"""`torch.nn.parallel.DistributedDataParallel` on backend `torchnative`, against upstream DDP over gloo.

Issue #247, `docs/distributed/BACKEND.md` D13 ("DDP gets a Python `Reducer`
under upstream's name, so user code runs unchanged"), SPEC S6.13. The design is
`docs/distributed/DDP.md`.

What is compared, and against what. Every scenario below is one **unchanged
upstream training loop** -- `DistributedDataParallel(model)`, `loss.backward()`,
`torch.optim.SGD.step()` -- run three times at each world size: on the shim over
`backend="torchnative"`, on upstream torch over gloo in float32, and on upstream
torch over gloo in float64. The float64 run is where the tolerance comes from
(`docs/numerics/AGREE.md` §2, both of its rules): a quantity agrees when the
shim is within upstream's own float32-vs-float64 error of upstream float32,
floored at 8 float32 ulp, or when the shim's own distance from the float64 run
is within 4x upstream's. Nothing here is a literal number.

The worlds are 2 and 3. Two is the smallest world that reduces anything; three
is the smallest at which "divide by the world" and "divide by two" differ and at
which the fold order of three contributions can show.

The inputs are a pure-Python generator seeded per rank and per tag, never
`torch.manual_seed`: the shim and upstream are two RNGs, and a shared seed only
means shared numbers when the generator is the same object (the same reasoning
`tests/distributed/test_collect2.py` gives). Every value is a multiple of 2**-13
in [-1, 1), exactly representable in float32 and float64, so the inputs are not
a source of difference between the runs.

The harness is `test_collect2.py`'s, imported rather than copied: real
`subprocess.Popen` ranks, an ephemeral port, every child killed on every path,
and output to files so a chatty rank cannot wedge its peers.
"""

import functools
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_DISTRIBUTED_TESTS = os.path.join(_HERE, "..", "distributed")
if _DISTRIBUTED_TESTS not in sys.path:
    sys.path.insert(0, _DISTRIBUTED_TESTS)

import _skip  # noqa: E402
import gloo_loopback  # noqa: E402
from test_collect2 import C2_F32_ULP, C2_ULP_FLOOR, _c2_flat, _c2_spawn  # noqa: E402
from test_shim import _CKPT_VENDOR_SHIM, _upstream_torch  # noqa: E402


#: The world sizes every value test runs at. See the module docstring.
DDP_WORLDS = (2, 3)

#: `docs/numerics/AGREE.md` §2's second rule: a difference above upstream's own
#: float32-vs-float64 error is still not a defect when the shim's distance from
#: the float64 run is within this factor of upstream's own distance. Stated
#: here so that widening it is a diff, and pinned by
#: `test_the_tolerance_is_agree_s_and_not_a_free_parameter`.
DDP_AGREE_FACTOR = 4

#: The scenarios whose numbers are compared with upstream, element by element.
DDP_VALUE_SCENARIOS = ("same_init", "different_init", "no_sync",
                       "zero_in_place", "one_byte_buckets")


# ---------------------------------------------------------------------------
# The probe, one source string shared verbatim by every side.
#
# Shared rather than written per side because the claim of this file is that
# the sides ran the same program on the same inputs; two copies of a training
# loop are two chances to drift, and a drift would read as a numerical defect.
# What differs between the workers is only the lines that choose a backend, and
# `shim`, which gates the probes that refuse here and that upstream would run.
# ---------------------------------------------------------------------------

_DDP_PROBE_SRC = r'''
import zlib

DDP_SEED = 20261113


def d_values(rank, n, tag):
    """`n` reproducible multiples of 2**-13 in [-1, 1) for `rank` and `tag`."""
    state = (DDP_SEED + rank * 1000003 + zlib.crc32(tag.encode())) % 2147483647
    out = []
    for _ in range(n):
        state = (state * 1103515245 + 12345) % 2147483648
        out.append(((state >> 8) % 16384 - 8192) / 8192.0)
    return out


def d_tensor(torch, DT, rank, shape, tag):
    n = 1
    for d in shape:
        n *= d
    return torch.tensor(d_values(rank, n, tag), dtype=DT).reshape(shape)


class DNet(torch.nn.Module):
    """Two `Linear`s and a `tanh`. `extra_unused` adds a third `Linear` that
    `forward` never reaches, which is the model upstream's unused-parameter
    error is about. Parameter order: fc1.weight, fc1.bias, fc2.weight,
    fc2.bias (then unused.weight, unused.bias)."""

    def __init__(self, extra_unused=False):
        super().__init__()
        self.fc1 = torch.nn.Linear(4, 8)
        self.fc2 = torch.nn.Linear(8, 3)
        if extra_unused:
            self.unused = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.fc2(torch.tanh(self.fc1(x)))


def d_model(torch, DT, init_rank, extra_unused=False):
    """A `DNet` initialised from the generator for `init_rank`."""
    model = DNet(extra_unused).to(DT)
    with torch.no_grad():
        for i, p in enumerate(model.parameters()):
            p.copy_(d_tensor(torch, DT, init_rank, tuple(p.shape), "init%d" % i))
    return model


def d_params(model):
    return [p.detach().reshape(-1).tolist() for p in model.parameters()]


def d_grads(model):
    return [None if p.grad is None else p.grad.detach().reshape(-1).tolist()
            for p in model.parameters()]


def d_mse(out, target):
    """Mean squared error spelled out, on the shim and the oracle alike.

    `F.mse_loss` reaches `torch._C._nn.mse_loss`, still a bootstrap.py stub
    (`tests/ops/test_tail1.py` pins that wall, docs/training/BACKWARD9.md §1
    spells its criterion the same way). What this suite checks is DDP's
    reducer, not the loss kernel.
    """
    return ((out - target) ** 2).mean()


def d_train(torch, dist, rank, DT, init_same=True, steps=3, ddp_kwargs=None,
            no_sync_first=False, zero_to_none=True):
    """One unchanged upstream DDP training loop. Returns what it observed."""
    model = d_model(torch, DT, 0 if init_same else rank)
    rec = {"init_local": d_params(model)}
    ddp = torch.nn.parallel.DistributedDataParallel(model, **(ddp_kwargs or {}))
    rec["after_ctor"] = d_params(model)
    rec["reducer_type"] = "%s.%s" % (type(ddp.reducer).__module__,
                                     type(ddp.reducer).__qualname__)
    opt = torch.optim.SGD(ddp.parameters(), lr=0.25)
    rec["losses"], rec["grads"], rec["params"], rec["grad_kept"] = [], [], [], []
    for step in range(steps):
        opt.zero_grad(set_to_none=zero_to_none)
        if no_sync_first and step == 0:
            with ddp.no_sync():
                out = ddp(d_tensor(torch, DT, rank, (5, 4), "xns%d" % step))
                d_mse(out, d_tensor(torch, DT, rank, (5, 3),
                                         "yns%d" % step)).backward()
            rec["grads_no_sync"] = d_grads(model)
        held = [p.grad for p in model.parameters()]
        out = ddp(d_tensor(torch, DT, rank, (5, 4), "x%d" % step))
        loss = d_mse(out, d_tensor(torch, DT, rank, (5, 3), "y%d" % step))
        loss.backward()
        rec["grad_kept"].append([h is not None and p.grad is h
                                 for p, h in zip(model.parameters(), held)])
        rec["losses"].append(loss.item())
        rec["grads"].append(d_grads(model))
        opt.step()
        rec["params"].append(d_params(model))
    data = ddp._get_ddp_logging_data()
    rec["logging"] = {k: data[k] for k in sorted(data)
                      if isinstance(data[k], (bool, int, float, str))}
    return rec


def d_unused(torch, dist, rank, DT):
    """A parameter `forward` never reaches, with `find_unused_parameters` off.

    Upstream's reducer cannot finish the iteration -- the parameter's bucket
    never completes -- and says so at the next forward. The probe records the
    first iteration's outcome and the second forward's exception.
    """
    model = d_model(torch, DT, 0, extra_unused=True)
    ddp = torch.nn.parallel.DistributedDataParallel(model)
    x = d_tensor(torch, DT, rank, (5, 4), "xu")
    y = d_tensor(torch, DT, rank, (5, 3), "yu")
    d_mse(ddp(x), y).backward()
    ddp(x)
    return "ACCEPTED: the second forward did not raise"


def d_run(torch, dist, rank, world, DT, shim):
    out = {}
    only = globals().get("D_ONLY")

    def probe(key, fn):
        if only is not None and key not in only:
            return
        try:
            out[key] = {"ok": fn()}
        except Exception as exc:
            out[key] = {"exc": "%s: %s" % (type(exc).__name__, exc)}

    probe("same_init", lambda: d_train(torch, dist, rank, DT))
    probe("different_init",
          lambda: d_train(torch, dist, rank, DT, init_same=False))
    probe("no_sync", lambda: d_train(torch, dist, rank, DT, steps=2,
                                     no_sync_first=True))
    probe("zero_in_place", lambda: d_train(torch, dist, rank, DT, steps=2,
                                           zero_to_none=False))
    # 1e-6 MiB is one byte, smaller than every parameter, so after the first
    # iteration's rebuild every parameter is its own bucket -- on upstream too,
    # whatever order its engine readies them in.
    probe("one_byte_buckets",
          lambda: d_train(torch, dist, rank, DT,
                          ddp_kwargs={"bucket_cap_mb": 1e-6}))
    probe("unused_parameter", lambda: d_unused(torch, dist, rank, DT))

    if shim:
        def ctor(kwargs):
            torch.nn.parallel.DistributedDataParallel(
                d_model(torch, DT, 0), **kwargs)
            return "ACCEPTED"
        probe("refuse_find_unused_parameters",
              lambda: ctor({"find_unused_parameters": True}))
        probe("refuse_gradient_as_bucket_view",
              lambda: ctor({"gradient_as_bucket_view": True}))
        probe("refuse_static_graph", lambda: ctor({"static_graph": True}))

        def comm_hook():
            ddp = torch.nn.parallel.DistributedDataParallel(d_model(torch, DT, 0))

            def noop(state, bucket):
                return bucket

            ddp.register_comm_hook(None, noop)
            return "ACCEPTED"
        probe("refuse_comm_hook", comm_hook)

        def join():
            ddp = torch.nn.parallel.DistributedDataParallel(d_model(torch, DT, 0))
            with ddp.join():
                pass
            return "ACCEPTED"
        probe("refuse_join", join)
    return out
'''


_DDP_SHIM_HEAD = (
    "import json, sys\n"
    "import torch\n"
    'assert hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded upstream torch, not the shim -- every value '
    'below would be upstream measuring itself")\n'
    "import torch.distributed as dist\n"
    "import torchnative.distributed  # noqa: F401 -- backend + reducer\n"
)

_DDP_SHIM_TAIL = r'''
rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
DT = getattr(torch, sys.argv[5])
dist.init_process_group(backend="torchnative",
                        init_method="tcp://127.0.0.1:%d" % port,
                        rank=rank, world_size=world)
out = d_run(torch, dist, rank, world, DT, True)
c10d = torch._C._distributed_c10d
out["_installed"] = {
    name: {"same_object": getattr(dist, name) is getattr(c10d, name),
           "module": getattr(getattr(c10d, name), "__module__", None)}
    for name in ("Reducer", "Logger", "_broadcast_coalesced",
                 "_compute_bucket_assignment_by_size",
                 "_verify_params_across_processes",
                 "_register_comm_hook", "_register_builtin_comm_hook")}
out["_meta"] = {"shim": True, "rank": dist.get_rank(),
                "world": dist.get_world_size(), "backend": "torchnative"}
with open(dest, "w") as handle:
    json.dump(out, handle)
'''

_DDP_SHIM_SRC = _DDP_SHIM_HEAD + _DDP_PROBE_SRC + _DDP_SHIM_TAIL

# The sabotage: the backend's allreduce still runs -- every rank stays in
# lockstep, nothing hangs -- but on a copy, so the caller's bucket comes back
# holding its own pre-scaled contribution. A reducer that really reduces
# through the backend must now disagree with upstream; one that computed the
# mean some other way, or copied local gradients, would not notice.
_DDP_SABOTAGE = r'''
from torch._C._distributed_c10d import ProcessGroupLocal as _PGL
_real_allreduce = _PGL.allreduce


def _dropped_allreduce(self, tensors, *args, **kwargs):
    return _real_allreduce(self, [t.clone() for t in tensors], *args, **kwargs)


_PGL.allreduce = _dropped_allreduce
D_ONLY = ("same_init",)
'''

_DDP_SABOTAGED_SRC = (_DDP_SHIM_HEAD + _DDP_SABOTAGE + _DDP_PROBE_SRC
                      + _DDP_SHIM_TAIL)

_DDP_GLOO_SRC = (
    gloo_loopback.child_pin_source()
    + "import json, os, sys\n"
    "import torch\n"
    'assert not hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded the shim, not upstream torch -- the oracle '
    'would be the thing under test")\n'
    "import torch.distributed as dist\n"
    + _DDP_PROBE_SRC
    + r'''
rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
DT = getattr(torch, sys.argv[5])
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(port)
dist.init_process_group(backend="gloo", rank=rank, world_size=world)
out = d_run(torch, dist, rank, world, DT, False)
out["_meta"] = {"shim": False, "rank": dist.get_rank(),
                "world": dist.get_world_size(), "backend": "gloo"}
with open(dest, "w") as handle:
    json.dump(out, handle)
dist.destroy_process_group()
'''
)


@functools.cache
def ddp_shim(world):
    return _c2_spawn(_DDP_SHIM_SRC, world, "float32", True, "ddp-shim%d" % world)


@functools.cache
def ddp_sabotaged(world):
    return _c2_spawn(_DDP_SABOTAGED_SRC, world, "float32", True,
                     "ddp-sabotage%d" % world)


@functools.cache
def ddp_gloo32(world):
    return _c2_spawn(_DDP_GLOO_SRC, world, "float32", False, "ddp-gloo32-%d" % world)


@functools.cache
def ddp_gloo64(world):
    return _c2_spawn(_DDP_GLOO_SRC, world, "float64", False, "ddp-gloo64-%d" % world)


def _need_shim():
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        raise _skip.Skip("no vendored shim installed (%s)" % _CKPT_VENDOR_SHIM)


def _need_oracle():
    _need_shim()
    if _upstream_torch is None:
        raise _skip.Skip("no upstream torch for the gloo oracle (docs/models/E2E.md)")


def _ok(report, key):
    entry = report[key]
    assert "ok" in entry, "%s did not run: %s" % (key, entry.get("exc"))
    return entry["ok"]


def _exc(report, key):
    entry = report[key]
    assert "exc" in entry, "%s was ACCEPTED where a refusal was required: %r" % (
        key, entry.get("ok"))
    return entry["exc"]


# ---------------------------------------------------------------------------
# Agreement, with the tolerance derived from upstream's own run
# ---------------------------------------------------------------------------

def ddp_compare(shim, up32, up64):
    """`(agrees, detail)` for one quantity, by `docs/numerics/AGREE.md` §2.

    Rule 1: within upstream's own float32-vs-float64 error, floored at 8 ulp.
    Rule 2: otherwise, the shim's own distance from float64 within
    `DDP_AGREE_FACTOR` times upstream's (floored the same way). Scaled by the
    largest float64 magnitude, at least 1, as every cross-interpreter
    comparison in this repository is.
    """
    a, b, c = _c2_flat(shim), _c2_flat(up32), _c2_flat(up64)
    assert len(a) == len(b) == len(c), (len(a), len(b), len(c))
    assert a, "compared an empty quantity, which would pass for any implementation"
    assert None not in a and None not in b, "a gradient was None: %r / %r" % (a, b)
    scale = max([abs(x) for x in c] + [1.0])
    floor = C2_ULP_FLOOR * C2_F32_ULP
    own = max(abs(x - y) for x, y in zip(b, c)) / scale
    worst = max(abs(x - y) for x, y in zip(a, b)) / scale
    if worst <= max(own, floor):
        return True, "within %.3e of float32 (tol %.3e)" % (worst, max(own, floor))
    to_truth = max(abs(x - y) for x, y in zip(a, c)) / scale
    bound = max(DDP_AGREE_FACTOR * own, floor)
    return (to_truth <= bound,
            "%.3e from float32 (tol %.3e); %.3e from float64 against %dx "
            "upstream's own %.3e" % (worst, max(own, floor), to_truth,
                                     DDP_AGREE_FACTOR, own))


def ddp_assert_agrees(what, shim, up32, up64):
    agrees, detail = ddp_compare(shim, up32, up64)
    assert agrees, "%s: %s\n  shim: %r\n  gloo: %r" % (what, detail, shim, up32)


def _fields(rec):
    """The per-step quantities of one scenario, as (name, value) pairs."""
    out = [("init_local", rec["init_local"]), ("after_ctor", rec["after_ctor"])]
    if "grads_no_sync" in rec:
        out.append(("grads_no_sync", rec["grads_no_sync"]))
    for step in range(len(rec["grads"])):
        out.append(("loss[%d]" % step, [rec["losses"][step]]))
        out.append(("grads[%d]" % step, rec["grads"][step]))
        out.append(("params[%d]" % step, rec["params"][step]))
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_ddp_training_agrees_with_upstream_ddp_over_gloo():
    """The whole claim. Every scenario, every rank, every step, every quantity.

    Fails if the reducer averages wrongly (sum instead of mean, a different
    divisor), copies the wrong bucket slice into a `.grad`, loses the
    `no_sync()` accumulation, double-counts a gradient the engine also
    accumulated, or broadcasts the wrong rank's initial parameters -- each of
    those moves `grads[k]` or `params[k]` by far more than float32 rounding.
    """
    _need_oracle()
    compared = 0
    for world in DDP_WORLDS:
        shim, up32, up64 = ddp_shim(world), ddp_gloo32(world), ddp_gloo64(world)
        for rank in range(world):
            for scenario in DDP_VALUE_SCENARIOS:
                a = _ok(shim[rank], scenario)
                b = _ok(up32[rank], scenario)
                c = _ok(up64[rank], scenario)
                fa, fb, fc = _fields(a), _fields(b), _fields(c)
                assert [n for n, _ in fa] == [n for n, _ in fb] == [n for n, _ in fc]
                for (name, va), (_, vb), (_, vc) in zip(fa, fb, fc):
                    ddp_assert_agrees("%s/%s at world %d rank %d"
                                      % (scenario, name, world, rank), va, vb, vc)
                    compared += 1
                # Not vacuous: every parameter received a non-zero gradient.
                for grads in a["grads"]:
                    for row in grads:
                        assert max(abs(v) for v in row) > 0.0, (scenario, grads)
    # 5 scenarios; 3/3/2/2/3 steps -> 2 + 3*steps fields, +1 for no_sync's
    # local gradients. Stated so a dropped scenario is a failure.
    per_rank = sum(2 + 3 * s for s in (3, 3, 2, 2, 3)) + 1
    assert compared == per_rank * sum(DDP_WORLDS), compared


def test_the_reducer_in_use_is_this_project_s_and_upstream_s_on_the_oracle():
    """Under the shim `ddp.reducer` is the Python reducer, installed under
    upstream's names in both namespaces as one object; under upstream it is the
    C++ one. Fails if `torchnative.distributed` stopped installing it (DDP would
    refuse at construction), if `torch.distributed.Reducer` and
    `torch._C._distributed_c10d.Reducer` diverged, or if the oracle ran the shim.
    """
    _need_oracle()
    for world in DDP_WORLDS:
        for report in ddp_shim(world):
            rec = _ok(report, "same_init")
            assert rec["reducer_type"] == "torchnative.distributed._reducer.Reducer", rec
            for name, row in report["_installed"].items():
                assert row["same_object"], (name, row)
                assert row["module"] == "torchnative.distributed._reducer", (name, row)
        for report in ddp_gloo32(world):
            rec = _ok(report, "same_init")
            # Upstream's pybind class; its exact `__module__` spelling is
            # upstream's business, so only "not ours, and a Reducer" is held.
            assert rec["reducer_type"].endswith(".Reducer"), rec
            assert "torchnative" not in rec["reducer_type"], rec


def test_every_rank_holds_rank_zero_s_parameters_after_construction():
    """DDP broadcasts rank 0's parameters at construction (`_sync_module_states`
    through `_broadcast_coalesced`). With a deliberately different init per rank
    the local inits differ -- checked, so the test is not vacuous -- and after
    the constructor every rank must hold rank 0's, **bit for bit**: a broadcast
    moves numbers and rounds none. After each step every rank's parameters are
    again bit-identical, because every rank applied the same reduced gradient.

    Fails if `_broadcast_coalesced` is a no-op, broadcasts from the wrong root,
    writes into a copy instead of the parameter's storage, or if the reduced
    gradient differs between ranks.
    """
    _need_shim()
    for world in DDP_WORLDS:
        recs = [_ok(r, "different_init") for r in ddp_shim(world)]
        assert recs[0]["init_local"] != recs[1]["init_local"], "inits were equal"
        for rank, rec in enumerate(recs):
            assert rec["after_ctor"] == recs[0]["init_local"], (world, rank)
            for step, params in enumerate(rec["params"]):
                assert params == recs[0]["params"][step], (world, rank, step)


def test_no_sync_accumulates_locally_and_the_next_step_reduces_the_sum():
    """`with ddp.no_sync(): loss.backward()` accumulates without communicating,
    and the next synced backward reduces the *accumulated* gradient.

    Local gradients inside `no_sync` differ across ranks (each rank's data
    differs; checked). The synced gradient is the same on every rank, bit for
    bit, and differs from the gradient of the synced batch alone (the
    `same_init` scenario's step 0), so the accumulated part reached the
    reduction. Agreement with upstream is in the first test. Fails if the hook
    reduced inside `no_sync`, or reduced only the incoming gradient rather than
    `.grad` plus it.
    """
    _need_shim()
    for world in DDP_WORLDS:
        recs = [_ok(r, "no_sync") for r in ddp_shim(world)]
        plain = [_ok(r, "same_init") for r in ddp_shim(world)]
        assert recs[0]["grads_no_sync"] != recs[1]["grads_no_sync"], world
        for rank, rec in enumerate(recs):
            assert rec["grads"][0] == recs[0]["grads"][0], (world, rank)
            assert rec["grads"][0] != plain[rank]["grads"][0], (world, rank)


def test_the_grad_tensor_keeps_its_identity_as_upstream_s_does():
    """Upstream accumulates into an existing `.grad` in place and the reducer
    copies the reduced bucket back into it, so a `.grad` that existed before
    `backward()` is the same object afterwards. The reducer here has to hand
    `.grad` to the engine and take it back (`docs/distributed/DDP.md` §3), and
    this is what holds it to restoring the caller's tensor. Fails if any
    parameter's identity differs from upstream's, in any scenario or step.
    """
    _need_oracle()
    for world in DDP_WORLDS:
        for rank in range(world):
            for scenario in DDP_VALUE_SCENARIOS:
                a = _ok(ddp_shim(world)[rank], scenario)["grad_kept"]
                b = _ok(ddp_gloo32(world)[rank], scenario)["grad_kept"]
                assert a == b, (scenario, world, rank, a, b)
    # Not vacuous: the in-place scenario has a pre-existing `.grad` to keep.
    kept = _ok(ddp_shim(2)[0], "zero_in_place")["grad_kept"]
    assert any(any(row) for row in kept), kept


def test_one_byte_buckets_give_the_same_numbers_as_one_bucket():
    """`bucket_cap_mb=1e-6` (one byte) makes every parameter its own bucket after
    the first iteration's rebuild; the default makes one. The allreduce is
    elementwise and folds ranks in ascending order either way, so the two runs
    must be **bit-identical**. Upstream rebuilds into one bucket per parameter
    too, and the bucket byte sizes match it as a multiset.

    Fails if a bucket's slice offsets are wrong, if a bucket is launched before
    all of its gradients arrived, if `_rebuild_buckets` drops or duplicates a
    parameter, or if the rebuild silently kept one bucket.
    """
    _need_oracle()
    for world in DDP_WORLDS:
        for rank in range(world):
            small = _ok(ddp_shim(world)[rank], "one_byte_buckets")
            one = _ok(ddp_shim(world)[rank], "same_init")
            assert small["grads"] == one["grads"], (world, rank)
            assert small["params"] == one["params"], (world, rank)

            ours = [int(v) for v in re.findall(
                r"\d+", small["logging"]["rebuilt_bucket_sizes"])]
            theirs_log = _ok(ddp_gloo32(world)[rank], "one_byte_buckets")["logging"]
            assert "rebuilt_bucket_sizes" in theirs_log, sorted(theirs_log)
            theirs = [int(v) for v in re.findall(
                r"\d+", theirs_log["rebuilt_bucket_sizes"])]
            # 4*8, 8, 8*3, 3 float32s: one bucket each.
            assert sorted(ours) == sorted(theirs) == [12, 32, 96, 128], (ours, theirs)
            single = [int(v) for v in re.findall(
                r"\d+", one["logging"]["rebuilt_bucket_sizes"])]
            assert single == [268], single


def test_logging_data_agrees_with_upstream_on_what_it_records():
    """`ddp._get_ddp_logging_data()` reads the `Logger`. The construction-time
    and rebuild facts it records are compared with upstream's; timing stats
    are not built and not compared (`docs/distributed/DDP.md` §5). Fails if a
    recorded fact is wrong or the logger stopped recording it.
    """
    _need_oracle()
    keys = ("world_size", "rank", "module_name", "bucket_cap_bytes",
            "find_unused_parameters", "gradient_as_bucket_view",
            "num_parameter_tensors", "total_parameter_size_bytes",
            "has_rebuilt_buckets")
    for world in DDP_WORLDS:
        for rank in range(world):
            ours = _ok(ddp_shim(world)[rank], "same_init")["logging"]
            theirs = _ok(ddp_gloo32(world)[rank], "same_init")["logging"]
            compared = [k for k in keys if k in theirs]
            assert len(compared) >= 6, (sorted(theirs), compared)
            for key in compared:
                assert key in ours, (key, sorted(ours))
                if isinstance(theirs[key], str):
                    assert ours[key] == theirs[key], (key, ours[key], theirs[key])
                else:
                    assert int(ours[key]) == int(theirs[key]), (
                        key, ours[key], theirs[key])
            assert ours["backend_name"] == "torchnative", ours


def test_an_unused_parameter_fails_the_next_iteration_as_upstream_does():
    """With `find_unused_parameters=False` a parameter that gets no gradient
    leaves its bucket unfinished; upstream raises at the next forward rather
    than stepping on a half-reduced model, and so must this. Fails if the
    reducer silently skipped the parameter, or reduced a partial bucket.
    """
    _need_oracle()
    marker = "Expected to have finished reduction in the prior iteration"
    for world in DDP_WORLDS:
        for rank in range(world):
            ours = _exc(ddp_shim(world)[rank], "unused_parameter")
            theirs = _exc(ddp_gloo32(world)[rank], "unused_parameter")
            assert theirs.startswith("RuntimeError") and marker in theirs, theirs
            assert ours.startswith("RuntimeError") and marker in ours, ours
            assert "unused.weight" in ours, ours


def test_a_backend_allreduce_that_drops_its_result_makes_the_comparison_fail():
    """The sabotage (AGENTS.md §17.5). In a child, `ProcessGroupLocal.allreduce`
    is replaced by one that runs on a copy, so every rank keeps its own
    contribution. The first step's gradients must then **disagree** with
    upstream and differ across ranks. If they still agreed, the agreement test
    above would be measuring something other than the backend's reduction.
    """
    _need_oracle()
    world = 2
    bad = ddp_sabotaged(world)
    disagreements = 0
    for rank in range(world):
        a = _ok(bad[rank], "same_init")["grads"][0]
        b = _ok(ddp_gloo32(world)[rank], "same_init")["grads"][0]
        c = _ok(ddp_gloo64(world)[rank], "same_init")["grads"][0]
        agrees, _ = ddp_compare(a, b, c)
        disagreements += not agrees
    assert disagreements == world, disagreements
    grads = [_ok(r, "same_init")["grads"][0] for r in bad]
    assert grads[0] != grads[1], "ranks agreed although nothing was reduced"


def test_what_is_not_built_refuses_by_name():
    """Each of these is refused with its own name, never accepted and
    approximated: `find_unused_parameters=True`, `gradient_as_bucket_view=True`,
    `static_graph=True`, a communication hook, and `join()`. Fails if any is
    accepted, or refuses without naming itself.
    """
    _need_shim()
    names = {
        "refuse_find_unused_parameters": "find_unused_parameters",
        "refuse_gradient_as_bucket_view": "gradient_as_bucket_view",
        "refuse_static_graph": "static_graph",
        "refuse_comm_hook": "_register_comm_hook",
        "refuse_join": "join",
    }
    for world in DDP_WORLDS:
        for report in ddp_shim(world):
            for key, name in names.items():
                message = _exc(report, key)
                assert message.startswith("NotImplementedError"), (key, message)
                assert name in message, (key, message)


def test_the_tolerance_is_agree_s_and_not_a_free_parameter():
    """`docs/numerics/AGREE.md` §2: 8 ulp floor, factor 4. And the comparison
    can fail: a gradient doubled (a sum where a mean belongs) is rejected, and
    upstream float32 against itself is accepted.
    """
    assert C2_ULP_FLOOR == 8 and C2_F32_ULP == 2.0 ** -23
    assert DDP_AGREE_FACTOR == 4
    up32 = [[0.5, -0.25, 0.125]]
    up64 = [[0.5, -0.25, 0.125]]
    assert ddp_compare(up32, up32, up64)[0]
    assert not ddp_compare([[1.0, -0.5, 0.25]], up32, up64)[0]


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_ddp")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
