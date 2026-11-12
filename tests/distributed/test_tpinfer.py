"""Tensor-parallel inference of a real `transformers` model on backend `torchnative`.

BACKEND.md section 9 step 9 (issue #242), SPEC S6.13. A tiny Qwen3 (the
project's test model family, owner 2026-10-03) is built by upstream with
`from_config` under a fixed seed and written once to a temporary checkpoint,
so every rank, every oracle and the single-process runs load identical
weights. It is then run:

* single-process on the shim and on upstream (float32 and float64): the
  world 1 control, the existing method of `tests/models/test_qwen3.py`;
* tensor-parallel at tp 2 and tp 3 over `init_process_group("torchnative")`
  with an explicit 1-D `DeviceMesh("cpu", (N,), ("tp",))`, and the same over
  upstream torch with gloo, in float32 and float64.

Why this file is in `tests/distributed/` rather than `tests/models/`: it reuses
`test_collect2.py`'s subprocess-world harness (`_c2_spawn`: ephemeral port,
per-rank files, a deadline for the spawn) and its gloo oracle by import. A
suite runs with its own folder as `sys.path[0]` and only `tests/_support/` on
PYTHONPATH (`tests/run.sh`), so the harness is importable from here and not
from `tests/models/`. Copying it would be two harnesses to keep honest.

**How the model is sharded, and what is ours.** The documented user line is
`from_pretrained(path, distributed_config=DistributedConfig(tp_size=N),
device_mesh=mesh)`. In transformers 5.15.1 that line gets through
`initialize_tensor_parallelism` and then refuses in
`integrations/accelerate.py:check_and_set_device_map`, because the explicit
mesh makes a `device_map` and a `device_map` requires `accelerate`. Upstream
refuses there too: `accelerate` is in neither `.caches/spike-venv` nor
`.github/scripts/gate-requirements.txt`, and adding it is the owner's decision
(AGENTS.md section 7). `test_the_users_from_pretrained_line_reaches_the_same_wall_on_both_sides`
holds both sides to that same wall, so the shim cannot stop earlier unseen.

So the sharding below is transformers' own, in from_pretrained's order, with a
loop of glue that is ours: `apply_tensor_parallelism` installs the hooks
(and runs `validate_module`) first, as `maybe_distribute_model` does before
weights load; then every parameter the plan names is cut by
`ALL_PARALLEL_STYLES[style].__class__(device_mesh, rank, empty_param)`
`.shard_tensor(...)` and set back with `update_module_attributes`, as
`core_model_loading.py:1683-1690` and `shard_and_distribute_module` do. The
only change is `device="cpu"` where transformers passes `device_map[""]`
(`cpu:LOCAL_RANK`), because `shard_and_distribute_module` itself passes the
integer rank as the device and `torch.device(<int>)` means "the current
accelerator", which neither side has.

**The tolerance is derived** (docs/numerics/AGREE.md section 2): upstream's
own float32-vs-float64 error on the tensor-parallel logits at that world size,
p90 floored at 8 float32 ulp, and not a defect within 4x upstream's own worst
distance from float64. There is no constant here to widen.

**What can fail** (AGENTS.md section 17.5):

* the shard shapes and the per-forward collective counts are asserted on
  every rank, so a model that silently stayed replicated (hooks not
  installed, weights not cut) fails by count before it can pass by value;
* the sabotage: the last rank's `dist.all_reduce` is replaced by one that
  runs the collective on a clone and discards the result. Its logits must
  leave the derived bound. A pure no-op on one rank is not a test here: the
  peers would block in the star exchange and the harness would report a
  timeout, which says nothing about values;
* every shim child asserts `hasattr(torch._C, "_aten_implemented")` and
  every upstream child asserts its absence (AGENTS.md section 15.2).

**What this cannot see** (AGENTS.md section 17.4): uneven shards (every dim
here divides by 2 and 3), packed or MoE styles, `embedding_rowwise`, a model
whose forward reads `config.num_attention_heads` instead of the local head
count, the TP backward (training is step 12), and any device other than CPU.
"""

import atexit
import functools
import json
import os
import shutil
import tempfile

import _skip
import gloo_loopback

from test_collect2 import _c2_spawn
from test_shim import _CKPT_VENDOR_DIR

_VENDOR_SHIM = os.path.join(_CKPT_VENDOR_DIR, "torch", "_C.abi3.so")

#: float32 machine epsilon, the floor's unit (docs/numerics/AGREE.md section 2).
_EPS32 = 2.0 ** -23

#: AGREE.md section 2: the floor is 8 ulp.
_ULP_FLOOR = 8

#: AGREE.md section 2's second rule: within this multiple of upstream's own
#: distance from float64 is not a defect.
_ORACLE_FACTOR = 4.0

#: The tensor-parallel world sizes. Three is not optional: a star whose hub is
#: special, or a shard split that is right only for a pair, passes at two.
_WORLDS = (2, 3)

_LAYERS = 2

#: Every dimension the Qwen3 plan shards divides by 2 and by 3: q_proj out
#: 12*8=96, k/v out 6*8=48, o_proj in 96, MLP 60, vocab 96 (lm_head is
#: `colwise_gather_output`, which requires vocab % N == 0). 12 heads over 6 kv
#: heads keeps grouped-query attention at every N (6/3, 4/2 heads per rank).
#: Untied, so sharding `lm_head` does not touch `embed_tokens`.
_TP_CFG = dict(
    vocab_size=96, hidden_size=48, intermediate_size=60,
    num_hidden_layers=_LAYERS, num_attention_heads=12, num_key_value_heads=6,
    head_dim=8, max_position_embeddings=64, tie_word_embeddings=False,
    rope_theta=1e6, bos_token_id=1, eos_token_id=2, pad_token_id=1,
)

_IDS = [5, 7, 9, 11, 13, 17, 19, 23]

#: (full shape, shard shape at world N) per parameter kind, from `_TP_CFG`.
def _expected_shards(world):
    h, q, kv, ff, v = 48, 96, 48, 60, 96
    return {
        "self_attn.q_proj.weight": ((q, h), (q // world, h)),
        "self_attn.k_proj.weight": ((kv, h), (kv // world, h)),
        "self_attn.v_proj.weight": ((kv, h), (kv // world, h)),
        "self_attn.o_proj.weight": ((h, q), (h, q // world)),
        "self_attn.q_norm.weight": ((8,), (8,)),
        "self_attn.k_norm.weight": ((8,), (8,)),
        "mlp.gate_proj.weight": ((ff, h), (ff // world, h)),
        "mlp.up_proj.weight": ((ff, h), (ff // world, h)),
        "mlp.down_proj.weight": ((h, ff), (h, ff // world)),
    }


# ---------------------------------------------------------------------------
# One probe body, shared verbatim by the shim and the upstream children.
# ---------------------------------------------------------------------------

_TP_PROBE_SRC = r'''
def tp_child(torch, dist, rank, world, A, shim, init):
    import os, traceback
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HOME"] = A["hf_home"]
    out = {"stage": "start", "rank": rank, "world": world, "shim": shim}
    try:
        from transformers import AutoModelForCausalLM, Qwen3Config
        dt = getattr(torch, A["dtype"])

        if A["mode"] == "write":
            out["stage"] = "write"
            torch.manual_seed(0)
            m = AutoModelForCausalLM.from_config(Qwen3Config(**A["config"])).eval()
            with torch.no_grad():
                for p in m.parameters():
                    # Away from the init so q_norm/k_norm and every shard matter.
                    p.normal_(0.0, 0.3)
            m.save_pretrained(A["path"])
            out["written"] = sorted(n for n, _ in m.named_parameters())
            out["stage"] = "done"
            return out

        ids = torch.tensor([A["ids"]])
        enc = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

        if A["mode"] == "single":
            out["stage"] = "load"
            model = AutoModelForCausalLM.from_pretrained(A["path"], dtype=dt).eval()
            out["stage"] = "forward"
            with torch.no_grad():
                out["logits"] = model(**enc).logits.tolist()
            out["stage"] = "done"
            return out

        # -- tensor parallel ------------------------------------------------
        os.environ["RANK"] = str(rank)
        os.environ["LOCAL_RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
        out["stage"] = "init_process_group"
        init()
        out["backend"] = dist.get_backend()

        out["stage"] = "device_mesh"
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.distributed_c10d import _get_default_group
        mesh = init_device_mesh("cpu", (world,), mesh_dim_names=("tp",))
        out["mesh"] = {
            "local_rank": mesh.get_local_rank(), "size": mesh.size(),
            "device_type": mesh.device_type,
            "dim_names": list(mesh.mesh_dim_names),
            "is_default_group": mesh.get_group() is _get_default_group(),
        }

        from transformers.distributed import DistributedConfig
        from transformers.integrations.tensor_parallel import (
            ALL_PARALLEL_STYLES, _get_parameter_tp_plan, apply_tensor_parallelism)

        # The user's line. Recorded, not required: see the module docstring.
        out["stage"] = "user_line"
        import importlib.util
        user = {"accelerate": importlib.util.find_spec("accelerate") is not None}
        try:
            um = AutoModelForCausalLM.from_pretrained(
                A["path"], dtype=dt, device_mesh=mesh,
                distributed_config=DistributedConfig(tp_size=world)).eval()
            with torch.no_grad():
                user["ok"] = um(**enc).logits.tolist()
            del um
        except Exception as exc:
            user["exc"] = "%s: %s" % (type(exc).__name__, exc)
        out["user_line"] = user

        out["stage"] = "load"
        model = AutoModelForCausalLM.from_pretrained(A["path"], dtype=dt).eval()

        # Hooks first, as `maybe_distribute_model` runs before weights load.
        out["stage"] = "apply_tensor_parallelism"
        apply_tensor_parallelism(model, None, DistributedConfig(tp_size=world), mesh)

        out["stage"] = "shard"
        plan = model.tp_plan
        sharded = {}
        for name, param in list(model.named_parameters()):
            style = _get_parameter_tp_plan(name, plan)
            if style is None:
                continue
            layer = ALL_PARALLEL_STYLES[style].__class__(
                device_mesh=mesh, rank=mesh.get_local_rank(),
                empty_param=param.detach())
            piece = layer.shard_tensor(param.detach(), tensor_idx=None,
                                       device="cpu", dtype=param.dtype)
            piece = piece.contiguous().clone()
            owner, attr = name.rsplit(".", 1)
            module = model.get_submodule(owner)
            setattr(module, attr, torch.nn.Parameter(piece, requires_grad=False))
            layer.update_module_attributes(module)
            sharded[name] = [list(param.shape), list(piece.shape), style]
        out["sharded"] = sharded

        real_ar, real_ag = dist.all_reduce, dist.all_gather

        def forward(counts, drop):
            def ar(tensor, *a, **k):
                counts["all_reduce"] += 1
                if drop:
                    # The collective still runs, so no peer blocks; this rank
                    # just never sees its result.
                    return real_ar(tensor.clone(), *a, **k)
                return real_ar(tensor, *a, **k)

            def ag(*a, **k):
                counts["all_gather"] += 1
                return real_ag(*a, **k)

            dist.all_reduce, dist.all_gather = ar, ag
            try:
                with torch.no_grad():
                    return model(**enc).logits.tolist()
            finally:
                dist.all_reduce, dist.all_gather = real_ar, real_ag

        out["stage"] = "forward"
        counts = {"all_reduce": 0, "all_gather": 0}
        out["logits"] = forward(counts, False)
        out["counts"] = counts

        out["stage"] = "sabotage"
        sab = {"all_reduce": 0, "all_gather": 0}
        out["sabotaged_logits"] = forward(sab, rank == A["sabotage_rank"])
        out["sabotage_counts"] = sab
        out["stage"] = "done"
    except BaseException as exc:
        # Recorded rather than raised: the parent names the stage, and a peer
        # blocked on this rank sees its socket close instead of a timeout.
        out["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:1500])
        out["traceback"] = traceback.format_exc()[-4000:]
    return out
'''

_TP_SHIM_SRC = (
    "import json, sys\n"
    "import torch\n"
    'assert hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded upstream torch, not the shim")\n'
    "import torch.distributed as dist\n"
    "import torchnative.distributed  # noqa: F401 (registers backend='torchnative')\n"
    + _TP_PROBE_SRC
    + r'''
rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
A = json.loads(sys.argv[5])
def init():
    dist.init_process_group(backend="torchnative",
                            init_method="tcp://127.0.0.1:%d" % port,
                            rank=rank, world_size=world)
out = tp_child(torch, dist, rank, world, A, True, init)
out["_meta"] = {"shim": hasattr(torch._C, "_aten_implemented"), "rank": rank}
with open(dest, "w") as handle:
    json.dump(out, handle)
'''
)

_TP_GLOO_SRC = (
    gloo_loopback.child_pin_source()
    + "import json, os, sys\n"
    "import torch\n"
    'assert not hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded the shim, not upstream torch")\n'
    "import torch.distributed as dist\n"
    + _TP_PROBE_SRC
    + r'''
rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
A = json.loads(sys.argv[5])
def init():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
out = tp_child(torch, dist, rank, world, A, False, init)
out["_meta"] = {"shim": hasattr(torch._C, "_aten_implemented"), "rank": rank}
with open(dest, "w") as handle:
    json.dump(out, handle)
if dist.is_initialized():
    dist.destroy_process_group()
'''
)


# ---------------------------------------------------------------------------
# Fixtures, each run once
# ---------------------------------------------------------------------------

def _require_shim():
    if not os.path.isfile(_VENDOR_SHIM):
        raise _skip.Skip(f"vendored tree not built: no {_VENDOR_SHIM} "
                         "(scripts/vendor/vendor_torch.sh + scripts/vendor/install_shim.sh)")


@functools.cache
def _workdir():
    root = tempfile.mkdtemp(prefix="tpinfer-")
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root


def _args(mode, dtype, **extra):
    base = {"mode": mode, "dtype": dtype, "ids": _IDS,
            "path": os.path.join(_workdir(), "tiny-qwen3"),
            "hf_home": os.path.join(_workdir(), "hf-home")}
    base.update(extra)
    return json.dumps(base)


def _reached(report, what):
    assert report.get("stage") == "done" and "error" not in report, (
        f"{what}: rank {report.get('rank')} stopped at stage "
        f"{report.get('stage')!r}: {report.get('error')}\n"
        f"{report.get('traceback', '')[-1500:]}")


@functools.cache
def _checkpoint():
    """Written once by upstream, read by every side after."""
    _require_shim()
    [rep] = _c2_spawn(_TP_GLOO_SRC, 1, _args("write", "float32", config=_TP_CFG),
                      False, "tp-write")
    _reached(rep, "write")
    assert rep["written"], rep
    return json.loads(_args("write", "float32"))["path"]


@functools.cache
def _single(shim, dtype):
    _checkpoint()
    src = _TP_SHIM_SRC if shim else _TP_GLOO_SRC
    [rep] = _c2_spawn(src, 1, _args("single", dtype), shim,
                      f"tp-single-{'shim' if shim else 'up'}-{dtype}")
    _reached(rep, f"single {'shim' if shim else 'upstream'} {dtype}")
    return rep


@functools.cache
def _tp(shim, world, dtype):
    _checkpoint()
    src = _TP_SHIM_SRC if shim else _TP_GLOO_SRC
    reps = _c2_spawn(src, world, _args("tp", dtype, sabotage_rank=world - 1), shim,
                     f"tp{world}-{'shim' if shim else 'gloo'}-{dtype}")
    for rep in reps:
        _reached(rep, f"tp {world} {'shim' if shim else 'gloo'} {dtype}")
    return reps


# ---------------------------------------------------------------------------
# The derived bound
# ---------------------------------------------------------------------------

def _np(values):
    import numpy as np
    return np.asarray(values, dtype=np.float64)


def _bound(up32, up64):
    """docs/numerics/AGREE.md section 2 for THIS output: (bound, tol, own, scale)."""
    import numpy as np
    u, u64 = _np(up32).ravel(), _np(up64).ravel()
    assert u.shape == u64.shape, (u.shape, u64.shape)
    assert np.all(np.isfinite(u64)), "upstream's float64 oracle is not finite"
    scale = float(np.max(np.abs(u64)))
    assert scale > 0, "degenerate output: every logit is zero"
    own = np.abs(u - u64) / scale
    p90 = float(np.sort(own)[int(0.9 * (own.size - 1))])
    tol = max(p90, _ULP_FLOOR * _EPS32)
    worst = float(own.max())
    return max(tol, _ORACLE_FACTOR * worst) * scale, tol, worst, scale


def _dist(a, b):
    import numpy as np
    a, b = _np(a), _np(b)
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.max(np.abs(a - b)))


def _tp_bound(world, rank, key="logits"):
    """Upstream's own float32-vs-float64 error on `rank`'s TP output at `world`."""
    up32, up64 = _tp(False, world, "float32"), _tp(False, world, "float64")
    return _bound(up32[rank][key], up64[rank][key])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_world_one_control_single_process_shim_agrees_with_upstream():
    """The existing method, unchanged, on this checkpoint: no process group."""
    assert (_ULP_FLOOR, _ORACLE_FACTOR, _EPS32) == (8, 4.0, 2.0 ** -23)
    shim, up32, up64 = _single(True, "float32"), _single(False, "float32"), _single(False, "float64")
    bound, tol, own, scale = _bound(up32["logits"], up64["logits"])
    d = _dist(shim["logits"], up32["logits"])
    assert d <= bound, (
        f"single-process shim logits differ from upstream by {d:.3e}; derived "
        f"bound {bound:.3e} (p90 {tol:.3e}, upstream own worst {own:.3e}, scale {scale:.3e})")


def _check_tp_agrees(world):
    shim = _tp(True, world, "float32")
    gloo = _tp(False, world, "float32")
    single_shim = _single(True, "float32")
    single_up = _single(False, "float32")
    for rank in range(world):
        bound, tol, own, scale = _tp_bound(world, rank)
        detail = (f"derived bound {bound:.3e} (p90 {tol:.3e}, upstream's own TP "
                  f"worst {own:.3e}, scale {scale:.3e})")
        # Premise: upstream's own TP run sits within its own bound of its own
        # single-process run. If not, the bound is not measuring the shim.
        d = _dist(gloo[rank]["logits"], single_up["logits"])
        assert d <= bound, (f"tp {world}: upstream gloo rank {rank} is {d:.3e} from "
                            f"upstream single-process; {detail}")
        assert shim[rank]["backend"] == "torchnative", shim[rank]["backend"]
        assert gloo[rank]["backend"] == "gloo", gloo[rank]["backend"]
        d_single = _dist(shim[rank]["logits"], single_shim["logits"])
        assert d_single <= bound, (
            f"tp {world}: shim rank {rank} logits are {d_single:.3e} from the "
            f"single-process shim logits; {detail}")
        d_gloo = _dist(shim[rank]["logits"], gloo[rank]["logits"])
        assert d_gloo <= bound, (
            f"tp {world}: shim rank {rank} logits are {d_gloo:.3e} from upstream "
            f"torch's TP-over-gloo logits for the same weights; {detail}")
    # The star's contract (FEDERATED4.md section 2): one fold on the hub, one
    # result sent back, so every rank holds the same bytes.
    first = shim[0]["logits"]
    for rank in range(1, world):
        assert shim[rank]["logits"] == first, (
            f"tp {world}: shim rank {rank} logits are not bit-identical to rank 0's")


def test_tp2_every_rank_agrees_with_single_process_and_with_upstream_gloo_tp():
    _check_tp_agrees(2)


def test_tp3_every_rank_agrees_with_single_process_and_with_upstream_gloo_tp():
    _check_tp_agrees(3)


def test_the_weights_are_sharded_and_the_collectives_run_on_every_rank():
    """A model that stayed replicated would agree by value; it fails here by count."""
    for world in _WORLDS:
        shim, gloo = _tp(True, world, "float32"), _tp(False, world, "float32")
        expected = _expected_shards(world)
        for rank in range(world):
            for side, rep in (("shim", shim[rank]), ("gloo", gloo[rank])):
                mesh = rep["mesh"]
                assert mesh == {"local_rank": rank, "size": world, "device_type": "cpu",
                                "dim_names": ["tp"], "is_default_group": True}, (side, mesh)
                sharded = rep["sharded"]
                assert len(sharded) == _LAYERS * len(expected) + 1, (side, sorted(sharded))
                for layer in range(_LAYERS):
                    for kind, (full, piece) in expected.items():
                        got = sharded[f"model.layers.{layer}.{kind}"]
                        assert (tuple(got[0]), tuple(got[1])) == (full, piece), (
                            side, world, rank, kind, got)
                head = sharded["lm_head.weight"]
                assert head[2] == "colwise_gather_output", head
                assert (tuple(head[0]), tuple(head[1])) == ((96, 48), (96 // world, 48)), head
                # One all_reduce after each rowwise layer (o_proj, down_proj),
                # one all_gather for the lm_head.
                assert rep["counts"] == {"all_reduce": 2 * _LAYERS, "all_gather": 1}, (
                    side, world, rank, rep["counts"])
            assert shim[rank]["sharded"] == gloo[rank]["sharded"], (world, rank)


def test_dropping_one_ranks_all_reduce_result_leaves_the_bound():
    """Sabotage, in the child: the last rank discards every all_reduce result."""
    for world in _WORLDS:
        sab = world - 1
        single_shim = _single(True, "float32")
        bound, _, _, _ = _tp_bound(world, sab)
        for side in (True, False):
            reps = _tp(side, world, "float32")
            assert reps[sab]["sabotage_counts"]["all_reduce"] == 2 * _LAYERS, reps[sab]
            ref = single_shim if side else _single(False, "float32")
            gap = _dist(reps[sab]["sabotaged_logits"], ref["logits"])
            assert gap > bound, (
                f"tp {world} {'shim' if side else 'gloo'}: rank {sab} ignored every "
                f"all_reduce result and its logits moved only {gap:.3e}, inside the "
                f"derived bound {bound:.3e}; the agreement tests could not see a "
                "missing collective")
        # The wrong answer is upstream's wrong answer too: the difference the
        # sabotage makes is the collective's result and nothing else.
        shim, gloo = _tp(True, world, "float32"), _tp(False, world, "float32")
        for rank in range(world):
            sbound, _, _, _ = _tp_bound(world, rank, "sabotaged_logits")
            d = _dist(shim[rank]["sabotaged_logits"], gloo[rank]["sabotaged_logits"])
            assert d <= sbound, (world, rank, d, sbound)


def test_the_users_from_pretrained_line_reaches_the_same_wall_on_both_sides():
    """`from_pretrained(..., distributed_config=..., device_mesh=mesh)`.

    Without `accelerate` transformers refuses in `check_and_set_device_map`,
    after `initialize_tensor_parallelism` has read the mesh. The shim must reach
    that same refusal, word for word, and not stop earlier. With `accelerate`
    installed on both sides the line must run and agree.
    """
    for world in _WORLDS:
        shim, gloo = _tp(True, world, "float32"), _tp(False, world, "float32")
        for rank in range(world):
            bound, _, _, _ = _tp_bound(world, rank)
            up, ours = gloo[rank]["user_line"], shim[rank]["user_line"]
            if "exc" in up:
                assert not up["accelerate"], up
                assert "requires `accelerate`" in up["exc"], up
                assert ours.get("exc") == up["exc"], (world, rank, ours, up)
            else:
                assert "ok" in ours, (world, rank, ours)
                assert _dist(ours["ok"], up["ok"]) <= bound, (world, rank)


def test_every_child_ran_the_torch_it_claims():
    """The premise of every number above, checked from outside the child."""
    assert _single(True, "float32")["_meta"]["shim"] is True
    for dtype in ("float32", "float64"):
        assert _single(False, dtype)["_meta"]["shim"] is False
    for world in _WORLDS:
        for rank, rep in enumerate(_tp(True, world, "float32")):
            assert rep["_meta"] == {"shim": True, "rank": rank}, rep["_meta"]
        for dtype in ("float32", "float64"):
            for rank, rep in enumerate(_tp(False, world, dtype)):
                assert rep["_meta"] == {"shim": False, "rank": rank}, rep["_meta"]


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = _skip.run_tests(items, suite="test_tpinfer")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
