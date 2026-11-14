"""Vertical federated learning: `torch.federated.FederatedFeatureParallel` (issue #293, SPEC S6.8).

The features are split across the ranks; every rank wraps a bottom module over
its own columns, and the label holder also holds the top module and the
labels. Upstream torch has no vertical-FL module, so there is no upstream
`FederatedFeatureParallel` to compare against. The oracles are:

* **(a) the joined model**: the same bottoms, the same top and the same SGD,
  trained in one process on the joined features with ordinary autograd. Run
  on the shim it must be **bit-equal** to the split run: the split run
  performs the same operations in the same order on bit-identical operands
  and only inserts copies (the wire is bit-exact, `test_p2p.py`). Run on
  upstream torch it is held within the tolerance AGENTS.md §16 derives:
  upstream's own float32-vs-float64 error on this same training run, floored
  at 8 ulp (`test_collect2.c2_assert_matches`).
* **(b) upstream torch + gloo doing the same exchange by hand**: every rank in
  its own process, the embedding sent with `dist.send`, the gradient returned
  with `dist.send`, `embedding.backward(gradient)` on the passive party. The
  shim is held within the tolerance derived from gloo's own float32-vs-float64
  run.

What crosses the ranks is asserted **from the backend**, not inferred: the
shim children wrap `transport.Mesh.send` (every point-to-point frame, decoded
from the bytes actually queued) and every collective method of
`ProcessGroupTorchnative`, and the raw feature rows, labels and sample ids are
searched for, byte for byte, in every payload sent while training.

Two nullifications run inside the suite (AGENTS.md §17.5): the gradient
returned to one passive party is replaced by zeros, and the batch digest is
replaced by a constant. Each must turn exactly the check it targets red.

Written before anything here was run (AGENTS.md rule 10: agents do not run
suites); the first run is CI's. Graded *builds* until then.
"""

import functools
import os

import gloo_loopback

from test_collect2 import _c2_spawn, c2_assert_matches, c2_exc, c2_ok
from test_p2p import _with_env


#: Party counts (= world sizes) the training tests run at. Three is where a
#: label holder that is not rank 0, and two passive parties of different
#: widths, can be told apart from "rank 0 and one peer".
VFL_PARTIES = (2, 3)


# ---------------------------------------------------------------------------
# The probe, shared verbatim by every side (shim and upstream)
# ---------------------------------------------------------------------------

_VFL_PROBE_SRC = r'''
import hashlib

VFL_BATCH = 4
VFL_STEPS = 3
VFL_CLASSES = 3
VFL_LR = 0.125
# Feature widths and embedding widths differ for every party, so a feature
# tensor and an embedding tensor never have the same shape.
VFL_LAYOUTS = {
    2: {"feats": (3, 5), "embs": (2, 3), "holder": 0},
    3: {"feats": (3, 5, 4), "embs": (2, 3, 2), "holder": 1},
}

def vfl_ints(key, n):
    state = int.from_bytes(hashlib.sha256(
        ("vfl-20261113/" + key).encode()).digest()[:4], "big") % 2147483647
    out = []
    for _ in range(n):
        state = (state * 1103515245 + 12345) % 2147483648
        out.append((state >> 11) % 1000 - 500)
    return out

def vfl_floats(key, n, scale):
    # Multiples of 2**-6 times `scale` (a power of two): exact in float32 and
    # float64, so both precisions start from the same numbers.
    return [(v % 256 - 128) / 64.0 * scale for v in vfl_ints(key, n)]

def vfl_plan(parties):
    lay = VFL_LAYOUTS[parties]
    n = VFL_BATCH * VFL_STEPS
    feats = [[vfl_floats("x%d/%d" % (k, i), f, 1.0) for i in range(n)]
             for k, f in enumerate(lay["feats"])]
    labels = [vfl_ints("y/%d" % i, 1)[0] % VFL_CLASSES for i in range(n)]
    order = sorted(range(n), key=lambda i: (vfl_ints("order/%d" % i, 1)[0], i))
    ids = [7001 + 13 * i for i in range(n)]
    total = sum(lay["embs"])
    weights = {}
    for k, (f, e) in enumerate(zip(lay["feats"], lay["embs"])):
        weights["W%d" % k] = vfl_floats("W%d" % k, e * f, 0.25)
        weights["b%d" % k] = vfl_floats("b%d" % k, e, 0.25)
    weights["Wt"] = vfl_floats("Wt", VFL_CLASSES * total, 0.25)
    weights["bt"] = vfl_floats("bt", VFL_CLASSES, 0.25)
    return {"lay": lay, "feats": feats, "labels": labels, "order": order,
            "ids": ids, "weights": weights, "total": total}

def vfl_rows(flat, rows, cols):
    return [flat[r * cols:(r + 1) * cols] for r in range(rows)]

def vfl_linear(torch, f, e, w, b, dtype):
    lin = torch.nn.Linear(f, e)
    lin.weight = torch.nn.Parameter(torch.tensor(vfl_rows(w, e, f), dtype=dtype))
    lin.bias = torch.nn.Parameter(torch.tensor(b, dtype=dtype))
    return lin

def vfl_bottom(torch, plan, k, dtype):
    f, e = plan["lay"]["feats"][k], plan["lay"]["embs"][k]
    w = plan["weights"]
    # The tanh is load-bearing for bit equality: its backward writes a fresh
    # contiguous tensor, so the bottom's Linear backward sees the same layout
    # whether its gradient came from a slice of cat's backward (joined) or
    # from a received contiguous copy (split).
    return torch.nn.Sequential(
        vfl_linear(torch, f, e, w["W%d" % k], w["b%d" % k], dtype),
        torch.nn.Tanh())

def vfl_top(torch, plan, dtype):
    w = plan["weights"]
    return vfl_linear(torch, plan["total"], VFL_CLASSES, w["Wt"], w["bt"], dtype)

def vfl_batch(torch, plan, k, s, dtype):
    idx = plan["order"][s * VFL_BATCH:(s + 1) * VFL_BATCH]
    x = torch.tensor([plan["feats"][k][i] for i in idx], dtype=dtype)
    ids = torch.tensor([plan["ids"][i] for i in idx], dtype=torch.int64)
    y = torch.tensor([plan["labels"][i] for i in idx], dtype=torch.int64)
    return x, ids, y

def vfl_flat(t):
    return None if t is None else t.detach().reshape(-1).tolist()

def vfl_state(prefix, module, grads):
    return {prefix + n: vfl_flat(p.grad if grads else p)
            for n, p in module.named_parameters()}

def vfl_record(rank, holder, bottom, top, grads):
    rec = vfl_state("bottom%d." % rank, bottom, grads)
    if rank == holder:
        rec.update(vfl_state("top.", top, grads))
    return rec

def vfl_joined(torch, parties, dtype):
    """Oracle (a): one process, the features joined, ordinary autograd."""
    F = torch.nn.functional
    plan = vfl_plan(parties)
    bottoms = [vfl_bottom(torch, plan, k, dtype) for k in range(parties)]
    top = vfl_top(torch, plan, dtype)
    opts = [torch.optim.SGD(m.parameters(), lr=VFL_LR) for m in bottoms + [top]]
    steps = []
    for s in range(VFL_STEPS):
        for o in opts:
            o.zero_grad()
        embs, y = [], None
        for k in range(parties):
            x, _ids, y = vfl_batch(torch, plan, k, s, dtype)
            embs.append(bottoms[k](x))
        loss = F.cross_entropy(top(torch.cat(embs, dim=1)), y)
        loss.backward()
        rec = {"loss": float(loss.item()), "grads": {}, "values": {}}
        for k, m in enumerate(bottoms):
            rec["grads"].update(vfl_state("bottom%d." % k, m, True))
        rec["grads"].update(vfl_state("top.", top, True))
        for o in opts:
            o.step()
        for k, m in enumerate(bottoms):
            rec["values"].update(vfl_state("bottom%d." % k, m, False))
        rec["values"].update(vfl_state("top.", top, False))
        steps.append(rec)
    return steps

def vfl_by_hand(torch, dist, rank, parties, dtype):
    """Oracle (b): the same exchange written out with send/recv."""
    F = torch.nn.functional
    plan = vfl_plan(parties)
    holder = plan["lay"]["holder"]
    bottom = vfl_bottom(torch, plan, rank, dtype)
    top = vfl_top(torch, plan, dtype) if rank == holder else None
    params = list(bottom.parameters()) + (list(top.parameters()) if top else [])
    opt = torch.optim.SGD(params, lr=VFL_LR)
    steps = []
    for s in range(VFL_STEPS):
        opt.zero_grad()
        x, _ids, y = vfl_batch(torch, plan, rank, s, dtype)
        emb = bottom(x)
        rec = {"loss": None}
        if rank != holder:
            dist.send(emb.detach().contiguous(), holder, tag=1)
            grad = torch.zeros(tuple(emb.shape), dtype=dtype)
            dist.recv(grad, holder, tag=2)
            emb.backward(grad)
        else:
            parts, leaves = [], {}
            for k in range(parties):
                if k == rank:
                    parts.append(emb)
                    continue
                buf = torch.zeros(VFL_BATCH, plan["lay"]["embs"][k], dtype=dtype)
                dist.recv(buf, k, tag=1)
                buf.requires_grad_(True)
                leaves[k] = buf
                parts.append(buf)
            loss = F.cross_entropy(top(torch.cat(parts, dim=1)), y)
            loss.backward()
            for k in sorted(leaves):
                dist.send(leaves[k].grad.contiguous(), k, tag=2)
            rec["loss"] = float(loss.item())
        rec["grads"] = vfl_record(rank, holder, bottom, top, True)
        opt.step()
        rec["values"] = vfl_record(rank, holder, bottom, top, False)
        steps.append(rec)
    return steps
'''


# ---------------------------------------------------------------------------
# The shim side: FederatedFeatureParallel, with the backend instrumented
# ---------------------------------------------------------------------------

_VFL_SHIM_PART = r'''
def vfl_instrument():
    """Record every frame and every collective the backend is handed."""
    from torchnative.distributed.backend import (
        ProcessGroupTorchnative, transport, wire)
    log = {"phase": "init", "frames": [], "collectives": []}
    original_send = transport.Mesh.send

    def logged_send(self, dst, tag, data, tensor=None):
        frame = wire.decode(data, self.rank)
        log["frames"].append({
            "phase": log["phase"], "dst": dst, "tag": frame.tag,
            "dtype": frame.dtype, "shape": list(frame.shape),
            "bytes": len(data), "payload": frame.payload})
        return original_send(self, dst, tag, data, tensor)

    transport.Mesh.send = logged_send
    names = ("allreduce", "allreduce_partial", "allreduce_coalesced",
             "broadcast", "allgather", "allgather_partial", "_allgather_base",
             "allgather_coalesced", "all_gather_single", "gather", "scatter",
             "reduce", "reduce_scatter", "_reduce_scatter_base",
             "reduce_scatter_single", "alltoall", "alltoall_base", "barrier")
    wrapped = []
    for name in names:
        original = getattr(ProcessGroupTorchnative, name, None)
        if original is None:
            continue
        def make(original=original, name=name):
            def inner(self, *args, **kwargs):
                log["collectives"].append({"phase": log["phase"], "name": name})
                return original(self, *args, **kwargs)
            return inner
        setattr(ProcessGroupTorchnative, name, make())
        wrapped.append(name)
    log["wrapped"] = wrapped
    return log

def vfl_ffp_train(torch, rank, parties, dtype, log, mutant):
    import torch.federated as tfed
    import torchnative.federated as nfed
    from torchnative.federated import feature_parallel as fp
    from torchnative.distributed.backend import wire
    F = torch.nn.functional
    plan = vfl_plan(parties)
    holder = plan["lay"]["holder"]
    res = {"same_class": tfed.FederatedFeatureParallel is nfed.FederatedFeatureParallel,
           "mutant": mutant}
    if mutant == "zero_grad_to_last":
        # Nullification 1: the label holder returns zeros to the last rank
        # instead of its embedding's gradient. Every other message is intact.
        original = fp.FederatedFeatureParallel._send_gradient
        def zeroed(self, peer, grad):
            if peer == parties - 1:
                grad = torch.zeros_like(grad)
            return original(self, peer, grad)
        fp.FederatedFeatureParallel._send_gradient = zeroed
    bottom = vfl_bottom(torch, plan, rank, dtype)
    top = vfl_top(torch, plan, dtype) if rank == holder else None
    log["phase"] = "construct"
    ffp = tfed.FederatedFeatureParallel(bottom, label_holder=holder, top=top)
    res["tag_base"] = ffp.tag_base
    res["is_label_holder"] = ffp.is_label_holder
    opt = torch.optim.SGD(ffp.parameters(), lr=VFL_LR)
    log["phase"] = "train"
    forbidden = []
    steps, emb_sha = [], []
    for s in range(VFL_STEPS):
        opt.zero_grad()
        x, ids, y = vfl_batch(torch, plan, rank, s, dtype)
        forbidden.append(("features", wire.tensor_bytes(x)[2]))
        for i in range(x.shape[0]):
            forbidden.append(("feature row %d" % i,
                              wire.tensor_bytes(x[i].contiguous())[2]))
        forbidden.append(("labels", wire.tensor_bytes(y)[2]))
        forbidden.append(("sample ids", wire.tensor_bytes(ids)[2]))
        out = ffp(x, sample_ids=ids)
        rec = {"loss": None}
        if ffp.is_label_holder:
            loss = F.cross_entropy(out, y)
            rec["loss"] = float(loss.item())
            if s == 0:
                loss.backward()        # the plain spelling
            else:
                ffp.backward(loss)     # the spelling that also settles
        else:
            emb_sha.append(hashlib.sha256(
                wire.tensor_bytes(out.detach())[2]).hexdigest())
            ffp.backward()
        rec["grads"] = vfl_record(rank, holder, ffp.module, ffp.top, True)
        opt.step()
        rec["values"] = vfl_record(rank, holder, ffp.module, ffp.top, False)
        steps.append(rec)
    log["phase"] = "done"
    res["steps"] = steps
    res["emb_sha"] = emb_sha
    res["logging"] = ffp._get_federated_logging_data()
    frames, leaks = [], []
    for f in log["frames"]:
        frames.append({"phase": f["phase"], "dst": f["dst"], "tag": f["tag"],
                       "dtype": f["dtype"], "shape": f["shape"],
                       "bytes": f["bytes"],
                       "sha": hashlib.sha256(f["payload"]).hexdigest()})
        if f["phase"] != "train":
            continue
        for what, raw in forbidden:
            if raw and raw in f["payload"]:
                leaks.append({"what": what, "tag": f["tag"], "shape": f["shape"]})
    res["frames"] = frames
    res["leaks"] = leaks
    res["forbidden_checked"] = len(forbidden)
    res["collectives"] = log["collectives"]
    res["wrapped"] = log["wrapped"]
    return res

def vfl_exc(fn):
    try:
        value = fn()
    except Exception as exc:
        return {"exc": "%s: %s" % (type(exc).__name__, exc),
                "class": type(exc).__name__,
                "is_value_error": isinstance(exc, ValueError),
                "mismatched": list(getattr(exc, "mismatched", ()) or ()),
                "step": getattr(exc, "step", None)}
    return {"exc": None, "value": value}

def vfl_refusals(torch, rank, parties, out, probe, mutant):
    import torch.federated as tfed
    FFP = tfed.FederatedFeatureParallel
    F = torch.nn.functional
    plan = vfl_plan(parties)
    holder = plan["lay"]["holder"]
    last = parties - 1
    f32 = torch.float32

    def bottom():
        return vfl_bottom(torch, plan, rank, f32)

    def top():
        return vfl_top(torch, plan, f32) if rank == holder else None

    def fresh():
        return FFP(bottom(), label_holder=holder, top=top())

    def batch(s):
        return vfl_batch(torch, plan, rank, s, f32)

    def step(ffp, s):
        x, ids, y = batch(s)
        out_ = ffp(x, sample_ids=ids)
        if ffp.is_label_holder:
            ffp.backward(F.cross_entropy(out_, y))
        else:
            ffp.backward()
        return "stepped"

    # -- misaligned batches (the refusal this class exists to make) ----------
    ffp = fresh()
    probe("misaligned_before", lambda: step(ffp, 0))

    def misaligned():
        x, ids, y = batch(1)
        if rank == last:
            # The same samples in reverse order: a set comparison would pass.
            ids = torch.tensor(list(reversed(ids.tolist())), dtype=torch.int64)
        def run():
            out_ = ffp(x, sample_ids=ids)
            # Reached only if the step was accepted: finish it, so the ranks
            # stay in step for whatever runs next.
            if ffp.is_label_holder:
                ffp.backward(F.cross_entropy(out_, y))
            else:
                ffp.backward()
            return "ACCEPTED"
        return vfl_exc(run)
    probe("misaligned", misaligned)
    if mutant == "no_digest":
        return
    probe("misaligned_after", lambda: step(ffp, 2))

    # -- one rank in no-grad mode --------------------------------------------
    ffp2 = fresh()
    def grad_mode():
        x, ids, _y = batch(0)
        def run():
            if rank == last:
                with torch.no_grad():
                    ffp2(x, sample_ids=ids)
            else:
                ffp2(x, sample_ids=ids)
            return "ACCEPTED"
        return vfl_exc(run)
    probe("grad_mode_mismatch", grad_mode)

    # -- a loss that reaches no passive embedding ----------------------------
    ffp3 = fresh()
    def not_reached():
        x, ids, _y = batch(0)
        out_ = ffp3(x, sample_ids=ids)
        def run():
            if ffp3.is_label_holder:
                ffp3.backward((ffp3.top.weight * ffp3.top.weight).sum())
            else:
                ffp3.backward()
            return "ACCEPTED"
        return vfl_exc(run)
    probe("not_reached", not_reached)
    probe("not_reached_after", lambda: step(ffp3, 1))

    # -- refusals made locally, before anything is sent ----------------------
    ffp4 = fresh()
    x0, ids0, _y0 = batch(0)
    probe("missing_ids", lambda: repr(ffp4(x0)))
    probe("float_ids", lambda: repr(ffp4(x0, sample_ids=torch.tensor(
        [0.5] * VFL_BATCH, dtype=torch.float32))))
    probe("backward_misuse", lambda: repr(
        ffp4.backward() if rank == holder else ffp4.backward(torch.tensor(1.0))))
    probe("backward_before_forward", lambda: (
        "holder" if rank == holder else repr(ffp4.backward())))

    class MetaBottom(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.w = torch.nn.Parameter(torch.zeros(1))
        def forward(self, x):
            return torch.empty(x.shape[0], 2, device="meta")
    ffp5 = FFP(MetaBottom(), label_holder=holder, top=top())
    probe("non_cpu", lambda: repr(ffp5(x0, sample_ids=ids0)))

    # -- constructor refusals: every rank refuses, before any collective -----
    probe("no_label_holder", lambda: repr(FFP(bottom(), top=top())))
    probe("label_holder_out_of_range",
          lambda: repr(FFP(bottom(), label_holder=parties + 3, top=top())))
    probe("top_misplaced", lambda: repr(FFP(
        bottom(), label_holder=holder,
        top=None if rank == holder else vfl_top(torch, plan, f32))))
    probe("merge_sum", lambda: repr(FFP(bottom(), label_holder=holder,
                                        top=top(), merge="sum")))
    probe("secure_aggregation", lambda: repr(FFP(
        bottom(), label_holder=holder, top=top(), secure_aggregation=True)))
    probe("differential_privacy", lambda: repr(FFP(
        bottom(), label_holder=holder, top=top(), differential_privacy=1.0)))
    probe("encryption", lambda: repr(FFP(
        bottom(), label_holder=holder, top=top(), encryption="aes-gcm")))
    probe("group", lambda: repr(FFP(bottom(), label_holder=holder,
                                    top=top(), group=object())))
    probe("module_none", lambda: repr(FFP(None, label_holder=holder, top=top())))
    # The instance counter did not move on any refusal above, so a fresh
    # instance still agrees across the ranks and still trains.
    probe("after_refusals", lambda: step(fresh(), 0))
'''


_VFL_SHIM_SRC = (
    "import datetime, json, os, sys, time\n"
    "import torch\n"
    'assert hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded upstream torch, not the shim")\n'
    "import torch.distributed as dist\n"
    "import torchnative.distributed  # noqa: F401 -- registers backend='torchnative'\n"
    + _VFL_PROBE_SRC
    + _VFL_SHIM_PART
    + r'''
rank, port, world, dest, dtype_name = (int(sys.argv[1]), int(sys.argv[2]),
                                       int(sys.argv[3]), sys.argv[4], sys.argv[5])
mode = os.environ.get("VFL_MODE", "train")
parties = int(os.environ.get("VFL_PARTIES", "2"))
dtype = getattr(torch, dtype_name)
out = {}

def probe(key, fn):
    try:
        out[key] = {"ok": fn()}
    except Exception as exc:
        out[key] = {"exc": "%s: %s" % (type(exc).__name__, exc)}

if mode == "joined":
    out["steps"] = vfl_joined(torch, parties, dtype)
else:
    log = vfl_instrument()
    if mode in ("refuse", "one"):
        import torch.federated as tfed
        probe("no_group", lambda: repr(tfed.FederatedFeatureParallel(
            torch.nn.Linear(2, 2), label_holder=0, top=torch.nn.Linear(2, 2))))
    dist.init_process_group(backend="torchnative",
                            init_method="tcp://127.0.0.1:%d" % port,
                            rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=120))
    if mode in ("train", "zero_grad_to_last"):
        out.update(vfl_ffp_train(torch, rank, parties, dtype, log,
                                 None if mode == "train" else mode))
    elif mode in ("refuse", "no_digest"):
        if mode == "no_digest":
            # Nullification 2: every batch digests to the same number.
            from torchnative.federated import feature_parallel as fp
            fp._batch_digest = lambda ids: 0
        vfl_refusals(torch, rank, parties, out, probe,
                     None if mode == "refuse" else mode)
    elif mode == "one":
        import torch.federated as tfed
        probe("world_of_one", lambda: repr(tfed.FederatedFeatureParallel(
            torch.nn.Linear(2, 2), label_holder=0, top=torch.nn.Linear(2, 2))))
out["_meta"] = {"shim": True, "rank": rank, "world": world, "mode": mode,
                "parties": parties, "dtype": dtype_name,
                "backend": dist.get_backend() if dist.is_initialized() else None}
with open(dest, "w") as handle:
    json.dump(out, handle)
'''
)


_VFL_GLOO_SRC = (
    gloo_loopback.child_pin_source()
    + "import json, os, sys, time\n"
    "import torch\n"
    'assert not hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded the shim, not upstream torch")\n'
    "import torch.distributed as dist\n"
    + _VFL_PROBE_SRC
    + r'''
rank, port, world, dest, dtype_name = (int(sys.argv[1]), int(sys.argv[2]),
                                       int(sys.argv[3]), sys.argv[4], sys.argv[5])
mode = os.environ.get("VFL_MODE", "hand")
parties = int(os.environ.get("VFL_PARTIES", "2"))
dtype = getattr(torch, dtype_name)
out = {}
backend = None
if mode == "joined":
    out["steps"] = vfl_joined(torch, parties, dtype)
else:
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world)
    backend = "gloo"
    out["steps"] = vfl_by_hand(torch, dist, rank, parties, dtype)
    dist.destroy_process_group()
out["_meta"] = {"shim": False, "rank": rank, "world": world, "mode": mode,
                "parties": parties, "dtype": dtype_name, "backend": backend}
with open(dest, "w") as handle:
    json.dump(out, handle)
'''
)


def _spawn(source, world, dtype, shim, what, mode, parties):
    return _with_env(
        {"VFL_MODE": mode, "VFL_PARTIES": str(parties),
         "TORCHNATIVE_DIST_TRANSPORT": None, "TORCHNATIVE_DIST_BIND_ADDR": None},
        lambda: _c2_spawn(source, world, dtype, shim, what))


@functools.cache
def vfl_shim(parties, mode="train"):
    return _spawn(_VFL_SHIM_SRC, parties, "float32", True,
                  "vfl-shim-%s-%d" % (mode, parties), mode, parties)


@functools.cache
def vfl_shim_joined(parties):
    return _spawn(_VFL_SHIM_SRC, 1, "float32", True,
                  "vfl-shim-joined-%d" % parties, "joined", parties)[0]


@functools.cache
def vfl_up_joined(parties, dtype):
    return _spawn(_VFL_GLOO_SRC, 1, dtype, False,
                  "vfl-up-joined-%s-%d" % (dtype, parties), "joined", parties)[0]


@functools.cache
def vfl_gloo_hand(parties, dtype):
    return _spawn(_VFL_GLOO_SRC, parties, dtype, False,
                  "vfl-gloo-hand-%s-%d" % (dtype, parties), "hand", parties)


@functools.cache
def vfl_shim_one():
    return _spawn(_VFL_SHIM_SRC, 1, "float32", True, "vfl-shim-one", "one", 1)[0]


def _helpers():
    namespace = {}
    exec(compile(_VFL_PROBE_SRC, "<vfl probe>", "exec"), namespace)
    return namespace


def _layout(parties):
    return _helpers()["VFL_LAYOUTS"][parties]


def _expected_names(rank, holder):
    names = {"bottom%d.0.weight" % rank, "bottom%d.0.bias" % rank}
    if rank == holder:
        names |= {"top.weight", "top.bias"}
    return names


def _exact_mismatches(report, joined, rank, holder):
    """Every (step, kind, name) where this rank's run differs from the joined model."""
    bad, seen = [], 0
    assert len(report["steps"]) == len(joined["steps"]) == 3, (
        len(report["steps"]), len(joined["steps"]))
    for s, (ours, theirs) in enumerate(zip(report["steps"], joined["steps"])):
        if rank == holder:
            seen += 1
            if ours["loss"] != theirs["loss"]:
                bad.append((s, "loss", "loss"))
        else:
            assert ours["loss"] is None, ("a passive party reported a loss", ours)
        for kind in ("grads", "values"):
            assert set(ours[kind]) == _expected_names(rank, holder), (
                rank, kind, sorted(ours[kind]))
            for name, value in ours[kind].items():
                assert value is not None, ("no %s for %s at step %d on rank %d"
                                           % (kind, name, s, rank))
                seen += 1
                if value != theirs[kind][name]:
                    bad.append((s, kind, name))
    return bad, seen


def _quantities(rank, holder):
    """How many numbers-lists `_exact_mismatches` compares on one rank."""
    per_step = 2 * len(_expected_names(rank, holder)) + (1 if rank == holder else 0)
    return 3 * per_step


# ---------------------------------------------------------------------------
# (a) the joined single-process model
# ---------------------------------------------------------------------------

def test_split_training_equals_the_joined_model_bit_for_bit_at_two_and_three_parties():
    """Loss, every parameter's gradient and every updated value, for three SGD steps.

    Bit equality, not a tolerance: the split run does the joined run's
    operations, in its order, on bit-identical operands (the wire copies
    bytes), so there is no rounding difference to allow for. A tolerance here
    would be chosen, which AGENTS.md §16 forbids.
    """
    for parties in VFL_PARTIES:
        holder = _layout(parties)["holder"]
        joined = vfl_shim_joined(parties)
        reports = vfl_shim(parties)
        total = 0
        for rank in range(parties):
            bad, seen = _exact_mismatches(reports[rank], joined, rank, holder)
            assert not bad, (
                "parties %d rank %d differs from the joined model at %s"
                % (parties, rank, bad))
            assert seen == _quantities(rank, holder), (parties, rank, seen)
            total += seen
        assert total == sum(_quantities(r, holder) for r in range(parties))


def test_the_label_holder_is_not_rank_zero_at_three_parties_and_its_slot_is_its_rank():
    """At three parties the label holder is rank 1, so rank order is not 'holder first'."""
    assert _layout(3)["holder"] == 1
    reports = vfl_shim(3)
    log = reports[1]["logging"]
    assert reports[1]["is_label_holder"] is True, reports[1]
    assert log["widths"] == list(_layout(3)["embs"]), log
    for rank in (0, 2):
        assert reports[rank]["is_label_holder"] is False, reports[rank]


def test_split_training_agrees_with_the_upstream_joined_model_within_its_derived_tolerance():
    """The same comparison against upstream torch, at upstream's own f32-vs-f64 error."""
    for parties in VFL_PARTIES:
        holder = _layout(parties)["holder"]
        up32, up64 = vfl_up_joined(parties, "float32"), vfl_up_joined(parties, "float64")
        reports = vfl_shim(parties)
        seen = 0
        for rank in range(parties):
            for s in range(3):
                ours = reports[rank]["steps"][s]
                if rank == holder:
                    c2_assert_matches("loss step %d" % s, parties, rank, ours["loss"],
                                      up32["steps"][s]["loss"], up64["steps"][s]["loss"])
                    seen += 1
                for kind in ("grads", "values"):
                    for name in sorted(_expected_names(rank, holder)):
                        c2_assert_matches(
                            "%s %s step %d" % (kind, name, s), parties, rank,
                            ours[kind][name], up32["steps"][s][kind][name],
                            up64["steps"][s][kind][name])
                        seen += 1
        assert seen == sum(_quantities(r, holder) for r in range(parties)), seen


# ---------------------------------------------------------------------------
# (b) upstream torch + gloo, the exchange written by hand
# ---------------------------------------------------------------------------

def test_the_gloo_exchange_by_hand_is_the_upstream_joined_model_bit_for_bit():
    """The oracle checked against the other oracle, so neither is trusted alone."""
    for parties in VFL_PARTIES:
        holder = _layout(parties)["holder"]
        joined = vfl_up_joined(parties, "float32")
        hand = vfl_gloo_hand(parties, "float32")
        for rank in range(parties):
            bad, seen = _exact_mismatches(hand[rank], joined, rank, holder)
            assert not bad, (
                "upstream by hand differs from upstream joined at parties %d "
                "rank %d: %s" % (parties, rank, bad))
            assert seen == _quantities(rank, holder)


def test_split_training_agrees_with_upstream_gloo_doing_the_exchange_by_hand():
    for parties in VFL_PARTIES:
        holder = _layout(parties)["holder"]
        g32, g64 = vfl_gloo_hand(parties, "float32"), vfl_gloo_hand(parties, "float64")
        reports = vfl_shim(parties)
        seen = 0
        for rank in range(parties):
            for s in range(3):
                ours = reports[rank]["steps"][s]
                if rank == holder:
                    c2_assert_matches("loss step %d" % s, parties, rank, ours["loss"],
                                      g32[rank]["steps"][s]["loss"],
                                      g64[rank]["steps"][s]["loss"])
                    seen += 1
                for kind in ("grads", "values"):
                    for name in sorted(_expected_names(rank, holder)):
                        c2_assert_matches(
                            "%s %s step %d" % (kind, name, s), parties, rank,
                            ours[kind][name], g32[rank]["steps"][s][kind][name],
                            g64[rank]["steps"][s][kind][name])
                        seen += 1
        assert seen == sum(_quantities(r, holder) for r in range(parties)), seen


# ---------------------------------------------------------------------------
# What crosses the ranks, read off the backend
# ---------------------------------------------------------------------------

def _train_frames(report):
    base = report["tag_base"]
    return [dict(f, kind=f["tag"] - base) for f in report["frames"]
            if f["phase"] == "train"]


def test_only_embedding_and_gradient_shaped_tensors_cross_ranks():
    """Every frame sent while training, decoded from the bytes the mesh was handed.

    Per step, a passive party sends the label holder exactly a `(6,)` int64
    header and a `(batch, its width)` embedding; the label holder sends each
    passive party exactly a `(5,)` int64 verdict, a `(2,)` int64 status and a
    `(batch, that party's width)` gradient. Nothing goes between passive
    parties, no collective runs while training, and the one collective of
    construction is the configuration digest.
    """
    tags = {"header": 0, "verdict": 1, "embedding": 2, "status": 3, "gradient": 4}
    for parties in VFL_PARTIES:
        lay = _layout(parties)
        holder = lay["holder"]
        reports = vfl_shim(parties)
        for rank in range(parties):
            r = reports[rank]
            assert r["wrapped"], "no collective method was instrumented"
            assert [c for c in r["collectives"] if c["phase"] == "train"] == [], (
                parties, rank, r["collectives"])
            assert [c["name"] for c in r["collectives"]
                    if c["phase"] == "construct"] == ["allgather"], r["collectives"]
            assert [f for f in r["frames"] if f["phase"] == "construct"] == []
            frames = _train_frames(r)
            if rank != holder:
                want = []
                for _s in range(3):
                    want.append((holder, tags["header"], "int64", [6]))
                    want.append((holder, tags["embedding"], "float32",
                                 [4, lay["embs"][rank]]))
                got = [(f["dst"], f["kind"], f["dtype"], f["shape"]) for f in frames]
                assert got == want, (parties, rank, got)
                # The embedding frame is this rank's embedding, byte for byte.
                shas = [f["sha"] for f in frames if f["kind"] == tags["embedding"]]
                assert shas == r["emb_sha"], (parties, rank)
            else:
                got = sorted((f["dst"], f["kind"], f["dtype"], f["shape"])
                             for f in frames)
                want = []
                for peer in range(parties):
                    if peer == holder:
                        continue
                    for _s in range(3):
                        want += [(peer, tags["verdict"], "int64", [5]),
                                 (peer, tags["status"], "int64", [2]),
                                 (peer, tags["gradient"], "float32",
                                  [4, lay["embs"][peer]])]
                assert got == sorted(want), (parties, rank, got)
            # No feature width appears as the width of anything sent.
            for f in frames:
                assert f["shape"] != [4, lay["feats"][rank]], f


def test_no_raw_feature_label_or_sample_id_byte_string_is_in_any_payload():
    """Searched for byte for byte: whole batches and every feature row."""
    for parties in VFL_PARTIES:
        for rank in range(parties):
            r = vfl_shim(parties)[rank]
            # 3 steps x (features + 4 rows + labels + ids).
            assert r["forbidden_checked"] == 3 * 7, r["forbidden_checked"]
            assert len(_train_frames(r)) > 0, "nothing was sent, so nothing was searched"
            assert r["leaks"] == [], (parties, rank, r["leaks"])


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

@functools.cache
def vfl_refuse(parties):
    return vfl_shim(parties, "refuse")


def test_a_misaligned_batch_refuses_by_name_on_every_rank_and_the_next_step_recovers():
    """One rank feeds the same samples in reverse order: every rank refuses, nothing is sent."""
    for parties in VFL_PARTIES:
        last = parties - 1
        reports = vfl_refuse(parties)
        for rank in range(parties):
            assert c2_ok(reports[rank], "misaligned_before") == "stepped"
            got = c2_ok(reports[rank], "misaligned")
            assert got["exc"] is not None, (
                "parties %d rank %d accepted a misaligned batch" % (parties, rank))
            assert got["class"] == "SamplesMisaligned", got
            assert got["is_value_error"] is True, got
            assert got["mismatched"] == [last], got
            assert got["step"] == 1, got
            assert "No embedding was sent" in got["exc"], got
            assert "sample ids" in got["exc"], got
            assert c2_ok(reports[rank], "misaligned_after") == "stepped"


def test_a_rank_in_no_grad_mode_refuses_the_step_on_every_rank():
    for parties in VFL_PARTIES:
        for rank in range(parties):
            got = c2_ok(vfl_refuse(parties)[rank], "grad_mode_mismatch")
            assert got["exc"] is not None, got
            assert got["class"] == "ValueError", got
            assert "whether gradients are enabled" in got["exc"], got


def test_a_loss_that_reaches_no_embedding_refuses_on_both_sides_instead_of_hanging():
    for parties in VFL_PARTIES:
        holder = _layout(parties)["holder"]
        for rank in range(parties):
            got = c2_ok(vfl_refuse(parties)[rank], "not_reached")
            assert got["exc"] is not None, got
            assert got["class"] == "RuntimeError", got
            assert "did not reach" in got["exc"], got
            if rank == holder:
                assert "Those ranks were told" in got["exc"], got
            assert c2_ok(vfl_refuse(parties)[rank], "not_reached_after") == "stepped"


def test_local_misuse_refuses_by_name_before_anything_is_sent():
    for parties in VFL_PARTIES:
        holder = _layout(parties)["holder"]
        for rank in range(parties):
            r = vfl_refuse(parties)[rank]
            msg = c2_exc(r, "missing_ids")
            assert msg.startswith("TypeError:") and "sample_ids= is required" in msg, msg
            msg = c2_exc(r, "float_ids")
            assert msg.startswith("TypeError:") and "integer tensor" in msg, msg
            msg = c2_exc(r, "backward_misuse")
            if rank == holder:
                assert msg.startswith("TypeError:") and "no loss" in msg, msg
            else:
                assert msg.startswith("ValueError:") and "holds no labels" in msg, msg
                msg = c2_exc(r, "backward_before_forward")
                assert msg.startswith("RuntimeError:") and "no step awaiting" in msg, msg
            msg = c2_exc(r, "non_cpu")
            assert msg.startswith("NotImplementedError:"), msg
            assert "is on meta" in msg and "S6.10" in msg, msg


def test_the_constructor_refuses_by_name_what_is_not_built():
    for parties in VFL_PARTIES:
        holder = _layout(parties)["holder"]
        for rank in range(parties):
            r = vfl_refuse(parties)[rank]
            msg = c2_exc(r, "no_label_holder")
            assert msg.startswith("TypeError:") and "no label_holder" in msg, msg
            msg = c2_exc(r, "label_holder_out_of_range")
            assert msg.startswith("ValueError:") and "is not a rank" in msg, msg
            msg = c2_exc(r, "top_misplaced")
            if rank == holder:
                assert msg.startswith("TypeError:") and "no top=" in msg, msg
            else:
                assert msg.startswith("ValueError:") and "passive party" in msg, msg
            msg = c2_exc(r, "merge_sum")
            assert msg.startswith("NotImplementedError:") and "'sum'" in msg, msg
            for key, word in (("secure_aggregation", "homomorphic"),
                              ("differential_privacy", "accountant"),
                              ("encryption", "key exchange"),
                              ("group", "issue #221")):
                msg = c2_exc(r, key)
                assert msg.startswith("NotImplementedError:"), (key, msg)
                assert key + "=" in msg and word in msg, (key, msg)
            msg = c2_exc(r, "module_none")
            assert msg.startswith("TypeError:") and "torch.nn.Module" in msg, msg
            assert c2_ok(r, "after_refusals") == "stepped"


def test_no_process_group_and_a_world_of_one_refuse_by_name():
    for parties in VFL_PARTIES:
        for rank in range(parties):
            msg = c2_exc(vfl_refuse(parties)[rank], "no_group")
            assert msg.startswith("RuntimeError:") and "no process group" in msg, msg
    one = vfl_shim_one()
    msg = c2_exc(one, "no_group")
    assert "no process group" in msg, msg
    msg = c2_exc(one, "world_of_one")
    assert msg.startswith("NotImplementedError:"), msg
    assert "a world of one" in msg and "FederatedFeatureParallel" in msg, msg


# ---------------------------------------------------------------------------
# Nullifications (AGENTS.md §17.5)
# ---------------------------------------------------------------------------

def test_nullified_zero_gradient_to_one_party_is_caught_by_the_joined_comparison():
    """The label holder sends rank 2 zeros instead of its gradient.

    The comparison of the first test must go red, and first where it should:
    at step 0 every gradient and value of rank 2's bottom module differs and
    nothing on rank 0 or on the label holder (rank 1) does, because their
    step-0 arithmetic is untouched. From step 1 on rank 2's stale bottom
    feeds a different embedding into the top module, so the other ranks may
    differ too; that is propagation, not locality, and is not asserted
    either way. A comparison that went red everywhere at step 0 would not
    show that it reads the gradient that came back.
    """
    parties = 3
    holder = _layout(parties)["holder"]
    joined = vfl_shim_joined(parties)
    reports = vfl_shim(parties, "zero_grad_to_last")
    for rank in (0, 1):
        bad, _ = _exact_mismatches(reports[rank], joined, rank, holder)
        assert [b for b in bad if b[0] == 0] == [], (rank, bad)
    bad, _ = _exact_mismatches(reports[2], joined, 2, holder)
    want = [(s, kind, name) for s in range(3) for kind in ("grads", "values")
            for name in sorted(_expected_names(2, holder))]
    assert sorted(bad) == sorted(want), bad
    # The mutant really ran: rank 2's gradients are exactly zero.
    for s in range(3):
        for name, value in reports[2]["steps"][s]["grads"].items():
            assert all(v == 0.0 for v in value), (s, name, value)


def test_nullified_batch_digest_lets_a_misaligned_batch_through():
    """With the digest replaced by a constant, the reversed batch is accepted.

    So the refusal in the misalignment test is the digest's, not a side
    effect of something else in the step.
    """
    reports = vfl_shim(2, "no_digest")
    for rank in range(2):
        assert c2_ok(reports[rank], "misaligned_before") == "stepped"
        got = c2_ok(reports[rank], "misaligned")
        assert got["exc"] is None and got["value"] == "ACCEPTED", got
    # And without the mutant, the same probe refuses (the control).
    for rank in range(2):
        assert c2_ok(vfl_refuse(2)[rank], "misaligned")["exc"] is not None


# ---------------------------------------------------------------------------

def test_the_add_hook_exports_the_same_class():
    for parties in VFL_PARTIES:
        for rank in range(parties):
            assert vfl_shim(parties)[rank]["same_class"] is True


def test_every_rank_really_ran_the_shim_and_the_oracles_really_ran_upstream():
    for parties in VFL_PARTIES:
        for rank in range(parties):
            meta = vfl_shim(parties)[rank]["_meta"]
            assert (meta["shim"], meta["rank"], meta["world"], meta["backend"]) == (
                True, rank, parties, "torchnative"), meta
            for dtype in ("float32", "float64"):
                meta = vfl_gloo_hand(parties, dtype)[rank]["_meta"]
                assert (meta["shim"], meta["rank"], meta["backend"], meta["dtype"]) == (
                    False, rank, "gloo", dtype), meta
        assert vfl_shim_joined(parties)["_meta"]["shim"] is True
        for dtype in ("float32", "float64"):
            assert vfl_up_joined(parties, dtype)["_meta"]["shim"] is False


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
