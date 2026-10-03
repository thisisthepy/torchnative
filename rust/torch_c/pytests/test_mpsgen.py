"""Greedy `generate` on mps, and the two walls it stopped at (issue #29).

Before this round greedy `generate` had never run on `mps` in this repository.
The wall list was **measured, not inferred**: a tiny Qwen3 (the config
`test_qwen3.py` uses) generated 20 greedy tokens on mps under a
`TorchDispatchMode` that routed every refused op through the CPU and recorded
it, until the run completed (`.scratch/walls.py` in the round that wrote this
file, 2026-10-03, main checkout's shim at 652c7d7). Every configuration
measured -- float32/bfloat16/float16 x sdpa/eager -- stopped at the same walls,
in this order:

    sdpa                                   eager adds
    1. aten.isin.Tensor_Tensor             aten.bitwise_and.Tensor  masking_utils.py:56 and_mask
       generation/utils.py:2094 _prepare_special_tokens
       generation/stopping_criteria.py:580 EosTokenCriteria
    2. torch.tensor([], device=mps)        aten.index.Tensor        masking_utils.py:177 inner_mask
       cache_utils.py:123-124 DynamicLayer.lazy_initialization
    3. aten.argmax.default                 generation/utils.py:2925 _sample
    4. aten.bitwise_or.Tensor              generation/stopping_criteria.py:610
    5. aten.bitwise_not.default            generation/utils.py:2936 _sample
    6. aten.bitwise_and.Tensor             generation/utils.py:2936 _sample (int64 & bool)
    7. aten.max.default                    generation/utils.py:2937 _sample

With all of them routed, the tokens matched the CPU run in every
configuration. **What that measurement cannot see**: the real 0.6B
checkpoint's path, `do_sample=True` (`multinomial` is on the refusal list), and
anything a longer prompt or batch > 1 reaches. Those are unmeasured.

What this file holds, at which grade:

* *agrees* -- greedy tokens against upstream, and each rewritten op against
  upstream element-wise, with the oracle in a **separate subprocess** that
  asserts it imported upstream torch. Every mps computation also runs in its
  own subprocess: a panic inside candle's Metal backend poisons the encoder's
  mutex and every later Metal op in that process panics too (measured on the
  zero-element walls), so one failure must not fail the rest of the file.
* *ran on the GPU* -- `_metal_counters()` bracketing only the dispatch:
  compute encoders > 0 and **zero** host downloads. A host-computed twin can
  keep every value right; it cannot launch a compute encoder, and it has to
  download its operand.
* *refused by name* -- `aten.index.Tensor` (eager attention's
  `padding_mask[batch_idx, kv_idx]`) is still a host readback and is pinned
  as one; it is the wall left for eager-mode generate.

Nullifications this file is meant to catch:

* `first_extremum_index` using candle's `argmax_keepdim` on the values
      -> the `[-inf, lowest]` row of test_rewritten_ops_agree_on_mps
* `nan_along_dim` dropped -> the NaN rows of the same test
* `bitwise_on_device` losing its sign plane -> the negative int64 rows
* `new_buffer_with_data` losing its zero-size arm (candle fork)
      -> test_empty_tensors_are_made_on_mps_without_an_upload
* `linear_split`/`dispatch_*` losing their zero-grid guard (candle fork)
      -> test_ops_on_empty_mps_tensors_agree_with_upstream
* any of the ten ops going back on `MPS_HOST_READBACK_OPS`
      -> test_the_ten_generate_ops_are_not_refused_on_mps
"""

import json
import os
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

from test_shim import _C
import _skip


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
#: The vendored tree the shim side imports `torch` from, as test_qwen3.py does.
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")


def _shim_tree_or_skip(what):
    if not os.path.isfile(os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")):
        _skip.skip("   (skipped %s: no vendored shim at %s -- run vendor/install_shim.sh)"
                   % (what, _VENDOR_DIR))
        return False
    return True


def _run(script, payload, upstream):
    """Run `script` in a child. upstream=True empties PYTHONPATH so the child
    can only import site-packages torch; otherwise the child imports the
    vendored shim tree. Either way the script asserts which torch it got."""
    env = dict(os.environ)
    env.pop("TORCH_C_ARTEFACT", None)
    if upstream:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    else:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run([sys.executable, "-c", script, json.dumps(payload)],
                          capture_output=True, text=True, timeout=1800, env=env)
    lines = proc.stdout.strip().splitlines()
    if not lines:
        raise AssertionError("child produced no JSON (rc=%d):\n%s"
                             % (proc.returncode, proc.stderr[-3000:]))
    return json.loads(lines[-1])


def _mps_or_skip(what):
    if not _shim_tree_or_skip(what):
        return False
    try:
        _C._aten_dispatch("aten.ones.default", [1], device=_C.device("mps"))
    except (NotImplementedError, RuntimeError) as e:
        _skip.skip("   (skipped %s: no mps device on this machine -- %s)"
                   % (what, str(e).splitlines()[0]))
        return False
    return True


# --------------------------------------------------------------------------
# greedy generate, end to end
# --------------------------------------------------------------------------

#: A vocabulary wide enough that one host-side argmax would download more than
#: the whole run is allowed to: 4096 float32 logits are 16 KiB per step.
_GEN_CFG = dict(
    vocab_size=4096, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    max_position_embeddings=64, tie_word_embeddings=True, rope_theta=1e6,
    bos_token_id=1, eos_token_id=[2, 3], pad_token_id=1,
)
_GEN_IDS = [5, 7, 9, 11, 13, 17, 19, 23]
_NEW_TOKENS = 20

#: The same weights on both sides without a file or an RNG: every parameter is
#: a deterministic function of its position, built on the CPU.
_GEN_SCRIPT = r"""
import json, sys, traceback
A = json.loads(sys.argv[1])
import torch
IS_SHIM = hasattr(torch._C, "_aten_implemented")
assert IS_SHIM == A["shim"], ("wrong torch in this child", IS_SHIM)
out = {}
try:
    from transformers import AutoModelForCausalLM, Qwen3Config
    m = AutoModelForCausalLM.from_config(Qwen3Config(**A["cfg"]),
                                         attn_implementation=A["attn"]).eval()
    with torch.no_grad():
        for i, (name, p) in enumerate(sorted(m.named_parameters())):
            n = p.numel()
            v = torch.sin(torch.arange(n, dtype=torch.float32) * 0.37 + i) * 0.3
            p.copy_(v.reshape(p.shape))
    ids = torch.tensor([A["ids"]])
    enc = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    if A["device"] != "cpu":
        m = m.to(A["device"])
        enc = {k: v.to(A["device"]) for k, v in enc.items()}
    before = torch._C._metal_counters() if IS_SHIM and A["device"] == "mps" else None
    with torch.no_grad():
        seq = m.generate(**enc, max_new_tokens=A["new"], do_sample=False)
    after = torch._C._metal_counters() if before is not None else None
    out["tokens"] = seq.cpu().tolist()
    if before is not None:
        out["counters"] = {k: after[k] - before[k] for k in before if k != "built"}
except BaseException as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:1500])
    out["traceback"] = traceback.format_exc()[-3000:]
print(json.dumps(out))
"""


def _generate(shim, device, attn):
    return _run(_GEN_SCRIPT, dict(shim=shim, device=device, attn=attn, cfg=_GEN_CFG,
                                  ids=_GEN_IDS, new=_NEW_TOKENS), upstream=not shim)


def test_greedy_generate_on_mps_agrees_with_upstream_token_for_token():
    """The user's line, `model.to("mps").generate(...)`, with the default
    `DynamicCache` and sdpa attention -- issue #29's done-when, and what
    replaces `test_qwen3.py`'s two `_pin_wall` tests on the tiny model.

    The download bound is derived, not tuned: `generate` reads back one bool
    per step (`this_peer_finished`) and the final sequence, so the whole run
    may move at most 64 bytes a step plus 1 KiB -- 2.3 KiB. A single
    host-side argmax over the 4096-wide logits is 16 KiB, so any one of the
    rewritten ops falling back to the host breaks the bound by itself.
    """
    if not _mps_or_skip("greedy generate on mps"):
        return
    ref = _generate(False, "cpu", "sdpa")
    assert "error" not in ref, "upstream oracle failed: %s" % ref.get("traceback")
    got = _generate(True, "mps", "sdpa")
    assert "error" not in got, (
        "greedy generate on mps stopped -- issue #29 is not closed:\n%s"
        % got.get("traceback"))
    assert got["tokens"] == ref["tokens"], (got["tokens"], ref["tokens"])
    c = got["counters"]
    assert c["compute_encoders"] >= _NEW_TOKENS, c
    bound = _NEW_TOKENS * 64 + 1024
    assert c["host_download_bytes"] <= bound, (
        "generate on mps downloaded %d bytes (bound %d): something computed on "
        "the host" % (c["host_download_bytes"], bound), c)


def test_greedy_generate_on_mps_with_eager_attention_stops_at_index_by_name():
    """Eager attention is the one configuration still walled, and the wall is
    named: `masking_utils.padding_mask_function` indexes a bool mask with two
    index tensors, `aten.index.Tensor`, whose kernel is still a host readback.
    When that op moves onto the device this test goes red; replace it with
    the sdpa test above run on `attn="eager"`."""
    if not _mps_or_skip("eager generate on mps"):
        return
    got = _generate(True, "mps", "eager")
    assert "error" in got, (
        "eager generate on mps now completes -- aten.index.Tensor moved; "
        "turn this into an agreement test")
    assert "aten.index.Tensor: not implemented for the mps device" in got["error"], got["error"]


# --------------------------------------------------------------------------
# the rewritten ops, one by one
# --------------------------------------------------------------------------

_LOWEST = {"float32": -3.4028234663852886e38, "float16": -65504.0,
           "bfloat16": -3.3895313892515355e38}
_INF = float("inf")
_NAN = float("nan")


def _op_cases():
    """`[(label, op, [(values, shape, dtype)], kwargs)]`, JSON-safe.

    Each row is aimed at one step of the device arithmetic; see the module
    docstring's nullification list and `first_extremum_index`'s header.
    """
    rows = []
    for dt in ("float32", "float16", "bfloat16"):
        lo = _LOWEST[dt]
        for label, vals in (
            ("tie", [3.0, 7.0, 7.0, 1.0]),
            ("all -inf", [-_INF] * 4),
            ("-inf before lowest", [-_INF, lo, -_INF, lo]),
            ("nan middle", [1.0, _NAN, 9.0, _NAN]),
            ("signed zeros", [0.0, -0.0, 0.0, -0.0]),
        ):
            for op, args in (("aten.argmax.default", {"dim": None}),
                             ("aten.argmax.default", {"dim": 1}),
                             ("aten.max.default", {}),
                             ("aten.min.default", {}),
                             ("aten.max.dim", {"dim": 1}),
                             ("aten.min.dim", {"dim": 0})):
                # The sign of a zero *value* is left out: upstream's CPU
                # `min()` of [0., -0., 0., -0.] answers -0.0 (measured,
                # 2026-10-03) where the first-index rule, old build and new,
                # answers 0.0. That is a vectorisation detail of upstream's
                # reduction, not a defined semantic; the index rows keep it.
                if label == "signed zeros" and op != "aten.argmax.default":
                    continue
                rows.append(("%s %s %s %s" % (op, dt, label, args), op,
                             [(vals, [2, 2], dt)], args))
        # Wider than one threadgroup (1024), with the maximum tied at both
        # ends and NaN-free: the fold's cross-threadgroup tie-break.
        wide = [0.0] * 5000
        wide[37] = wide[4990] = 5.0
        rows.append(("argmax %s 5000 wide, tie across threadgroups" % dt,
                     "aten.argmax.default", [(wide, [5000], dt)], {"dim": 0}))
        # A strided reduction (dim 0 of a 2-D tensor is not the last dim).
        rows.append(("argmax %s strided dim 0" % dt, "aten.argmax.default",
                     [([1.0, 9.0, 9.0, 1.0, 9.0, 0.0], [3, 2], dt)], {"dim": 0}))
        rows.append(("isin %s nan/-0.0/inf" % dt, "aten.isin.Tensor_Tensor",
                     [([_NAN, 0.0, -0.0, _INF, 1.5], [5], dt),
                      ([_NAN, -0.0, _INF], [3], dt)], {}))
    for dt in ("int64", "uint8"):
        vals = [255, 1, 128, 0] if dt == "uint8" else [-(2 ** 62), 5, -1, 5]
        for op, args in (("aten.argmax.default", {"dim": 0}), ("aten.max.default", {}),
                         ("aten.max.dim", {"dim": 0})):
            rows.append(("%s %s" % (op, dt), op, [(vals, [4], dt)], args))
    rows.append(("isin int64 eos", "aten.isin.Tensor_Tensor",
                 [([5, 7, 9, 2, 3], [1, 5], "int64"), ([2, 3], [2], "int64")], {}))
    rows.append(("isin int64 invert", "aten.isin.Tensor_Tensor",
                 [([5, 7, 9, 2, 3], [1, 5], "int64"), ([2, 3], [2], "int64")],
                 {"invert": True}))
    signed = [-(2 ** 62), -1, -2, 0, 5, 2 ** 40, -(2 ** 53) - 8]
    for op in ("aten.bitwise_and.Tensor", "aten.bitwise_or.Tensor", "aten.bitwise_xor.Tensor"):
        rows.append(("%s int64 negatives" % op, op,
                     [(signed, [7], "int64"), (list(reversed(signed)), [7], "int64")], {}))
        rows.append(("%s int64 & bool" % op, op,
                     [([0, 1, 1, 5, -3, 2 ** 40], [6], "int64"),
                      ([True, False, True, True, True, False], [6], "bool")], {}))
        rows.append(("%s bool truth table" % op, op,
                     [([True, False, True, False], [4], "bool"),
                      ([True, True, False, False], [4], "bool")], {}))
        rows.append(("%s uint8" % op, op,
                     [([255, 1, 128, 77], [4], "uint8"), ([15, 3, 129, 200], [4], "uint8")], {}))
    for dt, vals in (("int64", [-(2 ** 62), -1, 0, 2 ** 40, 7]), ("uint8", [0, 1, 128, 254, 255]),
                     ("bool", [True, False, True, False, False])):
        rows.append(("bitwise_not %s" % dt, "aten.bitwise_not.default", [(vals, [5], dt)], {}))
    return rows


_OP_SCRIPT = r"""
import json, math, sys, traceback
A = json.loads(sys.argv[1])
import torch
IS_SHIM = hasattr(torch._C, "_aten_implemented")
assert IS_SHIM == A["shim"], ("wrong torch in this child", IS_SHIM)
OPS = {
    "aten.argmax.default": lambda xs, kw: torch.argmax(xs[0], **kw),
    "aten.max.default": lambda xs, kw: torch.max(xs[0]),
    "aten.min.default": lambda xs, kw: torch.min(xs[0]),
    "aten.max.dim": lambda xs, kw: torch.max(xs[0], kw["dim"]),
    "aten.min.dim": lambda xs, kw: torch.min(xs[0], kw["dim"]),
    "aten.isin.Tensor_Tensor": lambda xs, kw: torch.isin(xs[0], xs[1], **kw),
    "aten.bitwise_and.Tensor": lambda xs, kw: xs[0] & xs[1],
    "aten.bitwise_or.Tensor": lambda xs, kw: xs[0] | xs[1],
    "aten.bitwise_xor.Tensor": lambda xs, kw: xs[0] ^ xs[1],
    "aten.bitwise_not.default": lambda xs, kw: ~xs[0],
}
def flat(t):
    t = t.cpu()
    if t.dtype in (torch.float16, torch.bfloat16, torch.float32):
        return [repr(float(v)) for v in t.double().flatten().tolist()], str(t.dtype)
    return [int(v) for v in t.flatten().tolist()], str(t.dtype)
out = []
for label, op, operands, kw in A["rows"]:
    rec = {"label": label}
    try:
        xs = [torch.tensor(v, dtype=getattr(torch, d)).reshape(s) for v, s, d in operands]
        if A["device"] != "cpu":
            xs = [x.to(A["device"]) for x in xs]
        before = torch._C._metal_counters() if IS_SHIM and A["device"] == "mps" else None
        r = OPS[op](xs, kw)
        after = torch._C._metal_counters() if before is not None else None
        rs = list(r) if isinstance(r, tuple) else [r]
        rec["shape"] = [list(t.shape) for t in rs]
        rec["device"] = [str(t.device) for t in rs]
        rec["values"] = [flat(t) for t in rs]
        if before is not None:
            rec["counters"] = {k: after[k] - before[k] for k in before if k != "built"}
    except BaseException as exc:
        rec["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:400])
    out.append(rec)
print(json.dumps(out))
"""


def test_rewritten_ops_agree_on_mps_and_metal_computed_them():
    """Each of the ten ops, on inputs aimed at its device arithmetic, against
    upstream on the CPU; and on mps, the dispatch alone launched compute
    encoders and downloaded nothing."""
    if not _mps_or_skip("issue #29 ops on mps"):
        return
    rows = _op_cases()
    ref = _run(_OP_SCRIPT, dict(shim=False, device="cpu", rows=rows), upstream=True)
    got = _run(_OP_SCRIPT, dict(shim=True, device="mps", rows=rows), upstream=False)
    problems = []
    for r, g in zip(ref, got):
        if "error" in r:
            problems.append("%s: upstream raised %s" % (r["label"], r["error"]))
            continue
        if "error" in g:
            problems.append("%s: mps raised %s" % (g["label"], g["error"]))
            continue
        if g["shape"] != r["shape"] or g["values"] != r["values"]:
            problems.append("%s: mps %s %s, upstream %s %s"
                            % (g["label"], g["shape"], g["values"], r["shape"], r["values"]))
        if not all(d.startswith("mps") for d in g["device"]):
            problems.append("%s: result on %s" % (g["label"], g["device"]))
        c = g["counters"]
        if c["compute_encoders"] < 1 or c["host_downloads"] != 0:
            problems.append("%s: counters %s -- not computed on Metal" % (g["label"], c))
    assert not problems, "\n".join(problems)


def test_rewritten_ops_agree_on_cpu():
    """The CPU runs the same candle-op formulas; the golden cases cover the
    dtypes, this covers the same rows the mps test does, in one place."""
    if not _shim_tree_or_skip("issue #29 ops on cpu"):
        return
    rows = _op_cases()
    ref = _run(_OP_SCRIPT, dict(shim=False, device="cpu", rows=rows), upstream=True)
    got = _run(_OP_SCRIPT, dict(shim=True, device="cpu", rows=rows), upstream=False)
    bad = ["%s: shim %s, upstream %s" % (g["label"], g.get("values", g.get("error")),
                                        r.get("values", r.get("error")))
           for r, g in zip(ref, got) if r.get("values") != g.get("values")]
    assert not bad, "\n".join(bad)


def test_the_ten_generate_ops_are_not_refused_on_mps():
    """Off the list by being rewritten -- the derivation scan in test_shim.py
    is what proves the kernels hold no readback; this checks the artefact's
    table, and that `index.Tensor` (the eager wall) is still on it."""
    refused = set(_C._shim_mps_host_readback_ops())
    for op in ("aten.argmax.default", "aten.max.default", "aten.min.default",
               "aten.max.dim", "aten.min.dim", "aten.isin.Tensor_Tensor",
               "aten.bitwise_and.Tensor", "aten.bitwise_or.Tensor",
               "aten.bitwise_xor.Tensor", "aten.bitwise_not.default"):
        assert op not in refused, op
    assert "aten.index.Tensor" in refused
    assert "aten.multinomial.default" in refused, (
        "do_sample=True's sampler left the list -- measure sampling on mps")


# --------------------------------------------------------------------------
# zero-element tensors on Metal
# --------------------------------------------------------------------------

_EMPTY_OPS = {
    "tensor([])": "torch.tensor([], device=D)",
    "tensor([], bool)": "torch.tensor([], dtype=torch.bool, device=D)",
    "tensor([], int64)": "torch.tensor([], dtype=torch.int64, device=D)",
    "tensor([], bfloat16)": "torch.tensor([], dtype=torch.bfloat16, device=D)",
    "cpu empty .to": "torch.empty(0, 4).to(D)",
    "arange(0)": "torch.arange(0, device=D)",
    "half": "E.half()",
    "long": "E.long()",
    "where": "torch.where(B, E, E)",
    "clamp": "E.clamp(0, 1)",
    "sum(1)": "E.sum(1)",
    "sum(0)": "E.sum(0)",
    "mean(1)": "E.mean(1)",
    "index_select": "K.index_select(0, torch.zeros(0, dtype=torch.int64).to(D))",
    "embedding": "torch.nn.functional.embedding(torch.zeros(0, dtype=torch.int64).to(D), K)",
    "logical_not": "B.logical_not()",
    "ones_like": "torch.ones_like(E)",
    "masked_fill": "E.masked_fill(B, 1.0)",
    "softmax": "E.softmax(-1)",
    "cat(empty1d, k)": "torch.cat([torch.tensor([], device=D), K], 0)",
    "argmax(dim=1)": "torch.zeros(0, 4).to(D).argmax(1)",
}

_EMPTY_SCRIPT = r"""
import json, sys
A = json.loads(sys.argv[1])
import torch
IS_SHIM = hasattr(torch._C, "_aten_implemented")
assert IS_SHIM == A["shim"], ("wrong torch in this child", IS_SHIM)
D = A["device"]
out = {}
try:
    E = torch.zeros(0, 4).to(D) if D == "cpu" else torch.zeros(0, 4, device=D)
    B = torch.zeros(0, 4, dtype=torch.bool, device=D)
    K = torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 3).to(D)
    before = torch._C._metal_counters() if IS_SHIM and D == "mps" else None
    r = eval(A["expr"])
    after = torch._C._metal_counters() if before is not None else None
    out = {"shape": list(r.shape), "dtype": str(r.dtype), "device": str(r.device),
           "values": r.cpu().double().flatten().tolist()}
    if before is not None:
        out["counters"] = {k: after[k] - before[k] for k in before if k != "built"}
except BaseException as exc:
    out = {"error": "%s: %s" % (type(exc).__name__, str(exc)[:300])}
print(json.dumps(out))
"""


def test_ops_on_empty_mps_tensors_agree_with_upstream():
    """Zero-element tensors on Metal: made, cast, reduced, indexed and
    combined -- shape, dtype and values as upstream gives them, device mps.

    One subprocess per expression, because before the candle fork's
    zero-grid guard a zero-length launch panicked in `linear_split`
    (`0.div_ceil(0)`) and poisoned the encoder for the rest of the process
    (measured: every later Metal op panicked with `PoisonError`)."""
    if not _mps_or_skip("zero-element tensors on mps"):
        return
    problems = []
    for name, expr in sorted(_EMPTY_OPS.items()):
        ref = _run(_EMPTY_SCRIPT, dict(shim=False, device="cpu", expr=expr), upstream=True)
        got = _run(_EMPTY_SCRIPT, dict(shim=True, device="mps", expr=expr), upstream=False)
        if "error" in ref:
            if "error" not in got:
                problems.append("%s: upstream refuses (%s), mps answered" % (name, ref["error"]))
            continue
        if "error" in got:
            problems.append("%s: %s" % (name, got["error"]))
            continue
        for k in ("shape", "dtype", "values"):
            if got[k] != ref[k]:
                problems.append("%s: %s %r vs upstream %r" % (name, k, got[k], ref[k]))
        if not got["device"].startswith("mps"):
            problems.append("%s: result on %s" % (name, got["device"]))
    assert not problems, "\n".join(problems)


def test_empty_tensors_are_made_on_mps_without_an_upload():
    """`torch.tensor([], device="mps")` is `DynamicCache`'s first line on mps
    (cache_utils.py:123). It used to fail with "Failed to create metal
    resource: Buffer" -- `newBufferWithBytes` with length 0 returns nil. The
    fork now takes a pooled buffer instead (1 byte, `buf_size(0)`), and since
    nothing crosses, no upload is counted."""
    if not _mps_or_skip("empty tensor creation on mps"):
        return
    got = _run(_EMPTY_SCRIPT, dict(shim=True, device="mps",
                                   expr="torch.tensor([], dtype=torch.float32, device=D)"),
               upstream=False)
    assert "error" not in got, got
    assert got["shape"] == [0] and got["dtype"] == "torch.float32", got
    assert got["device"].startswith("mps"), got
    assert got["counters"]["host_upload_bytes"] == 0, got["counters"]


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_mpsgen")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
