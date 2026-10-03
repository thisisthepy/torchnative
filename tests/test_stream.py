"""Streaming `generate` against upstream: `TextIteratorStreamer` yields the same
tokens upstream does, plain and through `TorchnativeConfig("q8_0")` (issue #11).

Gemstone M1 streams a quantised Qwen3 / SmolLM2. Before this file **no suite
used a streamer** (`grep -l Streamer tests/` was empty), so the path
`generate(streamer=...)` -> `streamer.put(ids)` -> `tokenizer.decode` ->
`queue.Queue` -> a consuming thread had never been run on the shim.

What is compared, on each side in its own subprocess (AGENTS.md §16, third
grade -- *agrees*, not *reaches*):

* every `put()` the streamer receives, as token ids -- the prompt first, then
  one id per step. This is the token-for-token claim; the text chunks the
  iterator yields are compared too, and the streamed ids are cross-checked
  against `generate`'s own return, so a streamer that drops or repeats a token
  is red against the shim's own output as well as upstream's.
* plain: shim float32 against upstream float32, exact.
* q8_0: the shim loads through the registered `HfQuantizer`
  (`from_pretrained(..., quantization_config=TorchnativeConfig("q8_0"))`),
  dequantises each `QuantizedLinear` it holds and dumps those weights; the
  upstream side loads the SAME checkpoint and overwrites each of those weights
  with the dequantised values, so the two sides compute on identical numbers
  and the only thing under test is the quantised matmul and the streaming path.
  The tokens are compared exactly; the prefill logits gap is measured and
  graded against a tolerance derived as docs/numerics/AGREE.md §2 says --
  upstream's own float32-vs-float64 distance on those dequantised weights, 4x
  its worst, floor 8 eps. The measured gap is printed by `_report()`.
  **candle's `QMatMul` quantises the activation to q8_0 too** (32-blocks, f16
  scale amax/127) and takes an integer dot, so "upstream on the dequantised
  weights" is not enough: the upstream oracle also fake-quantises the
  activation entering each quantised layer the same way. The gap to the plain
  dense-activation oracle (about 2% of the logit scale on the tiny models) is
  reported as information, not graded.

Models: a tiny Qwen3 and a tiny Llama (SmolLM2's architecture), written by
upstream into a temporary directory under `.scratch/` with a word-level
tokenizer so a real `TextIteratorStreamer` has something to decode; and the real
`Qwen/Qwen3-0.6B` and `HuggingFaceTB/SmolLM2-135M` from the repository's own
HF cache, skipped by name when absent. HF_HUB_OFFLINE=1 throughout.

mps: greedy `generate` is blocked at issue #29's walls (isin readback, the
zero-byte Metal buffer, argmax). They are PINNED by name in the pattern of
test_qwen3.py -- green while the wall stands, red if the line stops anywhere
else or gets past it, with `_agree_on_mps_stream` as the one-line replacement.

NOT MEASURED here: streaming under sampling (`do_sample=True`; the streamer is
the same object but the token source is a random draw, which two processes
cannot be compared on), and divergence that appears only in a larger model than
0.6B / 135M, or past the 20 generated tokens.
"""

import atexit
import functools
import glob
import json
import os
import shutil
import subprocess
import sys
import tempfile

import _skip

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_VENDOR_DIR = os.environ.get("TORCH_C_STREAM_VENDOR") or os.path.join(
    _REPO_ROOT, "python")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

_EPS32 = 1.1920929e-07
_ORACLE_FACTOR = 4.0
_NEW_TOKENS = 20
_TINY_NEW_TOKENS = 12
_TINY_IDS = [5, 7, 9, 11, 13, 17, 19, 23]
_PROMPT = "Give me a short introduction to large language models."

_REAL = {
    "qwen3": ("Qwen/Qwen3-0.6B", 28),
    "smollm2": ("HuggingFaceTB/SmolLM2-135M", 30),
}

# q8_0 stores 32-element blocks, so every Linear's in_features must divide by
# 32: intermediate_size 64, not 48.
_TINY_CFG = {
    "qwen3": dict(
        model_type="qwen3", vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=64, tie_word_embeddings=True,
        rope_theta=1e6, bos_token_id=1, eos_token_id=[2, 3], pad_token_id=1),
    "smollm2": dict(
        model_type="llama", vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=True, rope_theta=1e5,
        bos_token_id=1, eos_token_id=2, pad_token_id=1),
}


def _main_checkout():
    marker = os.sep + ".worktrees" + os.sep
    if marker in _REPO_ROOT + os.sep:
        return _REPO_ROOT.split(marker)[0]
    return _REPO_ROOT


def _snapshot(model_id):
    folder = "models--" + model_id.replace("/", "--")
    roots = [os.environ.get("HF_HOME"),
             os.path.join(_REPO_ROOT, ".caches", "hf-home"),
             os.path.join(_main_checkout(), ".caches", "hf-home")]
    for root in roots:
        if not root:
            continue
        for snap in sorted(glob.glob(os.path.join(root, "hub", folder, "snapshots", "*"))):
            if os.path.isfile(os.path.join(snap, "model.safetensors")) and \
                    os.path.isfile(os.path.join(snap, "tokenizer.json")):
                return snap
    return None


# --------------------------------------------------------------------------
# One script, run on either side
# --------------------------------------------------------------------------

_SIDE_SCRIPT = r"""
import json, os, sys, threading, traceback
A = json.loads(sys.argv[1])
import torch
IS_SHIM = hasattr(torch._C, "_aten_implemented")
assert IS_SHIM == A["shim"], ("wrong torch on this side", IS_SHIM, A["shim"])
out = {"stage": "start", "shim": IS_SHIM, "completed": [], "errors": {}, "tag": A["tag"]}


def raw(t):
    t = t.detach().contiguous()
    if IS_SHIM:
        return torch._C._shim_tensor_bytes(t)
    return t.numpy().tobytes()


def dump(name, t):
    path = os.path.join(A["out"], A["tag"] + "." + name)
    wide = t.detach().double() if A["dtype"] == "float64" else t.detach().float()
    with open(path, "wb") as fh:
        fh.write(raw(wide.cpu()))
    out[name] = {"path": path, "shape": [int(d) for d in t.shape],
                 "dtype": str(t.dtype), "device": str(t.device),
                 "itemsize": 8 if A["dtype"] == "float64" else 4}


try:
    if A.get("write_tiny"):
        out["stage"] = "write_tiny"
        from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
        from tokenizers import Tokenizer, models
        cfg = dict(A["write_tiny"])
        mt = cfg.pop("model_type")
        torch.manual_seed(0)
        m = AutoModelForCausalLM.from_config(AutoConfig.for_model(mt, **cfg)).eval()
        with torch.no_grad():
            for p in m.parameters():
                p.normal_(0.0, 0.3)
        m.save_pretrained(A["path"])
        n = cfg["vocab_size"]
        tk = Tokenizer(models.WordLevel({"w%d" % i: i for i in range(n)}, unk_token="w0"))
        PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="w0", bos_token="w1",
                                eos_token="w2", pad_token="w1").save_pretrained(A["path"])
        del m

    if IS_SHIM and A["device"] == "mps":
        if not torch._C._metal_counters()["built"]:
            out["no_mps"] = "this build has no Metal backend"
            print(json.dumps(out)); raise SystemExit(0)

    out["stage"] = "load"
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
    kw = {"dtype": getattr(torch, A["dtype"])}
    if A["quant"]:
        from torchnative.quant import TorchnativeConfig
        kw["quantization_config"] = TorchnativeConfig(A["quant"])
    model = AutoModelForCausalLM.from_pretrained(A["path"], **kw).eval()
    if A["quant"]:
        from torchnative.quant import QuantizedLinear
        qlin = {n: m for n, m in model.named_modules() if isinstance(m, QuantizedLinear)}
        out["n_quantized"] = len(qlin)
        out["n_linear"] = sum(1 for m in model.modules() if isinstance(m, torch.nn.Linear))
        out["quantized_names"] = sorted(qlin)
        if A.get("deq_out"):
            os.makedirs(A["deq_out"], exist_ok=True)
            for n, m in qlin.items():
                w = torch._C._dequantize(m.qweight)
                with open(os.path.join(A["deq_out"], n + ".bin"), "wb") as fh:
                    fh.write(raw(w.float().cpu()))
        if A.get("mutant") == "qscale":
            # The dequant scale, wrong by 1%: every quantised layer's output.
            orig = QuantizedLinear.forward
            QuantizedLinear.forward = lambda self, x: orig(self, x) * 1.01
    if A.get("deq_from"):
        import numpy as np
        mods = dict(model.named_modules())
        for n in A["deq_names"]:
            w = mods[n].weight
            arr = np.fromfile(os.path.join(A["deq_from"], n + ".bin"), dtype="<f4")
            t = torch.from_numpy(arr.copy()).reshape(w.shape)
            with torch.no_grad():
                w.copy_(t.to(w.dtype) * A.get("deq_scale", 1.0))
        out["deq_applied"] = len(A["deq_names"])
        if A.get("act_q8"):
            # candle's QMatMul quantises the ACTIVATION to q8_0 as well (blocks
            # of 32, scale amax/127 stored as f16, round half away from zero)
            # and takes an integer dot. The oracle for the shim's quantised
            # linear is therefore the dequantised weight applied to the
            # fake-quantised activation, not to the dense one.
            def act_q8(mod, args):
                x = args[0]
                xb = x.reshape(-1, 32)
                d = xb.abs().amax(-1, keepdim=True) / 127.0
                idv = torch.where(d > 0, 1.0 / torch.where(d > 0, d, torch.ones_like(d)),
                                  torch.zeros_like(d))
                v = xb * idv
                q = torch.sign(v) * torch.floor(v.abs() + 0.5)
                dh = d.float().half().float().to(x.dtype)
                return ((q * dh).reshape(x.shape),) + tuple(args[1:])
            for n in A["deq_names"]:
                mods[n].register_forward_pre_hook(act_q8)

    out["stage"] = "inputs"
    tok = AutoTokenizer.from_pretrained(A["path"])
    if A.get("prompt") is not None:
        enc = dict(tok(A["prompt"], return_tensors="pt"))
    else:
        ids = torch.tensor([A["ids"]])
        enc = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
    out["input_ids"] = enc["input_ids"].tolist()

    out["stage"] = "to(device)"
    if A["device"] != "cpu":
        model = model.to(A["device"])
        enc = {k: v.to(A["device"]) for k, v in enc.items()}

    def attempt(name, fn):
        out["stage"] = name
        try:
            fn()
        except BaseException as exc:
            out["errors"][name] = {
                "error": "%s: %s" % (type(exc).__name__, str(exc)[:1500]),
                "traceback": traceback.format_exc()[-4000:]}
        else:
            out["completed"].append(name)

    def logits():
        with torch.no_grad():
            dump("logits", model(**enc).logits)

    def stream():
        puts = []
        mutant = A.get("mutant")

        class Rec(TextIteratorStreamer):
            def put(self, value):
                puts.append([int(x) for x in value.reshape(-1).cpu().tolist()])
                if mutant == "drop_token" and len(puts) == 5:
                    return  # the streamer path loses one token
                super().put(value)

        streamer = Rec(tok, skip_prompt=True, timeout=600)
        box = {}

        def work():
            try:
                with torch.no_grad():
                    box["seq"] = model.generate(
                        **enc, max_new_tokens=A["generate"], do_sample=False,
                        streamer=streamer)
            except BaseException as exc:
                box["exc"] = exc
                box["tb"] = traceback.format_exc()
                streamer.end()

        th = threading.Thread(target=work)
        th.start()
        chunks = [c for c in streamer]
        th.join()
        if "exc" in box:
            raise RuntimeError("%s: %s\n%s" % (
                type(box["exc"]).__name__, str(box["exc"])[:1500], box["tb"][-3000:]))
        out["generated"] = box["seq"].cpu().tolist()
        out["puts"] = puts
        out["chunks"] = chunks
        out["text"] = "".join(chunks)
        out["decoded"] = tok.decode(out["generated"][0][len(out["input_ids"][0]):],
                                    skip_special_tokens=False)

    if A.get("logits"):
        attempt("logits", logits)
    attempt("stream", stream)
    if out["errors"]:
        raise RuntimeError("stage(s) stopped: " + ", ".join(sorted(out["errors"])))
    out["stage"] = "done"
except SystemExit:
    raise
except BaseException as exc:
    out["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:1500])
    out["traceback"] = traceback.format_exc()[-4000:]
print(json.dumps(out))
"""


def _run_side(args, timeout=3600):
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    env["HF_HUB_DISABLE_TELEMETRY"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    if args.get("hf_home"):
        env["HF_HOME"] = args["hf_home"]
    if args["shim"]:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", _SIDE_SCRIPT, json.dumps(args)],
        capture_output=True, text=True, env=env, timeout=timeout)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    if proc.returncode != 0 or not lines:
        raise RuntimeError(
            f"{args['tag']}: subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout[-3000:]}\n--- stderr ---\n{proc.stderr[-3000:]}")
    return json.loads(lines[-1])


def _read(rec, name):
    import numpy as np

    meta = rec[name]
    dt = "<f8" if meta["itemsize"] == 8 else "<f4"
    arr = np.fromfile(meta["path"], dtype=dt).astype("<f8")
    assert arr.size == int(np.prod(meta["shape"])), (name, arr.size, meta["shape"])
    return arr


def _reached(rec, what="done"):
    if rec.get("no_mps"):
        raise _skip.Skip(rec["no_mps"])
    ok = rec["stage"] == "done" if what == "done" else what in rec.get("completed", [])
    err = rec.get("errors", {}).get(what) or {
        "error": rec.get("error"), "traceback": rec.get("traceback", "")}
    assert ok, (
        f"{rec.get('tag', '')} did not get through {what!r}: stopped at stage "
        f"{rec.get('stage')!r}: {err['error']}\n{err['traceback'][-1200:]}")


# --------------------------------------------------------------------------
# Cases
# --------------------------------------------------------------------------

def _require_shim():
    if not os.path.isfile(_VENDOR_SHIM):
        raise _skip.Skip(f"vendored tree not built: no {_VENDOR_SHIM} "
                         "(scripts/vendor/vendor_torch.sh + scripts/vendor/install_shim.sh)")


@functools.lru_cache(maxsize=None)
def _workdir():
    # Inside the repository (AGENTS.md §15.1): `.scratch/` is gitignored.
    base = os.path.join(_REPO_ROOT, ".scratch")
    os.makedirs(base, exist_ok=True)
    root = tempfile.mkdtemp(prefix="stream-", dir=base)
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root


@functools.lru_cache(maxsize=None)
def _case(name):
    """`name` is `tiny-qwen3`, `tiny-smollm2`, `qwen3` or `smollm2`."""
    _require_shim()
    if name.startswith("tiny-"):
        fam = name[5:]
        path = os.path.join(_workdir(), name + "-ckpt")
        rec = _run_side({"shim": False, "tag": name + "-write", "out": _workdir(),
                         "path": path, "write_tiny": _TINY_CFG[fam], "dtype": "float32",
                         "device": "cpu", "ids": _TINY_IDS, "quant": None,
                         "generate": 1, "hf_home": os.path.join(_workdir(), "hf-home")})
        _reached(rec)
        return {"path": path, "ids": _TINY_IDS, "prompt": None, "new": _TINY_NEW_TOKENS,
                "hf_home": os.path.join(_workdir(), "hf-home"), "layers": 2}
    model_id, layers = _REAL[name]
    snap = _snapshot(model_id)
    if snap is None:
        raise _skip.Skip(
            f"{model_id} is not cached under {_REPO_ROOT}/.caches/hf-home, "
            f"{_main_checkout()}/.caches/hf-home or $HF_HOME; fetch it with "
            f"HF_HOME=<repo>/.caches/hf-home -- the gate never downloads")
    return {"path": snap, "ids": None, "prompt": _PROMPT, "new": _NEW_TOKENS,
            "hf_home": os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(snap)))),
            "layers": layers}


def _args(case, shim, dtype, device, quant, tag, **extra):
    c = _case(case)
    a = {"shim": shim, "dtype": dtype, "device": device, "quant": quant,
         "tag": f"{case}-{tag}", "out": _workdir(), "path": c["path"], "ids": c["ids"],
         "prompt": c["prompt"], "hf_home": c["hf_home"], "generate": c["new"],
         "logits": device == "cpu"}
    a.update(extra)
    return a


@functools.lru_cache(maxsize=None)
def _plain(case, shim, device="cpu"):
    return _run_side(_args(case, shim, "float32", device, None,
                           f"{'shim' if shim else 'up'}-plain-{device}"))


@functools.lru_cache(maxsize=None)
def _q8_shim(case, device="cpu", mutant=None):
    deq = os.path.join(_workdir(), f"{case}-deq")
    return _run_side(_args(case, True, "float32", device, "q8_0",
                           f"shim-q8-{device}-{mutant}", deq_out=deq, mutant=mutant))


@functools.lru_cache(maxsize=None)
def _q8_up(case, dtype="float32", act_q8=True):
    """Upstream on the shim's dequantised weights. `act_q8` also fake-quantises
    the activations the way candle's QMatMul does; without it the oracle is the
    dense-activation one and the gap is the activation quantisation's."""
    shim = _q8_shim(case)
    _reached(shim, "stream")
    deq = os.path.join(_workdir(), f"{case}-deq")
    return _run_side(_args(case, False, dtype, "cpu", None, f"up-deq-{dtype}-{act_q8}",
                           deq_from=deq, deq_names=shim["quantized_names"],
                           act_q8=act_q8))


@functools.lru_cache(maxsize=None)
def _plain_mutant(case):
    return _run_side(_args(case, True, "float32", "cpu", None, "shim-plain-cpu-drop",
                           mutant="drop_token"))


# --------------------------------------------------------------------------
# The checks
# --------------------------------------------------------------------------

def _stream_problem(up, shim):
    """None if the two streams agree token for token, else why not."""
    if shim["input_ids"] != up["input_ids"]:
        return f"the sides fed different tokens: {shim['input_ids']} vs {up['input_ids']}"
    plen = len(up["input_ids"][0])
    up_new = up["generated"][0][plen:]
    if not up_new:
        return "upstream generated nothing"
    if shim["generated"] != up["generated"]:
        return (f"generate diverged\n  upstream {up_new}\n  shim     "
                f"{shim['generated'][0][plen:]}")
    # put(): the prompt first (skip_prompt only affects the text), then one id per step.
    for name, rec in (("upstream", up), ("shim", shim)):
        flat = [t for p in rec["puts"] for t in p]
        if flat != rec["generated"][0]:
            return (f"{name}: the streamer saw {flat}, generate returned "
                    f"{rec['generated'][0]}")
    if shim["puts"] != up["puts"]:
        return f"put() sequences differ: upstream {up['puts']} shim {shim['puts']}"
    if shim["chunks"] != up["chunks"]:
        return f"text chunks differ: upstream {up['chunks']} shim {shim['chunks']}"
    for name, rec in (("upstream", up), ("shim", shim)):
        if rec["text"] != rec["decoded"]:
            return f"{name}: streamed text {rec['text']!r} != decode {rec['decoded']!r}"
    return None


def _bound(up, up64):
    """docs/numerics/AGREE.md §2: (bound in output units, p90 tolerance,
    upstream's own worst relative distance from float64, scale)."""
    import numpy as np

    scale = float(np.max(np.abs(up64)))
    assert scale > 0, "degenerate output: every logit is zero"
    own = np.abs(up - up64) / scale
    p90 = float(np.sort(own)[int(0.9 * (own.size - 1))])
    tol = max(p90, 8 * _EPS32)
    worst = float(own.max())
    return max(tol, _ORACLE_FACTOR * worst) * scale, tol, worst, scale


_MEASURED = []


def _report(line):
    _MEASURED.append(line)
    print("MEASURED " + line)


def _q8_gap(case, shim_rec):
    """(distance of the shim's q8 prefill logits from upstream on the same
    dequantised weights, the derived bound, p90 tolerance, upstream's own
    worst, scale)."""
    import numpy as np

    up, up64 = _q8_up(case), _q8_up(case, "float64")
    _reached(up)
    _reached(up64)
    _reached(shim_rec, "logits")
    a, u, u64 = _read(shim_rec, "logits"), _read(up, "logits"), _read(up64, "logits")
    assert np.all(np.isfinite(u64)), "upstream's float64 oracle is not finite"
    bound, tol, own, scale = _bound(u, u64)
    return float(np.max(np.abs(a - u))), bound, tol, own, scale


def _check_plain(case):
    up, shim = _plain(case, False), _plain(case, True)
    _reached(up)
    _reached(shim)
    problem = _stream_problem(up, shim)
    assert problem is None, f"{case} plain cpu: {problem}"
    plen = len(up["input_ids"][0])
    _report(f"{case} plain cpu: {len(up['generated'][0]) - plen} tokens, "
            f"{len(up['chunks'])} chunks, identical")


def _check_q8(case):
    up, shim = _q8_up(case), _q8_shim(case)
    _reached(shim)
    _reached(up)
    assert shim["n_quantized"] >= 1, shim
    assert "lm_head" not in shim["quantized_names"], shim["quantized_names"]
    assert up["deq_applied"] == shim["n_quantized"], (up["deq_applied"], shim["n_quantized"])
    problem = _stream_problem(up, shim)
    assert problem is None, f"{case} q8_0 cpu: {problem}"
    dist, bound, tol, own, scale = _q8_gap(case, shim)
    import numpy as np

    dense_up = _q8_up(case, "float32", False)
    _reached(dense_up)
    dense_gap = float(np.max(np.abs(_read(shim, "logits") - _read(dense_up, "logits"))))
    dense_same = dense_up["generated"] == shim["generated"]
    _report(f"{case} q8_0 cpu: (information) against the DENSE-activation oracle the gap is "
            f"{dense_gap:.3e} (rel {dense_gap / scale:.3e}), tokens "
            f"{'identical' if dense_same else 'DIFFER'} -- candle's QMatMul also "
            "quantises the activation to q8_0")
    _report(f"{case} q8_0 cpu: tokens identical; {shim['n_quantized']} layers quantised; "
            f"logits gap {dist:.3e} (rel {dist / scale:.3e}) vs derived bound {bound:.3e} "
            f"(p90 {tol:.3e}, upstream f32-vs-f64 worst {own:.3e}, scale {scale:.3e})")
    assert dist <= bound, (
        f"{case} q8_0: shim logits differ from upstream-on-dequantised-weights by "
        f"{dist:.3e}; derived bound {bound:.3e}")
    return dist, bound


# --- the comparators can fail --------------------------------------------

def _check_drop_token_is_seen(case):
    up = _plain(case, False)
    mut = _plain_mutant(case)
    _reached(up)
    _reached(mut)
    problem = _stream_problem(up, mut)
    assert problem is not None, (
        f"{case}: a streamer that drops its 5th token still compared equal -- "
        "the streaming comparison cannot see a lost token")
    return problem


def _check_dequant_scale_is_seen(case):
    shim = _q8_shim(case, "cpu", "qscale")
    _reached(shim)
    dist, bound, *_ = _q8_gap(case, shim)
    assert dist > bound, (
        f"{case}: a 1% error in the quantised layers' scale moved the logits by "
        f"{dist:.3e}, inside the derived bound {bound:.3e} -- the q8_0 comparison "
        "could not tell a wrong scale from a right one")
    return dist, bound


# --------------------------------------------------------------------------
# mps: pinned at issue #29's walls (pattern of test_qwen3.py)
# --------------------------------------------------------------------------
#
# Known at 652c7d7: `generate` on mps stops at `aten.isin.Tensor_Tensor`
# (refused by name as a host readback) before its first forward; behind it are
# the zero-byte Metal buffer (`DynamicCache` allocates `torch.tensor([])`) and
# argmax. The pin accepts any of the three #29 walls, by name, at the `stream`
# stage, and goes red on any other stop and on success.

_WALLS = (
    "NotImplementedError: aten.isin.Tensor_Tensor: not implemented for the mps device",
    "candle: Metal error Failed to create metal resource: Buffer",
    "argmax",
)


def _is_wall(err):
    text = err["error"]
    return any(w in text for w in _WALLS[:2]) or ("argmax" in text and "mps" in text)


def _pin_wall(case, quant=False):
    rec = _q8_shim(case, "mps") if quant else _plain(case, True, "mps")
    if rec.get("no_mps"):
        raise _skip.Skip(rec["no_mps"])
    assert "stream" not in rec["completed"], (
        f"issue #29 moved: streaming generate on mps ({case}, quant={quant}) now "
        f"completes. Replace this pin with `_agree_on_mps_stream({case!r}, {quant})`.")
    err = rec["errors"].get("stream")
    assert err is not None, (
        f"{case} stream on mps neither completed nor reached its stage: stopped at "
        f"{rec['stage']!r}: {rec.get('error')}\n{rec.get('traceback', '')[-1200:]}")
    assert _is_wall(err), (
        f"{case} stream on mps stopped, but not at one of #29's walls -- a new or "
        f"changed wall, which needs diagnosing rather than re-pinning:\n"
        f"{err['error']}\n{err['traceback'][-1200:]}")


def _agree_on_mps_stream(case, quant=False):
    """What replaces the pin once #29's walls are gone."""
    if quant:
        shim = _q8_shim(case, "mps")
        _reached(shim, "stream")
        problem = _stream_problem(_q8_up(case), shim)
    else:
        shim = _plain(case, True, "mps")
        _reached(shim, "stream")
        problem = _stream_problem(_plain(case, False), shim)
    assert problem is None, f"{case} mps: {problem}"


# --------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------

def test_tiny_qwen3_streamed_tokens_match_upstream_on_cpu():
    _check_plain("tiny-qwen3")


def test_tiny_smollm2_streamed_tokens_match_upstream_on_cpu():
    _check_plain("tiny-smollm2")


def test_tiny_qwen3_q8_0_streamed_tokens_match_upstream_on_the_dequantised_weights():
    _check_q8("tiny-qwen3")


def test_tiny_smollm2_q8_0_streamed_tokens_match_upstream_on_the_dequantised_weights():
    _check_q8("tiny-smollm2")


def test_tiny_the_streaming_comparison_sees_a_dropped_token():
    for case in ("tiny-qwen3", "tiny-smollm2"):
        _check_drop_token_is_seen(case)


def test_tiny_the_q8_0_comparison_sees_a_wrong_dequant_scale():
    for case in ("tiny-qwen3", "tiny-smollm2"):
        _check_dequant_scale_is_seen(case)


def test_tiny_qwen3_streaming_generate_on_mps_is_pinned_at_issue_29():
    _pin_wall("tiny-qwen3")


def test_tiny_smollm2_streaming_generate_on_mps_is_pinned_at_issue_29():
    _pin_wall("tiny-smollm2")


def test_tiny_qwen3_q8_0_streaming_generate_on_mps_is_pinned_at_issue_29():
    _pin_wall("tiny-qwen3", quant=True)


def test_qwen3_0_6b_streamed_tokens_match_upstream_on_cpu():
    _check_plain("qwen3")


def test_qwen3_0_6b_q8_0_streamed_tokens_match_upstream_on_the_dequantised_weights():
    _check_q8("qwen3")


def test_qwen3_0_6b_streaming_generate_on_mps_is_pinned_at_issue_29():
    _pin_wall("qwen3")


def test_smollm2_135m_streamed_tokens_match_upstream_on_cpu():
    _check_plain("smollm2")


def test_smollm2_135m_q8_0_streamed_tokens_match_upstream_on_the_dequantised_weights():
    _check_q8("smollm2")


def test_smollm2_135m_streaming_generate_on_mps_is_pinned_at_issue_29():
    _pin_wall("smollm2")


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    sel = os.environ.get("TORCH_C_STREAM_ONLY")
    if sel:
        items = [(n, f) for n, f in items if sel in n]
    failures = _skip.run_tests(items, suite="test_stream")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
