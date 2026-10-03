"""Qwen3 against upstream: load bit for bit, forward at a derived tolerance,
greedy `generate` token for token -- on cpu and on mps (issue #23).

On 2026-10-03 the user made the Qwen family the test model (Llama 3.2 is gated
on the Hub; Qwen3 is Apache-2.0). Until this file, Qwen3 appeared here only in
synthetic and structural tests; **no real Qwen3 checkpoint had been measured
against upstream.** Issue #11 (streaming + q8_0) builds on these.

What Qwen3 adds over the Llama path the rest of the suite already agrees on
(read off `transformers/models/qwen3/modeling_qwen3.py` in the pinned 5.15.1,
diffed against `modeling_llama.py`):

* `q_norm` / `k_norm` -- an RMSNorm over the **head dim** of q and k, after
  the projection and before rope. Llama has neither.
* `head_dim` is a config field, **decoupled from `hidden_size`**: Qwen3-0.6B is
  1024 wide with 16 heads of 128, so `q_proj` is 1024 -> 2048 and `o_proj`
  2048 -> 1024. Every Llama case here has `head_dim * heads == hidden_size`.
* tied embeddings (`tie_word_embeddings: true`), with `lm_head.weight` *also*
  stored in the file, byte-identical to `embed_tokens`.
* `rope_theta` 1e6, default rope; `sliding_window` is null and
  `use_sliding_window` false, so every layer is `full_attention` -- the
  sliding-window mask path exists in the module but is not taken.
* the checkpoint is bfloat16, and transformers 5's `from_pretrained` defaults
  to `dtype="auto"`, so **the user's line computes in bfloat16.** Both the
  user's dtype and float32 are asserted below; they are separate claims.
* `generation_config.json` sets `do_sample: true`; every `generate` here
  passes `do_sample=False`, so the comparison is greedy and exact.

The bar is AGENTS.md §16's third grade, *agrees*: element-wise against
upstream, in a **separate subprocess** on each side. The tolerance is derived
per model, docs/numerics/AGREE.md §2: upstream re-run in float64, the p90 of
its own float32 (or bfloat16) error relative to the output's scale, floored at
8 ulp -- and, AGREE.md §2's second rule, a distance within 4x upstream's own
distance from float64 is not a defect. There is no constant here to widen.

AGENTS.md §17.5 -- a check that cannot fail is not a check. Two things make
these able to fail:

* **the control** (`..._the_bound_can_see_q_norm_and_k_norm`): upstream's own
  model with `q_norm`/`k_norm` replaced by the identity must land *outside*
  the derived bound. If it did not, a shim that skipped the one operation
  Qwen3 adds would pass every logit test here.
* **the weights are compared to the file**, read with `struct` and `numpy`
  alone, not to anything the library under test produced.

On mps the value is not the evidence (AGENTS.md §16): `_metal_counters()`
must show compute encoders opened during the forward, and fewer bytes read
back than one hidden-state row -- a host-computed twin would read back
activations.

Two families of test, so the gate always has something real to say:

* `test_tiny_qwen3_*` -- a 2-layer Qwen3 with every structural property above
  (decoupled head_dim, GQA, tied, bfloat16, q_norm/k_norm weights moved away
  from 1.0 so they matter), written by upstream into a temporary directory.
  Runs everywhere the vendored shim is built.
* `test_qwen3_0_6b_*` -- the real `Qwen/Qwen3-0.6B`, read from a Hugging Face
  cache **inside this repository** and never downloaded (AGENTS.md §2, §15.1).
  Skips by name when the snapshot is not cached.

Known walls at 652c7d7, measured on the tiny model with the main checkout's
built shim (reaches, not agrees -- AGENTS.md §16): cpu forward and generate
reach and match in both dtypes; on mps the forward reaches only with
`use_cache=False` -- with the default cache, `DynamicCache.lazy_initialization`
calls `torch.tensor([], device=mps)` and candle cannot create a zero-byte Metal
buffer (docs/devices/MPSFWD.md §6 recorded the same wall) -- and `generate`
stops at `aten.isin.Tensor_Tensor`, refused by name on mps as a host readback
(docs/devices/MPS.md). Both walls are issue #29 and are PINNED below, by
name, in the pattern of test_constset.py's NaN-seed pin: the gate stays green
while they stand, and goes red if the line stops anywhere else or gets past
them. The `use_cache=False` forward, which does reach mps, is held to the same
derived bound and counter evidence as the cpu.
"""

import atexit
import functools
import glob
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import _skip

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "python")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

#: float32 machine epsilon. The floor under every derived tolerance.
_EPS32 = 1.1920929e-07

#: docs/numerics/AGREE.md §2's second rule: within this multiple of upstream's
#: own distance from float64 is not a defect.
_ORACLE_FACTOR = 4.0

_MODEL_ID = "Qwen/Qwen3-0.6B"

#: The prompt is the user's kind of input, tokenized on each side by the
#: checkpoint's own tokenizer -- the shim side's `return_tensors="pt"` is part
#: of the user's line.
_PROMPT = "Give me a short introduction to large language models."

#: 20, not 1: docs/devices/MPSFWD.md §6 -- an agreeing argmax is a fact about
#: the first token, not about twenty.
_NEW_TOKENS = 20

#: The tiny model's prompt, as ids (it has no tokenizer).
_TINY_IDS = [5, 7, 9, 11, 13, 17, 19, 23]

_TINY_CFG = dict(
    vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    max_position_embeddings=64, tie_word_embeddings=True, rope_theta=1e6,
    bos_token_id=1, eos_token_id=[2, 3], pad_token_id=1,
)


# --------------------------------------------------------------------------
# Where things are
# --------------------------------------------------------------------------

def _main_checkout():
    """The main checkout's root when this file runs from `.worktrees/<name>`,
    else this checkout's root. The model cache lives in the main checkout
    (AGENTS.md §3: worktrees link large artefacts rather than copy them)."""
    marker = os.sep + ".worktrees" + os.sep
    if marker in _REPO_ROOT + os.sep:
        return _REPO_ROOT.split(marker)[0]
    return _REPO_ROOT


def _snapshot():
    """The cached `Qwen/Qwen3-0.6B` snapshot directory, or None. Never downloads.

    Only caches inside this repository are searched, plus an explicit
    `HF_HOME`: `~/.cache/huggingface` is outside the repository (AGENTS.md
    §15.1), and a snapshot there is not one this project put there.
    """
    roots = [os.environ.get("HF_HOME"),
             os.path.join(_REPO_ROOT, ".caches", "hf-home"),
             os.path.join(_main_checkout(), ".caches", "hf-home")]
    folder = "models--" + _MODEL_ID.replace("/", "--")
    for root in roots:
        if not root:
            continue
        for snap in sorted(glob.glob(os.path.join(root, "hub", folder, "snapshots", "*"))):
            if os.path.isfile(os.path.join(snap, "model.safetensors")) and \
                    os.path.isfile(os.path.join(snap, "tokenizer.json")):
                return snap
    return None


# --------------------------------------------------------------------------
# The oracle for the weights: the file, read without torch
# --------------------------------------------------------------------------

def _file_digests(snapshot):
    """sha256 of every tensor in `model.safetensors`, as little-endian float32.

    Read with `struct` and `numpy` alone, so a marshalling defect in the
    library under test cannot agree with itself. bfloat16 is the top half of
    float32, so `u16 << 16` is an exact widening -- the same bytes a correct
    `from_pretrained(dtype=torch.float32)` must hold, and the same bytes a
    correct bfloat16 load widens to.
    """
    import numpy as np

    path = os.path.join(snapshot, "model.safetensors")
    out = {}
    with open(path, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(n))
        base = 8 + n
        for name, entry in sorted(header.items()):
            if name == "__metadata__":
                continue
            a, b = entry["data_offsets"]
            fh.seek(base + a)
            raw = fh.read(b - a)
            if entry["dtype"] == "BF16":
                wide = (np.frombuffer(raw, dtype="<u2").astype("<u4") << 16).tobytes()
            elif entry["dtype"] == "F32":
                wide = raw
            else:
                raise AssertionError(f"{name}: unexpected dtype {entry['dtype']}")
            out[name] = (hashlib.sha256(wide).hexdigest(), entry["dtype"])
            del raw, wide
    return out


# --------------------------------------------------------------------------
# One script, run on either side
# --------------------------------------------------------------------------

#: Run by upstream torch (no PYTHONPATH) and by the shim (the vendored tree on
#: PYTHONPATH). Everything it reports goes to stdout as one JSON object; the
#: logits go to files, as raw little-endian bytes, because 151936 columns do not
#: belong in JSON. It never raises: a stop is reported with its stage and
#: traceback, so a test can name the wall instead of a bare exit code.
_SIDE_SCRIPT = r"""
import hashlib, json, os, sys, traceback
A = json.loads(sys.argv[1])
import torch
IS_SHIM = hasattr(torch._C, "_aten_implemented")
assert IS_SHIM == A["shim"], ("wrong torch on this side", IS_SHIM, A["shim"])
out = {"stage": "start", "shim": IS_SHIM, "completed": [], "errors": {}, "tag": A["tag"]}


def raw(t):
    t = t.detach()
    if t.dtype in (torch.bfloat16, torch.float16):
        t = t.float()
    t = t.contiguous()
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


def counters():
    if not (IS_SHIM and A["device"] == "mps"):
        return None
    c = torch._C._metal_counters()
    return {k: v for k, v in c.items() if k != "built"}


try:
    if A.get("write_tiny"):
        out["stage"] = "write_tiny"
        from transformers import AutoModelForCausalLM, Qwen3Config
        torch.manual_seed(0)
        m = AutoModelForCausalLM.from_config(Qwen3Config(**A["write_tiny"])).eval()
        with torch.no_grad():
            for p in m.parameters():
                # Away from 1.0 so q_norm/k_norm are not the identity.
                p.normal_(0.0, 0.3)
        m.to(torch.bfloat16).save_pretrained(A["path"])
        del m

    if IS_SHIM and A["device"] == "mps":
        if not torch._C._metal_counters()["built"]:
            out["no_mps"] = "this build has no Metal backend"
            print(json.dumps(out)); raise SystemExit(0)

    out["stage"] = "load"
    from transformers import AutoModelForCausalLM, AutoTokenizer
    kw = {} if A["dtype"] == "auto" else {"dtype": getattr(torch, A["dtype"])}
    if A.get("attn"):
        kw["attn_implementation"] = A["attn"]
    model = AutoModelForCausalLM.from_pretrained(A["path"], **kw).eval()
    out["attn_implementation"] = model.config._attn_implementation
    out["param_dtypes"] = sorted({str(p.dtype) for p in model.parameters()})

    if A.get("digest"):
        out["stage"] = "digest"
        out["digests"] = {k: hashlib.sha256(raw(v)).hexdigest()
                          for k, v in sorted(model.state_dict().items())}
        out["completed"].append("digest")

    out["stage"] = "inputs"
    if A.get("prompt") is not None:
        tok = AutoTokenizer.from_pretrained(A["path"])
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
        # Each stage records its own outcome, so a wall in one (the mps
        # pins, issue #29) does not hide whether the others got through.
        out["stage"] = name
        try:
            fn()
        except BaseException as exc:
            out["errors"][name] = {
                "error": "%s: %s" % (type(exc).__name__, str(exc)[:1500]),
                "traceback": traceback.format_exc()[-4000:]}
        else:
            out["completed"].append(name)

    def forward():
        before = counters()
        with torch.no_grad():
            logits = model(**enc).logits
        after = counters()
        if before is not None:
            out["forward_counters"] = {k: after[k] - before[k] for k in before}
        dump("logits", logits)

    def forward_nocache():
        # The same forward without a KV cache: on mps this steps around the
        # zero-byte `torch.tensor([])` that `DynamicCache` allocates (#29).
        before = counters()
        with torch.no_grad():
            logits = model(**enc, use_cache=False).logits
        after = counters()
        if before is not None:
            out["forward_nocache_counters"] = {k: after[k] - before[k] for k in before}
        dump("logits_nocache", logits)

    def control():
        # q_norm / k_norm replaced by the identity: Qwen3 without the one
        # operation it adds over Llama.
        hooks = [mod.register_forward_hook(lambda mod, args, res: args[0])
                 for name, mod in model.named_modules()
                 if name.endswith(".q_norm") or name.endswith(".k_norm")]
        out["control_hooks"] = len(hooks)
        try:
            with torch.no_grad():
                dump("control", model(**enc).logits)
        finally:
            for h in hooks:
                h.remove()

    def generate():
        with torch.no_grad():
            seq = model.generate(**enc, max_new_tokens=A["generate"], do_sample=False)
        out["generated"] = seq.cpu().tolist()

    attempt("forward", forward)
    if A.get("nocache"):
        attempt("forward_nocache", forward_nocache)
    if A.get("control"):
        attempt("control", control)
    if A.get("generate"):
        attempt("generate", generate)
    if out["errors"]:
        raise RuntimeError(
            "stage(s) stopped: " + ", ".join(sorted(out["errors"])))

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
        capture_output=True, text=True, env=env, timeout=timeout,
    )
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
    with open(meta["path"], "rb") as fh:
        arr = np.frombuffer(fh.read(), dtype=dt).astype("<f8")
    assert arr.size == int(np.prod(meta["shape"])), (name, arr.size, meta["shape"])
    return arr


def _reached(rec, what="done"):
    """Assert the side got through `what` ("forward", "generate", or "done"
    for everything it was asked to do), naming the wall if it did not.

    Per stage, so a wall in `generate` does not also fail the forward test
    run by the same subprocess -- each test reports its own wall.
    """
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
    root = tempfile.mkdtemp(prefix="qwen3-agree-")
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root


@functools.lru_cache(maxsize=None)
def _case(name):
    """Fixed inputs for one model: where it is and what to feed it."""
    _require_shim()
    if name == "tiny":
        path = os.path.join(_workdir(), "tiny-ckpt")
        hf_home = os.path.join(_workdir(), "hf-home")
        rec = _run_side({"shim": False, "tag": "tiny-write", "out": _workdir(),
                         "path": path, "write_tiny": _TINY_CFG, "dtype": "auto",
                         "device": "cpu", "ids": _TINY_IDS, "hf_home": hf_home})
        _reached(rec)
        return {"path": path, "ids": _TINY_IDS, "prompt": None,
                "hf_home": hf_home, "tied_from": "model.embed_tokens.weight"}
    snap = _snapshot()
    if snap is None:
        raise _skip.Skip(
            f"{_MODEL_ID} is not cached under {_REPO_ROOT}/.caches/hf-home, "
            f"{_main_checkout()}/.caches/hf-home or $HF_HOME; fetch it with "
            f"HF_HOME=<repo>/.caches/hf-home -- the gate never downloads")
    return {"path": snap, "ids": None, "prompt": _PROMPT,
            # <root>/hub/models--Qwen--Qwen3-0.6B/snapshots/<sha>
            "hf_home": os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(snap)))),
            "tied_from": "model.embed_tokens.weight"}


@functools.lru_cache(maxsize=None)
def _side(case, shim, dtype, device="cpu", attn=None):
    c = _case(case)
    upstream_f32 = (not shim and dtype == "float32")
    args = {
        "shim": shim, "dtype": dtype, "device": device, "attn": attn,
        "tag": f"{case}-{'shim' if shim else 'up'}-{dtype}-{device}-{attn}",
        "out": _workdir(), "path": c["path"], "ids": c["ids"],
        "prompt": c["prompt"], "hf_home": c["hf_home"],
        # Weights are digested where they are loaded on the cpu, in both dtypes.
        "digest": device == "cpu" and dtype != "float64",
        "control": upstream_f32,
        "nocache": shim and device != "cpu",
        "generate": None if dtype == "float64" else
                    (_NEW_TOKENS if c["prompt"] is not None else 8),
    }
    return _run_side(args)


@functools.lru_cache(maxsize=None)
def _oracle(case):
    """Expected digest per state-dict key, from the file."""
    c = _case(case)
    file = _file_digests(c["path"])
    return {k: v[0] for k, v in file.items()}


# --------------------------------------------------------------------------
# The checks, shared by both cases
# --------------------------------------------------------------------------

def _check_bit_for_bit(case, dtype):
    want = _oracle(case)
    up = _side(case, False, dtype)
    shim = _side(case, True, dtype)
    _reached(up, "digest")
    _reached(shim, "digest")
    expected_dtype = "torch.float32" if dtype == "float32" else "torch.bfloat16"
    assert shim["param_dtypes"] == [expected_dtype], shim["param_dtypes"]
    assert up["param_dtypes"] == [expected_dtype], up["param_dtypes"]
    keys = sorted(up["digests"])
    assert sorted(shim["digests"]) == keys, (
        sorted(set(keys) ^ set(shim["digests"])))
    tied = _case(case)["tied_from"]
    checked = 0
    for k in keys:
        # `lm_head.weight` is tied: when the file does not store it, its
        # bytes are `embed_tokens`'.
        expected = want.get(k) or (want[tied] if k == "lm_head.weight" else None)
        assert expected is not None, f"{k}: in the state dict but not in the file"
        # The oracle is checked against upstream first: if upstream does not
        # reproduce the file, the comparison below is measuring nothing.
        assert up["digests"][k] == expected, f"upstream does not reproduce the file at {k}"
        assert shim["digests"][k] == expected, f"{k}: the shim's bytes are not the file's"
        checked += 1
    assert checked == len(keys) and checked >= len(want), (checked, len(want))
    return checked


def _bound(up, up64):
    """docs/numerics/AGREE.md §2, derived for THIS model and THIS dtype.

    Returns (bound in output units, the p90 tolerance, upstream's own worst
    relative distance from float64, scale).
    """
    import numpy as np

    scale = float(np.max(np.abs(up64)))
    assert scale > 0, "degenerate output: every logit is zero"
    own = np.abs(up - up64) / scale
    p90 = float(np.sort(own)[int(0.9 * (own.size - 1))])
    tol = max(p90, 8 * _EPS32)
    worst_own = float(own.max())
    return max(tol, _ORACLE_FACTOR * worst_own) * scale, tol, worst_own, scale


def _check_logits(case, dtype, device="cpu", attn=None, stage="forward"):
    """`stage="forward_nocache"` compares the shim's `use_cache=False`
    forward against upstream's ordinary one: a single prefill computes the
    same logits with or without a cache, the cache only keeps k and v."""
    import numpy as np

    up = _side(case, False, dtype, "cpu", attn)
    up64 = _side(case, False, "float64", "cpu", attn)
    shim = _side(case, True, dtype, device, attn)
    _reached(up)
    _reached(up64)
    _reached(shim, stage)
    out = "logits" if stage == "forward" else "logits_nocache"
    assert shim["input_ids"] == up["input_ids"] == up64["input_ids"], (
        "the two sides did not feed the same tokens", shim["input_ids"], up["input_ids"])
    assert shim["attn_implementation"] == up["attn_implementation"], (
        shim["attn_implementation"], up["attn_implementation"])
    assert shim[out]["shape"] == up["logits"]["shape"], (
        shim[out]["shape"], up["logits"]["shape"])
    assert shim[out]["dtype"] == up["logits"]["dtype"], (
        shim[out]["dtype"], up["logits"]["dtype"])
    if device != "cpu":
        assert shim[out]["device"].startswith(device), shim[out]["device"]
    a, u, u64 = _read(shim, out), _read(up, "logits"), _read(up64, "logits")
    assert np.all(np.isfinite(u64)), "upstream's float64 oracle is not finite"
    bound, tol, own, scale = _bound(u, u64)
    dist = float(np.max(np.abs(a - u)))
    assert dist <= bound, (
        f"{case}/{dtype}/{device}/{stage}: shim logits differ from upstream by {dist:.3e} "
        f"(relative {dist / scale:.3e}); the derived bound is {bound:.3e} "
        f"(p90 tolerance {tol:.3e}, upstream's own worst {own:.3e}, scale {scale:.3e})")
    return dist, bound


def _check_control(case):
    import numpy as np

    up = _side(case, False, "float32")
    up64 = _side(case, False, "float64")
    _reached(up)
    _reached(up64)
    layers = 2 if case == "tiny" else 28
    assert up["control_hooks"] == 2 * layers, up["control_hooks"]
    bound, _, _, _ = _bound(_read(up, "logits"), _read(up64, "logits"))
    gap = float(np.max(np.abs(_read(up, "control") - _read(up, "logits"))))
    assert gap > bound, (
        f"{case}: removing q_norm/k_norm moved the logits by only {gap:.3e}, "
        f"inside the derived bound {bound:.3e} -- the logit tests could not "
        "tell a shim that skips them from one that runs them")
    return gap, bound


def _check_generate(case, dtype, device="cpu"):
    up = _side(case, False, dtype)
    shim = _side(case, True, dtype, device)
    _reached(up)
    _reached(shim, "generate")
    assert shim["input_ids"] == up["input_ids"], (shim["input_ids"], up["input_ids"])
    prompt_len = len(up["input_ids"][0])
    assert len(up["generated"][0]) > prompt_len, "upstream generated nothing"
    if shim["generated"] == up["generated"]:
        return None
    message = (f"{case}/{dtype}/{device}: greedy generate diverged\n"
               f"  upstream {up['generated'][0][prompt_len:]}\n"
               f"  shim     {shim['generated'][0][prompt_len:]}")
    # float32 has no tie allowance: it must be token for token.
    assert dtype != "float32", message
    return _check_divergence_is_a_tie(case, dtype, device, up, shim, message)


def _check_divergence_is_a_tie(case, dtype, device, up, shim, message):
    """A low-precision greedy run may part from upstream only at a tie that
    the derived bound cannot resolve -- and nowhere else.

    The two sides share every token before the first divergence, so both are
    asked for the logits of that shared prefix, with a float64 upstream as the
    oracle. Accepted only if (a) the shim's logits there are within the bound
    derived for this model and dtype (docs/numerics/AGREE.md §2), and (b)
    upstream's own margin between its token and the shim's token is within
    that bound too: a margin the bound can see would make the shim's choice a
    defect, not a tie. After the divergence the prefixes differ, so nothing
    later is compared.

    What this cannot see: a defect that only ever moves near-tied logits. The
    logit tests are what bound that.
    """
    import numpy as np

    u_seq, s_seq = up["generated"][0], shim["generated"][0]
    k = next(i for i, (a, b) in enumerate(zip(u_seq, s_seq)) if a != b)
    assert k >= len(up["input_ids"][0]), "the sides differ inside the prompt"
    prefix = u_seq[:k]
    ups = _side_ids(case, False, dtype, "cpu", prefix)
    up64 = _side_ids(case, False, "float64", "cpu", prefix)
    sh = _side_ids(case, True, dtype, device, prefix)
    for rec in (ups, up64, sh):
        _reached(rec)
    u = _read(ups, "logits").reshape(ups["logits"]["shape"])[0, -1]
    u64 = _read(up64, "logits").reshape(up64["logits"]["shape"])[0, -1]
    s = _read(sh, "logits").reshape(sh["logits"]["shape"])[0, -1]
    report = _judge_tie(u, u64, s, u_seq[k], s_seq[k], k, message)
    print("     tie: " + report.replace("\n", "\n     "))
    return k


def _judge_tie(u, u64, s, a, b, k, message=""):
    """The decision alone, on one row of logits: upstream `u`, its float64
    oracle `u64`, the shim `s`; upstream chose token `a`, the shim `b`."""
    import numpy as np

    bound, tol, own, scale = _bound(u, u64)
    dist = float(np.max(np.abs(s - u)))
    margin = float(u[a] - u[b])
    margin64 = float(u64[a] - u64[b])
    report = (f"{message}\n  first divergence at position {k}: upstream chose {a}, "
              f"shim chose {b}\n  shim-vs-upstream logit distance {dist:.3e}, "
              f"upstream margin {margin:.3e} (float64 oracle {margin64:.3e}), "
              f"derived bound {bound:.3e} (p90 {tol:.3e}, own worst {own:.3e}, "
              f"scale {scale:.3e})")
    assert dist <= bound, report + "\n  -> the shim's logits are outside the bound: a defect"
    assert abs(margin) <= bound, (
        report + "\n  -> upstream's margin is larger than the bound, so the bound "
        "can resolve this choice: the shim picked the wrong token, a defect")
    return report


def _side_ids(case, shim, dtype, device, ids):
    """One forward over explicit token ids (no generate, no digest)."""
    c = _case(case)
    args = {
        "shim": shim, "dtype": dtype, "device": device, "attn": None,
        "tag": f"{case}-{'shim' if shim else 'up'}-{dtype}-{device}-prefix{len(ids)}",
        "out": _workdir(), "path": c["path"], "ids": list(ids),
        "prompt": None, "hf_home": c["hf_home"],
        "digest": False, "control": False, "nocache": False, "generate": None,
    }
    return _run_side(args)


def _check_metal_ran(case, dtype, stage="forward"):
    shim = _side(case, True, dtype, "mps")
    _reached(shim, stage)
    c = shim[stage + "_counters"]
    layers = 2 if case == "tiny" else 28
    # A lower bound on GPU work: at least one compute encoder per layer.
    assert c["compute_encoders"] >= layers, c
    # A host-computed twin reads activations back; one hidden-state row is
    # `hidden_size * 4` bytes. Measured on the tiny model: 1 byte (a bool).
    hidden = _TINY_CFG["hidden_size"] if case == "tiny" else 1024
    assert c["host_download_bytes"] < hidden * 4, c


# --------------------------------------------------------------------------
# mps: the user's line is pinned at issue #29's two walls
# --------------------------------------------------------------------------
#
# The pattern of test_constset.py's
# `test_the_nan_seed_on_mps_is_either_refused_or_a_recorded_defect`: a known
# wall is asserted BY NAME, so the gate stays green while it stands, goes red
# if the line stops anywhere else, and goes red when the wall falls -- forcing
# whoever fixed it here to put the agreement assertion in its place. The
# agreement and counter assertions are kept below as `_agree_on_mps_*`, so
# the replacement is one line.

#: Wall 1: `DynamicCache.lazy_initialization` calls
#: `torch.tensor([], device=mps)` and candle cannot create a zero-byte Metal
#: buffer (docs/devices/MPSFWD.md §6 recorded it first).
_WALL1_TEXT = "candle: Metal error Failed to create metal resource: Buffer"
_WALL1_FRAME = "in lazy_initialization"

#: Wall 2: `generate`'s `_prepare_special_tokens` calls `torch.isin` on mps
#: tensors, and `aten.isin.Tensor_Tensor` is refused there as a host readback
#: (docs/devices/MPS.md). It is reached before generate's first forward.
_WALL2_TEXT = "NotImplementedError: aten.isin.Tensor_Tensor: not implemented for the mps device"


def _is_wall(name, err):
    if name == "metal_zero_byte_buffer":
        return (err["error"].startswith("RuntimeError: torch.tensor: ")
                and _WALL1_TEXT in err["error"]
                and _WALL1_FRAME in err["traceback"])
    if name == "isin_host_readback":
        return err["error"].startswith(_WALL2_TEXT)
    raise AssertionError(f"no such wall: {name}")


_REPLACEMENT = {"forward": "_agree_on_mps_forward", "generate": "_agree_on_mps_generate"}


def _pin_wall(case, stage, wall):
    rec = _side(case, True, "float32", "mps")
    if rec.get("no_mps"):
        raise _skip.Skip(rec["no_mps"])
    replacement = f"{_REPLACEMENT[stage]}({case!r})"
    assert stage not in rec["completed"], (
        f"issue #29 moved: {case} {stage} on mps now gets past the {wall} wall. "
        f"This pin must be replaced by the agreement and counter assertions -- "
        f"change the test body to `{replacement}` (and close or update #29)")
    err = rec["errors"].get(stage)
    assert err is not None, (
        f"{case} {stage} on mps neither completed nor reached its stage: the "
        f"side stopped at {rec['stage']!r}: {rec.get('error')}\n"
        f"{rec.get('traceback', '')[-1200:]}")
    assert _is_wall(wall, err), (
        f"{case} {stage} on mps stopped, but not at #29's {wall} wall -- a new "
        f"or changed wall, which needs diagnosing rather than re-pinning:\n"
        f"{err['error']}\n{err['traceback'][-1200:]}")


def _agree_on_mps_forward(case):
    """What replaces the forward pin once #29's wall 1 is gone."""
    _check_logits(case, "float32", "mps")
    _check_metal_ran(case, "float32")


def _agree_on_mps_generate(case):
    """What replaces the generate pin once #29's walls are gone."""
    _check_generate(case, "float32", "mps")


# --------------------------------------------------------------------------
# The tiny Qwen3: runs wherever the vendored shim is built
# --------------------------------------------------------------------------

def test_tiny_qwen3_from_pretrained_loads_bit_for_bit_in_its_own_bfloat16():
    _check_bit_for_bit("tiny", "auto")


def test_tiny_qwen3_from_pretrained_loads_bit_for_bit_widened_to_float32():
    _check_bit_for_bit("tiny", "float32")


def test_tiny_qwen3_float32_logits_agree_with_upstream_on_cpu():
    _check_logits("tiny", "float32")


def test_tiny_qwen3_float32_logits_agree_with_upstream_on_cpu_with_eager_attention():
    _check_logits("tiny", "float32", attn="eager")


def test_tiny_qwen3_default_dtype_logits_agree_with_upstream_on_cpu():
    _check_logits("tiny", "auto")


def test_tiny_qwen3_the_bound_can_see_q_norm_and_k_norm():
    _check_control("tiny")


def test_the_tie_allowance_refuses_a_margin_the_bound_can_see():
    """Nullification of the bf16 generate criterion, without a model: the
    tie allowance must reject (a) a shim choice upstream separates by more
    than the bound, and (b) shim logits outside the bound -- and accept a
    genuine tie."""
    import numpy as np

    rng = np.random.default_rng(0)
    u64 = rng.normal(size=4096) * 10.0
    u = u64 + rng.normal(size=4096) * 1e-3          # upstream's own error
    bound = _bound(u, u64)[0]
    a, b = 7, 11
    tie = u.copy(); tie[a] = 50.0; tie[b] = 50.0 - bound / 4
    tie64 = u64.copy(); tie64[a] = 50.0; tie64[b] = 50.0 - bound / 4
    _judge_tie(tie, tie64, tie, a, b, 0)              # a tie: accepted
    wide = tie.copy(); wide[b] = 50.0 - 1000 * bound
    wide64 = tie64.copy(); wide64[b] = 50.0 - 1000 * bound
    try:
        _judge_tie(wide, wide64, wide, a, b, 0)
    except AssertionError as e:
        assert "margin is larger than the bound" in str(e), e
    else:
        raise AssertionError("a margin 1000x the bound was accepted as a tie")
    off = tie.copy(); off += 100 * bound
    try:
        _judge_tie(tie, tie64, off, a, b, 0)
    except AssertionError as e:
        assert "outside the bound" in str(e), e
    else:
        raise AssertionError("shim logits 100x outside the bound were accepted")


def test_tiny_qwen3_greedy_generate_matches_upstream_on_cpu_float32():
    _check_generate("tiny", "float32")


def test_tiny_qwen3_greedy_generate_matches_upstream_on_cpu_default_dtype():
    _check_generate("tiny", "auto")


def test_tiny_qwen3_forward_on_mps_is_pinned_at_the_zero_byte_metal_buffer():
    """Pinned, issue #29: the user's forward stops at wall 1. When it gets
    past, replace the body with `_agree_on_mps_forward("tiny")`."""
    _pin_wall("tiny", "forward", "metal_zero_byte_buffer")


def test_tiny_qwen3_generate_on_mps_is_pinned_at_the_isin_readback_refusal():
    """Pinned, issue #29: greedy `generate` stops at wall 2, before its first
    forward. When it gets past, replace the body with
    `_agree_on_mps_generate("tiny")`."""
    _pin_wall("tiny", "generate", "isin_host_readback")


def test_tiny_qwen3_float32_logits_without_a_cache_agree_on_mps_and_metal_computed_them():
    """`use_cache=False` steps around wall 1 and reaches mps today; this is
    the agreement and counter evidence for that forward."""
    _check_logits("tiny", "float32", "mps", stage="forward_nocache")
    _check_metal_ran("tiny", "float32", stage="forward_nocache")


# --------------------------------------------------------------------------
# The real Qwen/Qwen3-0.6B: skips by name when it is not cached
# --------------------------------------------------------------------------

def test_qwen3_0_6b_from_pretrained_loads_bit_for_bit_in_its_own_bfloat16():
    """The user's line, `from_pretrained(path)`: transformers 5 defaults to
    `dtype="auto"`, so this is bfloat16 -- 311 tensors from the file plus
    the tied head."""
    _check_bit_for_bit("0.6b", "auto")


def test_qwen3_0_6b_from_pretrained_loads_bit_for_bit_widened_to_float32():
    _check_bit_for_bit("0.6b", "float32")


def test_qwen3_0_6b_float32_logits_agree_with_upstream_on_cpu():
    _check_logits("0.6b", "float32")


def test_qwen3_0_6b_default_dtype_logits_agree_with_upstream_on_cpu():
    """bfloat16, the user's dtype. The bound is derived from upstream's own
    bfloat16-vs-float64 error on this model -- not borrowed from float32."""
    _check_logits("0.6b", "auto")


def test_qwen3_0_6b_the_bound_can_see_q_norm_and_k_norm():
    _check_control("0.6b")


def test_qwen3_0_6b_greedy_generate_matches_upstream_on_cpu_float32():
    _check_generate("0.6b", "float32")


def test_qwen3_0_6b_greedy_generate_matches_upstream_on_cpu_default_dtype():
    _check_generate("0.6b", "auto")


def test_qwen3_0_6b_forward_on_mps_is_pinned_at_the_zero_byte_metal_buffer():
    """Pinned, issue #29: the user's forward stops at wall 1. When it gets
    past, replace the body with `_agree_on_mps_forward("0.6b")`."""
    _pin_wall("0.6b", "forward", "metal_zero_byte_buffer")


def test_qwen3_0_6b_generate_on_mps_is_pinned_at_the_isin_readback_refusal():
    """Pinned, issue #29: greedy `generate` stops at wall 2, before its first
    forward. When it gets past, replace the body with
    `_agree_on_mps_generate("0.6b")`."""
    _pin_wall("0.6b", "generate", "isin_host_readback")


def test_qwen3_0_6b_float32_logits_without_a_cache_agree_on_mps_and_metal_computed_them():
    """`use_cache=False` steps around wall 1 and reaches mps today; this is
    the agreement and counter evidence for that forward."""
    _check_logits("0.6b", "float32", "mps", stage="forward_nocache")
    _check_metal_ran("0.6b", "float32", stage="forward_nocache")


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = _skip.run_tests(items, suite="test_qwen3")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
