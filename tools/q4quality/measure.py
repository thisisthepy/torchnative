#!/usr/bin/env python3
"""Q4 quality on a real model, against upstream and against llama.cpp (issue #16).

    .caches/spike-venv/bin/python tools/q4quality/measure.py run \\
        --model qwen3-0.6b --formats q4_0,q4_k \\
        --gguf-py <dir containing gguf/, or a gguf-*.whl> \\
        --out .scratch/target/q4quality/qwen3-0.6b

`--out` must be inside this repository (AGENTS.md §2), and under a directory
named `target`: `test_docrefs.py` reads every file in the tree as text except
under `target/`, and the logits this writes are gigabytes.

Every phase runs in its own subprocess, foreground, in order; nothing is
backgrounded. Each writes `<out>/<phase>.json`, and `report` merges them into
`<out>/results.json` and `<out>/results.md` -- the second is what a document
quotes, and every number in it carries its grade (AGENTS.md §16).

======================================================================
The design, and why each part is there
======================================================================

**Two references, because one cannot tell "lossy" from "wrong".** A buggy Q4
and a correct-but-lossy Q4 can show the same perplexity. So:

  R0  upstream torch, dense float32.            What quantisation costs.
  R1  upstream torch, weights = llama.cpp's     Whether *our* Q4 is right:
      reader (gguf-py) applied to OUR bytes.    same bytes, independent reader,
                                                independent stack.
  R2  upstream torch, weights = llama.cpp's     What llama.cpp's own writer
      writer applied to the same dense W.       costs. For q4_0 this is gguf-py's
                                                writer (expected byte-identical
                                                to ours, and checked); for q4_k
                                                it is `gguf_ref.quantize_q4_k_llamacpp`
                                                because candle's writer is NOT
                                                llama.cpp's (make_qkx1 vs make_qkx2).
  R3  (--gguf FILE) upstream torch, weights     The same question without a
      read from a real llama.cpp GGUF.          transcription in the way.

and on the shim side

  S0  shim, dense float32.                      The stack's own floor vs R0.
  S1  shim dense, weights = candle's reader     Must AGREE with R1 (same weights,
      applied to our bytes.                     dense compute): AGREE.md's rule.
  S2  shim, TorchnativeConfig(fmt).             The thing being shipped.

**Correctness is established layer by layer, not inferred from a model.**
AGENTS.md §17.5: a model exercises a fraction of the kernels, and a wrong
kernel can leave a model replay green. So the chain is

  1. bytes:   our q4_0 blob == gguf-py's writer, every converted layer   (exact)
  2. reader:  candle's dequant == gguf-py's dequant, sampled rows        (bit-exact)
  3. matmul:  S2's QuantizedLinear output vs the exact value of the
              Q8-activation x Q4-weight product, on real activations   (derived bound)
  4. stack:   S1 vs R1, element-wise, AGREE.md's per-model rule        (agrees / diverge)

and only after that do the model-level numbers mean "this is the cost of the
format" rather than "this is the cost of the format plus a bug".

**Why S2 is not held to R1 element-wise.** Candle (and llama.cpp's CPU path)
quantises the *activation* to Q8_0/Q8_K inside the matmul. That rounding is a
step function of its input, so a last-bit difference upstream of a layer can
move one activation across a rounding boundary and change the layer output by
a whole Q8 step. There is no element-wise tolerance that admits that and
refuses a bug, so step 3 checks the matmul on identical inputs instead, and
S2's model-level numbers are graded *measured*, never *agrees*.

**What is measured, and what each metric misses.**

  * Per-token KL(R || S) over the full vocabulary (mean, median, p90, p99,
    max). The most sensitive: it sees every logit. It is what llama.cpp's
    `llama-perplexity --kl-divergence` reports, so the numbers are comparable
    with the community's Q4 tables. *Misses*: whether the shift lands on the
    token that matters -- a KL spread over junk tokens and a KL that flips
    the argmax can have the same mean.
  * Perplexity ratio, `exp(mean(nll_S - nll_R))`, with its standard error.
    The number a user knows. *Misses*: everything about the distribution
    except the probability of the actual next token; errors that cancel over
    the text average away.
  * Top-1 agreement (teacher-forced): fraction of positions with the same
    argmax. *Misses*: compounding -- every position is given the reference
    prefix.
  * Greedy generation: tokens until the first divergence from R0, per fixed
    prompt. What a user of `generate` sees, compounding included. *Misses*:
    after the first divergence the two texts are unrelated, so only the
    divergence point is reported, never "agreement" past it.

**Thresholds.** Only correctness checks have pass/fail lines, and every one is
derived: bytes and bits are exact; the matmul bound is Higham's gamma_K with K
counted from candle's kernels (`gguf_ref.py`); S1-vs-R1 uses AGREE.md's rule
with the tolerance re-derived for this model from R1's own float32-vs-float64
error (`--oracle-f64`, on by default). The quality numbers are not thresholded:
"acceptable" for Gemstone means "no worse than llama.cpp on the same weights,
text and metric", so they are reported beside R2/R3 with paired standard
errors, and the report says whether the difference is within them. No number
in this file was picked to make an outcome come out.

**The text.** wikitext-2-raw `wiki.test.raw`, from the exact zip llama.cpp's
`scripts/get-wikitext-2.sh` downloads (`ggml-org/ci`, sha256 pinned below),
scored the way `llama-perplexity` scores it: chunks of `--ctx` tokens, the
first half as context, the rest scored. Same text, same scoring, so a number
here can stand next to a llama.cpp number on the same model.

**What this harness cannot find** (AGENTS.md §17.4): quality on tasks rather
than text (no benchmark suite); anything the converted set does not cover
(the tied `lm_head` and the embedding stay dense by default, while a llama.cpp
GGUF quantises its embedding, so R3 is replaced only on the converted set and
the report lists what it skipped); bf16/f16 activations (candle's QMatMul is
f32 here); and any device but the CPU.

**Getting gguf-py without installing into the pinned venv** (AGENTS.md §15.2):
download the wheel into this repository and point `--gguf-py` at it; the
package is pure Python and imports from the zip.

    .caches/spike-venv/bin/python -m pip download --no-deps --no-cache-dir \\
        gguf -d .caches/gguf-py
    --gguf-py .caches/gguf-py/gguf-<version>-py3-none-any.whl
"""

import argparse
import hashlib
import io
import json
import math
import os
import platform
import struct
import subprocess
import sys
import time
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
PYTESTS = os.path.join(REPO, "rust", "torch_c", "pytests")
VENDOR = os.path.join(REPO, "torchnative", "src", "main")


def _caches_root():
    """`.caches/` of the main checkout, which a worktree does not have.

    Worktrees live in `.worktrees/<name>` and link large artefacts rather than
    copy them (AGENTS.md §3), so the model cache is the main checkout's. Git's
    common dir is the main checkout's `.git`; its parent is that checkout.
    """
    local = os.path.join(REPO, ".caches")
    if os.path.isdir(local):
        return local
    p = subprocess.run(["git", "-C", REPO, "rev-parse", "--path-format=absolute", "--git-common-dir"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"cannot locate the main checkout: {p.stderr.strip()}")
    return os.path.join(os.path.dirname(p.stdout.strip()), ".caches")


CACHES = _caches_root()
DEFAULT_PY = os.path.join(CACHES, "spike-venv", "bin", "python")
HF_HOME = os.path.join(CACHES, "hf-home")

MODELS = {
    "qwen3-0.6b": "Qwen/Qwen3-0.6B",
    "smollm2-135m": "HuggingFaceTB/SmolLM2-135M",
}

# llama.cpp's own perplexity text: scripts/get-wikitext-2.sh fetches this zip.
# The sha256 is the hub's X-Linked-Etag for the LFS object (checked 2026-10-03).
TEXT = {
    "repo_id": "ggml-org/ci",
    "repo_type": "dataset",
    "filename": "wikitext-2-raw-v1.zip",
    "member": "wikitext-2-raw/wiki.test.raw",
    "sha256": "ef7edb566e3e2b2d31b29c1fdb0c89a4cc683597484c3dc2517919c615435a11",
}

# Fixed prompts for the greedy check. Plain continuations, no chat template,
# so the two stacks see the same token ids from the same tokenizer call.
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "In 1905, Albert Einstein published four papers that",
    "Photosynthesis is the process by which",
    "The three primary colours of light are",
]

PHASES = ("quantise", "weights", "reference", "candidate", "report")


# ======================================================================
# shared helpers (stdlib + numpy only)
# ======================================================================


def _np():
    import numpy as np

    return np


def _log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", file=sys.stderr, flush=True)


def _load(out, name):
    with open(os.path.join(out, name)) as fh:
        return json.load(fh)


def _dump(out, name, obj):
    path = os.path.join(out, name)
    with open(path + ".partial", "w") as fh:
        json.dump(obj, fh, indent=1, sort_keys=True, default=_json_default)
        fh.write("\n")
    os.replace(path + ".partial", path)


def _json_default(o):
    np = _np()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def _snapshot_dir(repo_id):
    """The cached snapshot under this repository's HF_HOME, never ~/.cache."""
    base = os.path.join(HF_HOME, "hub", "models--" + repo_id.replace("/", "--"), "snapshots")
    snaps = sorted(os.listdir(base)) if os.path.isdir(base) else []
    if len(snaps) != 1:
        raise RuntimeError(
            f"expected exactly one cached snapshot of {repo_id} under {base}, found {snaps}. "
            f"Fetch it with HF_HOME={HF_HOME} (never ~/.cache)."
        )
    return os.path.join(base, snaps[0]), snaps[0]


def _safetensors_index(model_dir):
    """`{tensor_name: (file, dtype, shape, start, end)}`, parsed by hand.

    By hand because the `safetensors` numpy loader has no bfloat16, and both
    models ship bf16. The header is a little-endian u64 length and JSON.
    """
    files = [f for f in sorted(os.listdir(model_dir)) if f.endswith(".safetensors")]
    out = {}
    for f in files:
        path = os.path.join(model_dir, f)
        with open(path, "rb") as fh:
            (n,) = struct.unpack("<Q", fh.read(8))
            header = json.loads(fh.read(n))
        base = 8 + n
        for k, v in header.items():
            if k == "__metadata__":
                continue
            s, e = v["data_offsets"]
            out[k] = (path, v["dtype"], tuple(v["shape"]), base + s, base + e)
    return out


def _read_f32(index, key):
    """One tensor as float32. bf16 -> f32 is exact (a 16-bit shift), which is
    also what `from_pretrained(dtype=float32)` does on both stacks."""
    np = _np()
    path, dtype, shape, s, e = index[key]
    with open(path, "rb") as fh:
        fh.seek(s)
        raw = fh.read(e - s)
    if dtype == "BF16":
        a = (np.frombuffer(raw, dtype="<u2").astype(np.uint32) << 16).view(np.float32)
    elif dtype == "F16":
        a = np.frombuffer(raw, dtype="<f2").astype(np.float32)
    elif dtype == "F32":
        a = np.frombuffer(raw, dtype="<f4").copy()
    else:
        raise ValueError(f"{key}: dtype {dtype} not handled")
    return a.reshape(shape)


def _blob_path(out, fmt, layer):
    return os.path.join(out, "blobs", fmt, layer + ".bin")


def _text_tokens(tokenizer, ctx, chunks):
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(TEXT["repo_id"], TEXT["filename"], repo_type=TEXT["repo_type"])
    raw = open(path, "rb").read()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != TEXT["sha256"]:
        raise RuntimeError(f"{TEXT['filename']} sha256 {digest} != pinned {TEXT['sha256']}")
    text = zipfile.ZipFile(io.BytesIO(raw)).read(TEXT["member"]).decode("utf-8")
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    need = ctx * chunks
    if len(ids) < need:
        raise RuntimeError(f"text has {len(ids)} tokens, need {need}")
    return ids[:need], {"zip_sha256": digest, "member": TEXT["member"], "tokens_total": len(ids)}


def _assert_stack(torch, want_shim):
    is_shim = hasattr(torch._C, "_aten_implemented")
    if is_shim != want_shim:
        raise RuntimeError(
            f"this phase needs the {'shim' if want_shim else 'upstream'} torch, got "
            f"{'shim' if is_shim else 'upstream'} from {torch.__file__} (AGENTS.md §15.2)"
        )


# --- metrics (numpy, float64; the judge never runs on the stack it judges) ---


def _log_softmax(x):
    np = _np()
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def token_metrics(ref_logits, cand_logits, targets):
    """Per scored position: KL(ref || cand), both NLLs, top-1 agreement."""
    np = _np()
    lr = _log_softmax(ref_logits)
    lc = _log_softmax(cand_logits)
    pr = np.exp(lr)
    kl = (pr * (lr - lc)).sum(axis=-1)
    rows = np.arange(len(targets))
    return {
        "kl": kl,
        "nll_ref": -lr[rows, targets],
        "nll_cand": -lc[rows, targets],
        "top1_same": (lr.argmax(axis=-1) == lc.argmax(axis=-1)).astype(np.float64),
    }


def summarise(per_token):
    np = _np()
    kl = np.asarray(per_token["kl"])
    d = np.asarray(per_token["nll_cand"]) - np.asarray(per_token["nll_ref"])
    n = kl.size
    return {
        "n_tokens": int(n),
        "kl_mean": float(kl.mean()),
        "kl_mean_se": float(kl.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "kl_median": float(np.median(kl)),
        "kl_p90": float(np.quantile(kl, 0.90)),
        "kl_p99": float(np.quantile(kl, 0.99)),
        "kl_max": float(kl.max()),
        "ppl_ref": float(math.exp(np.mean(per_token["nll_ref"]))),
        "ppl_cand": float(math.exp(np.mean(per_token["nll_cand"]))),
        "ln_ppl_ratio": float(d.mean()),
        "ln_ppl_ratio_se": float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else float("nan"),
        "top1_agreement": float(np.mean(per_token["top1_same"])),
    }


def first_divergence(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


def _scored(ctx):
    """llama-perplexity's scoring window: positions [ctx/2, ctx-1) predict the next token."""
    first = ctx // 2
    return first, ctx - 1 - first


# ======================================================================
# phase: quantise (shim) -- our bytes, and candle's reading of them
# ======================================================================


def phase_quantise(args):
    import torch

    _assert_stack(torch, want_shim=True)
    from transformers import AutoModelForCausalLM
    from torchnative.quant import TorchnativeConfig

    model_dir, rev = _snapshot_dir(MODELS[args.model])
    res = {"model": MODELS[args.model], "revision": rev, "formats": {}}
    for fmt in args.formats:
        _log(f"quantise {fmt}")
        try:
            m = AutoModelForCausalLM.from_pretrained(
                model_dir, dtype=torch.float32, quantization_config=TorchnativeConfig(fmt),
                attn_implementation="eager",
            )
        except ValueError as exc:
            # The plugin refuses a format the model's widths cannot hold
            # (SmolLM2 is 576 wide; no 256-block k-quant fits). That is an
            # answer, recorded by name, not a crash and not a skip.
            res["formats"][fmt] = {"refused": str(exc)}
            continue
        rep = m.torchnative_quantization
        layers = {}
        os.makedirs(os.path.join(args.out, "blobs", fmt), exist_ok=True)
        for name in rep.converted:
            mod = m.get_submodule(name)
            blob = torch._C._quantized_blob(mod.qweight)
            with open(_blob_path(args.out, fmt, name), "wb") as fh:
                fh.write(blob)
            # Candle's reading of the first rows, for the bit check in
            # `weights`. A sample, because the shim has no bulk export and a
            # full `tolist()` of 0.44B weights is not a measurement.
            rows = min(args.sample_rows, mod.out_features)
            deq = torch._C._dequantize(mod.qweight)[:rows].tolist()
            layers[name] = {
                "shape": [mod.out_features, mod.in_features],
                "bytes": len(blob),
                "sha256": hashlib.sha256(blob).hexdigest(),
                "candle_dequant_rows": rows,
                "candle_dequant_f32": struct.pack(f"<{rows * mod.in_features}f",
                                                  *[v for r in deq for v in r]).hex(),
            }
        res["formats"][fmt] = {
            "converted": list(rep.converted),
            "left_dense": [list(s) for s in rep.skipped],
            "modules_to_not_convert": list(rep.modules_to_not_convert),
            "swapped_before_weights": rep.swapped_before_weights,
            "layers": layers,
        }
        del m
    _dump(args.out, "quantise.json", res)


# ======================================================================
# phase: weights (numpy + gguf-py) -- bytes, bits, and the writers' error
# ======================================================================


def phase_weights(args):
    np = _np()
    sys.path.insert(0, HERE)
    import gguf_ref

    gq, prov = gguf_ref.load_gguf_quants(args.gguf_py)
    q = _load(args.out, "quantise.json")
    model_dir, _ = _snapshot_dir(MODELS[args.model])
    index = _safetensors_index(model_dir)
    res = {"gguf_py": prov, "formats": {}}
    for fmt, fq in q["formats"].items():
        if "refused" in fq:
            res["formats"][fmt] = {"refused": fq["refused"]}
            continue
        per = {}
        os.makedirs(os.path.join(args.out, "blobs", fmt + "_llamacpp"), exist_ok=True)
        tot = {"bytes_equal_layers": 0, "bytes_differ": 0, "bytes_total": 0,
               "bits_equal_layers": 0, "layers": 0}
        for name, info in fq["layers"].items():
            _log(f"weights {fmt} {name}")
            out_f, in_f = info["shape"]
            w = _read_f32(index, name + ".weight")
            ours = open(_blob_path(args.out, fmt, name), "rb").read()
            if hashlib.sha256(ours).hexdigest() != info["sha256"]:
                raise RuntimeError(f"{name}: blob on disk is not the one `quantise` wrote")
            if fmt == "q4_0":
                ref = gguf_ref.gguf_quantize(gq, fmt, w)
            elif fmt == "q4_k":
                ref = gguf_ref.quantize_q4_k_llamacpp(w)
            else:
                raise ValueError(f"no reference writer for {fmt}")
            with open(_blob_path(args.out, fmt + "_llamacpp", name), "wb") as fh:
                fh.write(ref)
            ob = np.frombuffer(ours, dtype=np.uint8)
            rb = np.frombuffer(ref, dtype=np.uint8)
            ndiff = int(np.sum(ob != rb))
            # Reader: candle's sampled rows against gguf-py's reading of the same bytes.
            rows = info["candle_dequant_rows"]
            nrow_bytes = (in_f // gguf_ref.BLOCK_SIZE[fmt]) * gguf_ref.TYPE_SIZE[fmt]
            ref_deq = gguf_ref.gguf_dequantize(gq, fmt, ours[: rows * nrow_bytes], rows * in_f)
            bits_equal = ref_deq.tobytes() == bytes.fromhex(info["candle_dequant_f32"])
            # Writers' reconstruction error, both read by gguf-py.
            w_ours = gguf_ref.gguf_dequantize(gq, fmt, ours, out_f * in_f)
            w_ref = gguf_ref.gguf_dequantize(gq, fmt, ref, out_f * in_f)
            wf = w.reshape(-1).astype(np.float64)
            norm = math.sqrt(float(np.mean(wf * wf)))
            per[name] = {
                "bytes_differing": ndiff,
                "bytes": int(ob.size),
                "reader_bits_equal_sampled_rows": bool(bits_equal),
                "rel_rms_ours": math.sqrt(float(np.mean((w_ours - wf) ** 2))) / norm,
                "rel_rms_llamacpp_writer": math.sqrt(float(np.mean((w_ref - wf) ** 2))) / norm,
            }
            tot["layers"] += 1
            tot["bytes_total"] += int(ob.size)
            tot["bytes_differ"] += ndiff
            tot["bytes_equal_layers"] += int(ndiff == 0)
            tot["bits_equal_layers"] += int(bits_equal)
        r_ours = [v["rel_rms_ours"] for v in per.values()]
        r_ref = [v["rel_rms_llamacpp_writer"] for v in per.values()]
        tot["rel_rms_ours_mean"] = float(np.mean(r_ours))
        tot["rel_rms_llamacpp_writer_mean"] = float(np.mean(r_ref))
        tot["writer_is_llamacpp_writer"] = tot["bytes_differ"] == 0
        res["formats"][fmt] = {"summary": tot, "layers": per}
    _dump(args.out, "weights.json", res)


# ======================================================================
# phase: reference (upstream torch) -- R0, R1, R2, R3 and the f64 oracle
# ======================================================================


def _gguf_tensors(path, hf_layers, model_type, n_head, n_kv, gq):
    """Dequantised float32 weights from a real llama.cpp GGUF, keyed by HF name.

    Name mapping is llama.cpp's `blk.N.*` convention. LLaMA-arch converters
    permute q/k rows for RoPE (`LlamaModel.permute` in llama.cpp's
    conversion/llama.py); Qwen2/3 do not. The permutation is undone here and
    then *checked*: both orientations are scored against the dense weight and
    the one used must be the closer, or the load is refused by name.
    """
    np = _np()
    import importlib

    reader_mod = importlib.import_module("gguf.gguf_reader")
    r = reader_mod.GGUFReader(path)
    by_name = {t.name: t for t in r.tensors}
    names = {
        "self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
        "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
        "mlp.down_proj": "ffn_down",
    }
    out, info = {}, {"types": {}, "unused_in_file": []}
    used = set()
    for hf in hf_layers:
        parts = hf.split(".")
        i, tail = parts[2], ".".join(parts[3:])
        gname = f"blk.{i}.{names[tail]}.weight"
        t = by_name[gname]
        used.add(gname)
        w = gq.dequantize(t.data, t.tensor_type).astype(np.float32)
        w = w.reshape(int(t.shape[1]), int(t.shape[0]))
        if model_type == "llama" and tail in ("self_attn.q_proj", "self_attn.k_proj"):
            h = n_head if tail.endswith("q_proj") else n_kv
            w = w.reshape(h, w.shape[0] // h // 2, 2, w.shape[1]).swapaxes(1, 2).reshape(w.shape)
        out[hf] = w
        info["types"][hf] = t.tensor_type.name
    info["unused_in_file"] = sorted((set(by_name) - used))
    return out, info


def phase_reference(args):
    np = _np()
    import torch

    _assert_stack(torch, want_shim=False)
    torch.set_num_threads(os.cpu_count() or 1)
    sys.path.insert(0, HERE)
    sys.path.insert(0, PYTESTS)
    import gguf_ref
    from agree_sweep import diff_stats
    from transformers import AutoModelForCausalLM, AutoTokenizer

    gq, prov = gguf_ref.load_gguf_quants(args.gguf_py)
    q = _load(args.out, "quantise.json")
    model_dir, rev = _snapshot_dir(MODELS[args.model])
    index = _safetensors_index(model_dir)
    tok = AutoTokenizer.from_pretrained(model_dir)
    ids, text_info = _text_tokens(tok, args.ctx, args.chunks)
    prompts = [tok(p, add_special_tokens=False)["input_ids"] for p in PROMPTS]
    _dump(args.out, "tokens.json", {"ids": ids, "prompts": prompts, "text": text_info})

    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager")
    model.eval()
    cfg = model.config
    first, nscored = _scored(args.ctx)
    V = cfg.vocab_size
    ref_dir = os.path.join(args.out, "ref")
    os.makedirs(ref_dir, exist_ok=True)

    def forward_all(store=None):
        logits = []
        with torch.no_grad():
            for c in range(args.chunks):
                x = torch.tensor([ids[c * args.ctx : (c + 1) * args.ctx]])
                lg = model(input_ids=x, use_cache=False).logits[0, first : args.ctx - 1].float().numpy()
                if store is not None:
                    store[c] = lg
                logits.append(lg)
        return logits

    def greedy():
        outs = []
        with torch.no_grad():
            for p in prompts:
                g = model.generate(torch.tensor([p]), max_new_tokens=args.gen_tokens, do_sample=False,
                                   num_beams=1, use_cache=True)
                outs.append(g[0, len(p):].tolist())
        return outs

    def targets(c):
        return np.array(ids[c * args.ctx + first + 1 : (c + 1) * args.ctx])

    def set_weights(mapping):
        with torch.no_grad():
            for name, w in mapping.items():
                model.get_submodule(name).weight.copy_(torch.from_numpy(np.ascontiguousarray(w)))

    def oracle_rel(window_logits):
        """Upstream's own float32-vs-float64 distance on window 0 (AGREE.md §2)."""
        if not args.oracle_f64:
            return None
        model.double()
        with torch.no_grad():
            x = torch.tensor([ids[: args.ctx]])
            lg64 = model(input_ids=x, use_cache=False).logits[0, first : args.ctx - 1].numpy()
        model.float()
        return diff_stats(window_logits, lg64)["rel"]

    res = {"model": MODELS[args.model], "revision": rev, "torch": torch.__version__,
           "gguf_py": prov, "configs": {}}

    def mm(name, shape):
        return np.lib.format.open_memmap(os.path.join(ref_dir, name + ".npy"), mode="w+",
                                         dtype=np.float32, shape=shape)

    # R0
    _log("reference R0 dense")
    r0 = mm("R0", (args.chunks, nscored, V))
    r0_logits = forward_all(r0)
    r0.flush()
    res["configs"]["R0"] = {"oracle_rel_f32_vs_f64": oracle_rel(r0_logits[0]), "greedy": greedy()}
    original = {}

    for fmt, fq in q["formats"].items():
        if "refused" in fq:
            continue
        conv = fq["converted"]
        for n in conv:
            if n not in original:
                original[n] = _read_f32(index, n + ".weight")

        def from_blobs(kind):
            return {
                n: gguf_ref.gguf_dequantize(
                    gq, fmt, open(_blob_path(args.out, kind, n), "rb").read(), int(np.prod(original[n].shape))
                ).reshape(original[n].shape)
                for n in conv
            }

        # R1: llama.cpp's reader on our bytes.
        _log(f"reference R1 {fmt}")
        set_weights(from_blobs(fmt))
        r1 = mm(f"R1_{fmt}", (args.chunks, nscored, V))
        r1_logits = forward_all(r1)
        r1.flush()
        per = [token_metrics(r0_logits[c], r1_logits[c], targets(c)) for c in range(args.chunks)]
        res["configs"][f"R1_{fmt}"] = {
            "vs_R0": summarise({k: np.concatenate([p[k] for p in per]) for k in per[0]}),
            "oracle_rel_f32_vs_f64": oracle_rel(r1_logits[0]),
            "greedy": greedy(),
        }
        np.savez(os.path.join(ref_dir, f"R1_{fmt}_vs_R0_tokens.npz"),
                 **{k: np.concatenate([p[k] for p in per]) for k in per[0]})
        del r1_logits

        # R2: llama.cpp's writer.
        _log(f"reference R2 {fmt}")
        set_weights(from_blobs(fmt + "_llamacpp"))
        r2_logits = forward_all()
        per = [token_metrics(r0_logits[c], r2_logits[c], targets(c)) for c in range(args.chunks)]
        res["configs"][f"R2_{fmt}"] = {
            "vs_R0": summarise({k: np.concatenate([p[k] for p in per]) for k in per[0]}),
            "greedy": greedy(),
            "writer": "gguf-py Q4_0.quantize" if fmt == "q4_0" else "gguf_ref.quantize_q4_k_llamacpp (transcription)",
        }
        np.savez(os.path.join(ref_dir, f"R2_{fmt}_vs_R0_tokens.npz"),
                 **{k: np.concatenate([p[k] for p in per]) for k in per[0]})
        del r2_logits
        set_weights(original)

    if args.gguf:
        _log("reference R3 from GGUF")
        conv = sorted({n for fq in q["formats"].values() if "refused" not in fq for n in fq["converted"]})
        for n in conv:
            if n not in original:
                original[n] = _read_f32(index, n + ".weight")
        w3, info3 = _gguf_tensors(args.gguf, conv, cfg.model_type, cfg.num_attention_heads,
                                  getattr(cfg, "num_key_value_heads", cfg.num_attention_heads), gq)
        # Orientation check for the q/k permutation: the dense weight must be
        # nearer the orientation used than the other one.
        for n in conv:
            if n.endswith(("q_proj", "k_proj")) and cfg.model_type == "llama":
                h = cfg.num_attention_heads if n.endswith("q_proj") else cfg.num_key_value_heads
                w = w3[n]
                other = w.reshape(h, 2, w.shape[0] // h // 2, w.shape[1]).swapaxes(1, 2).reshape(w.shape)
                if np.linalg.norm(w - original[n]) >= np.linalg.norm(other - original[n]):
                    raise RuntimeError(f"R3: {n} is closer to the dense weight with llama.cpp's q/k "
                                       "permutation left in than with it undone; the file's layout is "
                                       "not the one assumed, so its numbers would be about the wrong rows")
        info3["rel_rms_vs_dense"] = {
            n: float(np.sqrt(np.mean((w3[n] - original[n]) ** 2)) / np.sqrt(np.mean(original[n] ** 2)))
            for n in conv
        }
        set_weights(w3)
        r3_logits = forward_all()
        per = [token_metrics(r0_logits[c], r3_logits[c], targets(c)) for c in range(args.chunks)]
        res["configs"]["R3_gguf"] = {
            "file": os.path.abspath(args.gguf),
            "file_sha256": hashlib.sha256(open(args.gguf, "rb").read()).hexdigest(),
            "vs_R0": summarise({k: np.concatenate([p[k] for p in per]) for k in per[0]}),
            "greedy": greedy(),
            "gguf": info3,
        }
        np.savez(os.path.join(ref_dir, "R3_gguf_vs_R0_tokens.npz"),
                 **{k: np.concatenate([p[k] for p in per]) for k in per[0]})
        set_weights(original)

    _dump(args.out, "reference.json", res)


# ======================================================================
# phase: candidate (shim) -- S0, S1, S2, and the matmul on real activations
# ======================================================================


def _shim_rows_to_numpy(t, step=8):
    """A shim tensor `(rows, cols)` to float32 numpy, in slices.

    The shim has no `numpy()` and no bulk byte export, and one `tolist()` of a
    255 x 151936 logit block is ~39M Python floats. Slicing keeps the peak at
    `step` rows; the values are exact (`tolist` of float32 is lossless).
    """
    np = _np()
    out = np.empty(tuple(t.shape), dtype=np.float32)
    for i in range(0, t.shape[0], step):
        out[i : i + step] = np.asarray(t[i : i + step].tolist(), dtype=np.float32)
    return out


def phase_candidate(args):
    np = _np()
    import torch

    _assert_stack(torch, want_shim=True)
    sys.path.insert(0, HERE)
    sys.path.insert(0, PYTESTS)
    import gguf_ref
    from agree_sweep import F32_EPS, diff_stats, verdict
    from transformers import AutoModelForCausalLM
    from torchnative.quant import QuantizedLinear, TorchnativeConfig

    gq, prov = gguf_ref.load_gguf_quants(args.gguf_py)
    q = _load(args.out, "quantise.json")
    ref = _load(args.out, "reference.json")
    toks = _load(args.out, "tokens.json")
    ids, prompts = toks["ids"], toks["prompts"]
    model_dir, _ = _snapshot_dir(MODELS[args.model])
    first, nscored = _scored(args.ctx)
    ref_dir = os.path.join(args.out, "ref")
    R = {name: np.load(os.path.join(ref_dir, name + ".npy"), mmap_mode="r")
         for name in ["R0"] + [f"R1_{f}" for f, fq in q["formats"].items() if "refused" not in fq]}
    res = {"configs": {}, "linear": {}}

    def targets(c):
        return np.array(ids[c * args.ctx + first + 1 : (c + 1) * args.ctx])

    def run(model, label, against):
        model.eval()
        per = {k: [] for k in against}
        win0 = None
        with torch.no_grad():
            for c in range(args.chunks):
                x = torch.tensor([ids[c * args.ctx : (c + 1) * args.ctx]])
                lg = _shim_rows_to_numpy(model(input_ids=x, use_cache=False).logits[0, first : args.ctx - 1])
                if c == 0:
                    win0 = lg
                for k in against:
                    per[k].append(token_metrics(np.asarray(R[k][c]), lg, targets(c)))
            outs = []
            for p in prompts:
                g = model.generate(torch.tensor([p]), max_new_tokens=args.gen_tokens, do_sample=False,
                                   num_beams=1, use_cache=True)
                outs.append(g[0, len(p):].tolist())
        entry = {"greedy": outs}
        for k in against:
            cat = {m: np.concatenate([p[m] for p in per[k]]) for m in per[k][0]}
            entry["vs_" + k] = summarise(cat)
            np.savez(os.path.join(ref_dir, f"{label}_vs_{k}_tokens.npz"), **cat)
        return entry, win0

    def agree(win0, refname, oracle):
        """AGREE.md §2's rule, with the tolerance re-derived for this model."""
        d = diff_stats(win0, np.asarray(R[refname][0]))
        tol = max(oracle, 8 * F32_EPS) if oracle is not None else None
        v = verdict(d["rel"], d["scale"], oracle, tol) if tol is not None else "no_oracle"
        return {"rel": d["rel"], "abs": d["abs"], "scale": d["scale"], "oracle_rel": oracle,
                "tol": tol, "verdict": v}

    # S0: the dense floor.
    _log("candidate S0 dense")
    dense = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32, attn_implementation="eager")
    e, win0 = run(dense, "S0", ["R0"])
    e["agree_vs_R0"] = agree(win0, "R0", ref["configs"]["R0"]["oracle_rel_f32_vs_f64"])
    res["configs"]["S0"] = e

    for fmt, fq in q["formats"].items():
        if "refused" in fq:
            res["configs"][f"S2_{fmt}"] = {"refused": fq["refused"]}
            continue
        # S1: candle's reader on our bytes, dense compute.
        _log(f"candidate S1 {fmt}")
        keep = {}
        with torch.no_grad():
            for n in fq["converted"]:
                mod = dense.get_submodule(n)
                blob = open(_blob_path(args.out, fmt, n), "rb").read()
                w = torch._C._dequantize(torch._C._quantized_from_blob(blob, list(mod.weight.shape), fmt))
                keep[n] = mod.weight
                mod.weight = torch.nn.Parameter(w, requires_grad=False)
        e, win0 = run(dense, f"S1_{fmt}", ["R0", f"R1_{fmt}"])
        e["agree_vs_R1"] = agree(win0, f"R1_{fmt}", ref["configs"][f"R1_{fmt}"]["oracle_rel_f32_vs_f64"])
        res["configs"][f"S1_{fmt}"] = e
        with torch.no_grad():
            for n, p in keep.items():
                dense.get_submodule(n).weight = p

    del dense

    for fmt, fq in q["formats"].items():
        if "refused" in fq:
            continue
        _log(f"candidate S2 {fmt}")
        m = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=torch.float32, quantization_config=TorchnativeConfig(fmt),
            attn_implementation="eager",
        )
        conv = list(m.torchnative_quantization.converted)
        if conv != fq["converted"]:
            raise RuntimeError(f"S2 {fmt}: converted set differs from the `quantise` phase's")
        for n in conv:
            if hashlib.sha256(torch._C._quantized_blob(m.get_submodule(n).qweight)).hexdigest() != \
                    fq["layers"][n]["sha256"]:
                raise RuntimeError(f"S2 {fmt}: {n} quantised to different bytes than R1 was built from")

        # The matmul on real activations: a first, a middle and a last layer,
        # attention and MLP, inputs captured from chunk 0.
        L = m.config.num_hidden_layers
        picks = [f"model.layers.0.self_attn.q_proj", f"model.layers.{L // 2}.mlp.down_proj",
                 f"model.layers.{L - 1}.mlp.gate_proj", f"model.layers.{L - 1}.self_attn.o_proj"]
        picks = [p for p in picks if p in conv]
        captured = {}

        def hook(name):
            def fn(mod, inputs, output):
                if name in captured:
                    return
                r = min(args.linear_rows, inputs[0].shape[1])
                captured[name] = (_shim_rows_to_numpy(inputs[0][0, :r]), _shim_rows_to_numpy(output[0, :r]),
                                  None if mod.bias is None else _shim_rows_to_numpy(mod.bias[None])[0])
            return fn

        handles = [m.get_submodule(p).register_forward_hook(hook(p)) for p in picks]
        e, _ = run(m, f"S2_{fmt}", ["R0", f"R1_{fmt}"])
        for h in handles:
            h.remove()
        res["configs"][f"S2_{fmt}"] = e

        for name, (x, y, bias) in captured.items():
            mod = m.get_submodule(name)
            assert isinstance(mod, QuantizedLinear)
            out_f, in_f = mod.out_features, mod.in_features
            blob = open(_blob_path(args.out, fmt, name), "rb").read()
            bs = gguf_ref.BLOCK_SIZE["q8_0" if fmt == "q4_0" else "q4_k"]
            tie = np.array([
                (gguf_ref.q8_0_ties(row) if fmt == "q4_0" else gguf_ref.q8_k_ties(row))
                + gguf_ref.signed_max_ties(row, bs)
                for row in x
            ])
            keep_rows = np.where(tie == 0)[0]
            xs, ys = x[keep_rows], y[keep_rows].astype(np.float64)
            rows = len(keep_rows)
            if fmt == "q4_0":
                xq = gguf_ref.gguf_quantize(gq, "q8_0", xs)
                exact, abs_sum, ref_err = gguf_ref.exact_linear_q4_0(xq, blob, rows, out_f, in_f)
            else:
                xd, xqk = gguf_ref.quantize_q8_k_ref(xs)
                exact, abs_sum, ref_err = gguf_ref.exact_linear_q4_k(
                    xd.reshape(rows, -1), xqk.reshape(rows, -1, 256), blob, rows, out_f, in_f)
            bound = gguf_ref.linear_bound(fmt, in_f, abs_sum, ref_err)
            if bias is not None:
                exact = exact + bias.astype(np.float64)
                bound = bound + gguf_ref.U32 * np.abs(exact)
            wdq = gguf_ref.gguf_dequantize(gq, fmt, blob, out_f * in_f).reshape(out_f, in_f).astype(np.float64)
            alt = xs.astype(np.float64) @ wdq.T + (0 if bias is None else bias.astype(np.float64))
            ratio = np.abs(ys - exact) / bound
            res["linear"][f"{fmt}:{name}"] = {
                "rows_checked": rows, "rows_refused_for_ties": int(np.sum(tie > 0)),
                "K": gguf_ref.accumulation_k(fmt, in_f),
                "max_ratio_to_bound": float(ratio.max()) if rows else None,
                # None, not False, when every row had a tie: nothing was graded.
                "within_bound": bool(ratio.max() <= 1.0) if rows else None,
                "no_activation_quant_min_ratio": float((np.abs(alt - exact) / bound).max(axis=1).min()) if rows else None,
            }
        del m

    _dump(args.out, "candidate.json", res)


# ======================================================================
# phase: report
# ======================================================================


def _paired(out, a, b):
    """mean(KL_a - KL_b) per token and its standard error, same tokens."""
    np = _np()
    pa = os.path.join(out, "ref", a + "_vs_R0_tokens.npz")
    pb = os.path.join(out, "ref", b + "_vs_R0_tokens.npz")
    if not (os.path.exists(pa) and os.path.exists(pb)):
        return None
    ka, kb = np.load(pa)["kl"], np.load(pb)["kl"]
    d = ka - kb
    se = float(d.std(ddof=1) / math.sqrt(d.size))
    return {"mean_diff": float(d.mean()), "se": se, "z": float(d.mean() / se) if se > 0 else float("nan")}


def phase_report(args):
    q = _load(args.out, "quantise.json")
    w = _load(args.out, "weights.json")
    r = _load(args.out, "reference.json")
    c = _load(args.out, "candidate.json")
    env = _load(args.out, "env.json")
    toks = _load(args.out, "tokens.json")
    results = {"env": env, "quantise": q, "weights": w, "reference": r, "candidate": c, "paired": {}}
    md = [f"# Q4 quality -- {q['model']} @ {q['revision'][:12]}", ""]
    md.append(f"text: wikitext-2-raw test, zip sha256 `{toks['text']['zip_sha256'][:16]}...`, "
              f"{args.chunks} x {args.ctx} tokens, scored {_scored(args.ctx)[1]} per chunk "
              f"(llama-perplexity's window). gguf-py quants.py sha256 `{w['gguf_py']['sha256'][:16]}...`.")
    md.append(f"load at start {env['load_start']}, shim commit `{env['commit']}`.")
    md.append("")
    md.append("## Correctness (graded)")
    md.append("")
    md.append("| check | format | result | grade |")
    md.append("|---|---|---|---|")
    for fmt, fw in w["formats"].items():
        if "refused" in fw:
            md.append(f"| load | {fmt} | refused: {fw['refused'].splitlines()[0][:90]} | refusal |")
            continue
        s = fw["summary"]
        md.append(f"| writer bytes == llama.cpp writer | {fmt} | {s['bytes_equal_layers']}/{s['layers']} layers, "
                  f"{s['bytes_differ']}/{s['bytes_total']} bytes differ | exact"
                  + (" (q4_k: different writer by design; reported, not graded)" if fmt == "q4_k" else "") + " |")
        md.append(f"| reader bits == gguf-py | {fmt} | {s['bits_equal_layers']}/{s['layers']} layers (sampled rows) | exact |")
    for k, v in c["linear"].items():
        if v["within_bound"] is None:
            md.append(f"| matmul vs exact | {k} | every captured row had a rounding tie | not graded |")
            continue
        md.append(f"| matmul vs exact, K={v['K']} | {k} | max {v['max_ratio_to_bound']:.3g}x bound; "
                  f"no-act-quant alt >= {v['no_activation_quant_min_ratio']:.3g}x; "
                  f"{v['rows_refused_for_ties']} tie rows refused | "
                  f"{'within derived bound' if v['within_bound'] else 'OUTSIDE BOUND'} |")
    for name, e in c["configs"].items():
        for key in ("agree_vs_R0", "agree_vs_R1"):
            if key in e:
                a = e[key]
                md.append(f"| {name} {key.replace('agree_', '')} element-wise | {name} | rel {a['rel']:.3g}, "
                          f"oracle {a['oracle_rel']} | {a['verdict']} |")
    md.append("")
    md.append("## Quality (measured, not thresholded)")
    md.append("")
    md.append("| config | vs | KL mean ± se | KL p99 | KL max | PPL ratio | top-1 | greedy first divergence vs R0 |")
    md.append("|---|---|---|---|---|---|---|---|")
    g0 = r["configs"]["R0"]["greedy"]

    def row(name, e, vs):
        s = e.get("vs_" + vs)
        if not s:
            return
        div = [first_divergence(a, b) for a, b in zip(g0, e["greedy"])]
        md.append(f"| {name} | {vs} | {s['kl_mean']:.4g} ± {s['kl_mean_se']:.2g} | {s['kl_p99']:.4g} | "
                  f"{s['kl_max']:.4g} | {math.exp(s['ln_ppl_ratio']):.4f} | {s['top1_agreement']:.4f} | "
                  f"{div} / {args.gen_tokens} |")

    for name, e in r["configs"].items():
        row(name, e, "R0")
    for name, e in c["configs"].items():
        if "refused" in e:
            md.append(f"| {name} | -- | refused | | | | | |")
            continue
        for vs in ("R0",) + tuple(k[3:] for k in e if k.startswith("vs_R1")):
            row(name, e, vs)
    md.append("")
    md.append("## Ours against llama.cpp on the same tokens (paired, KL vs R0)")
    md.append("")
    md.append("| ours | llama.cpp | mean(KL_ours - KL_ref) | se | z |")
    md.append("|---|---|---|---|---|")
    for fmt in q["formats"]:
        for refname in (f"R2_{fmt}", "R3_gguf"):
            p = _paired(args.out, f"S2_{fmt}", refname)
            if p:
                results["paired"][f"S2_{fmt}-{refname}"] = p
                md.append(f"| S2_{fmt} | {refname} | {p['mean_diff']:.4g} | {p['se']:.2g} | {p['z']:.2f} |")
    md.append("")
    md.append("Grades: *exact* and *within derived bound* and *agree* are pass/fail with derived lines "
              "(tools/q4quality/measure.py docstring). Everything under Quality is *measured*.")
    _dump(args.out, "results.json", results)
    with open(os.path.join(args.out, "results.md"), "w") as fh:
        fh.write("\n".join(md) + "\n")
    print("\n".join(md))


# ======================================================================
# orchestration
# ======================================================================


def _child_env(args, shim):
    env = dict(os.environ)
    for var in ("CANDLE_DEQUANTIZE_ALL", "CANDLE_DEQUANTIZE_ALL_F16"):
        if env.get(var, "") not in ("", "0"):
            # It turns every quantised matmul into a dense one over dequantised
            # weights -- the measurement would then be of a different thing.
            raise SystemExit(f"{var} is set; refusing to measure (it bypasses the Q4 matmul)")
    scratch = os.path.join(os.path.abspath(args.out), "tmp")
    os.makedirs(scratch, exist_ok=True)
    env.update({
        "HF_HOME": HF_HOME,  # never ~/.cache
        "HF_HUB_DISABLE_TELEMETRY": "1",
        "TMPDIR": scratch,  # AGENTS.md §2: nothing outside this repository
        "XDG_CACHE_HOME": scratch,
        "TORCH_HOME": scratch,
        "PYTHONDONTWRITEBYTECODE": "1",
    })
    if args.offline:
        env["HF_HUB_OFFLINE"] = "1"
    if shim:
        env["PYTHONPATH"] = VENDOR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1, as test_shim.py sets it
    else:
        env.pop("PYTHONPATH", None)
    return env


def _git_commit():
    p = subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True)
    dirty = subprocess.run(["git", "-C", REPO, "status", "--porcelain"], capture_output=True, text=True)
    return p.stdout.strip() + ("+dirty" if dirty.stdout.strip() else "")


def run_all(args):
    out = os.path.abspath(args.out)
    if not out.startswith(REPO + os.sep):
        raise SystemExit(f"--out {out} is outside this repository (AGENTS.md §2)")
    if "target" not in os.path.relpath(out, REPO).split(os.sep):
        raise SystemExit(f"--out {out} is not under a `target/` directory; test_docrefs.py would "
                         "read the gigabytes of logits written there as text")
    os.makedirs(os.path.join(out, "logs"), exist_ok=True)
    load = os.getloadavg()
    cores = os.cpu_count() or 1
    if load[0] > cores and not args.allow_loaded:
        raise SystemExit(f"load {load[0]:.1f} on {cores} cores; measurements run alone (AGENTS.md §16). "
                         "--allow-loaded records it and proceeds.")
    if not os.path.exists(os.path.join(VENDOR, "torch", "__init__.py")):
        raise SystemExit(f"{VENDOR}/torch is not built (AGENTS.md §14.1): "
                         "sh vendor/vendor_torch.sh && bash vendor/install_shim.sh")
    _dump(out, "env.json", {
        "load_start": list(load), "cores": cores, "commit": _git_commit(), "platform": platform.platform(),
        "python": args.python, "argv": sys.argv, "formats": args.formats, "ctx": args.ctx, "chunks": args.chunks,
    })
    common = ["--model", args.model, "--formats", ",".join(args.formats), "--out", out,
              "--ctx", str(args.ctx), "--chunks", str(args.chunks), "--gen-tokens", str(args.gen_tokens),
              "--linear-rows", str(args.linear_rows), "--sample-rows", str(args.sample_rows)]
    if args.gguf_py:
        common += ["--gguf-py", os.path.abspath(args.gguf_py)]
    if args.gguf:
        common += ["--gguf", os.path.abspath(args.gguf)]
    if not args.oracle_f64:
        common += ["--no-oracle-f64"]
    stack = {"quantise": True, "weights": False, "reference": False, "candidate": True, "report": False}
    for phase in args.phases:
        log = os.path.join(out, "logs", phase + ".log")
        _log(f"phase {phase} -> {log}")
        with open(log, "w") as fh:
            proc = subprocess.run([args.python, os.path.abspath(__file__), phase] + common,
                                  stdout=fh, stderr=subprocess.STDOUT, env=_child_env(args, stack[phase]))
        if proc.returncode != 0:
            raise SystemExit(f"phase {phase} exited {proc.returncode}; see {log}")
    env = _load(out, "env.json")
    env["load_end"] = list(os.getloadavg())
    _dump(out, "env.json", env)
    print(open(os.path.join(out, "results.md")).read() if "report" in args.phases else "done")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Q4 quality against upstream and llama.cpp (issue #16)")
    ap.add_argument("phase", choices=("run",) + PHASES)
    ap.add_argument("--model", choices=sorted(MODELS), default="qwen3-0.6b")
    ap.add_argument("--formats", default="q4_0,q4_k")
    ap.add_argument("--out", required=True)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--chunks", type=int, default=4)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--linear-rows", type=int, default=64)
    ap.add_argument("--sample-rows", type=int, default=4)
    ap.add_argument("--gguf-py", default=None)
    ap.add_argument("--gguf", default=None, help="a llama.cpp-written GGUF of the same model (R3)")
    ap.add_argument("--no-oracle-f64", dest="oracle_f64", action="store_false")
    ap.add_argument("--python", default=DEFAULT_PY)
    ap.add_argument("--phases", default=",".join(PHASES))
    ap.add_argument("--allow-loaded", action="store_true")
    ap.add_argument("--offline", action="store_true", help="HF_HUB_OFFLINE=1 (text zip must be cached)")
    args = ap.parse_args(argv)
    args.formats = [f for f in args.formats.split(",") if f]
    for f in args.formats:
        if f not in ("q4_0", "q4_k"):
            raise SystemExit(f"--formats: {f} is not a Q4 format this harness has a reference for")
    if args.phase == "run":
        args.phases = [p for p in args.phases.split(",") if p]
        return run_all(args)
    os.makedirs(args.out, exist_ok=True)
    {"quantise": phase_quantise, "weights": phase_weights, "reference": phase_reference,
     "candidate": phase_candidate, "report": phase_report}[args.phase](args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
