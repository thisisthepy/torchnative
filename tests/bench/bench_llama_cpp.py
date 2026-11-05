"""Benchmark and profile torchnative against llama.cpp.

Capabilities and comparisons:
  * Models: SmolLM2-135M and Qwen3-0.6B.
  * Quantization formats: Q8_0 and Q4_0 (plus optional dense float32 baseline).
  * Backends: torchnative (CPU, MPS) and llama.cpp (llama-bench, llama-completion).
  * Prefill tok/s and TTFT (time to first token in ms) at prompt lengths 128 and 512.
  * Decode tok/s over 128 generated tokens (and TTFT for 5-token prompt).
  * Token-for-token and text greedy verification on "The capital of France is".
  * Op-level profiling (--profile) with Metal counter tracking on MPS.

This file lives inside tests/bench/ in torchnative per AGENTS.md rules.
No U+2014 em-dashes are used anywhere in this file.
"""

import argparse
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple


def _stats(samples: List[float]) -> Dict[str, float]:
    if not samples:
        return {"median": 0.0, "mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    med = float(statistics.median(samples))
    mean = float(statistics.mean(samples))
    std = float(statistics.stdev(samples)) if len(samples) > 1 else 0.0
    return {
        "median": med,
        "mean": mean,
        "std": std,
        "min": float(min(samples)),
        "max": float(max(samples)),
    }

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = str(
    next(
        p
        for p in __import__("pathlib").Path(_HERE).parents
        if (p / "AGENTS.md").is_file()
    )
)
_SHIM_PATH = os.path.join(_ROOT, "torchnative", "python")


def _resolve_caches() -> str:
    env = os.environ.get("TORCHNATIVE_CACHES")
    if env and os.path.isdir(env):
        return env
    here = os.path.join(_ROOT, ".caches")
    if os.path.isdir(here):
        return here
    parts = _ROOT.split(os.sep)
    if ".worktrees" in parts:
        base = os.sep.join(parts[: parts.index(".worktrees")])
        cand = os.path.join(base, ".caches")
        if os.path.isdir(cand):
            return cand
    return here


_CACHES = _resolve_caches()
_DEFAULT_HF_HOME = os.path.join(_CACHES, "hf-home")
_DEFAULT_GGUF_DIR = os.path.join(_CACHES, "gguf")
_DEFAULT_GGUF_VENV = os.path.join(_CACHES, "gguf-venv")
_DEFAULT_LLAMA_BENCH = os.path.join(
    _CACHES, "llama.cpp", "build", "bin", "llama-bench"
)
_DEFAULT_LLAMA_COMPLETION = os.path.join(
    _CACHES, "llama.cpp", "build", "bin", "llama-completion"
)

SMOLLM2_SNAP = os.path.join(
    _DEFAULT_HF_HOME,
    "hub",
    "models--HuggingFaceTB--SmolLM2-135M",
    "snapshots",
    "93efa2f097d58c2a74874c7e644dbc9b0cee75a2",
)
QWEN3_SNAP = os.path.join(
    _DEFAULT_HF_HOME,
    "hub",
    "models--Qwen--Qwen3-0.6B",
    "snapshots",
    "c1899de289a04d12100db370d81485cdf75e47ca",
)

MODEL_CONFIGS: Dict[str, Dict[str, Any]] = {
    "smollm2-135m": {
        "display_name": "SmolLM2-135M",
        "hf_snapshot": SMOLLM2_SNAP,
        "arch": "llama",
        "vocab_size": 49152,
        "gguf_files": {
            "q8_0": "SmolLM2-135M-Q8_0.gguf",
            "q4_0": "SmolLM2-135M-Q4_0.gguf",
        },
    },
    "qwen3-0.6b": {
        "display_name": "Qwen3-0.6B",
        "hf_snapshot": QWEN3_SNAP,
        "arch": "qwen3",
        "vocab_size": 151936,
        "gguf_files": {
            "q8_0": "Qwen3-0.6B-Q8_0.gguf",
            "q4_0": "Qwen3-0.6B-Q4_0.gguf",
        },
    },
}

VERIFY_PROMPT = "The capital of France is"
PREFILL_LENGTHS = [128, 512]
DECODE_STEPS = 128
PROFILE_DECODE_STEPS = 32

_METAL_COUNTER_NAMES = (
    "compute_encoders",
    "blit_encoders",
    "host_uploads",
    "host_downloads",
    "host_download_bytes",
)


def loadavg() -> Tuple[float, float, float]:
    return os.getloadavg()


def find_gguf_site(gguf_venv_path: str) -> Optional[str]:
    pattern = os.path.join(
        gguf_venv_path, "lib", "python3*", "site-packages"
    )
    matches = glob.glob(pattern)
    return matches[0] if matches else None


# --------------------------------------------------------------------------
# Op-level Profiler Scope (torch._C._aten_dispatch & _C._quantized_linear)
# --------------------------------------------------------------------------


class ProfileScope:
    """Wraps torch._C._aten_dispatch and torch._C._quantized_linear to record op stats."""

    def __init__(self, torch_mod: Any, device: str):
        self.torch = torch_mod
        self.device = device
        self.records: Dict[str, Dict[str, Any]] = {}
        self.orig_aten_dispatch = None
        self.orig_quantized_linear = None
        self.has_metal = (
            device == "mps"
            and hasattr(torch_mod._C, "_metal_counters")
            and torch_mod._C._metal_counters().get("built", False)
        )

    def _get_metal(self) -> Dict[str, int]:
        if self.has_metal:
            return self.torch._C._metal_counters()
        return {}

    def __enter__(self):
        from torch.utils._python_dispatch import TorchDispatchMode

        c = self.torch._C
        self.orig_quantized_linear = getattr(c, "_quantized_linear", None)
        has_metal = self.has_metal
        records = self.records
        orig_quant = self.orig_quantized_linear

        class _OpProfilerMode(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if kwargs is None:
                    kwargs = {}
                op = str(func)
                if has_metal:
                    c0 = c._metal_counters()
                t0 = time.perf_counter()
                res = func(*args, **kwargs)
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0

                rec = records.setdefault(
                    op,
                    {
                        "count": 0,
                        "total_ms": 0.0,
                        "compute_encoders": 0,
                        "blit_encoders": 0,
                        "host_uploads": 0,
                        "host_downloads": 0,
                        "host_download_bytes": 0,
                    },
                )
                rec["count"] += 1
                rec["total_ms"] += elapsed_ms
                if has_metal:
                    c1 = c._metal_counters()
                    rec["compute_encoders"] += max(
                        0,
                        c1.get("compute_encoders", 0)
                        - c0.get("compute_encoders", 0),
                    )
                    rec["blit_encoders"] += max(
                        0, c1.get("blit_encoders", 0) - c0.get("blit_encoders", 0)
                    )
                    rec["host_uploads"] += max(
                        0, c1.get("host_uploads", 0) - c0.get("host_uploads", 0)
                    )
                    rec["host_downloads"] += max(
                        0,
                        c1.get("host_downloads", 0)
                        - c0.get("host_downloads", 0),
                    )
                    rec["host_download_bytes"] += max(
                        0,
                        c1.get("host_download_bytes", 0)
                        - c0.get("host_download_bytes", 0),
                    )
                return res

        self._mode = _OpProfilerMode()
        self._mode.__enter__()

        if orig_quant is not None:
            quant_name = "torch._C._quantized_linear"

            def wrapped_quant(*args, **kwargs):
                if has_metal:
                    c0 = c._metal_counters()
                t0 = time.perf_counter()
                res = orig_quant(*args, **kwargs)
                t1 = time.perf_counter()
                elapsed_ms = (t1 - t0) * 1000.0

                rec = records.setdefault(
                    quant_name,
                    {
                        "count": 0,
                        "total_ms": 0.0,
                        "compute_encoders": 0,
                        "blit_encoders": 0,
                        "host_uploads": 0,
                        "host_downloads": 0,
                        "host_download_bytes": 0,
                    },
                )
                rec["count"] += 1
                rec["total_ms"] += elapsed_ms
                if has_metal:
                    c1 = c._metal_counters()
                    rec["compute_encoders"] += max(
                        0,
                        c1.get("compute_encoders", 0)
                        - c0.get("compute_encoders", 0),
                    )
                    rec["blit_encoders"] += max(
                        0, c1.get("blit_encoders", 0) - c0.get("blit_encoders", 0)
                    )
                    rec["host_uploads"] += max(
                        0, c1.get("host_uploads", 0) - c0.get("host_uploads", 0)
                    )
                    rec["host_downloads"] += max(
                        0,
                        c1.get("host_downloads", 0)
                        - c0.get("host_downloads", 0),
                    )
                    rec["host_download_bytes"] += max(
                        0,
                        c1.get("host_download_bytes", 0)
                        - c0.get("host_download_bytes", 0),
                    )
                return res

            c._quantized_linear = wrapped_quant
            if "_C" in sys.modules and hasattr(
                sys.modules["_C"], "_quantized_linear"
            ):
                sys.modules["_C"]._quantized_linear = wrapped_quant

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if getattr(self, "_mode", None) is not None:
            self._mode.__exit__(exc_type, exc_val, exc_tb)
            self._mode = None
        c = self.torch._C
        if self.orig_quantized_linear is not None:
            c._quantized_linear = self.orig_quantized_linear
            if "_C" in sys.modules and hasattr(
                sys.modules["_C"], "_quantized_linear"
            ):
                sys.modules["_C"]._quantized_linear = self.orig_quantized_linear


# --------------------------------------------------------------------------
# Helper for SmolLM2 GGUF Block Loading
# --------------------------------------------------------------------------


def load_smollm2_from_gguf_blocks(
    snap_or_config: str, gguf_path: str, device: str = "cpu"
) -> Any:
    """Helper to load SmolLM2 and populate its linear layers with GGUF blocks.

    Reads exact GGUF blocks from SmolLM2-135M-Q8_0.gguf / SmolLM2-135M-Q4_0.gguf
    using torchnative.gguf.quantized_linear.
    """
    import torch
    from torchnative.gguf import GGUFFile, quantized_linear
    from transformers import AutoModelForCausalLM

    f = GGUFFile(gguf_path)
    model = AutoModelForCausalLM.from_pretrained(
        snap_or_config, dtype=torch.float32, attn_implementation="sdpa"
    )
    for i, layer in enumerate(model.model.layers):
        mapping = {
            f"blk.{i}.attn_q.weight": "q_proj",
            f"blk.{i}.attn_k.weight": "k_proj",
            f"blk.{i}.attn_v.weight": "v_proj",
            f"blk.{i}.attn_output.weight": "o_proj",
        }
        for gguf_name, attr in mapping.items():
            if gguf_name in f.tensors:
                setattr(layer.self_attn, attr, quantized_linear(f, gguf_name))

        mlp_mapping = {
            f"blk.{i}.ffn_gate.weight": "gate_proj",
            f"blk.{i}.ffn_up.weight": "up_proj",
            f"blk.{i}.ffn_down.weight": "down_proj",
        }
        for gguf_name, attr in mlp_mapping.items():
            if gguf_name in f.tensors:
                setattr(layer.mlp, attr, quantized_linear(f, gguf_name))

    if "output.weight" in f.tensors:
        model.lm_head = quantized_linear(f, "output.weight")
    if device != "cpu":
        # Note: QuantizedLinear.qweight is a QTensor; moving dense buffers to device:
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        model.model.norm = model.model.norm.to(device)
        for layer in model.model.layers:
            layer.input_layernorm = layer.input_layernorm.to(device)
            layer.post_attention_layernorm = (
                layer.post_attention_layernorm.to(device)
            )
    return model


# --------------------------------------------------------------------------
# Worker side implementation
# --------------------------------------------------------------------------


def _sync_device(tensor: Any, device: str) -> None:
    """Flush and wait on MPS/CPU by reading back at least 1 element to CPU."""
    if tensor is not None and hasattr(tensor, "cpu"):
        if tensor.numel() > 0:
            _ = tensor.reshape(-1)[0].cpu()


def worker_main(task_json: str) -> None:
    os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
    task = json.loads(task_json)

    model_key = task["model"]
    format_key = task["format"]  # "q8_0", "q4_0", or "f32"
    device_name = task["device"]  # "cpu" or "mps"
    mode = task["mode"]  # "quant", "gguf_dense", "gguf+quant", "gguf_blocks", "dense"
    reps = int(task.get("reps", 3))
    warmup = int(task.get("warmup", 1))
    do_profile = bool(task.get("profile", False))
    gguf_dir = task.get("gguf_dir", _DEFAULT_GGUF_DIR)

    site = task.get("gguf_site")
    if site and site not in sys.path:
        sys.path.append(site)

    import torch
    import torchnative.quant as tn_quant
    from torchnative.transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    cfg = MODEL_CONFIGS[model_key]
    snap_path = cfg["hf_snapshot"]
    gguf_filename = cfg["gguf_files"].get(format_key)
    gguf_path = os.path.join(gguf_dir, gguf_filename) if gguf_filename else None

    out: Dict[str, Any] = {
        "model": model_key,
        "format": format_key,
        "device": device_name,
        "mode": mode,
        "torch_version": torch.__version__,
        "is_shim": hasattr(torch._C, "_aten_implemented"),
    }

    # 2. Load model according to model_key and mode
    tokenizer = AutoTokenizer.from_pretrained(snap_path)
    model = None

    if mode == "gguf_blocks":
        # S2.5: the file's own GGML blocks, byte for byte, in QuantizedLinear
        # leaves (lm_head too when the file stores it quantised). This is the
        # model llama.cpp runs from the same file, so it is the like-for-like
        # comparison; "quant" re-quantises dense HF weights instead.
        if not gguf_path or not os.path.isfile(gguf_path):
            raise FileNotFoundError(f"GGUF file missing: {gguf_path}")
        model = AutoModelForCausalLM.from_pretrained(
            gguf_dir, gguf_file=gguf_filename, keep_blocks=True, dtype=torch.float32
        ).to(device_name)
        out["quant_layers"] = sum(
            1 for m in model.modules() if isinstance(m, tn_quant.QuantizedLinear)
        )
    elif model_key == "smollm2-135m":
        if mode == "dense":
            model = AutoModelForCausalLM.from_pretrained(
                snap_path, dtype=torch.float32, attn_implementation="sdpa"
            ).to(device_name)
        elif mode == "quant":
            # Load dense float32, move to device FIRST, then quantize_
            model = AutoModelForCausalLM.from_pretrained(
                snap_path, dtype=torch.float32, attn_implementation="sdpa"
            ).to(device_name)
            q_report = tn_quant.quantize_(model, format=format_key)
            out["quant_report"] = str(q_report)
        elif mode == "exact_gguf_blocks":
            if not gguf_path or not os.path.isfile(gguf_path):
                raise FileNotFoundError(f"GGUF file missing: {gguf_path}")
            model = load_smollm2_from_gguf_blocks(
                snap_path, gguf_path, device=device_name
            )
        else:
            raise ValueError(f"Unknown mode {mode!r} for smollm2-135m")
    elif model_key == "qwen3-0.6b":
        if mode == "dense":
            model = AutoModelForCausalLM.from_pretrained(
                snap_path, dtype=torch.float32, attn_implementation="sdpa"
            ).to(device_name)
        elif mode == "gguf_dense":
            if not gguf_path or not os.path.isfile(gguf_path):
                raise FileNotFoundError(f"GGUF file missing: {gguf_path}")
            model = AutoModelForCausalLM.from_pretrained(
                gguf_dir, gguf_file=gguf_filename, dtype=torch.float32
            ).to(device_name)
        elif mode == "gguf+quant":
            if not gguf_path or not os.path.isfile(gguf_path):
                raise FileNotFoundError(f"GGUF file missing: {gguf_path}")
            model = AutoModelForCausalLM.from_pretrained(
                gguf_dir, gguf_file=gguf_filename, dtype=torch.float32
            ).to(device_name)
            q_report = tn_quant.quantize_(model, format=format_key)
            out["quant_report"] = str(q_report)
        elif mode == "quant":
            model = AutoModelForCausalLM.from_pretrained(
                snap_path, dtype=torch.float32, attn_implementation="sdpa"
            ).to(device_name)
            q_report = tn_quant.quantize_(model, format=format_key)
            out["quant_report"] = str(q_report)
        else:
            raise ValueError(f"Unknown mode {mode!r} for qwen3-0.6b")

    model.eval()

    # 3. Prefill Benchmarks (S in [128, 512])
    prefill_results: Dict[str, Any] = {}
    for S in PREFILL_LENGTHS:
        ids = (
            torch.arange(1, S + 1, dtype=torch.int64).reshape(1, S) % 4096 + 1
        ).to(device_name)

        # Warmup
        for _ in range(warmup):
            with torch.no_grad():
                logits = model(ids).logits
                next_tok = logits[0, -1].argmax()
                _sync_device(next_tok, device_name)

        ttft_samples_ms: List[float] = []
        tok_s_samples: List[float] = []
        for _ in range(reps):
            with torch.no_grad():
                t0 = time.perf_counter()
                logits = model(ids).logits
                next_tok = logits[0, -1].argmax()
                _sync_device(next_tok, device_name)
                t1 = time.perf_counter()
            elapsed_ms = (t1 - t0) * 1000.0
            ttft_samples_ms.append(elapsed_ms)
            tok_s_samples.append(S / (elapsed_ms / 1000.0))

        prefill_results[str(S)] = {
            "samples_ttft_ms": ttft_samples_ms,
            "samples_prefill_tok_s": tok_s_samples,
            "stats_ttft_ms": _stats(ttft_samples_ms),
            "stats_prefill_tok_s": _stats(tok_s_samples),
            "median_ttft_ms": _stats(ttft_samples_ms)["median"],
            "std_ttft_ms": _stats(ttft_samples_ms)["std"],
            "median_prefill_tok_s": _stats(tok_s_samples)["median"],
            "std_prefill_tok_s": _stats(tok_s_samples)["std"],
            "mean_ttft_ms": sum(ttft_samples_ms) / len(ttft_samples_ms),
            "mean_prefill_tok_s": sum(tok_s_samples) / len(tok_s_samples),
        }
    out["prefill"] = prefill_results

    # 4. Decode Benchmarks (128 steps from 5-token prompt)
    prompt_inputs = tokenizer(VERIFY_PROMPT, return_tensors="pt")
    prompt_ids = prompt_inputs["input_ids"].to(device_name)

    # Warmup 1 decode run of 8 steps
    with torch.no_grad():
        w_out = model(prompt_ids, use_cache=True)
        w_cur = w_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        w_past = w_out.past_key_values
        for _ in range(8):
            w_out = model(w_cur, past_key_values=w_past, use_cache=True)
            w_past = w_out.past_key_values
            w_cur = w_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            _sync_device(w_cur, device_name)

    decode_samples_ms: List[float] = []
    decode_tok_s_samples: List[float] = []
    ttft_prompt5_samples_ms: List[float] = []
    last_tokens: List[int] = []

    for _ in range(reps):
        with torch.no_grad():
            t0_p5 = time.perf_counter()
            p_out = model(prompt_ids, use_cache=True)
            cur_tok = p_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            _sync_device(cur_tok, device_name)
            t1_p5 = time.perf_counter()
            ttft_p5_ms = (t1_p5 - t0_p5) * 1000.0
            ttft_prompt5_samples_ms.append(ttft_p5_ms)

            past = p_out.past_key_values
            tokens = [int(cur_tok[0, 0].cpu().item())]

            t0_dec = time.perf_counter()
            for step in range(1, DECODE_STEPS):
                s_out = model(cur_tok, past_key_values=past, use_cache=True)
                past = s_out.past_key_values
                cur_tok = s_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                tok_id = int(cur_tok[0, 0].cpu().item())
                tokens.append(tok_id)
            t1_dec = time.perf_counter()

        dec_ms = (t1_dec - t0_dec) * 1000.0
        decode_samples_ms.append(dec_ms)
        decode_tok_s_samples.append((DECODE_STEPS - 1) / (dec_ms / 1000.0))
        last_tokens = tokens

    # Also measure model.generate time
    gen_samples_ms: List[float] = []
    gen_tok_s_samples: List[float] = []
    for _ in range(reps):
        with torch.no_grad():
            t0_g = time.perf_counter()
            g_out = model.generate(
                prompt_ids,
                max_new_tokens=DECODE_STEPS,
                min_new_tokens=DECODE_STEPS,
                do_sample=False,
                use_cache=True,
            )
            _sync_device(g_out, device_name)
            t1_g = time.perf_counter()
        g_ms = (t1_g - t0_g) * 1000.0
        gen_samples_ms.append(g_ms)
        gen_tok_s_samples.append(DECODE_STEPS / (g_ms / 1000.0))

    out["decode"] = {
        "steps": DECODE_STEPS - 1,
        "samples_decode_128_ms": decode_samples_ms,
        "samples_decode_tok_s": decode_tok_s_samples,
        "stats_decode_128_ms": _stats(decode_samples_ms),
        "stats_decode_tok_s": _stats(decode_tok_s_samples),
        "median_decode_128_ms": _stats(decode_samples_ms)["median"],
        "std_decode_128_ms": _stats(decode_samples_ms)["std"],
        "median_decode_tok_s": _stats(decode_tok_s_samples)["median"],
        "std_decode_tok_s": _stats(decode_tok_s_samples)["std"],
        "mean_decode_128_ms": sum(decode_samples_ms) / len(decode_samples_ms),
        "mean_decode_tok_s": sum(decode_tok_s_samples)
        / len(decode_tok_s_samples),
        "samples_ttft_prompt5_ms": ttft_prompt5_samples_ms,
        "stats_ttft_prompt5_ms": _stats(ttft_prompt5_samples_ms),
        "median_ttft_prompt5_ms": _stats(ttft_prompt5_samples_ms)["median"],
        "std_ttft_prompt5_ms": _stats(ttft_prompt5_samples_ms)["std"],
        "mean_ttft_prompt5_ms": sum(ttft_prompt5_samples_ms)
        / len(ttft_prompt5_samples_ms),
        "samples_gen_128_ms": gen_samples_ms,
        "samples_gen_tok_s": gen_tok_s_samples,
        "stats_gen_128_ms": _stats(gen_samples_ms),
        "stats_gen_tok_s": _stats(gen_tok_s_samples),
        "median_gen_tok_s": _stats(gen_tok_s_samples)["median"],
        "std_gen_tok_s": _stats(gen_tok_s_samples)["std"],
        "mean_gen_tok_s": sum(gen_tok_s_samples) / len(gen_tok_s_samples),
    }

    # 5. Greedy Output Token Verification (20 tokens and 128 tokens)
    text_20 = tokenizer.decode(last_tokens[:20], skip_special_tokens=True)
    text_128 = tokenizer.decode(last_tokens[:128], skip_special_tokens=True)
    out["verify"] = {
        "tokens_20": last_tokens[:20],
        "text_20": text_20,
        "tokens_128": last_tokens[:128],
        "text_128": text_128,
    }

    # 6. Op-Level Profiling Mode (--profile)
    if do_profile:
        prof_res: Dict[str, Any] = {}
        has_metal = (
            device_name == "mps"
            and hasattr(torch._C, "_metal_counters")
            and torch._C._metal_counters().get("built", False)
        )
        # Profile prefill (S=128)
        ids_128 = (
            torch.arange(1, 129, dtype=torch.int64).reshape(1, 128) % 4096 + 1
        ).to(device_name)
        if has_metal:
            c_before_prefill = torch._C._metal_counters()
        t0_prof_p = time.perf_counter()
        with ProfileScope(torch, device_name) as ps_prefill:
            with torch.no_grad():
                l = model(ids_128).logits
                nt = l[0, -1].argmax()
                _sync_device(nt, device_name)
        t1_prof_p = time.perf_counter()
        prof_res["prefill_wall_ms"] = (t1_prof_p - t0_prof_p) * 1000.0
        if has_metal:
            c_after_prefill = torch._C._metal_counters()
            prof_res["prefill_metal_delta"] = {
                k: c_after_prefill.get(k, 0) - c_before_prefill.get(k, 0)
                for k in (
                    "compute_encoders",
                    "blit_encoders",
                    "host_uploads",
                    "host_downloads",
                    "host_download_bytes",
                )
            }
        prof_res["prefill_ops"] = ps_prefill.records

        # Profile decode (32 steps)
        with torch.no_grad():
            p_out = model(prompt_ids, use_cache=True)
            c_tok = p_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            _sync_device(c_tok, device_name)
            past_kv = p_out.past_key_values

        if has_metal:
            c_before_decode = torch._C._metal_counters()
        t0_prof_d = time.perf_counter()
        with ProfileScope(torch, device_name) as ps_decode:
            with torch.no_grad():
                for _ in range(PROFILE_DECODE_STEPS):
                    s_out = model(c_tok, past_key_values=past_kv, use_cache=True)
                    past_kv = s_out.past_key_values
                    c_tok = s_out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                    _sync_device(c_tok, device_name)
        t1_prof_d = time.perf_counter()
        prof_res["decode_wall_ms"] = (t1_prof_d - t0_prof_d) * 1000.0
        if has_metal:
            c_after_decode = torch._C._metal_counters()
            prof_res["decode_metal_delta"] = {
                k: c_after_decode.get(k, 0) - c_before_decode.get(k, 0)
                for k in (
                    "compute_encoders",
                    "blit_encoders",
                    "host_uploads",
                    "host_downloads",
                    "host_download_bytes",
                )
            }
        prof_res["decode_ops"] = ps_decode.records
        out["profile"] = prof_res

    sys.stdout.write("@@JSON@@" + json.dumps(out) + "\n")
    sys.stdout.flush()


# --------------------------------------------------------------------------
# Driver side implementation
# --------------------------------------------------------------------------


def build_worker_env(threads: int, gguf_venv: str) -> Dict[str, str]:
    env = dict(os.environ)
    env["RAYON_NUM_THREADS"] = str(threads)
    env["CANDLE_NUM_THREADS"] = str(threads)
    env["OMP_NUM_THREADS"] = str(threads)
    env["VECLIB_MAXIMUM_THREADS"] = str(threads)
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HOME"] = _DEFAULT_HF_HOME

    existing_pythonpath = [
        p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p
    ]
    env["PYTHONPATH"] = os.pathsep.join([_SHIM_PATH] + existing_pythonpath)

    site = find_gguf_site(gguf_venv)
    if site:
        env["TN_GGUF_SITE"] = site
    return env


def run_torchnative_task(
    python_bin: str, task: Dict[str, Any], threads: int, gguf_venv: str
) -> Dict[str, Any]:
    task["gguf_site"] = find_gguf_site(gguf_venv)
    task_json = json.dumps(task)
    env = build_worker_env(threads, gguf_venv)

    load_before = loadavg()
    proc = subprocess.run(
        [
            python_bin,
            os.path.abspath(__file__),
            "--worker",
            "--worker-task",
            task_json,
        ],
        capture_output=True,
        text=True,
        env=env,
        cwd=_ROOT,
    )
    load_after = loadavg()

    if proc.returncode != 0:
        sys.stderr.write(
            f"Worker failed for {task['model']}/{task['format']}/{task['device']}:\n"
        )
        sys.stderr.write(proc.stdout[-3000:] + "\n" + proc.stderr[-3000:] + "\n")
        raise RuntimeError(
            f"torchnative worker exited with code {proc.returncode}"
        )

    lines = [
        line
        for line in proc.stdout.splitlines()
        if line.startswith("@@JSON@@")
    ]
    if not lines:
        sys.stderr.write(proc.stdout[-3000:] + "\n" + proc.stderr[-3000:] + "\n")
        raise RuntimeError("Worker produced no @@JSON@@ output line")

    payload = json.loads(lines[-1][len("@@JSON@@") :])
    payload["load_before"] = load_before
    payload["load_after"] = load_after
    return payload


def run_llama_bench(
    bench_bin: str,
    gguf_path: str,
    threads: int,
    ngl: int,
    reps: int,
) -> Dict[str, Any]:
    if not os.path.isfile(bench_bin):
        return {"error": f"llama-bench not found at {bench_bin}"}
    if not os.path.isfile(gguf_path):
        return {"error": f"GGUF file not found at {gguf_path}"}

    cmd = [
        bench_bin,
        "-m",
        gguf_path,
        "-p",
        "128,512",
        "-n",
        "128",
        "-t",
        str(threads),
        "-ngl",
        str(ngl),
        "-r",
        str(reps),
        "-o",
        "json",
    ]
    load_before = loadavg()
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_ROOT)
    load_after = loadavg()

    if proc.returncode != 0:
        return {
            "error": f"llama-bench exited with {proc.returncode}: {proc.stderr[:1000]}"
        }

    try:
        entries = json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        return {"error": f"Failed to parse llama-bench JSON: {exc}"}

    parsed: Dict[str, Any] = {
        "load_before": load_before,
        "load_after": load_after,
        "prefill": {},
        "decode": {},
    }
    for e in entries:
        n_p = e.get("n_prompt", 0)
        n_g = e.get("n_gen", 0)
        samples_ns = e.get("samples_ns", [])
        samples_ts = e.get("samples_ts", [])
        if n_p in (128, 512) and n_g == 0:
            ttft_ms = [ns / 1e6 for ns in samples_ns]
            parsed["prefill"][str(n_p)] = {
                "samples_ttft_ms": ttft_ms,
                "samples_prefill_tok_s": samples_ts,
                "stats_ttft_ms": _stats(ttft_ms),
                "stats_prefill_tok_s": _stats(samples_ts),
                "median_ttft_ms": _stats(ttft_ms)["median"],
                "std_ttft_ms": _stats(ttft_ms)["std"],
                "median_prefill_tok_s": _stats(samples_ts)["median"],
                "std_prefill_tok_s": _stats(samples_ts)["std"],
                "mean_ttft_ms": sum(ttft_ms) / len(ttft_ms) if ttft_ms else 0,
                "mean_prefill_tok_s": sum(samples_ts) / len(samples_ts)
                if samples_ts
                else 0,
            }
        elif n_p == 0 and n_g == 128:
            dec_ms = [ns / 1e6 for ns in samples_ns]
            parsed["decode"] = {
                "steps": 128,
                "samples_decode_128_ms": dec_ms,
                "samples_decode_tok_s": samples_ts,
                "stats_decode_128_ms": _stats(dec_ms),
                "stats_decode_tok_s": _stats(samples_ts),
                "median_decode_128_ms": _stats(dec_ms)["median"],
                "std_decode_128_ms": _stats(dec_ms)["std"],
                "median_decode_tok_s": _stats(samples_ts)["median"],
                "std_decode_tok_s": _stats(samples_ts)["std"],
                "mean_decode_128_ms": sum(dec_ms) / len(dec_ms)
                if dec_ms
                else 0,
                "mean_decode_tok_s": sum(samples_ts) / len(samples_ts)
                if samples_ts
                else 0,
            }
    return parsed


def run_llama_completion(
    completion_bin: str,
    gguf_path: str,
    prompt: str,
    n_tokens: int,
    threads: int,
    ngl: int,
) -> Dict[str, Any]:
    if not os.path.isfile(completion_bin):
        return {"error": f"llama-completion not found at {completion_bin}"}
    if not os.path.isfile(gguf_path):
        return {"error": f"GGUF file not found at {gguf_path}"}

    cmd = [
        completion_bin,
        "-m",
        gguf_path,
        "-p",
        prompt,
        "-n",
        str(n_tokens),
        "-no-cnv",
        "-c",
        "2048",
        "--temp",
        "0",
        "--top-k",
        "1",
        "--top-p",
        "1.0",
        "--min-p",
        "0.0",
        "--seed",
        "42",
        "-ngl",
        str(ngl),
        "-t",
        str(threads),
        "-tb",
        str(threads),
        "--perf",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=_ROOT)
    if proc.returncode != 0:
        return {
            "error": f"llama-completion failed ({proc.returncode}): {proc.stderr[:1000]}"
        }

    raw_stdout = proc.stdout
    # Text generated by llama-completion contains prompt followed by generation
    gen_text = raw_stdout
    if gen_text.startswith(prompt):
        gen_text = gen_text[len(prompt) :]
    gen_text = gen_text.strip()

    # Parse perf metrics from stderr
    stderr = proc.stderr
    p_eval_m = re.search(
        r"prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens.*?([0-9.]+)\s*tokens per second",
        stderr,
    )
    eval_m = re.search(
        r"\beval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*runs.*?([0-9.]+)\s*tokens per second",
        stderr,
    )

    return {
        "text": gen_text,
        "prompt_eval_ms": float(p_eval_m.group(1)) if p_eval_m else None,
        "prompt_eval_tok_s": float(p_eval_m.group(3)) if p_eval_m else None,
        "eval_ms": float(eval_m.group(1)) if eval_m else None,
        "eval_tok_s": float(eval_m.group(3)) if eval_m else None,
    }


def compare_text(tn_text: str, llama_text: str) -> Dict[str, Any]:
    tn_s = tn_text.strip()
    ll_s = llama_text.strip()
    exact_match = tn_s == ll_s
    char_divergence = None
    if not exact_match:
        for idx, (c1, c2) in enumerate(zip(tn_s, ll_s)):
            if c1 != c2:
                char_divergence = idx
                break
        if char_divergence is None:
            char_divergence = min(len(tn_s), len(ll_s))

    return {
        "match": exact_match,
        "first_diff_char": char_divergence,
        "tn_len": len(tn_s),
        "llama_len": len(ll_s),
    }


def format_profile_table(ops_dict: Dict[str, Dict[str, Any]]) -> str:
    lines = []
    lines.append(
        f"{'Operator':<38} {'Calls':>6} {'Total(ms)':>10} {'Avg(ms)':>8} "
        f"{'Enc':>5} {'Blit':>5} {'Up':>5} {'Down':>5} {'DownBytes':>10}"
    )
    lines.append("-" * 100)
    sorted_ops = sorted(
        ops_dict.items(), key=lambda kv: kv[1]["total_ms"], reverse=True
    )
    for op, stats in sorted_ops:
        cnt = stats["count"]
        tot = stats["total_ms"]
        avg = tot / cnt if cnt else 0.0
        enc = stats.get("compute_encoders", 0)
        blit = stats.get("blit_encoders", 0)
        up = stats.get("host_uploads", 0)
        down = stats.get("host_downloads", 0)
        down_bytes = stats.get("host_download_bytes", 0)
        lines.append(
            f"{op:<38} {cnt:>6} {tot:>10.2f} {avg:>8.3f} "
            f"{enc:>5} {blit:>5} {up:>5} {down:>5} {down_bytes:>10}"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Main entry point & CLI parser
# --------------------------------------------------------------------------


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark and profile torchnative against llama.cpp"
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="Internal worker flag (run single task in subprocess)",
    )
    parser.add_argument(
        "--worker-task",
        type=str,
        default="",
        help="JSON payload for worker task",
    )
    parser.add_argument(
        "--model",
        choices=["smollm2-135m", "qwen3-0.6b", "all"],
        default="all",
        help="Model to benchmark (default: all)",
    )
    parser.add_argument(
        "--format",
        choices=["q8_0", "q4_0", "all"],
        default="all",
        help="Quantization format to benchmark (default: all)",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "mps", "all"],
        default="all",
        help="Device to benchmark on (default: all)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Thread count matching M1 performance cores (default: 4)",
    )
    parser.add_argument(
        "--reps",
        type=int,
        default=3,
        help="Repetitions for each timed measurement (default: 3)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="Warmup iterations (default: 1)",
    )
    parser.add_argument(
        "--include-dense",
        action="store_true",
        help="Include float32 dense baseline",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Instrument torchnative op-level dispatch and Metal counters",
    )
    parser.add_argument(
        "--skip-llama",
        action="store_true",
        help="Skip llama.cpp benchmarks",
    )
    parser.add_argument(
        "--skip-torchnative",
        action="store_true",
        help="Skip torchnative benchmarks",
    )
    parser.add_argument(
        "--tn-modes",
        nargs="+",
        default=None,
        choices=["quant", "gguf_dense", "gguf+quant", "gguf_blocks"],
        help="torchnative load modes to run (default: quant and gguf_blocks for "
        "SmolLM2; gguf_dense, gguf+quant and gguf_blocks for Qwen3)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default="",
        help="Path to write JSON output results",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=os.environ.get("PYTHON", sys.executable),
        help="Python binary for torchnative worker (default: sys.executable)",
    )
    parser.add_argument(
        "--gguf-dir",
        type=str,
        default=_DEFAULT_GGUF_DIR,
        help=f"Directory with GGUF files (default: {_DEFAULT_GGUF_DIR})",
    )
    parser.add_argument(
        "--llama-bench",
        type=str,
        default=_DEFAULT_LLAMA_BENCH,
        help=f"Path to llama-bench (default: {_DEFAULT_LLAMA_BENCH})",
    )
    parser.add_argument(
        "--llama-completion",
        type=str,
        default=_DEFAULT_LLAMA_COMPLETION,
        help=f"Path to llama-completion (default: {_DEFAULT_LLAMA_COMPLETION})",
    )
    parser.add_argument(
        "--gguf-venv",
        type=str,
        default=_DEFAULT_GGUF_VENV,
        help=f"Path to gguf-venv (default: {_DEFAULT_GGUF_VENV})",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.worker:
        if not args.worker_task:
            sys.stderr.write("Worker invoked without --worker-task\n")
            sys.exit(1)
        worker_main(args.worker_task)
        return

    models = (
        ["smollm2-135m", "qwen3-0.6b"]
        if args.model == "all"
        else [args.model]
    )
    formats = ["q8_0", "q4_0"] if args.format == "all" else [args.format]
    if args.include_dense:
        formats.append("f32")

    devices = ["cpu", "mps"] if args.device == "all" else [args.device]

    print("=" * 80)
    print("TORCHNATIVE vs LLAMA.CPP BENCHMARK & PROFILING HARNESS")
    print(f"Repo:           {_ROOT}")
    print(f"Python:         {args.python}")
    print(f"Threads:        {args.threads}")
    print(f"Reps:           {args.reps} (warmup: {args.warmup})")
    print(f"Profile:        {args.profile}")
    print(f"Initial Load:   {loadavg()}")
    print("=" * 80)

    overall_results: Dict[str, Any] = {
        "metadata": {
            "root": _ROOT,
            "python": args.python,
            "threads": args.threads,
            "reps": args.reps,
            "warmup": args.warmup,
            "profile": args.profile,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        },
        "models": {},
        "comparisons": [],
    }

    for model_key in models:
        mcfg = MODEL_CONFIGS[model_key]
        print(f"\n>>> MODEL: {mcfg['display_name']} ({model_key})")
        overall_results["models"][model_key] = {"torchnative": {}, "llama_cpp": {}}

        for fmt in formats:
            gguf_name = mcfg["gguf_files"].get(fmt)
            gguf_path = (
                os.path.join(args.gguf_dir, gguf_name) if gguf_name else ""
            )

            for dev in devices:
                key = f"{fmt}_{dev}"
                print(f"\n--- Format: {fmt} | Device: {dev} ---")

                # -------------------------------------------------------------
                # 1. torchnative measurements
                # -------------------------------------------------------------
                if not args.skip_torchnative:
                    # Select modes:
                    # For smollm2: "quant" (and "dense" for f32)
                    # For qwen3: "gguf_dense" and "gguf+quant" (and "dense" for f32)
                    tn_modes = []
                    if fmt == "f32":
                        tn_modes = ["dense"]
                    elif args.tn_modes:
                        tn_modes = args.tn_modes
                    elif model_key == "smollm2-135m":
                        tn_modes = ["quant", "gguf_blocks"]
                    elif model_key == "qwen3-0.6b":
                        tn_modes = ["gguf_dense", "gguf+quant", "gguf_blocks"]

                    for mode in tn_modes:
                        mode_label = f"{key}_{mode}"
                        print(f"  [torchnative] Mode: {mode} ...", end=" ", flush=True)
                        t_task = {
                            "model": model_key,
                            "format": fmt,
                            "device": dev,
                            "mode": mode,
                            "threads": args.threads,
                            "reps": args.reps,
                            "warmup": args.warmup,
                            "profile": args.profile,
                            "gguf_dir": args.gguf_dir,
                        }
                        try:
                            tn_res = run_torchnative_task(
                                args.python, t_task, args.threads, args.gguf_venv
                            )
                            overall_results["models"][model_key]["torchnative"][
                                mode_label
                            ] = tn_res
                            p128 = tn_res["prefill"]["128"]["mean_prefill_tok_s"]
                            p512 = tn_res["prefill"]["512"]["mean_prefill_tok_s"]
                            dec = tn_res["decode"]["mean_decode_tok_s"]
                            ttft5 = tn_res["decode"]["mean_ttft_prompt5_ms"]
                            print(
                                f"done! Prefill S128: {p128:6.1f} tok/s | "
                                f"S512: {p512:6.1f} tok/s | "
                                f"Decode: {dec:6.1f} tok/s (TTFT5: {ttft5:5.1f} ms)"
                            )

                            if args.profile and "profile" in tn_res:
                                print("    Profiling Prefill (S=128) Top Ops:")
                                prof_p = tn_res["profile"].get("prefill_ops", {})
                                p_lines = format_profile_table(prof_p).splitlines()
                                for pl in p_lines[:8]:
                                    print("     ", pl)
                                print("    Profiling Decode (32 steps) Top Ops:")
                                prof_d = tn_res["profile"].get("decode_ops", {})
                                d_lines = format_profile_table(prof_d).splitlines()
                                for dl in d_lines[:8]:
                                    print("     ", dl)

                        except Exception as exc:
                            print(f"FAILED: {exc}")
                            overall_results["models"][model_key]["torchnative"][
                                mode_label
                            ] = {"error": str(exc)}

                # -------------------------------------------------------------
                # 2. llama.cpp measurements
                # -------------------------------------------------------------
                if not args.skip_llama and gguf_path and os.path.isfile(gguf_path):
                    ngl = 99 if dev == "mps" else 0
                    llama_key = f"{fmt}_{dev}"
                    print(
                        f"  [llama.cpp]   Benchmarking (ngl={ngl}) ...",
                        end=" ",
                        flush=True,
                    )
                    l_bench = run_llama_bench(
                        args.llama_bench,
                        gguf_path,
                        args.threads,
                        ngl,
                        args.reps,
                    )
                    overall_results["models"][model_key]["llama_cpp"][
                        f"{llama_key}_bench"
                    ] = l_bench

                    if "error" not in l_bench:
                        lp128 = l_bench["prefill"]["128"]["mean_prefill_tok_s"]
                        lp512 = l_bench["prefill"]["512"]["mean_prefill_tok_s"]
                        ldec = l_bench["decode"]["mean_decode_tok_s"]
                        print(
                            f"done! Prefill S128: {lp128:6.1f} tok/s | "
                            f"S512: {lp512:6.1f} tok/s | "
                            f"Decode: {ldec:6.1f} tok/s"
                        )
                    else:
                        print(f"FAILED: {l_bench['error']}")

                    # Completion verification for 20 and 128 tokens
                    l_comp_20 = run_llama_completion(
                        args.llama_completion,
                        gguf_path,
                        VERIFY_PROMPT,
                        20,
                        args.threads,
                        ngl,
                    )
                    l_comp_128 = run_llama_completion(
                        args.llama_completion,
                        gguf_path,
                        VERIFY_PROMPT,
                        128,
                        args.threads,
                        ngl,
                    )
                    overall_results["models"][model_key]["llama_cpp"][
                        f"{llama_key}_comp_20"
                    ] = l_comp_20
                    overall_results["models"][model_key]["llama_cpp"][
                        f"{llama_key}_comp_128"
                    ] = l_comp_128

                    # Compare greedy output text with torchnative if available
                    for m_lbl, t_data in overall_results["models"][model_key][
                        "torchnative"
                    ].items():
                        if m_lbl.startswith(f"{fmt}_{dev}") and "verify" in t_data:
                            t_verify = t_data["verify"]
                            comp_20 = compare_text(
                                t_verify["text_20"], l_comp_20.get("text", "")
                            )
                            comp_128 = compare_text(
                                t_verify["text_128"], l_comp_128.get("text", "")
                            )
                            overall_results["comparisons"].append(
                                {
                                    "model": model_key,
                                    "format": fmt,
                                    "device": dev,
                                    "torchnative_mode": m_lbl,
                                    "verify_20": comp_20,
                                    "verify_128": comp_128,
                                }
                            )
                            print(
                                f"    [Greedy Verify vs {m_lbl}] 20 tok match: {comp_20['match']} | "
                                f"128 tok match: {comp_128['match']}"
                            )

    print("\n" + "=" * 80)
    print(f"Final Load Average: {loadavg()}")
    print("=" * 80)

    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(overall_results, f, indent=2)
        print(f"Results written to: {args.output_json}")


if __name__ == "__main__":
    main()
