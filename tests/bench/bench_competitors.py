"""Compare torchnative with ONNX Runtime and LiteRT on one model and one host.

Bench-only (leader decision D11, issue #283). The result format follows
bench_llama_cpp.py (`models -> <model> -> <backend> -> prefill / decode / verify`,
the same `median_*` key names), so the two files can be read side by side.

Backends (a requested backend that is absent is recorded as {"refused": "<name>: <reason>"}
and printed as `SKIP <backend>: <reason>`; CPU is never substituted for an accelerator):
  torchnative-cpu, torchnative-mps   the shim, run through bench_llama_cpp's own worker
  onnxruntime-cpu / -coreml / -qnn   upstream torch's dynamo exporter -> ORT session
  litert-cpu / litert-gpu            litert-torch's generative converter -> ai_edge_litert

Conversion paths and their limits (they are written into every result, never implied):
  * ONNX: `torch.onnx.export(dynamo=True)` of a wrapper (input_ids -> logits) with a dynamic
    sequence axis. There is NO KV cache: every decode step recomputes the whole sequence, so its
    decode tok/s is not comparable with a KV-cached decode. The result says
    `kv_cache: "none (full-sequence recompute per step)"`. An exporter failure is a refusal.
  * LiteRT: litert_torch.generative converter (SmolLM example builder, then convert_to_litert),
    which writes a .tflite with `prefill*` and `decode` signatures and an explicit KV cache. The
    prompt is fed through `decode` token by token. The generative API is read from the installed
    package at run time; a missing builder or changed signature is a refusal naming it.
  * Accelerator providers (CoreML, QNN) always get CPU appended by ORT for unsupported nodes.
    The harness records `session.get_providers()` and refuses unless the requested provider is
    first, but it cannot see how many nodes ran on it: that is recorded as `partition: "unknown"`.

Grades (AGENTS.md §16): a backend's tokens are called "agrees" only when they equal upstream
torch float32 greedy tokens on the same prompt; otherwise the first divergence index is recorded.

Run (never in the gate; measurements run alone, AGENTS.md rule 8):
  uv venv .caches/bench-venv --python 3.13
  uv pip install --python .caches/bench-venv/bin/python -r tests/bench/requirements-competitors.txt
  .caches/spike-venv/bin/python tests/bench/bench_competitors.py --backends onnxruntime-cpu,litert-cpu,torchnative-cpu

Everything it makes goes to .scratch/bench-competitors/ (git-ignored). No U+2014 anywhere.
"""

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

import bench_llama_cpp as L  # stdlib-only at import time; main() is guarded

ALL_BACKENDS = [
    "torchnative-cpu", "torchnative-mps",
    "onnxruntime-cpu", "onnxruntime-coreml", "onnxruntime-qnn",
    "litert-cpu", "litert-gpu",
]
ORT_PROVIDER = {
    "onnxruntime-cpu": "CPUExecutionProvider",
    "onnxruntime-coreml": "CoreMLExecutionProvider",
    "onnxruntime-qnn": "QNNExecutionProvider",
}
VERIFY_TOKENS = 32
SCRATCH = os.path.join(L._ROOT, ".scratch", "bench-competitors")
DEFAULT_BENCH_PY = os.path.join(L._CACHES, "bench-venv", "bin", "python")
DEFAULT_TN_PY = os.path.join(L._CACHES, "spike-venv", "bin", "python")


# ---------------------------------------------------------------- worker side

class Refused(Exception):
    """The backend cannot run here; the message becomes {"refused": "<name>: <reason>"}."""


def _timed_stats(samples_ms: List[float], count: int) -> Dict[str, Any]:
    tok_s = [count / (ms / 1000.0) for ms in samples_ms]
    return {
        "samples_ms": samples_ms,
        "stats_ms": L._stats(samples_ms),
        "median_ms": L._stats(samples_ms)["median"],
        "samples_tok_s": tok_s,
        "stats_tok_s": L._stats(tok_s),
        "median_tok_s": L._stats(tok_s)["median"],
    }


def _load_reference(torch, snap):
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float32, attn_implementation="eager").eval()


class OnnxRunner:
    """Full-sequence ONNX model. kv_cache is none by construction (see module docstring)."""
    kv_cache = "none (full-sequence recompute per step)"

    def __init__(self, torch, snap, backend, threads, scratch):
        import numpy as np
        try:
            import onnxruntime as ort
        except ImportError as e:
            raise Refused(f"{backend}: onnxruntime is not installed ({e})")
        self.np = np
        want = ORT_PROVIDER[backend]
        avail = ort.get_available_providers()
        if want not in avail:
            raise Refused(f"{backend}: {want} not in onnxruntime.get_available_providers() {avail}")
        path = os.path.join(scratch, "smollm2-135m.onnx")
        if not os.path.isfile(path):
            try:
                import onnx  # noqa: F401
                import onnxscript  # noqa: F401
            except ImportError as e:
                raise Refused(f"{backend}: ONNX export needs onnx and onnxscript ({e})")

            class Wrap(torch.nn.Module):
                def __init__(self, m):
                    super().__init__()
                    self.m = m

                def forward(self, input_ids):
                    return self.m(input_ids, use_cache=False).logits

            ids = torch.arange(1, 9, dtype=torch.int64).reshape(1, 8)
            seq = torch.export.Dim("seq", min=1, max=2048)
            try:
                prog = torch.onnx.export(
                    Wrap(_load_reference(torch, snap)), (ids,), dynamo=True,
                    input_names=["input_ids"], output_names=["logits"],
                    dynamic_shapes={"input_ids": {1: seq}},
                )
                prog.save(path)
            except Exception as e:  # exporter walls are findings, not crashes
                raise Refused(f"{backend}: torch.onnx.export(dynamo=True) failed: {type(e).__name__}: {str(e)[:300]}")
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        opts: List[Dict[str, str]] = [{}]
        if backend == "onnxruntime-qnn":
            be = os.environ.get("TORCHNATIVE_QNN_BACKEND")
            if not be:
                raise Refused(f"{backend}: set TORCHNATIVE_QNN_BACKEND to the QNN backend library path")
            opts = [{"backend_path": be}]
        try:
            self.sess = ort.InferenceSession(path, so, providers=[want], provider_options=opts)
        except Exception as e:
            raise Refused(f"{backend}: session creation with only {want} failed: {type(e).__name__}: {str(e)[:300]}")
        used = self.sess.get_providers()
        if not used or used[0] != want:
            raise Refused(f"{backend}: session used {used}, not {want} first")
        self.info = {"onnx": path, "providers_used": used, "partition": "unknown",
                     "onnxruntime": ort.__version__}

    def logits(self, ids: List[int]):
        x = self.np.asarray([ids], dtype=self.np.int64)
        return self.sess.run(["logits"], {"input_ids": x})[0][0, -1]

    # one object exposes: prefill(ids) -> argmax, greedy(prompt, n) -> tokens
    def prefill(self, ids):
        return int(self.logits(ids).argmax())

    def greedy(self, prompt, n):
        seq = list(prompt)
        out = []
        for _ in range(n):
            t = int(self.logits(seq).argmax())
            out.append(t)
            seq.append(t)
        return out


class LiteRTRunner:
    kv_cache = "explicit KV cache signatures from the generative converter"

    def __init__(self, torch, snap, backend, threads, scratch, prompt_lens, decode_tokens):
        import numpy as np
        self.np = np
        try:
            from ai_edge_litert import interpreter as itp
        except ImportError as e:
            raise Refused(f"{backend}: ai_edge_litert is not installed ({e})")
        try:
            from litert_torch.generative.examples.smollm import smollm
            from litert_torch.generative.utilities import converter
        except ImportError as e:
            raise Refused(f"{backend}: litert_torch generative API not importable ({e})")
        kv_max = max(prompt_lens) + decode_tokens + 8
        prefix = "smollm2"
        outdir = os.path.join(scratch, "litert")
        os.makedirs(outdir, exist_ok=True)
        tfl = [f for f in os.listdir(outdir) if f.endswith(".tflite")]
        if not tfl:
            builder = getattr(smollm, "build_model_v2", None) or getattr(smollm, "build_model", None)
            if builder is None:
                raise Refused(f"{backend}: litert_torch smollm example has no build_model_v2 or build_model")
            try:
                model = builder(snap, mask_cache_size=kv_max)
                converter.convert_to_litert(
                    model, output_path=outdir, output_name_prefix=prefix,
                    prefill_seq_len=sorted(prompt_lens), kv_cache_max_len=kv_max, quantize="none",
                )
            except Exception as e:
                raise Refused(f"{backend}: litert_torch conversion failed: {type(e).__name__}: {str(e)[:300]}")
            tfl = [f for f in os.listdir(outdir) if f.endswith(".tflite")]
            if not tfl:
                raise Refused(f"{backend}: converter wrote no .tflite into {outdir}")
        path = os.path.join(outdir, sorted(tfl)[0])
        delegates = None
        if backend == "litert-gpu":
            lib = os.environ.get("TORCHNATIVE_LITERT_GPU_DELEGATE")
            if not lib:
                raise Refused(f"{backend}: set TORCHNATIVE_LITERT_GPU_DELEGATE to the GPU delegate library path")
            try:
                delegates = [itp.load_delegate(lib)]
            except Exception as e:
                raise Refused(f"{backend}: delegate {lib} not loadable: {type(e).__name__}: {str(e)[:200]}")
        try:
            self.interp = itp.Interpreter(model_path=path, num_threads=threads,
                                          experimental_delegates=delegates)
            self.sigs = self.interp.get_signature_list()
        except Exception as e:
            raise Refused(f"{backend}: interpreter creation failed: {type(e).__name__}: {str(e)[:300]}")
        if "decode" not in self.sigs:
            raise Refused(f"{backend}: no 'decode' signature in {path}; has {sorted(self.sigs)}")
        self.prefill_sigs: Dict[int, str] = {}
        for name, sig in self.sigs.items():
            if name.startswith("prefill"):
                shp = self.interp.get_signature_runner(name).get_input_details()
                if "tokens" in shp:
                    self.prefill_sigs[int(shp["tokens"]["shape"][1])] = name
        self.info = {"tflite": path, "signatures": sorted(self.sigs), "gpu_delegate": bool(delegates),
                     "kv_cache_max_len": kv_max}

    def _zero_kv(self, runner):
        return {k: self.np.zeros(v["shape"], dtype=v["dtype"])
                for k, v in runner.get_input_details().items() if k.startswith("kv_cache")}

    def _step(self, runner, tokens, pos, kv):
        out = runner(tokens=tokens, input_pos=pos, **kv)
        new_kv = {k: v for k, v in out.items() if k.startswith("kv_cache")}
        return out["logits"], new_kv

    def prefill_fn(self, S):
        name = self.prefill_sigs.get(S)
        if name is None:
            raise Refused(f"no prefill signature of length {S}; has {sorted(self.prefill_sigs)}")
        runner = self.interp.get_signature_runner(name)
        np = self.np
        toks = (np.arange(1, S + 1, dtype=np.int32).reshape(1, S) % 4096 + 1).astype(np.int32)
        pos = np.arange(S, dtype=np.int32)
        kv = self._zero_kv(runner)

        def run():
            self._step(runner, toks, pos, kv)
        return run

    def greedy(self, prompt, n):
        np = self.np
        runner = self.interp.get_signature_runner("decode")
        kv = self._zero_kv(runner)
        out: List[int] = []
        logits = None
        for i, t in enumerate(prompt):
            logits, kv = self._step(runner, np.asarray([[t]], dtype=np.int32),
                                    np.asarray([i], dtype=np.int32), kv)
        pos = len(prompt)
        for _ in range(n):
            t = int(np.asarray(logits).reshape(-1, np.asarray(logits).shape[-1])[-1].argmax())
            out.append(t)
            logits, kv = self._step(runner, np.asarray([[t]], dtype=np.int32),
                                    np.asarray([pos], dtype=np.int32), kv)
            pos += 1
        return out


def worker_main(task: Dict[str, Any]) -> Dict[str, Any]:
    os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
    import torch
    assert not hasattr(torch._C, "_aten_implemented"), "competitor worker imported the shim, not upstream torch"
    from transformers import AutoTokenizer
    snap = L.SMOLLM2_SNAP
    tok = AutoTokenizer.from_pretrained(snap)
    prompt = tok(L.VERIFY_PROMPT)["input_ids"]
    backend = task["backend"]
    out: Dict[str, Any] = {"backend": backend, "torch_version": torch.__version__, "is_shim": False}

    if backend == "reference":
        model = _load_reference(torch, snap)
        with torch.no_grad():
            g = model.generate(torch.tensor([prompt]), max_new_tokens=VERIFY_TOKENS, min_new_tokens=VERIFY_TOKENS,
                               do_sample=False)
        out["tokens"] = g[0, len(prompt):].tolist()
        return out

    reps, warm, lens, ndec = int(task["reps"]), 1, task["prompt_lens"], int(task["decode_tokens"])
    threads = int(task["threads"])
    scratch = task["scratch"]
    os.makedirs(scratch, exist_ok=True)
    try:
        if backend in ORT_PROVIDER:
            r: Any = OnnxRunner(torch, snap, backend, threads, scratch)
        else:
            r = LiteRTRunner(torch, snap, backend, threads, scratch, lens, ndec)
        out.update(r.info)
        out["kv_cache"] = r.kv_cache

        prefill: Dict[str, Any] = {}
        for S in lens:
            if isinstance(r, OnnxRunner):
                ids = [(i % 4096) + 1 for i in range(1, S + 1)]
                fn = (lambda ids=ids: r.prefill(ids))
            else:
                fn = r.prefill_fn(S)
            for _ in range(warm):
                fn()
            ms = []
            for _ in range(reps):
                t0 = time.perf_counter()
                fn()
                ms.append((time.perf_counter() - t0) * 1000.0)
            d = _timed_stats(ms, S)
            prefill[str(S)] = {
                "samples_ttft_ms": ms, "stats_ttft_ms": d["stats_ms"], "median_ttft_ms": d["median_ms"],
                "samples_prefill_tok_s": d["samples_tok_s"], "stats_prefill_tok_s": d["stats_tok_s"],
                "median_prefill_tok_s": d["median_tok_s"],
            }
        out["prefill"] = prefill

        r.greedy(prompt, 4)  # warmup
        ms, toks = [], []
        for _ in range(reps):
            t0 = time.perf_counter()
            toks = r.greedy(prompt, ndec)
            ms.append((time.perf_counter() - t0) * 1000.0)
        d = _timed_stats(ms, ndec)
        out["decode"] = {
            "steps": ndec, "includes_prompt_feed_of_tokens": len(prompt),
            "samples_decode_ms": ms, "stats_decode_ms": d["stats_ms"],
            "samples_decode_tok_s": d["samples_tok_s"], "stats_decode_tok_s": d["stats_tok_s"],
            "median_decode_tok_s": d["median_tok_s"],
        }
        out["verify"] = {"tokens_32": toks[:VERIFY_TOKENS], "text_32": tok.decode(toks[:VERIFY_TOKENS])}
    except Refused as e:
        return {"backend": backend, "refused": str(e)}
    return out


# ---------------------------------------------------------------- driver side

def _uptime() -> str:
    return subprocess.run(["uptime"], capture_output=True, text=True).stdout.strip()


def host_info() -> Dict[str, Any]:
    cpu = platform.processor()
    if sys.platform == "darwin":
        r = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
        cpu = r.stdout.strip() or cpu
    return {"platform": platform.platform(), "machine": platform.machine(), "cpu": cpu,
            "cpu_count": os.cpu_count(), "uptime_start": _uptime(), "loadavg_start": os.getloadavg()}


def _run_worker(python_bin: str, task: Dict[str, Any], shim: bool) -> Dict[str, Any]:
    env = dict(os.environ)
    parts = [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and os.path.abspath(p) != L._SHIM_PATH]
    if shim:
        parts = [L._SHIM_PATH] + parts
    env["PYTHONPATH"] = os.pathsep.join(parts)
    for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        env[k] = str(task["threads"])
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HOME"] = L._DEFAULT_HF_HOME
    proc = subprocess.run([python_bin, os.path.abspath(__file__), "--worker", "--worker-task", json.dumps(task)],
                          capture_output=True, text=True, env=env, cwd=L._ROOT)
    lines = [x for x in proc.stdout.splitlines() if x.startswith("@@JSON@@")]
    if proc.returncode != 0 or not lines:
        return {"refused": f"{task['backend']}: worker exit {proc.returncode}: "
                           f"{(proc.stderr or proc.stdout)[-400:].strip()}"}
    return json.loads(lines[-1][len("@@JSON@@"):])


def first_divergence(ref: List[int], got: List[int]) -> Optional[int]:
    for i in range(min(len(ref), len(got))):
        if ref[i] != got[i]:
            return i
    return None if len(ref) == len(got) else min(len(ref), len(got))


def _record_refusal(res: Dict[str, Any], backend: str, why: str) -> None:
    res[backend] = {"refused": why if why.startswith(backend) else f"{backend}: {why}"}
    print(f"SKIP {backend}: {res[backend]['refused'].split(': ', 1)[-1]}")


def parse_args():
    p = argparse.ArgumentParser(description="torchnative vs ONNX Runtime vs LiteRT (bench-only, D11 #283)")
    p.add_argument("--model", choices=["smollm2-135m"], default="smollm2-135m")
    p.add_argument("--backends", default=",".join(ALL_BACKENDS))
    p.add_argument("--prompt-lens", default="128,512")
    p.add_argument("--decode-tokens", type=int, default=128)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--max-load", type=float, default=2.0)
    p.add_argument("--bench-python", default=DEFAULT_BENCH_PY, help="interpreter of .caches/bench-venv")
    p.add_argument("--tn-python", default=DEFAULT_TN_PY, help="interpreter that runs the shim")
    p.add_argument("--out", default=os.path.join("tests", "bench", "results_competitors.json"))
    p.add_argument("--worker", action="store_true")
    p.add_argument("--worker-task", default="")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if args.worker:
        res = worker_main(json.loads(args.worker_task))
        sys.stdout.write("@@JSON@@" + json.dumps(res) + "\n")
        return 0

    load1 = os.getloadavg()[0]
    if load1 > args.max_load:
        sys.stderr.write(f"refusing to start: 1-minute load {load1:.2f} is above --max-load {args.max_load} "
                         f"(measurements run alone, AGENTS.md rule 8)\n")
        return 2
    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    unknown = [b for b in backends if b not in ALL_BACKENDS]
    if unknown:
        sys.stderr.write(f"unknown backends {unknown}; choose from {ALL_BACKENDS}\n")
        return 2
    lens = [int(x) for x in args.prompt_lens.split(",")]
    scratch = os.path.join(SCRATCH, args.model)
    os.makedirs(scratch, exist_ok=True)

    meta: Dict[str, Any] = {
        "root": L._ROOT, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "prompt_lens": lens, "decode_tokens": args.decode_tokens, "repeats": args.repeats,
        "threads": args.threads, "verify_prompt": L.VERIFY_PROMPT, "verify_tokens": VERIFY_TOKENS,
        "reference": "upstream torch float32, transformers generate, greedy, eager attention",
        "host": host_info(), "bench_python": args.bench_python, "tn_python": args.tn_python,
    }
    results: Dict[str, Any] = {}

    need_ref = bool(backends)
    ref_tokens: Optional[List[int]] = None
    if need_ref:
        if not os.path.isfile(args.bench_python):
            meta["reference_refused"] = f"reference: no bench venv at {args.bench_python} (see requirements-competitors.txt)"
            print("SKIP reference: " + meta["reference_refused"])
        else:
            ref = _run_worker(args.bench_python, {"backend": "reference", "threads": args.threads}, shim=False)
            if "refused" in ref:
                meta["reference_refused"] = ref["refused"]
                print("SKIP reference: " + ref["refused"])
            else:
                ref_tokens = ref["tokens"]
                meta["bench_torch_version"] = ref["torch_version"]

    for b in backends:
        if b.startswith("torchnative"):
            dev = b.split("-", 1)[1]
            if lens != L.PREFILL_LENGTHS or args.decode_tokens != L.DECODE_STEPS:
                _record_refusal(results, b, f"bench_llama_cpp's worker fixes prompt lengths {L.PREFILL_LENGTHS} "
                                            f"and {L.DECODE_STEPS} decode steps; got {lens} and {args.decode_tokens}")
                continue
            if not os.path.isfile(args.tn_python):
                _record_refusal(results, b, f"no interpreter at {args.tn_python}")
                continue
            try:
                r = L.run_torchnative_task(
                    args.tn_python,
                    {"model": args.model, "format": "f32", "device": dev, "mode": "dense",
                     "reps": args.repeats, "warmup": 1, "profile": False, "kv_cache": "dynamic"},
                    args.threads, L._DEFAULT_GGUF_VENV)
            except Exception as e:  # noqa: BLE001
                _record_refusal(results, b, f"worker failed: {e}")
                continue
            if not r.get("is_shim"):
                _record_refusal(results, b, "worker did not run the shim (torch._C has no _aten_implemented)")
                continue
            r["kv_cache"] = "dynamic cache (transformers default)"
            toks = r.get("verify", {}).get("tokens_128", [])[:VERIFY_TOKENS]
        else:
            if not os.path.isfile(args.bench_python):
                _record_refusal(results, b, f"no bench venv interpreter at {args.bench_python}")
                continue
            r = _run_worker(args.bench_python, {
                "backend": b, "reps": args.repeats, "prompt_lens": lens, "decode_tokens": args.decode_tokens,
                "threads": args.threads, "scratch": scratch}, shim=False)
            if "refused" in r:
                _record_refusal(results, b, r["refused"])
                continue
            toks = r.get("verify", {}).get("tokens_32", [])
        if ref_tokens is not None:
            div = first_divergence(ref_tokens, toks)
            r["agreement"] = {"grade": "agrees" if div is None else "diverges", "first_divergence_index": div,
                              "compared_tokens": min(len(ref_tokens), len(toks)),
                              "oracle": "upstream torch float32 greedy"}
        else:
            r["agreement"] = {"grade": "not measured: reference unavailable"}
        results[b] = r
        print(f"done {b}: " + (f"decode median tok/s {r['decode'].get('median_decode_tok_s')}"
                                 if "decode" in r else "no decode"))

    meta["host"]["uptime_end"] = _uptime()
    meta["host"]["loadavg_end"] = os.getloadavg()
    meta["reference_tokens"] = ref_tokens
    out_path = args.out if os.path.isabs(args.out) else os.path.join(L._ROOT, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"metadata": meta, "models": {args.model: results}}, f, indent=2)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
