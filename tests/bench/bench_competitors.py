"""Compare torchnative with ONNX Runtime and LiteRT on one model and one host.

Bench-only (leader decision D11, issue #283). The result format follows
bench_llama_cpp.py (`models -> <model> -> <backend> -> prefill / decode / verify`,
the same `median_*` key names), so the two files can be read side by side.

Backends (a requested backend that is absent is recorded as {"refused": "<name>: <reason>"}
and printed as `SKIP <backend>: <reason>`; CPU is never substituted for an accelerator):
  torchnative-cpu, torchnative-mps   the shim, run through bench_llama_cpp's own worker
  onnxruntime-cpu / -coreml / -qnn   upstream torch's dynamo exporter -> ORT session
  litert-cpu / litert-gpu            litert_torch.convert -> .tflite -> ai_edge_litert Interpreter

One static-shape scheme for ONNX and LiteRT (no KV cache, labelled in every result):
  * A wrapper `input_ids[1, N] -> logits[1, N, V]` is converted once per N and cached under
    .scratch/bench-competitors/ keyed by model, N and package versions.
  * Prefill: one unpadded graph with N = prompt length, timed on S real tokens (comparable with
    torchnative and llama.cpp prefill, which also compute all positions).
  * Decode: N = len(prompt) + decode_tokens. The prompt is right-padded to N and the logits at
    position t give the next token; every step recomputes the whole fixed-length sequence.
    `kv_cache: "none (fixed-length full recompute)"`. These decode numbers are therefore NOT
    comparable with KV-cached decode (torchnative, llama.cpp) and are recorded for completeness.
  * Causal attention means positions after t cannot influence position t. This is asserted once
    per conversion: the padded run's logits[t] are compared with an unpadded upstream float32
    run, and the backend refuses by name if they differ by more than 1e-4 relative.
  * ONNX: `torch.onnx.export(dynamo=True)` with static shapes, then an ORT session limited to the
    requested provider. CoreML and QNN always get CPU appended by ORT for unsupported nodes: the
    harness refuses unless the requested provider is first in `session.get_providers()`, but cannot
    see the node partition (`partition: "unknown"`).
  * LiteRT: `litert_torch.convert(wrapper, (input_ids,))` (the plain converter; the generative API
    imports tensorflow, which is not approved), `LiteRTModel.export(<.tflite>)`, and timing with
    ai_edge_litert's Interpreter (set_tensor, invoke, get_tensor) to avoid Python per-op overhead.
    litert-cpu is the Interpreter's default CPU path (XNNPACK, from its log line). litert-gpu
    refuses unless TORCHNATIVE_LITERT_GPU_DELEGATE names a library that `load_delegate` loads.
  * torchnative runs through bench_llama_cpp's own worker (dense f32, dynamic KV cache).

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


class StaticRunner:
    """One static-shape scheme for ONNX and LiteRT: input_ids[1, N] -> logits[1, N, V].

    `graph(N)` converts (or loads from the cache) one graph per N and returns
    `run(ids_int_array_of_length_N) -> logits ndarray [1, N, V]`. No KV cache.
    """
    kv_cache = "none (fixed-length full recompute)"

    def __init__(self, torch, snap, backend, threads, scratch):
        import numpy as np
        self.np, self.torch, self.snap = np, torch, snap
        self.backend, self.threads, self.scratch = backend, threads, scratch
        self._model = None
        self._graphs: Dict[int, Any] = {}
        self.info: Dict[str, Any] = {"files": {}}

    def model(self):
        if self._model is None:
            self._model = _load_reference(self.torch, self.snap)
        return self._model

    def wrapper(self):
        torch = self.torch
        inner = self.model()

        class Wrap(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.m = inner

            def forward(self, input_ids):
                return self.m(input_ids, use_cache=False).logits
        return Wrap().eval()

    def sample(self, N):
        return self.torch.arange(1, N + 1, dtype=self.torch.int64).reshape(1, N) % 4096

    def graph(self, N):
        if N not in self._graphs:
            self._graphs[N] = self._build(N)
        return self._graphs[N]

    def _build(self, N):
        raise NotImplementedError


class OnnxRunner(StaticRunner):
    def __init__(self, torch, snap, backend, threads, scratch):
        super().__init__(torch, snap, backend, threads, scratch)
        try:
            import onnxruntime as ort
            import onnx
            import onnxscript  # noqa: F401
        except ImportError as e:
            raise Refused(f"{backend}: onnxruntime, onnx and onnxscript are needed ({e})")
        self.ort = ort
        self.want = ORT_PROVIDER[backend]
        avail = ort.get_available_providers()
        if self.want not in avail:
            raise Refused(f"{backend}: {self.want} not in onnxruntime.get_available_providers() {avail}")
        self.key = f"torch{torch.__version__}-onnx{onnx.__version__}"
        self.info.update({"onnxruntime": ort.__version__, "onnx": onnx.__version__, "partition": "unknown",
                          "export": "torch.onnx.export(dynamo=True), static shapes"})

    def _build(self, N):
        path = os.path.join(self.scratch, f"onnx-smollm2-135m-N{N}-{self.key}.onnx")
        if not os.path.isfile(path):
            try:
                prog = self.torch.onnx.export(self.wrapper(), (self.sample(N),), dynamo=True,
                                              input_names=["input_ids"], output_names=["logits"])
                prog.save(path)
            except Exception as e:  # exporter walls are findings, not crashes
                raise Refused(f"{self.backend}: torch.onnx.export(dynamo=True) failed at N={N}: "
                              f"{type(e).__name__}: {str(e)[:300]}")
        so = self.ort.SessionOptions()
        so.intra_op_num_threads = self.threads
        opts: List[Dict[str, str]] = [{}]
        if self.backend == "onnxruntime-qnn":
            be = os.environ.get("TORCHNATIVE_QNN_BACKEND")
            if not be:
                raise Refused(f"{self.backend}: set TORCHNATIVE_QNN_BACKEND to the QNN backend library path")
            opts = [{"backend_path": be}]
        try:
            sess = self.ort.InferenceSession(path, so, providers=[self.want], provider_options=opts)
        except Exception as e:
            raise Refused(f"{self.backend}: session with only {self.want} failed: {type(e).__name__}: {str(e)[:300]}")
        used = sess.get_providers()
        if not used or used[0] != self.want:
            raise Refused(f"{self.backend}: session used {used}, not {self.want} first")
        self.info["files"][str(N)] = path
        self.info["providers_used"] = used
        np = self.np
        return lambda ids: sess.run(["logits"], {"input_ids": np.asarray(ids, dtype=np.int64).reshape(1, N)})[0]


class LiteRTRunner(StaticRunner):
    """litert_torch.convert (plain converter; the generative API needs tensorflow, not approved),
    exported to a .tflite in .scratch and run with ai_edge_litert's Interpreter (no Python
    per-op overhead; set_tensor, invoke, get_tensor per forward)."""

    def __init__(self, torch, snap, backend, threads, scratch):
        super().__init__(torch, snap, backend, threads, scratch)
        try:
            from ai_edge_litert import interpreter as itp
            import litert_torch
            from importlib import metadata
        except ImportError as e:
            raise Refused(f"{backend}: ai_edge_litert and litert_torch are needed ({e})")
        self.itp, self.lt = itp, litert_torch
        self.delegates = None
        if backend == "litert-gpu":
            lib = os.environ.get("TORCHNATIVE_LITERT_GPU_DELEGATE")
            if not lib:
                raise Refused(f"{backend}: no GPU delegate: set TORCHNATIVE_LITERT_GPU_DELEGATE to its library path")
            try:
                self.delegates = [itp.load_delegate(lib)]
            except Exception as e:
                raise Refused(f"{backend}: GPU delegate {lib} not loadable: {type(e).__name__}: {str(e)[:200]}")
            self.info["gpu_delegate_library"] = lib
        v = metadata.version
        self.key = f"torch{torch.__version__}-litert_torch{v('litert-torch')}"
        self.info.update({"litert_torch": v("litert-torch"), "ai_edge_litert": v("ai-edge-litert"),
                          "conversion": "litert_torch.convert(wrapper, (input_ids,)) then LiteRTModel.export(.tflite)",
                          "runtime": "ai_edge_litert.interpreter.Interpreter on the exported .tflite",
                          "cpu_path": ("XNNPACK CPU delegate is the Interpreter default (log line 'Created TensorFlow "
                                       "Lite XNNPACK delegate for CPU'); not asserted by this harness")
                          if backend == "litert-cpu" else None})

    def _build(self, N):
        path = os.path.join(self.scratch, f"litert-smollm2-135m-N{N}-{self.key}.tflite")
        if not os.path.isfile(path):
            try:
                em = self.lt.convert(self.wrapper(), (self.sample(N),))
                em.export(path)
            except Exception as e:
                raise Refused(f"{self.backend}: litert_torch.convert failed at N={N}: {type(e).__name__}: {str(e)[:300]}")
        try:
            it = self.itp.Interpreter(model_path=path, num_threads=self.threads,
                                      experimental_delegates=self.delegates)
            it.allocate_tensors()
        except Exception as e:
            raise Refused(f"{self.backend}: Interpreter creation failed: {type(e).__name__}: {str(e)[:300]}")
        inp, outp = it.get_input_details()[0], it.get_output_details()[0]
        self.info["files"][str(N)] = path
        self.info["input_dtype"] = str(inp["dtype"])
        np = self.np

        def run(ids):
            it.set_tensor(inp["index"], np.asarray(ids).reshape(1, N).astype(inp["dtype"]))
            it.invoke()
            return it.get_tensor(outp["index"])
        return run


def check_padding(r, N, prompt, torch):
    """Causal attention means positions after t cannot change logits[t]. Assert it once per
    conversion: padded run at N versus the unpadded upstream float32 run."""
    np = r.np
    t = len(prompt) - 1
    padded = np.zeros(N, dtype=np.int64)
    padded[: len(prompt)] = prompt
    got = np.asarray(r.graph(N)(padded))[0, t]
    with torch.no_grad():
        want = r.model()(torch.tensor([prompt]), use_cache=False).logits[0, -1].numpy()
    rel = float(np.abs(got - want).max() / max(float(np.abs(want).max()), 1e-30))
    r.info["padding_check"] = {"N": N, "position": t, "max_rel_diff": rel, "limit": 1e-4}
    if rel > 1e-4:
        raise Refused(f"{r.backend}: logits[{t}] of the padded run at N={N} differ from upstream unpadded "
                      f"by relative {rel:.2e} (> 1e-4)")


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

    reps, lens, ndec = int(task["reps"]), task["prompt_lens"], int(task["decode_tokens"])
    scratch = task["scratch"]
    os.makedirs(scratch, exist_ok=True)
    try:
        cls = OnnxRunner if backend in ORT_PROVIDER else LiteRTRunner
        r = cls(torch, snap, backend, int(task["threads"]), scratch)
        np = r.np
        out["kv_cache"] = r.kv_cache

        prefill: Dict[str, Any] = {}
        for S in lens:  # one unpadded graph of exactly S tokens per prompt length
            run = r.graph(S)
            ids = np.asarray((np.arange(1, S + 1) % 4096), dtype=np.int64)
            run(ids)  # warmup
            ms = []
            for _ in range(reps):
                t0 = time.perf_counter()
                run(ids)
                ms.append((time.perf_counter() - t0) * 1000.0)
            d = _timed_stats(ms, S)
            prefill[str(S)] = {
                "samples_ttft_ms": ms, "stats_ttft_ms": d["stats_ms"], "median_ttft_ms": d["median_ms"],
                "samples_prefill_tok_s": d["samples_tok_s"], "stats_prefill_tok_s": d["stats_tok_s"],
                "median_prefill_tok_s": d["median_tok_s"],
            }
        out["prefill"] = prefill

        N = len(prompt) + ndec  # decode graph: right-padded, read the logits at position t
        check_padding(r, N, prompt, torch)
        run = r.graph(N)

        def greedy():
            seq = np.zeros(N, dtype=np.int64)
            seq[: len(prompt)] = prompt
            t, toks = len(prompt) - 1, []
            for _ in range(ndec):
                nxt = int(np.asarray(run(seq))[0, t].argmax())
                toks.append(nxt)
                t += 1
                seq[t] = nxt
            return toks

        run(np.zeros(N, dtype=np.int64))  # warmup
        ms, toks = [], []
        for _ in range(reps):
            t0 = time.perf_counter()
            toks = greedy()
            ms.append((time.perf_counter() - t0) * 1000.0)
        d = _timed_stats(ms, ndec)
        out["decode"] = {
            "steps": ndec, "graph_length_N": N, "comparable_to_kv_cached_decode": False,
            "samples_decode_ms": ms, "stats_decode_ms": d["stats_ms"],
            "samples_decode_tok_s": d["samples_tok_s"], "stats_decode_tok_s": d["stats_tok_s"],
            "median_decode_tok_s": d["median_tok_s"],
        }
        out["verify"] = {"tokens_32": toks[:VERIFY_TOKENS], "text_32": tok.decode(toks[:VERIFY_TOKENS])}
        out.update(r.info)
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
            div = first_divergence(ref_tokens[: len(toks)], toks)
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
