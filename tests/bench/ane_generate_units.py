"""SmolLM2-135M `generate` through `to(device.npu)`, CPU_ONLY against CPU_AND_NE.

Evidence that does not come from `MLComputePlan`: the same lowered model is
built twice, once with `compute_units=CPU_ONLY` and once with `CPU_AND_NE`,
and each decodes the same greedy tokens in a fresh subprocess. If the Neural
Engine runs the group programs, the CPU_AND_NE run spends less process CPU
time per token (the work leaves the CPU cores) even when wall time is similar.

Run alone, after checking `uptime` (AGENTS.md rule 8):

    PYTHONPATH=<stage>:tests/_support .caches/spike-venv/bin/python \
        tests/bench/ane_generate_units.py [rounds]

For the direct reading, run this alongside a CPU_AND_NE round (needs sudo):

    sudo powermetrics --samplers ane_power,cpu_power -i 500 -n 20
"""

import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_VENDOR = os.path.join(_HERE, "..", "..", "torchnative", "python")

_SCRIPT = r"""
import io, json, os, sys, time, warnings
_stdout = os.fdopen(os.dup(1), "w")
_sink = os.open(os.devnull, os.O_WRONLY); os.dup2(_sink, 1); os.close(_sink)
sys.stdout = io.StringIO()
warnings.simplefilter("ignore")
import torch
assert hasattr(torch._C, "_aten_implemented")
import coremltools as ct
from torchnative.export import coreml as C
from transformers import AutoModelForCausalLM, AutoTokenizer
name = "HuggingFaceTB/SmolLM2-135M"
tok = AutoTokenizer.from_pretrained(name)
m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32).eval()
# The internal entry `to(device.npu)` calls, with compute_units exposed.
m, _ = C._compile_model(m, compute_units=getattr(ct.ComputeUnit, sys.argv[1]))
ids = tok("The capital of France is", return_tensors="pt").input_ids
with torch.no_grad():
    m.generate(ids, max_new_tokens=4, do_sample=False)  # warm the programs
    n = 32
    w0, c0 = time.perf_counter(), time.process_time()
    g = m.generate(ids, max_new_tokens=n, do_sample=False)
    w1, c1 = time.perf_counter(), time.process_time()
print(json.dumps({"units": sys.argv[1], "tokens": g[0, ids.shape[1]:].tolist(),
                  "wall_ms_per_token": 1000 * (w1 - w0) / n,
                  "cpu_ms_per_token": 1000 * (c1 - c0) / n}), file=_stdout, flush=True)
"""


def _one(units):
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.abspath(_VENDOR)
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run([sys.executable, "-c", _SCRIPT, units],
                          capture_output=True, text=True, env=env, timeout=1800)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr[-2000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main():
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    print("uptime:", os.popen("uptime").read().strip())
    rows = []
    for _ in range(rounds):
        for units in ("CPU_ONLY", "CPU_AND_NE"):
            r = _one(units)
            rows.append(r)
            print(f"{r['units']:<11} wall {r['wall_ms_per_token']:7.2f} ms/token  "
                  f"cpu {r['cpu_ms_per_token']:7.2f} ms/token")
    same = len({tuple(r["tokens"]) for r in rows}) == 1
    print("same tokens across every run:", same)


if __name__ == "__main__":
    main()
