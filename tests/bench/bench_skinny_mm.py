"""A/B the decode-shaped `aten.mm` against upstream torch and against the
shim's own general path, at SmolLM2-135M's projection shapes.

Bench-only, never in the gate (AGENTS.md rule 8: measurements run alone).
Written for the CI runner where the gap was measured (Gemstone #74, run
44792524320: 90.8 ms of a 122.9 ms torchnative decode step in `aten.mm`, 211
calls per step), so that the fix in `torchnative/rust/torch_c/src/skinny_gemm.rs`
is measured where it was motivated and not on the Accelerate build, where it
is not routed. docs/perf/MM_DECODE.md.

Three sides, each in its own subprocess, interleaved per round so machine
drift cannot land on one side (the shape of `ab_upstream.py`):

  upstream   the real torch on PATH (no PYTHONPATH)
  shim       the vendored tree, kernel on (the shipped default)
  shim-off   the vendored tree with TORCHNATIVE_SKINNY_GEMM=0: the general
             path the kernel replaced, same binary

Every item is `torch.ops.aten.mm.default(x, w.t())`, the literal operands
`bootstrap.py::linear` hands the kernel after the fold (`t, view, mm,
_unsafe_view`), at the five weight shapes of SmolLM2-135M and the two row
counts of the breakdown (batch 1 and batch 4, one token each). The shim sides
also report `torch._C._skinny_gemm_counters()` deltas, so the artefact says
which kernel the timed calls took rather than leaving it to the numbers.

`per-step estimate` is the sum over the five shapes of the per-call median
times the number of calls one decode step makes (30 layers: 60 q/o_proj,
60 k/v_proj, 60 gate/up_proj, 30 down_proj, 1 lm_head = 211). It is an
estimate from isolated calls, not a decode step; the step itself is
`ab_upstream.py --suite model`.

Run (CI; locally only when told to, alone, after `uptime`)::

    PYTHONPATH= python tests/bench/bench_skinny_mm.py --python "$(which python)" \\
        --rounds 3 --out "$RUNNER_TEMP/skinny_mm.json"

    # sweep the split floor without a rebuild (bytes of weight per call)
    ... --thread-bytes 0,262144,1048576,4194304

Reporting: median of the pooled timed repetitions with min and max, the load
average around every worker run, and the counter deltas. No U+2014 anywhere.
"""

import argparse
import json
import os
import statistics
import subprocess
import sys
import time

REPO = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "AGENTS.md").is_file()))
SHIM_PATH = os.path.join(REPO, "torchnative", "python")

WARMUP = 10
REPS = 40

# (name, out_features n, in_features k, calls per decode step of SmolLM2-135M)
SHAPES = [
    ("q/o_proj 576x576", 576, 576, 60),
    ("k/v_proj 192x576", 192, 576, 60),
    ("gate/up 1536x576", 1536, 576, 60),
    ("down 576x1536", 576, 1536, 30),
    ("lm_head 49152x576", 49152, 576, 1),
]
ROWS = (1, 4)


def loadavg():
    return os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)


# --------------------------------------------------------------------------
# worker side
# --------------------------------------------------------------------------


def _time(fn, warmup, reps):
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append((time.perf_counter() - t0) * 1000.0)
    return out


def worker():
    import torch

    is_shim = hasattr(torch._C, "_aten_implemented")
    counters = getattr(torch._C, "_skinny_gemm_counters", None)
    aten = torch.ops.aten
    torch.manual_seed(0)
    data = {}
    taken = {}
    for name, n, k, calls in SHAPES:
        w = torch.randn(n, k)
        wt = w.t()
        for rows in ROWS:
            x = torch.randn(rows, k)
            key = f"{name} rows={rows}"
            before = counters()["calls"] if counters else None
            # lm_head is the one shape that is slow enough to need fewer reps.
            reps = REPS if n < 10000 else max(10, REPS // 4)
            data[key] = {"t": _time(lambda: aten.mm.default(x, wt), WARMUP, reps),
                         "calls_per_step": calls, "rows": rows}
            if counters:
                taken[key] = counters()["calls"] - before
    out = {
        "torch": torch.__version__,
        "file": torch.__file__,
        "shim": is_shim,
        "threads": torch.get_num_threads() if hasattr(torch, "get_num_threads") else None,
        "counters": counters() if counters else None,
        "taken": taken,
        "data": data,
    }
    print("@@JSON@@" + json.dumps(out))


# --------------------------------------------------------------------------
# driver side
# --------------------------------------------------------------------------


def run_side(python, side, thread_bytes=None):
    env = dict(os.environ)
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    if side == "upstream":
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = SHIM_PATH
    if side == "shim-off":
        env["TORCHNATIVE_SKINNY_GEMM"] = "0"
    if thread_bytes is not None:
        env["TORCHNATIVE_SKINNY_GEMM_THREAD_BYTES"] = str(thread_bytes)
    before = loadavg()
    p = subprocess.run([python, os.path.abspath(__file__), "--worker"],
                       env=env, capture_output=True, text=True, cwd=REPO)
    after = loadavg()
    if p.returncode != 0:
        sys.stderr.write(p.stdout[-4000:] + "\n" + p.stderr[-4000:] + "\n")
        raise SystemExit("%s worker failed (exit %d)" % (side, p.returncode))
    line = [l for l in p.stdout.splitlines() if l.startswith("@@JSON@@")]
    if not line:
        sys.stderr.write(p.stdout[-4000:] + "\n")
        raise SystemExit("%s worker produced no result" % side)
    res = json.loads(line[-1][len("@@JSON@@"):])
    if side == "upstream" and res["shim"]:
        raise SystemExit("the upstream side imported the shim: check PYTHONPATH")
    if side != "upstream" and not res["shim"]:
        raise SystemExit("the shim side imported upstream torch: vendor the tree first (AGENTS.md §14.1)")
    res["load_before"] = before
    res["load_after"] = after
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--python", default=os.environ.get("PYTHON", sys.executable))
    ap.add_argument("--sides", default="upstream,shim,shim-off")
    ap.add_argument("--thread-bytes", default="",
                    help="comma-separated TORCHNATIVE_SKINNY_GEMM_THREAD_BYTES values; "
                         "adds one 'shim@<bytes>' side per value")
    ap.add_argument("--out", default="", help="write the pooled results as JSON here")
    args = ap.parse_args()

    if args.worker:
        worker()
        return

    sides = [(s, None) for s in args.sides.split(",") if s]
    for raw in [v for v in args.thread_bytes.split(",") if v]:
        sides.append((f"shim@{int(raw)}", int(raw)))

    pooled = {label: {} for label, _ in sides}
    meta = {}
    loads = []
    for r in range(args.rounds):
        for label, thread_bytes in sides:
            side = "shim" if label.startswith("shim@") else label
            res = run_side(args.python, side, thread_bytes)
            meta[label] = {k: res[k] for k in ("torch", "file", "threads", "counters", "taken")}
            loads.append((r + 1, label, res["load_before"][0], res["load_after"][0]))
            for k, v in res["data"].items():
                pooled[label].setdefault(k, []).extend(v["t"])

    print("repo:    %s" % REPO)
    print("python:  %s" % args.python)
    print("rounds:  %d   warmup %d, reps %d per round and item (lm_head: reps/4)" % (args.rounds, WARMUP, REPS))
    for label, _ in sides:
        m = meta[label]
        c = m["counters"]
        where = ("skinny routed=%s threads=%s thread_bytes=%s simd=%s" %
                 (c["routed"], c["threads"], c["thread_bytes"], c["simd"])) if c else "upstream"
        print("%-12s torch %s threads=%s  %s" % (label, m["torch"], m["threads"], where))
    print()
    print("load average (1 min) around each worker run:")
    for r, label, b, a in loads:
        print("  round %d %-12s before %.2f  after %.2f" % (r, label, b, a))
    lo = min(min(b, a) for _, _, b, a in loads)
    hi = max(max(b, a) for _, _, b, a in loads)
    print("  range %.2f -- %.2f%s" % (lo, hi, "   *** load moved, treat as void" if hi - lo > 2.0 else ""))
    print()

    labels = [label for label, _ in sides]
    header = "%-28s" % "ms per call, median [min-max]"
    for label in labels:
        header += " %22s" % label
    header += "  taken(shim)"
    print(header)
    print("-" * len(header))
    keys = list(pooled[labels[0]].keys())
    steps = {label: {rows: 0.0 for rows in ROWS} for label in labels}
    for key in keys:
        line = "%-28s" % key
        for label in labels:
            t = sorted(pooled[label].get(key, []))
            if not t:
                line += " %22s" % "-"
                continue
            med = statistics.median(t)
            line += " %8.4f [%6.4f-%7.4f]" % (med, t[0], t[-1])
            rows = int(key.rsplit("rows=", 1)[1])
            calls = next(c for name, _, _, c in SHAPES if key.startswith(name))
            steps[label][rows] += med * calls
        taken = meta.get("shim", {}).get("taken", {}).get(key)
        per_round = WARMUP + (REPS if not key.startswith("lm_head") else max(10, REPS // 4))
        line += "  %s" % ("" if taken is None else "%d/%d per round" % (taken, per_round))
        print(line)
    print()
    print("per-step estimate (sum of median x calls per step, 211 mm calls), ms:")
    for rows in ROWS:
        line = "  batch %d:" % rows
        for label in labels:
            line += "  %s %.2f" % (label, steps[label][rows])
        if "upstream" in labels:
            for label in labels:
                if label != "upstream" and steps["upstream"][rows]:
                    line += "  (%s/upstream %.2fx)" % (label, steps[label][rows] / steps["upstream"][rows])
        print(line)
    print()
    print("'taken(shim)' is the skinny-kernel call count over the last round's warmup and timed")
    print("calls of the shim side; a shape whose count is 0 was measured on the general path.")

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"meta": meta, "loads": loads, "pooled": pooled, "steps": steps,
                       "shapes": SHAPES, "rows": ROWS, "warmup": WARMUP, "reps": REPS}, f)
        print("written: %s" % args.out)


if __name__ == "__main__":
    main()
