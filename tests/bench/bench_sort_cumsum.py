"""A/B `aten.cumsum.default` and `aten.sort.default` against upstream torch on
the sampling shape: [4, 49152] float32 rows (Gemstone #74: top-p and draw call
both on a vocabulary row every decode step).

Bench-only, never in the gate (AGENTS.md rule 8: measurements run alone).
The CPU `cumsum` used to be one candle kernel per element along the dimension
(about 49 thousand launches per row); it is now one host scan. `sort` orders
the rows across the crate's thread pool, with the same per-row algorithm.

Two sides, each in its own subprocess, interleaved per round so machine drift
cannot land on one side (the shape of `bench_skinny_mm.py`):

  upstream   the real torch on PATH (no PYTHONPATH)
  shim       the vendored tree

Items: cumsum dim=-1, cumsum dim=0 (strided lanes), sort ascending,
sort descending, sort.stable(stable=True), each on the `[4, 49152]` row set
and on one `[1, 49152]` row (the single-lane case, which cannot be split).

Run (CI; locally only when told to, alone, after `uptime`)::

    PYTHONPATH= python tests/bench/bench_sort_cumsum.py --python "$(which python)" \\
        --rounds 3 --out "$RUNNER_TEMP/sort_cumsum.json"

Reporting: median of the pooled timed repetitions with min and max, and the
load average around every worker run. No U+2014 anywhere.
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

WARMUP = 5
REPS = 30
ROWS = (4, 1)
WIDTH = 49152


def loadavg():
    return os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)


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

    aten = torch.ops.aten
    torch.manual_seed(0)
    data = {}
    for rows in ROWS:
        x = torch.randn(rows, WIDTH)
        items = {
            f"cumsum dim=-1 [{rows}, {WIDTH}]": lambda x=x: aten.cumsum.default(x, -1),
            f"cumsum dim=0 [{rows}, {WIDTH}]": lambda x=x: aten.cumsum.default(x, 0),
            f"sort asc [{rows}, {WIDTH}]": lambda x=x: aten.sort.default(x, -1, False),
            f"sort desc [{rows}, {WIDTH}]": lambda x=x: aten.sort.default(x, -1, True),
            f"sort stable=True [{rows}, {WIDTH}]": lambda x=x: aten.sort.stable(x, stable=True, dim=-1, descending=False),
        }
        for key, fn in items.items():
            data[key] = _time(fn, WARMUP, REPS)
    out = {
        "torch": torch.__version__,
        "file": torch.__file__,
        "shim": hasattr(torch._C, "_aten_implemented"),
        "threads": torch.get_num_threads() if hasattr(torch, "get_num_threads") else None,
        "data": data,
    }
    print("@@JSON@@" + json.dumps(out))


def run_side(python, side):
    env = dict(os.environ)
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    if side == "upstream":
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = SHIM_PATH
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
    ap.add_argument("--sides", default="upstream,shim")
    ap.add_argument("--out", default="", help="write the pooled results as JSON here")
    args = ap.parse_args()

    if args.worker:
        worker()
        return

    sides = [s for s in args.sides.split(",") if s]
    pooled = {s: {} for s in sides}
    meta = {}
    loads = []
    for r in range(args.rounds):
        for side in sides:
            res = run_side(args.python, side)
            meta[side] = {k: res[k] for k in ("torch", "file", "threads")}
            loads.append((r + 1, side, res["load_before"][0], res["load_after"][0]))
            for k, v in res["data"].items():
                pooled[side].setdefault(k, []).extend(v)

    print("repo:    %s" % REPO)
    print("python:  %s" % args.python)
    print("rounds:  %d   warmup %d, reps %d per round and item" % (args.rounds, WARMUP, REPS))
    for side in sides:
        print("%-10s torch %s threads=%s" % (side, meta[side]["torch"], meta[side]["threads"]))
    print()
    for r, side, b, a in loads:
        print("  round %d %-10s load before %.2f  after %.2f" % (r, side, b, a))
    lo = min(min(b, a) for _, _, b, a in loads)
    hi = max(max(b, a) for _, _, b, a in loads)
    print("  range %.2f -- %.2f%s" % (lo, hi, "   *** load moved, treat as void" if hi - lo > 2.0 else ""))
    print()

    header = "%-34s" % "ms per call, median [min-max]"
    for side in sides:
        header += " %24s" % side
    if "upstream" in sides and "shim" in sides:
        header += "  shim/upstream"
    print(header)
    print("-" * len(header))
    for key in pooled[sides[0]]:
        line = "%-34s" % key
        meds = {}
        for side in sides:
            t = sorted(pooled[side].get(key, []))
            if not t:
                line += " %24s" % "-"
                continue
            meds[side] = statistics.median(t)
            line += " %9.3f [%6.3f-%8.3f]" % (meds[side], t[0], t[-1])
        if "upstream" in meds and "shim" in meds and meds["upstream"]:
            line += "  %.2fx" % (meds["shim"] / meds["upstream"])
        print(line)

    if args.out:
        with open(args.out, "w") as f:
            json.dump({"meta": meta, "loads": loads, "pooled": pooled,
                       "warmup": WARMUP, "reps": REPS}, f)
        print("written: %s" % args.out)


if __name__ == "__main__":
    main()
