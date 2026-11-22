"""Per-operator dispatch cost of the decode step's cheapest ops: upstream torch
against the shim, in nanoseconds per call, plus what the shim's Python binding
layer (`bootstrap.py`'s `resolve`) is of that.

Why this exists. Gemstone run 44874740472 (Intel Xeon 8573C runner, 3.15t,
SmolLM2-135M batch-4 decode) timed every operator of one step under the same
`TorchDispatchMode` on both substrates: `aten.view` 361 calls cost 0.80 ms on
upstream and 2.9 ms on torchnative (2.2 vs 8.0 us each), `aten.mul` 275 calls
0.87 vs 2.4 ms. About 6 us more per op over roughly 1500 ops is about 9 ms of
the step at either batch size, the third line after the matmul
(docs/perf/MM_DECODE.md) and attention (docs/perf/SDPA_DECODE.md).
docs/perf/DISPATCH2.md measured the same gap on the M1 at 0.7 us (1.5 vs 0.8)
and left a per-schema positional fast path for a later round. This file is
the measurement that round needs on the runner where the ratio is 4x, not 2x.

Not part of the gate. Run (CI; locally only when told to, alone)::

    PYTHONPATH= python tests/bench/bench_dispatch.py --python "$(which python)" --rounds 3

Each side runs in its own interpreter (`upstream` with no `PYTHONPATH`, `shim`
with the source tree). Every op is called through `torch.ops.aten.<op>.<overload>`,
the spelling the model's decomposed step uses, and once through the method
spelling for comparison. The shim side also profiles one mixed loop with
`cProfile` and reports the share of `resolve` and the top functions by own
time, so the next change can be aimed. No U+2014 anywhere.
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

WARMUP = 2000
REPS = 20000
CHUNK = 1000


def loadavg():
    return os.getloadavg() if hasattr(os, "getloadavg") else (0.0, 0.0, 0.0)


def _ns_per_call(fn, warmup=WARMUP, reps=REPS, chunk=CHUNK):
    """Median over chunks of `chunk` calls of the per-call time in ns."""
    for _ in range(warmup):
        fn()
    out = []
    for _ in range(reps // chunk):
        t0 = time.perf_counter_ns()
        for _ in range(chunk):
            fn()
        out.append((time.perf_counter_ns() - t0) / chunk)
    return out


def _ops(torch):
    aten = torch.ops.aten
    x = torch.randn(4, 576)
    y = torch.randn(4, 576)
    w = torch.randn(576, 576)
    h = torch.randn(4, 1, 9, 64)
    # (name, callable) in the spelling the decomposed decode step uses.
    return [
        ("view (4,576)->(1,4,576)", lambda: aten.view.default(x, [1, 4, 576])),
        ("_unsafe_view (4,576)->(4,1,576)", lambda: aten._unsafe_view.default(x, [4, 1, 576])),
        ("t (576,576)", lambda: aten.t.default(w)),
        ("transpose (4,1,9,64) 1,2", lambda: aten.transpose.int(h, 1, 2)),
        ("slice (4,576)[:, :64]", lambda: aten.slice.Tensor(x, 1, 0, 64)),
        ("unsqueeze (4,576) 1", lambda: aten.unsqueeze.default(x, 1)),
        ("mul (4,576)*(4,576)", lambda: aten.mul.Tensor(x, y)),
        ("add (4,576)+(4,576)", lambda: aten.add.Tensor(x, y)),
        ("mul scalar (4,576)*0.5", lambda: aten.mul.Tensor(x, 0.5)),
        ("method x.view(1,4,576)", lambda: x.view(1, 4, 576)),
        ("method x.transpose(0,1)", lambda: x.transpose(0, 1)),
        ("method x * y", lambda: x * y),
    ]


def worker():
    import cProfile
    import pstats

    import torch

    is_shim = hasattr(torch._C, "_aten_implemented")
    data = {}
    for name, fn in _ops(torch):
        data[name] = _ns_per_call(fn)

    # The same ops under a pass-through TorchDispatchMode: what Gemstone's
    # per-operator timer (benchmarks/batching_breakdown.py::time_operators)
    # adds to every op on each substrate. Production decode runs no mode, so
    # the difference between this and the row above is the timer's own
    # inflation, not a cost of the step; on the shim the mode path is several
    # Python calls per op (the active-mode check, the suppression check, the
    # OpOverload call), on upstream one.
    under_mode = {}
    try:
        from torch.utils._python_dispatch import TorchDispatchMode

        class PassThrough(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                return func(*args, **(kwargs or {}))

        with PassThrough():
            for name, fn in _ops(torch):
                under_mode[name] = _ns_per_call(fn, warmup=WARMUP // 4, reps=REPS // 4)
    except Exception as exc:  # noqa: BLE001 -- reported, the plain rows stand
        under_mode = {"error": repr(exc)[:300]}

    # One mixed loop under cProfile: the share of the binding layer by name.
    ops = _ops(torch)
    prof = cProfile.Profile()
    prof.enable()
    for _ in range(2000):
        for _, fn in ops:
            fn()
    prof.disable()
    stats = pstats.Stats(prof)
    total = stats.total_tt
    rows = []
    for (file, line, func), (cc, nc, tt, ct, callers) in stats.stats.items():
        rows.append({"func": func, "file": os.path.basename(file), "line": line,
                     "calls": nc, "own_ms": tt * 1000.0, "cum_ms": ct * 1000.0})
    rows.sort(key=lambda r: r["own_ms"], reverse=True)
    resolve_ms = sum(r["own_ms"] for r in rows if r["func"] == "resolve")
    print("@@JSON@@" + json.dumps({
        "torch": torch.__version__,
        "file": torch.__file__,
        "shim": is_shim,
        "python": sys.version.split()[0],
        "gil": getattr(sys, "_is_gil_enabled", lambda: True)(),
        "data": data,
        "under_mode": under_mode,
        "profile": {"total_ms": total * 1000.0, "resolve_own_ms": resolve_ms, "top": rows[:12]},
    }))


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
    pooled = {label: {} for label in sides}
    moded = {label: {} for label in sides}
    meta, loads = {}, []
    for r in range(args.rounds):
        for label in sides:
            res = run_side(args.python, label)
            meta[label] = {k: res[k] for k in ("torch", "file", "python", "gil", "profile")}
            loads.append((r + 1, label, res["load_before"][0], res["load_after"][0]))
            for k, v in res["data"].items():
                pooled[label].setdefault(k, []).extend(v)
            for k, v in res.get("under_mode", {}).items():
                if isinstance(v, list):
                    moded[label].setdefault(k, []).extend(v)
                else:
                    meta[label]["under_mode_error"] = v

    print("dispatch: median ns per call (min .. max over chunks of %d), %d rounds x %d calls"
          % (CHUNK, args.rounds, REPS))
    print("%-34s" % "op" + "".join("%22s" % label for label in sides) + ("%10s" % "ratio" if len(sides) == 2 else ""))
    names = list(pooled[sides[0]].keys())
    for name in names:
        line = "%-34s" % name
        meds = []
        for label in sides:
            t = pooled[label].get(name)
            if t:
                med = statistics.median(t)
                meds.append(med)
                line += "%22s" % ("%7.0f (%5.0f..%5.0f)" % (med, min(t), max(t)))
            else:
                line += "%22s" % "-"
        if len(meds) == 2 and meds[0] > 0:
            line += "%10.2fx" % (meds[1] / meds[0])
        print(line)
    print()
    print("the same ops under a pass-through TorchDispatchMode (median ns per call; the timer's inflation, not a step cost)")
    print("%-34s" % "op" + "".join("%22s" % label for label in sides) + ("%10s" % "ratio" if len(sides) == 2 else ""))
    for name in names:
        line = "%-34s" % name
        meds = []
        for label in sides:
            t = moded[label].get(name)
            if t:
                med = statistics.median(t)
                meds.append(med)
                line += "%22s" % ("%7.0f (%5.0f..%5.0f)" % (med, min(t), max(t)))
            else:
                line += "%22s" % meta[label].get("under_mode_error", "-")[:20]
        if len(meds) == 2 and meds[0] > 0:
            line += "%10.2fx" % (meds[1] / meds[0])
        print(line)
    print()
    for label in sides:
        m = meta[label]
        prof = m["profile"]
        print("%-10s torch %s python %s gil=%s  profile total %.1f ms, resolve own %.1f ms (%.1f%%)"
              % (label, m["torch"], m["python"], m["gil"], prof["total_ms"], prof["resolve_own_ms"],
                 100.0 * prof["resolve_own_ms"] / prof["total_ms"] if prof["total_ms"] else 0.0))
        for row in prof["top"][:8]:
            print("    %8.1f ms own %8.1f ms cum %8d calls  %s (%s:%s)"
                  % (row["own_ms"], row["cum_ms"], row["calls"], row["func"], row["file"], row["line"]))
    print()
    print("load average (1 min) around each worker run:")
    for r, label, lb, la in loads:
        print("  round %d %-10s before %.2f after %.2f" % (r, label, lb, la))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"pooled": pooled, "under_mode": moded, "meta": meta, "loads": loads, "warmup": WARMUP,
                       "reps": REPS, "chunk": CHUNK}, fh, indent=1)
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
