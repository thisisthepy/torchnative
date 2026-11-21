"""Decode-shaped attention: upstream torch vs the shim vs the shim with the
decode kernel off, on the shapes one SmolLM2-135M decode step passes to
`aten._scaled_dot_product_flash_attention_for_cpu` (docs/perf/SDPA_DECODE.md).

Not part of the gate. The gate's claim is counts and bounds
(`tests/ops/test_sdpa_decode.py`); this file's claim is time, and a time is
only evidence when the machine is otherwise idle, so it prints the load
average around every run and refuses to interpret its own numbers.

Run (CI; locally only when told to, alone, after `uptime`)::

    PYTHONPATH= python tests/bench/bench_sdpa_decode.py --python "$(which python)" --rounds 3

    # sweep the split floor (multiply-adds per call) without a rebuild
    ... --thread-madds 0,65536,1048576

Each side runs in its own interpreter: `upstream` with no `PYTHONPATH`, `shim`
with the source tree, `shim-off` the same with `TORCHNATIVE_SDPA_DECODE=0`.
The shim sides report the counter deltas per shape, so a faster number can be
read against whether the timed calls took the kernel at all. No U+2014 anywhere.
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
REPS = 60

# (name, batch, query heads, kv heads, q_len, kv_len, head_dim, mask kind)
# SmolLM2-135M: 9 query heads over 3 key/value heads, head dim 64, 30 layers.
SHAPES = [
    ("decode b4 kv74 nomask", 4, 9, 3, 1, 74, 64, None),
    ("decode b4 kv74 (b,1,1,s)", 4, 9, 3, 1, 74, 64, "b11s"),
    ("decode b4 kv512 (b,1,1,s)", 4, 9, 3, 1, 512, 64, "b11s"),
    ("decode b1 kv20 nomask", 1, 9, 3, 1, 20, 64, None),
    ("packed b1 q4 kv300 (1,1,l,s)", 1, 9, 3, 4, 300, 64, "11ls"),
    ("decode b4 kv74 repeated heads", 4, 9, 9, 1, 74, 64, "b11s"),
]


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

    is_shim = hasattr(torch._C, "_aten_implemented")
    counters = getattr(torch._C, "_sdpa_decode_counters", None)
    op = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu
    torch.manual_seed(0)
    data, taken = {}, {}
    for name, b, h, hk, l, s, d, mask_kind in SHAPES:
        # The query as the model passes it: a transposed (batch, len, heads, dim) view.
        q = torch.randn(b, l, h, d).transpose(1, 2)
        k = torch.randn(b, hk, s, d)
        v = torch.randn(b, hk, s, d)
        mask = None
        if mask_kind == "b11s":
            mask = torch.zeros(b, 1, 1, s)
            mask[:, :, :, : s // 8] = float("-inf")
        elif mask_kind == "11ls":
            mask = torch.zeros(1, 1, l, s)
            mask[:, :, :, : s // 8] = float("-inf")
        key = name
        before = counters()["calls"] if counters else None
        try:
            data[key] = {"t": _time(lambda: op(q, k, v, 0.0, False, attn_mask=mask), WARMUP, REPS)}
        except Exception as exc:  # noqa: BLE001 -- recorded, the other shapes still run
            data[key] = {"error": repr(exc)[:300]}
        if counters:
            taken[key] = counters()["calls"] - before
    print("@@JSON@@" + json.dumps({
        "torch": torch.__version__,
        "file": torch.__file__,
        "shim": is_shim,
        "threads": torch.get_num_threads() if hasattr(torch, "get_num_threads") else None,
        "counters": counters() if counters else None,
        "taken": taken,
        "data": data,
    }))


def run_side(python, side, thread_madds=None):
    env = dict(os.environ)
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    if side == "upstream":
        env.pop("PYTHONPATH", None)
    else:
        env["PYTHONPATH"] = SHIM_PATH
    if side == "shim-off":
        env["TORCHNATIVE_SDPA_DECODE"] = "0"
    if thread_madds is not None:
        env["TORCHNATIVE_SDPA_DECODE_THREAD_MADDS"] = str(thread_madds)
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
    ap.add_argument("--thread-madds", default="",
                    help="comma-separated TORCHNATIVE_SDPA_DECODE_THREAD_MADDS values; "
                         "adds one 'shim@<madds>' side per value")
    ap.add_argument("--out", default="", help="write the pooled results as JSON here")
    args = ap.parse_args()

    if args.worker:
        worker()
        return

    sides = [(s, None) for s in args.sides.split(",") if s]
    for raw in [v for v in args.thread_madds.split(",") if v]:
        sides.append((f"shim@{int(raw)}", int(raw)))

    pooled = {label: {} for label, _ in sides}
    errors = {label: {} for label, _ in sides}
    meta, loads = {}, []
    for r in range(args.rounds):
        for label, thread_madds in sides:
            side = "shim" if label.startswith("shim@") else label
            res = run_side(args.python, side, thread_madds)
            meta[label] = {k: res[k] for k in ("torch", "file", "threads", "counters", "taken")}
            loads.append((r + 1, label, res["load_before"][0], res["load_after"][0]))
            for k, v in res["data"].items():
                if "t" in v:
                    pooled[label].setdefault(k, []).extend(v["t"])
                else:
                    errors[label][k] = v["error"]

    print("sdpa decode: median ms per call (min .. max), %d rounds x %d reps, warmup %d"
          % (args.rounds, REPS, WARMUP))
    header = "%-34s" % "shape" + "".join("%16s" % label for label, _ in sides) + "   taken(shim)"
    print(header)
    for name, *_ in SHAPES:
        line = "%-34s" % name
        for label, _ in sides:
            t = pooled[label].get(name)
            if t:
                line += "%8.3f %s" % (statistics.median(t), "(%.2f..%.2f)" % (min(t), max(t)))[:16].rjust(16)
            else:
                line += "%16s" % ("error" if name in errors[label] else "-")
        shim_meta = meta.get("shim")
        line += "   %s" % (shim_meta["taken"].get(name, "?") if shim_meta and shim_meta["taken"] else "n/a")
        print(line)
    print()
    for label, _ in sides:
        m = meta[label]
        c = m["counters"]
        where = ("decode routed=%s threads=%s thread_madds=%s simd=%s" %
                 (c["routed"], c["threads"], c["thread_madds"], c["simd"])) if c else "upstream"
        print("%-12s torch %s threads=%s  %s" % (label, m["torch"], m["threads"], where))
        for name, err in errors[label].items():
            print("%-12s   %s: %s" % ("", name, err))
    print()
    print("load average (1 min) around each worker run:")
    for r, label, lb, la in loads:
        print("  round %d %-12s before %.2f after %.2f" % (r, label, lb, la))
    if args.out:
        with open(args.out, "w") as fh:
            json.dump({"shapes": SHAPES, "pooled": pooled, "errors": errors, "meta": meta,
                       "loads": loads, "warmup": WARMUP, "reps": REPS}, fh, indent=1)
        print("wrote %s" % args.out)


if __name__ == "__main__":
    main()
