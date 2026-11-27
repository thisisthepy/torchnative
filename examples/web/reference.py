"""Write `reference.json`: upstream torch's answers to `basics.py`'s cases.

Runs on a host with **upstream** torch (not torchnative), so the browser's results are
checked against upstream and not against torchnative itself:

    uv run --no-project --with torch==2.13.0 python reference.py

Each case is run in float32 (the value the page must reproduce) and, for `float` cases,
again in float64; the difference between the two is upstream's own rounding error, and
`basics.tolerance` turns it into the per-element tolerance (AGENTS.md §16).

It refuses to write if the torch it imported is torchnative's.
"""

import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from basics import CASES, compute, tolerance  # noqa: E402

if hasattr(torch._C, "_aten_implemented"):
    raise SystemExit("this torch is torchnative's; the reference must come from upstream torch")

f32 = compute(torch, torch.float32)
f64 = compute(torch, torch.float64)
cases = {}
for name, kind, _fn in CASES:
    if "error" in f32[name] or "error" in f64[name]:
        raise SystemExit(f"upstream raised on case {name!r}: {f32[name].get('error') or f64[name]['error']}")
    entry = {"kind": kind, "value": f32[name]["value"]}
    if kind == "float":
        entry["tol"] = tolerance(f32[name]["value"], f64[name]["value"])
    cases[name] = entry
record = {"torch": torch.__version__, "cases": cases}
with open(os.path.join(HERE, "reference.json"), "w", encoding="utf-8") as fh:
    json.dump(record, fh, indent=1)
    fh.write("\n")
print(json.dumps({"torch": torch.__version__, "cases": len(cases)}))
