"""Record UPSTREAM torch's tie order for sort/argsort/topk (issue #33).

Run with the spike-venv interpreter and WITHOUT the vendored tree on the path:
    .caches/spike-venv/bin/python tests/_support/gen_sortties_oracle.py
Writes tests/ops/sortties_oracle.json.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path = [p for p in sys.path if "torchnative/python" not in p]
import torch  # noqa: E402
import sortties_cells as sc  # noqa: E402

assert not hasattr(torch._C, "_aten_implemented"), "this is the shim, not upstream"
out = {"torch": torch.__version__, "cells": {}}
for c in sc.cells():
    try:
        out["cells"][c["id"]] = sc.encode(sc.run_cell(torch, c))
    except RuntimeError as e:
        out["cells"][c["id"]] = {"error": str(e)}
dest = os.path.join(HERE, "..", "ops", "sortties_oracle.json")
with open(dest, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print(len(out["cells"]), "cells", torch.__version__)
