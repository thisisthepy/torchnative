"""Record UPSTREAM torch's tie order for sort/argsort/topk (issue #33).

Run with the spike-venv interpreter and WITHOUT the vendored tree on the path:
    .caches/spike-venv/bin/python tests/_support/gen_sortties_oracle.py
Writes tests/ops/sortties_oracle.json (macOS, libc++), or on Linux
tests/ops/sortties_oracle_linux.json (the manylinux wheel, libstdc++; issue
#81), or on Windows tests/ops/sortties_oracle_windows.json (the win_amd64 wheel,
MSVC's STL; issue #81; run by .github/workflows/sortties-oracle-windows.yml).
The Linux file was recorded in Docker python:3.13 with torch==2.13.0
from the PyTorch CPU index, on linux/arm64 and linux/amd64: the two runs were
byte-identical, so one file serves both architectures.
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path = [p for p in sys.path if "torchnative/python" not in p.replace("\\", "/")]
import torch  # noqa: E402
import sortties_cells as sc  # noqa: E402

assert not hasattr(torch._C, "_aten_implemented"), "this is the shim, not upstream"
out = {"torch": torch.__version__, "cells": {}}
for c in sc.cells():
    try:
        out["cells"][c["id"]] = sc.encode(sc.run_cell(torch, c))
    except RuntimeError as e:
        out["cells"][c["id"]] = {"error": str(e)}
if sys.platform.startswith("linux"):
    name = "sortties_oracle_linux.json"
elif sys.platform == "win32":
    name = "sortties_oracle_windows.json"
else:
    name = "sortties_oracle.json"
dest = os.path.join(HERE, "..", "ops", name)
with open(dest, "w") as f:
    json.dump(out, f, separators=(",", ":"))
print(len(out["cells"]), "cells", torch.__version__)
