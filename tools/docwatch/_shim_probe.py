#!/usr/bin/env python3
"""Subprocess helper for `tools/docwatch/check_docs.py`.

Kept out-of-process from `check_docs.py`'s own interpreter, and out of the
main module entirely, for one reason: `check_docs.py --no-live` (the
static-only mode that checks `symbol-in-file`/`json-key` markers) must be
able to run without importing torch at all -- there are hosts where no
upstream torch is installed and the doc-only checks should still work.
Importing this file only happens when a marker actually needs a live shim
fact.

Loads the built `torch._C` shim the same way `tools/golden/compare.py`
does (`tools/golden/loader.py`, `TORCH_C_ARTEFACT` env var), and separately
imports real upstream torch from whatever `sys.path` this interpreter
already has (the project's spike-venv has it -- see that venv's own
`compare.py` docstring). Answers two kinds of live fact:

  - which aten ops `_C._aten_implemented()` reports (backs the
    `op-implemented` / `op-not-implemented` markers)
  - `hasattr(<upstream> torch, attr)` for a requested set of names (backs
    the `hasattr` marker -- this is the exact measurement ARCH20.md /
    SPELLINGS.md used to tell "no bare torch.<name> upstream" apart from
    "the shim refuses a name upstream does have")

Protocol: reads a JSON object from stdin, `{"attrs": [...]}`. Prints one
JSON object to stdout, `{"implemented": [...], "hasattr": {attr: bool,
...}}`, or `{"implemented_error": "...", "hasattr": {...}}` if the shim
could not be loaded (hasattr can still be answered without a shim).
"""
from __future__ import annotations

import json
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "golden"))

import loader  # noqa: E402  (path is inserted above, first-party)


def main() -> int:
    request = json.loads(sys.stdin.read() or "{}")
    attrs = request.get("attrs", [])

    result: dict = {}
    try:
        shim = loader.load_shim()
        result["implemented"] = list(shim._aten_implemented())
    except loader.ShimLoadError as e:
        result["implemented_error"] = str(e)

    import torch  # upstream, real -- see module docstring above

    result["hasattr"] = {attr: bool(hasattr(torch, attr)) for attr in attrs}
    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
