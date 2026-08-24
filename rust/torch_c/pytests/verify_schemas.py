#!/usr/bin/env python3
"""Check `src/overloads.json` and `src/methods.json` against upstream torch,
schema string by schema string.

Why this exists
---------------
`surface.json` is generated from the vendored tree's own `.pyi` stubs, and
docs/IMPORT_TORCH.md §1 gives the reason: borrowing names from the upstream
`_C.so` would make the build depend on the binary we are replacing. The
overload table cannot be produced the same way. The tree carries the aten
*overload names* (`aten.arange.start_step` and 980 more appear literally in
`_decomp`/`_meta_registrations`/`_refs`) and it carries Python-level
signatures (`torch/_C/_VariableFunctions.pyi`), but nothing in it joins the
two -- a `.pyi` overload does not say which aten overload it lowers to. See
docs/OVERLOAD.md §2.

So the table is transcribed, and a transcription needs a check. This script
is that check: it re-derives every schema from an installed upstream torch
and diffs. It is *not* part of the build -- the table is compiled into the
artefact and `cargo build` needs no torch -- it is the same kind of tool as
`tools/golden/compare.py`: run it when upstream moves.

Usage
-----
    /Volumes/macMini/caches/spike-venv/bin/python rust/torch_c/pytests/verify_schemas.py

Exit code is 0 iff every entry matched. Read the exit code; do not grep.
"""

from __future__ import annotations

import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, os.pardir, "src")
TABLES = (
    ("overloads.json", os.path.join(SRC, "overloads.json")),
    ("methods.json", os.path.join(SRC, "methods.json")),
)


def _aten_name(schema: str) -> tuple[str, str]:
    """`aten::mul.Tensor(...) -> ...` -> `("mul", "Tensor")`.

    Derived from the schema string rather than from the table *key*, which is
    the only thing that works for both tables: in `overloads.json` the key is
    the op name, but in `methods.json` it is the Python method name, and the
    two are not always the same -- `item` is `_local_scalar_dense`, `__mul__`
    is `mul`, `__invert__` is `bitwise_not`.
    """
    head = schema.split("(", 1)[0].strip()
    _, _, rest = head.rpartition("::")
    name, _, overload = rest.partition(".")
    return name, overload or "default"


def check(label: str, path: str, torch) -> tuple[int, int]:
    with open(path, encoding="utf-8") as fh:
        table = json.load(fh)

    failures = 0
    checked = 0
    for name, schemas in sorted(table.items()):
        if name.startswith("_README"):
            continue  # the embedded README
        for schema in schemas:
            checked += 1
            op, overload = _aten_name(schema)
            try:
                packet = getattr(torch.ops.aten, op)
                upstream = {
                    str(getattr(packet, ov)._schema) for ov in packet.overloads()
                }
            except Exception as exc:
                print(f"FAIL {label} {name}: no upstream op packet for aten::{op}: {exc}")
                failures += 1
                continue
            if schema not in upstream:
                failures += 1
                print(f"FAIL {label} {name} ({op}.{overload}): "
                      f"table entry is not an upstream schema:")
                print(f"     table:    {schema}")
                # Show the closest upstream sibling by overload name, which is
                # what a drift looks like in practice (an argument gained a
                # default, or `int` became `SymInt`).
                head = schema.split("(", 1)[0]
                for candidate in sorted(upstream):
                    if candidate.split("(", 1)[0] == head:
                        print(f"     upstream: {candidate}")
    return checked, failures


def main() -> int:
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment problem
        print(f"FATAL: needs an environment with upstream torch installed: {exc}",
              file=sys.stderr)
        return 2

    print(f"torch {torch.__version__}")
    total = 0
    failed = 0
    for label, path in TABLES:
        checked, failures = check(label, path, torch)
        print(f"  {label}: {checked - failures}/{checked} matched")
        total += checked
        failed += failures

    # The other direction is deliberately *not* an error. The tables list the
    # overloads torch's Python bindings expose, which is a subset: `aten::pow`
    # has fifteen overloads and `torch.pow` reaches three of them.
    print()
    print(f"SUMMARY: {total - failed}/{total} table entries matched upstream, "
          f"{failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
