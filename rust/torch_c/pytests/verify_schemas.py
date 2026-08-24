#!/usr/bin/env python3
"""Check `src/overloads.json` against upstream torch, schema string by schema
string.

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
TABLE = os.path.join(HERE, os.pardir, "src", "overloads.json")


def main() -> int:
    with open(TABLE, encoding="utf-8") as fh:
        table = json.load(fh)

    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment problem
        print(f"FATAL: needs an environment with upstream torch installed: {exc}",
              file=sys.stderr)
        return 2

    print(f"torch {torch.__version__}")
    failures = 0
    checked = 0

    for name, schemas in sorted(table.items()):
        if name.startswith("_"):
            continue  # the embedded README
        try:
            packet = getattr(torch.ops.aten, name)
            upstream = {str(getattr(packet, ov)._schema) for ov in packet.overloads()}
        except Exception as exc:
            print(f"FAIL {name}: no upstream op packet: {exc}")
            failures += 1
            continue

        for schema in schemas:
            checked += 1
            if schema not in upstream:
                failures += 1
                print(f"FAIL {name}: table entry is not an upstream schema:")
                print(f"     table:    {schema}")
                # Show the closest upstream sibling by overload name, which is
                # what a drift looks like in practice (an argument gained a
                # default, or `int` became `SymInt`).
                head = schema.split("(", 1)[0]
                for candidate in sorted(upstream):
                    if candidate.split("(", 1)[0] == head:
                        print(f"     upstream: {candidate}")

    # The other direction is deliberately *not* an error. The table lists the
    # overloads torch's Python binding exposes, which is a subset: `aten::pow`
    # has fifteen overloads and `torch.pow` reaches three of them.
    print()
    print(f"SUMMARY: {checked - failures}/{checked} table entries matched upstream, "
          f"{failures} failed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
