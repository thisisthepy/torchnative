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

import ast
import json
import os
import re
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


# The `_c10d_functional` family, which is not a JSON table but a tuple literal
# in `bootstrap.py`. Same problem, same answer: the text was transcribed from
# upstream's own registry (docs/DISTRIBUTED.md), so it needs the same check.
#
# It is read out of the source with `ast` rather than imported, because
# importing `bootstrap.py` means importing the shim, and this script has to run
# with *upstream* torch on the path -- putting both in one interpreter is the
# thing the golden harness goes to a second process to avoid.
BOOTSTRAP = os.path.join(SRC, "bootstrap.py")
NON_ATEN_NAMESPACES = ("_c10d_functional", "_c10d_functional_autograd", "_dtensor")


def _non_aten_schema_table() -> list:
    with open(BOOTSTRAP, encoding="utf-8") as fh:
        source = fh.read()
    tree = ast.parse(source)
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
        if "_NON_ATEN_SCHEMA_TEXT" in targets:
            return list(ast.literal_eval(node.value))
    raise RuntimeError(f"_NON_ATEN_SCHEMA_TEXT not found in {BOOTSTRAP}")


def check_non_aten(torch) -> tuple[int, int]:
    """Both directions, unlike the aten tables.

    A *missing* entry matters here in a way it does not for aten: the aten
    tables are deliberately a subset of what the dispatcher knows, but this
    table's whole job is to be the schemas the tree will ask for, and the tree
    asks for all of them. So an op upstream registers in one of these
    namespaces and this table lacks is reported.
    """
    # Importing this is what registers the namespaces; without it upstream's
    # own registry does not have them either.
    import torch.distributed._functional_collectives  # noqa: F401

    upstream = {}
    for schema in torch._C._jit_get_all_schemas():
        namespace = schema.name.split("::")[0]
        if namespace in NON_ATEN_NAMESPACES:
            upstream[str(schema).split("(")[0]] = str(schema)

    table = _non_aten_schema_table()
    failures = 0
    seen = set()
    for text in table:
        head = text.split("(")[0]
        seen.add(head)
        candidate = upstream.get(head)
        if candidate is None:
            failures += 1
            print(f"FAIL c10d schemas {head}: upstream has no such operator")
        elif _normalise(candidate) != _normalise(text):
            failures += 1
            print(f"FAIL c10d schemas {head}:")
            print(f"     table:    {text}")
            print(f"     upstream: {candidate}")
    for head in sorted(set(upstream) - seen):
        failures += 1
        print(f"FAIL c10d schemas {head}: upstream registers it and the table "
              "does not carry it")
    return len(table), failures


def _normalise(schema: str) -> str:
    return re.sub(r"\s+", " ", schema).strip()


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

    checked, failures = check_non_aten(torch)
    print(f"  bootstrap.py _NON_ATEN_SCHEMA_TEXT: {checked - failures}/{checked} matched")
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
