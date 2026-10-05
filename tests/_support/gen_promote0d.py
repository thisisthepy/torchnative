"""Generate tests/numerics/promote0d_table.json from UPSTREAM torch (issue #31).

Run with the spike-venv interpreter and the vendored tree NOT on the path:
    PYTHONPATH= .caches/spike-venv/bin/python tests/_support/gen_promote0d.py OUT.json
Each cell: op, operand kinds+dtypes, both orders, result dtype and value
(or "ERR"). Comparison cells record the value too, because the dtype a
comparison computes in is invisible in its bool result but not in its value.
"""
import json
import sys

import torch

assert not hasattr(torch._C, "_aten_implemented"), "this is the shim, not upstream"

DT = ["bool", "uint8", "int8", "int16", "int32", "int64", "float16", "bfloat16",
      "float32", "float64", "complex64", "complex128"]
PYS = ["bool", "int", "float", "complex"]
VAL = {"bool": True, "int": 3, "float": 3.1, "complex": 3.1 + 1j}


def cat(name):
    if name == "bool":
        return "bool"
    if name in ("int", "uint8", "int8", "int16", "int32", "int64"):
        return "int"
    if name.startswith("float") or name == "bfloat16":
        return "float"
    return "complex"


def operand(kind, name):
    v = VAL[cat(name)]
    if kind == "py":
        return v
    t = torch.tensor(v, dtype=getattr(torch, name))
    return t.reshape(1) if kind == "dim" else t


def enc(x):
    if isinstance(x, torch.Tensor):
        v = x.tolist()
        return [str(x.dtype).replace("torch.", ""), _cv(v)]
    return ["py", _cv(x)]


def _cv(v):
    if isinstance(v, list):
        return [_cv(i) for i in v]
    if isinstance(v, complex):
        return [v.real, v.imag]
    return v


OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "eq": lambda a, b: a == b,
    "lt": lambda a, b: a < b,
    "where": lambda a, b: torch.where(torch.tensor([True]), a, b),
    "result_type": lambda a, b: torch.result_type(a, b),
}

operands = [("dim", n) for n in DT] + [("zero", n) for n in DT] + [("py", n) for n in PYS]
cells = []
for op, fn in OPS.items():
    for ka, na in operands:
        for kb, nb in operands:
            if ka == "py" and kb == "py":
                continue
            a, b = operand(ka, na), operand(kb, nb)
            try:
                r = fn(a, b)
                res = ["dtype", str(r).replace("torch.", "")] if op == "result_type" else enc(r)
            except Exception:
                res = "ERR"
            cells.append({"op": op, "a": [ka, na], "b": [kb, nb], "res": res})
json.dump({"torch": torch.__version__, "cells": cells}, open(sys.argv[1], "w"))
print(len(cells), "cells")
