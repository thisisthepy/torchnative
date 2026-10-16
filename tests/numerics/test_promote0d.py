"""Issue #31: a 0-d tensor operand must not raise the result dtype.

Upstream's rule (`torch.result_type`, c10 `ResultTypeState`): dimensioned
tensors > 0-d tensors > Python scalars; a lower-priority operand only raises the
result when its category (bool < integral < floating < complex) is higher.

`promote0d_table.json` is the specification. It was measured against upstream
torch by `tests/_support/gen_promote0d.py`: operand kinds {dimensioned, 0-d,
Python scalar} x 12 dtypes (scalars: bool/int/float/complex) x both orders x
ops {add, sub, mul, div, eq, lt (operator syntax, so a Python scalar can come first), where, result_type}. Every cell here is a check
of result dtype AND value (a comparison's bool result hides the dtype it
computed in; its value does not). A cell upstream refuses must be refused. Cells the shim refuses by name for
reasons that are not this issue (complex tensors, bool arithmetic: see `_gap`)
are counted apart as `gap:*`, never as agreement, and never silently.

What this cannot see: devices (mps/cuda/vulkan) that compute dtype separately;
every cell runs on cpu. Only the (op, operand) pairs above are covered.
"""
import json
import os
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "AGENTS.md").is_file()))
sys.path.insert(0, os.path.join(_ROOT, "torchnative", "python"))

import torch  # noqa: E402

assert hasattr(torch._C, "_aten_implemented"), "this is upstream torch, not the shim"

import _skip  # noqa: E402

_TABLE = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "promote0d_table.json")))
CELLS = _TABLE["cells"]
EXPECTED_CELLS = 6144
VAL = {"bool": True, "int": 3, "float": 3.1, "complex": 3.1 + 1j}


def _cat(name):
    if name == "bool":
        return "bool"
    if name in ("int", "uint8", "int8", "int16", "int32", "int64"):
        return "int"
    if name.startswith("float") or name == "bfloat16":
        return "float"
    return "complex"


def _operand(kind, name):
    v = VAL[_cat(name)]
    if kind == "py":
        return v
    t = torch.tensor(v, dtype=getattr(torch, name))
    return t.reshape(1) if kind == "dim" else t


def _cv(v):
    if isinstance(v, list):
        return [_cv(i) for i in v]
    if isinstance(v, complex):
        return [v.real, v.imag]
    return v


def _enc(x):
    if isinstance(x, torch.Tensor):
        return [str(x.dtype).replace("torch.", ""), _cv(x.tolist())]
    return ["py", _cv(x)]


def _close(a, b):
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(_close(x, y) for x, y in zip(a, b))
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    return abs(a - b) <= 1e-3 * max(1.0, abs(b))


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


def _gap(cell):
    """The declared, named refusals this suite tolerates (and counts apart).

    complex: the shim holds a complex tensor as a pair of real tensors and has
        no complex arithmetic, `torch.tensor(complex)` or comparison kernels
        (docs/kernels/COMPLEX.md). Any cell with a complex operand other than
        `result_type` on a Python complex scalar is such a refusal.
    bool: `arith_tag` refuses bool-tensor operands of add/sub/mul/div by name
        (docs/numerics/BOOL.md); upstream computes. Not part of issue #31.
    """
    ops = (cell["a"], cell["b"])
    has_complex = any(_cat(n) == "complex" for _k, n in ops)
    complex_tensor = any(k != "py" and _cat(n) == "complex" for k, n in ops)
    if has_complex and (cell["op"] != "result_type" or complex_tensor):
        return "complex"
    if cell["op"] in ("add", "sub", "mul", "div") and any(k != "py" and n == "bool" for k, n in ops):
        return "bool"
    return None


def _check(cell):
    """(status, why): status is "agree", "gap:<name>" or "fail"."""
    ka, na = cell["a"]
    kb, nb = cell["b"]
    err = ""
    try:
        a, b = _operand(ka, na), _operand(kb, nb)
        r = OPS[cell["op"]](a, b)
        got = ["dtype", str(r).replace("torch.", "")] if cell["op"] == "result_type" else _enc(r)
    except Exception as e:  # noqa: BLE001
        got = "ERR"
        err = f"{type(e).__name__}: {str(e)[:80]}"
    want = cell["res"]
    if want == "ERR":
        return ("agree", None) if got == "ERR" else ("fail", f"upstream refuses, shim answers {got}")
    if got == "ERR":
        gap = _gap(cell)
        if gap:
            return f"gap:{gap}", err
        return "fail", f"upstream answers {want}, shim raises {err}"
    if got[0] != want[0]:
        return "fail", f"dtype {got[0]}, upstream {want[0]}"
    if (got[1] != want[1]) if isinstance(want[1], str) else not _close(got[1], want[1]):
        return "fail", f"value {got[1]}, upstream {want[1]} (dtype {got[0]})"
    return "agree", None


def _sweep(op):
    cells = [c for c in CELLS if c["op"] == op]
    bad, counts = [], {}
    for c in cells:
        status, why = _check(c)
        counts[status] = counts.get(status, 0) + 1
        if status == "fail":
            bad.append(f"{op}({c['a'][0]}:{c['a'][1]}, {c['b'][0]}:{c['b'][1]}): {why}")
    print(f"  {op}: {counts.get('agree', 0)} of {len(cells)} cells agree; "
          + ", ".join(f"{k}={v}" for k, v in sorted(counts.items()) if k != "agree"))
    assert not bad, f"{op}: {len(bad)} of {len(cells)} cells disagree with upstream:\n" + "\n".join(bad[:40])


def test_table_is_whole():
    assert len(CELLS) == EXPECTED_CELLS, (len(CELLS), EXPECTED_CELLS)
    assert {c["op"] for c in CELLS} == set(OPS)


def test_issue_31_examples():
    a = torch.ones(3, dtype=torch.int32)
    assert (a * torch.tensor(2)).dtype == torch.int32
    f = torch.ones(3, dtype=torch.float32)
    assert (f * torch.tensor(2.0, dtype=torch.float64)).dtype == torch.float32


def _mk(op):
    return lambda: _sweep(op)


_ITEMS = [("test_table_is_whole", test_table_is_whole), ("test_issue_31_examples", test_issue_31_examples)]
_ITEMS += [(f"test_promote0d_{op}", _mk(op)) for op in OPS]


def test_reduced_float_mul_div_with_a_0d_operand():
    # Values the main table does not probe. Upstream reads a 0-d *right*
    # operand of `mul`/`div` at its original value on bfloat16/float16 and
    # narrows a 0-d *left* operand; the first version of the fix did both
    # sides and was 5 of 144 cells off (mul with the 0-d on the left).
    import subprocess
    here = os.path.dirname(os.path.abspath(__file__))
    want = json.load(open(os.path.join(here, "promote0d_reduced.json")))
    gen = os.path.join(here, "..", "_support", "gen_promote0d_reduced.py")
    proc = subprocess.run([sys.executable, gen], capture_output=True, text=True,
                          env=dict(os.environ, TORCH_USE_RTLD_GLOBAL="1",
                                   PYTHONPATH=os.path.join(_ROOT, "torchnative", "python")),
                          timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    bad = [(w, g) for w, g in zip(want, got) if w != g]
    assert len(got) == len(want) == 144, (len(got), len(want))
    assert not bad, f"{len(bad)} of {len(want)} cells differ: " + "; ".join(
        f"{w[:5]} upstream {w[5:]} shim {g[5:]}" for w, g in bad[:5])


_ITEMS.append(("test_reduced_float_mul_div_with_a_0d_operand", test_reduced_float_mul_div_with_a_0d_operand))


if __name__ == "__main__":
    raise SystemExit(1 if _skip.run_tests(_ITEMS, "test_promote0d") else 0)
