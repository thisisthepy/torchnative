"""Upstream oracle for tests/numerics/test_promote0d.py::test_reduced_float_mul_div_with_a_0d_operand.

Run with upstream torch (spike-venv, no vendored tree on the path) and save stdout
as tests/numerics/promote0d_reduced.json. The suite runs this same script against
the shim and compares cell by cell.
"""
import os, json, sys
os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
import torch
out = []
dims = [torch.bfloat16, torch.float16]
zs = [torch.float32, torch.float64, torch.bfloat16, torch.float16]
for d in dims:
    for z in zs:
        for vals, s in [([3.0, 1.7, 2.9], 0.3), ([3.1], 3.1), ([-2.5, 7.3], 1.9)]:
            a = torch.tensor(vals, dtype=d); b = torch.tensor(s, dtype=z)
            for name, f in [("mul_r", lambda: a * b), ("mul_l", lambda: b * a), ("div_r", lambda: a / b),
                            ("div_l", lambda: b / a), ("both0d_mul", lambda: torch.tensor(vals[0], dtype=d) * b),
                            ("both0d_div", lambda: torch.tensor(vals[0], dtype=d) / b)]:
                try:
                    r = f(); out.append([str(d), str(z), vals, s, name, str(r.dtype), r.float().reshape(-1).tolist()])
                except Exception as e:
                    out.append([str(d), str(z), vals, s, name, "ERR", type(e).__name__])
print(json.dumps(out))
