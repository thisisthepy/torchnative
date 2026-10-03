"""Build matching torch / `_C` tensors from the same flat data.

`_C._tensor_from_flat(values, shape, dtype=None, device=None)` is the shim's
documented, deliberately unpromoted scaffolding for getting real data into a
`TensorBase` (see its docstring: there is no aten op yet that takes a Python
list of numbers). This module is the one place a golden case has to state
its numbers, so both sides are guaranteed to start from identical input.
"""

from __future__ import annotations

import dtypes as dt


def pair_from_flat(torch_module, c_module, flat, shape, dtype_name: str):
    """Return `(torch_tensor, c_tensor)` built from the same flat values."""
    t_dtype = dt.torch_dtype(torch_module, dtype_name)
    c_dt = dt.c_dtype(c_module, dtype_name)
    torch_t = torch_module.tensor(list(flat), dtype=t_dtype).reshape(list(shape))
    c_t = c_module._tensor_from_flat(list(flat), list(shape), dtype=c_dt)
    return torch_t, c_t


def scalar_pair(torch_module, c_module, value, dtype_name: str):
    """A 0-d tensor pair, e.g. for testing broadcasting against a scalar."""
    return pair_from_flat(torch_module, c_module, [value], [], dtype_name)
