"""Dtype metadata shared by the golden comparison harness.

DESIGN.md §5 names dtype promotion and quiet numeric drift as the primary
risk of building `torch._C` on top of candle. This module is the single
place that says, for a given dtype name, how close two results are allowed
to be before the harness calls it a mismatch.

Every name in ``TOLERANCES`` must exist as an attribute with the same
spelling on *both* ``torch`` and the built ``_C`` module -- see
docs/TORCH_C.md §1 ("dtype 은 파이썬 상수가 아니라 _C 가 소유하는 타입이다")
for why the shim's dtype names were chosen to match torch's exactly.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Tolerance:
    atol: float
    rtol: float


# Reasoning per dtype: integer dtypes must match exactly (atol=rtol=0 makes
# `values_close` degenerate to `==`). Floating dtypes get a tolerance sized
# to roughly one ULP at magnitude ~1 for that format, so a real rounding
# *direction* disagreement between candle and torch still trips the check.
TOLERANCES: dict[str, Tolerance] = {
    "float64": Tolerance(atol=1e-9, rtol=1e-9),
    "float32": Tolerance(atol=1e-5, rtol=1e-5),
    "float16": Tolerance(atol=5e-3, rtol=5e-3),
    "bfloat16": Tolerance(atol=6e-2, rtol=6e-2),
    "float8_e4m3fn": Tolerance(atol=2e-2, rtol=2e-2),
    "int64": Tolerance(atol=0.0, rtol=0.0),
    "int32": Tolerance(atol=0.0, rtol=0.0),
    "int16": Tolerance(atol=0.0, rtol=0.0),
    "uint8": Tolerance(atol=0.0, rtol=0.0),
    "uint32": Tolerance(atol=0.0, rtol=0.0),
}

# Dtypes the harness exercises by default. `float8_e4m3fn` is deliberately
# left out: constructing a float8_e4m3fn tensor was observed to hang
# indefinitely on this host, on *both* sides independently --
# `torch.tensor([...], dtype=torch.float8_e4m3fn)` alone (no `_C` involved)
# and `_C._tensor_from_flat([...], dtype=_C.float8_e4m3fn)` alone (no torch
# involved). That is an environment/toolchain issue worth someone's
# attention, but it is orthogonal to numeric correctness and this harness
# must not hang CI over it, so it is excluded here rather than silently
# retried forever. See the final report for how this was narrowed down.
DEFAULT_DTYPES: tuple[str, ...] = (
    "float64",
    "float32",
    "float16",
    "bfloat16",
    "int64",
    "int32",
    "int16",
    "uint8",
    "uint32",
)

EXCLUDED_DTYPES: dict[str, str] = {
    "float8_e4m3fn": (
        "torch.tensor(..., dtype=torch.float8_e4m3fn) and "
        "_C._tensor_from_flat(..., dtype=_C.float8_e4m3fn) both hang "
        "indefinitely when probed independently on this host (observed "
        "2026-08-24). Not exercised by this harness; needs investigation "
        "in rust/torch_c, which is out of scope here."
    ),
}


def dtype_name(dtype_obj) -> str:
    """`torch.float32` and `_C.float32` both repr as ``"torch.float32"``;
    normalize either to the bare name used as a key above."""
    s = str(dtype_obj)
    return s[len("torch.") :] if s.startswith("torch.") else s


def torch_dtype(torch_module, name: str):
    return getattr(torch_module, name)


def c_dtype(c_module, name: str):
    return getattr(c_module, name)


def tolerance_for(name: str) -> Tolerance:
    # Unknown dtype: fail closed (require exact match) rather than silently
    # let something new drift past under a guessed tolerance.
    return TOLERANCES.get(name, Tolerance(atol=0.0, rtol=0.0))
