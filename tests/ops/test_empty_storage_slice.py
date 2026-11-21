"""Issue #385: audit of CPU kernels that slice a zero-element tensor's storage
(docs/kernels/STRIDED.md section 8.9, SPEC S1.13).

A zero-element tensor on the CPU can hold a layout whose `start_offset` lies
past the end of its storage (`as_strided_((0, 3), (3, 1), 100)` over 12
elements) or whose strides step past a 0-byte storage (`x[:, 0:0].clone()`,
which keeps stride `(4, 1)` over an empty buffer). Any CPU kernel that slices
`&storage[start_offset..]` or steps outer blocks without checking for zero
elements first turns an empty operation into a Rust index-out-of-bounds panic
(`pyo3_runtime.PanicException`).

This suite feeds both layouts to every implemented operator in
`torch._C._aten_all_implemented()` that takes a tensor argument, in a separate
subprocess per family on both upstream torch and the shim, and asserts:

1. Every tensor-taking op in `torch._C._aten_all_implemented()` is covered by
   at least one family, and the factory-only exemption list contains only ops
   that take no tensor input.
2. No family subprocess dies, and zero rows on the shim raise `PanicException`.
3. Every row where both upstream and the shim return tensor(s) agrees on output
   shapes and dtypes; where upstream raises an exception (such as `amax` or
   `_local_scalar_dense` on 0 elements) or the shim raises a named
   `NotImplementedError`, the shim never panics.
"""

import json
import os
import subprocess
import sys

import _skip

_REPO_ROOT = str(
    next(
        p
        for p in __import__("pathlib").Path(__file__).resolve().parents
        if (p / "AGENTS.md").is_file()
    )
)
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "python")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")


def _available():
    return os.path.isfile(_VENDOR_SHIM)


#: Operators in `_aten_all_implemented()` that construct a tensor from sizes or
#: scalars and take no input tensor.
FACTORY_OPS_WITHOUT_TENSOR_INPUT = frozenset({
    "aten.arange.default",
    "aten.arange.start",
    "aten.arange.start_step",
    "aten.empty.memory_format",
    "aten.empty_strided.default",
    "aten.eye.default",
    "aten.eye.m",
    "aten.full.default",
    "aten.hann_window.default",
    "aten.hann_window.periodic",
    "aten.kaiser_window.beta",
    "aten.kaiser_window.default",
    "aten.kaiser_window.periodic",
    "aten.linspace.default",
    "aten.ones.default",
    "aten.randint.default",
    "aten.randint.low",
    "aten.randperm.default",
    "aten.scalar_tensor.default",
    "aten.zeros.default",
})

FAMILIES = (
    "unary_and_prims",
    "views_and_copies",
    "binary_and_comparisons",
    "reductions_and_norms",
    "linalg_and_attention",
    "conv_pool_upsample_pad_fft_rnn",
    "indexing_scatter_cat_sort",
    "inplace",
)

_PROBE = r'''
import json
import sys
import torch

family = sys.argv[1]
is_shim = hasattr(torch._C, "_aten_all_implemented")


def resolve_op(key):
    ns, rest = key.split(".", 1)
    name, overload = rest.split(".", 1)
    packet = getattr(getattr(torch.ops, ns), name)
    return getattr(packet, overload)


def call_op(key, *args, **kwargs):
    if is_shim:
        return torch._C._aten_dispatch(key, *args, **kwargs)
    if key == "aten.reshape_as.default":
        return args[0].reshape_as(args[1])
    if key == "aten.contiguous.default":
        return args[0].contiguous(*args[1:], **kwargs)
    return resolve_op(key)(*args, **kwargs)


def make_past_1d(dt=torch.float32):
    base = torch.zeros(12, dtype=dt)
    return base.as_strided_((0,), (1,), 100)


def make_clone_1d(dt=torch.float32):
    base = torch.zeros(12, dtype=dt).view(3, 4)
    return base[:, 0:0].clone().reshape(0)


def make_past_2d(dt=torch.float32):
    base = torch.zeros(12, dtype=dt)
    return base.as_strided_((0, 3), (3, 1), 100)


def make_clone_2d(dt=torch.float32):
    base = torch.zeros(12, dtype=dt).view(3, 4)
    return base[:, 0:0].clone()


def make_past_3d(dt=torch.float32):
    base = torch.zeros(12, dtype=dt)
    return base.as_strided_((0, 2, 4), (8, 4, 1), 100)


def make_clone_3d(dt=torch.float32):
    base = torch.zeros(24, dtype=dt).view(2, 3, 4)
    return base[:, :, 0:0].clone()


def make_past_4d(dt=torch.float32):
    base = torch.zeros(12, dtype=dt)
    return base.as_strided_((0, 2, 4, 4), (32, 16, 4, 1), 100)


def make_clone_4d(dt=torch.float32):
    base = torch.zeros(48, dtype=dt).view(2, 2, 3, 4)
    return base[:, :, :, 0:0].clone()


def make_past_5d(dt=torch.float32):
    base = torch.zeros(12, dtype=dt)
    return base.as_strided_((0, 2, 3, 3, 3), (54, 27, 9, 3, 1), 100)


def make_clone_5d(dt=torch.float32):
    base = torch.zeros(36, dtype=dt).view(1, 2, 2, 3, 3)
    return base[:, :, :, :, 0:0].clone()


def summarize(val):
    if isinstance(val, torch.Tensor):
        return {"kind": "tensor", "shape": list(val.shape), "dtype": str(val.dtype)}
    if isinstance(val, (tuple, list)):
        return {"kind": "seq", "items": [summarize(v) for v in val]}
    if isinstance(val, (bool, int, float)):
        return {"kind": "scalar", "type": type(val).__name__}
    return {"kind": "other", "type": type(val).__name__}


def run_case(fn):
    try:
        res = fn()
        return {"ok": True, "result": summarize(res)}
    except BaseException as exc:
        if isinstance(exc, (KeyboardInterrupt, SystemExit)):
            raise
        return {
            "ok": False,
            "exc": type(exc).__name__,
            "is_panic": "PanicException" in type(exc).__name__,
            "msg": str(exc),
        }


rows = {}
covered = set()


# Rows where upstream torch 2.13.0 itself crashes with SIGSEGV (returncode -11)
# on macOS arm64 (CI run 44517635477, isolated per-op on 2026-11-19):
# `aten._weight_norm_interface.default(clone_2d, clone_2d, 0)` where `clone_2d`
# has shape (3, 0) and non-zero stride (4, 1). The shim still runs this row and
# is checked for no Rust panic; only the upstream probe skips it so the
# upstream subprocess does not die before reporting the rest of the family.
import os

UPSTREAM_SIGSEGV_ROWS = frozenset({
    "aten._weight_norm_interface.default::clone",
}) | frozenset(
    r for r in os.environ.get("UPSTREAM_SKIP_ROWS", "").split(",") if r
)
_TRACE_ROW = os.environ.get("TRACE_ROW") == "1"


def record(op, tag, fn):
    covered.add(op)
    row_key = f"{op}::{tag}"
    if not is_shim and row_key in UPSTREAM_SIGSEGV_ROWS:
        rows[row_key] = {"ok": False, "upstream_sigsegv": True}
        return
    if _TRACE_ROW:
        sys.stderr.write(f"ROW {row_key}\n")
        sys.stderr.flush()
    rows[row_key] = run_case(fn)


def both_2d(dt=torch.float32):
    return (("past", lambda: make_past_2d(dt)), ("clone", lambda: make_clone_2d(dt)))


def both_1d(dt=torch.float32):
    return (("past", lambda: make_past_1d(dt)), ("clone", lambda: make_clone_1d(dt)))


def both_3d(dt=torch.float32):
    return (("past", lambda: make_past_3d(dt)), ("clone", lambda: make_clone_3d(dt)))


def both_4d(dt=torch.float32):
    return (("past", lambda: make_past_4d(dt)), ("clone", lambda: make_clone_4d(dt)))


if family == "unary_and_prims":
    FLOAT_UNARY = (
        "aten.abs.default",
        "aten.acos.default",
        "aten.ceil.default",
        "aten.cos.default",
        "aten.detach.default",
        "aten.erf.default",
        "aten.erfinv.default",
        "aten.exp.default",
        "aten.expm1.default",
        "aten.floor.default",
        "aten.gelu.default",
        "aten.hardtanh.default",
        "aten.i0.default",
        "aten.isinf.default",
        "aten.isnan.default",
        "aten.isposinf.default",
        "aten.isneginf.default",
        "aten.is_floating_point.default",
        "aten.leaky_relu.default",
        "aten.lift_fresh.default",
        "aten.lift_fresh_copy.default",
        "aten.log.default",
        "aten.log2.default",
        "aten.logical_not.default",
        "aten.neg.default",
        "aten.ones_like.default",
        "aten.reciprocal.default",
        "aten.relu.default",
        "aten.round.default",
        "aten.rsqrt.default",
        "aten.sigmoid.default",
        "aten.sign.default",
        "aten.silu.default",
        "aten.sin.default",
        "aten.sinc.default",
        "aten.softplus.default",
        "aten.sqrt.default",
        "aten.tanh.default",
        "aten.tril.default",
        "aten.triu.default",
        "aten.zeros_like.default",
        "aten.empty_like.default",
        "prims.clone.default",
        "prims.cos.default",
        "prims.erf.default",
        "prims.neg.default",
        "prims.reciprocal.default",
        "prims.rsqrt.default",
        "prims.sin.default",
        "prims.sqrt.default",
        "prims.tanh.default",
        "prims.view_of.default",
    )
    for op in FLOAT_UNARY:
        for tag, mk in both_2d(torch.float32):
            record(op, tag, lambda op=op, mk=mk: call_op(op, mk()))

    for tag, mk in both_2d(torch.int64):
        record("aten.bitwise_not.default", tag, lambda mk=mk: call_op("aten.bitwise_not.default", mk()))
        record("aten.one_hot.default", tag, lambda mk=mk: call_op("aten.one_hot.default", mk(), 4))

    for tag, mk in both_2d(torch.float32):
        record("aten.clamp.default", tag, lambda mk=mk: call_op("aten.clamp.default", mk(), -1.0, 1.0))
        record("aten.clamp_min.default", tag, lambda mk=mk: call_op("aten.clamp_min.default", mk(), 0.0))
        record("aten.clip.default", tag, lambda mk=mk: call_op("aten.clip.default", mk(), -1.0, 1.0))
        record("aten.round.decimals", tag, lambda mk=mk: call_op("aten.round.decimals", mk(), decimals=2))
        record("aten.full_like.default", tag, lambda mk=mk: call_op("aten.full_like.default", mk(), 3.5))
        record("aten.native_dropout.default", tag, lambda mk=mk: call_op("aten.native_dropout.default", mk(), 0.5, False))
        record("aten.native_dropout.default", tag + "_train", lambda mk=mk: call_op("aten.native_dropout.default", mk(), 0.5, True))
        record("aten._to_copy.default", tag + "_same", lambda mk=mk: call_op("aten._to_copy.default", mk()))
        record("aten._to_copy.default", tag + "_f64", lambda mk=mk: call_op("aten._to_copy.default", mk(), dtype=torch.float64))
        record("aten.new_empty.default", tag, lambda mk=mk: call_op("aten.new_empty.default", mk(), [0, 2]))
        record("aten.new_full.default", tag, lambda mk=mk: call_op("aten.new_full.default", mk(), [0, 2], 1.0))
        record("aten.new_ones.default", tag, lambda mk=mk: call_op("aten.new_ones.default", mk(), [0, 2]))
        record("aten.new_zeros.default", tag, lambda mk=mk: call_op("aten.new_zeros.default", mk(), [0, 2]))
        record("aten._local_scalar_dense.default", tag, lambda mk=mk: call_op("aten._local_scalar_dense.default", mk()))
        record("prims.broadcast_in_dim.default", tag, lambda mk=mk: call_op("prims.broadcast_in_dim.default", mk(), [2, 0, 3], [1, 2]))
        record("prims.split_dim.default", tag, lambda mk=mk: call_op("prims.split_dim.default", mk(), 0, 2))
        record("prims.transpose.default", tag, lambda mk=mk: call_op("prims.transpose.default", mk(), [1, 0]))

elif family == "views_and_copies":
    for tag, mk in both_2d(torch.float32):
        record("aten.alias.default", tag, lambda mk=mk: call_op("aten.alias.default", mk()))
        record("aten.alias_copy.default", tag, lambda mk=mk: call_op("aten.alias_copy.default", mk()))
        record("aten.as_strided.default", tag, lambda mk=mk: call_op("aten.as_strided.default", mk(), [0, 2], [2, 1], 0))
        record("aten.as_strided_copy.default", tag, lambda mk=mk: call_op("aten.as_strided_copy.default", mk(), [0, 2], [2, 1], 0))
        record("aten.clone.default", tag, lambda mk=mk: call_op("aten.clone.default", mk()))
        record("aten.contiguous.default", tag, lambda mk=mk: call_op("aten.contiguous.default", mk()))
        record("aten.detach_copy.default", tag, lambda mk=mk: call_op("aten.detach_copy.default", mk()))
        record("aten.expand.default", tag, lambda mk=mk: call_op("aten.expand.default", mk(), [2, -1, -1]))
        record("aten.expand_copy.default", tag, lambda mk=mk: call_op("aten.expand_copy.default", mk(), [2, -1, -1]))
        record("aten.permute.default", tag, lambda mk=mk: call_op("aten.permute.default", mk(), [1, 0]))
        record("aten.permute_copy.default", tag, lambda mk=mk: call_op("aten.permute_copy.default", mk(), [1, 0]))
        record("aten.reshape.default", tag, lambda mk=mk: call_op("aten.reshape.default", mk(), [0]))
        record("aten.reshape_as.default", tag, lambda mk=mk: call_op("aten.reshape_as.default", mk(), torch.zeros(0)))
        record("aten.select.int", tag, lambda mk=mk: call_op("aten.select.int", mk(), 1, 0) if mk().shape[1] > 0 else call_op("aten.select.int", mk(), 0, 0))
        record("aten.select_copy.int", tag, lambda mk=mk: call_op("aten.select_copy.int", mk(), 1, 0) if mk().shape[1] > 0 else call_op("aten.select_copy.int", mk(), 0, 0))
        record("aten.slice.Tensor", tag, lambda mk=mk: call_op("aten.slice.Tensor", mk(), 0, 0, 0, 1))
        record("aten.slice_copy.Tensor", tag, lambda mk=mk: call_op("aten.slice_copy.Tensor", mk(), 0, 0, 0, 1))
        record("aten.split.Tensor", tag, lambda mk=mk: call_op("aten.split.Tensor", mk(), 2, 0))
        record("aten.split_copy.Tensor", tag, lambda mk=mk: call_op("aten.split_copy.Tensor", mk(), 2, 0))
        record("aten.unsafe_split.Tensor", tag, lambda mk=mk: call_op("aten.unsafe_split.Tensor", mk(), 2, 0))
        record("aten.split_with_sizes.default", tag, lambda mk=mk: call_op("aten.split_with_sizes.default", mk(), [0, 0], 0))
        record("aten.split_with_sizes_copy.default", tag, lambda mk=mk: call_op("aten.split_with_sizes_copy.default", mk(), [0, 0], 0))
        record("aten.squeeze.default", tag, lambda mk=mk: call_op("aten.squeeze.default", mk()))
        record("aten.squeeze.dim", tag, lambda mk=mk: call_op("aten.squeeze.dim", mk(), 0))
        record("aten.squeeze.dims", tag, lambda mk=mk: call_op("aten.squeeze.dims", mk(), [0]))
        record("aten.squeeze_copy.default", tag, lambda mk=mk: call_op("aten.squeeze_copy.default", mk()))
        record("aten.squeeze_copy.dim", tag, lambda mk=mk: call_op("aten.squeeze_copy.dim", mk(), 0))
        record("aten.squeeze_copy.dims", tag, lambda mk=mk: call_op("aten.squeeze_copy.dims", mk(), [0]))
        record("aten.t.default", tag, lambda mk=mk: call_op("aten.t.default", mk()))
        record("aten.t_copy.default", tag, lambda mk=mk: call_op("aten.t_copy.default", mk()))
        record("aten.transpose.int", tag, lambda mk=mk: call_op("aten.transpose.int", mk(), 0, 1))
        record("aten.transpose_copy.int", tag, lambda mk=mk: call_op("aten.transpose_copy.int", mk(), 0, 1))
        record("aten.unbind.int", tag, lambda mk=mk: call_op("aten.unbind.int", mk(), 0))
        record("aten.unbind_copy.int", tag, lambda mk=mk: call_op("aten.unbind_copy.int", mk(), 0))
        record("aten.unflatten.int", tag, lambda mk=mk: call_op("aten.unflatten.int", mk(), 0, [0, 2]))
        record("aten.unfold.default", tag, lambda mk=mk: call_op("aten.unfold.default", mk(), 0, 1, 1))
        record("aten.unfold_copy.default", tag, lambda mk=mk: call_op("aten.unfold_copy.default", mk(), 0, 1, 1))
        record("aten.unsqueeze.default", tag, lambda mk=mk: call_op("aten.unsqueeze.default", mk(), 0))
        record("aten.unsqueeze_copy.default", tag, lambda mk=mk: call_op("aten.unsqueeze_copy.default", mk(), 0))
        record("aten.view.default", tag, lambda mk=mk: call_op("aten.view.default", mk(), [0]))
        record("aten._unsafe_view.default", tag, lambda mk=mk: call_op("aten._unsafe_view.default", mk(), [0]))
        record("aten.view_as.default", tag, lambda mk=mk: call_op("aten.view_as.default", mk(), torch.zeros(0)))
        record("aten.view.dtype", tag, lambda mk=mk: call_op("aten.view.dtype", mk(), torch.int32))
        record("aten.view_copy.default", tag, lambda mk=mk: call_op("aten.view_copy.default", mk(), [0]))
        record("aten.view_copy.dtype", tag, lambda mk=mk: call_op("aten.view_copy.dtype", mk(), torch.int32))
        record("aten.chunk.default", tag, lambda mk=mk: call_op("aten.chunk.default", mk(), 2, 0))
        record("aten.diag.default", tag, lambda mk=mk: call_op("aten.diag.default", mk()))
        record("aten.diag_embed.default", tag, lambda mk=mk: call_op("aten.diag_embed.default", mk()))
        record("aten.diagonal.default", tag, lambda mk=mk: call_op("aten.diagonal.default", mk()))
        record("aten.diagonal_copy.default", tag, lambda mk=mk: call_op("aten.diagonal_copy.default", mk()))
        record("aten.flip.default", tag, lambda mk=mk: call_op("aten.flip.default", mk(), [0]))
        record("aten.repeat.default", tag, lambda mk=mk: call_op("aten.repeat.default", mk(), [2, 3]))
        record("aten.roll.default", tag, lambda mk=mk: call_op("aten.roll.default", mk(), [1], [0]))
        record("aten.broadcast_tensors.default", tag, lambda mk=mk: call_op("aten.broadcast_tensors.default", [mk(), mk()]))

elif family == "binary_and_comparisons":
    TENSOR_BINARY = (
        "aten.add.Tensor",
        "aten.div.Tensor",
        "aten.eq.Tensor",
        "aten.floor_divide.default",
        "aten.fmod.Tensor",
        "aten.ge.Tensor",
        "aten.greater.Tensor",
        "aten.gt.Tensor",
        "aten.isin.Tensor_Tensor",
        "aten.le.Tensor",
        "aten.logical_and.default",
        "aten.lt.Tensor",
        "aten.max.other",
        "aten.maximum.default",
        "aten.min.other",
        "aten.minimum.default",
        "aten.mul.Tensor",
        "aten.multiply.Tensor",
        "aten.ne.Tensor",
        "aten.polar.default",
        "aten.pow.Tensor_Tensor",
        "aten.remainder.Tensor",
        "aten.sub.Tensor",
    )
    for op in TENSOR_BINARY:
        for tag, mk in both_2d(torch.float32):
            record(op, tag, lambda op=op, mk=mk: call_op(op, mk(), mk()))

    SCALAR_BINARY = (
        "aten.add.Scalar",
        "aten.div.Scalar",
        "aten.eq.Scalar",
        "aten.floor_divide.Scalar",
        "aten.fmod.Scalar",
        "aten.ge.Scalar",
        "aten.greater.Scalar",
        "aten.gt.Scalar",
        "aten.le.Scalar",
        "aten.lt.Scalar",
        "aten.mul.Scalar",
        "aten.multiply.Scalar",
        "aten.ne.Scalar",
        "aten.pow.Tensor_Scalar",
        "aten.remainder.Scalar",
        "aten.rsub.Scalar",
        "aten.sub.Scalar",
    )
    for op in SCALAR_BINARY:
        for tag, mk in both_2d(torch.float32):
            record(op, tag, lambda op=op, mk=mk: call_op(op, mk(), 2.0))

    for tag, mk in both_2d(torch.float32):
        record("aten.pow.Scalar", tag, lambda mk=mk: call_op("aten.pow.Scalar", 2.0, mk()))
        record("aten.div.Scalar_mode", tag, lambda mk=mk: call_op("aten.div.Scalar_mode", mk(), 2.0, rounding_mode="trunc"))
        record("aten.div.Tensor_mode", tag, lambda mk=mk: call_op("aten.div.Tensor_mode", mk(), mk(), rounding_mode="trunc"))
        record("aten.allclose.default", tag, lambda mk=mk: call_op("aten.allclose.default", mk(), mk()))
        record("aten.equal.default", tag, lambda mk=mk: call_op("aten.equal.default", mk(), mk()))
        record("aten.where.self", tag, lambda mk=mk: call_op("aten.where.self", mk().to(torch.bool), mk(), mk()))
        record("aten.where.ScalarSelf", tag, lambda mk=mk: call_op("aten.where.ScalarSelf", mk().to(torch.bool), 1.0, mk()))
        record("aten.where.ScalarOther", tag, lambda mk=mk: call_op("aten.where.ScalarOther", mk().to(torch.bool), mk(), 1.0))
        record("aten.where.Scalar", tag, lambda mk=mk: call_op("aten.where.Scalar", mk().to(torch.bool), 1.0, 0.0))
        record("aten.where.default", tag, lambda mk=mk: call_op("aten.where.default", mk().to(torch.bool)))

    INT_BINARY_TENSOR = (
        "aten.bitwise_and.Tensor",
        "aten.bitwise_or.Tensor",
        "aten.bitwise_xor.Tensor",
    )
    for op in INT_BINARY_TENSOR:
        for tag, mk in both_2d(torch.int64):
            record(op, tag, lambda op=op, mk=mk: call_op(op, mk(), mk()))

    INT_BINARY_SCALAR = (
        "aten.bitwise_and.Scalar",
        "aten.bitwise_or.Scalar",
        "aten.bitwise_xor.Scalar",
    )
    for op in INT_BINARY_SCALAR:
        for tag, mk in both_2d(torch.int64):
            record(op, tag, lambda op=op, mk=mk: call_op(op, mk(), 3))

elif family == "reductions_and_norms":
    for tag, mk in both_2d(torch.float32):
        record("aten.all.default", tag, lambda mk=mk: call_op("aten.all.default", mk().to(torch.bool)))
        record("aten.all.dim", tag, lambda mk=mk: call_op("aten.all.dim", mk().to(torch.bool), 0))
        record("aten.all.dims", tag, lambda mk=mk: call_op("aten.all.dims", mk().to(torch.bool), [0]))
        record("aten.any.default", tag, lambda mk=mk: call_op("aten.any.default", mk().to(torch.bool)))
        record("aten.any.dim", tag, lambda mk=mk: call_op("aten.any.dim", mk().to(torch.bool), 0))
        record("aten.any.dims", tag, lambda mk=mk: call_op("aten.any.dims", mk().to(torch.bool), [0]))
        record("aten._is_all_true.default", tag, lambda mk=mk: call_op("aten._is_all_true.default", mk().to(torch.bool)))
        record("aten.amax.default", tag + "_dim0", lambda mk=mk: call_op("aten.amax.default", mk(), [0]))
        record("aten.amax.default", tag + "_dim1", lambda mk=mk: call_op("aten.amax.default", mk(), [1]))
        record("aten.aminmax.default", tag, lambda mk=mk: call_op("aten.aminmax.default", mk()))
        record("aten.aminmax.default", tag + "_dim0", lambda mk=mk: call_op("aten.aminmax.default", mk(), dim=0))
        record("aten.aminmax.default", tag + "_dim1_keepdim", lambda mk=mk: call_op("aten.aminmax.default", mk(), dim=1, keepdim=True))
        record("aten.argmax.default", tag, lambda mk=mk: call_op("aten.argmax.default", mk()))
        record("aten.cumprod.default", tag, lambda mk=mk: call_op("aten.cumprod.default", mk(), 0))
        record("aten.cumsum.default", tag, lambda mk=mk: call_op("aten.cumsum.default", mk(), 0))
        record("aten.diff.default", tag, lambda mk=mk: call_op("aten.diff.default", mk(), 1, 0))
        record("aten.histc.default", tag, lambda mk=mk: call_op("aten.histc.default", mk(), 4, 0.0, 1.0))
        record("aten.linalg_vector_norm.default", tag, lambda mk=mk: call_op("aten.linalg_vector_norm.default", mk()))
        record("aten.logsumexp.default", tag + "_dim0", lambda mk=mk: call_op("aten.logsumexp.default", mk(), [0]))
        record("aten.logsumexp.default", tag + "_dim1", lambda mk=mk: call_op("aten.logsumexp.default", mk(), [1]))
        record("aten.max.default", tag, lambda mk=mk: call_op("aten.max.default", mk()))
        record("aten.max.dim", tag + "_dim0", lambda mk=mk: call_op("aten.max.dim", mk(), 0))
        record("aten.max.dim", tag + "_dim1", lambda mk=mk: call_op("aten.max.dim", mk(), 1))
        record("aten.mean.default", tag, lambda mk=mk: call_op("aten.mean.default", mk()))
        record("aten.mean.dim", tag, lambda mk=mk: call_op("aten.mean.dim", mk(), [0]))
        record("aten.min.default", tag, lambda mk=mk: call_op("aten.min.default", mk()))
        record("aten.min.dim", tag + "_dim0", lambda mk=mk: call_op("aten.min.dim", mk(), 0))
        record("aten.min.dim", tag + "_dim1", lambda mk=mk: call_op("aten.min.dim", mk(), 1))
        record("aten.nanmean.default", tag, lambda mk=mk: call_op("aten.nanmean.default", mk()))
        record("aten.nansum.default", tag, lambda mk=mk: call_op("aten.nansum.default", mk()))
        record("aten.norm.ScalarOpt_dim", tag, lambda mk=mk: call_op("aten.norm.ScalarOpt_dim", mk(), 2.0, [0]))
        record("aten.prod.default", tag, lambda mk=mk: call_op("aten.prod.default", mk()))
        record("aten.prod.dim_int", tag, lambda mk=mk: call_op("aten.prod.dim_int", mk(), 0))
        record("aten.std.default", tag, lambda mk=mk: call_op("aten.std.default", mk()))
        record("aten.std.dim", tag, lambda mk=mk: call_op("aten.std.dim", mk(), [0]))
        record("aten.std.correction", tag, lambda mk=mk: call_op("aten.std.correction", mk(), [0]))
        record("aten.sum.default", tag, lambda mk=mk: call_op("aten.sum.default", mk()))
        record("aten.sum.dim_IntList", tag + "_dim0", lambda mk=mk: call_op("aten.sum.dim_IntList", mk(), [0]))
        record("aten.sum.dim_IntList", tag + "_dim1", lambda mk=mk: call_op("aten.sum.dim_IntList", mk(), [1]))
        record("aten.var.default", tag, lambda mk=mk: call_op("aten.var.default", mk()))
        record("aten.var.dim", tag, lambda mk=mk: call_op("aten.var.dim", mk(), [0]))
        record("aten.var.correction", tag, lambda mk=mk: call_op("aten.var.correction", mk(), [0]))
        record("aten.var_mean.default", tag, lambda mk=mk: call_op("aten.var_mean.default", mk()))
        record("aten.var_mean.dim", tag, lambda mk=mk: call_op("aten.var_mean.dim", mk(), [0]))
        record("aten.var_mean.correction", tag, lambda mk=mk: call_op("aten.var_mean.correction", mk(), [0]))
        record("aten._log_softmax.default", tag, lambda mk=mk: call_op("aten._log_softmax.default", mk(), -1, False))
        record("aten._safe_softmax.default", tag, lambda mk=mk: call_op("aten._safe_softmax.default", mk(), -1))
        record("aten._softmax.default", tag, lambda mk=mk: call_op("aten._softmax.default", mk(), -1, False))
        record("aten._weight_norm_interface.default", tag, lambda mk=mk: call_op("aten._weight_norm_interface.default", mk(), mk(), 0))
        record("aten.nll_loss_forward.default", tag, lambda mk=mk: call_op("aten.nll_loss_forward.default", mk(), make_past_1d(torch.int64), None, 1, -100))

    record("aten.native_layer_norm.default", "past", lambda: call_op("aten.native_layer_norm.default", make_past_2d(), [3], torch.ones(3), torch.zeros(3), 1e-5))
    record("aten.native_layer_norm.default", "clone", lambda: call_op("aten.native_layer_norm.default", make_clone_2d(), [0], torch.ones(0), torch.zeros(0), 1e-5))
    record("aten.rms_norm.default", "past", lambda: call_op("aten.rms_norm.default", make_past_2d(), [3], torch.ones(3), 1e-5))
    record("aten.rms_norm.default", "clone", lambda: call_op("aten.rms_norm.default", make_clone_2d(), [0], torch.ones(0), 1e-5))

    for tag, mk in both_4d(torch.float32):
        record("aten.native_batch_norm.default", tag, lambda mk=mk: call_op("aten.native_batch_norm.default", mk(), torch.ones(2), torch.zeros(2), torch.zeros(2), torch.ones(2), False, 0.1, 1e-5))
        record("aten.native_group_norm.default", tag, lambda mk=mk: call_op("aten.native_group_norm.default", mk(), torch.ones(2), torch.zeros(2), mk().shape[0], 2, mk().shape[2] * mk().shape[3], 1, 1e-5))

elif family == "linalg_and_attention":
    record("aten.mm.default", "past", lambda: call_op("aten.mm.default", make_past_2d(), torch.zeros(3, 2)))
    record("aten.mm.default", "clone", lambda: call_op("aten.mm.default", make_clone_2d(), torch.zeros(0, 2)))
    record("aten.matmul.default", "past", lambda: call_op("aten.matmul.default", make_past_2d(), torch.zeros(3, 2)))
    record("aten.matmul.default", "clone", lambda: call_op("aten.matmul.default", make_clone_2d(), torch.zeros(0, 2)))
    record("aten.addmm.default", "past", lambda: call_op("aten.addmm.default", torch.zeros(0, 2), make_past_2d(), torch.zeros(3, 2)))
    record("aten.addmm.default", "clone", lambda: call_op("aten.addmm.default", torch.zeros(3, 2), make_clone_2d(), torch.zeros(0, 2)))
    record("aten.bmm.default", "past", lambda: call_op("aten.bmm.default", make_past_3d(), make_past_3d().transpose(1, 2)))
    record("aten.bmm.default", "clone", lambda: call_op("aten.bmm.default", make_clone_3d(), torch.zeros(2, 0, 2)))
    record("aten.baddbmm.default", "past", lambda: call_op("aten.baddbmm.default", torch.zeros(0, 2, 2), make_past_3d(), make_past_3d().transpose(1, 2)))
    record("aten.baddbmm.default", "clone", lambda: call_op("aten.baddbmm.default", torch.zeros(2, 3, 2), make_clone_3d(), torch.zeros(2, 0, 2)))
    record("aten.linalg_qr.default", "past", lambda: call_op("aten.linalg_qr.default", make_past_2d()))
    record("aten.linalg_qr.default", "clone", lambda: call_op("aten.linalg_qr.default", make_clone_2d()))
    record("aten.glu.default", "past", lambda: call_op("aten.glu.default", make_past_3d(), -1))
    record("aten.glu.default", "clone", lambda: call_op("aten.glu.default", make_clone_3d(), -1))
    record("aten._grouped_mm.default", "past", lambda: call_op("aten._grouped_mm.default", make_past_2d(torch.bfloat16), torch.zeros(1, 3, 4, dtype=torch.bfloat16), offs=torch.tensor([0], dtype=torch.int32)))
    record("aten._grouped_mm.default", "clone", lambda: call_op("aten._grouped_mm.default", make_clone_2d(torch.bfloat16), torch.zeros(1, 0, 4, dtype=torch.bfloat16), offs=torch.tensor([3], dtype=torch.int32)))

    record(
        "aten._scaled_dot_product_flash_attention_for_cpu.default",
        "past_batch0",
        lambda: call_op(
            "aten._scaled_dot_product_flash_attention_for_cpu.default",
            make_past_4d(),
            torch.zeros(0, 2, 4, 4),
            torch.zeros(0, 2, 4, 4),
        ),
    )
    record(
        "aten._scaled_dot_product_flash_attention_for_cpu.default",
        "clone_qlen0",
        lambda: call_op(
            "aten._scaled_dot_product_flash_attention_for_cpu.default",
            make_clone_4d().transpose(2, 3),
            torch.zeros(2, 2, 3, 3),
            torch.zeros(2, 2, 3, 3),
        ),
    )

elif family == "conv_pool_upsample_pad_fft_rnn":
    w1 = torch.zeros(2, 2, 1)
    w2 = torch.zeros(2, 2, 1, 1)
    record("aten.convolution.default", "past_1d", lambda: call_op("aten.convolution.default", make_past_3d(), w1, None, [1], [0], [1], False, [0], 1))
    record("aten.convolution.default", "clone_1d", lambda: call_op("aten.convolution.default", make_clone_3d(), w1, None, [1], [0], [1], False, [0], 1))
    record("aten.convolution.default", "past_2d", lambda: call_op("aten.convolution.default", make_past_4d(), w2, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1))
    record("aten.convolution.default", "clone_2d", lambda: call_op("aten.convolution.default", make_clone_4d(), w2, None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1))
    record("aten.convolution.default", "past_conv_t1d", lambda: call_op("aten.convolution.default", make_past_3d(), w1, None, [1], [0], [1], True, [0], 1))
    record("aten.convolution.default", "past_conv_t2d", lambda: call_op("aten.convolution.default", make_past_4d(), w2, None, [1, 1], [0, 0], [1, 1], True, [0, 0], 1))

    for tag, mk in both_3d(torch.float32):
        record("aten.adaptive_avg_pool1d.default", tag, lambda mk=mk: call_op("aten.adaptive_avg_pool1d.default", mk(), [1]))
        record("aten.max_pool1d.default", tag, lambda mk=mk: call_op("aten.max_pool1d.default", mk(), [1], [1], [0], [1], False))
        record("aten.upsample_linear1d.default", tag, lambda mk=mk: call_op("aten.upsample_linear1d.default", mk(), [2], False))
        record("aten.upsample_nearest1d.default", tag, lambda mk=mk: call_op("aten.upsample_nearest1d.default", mk(), [2]))
        record("aten.reflection_pad1d.default", tag, lambda mk=mk: call_op("aten.reflection_pad1d.default", mk(), [0, 0]))
        record("aten.replication_pad1d.default", tag, lambda mk=mk: call_op("aten.replication_pad1d.default", mk(), [0, 0]))
        record("aten.col2im.default", tag, lambda mk=mk: call_op("aten.col2im.default", mk(), [2, 2], [1, 1], [1, 1], [0, 0], [1, 1]))

    for tag, mk in both_4d(torch.float32):
        record("aten.adaptive_avg_pool2d.default", tag, lambda mk=mk: call_op("aten.adaptive_avg_pool2d.default", mk(), [1, 1]))
        record("aten.avg_pool2d.default", tag, lambda mk=mk: call_op("aten.avg_pool2d.default", mk(), [1, 1], [1, 1], [0, 0], False, True, None))
        record("aten._adaptive_avg_pool2d.default", tag, lambda mk=mk: call_op("aten._adaptive_avg_pool2d.default", mk(), [1, 1]))
        record("aten.max_pool2d.default", tag, lambda mk=mk: call_op("aten.max_pool2d.default", mk(), [1, 1], [1, 1], [0, 0], [1, 1], False))
        record("aten.max_pool2d_with_indices.default", tag, lambda mk=mk: call_op("aten.max_pool2d_with_indices.default", mk(), [1, 1], [1, 1], [0, 0], [1, 1], False))
        record("aten.im2col.default", tag, lambda mk=mk: call_op("aten.im2col.default", mk(), [1, 1], [1, 1], [0, 0], [1, 1]))
        record("aten.upsample_bicubic2d.default", tag, lambda mk=mk: call_op("aten.upsample_bicubic2d.default", mk(), [2, 2], False))
        record("aten.upsample_bilinear2d.default", tag, lambda mk=mk: call_op("aten.upsample_bilinear2d.default", mk(), [2, 2], False))
        record("aten.upsample_nearest2d.default", tag, lambda mk=mk: call_op("aten.upsample_nearest2d.default", mk(), [2, 2]))
        record("aten.reflection_pad2d.default", tag, lambda mk=mk: call_op("aten.reflection_pad2d.default", mk(), [0, 0, 0, 0]))
        record("aten.replication_pad2d.default", tag, lambda mk=mk: call_op("aten.replication_pad2d.default", mk(), [0, 0, 0, 0]))
        record("aten.constant_pad_nd.default", tag, lambda mk=mk: call_op("aten.constant_pad_nd.default", mk(), [0, 0]))

    for tag in ("past", "clone"):
        mk5 = make_past_5d if tag == "past" else make_clone_5d
        record("aten.reflection_pad3d.default", tag, lambda mk5=mk5: call_op("aten.reflection_pad3d.default", mk5(), [0, 0, 0, 0, 0, 0]))
        record("aten.replication_pad3d.default", tag, lambda mk5=mk5: call_op("aten.replication_pad3d.default", mk5(), [0, 0, 0, 0, 0, 0]))

    for tag, mk in both_2d(torch.float32):
        record("aten.complex.default", tag, lambda mk=mk: call_op("aten.complex.default", mk(), mk()))
        record("aten.real.default", tag, lambda mk=mk: call_op("aten.real.default", mk()))
        record("aten.imag.default", tag, lambda mk=mk: call_op("aten.imag.default", call_op("aten.complex.default", mk(), mk())))
        record("aten._fft_r2c.default", tag, lambda mk=mk: call_op("aten._fft_r2c.default", mk(), [0], 0, True))
        record("aten._fft_c2c.default", tag, lambda mk=mk: call_op("aten._fft_c2c.default", call_op("aten.complex.default", mk(), mk()), [0], 0, True))
        record("aten._fft_c2r.default", tag, lambda mk=mk: call_op("aten._fft_c2r.default", call_op("aten.complex.default", mk(), mk()), [0], 0, 0))
        record("aten.stft.default", tag, lambda mk=mk: call_op("aten.stft.default", mk(), 4, 2, 4, None, False, True, False))
        record("aten.stft.center", tag, lambda mk=mk: call_op("aten.stft.center", mk(), 4, 2, 4, None, False, "reflect", False, True, False))

    record("aten.view_as_complex.default", "past", lambda: call_op("aten.view_as_complex.default", torch.zeros(12).as_strided_((0, 2), (2, 1), 100)))
    record("aten.view_as_complex.default", "clone", lambda: call_op("aten.view_as_complex.default", make_clone_3d()[:, :, :2] if make_clone_3d().shape[-1] >= 2 else torch.zeros(12).as_strided_((0, 2), (2, 1), 100)))
    record("aten.view_as_real.default", "past", lambda: call_op("aten.view_as_real.default", call_op("aten.complex.default", make_past_2d(), make_past_2d())))
    record("aten.view_as_real.default", "clone", lambda: call_op("aten.view_as_real.default", call_op("aten.complex.default", make_clone_2d(), make_clone_2d())))

    h0 = torch.zeros(1, 2, 4)
    c0 = torch.zeros(1, 2, 4)
    w_ih = torch.zeros(16, 4)
    w_hh = torch.zeros(16, 4)
    b_ih = torch.zeros(16)
    b_hh = torch.zeros(16)
    record("aten.lstm.input", "past", lambda: call_op("aten.lstm.input", make_past_3d(), [h0, c0], [w_ih, w_hh, b_ih, b_hh], True, 1, 0.0, False, False, False))
    record("aten.lstm.input", "clone", lambda: call_op("aten.lstm.input", make_clone_3d(), [h0, c0], [w_ih, w_hh, b_ih, b_hh], True, 1, 0.0, False, False, False))

elif family == "indexing_scatter_cat_sort":
    for tag, mk in both_2d(torch.float32):
        record("aten.argsort.default", tag, lambda mk=mk: call_op("aten.argsort.default", mk(), -1, False))
        record("aten.argsort.stable", tag, lambda mk=mk: call_op("aten.argsort.stable", mk(), stable=True, dim=-1, descending=False))
        record("aten.sort.default", tag, lambda mk=mk: call_op("aten.sort.default", mk(), -1, False))
        record("aten.sort.stable", tag, lambda mk=mk: call_op("aten.sort.stable", mk(), stable=True, dim=-1, descending=False))
        record("aten.topk.default", tag, lambda mk=mk: call_op("aten.topk.default", mk(), 0, -1, True, True))
        record("aten.cat.default", tag + "_dim0", lambda mk=mk: call_op("aten.cat.default", [mk(), mk()], 0))
        record("aten.cat.default", tag + "_dim1", lambda mk=mk: call_op("aten.cat.default", [mk(), mk()], 1))
        record("aten.stack.default", tag, lambda mk=mk: call_op("aten.stack.default", [mk(), mk()], 0))
        record("aten.bucketize.Tensor", tag, lambda mk=mk: call_op("aten.bucketize.Tensor", mk(), torch.tensor([0.0, 1.0])))
        record("aten.bucketize.Scalar", tag, lambda mk=mk: call_op("aten.bucketize.Scalar", 0.5, mk().reshape(0)))
        record("aten.gather.default", tag, lambda mk=mk: call_op("aten.gather.default", mk(), 0, mk().to(torch.int64)))
        record("aten.index.Tensor", tag, lambda mk=mk: call_op("aten.index.Tensor", mk(), [make_past_1d(torch.int64)]))
        record("aten._unsafe_index.Tensor", tag, lambda mk=mk: call_op("aten._unsafe_index.Tensor", mk(), [make_past_1d(torch.int64)]))
        record("aten.index_select.default", tag, lambda mk=mk: call_op("aten.index_select.default", mk(), 0, make_past_1d(torch.int64)))
        record("aten.index_add.default", tag, lambda mk=mk: call_op("aten.index_add.default", mk(), 0, make_past_1d(torch.int64), mk()))
        record("aten.index_copy.default", tag, lambda mk=mk: call_op("aten.index_copy.default", mk(), 0, make_past_1d(torch.int64), mk()))
        record("aten.masked_fill.Scalar", tag, lambda mk=mk: call_op("aten.masked_fill.Scalar", mk(), mk().to(torch.bool), 1.0))
        record("aten.masked_fill.Tensor", tag, lambda mk=mk: call_op("aten.masked_fill.Tensor", mk(), mk().to(torch.bool), torch.tensor(1.0)))
        record("aten.masked_scatter.default", tag, lambda mk=mk: call_op("aten.masked_scatter.default", mk(), mk().to(torch.bool), mk()))
        record("aten.masked_select.default", tag, lambda mk=mk: call_op("aten.masked_select.default", mk(), mk().to(torch.bool)))
        record("aten.nonzero.default", tag, lambda mk=mk: call_op("aten.nonzero.default", mk()))
        record("aten._unique2.default", tag, lambda mk=mk: call_op("aten._unique2.default", mk(), True, False, False))
        record("aten.scatter.src", tag, lambda mk=mk: call_op("aten.scatter.src", mk(), 0, mk().to(torch.int64), mk()))
        record("aten.scatter.value", tag, lambda mk=mk: call_op("aten.scatter.value", mk(), 0, mk().to(torch.int64), 1.0))
        record("aten.scatter_reduce.two", tag, lambda mk=mk: call_op("aten.scatter_reduce.two", mk(), 0, mk().to(torch.int64), mk(), "sum"))
        record("aten.multinomial.default", tag, lambda mk=mk: call_op("aten.multinomial.default", mk(), 1, True))

    record("aten.cat.default", "clone_with_nonempty_dim1", lambda: call_op("aten.cat.default", [make_clone_2d(), torch.ones(3, 2)], 1))
    record("aten.embedding.default", "past", lambda: call_op("aten.embedding.default", torch.zeros(4, 3), make_past_2d(torch.int64)))
    record("aten.embedding.default", "clone", lambda: call_op("aten.embedding.default", torch.zeros(4, 3), make_clone_2d(torch.int64)))
    record("aten.repeat_interleave.Tensor", "past", lambda: call_op("aten.repeat_interleave.Tensor", make_past_1d(torch.int64)))
    record("aten.repeat_interleave.Tensor", "clone", lambda: call_op("aten.repeat_interleave.Tensor", make_clone_1d(torch.int64)))

elif family == "inplace":
    INPLACE_UNARY = (
        "aten.abs_.default",
        "aten.ceil_.default",
        "aten.cos_.default",
        "aten.erf_.default",
        "aten.exp_.default",
        "aten.expm1_.default",
        "aten.floor_.default",
        "aten.log2_.default",
        "aten.log_.default",
        "aten.logical_not_.default",
        "aten.neg_.default",
        "aten.reciprocal_.default",
        "aten.relu_.default",
        "aten.round_.default",
        "aten.rsqrt_.default",
        "aten.sigmoid_.default",
        "aten.sin_.default",
        "aten.sqrt_.default",
        "aten.t_.default",
        "aten.tanh_.default",
        "aten.tril_.default",
        "aten.triu_.default",
        "aten.zero_.default",
    )
    for op in INPLACE_UNARY:
        for tag, mk in both_2d(torch.float32):
            record(op, tag, lambda op=op, mk=mk: call_op(op, mk()))

    INPLACE_SCALAR = (
        "aten.add_.Scalar",
        "aten.div_.Scalar",
        "aten.fill_.Scalar",
        "aten.mul_.Scalar",
        "aten.sub_.Scalar",
    )
    for op in INPLACE_SCALAR:
        for tag, mk in both_2d(torch.float32):
            record(op, tag, lambda op=op, mk=mk: call_op(op, mk(), 2.0))

    INPLACE_TENSOR = (
        "aten.add_.Tensor",
        "aten.copy_.default",
        "aten.div_.Tensor",
        "aten.mul_.Tensor",
        "aten.sub_.Tensor",
    )
    for op in INPLACE_TENSOR:
        for tag, mk in both_2d(torch.float32):
            record(op, tag, lambda op=op, mk=mk: call_op(op, mk(), mk()))

    for tag, mk in both_2d(torch.float32):
        record("aten.as_strided_.default", tag, lambda mk=mk: call_op("aten.as_strided_.default", mk(), [0, 3], [3, 1], 100 if is_shim or tag == "past" else 0))
        record("aten.bernoulli_.float", tag, lambda mk=mk: call_op("aten.bernoulli_.float", mk(), 0.5))
        record("aten.clamp_.default", tag, lambda mk=mk: call_op("aten.clamp_.default", mk(), -1.0, 1.0))
        record("aten.clamp_min_.default", tag, lambda mk=mk: call_op("aten.clamp_min_.default", mk(), 0.0))
        record("aten.fill_.Tensor", tag, lambda mk=mk: call_op("aten.fill_.Tensor", mk(), torch.tensor(2.0)))
        record("aten.index_add_.default", tag, lambda mk=mk: call_op("aten.index_add_.default", mk(), 0, make_past_1d(torch.int64), mk()))
        record("aten.index_copy_.default", tag, lambda mk=mk: call_op("aten.index_copy_.default", mk(), 0, make_past_1d(torch.int64), mk()))
        record("aten.index_put_.default", tag, lambda mk=mk: call_op("aten.index_put_.default", mk(), [make_past_1d(torch.int64)], mk(), False))
        record("aten.masked_fill_.Scalar", tag, lambda mk=mk: call_op("aten.masked_fill_.Scalar", mk(), mk().to(torch.bool), 1.0))
        record("aten.normal_.default", tag, lambda mk=mk: call_op("aten.normal_.default", mk(), 0.0, 1.0))
        record("aten.round_.decimals", tag, lambda mk=mk: call_op("aten.round_.decimals", mk(), decimals=2))
        record("aten.scatter_.src", tag, lambda mk=mk: call_op("aten.scatter_.src", mk(), 0, mk().to(torch.int64), mk()))
        record("aten.scatter_.value", tag, lambda mk=mk: call_op("aten.scatter_.value", mk(), 0, mk().to(torch.int64), 1.0))
        record("aten.squeeze_.dim", tag, lambda mk=mk: call_op("aten.squeeze_.dim", mk(), 0))
        record("aten.transpose_.default", tag, lambda mk=mk: call_op("aten.transpose_.default", mk(), 0, 1))
        record("aten.uniform_.default", tag, lambda mk=mk: call_op("aten.uniform_.default", mk(), 0.0, 1.0))

    record(
        "aten.index_copy_.default",
        "clone_nonempty_idx",
        lambda: call_op(
            "aten.index_copy_.default",
            make_clone_2d(),
            0,
            torch.tensor([0, 2], dtype=torch.int64),
            torch.zeros(2, 0),
        ),
    )

else:
    raise ValueError(f"unknown family: {family}")

print(
    json.dumps(
        {
            "is_shim": is_shim,
            "family": family,
            "covered": sorted(covered),
            "rows": rows,
        }
    )
)
'''


def _run_family(family, *, use_shim):
    if use_shim:
        env = dict(os.environ)
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1, as the bindings suites set it
    else:
        env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "TORCH_USE_RTLD_GLOBAL")}
    skip_rows: list[str] = []
    for _attempt in range(8):
        run_env = dict(env)
        if skip_rows:
            run_env["UPSTREAM_SKIP_ROWS"] = ",".join(skip_rows)
        proc = subprocess.run(
            [sys.executable, "-c", _PROBE, family],
            env=run_env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        if proc.returncode == 0:
            break
        if not use_shim and proc.returncode < 0:
            trace_env = dict(run_env)
            trace_env["TRACE_ROW"] = "1"
            traced = subprocess.run(
                [sys.executable, "-c", _PROBE, family],
                env=trace_env,
                capture_output=True,
                text=True,
                timeout=120,
            )
            row_lines = [
                ln[4:].strip()
                for ln in traced.stderr.splitlines()
                if ln.startswith("ROW ")
            ]
            if row_lines and row_lines[-1] not in skip_rows:
                culprit = row_lines[-1]
                sys.stderr.write(
                    f"[test_empty_storage_slice] upstream {family!r} died with "
                    f"signal {proc.returncode} on {culprit}; skipping upstream row\n"
                )
                skip_rows.append(culprit)
                continue
        break
    assert proc.returncode == 0, (
        f"family {family!r} ({'shim' if use_shim else 'upstream'}) died with "
        f"returncode {proc.returncode}:\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    )
    data = json.loads(proc.stdout)
    assert data["is_shim"] is use_shim, (family, use_shim, data["is_shim"])
    return data


def _all_implemented_on_shim():
    env = dict(os.environ)
    env["PYTHONPATH"] = _VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, torch; print(json.dumps(sorted(torch._C._aten_all_implemented())))",
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    return set(json.loads(proc.stdout))


def test_no_implemented_aten_op_panics_on_empty_offset_or_strided_clone_layouts():
    assert _available(), f"built shim missing at {_VENDOR_SHIM}"
    all_ops = _all_implemented_on_shim()
    unknown_exempt = FACTORY_OPS_WITHOUT_TENSOR_INPUT - all_ops
    assert not unknown_exempt, f"exempt ops not in _aten_all_implemented(): {sorted(unknown_exempt)}"
    expected_tensor_ops = all_ops - FACTORY_OPS_WITHOUT_TENSOR_INPUT

    covered_union = set()
    panics = []
    shape_mismatches = []

    for family in FAMILIES:
        up = _run_family(family, use_shim=False)
        sh = _run_family(family, use_shim=True)
        covered_union.update(sh["covered"])

        assert set(up["rows"]) == set(sh["rows"]), (
            family,
            sorted(set(up["rows"]) ^ set(sh["rows"])),
        )
        for row_key, sh_row in sorted(sh["rows"].items()):
            up_row = up["rows"][row_key]
            if not sh_row["ok"] and sh_row.get("is_panic"):
                panics.append((family, row_key, sh_row))
            if up_row["ok"] and sh_row["ok"]:
                if up_row["result"] != sh_row["result"]:
                    shape_mismatches.append((family, row_key, up_row["result"], sh_row["result"]))

    missing_ops = sorted(expected_tensor_ops - covered_union)
    assert not missing_ops, (
        f"tensor-taking ops in _aten_all_implemented() not exercised by any family: {missing_ops}"
    )
    assert not panics, f"Rust PanicException reached Python on empty layouts: {panics}"
    assert not shape_mismatches, (
        f"output shape/dtype disagrees with upstream on empty layouts: {shape_mismatches}"
    )


if __name__ == "__main__":
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    raise SystemExit(1 if _skip.run_tests(items, suite="test_empty_storage_slice") else 0)
