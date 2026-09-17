"""A meta tensor's stride, storage offset and storage are real -- `docs/graph/STRIDE.md`.

`docs/graph/EXPORT6.md` §6 named the change and why it had to be one change:
a stride field on `Repr::Meta`, `stride()` reading it, `docs/graph/EXPORT4.md`
§6.5's invariant test rewritten, and `as_strided` / `t` / `slice` given meta
kernels -- because any subset leaves a meta tensor whose `stride()` lies.

Measuring first showed the lie was already live: `t`, `transpose`, `permute`,
`slice`, `expand` and every pointwise op already had meta arms, and every one of
them answered through a fresh contiguous tensor.  `torch.empty(3, 4,
device="meta").t().stride()` was `(3, 1)` here and `(1, 4)` upstream.
EXPORT4's invariant test did not see it because it only ever asked a freshly
constructed tensor.

Every assertion below is **upstream's answer, computed in a separate
subprocess** from the same probe text -- never a table written beside the
implementation.  The comparisons are exact: strides and offsets are integers,
and there is no tolerance to widen.

What this file cannot see, stated so nobody reads more into it: it compares the
layouts the probe builds.  An op missing from the probe is not judged here; for
those, `test_a_stride_unaware_meta_kernel_refuses_a_non_contiguous_input`
asserts the gate that stops them answering at all.
"""

import json
import os
import subprocess
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

#: float32 machine epsilon.  The one numeric comparison in this file (the
#: exported program's replay) uses a multiple of this and of nothing else;
#: `docs/numerics/AGREE.md` is the method.
_EPS32 = 1.1920929e-07


def _available():
    return os.path.isfile(_VENDOR_SHIM)


def _run(script, shim, timeout=900):
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    # The side each probe landed on is asserted, not assumed: an empty vendored
    # tree makes the "shim" side silently import upstream (CLAUDE.md §3).
    assert out["is_shim"] is shim, f"probe meant for shim={shim} ran on the other side"
    return out


# ---------------------------------------------------------------------------
# The probe.  Identical text on both sides; only the answers can differ.
# ---------------------------------------------------------------------------

_LAYOUT_PROBE = r"""
import json
import torch

aten = torch.ops.aten
prims = torch.ops.prims
out = {"is_shim": hasattr(torch._C, "_aten_implemented"), "cases": {}}


def m(*shape):
    return torch.empty(tuple(shape), device="meta")


def describe(t, base):
    return {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "offset": t.storage_offset(),
        "contiguous": t.is_contiguous(),
        "channels_last": t.is_contiguous(memory_format=torch.channels_last)
        if t.dim() == 4 else None,
        "dtype": str(t.dtype),
        "shares_storage": None if base is None
        else t.untyped_storage()._cdata == base.untyped_storage()._cdata,
        "nbytes": t.untyped_storage().nbytes(),
        "type": type(t).__name__,
    }


def case(name, build):
    try:
        base, result = build()
        if isinstance(result, (tuple, list)):
            out["cases"][name] = [describe(r, base) for r in result]
        else:
            out["cases"][name] = describe(result, base)
    except Exception as exc:
        out["cases"][name] = {"raised": type(exc).__name__, "message": str(exc)}


def on(base, f):
    return lambda: (base, f(base))


# ---- as_strided: the op EXPORT6 §6 names ----------------------------------
case("as_strided", lambda: on(m(3, 4), lambda b: b.as_strided((4, 3), (1, 4), 2))())
case("as_strided_default_offset", lambda: on(m(5, 4), lambda b: b[1:].as_strided((2, 2), (1, 1)))())
case("as_strided_of_view", lambda: on(m(6, 4), lambda b: b.t()[:, 2:].as_strided((3, 2), (2, 1), 5))())
case("as_strided_numel_0", lambda: on(m(3, 4), lambda b: b.as_strided((0, 4), (400, 1), 50))())
case("as_strided_0d", lambda: on(m(3, 4), lambda b: b.as_strided((), (), 7))())
case("as_strided_overlap", lambda: on(m(3, 4), lambda b: b.as_strided((3, 4), (1, 1), 0))())
case("as_strided_neg_stride", lambda: on(m(3, 4), lambda b: b.as_strided((2, 2), (-1, 1)))())
case("as_strided_len_mismatch", lambda: on(m(3, 4), lambda b: b.as_strided((2, 2), (1,)))())
case("as_strided_neg_offset", lambda: on(m(3, 4), lambda b: b.as_strided((2, 2), (1, 1), -1))())
case("aten_as_strided_kw", lambda: on(m(3, 4), lambda b: aten.as_strided.default(b, [2, 3], [1, 2], storage_offset=3))())

# ---- the views that already had meta arms and answered contiguously --------
case("t", lambda: on(m(3, 4), lambda b: b.t())())
case("t_1d", lambda: on(m(5), lambda b: b.t())())
case("t_of_t", lambda: on(m(3, 4), lambda b: b.t().t())())
case("transpose", lambda: on(m(3, 4, 5), lambda b: b.transpose(0, 2))())
case("transpose_neg", lambda: on(m(3, 4, 5), lambda b: b.transpose(-1, 1))())
case("permute", lambda: on(m(3, 4, 5), lambda b: b.permute(2, 0, 1))())
case("permute_of_slice", lambda: on(m(3, 4, 6), lambda b: b[:, 1:, ::2].permute(1, 2, 0))())
case("slice", lambda: on(m(3, 4, 5), lambda b: b[:, 1:3])())
case("slice_step", lambda: on(m(3, 8), lambda b: b[:, 1::3])())
case("slice_empty", lambda: on(m(3, 8), lambda b: b[:, 6:2])())
case("slice_of_t", lambda: on(m(3, 8), lambda b: b.t()[2:5])())
case("aten_slice", lambda: on(m(3, 8), lambda b: aten.slice.Tensor(b, 1, -5, 100, 2))())
case("select", lambda: on(m(3, 4, 5), lambda b: b[1])())
case("select_neg", lambda: on(m(3, 4, 5), lambda b: b.select(2, -2))())
case("select_of_permute", lambda: on(m(3, 4, 5), lambda b: b.permute(2, 0, 1)[:, 2])())
case("expand", lambda: on(m(3, 1), lambda b: b.expand(3, 4))())
case("expand_new_dims", lambda: on(m(3, 1), lambda b: b.expand(2, 3, 4))())
case("expand_new_unit_dim", lambda: on(m(3, 4), lambda b: b.t().expand(1, 1, 4, 3))())
case("expand_of_t", lambda: on(m(1, 3), lambda b: b.t().expand(3, 5))())
case("expand_minus_one", lambda: on(m(3, 1), lambda b: b[1:].expand(-1, 6))())
case("unsqueeze_of_t", lambda: on(m(3, 4), lambda b: b.t().unsqueeze(1))())
case("unsqueeze_front_of_slice", lambda: on(m(3, 4), lambda b: b[:, 1:].unsqueeze(0))())
case("unsqueeze_back_of_slice", lambda: on(m(3, 4), lambda b: b[:, 1:].unsqueeze(-1))())
case("squeeze_all", lambda: on(m(3, 1, 5, 1), lambda b: b.permute(3, 2, 1, 0).squeeze())())
case("squeeze_dim", lambda: on(m(3, 1, 5), lambda b: b.permute(2, 1, 0).squeeze(1))())
case("squeeze_dims", lambda: on(m(3, 1, 5, 1), lambda b: b.permute(1, 2, 3, 0).squeeze((0, 2)))())
case("split", lambda: on(m(3, 7), lambda b: b.t().split(3))())
case("split_with_sizes", lambda: on(m(7, 3), lambda b: b[:, 1:].split([2, 5]))())
case("detach", lambda: on(m(3, 4), lambda b: b.t().detach())())
case("alias", lambda: on(m(3, 4), lambda b: aten.alias.default(b[:, 1:]))())
case("lift_fresh", lambda: on(m(3, 4), lambda b: aten.lift_fresh.default(b.t()))())
# in-place initialisers and copy_ return their receiver, layout and storage kept
case("zero_on_t", lambda: on(m(3, 4), lambda b: aten.zero_.default(b.t()))())
case("fill_on_slice", lambda: on(m(3, 4), lambda b: aten.fill_.Scalar(b[:, 1::2], 2.0))())
case("uniform_on_t", lambda: on(m(3, 4), lambda b: aten.uniform_.default(b.t()))())
case("normal_on_expand", lambda: on(m(3, 1), lambda b: aten.normal_.default(b.t()))())
case("copy_into_t", lambda: on(m(3, 4), lambda b: aten.copy_.default(b.t(), m(3)))())
case("prims_view_of", lambda: on(m(3, 4), lambda b: prims.view_of.default(b.t()))())
case("prims_collapse_view_gappy", lambda: on(m(2, 3, 8), lambda b: prims.collapse_view.default(b[:, :, ::2], 0, 1))())
case("prims_collapse_view_gappy_tail", lambda: on(m(2, 3, 8), lambda b: prims.collapse_view.default(b[:, :, ::2], 1, 2))())
case("prims_collapse_view_all", lambda: on(m(2, 3, 4), lambda b: prims.collapse_view.default(b[1:], 0, 2))())
case("prims_collapse_view_unit", lambda: on(m(3, 1, 4), lambda b: prims.collapse_view.default(b.permute(0, 2, 1), 1, 2))())
case("prims_collapse_view_same", lambda: on(m(3, 4), lambda b: prims.collapse_view.default(b.t(), 1, 1))())
case("prims_collapse_view_0d", lambda: on(m(), lambda b: prims.collapse_view.default(b, 0, 0))())
case("prims_collapse_view_empty", lambda: on(m(0, 3), lambda b: prims.collapse_view.default(b.t(), 0, 1))())
case("prims_collapse_view_zero_inside", lambda: on(m(3, 0, 4), lambda b: prims.collapse_view.default(b, 0, 1))())
case("prims_collapse_view_impossible", lambda: on(m(3, 4), lambda b: prims.collapse_view.default(b.t(), 0, 1))())
case("prims_collapse_view_backwards", lambda: on(m(3, 4), lambda b: prims.collapse_view.default(b, 1, 0))())
case("prims_collapse_view_out_of_range", lambda: on(m(3, 4), lambda b: prims.collapse_view.default(b, 0, 2))())
case("prims_split_dim", lambda: on(m(3, 4, 6), lambda b: prims.split_dim.default(b.permute(2, 0, 1), 0, 2))())

# ---- empty_strided: the constructor behind every fake tensor ---------------
case("empty_strided_meta_transposed", lambda: (None, torch.empty_strided((2, 3), (1, 2), device="meta")))
case("empty_strided_meta_overlap", lambda: (None, torch.empty_strided((3, 4), (1, 1), device="meta")))
case("empty_strided_meta_expanded", lambda: (None, torch.empty_strided((3, 4), (0, 1), device="meta")))
case("empty_strided_meta_empty", lambda: (None, torch.empty_strided((0, 4), (9, 1), device="meta")))
case("empty_strided_meta_channels_last", lambda: (None, torch.empty_strided((2, 3, 4, 5), (60, 1, 15, 3), device="meta")))

# ---- view: legal only where the layout allows it ---------------------------
case("view_contiguous", lambda: on(m(3, 4), lambda b: b.view(2, 6))())
case("view_of_t_splitting_nothing", lambda: on(m(3, 4), lambda b: b.t().view(4, 3, 1))())
case("view_of_t_across_subspaces", lambda: on(m(3, 4), lambda b: b.t().view(12))())
case("view_of_permute_merging_a_block", lambda: on(m(2, 3, 4), lambda b: b.permute(1, 2, 0).view(12, 2))())
case("view_of_slice_step", lambda: on(m(4, 6), lambda b: b[:, ::2].view(2, 2, 3))())
case("view_of_expand", lambda: on(m(3, 1), lambda b: b.expand(3, 4).view(3, 2, 2))())
case("view_of_empty_t", lambda: on(m(0, 3), lambda b: b.t().view(3, 1, 0))())
case("view_0d", lambda: on(m(), lambda b: b.view(1, 1))())
case("reshape_as_view", lambda: on(m(3, 4), lambda b: b.t().reshape(2, 2, 3))())
case("reshape_as_copy", lambda: on(m(3, 4), lambda b: b.t().reshape(12))())
case("reshape_contiguous", lambda: on(m(3, 4), lambda b: b.reshape(6, 2))())

# ---- ops whose output layout follows the input's ---------------------------
LAYOUTS = {
    "t": lambda: m(3, 4).t(),
    "perm3": lambda: m(3, 4, 5).permute(2, 0, 1),
    "perm_step": lambda: m(3, 4, 6).permute(2, 0, 1)[:, :, ::2],
    "step": lambda: m(3, 8)[:, ::2],
    "offset": lambda: m(5, 4)[1:3],
    "expand": lambda: m(3, 1).expand(3, 4),
    "expand_t": lambda: m(1, 3).expand(4, 3).t(),
    "channels_last": lambda: m(2, 4, 5, 3).permute(0, 3, 1, 2),
    "channels_last_c1": lambda: m(40).as_strided((2, 1, 4, 5), (20, 1, 5, 1)),
    "odd": lambda: m(100).as_strided((3, 4), (5, 20), 1),
    "overlap": lambda: m(100).as_strided((3, 4), (1, 1), 0),
    "perm_c1": lambda: m(3, 1, 5).permute(1, 2, 0),
    "nchw_c1": lambda: m(2, 1, 4, 5),
    "empty_t": lambda: m(0, 3).t(),
    "vector_step": lambda: m(9)[::3],
}
FOLLOWERS = {
    "relu": lambda x: aten.relu.default(x),
    "neg": lambda x: aten.neg.default(x),
    "sin": lambda x: aten.sin.default(x),
    "rsqrt": lambda x: aten.rsqrt.default(x),
    "cos": lambda x: aten.cos.default(x),
    "erf": lambda x: aten.erf.default(x),
    "exp": lambda x: aten.exp.default(x),
    "expm1": lambda x: aten.expm1.default(x),
    "log": lambda x: aten.log.default(x),
    "log2": lambda x: aten.log2.default(x),
    "sinc": lambda x: aten.sinc.default(x),
    "sqrt": lambda x: aten.sqrt.default(x),
    "prims_cos": lambda x: prims.cos.default(x),
    "prims_neg": lambda x: prims.neg.default(x),
    "prims_erf": lambda x: prims.erf.default(x),
    "prims_rsqrt": lambda x: prims.rsqrt.default(x),
    "prims_sqrt": lambda x: prims.sqrt.default(x),
    "prims_tanh": lambda x: prims.tanh.default(x),
    "prims_reciprocal": lambda x: prims.reciprocal.default(x),
    "clip": lambda x: aten.clip.default(x, None, 1),
    "pow_tensor": lambda x: aten.pow.Tensor_Tensor(x, x),
    "add_Scalar": lambda x: aten.add.Scalar(x, 1),
    "sub_Scalar": lambda x: aten.sub.Scalar(x, 1),
    "mul_Scalar": lambda x: aten.mul.Scalar(x, 3),
    "sub_0d": lambda x: aten.sub.Tensor(x, m()),
    "mul_0d_first": lambda x: aten.mul.Tensor(m(), x),
    "div_self": lambda x: aten.div.Tensor(x, x),
    "mul_self": lambda x: aten.mul.Tensor(x, x),
    "sub_self": lambda x: aten.sub.Tensor(x, x),
    "ne_scalar": lambda x: aten.ne.Scalar(x, 0),
    "ge_scalar": lambda x: aten.ge.Scalar(x, 0),
    "gt_scalar": lambda x: aten.gt.Scalar(x, 0),
    "le_scalar": lambda x: aten.le.Scalar(x, 0),
    "lt_scalar": lambda x: aten.lt.Scalar(x, 0),
    "eq_self": lambda x: aten.eq.Tensor(x, x),
    "ne_self": lambda x: aten.ne.Tensor(x, x),
    "ge_self": lambda x: aten.ge.Tensor(x, x),
    "gt_self": lambda x: aten.gt.Tensor(x, x),
    "le_self": lambda x: aten.le.Tensor(x, x),
    "where_scalar_other": lambda x: aten.where.ScalarOther(aten.gt.Scalar(x, 0), x, 0.0),
    "where_scalar_self": lambda x: aten.where.ScalarSelf(aten.gt.Scalar(x, 0), 0.0, x),
    "gelu": lambda x: aten.gelu.default(x),
    "silu": lambda x: aten.silu.default(x),
    "tanh": lambda x: aten.tanh.default(x),
    "reciprocal": lambda x: aten.reciprocal.default(x),
    "pow": lambda x: aten.pow.Tensor_Scalar(x, 2),
    "clamp": lambda x: aten.clamp.default(x, 0),
    "clamp_min": lambda x: aten.clamp_min.default(x, 0),
    "bitwise_not": lambda x: aten.bitwise_not.default(aten._to_copy.default(x, dtype=torch.int64)),
    "clone": lambda x: aten.clone.default(x),
    "to_copy": lambda x: aten._to_copy.default(x, dtype=torch.float64),
    "empty_like": lambda x: aten.empty_like.default(x),
    "zeros_like": lambda x: aten.zeros_like.default(x),
    "add_self": lambda x: aten.add.Tensor(x, x),
    "div_Scalar": lambda x: aten.div.Scalar(x, 2.0),
    "rsub": lambda x: aten.rsub.Scalar(x, 1),
    "add_contiguous": lambda x: aten.add.Tensor(x, m(*x.shape)),
    "sub_contiguous_first": lambda x: aten.sub.Tensor(m(*x.shape), x),
    "eq_scalar": lambda x: aten.eq.Scalar(x, 0),
    "lt_self": lambda x: aten.lt.Tensor(x, x),
    "where": lambda x: aten.where.self(aten.gt.Scalar(x, 0), x, x),
    "prims_sin": lambda x: prims.sin.default(x),
    "prims_clone": lambda x: prims.clone.default(x, memory_format=None),
    "contiguous": lambda x: x.contiguous(),
    "clone_contiguous_format": lambda x: aten.clone.default(x, memory_format=torch.contiguous_format),
    "clone_preserve_format": lambda x: aten.clone.default(x, memory_format=torch.preserve_format),
    "to_copy_contiguous_format": lambda x: aten._to_copy.default(x, memory_format=torch.contiguous_format),
    "tril": lambda x: aten.tril.default(x) if x.dim() >= 2 else x,
}
for lname, make in LAYOUTS.items():
    case("layout:" + lname, lambda make=make: (None, make()))
    for oname, f in FOLLOWERS.items():
        case(oname + ":" + lname, lambda make=make, f=f: (None, f(make())))

# ---- ops whose output is contiguous whatever the input's layout -------------
def rev(*shape):
    # the same shape, strides reversed
    return m(*reversed(shape)).permute(*reversed(range(len(shape))))


def gap(*shape):
    # a stepped last axis: not dense
    return m(*shape[:-1], shape[-1] * 2)[..., ::2]


def cl(*shape):
    # channels-last where the rank allows it, reversed otherwise
    if len(shape) != 4:
        return rev(*shape)
    n, c, h, w = shape
    return m(n, h, w, c).permute(0, 3, 1, 2)


def off(*shape):
    return m(shape[0] + 2, *shape[1:])[2:]


idx = lambda *s: torch.zeros(s, dtype=torch.long, device="meta")
CONTIGUOUS = {
    "mm": lambda L: aten.mm.default(L(3, 4), L(4, 5)),
    "bmm": lambda L: aten.bmm.default(L(2, 3, 4), L(2, 4, 5)),
    "addmm": lambda L: aten.addmm.default(L(5), L(3, 4), L(4, 5)),
    "addmm_bias_2d": lambda L: aten.addmm.default(L(3, 5), L(3, 4), L(4, 5)),
    "matmul": lambda L: aten.matmul.default(L(2, 3, 4), L(4, 5)),
    "matmul_4d": lambda L: aten.matmul.default(L(2, 2, 3, 4), L(2, 2, 4, 3)),
    "sum": lambda L: aten.sum.default(L(3, 4)),
    "sum_dim": lambda L: aten.sum.dim_IntList(L(3, 4, 5), [1]),
    "sum_dim_keep": lambda L: aten.sum.dim_IntList(L(2, 3, 4, 5), [1], True),
    "mean": lambda L: aten.mean.default(L(3, 4)),
    "mean_dim_keep": lambda L: aten.mean.dim(L(2, 3, 4, 5), [-1], True),
    "amax": lambda L: aten.amax.default(L(2, 3, 4, 5), [1], True),
    "argmax": lambda L: aten.argmax.default(L(3, 4, 5), 1, True),
    "max_dim": lambda L: aten.max.dim(L(2, 3, 4, 5), 1, True),
    "min_dim": lambda L: aten.min.dim(L(3, 4, 5), 2),
    "topk": lambda L: aten.topk.default(L(3, 4, 5), 2, 1),
    "cumsum": lambda L: aten.cumsum.default(L(2, 3, 4, 5), 1),
    "any": lambda L: aten.any.default(L(3, 4)),
    "norm": lambda L: aten.norm.ScalarOpt_dim(L(3, 4, 5), 2, [1], True),
    "embedding": lambda L: aten.embedding.default(L(10, 4), idx(5, 3).t()),
    "gather": lambda L: aten.gather.default(L(3, 4, 5), 1, idx(3, 2, 5)),
    "native_layer_norm": lambda L: aten.native_layer_norm.default(L(3, 4, 5), [5], L(5), L(5), 1e-5),
    "native_layer_norm_4d": lambda L: aten.native_layer_norm.default(L(2, 3, 4, 5), [5], None, None, 1e-5),
    "convolution": lambda L: aten.convolution.default(
        L(2, 3, 6, 6), L(4, 3, 3, 3), None, [1, 1], [0, 0], [1, 1], False, [0, 0], 1),
    "convolution_1d": lambda L: aten.convolution.default(
        L(2, 3, 6), L(4, 3, 3), L(4), [1], [0], [1], False, [0], 1),
    "repeat": lambda L: aten.repeat.default(L(3, 4), [2, 1]),
    "new_zeros": lambda L: aten.new_zeros.default(L(3, 4), [2, 5]),
    "new_empty": lambda L: aten.new_empty.default(L(3, 4), [2, 5]),
    "new_ones": lambda L: aten.new_ones.default(L(2, 3, 4, 5), [2, 5]),
    "tril": lambda L: aten.tril.default(L(3, 4)),
    "triu": lambda L: aten.triu.default(L(2, 3, 4, 5)),
}
BUILDERS = {"contig": m, "rev": rev, "gap": gap, "cl": cl, "off": off}
# `cat` follows the inputs' common memory format, so it is judged separately
CAT = {
    "cat_0": lambda L: aten.cat.default([L(3, 4), L(3, 4)], 0),
    "cat_mixed": lambda L: aten.cat.default([L(3, 4), m(3, 4)], 1),
    "cat_4d": lambda L: aten.cat.default([L(2, 3, 4, 5), L(2, 3, 4, 5)], 1),
    "cat_4d_dim0": lambda L: aten.cat.default([L(2, 3, 4, 5), L(1, 3, 4, 5)], 0),
    "cat_4d_mixed": lambda L: aten.cat.default([L(2, 3, 4, 5), m(2, 3, 4, 5)], 1),
    "cat_4d_skip": lambda L: aten.cat.default([m(0), L(2, 3, 4, 5), L(2, 3, 4, 5)], 3),
    "cat_4d_c1": lambda L: aten.cat.default([L(2, 1, 4, 5), L(2, 1, 4, 5)], 2),
    "cat_n111": lambda L: aten.cat.default([L(2, 1, 1, 1), L(2, 1, 1, 1)], 0),
}
for oname, f in CAT.items():
    for bname, L in BUILDERS.items():
        case("cat:" + oname + ":" + bname, lambda f=f, L=L: (None, f(L)))
for oname, f in CONTIGUOUS.items():
    for bname, L in BUILDERS.items():
        case("free:" + oname + ":" + bname, lambda f=f, L=L: (None, f(L)))

# ---- attention: `empty_like(query)` and a transposed logsumexp --------------
SDPA_LAYOUTS = {"contig": m, "rev": rev, "gap": gap, "cl": cl, "off": off,
                "bhtk": lambda *s: m(s[0], s[2], s[1], s[3]).transpose(1, 2)}
for bname, L in SDPA_LAYOUTS.items():
    case("sdpa:" + bname, lambda L=L: (None, aten._scaled_dot_product_flash_attention_for_cpu.default(
        L(2, 3, 5, 4), L(2, 3, 5, 4), L(2, 3, 5, 4))))
case("sdpa:functional", lambda: (None, torch.nn.functional.scaled_dot_product_attention(
    m(2, 3, 5, 4), m(2, 3, 5, 4), m(2, 3, 5, 4))))

# ---- the gated arms, on contiguous inputs only -------------------------------
# `meta_stride_rule`'s `Unverified` arms answer only when every meta input is
# contiguous; that their answer is upstream's *there* is measured here, op by op.
GATED = {
    "sort": lambda: aten.sort.default(m(3, 4, 5), 1),
    "sort_last_desc": lambda: aten.sort.default(m(2, 3, 4, 5), -1, True),
    "index_none_first": lambda: aten.index.Tensor(m(3, 4, 5), [None, idx(2)]),
    "index_first": lambda: aten.index.Tensor(m(3, 4, 5), [idx(2, 1)]),
    "index_two": lambda: aten.index.Tensor(m(3, 4, 5), [idx(2), None, idx(2)]),
    "weight_norm_0": lambda: aten._weight_norm_interface.default(m(4, 3), m(4, 1), 0),
    "weight_norm_last": lambda: aten._weight_norm_interface.default(m(4, 3, 2), m(1, 1, 2), 2),
}
for oname, f in GATED.items():
    case("gated:" + oname, lambda f=f: (None, f()))

# broadcasting: the elementwise rule runs over the operands *after* expansion
case("add_broadcast_row", lambda: (None, m(3, 4).t() + m(3)))
case("add_broadcast_col", lambda: (None, m(4, 1) + m(3, 4).t()))
case("mul_broadcast_rank", lambda: (None, m(5) * m(3, 4, 5).permute(1, 0, 2)))
case("where_broadcast", lambda: (None, torch.where(m(4, 1) > 0, m(3, 4).t(), m(3))))

# ---- channels-last is a fact about strides, not about a constructor ---------
case("dense_channels_last", lambda: (None, torch.zeros(2, 4, 5, 3).permute(0, 3, 1, 2)))
case("dense_nchw", lambda: (None, torch.zeros(2, 3, 4, 5)))
case("dense_nchw_c1", lambda: (None, torch.zeros(2, 1, 4, 5)))
case("dense_empty_channels_last", lambda: (None, torch.zeros(0, 4, 5, 3).permute(0, 3, 1, 2)))

# ---- set_ from a meta storage now keeps the layout it is given --------------
def set_case(size, stride, offset, storage_numel=24):
    def build():
        s = m(storage_numel).untyped_storage()
        x = m(0)
        return None, x.set_(s, offset, size, stride)
    return build
case("set_contiguous", set_case((3, 4), (4, 1), 0))
case("set_transposed", set_case((4, 3), (1, 4), 0, storage_numel=12))
case("set_offset", set_case((2, 3), (3, 1), 5))
case("set_strided_offset", set_case((3, 2), (1, 3), 4))


def set_grows():
    # past the storage's end: upstream grows the one storage every handle and
    # every view addresses, so all of them must report the new size
    base = m(4)
    view = base[1:]
    s = base.untyped_storage()
    x = m(0).set_(s, 2, (3,), (1,))
    return None, x, [s.nbytes(), base.untyped_storage().nbytes(), view.untyped_storage().nbytes()]


def set_grows_case():
    try:
        _, x, sizes = set_grows()
        out["cases"]["set_grows"] = describe(x, None)
        out["cases"]["set_grows_seen_by"] = {"sizes": sizes}
    except Exception as exc:
        out["cases"]["set_grows"] = {"raised": type(exc).__name__, "message": str(exc)}


set_grows_case()

print(json.dumps(out))
"""


def _layouts(_cache={}):
    if "v" not in _cache:
        _cache["v"] = (_run(_LAYOUT_PROBE, shim=True), _run(_LAYOUT_PROBE, shim=False))
    return _cache["v"]


def _diff(names):
    shim, up = _layouts()
    bad = []
    for name in names:
        s, u = shim["cases"][name], up["cases"][name]
        if isinstance(u, dict) and "raised" in u:
            # A refusal is compared by type and message: a shim that refuses
            # for a different reason has not reproduced the refusal.
            if not (isinstance(s, dict) and s.get("raised") == u["raised"]
                    and s.get("message") == u["message"]):
                bad.append((name, s, u))
        elif s != u:
            bad.append((name, s, u))
    return bad


def _names(prefix=None, exclude=()):
    shim, up = _layouts()
    out = []
    for name in up["cases"]:
        if prefix is not None and not name.startswith(prefix):
            continue
        if any(name.startswith(e) for e in exclude):
            continue
        out.append(name)
    return out


def _report(bad):
    return "\n".join(f"  {n}\n    shim     {s}\n    upstream {u}" for n, s, u in bad[:8])


def test_as_strided_on_meta_answers_upstreams_layout_and_refusals():
    """EXPORT6 §6's named op: size, stride and offset are *stored*, not derived.

    The non-contiguous cases are what give this test teeth: an implementation
    that ignored the requested stride and computed a contiguous one from the
    shape would report `(3, 1)` for `as_strided((4, 3), (1, 4), 2)` and fail
    here.  So would one that dropped the offset, or that took the default
    offset as `0` rather than the input's own.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    names = _names("as_strided") + ["aten_as_strided_kw"]
    bad = _diff(names)
    assert not bad, f"{len(bad)} of {len(names)} as_strided cases differ\n" + _report(bad)
    shim, _ = _layouts()
    assert shim["cases"]["as_strided"]["stride"] == [1, 4], shim["cases"]["as_strided"]
    print(f"AS_STRIDED: {len(names)} of {len(names)} cases agree with upstream")


def test_view_kernels_on_meta_carry_the_real_stride_offset_and_storage():
    """The lie that was already live, closed: every meta view reports its layout.

    `t`, `transpose`, `permute`, `slice`, `select`, `expand`, `squeeze`,
    `unsqueeze`, `split` and the pass-through views all had meta arms before
    this round, and all of them answered a fresh contiguous tensor with a fresh
    storage.  Both halves are compared: the stride/offset, and that the result
    shares its input's storage -- `meta_utils.py`'s storage memo keys on that.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    names = [n for n in _names(exclude=("as_strided", "aten_as_strided", "view_", "reshape_",
                                        "layout:", "set_", "dense_", "add_broadcast",
                                        "mul_broadcast", "where_broadcast"))
             if ":" not in n]
    assert len(names) >= 30, names
    bad = _diff(names)
    assert not bad, f"{len(bad)} of {len(names)} view cases differ\n" + _report(bad)
    shim, _ = _layouts()
    assert shim["cases"]["t"]["stride"] == [1, 4], shim["cases"]["t"]
    assert shim["cases"]["slice"]["offset"] == 5, shim["cases"]["slice"]
    assert shim["cases"]["t"]["shares_storage"] is True, shim["cases"]["t"]
    print(f"VIEWS: {len(names)} of {len(names)} meta view cases agree with upstream")


def test_view_on_meta_is_legal_exactly_where_upstreams_layout_rule_says():
    """`view` of a non-contiguous tensor is sometimes a view and sometimes an error.

    Before this round a meta `view` could never refuse on layout, because every
    meta tensor was contiguous.  Now `t().view(12)` must raise upstream's own
    message, `t().view(4, 3, 1)` must succeed with upstream's stride, and
    `reshape` must pick view or copy by the same rule -- a copy has a fresh
    storage, a view does not.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    names = _names("view_") + _names("reshape_")
    bad = _diff(names)
    assert not bad, f"{len(bad)} of {len(names)} view/reshape cases differ\n" + _report(bad)
    shim, _ = _layouts()
    assert shim["cases"]["view_of_t_across_subspaces"]["raised"] == "RuntimeError"
    assert shim["cases"]["reshape_as_copy"]["shares_storage"] is False
    assert shim["cases"]["reshape_as_view"]["shares_storage"] is True
    print(f"VIEW RULE: {len(names)} of {len(names)} cases agree with upstream")


def test_layout_following_meta_kernels_answer_upstreams_output_stride():
    """Pointwise and `preserve_format` kernels: the output layout follows the input's.

    Measured on upstream first, across the twelve layouts below: every one of
    these ops answers `torch._prims_common.compute_elementwise_output_strides`
    of its operands, including on the layouts (`perm_step`, `odd`, `overlap`)
    where that differs from the simpler "keep the stride if dense" rule.
    `contiguous` and `tril` are here as the opposite answer, always contiguous.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    names = [n for n in _names(exclude=("free:", "cat:", "sdpa:", "gated:")) if ":" in n] + \
        _names("add_broadcast") + \
        _names("mul_broadcast") + _names("where_broadcast")
    assert len(names) >= 300, len(names)
    bad = _diff(names)
    assert not bad, f"{len(bad)} of {len(names)} layout-following cases differ\n" + _report(bad)
    print(f"FOLLOWERS: {len(names)} of {len(names)} (op x layout) cases agree with upstream")


def test_always_contiguous_meta_kernels_are_contiguous_on_every_input_layout():
    """`meta_stride_rule`'s `AlwaysContiguous` list, held to upstream.

    These arms build a fresh contiguous result and are allowed past the gate
    with a non-contiguous input *because* upstream answers contiguously there
    too.  That is a claim per op, and it is checked per op across five input
    layouts -- including channels-last, where `convolution` and `cat` are the
    ops one would expect to follow the input.  An op that does not belong on
    the list fails here rather than lying quietly.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    names = _names("free:")
    assert len(names) >= 150, len(names)
    bad = _diff(names)
    assert not bad, f"{len(bad)} of {len(names)} always-contiguous cases differ\n" + _report(bad)
    print(f"ALWAYS_CONTIGUOUS: {len(names)} of {len(names)} (op x layout) cases agree with upstream")


def test_cat_on_meta_follows_its_inputs_common_memory_format():
    """`cat` is neither always-contiguous nor elementwise.

    Measured: two NHWC-permuted inputs give an NHWC-strided result, a mixed
    pair gives a contiguous one, and the ambiguous `N111` / `C == 1` layouts
    fall back to contiguous through upstream's own tie-breaks.  Judged on the
    same five input layouts as the always-contiguous list, where it first
    showed up as the one member that did not belong.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    names = _names("cat:")
    bad = _diff(names)
    assert not bad, f"{len(bad)} of {len(names)} cat cases differ\n" + _report(bad)
    shim, _ = _layouts()
    assert shim["cases"]["cat:cat_4d:cl"]["channels_last"] is True
    print(f"CAT: {len(names)} of {len(names)} cases agree with upstream")


def test_attention_on_meta_lays_out_logsumexp_transposed_as_upstream_does():
    """A wrong stride on the path the gate lets through, found and closed.

    Upstream's CPU flash-attention meta is `empty_like(query)` and
    `empty(B, T, H).transpose(1, 2)`, so `logsumexp` is non-contiguous **for a
    contiguous query** -- `(15, 1, 3)` for `(2, 3, 5)`.  The shim answered
    `(15, 5, 1)`, and the stride gate could not see it because the inputs were
    contiguous.  Judged across six query layouts and through the functional
    door, which also returned a bare `TensorBase` before the pair was promoted.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    names = _names("sdpa:")
    bad = _diff(names)
    assert not bad, f"{len(bad)} of {len(names)} attention cases differ\n" + _report(bad)
    shim, _ = _layouts()
    assert shim["cases"]["sdpa:contig"][1]["stride"] == [15, 1, 3], shim["cases"]["sdpa:contig"]
    print(f"SDPA: {len(names)} of {len(names)} cases agree with upstream")


def test_gated_meta_kernels_agree_with_upstream_on_contiguous_inputs():
    """The premise under `meta_stride_rule`'s `Unverified` class, measured per op.

    Those arms answer only when every meta input is contiguous, on the theory
    that their contiguous answer is right there.  That theory is not a law --
    attention broke it (above) -- so every arm still in the class is compared
    with upstream on contiguous inputs.  An arm added to the class later is
    not judged until it is added here; `meta_stride_rule`'s docstring says so.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    names = _names("gated:")
    bad = _diff(names)
    assert not bad, f"{len(bad)} of {len(names)} gated cases differ\n" + _report(bad)
    print(f"GATED: {len(names)} of {len(names)} cases agree with upstream")


def test_channels_last_contiguity_is_read_off_the_stride():
    """`is_contiguous(memory_format=channels_last)` is a fact about the stride.

    `test_export5.py` asserted it was `False` for every tensor because the build
    "cannot make" a channels-last tensor, and checked that by trying
    `.to(memory_format=channels_last)`.  A plain `permute(0, 3, 1, 2)` of an
    NHWC tensor *is* channels-last, on the dense side as well, and the shim
    answered `False` for it.  With `as_strided` open on meta the same layout is
    constructible there too, so the answer is now computed on both arms.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    names = ["dense_channels_last", "dense_nchw", "dense_nchw_c1", "dense_empty_channels_last",
             "layout:channels_last", "layout:channels_last_c1", "layout:nchw_c1",
             "layout:perm3", "empty_strided_meta_channels_last"]
    bad = _diff(names)
    assert not bad, f"{len(bad)} of {len(names)} channels-last cases differ\n" + _report(bad)
    shim, _ = _layouts()
    assert shim["cases"]["dense_channels_last"]["channels_last"] is True
    assert shim["cases"]["layout:channels_last"]["channels_last"] is True
    print(f"CHANNELS_LAST: {len(names)} of {len(names)} cases agree with upstream")


def test_set_on_a_meta_tensor_adopts_a_non_contiguous_stride_and_an_offset():
    """EXPORT6 §2.5's two named refusals, lifted now that there is somewhere to put them.

    That round refused a non-contiguous stride and a non-zero offset *because*
    `Repr::Meta` had no field for either.  Both fields exist now, so both are
    adopted -- and asserted against upstream, so an implementation that
    accepted them and then answered a contiguous stride would fail.  A layout
    past the storage's end grows the storage, as upstream's does, and every
    handle and view must see the new size.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    names = _names("set_")
    assert "set_grows_seen_by" in names, names
    bad = _diff(names)
    assert not bad, f"{len(bad)} of {len(names)} set_ cases differ\n" + _report(bad)
    shim, _ = _layouts()
    assert shim["cases"]["set_strided_offset"]["stride"] == [1, 3]
    assert shim["cases"]["set_strided_offset"]["offset"] == 4
    # the growth is seen through the handle, the base and a view alike
    assert shim["cases"]["set_grows_seen_by"]["sizes"] == [20, 20, 20]
    print(f"SET_: {len(names)} of {len(names)} cases agree with upstream")


# ---------------------------------------------------------------------------
# Refusals that are the shim's own, asserted by name
# ---------------------------------------------------------------------------

_REFUSAL_PROBE = r"""
import json
import torch

out = {"is_shim": hasattr(torch._C, "_aten_implemented")}


def attempt(f):
    try:
        f()
        return {"ok": True}
    except Exception as exc:
        return {"raised": type(exc).__name__, "message": str(exc)}


m = lambda *s: torch.empty(*s, device="meta")
# an op with a meta kernel whose output layout this shim has not been shown to
# follow: it must refuse a non-contiguous input rather than answer contiguously
out["sort_t"] = attempt(lambda: torch.ops.aten.sort.default(m(3, 4).t(), 0))
out["sort_contiguous"] = attempt(lambda: torch.ops.aten.sort.default(m(3, 4), 0))
out["set_negative_stride"] = attempt(lambda: m(0).set_(m(12).untyped_storage(), 0, (3, 4), (-4, 1)))
print(json.dumps(out))
"""


def test_a_stride_unaware_meta_kernel_refuses_a_non_contiguous_input():
    """The gate that makes an unverified output stride unrepresentable.

    Every meta arm that has not been shown to answer upstream's output layout
    refuses when any meta input is not in the canonical contiguous layout --
    by the op's name, naming the stride.  Without the gate `sort` would
    answer a fresh contiguous tensor for a transposed input, which is the lie
    this round exists to remove.  The contiguous half is asserted too, so the
    gate cannot pass by refusing everything.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    r = _run(_REFUSAL_PROBE, shim=True)
    e = r["sort_t"]
    assert e.get("raised") == "NotImplementedError", e
    assert "aten.sort.default" in e["message"] and "stride" in e["message"], e
    assert r["sort_contiguous"] == {"ok": True}, r["sort_contiguous"]


def test_set_on_meta_refuses_a_negative_stride_by_name():
    """A negative stride is the one `set_` layout still refused on meta.

    Upstream accepts one and reads back a *different* stride (`(-1, 3)` comes
    back as `(6, 3)`, measured), which is not a layout this shim can claim to
    reproduce; the dense `set_` refuses it too.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    r = _run(_REFUSAL_PROBE, shim=True)
    e = r["set_negative_stride"]
    assert e.get("raised") == "NotImplementedError", e
    assert "negative stride" in e["message"], e


# ---------------------------------------------------------------------------
# The bar: exported, replayed, agreed -- on a module that needs meta strides
# ---------------------------------------------------------------------------

_EXPORT_PROBE = r"""
import json
import torch

out = {"is_shim": hasattr(torch._C, "_aten_implemented")}


class Strided(torch.nn.Module):
    # every intermediate is a strided view or an op over strided operands, so
    # the fake tensors export records are the meta layouts this file is about
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.lin = torch.nn.Linear(8, 8)
        self.w = torch.nn.Parameter(torch.randn(3, 7))

    def forward(self, x):
        a = self.lin(x).t()[1:, ::2]
        b = x.t()[1:, 1:4]
        c = torch.relu(a + b)
        return torch.matmul(c, self.w.t()[:, :4].t()), c.transpose(0, 1) * 2


torch.manual_seed(0)
mod = Strided().eval()
x = torch.randn(5, 8)
with torch.no_grad():
    eager = mod(x)
out["eager"] = [t.flatten().tolist() for t in eager]
out["shapes"] = [list(t.shape) for t in eager]
if not out["is_shim"]:
    # upstream's own float32-vs-float64 distance: the tolerance's source
    with torch.no_grad():
        exact = Strided().double().eval()(x.double())
    out["oracle"] = [(e.double() - f).abs().flatten().tolist() for e, f in zip(eager, exact)]
stage = "export"
try:
    ep = torch.export.export(mod, (x,))
    def lay(v):
        return [list(v.shape), list(v.stride()), v.storage_offset()]

    out["layouts"] = []
    for n in ep.graph.nodes:
        v = n.meta.get("val")
        if n.op != "call_function" or not isinstance(v, torch.Tensor):
            continue
        first = n.args[0].meta.get("val") if n.args and hasattr(n.args[0], "meta") else None
        src = lay(first) if isinstance(first, torch.Tensor) else None
        out["layouts"].append([str(n.target), src, lay(v)])
    stage = "replay"
    with torch.no_grad():
        replayed = ep.module()(x)
    out["replayed"] = [t.flatten().tolist() for t in replayed]
    out["stage"] = "done"
except Exception:
    import traceback
    out["stage"] = stage
    out["error"] = traceback.format_exc()[-3000:]
print(json.dumps(out))
"""


def test_a_module_whose_trace_needs_real_meta_strides_exports_replays_and_agrees():
    """The replay-and-agree bar, and the layouts the exported graph records.

    Three verdicts, separately: the export returns, the exported module
    replays, and its outputs agree element-wise with the module's own eager
    outputs -- and the shim's eager outputs agree with upstream's.  The
    tolerance is derived, `docs/numerics/AGREE.md` §2's method on these
    outputs: the p90 of upstream's own float32-vs-float64 relative error,
    floored at 8 ulp.

    Agreement alone could not see a wrong meta stride -- replay runs dense
    kernels -- so the fake layouts `torch.export` recorded on every call node
    are compared with upstream's too.  The two graphs are not node-for-node
    identical (this shim decomposes `linear`; upstream keeps it, and spells a
    no-op slice `alias`), so nodes are matched by `(op, input layout, output
    shape)` and every match must carry upstream's stride and offset.  A meta
    kernel answering a contiguous stride fails here either way: its node
    disagrees, or the nodes after it stop matching and the count falls.

    The module stays 2-D on purpose: a `view` of a 3-D activation stops at
    `FakeTensorDeviceMismatchError`, because this shim does not route a tensor
    subclass's own `__torch_dispatch__` (docs/graph/STRIDE.md §6).
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim = _run(_EXPORT_PROBE, shim=True)
    up = _run(_EXPORT_PROBE, shim=False)
    assert up["stage"] == "done", up.get("error")
    assert shim["stage"] == "done", shim.get("error")
    assert shim["shapes"] == up["shapes"]

    worst_rel = 0.0
    for i, (eager_s, replay_s, eager_u, oracle) in enumerate(
            zip(shim["eager"], shim["replayed"], up["eager"], up["oracle"])):
        scale = max(abs(v) for v in eager_u)
        rel = sorted(e / scale for e in oracle)
        tol = max(rel[int(0.9 * (len(rel) - 1))], 8 * _EPS32) * scale
        replay = max(abs(a - b) for a, b in zip(replay_s, eager_s))
        cross = max(abs(a - b) for a, b in zip(eager_s, eager_u))
        assert replay <= tol, f"output {i}: replay disagrees with eager: {replay} > {tol}"
        assert cross <= tol, f"output {i}: shim eager disagrees with upstream: {cross} > {tol}"
        worst_rel = max(worst_rel, replay / scale, cross / scale)

    def key(target, src, result):
        return (target, json.dumps(src), tuple(result[0]))

    upstream = {}
    for target, src, result in up["layouts"]:
        upstream.setdefault(key(target, src, result), []).append(result)
    matched, wrong, ops = 0, [], set()
    for target, src, result in shim["layouts"]:
        want = upstream.get(key(target, src, result))
        if want is None:
            continue
        matched += 1
        ops.add(target)
        if result not in want:
            wrong.append((target, src, result, want))
    assert not wrong, f"{len(wrong)} exported layouts differ from upstream: {wrong}"
    needed = {"aten.slice.Tensor", "aten.t.default", "aten.add.Tensor",
              "aten.relu.default", "aten.transpose.int", "aten.matmul.default"}
    assert matched >= 10 and needed <= ops, (
        f"only {matched} exported nodes line up with upstream's, covering {sorted(ops)}; "
        f"the layouts feeding them must have diverged")
    print(f"EXPORT: exported+replayed+agreed, worst relative {worst_rel:.3e}; "
          f"{matched} exported layouts identical to upstream's")


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failed else 0)
