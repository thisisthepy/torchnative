"""Symbolic shapes (SymInt / SymFloat / SymBool) for the shim's fake tensors.

Issue #262, INTENT §5, SPEC S1.8. Owner, 2026-11-13: "시퀀스 길이를 기호로 다루는거
좋은거 같아. 진행해. 동적 형태 구현해."

The shim's kernels and its `Repr::Meta` tensors work on concrete `usize` sizes. Upstream's
fake tensors carry `SymInt` sizes in C++. This module bridges the two without touching
the Rust side:

1. **Every symbolic value has a hint.** A backed `SymInt` (one Dynamo allocated for an
   input dimension) knows the concrete value it was traced with (`node.hint`). The shim
   kernels run on hints, so the meta computation is the one the shim already does.
2. **The symbolic sizes ride beside the hint tensor.** A `FakeTensor` whose sizes are
   symbolic carries `_tn_sym = (sizes, strides, storage_offset)` in its `__dict__`;
   `FakeTensor.size()`, `.shape`, `.stride()`, `.numel()` and `.storage_offset()` answer
   from it. Only `FakeTensor` is patched, so eager tensors pay nothing.
3. **Overload resolution sees an int.** `_Overloads.resolve` is wrapped: each symbolic
   argument is replaced by a `_SymHint`, an `int` subclass holding the hint and the
   symbol. Resolution and the Rust extraction see an `int`; the fake-tensor dispatch
   below sees the symbol again.
4. **Each fake op computes its output's symbolic sizes from a rule.** The wrapped
   `FakeTensorMode._dispatch_impl` runs the shim's meta kernel on hints, then a rule for
   the op computes the symbolic sizes from the symbolic inputs. Comparisons a rule makes
   on symbols (`if size == 1`) go through the ShapeEnv and become guards, exactly as in
   upstream's meta functions. **A rule's result is checked against the kernel's hint
   result**, so a wrong rule refuses by name instead of tracing a wrong shape.
5. **An op with no rule refuses by name** (`SymbolicShapeUnsupported`). The engine
   catches it and translates that frame again with static shapes, and counts it in
   `stats["dynamic_refused"]`; it never guesses a symbolic shape.

What this cannot do: unbacked symbols (data-dependent sizes such as `nonzero`) have no
hint, so they refuse by name. Ops outside the rule table make the frame static.
"""

import functools
import operator
import sys

__all__ = [
    "SymbolicShapeUnsupported",
    "install",
    "has_rule",
]


class SymbolicShapeUnsupported(RuntimeError):
    """A fake op met a symbolic size this module has no rule for. Names the op."""


_installed = False
_torch = None
_SymTypes = ()
_RULES = {}


# ---------------------------------------------------------------------------
# Hints
# ---------------------------------------------------------------------------


class _SymHint(int):
    """An `int` that is the hint of a symbolic value; `.sym` is the symbol itself."""


    def __new__(cls, hint, sym):
        o = int.__new__(cls, hint)
        o.sym = sym
        return o

    def __repr__(self):
        return f"_SymHint({int(self)}, {self.sym!r})"


def _is_sym(v):
    return isinstance(v, _SymTypes)


def _hint_of(v):
    """Concrete value a symbolic value was traced with. Refuses an unbacked symbol."""
    node = v.node
    h = node.hint
    if h is None:
        raise SymbolicShapeUnsupported(
            f"torchnative symbolic shapes: {v!r} has no hint (an unbacked, data-dependent "
            f"size); the shim's kernels need a concrete size to run"
        )
    if isinstance(v, _torch.SymBool):
        return bool(h)
    if isinstance(v, _torch.SymFloat):
        return float(h)
    return int(h)


def _to_hint(v):
    """Symbol → `_SymHint` (ints) or plain hint (floats, bools); recurse into lists."""
    if isinstance(v, _SymHint):
        return v
    if _is_sym(v):
        h = _hint_of(v)
        if isinstance(v, _torch.SymInt):
            return _SymHint(h, v)
        return h
    if isinstance(v, (list, tuple)) and v and any(_has_sym_scalar(x) for x in v):
        return type(v)(_to_hint(x) for x in v) if type(v) in (list, tuple) else [_to_hint(x) for x in v]
    return v


def _has_sym_scalar(v):
    if _is_sym(v):
        return True
    if isinstance(v, (list, tuple)):
        return any(_has_sym_scalar(x) for x in v)
    return False


def _plain(v):
    """`_SymHint` → its symbol; plain values unchanged (recursing into lists)."""
    if isinstance(v, _SymHint):
        return v.sym
    if isinstance(v, (list, tuple)):
        return type(v)(_plain(x) for x in v) if type(v) in (list, tuple) else [_plain(x) for x in v]
    return v


def _strip(v):
    """`_SymHint` → plain `int`, so nothing downstream keeps the symbol by accident."""
    if isinstance(v, _SymHint):
        return int(v)
    if _is_sym(v):
        # A raw symbol (an OpOverload called directly, not through resolve): its hint.
        # Handing it to the kernel would call `__index__`, which guards and specializes.
        return _hint_of(v)
    if isinstance(v, (list, tuple)):
        if any(isinstance(x, (_SymHint, list, tuple)) or _is_sym(x) for x in v):
            return type(v)(_strip(x) for x in v) if type(v) in (list, tuple) else [_strip(x) for x in v]
    return v


def _concrete(v):
    """Hint of an int-like (SymInt, _SymHint or int)."""
    if isinstance(v, _SymHint):
        return int(v)
    if _is_sym(v):
        return _hint_of(v)
    return v


# ---------------------------------------------------------------------------
# Symbolic metadata on fake tensors
# ---------------------------------------------------------------------------


def _sym_meta(t):
    """`(sizes, strides, offset)` with symbols where `t` has them, else `None`."""
    d = getattr(t, "__dict__", None)
    if d is None:
        return None
    return d.get("_tn_sym")


def _any_symbolic(sizes):
    return any(_is_sym(s) for s in sizes)


def _set_sym(t, sizes, strides, offset=0):
    """Attach symbolic metadata, or clear it when every entry is concrete."""
    sizes = tuple(sizes)
    strides = tuple(strides)
    if _any_symbolic(sizes) or _any_symbolic(strides) or _is_sym(offset):
        t.__dict__["_tn_sym"] = (sizes, strides, offset)
    else:
        t.__dict__.pop("_tn_sym", None)


def _base(name):
    return getattr(_torch._C.TensorBase, name)


def sizes_of(t):
    m = _sym_meta(t)
    if m is not None:
        return m[0]
    return tuple(_base("size")(t))


def strides_of(t):
    m = _sym_meta(t)
    if m is not None:
        return m[1]
    return tuple(_base("stride")(t))


def offset_of(t):
    m = _sym_meta(t)
    if m is not None:
        return m[2]
    return _base("storage_offset")(t)


def _install_fake_tensor_methods(FakeTensor):
    Size = _torch.Size
    base_size = _base("size")
    base_stride = _base("stride")
    base_numel = _base("numel")
    base_offset = _base("storage_offset")

    def size(self, dim=None):
        m = self.__dict__.get("_tn_sym")
        if m is None:
            return base_size(self) if dim is None else base_size(self, dim)
        if dim is None:
            return Size(m[0])
        return m[0][dim]

    def stride(self, dim=None):
        m = self.__dict__.get("_tn_sym")
        if m is None:
            return base_stride(self) if dim is None else base_stride(self, dim)
        if dim is None:
            return tuple(m[1])
        return m[1][dim]

    def numel(self):
        m = self.__dict__.get("_tn_sym")
        if m is None:
            return base_numel(self)
        return functools.reduce(operator.mul, m[0], 1)

    def storage_offset(self):
        m = self.__dict__.get("_tn_sym")
        if m is None:
            return base_offset(self)
        return m[2]

    FakeTensor.size = size
    FakeTensor.stride = stride
    FakeTensor.numel = numel
    FakeTensor.nelement = numel
    FakeTensor.storage_offset = storage_offset
    FakeTensor.shape = property(lambda self: size(self))

    # Shim methods written in Python that read `self.numel()` / `self.size()` for their
    # own bookkeeping must see the hint tensor, not the symbols.
    for name in _HINT_ONLY_METHODS:
        base = getattr(FakeTensor, name, None)
        if base is None or getattr(base, "_tn_symbolic", False):
            continue
        setattr(FakeTensor, name, _hint_only(base))


_HINT_ONLY_METHODS = ("_is_view",)


def _hint_only(base):
    def call(self, *args, **kwargs):
        d = self.__dict__
        m = d.pop("_tn_sym", None)
        try:
            return base(self, *args, **kwargs)
        finally:
            if m is not None:
                d["_tn_sym"] = m

    call._tn_symbolic = True
    call.__name__ = getattr(base, "__name__", "call")
    return call


def contiguous_strides(sizes):
    strides = []
    acc = 1
    for s in reversed(sizes):
        strides.append(acc)
        acc = acc * _max1(s)
    return tuple(reversed(strides))


def _max1(s):
    # Upstream's contiguous stride uses max(size, 1); a symbol with hint >= 2 is
    # assumed >= 2 by Dynamo's 0/1 specialization, so it is its own max.
    if _is_sym(s):
        return s
    return max(s, 1)


def _like_strides(sizes, hint_strides, hint_sizes):
    """Strides for a fresh (non-view) output: same dense order as the kernel produced."""
    if not _any_symbolic(sizes):
        return tuple(hint_strides)
    n = len(sizes)
    if n == 0:
        return ()
    # Order dimensions from innermost (smallest stride) out, as the kernel laid them.
    order = sorted(range(n), key=lambda i: (hint_strides[i], -i))
    out = [None] * n
    acc = 1
    hacc = 1
    for i in order:
        if hint_strides[i] != hacc and hint_sizes[i] != 1:
            raise SymbolicShapeUnsupported(
                "torchnative symbolic shapes: kernel output is not dense "
                f"(sizes {tuple(hint_sizes)}, strides {tuple(hint_strides)})"
            )
        out[i] = acc
        acc = acc * _max1(sizes[i])
        hacc = hacc * max(hint_sizes[i], 1)
    return tuple(out)


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


def rule(*names):
    def deco(fn):
        for n in names:
            _RULES[n] = fn
        return fn

    return deco


def has_rule(name):
    return name in _RULES


def _norm_dim(d, n):
    d = _concrete(d)
    return d + n if d < 0 else d


def broadcast(*shapes):
    n = max((len(s) for s in shapes), default=0)
    out = []
    for i in range(n):
        dim = 1
        for s in shapes:
            j = i - (n - len(s))
            if j < 0:
                continue
            v = s[j]
            # Comparisons on symbols become guards through the ShapeEnv.
            if _concrete(v) == 1:
                continue
            if _concrete(dim) == 1:
                dim = v
            elif _is_sym(v) and not _is_sym(dim):
                dim = v
        out.append(dim)
    return tuple(out)


def _tensor_sizes(a):
    if isinstance(a, _torch.Tensor):
        return sizes_of(a)
    return ()


def _elementwise(func, args, kwargs, out):
    shapes = []
    for a in list(args) + list(kwargs.values()):
        if isinstance(a, _torch.Tensor):
            shapes.append(sizes_of(a))
    return [broadcast(*shapes)]


for _n in (
    "add", "sub", "mul", "div", "pow", "rsub", "remainder", "fmod", "maximum", "minimum",
    "eq", "ne", "lt", "le", "gt", "ge", "logical_and", "logical_or", "logical_xor",
    "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_not", "logical_not", "where",
    "masked_fill", "neg", "exp", "log", "sqrt", "rsqrt", "abs", "sin", "cos", "tanh",
    "sigmoid", "silu", "gelu", "relu", "erf", "reciprocal", "sign", "clamp", "clamp_min",
    "clamp_max", "clone", "contiguous", "_to_copy", "to", "copy", "copy_", "fill",
    "fill_", "zero_", "add_", "mul_", "div_", "sub_", "masked_fill_", "neg_", "exp_",
    "_softmax", "softmax", "_log_softmax", "log_softmax", "lift_fresh_copy", "lift_fresh",
    "alias", "detach", "_safe_softmax", "pow_", "tril", "triu", "native_dropout_backward",
    "type_as", "square", "mish", "hardtanh", "leaky_relu", "elu", "softplus", "isnan",
    "isinf", "logical_not_", "float_power", "atan2", "addcmul", "addcdiv", "lerp",
    "nan_to_num", "dropout", "_conj", "resolve_conj", "resolve_neg", "positive",
    "masked_scatter", "sigmoid_", "tanh_", "relu_", "silu_", "erf_",
):
    _RULES[_n] = _elementwise


@rule("native_dropout")
def _r_native_dropout(func, args, kwargs, out):
    s = sizes_of(args[0])
    return [s, s]


@rule("ones_like", "zeros_like", "empty_like", "full_like", "rand_like", "randn_like")
def _r_like(func, args, kwargs, out):
    return [sizes_of(args[0])]


def _size_arg(v):
    if isinstance(v, (list, tuple)):
        return tuple(_plain(x) for x in v)
    return (_plain(v),)


@rule("ones", "zeros", "empty", "full", "rand", "randn", "empty_strided", "empty_permuted")
def _r_factory(func, args, kwargs, out):
    size = args[0] if args else kwargs.get("size")
    return [_size_arg(size)]


@rule("new_ones", "new_zeros", "new_empty", "new_full", "new_empty_strided")
def _r_new(func, args, kwargs, out):
    size = args[1] if len(args) > 1 else kwargs.get("size")
    return [_size_arg(size)]


@rule("arange")
def _r_arange(func, args, kwargs, out):
    vals = [_plain(a) for a in args if not isinstance(a, _torch.Tensor)]
    if "end" in kwargs:
        vals = [_plain(kwargs.get("start", 0)), _plain(kwargs["end"])]
        if "step" in kwargs:
            vals.append(_plain(kwargs["step"]))
    if len(vals) == 1:
        start, end, step = 0, vals[0], 1
    elif len(vals) == 2:
        start, end, step = vals[0], vals[1], 1
    else:
        start, end, step = vals[0], vals[1], vals[2]
    if _concrete(step) != 1 or isinstance(_concrete(end), float) or isinstance(_concrete(start), float):
        if _is_sym(end) or _is_sym(start):
            raise SymbolicShapeUnsupported(
                "torchnative symbolic shapes: aten.arange with a symbolic bound needs step 1 "
                "and integer bounds"
            )
        return None  # concrete bounds: nothing symbolic to attach
    return [(end - start,)]


@rule("mm")
def _r_mm(func, args, kwargs, out):
    return [(sizes_of(args[0])[0], sizes_of(args[1])[1])]


@rule("addmm")
def _r_addmm(func, args, kwargs, out):
    return [(sizes_of(args[1])[0], sizes_of(args[2])[1])]


@rule("bmm", "baddbmm")
def _r_bmm(func, args, kwargs, out):
    i = 1 if str(func).startswith("aten.baddbmm") else 0
    a, b = sizes_of(args[i]), sizes_of(args[i + 1])
    return [(a[0], a[1], b[2])]


@rule("matmul")
def _r_matmul(func, args, kwargs, out):
    a, b = sizes_of(args[0]), sizes_of(args[1])
    if len(a) == 1 and len(b) == 1:
        return [()]
    if len(b) == 1:
        return [tuple(a[:-1])]
    if len(a) == 1:
        return [tuple(b[:-2]) + (b[-1],)]
    batch = broadcast(tuple(a[:-2]), tuple(b[:-2]))
    return [batch + (a[-2], b[-1])]


@rule("linear")
def _r_linear(func, args, kwargs, out):
    a, w = sizes_of(args[0]), sizes_of(args[1])
    return [tuple(a[:-1]) + ((w[0],) if len(w) == 2 else ())]


@rule("embedding")
def _r_embedding(func, args, kwargs, out):
    w, idx = sizes_of(args[0]), sizes_of(args[1])
    return [tuple(idx) + (w[-1],)]


@rule("native_layer_norm")
def _r_native_layer_norm(func, args, kwargs, out):
    s = sizes_of(args[0])
    nshape = args[1] if len(args) > 1 else kwargs["normalized_shape"]
    k = len(s) - len(nshape)
    stat = tuple(s[:k]) + (1,) * len(nshape)
    return [s, stat, stat]


@rule("layer_norm", "rms_norm", "group_norm")
def _r_norm(func, args, kwargs, out):
    return [sizes_of(args[0])]


@rule("_fused_rms_norm")
def _r_fused_rms(func, args, kwargs, out):
    s = sizes_of(args[0])
    nshape = args[1]
    k = len(s) - len(nshape)
    return [s, tuple(s[:k]) + (1,) * len(nshape)]


@rule("scaled_dot_product_attention")
def _r_sdpa(func, args, kwargs, out):
    q, v = sizes_of(args[0]), sizes_of(args[2])
    return [tuple(q[:-1]) + (v[-1],)]


@rule("_scaled_dot_product_flash_attention_for_cpu")
def _r_sdpa_flash_cpu(func, args, kwargs, out):
    # (output, logsumexp): logsumexp is (B, H, L), batched like the query.
    q, v = sizes_of(args[0]), sizes_of(args[2])
    return [tuple(q[:-1]) + (v[-1],), tuple(q[:-1])]


@rule("_scaled_dot_product_attention_math")
def _r_sdpa_math(func, args, kwargs, out):
    # (output, attention weights (..., L, S)).
    q, k, v = sizes_of(args[0]), sizes_of(args[1]), sizes_of(args[2])
    return [tuple(q[:-1]) + (v[-1],), tuple(q[:-1]) + (k[-2],)]


def _reduce_shape(s, dims, keepdim):
    n = len(s)
    if dims is None or (isinstance(dims, (list, tuple)) and len(dims) == 0):
        dims = list(range(n))
    if not isinstance(dims, (list, tuple)):
        dims = [dims]
    dims = {_norm_dim(d, n) for d in dims}
    out = []
    for i, v in enumerate(s):
        if i in dims:
            if keepdim:
                out.append(1)
        else:
            out.append(v)
    return tuple(out)


@rule("sum", "mean", "amax", "amin", "prod", "var", "std", "norm", "linalg_vector_norm",
      "any", "all", "logsumexp")
def _r_reduce(func, args, kwargs, out):
    s = sizes_of(args[0])
    name = str(func)
    if name.endswith(".default") and len(args) == 1 and not kwargs.get("dim"):
        return [()]
    dims = args[1] if len(args) > 1 else kwargs.get("dim")
    if name.startswith(("aten.norm", "aten.linalg_vector_norm")):
        dims = args[2] if len(args) > 2 else kwargs.get("dim")
        keep = args[3] if len(args) > 3 else kwargs.get("keepdim", False)
    elif name.startswith(("aten.var", "aten.std")):
        # var.dim(self, dim, unbiased, keepdim) / var.correction(self, dim, *, correction, keepdim)
        keep = kwargs.get("keepdim", args[3] if len(args) > 3 else False)
    else:
        keep = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
    return [_reduce_shape(s, dims, bool(keep))]


@rule("max", "min", "argmax", "argmin", "cumsum", "cumprod", "sort", "topk")
def _r_dim_reduce(func, args, kwargs, out):
    s = sizes_of(args[0])
    name = str(func)
    if name in ("aten.max.default", "aten.min.default"):
        return [()]
    if name.startswith(("aten.cumsum", "aten.cumprod")):
        return [s]
    if name.startswith("aten.sort"):
        return [s, s]
    if name.startswith("aten.topk"):
        k = _plain(args[1])
        dim = _norm_dim(args[2] if len(args) > 2 else kwargs.get("dim", -1), len(s))
        r = tuple(k if i == dim else v for i, v in enumerate(s))
        return [r, r]
    if name.startswith(("aten.max.other", "aten.min.other")):
        return [broadcast(s, sizes_of(args[1]))]
    dim = args[1] if len(args) > 1 else kwargs.get("dim")
    keep = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
    r = _reduce_shape(s, dim, bool(keep))
    if name.startswith(("aten.max.dim", "aten.min.dim")):
        return [r, r]
    return [r]


# -- views ------------------------------------------------------------------


def _view_result(sizes, strides, offset):
    return ("view", tuple(sizes), tuple(strides), offset)


def _infer_size(shape, numel):
    shape = [_plain(x) for x in shape]
    neg = [i for i, v in enumerate(shape) if not _is_sym(v) and v == -1]
    if len(neg) > 1:
        raise SymbolicShapeUnsupported("view: more than one -1")
    if neg:
        known = 1
        for i, v in enumerate(shape):
            if i != neg[0]:
                known = known * v
        shape[neg[0]] = numel // known
    return tuple(shape)


def _numel(sizes):
    return functools.reduce(operator.mul, sizes, 1)


@rule("view", "_unsafe_view", "reshape", "_reshape_alias", "view_copy")
def _r_view(func, args, kwargs, out):
    t = args[0]
    shape = args[1] if len(args) > 1 else kwargs.get("size", kwargs.get("shape"))
    if not isinstance(shape, (list, tuple)):
        shape = [shape]
    s = sizes_of(t)
    new = _infer_size(shape, _numel(s))
    if not _base("is_contiguous")(t):
        # A view of a non-contiguous tensor: only correct as a fresh layout here.
        return [new]
    return [_view_result(new, contiguous_strides(new), offset_of(t))]


@rule("expand", "expand_copy")
def _r_expand(func, args, kwargs, out):
    t = args[0]
    shape = [_plain(x) for x in args[1]]
    s, st = sizes_of(t), strides_of(t)
    k = len(shape) - len(s)
    sizes, strides = [], []
    for i, v in enumerate(shape):
        if i < k:
            sizes.append(v)
            strides.append(0)
            continue
        cur = s[i - k]
        if not _is_sym(v) and v == -1:
            sizes.append(cur)
            strides.append(st[i - k])
        elif _concrete(cur) == 1 and _concrete(v) != 1:
            sizes.append(v)
            strides.append(0)
        else:
            sizes.append(cur if _is_sym(cur) else v)
            strides.append(st[i - k])
    return [_view_result(sizes, strides, offset_of(t))]


@rule("permute")
def _r_permute(func, args, kwargs, out):
    t = args[0]
    dims = [_norm_dim(d, t.dim()) for d in args[1]]
    s, st = sizes_of(t), strides_of(t)
    return [_view_result([s[d] for d in dims], [st[d] for d in dims], offset_of(t))]


@rule("transpose", "swapaxes")
def _r_transpose(func, args, kwargs, out):
    t = args[0]
    n = t.dim()
    a, b = _norm_dim(args[1], n), _norm_dim(args[2], n)
    s, st = list(sizes_of(t)), list(strides_of(t))
    s[a], s[b] = s[b], s[a]
    st[a], st[b] = st[b], st[a]
    return [_view_result(s, st, offset_of(t))]


@rule("t")
def _r_t(func, args, kwargs, out):
    t = args[0]
    s, st = list(sizes_of(t)), list(strides_of(t))
    return [_view_result(s[::-1], st[::-1], offset_of(t))]


@rule("unsqueeze")
def _r_unsqueeze(func, args, kwargs, out):
    t = args[0]
    n = t.dim()
    d = _concrete(args[1])
    d = d + n + 1 if d < 0 else d
    s, st = list(sizes_of(t)), list(strides_of(t))
    new_st = st[d] * _max1(s[d]) if d < n else 1
    s.insert(d, 1)
    st.insert(d, new_st)
    return [_view_result(s, st, offset_of(t))]


@rule("squeeze")
def _r_squeeze(func, args, kwargs, out):
    t = args[0]
    n = t.dim()
    s, st = list(sizes_of(t)), list(strides_of(t))
    if len(args) > 1:
        dims = args[1] if isinstance(args[1], (list, tuple)) else [args[1]]
        dims = {_norm_dim(d, n) for d in dims}
    else:
        dims = set(range(n))
    keep = [i for i in range(n) if not (i in dims and _concrete(s[i]) == 1 and not _is_sym(s[i]))]
    return [_view_result([s[i] for i in keep], [st[i] for i in keep], offset_of(t))]


@rule("select")
def _r_select(func, args, kwargs, out):
    t = args[0]
    n = t.dim()
    d = _norm_dim(args[1], n)
    idx = _plain(args[2])
    s, st = list(sizes_of(t)), list(strides_of(t))
    if not _is_sym(idx) and idx < 0:
        idx = s[d] + idx
    off = offset_of(t) + idx * st[d]
    del s[d]
    del st[d]
    return [_view_result(s, st, off)]


@rule("slice")
def _r_slice(func, args, kwargs, out):
    t = args[0]
    n = t.dim()
    d = _norm_dim(args[1] if len(args) > 1 else kwargs.get("dim", 0), n)
    start = _plain(args[2] if len(args) > 2 else kwargs.get("start"))
    end = _plain(args[3] if len(args) > 3 else kwargs.get("end"))
    step = _plain(args[4] if len(args) > 4 else kwargs.get("step", 1))
    s, st = list(sizes_of(t)), list(strides_of(t))
    size = s[d]
    start = 0 if start is None else start
    end = size if end is None else end
    if _concrete(step) != 1 and (_is_sym(size) or _is_sym(start) or _is_sym(end)):
        raise SymbolicShapeUnsupported("torchnative symbolic shapes: aten.slice with step != 1 on a symbolic dim")
    # Clamp the way upstream's slice does; comparisons on symbols become guards.
    if not _is_sym(start) and start < 0:
        start = start + size
    if not _is_sym(end) and end < 0:
        end = end + size
    if _concrete(start) < 0:
        start = 0
    if _concrete(end) > _concrete(size):
        end = size
    if _concrete(start) > _concrete(end):
        length = 0
    else:
        length = end - start
    if _concrete(step) != 1:
        length = (length + step - 1) // step
    off = offset_of(t) + start * st[d]
    s[d] = length
    st[d] = st[d] * step
    return [_view_result(s, st, off)]


@rule("narrow")
def _r_narrow(func, args, kwargs, out):
    t = args[0]
    d = _norm_dim(args[1], t.dim())
    start, length = _plain(args[2]), _plain(args[3])
    s, st = list(sizes_of(t)), list(strides_of(t))
    off = offset_of(t) + start * st[d]
    s[d] = length
    return [_view_result(s, st, off)]


@rule("split", "split_with_sizes", "chunk", "unbind", "unsafe_split")
def _r_split(func, args, kwargs, out):
    t = args[0]
    name = str(func)
    s, st = list(sizes_of(t)), list(strides_of(t))
    n = len(s)
    outs = out if isinstance(out, (list, tuple)) else [out]
    if name.startswith("aten.unbind"):
        d = _norm_dim(args[1] if len(args) > 1 else kwargs.get("dim", 0), n)
        res = []
        for i in range(len(outs)):
            ss, sst = s[:d] + s[d + 1:], st[:d] + st[d + 1:]
            res.append(_view_result(ss, sst, offset_of(t) + i * st[d]))
        return res
    d = _norm_dim(args[2] if len(args) > 2 else kwargs.get("dim", 0), n)
    if name.startswith("aten.split_with_sizes"):
        pieces = [_plain(x) for x in args[1]]
    elif name.startswith("aten.chunk"):
        chunks = _concrete(args[1])
        size = s[d]
        each = (size + chunks - 1) // chunks
        pieces = [each] * (len(outs) - 1)
        pieces.append(size - each * (len(outs) - 1))
    else:
        each = _plain(args[1])
        size = s[d]
        pieces = [each] * (len(outs) - 1)
        pieces.append(size - each * (len(outs) - 1))
    res = []
    off = offset_of(t)
    for p in pieces:
        ss = list(s)
        ss[d] = p
        res.append(_view_result(ss, st, off))
        off = off + p * st[d]
    return res


@rule("cat", "concat", "concatenate")
def _r_cat(func, args, kwargs, out):
    ts = [x for x in args[0]]
    d = args[1] if len(args) > 1 else kwargs.get("dim", 0)
    shapes = [sizes_of(x) for x in ts if not (x.dim() == 1 and _concrete(sizes_of(x)[0]) == 0)]
    if not shapes:
        return [sizes_of(ts[0])]
    n = len(shapes[0])
    d = _norm_dim(d, n)
    total = 0
    for sh in shapes:
        total = total + sh[d]
    r = list(shapes[0])
    for sh in shapes:
        for i in range(n):
            if i != d and _is_sym(sh[i]) and not _is_sym(r[i]):
                r[i] = sh[i]
    r[d] = total
    return [tuple(r)]


@rule("stack")
def _r_stack(func, args, kwargs, out):
    ts = list(args[0])
    s = list(sizes_of(ts[0]))
    d = _concrete(args[1] if len(args) > 1 else kwargs.get("dim", 0))
    d = d + len(s) + 1 if d < 0 else d
    s.insert(d, len(ts))
    return [tuple(s)]


@rule("index_select")
def _r_index_select(func, args, kwargs, out):
    t = args[0]
    d = _norm_dim(args[1], t.dim())
    s = list(sizes_of(t))
    idx = sizes_of(args[2])
    s[d] = idx[0] if idx else 1
    return [tuple(s)]


@rule("gather")
def _r_gather(func, args, kwargs, out):
    return [sizes_of(args[2])]


@rule("index")
def _r_index(func, args, kwargs, out):
    t = args[0]
    indices = args[1]
    s = list(sizes_of(t))
    tens = [(i, x) for i, x in enumerate(indices) if x is not None]
    if any(x.dtype == _torch.bool for _, x in tens):
        raise SymbolicShapeUnsupported("torchnative symbolic shapes: boolean-mask index has a data-dependent size")
    bshape = broadcast(*[sizes_of(x) for _, x in tens])
    pos = [i for i, _ in tens]
    contiguous_block = pos == list(range(pos[0], pos[0] + len(pos)))
    rest = [s[i] for i in range(len(s)) if i not in pos]
    if contiguous_block:
        before = [s[i] for i in range(pos[0])]
        after = [s[i] for i in range(pos[-1] + 1, len(s))]
        return [tuple(before) + bshape + tuple(after)]
    return [bshape + tuple(rest)]


@rule("repeat")
def _r_repeat(func, args, kwargs, out):
    s = sizes_of(args[0])
    reps = [_plain(r) for r in args[1]]
    k = len(reps) - len(s)
    s = (1,) * k + tuple(s)
    return [tuple(a * b for a, b in zip(s, reps))]


@rule("repeat_interleave")
def _r_repeat_interleave(func, args, kwargs, out):
    t = args[0]
    reps = args[1] if len(args) > 1 else kwargs.get("repeats")
    if isinstance(reps, _torch.Tensor):
        raise SymbolicShapeUnsupported("repeat_interleave with tensor repeats")
    d = args[2] if len(args) > 2 else kwargs.get("dim")
    s = list(sizes_of(t))
    if d is None:
        return [(_numel(s) * _plain(reps),)]
    d = _norm_dim(d, len(s))
    s[d] = s[d] * _plain(reps)
    return [tuple(s)]


@rule("index_put_", "index_copy_", "index_add_", "scatter_", "scatter_add_")
def _r_inplace_index(func, args, kwargs, out):
    t = args[0]
    return [_view_result(sizes_of(t), strides_of(t), offset_of(t))]


@rule("index_put", "index_copy", "index_add", "scatter", "scatter_add")
def _r_index_write(func, args, kwargs, out):
    return [sizes_of(args[0])]


@rule("sym_size", "sym_stride", "sym_numel", "sym_storage_offset", "_local_scalar_dense",
      "item", "_assert_tensor_metadata", "_assert_async", "set_")
def _r_noshape(func, args, kwargs, out):
    return None


def _aten_name(func):
    s = str(func)
    if s.startswith("aten."):
        return s[5:].split(".")[0]
    parts = s.split(".")
    if len(parts) >= 2:
        return parts[0] + "." + parts[1]
    return None


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------


def _symbolic_inputs(flat):
    for a in flat:
        if isinstance(a, _SymHint) or _is_sym(a):
            return True
        if isinstance(a, _torch.Tensor) and _sym_meta(a) is not None:
            return True
    return False


def _flatten(args, kwargs):
    flat = []

    def walk(v):
        if isinstance(v, (list, tuple)):
            for x in v:
                walk(x)
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        else:
            flat.append(v)

    walk(args)
    walk(kwargs)
    return flat


def _attach(out, specs, func):
    outs = list(out) if isinstance(out, (list, tuple)) else [out]
    if specs is None:
        return
    if len(specs) < len([o for o in outs if isinstance(o, _torch.Tensor)]):
        specs = list(specs) + [None] * (len(outs) - len(specs))
    for o, spec in zip(outs, specs):
        if spec is None or not isinstance(o, _torch.Tensor):
            continue
        hs = tuple(_base("size")(o))
        hst = tuple(_base("stride")(o))
        if isinstance(spec, tuple) and spec and spec[0] == "view":
            _, sizes, strides, offset = spec
        else:
            sizes = tuple(spec)
            strides = None
            offset = 0
        sizes = tuple(_plain(x) for x in sizes)
        if tuple(_concrete(x) for x in sizes) != hs:
            raise SymbolicShapeUnsupported(
                f"torchnative symbolic shapes: rule for {func} gave sizes {sizes} "
                f"(hint {tuple(_concrete(x) for x in sizes)}) but the kernel produced {hs}"
            )
        if strides is None:
            strides = _like_strides(sizes, hst, hs)
            offset = _base("storage_offset")(o)
        else:
            strides = tuple(_plain(x) for x in strides)
            if tuple(_concrete(x) for x in strides) != hst and _numel(hs) > 1:
                # The kernel laid the view out differently (it copied); describe its layout.
                strides = _like_strides(sizes, hst, hs)
                offset = _base("storage_offset")(o)
        _set_sym(o, sizes, strides, offset)


#: op -> how many fake dispatches used upstream's Python meta registration because the
#: shim's meta kernel refused (#263). Read by tests; never consulted for values.
UPSTREAM_META_USED = {}

#: The ops whose upstream Python meta registration may stand in for a refusing shim
#: meta kernel. A list, not "anything in meta_table": each entry was traced through a
#: real model (the in-place KV-cache writes of transformers' StaticCache).
_UPSTREAM_META_OPS = ("aten.index_put_.default", "aten.index_put.default")


def _upstream_meta(func):
    if str(func) not in _UPSTREAM_META_OPS:
        return None
    import torch._decomp as decomp
    import torch._meta_registrations  # noqa: F401  (populates meta_table)

    return decomp.meta_table.get(func)


def _install_dispatch(fake_tensor_mod):
    FakeTensorMode = fake_tensor_mod.FakeTensorMode
    orig = FakeTensorMode._dispatch_impl
    if getattr(orig, "_tn_symbolic", False):
        return

    def run_kernel(self, func, types, args, kwargs):
        try:
            return orig(self, func, types, args, kwargs)
        except NotImplementedError as exc:
            meta = _upstream_meta(func)
            if meta is None or "torch._C shim" not in str(exc):
                raise
            # The shim has no meta kernel (or no layout claim) for this op; upstream's
            # own Python meta registration computes the metadata. Values still come
            # from the shim's kernel at run time. Counted, never silent.
            UPSTREAM_META_USED[str(func)] = UPSTREAM_META_USED.get(str(func), 0) + 1
            return meta(*args, **kwargs)

    def _dispatch_impl(self, func, types, args, kwargs):
        flat = _flatten(args, kwargs)
        if not _symbolic_inputs(flat):
            return run_kernel(self, func, types, args, kwargs)
        name = _aten_name(func)
        r = _RULES.get(name)
        if r is None:
            raise SymbolicShapeUnsupported(
                f"torchnative symbolic shapes: no symbolic shape rule for {func}; "
                f"the frame is compiled with static shapes instead"
            )
        sargs = _plain(tuple(args))
        skw = {k: _plain(v) for k, v in kwargs.items()}
        hargs = _strip(tuple(args))
        hkw = {k: _strip(v) for k, v in kwargs.items()}
        # The kernel (or the decomposition FakeTensorMode picks for it) runs on hints
        # only: the inputs' symbols are hidden for the call, so nothing below this op
        # needs a rule of its own. The symbolic answer is the rule's, checked below.
        hidden = []
        for a in flat:
            if isinstance(a, _torch.Tensor):
                m = a.__dict__.pop("_tn_sym", None)
                if m is not None:
                    hidden.append((a, m))
        try:
            out = run_kernel(self, func, types, hargs, hkw)
        finally:
            for a, m in hidden:
                a.__dict__["_tn_sym"] = m
        specs = r(func, sargs, skw, out)
        _attach(out, specs, func)
        return out

    _dispatch_impl._tn_symbolic = True
    FakeTensorMode._dispatch_impl = _dispatch_impl

    # The fake-tensor cache synthesizes its entry's output with `Tensor.set_(storage,
    # offset, shape, stride)`; with symbolic metadata the shim's `set_` would take each
    # size through `__index__`, which evaluates (and specializes) the symbol. A call with
    # symbolic inputs therefore bypasses the cache; every other call keeps it.
    orig_cached = FakeTensorMode._cached_dispatch_impl

    def _cached_dispatch_impl(self, func, types, args, kwargs):
        if _symbolic_inputs(_flatten(args, kwargs)):
            return self._dispatch_impl(func, types, args, kwargs)
        return orig_cached(self, func, types, args, kwargs)

    _cached_dispatch_impl._tn_symbolic = True
    FakeTensorMode._cached_dispatch_impl = _cached_dispatch_impl


def _install_resolve(boot):
    Overloads = boot._Overloads
    orig = Overloads.resolve
    if getattr(orig, "_tn_symbolic", False):
        return

    def resolve(self, args, kwargs):
        try:
            return orig(self, args, kwargs)
        except TypeError:
            if not (any(_has_sym_scalar(a) for a in args) or any(_has_sym_scalar(v) for v in kwargs.values())):
                raise
        hargs = tuple(_to_hint(a) for a in args)
        hkw = {k: _to_hint(v) for k, v in kwargs.items()}
        return orig(self, hargs, hkw)

    resolve._tn_symbolic = True
    Overloads.resolve = resolve


def _install_factories():
    """`torch.empty_strided` called with symbols outside a mode (meta_utils, under no_dispatch)."""
    import torch

    orig = torch.empty_strided
    if getattr(orig, "_tn_symbolic", False):
        return

    def empty_strided(size, stride, *args, **kwargs):
        if not (_has_sym_scalar(size) or _has_sym_scalar(stride)):
            return orig(size, stride, *args, **kwargs)
        sizes = tuple(size)
        strides = tuple(stride)
        r = orig([_concrete(x) for x in sizes], [_concrete(x) for x in strides], *args, **kwargs)
        _set_sym(r, sizes, strides, 0)
        return r

    empty_strided._tn_symbolic = True
    torch.empty_strided = empty_strided


def _install_make_subclass():
    """`Tensor._make_subclass` builds the `FakeTensor` around a meta tensor: keep its symbols."""
    import torch

    T = torch._C.TensorBase
    sm = T.__dict__.get("_make_subclass")
    func = sm.__func__ if isinstance(sm, staticmethod) else sm
    if func is None or getattr(func, "_tn_symbolic", False):
        return

    # `functools.wraps`: Dynamo's polyfill for `_make_subclass` compares signatures.
    @functools.wraps(func)
    def _make_subclass(cls, data, *args, **kwargs):
        made = func(cls, data, *args, **kwargs)
        m = _sym_meta(data)
        if m is not None:
            made.__dict__["_tn_sym"] = m
        return made

    _make_subclass._tn_symbolic = True
    T._make_subclass = staticmethod(_make_subclass)


def install():
    """Install the symbolic-shape bridge. Idempotent."""
    global _installed, _torch, _SymTypes
    if _installed:
        return
    import torch
    import torch._subclasses.fake_tensor as ft

    _torch = torch
    _SymTypes = (torch.SymInt, torch.SymFloat, torch.SymBool)
    boot = sys.modules.get("_torch_c_bootstrap")
    if boot is None or not hasattr(boot, "_Overloads"):
        raise RuntimeError(
            "torchnative symbolic shapes: the shim's bootstrap (_torch_c_bootstrap._Overloads) "
            "is not loaded; this is not torchnative's torch._C"
        )
    _install_resolve(boot)
    _install_fake_tensor_methods(ft.FakeTensor)
    _install_dispatch(ft)
    _install_factories()
    _install_make_subclass()
    _installed = True


@rule("as_strided", "as_strided_copy")
def _r_as_strided(func, args, kwargs, out):
    t = args[0]
    size = args[1] if len(args) > 1 else kwargs["size"]
    stride = args[2] if len(args) > 2 else kwargs["stride"]
    off = args[3] if len(args) > 3 else kwargs.get("storage_offset")
    if off is None:
        off = offset_of(t)
    return [_view_result([_plain(x) for x in size], [_plain(x) for x in stride], _plain(off))]


@rule("diff")
def _r_diff(func, args, kwargs, out):
    t = args[0]
    n = _concrete(args[1] if len(args) > 1 else kwargs.get("n", 1))
    s = list(sizes_of(t))
    d = _norm_dim(args[2] if len(args) > 2 else kwargs.get("dim", -1), len(s))
    pre = args[3] if len(args) > 3 else kwargs.get("prepend")
    app = args[4] if len(args) > 4 else kwargs.get("append")
    size = s[d]
    if isinstance(pre, _torch.Tensor):
        size = size + sizes_of(pre)[d]
    if isinstance(app, _torch.Tensor):
        size = size + sizes_of(app)[d]
    s[d] = size - n
    return [tuple(s)]


@rule("prims.broadcast_in_dim")
def _r_broadcast_in_dim(func, args, kwargs, out):
    t = args[0]
    shape = [_plain(x) for x in args[1]]
    bdims = list(args[2])
    s, st = sizes_of(t), strides_of(t)
    strides = [0] * len(shape)
    for i, d in enumerate(bdims):
        if not (_concrete(s[i]) == 1 and _concrete(shape[d]) != 1):
            strides[d] = st[i]
    return [_view_result(shape, strides, offset_of(t))]


for _n in (
    "convert_element_type", "add", "sub", "mul", "div", "neg", "exp", "log", "sqrt",
    "rsqrt", "sin", "cos", "tanh", "abs", "eq", "ne", "lt", "le", "gt", "ge", "where",
    "maximum", "minimum", "pow", "bitwise_and", "bitwise_or", "bitwise_not", "sign",
    "reciprocal", "erf", "fill", "clone", "copy_to", "sigmoid", "remainder", "fmod",
):
    _RULES["prims." + _n] = _elementwise


@rule("prims.view_of", "prims.view_element_type")
def _r_view_of(func, args, kwargs, out):
    t = args[0]
    return [_view_result(sizes_of(t), strides_of(t), offset_of(t))]


@rule("prims.squeeze")
def _r_prims_squeeze(func, args, kwargs, out):
    t = args[0]
    dims = {_norm_dim(d, t.dim()) for d in args[1]}
    s, st = sizes_of(t), strides_of(t)
    keep = [i for i in range(len(s)) if i not in dims]
    return [_view_result([s[i] for i in keep], [st[i] for i in keep], offset_of(t))]


@rule("prims.transpose")
def _r_prims_transpose(func, args, kwargs, out):
    return _r_permute(func, args, kwargs, out)


@rule("prims.sum", "prims.amax", "prims.amin", "prims.prod", "prims.var")
def _r_prims_reduce(func, args, kwargs, out):
    return [_reduce_shape(sizes_of(args[0]), list(args[1]) if args[1] is not None else None, False)]
