"""The fixed fusion pattern library and its matcher (issue #68, docs/graph/FUSION.md §5.5).

**Owner decisions this module rests on** (2026-11-13, owner, relayed by the
leader; recorded in FUSION.md §1 and §8):

* FUSION.md §8 Q1 (b): fused kernels come from **a fixed pattern library
  matched at run time**. torchnative ships a fixed set of fused kernels; the
  compile backend matches captured regions against that set. There is no
  per-model build-time capture pipeline.
* Symbolic sequence length is approved: a kernel takes the length as a
  symbolic dimension, not bucketed shapes, where the pattern allows.

Everything else here (the first five patterns, the matching rule, the report
shape) is an **agent proposal, not confirmed**.

What it does: for each region `regions.partition` reports, it names the one
pattern the region is, or lists why it is none. It does not compile, fuse or
dispatch anything, and there are no kernels yet. Pure Python, no dependency
beyond `torch` (for the dtypes) and `torch.fx` (the graph it reads).

## The matching rule

A region matches a pattern spelling when all of these hold:

1. **Op sequence.** The region's work ops (its members minus views), in graph
   order, are exactly the spelling's steps' op names. Views are transparent:
   they are index maps and cost nothing in a kernel.
2. **Structure.** Each step's operands are what the step says: a ref to a
   pattern input or an earlier step (resolved backwards through the region's
   own views), a half of the last dim (`Half`, for rotary), a constant
   (`Const`), a literal, or a sequence of those (`Seq`). Commutative steps
   (`add`, `mul`) may have their two operands either way round. A keyword the
   step does not name refuses the match. So an operand wired to the wrong
   node fails even when the op sequence is equal.
3. **Outputs.** The region's outputs (members read outside it) are the
   spelling's output steps, all of them except those listed in
   `optional_outputs` (`add_rms_norm`'s sum). A region whose intermediate value
   is also read outside is not this kernel.
4. **Shapes, symbolic.** Each bound input's shape (from Dynamo's
   `example_value`, FX's `val` or `tensor_meta`) fits its `Spec`. Dims with
   the same name must be equal. A dim is *symbolic* when its size is not an
   `int` (a `SymInt`); every dim may be symbolic except those the pattern lists
   in `static` (the dims a kernel tiles over: hidden, head, intermediate). The
   match records which dims were symbolic.
5. **Dtypes.** Inputs with the same dtype variable have the same dtype, and it
   is one of the variable's allowed dtypes.
6. **Relations.** The pattern's own `check` (for attention: grouped-query heads,
   mask broadcast, no dropout).

Missing meta is not guessed: the region is listed as unmatched, naming the
input. Two pattern names matching one region is a library defect, listed as
`ambiguous`, never resolved by order.

## What happens to a region that matches nothing

It is listed in `FusionReport.unmatched` with the reason per candidate, and it
runs unfused: each of its ops through `_aten_dispatch` on its own, exactly as
without fusion. Whether such a region should instead be refused is the
owner's open question (FUSION.md §8 Q2); until it is answered the matcher
lists and does not refuse, and says so in every report (`UNMATCHED_RUNS`).

## How a matched region would dispatch (not built)

One op, `torchnative_fused.<pattern>`, through the one door `_aten_dispatch`
(AGENTS.md §11): no bypass. The region's views stay ordinary view ops in
front of it. Wiring that into the `torchnative.compile` backend waits for the
owner's other answers (FUSION.md §8) and for #268/#269.

What this cannot find: whether a real Dynamo graph spells these ops the way
the patterns do (`tests/export/test_fusion_capture.py` observes that once
#268/#269 let the decode step trace), view parameters other than the rotary
halves (views not named by a step are passed through), and anything about
numbers: there is no kernel to compare.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Callable

import torch

from .regions import ATTENTION, ELEMENTWISE, REDUCTION, op_name

__all__ = [
    "PATTERNS",
    "Const",
    "FusionReport",
    "Half",
    "Match",
    "Opt",
    "Pattern",
    "Seq",
    "Spec",
    "Step",
    "check_registry",
    "match_graph",
    "match_partition",
    "match_region",
]


# ---------------------------------------------------------------------------
# The declarative vocabulary
# ---------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Const:
    """A constant operand that satisfies `pred`; `text` says what it must be."""

    pred: Callable[[Any], bool]
    text: str


@dataclasses.dataclass(frozen=True)
class Opt:
    """A keyword that may be absent (or `None`); when present it must match `spec`."""

    spec: Any


@dataclasses.dataclass(frozen=True)
class Seq:
    """A list or tuple operand (`torch.cat`'s tensors), matched element by element."""

    items: tuple

    def __init__(self, *items):
        object.__setattr__(self, "items", tuple(items))


@dataclasses.dataclass(frozen=True)
class Half:
    """`ref[..., :D//2]` (`which=0`) or `ref[..., D//2:]` (`which=1`), D the last dim."""

    ref: str
    which: int


@dataclasses.dataclass(frozen=True)
class Spec:
    """A pattern input's shape and dtype.

    `dims` names each dim; a name starting with `...` binds any number of
    leading dims as one tuple; an `int` is a literal size. `dtype` names a
    dtype variable of the pattern, or `None` for any.
    """

    dims: tuple
    dtype: str | None = None


@dataclasses.dataclass(frozen=True)
class Step:
    """One work op of a pattern.

    `args` are the positional operands. `pos_kw` names the positional
    arguments past `args` (so `mean(p, -1, keepdim=True)` and
    `mean(p, dim=-1, keepdim=True)` read the same), and `kwargs` holds the
    spec of every keyword the step allows.
    """

    name: str
    op: str
    args: tuple = ()
    kwargs: dict = dataclasses.field(default_factory=dict)
    commutative: bool = False
    pos_kw: tuple = ()


@dataclasses.dataclass(frozen=True)
class Pattern:
    """One spelling of a fused kernel's computation."""

    inputs: dict           # name -> Spec, in binding order
    steps: tuple           # (Step, ...) in graph order
    outputs: tuple         # step names whose values leave the region
    dtypes: dict           # dtype variable -> frozenset of allowed dtypes
    static: frozenset      # dim names that must be concrete ints
    optional: frozenset = frozenset()   # inputs that may be absent
    optional_outputs: frozenset = frozenset()  # outputs that may be read by nothing outside
    check: Callable | None = None       # relations: returns a reason or None
    doc: str = ""


def _is_number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


EPS = Const(lambda v: _is_number(v) and v > 0, "a positive number (epsilon)")
LAST_DIM = Const(lambda v: v in (-1, (-1,), [-1]), "-1 (the last dim)")
ANY_DTYPE = Const(lambda v: isinstance(v, torch.dtype), "a dtype")
BOOL = Const(lambda v: isinstance(v, bool), "a bool")
SCALE = Const(lambda v: v is None or _is_number(v), "a number or None")
NO_DROPOUT = Const(lambda v: _is_number(v) and v == 0, "0 (inference: no dropout)")

FLOATS = frozenset({torch.float32, torch.float16, torch.bfloat16})


# ---------------------------------------------------------------------------
# The first set (agent proposal, not confirmed; FUSION.md §5.5 says why these)
# ---------------------------------------------------------------------------


def _rms_steps(x, prefix=()):
    """Qwen3RMSNorm / LlamaRMSNorm (transformers 5.15.1, modeling_qwen3.py:59-64):
    `h = x.to(f32); h * rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)`, times the
    weight, with and without the cast back to the input dtype."""
    core = prefix + (
        Step("h", "to", (x,), {"dtype": torch.float32}, pos_kw=("dtype",)),
        Step("p", "pow", ("h", 2)),
        Step("m", "mean", ("p",), {"dim": LAST_DIM, "keepdim": True},
             pos_kw=("dim", "keepdim")),
        Step("a", "add", ("m", EPS), commutative=True),
        Step("r", "rsqrt", ("a",)),
        Step("hn", "mul", ("h", "r"), commutative=True),
    )
    cast_back = core + (
        Step("c", "to", ("hn",), {"dtype": ANY_DTYPE}, pos_kw=("dtype",)),
        Step("y", "mul", ("w", "c"), commutative=True),
    )
    plain = core + (Step("y", "mul", ("w", "hn"), commutative=True),)
    return cast_back, plain


def _rms_norm():
    def make(steps):
        return Pattern(
            inputs={"x": Spec(("...lead", "N"), "T"), "w": Spec(("N",), "W")},
            steps=steps, outputs=("y",),
            dtypes={"T": FLOATS, "W": FLOATS}, static=frozenset({"N"}),
            doc="RMSNorm over the last dim, accumulated in float32",
        )
    return tuple(make(s) for s in _rms_steps("x"))


def _add_rms_norm():
    def make(steps):
        return Pattern(
            inputs={"res": Spec(("...lead", "N"), "T"), "xin": Spec(("...lead", "N"), "T"),
                    "w": Spec(("N",), "W")},
            steps=steps, outputs=("s", "y"), optional_outputs=frozenset({"s"}),
            dtypes={"T": FLOATS, "W": FLOATS}, static=frozenset({"N"}),
            doc="residual add, then RMSNorm of the sum; the norm leaves, and the sum "
                "too when something outside reads it (the final norm's sum is read by "
                "nothing else)",
        )
    prefix = (Step("s", "add", ("res", "xin"), commutative=True),)
    return tuple(make(s) for s in _rms_steps("s", prefix))


def _silu_mul():
    return (Pattern(
        inputs={"g": Spec(("...lead", "I"), "T"), "u": Spec(("...lead", "I"), "T")},
        steps=(
            Step("s", "silu", ("g",), {"inplace": Opt(False)}, pos_kw=("inplace",)),
            Step("y", "mul", ("s", "u"), commutative=True),
        ),
        outputs=("y",), dtypes={"T": FLOATS}, static=frozenset({"I"}),
        doc="SwiGLU: silu(gate) * up (Qwen3MLP, LlamaMLP)",
    ),)


def _rotary():
    return (Pattern(
        inputs={"x": Spec(("B", "H", "S", "D"), "T"), "cos": Spec(("B", 1, "S", "D"), "T"),
                "sin": Spec(("B", 1, "S", "D"), "T")},
        steps=(
            Step("a", "mul", ("x", "cos"), commutative=True),
            Step("n", "neg", (Half("x", 1),)),
            Step("c", "cat", (Seq("n", Half("x", 0)),), {"dim": LAST_DIM}, pos_kw=("dim",)),
            Step("m", "mul", ("c", "sin"), commutative=True),
            Step("y", "add", ("a", "m"), commutative=True),
        ),
        outputs=("y",), dtypes={"T": FLOATS}, static=frozenset({"H", "D"}),
        doc="x * cos + rotate_half(x) * sin (apply_rotary_pos_emb, modeling_qwen3.py:140-170)",
    ),)


def _sdpa_check(m):
    d, node = m.dims, m.steps["o"]
    gqa = node.kwargs.get("enable_gqa", False)
    hq, hkv = d["Hq"], d["Hkv"]
    if hq % hkv:
        return "Hq %s is not a multiple of Hkv %s" % (hq, hkv)
    if hq != hkv and not gqa:
        return "Hq %s != Hkv %s and enable_gqa is not set" % (hq, hkv)
    if "mask" in m.inputs:
        for name, full in (("Bm", "B"), ("Hm", "Hq")):
            v = d[name]
            if not (_same(v, 1) or _same(v, d[full])):
                return "mask dim %s = %s is neither 1 nor %s = %s" % (name, v, full, d[full])
        mdt = _meta(m.inputs["mask"]).dtype
        if mdt != torch.bool and mdt != m.dtypes["T"]:
            return "mask dtype %s is neither torch.bool nor T = %s" % (mdt, m.dtypes["T"])
    return None


def _sdpa_kv():
    return (Pattern(
        inputs={"q": Spec(("B", "Hq", "Sq", "D"), "T"), "k": Spec(("B", "Hkv", "Skv", "D"), "T"),
                "v": Spec(("B", "Hkv", "Skv", "D"), "T"),
                "mask": Spec(("Bm", "Hm", "Sq", "Skv"))},
        steps=(Step("o", "scaled_dot_product_attention", ("q", "k", "v"), {
            "attn_mask": Opt("mask"), "dropout_p": Opt(NO_DROPOUT), "scale": Opt(SCALE),
            "is_causal": Opt(BOOL), "enable_gqa": Opt(BOOL),
        }, pos_kw=("attn_mask", "dropout_p", "is_causal", "scale", "enable_gqa")),),
        outputs=("o",), dtypes={"T": FLOATS}, static=frozenset({"Hq", "Hkv", "D"}),
        optional=frozenset({"mask"}), check=_sdpa_check,
        doc="attention over a KV cache: the query and cache lengths are symbolic",
    ),)


#: name -> spellings. Order is the report's order only; it never breaks a tie.
PATTERNS = {
    "rms_norm": _rms_norm(),
    "add_rms_norm": _add_rms_norm(),
    "silu_mul": _silu_mul(),
    "rotary": _rotary(),
    "sdpa_kv": _sdpa_kv(),
}


# ---------------------------------------------------------------------------
# The registry checks itself
# ---------------------------------------------------------------------------


def _refs(spec):
    if isinstance(spec, str):
        yield spec
    elif isinstance(spec, Half):
        yield spec.ref
    elif isinstance(spec, Opt):
        yield from _refs(spec.spec)
    elif isinstance(spec, Seq):
        for s in spec.items:
            yield from _refs(s)


def check_registry(patterns):
    """Raise `ValueError` naming every defect; return `None` when there is none.

    A pattern whose step reads an undefined name, whose op the partitioner can
    never put in a region, or whose required input no step reads, could never
    match, so it would be dead weight that looks like coverage.
    """
    fusable = ELEMENTWISE | REDUCTION | ATTENTION
    errors = []
    for name, spellings in patterns.items():
        if not spellings:
            errors.append("%s: no spelling" % name)
        for i, p in enumerate(spellings):
            where = "%s[%d]" % (name, i)
            defined = set(p.inputs)
            used = set()
            for s in p.steps:
                if s.name in defined:
                    errors.append("%s: step %s reuses a name" % (where, s.name))
                if s.op not in fusable:
                    errors.append("%s: step %s op %s is in no fusable category"
                                  % (where, s.name, s.op))
                for spec in tuple(s.args) + tuple(s.kwargs.values()):
                    for ref in _refs(spec):
                        used.add(ref)
                        if ref not in defined:
                            errors.append("%s: step %s reads %s, which is no input or "
                                          "earlier step" % (where, s.name, ref))
                defined.add(s.name)
            for out in p.outputs:
                if out not in {s.name for s in p.steps}:
                    errors.append("%s: output %s is no step" % (where, out))
            for out in p.optional_outputs:
                if out not in p.outputs:
                    errors.append("%s: optional output %s is no output" % (where, out))
            for inp, spec in p.inputs.items():
                if inp not in used:
                    errors.append("%s: input %s is read by no step" % (where, inp))
                if spec.dtype is not None and spec.dtype not in p.dtypes:
                    errors.append("%s: input %s has dtype variable %s with no allowed set"
                                  % (where, inp, spec.dtype))
            names = {d for spec in p.inputs.values() for d in spec.dims if isinstance(d, str)}
            for dim in p.static:
                if dim not in names:
                    errors.append("%s: static dim %s is in no input" % (where, dim))
    if errors:
        raise ValueError("; ".join(errors))
    return None


check_registry(PATTERNS)


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


class _NoMatch(Exception):
    pass


def _same(a, b):
    if isinstance(a, tuple) or isinstance(b, tuple):
        return (isinstance(a, tuple) and isinstance(b, tuple) and len(a) == len(b)
                and all(_same(x, y) for x, y in zip(a, b)))
    if isinstance(a, int) and isinstance(b, int):
        return a == b
    if isinstance(a, int) or isinstance(b, int):
        return False  # a symbol is not provably a number
    return a is b or str(a) == str(b)


def _symbolic(v):
    if isinstance(v, tuple):
        return any(_symbolic(x) for x in v)
    return not isinstance(v, int)


def _eq(a, b):
    try:
        return bool(a == b)
    except Exception:  # noqa: BLE001  (a tensor constant compares elementwise)
        return False


def _meta(node):
    for key in ("example_value", "val", "tensor_meta"):
        v = node.meta.get(key)
        if v is not None and hasattr(v, "shape") and hasattr(v, "dtype"):
            return v
    return None


def _node(x):
    import torch.fx
    return isinstance(x, torch.fx.Node)


class Match:
    """A region bound to one pattern spelling."""

    def __init__(self, name, spelling, pattern, region):
        self.name = name
        self.spelling = spelling
        self.pattern = pattern
        self.region = region
        self.inputs = {}   # pattern input -> the operand node as the step reads it
        self.steps = {}    # step name -> node
        self.dims = {}     # dim name -> int, symbol, or tuple of them for `...`
        self.dtypes = {}   # dtype variable -> dtype

    @property
    def symbolic(self):
        """The dim names bound to a symbol (or a `...` tuple holding one), sorted."""
        return tuple(sorted(k for k, v in self.dims.items() if _symbolic(v)))

    def as_dict(self):
        return {
            "pattern": self.name,
            "spelling": self.spelling,
            "ops": list(self.region.ops),
            "inputs": {k: n.name for k, n in self.inputs.items()},
            "outputs": list(self.pattern.outputs),
            "dims": {k: (list(map(str, v)) if isinstance(v, tuple) else str(v))
                     for k, v in self.dims.items()},
            "symbolic": list(self.symbolic),
            "dtypes": {k: str(v) for k, v in self.dtypes.items()},
        }

    def __repr__(self):
        return "Match(%s, %s)" % (self.name, self.region.ops)


class _Ctx:
    def __init__(self, region, root):
        self.region = region
        self.views = {n for n, cat in region.members if cat == "view"}
        self.work = [n for n, cat in region.members if cat != "view"]
        self.work_set = set(self.work)
        self.names = {n: op_name(n, root) for n in self.work}
        self.names.update({n: op_name(n, root) for n in self.views})

    def through_views(self, n):
        while n in self.views:
            base = n.args[0] if n.args else None
            if not _node(base):
                break
            n = base
        return n

    def chain(self, n):
        """The view chain from an operand back to its root, without node args."""
        out = []
        while n in self.views:
            out.append((self.names[n], tuple(a for a in n.args[1:] if not _node(a)),
                        tuple(sorted((k, v) for k, v in n.kwargs.items() if not _node(v)))))
            base = n.args[0] if n.args else None
            if not _node(base):
                break
            n = base
        return n, out


def _operand(spec, arg, m, ctx, where):
    """Match one operand; raise `_NoMatch` with a reason naming what differs."""
    if isinstance(spec, Opt):
        if arg is None:
            return
        _operand(spec.spec, arg, m, ctx, where)
        return
    if isinstance(spec, str):
        if not _node(arg):
            raise _NoMatch("%s: operand is the constant %r, the pattern wants the tensor %s"
                           % (where, arg, spec))
        root = ctx.through_views(arg)
        if spec in m.steps:
            if root is not m.steps[spec]:
                raise _NoMatch("%s: operand %s is not step %s (%s)"
                               % (where, arg.name, spec, m.steps[spec].name))
            return
        if root in ctx.work_set:
            raise _NoMatch("%s: operand %s comes from %s inside the region, the pattern "
                           "wants its input %s" % (where, arg.name, root.name, spec))
        if spec in m.inputs:
            prev = m.inputs[spec]
            if prev is not arg and ctx.chain(prev) != ctx.chain(arg):
                raise _NoMatch("%s: operand %s is not the same input %s as %s"
                               % (where, arg.name, spec, prev.name))
            return
        m.inputs[spec] = arg
        return
    if isinstance(spec, Half):
        _half(spec, arg, m, ctx, where)
        return
    if isinstance(spec, Seq):
        if not isinstance(arg, (list, tuple)) or len(arg) != len(spec.items):
            raise _NoMatch("%s: operand %r is not a sequence of %d"
                           % (where, arg, len(spec.items)))
        for i, (s, a) in enumerate(zip(spec.items, arg)):
            _operand(s, a, m, ctx, "%s[%d]" % (where, i))
        return
    if _node(arg):
        raise _NoMatch("%s: operand %s is a tensor, the pattern wants a constant"
                       % (where, arg.name))
    if isinstance(spec, Const):
        if not spec.pred(arg):
            raise _NoMatch("%s is %r, the pattern wants %s" % (where, arg, spec.text))
        return
    if isinstance(arg, bool) != isinstance(spec, bool) or not _eq(arg, spec):
        raise _NoMatch("%s is %r, the pattern wants %r" % (where, arg, spec))


def _half(spec, arg, m, ctx, where):
    which = "first" if spec.which == 0 else "second"
    if not _node(arg) or arg not in ctx.views or ctx.names[arg] not in ("getitem", "narrow"):
        raise _NoMatch("%s: operand %s is not a slice of %s (the %s half of the last dim)"
                       % (where, getattr(arg, "name", repr(arg)), spec.ref, which))
    base = arg.args[0]
    meta = _meta(base) if _node(base) else None
    if meta is None:
        raise _NoMatch("%s: no shape meta on %s to check the half" % (where, base))
    rank, dim = len(meta.shape), meta.shape[-1]
    if not isinstance(dim, int) or dim % 2:
        raise _NoMatch("%s: last dim of %s is %s, not a static even size, so no half"
                       % (where, base.name, dim))
    h = dim // 2
    if ctx.names[arg] == "narrow":
        d, start, length = (list(arg.args[1:4]) + [None] * 3)[:3]
        ok = d in (-1, rank - 1) and start == spec.which * h and length == h
    else:
        idx = arg.args[1]
        idx = idx if isinstance(idx, tuple) else (idx,)
        sl, lead = idx[-1], idx[:-1]
        full = slice(None, None, None)
        ok = (isinstance(sl, slice) and sl.step in (None, 1)
              and all(e is Ellipsis or e == full for e in lead)
              and (Ellipsis in lead or len(idx) == rank))
        if ok and spec.which == 0:
            ok = sl.start in (None, 0) and sl.stop == h
        elif ok:
            ok = sl.start == h and (sl.stop is None or (isinstance(sl.stop, int) and sl.stop >= dim))
    if not ok:
        raise _NoMatch("%s: %s is not the %s half of the last dim of %s"
                       % (where, arg.name, which, base.name))
    _operand(spec.ref, base, m, ctx, where)


def _step(step, node, m, ctx):
    where = "%s (%s)" % (step.name, node.name)
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) < len(step.args):
        raise _NoMatch("%s: %d operands, the pattern wants %d" % (where, len(args), len(step.args)))
    extra = args[len(step.args):]
    if len(extra) > len(step.pos_kw):
        raise _NoMatch("%s: %d operands, the pattern allows %d"
                       % (where, len(args), len(step.args) + len(step.pos_kw)))
    for key, value in zip(step.pos_kw, extra):
        if key in kwargs:
            raise _NoMatch("%s: %s given twice" % (where, key))
        kwargs[key] = value
    args = args[:len(step.args)]
    for key in kwargs:
        if key not in step.kwargs:
            raise _NoMatch("%s: keyword %s is not in the pattern" % (where, key))
    for key, spec in step.kwargs.items():
        if key not in kwargs:
            if isinstance(spec, Opt):
                continue
            raise _NoMatch("%s: keyword %s is missing" % (where, key))
        _operand(spec, kwargs[key], m, ctx, "%s keyword %s" % (where, key))
    orders = [list(step.args)]
    if step.commutative and len(step.args) == 2:
        orders.append(list(reversed(step.args)))
    first = None
    for order in orders:
        saved = dict(m.inputs)
        try:
            for i, (spec, arg) in enumerate(zip(order, args)):
                _operand(spec, arg, m, ctx, "%s operand %d" % (where, i))
        except _NoMatch as e:
            m.inputs = saved
            first = first or e
            continue
        m.steps[step.name] = node
        return
    raise first


def _bind_dims(name, spec, meta, m, static):
    shape = tuple(meta.shape)
    dims = spec.dims
    if dims and isinstance(dims[0], str) and dims[0].startswith("..."):
        fixed = len(dims) - 1
        if len(shape) < fixed:
            raise _NoMatch("%s: rank %d, the pattern wants at least %d" % (name, len(shape), fixed))
        pairs = [(dims[0][3:], shape[:len(shape) - fixed])] + list(zip(dims[1:], shape[len(shape) - fixed:]))
    else:
        if len(shape) != len(dims):
            raise _NoMatch("%s: rank %d, the pattern wants %d" % (name, len(shape), len(dims)))
        pairs = list(zip(dims, shape))
    for dim, size in pairs:
        if isinstance(dim, int):
            if not _same(size, dim):
                raise _NoMatch("%s: a dim is %s, the pattern wants %d" % (name, size, dim))
            continue
        if dim in static and _symbolic(size):
            raise _NoMatch("%s: dim %s is symbolic (%s); the pattern needs it static, "
                           "its kernel tiles over it" % (name, dim, size))
        if dim in m.dims:
            if not _same(m.dims[dim], size):
                raise _NoMatch("%s: dim %s is %s, but %s was bound to %s"
                               % (name, dim, size, dim, m.dims[dim]))
        else:
            m.dims[dim] = size


def _try(name, spelling, pattern, region, ctx):
    ops = [ctx.names[n] for n in ctx.work]
    want = [s.op for s in pattern.steps]
    if ops != want:
        raise _NoMatch("op sequence (%s) is not (%s)" % (" ".join(ops), " ".join(want)))
    m = Match(name, spelling, pattern, region)
    for step, node in zip(pattern.steps, ctx.work):
        _step(step, node, m, ctx)
    produced = {m.steps[o] for o in pattern.outputs}
    reached = set()
    for out in region.outputs:
        root = ctx.through_views(out)
        if root not in produced:
            raise _NoMatch("region output %s (from %s) is not a pattern output"
                           % (out.name, root.name))
        reached.add(root)
    for o in pattern.outputs:
        if m.steps[o] not in reached and o not in pattern.optional_outputs:
            raise _NoMatch("pattern output %s (%s) is not read outside the region"
                           % (o, m.steps[o].name))
    for inp, spec in pattern.inputs.items():
        if inp not in m.inputs:
            if inp in pattern.optional:
                continue
            raise _NoMatch("input %s is bound to no operand" % inp)
        node = m.inputs[inp]
        meta = _meta(node)
        if meta is None:
            raise _NoMatch("%s: no shape/dtype meta (example_value, val or tensor_meta) on "
                           "%s; the matcher does not guess" % (inp, node.name))
        _bind_dims(inp, spec, meta, m, pattern.static)
        if spec.dtype is not None:
            allowed = pattern.dtypes[spec.dtype]
            if meta.dtype not in allowed:
                raise _NoMatch("%s: dtype %s is not one of %s" % (
                    inp, meta.dtype, ", ".join(sorted(map(str, allowed)))))
            bound = m.dtypes.setdefault(spec.dtype, meta.dtype)
            if bound != meta.dtype:
                raise _NoMatch("%s: dtype %s differs from %s = %s"
                               % (inp, meta.dtype, spec.dtype, bound))
    if pattern.check is not None:
        why = pattern.check(m)
        if why:
            raise _NoMatch(why)
    return m


def match_region(region, root=None, patterns=None):
    """`(Match, None)` when exactly one pattern name matches, else `(None, reason)`."""
    patterns = PATTERNS if patterns is None else patterns
    ctx = _Ctx(region, root)
    found, reasons, sequence_seen = [], [], False
    for name, spellings in patterns.items():
        for i, pattern in enumerate(spellings):
            if [ctx.names[n] for n in ctx.work] == [s.op for s in pattern.steps]:
                sequence_seen = True
            try:
                found.append(_try(name, i, pattern, region, ctx))
                break  # one spelling per name is enough
            except _NoMatch as e:
                reasons.append("%s[%d]: %s" % (name, i, e))
    names = [m.name for m in found]
    if len(names) == 1:
        return found[0], None
    if len(names) > 1:
        return None, "ambiguous: %s all match; the library is defective" % ", ".join(names)
    if not sequence_seen:
        return None, "op sequence (%s) is no pattern's" % " ".join(ctx.names[n] for n in ctx.work)
    return None, "; ".join(r for r in reasons if "op sequence" not in r)


class FusionReport:
    """Every reported region of a partition, matched or listed."""

    UNMATCHED_RUNS = (
        "unfused: each op runs through `_aten_dispatch` on its own; listed here, not "
        "refused (refuse or run unfused is the owner's open question, FUSION.md §8 Q2)"
    )

    def __init__(self):
        self.matched = []    # [Match]
        self.unmatched = []  # [(Region, reason)]
        self.regions = 0

    def accounted(self):
        return len(self.matched) + len(self.unmatched) == self.regions

    def as_dict(self):
        return {
            "matched": [m.as_dict() for m in self.matched],
            "unmatched": [{"kind": r.kind, "ops": list(r.ops), "reason": why,
                           "runs": self.UNMATCHED_RUNS} for r, why in self.unmatched],
            "regions": self.regions,
            "accounted": self.accounted(),
        }


def match_partition(part, root=None, patterns=None):
    """Match every region of `part` (`regions.partition`'s result).

    Singles (one work op) are not reported: there is nothing to fuse, and they
    run as the op they are.
    """
    rep = FusionReport()
    for region in part.regions:
        rep.regions += 1
        m, why = match_region(region, root, patterns)
        if m is not None:
            rep.matched.append(m)
        else:
            rep.unmatched.append((region, why))
    return rep


def match_graph(graph_or_module, root=None, patterns=None):
    """`partition` then `match_partition`, for a graph or a `GraphModule`."""
    from .regions import partition

    if root is None and hasattr(graph_or_module, "graph"):
        root = graph_or_module
    return match_partition(partition(graph_or_module, root), root, patterns)
