"""Build the `torch._C` *name* surface, at `_C` import time, inside the `.so`.

Read docs/IMPORT_TORCH.md first; this is the file that document describes.

Why this file exists
--------------------
VENDOR.md measured what the vendored Python tree demands of `_C` before
`import torch` finishes: 989 names, 27-32 C submodules that are themselves
packages, a 694-member `TensorBase`, a 985-member `_VariableFunctions`, and
74 enum instances written into the `torch` module rather than into `_C`.
None of that is computation. It is names, types, and metaclasses -- built
dynamically, once, at import.

Doing it here rather than in Rust is a mechanical choice, not an
architectural one: the operations are identical either way (create a heap
type, set an attribute, insert into `sys.modules`), Rust would spell them
through `Bound<'_, PyAny>` at several times the length, and it would still
run at exactly this moment. `_C` remains one native extension module -- this
file is compiled into it with `include_str!` and executed by
`run_bootstrap` in `lib.rs`.

Everything with *behaviour* stays in Rust: dtype, device, TensorBase, and
the aten dispatcher. Nothing here computes. Every callable it creates either
routes into `_C._aten_dispatch` -- the single door DESIGN.md §6 depends on --
or raises `NotImplementedError` naming itself.

The three rules
---------------
1. **No module-level `__getattr__` on `_C`.** The vendored tree turns whole
   subsystems off by asking `hasattr(torch._C, "_c10d_init")`
   (VENDOR.md wall 11). A catch-all would answer yes to every one of those
   and switch on distributed, RPC, CUDA, XPU, MTIA and MPS at once. A name is
   present here only if the surface table has it and the tree does not merely
   *probe* for it.

2. **Types and submodules do get a catch-all**, because the stubs are
   incomplete for them (`_special`, `_fft` and `_linalg` have no stub at all)
   and because nothing gates on a *member* of a `_C` type the way it gates on
   a name in `_C`. The catch-all hands back an object that raises when used.

3. **Anything not implemented names itself.** DESIGN.md §6 -- the shim is its
   own instrument, so running the tree produces the work queue.

Where the names come from
-------------------------
`vendor/gen_surface.py`, from `vendor/torch/_C/*.pyi` -- the vendored tree's
own stubs. Not from an installed upstream `_C.so`. The distinction matters:
the stubs are the tree's statement of what it expects, they ship under the
same BSD licence as the rest of the vendored tree, and using them keeps the
build from needing real torch present.
"""

from __future__ import annotations

import builtins
import enum
import importlib.util
import json
import sys
import types

# ---------------------------------------------------------------------------
# Deliberate omissions
# ---------------------------------------------------------------------------
#
# VENDOR.md wall 11, generalised. Each of these is a name the vendored tree
# probes with `hasattr(torch._C, ...)` in order to decide whether a subsystem
# exists. Leaving the name out is the off-switch upstream itself provides, and
# it is the only one it provides -- there is no module-level flag. The list was
# produced by grepping the vendored tree for those probes, so every entry is a
# switch that is really wired, not a guess about what might be optional.
#
# This is not the same as "not implemented". A name that is merely unbuilt
# still appears, and raises when used. A name in this set must *not* appear,
# because its absence is the answer to a question the tree asks.
# `vendor/gen_surface.py` derives the real list by scanning the tree for
# `hasattr(torch._C, "...")` and `getattr(torch._C, "...", ...)`; it arrives in
# the surface under "probes", and `install` reads it from there. Deriving it
# beats hand-listing: the hand-written version this replaced was already
# missing `_cuda_isInBadFork`, `_mps_is_in_bad_fork` and `_xpu_isInBadFork`.
#
# These are the ones a scan cannot see.
EXTRA_OFF_SWITCHES = frozenset(
    {
        # `torch/cuda/streams.py:11` and `torch/xpu/streams.py:11` guard on the
        # `*StreamBase` sibling and then use the event type unconditionally
        # inside that guard, so the event type has to go when the stream does.
        "_CudaEventBase",
        "_XpuEventBase",
    }
)

# Dunders `TensorBase` must *not* be given, even though the stub declares them.
#
# The rest it must: `torch/_tensor.py:1115` is `__itruediv__ =
# _C.TensorBase.__idiv__`, so the operator set is demanded by name in the class
# body, the same way the ordinary methods are. But a raising stand-in for any
# of these breaks the object rather than making it loud --
#
#   `__getattribute__`/`__setattr__`/`__getattr__`  intercept everything
#   `__del__`                                       raises during collection
#   `__init__`/`__new__`                            stop construction
#   `__hash__`/`__eq__`                             break dict and set use
#   `__repr__`/`__str__`/`__format__`/`__dir__`     break every traceback
#   `__reduce__`/`__reduce_ex__`/`__sizeof__`       break copy and pickle
#
# -- and several of them are consulted by the interpreter itself, where an
# exception has nowhere to go. They are left absent so the failure, if one
# comes, is an `AttributeError` naming the member rather than a `TensorBase`
# that cannot be printed.
UNSAFE_DUNDERS = frozenset(
    {
        "__init__",
        "__new__",
        "__init_subclass__",
        "__subclasshook__",
        "__class__",
        "__getattribute__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
        "__del__",
        "__dir__",
        "__doc__",
        "__module__",
        "__dict__",
        "__slots__",
        "__repr__",
        "__str__",
        "__format__",
        "__hash__",
        "__eq__",
        "__ne__",
        "__sizeof__",
        "__reduce__",
        "__reduce_ex__",
    }
)

_BUILTIN_EXCEPTION_BASES = {
    name
    for name in dir(builtins)
    if isinstance(getattr(builtins, name), type)
    and issubclass(getattr(builtins, name), BaseException)
}


class _Unimplemented:
    """A name that exists and refuses.

    DESIGN.md §6: the message is the work queue, so it always carries the
    fully-qualified name. It is deliberately *not* a chameleon -- the record
    mode of `vendor/probe.py` uses one of those, which is right for an
    instrument and wrong here, because a placeholder that answers every
    question quietly produces wrong behaviour instead of a work item.

    `__bool__` is part of that, and it was missing. Without it every instance
    is truthy, so `if torch._C._has_cudnn:` answered *yes* -- the class doing
    precisely what its own docstring says it must not. One site was measured to
    reach that branch on the `import torch` path
    (`torch/backends/cudnn/__init__.py:231`, a class body: `benchmark_limit`
    became a pair of placeholders where upstream without cuDNN leaves it
    `None`). It raises now.

    Note which question this refuses. Names the stubs declare as functions come
    from `_make_function` and are ordinary Python functions -- truthy, as
    upstream's builtins are. An `_Unimplemented` is what is left when the stubs
    say *nothing* about a name, so "is it there?" is a question the shim has no
    grounds to answer either way, and saying so is the honest reply.
    """

    __slots__ = ("_qualname", "__dict__")

    def __init__(self, qualname: str) -> None:
        self._qualname = qualname

    def __call__(self, *args, **kwargs):
        raise NotImplementedError(
            f"not implemented in torch._C shim: {self._qualname}"
        )

    def __bool__(self):
        raise NotImplementedError(
            "torch._C shim cannot answer whether this exists: "
            f"{self._qualname}. It is a placeholder, and a truth test on one "
            "used to silently mean yes."
        )

    def __repr__(self) -> str:
        return f"<torch._C shim: {self._qualname} (not implemented)>"


def _make_function(qualname: str, name: str):
    """A real Python function, not a `_Unimplemented`.

    Two places need the difference. `torch/__init__.py:2212` assigns
    `__module__` on every harvested `_VariableFunctions` member, and
    `_C._add_docstr` assigns `__doc__` on hundreds of `TensorBase` members --
    both are writable on a function and neither is on a builtin.
    """

    def fn(*args, **kwargs):
        raise NotImplementedError(f"not implemented in torch._C shim: {qualname}")

    fn.__name__ = name
    fn.__qualname__ = qualname
    return fn


def _make_property(qualname: str):
    """`TensorBase.is_sparse` and friends.

    `torch/_prims_common/__init__.py:90` pulls the *descriptor* out of the
    class (`torch.Tensor.is_sparse.__get__`), so these cannot be plain values
    (VENDOR.md wall 10).
    """

    def getter(self):
        raise NotImplementedError(f"not implemented in torch._C shim: {qualname}")

    return property(getter)


class _ShimMeta(type):
    """Metatype for synthesised `_C` types.

    Two unrelated jobs, both measured in VENDOR.md.

    *Members.* `torch/_torch_docs.py` reaches into `_C` types at import time
    (`_add_docstr(torch.Stream.query, ...)`), and the stubs do not list every
    member. Unknown ones are answered here rather than left to fail, because a
    missing member of a type is a hole, not a switch (rule 2 in the module
    docstring).

    *Identity.* `torch/_awaits/__init__.py:12` builds
    `class _PyAwaitMeta(type(torch._C._Await), type(Generic))`, which fails
    with `duplicate base class` if `type(_Await)` is plain `type`
    (VENDOR.md wall 12). Upstream's answer is pybind11's metatype; this is
    ours. The mirror case -- `torch/autograd/variable.py:14` needs
    `type(_C._LegacyVariableBase)` to *be* `type` -- is handled by
    `PLAIN_TYPE_METACLASS` below.
    """

    def __getattr__(cls, name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        value = _Unimplemented(f"{cls.__module__}.{cls.__name__}.{name}")
        setattr(cls, name, value)
        return value


# Types whose metatype must be exactly `type`. `torch/autograd/variable.py:14`
# is `class Variable(_C._LegacyVariableBase, metaclass=VariableMeta)` where
# `VariableMeta(type)`; a richer metatype on the base is a metaclass conflict.
# and `torch/autograd/function.py:365` is `class _SingleLevelFunction(
# _C._FunctionBase, ..., metaclass=FunctionMeta)` with `FunctionMeta(type)`.
#
# VENDOR.md wall 12 measured upstream's distribution -- 135 `pybind11_type`,
# 51 `type`, 2 `_TensorMeta`, 1 `OpaqueBaseMeta` -- and concluded that no
# blanket rule works because the tree depends on the difference in *both*
# directions. This is the other half of that: the default is `_ShimMeta`
# (needed by `_Await`, which must not be plain `type`), and the exceptions are
# found by running the tree, because a metaclass conflict is only visible when
# the class statement executes.
PLAIN_TYPE_METACLASS = frozenset({"_LegacyVariableBase", "_FunctionBase"})

# Submodules the stubs do not declare but the tree imports anyway. Upstream
# registers 32 from C; the stub directory covers 27. The rest are found by
# running the tree, which is the only way they can be found -- see
# VENDOR.md §6 item 2 on requirements that no amount of reading reveals.
EXTRA_SUBMODULES = (
    "_special",
    "_fft",
    "_linalg",
    "_sparse",
    "_nested",
    "_nn_functional",
    "_VariableFunctions",
)


class _SubmoduleFinder:
    """Serves `torch._C.<known>` and anything nested under it.

    `from torch._C._jit_tree_views import SourceRangeFactory` is an *import
    statement*, so an attribute on `_C` does not satisfy it -- `_C` is a
    compiled module and has no `__path__` for the machinery to search
    (VENDOR.md wall 8). Upstream registers its submodules into `sys.modules`
    from C; this does the same, and adds a finder for the nested case
    (`torch/utils/_python_dispatch.py:22` imports `torch._C._dynamo.guards`,
    so the submodules are themselves packages).

    The finder answers only for names below a submodule it already created.
    An unknown `torch._C.<name>` still fails, so `try: import ... except
    ImportError` stays a working question.
    """

    def __init__(self, prefix: str, roots: set[str]) -> None:
        self._prefix = prefix + "."
        self._depth = prefix.count(".") + 1
        self._roots = roots

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(self._prefix):
            return None
        head = fullname[len(self._prefix) :].split(".")[0]
        if head not in self._roots:
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        module = types.ModuleType(spec.name)
        module.__path__ = []
        _attach_module_catchall(module)
        return module

    def exec_module(self, module):
        return None


def _attach_module_catchall(module) -> None:
    """PEP 562 `__getattr__`, with a cache so identity holds.

    `torch/_ops.py:132` does `isinstance(k, TransformType)` where both the
    value and the class came out of `torch._C._functorch`. Handing out a fresh
    object per lookup breaks that for a reason that has nothing to do with
    torch, so every synthesised member is remembered.
    """
    name = module.__name__

    def __getattr__(attr):
        if attr.startswith("__") and attr.endswith("__"):
            raise AttributeError(attr)
        qualname = f"{name}.{attr}"
        if attr[:1].isupper() or attr.lstrip("_")[:1].isupper():
            value = _ShimMeta(attr, (), {"__module__": name, "__init__": _permissive_init})
        else:
            value = _Unimplemented(qualname)
        setattr(module, attr, value)
        return value

    module.__getattr__ = __getattr__


def _permissive_init(self, *args, **kwargs):
    """Synthesised `_C` types have to be *constructible*, with arguments.

    `torch/_sources.py:87` is `class SourceContext(SourceRangeFactory)` and its
    `__init__` calls `super().__init__(source, filename, file_lineno,
    leading_whitespace_len)` -- while `torch/nn/functional.py` is still
    importing, from inside the TorchScript source parser that `@_overload` runs
    at decoration time (VENDOR.md wall 13). So existing is not enough.
    """
    return None


def _build_type(name, spec, module_name, resolved):
    """One synthesised `_C` type."""
    bases = []
    for base in spec.get("bases", ()):
        if base in resolved:
            bases.append(resolved[base])
        elif base in _BUILTIN_EXCEPTION_BASES:
            bases.append(getattr(builtins, base))
        # `Tensor`, `Protocol`, `Generic[...]` and the like are dropped: they
        # are annotation-only bases in the stub, and `Tensor` in particular is
        # `torch.Tensor`, which does not exist while `_C` is being imported.

    if "Enum" in spec.get("bases", ()) or "IntEnum" in spec.get("bases", ()):
        # A real `enum.Enum`, because the tree type-checks against these:
        # `torch/_ops.py:139` asserts `isinstance(k, DispatchKey)`
        # (VENDOR.md wall 14). A stand-in class with class attributes would
        # pass the name lookup and fail the assertion.
        members = [m for m in spec.get("attrs", ()) if not m.startswith("__")]
        base = enum.IntEnum if "IntEnum" in spec.get("bases", ()) else enum.Enum
        cls = base(name, members, module=module_name)
        return cls

    body = {
        "__module__": module_name,
        "__init__": _permissive_init,
        "__doc__": f"torch._C shim placeholder for {module_name}.{name}",
    }
    for member in spec.get("methods", ()):
        if member.startswith("__") and member.endswith("__"):
            continue
        body[member] = _make_function(f"{name}.{member}", member)

    if "get" in spec.get("methods", ()) and "JitType" in spec.get("bases", ()):
        # The 14 TorchScript type singletons -- `_C.IntType.get()`,
        # `_C.TensorType.get()` and so on. `torch/_higher_order_ops/schema.py:56`
        # builds a dict of them at import, so `get` must return an object, and
        # the same object each time: the tree uses them as dict *keys* and as
        # identity comparisons. A fresh instance per call would be a subtle,
        # silent wrong answer rather than a loud one.
        body["get"] = classmethod(_singleton_getter())
    for member in spec.get("attrs", ()):
        if member.startswith("__") and member.endswith("__"):
            continue
        body.setdefault(member, _make_property(f"{name}.{member}"))

    meta = type if name in PLAIN_TYPE_METACLASS else _ShimMeta
    if bases:
        # An exception type must really derive from the exception it claims to
        # (`torch/distributed/__init__.py:41` catches `_C._DistError`), and a
        # metatype cannot be imposed on a base that has a stricter one.
        meta = type(bases[0]) if isinstance(bases[0], type) and issubclass(
            bases[0], BaseException
        ) else meta
    return meta(name, tuple(bases), body)


def _singleton_getter():
    cache: dict = {}

    def get(cls):
        if cls not in cache:
            cache[cls] = cls()
        return cache[cls]

    return get


def _order_types(types_spec):
    """Bases before subclasses. 26 `_C` types derive from `JitType`."""
    remaining = dict(types_spec)
    ordered = []
    made = set()
    while remaining:
        progressed = False
        for name in list(remaining):
            spec = remaining[name]
            deps = [b for b in spec.get("bases", ()) if b in types_spec]
            if all(d in made for d in deps):
                ordered.append((name, spec))
                made.add(name)
                del remaining[name]
                progressed = True
        if not progressed:
            # A base cycle in the stub. Emit the rest without their bases
            # rather than looping; if this ever fires it is a real finding.
            for name, spec in remaining.items():
                spec = dict(spec, bases=[])
                ordered.append((name, spec))
            break
    return ordered


# ---------------------------------------------------------------------------
# The op registry -- `torch.ops.aten.<op>.<overload>`
# ---------------------------------------------------------------------------


class _SchemaType:
    """A type inside a schema, kept as its source spelling.

    Not comparable to the TorchScript type singletons (`_C.TensorType.get()`
    and friends): `torch/_library/utils.py:163` decides "is this a tensor
    argument" by comparing against those objects, and answering `True` from a
    string match would be claiming a correspondence the shim has not built.
    Answering `False` makes the callers take their conservative branch, which
    is the safe direction.
    """

    __slots__ = ("_spelling",)

    def __init__(self, spelling: str) -> None:
        self._spelling = spelling

    def __str__(self) -> str:
        return self._spelling

    def __repr__(self) -> str:
        return self._spelling

    def __eq__(self, other) -> bool:
        return isinstance(other, _SchemaType) and self._spelling == other._spelling

    def __hash__(self) -> int:
        return hash(self._spelling)


class _AliasInfo:
    __slots__ = ("is_write", "before_set", "after_set")

    def __init__(self, is_write: bool, symbols) -> None:
        self.is_write = is_write
        self.before_set = set(symbols)
        self.after_set = set(symbols)


class _Argument:
    __slots__ = ("name", "type", "kwarg_only", "default_value", "alias_info", "N")

    def __init__(self, name, typ, kwarg_only, default_value, alias_info, N=None):
        self.name = name
        self.type = typ
        self.kwarg_only = kwarg_only
        self.default_value = default_value
        self.alias_info = alias_info
        self.N = N

    def has_default_value(self):
        return self.default_value is not None

    def __repr__(self):
        return f"{self.type} {self.name}"


def _split_top_level(text: str) -> list:
    """Split on commas that are not inside (), [] or ''."""
    out, depth, quote, start = [], 0, "", 0
    for i, ch in enumerate(text):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "\"'":
            quote = ch
        elif ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "," and depth == 0:
            out.append(text[start:i])
            start = i + 1
    tail = text[start:]
    if tail.strip():
        out.append(tail)
    return [chunk.strip() for chunk in out if chunk.strip()]


class _Schema:
    """`torch._C.FunctionSchema`, parsed rather than stubbed.

    `torch._C.parse_schema` is called at import from seven places
    (`torch/library.py:80` and `:151`, `torch/_prims/__init__.py:322`,
    `torch/_library/custom_ops.py:742`, ...), and what those callers do with
    the result -- count arguments, look for `kwarg_only` tensors, look for
    `alias_info.is_write` -- needs the schema to have really been read. A stub
    with an empty argument list would answer "no mutable arguments" to every
    question, which is a wrong answer rather than a missing one.

    The grammar handled is torch's own:
    `ns::name.overload(Type(alias) name=default, *, ...) -> (R1, R2)`.
    """

    __slots__ = ("name", "overload_name", "arguments", "returns", "_source")

    def __init__(self, qualname: str, overload: str, source: str = "") -> None:
        self.name = qualname
        self.overload_name = overload
        self.arguments = []
        self.returns = []
        self._source = source

    @classmethod
    def parse(cls, text: str) -> "_Schema":
        text = text.strip()
        open_paren = text.index("(")
        head = text[:open_paren].strip()
        # The *matching* close paren, found by depth. `rindex(")")` looked
        # right and was wrong: in
        # `aten::add_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)`
        # the last `)` belongs to the return type's alias annotation, so the
        # argument list swallowed the arrow and produced two arguments, the
        # second of them named `Tensor(a!`.
        depth = 0
        close_paren = -1
        for i in range(open_paren, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    close_paren = i
                    break
        if close_paren < 0:
            raise RuntimeError(f"torch._C shim: unbalanced schema: {text}")
        body = text[open_paren + 1 : close_paren]
        tail = text[close_paren + 1 :].strip()

        namespace, _, rest = head.rpartition("::")
        name, _, overload = rest.partition(".")
        qualname = f"{namespace}::{name}" if namespace else name

        schema = cls(qualname, overload, text)
        kwarg_only = False
        for chunk in _split_top_level(body):
            if chunk == "*":
                kwarg_only = True
                continue
            schema.arguments.append(_parse_argument(chunk, kwarg_only))

        if tail.startswith("->"):
            returns = tail[2:].strip()
            if returns.startswith("(") and returns.endswith(")"):
                returns = returns[1:-1]
            for chunk in _split_top_level(returns):
                schema.returns.append(_parse_argument(chunk, False))
        return schema

    def is_mutable(self) -> bool:
        return any(
            a.alias_info is not None and a.alias_info.is_write for a in self.arguments
        )

    def _is_view_op(self) -> bool:
        """A view op aliases an argument into a return without writing to it.

        `torch/_library/custom_ops.py:794` asks this during
        `@torch.library.custom_op` registration, which runs at import.
        Replicated from `MathBitsFallback.h`, the same rule `OpOverload`
        applies to its own schema (`torch/_ops.py:838`).
        """
        writes = [
            a.alias_info.is_write for a in self.arguments if a.alias_info is not None
        ]
        if not writes:
            return False
        return not any(writes) and any(r.alias_info is not None for r in self.returns)

    def __str__(self) -> str:
        if self._source:
            return self._source
        suffix = f".{self.overload_name}" if self.overload_name else ""
        return f"{self.name}{suffix}(...) -> ..."

    def __repr__(self) -> str:
        return str(self)


def _parse_argument(chunk: str, kwarg_only: bool) -> _Argument:
    default = None
    depth = 0
    for i, ch in enumerate(chunk):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "=" and depth == 0:
            default = chunk[i + 1 :].strip()
            chunk = chunk[:i].strip()
            break

    spelling, _, name = chunk.rpartition(" ")
    if not spelling:
        spelling, name = chunk, ""

    alias_info = None
    if "(" in spelling and spelling.endswith(")"):
        inner = spelling[spelling.index("(") + 1 : -1]
        # `Tensor(a!)` mutates; `Tensor(a)` only aliases.
        alias_info = _AliasInfo("!" in inner, inner.replace("!", "").split("|"))
    return _Argument(name.strip(), _SchemaType(spelling.strip()), kwarg_only, default,
                     alias_info)


# aten really does have dunder-named operators, and the tree registers
# decompositions for all ten at import (`torch/_decomp/decompositions.py:6239`
# is `register_inplace(aten.__iand__, aten.__and__)`). Everything else spelled
# with dunders that reaches the op registry is Python asking a question about
# the *namespace object* -- `__origin__`, `__deepcopy__`, `__wrapped__` -- and
# must be refused so that `getattr(..., default)` and `copy` keep working.
# The list is the tree's own: `grep -o 'aten\.__[a-z_]*__'`.
ATEN_DUNDER_OPS = frozenset(
    {
        "__and__",
        "__iand__",
        "__ilshift__",
        "__ior__",
        "__irshift__",
        "__ixor__",
        "__lshift__",
        "__or__",
        "__rshift__",
        "__xor__",
    }
)


def _is_refused_op_name(name: str) -> bool:
    return (
        name.startswith("__")
        and name.endswith("__")
        and name not in ATEN_DUNDER_OPS
    )


def _op_callable(dispatch, qualname: str, overload: str):
    """The one door, wearing the shape `torch/_ops.py` expects.

    `aten::add` + `Tensor` becomes the dispatch key `aten.add.Tensor`, which is
    exactly what `_C._aten_dispatch` takes and exactly the spelling
    `_C._aten_implemented()` reports. Overload is part of the key on purpose
    (docs/TORCH_C.md §1): folding `add.Tensor` and `add.Scalar` together would
    make one implementation look like two.
    """
    namespace, _, name = qualname.partition("::")
    key = f"{namespace}.{name}.{overload or 'default'}"

    def op(*args, **kwargs):
        return dispatch(key, *args, **kwargs)

    op.__name__ = f"{name}.{overload or 'default'}"
    op.__qualname__ = op.__name__
    op._shim_aten_key = key
    return op


# ---------------------------------------------------------------------------
# Overload resolution -- `torch.<op>(...)` -> `aten.<op>.<overload>`
# ---------------------------------------------------------------------------
#
# `torch.full` is not an aten op. Upstream it is a C binding that reads the
# actual arguments, picks one of several native functions, and calls it; the
# aten overload name is a property of the *native function it picked*, not of
# the name the caller typed. So a shim that stops at `torch.ops.aten.<op>.<ov>`
# leaves the whole user-facing API dark, which is where this project was:
#
#     torch.ops.aten.full.default([2], True).dtype   ->  torch.bool
#     torch.full((2,), True)                         ->  NotImplementedError
#
# and the vendored tree plus transformers call the second spelling.
#
# What is reproduced here is `PythonArgParser::raw_parse`: a signature list per
# name, tried in order, first one that binds wins. That is deliberately the
# same algorithm rather than a lookup table of special cases, because the
# ordering *is* the semantics -- see the note in `overloads.json` about
# `arange.start` versus `arange.start_step`.
#
# The single door is untouched. Resolution decides *which key* to hand
# `_C._aten_dispatch`; it never computes, never reaches a kernel by another
# route, and an op it resolves to but that has no kernel raises the same
# `aten op not implemented in torch._C shim: <key>` as before. A name with no
# table entry keeps the old refusal. So the instrument gains resolution without
# gaining a second entrance.

_SCHEMA_BASE_TYPES = frozenset(
    {
        "Tensor",
        "Scalar",
        "SymInt",
        "int",
        "float",
        "bool",
        "str",
        "ScalarType",
        "Layout",
        "Device",
        "MemoryFormat",
        "Generator",
    }
)


def _decompose_type(spelling: str):
    """`SymInt[]?` -> `("SymInt", True, True, 0)`, `Tensor(a!)` -> `("Tensor", False, False, None)`.

    Returns `(base, is_list, is_optional, list_size)`. The order of the strips
    matters: `?` binds outermost (`int[]?` is an optional list, not a list of
    optional ints), and an alias annotation is attached to the base.

    `list_size` is the number inside the brackets -- `int[1]` gives `1`, a bare
    `int[]` gives `0`, and a non-list gives `None`. It is not decoration:
    `FunctionParameter::check` accepts a *bare int* wherever the declared list
    has a size, which is the whole reason `x.sum(0)` binds
    `sum.dim_IntList(Tensor self, int[1]? dim, ...)`. Dropping the number and
    demanding a real sequence would make that call report "no matching
    overload" -- a wrong answer in the shape of a right one.
    """
    text = spelling.strip()
    optional = text.endswith("?")
    if optional:
        text = text[:-1].strip()
    is_list = False
    list_size = None
    if text.endswith("]"):
        is_list = True
        open_bracket = text.rindex("[")
        inside = text[open_bracket + 1 : -1].strip()
        list_size = int(inside) if inside.isdigit() else 0
        text = text[:open_bracket].strip()
    # `Tensor(a!)`, `Tensor(a)` -- the alias annotation is `_AliasInfo`'s
    # business (it is what `is_mutable()` reads); for type checking it is not
    # part of the type.
    if text.endswith(")") and "(" in text:
        text = text[: text.index("(")].strip()
    return text, is_list, optional, list_size


class _TypeChecker:
    """"Does this Python value satisfy this schema type?"

    `FunctionParameter::check` in `python_arg_parser.cpp`, restricted to the
    spellings the table contains -- `install` refuses to start if the table
    grows one this does not know, so the restriction is enforced rather than
    assumed.

    Three of these rules are torch's and would be got wrong by intuition:

      * `bool` does **not** satisfy `int`. `bool` subclasses `int` in Python,
        but torch's `THPUtils_checkLong` excludes it explicitly, and it has to
        -- `torch.full((2,), True)` must reach `Scalar` as a bool so the result
        is `torch.bool` and not `int64`.
      * `int` **does** satisfy `float`, one way only.
      * a zero-dim tensor satisfies `Scalar`. torch's `DOUBLE`/`SCALAR` case
        falls through to accepting a 0-dim `THPVariable`, which is why
        `torch.pow(t, other_0d_tensor)` picks `pow.Tensor_Tensor` -- declared
        first -- rather than `pow.Tensor_Scalar`.
    """

    def __init__(self, module) -> None:
        self._module = module
        self._tensor = module.TensorBase
        self._dtype = module.dtype
        self._layout = getattr(module, "layout", None)
        self._memory_format = getattr(module, "memory_format", None)
        self._device = module.device
        self._generator = getattr(module, "Generator", None)

    def check(self, spelling, value) -> bool:
        base, is_list, optional, list_size = _decompose_type(str(spelling))
        if optional and value is None:
            return True
        if is_list:
            if isinstance(value, (list, tuple)):
                return all(self._base(base, item) for item in value)
            # torch: "if a size is specified (e.g. IntArrayRef[2]) we also
            # allow passing a single int". `x.sum(0)` is that rule.
            return (
                list_size is not None
                and list_size > 0
                and base in ("int", "SymInt")
                and isinstance(value, int)
                and not isinstance(value, bool)
            )
        return self._base(base, value)

    @staticmethod
    def coerce(spelling, value):
        """The value to actually bind, once `check` has said yes.

        Only one rule: a bare int that satisfied a sized int list is widened to
        a one-element tuple, so the kernel behind the key always sees a list
        where the schema says list. torch does the same thing one layer down
        (`IntArrayRef` of length one).
        """
        base, is_list, _, list_size = _decompose_type(str(spelling))
        if (
            is_list
            and list_size
            and base in ("int", "SymInt")
            and isinstance(value, int)
            and not isinstance(value, bool)
        ):
            return (value,)
        return value

    def _base(self, base: str, value) -> bool:
        if base == "Tensor":
            return isinstance(value, self._tensor)
        if base == "Scalar":
            if isinstance(value, (bool, int, float, complex)):
                return True
            return isinstance(value, self._tensor) and value.dim() == 0
        if base in ("int", "SymInt"):
            return isinstance(value, int) and not isinstance(value, bool)
        if base == "float":
            return isinstance(value, (int, float)) and not isinstance(value, bool)
        if base == "bool":
            return isinstance(value, bool)
        if base == "str":
            return isinstance(value, str)
        if base == "ScalarType":
            return isinstance(value, self._dtype)
        if base == "Layout":
            return self._layout is not None and isinstance(value, self._layout)
        if base == "MemoryFormat":
            return self._memory_format is not None and isinstance(
                value, self._memory_format
            )
        if base == "Device":
            # torch takes a bare string wherever a device is taken, and
            # `device_arg` in aten.rs already accepts one.
            return isinstance(value, (self._device, str))
        if base == "Generator":
            return self._generator is not None and isinstance(value, self._generator)
        raise RuntimeError(f"torch._C shim: unhandled schema type: {base}")


def _is_schema_default(value, default_source) -> bool:
    """Is this the value the schema would have used anyway?

    Arguments equal to their own default carry no information, and dropping
    them is what keeps `torch.ones(2, pin_memory=False)` from arriving at a
    kernel that refuses every non-`None` `pin_memory`. It also means a kernel
    only ever sees arguments a caller actually asked for.

    Comparison is restricted to literal defaults against scalar values on
    purpose: `==` against a `TensorBase` would go through a synthesised
    `__eq__` that raises.
    """
    if default_source is None:
        return False
    if default_source == "None":
        return value is None
    if default_source == "True":
        return value is True
    if default_source == "False":
        return value is False
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    for parse in (int, float):
        try:
            return value == parse(default_source)
        except (TypeError, ValueError):
            continue
    return False


class _Overloads:
    """The ordered signature list for one `torch.<name>` or one `Tensor.<name>`.

    `self_bound` is the only difference between the two. A method's receiver is
    already chosen before any signature is considered -- upstream binds it
    outside `PythonArgParser` entirely -- so it is passed in as `args[0]` and
    the parts of the algorithm that count positional arguments skip over it.
    Concretely that is the varargs int-list rule: `x.view(2, 3)` has to mean
    `view([2, 3])`, and the precondition for that rule is "the signature has
    exactly one positional argument", which is only true of `view` once `self`
    is out of the count.
    """

    __slots__ = ("name", "schemas", "keys", "self_bound", "_checker_source")

    def __init__(self, name: str, schemas, checker_source, self_bound: bool = False) -> None:
        self.name = name
        self.self_bound = self_bound
        self.schemas = [_Schema.parse(text) for text in schemas]
        # A callable rather than a `_TypeChecker`: `layout` and `memory_format`
        # are synthesised later in `install` than this table is parsed, and the
        # first *call* is long after both. Parsing and validating the table
        # still happens now -- see the `_SCHEMA_BASE_TYPES` check below.
        self._checker_source = checker_source
        self.keys = []
        for schema in self.schemas:
            namespace, _, op = schema.name.partition("::")
            self.keys.append(f"{namespace}.{op}.{schema.overload_name or 'default'}")
            if self_bound and not schema.arguments:
                raise RuntimeError(
                    f"torch._C shim: methods.json entry {name!r} has a schema with "
                    f"no `self` to bind the receiver into: {schema}"
                )
            for argument in schema.arguments:
                base, _, _, _ = _decompose_type(str(argument.type))
                if base not in _SCHEMA_BASE_TYPES:
                    # At install time, not at call time. A spelling nobody
                    # taught `_TypeChecker` would otherwise make every call to
                    # that overload silently fail to match, which reads as "no
                    # overload matched" -- a wrong answer wearing the shape of
                    # a right one.
                    raise RuntimeError(
                        f"torch._C shim: overloads.json entry {name!r} uses schema "
                        f"type {base!r}, which _TypeChecker does not handle: "
                        f"{schema}"
                    )

    def _bind(self, schema, args, kwargs):
        """torch's `FunctionSignature::parse`, minus the parts nothing uses.

        Returns the bound arguments, or `None` if this schema does not accept
        the call.
        """
        checker = self._checker_source()
        positional = [a for a in schema.arguments if not a.kwarg_only]
        by_name = {a.name: a for a in schema.arguments}
        # `self` is bound before any signature is looked at, so it is not part
        # of what the parser counts (see the class docstring).
        skip = 1 if self.self_bound else 0

        # The varargs int-list rule, with torch's exact precondition: it
        # applies only when the signature has a *single* positional argument
        # and that argument is an int list, which is what makes
        # `torch.ones(2, 3)` mean `ones([2, 3])` while `torch.full(2, 3)` stays
        # an error rather than becoming `full([2], 3)`.
        if len(positional) - skip == 1 and len(args) > skip:
            base, is_list, _, _ = _decompose_type(str(positional[skip].type))
            if (
                is_list
                and base in ("int", "SymInt")
                and isinstance(args[skip], int)
                and not isinstance(args[skip], bool)
            ):
                args = tuple(args[:skip]) + (tuple(args[skip:]),)

        if len(args) > len(positional):
            return None

        bound = {}
        for value, parameter in zip(args, positional):
            if parameter.name in kwargs:
                return None  # given twice
            if not checker.check(parameter.type, value):
                return None
            bound[parameter.name] = checker.coerce(parameter.type, value)

        for key, value in kwargs.items():
            parameter = by_name.get(key)
            if parameter is None or key in bound:
                return None
            if not checker.check(parameter.type, value):
                return None
            bound[key] = checker.coerce(parameter.type, value)

        for parameter in schema.arguments:
            if parameter.name not in bound and not parameter.has_default_value():
                return None

        return {
            name: value
            for name, value in bound.items()
            if not _is_schema_default(value, by_name[name].default_value)
        }

    def resolve(self, args, kwargs):
        for schema, key in zip(self.schemas, self.keys):
            bound = self._bind(schema, args, kwargs)
            if bound is not None:
                return key, bound
        owner = "Tensor." if self.self_bound else "torch."
        shown = args[1:] if self.self_bound else args
        raise TypeError(
            f"{owner}{self.name}(): no matching overload in torch._C shim for "
            f"({_describe_call(shown, kwargs)}). Candidates tried, in order:\n"
            + "\n".join(f"  {schema}" for schema in self.schemas)
        )


def _describe_call(args, kwargs) -> str:
    parts = [type(a).__name__ for a in args]
    parts += [f"{k}={type(v).__name__}" for k, v in kwargs.items()]
    return ", ".join(parts)


# Python-level keyword arguments torch's factory functions accept that no aten
# schema mentions. They are the autograd knob, and this shim has no autograd
# (DESIGN.md §3 stage 0) -- so `requires_grad=False` is dropped and
# `requires_grad=True` is refused by name rather than ignored, which would hand
# back a tensor that silently does not record anything.
def _strip_python_only_kwargs(name: str, kwargs: dict) -> dict:
    kwargs = dict(kwargs)
    if kwargs.pop("requires_grad", False):
        raise NotImplementedError(
            f"not implemented in torch._C shim: torch.{name}(requires_grad=True) "
            f"-- there is no autograd behind this shim, and returning a tensor "
            f"that quietly records nothing would be worse than refusing"
        )
    # An explicit `out=None` is how the vendored tree spells "no out tensor".
    # Left in place it would fail to bind the `.out` overload (`out` is not
    # optional there) *and* fail to bind the plain one (`out` is not an
    # argument of it), so the call would report no matching overload.
    if "out" in kwargs and kwargs["out"] is None:
        del kwargs["out"]
    return kwargs


def install(module, surface_json: str, overloads_json: str, methods_json: str) -> None:
    surface = json.loads(surface_json)
    dispatch = module._aten_dispatch
    real = set(vars(module))
    off = frozenset(surface.get("probes", ())) | EXTRA_OFF_SWITCHES
    # Readable from Python so the decision can be inspected rather than
    # reverse-engineered from what is absent.
    module._shim_off_switches = sorted(off)

    # -- overload resolution ----------------------------------------------
    #
    # Parsed and validated now, so a malformed table stops `import _C` at the
    # line that is wrong. The `_TypeChecker` it will use is built on first
    # call, because `layout` and `memory_format` do not exist yet.
    checker_cell = []

    def _checker_source():
        if not checker_cell:
            checker_cell.append(_TypeChecker(module))
        return checker_cell[0]

    overloads = {
        name: _Overloads(name, schemas, _checker_source)
        for name, schemas in json.loads(overloads_json).items()
        if not name.startswith("_")  # the table's embedded README
    }
    # Inspectable for the same reason as `_shim_off_switches`: which aten key a
    # `torch.<op>` call can reach should be answerable by asking, not by
    # reading the table back out of the artefact.
    module._shim_overloads = {
        name: list(entry.keys) for name, entry in sorted(overloads.items())
    }

    # The same table, for `tensor.<method>(...)`. Separate file and separate
    # dict because they are separate bindings upstream (see methods.json's
    # `_README`), and separate here so `_shim_overloads` keeps meaning exactly
    # "what `torch.<op>` can reach".
    methods = {
        name: _Overloads(name, schemas, _checker_source, self_bound=True)
        for name, schemas in json.loads(methods_json).items()
        if not name.startswith("_README")
    }
    module._shim_methods = {
        name: list(entry.keys) for name, entry in sorted(methods.items())
    }

    # -- types ------------------------------------------------------------
    resolved = {}
    for name, spec in _order_types(surface["types"]):
        if name in real or name in off:
            continue
        resolved[name] = _build_type(name, spec, "torch._C", resolved)
        setattr(module, name, resolved[name])

    # `torch.Size` is a real tuple subclass upstream and the tree relies on it
    # being one (`isinstance(x.shape, tuple)`, unpacking, slicing).
    class Size(tuple):
        __module__ = "torch"

        def numel(self):
            n = 1
            for d in self:
                n *= d
            return n

    module.Size = Size
    resolved["Size"] = Size

    # -- TensorBase members ----------------------------------------------
    #
    # VENDOR.md wall 3: `class Tensor(torch._C.TensorBase)` in
    # `torch/_tensor.py` runs `_C._add_docstr(_C.TensorBase.<name>, ...)`
    # hundreds of times in its *class body*, so the whole member set is
    # demanded before `import torch` returns. PyO3's type objects are ordinary
    # heap types and accept `setattr`, which is why `TensorBase` stays the
    # native class rather than becoming a Python subclass -- a subclass would
    # have broken `isinstance(op_result, TensorBase)`, since results come out
    # of Rust as the native type.
    tensorbase = module.TensorBase
    existing = set(vars(tensorbase))

    def _wanted(member):
        return member not in existing and member not in UNSAFE_DUNDERS

    for member in surface["tensorbase"]["methods"]:
        if _wanted(member):
            setattr(tensorbase, member, _make_function(f"TensorBase.{member}", member))
    for member in surface["tensorbase"]["attrs"]:
        if _wanted(member):
            setattr(tensorbase, member, _make_property(f"TensorBase.{member}"))

    # ...and then the ones that are not stubs. Installed *after* the stub loop
    # so a real implementation always wins over the placeholder that the stub
    # surface would otherwise have left in place, and installed unconditionally
    # rather than through `_wanted` -- several of these (`__eq__`, `__ne__`)
    # are in `UNSAFE_DUNDERS`, which is a rule about *raising* stand-ins, not
    # about working implementations. A raising `__eq__` breaks dict and set
    # use; one that returns a mask is what upstream has.
    _install_tensor_methods(module, tensorbase, dispatch, methods)

    # `type(TensorBase)`. Upstream's is a distinct pybind11-adjacent metatype;
    # ours is plain `type`, because a PyO3 type's metatype cannot be replaced
    # after creation (`__class__` assignment refuses: `type` is not a heap
    # type). `torch/nn/parameter.py:19` does `class _ParameterMeta(_TensorMeta)`
    # and then uses it as the metaclass of a `torch.Tensor` subclass, which
    # works either way. Recorded in docs/IMPORT_TORCH.md as a difference,
    # because `isinstance(X, _TensorMeta)` is now true of every class.
    module._TensorMeta = type

    # -- submodules -------------------------------------------------------
    # Keyed on the module's *real* name, not the literal "torch._C".
    #
    # The golden harness (`tools/golden/loader.py`) loads this artefact
    # standalone, as a module called `_C`, and then imports real upstream torch
    # in the same interpreter to compare against. Registering `torch._C._onnx`
    # in `sys.modules` from a shim that is not `torch._C` made real torch's own
    # `_C` initialisation fail -- `generic_type: cannot initialize type
    # "TensorProtoDataType": an object with that name is already defined`. The
    # shim was reaching into a namespace it does not own.
    prefix = module.__name__
    roots = set(surface["submodules"]) | set(EXTRA_SUBMODULES)
    sys.meta_path.append(_SubmoduleFinder(prefix, roots))
    for name in sorted(roots):
        spec = surface["submodules"].get(name, {})
        sub = types.ModuleType(f"{prefix}.{name}")
        sub.__path__ = []
        sub_types = {}
        for tname, tspec in _order_types(spec.get("types", {})):
            sub_types[tname] = _build_type(tname, tspec, sub.__name__, sub_types)
            setattr(sub, tname, sub_types[tname])
        for fname in spec.get("functions", ()):
            setattr(sub, fname, _make_function(f"torch._C.{name}.{fname}", fname))
        for vname in spec.get("values", ()):
            setattr(sub, vname, _Unimplemented(f"torch._C.{name}.{vname}"))
        _attach_module_catchall(sub)
        setattr(module, name, sub)
        sys.modules[f"{prefix}.{name}"] = sub

    # -- `_dynamo.eval_frame` real no-ops ---------------------------------
    #
    # DYNAMO.md: `torch._dynamo` is an unconditional import (`transformers`
    # pulls it in through `masking_utils.py:42` on `torch >= 2.6`, no
    # `hasattr` gate exists to skip it -- DYNAMO.md §6). It reaches 52 names
    # under `_C._dynamo`, but only two are ever *called* rather than merely
    # referenced: `eval_frame.set_guard_error_hook` and
    # `eval_frame.set_code_exec_strategy`, both at module scope in
    # `torch/_dynamo/guards.py:5457` and `torch/_dynamo/decorators.py:125`.
    # Both calls discard the return value -- they register a hook / tag a
    # code object for an eval-frame hooking mechanism that only
    # `torch.compile` installs, which this project never calls (DYNAMO.md
    # §3.2). So the only requirement is that the call not raise.
    #
    # `_SubmoduleFinder` above already answers `torch._C._dynamo.<anything>`
    # generically -- including two levels deep, since it only checks that the
    # first path segment (`_dynamo`) is a known root -- so every other name
    # under `eval_frame`, `guards`, `utils` and `compiled_autograd` already
    # exists via the lazily-created catch-all module (DYNAMO.md §3.3, §7
    # item 4: `utils`/`compiled_autograd` are 0-access in this path and stay
    # empty). Only these two need a real body instead of the catch-all's
    # `_Unimplemented`, which raises on call. Pre-registering the module here
    # (rather than teaching the finder to special-case two names) means the
    # finder never runs for this particular submodule: Python's import system
    # checks `sys.modules` before consulting `sys.meta_path`.
    if "_dynamo" in roots:
        eval_frame = types.ModuleType(f"{prefix}._dynamo.eval_frame")
        eval_frame.__path__ = []
        _attach_module_catchall(eval_frame)

        def set_guard_error_hook(hook):
            return None

        def set_code_exec_strategy(code, strategy):
            return None

        set_guard_error_hook.__name__ = set_guard_error_hook.__qualname__ = (
            "set_guard_error_hook"
        )
        set_code_exec_strategy.__name__ = set_code_exec_strategy.__qualname__ = (
            "set_code_exec_strategy"
        )
        eval_frame.set_guard_error_hook = set_guard_error_hook
        eval_frame.set_code_exec_strategy = set_code_exec_strategy

        dynamo_sub = getattr(module, "_dynamo")
        setattr(dynamo_sub, "eval_frame", eval_frame)
        sys.modules[f"{prefix}._dynamo.eval_frame"] = eval_frame

    # -- _VariableFunctions ----------------------------------------------
    #
    # VENDOR.md wall 5. `torch.add`, `torch.mm`, `torch.full` and ~620 more
    # public names are written nowhere in the Python tree; `torch/__init__.py`
    # harvests them off this one object at import time and assigns
    # `__module__` on each. Instance attributes, not class attributes: a
    # function reached through a class becomes a bound method and
    # `method.__module__` is read-only.
    class _VariableFunctionsHolder:
        """Enumerable, non-binding, and open at the edges.

        *Enumerable* because `torch/__init__.py:2212` harvests `dir()` of this
        object to create `torch.add` and 620 more. *Non-binding* because every
        function lives in the instance `__dict__`, not the class: reached
        through a class a function becomes a bound method and
        `method.__module__` is read-only, which that same loop assigns.
        *Open* because the stub lists 976 names and upstream has 985 --
        `torch/jit/_builtins.py:117` wants `torch._VF.stft`, which is not one
        of the 976. A member of this object is a hole, never a switch, so the
        catch-all costs nothing (rule 2).
        """

        __module__ = "torch._C"

        def __getattr__(self, name):
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)
            fn = _torch_level_function(name, dispatch, overloads)
            setattr(self, name, fn)
            return fn

    varfns = _VariableFunctionsHolder()
    for name in surface["varfns"]:
        if name.startswith("__"):
            continue
        setattr(varfns, name, _torch_level_function(name, dispatch, overloads))
    # `torch.tensor` is the one name on this object that is not an overload set
    # -- see `_tensor_factory`.
    varfns.tensor = _tensor_factory(module, dispatch)
    module._VariableFunctions = varfns
    module._VariableFunctionsClass = type(varfns)
    module._TensorBase = module.TensorBase
    _install_grad_mode(module, varfns)

    # -- `_C._nn`, and the two composites that are not overload sets -------
    _install_nn(module, dispatch)
    _install_composites(module, varfns, dispatch)

    # -- the enum instances `_initExtension` writes into `torch` -----------
    _install_namespace_types(module, surface["namespace"])

    # -- module-level names ------------------------------------------------
    for name, kind in surface["module"].items():
        if name in off or hasattr(module, name):
            continue
        if kind == "function":
            setattr(module, name, _make_function(f"torch._C.{name}", name))
        elif kind == "bool":
            # A build-configuration flag. `_BUILD_FLAGS` installs the value
            # later (after `_install_behaviour`); what happens here is the
            # check that there *is* one. Falling through to a placeholder
            # would put a truthy object in a branch, which is the bug this
            # kind exists to prevent -- so the surface refuses to build
            # instead, naming the flag that needs deciding.
            if name not in _BUILD_FLAGS:
                raise RuntimeError(
                    f"torch._C shim: the stubs declare {name} as a build "
                    "flag (_bool) and _BUILD_FLAGS has no answer for it. A "
                    "flag has two answers and both change behaviour; decide "
                    "one in bootstrap.py rather than leaving a placeholder, "
                    "which is truthy."
                )
        elif name.lstrip("_")[:1].isupper():
            # Capitalised means a type, and it has to really be one: the names
            # the stubs miss turn up in *annotations*, which Python evaluates
            # at function definition time. `torch/nn/functional.py:7170` is
            # `scale_recipe_a: ScalingType | list[ScalingType]`, and `|` on a
            # non-type is a TypeError -- a placeholder object cannot stand in.
            setattr(module, name, _ShimMeta(
                name, (), {"__module__": "torch._C", "__init__": _permissive_init}))
        else:
            setattr(module, name, _Unimplemented(f"torch._C.{name}"))

    _install_behaviour(module, dispatch)

    # PyO3 emits `__all__` on `#[pymodule]` modules, so `from torch._C import *`
    # -- which is how most of the `torch` namespace comes into being
    # (`torch/__init__.py:445`) -- copies only what is listed. Setting an
    # attribute is not enough (VENDOR.md wall 6).
    module.__all__ = sorted(
        n for n in vars(module) if not n.startswith("_") and n not in off
    )


def _torch_level_function(name: str, dispatch, overloads):
    """A `torch.<name>` harvested from `_VariableFunctions`.

    With a table entry it resolves: the arguments choose an aten overload and
    the call goes through `_C._aten_dispatch` with the resolved key. Without
    one it still refuses, and still will not guess -- `torch.add(...)` picks
    among `add.Tensor`, `add.Scalar` and `add.out` by argument type, and
    sending every call to `.default` would name a key that mostly does not
    exist, poisoning the work queue DESIGN.md §6 is built on.

    Note what stays true either way: resolution decides *which key*, and the
    key still goes to the one door. An op that resolves but has no kernel
    raises `aten op not implemented in torch._C shim: <key>` from `aten.rs` --
    which is a strictly better work item than the old blanket refusal, because
    it names the overload the caller actually needed rather than the whole
    family.
    """
    entry = overloads.get(name)

    if entry is None:

        def fn(*args, **kwargs):
            raise NotImplementedError(
                f"not implemented in torch._C shim: torch.{name}(...) -- overload "
                f"resolution has no table entry for this op "
                f"(rust/torch_c/src/overloads.json); call "
                f"torch.ops.aten.{name}.<overload>, which carries the overload "
                f"and reaches the same dispatcher"
            )

    else:

        def fn(*args, **kwargs):
            key, bound = entry.resolve(args, _strip_python_only_kwargs(name, kwargs))
            return dispatch(key, **bound)

    fn.__name__ = name
    fn.__qualname__ = name
    return fn


# ---------------------------------------------------------------------------
# `TensorBase` members that do something
# ---------------------------------------------------------------------------
#
# docs/C_SURFACE.md §4 measured a small Llama forward plus greedy `generate`
# and found 50 of `TensorBase`'s 694 members actually used. Everything here
# serves that list, and nothing here is a second entrance: a method resolves an
# overload and calls `_C._aten_dispatch`, exactly like `torch.<op>` does.
#
# Three groups, and the difference between them is worth keeping visible:
#
#   1. table-driven (`methods.json`)  -- the overload machine, `self` bound in.
#   2. Python-level                   -- `to`, `item`, `__bool__`, `__getitem__`.
#      Upstream's binding for each of these is not a plain overload set either:
#      `THPVariable_to` reads its arguments and picks between several aten
#      calls, `THPVariable_getitem` walks the index and emits a *sequence* of
#      them. So these are written out, and each one still ends at the one door.
#   3. autograd-shaped               -- `requires_grad`, `grad_fn`, `data`.
#      These are the honest edge of the shim; see `_install_autograd_shape`.


def _tensor_method(name: str, dispatch, entry):
    """One `TensorBase.<name>`, resolving against `entry` with `self` bound.

    Operator dunders answer `NotImplemented` when nothing binds, rather than
    raising. That is upstream's behaviour and it is load-bearing rather than
    polite: the vendored tree compares tensors against strings and against
    `None` in several places, and Python only gets to fall back to its own
    identity comparison if `__eq__` declines. A `TypeError` out of `__eq__`
    would turn `x == "cpu"` into a crash where upstream gives `False`.
    """
    is_dunder = name.startswith("__") and name.endswith("__")

    def method(self, *args, **kwargs):
        try:
            key, bound = entry.resolve(
                (self,) + args, _strip_python_only_kwargs(name, kwargs)
            )
        except TypeError:
            if is_dunder:
                return NotImplemented
            raise
        return dispatch(key, **bound)

    method.__name__ = name
    method.__qualname__ = f"TensorBase.{name}"
    return method


def _install_tensor_methods(module, tensorbase, dispatch, methods) -> None:
    for name, entry in methods.items():
        setattr(tensorbase, name, _tensor_method(name, dispatch, entry))

    _install_tensor_conversions(module, tensorbase, dispatch)
    _install_tensor_scalars(tensorbase, dispatch)
    _install_tensor_indexing(tensorbase, dispatch)
    _install_tensor_softmax(tensorbase, dispatch)
    _install_autograd_shape(tensorbase)


# `x.float()` and friends. Upstream has no `aten::float`: `THPVariable_float`
# is `self.to(ScalarType::Float)`, and a `TorchDispatchMode` logger confirms it
# -- `f.float()` on a tensor that is already float32 produces *no* aten record
# at all, while `i.float()` produces `aten._to_copy.default`. Both halves are
# reproduced below: the no-change case returns `self` without dispatching, and
# the change case names `_to_copy`, which is the key upstream's dispatcher
# really sees. (The parser-level key would be `aten.to.dtype`; that overload is
# composite and never reaches a kernel. Naming the one that does keeps the work
# queue of DESIGN.md §6 pointing at something implementable.)
_DTYPE_METHODS = {
    "float": "float32",
    "double": "float64",
    "half": "float16",
    "bfloat16": "bfloat16",
    "long": "int64",
    "int": "int32",
    "short": "int16",
    "char": "int8",
    "byte": "uint8",
    "bool": "bool",
}


def _install_tensor_conversions(module, tensorbase, dispatch) -> None:
    def _to_copy(self, dtype=None, device=None, copy=False):
        if isinstance(device, str):
            device = module.device(device)
        # `!=`, not `is not`: `self.dtype` builds a fresh `PyDtype` on every
        # read (dtype.rs owns the tag, not a singleton table), so identity is
        # never true even for the same dtype. Both types define `__eq__`.
        wants_dtype = dtype is not None and dtype != self.dtype
        wants_device = device is not None and device != self.device
        if not (wants_dtype or wants_device or copy):
            return self
        return dispatch(
            "aten._to_copy.default",
            self,
            dtype=dtype if wants_dtype else None,
            device=device if wants_device else None,
        )

    for name, dtype_name in _DTYPE_METHODS.items():
        def convert(self, _dtype_name=dtype_name):
            return _to_copy(self, dtype=getattr(module, _dtype_name))

        convert.__name__ = name
        convert.__qualname__ = f"TensorBase.{name}"
        setattr(tensorbase, name, convert)

    def to(self, *args, **kwargs):
        """`Tensor.to`, in upstream's own argument shapes.

        This is the clearest case of docs/OVERLOAD.md §9 item 7 -- a Python
        binding whose signatures do not line up with any aten schema. torch's
        parser takes `to(Device device=None, ScalarType dtype=None, ...)`,
        `to(ScalarType dtype, ...)` and `to(Tensor other, ...)`; the aten
        overloads are `to.dtype`, `to.device` (which requires *both*),
        `to.other` and `to.dtype_layout` (all-keyword). `x.to('cpu')` binds the
        first parser signature and none of the aten ones, so a table entry
        would report "no matching overload" for a call real torch accepts.
        """
        kwargs = dict(kwargs)
        copy = bool(kwargs.pop("copy", False))
        kwargs.pop("non_blocking", None)
        kwargs.pop("memory_format", None)
        dtype = kwargs.pop("dtype", None)
        device = kwargs.pop("device", None)
        other = kwargs.pop("other", None)
        if kwargs:
            raise TypeError(
                f"Tensor.to(): unexpected keyword argument(s) "
                f"{sorted(kwargs)} in torch._C shim"
            )
        for value in args:
            if isinstance(value, module.dtype):
                dtype = value
            elif isinstance(value, (module.device, str)):
                device = value
            elif isinstance(value, tensorbase):
                other = value
            elif isinstance(value, bool):
                copy = copy or value  # non_blocking, then copy -- both bools
            else:
                raise TypeError(
                    f"Tensor.to(): torch._C shim does not understand argument "
                    f"{value!r} of type {type(value).__name__}"
                )
        if other is not None:
            dtype, device = other.dtype, other.device
        return _to_copy(self, dtype=dtype, device=device, copy=copy)

    to.__name__ = "to"
    to.__qualname__ = "TensorBase.to"
    setattr(tensorbase, "to", to)
    setattr(tensorbase, "type_as", lambda self, other: _to_copy(self, dtype=other.dtype))


def _install_tensor_softmax(tensorbase, dispatch) -> None:
    """`Tensor.softmax(dim, dtype=None)`.

    `methods.json`'s `_README` explains why this is not a table entry: the
    parser-level key for `Tensor.softmax` is `aten::softmax.int`, which is
    `CompositeImplicitAutograd` and never reaches a kernel, while the key
    upstream's dispatcher actually sees is `aten._softmax.default` (measured
    with a `TorchDispatchMode` logger on torch 2.13.0, docs/NN_SURFACE.md §6):

        x.softmax(dim)                      -> aten._softmax.default(x, dim, False)
        x.softmax(dim, dtype=torch.float64)  -> aten._to_copy.default(x, dtype=float64)
                                                 then aten._softmax.default(_, dim, False)
        x.softmax(dim, dtype=x.dtype)        -> aten._softmax.default(x, dim, False)
                                                 (no `_to_copy` -- measured: a no-op dtype
                                                 emits no conversion call either)

    `half_to_float` stays `False` unconditionally: it is an autocast signal
    (true only when the dispatcher itself decides to keep a fp16 input in
    fp32 accumulation), this shim has no autocast, and the trace above shows
    upstream's own eager path never sets it for a plain `dtype=None` call or
    an explicit-`dtype` one either. Same reasoning, same shape as `to.dtype`
    versus `_to_copy` in `_install_tensor_conversions`.
    """

    def softmax(self, dim, dtype=None):
        source = self
        if dtype is not None and dtype != self.dtype:
            source = dispatch("aten._to_copy.default", self, dtype=dtype)
        return dispatch("aten._softmax.default", source, dim, False)

    softmax.__name__ = "softmax"
    softmax.__qualname__ = "TensorBase.softmax"
    setattr(tensorbase, "softmax", softmax)


def _install_tensor_scalars(tensorbase, dispatch) -> None:
    """`item()` and `__bool__`, both of which leave the tensor world.

    `aten::item` exists, but it is not what upstream reaches: a
    `TorchDispatchMode` logger over `t.item()` records exactly
    `aten._local_scalar_dense.default`, and over `bool(t)` the same one. So
    that is the key, and the numel check that upstream does before it stays on
    this side, where it can carry torch's own message.
    """

    def item(self):
        if self.numel() != 1:
            raise RuntimeError(
                f"a Tensor with {self.numel()} elements cannot be converted to Scalar"
            )
        return dispatch("aten._local_scalar_dense.default", self)

    def __bool__(self):
        if self.numel() == 0:
            raise RuntimeError("Boolean value of Tensor with no values is ambiguous")
        if self.numel() != 1:
            raise RuntimeError(
                "Boolean value of Tensor with more than one value is ambiguous"
            )
        return bool(dispatch("aten._local_scalar_dense.default", self))

    def __float__(self):
        return float(item(self))

    def __int__(self):
        return int(item(self))

    def __index__(self):
        value = item(self)
        if isinstance(value, float):
            raise TypeError(
                "only integer tensors of a single element can be converted to an index"
            )
        return int(value)

    for name, fn in (
        ("item", item),
        ("__bool__", __bool__),
        ("__float__", __float__),
        ("__int__", __int__),
        ("__index__", __index__),
    ):
        fn.__name__ = name
        fn.__qualname__ = f"TensorBase.{name}"
        setattr(tensorbase, name, fn)


def _install_tensor_indexing(tensorbase, dispatch) -> None:
    """`x[...]`, decomposed the way `THPVariable_getitem` decomposes it.

    Upstream's indexing is not one aten call; it is a walk over the index that
    emits `select.int` for an integer, `slice.Tensor` for a slice, `unsqueeze`
    for a `None`, and one `index.Tensor` at the end if any index was a tensor.
    Measured, on torch 2.13.0:

        f[0]        -> [select.int]
        f[0, 1]     -> [select.int, select.int]
        f[:, 1]     -> [select.int]          (the full slice emits nothing)
        f[0:1]      -> [slice.Tensor]
        f[None]     -> [unsqueeze]
        f[bool_t]   -> [index.Tensor]

    So this reproduces the walk rather than inventing a single `getitem` op,
    and every step goes through `_aten_dispatch`. What it does *not* do is
    mixed basic-and-advanced indexing: an index containing both a tensor and a
    non-trivial slice is refused by name instead of being approximated.
    """

    def __getitem__(self, index):
        if not isinstance(index, tuple):
            index = (index,)
        # Located with `is`, never with `==` or `.index()`: an index tuple may
        # hold a tensor, and `TensorBase.__eq__` now answers with a mask.
        # `tuple.index(Ellipsis)` would compare its way there elementwise.
        ellipses = [k for k, item in enumerate(index) if item is Ellipsis]
        if ellipses:
            if len(ellipses) > 1:
                raise IndexError("an index can only have a single ellipsis ('...')")
            consumed = sum(
                1 for item in index if item is not None and item is not Ellipsis
            )
            at = ellipses[0]
            fill = (slice(None),) * max(self.dim() - consumed, 0)
            index = index[:at] + fill + index[at + 1 :]

        def _is_full_slice(item):
            return (
                isinstance(item, slice)
                and item.start is None
                and item.stop is None
                and item.step is None
            )

        if any(isinstance(item, tensorbase) for item in index):
            if any(
                not (item is None or isinstance(item, tensorbase) or _is_full_slice(item))
                for item in index
            ):
                raise NotImplementedError(
                    "not implemented in torch._C shim: TensorBase.__getitem__ mixing "
                    "a tensor index with integer or slice indices -- upstream applies "
                    "basic indexing first and then aten.index.Tensor, and this shim "
                    "does not reproduce that composition yet"
                )
            indices = [item if isinstance(item, tensorbase) else None for item in index]
            return dispatch("aten.index.Tensor", self, indices)

        result = self
        dim = 0
        for item in index:
            if item is None:
                result = dispatch("aten.unsqueeze.default", result, dim)
                dim += 1
            elif isinstance(item, bool):
                # torch treats a plain `True`/`False` as a zero-dim mask, which
                # adds a dimension of length 1 or 0. Not measured as used, and
                # guessing it is the kind of thing this shim refuses.
                raise NotImplementedError(
                    "not implemented in torch._C shim: TensorBase.__getitem__ with a "
                    "Python bool index"
                )
            elif isinstance(item, int):
                result = dispatch("aten.select.int", result, dim, item)
            elif isinstance(item, slice):
                if _is_full_slice(item):
                    dim += 1
                    continue
                result = dispatch(
                    "aten.slice.Tensor",
                    result,
                    dim,
                    item.start,
                    item.stop,
                    1 if item.step is None else item.step,
                )
                dim += 1
            else:
                raise NotImplementedError(
                    f"not implemented in torch._C shim: TensorBase.__getitem__ with "
                    f"an index of type {type(item).__name__}"
                )
        return result

    __getitem__.__name__ = "__getitem__"
    __getitem__.__qualname__ = "TensorBase.__getitem__"
    setattr(tensorbase, "__getitem__", __getitem__)

    def __len__(self):
        if self.dim() == 0:
            raise TypeError("len() of a 0-d tensor")
        return self.shape[0]

    setattr(tensorbase, "__len__", __len__)


def _install_autograd_shape(tensorbase) -> None:
    """`requires_grad`, `grad_fn`, `data` -- the papered-over part.

    **This is the one group here that is not an implementation.** There is no
    autograd behind this shim (DESIGN.md §3 stage 0), and `from_config` reaches
    `TensorBase.requires_grad_` before it reaches anything else interesting, so
    the choice was between stopping there and carrying an inert flag.

    The flag is carried, and the boundary is drawn where it can be seen:

      * `requires_grad` stores and reports what was set. Nothing reads it.
      * `grad_fn` and `grad` are always `None`, which is the truth -- no graph
        node was ever created and no gradient was ever accumulated.
      * `backward()` stays a raising stub, so code that actually depends on
        the flag meaning something fails by name rather than silently getting
        zeros.

    `data` returns `self`, not a detached alias. Upstream's `.data` shares
    storage with the original, so `p.data.normal_()` writes through to `p`;
    returning `self` gives that same write-through with the same object, and
    differs in that our `.data` still reports the original `requires_grad`.
    Recorded in docs/TENSORBASE.md rather than hidden.
    """

    def _make_subclass(cls, data, require_grad=False, dispatch_strides=False,
                       dispatch_device=False, device_for_backend_keys=None):
        """`torch.Tensor._make_subclass`, which is how a `Parameter` is born.

        `torch/nn/parameter.py:57` is `torch.Tensor._make_subclass(cls, data,
        requires_grad)` and two more sites do the same. Upstream re-wraps the
        same `TensorImpl` in a new Python object of class `cls`; here `cls(data)`
        reaches `TensorBase`'s `#[new]`, which PyO3 allocates with the subtype
        it was called on -- so the result really is a `Parameter`, and
        `nn.Module.__setattr__`'s `isinstance(value, Parameter)` is satisfied.

        The trailing three arguments are dispatch-key plumbing for subclasses
        that override strides or device; there is no dispatcher key set here,
        so they are accepted and ignored rather than refused -- refusing would
        stop a call that upstream makes with their defaults.
        """
        # `TensorBase.__new__(cls, ...)`, not `cls(...)`. `Parameter.__new__`
        # is the caller here, and calling `cls(data)` would re-enter it --
        # measured, as a `RecursionError` five frames deep. Upstream allocates
        # directly too; `_make_subclass` is below `__new__`, not above it.
        made = tensorbase.__new__(cls, data)
        made.requires_grad = bool(require_grad)
        return made

    # A *static* method, not a class method: upstream's is called as
    # `torch.Tensor._make_subclass(cls, data, requires_grad)` with the target
    # class passed explicitly, so binding the receiver would shift every
    # argument one place left and `Parameter` would arrive as the data.
    tensorbase._make_subclass = staticmethod(_make_subclass)

    def requires_grad_(self, mode=True):
        self.requires_grad = bool(mode)
        return self

    requires_grad_.__name__ = "requires_grad_"
    requires_grad_.__qualname__ = "TensorBase.requires_grad_"
    setattr(tensorbase, "requires_grad_", requires_grad_)
    setattr(tensorbase, "grad_fn", property(lambda self: None))
    setattr(tensorbase, "grad", property(lambda self: None))
    setattr(tensorbase, "is_leaf", property(lambda self: True))
    setattr(tensorbase, "data", property(lambda self: self))
    setattr(tensorbase, "retain_grad", lambda self: None)


def _install_grad_mode(module, varfns) -> None:
    """The grad-mode flags `torch.no_grad()` turns on and off.

    docs/FROM_CONFIG.md §2.2 measured `_set_grad_enabled` at **84 calls** during
    `AutoModelForCausalLM.from_config` -- the single most-called name in the
    whole trace, because every `@torch.no_grad()`-decorated initialiser flips it
    twice. So this is not an edge case that can be left refusing.

    **What is implemented is the flag, not what the flag means.** The state
    round-trips exactly: `no_grad()` reads the previous value, sets `False`,
    restores it, and gets back what it stored. That is the whole of the
    observable contract at this layer, and `torch/autograd/grad_mode.py` needs
    nothing else. What does *not* exist is the thing the flag would govern --
    there is no graph, so turning recording "on" records nothing either. It is
    the same boundary `TensorBase.requires_grad` draws (see
    `_install_autograd_shape`), and it is drawn in the same place: `backward()`
    stays a raising stub.

    `is_grad_enabled` is overwritten on `_VariableFunctions` as well as on the
    module. `torch/__init__.py` harvests `torch.is_grad_enabled` off that
    object, so leaving the harvested copy as the table-less refusal would make
    `torch.is_grad_enabled()` and `torch._C.is_grad_enabled()` disagree.
    """
    state = {
        "grad": True,
        # `torch/autograd/grad_mode.py:340` and `:393`. Both are real
        # backend-configuration switches (thread pool, layout enforcement) with
        # nothing behind them here, and both are context managers that restore
        # what they read -- so, like `grad`, they have to round-trip.
        "multithreading": True,
        "layout_enforcement": False,
    }

    def is_grad_enabled():
        return state["grad"]

    def _set_grad_enabled(mode):
        state["grad"] = bool(mode)

    def _is_multithreading_enabled():
        return state["multithreading"]

    def _set_multithreading_enabled(mode):
        state["multithreading"] = bool(mode)

    def _is_grad_layout_enforcement_enabled():
        return state["layout_enforcement"]

    def _set_grad_layout_enforcement_enabled(mode):
        state["layout_enforcement"] = bool(mode)

    for name, fn in (
        ("is_grad_enabled", is_grad_enabled),
        ("_set_grad_enabled", _set_grad_enabled),
        ("_is_multithreading_enabled", _is_multithreading_enabled),
        ("_set_multithreading_enabled", _set_multithreading_enabled),
        ("_is_grad_layout_enforcement_enabled", _is_grad_layout_enforcement_enabled),
        ("_set_grad_layout_enforcement_enabled", _set_grad_layout_enforcement_enabled),
    ):
        fn.__name__ = name
        fn.__qualname__ = f"torch._C.{name}"
        setattr(module, name, fn)
    varfns.is_grad_enabled = is_grad_enabled
    # Readable so the state can be inspected rather than inferred.
    module._shim_grad_state = state


def _tensor_factory(module, dispatch):
    """`torch.tensor(...)`, which has no overload set to resolve against.

    Every other name in `overloads.json` maps to aten overloads. This one does
    not: upstream's `torch.tensor` is `THPVariable_tensor` ->
    `internal_new_from_data`, a `_C` function, and a real `torch.tensor([1, 2])`
    produces exactly one aten record -- `aten.lift_fresh.default` (measured).
    `aten::tensor` does exist, but it is a TorchScript builtin that this path
    never reaches, so routing here would have named a key upstream never names.

    So the split is upstream's own: `_C` builds the data, and the single aten
    call upstream really makes is the single aten call made here. The door is
    still one door -- `lift_fresh` goes through `_aten_dispatch` like anything
    else.
    """

    def tensor(data, *, dtype=None, device=None, requires_grad=False, pin_memory=False):
        if requires_grad:
            raise NotImplementedError(
                "not implemented in torch._C shim: torch.tensor(requires_grad=True) "
                "-- there is no autograd behind this shim"
            )
        if pin_memory:
            raise NotImplementedError(
                "not implemented in torch._C shim: torch.tensor(pin_memory=True)"
            )
        if isinstance(device, str):
            device = module.device(device)
        return dispatch(
            "aten.lift_fresh.default",
            module._tensor_new_from_data(data, dtype, device),
        )

    tensor.__name__ = "tensor"
    tensor.__qualname__ = "tensor"
    return tensor


def _install_namespace_types(module, namespace) -> None:
    """`torch.strided`, `torch.contiguous_format`, `torch.per_tensor_affine`.

    `dtype` is not here: `_C` owns that type in Rust and registers every
    instance itself (see `dtype.rs`), because a dtype means something. These
    three are pure labels -- there is no candle concept behind `torch.strided`
    -- so they are built here, in the same spirit as `device`: a label, and the
    thing that uses it is what fails.
    """
    for kind in ("layout", "memory_format", "qscheme"):
        cls = getattr(module, kind, None)
        if cls is None or not isinstance(cls, type):
            cls = _ShimMeta(kind, (), {"__module__": "torch", "__init__": _permissive_init})
            setattr(module, kind, cls)
        for name, value_kind in namespace.items():
            if value_kind != kind or hasattr(module, name):
                continue
            instance = cls.__new__(cls)
            instance._shim_name = name
            setattr(module, name, instance)
            cls.__repr__ = lambda self: f"torch.{self._shim_name}"
            cls.__str__ = cls.__repr__


def _constant_function(qualname: str, value):
    def fn(*args, **kwargs):
        return value

    fn.__name__ = qualname.rsplit(".", 1)[-1]
    fn.__qualname__ = qualname
    return fn


# `_C` functions whose *return value* the import path consumes, with the site
# that consumes it. A `NotImplementedError` here is not a work item -- it is a
# stop, because the tree has no fallback. Each entry states what upstream
# returns and what the shim returns instead.
_DISCOVERED_RETURNS = {
    # `torch/_tensor.py:105` -- `__dlpack_c_exchange_api__: object =
    # torch._C._dlpack_exchange_api()`, in the `class Tensor` body. Upstream
    # hands back a PyCapsule of C function pointers for zero-copy tensor
    # exchange. The shim has no DLPack bridge, and `None` is what the attribute
    # would hold on a build without one.
    "_dlpack_exchange_api": None,
    # `torch/_library/fake_class_registry.py:420`, from the `@register_fake_class`
    # decorator that `torch/_library/opaque_object.py:72` applies at import.
    # Upstream returns the registered `torch::class_` wrapper and the caller
    # discards it -- the call is there to assert the class exists. There are no
    # custom classes in this shim, so nothing to assert and nothing to return.
    "_get_custom_class_python_wrapper": None,
    # `torch/_library/opaque_object.py:268`. Upstream records the name in the
    # C++ type registry so a custom op schema can mention it; the return value is
    # unused. The shim has no schema registry, so this is a no-op -- recorded
    # rather than hidden, because if opaque types ever have to round-trip through
    # a schema this is where that starts.
    "_register_opaque_type": None,
    # `torch/jit/_trace.py:616`, at module scope. Upstream flips a global in the
    # TorchScript tracer so that tracing a Python-only op warns instead of
    # silently producing a constant. There is no tracer here, so there is nothing
    # to flip; if TorchScript tracing is ever wanted this is one of its entry
    # points.
    "_tracer_warn_use_python": None,
    # `torch/library.py:1767`, reached from `torch/_native/registry.py:894` while
    # that module registers its overrides at import. Upstream returns an opaque
    # handle to the kernel already registered at (op, dispatch_key), which the
    # caller keeps so it can re-enter the dispatcher without recursing through its
    # own override. There is no dispatcher here and therefore no prior kernel;
    # `None` says so. The override machinery still records itself, and still has
    # nothing to override.
    "_dispatch_get_computed_kernel_for_dispatch_key": None,
    # `torch/xpu/__init__.py:278`, reached from `torch.xpu.is_available()`,
    # reached from `transformers/masking_utils.py:39` -- at *import* of
    # `transformers.generation.utils`, which is the lazy import
    # docs/IMPORT_TORCH.md §11 item 3 recorded `from_config` as dying in.
    #
    # This one is a real answer, not a stand-in: upstream returns the number of
    # XPU devices, and a build with no XPU support has none. The name has to
    # exist -- `torch.xpu` is an ordinary Python package that is always
    # importable, so the absence trick (VENDOR.md wall 11) does not apply here;
    # `hasattr` never gets a chance to be the question.
    "_xpu_getDeviceCount": 0,
    # `torch/mtia/__init__.py:152`, reached from
    # `torch/_dynamo/device_interface.py:297` at module scope, reached from
    # `torch/_inductor/utils.py:115`, reached from the same transformers lazy
    # import as the entry above. Upstream returns whether the build has MTIA
    # support compiled in; this one does not. Same shape of answer, same
    # reason the name cannot simply be absent.
    "_mtia_isBuilt": False,
    # `torch/mtia/__init__.py:81`, reached from `torch/random.py:82` -- that
    # is, from `torch.manual_seed`, which walks every accelerator's seeding
    # entry point before it reaches the CPU generator it was actually asked
    # about. Upstream answers "was this process forked after MTIA was
    # initialised?"; MTIA is never initialised here, so `False` is true and
    # `manual_seed_all` then queues its callback instead of running it.
    "_mtia_isInBadFork": False,
    # `torch/random.py:127`, the last stop in `torch.manual_seed` before the
    # CPU generator. Upstream's default is the literal string
    # `"privateuseone"`, and the caller then does `hasattr(torch, name)` --
    # nothing is registered under that name here, so the branch is skipped
    # exactly as it is upstream on a stock build. Returning the real default
    # rather than `None` matters: `hasattr(torch, None)` is a TypeError.
    "_get_privateuse1_backend_name": "privateuseone",
    # `torch/nn/modules/module.py:530` -- the first line of
    # `nn.Module.__init__`, so *every* module ever constructed calls it, which
    # makes it the very next wall after `import torch` on the road to
    # `from_config`. Upstream increments an internal usage counter and returns
    # nothing; there is no counter here and nothing reads the result. This is
    # the cleanest member of docs/C_SURFACE.md §7's second tier -- a name whose
    # existence is the whole requirement.
    "_log_api_usage_once": None,
    # `torch/nn/modules/module.py`, the same shape: upstream records that a
    # module of this class was instantiated. Returns nothing, read by nobody.
    "_log_api_usage_metadata": None,
    # The `__torch_function__` fast-path predicates. `torch/nn/init.py:597`
    # is the first of hundreds of sites: *every* function in `torch/nn/init.py`,
    # `torch/nn/functional.py` and `torch/_tensor.py` opens with
    # `if torch.overrides.has_torch_function_variadic(...)`, and upstream's C
    # implementation answers "does any argument's type override
    # `__torch_function__`?".
    #
    # `False` is the true answer here, not a stand-in: the protocol is a real
    # dispatch mechanism this shim does not implement, and there is no type in
    # the vendored tree's inference path that overrides it -- `Parameter` and
    # `Tensor` inherit the default. If a subclass ever does override it, this
    # is the single place that has to learn to say so, and until then a `True`
    # would send every one of those call sites into a handler that cannot run.
    # `torch/_tensor.py:1186`, in `Tensor.__len__`, and a dozen more places
    # that warn only while TorchScript tracing. Upstream returns the active
    # `TracingState` or `None`; there is no tracer here, so `None` is the true
    # answer -- and it has to be *callable and answering*, not merely present,
    # because `len(tensor)` is on the path. (docs/C_SURFACE.md §1-3 noticed
    # this name being looked up during `import torch`; it is called later.)
    "_get_tracing_state": None,
    "_has_torch_function": False,
    "_has_torch_function_unary": False,
    "_has_torch_function_variadic": False,
    # The mode stack, which `torch/overrides.py` and
    # `torch/utils/_python_dispatch.py` consult before the predicates above.
    # No mode is ever pushed here, so the stack is empty and disabled.
    "_is_torch_function_enabled": False,
    "_is_torch_function_mode_enabled": False,
    "_is_torch_function_all_disabled": False,
    "_len_torch_function_stack": 0,
    "_len_torch_dispatch_stack": 0,
}

# Build-configuration flags: plain `bool` *values* upstream, not callables, so
# they cannot go in `_DISCOVERED_RETURNS` above (which wraps everything in a
# function).
#
# These were being answered by `_Unimplemented`, which is **truthy**. That is
# the one shape of placeholder that is worse than absence for a name spelled
# `_has_<backend>`: every `if torch._C._has_mps:` in the tree took the branch
# that assumes a backend exists, and did so silently. It cost a wall to
# notice -- `torch.manual_seed(0)` reaches `torch/mps/__init__.py:67`, whose
# whole guard is `if not torch._C._has_mps: return`, and went straight past it
# into `_mps_get_default_generator`.
#
# `False` is not a stand-in here, it is the fact: this shim's tensor engine is
# candle on the CPU (DESIGN.md §4) and there is no CUDA, MPS, XPU or MKLDNN
# behind any of these names. `torch.cuda.is_available()` already answered
# `False` by a different route; this makes the rest of the family agree with
# it instead of each guarding site getting a different answer depending on
# which spelling it happened to use.
#
# The table is no longer a hand list of the ones that happened to cost a wall.
# `install` requires an entry for **every** module-level name the stubs
# annotate `_bool` (`surface.json` kind `"bool"`), because a build flag has
# exactly two answers and both change behaviour -- there is no third,
# placeholder-shaped answer to fall back on. Adding a `_bool` to the stubs and
# not deciding here is now an import-time error naming the flag, rather than a
# truthy object waiting in a branch.
#
# Every value below is `False`, and that is a fact about this build rather than
# a default: candle on the CPU links none of these libraries. Two entries are
# worth their own note --
#
#   `_has_kleidiai`  Upstream on the host that measured this answers *True*
#                    (KleidiAI is ARM's kernel library and an arm64 mac build
#                    picks it up). That is a fact about upstream's build, not
#                    about the API, and copying it would be the same error as
#                    the truthy placeholder in a politer form.
#
#   `_GLIBCXX_USE_CXX11_ABI`
#                    The only one where the question does not really apply:
#                    there is no libstdc++ under this shim at all, so neither
#                    answer describes it. `False` is chosen because it is the
#                    one that cannot make a caller assume a GNU C++ ABI is
#                    present. `torch/__init__.py:2354`
#                    (`compiled_with_cxx11_abi`) is the only reader and nothing
#                    on the `import torch` path calls it. Before this it was
#                    not even a value -- the install loop reads a leading
#                    capital as a type name, so `_GLIBCXX_USE_CXX11_ABI` was
#                    being answered with a *class*.
_BUILD_FLAGS = {
    # Reached through `torch.backends.<name>.is_available()`, one per line.
    "_has_mps": False,
    "_has_cuda": False,
    "_has_xpu": False,
    "_has_mkldnn": False,
    # `torch/backends/cudnn/__init__.py:126`, and :231 in a class body -- the
    # one flag measured to change a branch during `import torch`.
    "_has_cudnn": False,
    # `torch/backends/cusparselt/__init__.py:51`.
    "_has_cusparselt": False,
    # `torch/backends/kleidiai/__init__.py:7`. See the note above.
    "_has_kleidiai": False,
    # `torch/cuda/__init__.py:155` -- `has_magma: bool = torch._C._has_magma`,
    # a module-level assignment, so the placeholder was being published as
    # `torch.cuda.has_magma` for anyone who read it.
    "_has_magma": False,
    # `torch/backends/mkldnn/__init__.py:25`. ACL is a sub-capability of
    # MKLDNN, so `True` here would contradict `_has_mkldnn` above.
    "_has_mkldnn_acl": False,
    # `torch/backends/mkl/__init__.py:7`, and `_meta_registrations.py:2864`,
    # which registers a meta kernel for `torch.ops.mkl._mkl_linear` when true.
    # This build has no such op, so the registration was for nothing.
    "has_mkl": False,
    # No LAPACK is linked; `torch.linalg` has no kernels here.
    "has_lapack": False,
    # candle parallelises with rayon, not OpenMP.
    "has_openmp": False,
    # FFT support. No `aten.fft*` kernel exists in this build.
    "has_spectral": False,
    # See the note above: the question does not apply, and this is the answer
    # that claims the least.
    "_GLIBCXX_USE_CXX11_ABI": False,
}

# The same, for members of synthesised `_C` types: `"<Type>.<member>": value`.
# Installed after the types are built.
_DISCOVERED_TYPE_RETURNS = {
    # `torch/_sources.py:122` -- `get_source_lines_and_file(fn,
    # ErrorReport.call_stack())`, reached from `parse_def`, reached from
    # `_check_overload_body`, which `@torch.jit._overload` runs *at decoration
    # time* while `torch/nn/functional.py` is importing (VENDOR.md wall 13).
    # Upstream returns TorchScript's compilation call stack as a string, used
    # only to decorate an error message. There is no such stack here, and the
    # empty string is what upstream returns when the stack is empty.
    "ErrorReport.call_stack": "",
}


def _install_dispatch_keys(module) -> None:
    """`DispatchKey` and `DispatchKeySet`, for real.

    VENDOR.md wall 14 -- `torch/_ops.py:139` asserts `isinstance(k,
    DispatchKey)` and IMPORT_WALLS reached the same line from the opposite
    direction, which is what makes it structure rather than a detour. It is
    also, unusually for this file, *cheap* structure: `DispatchKey` is an
    enumeration and `DispatchKeySet` is a set of them, and both fall out of
    `enum` and `frozenset` with no dispatcher behind them.

    Every `HigherOrderOperator.__init__` runs `_dispatch_keyset_full()` and
    then removes keys from it (`torch/_ops.py:304`), so this executes during
    `import torch`, not merely on use. What is *not* here is anything that
    consults the set to route a call -- `_aten_dispatch` remains the one door.
    """
    DispatchKey = module.DispatchKey

    class DispatchKeySet:
        __module__ = "torch._C"
        __slots__ = ("_keys",)

        def __init__(self, key=None):
            if isinstance(key, DispatchKeySet):
                self._keys = key._keys
            elif key is None:
                self._keys = frozenset()
            else:
                self._keys = frozenset({key})

        @classmethod
        def _of(cls, keys):
            made = cls.__new__(cls)
            made._keys = frozenset(keys)
            return made

        def has(self, key):
            return key in self._keys

        def add(self, key):
            return self._of(self._keys | {key})

        def remove(self, key):
            return self._of(self._keys - {key})

        def __or__(self, other):
            return self._of(self._keys | DispatchKeySet(other)._keys)

        def __and__(self, other):
            return self._of(self._keys & DispatchKeySet(other)._keys)

        def __sub__(self, other):
            return self._of(self._keys - DispatchKeySet(other)._keys)

        def __iter__(self):
            return iter(sorted(self._keys, key=lambda k: k.name))

        def __contains__(self, key):
            return key in self._keys

        def __len__(self):
            return len(self._keys)

        def __bool__(self):
            return bool(self._keys)

        def __eq__(self, other):
            return isinstance(other, DispatchKeySet) and self._keys == other._keys

        def __hash__(self):
            return hash(self._keys)

        def highestPriorityTypeId(self):
            # Priority is the enum's declaration order in the stub, which is
            # upstream's own order. Not a claim that the shim dispatches by it.
            return max(self._keys, key=lambda k: list(DispatchKey).index(k))

        def raw_repr(self):
            return sum(1 << list(DispatchKey).index(k) for k in self._keys)

        def __repr__(self):
            return "DispatchKeySet(" + ", ".join(k.name for k in self) + ")"

    module.DispatchKeySet = DispatchKeySet
    every = DispatchKeySet._of(DispatchKey)
    module._dispatch_keyset_full = lambda: every
    module._dispatch_keyset_full_after = lambda key: DispatchKeySet._of(
        list(DispatchKey)[list(DispatchKey).index(key) + 1 :]
    )
    module._dispatch_keyset_to_string = lambda keyset: repr(keyset)
    # "Does the dispatcher know this op at all?" `torch/_decomp/__init__.py:90`
    # uses it to filter TorchScript's junk overloads (`aten.add.float_int`) out
    # of the decomposition registry.
    #
    # PAPERED OVER, and the direction was chosen rather than defaulted. `False`
    # would empty the decomposition table, which is the mechanism DESIGN.md §2
    # relies on for everything outside Core ATen -- so the table would be gone
    # in exchange for tidiness. `True` keeps the table and lets a handful of
    # junk overloads into it, where they are inert. The cost of the lie is a
    # few extra registry entries; the cost of the truth would be the feature.
    module._dispatch_has_kernel = lambda *a, **k: True
    # The narrower question -- "for *this* backend key?" -- is answered
    # conservatively, because a `True` there would claim a specific kernel.
    module._dispatch_has_kernel_for_dispatch_key = lambda *a, **k: False
    module._dispatch_has_kernel_for_any_dispatch_key = lambda *a, **k: False

    # A `DispatchKeySet` *value*, not a function.
    # `torch/_subclasses/functional_tensor.py:146` does
    # `torch._C._additional_keys_to_prop_for_wrapper_tensors.add(...)` in a
    # class body. Upstream seeds it with the keys a wrapper subclass should
    # inherit from its wrapped tensor; empty is the honest starting point for a
    # shim with no key propagation, and `.add` still works because the set is
    # real.
    module._additional_keys_to_prop_for_wrapper_tensors = DispatchKeySet()

    def _parse_dispatch_key(name):
        """`torch/library.py:915` -- "is this string a dispatch key?".

        Upstream returns the `DispatchKey` or `None`, and the caller branches
        on `None` to decide whether the string was a device type instead. The
        enum comes from the vendored stub, so this is a real lookup, not a
        guess.
        """
        return getattr(DispatchKey, name, None)

    module._parse_dispatch_key = _parse_dispatch_key
    module._dispatch_key_parse = _parse_dispatch_key


def _install_library(module) -> None:
    """`torch._C._dispatch_library` -- the operator registry, as a recorder.

    `torch/library.py:244` builds one of these per `Library(...)`, and
    `torch/_meta_registrations.py`, `torch/_decomp/` and several `torch/_refs`
    modules construct libraries and call `define`/`impl` *at import time*, so
    this is import-blocking rather than lazy.

    **This is the largest thing in this file that is papered over, and it is
    worth being precise about what it costs.** Upstream's object writes into
    the C++ dispatcher, so a kernel registered from Python is afterwards
    reachable through `torch.ops`. Here the registrations are recorded and
    dropped: `_aten_dispatch` is the only thing that answers a call, and it
    knows nothing about them. Concretely, `torch.library.impl(...)` appears to
    succeed and then has no effect.

    Recording rather than discarding, and exposing it as
    `_C._shim_registrations`, is what keeps that from being invisible -- the
    size of that list is the size of the gap.
    """
    registrations: list = []
    module._shim_registrations = registrations

    class _DispatchLibrary:
        __module__ = "torch._C"
        __slots__ = ("kind", "ns", "dispatch_key")

        def __init__(self, kind, ns, dispatch_key, filename="", lineno=0):
            self.kind = kind
            self.ns = ns
            self.dispatch_key = dispatch_key

        def define(self, schema, alias_analysis="", tags=()):
            # `torch/library.py:313` keeps the result and then recomputes the
            # name itself from the schema, so what matters is only that this is
            # the op name -- upstream returns exactly that.
            name = schema.split("(")[0].strip()
            registrations.append(("define", self.ns, name, schema))
            return name

        def impl(self, name, dispatch_key, fn, with_keyset=False):
            registrations.append(("impl", self.ns, name, dispatch_key))

        def impl_with_aoti_compile(self, ns, name, dispatch_key):
            registrations.append(("impl_with_aoti_compile", ns, name, dispatch_key))

        def fallback(self, dispatch_key, fn, with_keyset=False):
            registrations.append(("fallback", self.ns, dispatch_key, None))

        def reset(self):
            return None

        def __getattr__(self, name):
            # The registry has more entry points than the five `torch/library.py`
            # uses (`register_ad_inplace_or_view_fallback`, and others reached
            # only from `torch/_library/`). They all record and return nothing,
            # so they are answered here rather than transcribed one at a time --
            # and every one of them lands in `_shim_registrations`, so the list
            # still measures the gap.
            if name.startswith("__") and name.endswith("__"):
                raise AttributeError(name)

            def recorder(*args, **kwargs):
                registrations.append((name, self.ns, args and args[0], None))

            return recorder

    module._dispatch_library = _DispatchLibrary


# ---------------------------------------------------------------------------
# `torch._C._nn` -- the three names a model path actually calls
# ---------------------------------------------------------------------------
#
# The submodule loop above gives `_nn` all 70 names the stubs declare, every
# one of them a raising stub. docs/NN_SURFACE.md measured which of them a
# 2-layer Llama forward plus greedy `generate` calls, on upstream torch 2.13.0,
# by wrapping each builtin rather than by reading the tree:
#
#     _C._nn.linear                          150 calls
#     _C._nn.scaled_dot_product_attention      10
#     _C._nn.silu                              20
#     ...and nothing else. 3 of 96.
#
# So this is not a port of `_C._nn`; it is the three names, and the other 67
# stay raising stubs on purpose (rule 2 -- a hole, never a switch).
#
# WHY THESE ARE PYTHON-LEVEL COMPOSITIONS AND NOT NEW KERNELS. `aten::linear`
# and `aten::dropout` are `CompositeImplicitAutograd` upstream: they have no
# kernel at all, and what a `TorchDispatchMode` logger sees under
# `F.linear(...)` is `t` + `addmm`/`mm` + `view`, never `aten.linear.default`.
# Reproducing that decomposition here is therefore *following* upstream rather
# than routing around it, and it keeps DESIGN.md §6's single door: every step
# below goes through `_C._aten_dispatch` with a key the golden harness already
# compares against upstream. Nothing here computes.
#
# `scaled_dot_product_attention` is the opposite shape -- upstream picks a
# backend and calls one fused op -- so it is a selection, not a decomposition.


def _install_nn(module, dispatch) -> None:
    nn = module._nn
    reachable = frozenset(module._aten_all_implemented())

    def _t(weight):
        return dispatch("aten.t.default", weight)

    # `aten::linear`, transcribed from what upstream *does*, measured branch by
    # branch against torch 2.13.0 (docs/NN_SURFACE.md §4):
    #
    #     2-D  + bias      t, addmm
    #     N-D  + bias      view, t, addmm, view          (contiguous input)
    #     any  + bias      t, ..., matmul, ..., add      (non-contiguous input)
    #     2-D, no bias     t, mm
    #     N-D, no bias     t, view, mm, _unsafe_view     (i.e. plain matmul)
    #
    # The no-bias case is exactly `matmul(input, weight.t())` in every rank, so
    # it is spelled that way -- `aten.matmul.default` has a kernel here and it
    # is the op upstream's own `at::native::linear` calls.
    #
    # PATCHED, AND SAY SO: the bias branches want `aten.addmm.default`, which
    # this shim has no kernel for. Until it does, bias goes through
    # `matmul` + `add.Tensor` -- which is a real upstream path (it is the
    # non-contiguous branch above), just taken under a condition upstream would
    # not take it under. The numbers agree to float rounding; the accumulation
    # does not, because `addmm` is a fused GEMM and this is a GEMM followed by
    # a separate broadcast add. The `if` below is written against
    # `_aten_all_implemented()` rather than hardcoded, so the day an `addmm`
    # kernel lands this takes upstream's path with no edit here.
    _HAS_ADDMM = "aten.addmm.default" in reachable

    def linear(input, weight, bias=None):
        wt = _t(weight)
        if bias is None:
            return dispatch("aten.matmul.default", input, wt)
        if _HAS_ADDMM and input.dim() == 2:
            return dispatch("aten.addmm.default", bias, input, wt)
        if _HAS_ADDMM and input.dim() > 2 and input.is_contiguous():
            sizes = tuple(input.shape)
            flat = dispatch("aten.view.default", input, [-1, sizes[-1]])
            out = dispatch("aten.addmm.default", bias, flat, wt)
            return dispatch(
                "aten.view.default", out, list(sizes[:-1]) + [tuple(out.shape)[-1]]
            )
        return dispatch(
            "aten.add.Tensor", dispatch("aten.matmul.default", input, wt), bias
        )

    def silu(input):
        return dispatch("aten.silu.default", input)

    # Upstream's `scaled_dot_product_attention` is a *backend selection*, and
    # the selection was measured rather than assumed (docs/NN_SURFACE.md §5).
    # On CPU, 4-D float inputs with `dropout_p == 0` go to
    # `aten._scaled_dot_product_flash_attention_for_cpu` -- for float32,
    # float64, float16 and bfloat16 alike, with or without a mask, with or
    # without `is_causal`, and (measured) with both at once. Everything else
    # falls back to the math backend, which is a different op sequence
    # (`mul.Scalar`, `expand`, `view`, `bmm`, `_safe_softmax`, ...).
    #
    # Only the flash path is wired. The math fallback is refused by name rather
    # than approximated, because `aten._safe_softmax.default` has no kernel
    # here and silently substituting a plain softmax would differ from upstream
    # exactly on the fully-masked rows that `_safe_softmax` exists for.
    #
    # The aten op returns `(output, logsumexp)`; upstream's binding returns the
    # first. `logsumexp` is dropped here for the same reason upstream drops it:
    # it is a backward-pass residual and this shim has no backward.
    _FLASH = "aten._scaled_dot_product_flash_attention_for_cpu.default"

    def scaled_dot_product_attention(
        query,
        key,
        value,
        attn_mask=None,
        dropout_p=0.0,
        is_causal=False,
        *,
        scale=None,
        enable_gqa=False,
    ):
        if dropout_p != 0.0:
            raise NotImplementedError(
                "not implemented in torch._C shim: "
                "scaled_dot_product_attention(dropout_p != 0) -- upstream drops to "
                "the math backend here, which needs aten._safe_softmax.default, "
                "aten.bernoulli_.float and aten.div_.Scalar; none has a kernel"
            )
        if query.dim() != 4:
            raise NotImplementedError(
                "not implemented in torch._C shim: "
                f"scaled_dot_product_attention on a {query.dim()}-D query -- upstream "
                "drops to the math backend for anything but 4-D {B, H, T, K}, which "
                "needs aten._safe_softmax.default; it has no kernel"
            )
        if attn_mask is not None and attn_mask.dtype == module.bool:
            # Measured: upstream converts a bool mask to an additive float one
            # with `scalar_tensor` + `where.self` and *then* calls flash.
            raise NotImplementedError(
                "not implemented in torch._C shim: "
                "scaled_dot_product_attention(attn_mask=<bool tensor>) -- upstream "
                "converts it with aten.scalar_tensor.default and aten.where.self "
                "before calling flash attention; neither has a kernel"
            )
        if enable_gqa:
            # Upstream hands the mismatched head counts straight to the flash
            # op, which broadcasts them. This shim's kernel does not.
            # transformers' Llama calls `repeat_kv` itself and leaves this
            # False, which is why it is refused rather than implemented.
            raise NotImplementedError(
                "not implemented in torch._C shim: "
                "scaled_dot_product_attention(enable_gqa=True) -- upstream's flash "
                "kernel broadcasts the key/value head dimension internally; this "
                "shim's does not. Repeat the heads before calling."
            )
        return dispatch(
            _FLASH,
            query,
            key,
            value,
            dropout_p,
            is_causal,
            attn_mask=attn_mask,
            scale=scale,
        )[0]

    for fn, name in (
        (linear, "linear"),
        (silu, "silu"),
        (scaled_dot_product_attention, "scaled_dot_product_attention"),
    ):
        fn.__name__ = fn.__qualname__ = name
        fn.__module__ = "torch._C._nn"
        setattr(nn, name, fn)

    # Readable for the same reason as `_shim_overloads`: which of `_nn`'s 70
    # names does something should be answerable by asking.
    module._shim_nn_implemented = ["linear", "scaled_dot_product_attention", "silu"]


def _install_composites(module, varfns, dispatch) -> None:
    """`torch.dropout`, the other `CompositeImplicitAutograd` on the path.

    `torch/nn/functional.py:1491` is `_VF.dropout(input, p, training)`, so
    every `nn.Dropout` in a model reaches `torch._C._VariableFunctions.dropout`
    -- 10 times in the measured Llama `generate` (docs/NN_SURFACE.md §3).

    It is not in `overloads.json` because it is not an overload set in the
    sense that table means. `aten::dropout`'s body short-circuits before the
    dispatcher: `if (p == 0 || !train || input.numel() == 0) return input;`.
    Measured, and the measurement is the point -- `F.dropout(x, 0.0, False)`
    produces *no* aten record at all on upstream torch. An eval-mode model
    therefore needs no dropout kernel, and routing this name through the
    overload table would have invented a requirement upstream does not have.

    `train=True` still resolves to `aten.dropout.default` and still raises from
    the one door, naming the kernel an inference-only shim does not need yet.
    """

    def dropout(input, p, train):
        if p == 0 or not train or input.numel() == 0:
            return input
        return dispatch("aten.dropout.default", input, p, train)

    def dropout_(input, p, train):
        if p == 0 or not train or input.numel() == 0:
            return input
        return dispatch("aten.dropout_.default", input, p, train)

    for fn, name in ((dropout, "dropout"), (dropout_, "dropout_")):
        fn.__name__ = fn.__qualname__ = name
        fn.__module__ = "torch._C"
        setattr(varfns, name, fn)


def _install_behaviour(module, dispatch) -> None:
    """The names that have to *do* something for the import to finish."""

    def _add_docstr(obj, doc):
        # `torch/_tensor.py` and `torch/_torch_docs.py` call this thousands of
        # times at import. Upstream returns the object; the callers rely on it.
        try:
            obj.__doc__ = doc
        except (AttributeError, TypeError):
            pass
        return obj

    module._add_docstr = _add_docstr

    def _initExtension(shm_manager_path):
        """VENDOR.md wall 2 -- the wall strict mode used to stop at.

        Upstream's does two things worth reproducing: it writes the dtype,
        layout, memory_format and qscheme instances into the *`torch`* module
        (wall 7 -- they are not attributes of `_C` upstream at all), and it
        takes the `torch_shm_manager` path that `_manager_path()` insisted on
        (wall 4) without running it.
        """
        torch_module = sys.modules.get("torch")
        if torch_module is None:
            return
        # Wall 20, and the one that decides whether a model can have
        # parameters at all. Upstream's `_C` never returns a bare `TensorBase`
        # -- `THPVariable_Wrap` instantiates `THPVariableClass`, which is
        # `torch._tensor.Tensor` -- and `torch/nn/parameter.py:54` branches on
        # `type(data) is torch.Tensor`. Get that wrong and `Parameter(...)`
        # takes its custom-tensor path, returns something that is not a
        # `Parameter`, and `nn.Module.__setattr__` files it as a plain
        # attribute: the model builds and has no parameters.
        #
        # This is the right moment for it, and it is upstream's own moment:
        # `torch/__init__.py:1931` runs `from torch._tensor import Tensor` and
        # `:2189` calls this function, so the class exists and no tensor has
        # been made yet.
        tensor_cls = getattr(torch_module, "Tensor", None)
        if tensor_cls is not None:
            module._set_tensor_class(tensor_cls)
        kinds = (module.dtype, module.layout, module.memory_format, module.qscheme)
        for name, value in list(vars(module).items()):
            # Filtered on the *type*, not on the leading underscore. Skipping
            # underscored names looked harmless and dropped `torch._mkldnn`,
            # which is a real layout that `torch/_export/serde/serialize.py:171`
            # reads at import of that module -- past the point this shim was
            # being tested, so it would have surfaced much later.
            if isinstance(value, kinds) and not hasattr(torch_module, name):
                setattr(torch_module, name, value)

    module._initExtension = _initExtension

    # VENDOR.md walls 15 and 16: neither of these is behind a `hasattr` gate.
    # `torch/autograd/__init__.py:653` raises if `_autograd_init()` is falsy
    # and `torch/jit/__init__.py:315` does the same for `_jit_init()`, so
    # "inference only, no backward" (DESIGN.md §3 stage 0) does not extend to
    # "do not import autograd". Returning True claims only that initialisation
    # succeeded, which for a shim with no autograd state is true.
    module._autograd_init = lambda: True
    module._jit_init = lambda: True
    module._init_names = lambda *a, **k: None

    def _multiprocessing_init():
        """VENDOR.md wall 17 -- C writing into a *Python* package's namespace.

        `torch/multiprocessing/__init__.py:37` calls this and `spawn.py:14`
        then imports `_prctl_pr_set_pdeathsig` from the package. That name is
        defined nowhere in the vendored source; only running the tree reveals
        the requirement.
        """
        mp = sys.modules.get("torch.multiprocessing")
        if mp is not None and not hasattr(mp, "_prctl_pr_set_pdeathsig"):
            mp._prctl_pr_set_pdeathsig = _Unimplemented(
                "torch.multiprocessing._prctl_pr_set_pdeathsig"
            )

    module._multiprocessing_init = _multiprocessing_init

    def _set_generator_metaclass(meta):
        """VENDOR.md wall 19, which turns out to be a supported hook.

        The wall was read as a bidirectional coupling -- `_C` having to import
        a metaclass out of the vendored Python tree. It is that, but upstream
        arranged it deliberately: `torch/_prims/rng_prims.py:411` says so in a
        comment ("Late-bind OpaqueBaseMeta as Generator's metaclass ... to
        avoid making torch._C depend on torch._opaque_base at init time") and
        calls this setter. So there is nothing to invert: `_C` exposes the
        setter and the tree hands the metaclass in.

        `Generator` is synthesised with `_ShimMeta`, a heap type, so the
        `__class__` assignment is legal. It would not be if `Generator` were a
        PyO3 type -- `type` is a static type and cannot be reassigned away
        from (measured).
        """
        module.Generator.__class__ = meta

    module._set_generator_metaclass = _set_generator_metaclass

    _install_dispatch_keys(module)
    _install_library(module)

    # -- op registry ------------------------------------------------------
    def _jit_get_operation(qualname):
        if "::" not in qualname:
            raise RuntimeError(f"torch._C shim: not a qualified op name: {qualname}")
        name = qualname.split("::", 1)[1]
        if _is_refused_op_name(name):
            raise RuntimeError(f"torch._C shim: no operator {qualname}")
        return _op_callable(dispatch, qualname, ""), ["default"]

    def _get_operation_overload(qualname, overload):
        name = qualname.split("::", 1)[-1]
        if _is_refused_op_name(name):
            return None
        op = _op_callable(dispatch, qualname, overload)

        def op_dk(dispatch_key, *args, **kwargs):
            return op(*args, **kwargs)

        return op, op_dk, []

    def _get_schema(qualname, overload):
        return _Schema(qualname, overload)

    # A plain function, not `_Schema.parse` itself: `torch/__init__.py:1091`
    # walks every public name in `dir(_C)` and assigns `__module__` on each
    # callable, and a bound classmethod has no writable `__module__`.
    def parse_schema(text):
        return _Schema.parse(text)

    module.parse_schema = parse_schema
    module.FunctionSchema = _Schema
    module.Argument = _Argument
    module.AliasInfo = _AliasInfo

    # -- names discovered by running the tree ------------------------------
    #
    # Everything below is here because `import torch` demanded it, at the file
    # and line noted. None of it is reachable by reading the vendored source
    # for `torch._C.<name>`: these are the calls whose *return value* is used,
    # as opposed to the vast majority which are only looked up.
    for name, value in _DISCOVERED_RETURNS.items():
        setattr(module, name, _constant_function(f"torch._C.{name}", value))
    for name, value in _BUILD_FLAGS.items():
        setattr(module, name, value)
    for dotted, value in _DISCOVERED_TYPE_RETURNS.items():
        type_name, member = dotted.split(".", 1)
        owner = getattr(module, type_name, None)
        if owner is not None:
            setattr(owner, member, staticmethod(
                _constant_function(f"torch._C.{dotted}", value)))

    module._jit_get_operation = _jit_get_operation
    module._get_operation_overload = _get_operation_overload
    module._get_schema = _get_schema

    _install_default_generator(module)


def _install_default_generator(module) -> None:
    """`torch.default_generator` -- an object with state, not a placeholder.

    `torch/random.py:24` is `from torch._C import default_generator` and every
    function in that module is a method call on it, so the name has to exist
    before `import torch` finishes. It has existed all along as an
    `_Unimplemented`; what changes here is that the CPU RNG behind it is real
    (`rng.rs`), so `manual_seed` can mean what it means upstream -- the same
    seed gives the same numbers as torch, bit for bit for `uniform_`.

    It is an *instance of the synthesised `Generator` class*, not a new type,
    for two reasons that both bite:

      * `_TypeChecker._base` answers the schema's `Generator?` parameter with
        `isinstance(value, module.Generator)`, so anything else would fail to
        bind `tensor.uniform_(a, b, generator=torch.default_generator)`.
      * `Generator` has to stay a heap type -- `_set_generator_metaclass`
        reassigns its `__class__` (VENDOR.md wall 19), which a PyO3 static type
        cannot do.

    Three methods are real and the rest keep refusing. `get_state`/`set_state`
    are the notable absence: upstream's is a 5056-byte legacy blob whose layout
    is the MT19937 struct itself (docs/RNG.md §1.1), and exchanging that blob
    with real torch is a separate piece of work from reproducing the stream.
    Left refusing by name rather than given a format of our own, because a
    round-trippable-but-incompatible blob is the kind of thing that looks like
    interop until someone tries it.
    """
    generator = module.Generator()

    # Read by `generator_arg` in aten.rs: it is how a kernel tells "the default
    # generator was named explicitly" from "some other generator was", and the
    # second is refused rather than quietly served from this stream.
    generator._shim_is_default_generator = True

    def manual_seed(seed):
        module._shim_manual_seed(int(seed))
        return generator

    def seed():
        return module._shim_reseed()

    def initial_seed():
        return module._shim_initial_seed()

    generator.manual_seed = manual_seed
    generator.seed = seed
    generator.initial_seed = initial_seed
    # `.device` is left alone on purpose: the synthesised `Generator` carries
    # it as a *property* (it is one on upstream's type too), and a property has
    # no setter, so an instance attribute cannot shadow it. Overwriting the
    # class attribute would change it for every `Generator`, which is the wrong
    # shape for a per-instance value -- and nothing reads it on this path.

    module.default_generator = generator
