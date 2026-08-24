"""Drive the vendored torch tree against our `_C` and record where it stops.

This is the experiment DESIGN.md §2 has never had run against it. Every number
in IMPORT_WALLS.md 3차/5차 was taken with *real* torch underneath; nothing has
ever checked what happens when the Python tree is real and `_C` is ours.

Two modes, and the difference between them is the whole point:

  strict   Load our `_C` as `torch._C` and let `import torch` fail wherever it
           fails. This is the honest answer to "how far do we get". Nothing is
           faked, so nothing can be over-claimed.

  record   Attach a module-level `__getattr__` to our `_C` that logs every name
           the Python tree asks for and hands back a permissive placeholder.
           This does NOT get us further in any real sense -- a placeholder is
           not an implementation. It is a *measuring instrument*: it answers
           "how much `torch._C` surface does the vendored tree demand, and of
           what kind", which is the number §2 needs and one run of strict mode
           cannot produce (strict mode reports one name per run).

Judgement is by exit code, never by scraping output. IMPORT_WALLS 2차 records
this project losing a whole round to a `grep -q MODEL_OK` that matched the
traceback echoing its own source line.
"""

from __future__ import annotations

import argparse
import importlib.machinery
import importlib.util
import json
import os
import sys
import types
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))

# This script used to sit *inside* the vendored tree's directory, so `HERE` and
# the tree were the same place. They are not any more: the tree lands in the
# package, at torchnative/src/main/, so that one build sees both `torch` and
# `torchnative` as top-level packages, while the scripts that assemble it stay
# in vendor/. Same override the shell scripts take.
VENDOR = os.environ.get(
    "TORCHNATIVE_VENDOR_DIR",
    os.path.join(os.path.dirname(HERE), "torchnative", "src", "main"),
)
SHIM = os.path.join(VENDOR, "torch", "_C.abi3.so")

# Names the Python tree uses as *capability probes* rather than as functions:
#
#     torch/distributed/__init__.py:28
#         def is_available(): return hasattr(torch._C, "_c10d_init")
#
# The tree asks whether `_C` has the name at all, and a subsystem switches off
# when it does not. Seeding these would flip subsystems on that our `_C` has no
# business claiming, and the run would then fail for reasons that say nothing
# about us -- `_c10d_init()` returning a falsy placeholder makes
# `torch/distributed/__init__.py:38` raise, which is an artefact of the
# instrument, not a wall.
#
# So they are deliberately withheld, which is also what the real `_C` will do.
# This is a finding in its own right: omission is the supported way to disable a
# subtree, and IMPORT_WALLS 4차 could not use it because it kept upstream's `_C`
# and tried to cut at the module level instead.
CAPABILITY_PROBES = frozenset({
    "_c10d_init",
    "_rpc_init",
    "_dist_autograd_init",
    "_faulty_agent_init",
    "_mtia_init",
    "_cuda_init",
})

# The opposite of CAPABILITY_PROBES: no `hasattr` gate, and a falsy return is
# fatal.
#
#     torch/autograd/__init__.py:653
#         if not torch._C._autograd_init():
#             raise RuntimeError("autograd initialization failed")
#
# Autograd is therefore *not* optional at import time, even for a build that
# will only ever do inference. Worth stating plainly, because "we do not need
# backward on device" is a live assumption in this project (DESIGN.md §3's
# stage 0) and it does not extend to importing torch.
# `torch/jit/__init__.py:315` is the same shape and equally unconditional --
# `torch.jit` is reached from `torch.distributions`, which
# `torch/__init__.py:2298` imports outright.
MANDATORY_INITS = frozenset({
    "_autograd_init",
    "_jit_init",
    "_init_names",
})


# Dunders cannot be refused wholesale on a `_C` *type* the way they can on the
# module: upstream `TensorBase` really does define `__idiv__`, `__ilshift__` and
# the rest of the in-place operator set, and `_tensor.py` rebinds them by name.
# What must still be refused is the handful the *interpreter itself* probes --
# answering those with a placeholder makes the class machinery behave in ways
# that have nothing to do with torch, and the run stops meaning anything.
INTERPRETER_PROBES = frozenset({
    "__abstractmethods__", "__mro_entries__", "__set_name__",
    "__init_subclass__", "__class_getitem__", "__isabstractmethod__",
    "__slots__", "__wrapped__", "__signature__", "__objclass__",
    "__get__", "__set__", "__delete__", "__origin__", "__args__",
    "__typing_subst__", "__no_type_check__", "__type_params__",
    # `typing` asks these of anything used as a type argument, and
    # `torch/nn/common_types.py:41` puts `Tensor` inside a `Union[...]`.
    "__typing_unpacked_tuple_args__", "__typing_is_unpacked_typevartuple__",
    "__typing_prepare_subst__", "__unpacked__", "__parameters__",
    "__module__", "__name__", "__qualname__", "__bases__", "__mro__",
})


def load_shim_as_torch_C():
    """Put our artefact into `sys.modules` under the name `torch._C`.

    Pre-seeding rather than letting `torch/__init__.py` find it on disk: it is
    the only way to get a handle on the module object *before* the Python tree
    starts pulling names out of it, which `record` mode needs. `from torch._C
    import *` then finds the entry already present and does not touch the file.

    The last component of the name has to stay `_C` -- CPython derives the init
    symbol (`PyInit__C`) from it, and that is what PyO3 exported.
    """
    if not os.path.exists(SHIM):
        raise SystemExit(f"no shim at {SHIM} -- run vendor/install_shim.sh")
    loader = importlib.machinery.ExtensionFileLoader("torch._C", SHIM)
    spec = importlib.util.spec_from_file_location("torch._C", SHIM, loader=loader)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules["torch._C"] = module
    return module


class Placeholder:
    """Answers to as many protocols as a name can be asked for at import time.

    IMPORT_WALLS 4차 hit exactly two of these before the wall that ended it --
    something had to be inheritable, and something had to behave as a
    container. Rather than rediscover them one exception at a time, this
    answers the cheap ones up front so the run reaches the *expensive* wall,
    which is the one worth reporting.
    """

    def __init__(self, name):
        self._name = name

    def __call__(self, *a, **k):
        return Placeholder(f"{self._name}()")

    def __getattr__(self, item):
        if item.startswith("__") and item.endswith("__"):
            raise AttributeError(item)
        return Placeholder(f"{self._name}.{item}")

    def __getitem__(self, item):
        return Placeholder(f"{self._name}[...]")

    def __setitem__(self, item, value):
        pass

    def __contains__(self, item):
        return False

    def __iter__(self):
        return iter(())

    def __len__(self):
        return 0

    def __bool__(self):
        return False

    # `torch/_prims_common/__init__.py:90` reaches for
    # `torch.Tensor.is_sparse.__get__`. A good part of upstream's `TensorBase`
    # surface is getset descriptors rather than methods -- `is_sparse`, `grad`,
    # `shape`, `data` -- and the Python tree pulls the descriptor object out and
    # keeps it. Being callable is not enough; the placeholder has to be one.
    def __get__(self, obj, objtype=None):
        return self

    def __or__(self, other):
        return Placeholder(f"{self._name}|...")

    def __ror__(self, other):
        return Placeholder(f"...|{self._name}")

    def __repr__(self):
        return f"<placeholder {self._name}>"


def attach_recorder(module, demanded):
    """PEP 562 module `__getattr__`, installed on an extension module.

    Works because `module_getattro` looks the hook up in the module's `__dict__`
    like any other module -- being a compiled module changes nothing here.
    """
    real = set(vars(module))

    def __getattr__(name):
        if name in real:  # pragma: no cover -- CPython would not call us
            return vars(module)[name]
        demanded.setdefault(name, 0)
        demanded[name] += 1
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        if name in CAPABILITY_PROBES:
            raise AttributeError(name)  # see CAPABILITY_PROBES
        # A type, so that `class X(torch._C.Foo)` and `x: torch._C.Foo` both
        # work -- IMPORT_WALLS categories 2 and 4. Leading underscores are
        # stripped first: `torch/cuda/green_contexts.py:18` subclasses
        # `torch._C._GreenContext`, and a non-class base makes Python treat the
        # object's *type* as the metaclass, which fails somewhere unrelated.
        if name.lstrip("_")[:1].isupper():
            return type(name, (), {"__module__": "torch._C"})
        return Placeholder(f"torch._C.{name}")

    module.__getattr__ = __getattr__
    return demanded


def attach_tensorbase_recorder(module, demanded):
    """Record the `TensorBase` member surface the Python tree binds at class time.

    `torch/_tensor.py:102` opens with `class Tensor(torch._C.TensorBase)` and
    then spends hundreds of lines doing `_C._add_docstr(_C.TensorBase.<name>,
    ...)` in the class body. Every one of those is resolved while the class
    statement runs, so the whole set is demanded before `import torch` returns.
    A module-level `__getattr__` cannot see them -- `TensorBase` itself exists,
    so the module hook is never consulted -- hence a second instrument.

    PyO3 pyclasses are heap types, but not writable ones, so the recorder is a
    Python subclass whose metaclass answers for the missing names. Subclassing
    keeps the real members real: only names our `_C` genuinely lacks reach the
    hook.
    """
    real_base = module.TensorBase

    # Dunders cannot be refused wholesale here the way they are on the module:
    # upstream `TensorBase` really does define `__idiv__`, `__ilshift__` and the
    # rest of the in-place operator set, and `_tensor.py` rebinds them by name.
    # What must still be refused is the handful the *interpreter itself* probes
    # -- answering those with a placeholder makes the class machinery behave in
    # ways that have nothing to do with torch, and the run stops meaning
    # anything.
    class _RecordingMeta(type(real_base)):
        def __getattr__(cls, name):
            if name in INTERPRETER_PROBES:
                raise AttributeError(name)
            key = f"TensorBase.{name}"
            demanded[key] = demanded.get(key, 0) + 1
            return Placeholder(f"torch._C.{key}")

    recording = _RecordingMeta("TensorBase", (real_base,), {"__module__": "torch._C"})
    module.TensorBase = recording

    # `torch/nn/parameter.py:26` does `class _ParameterMeta(torch._C._TensorMeta)`
    # and then `class Parameter(torch.Tensor, metaclass=_ParameterMeta)`. That
    # only type-checks if `_TensorMeta` really *is* the metaclass of
    # `TensorBase` -- upstream satisfies `type(TensorBase) is _C._TensorMeta`.
    # Our `_C` will have to export its pyclass metatype under that name.
    module._TensorMeta = _RecordingMeta
    return demanded


def dump_surface(path):
    """Write the *name* surface of a real upstream `torch._C` to JSON.

    Run against an installed upstream torch. Only names and a coarse kind are
    taken -- no code, no values -- because the point is to size the hole our
    `_C` has to fill, not to borrow anything from upstream at runtime.
    """
    # The vendored tree would shadow the installed one, and this mode is the one
    # place that must reach upstream. Drop both the vendor dir and this script's
    # own directory: they used to be the same path and no longer are, so
    # dropping only `HERE` would leave the tree in front of upstream.
    _shadow = {os.path.abspath(HERE), os.path.abspath(VENDOR)}
    sys.path = [p for p in sys.path if os.path.abspath(p or ".") not in _shadow]

    import torch._C as real  # noqa: PLC0415 -- upstream, deliberately

    surface = {}
    for name in dir(real):
        value = getattr(real, name)
        if isinstance(value, type):
            kind = "type"
        elif isinstance(value, __import__("types").ModuleType):
            kind = "module"
        elif callable(value):
            kind = "callable"
        else:
            kind = "value"
        surface[name] = kind
    # Per-type detail. Two things matter and neither can be guessed:
    #
    #   meta      `torch/autograd/variable.py:14` writes
    #             `class Variable(_C._LegacyVariableBase, metaclass=VariableMeta)`
    #             where `VariableMeta(type)`, so that base must have plain `type`
    #             as its metatype -- while `torch/_awaits/__init__.py:12` needs
    #             `type(_C._Await)` to be something *other* than `type`. Giving
    #             every `_C` type the same metatype breaks one or the other.
    #
    #   members   so a seeded type can carry the names the tree binds off it
    #             (`torch.Stream.query`) without a catch-all hook, which is what
    #             forces the metatype question in the first place.
    obj_members = set(dir(object))
    surface["#types"] = {
        name: {
            "meta": type(getattr(real, name)).__name__,
            "members": sorted(set(dir(getattr(real, name))) - obj_members),
            # Members that are instances of their own type -- pybind11 enum
            # values such as `DispatchKey.CPU` and `TransformType.Jvp`.
            # `torch/_ops.py:132` gates on `isinstance(k, TransformType)`, so a
            # stand-in has to be an instance and not merely a distinct object.
            "enum_members": sorted(
                m for m in dir(getattr(real, name))
                if not m.startswith("__")
                and isinstance(getattr(getattr(real, name), m, None),
                               getattr(real, name))
            ),
        }
        for name in dir(real)
        if isinstance(getattr(real, name), type)
    }
    surface["#TensorBase"] = sorted(dir(real.TensorBase))
    # `torch/__init__.py:2212` builds the entire `torch.<op>` namespace by
    # iterating this one object, so its member list is not incidental -- it is
    # the shape of the public torch API.
    surface["#_VariableFunctions"] = sorted(dir(real._VariableFunctions))

    # Names that `_C` writes into the *`torch`* namespace from C, during
    # `_initExtension`. `torch.float32`, `torch.strided`,
    # `torch.contiguous_format` and friends are not attributes of `torch._C`
    # and are not assigned anywhere in the Python tree; upstream's
    # `initializeDtypes()` imports the `torch` module and calls
    # `PyModule_AddObject` on it. Nothing in the vendored tree reveals that they
    # have to exist, which is why they are collected here.
    import torch as real_torch  # noqa: PLC0415

    enum_types = (real_torch.dtype, real_torch.layout,
                  real_torch.memory_format, real_torch.qscheme)
    surface["#torch_namespace"] = {
        name: type(getattr(real_torch, name)).__name__
        for name in dir(real_torch)
        if isinstance(getattr(real_torch, name), enum_types)
    }
    with open(path, "w") as fh:
        json.dump(surface, fh, indent=1)
    print(f"wrote {len(surface) - 1} torch._C names and "
          f"{len(surface['#TensorBase'])} TensorBase members to {path}")


def _seeded_meta(demanded):
    """Stand-in for pybind11's metatype -- see `seed_surface`.

    It also answers for members, because `torch/_torch_docs.py:13691` does
    `_add_docstr(torch.Stream.query, ...)`: a seeded `_C` type is not just a
    name, the tree reaches into it at import time the same way it reaches into
    `TensorBase`.
    """

    class _SeededMeta(type):
        def __getattr__(cls, name):
            if name in INTERPRETER_PROBES:
                raise AttributeError(name)
            key = f"{cls.__name__}.{name}"
            demanded[key] = demanded.get(key, 0) + 1
            if name.lstrip("_")[:1].isupper():
                return cls()   # pybind11 enum value; see `enum_members`
            return Placeholder(f"torch._C.{key}")

    return _SeededMeta


def _seeded_type(name, qualname, demanded, meta, members=(), enum_members=()):
    """A `_C` type that can also be constructed and subclassed.

    `torch/_sources.py:87` is `class SourceContext(SourceRangeFactory)` whose
    `__init__` calls `super().__init__(source, filename, file_lineno,
    leading_whitespace_len)`. So it is not enough for a `_C` type to exist and
    answer for members -- the tree instantiates it, with arguments, while
    `torch.nn.functional` is still importing.
    """
    body = {
        "__module__": qualname.rsplit(".", 1)[0] if "." in qualname else "torch._C",
        "__init__": lambda self, *a, **k: None,
        "__getattr__": lambda self, item: (
            Placeholder(f"{qualname}().{item}")
            if not (item.startswith("__") and item.endswith("__"))
            else (_ for _ in ()).throw(AttributeError(item))
        ),
    }
    # Members are written into the class dict rather than answered by a
    # metaclass hook, so that a type whose upstream metatype is plain `type`
    # can still carry them.
    for member in members:
        if member.startswith("__") and member.endswith("__"):
            continue
        body.setdefault(member, Placeholder(f"{qualname}.{member}"))
    demanded.setdefault(f"type:{qualname}", 0)
    cls = meta(name, (), body)
    for member in enum_members:
        setattr(cls, member, cls())
    return cls


class _CSubmoduleFinder:
    """Synthesise `torch._C.<anything>` on demand.

    `torch/utils/_python_dispatch.py:22` imports `torch._C._dynamo.guards`, so
    the C submodules are themselves packages. Enumerating them is hopeless --
    upstream registers 32 at the top level and an unknown number below -- so the
    instrument answers for the whole namespace instead. Each synthesised module
    carries `__path__ = []`, without which Python refuses at the parent with
    "is not a package" before any finder is consulted.
    """

    def __init__(self, demanded, meta):
        self.demanded = demanded
        self.meta = meta

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith("torch._C."):
            return None
        return importlib.util.spec_from_loader(fullname, self)

    def create_module(self, spec):
        mod = types.ModuleType(spec.name)
        mod.__path__ = []
        mod.__getattr__ = _submodule_recorder(spec.name, self.demanded, self.meta)
        self.demanded[f"submodule:{spec.name}"] = 0
        return mod

    def exec_module(self, module):
        return None


def _submodule_recorder(qualname, demanded, meta):
    # Cached, because identity matters: `torch/_ops.py:132` does
    # `isinstance(k, TransformType)` where both the value and the class were
    # fetched from `torch._C._functorch`. Handing out a fresh class per lookup
    # makes that check fail for a reason that has nothing to do with torch.
    cache: dict[str, object] = {}

    def __getattr__(name):
        if name.startswith("__") and name.endswith("__"):
            raise AttributeError(name)
        key = f"{qualname}.{name}"
        demanded[key] = demanded.get(key, 0) + 1
        if name in cache:
            return cache[name]
        if name.lstrip("_")[:1].isupper():
            # Nothing is known about a synthesised submodule's types, so the
            # metaclass answers for members, and any capitalised member is
            # treated as an enum value (`TransformType.Jvp`) -- that is the
            # convention pybind11 enums in `_C` follow.
            cache[name] = _seeded_type(name, key, demanded, meta)
        else:
            cache[name] = Placeholder(key)
        return cache[name]
    return __getattr__


def seed_surface(module, surface, demanded):
    """Fill every name our `_C` lacks with a placeholder, before `import torch`.

    This exists because a module-level `__getattr__` cannot serve
    `from torch._C import *`, and that star import at `torch/__init__.py:445`
    is how most of the `torch` namespace comes into being. Recording hooks see
    nothing of it.

    Seeding is a *measurement*, not progress: a placeholder `torch.float32` is
    not a dtype. What it buys is the ability to tell breadth-walls (a name is
    missing) apart from depth-walls (the name is there and the tree needs it to
    behave), which is the distinction DESIGN.md §2 actually turns on.
    """
    seeded_meta = _seeded_meta(demanded)
    type_detail = surface.pop("#types", {})
    tensorbase_members = surface.pop("#TensorBase", [])
    varfns = surface.pop("#_VariableFunctions", [])
    namespace = surface.pop("#torch_namespace", {})

    # Model upstream faithfully: these names appear when `_initExtension` runs,
    # because that is when the C side does `PyModule_AddObject` on the `torch`
    # module. Doing it here rather than seeding `torch` afterwards keeps the
    # ordering honest -- `torch/__init__.py` uses some of them below the call.
    def _initExtension(*a, **k):
        demanded["_initExtension()"] = demanded.get("_initExtension()", 0) + 1
        torch_mod = sys.modules["torch"]
        made: dict[str, type] = {}
        for name, type_name in namespace.items():
            # Our `_C` already exports eleven real dtypes through `__all__`, and
            # the star import above put them on `torch`. Those are not holes.
            if hasattr(torch_mod, name):
                continue
            demanded[f"torch.{name}"] = demanded.get(f"torch.{name}", 0) + 1
            if type_name not in made:
                base = getattr(module, type_name, None)
                try:
                    made[type_name] = type(type_name, (base,), {})
                except TypeError:
                    # PyO3 pyclasses are final unless declared `subclass`, so a
                    # stand-in cannot inherit -- `isinstance(x, torch.dtype)`
                    # will not hold for these. Noted rather than papered over.
                    made[type_name] = type(type_name, (), {})
            try:
                value = made[type_name]()
            except TypeError:
                value = Placeholder(f"torch.{name}")
            setattr(torch_mod, name, value)

    module._initExtension = _initExtension
    surface.pop("_initExtension", None)

    # `torch/__init__.py:2212-2224`:
    #
    #     for __name in dir(_C._VariableFunctions):
    #         __obj = getattr(_C._VariableFunctions, __name)
    #         __obj.__module__ = __name__
    #         globals()[__name] = __obj
    #
    # so `torch.add`, `torch.mm`, `torch.full` and 622 more public names are not
    # written anywhere in the Python tree -- they are *harvested* from one `_C`
    # object at import time. Each must be enumerable by `dir()` and each must
    # accept `__module__` being assigned, which rules out builtin functions and
    # bound methods. The instrument therefore hands out real Python functions.
    if varfns:
        ns = {}
        for name in varfns:
            if name.startswith("__"):
                continue

            def _fn(*a, _n=name, **k):
                demanded[f"_VariableFunctions.{_n}()"] = \
                    demanded.get(f"_VariableFunctions.{_n}()", 0) + 1
                return Placeholder(f"torch.{_n}(...)")

            _fn.__name__ = name
            _fn.__qualname__ = name
            ns[name] = _fn
        # Instance attributes, not class attributes: a function reached through
        # a class becomes a bound method, and `method.__module__` is read-only.
        # Upstream gets away with it because these are `builtin_function_or_method`
        # objects that never bind. Anything we build in Rust has to have the
        # same non-binding property.
        holder = types.SimpleNamespace(**ns)
        module._VariableFunctions = holder
        module._VariableFunctionsClass = type(holder)
        surface.pop("_VariableFunctions", None)
        surface.pop("_VariableFunctionsClass", None)

    seeded = 0
    for name, kind in surface.items():
        if hasattr(module, name) or name in CAPABILITY_PROBES:
            continue
        demanded[f"seeded:{name}"] = 0
        if kind == "type":
            # The metatype is copied from upstream rather than chosen, because
            # the tree depends on it in both directions:
            #   `torch/_awaits/__init__.py:12` needs `type(_C._Await)` to differ
            #   from `type`; `torch/autograd/variable.py:14` needs
            #   `type(_C._LegacyVariableBase)` to *be* `type`.
            detail = type_detail.get(name, {})
            meta = type if detail.get("meta", "type") == "type" else seeded_meta
            setattr(module, name,
                    _seeded_type(name, f"torch._C.{name}", demanded, meta,
                                 detail.get("members", ()),
                                 detail.get("enum_members", ())))
        elif kind == "module":
            # `torch/_sources.py:9` does `from torch._C._jit_tree_views import
            # SourceRangeFactory`. That is an *import statement*, so an
            # attribute is not enough -- Python looks in `sys.modules` (`_C` is
            # a compiled module and therefore not a package, so there is no
            # `__path__` to search). Upstream's `_C` registers 32 such
            # submodules from C. Ours will have to as well; PyO3 spells this
            # `PyModule::new` + an explicit `sys.modules` insert.
            sub = types.ModuleType(f"torch._C.{name}")
            sub.__path__ = []
            sub.__getattr__ = _submodule_recorder(f"torch._C.{name}", demanded,
                                                  seeded_meta)
            setattr(module, name, sub)
            sys.modules[f"torch._C.{name}"] = sub
        elif name == "_jit_get_operation":
            # TORCH_C.md §5-4 listed this as the entrance whose *shape* could
            # not be settled without running step 1. This is step 1, and the
            # shape is now known: it returns the pair `(op, overload_names)`,
            # `op` must be callable with a settable `__module__`, and
            # `torch/_ops.py:1415` hands it straight to
            # `torch.jit._builtins._register_builtin`. `torch/fx/node.py:102`
            # resolves `torch.ops.aten._assert_async.msg` while `import torch`
            # is still running, so this is not a lazily-reached API.
            def _jit_get_operation(qualname, *a, **k):
                demanded[f"_jit_get_operation:{qualname}"] = \
                    demanded.get(f"_jit_get_operation:{qualname}", 0) + 1

                def op(*args, **kwargs):
                    return Placeholder(f"{qualname}(...)")

                op.__name__ = qualname.replace("::", "_")
                return op, ["default"]
            setattr(module, name, _jit_get_operation)
        elif name == "_get_operation_overload":
            # `torch/_ops.py:1238` -- the second half of the same entrance, and
            # it returns a *triple*: `(op, op_dk, tags)`. Together with
            # `_get_schema` this is what `torch.ops.aten.add.Tensor` is made of.
            def _get_operation_overload(qualname, overload, *a, **k):
                key = f"_get_operation_overload:{qualname}.{overload or 'default'}"
                demanded[key] = demanded.get(key, 0) + 1

                def op(*args, **kwargs):
                    return Placeholder(f"{qualname}(...)")

                def op_dk(dk, *args, **kwargs):
                    return Placeholder(f"{qualname}[{dk}](...)")

                return op, op_dk, []
            setattr(module, name, _get_operation_overload)
        elif name == "_multiprocessing_init":
            # The second instance of "C writes into a Python module's
            # namespace", after `_initExtension`. `torch/multiprocessing/
            # __init__.py:37` calls this and then `spawn.py:14` imports
            # `_prctl_pr_set_pdeathsig` from the package -- a name that appears
            # nowhere in the vendored source. Grep will never find these; only
            # running the tree does.
            def _multiprocessing_init(*a, **k):
                demanded["_multiprocessing_init()"] = \
                    demanded.get("_multiprocessing_init()", 0) + 1
                mp = sys.modules.get("torch.multiprocessing")
                if mp is not None:
                    mp._prctl_pr_set_pdeathsig = Placeholder(
                        "torch.multiprocessing._prctl_pr_set_pdeathsig")
            setattr(module, name, _multiprocessing_init)
        elif name in MANDATORY_INITS:
            def _init(*a, _n=name, **k):
                demanded[f"{_n}()"] = demanded.get(f"{_n}()", 0) + 1
                return True
            setattr(module, name, _init)
        else:
            setattr(module, name, Placeholder(f"torch._C.{name}"))
        seeded += 1

    # PyO3 0.29 emits `__all__` on `#[pymodule]` modules, so `from torch._C
    # import *` copies only what the crate declared -- setting an attribute is
    # not enough to make it visible to the star import that builds most of the
    # `torch` namespace. Worth knowing outside this instrument too: whatever we
    # want re-exported from the real `_C` has to be *registered*, not merely
    # reachable.
    if hasattr(module, "__all__"):
        module.__all__ = sorted(set(module.__all__) | {
            n for n in surface if not n.startswith("_")
        })
    return seeded, tensorbase_members


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("strict", "record"), default="strict")
    ap.add_argument("--dump-surface", default=None,
                    help="run under upstream torch and write its _C name surface here")
    ap.add_argument("--surface", default=None,
                    help="seed missing _C names from this surface JSON (record mode only)")
    ap.add_argument("--target", default="torch",
                    help="what to import: torch | transformers | from_config")
    ap.add_argument("--report", default=None, help="write demanded names as JSON here")
    args = ap.parse_args()

    if args.dump_surface:
        dump_surface(args.dump_surface)
        return 0

    module = load_shim_as_torch_C()
    demanded: dict[str, int] = {}
    if args.mode == "record":
        if args.surface:
            sys.meta_path.insert(0, _CSubmoduleFinder(demanded, _seeded_meta(demanded)))
            with open(args.surface) as fh:
                seeded, _ = seed_surface(module, json.load(fh), demanded)
            print(f"seeded {seeded} placeholder names into torch._C", file=sys.stderr)
        attach_recorder(module, demanded)
        attach_tensorbase_recorder(module, demanded)

    status = 0
    try:
        if args.target == "torch":
            import torch
            print("imported torch from", torch.__file__)
            print("torch.__version__ =", torch.__version__)
        elif args.target == "transformers":
            import transformers
            print("transformers", transformers.__version__)
            from transformers.utils import import_utils
            print("is_torch_available()", import_utils.is_torch_available())
        elif args.target == "from_config":
            from transformers import AutoConfig, AutoModelForCausalLM
            cfg = AutoConfig.for_model(
                "llama", hidden_size=64, intermediate_size=128,
                num_hidden_layers=2, num_attention_heads=4,
                num_key_value_heads=4, vocab_size=128,
            )
            model = AutoModelForCausalLM.from_config(cfg)
            print("built", type(model).__name__)
    except BaseException:
        traceback.print_exc()
        status = 1

    if demanded:
        print(f"\n--- torch._C names demanded: {len(demanded)} ---", file=sys.stderr)
        for name in sorted(demanded):
            print(f"  {name} x{demanded[name]}", file=sys.stderr)
    if args.report:
        with open(args.report, "w") as fh:
            json.dump({"mode": args.mode, "target": args.target,
                       "status": status, "demanded": demanded}, fh, indent=1)
    return status


if __name__ == "__main__":
    sys.exit(main())
