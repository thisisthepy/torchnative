"""Which overload `torch.ops.aten.<name>(...)` calls when the packet has no
`default` overload (`bootstrap.py::_packet_overload`;
docs/design/FUNCTIONALIZATION.md section 5.5).

Upstream's packet tries each overload's schema in its registration order and
calls the first that accepts the arguments by count, keyword names and kinds
(TorchScript's `getOpWithStack`); when none does it raises `RuntimeError`
with one error per overload. The shim binds by the same count and keyword
rules and checks kinds with `_TypeChecker`'s predicates; it calls the one
overload that binds, and refuses by name when several do.

The packets compared are computed, not listed: every aten name with an
overload in `torch._C._aten_all_implemented()` whose shim packet has no
`default` (printed). `CALLS` must have a row for each, so a new packet of that
kind turns `test_every_packet_without_a_default_has_a_row` red until it gets
one. **Every expectation is upstream's answer**, computed in a separate
subprocess from the same probe text: upstream's overload is
`torch._C._jit_resolve_packet` (cross-checked against what a
`TorchDispatchMode` records when the mode sees the packet's op), the shim's
is what the same mode records on the shim, or, for an op the mode does not see
(a CompositeImplicitAutograd decomposition), `torch._C._shim_packet_overload`.

What each test turns red under (nullifications, none run locally; CI runs
them):

* `_packet_overload` answering the old `<name>.default` key, or HEAD~1's
  arity-only rule: `test_packets_resolve_upstreams_overload` (every
  `.default` row; the arity rule leaves `add_`, `pow`, `bitwise_*`, ... with
  several candidates and refuses them);
* the kind check removed from `_packet_bind` (`predicate` treated as always
  true): `test_packets_resolve_upstreams_overload` (two-overload rows such as
  `rsub(t, 2)` become ambiguous) and `test_kind_mismatches_refuse_as_upstreams`
  (the calls bind instead of raising);
* the `number` narrowing removed (`_TypeChecker`'s `Scalar`, which takes a
  0-dim tensor): the `fill_`, `masked_fill` and `masked_fill_` rows with a
  0-dim tensor value become ambiguous and refuse;
* the `Tensor?[]` element rule removed from `_TypeChecker._base_predicate`:
  the `_unsafe_index` and `index` rows (the predicate raises, the overload is
  undecided and the call refuses by name);
* `_packet_numbers_as_tensors` emptied: the `add_`, `sub_`,
  `mul_`, `div_` and `multiply` number rows of
  `test_packets_resolve_upstreams_overload` (a number then binds `.Scalar`
  instead of `.Tensor`, where upstream takes `.Tensor`);
* `_PACKET_OVERLOAD_ORDER` emptied: `multiply|number` (where the yaml declares
  `.out` between `.Tensor` and `.Scalar`) or any two-overload call on a packet
  where the yaml puts `.Scalar` ahead of `.Tensor`;
* the TorchScript kind adjustments (`bool` for `int`, `None` for `Tensor`, a
  bare tensor for `Tensor?[]`) removed from `_packet_predicate`:
  `transpose|bool`, `rsub|none_tensor` and `index|bare_tensor` in
  `test_packets_resolve_upstreams_overload`.

Packets that have a `default` overload (section 5.6) are compared too:
every implemented `aten.<name>.default` key with a call built from its
schema (`test_every_implemented_default_packet_resolves_to_default`, red when
the yaml's `""` is read as "no default" again, CI run 43841297493), the
packets that run refused, called for real
(`test_the_ci_packets_dispatch_their_default`), and the measured
two-overload rows (`test_packets_with_a_default_choose_upstreams_overload`,
red when `_PACKET_BEFORE_DEFAULT` is dropped: `range|two`, where both
`step` and `default` bind and upstream takes `step`, its first).

What this cannot see: a representative call is one call per key, so a
default packet whose other overloads bind only other argument forms is
checked on those forms only where `RESOLVE` lists one; overloads upstream
registers that the shim does not list (`select.t`,
`index.str`, `pow.int`, ... come from TorchScript's prim registry and have no
schema here); devices; and whether the result values agree (the op suites own
that; only shape and dtype are compared here).
"""

import json
import os
import subprocess
import sys

import _skip

_REPO_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "python")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")


def _available():
    return os.path.isfile(_VENDOR_SHIM)


_PROBE = r'''
import json
import sys
import torch
from torch.utils._python_dispatch import TorchDispatchMode

IS_SHIM = hasattr(torch._C, "_aten_implemented")
A = torch.ops.aten
out = {"is_shim": IS_SHIM, "impl": None, "packets": [], "rows": {},
       "mismatch": {}, "narrowing": {}}

if IS_SHIM:
    impl = sorted(torch._C._aten_all_implemented())
    out["impl"] = impl
else:
    impl = json.loads(sys.argv[1])
implemented = set(impl)

# Every aten name with an implemented overload whose packet has no `default`,
# from this side's own packets.
for name in sorted({k.split(".")[1] for k in impl if k.startswith("aten.")}):
    try:
        overloads = getattr(A, name).overloads()
    except Exception as exc:  # noqa: BLE001 - recorded, compared below
        out["packets"].append([name, None, f"{type(exc).__name__}: {exc}"])
        continue
    if "default" not in overloads:
        out["packets"].append([name, [o for o in overloads if f"aten.{name}.{o}" in implemented], None])


def failed(exc):
    if isinstance(exc, (KeyboardInterrupt, SystemExit)):
        raise exc
    return {"raised": type(exc).__name__, "message": str(exc)}


def summary(r):
    if isinstance(r, torch.Tensor):
        return {"shape": list(r.shape), "dtype": str(r.dtype)}
    if isinstance(r, (list, tuple)):
        return [summary(x) for x in r]
    return repr(r)


class Record(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.seen = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen.append(str(func))
        return func(*args, **(kwargs or {}))


t = lambda: torch.arange(6.).reshape(2, 3)
b = lambda: torch.tensor([1, 2, 3])
m = lambda: torch.tensor([True, False, True])
i = lambda: torch.tensor([0])
idx = lambda: torch.tensor([[0, 1, 0]])
bd = lambda: torch.tensor([1., 2.])

def lstm_args():
    x = torch.arange(30.).reshape(5, 2, 3) / 30
    h = (torch.zeros(1, 2, 4), torch.zeros(1, 2, 4))
    w = [torch.full((16, 3), 0.1), torch.full((16, 4), 0.1), torch.zeros(16), torch.zeros(16)]
    return (x, h, w, True, 1, 0.0, False, False, False), {}

# packet -> [(label, () -> (args, kwargs))]. Each call is valid upstream.
CALLS = {
    "_unsafe_index": [("list", lambda: ((t(), [i()]), {})),
                      ("none_first", lambda: ((t(), [None, i()]), {}))],
    "add_": [("tensor", lambda: ((t(), t()), {})), ("alpha", lambda: ((t(), t()), {"alpha": 2})),
             ("number", lambda: ((t(), 2), {}))],
    "bernoulli_": [("float", lambda: ((t(), 0.5), {})),
                   ("tensor", lambda: ((t(), torch.full((2, 3), 0.5)), {}))],
    "bitwise_and": [("tensor", lambda: ((b(), b()), {})), ("number", lambda: ((b(), 1), {})),
                    ("number_first", lambda: ((1, b()), {}))],
    "bitwise_or": [("tensor", lambda: ((b(), b()), {})), ("number", lambda: ((b(), 1), {}))],
    "bitwise_xor": [("tensor", lambda: ((b(), b()), {})), ("number", lambda: ((b(), 1), {}))],
    "bucketize": [("tensor", lambda: ((t(), bd()), {})), ("number", lambda: ((1.5, bd()), {}))],
    "div_": [("tensor", lambda: ((t(), t() + 1), {})),
             ("mode", lambda: ((t(), t() + 1), {"rounding_mode": "floor"})),
             ("number", lambda: ((t(), 2), {}))],
    "empty": [("list", lambda: (([2, 3],), {})),
              ("dtype", lambda: (([2],), {"dtype": torch.float64})),
              ("size", lambda: ((torch.Size([2]),), {}))],
    "fill_": [("number", lambda: ((t(), 1), {})), ("zero_dim", lambda: ((t(), torch.tensor(1.)), {}))],
    "greater": [("tensor", lambda: ((t(), t()), {})), ("number", lambda: ((t(), 0), {}))],
    "index": [("list", lambda: ((t(), [i()]), {})), ("none_first", lambda: ((t(), [None, i()]), {})),
              ("keyword", lambda: ((t(),), {"indices": [i()]})),
              ("bare_tensor", lambda: ((t(), i()), {}))],
    "isin": [("tensor", lambda: ((b(), b()), {})), ("number", lambda: ((b(), 1), {}))],
    "lstm": [("input", lstm_args)],
    "masked_fill": [("number", lambda: ((b(), m(), 0), {})),
                    ("zero_dim", lambda: ((b(), m(), torch.tensor(5)), {}))],
    "masked_fill_": [("number", lambda: ((b(), m(), 0), {})),
                     ("zero_dim", lambda: ((b(), m(), torch.tensor(5)), {}))],
    "mul_": [("tensor", lambda: ((t(), t()), {})), ("number", lambda: ((t(), 2.0), {}))],
    "multiply": [("tensor", lambda: ((t(), t()), {})), ("number", lambda: ((t(), 2), {}))],
    "norm": [("dim", lambda: ((t(), 2, [1]), {})), ("keywords", lambda: ((t(),), {"p": 2, "dim": [1]})),
             ("plain", lambda: ((t(),), {}))],
    "pow": [("tensor", lambda: ((t(), t()), {})), ("number", lambda: ((t(), 2), {})),
            ("number_first", lambda: ((2, t()), {}))],
    "repeat_interleave": [("repeats", lambda: ((b(),), {})), ("int", lambda: ((t(), 2), {})),
                          ("one_element_tensor", lambda: ((t(), torch.tensor([2])), {}))],
    "rsub": [("number", lambda: ((t(), 2), {})), ("tensor", lambda: ((t(), t()), {}))],
    "scatter": [("value", lambda: ((t(), 0, idx(), 1.0), {})), ("src", lambda: ((t(), 0, idx(), t()), {}))],
    "scatter_": [("src", lambda: ((t(), 0, idx(), t()), {})), ("value", lambda: ((t(), 0, idx(), 1.0), {}))],
    "scatter_reduce": [("two", lambda: ((t(), 0, idx(), t(), "sum"), {}))],
    "select": [("int", lambda: ((t(), 0, 1), {}))],
    "select_copy": [("int", lambda: ((t(), 0, 1), {}))],
    "slice": [("tensor", lambda: ((t(), 1, 0, 2), {})), ("defaults", lambda: ((t(),), {}))],
    "slice_copy": [("tensor", lambda: ((t(), 1, 0, 2), {}))],
    "split_copy": [("tensor", lambda: ((t(), 1), {}))],
    "sub_": [("tensor", lambda: ((t(), t()), {})), ("number", lambda: ((t(), 1), {}))],
    "transpose": [("int", lambda: ((t(), 0, 1), {})), ("keywords", lambda: ((t(),), {"dim0": 0, "dim1": 1})),
                  ("bool", lambda: ((t(), True, 1), {}))],
    "transpose_copy": [("int", lambda: ((t(), 0, 1), {}))],
    "unbind": [("default_dim", lambda: ((t(),), {})), ("dim", lambda: ((t(), 1), {}))],
    "unbind_copy": [("int", lambda: ((t(), 0), {}))],
    "unflatten": [("tuple", lambda: ((t(), 1, (3, 1)), {}))],
    # Packets upstream gives a `default` that the shim does not list: the
    # TorchScript scalar op (`aten::add(Scalar a, Scalar b) -> Scalar`,
    # `aten::split(Tensor self, int[] split_sizes, int dim=0)`), registered
    # last. The shim's packet has no `default`, so it resolves like the rest.
    "add": [("tensor", lambda: ((t(), t()), {})), ("alpha", lambda: ((t(), t()), {"alpha": 2}))],
    "sub": [("tensor", lambda: ((t(), t()), {}))],
    "mul": [("tensor", lambda: ((t(), t()), {}))],
    "div": [("tensor", lambda: ((t(), t() + 1), {})),
            ("mode", lambda: ((t(), t() + 1), {"rounding_mode": "floor"}))],
    "eq": [("tensor", lambda: ((t(), t()), {})), ("number", lambda: ((t(), 0), {}))],
    "ne": [("tensor", lambda: ((t(), t()), {})), ("number", lambda: ((t(), 0), {}))],
    "lt": [("tensor", lambda: ((t(), t()), {})), ("number", lambda: ((t(), 0), {}))],
    "le": [("tensor", lambda: ((t(), t()), {})), ("number", lambda: ((t(), 0), {}))],
    "gt": [("tensor", lambda: ((t(), t()), {})), ("number", lambda: ((t(), 0), {}))],
    "ge": [("tensor", lambda: ((t(), t()), {})), ("number", lambda: ((t(), 0), {}))],
    "fmod": [("tensor", lambda: ((t(), t() + 1), {})), ("number", lambda: ((t(), 2), {}))],
    "remainder": [("tensor", lambda: ((t(), t() + 1), {})), ("number", lambda: ((t(), 2), {}))],
    "split": [("int", lambda: ((t(), 1), {})), ("sizes", lambda: ((t(), [1, 1]), {}))],
}
out["calls"] = sorted(CALLS)


def resolved(name, make, seen):
    for s in seen:
        parts = s.split(".")
        if len(parts) == 3 and parts[0] == "aten" and parts[1] == name:
            return parts[2], "mode"
    args, kwargs = make()
    try:
        if IS_SHIM:
            return torch._C._shim_packet_overload(
                "aten::" + name, getattr(A, name).overloads(), args, kwargs), "resolver"
        return torch._C._jit_resolve_packet("aten::" + name, *args, **kwargs), "jit_resolve_packet"
    except Exception as exc:  # noqa: BLE001 - recorded
        return None, f"{type(exc).__name__}: {exc}"


for name, rows in CALLS.items():
    for label, make in rows:
        key = f"{name}|{label}"
        row = {}
        mode = Record()
        try:
            args, kwargs = make()
            with mode:
                r = getattr(A, name)(*args, **kwargs)
            row["outcome"] = {"ok": summary(r)}
        except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
            row["outcome"] = failed(exc)
        row["overload"], row["via"] = resolved(name, make, mode.seen)
        if not IS_SHIM:
            args, kwargs = make()
            try:
                row["jit"] = torch._C._jit_resolve_packet("aten::" + name, *args, **kwargs)
            except Exception as exc:  # noqa: BLE001 - recorded
                row["jit"] = f"{type(exc).__name__}: {exc}"
            row["mode_seen"] = [s for s in mode.seen if s.split(".")[1:2] == [name]][:1]
            # A CompositeImplicitAutograd overload never reaches a mode: upstream
            # runs its decomposition, and the mode sees what that calls
            # (`repeat_interleave.self_Tensor` -> `repeat_interleave.Tensor`).
            row["composite_implicit"] = isinstance(row["jit"], str) and bool(
                torch._C._dispatch_has_kernel_for_dispatch_key(
                    f"aten::{name}.{row['jit']}", "CompositeImplicitAutograd"))
        out["rows"][key] = row

# Calls upstream refuses: wrong kinds, counts and keyword names.
MISMATCH = [
    ("transpose|str", "transpose", lambda: ((t(), "a", 1), {})),
    ("transpose|float", "transpose", lambda: ((t(), 0.5, 1), {})),
    ("transpose|none", "transpose", lambda: ((t(), None, 1), {})),
    ("transpose|list", "transpose", lambda: ((t(), [0], 1), {})),
    ("transpose|missing", "transpose", lambda: ((t(), 0), {})),
    ("transpose|too_many", "transpose", lambda: ((t(), 0, 1, 2), {})),
    ("unbind|str", "unbind", lambda: ((t(), "a"), {})),
    ("select|str", "select", lambda: ((t(), "a", 0), {})),
    ("unflatten|int", "unflatten", lambda: ((t(), 1, 3), {})),
    ("unflatten|float_element", "unflatten", lambda: ((t(), 1, [3.0, 1]), {})),
    ("_unsafe_index|int", "_unsafe_index", lambda: ((t(), 3), {})),
    ("_unsafe_index|int_element", "_unsafe_index", lambda: ((t(), [0]), {})),
    ("_unsafe_index|missing", "_unsafe_index", lambda: ((t(),), {})),
    ("index|int_element", "index", lambda: ((t(), [0]), {})),
    ("empty|int", "empty", lambda: ((2,), {})),
    ("empty|float_element", "empty", lambda: (([2.0],), {})),
    ("empty|unknown_keyword", "empty", lambda: (([2],), {"bogus": 1})),
    ("empty|positional_dtype", "empty", lambda: (([2], torch.float64), {})),
    ("empty|device_int", "empty", lambda: (([2],), {"device": 0})),
    ("slice|float", "slice", lambda: ((t(), 1, 1.0, 2), {})),
    ("scatter_reduce|reduce_int", "scatter_reduce", lambda: ((t(), 0, idx(), t(), 1), {})),
    ("fill_|str", "fill_", lambda: ((t(), "x"), {})),
    ("fill_|missing", "fill_", lambda: ((t(),), {})),
    ("rsub|str", "rsub", lambda: ((t(), "x"), {})),
    ("rsub|list", "rsub", lambda: ((t(), [1]), {})),
    ("add_|str", "add_", lambda: ((t(), "x"), {})),
    ("bitwise_and|numbers", "bitwise_and", lambda: ((1, 1), {})),
    ("bitwise_and|missing", "bitwise_and", lambda: ((b(),), {})),
    ("isin|numbers", "isin", lambda: ((1, 1), {})),
    ("masked_fill|str", "masked_fill", lambda: ((b(), m(), "x"), {})),
    ("div_|mode_int", "div_", lambda: ((t(), t() + 1), {"rounding_mode": 1})),
]
for key, name, make in MISMATCH:
    try:
        args, kwargs = make()
        r = getattr(A, name)(*args, **kwargs)
        out["mismatch"][key] = {"ok": summary(r)}
    except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
        out["mismatch"][key] = failed(exc)

# Issue #386 moved the former section 5.5 narrowings into `CALLS` and `AGREE`
# by reproducing upstream's overload registration order and TorchScript's kind
# rules (`bool` for `int`, `None` for `Tensor`, a bare tensor for `Tensor?[]`).
NARROWING = []
for key, name, make, _ in NARROWING:
    try:
        args, kwargs = make()
        r = getattr(A, name)(*args, **kwargs)
        out["narrowing"][key] = {"ok": summary(r)}
    except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
        out["narrowing"][key] = failed(exc)
out["narrowing_expected"] = {key: kind for key, _, _, kind in NARROWING}

# Calls that left NARROWING because the shim now computes them (CI run
# 43853489734: `aten.add(t, 2)`, run 43859582176: `aten.mul(t, 2)`, and issue
# #386: `add_`, `sub_`, `mul_`, `div_`, `multiply`, `transpose|bool` and
# `index|bare_tensor`). Both sides record the values, and the test below
# requires them to agree.
AGREE = [
    ("add|number", "add", lambda: ((t(), 2), {})),
    ("mul|number", "mul", lambda: ((t(), 2), {})),
    ("add_|number", "add_", lambda: ((t(), 2), {})),
    ("sub_|number", "sub_", lambda: ((t(), 1), {})),
    ("mul_|number", "mul_", lambda: ((t(), 2.0), {})),
    ("div_|number", "div_", lambda: ((t(), 2), {})),
    ("multiply|number", "multiply", lambda: ((t(), 2), {})),
    ("transpose|bool", "transpose", lambda: ((t(), True, 1), {})),
    ("index|bare_tensor", "index", lambda: ((t(), i()), {})),
]
out["agree"] = {}
for key, name, make in AGREE:
    try:
        args, kwargs = make()
        r = getattr(A, name)(*args, **kwargs)
        out["agree"][key] = {"ok": summary(r), "values": r.tolist()}
    except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
        out["agree"][key] = failed(exc)


# Every implemented `aten.<name>.default`: the packet with that `default`,
# called with one representative value per required argument, built from
# this side's own `default` schema (`str(argument.type)`, upstream's text on
# both sides). Upstream's answer is `_jit_resolve_packet`; the shim's is
# `_shim_packet_overload`, which must not raise for a packet with a default
# (CI run 43841297493: `im2col`, `_unique2`, `alias`, ... dispatched
# `aten.<name>.` and refused, and `upsample_nearest1d` refused as ambiguous).
def representative(spelling):
    if spelling.startswith("Optional["):
        return None
    simple = {"Tensor": lambda: torch.zeros(2), "int": lambda: 1, "float": lambda: 1.0,
              "bool": lambda: False, "number": lambda: 1, "str": lambda: "mean",
              "Device": lambda: torch.device("cpu"),
              "List[int]": lambda: [1], "List[float]": lambda: [1.0],
              "List[bool]": lambda: [False], "List[Tensor]": lambda: [torch.zeros(2)],
              "List[Optional[Tensor]]": lambda: [torch.zeros(2, dtype=torch.int64)]}
    if spelling not in simple:
        raise LookupError(spelling)
    return simple[spelling]()


def default_calls(name):
    schema = getattr(A, name).default._schema
    positional, keyword, every = [], {}, {}
    for a in schema.arguments:
        if a.has_default_value():
            continue
        value = representative(str(a.type))
        every[a.name] = value
        if a.kwarg_only:
            keyword[a.name] = value
        else:
            positional.append(value)
    return {"positional": (tuple(positional), keyword), "keyword": ((), every)}


out["with_default"] = {}
out["with_default_skipped"] = {}
for key in sorted(k for k in impl if k.startswith("aten.") and k.endswith(".default")):
    name = key.split(".")[1]
    try:
        overloads = getattr(A, name).overloads()
        calls = default_calls(name)
    except LookupError as exc:
        out["with_default_skipped"][key] = f"no representative for {exc}"
        continue
    except Exception as exc:  # noqa: BLE001 - recorded, asserted below
        out["with_default"][key] = {"packet": f"{type(exc).__name__}: {exc}"}
        continue
    row = {"overloads": list(overloads)}
    if IS_SHIM:
        # The packet's own list, as `_jit_get_operation` holds it: the yaml
        # spells the default `""` there, which is the spelling that misfired.
        raw = list(torch._C._jit_get_operation("aten::" + name)[1])
        row["raw"] = raw
    for form, (args, kwargs) in calls.items():
        try:
            if IS_SHIM:
                row[form] = torch._C._shim_packet_overload("aten::" + name, raw, args, kwargs)
            else:
                row[form] = torch._C._jit_resolve_packet("aten::" + name, *args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - recorded, asserted below
            row[form] = {"raised": type(exc).__name__, "message": str(exc)[:300]}
    if IS_SHIM:
        try:
            row["no_arguments"] = torch._C._shim_packet_overload("aten::" + name, raw, (), {})
        except Exception as exc:  # noqa: BLE001 - recorded, asserted below
            row["no_arguments"] = {"raised": type(exc).__name__, "message": str(exc)[:300]}
    out["with_default"][key] = row


# The CI run's packets, called for real: the key the packet dispatches is
# what a mode records, and the mode stops the call there.
class Stop(Exception):
    pass


class First(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.seen = None

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.seen = str(func)
        raise Stop()


line = lambda: torch.arange(12.).reshape(1, 3, 4)
DISPATCHED = {
    "upsample_nearest1d|size_none": ("upsample_nearest1d", lambda: ((line(), [8], None), {}), "default"),
    "upsample_linear1d|size_none": ("upsample_linear1d", lambda: ((line(), [8], False, None), {}), "default"),
    "upsample_linear1d|scale": ("upsample_linear1d", lambda: ((line(), [8], False, 2.0), {}), "default"),
    "im2col": ("im2col", lambda: ((torch.ones(1, 1, 4, 4), [2, 2], [1, 1], [0, 0], [1, 1]), {}), "default"),
    "col2im": ("col2im", lambda: ((torch.ones(1, 4, 9), [4, 4], [2, 2], [1, 1], [0, 0], [1, 1]), {}), "default"),
    "_unique2": ("_unique2", lambda: ((torch.tensor([1, 2, 1]),), {}), "default"),
    "_unique2|keywords": ("_unique2", lambda: ((torch.tensor([1, 2, 1]),),
                                               {"sorted": True, "return_inverse": True}), "default"),
    "alias": ("alias", lambda: ((line(),), {}), "default"),
    "lift_fresh": ("lift_fresh", lambda: ((line(),), {}), "default"),
    "detach": ("detach", lambda: ((line(),), {}), "default"),
    "_unsafe_view": ("_unsafe_view", lambda: ((line(), [12]), {}), "default"),
}
out["dispatched"] = {}
for key, (name, make, _) in DISPATCHED.items():
    mode = First()
    try:
        args, kwargs = make()
        with mode:
            getattr(A, name)(*args, **kwargs)
        out["dispatched"][key] = {"ran_past_the_mode": mode.seen}
    except Stop:
        out["dispatched"][key] = {"seen": mode.seen}
    except BaseException as exc:  # noqa: BLE001 - one row, see `failed`
        out["dispatched"][key] = failed(exc)
out["dispatched_expected"] = {key: f"aten.{name}.{ov}" for key, (name, _, ov) in DISPATCHED.items()}

# Packets with a `default` and another overload that both or either accept:
# the overload chosen, at the resolver on each side (`.vec` is
# CompositeImplicitAutograd upstream, so a mode sees `.default` either way).
RESOLVE = {
    "upsample_nearest1d|both_bind": ("upsample_nearest1d", lambda: ((line(), [8], None), {})),
    "upsample_nearest1d|vec_only": ("upsample_nearest1d", lambda: ((line(), None, [2.0]), {})),
    "upsample_nearest1d|vec_keywords": ("upsample_nearest1d", lambda: (
        (line(),), {"output_size": [8], "scale_factors": None})),
    "upsample_linear1d|both_bind": ("upsample_linear1d", lambda: ((line(), [8], False, None), {})),
    "upsample_linear1d|vec_only": ("upsample_linear1d", lambda: ((line(), None, False, [2.0]), {})),
    "sum|dim": ("sum", lambda: ((line(), [0]), {})),
    "sum|dtype": ("sum", lambda: ((line(),), {"dtype": torch.float64})),
    "range|two": ("range", lambda: ((0, 5), {})),
    "max|other": ("max", lambda: ((line(), line()), {})),
    "max|plain": ("max", lambda: ((line(),), {})),
    "view|dtype": ("view", lambda: ((line(), torch.int32), {})),
    "view|size": ("view", lambda: ((line(), [12]), {})),
    "mm|out": ("mm", lambda: ((torch.ones(2, 2), torch.ones(2, 2)), {"out": torch.empty(2, 2)})),
    "rsub|none_tensor": ("rsub", lambda: ((line(), None), {})),
    "add_|none_tensor": ("add_", lambda: ((line(), None), {})),
}
out["resolve"] = {}
for key, (name, make) in RESOLVE.items():
    args, kwargs = make()
    try:
        if IS_SHIM:
            out["resolve"][key] = torch._C._shim_packet_overload(
                "aten::" + name, getattr(A, name).overloads(), args, kwargs)
        else:
            out["resolve"][key] = torch._C._jit_resolve_packet("aten::" + name, *args, **kwargs)
    except Exception as exc:  # noqa: BLE001 - recorded, asserted below
        out["resolve"][key] = {"raised": type(exc).__name__, "message": str(exc)[:300]}

print(json.dumps(out))
'''


def _run(shim, impl=None, _cache={}):
    if shim in _cache:
        return _cache[shim]
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    argv = [sys.executable, "-c", _PROBE] + ([] if shim else [json.dumps(impl)])
    proc = subprocess.run(argv, capture_output=True, text=True, env=env, cwd=_REPO_ROOT, timeout=900)
    assert proc.returncode == 0, (
        f"the {'shim' if shim else 'upstream'} probe did not run:\n{proc.stderr[-3000:]}"
    )
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    # An empty vendored tree makes the shim side import upstream silently
    # (AGENTS.md section 15.2), so the side is asserted, not assumed.
    assert data["is_shim"] is shim, f"the probe meant for shim={shim} ran on the other side"
    _cache[shim] = data
    return data


def _both():
    if not _available():
        _skip.skip("no vendored shim installed (scripts/vendor/install_shim.sh)")
        return None
    shim = _run(True)
    return shim, _run(False, shim["impl"])


def _chunks(message):
    """The first line of each per-overload error in a refusal."""
    prefix = "Overloaded torch operator invoked from Python failed to match any schema:\n"
    body = message[len(prefix):] if message.startswith(prefix) else message
    return [c.splitlines()[0] for c in body.split("\n\n") if c.strip() and c.startswith(("aten::", "Unknown keyword"))]


def test_every_packet_without_a_default_has_a_row():
    """The shim's packets without a `default` that have an implemented
    overload, against upstream's for the same implemented keys; both printed.
    Red when a packet upstream gives no `default` has one here, or when a
    packet without one here has no row in `CALLS`."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    print("   shim packets without default:", json.dumps(shim["packets"]))
    assert not [p for p in shim["packets"] if p[2]], [p for p in shim["packets"] if p[2]]
    ours = {p[0] for p in shim["packets"]}
    theirs = {p[0] for p in up["packets"]}
    # Every packet upstream gives no `default` has none here either. The
    # converse does not hold: upstream registers a TorchScript `default`
    # (`aten::add(Scalar a, Scalar b)`) for 13 names the shim lists without
    # one; those are printed and resolve like the rest.
    assert theirs <= ours, f"upstream has no default, the shim has one: {sorted(theirs - ours)}"
    print("   no default here, a TorchScript default upstream:", sorted(ours - theirs))
    missing = sorted(ours - set(shim["calls"]))
    assert not missing, f"packets without a default and no CALLS row: {missing}"
    assert len(theirs) >= 36, len(theirs)


def test_packets_resolve_upstreams_overload():
    """Each row of `CALLS`: the overload the shim's packet reached equals the
    overload upstream's packet reached. Where that overload is not
    implemented here, the shim's refusal names it."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    implemented = set(shim["impl"])
    bad = []
    for key in sorted(up["rows"]):
        u, s = up["rows"][key], shim["rows"].get(key)
        name = key.split("|")[0]
        # The oracle agrees with itself: what the mode saw is what
        # `_jit_resolve_packet` answers.
        # (A CompositeImplicitAutograd overload is the exception: the mode sees
        # the decomposition's call, another overload of the same packet.)
        if u["mode_seen"] and not u["composite_implicit"]:
            assert u["mode_seen"][0] == f"aten.{name}.{u['jit']}", (key, u)
        assert "ok" in u["outcome"], f"{key}: upstream raised {u['outcome']}"
        if s is None or s["overload"] != u["jit"]:
            bad.append(f"  {key}: upstream {u['jit']}, shim {s and (s['overload'], s['via'])}")
            continue
        op = f"aten.{name}.{u['jit']}"
        if op not in implemented and "raised" in s["outcome"] and op not in s["outcome"]["message"]:
            bad.append(f"  {key}: {op} is not implemented and the refusal does not name it: {s['outcome']}")
        if op in implemented and "raised" in s["outcome"]:
            bad.append(f"  {key}: {op} is implemented and the call raised: {s['outcome']}")
    print(f"   {len(up['rows']) - len(bad)} of {len(up['rows'])} rows resolve upstream's overload")
    assert not bad, f"{len(bad)} of {len(up['rows'])} rows differ:\n" + "\n".join(bad)


def test_the_resolved_overloads_answer_upstreams_shape_and_dtype():
    """Rows that compute on both sides: the result's shape and dtype (lists
    and tuples element by element). Red when a packet reaches an overload
    whose result is not upstream's."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    keys = [k for k in sorted(up["rows"]) if "ok" in shim["rows"].get(k, {}).get("outcome", {})]
    bad = [f"  {k}\n    upstream {up['rows'][k]['outcome']}\n    shim     {shim['rows'][k]['outcome']}"
           for k in keys if shim["rows"][k]["outcome"] != up["rows"][k]["outcome"]]
    print(f"   {len(keys) - len(bad)} of {len(keys)} computed rows agree on shape and dtype")
    assert keys, "no row computed on the shim"
    assert not bad, f"{len(bad)} of {len(keys)} rows differ:\n" + "\n".join(bad)


def test_kind_mismatches_refuse_as_upstreams():
    """`MISMATCH`: calls upstream's packet refuses (a wrong kind, a missing
    or extra argument, an unknown keyword). The shim raises the same type,
    and the first line of each of its per-overload errors is one of
    upstream's (upstream lists more overloads than the shim for `select`,
    `index` and `_unsafe_index`, so the shim's errors are a subset)."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    bad = []
    for key in sorted(up["mismatch"]):
        u, s = up["mismatch"][key], shim["mismatch"].get(key)
        assert u.get("raised") == "RuntimeError", f"{key}: upstream {u}"
        if s is None or s.get("raised") != "RuntimeError":
            bad.append(f"  {key}: upstream RuntimeError, shim {s}")
            continue
        ours = _chunks(s["message"])
        theirs = _chunks(u["message"])
        if not ours or [c for c in ours if c not in theirs]:
            bad.append(f"  {key}\n    upstream {theirs}\n    shim     {ours}")
    print(f"   {len(up['mismatch']) - len(bad)} of {len(up['mismatch'])} refusals agree")
    assert not bad, f"{len(bad)} of {len(up['mismatch'])} refusals differ:\n" + "\n".join(bad)


def test_known_narrowings_refuse_by_name():
    """`NARROWING`: calls upstream computes and the shim refuses, pinned so
    that a change in either direction is seen. Several overloads binding
    refuses with `NotImplementedError` naming the packet and the overloads;
    a value TorchScript accepts and `_TypeChecker` does not refuses with the
    `RuntimeError` whose last line says the kinds are `_TypeChecker`'s."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    bad = []
    for key, kind in sorted(up["narrowing_expected"].items()):
        u, s = up["narrowing"][key], shim["narrowing"][key]
        assert "ok" in u, f"{key}: upstream {u}"
        if s.get("raised") != kind:
            bad.append(f"  {key}: expected {kind}, shim {s}")
            continue
        name = key.split("|")[0]
        if kind == "NotImplementedError" and f"aten.{name} cannot choose" not in s["message"]:
            bad.append(f"  {key}: the refusal does not name the packet: {s['message']}")
        if kind == "RuntimeError" and "_TypeChecker" not in s["message"]:
            bad.append(f"  {key}: the refusal does not say whose kinds it checked: {s['message']}")
    assert not bad, "\n".join(bad)


def test_calls_that_left_the_narrowings_compute_upstreams_answer():
    """`AGREE`: calls the shim used to refuse and now computes. Each must give
    upstream's shape, dtype and values, so a wrong overload choice is red."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    bad = []
    for key in sorted(up["agree"]):
        u, s = up["agree"][key], shim["agree"].get(key)
        assert "ok" in u, f"{key}: upstream {u}"
        if s is None or s != u:
            bad.append(f"  {key}: upstream {u}, shim {s}")
    assert not bad, "\n".join(bad)


def test_every_implemented_default_packet_resolves_to_default():
    """Every implemented `aten.<name>.default` (printed): its packet lists
    `default`, and a call built from the `default` schema's required
    arguments, positionally and by keyword, resolves to `default` wherever
    upstream's `_jit_resolve_packet` does (217 of 217 upstream on 2.13.0,
    2026-11-16), never to an overload upstream did not choose, and the
    resolver does not raise, not even for a call with no arguments (the
    packet then dispatches `default` and its kernel answers, as before this
    branch). Red under CI run 43841297493's defect: the yaml's `""` read as
    "no default" (`im2col` resolved `""` and dispatched `aten.im2col.`;
    `upsample_nearest1d` refused as ambiguous)."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    rows = shim["with_default"]
    print(f"   implemented .default keys: {len(rows)}; skipped (no representative): "
          f"{json.dumps(shim['with_default_skipped'])}")
    print("   keys:", " ".join(sorted(rows)))
    assert len(rows) >= 200, len(rows)
    bad = []
    for key in sorted(rows):
        s, u = rows[key], up["with_default"].get(key, {})
        if "packet" in s:
            bad.append(f"  {key}: the packet did not build: {s['packet']}")
            continue
        if "default" not in s["overloads"]:
            bad.append(f"  {key}: the packet does not list default: {s['overloads']}")
        for form in ("positional", "keyword"):
            sv, uv = s.get(form), u.get(form)
            if not isinstance(sv, str):
                bad.append(f"  {key} {form}: the resolver raised: {sv}")
            elif uv == "default" and sv != "default":
                bad.append(f"  {key} {form}: upstream default, shim {sv!r}")
            elif isinstance(uv, str) and sv not in (uv, "default"):
                bad.append(f"  {key} {form}: upstream {uv!r}, shim {sv!r}")
            elif not isinstance(uv, str) and sv != "default":
                bad.append(f"  {key} {form}: upstream refused, shim chose {sv!r}")
        if not isinstance(s.get("no_arguments"), str):
            bad.append(f"  {key} no arguments: the resolver raised: {s.get('no_arguments')}")
    print(f"   {len(rows) - len({b.split()[0] for b in bad})} of {len(rows)} packets resolve as upstream")
    assert not bad, f"{len(bad)} rows differ:\n" + "\n".join(bad)


def test_the_ci_packets_dispatch_their_default():
    """`DISPATCHED`: the packets CI run 43841297493 refused, called for real
    under a mode that records the first op and stops there. Each reaches
    `aten.<name>.default`, as upstream's does. Red when the packet route
    dispatches `aten.<name>.` or refuses before the kernel."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    bad = []
    for key, expected in sorted(shim["dispatched_expected"].items()):
        u, s = up["dispatched"][key], shim["dispatched"][key]
        assert u == {"seen": expected}, f"{key}: upstream {u}"
        if s != {"seen": expected}:
            bad.append(f"  {key}: expected {expected}, shim {s}")
    assert not bad, f"{len(bad)} of {len(shim['dispatched'])} rows differ:\n" + "\n".join(bad)


def test_packets_with_a_default_choose_upstreams_overload():
    """`RESOLVE`: packets with a `default` and other overloads. The shim's
    choice is upstream's (`upsample_nearest1d(x, [8], None)` binds both and
    takes `default`, registered first; `(x, None, [2.0])` binds `.vec`
    alone; `range(0, 5)` takes `.step`, registered before `default`), or
    `default` where upstream's choice has no kernel here and `default` has
    one (the key the call had before this branch)."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    implemented = set(shim["impl"])
    assert up["resolve"]["upsample_nearest1d|both_bind"] == "default", up["resolve"]
    assert up["resolve"]["upsample_nearest1d|vec_only"] == "vec", up["resolve"]
    assert up["resolve"]["range|two"] == "step", up["resolve"]
    bad = []
    for key in sorted(up["resolve"]):
        u, s = up["resolve"][key], shim["resolve"][key]
        name = key.split("|")[0]
        assert isinstance(u, str), f"{key}: upstream {u}"
        if s == u:
            continue
        if (s == "default" and f"aten.{name}.{u}" not in implemented
                and f"aten.{name}.default" in implemented):
            continue
        bad.append(f"  {key}: upstream {u!r}, shim {s!r}")
    print(f"   {len(up['resolve']) - len(bad)} of {len(up['resolve'])} rows choose upstream's overload")
    assert not bad, "\n".join(bad)


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = _skip.run_tests(items, suite="test_packet_overload")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
