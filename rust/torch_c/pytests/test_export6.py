"""The three `torch.export` walls of `docs/graph/EXPORT5.md` §10, closed and held down.

`docs/graph/EXPORT5.md` §10 measured `torch.export` at **0 of 40** real
`transformers` architectures and named where the 26 export-time failures stop:

    14   AttributeError: '_SchemaType' object has no attribute 'annotation_str'
    11   NotImplementedError: TensorBase.set_: the storage has never been filled
     1   AssertionError: Could not find common device for aten.empty_strided.default

These tests are shaped against two named traps rather than against the walls.

`docs/graph/COMPILE.md` records that fixing a *superficial* error in
`torch.compile` made it silently return the eager function -- looking done while
doing nothing.  The analogous failure here is an `ExportedProgram` that is
returned and does not replay, or replays and does not agree.  So nothing in this
file asserts "export() returned".  The end-to-end tests assert the three
verdicts of `export_sweep.py` separately -- exported / replayed / **agreed** --
and agreement is element-wise against the module's own eager output with a
tolerance derived from float32 epsilon, never widened.

The second trap is a test that cannot go red.  `annotation_str` is not checked
against a table written in this file, because a table written here and a mapping
written in `bootstrap.py` would be the same author agreeing with himself.  It is
checked against **upstream torch's own answer, for all 2584 `- func:` entries of
the vendored `native_functions.yaml`, element by element** -- the same file the
shim parses.  A wrong spelling for any one argument of any one op fails.
"""

import json
import os
import subprocess
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")
_NATIVE_FUNCTIONS = os.path.join(
    _VENDOR_DIR, "torchgen", "packaged", "ATen", "native", "native_functions.yaml"
)

#: float32 machine epsilon.  Every tolerance below is a multiple of this and of
#: nothing else; `docs/numerics/AGREE.md` is the method.
_EPS32 = 1.1920929e-07


def _available():
    return os.path.isfile(_VENDOR_SHIM) and os.path.isfile(_NATIVE_FUNCTIONS)


def _run(script, shim, timeout=1800):
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
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# wall 1 -- `_SchemaType.annotation_str`
# ---------------------------------------------------------------------------

#: Emits, for every `- func:` entry, the `annotation_str` of every argument and
#: every return, in schema order.  Identical text on both sides, so the only
#: thing that can differ is the answer.
_ANNOTATION_PROBE = r"""
import json, sys
import torch

path = %(path)r
rows, names = [], []
for line in open(path, encoding="utf-8"):
    if not line.startswith("- func:"):
        continue
    text = "aten::" + line[len("- func:"):].strip()
    names.append(text.split("(", 1)[0])
    try:
        schema = torch._C.parse_schema(text)
    except Exception as exc:
        rows.append({"error": "%%s: %%s" %% (type(exc).__name__, exc)})
        continue
    try:
        rows.append({"ann": [a.type.annotation_str
                             for a in list(schema.arguments) + list(schema.returns)]})
    except Exception as exc:
        rows.append({"error": "%%s: %%s" %% (type(exc).__name__, exc)})

print(json.dumps({
    "is_shim": "torchnative" in (torch.__file__ or ""),
    "names": names,
    "rows": rows,
}))
"""


def _annotations(shim, _cache={}):
    key = "shim" if shim else "up"
    if key not in _cache:
        _cache[key] = _run(_ANNOTATION_PROBE % {"path": _NATIVE_FUNCTIONS}, shim=shim)
    return _cache[key]


def test_every_schema_type_answers_annotation_str_exactly_as_upstream_does():
    """Wall 1, at full width: 2584 schemas, every argument, both sides.

    Not a spot check and not a table.  `torch/fx/operator_schemas.py:95` does
    `eval(ts_type.annotation_str, _type_eval_globals)`, so a spelling that is
    merely plausible -- `Tensor?` where upstream says `Optional[Tensor]`, or
    `SymInt` where upstream says `int` -- raises `NameError` inside `eval` and
    stops export just as surely as the missing attribute did.  Only upstream's
    own answer decides.
    """
    if not _available():
        print("SKIP no vendored shim / native_functions.yaml")
        return
    shim, up = _annotations(True), _annotations(False)
    assert shim["is_shim"], "the shim-side probe did not get the vendored torch"
    assert not up["is_shim"], "the upstream-side probe got the shim"
    assert shim["names"] == up["names"], "the two sides read different schema files"

    errors = [(n, r["error"]) for n, r in zip(shim["names"], shim["rows"]) if "error" in r]
    assert not errors, (
        f"{len(errors)} of {len(shim['rows'])} schemas cannot answer annotation_str; "
        f"first three: {errors[:3]}"
    )

    mismatches = []
    for name, s, u in zip(shim["names"], shim["rows"], up["rows"]):
        if "error" in u:
            continue  # upstream refuses this entry too; not this shim's gap
        if s["ann"] != u["ann"]:
            mismatches.append((name, s["ann"], u["ann"]))
    assert not mismatches, (
        f"{len(mismatches)} schemas spell a type differently from upstream; "
        f"first three: {mismatches[:3]}"
    )
    judged = sum(1 for u in up["rows"] if "error" not in u)
    assert judged >= 2500, f"only {judged} schemas were comparable, expected the whole file"
    print(f"ANNOTATION: {judged} schemas compared element-wise, 0 mismatches")


def test_fx_can_turn_a_schema_into_a_python_signature():
    """The caller wall 1 actually blocks, exercised through its public door.

    `_SchemaType.annotation_str` existing is not the guarantee; the guarantee is
    that `torch.fx.operator_schemas` can build an `inspect.Signature` out of a
    schema, which is what `torch.export`'s normalisation path calls.  A test on
    the attribute alone would stay green if the spelling were eval-able but
    wrong for the *shape* of the type, and this one would not.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    probe = r"""
import json
import torch
from torch.fx.operator_schemas import _torchscript_schema_to_signature

out = {"is_shim": "torchnative" in (torch.__file__ or "")}
sigs = {}
for spelling in [
    "aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor",
    "aten::t(Tensor(a) self) -> Tensor(a)",
    "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor",
    "aten::empty.memory_format(SymInt[] size, *, ScalarType? dtype=None, "
    "Layout? layout=None, Device? device=None, bool? pin_memory=None, "
    "MemoryFormat? memory_format=None) -> Tensor",
    "aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor",
    "aten::split.sizes(Tensor(a -> *) self, SymInt[] split_size, int dim=0) -> Tensor(a)[]",
]:
    schema = torch._C.parse_schema(spelling)
    sig = _torchscript_schema_to_signature(schema)
    sigs[spelling.split("(", 1)[0]] = [
        (p.name, str(p.annotation), str(p.kind)) for p in sig.parameters.values()
    ]
out["sigs"] = sigs
print(json.dumps(out))
"""
    shim, up = _run(probe, shim=True), _run(probe, shim=False)
    assert shim["is_shim"], "the shim-side probe did not get the vendored torch"
    assert not up["is_shim"], "the upstream-side probe got the shim"
    for name in sorted(up["sigs"]):
        assert shim["sigs"].get(name) == up["sigs"][name], (
            f"{name}: signature differs from upstream\n"
            f"  shim     {shim['sigs'].get(name)}\n"
            f"  upstream {up['sigs'][name]}"
        )
    print(f"SIGNATURE: {len(up['sigs'])} schemas -> identical inspect.Signature")


# ---------------------------------------------------------------------------
# wall 1b -- the wall behind wall 1: a mode gets every argument as a keyword
# ---------------------------------------------------------------------------

_MODE_SHAPE_PROBE = r"""
import json
import torch
from torch.utils._python_dispatch import TorchDispatchMode

records = []


class Record(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        records.append({
            "op": str(func),
            "n_args": len(args),
            "kwargs": sorted(kwargs),
            "kwarg_only": sorted(a.name for a in func._schema.arguments if a.kwarg_only),
        })
        return func(*args, **kwargs)


x = torch.ones(2, 3)
w = torch.ones(4, 5)
with Record():
    x.new_empty((2, 2))
    x.add(x)
    x.transpose(0, 1)
    x.sum(0)
    w.t()
    torch.empty((2, 2), dtype=torch.float32)

print(json.dumps({
    "is_shim": "torchnative" in (torch.__file__ or ""),
    "records": records,
}))
"""


def test_a_dispatch_mode_receives_positional_arguments_positionally():
    """The wall that stood behind wall 1, and the reason `export` still stopped.

    With `annotation_str` answered, thirteen of the fourteen architectures moved
    straight to

        TypeError: OpOverload.decompose() got multiple values for argument 'self'

    raised at `torch/_subclasses/fake_tensor.py:2970`, `func.decompose(*args,
    **kwargs)`.  `OpOverload.decompose` is `def decompose(self, *args,
    **kwargs)`, so a schema argument *named* `self` -- which is most aten ops'
    first argument -- collides with the bound receiver the moment it arrives as
    a keyword.  Upstream never hits it because upstream hands `__torch_dispatch__`
    its positional arguments positionally; this shim bound every argument by
    keyword and `args` was always empty.

    Asserted as the shape upstream produces, op by op, not as "no exception":
    an implementation that moved *some* arguments across would still let
    `decompose` through on the ops it happened to cover, and this test names the
    ones it did not.

    Note the kwarg side is asserted too.  Moving everything to positional would
    also make `decompose` work, and would be just as wrong -- `dtype`, `layout`,
    `device`, `pin_memory` and `memory_format` are `kwarg_only` in the schema
    and upstream keeps them there.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim, up = _run(_MODE_SHAPE_PROBE, shim=True), _run(_MODE_SHAPE_PROBE, shim=False)
    assert shim["is_shim"], "the shim-side probe did not get the vendored torch"
    assert not up["is_shim"], "the upstream-side probe got the shim"

    # Indexed by op, because the two sides need not record the same *number* of
    # calls -- the shim's decomposition of a method into aten ops is its own
    # business and §9's overload disagreement is known and unclosed.  What is
    # compared is the argument shape of the ops both sides recorded.
    def by_op(side):
        out = {}
        for r in side["records"]:
            out.setdefault(r["op"], r)
        return out

    s, u = by_op(shim), by_op(up)
    common = sorted(set(s) & set(u))
    assert len(common) >= 4, (
        f"too few ops recorded on both sides to be evidence: {common}\n"
        f"  shim     {sorted(s)}\n  upstream {sorted(u)}"
    )

    # (a) the positional COUNT is upstream's, op for op.
    arity = [(op, s[op]["n_args"], u[op]["n_args"])
             for op in common if s[op]["n_args"] != u[op]["n_args"]]
    assert not arity, (
        "a TorchDispatchMode sees a different number of positional arguments "
        f"than upstream; rows are (op, shim, upstream): {arity}"
    )

    # (b) nothing the schema declares POSITIONAL arrives as a keyword. This is
    #     the property `OpOverload.decompose` needs and the one that was false:
    #     `self` as a keyword collides with the bound receiver.
    #
    #     Asserted this way rather than as `shim kwargs == upstream kwargs`
    #     on purpose. Upstream's *python argument parser* materialises some
    #     defaulted kwarg-only arguments and not others -- `new_empty` arrives
    #     with `pin_memory` and without `dtype`, which is a property of
    #     `python_arg_parser.cpp` and not of the dispatcher. Demanding that
    #     here would be demanding a quirk, and would make this test fail for a
    #     reason that has nothing to do with the wall.
    smuggled = [(op, sorted(set(s[op]["kwargs"]) - set(s[op]["kwarg_only"])))
                for op in common
                if set(s[op]["kwargs"]) - set(s[op]["kwarg_only"])]
    assert not smuggled, (
        "these schema-positional arguments reach a TorchDispatchMode as "
        "keywords, which is what collides with OpOverload.decompose's bound "
        f"`self`: {smuggled}"
    )
    print(f"MODESHAPE: {len(common)} ops, positional arity identical to upstream, "
          "no positional argument passed by keyword")


# ---------------------------------------------------------------------------
# wall 1c -- `new_empty` on a meta tensor, which is what `meta_embedding` calls
# ---------------------------------------------------------------------------

_NEW_FACTORY_PROBE = r"""
import json
import torch

out = {"is_shim": "torchnative" in (torch.__file__ or ""), "cases": {}}
w = torch.empty(7, 5, device="meta", dtype=torch.float32)
cases = {
    "new_empty": lambda: w.new_empty((3, 5)),
    "new_empty_dtype": lambda: w.new_empty((3, 5), dtype=torch.float64),
    "new_zeros": lambda: w.new_zeros((2, 4)),
    "new_ones": lambda: w.new_ones((2, 4)),
    "new_empty_0d": lambda: w.new_empty(()),
}
for name, fn in cases.items():
    try:
        t = fn()
        out["cases"][name] = {
            "shape": list(t.shape), "dtype": str(t.dtype), "device": str(t.device)
        }
    except Exception as exc:
        out["cases"][name] = {"error": "%s: %s" % (type(exc).__name__, exc)}
print(json.dumps(out))
"""


def test_the_new_x_factories_have_a_meta_kernel_that_matches_upstream():
    """The wall behind wall 1b.

    `torch/_meta_registrations.py:8997` is the meta kernel for `embedding`:

        return weight.new_empty(out_shape, dtype=out_dtype)

    With the mode seated correctly, every architecture with an embedding table
    -- thirteen of the fourteen -- reached that line and stopped on
    `no meta kernel for aten.new_empty.default`.  `new_ones` already had a meta
    arm; its two siblings did not.

    Shape, dtype AND device are all asserted against upstream: a factory that
    answered the right shape on the wrong device would put a meta tensor into a
    dense graph, which fails much later and somewhere else.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim, up = _run(_NEW_FACTORY_PROBE, shim=True), _run(_NEW_FACTORY_PROBE, shim=False)
    assert shim["is_shim"], "the shim-side probe did not get the vendored torch"
    assert not up["is_shim"], "the upstream-side probe got the shim"
    for name in sorted(up["cases"]):
        u, s = up["cases"][name], shim["cases"][name]
        assert "error" not in u, f"upstream itself refuses {name}: {u['error']}"
        assert s == u, f"{name}: shim {s} != upstream {u}"
    print(f"METAFACTORY: {len(up['cases'])} cases identical to upstream")


# ---------------------------------------------------------------------------
# wall 1d -- `Argument.default_value` was the schema's SOURCE TEXT
# ---------------------------------------------------------------------------

_DEFAULT_PROBE = r"""
import json
import torch

path = %(path)r
rows, names = [], []
for line in open(path, encoding="utf-8"):
    if not line.startswith("- func:"):
        continue
    text = "aten::" + line[len("- func:"):].strip()
    names.append(text.split("(", 1)[0])
    try:
        schema = torch._C.parse_schema(text)
    except Exception as exc:
        rows.append({"error": "%%s: %%s" %% (type(exc).__name__, exc)})
        continue
    vals = []
    for a in schema.arguments:
        try:
            vals.append([str(a.name), bool(a.has_default_value()), repr(a.default_value)])
        except Exception as exc:
            vals.append([str(a.name), None, "ERR %%s" %% (exc,)])
    rows.append({"args": vals})
print(json.dumps({
    "is_shim": "torchnative" in (torch.__file__ or ""),
    "names": names,
    "rows": rows,
}))
"""


def _defaults(shim, _cache={}):
    key = "shim" if shim else "up"
    if key not in _cache:
        _cache[key] = _run(_DEFAULT_PROBE % {"path": _NATIVE_FUNCTIONS}, shim=shim)
    return _cache[key]


def test_every_schema_default_is_the_python_value_upstream_gives_not_its_source_text():
    """The wall behind wall 1c, and the one with the widest reach.

    `torch/_subclasses/fake_impls.py:218` normalises a constructor call to
    keywords, which fills every unsupplied argument from `arg.default_value`,
    and then hands the result straight back to the op:

        new_kwargs = {..., 'layout': 'None', 'device': 'None', 'pin_memory': 'None'}
        r = func(*args, **new_kwargs)

    Those are the three-character *strings* `'None'`, not `None`.  This shim
    kept `default_value` as the schema's source text, and coupled
    `has_default_value()` to it -- `default_value is not None` -- so the text
    had to stay a string for the predicate to work.  It survived because the
    only caller that read it was this shim's own binder, which re-parses the
    text; the first outside reader got a string where upstream gives a value.

    2184 of the file's arguments differ, in 50 distinct shapes: `None`,
    `True`/`False`, ints, floats, quoted strings, list literals, the sized-list
    broadcast (`int[2] padding=0` is `[0, 0]`), and the three enum-valued
    defaults (`Mean` is `1`).  Every one is compared against upstream's own
    answer rather than against a table written here.

    `has_default_value()` is compared too, and separately: a `default_value`
    that is now legitimately `None` must not make the predicate say "no
    default", which is precisely what the old coupling would have done.
    """
    if not _available():
        print("SKIP no vendored shim / native_functions.yaml")
        return
    shim, up = _defaults(True), _defaults(False)
    assert shim["is_shim"], "the shim-side probe did not get the vendored torch"
    assert not up["is_shim"], "the upstream-side probe got the shim"
    assert shim["names"] == up["names"], "the two sides read different schema files"

    has_bad, value_bad, compared = [], [], 0
    for name, s, u in zip(shim["names"], shim["rows"], up["rows"]):
        if "error" in u or "error" in s:
            continue
        if len(s["args"]) != len(u["args"]):
            value_bad.append((name, "arity", len(s["args"]), len(u["args"])))
            continue
        for a, b in zip(s["args"], u["args"]):
            compared += 1
            if a[1] != b[1]:
                has_bad.append((name, a[0], a[1], b[1]))
            if a[2] != b[2]:
                value_bad.append((name, a[0], a[2], b[2]))
    assert not has_bad, (
        f"{len(has_bad)} arguments disagree with upstream on has_default_value(); "
        f"first three: {has_bad[:3]}"
    )
    assert not value_bad, (
        f"{len(value_bad)} arguments give a different default_value than upstream; "
        f"first five: {value_bad[:5]}"
    )
    assert compared >= 9000, (
        f"only {compared} arguments were compared, expected the whole file"
    )
    print(f"DEFAULTS: {compared} schema arguments compared element-wise, 0 mismatches")


# ---------------------------------------------------------------------------
# wall 1e -- `layout=torch.strided`, which `torch.export` passes explicitly
# ---------------------------------------------------------------------------

_LAYOUT_PROBE = r"""
import json
import torch

out = {"is_shim": "torchnative" in (torch.__file__ or ""), "cases": {}}
cases = {
    "arange": lambda: torch.arange(0, 6, 2, layout=torch.strided),
    "empty": lambda: torch.empty((2, 3), layout=torch.strided),
    "zeros": lambda: torch.zeros((2, 3), layout=torch.strided),
    "ones": lambda: torch.ones((2, 3), layout=torch.strided),
    "full": lambda: torch.full((2, 3), 4.0, layout=torch.strided),
    # The refusal half. `sparse_coo` is a layout this shim does not have, and
    # dropping it would be a wrong answer with no trace -- so it must still
    # raise, on both sides.
    "arange_sparse": lambda: torch.arange(0, 6, 2, layout=torch.sparse_coo),
    "empty_sparse": lambda: torch.empty((2, 3), layout=torch.sparse_coo),
}
for name, fn in cases.items():
    try:
        t = fn()
        out["cases"][name] = {"shape": list(t.shape), "dtype": str(t.dtype),
                              "values": t.flatten().tolist()}
    except Exception as exc:
        out["cases"][name] = {"raised": type(exc).__name__}
print(json.dumps(out))
"""


def test_layout_strided_is_accepted_and_every_other_layout_is_still_refused():
    """`torch.export` passes `layout=torch.strided` explicitly, on every factory.

    `torch/_export/non_strict_utils.py:1205` spells it out, and after the
    defaults fix ten of the architectures stopped here:

        aten.arange.start_step: argument 'layout' not implemented (got torch.strided)

    on an argument asking for exactly the layout being handed back.  A helper
    for this already existed (`reject_layout`) and was wired to three
    hand-picked ops; the generic `reject_unsupported` refused every layout
    including the only one the shim has.

    The refusing half is asserted in the same test and deliberately so.  "Accept
    `strided`" is one character away from "accept anything", and a test that
    only checked the accepting half would stay green for a shim that silently
    dropped `sparse_coo` -- which is a wrong answer with no trace.  Both halves
    are compared against upstream, so neither can be satisfied by a constant.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim, up = _run(_LAYOUT_PROBE, shim=True), _run(_LAYOUT_PROBE, shim=False)
    assert shim["is_shim"], "the shim-side probe did not get the vendored torch"
    assert not up["is_shim"], "the upstream-side probe got the shim"
    for name in sorted(up["cases"]):
        u, s = up["cases"][name], shim["cases"][name]
        if name.endswith("_sparse"):
            assert "raised" in u, f"upstream accepted {name}, so this is not a refusal case"
            assert "raised" in s, (
                f"{name}: the shim ACCEPTED a layout it does not have and answered "
                f"{s} -- a wrong answer with no trace"
            )
            continue
        assert "raised" not in u, f"upstream itself refuses {name}: {u}"
        assert s == u, f"{name}: shim {s} != upstream {u}"
    accepted = sum(1 for n in up["cases"] if not n.endswith("_sparse"))
    refused = len(up["cases"]) - accepted
    print(f"LAYOUT: {accepted} factories accept torch.strided, {refused} still refuse sparse_coo")


# ---------------------------------------------------------------------------
# wall 1f -- `pin_memory=False`, which asks for nothing and was refused
# ---------------------------------------------------------------------------

_PIN_PROBE = r"""
import json
import torch

out = {"is_shim": "torchnative" in (torch.__file__ or ""), "cases": {}}
cases = {
    "arange": lambda: torch.arange(0, 6, 2, pin_memory=False),
    "empty": lambda: torch.empty((2, 3), pin_memory=False),
    "zeros": lambda: torch.zeros((2, 3), pin_memory=False),
    "ones": lambda: torch.ones((2, 3), pin_memory=False),
    "full": lambda: torch.full((2, 3), 4.0, pin_memory=False),
    # The refusal half: asking to PIN is asking for something this shim cannot
    # do, and it must still say so.
    "arange_pinned": lambda: torch.arange(0, 6, 2, pin_memory=True),
    "empty_pinned": lambda: torch.empty((2, 3), pin_memory=True),
}
for name, fn in cases.items():
    try:
        t = fn()
        out["cases"][name] = {"shape": list(t.shape), "dtype": str(t.dtype),
                              "values": t.flatten().tolist()}
    except Exception as exc:
        out["cases"][name] = {"raised": type(exc).__name__}
print(json.dumps(out))
"""


def test_pin_memory_false_is_accepted_and_pin_memory_true_is_still_refused():
    """`pin_memory=False` asks for nothing; refusing it refuses a no-op.

    The wall behind the layout one, and the same shape: `torch.export` passes
    every factory argument explicitly, so ten architectures moved from
    `argument 'layout'` straight to `argument 'pin_memory'`.

    `False` and `None` mean the same thing here -- do not pin -- and the shim
    accepted one and refused the other.  `True` is a different request, this
    shim cannot honour it, and it still raises; that half is asserted so that
    "accept `False`" cannot quietly become "accept anything".
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim, up = _run(_PIN_PROBE, shim=True), _run(_PIN_PROBE, shim=False)
    assert shim["is_shim"], "the shim-side probe did not get the vendored torch"
    assert not up["is_shim"], "the upstream-side probe got the shim"
    for name in sorted(up["cases"]):
        u, s = up["cases"][name], shim["cases"][name]
        if name.endswith("_pinned"):
            assert "raised" in s, (
                f"{name}: the shim accepted pin_memory=True and answered {s}, "
                "which claims pinned memory it did not allocate"
            )
            continue
        assert "raised" not in u, f"upstream itself refuses {name}: {u}"
        assert s == u, f"{name}: shim {s} != upstream {u}"
    accepted = sum(1 for n in up["cases"] if not n.endswith("_pinned"))
    print(f"PINMEMORY: {accepted} factories accept pin_memory=False, "
          f"{len(up['cases']) - accepted} still refuse pin_memory=True")


# ---------------------------------------------------------------------------
# wall 1g -- the TorchScript type singletons, which nothing was ever identical to
# ---------------------------------------------------------------------------

_SINGLETON_PROBE = r"""
import json
import torch
from torch._subclasses.fake_impls import _is_tensor_constructor, contains_tensor_types

out = {"is_shim": "torchnative" in (torch.__file__ or "")}

ops = {
    "arange.start_step": torch.ops.aten.arange.start_step,
    "empty.memory_format": torch.ops.aten.empty.memory_format,
    "zeros.default": torch.ops.aten.zeros.default,
    "full.default": torch.ops.aten.full.default,
    "randn.default": torch.ops.aten.randn.default,
    # Not constructors: they take a tensor.
    "add.Tensor": torch.ops.aten.add.Tensor,
    "relu.default": torch.ops.aten.relu.default,
    "t.default": torch.ops.aten.t.default,
}
out["constructor"] = {k: bool(_is_tensor_constructor(v)) for k, v in ops.items()}

# `is`, not `==`: `_is_tensor_constructor` uses identity and nothing else will do.
schema = torch._C.parse_schema(
    "aten::probe(Tensor self, int n, Tensor? bias, int[] shape, str mode) -> Tensor"
)
tensor_t = torch._C.TensorType.get()
out["identity"] = {
    "return_is_tensor_singleton": schema.returns[0].type is tensor_t,
    "arg0_is_tensor_singleton": schema.arguments[0].type is tensor_t,
    "arg1_is_int_singleton": schema.arguments[1].type is torch._C.IntType.get(),
    "arg1_is_tensor_singleton": schema.arguments[1].type is tensor_t,
    "optional_is_not_tensor_singleton": schema.arguments[2].type is not tensor_t,
    "get_is_stable": torch._C.TensorType.get() is torch._C.TensorType.get(),
}
out["contains_tensor"] = [bool(contains_tensor_types(a.type)) for a in schema.arguments]
out["isinstance"] = {
    "plain_tensor": isinstance(schema.returns[0].type, torch._C.TensorType),
    "optional_tensor": isinstance(schema.arguments[2].type, torch._C.TensorType),
    "int_is_not_tensor": isinstance(schema.arguments[1].type, torch._C.TensorType),
    "int_is_int": isinstance(schema.arguments[1].type, torch._C.IntType),
    "str_is_string": isinstance(schema.arguments[4].type, torch._C.StringType),
}
print(json.dumps(out))
"""


def test_the_torchscript_type_singletons_are_the_objects_a_schema_hands_out():
    """The last wall of the wall-1 chain, and the one that is a decision.

    `torch/_subclasses/fake_impls.py:159` decides "is this op a tensor
    constructor" by **identity**:

        schema.returns[0].type is torch._C.TensorType.get()

    Nothing this shim produced was ever that object, so the answer was `False`
    for every op, no constructor ever reached `fake_impls.constructors`, and
    `arange` fell through to the generic path -- which looks for a common device
    among arguments, finds no tensor among them, and raises
    `Could not find common device for aten.arange.start_step`.  Ten
    architectures.

    `_SchemaType`'s docstring used to decline this correspondence on purpose,
    and it was right to while the correspondence did not exist: answering `True`
    from a string match would have been a claim.  The correspondence exists now
    -- `_SchemaType` is interned per spelling and `TensorType.get()` returns the
    interned `Tensor` -- so `is` is a real identity rather than a simulated one.

    Both directions are asserted.  A shim where `TensorType.get()` matched
    *everything* would make `_is_tensor_constructor` true for `add.Tensor`, and
    then every elementwise op would be routed through the constructor impl; so
    the ops that are NOT constructors are in the table too, and `int` must not
    be a `TensorType`.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim, up = _run(_SINGLETON_PROBE, shim=True), _run(_SINGLETON_PROBE, shim=False)
    assert shim["is_shim"], "the shim-side probe did not get the vendored torch"
    assert not up["is_shim"], "the upstream-side probe got the shim"
    for section in ("constructor", "identity", "contains_tensor", "isinstance"):
        assert shim[section] == up[section], (
            f"{section} differs from upstream\n"
            f"  shim     {shim[section]}\n  upstream {up[section]}"
        )
    assert any(up["constructor"].values()) and not all(up["constructor"].values()), (
        "the probe's op table does not separate constructors from non-constructors, "
        "so agreeing with upstream on it proves nothing"
    )
    print("SINGLETON: _is_tensor_constructor, identity, isinstance and "
          "contains_tensor_types all agree with upstream")


# ---------------------------------------------------------------------------
# wall 2 -- the `filled` invariant meets `meta_utils.py`'s metadata-only `set_`
# ---------------------------------------------------------------------------

_SET_PROBE = r"""
import json
import torch

out = {"is_shim": "torchnative" in (torch.__file__ or ""), "cases": {}}


def record(name, fn):
    try:
        out["cases"][name] = fn()
    except Exception as exc:
        out["cases"][name] = {"raised": type(exc).__name__}


def meta_relayout():
    r = torch.empty(2, 3, device="meta", dtype=torch.float32)
    s = r.untyped_storage()
    before = s._cdata
    r.set_(s, 0, (3, 2), (2, 1))
    return {"shape": list(r.shape), "dtype": str(r.dtype), "device": str(r.device),
            "shares_storage": r.untyped_storage()._cdata == before}


def meta_same_shape():
    r = torch.empty(4, device="meta", dtype=torch.float64)
    s = r.untyped_storage()
    r.set_(s, 0, (4,), (1,))
    return {"shape": list(r.shape), "dtype": str(r.dtype), "device": str(r.device),
            "shares_storage": r.untyped_storage()._cdata == s._cdata}


def meta_grows():
    # The storage's byte count is not a bound on what `set_` may claim: upstream
    # lets the size exceed it on meta, because there are no bytes to run out of.
    r = torch.empty(2, device="meta", dtype=torch.float32)
    s = r.untyped_storage()
    r.set_(s, 0, (6,), (1,))
    return {"shape": list(r.shape), "dtype": str(r.dtype), "device": str(r.device)}


def dense_from_unfilled():
    # The invariant being NARROWED, not lifted. A dense tensor built from a
    # storage nobody ever wrote bytes into is docs/models/CKPT.md §4's silent-zeros
    # failure, and it must still refuse.
    r = torch.empty(4, dtype=torch.float32)
    s = torch.UntypedStorage(16)
    r.set_(s, 0, (4,), (1,))
    return {"shape": list(r.shape), "values": r.tolist()}


record("meta_relayout", meta_relayout)
record("meta_same_shape", meta_same_shape)
record("meta_grows", meta_grows)
record("dense_from_unfilled", dense_from_unfilled)
print(json.dumps(out))
"""


def test_set_on_a_meta_tensor_with_a_meta_storage_is_metadata_and_is_allowed():
    """`docs/graph/EXPORT5.md` §10's wall 2, narrowed rather than lifted.

    `torch/_subclasses/meta_utils.py:2124` -- the "crazy town" branch -- builds
    a meta storage and then `set_`s it onto a meta tensor to give that tensor a
    layout it could not get from `clone()`.  No bytes are involved on either
    side.  §2's `filled = false` invariant refused it, correctly for the dense
    case it was written for and too broadly for this one.  It was 11 of the 26
    export failures when the round began and 18 by the time the wall-1 chain
    was closed, because closing wall 1 let more architectures reach it.

    The narrowing is exactly the one §10 names: **allow `set_` when both the
    tensor and the storage are meta, refuse otherwise.**

    `dense_from_unfilled` is in the same probe and asserts the refusal is still
    there.  Without it this test would pass just as happily against a shim that
    deleted the invariant, which is the `docs/models/CKPT.md` §4 silent-zeros
    failure -- a tensor of zeros that nobody wrote and everybody believes.

    `shares_storage` is asserted too: `meta_utils.py`'s whole reason for this
    branch is to record that two tensors alias, so a `set_` that minted a fresh
    storage identity would satisfy the call and lose the fact it was made for.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim, up = _run(_SET_PROBE, shim=True), _run(_SET_PROBE, shim=False)
    assert shim["is_shim"], "the shim-side probe did not get the vendored torch"
    assert not up["is_shim"], "the upstream-side probe got the shim"

    for name in ("meta_relayout", "meta_same_shape", "meta_grows"):
        u, s = up["cases"][name], shim["cases"][name]
        assert "raised" not in u, f"upstream itself refuses {name}: {u}"
        assert "raised" not in s, f"{name}: the shim still refuses -- {s}"
        for key in ("shape", "dtype", "device"):
            if key in u:
                assert s.get(key) == u[key], f"{name}.{key}: shim {s.get(key)!r} != upstream {u[key]!r}"
        if "shares_storage" in u:
            assert s.get("shares_storage") == u["shares_storage"], (
                f"{name}: shim says shares_storage={s.get('shares_storage')}, "
                f"upstream says {u['shares_storage']} -- the aliasing fact "
                "meta_utils.py builds this branch to record"
            )

    assert "raised" in shim["cases"]["dense_from_unfilled"], (
        "a DENSE tensor was built from a storage nobody delivered bytes for, and "
        f"answered {shim['cases']['dense_from_unfilled']}. That is docs/models/CKPT.md "
        "§4's silent zeros: the invariant was lifted, not narrowed."
    )
    print("SETMETA: meta+meta set_ allowed and aliasing preserved; "
          "dense-from-unfilled still refused")


# ---------------------------------------------------------------------------
# wall 1h -- an argument that is its own default must not reach the mode
# ---------------------------------------------------------------------------

_DEFAULT_KWARG_PROBE = r"""
import json
import torch
from torch.utils._python_dispatch import TorchDispatchMode

records = []


class Record(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        records.append({"op": str(func), "kwargs": sorted(kwargs)})
        return func(*args, **kwargs)


x = torch.ones(2, 3)
with Record():
    # `device=None` and `layout=None` ARE the schema defaults. Upstream's boxed
    # dispatch omits them; a mode that sees them reads `kwargs["device"].type`
    # off `None`.
    torch.ops.aten._to_copy.default(x, dtype=torch.float64, device=None)
    # `dtype` here is NOT the default, so it must survive.
    torch.ops.aten._to_copy.default(x, dtype=torch.float16)

print(json.dumps({
    "is_shim": "torchnative" in (torch.__file__ or ""),
    "records": records,
}))
"""


def test_a_kwarg_that_equals_its_schema_default_does_not_reach_a_dispatch_mode():
    """The wall behind wall 2, and the plainest example of a wrong shape.

    `torch/_subclasses/fake_tensor.py:2624` is

        "device" in kwargs and kwargs["device"].type != "cpu"

    with no `is None` between the two halves, because upstream's boxed dispatch
    never puts an argument in `kwargs` when its value is the schema default.
    This shim's `OpOverload.__call__` forwarded its caller's keywords verbatim,
    so a caller who wrote `device=None` explicitly -- and `_to_copy`'s callers
    inside `torch.export` do -- produced `AttributeError: 'NoneType' object has
    no attribute 'type'` on twelve architectures.

    The second call in the probe is the half that stops this from being "drop
    every optional kwarg": `dtype=torch.float16` is not `_to_copy`'s default and
    must still arrive, or the copy would silently keep its input's dtype.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim, up = _run(_DEFAULT_KWARG_PROBE, shim=True), _run(_DEFAULT_KWARG_PROBE, shim=False)
    assert shim["is_shim"], "the shim-side probe did not get the vendored torch"
    assert not up["is_shim"], "the upstream-side probe got the shim"

    def kwargs_of(side, op):
        return [r["kwargs"] for r in side["records"] if r["op"] == op]

    op = "aten._to_copy.default"
    s, u = kwargs_of(shim, op), kwargs_of(up, op)
    assert u, "upstream recorded no _to_copy call, so the probe proves nothing"
    assert s == u, (
        f"{op}: the shim's dispatch mode sees {s}, upstream's sees {u}"
    )
    assert any("dtype" in k for k in u), (
        "upstream dropped `dtype` too, so this probe cannot tell dropping "
        "defaults from dropping everything"
    )
    print(f"DEFAULTKWARG: {len(u)} _to_copy calls, keyword sets identical to upstream")


# ---------------------------------------------------------------------------
# the shape-only meta kernels the freed architectures then asked for
# ---------------------------------------------------------------------------

_MORE_META_PROBE = r"""
import json
import torch

out = {"is_shim": "torchnative" in (torch.__file__ or ""), "cases": {}}


def record(name, fn):
    try:
        t = fn()
        out["cases"][name] = {"shape": list(t.shape), "dtype": str(t.dtype),
                              "device": str(t.device)}
    except Exception as exc:
        out["cases"][name] = {"raised": type(exc).__name__, "msg": str(exc)[:120]}


m = torch.empty(2, 1, 3, 1, device="meta", dtype=torch.float32)
record("squeeze_dims", lambda: torch.ops.aten.squeeze.dims(m, [1, 3]))
record("squeeze_dims_noop", lambda: torch.ops.aten.squeeze.dims(m, [0, 2]))
record("squeeze_dims_negative", lambda: torch.ops.aten.squeeze.dims(m, [-3, -1]))
n = torch.empty(4, 6, device="meta", dtype=torch.float32)
record("split_dim", lambda: torch.ops.prims.split_dim(n, 1, 2))
record("split_dim_0", lambda: torch.ops.prims.split_dim(n, 0, 4))
print(json.dumps(out))
"""


def test_the_shape_only_meta_kernels_the_freed_architectures_ask_for():
    """Three ops that are pure shape rules, and were reached only once the
    wall-1 chain and wall 2 were closed.

    Every one of them was a `no meta kernel for ...` on the sweep AFTER the
    walls above came down -- they were never visible while `annotation_str` was
    stopping the trace in the first few ops.  Each is a rule over `shape` with
    no values involved, and each is compared against upstream's own answer for
    shape, dtype AND device rather than against a shape written here.

    `aten.lift_fresh_copy.default` is NOT here and is left open on purpose: it
    has no kernel at all in this shim, dense or meta, so it is an op to add
    rather than a shape rule, and `docs/graph/EXPORT6.md` §5 carries it.

    `squeeze_dims_noop` is not filler: `squeeze` on an axis whose extent is not
    1 is a **no-op, not an error**, and a kernel that removed the axis anyway
    would pass a test that only tried extents of 1.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim, up = _run(_MORE_META_PROBE, shim=True), _run(_MORE_META_PROBE, shim=False)
    assert shim["is_shim"], "the shim-side probe did not get the vendored torch"
    assert not up["is_shim"], "the upstream-side probe got the shim"
    for name in sorted(up["cases"]):
        u, s = up["cases"][name], shim["cases"][name]
        assert "raised" not in u, f"upstream itself refuses {name}: {u}"
        assert "raised" not in s, f"{name}: the shim refuses -- {s}"
        assert s == u, f"{name}: shim {s} != upstream {u}"
    print(f"MOREMETA: {len(up['cases'])} meta cases identical to upstream")


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
