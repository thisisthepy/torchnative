"""Issue #306, wall W6: the shim's JIT types answer as upstream's do.

`executorch.exir` reads schema argument types the way upstream code does:

    isinstance(schema_arg.type, torch.TensorType)
        or schema_arg.type == torch.OptionalType.ofTensor()     # verification/arg_validator.py:82-83
    schema_arg.type == torch.ListType.ofTensors()                # arg_validator.py:89
    if isinstance(val_type, torch.OptionalType):                 # emit/_emitter.py:561-566
        val_type = val_type.getElementType()
    assert isinstance(val_type, torch.ListType)
    torch.ListType(self._get_list_tuple_jit_type(val))           # _emitter.py:558

Before this, `OptionalType.ofTensor()` and `ListType.ofTensors()` refused,
`getElementType` did not exist, `isinstance(arg.type, ListType)` was False
for every list, and `torch.ListType(x)` returned a placeholder for any `x`
(`ListType()`, `ListType(int)` included): a check that could not fail.

Two tests, both against upstream (torch 2.13.0, `sys.executable` without the
vendored tree on its path), each in a subprocess:

  * the full width: every argument and return of every schema in the vendored
    `native_functions.yaml`, fingerprinted the same way on both sides
    (isinstance over the `_C` type classes, `kind()`, element chain, `==`,
    `!=`, `isSubtypeOf` in both directions and `containedTypes` against a set
    of reference types built from the public constructors);
  * a measured table: exir's own expressions on real operator schemas, the
    constructors, the singletons, the subtype rules and the refusals, with
    upstream's answers written down below and re-checked against upstream
    when it is importable.

What these cannot find: `str()` of a type (the shim prints the schema
spelling `int[]` where upstream prints `List[int]`; the binder and `aten.rs`
read that spelling, so it is out of scope here), `Argument.real_type` (absent
on the shim, an `AttributeError`), identity of fresh constructions (the shim
interns, so `ListType(IntType.get()) is ListType.ofInts()`; upstream gives a
fresh object), and the `type(t)` of a schema type (`_SchemaType` here, the
named class upstream; `executorch/exir/emit/_emitter.py:1552` asserts on it
for symbolic ops).
"""
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
VENDOR = ROOT / "torchnative" / "python"
NATIVE_FUNCTIONS = VENDOR / "torchgen" / "packaged" / "ATen" / "native" / "native_functions.yaml"

sys.path.insert(0, str(ROOT / "tests" / "_support"))
import _skip  # noqa: E402


def _run(script, shim, *argv, timeout=1800):
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = str(VENDOR)
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run([sys.executable, "-c", script, *argv], capture_output=True,
                          text=True, env=env, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"probe ({'shim' if shim else 'upstream'}) exited {proc.returncode}\n"
            f"--- stderr ---\n{proc.stderr[-4000:]}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _have_shim():
    return (VENDOR / "torch" / "__init__.py").is_file()


def _have_upstream():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "TORCH_USE_RTLD_GLOBAL")}
    proc = subprocess.run([sys.executable, "-c", "import torch, sys; "
                           "sys.exit(1 if hasattr(torch._C, '_aten_implemented') else 0)"],
                          capture_output=True, env=env, timeout=600)
    return proc.returncode == 0


# ---------------------------------------------------------------------------
# full width
# ---------------------------------------------------------------------------

_FINGERPRINT_PROBE = r'''
import json, sys
import torch

path = sys.argv[1]
C = torch._C
L, O = C.ListType, C.OptionalType
CLASSES = ["Type", "TensorType", "IntType", "SymIntType", "SymBoolType", "FloatType",
           "ComplexType", "BoolType", "StringType", "NumberType", "NoneType", "DeviceObjType",
           "AnyType", "StreamObjType", "_GeneratorType", "ListType", "OptionalType",
           "TupleType", "DictType"]
CLASSES = [(n, getattr(C, n)) for n in CLASSES]
G = {n: getattr(C, n).get() for n in ("TensorType", "IntType", "SymIntType", "SymBoolType",
     "FloatType", "ComplexType", "BoolType", "StringType", "NumberType", "NoneType",
     "DeviceObjType", "AnyType", "_GeneratorType", "StreamObjType")}
REFS = dict(G)
REFS.update({
    "ofTensor": O.ofTensor(), "ofTensors": L.ofTensors(), "ofInts": L.ofInts(),
    "ofFloats": L.ofFloats(), "ofBools": L.ofBools(), "ofStrings": L.ofStrings(),
    "ofComplexDoubles": L.ofComplexDoubles(),
    "L(SymInt)": L(G["SymIntType"]), "L(Number)": L(G["NumberType"]),
    "L(ofTensor)": L(O.ofTensor()), "O(ofInts)": O(L.ofInts()), "O(ofFloats)": O(L.ofFloats()),
    "O(L(SymInt))": O(L(G["SymIntType"])), "O(SymInt)": O(G["SymIntType"]),
})
for n in ("IntType", "FloatType", "BoolType", "StringType", "NumberType", "DeviceObjType",
          "_GeneratorType", "SymBoolType", "ComplexType", "StreamObjType"):
    REFS[f"O({n})"] = O(G[n])

def fp(t, depth=0):
    out = {"isinstance": [n for n, c in CLASSES if isinstance(t, c)]}
    try:
        out["kind"] = t.kind()
    except Exception as e:
        out["kind"] = "RAISE " + type(e).__name__
    out["is_Tensor_singleton"] = t is G["TensorType"]
    out["eq"] = sorted(k for k, v in REFS.items() if t == v)
    out["ne"] = sorted(k for k, v in REFS.items() if not (t != v))
    out["sub"] = sorted(k for k, v in REFS.items() if t.isSubtypeOf(v))
    out["super"] = sorted(k for k, v in REFS.items() if v.isSubtypeOf(t))
    out["contained"] = [x.kind() for x in t.containedTypes()]
    if hasattr(t, "getElementType"):
        out["elem"] = fp(t.getElementType(), depth + 1) if depth < 4 else "deep"
    else:
        out["elem"] = None
    return out

rows, names, spell = [], [], []
for line in open(path, encoding="utf-8"):
    if not line.startswith("- func:"):
        continue
    text = "aten::" + line[len("- func:"):].strip()
    try:
        s = C.parse_schema(text)
    except Exception:
        continue
    names.append(text.split("(", 1)[0])
    items = list(s.arguments) + list(s.returns)
    rows.append([fp(a.type) for a in items])
    spell.append([str(a.type) for a in items])
refs = {k: fp(v) for k, v in REFS.items()}
print(json.dumps({"is_shim": hasattr(C, "_aten_implemented"), "names": names, "rows": rows,
                  "spell": spell, "refs": refs}))
'''


def _alias_annotated(spelling):
    return "(" in spelling and not spelling.rstrip().endswith(("]", "?"))


def _diff(a, b, path=""):
    if not isinstance(a, dict) or not isinstance(b, dict):
        return [] if a == b else [path or "root"]
    out = []
    for key in a:
        if key == "elem":
            out += _diff(a[key], b.get(key), path + "elem.")
        elif a[key] != b.get(key):
            out.append(path + key)
    return out


def test_every_schema_type_has_upstreams_kind_elements_equality_and_subtypes():
    """All 2584 schemas, every argument and return, both sides.

    One known divergence is allowed, by name and by shape: an alias-annotated
    top-level `Tensor(a!)` is `TensorType.get()` *by identity* upstream and is
    not here, because the argument's spelling keeps the annotation (the schema
    printer and the binder read it). `==`, `isinstance` and `kind()` agree for
    those too; only `is` differs. Every other difference is a failure.
    """
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    if not _have_upstream():
        _skip.skip("no upstream torch on sys.executable to compare with")
        return
    shim = _run(_FINGERPRINT_PROBE, True, str(NATIVE_FUNCTIONS))
    up = _run(_FINGERPRINT_PROBE, False, str(NATIVE_FUNCTIONS))
    assert shim["is_shim"] and not up["is_shim"], "the two probes got the wrong torch"
    assert shim["names"] == up["names"], "the two sides parsed different schema sets"
    compared, known, bad = 0, 0, []
    for name, spellings, srow, urow in zip(shim["names"], shim["spell"], shim["rows"], up["rows"]):
        assert len(srow) == len(urow), (name, len(srow), len(urow))
        for spelling, s, u in zip(spellings, srow, urow):
            compared += 1
            fields = _diff(u, s)
            if fields == ["is_Tensor_singleton"] and _alias_annotated(spelling):
                known += 1
                continue
            if fields:
                bad.append((name, spelling, fields, {f: (u.get(f), s.get(f)) for f in fields}))
    for ref, u in up["refs"].items():
        fields = _diff(u, shim["refs"].get(ref))
        if fields:
            bad.append(("<reference>", ref, fields, None))
    assert compared >= 12000, f"only {compared} schema types compared"
    assert not bad, f"{len(bad)} of {compared} schema types differ from upstream; first five: {bad[:5]}"
    print(f"JITTYPES: {compared} schema types and {len(up['refs'])} reference types compared, "
          f"0 differ; {known} alias-annotated tensors differ only in `is TensorType.get()`")


# ---------------------------------------------------------------------------
# the measured table
# ---------------------------------------------------------------------------

_TABLE_PROBE = r'''
import json
import torch

C = torch._C


def r(fn):
    try:
        v = fn()
    except Exception as exc:
        return "raises " + type(exc).__name__
    if isinstance(v, (bool, int, str, type(None))):
        return v
    if isinstance(v, (list, tuple)):
        return [x if isinstance(x, (bool, int, str, type(None))) else _kind(x) for x in v]
    return _kind(v)


def _kind(v):
    try:
        return "type " + v.kind()
    except Exception:
        return "object " + type(v).__name__


def arg(op, i):
    schema = op._schema
    return (schema.returns[0] if i == -1 else schema.arguments[i]).type


A = torch.ops.aten
bias = lambda: arg(A.convolution.default, 2)       # Tensor? bias
stride = lambda: arg(A.convolution.default, 3)     # SymInt[] stride
tensors = lambda: arg(A.cat.default, 0)            # Tensor[] tensors
self_ = lambda: arg(A.add_.Tensor, 0)              # Tensor(a!) self
indices = lambda: arg(A.index.Tensor, 1)           # Tensor?[] indices
dim = lambda: arg(A.sum.dim_IntList, 1)            # int[1]? dim
dtype = lambda: arg(A.sum.dim_IntList, 3)          # ScalarType? dtype
size = lambda: arg(A.zeros.default, 0)             # SymInt[] size
cmin = lambda: arg(A.clamp.default, 1)             # Scalar? min
sym_ret = lambda: arg(A.sym_size.int, -1)          # -> SymInt
split_ret = lambda: arg(A.split.Tensor, -1)        # -> Tensor(a)[]


def emit_list_branch(val_type):
    """executorch/exir/emit/_emitter.py:558-566 then `_emit_list` 307-331, as written there:
    which branch an argument of this schema type takes."""
    if isinstance(val_type, torch.OptionalType):
        val_type = val_type.getElementType()
    assert isinstance(val_type, torch.ListType)
    val_type = val_type.getElementType()
    for name in ("BoolType", "IntType", "FloatType", "TensorType"):
        if isinstance(val_type, getattr(torch, name)):
            return name
    if isinstance(val_type, torch.OptionalType):
        actual_type = val_type.getElementType()
        if isinstance(actual_type, torch.TensorType):
            return "OptionalTensorList"
    return "Unknown list type"


def arg_validator_kind(t):
    """executorch/exir/verification/arg_validator.py:81-89, as written there."""
    if isinstance(t, torch.TensorType) or t == torch.OptionalType.ofTensor():
        return "tensor"
    elif t == torch.ListType.ofTensors():
        return "tensor list"
    return "other"


out = {
    # exir, arg_validator.py:81-89
    "exir av bias": r(lambda: arg_validator_kind(bias())),
    "exir av self": r(lambda: arg_validator_kind(self_())),
    "exir av tensors": r(lambda: arg_validator_kind(tensors())),
    "exir av split ret": r(lambda: arg_validator_kind(split_ret())),
    "exir av indices": r(lambda: arg_validator_kind(indices())),
    "exir av stride": r(lambda: arg_validator_kind(stride())),
    # exir, _emitter.py:558-566 + 307-331
    "exir emit dim": r(lambda: emit_list_branch(dim())),
    "exir emit stride": r(lambda: emit_list_branch(stride())),
    "exir emit tensors": r(lambda: emit_list_branch(tensors())),
    "exir emit indices": r(lambda: emit_list_branch(indices())),
    "exir emit inferred int list": r(lambda: emit_list_branch(torch.ListType(torch.IntType.get()))),
    "exir emit inferred tensor list": r(lambda: emit_list_branch(torch.ListType(torch.TensorType.get()))),
    # exir, _emitter.py:1268
    "exir emit ofInts": r(lambda: emit_list_branch(torch.ListType.ofInts())),
    # the parametrised types themselves
    "ofTensor": r(lambda: C.OptionalType.ofTensor()),
    "ofTensor elem is Tensor": r(lambda: C.OptionalType.ofTensor().getElementType() is C.TensorType.get()),
    "ofTensor is ofTensor": r(lambda: C.OptionalType.ofTensor() is C.OptionalType.ofTensor()),
    "ofTensors elem": r(lambda: C.ListType.ofTensors().getElementType()),
    "ofInts elem": r(lambda: C.ListType.ofInts().getElementType()),
    "ofFloats elem": r(lambda: C.ListType.ofFloats().getElementType()),
    "ofBools elem": r(lambda: C.ListType.ofBools().getElementType()),
    "ofStrings elem": r(lambda: C.ListType.ofStrings().getElementType()),
    "ofComplexDoubles elem": r(lambda: C.ListType.ofComplexDoubles().getElementType()),
    "ListType(Int) == ofInts": r(lambda: C.ListType(C.IntType.get()) == C.ListType.ofInts()),
    "ListType(SymInt) == ofInts": r(lambda: C.ListType(C.SymIntType.get()) == C.ListType.ofInts()),
    "ListType(SymInt) elem": r(lambda: C.ListType(C.SymIntType.get()).getElementType()),
    "Optional(Optional(Tensor)) == ofTensor": r(lambda: C.OptionalType(C.OptionalType.ofTensor()) == C.OptionalType.ofTensor()),
    "Optional(ofInts) elem": r(lambda: C.OptionalType(C.ListType.ofInts()).getElementType()),
    "ListType(ofTensor) == indices": r(lambda: C.ListType(C.OptionalType.ofTensor()) == indices()),
    "ofTensors == ofInts": r(lambda: C.ListType.ofTensors() == C.ListType.ofInts()),
    "ofTensor == Tensor": r(lambda: C.OptionalType.ofTensor() == C.TensorType.get()),
    # schema argument types
    "bias == ofTensor": r(lambda: bias() == C.OptionalType.ofTensor()),
    "bias != ofTensor": r(lambda: bias() != C.OptionalType.ofTensor()),
    "bias isinstance OptionalType": r(lambda: isinstance(bias(), C.OptionalType)),
    "bias isinstance TensorType": r(lambda: isinstance(bias(), C.TensorType)),
    "stride == ofInts": r(lambda: stride() == C.ListType.ofInts()),
    "stride elem isinstance IntType": r(lambda: isinstance(stride().getElementType(), C.IntType)),
    "stride elem isinstance SymIntType": r(lambda: isinstance(stride().getElementType(), C.SymIntType)),
    "size == ofInts": r(lambda: size() == C.ListType.ofInts()),
    "tensors == ofTensors": r(lambda: tensors() == C.ListType.ofTensors()),
    "split ret == ofTensors": r(lambda: split_ret() == C.ListType.ofTensors()),
    "self == Tensor": r(lambda: self_() == C.TensorType.get()),
    "self isinstance TensorType": r(lambda: isinstance(self_(), C.TensorType)),
    "indices isinstance ListType": r(lambda: isinstance(indices(), C.ListType)),
    "indices isinstance OptionalType": r(lambda: isinstance(indices(), C.OptionalType)),
    "indices elem": r(lambda: indices().getElementType()),
    "indices elem elem is Tensor": r(lambda: indices().getElementType().getElementType() is C.TensorType.get()),
    "indices == ofTensors": r(lambda: indices() == C.ListType.ofTensors()),
    "dim kind": r(lambda: dim().kind()),
    "dim elem": r(lambda: dim().getElementType()),
    "dim elem elem": r(lambda: dim().getElementType().getElementType()),
    "dim == Optional(ofInts)": r(lambda: dim() == C.OptionalType(C.ListType.ofInts())),
    "dtype == Optional(Int)": r(lambda: dtype() == C.OptionalType(C.IntType.get())),
    "clamp min == Optional(Number)": r(lambda: cmin() == C.OptionalType(C.NumberType.get())),
    "sym ret == Int": r(lambda: sym_ret() == C.IntType.get()),
    "sym ret == SymInt": r(lambda: sym_ret() == C.SymIntType.get()),
    "sym ret isinstance IntType": r(lambda: isinstance(sym_ret(), C.IntType)),
    "sym ret isinstance SymIntType": r(lambda: isinstance(sym_ret(), C.SymIntType)),
    "hasattr Tensor getElementType": r(lambda: hasattr(C.TensorType.get(), "getElementType")),
    "hasattr ofInts getElementType": r(lambda: hasattr(C.ListType.ofInts(), "getElementType")),
    "bias isinstance Type": r(lambda: isinstance(bias(), C.Type)),
    "containedTypes indices": r(lambda: indices().containedTypes()),
    # singletons
    "SymIntType.get kind": r(lambda: C.SymIntType.get().kind()),
    "SymIntType.get == IntType.get": r(lambda: C.SymIntType.get() == C.IntType.get()),
    "SymBoolType.get kind": r(lambda: C.SymBoolType.get().kind()),
    "SymBoolType.get == BoolType.get": r(lambda: C.SymBoolType.get() == C.BoolType.get()),
    "NumberType.get kind": r(lambda: C.NumberType.get().kind()),
    # subtypes
    "Int <: Number": r(lambda: C.IntType.get().isSubtypeOf(C.NumberType.get())),
    "Bool <: Number": r(lambda: C.BoolType.get().isSubtypeOf(C.NumberType.get())),
    "SymInt <: Int": r(lambda: C.SymIntType.get().isSubtypeOf(C.IntType.get())),
    "Tensor <: ofTensor": r(lambda: C.TensorType.get().isSubtypeOf(C.OptionalType.ofTensor())),
    "ofTensor <: Tensor": r(lambda: C.OptionalType.ofTensor().isSubtypeOf(C.TensorType.get())),
    "None <: ofTensor": r(lambda: C.NoneType.get().isSubtypeOf(C.OptionalType.ofTensor())),
    "ofInts <: List(Number)": r(lambda: C.ListType.ofInts().isSubtypeOf(C.ListType(C.NumberType.get()))),
    "stride <: ofInts": r(lambda: stride().isSubtypeOf(C.ListType.ofInts())),
    # what upstream refuses, by exception type
    "ListType()": r(lambda: C.ListType()),
    "ListType(int)": r(lambda: C.ListType(int)),
    "ListType('int')": r(lambda: C.ListType("int")),
    "OptionalType()": r(lambda: C.OptionalType()),
    "OptionalType(None type)": r(lambda: C.OptionalType(C.NoneType.get())),
    "TensorType()": r(lambda: C.TensorType()),
    "IntType()": r(lambda: C.IntType()),
    "Tensor.getElementType()": r(lambda: C.TensorType.get().getElementType()),
    "isSubtypeOf(int)": r(lambda: C.IntType.get().isSubtypeOf(int)),
}
print(json.dumps({"is_shim": hasattr(C, "_aten_implemented"), "out": out}))
'''

#: Upstream's answers, torch 2.13.0 (`.caches/spike-venv`), measured with
#: `_TABLE_PROBE` on 2026-11-15. "raises X" is the exception type upstream
#: raises; "type K" is a JIT type whose `kind()` is K.
UPSTREAM_TABLE = {
    'exir av bias': 'tensor',
    'exir av self': 'tensor',
    'exir av tensors': 'tensor list',
    'exir av split ret': 'tensor list',
    'exir av indices': 'other',
    'exir av stride': 'other',
    'exir emit dim': 'IntType',
    'exir emit stride': 'IntType',
    'exir emit tensors': 'TensorType',
    'exir emit indices': 'OptionalTensorList',
    'exir emit inferred int list': 'IntType',
    'exir emit inferred tensor list': 'TensorType',
    'exir emit ofInts': 'IntType',
    'ofTensor': 'type OptionalType',
    'ofTensor elem is Tensor': True,
    'ofTensor is ofTensor': True,
    'ofTensors elem': 'type TensorType',
    'ofInts elem': 'type IntType',
    'ofFloats elem': 'type FloatType',
    'ofBools elem': 'type BoolType',
    'ofStrings elem': 'type StringType',
    'ofComplexDoubles elem': 'type ComplexType',
    'ListType(Int) == ofInts': True,
    'ListType(SymInt) == ofInts': False,
    'ListType(SymInt) elem': 'type SymIntType',
    'Optional(Optional(Tensor)) == ofTensor': True,
    'Optional(ofInts) elem': 'type ListType',
    'ListType(ofTensor) == indices': True,
    'ofTensors == ofInts': False,
    'ofTensor == Tensor': False,
    'bias == ofTensor': True,
    'bias != ofTensor': False,
    'bias isinstance OptionalType': True,
    'bias isinstance TensorType': False,
    'stride == ofInts': True,
    'stride elem isinstance IntType': True,
    'stride elem isinstance SymIntType': False,
    'size == ofInts': True,
    'tensors == ofTensors': True,
    'split ret == ofTensors': True,
    'self == Tensor': True,
    'self isinstance TensorType': True,
    'indices isinstance ListType': True,
    'indices isinstance OptionalType': False,
    'indices elem': 'type OptionalType',
    'indices elem elem is Tensor': True,
    'indices == ofTensors': False,
    'dim kind': 'OptionalType',
    'dim elem': 'type ListType',
    'dim elem elem': 'type IntType',
    'dim == Optional(ofInts)': True,
    'dtype == Optional(Int)': True,
    'clamp min == Optional(Number)': True,
    'sym ret == Int': True,
    'sym ret == SymInt': False,
    'sym ret isinstance IntType': True,
    'sym ret isinstance SymIntType': False,
    'hasattr Tensor getElementType': False,
    'hasattr ofInts getElementType': True,
    'bias isinstance Type': True,
    'containedTypes indices': ['type OptionalType'],
    'SymIntType.get kind': 'SymIntType',
    'SymIntType.get == IntType.get': False,
    'SymBoolType.get kind': 'SymBoolType',
    'SymBoolType.get == BoolType.get': False,
    'NumberType.get kind': 'NumberType',
    'Int <: Number': True,
    'Bool <: Number': False,
    'SymInt <: Int': False,
    'Tensor <: ofTensor': True,
    'ofTensor <: Tensor': False,
    'None <: ofTensor': True,
    'ofInts <: List(Number)': False,
    'stride <: ofInts': True,
    'ListType()': 'raises TypeError',
    'ListType(int)': 'raises TypeError',
    "ListType('int')": 'raises TypeError',
    'OptionalType()': 'raises TypeError',
    'OptionalType(None type)': 'raises RuntimeError',
    'TensorType()': 'raises TypeError',
    'IntType()': 'raises TypeError',
    'Tensor.getElementType()': 'raises AttributeError',
    'isSubtypeOf(int)': 'raises TypeError',
}


def test_exir_expressions_and_the_constructors_answer_as_upstream_measured():
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    got = _run(_TABLE_PROBE, True)
    assert got["is_shim"], "the shim-side probe got upstream torch"
    wrong = {k: (v, got["out"].get(k)) for k, v in UPSTREAM_TABLE.items() if got["out"].get(k) != v}
    assert not wrong, f"{len(wrong)} of {len(UPSTREAM_TABLE)} rows differ (upstream, shim): {wrong}"
    assert set(got["out"]) == set(UPSTREAM_TABLE), "the probe and the table name different rows"
    if _have_upstream():
        # The table is upstream's, not a wish: re-measure it.
        up = _run(_TABLE_PROBE, False)
        assert not up["is_shim"]
        drift = {k: (v, up["out"].get(k)) for k, v in UPSTREAM_TABLE.items() if up["out"].get(k) != v}
        assert not drift, f"upstream no longer answers the recorded table: {drift}"
    print(f"JITTYPES: {len(UPSTREAM_TABLE)} measured rows agree")


_REFUSAL_PROBE = r'''
import json
import torch

C = torch._C
assert hasattr(C, "_aten_implemented"), "upstream torch, not the shim"
out = {}
def r(name, fn):
    try:
        v = fn()
    except Exception as exc:
        out[name] = [type(exc).__name__, str(exc)]
    else:
        out[name] = ["returned", type(v).__name__]
r("TupleType([Int])", lambda: C.TupleType([C.IntType.get()]))
r("DictType(str, Tensor)", lambda: C.DictType(C.StringType.get(), C.TensorType.get()))
r("UnionType([Int, Tensor])", lambda: C.UnionType([C.IntType.get(), C.TensorType.get()]))
r("FutureType(Tensor)", lambda: C.FutureType(C.TensorType.get()))
r("kind of an unmeasured base", lambda: C.parse_schema("aten::x(Dimname d) -> Tensor").arguments[0].type.kind())
print(json.dumps(out))
'''


def test_a_type_the_shim_cannot_represent_refuses_by_name():
    """Upstream builds these (`Tuple[int]`, `Dict[str, Tensor]`, ...). The shim
    has no spelling for them, so each one refuses naming what it was asked to
    build, instead of handing back the placeholder instance it used to. The
    last row is a base spelling whose JIT kind was never measured (upstream's
    own parser rejects `Dimname`; the shim's accepts it)."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    got = _run(_REFUSAL_PROBE, True)
    for name, (kind, text) in got.items():
        assert kind == "NotImplementedError", (name, kind, text)
        assert "torch._C shim" in text, (name, text)
    assert "TupleType" in got["TupleType([Int])"][1]
    assert "DictType" in got["DictType(str, Tensor)"][1]
    assert "Dimname" in got["kind of an unmeasured base"][1]


if __name__ == "__main__":
    sys.exit(1 if _skip.run_tests(
        [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)],
        suite="test_jit_types",
    ) else 0)
