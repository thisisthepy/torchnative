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
    (isinstance over the `_C` type classes, the exact class `type(t)`, `kind()`, element chain, `==`,
    `!=`, `isSubtypeOf` in both directions and `containedTypes` against a set
    of reference types built from the public constructors);
  * a measured table: exir's own expressions on real operator schemas, the
    constructors, the singletons, the subtype rules and the refusals, with
    upstream's answers written down below and re-checked against upstream
    when it is importable.

A third test (wall W8, CI run 43793269977) compares `str()` and `repr()` of
every one of those types with upstream's text (`List[int]`,
`Optional[Tensor]`), and pins exir's own reading of it: its edge verifier
keeps an argument in the dtype table only when `str(arg.type)` is `Tensor`,
`Optional[Tensor]` or `List[Tensor]`, and the schema spelling `Tensor?` made
it drop `linear`'s `bias` and report float32 against float32 as a mismatch.

A fourth compares `Argument`'s attribute surface and its `is_out` /
`is_write` / `real_type` answers with upstream's on seven schemas (CI run
43808445902, exir's portable stage).

What these cannot find: `Argument.real_type` beyond those seven schemas (it
refuses by name for the `ScalarType`, `Layout`, `MemoryFormat` and
`DeviceIndex` bases, which upstream reads as types of their own), identity of fresh constructions (the shim
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
    # `type(t)` itself: exir's emitter and torch/fx/passes/reinplace.py compare
    # classes, not isinstance (#423).
    out["type"] = next((n for n, c in CLASSES if type(t) is c), "other " + type(t).__name__)
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
    # The schema spelling (`Tensor(a!)`), not `str()`: the allowance below
    # is decided on the alias annotation, which upstream's text drops.
    spell.append([getattr(a.type, "_spelling", str(a.type)) for a in items])
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


# ---------------------------------------------------------------------------
# str() and repr(), wall W8
# ---------------------------------------------------------------------------

_STR_PROBE = r"""
import json, sys
import torch

C = torch._C
L, O = C.ListType, C.OptionalType
rows, names = [], []
for line in open(sys.argv[1], encoding="utf-8"):
    if not line.startswith("- func:"):
        continue
    text = "aten::" + line[len("- func:"):].strip()
    try:
        s = C.parse_schema(text)
    except Exception:
        continue
    names.append(text.split("(", 1)[0])
    rows.append([[str(a.type), repr(a.type)] for a in list(s.arguments) + list(s.returns)])
refs = {n: str(getattr(C, n).get()) for n in (
    "TensorType", "IntType", "SymIntType", "SymBoolType", "FloatType", "ComplexType",
    "BoolType", "StringType", "NumberType", "NoneType", "DeviceObjType", "AnyType",
    "_GeneratorType", "StreamObjType")}
refs.update({
    "ofTensor": str(O.ofTensor()), "ofTensors": str(L.ofTensors()), "ofInts": str(L.ofInts()),
    "L(SymInt)": str(L(C.SymIntType.get())), "L(ofTensor)": str(L(O.ofTensor())),
    "O(ofInts)": str(O(L.ofInts())), "O(SymInt)": str(O(C.SymIntType.get())),
})
ops = {}
for op in ("linear.default", "split_with_sizes.default", "add_.Tensor", "sum.dim_IntList",
           "index.Tensor", "clamp.default", "native_layer_norm.default"):
    p, o = op.split(".")
    s = getattr(getattr(torch.ops.aten, p), o)._schema
    ops[op] = [[a.name, str(a.type)] for a in list(s.arguments) + list(s.returns)]
# executorch/exir/dialects/edge/spec/utils.py `is_tensor_arg`, verbatim.
ops["exir is_tensor_arg(linear)"] = [
    a.name for a in list(torch.ops.aten.linear.default._schema.arguments)
    + list(torch.ops.aten.linear.default._schema.returns)
    if str(a.type) in ["Tensor", "Optional[Tensor]", "List[Tensor]"]]
print(json.dumps({"is_shim": hasattr(C, "_aten_implemented"), "names": names,
                  "rows": rows, "refs": refs, "ops": ops}))
"""

#: Upstream torch 2.13.0's answers, measured with the probe above (no vendored
#: tree on the path). `str` and `repr` agree upstream for every type measured.
UPSTREAM_STR_OPS = {
    "linear.default": [["input", "Tensor"], ["weight", "Tensor"],
                       ["bias", "Optional[Tensor]"], ["", "Tensor"]],
    "split_with_sizes.default": [["self", "Tensor"], ["split_sizes", "List[int]"],
                                 ["dim", "int"], ["", "List[Tensor]"]],
    "add_.Tensor": [["self", "Tensor"], ["other", "Tensor"], ["alpha", "number"],
                    ["", "Tensor"]],
    "sum.dim_IntList": [["self", "Tensor"], ["dim", "Optional[List[int]]"],
                        ["keepdim", "bool"], ["dtype", "Optional[int]"], ["", "Tensor"]],
    "index.Tensor": [["self", "Tensor"], ["indices", "List[Optional[Tensor]]"],
                     ["", "Tensor"]],
    "clamp.default": [["self", "Tensor"], ["min", "Optional[number]"],
                      ["max", "Optional[number]"], ["", "Tensor"]],
    "native_layer_norm.default": [["input", "Tensor"], ["normalized_shape", "List[int]"],
                                  ["weight", "Optional[Tensor]"], ["bias", "Optional[Tensor]"],
                                  ["eps", "float"], ["", "Tensor"], ["", "Tensor"],
                                  ["", "Tensor"]],
    "exir is_tensor_arg(linear)": ["input", "weight", "bias", ""],
}

UPSTREAM_STR_REFS = {
    "TensorType": "Tensor", "IntType": "int", "SymIntType": "int", "SymBoolType": "bool",
    "FloatType": "float", "ComplexType": "complex", "BoolType": "bool", "StringType": "str",
    "NumberType": "number", "NoneType": "NoneType", "DeviceObjType": "Device",
    "AnyType": "Any", "_GeneratorType": "Generator", "StreamObjType": "Stream",
    "ofTensor": "Optional[Tensor]", "ofTensors": "List[Tensor]", "ofInts": "List[int]",
    "L(SymInt)": "List[int]", "L(ofTensor)": "List[Optional[Tensor]]",
    "O(ofInts)": "Optional[List[int]]", "O(SymInt)": "Optional[int]",
}


def test_str_of_every_schema_type_is_upstreams_text():
    """`str(arg.type)` is upstream's `List[int]` / `Optional[Tensor]`, not the
    schema spelling, for every argument and return of every schema, and
    `repr` with it. Measured upstream: `str(t) == t.annotation_str` for all
    19600 types of `_jit_get_all_schemas()`.

    The measured rows hold on the shim without upstream on the path; the full
    width, when upstream is importable, compares every type with upstream's
    own answer rather than with the table.
    """
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    shim = _run(_STR_PROBE, True, str(NATIVE_FUNCTIONS))
    assert shim["is_shim"], "the shim-side probe got upstream torch"
    wrong = {k: (v, shim["ops"].get(k)) for k, v in UPSTREAM_STR_OPS.items()
             if shim["ops"].get(k) != v}
    assert not wrong, f"{len(wrong)} of {len(UPSTREAM_STR_OPS)} ops differ (upstream, shim): {wrong}"
    wrong = {k: (v, shim["refs"].get(k)) for k, v in UPSTREAM_STR_REFS.items()
             if shim["refs"].get(k) != v}
    assert not wrong, f"{len(wrong)} reference types differ (upstream, shim): {wrong}"
    if not _have_upstream():
        _skip.skip("no upstream torch on sys.executable for the full-width str comparison")
        return
    up = _run(_STR_PROBE, False, str(NATIVE_FUNCTIONS))
    assert not up["is_shim"], "the upstream-side probe got the shim"
    assert up["ops"] == UPSTREAM_STR_OPS, "upstream no longer answers the recorded rows"
    assert up["refs"] == UPSTREAM_STR_REFS, "upstream no longer answers the recorded references"
    assert shim["names"] == up["names"], "the two sides parsed different schema sets"
    compared, bad = 0, []
    for name, srow, urow in zip(shim["names"], shim["rows"], up["rows"]):
        assert len(srow) == len(urow), (name, len(srow), len(urow))
        for s, u in zip(srow, urow):
            compared += 1
            if s != u:
                bad.append((name, u, s))
    assert compared >= 12000, f"only {compared} schema types compared"
    assert not bad, f"{len(bad)} of {compared} types print differently (upstream, shim); first five: {bad[:5]}"
    print(f"JITTYPES: str/repr of {compared} schema types agree with upstream")


#: `Argument` of a parsed schema, field by field: name, `str(type)`,
#: `str(real_type)`, N, kwarg_only, is_out, is_write, has_default_value(),
#: `repr(default_value)`, alias_info as (is_write, before_set, after_set).
#: The same text runs on both sides.
_ARG_PROBE = r'''
import json, sys
import torch
SCHEMAS = [
    "aten::relu.out(Tensor self, *, Tensor(a!) out) -> Tensor(a!)",
    "aten::permute_copy.out(Tensor self, int[] dims, *, Tensor(a!) out) -> Tensor(a!)",
    "aten::addmm.out(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1, Scalar alpha=1, Tensor(a!) out) -> Tensor(a!)",
    "aten::linear(Tensor input, Tensor weight, Tensor? bias=None) -> Tensor",
    "aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)",
    "aten::max.dim_max(Tensor self, int dim, bool keepdim=False, *, Tensor(a!) max, Tensor(b!) max_values) -> (Tensor(a!) values, Tensor(b!) indices)",
    "aten::view(Tensor(a) self, SymInt[] size) -> Tensor(a)",
]
def row(a):
    ai = a.alias_info
    try:
        real = str(a.real_type)
    except Exception as exc:
        real = "raised " + type(exc).__name__
    return [a.name, str(a.type), real, a.N, a.kwarg_only, a.is_out, a.is_write,
            a.has_default_value(), repr(a.default_value),
            None if ai is None else [ai.is_write, sorted(ai.before_set), sorted(ai.after_set)]]
out = {"is_shim": hasattr(torch._C, "_aten_implemented"), "rows": {}, "dir": []}
for text in SCHEMAS:
    s = torch._C.parse_schema(text)
    out["rows"][text.split("(")[0]] = {"arguments": [row(a) for a in s.arguments],
                                       "returns": [row(r) for r in s.returns]}
a = torch._C.parse_schema(SCHEMAS[0]).arguments[1]
out["dir"] = sorted(n for n in dir(a) if not n.startswith("_"))
print(json.dumps(out))
'''

#: Upstream's answers, measured on torch 2.13.0 (2026-11-16) with `_ARG_PROBE`.
UPSTREAM_ARGUMENTS = {'dir': ['N',
         'alias_info',
         'default_value',
         'has_default_value',
         'is_out',
         'is_write',
         'kwarg_only',
         'name',
         'real_type',
         'type'],
 'rows': {'aten::add_.Tensor': {'arguments': [['self',
                                               'Tensor',
                                               'Tensor',
                                               None,
                                               False,
                                               False,
                                               True,
                                               False,
                                               'None',
                                               [True, ['a'], ['a']]],
                                              ['other',
                                               'Tensor',
                                               'Tensor',
                                               None,
                                               False,
                                               False,
                                               False,
                                               False,
                                               'None',
                                               None],
                                              ['alpha',
                                               'number',
                                               'number',
                                               None,
                                               True,
                                               False,
                                               False,
                                               True,
                                               '1',
                                               None]],
                                'returns': [['',
                                             'Tensor',
                                             'Tensor',
                                             None,
                                             False,
                                             False,
                                             True,
                                             False,
                                             'None',
                                             [True, ['a'], ['a']]]]},
          'aten::addmm.out': {'arguments': [['self',
                                             'Tensor',
                                             'Tensor',
                                             None,
                                             False,
                                             False,
                                             False,
                                             False,
                                             'None',
                                             None],
                                            ['mat1',
                                             'Tensor',
                                             'Tensor',
                                             None,
                                             False,
                                             False,
                                             False,
                                             False,
                                             'None',
                                             None],
                                            ['mat2',
                                             'Tensor',
                                             'Tensor',
                                             None,
                                             False,
                                             False,
                                             False,
                                             False,
                                             'None',
                                             None],
                                            ['beta',
                                             'number',
                                             'number',
                                             None,
                                             True,
                                             False,
                                             False,
                                             True,
                                             '1',
                                             None],
                                            ['alpha',
                                             'number',
                                             'number',
                                             None,
                                             True,
                                             False,
                                             False,
                                             True,
                                             '1',
                                             None],
                                            ['out',
                                             'Tensor',
                                             'Tensor',
                                             None,
                                             True,
                                             True,
                                             True,
                                             False,
                                             'None',
                                             [True, ['a'], ['a']]]],
                              'returns': [['',
                                           'Tensor',
                                           'Tensor',
                                           None,
                                           False,
                                           False,
                                           True,
                                           False,
                                           'None',
                                           [True, ['a'], ['a']]]]},
          'aten::linear': {'arguments': [['input',
                                          'Tensor',
                                          'Tensor',
                                          None,
                                          False,
                                          False,
                                          False,
                                          False,
                                          'None',
                                          None],
                                         ['weight',
                                          'Tensor',
                                          'Tensor',
                                          None,
                                          False,
                                          False,
                                          False,
                                          False,
                                          'None',
                                          None],
                                         ['bias',
                                          'Optional[Tensor]',
                                          'Optional[Tensor]',
                                          None,
                                          False,
                                          False,
                                          False,
                                          True,
                                          'None',
                                          None]],
                           'returns': [['',
                                        'Tensor',
                                        'Tensor',
                                        None,
                                        False,
                                        False,
                                        False,
                                        False,
                                        'None',
                                        None]]},
          'aten::max.dim_max': {'arguments': [['self',
                                               'Tensor',
                                               'Tensor',
                                               None,
                                               False,
                                               False,
                                               False,
                                               False,
                                               'None',
                                               None],
                                              ['dim',
                                               'int',
                                               'int',
                                               None,
                                               False,
                                               False,
                                               False,
                                               False,
                                               'None',
                                               None],
                                              ['keepdim',
                                               'bool',
                                               'bool',
                                               None,
                                               False,
                                               False,
                                               False,
                                               True,
                                               'False',
                                               None],
                                              ['max',
                                               'Tensor',
                                               'Tensor',
                                               None,
                                               True,
                                               True,
                                               True,
                                               False,
                                               'None',
                                               [True, ['a'], ['a']]],
                                              ['max_values',
                                               'Tensor',
                                               'Tensor',
                                               None,
                                               True,
                                               True,
                                               True,
                                               False,
                                               'None',
                                               [True, ['b'], ['b']]]],
                                'returns': [['values',
                                             'Tensor',
                                             'Tensor',
                                             None,
                                             False,
                                             False,
                                             True,
                                             False,
                                             'None',
                                             [True, ['a'], ['a']]],
                                            ['indices',
                                             'Tensor',
                                             'Tensor',
                                             None,
                                             False,
                                             False,
                                             True,
                                             False,
                                             'None',
                                             [True, ['b'], ['b']]]]},
          'aten::permute_copy.out': {'arguments': [['self',
                                                    'Tensor',
                                                    'Tensor',
                                                    None,
                                                    False,
                                                    False,
                                                    False,
                                                    False,
                                                    'None',
                                                    None],
                                                   ['dims',
                                                    'List[int]',
                                                    'List[int]',
                                                    None,
                                                    False,
                                                    False,
                                                    False,
                                                    False,
                                                    'None',
                                                    None],
                                                   ['out',
                                                    'Tensor',
                                                    'Tensor',
                                                    None,
                                                    True,
                                                    True,
                                                    True,
                                                    False,
                                                    'None',
                                                    [True, ['a'], ['a']]]],
                                     'returns': [['',
                                                  'Tensor',
                                                  'Tensor',
                                                  None,
                                                  False,
                                                  False,
                                                  True,
                                                  False,
                                                  'None',
                                                  [True, ['a'], ['a']]]]},
          'aten::relu.out': {'arguments': [['self',
                                            'Tensor',
                                            'Tensor',
                                            None,
                                            False,
                                            False,
                                            False,
                                            False,
                                            'None',
                                            None],
                                           ['out',
                                            'Tensor',
                                            'Tensor',
                                            None,
                                            True,
                                            True,
                                            True,
                                            False,
                                            'None',
                                            [True, ['a'], ['a']]]],
                             'returns': [['',
                                          'Tensor',
                                          'Tensor',
                                          None,
                                          False,
                                          False,
                                          True,
                                          False,
                                          'None',
                                          [True, ['a'], ['a']]]]},
          'aten::view': {'arguments': [['self',
                                        'Tensor',
                                        'Tensor',
                                        None,
                                        False,
                                        False,
                                        False,
                                        False,
                                        'None',
                                        [False, ['a'], ['a']]],
                                       ['size',
                                        'List[int]',
                                        'List[int]',
                                        None,
                                        False,
                                        False,
                                        False,
                                        False,
                                        'None',
                                        None]],
                         'returns': [['',
                                      'Tensor',
                                      'Tensor',
                                      None,
                                      False,
                                      False,
                                      False,
                                      False,
                                      'None',
                                      [False, ['a'], ['a']]]]}}}


def test_argument_surface_and_out_flags_are_upstreams():
    """exir's portable stage reads `Argument.is_out` (`exir/common.py:72`,
    `exir/emit/_emitter.py:1542`) and its serializer `real_type`; the shim's
    `_Argument` had neither and the stage stopped on `'_Argument' object has
    no attribute 'is_out'` (CI run 43808445902). Every public attribute
    upstream's `Argument` has exists here, and for `relu.out`,
    `permute_copy.out`, `addmm.out`, `linear`, `add_.Tensor` (a written `self`
    that is not an out), `max.dim_max` (two outs) and `view` every field
    equals upstream's. Red when `is_out` ignores `kwarg_only` (the `add_`
    row), when it ignores the write flag (`addmm.out`'s `beta`), or when
    either property is removed."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    shim = _run(_ARG_PROBE, True)
    assert shim["is_shim"], "the shim-side probe got upstream torch"
    missing = sorted(set(UPSTREAM_ARGUMENTS["dir"]) - set(shim["dir"]))
    assert not missing, f"the shim's Argument lacks upstream's attributes: {missing}"
    bad = {k: (v, shim["rows"].get(k)) for k, v in UPSTREAM_ARGUMENTS["rows"].items()
           if shim["rows"].get(k) != v}
    assert not bad, f"{len(bad)} of {len(UPSTREAM_ARGUMENTS['rows'])} schemas differ (upstream, shim): {bad}"
    if not _have_upstream():
        _skip.skip("no upstream torch on sys.executable to re-measure the table")
        return
    up = _run(_ARG_PROBE, False)
    assert not up["is_shim"], "the upstream-side probe got the shim"
    assert {"dir": up["dir"], "rows": up["rows"]} == UPSTREAM_ARGUMENTS, \
        "upstream no longer answers the recorded Argument table"


_MEMORY_FORMAT_PROBE = r"""
import copy, json, pickle, torch

class PermuteClone(torch.nn.Module):
    def forward(self, x):
        return x.permute(1, 0).clone(memory_format=torch.contiguous_format)

ep = torch.export.export(PermuteClone(), (torch.zeros(2, 3),))
clone_node = next(
    n for n in ep.graph.nodes
    if n.op == "call_function" and "clone" in str(n.target)
)
mf = clone_node.kwargs["memory_format"]
# `executorch.exir.passes.memory_format_ops_pass.MemoryFormatOpsPass` runs
# `nkwargs = dict(copy.deepcopy(kwargs))` and then hands
# `nkwargs.pop("memory_format", torch.preserve_format)` to `get_dim_order`.
cmf = dict(copy.deepcopy(clone_node.kwargs))["memory_format"]
pmf = pickle.loads(pickle.dumps(mf))


def check(v, target):
    return {
        "repr": repr(v),
        "is": v is target,
        "eq": v == target,
        "in_list": v in [target],
        "in_none_list": v in [None, target],
        "hash_eq": hash(v) == hash(target),
        "ne_preserve": v != torch.preserve_format,
    }


cannot_create = {}
for kind in ("memory_format", "layout", "qscheme"):
    cls = getattr(torch, kind)
    try:
        cls()
        cannot_create[kind] = "CREATED"
    except TypeError as exc:
        cannot_create[kind] = str(exc)

singletons = {}
for name in ("contiguous_format", "preserve_format", "channels_last",
             "strided", "per_tensor_affine"):
    obj = getattr(torch, name)
    c = copy.copy(obj)
    d = copy.deepcopy(obj)
    p = pickle.loads(pickle.dumps(obj))
    singletons[name] = {
        "copy_is": c is obj,
        "deepcopy_is": d is obj,
        "deepcopy_eq": d == obj,
        "pickle_is": p is obj,
        "pickle_eq": p == obj,
    }

print(json.dumps({
    "is_shim": hasattr(torch._C, "_aten_implemented"),
    "node": check(mf, torch.contiguous_format),
    "after_deepcopy": check(cmf, torch.contiguous_format),
    "after_pickle": check(pmf, torch.contiguous_format),
    "cannot_create": cannot_create,
    "singletons": singletons,
}))
"""


def test_clone_node_memory_format_survives_graph_and_deepcopy():
    """Issue #306 (CI run 44423485167): inside `ConvertToSDPAPass`,
    `executorch/exir/dim_order_utils.py::get_dim_order` raised
    `Failed to generate dim_order for a given memory format:
    torch.contiguous_format` on `aten.clone.default(memory_format=
    torch.contiguous_format)`. `MemoryFormatOpsPass.call_operator` runs
    `nkwargs = dict(copy.deepcopy(kwargs))` before calling `get_dim_order`,
    and the shim's `torch.memory_format` had `_permissive_init` with no
    `__deepcopy__`, `__reduce__`, `__eq__` or `__hash__`, so `copy.deepcopy`
    constructed a fresh empty `memory_format` instance that was neither `is`
    nor `==` `torch.contiguous_format`. Checks `is`, `==`, `in [...]` and
    `hash` on the exported `clone` node's `kwargs['memory_format']` before and
    after `copy.deepcopy` and `pickle`, on the shim and on upstream side by
    side."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    if not _have_upstream():
        _skip.skip("no upstream torch on sys.executable")
        return
    shim = _run(_MEMORY_FORMAT_PROBE, True)
    up = _run(_MEMORY_FORMAT_PROBE, False)
    assert shim["is_shim"] and not up["is_shim"]
    print("   upstream clone kwargs['memory_format']:",
          json.dumps({k: up[k] for k in ("node", "after_deepcopy", "after_pickle")}))
    print("   shim     clone kwargs['memory_format']:",
          json.dumps({k: shim[k] for k in ("node", "after_deepcopy", "after_pickle")}))
    for key in ("node", "after_deepcopy", "after_pickle", "cannot_create", "singletons"):
        assert shim[key] == up[key], (
            f"{key} differs:\n  upstream {up[key]}\n  shim     {shim[key]}"
        )


if __name__ == "__main__":
    sys.exit(1 if _skip.run_tests(
        [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)],
        suite="test_jit_types",
    ) else 0)
