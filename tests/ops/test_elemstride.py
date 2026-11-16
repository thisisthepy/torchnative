"""A fresh elementwise result is laid out as upstream lays it out (issue #383).

Upstream's TensorIterator gives a freshly allocated result the stride order of
its operands: `exp` of a `permute(2, 0, 1)` input is not contiguous upstream,
it has the input's permutation, densely packed. This shim built every such
result row-major, so the values and the aliasing were right and the layout was
not, and a later `.view(-1)` succeeded here where upstream refuses (or the
reverse). CI run 43819391187 counted 215 (Linux) and 246 (macOS) rows of
`tests/bindings/test_cia_frame.py` whose only difference from upstream was
`is_contiguous()` of a descriptor, first differing at `aten.mul.Scalar` (114
and 145), `aten._scaled_dot_product_flash_attention_for_cpu.default` (99) and
`aten.bitwise_and.Tensor` (2).

The rule, measured on torch 2.13.0 before the implementation
(docs/kernels/VIEWS.md section 8): the result's stride is
`torch._prims_common.compute_elementwise_output_strides` of the tensor
operands broadcast to the result's shape, in argument order (the first
operand that orders two axes decides; a zero stride orders nothing); a
`preserve_format` copy (`_to_copy` with a dtype change, `bool()`, the
`*_like` factories) keeps a dense input's strides and otherwise takes that
permutation; the CPU flash attention output is `empty_like(query)` and its
logsumexp is a `(b, L, h)` buffer transposed. `masked_fill`, `pow.Scalar`,
reductions, `softmax`, `mm`, `cat` and `index_select` are row-major upstream
and stay row-major here.

**Every assertion is upstream's answer, computed in a separate subprocess from
the same probe text.** The one exception is
`test_the_oracle_is_what_the_measurement_recorded`, which checks the oracle
itself against the recorded measurement first. Layout fields (shape, stride,
offset, `is_contiguous`, dtype, storage sharing with the inputs, whether
`.view(-1)` succeeds) are compared exactly; values with a relative tolerance
of 1e-5, so an ulp difference a kernel already had is not read as a layout
defect (the values are compared exactly elsewhere, in golden).

What each test turns red under (nullifications, none run locally; CI runs them):

* `relay_dense_elementwise` returning the kernel's result unchanged, or the
  device scan never setting `strided`: `test_unary_and_scalar_results_...`
  (every permuted, transposed and permuted-with-gap row: stride,
  `is_contiguous`, `view_flat`), `test_tensor_tensor_results_...` and
  `test_the_dense_layout_is_the_meta_layout` (dense row-major, meta permuted);
* `dense_follows_operands` without its extra list (only the meta list): the
  `abs`, `sigmoid`, `floor`, `round`, `logical_not`, `bitwise_and`,
  `maximum` and `logical_and` rows;
* `relay_fresh_dense` (or `layout::copy_into_stride`) giving the row-major
  values the new strides without moving them: every relaid row's values;
* `elementwise_stride` reading the operands in reverse order: the `c+p` and
  `p+c`, `t01+t12` and `t12+t01` rows;
* adding `masked_fill` or `pow.Scalar` to the list, or relaying reductions:
  `test_row_major_results_stay_row_major`;
* the relay ignoring `memory_format=contiguous_format`:
  `zeros_like_contig|*` in `test_conversions_...`;
* `converted_in_format` returning candle's row-major conversion: the
  `double`, `bool`, `to_int32`, `to_bf16` and `to_copy_f64` rows of
  `test_conversions_...`;
* the flash relayout removed from `sdpa_flash_cpu`: every logsumexp row and
  the output rows for a non-contiguous query in
  `test_flash_attention_outputs_...`.

What this file cannot see: Metal and CUDA results (row-major there by
decision, docs/kernels/VIEWS.md section 8.4; no CI device run), the cost of the
extra copy (no measurement), results with no elements (left row-major: upstream
follows no single rule there) and ops with `out=` (out of scope).

Nothing here needs numpy or a network.
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


# ---------------------------------------------------------------------------
# The probe. Identical text on both sides; only the answers can differ.
# ---------------------------------------------------------------------------

_PROBE = r'''
import json
import math
import torch
import torch.nn.functional as F

A = torch.ops.aten
out = {"is_shim": hasattr(torch._C, "_aten_implemented")}


def flat(t):
    acc = []

    def walk(x):
        if isinstance(x, list):
            for y in x:
                walk(y)
        else:
            acc.append(float(x))

    walk(t.tolist())
    return acc


def describe(t, inputs, values=True):
    d = {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "offset": t.storage_offset(),
        "contiguous": t.is_contiguous(),
        "dtype": str(t.dtype),
    }
    if t.device.type != "meta":
        mine = t.untyped_storage()._cdata
        d["shares"] = any(mine == x.untyped_storage()._cdata
                          for x in inputs if isinstance(x, torch.Tensor))
        try:
            t.view(-1)
            d["view_flat"] = "ok"
        except Exception as exc:
            d["view_flat"] = type(exc).__name__
        if values:
            d["values"] = flat(t)
    return d


def run(section, name, fn, values=True):
    try:
        result, inputs = fn()
        out.setdefault(section, {})[name] = describe(result, inputs, values)
    except Exception as exc:
        out.setdefault(section, {})[name] = {"raised": type(exc).__name__, "message": str(exc)[:300]}


# Every input is built from `base(n, device)`, a 1-D tensor of n float32
# values, through views only, so its layout is the point.
def base(n, device="cpu"):
    if device == "meta":
        return torch.empty(n, device="meta")
    return torch.arange(n, dtype=torch.float32) / 7 + 0.25


LAYOUTS = {
    "contig": lambda d: base(24, d).view(2, 3, 4),
    "perm": lambda d: base(24, d).view(3, 4, 2).permute(2, 0, 1),
    "t01": lambda d: base(24, d).view(3, 2, 4).transpose(0, 1),
    "t12": lambda d: base(24, d).view(2, 4, 3).transpose(1, 2),
    "gap": lambda d: base(48, d).view(2, 3, 8)[:, :, ::2],
    "perm_gap": lambda d: base(48, d).view(3, 8, 2).permute(2, 0, 1)[:, :, ::2],
    "expanded": lambda d: base(4, d).view(1, 1, 4).expand(2, 3, 4),
    "expanded_mid": lambda d: base(8, d).view(2, 1, 4).expand(2, 3, 4),
    "offset": lambda d: base(30, d)[6:].view(2, 3, 4),
    "t2d": lambda d: base(6, d).view(3, 2).t(),
    "one_t": lambda d: base(1, d).view(1, 1).t(),
    "strided1d": lambda d: base(8, d)[::2],
    "unit_col": lambda d: base(8, d).view(2, 4)[:, :1],
}

# (i) unary ops and tensor-scalar binary ops, each `x -> result`.
UNARY = {
    "exp": lambda x: x.exp(),
    "neg": lambda x: -x,
    "abs": lambda x: x.abs(),
    "sigmoid": lambda x: x.sigmoid(),
    "relu": lambda x: x.relu(),
    "gelu": lambda x: F.gelu(x),
    "silu": lambda x: F.silu(x),
    "sqrt": lambda x: x.sqrt(),
    "tanh": lambda x: x.tanh(),
    "sin": lambda x: x.sin(),
    "floor": lambda x: x.floor(),
    "round": lambda x: x.round(),
    "clamp": lambda x: x.clamp(1.0, 2.5),
    "mul_scalar": lambda x: x * 2.0,
    "add_scalar_int": lambda x: x + 1,
    "rsub_scalar": lambda x: 1.0 - x,
    "div_scalar": lambda x: x / 3.0,
    "pow_tensor_scalar": lambda x: x ** 2,
    "remainder_scalar": lambda x: x % 0.75,
    "gt_scalar": lambda x: x > 1.5,
    "eq_scalar": lambda x: x == 0.25,
    "logical_not": lambda x: torch.logical_not(x > 1.5),
    "where_scalar_other": lambda x: torch.where(x > 1.5, x, 0.0),
    "where_scalar_self": lambda x: torch.where(x > 1.5, 0.0, x),
    "int_bitwise_and_scalar": lambda x: (x * 7).to(torch.int64) & 3,
    "int_times_float": lambda x: (x * 7).to(torch.int64) * 0.5,
}
out["unary"] = {}
for op_name, fn in UNARY.items():
    for key, make in LAYOUTS.items():
        run("unary", op_name + "|" + key, lambda fn=fn, make=make: (lambda x: (fn(x), [x]))(make("cpu")))

# (ii) tensor-tensor operands: (name, a, b). `row` and `col` broadcast.
def row(d):
    return base(4, d)


def col(d):
    return base(3, d).view(3, 1)


def zero_d(d):
    return base(1, d).view(())


def one(d):
    return base(1, d)


PAIRS = {
    "c+c": (LAYOUTS["contig"], LAYOUTS["contig"]),
    "p+p": (LAYOUTS["perm"], LAYOUTS["perm"]),
    "c+p": (LAYOUTS["contig"], LAYOUTS["perm"]),
    "p+c": (LAYOUTS["perm"], LAYOUTS["contig"]),
    "t01+t12": (LAYOUTS["t01"], LAYOUTS["t12"]),
    "t12+t01": (LAYOUTS["t12"], LAYOUTS["t01"]),
    "p+row": (LAYOUTS["perm"], row),
    "row+p": (row, LAYOUTS["perm"]),
    "t12+col": (LAYOUTS["t12"], col),
    "p+expanded": (LAYOUTS["perm"], LAYOUTS["expanded"]),
    "expanded+p": (LAYOUTS["expanded"], LAYOUTS["perm"]),
    "expanded+c": (LAYOUTS["expanded"], LAYOUTS["contig"]),
    "gap+p": (LAYOUTS["gap"], LAYOUTS["perm"]),
    "p+gap": (LAYOUTS["perm"], LAYOUTS["gap"]),
    "p+0d": (LAYOUTS["perm"], zero_d),
    "0d+p": (zero_d, LAYOUTS["perm"]),
    "p+one": (LAYOUTS["perm"], one),
    "t12slice+t12": (lambda d: LAYOUTS["t12"](d)[:, :, :1], LAYOUTS["t12"]),
    "pcol+row3": (lambda d: base(6, d).view(3, 2).t().unsqueeze(-1), lambda d: base(4, d).view(1, 1, 4)),
    "row3+pcol": (lambda d: base(4, d).view(1, 1, 4), lambda d: base(6, d).view(3, 2).t().unsqueeze(-1)),
    "t2d+lead": (LAYOUTS["t2d"], lambda d: base(4, d).view(4, 1, 1)),
    "lead+t2d": (lambda d: base(4, d).view(4, 1, 1), LAYOUTS["t2d"]),
}
BINARY = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "div": lambda a, b: a / b,
    "eq": lambda a, b: a == b,
    "lt": lambda a, b: a < b,
    "maximum": lambda a, b: torch.maximum(a, b),
    "bitwise_and_bool": lambda a, b: (a > 1.0) & (b > 1.0),
    "logical_and": lambda a, b: torch.logical_and(a > 1.0, b > 1.0),
    "where_self": lambda a, b: torch.where(a > 1.0, a, b),
    "int_plus_float": lambda a, b: (a * 7).to(torch.int64) + b,
}
out["binary"] = {}
for op_name, fn in BINARY.items():
    for key, (ma, mb) in PAIRS.items():
        run("binary", op_name + "|" + key,
            lambda fn=fn, ma=ma, mb=mb: (lambda a, b: (fn(a, b), [a, b]))(ma("cpu"), mb("cpu")))

# (iii) converting copies and the `*_like` factories.
CONVERT = {
    "double": (lambda x: x.double(), True),
    "double_float": (lambda x: x.double().float(), True),
    "bool": (lambda x: (x - 0.25).bool(), True),
    "to_int32": (lambda x: (x * 7).to(torch.int32), True),
    "to_bf16": (lambda x: x.to(torch.bfloat16), True),
    "to_copy_f64": (lambda x: A._to_copy.default(x, dtype=torch.float64), True),
    "to_copy_f64_contig": (lambda x: A._to_copy.default(x, dtype=torch.float64,
                                                       memory_format=torch.contiguous_format), True),
    "zeros_like": (lambda x: torch.zeros_like(x), True),
    "ones_like": (lambda x: torch.ones_like(x), True),
    "full_like": (lambda x: torch.full_like(x, 3.0), True),
    "empty_like": (lambda x: torch.empty_like(x), False),
    "zeros_like_int": (lambda x: torch.zeros_like(x, dtype=torch.int64), True),
    "zeros_like_contig": (lambda x: torch.zeros_like(x, memory_format=torch.contiguous_format), True),
}
out["convert"] = {}
for op_name, (fn, values) in CONVERT.items():
    for key, make in LAYOUTS.items():
        run("convert", op_name + "|" + key,
            lambda fn=fn, make=make: (lambda x: (fn(x), [x]))(make("cpu")), values)

# Controls: upstream answers these row-major whatever the input layout.
ROWMAJOR = {
    "masked_fill": lambda x: x.masked_fill(x > 1.5, 0.0),
    "pow_scalar_base": lambda x: 2.0 ** x,
    "softmax": lambda x: x.softmax(-1),
    "sum_last": lambda x: x.sum(-1),
    "mean_first": lambda x: x.mean(0),
    "cat": lambda x: torch.cat([x, x], 0),
    "index_select": lambda x: x.index_select(0, torch.tensor([1, 0])),
}
out["rowmajor"] = {}
for op_name, fn in ROWMAJOR.items():
    for key in ("perm", "t01", "t12", "perm_gap"):
        run("rowmajor", op_name + "|" + key,
            lambda fn=fn, key=key: (lambda x: (fn(x), [x]))(LAYOUTS[key]("cpu")))
run("rowmajor", "mm|t2d", lambda: (lambda a: (torch.mm(a, base(12).view(3, 4)), [a]))(LAYOUTS["t2d"]("cpu")))

# In-place: the receiver keeps its own layout.
def inplace(op):
    def build():
        y = LAYOUTS["perm"]("cpu").clone()
        r = op(y)
        assert r is y
        return y, []
    return build


run("inplace", "mul_", inplace(lambda y: y.mul_(2.0)))
run("inplace", "add_", inplace(lambda y: y.add_(y)))
run("inplace", "exp_", inplace(lambda y: y.exp_()))

# (iv) attention. Deterministic values, so both sides attend to the same
# numbers without a shared generator.
def det(shape, phase):
    n = 1
    for s in shape:
        n *= s
    return torch.sin(torch.arange(n, dtype=torch.float32) * 0.37 + phase).view(shape)


QUERIES = {
    "contig": lambda p: det((2, 4, 5, 8), p),
    "bLhd_T": lambda p: det((2, 5, 4, 8), p).transpose(1, 2),
    "Lbhd_perm": lambda p: det((5, 2, 4, 8), p).permute(1, 2, 0, 3),
    "bhdL_T": lambda p: det((2, 4, 8, 5), p).transpose(2, 3),
    "bLhd_T_sliced": lambda p: det((2, 7, 4, 8), p).transpose(1, 2)[:, :, :5],
    "L1": lambda p: det((2, 4, 1, 8), p),
    "h1_T": lambda p: det((2, 5, 1, 8), p).transpose(1, 2),
}
out["attention"] = {}
for key, make in QUERIES.items():
    for causal in (False, True):
        def flash(make=make, causal=causal):
            q, k, v = make(0.0), make(1.0), make(2.0)
            return A._scaled_dot_product_flash_attention_for_cpu.default(q, k, v, 0.0, causal), [q, k, v]
        name = key + ("|causal" if causal else "")
        try:
            (o, lse), inputs = flash()
            out["attention"][name + "|out"] = describe(o, inputs)
            out["attention"][name + "|logsumexp"] = describe(lse, inputs)
        except Exception as exc:
            out["attention"][name + "|out"] = {"raised": type(exc).__name__, "message": str(exc)[:300]}
            out["attention"][name + "|logsumexp"] = out["attention"][name + "|out"]
    run("attention", key + "|F.sdpa",
        lambda make=make: (lambda q, k, v: (F.scaled_dot_product_attention(q, k, v), [q, k, v]))(
            make(0.0), make(1.0), make(2.0)))
    run("attention", key + "|math",
        lambda make=make: (lambda q, k, v: (A._scaled_dot_product_attention_math.default(q, k, v)[0], [q, k, v]))(
            make(0.0), make(1.0), make(2.0)))

# The same chains on meta inputs, for the ops with a meta arm the meta side
# lays out by operands (`MetaStrideRule::Elementwise` and `PreserveFormat`).
META_UNARY = ["exp", "neg", "sqrt", "tanh", "relu", "silu", "gelu", "clamp", "mul_scalar",
              "add_scalar_int", "rsub_scalar", "div_scalar", "pow_tensor_scalar", "gt_scalar",
              "eq_scalar", "where_scalar_other"]
META_BINARY = ["add", "sub", "mul", "div", "eq", "lt", "where_self"]
META_CONVERT = ["double", "zeros_like", "empty_like", "to_copy_f64"]
META_LAYOUTS = ["contig", "perm", "t01", "t12", "gap", "perm_gap", "expanded", "t2d"]
META_PAIRS = ["c+p", "p+c", "t01+t12", "p+row", "row+p", "expanded+p", "gap+p", "p+0d", "t2d+lead"]
out["meta"] = {}
for op_name in META_UNARY:
    for key in META_LAYOUTS:
        run("meta", "unary|" + op_name + "|" + key,
            lambda fn=UNARY[op_name], key=key: (fn(LAYOUTS[key]("meta")), []))
for op_name in META_CONVERT:
    for key in META_LAYOUTS:
        run("meta", "convert|" + op_name + "|" + key,
            lambda fn=CONVERT[op_name][0], key=key: (fn(LAYOUTS[key]("meta")), []))
for op_name in META_BINARY:
    for key in META_PAIRS:
        ma, mb = PAIRS[key]
        run("meta", "binary|" + op_name + "|" + key,
            lambda fn=BINARY[op_name], ma=ma, mb=mb: (fn(ma("meta"), mb("meta")), []))

print(json.dumps(out))
'''


def _run(shim, _cache={}):
    if shim in _cache:
        return _cache[shim]
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        capture_output=True, text=True, env=env, cwd=_REPO_ROOT, timeout=900,
    )
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
    return _run(True), _run(False)


_LAYOUT_KEYS = ("shape", "stride", "offset", "contiguous", "dtype", "shares", "view_flat", "raised")


def _close(a, b, rel=1e-5, abs_=1e-6):
    if a is None or b is None or len(a) != len(b):
        return a == b
    for x, y in zip(a, b):
        if x != x and y != y:
            continue
        if abs(x - y) > abs_ + rel * abs(y):
            return False
    return True


def _diff(section, names, shim, up, rel=1e-5, abs_=1e-6):
    bad = []
    for name in names:
        u, s = up[section][name], shim[section].get(name) or {}
        layout_u = {k: u.get(k) for k in _LAYOUT_KEYS}
        layout_s = {k: s.get(k) for k in _LAYOUT_KEYS}
        if layout_u != layout_s or not _close(s.get("values"), u.get("values"), rel, abs_):
            if "values" in u and "values" in s and layout_u == layout_s:
                layout_u, layout_s = dict(layout_u, values=u["values"]), dict(layout_s, values=s["values"])
            bad.append(f"  {name}\n    upstream {layout_u}\n    shim     {layout_s}")
    return bad


def _checked(section, shim, up, minimum):
    names = sorted(up[section])
    assert len(names) >= minimum, f"the probe lost rows of {section}: {len(names)}"
    refused = [n for n in names if "raised" in up[section][n]]
    assert not refused, f"upstream refused rows of {section}: {refused}"
    return names


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------


def test_the_oracle_is_what_the_measurement_recorded():
    """The oracle first, against docs/kernels/VIEWS.md section 8's table: a
    permuted operand's permutation survives `exp` and `* 2.0`, the first
    operand decides (`c+p` contiguous, `p+c` permuted), a gap repacks
    row-major, `double()` of a transposed `(3, 2)` keeps `(1, 2)`,
    `masked_fill` is row-major, and the flash output of a transposed query is
    `(160, 8, 32, 1)` with a `(20, 1, 4)` logsumexp."""
    pair = _both()
    if pair is None:
        return
    _, up = pair
    assert up["unary"]["exp|perm"]["stride"] == [1, 8, 2], up["unary"]["exp|perm"]
    assert up["unary"]["mul_scalar|perm"]["contiguous"] is False
    assert up["unary"]["mul_scalar|gap"]["stride"] == [12, 4, 1]
    assert up["binary"]["add|c+p"]["stride"] == [12, 4, 1]
    assert up["binary"]["add|p+c"]["stride"] == [1, 8, 2]
    assert up["binary"]["bitwise_and_bool|p+row"]["stride"] == [1, 8, 2]
    assert up["convert"]["double|t2d"]["stride"] == [1, 2]
    assert up["convert"]["to_copy_f64_contig|t2d"]["stride"] == [3, 1]
    assert up["rowmajor"]["masked_fill|perm"]["stride"] == [12, 4, 1]
    assert up["attention"]["bLhd_T|out"]["stride"] == [160, 8, 32, 1]
    assert up["attention"]["contig|out"]["stride"] == [160, 40, 8, 1]
    assert up["attention"]["contig|logsumexp"]["stride"] == [20, 1, 4]
    assert up["attention"]["bLhd_T|logsumexp"]["contiguous"] is False


def test_unary_and_scalar_results_are_laid_out_as_upstreams():
    """Family (i), the `aten.mul.Scalar` rows of CI run 43819391187: every
    unary op and tensor-scalar op in `UNARY` on every layout in `LAYOUTS`
    (contiguous, permuted, transposed two ways, a gap, a permuted gap, two
    expanded, an offset, a 2-D transpose, one element, a strided 1-D, an
    extent-1 column)."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = _checked("unary", shim, up, 300)
    bad = _diff("unary", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} unary rows differ from upstream:\n" + "\n".join(bad)
    moved = sum(1 for n in names if not up["unary"][n]["contiguous"])
    print(f"ELEMSTRIDE unary: {len(names)} of {len(names)} rows equal upstream's ({moved} not contiguous)")


def test_tensor_tensor_results_are_laid_out_as_upstreams():
    """Family (ii), the `aten.bitwise_and.Tensor` rows: two tensor operands of
    the same, different, broadcast, expanded, gapped, 0-D and one-element
    layouts, including the order of the two (the first operand that orders two
    axes decides), int + float promotion, comparisons, logical and bitwise ops
    on bool, and `where.self`."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = _checked("binary", shim, up, 200)
    bad = _diff("binary", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} binary rows differ from upstream:\n" + "\n".join(bad)
    print(f"ELEMSTRIDE binary: {len(names)} of {len(names)} rows equal upstream's")


def test_conversions_and_like_factories_are_laid_out_as_upstreams():
    """Family (iii): a dtype-changing `_to_copy` (`double`, `float`, `bool`,
    `to(int32)`, `to(bfloat16)`, the aten op with `preserve_format` and with
    `contiguous_format`) and `zeros_like`, `ones_like`, `full_like`,
    `empty_like` (layout only), `zeros_like(dtype=int64)` and
    `zeros_like(memory_format=contiguous_format)`, on every layout."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = _checked("convert", shim, up, 150)
    bad = _diff("convert", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} conversion rows differ from upstream:\n" + "\n".join(bad)


def test_flash_attention_outputs_are_laid_out_as_upstreams():
    """Family (iv), the 99 `aten._scaled_dot_product_flash_attention_for_cpu`
    rows: the output and the logsumexp of the CPU flash kernel for a
    contiguous query, the usual `(b, L, h, d).transpose(1, 2)` query, a
    permuted one, a `(b, h, d, L)` transposed one, a sliced (not dense) one and
    extent-1 heads and lengths, causal and not; plus
    `F.scaled_dot_product_attention` (the flash output through the CIA body)
    and the math op (row-major upstream). Values within 1e-4: the flash
    kernel's default arithmetic agrees with upstream's only to a tolerance
    (docs/kernels/SDPA.md section 3)."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = _checked("attention", shim, up, 40)
    bad = _diff("attention", names, shim, up, rel=1e-4, abs_=1e-4)
    assert not bad, f"{len(bad)} of {len(names)} attention rows differ from upstream:\n" + "\n".join(bad)


def test_row_major_results_stay_row_major():
    """The controls: `masked_fill`, `2.0 ** x` (`pow.Scalar`), `softmax`,
    `sum`, `mean`, `cat`, `index_select` and `mm` are row-major upstream for
    every input layout, and stay row-major here. Red when the relay is applied
    to an op upstream does not lay out by its operands."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = _checked("rowmajor", shim, up, 25)
    for name in names:
        assert up["rowmajor"][name]["contiguous"] is True, (name, up["rowmajor"][name])
    bad = _diff("rowmajor", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} row-major rows differ:\n" + "\n".join(bad)


def test_in_place_receivers_keep_their_layout():
    """`mul_`, `add_` and `exp_` on a permuted receiver return the receiver,
    whose stride stays `(1, 8, 2)`: the relay touches only fresh results."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = _checked("inplace", shim, up, 3)
    for name in names:
        assert up["inplace"][name]["stride"] == [1, 8, 2], (name, up["inplace"][name])
    bad = _diff("inplace", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} in-place rows differ:\n" + "\n".join(bad)


def test_the_dense_layout_is_the_meta_layout():
    """The meta side already laid these ops out by their operands
    (`relay_elementwise`); the dense side now does too, so for every meta row
    the shim's meta layout, the shim's dense layout of the same chain and
    upstream's dense layout are one answer (shape, stride, `is_contiguous`,
    dtype)."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(up["meta"])
    assert len(names) >= 200, len(names)
    keys = ("shape", "stride", "contiguous", "dtype", "raised")
    bad = []
    for name in names:
        family, op_name, key = name.split("|", 2)
        section = {"unary": "unary", "binary": "binary", "convert": "convert"}[family]
        dense_name = op_name + "|" + key
        want = {k: up[section][dense_name].get(k) for k in keys}
        dense = {k: shim[section].get(dense_name, {}).get(k) for k in keys}
        meta = {k: shim["meta"].get(name, {}).get(k) for k in keys}
        if not (dense == meta == want):
            bad.append(f"  {name}\n    upstream dense {want}\n    shim meta      {meta}\n    shim dense     {dense}")
    assert not bad, f"{len(bad)} of {len(names)} rows disagree:\n" + "\n".join(bad)


def test_a_view_of_the_result_succeeds_exactly_where_upstreams_does():
    """The consequence #383 is about, counted on its own: across every row of
    every section, `.view(-1)` of the result succeeds here exactly where it
    succeeds upstream. Before #383 every row whose upstream answer is not
    contiguous viewed here and refused upstream."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    bad, refusing = [], 0
    for section in ("unary", "binary", "convert", "rowmajor", "attention"):
        for name, u in up[section].items():
            if "view_flat" not in u:
                continue
            refusing += u["view_flat"] != "ok"
            if shim[section].get(name, {}).get("view_flat") != u["view_flat"]:
                bad.append(f"  {section} {name}: upstream {u['view_flat']}, shim "
                           f"{shim[section].get(name, {}).get('view_flat')}")
    assert refusing >= 100, f"the probe lost the rows whose view refuses upstream: {refusing}"
    assert not bad, f"{len(bad)} rows differ in whether view(-1) succeeds:\n" + "\n".join(bad)


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = _skip.run_tests(items, suite="test_elemstride")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
