"""The dense view family is a view: `view`, `_unsafe_view`, `reshape`, `view_as`,
`reshape_as` and `unflatten` on a non-contiguous input (issue #380).

Before this suite the six dense kernels ran `.contiguous().reshape(target)`
whatever the input's layout. On a non-contiguous input upstream can view, that
copies, silently: the result reported contiguous strides, a write through it
never reached the base (nor a write to the base the result), `_is_view()` was
False and `_base` None, and a `view` upstream refuses succeeded here. CI
measured 263 of 777 (Linux) and 294 of 777 (macOS) decomposition-sequence rows
of `tests/bindings/test_cia_frame.py` differing from upstream in the contiguity
flag alone.

Upstream's rule, which the meta arms already followed through
`layout.rs::view_stride` (the port of `computeStride`): a view where
`computeStride` finds strides for the new shape, over the input's storage at the
input's offset; otherwise `view`, `_unsafe_view` and `view_as` refuse and
`reshape`, `reshape_as` and `unflatten` copy.

**Every assertion is upstream's answer, computed in a separate subprocess from
the same probe text**, never a table written beside the implementation. The one
exception is `test_the_two_380_examples_are_what_upstream_measured`, which
asserts the issue's two measured layouts on the upstream side first, so the
oracle itself is checked before the shim is compared with it.

What each test turns red under (nullifications, none run locally; CI runs them):

* the kernels back to `.contiguous().reshape(...)`: every test below except the
  refusal and copy tests (stride, `is_contiguous`, sharing, `_is_view`, writes);
* `view_with_layout` building a copy instead of sharing the `Arc`:
  `..._share_storage_and_report_upstreams_layout` (shares) and
  `test_writes_reach_the_base_and_the_base_reaches_the_view` (values);
* the helper copying instead of refusing for `view`/`_unsafe_view`/`view_as`:
  `test_a_view_upstream_refuses_refuses_here_in_upstreams_words`;
* the helper refusing instead of copying for `reshape`/`reshape_as`/`unflatten`:
  `test_reshape_of_an_unviewable_input_is_an_independent_copy`;
* the helper using candle's contiguous strides instead of `view_stride`'s:
  `test_the_dense_layout_is_the_meta_layout` and the zero-element rows;
* `note_view` not marking `_unsafe_view`'s result as a non-autograd alias:
  `test_unsafe_view_aliases_but_is_not_an_autograd_view` (offset row);
* `expand_default` keeping candle's zero stride on a new leading axis of
  extent 1: the `expand_new_leading_one` row.

What this file cannot see: device kernels (Metal, CUDA, Vulkan) reading a
strided view result. The probe is CPU only; no CI device run covers it.

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
import torch

A = torch.ops.aten
out = {"is_shim": hasattr(torch._C, "_aten_implemented"),
       "cases": {}, "refusals": {}, "writes": {}, "meta": {}}


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


def base_kind(t, root):
    try:
        b = t._base
    except Exception as exc:
        return "raised: " + type(exc).__name__
    if b is None:
        return "none"
    return "root" if b is root else "other"


def describe(root, t, values=True):
    d = {
        "shape": list(t.shape),
        "stride": list(t.stride()),
        "offset": t.storage_offset(),
        "contiguous": t.is_contiguous(),
        "is_view": t._is_view(),
        "base": base_kind(t, root),
    }
    if root is not None and root.device.type != "meta":
        d["shares"] = t.untyped_storage()._cdata == root.untyped_storage()._cdata
    if values:
        d["values"] = flat(t)
    return d


# Each input is `factory -> (root, input)`; `factory(n)` is a 1-D tensor of n
# elements: `arange` for dense, `empty(device="meta")` for the meta rows.
INPUTS = {
    "contig": lambda f: (lambda r: (r, r.view(2, 3, 4)))(f(24)),
    "root": lambda f: (lambda r: (r, r))(f(24)),
    "transposed": lambda f: (lambda r: (r, r.view(2, 3, 4).transpose(0, 1)))(f(24)),
    "permuted": lambda f: (lambda r: (r, r.view(5, 2, 3, 2).permute(1, 2, 3, 0)))(f(60)),
    "t2d": lambda f: (lambda r: (r, r.view(3, 4).t()))(f(12)),
    "stepped": lambda f: (lambda r: (r, r.view(3, 4)[:, ::2]))(f(12)),
    "offset": lambda f: (lambda r: (r, r.view(3, 4)[:, 1:3]))(f(12)),
    "expanded": lambda f: (lambda r: (r, r.view(1, 4).expand(3, 4)))(f(4)),
    "expanded_scalar": lambda f: (lambda r: (r, r.expand(6)))(f(1)),
    "ex2": lambda f: (lambda r: (r, r.view(2, 3, 8).transpose(0, 1).expand(3, 2, 8)))(f(48)),
    "zero": lambda f: (lambda r: (r, r.view(3, 4)[:, 0:0]))(f(12)),
    "zero_t": lambda f: (lambda r: (r, r.view(3, 4)[:, 0:0].t()))(f(12)),
    "one": lambda f: (lambda r: (r, r.view(2, 2)[1:, 1:]))(f(4)),
    "strided1d": lambda f: (lambda r: (r, r.view(2, 2)[:, 1]))(f(4)),
    "scalar0d": lambda f: (lambda r: (r, r[2]))(f(4)),
}


def dense(n):
    return torch.arange(float(n))


def meta(n):
    return torch.empty(n, device="meta")


def apply(op, x, arg):
    if op == "view":
        return x.view(arg)
    if op == "unsafe":
        return A._unsafe_view.default(x, arg)
    if op == "reshape":
        return x.reshape(arg)
    if op == "view_as":
        return x.view_as(torch.empty(arg, device=x.device))
    if op == "reshape_as":
        return x.reshape_as(torch.empty(arg, device=x.device))
    if op == "unflatten":
        return x.unflatten(arg[0], arg[1])
    if op == "expand":
        return x.expand(arg)
    raise AssertionError(op)


# (name, input, op, argument). Upstream answers every one of these with a
# tensor; which of them are views is upstream's answer, not this table's.
CASES = [
    # The issue's two examples.
    ("ex1_view", "permuted", "view", [12, 5]),
    ("ex2_view", "ex2", "view", [3, 2, 8]),
    # The rest of the permuted input.
    ("ex1_view_infer", "permuted", "view", [-1, 5]),
    ("ex1_unsafe", "permuted", "unsafe", [12, 5]),
    ("ex1_reshape", "permuted", "reshape", [12, 5]),
    ("ex1_view_as", "permuted", "view_as", [12, 5]),
    ("ex1_reshape_as", "permuted", "reshape_as", [12, 5]),
    ("ex1_reshape_copy", "permuted", "reshape", [60]),
    ("ex1_unflatten_last", "permuted", "unflatten", [3, (5, 1)]),
    ("ex1_unflatten_infer", "permuted", "unflatten", [0, (1, -1)]),
    ("ex2_reshape", "ex2", "reshape", [3, 2, 8]),
    ("ex2_unsafe", "ex2", "unsafe", [3, 2, 8]),
    # Transposed.
    ("T_view", "transposed", "view", [3, 2, 2, 2]),
    ("T_view_infer", "transposed", "view", [3, 2, -1, 2]),
    ("T_unsafe", "transposed", "unsafe", [3, 2, 2, 2]),
    ("T_reshape", "transposed", "reshape", [3, 2, 2, 2]),
    ("T_unflatten_2", "transposed", "unflatten", [2, (2, 2)]),
    ("T_unflatten_1", "transposed", "unflatten", [1, (2, 1)]),
    ("T_unflatten_0", "transposed", "unflatten", [0, (3, 1)]),
    ("T_reshape_copy", "transposed", "reshape", [6, 4]),
    ("T_reshape_copy_infer", "transposed", "reshape", [-1]),
    ("T_reshape_as_copy", "transposed", "reshape_as", [6, 4]),
    # t() of a 2-D tensor.
    ("t_view_trailing_one", "t2d", "view", [4, 3, 1]),
    ("t_view_middle_one", "t2d", "view", [4, 1, 3]),
    ("t_view_split", "t2d", "view", [2, 2, 3]),
    ("t_view_as", "t2d", "view_as", [4, 3]),
    ("t_reshape_view", "t2d", "reshape", [2, 2, 3]),
    ("t_reshape_copy", "t2d", "reshape", [12]),
    # A stepped slice and an offset slice.
    ("stepped_view", "stepped", "view", [3, 2, 1]),
    ("stepped_view_flat", "stepped", "view", [6]),
    ("stepped_unsafe", "stepped", "unsafe", [3, 1, 2]),
    ("offset_view", "offset", "view", [3, 1, 2]),
    ("offset_unsafe", "offset", "unsafe", [3, 1, 2]),
    ("offset_reshape_copy", "offset", "reshape", [6]),
    # Expanded (stride 0).
    ("expanded_view", "expanded", "view", [3, 2, 2]),
    ("expanded_unsafe", "expanded", "unsafe", [3, 2, 2]),
    ("expanded_reshape_copy", "expanded", "reshape", [12]),
    ("expanded_scalar_view", "expanded_scalar", "view", [2, 3]),
    ("expanded_scalar_reshape", "expanded_scalar", "reshape", [3, 2]),
    # Zero elements.
    ("zero_view_rows", "zero", "view", [0, 3]),
    ("zero_view_flat", "zero", "view", [0]),
    ("zero_view_infer", "zero", "view", [-1]),
    ("zero_view_same", "zero", "view", [3, 0]),
    ("zero_view_rank3", "zero", "view", [0, 5, 2]),
    ("zero_reshape", "zero", "reshape", [0, 3]),
    ("zero_t_view_same", "zero_t", "view", [0, 3]),
    ("zero_t_view_infer", "zero_t", "view", [-1]),
    ("zero_t_view_other", "zero_t", "view", [2, 0]),
    # One element, a strided 1-D, a 0-D.
    ("one_view_flat", "one", "view", [1]),
    ("one_view_infer", "one", "view", [-1]),
    ("one_view_rank3", "one", "view", [1, 1, 1]),
    ("one_view_0d", "one", "view", []),
    ("one_unsafe", "one", "unsafe", [1]),
    ("strided1d_view_col", "strided1d", "view", [2, 1]),
    ("strided1d_view_row", "strided1d", "view", [1, 2]),
    ("strided1d_view_infer", "strided1d", "view", [-1]),
    ("strided1d_reshape", "strided1d", "reshape", [2, 1]),
    ("scalar0d_view", "scalar0d", "view", [1, 1]),
    ("scalar0d_view_infer", "scalar0d", "view", [-1]),
    ("scalar0d_reshape", "scalar0d", "reshape", []),
    # Already contiguous: the path that did not change.
    ("contig_view", "contig", "view", [6, 4]),
    ("contig_unsafe", "contig", "unsafe", [6, 4]),
    ("contig_reshape", "contig", "reshape", [4, 6]),
    ("contig_view_as", "contig", "view_as", [6, 4]),
    ("contig_reshape_as", "contig", "reshape_as", [6, 4]),
    ("contig_unflatten", "contig", "unflatten", [2, (2, 2)]),
    ("root_view", "root", "view", [6, 4]),
    ("root_unsafe", "root", "unsafe", [6, 4]),
    # expand's stride on a new leading axis of extent 1 (inferExpandGeometry).
    ("expand_new_leading_one", "t2d", "expand", [1, 4, 3]),
    ("expand_new_leading_two", "t2d", "expand", [2, 4, 3]),
]

for name, key, op, arg in CASES:
    try:
        root, x = INPUTS[key](dense)
        out["cases"][name] = describe(root, apply(op, x, arg))
    except Exception as exc:
        out["cases"][name] = {"raised": type(exc).__name__, "message": str(exc)}


# Upstream refuses each of these (`computeStride` finds no strides).
REFUSALS = [
    ("T_view", "transposed", "view", [6, 4]),
    ("T_view_rows", "transposed", "view", [3, 8]),
    ("T_unsafe", "transposed", "unsafe", [6, 4]),
    ("T_view_as", "transposed", "view_as", [6, 4]),
    ("t_view_flat", "t2d", "view", [12]),
    ("offset_view_flat", "offset", "view", [6]),
    ("expanded_view_flat", "expanded", "view", [12]),
    ("ex1_view_flat", "permuted", "view", [60]),
]

for name, key, op, arg in REFUSALS:
    try:
        root, x = INPUTS[key](dense)
        r = apply(op, x, arg)
        out["refusals"][name] = {"ok": describe(root, r)}
    except Exception as exc:
        out["refusals"][name] = {"raised": type(exc).__name__, "message": str(exc)}


# Writes, both directions. Each returns the side that did NOT take the write.
def w_view_to_base():
    r, x = INPUTS["transposed"](dense)
    v = x.view(3, 2, 2, 2)
    v.add_(100.0)
    v[1, 0, 1, 1] = -1.0
    return flat(r)


def w_base_to_view():
    r, x = INPUTS["transposed"](dense)
    v = x.view(3, 2, 2, 2)
    r[5] = -7.0
    r.mul_(2.0)
    return flat(v)


def w_ex1_row():
    r, x = INPUTS["permuted"](dense)
    v = x.view(12, 5)
    v[0].fill_(-1.0)
    return flat(r)


def w_ex2_inplace():
    r, x = INPUTS["ex2"](dense)
    v = x.view(3, 2, 8)
    v.add_(1.0)
    return flat(r)


def w_reshape_view():
    r, x = INPUTS["t2d"](dense)
    v = x.reshape(2, 2, 3)
    v.zero_()
    return flat(r)


def w_reshape_copy_to_base():
    r, x = INPUTS["transposed"](dense)
    v = x.reshape(6, 4)
    v.fill_(-5.0)
    return flat(r)


def w_reshape_copy_from_base():
    r, x = INPUTS["transposed"](dense)
    v = x.reshape(6, 4)
    r.fill_(1.0)
    return flat(v)


def w_stepped():
    r, x = INPUTS["stepped"](dense)
    x.view(3, 2, 1).fill_(5.0)
    return flat(r)


def w_offset():
    r, x = INPUTS["offset"](dense)
    x.view(3, 1, 2).mul_(-1.0)
    return flat(r)


def w_unflatten():
    r, x = INPUTS["transposed"](dense)
    x.unflatten(2, (2, 2)).sub_(1.0)
    return flat(r)


def w_view_as_from_base():
    r, x = INPUTS["t2d"](dense)
    v = x.view_as(torch.empty(4, 3))
    r.add_(1.0)
    return flat(v)


def w_unsafe():
    r, x = INPUTS["offset"](dense)
    A._unsafe_view.default(x, [3, 1, 2]).fill_(0.0)
    return flat(r)


def w_expanded_add():
    r, x = INPUTS["expanded"](dense)
    x.view(3, 2, 2).add_(1.0)
    return flat(r)


def w_expanded_fill():
    r, x = INPUTS["expanded"](dense)
    x.view(3, 2, 2).fill_(9.0)
    return flat(r)


for name, fn in sorted(globals().copy().items()):
    if name.startswith("w_") and callable(fn):
        try:
            out["writes"][name] = {"ok": fn()}
        except Exception as exc:
            out["writes"][name] = {"raised": type(exc).__name__, "message": str(exc)}


# The meta rows: the same chains, on meta tensors. Only `view` and `reshape`
# have meta kernels.
META = [c for c in CASES if c[2] in ("view", "reshape")
        and c[1] in ("transposed", "permuted", "t2d", "stepped", "offset",
                     "expanded", "ex2", "zero", "zero_t", "contig")]
for name, key, op, arg in META:
    try:
        root, x = INPUTS[key](meta)
        out["meta"][name] = describe(root, apply(op, x, arg), values=False)
    except Exception as exc:
        out["meta"][name] = {"raised": type(exc).__name__, "message": str(exc)}

# The copies whose layout a later `view` depends on: a fresh tensor upstream
# lays out row-major (or, for `preserve_format`, in the input's permutation)
# must not come back here with the input's zero stride, gap or transposition,
# or the `view` after it refuses where upstream's succeeds (CI run 43808445902).
COPY_OPS = {
    "clone": lambda x: x.clone(),
    "clone_contig": lambda x: x.clone(memory_format=torch.contiguous_format),
    "to_copy": lambda x: A._to_copy.default(x),
    "to_copy_contig": lambda x: A._to_copy.default(x, memory_format=torch.contiguous_format),
    "to_same_dtype_copy": lambda x: x.to(torch.float32, copy=True),
    "cat_one": lambda x: torch.cat([x]),
    "cat_two_last": lambda x: torch.cat([x, x], -1),
    "repeat_ones": lambda x: x.repeat([1] * x.dim()),
    "repeat_two": lambda x: x.repeat([2] + [1] * (x.dim() - 1)),
    "index_add_empty": lambda x: x.index_add(0, torch.zeros(0, dtype=torch.long),
                                             torch.zeros([0] + list(x.shape[1:]))),
    "index_copy_empty": lambda x: x.index_copy(0, torch.zeros(0, dtype=torch.long),
                                               torch.zeros([0] + list(x.shape[1:]))),
    "index_copy_one": lambda x: x.index_copy(0, torch.tensor([0]),
                                             torch.full([1] + list(x.shape[1:]), -1.0)),
}
COPY_INPUTS = ["contig", "transposed", "permuted", "t2d", "stepped", "offset",
               "expanded", "expanded_scalar", "ex2", "zero_t", "strided1d", "one"]
out["copies"] = {}
for op_name, fn in COPY_OPS.items():
    for key in COPY_INPUTS:
        if op_name.startswith("index") and key in ("zero_t",):
            continue
        name = op_name + "|" + key
        try:
            root, x = INPUTS[key](dense)
            r = fn(x)
            d = describe(root, r)
            try:
                r.view(-1)
                d["view_flat"] = "ok"
            except Exception as exc:
                d["view_flat"] = type(exc).__name__ + ": " + str(exc)
            out["copies"][name] = d
        except Exception as exc:
            out["copies"][name] = {"raised": type(exc).__name__, "message": str(exc)}


# The CI scenarios themselves: composites whose `clone` feeds a `view` or an
# `_unsafe_view` (bootstrap.py's `repeat_interleave`, and the `reshape` and
# `matmul` bodies reached through `_op_dk(CompositeImplicitAutograd)`).
CIA = torch._C.DispatchKey.CompositeImplicitAutograd
M = torch.arange(6.0).view(2, 3)
SCENARIOS = {
    "repeat_interleave_dim_last": lambda: torch.repeat_interleave(M, 2, dim=-1),
    "repeat_interleave_flat": lambda: torch.repeat_interleave(M, 2),
    "repeat_interleave_t": lambda: torch.repeat_interleave(M.t(), 2, dim=0),
    "reshape_body_transposed": lambda: A.reshape.default._op_dk(
        CIA, torch.arange(6.0).reshape(3, 2).t(), [6]),
    "reshape_body_expand": lambda: A.reshape.default._op_dk(
        CIA, torch.arange(6.0).view(2, 1, 3).expand(2, 4, 3), [8, 3]),
    "matmul_body_broadcast": lambda: A.matmul.default._op_dk(
        CIA, torch.arange(24.0).view(2, 1, 3, 4), torch.arange(40.0).view(5, 4, 2)),
    "matmul_body_fold": lambda: A.matmul.default._op_dk(
        CIA, torch.arange(4.0).requires_grad_(), torch.arange(24.0).view(2, 4, 3)),
}
out["scenarios"] = {}
for name, fn in SCENARIOS.items():
    try:
        out["scenarios"][name] = describe(None, fn().detach())
    except Exception as exc:
        out["scenarios"][name] = {"raised": type(exc).__name__, "message": str(exc)}

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
    # (AGENTS.md §15.2), so the side is asserted, not assumed.
    assert data["is_shim"] is shim, f"the probe meant for shim={shim} ran on the other side"
    _cache[shim] = data
    return data


def _both():
    if not _available():
        _skip.skip("no vendored shim installed (scripts/vendor/install_shim.sh)")
        return None
    return _run(True), _run(False)


def _diff(section, names, shim, up):
    bad = []
    for name in names:
        if shim[section].get(name) != up[section][name]:
            bad.append(f"  {name}\n    upstream {up[section][name]}\n    shim     {shim[section].get(name)}")
    return bad


# ---------------------------------------------------------------------------
# The tests
# ---------------------------------------------------------------------------


def test_the_two_380_examples_are_what_upstream_measured():
    """The oracle first: upstream's two #380 examples are shared-storage,
    non-contiguous views with strides (1, 12) and (8, 24, 1). Then the shim's
    answer is the same, field for field. Red when the kernel copies: the shim
    reports (5, 1) and (16, 8, 1), contiguous, not sharing."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    one, two = up["cases"]["ex1_view"], up["cases"]["ex2_view"]
    assert (one["stride"], one["contiguous"], one["shares"]) == ([1, 12], False, True), one
    assert (two["stride"], two["contiguous"], two["shares"]) == ([8, 24, 1], False, True), two
    assert one["is_view"] and one["base"] == "root", one
    bad = _diff("cases", ["ex1_view", "ex2_view"], shim, up)
    assert not bad, "the #380 examples differ from upstream:\n" + "\n".join(bad)


def test_every_view_family_result_is_upstreams_layout_and_aliasing():
    """Every row of `CASES`: shape, stride, storage offset, `is_contiguous`,
    whether it shares the root's storage, `_is_view()`, what `_base` is, and
    the values. Upstream answers every row with a tensor."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(up["cases"])
    refused = [n for n in names if "raised" in up["cases"][n]]
    assert not refused, f"upstream refused rows this table expects it to answer: {refused}"
    bad = _diff("cases", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} rows differ from upstream:\n" + "\n".join(bad)
    print(f"VIEWSTRIDE: {len(names)} of {len(names)} rows equal upstream's")


def test_viewable_inputs_share_storage_and_report_upstreams_layout():
    """The half of the table the defect was about, asserted on its own so that
    a regression names it: every row upstream answers with a non-contiguous
    shared view is one here too."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(n for n, d in up["cases"].items()
                   if d.get("shares") and not d.get("contiguous"))
    assert len(names) >= 25, f"the probe lost its non-contiguous views: {names}"
    bad = [n for n in names
           if not shim["cases"][n].get("shares")
           or shim["cases"][n].get("stride") != up["cases"][n]["stride"]
           or shim["cases"][n].get("offset") != up["cases"][n]["offset"]]
    assert not bad, f"{len(bad)} of {len(names)} non-contiguous views are not views here: {bad}"


def test_a_view_upstream_refuses_refuses_here_in_upstreams_words():
    """`view`, `_unsafe_view` and `view_as` of a layout `computeStride` cannot
    express raise upstream's RuntimeError, word for word. Red when the helper
    copies instead (the old behaviour: the view "succeeded")."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(up["refusals"])
    for name in names:
        assert up["refusals"][name].get("raised") == "RuntimeError", (name, up["refusals"][name])
        assert "view size is not compatible" in up["refusals"][name]["message"], up["refusals"][name]
    bad = _diff("refusals", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} refusals differ:\n" + "\n".join(bad)


def test_reshape_of_an_unviewable_input_is_an_independent_copy():
    """`reshape`, `reshape_as` and `unflatten`'s copy arm: contiguous, a storage
    of its own, `_base` None, upstream's values. Red when the helper refuses
    for them too, or when the copy aliases its input."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = ["T_reshape_copy", "T_reshape_copy_infer", "T_reshape_as_copy",
             "ex1_reshape_copy", "t_reshape_copy", "offset_reshape_copy",
             "expanded_reshape_copy"]
    for name in names:
        u = up["cases"][name]
        assert (u["shares"], u["contiguous"], u["is_view"], u["base"]) == (False, True, False, "none"), (name, u)
    bad = _diff("cases", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} copies differ:\n" + "\n".join(bad)


def test_writes_reach_the_base_and_the_base_reaches_the_view():
    """Write through the view, read the base; write to the base, read the view;
    in-place ops (`add_`, `mul_`, `sub_`, `zero_`, `fill_`) and `__setitem__`
    on the view; the copy arm in both directions (the write must NOT arrive);
    an expanded view (upstream refuses `add_` on it and lets `fill_` through).
    Red when the view is a copy: the base keeps its old values."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(up["writes"])
    assert len(names) == 14, names
    assert up["writes"]["w_expanded_add"].get("raised") == "RuntimeError", up["writes"]["w_expanded_add"]
    assert up["writes"]["w_view_to_base"]["ok"][7] == -1.0, up["writes"]["w_view_to_base"]
    bad = _diff("writes", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} write rows differ:\n" + "\n".join(bad)


def test_unsafe_view_aliases_but_is_not_an_autograd_view():
    """Upstream's `_unsafe_view` shares its input's storage and is not an
    autograd view: `_is_view()` False, `_base` None, even when its layout has
    an offset. Red when `note_view` leaves the offset row to the layout rule
    (`_is_view()` True, `_base` refuses)."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(n for n in up["cases"] if "unsafe" in n)
    for name in names:
        u = up["cases"][name]
        assert (u["shares"], u["is_view"], u["base"]) == (True, False, "none"), (name, u)
    bad = _diff("cases", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} _unsafe_view rows differ:\n" + "\n".join(bad)


def test_the_dense_layout_is_the_meta_layout():
    """The meta arms of `view` and `reshape` already used `view_stride`; the
    dense kernels now do too, so the two agree on shape, stride, offset,
    contiguity and view-ness for every row both can build, and both agree
    with upstream's meta answer."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(up["meta"])
    assert len(names) >= 20, names
    keys = ("shape", "stride", "offset", "contiguous", "is_view", "base")
    bad = []
    for name in names:
        dense = {k: shim["cases"][name].get(k) for k in keys}
        meta = {k: shim["meta"].get(name, {}).get(k) for k in keys}
        want = {k: up["meta"][name].get(k) for k in keys}
        if not (dense == meta == want):
            bad.append(f"  {name}\n    upstream meta {want}\n    shim meta     {meta}\n    shim dense    {dense}")
    assert not bad, f"{len(bad)} of {len(names)} rows disagree:\n" + "\n".join(bad)


def test_copies_are_laid_out_as_upstreams_and_a_view_of_them_succeeds():
    """`clone` (both formats), `_to_copy` (both formats), a same-dtype `to`,
    `cat` of one and of two strided entries, `repeat`, and the out-of-place
    `index_add`/`index_copy`, on every strided input: shape, stride, offset,
    contiguity, no sharing with the input, values, and whether `.view(-1)` of
    the result succeeds. Red under each nullification (none run locally):

    * `clone_default` back to `Tensor::copy` (keeps the input's layout): the
      `clone|expanded`, `clone_contig|*` rows (stride, `view_flat`);
    * `copy_in_format` ignoring `contiguous_format`: `clone_contig|transposed`;
    * `copy_in_format` keeping a non-dense stride instead of the permutation:
      `clone|stepped`, `clone|offset` (offset 1 vs 0);
    * `_to_copy`'s same-dtype path back to `Tensor::copy`: `to_copy|expanded`;
    * `cat_default` without the closing `force_contiguous`: `cat_one|*`
      (`shares` True) and `cat_two_last|transposed` (stride);
    * `repeat_default` back to `Tensor::copy`: `repeat_ones|t2d`;
    * the index ops' targets back to `Tensor::copy`: `index_*|t2d`."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(up["copies"])
    assert len(names) >= 130, len(names)
    refused = [n for n in names if "raised" in up["copies"][n]]
    assert not refused, f"upstream refused copy rows: {refused}"
    for name in names:
        u = up["copies"][name]
        assert u["shares"] is False and u["base"] == "none" and u["offset"] == 0, (name, u)
    # The oracle's own layout, for the rows the CI failures were about.
    assert up["copies"]["clone|expanded"]["stride"] == [4, 1], up["copies"]["clone|expanded"]
    assert up["copies"]["clone_contig|transposed"]["stride"] == [8, 4, 1]
    assert up["copies"]["clone|transposed"]["stride"] == [4, 12, 1]
    bad = _diff("copies", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} copy rows differ:\n" + "\n".join(bad)


def test_the_ci_scenarios_compute_upstreams_answer():
    """CI run 43808445902's refusals, each reproduced: `repeat_interleave`
    (bootstrap.py, `expand` then `clone` then `view`) and the `reshape` and
    `matmul` bodies (`clone(memory_format=contiguous_format)` then
    `_unsafe_view`). Upstream computes every one; the shim raised `view size
    is not compatible ...`. Red when `clone` keeps its input's layout."""
    pair = _both()
    if pair is None:
        return
    shim, up = pair
    names = sorted(up["scenarios"])
    assert len(names) == 7, names
    refused = [n for n in names if "raised" in up["scenarios"][n]]
    assert not refused, f"upstream refused: {refused}"
    bad = _diff("scenarios", names, shim, up)
    assert not bad, f"{len(bad)} of {len(names)} scenarios differ:\n" + "\n".join(bad)


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = _skip.run_tests(items, suite="test_viewstride")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
