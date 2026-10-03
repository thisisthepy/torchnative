"""The ten operators the deepened derivation found, and the refusal they now get.

`docs/devices/matrix.md` §7.14 measured that twelve cells published AGREES in
§6's `mps` columns were computed on the host, and left the fix as a capability
decision. §7.16 takes it: the operators are refused by name, and the
derivation that failed to see them follows calls instead of matching a fixed
set of helper names.

**What this file is for, and what it deliberately is not.** It is not another
counter test -- `test_metalplace.py` owns the counters and `test_metalcount.py`
owns their calibration. It is about the two things a refusal has to be, and
neither is checkable by a counter:

1. **The refusal names what still works, and the name is measured.**
   `device.rs::MPS_HOST_READBACK_NOTES` says, per op, which dtypes agree with
   upstream on the CPU. A table of comfort written beside the implementation
   is exactly the §7.9 `clamp` shape -- a claim nobody re-ran. So every dtype
   in every note is re-measured here against an oracle computed in a
   **separate subprocess**, at `tests/golden/dtypes.py`'s derived tolerance.
   A note that drifts turns the gate red.

2. **The derivation finds the ten without being told them.** A derivation that
   works because somebody added the answer to a list is the defect wearing a
   fix. This proves the independence mechanically rather than by reading:
   `_C._shim_mps_host_readback_ops()` is stubbed to the empty list and the
   derivation is re-run, and the ten still come out of it.

**What is withdrawn, stated plainly.** Twelve `AGREES` cells, all integral
dtypes. The float columns of these ops were *already* not running on `mps`:
`sort` on `float32`/`mps` raises `Metal contiguous to_dtype F32 F64 not
implemented` from inside candle, because `read_flat` widens to `f64` and Metal
has no double. So the refusal costs the integral columns and re-labels the
rest; it does not take a working float path away. That is asserted below from
the document rather than left as prose.
"""

import json
import os
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import test_shim
from test_shim import _C
import _skip

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "tests", "golden"))
import dtypes as dt_utils  # noqa: E402

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    "..")

# The ten, as data. Grown only by a visible edit here, and asserted in both
# directions against the artefact's own table.
NEWLY_REFUSED = (
    "aten.argsort.default",
    "aten.argsort.stable",
    "aten.floor_divide.Scalar",
    "aten.floor_divide.default",
    "aten.linalg_vector_norm.default",
    "aten.norm.ScalarOpt_dim",
    "aten.scatter_.src",
    "aten.scatter_.value",
    "aten.sort.default",
    "aten.topk.default",
)

# §6's column order, and the note cross-check below depends on it.
_CPU_COLUMNS = ("float32", "float16", "bfloat16", "float64",
                "int64", "int32", "int8", "bool")

_SHAPE = [2, 3]
_VALUES = [-3.0, -1.0, 0.5, 2.0, -7.0, 5.0]
_OTHER = [1.0, 2.0, 4.0, -1.0, 8.0, -2.0]
_INDEX = [0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

# Integral dtypes take integral values: `0.5` is not representable and a
# truncation would make `sort` compare a different tensor than the oracle
# does. The oracle is handed the same list, so both sides sort the same data.
_INT_VALUES = [-3.0, -1.0, 1.0, 2.0, -7.0, 5.0]
_BOOL_VALUES = [1.0, 0.0, 1.0, 0.0, 1.0, 1.0]


def _values_for(dtype):
    if dtype == "bool":
        return _BOOL_VALUES
    if dtype.startswith("int") or dtype.startswith("uint"):
        return _INT_VALUES
    return _VALUES


def _other_for(dtype):
    # No zero: `floor_divide` by zero is defined differently per dtype and is
    # not the subject here.
    if dtype == "bool":
        return [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    return _OTHER


def _mps_or_skip(what):
    if not _C._metal_counters()["built"]:
        _skip.skip("   (skipped %s: not an Apple build)" % what)
        return None
    try:
        _C._aten_dispatch("aten.ones.default", [1], device=_C.device("mps"))
    except (NotImplementedError, RuntimeError) as e:
        _skip.skip("   (skipped %s: no mps device on this machine -- %s)"
                   % (what, str(e).splitlines()[0]))
        return None
    return _C.device("mps")


def _tensor(values, dtype, device=None):
    return _C._tensor_from_flat(list(values), list(_SHAPE),
                                dt_utils.c_dtype(_C, dtype), device) \
        if device is not None else \
        _C._tensor_from_flat(list(values), list(_SHAPE),
                             dt_utils.c_dtype(_C, dtype))


def _call(op, dtype, device=None):
    """`(args, kwargs)` for one dispatch of `op` at `dtype`.

    One place, shared by the shim call and by the oracle payload, so the two
    sides cannot drift into testing different calls.
    """
    if op in ("aten.sort.default", "aten.argsort.default"):
        return (0,), {}
    if op == "aten.argsort.stable":
        return (True, 0), {}
    if op == "aten.topk.default":
        return (2,), {}
    if op == "aten.floor_divide.default":
        return (_tensor(_other_for(dtype), dtype, device),), {}
    if op == "aten.floor_divide.Scalar":
        return (2,), {}
    if op == "aten.scatter_.src":
        return (1, _tensor(_INDEX, "int64", device),
                _tensor(_other_for(dtype), dtype, device)), {}
    if op == "aten.scatter_.value":
        return (1, _tensor(_INDEX, "int64", device), 7.0), {}
    if op == "aten.linalg_vector_norm.default":
        return (2.0,), {}
    if op == "aten.norm.ScalarOpt_dim":
        return (2, [1], False), {}
    raise AssertionError("no call shape for " + op)


_ORACLE = r"""
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented"), (
    "the oracle subprocess imported the shim, not upstream torch")

def flat(x):
    return [float(v) for v in x.to(torch.float64).flatten().tolist()]

req = json.loads(sys.argv[1])
out = {}
for key, case in req.items():
    dt = getattr(torch, case["dtype"])
    shape = case["shape"]
    t = torch.tensor(case["values"]).reshape(shape).to(dt)
    other = None
    if case["other"] is not None:
        other = torch.tensor(case["other"]).reshape(shape).to(dt)
    index = torch.tensor(case["index"]).reshape(shape).to(torch.int64)
    op = case["op"]
    try:
        if op == "aten.sort.default":
            r = torch.ops.aten.sort(t, 0)
            v = flat(r[0]) + flat(r[1])
        elif op == "aten.argsort.default":
            v = flat(torch.ops.aten.argsort(t, 0))
        elif op == "aten.argsort.stable":
            v = flat(torch.ops.aten.argsort(t, stable=True, dim=0))
        elif op == "aten.topk.default":
            r = torch.ops.aten.topk(t, 2)
            v = flat(r[0]) + flat(r[1])
        elif op == "aten.floor_divide.default":
            v = flat(torch.ops.aten.floor_divide(t, other))
        elif op == "aten.floor_divide.Scalar":
            v = flat(torch.ops.aten.floor_divide(t, 2))
        elif op == "aten.scatter_.src":
            v = flat(t.clone().scatter_(1, index, other))
        elif op == "aten.scatter_.value":
            v = flat(t.clone().scatter_(1, index, 7.0))
        elif op == "aten.linalg_vector_norm.default":
            v = flat(torch.ops.aten.linalg_vector_norm(t, 2.0))
        elif op == "aten.norm.ScalarOpt_dim":
            v = flat(torch.ops.aten.norm(t, 2, [1], False))
        else:
            raise AssertionError(op)
    except Exception as e:                       # noqa: BLE001
        out[key] = {"error": "%s: %s" % (type(e).__name__, e)}
        continue
    out[key] = {"values": v}
print(json.dumps(out))
"""


def _oracle(payload):
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("TORCH_C_ARTEFACT", None)
    proc = subprocess.run([sys.executable, "-c", _ORACLE, json.dumps(payload)],
                          capture_output=True, text=True, timeout=900, env=env)
    if proc.returncode != 0:
        raise AssertionError("upstream oracle subprocess failed:\n"
                             + proc.stderr[-3000:])
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _flat(t):
    h = t.cpu() if str(t.device) != "cpu" else t
    return [float(v) for v in h.to(_C.float64).flatten().tolist()]


def _shim_values(op, dtype):
    args, kwargs = _call(op, dtype)
    t = _tensor(_values_for(dtype), dtype)
    r = _C._aten_dispatch(op, t, *args, **kwargs)
    if op in ("aten.sort.default", "aten.topk.default"):
        return _flat(r[0]) + _flat(r[1])
    if op in ("aten.scatter_.src", "aten.scatter_.value"):
        return _flat(t)
    return _flat(r)


def _notes():
    return dict(_C._shim_mps_host_readback_notes())


# ---------------------------------------------------------------------------
# 1. the refusal
# ---------------------------------------------------------------------------

def test_the_ten_newly_refused_ops_are_refused_and_the_message_says_what_works():
    """Half one: the refusal happens, and it is a direction rather than a wall.

    Four things are asserted and each is separately load-bearing. That it
    raises `NotImplementedError` -- a `RuntimeError` would not be caught by a
    caller written against the other 89. That the message names the **op** and
    the **device**, because a refusal that does not say which op is one nobody
    can act on. That it offers `.cpu()`. And that it carries the op's own
    note, so a user hitting `sort` on `int64`/`mps` learns where the operator
    *does* agree instead of only that this failed.

    The note is compared to the artefact's table rather than to a string
    written here: a test holding its own copy of the wording would agree with
    itself after somebody changed the build.
    """
    mps = _mps_or_skip("the ten newly refused operators")
    if mps is None:
        return
    refused = set(_C._shim_mps_host_readback_ops())
    notes = _notes()
    assert set(notes) == set(NEWLY_REFUSED), (
        "MPS_HOST_READBACK_NOTES is meant to cover exactly the ten §7.16 "
        "added: " + repr(sorted(set(notes) ^ set(NEWLY_REFUSED))))

    for op in NEWLY_REFUSED:
        assert op in refused, (
            "%s is not in MPS_HOST_READBACK_OPS. It reaches read_flat on "
            "every path -- docs/devices/matrix.md §7.16 -- so taking it off "
            "the list without removing the readback is the MPSATTN.md §3.1 "
            "defeat." % op)
        args, kwargs = _call(op, "int64", mps)
        t = _tensor(_values_for("int64"), "int64", mps)
        try:
            _C._aten_dispatch(op, t, *args, **kwargs)
        except NotImplementedError as e:
            message = str(e)
        else:
            raise AssertionError(
                "%s answered an mps dispatch. It computes on the host; §6's "
                "table publishes it REFUSES on every mps dtype." % op)
        assert op in message, message
        assert "mps" in message, message
        assert ".cpu()" in message, message
        assert notes[op] in message, (
            "the refusal for %s does not carry its note (%r). Without it the "
            "message says what failed and not what works." % (op, notes[op]))


def test_no_newly_refused_op_answers_any_mps_dtype():
    """The withdrawal is dtype-independent, and that is measured not assumed.

    The gate reads the op name only. If some dtype still got through, the
    published grade for that cell would be wrong in the other direction --
    §6's table says REFUSES for all sixteen `mps` cells of these ten
    operators, and a table that is wrong optimistically is the failure this
    whole section exists to correct.

    `int8` and `float64` are excluded: `_tensor_from_flat` cannot build them
    on Metal at all (§7.13), so the refusal that fires there is a different
    one and asserting this gate for them would be asserting the wrong cause.
    """
    mps = _mps_or_skip("the dtype independence of the ten refusals")
    if mps is None:
        return
    checked = 0
    for op in NEWLY_REFUSED:
        for dtype in ("float32", "float16", "bfloat16", "int64", "int32", "bool"):
            try:
                t = _tensor(_values_for(dtype), dtype, mps)
                args, kwargs = _call(op, dtype, mps)
            except (NotImplementedError, RuntimeError):
                continue      # the operand cannot exist on Metal; not this gate
            try:
                _C._aten_dispatch(op, t, *args, **kwargs)
            except NotImplementedError as e:
                assert "reads the tensor back to host memory" in str(e), (
                    op, dtype, str(e))
                checked += 1
            except RuntimeError as e:
                # candle bounced it before this gate could. That means the op
                # is no longer on `MPS_HOST_READBACK_OPS` -- the readback is
                # still in the kernel and only the float dtypes happen to be
                # stopped, by accident, one layer further in. This is the
                # docs/devices/MPSATTN.md §3.1 defeat and it is named rather
                # than allowed to surface as a bare exception.
                raise AssertionError(
                    "%s at %s was not stopped by the host-readback gate; "
                    "candle refused it instead (%s). The gate is name-based, "
                    "so this means the name came off the list while the "
                    "readback stayed in the kernel."
                    % (op, dtype, str(e).splitlines()[0]))
            else:
                raise AssertionError(
                    "%s answered on mps at %s -- the host-readback gate is "
                    "name-based and must fire for every dtype" % (op, dtype))
    assert checked >= 40, (
        "only %d (op, dtype) pairs reached the gate; the operands stopped "
        "building on Metal and this test has quietly become empty" % checked)


# ---------------------------------------------------------------------------
# 2. the note is measured, not asserted
# ---------------------------------------------------------------------------

def test_every_dtype_a_refusal_note_advertises_agrees_on_the_cpu():
    """Half two, and the grade is **AGREES**.

    Each note names the dtypes whose CPU column agrees with upstream. This
    runs every one of them through the shim on the CPU and compares
    element-wise with upstream computed in a **separate subprocess**, at the
    derived tolerance from `tests/golden/dtypes.py` -- never widened here.

    Sort-family operators are compared on values *and* indices, because a
    permutation that is right elementwise on values can still be the wrong
    permutation; `scatter_` is compared on the mutated `self`, which is the
    whole output of an in-place op.

    **This is the test that stops the note being comfort.** A note is a claim
    about where the op works, printed to a user at the moment they are
    deciding what to do next, and §7.9's `clamp` is what an unmeasured claim
    of that kind costs.
    """
    notes = _notes()
    payload, plan = {}, []
    for op in sorted(notes):
        for dtype in [d.strip() for d in notes[op].split(",")]:
            key = "%s|%s" % (op, dtype)
            payload[key] = {
                "op": op,
                "dtype": dtype,
                "shape": _SHAPE,
                "values": _values_for(dtype),
                "other": _other_for(dtype),
                "index": _INDEX,
            }
            plan.append((key, op, dtype))
    assert len(plan) >= 40, len(plan)

    upstream = _oracle(payload)
    bad = []
    for key, op, dtype in plan:
        got = upstream[key]
        if "error" in got:
            bad.append("%s: the note advertises this dtype but upstream itself "
                       "refuses it (%s)" % (key, got["error"]))
            continue
        try:
            ours = _shim_values(op, dtype)
        except (NotImplementedError, RuntimeError, TypeError) as e:
            bad.append("%s: the note advertises this dtype but the shim's CPU "
                       "path raises %s: %s" % (key, type(e).__name__, e))
            continue
        tol = dt_utils.tolerance_for(dtype)
        if len(ours) != len(got["values"]):
            bad.append("%s: %d values vs upstream's %d"
                       % (key, len(ours), len(got["values"])))
            continue
        for i, (x, y) in enumerate(zip(ours, got["values"])):
            if x != x and y != y:
                continue
            if abs(x - y) > tol.atol + tol.rtol * abs(y):
                bad.append("%s: index %d shim=%r upstream=%r (atol=%r rtol=%r)"
                           % (key, i, x, y, tol.atol, tol.rtol))
                break
    assert not bad, (
        "these dtypes are advertised by a refusal note and do not agree with "
        "upstream on the CPU. Narrow the note in device.rs to what the build "
        "actually does -- do not widen the tolerance:\n  " + "\n  ".join(bad))


# ---------------------------------------------------------------------------
# 3. the derivation found them, and not from a list
# ---------------------------------------------------------------------------

def test_the_derivation_finds_the_ten_without_being_told_their_names():
    """Half three, and the only one of the three that is the actual deliverable.

    Adding ten names to a list closes twelve cells once. What stops the class
    from recurring is a derivation that follows calls rather than matching
    names, and the way to know it does is to take the answer away from it.

    So `_C._shim_mps_host_readback_ops()` is stubbed to the empty list and
    `test_shim._ops_that_reach_the_host()` is re-run. It reads `aten.rs` and
    its siblings and nothing else, so the ten must still come out -- each with
    a witness path ending at a function that really holds a readback marker.
    The witness is checked, not just the membership: a derivation that
    returned every op would satisfy the membership assertion and be useless.
    """
    witness = test_shim._ops_that_reach_the_host()
    if witness is None:
        _skip.skip("   (skipped: crates/torch_c/src is not beside this file -- "
                   "installed rather than in-tree)")
        return

    saved = _C._shim_mps_host_readback_ops
    try:
        _C._shim_mps_host_readback_ops = lambda: []
        blinded = test_shim._ops_that_reach_the_host()
    finally:
        _C._shim_mps_host_readback_ops = saved
    assert blinded == witness, (
        "the derivation changed when the refusal list was emptied, so it is "
        "reading the answer rather than deriving it")

    mods = test_shim._all_rs_modules()
    missing = [op for op in NEWLY_REFUSED if op not in witness]
    assert not missing, (
        "the derivation no longer reaches %r. These kernels read their "
        "operand to the host through an un-named hop (order_along, "
        "argsort_core, floor_divide_impl, norm_pow_walk, scatter_src); a "
        "derivation that cannot follow that is the one §7.14 found twelve "
        "published cells sitting in." % (missing,))

    for op in NEWLY_REFUSED:
        path = witness[op]
        assert len(path) >= 3, (
            "%s is derived through %r -- one hop. If that is now true the "
            "kernel changed, and this test is no longer proving the "
            "transitive half works." % (op, path))
        last_module, last_fn = path[-1]
        body = mods[last_module][0][last_fn]
        assert test_shim._MPS_READBACK_MARKERS.search(body), (
            op, path, "the witness path does not end at a real readback")

    # And the derivation is not simply everything: most ops do not reach the
    # host, and an assertion set that passed on a derivation returning all of
    # them would prove nothing.
    _, text = test_shim._aten_rs_functions()
    total = len(test_shim._aten_dispatch_targets(text))
    assert 40 < len(witness) < total * 0.6, (
        "the derivation returned %d of %d dispatched ops -- it has stopped "
        "discriminating" % (len(witness), total))


def test_a_readback_behind_two_un_named_hops_is_still_derived():
    """The mutant §7.12 planted, written as a test instead of built as a fork.

    §7.12's M2 hid a readback one file over and the whole gate stayed green.
    §7.14 found twelve production cells hidden by an un-named hop in the *same*
    file. Both are the same defect in the instrument, and both are now
    supposed to be impossible.

    This asserts it on synthetic sources rather than on the real tree, so it
    runs on a machine with no Metal and cannot be satisfied by an accident of
    what `aten.rs` currently contains: a kernel calling a helper calling
    another helper that holds the marker, in the same file and across files,
    must be derived -- and a kernel calling a *method* named `to_le_bytes`
    must not be, which is the false-positive direction a previous round found
    the hard way.
    """
    import re as _re

    def derive(sources):
        mods = {name: test_shim._rs_functions_from_text(text)
                for name, text in sources.items()}
        graph, direct = {}, set()
        for module, (bodies, _) in mods.items():
            for fn, body in bodies.items():
                graph[(module, fn)] = test_shim._callees(mods, module, body)
                if test_shim._MPS_READBACK_MARKERS.search(body):
                    direct.add((module, fn))
        seen = set(direct)
        frontier = list(direct)
        callers = {}
        for node, callees in graph.items():
            for callee in callees:
                callers.setdefault(callee, set()).add(node)
        while frontier:
            nxt = []
            for node in frontier:
                for caller in callers.get(node, ()):
                    if caller not in seen:
                        seen.add(caller)
                        nxt.append(caller)
            frontier = nxt
        return seen

    sources = {
        "aten": """
fn deep_kernel(t: &Tensor) -> PyResult<Tensor> { helper_one(t) }
fn helper_one(t: &Tensor) -> PyResult<Tensor> { helper_two(t) }
fn helper_two(t: &Tensor) -> PyResult<Tensor> { let v = t.to_vec1::<i64>()?; ok(v) }
fn across(t: &Tensor) -> PyResult<Tensor> { crate::tensor::twin(t) }
fn clean_kernel(x: i64) -> Vec<u8> { x.to_le_bytes().to_vec() }
fn also_clean(t: &Tensor) -> PyResult<Tensor> { t.affine(1.0, 0.0) }
""",
        "tensor": """
fn twin(t: &Tensor) -> PyResult<Tensor> { let v = t.to_vec1::<f32>()?; ok(v) }
fn to_le_bytes(t: &Tensor) -> PyResult<Vec<u8>> { let v = t.to_vec1::<u8>()?; ok(v) }
""",
    }
    reached = derive(sources)
    for fn in ("deep_kernel", "helper_one", "helper_two", "across"):
        assert ("aten", fn) in reached, (
            "%s reaches a readback and the derivation missed it -- this is the "
            "un-named-hop blind spot reopening" % fn)
    for fn in ("clean_kernel", "also_clean"):
        assert ("aten", fn) not in reached, (
            "%s does not touch a dispatched tensor and was derived anyway. "
            "`to_le_bytes` is an inherent method on every Rust integer; "
            "matching it by bare name flagged a dozen clean kernels once "
            "already." % fn)
    assert _re.search(r"to_le_bytes", sources["aten"]), "fixture lost its point"


# ---------------------------------------------------------------------------
# 4. the document moved with the build
# ---------------------------------------------------------------------------

def test_the_matrix_grades_every_mps_cell_of_the_ten_as_refuses():
    """§6's table has to say what the build does, and this is what forces it.

    The gate is name-based, so every `mps` cell of these ten operators is now
    a refusal. Twelve of them were published `AGREES`. A test that only
    checked the build would leave the document free to keep the old grade --
    which is exactly how twelve wrong cells survived four rounds of
    re-measurement.

    The CPU columns are checked too, in the other direction: they must **not**
    all have become REFUSES. The withdrawal is scoped to `mps`, and a change
    that took the CPU with it would be a much larger one than §7.16 describes.
    """
    path = os.path.join(REPO, "docs", "devices", "matrix.md")
    notes = _notes()
    rows = {}
    for line in open(path, encoding="utf-8"):
        if not line.startswith("| aten."):
            continue
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        rows.setdefault(cells[0], cells[1:])
    for op in NEWLY_REFUSED:
        assert op in rows, (op, "no row in §6's table")
        cells = rows[op]
        assert len(cells) == 16, (op, len(cells))
        mps = cells[8:]
        assert all(c == "REFUSES" for c in mps), (
            "%s is refused on mps by name for every dtype, but §6's table "
            "grades its mps columns %r. docs/devices/matrix.md §7.16 and the "
            "table have to move with the build." % (op, mps))
        # Every dtype a note advertises must be graded AGREES in §6's CPU
        # column for that operator.
        #
        # **This is here because the subprocess oracle alone was not enough,
        # and that was measured rather than assumed.** A mutant that added
        # `float32` to `scatter_.src`'s note survived
        # `test_every_dtype_a_refusal_note_advertises_agrees_on_the_cpu`: at
        # the one shape that test uses, `[2, 3]` with an `int64` index, the
        # shim and upstream do agree -- while §6 grades the cell BREAKS from
        # a different probe. That is §7.9's `clamp` in miniature: a cell
        # passing is not an operator being right, and one shape cannot speak
        # for a column. Tying the note to §6's grade puts the whole matrix
        # sweep behind it rather than one tensor.
        cpu = cells[:8]
        for dtype in [d.strip() for d in notes.get(op, "").split(",") if d.strip()]:
            assert dtype in _CPU_COLUMNS, (op, dtype)
            grade = cpu[_CPU_COLUMNS.index(dtype)]
            assert grade == "AGREES", (
                "%s's refusal note advertises %s, but §6's table grades its "
                "%s_cpu cell %s. The note is printed to a user deciding what "
                "to do next; advertising a dtype the matrix says does not "
                "agree sends them somewhere that does not work."
                % (op, dtype, dtype, grade))

        assert any(c == "AGREES" for c in cpu), (
            "%s has no AGREES left on the CPU. The refusal is scoped to mps; "
            "if the CPU went with it that is a far bigger change than §7.16 "
            "describes." % op)


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)
             and getattr(fn, "__module__", None) == __name__]
    failures = _skip.run_tests(items, suite="test_mpsrefuse")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
