"""`_dispatch_is_included_in_alias(key, alias)` -- the thing `resolve_key` dies on.

`docs/graph/METAKEY.md` §2.2 measured the Meta predicate's `False`-where-upstream-
says-`True` and found the harm is not that a caller takes a slower path: it is
that `resolve_key` (`torch/_ops.py:213`) falls past its branch 1 into branch 2.1,
which calls this predicate, which was an `_Unimplemented` and **raised**.
METAKEY §5 left it open on purpose and named it as the gap that has to close
first. This file closes it.

Every upstream answer here is computed **in a separate subprocess**, never read
from a table written beside the implementation. What that measurement found:

* Upstream's answer is a total function of two `DispatchKey`s with exactly two
  edge rules and one lookup. Over the 145x145 cross product it answers 20881
  pairs and **raises for 144** -- every one of them `alias == Undefined`, an
  internal assert in `c10/core/DispatchKeySet.cpp:9`. `key == Undefined`
  answers `False` and does not raise, because upstream tests that first.

* **Only six aliases expand beyond themselves**: `Autograd` (27),
  `CompositeExplicitAutograd` (82), `CompositeExplicitAutogradNonFunctional`
  (57), `CompositeImplicitAutograd` (122),
  `CompositeImplicitAutogradNestedTensor` (26) and
  `FuncTorchBatchedDecomposition` (7). Every other one of the 145 keys includes
  only itself. Notably `ADInplaceOrView`, which reads like an alias and is not
  one here.

* So the rule is `k != Undefined and (k == alias or k in EXPANSION[alias])`,
  and that rule reproduces upstream on **20869 of 20881** pairs. The 12 misses
  are six symmetric pairs of *enum-value aliases* -- `EndOfDenseBackends` is
  the same number as `Meta`, `EndOfSparseBackends` the same as `SparseMeta` --
  which upstream's pybind enum exposes as distinct names for one value. None
  of those six names exists in the vendored `DispatchKey`, so the rule is exact
  on everything this shim can be asked about. That is measured, not assumed:
  `test_the_six_value_alias_pairs_the_name_rule_cannot_see_are_all_absent_here`.

**The vendored enum and upstream's runtime enum are not the same enum**, and it
is upstream's own stub that is behind, not this shim's copy of it.
`torch/_C/__init__.pyi` declares 140 keys, `surface.json` mirrors all 140
exactly, and upstream's pybind enum has 145 -- 123 shared, 17 stub-only
(`Vulkan`, `MKLDNN`, `Named`, `Tracer`, ...) and 22 runtime-only (the
`StartOf*`/`EndOf*` sentinels, the `MAIA` backends, `Quantized`). "Agreement
with upstream" is therefore only *definable* on the 123, and
`test_the_enum_divergence_that_bounds_what_agreement_can_mean` pins the
partition so that a vendor bump which changes it reddens here rather than
silently shrinking what the agreement test covers.

That last test is what keeps the expansion lists from being a bare list.
`docs/graph/METAKEY.md` records that this project spent a week removing rotted
lists; the difference is that these are re-derived from a live upstream every
run and the partition they are valid over is asserted, so staleness is a RED
test rather than a wrong answer.

What this file does NOT establish: that closing the gap moves anything
downstream. It does not.  `test_closing_the_gap_lets_resolve_key_answer_where_it_used_to_raise`
measures the one thing that changes -- `resolve_key` returns instead of raising
-- and `test_the_export_that_motivated_this_still_makes_zero_alias_queries`
prints, without asserting, that the export METAKEY §2.2 used still reaches this
predicate zero times. See `docs/graph/ALIASINC.md` §4.
"""

import json
import os
import subprocess
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "python")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

#: The six aliases upstream expands beyond themselves. Asserted against a live
#: upstream by `test_exactly_six_aliases_expand_beyond_themselves`, so a new
#: alias key upstream reddens rather than being silently answered `False`.
_EXPANDING = (
    "Autograd",
    "CompositeExplicitAutograd",
    "CompositeExplicitAutogradNonFunctional",
    "CompositeImplicitAutograd",
    "CompositeImplicitAutogradNestedTensor",
    "FuncTorchBatchedDecomposition",
)

#: The four `resolve_key` actually consults (`torch/_ops.py:218-254` consults
#: six `cand`s, and these are the ones reached before the backend fallback).
_RESOLVE_KEY_CANDIDATES = (
    "CompositeExplicitAutogradNonFunctional",
    "CompositeExplicitAutograd",
    "CompositeImplicitAutogradNestedTensor",
    "CompositeImplicitAutograd",
    "Autograd",
    "FuncTorchBatchedDecomposition",
)


def _available():
    return os.path.isfile(_VENDOR_SHIM)


def _run(script, shim, stdin="", timeout=900):
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", script], input=stdin,
        capture_output=True, text=True, env=env, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    # The side each probe landed on is asserted, not assumed: an empty vendored
    # tree makes the "shim" side silently import upstream.
    assert out["is_shim"] is shim, f"probe meant for shim={shim} ran on the other side"
    return out


_PREAMBLE = """
import json, warnings
warnings.filterwarnings("ignore")
import torch
IS_SHIM = hasattr(torch._C, "_aten_implemented")
from torch._C import DispatchKey
NAMES = sorted(DispatchKey.__members__) if hasattr(DispatchKey, "__members__") \\
    else sorted(k.name for k in DispatchKey)
"""


# --- the full key-by-alias cross product, on whichever side the probe runs --
_CROSS = _PREAMBLE + """
rows, errs = {}, {}
for a in NAMES:
    for b in NAMES:
        try:
            rows[a + "|" + b] = bool(torch._C._dispatch_is_included_in_alias(
                getattr(DispatchKey, a), getattr(DispatchKey, b)))
        except Exception as exc:
            errs[a + "|" + b] = type(exc).__name__
print(json.dumps({"is_shim": IS_SHIM, "names": NAMES, "rows": rows, "errs": errs}))
"""


# --- which names each side's enum has -------------------------------------
_NAMES = _PREAMBLE + """
print(json.dumps({"is_shim": IS_SHIM, "names": NAMES}))
"""


# --- does `resolve_key` answer, for the op METAKEY 2.2 measured? -----------
_RESOLVE = _PREAMBLE + """
from torch._ops import resolve_key
out = {}
for spelling in ["as_strided.default", "native_layer_norm.default", "view.default",
                 "clone.default", "matmul.default", "add.Tensor"]:
    name, ov = spelling.split(".")
    try:
        op = getattr(getattr(torch.ops.aten, name), ov)
    except Exception as exc:
        out[spelling] = "NO-OP:" + type(exc).__name__
        continue
    try:
        out[spelling] = getattr(resolve_key(op, DispatchKey.Meta), "name", "?")
    except Exception as exc:
        out[spelling] = type(exc).__name__ + ":" + str(exc)[:80]
print(json.dumps({"is_shim": IS_SHIM, "resolved": out}))
"""


# --- how many alias queries does the motivating export actually make? ------
_EXPORT_QUERIES = _PREAMBLE + """
import collections, traceback
original = torch._C._dispatch_is_included_in_alias
sites = collections.Counter()

def recording(k, alias, *a, **kw):
    frame = traceback.extract_stack()[-2]
    sites[f"{frame.filename.split('/torch/')[-1]}:{frame.name}"] += 1
    return original(k, alias, *a, **kw)

torch._C._dispatch_is_included_in_alias = recording

class M(torch.nn.Module):
    def __init__(self):
        super().__init__(); self.ln = torch.nn.LayerNorm(6)
    def forward(self, x): return self.ln(x)

exported = True
try:
    torch.export.export(M().eval(), (torch.randn(4, 6),))
except Exception:
    exported = False
torch._C._dispatch_is_included_in_alias = original
print(json.dumps({"is_shim": IS_SHIM, "exported": exported,
                  "queries": sum(sites.values()), "sites": dict(sites)}))
"""


_CACHE = {}


def _cross(shim):
    key = ("cross", shim)
    if key not in _CACHE:
        _CACHE[key] = _run(_CROSS, shim)
    return _CACHE[key]


def _names(shim):
    key = ("names", shim)
    if key not in _CACHE:
        _CACHE[key] = _run(_NAMES, shim)
    return _CACHE[key]["names"]


def _skip(name):
    print(f"SKIP {name}: no vendored shim at {_VENDOR_SHIM}")


# ---------------------------------------------------------------------------
# 1. The bar: agreement with upstream, key by key and alias by alias.
# ---------------------------------------------------------------------------
def test_the_shim_agrees_with_upstream_over_the_full_key_by_alias_cross_product():
    """Every shared (key, alias) pair, both sides, separate subprocesses.

    This is the round's bar and it is deliberately not "returns a bool without
    raising". It compares 123 x 123 = 15129 answers. A rule that got the
    `CompositeExplicitAutograd` expansion wrong by one key reddens here with
    that key named.
    """
    if not _available():
        return _skip("cross product")
    up, shim = _cross(False), _cross(True)
    shared = sorted(set(up["names"]) & set(shim["names"]))
    assert len(shared) > 100, f"only {len(shared)} shared keys -- vendored enum is wrong"

    disagree = []
    for a in shared:
        for b in shared:
            pair = a + "|" + b
            u = up["rows"].get(pair, "RAISED:" + up["errs"].get(pair, "?"))
            s = shim["rows"].get(pair, "RAISED:" + shim["errs"].get(pair, "?"))
            if u != s:
                disagree.append(f"({a}, {b}): upstream={u} shim={s}")
    assert not disagree, (
        f"{len(disagree)} of {len(shared) ** 2} pairs disagree with upstream; "
        f"first 20:\n" + "\n".join(disagree[:20])
    )
    print(f"ALIASINC cross product: {len(shared) ** 2} shared pairs, 0 disagreements")


# ---------------------------------------------------------------------------
# 2. The two edge rules, which are not symmetric.
# ---------------------------------------------------------------------------
def test_an_undefined_alias_raises_while_an_undefined_key_answers_false():
    """Upstream's asymmetry, reproduced rather than tidied away.

    `k == Undefined` returns `False`; `alias == Undefined` trips
    `t != DispatchKey::Undefined INTERNAL ASSERT FAILED`. Both sides, all 144
    upstream pairs. A shim that "helpfully" returned `False` for an Undefined
    alias would hide a caller bug that upstream makes loud.
    """
    if not _available():
        return _skip("undefined edges")
    up, shim = _cross(False), _cross(True)
    shared = sorted(set(up["names"]) & set(shim["names"]))
    assert "Undefined" in shared

    assert all(k + "|Undefined" in up["errs"] for k in shared if k != "Undefined"), \
        "upstream stopped raising for an Undefined alias -- re-derive the edge rule"
    raised = [k for k in shared if k != "Undefined" and k + "|Undefined" in shim["errs"]]
    assert len(raised) == len(shared) - 1, (
        f"shim raises for only {len(raised)} of {len(shared) - 1} Undefined-alias pairs"
    )

    assert up["rows"]["Undefined|Undefined"] is False
    assert shim["rows"]["Undefined|Undefined"] is False
    for b in shared:
        if b == "Undefined":
            continue
        assert shim["rows"]["Undefined|" + b] is False, f"Undefined key answered True for {b}"


# ---------------------------------------------------------------------------
# 3. The shape: six expanding aliases, and every other key includes only itself.
# ---------------------------------------------------------------------------
def test_exactly_six_aliases_expand_beyond_themselves():
    """What makes a six-entry table the right size, measured on upstream.

    If upstream adds an alias key -- or promotes an existing key to one -- this
    reddens and names it, which is the whole reason the expansion lists are
    allowed to be lists. `ADInplaceOrView` is in the assertion's negative half
    on purpose: it reads like an alias and upstream expands it to itself alone.
    """
    if not _available():
        return _skip("expanding aliases")
    up = _cross(False)
    expanding = sorted(
        b for b in up["names"]
        if sum(1 for a in up["names"] if up["rows"].get(a + "|" + b)) > 2
    )
    assert expanding == sorted(_EXPANDING), (
        f"upstream's expanding aliases changed: {expanding}\n"
        f"expected: {sorted(_EXPANDING)}\n"
        "Re-derive the expansion lists in bootstrap.py::_ALIAS_EXPANSION."
    )
    assert "ADInplaceOrView" not in expanding
    sizes = {b: sum(1 for a in up["names"] if up["rows"].get(a + "|" + b)) for b in expanding}
    assert sizes == {
        "Autograd": 27,
        "CompositeExplicitAutograd": 82,
        "CompositeExplicitAutogradNonFunctional": 57,
        "CompositeImplicitAutograd": 122,
        "CompositeImplicitAutogradNestedTensor": 26,
        "FuncTorchBatchedDecomposition": 7,
    }, f"upstream expansion sizes moved: {sizes}"


def test_the_six_value_alias_pairs_the_name_rule_cannot_see_are_all_absent_here():
    """The rule is name-based; upstream's enum has six same-value name pairs.

    `EndOfDenseBackends` *is* `Meta` numerically, so upstream answers `True`
    for both orders of that pair while a name-equality rule answers `False`.
    Those 12 pairs are the rule's entire error, and none of the six offending
    names exists in the vendored enum -- so the error is unreachable here.
    This test is what says so, rather than the docstring saying so.
    """
    if not _available():
        return _skip("value aliases")
    up, shim = _cross(False), _cross(True)
    names = up["names"]
    expansion = {b: {a for a in names if up["rows"].get(a + "|" + b)} for b in _EXPANDING}
    misses = [
        (a, b) for a in names for b in names
        if up["rows"].get(a + "|" + b) is True
        and not (a != "Undefined" and (a == b or a in expansion.get(b, ())))
    ]
    assert len(misses) == 12, f"the name rule's error is no longer 12 pairs: {misses}"
    # A pair is unaskable here as soon as *either* member is missing from the
    # vendored enum -- `Meta` is present, `EndOfDenseBackends` is not, so the
    # pair cannot be formed. Asserting on the union of names instead would fail
    # on `Meta` and prove nothing; this is the invariant that actually holds.
    askable = [
        (a, b) for a, b in misses
        if a in set(shim["names"]) and b in set(shim["names"])
    ]
    assert not askable, (
        f"value-alias pairs now formable in the vendored enum: {askable}. "
        "The name-equality rule answers False for them and upstream answers True; "
        "bootstrap.py must compare enum values, not names."
    )
    absent = sorted({b for a, b in misses} - set(shim["names"]))
    assert absent == sorted([
        "EndOfAutogradFunctionalityBackends", "EndOfDenseBackends",
        "EndOfNestedTensorBackends", "EndOfQuantizedBackends",
        "EndOfSparseBackends", "EndOfSparseCsrBackends",
    ]), f"the absent half of the value-alias pairs changed: {absent}"


# ---------------------------------------------------------------------------
# 4. Staleness: the partition the agreement above is valid over.
# ---------------------------------------------------------------------------
def test_the_enum_divergence_that_bounds_what_agreement_can_mean():
    """123 shared, 17 stub-only, 22 runtime-only -- and the stub is the stale one.

    `test_the_shim_agrees_..._cross_product` can only compare names both enums
    have. Without this test a vendor bump that dropped 40 keys from the stub
    would make that test compare less and still pass green. Here the partition
    itself is the assertion, so the coverage shrinking is the failure.

    The stub-only names are not this shim's invention: `surface.json` mirrors
    `torch/_C/__init__.pyi` exactly, and it is upstream's own stub that still
    declares `Vulkan`, `MKLDNN`, `Named` and `Tracer` after its C++ enum
    dropped them.
    """
    if not _available():
        return _skip("enum divergence")
    up, shim = set(_names(False)), set(_names(True))
    shared, stub_only, runtime_only = up & shim, shim - up, up - shim
    assert (len(shared), len(stub_only), len(runtime_only)) == (123, 17, 22), (
        f"vendored/upstream DispatchKey partition moved to "
        f"({len(shared)}, {len(stub_only)}, {len(runtime_only)}).\n"
        f"stub-only:    {sorted(stub_only)}\n"
        f"runtime-only: {sorted(runtime_only)}\n"
        "Re-derive _ALIAS_EXPANSION in bootstrap.py against the new upstream and "
        "re-check docs/graph/ALIASINC.md 2."
    )
    # The stub-only 17 are answered `False` by every alias, because upstream
    # cannot be asked about them. That is the conservative direction: it never
    # claims a kernel exists. Asserted so a future round cannot guess instead.
    cross = _cross(True)
    for a in sorted(stub_only):
        for b in _EXPANDING:
            assert cross["rows"][a + "|" + b] is False, (
                f"{a} is not in upstream's enum, so {b} must not claim to include it"
            )


# ---------------------------------------------------------------------------
# 5. What closing the gap actually changes -- and what it does not.
# ---------------------------------------------------------------------------
def test_closing_the_gap_lets_resolve_key_answer_where_it_used_to_raise():
    """METAKEY 2.2's measurement, re-run: the `NotImplementedError` is gone.

    That document recorded `resolve_key(aten.as_strided.default, Meta)` raising
    `NotImplementedError: _dispatch_is_included_in_alias` here while upstream
    returned `Meta`. This asserts the predicate is no longer what stops it.
    It deliberately does not assert the *same key* comes back: the Meta
    predicate is still a constant `False` by METAKEY 1's decision, so the shim
    reaches a different branch, and pinning the returned key here would be
    pinning that unrelated decision.
    """
    if not _available():
        return _skip("resolve_key")
    shim = _run(_RESOLVE, True)["resolved"]
    blamed = {k: v for k, v in shim.items()
              if "_dispatch_is_included_in_alias" in v}
    assert not blamed, f"resolve_key still dies on the predicate: {blamed}"
    print("ALIASINC resolve_key(_, Meta):", json.dumps(shim, sort_keys=True))



# --- resolve_key over the whole aten surface, three keys ------------------
_SWEEP = _PREAMBLE + """
from torch._ops import resolve_key
BLIND = %r
if BLIND:
    def _raise(*a, **k):
        raise NotImplementedError(
            "not implemented in torch._C shim: torch._C._dispatch_is_included_in_alias")
    torch._C._dispatch_is_included_in_alias = _raise
    import torch._ops as _ops
    _ops.is_included_in_alias = _raise
out = {}
for opname in dir(torch.ops.aten):
    if opname.startswith("__"):
        continue
    try:
        packet = getattr(torch.ops.aten, opname)
        overloads = packet.overloads()
    except Exception:
        continue
    for ov in overloads:
        try:
            op = getattr(packet, ov)
        except Exception:
            continue
        for kn in ("Meta", "CPU", "AutogradCPU"):
            try:
                r = getattr(resolve_key(op, getattr(DispatchKey, kn)), "name", "?")
            except Exception as exc:
                blames = "is_included_in_alias" in str(exc)
                r = "RAISED:ALIAS" if blames else "RAISED:" + str(exc)[-60:]
            out[opname + "." + ov + "|" + kn] = r
print(json.dumps({"is_shim": IS_SHIM, "resolved": out}))
"""

_SWEEP_LIVE = _SWEEP % (False,)
_SWEEP_BLIND = _SWEEP % (True,)


def test_no_resolve_key_result_on_the_whole_aten_surface_still_blames_this_predicate():
    """The one durable guarantee this round adds, over 4893 results not six.

    `resolve_key(op, k)` for every aten overload against `Meta`, `CPU` and
    `AutogradCPU`. Before this round **3547 of 4893 died on
    `_dispatch_is_included_in_alias`**; the assertion is that none does now.

    The same sweep with the predicate monkeypatched back to raising is run as
    the control, so this test measures its own premise instead of trusting the
    history: if the sweep stopped reaching the predicate at all, the control
    would also show zero and that is asserted against.

    What is *printed and not asserted* is where those 3547 go now
    (`_dispatch_get_backend_keyset_from_autograd`, the next unimplemented
    name) and how many resolve. Asserting those would be asserting the next
    gap stays open -- a test against progress, `docs/graph/METAKEY.md` 5.
    """
    if not _available():
        return _skip("resolve_key sweep")
    live = _run(_SWEEP_LIVE, True)["resolved"]
    blind = _run(_SWEEP_BLIND, True)["resolved"]
    assert len(live) > 1000, f"sweep only produced {len(live)} results"

    control = [k for k, v in blind.items() if v == "RAISED:ALIAS"]
    assert control, (
        "the control sweep never reached the predicate, so this test could not "
        "have failed -- the sweep no longer exercises resolve_key's alias branches"
    )

    blamed = [k for k, v in live.items() if v == "RAISED:ALIAS"]
    assert not blamed, (
        f"{len(blamed)} of {len(live)} resolve_key results still die on "
        f"_dispatch_is_included_in_alias; first 10: {blamed[:10]}"
    )

    import collections
    after = collections.Counter(
        v if not v.startswith("RAISED:") else "RAISED:" + v[-45:] for v in live.values()
    )
    print(f"ALIASINC sweep: {len(live)} results, {len(control)} used to blame this "
          f"predicate, 0 do now. Where they land instead: {after.most_common(4)}")

def test_the_export_that_motivated_this_still_makes_zero_alias_queries():
    """Printed, not asserted -- and the honest answer is that nothing moved.

    METAKEY 2.2 measured that the shim makes zero `Meta` queries during this
    export because nothing here enters `_get_dispatch`, and both exports
    succeed anyway. The same is true of this predicate. The count is printed so
    a later round can see it move; asserting it would be a test against
    progress, exactly as METAKEY 5 argued.
    """
    if not _available():
        return _skip("export queries")
    shim = _run(_EXPORT_QUERIES, True)
    print(f"ALIASINC export: exported={shim['exported']} "
          f"alias_queries={shim['queries']} sites={shim['sites']}")


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failed else 0)
