"""`_dispatch_get_backend_keyset_from_autograd` -- the name `resolve_key` dies on now.

`docs/graph/ALIASINC.md` §4 closed the previous gap and measured what it bought:
of 4893 `resolve_key(op, key)` results over the aten surface, **3547 stopped
dying on `_dispatch_is_included_in_alias` and started dying on this name, and
zero newly resolved.** This file is the next name on that list, and it answers
the same question about itself first.

**The headline is that implementing this one alone also unblocks nothing**, and
that is measured, not inferred: `test_the_staged_sweep_shows_this_name_alone_unblocks_nothing`
patches each remaining name in turn and counts. The 3547 do not resolve; they
*split*, 2719 onto `_dispatch_is_alias_key` and 828 onto
`_dispatch_has_backend_fallback`.

What is new, and is worth more than the implementation, is that **the chain
terminates and it is exactly three more names.** With all of them answered from
a live upstream, every one of the 4893 either resolves or raises upstream's own
`could not find kernel` -- no result dies on an unimplemented name. That is the
map `docs/graph/BKEYSET.md` §3 records, and it is what says this line of work
ends rather than continuing indefinitely.

Every upstream number here is recomputed in a **separate subprocess** on every
run. There is no expected-value table written beside the implementation.

Three measured facts shape the implementation, and none is guessable:

* **Only 16 of upstream's 145 keys answer a non-empty keyset.** Every other key
  -- including every non-autograd key and `Autograd` itself -- answers the empty
  set. So this is a 14-entry table (two of the 16 are keys the vendored enum
  does not have), not a 145-row one.

* **Four `Autograd<Backend>` keys answer empty**: `AutogradHIP`, `AutogradVE`,
  `AutogradMTIA` and `AutogradFunctionality`. The obvious rule -- strip
  `Autograd`, return that backend -- is wrong for exactly those four, because
  upstream's `getBackendKeySetFromAutograd` is a `switch` with explicit cases
  and those are not among them. `test_four_autograd_keys_answer_empty_so_the_name_rule_is_wrong`
  is what says so.

* **`DispatchKeySet.has()` cannot be trusted on the alias keys.** Upstream's
  *empty* keyset answers `has()` `True` for all six alias keys and for
  `Undefined`. Membership is therefore probed off the other 138 only, and
  `test_the_membership_probe_is_only_valid_away_from_the_alias_keys` pins that
  the spurious set is exactly those seven -- so if upstream ever fixes it, this
  reddens and the probe can widen instead of silently continuing to skip them.

A fourth fact is a near-miss worth recording: the first table built for this
round was read out of `repr(keyset)`, which prints **C++ legacy names**
(`Vulkan`, `Metal`, `MkldnnCPU`) that are not in upstream's Python enum at all.
`docs/graph/ALIASINC.md` §2.1 had already measured `Vulkan` as stub-only. Every
number below is therefore built from `__members__` and `has()`, never from a
repr string.
"""

import collections
import json
import os
import subprocess
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "python")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

#: The keys upstream answers a non-empty keyset for. Asserted against a live
#: upstream by `test_only_sixteen_keys_answer_a_nonempty_backend_keyset`, so a
#: new one reddens rather than being silently answered empty.
_NONEMPTY = (
    "AutogradCPU", "AutogradCUDA", "AutogradHPU", "AutogradIPU", "AutogradLazy",
    "AutogradMAIA", "AutogradMPS", "AutogradMeta", "AutogradNestedTensor",
    "AutogradOther", "AutogradPrivateUse1", "AutogradPrivateUse2",
    "AutogradPrivateUse3", "AutogradXLA", "AutogradXPU",
    "EndOfAutogradFunctionalityBackends",
)

#: `Autograd<X>` keys upstream maps to the empty set anyway. The whole point of
#: not deriving this table from the names.
_AUTOGRAD_KEYS_THAT_ANSWER_EMPTY = (
    "Autograd", "AutogradFunctionality", "AutogradHIP", "AutogradMTIA", "AutogradVE",
)

#: The remaining unimplemented names on `resolve_key`'s path, in the order the
#: staged sweep patches them. `test_the_chain_terminates_and_no_fourth_name_appears`
#: asserts nothing outside this list is ever blamed.
_REMAINING_CHAIN = (
    "_dispatch_get_backend_keyset_from_autograd",
    "_dispatch_is_alias_key",
    "_dispatch_autogradother_backends",
    "_dispatch_has_backend_fallback",
)


def _available():
    return os.path.isfile(_VENDOR_SHIM)


def _skip(name):
    print(f"SKIP {name}: no vendored shim at {_VENDOR_SHIM}")


def _run(script, shim, argv=(), timeout=1800):
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", script, *[str(a) for a in argv]],
        capture_output=True, text=True, env=env, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    # Which side the probe landed on is asserted, not assumed: an empty
    # vendored tree makes the "shim" side silently import upstream and every
    # comparison below would then be upstream against itself.
    assert out["is_shim"] is shim, f"probe meant for shim={shim} ran on the other side"
    return out


_PREAMBLE = """
import json, sys, warnings, collections
warnings.filterwarnings("ignore")
import torch
C = torch._C
IS_SHIM = hasattr(C, "_aten_implemented")
DK = C.DispatchKey
NAMES = sorted(DK.__members__) if hasattr(DK, "__members__") \\
    else sorted(k.name for k in DK)
"""


# --- the whole table, by enum membership rather than by repr ---------------
_TABLE = _PREAMBLE + """
try:
    ALIAS = [n for n in NAMES if C._dispatch_is_alias_key(getattr(DK, n))]
except Exception:
    # The shim has no `_dispatch_is_alias_key` yet; the six are stable and
    # `spurious` below is measured separately, so fall back to it.
    ALIAS = []
rows, errs = {}, {}
for n in NAMES:
    try:
        ks = C._dispatch_get_backend_keyset_from_autograd(getattr(DK, n))
    except Exception as exc:
        errs[n] = type(exc).__name__ + ":" + str(exc).splitlines()[0][:140]
        continue
    mem, bad = [], []
    for m in NAMES:
        try:
            if ks.has(getattr(DK, m)):
                mem.append(m)
        except Exception:
            bad.append(m)
    rows[n] = {"members": sorted(mem), "unprobeable": sorted(bad)}
spurious = sorted(rows.get("CPU", {}).get("members", []))
print(json.dumps({"is_shim": IS_SHIM, "names": NAMES, "alias_keys": sorted(ALIAS),
                  "rows": rows, "errs": errs, "spurious_on_empty": spurious}))
"""


# --- resolve_key over the whole aten surface, with the chain staged --------
_SWEEP = _PREAMBLE + """
STAGE = int(sys.argv[1])
KS = C.DispatchKeySet

def _mk(names):
    return KS._of([getattr(DK, n) for n in names if n in NAMES])

# The staged patches are built from the *upstream* answers handed in through
# argv[2] as JSON, never from anything written in this file.
HANDED = json.loads(sys.argv[2])
if STAGE < 0:
    # The control: this name put back to what an `_Unimplemented` does, so the
    # sweep's "before" is measured on this tree rather than quoted from
    # `docs/graph/ALIASINC.md`.
    def _raise(*a, **kw):
        raise NotImplementedError(
            "not implemented in torch._C shim: "
            "torch._C._dispatch_get_backend_keyset_from_autograd")
    C._dispatch_get_backend_keyset_from_autograd = _raise
if STAGE >= 1:
    BK = {n: _mk(HANDED["backend_keyset"].get(n, [])) for n in NAMES}
    C._dispatch_get_backend_keyset_from_autograd = lambda k: BK[getattr(k, "name", k)]
if STAGE >= 2:
    AL = set(HANDED["alias_keys"])
    C._dispatch_is_alias_key = lambda k: getattr(k, "name", k) in AL
if STAGE >= 3:
    C._dispatch_autogradother_backends = _mk(HANDED["autogradother"])
if STAGE >= 4:
    FB = set(HANDED["backend_fallback"])
    C._dispatch_has_backend_fallback = lambda k: getattr(k, "name", k) in FB

from torch._ops import resolve_key
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
                r = getattr(resolve_key(op, getattr(DK, kn)), "name", "?")
            except NotImplementedError as exc:
                s = str(exc)
                if "not implemented in torch._C shim" in s or "_Unimplemented" in s:
                    r = "GAP:" + s.rsplit(".", 1)[-1].strip()
                elif "could not find kernel" in s:
                    r = "NOKERNEL"
                else:
                    r = "RAISED:" + s[-60:]
            except Exception as exc:
                r = "RAISED:" + type(exc).__name__ + ":" + str(exc)[-60:]
            out[opname + "." + ov + "|" + kn] = r
print(json.dumps({"is_shim": IS_SHIM, "stage": STAGE, "n": len(out),
                  "counts": dict(collections.Counter(out.values())),
                  "resolved": sorted(k for k, v in out.items()
                                     if not v.startswith(("GAP:", "RAISED:", "NOKERNEL")))}))
"""


# --- the neighbours the staged sweep needs, read off a live upstream -------
_NEIGHBOURS = _PREAMBLE + """
ALIAS = [n for n in NAMES if C._dispatch_is_alias_key(getattr(DK, n))]
PROBE = [n for n in NAMES if n not in ALIAS and n != "Undefined"]
def mem(ks):
    return sorted(m for m in PROBE if ks.has(getattr(DK, m)))
print(json.dumps({
    "is_shim": IS_SHIM,
    "backend_keyset": {n: mem(C._dispatch_get_backend_keyset_from_autograd(getattr(DK, n)))
                       for n in NAMES},
    "alias_keys": sorted(ALIAS),
    "autogradother": mem(C._dispatch_autogradother_backends),
    "backend_fallback": sorted(n for n in NAMES
                               if C._dispatch_has_backend_fallback(getattr(DK, n))),
}))
"""


_CACHE = {}


def _table(shim):
    if ("table", shim) not in _CACHE:
        _CACHE[("table", shim)] = _run(_TABLE, shim)
    return _CACHE[("table", shim)]


def _neighbours():
    if "neigh" not in _CACHE:
        _CACHE["neigh"] = _run(_NEIGHBOURS, False)
    return _CACHE["neigh"]


def _sweep(stage):
    if ("sweep", stage) not in _CACHE:
        handed = json.dumps(_neighbours())
        _CACHE[("sweep", stage)] = _run(_SWEEP, True, argv=(stage, handed))
    return _CACHE[("sweep", stage)]


# ---------------------------------------------------------------------------
# 1. The bar: element-wise agreement with a live upstream.
# ---------------------------------------------------------------------------
def test_the_shim_agrees_with_upstream_on_the_backend_keyset_for_every_shared_key():
    """Every shared `DispatchKey`, member by member, both sides, separate procs.

    Not "returns a keyset without raising". For each of the 123 keys both enums
    have, the *set of members* is compared name by name over the same 123. A
    table that got `AutogradNestedTensor` wrong by one backend reddens here
    with that backend named.

    Members upstream reports that the vendored enum cannot spell (the
    `StartOf*`/`EndOf*` sentinels, `NestedTensorMAIA`) are outside the
    comparison for the same reason `docs/graph/ALIASINC.md` §2.1 gives: they
    cannot be passed in, so there is nothing to agree about. The partition that
    bounds this is asserted separately, so it cannot quietly grow.
    """
    if not _available():
        return _skip("backend keyset agreement")
    up, shim = _table(False), _table(True)
    shared = sorted(set(up["names"]) & set(shim["names"]))
    assert len(shared) > 100, f"only {len(shared)} shared keys -- vendored enum is wrong"
    assert not shim["errs"], (
        f"the shim raised for {len(shim['errs'])} of {len(shim['names'])} keys; "
        f"first three: {dict(sorted(shim['errs'].items())[:3])}")
    assert not up["errs"], (
        f"upstream raised for {len(up['errs'])} keys: {dict(sorted(up['errs'].items())[:3])}")

    # Upstream's `has()` answers `True` for the six alias keys and `Undefined`
    # on *every* keyset, including the empty one -- measured by
    # `test_the_membership_probe_is_only_valid_away_from_the_alias_keys`. The
    # shim's `has()` does not do that, and must not be taught to: it is a bug
    # in a bit-set representation, not a membership. So those seven are outside
    # the comparison on both sides. Leaving them in made all 123 keys disagree.
    untrustworthy = set(up["alias_keys"]) | {"Undefined"}
    comparable = set(shared) - untrustworthy
    disagree = []
    for k in shared:
        u = set(up["rows"][k]["members"]) & comparable
        s = set(shim["rows"][k]["members"]) & comparable
        if u != s:
            disagree.append(
                f"{k}: upstream-only={sorted(u - s)} shim-only={sorted(s - u)}")
    assert not disagree, (
        f"{len(disagree)} of {len(shared)} keys disagree with upstream;\n"
        + "\n".join(disagree[:20])
    )
    compared = sum(len(set(up["rows"][k]["members"]) & comparable) for k in shared)
    print(f"BKEYSET agreement: {len(shared)} keys x {len(comparable)} comparable "
          f"members = {len(shared) * len(comparable)} membership questions, "
          f"{compared} true memberships, 0 disagreements")


# ---------------------------------------------------------------------------
# 2. The shape, which is what makes a 14-entry table the right size.
# ---------------------------------------------------------------------------
def test_only_sixteen_keys_answer_a_nonempty_backend_keyset():
    """129 of upstream's 145 keys answer the empty set. Measured on upstream.

    If upstream adds an autograd backend -- or starts answering for one of the
    four in the next test -- this reddens and names it, which is the whole
    reason the table below is allowed to be a table.
    """
    if not _available():
        return _skip("nonempty shape")
    up = _table(False)
    alias = set(up["alias_keys"])
    real = {
        k: sorted(set(v["members"]) - alias - {"Undefined"})
        for k, v in up["rows"].items()
    }
    nonempty = sorted(k for k, v in real.items() if v)
    assert nonempty == sorted(_NONEMPTY), (
        f"upstream's non-empty backend keysets changed to {nonempty}\n"
        f"expected {sorted(_NONEMPTY)}\n"
        "Re-derive _AUTOGRAD_BACKEND_KEYSET in bootstrap.py."
    )
    assert len(up["rows"]) - len(nonempty) == 129, (
        f"{len(up['rows']) - len(nonempty)} keys answer empty, expected 129")
    # These are upstream's own member counts, so they include the
    # `StartOf*`/`EndOf*` sentinels that share a value with a real key
    # (`AutogradCPU` -> `CPU`, `Dense`, `StartOfDenseBackends`). The vendored
    # enum cannot spell those, which is why the table in bootstrap.py is
    # smaller than these numbers; the partition test below bounds that.
    sizes = {k: len(real[k]) for k in nonempty}
    assert sizes["AutogradCPU"] == 3 and sizes["AutogradNestedTensor"] == 19 \
        and sizes["AutogradOther"] == 57, f"upstream keyset sizes moved: {sizes}"


def test_four_autograd_keys_answer_empty_so_the_name_rule_is_wrong():
    """The rule "strip `Autograd`, return that backend" is wrong for four keys.

    `AutogradHIP`, `AutogradVE`, `AutogradMTIA` and `AutogradFunctionality` all
    exist in upstream's enum and all answer the **empty** set, because
    `getBackendKeySetFromAutograd` is a `switch` with explicit cases and those
    are not among them. A derived implementation would have invented three
    backend memberships; this asserts the exception rather than trusting it.

    If upstream ever adds those cases, this reddens -- and at that point the
    derived rule becomes correct and the table should go.
    """
    if not _available():
        return _skip("name rule")
    up = _table(False)
    alias = set(up["alias_keys"])
    empty_autograds = sorted(
        k for k in up["rows"]
        if k.startswith("Autograd")
        and not (set(up["rows"][k]["members"]) - alias - {"Undefined"})
    )
    assert empty_autograds == sorted(_AUTOGRAD_KEYS_THAT_ANSWER_EMPTY), (
        f"upstream's empty-answering Autograd keys changed: {empty_autograds}")
    # And the shim answers empty for them too -- the direction that never
    # invents a backend. The shim has to *answer*: if it raises, that is this
    # test failing, not this test skipping a half.
    shim = _table(True)
    assert not shim["errs"], (
        f"the shim raised for {len(shim['errs'])} keys instead of answering; "
        f"first three: {dict(sorted(shim['errs'].items())[:3])}")
    for k in empty_autograds:
        if k not in shim["names"]:
            continue
        real = set(shim["rows"][k]["members"]) - alias - {"Undefined"}
        assert not real, f"the shim invented a backend keyset for {k}: {sorted(real)}"


def test_the_autogradother_keyset_is_exactly_the_dispatch_autogradother_backends_value():
    """Two upstream surfaces that must agree, cross-checked on upstream itself.

    `resolve_key` consults `_dispatch_autogradother_backends` (the value) a few
    lines after `_dispatch_get_backend_keyset_from_autograd(AutogradOther)`
    (the function). Upstream builds both from the same `c10` set. Asserting the
    identity on a live upstream means the 57-member entry in the table is
    corroborated by a second upstream surface rather than by one probe.
    """
    if not _available():
        return _skip("autogradother identity")
    neigh = _neighbours()
    a = set(neigh["backend_keyset"]["AutogradOther"])
    b = set(neigh["autogradother"])
    assert a == b, (
        f"upstream's two AutogradOther surfaces diverged: "
        f"function-only={sorted(a - b)} value-only={sorted(b - a)}")
    assert len(a) == 57, f"AutogradOther's backend set is now {len(a)} keys"


# ---------------------------------------------------------------------------
# 3. The two things that would make the agreement test quietly compare less.
# ---------------------------------------------------------------------------
def test_the_enum_divergence_that_bounds_what_this_agreement_can_mean():
    """123 shared / 17 stub-only / 22 runtime-only, and the stub is the stale one.

    Same guard `docs/graph/ALIASINC.md` §2.1 installed, restated for this
    function because it bounds *this* comparison too. Without it a vendor bump
    that dropped forty names from the stub would make the agreement test
    compare less and still report green.

    The 17 stub-only keys must answer the empty set: upstream cannot be asked
    about them, and empty is the direction that never claims a backend.
    """
    if not _available():
        return _skip("enum divergence")
    up, shim = _table(False), _table(True)
    U, S = set(up["names"]), set(shim["names"])
    shared, stub_only, runtime_only = U & S, S - U, U - S
    assert (len(shared), len(stub_only), len(runtime_only)) == (123, 17, 22), (
        f"vendored/upstream DispatchKey partition moved to "
        f"({len(shared)}, {len(stub_only)}, {len(runtime_only)}).\n"
        f"stub-only:    {sorted(stub_only)}\n"
        f"runtime-only: {sorted(runtime_only)}\n"
        "Re-derive _AUTOGRAD_BACKEND_KEYSET in bootstrap.py and re-check "
        "docs/graph/BKEYSET.md §2."
    )
    alias = set(up["alias_keys"])
    assert not shim["errs"], (
        f"the shim raised for {len(shim['errs'])} keys instead of answering; "
        f"first three: {dict(sorted(shim['errs'].items())[:3])}")
    for k in sorted(stub_only):
        real = set(shim["rows"][k]["members"]) - alias - {"Undefined"}
        assert not real, (
            f"{k} is not in upstream's enum, so the shim must not claim it maps "
            f"to a backend keyset; it answered {sorted(real)}")
    # Two of the 16 non-empty keys are runtime-only, so the table has 14
    # reachable entries. Stated here so that a bump which introduces them is a
    # RED test rather than two answers that quietly go missing.
    unreachable = sorted(set(_NONEMPTY) & runtime_only)
    assert unreachable == ["AutogradMAIA", "EndOfAutogradFunctionalityBackends"], (
        f"the unreachable half of the non-empty keys changed: {unreachable}")


def test_the_membership_probe_is_only_valid_away_from_the_alias_keys():
    """`has()` on an *empty* upstream keyset answers True seven times.

    Every membership number in this file is read with `DispatchKeySet.has()`,
    and upstream's own empty keyset reports `True` for all six alias keys and
    for `Undefined`. So the probe excludes those seven -- and this asserts the
    spurious set is exactly those seven, on both sides, so that the exclusion
    is a measured bound rather than a habit. If upstream fixes `has()`, this
    reddens and the probe can widen.

    This is also why nothing here is read out of `repr(keyset)`: that prints
    C++ legacy names (`Vulkan`, `Metal`, `MkldnnCPU`) which are not in
    upstream's Python enum at all -- `docs/graph/ALIASINC.md` §2.1 measured
    `Vulkan` as stub-only, and a repr-derived table would have contradicted it.
    """
    if not _available():
        return _skip("membership probe validity")
    up = _table(False)
    # `CPU` is one of the 129 keys whose real answer is the empty set.
    assert sorted(up["spurious_on_empty"]) == sorted(
        list(up["alias_keys"]) + ["Undefined"]), (
        f"upstream's empty keyset now answers has()=True for "
        f"{up['spurious_on_empty']}, not the six aliases plus Undefined")
    assert len(up["alias_keys"]) == 6, f"upstream has {len(up['alias_keys'])} alias keys"
    shim = _table(True)
    assert sorted(shim["spurious_on_empty"]) == [], (
        f"the shim's empty keyset claims members: {shim['spurious_on_empty']}. "
        "That is a different bug from upstream's and must not be copied.")


# ---------------------------------------------------------------------------
# 4. What closing it does -- and the map of what is left.
# ---------------------------------------------------------------------------
def test_no_resolve_key_result_on_the_whole_aten_surface_still_blames_this_name():
    """The durable guarantee, over 4893 results.

    `resolve_key(op, k)` for every aten overload against `Meta`, `CPU` and
    `AutogradCPU`. `docs/graph/ALIASINC.md` §4 measured 3547 of 4893 dying on
    this name; the assertion is that none does now.

    The control is the same sweep with this name patched back to raising what
    an `_Unimplemented` raises. Without it a green result would be worthless:
    if the sweep stopped reaching the name at all, "nothing blames it" is
    also what that looks like.
    """
    if not _available():
        return _skip("resolve_key sweep")
    live = _sweep(0)
    assert live["n"] > 1000, f"sweep only produced {live['n']} results"
    blamed = live["counts"].get("GAP:_dispatch_get_backend_keyset_from_autograd", 0)
    assert blamed == 0, f"{blamed} of {live['n']} resolve_key results still die on this name"

    control = _sweep(-1)
    reached = control["counts"].get("GAP:_dispatch_get_backend_keyset_from_autograd", 0)
    assert reached > 1000, (
        f"the control sweep blamed this name only {reached} times, so the test "
        "could not have failed -- resolve_key no longer reaches it")
    print(f"BKEYSET sweep: {live['n']} results; the unpatched shim blames this "
          f"name {reached} times, the implementation 0")


def test_the_staged_sweep_shows_this_name_alone_unblocks_nothing():
    """The honest headline, and it is a negative one.

    The 3547 that reach this name do **not** resolve once it answers. They
    split onto the next two names. This asserts the count of *resolved*
    results does not shrink and prints what it is -- asserting that it does not
    *grow* would be a test against progress (`docs/graph/METAKEY.md` §5), so
    the guarantee is one-directional and the number is printed beside it.
    """
    if not _available():
        return _skip("staged unblock")
    before, after = _sweep(-1), _sweep(0)
    b = set(before["resolved"])
    a = set(after["resolved"])
    assert not (b - a), (
        f"{len(b - a)} results resolved before this name was answered and no "
        f"longer do -- a regression: {sorted(b - a)[:10]}")
    newly = sorted(a - b)
    print(f"BKEYSET unblock: {len(b)} of {before['n']} resolved before, "
          f"{len(a)} after; newly resolved = {len(newly)}")
    print(f"BKEYSET landing: {json.dumps(after['counts'], sort_keys=True)}")


def test_the_chain_terminates_and_no_fourth_name_appears():
    """The map: three more names and `resolve_key` stops dying on gaps entirely.

    Each stage patches one more of `_REMAINING_CHAIN` from a live upstream and
    re-sweeps. Two things are asserted. **Every gap blamed at every stage is
    one of the four names** -- so a fifth unimplemented name hiding behind
    these would redden here rather than being discovered a round later. And
    **at the last stage no result dies on a gap at all**: they resolve or they
    raise upstream's own `could not find kernel`.

    That second assertion is what says this line of work terminates. It is
    measured with upstream's answers patched in, so it is a claim about the
    *shape* of the remaining work, not a claim that the shim does it.
    """
    if not _available():
        return _skip("chain map")
    allowed = {"GAP:" + n for n in _REMAINING_CHAIN}
    table = []
    for stage in range(-1, len(_REMAINING_CHAIN) + 1):
        counts = _sweep(stage)["counts"]
        gaps = {k: v for k, v in counts.items() if k.startswith("GAP:")}
        unexpected = sorted(set(gaps) - allowed)
        assert not unexpected, (
            f"stage {stage} blamed a name outside the known chain: {unexpected}. "
            "The remaining chain in docs/graph/BKEYSET.md §3 is incomplete.")
        raised = {k: v for k, v in counts.items() if k.startswith("RAISED:")}
        assert not raised, f"stage {stage} produced unexpected exceptions: {raised}"
        table.append((stage, len(_sweep(stage)["resolved"]), gaps))
    last = table[-1][2]
    assert not last, (
        f"with all {len(_REMAINING_CHAIN)} names answered, {sum(last.values())} "
        f"results still die on a gap: {last}. The chain does not terminate here.")
    for stage, resolved, gaps in table:
        print(f"BKEYSET chain stage {stage}: resolved={resolved} gaps={gaps}")


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
