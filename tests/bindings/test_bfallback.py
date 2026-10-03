"""`_dispatch_is_alias_key` and `_dispatch_has_backend_fallback` -- the last two names.

`docs/graph/BKEYSET.md` §3 mapped the remaining chain and said it terminates in
exactly these two (plus `_dispatch_autogradother_backends`, which that sweep
never consults). This file is both of them, and the second one is not a lookup.

**The headline is that the honest answer lands at 1440, not 3301.** BKEYSET's
stage 4 measured 3301 with *upstream's* fallback set patched in. That set is the
37 dispatch keys upstream's C++ build registered fallback kernels at. **This
shim has registered none**, and answering `True` for them would hand back the
key itself for 1861 more `(op, key)` pairs on the promise that a fallback will
catch the call -- 1861 kernels that do not exist. That is the direction
`_dispatch_registrations` already refuses by name rather than commit.

So the numbers this round reports are:

    resolved, before                              1346
    resolved, + `_dispatch_is_alias_key`          1440
    resolved, + the honest fallback answer        1440   (unchanged)
    resolved, + *upstream's* fallback set         3301   (1861 false claims)

and the thing that did move is what the other 3453 do: they stop dying on an
unimplemented name and start raising upstream's own `could not find kernel`.
**No `resolve_key` result on the aten surface dies on a gap in this shim any
more.** That is what BKEYSET meant by the chain terminating, arrived at
honestly rather than by copying.

The cost is counted rather than waved at: `test_the_capability_gaps_this_honest_answer_names`
lists the 32 spellable dispatch keys where upstream answers `True` and this
shim answers `False`. Each is a real missing fallback kernel, and the list is
worth as much as the function.

Every upstream number here is recomputed in a **separate subprocess** on every
run. There is no expected-value table written beside the implementation.
"""

import json
import os
import subprocess
import sys


_REPO_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "python")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")


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
    # vendored tree makes the "shim" side silently import upstream, and every
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


# --- both predicates, key by key, plus the shim's own fallback registry ----
_TABLE = _PREAMBLE + """
alias, alias_err, fb, fb_err = {}, {}, {}, {}
for n in NAMES:
    try:
        alias[n] = bool(C._dispatch_is_alias_key(getattr(DK, n)))
    except Exception as exc:
        alias_err[n] = type(exc).__name__ + ":" + str(exc).splitlines()[0][:140]
    try:
        fb[n] = bool(C._dispatch_has_backend_fallback(getattr(DK, n)))
    except Exception as exc:
        fb_err[n] = type(exc).__name__ + ":" + str(exc).splitlines()[0][:140]

# Which keys expand beyond themselves under `_dispatch_is_included_in_alias`.
# The identity between this and `alias` is what lets the implementation derive
# the predicate from `_ALIAS_EXPANSION` instead of adding a second table.
# Compared by enum *value*, not by name. Upstream spells six keys twice
# (`EndOfDenseBackends` **is** `Meta`), and a name-inequality test reports all
# twelve of those names as expanding -- which is `docs/graph/ALIASINC.md` §3's
# finding arriving from the other direction, and would make this identity look
# false for a reason that has nothing to do with alias keys.
expanding = []
for a in NAMES:
    av = getattr(DK, a)
    for k in NAMES:
        kv = getattr(DK, k)
        if kv == av:
            continue
        try:
            if C._dispatch_is_included_in_alias(kv, av):
                expanding.append(a)
                break
        except Exception:
            continue

print(json.dumps({
    "is_shim": IS_SHIM, "names": NAMES,
    "alias_true": sorted(k for k, v in alias.items() if v), "alias_err": alias_err,
    "fallback_true": sorted(k for k, v in fb.items() if v), "fb_err": fb_err,
    "expanding_aliases": sorted(set(expanding)),
    "shim_backend_fallbacks": sorted(getattr(C, "_shim_backend_fallbacks", lambda: [])()),
    "recorded_fallbacks": [list(map(str, r))
                           for r in getattr(C, "_shim_registrations", [])
                           if r and r[0] == "fallback"],
}))
"""


# --- a fallback pushed through the only registration door there is --------
_REGISTER = _PREAMBLE + """
before = sorted(C._shim_backend_fallbacks())
lib = C._dispatch_library("IMPL", "aten", "CPU")
lib.fallback("CPU", lambda *a, **kw: None)
after = sorted(C._shim_backend_fallbacks())
recorded = [list(map(str, r)) for r in C._shim_registrations if r and r[0] == "fallback"]
print(json.dumps({
    "is_shim": IS_SHIM, "before": before, "after": after, "recorded": recorded,
    "predicate_after": bool(C._dispatch_has_backend_fallback(DK.CPU)),
}))
"""


# --- resolve_key over the whole aten surface, three ways -------------------
_SWEEP = _PREAMBLE + """
MODE = sys.argv[1]
HANDED = json.loads(sys.argv[2])
if MODE == "upstream_fallbacks":
    # BKEYSET §3 stage 4, reproduced: upstream's registered-fallback set patched
    # in. This is the control the honest answer is measured against, and it is
    # built from a live upstream handed in through argv, never from this file.
    FB = set(HANDED["backend_fallback"])
    C._dispatch_has_backend_fallback = lambda k: getattr(k, "name", k) in FB
elif MODE == "alias_only":
    # The control for "the fallback answer changed nothing": this name put back
    # to what an `_Unimplemented` does, so the before is measured on this tree
    # rather than quoted from `docs/graph/BKEYSET.md`.
    def _raise(*a, **kw):
        raise NotImplementedError(
            "not implemented in torch._C shim: "
            "torch._C._dispatch_has_backend_fallback")
    C._dispatch_has_backend_fallback = _raise
elif MODE != "honest":
    raise SystemExit("unknown mode " + MODE)

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
print(json.dumps({"is_shim": IS_SHIM, "mode": MODE, "n": len(out),
                  "counts": dict(collections.Counter(out.values())),
                  "resolved": sorted(k for k, v in out.items()
                                     if not v.startswith(("GAP:", "RAISED:", "NOKERNEL")))}))
"""


_CACHE = {}


def _table(shim):
    if ("table", shim) not in _CACHE:
        _CACHE[("table", shim)] = _run(_TABLE, shim)
    return _CACHE[("table", shim)]


def _sweep(mode):
    if ("sweep", mode) not in _CACHE:
        handed = json.dumps({"backend_fallback": _table(False)["fallback_true"]})
        _CACHE[("sweep", mode)] = _run(_SWEEP, True, argv=(mode, handed))
    return _CACHE[("sweep", mode)]


# ---------------------------------------------------------------------------
# 1. `_dispatch_is_alias_key`: agreement, and why it needs no table.
# ---------------------------------------------------------------------------
def test_the_alias_key_predicate_agrees_with_upstream_on_every_shared_key():
    """Key by key over the shared enum, both sides, separate subprocesses.

    Not "returns a bool". Every one of the 123 keys both enums spell is asked
    on both sides and the answers are compared one at a time, so a predicate
    that got a single key wrong reddens here with that key named.
    """
    if not _available():
        return _skip("alias key agreement")
    up, shim = _table(False), _table(True)
    shared = sorted(set(up["names"]) & set(shim["names"]))
    assert len(shared) > 100, f"only {len(shared)} shared keys -- vendored enum is wrong"
    assert not shim["alias_err"], (
        f"the shim raised for {len(shim['alias_err'])} keys; "
        f"first three: {dict(sorted(shim['alias_err'].items())[:3])}")
    assert not up["alias_err"], f"upstream raised: {dict(sorted(up['alias_err'].items())[:3])}"
    u, s = set(up["alias_true"]) & set(shared), set(shim["alias_true"]) & set(shared)
    assert u == s, f"upstream-only={sorted(u - s)} shim-only={sorted(s - u)}"
    print(f"ALIASKEY agreement: {len(shared)} keys compared one at a time, "
          f"{len(u)} True, 0 disagreements")


def test_the_alias_keys_are_exactly_the_keys_that_expand_beyond_themselves():
    """Measured on upstream: `is_alias_key` == "expands under `is_included_in_alias`".

    This is the licence for the implementation to answer from the existing
    `_ALIAS_EXPANSION` instead of adding a second table that could rot away
    from the first. `docs/graph/ALIASINC.md` §1.1 measured the six; this
    re-derives the identity from a live upstream every run, so if upstream ever
    promotes a key to an alias without giving it an expansion -- or the other
    way round -- this reddens and prints both sides.
    """
    if not _available():
        return _skip("alias identity")
    up = _table(False)
    alias, expanding = set(up["alias_true"]), set(up["expanding_aliases"])
    assert alias == expanding, (
        f"upstream's two surfaces disagree: alias-only={sorted(alias - expanding)} "
        f"expanding-only={sorted(expanding - alias)}")
    assert len(alias) == 6, f"upstream now has {len(alias)} alias keys: {sorted(alias)}"
    print(f"ALIASKEY identity: {sorted(alias)}")


# ---------------------------------------------------------------------------
# 2. `_dispatch_has_backend_fallback`: the judgement, and its evidence.
# ---------------------------------------------------------------------------
def test_this_shim_has_registered_no_backend_fallback_and_the_predicate_says_so():
    """The evidence for answering `False` everywhere: the registry is empty.

    A backend fallback is a kernel registered at a dispatch key that catches
    *every* operator. The only door a registration can arrive through here is
    `_C._dispatch_library(...)`, and after a full `import torch` **zero**
    `fallback` registrations have come through it. So the honest answer is not
    a choice between upstream's 37 and something smaller -- it is that this
    shim has none, derived from its own registry rather than asserted.

    Both halves are pinned: the registry is empty *and* the predicate answers
    `False` for all 140 keys. A later round that starts answering `True` from a
    table rather than from the registry reddens here.
    """
    if not _available():
        return _skip("empty fallback registry")
    shim = _table(True)
    assert not shim["fb_err"], (
        f"the shim raised for {len(shim['fb_err'])} keys; "
        f"first three: {dict(sorted(shim['fb_err'].items())[:3])}")
    assert shim["shim_backend_fallbacks"] == [], (
        "this shim now has effective backend fallbacks at "
        f"{shim['shim_backend_fallbacks']} -- the doc's count and the gap list "
        "below both have to be redone")
    assert shim["recorded_fallbacks"] == [], (
        "a Python-level fallback is now registered during import "
        f"({shim['recorded_fallbacks']}); decide whether it is effective "
        "before the predicate can keep answering False for its key")
    assert shim["fallback_true"] == [], (
        f"the predicate answers True for {shim['fallback_true']} while the "
        "registry is empty -- that is a claimed kernel")
    print(f"BFALLBACK: {len(shim['names'])} keys, 0 True, registry empty")


def test_a_python_registered_fallback_is_recorded_but_is_not_effective():
    """The asymmetry that keeps the predicate from becoming a lie, pinned.

    `_dispatch_library(...).fallback(...)` is recorded into
    `_shim_registrations` and then **dropped** -- `_aten_dispatch` is the only
    thing that answers a call and it knows nothing about it. So a recorded
    fallback must not make the predicate answer `True`: that would claim a
    kernel that will never run, which is exactly what this round refused to do
    with upstream's 37.

    This is the deliberate part, so it is asserted rather than left to a
    comment. The day a round wires Python fallbacks through to the dispatcher,
    this test reddens and forces both sides to be updated together.
    """
    if not _available():
        return _skip("recorded but not effective")
    out = _run(_REGISTER, True)
    assert out["before"] == [], f"effective set was not empty to start: {out['before']}"
    assert len(out["recorded"]) == 1, (
        f"the registration did not arrive in `_shim_registrations`: {out['recorded']}")
    assert out["after"] == [], (
        f"a dropped Python fallback became effective: {out['after']}")
    assert out["predicate_after"] is False, (
        "the predicate answered True for a fallback that is recorded and dropped")
    print("BFALLBACK asymmetry: recorded=1 effective=0 predicate(CPU)=False")


def test_the_capability_gaps_this_honest_answer_names():
    """The cost, counted and named rather than waved at.

    Every key where upstream answers `True` and this shim answers `False` is a
    backend fallback kernel upstream's build has and this one does not. Upstream
    has 37; five cannot be spelled in the vendored enum (and three of those five
    are second names for keys that can), leaving **32** real, nameable gaps.

    Derived from both live sides every run, so the count moves the day either
    side does -- which is the point of printing it next to the 1440.
    """
    if not _available():
        return _skip("capability gaps")
    up, shim = _table(False), _table(True)
    names = set(shim["names"])
    gaps = sorted(set(up["fallback_true"]) - set(shim["fallback_true"]))
    spellable = sorted(g for g in gaps if g in names)
    unspellable = sorted(g for g in gaps if g not in names)
    assert len(gaps) == len(up["fallback_true"]), (
        "the shim claims a fallback upstream does not have: "
        f"{sorted(set(shim['fallback_true']) - set(up['fallback_true']))}")
    assert spellable, "no spellable gaps at all -- the probe is not comparing anything"
    print(f"BFALLBACK gaps: upstream {len(up['fallback_true'])} True, this shim 0. "
          f"{len(spellable)} spellable: {spellable}")
    print(f"BFALLBACK gaps: {len(unspellable)} not in the vendored enum: {unspellable}")


# ---------------------------------------------------------------------------
# 3. What it costs and what it buys, over the whole aten surface.
# ---------------------------------------------------------------------------
def test_the_honest_fallback_answer_resolves_nothing_extra_and_that_is_the_result():
    """`resolve_key` over 4893 `(op, key)` results, honest vs. the `_Unimplemented`.

    The honest answer adds **zero** resolutions. That is reported rather than
    buried, and the control is run on this tree: `alias_only` puts this name
    back to raising, so "nothing changed" cannot pass by the sweep having
    stopped reaching it.

    The guarantee is one-directional -- nothing that resolved before may stop
    resolving. The count itself is printed, not asserted; asserting it would be
    a test against progress (`docs/graph/METAKEY.md` §5).
    """
    if not _available():
        return _skip("honest sweep")
    before, after = _sweep("alias_only"), _sweep("honest")
    assert before["n"] == after["n"], f"{before['n']} vs {after['n']} results swept"
    assert before["counts"].get("GAP:_dispatch_has_backend_fallback", 0) > 1000, (
        "the control did not blame this name, so the sweep is not reaching it: "
        f"{before['counts']}")
    lost = set(before["resolved"]) - set(after["resolved"])
    assert not lost, f"{len(lost)} results stopped resolving: {sorted(lost)[:10]}"
    # The teeth, and the reason this is not merely a one-directional guard:
    # extra resolutions are *allowed*, but only if the registry can pay for
    # them. Growth with an empty `_shim_backend_fallbacks()` is the claimed-
    # kernel direction, and it is what a later round copying upstream's set
    # would look like. This still lets a real fallback land and raise the
    # count, so it is not a test against progress.
    gained = set(after["resolved"]) - set(before["resolved"])
    if gained:
        assert _table(True)["shim_backend_fallbacks"], (
            f"{len(gained)} more results resolve because of this name while the "
            "shim's fallback registry is empty -- those are claimed kernels")
    print(f"BFALLBACK sweep: resolved {len(before['resolved'])} -> "
          f"{len(after['resolved'])} of {after['n']}, "
          f"registry={_table(True)['shim_backend_fallbacks']}")


def test_no_resolve_key_result_on_the_whole_aten_surface_dies_on_a_gap_any_more():
    """The chain terminates -- honestly, at 1440.

    With both names answered, every one of the 4893 results either resolves or
    raises upstream's own `could not find kernel`. **None dies on a name this
    shim has not implemented.** That is what `docs/graph/BKEYSET.md` §3
    predicted and it is the last step of it.

    `_dispatch_autogradother_backends` is still an `_Unimplemented` and is still
    never blamed, because `resolve_key` only reads it when `k == AutogradOther`
    and these three keys never are. If that ever stops being true this reddens
    and names it.
    """
    if not _available():
        return _skip("chain terminates")
    after = _sweep("honest")
    gaps = {k: v for k, v in after["counts"].items() if k.startswith("GAP:")}
    raised = {k: v for k, v in after["counts"].items() if k.startswith("RAISED:")}
    assert not gaps, f"{sum(gaps.values())} results still die on an unimplemented name: {gaps}"
    assert not raised, f"{sum(raised.values())} results raised unexpectedly: {raised}"
    print(f"BFALLBACK terminal: {after['counts']}")


def test_upstreams_fallback_set_would_resolve_1861_more_by_claiming_kernels():
    """The number this round declined, measured rather than described.

    Patching upstream's 37-key set in resolves far more -- and every extra one
    is `resolve_key` handing back the key itself on the promise that a backend
    fallback will catch the call. There is no such fallback here, so each is a
    claimed kernel. The gap between the two sweeps is the size of the claim.

    This is the test that would go green if a later round copied upstream's
    table, so it asserts the *difference is real* rather than asserting a
    constant: the honest sweep must resolve strictly fewer, and every extra
    resolution must be a key the honest sweep refused.
    """
    if not _available():
        return _skip("the declined claim")
    honest, copied = _sweep("honest"), _sweep("upstream_fallbacks")
    extra = set(copied["resolved"]) - set(honest["resolved"])
    assert extra, (
        "copying upstream's fallback set resolves nothing extra -- either the "
        "control is broken or the honest answer already claims them")
    assert not (set(honest["resolved"]) - set(copied["resolved"])), (
        "the honest answer resolves something the copied one does not")
    print(f"BFALLBACK declined: copying upstream's {len(_table(False)['fallback_true'])}-key "
          f"set would resolve {len(copied['resolved'])} instead of "
          f"{len(honest['resolved'])} -- {len(extra)} claimed kernels")


def test_the_enum_divergence_that_bounds_what_this_agreement_can_mean():
    """The partition both comparisons above are only valid over.

    `docs/graph/ALIASINC.md` §2.1 measured it and this re-measures it, because
    without it a vendor bump that dropped keys from the stub would make the
    agreement test compare less and still report green. Here the partition is
    the assertion, so coverage shrinking is a failure rather than a smaller
    number nobody reads.
    """
    if not _available():
        return _skip("enum partition")
    up, shim = _table(False), _table(True)
    u, s = set(up["names"]), set(shim["names"])
    shared, stub_only, runtime_only = u & s, s - u, u - s
    assert (len(shared), len(stub_only), len(runtime_only)) == (123, 17, 22), (
        f"the enum partition moved: shared={len(shared)} stub_only={len(stub_only)} "
        f"runtime_only={len(runtime_only)}; stub_only={sorted(stub_only)} "
        f"runtime_only={sorted(runtime_only)}")
    # The 17 the stub has and upstream's runtime enum does not cannot be asked
    # about upstream, so both predicates must answer the direction that never
    # claims anything.
    claimed = sorted(stub_only & (set(shim["alias_true"]) | set(shim["fallback_true"])))
    assert not claimed, f"the shim invented an answer for stub-only keys: {claimed}"
    print(f"BFALLBACK partition: {len(shared)} shared | {len(stub_only)} stub-only "
          f"| {len(runtime_only)} runtime-only")


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
