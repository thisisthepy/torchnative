"""The fixed fusion pattern library and its matcher (issue #68, docs/graph/FUSION.md §5.5).

The owner chose, 2026-11-13 (relayed by the leader), that fused kernels come
from **a fixed pattern library matched at run time** (FUSION.md §8 Q1 (b)),
and approved **a symbolic sequence length** (kernels take the length as a
symbolic dimension, not bucketed shapes, where the pattern allows). This
suite tests the matcher only: `torchnative/export/patterns.py` reads the
regions `torchnative/export/regions.py` finds and names the pattern each one
is, or lists why it is none. Nothing is compiled or dispatched; there are no
kernels yet.

Every graph here is built by hand with `torch.fx.Graph`. Each input carries a
stand-in for Dynamo's `example_value` meta (`_Meta`: a shape and a dtype), and
a symbolic size is a stand-in for a `SymInt` (`_Sym`, an object that is not an
`int`, which is how the matcher tells a symbol from a concrete size). No model,
no Dynamo, no FakeTensorMode: the vendored tree is enough.

What is asserted:

* each first-set pattern (`rms_norm`, `add_rms_norm`, `silu_mul`, `rotary`,
  `sdpa_kv`) matches the region the partitioner finds, at two concrete
  lengths and at a symbolic one, with the length bound as a symbol and the
  static dims bound to the same numbers;
* near-misses do not match and are listed with a reason: wrong op order, an
  extra op, a wrong dtype (meta and cast constant), wrong wiring with the
  same op sequence (a negative control shows the sequence alone is equal),
  a static dim given a symbol, missing meta, an ambiguous library;
* a deliberately broken pattern (three mutants of `rms_norm`) makes the
  positive check fail, so that check can fail (AGENTS.md §17.5);
* the registry checks itself, and refuses a pattern with an undefined ref.

**Expected red before `patterns.py` exists:** every test, because the import of
`torchnative.export.patterns` below fails, so the suite fails as a whole at
import. Not observed: the round that wrote this was not allowed to run a
suite (AGENTS.md §10); CI runs it.

What this cannot find: whether the real Dynamo graph of a model spells these
ops the way the hand-built graphs do (the decode-step suite,
`test_fusion_capture.py`, is where that is observed, and it is blocked by
#268/#269), whether a matched region would compute the right numbers in a
kernel (no kernel exists), and view parameters other than the rotary halves
(the matcher passes through views it does not name).
"""

import dataclasses
import operator
import os
import pathlib
import sys

_TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]
_SUPPORT_DIR = _TESTS_DIR / "_support"
if str(_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPPORT_DIR))

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_VENDOR_DIR = str(_REPO_ROOT / "torchnative" / "python")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")
_MAIN_VENDOR_DIR = "/Volumes/macMini/thisisthepy/torchnative/torchnative/python"
_MAIN_VENDOR_SHIM = os.path.join(_MAIN_VENDOR_DIR, "torch", "_C.abi3.so")

if os.path.isfile(_VENDOR_SHIM):
    _PATH = [_VENDOR_DIR]
elif os.path.isfile(_MAIN_VENDOR_SHIM):
    _PATH = [_VENDOR_DIR, _MAIN_VENDOR_DIR]
else:
    _PATH = [_VENDOR_DIR]
for _p in reversed(_PATH):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import _skip  # noqa: E402

import torch  # noqa: E402
import torch.fx  # noqa: E402

assert hasattr(torch._C, "_aten_implemented"), "imported upstream torch, not the shim"

from torchnative.export.patterns import (  # noqa: E402
    PATTERNS,
    Pattern,
    Step,
    check_registry,
    match_partition,
)
from torchnative.export.regions import partition  # noqa: E402

F32, F16 = torch.float32, torch.float16


# ---------------------------------------------------------------------------
# Stand-ins. The callables carry the op name the rule reads and are never called.
# ---------------------------------------------------------------------------


def linear(*args, **kwargs):
    raise AssertionError("stand-in, never called")


def silu(*args, **kwargs):
    raise AssertionError("stand-in, never called")


def cat(*args, **kwargs):
    raise AssertionError("stand-in, never called")


def scaled_dot_product_attention(*args, **kwargs):
    raise AssertionError("stand-in, never called")


class _Meta:
    """What the matcher reads from Dynamo's `example_value`: a shape and a dtype."""

    def __init__(self, shape, dtype):
        self.shape = tuple(shape)
        self.dtype = dtype


class _Sym:
    """A symbolic size. Not an `int`, as a `SymInt` is not; prints like one (`s0`)."""

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    __repr__ = __str__


def _m(node, shape, dtype=F32):
    node.meta["example_value"] = _Meta(shape, dtype)
    return node


# ---------------------------------------------------------------------------
# Graph builders, one per pattern, with the knobs the near-misses turn
# ---------------------------------------------------------------------------


def _rms_graph(S, *, residual=False, keep_sum=True, cast_back=True, upcast=F32, xdtype=F32,
               N=32, misorder=False, uncast_mul=False, meta=True):
    """Qwen3RMSNorm.forward as Dynamo spells it (transformers 5.15.1,
    `modeling_qwen3.py:59-64`), optionally after a residual add."""
    g = torch.fx.Graph()
    shape = (1, S, N)
    mark = _m if meta else (lambda n, *a, **k: n)
    if residual:
        a = mark(g.placeholder("a"), shape, xdtype)
        b = mark(g.placeholder("b"), shape, xdtype)
        x = g.call_function(operator.add, (a, b))
    else:
        x = mark(g.placeholder("x"), shape, xdtype)
    w = mark(g.placeholder("w"), (N,), F32)
    h = g.call_method("to", (x, upcast))
    p = g.call_method("pow", (h, 2))
    m = g.call_method("mean", (p, -1), {"keepdim": True})
    if misorder:
        r = g.call_method("rsqrt", (m,))
        r = g.call_function(operator.add, (r, 1e-6))
    else:
        r = g.call_function(operator.add, (m, 1e-6))
        r = g.call_method("rsqrt", (r,))
    hn = g.call_function(operator.mul, (x if uncast_mul else h, r))
    if cast_back:
        hn = g.call_method("to", (hn, xdtype))
    y = g.call_function(operator.mul, (w, hn))
    g.output((x, y) if residual and keep_sum else y)
    return g


def _silu_mul_graph(S, *, I=48, gdtype=F32, udtype=F32, extra=False):
    """Qwen3MLP's `act_fn(gate_proj(x)) * up_proj(x)` between the projections."""
    g = torch.fx.Graph()
    gate = _m(g.placeholder("gate"), (1, S, I), gdtype)
    up = _m(g.placeholder("up"), (1, S, I), udtype)
    s = g.call_function(silu, (gate,))
    y = g.call_function(operator.mul, (s, up))
    if extra:
        y = g.call_method("tanh", (y,))
    g.output(y)
    return g


def _rotary_graph(S, *, H=4, D=16, swap_cat=False, neg_first_half=False):
    """`apply_rotary_pos_emb` for one of q/k (`modeling_qwen3.py:140-170`): the
    head comes out of a projection as views, so the whole chain is one region."""
    g = torch.fx.Graph()
    hid = g.placeholder("hidden")
    wq = g.placeholder("wq")
    cos = _m(g.placeholder("cos"), (1, 1, S, D))
    sin = _m(g.placeholder("sin"), (1, 1, S, D))
    lin = g.call_function(linear, (hid, wq))
    v = g.call_method("view", (lin, 1, -1, H, D))
    x = _m(g.call_method("transpose", (v, 1, 2)), (1, H, S, D))
    a = g.call_function(operator.mul, (x, cos))
    x1 = g.call_function(operator.getitem, (x, (Ellipsis, slice(None, D // 2, None))))
    x2 = g.call_function(operator.getitem, (x, (Ellipsis, slice(D // 2, None, None))))
    n = g.call_function(operator.neg, (x1 if neg_first_half else x2,))
    parts = (x1, n) if swap_cat else (n, x1 if not neg_first_half else x2)
    c = g.call_function(cat, (parts,), {"dim": -1})
    mm = g.call_function(operator.mul, (c, sin))
    y = g.call_function(operator.add, (a, mm))
    g.output(y)
    return g


def _sdpa_graph(Skv, *, Sq=1, Hq=4, Hkv=4, D=16, gqa=None, dropout=0.0, mask=True):
    """`sdpa_attention_forward`'s call (`integrations/sdpa_attention.py:154-163`)
    on a KV cache of length `Skv`."""
    g = torch.fx.Graph()
    q = _m(g.placeholder("q"), (1, Hq, Sq, D))
    k = _m(g.placeholder("k"), (1, Hkv, Skv, D))
    v = _m(g.placeholder("v"), (1, Hkv, Skv, D))
    # a symbolic D has no number to take a root of; the pattern accepts any number as the scale
    scale = D ** -0.5 if isinstance(D, int) else 0.25
    kwargs = {"dropout_p": dropout, "scale": scale, "is_causal": False}
    if mask:
        kwargs["attn_mask"] = _m(g.placeholder("mask"), (1, 1, Sq, Skv), torch.bool)
    if gqa is not None:
        kwargs["enable_gqa"] = gqa
    o = g.call_function(scaled_dot_product_attention, (q, k, v), kwargs)
    g.output(o)
    return g


def _report(g, patterns=None):
    part = partition(g)
    rep = match_partition(part, patterns=PATTERNS if patterns is None else patterns)
    assert rep.accounted(), rep.as_dict()
    return rep


def _expect(g, names, patterns=None):
    """The positive check: the graph's regions match exactly these pattern names."""
    rep = _report(g, patterns)
    got = [m.name for m in rep.matched]
    assert got == names and rep.unmatched == [], rep.as_dict()
    return rep


def _expect_listed(g, *needles):
    """The negative check: nothing matches, and the one region is listed with a
    reason that names what differs."""
    rep = _report(g)
    assert rep.matched == [], rep.as_dict()
    assert len(rep.unmatched) == 1, rep.as_dict()
    why = rep.as_dict()["unmatched"][0]["reason"]
    for needle in needles:
        assert needle in why, (needle, why)
    return rep


# ---------------------------------------------------------------------------
# Each pattern matches, with the length symbolic
# ---------------------------------------------------------------------------


def test_patterns_rms_norm_matches_at_two_lengths_and_a_symbolic_one():
    seen = []
    for S in (3, 11, _Sym("s0")):
        for cast_back in (True, False):
            m = _expect(_rms_graph(S, cast_back=cast_back), ["rms_norm"]).matched[0]
            assert m.dims["N"] == 32 and m.dims["lead"] == (1, S), m.as_dict()
            assert m.dtypes == {"T": F32, "W": F32}, m.as_dict()
            seen.append((S, m.symbolic, m.spelling))
    assert [s for _, s, _ in seen] == [(), (), (), (), ("lead",), ("lead",)], seen
    assert len({sp for _, _, sp in seen}) == 2, "both spellings must be exercised: %s" % seen


def test_patterns_add_rms_norm_matches_the_residual_and_returns_both_values():
    for S in (3, 11, _Sym("s0")):
        rep = _expect(_rms_graph(S, residual=True), ["add_rms_norm"])
        m = rep.matched[0]
        assert m.dims["N"] == 32 and m.dims["lead"] == (1, S), m.as_dict()
        assert m.as_dict()["outputs"] == ["s", "y"], m.as_dict()
    # The model's final norm: the sum is read by nothing but the norm.
    m = _expect(_rms_graph(7, residual=True, keep_sum=False), ["add_rms_norm"]).matched[0]
    assert m.region.outputs == [m.steps["y"]], m.as_dict()
    # Without the optional mark, that region is not this kernel (the rule can fail).
    strict = {"add_rms_norm": tuple(dataclasses.replace(p, optional_outputs=frozenset())
                                    for p in PATTERNS["add_rms_norm"])}
    rep = _report(_rms_graph(7, residual=True, keep_sum=False), patterns=strict)
    assert rep.matched == [] and "not read outside" in rep.as_dict()["unmatched"][0]["reason"]


def test_patterns_silu_mul_matches_at_two_lengths_and_a_symbolic_one():
    for S in (3, 11, _Sym("s0")):
        m = _expect(_silu_mul_graph(S), ["silu_mul"]).matched[0]
        assert m.dims["I"] == 48 and m.dims["lead"] == (1, S), m.as_dict()
        assert m.symbolic == (("lead",) if isinstance(S, _Sym) else ()), m.as_dict()


def test_patterns_rotary_matches_through_the_projection_views():
    for S in (3, 11, _Sym("s0")):
        rep = _expect(_rotary_graph(S), ["rotary"])
        m = rep.matched[0]
        assert m.dims["S"] == S and m.dims["D"] == 16 and m.dims["H"] == 4, m.as_dict()
        assert m.inputs["x"].op == "call_method" and m.inputs["x"].target == "transpose"
        assert m.symbolic == (("S",) if isinstance(S, _Sym) else ()), m.as_dict()


def test_patterns_sdpa_matches_a_kv_cache_of_any_length_and_gqa():
    for Skv in (9, 13, _Sym("s1")):
        m = _expect(_sdpa_graph(Skv), ["sdpa_kv"]).matched[0]
        assert m.dims["Skv"] == Skv and m.dims["Sq"] == 1 and m.dims["D"] == 16, m.as_dict()
        assert "mask" in m.inputs, m.as_dict()
    m = _expect(_sdpa_graph(9, Hkv=2, gqa=True, mask=False), ["sdpa_kv"]).matched[0]
    assert m.dims["Hq"] == 4 and m.dims["Hkv"] == 2 and "mask" not in m.inputs, m.as_dict()
    # prefill: the query length is symbolic too
    m = _expect(_sdpa_graph(_Sym("s1"), Sq=_Sym("s1"), mask=False), ["sdpa_kv"]).matched[0]
    assert m.symbolic == ("Skv", "Sq"), m.as_dict()


# ---------------------------------------------------------------------------
# Near-misses: they do not match, and they are listed with a reason
# ---------------------------------------------------------------------------


def test_patterns_wrong_op_order_is_listed():
    _expect_listed(_rms_graph(5, misorder=True), "op sequence", "rsqrt")


def test_patterns_an_extra_op_is_listed():
    _expect_listed(_silu_mul_graph(5, extra=True), "op sequence", "tanh")


def test_patterns_a_wrong_dtype_is_listed():
    _expect_listed(_silu_mul_graph(5, gdtype=F32, udtype=F16), "dtype")
    _expect_listed(_rms_graph(5, xdtype=torch.int64), "dtype", "int64")
    # the upcast constant: an RMSNorm that accumulates in float16 is another kernel
    _expect_listed(_rms_graph(5, upcast=F16, xdtype=F16), "float16")


def test_patterns_wrong_wiring_with_the_same_op_sequence_is_listed():
    # Negative control: the near-misses have exactly the op sequence of the
    # good graphs, so only the structural (edge) check can reject them.
    for good, bad in ((_rms_graph(5), _rms_graph(5, uncast_mul=True)),
                      (_rotary_graph(5), _rotary_graph(5, swap_cat=True)),
                      (_rotary_graph(5), _rotary_graph(5, neg_first_half=True))):
        pg, pb = partition(good), partition(bad)
        assert pg.summary() == pb.summary(), (pg.summary(), pb.summary())
    _expect_listed(_rms_graph(5, uncast_mul=True), "operand")
    _expect_listed(_rotary_graph(5, swap_cat=True), "operand")
    _expect_listed(_rotary_graph(5, neg_first_half=True), "half")


def test_patterns_a_static_dim_given_a_symbol_is_listed():
    # The length may be symbolic; the normalised dim is a kernel's tile and may not.
    _expect_listed(_rms_graph(5, N=_Sym("s2")), "N", "static")
    _expect_listed(_sdpa_graph(9, D=_Sym("s3")), "D", "static")


def test_patterns_sdpa_near_misses_are_listed():
    _expect_listed(_sdpa_graph(9, dropout=0.1), "dropout_p")
    _expect_listed(_sdpa_graph(9, Hkv=2, mask=False), "enable_gqa")


def test_patterns_missing_meta_is_listed_not_guessed():
    _expect_listed(_rms_graph(5, meta=False), "meta")


# ---------------------------------------------------------------------------
# The report, the registry, and a check that can fail
# ---------------------------------------------------------------------------


def test_patterns_report_lists_every_region_and_says_how_the_unmatched_run():
    g = torch.fx.Graph()
    x = _m(g.placeholder("x"), (1, 4, 32))
    w = _m(g.placeholder("w"), (32,))
    h = g.call_method("to", (x, F32))
    p = g.call_method("pow", (h, 2))
    m = g.call_method("mean", (p, -1), {"keepdim": True})
    a = g.call_function(operator.add, (m, 1e-6))
    r = g.call_method("rsqrt", (a,))
    y = g.call_function(operator.mul, (w, g.call_function(operator.mul, (h, r))))
    gate = _m(g.call_function(linear, (y, w)), (1, 4, 48))
    up = _m(g.call_function(linear, (y, w)), (1, 4, 48))
    s = g.call_function(silu, (gate,))
    mm = g.call_function(operator.mul, (s, up))
    t = g.call_method("exp", (g.call_function(linear, (mm, w)),))
    t = g.call_method("cos", (t,))
    g.output(t)
    rep = _report(g)
    d = rep.as_dict()
    assert [e["pattern"] for e in d["matched"]] == ["rms_norm", "silu_mul"], d
    assert [e["ops"] for e in d["unmatched"]] == [["exp", "cos"]], d
    assert d["unmatched"][0]["runs"] == rep.UNMATCHED_RUNS, d
    assert "_aten_dispatch" in rep.UNMATCHED_RUNS and "Q2" in rep.UNMATCHED_RUNS
    assert d["regions"] == 3 and d["accounted"], d


def test_patterns_an_ambiguous_library_is_listed_not_resolved():
    twin = {"rms_norm": PATTERNS["rms_norm"], "rms_norm_twin": PATTERNS["rms_norm"]}
    rep = _report(_rms_graph(5), patterns=twin)
    assert rep.matched == [], rep.as_dict()
    why = rep.as_dict()["unmatched"][0]["reason"]
    assert "ambiguous" in why and "rms_norm_twin" in why, why


def test_patterns_the_registry_checks_itself():
    assert check_registry(PATTERNS) is None
    assert list(PATTERNS) == ["rms_norm", "add_rms_norm", "silu_mul", "rotary", "sdpa_kv"]
    good = PATTERNS["silu_mul"][0]
    bad = dataclasses.replace(good, steps=(good.steps[0], Step("y", "mul", ("s", "nope"),
                                                               commutative=True)))
    try:
        check_registry({"broken": (bad,)})
    except ValueError as e:
        assert "nope" in str(e), e
    else:
        raise AssertionError("check_registry accepted a pattern with an undefined ref")
    assert isinstance(good, Pattern)


def _mutants():
    """Three ways to break `rms_norm`: a constant, an edge, a dropped step."""

    def pow_three(p):
        steps = tuple(dataclasses.replace(s, args=(s.args[0], 3)) if s.name == "p" else s
                      for s in p.steps)
        return dataclasses.replace(p, steps=steps)

    def mul_reads_the_input(p):
        steps = tuple(dataclasses.replace(s, args=("x", "r")) if s.name == "hn" else s
                      for s in p.steps)
        return dataclasses.replace(p, steps=steps)

    def no_rsqrt(p):
        steps = tuple(s for s in p.steps if s.name != "r")
        steps = tuple(dataclasses.replace(s, args=("h", "a")) if s.name == "hn" else s
                      for s in steps)
        return dataclasses.replace(p, steps=steps)

    return [pow_three, mul_reads_the_input, no_rsqrt]


def test_patterns_a_broken_pattern_makes_the_positive_check_fail():
    g = _rms_graph(5)
    _expect(g, ["rms_norm"], patterns={"rms_norm": PATTERNS["rms_norm"]})
    survived = []
    for mutate in _mutants():
        broken = {"rms_norm": tuple(mutate(p) for p in PATTERNS["rms_norm"])}
        try:
            _expect(g, ["rms_norm"], patterns=broken)
        except AssertionError:
            continue
        survived.append(mutate.__name__)
    assert not survived, "mutants the positive check did not catch: %s" % survived


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = _skip.run_tests(items, suite="test_fusion_patterns")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
