"""Fusable regions of a captured decode step (issue #68, docs/graph/FUSION.md).

Two halves, and they are graded differently.

**The partitioner on hand-built FX graphs** (`test_partition_*`). No model,
no Dynamo: each graph is built node by node with `torch.fx.Graph`, so the
expected regions are derived by hand from the rule in
`torchnative/export/regions.py` and written down exactly. These need only
the vendored tree; they do not depend on `torchnative.compile`.

**One model's decode step captured through `torchnative.compile`**
(`test_decode_step_*`). A tiny random-weight Qwen3 (the `_TINY_CFG` of
`tests/models/test_qwen3.py`: two layers, GQA 4/2, `head_dim` 16, SDPA, no
download) is prefilled eagerly, then three decode steps run through
`torchnative.compile.Engine` with a recording backend. Every graph Dynamo
hands the backend is partitioned and the region list is asserted. Each case
runs in a fresh subprocess that imports the vendored tree and asserts
`hasattr(torch._C, "_aten_implemented")`.

**The decode expectations are not yet observed.** They were derived by
reading `transformers` 5.15.1's `modeling_qwen3.py` and `cache_utils.py`
against the rule; no model run was made when they were written (the agent
round that wrote them was not allowed to run one). The first CI run confirms
or refutes them, and a refutation is a finding about the rule or the
reading, to be reported, not a number to be edited until green. The
expectations, per decode step with L = 2 layers:

* L `attention` regions (one SDPA per layer);
* 4L + 1 RMSNorm regions (`input_layernorm`, `q_norm`, `k_norm`,
  `post_attention_layernorm` per layer, plus the final norm), each one
  region with `pow`, `mean` and `rsqrt`;
* L MLP regions that are exactly `silu, mul`;
* `DynamicCache`: no mutation, no refusal, and a **new compile on every
  step** (the past length is a shape, and `torchnative.compile` runs with
  static shapes today: FUSION.md §3). The owner approved a symbolic sequence
  length on 2026-11-13 (relayed by the leader); how `torchnative.compile`
  makes the length symbolic is still open (FUSION.md §8 Q4), and when it
  does, this expectation flips to "no new compile" and the test changes with
  it;
* `StaticCache`: `index_copy_` 2L times and `add_` L times
  (`cumulative_length`) listed as mutations outside every region, and **no
  new compile** after the first step;
* compiled logits equal to an eager twin's, bit for bit, every step.

`torchnative.compile` is PR #208 (branch `feat/dynamo-guards-manager`).
Without it the decode tests **fail by name**, naming #208; they never pass
or skip silently.

What this cannot find: whether a listed region is profitable, whether the
kernel compiler can take it, and anything on a device. It lists regions; it
fuses nothing.
"""

import json
import operator
import os
import pathlib
import subprocess
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


def _python_path():
    """This tree's `torchnative` first; `torch` from here if built, else from the
    main checkout (a worktree's `torch/` holds only tracked files)."""
    if os.path.isfile(_VENDOR_SHIM):
        return [_VENDOR_DIR]
    if os.path.isfile(_MAIN_VENDOR_SHIM):
        return [_VENDOR_DIR, _MAIN_VENDOR_DIR]
    return None


for _p in reversed(_python_path() or [_VENDOR_DIR]):
    if _p not in sys.path:
        sys.path.insert(0, _p)

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import _skip  # noqa: E402

import torch  # noqa: E402
import torch.fx  # noqa: E402

assert hasattr(torch._C, "_aten_implemented"), "imported upstream torch, not the shim"

from torchnative.export.regions import classify, extract, op_name, partition  # noqa: E402


# ---------------------------------------------------------------------------
# Hand-built graphs. Stand-in callables carry the op name the rule reads;
# they are never called.
# ---------------------------------------------------------------------------


def linear(*args, **kwargs):
    raise AssertionError("stand-in, never called")


def scaled_dot_product_attention(*args, **kwargs):
    raise AssertionError("stand-in, never called")


def silu(*args, **kwargs):
    raise AssertionError("stand-in, never called")


def dropout(*args, **kwargs):
    raise AssertionError("stand-in, never called")


def mystery(*args, **kwargs):
    raise AssertionError("stand-in, never called")


def _rms_norm(g, x, w):
    """Qwen3RMSNorm.forward as Dynamo spells it; returns (first node, output)."""
    h = g.call_method("to", (x, torch.float32))
    p = g.call_method("pow", (h, 2))
    m = g.call_method("mean", (p, -1), {"keepdim": True})
    a = g.call_function(operator.add, (m, 1e-6))
    r = g.call_method("rsqrt", (a,))
    hn = g.call_function(operator.mul, (h, r))
    return h, g.call_function(operator.mul, (w, hn))


def _decoder_layer_graph():
    """One simplified decoder layer: norm, q/k/v projections, SDPA, o_proj,
    residual, SiLU-gated MLP, residual."""
    g = torch.fx.Graph()
    x = g.placeholder("x")
    w_norm = g.placeholder("w_norm")
    wq, wk, wv, wo = (g.placeholder(n) for n in ("wq", "wk", "wv", "wo"))
    wg, wu, wd = (g.placeholder(n) for n in ("wg", "wu", "wd"))
    _, n1 = _rms_norm(g, x, w_norm)
    heads = []
    for w in (wq, wk, wv):
        y = g.call_function(linear, (n1, w))
        y = g.call_method("view", (y, 1, 1, 4, 8))
        heads.append(g.call_method("transpose", (y, 1, 2)))
    att = g.call_function(scaled_dot_product_attention, tuple(heads))
    at = g.call_method("transpose", (att, 1, 2))
    ar = g.call_method("reshape", (at, 1, 1, 32))
    o = g.call_function(linear, (ar, wo))
    res = g.call_function(operator.add, (x, o))
    gate = g.call_function(linear, (res, wg))
    s = g.call_function(silu, (gate,))
    up = g.call_function(linear, (res, wu))
    mm = g.call_function(operator.mul, (s, up))
    d = g.call_function(linear, (mm, wd))
    out = g.call_function(operator.add, (res, d))
    g.output(out)
    return g, x, w_norm, n1


def _contracted_is_acyclic(graph, groups):
    """Contract each group of nodes to one vertex and report whether the
    result is still a DAG. `groups` is a list of node lists."""
    rep = {}
    for i, group in enumerate(groups):
        for n in group:
            rep[n] = ("g", i)
    edges = {}
    for n in graph.nodes:
        a = rep.get(n, n)
        for p in n.all_input_nodes:
            b = rep.get(p, p)
            if a != b:
                edges.setdefault(b, set()).add(a)
    state = {}

    def visit(v):
        state[v] = 1
        for w in edges.get(v, ()):
            if state.get(w) == 1:
                return False
            if w not in state and not visit(w):
                return False
        state[v] = 2
        return True

    vertices = {rep.get(n, n) for n in graph.nodes}
    return all(visit(v) for v in vertices if v not in state)


def _all_groups(part):
    return [r.nodes for r in part.regions + part.singles]


# ---------------------------------------------------------------------------
# The partitioner, no model
# ---------------------------------------------------------------------------


def test_partition_a_decoder_layer_into_norm_attention_and_mlp_regions():
    g, x, w_norm, n1 = _decoder_layer_graph()
    part = partition(g)
    assert part.summary() == [
        ("reduction", ("to", "pow", "mean", "add", "rsqrt", "mul", "mul")),
        ("attention", ("scaled_dot_product_attention",)),
        ("elementwise", ("silu", "mul")),
    ], part.summary()
    assert [r.ops for r in part.singles] == [["add"], ["add"]], part.as_dict()
    assert part.views_only == 4, part.as_dict()
    assert [op for _, op in part.extern] == ["linear"] * 7, part.as_dict()
    assert part.mutations == [] and part.refused == [] and part.ignored == []
    assert part.calls == 27 and part.accounted(), part.as_dict()
    norm = part.regions[0]
    assert norm.inputs == [x, w_norm], norm.inputs
    assert norm.outputs == [n1], norm.outputs
    assert _contracted_is_acyclic(g, _all_groups(part))


def test_partition_extract_hands_off_a_graph_that_computes_the_region():
    g, x, w_norm, n1 = _decoder_layer_graph()
    norm = partition(g).regions[0]
    sub = extract(norm)
    kinds = [n.op for n in sub.nodes]
    assert kinds.count("placeholder") == 2 and kinds.count("call_method") + kinds.count(
        "call_function") == 7 and kinds[-1] == "output", kinds
    gm = torch.fx.GraphModule(torch.nn.Module(), sub)
    torch.manual_seed(68)
    xs, ws = torch.randn(2, 8), torch.randn(8)
    h = xs.to(torch.float32)
    want = ws * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + 1e-6))
    got = gm(xs, ws)
    assert torch.equal(got, want), (got, want)


def test_partition_keeps_kv_writes_outside_every_region():
    """The StaticCache shape: rotary, then `index_copy_` into the cache, then a read."""
    g = torch.fx.Graph()
    k = g.placeholder("k_new")
    cos = g.placeholder("cos")
    cache = g.placeholder("cache")
    pos = g.placeholder("pos")
    a = g.call_function(operator.mul, (k, cos))
    b = g.call_method("neg", (a,))
    c = g.call_function(operator.add, (a, b))
    g.call_method("index_copy_", (cache, 2, pos, c))
    d = g.call_method("transpose", (cache, 2, 3))
    e = g.call_function(operator.mul, (d, 2.0))
    g.call_function(operator.setitem, (cache, 0, e))
    g.output(e)
    part = partition(g)
    assert part.summary() == [("elementwise", ("mul", "neg", "add"))], part.summary()
    assert [op for _, op, _ in part.mutations] == ["index_copy_", "setitem"], part.as_dict()
    why = part.mutations[0][2]
    assert "index_copy_" in why and "in place" in why, why
    assert [r.ops for r in part.singles] == [["mul"]], part.as_dict()
    assert part.accounted() and _contracted_is_acyclic(g, _all_groups(part))

    # The write is a barrier even for a region it does not read: `a` reads the
    # cache before `index_copy_`, `b` after it, and no FX edge says so.
    g2 = torch.fx.Graph()
    cache2 = g2.placeholder("cache")
    pos2 = g2.placeholder("pos")
    src = g2.placeholder("src")
    v = g2.call_method("view", (cache2, -1))
    a2 = g2.call_function(operator.mul, (v, 2))
    g2.call_method("index_copy_", (cache2, 0, pos2, src))
    b2 = g2.call_function(operator.add, (a2, 1))
    g2.output(b2)
    part2 = partition(g2)
    assert part2.regions == [], part2.summary()
    assert [r.ops for r in part2.singles] == [["mul"], ["add"]], part2.as_dict()


def test_partition_refuses_an_op_without_a_rule_by_name():
    g = torch.fx.Graph()
    x = g.placeholder("x")
    a = g.call_method("exp", (x,))
    b = g.call_function(mystery, (a,))
    c = g.call_method("sin", (b,))
    d = g.call_method("cos", (c,))
    g.output(d)
    part = partition(g)
    assert [(n.name, op) for n, op, _ in part.refused] == [(b.name, "mystery")], part.as_dict()
    assert "mystery" in part.refused[0][2] and "no fusion rule" in part.refused[0][2]
    assert part.summary() == [("elementwise", ("sin", "cos"))], part.summary()
    assert [r.ops for r in part.singles] == [["exp"]], part.as_dict()
    assert part.accounted()


def test_partition_joins_the_most_recent_producer_and_never_makes_a_cycle():
    g = torch.fx.Graph()
    x = g.placeholder("x")
    y = g.placeholder("y")
    p = g.call_method("exp", (x,))
    q = g.call_method("sin", (y,))
    z = g.call_function(operator.add, (p, q))
    w = g.call_function(operator.mul, (p, 2))
    g.output((z, w))
    part = partition(g)
    assert part.summary() == [("elementwise", ("sin", "add"))], part.summary()
    assert [r.ops for r in part.singles] == [["exp"], ["mul"]], part.as_dict()

    # A naive "group connected fusable nodes" rule fuses `a` with `c` around
    # the extern `b`; contracting that group is a cycle (a -> b -> c -> a).
    # The helper must see it, or the acyclicity checks above cannot fail.
    g2 = torch.fx.Graph()
    x2 = g2.placeholder("x")
    w2 = g2.placeholder("w")
    a = g2.call_method("exp", (x2,))
    b = g2.call_function(linear, (a, w2))
    c = g2.call_function(operator.add, (a, b))
    g2.output(c)
    assert not _contracted_is_acyclic(g2, [[a, c]]), "the cycle check cannot fail"
    part2 = partition(g2)
    assert part2.regions == [] and [r.ops for r in part2.singles] == [["exp"], ["add"]]
    assert _contracted_is_acyclic(g2, _all_groups(part2))


def test_partition_classifies_the_spellings_that_look_alike():
    g = torch.fx.Graph()
    x = g.placeholder("x")
    idx = g.placeholder("idx")
    cases = [
        (g.call_function(operator.and_, (x, x)), "elementwise"),
        (g.call_method("add_", (x, 1)), "mutation"),
        (g.call_function(operator.iadd, (x, 1)), "mutation"),
        (g.call_method("cumsum", (x, 0)), "refused"),
        (g.call_function(dropout, (x, 0.1, False)), "elementwise"),
        (g.call_function(dropout, (x, 0.1, True)), "refused"),
        (g.call_function(operator.getitem, (x, idx)), "extern"),
        (g.call_function(operator.getitem, (x, slice(0, 2))), "view"),
        (g.call_method("size", (x,)), "ignored"),
    ]
    got = [(n.name, classify(n)[0]) for n, _ in cases]
    want = [(n.name, cat) for n, cat in cases]
    assert got == want, got
    assert "random" in classify(cases[5][0])[1]


def test_op_name_reads_the_shims_own_spellings():
    """Dynamo graphs name torch-level callables and the door's OpOverloads; the
    rule reads their names, so the shim's names must be the base op names."""
    import torch.nn.functional as F

    targets = {
        "add": [operator.add, torch.add, torch.ops.aten.add.Tensor],
        "rsqrt": [torch.rsqrt],
        "linear": [F.linear],
        "silu": [F.silu],
        "scaled_dot_product_attention": [F.scaled_dot_product_attention],
        "index_copy_": [torch.ops.aten.index_copy_.default],
    }
    g = torch.fx.Graph()
    x = g.placeholder("x")
    got = {}
    for want, fns in targets.items():
        got[want] = [op_name(g.call_function(fn, (x,))) for fn in fns]
    bad = {k: v for k, v in got.items() if any(name != k for name in v)}
    assert not bad, bad


# ---------------------------------------------------------------------------
# One model's decode step, through torchnative.compile (needs #208)
# ---------------------------------------------------------------------------

#: `tests/models/test_qwen3.py`'s `_TINY_CFG`, copied so that importing this
#: suite does not import that one.
_TINY_CFG = dict(
    vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    max_position_embeddings=64, tie_word_embeddings=True, rope_theta=1e6,
    bos_token_id=1, eos_token_id=[2, 3], pad_token_id=1,
)
_TINY_IDS = [5, 7, 9, 11, 13, 17, 19, 23]
_STEPS = 3
_LAYERS = _TINY_CFG["num_hidden_layers"]

_CHILD = r"""
import json, sys
req = json.loads(sys.argv[1])
import torch
assert hasattr(torch._C, "_aten_implemented"), "the child imported upstream torch, not the shim"
out = {"no_compile": None}
try:
    from torchnative.compile import Engine
except ImportError as e:
    out["no_compile"] = "%s: %s" % (type(e).__name__, e)
    print(json.dumps(out)); raise SystemExit(0)
from torchnative.export.regions import partition
from transformers import DynamicCache, Qwen3Config, Qwen3ForCausalLM, StaticCache

cfg = Qwen3Config(**req["cfg"])
cfg._attn_implementation = "sdpa"
torch.manual_seed(68)
model = Qwen3ForCausalLM(cfg).eval()
ids = torch.tensor([req["ids"]])
T = ids.shape[1]

def make_cache():
    if req["cache"] == "dynamic":
        return DynamicCache(config=cfg)
    return StaticCache(config=cfg, max_cache_len=req["max_cache_len"])

graphs = []
def backend(gm, example_inputs):
    graphs.append(gm)
    return gm.forward

eng = Engine(backend)
step = eng.compile_method(model)
cache_c, cache_e = make_cache(), make_cache()
per_step, equal = [], []
with torch.no_grad():
    model(ids, past_key_values=cache_c, use_cache=True)
    first = model(ids, past_key_values=cache_e, use_cache=True)
    nxt = first.logits[:, -1:].argmax(-1)
    for s in range(req["steps"]):
        pos = torch.tensor([[T + s]])
        g0 = len(graphs)
        lc = step(input_ids=nxt, past_key_values=cache_c, use_cache=True, position_ids=pos).logits
        le = model(input_ids=nxt, past_key_values=cache_e, use_cache=True, position_ids=pos).logits
        per_step.append(len(graphs) - g0)
        equal.append(bool(torch.equal(lc, le)))
        nxt = le[:, -1:].argmax(-1)

merged = {"regions": [], "singles": [], "extern": [], "mutations": [], "ignored": [],
          "refused": [], "views_only": 0, "accounted": True}
for gm in graphs[:per_step[0]]:
    d = partition(gm).as_dict()
    for key in ("regions", "singles", "extern", "mutations", "ignored", "refused"):
        merged[key] += d[key]
    merged["views_only"] += d["views_only"]
    merged["accounted"] = merged["accounted"] and d["accounted"]
out.update(merged)
out["graphs_per_step"] = per_step
out["equal"] = equal
out["stats"] = dict(eng.stats)
print(json.dumps(out))
"""

_RESULTS = {}


def _run(cache):
    if cache in _RESULTS:
        return _RESULTS[cache]
    path = _python_path()
    if path is None:
        raise _skip.Skip("vendored tree not built: no %s and no %s" % (_VENDOR_SHIM, _MAIN_VENDOR_SHIM))
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(path)
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    req = {"cache": cache, "cfg": _TINY_CFG, "ids": _TINY_IDS, "steps": _STEPS,
           "max_cache_len": 16}
    proc = subprocess.run([sys.executable, "-c", _CHILD, json.dumps(req)],
                          capture_output=True, text=True, timeout=900, env=env)
    lines = [l for l in proc.stdout.splitlines() if l.startswith("{")]
    assert proc.returncode == 0 and lines, (
        "the %s-cache child exited %d:\n%s" % (cache, proc.returncode, proc.stderr[-3000:]))
    out = json.loads(lines[-1])
    assert out["no_compile"] is None, (
        "torchnative.compile is absent (%s). The decode-step capture needs PR #208 "
        "(hook-free Dynamo, branch feat/dynamo-guards-manager) merged into this "
        "branch; it refuses here by name rather than skip." % out["no_compile"])
    _RESULTS[cache] = out
    return out


def _norms(out):
    return [ops for kind, ops in out["regions"]
            if {"pow", "mean", "rsqrt"} <= set(ops)]


def _check_common(out, cache):
    assert out["graphs_per_step"][0] >= 1 and out["stats"]["fail"] == 0, (
        "%s: the first decode step compiled no graph (a silent eager run): %s"
        % (cache, out["stats"]))
    assert out["accounted"], "%s: a call node is in no field of the partition" % cache
    attention = [ops for kind, ops in out["regions"] if kind == "attention"]
    assert len(attention) == _LAYERS, (cache, attention, out["regions"])
    assert len(_norms(out)) == 4 * _LAYERS + 1, (cache, _norms(out), out["regions"])
    assert out["equal"] == [True] * _STEPS, (cache, out["equal"])


def test_decode_step_with_dynamic_cache_lists_its_fusable_regions():
    out = _run("dynamic")
    _check_common(out, "dynamic")
    mlp = [ops for kind, ops in out["regions"] if ops == ["silu", "mul"]]
    assert len(mlp) == _LAYERS, (mlp, out["regions"])
    assert out["mutations"] == [], out["mutations"]
    assert out["refused"] == [], out["refused"]
    print("    fusion: DynamicCache decode step: %d regions (%d attention, %d norm, %d mlp), "
          "%d singles, %d extern, refused %s"
          % (len(out["regions"]), _LAYERS, len(_norms(out)), len(mlp), len(out["singles"]),
             len(out["extern"]), out["refused"]))
    for kind, ops in out["regions"]:
        print("      %-11s %s" % (kind, " ".join(ops)))


def test_decode_step_with_dynamic_cache_recompiles_every_step_under_static_shapes():
    out = _run("dynamic")
    assert all(n >= 1 for n in out["graphs_per_step"]), (
        "expected a new compile on every DynamicCache step (the past length is a "
        "shape and torchnative.compile runs static today; the owner approved a "
        "symbolic length on 2026-11-13, and once torchnative.compile marks it "
        "symbolic this expectation flips: FUSION.md §3.3, §8 Q4): %s" % out["graphs_per_step"])
    print("    fusion: DynamicCache: graphs compiled per decode step %s" % out["graphs_per_step"])


def test_decode_step_with_static_cache_keeps_kv_writes_outside_regions():
    out = _run("static")
    _check_common(out, "static")
    muts = out["mutations"]
    assert muts.count("index_copy_") == 2 * _LAYERS, muts
    assert muts.count("add_") == _LAYERS, muts
    assert out["graphs_per_step"][1:] == [0] * (_STEPS - 1), (
        "expected one StaticCache decode graph to serve every step: %s" % out["graphs_per_step"])
    print("    fusion: StaticCache decode step: %d regions, mutations %s, refused %s, "
          "graphs per step %s"
          % (len(out["regions"]), muts, out["refused"], out["graphs_per_step"]))
    for kind, ops in out["regions"]:
        print("      %-11s %s" % (kind, " ".join(ops)))


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = _skip.run_tests(items, suite="test_fusion_capture")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
