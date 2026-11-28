"""Fusable regions of a captured FX graph (issue #68, docs/graph/FUSION.md).

**Agent proposal, not confirmed by the owner.** The rule below, the category
tables and the handoff shape (`extract`) are what docs/graph/FUSION.md §5
proposes. None of it is a decision; the owner's questions are FUSION.md §8.
The owner did decide (2026-11-13, relayed by the leader) that fused kernels
come from a fixed pattern library matched at run time: `patterns.py` matches
the regions found here against that library.

What this module does: it reads an FX graph (the one Dynamo hands a
`torchnative.compile` backend, or an ATen graph after decomposition) and
groups its call nodes into regions that one fused kernel could compute.
It does not fuse, compile or dispatch anything. It is pure Python and needs
nothing beyond `torch.fx`.

## The rule

Every call node falls into exactly one category, by its op name:

| category | what | in a region? |
|---|---|---|
| `elementwise` | one output element from the same element of each input | yes |
| `reduction` | reduces one axis (`sum`, `mean`, `softmax`, norms) | yes |
| `view` | changes indexing, not data (`view`, `transpose`, `getitem` by slice) | yes, but counts as no work |
| `attention` | `scaled_dot_product_attention` and its ATen spellings | its own region, one node |
| `extern` | a library kernel: GEMM, convolution, gather, allocation | no; it ends regions |
| `mutation` | writes into an argument (`index_copy_`, `add_`, `setitem`) | no; it ends **every** open region |
| `ignored` | metadata and global state (`size`, `_set_grad_enabled`) | no |
| `refused` | no rule for it, or random | no; it ends **every** open region and is listed by name |

Regions are grown in graph order. A fusable node joins the **open** region
of its most recent producer, and every other open region it reads from is
closed. A fusable node with no open producer starts a region. A node that is
not fusable closes every open region it reads from. A closed region takes no
new members.

A mutation closes every open region, read or not, and so does a refused op
(it may write; nothing says it does not). The reason: a pre-functionalization
FX graph (Dynamo's) has no edge from `index_copy_(cache, ...)` to a later
read of `cache`; the order is only the node order. A region that straddled
the write would compute some of its members on the wrong side of it.

Why closing is enough to stay acyclic: an open region has, by construction,
no consumer outside itself. So a path from a region back into it would have
to leave it first, and leaving closes it. Contracting each region to one
kernel therefore never creates a cycle, which is the property a fused graph
must keep to be executable at all.

Why "most recent" and not "merge": merging two open regions at a shared
consumer is also acyclic, but it grows regions across unrelated chains (the
rotary `cos`/`sin` of the whole model would merge every layer's `q` and `k`
into one region). Joining one and closing the others keeps regions to one
chain each. It loses some horizontal fusion; FUSION.md §5.2 names that cost.

A region with two or more non-view ops is reported in `Partition.regions`.
A region with one is in `Partition.singles` (nothing to fuse: it runs
unfused). A region of views only is dropped, because views need no kernel.

## What it refuses, by name

`Partition.refused` lists every node with no rule, as `(node, op, reason)`.
Nothing is dropped silently: every call node lands in exactly one of
`regions`, `singles`, `extern`, `mutations`, `ignored`, `refused` or a
views-only region, and `Partition.accounted()` checks that count.

What this cannot find: whether a region is *worth* fusing, whether a
reduction's axis allows fusing it with the elementwise ops around it (two
reductions over different axes in one region are legal here and may need
splitting by the kernel compiler), and anything that depends on shapes. It
reads op names only.
"""

from __future__ import annotations

__all__ = [
    "ATTENTION",
    "ELEMENTWISE",
    "EXTERN",
    "IGNORED",
    "REDUCTION",
    "VIEW",
    "Partition",
    "Region",
    "classify",
    "extract",
    "op_name",
    "partition",
]

ELEMENTWISE = frozenset({
    "abs", "acos", "add", "addcdiv", "addcmul", "and_", "asin", "atan", "atan2",
    "bfloat16", "bitwise_and", "bitwise_not", "bitwise_or", "bitwise_xor", "bool",
    "ceil", "clamp", "clamp_max", "clamp_min", "clip", "clone", "convert_element_type",
    "copy", "cos", "cosh", "div", "double", "elu", "eq", "erf", "exp", "exp2", "expm1",
    "fill", "float", "floor", "floordiv", "fmod", "ge", "gelu", "gt", "half",
    "hardsigmoid", "hardswish", "hardtanh", "hypot", "int", "invert", "isinf", "isnan",
    "le", "leaky_relu", "lerp", "lift_fresh_copy", "log", "log1p", "log2", "logaddexp",
    "logical_and", "logical_not", "logical_or", "long", "lt", "masked_fill", "maximum",
    "minimum", "mish", "mod", "mul", "nan_to_num", "ne", "neg", "not_", "or_", "pow",
    "reciprocal", "relu", "relu6", "remainder", "round", "rsqrt", "rsub", "sgn",
    "sigmoid", "sign", "silu", "sin", "sinh", "softplus", "sqrt", "square", "sub", "tan",
    "tanh", "to", "_to_copy", "tril", "triu", "true_divide", "truediv", "trunc",
    "type_as", "where", "xor",
    # indexed copies that a fused kernel computes by index arithmetic
    "cat", "stack", "repeat",
    # factories: an iota or a constant fill is an elementwise kernel with no input
    "arange", "full", "full_like", "ones", "ones_like", "scalar_tensor", "zeros",
    "zeros_like",
})

REDUCTION = frozenset({
    "all", "amax", "amin", "any", "argmax", "argmin", "group_norm", "layer_norm",
    "log_softmax", "_log_softmax", "logsumexp", "max", "mean", "min",
    "native_group_norm", "native_layer_norm", "norm", "prod", "rms_norm",
    "_fused_rms_norm", "softmax", "_softmax", "std", "std_mean", "sum", "var", "var_mean",
})

ATTENTION = frozenset({
    "scaled_dot_product_attention",
    "_scaled_dot_product_attention_math",
    "_scaled_dot_product_cudnn_attention",
    "_scaled_dot_product_efficient_attention",
    "_scaled_dot_product_flash_attention",
    "_scaled_dot_product_flash_attention_for_cpu",
    "_scaled_dot_product_fused_attention_overrideable",
})

VIEW = frozenset({
    "_reshape_alias", "_unsafe_view", "alias", "as_strided", "chunk", "contiguous",
    "detach", "expand", "expand_as", "flatten", "getitem", "movedim", "narrow",
    "permute", "reshape", "select", "slice", "split", "split_with_sizes", "squeeze",
    "swapaxes", "t", "transpose", "unbind", "unflatten", "unsqueeze", "view", "view_as",
})

EXTERN = frozenset({
    "addmm", "addmv", "baddbmm", "bmm", "conv1d", "conv2d", "conv3d",
    "conv_transpose1d", "conv_transpose2d", "convolution", "dot", "einsum",
    "embedding", "empty", "empty_like", "empty_strided", "gather", "index",
    "index_copy", "index_put", "index_select", "linear", "masked_scatter", "matmul",
    "mm", "mv", "new_empty", "new_ones", "new_zeros", "repeat_interleave", "scatter",
    "scatter_add", "scatter_reduce", "take",
})

IGNORED = frozenset({
    "_assert", "_assert_async", "_assert_scalar", "_check", "_check_is_size",
    "_enter_autocast", "_exit_autocast", "_log_api_usage_once",
    "_record_function_enter_new", "_record_function_exit", "_set_grad_enabled",
    "dim", "getattr", "is_floating_point", "numel", "set_grad_enabled", "size",
    "stride", "sym_constrain_range_for_size", "sym_numel", "sym_size",
    "sym_storage_offset", "sym_stride",
})

RANDOM = frozenset({
    "bernoulli", "dropout", "multinomial", "native_dropout", "normal", "rand",
    "rand_like", "randint", "randint_like", "randn", "randn_like", "randperm", "uniform",
})

#: In-place spellings whose name does not end in `_` (the `operator` module's).
_MUTATING = frozenset({
    "iadd", "iand", "ifloordiv", "ilshift", "imatmul", "imod", "imul", "ior", "ipow",
    "irshift", "isub", "itruediv", "ixor", "setitem",
})

#: Names that end in `_` and do not write: `operator.and_` and its siblings.
_NOT_MUTATING = frozenset({"and_", "not_", "or_"})

_MODULE_OPS = {
    "Dropout": "dropout", "Embedding": "embedding", "GELU": "gelu",
    "Identity": "alias", "LayerNorm": "layer_norm", "Linear": "linear",
    "ReLU": "relu", "SiLU": "silu", "Sigmoid": "sigmoid", "Softmax": "softmax",
    "Tanh": "tanh",
}

_FUSABLE = ("elementwise", "reduction", "view")


def op_name(node, root=None):
    """The base op name of a call node: `aten.add.Tensor`, `operator.add`,
    `torch.add` and the method `x.add` all answer `add`.

    `root` is the owning module, needed only for `call_module` nodes.
    """
    if node.op == "call_method":
        return str(node.target)
    if node.op == "call_module":
        sub = root.get_submodule(node.target) if root is not None else None
        kind = type(sub).__name__ if sub is not None else str(node.target)
        return _MODULE_OPS.get(kind, "module:" + kind)
    target = node.target
    name = getattr(target, "__name__", None) or str(target)
    # An OpOverload is named `add.Tensor` (upstream and the shim alike); an
    # OpOverloadPacket `add`; `str()` of either may carry the `aten.` prefix.
    if name.startswith(("aten.", "prims.", "aten::", "prims::")):
        name = name.replace("::", ".").split(".", 1)[1]
    return name.split(".", 1)[0]


def _is_mutating(name):
    if name in _MUTATING:
        return True
    return name.endswith("_") and not name.endswith("__") and name not in _NOT_MUTATING


def _arg(node, index, key, default):
    if key in node.kwargs:
        return node.kwargs[key]
    if len(node.args) > index:
        return node.args[index]
    return default


def classify(node, root=None):
    """`(category, reason)` for one call node; `reason` is `None` unless refused."""
    name = op_name(node, root)
    if name in IGNORED:
        return "ignored", None
    if _is_mutating(name):
        return "mutation", (
            f"{name} writes in place; a region is single-assignment, so the write "
            "stays outside every region and runs through `_aten_dispatch` in "
            "program order (docs/graph/FUSION.md §4)"
        )
    if name in RANDOM:
        if name == "dropout" and (_arg(node, 2, "training", True) is False
                                  or _arg(node, 1, "p", 0.5) == 0):
            return "elementwise", None  # identity in eval: nothing is drawn
        return "refused", (
            f"{name} draws random numbers; a fused kernel would need the same "
            "generator state as eager, which no region carries (docs/graph/FUSION.md §6)"
        )
    if name == "getitem" and len(node.all_input_nodes) > 1:
        return "extern", None  # indexing by a tensor is a gather, not a view
    if name in VIEW:
        return "view", None
    if name in ELEMENTWISE:
        return "elementwise", None
    if name in REDUCTION:
        return "reduction", None
    if name in ATTENTION:
        return "attention", None
    if name in EXTERN:
        return "extern", None
    return "refused", (
        f"{name}: no fusion rule; it runs unfused through `_aten_dispatch` and is "
        "listed here by name (docs/graph/FUSION.md §6)"
    )


class Region:
    """One kernel's worth of nodes. `kind` is `elementwise`, `reduction` or `attention`.

    `nodes` are the members in graph order, `ops` the names of the members
    that do work (views left out), `inputs` the outside nodes the members
    read, and `outputs` the members something outside the region reads.
    """

    def __init__(self, kind, members, names):
        self.kind = kind
        self.members = list(members)  # [(node, category)] in graph order
        self.nodes = [n for n, _ in self.members]
        self.ops = [names[n] for n, cat in self.members if cat != "view"]
        inside = set(self.nodes)
        inputs, seen = [], set()
        for n in self.nodes:
            for p in n.all_input_nodes:
                if p not in inside and p not in seen:
                    seen.add(p)
                    inputs.append(p)
        self.inputs = inputs
        self.outputs = [n for n in self.nodes if any(u not in inside for u in n.users)]

    def __repr__(self):
        return f"Region({self.kind}, {self.ops})"


class Partition:
    """The result of `partition`. Every call node is in exactly one field."""

    def __init__(self):
        self.regions = []    # [Region], two or more non-view ops, or attention
        self.singles = []    # [Region], one non-view op: nothing to fuse
        self.views_only = 0  # regions of views alone, dropped
        self.extern = []     # [(node, op)]
        self.mutations = []  # [(node, op, reason)]
        self.ignored = []    # [(node, op)]
        self.refused = []    # [(node, op, reason)]
        self.calls = 0       # call nodes seen
        self._view_nodes = 0

    def accounted(self):
        """True when every call node is in exactly one field (the conservation check)."""
        in_regions = sum(len(r.nodes) for r in self.regions + self.singles)
        total = (in_regions + self._view_nodes + len(self.extern) + len(self.mutations)
                 + len(self.ignored) + len(self.refused))
        return total == self.calls

    def summary(self):
        """`[(kind, (op, ...)), ...]` for the reported regions, in graph order."""
        return [(r.kind, tuple(r.ops)) for r in self.regions]

    def as_dict(self):
        """Names only, JSON-ready: what a test or a log records."""
        return {
            "regions": [[r.kind, list(r.ops)] for r in self.regions],
            "singles": [list(r.ops) for r in self.singles],
            "views_only": self.views_only,
            "extern": [op for _, op in self.extern],
            "mutations": [op for _, op, _ in self.mutations],
            "ignored": [op for _, op in self.ignored],
            "refused": [[n.name, op, why] for n, op, why in self.refused],
            "calls": self.calls,
            "accounted": self.accounted(),
        }


def partition(graph_or_module, root=None):
    """Group the call nodes of an FX graph into fusable regions (the rule above)."""
    graph = getattr(graph_or_module, "graph", graph_or_module)
    if root is None and hasattr(graph_or_module, "graph"):
        root = graph_or_module
    out = Partition()
    names = {}
    order = {}
    run_of = {}   # node -> run index
    runs = []     # [{"members": [(node, cat)], "open": bool}]

    def close(indices):
        for i in indices:
            runs[i]["open"] = False

    for position, node in enumerate(graph.nodes):
        order[node] = position
        if node.op in ("placeholder", "get_attr", "output"):
            continue
        out.calls += 1
        name = op_name(node, root)
        names[node] = name
        category, reason = classify(node, root)
        producers = [p for p in node.all_input_nodes if p in run_of]
        open_runs = []
        for p in producers:
            i = run_of[p]
            if runs[i]["open"] and i not in open_runs:
                open_runs.append(i)

        if category in _FUSABLE:
            if open_runs:
                def latest(i):
                    return max(order[p] for p in producers if run_of[p] == i)
                target = max(open_runs, key=latest)
                close(i for i in open_runs if i != target)
            else:
                runs.append({"members": [], "open": True})
                target = len(runs) - 1
            runs[target]["members"].append((node, category))
            run_of[node] = target
            continue

        if category in ("mutation", "refused"):
            close(range(len(runs)))  # a write is a barrier in program order
        else:
            close(open_runs)
        if category == "attention":
            out.regions.append(Region("attention", [(node, category)], names))
        elif category == "extern":
            out.extern.append((node, name))
        elif category == "mutation":
            out.mutations.append((node, name, reason))
        elif category == "ignored":
            out.ignored.append((node, name))
        else:
            out.refused.append((node, name, reason))

    for run in runs:
        members = run["members"]
        work = [cat for _, cat in members if cat != "view"]
        if not work:
            out.views_only += 1
            out._view_nodes += len(members)
            continue
        kind = "reduction" if "reduction" in work else "elementwise"
        region = Region(kind, members, names)
        (out.regions if len(work) >= 2 else out.singles).append(region)

    out.regions.sort(key=lambda r: order[r.nodes[0]])
    out.singles.sort(key=lambda r: order[r.nodes[0]])
    return out


def extract(region):
    """The region as its own `torch.fx.Graph`: inputs become placeholders, the
    values used outside the region become the output (a tuple when more than one).

    This is the handoff shape FUSION.md §5.3 proposes to TypedPython's kernel
    compiler (agent proposal, not confirmed). Node `meta` is copied, so a
    Dynamo graph's `example_value` (shape, dtype) travels with each node.
    """
    import torch.fx

    graph = torch.fx.Graph()
    env = {}
    for n in region.inputs:
        env[n] = graph.placeholder(n.name)
        env[n].meta.update(n.meta)
    for n in region.nodes:
        env[n] = graph.node_copy(n, lambda a: env[a])
    outs = [env[n] for n in region.outputs]
    graph.output(outs[0] if len(outs) == 1 else tuple(outs))
    return graph
