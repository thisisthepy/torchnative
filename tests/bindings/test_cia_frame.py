"""Issue #372: the CompositeImplicitAutograd (CIA) decomposition frame.

Direction: leader decision 2026-11-16, agent proposal; not owner-confirmed.
CIA decomposition of aten ops comes before functionalization step 3. The
leader's conditions: (1) the ops covered are those upstream's exports actually
decompose, measured, and everything else refuses by name; (2) upstream's own
Python decompositions are used where they exist, own compositions only where
they do not; (3) results are compared with upstream's op sequence node for
node; (4) this PR is the frame plus the Linear+ReLU path only.
docs/design/DECOMPOSE.md describes the frame.

What the shim has (bootstrap.py, aten.rs):

* `_CIA_TABLE`: op key -> body, its source and the upstream measurement it
  was checked against. `torch._C._shim_cia_table()` reads it back, with the
  upstream-CIA ops the shim runs a native kernel for and has no body for
  (`pending`).
* The door's two phases (`_shim_cia_dispatch`): `door`, above Functionalize
  and the modes, and `backend`, just before the native table.
* `OpOverload._op_dk(CompositeImplicitAutograd, ...)`: the body, or a refusal
  naming the op.
* `torch._C._shim_cia_counters()`: per op, how many times its body ran through
  each entry, so a test can tell the decomposition from a kernel (AGENTS.md
  13.1).

Every upstream fact a test compares with is re-derived from upstream (torch
2.13.0) in a separate subprocess on every run; `UPSTREAM_FACTS` is the one
recorded table, and `test_upstream_is_the_recorded_table` re-measures it.

What these tests cannot see: whether a body agrees with upstream in value
(the bodies emit ops whose kernels the golden harness compares; the frame
claims the op sequence, grade "reaches", AGENTS.md 16), any op outside
`CASES` and `NO_KERNEL`, and the memory formats other than `contiguous_format`
(this shim keeps no channels-last strides, so `contiguous` is held to
upstream's sequence for `contiguous_format` only).

The comparison with upstream's op sequence is exact except for four separated
kinds, each pinned per platform as an upper bound (a count above its cap
fails; below it a `CIAFRAME CAP ... can be lowered` line is printed): rows
that differ only in the contiguity flag of tensor descriptors (issue #383:
fresh results keep their inputs' stride order upstream and are row-major in
the shim), and, off macOS, reordered, reordered-and-flag and C++-type-name
rows. CI run 43814367564: Linux 215 layout only, 55 reordered and flag, 49
reordered, 3 type-name; macOS 246 layout only. Separated rather than fixed
first (leader, 2026-11-16) because the values and the aliasing of a fresh
result are right and the effect shows by name in individual tests. What this
cannot find: a flag difference that is a real aliasing defect hides in the
separated count (the view kernels were that; tests/ops/test_viewstride.py
guards them).

These tests were written before the implementation and have not run on a
built shim (leader rule: agents compile only). The upstream half was run on
2026-11-16 by hand for `UPSTREAM_FACTS`.
"""
import hashlib
import json
import os
import pathlib
import re
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
VENDOR = ROOT / "torchnative" / "python"

sys.path.insert(0, str(ROOT / "tests" / "_support"))
import _skip  # noqa: E402


def _run(script, shim, *extra, timeout=1200):
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = str(VENDOR)
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run([sys.executable, "-c", script, "shim" if shim else "upstream", *extra],
                          capture_output=True, text=True, env=env, timeout=timeout)
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
# The cases each body is held to. ADDING AN OP TO `_CIA_TABLE` MEANS ADDING
# ITS CASES HERE: `test_the_table_is_truthful_against_upstream` fails for a
# covered op with no cases. Each case is an argument tuple as a Python
# expression, evaluated after `torch.manual_seed(0)` on both sides.
# ---------------------------------------------------------------------------

def _qkv(dtype="torch.float32", batch=2, heads=3, rows=4, cols=5, head_dim=8, kv_heads=None):
    """`query, key, value` as an expression, for the attention cases."""
    kv_heads = heads if kv_heads is None else kv_heads
    return (f"torch.randn({batch}, {heads}, {rows}, {head_dim}, dtype={dtype}), "
            f"torch.randn({batch}, {kv_heads}, {cols}, {head_dim}, dtype={dtype}), "
            f"torch.randn({batch}, {kv_heads}, {cols}, {head_dim}, dtype={dtype})")


def _sdpa(kw="", qkv=None):
    """An attention case: `query, key, value` positional, `kw` the keywords."""
    return '{"args": (' + (_qkv() if qkv is None else qkv) + ',), "kw": {' + kw + '}}'


def _one(kw=""):
    """A `to.dtype_layout` case on a 2-by-3 float32 tensor (keyword-only)."""
    return '{"args": (torch.randn(2, 3),), "kw": {' + kw + '}}'


def _lstm(steps=5, batch=2, features=3, hidden=4, layers=1, bidirectional=False, bias=True,
          batch_first=False, proj=0, dropout=0.0, train=False, dtype="torch.float64",
          zero_state=False):
    """An `aten.lstm.input` case: `(input, [h0, c0], params, has_biases,
    num_layers, dropout, train, bidirectional, batch_first)` as an
    expression, the parameters in `nn.LSTM`'s `_flat_weights` order (per
    layer and direction `w_ih, w_hh[, b_ih, b_hh][, w_hr]`)."""
    dirs = 2 if bidirectional else 1
    out_size = proj or hidden
    shape = (batch, steps, features) if batch_first else (steps, batch, features)
    state = "torch.zeros" if zero_state else "torch.randn"
    params = []
    for layer in range(layers):
        width = features if layer == 0 else out_size * dirs
        for _ in range(dirs):
            params += [f"torch.randn({4 * hidden}, {width}, dtype={dtype})",
                       f"torch.randn({4 * hidden}, {out_size}, dtype={dtype})"]
            if bias:
                params += [f"torch.randn({4 * hidden}, dtype={dtype})"] * 2
            if proj:
                params += [f"torch.randn({proj}, {hidden}, dtype={dtype})"]
    return (f"(torch.randn{shape!r}.to({dtype}), "
            f"[{state}({layers * dirs}, {batch}, {out_size}, dtype={dtype}), "
            f"{state}({layers * dirs}, {batch}, {hidden}, dtype={dtype})], "
            f"[{', '.join(params)}], {bias}, {layers}, {dropout}, {train}, "
            f"{bidirectional}, {batch_first})")


#: float32 `lstm` rows. Upstream's C++ body takes oneDNN for them
#: (`use_mkldnn`: built with oneDNN, a float32 CPU input) on the Linux x86_64
#: wheel and the native path on the macOS arm64 one, which has no oneDNN
#: (`torch.backends.mkldnn.is_available()` False there, measured 2026-11-21).
#: This shim answers `torch._C._has_mkldnn` False, so its body takes the native
#: path everywhere: these rows are `CASES` on macOS only, and
#: `test_lstm_float32_rows_take_onednn_upstream_only_where_it_is_built` holds
#: the other platforms to the difference. float64 rows (no oneDNN for float64
#: on any platform) are in `CASES` everywhere.
_LSTM_FLOAT32 = [
    ("float32 1 layer", _lstm(dtype="torch.float32")),
    ("float32 2 layers bidirectional batch_first",
     _lstm(dtype="torch.float32", layers=2, bidirectional=True, batch_first=True)),
]

#: A 4-D tensor whose strides are channels-last (a permuted NHWC tensor).
_CL = "torch.randn(2, 4, 5, 3).permute(0, 3, 1, 2)"
_CPU = "torch.device('cpu')"


#: Cases whose upstream sequence the body reproduces exactly. An expression
#: is the argument tuple, or `{"args": (...), "kw": {...}}` for keyword-only
#: arguments. Measured on torch 2.13.0, 2026-11-16 (#375).
CASES = {
    # The metadata reads (#423): nothing is emitted, the answer is the
    # tensor's own; a view with an offset and non-unit strides, and a 0-D one.
    "aten.sym_size.int": [
        ("view dim 1", "(torch.ones(2, 3)[:, 1:], 1)"),
        ("view dim -2", "(torch.ones(2, 3)[:, 1:], -2)"),
    ],
    "aten.sym_stride.int": [
        ("view dim 0", "(torch.ones(2, 3)[:, 1:], 0)"),
        ("transposed dim 1", "(torch.ones(2, 3).t(), 1)"),
    ],
    "aten.sym_numel.default": [
        ("view", "(torch.ones(2, 3)[:, 1:],)"),
        ("0-d", "(torch.tensor(1.5),)"),
        ("empty", "(torch.ones(0, 4),)"),
    ],
    "aten.sym_storage_offset.default": [
        ("view", "(torch.ones(2, 3)[:, 1:],)"),
        ("base", "(torch.ones(2, 3),)"),
    ],
    "aten.dim.default": [
        ("2-d", "(torch.ones(2, 3),)"),
        ("0-d", "(torch.tensor(1.5),)"),
    ],
    "aten.linear.default": [
        ("2d+bias", "(torch.randn(2, 8), torch.randn(4, 8), torch.randn(4))"),
        ("2d+bias f64", "(torch.randn(2, 8, dtype=torch.float64), torch.randn(4, 8, dtype=torch.float64),"
                        " torch.randn(4, dtype=torch.float64))"),
        ("2d+bias2d", "(torch.randn(2, 8), torch.randn(4, 8), torch.randn(2, 4))"),
        ("2d nc+bias", "(torch.randn(8, 2).t(), torch.randn(4, 8), torch.randn(4))"),
        ("0rows 2d+bias", "(torch.randn(0, 8), torch.randn(4, 8), torch.randn(4))"),
        ("1d+bias", "(torch.randn(8), torch.randn(4, 8), torch.randn(4))"),
        ("3d+bias", "(torch.randn(3, 2, 8), torch.randn(4, 8), torch.randn(4))"),
        ("4d+bias", "(torch.randn(2, 3, 2, 8), torch.randn(4, 8), torch.randn(4))"),
        ("3d+bias[1]", "(torch.randn(3, 2, 8), torch.randn(4, 8), torch.randn(1))"),
        ("3d+bias[1,4]", "(torch.randn(3, 2, 8), torch.randn(4, 8), torch.randn(1, 4))"),
        ("1d+bias[1,4]", "(torch.randn(8), torch.randn(4, 8), torch.randn(1, 4))"),
        # The branches that emit `matmul` (CIA, now with its own body): a
        # logging mode sees matmul's decomposition, as upstream's does.
        ("2d nobias", "(torch.randn(2, 8), torch.randn(4, 8))"),
        ("3d nobias", "(torch.randn(3, 2, 8), torch.randn(4, 8))"),
        ("3d nc+bias", "(torch.randn(2, 3, 8).transpose(0, 1), torch.randn(4, 8), torch.randn(4))"),
        ("3d+bias2d", "(torch.randn(3, 2, 8), torch.randn(4, 8), torch.randn(2, 4))"),
        # Moved here from NO_KERNEL when `squeeze_.dim` got a kernel (#374):
        # t, unsqueeze, mm, squeeze_.dim.
        ("1d nobias", "(torch.randn(8), torch.randn(4, 8))"),
    ],
    # matmul: every branch of upstream's C++ kernel that does not need `dot`
    # or `mv` (those are in NO_KERNEL). `1d@2d` moved here from NO_KERNEL when
    # `squeeze_.dim` got a kernel (#374).
    "aten.matmul.default": [
        ("2d@2d", "(torch.randn(3, 4), torch.randn(4, 5))"),
        ("1d@2d", "(torch.randn(4), torch.randn(4, 5))"),
        ("3d@2d fold", "(torch.randn(2, 3, 4), torch.randn(4, 5))"),
        ("4d@2d fold", "(torch.randn(2, 3, 4, 5), torch.randn(5, 6))"),
        ("4d strided@2d fold", "(torch.randn(2, 3, 4, 5)[:, :, ::2], torch.randn(5, 6))"),
        ("0 batch fold", "(torch.randn(0, 3, 4), torch.randn(4, 5))"),
        ("0 inner fold", "(torch.randn(2, 3, 0), torch.randn(0, 0))"),
        ("3d nc@2d no fold", "(torch.randn(2, 3, 4).transpose(1, 2), torch.randn(3, 5))"),
        ("3d nc@2d no fold 2", "(torch.randn(4, 3, 2).transpose(0, 2), torch.randn(4, 5))"),
        ("3d@2d grad fold", "(torch.randn(2, 3, 4), torch.randn(4, 5, requires_grad=True))"),
        ("2d@3d no fold", "(torch.randn(3, 4), torch.randn(2, 4, 5))"),
        ("2d grad@3d fold transposed", "(torch.randn(4, 5, requires_grad=True), torch.randn(2, 5, 6))"),
        ("2d@3d zero batch", "(torch.randn(3, 4), torch.randn(0, 4, 5))"),
        ("1d@3d", "(torch.randn(4), torch.randn(2, 4, 5))"),
        ("3d@1d nc bmm", "(torch.randn(4, 3, 2).transpose(0, 2), torch.randn(4))"),
        ("3d@3d", "(torch.randn(2, 3, 4), torch.randn(2, 4, 5))"),
        ("3d@3d [1,1]", "(torch.randn(1, 3, 4), torch.randn(1, 4, 5))"),
        ("3d@3d bcast right", "(torch.randn(2, 3, 4), torch.randn(1, 4, 5))"),
        ("3d@3d bcast left", "(torch.randn(1, 3, 4), torch.randn(2, 4, 5))"),
        ("3d@3d bcast left, grad", "(torch.randn(1, 3, 4, requires_grad=True), torch.randn(2, 4, 5))"),
        ("3d@3d bcast right, grad", "(torch.randn(2, 3, 4), torch.randn(1, 4, 5, requires_grad=True))"),
        ("4d@3d broadcast copy", "(torch.randn(2, 1, 3, 4), torch.randn(5, 4, 6))"),
        ("4d@4d broadcast", "(torch.randn(2, 1, 3, 4), torch.randn(1, 4, 4, 5))"),
        ("5d@5d", "(torch.randn(2, 1, 3, 4, 5), torch.randn(1, 4, 3, 5, 6))"),
        ("0d refused", "(torch.tensor(1.0), torch.randn(4))"),
        ("batch mismatch refused", "(torch.randn(2, 3, 4), torch.randn(3, 4, 5))"),
    ],
    "aten.reshape.default": [
        ("view", "(torch.randn(2, 3, 4), [6, 4])"),
        ("view -1", "(torch.randn(2, 3, 4), [-1, 4])"),
        ("view flatten", "(torch.randn(2, 3, 4), [-1])"),
        ("view same", "(torch.randn(2, 3, 4), [2, 3, 4])"),
        ("view -1 middle", "(torch.randn(2, 3, 4), [2, -1, 4])"),
        ("view tuple", "(torch.randn(2, 3, 4), (6, 4))"),
        ("view 0-d to 1", "(torch.tensor(1.0), [1])"),
        ("view to 0-d", "(torch.randn(1), [])"),
        ("view transposed", "(torch.randn(2, 3).t(), [3, 2])"),
        ("view with unit dim", "(torch.randn(3, 1, 4).transpose(0, 1), [3, 4])"),
        ("view slice", "(torch.randn(6, 4)[::2], [3, 2, 2])"),
        ("view zero numel", "(torch.randn(0, 3), [-1, 3])"),
        ("view zero numel 2", "(torch.randn(0, 3), [0, 0, 3])"),
        ("view int32", "(torch.randn(2, 3).int(), [6])"),
        ("copy transposed", "(torch.randn(2, 3, 4).transpose(0, 1), [6, 4])"),
        ("copy transposed -1", "(torch.randn(2, 3, 4).transpose(0, 1), [-1])"),
        ("copy transposed 2", "(torch.randn(2, 3, 4).transpose(1, 2), [-1, 3])"),
        ("copy slice", "(torch.randn(6, 4)[::2], [-1])"),
        ("copy expand", "(torch.randn(2, 1, 3).expand(2, 4, 3), [8, 3])"),
        ("copy sliced inner", "(torch.randn(4, 5)[:, 1:3], [8])"),
        ("copy tuple", "(torch.randn(2, 3, 4).transpose(0, 1), (6, 4))"),
        ("bad size refused", "(torch.randn(2, 3), [7])"),
        ("two -1 refused", "(torch.randn(2, 3), [-1, -1])"),
        ("bad -1 refused", "(torch.randn(2, 3), [4, -1])"),
        ("negative refused", "(torch.randn(2, 3), [-2, 3])"),
    ],
    "aten.contiguous.default": [
        ("contiguous", "(torch.randn(2, 3),)"),
        ("transposed", "(torch.randn(2, 3).t(),)"),
        ("1xN transposed", "(torch.randn(1, 3).t(),)"),
        ("unit dim transposed", "(torch.randn(2, 1, 3).transpose(0, 1),)"),
        ("0 rows", "(torch.randn(0, 3),)"),
        ("0-d", "(torch.tensor(1.0),)"),
        ("contiguous_format kw", '{"args": (torch.randn(2, 3).t(),), "kw": {"memory_format": torch.contiguous_format}}'),
        ("preserve on contiguous", '{"args": (torch.randn(2, 3),), "kw": {"memory_format": torch.preserve_format}}'),
        ("preserve on transposed refused",
         '{"args": (torch.randn(2, 3).t(),), "kw": {"memory_format": torch.preserve_format}}'),
        ("channels_last rank 3 refused",
         '{"args": (torch.randn(2, 3, 4),), "kw": {"memory_format": torch.channels_last}}'),
        ("channels_last_3d rank 4 refused",
         '{"args": (torch.randn(2, 3, 4, 5),), "kw": {"memory_format": torch.channels_last_3d}}'),
    ],
    "aten.narrow.default": [
        ("start 1 len 2", "(torch.randn(4, 5), 0, 1, 2)"),
        ("negative start", "(torch.randn(4, 5), 1, -3, 2)"),
        ("negative dim", "(torch.randn(4, 5), -1, 0, 5)"),
        ("negative start whole", "(torch.randn(4, 5), 0, -4, 4)"),
        ("start at end len 0", "(torch.randn(4, 5), 0, 4, 0)"),
        ("last row", "(torch.randn(4, 5), 0, -1, 1)"),
        ("len 0", "(torch.randn(4, 5), 0, 0, 0)"),
        ("transposed", "(torch.randn(4, 5).t(), 0, 1, 2)"),
        ("end exceeds refused", "(torch.randn(4, 5), 0, 3, 2)"),
        ("start beyond refused", "(torch.randn(4, 5), 0, 5, 0)"),
        ("start below refused", "(torch.randn(4, 5), 0, -5, 1)"),
        ("0-d refused", "(torch.tensor(1.0), 0, 0, 1)"),
        ("negative length refused", "(torch.randn(4, 5), 0, 1, -1)"),
        ("dim out of range refused", "(torch.randn(4, 5), 2, 0, 1)"),
    ],
    "aten.softmax.int": [
        ("dim 1", "(torch.randn(2, 3), 1)"),
        ("dim -1", "(torch.randn(2, 3), -1)"),
        ("dim 0", "(torch.randn(2, 3), 0)"),
        ("3d", "(torch.randn(2, 3, 4), 1)"),
        ("0 rows", "(torch.randn(0, 3), 1)"),
        ("transposed", "(torch.randn(2, 3).t(), 1)"),
        ("0-d", "(torch.tensor(1.0), 0)"),
        ("dtype float64", "(torch.randn(2, 3), 1, torch.float64)"),
        ("dtype same", "(torch.randn(2, 3), 1, torch.float32)"),
        ("dtype from float64", "(torch.randn(2, 3, dtype=torch.float64), 1, torch.float32)"),
        ("dtype from int32", "(torch.randn(2, 3).int(), 1, torch.float32)"),
        ("dtype kw", '{"args": (torch.randn(2, 3), 1), "kw": {"dtype": torch.float64}}'),
        ("dtype kw None", '{"args": (torch.randn(2, 3), 1), "kw": {"dtype": None}}'),
    ],
    "aten.where.ScalarSelf": [
        ("float scalar", "(torch.randn(2, 3) > 0, 1.0, torch.randn(2, 3))"),
        ("int scalar, float other", "(torch.randn(2, 3) > 0, 1, torch.randn(2, 3))"),
        ("int scalar, int32 other", "(torch.randn(2, 3) > 0, 1, torch.randn(2, 3).int())"),
        ("int scalar, int64 other", "(torch.randn(2, 3) > 0, 2, torch.randn(2, 3).long())"),
        ("float scalar, int32 other", "(torch.randn(2, 3) > 0, 1.5, torch.randn(2, 3).int())"),
        ("float scalar, int64 other", "(torch.randn(2, 3) > 0, 1.0, torch.randn(2, 3).long())"),
        ("float scalar, f64 other", "(torch.randn(2, 3) > 0, 1.5, torch.randn(2, 3, dtype=torch.float64))"),
        ("bool scalar, bool other", "(torch.randn(2, 3) > 0, True, torch.randn(2, 3) > 0)"),
        ("int scalar, bool other", "(torch.randn(2, 3) > 0, 1, torch.randn(2, 3) > 0)"),
        ("condition broadcast", "(torch.randn(3) > 0, 1.0, torch.randn(2, 3))"),
        ("0-d other", "(torch.randn(2, 3) > 0, 1.0, torch.tensor(2.0))"),
        ("transposed other", "(torch.randn(3, 2) > 0, 1.0, torch.randn(2, 3).t())"),
        ("non-bool condition refused", "(torch.randn(2, 3), 1.0, torch.randn(2, 3))"),
    ],
    "aten.view_as.default": [
        ("2d to 2d", "(torch.randn(2, 3), torch.empty(3, 2))"),
        ("2d to 1d", "(torch.randn(2, 3), torch.empty(6))"),
    ],
    "aten.argsort.default": [
        ("default", "(torch.randn(2, 3),)"),
        ("dim 0 descending", "(torch.randn(2, 3), 0, True)"),
    ],
    "aten.argsort.stable": [
        ("stable true", '{"args": (torch.randn(2, 3),), "kw": {"stable": True, "dim": 0, "descending": True}}'),
        ("stable false", '{"args": (torch.randn(2, 3),), "kw": {"stable": False}}'),
    ],
    "aten.chunk.default": [
        ("even", "(torch.randn(2, 6), 3, 1)"),
        ("uneven", "(torch.randn(2, 5), 3, 1)"),
        ("dim 0", "(torch.randn(4, 3), 2, 0)"),
        ("zero size", "(torch.randn(2, 0), 3, 1)"),
    ],
    "aten.unflatten.int": [
        ("basic", "(torch.randn(2, 6), 1, [2, 3])"),
        ("neg dim and inferred", "(torch.randn(2, 6), -1, [2, -1])"),
        ("transposed", "(torch.randn(2, 6).t(), 0, [2, 3])"),
    ],
    "aten.broadcast_tensors.default": [
        ("broadcast pair", "([torch.randn(1, 3), torch.randn(2, 1)],)"),
        ("same shape", "([torch.randn(2, 3), torch.randn(2, 3)],)"),
    ],
    "aten.is_floating_point.default": [
        ("float32", "(torch.randn(2, 3),)"),
        ("int64", "(torch.randn(2, 3).long(),)"),
    ],
    "aten.std.default": [
        ("unbiased", "(torch.randn(2, 3),)"),
        ("biased", "(torch.randn(2, 3), False)"),
    ],
    "aten.std.dim": [
        ("dim 1", "(torch.randn(2, 3), [1], True, False)"),
        ("keepdim", "(torch.randn(2, 3), [1], False, True)"),
    ],
    "aten.var.default": [
        ("unbiased", "(torch.randn(2, 3),)"),
        ("biased", "(torch.randn(2, 3), False)"),
    ],
    "aten.var.dim": [
        ("dim 1", "(torch.randn(2, 3), [1], True, False)"),
        ("keepdim", "(torch.randn(2, 3), [1], False, True)"),
    ],
    "aten.var_mean.default": [
        ("unbiased", "(torch.randn(2, 3),)"),
        ("biased", "(torch.randn(2, 3), False)"),
    ],
    "aten.var_mean.dim": [
        ("dim 1", "(torch.randn(2, 3), [1], True, False)"),
        ("keepdim", "(torch.randn(2, 3), [1], False, True)"),
    ],
    "aten.where.ScalarOther": [
        ("float scalar", "(torch.randn(2, 3) > 0, torch.randn(2, 3), 0.0)"),
        ("int scalar, float self", "(torch.randn(2, 3) > 0, torch.randn(2, 3), 1)"),
        ("float scalar, int64 self", "(torch.randn(2, 3) > 0, torch.randn(2, 3).long(), 0.5)"),
        ("bool scalar, bool self", "(torch.randn(2, 3) > 0, torch.randn(2, 3) > 0, False)"),
        ("transposed self", "(torch.randn(3, 2) > 0, torch.randn(2, 3).t(), 0.0)"),
    ],
    "aten.where.Scalar": [
        ("float float", "(torch.randn(2, 3) > 0, 1.0, 2.0)"),
        ("int float", "(torch.randn(2, 3) > 0, 1, 2.5)"),
        ("int int", "(torch.randn(2, 3) > 0, 1, 2)"),
        ("bool bool", "(torch.randn(2, 3) > 0, True, False)"),
    ],
    "aten.where.default": [
        ("2d", "(torch.randn(2, 3) > 0,)"),
        ("1d", "(torch.randn(4) > 0,)"),
        ("0d", "(torch.tensor(True),)"),
    ],
    "aten.greater.Tensor": [
        ("float", "(torch.randn(2, 3), torch.randn(2, 3))"),
        ("broadcast", "(torch.randn(2, 3), torch.randn(3))"),
        ("transposed", "(torch.randn(2, 3).t(), torch.randn(3, 2))"),
    ],
    "aten.greater.Scalar": [
        ("float", "(torch.randn(2, 3), 0.5)"),
        ("int", "(torch.randn(2, 3).long(), 0)"),
    ],
    "aten.multiply.Tensor": [
        ("float", "(torch.randn(2, 3), torch.randn(2, 3))"),
        ("broadcast", "(torch.randn(2, 3), torch.randn(3))"),
        ("transposed", "(torch.randn(2, 3).t(), torch.randn(3, 2))"),
    ],
    "aten.multiply.Scalar": [
        ("float", "(torch.randn(2, 3), 2.5)"),
        ("int", "(torch.randn(2, 3).long(), 3)"),
    ],
    "aten.max.other": [
        ("float", "(torch.randn(2, 3), torch.randn(2, 3))"),
        ("broadcast", "(torch.randn(2, 3), torch.randn(3))"),
    ],
    "aten.min.other": [
        ("float", "(torch.randn(2, 3), torch.randn(2, 3))"),
        ("broadcast", "(torch.randn(2, 3), torch.randn(3))"),
    ],
    "aten.adaptive_avg_pool1d.default": [
        ("batched", "(torch.randn(2, 3, 8), [4])"),
        ("unbatched", "(torch.randn(3, 8), [3])"),
        ("to one", "(torch.randn(2, 3, 8), [1])"),
    ],
    "aten.adaptive_avg_pool2d.default": [
        ("batched", "(torch.randn(2, 3, 8, 8), [4, 4])"),
        ("uneven", "(torch.randn(2, 3, 7, 5), [3, 2])"),
        ("global", "(torch.randn(3, 8, 8), [1, 1])"),
    ],
    "aten.max_pool1d.default": [
        ("kernel only", "(torch.randn(2, 3, 8), [2])"),
        ("every argument", "(torch.randn(2, 3, 8), [3], [2], [1], [1], True)"),
    ],
    "aten.max_pool2d.default": [
        ("kernel only", "(torch.randn(2, 3, 8, 8), [2, 2])"),
        ("every argument", "(torch.randn(2, 3, 8, 8), [3, 3], [2, 2], [1, 1], [1, 1], True)"),
        ("stride given", "(torch.randn(1, 2, 6, 6), [2, 2], [1, 1])"),
    ],
    "aten.nanmean.default": [
        ("all", "(torch.tensor([[1.0, float('nan'), 3.0], [4.0, 5.0, float('nan')]]),)"),
        ("dim keepdim", "(torch.tensor([[1.0, float('nan'), 3.0], [4.0, 5.0, 6.0]]), [1], True)"),
        ("dim", "(torch.randn(2, 3), [0])"),
    ],
    "aten.diag.default": [
        ("1d", "(torch.randn(3),)"),
        ("1d offset", "(torch.randn(3), 2)"),
        ("1d negative offset", "(torch.randn(3), -1)"),
        ("2d", "(torch.randn(3, 4),)"),
        ("2d offset", "(torch.randn(3, 4), 1)"),
    ],
    "aten.one_hot.default": [
        ("given classes", "(torch.tensor([0, 2, 1]), 3)"),
        ("inferred classes", "(torch.tensor([[0, 1], [2, 0]]),)"),
        ("empty, given classes", "(torch.tensor([], dtype=torch.long), 4)"),
    ],
    # #374. stft's rows: center with each pad mode (reflect, constant,
    # replicate, circular), a short window (zeros, slice, copy_ or fill_),
    # align_to_window, normalized, two-sided, return_complex False
    # (view_as_real), a 1-D and a 2-D signal, keywords, and the checks.
    "aten.stft.center": [
        ("1d reflect no window", "(torch.randn(64), 16, None, None, None, True, 'reflect', False, None, True)"),
        ("1d hann hop 4", "(torch.randn(64), 16, 4, 16, torch.hann_window(16), True, 'reflect', False, None, True)"),
        ("2d short window constant normalized two-sided",
         "(torch.randn(2, 64), 16, 4, 12, torch.hann_window(12), True, 'constant', True, False, True)"),
        ("2d replicate", "(torch.randn(2, 64), 16, 4, None, None, True, 'replicate', False, None, True)"),
        ("2d circular", "(torch.randn(2, 64), 16, 4, None, None, True, 'circular', False, None, True)"),
        ("2d no center", "(torch.randn(2, 64), 16, 4, None, None, False, 'reflect', False, None, True)"),
        ("1d return_complex False", "(torch.randn(64), 16, None, None, None, True, 'reflect', False, None, False)"),
        ("1d aligned short window no center",
         "(torch.randn(64), 16, 4, 8, None, False, 'reflect', False, None, True, True)"),
        ("float64 normalized", "(torch.randn(64, dtype=torch.float64), 16, 4, None,"
                               " torch.hann_window(16, dtype=torch.float64), True, 'reflect', True, None, True)"),
        ("keywords", '{"args": (torch.randn(64), 16), "kw": {"window": torch.hann_window(16), '
                     '"return_complex": True}}'),
        ("3d refused", "(torch.randn(2, 2, 64), 16, None, None, None, True, 'reflect', False, None, True)"),
        ("return_complex missing refused", "(torch.randn(64), 16, None, None, None, True, 'reflect', False, None, None)"),
        ("align with center refused",
         "(torch.randn(64), 16, 4, None, None, True, 'reflect', False, None, True, True)"),
        ("unknown pad mode refused", "(torch.randn(64), 16, 4, None, None, True, 'bogus', False, None, True)"),
    ],
    "aten.stft.default": [
        ("1d no window", "(torch.randn(64), 16, None, None, None, False, None, True)"),
        ("1d short window", "(torch.randn(64), 16, 4, 8, None, False, None, True)"),
        ("1d short hann aligned", "(torch.randn(64), 16, 4, 8, torch.hann_window(8), False, None, True, True)"),
        ("1d two-sided", "(torch.randn(64), 16, 4, None, None, False, False, True)"),
        ("2d window normalized real output",
         "(torch.randn(2, 64), 16, 4, None, torch.hann_window(16), True, True, False)"),
        ("float64 window normalized", "(torch.randn(64, dtype=torch.float64), 16, 4, None,"
                                      " torch.hann_window(16, dtype=torch.float64), True, None, True)"),
        ("int64 refused", "(torch.arange(64), 16, None, None, None, False, None, True)"),
        ("n_fft too large refused", "(torch.randn(8), 16, None, None, None, False, None, True)"),
        ("win_length too large refused", "(torch.randn(64), 16, 4, 20, None, False, None, True)"),
        ("window size refused", "(torch.randn(64), 16, 4, None, torch.ones(8), False, None, True)"),
        ("hop 0 refused", "(torch.randn(64), 16, 0, None, None, False, None, True)"),
        ("return_complex missing refused", "(torch.randn(64), 16)"),
    ],
    # #374. lstm's rows, float64 so that no platform's upstream takes oneDNN
    # (`_LSTM_FLOAT32` above for float32).
    "aten.lstm.input": [
        ("1 layer", _lstm()),
        ("1 layer zero state", _lstm(zero_state=True)),
        ("one step", _lstm(steps=1)),
        ("batch_first", _lstm(batch_first=True)),
        ("bidirectional", _lstm(bidirectional=True)),
        ("2 layers", _lstm(layers=2)),
        ("2 layers bidirectional batch_first", _lstm(layers=2, bidirectional=True, batch_first=True)),
        ("no bias", _lstm(bias=False)),
        ("no bias bidirectional 2 layers", _lstm(bias=False, bidirectional=True, layers=2)),
        ("projections", _lstm(proj=3)),
        ("projections bidirectional no bias", _lstm(proj=3, bidirectional=True, bias=False)),
        ("dropout eval", _lstm(layers=2, dropout=0.3)),
        ("dropout train", _lstm(layers=2, dropout=0.3, train=True)),
        ("one hidden state refused",
         "(torch.randn(5, 2, 3, dtype=torch.float64), [torch.randn(1, 2, 4, dtype=torch.float64)],"
         " [torch.randn(16, 3, dtype=torch.float64)], True, 1, 0.0, False, False, False)"),
        ("parameter count refused",
         "(torch.randn(5, 2, 3, dtype=torch.float64), [torch.randn(1, 2, 4, dtype=torch.float64),"
         " torch.randn(1, 2, 4, dtype=torch.float64)], [torch.randn(16, 3, dtype=torch.float64)],"
         " True, 1, 0.0, False, False, False)"),
    ] + (_LSTM_FLOAT32 if sys.platform == "darwin" else []),
    "aten.clip.default": [
        ("both", "(torch.randn(2, 3), -0.5, 0.5)"),
        ("min only", "(torch.randn(2, 3), 0.0, None)"),
        ("max only", "(torch.randn(2, 3), None, 0.5)"),
        ("min default", "(torch.randn(2, 3), 0.0)"),
    ],
    "aten.reshape_as.default": [
        ("contiguous", "(torch.randn(2, 3), torch.empty(3, 2))"),
        ("noncontiguous", "(torch.randn(2, 3).t(), torch.empty(6))"),
    ],
    "aten.real.default": [
        ("complex", "(torch.complex(torch.randn(2, 3), torch.randn(2, 3)),)"),
        ("float", "(torch.randn(2, 3),)"),
    ],
    "aten.imag.default": [
        ("complex", "(torch.complex(torch.randn(2, 3), torch.randn(2, 3)),)"),
        ("float refused", "(torch.randn(2, 3),)"),
    ],
    "aten.rms_norm.default": [
        ("last dim", "(torch.randn(2, 3), [3])"),
        ("with weight", "(torch.randn(2, 3), [3], torch.randn(3))"),
        ("with eps", "(torch.randn(2, 3), [3], None, 1e-5)"),
        ("two dims", "(torch.randn(2, 3), [2, 3], torch.randn(2, 3), 1e-6)"),
        ("float16", "(torch.randn(2, 3, dtype=torch.float16), [3])"),
    ],
    "aten.__and__.Tensor": [
        ("bool", "(torch.randn(2, 3) > 0, torch.randn(2, 3) > 0)"),
        ("int32", "(torch.randn(2, 3).int(), torch.randn(2, 3).int())"),
        ("int64", "(torch.randn(2, 3).long(), torch.randn(2, 3).long())"),
        ("broadcast", "(torch.randn(2, 3).int(), torch.randn(3).int())"),
        ("bool and 0-d bool", "(torch.randn(2, 3) > 0, torch.tensor(True))"),
        ("0-d bool and bool", "(torch.tensor(True), torch.randn(2, 3) > 0)"),
        ("transposed bool", "(torch.randn(2, 3).t() > 0, torch.randn(2, 3).t() > 0)"),
    ],
    "aten.diff.default": [
        ("default", "(torch.randn(4, 5),)"),
        ("n 1", "(torch.randn(4, 5), 1)"),
        ("dim 0", "(torch.randn(4, 5), 1, 0)"),
        ("n 2", "(torch.randn(4, 5), 2)"),
        ("n 2 dim 0", "(torch.randn(4, 5), 2, 0)"),
        ("n 3 dim 0", "(torch.randn(4, 5), 3, 0)"),
        ("n = size", "(torch.randn(4, 5), 5, 1)"),
        ("n > size", "(torch.randn(4, 5), 6, 1)"),
        ("n 0", "(torch.randn(4, 5), 0)"),
        ("n 0 bad dim", "(torch.randn(4, 5), 0, 2)"),
        ("dim -1 explicit", "(torch.randn(4, 5), 1, -1)"),
        ("1d", "(torch.randn(4), 1, 0)"),
        ("size 1", "(torch.randn(1, 5), 1, 0)"),
        ("size 0 on dim", "(torch.randn(0, 5), 1, 0)"),
        ("size 0 other dim", "(torch.randn(4, 0), 2, 0)"),
        ("int32", "(torch.randn(4, 5).int(), 1)"),
        ("prepend", '{"args": (torch.randn(4, 5), 1, 1), "kw": {"prepend": torch.randn(4, 2)}}'),
        ("append", '{"args": (torch.randn(4, 5), 1, 1), "kw": {"append": torch.randn(4, 2)}}'),
        ("prepend+append", '{"args": (torch.randn(4, 5), 1, 1),'
                           ' "kw": {"prepend": torch.randn(4, 1), "append": torch.randn(4, 2)}}'),
        ("prepend dim 0", '{"args": (torch.randn(4, 5), 1, 0),'
                          ' "kw": {"prepend": torch.randn(1, 5), "append": torch.randn(2, 5)}}'),
        ("prepend dim -1", '{"args": (torch.randn(4, 5), 1, -1), "kw": {"prepend": torch.randn(4, 2)}}'),
        ("prepend dim -2", '{"args": (torch.randn(4, 5), 1, -2), "kw": {"prepend": torch.randn(2, 5)}}'),
        ("prepend n 2", '{"args": (torch.randn(4, 5), 2, 1), "kw": {"prepend": torch.randn(4, 1)}}'),
        ("prepend n 0", '{"args": (torch.randn(4, 5), 0, 1), "kw": {"prepend": torch.randn(4, 2)}}'),
        ("prepend positional", "(torch.randn(4, 5), 1, 1, torch.randn(4, 2))"),
        ("prepend other dtype", '{"args": (torch.randn(4, 5), 1, 1), "kw": {"prepend": torch.randn(4, 2).int()}}'),
        ("prepend on empty dim", '{"args": (torch.randn(0, 5), 1, 0), "kw": {"prepend": torch.randn(1, 5)}}'),
        ("negative n refused", "(torch.randn(4, 5), -1, 2)"),
        ("0-d refused", "(torch.tensor(1.0), 1)"),
        ("bad dim refused", "(torch.randn(4, 5), 1, 2)"),
        ("prepend rank refused", '{"args": (torch.randn(4, 5), 1, 1), "kw": {"prepend": torch.tensor(1.0)}}'),
        ("append rank refused", '{"args": (torch.randn(4, 5), 1, 1), "kw": {"append": torch.randn(4, 2, 1)}}'),
        ("prepend size refused", '{"args": (torch.randn(4, 5), 1, 1), "kw": {"prepend": torch.randn(3, 2)}}'),
        ("prepend 3d size refused",
         '{"args": (torch.randn(2, 3, 4), 1, 1), "kw": {"prepend": torch.randn(3, 3, 5)}}'),
        ("prepend bad dim refused", '{"args": (torch.randn(4, 5), 1, 3), "kw": {"prepend": torch.randn(4, 2)}}'),
        ("n 0 prepend size refused", '{"args": (torch.randn(4, 5), 0, 0), "kw": {"prepend": torch.randn(4, 2)}}'),
    ],
    # to.dtype / to.dtype_layout (#377): `to_impl`. Nothing is logged, and
    # `self` itself comes back (the probe records identity for these two),
    # when dtype, layout and device already hold, `copy` is False and the
    # memory format is preserve or the one `self` suggests; otherwise one
    # `_to_copy` with dtype (always, for to.dtype), non_blocking only when True
    # and memory_format only when given, `copy` never forwarded.
    "aten.to.dtype": [
        ("same dtype is self", "(torch.randn(2, 3), torch.float32)"),
        ("same dtype non_blocking is self", "(torch.randn(2, 3), torch.float32, True)"),
        ("same dtype copy kw", '{"args": (torch.randn(2, 3), torch.float32), "kw": {"copy": True}}'),
        ("same dtype non_blocking copy", "(torch.randn(2, 3), torch.float32, True, True)"),
        ("same dtype preserve is self",
         '{"args": (torch.randn(2, 3), torch.float32), "kw": {"memory_format": torch.preserve_format}}'),
        ("same dtype contiguous is self",
         '{"args": (torch.randn(2, 3), torch.float32), "kw": {"memory_format": torch.contiguous_format}}'),
        ("same dtype transposed contiguous is self",
         '{"args": (torch.randn(2, 3).t(), torch.float32), "kw": {"memory_format": torch.contiguous_format}}'),
        ("same dtype transposed copy", '{"args": (torch.randn(2, 3).t(), torch.float32), "kw": {"copy": True}}'),
        ("same dtype channels_last strides, channels_last is self",
         '{"args": (' + _CL + ', torch.float32), "kw": {"memory_format": torch.channels_last}}'),
        ("same dtype channels_last strides, contiguous copies",
         '{"args": (' + _CL + ', torch.float32), "kw": {"memory_format": torch.contiguous_format}}'),
        ("same dtype 0-d", "(torch.tensor(1.0), torch.float32)"),
        ("float64", "(torch.randn(2, 3), torch.float64)"),
        ("int32", "(torch.randn(2, 3), torch.int32)"),
        ("bool", "(torch.randn(2, 3), torch.bool)"),
        ("float16", "(torch.randn(2, 3), torch.float16)"),
        ("from int32", "(torch.randn(2, 3).int(), torch.float32)"),
        ("non_blocking positional", "(torch.randn(2, 3), torch.float64, True)"),
        ("non_blocking kw", '{"args": (torch.randn(2, 3), torch.float64), "kw": {"non_blocking": True}}'),
        ("non_blocking False", "(torch.randn(2, 3), torch.float64, False)"),
        ("copy kw", '{"args": (torch.randn(2, 3), torch.float64), "kw": {"copy": True}}'),
        ("non_blocking copy positional", "(torch.randn(2, 3), torch.float64, True, True)"),
        ("contiguous", '{"args": (torch.randn(2, 3), torch.float64), "kw": {"memory_format": torch.contiguous_format}}'),
        ("preserve", '{"args": (torch.randn(2, 3), torch.float64), "kw": {"memory_format": torch.preserve_format}}'),
        ("memory_format positional", "(torch.randn(2, 3), torch.float64, False, False, torch.contiguous_format)"),
        ("transposed contiguous",
         '{"args": (torch.randn(2, 3).t(), torch.float64), "kw": {"memory_format": torch.contiguous_format}}'),
        ("0-d", "(torch.tensor(1.0), torch.float64)"),
        ("zero numel", "(torch.randn(0, 3), torch.float64)"),
    ],
    "aten.to.dtype_layout": [
        ("nothing is self", _one()),
        ("dtype same is self", _one("'dtype': torch.float32")),
        ("device same is self", _one("'device': " + _CPU)),
        ("device string same is self", _one("'device': 'cpu'")),
        ("layout strided is self", _one("'layout': torch.strided")),
        ("dtype layout device same is self",
         _one("'dtype': torch.float32, 'layout': torch.strided, 'device': " + _CPU)),
        ("all same is self",
         _one("'dtype': torch.float32, 'layout': torch.strided, 'device': " + _CPU +
              ", 'pin_memory': False, 'non_blocking': False, 'copy': False, 'memory_format': None")),
        ("pin_memory True same dtype is self", _one("'dtype': torch.float32, 'pin_memory': True")),
        ("non_blocking same is self", _one("'dtype': torch.float32, 'non_blocking': True")),
        ("preserve is self", _one("'memory_format': torch.preserve_format")),
        ("contiguous is self", _one("'memory_format': torch.contiguous_format")),
        ("transposed contiguous is self",
         '{"args": (torch.randn(2, 3).t(),), "kw": {"memory_format": torch.contiguous_format}}'),
        ("channels_last strides, channels_last is self",
         '{"args": (' + _CL + ',), "kw": {"memory_format": torch.channels_last}}'),
        ("channels_last strides, contiguous copies",
         '{"args": (' + _CL + ',), "kw": {"memory_format": torch.contiguous_format}}'),
        ("dtype only", _one("'dtype': torch.float64")),
        ("dtype device", _one("'dtype': torch.float64, 'device': " + _CPU)),
        ("dtype device string", _one("'dtype': torch.float64, 'device': 'cpu'")),
        ("dtype layout", _one("'dtype': torch.float64, 'layout': torch.strided")),
        ("dtype pin_memory False", _one("'dtype': torch.float64, 'pin_memory': False")),
        ("dtype pin_memory None", _one("'dtype': torch.float64, 'pin_memory': None")),
        ("dtype non_blocking", _one("'dtype': torch.float64, 'non_blocking': True")),
        ("dtype non_blocking False", _one("'dtype': torch.float64, 'non_blocking': False")),
        ("dtype copy", _one("'dtype': torch.float64, 'copy': True")),
        ("dtype contiguous", _one("'dtype': torch.float64, 'memory_format': torch.contiguous_format")),
        ("dtype preserve", _one("'dtype': torch.float64, 'memory_format': torch.preserve_format")),
        ("dtype transposed contiguous",
         '{"args": (torch.randn(2, 3).t(),), "kw": {"dtype": torch.float64, "memory_format": torch.contiguous_format}}'),
        ("every option, different dtype",
         _one("'dtype': torch.float64, 'layout': torch.strided, 'device': " + _CPU +
              ", 'pin_memory': False, 'non_blocking': True, 'copy': True, 'memory_format': torch.contiguous_format")),
        ("same dtype copy", _one("'dtype': torch.float32, 'copy': True")),
        ("same dtype layout device copy",
         _one("'dtype': torch.float32, 'layout': torch.strided, 'device': " + _CPU + ", 'copy': True")),
        ("device copy", _one("'device': " + _CPU + ", 'copy': True")),
        ("layout copy", _one("'layout': torch.strided, 'copy': True")),
        ("device meta", _one("'device': torch.device('meta')")),
        ("channels_last strides, contiguous, no dtype",
         '{"args": (' + _CL + ',), "kw": {"memory_format": torch.contiguous_format}}'),
        ("transposed contiguous copy",
         '{"args": (torch.randn(2, 3).t(),), "kw": {"memory_format": torch.contiguous_format, "copy": True}}'),
    ],
    # scaled_dot_product_attention (#377). The flash branches (the call logs one
    # `_scaled_dot_product_flash_attention_for_cpu`, a bool mask first becoming
    # additive), the zero-size branch, the math branches (what the math op's
    # own body logs, `matmul` decomposed, dropout as `torch.dropout` decomposes)
    # and the checks that raise. Errors compare by message.
    "aten.scaled_dot_product_attention.default": [
        # Flash.
        ("flash none", _sdpa()),
        ("flash bool mask 2d", _sdpa('"attn_mask": torch.rand(4, 5) > 0.5')),
        ("flash bool mask 4d", _sdpa('"attn_mask": torch.rand(2, 3, 4, 5) > 0.5')),
        ("flash bool mask (L,1)", _sdpa('"attn_mask": torch.rand(4, 1) > 0.5')),
        ("flash float mask 2d", _sdpa('"attn_mask": torch.randn(4, 5)')),
        ("flash float mask 4d", _sdpa('"attn_mask": torch.randn(2, 3, 4, 5)')),
        ("flash float mask (L,1)", _sdpa('"attn_mask": torch.randn(4, 1)')),
        ("flash float mask (1,1,L,S)", _sdpa('"attn_mask": torch.randn(1, 1, 4, 5)')),
        ("flash float mask (B,H,1,S)", _sdpa('"attn_mask": torch.randn(2, 3, 1, 5)')),
        ("flash float mask transposed", _sdpa('"attn_mask": torch.randn(5, 4).t()')),
        ("flash causal", _sdpa('"is_causal": True')),
        ("flash causal L>S", _sdpa('"is_causal": True', _qkv(rows=6))),
        ("flash causal + float mask", _sdpa('"is_causal": True, "attn_mask": torch.randn(4, 5)')),
        ("flash causal + bool mask", _sdpa('"is_causal": True, "attn_mask": torch.rand(4, 5) > 0.5')),
        ("flash dropout 0.0 explicit", _sdpa('"dropout_p": 0.0')),
        ("flash scale", _sdpa('"scale": 0.3')),
        ("flash scale negative", _sdpa('"scale": -0.3')),
        ("flash scale + mask + causal", _sdpa('"scale": 0.3, "is_causal": True, "attn_mask": torch.randn(4, 5)')),
        ("flash gqa", _sdpa('"enable_gqa": True', _qkv(heads=4, kv_heads=2))),
        ("flash gqa same heads", _sdpa('"enable_gqa": True')),
        ("flash gqa mask 4d", _sdpa('"enable_gqa": True, "attn_mask": torch.randn(2, 4, 4, 5)',
                                    _qkv(heads=4, kv_heads=2))),
        ("flash gqa bool mask", _sdpa('"enable_gqa": True, "attn_mask": torch.rand(4, 5) > 0.5',
                                      _qkv(heads=4, kv_heads=2))),
        ("flash gqa causal", _sdpa('"enable_gqa": True, "is_causal": True', _qkv(heads=4, kv_heads=2))),
        ("flash float64", _sdpa("", _qkv("torch.float64"))),
        ("flash float64 bool mask", _sdpa('"attn_mask": torch.rand(4, 5) > 0.5', _qkv("torch.float64"))),
        ("flash float16", _sdpa("", _qkv("torch.float16"))),
        ("flash float16 bool mask", _sdpa('"attn_mask": torch.rand(4, 5) > 0.5', _qkv("torch.float16"))),
        ("flash float16 float32 mask", _sdpa('"attn_mask": torch.randn(4, 5)', _qkv("torch.float16"))),
        ("flash float16 causal", _sdpa('"is_causal": True', _qkv("torch.float16"))),
        ("flash bfloat16", _sdpa("", _qkv("torch.bfloat16"))),
        ("flash bfloat16 bool mask", _sdpa('"attn_mask": torch.rand(4, 5) > 0.5', _qkv("torch.bfloat16"))),
        ("flash strided heads",
         _sdpa("", "torch.randn(2, 4, 3, 8).transpose(1, 2), torch.randn(2, 5, 3, 8).transpose(1, 2),"
                   " torch.randn(2, 5, 3, 8).transpose(1, 2)")),
        ("flash size-1 head dim", _sdpa("", "torch.randn(2, 3, 4, 1), torch.randn(2, 3, 5, 1),"
                                            " torch.randn(2, 3, 5, 1)")),
        # Zero-size inputs: `zeros` plus `(sum(q) + sum(k) + sum(v)) * 0`.
        ("zero query rows", _sdpa("", _qkv(rows=0))),
        ("zero keys", _sdpa("", "torch.randn(2, 3, 4, 8), torch.randn(2, 3, 0, 8), torch.randn(2, 3, 0, 8)")),
        ("zero batch", _sdpa("", _qkv(batch=0))),
        ("zero head dim", _sdpa("", "torch.randn(2, 3, 4, 0), torch.randn(2, 3, 5, 0), torch.randn(2, 3, 5, 0)")),
        ("zero value head dim", _sdpa("", "torch.randn(2, 3, 4, 8), torch.randn(2, 3, 5, 8),"
                                          " torch.randn(2, 3, 5, 0)")),
        ("zero keys causal", _sdpa('"is_causal": True', "torch.randn(2, 3, 4, 8), torch.randn(2, 3, 0, 8),"
                                                        " torch.randn(2, 3, 0, 8)")),
        ("zero query rows float16", _sdpa("", _qkv("torch.float16", rows=0))),
        ("zero query rows 3d", _sdpa("", "torch.randn(2, 0, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)")),
        ("zero query rows dropout", _sdpa('"dropout_p": 0.3', _qkv(rows=0))),
        # Math, because dropout_p > 0 (train and eval are the caller's: only
        # the number matters).
        ("math dropout", _sdpa('"dropout_p": 0.3')),
        ("math dropout 1.0", _sdpa('"dropout_p": 1.0')),
        ("math dropout bool mask", _sdpa('"dropout_p": 0.3, "attn_mask": torch.rand(4, 5) > 0.5')),
        ("math dropout float mask", _sdpa('"dropout_p": 0.3, "attn_mask": torch.randn(4, 5)')),
        ("math dropout causal", _sdpa('"dropout_p": 0.3, "is_causal": True')),
        ("math dropout scale", _sdpa('"dropout_p": 0.3, "scale": 0.2')),
        ("math dropout scale negative", _sdpa('"dropout_p": 0.3, "scale": -0.2')),
        ("math dropout scale zero", _sdpa('"dropout_p": 0.3, "scale": 0.0')),
        ("math dropout gqa", _sdpa('"dropout_p": 0.3, "enable_gqa": True', _qkv(heads=4, kv_heads=2))),
        ("math dropout gqa same heads", _sdpa('"dropout_p": 0.3, "enable_gqa": True')),
        ("math dropout gqa causal", _sdpa('"dropout_p": 0.3, "enable_gqa": True, "is_causal": True',
                                          _qkv(heads=4, kv_heads=2))),
        ("math dropout float64", _sdpa('"dropout_p": 0.3', _qkv("torch.float64"))),
        ("math dropout float16", _sdpa('"dropout_p": 0.3', _qkv("torch.float16"))),
        ("math dropout float16 bool mask",
         _sdpa('"dropout_p": 0.3, "attn_mask": torch.rand(4, 5) > 0.5', _qkv("torch.float16"))),
        ("math dropout float16 float16 mask",
         _sdpa('"dropout_p": 0.3, "attn_mask": torch.randn(4, 5, dtype=torch.float16)', _qkv("torch.float16"))),
        ("math dropout float16 causal", _sdpa('"dropout_p": 0.3, "is_causal": True', _qkv("torch.float16"))),
        ("math dropout bfloat16", _sdpa('"dropout_p": 0.3', _qkv("torch.bfloat16"))),
        # Math, because flash is not eligible.
        ("math 3d", _sdpa("", "torch.randn(3, 4, 8), torch.randn(3, 5, 8), torch.randn(3, 5, 8)")),
        ("math 3d bool mask", _sdpa('"attn_mask": torch.rand(4, 5) > 0.5',
                                    "torch.randn(3, 4, 8), torch.randn(3, 5, 8), torch.randn(3, 5, 8)")),
        ("math 3d float mask", _sdpa('"attn_mask": torch.randn(4, 5)',
                                     "torch.randn(3, 4, 8), torch.randn(3, 5, 8), torch.randn(3, 5, 8)")),
        ("math 3d causal", _sdpa('"is_causal": True',
                                 "torch.randn(3, 4, 8), torch.randn(3, 5, 8), torch.randn(3, 5, 8)")),
        ("math 3d scale", _sdpa('"scale": 0.5', "torch.randn(3, 4, 8), torch.randn(3, 5, 8), torch.randn(3, 5, 8)")),
        ("math 3d gqa", _sdpa('"enable_gqa": True', "torch.randn(4, 4, 8), torch.randn(2, 5, 8), torch.randn(2, 5, 8)")),
        ("math 3d float16", _sdpa("", "torch.randn(3, 4, 8, dtype=torch.float16), torch.randn(3, 5, 8, dtype=torch.float16),"
                                      " torch.randn(3, 5, 8, dtype=torch.float16)")),
        ("math 2d", _sdpa("", "torch.randn(4, 8), torch.randn(5, 8), torch.randn(5, 8)")),
        ("math 2d bool mask float64", _sdpa('"attn_mask": torch.rand(4, 5) > 0.5',
                                            "torch.randn(4, 8, dtype=torch.float64), torch.randn(5, 8, dtype=torch.float64),"
                                            " torch.randn(5, 8, dtype=torch.float64)")),
        ("math 5d", _sdpa("", "torch.randn(2, 2, 3, 4, 8), torch.randn(2, 2, 3, 5, 8), torch.randn(2, 2, 3, 5, 8)")),
        ("math one kv head, no gqa", _sdpa("", _qkv(kv_heads=1))),
        ("math one kv head float mask", _sdpa('"attn_mask": torch.randn(4, 5)', _qkv(kv_heads=1))),
        ("math head dim differs from value's", _sdpa("", "torch.randn(2, 3, 4, 8), torch.randn(2, 3, 5, 8),"
                                                         " torch.randn(2, 3, 5, 6)")),
        ("math key batch 1", _sdpa("", "torch.randn(2, 3, 4, 8), torch.randn(1, 3, 5, 8), torch.randn(1, 3, 5, 8)")),
        ("math mask (H,L,S)", _sdpa('"attn_mask": torch.randn(3, 4, 5)')),
        ("math mask (1,L,S)", _sdpa('"attn_mask": torch.randn(1, 4, 5)')),
        ("math head dim stride not 1", _sdpa("", "torch.randn(2, 3, 8, 4).transpose(-1, -2), torch.randn(2, 3, 5, 8),"
                                                 " torch.randn(2, 3, 5, 8)")),
        ("math key stride not 1", _sdpa("", "torch.randn(2, 3, 4, 8), torch.randn(2, 3, 8, 5).transpose(-1, -2),"
                                            " torch.randn(2, 3, 5, 8)")),
        ("math float16 3d scale negative",
         _sdpa('"scale": -0.4', "torch.randn(3, 4, 8, dtype=torch.float16), torch.randn(3, 5, 8, dtype=torch.float16),"
                                " torch.randn(3, 5, 8, dtype=torch.float16)")),
        # The checks that raise (message only).
        ("dtype mismatch refused", _sdpa("", "torch.randn(2, 3, 4, 8), torch.randn(2, 3, 5, 8, dtype=torch.float64),"
                                              " torch.randn(2, 3, 5, 8)")),
        ("half and bfloat16 mismatch refused", _sdpa("", "torch.randn(2, 3, 4, 8, dtype=torch.float16),"
                                                         " torch.randn(2, 3, 5, 8, dtype=torch.bfloat16),"
                                                         " torch.zeros(2, 3, 5, 8, dtype=torch.int32)")),
        ("1d refused", _sdpa("", "torch.randn(8), torch.randn(8), torch.randn(8)")),
        ("one operand 1d refused", _sdpa("", "torch.randn(2, 3, 4, 8), torch.randn(8), torch.randn(2, 3, 5, 8)")),
        ("int64 mask refused", _sdpa('"attn_mask": torch.randint(0, 2, (4, 5))')),
        ("float64 mask on float32 refused", _sdpa('"attn_mask": torch.randn(4, 5, dtype=torch.float64)')),
        ("float16 mask on float32 refused", _sdpa('"attn_mask": torch.randn(4, 5, dtype=torch.float16)')),
        ("1d mask refused", _sdpa('"attn_mask": torch.randn(5)')),
        ("causal with mask and dropout refused", _sdpa('"dropout_p": 0.3, "is_causal": True, "attn_mask": torch.randn(4, 5)')),
        ("gqa not divisible refused", _sdpa('"enable_gqa": True', _qkv(heads=3, kv_heads=2))),
        ("gqa fewer query heads refused", _sdpa('"enable_gqa": True', _qkv(heads=2, kv_heads=4))),
        ("heads mismatch without gqa refused", _sdpa("", _qkv(heads=4, kv_heads=2))),
        ("batch mismatch refused", _sdpa("", "torch.randn(2, 3, 4, 8), torch.randn(3, 3, 5, 8), torch.randn(3, 3, 5, 8)")),
        ("dropout above 1 refused", _sdpa('"dropout_p": 1.5')),
    ],
    "aten._scaled_dot_product_attention_math.default": [
        ("none", _sdpa()),
        ("float mask", _sdpa('"attn_mask": torch.randn(4, 5)')),
        ("float mask 4d", _sdpa('"attn_mask": torch.randn(2, 3, 4, 5)')),
        ("causal", _sdpa('"is_causal": True')),
        ("causal + mask refused", _sdpa('"is_causal": True, "attn_mask": torch.randn(4, 5)')),
        ("scale", _sdpa('"scale": 0.3')),
        ("scale negative", _sdpa('"scale": -0.3')),
        ("gqa", _sdpa('"enable_gqa": True', _qkv(heads=4, kv_heads=2))),
        ("gqa same heads", _sdpa('"enable_gqa": True')),
        ("gqa not divisible refused", _sdpa('"enable_gqa": True', _qkv(heads=3, kv_heads=2))),
        ("dropout", _sdpa('"dropout_p": 0.3')),
        ("dropout float mask", _sdpa('"dropout_p": 0.3, "attn_mask": torch.randn(4, 5)')),
        ("dropout mask", _sdpa('"dropout_p": 0.3, "dropout_mask": torch.rand(2, 3, 4, 5) > 0.5')),
        ("3d", _sdpa("", "torch.randn(3, 4, 8), torch.randn(3, 5, 8), torch.randn(3, 5, 8)")),
        ("2d", _sdpa("", "torch.randn(4, 8), torch.randn(5, 8), torch.randn(5, 8)")),
        ("float64", _sdpa("", _qkv("torch.float64"))),
        ("float16", _sdpa("", _qkv("torch.float16"))),
        ("float16 float32 mask", _sdpa('"attn_mask": torch.randn(4, 5)', _qkv("torch.float16"))),
        ("float16 causal", _sdpa('"is_causal": True', _qkv("torch.float16"))),
        ("bfloat16 dropout", _sdpa('"dropout_p": 0.3', _qkv("torch.bfloat16"))),
        ("zero query rows", _sdpa("", _qkv(rows=0))),
        ("zero keys", _sdpa("", "torch.randn(2, 3, 4, 8), torch.randn(2, 3, 0, 8), torch.randn(2, 3, 0, 8)")),
        ("float16 zero query rows", _sdpa("", _qkv("torch.float16", rows=0))),
    ],
}

#: Branches whose upstream sequence the body reproduces, but whose emitted
#: ops include one this shim has no kernel for (aten.rs `IMPLEMENTED`): the
#: test asserts upstream's measured result and that the shim refuses naming
#: that op, and nothing else. The day the kernel lands these fail and move to
#: `CASES`. Each row: (label, expression, the op the refusal must name).
NO_KERNEL = {
    "aten.matmul.default": [
        ("1d@1d", "(torch.randn(4), torch.randn(4))", "dot"),
        ("2d@1d", "(torch.randn(3, 4), torch.randn(4))", "mv"),
        ("3d@1d fold", "(torch.randn(2, 3, 4), torch.randn(4))", "mv"),
        ("4d@1d fold", "(torch.randn(2, 3, 4, 5), torch.randn(5))", "mv"),
        ("1d grad@3d fold", "(torch.randn(5, requires_grad=True), torch.randn(2, 5, 6))", "mv"),
    ],
    "aten.diff.default": [
        ("bool", "(torch.randn(4) > 0,)", "logical_xor"),
        ("bool n 2", "(torch.randn(4) > 0, 2, 0)", "logical_xor"),
    ],
}

#: Branches whose upstream sequence the body reproduces (the flash op is
#: emitted, as upstream's body emits it) and whose kernel this shim refuses BY
#: NAME on purpose, not for want of a kernel (that is `NO_KERNEL`). The test
#: asserts upstream's measured sequence contains the op, and that the shim's
#: error is a `NotImplementedError` naming it. Each row: (label, expression,
#: the op the refusal must name, further text it must carry within the first
#: 300 characters, which is all a row keeps).
#: The row below: a float32 mask on a float64 query. Upstream's flash kernel
#: runs it and is wrong in a way that depends on the CPU's vector width
#: (`tests/_support/test_shim.py`, the test that records it); the shim refuses
#: (leader decision 2026-11-16, not the owner's words). The day upstream
#: agrees with the reference and the refusal is lifted, the row moves back to
#: `CASES`.
REFUSED_BY_NAME = {
    "aten.scaled_dot_product_attention.default": [
        ("float32 mask on float64 refused",
         _sdpa('"attn_mask": torch.randn(4, 5)', _qkv("torch.float64")),
         "aten._scaled_dot_product_flash_attention_for_cpu",
         ("NotImplementedError", "float32 attn_mask", "float64 query")),
    ],
}

#: The public spelling that reaches each op, where one exists.
PUBLIC = {
    "aten.linear.default": "torch.nn.functional.linear",
    "aten.scaled_dot_product_attention.default": "torch.nn.functional.scaled_dot_product_attention",
}

#: Ops whose rows also record whether the result IS the first argument: `to`
#: returns `self` itself (nothing logged) when it has nothing to do, and that
#: identity is part of upstream's contract (`y = x.to(x.dtype); y.fill_(0)`
#: changes `x`).
IDENTITY = ("aten.to.dtype", "aten.to.dtype_layout")


_SEQ_PROBE = r'''
import json, sys
import torch
IS_SHIM = hasattr(torch._C, "_aten_implemented")
assert IS_SHIM == (sys.argv[1] == "shim"), ("shim" if IS_SHIM else "upstream", sys.argv[1])  # AGENTS.md 15.2
from torch.utils._python_dispatch import TorchDispatchMode

CIA = torch._C.DispatchKey.CompositeImplicitAutograd
groups = json.loads(sys.argv[2])
public = json.loads(sys.argv[3])
identity = json.loads(sys.argv[4])


def desc(a):
    if isinstance(a, torch.Tensor):
        return ["T", list(a.shape), bool(a.is_contiguous()), str(a.dtype)]
    if isinstance(a, (list, tuple)):
        return [desc(x) for x in a]
    if a is None or isinstance(a, (bool, int, float, str)):
        return a
    return str(a)


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (list, tuple)):
        return [t for item in value for t in tensors(item)]
    if isinstance(value, dict):
        return [t for item in value.values() for t in tensors(item)]
    return []


class Log(TorchDispatchMode):
    """The log, and beside it the data flow between its steps: `deps[i]` is
    the steps whose output tensor (by object identity) step `i` consumes,
    `mut[i]` whether step `i` writes an argument (its schema says so, or it
    returns one of its argument objects). Every tensor seen is kept alive, so
    an `id` cannot be reused within one row."""

    def __init__(self):
        super().__init__()
        self.log = []
        self.deps = []
        self.mut = []
        self._producer = {}
        self._keep = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        ins = tensors(args) + tensors(kwargs)
        out = func(*args, **kwargs)
        outs = tensors(out)
        step = len(self.log)
        self.log.append([str(func), [desc(a) for a in args],
                         {k: desc(v) for k, v in sorted(kwargs.items())}, desc(out)])
        self.deps.append(sorted({self._producer[id(t)] for t in ins if id(t) in self._producer}))
        try:
            writes = any(a.alias_info is not None and a.alias_info.is_write
                         for a in func._schema.arguments)
        except Exception:  # noqa: BLE001 - unknown is treated as a write
            writes = True
        self.mut.append(bool(writes or any(o is t for o in outs for t in ins)))
        for t in outs:
            self._producer[id(t)] = step
        self._keep.extend(ins + outs)
        return out


def overload(key):
    ns, name, ov = key.split(".", 2)
    return getattr(getattr(getattr(torch.ops, ns), name), ov)


out = {}
flow = {}
for group, table in groups.items():
    for key, cases in table.items():
        op = overload(key)
        for label, expr in cases:
            entries = {"op_dk": lambda a, kw: op._op_dk(CIA, *a, **kw),
                       "call": lambda a, kw: op(*a, **kw)}
            if key in public:
                fn = eval(public[key])
                entries["public"] = lambda a, kw, fn=fn: fn(*a, **kw)
            for how, call in entries.items():
                torch.manual_seed(0)
                args = eval(expr)
                kw = {}
                if isinstance(args, dict):
                    args, kw = args["args"], args["kw"]
                row = {}
                try:
                    with Log() as mode:
                        result = call(args, kw)
                    row["log"] = mode.log
                    row["result"] = desc(result)
                    if key in identity:
                        row["is_self"] = bool(result is args[0])
                    # Kept apart from the row, so the row's exact comparison
                    # with upstream is what it was before this was recorded.
                    flow[f"{group}|{key}|{label}|{how}"] = {"deps": mode.deps, "mut": mode.mut}
                except Exception as exc:  # noqa: BLE001
                    row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                out[f"{group}|{key}|{label}|{how}"] = row
print(json.dumps({"is_shim": IS_SHIM, "out": out, "flow": flow}))
'''


_TABLE_PROBE = r'''
import json, sys
import torch
IS_SHIM = hasattr(torch._C, "_aten_implemented")
assert IS_SHIM == (sys.argv[1] == "shim"), ("shim" if IS_SHIM else "upstream", sys.argv[1])  # AGENTS.md 15.2
names = json.loads(sys.argv[2])
CIA = torch._C.DispatchKey.CompositeImplicitAutograd
out = {"has_cia": {}}
for name in names:
    try:
        out["has_cia"][name] = bool(torch._C._dispatch_has_kernel_for_dispatch_key(name, CIA))
    except Exception as exc:  # noqa: BLE001
        out["has_cia"][name] = f"{type(exc).__name__}: {exc}"
if IS_SHIM:
    out["table"] = torch._C._shim_cia_table()
    out["armed"] = list(torch._C._shim_armed_cia())
    out["native"] = sorted(torch._C._aten_all_implemented())
else:
    out["upstream_registrations"] = sorted(
        torch._C._dispatch_get_registrations_for_dispatch_key("CompositeImplicitAutograd"))
print(json.dumps({"is_shim": IS_SHIM, "out": out}))
'''


_SHIM_PROBE = r'''
import json, sys
import torch
IS_SHIM = hasattr(torch._C, "_aten_implemented")
assert IS_SHIM, "this probe runs on the shim only"  # AGENTS.md 15.2
from torch.utils._python_dispatch import TorchDispatchMode

CIA = torch._C.DispatchKey.CompositeImplicitAutograd
aten = torch.ops.aten
F = torch.nn.functional
out = {}


class Log(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.log = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.log.append(str(func))
        return func(*args, **(kwargs or {}))


def counters():
    return torch._C._shim_cia_counters().get("aten.linear.default", {"door": 0, "backend": 0, "op_dk": 0})


def delta(before):
    after = counters()
    return {k: after.get(k, 0) - before.get(k, 0) for k in ("door", "backend", "op_dk")}


def refusal(fn):
    try:
        fn()
    except NotImplementedError as exc:
        return str(exc)
    except Exception as exc:  # noqa: BLE001
        return f"WRONG {type(exc).__name__}: {exc}"
    return None


x, w, b = torch.randn(2, 8), torch.randn(4, 8), torch.randn(4)

# Where the body ran, by entry.
c = counters(); F.linear(x, w, b); out["eager"] = delta(c)
c = counters(); aten.linear.default._op_dk(CIA, x, w, b); out["op_dk"] = delta(c)
c = counters()
with Log() as m:
    F.linear(x, w, b)
out["mode"] = {"log": m.log, "counters": delta(c)}
c = counters()
with torch.inference_mode():
    with Log() as m:
        F.linear(x, w, b)
out["inference_mode"] = {"log": m.log, "counters": delta(c)}
c = counters()
with torch.inference_mode():
    F.linear(x, w, b)
out["inference_mode_no_mode"] = delta(c)
c = counters()
with torch._C._AutoDispatchBelowAutograd():
    with Log() as m:
        F.linear(x, w, b)
out["below_autograd"] = {"log": m.log, "counters": delta(c)}
c = counters()
r = aten.linear.default.decompose(x, w, b)
out["decompose"] = {"delta": delta(c), "equal_to_eager": bool(torch.equal(r, F.linear(x, w, b)))}

# Refusals by name.
stand_in = sys.argv[2]
ns_, name_, ov_ = stand_in.split(".", 2)
stand_in_op = getattr(getattr(torch.ops.aten, name_), ov_)
out["refuse_uncovered"] = refusal(lambda: stand_in_op._op_dk(CIA, torch.randn(2, 2), torch.randn(2, 2)))
out["refuse_not_cia"] = refusal(lambda: aten.t.default._op_dk(CIA, torch.randn(2, 3)))
out["decompose_uncovered"] = repr(stand_in_op.decompose(torch.randn(2, 2), torch.randn(2, 2)))[:40]
out["stand_in_native"] = stand_in in set(torch._C._aten_all_implemented())
out["pending"] = torch._C._shim_cia_table()["pending"]
# The former pending example now has a body: `_op_dk` runs it.
out["stft_op_dk"] = list(aten.stft.default._op_dk(CIA, torch.randn(64), 16,
                                                   return_complex=True).shape)

# Nullification: without the body armed, the door has no kernel for linear.
was = torch._C._shim_disarm_cia("aten.linear.default")
out["disarmed_was_armed"] = bool(was)
out["disarmed"] = refusal(lambda: F.linear(x, w, b))
torch._C._shim_arm_cia("aten.linear.default")
c = counters(); F.linear(x, w, b); out["rearmed"] = delta(c)
out["arm_non_aten"] = refusal(lambda: torch._C._shim_arm_cia("tnpl.f.default"))
print(json.dumps({"is_shim": IS_SHIM, "out": out}))
'''


_FACTS_PROBE = r'''
import json, sys
import torch
assert not hasattr(torch._C, "_aten_implemented")  # upstream only, AGENTS.md 15.2
from torch.utils._python_dispatch import TorchDispatchMode
from torch._dispatch.python import enable_python_dispatcher
from torch._decomp import decomposition_table
import torch._refs

CIA = torch._C.DispatchKey.CompositeImplicitAutograd
CEA = torch._C.DispatchKey.CompositeExplicitAutograd
aten = torch.ops.aten
F = torch.nn.functional


class Log(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.log = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.log.append(str(func))
        return func(*args, **(kwargs or {}))


out = {"cia": {}, "cea": {}}
for name in ("aten::linear", "aten::t", "aten::addmm", "aten::relu", "aten::permute",
             "aten::transpose.int", "aten::matmul", "aten::reshape", "aten::to.dtype"):
    out["cia"][name] = torch._C._dispatch_has_kernel_for_dispatch_key(name, CIA)
    out["cea"][name] = torch._C._dispatch_has_kernel_for_dispatch_key(name, CEA)
out["linear_in_decomposition_table"] = aten.linear.default in decomposition_table
out["t_decomposition"] = getattr(decomposition_table.get(aten.t.default), "__module__", None)
out["refs_has_linear"] = hasattr(torch._refs, "linear")
out["linear_py_kernel_cia"] = CIA in aten.linear.default.py_kernels
x, w, b = torch.randn(2, 8), torch.randn(4, 8), torch.randn(4)
out["decompose_not_implemented"] = {
    "t": aten.t.default.decompose(w) is NotImplemented,
    "addmm": aten.addmm.default.decompose(b, x, w.t()) is NotImplemented,
    "relu": aten.relu.default.decompose(x) is NotImplemented,
}
logs = {}
with Log() as m:
    F.linear(x, w, b)
logs["mode"] = m.log
with torch.inference_mode():
    with Log() as m:
        F.linear(x, w, b)
logs["inference_mode"] = m.log
with torch._C._AutoDispatchBelowAutograd():
    with Log() as m:
        F.linear(x, w, b)
logs["below_autograd"] = m.log
with enable_python_dispatcher():
    with Log() as m:
        F.linear(x, w, b)
logs["python_dispatcher_no_override"] = m.log
out["logs"] = logs
print(json.dumps({"is_shim": False, "out": out}))
'''

#: Upstream torch 2.13.0, measured 2026-11-16 (the `.caches/spike-venv` oracle).
#: `aten.t.default._op_dk(CompositeImplicitAutograd, x)` is not in it: it
#: crashes the upstream process (exit 139), so it is a fact recorded in
#: docs/design/DECOMPOSE.md and not re-run.
UPSTREAM_FACTS = {
    "cia": {"aten::linear": True, "aten::t": False, "aten::addmm": False, "aten::relu": False,
            "aten::permute": False, "aten::transpose.int": False, "aten::matmul": True,
            "aten::reshape": True, "aten::to.dtype": True},
    "cea": {"aten::linear": False, "aten::t": True, "aten::addmm": False, "aten::relu": False,
            "aten::permute": True, "aten::transpose.int": True, "aten::matmul": False,
            "aten::reshape": False, "aten::to.dtype": False},
    "linear_in_decomposition_table": False,
    "t_decomposition": "torch._refs",
    "refs_has_linear": False,
    "linear_py_kernel_cia": False,
    "decompose_not_implemented": {"t": True, "addmm": True, "relu": True},
    "logs": {"mode": ["aten.t.default", "aten.addmm.default"],
             "inference_mode": ["aten.linear.default"],
             "below_autograd": ["aten.linear.default"],
             "python_dispatcher_no_override": ["aten.t.default", "aten.addmm.default"]},
}

_cache = {}


def _seq(shim):
    key = ("seq", shim)
    if key not in _cache:
        groups = {"cases": CASES,
                  "nokernel": {k: [(label, expr) for label, expr, _ in rows]
                               for k, rows in NO_KERNEL.items()},
                  "refused": {k: [(row[0], row[1]) for row in rows]
                              for k, rows in REFUSED_BY_NAME.items()},
                  "onednn": {"aten.lstm.input": _LSTM_FLOAT32}}
        got = _run(_SEQ_PROBE, shim, json.dumps(groups), json.dumps(PUBLIC),
                   json.dumps(list(IDENTITY)))
        _cache[key] = got["out"]
        _cache[("flow", shim)] = got["flow"]
    return _cache[key]


def _flow(shim):
    """Per row, the data flow `_SEQ_PROBE`'s `Log` recorded beside the log."""
    _seq(shim)
    return _cache[("flow", shim)]


def _step_signatures(log, deps):
    """Per step of one recorded log, a signature: the step itself, how many
    steps consume it, and the sorted signatures of its producers, all the way
    down; None when `deps` is not a recorded order.

    Steps are matched by what they are AND by what they consume: matching
    "the earliest equal step" was not enough. With three scalar `sum` steps
    that print alike, it paired upstream's first `sum` with the shim's first
    `sum` although gcc had evaluated the operands of `a + b + c` right to
    left, and then found no `add` whose producers were placed (CI run
    43819391187: 24 Linux rows, the zero-size attention path, every one
    `sum, add` against `add, sum`). Two steps with one signature are
    interchangeable: nothing recorded tells them apart on either side.

    A signature is a SHA-256 digest of that triple, not the triple's text.
    The text nested every producer's whole text, so its length doubled with
    each step that two later steps consume: a 2-layer bidirectional
    `lstm.input` row (300 steps) needed about 1.2e12 bytes, and CI run
    45131459631's Linux gate killed test_cia_frame at 14 GB. Equal digests
    stand for equal triples (a collision is not a practical concern)."""
    users = [0] * len(log)
    for producers in deps:
        for p in producers:
            users[p] += 1
    out = []
    for index, step in enumerate(log):
        if any(p >= index for p in deps[index]):
            return None                      # not a recorded order
        text = json.dumps([step, users[index], sorted(out[p] for p in deps[index])],
                          sort_keys=True)
        out.append(hashlib.sha256(text.encode()).hexdigest())
    return out


def _reordering(want, got, want_flow, got_flow):
    """How many steps moved, if `got` is `want` with independent steps
    reordered and nothing else changed; None otherwise.

    The rule, checked on the recorded data flow (tensor object identity in the
    probe, not inferred from shapes): every step of `got` pairs with one step
    of `want` that is equal to it, is consumed by as many steps, and consumes
    producers that pair the same way, all the way down (so each side's own
    order is a topological order of one and the same dependency graph, and a
    side whose identities were lost, which has no edges, pairs with nothing
    that has any); and a step that writes an argument is a barrier nothing crosses,
    because a reader moved past a writer reads another value and the
    producer edges alone do not show that. Everything outside the log
    (`result`, `is_self`) must be equal.
    """
    if not (isinstance(want, dict) and isinstance(got, dict)):
        return None
    if "log" not in want or "log" not in got or want_flow is None or got_flow is None:
        return None
    if {k: v for k, v in want.items() if k != "log"} != {k: v for k, v in got.items() if k != "log"}:
        return None
    up, mine = want["log"], got["log"]
    up_deps, up_mut = want_flow["deps"], want_flow["mut"]
    my_deps, my_mut = got_flow["deps"], got_flow["mut"]
    if not (len(up) == len(mine) == len(up_deps) == len(my_deps) == len(up_mut) == len(my_mut)):
        return None
    up_sig, my_sig = _step_signatures(up, up_deps), _step_signatures(mine, my_deps)
    if up_sig is None or my_sig is None or sorted(up_sig) != sorted(my_sig):
        return None
    placed, order = set(), []
    for k in range(len(mine)):
        pick = next((i for i in range(len(up)) if i not in placed and up_sig[i] == my_sig[k]), None)
        if pick is None:
            return None
        placed.add(pick)
        order.append(pick)
    for k, i in enumerate(order):
        if (up_mut[i] or my_mut[k]) and (i != k or set(order[:k]) != set(range(k))):
            return None
    return sum(1 for k, i in enumerate(order) if i != k)


#: The C++ spellings of one type that upstream's messages print through
#: `c10::util::get_fully_qualified_type_name` (`__PRETTY_FUNCTION__`), which
#: depend on the compiler and the platform's `int64_t`: clang on macOS prints
#: `long long` (measured on 2.13.0, 2026-11-16: "attn_mask.dtype: long long
#: and  query.dtype: float instead."), and the Linux wheel's own text differed
#: from that in CI run 43802210944 (not shown in that log).
_CPP_TYPE_SPELLINGS = re.compile(
    r"\b(?:long long int|long long|long int|long|__int64|short int|short)\b")


def _same_but_cpp_type_names(want, got):
    """Both rows raised and the messages differ only where a C++ type name
    is spelled: compared with each spelling replaced by one token, so the
    rest of the message (the check, its order, the other dtype) still has to
    be upstream's character for character."""
    if not (isinstance(want, dict) and isinstance(got, dict)):
        return False
    if set(want) != {"error"} or set(got) != {"error"} or want == got:
        return False
    return (_CPP_TYPE_SPELLINGS.sub("<c++ type>", want["error"])
            == _CPP_TYPE_SPELLINGS.sub("<c++ type>", got["error"]))


def _shim_only():
    if "shim_only" not in _cache:
        _cache["shim_only"] = _run(_SHIM_PROBE, True, _UNCOVERED_STAND_IN)["out"]
    return _cache["shim_only"]


def _need_both():
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return False
    if not _have_upstream():
        _skip.skip("upstream torch not importable without the vendored tree")
        return False
    return True



#: See the pending check in the table test (issue #374). Empty since the
#: last three bodies (`stft.center`, `stft.default`, `lstm.input`) landed:
#: every upstream-CIA op with a native kernel here has a body.
_PENDING_PINNED = ()

#: The upstream-CIA op the refusal path is exercised with now that nothing is
#: pending: no body in `_CIA_TABLE` and no native kernel either (aten.rs
#: `IMPLEMENTED`), so `_op_dk(CompositeImplicitAutograd)` refuses naming it
#: and `decompose` answers NotImplemented. Upstream answers True for it
#: (measured on 2.13.0, 2026-11-21) and has no Python kernel for it at that
#: key. When it gains a body or a kernel, pick another such op.
_UNCOVERED_STAND_IN = "aten.kron.default"

def test_upstream_is_the_recorded_table():
    """Re-derive `UPSTREAM_FACTS`: which ops are CompositeImplicitAutograd,
    where their Python decompositions are (none for `linear`; `t`'s is
    `torch._refs.t`, and `t` is CompositeExplicitAutograd, not CIA), and where
    a mode sees `linear` itself (inference mode, below autograd) or its body's
    ops (plain eager, and the Python dispatcher with nothing registered)."""
    if not _have_upstream():
        _skip.skip("upstream torch not importable without the vendored tree")
        return
    got = _run(_FACTS_PROBE, False)["out"]
    assert got == UPSTREAM_FACTS, {k: (UPSTREAM_FACTS.get(k), got.get(k))
                                   for k in set(UPSTREAM_FACTS) | set(got)
                                   if UPSTREAM_FACTS.get(k) != got.get(k)}


def test_the_table_is_truthful_against_upstream():
    """`_dispatch_has_kernel_for_dispatch_key(op, CompositeImplicitAutograd)`
    is True on the shim exactly for the covered ops, each of which upstream
    answers True for; the door's armed set is the table; every covered op has
    a source, a measurement and cases here; and the `pending` list (upstream
    CIA, a native kernel here, no body) is exactly what its definition says,
    answers False, and is printed so it can be watched shrinking."""
    if not _need_both():
        return
    shim = _run(_TABLE_PROBE, True, json.dumps([]))["out"]
    table = shim["table"]
    covered = sorted(table["covered"])
    pending = table["pending"]
    assert "aten.linear.default" in covered, table
    assert shim["armed"] == covered, ("the door's armed set is not the table", shim["armed"], covered)
    for op, row in table["covered"].items():
        assert row["source"] in ("torchnative composition", "upstream torch._decomp",
                                 "upstream torch._refs"), (op, row)
        assert row["checked"], (op, "no upstream measurement named")
        assert CASES.get(op), f"{op} is in _CIA_TABLE and has no CASES in this file"
    assert not set(covered) & set(pending), set(covered) & set(pending)
    assert set(pending) <= set(shim["native"]), sorted(set(pending) - set(shim["native"]))

    def qual(key):
        ns, name, ov = key.split(".", 2)
        return f"{ns}::{name}" + ("" if ov == "default" else f".{ov}")

    non_cia = ["aten::t", "aten::addmm", "aten::relu", "aten::permute", "aten::mm"]
    stand_in = qual(_UNCOVERED_STAND_IN)
    names = [qual(k) for k in covered + pending] + non_cia + [stand_in]
    shim = _run(_TABLE_PROBE, True, json.dumps(names))["out"]
    up = _run(_TABLE_PROBE, False, json.dumps(names))["out"]
    up_keys = set()
    for name in up["upstream_registrations"]:
        ns, _, rest = name.partition("::")
        op, _, ov = rest.partition(".")
        up_keys.add(f"{ns}.{op}.{ov or 'default'}")
    for key in covered:
        assert up["has_cia"][qual(key)] is True, (key, "covered, but upstream has no CIA kernel")
        assert shim["has_cia"][qual(key)] is True, (key, shim["has_cia"][qual(key)])
    for key in pending:
        assert key in up_keys and up["has_cia"][qual(key)] is True, (key, "pending, but not CIA upstream")
        assert shim["has_cia"][qual(key)] is False, (key, shim["has_cia"][qual(key)])
    for name in non_cia:
        assert up["has_cia"][name] is False and shim["has_cia"][name] is False, (
            name, up["has_cia"][name], shim["has_cia"][name])
    # The refusal tests' stand-in: CIA upstream, no body and no kernel here.
    assert up["has_cia"][stand_in] is True and shim["has_cia"][stand_in] is False, (
        stand_in, up["has_cia"][stand_in], shim["has_cia"][stand_in])
    assert _UNCOVERED_STAND_IN not in table["covered"], _UNCOVERED_STAND_IN
    assert _UNCOVERED_STAND_IN not in shim["native"], _UNCOVERED_STAND_IN
    # The pending list may only shrink (issue #374; leader decision
    # 2026-11-16). `_PENDING_PINNED` is the list on the day the frame landed:
    # the 42 ops of aten.rs's implemented lists for which upstream 2.13.0
    # answers True to `_dispatch_has_kernel_for_dispatch_key(op,
    # CompositeImplicitAutograd)`, derived 2026-11-17 by asking upstream about
    # each. A name outside it means a new native kernel for an upstream-CIA
    # op landed without a body: that fails here. When a body lands, its op
    # leaves the shim's list; remove it from the pin in the same change (a
    # stale pin entry fails too, so the pin cannot hide a regrowth). Since
    # #374's last three bodies the pin is empty, so any pending op fails here.
    def norm(key):
        return key if key.count(".") >= 2 else key + ".default"
    now = {norm(k) for k in pending}
    grown = sorted(now - set(_PENDING_PINNED))
    assert not grown, f"upstream-CIA ops gained a native kernel without a decomposition body (#374): {grown}"
    stale = sorted(set(_PENDING_PINNED) - now)
    assert not stale, f"these left the pending list; remove them from _PENDING_PINNED (#374): {stale}"
    print(f"CIAFRAME covered {len(covered)}: {covered}")
    print(f"CIAFRAME pending {len(pending)} (upstream CIA, native kernel here, no body; "
          f"has_kernel False): {pending}")
    print(f"CIAFRAME upstream CIA aten overloads {table['upstream_cia']}, "
          f"without a body {table['uncovered']}")


def _strip_layout(value):
    """`value` with the contiguity flag of every tensor descriptor removed.
    A descriptor is `["T", shape, is_contiguous, dtype]`; only its third
    element goes, so shapes, dtypes, op names, arguments, keyword arguments
    and error texts stay compared exactly."""
    if isinstance(value, list):
        if len(value) == 4 and value[0] == "T" and isinstance(value[2], bool):
            return ["T", value[1], None, value[3]]
        return [_strip_layout(item) for item in value]
    if isinstance(value, dict):
        return {key: _strip_layout(item) for key, item in value.items()}
    return value


def _classify(want, got):
    """What kind of difference a row has, most structural first."""
    if not isinstance(got, dict) or not isinstance(want, dict):
        return f"not a log: upstream {str(want)[:160]!r} shim {str(got)[:160]!r}"
    if "log" not in got or "log" not in want:
        return f"no log: upstream keys {sorted(want)} shim {str(got)[:300]!r}"
    wn, gn = [step[0] for step in want["log"]], [step[0] for step in got["log"]]
    if wn != gn:
        return f"op names: upstream {wn} shim {gn}"
    strip = _strip_layout
    if strip(want) == strip(got):
        where = [f"{i}:{step[0]}" for i, (step, other) in enumerate(zip(want["log"], got["log"]))
                 if step != other]
        return f"contiguity flag only: steps {where or ['result']}"
    for i, (step, other) in enumerate(zip(want["log"], got["log"])):
        if strip(step) != strip(other):
            return (f"step {i} {step[0]} beyond contiguity: upstream {json.dumps(step)[:260]} "
                    f"shim {json.dumps(other)[:260]}")
    rest_w = {key: value for key, value in want.items() if key != "log"}
    rest_g = {key: value for key, value in got.items() if key != "log"}
    return f"outside the log: upstream {json.dumps(rest_w)[:260]} shim {json.dumps(rest_g)[:260]}"


#: Rows that equal upstream's up to a stated allowance (issue #383 and the
#: compiler-dependent text of DECOMPOSE.md section 6.2), pinned per platform
#: as UPPER bounds: the test fails above the cap and, below it, prints that
#: the cap can be lowered (lowering is a manual follow-up). A platform that
#: is not a key has no cap; the count is only printed.
#: LAYOUT: only the contiguity flag of descriptors differs (#383). Run
#: 43874483840, with fresh results laid out as upstream lays them out, measured
#: 0 on Linux and on macOS (it was 215 and 246 before), so the cap is 0.
_LAYOUT_ONLY_CAP = {"linux": 0, "darwin": 0}
#: REORDERED_LAYOUT: equal after the flags are removed from both sides, up to
#: a dependency-respecting reordering (off macOS only). It was 31 on Linux
#: (10 math attention, 21 attention); run 43874483840 measured 0, because the
#: layout difference is gone and those rows are plain "reordered" now.
_REORDERED_LAYOUT_CAP = {"linux": 0}
#: The `+ 26` in both caps below is #374's `aten.lstm.input` rows, derived
#: from upstream's source and not measured: `_lstm_impl` returns
#: `std::make_tuple(outputs, at::stack(hy, 0), at::stack(cy, 0))`, whose
#: arguments gcc evaluates in its own order, so on Linux the two final `stack`
#: steps may come in the other order (13 float64 rows that compute, times
#: `op_dk` and `call`). If CI measures fewer, the `can be lowered` line says so.
_REORDERED_ALL_CAP = {"linux": 104 + 26}
#: REORDERED: exact apart from the order of independent steps (off macOS only).
#: Run 43874483840 measured 104 on Linux: 46 `aten.diff.default`, 10
#: `aten._scaled_dot_product_attention_math.default` and 48
#: `aten.scaled_dot_product_attention.default` (73 before, plus the 31 rows
#: that had a layout flag as well).
_REORDERED_CAP = {"linux": 104 + 26}
#: RENAMED: error texts equal up to C++ type-name spelling (off macOS only).
_RENAMED_CAP = {"linux": 3}


def _row_kind(want, got, want_flow, got_flow, allowances):
    """Tell one row apart: `"exact"`, `"layout"`, `"reordered"`,
    `"renamed"`, `"reordered_layout"` or `"fail"`, with a detail string.

    `allowances` is False on macOS, where reordering and C++ type names are
    not allowed. A row where the shim has no log (an error where upstream
    computes) is never separated: it is `"fail"`."""
    if got == want:
        return "exact", ""
    both_logs = (isinstance(want, dict) and isinstance(got, dict)
                 and "log" in want and "log" in got)
    if allowances:
        if _reordering(want, got, want_flow, got_flow):
            return "reordered", ""
        if _same_but_cpp_type_names(want, got):
            return "renamed", ""
    if both_logs:
        names_equal = [s[0] for s in want["log"]] == [s[0] for s in got["log"]]
        if names_equal and _strip_layout(want) == _strip_layout(got):
            return "layout", _first_layout_step(want, got)
        if allowances and _reordering(_strip_layout(want), _strip_layout(got),
                                      want_flow, got_flow):
            return "reordered_layout", ""
    return "fail", _classify(want, got)


def _first_layout_step(want, got):
    """The op name of the first step whose descriptors differ, else `result`."""
    for step, other in zip(want["log"], got["log"]):
        if step != other:
            return step[0]
    return "result"


def _cap_verdict(name, count, cap):
    """`(failure, note)` for one count against its upper bound. `cap` None
    (a platform without a cap): neither. Above the cap: a failure. Below it:
    a note that the cap can be lowered; the test still passes."""
    if cap is None:
        return None, None
    if count > cap:
        return (f"{name}: {count} rows, above the cap {cap}: these rows grew "
                f"(#383 for layout, DECOMPOSE.md section 6.2 otherwise)"), None
    if count < cap:
        return None, f"CIAFRAME CAP {name} can be lowered: measured {count}, cap {cap}"
    return None, None


def _per_key_op(keys):
    counts = {}
    for key in keys:
        op = key.split("|")[1] if "|" in key else key
        counts[op] = counts.get(op, 0) + 1
    return counts


def test_each_body_emits_upstreams_sequence():
    """For every case in `CASES`, under a logging `TorchDispatchMode`, the
    ops the body emits (name, argument shapes and contiguity, keyword
    arguments, result) equal upstream's, through `_op_dk(CIA)`, through the
    op itself (the door's CIA phase) and through the public spelling, except
    for the separated kinds below.

    Separated, never silently passed, each with an upper bound per platform
    (a count above its cap fails; below it a `CIAFRAME CAP ... can be
    lowered` line is printed, and lowering is a manual follow-up):

    * layout only (#383, `_LAYOUT_ONLY_CAP`): the row differs from upstream
      only in the contiguity flag of tensor descriptors. Counted per op of
      the first differing step on `CIAFRAME LAYOUT ` lines
      (`TORCHNATIVE_CIAFRAME_VERBOSE=1` prints every row). A row where the
      shim has no log is never separated.
    * reordered and layout flag (off macOS, `_REORDERED_LAYOUT_CAP`): equal
      after the flags are removed from both sides, up to a
      dependency-respecting reordering, with `_reordering`'s data-flow check
      unchanged.
    * reordered (off macOS, `_REORDERED_CAP`) and C++ type-name spelling in
      an error text (off macOS, `_RENAMED_CAP`).

    Measured in CI run 43814367564 (head 55cc039): Linux 215 layout only, 55
    rows that the log shows with the same op multiset but another order (the
    data flow check had not run on them: stripped, they are the reordered and
    layout kind), 49 reordered (46 `aten.diff.default`, 3
    `aten.scaled_dot_product_attention.default`), 3 type-name; macOS 246
    layout only and nothing else. Why separated and not fixed first (leader,
    2026-11-16): for a fresh result the values and the aliasing are right, and
    the effect shows by name in individual tests.

    What this cannot find: a layout flag difference that is a real aliasing
    defect hides in the separated count. The view kernels were that, and
    tests/ops/test_viewstride.py now guards them."""
    if not _need_both():
        return
    shim, up = _seq(True), _seq(False)
    rows = [k for k in up if k.startswith("cases|")]
    assert rows, "no cases ran upstream"
    # Allowances off on macOS, where the bodies were measured and every row
    # must equal upstream's up to the layout flag. Elsewhere upstream's own
    # text is compiler-dependent (CI run 43802210944, Linux only;
    # docs/design/DECOMPOSE.md section 6.2): the order of independent ops (a
    # C++ CompositeImplicit kernel's function arguments are evaluated in an
    # unspecified order, gcc right to left, clang left to right) and C++ type
    # names in a message.
    allowances = sys.platform != "darwin"
    up_flow, shim_flow = _flow(False), _flow(True)
    verbose = os.environ.get("TORCHNATIVE_CIAFRAME_VERBOSE") == "1"
    kinds = {"layout": {}, "reordered": {}, "renamed": {}, "reordered_layout": {}}
    bad = {}
    for k in sorted(rows):
        kind, detail = _row_kind(up[k], shim.get(k), up_flow.get(k), shim_flow.get(k), allowances)
        if kind == "fail":
            bad[k] = detail
        elif kind != "exact":
            kinds[kind][k] = detail
    for k in sorted(kinds["reordered"]):
        print(f"CIAFRAME REORDERED {k} :: data flow identical")
    for k in sorted(kinds["renamed"]):
        print(f"CIAFRAME CPPNAME {k} :: equal with C++ type names unified: "
              f"upstream {up[k]['error'][:200]!r} shim {shim[k]['error'][:200]!r}")
    print(f"CIAFRAME on {sys.platform}: {len(kinds['reordered'])} of {len(rows)} rows equal upstream's "
          f"only up to a dependency-respecting reordering, {len(kinds['renamed'])} only up to C++ "
          f"type-name spelling, {len(kinds['reordered_layout'])} reordered and layout flag, "
          f"{len(kinds['layout'])} layout flag only (#383); the first two allowances and the third "
          f"are off on darwin")
    caps = {"layout": ("layout only", _LAYOUT_ONLY_CAP),
            "reordered_layout": ("reordered and layout flag", _REORDERED_LAYOUT_CAP),
            "reordered": ("reordered", _REORDERED_CAP),
            "renamed": ("C++ type names", _RENAMED_CAP)}
    failures, notes = [], []
    for kind, (name, table) in caps.items():
        rows_of_kind = kinds[kind]
        if kind == "layout":
            per_op = {}
            for detail in rows_of_kind.values():
                per_op[detail] = per_op.get(detail, 0) + 1
        else:
            per_op = _per_key_op(rows_of_kind)
        cap = table.get(sys.platform)
        for op, n in sorted(per_op.items()):
            print(f"CIAFRAME LAYOUT {name}: {op}: {n}")
        print(f"CIAFRAME LAYOUT {name}: total {len(rows_of_kind)} of {len(rows)} rows, "
              f"cap {cap if cap is not None else 'none on ' + sys.platform}")
        if verbose:
            for k, detail in sorted(rows_of_kind.items()):
                print(f"CIAFRAME LAYOUT ROW {name}: {k} :: {detail}")
        failure, note = _cap_verdict(name, len(rows_of_kind), cap)
        if failure:
            failures.append(failure)
        if note:
            notes.append(note)
    both = len(kinds["reordered"]) + len(kinds["reordered_layout"])
    failure, note = _cap_verdict("reordered, with and without a layout flag", both,
                                 _REORDERED_ALL_CAP.get(sys.platform))
    if failure:
        failures.append(failure)
    if note:
        notes.append(note)
    for note in notes:
        print(note)
    # One line per differing row, so a CI log holds all of them (run
    # 43796765257 cut the JSON of some 200 rows off after 6000 characters).
    counts = {}
    for k in sorted(bad):
        print(f"CIAFRAME DIFF {k} :: {bad[k]}")
        key = bad[k].split(":")[0]
        counts[key] = counts.get(key, 0) + 1
    for kind, n in sorted(counts.items()):
        print(f"CIAFRAME DIFF KIND {kind}: {n} of {len(rows)} rows")
    assert not bad, f"{len(bad)} of {len(rows)} rows differ from upstream: {counts}"
    assert not failures, "; ".join(failures)
    print(f"CIAFRAME sequences: {len(rows)} (case, entry) rows equal to upstream's "
          f"or separated as above")


def test_the_row_kinds_are_told_apart_on_recorded_rows():
    """The separation's logic on hand-written miniature rows, without torch:
    each kind is told from the others, and a cap fails only above its bound.

    Nullification guard (AGENTS.md 17.5): making the layout stripping too
    greedy (stripping shapes, dtypes or a whole descriptor) turns the
    `real difference` cases red; making it too narrow turns the layout cases
    red."""
    def T(shape, contig=True, dtype="f32"):
        return ["T", shape, contig, dtype]

    def row(*steps, result=None):
        return {"log": [list(s) for s in steps], "result": result if result is not None else T([2])}

    a = ["aten.mul.Scalar", [T([2, 3])], {}, T([2, 3])]
    b = ["aten.add.Tensor", [T([2, 3]), T([2, 3])], {}, T([2, 3])]
    flow = {"deps": [[], []], "mut": [False, False]}
    flow_swapped = {"deps": [[], []], "mut": [False, False]}

    def kind(want, got, wf=flow, gf=flow_swapped, allow=True):
        return _row_kind(want, got, wf, gf, allow)[0]

    base = row(a, b)
    assert kind(base, row(a, b)) == "exact"
    # Layout only: the flag of one descriptor in an argument and in the result.
    a_flag = ["aten.mul.Scalar", [T([2, 3], False)], {}, T([2, 3], False)]
    layout = row(a_flag, b)
    assert kind(base, layout) == "layout"
    assert kind(base, layout, allow=False) == "layout"
    assert _row_kind(base, layout, flow, flow, True)[1] == "aten.mul.Scalar"
    # Reordered: the same steps swapped, independent.
    assert kind(base, row(b, a)) == "reordered"
    assert kind(base, row(b, a), allow=False) == "fail"
    # Reordered and layout flag.
    assert kind(base, row(b, a_flag)) == "reordered_layout"
    assert kind(base, row(b, a_flag), allow=False) == "fail"
    # A real difference in an argument (a shape) is never separated, even
    # next to a flag difference.
    a_shape = ["aten.mul.Scalar", [T([3, 2], False)], {}, T([2, 3])]
    assert kind(base, row(a_shape, b)) == "fail"
    a_kw = ["aten.mul.Scalar", [T([2, 3])], {"alpha": 2}, T([2, 3])]
    assert kind(base, row(a_kw, b)) == "fail"
    a_dtype = ["aten.mul.Scalar", [T([2, 3])], {}, T([2, 3], True, "f64")]
    assert kind(base, row(a_dtype, b)) == "fail"
    assert kind(base, row(a, b, result=T([2, 2]))) == "fail"
    # A real difference in op names (different multisets), with a flag
    # difference too, is a failure.
    c = ["aten.sub.Tensor", [T([2, 3]), T([2, 3])], {}, T([2, 3])]
    assert kind(base, row(a_flag, c)) == "fail"
    assert kind(base, row(c, a_flag)) == "fail"
    # A reordering whose data flow differs is refused (a dependency).
    dep = {"deps": [[], [0]], "mut": [False, False]}
    assert kind(base, row(b, a_flag), wf=dep, gf=flow) == "fail"
    # Steps that print alike (three scalar sums of `a + b + c`, CI run
    # 43819391187): clang evaluates sum, sum, add, sum, add and gcc sum, sum,
    # sum, add, add. The matching is by what each step consumes, so this is a
    # reordering whichever `sum` the first `add` took.
    sm = ["aten.sum.default", [T([0, 3])], {}, T([])]
    ad = ["aten.add.Tensor", [T([]), T([])], {}, T([])]
    clang = row(sm, sm, ad, sm, ad)
    gcc = row(sm, sm, sm, ad, ad)
    clang_flow = {"deps": [[], [], [0, 1], [], [2, 3]], "mut": [False] * 5}
    gcc_flow = {"deps": [[], [], [], [1, 2], [0, 3]], "mut": [False] * 5}
    assert kind(gcc, clang, wf=gcc_flow, gf=clang_flow) == "reordered"
    # The same step lists with another data flow (the second add consuming
    # two sums and not the first add) are not a reordering.
    other_flow = {"deps": [[], [], [], [1, 2], [0, 1]], "mut": [False] * 5}
    assert kind(gcc, clang, wf=other_flow, gf=clang_flow) == "fail"
    # A step consumed twice on one side and once on the other is not either.
    twice_flow = {"deps": [[], [], [0, 0], [], [2, 3]], "mut": [False] * 5}
    assert kind(gcc, clang, wf=gcc_flow, gf=twice_flow) == "fail"
    # A writer is still a barrier.
    writer_flow = {"deps": [[], [], [0, 1], [], [2, 3]], "mut": [False, False, True, False, False]}
    assert kind(gcc, clang, wf=gcc_flow, gf=writer_flow) == "fail"
    # A row where the shim has no log is never separated.
    assert kind(base, {"error": "NotImplementedError: aten.x"}) == "fail"
    assert kind({"error": "boom"}, base) == "fail"
    assert kind(base, None) == "fail"
    # C++ type names in an error text.
    assert kind({"error": "got long int and float"}, {"error": "got long long and float"}) == "renamed"
    assert kind({"error": "got long int and float"}, {"error": "got long long and double"}) == "fail"
    # The cap rule: fails above, only a note below, silent at the cap and
    # without a cap.
    assert _cap_verdict("x", 6, 5)[0] and "grew" in _cap_verdict("x", 6, 5)[0]
    assert _cap_verdict("x", 4, 5) == (None, "CIAFRAME CAP x can be lowered: measured 4, cap 5")
    assert _cap_verdict("x", 5, 5) == (None, None)
    assert _cap_verdict("x", 99, None) == (None, None)
    assert _LAYOUT_ONLY_CAP.get("win32") is None


def _reordered_copy(row, flow):
    """`row` and `flow` with the steps between two writers put in another
    topological order (the latest ready step first), the dependency indices
    renumbered to match; with the number of steps that moved."""
    log, deps, mut = row["log"], flow["deps"], flow["mut"]
    order, segment = [], []

    def drain():
        ready_of = set(segment)
        placed = set()
        while len(placed) < len(segment):
            ready = [i for i in segment if i not in placed
                     and all(p in placed or p not in ready_of for p in deps[i])]
            pick = max(ready)
            placed.add(pick)
            order.append(pick)
        segment.clear()

    for i in range(len(log)):
        if mut[i]:
            drain()
            order.append(i)
        else:
            segment.append(i)
    drain()
    where = {old: new for new, old in enumerate(order)}
    got = dict(row, log=[log[i] for i in order])
    got_flow = {"deps": [sorted(where[p] for p in deps[i]) for i in order],
                "mut": [mut[i] for i in order]}
    return got, got_flow, sum(1 for k, i in enumerate(order) if i != k)


def _signature_moves(row, flow, got, got_flow):
    """How many positions hold a step of another signature after reordering.
    Swapping steps with one signature moves nothing `_reordering` can see
    (they are interchangeable), so the generated count of moved steps is not
    what it is held to; this is."""
    a = _step_signatures(row["log"], flow["deps"])
    b = _step_signatures(got["log"], got_flow["deps"])
    return sum(1 for x, y in zip(a, b) if x != y)


def test_reordering_stays_small_on_every_recorded_row():
    """`_reordering` on upstream's own rows: each row against itself is 0
    steps moved, and each row against a copy with its independent steps put
    in another order is recognised as a reordering, which moved steps exactly
    when a position now holds a step of another signature; every signature is
    a 64-character digest.

    Red before the digest (CI run 45131459631, Linux, where reordering is
    allowed): the signature text nested each producer's whole text, about
    1.2e12 bytes for a 2-layer bidirectional `lstm.input` row, and the gate's
    memory watchdog killed this suite at 14 GB. On macOS the rows were exact
    and `_reordering` never ran, so nothing local showed it.

    What this cannot find: a digest collision that pairs two different steps
    (SHA-256; not a practical concern), and a reordering the shim really
    emits that these generated orders do not resemble."""
    if not _have_upstream():
        _skip.skip("upstream torch not importable without the vendored tree")
        return
    up, up_flow = _seq(False), _flow(False)
    rows = [k for k, row in up.items()
            if isinstance(row, dict) and "log" in row and up_flow.get(k)]
    lstm = [k for k in rows if "|aten.lstm.input|" in k]
    assert lstm, "no lstm.input row recorded a log"
    moved_rows = 0
    for k in rows:
        row, flow = up[k], up_flow[k]
        sig = _step_signatures(row["log"], flow["deps"])
        assert sig is not None and all(len(x) == 64 for x in sig), k
        assert _reordering(row, row, flow, flow) == 0, k
        got, got_flow, _ = _reordered_copy(row, flow)
        counted = _reordering(row, got, flow, got_flow)
        moved = _signature_moves(row, flow, got, got_flow)
        assert counted is not None and (counted > 0) == (moved > 0), (k, moved, counted)
        moved_rows += moved > 0
    assert moved_rows > 0, "no generated order moved a step: the check compared nothing"
    # A row whose reordering swaps only interchangeable steps: the same op on
    # two inputs of one shape, as upstream's oneDNN lstm path does with h0 and
    # c0 (CI run 45460282414, Linux: 'onednn|aten.lstm.input|float32 1 layer|
    # op_dk' moved 4 steps, all interchangeable, and `_reordering` rightly
    # counted 0). Recorded on every platform, so macOS exercises it too.
    step = ["aten.squeeze.dim", [["float32", [1, 2, 4], True], 0], {}, ["float32", [2, 4], True]]
    twins = {"log": [step, step], "result": "r"}
    twins_flow = {"deps": [[], []], "mut": [False, False]}
    got, got_flow, swapped = _reordered_copy(twins, twins_flow)
    assert swapped == 2, swapped
    assert _signature_moves(twins, twins_flow, got, got_flow) == 0
    assert _reordering(twins, got, twins_flow, got_flow) == 0
    print(f"CIAFRAME reordering: {len(rows)} upstream rows ({len(lstm)} lstm.input), "
          f"{moved_rows} with a generated reordering, each told apart in bounded memory")


def test_branches_without_a_shim_kernel_refuse_by_name():
    """`NO_KERNEL`: upstream runs each (and its sequence is measured); the
    body here reaches an op this shim has no kernel for and the door refuses,
    naming it. Through `_op_dk` and through the op itself. This asserts a gap
    on purpose: when `dot`, `mv` or `logical_xor` get kernels the
    rows move to `CASES` (and fail here until they do)."""
    if not _need_both():
        return
    shim, up = _seq(True), _seq(False)
    count = 0
    for op, rows in NO_KERNEL.items():
        for label, _, name in rows:
            for how in ("op_dk", "call"):
                key = f"nokernel|{op}|{label}|{how}"
                assert "error" not in up[key] and up[key]["log"], (key, up[key])
                row = shim[key]
                assert "error" in row, (key, "the shim ran it: move this row to CASES", row)
                assert name in row["error"], (key, name, row["error"])
                count += 1
    print(f"CIAFRAME no-kernel branches: {count} (row, entry) pairs refuse naming the missing op")


def test_branches_the_shim_refuses_by_name_run_upstream_and_name_the_refusal():
    """`REFUSED_BY_NAME`: upstream runs each, and its measured sequence
    contains the flash op the body chose (the body emits it as upstream's
    does); the shim reaches the same op and its kernel refuses, naming the op
    and the combination, through `_op_dk`, the op itself and the public
    spelling. Moved here from `CASES` ("float32 mask on float64 allowed", three
    rows) when the kernel began to refuse that combination (leader decision
    2026-11-16)."""
    if not _need_both():
        return
    shim, up = _seq(True), _seq(False)
    count = 0
    for op, rows in REFUSED_BY_NAME.items():
        for label, _, name, parts in rows:
            for how in ("op_dk", "call", "public"):
                key = f"refused|{op}|{label}|{how}"
                assert "error" not in up[key] and up[key]["log"], (key, up[key])
                assert any(step[0] == "aten._scaled_dot_product_flash_attention_for_cpu.default"
                           for step in up[key]["log"]), (key, [s[0] for s in up[key]["log"]])
                row = shim[key]
                assert "error" in row, (key, "the shim ran it: the refusal is gone", row)
                assert name in row["error"], (key, name, row["error"])
                for part in parts:
                    assert part in row["error"], (key, part, row["error"])
                count += 1
    print(f"CIAFRAME refused-by-name branches: {count} (row, entry) pairs refuse naming the op")


def test_lstm_float32_rows_take_onednn_upstream_only_where_it_is_built():
    """`_LSTM_FLOAT32`: upstream's C++ `lstm` takes oneDNN
    (`aten.mkldnn_rnn_layer.default`, one call per layer and direction, with
    a oneDNN workspace as its fourth result) for a float32 CPU input on a
    build with oneDNN, and the native path on one without (macOS arm64,
    measured 2026-11-21). This shim reports no oneDNN, so its body is the
    native path on every platform.

    Where upstream took the native path, the shim's row must equal it (on
    macOS these rows are also in `CASES`). Where upstream took oneDNN, the
    shim's row must be the native sequence (no oneDNN step) and its result
    (output, h_n, c_n) must have upstream's shapes, dtypes and contiguity.
    On macOS upstream must not take oneDNN. What this cannot see: the
    workspace's size, which is oneDNN's own and which the shim has no op for
    (#374 names `aten.mkldnn_rnn_layer.default` as the kernel a oneDNN-equal
    body would need)."""
    if not _need_both():
        return
    shim, up = _seq(True), _seq(False)
    onednn = native = 0
    for label, _ in _LSTM_FLOAT32:
        for how in ("op_dk", "call"):
            key = f"onednn|aten.lstm.input|{label}|{how}"
            want, got = up[key], shim[key]
            assert "log" in want, (key, want)
            assert "log" in got, (key, got)
            assert all(step[0] != "aten.mkldnn_rnn_layer.default" for step in got["log"]), key
            if any(step[0] == "aten.mkldnn_rnn_layer.default" for step in want["log"]):
                assert sys.platform != "darwin", (key, "upstream took oneDNN on macOS")
                assert got["result"] == want["result"], (key, want["result"], got["result"])
                onednn += 1
            else:
                kind, detail = _row_kind(want, got, _flow(False).get(key), _flow(True).get(key),
                                         sys.platform != "darwin")
                assert kind in ("exact", "reordered"), (key, kind, detail)
                native += 1
    print(f"CIAFRAME ONEDNN on {sys.platform}: {onednn} float32 lstm (row, entry) pairs where "
          f"upstream took oneDNN and the shim the native path, {native} where both took the "
          f"native path")


def test_the_counters_show_which_entry_ran_the_body():
    """`_shim_cia_counters()` per entry, against upstream's measured dispatch:
    eager `F.linear` runs the body at the door; `_op_dk` and `decompose` run
    it directly; under a mode the mode sees upstream's `t, addmm`; in
    inference mode and below autograd the mode sees `aten.linear.default`
    itself (upstream's log, `UPSTREAM_FACTS`) and the backend phase runs the
    body afterwards."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    got = _shim_only()
    one = lambda **k: {"door": 0, "backend": 0, "op_dk": 0, **k}  # noqa: E731
    assert got["eager"] == one(door=1), got["eager"]
    assert got["op_dk"] == one(op_dk=1), got["op_dk"]
    assert got["mode"]["log"] == UPSTREAM_FACTS["logs"]["mode"], got["mode"]
    assert got["mode"]["counters"] == one(door=1), got["mode"]
    assert got["inference_mode"]["log"] == UPSTREAM_FACTS["logs"]["inference_mode"], got["inference_mode"]
    assert got["inference_mode"]["counters"] == one(backend=1), got["inference_mode"]
    assert got["inference_mode_no_mode"] == one(backend=1), got["inference_mode_no_mode"]
    assert got["below_autograd"]["log"] == UPSTREAM_FACTS["logs"]["below_autograd"], got["below_autograd"]
    assert got["below_autograd"]["counters"] == one(backend=1), got["below_autograd"]
    assert got["decompose"]["delta"]["op_dk"] == 1, got["decompose"]
    assert got["decompose"]["equal_to_eager"] is True, got["decompose"]


def test_an_uncovered_op_refuses_by_name():
    """`_op_dk(CompositeImplicitAutograd)` for an op with no body refuses
    naming the op and the reason: an upstream-CIA op with no body and no
    kernel here (`_UNCOVERED_STAND_IN`; nothing is pending any more, #374)
    with "CompositeImplicit decomposition not implemented", the issue and
    "no kernel for it either"; an op upstream has no CIA kernel for (`t`;
    upstream's process crashes) with that fact. `decompose` of the stand-in
    answers NotImplemented (its key answer is False), the pending list is
    empty, `stft.default`, the former pending example, runs its body, and
    arming a non-aten key is refused."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    got = _shim_only()
    msg = got["refuse_uncovered"]
    assert msg and not msg.startswith("WRONG"), msg
    for part in (_UNCOVERED_STAND_IN, "CompositeImplicit decomposition not implemented", "#372",
                 "this shim has no kernel for it either"):
        assert part in msg, (part, msg)
    assert got["stand_in_native"] is False, (_UNCOVERED_STAND_IN, "has a native kernel now: pick another")
    msg = got["refuse_not_cia"]
    assert msg and not msg.startswith("WRONG"), msg
    assert "aten.t.default" in msg and "no CompositeImplicitAutograd kernel upstream" in msg, msg
    assert got["decompose_uncovered"] == "NotImplemented", got["decompose_uncovered"]
    assert got["pending"] == [], got["pending"]
    assert got["stft_op_dk"] == [9, 13], got["stft_op_dk"]
    assert got["arm_non_aten"] is not None and "tnpl.f.default" in got["arm_non_aten"], got["arm_non_aten"]


def test_disarming_the_body_takes_linear_away():
    """Nullification (AGENTS.md 17.5): with `aten.linear.default` disarmed the
    door has nothing for it (no native kernel), so `F.linear` refuses naming
    the op; armed again, the body runs and the counter moves. This is what
    makes the counter tests above evidence of the body, not of a kernel."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    got = _shim_only()
    assert got["disarmed_was_armed"] is True, got
    assert got["disarmed"] and "aten.linear.default" in got["disarmed"], got["disarmed"]
    assert got["rearmed"]["door"] == 1, got["rearmed"]


if __name__ == "__main__":
    sys.exit(1 if _skip.run_tests(
        [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)],
        suite="test_cia_frame",
    ) else 0)
