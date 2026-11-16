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

These tests were written before the implementation and have not run on a
built shim (leader rule: agents compile only). The upstream half was run on
2026-11-16 by hand for `UPSTREAM_FACTS`.
"""
import json
import os
import pathlib
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


#: A 4-D tensor whose strides are channels-last (a permuted NHWC tensor).
_CL = "torch.randn(2, 4, 5, 3).permute(0, 3, 1, 2)"
_CPU = "torch.device('cpu')"


#: Cases whose upstream sequence the body reproduces exactly. An expression
#: is the argument tuple, or `{"args": (...), "kw": {...}}` for keyword-only
#: arguments. Measured on torch 2.13.0, 2026-11-16 (#375).
CASES = {
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
    ],
    # matmul: every branch of upstream's C++ kernel that does not need `dot`,
    # `mv`, `squeeze_` (those are in NO_KERNEL).
    "aten.matmul.default": [
        ("2d@2d", "(torch.randn(3, 4), torch.randn(4, 5))"),
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
        ("float32 mask on float64 allowed", _sdpa('"attn_mask": torch.randn(4, 5)', _qkv("torch.float64"))),
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
        ("1d@2d", "(torch.randn(4), torch.randn(4, 5))", "squeeze_"),
        ("3d@1d fold", "(torch.randn(2, 3, 4), torch.randn(4))", "mv"),
        ("4d@1d fold", "(torch.randn(2, 3, 4, 5), torch.randn(5))", "mv"),
        ("1d grad@3d fold", "(torch.randn(5, requires_grad=True), torch.randn(2, 5, 6))", "mv"),
    ],
    "aten.linear.default": [
        ("1d nobias", "(torch.randn(8), torch.randn(4, 8))", "squeeze_"),
    ],
    "aten.diff.default": [
        ("bool", "(torch.randn(4) > 0,)", "logical_xor"),
        ("bool n 2", "(torch.randn(4) > 0, 2, 0)", "logical_xor"),
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


class Log(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.log = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        out = func(*args, **kwargs)
        self.log.append([str(func), [desc(a) for a in args],
                         {k: desc(v) for k, v in sorted(kwargs.items())}, desc(out)])
        return out


def overload(key):
    ns, name, ov = key.split(".", 2)
    return getattr(getattr(getattr(torch.ops, ns), name), ov)


out = {}
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
                except Exception as exc:  # noqa: BLE001
                    row["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                out[f"{group}|{key}|{label}|{how}"] = row
print(json.dumps({"is_shim": IS_SHIM, "out": out}))
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
out["refuse_pending"] = refusal(lambda: aten.view_as.default._op_dk(CIA, torch.randn(2, 3), torch.randn(3, 2)))
out["refuse_not_cia"] = refusal(lambda: aten.t.default._op_dk(CIA, torch.randn(2, 3)))
out["decompose_pending"] = repr(aten.view_as.default.decompose(torch.randn(2, 3), torch.randn(3, 2)))[:40]

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
                               for k, rows in NO_KERNEL.items()}}
        _cache[key] = _run(_SEQ_PROBE, shim, json.dumps(groups), json.dumps(PUBLIC),
                           json.dumps(list(IDENTITY)))["out"]
    return _cache[key]


def _shim_only():
    if "shim_only" not in _cache:
        _cache["shim_only"] = _run(_SHIM_PROBE, True)["out"]
    return _cache["shim_only"]


def _need_both():
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return False
    if not _have_upstream():
        _skip.skip("upstream torch not importable without the vendored tree")
        return False
    return True



#: See the pending check in the table test (issue #374).
_PENDING_PINNED = (
    "aten.adaptive_avg_pool1d.default",
    "aten.adaptive_avg_pool2d.default",
    "aten.argsort.default",
    "aten.argsort.stable",
    "aten.broadcast_tensors.default",
    "aten.chunk.default",
    "aten.clip.default",
    "aten.diag.default",
    "aten.greater.Scalar",
    "aten.greater.Tensor",
    "aten.imag.default",
    "aten.is_floating_point.default",
    "aten.lstm.input",
    "aten.max.other",
    "aten.max_pool1d.default",
    "aten.max_pool2d.default",
    "aten.min.other",
    "aten.multiply.Scalar",
    "aten.multiply.Tensor",
    "aten.nanmean.default",
    "aten.one_hot.default",
    "aten.real.default",
    "aten.reshape_as.default",
    "aten.rms_norm.default",
    "aten.std.default",
    "aten.std.dim",
    "aten.stft.center",
    "aten.stft.default",
    "aten.unflatten.int",
    "aten.var.default",
    "aten.var.dim",
    "aten.var_mean.default",
    "aten.var_mean.dim",
    "aten.view_as.default",
    "aten.where.Scalar",
    "aten.where.ScalarOther",
    "aten.where.default",
)

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
    names = [qual(k) for k in covered + pending] + non_cia
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
    # The pending list may only shrink (issue #374; leader decision
    # 2026-11-16). `_PENDING_PINNED` is the list on the day the frame landed:
    # the 42 ops of aten.rs's implemented lists for which upstream 2.13.0
    # answers True to `_dispatch_has_kernel_for_dispatch_key(op,
    # CompositeImplicitAutograd)`, derived 2026-11-17 by asking upstream about
    # each. A name outside it means a new native kernel for an upstream-CIA
    # op landed without a body: that fails here. When a body lands, its op
    # leaves the shim's list; remove it from the pin in the same change (a
    # stale pin entry fails too, so the pin cannot hide a regrowth).
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


def test_each_body_emits_upstreams_sequence():
    """For every case in `CASES`, under a logging `TorchDispatchMode`, the
    ops the body emits (name, argument shapes and contiguity, keyword
    arguments, result) equal upstream's, through `_op_dk(CIA)`, through the
    op itself (the door's CIA phase) and through the public spelling."""
    if not _need_both():
        return
    shim, up = _seq(True), _seq(False)
    rows = [k for k in up if k.startswith("cases|")]
    assert rows, "no cases ran upstream"
    bad = {k: (up[k], shim.get(k)) for k in rows if shim.get(k) != up[k]}
    assert not bad, json.dumps(bad, indent=1)[:6000]
    print(f"CIAFRAME sequences: {len(rows)} (case, entry) rows equal to upstream's")


def test_branches_without_a_shim_kernel_refuse_by_name():
    """`NO_KERNEL`: upstream runs each (and its sequence is measured); the
    body here reaches an op this shim has no kernel for and the door refuses,
    naming it. Through `_op_dk` and through the op itself. This asserts a gap
    on purpose: when `dot`, `mv`, `squeeze_` or `logical_xor` get kernels the
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
    naming the op and the reason: an upstream-CIA op (`view_as`, pending)
    with "CompositeImplicit decomposition not implemented" and the issue; an op
    upstream has no CIA kernel for (`t`; upstream's process crashes) with that
    fact. `decompose` of a pending op answers NotImplemented (its key answer is
    False), and arming a non-aten key is refused."""
    if not _have_shim():
        _skip.skip("vendored tree not built")
        return
    got = _shim_only()
    msg = got["refuse_pending"]
    assert msg and not msg.startswith("WRONG"), msg
    for part in ("aten.view_as.default", "CompositeImplicit decomposition not implemented", "#372"):
        assert part in msg, (part, msg)
    msg = got["refuse_not_cia"]
    assert msg and not msg.startswith("WRONG"), msg
    assert "aten.t.default" in msg and "no CompositeImplicitAutograd kernel upstream" in msg, msg
    assert got["decompose_pending"] == "NotImplemented", got["decompose_pending"]
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
