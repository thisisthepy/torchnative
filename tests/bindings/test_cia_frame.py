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
claims the op sequence, grade "reaches", AGENTS.md 16), and any op outside
`CASES` and `MATMUL_BRANCH`.

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

#: Cases whose upstream sequence the body reproduces exactly.
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
    ],
}

#: Cases where the body emits `aten.matmul.default`, which is CIA upstream and
#: has no body in `_CIA_TABLE` yet: a logging mode sees `matmul` on the shim
#: where upstream's sees its decomposition. The test asserts exactly that,
#: and nothing else differs, so the day `matmul` gets a body these cases fail
#: and move to `CASES`.
MATMUL_BRANCH = {
    "aten.linear.default": [
        ("2d nobias", "(torch.randn(2, 8), torch.randn(4, 8))"),
        ("3d nobias", "(torch.randn(3, 2, 8), torch.randn(4, 8))"),
        ("3d nc+bias", "(torch.randn(2, 3, 8).transpose(0, 1), torch.randn(4, 8), torch.randn(4))"),
        ("3d+bias2d", "(torch.randn(3, 2, 8), torch.randn(4, 8), torch.randn(2, 4))"),
    ],
}
# Not here: a 1-D input on this branch (`1d+bias0d`, or no bias). Upstream's
# matmul handles it (`unsqueeze, mm, squeeze_.dim`); the shim's native
# `matmul` refuses a 1-D operand by name (aten.rs `matmul_default`), a wall of
# that kernel and not of the frame, which `matmul`'s body will remove.

#: The public spelling that reaches each op, where one exists.
PUBLIC = {"aten.linear.default": "torch.nn.functional.linear"}


_SEQ_PROBE = r'''
import json, sys
import torch
IS_SHIM = hasattr(torch._C, "_aten_implemented")
assert IS_SHIM == (sys.argv[1] == "shim"), ("shim" if IS_SHIM else "upstream", sys.argv[1])  # AGENTS.md 15.2
from torch.utils._python_dispatch import TorchDispatchMode

CIA = torch._C.DispatchKey.CompositeImplicitAutograd
groups = json.loads(sys.argv[2])
public = json.loads(sys.argv[3])


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
            entries = {"op_dk": lambda a: op._op_dk(CIA, *a), "call": lambda a: op(*a)}
            if key in public:
                fn = eval(public[key])
                entries["public"] = lambda a, fn=fn: fn(*a)
            for how, call in entries.items():
                torch.manual_seed(0)
                args = eval(expr)
                row = {}
                try:
                    with Log() as mode:
                        result = call(args)
                    row["log"] = mode.log
                    row["result"] = desc(result)
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
out["refuse_pending"] = refusal(lambda: aten.reshape.default._op_dk(CIA, torch.randn(2, 3), [6]))
out["refuse_not_cia"] = refusal(lambda: aten.t.default._op_dk(CIA, torch.randn(2, 3)))
out["decompose_pending"] = repr(aten.reshape.default.decompose(torch.randn(2, 3), [6]))[:40]

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
        groups = {"cases": CASES, "matmul": MATMUL_BRANCH}
        _cache[key] = _run(_SEQ_PROBE, shim, json.dumps(groups), json.dumps(PUBLIC))["out"]
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


def test_the_matmul_branch_stops_at_matmul():
    """The pinned gap, measured: in `MATMUL_BRANCH` the shim's log is
    upstream's with upstream's `matmul` decomposition replaced by one
    `aten.matmul.default` (CIA upstream, no body here yet), and the result
    matches. Fails, and says so, the day `matmul` gets a body."""
    if not _need_both():
        return
    shim, up = _seq(True), _seq(False)
    rows = [k for k in up if k.startswith("matmul|")]
    assert rows, "no matmul-branch cases ran upstream"
    for k in rows:
        s_row, u_row = shim[k], up[k]
        assert "error" not in u_row and "error" not in s_row, (k, u_row, s_row)
        names = [e[0] for e in s_row["log"]]
        assert names.count("aten.matmul.default") == 1, (
            k, "the shim no longer emits matmul here: move this case to CASES", names)
        i = names.index("aten.matmul.default")
        tail = len(names) - i - 1
        u_log = u_row["log"]
        assert s_row["log"][:i] == u_log[:i], (k, u_log, s_row["log"])
        assert s_row["log"][i + 1:] == (u_log[len(u_log) - tail:] if tail else []), (k, u_log, s_row["log"])
        assert "aten.matmul.default" not in [e[0] for e in u_log], (k, u_log)
        assert s_row["result"] == u_row["result"], (k, u_row["result"], s_row["result"])
    print(f"CIAFRAME matmul branch: {len(rows)} rows differ from upstream only by an "
          f"undecomposed aten.matmul.default")


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
    naming the op and the reason: an upstream-CIA op (`reshape`, pending)
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
    for part in ("aten.reshape.default", "CompositeImplicit decomposition not implemented", "#372"):
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
