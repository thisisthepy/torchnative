"""Known divergences from upstream, each pinned by name to its issue (#358, #359).

Leader rule, 2026-11-15: a difference from upstream found while fixing a test
is not worked around silently. Each test below asserts **upstream's**
behaviour (measured on torch 2.13.0) on the shim, in a subprocess. While the
shim still differs, `_skip.known_divergence` turns the measured symptom into a
SKIP that names the issue; any other failure stays a FAIL; the day the shim is
fixed the test is an `ok` and the SKIP count drops by one.

**What makes these tests able to fail.** Each runs the shim and compares with
upstream's measured answer; a different symptom than the pinned one is a FAIL.

**What this cannot find.** Any other case of the same defect: one graph shape
for #358, one operator for #359.
"""

import json
import os
import pathlib
import subprocess
import sys

_TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]
_SUPPORT_DIR = _TESTS_DIR / "_support"
if str(_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPPORT_DIR))

import _skip  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_VENDOR_DIR = str(_REPO_ROOT / "torchnative" / "python")

_PROBE = r'''
import json, sys
import torch
assert hasattr(torch._C, "_aten_implemented"), torch.__file__

def independent_graph():
    # Upstream 2.13.0: [3.0, 3.0, 3.0].
    w = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    loss = (w * w).sum()
    loss2 = (w * 3.0).sum()
    loss.backward()
    (g,) = torch.autograd.grad(loss2, w)
    return g.tolist()

def subclass_overload():
    # Upstream 2.13.0: ['aten.add.Tensor', 'aten.add_.Tensor'] for a plain
    # tensor subclass made with _make_subclass.
    seen = []

    class S(torch.Tensor):
        @classmethod
        def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
            seen.append(str(func))
            with torch._C._DisableTorchDispatch():
                return func(*args, **(kwargs or {}))

    t = torch.Tensor._make_subclass(S, torch.ones(3))
    t + 1
    t.add_(1)
    return seen

try:
    out = {"independent_graph": independent_graph, "subclass_overload": subclass_overload}[sys.argv[1]]()
    print("RESULT " + json.dumps({"ok": out}))
except Exception as exc:
    print("RESULT " + json.dumps({"error": f"{type(exc).__name__}: {exc}"}))
'''


def _shim(case):
    env = dict(os.environ)
    env["PYTHONPATH"] = _VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run([sys.executable, "-c", _PROBE, case], capture_output=True,
                          text=True, env=env, timeout=300)
    line = next((ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT ")), None)
    assert line is not None, f"the shim probe printed no result (exit {proc.returncode}):\n{proc.stderr[-1500:]}"
    return json.loads(line[len("RESULT "):])


@_skip.known_divergence("#358", "a backward frees an independent graph",
                        "backward through the graph a second time")
def test_an_independent_graph_survives_another_graphs_backward():
    got = _shim("independent_graph")
    assert got == {"ok": [3.0, 3.0, 3.0]}, f"upstream gives [3.0, 3.0, 3.0]; the shim gave {got}"
    print("ok   known_divergences: an independent graph survives another graph's backward (#358 fixed)")


@_skip.known_divergence("#359", "a subclass sees the .Scalar overload", "aten.add.Scalar")
def test_a_subclass_sees_upstreams_overload_for_a_python_number():
    got = _shim("subclass_overload")
    want = {"ok": ["aten.add.Tensor", "aten.add_.Tensor"]}
    assert got == want, f"upstream hands the class {want['ok']}; the shim handed {got}"
    print("ok   known_divergences: a subclass sees upstream's overload (#359 fixed)")


def _main():
    tests = [
        ("test_an_independent_graph_survives_another_graphs_backward",
         test_an_independent_graph_survives_another_graphs_backward),
        ("test_a_subclass_sees_upstreams_overload_for_a_python_number",
         test_a_subclass_sees_upstreams_overload_for_a_python_number),
    ]
    return _skip.run_tests(tests, suite="test_known_divergences")


if __name__ == "__main__":
    raise SystemExit(_main())
