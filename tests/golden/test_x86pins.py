"""The golden harness's x86_64 Linux pins (issue #69): named, live, and narrow.

`tests/golden/compare.py`'s `KNOWN_X86_64_LINUX` demotes a failing golden case
to `X86KNOWN` (counted in `x86_known`, not in `failed`) only on x86_64 Linux and
only when the case is named in the table. This suite checks three things:

  (a) every pin names a case `cases.py` really builds (no dead pins: a renamed
      case would otherwise leave its pin matching nothing, and the real case
      would go back to FAIL with nobody told why the pin stopped working);
  (b) with the platform forced true through `classify`'s argument (not an env
      var production could set), a failing pinned case is X86KNOWN, a failing
      unpinned case stays FAIL, and a passing pinned case is a plain PASS;
  (c) with the platform false, the same pinned failure stays FAIL.

What this cannot find: whether the pins are the *right* set on the runner (only
a CI run of compare.py measures that), and whether a pinned case would fail for
a new, different reason on x86 (the pin is by name, so it would be hidden as
X86KNOWN; the detail is still printed on the X86KNOWN line).
"""

import json
import os
import pathlib
import subprocess
import sys

import _skip

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import compare  # noqa: E402

_NAMES_SCRIPT = r"""
import json, sys
sys.path.insert(0, sys.argv[1])
import compare
from loader import load_shim, resolve_torch_overload
import torch
c = load_shim(None)
compare._DTYPE_OWNERS[:] = [torch, c]
out = {}
for op in compare.KNOWN_X86_64_LINUX:
    builder = compare.CASE_BUILDERS[op]
    out[op] = [case.name for case in builder(torch, c, resolve_torch_overload(torch, op))]
print("NAMES " + json.dumps(out))
"""


def _staged_artefact():
    if os.environ.get("TORCH_C_ARTEFACT"):
        return os.environ["TORCH_C_ARTEFACT"]
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        cand = pathlib.Path(entry) / "_C.abi3.so"
        if entry and cand.is_file():
            return str(cand)
    return None


def _case_names():
    # A subprocess without PYTHONPATH: the gate's PYTHONPATH puts the vendored
    # shim `torch` first, and the builders need upstream torch, as compare.py
    # itself runs.
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    art = _staged_artefact()
    if art:
        env["TORCH_C_ARTEFACT"] = art
    proc = subprocess.run(
        [sys.executable, "-c", _NAMES_SCRIPT, str(HERE)],
        capture_output=True, text=True, env=env, timeout=600,
    )
    lines = [l for l in proc.stdout.splitlines() if l.startswith("NAMES ")]
    assert proc.returncode == 0 and lines, (
        f"could not build the golden case names (exit {proc.returncode}): {proc.stderr[-800:]}"
    )
    return json.loads(lines[0][len("NAMES "):])


def test_every_pin_names_a_case_cases_py_builds():
    names = _case_names()
    dead = []
    total = 0
    for op, pins in compare.KNOWN_X86_64_LINUX.items():
        for name, issue, reason in pins:
            total += 1
            assert issue and reason, (op, name)
            if name not in names[op]:
                dead.append((op, name))
    assert not dead, f"{len(dead)} pin(s) name no case cases.py builds: {dead}"
    assert total == 54, f"expected the 54 pinned names of the PR #90 CI run, found {total}"


def test_no_pin_is_a_whole_op_or_a_duplicate():
    for op, pins in compare.KNOWN_X86_64_LINUX.items():
        names = [p[0] for p in pins]
        assert len(names) == len(set(names)), f"{op}: duplicate pin"
        assert op not in names and "*" not in "".join(names), f"{op}: pin is not an exact case name"


def _one_pin():
    op, pins = next(iter(compare.KNOWN_X86_64_LINUX.items()))
    return op, pins[0][0]


def test_on_x86_a_failing_pinned_case_is_x86known_and_an_unpinned_one_stays_failed():
    op, name = _one_pin()
    assert compare.classify(op, name, passed=False, on_x86_64_linux=True) == "X86KNOWN"
    assert compare.classify(op, name + " (not pinned)", passed=False, on_x86_64_linux=True) == "FAIL"
    assert compare.classify("aten.not_pinned.default", name, passed=False, on_x86_64_linux=True) == "FAIL"
    assert compare.classify(op, name, passed=True, on_x86_64_linux=True) == "PASS"
    assert compare.x86_pin(op, name, True)[0].startswith("#")


def test_off_x86_the_same_pinned_failure_stays_failed():
    op, name = _one_pin()
    assert compare.classify(op, name, passed=False, on_x86_64_linux=False) == "FAIL"
    assert compare.x86_pin(op, name, False) is None


def test_the_production_predicate_is_skips_own():
    assert compare._on_x86_64_linux() == _skip.on_x86_64_linux()


if __name__ == "__main__":
    raise SystemExit(
        _skip.run_tests(
            [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)],
            suite="test_x86pins",
        )
    )
