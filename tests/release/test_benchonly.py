"""The competitor packages are bench-only: nothing outside tests/bench/ imports them (D11, #283).

onnxruntime, ai_edge_litert, ai_edge_torch, litert_torch and tflite_runtime are installed only
into .caches/bench-venv from tests/bench/requirements-competitors.txt. The gate, the wheel and
torchnative/ must never import them, or a user install would need them. Two checks:

  * no tracked .py file outside tests/bench/ has an `import X` / `from X import ...` of those
    top-level names (this file is exempt: it names them as data, not as imports);
  * every requirement line in requirements-competitors.txt is an exact `==` pin.

Nullification (AGENTS.md §17.5): add `import onnxruntime` to any tracked file under
torchnative/python/torchnative/, or change `torch==2.13.0` in the requirements file to
`torch>=2.13.0`; the matching test goes red and names the file or the line. The tests
`test_the_import_scan_can_fail` and `test_the_pin_check_can_fail` run the same checks on
synthetic input and require them to report.
"""

import pathlib
import re
import subprocess

REPO = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
BENCH_ONLY = ("onnxruntime", "ai_edge_litert", "ai_edge_torch", "litert_torch", "tflite_runtime")
REQ = "tests/bench/requirements-competitors.txt"
_IMPORT = re.compile(r"^\s*(?:import|from)\s+(" + "|".join(BENCH_ONLY) + r")\b", re.M)


def _tracked_py():
    done = subprocess.run(["git", "ls-files", "-z", "*.py"], cwd=REPO, capture_output=True)
    assert done.returncode == 0, done.stderr.decode()
    return [n.decode() for n in done.stdout.split(b"\0") if n]


def _offenders(read, names):
    out = []
    for n in names:
        if n.startswith("tests/bench/") or n == "tests/release/test_benchonly.py":
            continue
        m = _IMPORT.search(read(n))
        if m:
            out.append((n, m.group(1)))
    return sorted(out)


def _unpinned(lines):
    bad = []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        if not re.fullmatch(r"[A-Za-z0-9._\-\[\]]+==[A-Za-z0-9._!+*\-]+", s):
            bad.append(s)
    return bad


def _read(name):
    return (REPO / name).read_text(encoding="utf-8", errors="replace")


def test_nothing_outside_tests_bench_imports_the_bench_only_packages():
    names = _tracked_py()
    assert any(n.startswith("tests/") for n in names), "git ls-files returned no tests: the scan is empty"
    bad = _offenders(_read, names)
    assert not bad, f"bench-only packages imported outside tests/bench/ (D11, #283): {bad}"


def test_the_import_scan_can_fail():
    files = {"torchnative/x.py": "import os\nimport onnxruntime as ort\n",
             "tests/bench/ok.py": "import onnxruntime\n",
             "a/y.py": "from ai_edge_litert import interpreter\n",
             "a/z.py": "x = 'import onnxruntime'\n"}
    assert _offenders(files.__getitem__, sorted(files)) == [
        ("a/y.py", "ai_edge_litert"), ("torchnative/x.py", "onnxruntime")]


def test_requirements_competitors_pins_every_line_with_exact_equals():
    text = _read(REQ)
    lines = [l for l in text.splitlines() if l.strip() and not l.strip().startswith("#")]
    assert lines, f"{REQ} has no requirement lines"
    bad = _unpinned(text.splitlines())
    assert not bad, f"{REQ}: lines not pinned with ==: {bad}"
    names = {l.split("==")[0].lower() for l in lines}
    for need in ("torch", "transformers", "onnxruntime", "ai-edge-litert"):
        assert need in names, f"{REQ} does not pin {need}"


def test_the_pin_check_can_fail():
    assert _unpinned(["torch==2.13.0", "# c", "", "onnx>=1.0", "numpy", "x~=1.2"]) == [
        "onnx>=1.0", "numpy", "x~=1.2"]


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
