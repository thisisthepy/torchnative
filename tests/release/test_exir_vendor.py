"""The vendored exir tree is what its manifest says, and imports nothing it must not (issue #423, step R5-V).

`torchnative/python/torchnative/_vendor/executorch` is exir's passes from
ExecuTorch v1.4.1 without its serializer, written by
`scripts/vendor/vendor_exir.py` (owner, 2026-11-21: "Rust로 재구현", option (c):
only exir's passes are vendored, no executorch Python package in the wheel).
`tests/export/test_et_vendored.py` checks that it lowers to executorch's bytes.

**What makes each test able to fail.**

* `test_the_tree_matches_its_manifest`: a hand edit, a deleted file or a stray
  file in the tree fails, naming the file.
* `test_the_manifest_check_can_fail`: the same check on a copy with one byte
  changed, one file removed and one added must report all three; a check that
  always answers "no problems" fails here.
* `test_no_vendored_module_imports_executorch`: an `import executorch...` the
  rename missed would make the tree depend on (or shadow) an installed
  executorch; it fails naming file and line.
* `test_every_third_party_import_is_guarded`: an import of ruamel, tabulate,
  torchao, numpy or flatbuffers outside a `try` that catches ImportError would
  make the wheel need that package; it fails naming file and line.
* `test_the_wheel_takes_the_tree_and_its_data`: pyproject's package list and
  package data must cover `torchnative._vendor.executorch` and its `edge.json`.

**What this suite cannot find.** Whether the tree lowers correctly: that is
`test_et_vendored.py`. Imports done through `importlib` with a computed name.
"""

import ast
import pathlib
import shutil
import sys
import tomllib
import uuid

REPO = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
sys.path.insert(0, str(REPO / "scripts" / "vendor"))

import vendor_exir  # noqa: E402

TREE = vendor_exir.DEST
THIRD_PARTY = ("ruamel", "tabulate", "torchao", "numpy", "flatbuffers")


def _modules():
    return sorted(p for p in TREE.rglob("*.py") if "__pycache__" not in p.parts)


def _imports(tree):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                yield node, a.name
        elif isinstance(node, ast.ImportFrom) and not node.level and node.module:
            yield node, node.module


def _guarded(tree, node):
    for t in ast.walk(tree):
        if isinstance(t, ast.Try) and t.body[0].lineno <= node.lineno <= t.body[-1].end_lineno:
            for h in t.handlers:
                if h.type is None:
                    return True
                names = [getattr(x, "id", getattr(x, "attr", ""))
                         for x in (h.type.elts if isinstance(h.type, ast.Tuple) else [h.type])]
                if {"ImportError", "ModuleNotFoundError"} & set(names):
                    return True
    return False


def test_the_tree_matches_its_manifest():
    problems = vendor_exir.check()
    assert not problems, "\n".join(problems)
    print(f"ok   exir_vendor: {len(_modules())} modules match {vendor_exir.MANIFEST}")


def test_the_manifest_check_can_fail():
    scratch = REPO / ".scratch" / f"exir-vendor-mutant-{uuid.uuid4().hex[:8]}"
    try:
        shutil.copytree(TREE, scratch / "executorch")
        shutil.copy(TREE.parent / "__init__.py", scratch / "__init__.py")
        dest = scratch / "executorch"
        assert vendor_exir.check(dest) == [], "the unmodified copy must pass"
        changed = dest / "exir" / "__init__.py"
        changed.write_bytes(changed.read_bytes() + b"\n")
        (dest / "exir" / "dialects" / "edge" / "edge.json").unlink()
        (dest / "stray.py").write_text("")
        problems = "\n".join(vendor_exir.check(dest))
        assert "changed: exir/__init__.py" in problems, problems
        assert "missing: exir/dialects/edge/edge.json" in problems, problems
        assert "not in the manifest: stray.py" in problems, problems
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def test_no_vendored_module_imports_executorch():
    bad = []
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node, name in _imports(tree):
            if name == "executorch" or name.startswith("executorch."):
                bad.append(f"{path.relative_to(TREE)}:{node.lineno} {name}")
    assert not bad, "\n".join(bad)


def test_every_third_party_import_is_guarded():
    bad, guarded = [], 0
    for path in _modules():
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node, name in _imports(tree):
            if name.split(".")[0] not in THIRD_PARTY:
                continue
            if _guarded(tree, node):
                guarded += 1
            else:
                bad.append(f"{path.relative_to(TREE)}:{node.lineno} {name}")
    assert not bad, "\n".join(bad)
    # The script patches three such imports into guarded ones; zero would mean
    # the scan saw nothing.
    assert guarded >= 3, guarded
    print(f"ok   exir_vendor: {guarded} third-party imports, all guarded")


def test_the_wheel_takes_the_tree_and_its_data():
    doc = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    st = doc["tool"]["setuptools"]
    assert st["package-dir"][""] == "torchnative/python"
    assert "torchnative*" in st["packages"]["find"]["include"]
    assert "**/*" in st["package-data"]["torchnative"]
    assert (TREE / "exir" / "dialects" / "edge" / "edge.json").is_file()
    assert (TREE / "__init__.py").is_file() and (TREE.parent / "__init__.py").is_file()


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
