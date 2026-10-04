"""A skip must be countable, in every suite, not just the ones fixed today.

The shape of the defect (see `_skip.py`'s docstring for the full story): a
test whose fixture was missing printed a hand-rolled

    print("   (skipped: <reason>)")
    return

and the suite's own `__main__` runner treats "returned without raising" as
success, so it printed `ok   {name}`. `suite_ledger.py`'s `tally()` only
counts lines matching ``^SKIP\\s`` -- three leading spaces and no ``SKIP``
prefix is invisible to it -- so the test landed in *neither* the ok nor the
SKIP column. A test gutted to ``pass`` behind such a guard would print an
identical log. 29 suite files had this shape; this file is what stops the
30th from reappearing without anyone noticing, here or in a suite written
after this one.

What this scan can see: the literal source shape ``print(<text containing
"(skipped">)`` immediately followed, in the same statement list, by a bare
``return``. That is every instance the original audit found (`grep -n
'(skipped' tests/*.py`, cross-checked against each file's
AST). What it cannot see, stated rather than implied:

  * A skip reported through some other unrecognised spelling that neither
    prints "(skipped" nor calls a registered skip helper -- this scan has no
    way to know a line is "the skip line" except by that substring or by a
    known helper call, so a new spelling invented without updating either
    list here would slip through silently, exactly like the defect it fixes.
  * A skip that does not `return` immediately after reporting (e.g. one that
    `continue`s a loop, or falls through to more code) -- the adjacency check
    only looks at the very next statement.
  * Two suites are deliberately excluded below (`test_vulkan4.py`,
    `test_coremlops.py`) because other rounds were editing them at the same
    time this file was written and collision was worse than a temporary gap.
    Removing that exclusion and re-running this test is the way to check
    they have since been fixed -- do not assume it from this file's history.
"""

import ast
import pathlib

HERE = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "AGENTS.md").is_file()) / "tests"

# See the module docstring: these two are mid-edit by other rounds as of
# this writing and are out of this file's scope. Re-check them by deleting
# this line and re-running.
_EXEMPT = {"test_vulkan4.py", "test_coremlops.py"}

# Spellings that register a skip so the ledger's runner can report it as
# SKIP instead of `ok`. A `print(...)` immediately before `return` is only a
# violation if nothing here fired first in the same statement list.
_SKIP_CALL_MARKERS = ("_skip.skip(", "_skipreg.skip(", "vulkan_coverage.vulkan_skip(")


def _suite_files():
    return sorted(
        p for p in HERE.rglob("test_*.py") if "__pycache__" not in p.parts
        if p.name not in _EXEMPT and p.name != "test_skipvis.py"
    )


def _is_print_call(stmt):
    return (isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call)
            and isinstance(stmt.value.func, ast.Name) and stmt.value.func.id == "print")


def _is_skip_registration(stmt, src_lines):
    seg = "\n".join(src_lines[stmt.lineno - 1:stmt.end_lineno])
    return any(marker in seg for marker in _SKIP_CALL_MARKERS)


def _silent_skip_sites(path):
    """[(lineno, snippet), ...] for every unreported "(skipped" + return."""
    src = path.read_text()
    lines = src.splitlines()
    tree = ast.parse(src, filename=str(path))
    violations = []

    class Visitor(ast.NodeVisitor):
        def visit_block(self, body):
            for i, stmt in enumerate(body):
                if _is_print_call(stmt):
                    seg = "\n".join(lines[stmt.lineno - 1:stmt.end_lineno])
                    if "(skipped" in seg and i + 1 < len(body) and isinstance(body[i + 1], ast.Return):
                        violations.append((stmt.lineno, seg.strip()))
                elif _is_skip_registration(stmt, lines):
                    # A registered skip is fine even if a neighbouring
                    # (unrelated) print happens to mention "(skipped" --
                    # nothing to flag here.
                    pass

        def generic_visit(self, node):
            for field, value in ast.iter_fields(node):
                if isinstance(value, list) and value and isinstance(value[0], ast.stmt):
                    self.visit_block(value)
                    for item in value:
                        self.visit(item)
                elif isinstance(value, ast.AST):
                    self.visit(value)
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, ast.AST):
                            self.visit(item)

    Visitor().visit(tree)
    return violations


def test_no_suite_reports_a_skip_through_a_line_the_ledger_cannot_count():
    offenders = {}
    for path in _suite_files():
        sites = _silent_skip_sites(path)
        if sites:
            offenders[path.name] = sites
    assert not offenders, (
        "these suites report a skip through a print the ledger's "
        "`^SKIP\\s` regex cannot see, so it is counted as neither ok nor "
        "skip (suite_ledger.py tally()): "
        + "; ".join(f"{name}:{[l for l, _ in s]}" for name, s in offenders.items())
    )


def test_every_suite_file_is_covered_by_this_scan_or_named_as_exempt():
    """A suite added later, with no `(skipped` text at all yet, should still
    be scanned -- this just proves the exemption list is closed and small,
    so a future silent skip in a NEW file cannot hide by never being
    checked. If this assertion ever needs to grow, that growth itself is
    the signal to look at."""
    all_suites = {p.name for p in HERE.rglob("test_*.py") if "__pycache__" not in p.parts}
    scanned = {p.name for p in _suite_files()}
    assert all_suites - scanned - {"test_skipvis.py"} == _EXEMPT, (
        "a suite file exists that this scan neither checks nor names as "
        "exempt -- add it to one"
    )


def test_the_scanner_itself_catches_the_defect_shape_when_present():
    """Break the invariant on purpose, in a string (not on disk), and
    confirm the scanner's AST walk actually flags it -- proof the detector
    is not vacuously green because `_suite_files()` came back empty or
    because the walk never reaches a nested function."""
    import tempfile
    bad_src = '''
def test_something():
    if True:
        print("   (skipped: no widget)")
        return
    assert False
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=str(HERE), prefix="test_skipvis_tmp_", delete=False
    ) as fh:
        fh.write(bad_src)
        tmp_path = pathlib.Path(fh.name)
    try:
        sites = _silent_skip_sites(tmp_path)
        assert sites, "the scanner did not flag a print+return it should have"
        assert sites[0][0] == 4, sites
    finally:
        tmp_path.unlink()


def test_a_registered_skip_is_not_flagged():
    """The positive case: `_skip.skip(...)` (or the vulkan/`_skipreg`
    equivalents) right before `return` is the fix, not a new violation."""
    import tempfile
    good_src = '''
def test_something():
    if True:
        _skip.skip("no widget")
        return
    assert False
'''
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".py", dir=str(HERE), prefix="test_skipvis_tmp_", delete=False
    ) as fh:
        fh.write(good_src)
        tmp_path = pathlib.Path(fh.name)
    try:
        assert _silent_skip_sites(tmp_path) == []
    finally:
        tmp_path.unlink()


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
