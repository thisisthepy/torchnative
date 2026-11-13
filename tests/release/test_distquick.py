"""`.github/scripts/dist_quick.py`: does the quick distributed run pick the
suites it claims, and can its verdict fail? (#273)

The quick run (`.github/workflows/dist-quick.yml`) is not the gate. What it
must not do is report green over fewer suites than it says, or over a suite
that failed. Each check below is paired with the mutation it catches
(AGENTS.md §17.5).
"""

import importlib.util
import os
import pathlib
import sys
import tempfile

REPO = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
SCRIPT = REPO / ".github/scripts/dist_quick.py"
WORKFLOW = REPO / ".github/workflows/dist-quick.yml"
SCRATCH = REPO / ".scratch"
SCRATCH.mkdir(exist_ok=True)


def _dq():
    spec = importlib.util.spec_from_file_location("_dist_quick", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _rels(dq, paths):
    return [dq._rel(p, dq.PYTESTS) for p in paths]


def test_every_distributed_suite_is_selected():
    dq = _dq()
    on_disk = sorted(p.relative_to(REPO / "tests").as_posix()
                     for p in (REPO / "tests/distributed").rglob("test_*.py"))
    assert on_disk, "no suites under tests/distributed"
    chosen = _rels(dq, dq.select())
    missing = [s for s in on_disk if s not in chosen]
    assert not missing, "distributed suites not selected: %s" % missing


def test_imported_suites_are_followed_and_the_follow_is_what_adds_them():
    """test_collect2 imports test_shim, so test_shim is selected; with the
    import-following step disabled it is not, so the step is what adds it."""
    dq = _dq()
    chosen = _rels(dq, dq.select())
    assert "_support/test_shim.py" in chosen, chosen
    bare = _rels(dq, dq.select(follow=False))
    assert "_support/test_shim.py" not in bare, bare


def test_imports_are_followed_transitively():
    dq = _dq()
    with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
        root = pathlib.Path(d)
        (root / "a").mkdir()
        (root / "a" / "test_one.py").write_text("import test_two\n")
        (root / "test_two.py").write_text("from test_three import x\nimport os\n")
        (root / "test_three.py").write_text("x = 1\n")
        (root / "test_four.py").write_text("x = 1\n")
        got = sorted(p.name for p in dq.select([root / "a"], pytests=root, repo=root))
    assert got == ["test_one.py", "test_three.py", "test_two.py"], got


def test_a_missing_root_refuses_by_name():
    dq = _dq()
    try:
        dq.select(["tests/no_such_dir"])
    except FileNotFoundError as e:
        assert "tests/no_such_dir" in str(e)
    else:
        raise AssertionError("a missing root was accepted")


def _run_one(dq, body, plan_runners=("linux", "macos")):
    with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
        root = pathlib.Path(d)
        suite = root / "test_case.py"
        suite.write_text(body)
        plan = {} if plan_runners is None else {"test_case.py": plan_runners}
        return dq.run([suite], root / "logs", sys.executable, dict(os.environ),
                      pytests=root, plan=plan)


def test_a_passing_suite_is_green():
    assert _run_one(_dq(), "print('ok   a')\n") == 0


def test_a_printed_fail_turns_the_run_red_even_at_exit_zero():
    assert _run_one(_dq(), "print('ok   a')\nprint('FAIL b: x')\n") == 1


def test_a_non_zero_exit_turns_the_run_red():
    assert _run_one(_dq(), "print('ok   a')\nraise SystemExit(3)\n") == 1


def test_an_unclassified_suite_turns_the_run_red():
    assert _run_one(_dq(), "print('ok   a')\n", plan_runners=None) == 1


def test_a_suite_not_planned_for_linux_is_skipped_not_run():
    # It would fail if run; the plan keeps it off Linux, so it is skipped by name.
    assert _run_one(_dq(), "raise SystemExit(5)\n", plan_runners=("macos",)) == 0


def test_an_empty_selection_is_red():
    dq = _dq()
    with tempfile.TemporaryDirectory(dir=SCRATCH) as d:
        assert dq.run([], pathlib.Path(d) / "logs", sys.executable, dict(os.environ),
                      pytests=pathlib.Path(d), plan={}) == 1


def test_the_workflow_is_dispatch_only_and_not_the_gate():
    text = WORKFLOW.read_text()
    head = text.split("\njobs:", 1)[0]
    assert "workflow_dispatch" in head
    for trigger in ("push:", "pull_request", "schedule:"):
        assert trigger not in head, "dist-quick.yml has a %s trigger" % trigger
    assert ".github/scripts/dist_quick.py run" in text
    assert "tests/run.sh" not in text, "dist-quick.yml runs the gate itself"


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
