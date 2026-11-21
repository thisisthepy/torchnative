"""Run the distributed suites alone, on a branch, without the full gate (#273).

Leader decision, 2026-11-13: a dispatch-only workflow
(`.github/workflows/dist-quick.yml`) runs `tests/distributed` plus the suites
those import, to shorten the turnaround of each step of
docs/distributed/BACKEND.md. **It is not the gate.** `test.yml` stays the merge
gate; a green run here says only that the suites listed in its `DIST QUICK:`
lines passed on Linux.

Selection (`select`):

  * every `test_*.py` under each root given (a directory or one file), and
  * every `test_*` module those suites import (`import test_x`,
    `from test_x import ...`), resolved against the `test_*.py` files under
    `tests/`, transitively. `test_collect2` imports `test_shim`, so
    `_support/test_shim.py` runs too: a helper that changes under a
    distributed suite is exercised in the same run.

What the selection cannot see: an import spelled through `importlib` or
`__import__`, and a suite that does not import a module but depends on it
through a subprocess. Those reach this run only when named as a root.

Running (`run`) honours the Linux plan in `gate_suites.py`: a selected suite
not planned for the Linux runner is not executed and gets the same
`SKIP <suite>: (whole suite)` line the ledger prints. Counts use
`suite_ledger.tally`, so they are counted the way the gate counts them.

Usage:
    python .github/scripts/dist_quick.py select [ROOT ...]
    python .github/scripts/dist_quick.py run --logs DIR --suite-env K=V ... [ROOT ...]
"""

import argparse
import ast
import importlib.util
import os
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
PYTESTS = REPO / "tests"
DEFAULT_ROOTS = ("tests/distributed",)
RUNNER = "linux"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def suite_files(pytests=PYTESTS):
    """{module name: path} for every suite under `pytests`."""
    found = {}
    for p in sorted(pathlib.Path(pytests).rglob("test_*.py")):
        if "__pycache__" not in p.parts:
            found[p.stem] = p
    return found


def imported_suites(path, known):
    """The `test_*` modules `path` imports that are suites in `known`."""
    tree = ast.parse(pathlib.Path(path).read_text(), filename=str(path))
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module.split(".")[0])
    return sorted(n for n in names if n.startswith("test_") and n in known)


def select(roots=DEFAULT_ROOTS, pytests=PYTESTS, repo=REPO, follow=True):
    """Sorted suite paths: the roots' suites plus what they import, transitively."""
    known = suite_files(pytests)
    queue = []
    for root in roots:
        p = pathlib.Path(root)
        p = p if p.is_absolute() else pathlib.Path(repo) / p
        if p.is_file():
            queue.append(p.resolve())
        elif p.is_dir():
            queue.extend(q.resolve() for q in sorted(p.rglob("test_*.py"))
                         if "__pycache__" not in q.parts)
        else:
            raise FileNotFoundError("dist_quick: no such suite or directory: %s" % root)
    chosen = set()
    while queue:
        p = queue.pop()
        if p in chosen:
            continue
        chosen.add(p)
        if follow:
            queue.extend(known[n].resolve() for n in imported_suites(p, known))
    return sorted(chosen)


def _rel(p, pytests):
    return pathlib.Path(p).resolve().relative_to(pathlib.Path(pytests).resolve()).as_posix()


def run(suites, logs, python, env, pytests=PYTESTS, plan=None, out=None):
    """Run `suites`; 1 if any failed, printed a FAIL, or is unclassified.

    `plan` maps a suite path relative to `pytests` to its runner tuple; it
    defaults to `gate_suites.SUITES` (tests pass their own)."""
    ledger = _load(PYTESTS / "suite_ledger.py", "_suite_ledger")
    gs = _load(REPO / ".github" / "scripts" / "gate_suites.py", "_gate_suites")
    plan = gs.SUITES if plan is None else plan
    out = sys.stdout if out is None else out

    def say(text):
        out.write(text)
        out.flush()

    logs = pathlib.Path(logs)
    logs.mkdir(parents=True, exist_ok=True)
    say(("DIST QUICK: this is NOT the gate (test.yml is). Suites: %d" % len(suites)) + "\n")
    totals = [0, 0, 0]
    bad = []
    for suite in suites:
        rel = _rel(suite, pytests)
        name = pathlib.Path(suite).name
        log = logs / (name + ".log")
        say(("--- %s ---" % rel) + "\n")
        runners = plan.get(rel)
        if runners is None:
            text = "FAIL %s: not classified in .github/scripts/gate_suites.py\n" % rel
            rc = 1
        elif RUNNER not in runners:
            text = gs.skip_line(name, RUNNER, gs.WHY_NOT.get(rel, "no reason recorded")) + "\n"
            rc = 0
        else:
            with open(log, "wb") as fh:
                rc = subprocess.call([python, str(suite)], stdout=fh,
                                     stderr=subprocess.STDOUT, env=env)
            text = log.read_bytes().decode("utf-8", "replace")
        say(text)
        counts = ledger.tally(text)
        for i in range(3):
            totals[i] += counts[i]
        say("DIST QUICK: %s ok=%d FAIL=%d SKIP=%d exit=%d\n" % ((rel,) + counts + (rc,)))
        if rc != 0 or counts[1]:
            bad.append(rel)
    say("DIST QUICK: suites=%d ok=%d FAIL=%d SKIP=%d (not the gate)\n"
        % ((len(suites),) + tuple(totals)))
    if bad:
        say(("DIST QUICK: failing suites: " + ", ".join(bad)) + "\n")
    return 1 if bad or not suites else 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select")
    s.add_argument("roots", nargs="*")
    r = sub.add_parser("run")
    r.add_argument("--logs", required=True)
    r.add_argument("--python", default=sys.executable)
    r.add_argument("--suite-env", action="append", default=[])
    r.add_argument("roots", nargs="*")
    args = ap.parse_args(argv)
    suites = select(args.roots or DEFAULT_ROOTS)
    if args.cmd == "select":
        for p in suites:
            print(_rel(p, PYTESTS))
        return 0
    env = dict(os.environ)
    for entry in args.suite_env:
        k, _, v = entry.partition("=")
        env[k] = v
    return run(suites, args.logs, args.python, env)


if __name__ == "__main__":
    raise SystemExit(main())
