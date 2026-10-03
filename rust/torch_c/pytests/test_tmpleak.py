"""The gate must not leave its temporary directories behind.

Measured on develop at 78cf555, one full `run.sh` with `TMPDIR` pointed at an
empty directory: **53 entries, 40 MB, left behind after the run had exited 0.**
Three of them were `golden-harness-*` -- one full copy of `_C.abi3.so`
(8.7 MB) for every golden invocation (compare, self-test, DOCWATCH), because
`tools/golden/loader.py` copies the artefact into a private `mkdtemp` and
never removes it. The rest were the per-fixture directories of `test_shim.py`,
`test_collect2.py`, `test_npu2.py` and `test_qnn.py`: checkpoints, federated
rendezvous files, and the `collect2-harness-probe-*` rank logs.

That last one is not only a storage problem. Its rank logs are 1 MB runs of a
single character, and when `TMPDIR` lives inside the checkout -- which the
rule "nothing is created outside this repository" (CLAUDE.md §7) asks for --
`test_docrefs.py` walked into it. `test_docrefs.py` read every file in the
tree, tracked or not, and its `DOCS_REF` pattern started with an unbounded
`[A-Za-z0-9_.:-]*` prefix, which is quadratic on a 1 MB run with no `docs/`
in it. A sibling worktree's gate sat in that suite at 100% CPU for over half
an hour. A leaked directory became a hung gate. (GitHub issue #8 fixed both
sides there: it now reads only `git ls-files`, its pattern starts at the
literal `docs/`, and a test bounds a 1 MB single-character input by timeout.
A leak is still a leak, which is what this file pins.)

What this file pins:

1. **Behaviourally, for the largest leak:** the golden loader, run in a
   subprocess against a private, empty `TMPDIR`, leaves that directory empty
   after the process exits -- and actually loaded the shim, so an empty
   directory cannot come from a loader that failed before creating anything.
2. **Structurally, for every `tempfile.mkdtemp` the gate reaches** (this
   directory and `tools/golden/`): each one is bound to a name, and that name
   is either handed to `atexit.register(<shutil>.rmtree, name, ...)` within
   three lines, or the next statement is a `try:` whose `finally` removes it.

What this CANNOT find (CLAUDE.md §5.4):

* directories created by third parties on our behalf -- coremltools'
  compiled `.mlmodelc` bundles (the `tmp*` entries), upstream torch's
  `torchinductor_<user>` cache, `publish-main-build.log`;
* `mkstemp`, `NamedTemporaryFile(delete=False)`, `TemporaryDirectory`
  objects that are never closed;
* anything left by a process killed with SIGKILL or by a subprocess
  timeout -- `atexit` does not run then;
* whether the structural rule's cleanup is *reached*: it is a source scan,
  and `docs/devices/MPSATTN.md` §3.1 records how a scan can be stepped
  around. Item 1 is the behavioural half, and it covers one site, not all.
"""

import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

import _skip

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[2]
GOLDEN = REPO / "tools" / "golden"

#: The files whose temporary directories the gate creates: every suite and
#: helper in this directory, and the golden harness `run.sh` invokes.
SCANNED = sorted(HERE.glob("*.py")) + sorted(GOLDEN.glob("*.py"))

MKDTEMP = re.compile(r"mkdtemp\(")
BOUND = re.compile(r"^\s*(?P<name>[A-Za-z_][A-Za-z0-9_]*)\s*=\s*tempfile\.mkdtemp\(")


def _artefact():
    """The `_C.abi3.so` this gate run staged, or None."""
    explicit = os.environ.get("TORCH_C_ARTEFACT")
    if explicit and os.path.isfile(explicit):
        return explicit
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep):
        candidate = os.path.join(entry, "_C.abi3.so")
        if entry and os.path.isfile(candidate):
            return candidate
    return None


def test_the_golden_loader_leaves_no_copy_of_the_artefact_behind():
    artefact = _artefact()
    if artefact is None:
        _skip.skip("no staged _C.abi3.so on PYTHONPATH and TORCH_C_ARTEFACT unset")
        return
    private = tempfile.mkdtemp(prefix="tmpleak-probe-")
    try:
        env = dict(os.environ)
        env["TMPDIR"] = private
        src = (
            "import sys\n"
            "sys.path.insert(0, %r)\n"
            "import loader\n"
            "m = loader.load_shim(%r)\n"
            "print('LOADED', hasattr(m, '_aten_implemented'))\n"
        ) % (str(GOLDEN), artefact)
        proc = subprocess.run(
            [sys.executable, "-c", src], capture_output=True, text=True,
            env=env, timeout=300,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        # Not vacuous: the copy was made and imported, so the directory
        # existed. An empty directory below therefore means it was removed.
        assert "LOADED True" in proc.stdout, proc.stdout[-2000:]
        left = sorted(os.listdir(private))
        assert left == [], (
            "the golden loader left %s behind in TMPDIR after its process "
            "exited -- each one is a full copy of _C.abi3.so" % left)
    finally:
        shutil.rmtree(private, ignore_errors=True)


def _violations(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    out = []
    for i, line in enumerate(lines):
        if not MKDTEMP.search(line) or line.lstrip().startswith("#"):
            continue
        m = BOUND.match(line)
        where = "%s:%d" % (path.relative_to(REPO), i + 1)
        if m is None:
            out.append("%s: mkdtemp() is not bound to a name, so nothing can "
                       "remove it: %s" % (where, line.strip()))
            continue
        name = m.group("name")
        window = "\n".join(lines[i + 1:i + 4])
        registered = re.search(
            r"atexit\.register\(\s*[A-Za-z_]*shutil\.rmtree\s*,\s*%s\b" % re.escape(name),
            window)
        following = next((l.strip() for l in lines[i + 1:] if l.strip()), "")
        guarded = following == "try:" and re.search(
            r"rmtree\(\s*%s\b" % re.escape(name), "\n".join(lines[i + 1:]))
        if not (registered or guarded):
            out.append("%s: `%s` from mkdtemp() is neither registered with "
                       "atexit nor removed in a following try/finally"
                       % (where, name))
    return out


def test_every_temporary_directory_the_gate_makes_is_removed():
    me = pathlib.Path(__file__).resolve()
    bad = []
    seen = 0
    for path in SCANNED:
        if path == me:
            continue  # this file names the pattern while explaining it
        found = _violations(path)
        bad.extend(found)
        seen += len(MKDTEMP.findall(path.read_text(encoding="utf-8")))
    # The scan must have something to judge: a glob that silently matched
    # nothing would pass with zero violations.
    assert seen >= 15, "only %d mkdtemp sites found -- is SCANNED right?" % seen
    assert not bad, "temporary directories that outlive the gate:\n  " + "\n  ".join(bad)


def test_the_structural_rule_rejects_the_shapes_it_exists_for():
    """The scanner, against inputs written to fail it, so it cannot be a no-op."""
    probe = REPO / ".scratch"
    probe.mkdir(exist_ok=True)
    cases = {
        "unbound": 'x = os.path.join(tempfile.mkdtemp(prefix="a-"), "f")\n',
        "never_removed": 'd = tempfile.mkdtemp()\nuse(d)\n',
        "removes_another": 'd = tempfile.mkdtemp()\natexit.register(shutil.rmtree, e)\n',
        "too_late": 'd = tempfile.mkdtemp()\na()\nb()\nc()\natexit.register(shutil.rmtree, d)\n',
    }
    accepted = {
        "atexit": 'd = tempfile.mkdtemp()\natexit.register(shutil.rmtree, d, ignore_errors=True)\n',
        "aliased": 'd = tempfile.mkdtemp()\natexit.register(_shutil.rmtree, d)\n',
        "finally": 'd = tempfile.mkdtemp()\ntry:\n    use(d)\nfinally:\n    shutil.rmtree(d)\n',
    }
    work = pathlib.Path(tempfile.mkdtemp(prefix="tmpleak-rule-", dir=probe))
    try:
        for label, text in list(cases.items()) + list(accepted.items()):
            f = work / ("%s.py" % label)
            f.write_text(text)
            # relative_to(REPO) inside _violations needs a path under REPO.
            got = _violations(f)
            if label in cases:
                assert got, "the rule accepted the %r shape" % label
            else:
                assert not got, "the rule rejected the %r shape: %s" % (label, got)
    finally:
        shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    failures = _skip.run_tests(
        [(name, fn) for name, fn in sorted(globals().items())
         if name.startswith("test_")],
        suite="test_tmpleak",
    )
    raise SystemExit(1 if failures else 0)
