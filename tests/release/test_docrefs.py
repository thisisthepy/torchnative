"""Do the documentation paths this tree writes down still point at documents?

`docs/` was a flat list of 183 files, and roughly six thousand strings across
Markdown prose, Rust comments, Python docstrings, DOCWATCH markers and workflow
files name those files by path. Sorting them into subfolders is only safe if
every one of those strings moves with its target, and a missed one is invisible:
nothing dereferences a path written in a comment.

The specific failure this file was written against is sharper than a broken
link, and it is the shape AGENTS.md §17.5 records repeatedly -- a check that
cannot fail. `run.sh` fed the documentation checker with a shell glob:

    "$repo_root"/docs/*.md "$repo_root/README.md"

A glob does not recurse. The moment a document moved into `docs/<folder>/` it
would drop out of DOCWATCH silently: the gate stays green, the marker count gets
*smaller*, and nothing says a word. `check_docs.py`'s own default was the same
glob (`DOCS_DIR.glob("*.md")`), which is the more dangerous of the two because
it under-reports for anyone who invokes the checker directly rather than through
the gate. Both are now recursive, and `test_docwatch_enumerates_docs_recursively`
below fails if either reverts.
What this cannot see, stated rather than implied:

  * **Interpolated paths.** `f"docs/platform/RELEASE_{version}.md"` has no literal
    filename, so neither the reference scan nor the split-literal scan can check
    it. Two of these were in `test_release.py` and the gate found them by
    FileNotFoundError, one run apart. If you build a documentation path from a
    variable, the suite that reads it is the only thing that will notice.
  * **References inside the vendored tree** (`torchnative/python/torch/`),
    which is upstream's and regenerated.
  * **Files git does not track.** The scan is `git ls-files` (GitHub issue #8):
    scratch notes, logs, caches and sibling worktrees inside the checkout are
    not documentation, and reading them made the gate fail on a stale note and
    hang on a 1 MB log. A new file is checked once it is `git add`ed.
"""

import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

REPO = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
DOCS = REPO / "docs"
RUN_SH = REPO / "tests/run.sh"
CHECK_DOCS = REPO / "tests/docwatch/check_docs.py"

# A `docs/NAME.md` string anywhere in a text file. References to *other*
# projects' docs -- `pypackpack/docs/SPEC.md`, torch-mlir's
# `.../blob/main/docs/architecture.md` -- are told apart by the whole line they
# sit on (FOREIGN_PREFIXES, in `_references_in`), not by the pattern.
#
# The pattern used to open with `(?P<prefix>[A-Za-z0-9_.:-]*/?)` for that
# purpose. The group was captured and never read once the line test replaced
# it, and it made matching quadratic: at every position of a long run of those
# characters the engine consumed the rest of the run before backtracking to
# look for `docs/`. A leaked 1 MB single-character log held the gate at 100%
# CPU for over half an hour (GitHub issue #8). Starting at the literal `docs/`
# finds the same matches -- the prefix was optional, so every match it made
# had a `docs/` the bare pattern also finds -- and
# `test_matching_is_linear_on_one_megabyte_of_a_single_character` fails, rather
# than hangs, if anything quadratic comes back.
#
# The folder group is what makes this match the post-move spelling. Without it
# the pattern could only match `docs/NAME.md`, so once every reference became
# `docs/<folder>/NAME.md` the check would have matched nothing and passed
# vacuously -- it did exactly that when first written, and the deliberate-fault
# run below is what caught it. A test that cannot fail is not a test.
DOCS_REF = re.compile(
    r"docs/(?:(?P<folder>[a-z_]+)/)?(?P<name>[A-Za-z0-9_.-]+\.md)"
)

# Relative Markdown links inside docs/, e.g. `[`docs/SCALAR.md`](SCALAR.md)`.
# These carry no `docs/` prefix, so the substitution that moved every
# `docs/NAME.md` string did not touch them -- and they only break once the
# linking and linked documents land in *different* folders, which is exactly
# what sorting into subfolders does. Resolved from the linking file's directory.
# The target may contain `/` -- after the sort these links are `../folder/NAME.md`.
# The first version of this pattern excluded `/`, so it matched nothing once the
# links were rewritten and passed vacuously; the deliberate-fault run caught it.
REL_LINK = re.compile(r"\]\((?!https?:|#)(?P<target>[A-Za-z0-9_./-]+\.md)\)")

# Documents named in prose that do not exist, with the reason each is allowed.
# Named, not pattern-matched: a new dangling reference must fail this test, so
# the exception list is the only thing that may grow, and only deliberately.
KNOWN_MISSING = {
    "docs/TAIL2.md": (
        "referenced by docs/kernels/TAIL3.md's opening sentence, but the document "
        "was never committed -- tests/ops/test_tail2.py exists, so the "
        "round happened and only its write-up is missing. Pre-existing; found "
        "while sorting docs/ into subfolders and deliberately not invented here. "
        "Delete this entry when TAIL2.md lands."
    ),
}

# Files whose `docs/...` strings are fixture paths in a throwaway repository they
# build, not references into this tree. Named, like KNOWN_MISSING, so the list
# only grows deliberately. `test-sync-release.sh` is shared across the
# thisisthepy repositories (here with its throwaway directory under `.scratch/`
# rather than `.tmp/`, AGENTS.md §2) and creates `docs/a.md`, `docs/sub/c.md`,
# ... there to test what `sync-release.sh` drops.
FIXTURE_FILES = {".github/scripts/release/test-sync-release.sh"}

# Prefixes that mean the reference belongs to another project, not this one.
FOREIGN_PREFIXES = ("pypackpack/", "torch-mlir/", "https://", "http://")


SELF = pathlib.Path(__file__).resolve()
BINARY_SUFFIXES = {".png", ".jpg", ".so", ".dylib", ".a", ".pt", ".bin", ".safetensors", ".whl", ".zip"}


def _tracked_files(repo=REPO):
    """The files git tracks in `repo`, from its index. Refuses by name rather
    than falling back to a directory walk: a walk is the defect this replaces."""
    try:
        out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"],
                             check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError) as e:
        detail = getattr(e, "stderr", b"") or b""
        raise RuntimeError(
            f"test_docrefs scans the files git tracks and could not list them in "
            f"{repo}: {e} {detail.decode(errors='replace').strip()}"
        ) from e
    return [repo / os.fsdecode(n) for n in out.split(b"\0") if n]


def _text_files(repo=REPO):
    """Every tracked text file -- `git ls-files`, not a directory walk.

    The walk this replaces (`rglob("*")` minus a skip list) matched "tracked"
    when it was written on 2026-09-07, because nothing untracked of size lived
    in the checkout. It stopped matching when `.worktrees/` and `.caches/`
    (2026-09-22) and `.scratch/` (2026-10-02) moved inside: a scratch note
    quoting an old path turned the gate red, the main checkout's gate re-read
    every worktree's copy of the docs, and a leaked 1 MB log hung it (GitHub
    issue #8). The index is the definition the docstring always gave.

    `torchnative/python/torch/` needs no special case any more: upstream's
    vendored files there are gitignored, so they are not listed, and the two
    of ours that are tracked beside them (`README.md`, `nn/federated.py`) are
    read like any other file. A file staged with `git add` is included; a new
    file that has not been added is not, until it is."""
    for p in _tracked_files(repo):
        if not p.is_file():
            continue  # deleted in the working tree, or a submodule entry
        if p.suffix.lower() in BINARY_SUFFIXES:
            continue
        try:
            yield p, p.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binary (SPIR-V, .npy, ...) -- no prose to check


def _references_in(text):
    """Every `docs/...md` reference in one file's text that is ours to resolve,
    as (ref, folder, name).

    Line by line, so each line is judged once. The previous form sliced the
    enclosing line out of the whole text for every match, which is quadratic
    on one long line holding many matches even with a linear pattern."""
    for line in text.split("\n"):
        if "docs/" not in line:
            continue
        # A URL or another project's checkout is not ours to resolve. Judge
        # by the whole line, not the few characters before the match:
        # `https://github.com/llvm/torch-mlir/blob/main/docs/architecture.md`
        # leaves only `main/` in front of `docs/`, which looks local.
        if any(f in line for f in FOREIGN_PREFIXES):
            continue
        for m in DOCS_REF.finditer(line):
            folder, name = m.group("folder"), m.group("name")
            ref = f"docs/{folder}/{name}" if folder else f"docs/{name}"
            if ref in KNOWN_MISSING:
                continue
            yield ref, folder, name


def _checked_references(repo=REPO):
    """(file, ref, exists) for every reference the dangling check judges."""
    docs = repo / "docs"
    for path, text in _text_files(repo):
        if path == SELF:
            continue  # this file quotes example paths while explaining itself
        rel = path.relative_to(repo).as_posix()
        if rel in FIXTURE_FILES:
            continue
        for ref, folder, name in _references_in(text):
            if folder:
                # Fully-qualified: it must exist exactly where it says.
                exists = (docs / folder / name).is_file()
            else:
                # Unqualified (a historical `git show <rev>:docs/NAME.md` path,
                # or a reference nobody re-pointed): accept it if the document
                # exists anywhere under docs/.
                exists = (docs / name).is_file() or bool(list(docs.glob(f"*/{name}")))
            yield rel, ref, exists


def _dangling_references(repo=REPO):
    dangling = {}
    for rel, ref, exists in _checked_references(repo):
        if not exists:
            dangling.setdefault(ref, []).append(rel)
    return dangling


def test_no_docs_reference_in_the_tree_dangles():
    """Every `docs/[<folder>/]NAME.md` string names a file that exists."""
    dangling = _dangling_references()
    assert not dangling, (
        "documentation references that point at no file:\n"
        + "\n".join(f"  {r}  <- {', '.join(sorted(set(w))[:4])}" for r, w in sorted(dangling.items()))
    )


def test_known_missing_documents_are_still_missing():
    """The exception list may not outlive its reason. If TAIL2.md is written,
    this fails and the entry must be deleted -- an allow-list that silently
    covers a file that now exists is how an exception becomes permanent."""
    for ref, why in KNOWN_MISSING.items():
        name = ref.split("/")[-1]
        found = list(DOCS.glob(f"*/{name}")) + ([DOCS / name] if (DOCS / name).exists() else [])
        assert not found, (
            f"{ref} now exists ({found[0].relative_to(REPO) if found else ''}) but is still "
            f"listed as KNOWN_MISSING. Remove the entry. Reason it was listed: {why}"
        )


def test_relative_markdown_links_inside_docs_resolve():
    """`[`docs/SCALAR.md`](SCALAR.md)` resolves against the linking file's own
    directory, so it survives a flat tree and breaks the moment the two files
    land in different folders. The `docs/NAME.md` rewrite does not see these."""
    broken = []
    for path in DOCS.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for m in REL_LINK.finditer(text):
            target = (path.parent / m.group("target")).resolve()
            if not target.exists():
                broken.append(f"{path.relative_to(REPO)} -> {m.group('target')}")
    assert not broken, "relative links that do not resolve:\n  " + "\n  ".join(broken)


def test_every_document_lives_in_a_subfolder():
    """The habit this sorting exists to break: a round drops its document at the
    top of `docs/` because there is no visible place for it. `docs/README.md` is
    the index. `INTENT.md` and `SPEC.md` are the project's intent and behavioural
    contract (AGENTS.md rule 5) and live here by the ecosystem-wide layout; they
    are named, not pattern-matched, so a round document still cannot slip in."""
    allowed = {"README.md", "INTENT.md", "SPEC.md"}
    loose = sorted(p.name for p in DOCS.glob("*.md") if p.name not in allowed)
    assert not loose, (
        f"{len(loose)} document(s) at the top level of docs/: {', '.join(loose)}.\n"
        "Put each in the folder matching its subject -- docs/README.md lists them "
        "and says what belongs in each."
    )


def test_docs_readme_names_every_folder_and_only_real_ones():
    """The index and the tree agree in both directions. A folder missing from the
    index is a folder the next round will not find; a folder in the index that
    does not exist sends them somewhere that is not there."""
    readme = (DOCS / "README.md").read_text(encoding="utf-8")
    on_disk = {p.name for p in DOCS.iterdir() if p.is_dir() and not p.name.startswith(".")}
    # Only the index table's first column, so prose mentioning `docs/`
    # is not mistaken for a folder name.
    named = set(re.findall(r"^\| `([a-z_]+)/` \|", readme, re.M))
    assert on_disk <= named, f"folders not described in docs/README.md: {sorted(on_disk - named)}"
    assert named <= on_disk, f"docs/README.md describes folders that do not exist: {sorted(named - on_disk)}"


def test_docwatch_enumerates_docs_recursively():
    """The nullification guard. Both enumerations must recurse; if either reverts
    to a non-recursive glob, every document in a subfolder leaves DOCWATCH
    silently and the marker count drops while the gate stays green."""
    run_sh = RUN_SH.read_text(encoding="utf-8")
    # Comment lines are stripped first: the fix's own comment quotes the glob it
    # replaced, and a check that cannot tell a comment from code would fail on
    # the explanation of the thing it is checking.
    code = "\n".join(l for l in run_sh.splitlines() if not l.lstrip().startswith("#"))
    invocation = code[code.index("check_docs.py"):]
    assert '"$repo_root"/docs/*.md' not in invocation, (
        "run.sh feeds check_docs.py with a non-recursive shell glob again. A glob "
        "does not descend into docs/<folder>/, so every document there would drop "
        "out of DOCWATCH with no error and a smaller marker count."
    )
    assert 'find "$repo_root/docs" -name' in code, (
        "run.sh no longer enumerates docs/ recursively before calling check_docs.py"
    )
    assert "$doc_files" in invocation, (
        "run.sh computes a recursive doc list but does not pass it to check_docs.py"
    )

    check = CHECK_DOCS.read_text(encoding="utf-8")
    assert 'DOCS_DIR.glob("*.md")' not in check, (
        "check_docs.py's default scan is a non-recursive glob again. This is the "
        "more dangerous of the two: it under-reports for anyone invoking the "
        "checker directly, not just through the gate."
    )
    assert 'DOCS_DIR.rglob("*.md")' in check, (
        "check_docs.py's default scan should be rglob so it finds documents in "
        "docs/<folder>/"
    )


def test_the_documents_are_all_still_here():
    """A move is not a deletion. The count is the one that was measured before
    sorting began; `ge` rather than `eq` because later rounds legitimately add
    documents, and a *drop* is the news."""
    n = len(list(DOCS.rglob("*.md"))) - 1  # minus docs/README.md, added by the sort
    assert n >= 183, (
        f"{n} documents under docs/, but 183 were there before they were sorted "
        f"into subfolders. A move must not lose one."
    )


# A path built from separate string literals -- `os.path.join(REPO, "docs",
# "CUDA.md")` or `root / "docs" / "SETITEM.md"` -- is invisible to any check
# that greps for `docs/NAME.md`, because that string never appears in the source.
# Four of these survived the textual rewrite and the gate caught them by
# FileNotFoundError, one suite at a time. This is the cheaper way to find them.
SPLIT_PATH = re.compile(r'"docs"\s*[,/]\s*"(?P<name>[A-Za-z0-9_.-]+\.md)"')


def test_no_path_is_built_from_split_docs_literals():
    """`"docs", "NAME.md"` does not resolve now that documents live in folders,
    and no textual rewrite of `docs/NAME.md` can see it."""
    bad = []
    for path, text in _text_files():
        if path == SELF:
            continue
        if path.suffix not in {".py", ".rs", ".sh", ".c"}:
            continue  # a path is built in code; prose quoting one is not a path
        for m in SPLIT_PATH.finditer(text):
            bad.append(f"{path.relative_to(REPO)}: \"docs\", \"{m.group('name')}\"")
    assert not bad, (
        "documentation paths built from split string literals, which point at "
        "the old flat layout and which a `docs/NAME.md` rewrite cannot see:\n  "
        + "\n  ".join(bad)
        + "\nWrite the folder in: os.path.join(REPO, \"docs\", \"<folder>\", \"NAME.md\")."
    )


# --- Scope and cost of the scan (GitHub issue #8) --------------------------


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True,
                   stdout=subprocess.DEVNULL)


def _scratch_repo(root):
    """A throwaway git checkout with one document and three files that each name
    a document that does not exist: one tracked, one untracked, one ignored."""
    repo = pathlib.Path(root) / "repo"
    (repo / "docs" / "kernels").mkdir(parents=True)
    (repo / "docs" / "kernels" / "REAL.md").write_text("# real\n")
    (repo / "tracked.md").write_text("see docs/kernels/TRACKED_GONE.md\n")
    (repo / "untracked.md").write_text("see docs/kernels/UNTRACKED_GONE.md\n")
    (repo / "scratch").mkdir()
    (repo / "scratch" / "note.md").write_text("see docs/kernels/IGNORED_GONE.md\n")
    (repo / ".gitignore").write_text("scratch/\n")
    _git(repo, "init", "-q")
    _git(repo, "add", "docs", "tracked.md", ".gitignore")
    return repo


def test_only_tracked_files_are_scanned_and_a_tracked_stale_reference_still_fails():
    """The docstring always said "every tracked text file"; the walk was
    `rglob("*")`, which was the same set only until `.worktrees/`, `.caches/`
    and `.scratch/` moved inside the checkout (2026-09-22 / 2026-10-02). A
    scratch note quoting an old path then turned the gate red. Both halves are
    asserted: the untracked and ignored references are not reported, and the
    tracked one is -- a scan that reads nothing would pass the first half."""
    root = tempfile.mkdtemp(prefix="docrefs-scope-")
    try:
        dangling = _dangling_references(_scratch_repo(root))
    finally:
        shutil.rmtree(root)
    assert set(dangling) == {"docs/kernels/TRACKED_GONE.md"}, (
        "expected exactly the tracked file's stale reference to be reported, got "
        f"{sorted(dangling)}"
    )


def test_every_tracked_file_that_names_docs_is_read():
    """The other direction, against the real tree: restricting the scan must not
    quietly drop tracked files (AGENTS.md §17.5). Computed independently of
    `_text_files` -- straight from `git ls-files` and the bytes on disk -- so a
    skip list that grows to cover a tracked directory turns this red."""
    out = subprocess.run(["git", "-C", str(REPO), "ls-files", "-z"], check=True,
                         capture_output=True).stdout
    scanned = {p.relative_to(REPO).as_posix() for p, _ in _text_files()}
    tracked = {os.fsdecode(n) for n in out.split(b"\0") if n}
    untracked_read = sorted(scanned - tracked)
    assert not untracked_read, (
        f"{len(untracked_read)} file(s) read that git does not track, e.g. "
        f"{untracked_read[:5]}"
    )
    missed = []
    for rel in sorted(tracked):
        p = REPO / rel
        if not p.is_file():
            continue
        data = p.read_bytes()
        if b"docs/" not in data:
            continue
        try:
            data.decode("utf-8")
        except UnicodeDecodeError:
            continue
        if rel not in scanned:
            missed.append(rel)
    assert not missed, (
        f"{len(missed)} tracked text file(s) mention `docs/` but are not scanned: "
        f"{missed[:10]}"
    )


# One megabyte of a single character is what hung the gate: a leaked
# `collect2-harness-probe-*` rank log. The other inputs aim at the same
# backtracking from different sides -- each character of the old prefix class,
# the same run with a `docs/` at its end (so a line filter that skips lines
# without `docs/` cannot hide a quadratic pattern), and many matches on one
# line (per-match line slicing is quadratic there even with a linear pattern).
_MB = 1 << 20
_PATHOLOGICAL = {
    **{f"run of {c!r}": c * _MB for c in "a0_.:-/"},
    **{f"run of {c!r} then docs/": c * _MB + "docs/" for c in "a0_.:-"},
    "docs/ repeated": "docs/" * (_MB // 5),
    "docs/<folder>/ repeated": "docs/kernels/" * (_MB // 13),
    "one long line of matches": "docs/kernels/REAL.md " * (_MB // 21),
    "split literal then spaces": '"docs"' + " " * _MB,
}

_TIMING_PROBE = r"""
import importlib.util, pathlib, sys, time
spec = importlib.util.spec_from_file_location("docrefs_probe", sys.argv[1])
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
for label, text in mod._PATHOLOGICAL.items():
    t0 = time.monotonic()
    # The pattern on its own as well as through `_references_in`: the line
    # filter there skips most of these inputs, and must not be what keeps a
    # quadratic pattern from showing.
    n = sum(1 for _ in mod.DOCS_REF.finditer(text))
    n += sum(1 for _ in mod._references_in(text))
    n += sum(1 for _ in mod.SPLIT_PATH.finditer(text))
    print(f"{time.monotonic() - t0:8.3f}s  {n:7d} refs  {label}", flush=True)
t0 = time.monotonic()
d = mod._dangling_references(pathlib.Path(sys.argv[2]))
print(f"{time.monotonic() - t0:8.3f}s  tracked 1 MB file through the full scan", flush=True)
assert set(d) == {"docs/kernels/TRACKED_GONE.md"}, sorted(d)
"""

#: The whole probe -- seventeen 1 MB inputs plus a full scan of a tracked 1 MB
#: file -- took between 0.3 and 1.2 s in total in every run of the round that
#: wrote it (2026-10-03), several at load averages far above the 8 cores. The
#: quadratic prefix ran for more than 30 minutes on one input without
#: finishing. The bound sits far from both.
_TIMING_BOUND_S = 60


def test_matching_is_linear_on_one_megabyte_of_a_single_character():
    """Run in a subprocess with a timeout so that a quadratic regression fails
    this test instead of hanging the gate, which is what the original did."""
    root = tempfile.mkdtemp(prefix="docrefs-linear-")
    try:
        repo = _scratch_repo(root)
        (repo / "probe.log").write_text("a" * _MB)
        _git(repo, "add", "probe.log")
        # Progress goes to a file, not a pipe: a pipe's partial output is lost
        # when the timeout kills the child, and which input stalled is the
        # useful half of the failure.
        progress = pathlib.Path(root) / "progress.txt"
        with open(progress, "w") as out:
            try:
                proc = subprocess.run(
                    [sys.executable, "-c", _TIMING_PROBE, str(SELF), str(repo)],
                    stdout=out, stderr=subprocess.STDOUT, timeout=_TIMING_BOUND_S,
                )
            except subprocess.TimeoutExpired:
                proc = None
        report = progress.read_text()
    finally:
        shutil.rmtree(root)
    print(report, end="")
    assert proc is not None, (
        f"scanning 1 MB inputs did not finish in {_TIMING_BOUND_S}s; matching is "
        f"not linear. Finished before the timeout:\n{report}"
    )
    assert proc.returncode == 0, report


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
