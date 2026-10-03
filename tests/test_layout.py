"""The repository root holds only the approved entries (AGENTS.md §2, issue #43).

"Do not add top-level folders" is a rule an agent can break without noticing,
and a stray root entry is invisible in a diff of a few hundred files. So the
gate compares the root against the approved list:

  * every top-level name `git ls-files` reports (tracked, or staged) must be on
    the tracked list;
  * of the git-ignored entries, `.caches/`, `.scratch/` and `.worktrees/` are
    approved when present, and must actually be ignored and untracked.

The list is written down here AND in AGENTS.md's table, and the two are
compared, so neither can be widened alone. The check itself is run once on a
root with a stray entry, to prove it can go red (AGENTS.md §17.5).
"""

import pathlib
import re
import subprocess

REPO = pathlib.Path(__file__).resolve().parents[1]

APPROVED_TRACKED = {
    "pyproject.toml", "setup.py", "README.md", "PROJECT.md",
    "AGENTS.md", "LICENSE", ".gitignore", ".github", "torchnative",
    "tests", "docs", "scripts", "vendor",
}
APPROVED_IGNORED = {".caches", ".scratch", ".worktrees"}


def _git(*args):
    done = subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True)
    assert done.returncode == 0, f"git {' '.join(args)} failed:\n{done.stderr}"
    return done.stdout


def _tracked_top_level():
    return {p.split("/", 1)[0] for p in _git("ls-files", "-z").split("\0") if p}


def _violations(tracked_top):
    return sorted(tracked_top - APPROVED_TRACKED)


def test_every_tracked_root_entry_is_approved():
    stray = _violations(_tracked_top_level())
    assert not stray, (
        f"unapproved root entries {stray} -- AGENTS.md §2: work goes inside the "
        f"existing directories; propose a new top-level entry and wait for approval")


def test_the_ignored_root_entries_present_are_ignored_and_untracked():
    tracked = _tracked_top_level()
    for name in sorted(APPROVED_IGNORED):
        if not (REPO / name).exists():
            continue
        assert name not in tracked, f"{name}/ is approved only as a git-ignored entry"
        done = subprocess.run(["git", "check-ignore", "-q", name + "/"], cwd=REPO)
        assert done.returncode == 0, f"{name}/ is present but not git-ignored"


def test_the_check_can_fail():
    """Nullification: a root with a stray directory must be reported, by name."""
    assert _violations({"README.md", "docs", "tools"}) == ["tools"]
    assert _violations(APPROVED_TRACKED) == []


def test_the_list_matches_agents_md():
    """AGENTS.md §2's table is the list a person reads; this file is the list
    the gate enforces. They must be the same list."""
    text = (REPO / "AGENTS.md").read_text(encoding="utf-8")
    rows = {}
    for kind in ("Tracked", "Git-ignored"):
        m = re.search(r"^\| " + kind + r" \| (.*) \|$", text, re.M)
        assert m, f"AGENTS.md §2 has no '{kind}' row in the approved root entries table"
        rows[kind] = {n.rstrip("/") for n in re.findall(r"`([^`]+)`", m.group(1))}
    assert rows["Tracked"] == APPROVED_TRACKED, (
        f"AGENTS.md lists {sorted(rows['Tracked'])}, this test {sorted(APPROVED_TRACKED)}")
    assert rows["Git-ignored"] == APPROVED_IGNORED, (
        f"AGENTS.md lists {sorted(rows['Git-ignored'])}, this test {sorted(APPROVED_IGNORED)}")


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
