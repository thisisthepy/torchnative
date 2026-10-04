"""Writing rules the gate enforces (AGENTS.md §23, issue #45).

1. No em-dash (U+2014) in any tracked file outside `vendor/`. The maintainer
   directed this on 2026-10-03: an em-dash joins two sentences that should be
   split, or hides the relation between a clause and its reason. `vendor/` is
   upstream's text and stays untouched.
2. Reader-facing install instructions (README, the Korean README, the guide) use
   uv or ppp, never `pip install`. Test and CI code that runs pip as a tool is
   not an instruction to a reader and is not scanned.

The character is written as an escape below, so this file does not contain the
thing it forbids. The detector is run on planted text to prove it can go red
(AGENTS.md §17.5); the same nullification was also done on a real tracked file.
"""

import pathlib
import subprocess

REPO = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
EM_DASH = "\u2014".encode("utf-8")
READER_FACING = ["README.md", "docs/locale/README_ko.md"]


def _git(*args):
    done = subprocess.run(["git", *args], cwd=REPO, capture_output=True)
    assert done.returncode == 0, f"git {' '.join(args)} failed:\n{done.stderr.decode()}"
    return done.stdout


def _tracked():
    names = [n.decode("utf-8") for n in _git("ls-files", "-z").split(b"\0") if n]
    return [n for n in names if not n.startswith("torchnative/rust/vendor/") and (REPO / n).is_file()]


def _em_dash_files(read, names):
    return sorted(n for n in names if EM_DASH in read(n))


def _read(name):
    return (REPO / name).read_bytes()


def test_no_em_dash_in_tracked_files_outside_vendor():
    bad = _em_dash_files(_read, _tracked())
    assert not bad, (
        f"em-dash (U+2014) in {len(bad)} tracked files, first {bad[:5]} -- AGENTS.md §23: "
        f"split the sentence, or use a colon, a comma or parentheses")


def test_the_em_dash_detector_can_fail():
    planted = {"a.md": b"one " + EM_DASH + b" two", "b.md": b"clean", "c.py": b"x" + EM_DASH}
    assert _em_dash_files(planted.__getitem__, sorted(planted)) == ["a.md", "c.py"]
    assert _em_dash_files(planted.__getitem__, ["b.md"]) == []


def test_the_scan_covers_the_files_it_claims_to():
    names = _tracked()
    assert "README.md" in names and "AGENTS.md" in names
    assert any(n.startswith("docs/") for n in names)
    assert any(n.endswith(".rs") for n in names)
    assert not any(n.startswith("torchnative/rust/vendor/") for n in names)


def _reader_files():
    return READER_FACING + sorted(
        str(p.relative_to(REPO)) for p in (REPO / "docs" / "guide").glob("*.html"))


def test_reader_facing_pages_do_not_tell_the_reader_to_pip_install():
    bad = [n for n in _reader_files() if b"pip install" in _read(n)]
    assert not bad, f"`pip install` in {bad} -- AGENTS.md §23: examples use uv or ppp"


def test_the_pip_check_can_fail():
    assert b"pip install" in b"```sh\npip install --pre torchnative\n```"
    assert b"pip install" not in b"uv add --prerelease allow torchnative"


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
