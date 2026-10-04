"""Does release-sync's main-only layout keep what `main` must carry -- and can
that check actually fail?

`main` is reached by one path only: `.github/workflows/release-sync.yml` runs
`.github/scripts/release/sync-release.sh`, which regenerates `release` from
`develop` minus some Markdown, and the maintainer merges the `release -> main`
pull request (docs/platform/PUBLISH.md). `pages.yml` then deploys `docs/guide/`
from `main`. So a layout change that dropped `docs/guide/` would empty the
published site while every check on `develop` stayed green. The older manual
publisher removed `docs/` entirely; it was deleted in #43.

`.github/scripts/release/test-sync-release.sh` tests the regeneration
mechanics in a synthetic repository. This file runs the real script over the
real tree, and runs a deliberately broken copy of it to prove the guide check
goes red when the layout loses the guide (AGENTS.md 17.5).
"""

import os
import pathlib
import shutil
import subprocess
import tempfile

REPO = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
SYNC = REPO / ".github/scripts/release/sync-release.sh"
GUIDE = "docs/guide"


#: Every workflow, named. Named rather than counted, for the reason build.py's
#: EXPECTED_TARGET_KEYS is named: a count cannot say which one moved.
MANUAL_WORKFLOWS = (
    "build-cuda-wheel.yml",
    "publish-pypi.yml",
    "test-published-wheel.yml",
    "test-qnn-lower.yml",
)
# The release flow shared by every thisisthepy repository
# (.github/scripts/release/README.md): event-driven, not manual, so they are
# exempt from the workflow_dispatch premise.
AUTOMATION_WORKFLOWS = (
    "pages.yml",
    "release-sync.yml",
)
EXPECTED_WORKFLOWS = tuple(sorted(MANUAL_WORKFLOWS + AUTOMATION_WORKFLOWS))


def test_every_workflow_is_named_and_the_manual_ones_are_workflow_dispatch():
    """GitHub only renders the "Run workflow" button for a workflow file present
    on the default branch, which `main` is; the main-only layout keeps
    `.github/` for that reason. If a manual workflow stops being
    `workflow_dispatch`, that reasoning should be re-argued, not inherited.

    `publish-pypi.yml` is also triggered by a version tag, which is not a
    counter-example: it is *additionally* `workflow_dispatch`, deliberately, so
    that the first use of a release workflow is a dry run rather than a
    release. docs/platform/PUBLISH_CI.md §6."""
    wfs = sorted((REPO / ".github/workflows").glob("*.yml"))
    names = tuple(w.name for w in wfs)
    assert names == EXPECTED_WORKFLOWS, (
        f"the workflow set is {list(names)}; expected {list(EXPECTED_WORKFLOWS)}")
    for wf in wfs:
        if wf.name not in MANUAL_WORKFLOWS:
            continue
        assert "workflow_dispatch" in wf.read_text(), (
            f"{wf.name} is no longer workflow_dispatch -- the main-only layout "
            f"keeps .github/ on main specifically so its Run workflow button exists."
        )


def test_release_sync_carries_no_guard_status():
    """Main is protected by the maintainer's repository settings, not by a
    status check this workflow fakes (#43). The guard workflow is gone, so a
    step that posts its verdict, and the permission it needed, must be too."""
    text = (REPO / ".github/workflows/release-sync.yml").read_text()
    assert "statuses: write" not in text, "release-sync.yml still asks for statuses: write"
    assert "only-release-into-main" not in text, (
        "release-sync.yml still posts the deleted main-source-guard's status")


def _git(cwd, *args):
    return subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True)


def _release_files(script):
    """Run `script` (a sync-release.sh) over a throwaway clone of this
    repository's HEAD and return (files in HEAD, files in `release`).

    A clone, not this checkout: the script writes `refs/heads/release`, and the
    gate must not move a ref in the repository it is testing. The clone lives
    under `.scratch/` (AGENTS.md §2), not in the system temp directory."""
    scratch = REPO / ".scratch"
    scratch.mkdir(exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="test_publish.", dir=scratch)
    try:
        clone = os.path.join(tmp, "repo")
        done = _git(REPO, "clone", "-q", "--shared", "--no-checkout", str(REPO), clone)
        assert done.returncode == 0, f"git clone failed:\n{done.stderr}"
        head = _git(REPO, "rev-parse", "HEAD").stdout.strip()
        done = subprocess.run(
            ["bash", str(script), "--source", head, "--target", "release"],
            cwd=clone, capture_output=True, text=True)
        assert done.returncode == 0, (
            f"sync-release.sh failed:\n{done.stdout}\n{done.stderr}")
        src = _git(clone, "ls-tree", "-r", "--name-only", head).stdout.split("\n")
        rel = _git(clone, "ls-tree", "-r", "--name-only", "release").stdout.split("\n")
        return {p for p in src if p}, {p for p in rel if p}
    finally:
        shutil.rmtree(tmp)


def _guide_losses(src, rel):
    guide = {p for p in src if p.startswith(GUIDE + "/")}
    assert f"{GUIDE}/index.html" in guide, (
        f"HEAD has no {GUIDE}/index.html -- the Pages site has nothing to deploy")
    return sorted(guide - rel)


def test_docs_guide_survives_the_main_only_layout():
    """`pages.yml` deploys `docs/guide/` from main. Every file of it, the
    landing page first, must reach `release`."""
    src, rel = _release_files(SYNC)
    assert f"{GUIDE}/index.html" in rel, (
        f"{GUIDE}/index.html does not survive the main-only layout -- merging "
        f"the release PR would empty the Pages site")
    lost = _guide_losses(src, rel)
    assert not lost, f"the main-only layout drops part of the Pages site: {lost}"
    # The layout is not a no-op: it does drop the develop-only Markdown.
    for gone in ("AGENTS.md", "PROJECT.md"):
        assert gone in src and gone not in rel, f"{gone} reaches release"


def test_the_guide_check_can_fail():
    """Nullification: a copy of sync-release.sh that also drops everything under
    `docs/` must make the guide check above go red. If it stays green, the
    check is reading something other than the generated tree."""
    text = SYNC.read_text()
    keep = "        docs/*/*) ;;           # inside a docs subdirectory: keep\n"
    assert keep in text, "sync-release.sh no longer has the docs-subdirectory keep rule"
    broken = text.replace(keep, "        docs/*/*) printf '%s\\0' \"$p\" ;;\n")
    scratch = REPO / ".scratch"
    scratch.mkdir(exist_ok=True)
    fd, path = tempfile.mkstemp(prefix="sync-release.broken.", suffix=".sh", dir=scratch)
    try:
        with os.fdopen(fd, "w") as fh:
            fh.write(broken)
        src, rel = _release_files(path)
    finally:
        os.unlink(path)
    assert f"{GUIDE}/index.html" not in rel, "the broken layout kept the guide"
    assert _guide_losses(src, rel), "the guide check cannot see a dropped guide"


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
