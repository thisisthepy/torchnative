# Release flow

    develop --push--> release (auto) --PR--> main

`release` is CI-generated and disposable, not a long-lived branch. It may not exist
between runs, and any existing `release` (e.g. a stale one from an older workflow) is
ignored and overwritten.

1. Every push to `develop` triggers `.github/workflows/release-sync.yml`, which runs
   `.github/scripts/release/sync-release.sh --push`. It rebuilds `release` from scratch as exactly
   one commit (`Release: sync from develop <sha>`) whose parent is that develop commit and
   whose tree is the develop tree minus the files below. This happens even when nothing
   is dropped. It is a no-op only if `release` already has that tree and that parent.
   The push uses `--force-with-lease` pinned to the remote sha the run observed.
2. The workflow opens (or updates) a pull request `release` -> `main` with the default
   `GITHUB_TOKEN`. This needs the repository setting *Actions -> General -> Allow GitHub
   Actions to create and approve pull requests*. No PAT is needed; an optional
   `RELEASE_PR_TOKEN` secret is used for the PR step when present.
3. Protecting `main` belongs to the maintainer, and the maintainer merges the release
   pull request. That merge is the only way `main` changes.
4. On push to `main`, `pages.yml` deploys `docs/guide/` to GitHub Pages. `docs/guide/`
   survives the main-only layout (it is not Markdown directly under `docs/`);
   `tests/release/test_publish.py` asserts that against the real tree.

## What is dropped on release

- every `*.md` at the repository root except `README.md` (including `CLAUDE.md`, `AGENTS.md`)
- every `*.md` directly inside `docs/`

Kept: everything else, including Markdown in `docs/<subfolder>/` and nested `README.md` files.
Only Markdown is ever dropped.

## Local use

    .github/scripts/release/sync-release.sh --dry-run     # list paths that would be dropped
    .github/scripts/release/sync-release.sh               # regenerate local release branch
    .github/scripts/release/sync-release.sh --push        # also push to origin
    .github/scripts/release/test-sync-release.sh          # tests (throwaway repo in .scratch/)

The script uses plumbing and a private index, so your working tree, index and HEAD are untouched.
It refuses to run while `release` is checked out.
