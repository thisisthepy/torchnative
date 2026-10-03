# `main` is published by release-sync, and only by release-sync

`develop` is the repository. `main` is what a person who has never read this
repository sees when they land on GitHub. Nobody develops on it and nothing is
merged into it by hand. There is **one** path to it:

```
push to develop
  -> .github/workflows/release-sync.yml
  -> .github/scripts/release/sync-release.sh --push   (regenerates `release`)
  -> pull request release -> main                      (opened or updated by the workflow)
  -> the maintainer merges it
```

Protecting `main` belongs to the maintainer, and the maintainer merges the
release pull request. Agents never push to `main` or `release` (AGENTS.md §4).

The script is `.github/scripts/release/sync-release.sh`; its own README is
`.github/scripts/release/README.md`.

<!-- DOCWATCH: symbol-in-file .github/scripts/release/sync-release.sh drop.z present -->

---

## 1. Why `release` is built, not merged

The obvious implementation is to merge `develop` into `main` and delete the
paths that should not ship. It fails quietly: a deletion made in a merge commit
does not survive the next merge once `develop` has touched the path again, so
every release would have to repeat the deletions by hand.

So `release`'s tree is **derived from** `develop`'s on every push, through a
private index, and committed as exactly one commit on top of that `develop`
commit (with `main` as a second parent when `main` holds commits `develop`
lacks, so the pull request always fast-forwards):

```sh
GIT_INDEX_FILE=<private> git read-tree <develop sha>
GIT_INDEX_FILE=<private> git update-index --force-remove ...   # the dropped paths
TREE=$(GIT_INDEX_FILE=<private> git write-tree)
git commit-tree "$TREE" -p <develop sha> [-p <main sha>]
```

The caller's working tree, index and `HEAD` are never touched.

<!-- DOCWATCH: symbol-in-file .github/scripts/release/sync-release.sh GIT_INDEX_FILE present -->
<!-- DOCWATCH: symbol-in-file .github/scripts/release/sync-release.sh commit-tree present -->

---

## 2. The main-only layout

Only Markdown is ever dropped:

- every `*.md` at the repository root except `README.md` (so `AGENTS.md` and
  `PROJECT.md` stay on `develop`);
- every `*.md` directly inside `docs/`.

Everything else is kept, including every subdirectory of `docs/`. That matters
for one directory in particular: **`docs/guide/` is the GitHub Pages site**, and
`.github/workflows/pages.yml` deploys it from `main`. A layout that dropped it
would empty the published site while every check on `develop` stayed green.
`tests/release/test_publish.py` therefore runs the real `sync-release.sh` over the real
tree and asserts that `docs/guide/index.html` and every other file under
`docs/guide/` reach `release`, and it runs a deliberately broken copy of the
script to prove that assertion can fail (AGENTS.md §17.5).

`.github/` is kept as well. The manually triggered workflows
(`build-cuda-wheel.yml`, `publish-pypi.yml`, `test-qnn-lower.yml`,
`test-published-wheel.yml`) are `workflow_dispatch`, and GitHub only renders the
"Run workflow" button for a workflow file present on the default branch, which
`main` is.

<!-- DOCWATCH: symbol-in-file .github/workflows/pages.yml docs/guide present -->

---

## 3. What used to be here

Before issue #43 a second, manual path existed: a local script that built
`main` from `develop` minus an exclusion list and had the coordinating session
push it. Its exclusion list removed `docs/` entirely, so a `main` published
through it would have emptied the Pages site, and no workflow called it. It was
deleted in #43 together with the tests that covered only it. The release notes
for 0.1.0b1 and 0.1.0b2 mention it; they describe what happened then and are
left as history.

---

## 4. The checks

| Check | What it can see |
|---|---|
| `.github/scripts/release/test-sync-release.sh` | the regeneration mechanics in a throwaway repository under `.scratch/`: what is dropped and kept, one commit over the source, no-op runs, the diverged-`main` second parent, `--push` with `--force-with-lease`, refusal while `release` is checked out |
| `tests/release/test_publish.py` | the real tree: `docs/guide/` survives the main-only layout, and the workflow set is the named one |

Neither check builds a wheel from the `release` tree; the wheel is built and
published from a `v*` tag by `publish-pypi.yml` (`docs/platform/PUBLISH_CI.md`).
PyPI uploads still need the user's explicit approval (AGENTS.md §17.7).
