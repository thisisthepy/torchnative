# `docs/` — what is here, and where a new document goes

183 documents accumulated flat in this directory. The cost was not untidiness:
a flat list gives a new round no place to *look*, so each round read the two or
three documents it already knew about and dropped its own at the top level. This
index exists so the next round has somewhere to put its document, and somewhere
to look before it re-measures something already measured.

**Put your document in the folder that matches its subject, not its round name.**
Several families here (`BACKWARD1-9`, `DEMAND1-8`, `TAIL*`) are named after the
round rather than the subject, which is why the folder assignments below were made
by reading the documents rather than by their filename prefix.

**`AGENTS.md` and `README.md` stay at the repository root.** They are not round
records; they are the front door.

| Folder | What belongs there |
|---|---|
| `design/` | What this project is and the shape of its core — the `torch._C` surface, the dispatcher's design, abi3, and the decisions that constrain everything else. |
| `bindings/` | The Python-facing surface: names, spellings, argument forms, overload resolution, `methods.json`/`overloads.json`, and whether a name reaches a kernel at all. |
| `kernels/` | Individual operator kernels and the rounds that added them — what a kernel does, what it cost, and where it still diverges. |
| `architectures/` | Sweeps over real model architectures and the demand lists derived from them: which models forward, which are blocked, and on what. |
| `numerics/` | Dtypes, promotion, scalar rules, RNG, and agreement with upstream's actual numbers. The "does it produce upstream's answer" axis. |
| `models/` | Running real checkpoints end to end: `from_pretrained`, `generate`, saving, loading, `transformers` compatibility. |
| `training/` | Autograd, the tape, backward passes, loss, and training mode. |
| `graph/` | Graph capture, decomposition, refolding, `torch.export`, `torch.compile`/Dynamo walls, and lowering to NPU/quantized backends. |
| `devices/` | Device backends and the device abstraction: CPU, `meta`, MPS, CUDA, Vulkan, and vendor NPUs. |
| `distributed/` | `torch.distributed`, collectives, transport, and federated rounds. |
| `platform/` | Building and shipping: cross-builds, wheels, WASM, iOS/Android/Linux/Windows targets, vendoring, release notes. |
| `perf/` | Measurements of speed and overhead, and the fixes made for speed. Numbers here are only valid unloaded — see AGENTS.md §16. |
| `api/` | The user-facing Python API this project offers in its own namespaces: `torchnative.device`, `torchnative.transformers`, and the argument forms and refusals they present. Distinct from `bindings/`, which is the `torch._C` surface we must reproduce; this folder is the surface we chose. |
| `verification/` | The checkers themselves and the audits of this documentation: DOCWATCH, the golden harness, and what they structurally cannot see. |
| `guide/` | The bilingual (en/ko) GitHub Pages site, deployed from `main` by `.github/workflows/pages.yml`. Static HTML/CSS/JS; `check_guide.py` is its test. |
| `locale/` | Translations of the front page: `README_ko.md` is the Korean README. |

This directory holds **documents only**. The three non-Markdown files that
used to sit here have moved to where the thing they are is kept:
`_profile_decode.py` and `_profile_sdpa_shapes.py` to `tests/bench/` (they are
run, not read), and `int8-candle-0.11.0-cpu.patch` to `torchnative/rust/vendor/`.

## The check that keeps this true

`tests/release/test_docrefs.py` runs in the gate and enforces:

1. **No `docs/*.md` reference in a tracked file dangles.** 7,081 references in
   438 tracked files (counted 2026-10-03) name documents by path, in Markdown
   prose, Rust comments, Python docstrings and workflow files. A move that
   misses one is invisible otherwise. The scan is `git ls-files`, so scratch
   notes, logs and worktrees inside the checkout are not read, and a test
   bounds the scan's cost on a 1 MB single-character file -- an earlier
   pattern was quadratic and hung the gate on a leaked log (GitHub issue #8).
2. **No new document lands at the top level of `docs/`.** That is the failure
   this index exists to stop, and prose asking for it would not have stopped it.
3. **No documentation path is built from split string literals.**
   `os.path.join(REPO, "docs", "CUDA.md")` never contains the string
   `docs/CUDA.md`, so no textual rewrite can see it. Four of these survived the
   sort and the gate found them one suite at a time by `FileNotFoundError`.
4. **DOCWATCH still enumerates documents recursively.** `run.sh` fed the
   documentation checker with `docs/*.md`, and a shell glob does not recurse — so
   the moment a document moved into a subfolder it would have silently dropped
   out of the check, leaving the gate green and the marker count *smaller*. The
   test asserts the non-recursive glob has not come back, in `run.sh` *and* in
   `check_docs.py`'s own default, which was the same glob and the more dangerous
   of the two because it under-reports for anyone invoking the checker directly.
5. **Every document is still here** (183 at the time of the sort, `ge`), and the
   folders on disk and the table above name each other exactly.

Each of these was verified by deliberately breaking it and watching the test go
red. Two of them were *vacuous when first written* -- their regexes could not
match `docs/<folder>/NAME.md` at all, so they matched nothing after the sort and
passed green. That is AGENTS.md §17.5 reproduced inside the test written to
prevent it; the comments in the test record it so the next round does not
reintroduce it.
