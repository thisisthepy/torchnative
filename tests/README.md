# `tests/` — VOICE4 (BigVGAN) fixtures

The scripts here are tracked. **The assets are not**, and that is deliberate:
`hf/` (430M), `bigvgan-converted/` (429M), `vendor/` (90M),
`voicestudio-5.16.0/`, `vs-src/`, `pkg/`, `stubs/`, `ref/` and `mlk24k.wav`
are model weights, a HuggingFace cache, vendored third-party source, and
generated reference output. All are reproducible from the scripts, and none
is this project's to redistribute.

## What the gate does with them

`rust/torch_c/pytests/test_voice4.py` reads `TORCH_C_VOICE4_ASSETS`. **Unset,
two BigVGAN tests skip by name** and the gate reports them under `SKIP`. Point
it at this directory to run them:

    TORCH_C_VOICE4_ASSETS=$(pwd)/tests bash rust/torch_c/pytests/run.sh

Until 2026-09-22 this directory lived outside the repository entirely, at
`/Volumes/macMini/voice4-work`, where nothing referenced it and nobody could
see that it was the thing keeping those two tests from running.

## The scripts

| file | what it does |
|---|---|
| `capture_ops.py` | records the operators a BigVGAN forward reaches |
| `make_ref.py`    | regenerates `ref/` — the upstream oracle the test compares against |
| `replay.py`      | replays a capture under the shim and prints `MARKER shim`/`upstream` |
| `nullify.sh`     | rebuilds, restages and runs `test_voice4.py`, printing ok/FAIL only |

`nullify.sh` took a hard-coded worktree path until 2026-09-22; it now derives
the repository root from its own location, so it works from any checkout.
