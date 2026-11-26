# macOS example

Runs the common task (`../README.md`) on one Mac with three devices side by side:

| Device | What it is | How the script reaches it |
|---|---|---|
| `cpu` | torchnative's CPU kernels | the model as loaded |
| `mps` | Metal, through candle's Metal backend | `model.to("mps")`, inputs moved to `mps` |
| `npu` | the Apple Neural Engine, through CoreML | `model.to(torchnative.device.npu)` |

If an op refuses on `mps`, the script prints the refusal and the list from
`torch._C._shim_mps_host_readback_ops()`: the ops this build answers on `mps` by reading back to
the host. A device that cannot run is reported as `FAILED` with its error, and the script exits
non-zero. It never runs the CPU in its place.

Hardware we test on: Mac mini M1.

## Run

```sh
uv run python run.py                 # cpu, mps, npu
uv run python run.py --device npu    # one device
```

`uv` installs torchnative (which provides `torch`) and `transformers` 5.15.1 from PyPI. The model is
downloaded into the Hugging Face cache on the first run.

To run against a torchnative checkout instead of the published wheel, build the shim
(`scripts/vendor/vendor_torch.sh`, then `scripts/vendor/install_shim.sh`) and run the script with that
checkout's `torchnative/python` first on `PYTHONPATH`.

## Expected output

For each device, the streamed text, then one line:

```
COMMON_TASK {"device": "cpu", "text": " a very powerful technique for learning from data. ...", "new_tokens": 32, "tokens_per_second": ..., "agrees_with_upstream": true, "ids": [...]}
```

`agrees_with_upstream` compares the token ids with `reference.json`, which `reference.py` wrote with
upstream torch 2.13.0 and transformers 5.15.1 (not with torchnative). The Android example uses the
same ids. Equal ids are the criterion on every device, `npu` included: `npu` computes in float16, and
there is no tolerance in its place (leader decision, 2026-11-26, not the owner's words).

## Last recorded result

Mac mini M1, 2026-11-26, branch `feat/437-examples-macos` with the shim built from the checkout:

| Device | Ids equal to upstream | Tokens per second |
|---|---|---|
| `cpu` | 32 of 32 | 33.76 |
| `mps` | 32 of 32 | 48.06 |
| `npu` | 32 of 32, in 3 of 3 runs | 3.2, 17.43, 7.87 |

The speeds were taken with other builds running (load average 5 to 6), so they are not a
measurement. On `npu` the first run includes compiling the CoreML programs.

`npu` computes in float16, and this task's greedy margins are as small as 0.026 logits. Before
2026-11-26 the run diverged from upstream at the second step. The cause was not float16: `generate`
on a compiled module uses a `StaticCache`, and the CoreML decoder groups fed its unwritten slots
to the program as keys and values. The logit error grew with the cache length (0.78 at 5 slots,
8.41 at 64) against 0.20 with a `DynamicCache`. The groups now read only the rows written
(`tests/devices/coreml/test_anecacherows.py`, `test_anegenerate.py`).
