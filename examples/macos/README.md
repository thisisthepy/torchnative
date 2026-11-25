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
same ids.

## Last recorded result

Not run yet as this example. Before it existed, the same task with the shim on the Mac mini M1 (cpu)
gave 32 of 32 ids equal to upstream's at 36.25 tokens per second (2026-11-25, from
`examples/android`'s task code, branch `feat/437-examples-android`).
