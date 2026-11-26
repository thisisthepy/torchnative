# Linux example

Runs the common task (`../README.md`) on one Linux machine, on every device it has:

| Device | What it is | How the script reaches it |
|---|---|---|
| `cpu` | torchnative's CPU kernels | the model as loaded |
| `vulkan` | torchnative's own Vulkan kernels | `model.to(torchnative.device.vulkan)`, inputs moved to `vulkan` |
| `cuda` | candle's CUDA backend | `model.to(torchnative.device.cuda)`, inputs moved to `cuda` |

The script first prints one `AVAILABILITY` line per device, with the reason its probe gave when the
device is not there (for example `no_loader` for Vulkan, `not_built` or `no_driver` for CUDA). With
no `--device`, it runs `cpu` and every available GPU and reports the rest as unavailable. A device
named with `--device` that cannot run is reported as `FAILED` with its error, and the script exits
non-zero. It never runs the CPU in its place.

Hardware we test on: aarch64 Linux (Docker on a Mac mini M1) and the x86_64 CI runner for `cpu`.
`vulkan` and `cuda` have not run on Linux yet (status in `../README.md`).

## Run

```sh
uv run python run.py                    # cpu, then every available GPU
uv run python run.py --device vulkan    # one device; it must be available
```

`uv` installs torchnative (which provides `torch`) and `transformers` 5.15.1 from PyPI. torchnative
publishes manylinux wheels for x86_64 and aarch64. The model is downloaded into the Hugging Face
cache on the first run.

To run against a torchnative checkout instead of the published wheel, build the shim
(`scripts/vendor/vendor_torch.sh`, then `scripts/vendor/install_shim.sh`) and run the script with that
checkout's `torchnative/python` first on `PYTHONPATH`.

## Expected output

The availability lines, then for each device the streamed text and one line:

```
AVAILABILITY {"device": "vulkan", "available": false, "reason": "no_loader"}
COMMON_TASK {"device": "cpu", "text": " a very powerful technique for learning from data. ...", "new_tokens": 32, "tokens_per_second": ..., "agrees_with_upstream": true, "ids": [...]}
```

`agrees_with_upstream` compares the token ids with `reference.json`, which `reference.py` wrote with
upstream torch 2.13.0 and transformers 5.15.1 (not with torchnative). The macOS and Android
examples use the same ids. Equal ids are the criterion on every device.

## Last recorded result

Not run yet.
