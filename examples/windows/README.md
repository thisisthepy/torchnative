# Windows example

Runs the common task (`../README.md`) on one Windows machine, on every device it has:

| Device | What it is | How the script reaches it |
|---|---|---|
| `cpu` | torchnative's CPU kernels | the model as loaded |
| `npu` | an Intel Core Ultra NPU through OpenVINO | `model.to(torchnative.device.npu)`; the model is lowered on the move, the inputs stay on the CPU |
| `cuda` | candle's CUDA backend | `model.to(torchnative.device.cuda)`, inputs moved to `cuda` |

The script first prints one `AVAILABILITY` line per device, with the reason its probe gave when the
device is not there (for example `unresolved` for `npu` without OpenVINO or without an Intel NPU,
`not_built` for `cuda`). With no `--device`, it runs `cpu` and every available accelerator and
reports the rest as unavailable. A device named with `--device` that cannot run is reported as
`FAILED` with its error, and the script exits non-zero. It never runs the CPU in its place.

Where it runs: the `windows` job of `.github/workflows/test.yml` runs this script after the gate, on
the shim that job built, on the x86_64 runner. The runner has no NPU and no GPU, so it runs `cpu`
and prints why `npu` and `cuda` are unavailable. `npu` on an Intel Core Ultra laptop is run by the
owner (status in `../README.md`).

## Run

```sh
uv run python run.py                  # cpu, then every available accelerator
uv run python run.py --device npu     # one device; it must be available
```

`uv` installs torchnative (which provides `torch`) and `transformers` 5.15.1 from PyPI. The model is
downloaded into the Hugging Face cache on the first run.

To run against a torchnative checkout instead of the published wheel, build the shim
(`scripts/vendor/vendor_torch.sh`, then `scripts/vendor/install_shim.sh`) and run the script with that
checkout's `torchnative/python` first on `PYTHONPATH`.

## Expected output

The availability lines, then for each device the streamed text and one line:

```
AVAILABILITY {"device": "npu", "available": false, "reason": "unresolved"}
COMMON_TASK {"device": "cpu", "text": " a very powerful technique for learning from data. ...", "new_tokens": 32, "tokens_per_second": ..., "agrees_with_upstream": true, "ids": [...]}
```

`agrees_with_upstream` compares the token ids with `reference.json`, which `reference.py` wrote with
upstream torch 2.13.0 and transformers 5.15.1 (not with torchnative). The macOS, Linux and Android
examples use the same ids. Equal ids are the criterion on every device.

## Last recorded result

Not run yet.
