"""The common task on Windows, on `cpu`, `npu` and `cuda` (examples/README.md).

SmolLM2-135M in float32, loaded with transformers' `from_pretrained`, greedy, 32 new
tokens after the prompt "On-device inference is", streamed. Each device's line names
the device, the text, tokens per second, and whether the token ids equal upstream's
(`reference.json`, written by `reference.py` with upstream torch, not torchnative).

    uv run python run.py                  # cpu, then every accelerator this machine has
    uv run python run.py --device npu     # one device; it must be there

- `npu` is `model.to(torchnative.device.npu)`: an Intel Core Ultra NPU through OpenVINO.
  The model is lowered on the move; the inputs stay on the CPU.
- `cuda` is `model.to(torchnative.device.cuda)`: candle's CUDA backend.

Before running, every device's `availability()` is printed with the reason its probe
gave. Without `--device`, an accelerator that is not available is reported and
skipped; a device named with `--device` that cannot run fails. A device that fails or
disagrees makes the script exit non-zero. Nothing falls back to the CPU silently.
"""

import argparse
import json
import os
import sys
import time

MODEL = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "On-device inference is"
NEW_TOKENS = 32
DEVICES = ("cpu", "npu", "cuda")
REFERENCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference.json")


def _load_reference():
    with open(REFERENCE, encoding="utf-8") as fh:
        reference = json.load(fh)
    if reference["model"] != MODEL or reference["prompt"] != PROMPT:
        raise RuntimeError(f"reference.json is for another task: {reference['model']!r}, {reference['prompt']!r}")
    return reference


def run(device, reference, emit):
    """Run the common task on `device`; `emit` receives each streamed piece of text."""
    import torch
    from torchnative import device as tn_device
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

    if device not in DEVICES:
        raise ValueError(f"device must be one of {DEVICES}, not {device!r}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    inputs = tokenizer(PROMPT, return_tensors="pt")
    if device == "npu":
        model.to(tn_device.npu)  # lowers to the NPU, or refuses by name
    elif device == "cuda":
        target = tn_device.cuda
        model.to(target)  # refuses by name when the device cannot run
        inputs = {k: v.to(target.torch_device) for k, v in inputs.items()}

    class _Streamer(TextStreamer):
        def on_finalized_text(self, text, stream_end=False):
            emit(text)

    start = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, max_new_tokens=NEW_TOKENS,
                             streamer=_Streamer(tokenizer, skip_prompt=True))
    seconds = time.perf_counter() - start
    ids = out[0, inputs["input_ids"].shape[1]:].tolist()
    return {
        "device": device,
        "text": tokenizer.decode(ids, skip_special_tokens=True),
        "new_tokens": len(ids),
        "tokens_per_second": round(len(ids) / seconds, 2),
        "agrees_with_upstream": ids == reference["ids"],
        "ids": ids,
    }


def _availability(device):
    from torchnative import device as tn_device

    if device == "cpu":
        return True, None
    found = getattr(tn_device, device).availability()
    return bool(found.available), found.reason


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=DEVICES, action="append",
                        help="run only this device (repeatable); it must be available")
    args = parser.parse_args(argv)

    import torch

    if not hasattr(torch._C, "_aten_implemented"):
        raise SystemExit("the torch imported here is not torchnative's (no torch._C._aten_implemented)")
    reference = _load_reference()

    chosen = []
    for device in DEVICES:
        ok, reason = _availability(device)
        print(f"AVAILABILITY {json.dumps({'device': device, 'available': ok, 'reason': reason})}")
        if args.device:
            if device in args.device:
                chosen.append(device)
        elif ok:
            chosen.append(device)

    failed = []
    for device in chosen:
        print(f"[{device}] {PROMPT}", end="", flush=True)
        try:
            result = run(device, reference, lambda t: print(t, end="", flush=True))
        except Exception as exc:  # reported by name, never replaced by a cpu run
            print(f"\n[{device}] FAILED {type(exc).__name__}: {exc}")
            failed.append(device)
            continue
        print()
        print("COMMON_TASK " + json.dumps(result, ensure_ascii=False))
        if not result["agrees_with_upstream"]:
            failed.append(device)
    if failed:
        print(f"failed or disagreed: {failed}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
