"""The common task on a Mac, on `cpu`, `mps` and `npu` side by side (examples/README.md).

SmolLM2-135M in float32, loaded with transformers' `from_pretrained`, greedy, 32 new
tokens after the prompt "On-device inference is", streamed. Each device's line names
the device, the text, tokens per second, and whether the token ids equal upstream's
(`reference.json`, written by `reference.py` with upstream torch, not torchnative).

    uv run python run.py                 # cpu, mps, npu
    uv run python run.py --device mps    # one device

- `mps` is `model.to("mps")` (Metal). If an op refuses there, the refusal is printed
  together with `torch._C._shim_mps_host_readback_ops()`, the ops this build answers
  on mps by reading back to the host.
- `npu` is `model.to(torchnative.device.npu)`: the Apple Neural Engine through CoreML.

A device that cannot run is reported as failed and the script exits non-zero. Nothing
falls back to the CPU silently.
"""

import argparse
import json
import os
import sys
import time

MODEL = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "On-device inference is"
NEW_TOKENS = 32
DEVICES = ("cpu", "mps", "npu")
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
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    inputs = tokenizer(PROMPT, return_tensors="pt")
    if device == "mps":
        model.to("mps")
        inputs = {k: v.to("mps") for k, v in inputs.items()}
    elif device == "npu":
        from torchnative import device as tn_device

        model.to(tn_device.npu)
    elif device != "cpu":
        raise ValueError(f"device must be one of {DEVICES}, not {device!r}")

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


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--device", choices=DEVICES, action="append",
                        help="run only this device (repeatable); default: all three")
    args = parser.parse_args(argv)

    import torch

    if not hasattr(torch._C, "_aten_implemented"):
        raise SystemExit("the torch imported here is not torchnative's (no torch._C._aten_implemented)")
    reference = _load_reference()

    failed = []
    for device in args.device or DEVICES:
        print(f"[{device}] {PROMPT}", end="", flush=True)
        try:
            result = run(device, reference, lambda t: print(t, end="", flush=True))
        except Exception as exc:  # reported by name, never replaced by a cpu run
            print(f"\n[{device}] FAILED {type(exc).__name__}: {exc}")
            if device == "mps":
                print(f"[mps] ops answered by host readback in this build: "
                      f"{list(torch._C._shim_mps_host_readback_ops())}")
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
