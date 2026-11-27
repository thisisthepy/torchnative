"""The common task every torchnative example runs first (examples/README.md).

SmolLM2-135M in float32, loaded with transformers' `from_pretrained`, greedy, 32 new
tokens after the prompt "On-device inference is", streamed. The result names the
device, the text, tokens per second, and whether the token ids equal upstream's:
`reference.json`, written on a host by `examples/android/reference.py` with upstream
torch (not torchnative), so the check is against upstream and not against ourselves.

`device` is "cpu" or "npu". "npu" is `model.to(torchnative.device.npu)` (#423). A
device that cannot run raises; nothing falls back to the CPU silently.
"""

import json
import os
import time

MODEL = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "On-device inference is"
NEW_TOKENS = 32
REFERENCE = os.path.join(os.path.dirname(__file__), "reference.json")


def run(device="cpu", emit=print):
    """Run the common task on `device`; `emit` receives each streamed piece of text."""
    import torch

    if not hasattr(torch._C, "_aten_implemented"):
        raise RuntimeError("the torch imported here is not torchnative's (no torch._C._aten_implemented)")
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

    with open(REFERENCE, encoding="utf-8") as fh:
        reference = json.load(fh)
    if reference["model"] != MODEL or reference["prompt"] != PROMPT:
        raise RuntimeError(f"reference.json is for another task: {reference['model']!r}, {reference['prompt']!r}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
    if device == "npu":
        from torchnative import device as tn_device

        model.to(tn_device.npu)
    elif device != "cpu":
        raise ValueError(f"device must be 'cpu' or 'npu', not {device!r}")

    class _Streamer(TextStreamer):
        def on_finalized_text(self, text, stream_end=False):
            emit(text)

    inputs = tokenizer(PROMPT, return_tensors="pt")
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


def run_json(device="cpu", emit=print):
    """`run`, as one JSON line (what the app writes to Documents/common-task.txt)."""
    return json.dumps(run(device, emit), ensure_ascii=False)
