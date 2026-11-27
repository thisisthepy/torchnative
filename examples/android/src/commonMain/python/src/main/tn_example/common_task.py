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
from typing import Callable, cast

MODEL = "HuggingFaceTB/SmolLM2-135M"
PROMPT = "On-device inference is"
NEW_TOKENS = 32
REFERENCE = os.path.join(os.path.dirname(__file__), "reference.json")


def run(device: str = "cpu", emit: Callable[[str], None] = print) -> dict[str, object]:
    """Run the common task on `device`; `emit` receives each streamed piece of text."""
    import torch

    if not hasattr(torch._C, "_aten_implemented"):
        raise RuntimeError("the torch imported here is not torchnative's (no torch._C._aten_implemented)")
    from transformers import (AutoModelForCausalLM, AutoTokenizer, BatchEncoding, PreTrainedModel,
                              PreTrainedTokenizerBase, TextStreamer)

    with open(REFERENCE, encoding="utf-8") as fh:
        reference = cast(dict[str, object], json.load(fh))
    if reference["model"] != MODEL or reference["prompt"] != PROMPT:
        raise RuntimeError(f"reference.json is for another task: {reference['model']!r}, {reference['prompt']!r}")

    loaded = AutoTokenizer.from_pretrained(MODEL)
    if loaded is None:
        raise RuntimeError(f"transformers found no tokenizer for {MODEL}")
    tokenizer: PreTrainedTokenizerBase = loaded
    model = cast(PreTrainedModel, AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)).eval()
    if device == "npu":
        from torchnative import device as tn_device

        # torchnative's `Module.to` takes its own `Device` objects; torchnative ships no
        # `py.typed`, so the checker sees neither, and the call is typed as torch's.
        cast(object, model.to(cast(torch.device, tn_device.npu)))  # `to` returns untyped
    elif device != "cpu":
        raise ValueError(f"device must be 'cpu' or 'npu', not {device!r}")

    class _Streamer(TextStreamer):
        def on_finalized_text(self, text: str, stream_end: bool = False) -> None:
            emit(text)

    inputs = cast(BatchEncoding, tokenizer(PROMPT, return_tensors="pt"))
    start = time.perf_counter()
    with torch.no_grad():
        # transformers leaves `generate`'s return untyped; the check below narrows it.
        out = cast(object, model.generate(**inputs, do_sample=False, max_new_tokens=NEW_TOKENS,
                                          streamer=_Streamer(tokenizer, skip_prompt=True)))
    seconds = time.perf_counter() - start
    if not isinstance(out, torch.Tensor):
        raise RuntimeError(f"generate returned {type(out).__name__}, not the token-id tensor")
    prompt_length = cast(torch.Tensor, inputs["input_ids"]).shape[1]
    ids = cast(list[int], out[0, prompt_length:].tolist())
    return {
        "device": device,
        "text": tokenizer.decode(ids, skip_special_tokens=True),
        "new_tokens": len(ids),
        "tokens_per_second": round(len(ids) / seconds, 2),
        "agrees_with_upstream": ids == reference["ids"],
        "ids": ids,
    }


def run_json(device: str = "cpu", emit: Callable[[str], None] = print) -> str:
    """`run`, as one JSON line (what the Android activity logs)."""
    return json.dumps(run(device, emit), ensure_ascii=False)
