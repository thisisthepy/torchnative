"""Write the common task's upstream token ids to the example's `reference.json`.

Runs on a host with **upstream** torch (not torchnative), so every device's run is
checked against upstream and not against torchnative itself:

    uv run --no-project --with torch==2.13.0 --with transformers==5.15.1 python reference.py

(`--no-project` keeps this example's torchnative dependency out: both provide `torch`.)

It refuses to write if the torch it imported is torchnative's.
"""

import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from run import MODEL, NEW_TOKENS, PROMPT, REFERENCE  # noqa: E402

if hasattr(torch._C, "_aten_implemented"):
    raise SystemExit("this torch is torchnative's; the reference must come from upstream torch")

tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32).eval()
inputs = tokenizer(PROMPT, return_tensors="pt")
with torch.no_grad():
    out = model.generate(**inputs, do_sample=False, max_new_tokens=NEW_TOKENS)
ids = out[0, inputs["input_ids"].shape[1]:].tolist()
record = {
    "model": MODEL,
    "prompt": PROMPT,
    "new_tokens": NEW_TOKENS,
    "ids": ids,
    "text": tokenizer.decode(ids, skip_special_tokens=True),
    "torch": torch.__version__,
    "transformers": __import__("transformers").__version__,
}
with open(REFERENCE, "w", encoding="utf-8") as fh:
    json.dump(record, fh, indent=2, ensure_ascii=False)
    fh.write("\n")
print(json.dumps(record, ensure_ascii=False))
