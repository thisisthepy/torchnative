"""Every aten op BigVGAN's generator forward dispatches, captured upstream."""
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from vsbig import BigVGANConfig, BigVGANModel

OUT = Path("/Volumes/macMini/voice4-work/ref")
CKPT = "/Volumes/macMini/voice4-work/bigvgan-converted"

seen = {}
class Capture(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        seen[name] = seen.get(name, 0) + 1
        return func(*args, **(kwargs or {}))

cfg = BigVGANConfig.from_pretrained(CKPT)
model = BigVGANModel.from_pretrained(CKPT, dtype=torch.float32).eval()
mel = torch.from_numpy(np.load(OUT / "mel_input.npy"))

with torch.no_grad(), Capture():
    out = model(input_features=mel)
print("audio", tuple(out.audio_values.shape))

ops = sorted(seen)
print("distinct ops:", len(ops))
for o in ops:
    print(f"  {seen[o]:7d}  {o}")
(OUT / "forward_ops.json").write_text(json.dumps(seen, indent=2, sort_keys=True))

# Construction is a separate question -- BigVGAN builds its anti-alias filters in
# __init__, which is where kaiser_window and sinc live.
seen.clear()
with Capture():
    BigVGANModel(cfg)
cons = sorted(seen)
print("\nconstruction distinct ops:", len(cons))
for o in cons:
    print(f"  {seen[o]:7d}  {o}")
(OUT / "construct_ops.json").write_text(json.dumps(seen, indent=2, sort_keys=True))
