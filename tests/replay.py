import json, os, sys
import numpy as np
import torch
from vsbig import BigVGANModel
marker = "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"
print("MARKER", marker, torch.__version__, flush=True)
import json as _j
mel = torch.tensor(_j.load(open("/Volumes/macMini/voice4-work/ref/mel_input.json")), dtype=torch.float32)
model = BigVGANModel.from_pretrained("/Volumes/macMini/voice4-work/bigvgan-converted",
                                     dtype=torch.float32).eval()
print("loaded", sum(p.numel() for p in model.parameters()), "params", flush=True)
with torch.no_grad():
    audio = model(input_features=mel).audio_values
print("audio", tuple(audio.shape), flush=True)
_j.dump(audio.reshape(-1).tolist(), open(f"/Volumes/macMini/voice4-work/ref/audio_{marker}_replay.json","w"))
print("saved", flush=True)
