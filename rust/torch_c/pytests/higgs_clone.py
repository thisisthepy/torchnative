#!/usr/bin/env python3
import argparse, glob, json, time, os, subprocess, sys
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")

CONVERSATION = [
    {"role": "system", "content": [{"type": "text", "text": "Generate audio."}]},
    {"role": "scene", "content": [
        {"type": "text", "text": "Audio is recorded from a quiet room."},
        {"type": "audio", "audio_url": "dummy"} 
    ]},
    {"role": "user", "content": [{"type": "text", "text": "This is a cloned voice."}]}
]

_STUB = '''
__version__ = "0.0.0-torchnative-stub"
__spec__ = None
import math
import torch
import torch.nn.functional as F

class Functional:
    @staticmethod
    def resample(waveform: torch.Tensor, orig_freq: int, new_freq: int, lowpass_filter_width: int = 6, rolloff: float = 0.99, resampling_method: str = "sinc_interp_hann", beta: float | None = None) -> torch.Tensor:
        if orig_freq == new_freq: return waveform
        gcd = math.gcd(int(orig_freq), int(new_freq))
        orig_freq = int(orig_freq) // gcd
        new_freq = int(new_freq) // gcd
        base_freq = min(orig_freq, new_freq) * rolloff
        width = math.ceil(lowpass_filter_width * orig_freq / base_freq)
        idx_dtype = waveform.dtype if waveform.dtype.is_floating_point else torch.float64
        idx = (torch.arange(-width, width + orig_freq, dtype=idx_dtype, device=waveform.device)[None, None] / orig_freq)
        t = (torch.arange(0, -new_freq, -1, dtype=waveform.dtype, device=waveform.device)[:, None, None] / new_freq + idx)
        t *= base_freq
        t = t.clamp(-lowpass_filter_width, lowpass_filter_width)
        if resampling_method == "sinc_interp_hann":
            window = torch.cos(t * math.pi / lowpass_filter_width / 2) ** 2
        else:
            if beta is None: beta = 14.769656459379492
            beta_tensor = torch.tensor(float(beta), dtype=waveform.dtype, device=waveform.device)
            window = torch.i0(beta_tensor * torch.sqrt(1 - (t / lowpass_filter_width) ** 2)) / torch.i0(beta_tensor)
        t *= math.pi
        scale = base_freq / orig_freq
        kernels = torch.where(t == 0, torch.tensor(1.0, dtype=waveform.dtype, device=waveform.device), torch.sin(t) / t)
        kernels *= window * scale
        if not waveform.dtype.is_floating_point: kernels = kernels.to(dtype=torch.float32)
        shape = waveform.size()
        waveform = waveform.view(-1, shape[-1])
        num_wavs, length = waveform.shape
        waveform = F.pad(waveform, (width, width + orig_freq))
        resampled = F.conv1d(waveform[:, None], kernels, stride=orig_freq)
        resampled = resampled.transpose(1, 2).reshape(num_wavs, -1)
        target_length = torch.ceil(torch.as_tensor(new_freq * length / orig_freq)).long()
        resampled = resampled[..., :target_length]
        return resampled.view(shape[:-1] + resampled.shape[-1:])

functional = Functional()
'''

def _write_stub(d):
    pkg = os.path.join(d, "stubs", "torchaudio")
    os.makedirs(pkg, exist_ok=True)
    with open(os.path.join(pkg, "__init__.py"), "w") as f: f.write(_STUB)
    return os.path.join(d, "stubs")

_RUN = r'''
import argparse, glob, json, time, os, sys
ap = argparse.ArgumentParser()
ap.add_argument("--side"); ap.add_argument("--out"); ap.add_argument("--tok-dtype")
a = ap.parse_args()

import torch, numpy as np
is_shim = hasattr(torch._C, "_aten_implemented")
assert is_shim == (a.side == "shim"), "wrong torch for side=%s" % a.side

GEN = glob.glob(os.environ["HF_HOME"] + "/hub/models--bosonai--higgs-audio-v2-generation-3B-base/snapshots/*")[0]
import sys
sys.modules.pop("torchaudio", None)
from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration
CONVERSATION = json.loads(os.environ["HIGGS_CONVERSATION"])

proc = AutoProcessor.from_pretrained(GEN)
dt = getattr(torch, a.tok_dtype)
proc.audio_tokenizer.to(dt); proc.audio_tokenizer.eval()

# Generate reference audio (random but deterministic)
np.random.seed(42)
dummy_audio = np.random.randn(24000).astype(np.float32 if a.tok_dtype == "float32" else np.float64)

text = proc.apply_chat_template(CONVERSATION, tokenize=False, add_generation_prompt=True)
orig_encode = proc.audio_tokenizer.encode
proc.audio_tokenizer.encode = lambda **kw: orig_encode(**{k: v.to(dt) if v.is_floating_point() else v for k, v in kw.items()})

from transformers import BatchFeature
proc.feature_extractor = lambda *a, **kw: BatchFeature({"input_values": torch.tensor(dummy_audio.tolist(), dtype=dt).unsqueeze(0).unsqueeze(0)})
inputs = proc(text=text, audio=[dummy_audio.tolist()], sampling_rate=24000, return_tensors="pt")
torch.save(inputs, a.out + "_inputs.pt")

pass

t = time.time()
torch.manual_seed(0)
model = HiggsAudioV2ForConditionalGeneration.from_pretrained(GEN, dtype=torch.bfloat16).eval()
with torch.no_grad():
    out = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=16, return_dict_in_generate=True, use_cache=True)
codes = out.audio_sequences
torch.save(codes, a.out + "_out_codes.pt")
'''

def _call(side, stubs, d, name, tok_dtype):
    out = os.path.join(d, name)
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/Volumes/macMini/caches/hf")
    if side == "shim":
        env["PYTHONPATH"] = _VENDOR_DIR + os.pathsep + stubs
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env["PYTHONPATH"] = stubs
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    env["HIGGS_CONVERSATION"] = json.dumps(CONVERSATION)
    
    t = time.time()
    proc = subprocess.run([sys.executable, "-c", _RUN, "--side", side, "--out", out, "--tok-dtype", tok_dtype], env=env, cwd=_REPO_ROOT)
    if proc.returncode != 0: raise SystemExit("run %s failed" % name)
    print("   %-16s %6.1fs" % (name, time.time() - t), flush=True)
    return out

def main():
    d = "/tmp/higgs_clone"
    os.makedirs(d, exist_ok=True)
    stubs = _write_stub(d)
    
    _call("upstream", stubs, d, "up_f32", "float32")
    _call("shim", stubs, d, "sh_f32", "float32")
    
    import torch
    up_in = torch.load(os.path.join(d, "up_f32_inputs.pt"), weights_only=False)
    sh_in = torch.load(os.path.join(d, "sh_f32_inputs.pt"), weights_only=False)
    
    up_codes = torch.load(os.path.join(d, "up_f32_out_codes.pt"), weights_only=False)
    sh_codes = torch.load(os.path.join(d, "sh_f32_out_codes.pt"), weights_only=False)
    
    report = {
        "encode": {
            "audio_input_ids_equal": bool(torch.equal(up_in["audio_input_ids"], sh_in["audio_input_ids"])),
            "input_ids_equal": bool(torch.equal(up_in["input_ids"], sh_in["input_ids"])),
            "n_audio_codes": int(up_in["audio_input_ids"].shape[1])
        },
        "generate": {
            "codes_total": int(up_codes.shape[1]),
            "codes_matching": int((up_codes == sh_codes).all(dim=-1).sum().item()),
            "first_differing_step": int((up_codes != sh_codes).any(dim=-1).nonzero()[0][1].item()) if not torch.equal(up_codes, sh_codes) else -1
        }
    }
    with open(os.path.join(_HERE, "higgs_clone.json"), "w") as f:
        json.dump(report, f, indent=1)
    print(json.dumps(report, indent=1))

if __name__ == "__main__": main()
