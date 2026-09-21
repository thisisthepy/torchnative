"""Run voicestudio's BigVGAN v2 (real NVIDIA weights) on real speech, upstream torch.

Writes the mel it vocodes and the waveform it produces, at float32 and at float64,
so a shim replay can be scored against both (docs/AGREE.md's method).
"""
import json, sys, wave
from pathlib import Path

import numpy as np
import torch

from vsbig import BigVGANConfig, BigVGANModel
from vsbig.modeling_bigvgan import mel_spectrogram

OUT = Path("/Volumes/macMini/voice4-work/ref")
OUT.mkdir(exist_ok=True)
CKPT = "/Volumes/macMini/voice4-work/bigvgan-converted"

# --- real speech in ---------------------------------------------------------
with wave.open("/Volumes/macMini/voice4-work/mlk24k.wav", "rb") as w:
    assert w.getframerate() == 24000 and w.getnchannels() == 1 and w.getsampwidth() == 2
    raw = w.readframes(w.getnframes())
pcm = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
# One second, from a voiced stretch rather than the leading silence.
SECONDS, START = 1.0, 24000
pcm = pcm[START : START + int(24000 * SECONDS)]
wav64 = torch.from_numpy(pcm)[None, :]           # (1, N) float64

cfg = BigVGANConfig.from_pretrained(CKPT)
print("sampling_rate", cfg.sampling_rate, "n_fft", cfg.n_fft, "hop", cfg.hop_length,
      "win", cfg.win_length, "mels", cfg.model_in_dim, "fmin", cfg.mel_fmin, "fmax", cfg.mel_fmax)

def build(dtype):
    m = BigVGANModel.from_pretrained(CKPT, dtype=dtype)
    m.eval()
    # weights arrive already folded: the converted checkpoint carries no parametrization
    return m

def mel_for(wav, dtype):
    mel = mel_spectrogram(
        wav.to(dtype), sampling_rate=cfg.sampling_rate, n_fft=cfg.n_fft,
        hop_length=cfg.hop_length, win_length=cfg.win_length,
        num_mel_bins=cfg.model_in_dim, fmin=cfg.mel_fmin, fmax=cfg.mel_fmax,
        centered=False,
    )
    from vsbig.modeling_bigvgan import dynamic_range_compression
    return dynamic_range_compression(mel)

# The mel front end is float32 internally whatever it is handed, so build it ONCE
# and give both model dtypes the identical input -- the comparison is about the
# generator, not about two different spectrograms.
mel32 = mel_for(wav64, torch.float32)
np.save(OUT / "mel_input.npy", mel32.detach().numpy())

res = {}
for name, dtype in (("f32", torch.float32), ("f64", torch.float64)):
    mel = mel32.to(dtype)
    model = build(dtype)
    with torch.no_grad():
        out = model(input_features=mel)
    a = out.audio_values
    print(name, "mel", tuple(mel.shape), "audio", tuple(a.shape),
          "absmax", float(a.abs().max()), "rms", float(a.double().pow(2).mean().sqrt()))
    np.save(OUT / f"mel_{name}.npy", mel.detach().numpy())
    np.save(OUT / f"audio_{name}.npy", a.detach().numpy())
    res[name] = dict(mel_shape=list(mel.shape), audio_shape=list(a.shape),
                     absmax=float(a.abs().max()))
    del model

# --- upstream's own float32-vs-float64 error, the tolerance source ----------
a32 = torch.from_numpy(np.load(OUT / "audio_f32.npy")).double()
a64 = torch.from_numpy(np.load(OUT / "audio_f64.npy")).double()
scale = float(a64.abs().max())
rel = float((a32 - a64).abs().max() / scale)
res["upstream_f32_vs_f64_rel"] = rel
res["scale"] = scale
res["num_params"] = 112414512
print("UPSTREAM f32-vs-f64 rel on the waveform:", rel, "scale", scale)
(OUT / "ref.json").write_text(json.dumps(res, indent=2))
