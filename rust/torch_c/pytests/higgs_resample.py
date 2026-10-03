#!/usr/bin/env python3
import json, os, subprocess, sys
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")

_STUB = '''
import math
import torch
import torch.nn.functional as F

def resample(waveform: torch.Tensor, orig_freq: int, new_freq: int, lowpass_filter_width: int = 6, rolloff: float = 0.99, resampling_method: str = "sinc_interp_hann", beta: float = None) -> torch.Tensor:
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
'''

_SRC = '''
import sys
import torch

def run(name, dtype):
    torch.manual_seed(42)
    waveform = torch.zeros(1, 24000) # 1 second of audio at 24kHz
    if dtype == "float64":
        waveform = waveform.to(torch.float64)
    import torchaudio
    out = torchaudio.functional.resample(waveform, 24000, 16000)
    out_f32 = out.to(torch.float32)
    import numpy as np
    np.save(f"/tmp/higgs_resample_{name}.npy", out_f32.cpu().numpy())

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--shim":
        import test_shim
    run(sys.argv[2], sys.argv[3])
'''

def main():
    d = "/tmp/higgs_resample"
    os.makedirs(d, exist_ok=True)
    
    stubs = os.path.join(d, "stubs")
    ta = os.path.join(stubs, "torchaudio")
    os.makedirs(ta, exist_ok=True)
    with open(os.path.join(stubs, "torchaudio", "__init__.py"), "w") as f:
        f.write("__version__ = '0.0.0-torchnative-stub'\nfrom . import functional\n")
    with open(os.path.join(stubs, "torchaudio", "functional.py"), "w") as f:
        f.write(_STUB)
        
    with open(os.path.join(d, "run.py"), "w") as f:
        f.write(_SRC)
        
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{stubs}:{_REPO_ROOT}/rust/torch_c/pytests"

    print("Running up_f64...")
    subprocess.run([sys.executable, os.path.join(d, "run.py"), "--upstream", "up_f64", "float64"], env=env, check=True)
    print("Running up_f32...")
    subprocess.run([sys.executable, os.path.join(d, "run.py"), "--upstream", "up_f32", "float32"], env=env, check=True)
    
    print("Running sh_f32...")
    env_sh = env.copy()
    env_sh["TORCH_USE_RTLD_GLOBAL"] = "1"
    env_sh["PYTHONPATH"] = f"{_VENDOR_DIR}:{_REPO_ROOT}/rust/torch_c/pytests"
    subprocess.run([sys.executable, os.path.join(d, "run.py"), "--shim", "sh_f32", "float32"], env=env_sh, check=True)

    up_f64 = np.load("/tmp/higgs_resample_up_f64.npy")
    up_f32 = np.load("/tmp/higgs_resample_up_f32.npy")
    sh_f32 = np.load("/tmp/higgs_resample_sh_f32.npy")
    
    # Calculate relative error over float64 oracle
    def rel_err(a, oracle):
        return float(np.abs(a - oracle).max() / np.abs(oracle).max())
        
    up_f32_err = rel_err(up_f32, up_f64)
    sh_f32_err = rel_err(sh_f32, up_f64)
    
    print(f"up_f32 vs f64 error: {up_f32_err}")
    print(f"sh_f32 vs f64 error: {sh_f32_err}")
    
    out = {
        "resample": {
            "shim_vs_f64": sh_f32_err,
            "upstream_f32_vs_f64": up_f32_err,
            "ratio": sh_f32_err / up_f32_err if up_f32_err > 0 else 0
        }
    }
    
    with open("rust/torch_c/pytests/higgs_resample.json", "w") as f:
        json.dump(out, f, indent=1)

if __name__ == "__main__":
    main()
