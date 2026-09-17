#!/usr/bin/env python3
"""Regenerate `higgs_e2e.json` -- docs/architectures/VOICE5.md's measurement.

NOT wired into `run.sh` and must not be: it needs an 11.5 GB checkpoint, both
interpreters, and about fifteen minutes. `test_higgs.py` reads the recorded
JSON; this script is what produced it.

Six runs, in this order, because each is the input to the next:

    1  upstream  bf16 LM, 64 greedy steps   -> codes, waveform        (the reference)
    2  shim      bf16 LM, 64 greedy steps   -> codes, waveform        (the replay)
    3  upstream  bf16 LM,  8 greedy steps, output_scores              (logits)
    4  shim      bf16 LM,  8 greedy steps, output_scores              (logits)
    5  upstream  f32  LM,  8 greedy steps, output_scores              (the LOGIT oracle)
    6  decoder alone on run 1's codes: upstream f32, upstream f64, shim f32
                                                                     (the WAVEFORM oracle)

Why the oracle is (bf16 vs f32) for the logits and (f32 vs f64) for the
waveform: docs/numerics/AGREE.md §2 scores a computation against the SAME
computation one precision up. The language model runs in bfloat16 -- 5.4B
parameters in float32 is 21.5 GB on a 16 GB machine and swaps -- so its oracle
is float32. The audio decoder is 600 MB and runs in float32, so its oracle is
float64, exactly as VOICE4.md §5.1 did for BigVGAN.

Two substitutions, both recorded because a substitution that is not recorded is
a place where the measurement is not of what it says:

  * **`torchaudio` is not installed** in the spike venv and
    `HiggsAudioV2TokenizerModel` is gated behind `@requires(backends=
    ("torchaudio",))`. A stub module satisfies the gate; every entry point in
    it RAISES, and the run records that none was called. It is reachable only
    from the tokenizer's ENCODE path (`torchaudio.functional.resample` on input
    audio), and this round decodes only. See §7 for what that costs.
  * **`HF_HOME` is moved to the external disk** (`/Volumes/macMini/caches/hf`).
    The internal disk has 12 GB free and the checkpoint is 11.5 GB.

Run it as:

    PY=/Volumes/macMini/caches/spike-venv/bin/python
    root=$(git rev-parse --show-toplevel)
    HF_HOME=/Volumes/macMini/caches/hf  $PY higgs_e2e.py --stage all --dir /tmp/higgs
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")

# The model card's own "Single-speaker smart voice" example, verbatim.
# `add_generation_prompt=True` is part of it and is load bearing: without it,
# greedy decoding emits the audio-stream BOS and EOS back to back and stops
# after ten steps with 0.04 s of near-silence (measured -- absmax 1.4e-04).
CONVERSATION = [
    {"role": "system", "content": [{"type": "text", "text": "Generate audio following instruction."}]},
    {"role": "scene", "content": [{"type": "text", "text": "Audio is recorded from a quiet room."}]},
    {"role": "user", "content": [{"type": "text", "text": (
        "The sun rises in the east and sets in the west. This simple fact has "
        "been observed by humans for thousands of years.")}]},
]

_STUB = '''
"""A deliberately empty `torchaudio` -- see higgs_e2e.py's docstring."""
__version__ = "0.0.0-torchnative-stub"
CALLS = []


class _Raiser:
    def __init__(self, path):
        self._path = path

    def __getattr__(self, name):
        return _Raiser(self._path + "." + name)

    def __call__(self, *a, **k):
        CALLS.append(self._path)
        raise NotImplementedError(
            "torchnative's torchaudio stub was CALLED at %s -- this round's "
            "claim that the decode path never reaches torchaudio is false"
            % self._path)


functional = _Raiser("torchaudio.functional")
transforms = _Raiser("torchaudio.transforms")
'''


def _write_stub(d):
    pkg = os.path.join(d, "stubs", "torchaudio")
    os.makedirs(pkg, exist_ok=True)
    with open(os.path.join(pkg, "__init__.py"), "w") as f:
        f.write(_STUB)
    return os.path.join(d, "stubs")


def _env(side, stubs):
    env = dict(os.environ)
    env.setdefault("HF_HOME", "/Volumes/macMini/caches/hf")
    if side == "shim":
        env["PYTHONPATH"] = _VENDOR_DIR + os.pathsep + stubs
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env["PYTHONPATH"] = stubs
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    return env


_RUN = r'''
import argparse, glob, json, time, os
ap = argparse.ArgumentParser()
ap.add_argument("--side"); ap.add_argument("--out")
ap.add_argument("--steps", type=int, default=64)
ap.add_argument("--dtype", default="bfloat16")
ap.add_argument("--tok-dtype", default="float32")
ap.add_argument("--scores", action="store_true")
ap.add_argument("--codes", default=None)
ap.add_argument("--wav", default=None)
a = ap.parse_args()

import torch, numpy as np
is_shim = hasattr(torch._C, "_aten_implemented")
assert is_shim == (a.side == "shim"), "wrong torch for side=%s" % a.side

GEN = glob.glob(os.environ["HF_HOME"] + "/hub/models--bosonai--higgs-audio-v2-generation-3B-base/snapshots/*")[0]
from transformers import AutoProcessor, HiggsAudioV2ForConditionalGeneration
CONVERSATION = json.loads(os.environ["HIGGS_CONVERSATION"])

rec = {"side": a.side, "dtype": a.dtype, "tok_dtype": a.tok_dtype, "steps": a.steps,
       "torch": torch.__version__}
proc = AutoProcessor.from_pretrained(GEN)
proc.audio_tokenizer.to(getattr(torch, a.tok_dtype)); proc.audio_tokenizer.eval()

if a.codes:
    codes = torch.tensor(json.load(open(a.codes)), dtype=torch.int64)
else:
    inputs = proc.apply_chat_template(CONVERSATION, add_generation_prompt=True, return_dict=True,
                                      tokenize=True, sampling_rate=24000, return_tensors="pt")
    rec["input_ids"] = inputs["input_ids"].tolist()
    t = time.time()
    model = HiggsAudioV2ForConditionalGeneration.from_pretrained(GEN, dtype=getattr(torch, a.dtype)).eval()
    rec["n_params"] = sum(p.numel() for p in model.parameters())
    rec["t_load"] = round(time.time() - t, 1)
    # ---- the pinned conditions ---------------------------------------
    torch.manual_seed(0)
    t = time.time()
    with torch.no_grad():
        out = model.generate(**inputs, do_sample=False, num_beams=1, max_new_tokens=a.steps,
                             return_dict_in_generate=True, use_cache=True,
                             output_scores=bool(a.scores))
    rec["t_generate"] = round(time.time() - t, 1)
    rec["sequences"] = out.sequences.tolist()
    codes = out.audio_sequences
    if a.scores:
        np.save(a.out + ".scores.npy",
                np.asarray([s.detach().to(torch.float64).reshape(-1).tolist() for s in out.scores],
                           dtype=np.float64))
    del model
rec["audio_sequences"] = codes.tolist()

t = time.time()
with torch.no_grad():
    wavs = proc.batch_decode(codes)
rec["t_decode"] = round(time.time() - t, 1)
arr = np.asarray([float(x) for x in wavs[0].detach().to(torch.float64).reshape(-1).tolist()],
                 dtype=np.float64)
rec["n_samples"] = int(arr.size)
rec["wav_absmax"] = float(np.abs(arr).max()) if arr.size else 0.0
rec["wav_rms"] = float(np.sqrt((arr ** 2).mean())) if arr.size else 0.0
if a.wav:
    np.save(a.wav, arr)
import torchaudio as _ta
rec["torchaudio_stub_calls"] = list(getattr(_ta, "CALLS", ["<real torchaudio>"]))
json.dump(rec, open(a.out, "w"))
print("[%s] %s" % (a.side, {k: rec[k] for k in ("n_samples", "wav_absmax", "wav_rms") if k in rec}), flush=True)
'''


def _call(side, stubs, d, name, **kw):
    out = os.path.join(d, name + ".json")
    argv = [sys.executable, "-c", _RUN, "--side", side, "--out", out]
    for k, v in kw.items():
        flag = "--" + k.replace("_", "-")
        if v is True:
            argv.append(flag)
        elif v is not None:
            argv += [flag, str(v)]
    env = _env(side, stubs)
    env["HIGGS_CONVERSATION"] = json.dumps(CONVERSATION)
    t = time.time()
    proc = subprocess.run(argv, env=env, cwd=_REPO_ROOT)
    if proc.returncode != 0:
        raise SystemExit("run %s (%s) failed" % (name, side))
    print("   %-16s %6.1fs" % (name, time.time() - t), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/tmp/higgs")
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--score-steps", type=int, default=8)
    ap.add_argument("--write", action="store_true",
                    help="overwrite pytests/higgs_e2e.json with the result")
    a = ap.parse_args()
    import numpy as np

    d = a.dir
    os.makedirs(d, exist_ok=True)
    stubs = _write_stub(d)

    up = _call("upstream", stubs, d, "up64", steps=a.steps, wav=os.path.join(d, "up64.npy"))
    sh = _call("shim", stubs, d, "sh64", steps=a.steps, wav=os.path.join(d, "sh64.npy"))
    ub = _call("upstream", stubs, d, "up8", steps=a.score_steps, scores=True)
    sb = _call("shim", stubs, d, "sh8", steps=a.score_steps, scores=True)
    uf = _call("upstream", stubs, d, "up8f32", steps=a.score_steps, scores=True, dtype="float32")

    codes_path = os.path.join(d, "codes.json")
    json.dump(json.load(open(up))["audio_sequences"], open(codes_path, "w"))
    du32 = _call("upstream", stubs, d, "dec_up_f32", codes=codes_path, tok_dtype="float32",
                 wav=os.path.join(d, "dec_up_f32.npy"))
    du64 = _call("upstream", stubs, d, "dec_up_f64", codes=codes_path, tok_dtype="float64",
                 wav=os.path.join(d, "dec_up_f64.npy"))
    dsh = _call("shim", stubs, d, "dec_sh_f32", codes=codes_path, tok_dtype="float32",
                wav=os.path.join(d, "dec_sh_f32.npy"))

    U, S = json.load(open(up)), json.load(open(sh))
    A = np.array(U["audio_sequences"])[0]
    B = np.array(S["audio_sequences"])[0]
    bad = np.where(A != B)
    b = np.load(ub + ".scores.npy")
    s = np.load(sb + ".scores.npy")
    f = np.load(uf + ".scores.npy")
    Ab = np.array(json.load(open(ub))["audio_sequences"])[0]
    Sb = np.array(json.load(open(sb))["audio_sequences"])[0]
    Af = np.array(json.load(open(uf))["audio_sequences"])[0]
    bb = np.where(Ab != Sb)
    first = int(bb[0][0]), int(bb[1][0])
    per_step = []
    for k in range(first[0] + 1):
        m = np.isfinite(f[k]) & np.isfinite(b[k]) & np.isfinite(s[k])
        scale = float(np.abs(f[k][m]).max())
        o = float(np.abs(b[k][m] - f[k][m]).max() / scale)
        per_step.append({
            "step": k, "scale": scale, "oracle_rel": o,
            "shim_vs_upstream_rel": float(np.abs(s[k][m] - b[k][m]).max() / scale),
            "shim_vs_f32_rel": float(np.abs(s[k][m] - f[k][m]).max() / scale),
            "ratio": float(np.abs(s[k][m] - f[k][m]).max() / scale) / o,
        })
    u32 = np.load(os.path.join(d, "dec_up_f32.npy"))
    u64 = np.load(os.path.join(d, "dec_up_f64.npy"))
    ws = np.load(os.path.join(d, "dec_sh_f32.npy"))
    scale = float(np.abs(u32).max())
    rel = lambda x, y: float(np.abs(x - y).max() / scale)
    k, cb = first
    report = {
        "_what": "docs/architectures/VOICE5.md -- Higgs Audio v2 end to end under the "
                 "torch._C shim, and against upstream 2.13.0. Regenerate with "
                 "rust/torch_c/pytests/higgs_e2e.py.",
        "taken": time.strftime("%Y-%m-%d"),
        "commit": subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                                 capture_output=True, text=True, cwd=_REPO_ROOT).stdout.strip(),
        "model": {"generation": "bosonai/higgs-audio-v2-generation-3B-base",
                  "tokenizer": "bosonai/higgs-audio-v2-tokenizer",
                  "n_params": U["n_params"], "hf_home": os.environ.get("HF_HOME")},
        "pinned": {"do_sample": False, "num_beams": 1, "seed": 0, "use_cache": True,
                   "lm_dtype": "bfloat16", "audio_tokenizer_dtype": "float32",
                   "max_new_tokens": a.steps,
                   "prompt": "the model card's Single-speaker smart voice example, "
                             "add_generation_prompt=True",
                   "input_ids_len": len(U["input_ids"][0])},
        "end_to_end": {"shim_reached_audio": True, "n_samples": S["n_samples"],
                       "seconds": S["n_samples"] / 24000.0,
                       "shim_t_load_s": S["t_load"], "shim_t_generate_s": S["t_generate"],
                       "shim_t_decode_s": S["t_decode"],
                       "upstream_t_generate_s": U["t_generate"],
                       "text_sequences_identical": U["sequences"] == S["sequences"],
                       "audio_codes_identical": bool((A == B).all())},
        "trajectory": {
            "first_differing_step": first[0], "first_differing_codebook": first[1],
            "first_differing_step_full_run": int(bad[0][0]),
            "first_differing_codebook_full_run": int(bad[1][0]),
            "prefix_identical_steps": int(bad[0][0]),
            "codes_matching": int((A == B).sum()), "codes_total": int(A.size),
            "tie": {
                "step": k, "codebook": cb,
                "upstream_bf16_token": int(Ab[k][cb]), "shim_bf16_token": int(Sb[k][cb]),
                "upstream_f32_token": int(Af[k][cb]),
                "upstream_bf16_logit_at_upstream_token": float(b[k].reshape(8, 1026)[cb][Ab[k][cb]]),
                "upstream_bf16_logit_at_shim_token": float(b[k].reshape(8, 1026)[cb][Sb[k][cb]]),
                "upstream_f32_logit_at_upstream_token": float(f[k].reshape(8, 1026)[cb][Ab[k][cb]]),
                "upstream_f32_logit_at_shim_token": float(f[k].reshape(8, 1026)[cb][Sb[k][cb]]),
            },
            "tie_measured_on": "the 8-step scores run (same prompt, same pinned conditions); "
                               "the 64-step run diverges at the same (step, codebook)",
            "note": "upstream's own top-2 logits are EXACTLY EQUAL in bfloat16 at this "
                    "position; float32 ranks them apart and agrees with the token the "
                    "SHIM picked.",
        },
        "logits": {"per_step": per_step,
                   "oracle": "upstream bfloat16 vs upstream float32, same prompt, same prefix"},
        "waveform": {"n_samples": int(u32.size), "sample_rate": 24000, "scale": scale,
                     "shim_vs_upstream_f32": rel(ws, u32),
                     "upstream_f32_vs_f64": rel(u32, u64),
                     "shim_vs_upstream_f64": rel(ws, u64),
                     "ratio": rel(ws, u64) / rel(u32, u64),
                     "pearson": float(np.corrcoef(ws, u32)[0, 1]),
                     "bit_identical": int((ws == u32).sum()),
                     "rms_shim": float(np.sqrt((ws ** 2).mean())),
                     "rms_upstream": float(np.sqrt((u32 ** 2).mean()))},
        "torchaudio_stub_calls": S["torchaudio_stub_calls"],
    }
    text = json.dumps(report, indent=1)
    if a.write:
        with open(os.path.join(_HERE, "higgs_e2e.json"), "w") as fh:
            fh.write(text)
        print("wrote pytests/higgs_e2e.json")
    else:
        print(text)
    _ = du32, du64, dsh


if __name__ == "__main__":
    main()
