"""`.to(device.npu)` on Android through ExecuTorch's XNNPACK delegate, in memory (issue #423, step R5a).

Owner's direction (docs/devices/ETDEVICE.md section 2): ExecuTorch run on the device
at `.to(device.npu)` time, JIT, programs in memory only, never a `.pte` file; XNNPACK
first; the sequence length symbolic. The shape tested here is the agent proposal
(i) to (iv) of that section, relayed as standing by the leader (2026-11-21): the
resolution says "CPU (ExecuTorch XNNPACK delegate)"; the same module comes back with
its forward swapped; the first call captures, lowers through the vendored exir,
serialises in memory through the Rust serializer and loads into `executorch_rt`; R5a
is the forward without a KV cache. Tested on the macOS arm64 gate, where the runtime
is built, with `TORCHNATIVE_DEVICE_HOST=android`.

**Grade (AGENTS section 16): agrees.** A tiny Qwen3 (weights set by formula, the same
on both sides) is run by upstream torch and by the shim, each in its own subprocess.
The bound is docs/numerics/AGREE.md section 2, derived for this model as
`tests/models/test_qwen3.py` derives it: upstream re-run in float64, the p90 of its
float32 error relative to the logits' scale floored at 8 ulp, or 4x upstream's own
worst distance from float64. Greedy `generate(use_cache=False)` must give upstream's
tokens exactly.

**What makes each test able to fail.**

* `test_android_resolves_to_executorch_where_the_runtime_is_built`: on macOS arm64
  the resolution must be `executorch` with the exact CPU unit (a build without the
  runtime, the XNNPACK backend, a serializer or the vendored exir fails here, never
  skips). Elsewhere `runtime_refusal()` must name what is missing and the composed
  refusal must carry it.
* `test_a_linear_runs_in_the_xnnpack_delegate`: the program's method must count one
  `XnnpackBackend` delegate call per execution and no portable `addmm`/`relu`; a
  lowering that left the graph on portable kernels, or a forward that ran eagerly,
  fails on the counters, whatever the values.
* `test_qwen3_logits_agree_with_upstream_at_two_lengths_from_one_program`: logits at
  lengths 7 and 11 against upstream within the derived bound, and both calls served
  by one compiled program (the symbolic length); a second compile fails.
* `test_greedy_generate_without_cache_gives_upstreams_tokens`: token for token, and
  the steps of one generate may not each compile (at most two programs for six
  steps).
* `test_a_length_past_the_range_compiles_once_more_with_a_wider_range`: a 20-token
  call with `max_seq_len=16` compiles exactly one more program, whose range is 32.
* `test_what_r5a_does_not_do_is_refused_by_name`: `use_cache=True` and an option the
  backend does not take refuse by name; `.to()` returns the same object.

**What this suite cannot find.** Whether a real Android device runs it (R5-D, the
Galaxy Tab, #21). Real checkpoints (the tiny model has the real architecture, not
its size). The KV cache path (R5b). Whether XNNPACK used SIMD or threads.
"""

import json
import os
import pathlib
import platform
import subprocess
import sys

_TESTS_DIR = pathlib.Path(__file__).resolve().parents[2]
_SUPPORT_DIR = _TESTS_DIR / "_support"
if str(_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPPORT_DIR))
_REPO_ROOT = _TESTS_DIR.parent
_VENDOR_DIR = str(_REPO_ROOT / "torchnative" / "python")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
# The vendored tree has no lib/libtorch_global_deps; upstream's __init__ skips
# it only with this set (as test_devicens.py does; run 44973431737).
os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import _skip  # noqa: E402

UNIT = "CPU (ExecuTorch XNNPACK delegate)"
_EPS32 = 1.1920929e-07
_ORACLE_FACTOR = 4.0
_MARK = "ETNPU_JSON "
_cache = {}


def _on_macos_arm64():
    return sys.platform == "darwin" and platform.machine() == "arm64"


#: Run by upstream torch (no PYTHONPATH) and by the shim (vendored tree first).
_SIDE = r'''
import json, math, os, sys
args = json.loads(sys.argv[1])
out = {"shim": args["shim"]}
import torch
out["is_shim"] = hasattr(torch._C, "_aten_implemented")
from transformers import Qwen3Config, Qwen3ForCausalLM

dtype = getattr(torch, args["dtype"])
cfg = Qwen3Config(vocab_size=128, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                  max_position_embeddings=128, tie_word_embeddings=True, rope_theta=1e6,
                  bos_token_id=1, eos_token_id=None, pad_token_id=0)
cfg._attn_implementation = "sdpa"
model = Qwen3ForCausalLM(cfg).eval()
with torch.no_grad():
    for k, (name, p) in enumerate(sorted(model.named_parameters())):
        n = p.numel()
        w = torch.sin(torch.arange(n, dtype=torch.float64) * 0.37 + k)
        if name.endswith("norm.weight"):
            w = 1.0 + 0.1 * w
        else:
            w = w * (1.0 / math.sqrt(max(p.shape[-1], 1)))
        p.copy_(w.reshape(p.shape).to(p.dtype))
model = model.to(dtype)

def ids(n):
    return torch.tensor([[(5 * i + 3) % 120 + 2 for i in range(n)]], dtype=torch.long)

def logits(n):
    x = ids(n)
    with torch.no_grad():
        return model(input_ids=x, attention_mask=torch.ones_like(x)).logits

def gen():
    x = ids(7)
    with torch.no_grad():
        y = model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=6,
                           do_sample=False, use_cache=False)
    return y[0].tolist()

if args["shim"]:
    import torchnative.device as D
    from torchnative.export import et_device
    os.environ["TORCHNATIVE_DEVICE_HOST"] = "android"
    res = D.npu.resolve()
    out["resolution"] = res.as_dict()
    # A Linear first: the counters are the evidence of where it ran.
    lin = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU()).eval()
    with torch.no_grad():
        lin[0].weight.copy_(torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32 - 0.5)
        lin[0].bias.copy_(torch.arange(4, dtype=torch.float32) / 4)
        xin = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16 - 0.5
        want = lin(xin)
    back = lin.to(D.npu)
    out["linear_same_object"] = back is lin
    out["linear_compiles_before_call"] = lin.torchnative_offload["compiles"]
    with torch.no_grad():
        got = lin(xin)
        got2 = lin(xin)
    out["linear_max_abs_diff"] = float((got - want).abs().max())
    out["linear_report"] = lin.torchnative_offload
    out["linear_counters"] = lin.forward.counters()

    back = model.to(D.npu, max_seq_len=16)
    out["same_object"] = back is model
    out["unit"] = model.torchnative_offload["unit"]
    out["logits7"] = logits(7).float().tolist()
    out["logits11"] = logits(11).float().tolist()
    out["compiles_after_forwards"] = model.torchnative_offload["compiles"]
    out["tokens"] = gen()
    out["compiles_after_generate"] = model.torchnative_offload["compiles"]
    logits(20)
    out["compiles_after_long"] = model.torchnative_offload["compiles"]
    out["programs"] = model.torchnative_offload["programs"]
    out["counters"] = model.forward.counters()
    refusals = {}
    try:
        x = ids(7)
        model.generate(input_ids=x, attention_mask=torch.ones_like(x), max_new_tokens=2,
                       do_sample=False, use_cache=True)
    except et_device.ExecuTorchUnsupported as exc:
        refusals["use_cache"] = str(exc)
    other = Qwen3ForCausalLM(cfg).eval()
    try:
        other.to(D.npu, precision="float32")
    except TypeError as exc:
        refusals["precision"] = str(exc)
    out["refusals"] = refusals
else:
    out["logits7"] = logits(7).float().tolist() if dtype != torch.float64 else logits(7).tolist()
    out["logits11"] = logits(11).float().tolist() if dtype != torch.float64 else logits(11).tolist()
    if dtype == torch.float32:
        out["tokens"] = gen()
print("ETNPU_JSON " + json.dumps(out))
'''


def _side(shim, dtype):
    key = (shim, dtype)
    if key in _cache:
        return _cache[key]
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    scratch = _REPO_ROOT / ".scratch"
    scratch.mkdir(exist_ok=True)
    env["TMPDIR"] = str(scratch)
    env.pop("TORCHNATIVE_DEVICE_HOST", None)
    if shim:
        env["PYTHONPATH"] = os.pathsep.join([_VENDOR_DIR, str(_SUPPORT_DIR)])
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run([sys.executable, "-c", _SIDE, json.dumps({"shim": shim, "dtype": dtype})],
                          capture_output=True, text=True, env=env, timeout=1800)
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith(_MARK)), None)
    if proc.returncode != 0 or line is None:
        raise AssertionError(f"{'shim' if shim else 'upstream'} {dtype} side exited {proc.returncode}; "
                             f"stderr tail:\n" + "\n".join(proc.stderr.splitlines()[-30:]))
    rec = json.loads(line[len(_MARK):])
    assert rec["is_shim"] == shim, f"the {'shim' if shim else 'upstream'} side imported the wrong torch"
    _cache[key] = rec
    return rec


def _need_runtime():
    if not _on_macos_arm64():
        from torchnative.export import et_device  # noqa: F401 - importable everywhere

        _skip.skip("the ExecuTorch runtime is built on macOS arm64 and Android only "
                   "(torch_c/Cargo.toml); the refusal elsewhere is "
                   "test_android_resolves_to_executorch_where_the_runtime_is_built")
        return False
    if not os.path.isfile(os.path.join(_VENDOR_DIR, "torch", "__init__.py")):
        _skip.skip("no vendored torch tree")
        return False
    return True


def _flat(x):
    if isinstance(x, list):
        for y in x:
            yield from _flat(y)
    else:
        yield float(x)


def _bound(up, up64):
    """docs/numerics/AGREE.md section 2, as tests/models/test_qwen3.py derives it."""
    scale = max(abs(v) for v in up64)
    assert scale > 0, "degenerate output: every logit is zero"
    own = sorted(abs(a - b) / scale for a, b in zip(up, up64))
    p90 = own[int(0.9 * (len(own) - 1))]
    tol = max(p90, 8 * _EPS32)
    return max(tol, _ORACLE_FACTOR * own[-1]) * scale, tol, own[-1], scale


def test_android_resolves_to_executorch_where_the_runtime_is_built():
    import torch  # noqa: F401 - the shim, from _VENDOR_DIR

    import torchnative.device as D
    from torchnative.export import et_device

    old = os.environ.get("TORCHNATIVE_DEVICE_HOST")
    os.environ["TORCHNATIVE_DEVICE_HOST"] = "android"
    try:
        assert D.npu_candidates()[0] == ("executorch", UNIT), D.npu_candidates()
        why = et_device.runtime_refusal()
        try:
            res = D.npu.resolve()
        except D.NpuUnresolved as exc:
            res, err = None, str(exc)
    finally:
        if old is None:
            os.environ.pop("TORCHNATIVE_DEVICE_HOST", None)
        else:
            os.environ["TORCHNATIVE_DEVICE_HOST"] = old
    if _on_macos_arm64():
        assert why is None, f"macOS arm64 builds the runtime, yet: {why}"
        assert res is not None and res.backend == "executorch" and res.unit == UNIT, res
        assert res.detail["delegate"] == "XnnpackBackend", res.detail
        print(f"ok   etnpu: android resolves to {res.unit} via {res.source}")
        return
    assert why, "runtime_refusal() answered None on a platform that does not build the runtime"
    if res is None:
        assert why in err, (why, err)
    else:
        assert res.backend != "executorch", res
    print(f"ok   etnpu: no runtime here, executorch refuses by name: {why}")


def test_a_linear_runs_in_the_xnnpack_delegate():
    if not _need_runtime():
        return
    rec = _side(True, "float32")
    assert rec["linear_same_object"], "to(npu) did not return the same module"
    assert rec["linear_compiles_before_call"] == 0, "lowering happened at .to(), not at the first call"
    rep = rec["linear_report"]
    assert rep["compiles"] == 1 and rep["fully_delegated"], rep
    assert rec["linear_max_abs_diff"] <= 1e-6, rec["linear_max_abs_diff"]
    (c,) = rec["linear_counters"]
    assert c["executions"] == 2, c
    assert c["delegate_calls"] == {"XnnpackBackend": 2}, c
    portable = {k: v for k, v in c["operator_calls"].items() if "addmm" in k or "relu" in k or "linear" in k}
    assert not portable, f"the Linear ran on portable kernels: {portable}"
    assert "XnnpackBackend" in rep["programs"][0]["method_backends"], rep["programs"][0]
    print(f"ok   etnpu: Linear+ReLU ran in the delegate ({c['delegate_calls']}), "
          f"max |diff| {rec['linear_max_abs_diff']:.3g}")


def _check_logits(name):
    shim, up, up64 = _side(True, "float32"), _side(False, "float32"), _side(False, "float64")
    a, u, u64 = list(_flat(shim[name])), list(_flat(up[name])), list(_flat(up64[name]))
    assert len(a) == len(u) == len(u64), (len(a), len(u), len(u64))
    bound, tol, own, scale = _bound(u, u64)
    dist = max(abs(x - y) for x, y in zip(a, u))
    assert dist <= bound, (f"{name}: ExecuTorch logits differ from upstream by {dist:.3e}; the derived "
                           f"bound is {bound:.3e} (p90 {tol:.3e}, upstream's own worst {own:.3e}, "
                           f"scale {scale:.3e})")
    return dist, bound


def test_qwen3_logits_agree_with_upstream_at_two_lengths_from_one_program():
    if not _need_runtime():
        return
    rec = _side(True, "float32")
    assert rec["same_object"] and rec["unit"] == UNIT, rec
    d7, b7 = _check_logits("logits7")
    d11, b11 = _check_logits("logits11")
    assert rec["compiles_after_forwards"] == 1, (
        f"lengths 7 and 11 took {rec['compiles_after_forwards']} programs; the sequence length "
        f"is symbolic, so one: {rec['programs'][:2]}")
    first = rec["programs"][0]
    assert first["shapes"] == "symbolic-seq", first
    assert first["delegated_subgraphs"] >= 1, first
    print(f"ok   etnpu: qwen3 logits agree at 7 ({d7:.2e} <= {b7:.2e}) and 11 ({d11:.2e} <= {b11:.2e}) "
          f"from one program; left outside the delegate: {first['left_outside_delegate']}")


def test_greedy_generate_without_cache_gives_upstreams_tokens():
    if not _need_runtime():
        return
    rec, up = _side(True, "float32"), _side(False, "float32")
    assert rec["tokens"] == up["tokens"], (rec["tokens"], up["tokens"])
    grew = rec["compiles_after_generate"] - rec["compiles_after_forwards"]
    assert grew <= 2, f"six generate steps compiled {grew} programs"
    print(f"ok   etnpu: generate(use_cache=False) gives upstream's tokens {rec['tokens'][7:]} "
          f"with {grew} more program(s)")


def test_a_length_past_the_range_compiles_once_more_with_a_wider_range():
    if not _need_runtime():
        return
    rec = _side(True, "float32")
    assert rec["compiles_after_long"] == rec["compiles_after_generate"] + 1, rec["programs"]
    last = rec["programs"][-1]
    assert last["shapes"] == "symbolic-seq" and last["max_seq_len"] == 32, last
    print("ok   etnpu: a 20-token call past max_seq_len=16 compiled one more program, range 32")


def test_what_r5a_does_not_do_is_refused_by_name():
    if not _need_runtime():
        return
    rec = _side(True, "float32")
    ref = rec["refusals"]
    assert "use_cache" in ref and "R5b" in ref["use_cache"] and "use_cache=False" in ref["use_cache"], ref
    assert "precision" in ref and "executorch" in ref["precision"], ref
    print("ok   etnpu: use_cache=True and precision= refuse by name")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        _skip._state["current"] = name
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            if name in _skip._state["skipped"]:
                print(f"SKIP test_etnpu: {name} -- {_skip._state['skipped'][name]}")
        _skip._state["current"] = None
    raise SystemExit(1 if failures else 0)
