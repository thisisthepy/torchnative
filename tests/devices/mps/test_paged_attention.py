"""Paged attention kernel on `cpu` and `mps` (issue #14, SPEC S7.3).

Criteria (issue #14):
1. A paged attention kernel (`torchnative.kernels.paged_attention`) agrees
   element-wise with upstream PyTorch's reference attention over the same
   gathered KV, in a separate subprocess, against a derived tolerance
   (`docs/numerics/AGREE.md`: within 4x upstream's own distance from its
   float64 truth, or exact/dtype-ulp when upstream float32 equals float64).
2. On `mps`, `torch._C._metal_counters()` proves it ran on the device:
   `compute_encoders >= 1` and `host_downloads == 0` (`host_download_bytes == 0`)
   mid-dispatch.
3. Unsupported block sizes, head layouts, ranks, dtypes, devices, or negative
   block indices refuse by name (`UnsupportedPagedAttentionLayout`).
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
from pathlib import Path

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
_VENDOR_DIR = str(_ROOT / "torchnative" / "python")
_VENDOR_SHIM = _ROOT / "torchnative" / "python" / "torch" / "_C.abi3.so"
if not _VENDOR_SHIM.is_file() and ".worktrees" in _ROOT.parts:
    _MAIN_ROOT = Path(*_ROOT.parts[: _ROOT.parts.index(".worktrees")])
    _MAIN_SHIM = _MAIN_ROOT / "torchnative" / "python" / "torch" / "_C.abi3.so"
    if _MAIN_SHIM.is_file():
        _VENDOR_SHIM = _MAIN_SHIM
        _MAIN_VENDOR_DIR = str(_MAIN_ROOT / "torchnative" / "python")
        if _MAIN_VENDOR_DIR not in sys.path:
            sys.path.insert(0, _MAIN_VENDOR_DIR)
_SUPPORT_DIR = str(_ROOT / "tests" / "_support")
if _SUPPORT_DIR not in sys.path:
    sys.path.insert(0, _SUPPORT_DIR)
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

import _skip  # noqa: E402


def _shim_tree_or_skip(what: str) -> bool:
    if not _VENDOR_SHIM.is_file():
        _skip.skip(
            f"   (skipped {what}: no vendored shim at {_VENDOR_SHIM} -- run scripts/vendor/install_shim.sh)"
        )
        return False
    return True


def _shim_pythonpath(existing: str = "") -> str:
    entries = [_VENDOR_DIR]
    if _VENDOR_SHIM.parent.parent != Path(_VENDOR_DIR):
        entries.append(str(_VENDOR_SHIM.parent.parent))
    for p in existing.split(os.pathsep):
        if p and p not in entries:
            entries.append(p)
    return os.pathsep.join(entries)


def _mps_available_in_shim() -> tuple[bool, str]:
    if not _VENDOR_SHIM.is_file():
        return (
            False,
            f"no vendored shim at {_VENDOR_SHIM} -- run scripts/vendor/install_shim.sh",
        )
    env = dict(os.environ)
    env["PYTHONPATH"] = _shim_pythonpath(env.get("PYTHONPATH", ""))
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    code = (
        "import json, torch\n"
        "try:\n"
        "    t = torch.ones(1, device='mps')\n"
        "    print(json.dumps({'ok': True, 'dev': str(t.device)}))\n"
        "except Exception as e:\n"
        "    print(json.dumps({'ok': False, 'why': f'{type(e).__name__}: {e}'}))\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        return False, proc.stderr.strip()[-300:] or "subprocess failed"
    data = json.loads(proc.stdout.strip().splitlines()[-1])
    return bool(data.get("ok")), data.get("why", "")


_UPSTREAM_REF_SCRIPT = r"""
import json, math, sys
import torch
import torch.nn.functional as F

assert not hasattr(torch._C, "_aten_implemented"), "expected upstream torch in oracle subprocess"
req = json.loads(sys.stdin.read())

dtype = getattr(torch, req["dtype"])
q32 = torch.tensor(req["query"], dtype=torch.float32)
kc32 = torch.tensor(req["key_cache"], dtype=torch.float32)
vc32 = torch.tensor(req["value_cache"], dtype=torch.float32)
bt = torch.tensor(req["block_table"], dtype=torch.int64)
ctx = req.get("context_lens")
scale = req.get("scale")
is_causal = bool(req.get("is_causal", False))

def ref_paged(q, kc, vc, calc_dtype):
    q_d = q.to(calc_dtype)
    kc_d = kc.to(calc_dtype)
    vc_d = vc.to(calc_dtype)
    B, H_q, Sq, D = q_d.shape
    num_blocks, block_size, H_kv, _ = kc_d.shape
    max_blocks = bt.shape[1]
    S_kv = max_blocks * block_size
    k_gat = kc_d[bt.reshape(-1)].reshape(B, S_kv, H_kv, D).transpose(1, 2).contiguous()
    v_gat = vc_d[bt.reshape(-1)].reshape(B, S_kv, H_kv, D).transpose(1, 2).contiguous()
    n_rep = H_q // H_kv
    if n_rep > 1:
        k_gat = k_gat[:, :, None, :, :].expand(B, H_kv, n_rep, S_kv, D).reshape(B, H_q, S_kv, D)
        v_gat = v_gat[:, :, None, :, :].expand(B, H_kv, n_rep, S_kv, D).reshape(B, H_q, S_kv, D)
    mask = None
    if ctx is not None or is_causal:
        pos = torch.arange(S_kv, dtype=torch.int64).view(1, 1, 1, S_kv)
        valid = torch.ones((B, 1, Sq, S_kv), dtype=torch.bool)
        if ctx is not None:
            c_t = torch.tensor(ctx, dtype=torch.int64).view(B, 1, 1, 1)
            valid = valid & (pos < c_t)
        if is_causal:
            if ctx is not None:
                c_t = torch.tensor(ctx, dtype=torch.int64).view(B, 1, 1, 1)
                q_idx = torch.arange(Sq, dtype=torch.int64).view(1, 1, Sq, 1)
                q_pos = c_t - Sq + q_idx
            else:
                q_pos = torch.arange(Sq, dtype=torch.int64).view(1, 1, Sq, 1)
            valid = valid & (pos <= q_pos)
        add_mask = torch.zeros((B, 1, Sq, S_kv), dtype=calc_dtype)
        add_mask = add_mask.masked_fill(~valid, float("-inf"))
        mask = add_mask
    out = F.scaled_dot_product_attention(q_d, k_gat, v_gat, attn_mask=mask, scale=scale, is_causal=False)
    return out

with torch.no_grad():
    out_dt = ref_paged(q32, kc32, vc32, dtype)
    out_f64 = ref_paged(q32, kc32, vc32, torch.float64)

print(json.dumps({
    "shape": list(out_dt.shape),
    "dtype": str(out_dt.dtype),
    "up_dt": out_dt.to(torch.float64).flatten().tolist(),
    "up_f64": out_f64.flatten().tolist(),
}))
"""

_SHIM_RUN_SCRIPT = r"""
import json, sys
import torch
from torchnative.kernels import paged_attention

assert hasattr(torch._C, "_aten_implemented"), "expected torchnative shim in subprocess"
req = json.loads(sys.stdin.read())
dev = req["device"]
dtype = getattr(torch, req["dtype"])

q = torch.tensor(req["query"], dtype=dtype, device=dev)
kc = torch.tensor(req["key_cache"], dtype=dtype, device=dev)
vc = torch.tensor(req["value_cache"], dtype=dtype, device=dev)
bt = torch.tensor(req["block_table"], dtype=torch.int64, device=dev)
ctx = req.get("context_lens")
ctx_t = torch.tensor(ctx, dtype=torch.int64, device=dev) if ctx is not None else None
scale = req.get("scale")
is_causal = bool(req.get("is_causal", False))

before = torch._C._metal_counters() if dev == "mps" else None
out = paged_attention(
    q,
    kc,
    vc,
    bt,
    context_lens=ctx_t,
    scale=scale,
    is_causal=is_causal,
)
after = torch._C._metal_counters() if before is not None else None

res = {
    "shape": list(out.shape),
    "dtype": str(out.dtype),
    "device": str(out.device),
    "values": out.cpu().to(torch.float64).flatten().tolist(),
}
if before is not None:
    res["counters"] = {k: after[k] - before[k] for k in before if k != "built"}
print(json.dumps(res))
"""


def _run_upstream(payload: dict) -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("TORCH_C_ARTEFACT", None)
    env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", _UPSTREAM_REF_SCRIPT],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"upstream reference subprocess failed (rc={proc.returncode}):\n{proc.stderr[-3000:]}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _run_shim(payload: dict) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = _shim_pythonpath(env.get("PYTHONPATH", ""))
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _SHIM_RUN_SCRIPT],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"shim subprocess failed (rc={proc.returncode}):\n{proc.stderr[-3000:]}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _make_case(
    *,
    seed: int,
    batch: int,
    q_heads: int,
    kv_heads: int,
    seqlen_q: int,
    head_dim: int,
    num_blocks: int,
    block_size: int,
    block_table: list[list[int]],
    context_lens: list[int] | None,
    dtype: str = "float32",
    is_causal: bool = False,
    scale: float | None = None,
) -> dict:
    rng = random.Random(seed)

    def rand_nested(shape):
        if len(shape) == 1:
            return [rng.uniform(-0.75, 0.75) for _ in range(shape[0])]
        return [rand_nested(shape[1:]) for _ in range(shape[0])]

    return {
        "query": rand_nested([batch, q_heads, seqlen_q, head_dim]),
        "key_cache": rand_nested([num_blocks, block_size, kv_heads, head_dim]),
        "value_cache": rand_nested([num_blocks, block_size, kv_heads, head_dim]),
        "block_table": block_table,
        "context_lens": context_lens,
        "dtype": dtype,
        "is_causal": is_causal,
        "scale": scale,
    }


def _check_agreement(case: dict, device: str) -> dict:
    up = _run_upstream(case)
    got = _run_shim({**case, "device": device})
    assert got["device"].startswith(device), (device, got["device"])
    assert got["shape"] == up["shape"], (got["shape"], up["shape"])
    assert got["dtype"] == up["dtype"], (got["dtype"], up["dtype"])

    up_dt = up["up_dt"]
    up_f64 = up["up_f64"]
    vals = got["values"]
    assert len(vals) == len(up_dt) == len(up_f64)

    worst_upstream = max(abs(a - b) for a, b in zip(up_dt, up_f64))
    worst_shim = max(abs(a - b) for a, b in zip(vals, up_f64))
    # Derived tolerance (docs/numerics/AGREE.md): 4x upstream's own distance
    # from float64 truth, with a 4-ulp floor if upstream happened to land on 0.
    ulp_floor = max(max(abs(x) for x in up_f64), 1.0) * (2.0**-21)
    bound = max(4.0 * worst_upstream, ulp_floor)
    assert worst_shim <= bound, (
        f"paged_attention on {device} ({case['dtype']}) max diff from f64 is "
        f"{worst_shim:.3e} > bound {bound:.3e} (upstream err {worst_upstream:.3e})"
    )
    return got


def test_paged_attention_agrees_with_upstream_reference_on_cpu_and_mps():
    if not _shim_tree_or_skip("paged_attention agreement"):
        return
    cases = [
        # 1. Single-query decode, GQA (q_heads=4, kv_heads=2), partial blocks with context_lens
        _make_case(
            seed=10,
            batch=2,
            q_heads=4,
            kv_heads=2,
            seqlen_q=1,
            head_dim=8,
            num_blocks=5,
            block_size=4,
            block_table=[[3, 1, 0], [4, 2, 1]],
            context_lens=[9, 11],
            dtype="float32",
        ),
        # 2. Multi-query chunked prefill + causal mask, MHA (q_heads=2, kv_heads=2)
        _make_case(
            seed=20,
            batch=2,
            q_heads=2,
            kv_heads=2,
            seqlen_q=3,
            head_dim=16,
            num_blocks=6,
            block_size=4,
            block_table=[[1, 4, 2], [0, 5, 3]],
            context_lens=[7, 10],
            dtype="float32",
            is_causal=True,
        ),
        # 3. Reduced precision (float16 and bfloat16) GQA decode
        _make_case(
            seed=30,
            batch=2,
            q_heads=6,
            kv_heads=2,
            seqlen_q=1,
            head_dim=8,
            num_blocks=4,
            block_size=4,
            block_table=[[2, 0], [1, 3]],
            context_lens=[6, 8],
            dtype="float16",
        ),
        _make_case(
            seed=40,
            batch=2,
            q_heads=6,
            kv_heads=2,
            seqlen_q=1,
            head_dim=8,
            num_blocks=4,
            block_size=4,
            block_table=[[2, 0], [1, 3]],
            context_lens=[6, 8],
            dtype="bfloat16",
        ),
    ]

    for case in cases:
        _check_agreement(case, "cpu")

    mps_ok, why = _mps_available_in_shim()
    if not mps_ok:
        _skip.skip(f"   (skipped mps agreement in test_paged_attention: {why})")
        return

    for case in cases:
        _check_agreement(case, "mps")


def test_paged_attention_on_mps_opens_compute_encoders_and_downloads_nothing():
    mps_ok, why = _mps_available_in_shim()
    if not mps_ok:
        _skip.skip(f"   (skipped mps counters in test_paged_attention: {why})")
        return

    for dtype in ("float32", "float16", "bfloat16"):
        case = _make_case(
            seed=55,
            batch=2,
            q_heads=4,
            kv_heads=2,
            seqlen_q=2,
            head_dim=8,
            num_blocks=4,
            block_size=8,
            block_table=[[2, 0], [1, 3]],
            context_lens=[11, 15],
            dtype=dtype,
            is_causal=True,
        )
        got = _check_agreement(case, "mps")
        counters = got["counters"]
        assert counters["compute_encoders"] >= 1, (
            f"paged_attention ({dtype}) on mps opened no Metal compute encoders: {counters}"
        )
        assert counters["host_downloads"] == 0, (
            f"paged_attention ({dtype}) on mps downloaded {counters['host_downloads']} "
            f"buffer(s) ({counters['host_download_bytes']} bytes) mid-dispatch: {counters}"
        )
        assert counters["host_download_bytes"] == 0, counters


def test_paged_attention_refuses_unsupported_block_or_head_layouts_by_name():
    import torch
    from torchnative.kernels import (
        UnsupportedPagedAttentionLayout,
        paged_attention,
    )

    q = torch.randn(2, 4, 1, 8)
    kc = torch.randn(4, 4, 2, 8)
    vc = torch.randn(4, 4, 2, 8)
    bt = torch.tensor([[0, 1], [2, 3]], dtype=torch.int64)

    def expect_refusal(match: str, fn):
        try:
            fn()
        except UnsupportedPagedAttentionLayout as exc:
            assert match in str(exc), f"expected {match!r} in {exc!r}"
        else:
            raise AssertionError(f"expected UnsupportedPagedAttentionLayout({match!r})")

    # 1. Non-4D query or KV cache refuses by name.
    expect_refusal(
        "4-D",
        lambda: paged_attention(torch.randn(2, 4, 8), kc, vc, bt),
    )
    expect_refusal(
        "4-D",
        lambda: paged_attention(q, torch.randn(4, 2, 8), vc, bt),
    )

    # 2. Mismatched key_cache and value_cache shapes refuse by name.
    expect_refusal(
        "key_cache",
        lambda: paged_attention(q, kc, torch.randn(4, 4, 2, 16), bt),
    )

    # 3. Query head count not divisible by KV head count refuses by name.
    expect_refusal(
        "num_kv_heads",
        lambda: paged_attention(torch.randn(2, 5, 1, 8), kc, vc, bt),
    )

    # 4. Head dimension mismatch between query and KV cache refuses by name.
    expect_refusal(
        "head_dim",
        lambda: paged_attention(torch.randn(2, 4, 1, 16), kc, vc, bt),
    )

    # 5. Non-2D block_table or batch mismatch refuses by name.
    expect_refusal(
        "block_table",
        lambda: paged_attention(q, kc, vc, torch.tensor([0, 1], dtype=torch.int64)),
    )
    expect_refusal(
        "batch",
        lambda: paged_attention(q, kc, vc, torch.tensor([[0, 1]], dtype=torch.int64)),
    )

    # 6. Non-integer block_table dtype refuses by name.
    expect_refusal(
        "block_table",
        lambda: paged_attention(q, kc, vc, torch.tensor([[0.0, 1.0], [2.0, 3.0]])),
    )

    # 7. Unsupported query/KV dtype (int64) refuses by name.
    expect_refusal(
        "dtype",
        lambda: paged_attention(
            torch.ones(2, 4, 1, 8, dtype=torch.int64),
            torch.ones(4, 4, 2, 8, dtype=torch.int64),
            torch.ones(4, 4, 2, 8, dtype=torch.int64),
            bt,
        ),
    )

    # 8. Invalid context_lens (exceeding allocated blocks or <= 0) refuses by name.
    expect_refusal(
        "context_lens",
        lambda: paged_attention(
            q, kc, vc, bt, context_lens=torch.tensor([99, 4], dtype=torch.int64)
        ),
    )


_UPSTREAM_FLASH_REF_SCRIPT = r"""
import json, math, sys
import torch
import torch.nn.functional as F

assert not hasattr(torch._C, "_aten_implemented"), "expected upstream torch in oracle subprocess"
req = json.loads(sys.stdin.read())
op = req["op"]
dtype = getattr(torch, req["dtype"])

q32 = torch.tensor(req["query"], dtype=torch.float32)
k32 = torch.tensor(req["key"], dtype=torch.float32)
v32 = torch.tensor(req["value"], dtype=torch.float32)
scale = req.get("scale")
is_causal = bool(req.get("is_causal", False))

def expand_kv(q, k, v):
    B, H_q, Sq, D = q.shape
    _, H_kv, Sk, Dv = v.shape
    n_rep = H_q // H_kv
    if n_rep > 1:
        k = k[:, :, None, :, :].expand(B, H_kv, n_rep, Sk, D).reshape(B, H_q, Sk, D)
        v = v[:, :, None, :, :].expand(B, H_kv, n_rep, Sk, Dv).reshape(B, H_q, Sk, Dv)
    return k, v

def ref_flash(q, k, v, calc_dtype):
    q_d = q.to(calc_dtype)
    k_d = k.to(calc_dtype)
    v_d = v.to(calc_dtype)
    k_d, v_d = expand_kv(q_d, k_d, v_d)
    mask = None
    if req.get("attn_mask") is not None:
        m = torch.tensor(req["attn_mask"])
        mask = m if m.dtype == torch.bool else m.to(calc_dtype)
    return F.scaled_dot_product_attention(
        q_d, k_d, v_d, attn_mask=mask, dropout_p=0.0, is_causal=is_causal, scale=scale
    )

def ref_linear(q, k, v, calc_dtype):
    # Compute in float32/float64 as the kernel does, or in calc_dtype
    q_d = q.to(calc_dtype).to(torch.float32 if calc_dtype in (torch.float16, torch.bfloat16) else calc_dtype)
    k_d = k.to(calc_dtype).to(q_d.dtype)
    v_d = v.to(calc_dtype).to(q_d.dtype)
    k_d, v_d = expand_kv(q_d, k_d, v_d)
    B, H, S, D = q_d.shape
    Dv = v_d.shape[-1]
    sc = (1.0 / math.sqrt(D)) if scale is None else float(scale)
    q_s = q_d * sc

    decay = req.get("decay")
    beta = req.get("beta")
    init_state = req.get("initial_state")
    ret_state = bool(req.get("return_final_state", False))

    if decay is not None or beta is not None or init_state is not None or ret_state:
        # Recurrent linear / gated-delta rule:
        # state_t = exp(decay_t) * state_{t-1} + k_t^T @ (beta_t * v_t)
        # out_t = q_t @ state_t
        state = (
            torch.tensor(init_state, dtype=calc_dtype).to(q_d.dtype)
            if init_state is not None
            else torch.zeros((B, H, D, Dv), dtype=q_d.dtype)
        )
        dec_t = (
            torch.tensor(decay, dtype=calc_dtype).to(q_d.dtype).reshape(B, H, S, 1, 1)
            if decay is not None
            else None
        )
        bet_t = (
            torch.tensor(beta, dtype=calc_dtype).to(q_d.dtype).reshape(B, H, S, 1)
            if beta is not None
            else None
        )
        outs = []
        for t in range(S):
            if dec_t is not None:
                state = state * torch.exp(dec_t[:, :, t])
            vt = v_d[:, :, t]
            if bet_t is not None:
                vt = vt * bet_t[:, :, t]
            kt = k_d[:, :, t]
            state = state + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            qt = q_s[:, :, t]
            ot = (qt.unsqueeze(-1) * state).sum(dim=-2)
            outs.append(ot)
        out = torch.stack(outs, dim=2).to(calc_dtype)
        return out, state.to(calc_dtype)

    scores = torch.matmul(q_s, k_d.transpose(-2, -1))
    if is_causal:
        idx = torch.arange(S)
        causal = idx.view(S, 1) >= idx.view(1, S)
        scores = torch.where(causal.view(1, 1, S, S), scores, torch.zeros_like(scores))
    out = torch.matmul(scores, v_d).to(calc_dtype)
    return out, None

with torch.no_grad():
    if op == "flash_attention":
        out_dt = ref_flash(q32, k32, v32, dtype)
        out_f64 = ref_flash(q32, k32, v32, torch.float64)
        st_dt = st_f64 = None
    elif op == "flash_linear_attention":
        out_dt, st_dt = ref_linear(q32, k32, v32, dtype)
        out_f64, st_f64 = ref_linear(q32, k32, v32, torch.float64)
    else:
        raise ValueError(op)

res = {
    "shape": list(out_dt.shape),
    "dtype": str(out_dt.dtype),
    "up_dt": out_dt.to(torch.float64).flatten().tolist(),
    "up_f64": out_f64.flatten().tolist(),
}
if st_dt is not None and req.get("return_final_state"):
    res["state_shape"] = list(st_dt.shape)
    res["state_up_dt"] = st_dt.to(torch.float64).flatten().tolist()
    res["state_up_f64"] = st_f64.flatten().tolist()
print(json.dumps(res))
"""

_SHIM_FLASH_SCRIPT = r"""
import json, sys
import torch
from torchnative.kernels import flash_attention, flash_linear_attention

assert hasattr(torch._C, "_aten_implemented"), "expected torchnative shim in subprocess"
req = json.loads(sys.stdin.read())
op = req["op"]
dev = req["device"]
dtype = getattr(torch, req["dtype"])

q = torch.tensor(req["query"], dtype=dtype, device=dev)
k = torch.tensor(req["key"], dtype=dtype, device=dev)
v = torch.tensor(req["value"], dtype=dtype, device=dev)
scale = req.get("scale")
is_causal = bool(req.get("is_causal", False))

mask_t = None
if req.get("attn_mask") is not None:
    m = torch.tensor(req["attn_mask"], device=dev)
    mask_t = m if m.dtype == torch.bool else m.to(dtype)

dec_t = (
    torch.tensor(req["decay"], dtype=dtype, device=dev)
    if req.get("decay") is not None
    else None
)
bet_t = (
    torch.tensor(req["beta"], dtype=dtype, device=dev)
    if req.get("beta") is not None
    else None
)
init_t = (
    torch.tensor(req["initial_state"], dtype=dtype, device=dev)
    if req.get("initial_state") is not None
    else None
)
ret_state = bool(req.get("return_final_state", False))

before = torch._C._metal_counters() if dev == "mps" else None
if op == "flash_attention":
    out = flash_attention(
        q, k, v, attn_mask=mask_t, dropout_p=0.0, is_causal=is_causal, scale=scale
    )
    st = None
elif op == "flash_linear_attention":
    res_out = flash_linear_attention(
        q,
        k,
        v,
        decay=dec_t,
        beta=bet_t,
        scale=scale,
        is_causal=is_causal,
        initial_state=init_t,
        return_final_state=ret_state,
    )
    if ret_state:
        out, st = res_out
    else:
        out, st = res_out, None
else:
    raise ValueError(op)
after = torch._C._metal_counters() if before is not None else None

res = {
    "shape": list(out.shape),
    "dtype": str(out.dtype),
    "device": str(out.device),
    "values": out.cpu().to(torch.float64).flatten().tolist(),
}
if st is not None:
    res["state_shape"] = list(st.shape)
    res["state_values"] = st.cpu().to(torch.float64).flatten().tolist()
if before is not None:
    res["counters"] = {k: after[k] - before[k] for k in before if k != "built"}
print(json.dumps(res))
"""


def _run_flash_upstream(payload: dict) -> dict:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("TORCH_C_ARTEFACT", None)
    env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", _UPSTREAM_FLASH_REF_SCRIPT],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"upstream flash reference subprocess failed (rc={proc.returncode}):\n{proc.stderr[-3000:]}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _run_flash_shim(payload: dict) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = _shim_pythonpath(env.get("PYTHONPATH", ""))
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _SHIM_FLASH_SCRIPT],
        input=json.dumps(payload),
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"shim flash subprocess failed (rc={proc.returncode}):\n{proc.stderr[-3000:]}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _check_flash_agreement(case: dict, device: str) -> dict:
    up = _run_flash_upstream(case)
    got = _run_flash_shim({**case, "device": device})
    assert got["device"].startswith(device), (device, got["device"])
    assert got["shape"] == up["shape"], (got["shape"], up["shape"])
    assert got["dtype"] == up["dtype"], (got["dtype"], up["dtype"])

    up_dt = up["up_dt"]
    up_f64 = up["up_f64"]
    vals = got["values"]
    assert len(vals) == len(up_dt) == len(up_f64)

    worst_upstream = max(abs(a - b) for a, b in zip(up_dt, up_f64))
    worst_shim = max(abs(a - b) for a, b in zip(vals, up_f64))
    ulp_floor = max(max(abs(x) for x in up_f64), 1.0) * (2.0**-21)
    bound = max(4.0 * worst_upstream, ulp_floor)
    assert worst_shim <= bound, (
        f"{case['op']} on {device} ({case['dtype']}) max diff from f64 is "
        f"{worst_shim:.3e} > bound {bound:.3e} (upstream err {worst_upstream:.3e})"
    )
    if case.get("return_final_state"):
        assert got["state_shape"] == up["state_shape"]
        s_up_dt = up["state_up_dt"]
        s_up_f64 = up["state_up_f64"]
        s_vals = got["state_values"]
        s_worst_up = max(abs(a - b) for a, b in zip(s_up_dt, s_up_f64))
        s_worst_shim = max(abs(a - b) for a, b in zip(s_vals, s_up_f64))
        s_floor = max(max(abs(x) for x in s_up_f64), 1.0) * (2.0**-21)
        s_bound = max(4.0 * s_worst_up, s_floor)
        assert s_worst_shim <= s_bound, (
            f"{case['op']} final_state on {device} ({case['dtype']}) max diff "
            f"{s_worst_shim:.3e} > bound {s_bound:.3e}"
        )
    return got


def _rand_nested(rng: random.Random, shape: list[int], lo: float = -0.75, hi: float = 0.75):
    if len(shape) == 1:
        return [rng.uniform(lo, hi) for _ in range(shape[0])]
    return [_rand_nested(rng, shape[1:], lo, hi) for _ in range(shape[0])]


def test_flash_attention_agrees_with_upstream_reference_on_cpu_and_mps():
    if not _shim_tree_or_skip("flash_attention agreement"):
        return
    rng = random.Random(101)
    cases = [
        # 1. Causal GQA prefill (q_heads=4, kv_heads=2, seqlen_q=6, seqlen_k=6), float32
        {
            "op": "flash_attention",
            "query": _rand_nested(rng, [2, 4, 6, 8]),
            "key": _rand_nested(rng, [2, 2, 6, 8]),
            "value": _rand_nested(rng, [2, 2, 6, 8]),
            "dtype": "float32",
            "is_causal": True,
            "scale": None,
        },
        # 2. Non-causal MHA with explicit additive mask and custom scale, float32
        {
            "op": "flash_attention",
            "query": _rand_nested(rng, [2, 2, 3, 16]),
            "key": _rand_nested(rng, [2, 2, 5, 16]),
            "value": _rand_nested(rng, [2, 2, 5, 16]),
            "attn_mask": _rand_nested(rng, [2, 1, 3, 5], -0.5, 0.0),
            "dtype": "float32",
            "is_causal": False,
            "scale": 0.2,
        },
        # 3. Causal GQA in float16 and bfloat16
        {
            "op": "flash_attention",
            "query": _rand_nested(rng, [2, 6, 4, 8]),
            "key": _rand_nested(rng, [2, 2, 4, 8]),
            "value": _rand_nested(rng, [2, 2, 4, 8]),
            "dtype": "float16",
            "is_causal": True,
        },
        {
            "op": "flash_attention",
            "query": _rand_nested(rng, [2, 6, 4, 8]),
            "key": _rand_nested(rng, [2, 2, 4, 8]),
            "value": _rand_nested(rng, [2, 2, 4, 8]),
            "dtype": "bfloat16",
            "is_causal": True,
        },
    ]

    for case in cases:
        _check_flash_agreement(case, "cpu")

    mps_ok, why = _mps_available_in_shim()
    if not mps_ok:
        _skip.skip(f"   (skipped mps agreement in test_flash_attention: {why})")
        return

    for case in cases:
        _check_flash_agreement(case, "mps")


def test_flash_attention_on_mps_opens_compute_encoders_and_downloads_nothing():
    mps_ok, why = _mps_available_in_shim()
    if not mps_ok:
        _skip.skip(f"   (skipped mps counters in test_flash_attention: {why})")
        return
    rng = random.Random(202)
    for dtype in ("float32", "float16", "bfloat16"):
        case = {
            "op": "flash_attention",
            "query": _rand_nested(rng, [2, 4, 5, 8]),
            "key": _rand_nested(rng, [2, 2, 5, 8]),
            "value": _rand_nested(rng, [2, 2, 5, 8]),
            "dtype": dtype,
            "is_causal": True,
        }
        got = _check_flash_agreement(case, "mps")
        counters = got["counters"]
        assert counters["compute_encoders"] >= 1, (
            f"flash_attention ({dtype}) on mps opened no Metal compute encoders: {counters}"
        )
        assert counters["host_downloads"] == 0, (
            f"flash_attention ({dtype}) on mps downloaded {counters['host_downloads']} "
            f"buffer(s) mid-dispatch: {counters}"
        )
        assert counters["host_download_bytes"] == 0, counters


def test_flash_linear_attention_agrees_with_upstream_reference_on_cpu_and_mps():
    if not _shim_tree_or_skip("flash_linear_attention agreement"):
        return
    rng = random.Random(303)
    cases = [
        # 1. Causal chunked linear attention with GQA (q_heads=4, kv_heads=2), float32
        {
            "op": "flash_linear_attention",
            "query": _rand_nested(rng, [2, 4, 6, 8]),
            "key": _rand_nested(rng, [2, 2, 6, 8]),
            "value": _rand_nested(rng, [2, 2, 6, 8]),
            "dtype": "float32",
            "is_causal": True,
        },
        # 2. Gated recurrent linear attention with decay, beta, initial_state, return_final_state
        {
            "op": "flash_linear_attention",
            "query": _rand_nested(rng, [2, 2, 5, 8]),
            "key": _rand_nested(rng, [2, 2, 5, 8]),
            "value": _rand_nested(rng, [2, 2, 5, 8]),
            "decay": _rand_nested(rng, [2, 2, 5], -0.4, -0.05),
            "beta": _rand_nested(rng, [2, 2, 5], 0.1, 0.9),
            "initial_state": _rand_nested(rng, [2, 2, 8, 8], -0.2, 0.2),
            "return_final_state": True,
            "dtype": "float32",
            "is_causal": True,
        },
        # 3. Reduced precision (float16 and bfloat16) gated linear attention
        {
            "op": "flash_linear_attention",
            "query": _rand_nested(rng, [2, 4, 4, 8]),
            "key": _rand_nested(rng, [2, 2, 4, 8]),
            "value": _rand_nested(rng, [2, 2, 4, 8]),
            "decay": _rand_nested(rng, [2, 4, 4], -0.3, -0.05),
            "return_final_state": True,
            "dtype": "float16",
            "is_causal": True,
        },
        {
            "op": "flash_linear_attention",
            "query": _rand_nested(rng, [2, 4, 4, 8]),
            "key": _rand_nested(rng, [2, 2, 4, 8]),
            "value": _rand_nested(rng, [2, 2, 4, 8]),
            "decay": _rand_nested(rng, [2, 4, 4], -0.3, -0.05),
            "return_final_state": True,
            "dtype": "bfloat16",
            "is_causal": True,
        },
    ]

    for case in cases:
        _check_flash_agreement(case, "cpu")

    mps_ok, why = _mps_available_in_shim()
    if not mps_ok:
        _skip.skip(f"   (skipped mps agreement in test_flash_linear_attention: {why})")
        return

    for case in cases:
        got = _check_flash_agreement(case, "mps")
        counters = got["counters"]
        assert counters["compute_encoders"] >= 1, counters
        assert counters["host_downloads"] == 0, counters
        assert counters["host_download_bytes"] == 0, counters


def test_flash_attention_and_linear_attention_refuse_unsupported_layouts_by_name():
    import torch
    from torchnative.kernels import (
        UnsupportedFlashAttentionLayout,
        flash_attention,
        flash_linear_attention,
    )

    q = torch.randn(2, 4, 4, 8)
    k = torch.randn(2, 2, 4, 8)
    v = torch.randn(2, 2, 4, 8)

    def expect_refusal(match: str, fn):
        try:
            fn()
        except UnsupportedFlashAttentionLayout as exc:
            assert match in str(exc), f"expected {match!r} in {exc!r}"
        else:
            raise AssertionError(f"expected UnsupportedFlashAttentionLayout({match!r})")

    # 1. Non-4D query/key/value
    expect_refusal("4-D", lambda: flash_attention(torch.randn(2, 4, 8), k, v))
    expect_refusal("4-D", lambda: flash_linear_attention(torch.randn(2, 4, 8), k, v))

    # 2. Non-divisible GQA head counts
    expect_refusal("num_kv_heads", lambda: flash_attention(torch.randn(2, 5, 4, 8), k, v))
    expect_refusal("num_kv_heads", lambda: flash_linear_attention(torch.randn(2, 5, 4, 8), k, v))

    # 3. Mismatched head_dim between query and key
    expect_refusal("head_dim", lambda: flash_attention(torch.randn(2, 4, 4, 16), k, v))
    expect_refusal("head_dim", lambda: flash_linear_attention(torch.randn(2, 4, 4, 16), k, v))

    # 4. Non-zero dropout_p in flash_attention
    expect_refusal("dropout_p", lambda: flash_attention(q, k, v, dropout_p=0.1))

    # 5. Unsupported dtype (int64)
    expect_refusal(
        "dtype",
        lambda: flash_attention(
            torch.ones(2, 4, 4, 8, dtype=torch.int64),
            torch.ones(2, 2, 4, 8, dtype=torch.int64),
            torch.ones(2, 2, 4, 8, dtype=torch.int64),
        ),
    )

    # 6. Mismatched seqlen in flash_linear_attention or invalid decay shape
    expect_refusal(
        "seqlen",
        lambda: flash_linear_attention(
            q, torch.randn(2, 2, 6, 8), torch.randn(2, 2, 6, 8)
        ),
    )
    expect_refusal(
        "decay",
        lambda: flash_linear_attention(q, k, v, decay=torch.randn(2, 9, 4)),
    )


if __name__ == "__main__":
    items = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    raise SystemExit(1 if _skip.run_tests(items, suite="test_paged_attention") else 0)
