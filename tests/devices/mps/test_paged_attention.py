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
_SUPPORT_DIR = str(_ROOT / "tests" / "_support")
if _SUPPORT_DIR not in sys.path:
    sys.path.insert(0, _SUPPORT_DIR)
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)

import _skip  # noqa: E402


def _mps_available_in_shim() -> tuple[bool, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [_VENDOR_DIR]
        + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
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
    env["PYTHONPATH"] = os.pathsep.join(
        [_VENDOR_DIR]
        + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p and p != _VENDOR_DIR]
    )
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


if __name__ == "__main__":
    items = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    raise SystemExit(1 if _skip.run_tests(items, suite="test_paged_attention") else 0)
