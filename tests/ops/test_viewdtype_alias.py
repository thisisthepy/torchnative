"""`aten.view.dtype` is a view: issue #436.

Upstream's `x.view(dtype)` reinterprets the bytes of `x`'s storage. The result
shares that storage, so a write through either tensor is seen through the
other, and both report one `data_ptr`. Until #436 the shim built the result
from a copy (`to_le_bytes` then `from_le_bytes` in `aten.rs::view_dtype`), so
every value matched at the moment of the call and every later write was lost
on the other side, with no error. transformers 5.18 continuous batching
depends on the aliasing: it allocates the paged cache as bytes, views it in
the cache dtype, and writes through the view (`copy_blocks`, offload restore).

Every case below runs twice, once on the vendored shim and once on upstream,
each in its own process, and the two answers are compared whole. The cases
cover both directions of the write, all three size relations (smaller to
larger, larger to smaller, equal), a sliced input (non-zero storage offset), a
strided input, a view of a view, the life of the storage after the base dies,
upstream's four refusals word for word, and the 5.18 byte-cache pattern.

Nullification this file is meant to catch: `view_dtype` going back to a copy
(every `*_then_read_*` case and every `same_ptr` field turns red).
"""

import json
import os
import subprocess
import sys

from test_shim import _C  # noqa: F401  (the shim must load before the probes)
import _skip

_REPO_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents
                      if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "python")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")


_PROBE_SCRIPT = r"""
import json
import torch

out = {"_marker": "shim" if hasattr(torch._C, "_aten_implemented") else "upstream"}


def desc(t):
    return {"vals": t.reshape(-1).tolist(), "shape": list(t.shape),
            "stride": list(t.stride()), "offset": t.storage_offset(),
            "dtype": str(t.dtype)}


def rec(name, fn):
    try:
        out[name] = {"ok": fn()}
    except Exception as e:
        out[name] = {"raised": type(e).__name__, "msg": str(e)}


ONE = list(torch.tensor([1.0], dtype=torch.float32).view(torch.uint8).tolist())


def u8_to_f32_write_base_then_read_view():
    x = torch.zeros(16, dtype=torch.uint8)
    y = x.view(torch.float32)
    x[4:8] = torch.tensor(ONE, dtype=torch.uint8)
    return {"y": desc(y), "same_ptr": x.data_ptr() == y.data_ptr()}


def u8_to_f32_write_view_then_read_base():
    x = torch.zeros(16, dtype=torch.uint8)
    y = x.view(torch.float32)
    y[2] = 2.5
    y[3].fill_(-1.0)
    return {"x": desc(x)}


def f32_to_u8_write_view_then_read_base():
    x = torch.zeros(3, dtype=torch.float32)
    y = x.view(torch.uint8)
    y[4:8] = torch.tensor(ONE, dtype=torch.uint8)
    return {"x": desc(x), "y": desc(y), "same_ptr": x.data_ptr() == y.data_ptr()}


def f32_to_i32_inplace_add_through_view():
    x = torch.zeros(4, dtype=torch.float32)
    y = x.view(torch.int32)
    y.add_(1)
    x.mul_(2.0)
    return {"x": desc(x), "y": desc(y)}


def i64_to_f64_round_trip():
    x = torch.arange(4, dtype=torch.int64)
    y = x.view(torch.float64)
    y[0] = 1.0
    return {"x": desc(x), "back": desc(y.view(torch.int64))}


def sliced_input():
    x = torch.arange(16, dtype=torch.uint8)
    s = x[4:12]
    y = s.view(torch.float32)
    y.zero_()
    return {"x": desc(x), "y": desc(y), "same_ptr": s.data_ptr() == y.data_ptr()}


def strided_input():
    x = torch.zeros(4, 8, dtype=torch.uint8)
    s = x[:, :4]
    y = s.view(torch.int32)
    y.fill_(0x01020304)
    return {"x": desc(x), "y": desc(y)}


def view_of_view():
    x = torch.zeros(8, dtype=torch.uint8)
    y = x.view(torch.float32).view(torch.int16)
    y[1] = 0x3F80
    return {"x": desc(x), "f": desc(x.view(torch.float32))}


def base_dies_storage_lives():
    x = torch.zeros(8, dtype=torch.uint8)
    y = x.view(torch.float32)
    z = x.view(torch.int32)
    del x
    y[0] = 1.0
    return {"z": desc(z)}


def same_untyped_storage():
    x = torch.zeros(8, dtype=torch.uint8)
    y = x.view(torch.float32)
    return {"storage_ptr": x.untyped_storage().data_ptr() == y.untyped_storage().data_ptr(),
            "nbytes": y.untyped_storage().nbytes()}


def byte_cache_pattern():
    # transformers 5.18 `PagedAttentionCache`: one byte buffer, viewed in the
    # cache dtype, written with index_copy_, read back through a fresh view.
    cache = torch.zeros(2 * 3 * 4 * 4, dtype=torch.uint8)
    kv = cache.view(torch.float32).view(2, 3, 4)
    src = torch.arange(2 * 2 * 4, dtype=torch.float32).reshape(2, 2, 4)
    kv.index_copy_(1, torch.tensor([2, 0]), src)
    again = cache.view(torch.float32).view(2, 3, 4)
    again[1, 1].fill_(9.0)
    return {"kv": desc(kv), "cache_sum": int(cache.to(torch.int64).sum())}


rec("u8_to_f32_write_base_then_read_view", u8_to_f32_write_base_then_read_view)
rec("u8_to_f32_write_view_then_read_base", u8_to_f32_write_view_then_read_base)
rec("f32_to_u8_write_view_then_read_base", f32_to_u8_write_view_then_read_base)
rec("f32_to_i32_inplace_add_through_view", f32_to_i32_inplace_add_through_view)
rec("i64_to_f64_round_trip", i64_to_f64_round_trip)
rec("sliced_input", sliced_input)
rec("strided_input", strided_input)
rec("view_of_view", view_of_view)
rec("base_dies_storage_lives", base_dies_storage_lives)
rec("same_untyped_storage", same_untyped_storage)
rec("byte_cache_pattern", byte_cache_pattern)

# Upstream's refusals.
rec("refuse_zero_dim", lambda: desc(torch.tensor(1, dtype=torch.uint8).view(torch.float32)))
rec("refuse_last_stride", lambda: desc(torch.zeros(4, 4, dtype=torch.uint8).t().view(torch.float32)))
rec("refuse_last_size", lambda: desc(torch.zeros(6, dtype=torch.uint8).view(torch.float32)))
rec("refuse_offset", lambda: desc(torch.zeros(9, dtype=torch.uint8)[1:].view(torch.float32)))
rec("refuse_other_stride", lambda: desc(torch.zeros(3, 6, dtype=torch.uint8)[:, :4].view(torch.float32)))

print(json.dumps(out))
"""

_cache = {}


def _run(side):
    """`side` is `"shim"` or `"upstream"`; `None` when the vendored shim is not
    installed (the skip `test_strided.py` takes)."""
    if side in _cache:
        return _cache[side]
    env = dict(os.environ)
    if side == "shim":
        if not os.path.isfile(_VENDOR_SHIM):
            _cache[side] = None
            return None
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run([sys.executable, "-c", _PROBE_SCRIPT],
                          capture_output=True, text=True, env=env, cwd=_REPO_ROOT)
    assert proc.returncode == 0, f"the {side} probe failed to run:\n{proc.stderr[-3000:]}"
    data = json.loads(proc.stdout)
    assert data["_marker"] == side, (
        f"the {side} probe imported the wrong torch ({data['_marker']})")
    _cache[side] = data
    return data


def _agree(*names):
    shim = _run("shim")
    if shim is None:
        return _skip.skip("the vendored shim is not installed")
    upstream = _run("upstream")
    for name in names:
        assert shim[name] == upstream[name], (name, "shim", shim[name], "upstream", upstream[name])


def test_a_write_through_the_base_is_seen_through_the_view():
    _agree("u8_to_f32_write_base_then_read_view")


def test_a_write_through_the_view_is_seen_through_the_base():
    _agree("u8_to_f32_write_view_then_read_base", "f32_to_u8_write_view_then_read_base",
           "f32_to_i32_inplace_add_through_view", "i64_to_f64_round_trip")


def test_sliced_and_strided_inputs_alias_at_their_offset_and_strides():
    _agree("sliced_input", "strided_input")


def test_a_view_of_a_view_reinterprets_the_same_bytes():
    _agree("view_of_view")


def test_the_storage_outlives_the_base_and_is_one_storage():
    _agree("base_dies_storage_lives", "same_untyped_storage")


def test_the_transformers_518_byte_cache_pattern():
    _agree("byte_cache_pattern")


def test_upstreams_refusals_word_for_word():
    _agree("refuse_zero_dim", "refuse_last_stride", "refuse_last_size", "refuse_offset",
           "refuse_other_stride")
    shim = _run("shim")
    if shim is not None:
        for name in ("refuse_zero_dim", "refuse_last_stride", "refuse_last_size",
                     "refuse_offset", "refuse_other_stride"):
            assert "raised" in shim[name], (name, shim[name])


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_viewdtype_alias")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
