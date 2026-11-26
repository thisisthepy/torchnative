"""The KV rows a CoreML decoder group reads are the cache's filled prefix (#437).

`to(torchnative.device.npu)` replaces Llama-shaped decoder layers with grouped
CoreML programs (`coreml._decoder_group`). Each step feeds the program the
cached key and value rows of every layer in the group, and the program attends
over all of them; its mask hides only the one zero row it prepends.

The group read `cache.layers[i].keys` whole. For a `DynamicCache` that is the
filled prefix. For a `StaticCache`, which `generate` uses by default on a
compiled module (`compile._install_static_cache_default`, owner 2026-11-13,
INTENT §5, #263), it is all `max_cache_len` slots, and the unwritten ones are
zeros the program attended to with score 0. Measured on SmolLM2-135M on an
Apple M-series Mac (2026-11-26, `examples/macos`): position 0 agreed with
upstream (the cache is not yet initialised there), and the prefill's last-row
logit error grew with the cache length, 0.78 at 5 slots, 2.79 at 16, 8.41 at
64, against 0.20 with a `DynamicCache` and 0.0 between the two caches upstream.

What this asserts: `coreml._cached_rows` returns exactly the rows written, for
both cache types, on any host (no coremltools needed), and refuses by name a
layer whose slots no longer hold positions in order (a sliding window that has
wrapped). `test_anegenerate.py` asserts the same thing end to end on the
Neural Engine. What this cannot see: the program's arithmetic.
"""

import os

import _skip
from test_shim import _CKPT_VENDOR_SHIM, _STDOUT_GUARD, _npu_fixture

_SCRIPT = r"""
@STDOUT_GUARD@
import json
import torch
assert hasattr(torch._C, "_aten_implemented")
from transformers import DynamicCache, LlamaConfig, StaticCache
from torchnative.export import coreml

cfg = LlamaConfig(hidden_size=16, num_attention_heads=2, num_key_value_heads=1,
                  num_hidden_layers=1, intermediate_size=32, vocab_size=8,
                  max_position_embeddings=64)
out = {}
for name, make in (("static", lambda: StaticCache(config=cfg, max_cache_len=16)),
                   ("dynamic", lambda: DynamicCache(config=cfg))):
    cache = make()
    before = coreml._cached_rows(cache.layers[0])
    written = []
    for pos in range(3):
        k = torch.full((1, 1, 1, 8), float(pos + 1))
        v = torch.full((1, 1, 1, 8), float(-(pos + 1)))
        cache.update(k, v, 0)
        written.append(pos + 1)
    k, v = coreml._cached_rows(cache.layers[0])
    out[name] = {
        "before": before == (None, None),
        "k_shape": list(k.shape), "v_shape": list(v.shape),
        "k_rows": [float(x) for x in k[0, 0, :, 0].tolist()],
        "v_rows": [float(x) for x in v[0, 0, :, 0].tolist()],
        "written": written,
    }

class Wrapped:
    # A layer whose positions outran its slots: a sliding window that rolled.
    is_initialized = True
    keys = torch.zeros(1, 1, 4, 8)
    values = torch.zeros(1, 1, 4, 8)
    def get_seq_length(self):
        return 6
try:
    coreml._cached_rows(Wrapped())
    out["wrapped"] = "accepted"
except coreml.CoreMLUnsupported as error:
    out["wrapped"] = str(error)
print(json.dumps(out), file=_stdout, flush=True)
"""

_CACHE = {}


def _run():
    if "out" not in _CACHE:
        if not os.path.isfile(_CKPT_VENDOR_SHIM):
            _skip.skip("vendored tree has no _C.abi3.so")
        _CACHE["out"] = _npu_fixture(_SCRIPT.replace("@STDOUT_GUARD@", _STDOUT_GUARD))
    return _CACHE["out"]


def test_a_static_cache_feeds_only_the_rows_written():
    got = _run()["static"]
    assert got["before"], got
    assert got["k_shape"] == [1, 1, 3, 8], got
    assert got["v_shape"] == [1, 1, 3, 8], got
    assert got["k_rows"] == [1.0, 2.0, 3.0], got
    assert got["v_rows"] == [-1.0, -2.0, -3.0], got


def test_a_dynamic_cache_feeds_the_same_rows():
    out = _run()
    assert out["dynamic"]["before"], out["dynamic"]
    for key in ("k_shape", "v_shape", "k_rows", "v_rows"):
        assert out["dynamic"][key] == out["static"][key], (key, out)


def test_a_wrapped_window_refuses_by_name():
    said = _run()["wrapped"]
    assert said != "accepted", said
    assert "6 positions" in said and "4 slots" in said, said


def _main():
    items = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    return 1 if _skip.run_tests(items, suite="test_anecacherows") else 0


if __name__ == "__main__":
    raise SystemExit(_main())
