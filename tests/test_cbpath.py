"""Issue #13: transformers continuous batching (`generate_batch`, `paged|sdpa` and
`paged|eager`) on cpu.

The op survey for this path was taken on upstream torch 2.13.0 + transformers
5.15.1 under a `TorchDispatchMode` (issue #13 carries the lists). Every ATen overload
it recorded on cpu already has a kernel here -- and continuous batching still
stopped at its first forward, on two walls the survey structurally cannot see,
because both sit *above* the dispatcher:

  * `torch.index_select(...)` had no `overloads.json` row. The paged cache
    reads every layer through that spelling
    (`generation/continuous_batching/cache.py:427`).
  * `TensorBase.__getitem__` refused an index that mixes an integer or a slice
    with a tensor. Every sampling request reaches one:
    `batch_data["input_ids"][0, logits_indices]`
    (`generation/continuous_batching/model_runner.py:187`).

Each side runs in its own process (the shim through the vendored tree, upstream
through the interpreter's own torch) and the results are compared element-wise,
dtype and shape included (AGENTS.md §16). The refusal test pins the shapes the
new walk does not reproduce to a named refusal rather than an answer (AGENTS.md
§18). Written before the build that would let it pass (AGENTS.md §22): against
the tree built from develop 652c7d7 all five tests fail.
"""

import json
import os
import subprocess
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# run.sh's own override, so the suite can be pointed at another built tree
_VENDOR_DIR = os.environ.get(
    "TORCHNATIVE_VENDOR_DIR", os.path.join(_REPO_ROOT, "torchnative", "python")
)


def _run_side(script: str, shim: bool, timeout: int = 300) -> dict:
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{'shim' if shim else 'upstream'} side failed "
            f"(rc={proc.returncode}):\n{proc.stderr[-4000:]}"
        )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out.pop("is_shim") is shim, "the side imported the wrong torch"
    return out


# ---------------------------------------------------------------------------
# Mixed basic + advanced indexing
# ---------------------------------------------------------------------------

_GETITEM_SCRIPT = r"""
import json, torch
out = {"is_shim": hasattr(torch._C, "_aten_implemented")}
x = torch.arange(2 * 5 * 6 * 7, dtype=torch.float32).reshape(2, 5, 6, 7)
y = torch.arange(2 * 5 * 6 * 7 * 3).reshape(2, 5, 6, 7, 3)
t = torch.tensor([3, 0, 4])
u = torch.tensor([1, 5, 2])
z = torch.tensor(1)
m = torch.tensor([True, False, True, True, False])
m2 = torch.zeros(5, 6, dtype=torch.bool)
m2[1, 2] = True
m2[3, 0] = True
m2[4, 5] = True
ids = torch.tensor([[11, 12, 13, 14, 15, 16, 17]], dtype=torch.int32)
li = torch.tensor([6, 0, 3], dtype=torch.int32)
cases = {
    # the continuous-batching call site itself: int32 ids, int32 positions
    "input_ids[0, logits_indices]": lambda: ids[0, li],
    "x[0, t]": lambda: x[0, t],
    "x[1, :, t]": lambda: x[1, :, t],
    "x[:, t, 0]": lambda: x[:, t, 0],
    # NumPy would give (3, 5, 7) here; upstream selects first and gives (5, 3, 7)
    "x[0, :, u]": lambda: x[0, :, u],
    "x[-1, t]": lambda: x[-1, t],
    "x[1:, t]": lambda: x[1:, t],
    "x[:, 1:4, u]": lambda: x[:, 1:4, u],
    "x[::2, t]": lambda: x[::2, t],
    "x[0, None, t]": lambda: x[0, None, t],
    "x[None, 0, t]": lambda: x[None, 0, t],
    "x[0, ..., u]": lambda: x[0, ..., u],
    "x[0, [1, 2]]": lambda: x[0, [1, 2]],
    "x[0, t, 1, u]": lambda: x[0, t, 1, u],
    "x[z, t]": lambda: x[z, t],
    "x[0, m]": lambda: x[0, m],
    "x[0, t[:, None]]": lambda: x[0, t[:, None]],
    "x[1, -2:, t]": lambda: x[1, -2:, t],
    "x[0, t, ..., 3]": lambda: x[0, t, ..., 3],
    # a 2-D mask advances the walk by two: the trailing int selects dim 2
    "y[0, m2, 1]": lambda: y[0, m2, 1],
    "y[0, m2, :, 1]": lambda: y[0, m2, :, 1],
    # ...and an ellipsis after it must count it as two
    "y[0, m2, ..., 2]": lambda: y[0, m2, ..., 2],
}
for key, fn in cases.items():
    r = fn()
    out[key] = [str(r.dtype), list(r.shape), r.reshape(-1).tolist()]
print(json.dumps(out))
"""


def test_mixed_basic_and_advanced_getitem_matches_upstream_element_wise():
    shim = _run_side(_GETITEM_SCRIPT, shim=True)
    upstream = _run_side(_GETITEM_SCRIPT, shim=False)
    assert set(shim) == set(upstream), (sorted(shim), sorted(upstream))
    wrong = {k: (shim[k][:2], upstream[k][:2]) for k in upstream if shim[k] != upstream[k]}
    assert not wrong, f"shim vs upstream (dtype, shape) where values differ: {wrong}"


_GETITEM_REFUSAL_SCRIPT = r"""
import json, torch
out = {"is_shim": hasattr(torch._C, "_aten_implemented")}
x = torch.arange(2 * 5 * 6 * 7, dtype=torch.float32).reshape(2, 5, 6, 7)
y = torch.arange(2 * 5 * 6 * 7 * 3).reshape(2, 5, 6, 7, 3)
t = torch.tensor([3, 0, 4])
m2 = torch.zeros(5, 6, dtype=torch.bool)
m2[1, 2] = True
for key, fn in {
    "python bool": lambda: x[0, t, True],
    "tensor after a 2-D mask": lambda: y[0, m2, t],
}.items():
    try:
        fn()
        out[key] = None
    except Exception as e:
        out[key] = [type(e).__name__, str(e)]
print(json.dumps(out))
"""


def test_mixed_getitem_refuses_the_unmeasured_shapes_by_name():
    # Upstream answers both of these; the shim does not reproduce them yet, and
    # must say which construct it is refusing rather than return something.
    shim = _run_side(_GETITEM_REFUSAL_SCRIPT, shim=True)
    expected = {
        "python bool": "Python bool index mixed with other indices",
        "tensor after a 2-D mask": "after a bool/uint8 mask of rank > 1",
    }
    for key, phrase in expected.items():
        got = shim[key]
        assert got is not None, f"{key}: computed instead of refusing"
        kind, message = got
        assert kind == "NotImplementedError", (key, kind, message)
        assert "not implemented in torch._C shim" in message, (key, message)
        assert phrase in message, (key, message)


# ---------------------------------------------------------------------------
# torch.index_select
# ---------------------------------------------------------------------------

_INDEX_SELECT_SCRIPT = r"""
import json, torch
out = {"is_shim": hasattr(torch._C, "_aten_implemented")}
# the paged cache's own shape: [num_pages, kv_heads, head_dim], int64 read index
cache = torch.arange(10 * 2 * 4, dtype=torch.float32).reshape(10, 2, 4)
read_index = torch.tensor([7, 2, 2, 9, 0])
r = torch.index_select(cache, 0, read_index)
out["dim0"] = [str(r.dtype), list(r.shape), r.reshape(-1).tolist()]
r = torch.index_select(cache, 2, torch.tensor([3, 0], dtype=torch.int32))
out["dim2_int32_index"] = [str(r.dtype), list(r.shape), r.reshape(-1).tolist()]
r = torch.index_select(cache.to(torch.bfloat16), 0, read_index)
out["bfloat16"] = [str(r.dtype), list(r.shape), r.float().reshape(-1).tolist()]
print(json.dumps(out))
"""


def test_torch_index_select_free_function_matches_upstream_element_wise():
    shim = _run_side(_INDEX_SELECT_SCRIPT, shim=True)
    upstream = _run_side(_INDEX_SELECT_SCRIPT, shim=False)
    assert shim == upstream, {k: (shim.get(k), upstream[k]) for k in upstream if shim.get(k) != upstream[k]}


# ---------------------------------------------------------------------------
# The path itself, end to end
# ---------------------------------------------------------------------------

_GENERATE_BATCH_SCRIPT = r"""
import json, random, sys
import torch
import transformers
from transformers import GenerationConfig
from transformers.generation.configuration_utils import ContinuousBatchingConfig
import transformers.generation.continuous_batching.input_outputs as cbio
import transformers.generation.continuous_batching.continuous_api as cbapi

IS_SHIM = hasattr(torch._C, "_aten_implemented")

# Upstream only: on a host where mps is available, upstream CB pins its cpu IO
# buffers (`len(get_available_devices()) > 1`), and upstream 2.13.0 answers
# `torch.zeros(..., device="cpu", pin_memory=True)` with an *mps* tensor -- so its
# own cpu forward then fails (issue #30, measured). That is upstream not running
# a cpu model on a Mac, not a property under test; forcing the unpinned branch is
# what lets the *reference* run at all. The shim gets no such help: it accepts
# `pin_memory=True` on cpu, serves an ordinary cpu tensor, and its
# `torch.mps.recommended_max_memory()` and two siblings answer from Metal
# (`test_cbwalls.py` pins each by itself). Both of #13's stubs are gone from this
# side, so `generate_batch` below runs on the shim's own answers.
if not IS_SHIM:
    cbio.get_available_devices = lambda: ["cpu"]

errors = []
_crit = cbapi.ContinuousBatchingManager._handle_critical_error
def _record(self, error, bp):
    errors.append(f"{type(error).__name__}: {error}")
    return _crit(self, error, bp)
cbapi.ContinuousBatchingManager._handle_critical_error = _record

attn, sampling, arch, dtype_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
common = dict(
    vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, head_dim=16,
    max_position_embeddings=256, tie_word_embeddings=False, attn_implementation=attn,
)
if arch == "qwen3":
    model = transformers.Qwen3ForCausalLM(transformers.Qwen3Config(**common))
else:
    model = transformers.LlamaForCausalLM(transformers.LlamaConfig(**common))
# Weights from Python's RNG: the two builds' torch RNGs differ, the model must not.
rng = random.Random(1234)
with torch.no_grad():
    for name, p in sorted(model.named_parameters()):
        vals = [rng.uniform(-0.2, 0.2) for _ in range(p.numel())]
        if name.endswith("norm.weight"):
            vals = [1.0 + v for v in vals]
        p.copy_(torch.tensor(vals, dtype=torch.float32).reshape(p.shape))
model = model.to(getattr(torch, dtype_name))
model.eval()

prng = random.Random(1)
# three requests of different lengths; block_size 16 and max_batch_tokens 32 make
# the 41-token prompt span three blocks and be prefilled in chunks beside the
# other two requests' decode steps
prompts = [[prng.randrange(3, 256) for _ in range(n)] for n in (5, 23, 41)]
gen = dict(max_new_tokens=6, eos_token_id=-1, pad_token_id=0)
extra = {}
if sampling == "greedy":
    gen.update(do_sample=False)
else:
    # Gemstone's stated sampler configuration; values are random, so only the
    # completion and the token count are compared for it
    gen.update(do_sample=True, temperature=0.7, top_k=20, top_p=0.9)
    extra = dict(per_request_processors=True, return_logprobs=True, allow_block_sharing=False)
cbc = ContinuousBatchingConfig(block_size=16, num_blocks=24, max_batch_tokens=32,
                               use_cuda_graph=False, **extra)
res = model.generate_batch(inputs=prompts, generation_config=GenerationConfig(**gen),
                           continuous_batching_config=cbc, progress_bar=False)
print(json.dumps({"is_shim": IS_SHIM, "errors": errors,
                  "tokens": [list(map(int, r.generated_tokens)) for r in res.values()]}))
"""


def _generate_batch(attn: str, sampling: str, shim: bool, arch: str = "qwen3",
                    dtype: str = "float32") -> dict:
    marker = "attn, sampling, arch, dtype_name = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]"
    assert marker in _GENERATE_BATCH_SCRIPT
    script = _GENERATE_BATCH_SCRIPT.replace(
        marker, f"attn, sampling, arch, dtype_name = {attn!r}, {sampling!r}, {arch!r}, {dtype!r}"
    )
    return _run_side(script, shim=shim, timeout=600)


def test_generate_batch_greedy_tokens_match_upstream_on_cpu_for_both_paged_attentions():
    # #13's configs, 2 architectures x 2 attentions x 2 dtypes, each cell its own
    # pair of subprocesses. The shim side runs with no stub (see the script).
    for arch in ("qwen3", "llama"):
        for attn in ("paged|sdpa", "paged|eager"):
            for dtype in ("float32", "bfloat16"):
                cell = (arch, attn, dtype)
                shim = _generate_batch(attn, "greedy", shim=True, arch=arch, dtype=dtype)
                upstream = _generate_batch(attn, "greedy", shim=False, arch=arch, dtype=dtype)
                assert not upstream["errors"], (cell, "upstream", upstream["errors"])
                assert not shim["errors"], (cell, "shim", shim["errors"])
                assert len(upstream["tokens"]) == 3 and all(
                    len(t) == 6 for t in upstream["tokens"]
                ), (cell, upstream)
                assert shim["tokens"] == upstream["tokens"], (
                    cell, shim["tokens"], upstream["tokens"])


def test_generate_batch_with_per_request_sampling_completes_on_cpu():
    shim = _generate_batch("paged|sdpa", "sample", shim=True)
    assert not shim["errors"], shim["errors"]
    assert len(shim["tokens"]) == 3 and all(len(t) == 6 for t in shim["tokens"]), shim["tokens"]
    assert all(0 <= tok < 256 for t in shim["tokens"] for tok in t), shim["tokens"]


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
