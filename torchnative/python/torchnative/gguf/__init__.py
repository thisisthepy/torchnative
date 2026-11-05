"""GGUF checkpoints -- the file format Ollama and llama.cpp ship models in.

    from torchnative.transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B-GGUF", gguf_file="Qwen3-0.6B-Q8_0.gguf"
    )

That call is the entry point, and it is upstream's own spelling: the same
`from_pretrained(..., gguf_file=...)` that `transformers` documents. What runs
underneath is upstream's, too -- its GGUF-to-config mapping, its GGUF-to-HF
tensor-name tables, its loader -- except for the one step this stack cannot run
as upstream wrote it, which is the step that turns GGML blocks into tensors
(upstream does it in numpy and crosses with `torch.from_numpy`, which this shim
refuses by name: docs/devices/INTELNPU.md §1.5). docs/graph/GGUF.md is the
design record and §1 says why it is this shape rather than the two
alternatives.

This package is the replacement for that one step, plus the reader under it:

  * `GGUFFile` (`reader.py`) -- the container, in plain Python, no `torch`.
  * `load_tensor(f, name, keep_blocks=...)` -- one tensor:

      - `F32`/`F16`/`BF16` become a dense tensor of that dtype, bit for bit.
      - A block type (`Q4_0` `Q4_1` `Q5_0` `Q5_1` `Q8_0` `Q2_K` .. `Q6_K`)
        becomes, with `keep_blocks=True`, a quantised tensor holding **the
        file's own bytes** (`torch._C._quantized_from_blob`) -- not decoded,
        not re-encoded -- and with `keep_blocks=False` candle's dequantisation
        of those bytes as `float32`.
      - Anything else refuses by name (`reader.GGML_TYPES` says why for each).

  * `quantized_linear(f, name, bias=None)` -- a `torchnative.quant.
    QuantizedLinear` whose weight *is* the GGUF tensor's blocks. That is the
    mapping straight onto torchnative's quantised modules, with no dequantise
    and no re-quantise in between; docs/graph/GGUF.md §2 is the argument that
    it is byte-exact and test_gguf.py the measurement.

**Keeping the blocks inside a loaded model.** By default `from_pretrained`
dequantises so the loaded model is dense, matching upstream's GGUF path. Passing
`keep_blocks=True` or `quantization_config=TorchnativeConfig(...)` replaces
eligible `nn.Linear` leaves with `QuantizedLinear` holding the file's own
per-layer GGUF blocks byte for byte (docs/graph/GGUF.md §5). Non-torchnative
`quantization_config` objects refuse by name rather than quietly re-quantising
already-quantised weights.

Importing this package imports neither `torch` nor `transformers`
(`torchnative/__init__.py` says why that matters); the functions below import
`torch` when called.
"""

from .reader import (
    GGML_TYPES,
    GGUFFile,
    GGUFFormatError,
    GGUFTensorInfo,
    UnsupportedGGUF,
    refusal,
    type_name,
)

__all__ = [
    "GGML_TYPES",
    "GGUFFile",
    "GGUFFormatError",
    "GGUFTensorInfo",
    "UnsupportedGGUF",
    "load_tensor",
    "quantized_linear",
    "type_name",
]

_MISSING_SHIM = (
    "torch._C has no `_quantized_from_blob`: GGUF blocks are handed to "
    "torchnative's `torch._C` shim, not to upstream torch, and the torch that is "
    "live is {module}. Put `torchnative/python` on PYTHONPATH "
    "(docs/platform/VENDOR.md)."
)

_TORCH_DTYPES = {"float32": "float32", "float16": "float16", "bfloat16": "bfloat16"}


def _torch():
    import torch

    if not hasattr(torch._C, "_quantized_from_blob"):
        raise RuntimeError(_MISSING_SHIM.format(module=getattr(torch, "__file__", "?")))
    return torch


def _permute_qk_rows_raw(raw, shape, n_head):
    """Reverse llama.cpp's q/k row permutation directly on raw GGUF row bytes.

    Upstream's `LlamaTensorProcessor._reverse_permute_weights` reshapes dim 0 of
    `weights` from `(n_head, dim, 2, ...)` to `(n_head, 2, dim, ...)` via
    `swapaxes(2, 1)`. Because GGUF blocks run along `shape[-1]`, each of the
    `n_rows = shape[0]` rows is an independent contiguous slice of
    `row_bytes = len(raw) // n_rows` bytes. Rearranging those row slices
    performs the exact permutation without decoding or re-encoding any block.
    """
    n_rows = shape[0]
    if n_head <= 0 or n_rows % (n_head * 2) != 0:
        raise UnsupportedGGUF(
            f"cannot reverse-permute Llama q/k rows of shape {shape} for n_head={n_head}"
        )
    dim = n_rows // n_head // 2
    row_bytes = len(raw) // n_rows
    out = bytearray(len(raw))
    dst_row = 0
    for h in range(n_head):
        for p in range(2):
            for d in range(dim):
                src_row = (h * dim + d) * 2 + p
                s0 = src_row * row_bytes
                d0 = dst_row * row_bytes
                out[d0 : d0 + row_bytes] = raw[s0 : s0 + row_bytes]
                dst_row += 1
    return bytes(out)


def _permute_qk_rows_dense(tensor, n_head):
    """Reverse llama.cpp's q/k row permutation on a dense tensor."""
    n_rows = tensor.shape[0]
    if n_head <= 0 or n_rows % (n_head * 2) != 0:
        raise UnsupportedGGUF(
            f"cannot reverse-permute Llama q/k rows of shape {tuple(tensor.shape)} for n_head={n_head}"
        )
    dim = n_rows // n_head // 2
    return (
        tensor.reshape(n_head, dim, 2, *tensor.shape[1:])
        .transpose(1, 2)
        .reshape(tensor.shape)
        .contiguous()
    )


def load_tensor(f, name, keep_blocks=False, permute_heads=None):
    """One tensor of `f` (a `GGUFFile`) as a `torch` tensor.

    `keep_blocks=True` keeps a block type as a quantised tensor carrying the
    file's bytes unchanged; `_C._quantized_blob(t)` on the result returns them.
    `keep_blocks=False` dequantises through candle and returns `float32`.
    Dense types ignore the flag: they come back in their own dtype, never
    widened, so that a caller who wants `float32` says so.
    When `permute_heads` is given (`LlamaTensorProcessor` on `.attn_q.` /
    `.attn_k.`), reverses llama.cpp's rotary row permutation.
    """
    torch = _torch()
    info = f.info(name)
    why = refusal(info.ggml_type, name)
    if why is not None:
        raise why
    raw = f.raw(name)
    kind = info.kind
    if kind == "dense":
        dtype = getattr(torch, _TORCH_DTYPES[GGML_TYPES[info.ggml_type][4]])
        # `torch.frombuffer` copies on this shim (lib.rs `_frombuffer`), so the
        # result does not alias the mmap and outlives the file.
        t = torch.frombuffer(raw, dtype=dtype).reshape(info.shape)
        if permute_heads is not None:
            t = _permute_qk_rows_dense(t, permute_heads)
        return t
    if kind == "block":
        if permute_heads is not None:
            raw = _permute_qk_rows_raw(raw, info.shape, permute_heads)
        q = torch._C._quantized_from_blob(raw, list(info.shape), info.format)
        if keep_blocks:
            return q
        return torch._C._dequantize(q)
    # `refusal` returned None, so the kind is one of the two above; anything
    # else means GGML_TYPES grew a kind this function was not taught.
    raise AssertionError(f"GGML_TYPES kind {kind!r} for {info.type_name} is not handled")


def quantized_linear(f, name, bias=None):
    """A `QuantizedLinear` whose weight is GGUF tensor `name`'s own blocks.

    `name` must be a 2-D block-type tensor -- `(out_features, in_features)` in
    torch order, which is how llama.cpp stores every linear weight. Its blocks
    run along `in_features`, which is the dimension `QMatMul` contracts, so the
    layout candle reads is the layout llama.cpp wrote and nothing is
    transposed or repacked. A dense tensor is refused by name: wrapping an
    `F32` weight in a "quantised" module would report a compression that did
    not happen.
    """
    from torchnative.quant import QuantizedLinear

    info = f.info(name)
    if info.kind != "block":
        why = refusal(info.ggml_type, name)
        if why is not None:
            raise why
        raise UnsupportedGGUF(
            f"tensor {name!r} is {info.type_name}, not a block-quantised type; "
            "build an nn.Linear from load_tensor() instead of a QuantizedLinear."
        )
    if len(info.shape) != 2:
        raise UnsupportedGGUF(
            f"tensor {name!r} has shape {info.shape}; a linear weight is 2-D"
        )
    out_features, in_features = info.shape
    qweight = load_tensor(f, name, keep_blocks=True)
    return QuantizedLinear(in_features, out_features, qweight, bias, info.format)
