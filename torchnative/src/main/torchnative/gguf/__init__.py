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

**What the `from_pretrained` door does not do yet.** It dequantises. A model
loaded through it is dense, exactly as upstream's GGUF path is dense -- upstream
refuses `quantization_config` together with `gguf_file`
(`modeling_utils.py`, "You cannot combine Quantization and loading a model from
a GGUF file"). Keeping the file's blocks inside a loaded model is the next step
(docs/graph/GGUF.md §5); until then the door refuses `quantization_config` by
name rather than quietly re-quantising already-quantised weights.

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
    "live is {module}. Put `torchnative/src/main` on PYTHONPATH "
    "(docs/platform/VENDOR.md)."
)

_TORCH_DTYPES = {"float32": "float32", "float16": "float16", "bfloat16": "bfloat16"}


def _torch():
    import torch

    if not hasattr(torch._C, "_quantized_from_blob"):
        raise RuntimeError(_MISSING_SHIM.format(module=getattr(torch, "__file__", "?")))
    return torch


def load_tensor(f, name, keep_blocks=False):
    """One tensor of `f` (a `GGUFFile`) as a `torch` tensor.

    `keep_blocks=True` keeps a block type as a quantised tensor carrying the
    file's bytes unchanged; `_C._quantized_blob(t)` on the result returns them.
    `keep_blocks=False` dequantises through candle and returns `float32`.
    Dense types ignore the flag: they come back in their own dtype, never
    widened, so that a caller who wants `float32` says so.
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
        return torch.frombuffer(raw, dtype=dtype).reshape(info.shape)
    if kind == "block":
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
