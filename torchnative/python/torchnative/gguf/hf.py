"""`from_pretrained(..., gguf_file=...)` on this stack.

Reached from `torchnative.transformers.<Auto*>.from_pretrained` when the
caller passes `gguf_file=`; nothing else imports it. It imports `transformers`
at module scope, which is why it is a separate module from the reader
(`torchnative/gguf/__init__.py` must stay cheap to import).

**What upstream does, step by step, and which step is replaced.**
`PreTrainedModel.from_pretrained` with `gguf_file` (transformers 5.15.1,
`modeling_utils.py`) does four things:

  1. reads the GGUF metadata into a config -- `AutoConfig.from_pretrained(...,
     gguf_file=...)` -> `load_gguf_checkpoint(return_tensors=False)`;
  2. builds the model on `meta` and asks `get_gguf_hf_weights_map` which GGUF
     tensor is which parameter;
  3. reads every tensor with the `gguf` package, dequantises it **in numpy**
     (`gguf.quants.dequantize`), passes it through the architecture's
     `TensorProcessor`, and crosses into torch with `torch.from_numpy`;
  4. hands that state dict to the ordinary loader.

Steps 1, 2 and 4 are upstream's code, called here unchanged. Step 3 cannot run
on this shim as written -- `torch.from_numpy` refuses by name
(docs/devices/INTELNPU.md §1.5, asserted in test_intelnpu.py) -- so it is
replaced: the blocks go to candle through `torchnative.gguf.load_tensor`, and
candle dequantises them. test_gguf.py holds that replacement to **bit
equality** with `gguf.quants.dequantize` on every tensor of two real files; a
tolerance is not acceptable for weights.

Upstream also requires `accelerate` for step 4 ("accelerate is required when
loading a GGUF file"). This route does not pass `gguf_file` to the inner
`from_pretrained` -- it passes the state dict, which is upstream's own
`state_dict=` path -- so that check is never reached and `accelerate` is not
needed. That is a difference from upstream and is stated, not hidden.

**What is refused, by name** (AGENTS.md §18):

  * the `gguf` package missing -- steps 1 and 2 are upstream's and they import
    it; this is upstream's own requirement, not one added here;
  * an architecture whose `TensorProcessor` rewrites tensors (Llama's q/k
    permutation, MoE expert splits, ...) -- those rewrites are numpy code in
    `modeling_gguf_pytorch_utils.py` and are not ported, so such a file would
    load with wrong weights. Only architectures whose processor is the
    identity (`TensorProcessor` itself: qwen2, qwen3, phi3, ...) are read;
  * `quantization_config=` -- upstream refuses it with `gguf_file`, and here it
    would mean dequantising the file's blocks and quantising them again, which
    is lossy twice. Keeping the file's own blocks is docs/graph/GGUF.md §5;
  * `state_dict=` -- upstream refuses it with `gguf_file`.
"""

import os

import torch
from transformers import AutoConfig, PreTrainedConfig
from transformers.modeling_gguf_pytorch_utils import (
    TENSOR_PROCESSORS,
    TensorProcessor,
    get_gguf_hf_weights_map,
)
from transformers.models.auto.auto_factory import _get_model_class
from transformers.utils import cached_file
from transformers.utils.import_utils import is_gguf_available

from . import GGML_TYPES, GGUFFile, UnsupportedGGUF, load_tensor

# The keyword arguments upstream's Auto `from_pretrained` treats as hub
# arguments (auto_factory.py, `hub_kwargs_names`).
_HUB_KWARGS = (
    "cache_dir",
    "force_download",
    "local_files_only",
    "proxies",
    "revision",
    "subfolder",
    "token",
)


class GGUFLoadReport:
    """What the GGUF door read, what it mapped, and what it left out.

    Published as `model.torchnative_gguf`, for the reason
    `torchnative.quant.hf._LoadReport` is published as
    `model.torchnative_quantization`: `from_pretrained` returns only a model,
    so what happened has to be reachable from it or it is not reachable.
    """

    def __init__(self, path, architecture, version):
        self.path = path
        self.architecture = architecture
        self.version = version
        self.loaded = {}  # hf name -> (gguf name, GGML type name)
        self.unmapped = []  # gguf names with no parameter in this model
        self.dequantised = True

    @property
    def types(self):
        out = {}
        for _, kind in self.loaded.values():
            out[kind] = out.get(kind, 0) + 1
        return dict(sorted(out.items()))

    def __str__(self):
        lines = [
            f"gguf v{self.version} {os.path.basename(self.path)} ({self.architecture}): "
            f"{len(self.loaded)} tensors loaded {self.types}, "
            f"{len(self.unmapped)} not mapped to a parameter",
            "  every block type was dequantised to float32 by candle; the model is dense",
        ]
        if self.unmapped:
            lines.append(f"  not mapped, e.g. {self.unmapped[0]}")
        return "\n".join(lines)

    __repr__ = __str__


def _refuse(kwargs):
    if kwargs.get("quantization_config") is not None:
        raise UnsupportedGGUF(
            "from_pretrained(gguf_file=..., quantization_config=...) is refused. "
            "Upstream refuses the same combination; here it would also mean "
            "dequantising the file's GGML blocks and quantising the result again, "
            "which loses precision twice and keeps none of the file's own bytes. "
            "Load without quantization_config to get the dense model upstream's "
            "GGUF path gives. Keeping the file's blocks in a loaded model is not "
            "implemented yet (docs/graph/GGUF.md §5); torchnative.gguf."
            "quantized_linear maps one tensor onto a QuantizedLinear byte for byte."
        )
    if kwargs.get("state_dict") is not None:
        raise ValueError(
            "`state_dict` cannot be passed together with a `gguf_file` -- the same "
            "refusal upstream's from_pretrained makes."
        )
    if not is_gguf_available():
        raise ImportError(
            "Loading a GGUF file goes through transformers' own GGUF config mapping "
            "and tensor-name tables, and both import the `gguf` package (upstream's "
            "requirement: pip install 'gguf>=0.10.0'). torchnative replaces only the "
            "step that turns GGML blocks into tensors (docs/graph/GGUF.md §1)."
        )


def _resolve(name, gguf_file, hub_kwargs):
    """`(config source, gguf file name, resolved path)`, the way upstream resolves.

    Upstream takes `gguf_file` as-is when it names an existing file and
    otherwise looks it up under `name` (`_get_resolved_checkpoint_files`); the
    same precedence is kept. The config is read from the same file.
    """
    if os.path.isfile(gguf_file):
        path = os.path.abspath(gguf_file)
        return os.path.dirname(path), os.path.basename(path), path
    if name is None:
        raise ValueError(
            f"gguf_file={gguf_file!r} is not a file, and there is no model id or "
            "directory to look it up in"
        )
    return name, gguf_file, cached_file(name, gguf_file, **hub_kwargs)


def state_dict_from_gguf(f, model_class, config):
    """`({hf name: float32 tensor}, GGUFLoadReport)` for model `model_class(config)`.

    Upstream's step 2 and a replacement for its step 3 (module docstring).
    """
    arch = f.metadata.get("general.architecture")
    processor_cls = TENSOR_PROCESSORS.get(arch, TensorProcessor)
    if processor_cls is not TensorProcessor:
        raise UnsupportedGGUF(
            f"GGUF architecture {arch!r}: transformers passes its tensors through "
            f"{processor_cls.__name__}, which rewrites them in numpy "
            "(modeling_gguf_pytorch_utils.py) before they become parameters. That "
            "rewrite is not ported to this stack, so the weights would load wrong; "
            "refused instead. Architectures whose processor is the identity "
            "(qwen2, qwen3, phi3, ...) load."
        )
    processor = processor_cls(config=config.to_dict())
    with torch.device("meta"):
        skeleton = model_class(config)
    mapping = get_gguf_hf_weights_map(skeleton, processor)
    del skeleton

    report = GGUFLoadReport(f.path, arch, f.version)
    state_dict = {}
    for gguf_name, info in f.tensors.items():
        hf_name = mapping.get(gguf_name)
        if hf_name is None:
            # Upstream drops these too (`if name not in tensor_key_mapping:
            # continue`); recording them is the difference.
            report.unmapped.append(gguf_name)
            continue
        tensor = load_tensor(f, gguf_name, keep_blocks=False)
        # Upstream's numpy path produces float32 for every GGML type
        # (`gguf.quants.dequantize` returns float32, F16 included), and the
        # loader casts from there to whatever `dtype=` asked for. Widening F16
        # and BF16 to float32 is exact, so following it costs no bits.
        if tensor.dtype != torch.float32:
            tensor = tensor.to(torch.float32)
        state_dict[hf_name] = tensor
        report.loaded[hf_name] = (gguf_name, GGML_TYPES[info.ggml_type][0])
    return state_dict, report


def from_pretrained(auto_cls, pretrained_model_name_or_path, *model_args, **kwargs):
    """`auto_cls.from_pretrained(name, gguf_file=...)`, routed as the module docstring says."""
    gguf_file = kwargs.pop("gguf_file")
    _refuse(kwargs)
    hub_kwargs = {k: kwargs.pop(k) for k in _HUB_KWARGS if k in kwargs}
    source, gguf_name, path = _resolve(pretrained_model_name_or_path, gguf_file, hub_kwargs)

    config = kwargs.pop("config", None)
    if not isinstance(config, PreTrainedConfig):
        # Mirrors auto_factory.py: `dtype="auto"` is meaningless to a config,
        # and the remaining kwargs are offered to the config first, exactly as
        # upstream offers them, so `num_hidden_layers=2` and friends still work.
        config_kwargs = dict(kwargs)
        for key in ("dtype", "torch_dtype"):
            if config_kwargs.get(key) == "auto":
                config_kwargs.pop(key)
        config, unused = AutoConfig.from_pretrained(
            source, gguf_file=gguf_name, return_unused_kwargs=True, **hub_kwargs, **config_kwargs
        )
        kwargs = {k: v for k, v in kwargs.items() if k in unused or k in ("dtype", "torch_dtype")}

    model_class = _get_model_class(config, auto_cls._model_mapping)
    with GGUFFile(path) as f:
        state_dict, report = state_dict_from_gguf(f, model_class, config)

    model = model_class.from_pretrained(None, *model_args, config=config, state_dict=state_dict, **kwargs)
    model.torchnative_gguf = report
    return model
