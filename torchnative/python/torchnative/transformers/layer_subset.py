"""Load only a contiguous range of a decoder's layers (BACKEND.md D12).

    from torchnative.transformers import load_layer_subset
    stage = load_layer_subset("path/to/checkpoint", layers=(2, 4), dtype=torch.float32)

D12 (decided, leader, 2026-11-08): "A pipeline rank may load only its layer
subset: a new loader option under `torchnative.transformers`." This is that
option. Its shape (a function, its name, its arguments) is an **agent
proposal, not confirmed** (docs/distributed/PPINFER.md section 3).

**What the installed loader already does, read off transformers 5.15.1.**
`PreTrainedModel.from_pretrained` builds the model under
`torch.device("meta")` (`get_init_context`, `modeling_utils.py:3796`), opens
each safetensors file with `safe_open` and takes `get_slice(k)` for every key,
which reads nothing (`modeling_utils.py:4470-4472`). It then materialises a
slice only when its (renamed) key is in the model's own meta `state_dict()`;
a checkpoint key the model does not have is recorded as unexpected and never
sliced (`core_model_loading.py:1608-1700`, `_materialize_copy` is the only
`tensor[...]`). So the loader can skip non-local layers **without
`device_map`**, provided the model it builds has only the local layers. It
has no option that builds such a model. This module supplies that one piece
and nothing else: a subclass of the real model class whose `__init__` runs
the real `__init__` (on meta, so the dropped layers cost nothing) and then
prunes. Every byte is still read by transformers' own loader, through its own
dtype handling, rotary-buffer initialisation, weight tying and generation
config. When loading is done the object's class is set back to the real
class, so what comes back is `Qwen3ForCausalLM`, not a wrapper.

**How the subset keeps the checkpoint's key names.** The kept layers are held
in a `StageLayers`, an `nn.ModuleList` whose children are registered under
their **global** index (`model.layers.2.mlp.up_proj.weight` on the stage that
owns layers 2 and 3), so `state_dict()` names equal the checkpoint's and no
key is renamed. Iteration is positional, which is all the model's forward
does (`for i, layer in enumerate(self.layers[:n])`).

**How each stage's cache holds only its layers.** Attention modules index the
cache by `self.layer_idx`. The kept layers' `layer_idx` attributes are
renumbered to their local position, the stage's `config.num_hidden_layers`
becomes the local count and `config.layer_types` is sliced, so
`DynamicCache(config=model.config)` has exactly one entry per local layer and
`cache.get_seq_length()` (layer 0 of the stage) is the stage's own length,
which the mask builder reads. The global indices are kept in
`model._torchnative_stage["global_layer_idx"]`.

**What each stage keeps besides its layers.** The first stage keeps
`embed_tokens`; the last keeps the final `norm` and `lm_head`; any other
stage has `embed_tokens` and `lm_head` set to `None` and `norm` replaced by
`nn.Identity()`, because the base model's forward calls `self.norm`
unconditionally and a stage that is not last must hand on the un-normed
hidden state. With tied embeddings (`tie_word_embeddings=True`, as in the
small Qwen3 checkpoints) the input and output embeddings are one tensor, so
the first and the last stage each keep both names for it (no extra memory)
and the checkpoint's one stored copy loads into it.

**Refused by name** (AGENTS.md section 18): a range that is empty, reversed or
outside the model; a model whose layers are not `model.model.layers` as an
`nn.ModuleList`, or whose layers carry no `layer_idx` to renumber; layer types
other than full and sliding attention (recurrent, linear or KV-shared layers
keep state this does not split); encoder-decoder models; `device_map`,
`tp_plan`, `distributed_config`, `device_mesh`, `state_dict`, `gguf_file` and
quantisation, none of which was built or tested with a subset; and a
checkpoint that lacks a key the subset needs (transformers would otherwise
initialise it randomly and say so only in a log line).

Graded *builds* (AGENTS.md section 16): this file compiled; nothing ran.
"""

from __future__ import annotations

import re

import torch.nn as nn

#: `from_pretrained` keywords refused with a subset, and why.
_REFUSED_KWARGS = {
    "export": "`export=` is not implemented in this build (see "
              "torchnative.transformers' refusal of it in from_pretrained)",
    "device_map": "a device_map places whole modules by name and was not built or "
                  "tested together with a layer subset; load the subset, then "
                  "`.to(device)` it",
    "tp_plan": "tensor parallelism inside a pipeline stage is not built",
    "distributed_config": "tensor parallelism inside a pipeline stage is not built",
    "device_mesh": "tensor parallelism inside a pipeline stage is not built",
    "state_dict": "a subset is read from the checkpoint files; an in-memory "
                  "state_dict is already fully materialised, so there is "
                  "nothing to skip",
    "gguf_file": "the GGUF route (torchnative/gguf/hf.py) dequantises whole "
                 "tensors through its own loader, which has no subset",
    "quantization_config": "quantised loading was not built or tested with a "
                           "subset",
    "load_in_8bit": "quantised loading was not built or tested with a subset",
    "load_in_4bit": "quantised loading was not built or tested with a subset",
    "output_loading_info": "this loader reads the loading info itself, to refuse a "
                           "missing key; it is on the returned model as "
                           "`_torchnative_stage['loading_info']`",
}

#: Layer types whose cache is one key/value tensor per layer, so splitting the
#: layer list splits the cache. Anything else keeps state this does not split.
_SPLITTABLE_LAYER_TYPES = ("full_attention", "sliding_attention")


class UnsupportedLayerSubset(NotImplementedError):
    """A layer subset this build cannot load, refused by name."""


def _refuse(why):
    raise UnsupportedLayerSubset("torchnative.transformers.load_layer_subset: " + why)


class StageLayers(nn.ModuleList):
    """The kept layers, registered under their global index.

    `state_dict()` therefore names `model.layers.<global>.*`, the checkpoint's
    own names. Iteration and integer indexing are positional. Slicing returns a
    positionally numbered `nn.ModuleList` (all the model's forward does with it
    is iterate). Growing, shrinking or reassigning it refuses: a stage's range
    is fixed when it is built.
    """

    def __init__(self, modules=None, first=0):
        nn.Module.__init__(self)
        for offset, module in enumerate(modules or ()):
            self.add_module(str(first + offset), module)

    def __getitem__(self, idx):
        values = list(self._modules.values())
        if isinstance(idx, slice):
            return nn.ModuleList(values[idx])
        return values[idx]

    def _fixed(self, *_a, **_k):
        _refuse("a pipeline stage's layer list is fixed when it is built; it "
                "cannot grow, shrink or be reassigned in place")

    __setitem__ = __delitem__ = append = extend = insert = pop = _fixed
    __iadd__ = __add__ = _fixed


# ---------------------------------------------------------------------------
# Pruning a model to one stage
# ---------------------------------------------------------------------------


def _check_range(layers, num_layers):
    ok = (isinstance(layers, (list, tuple)) and len(layers) == 2
          and all(isinstance(v, int) and not isinstance(v, bool) for v in layers))
    if not ok:
        _refuse("`layers` must be one contiguous (start, stop) pair of layer "
                "indices, stop exclusive; got %r. A non-contiguous set of layers "
                "is not a pipeline stage" % (layers,))
    start, stop = layers
    if not 0 <= start < stop <= num_layers:
        _refuse("layer range (%d, %d) is empty, reversed or outside a model of %d "
                "layers; a stage must own at least one layer"
                % (start, stop, num_layers))
    return start, stop


def _check_model(model):
    cls = type(model).__name__
    config = model.config
    if getattr(config, "is_encoder_decoder", False):
        _refuse("%s is an encoder-decoder model; only decoder-only causal LMs are "
                "split here" % cls)
    base = getattr(model, getattr(model, "base_model_prefix", "model"), None)
    layers = getattr(base, "layers", None)
    if base is None or not isinstance(layers, nn.ModuleList):
        _refuse("%s has no `model.layers` nn.ModuleList, so there is no layer list "
                "this loader understands" % cls)
    for name in ("embed_tokens", "norm"):
        if not isinstance(getattr(base, name, None), nn.Module):
            _refuse("%s.model has no `%s` module; the stage split assumes the "
                    "embed -> layers -> norm -> lm_head shape" % (cls, name))
    if not isinstance(getattr(model, "lm_head", None), nn.Module):
        _refuse("%s has no `lm_head` module; the stage split assumes the "
                "embed -> layers -> norm -> lm_head shape" % cls)
    if getattr(config, "num_kv_shared_layers", None):
        _refuse("%s shares key/value states across layers (num_kv_shared_layers), "
                "so a stage's cache would need another stage's layers" % cls)
    types = getattr(config, "layer_types", None)
    if types is not None:
        odd = sorted(set(types) - set(_SPLITTABLE_LAYER_TYPES))
        if odd:
            _refuse("%s has layer types %s, whose cache is not one key/value pair "
                    "per layer; only %s split here"
                    % (cls, odd, list(_SPLITTABLE_LAYER_TYPES)))
    return base


def _renumber(layer, global_idx, local_idx):
    """Set every `layer_idx == global_idx` under `layer` to `local_idx`."""
    hits = 0
    for module in layer.modules():
        if isinstance(getattr(module, "layer_idx", None), int) and not isinstance(
                module.layer_idx, bool):
            if module.layer_idx != global_idx:
                _refuse("layer %d holds a submodule %s whose layer_idx is %d, not %d; "
                        "the cache index cannot be renumbered safely"
                        % (global_idx, type(module).__name__, module.layer_idx,
                           global_idx))
            module.layer_idx = local_idx
            hits += 1
    if hits == 0:
        _refuse("layer %d (%s) carries no integer `layer_idx`, so there is no cache "
                "index to renumber; this layer type is not one this loader "
                "understands" % (global_idx, type(layer).__name__))


def prune_to_stage(model, layers, embed=None, head=None):
    """Prune `model` in place to the layers `layers = (start, stop)`.

    Used by `load_layer_subset` on the meta model before any weight is read.
    `embed` defaults to `start == 0` and `head` to `stop == num_layers`.
    Returns `model`.
    """
    if getattr(model, "_torchnative_stage", None) is not None:
        _refuse("this model is already a stage of layers %r; prune the full model, "
                "not a stage" % (model._torchnative_stage["layers"],))
    base = _check_model(model)
    config = model.config
    num_layers = len(base.layers)
    if getattr(config, "num_hidden_layers", num_layers) != num_layers:
        _refuse("config.num_hidden_layers is %r but the model has %d layers; it was "
                "already pruned or is not one this loader understands"
                % (config.num_hidden_layers, num_layers))
    start, stop = _check_range(layers, num_layers)
    embed = (start == 0) if embed is None else bool(embed)
    head = (stop == num_layers) if head is None else bool(head)
    tied = bool(getattr(model, "all_tied_weights_keys", None))

    kept = [base.layers[i] for i in range(start, stop)]
    for local, layer in enumerate(kept):
        _renumber(layer, start + local, local)
    base.layers = StageLayers(kept, first=start)

    # Tied embeddings are one tensor under two names: a stage that keeps
    # either end keeps both names for it.
    keep_embed = embed or (tied and head)
    keep_head = head or (tied and embed)
    if not keep_embed:
        base.embed_tokens = None
    if not head:
        base.norm = nn.Identity()
    if not keep_head:
        model.lm_head = None

    present = {n for n, _ in model.named_parameters(remove_duplicate=False)}
    if tied:
        model.all_tied_weights_keys = {
            t: s for t, s in model.all_tied_weights_keys.items()
            if t in present and s in present}

    types = getattr(config, "layer_types", None)
    if types is not None:
        config.layer_types = list(types[start:stop])
    config.num_hidden_layers = stop - start

    # Checkpoint keys this stage does not own: not an error, so keep them out
    # of the load report. Matched by regex, as transformers matches them.
    prefix = re.escape(getattr(model, "base_model_prefix", "model"))
    own = "|".join(str(i) for i in range(start, stop))
    ignore = {r"^%s\.layers\.(?!(?:%s)\.)\d+\." % (prefix, own)}
    if not keep_embed:
        ignore.add(r"^%s\.embed_tokens\." % prefix)
    if not head:
        ignore.add(r"^%s\.norm\." % prefix)
    if not keep_head:
        ignore.add(r"^lm_head\.")
    model._keys_to_ignore_on_load_unexpected = set(
        getattr(model, "_keys_to_ignore_on_load_unexpected", None) or ()) | ignore

    model._torchnative_stage = {
        "layers": (start, stop),
        "num_layers": num_layers,
        "embed": embed,
        "head": head,
        "tied": tied,
        "global_layer_idx": {local: start + local for local in range(stop - start)},
    }
    return model


# ---------------------------------------------------------------------------
# The loader option
# ---------------------------------------------------------------------------


def _model_class(config, auto_class):
    from transformers.models.auto.auto_factory import _get_model_class

    if auto_class is None:
        from transformers import AutoModelForCausalLM as auto_class
    mapping = auto_class._model_mapping
    if type(config) not in mapping:
        _refuse("%s has no %s class for config %s"
                % (auto_class.__name__, auto_class.__name__, type(config).__name__))
    return _get_model_class(config, mapping)


def load_layer_subset(pretrained_model_name_or_path, layers, *, embed=None, head=None,
                      auto_class=None, **kwargs):
    """Load layers `start..stop-1` of a causal LM, reading only their weights.

    `layers` is `(start, stop)`, stop exclusive, in the full model's layer
    numbering. `embed` (default: `start == 0`) keeps the input embedding,
    `head` (default: `stop == num_layers`) the final norm and `lm_head`.
    Every other keyword goes to the real `from_pretrained` (for example
    `dtype=`), except the ones refused by name in `_REFUSED_KWARGS`.

    Returns the real model class's instance, pruned to the subset, with
    `model._torchnative_stage` describing the range and the loading info.
    Refuses when the checkpoint lacks a key the subset needs.
    """
    for name, why in _REFUSED_KWARGS.items():
        if name in kwargs and kwargs[name] not in (None, False):
            _refuse("`%s=` is refused with a layer subset: %s" % (name, why))
        kwargs.pop(name, None)

    from transformers import AutoConfig

    config = kwargs.pop("config", None)
    if config is None:
        config_kwargs = {k: kwargs[k] for k in ("revision", "trust_remote_code",
                                                "cache_dir", "local_files_only",
                                                "token") if k in kwargs}
        config = AutoConfig.from_pretrained(pretrained_model_name_or_path,
                                            **config_kwargs)
    if getattr(config, "quantization_config", None) is not None:
        _refuse("the checkpoint is quantised (config.quantization_config); "
                "quantised loading was not built or tested with a subset")
    # Checked before anything is read, as well as again inside the prune.
    _check_range(layers, int(getattr(config, "num_hidden_layers", 0) or 0))
    real = _model_class(config, auto_class)
    spec = (tuple(layers), embed, head)

    def __init__(self, cfg, *args, **kw):
        real.__init__(self, cfg, *args, **kw)
        prune_to_stage(self, spec[0], embed=spec[1], head=spec[2])

    stage_cls = type(real.__name__, (real,), {
        "__init__": __init__,
        "__module__": real.__module__,
        "__qualname__": real.__qualname__,
    })
    model, info = stage_cls.from_pretrained(
        pretrained_model_name_or_path, config=config, output_loading_info=True,
        **kwargs)
    # Back to the real class: the subclass only existed to prune in __init__.
    model.__class__ = real

    missing = sorted(info.get("missing_keys", ()) or ())
    if missing:
        _refuse("the checkpoint at %r lacks %d key(s) this subset needs, which "
                "transformers would have initialised randomly: %s"
                % (pretrained_model_name_or_path, len(missing), missing[:8]))
    mismatched = sorted(info.get("mismatched_keys", ()) or ())
    if mismatched:
        _refuse("the checkpoint at %r has %d key(s) whose shape does not match "
                "the subset: %s" % (pretrained_model_name_or_path, len(mismatched),
                                   mismatched[:8]))
    model._torchnative_stage["loading_info"] = {
        k: sorted(v) if isinstance(v, (set, list, tuple)) else v
        for k, v in info.items()}
    return model
