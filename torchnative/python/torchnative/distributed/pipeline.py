"""Pipeline-parallel generation over point-to-point (BACKEND.md section 6.2, route B).

    import torch.distributed as dist
    import torchnative.distributed  # registers backend="torchnative"
    from torchnative.distributed.pipeline import PipelineStage

    dist.init_process_group("torchnative")             # one rank per stage
    stage = PipelineStage.from_pretrained(path, split=[3, 1], dtype=torch.float32)
    tokens = stage.generate(input_ids, max_new_tokens=16)   # same call on every rank

BACKEND.md section 9 step 10, issue #295, SPEC S6.13. Graded *builds*: this
file compiled and nothing ran (AGENTS.md sections 10 and 16).

Provenance. Pipeline parallelism as the uneven-device tool is D11 and a rank
loading only its layer subset is D12 (decided, leader, 2026-11-08). Route B
itself (a thin stage runner rather than `torch.distributed.pipelining`) was an
agent proposal in BACKEND.md section 6.2 and this round took it; **everything
below is an agent proposal, not confirmed**: the class and function names, the
split argument, which stage drives, the message layout and tags, the refusals
(docs/distributed/PPINFER.md).

**What runs where.** The model is the real transformers class, loaded per rank
by `torchnative.transformers.load_layer_subset` with only that rank's
contiguous layer range. Stage 0 keeps the input embedding; the last stage
keeps the final norm and `lm_head`. The **last stage runs transformers' own
`generate`** (greedy): its `forward` is wrapped for the duration of the call
so that, each step, it sends the step's token ids to stage 0, receives the
hidden state from the stage before it, and then calls the real `forward` with
`inputs_embeds=` that hidden state. Token selection, logits processors,
stopping criteria, padding of finished rows and the attention-mask and
position bookkeeping are therefore transformers' code, unchanged. The other
stages serve: each step they receive (token ids from the last stage, at stage
0; a hidden state from the previous stage, elsewhere), run the real base
model's `forward` over their own layers with their own cache, and send the
hidden state on.

**What crosses a boundary, per step.** On every edge a 4-element int64 header
`[kind, batch, length, past]` precedes the payload. Forward edges
(stage s to s+1) carry the hidden state `(batch, length, hidden_size)` in the
model's dtype; the return edge (last stage to stage 0) carries the step's
token ids `(batch, length)` int64. When `generate` returns, the last stage
sends the final sequences down the chain once, so every rank returns the same
tensor. Nothing vocabulary-sized crosses. Position ids and the attention mask
are rebuilt on every stage from `(batch, past + length)`, as transformers'
`_prepare_position_ids_for_generation` builds them for an all-ones mask, which
is why padded inputs refuse.

**Each stage keeps its own cache** for its own layers only: a
`DynamicCache(config=stage_model.config)` whose entry count is the stage's
layer count (the loader renumbers the layers' `layer_idx` and slices
`config.layer_types`). The header's `past` is the sender's cache length; a
receiver whose own cache disagrees raises `PipelineDesync` naming both,
instead of computing on a cache that drifted.

**Refused by name** (AGENTS.md section 18): sampling (`do_sample=True`), beam
search and multiple return sequences, assisted or prompt-lookup decoding,
streamers and every other `generate` keyword not listed in `_ALLOWED`,
padded input (an attention mask that is not all ones), `use_cache=False`, a
static or quantised cache implementation, a pipeline of one stage, a split
that leaves a stage empty or is not contiguous, a split whose stage count is
not the world size, a group other than the default group (`new_group` is
BACKEND.md step 8), and a stage whose loaded range is not the one the split
assigns to its rank.

**Placement hook.** `split` may be a callable `split(num_layers, world_size)`
returning the ranges; issue #296 builds the capability-weighted placement that
plugs in there.
"""

from __future__ import annotations

import functools

#: Message tags. Distinct per message class so the backend's traffic record
#: (`torchnative.distributed.backend.record_traffic`) can be read by class.
TAG_HEADER = 7301
TAG_HIDDEN = 7302
TAG_IDS = 7303
TAG_RESULT = 7304

_STOP = 0
_STEP = 1

#: `generate` keywords this runner accepts. Everything else refuses by name.
_ALLOWED = frozenset({
    "max_new_tokens", "min_new_tokens", "max_length", "eos_token_id",
    "pad_token_id", "attention_mask", "do_sample", "num_beams", "use_cache",
})


class UnsupportedPipeline(NotImplementedError):
    """A pipeline configuration or generate option this build refuses by name."""


class PipelineDesync(RuntimeError):
    """A stage's cache length disagrees with the message it received."""


def _refuse(what, why):
    raise UnsupportedPipeline("torchnative.distributed.pipeline.%s: %s" % (what, why))


# ---------------------------------------------------------------------------
# The split
# ---------------------------------------------------------------------------


def split_layers(num_layers, world_size, split):
    """`[(start, stop), ...]`, one contiguous non-empty range per rank.

    `split` is one of:

    * a list of per-stage layer counts, e.g. `[3, 1]` (uneven is allowed, D11);
    * a list of `(start, stop)` pairs, stop exclusive, that tile
      `0..num_layers` in order;
    * a callable `split(num_layers, world_size)` returning either of the
      above (the placement hook, issue #296).

    Refuses by name: an empty stage, a gap, an overlap, a reordering, a range
    outside the model, and a stage count that is not `world_size`.
    """
    if callable(split):
        split = split(num_layers, world_size)
    if not isinstance(split, (list, tuple)) or not split:
        _refuse("split_layers", "split must be a non-empty list of layer counts or "
                "(start, stop) pairs, or a callable returning one; got %r" % (split,))
    if world_size < 2:
        _refuse("split_layers", "a pipeline of %d stage is the model itself; call "
                "the model's own generate" % world_size)
    if all(isinstance(c, int) and not isinstance(c, bool) for c in split):
        if any(c < 1 for c in split):
            _refuse("split_layers", "split %r leaves a stage with no layers; every "
                    "stage must own at least one" % (list(split),))
        ranges, start = [], 0
        for count in split:
            ranges.append((start, start + count))
            start += count
    elif all(isinstance(r, (list, tuple)) and len(r) == 2 for r in split):
        ranges = [(r[0], r[1]) for r in split]
        for r in ranges:
            if not all(isinstance(v, int) and not isinstance(v, bool) for v in r):
                _refuse("split_layers", "range %r is not a pair of layer indices" % (r,))
    else:
        _refuse("split_layers", "split %r mixes counts and ranges, or holds something "
                "that is neither" % (split,))
    expected = 0
    for index, (start, stop) in enumerate(ranges):
        if stop <= start:
            _refuse("split_layers", "stage %d's range (%d, %d) is empty; every stage "
                    "must own at least one layer" % (index, start, stop))
        if start != expected:
            _refuse("split_layers", "stage %d starts at layer %d but the stage before "
                    "it ends at %d; the stages must be contiguous and in order (a "
                    "non-contiguous split is not built)" % (index, start, expected))
        expected = stop
    if expected != num_layers:
        _refuse("split_layers", "the split covers layers 0..%d but the model has %d"
                % (expected, num_layers))
    if len(ranges) != world_size:
        _refuse("split_layers", "the split has %d stages but the world has %d ranks; "
                "one rank runs one stage" % (len(ranges), world_size))
    return ranges


# ---------------------------------------------------------------------------
# The stage
# ---------------------------------------------------------------------------


class PipelineStage:
    """One rank's stage: its layer range, its cache, its part of `generate`."""

    def __init__(self, model, ranges, group=None):
        import torch.distributed as dist

        if not dist.is_initialized():
            _refuse("PipelineStage", "torch.distributed is not initialised; call "
                    "init_process_group first (one rank per stage)")
        if group is not None and group is not dist.distributed_c10d._get_default_group():
            _refuse("PipelineStage", "only the default process group is supported; a "
                    "sub-group needs new_group, which is not built "
                    "(docs/distributed/BACKEND.md section 9 step 8)")
        info = getattr(model, "_torchnative_stage", None)
        if info is None:
            _refuse("PipelineStage", "the model is not a layer subset; load it with "
                    "torchnative.transformers.load_layer_subset or "
                    "PipelineStage.from_pretrained")
        world, rank = dist.get_world_size(), dist.get_rank()
        ranges = split_layers(info["num_layers"], world, list(ranges))
        if tuple(info["layers"]) != tuple(ranges[rank]):
            _refuse("PipelineStage", "rank %d holds layers %r but the split assigns it "
                    "%r" % (rank, tuple(info["layers"]), tuple(ranges[rank])))
        if info["embed"] != (rank == 0) or info["head"] != (rank == world - 1):
            _refuse("PipelineStage", "rank %d's model has embed=%r head=%r; stage 0 "
                    "must hold the embedding and only the last stage the head"
                    % (rank, info["embed"], info["head"]))
        self.model = model
        self.ranges = ranges
        self.stage = rank
        self.num_stages = world
        self.last = world - 1
        self.hidden_size = int(model.config.hidden_size)
        params = list(model.parameters())
        self.device = params[0].device
        self.dtype = next((p.dtype for p in params if p.is_floating_point()), None)
        #: This stage's cache from the latest `generate`, for inspection.
        self.cache = None
        #: Forward steps this stage ran in the latest `generate`.
        self.steps = 0
        self._pending = []

    @property
    def layer_range(self):
        return tuple(self.ranges[self.stage])

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, split, group=None,
                        **kwargs):
        """Load this rank's layers only (D12) and wrap them as its stage."""
        import torch.distributed as dist
        from transformers import AutoConfig

        from torchnative.transformers.layer_subset import load_layer_subset

        if not dist.is_initialized():
            _refuse("PipelineStage.from_pretrained", "torch.distributed is not "
                    "initialised; call init_process_group first (one rank per stage)")
        config = kwargs.pop("config", None)
        if config is None:
            config = AutoConfig.from_pretrained(pretrained_model_name_or_path)
        world, rank = dist.get_world_size(), dist.get_rank()
        ranges = split_layers(int(config.num_hidden_layers), world, split)
        model = load_layer_subset(pretrained_model_name_or_path, ranges[rank],
                                  config=config, **kwargs)
        return cls(model, ranges, group=group)

    # -- messages ------------------------------------------------------------
    def _isend(self, tensor, dst, tag):
        import torch.distributed as dist

        if tensor.device.type != "cpu":
            # The backend is registered for CPU tensors only (BACKEND.md section
            # 4.3): a device rank stages through host memory. Untested here.
            tensor = tensor.cpu()
        tensor = tensor.contiguous()
        self._pending.append((dist.isend(tensor, dst, tag=tag), tensor))

    def _recv(self, shape, dtype, src, tag):
        import torch
        import torch.distributed as dist

        buf = torch.empty(shape, dtype=dtype)
        dist.irecv(buf, src, tag=tag).wait()
        return buf

    def _flush(self):
        pending, self._pending = self._pending, []
        for work, _tensor in pending:
            work.wait()

    def _send_header(self, dst, kind, d0, d1, past):
        import torch

        self._isend(torch.tensor([kind, d0, d1, past], dtype=torch.int64), dst,
                    TAG_HEADER)

    def _recv_header(self, src):
        import torch

        kind, d0, d1, past = (int(v) for v in
                              self._recv((4,), torch.int64, src, TAG_HEADER).tolist())
        return kind, d0, d1, past

    def _send_hidden(self, hidden, past):
        """Header and hidden state to the next stage. A seam the tests sabotage."""
        batch, length = int(hidden.shape[0]), int(hidden.shape[1])
        self._send_header(self.stage + 1, _STEP, batch, length, past)
        self._isend(hidden, self.stage + 1, TAG_HIDDEN)

    def _new_cache(self):
        """This stage's cache: one entry per local layer. A seam the tests sabotage."""
        from transformers import DynamicCache

        return DynamicCache(config=self.model.config)

    def _check_past(self, past, src):
        own = self.cache.get_seq_length()
        if own != past:
            raise PipelineDesync(
                "torchnative.distributed.pipeline: stage %d (layers %r) holds %d "
                "cached positions but stage %d sent a step at position %d; the "
                "caches drifted apart, so this stage refuses to compute on its"
                % (self.stage, self.layer_range, own, src, past))

    # -- generate ------------------------------------------------------------
    def _validate(self, kwargs):
        for name in sorted(kwargs):
            if name not in _ALLOWED:
                _refuse("PipelineStage.generate", "`%s=` is not supported by the "
                        "pipeline runner; it accepts only %s. Beam search, sampling, "
                        "assisted decoding, streamers and output dictionaries are not "
                        "built" % (name, sorted(_ALLOWED)))
        if kwargs.get("do_sample"):
            _refuse("PipelineStage.generate", "sampling (do_sample=True) is not "
                    "built; only greedy decoding, whose tokens are checked against "
                    "the single-process run, is")
        if (kwargs.get("num_beams") or 1) != 1:
            _refuse("PipelineStage.generate", "beam search (num_beams=%r) is not "
                    "built: reordering beams would have to reorder every stage's "
                    "cache in step" % kwargs["num_beams"])
        if kwargs.get("use_cache", True) is False:
            _refuse("PipelineStage.generate", "use_cache=False is not supported: "
                    "every stage keeps its own cache, which is the point of the "
                    "split")
        gc = getattr(self.model, "generation_config", None)
        if gc is not None:
            if getattr(gc, "cache_implementation", None) not in (None, "dynamic"):
                _refuse("PipelineStage.generate", "generation_config.cache_"
                        "implementation=%r is not supported; each stage keeps a "
                        "DynamicCache" % gc.cache_implementation)
            if (getattr(gc, "num_return_sequences", 1) or 1) != 1:
                _refuse("PipelineStage.generate", "num_return_sequences=%r is not "
                        "built" % gc.num_return_sequences)

    def generate(self, input_ids=None, **kwargs):
        """Greedy generation across the pipeline; call it on every rank.

        Every rank passes the same arguments (they are validated everywhere,
        so a refusal is raised on every rank before any message is sent); the
        last stage reads `input_ids` and runs transformers' `generate`.
        Returns the final sequences `(batch, prompt + new)` on every rank.
        """
        self._validate(kwargs)
        self.steps = 0
        self.cache = self._new_cache()
        try:
            if self.stage == self.last:
                return self._drive(input_ids, kwargs)
            return self._serve()
        finally:
            self._flush()

    def _drive(self, input_ids, kwargs):
        import torch

        if input_ids is None or input_ids.dim() != 2:
            _refuse("PipelineStage.generate", "the last stage needs input_ids of shape "
                    "(batch, length)")
        mask = kwargs.pop("attention_mask", None)
        if mask is not None and (tuple(mask.shape) != tuple(input_ids.shape)
                                 or not bool((mask == 1).all())):
            _refuse("PipelineStage.generate", "a padded attention_mask is not "
                    "supported: the stages rebuild position ids and masks from the "
                    "sequence length alone, so only an all-ones mask is honoured")
        model, stage = self.model, self
        real_forward = model.forward

        @functools.wraps(real_forward)
        def forward(*args, **kw):
            if args:
                _refuse("PipelineStage.generate", "the last stage's forward takes "
                        "keyword arguments only (transformers' generate passes them so)")
            ids = kw.pop("input_ids", None)
            if kw.pop("inputs_embeds", None) is not None or ids is None:
                _refuse("PipelineStage.generate", "generation from inputs_embeds is "
                        "not built; pass input_ids")
            cache = kw.get("past_key_values")
            if cache is not stage.cache:
                raise PipelineDesync(
                    "torchnative.distributed.pipeline: transformers' generate handed "
                    "the last stage a cache that is not the stage's own")
            past = cache.get_seq_length()
            batch, length = int(ids.shape[0]), int(ids.shape[1])
            stage._flush()
            stage._send_header(0, _STEP, batch, length, past)
            stage._isend(ids, 0, TAG_IDS)
            prev = stage.stage - 1
            kind, d0, d1, their_past = stage._recv_header(prev)
            if (kind, d0, d1) != (_STEP, batch, length):
                raise PipelineDesync(
                    "torchnative.distributed.pipeline: the last stage expected a step "
                    "of (batch %d, length %d) from stage %d and got kind %d (%d, %d)"
                    % (batch, length, prev, kind, d0, d1))
            stage._check_past(their_past, prev)
            hidden = stage._recv((batch, length, stage.hidden_size), stage.dtype,
                                 prev, TAG_HIDDEN).to(stage.device)
            stage.steps += 1
            return real_forward(input_ids=None, inputs_embeds=hidden, **kw)

        model.forward = forward
        try:
            with torch.no_grad():
                sequences = model.generate(
                    input_ids, attention_mask=torch.ones_like(input_ids),
                    past_key_values=self.cache, do_sample=False, num_beams=1,
                    **{k: v for k, v in kwargs.items()
                       if k not in ("do_sample", "num_beams", "use_cache")})
        finally:
            del model.forward
        sequences = sequences.cpu()
        self._flush()
        batch, total = int(sequences.shape[0]), int(sequences.shape[1])
        self._send_header(0, _STOP, batch, total, 0)
        self._isend(sequences, 0, TAG_RESULT)
        return sequences

    def _serve(self):
        import torch

        src = self.last if self.stage == 0 else self.stage - 1
        base = getattr(self.model, getattr(self.model, "base_model_prefix", "model"))
        while True:
            kind, d0, d1, past = self._recv_header(src)
            if kind == _STOP:
                sequences = self._recv((d0, d1), torch.int64, src, TAG_RESULT)
                if self.stage + 1 < self.last:
                    self._flush()
                    self._send_header(self.stage + 1, _STOP, d0, d1, 0)
                    self._isend(sequences, self.stage + 1, TAG_RESULT)
                return sequences
            if kind != _STEP:
                raise PipelineDesync(
                    "torchnative.distributed.pipeline: stage %d got a header of kind "
                    "%d from stage %d" % (self.stage, kind, src))
            self._check_past(past, src)
            if self.stage == 0:
                ids = self._recv((d0, d1), torch.int64, src, TAG_IDS).to(self.device)
                inputs = {"input_ids": ids}
            else:
                hidden = self._recv((d0, d1, self.hidden_size), self.dtype, src,
                                    TAG_HIDDEN).to(self.device)
                inputs = {"inputs_embeds": hidden}
            # transformers' generate, for an all-ones mask: the mask spans the
            # cache plus this step, positions are its cumsum - 1, sliced.
            mask = torch.ones((d0, past + d1), dtype=torch.long, device=self.device)
            positions = (mask.cumsum(-1) - 1)[:, -d1:]
            with torch.no_grad():
                out = base(**inputs, attention_mask=mask, position_ids=positions,
                           past_key_values=self.cache, use_cache=True)
            self.steps += 1
            self._flush()
            self._send_hidden(out.last_hidden_state, past)
