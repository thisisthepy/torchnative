"""A preallocated KV cache for transformers' `generate` (S7.4, issue #180).

transformers' default `DynamicLayer.update` is

    self.keys = torch.cat([self.keys, key_states], dim=-2)

once per layer per step, so a decode of `n` tokens copies the whole cache `n`
times: O(n^2) bytes moved, and on `mps` two fresh device buffers per layer per
step. `PreallocatedLayer` keeps one buffer per layer with room to spare, writes
each step's keys and values into it in place, and hands attention a narrowed
view of the filled part. A buffer grows only when it is full, to the next
multiple of `chunk` at least double its size, so growth copies are amortised
O(n).

    from torchnative.kernels.kv_cache import PreallocatedCache
    cache = PreallocatedCache(config=model.config)
    model.generate(ids, past_key_values=cache, max_new_tokens=128)

**What attention sees is unchanged.** `keys`/`values` are
`[batch, kv_heads, seq_len, head_dim]` with the same values `DynamicLayer`
would hold, `get_seq_length` and `get_mask_sizes` answer the same, and no
padding is ever exposed, so the attention mask (or its absence) is the one
transformers builds for a `DynamicCache`. The only difference is the layout:
the view's head stride is the buffer's, not `seq_len * head_dim`.
`torch_c/src/sdpa_metal.rs` reads that layout without a copy.

**What it refuses.** Layer types other than `full_attention` (sliding-window,
chunked, linear-attention and compressed layers each have their own update
rule in transformers) raise `UnsupportedCacheLayout` naming them, rather than
being given full-attention semantics they do not have.
"""

from __future__ import annotations

__all__ = ["PreallocatedCache", "PreallocatedLayer", "UnsupportedCacheLayout"]


class UnsupportedCacheLayout(NotImplementedError):
    """A model whose cache layout `PreallocatedCache` does not implement."""


def _classes():
    from transformers.cache_utils import Cache, DynamicLayer, get_layer_types_and_kwargs

    return Cache, DynamicLayer, get_layer_types_and_kwargs


def _make_layer_class():
    Cache, DynamicLayer, _ = _classes()

    class PreallocatedLayer(DynamicLayer):
        """`DynamicLayer` with an in-place, amortised-growth buffer instead of `torch.cat`."""

        def __init__(self, chunk: int = 256, **kwargs):
            super().__init__(**kwargs)
            if int(chunk) < 1:
                raise ValueError(f"chunk must be a positive number of tokens, got {chunk!r}")
            self.chunk = int(chunk)
            self._kbuf = None
            self._vbuf = None
            self._len = 0

        def lazy_initialization(self, key_states, value_states):
            self.dtype, self.device = key_states.dtype, key_states.device
            self._len = 0
            self._kbuf = None
            self._vbuf = None
            self.keys = key_states[..., :0, :]
            self.values = value_states[..., :0, :]
            self.is_initialized = True

        def _capacity(self):
            return 0 if self._kbuf is None else self._kbuf.shape[-2]

        def _grow(self, key_states, value_states, need):
            import torch

            cap = self._capacity()
            new_cap = max(need, 2 * cap, self.chunk)
            new_cap = -(-new_cap // self.chunk) * self.chunk
            kshape = list(key_states.shape)
            vshape = list(value_states.shape)
            kshape[-2] = new_cap
            vshape[-2] = new_cap
            kbuf = torch.empty(kshape, dtype=key_states.dtype, device=key_states.device)
            vbuf = torch.empty(vshape, dtype=value_states.dtype, device=value_states.device)
            if self._len:
                kbuf[..., : self._len, :] = self._kbuf[..., : self._len, :]
                vbuf[..., : self._len, :] = self._vbuf[..., : self._len, :]
            self._kbuf, self._vbuf = kbuf, vbuf

        def update(self, key_states, value_states, *args, **kwargs):
            if not self.is_initialized:
                self.lazy_initialization(key_states, value_states)
            n = key_states.shape[-2]
            end = self._len + n
            if (
                self._kbuf is None
                or end > self._capacity()
                or self._kbuf.shape[:-2] != key_states.shape[:-2]
            ):
                if self._kbuf is not None and self._kbuf.shape[:-2] != key_states.shape[:-2]:
                    raise UnsupportedCacheLayout(
                        f"PreallocatedLayer: the new states' batch/head shape "
                        f"{tuple(key_states.shape[:-2])} differs from the cache's "
                        f"{tuple(self._kbuf.shape[:-2])}"
                    )
                self._grow(key_states, value_states, end)
            self._kbuf[..., self._len : end, :] = key_states
            self._vbuf[..., self._len : end, :] = value_states
            self._len = end
            self.keys = self._kbuf[..., :end, :]
            self.values = self._vbuf[..., :end, :]
            return self.keys, self.values

        def get_seq_length(self) -> int:
            return self._len if self.is_initialized else 0

        def reset(self) -> None:
            if self.is_initialized:
                self._len = 0
                self.keys = self._kbuf[..., :0, :] if self._kbuf is not None else self.keys[..., :0, :]
                self.values = self._vbuf[..., :0, :] if self._vbuf is not None else self.values[..., :0, :]

        def crop(self, tokens_to_remove: int) -> None:
            if tokens_to_remove > 0:
                # transformers' deprecated absolute form: the size to keep.
                keep = min(tokens_to_remove, self._len)
            else:
                keep = max(self._len - abs(tokens_to_remove), 0)
            if keep == self._len or self._kbuf is None:
                return
            self._len = keep
            self.keys = self._kbuf[..., :keep, :]
            self.values = self._vbuf[..., :keep, :]

        def _rebuild(self, keys, values):
            # Batch reshuffles materialise new tensors; the buffer becomes them.
            self._kbuf = keys.contiguous()
            self._vbuf = values.contiguous()
            self._len = self._kbuf.shape[-2]
            self.keys = self._kbuf[..., : self._len, :]
            self.values = self._vbuf[..., : self._len, :]

        def reorder_cache(self, beam_idx) -> None:
            if self._len > 0:
                idx = beam_idx.to(self.keys.device)
                self._rebuild(self.keys.index_select(0, idx), self.values.index_select(0, idx))

        def batch_repeat_interleave(self, repeats: int) -> None:
            if self._len > 0:
                self._rebuild(
                    self.keys.repeat_interleave(repeats, dim=0),
                    self.values.repeat_interleave(repeats, dim=0),
                )

        def batch_select_indices(self, indices) -> None:
            if self._len > 0:
                self._rebuild(self.keys[indices, ...], self.values[indices, ...])

    return PreallocatedLayer


def _make_cache_class(layer_cls):
    Cache, _, get_layer_types_and_kwargs = _classes()

    class PreallocatedCache(Cache):
        """A `Cache` of `PreallocatedLayer`s; pass it to `generate(past_key_values=...)`."""

        def __init__(self, config=None, chunk: int = 256):
            if config is not None:
                decoder_config = config.get_text_config(decoder=True)
                layer_types, _ = get_layer_types_and_kwargs(decoder_config)
                other = sorted({t for t in layer_types if t != "full_attention"})
                if other:
                    raise UnsupportedCacheLayout(
                        f"PreallocatedCache implements full_attention layers only; this model "
                        f"also has {other}, whose cache update in transformers is not a plain "
                        "append. Use transformers' DynamicCache for it."
                    )
                super().__init__(layers=[layer_cls(chunk=chunk) for _ in layer_types])
            else:
                super().__init__(layer_class_to_replicate=lambda: layer_cls(chunk=chunk))

    return PreallocatedCache


_BUILT = {}


def __getattr__(name):
    # Built on first use so `import torchnative.kernels` does not import transformers.
    if name in ("PreallocatedLayer", "PreallocatedCache"):
        if "PreallocatedLayer" not in _BUILT:
            layer = _make_layer_class()
            _BUILT["PreallocatedLayer"] = layer
            _BUILT["PreallocatedCache"] = _make_cache_class(layer)
        return _BUILT[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
