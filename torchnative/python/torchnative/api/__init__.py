"""Deployment, lifetime policy and device orchestration.

The `torch` import is deferred rather than done at module scope. This
distribution *provides* `torch` -- the vendored tree plus our `_C` -- rather
than depending on someone else's, so at import time there may not be one yet:
the wheels that carry the tree are not wired up, and until they are, importing
`torchnative` should not require a PyTorch that this package is itself meant to
supply. See DESIGN.md §2 and the README's Status section.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable

if TYPE_CHECKING:
    from torch import nn
    from torchnative.delta import BufferSnapshot, Delta
    from torchnative.device import Device

__all__ = ["TorchNativeAPI"]


def _resolve_device(device: Device | str | None) -> Device:
    from torchnative import device as tndevice

    if device is None:
        return tndevice.cpu
    if tndevice.is_device(device):
        return device
    if isinstance(device, str):
        known = {d.type: d for d in tndevice.members()}
        if device not in known:
            raise ValueError(
                f"torchnative.api: unknown device {device!r}; known devices are "
                f"{sorted(known)}"
            )
        return known[device]
    raise TypeError(
        f"torchnative.api: device must be a torchnative.device.Device or a device "
        f"name string, got {type(device).__name__}"
    )


def _require_module(model: Any, caller: str) -> nn.Module:
    import torch.nn as nn

    if not isinstance(model, nn.Module):
        raise TypeError(
            f"torchnative.api.{caller}: expected a torch.nn.Module instance, "
            f"got {type(model).__name__}"
        )
    return model


class TorchNativeAPI(object):
    """On-device model deployment, lifetime policy, and device orchestration.

    See ``docs/INTENT.md`` §4 and §7, ``docs/design/DESIGN.md`` §2, §3 and §10,
    and ``docs/SPEC.md`` S7.2:

    * ``deploy(model, ...)`` returns the **exact same** ``nn.Module`` instance
      (never a wrapper), after optionally quantising ``nn.Linear`` leaves in
      place via ``torchnative.quant.quantize_``, placing or lowering the model
      onto the resolved ``torchnative.device.Device`` via ``model.to(...)``,
      and opening parameter ``Delta`` / ``BufferSnapshot`` lifetime state.
    * ``record()``, ``revert()``, ``apply()``, ``persist()``, and
      ``load_delta()`` orchestrate the parameter ``Delta`` and
      ``BufferSnapshot`` across system events (backgrounding, user switch,
      process restart) bit-for-bit.
    * ``save_weights()`` and ``load_weights()`` round-trip model state via
      safetensors for the weight deployment channel (DESIGN.md §10).
    """

    def __init__(
        self,
        device: Device | str | None = None,
        *,
        quantize: str | None = None,
        quantize_predicate: Callable[[str, Any], bool] | None = None,
        delta_names: Iterable[str] | None = None,
        buffer_names: Iterable[str] | None = None,
        compile_options: dict[str, Any] | None = None,
    ) -> None:
        self.device: Device = _resolve_device(device)
        self.quantize: str | None = quantize
        self.quantize_predicate = quantize_predicate
        self.delta_names: tuple[str, ...] | None = (
            None if delta_names is None else tuple(delta_names)
        )
        self.buffer_names: tuple[str, ...] | None = (
            None if buffer_names is None else tuple(buffer_names)
        )
        self.compile_options: dict[str, Any] = dict(compile_options or {})
        self.model: nn.Module | None = None
        self.delta: Delta | None = None
        self.buffer_snapshot: BufferSnapshot | None = None
        self.deployment_report: dict[str, Any] | None = None

    def deploy(
        self,
        model: nn.Module,
        *,
        device: Device | str | None = None,
        quantize: str | None = None,
        quantize_predicate: Callable[[str, Any], bool] | None = None,
        delta_names: Iterable[str] | None = None,
        buffer_names: Iterable[str] | None = None,
        **compile_options: Any,
    ) -> nn.Module:
        """Deploy ``model`` in place onto the target device and return ``model``."""
        _require_module(model, "deploy")
        target_device = self.device if device is None else _resolve_device(device)
        q_fmt = self.quantize if quantize is None else quantize
        q_pred = (
            self.quantize_predicate
            if quantize_predicate is None
            else quantize_predicate
        )
        d_names = (
            self.delta_names if delta_names is None else tuple(delta_names)
        )
        b_names = (
            self.buffer_names if buffer_names is None else tuple(buffer_names)
        )
        merged_options = dict(self.compile_options)
        merged_options.update(compile_options)

        quant_summary: dict[str, Any] | None = None
        if q_fmt is not None:
            from torchnative import quant as tnquant

            known_formats = tuple(tnquant.FORMATS())
            if q_fmt not in known_formats:
                raise ValueError(
                    f"torchnative.api.deploy: unknown quantisation format {q_fmt!r}; "
                    f"supported formats are {list(known_formats)}"
                )
            q_report = tnquant.quantize_(model, format=q_fmt, predicate=q_pred)
            quant_summary = {
                "format": q_report.format,
                "replaced": list(q_report.replaced),
                "skipped": list(q_report.skipped),
                "dense_bytes": q_report.dense_bytes,
                "quantized_bytes": q_report.quantized_bytes,
                "ratio": q_report.ratio,
            }

        if target_device.eager:
            if merged_options:
                raise TypeError(
                    f"torchnative.api.deploy: compile options {sorted(merged_options)} "
                    f"are only valid for compiled devices, not eager device "
                    f"torchnative.device.{target_device.type}"
                )
            deployed = model.to(target_device)
        else:
            deployed = model.to(target_device, **merged_options)

        if d_names is not None:
            self.open_delta(deployed, d_names)
        if b_names is not None:
            self.open_buffer_snapshot(deployed, b_names)

        self.device = target_device
        self.model = deployed
        report = {
            "device": target_device.type,
            "eager": bool(target_device.eager),
            "quantize": quant_summary,
            "delta_covers": None if self.delta is None else list(self.delta.covers),
            "buffer_covers": (
                None
                if self.buffer_snapshot is None
                else list(self.buffer_snapshot.covers)
            ),
            "offload": getattr(deployed, "torchnative_offload", None),
        }
        self.deployment_report = report
        deployed.torchnative_deployment = report
        return deployed

    def open_delta(self, model: nn.Module, names: Iterable[str]) -> Delta:
        """Open a parameter ``Delta`` over ``names`` on ``model``."""
        _require_module(model, "open_delta")
        from torchnative.delta import Delta

        self.model = model
        self.delta = Delta.over(model, names)
        return self.delta

    def open_buffer_snapshot(
        self, model: nn.Module, names: Iterable[str]
    ) -> BufferSnapshot:
        """Open a ``BufferSnapshot`` over ``names`` on ``model``."""
        _require_module(model, "open_buffer_snapshot")
        from torchnative.delta import BufferSnapshot

        self.model = model
        self.buffer_snapshot = BufferSnapshot.over(model, names)
        return self.buffer_snapshot

    def _target_model(self, model: nn.Module | None, caller: str) -> nn.Module:
        target = self.model if model is None else model
        if target is None:
            raise ValueError(
                f"torchnative.api.{caller}: no model is bound to this TorchNativeAPI "
                f"instance; pass `model` explicitly or call `deploy(model)` first"
            )
        return _require_module(target, caller)

    def record(self, model: nn.Module | None = None) -> Delta:
        """Record the covered parameter offsets from base into the active ``Delta``."""
        if self.delta is None:
            raise ValueError(
                "torchnative.api.record: no parameter delta is open; pass "
                "`delta_names` to TorchNativeAPI/deploy or call `open_delta(model, names)`"
            )
        target = self._target_model(model, "record")
        return self.delta.record(target)

    def revert(self, model: nn.Module | None = None) -> nn.Module:
        """Restore base parameters and/or snapshotted buffers bit-for-bit."""
        if self.delta is None and self.buffer_snapshot is None:
            raise ValueError(
                "torchnative.api.revert: neither a parameter delta nor a buffer "
                "snapshot is open to revert"
            )
        target = self._target_model(model, "revert")
        if self.delta is not None:
            self.delta.revert(target)
        if self.buffer_snapshot is not None:
            self.buffer_snapshot.revert(target)
        return target

    def apply(self, model: nn.Module | None = None) -> nn.Module:
        """Apply the recorded parameter ``Delta`` offset onto ``model``."""
        if self.delta is None:
            raise ValueError(
                "torchnative.api.apply: no parameter delta is open to apply"
            )
        target = self._target_model(model, "apply")
        self.delta.apply(target)
        return target

    def persist(self, path: str | Path, what: str = "value") -> str | Path:
        """Write the active ``Delta`` to ``path`` as safetensors."""
        if self.delta is None:
            raise ValueError(
                "torchnative.api.persist: no parameter delta is open to persist"
            )
        return self.delta.persist(path, what=what)

    @staticmethod
    def load_delta(path: str | Path) -> dict[str, Any]:
        """Load a persisted delta tensor table from ``path`` via safetensors."""
        from torchnative.delta import Delta

        return Delta.load(path)

    @staticmethod
    def save_weights(model: nn.Module, path: str | Path) -> Path:
        """Write floating-point parameters and buffers of ``model`` to safetensors."""
        _require_module(model, "save_weights")
        import json
        import struct
        from torchnative.delta import Delta

        state = model.state_dict()
        if not state:
            raise ValueError(
                f"torchnative.api.save_weights: {type(model).__name__} has an empty "
                f"state_dict"
            )
        header: dict[str, Any] = {}
        blob = bytearray()
        offset = 0
        for name in sorted(state):
            dtype_name, raw = Delta._bytes(state[name])
            header[name] = {
                "dtype": dtype_name,
                "shape": list(state[name].shape),
                "data_offsets": [offset, offset + len(raw)],
            }
            blob += raw
            offset += len(raw)
        encoded = json.dumps(header, separators=(",", ":")).encode("utf-8")
        encoded += b" " * ((8 - len(encoded) % 8) % 8)
        out_path = Path(path)
        with open(out_path, "wb") as handle:
            handle.write(struct.pack("<Q", len(encoded)))
            handle.write(encoded)
            handle.write(bytes(blob))
        return out_path

    def load_weights(
        self, model: nn.Module, path: str | Path, *, strict: bool = True
    ) -> nn.Module:
        """Load safetensors weights from ``path`` into ``model`` and return ``model``."""
        _require_module(model, "load_weights")
        from safetensors.torch import load_file

        tensors = load_file(str(path), device="cpu")
        model.load_state_dict(tensors, strict=strict)
        return model
