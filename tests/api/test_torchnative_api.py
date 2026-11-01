"""Tests for `torchnative.api.TorchNativeAPI` (SPEC S7.2).

Exercises:
1. Lazy import: `import torchnative.api` does not import `torch`.
2. Device orchestration & in-place deployment: `deploy(model)` returns the
   exact same `nn.Module` (`deployed is model`), still supports `backward()`,
   and records `model.torchnative_deployment` / `api.deployment_report`.
3. Quantisation orchestration: `deploy(model, quantize="q8_0")` replaces
   eligible `nn.Linear` leaves in place via `torchnative.quant.quantize_`.
4. Lifetime policy orchestration: `Delta` (`delta_names=...`) and
   `BufferSnapshot` (`buffer_names=...`) across `record()`, `revert()`,
   `apply()`, and `persist()`/`load_delta()`, restoring base weights and
   buffers bit-for-bit.
5. Weight deployment channel: `save_weights(model, path)` and
   `load_weights(model, path)` round-trip via safetensors bit-for-bit.
6. Refusals by name: invalid device names/objects, unknown quantisation
   formats, non-`nn.Module` inputs, or missing delta/model state refuse by
   name.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
_ROOT = next(p for p in Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
_PY = str(_ROOT / "torchnative" / "python")
_SUPPORT = str(_ROOT / "tests" / "_support")
if _SUPPORT not in sys.path:
    sys.path.insert(0, _SUPPORT)
if _PY not in sys.path:
    sys.path.insert(0, _PY)

_VENDOR_SHIM = _ROOT / "torchnative" / "python" / "torch" / "_C.abi3.so"


def _assert_raises(exc_type, match, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type as exc:
        if match and match not in str(exc):
            raise AssertionError(
                f"expected {exc_type.__name__} matching {match!r}, got {exc!r}"
            ) from exc
        return exc
    raise AssertionError(f"expected {exc_type.__name__}({match!r}), nothing raised")


def test_import_torchnative_api_does_not_import_torch():
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [_PY] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, torchnative.api; print('torch' in sys.modules)",
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert proc.stdout.strip() == "False", proc.stdout


def test_deploy_returns_same_nn_module_and_places_on_device():
    import torch
    import torch.nn as nn
    import torchnative.device as tndevice
    from torchnative.api import TorchNativeAPI

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 4)
            self.fc2 = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc2(torch.relu(self.fc1(x)))

    torch.manual_seed(0)
    model = TinyModel()
    api = TorchNativeAPI(device=tndevice.cpu)
    deployed = api.deploy(model)

    # INTENT §7: deploy returns the exact same nn.Module, never a wrapper.
    assert deployed is model
    assert api.device is tndevice.cpu
    assert hasattr(model, "torchnative_deployment")
    assert api.deployment_report == model.torchnative_deployment
    assert api.deployment_report["device"] == "cpu"
    assert api.deployment_report["eager"] is True
    assert api.deployment_report["quantize"] is None

    # The deployed model still runs forward and backward.
    x = torch.randn(3, 8)
    out = deployed(x)
    loss = (out * out).sum()
    loss.backward()
    assert model.fc1.weight.grad is not None
    assert float(model.fc1.weight.grad.abs().max().item()) > 0.0


def test_deploy_with_quantization_replaces_linear_leaves_in_place():
    import torch

    if not hasattr(torch._C, "_quantize"):
        import _skip

        _skip.skip(
            f"   (skipped test_deploy_with_quantization_replaces_linear_leaves_in_place: "
            f"no vendored shim at {_VENDOR_SHIM} -- run scripts/vendor/install_shim.sh)"
        )
        return
    import torch.nn as nn
    from torchnative.api import TorchNativeAPI
    from torchnative.quant import QuantizedLinear

    class BlockModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(32, 16)
            self.head = nn.Linear(16, 4)

        def forward(self, x):
            return self.head(self.proj(x))

    torch.manual_seed(1)
    model = BlockModel()
    api = TorchNativeAPI(
        device="cpu",
        quantize="q8_0",
        quantize_predicate=lambda name, _mod: name == "proj",
    )
    deployed = api.deploy(model)
    assert deployed is model
    assert isinstance(model.proj, QuantizedLinear)
    assert isinstance(model.head, nn.Linear)
    qrep = api.deployment_report["quantize"]
    assert qrep is not None
    assert qrep["format"] == "q8_0"
    assert qrep["replaced"] == ["proj"]

    x = torch.randn(2, 32)
    y = model(x)
    assert list(y.shape) == [2, 4]


def test_lifetime_policy_delta_and_buffer_snapshot_record_revert_apply_persist():
    import torch
    import torch.nn as nn
    from torchnative.api import TorchNativeAPI

    class AdaptableModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 4)
            self.bn = nn.BatchNorm1d(4)

        def forward(self, x):
            return self.bn(self.fc(x))

    torch.manual_seed(2)
    model = AdaptableModel()
    base_weight_bytes = model.fc.weight.detach().clone().tolist()
    base_mean_bytes = model.bn.running_mean.detach().clone().tolist()
    base_batches = int(model.bn.num_batches_tracked.item())

    api = TorchNativeAPI(
        device="cpu",
        delta_names=("fc.weight", "fc.bias"),
        buffer_names=("bn.running_mean", "bn.running_var", "bn.num_batches_tracked"),
    )
    deployed = api.deploy(model)
    assert deployed is model
    assert api.delta is not None
    assert api.buffer_snapshot is not None
    assert api.delta.covers == ("fc.weight", "fc.bias")

    # Run a forward/backward step that updates both parameters and buffers.
    model.train()
    x = torch.randn(6, 4)
    out = model(x)
    loss = (out * out).sum()
    loss.backward()
    with torch.no_grad():
        model.fc.weight.data.copy_(model.fc.weight.detach() - 0.1 * model.fc.weight.grad)
        model.fc.bias.data.copy_(model.fc.bias.detach() - 0.1 * model.fc.bias.grad)

    adapted_weight = model.fc.weight.detach().clone().tolist()
    assert adapted_weight != base_weight_bytes
    assert model.bn.running_mean.detach().clone().tolist() != base_mean_bytes
    assert int(model.bn.num_batches_tracked.item()) == base_batches + 1

    # Record parameter delta.
    delta = api.record()
    assert delta is api.delta
    assert delta.norm() > 0.0

    # Revert on system event (background / user switch) restores base bit-for-bit.
    api.revert()
    assert model.fc.weight.detach().clone().tolist() == base_weight_bytes
    assert model.bn.running_mean.detach().clone().tolist() == base_mean_bytes
    assert int(model.bn.num_batches_tracked.item()) == base_batches

    # Apply puts base + delta back onto covered parameters.
    api.apply()
    assert model.fc.weight.detach().clone().tolist() != base_weight_bytes

    # Persist and load_delta round-trip bit-for-bit via safetensors.
    with tempfile.TemporaryDirectory() as tmp:
        delta_path = Path(tmp) / "delta.safetensors"
        saved_path = api.persist(delta_path)
        assert Path(saved_path) == delta_path
        loaded = api.load_delta(delta_path)
        assert set(loaded.keys()) == {"fc.weight", "fc.bias"}
        assert loaded["fc.weight"].tolist() == api.delta.value["fc.weight"].tolist()


def test_weights_save_and_load_round_trip():
    import torch
    import torch.nn as nn
    from torchnative.api import TorchNativeAPI

    class SmallNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(6, 3)
            self.l2 = nn.Linear(3, 2)

        def forward(self, x):
            return self.l2(torch.relu(self.l1(x)))

    torch.manual_seed(3)
    src = SmallNet()
    dst = SmallNet()
    assert src.l1.weight.tolist() != dst.l1.weight.tolist()

    api = TorchNativeAPI(device="cpu")
    with tempfile.TemporaryDirectory() as tmp:
        weights_path = Path(tmp) / "weights.safetensors"
        api.save_weights(src, weights_path)
        loaded_model = api.load_weights(dst, weights_path)
        assert loaded_model is dst
        assert dst.l1.weight.tolist() == src.l1.weight.tolist()
        assert dst.l1.bias.tolist() == src.l1.bias.tolist()
        assert dst.l2.weight.tolist() == src.l2.weight.tolist()
        assert dst.l2.bias.tolist() == src.l2.bias.tolist()


def test_unsupported_device_quantize_or_arguments_refuse_by_name():
    import torch.nn as nn
    from torchnative.api import TorchNativeAPI

    # Unknown device name refuses by name.
    _assert_raises(ValueError, "tpu", TorchNativeAPI, device="tpu")

    # Wrong device type refuses by name.
    _assert_raises(TypeError, "device", TorchNativeAPI, device=123)

    # Unknown quantize format refuses by name.
    api_bad_q = TorchNativeAPI(device="cpu", quantize="int2_magic")
    _assert_raises(ValueError, "int2_magic", api_bad_q.deploy, nn.Linear(4, 4))

    # Non-nn.Module refuses by name.
    api = TorchNativeAPI(device="cpu")
    _assert_raises(TypeError, "nn.Module", api.deploy, {"not": "a_module"})

    # Calling record/revert/persist before opening a delta refuses by name.
    _assert_raises(ValueError, "delta", api.record, nn.Linear(4, 4))
    _assert_raises(ValueError, "revert", api.revert, nn.Linear(4, 4))
    _assert_raises(ValueError, "delta", api.persist, "unused.safetensors")


if __name__ == "__main__":
    import _skip

    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    raise SystemExit(1 if _skip.run_tests(tests, suite="test_torchnative_api") else 0)
