"""Tests for the Hugging Face resolver protocol and mobile build-time kernel bundling (S7.1)."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")


def _require_kernels():
    try:
        from kernels._rust import KernelDependency, KernelVersion
        from kernels.resolver import Resolver
    except ImportError:
        import _skip

        raise _skip.Skip("optional Hugging Face kernels package is not installed")
    return KernelDependency, KernelVersion, Resolver


def _write_local_kernel_repo(
    root: Path,
    repo_name: str,
    *,
    variants: dict[str, tuple[str, str]],
    version: int = 1,
) -> Path:
    """Create a real local HF kernel repository with `build/<variant>/metadata.json` and `__init__.py`."""
    from kernels._rust import Digest, DigestAlgorithm

    repo_dir = root / repo_name
    build_dir = repo_dir / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    py_name = repo_name.replace("-", "_")

    for variant_str, (backend_type, code) in variants.items():
        vdir = build_dir / variant_str
        vdir.mkdir(parents=True, exist_ok=True)
        (vdir / "__init__.py").write_text(code, encoding="utf-8")
        digest = Digest.hash_variant(DigestAlgorithm.SHA256, vdir)
        meta = {
            "id": f"{py_name}_v{version}_{variant_str.replace('-', '_')}",
            "name": repo_name,
            "version": version,
            "license": "Apache-2.0",
            "python-depends": [],
            "kernel-depends": [],
            "backend": {"type": backend_type},
            "digest": {
                "algorithm": str(digest.algorithm).lower(),
                "files": dict(digest.files),
            },
        }
        (vdir / "metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return repo_dir


def test_bundle_resolver_uses_exact_dependency_and_preselected_path():
    KernelDependency, KernelVersion, Resolver = _require_kernels()
    from torchnative.kernels import BundleResolver

    dep = KernelDependency(repo_id="example/kernel", version=KernelVersion.Version(1))
    other = KernelDependency(repo_id="example/kernel", version=KernelVersion.Version(2))
    path = Path("/bundle/ios-arm64/variant")
    marker = object()
    resolver = BundleResolver({dep: path})
    with patch("kernels.resolver.resolve_local_kernel", return_value=marker) as load:
        assert isinstance(resolver, Resolver)
        assert resolver.resolve(api=None, backend="metal", kernel=dep) is marker
        load.assert_called_once_with(path, backend="metal")
        assert resolver.resolve(api=None, backend="metal", kernel=other) is None
        assert load.call_count == 1


def test_mobile_target_profiles_and_unsupported_backend_refusals():
    from torchnative.kernels import (
        UnsupportedKernelBackend,
        UnsupportedKernelTarget,
        target_profile,
    )

    android = target_profile("android-arm64")
    assert android.target == "android-arm64"
    assert android.resolution == "build-time"
    assert android.backends == ("cpu",)

    android_v8a = target_profile("android-arm64-v8a")
    assert android_v8a.target == "android-arm64"
    assert android_v8a.resolution == "build-time"
    assert android_v8a.backends == ("cpu",)

    ios = target_profile("ios-arm64")
    assert ios.target == "ios-arm64"
    assert ios.resolution == "build-time"
    assert ios.backends == ("metal", "cpu")

    ios_sim = target_profile("ios-arm64-sim")
    assert ios_sim.target == "ios-arm64-sim"
    assert ios_sim.resolution == "build-time"
    assert ios_sim.backends == ("metal", "cpu")

    macos = target_profile("macos-arm64")
    assert macos.resolution == "runtime"
    assert macos.backends == ("metal", "cpu")

    linux = target_profile("linux-x86_64")
    assert linux.resolution == "runtime"
    assert linux.backends == ("cuda", "cpu")

    windows = target_profile("windows-x86_64")
    assert windows.resolution == "runtime"
    assert windows.backends == ("cuda", "cpu")

    for bad_target in ("wasm32", "android-x86_64", "unknown-os"):
        try:
            target_profile(bad_target)
        except UnsupportedKernelTarget as exc:
            assert bad_target in str(exc), exc
        else:
            raise AssertionError(f"target {bad_target!r} was not refused by name")

    for bad_backend in ("vulkan", "wasm"):
        try:
            target_profile("android-arm64", backend=bad_backend)
        except UnsupportedKernelBackend as exc:
            assert bad_backend in str(exc), exc
        else:
            raise AssertionError(f"backend {bad_backend!r} was not refused by name")


def test_build_time_selection_and_bundle_staging_for_mobile_targets():
    KernelDependency, KernelVersion, Resolver = _require_kernels()
    from torchnative.kernels import (
        BundleResolver,
        UnsupportedKernelTarget,
        select_mobile_variant,
        stage_bundle,
    )

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        tmp_path = Path(tmp)
        repo_dir = _write_local_kernel_repo(
            tmp_path / "sources",
            "rms-norm",
            variants={
                "torch-cpu": (
                    "cpu",
                    "BACKEND = 'cpu'\ndef forward(x):\n    return x * 2\n",
                ),
                "torch-metal": (
                    "metal",
                    "BACKEND = 'metal'\ndef forward(x):\n    return x * 3\n",
                ),
            },
            version=1,
        )

        # Build-time variant selection for Android picks torch-cpu; for iOS picks torch-metal first.
        android_v = select_mobile_variant(repo_dir, target="android-arm64")
        assert android_v.name == "torch-cpu", android_v

        ios_v = select_mobile_variant(repo_dir, target="ios-arm64")
        assert ios_v.name == "torch-metal", ios_v

        ios_cpu_v = select_mobile_variant(repo_dir, target="ios-arm64", backend="cpu")
        assert ios_cpu_v.name == "torch-cpu", ios_cpu_v

        # Desktop targets refuse build-time mobile variant staging by name.
        try:
            select_mobile_variant(repo_dir, target="linux-x86_64")
        except UnsupportedKernelTarget as exc:
            assert "linux-x86_64" in str(exc), exc
        else:
            raise AssertionError("desktop target was not refused by select_mobile_variant")

        dep = KernelDependency(repo_id="kernels-community/rms-norm", version=KernelVersion.Version(1))
        bundle_dir = tmp_path / "app-bundle-android"
        manifest_path = stage_bundle(
            {dep: repo_dir},
            target="android-arm64",
            output_dir=bundle_dir,
        )
        assert manifest_path == bundle_dir / "bundle.json"
        assert manifest_path.is_file()

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["target"] == "android-arm64"
        assert manifest["resolution"] == "build-time"
        assert "kernels-community/rms-norm" in json.dumps(manifest["kernels"])

        resolver = BundleResolver.from_bundle(bundle_dir)
        assert isinstance(resolver, Resolver)
        loc = resolver.resolve(api=None, backend="cpu", kernel=dep)
        assert loc is not None
        assert loc.variant_str == "torch-cpu"
        assert (loc.variant_path / "metadata.json").is_file()


def test_bundle_resolver_loads_real_local_kernel_via_get_kernel_without_network():
    KernelDependency, KernelVersion, _ = _require_kernels()
    from torchnative.kernels import BundleResolver, get_kernel, stage_bundle

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        tmp_path = Path(tmp)
        repo_dir = _write_local_kernel_repo(
            tmp_path / "sources",
            "fast-act",
            variants={
                "torch-cpu": (
                    "cpu",
                    "KERNEL_TAG = 'android-bundled-cpu'\ndef apply_act(x):\n    return x + 42\n",
                ),
            },
            version=1,
        )
        dep = KernelDependency(repo_id="kernels-community/fast-act", version=KernelVersion.Version(1))
        bundle_dir = tmp_path / "bundle"
        stage_bundle({dep: repo_dir}, target="android-arm64", output_dir=bundle_dir)
        resolver = BundleResolver.from_bundle(bundle_dir)

        with patch("huggingface_hub.hf_api.HfApi.hf_hub_download", side_effect=AssertionError("network call")), \
             patch("huggingface_hub.hf_api.HfApi.snapshot_download", side_effect=AssertionError("network call")):
            mod = get_kernel(
                "kernels-community/fast-act",
                version=1,
                backend="cpu",
                resolver=resolver,
            )
        assert mod.KERNEL_TAG == "android-bundled-cpu"
        assert mod.apply_act(8) == 50


def test_use_bundle_routes_hf_kernels_get_kernel_through_the_bundle():
    KernelDependency, KernelVersion, _ = _require_kernels()
    import kernels as hf_kernels
    from torchnative.kernels import BundleResolver, stage_bundle, use_bundle

    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as tmp:
        tmp_path = Path(tmp)
        repo_dir = _write_local_kernel_repo(
            tmp_path / "sources",
            "bundled-attn",
            variants={
                "torch-cpu": (
                    "cpu",
                    "ORIGIN = 'bundle-hook'\ndef scale(x):\n    return x * 5\n",
                ),
            },
            version=2,
        )
        dep = KernelDependency(repo_id="kernels-community/bundled-attn", version=KernelVersion.Version(2))
        bundle_dir = tmp_path / "bundle-ios"
        stage_bundle({dep: repo_dir}, target="ios-arm64", backend="cpu", output_dir=bundle_dir)
        resolver = BundleResolver.from_bundle(bundle_dir)

        orig_get_kernel = hf_kernels.get_kernel
        with use_bundle(resolver, backend="cpu"):
            with patch("huggingface_hub.hf_api.HfApi.snapshot_download", side_effect=AssertionError("network call")):
                mod = hf_kernels.get_kernel("kernels-community/bundled-attn", version=2)
            assert mod.ORIGIN == "bundle-hook"
            assert mod.scale(7) == 35
        assert hf_kernels.get_kernel is orig_get_kernel


if __name__ == "__main__":
    import _skip

    raise SystemExit(
        1
        if _skip.run_tests(
            [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")],
            suite="test_kernel_bundle",
        )
        else 0
    )
