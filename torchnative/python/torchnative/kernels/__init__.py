"""Bundle resolver for Hugging Face `kernels`.

Adopts the `kernels` discovery contract and inverts distribution: on mobile,
variants are selected and compiled at build time and read from the app bundle,
because iOS will not execute downloaded native code. Desktop keeps resolving
against the Hub. See DESIGN.md §8.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import json
from pathlib import Path
import shutil


__all__ = [
    "BundleResolver",
    "TargetProfile",
    "UnsupportedKernelBackend",
    "UnsupportedKernelTarget",
    "get_kernel",
    "select_mobile_variant",
    "stage_bundle",
    "target_profile",
    "use_bundle",
]


class UnsupportedKernelTarget(ValueError):
    """A target platform has no HF `kernels` profile in this build."""


class UnsupportedKernelBackend(ValueError):
    """A backend is not supported by the HF `kernels` target profile."""


@dataclass(frozen=True)
class TargetProfile:
    """HF `kernels` resolution policy and backend priority for one target platform.

    From `docs/design/DESIGN.md` §8 ("타깃 매트릭스"):
      - `android-arm64`: build-time resolution, `("cpu",)` (Android GPU `vulkan`
        is not in the HF `kernels` backend vocabulary).
      - `ios-arm64` / `ios-arm64-sim`: build-time resolution, `("metal", "cpu")`.
      - `macos-arm64`: runtime Hub resolution, `("metal", "cpu")`.
      - `linux-x86_64` / `linux-aarch64`: runtime Hub resolution, `("cuda", "cpu")`.
      - `windows-x86_64` / `windows-arm64`: runtime Hub resolution, `("cuda", "cpu")`.
    """

    target: str
    resolution: str
    backends: tuple[str, ...]
    cpu: str
    os: str


_TARGET_ALIASES = {
    "android-arm64-v8a": "android-arm64",
    "android_arm64_v8a": "android-arm64",
    "android_arm64": "android-arm64",
    "ios_arm64": "ios-arm64",
    "ios_arm64_sim": "ios-arm64-sim",
    "darwin-arm64": "macos-arm64",
    "macos_arm64": "macos-arm64",
    "linux_x86_64": "linux-x86_64",
    "linux_aarch64": "linux-aarch64",
    "win_amd64": "windows-x86_64",
    "windows_x86_64": "windows-x86_64",
    "win_arm64": "windows-arm64",
    "windows_arm64": "windows-arm64",
}

_TARGET_PROFILES = {
    "android-arm64": TargetProfile(
        target="android-arm64",
        resolution="build-time",
        backends=("cpu",),
        cpu="aarch64",
        os="android",
    ),
    "ios-arm64": TargetProfile(
        target="ios-arm64",
        resolution="build-time",
        backends=("metal", "cpu"),
        cpu="aarch64",
        os="ios",
    ),
    "ios-arm64-sim": TargetProfile(
        target="ios-arm64-sim",
        resolution="build-time",
        backends=("metal", "cpu"),
        cpu="aarch64",
        os="ios",
    ),
    "macos-arm64": TargetProfile(
        target="macos-arm64",
        resolution="runtime",
        backends=("metal", "cpu"),
        cpu="aarch64",
        os="darwin",
    ),
    "linux-x86_64": TargetProfile(
        target="linux-x86_64",
        resolution="runtime",
        backends=("cuda", "cpu"),
        cpu="x86_64",
        os="linux",
    ),
    "linux-aarch64": TargetProfile(
        target="linux-aarch64",
        resolution="runtime",
        backends=("cuda", "cpu"),
        cpu="aarch64",
        os="linux",
    ),
    "windows-x86_64": TargetProfile(
        target="windows-x86_64",
        resolution="runtime",
        backends=("cuda", "cpu"),
        cpu="x86_64",
        os="windows",
    ),
    "windows-arm64": TargetProfile(
        target="windows-arm64",
        resolution="runtime",
        backends=("cuda", "cpu"),
        cpu="aarch64",
        os="windows",
    ),
}


def target_profile(target: str, *, backend: str | None = None) -> TargetProfile:
    """Return the `TargetProfile` for `target`, or refuse unsupported targets/backends by name."""
    if not isinstance(target, str) or not target.strip():
        raise UnsupportedKernelTarget(f"torchnative.kernels: invalid target {target!r}")
    canonical = _TARGET_ALIASES.get(target, target)
    profile = _TARGET_PROFILES.get(canonical)
    if profile is None:
        raise UnsupportedKernelTarget(
            f"torchnative.kernels: unsupported target {target!r}. "
            f"Supported targets: {', '.join(sorted(_TARGET_PROFILES))} "
            "(DESIGN.md §8 and §10: android-x86_64 and wasm have no HF kernels target)."
        )
    if backend is not None:
        if backend not in profile.backends:
            raise UnsupportedKernelBackend(
                f"torchnative.kernels: backend {backend!r} is not supported on target "
                f"{profile.target!r} (allowed backends: {', '.join(profile.backends)}; "
                "DESIGN.md §8: HF kernels has no 'vulkan' or 'wasm' backend slot)."
            )
    return profile


def _candidate_variant_names(profile: TargetProfile, backend: str) -> tuple[str, ...]:
    """Ordered variant directory names compatible with `(profile, backend)`."""
    return (
        f"torch-stable-abi213-{backend}-{profile.cpu}-{profile.os}",
        f"torch213-{backend}-{profile.cpu}-{profile.os}",
        f"torch-{backend}",
        "torch-universal",
    )


def select_mobile_variant(
    repo_path: str | Path,
    *,
    target: str,
    backend: str | None = None,
) -> Path:
    """Select a build variant directory from `repo_path` at build time for a mobile target.

    Refuses desktop (`runtime`) targets and unsupported backends by name.
    """
    profile = target_profile(target, backend=backend)
    if profile.resolution != "build-time":
        raise UnsupportedKernelTarget(
            f"torchnative.kernels.select_mobile_variant: target {target!r} resolves at "
            f"{profile.resolution}, not build-time (mobile targets: android-arm64, ios-arm64, ios-arm64-sim)."
        )
    root = Path(repo_path)
    if not root.exists():
        raise FileNotFoundError(f"torchnative.kernels: kernel repository path does not exist: {root}")

    backends = (backend,) if backend is not None else profile.backends
    search_bases = [root / "build", root]

    for b in backends:
        for base in search_bases:
            if not base.is_dir():
                continue
            for vname in _candidate_variant_names(profile, b):
                candidate = base / vname
                if (candidate / "metadata.json").is_file():
                    return candidate
            # Also check any arch variant matching `*-<backend>-<cpu>-<os>` with metadata.json.
            suffix = f"-{b}-{profile.cpu}-{profile.os}"
            for entry in sorted(base.iterdir()):
                if entry.is_dir() and entry.name.endswith(suffix) and (entry / "metadata.json").is_file():
                    return entry

    if (root / "metadata.json").is_file():
        return root

    raise FileNotFoundError(
        f"torchnative.kernels: no compatible variant found in {root} for mobile target "
        f"{profile.target!r} (tried backends: {', '.join(backends)})."
    )


def _slug_repo_id(repo_id: str) -> str:
    return repo_id.strip("/").replace("/", "__")


def stage_bundle(
    kernel_sources: dict,
    *,
    target: str,
    output_dir: str | Path,
    backend: str | None = None,
) -> Path:
    """Stage selected mobile variants into `output_dir` and write `bundle.json`.

    `kernel_sources` maps `KernelDependency -> source_repo_or_variant_path`.
    Returns the path to `output_dir / "bundle.json"`.
    """
    from kernels._rust import KernelDependency, KernelPaths

    profile = target_profile(target, backend=backend)
    if profile.resolution != "build-time":
        raise UnsupportedKernelTarget(
            f"torchnative.kernels.stage_bundle: target {target!r} uses {profile.resolution} "
            "resolution, not build-time bundling."
        )

    out_root = Path(output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    staged_paths: dict[KernelDependency, Path] = {}
    manifest_entries = []

    for dep, src in kernel_sources.items():
        if not isinstance(dep, KernelDependency):
            raise TypeError(
                f"torchnative.kernels.stage_bundle: expected KernelDependency key, got {type(dep).__name__}"
            )
        variant_src = select_mobile_variant(src, target=profile.target, backend=backend)
        dest_dir = out_root / "variants" / _slug_repo_id(dep.repo_id) / variant_src.name
        if dest_dir.exists():
            shutil.rmtree(dest_dir)
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(variant_src, dest_dir)
        staged_paths[dep] = dest_dir
        manifest_entries.append(
            {
                "repo_id": dep.repo_id,
                "version": repr(dep.version),
                "variant": variant_src.name,
                "relative_path": dest_dir.relative_to(out_root).as_posix(),
            }
        )

    rel_paths = {dep: path.relative_to(out_root) for dep, path in staged_paths.items()}
    kernel_paths_json = json.loads(KernelPaths(rel_paths).to_json())
    manifest = {
        "schema_version": 1,
        "target": profile.target,
        "resolution": profile.resolution,
        "backends": list(profile.backends),
        "selected_backend": backend,
        "kernels": manifest_entries,
        "kernel_paths": kernel_paths_json,
    }
    manifest_path = out_root / "bundle.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest_path


class BundleResolver:
    """Resolve exact HF kernel dependencies to already-selected bundle variants.

    The mapping is prepared by the application build. This resolver never asks
    the supplied Hub API to download anything. The optional ``kernels`` package
    supplies metadata parsing and the returned ``LocalKernel`` implementation.
    """

    def __init__(self, kernel_paths, *, target: str | None = None):
        self._paths = {dependency: Path(path) for dependency, path in kernel_paths.items()}
        self.target = target

    @classmethod
    def from_bundle(cls, bundle_dir: str | Path) -> "BundleResolver":
        """Load a `BundleResolver` from a directory staged by `stage_bundle`."""
        from kernels._rust import KernelPaths

        root = Path(bundle_dir)
        manifest_path = root / "bundle.json" if root.is_dir() else root
        base_dir = manifest_path.parent
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        kp = KernelPaths.from_json(json.dumps(data["kernel_paths"]))
        resolved = {
            dep: (path if path.is_absolute() else base_dir / path)
            for dep, path in kp.items()
        }
        return cls(resolved, target=data.get("target"))

    @property
    def kernel_paths(self) -> dict:
        return dict(self._paths)

    def resolve(self, *, api, backend, kernel):
        path = self._paths.get(kernel)
        if path is None:
            return None
        from kernels.resolver import resolve_local_kernel

        return resolve_local_kernel(path, backend=backend)


def get_kernel(
    repo_id: str,
    *,
    revision: str | None = None,
    version: int | None = None,
    backend: str | None = None,
    resolver: BundleResolver | None = None,
    bundle_dir: str | Path | None = None,
    check_arch: bool = False,
):
    """Load a kernel from a `BundleResolver` (or `bundle_dir`) without network access."""
    from huggingface_hub import HfApi
    from kernels._rust import KernelDependency
    from kernels._versions import revision_or_version
    from kernels.load import get_kernel_with_resolver
    from kernels.validate import (
        AllKernelValidator,
        AllMetadataValidator,
        ArchValidator,
        default_kernel_validators,
        default_metadata_validators,
    )

    if resolver is None:
        if bundle_dir is None:
            raise ValueError("torchnative.kernels.get_kernel requires either `resolver` or `bundle_dir`")
        resolver = BundleResolver.from_bundle(bundle_dir)

    kernel_version = revision_or_version(revision=revision, version=version)
    dep = KernelDependency(repo_id=repo_id, version=kernel_version)

    validators = default_metadata_validators()
    if check_arch:
        validators.append(ArchValidator())

    return get_kernel_with_resolver(
        api=HfApi(),
        backend=backend,
        kernel=dep,
        resolver=resolver,
        kernel_validator=AllKernelValidator(validators=default_kernel_validators()),
        metadata_validator=AllMetadataValidator(validators=validators),
    )


@contextmanager
def use_bundle(resolver: BundleResolver | str | Path, *, backend: str | None = None):
    """Route `kernels.get_kernel` and `kernels.load.get_kernel` through `resolver` within the `with` block."""
    import kernels as hf_kernels
    import kernels.load as hf_load

    active_resolver = (
        resolver if isinstance(resolver, BundleResolver) else BundleResolver.from_bundle(resolver)
    )
    orig_top = hf_kernels.get_kernel
    orig_load = hf_load.get_kernel

    def _bundled_get_kernel(
        repo_id: str,
        *,
        revision: str | None = None,
        version: int | None = None,
        backend: str | None = backend,
        user_agent=None,
        trust_remote_code: bool | list[str] = False,
        check_arch: bool = False,
    ):
        return get_kernel(
            repo_id,
            revision=revision,
            version=version,
            backend=backend,
            resolver=active_resolver,
            check_arch=check_arch,
        )

    hf_kernels.get_kernel = _bundled_get_kernel
    hf_load.get_kernel = _bundled_get_kernel
    try:
        yield active_resolver
    finally:
        hf_kernels.get_kernel = orig_top
        hf_load.get_kernel = orig_load
