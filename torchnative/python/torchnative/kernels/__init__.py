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
    "UnsupportedFlashAttentionLayout",
    "UnsupportedKernelBackend",
    "UnsupportedKernelTarget",
    "UnsupportedPagedAttentionLayout",
    "flash_attention",
    "flash_linear_attention",
    "gather_paged_kv",
    "get_kernel",
    "paged_attention",
    "select_mobile_variant",
    "stage_bundle",
    "target_profile",
    "use_bundle",
]


class UnsupportedKernelTarget(ValueError):
    """A target platform has no HF `kernels` profile in this build."""


class UnsupportedKernelBackend(ValueError):
    """A backend is not supported by the HF `kernels` target profile."""


class UnsupportedPagedAttentionLayout(ValueError):
    """A block size, head layout, dtype, or device is not supported by `paged_attention`."""


class UnsupportedFlashAttentionLayout(ValueError):
    """A head layout, rank, mask, dtype, or device is not supported by `flash_attention` / `flash_linear_attention`."""


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
    import kernels.validate as kv

    if resolver is None:
        if bundle_dir is None:
            raise ValueError("torchnative.kernels.get_kernel requires either `resolver` or `bundle_dir`")
        resolver = BundleResolver.from_bundle(bundle_dir)

    kernel_version = revision_or_version(revision=revision, version=version)
    dep = KernelDependency(repo_id=repo_id, version=kernel_version)

    validators = kv.default_metadata_validators()
    if check_arch:
        validators.append(kv.ArchValidator())

    # `kernels` 0.18 splits `AllValidator` into `AllMetadataValidator` and
    # `AllKernelValidator`, while `kernels` <= 0.17 uses `AllValidator` and
    # passes only `metadata_validator` to `get_kernel_with_resolver`.
    if hasattr(kv, "AllMetadataValidator") and hasattr(kv, "AllKernelValidator"):
        return get_kernel_with_resolver(
            api=HfApi(),
            backend=backend,
            kernel=dep,
            resolver=resolver,
            kernel_validator=kv.AllKernelValidator(validators=kv.default_kernel_validators()),
            metadata_validator=kv.AllMetadataValidator(validators=validators),
        )
    return get_kernel_with_resolver(
        api=HfApi(),
        backend=backend,
        kernel=dep,
        resolver=resolver,
        metadata_validator=kv.AllValidator(validators=validators),
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


# ---------------------------------------------------------------------------
# Paged attention kernel on cpu and mps (issue #14, SPEC S7.3)
# ---------------------------------------------------------------------------

_PAGED_DEVICES = ("cpu", "mps")
_PAGED_MIN_BLOCK_SIZE = 1
_PAGED_MAX_BLOCK_SIZE = 1024


def _validate_paged_inputs(
    query,
    key_cache,
    value_cache,
    block_table,
    context_lens=None,
) -> tuple[int, int, int, int, int, int, int, int]:
    import torch

    for name, tensor in (
        ("query", query),
        ("key_cache", key_cache),
        ("value_cache", value_cache),
        ("block_table", block_table),
    ):
        if not isinstance(tensor, torch.Tensor):
            raise UnsupportedPagedAttentionLayout(
                f"torchnative.kernels.paged_attention: `{name}` must be a torch.Tensor, "
                f"got {type(tensor).__name__}"
            )

    if query.dim() != 4:
        raise UnsupportedPagedAttentionLayout(
            f"torchnative.kernels.paged_attention: `query` must be 4-D "
            f"[batch, num_heads, seqlen_q, head_dim], got {query.dim()}-D shape {list(query.shape)}"
        )
    if key_cache.dim() != 4 or value_cache.dim() != 4:
        raise UnsupportedPagedAttentionLayout(
            f"torchnative.kernels.paged_attention: `key_cache` and `value_cache` must be 4-D "
            f"[num_blocks, block_size, num_kv_heads, head_dim], got key_cache={list(key_cache.shape)} "
            f"and value_cache={list(value_cache.shape)}"
        )
    if list(key_cache.shape) != list(value_cache.shape):
        raise UnsupportedPagedAttentionLayout(
            f"torchnative.kernels.paged_attention: `key_cache` shape {list(key_cache.shape)} "
            f"and `value_cache` shape {list(value_cache.shape)} must match"
        )

    dev_type = query.device.type
    if dev_type not in _PAGED_DEVICES:
        raise UnsupportedPagedAttentionLayout(
            f"torchnative.kernels.paged_attention: device {dev_type!r} is not supported; "
            f"supported devices are {_PAGED_DEVICES}"
        )
    for name, tensor in (
        ("key_cache", key_cache),
        ("value_cache", value_cache),
        ("block_table", block_table),
    ):
        if tensor.device.type != dev_type:
            raise UnsupportedPagedAttentionLayout(
                f"torchnative.kernels.paged_attention: `{name}` is on {tensor.device} "
                f"while `query` is on {query.device}"
            )

    allowed_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    if query.dtype not in allowed_dtypes:
        raise UnsupportedPagedAttentionLayout(
            f"torchnative.kernels.paged_attention: unsupported query dtype {query.dtype}; "
            f"supported dtypes are float32, float16, bfloat16"
        )
    if key_cache.dtype != query.dtype or value_cache.dtype != query.dtype:
        raise UnsupportedPagedAttentionLayout(
            f"torchnative.kernels.paged_attention: query dtype ({query.dtype}), "
            f"key_cache dtype ({key_cache.dtype}), and value_cache dtype ({value_cache.dtype}) must match"
        )

    batch, num_heads, seqlen_q, head_dim = query.shape
    num_blocks, block_size, num_kv_heads, kv_head_dim = key_cache.shape

    if batch <= 0 or seqlen_q <= 0 or head_dim <= 0:
        raise UnsupportedPagedAttentionLayout(
            f"torchnative.kernels.paged_attention: invalid query shape {list(query.shape)}"
        )
    if (
        num_blocks <= 0
        or block_size < _PAGED_MIN_BLOCK_SIZE
        or block_size > _PAGED_MAX_BLOCK_SIZE
    ):
        raise UnsupportedPagedAttentionLayout(
            f"torchnative.kernels.paged_attention: unsupported block_size={block_size} "
            f"or num_blocks={num_blocks} in key_cache shape {list(key_cache.shape)}"
        )
    if kv_head_dim != head_dim:
        raise UnsupportedPagedAttentionLayout(
            f"torchnative.kernels.paged_attention: query head_dim ({head_dim}) does not match "
            f"KV cache head_dim ({kv_head_dim})"
        )
    if num_kv_heads <= 0 or num_heads <= 0 or num_heads % num_kv_heads != 0:
        raise UnsupportedPagedAttentionLayout(
            f"torchnative.kernels.paged_attention: query num_heads ({num_heads}) must be a "
            f"positive multiple of num_kv_heads ({num_kv_heads})"
        )

    if block_table.dim() != 2:
        raise UnsupportedPagedAttentionLayout(
            f"torchnative.kernels.paged_attention: `block_table` must be 2-D "
            f"[batch, max_blocks_per_seq], got {block_table.dim()}-D shape {list(block_table.shape)}"
        )
    if block_table.shape[0] != batch or block_table.shape[1] <= 0:
        raise UnsupportedPagedAttentionLayout(
            f"torchnative.kernels.paged_attention: `block_table` shape {list(block_table.shape)} "
            f"does not match query batch size {batch}"
        )
    if block_table.dtype not in (torch.int64, torch.int32):
        raise UnsupportedPagedAttentionLayout(
            f"torchnative.kernels.paged_attention: `block_table` must have dtype int64 or int32, "
            f"got {block_table.dtype}"
        )

    max_blocks_per_seq = block_table.shape[1]
    max_seqlen_k = max_blocks_per_seq * block_size

    if context_lens is not None:
        if not isinstance(context_lens, torch.Tensor):
            raise UnsupportedPagedAttentionLayout(
                f"torchnative.kernels.paged_attention: `context_lens` must be a 1-D torch.Tensor, "
                f"got {type(context_lens).__name__}"
            )
        if context_lens.dim() != 1 or context_lens.shape[0] != batch:
            raise UnsupportedPagedAttentionLayout(
                f"torchnative.kernels.paged_attention: `context_lens` must have shape [{batch}], "
                f"got {list(context_lens.shape)}"
            )
        if context_lens.dtype not in (torch.int64, torch.int32):
            raise UnsupportedPagedAttentionLayout(
                f"torchnative.kernels.paged_attention: `context_lens` must have dtype int64 or int32, "
                f"got {context_lens.dtype}"
            )
        if context_lens.device.type != dev_type:
            raise UnsupportedPagedAttentionLayout(
                f"torchnative.kernels.paged_attention: `context_lens` is on {context_lens.device} "
                f"while `query` is on {query.device}"
            )
        # Validate context_lens bounds on CPU only; on mps we avoid any host readback mid-dispatch.
        if dev_type == "cpu":
            lens_list = [int(x) for x in context_lens.tolist()]
            if any(c <= 0 or c > max_seqlen_k for c in lens_list):
                raise UnsupportedPagedAttentionLayout(
                    f"torchnative.kernels.paged_attention: `context_lens` values {lens_list} "
                    f"must be in 1..={max_seqlen_k} (max_blocks_per_seq={max_blocks_per_seq} * "
                    f"block_size={block_size})"
                )

    if dev_type == "cpu":
        bt_flat = [int(x) for x in block_table.reshape(-1).tolist()]
        if any(b < 0 or b >= num_blocks for b in bt_flat):
            raise UnsupportedPagedAttentionLayout(
                f"torchnative.kernels.paged_attention: `block_table` contains out-of-range "
                f"block index for num_blocks={num_blocks}: {bt_flat}"
            )

    return (
        batch,
        num_heads,
        seqlen_q,
        head_dim,
        num_blocks,
        block_size,
        num_kv_heads,
        max_blocks_per_seq,
    )


def gather_paged_kv(
    key_cache,
    value_cache,
    block_table,
):
    """Gather paged `[num_blocks, block_size, num_kv_heads, head_dim]` caches into `[B, H_kv, S_kv, D]`."""
    import torch

    if key_cache.dim() != 4 or value_cache.dim() != 4:
        raise UnsupportedPagedAttentionLayout(
            "torchnative.kernels.gather_paged_kv: `key_cache` and `value_cache` must be 4-D "
            "[num_blocks, block_size, num_kv_heads, head_dim]"
        )
    num_blocks, block_size, num_kv_heads, head_dim = key_cache.shape
    batch, max_blocks_per_seq = block_table.shape
    seqlen_k = max_blocks_per_seq * block_size

    flat_bt = block_table.reshape(-1)
    if flat_bt.dtype != torch.int64:
        flat_bt = flat_bt.to(torch.int64)

    row_width = block_size * num_kv_heads * head_dim
    k_flat = torch.index_select(key_cache.reshape(num_blocks, row_width), 0, flat_bt)
    v_flat = torch.index_select(value_cache.reshape(num_blocks, row_width), 0, flat_bt)

    k_gathered = k_flat.reshape(batch, seqlen_k, num_kv_heads, head_dim).transpose(1, 2)
    v_gathered = v_flat.reshape(batch, seqlen_k, num_kv_heads, head_dim).transpose(1, 2)
    return k_gathered, v_gathered


def paged_attention(
    query,
    key_cache,
    value_cache,
    block_table,
    *,
    context_lens=None,
    scale: float | None = None,
    is_causal: bool = False,
    attn_mask=None,
):
    """Compute paged attention over a block-allocated KV cache on `cpu` or `mps`.

    All operations execute on the input tensor's device (`cpu` or `mps`) with zero
    host downloads mid-dispatch on `mps`.
    """
    import torch
    import torch.nn.functional as F

    (
        batch,
        num_heads,
        seqlen_q,
        head_dim,
        _num_blocks,
        block_size,
        num_kv_heads,
        max_blocks_per_seq,
    ) = _validate_paged_inputs(
        query, key_cache, value_cache, block_table, context_lens=context_lens
    )

    seqlen_k = max_blocks_per_seq * block_size
    k_gat, v_gat = gather_paged_kv(key_cache, value_cache, block_table)

    n_rep = num_heads // num_kv_heads
    if n_rep > 1:
        k_gat = (
            k_gat[:, :, None, :, :]
            .expand(batch, num_kv_heads, n_rep, seqlen_k, head_dim)
            .reshape(batch, num_heads, seqlen_k, head_dim)
        )
        v_gat = (
            v_gat[:, :, None, :, :]
            .expand(batch, num_kv_heads, n_rep, seqlen_k, head_dim)
            .reshape(batch, num_heads, seqlen_k, head_dim)
        )

    dev = query.device
    dtype = query.dtype
    combined_mask = None

    if context_lens is not None or is_causal:
        # Build 0..seqlen_k-1 on the device using ones + cumsum - 1 so no host buffer
        # upload is needed inside the dispatch bracket.
        ones_k = torch.ones((1, 1, 1, seqlen_k), dtype=torch.int64, device=dev)
        pos_k = torch.cumsum(ones_k, dim=-1) - 1

        valid = None
        if context_lens is not None:
            ctx_i64 = (
                context_lens
                if context_lens.dtype == torch.int64
                else context_lens.to(torch.int64)
            ).reshape(batch, 1, 1, 1)
            valid = pos_k < ctx_i64

        if is_causal:
            ones_q = torch.ones((1, 1, seqlen_q, 1), dtype=torch.int64, device=dev)
            q_idx = torch.cumsum(ones_q, dim=2) - 1
            if context_lens is not None:
                ctx_i64 = (
                    context_lens
                    if context_lens.dtype == torch.int64
                    else context_lens.to(torch.int64)
                ).reshape(batch, 1, 1, 1)
                q_pos = (ctx_i64 - seqlen_q) + q_idx
            else:
                q_pos = q_idx
            causal_valid = pos_k <= q_pos
            valid = causal_valid if valid is None else (valid & causal_valid)

        zero = torch.zeros((1,), dtype=dtype, device=dev)
        neg_inf = torch.full((1,), float("-inf"), dtype=dtype, device=dev)
        combined_mask = torch.where(valid, zero, neg_inf)

    if attn_mask is not None:
        if not isinstance(attn_mask, torch.Tensor) or attn_mask.device.type != dev.type:
            raise UnsupportedPagedAttentionLayout(
                "torchnative.kernels.paged_attention: `attn_mask` must be a Tensor on the same device as `query`"
            )
        if attn_mask.dtype == torch.bool:
            zero = torch.zeros((1,), dtype=dtype, device=dev)
            neg_inf = torch.full((1,), float("-inf"), dtype=dtype, device=dev)
            add_mask = torch.where(attn_mask, zero, neg_inf)
        elif attn_mask.dtype == dtype:
            add_mask = attn_mask
        else:
            raise UnsupportedPagedAttentionLayout(
                f"torchnative.kernels.paged_attention: `attn_mask` dtype ({attn_mask.dtype}) "
                f"must match query dtype ({dtype}) or be bool"
            )
        combined_mask = add_mask if combined_mask is None else (combined_mask + add_mask)

    return F.scaled_dot_product_attention(
        query.contiguous(),
        k_gat.contiguous(),
        v_gat.contiguous(),
        attn_mask=combined_mask,
        dropout_p=0.0,
        is_causal=False,
        scale=scale,
    )


# ---------------------------------------------------------------------------
# Flash-attention and flash-linear-attention kernels on cpu and mps (SPEC S7.3)
# ---------------------------------------------------------------------------


def _validate_flash_qkv(
    fn_name: str,
    query,
    key,
    value,
    *,
    require_equal_seqlen: bool = False,
) -> tuple[int, int, int, int, int, int]:
    import torch

    for name, tensor in (("query", query), ("key", key), ("value", value)):
        if not isinstance(tensor, torch.Tensor):
            raise UnsupportedFlashAttentionLayout(
                f"torchnative.kernels.{fn_name}: `{name}` must be a torch.Tensor, "
                f"got {type(tensor).__name__}"
            )
        if tensor.dim() != 4:
            raise UnsupportedFlashAttentionLayout(
                f"torchnative.kernels.{fn_name}: `{name}` must be 4-D "
                f"[batch, num_heads, seqlen, head_dim], got {tensor.dim()}-D shape {list(tensor.shape)}"
            )

    dev_type = query.device.type
    if dev_type not in _PAGED_DEVICES:
        raise UnsupportedFlashAttentionLayout(
            f"torchnative.kernels.{fn_name}: device {dev_type!r} is not supported; "
            f"supported devices are {_PAGED_DEVICES}"
        )
    for name, tensor in (("key", key), ("value", value)):
        if tensor.device.type != dev_type:
            raise UnsupportedFlashAttentionLayout(
                f"torchnative.kernels.{fn_name}: `{name}` is on {tensor.device} "
                f"while `query` is on {query.device}"
            )

    allowed_dtypes = (torch.float32, torch.float16, torch.bfloat16)
    if query.dtype not in allowed_dtypes:
        raise UnsupportedFlashAttentionLayout(
            f"torchnative.kernels.{fn_name}: unsupported query dtype {query.dtype}; "
            "supported dtypes are float32, float16, bfloat16"
        )
    if key.dtype != query.dtype or value.dtype != query.dtype:
        raise UnsupportedFlashAttentionLayout(
            f"torchnative.kernels.{fn_name}: query dtype ({query.dtype}), "
            f"key dtype ({key.dtype}), and value dtype ({value.dtype}) must match"
        )

    batch, num_heads, seqlen_q, head_dim = query.shape
    k_batch, num_kv_heads, seqlen_k, k_head_dim = key.shape
    v_batch, v_kv_heads, seqlen_v, v_head_dim = value.shape

    if batch <= 0 or seqlen_q <= 0 or seqlen_k <= 0 or head_dim <= 0 or v_head_dim <= 0:
        raise UnsupportedFlashAttentionLayout(
            f"torchnative.kernels.{fn_name}: invalid shapes query={list(query.shape)}, "
            f"key={list(key.shape)}, value={list(value.shape)}"
        )
    if k_batch != batch or v_batch != batch:
        raise UnsupportedFlashAttentionLayout(
            f"torchnative.kernels.{fn_name}: batch size mismatch across query ({batch}), "
            f"key ({k_batch}), and value ({v_batch})"
        )
    if seqlen_v != seqlen_k:
        raise UnsupportedFlashAttentionLayout(
            f"torchnative.kernels.{fn_name}: key seqlen ({seqlen_k}) and value seqlen ({seqlen_v}) must match"
        )
    if require_equal_seqlen and seqlen_q != seqlen_k:
        raise UnsupportedFlashAttentionLayout(
            f"torchnative.kernels.{fn_name}: query seqlen ({seqlen_q}) and key seqlen ({seqlen_k}) must match"
        )
    if k_head_dim != head_dim:
        raise UnsupportedFlashAttentionLayout(
            f"torchnative.kernels.{fn_name}: query head_dim ({head_dim}) does not match "
            f"key head_dim ({k_head_dim})"
        )
    if v_kv_heads != num_kv_heads or num_kv_heads <= 0 or num_heads <= 0 or num_heads % num_kv_heads != 0:
        raise UnsupportedFlashAttentionLayout(
            f"torchnative.kernels.{fn_name}: query num_heads ({num_heads}) must be a positive "
            f"multiple of num_kv_heads ({num_kv_heads}), and key/value head counts must match ({v_kv_heads})"
        )

    return batch, num_heads, num_kv_heads, seqlen_q, seqlen_k, head_dim


def flash_attention(
    query,
    key,
    value,
    *,
    attn_mask=None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    scale: float | None = None,
):
    """Scaled dot-product flash attention over 4-D `[batch, num_heads, seqlen, head_dim]` on `cpu` and `mps`.

    Supports MHA and GQA (`num_heads % num_kv_heads == 0`), causal and explicit
    masks, custom `scale`, and `float32`/`float16`/`bfloat16` with zero host
    downloads mid-dispatch on `mps`.
    """
    import torch
    import torch.nn.functional as F

    batch, num_heads, num_kv_heads, seqlen_q, seqlen_k, head_dim = _validate_flash_qkv(
        "flash_attention", query, key, value
    )
    if dropout_p != 0.0:
        raise UnsupportedFlashAttentionLayout(
            f"torchnative.kernels.flash_attention: non-zero dropout_p ({dropout_p}) is not supported"
        )

    dev = query.device
    dtype = query.dtype
    v_head_dim = value.shape[-1]

    n_rep = num_heads // num_kv_heads
    if n_rep > 1:
        key = (
            key[:, :, None, :, :]
            .expand(batch, num_kv_heads, n_rep, seqlen_k, head_dim)
            .reshape(batch, num_heads, seqlen_k, head_dim)
        )
        value = (
            value[:, :, None, :, :]
            .expand(batch, num_kv_heads, n_rep, seqlen_k, v_head_dim)
            .reshape(batch, num_heads, seqlen_k, v_head_dim)
        )

    if attn_mask is not None:
        if not isinstance(attn_mask, torch.Tensor) or attn_mask.device.type != dev.type:
            raise UnsupportedFlashAttentionLayout(
                "torchnative.kernels.flash_attention: `attn_mask` must be a Tensor on the same device as `query`"
            )
        if attn_mask.dtype not in (torch.bool, dtype):
            raise UnsupportedFlashAttentionLayout(
                f"torchnative.kernels.flash_attention: `attn_mask` dtype ({attn_mask.dtype}) "
                f"must match query dtype ({dtype}) or be bool"
            )

    return F.scaled_dot_product_attention(
        query.contiguous(),
        key.contiguous(),
        value.contiguous(),
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
    )


def flash_linear_attention(
    query,
    key,
    value,
    *,
    decay=None,
    beta=None,
    scale: float | None = None,
    is_causal: bool = True,
    initial_state=None,
    return_final_state: bool = False,
):
    """Linear and gated-delta recurrent attention over 4-D `[batch, num_heads, seqlen, head_dim]` on `cpu` and `mps`.

    Supports MHA and GQA (`num_heads % num_kv_heads == 0`), optional log-decay
    (`decay`) and input gate (`beta`) tensors of shape `[batch, num_heads, seqlen]`,
    optional `initial_state` `[batch, num_heads, head_dim, head_dim_v]`, and
    `return_final_state=True`, executing entirely on device with zero host
    downloads mid-dispatch on `mps`.
    """
    import math
    import torch

    batch, num_heads, num_kv_heads, seqlen, _seqlen_k, head_dim = _validate_flash_qkv(
        "flash_linear_attention", query, key, value, require_equal_seqlen=True
    )
    dev = query.device
    dtype = query.dtype
    v_head_dim = value.shape[-1]

    for gate_name, gate in (("decay", decay), ("beta", beta)):
        if gate is None:
            continue
        if not isinstance(gate, torch.Tensor) or gate.device.type != dev.type:
            raise UnsupportedFlashAttentionLayout(
                f"torchnative.kernels.flash_linear_attention: `{gate_name}` must be a Tensor on {dev}"
            )
        if gate.dtype != dtype:
            raise UnsupportedFlashAttentionLayout(
                f"torchnative.kernels.flash_linear_attention: `{gate_name}` dtype ({gate.dtype}) "
                f"must match query dtype ({dtype})"
            )
        valid_shapes = (
            [batch, num_heads, seqlen],
            [batch, num_heads, seqlen, 1],
        )
        if list(gate.shape) not in valid_shapes:
            raise UnsupportedFlashAttentionLayout(
                f"torchnative.kernels.flash_linear_attention: `{gate_name}` shape {list(gate.shape)} "
                f"must be [{batch}, {num_heads}, {seqlen}] (or with trailing 1)"
            )

    if initial_state is not None:
        if not isinstance(initial_state, torch.Tensor) or initial_state.device.type != dev.type:
            raise UnsupportedFlashAttentionLayout(
                f"torchnative.kernels.flash_linear_attention: `initial_state` must be a Tensor on {dev}"
            )
        if initial_state.dtype != dtype:
            raise UnsupportedFlashAttentionLayout(
                f"torchnative.kernels.flash_linear_attention: `initial_state` dtype ({initial_state.dtype}) "
                f"must match query dtype ({dtype})"
            )
        if list(initial_state.shape) != [batch, num_heads, head_dim, v_head_dim]:
            raise UnsupportedFlashAttentionLayout(
                f"torchnative.kernels.flash_linear_attention: `initial_state` shape {list(initial_state.shape)} "
                f"must be [{batch}, {num_heads}, {head_dim}, {v_head_dim}]"
            )

    n_rep = num_heads // num_kv_heads
    if n_rep > 1:
        key = (
            key[:, :, None, :, :]
            .expand(batch, num_kv_heads, n_rep, seqlen, head_dim)
            .reshape(batch, num_heads, seqlen, head_dim)
        )
        value = (
            value[:, :, None, :, :]
            .expand(batch, num_kv_heads, n_rep, seqlen, v_head_dim)
            .reshape(batch, num_heads, seqlen, v_head_dim)
        )

    calc_dtype = torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype
    q_c = query.to(calc_dtype) if query.dtype != calc_dtype else query
    k_c = key.to(calc_dtype) if key.dtype != calc_dtype else key
    v_c = value.to(calc_dtype) if value.dtype != calc_dtype else value

    sc = (1.0 / math.sqrt(head_dim)) if scale is None else float(scale)
    q_s = q_c * sc

    if decay is not None or beta is not None or initial_state is not None or return_final_state:
        state = (
            initial_state.to(calc_dtype)
            if initial_state is not None
            else torch.zeros((batch, num_heads, head_dim, v_head_dim), dtype=calc_dtype, device=dev)
        )
        dec_c = (
            decay.to(calc_dtype).reshape(batch, num_heads, seqlen, 1, 1)
            if decay is not None
            else None
        )
        bet_c = (
            beta.to(calc_dtype).reshape(batch, num_heads, seqlen, 1)
            if beta is not None
            else None
        )
        outs = []
        for t in range(seqlen):
            if dec_c is not None:
                state = state * torch.exp(dec_c[:, :, t])
            vt = v_c[:, :, t]
            if bet_c is not None:
                vt = vt * bet_c[:, :, t]
            kt = k_c[:, :, t]
            state = state + kt.unsqueeze(-1) * vt.unsqueeze(-2)
            qt = q_s[:, :, t]
            ot = (qt.unsqueeze(-1) * state).sum(dim=-2)
            outs.append(ot)
        out = torch.stack(outs, dim=2)
        if out.dtype != dtype:
            out = out.to(dtype)
        if return_final_state:
            final_state = state.to(dtype) if state.dtype != dtype else state
            return out, final_state
        return out

    scores = torch.matmul(q_s, k_c.transpose(-2, -1))
    if is_causal:
        ones_s = torch.ones((seqlen,), dtype=torch.int64, device=dev)
        idx = torch.cumsum(ones_s, dim=0) - 1
        causal = idx.reshape(1, 1, seqlen, 1) >= idx.reshape(1, 1, 1, seqlen)
        zero = torch.zeros((1,), dtype=calc_dtype, device=dev)
        scores = torch.where(causal, scores, zero)
    out = torch.matmul(scores, v_c)
    if out.dtype != dtype:
        out = out.to(dtype)
    return out

