"""Tests for the Hugging Face resolver protocol using preselected local variants."""
from pathlib import Path
from unittest.mock import patch


def test_bundle_resolver_uses_exact_dependency_and_preselected_path():
    try:
        from kernels._rust import KernelDependency, KernelVersion
        from kernels.resolver import Resolver
    except ImportError:
        import _skip
        raise _skip.Skip("optional Hugging Face kernels package is not installed")
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


if __name__ == "__main__":
    import _skip
    raise SystemExit(_skip.run_tests(
        [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")],
        suite="test_kernel_bundle",
    ))
