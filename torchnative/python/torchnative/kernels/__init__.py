"""Bundle-backed resolver for the Hugging Face ``kernels`` contract.

The contract is adopted as-is; only resolution is inverted. On mobile, kernel
variants are selected and compiled at build time and read from the app bundle,
because iOS will not execute downloaded native code. Desktop keeps resolving
against the Hub. See DESIGN.md §8.
"""


class BundleResolver:
    """Resolve exact HF kernel dependencies to already-selected bundle variants.

    The mapping is prepared by the application build. This resolver never asks
    the supplied Hub API to download anything. The optional ``kernels`` package
    supplies metadata parsing and the returned ``LocalKernel`` implementation.
    """

    def __init__(self, kernel_paths):
        from pathlib import Path

        self._paths = {dependency: Path(path) for dependency, path in kernel_paths.items()}

    def resolve(self, *, api, backend, kernel):
        path = self._paths.get(kernel)
        if path is None:
            return None
        from kernels.resolver import resolve_local_kernel

        return resolve_local_kernel(path, backend=backend)
