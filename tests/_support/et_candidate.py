"""Take Android's ExecuTorch candidate out of `npu` resolution, for tests of the branch behind it.

Since issue #423 Android's candidates are `("executorch", "CPU (ExecuTorch XNNPACK
delegate)")` then `("qnn", "Qualcomm Hexagon NPU")`, and the first resolves wherever
the runtime is built (the macOS arm64 gate, with the host moved to android). A test
whose subject is the QNN resolver (its success branch against a fake probe, its
Exynos and unknown-SoC refusals, the `.to()` refusal) wraps itself in
`no_executorch()`, which makes `_resolve_executorch` refuse by name, so the QNN
branch is what runs. A helper, not a suite.
"""

import contextlib


@contextlib.contextmanager
def no_executorch(D):
    saved = D.NpuDevice._resolve_executorch

    def refuse(self, h, backend, unit):
        raise D.NpuUnresolved(
            f"{backend} ({unit}): taken out of resolution by this test "
            f"(tests/_support/et_candidate.py) so the qnn branch runs")

    D.NpuDevice._resolve_executorch = refuse
    try:
        yield
    finally:
        D.NpuDevice._resolve_executorch = saved
