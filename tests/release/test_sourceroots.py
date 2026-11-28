"""`build.py`'s freshness check must accept the host shim its own install built.

`artefact_verdict` compares an artefact's cargo dep-info against the source on
disk, and an input outside `_source_roots()` makes the verdict UNKNOWN ("built
from a different checkout"), which stops the wheel build. The roots were the
crate and its `[patch]` paths. Since the `executorch` feature (#315),
`install_shim.sh` builds the macOS arm64 host shim with the path dependency
`../executorch_rt`, whose build script reads the committed sources under
`torchnative/rust/vendor/` (cpuinfo, xnnpack, ...). Those were not roots, so
every wheel build after a default `install_shim.sh` on macOS arm64 refused
(measured 2026-11-25 building the cp315t Android wheel, #437).

What this cannot see: a dep-info that lists an input under a root by a path
that does not resolve there; `artefact_verdict` realpaths each input, so the
roots are compared in the same form here.
"""

import os
import pathlib
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "_support"))
import _skip  # noqa: E402

REPO = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
sys.path.insert(0, str(REPO / "scripts" / "wheel"))

import build  # noqa: E402

RUST = REPO / "torchnative" / "rust"


def test_the_path_dependencies_and_the_committed_vendor_tree_are_roots():
    roots = build._source_roots()
    for want in (RUST / "torch_c", RUST / "vendor" / "candle-core", RUST / "executorch_rt",
                 RUST / "vendor"):
        assert os.path.realpath(want) in roots, (want, roots)


def test_the_roots_stay_declared_and_inside_the_repository():
    repo = os.path.realpath(REPO)
    for root in build._source_roots():
        assert root.startswith(repo + os.sep), root
        assert root != repo, "the whole repository is not a source root"


def test_a_stray_corner_of_the_repository_is_still_foreign():
    roots = build._source_roots()
    stray = os.path.realpath(REPO / "tests")
    assert not any(stray == r or stray.startswith(r + os.sep) for r in roots), roots


if __name__ == "__main__":
    raise SystemExit(_skip.run_tests(
        [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)],
        "release/test_sourceroots",
    ))
