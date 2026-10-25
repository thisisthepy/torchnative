"""Issue #54: an op defined through `torch.library` keeps the tags it was
defined with. `torch/_prims/__init__.py` defines every prim with
`pt2_compliant_tag`; `_DispatchLibrary.define` used to drop the argument, so
`torch.ops.prims.sin.default.tags` was `[]` here and `[pt2_compliant_tag]`
upstream. Run in the vendored tree, in a subprocess, like every probe.
"""
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
VENDOR = ROOT / "torchnative" / "python"

_PROBE = r"""
import json, torch, torch._prims
assert hasattr(torch._C, "_aten_implemented"), "upstream torch, not the shim"
out = {}
for name in ("sin", "cos", "neg", "clone", "transpose", "broadcast_in_dim"):
    out[name] = sorted(t.name for t in getattr(torch.ops.prims, name).default.tags)
out["aten.add.Tensor"] = sorted(t.name for t in torch.ops.aten.add.Tensor.tags)
# The closed registry (issue #54): upstream has no aten op by these names.
out["absent"] = [n for n in ("broadcast_in_dim", "split_dim", "view_of", "not_an_op")
                 if getattr(torch.ops.aten, n, None) is not None]
# Names outside the yaml that upstream does have still resolve.
out["outside_yaml"] = sorted(torch.ops.aten.uniform.overloads())
out["jit_only"] = hasattr(torch.ops.aten, "_list_to_tensor")
print(json.dumps(out))
"""


def test_prims_keep_their_defined_tags():
    if not (VENDOR / "torch" / "__init__.py").is_file():
        print("SKIP test_prims_keep_their_defined_tags: vendored tree not built")
        return
    env = dict(os.environ, PYTHONPATH=str(VENDOR), TORCH_USE_RTLD_GLOBAL="1")
    proc = subprocess.run([sys.executable, "-c", _PROBE], capture_output=True,
                          text=True, env=env, timeout=300)
    assert proc.returncode == 0, proc.stderr[-3000:]
    got = json.loads(proc.stdout.strip().splitlines()[-1])
    for name in ("sin", "cos", "neg", "clone", "transpose", "broadcast_in_dim"):
        assert "pt2_compliant_tag" in got[name], (name, got[name])
    # The yaml path still answers for aten.
    assert "pt2_compliant_tag" in got["aten.add.Tensor"], got["aten.add.Tensor"]
    assert "core" in got["aten.add.Tensor"], got["aten.add.Tensor"]
    assert got["absent"] == [], got["absent"]
    assert got["outside_yaml"] == ["default", "out"], got["outside_yaml"]
    assert got["jit_only"] is True


if __name__ == "__main__":
    test_prims_keep_their_defined_tags()
    print("ok   test_prims_keep_their_defined_tags")
