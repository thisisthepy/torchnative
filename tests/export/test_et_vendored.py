"""exir's passes, vendored, lower to executorch 1.4.1's bytes without executorch (issue #423, step R5-V).

Owner, 2026-11-21 (asked by the leader, option (c)): "Rust로 재구현": the serializer
step is re-implemented in Rust (R5-S1, `test_et_serialize.py`) and only exir's
passes are vendored, so no executorch Python package goes into the wheel. The tree
is `torchnative/python/torchnative/_vendor/executorch`, written by
`scripts/vendor/vendor_exir.py`; its manifest and import rules are checked by
`tests/release/test_exir_vendor.py`.

**Grade (AGENTS section 16): agrees, byte for byte.** The specification is the
program executorch 1.4.1 itself writes for the same model and options.

How. `tests/_support/et_vendored_probe.py` runs twice in the interpreter named by
`TORCHNATIVE_EXIR_PYTHON` (executorch 1.4.1 installed, the vendored torch tree on
`PYTHONPATH`, so `torch` is the shim: issue #306): once lowering with the installed
executorch, once with `torchnative._vendor.executorch` while `executorch`,
`flatbuffers`, `ruamel`, `tabulate` and `torchao` are absent (`sys.modules[name] =
None`) and flatc cannot start. The eight lowerings are `et_serialize_probe.py`'s.

**What makes each test able to fail.**

* `test_the_vendored_lowering_runs_without_executorch`: an un-renamed
  `import executorch...` anywhere on the path fails the lowering (the package is
  absent) and is listed; a module of the installed executorch that got loaded
  anyway is listed; a flatc start is listed. At least the four portable lowerings
  must complete, so the check cannot pass on zero lowerings.
* `test_every_portable_program_equals_executorchs`: one byte of difference in a
  portable program (a pass that changed, a trimmed import that changed a path, the
  Rust serializer) fails and names the first offset.
* `test_every_xnnpack_program_equals_executorchs`: the same for the four XNNPACK
  lowerings where `torch._C._et_serialize_xnnpack` is built (macOS arm64). Where it
  is not, each must be refused naming `_et_serialize_xnnpack`, never answered.

**What this suite cannot find.** Passes no lowering here reaches (quantised
graphs, dynamic shapes, other partitioners). Whether the wheel carries the tree:
that is `tests/release/test_exir_vendor.py` and the wheel job. Android: R5-D.
"""

import base64
import json
import os
import pathlib
import platform
import subprocess
import sys

_TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]
_SUPPORT_DIR = _TESTS_DIR / "_support"
if str(_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPPORT_DIR))

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_VENDOR_DIR = str(_REPO_ROOT / "torchnative" / "python")
_PROBE = str(_SUPPORT_DIR / "et_vendored_probe.py")
_MARK = "ET_VENDORED_JSON "

import _skip  # noqa: E402

_cache = {}


def _on_macos_arm64():
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _exir_python():
    path = os.environ.get("TORCHNATIVE_EXIR_PYTHON")
    return path if path and os.path.isfile(path) else None


def _run(mode):
    scratch = _REPO_ROOT / ".scratch"
    scratch.mkdir(exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = _VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    env["TMPDIR"] = str(scratch)
    proc = subprocess.run([_exir_python(), _PROBE, mode], env=env, cwd=str(_SUPPORT_DIR),
                          capture_output=True, text=True, timeout=1800)
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith(_MARK)), None)
    if line is None:
        raise AssertionError(f"the {mode} probe exited {proc.returncode} without a report; stderr tail:\n"
                             + "\n".join(proc.stderr.splitlines()[-25:]))
    report = json.loads(line[len(_MARK):])
    assert "fatal" not in report, report.get("fatal")
    assert report["shim"], f"the {mode} probe imported upstream torch, not the shim"
    assert report["pte_binding"], f"the {mode} probe's torch._C has no _et_serialize_pte"
    return report


def _reports():
    """(upstream, vendored), or None after a skip."""
    if not os.path.isfile(os.path.join(_VENDOR_DIR, "torch", "__init__.py")):
        _skip.skip("no vendored torch tree")
        return None
    if _exir_python() is None:
        _skip.skip("TORCHNATIVE_EXIR_PYTHON unset or not a file: no interpreter with executorch "
                   "(the gate builds one with executorch==1.4.1; issue #306)")
        return None
    if "reports" not in _cache:
        _cache["reports"] = (_run("upstream"), _run("vendored"))
    return _cache["reports"]


def _compare(names):
    up, ven = _reports()
    bad = {}
    for name in names:
        u, v = up["lowered"][name], ven["lowered"][name]
        assert u["ok"], f"executorch itself failed to lower {name}: {u}"
        if not v["ok"]:
            bad[name] = {k: v[k] for k in ("error", "where")}
            continue
        want, got = base64.b64decode(u["b64"]), base64.b64decode(v["b64"])
        if want != got:
            n = min(len(want), len(got))
            bad[name] = {"upstream_bytes": len(want), "vendored_bytes": len(got),
                         "first_difference": next((i for i in range(n) if want[i] != got[i]), n)}
        else:
            print(f"ok   et_vendored: {name}: {len(got)} bytes, equal to executorch's")
    assert not bad, json.dumps(bad, indent=1)


def test_the_vendored_lowering_runs_without_executorch():
    reports = _reports()
    if reports is None:
        return
    _, ven = reports
    assert not ven["executorch_modules"], ven["executorch_modules"]
    assert not [m for m in ven["refused_imports"] if m.split(".")[0] == "executorch"], ven["refused_imports"]
    assert not ven["flatc_attempts"], ven["flatc_attempts"]
    assert ven["vendored_modules"] > 0, "no torchnative._vendor.executorch module was imported"
    portable = [n for n in ven["lowered"] if n.startswith("portable_")]
    assert len(portable) == 4 and all(ven["lowered"][n]["ok"] for n in portable), \
        {n: ven["lowered"][n] for n in portable if not ven["lowered"][n]["ok"]}
    print(f"ok   et_vendored: {ven['vendored_modules']} vendored modules, no executorch; "
          f"guarded third-party imports refused: {ven['refused_imports']}")


def test_every_portable_program_equals_executorchs():
    if _reports() is None:
        return
    _compare([n for n in _reports()[0]["lowered"] if n.startswith("portable_")])


def test_every_xnnpack_program_equals_executorchs():
    reports = _reports()
    if reports is None:
        return
    up, ven = reports
    names = [n for n in up["lowered"] if n.startswith("xnnpack_")]
    assert len(names) == 4
    if ven["xnnpack_binding"]:
        _compare(names)
        return
    assert not _on_macos_arm64(), "macOS arm64 builds the runtime, so _et_serialize_xnnpack must be there"
    for name in names:
        v = ven["lowered"][name]
        assert not v["ok"] and "_et_serialize_xnnpack" in v["error"], (name, v)
    print("ok   et_vendored: no _et_serialize_xnnpack here; the four XNNPACK lowerings refuse by name")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        _skip._state["current"] = name
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            if name in _skip._state["skipped"]:
                print(f"SKIP test_et_vendored: {name} -- {_skip._state['skipped'][name]}")
        _skip._state["current"] = None
    raise SystemExit(1 if failures else 0)
