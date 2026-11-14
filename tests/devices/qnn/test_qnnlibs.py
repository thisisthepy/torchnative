"""QNN libraries, route (a): never shipped, found at run time, refused by name (#260).

Owner, 2026-11-13 (relayed by the leader): "(a) 로 가자". torchnative's wheel
does not contain `libQnnHtp*.so`; on the device it finds the Qualcomm runtime
libraries already there or inside the app (an Android app adds
`com.qualcomm.qti:qnn-runtime` from Maven Central), and refuses by name when
they are absent.

No device or torch build is needed: the filesystem, the environment and
`/proc/self/maps` are passed in, because the property under test is where the
search looks and what it says when it finds nothing.

Nullifications this file is meant to catch:
* the resolver's library gate removed
      -> `test_on_the_device_npu_refuses_without_the_libraries`
* the app's native-library directory not searched
      -> `test_the_apps_own_native_library_directory_is_searched`
* a QNN library added to a wheel build script
      -> `test_no_wheel_build_script_ships_a_qnn_library`
"""

import os
import pathlib
import re
import sys
import _skip

_ROOT = pathlib.Path(next(p for p in pathlib.Path(__file__).resolve().parents if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = str(_ROOT / "torchnative" / "python")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

#: A PythonMultiplatform-style app: libpython and the AAR's .so files extracted
#: into the same native-library directory.
_APP_LIB = "/data/app/~~abc/org.example.app-xyz/lib/arm64"
_MAPS = (
    f"7a00000000-7a00100000 r-xp 00000000 fd:05 123 {_APP_LIB}/libpython3.13.so\n"
    "7b00000000-7b00010000 r-xp 00000000 fd:05 456 /system/lib64/libc.so\n"
    "7c00000000-7c00010000 rw-p 00000000 00:00 0 [anon:scudo]\n"
)
_V69 = ("libQnnHtp.so", "libQnnSystem.so", "libQnnHtpV69Stub.so", "libQnnHtpV69Skel.so")


def _qd():
    if not os.path.isdir(os.path.join(_VENDOR_DIR, "torchnative", "export")):
        return None
    import importlib

    return importlib.import_module("torchnative.export.qnn_device")


def test_the_apps_own_native_library_directory_is_searched():
    QD = _qd()
    if QD is None:
        _skip.skip("   (skipped: no vendored torchnative tree)")
        return
    files = {f"{_APP_LIB}/{n}" for n in _V69}
    found = QD.find_qnn_libraries(69, environ={}, maps_text=_MAPS, exists=files.__contains__)
    assert found == {n: f"{_APP_LIB}/{n}" for n in _V69}, found
    assert QD.qnn_library_names(69) == _V69
    print("ok   qnnlibs: the app's own native-library directory (from /proc/self/maps) is searched")


def test_search_order_explicit_dir_first():
    QD = _qd()
    if QD is None:
        _skip.skip("   (skipped: no vendored torchnative tree)")
        return
    env = {"TORCHNATIVE_QNN_LIB_DIR": "/opt/qnn", "ADSP_LIBRARY_PATH": "/dsp/a;/dsp/b",
           "LD_LIBRARY_PATH": "/ld/one:/ld/two"}
    dirs = QD.qnn_library_search_dirs(environ=env, maps_text=_MAPS)
    assert dirs[:6] == ["/opt/qnn", "/dsp/a", "/dsp/b", "/ld/one", "/ld/two", _APP_LIB], dirs
    assert dirs[6:] == ["/vendor/lib64", "/system/lib64", QD.DEVICE_DIR], dirs
    both = {f"/opt/qnn/{n}" for n in _V69} | {f"{_APP_LIB}/{n}" for n in _V69}
    found = QD.find_qnn_libraries(69, environ=env, maps_text=_MAPS, exists=both.__contains__)
    assert all(p.startswith("/opt/qnn/") for p in found.values()), found
    print("ok   qnnlibs: the explicit directory wins, then the DSP path, LD path, app, vendor, staging")


def test_a_missing_library_is_named_with_the_maven_coordinate():
    QD = _qd()
    if QD is None:
        _skip.skip("   (skipped: no vendored torchnative tree)")
        return
    files = {f"{_APP_LIB}/{n}" for n in _V69 if "Skel" not in n}
    try:
        QD.require_qnn_libraries(69, environ={}, maps_text=_MAPS, exists=files.__contains__)
    except QD.QnnLibrariesMissing as exc:
        text = str(exc)
        assert "libQnnHtpV69Skel.so" in text and "libQnnHtp.so," not in text, text
        assert "com.qualcomm.qti:qnn-runtime" in text and _APP_LIB in text, text
        assert "does not ship" in text and "TORCHNATIVE_QNN_LIB_DIR" in text, text
    else:
        raise AssertionError("a missing skel was not refused")
    print("ok   qnnlibs: a missing library is refused by name, with where it looked and the Maven artefact")


def test_on_the_device_npu_refuses_without_the_libraries():
    QD = _qd()
    if QD is None:
        _skip.skip("   (skipped: no vendored torchnative tree)")
        return
    import importlib

    D = importlib.import_module("torchnative.device")
    base = {
        "transport": "local", "reachable": True, "reason": None, "soc_status": "known",
        "soc_raw": "SM8475", "htp_arch": 69, "htp_reachable": True,
        "htp_unreachable_reason": None, "cdsp_fastrpc": (),
    }
    absent = dict(base, qnn_libraries={n: None for n in _V69}, qnn_libraries_complete=False,
                  qnn_libraries_missing_reason="torchnative qnn_device: ... com.qualcomm.qti:qnn-runtime ...")
    present = dict(base, qnn_libraries={n: f"{_APP_LIB}/{n}" for n in _V69}, qnn_libraries_complete=True)
    saved = QD.device_report
    try:
        QD.device_report = lambda: dict(absent)
        try:
            D.npu._resolve_qnn("android", "qnn", "Qualcomm Hexagon NPU")
        except D.NpuUnresolved as exc:
            assert "qnn-runtime" in str(exc) and "V69" in str(exc), exc
        else:
            raise AssertionError("device.npu resolved on the device with no QNN runtime libraries")
        QD.device_report = lambda: dict(present)
        res = D.npu._resolve_qnn("android", "qnn", "Qualcomm Hexagon NPU")
        assert res.unit == "Qualcomm Hexagon NPU", res
    finally:
        QD.device_report = saved
    print("ok   qnnlibs: on the device, device.npu refuses without the libraries and resolves with them")


def test_no_wheel_build_script_ships_a_qnn_library():
    roots = [_ROOT / "scripts" / "wheel", _ROOT / "setup.py", _ROOT / "pyproject.toml",
             _ROOT / ".github" / "workflows" / "publish-pypi.yml"]
    hits = []
    for root in roots:
        paths = [root] if root.is_file() else sorted(root.rglob("*")) if root.is_dir() else []
        for path in paths:
            if path.is_file() and path.suffix in (".py", ".toml", ".yml", ".yaml", ".sh", ""):
                for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
                    if re.search(r"libQnn|libSnpe|qnn-runtime", line):
                        hits.append(f"{path.relative_to(_ROOT)}:{i}: {line.strip()}")
    assert not hits, "route (a): torchnative ships no Qualcomm library, but:\n" + "\n".join(hits)
    print("ok   qnnlibs: no wheel build script references a Qualcomm library")


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
                print(f"SKIP test_qnnlibs: {name} -- {_skip._state['skipped'][name]}")
        _skip._state["current"] = None
    raise SystemExit(1 if failures else 0)
