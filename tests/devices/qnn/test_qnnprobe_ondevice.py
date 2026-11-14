"""The three probe defects of #21, fixed in #217, and what still refuses.

Measured on the Galaxy Tab S9 Ultra (SM-X910, SM8550, Android 16 / API 36,
QNN 2.37) on 2026-11-07 and recorded on #21:

1. `SOC_PROPERTIES` read `ro.soc.manufacturer` first. It reads `QTI`, a vendor
   name, so the probe reported `soc_raw=QTI`, `not-in-table`, `htp_arch=null`
   for a chipset the table knows.
2. `htp_reachable()` said no because `/dev/cdsprpc-smd` is absent, while
   `qnn-platform-validator --backend dsp --testBackend` passed through
   `/dev/adsprpc-smd` (V73), and failed with the skeleton hidden.
3. On the device itself the probe asked for adb and `ANDROID_SERIAL`, so
   `model.to(torchnative.device.npu)` there raised "no adb on PATH".

No device, executorch or torch build is needed: the shell transport, the SoC
table and the FastRPC library are replaced with fakes, because the property
under test is what the probe concludes from evidence. The two validator logs
below are the recorded output, not invented text.

Nullifications this file is meant to catch:

* `ro.soc.manufacturer` put back first in `SOC_PROPERTIES`
      -> `test_the_part_number_is_read_before_the_vendor_name`
* the runtime probe's verdict ignored (adsprpc alone counted, or never counted)
      -> `test_adsprpc_counts_only_with_a_runtime_yes`
* the validator verdict read from the exit code or from one line only
      -> `test_validator_logs_parse_both_ways`
* `device_report` gating on adb when running on the device
      -> `test_on_device_the_probe_needs_no_adb`
* the measured architecture not used when the table is unavailable
      -> `test_on_device_the_measured_arch_fills_a_missing_table`
* a table/runtime architecture mismatch accepted
      -> `test_an_arch_mismatch_refuses`
"""

import os
import sys
import _skip

_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "python")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")


#: `qnn-platform-validator --backend dsp --testBackend`, Tab S9 Ultra,
#: 2026-11-07, with the V73 stub/skel and calculator pair staged. Exit 0.
VALIDATOR_PASS = """\
PF_VALIDATOR: DEBUG: dlOpen successfull for library : libadsprpc.so
PF_VALIDATOR: DEBUG: dlOpen successfull for library : libcdsprpc.so
Backend DSP Prerequisites: Present.
PF_VALIDATOR: ERROR: The fastRPC library version is not implemented yet.

Library version of the backend DSP: Not Found.
PF_VALIDATOR: DEBUG: Calling PlatformValidator->getCoreVersion
Core Version of the backend DSP: Hexagon Architecture V73
PF_VALIDATOR: DEBUG: Loading sample stub: libQnnHtpV73CalculatorStub.so
PF_VALIDATOR: DEBUG: Success in executing the sum function
Unit Test on the backend DSP: Passed.
QNN is supported for backend DSP on the device.
*********** Results Summary ***********
Backend = DSP
{
  Backend Hardware  : Supported
  Backend Libraries : Found
  Library Version   : Not Found
  Core Version      : Hexagon Architecture V73
  Unit Test         : Passed
}
"""

#: The negative control, same device and day, skeleton hidden. Also exit 0.
VALIDATOR_FAIL = """\
PF_VALIDATOR: ERROR: Also make sure ADSP_LIBRARY_PATH points to directory containing skels.
Unit Test on the backend DSP: Failed.
QNN is NOT supported for backend DSP on the device.
*********** Results Summary ***********
Backend = DSP
{
  Backend Hardware  : Supported
  Backend Libraries : Found
  Library Version   : Not Queried
  Core Version      : Not Queried
  Unit Test         : Failed
}
"""

#: The tablet's properties and nodes as recorded on #21.
_PROPS = {
    "ro.soc.manufacturer": "QTI",
    "ro.soc.model": "SM8550",
    "ro.board.platform": "kalama",
    "ro.hardware": "qcom",
    "ro.product.cpu.abi": "arm64-v8a",
}


def _modules():
    if not os.path.isdir(os.path.join(_VENDOR_DIR, "torchnative", "export")):
        return None
    import importlib

    return (
        importlib.import_module("torchnative.export.qnn_device"),
        importlib.import_module("torchnative.export.qnn"),
        importlib.import_module("torchnative.device"),
    )


class _FakeDevice:
    """A shell that answers like the tablet: props, nodes, staged files."""

    def __init__(self, props=_PROPS, nodes=("/dev/adsprpc-smd", "/dev/adsprpc-smd-secure"),
                 sysfs=("adsprpc-smd", "adsprpc-smd-secure"), validator=None):
        self.props = dict(props)
        self.nodes = set(nodes)
        self.sysfs = tuple(sysfs)
        self.validator = validator
        self.commands = []

    def __call__(self, command, timeout=300):
        self.commands.append(command)
        if command.startswith("getprop "):
            return self.props.get(command.split(None, 1)[1].strip(), "") + "\n"
        if command.startswith("ls -1 /sys/class/fastrpc"):
            return "\n".join(self.sysfs) + "\n"
        if command.startswith("ls -d "):
            path = command.split()[2]
            if path in self.nodes:
                return path + "\n"
            if path.endswith("/qnn-platform-validator") and self.validator is not None:
                return path + "\n"
            return ""
        if command.startswith("ls -1 /data/local/tmp/bw_qnn"):
            return ""
        if "qnn-platform-validator --backend dsp" in command:
            # Measured on the Flip 4: without --coreVersion the arch is
            # "Not Queried", and without --targetPath the tool writes outside
            # DEVICE_DIR.
            assert "--coreVersion" in command, command
            assert "--targetPath /data/local/tmp/" in command, command
            return self.validator or ""
        raise AssertionError(f"unexpected device command: {command!r}")


class _Patched:
    """Swap attributes for the duration of a `with`, restore them after."""

    def __init__(self, *triples):
        self.triples = triples
        self.saved = []

    def __enter__(self):
        for obj, name, value in self.triples:
            self.saved.append((obj, name, getattr(obj, name)))
            setattr(obj, name, value)
        return self

    def __exit__(self, *exc):
        for obj, name, value in reversed(self.saved):
            setattr(obj, name, value)


def _fake_table(Q):
    """`qnn.resolve_soc` / `soc_targets` without executorch: SM8550 -> V73 only."""

    class _Member:
        name = "SM8550"

    def resolve(raw):
        if raw.strip().upper() == "SM8550":
            return _Member, 43, 73
        raise Q.QnnRefused(f"{raw!r} not in the fake table")

    return resolve, (lambda: {"SM8550": (43, 73)})


def _no_table(Q):
    def refuse(*_a, **_k):
        raise Q.QnnRefused("executorch is not importable here")

    return refuse, refuse


def test_the_part_number_is_read_before_the_vendor_name():
    mods = _modules()
    if mods is None:
        _skip.skip("   (skipped: no vendored torchnative tree)")
        return
    QD, Q, _D = mods
    assert "ro.soc.manufacturer" not in QD.SOC_PROPERTIES, QD.SOC_PROPERTIES
    assert QD.SOC_PROPERTIES[0] == "ro.soc.model", QD.SOC_PROPERTIES
    resolve, targets = _fake_table(Q)
    dev = _FakeDevice()
    with _Patched((QD, "shell", dev), (Q, "resolve_soc", resolve), (Q, "soc_targets", targets)):
        prop, raw, status, chipset, _model, arch = QD.device_soc()
    assert (prop, raw, status, chipset, arch) == ("ro.soc.model", "SM8550", QD.SOC_KNOWN, "SM8550", 73), (
        prop, raw, status, chipset, arch)
    # Exynos is still recognised from the manufacturer property.
    dev = _FakeDevice(props=dict(_PROPS, **{"ro.soc.manufacturer": "Samsung", "ro.soc.model": "s5e9925"}))
    with _Patched((QD, "shell", dev), (Q, "resolve_soc", resolve), (Q, "soc_targets", targets)):
        prop, raw, status, *_ = QD.device_soc()
    assert status == QD.SOC_EXYNOS_UNIMPLEMENTED and prop == "ro.soc.manufacturer", (prop, raw, status)
    print("ok   qnnprobe_ondevice: ro.soc.model is read before the vendor name; Exynos still recognised")


def test_validator_logs_parse_both_ways():
    mods = _modules()
    if mods is None:
        _skip.skip("   (skipped: no vendored torchnative tree)")
        return
    QD = mods[0]
    assert QD.parse_platform_validator(VALIDATOR_PASS) == (True, 73, None)
    passed, arch, why = QD.parse_platform_validator(VALIDATOR_FAIL)
    assert passed is False and arch is None, (passed, arch)
    assert "ADSP_LIBRARY_PATH" in why, why
    # One verdict line without the other is not a pass.
    half = VALIDATOR_PASS.replace("QNN is supported for backend DSP on the device.\n", "")
    assert QD.parse_platform_validator(half)[0] is not True
    # Output with no verdict at all is a refusal to answer, not a no.
    passed, _arch, why = QD.parse_platform_validator("Segmentation fault\n")
    assert passed is None and why, (passed, why)
    print("ok   qnnprobe_ondevice: validator verdicts read from its text, pass, fail and silent")


def test_adsprpc_counts_only_with_a_runtime_yes():
    mods = _modules()
    if mods is None:
        _skip.skip("   (skipped: no vendored torchnative tree)")
        return
    QD = mods[0]
    no_fastrpc = lambda lib=None: (None, None, None, "not on the device")  # noqa: E731
    # adsprpc only, nothing staged, no FastRPC query possible: still no.
    with _Patched((QD, "shell", _FakeDevice()), (QD, "fastrpc_dsp_info", no_fastrpc)):
        ok, why = QD.htp_reachable()
    assert ok is False and "cdsprpc" in why and "No runtime probe" in why, (ok, why)
    # The recorded passing validator: yes.
    with _Patched((QD, "shell", _FakeDevice(validator=VALIDATOR_PASS)), (QD, "fastrpc_dsp_info", no_fastrpc)):
        ok, why = QD.htp_reachable()
        probe = QD.htp_runtime_probe()
    assert ok is True and why is None, (ok, why)
    assert probe["method"] == "platform-validator" and probe["arch"] == 73, probe
    # The recorded negative control: no, and it says the probe answered no.
    with _Patched((QD, "shell", _FakeDevice(validator=VALIDATOR_FAIL)), (QD, "fastrpc_dsp_info", no_fastrpc)):
        ok, why = QD.htp_reachable()
    assert ok is False and "answered no" in why, (ok, why)
    # FastRPC's own answer, when that is the only one.
    yes = lambda lib=None: (True, 69, True, None)  # noqa: E731
    with _Patched((QD, "shell", _FakeDevice()), (QD, "fastrpc_dsp_info", yes)):
        ok, _why = QD.htp_reachable()
        probe = QD.htp_runtime_probe()
    assert ok is True and probe["method"] == "fastrpc-dsp-info" and probe["arch"] == 69, probe
    print("ok   qnnprobe_ondevice: adsprpc counts only when a runtime probe answers yes")


def test_fastrpc_dsp_info_decodes_the_capability_query():
    mods = _modules()
    if mods is None:
        _skip.skip("   (skipped: no vendored torchnative tree)")
        return
    QD = mods[0]
    import ctypes

    class _Fn:
        def __init__(self, answers, rc=0):
            self.answers, self.rc, self.calls = answers, rc, []

        def __call__(self, req, ptr, size):
            cap = ctypes.cast(ptr, ctypes.POINTER(QD._RemoteDspCapability)).contents
            self.calls.append((req, cap.domain, cap.attribute_ID, size))
            cap.capability = self.answers.get(cap.attribute_ID, 0)
            return self.rc

    class _Lib:
        def __init__(self, fn):
            self.remote_handle_control = fn

    fn = _Fn({0: 1, 6: 0x73, 7: 2})
    assert QD.fastrpc_dsp_info(_Lib(fn)) == (True, 73, True, None)
    assert all(c[0] == 2 and c[1] == 3 and c[3] == 12 for c in fn.calls), fn.calls
    assert [c[2] for c in fn.calls] == [0, 6, 7], fn.calls
    # The Galaxy Z Flip 4 (SM8475, 2026-11-13): DOMAIN_SUPPORT answers 0 for
    # every domain, ARCH_VER answers 0x1008c69 for the CDSP, HMX depth 0.
    # That is a V69 compute DSP, not a "not supported".
    assert QD.fastrpc_dsp_info(_Lib(_Fn({0: 0, 6: 0x1008C69, 7: 0}))) == (True, 69, None, None)
    # Nothing identifies a compute DSP: no answer, never a no.
    supported, *_rest, why = QD.fastrpc_dsp_info(_Lib(_Fn({0: 0})))
    assert supported is None and "DOMAIN_SUPPORT=0 is not read as a no" in why, why
    supported, *_rest, why = QD.fastrpc_dsp_info(_Lib(_Fn({0: 1}, rc=0x80000406)))
    assert supported is None and "0x80000406" in why, (supported, why)
    supported, *_rest, why = QD.fastrpc_dsp_info(_Lib(_Fn({}, rc=0x4E)))
    assert supported is None and "0x4e" in why, why
    # Off the device, no library is guessed at.
    with _Patched((QD, "on_device", lambda: False)):
        assert QD.fastrpc_dsp_info()[0] is None
    print("ok   qnnprobe_ondevice: the FastRPC capability query asks CDSP and decodes V73 / HMX")


def test_on_device_the_probe_needs_no_adb():
    mods = _modules()
    if mods is None:
        _skip.skip("   (skipped: no vendored torchnative tree)")
        return
    QD, Q, _D = mods
    resolve, targets = _fake_table(Q)
    dev = _FakeDevice(validator=VALIDATOR_PASS)
    saved_serial = os.environ.pop("ANDROID_SERIAL", None)
    try:
        with _Patched((QD, "on_device", lambda: True), (QD, "_local_sh", dev),
                      (QD, "adb", lambda *a, **k: (_ for _ in ()).throw(AssertionError("adb used on the device"))),
                      (QD.shutil, "which", lambda name: None),
                      (Q, "resolve_soc", resolve), (Q, "soc_targets", targets)):
            report = QD.device_report()
    finally:
        if saved_serial is not None:
            os.environ["ANDROID_SERIAL"] = saved_serial
    assert report["reachable"] is True and report["transport"] == "local", report
    assert report["htp_arch"] == 73 and report["htp_reachable"] is True, report
    assert any(c.startswith("getprop ") for c in dev.commands), dev.commands
    # And on a host, with no adb, it still refuses by name.
    with _Patched((QD, "on_device", lambda: False), (QD.shutil, "which", lambda name: None)):
        report = QD.device_report()
    assert report["reachable"] is False and "adb" in report["reason"], report
    print("ok   qnnprobe_ondevice: on the device the probe runs locally, with no adb")


def test_on_device_the_measured_arch_fills_a_missing_table():
    mods = _modules()
    if mods is None:
        _skip.skip("   (skipped: no vendored torchnative tree)")
        return
    QD, Q, D = mods
    refuse, refuse_targets = _no_table(Q)
    dev = _FakeDevice(validator=VALIDATOR_PASS)
    # The app ships the QNN runtime (route (a), #260); its absence is
    # test_qnnlibs.py's subject.
    present = lambda arch, **k: {n: f"/app/lib/{n}" for n in QD.qnn_library_names(arch)}  # noqa: E731
    with _Patched((QD, "on_device", lambda: True), (QD, "_local_sh", dev),
                  (QD, "find_qnn_libraries", present),
                  (Q, "resolve_soc", refuse), (Q, "soc_targets", refuse_targets)):
        report = QD.device_report()
    assert report["soc_status"] == QD.SOC_TABLE_UNAVAILABLE, report
    assert report["htp_arch"] == 73 and report["htp_arch_source"] == "runtime:platform-validator", report
    assert "libQnnHtpV73Skel.so" in report["htp_libraries"], report
    with _Patched((QD, "device_report", lambda: dict(report))):
        res = D.npu._resolve_qnn("android", "qnn", "Qualcomm Hexagon NPU")
    assert res.unit == "Qualcomm Hexagon NPU", res
    # With the negative control, nothing is filled in and the resolver refuses.
    dev = _FakeDevice(validator=VALIDATOR_FAIL)
    with _Patched((QD, "on_device", lambda: True), (QD, "_local_sh", dev),
                  (QD, "fastrpc_dsp_info", lambda lib=None: (None, None, None, "fake")),
                  (Q, "resolve_soc", refuse), (Q, "soc_targets", refuse_targets)):
        report = QD.device_report()
    assert report["htp_arch"] is None and report["htp_reachable"] is False, report
    with _Patched((QD, "device_report", lambda: dict(report))):
        try:
            D.npu._resolve_qnn("android", "qnn", "Qualcomm Hexagon NPU")
        except D.NpuUnresolved:
            pass
        else:
            raise AssertionError("resolved with a failed runtime probe and no table")
    print("ok   qnnprobe_ondevice: the DSP's own V73 fills a missing table, and only on a yes")


def test_an_arch_mismatch_refuses():
    mods = _modules()
    if mods is None:
        _skip.skip("   (skipped: no vendored torchnative tree)")
        return
    QD, Q, D = mods
    resolve, targets = _fake_table(Q)
    dev = _FakeDevice(validator=VALIDATOR_PASS.replace("V73", "V69"))
    with _Patched((QD, "shell", dev), (Q, "resolve_soc", resolve), (Q, "soc_targets", targets),
                  (QD, "adb_available", lambda: True)):
        report = QD.device_report()
    assert report["htp_arch"] == 73 and report["htp_arch_runtime"] == 69, report
    with _Patched((QD, "device_report", lambda: dict(report))):
        try:
            D.npu._resolve_qnn("android", "qnn", "Qualcomm Hexagon NPU")
        except D.NpuUnresolved as exc:
            assert "V73" in str(exc) and "V69" in str(exc), exc
        else:
            raise AssertionError("a V73 table answer and a V69 DSP answer resolved")
    print("ok   qnnprobe_ondevice: a table / DSP architecture mismatch refuses by name")


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
                print(f"SKIP test_qnnprobe_ondevice: {name} -- {_skip._state['skipped'][name]}")
        _skip._state["current"] = None
    raise SystemExit(1 if failures else 0)
