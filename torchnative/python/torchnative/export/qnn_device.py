"""Ask an attached Snapdragon device what it is, over `adb`. Never assume.

`torchnative.export.nnapi_device` is the shape this follows, including the two
rules that document earned:

* **`ANDROID_SERIAL` only, never "the one attached device".** The devices on
  this machine are shared with other work, and a command that picks its own
  target lands on somebody else's.
* **Write only under one directory this project owns by name**, install
  nothing, and remove what was pushed.

What is different is the question. NNAPI's runtime enumerates its own drivers,
so `nnapi_device.devices()` could ask the *runtime* what it had. QNN has no
such call before a backend is initialised, and initialising one needs the
Qualcomm libraries already staged on the device. So the SoC is read from
Android's own properties and mapped through **ExecuTorch's** chipset table
(`torchnative.export.qnn.resolve_soc`), and the HTP libraries are located by
looking for the files.

Nothing here executes a model. docs/devices/QNN.md §6.4 says why: this round could not
lower a QNN artefact -- the host cannot -- so there is nothing to execute, and
a module that ran *something* and reported success would be reporting on the
CPU path. What this module does is answer the two questions the device
procedure in docs/devices/QNN.md §5 starts with, from the device rather than from a
guess: **which SoC is this** and **is the HTP reachable**.

Those two are answered by different evidence and must not be run together. The
SoC is a *name*, mapped through a table; it describes the part number, and it
reads the same on a unit whose cDSP is fused off, powered down, or not exposed
to userspace at all. Only `htp_reachable` answers the second question, and it
answers it from the FastRPC compute-DSP endpoints -- see `CDSP_FASTRPC_NODES`
for the measurement that forced the two apart.
"""

from __future__ import annotations

import ctypes
import os
import re
import shutil
import subprocess
import sys


__all__ = [
    "QnnDeviceRefused",
    "QnnLibrariesMissing",
    "QNN_LIB_DIR_ENV",
    "QNN_MAVEN_COORDINATE",
    "qnn_library_names",
    "qnn_library_search_dirs",
    "find_qnn_libraries",
    "require_qnn_libraries",
    "DEVICE_DIR",
    "SOC_PROPERTIES",
    "SOC_MANUFACTURER_PROPERTY",
    "HTP_RUNTIME_LIBRARIES",
    "FASTRPC_NODES",
    "CDSP_FASTRPC_NODES",
    "SOC_KNOWN",
    "SOC_NOT_IN_TABLE",
    "SOC_TABLE_UNAVAILABLE",
    "SOC_EXYNOS_UNIMPLEMENTED",
    "serial",
    "on_device",
    "transport",
    "adb_available",
    "adb",
    "shell",
    "getprop",
    "device_soc",
    "fastrpc_nodes",
    "cdsp_fastrpc_nodes",
    "htp_reachable",
    "htp_runtime_probe",
    "parse_platform_validator",
    "fastrpc_dsp_info",
    "PLATFORM_VALIDATOR",
    "device_abi",
    "htp_stub_for",
    "staged_libraries",
    "device_report",
    "cleanup",
]


class QnnDeviceRefused(RuntimeError):
    """No device, no serial, or the device cannot answer the question."""


#: The only path on the device this module writes to. Chosen to match the
#: instruction this round was given; `nnapi_device` owns `bw_device` next to it.
DEVICE_DIR = "/data/local/tmp/bw_qnn"

#: Read in order; the first that answers wins. `ro.soc.model` is the part
#: number Android documents for CDD Section 3.2.2 compliance; `ro.board.platform`
#: and `ro.hardware` are legacy fallbacks.
#:
#: `ro.soc.manufacturer` is **not** in this tuple, and it used to be first. It
#: is a vendor name, not a part number: on the Galaxy Tab S9 Ultra (SM8550,
#: 2026-11-07, #21) it reads `QTI`, so the first-answer-wins loop stopped there
#: and reported `soc_raw=QTI`, `not-in-table`, `htp_arch=null` for a chipset
#: the table knows. It is read separately (`SOC_MANUFACTURER_PROPERTY`), only
#: to recognise Exynos.
SOC_PROPERTIES = ("ro.soc.model", "ro.board.platform", "ro.hardware")
SOC_MANUFACTURER_PROPERTY = "ro.soc.manufacturer"

#: The Qualcomm runtime libraries an HTP-delegated `.pte` needs beside it.
#: `libQnnHtp.so` is the backend, `libQnnSystem.so` its shared system layer,
#: `...V<arch>Stub.so` the CPU-side stub for one HTP generation and
#: `...V<arch>Skel.so` the matching skeleton that is loaded *on the DSP* over
#: FastRPC. The stub/skel pair is per-architecture, which is why `htp_stub_for`
#: takes the arch rather than returning a fixed list: pushing the V73 pair to a
#: V75 device stages files that will never be opened.
HTP_RUNTIME_LIBRARIES = ("libQnnHtp.so", "libQnnSystem.so")


def serial():
    """The device this module talks to, or `None`. `ANDROID_SERIAL` only."""
    return os.environ.get("ANDROID_SERIAL") or None


def on_device():
    """True when this interpreter runs *on* Android, not on a host with adb.

    Same test `torchnative.device.host()` uses. On the device there is no adb
    and no `ANDROID_SERIAL`, and asking for them was defect 3 of #21:
    `model.to(torchnative.device.npu)` on the tablet raised "no adb on PATH".
    """
    return hasattr(sys, "getandroidapilevel")


def transport():
    """`"local"` on the device, `"adb"` on a host."""
    return "local" if on_device() else "adb"


def adb_available():
    return shutil.which("adb") is not None and serial() is not None


def _local_sh(command, timeout=300):
    """Run one `sh -c` command on this (Android) device and return stdout."""
    sh = "/system/bin/sh" if os.path.exists("/system/bin/sh") else "sh"
    proc = subprocess.run(
        [sh, "-c", command], capture_output=True, timeout=timeout
    )
    return proc.stdout.decode(errors="replace")


def shell(command, timeout=300):
    """One shell command on the device: locally on Android, over adb on a host.

    Every probe below goes through this, so the evidence is the same commands
    whichever side of the cable the interpreter is on.
    """
    if on_device():
        return _local_sh(command, timeout=timeout)
    return adb("shell", command, timeout=timeout)


def adb(*args, check=True, binary=False, timeout=300):
    if shutil.which("adb") is None:
        raise QnnDeviceRefused(
            "torchnative qnn_device: no `adb` on PATH. Add "
            "$ANDROID_SDK_ROOT/platform-tools."
        )
    if serial() is None:
        raise QnnDeviceRefused(
            "torchnative qnn_device: ANDROID_SERIAL is unset. It is required "
            "rather than inferred: the Android devices on this machine are "
            "shared, and `adb` with one device attached would silently pick "
            "whichever one that is."
        )
    proc = subprocess.run(
        ["adb", "-s", serial(), *args], capture_output=True, timeout=timeout
    )
    if check and proc.returncode != 0:
        raise QnnDeviceRefused(
            f"torchnative qnn_device: adb {' '.join(args)} exited "
            f"{proc.returncode}: {proc.stderr.decode(errors='replace').strip()}"
        )
    return proc.stdout if binary else proc.stdout.decode(errors="replace")


def getprop(name):
    """One Android system property, stripped. Empty string if it is unset."""
    return shell(f"getprop {name}").strip()


def device_abi():
    """`ro.product.cpu.abi`. QNN's device libraries are `arm64-v8a` only."""
    return getprop("ro.product.cpu.abi")


#: `device_soc`'s third element. Four outcomes, and conflating any two of
#: them produces a report that is worse than no report.
SOC_KNOWN = "known"                       # in ExecuTorch's QcomChipset
SOC_NOT_IN_TABLE = "not-in-table"         # the device answered; the table does not have it
SOC_TABLE_UNAVAILABLE = "table-unavailable"   # this *host* cannot consult the table
SOC_EXYNOS_UNIMPLEMENTED = "exynos-unimplemented"  # Samsung Exynos SoC detected


def _is_exynos(prop: str, raw: str) -> bool:
    r = raw.lower()
    if prop == "ro.soc.manufacturer" and r in ("samsung", "exynos"):
        return True
    if "exynos" in r or r.startswith("s5e") or r.startswith("universal"):
        return True
    return False


def device_soc():
    """`(property, raw, status, chipset, soc_model, htp_arch)` read from the device.

    **`status` exists because the first version of this function did not have
    it, and it lied.** Run against a real Snapdragon 8 Gen 2 whose
    `ro.soc.model` is `SM8550` -- a chipset ExecuTorch knows perfectly well --
    it reported `chipset: null`, because `qnn.resolve_soc` needs executorch to
    read `QcomChipset` and the interpreter it was called from did not have
    executorch installed. A refusal about the *host* was rendered as a verdict
    about the *device*.

    Those two are not close. `SOC_NOT_IN_TABLE` means what
    pytorch/executorch#16465 describes -- a device reporting `CQ8750S`, on
    which QNN's own SoC detection later fails with "No Snapdragon SOC
    detected", and no lowering can help. `SOC_TABLE_UNAVAILABLE` means install
    executorch. Reporting the second as the first sends the reader to buy
    different hardware.

    The raw string is always carried, whatever the status: it is the only
    thing here that came from the device, and it is what a person needs when
    the table does not have it.
    """
    from torchnative.export import qnn

    manufacturer = getprop(SOC_MANUFACTURER_PROPERTY)
    if manufacturer and _is_exynos(SOC_MANUFACTURER_PROPERTY, manufacturer):
        model = getprop("ro.soc.model") or manufacturer
        return (
            SOC_MANUFACTURER_PROPERTY, manufacturer, SOC_EXYNOS_UNIMPLEMENTED,
            "Samsung Exynos", model, None,
        )
    for prop in SOC_PROPERTIES:
        raw = getprop(prop)
        if not raw:
            continue
        if _is_exynos(prop, raw):
            model = getprop("ro.soc.model") or raw
            return prop, raw, SOC_EXYNOS_UNIMPLEMENTED, "Samsung Exynos", model, None
        try:
            member, soc_model, htp_arch = qnn.resolve_soc(raw)
        except qnn.QnnRefused as exc:
            # Distinguish "the table says no" from "there is no table here" by
            # asking whether the table can be read at all, not by parsing the
            # message.
            try:
                qnn.soc_targets()
            except qnn.QnnRefused:
                return prop, raw, SOC_TABLE_UNAVAILABLE, None, None, None
            return prop, raw, SOC_NOT_IN_TABLE, None, None, None
        return prop, raw, SOC_KNOWN, member.name, soc_model, htp_arch
    raise QnnDeviceRefused(
        "torchnative qnn_device: none of "
        f"{', '.join(SOC_PROPERTIES)} is set on {serial() or 'this device'}. This is not a "
        "device that can be targeted by name, and guessing a chipset would "
        "produce an artefact QNN refuses at init with an architecture "
        "mismatch."
    )


def htp_stub_for(htp_arch):
    """The stub/skel pair for one HTP architecture, e.g. `(…V75Stub.so, …V75Skel.so)`.

    Derived from the architecture number rather than listed, because the list
    grows every generation and a stale copy of it here would silently omit the
    newest device -- the same reason `qnn.soc_targets()` reads ExecuTorch's
    table instead of transcribing it.
    """
    if not isinstance(htp_arch, int) or htp_arch <= 0:
        raise QnnDeviceRefused(
            f"torchnative qnn_device: {htp_arch!r} is not an HTP architecture "
            "number. It comes from ExecuTorch's `_soc_info_table` via "
            "`qnn.resolve_soc`, and there is no default that is safe to "
            "invent -- the stub must match the silicon."
        )
    return (f"libQnnHtpV{htp_arch}Stub.so", f"libQnnHtpV{htp_arch}Skel.so")


#: Every FastRPC character device this module looks for, compute and audio
#: alike. Reported for context; **not** evidence of an NPU on its own, for the
#: reason `CDSP_FASTRPC_NODES` gives.
FASTRPC_NODES = (
    "/dev/adsprpc-smd",
    "/dev/adsprpc-smd-secure",
    "/dev/fastrpc-adsp",
    "/dev/cdsprpc-smd",
    "/dev/cdsprpc-smd-secure",
    "/dev/fastrpc-cdsp",
)

#: The subset of `FASTRPC_NODES` that reaches the **compute** DSP, which is the
#: only DSP the HTP runs on. `libQnnHtpV<arch>Skel.so` is loaded *onto the cDSP*
#: over one of these, so their absence means no Hexagon NPU is reachable
#: however good the artefact is and whatever the SoC is called.
#:
#: **Separating these two tuples is the whole point.** The first version of this
#: module had one flat tuple with `/dev/adsprpc-smd` at the front of it, and
#: `device_report` reported `fastrpc: ("/dev/adsprpc-smd",)` on the Galaxy Tab
#: S9 Ultra as if that were a positive finding. It is not. `adsprpc` is the
#: **audio** DSP: it is present on essentially every Qualcomm phone and tablet
#: ever shipped, it is present when the cDSP is fused off, and nothing that
#: runs on it is an NPU workload. Measured on that tablet (SM8550, Android 16,
#: API 36): `/sys/class/fastrpc/` -- the kernel driver's own registration list,
#: which is what this should have been reading -- contains exactly
#: `adsprpc-smd` and `adsprpc-smd-secure` and **no cDSP endpoint at all**,
#: while `/dev/adsprpc-smd` is `crw-rw-r-- system:system` and so is not even
#: openable read-write by the `shell` user that would have to open it.
#:
#: Both the downstream (`cdsprpc-smd`) and mainline (`fastrpc-cdsp`) names are
#: listed because which one a kernel exposes is not fixed.
CDSP_FASTRPC_NODES = (
    "/dev/cdsprpc-smd",
    "/dev/cdsprpc-smd-secure",
    "/dev/fastrpc-cdsp",
)


def fastrpc_nodes():
    """Which of `FASTRPC_NODES` exist on the device. Stat, never `ls /dev`.

    Each path is stat'd by name rather than the directory being listed, so that
    a device whose `/dev` the `shell` user cannot enumerate is not reported as
    having no DSP.

    **A non-empty return is not an NPU.** Nearly every entry that comes back on
    a real handset is `adsprpc`, the audio DSP. Use `cdsp_fastrpc_nodes` for the
    question anybody actually means.
    """
    found = []
    for path in FASTRPC_NODES:
        out = shell(f"ls -d {path} 2>/dev/null || true").strip()
        if out == path:
            found.append(path)
    return tuple(found)


def cdsp_fastrpc_nodes():
    """Which compute-DSP FastRPC endpoints exist. The HTP is reached over these.

    Two sources, and the disagreement between them matters, so both are
    consulted and the union is returned:

    * `/sys/class/fastrpc/` -- the FastRPC driver's own list of registered
      endpoints. This is the authoritative one: a name here was registered by
      the kernel driver, not merely left in a `/dev` populated by ueventd.
    * `ls -d` on each of `CDSP_FASTRPC_NODES`, for kernels that do not export
      the sysfs class.
    """
    found = []
    listing = shell("ls -1 /sys/class/fastrpc 2>/dev/null || true").split()
    for path in CDSP_FASTRPC_NODES:
        name = path.rsplit("/", 1)[-1]
        if name in listing:
            found.append(path)
            continue
        out = shell(f"ls -d {path} 2>/dev/null || true").strip()
        if out == path:
            found.append(path)
    return tuple(found)


#: Qualcomm's own on-device checker, from the QNN SDK
#: (`bin/aarch64-android/qnn-platform-validator`). Never pushed by this module:
#: it is used only if someone staged it under `DEVICE_DIR`, beside the
#: `libQnnHtpV<arch>CalculatorStub.so` / `libCalculator_skel.so` pair its DSP
#: unit test loads.
PLATFORM_VALIDATOR = "qnn-platform-validator"


def parse_platform_validator(text):
    """`(passed, arch, reason)` from `qnn-platform-validator --backend dsp --testBackend`.

    **The exit code is useless**: both recorded runs on the Tab S9 Ultra
    (2026-11-07) exited 0, the passing one and the negative control with the
    skeleton hidden. So the verdict is read from the two lines the tool prints,
    and both must agree:

        Unit Test on the backend DSP: Passed.
        QNN is supported for backend DSP on the device.

    `arch` is the HTP generation from `Core Version of the backend DSP: Hexagon
    Architecture V73`, or `None` when the tool did not query it (it prints
    `Not Queried` on the failing path). Anything that is neither verdict is a
    refusal to answer, reported as `passed=None`, never as a no.
    """
    arch = None
    m = re.search(r"Core Version of the backend DSP:\s*Hexagon Architecture V(\d+)", text)
    if m:
        arch = int(m.group(1))
    unit_pass = re.search(r"Unit Test on the backend DSP:\s*Passed", text) is not None
    unit_fail = re.search(r"Unit Test on the backend DSP:\s*Failed", text) is not None
    supported = "QNN is supported for backend DSP on the device" in text
    unsupported = "QNN is NOT supported for backend DSP on the device" in text
    if unit_pass and supported and not (unit_fail or unsupported):
        return True, arch, None
    if unit_fail or unsupported:
        errors = [
            line.split("ERROR:", 1)[1].strip()
            for line in text.splitlines() if "ERROR:" in line
        ]
        return False, arch, (
            "qnn-platform-validator's DSP unit test failed"
            + (f": {'; '.join(errors)}" if errors else "")
        )
    return None, arch, (
        "qnn-platform-validator printed no DSP verdict (neither 'Unit Test on "
        "the backend DSP: Passed' nor 'Failed')"
    )


#: FastRPC's capability query, `remote_handle_control(DSPRPC_GET_DSP_INFO, ...)`
#: from the Hexagon SDK's `remote.h`. It asks the FastRPC driver about one DSP
#: domain without loading any skeleton onto it.
_DSPRPC_GET_DSP_INFO = 2
_CDSP_DOMAIN_ID = 3
_ATTR_DOMAIN_SUPPORT = 0
_ATTR_ARCH_VER = 6
_ATTR_HMX_SUPPORT_DEPTH = 7


class _RemoteDspCapability(ctypes.Structure):
    _fields_ = [
        ("domain", ctypes.c_uint32),
        ("attribute_ID", ctypes.c_uint32),
        ("capability", ctypes.c_uint32),
    ]


def _load_cdsprpc():
    """`libcdsprpc.so` from the vendor partition, or `None`. Device only."""
    for name in ("libcdsprpc.so", "/vendor/lib64/libcdsprpc.so"):
        try:
            return ctypes.CDLL(name)
        except OSError:
            continue
    return None


def fastrpc_dsp_info(lib=None):
    """`(supported, arch, hmx, reason)` for the compute DSP, asked of FastRPC itself.

    Runs only on the device (`on_device()`): it dlopens the vendor's
    `libcdsprpc.so`, which the platform validator loads successfully from a
    shell on the Tab S9 Ultra. `supported` is `None` when the query could not be
    made at all, which is not a no. `arch` decodes `ARCH_VER` the way the
    Hexagon SDK's `get_hex_arch_ver` does (low byte, hex digits: `0x73` is V73).

    **`DOMAIN_SUPPORT` is not evidence either way.** Measured on a Galaxy Z
    Flip 4 (SM8475, 2026-11-13, through both `libcdsprpc.so` and
    `libadsprpc.so`): it answers `rc=0, capability=0` for *every* domain,
    including the audio DSP that is certainly present, while the platform
    validator's DSP unit test passes on the same device. What does
    discriminate is `ARCH_VER`: `rc=0x4e` for domains 0 to 2 and `rc=0,
    0x1008c69` for the CDSP (low byte `0x69`: V69, matching the SoC table and
    the validator). So the verdict comes from `ARCH_VER`; `DOMAIN_SUPPORT`
    can only add a yes. `HMX_SUPPORT_DEPTH` read 0 on that V69 part, so a 0
    there is reported as unknown, never as "no HMX".
    """
    if lib is None:
        if not on_device():
            return None, None, None, "the FastRPC query runs only on the device"
        lib = _load_cdsprpc()
        if lib is None:
            return None, None, None, "libcdsprpc.so could not be loaded on this device"
    try:
        control = lib.remote_handle_control
    except AttributeError:
        return None, None, None, "libcdsprpc.so has no remote_handle_control"
    control.restype = ctypes.c_int
    control.argtypes = [ctypes.c_uint32, ctypes.c_void_p, ctypes.c_uint32]

    def ask(attribute):
        cap = _RemoteDspCapability(_CDSP_DOMAIN_ID, attribute, 0)
        rc = control(_DSPRPC_GET_DSP_INFO, ctypes.byref(cap), ctypes.sizeof(cap))
        return rc, cap.capability

    rc_dom, supported = ask(_ATTR_DOMAIN_SUPPORT)
    rc_arch, raw = ask(_ATTR_ARCH_VER)
    arch = None
    if rc_arch == 0 and raw:
        try:
            arch = int(f"{raw & 0xFF:x}")
        except ValueError:
            arch = None
    if arch is None and not (rc_dom == 0 and supported):
        return None, None, None, (
            "FastRPC did not identify a compute DSP: "
            f"DSPRPC_GET_DSP_INFO(CDSP, ARCH_VER) returned rc={rc_arch & 0xFFFFFFFF:#x}, "
            f"value={raw:#x}; DOMAIN_SUPPORT returned rc={rc_dom & 0xFFFFFFFF:#x}, "
            f"value={supported}. DOMAIN_SUPPORT=0 is not read as a no: it "
            "answers 0 for every domain on some devices"
        )
    rc_hmx, depth = ask(_ATTR_HMX_SUPPORT_DEPTH)
    hmx = True if rc_hmx == 0 and depth else None
    return True, arch, hmx, None


def _run_staged_validator():
    """The staged platform validator's output, or `None` if it is not staged."""
    path = f"{DEVICE_DIR}/{PLATFORM_VALIDATOR}"
    present = shell(f"ls -d {path} 2>/dev/null || true").strip()
    if present != path:
        return None
    # `--coreVersion`: without it the tool prints `Core Version: Not Queried`
    # (Galaxy Z Flip 4, SM8475, QNN 2.37, 2026-11-13); with it, `Hexagon
    # Architecture V69`. `--targetPath`: by default the tool writes
    # `Result.csv` to /data/local/tmp/platformValidator/output, outside the
    # one directory this module may write to.
    return shell(
        f"cd {DEVICE_DIR} && LD_LIBRARY_PATH={DEVICE_DIR} "
        f"ADSP_LIBRARY_PATH={DEVICE_DIR} ./{PLATFORM_VALIDATOR} "
        f"--backend dsp --coreVersion --testBackend "
        f"--targetPath {DEVICE_DIR}/validator-output 2>&1 || true",
        timeout=120,
    )


def htp_runtime_probe():
    """Runtime evidence that the compute DSP answers. A dict; never raises.

    `{"method", "ok", "arch", "hmx", "detail"}`. `ok` is `True` (it answered),
    `False` (it answered no) or `None` (no runtime probe could be run here,
    which is not a no and is never reported as a yes). Methods, first that
    gives a verdict wins:

    * `platform-validator`: Qualcomm's `qnn-platform-validator`, if staged
      under `DEVICE_DIR`, over either transport. This is the measurement in
      #21 (Tab S9 Ultra, QNN 2.37: passed, V73; failed with the skel hidden).
    * `fastrpc-dsp-info`: on the device only, FastRPC's capability query for
      the CDSP domain (`fastrpc_dsp_info`).
    """
    tried = []
    try:
        text = _run_staged_validator()
    except QnnDeviceRefused as exc:
        text = None
        tried.append(f"platform-validator: {exc}")
    if text is None:
        if not tried:
            tried.append(f"platform-validator: not staged under {DEVICE_DIR}")
    else:
        passed, arch, why = parse_platform_validator(text)
        if passed is not None:
            return {
                "method": "platform-validator", "ok": passed, "arch": arch,
                "hmx": None, "detail": why,
            }
        tried.append(f"platform-validator: {why}")
    supported, arch, hmx, why = fastrpc_dsp_info()
    if supported is not None:
        return {
            "method": "fastrpc-dsp-info", "ok": supported, "arch": arch,
            "hmx": hmx, "detail": why,
        }
    tried.append(f"fastrpc-dsp-info: {why}")
    return {"method": None, "ok": None, "arch": None, "hmx": None,
            "detail": "; ".join(tried)}


def htp_reachable(probe=None):
    """`(bool, reason)` -- is there a path to the Hexagon NPU on this device?

    Deliberately **not** a function of the SoC name. `device_soc` answering
    `SM8550`/`htp_arch=73` says the silicon that model number denotes has a V73
    HTP in its datasheet; it does not say this unit's cDSP is powered, fused in,
    exposed to userspace, or reachable by the uid holding the adb shell. Those
    are different claims and only the second one licenses `device.npu`.

    A dedicated compute-DSP node is enough. Without one, the answer comes
    from `htp_runtime_probe`, never from a node's name: on the Tab S9 Ultra
    (2026-11-07, #21) there is no `cdsprpc` node at all, yet the platform
    validator's DSP unit test passes, because that kernel reaches the cDSP
    through `/dev/adsprpc-smd`. An `adsprpc` node on its own is still not
    evidence (it is the audio DSP on every Qualcomm part); only a runtime probe
    that answered yes turns it into one.
    """
    nodes = cdsp_fastrpc_nodes()
    if nodes:
        return True, None
    if probe is None:
        probe = htp_runtime_probe()
    if probe["ok"] is True:
        return True, None
    no_node = (
        "no compute-DSP FastRPC endpoint on the device: none of "
        f"{', '.join(CDSP_FASTRPC_NODES)} exists and /sys/class/fastrpc "
        "registers none. (An `adsprpc` node is the *audio* DSP and does not "
        "substitute on its own.)"
    )
    if probe["ok"] is False:
        return False, (
            f"{no_node} The runtime probe ({probe['method']}) answered no: "
            f"{probe['detail']}"
        )
    return False, (
        f"{no_node} No runtime probe could answer either ({probe['detail']}). "
        "On kernels that reach the cDSP through /dev/adsprpc-smd, stage "
        f"{PLATFORM_VALIDATOR} and its calculator stub/skel under {DEVICE_DIR}, "
        "or run torchnative on the device itself."
    )


def staged_libraries():
    """Names of files already under `DEVICE_DIR`, or `()` if it does not exist."""
    out = shell(f"ls -1 {DEVICE_DIR} 2>/dev/null || true")
    return tuple(sorted(line.strip() for line in out.splitlines() if line.strip()))


class QnnLibrariesMissing(QnnDeviceRefused):
    """The Qualcomm runtime libraries are not on this device or in this app."""


#: Where a caller can point torchnative at a directory of QNN libraries.
QNN_LIB_DIR_ENV = "TORCHNATIVE_QNN_LIB_DIR"

#: The Maven Central artefact an Android app adds to get them.
QNN_MAVEN_COORDINATE = "com.qualcomm.qti:qnn-runtime"


def qnn_library_names(htp_arch):
    """The files the HTP backend needs for one architecture, CPU side and DSP side."""
    return HTP_RUNTIME_LIBRARIES + htp_stub_for(htp_arch)


def _proc_maps_text():
    try:
        with open("/proc/self/maps", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def _app_native_lib_dirs(maps_text):
    """Directories of the shared libraries this process already mapped.

    An Android app's `jniLibs` (where Gradle puts the `qnn-runtime` AAR's
    `.so` files) are extracted beside the app's other native libraries, so the
    directory libpython was loaded from is the app's native-library directory.
    Read from `/proc/self/maps`, not guessed from a package name.
    """
    dirs = []
    for line in maps_text.splitlines():
        parts = line.split()
        if len(parts) < 6:
            continue
        path = parts[-1]
        if not path.startswith("/") or not path.endswith(".so"):
            continue
        base = path.rsplit("/", 1)[-1]
        if base.startswith("libpython") or base.startswith("libtorch_global_deps"):
            d = path.rsplit("/", 1)[0]
            if d not in dirs:
                dirs.append(d)
    return dirs


def qnn_library_search_dirs(environ=None, maps_text=None):
    """Where `find_qnn_libraries` looks, in order, with duplicates removed.

    1. `TORCHNATIVE_QNN_LIB_DIR`, if set: an explicit answer wins.
    2. `ADSP_LIBRARY_PATH` (`;`- or `:`-separated): where FastRPC loads skels from.
    3. `LD_LIBRARY_PATH`.
    4. The app's own native-library directory (`_app_native_lib_dirs`).
    5. `/vendor/lib64`, `/system/lib64`: some OEM images carry the QNN runtime.
    6. `DEVICE_DIR`, this module's staging directory.

    torchnative ships none of these files (owner, 2026-11-13: "(a) 로 가자").
    """
    env = os.environ if environ is None else environ
    maps = _proc_maps_text() if maps_text is None else maps_text
    out = []

    def add(d):
        d = d.strip()
        if d and d not in out:
            out.append(d)

    if env.get(QNN_LIB_DIR_ENV):
        add(env[QNN_LIB_DIR_ENV])
    for d in re.split(r"[;:]", env.get("ADSP_LIBRARY_PATH", "")):
        add(d)
    for d in env.get("LD_LIBRARY_PATH", "").split(":"):
        add(d)
    for d in _app_native_lib_dirs(maps):
        add(d)
    for d in ("/vendor/lib64", "/system/lib64", DEVICE_DIR):
        add(d)
    return out


def find_qnn_libraries(htp_arch, *, environ=None, maps_text=None, exists=None):
    """`{name: path or None}` for `qnn_library_names(htp_arch)`, on this device.

    Runs locally (on the device, in the app's process): the question is what
    *this* process can load, which an adb shell on a host cannot answer.
    """
    exists = os.path.isfile if exists is None else exists
    dirs = qnn_library_search_dirs(environ=environ, maps_text=maps_text)
    found = {}
    for name in qnn_library_names(htp_arch):
        found[name] = next(
            (f"{d}/{name}" for d in dirs if exists(f"{d}/{name}")), None
        )
    return found


def require_qnn_libraries(htp_arch, **kwargs):
    """`find_qnn_libraries`, or `QnnLibrariesMissing` naming what is absent."""
    found = find_qnn_libraries(htp_arch, **kwargs)
    missing = [name for name, path in found.items() if path is None]
    if missing:
        dirs = qnn_library_search_dirs(
            environ=kwargs.get("environ"), maps_text=kwargs.get("maps_text")
        )
        raise QnnLibrariesMissing(
            f"torchnative qnn_device: the Qualcomm QNN runtime libraries for "
            f"HTP V{htp_arch} are not on this device or in this app: missing "
            f"{', '.join(missing)}. Looked in: {', '.join(dirs)}. torchnative "
            f"does not ship them; an Android app adds {QNN_MAVEN_COORDINATE} "
            f"from Maven Central in Gradle, or set {QNN_LIB_DIR_ENV} to a "
            f"directory that holds them."
        )
    return found


def device_report():
    """Everything this module can establish about the attached device.

    A dict rather than an exception on the negative paths, for the reason
    `qnn.probe()` gives: when the answer is "no NPU here", the useful output is
    which half is missing.
    """
    if not on_device() and not adb_available():
        return {
            "serial": serial(),
            "transport": "adb",
            "reachable": False,
            "reason": (
                "no adb on PATH, or ANDROID_SERIAL is unset"
                if serial() is None
                else "no adb on PATH"
            ),
        }
    report = {
        "serial": None if on_device() else serial(),
        "transport": transport(),
        "reachable": True,
        "reason": None,
    }
    try:
        report["abi"] = device_abi()
        prop, raw, status, chipset, soc_model, htp_arch = device_soc()
        report.update(
            soc_property=prop,
            soc_raw=raw,
            soc_status=status,
            chipset=chipset,
            soc_model=soc_model,
            htp_arch=htp_arch,
        )
        if status == SOC_EXYNOS_UNIMPLEMENTED:
            report["is_exynos"] = True
            report["htp_reachable"] = False
            report["htp_unreachable_reason"] = (
                f"Samsung Exynos SoC detected ({soc_model or raw}), but Exynos NPU support is unimplemented in torchnative. "
                f"Use torchnative.device.cpu or torchnative.device.vulkan for execution on this device."
            )
            return report
        report["htp_arch_source"] = "soc-table" if htp_arch else None
        report["fastrpc"] = fastrpc_nodes()
        report["cdsp_fastrpc"] = cdsp_fastrpc_nodes()
        probe = None if report["cdsp_fastrpc"] else htp_runtime_probe()
        # The SoC name says what the datasheet has; this says what the device
        # will actually let a caller reach. `device.npu` gates on this one.
        reachable, why = htp_reachable(probe)
        report["htp_reachable"] = reachable
        report["htp_unreachable_reason"] = why
        if probe is not None:
            report["htp_runtime_probe"] = probe
            # On the device there is usually no executorch, so the SoC table
            # cannot be read (`table-unavailable`). The architecture the DSP
            # itself reported is a measurement, and is used and labelled so.
            if probe["ok"] and probe["arch"]:
                report["htp_arch_runtime"] = probe["arch"]
            if not htp_arch and probe["ok"] and probe["arch"]:
                htp_arch = probe["arch"]
                report["htp_arch"] = htp_arch
                report["htp_arch_source"] = f"runtime:{probe['method']}"
        report["htp_libraries"] = (
            HTP_RUNTIME_LIBRARIES + htp_stub_for(htp_arch)
            if htp_arch
            else HTP_RUNTIME_LIBRARIES
        )
        # Only the process that will load them can answer this, so only the
        # local transport asks (#260).
        if on_device() and htp_arch:
            libs = find_qnn_libraries(htp_arch)
            report["qnn_libraries"] = libs
            report["qnn_libraries_complete"] = all(libs.values())
            if not report["qnn_libraries_complete"]:
                try:
                    require_qnn_libraries(htp_arch)
                except QnnLibrariesMissing as exc:
                    report["qnn_libraries_missing_reason"] = str(exc)
        report["staged"] = staged_libraries()
        report["staged_complete"] = all(
            name in report["staged"] for name in report["htp_libraries"]
        )
    except QnnDeviceRefused as exc:
        report["reachable"] = False
        report["reason"] = str(exc)
    return report


def cleanup():
    """Remove `DEVICE_DIR` entirely. The devices here are shared.

    Returns True if the directory is gone afterwards. Never touches anything
    outside it and never uninstalls anything, because this module never
    installed anything.
    """
    adb("shell", f"rm -rf {DEVICE_DIR}", check=False)
    return not staged_libraries()
