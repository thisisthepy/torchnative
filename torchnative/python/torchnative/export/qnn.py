"""ExecuTorch's Qualcomm (QNN) backend: host capability probes and the SoC table.

docs/devices/QNN.md is this module's design record. The `.pte` writers
(`lower`, `lower_cpu_reference`; #277) and the `.pte` reader and loader
(`read_artefact`, `PteArtefact`, `match_device`, `ExecuTorchModule`,
`QnnModule`; #277, owner decision D12 part 2) were removed by owner decision.
The on-device route runs ExecuTorch programs from in-memory buffers produced at
`.to(device)` time, and no `.pte` file is written or read (leader, 2026-11-13).

What remains is what the in-memory route still asks of this host: which
ExecuTorch and QNN SDK versions are installed, why the QNN ahead-of-time half
cannot run here, which SoCs ExecuTorch's own tables know, and which backends
this process's ExecuTorch runtime has registered.

## What this host can and cannot do

Measured, not assumed -- `probe()` re-measures it on every call:

* The ExecuTorch runtime works on macOS arm64.
* The **QNN** half does not. Every one of ExecuTorch's 112 QNN op builders
  begins `import executorch.backends.qualcomm.python.PyQnnManagerAdaptor`, and
  that package is absent from the macOS wheel -- it is a pybind extension built
  against `$QNN_SDK_ROOT`, and the SDK ships host libraries for
  `lib/x86_64-linux-clang` only. `qnn_aot_refusal()` returns that sentence with
  the module name in it rather than letting an ImportError surface from
  eighteen frames down.

## What this module will not claim

Getting the right answer out of an ExecuTorch program is **not** evidence that
it ran on the HTP. docs/graph/NPU2.md is this project's record of paying for
that mistake twice: the CoreML models docs/graph/NPU.md called "executed" had
run on the CPU, and every NNAPI driver on the emulator was software. The
results were correct either way. Nothing here answers "did it run on the NPU";
`runtime_backends()` answers only whether the runtime could have.
"""

from __future__ import annotations

import importlib
import platform


__all__ = [
    "QnnRefused",
    "QNN_BACKEND_ID",
    "host_platform",
    "executorch_version",
    "qnn_sdk_version",
    "qnn_aot_refusal",
    "qnn_aot_available",
    "soc_targets",
    "resolve_soc",
    "probe",
    "runtime_backends",
]


class QnnRefused(RuntimeError):
    """This host, this artefact or this runtime cannot do what was asked.

    Always carries the name of the missing thing. A refusal that says only
    "unsupported" sends the reader to the wrong place; docs/graph/NPU.md §4 is the
    precedent -- an op with no calling convention is *absent* and named, never
    approximated.
    """


#: The `backend_id` ExecuTorch's Qualcomm backend registers itself under, and
#: the string that appears in an ExecuTorch program's delegate table. It is
#: compared against `runtime_backends()`.
QNN_BACKEND_ID = "QnnBackend"


# --------------------------------------------------------------------------
# Host capability. Every one of these measures; none of them recalls.
# --------------------------------------------------------------------------


def host_platform():
    """`(system, machine)` lowercased, e.g. `("darwin", "arm64")`."""
    return platform.system().lower(), platform.machine().lower()


def _import(name):
    try:
        return importlib.import_module(name)
    except Exception as exc:  # ImportError, and whatever a broken install raises
        raise QnnRefused(
            f"torchnative qnn: cannot import {name!r}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc


def executorch_version():
    """The installed ExecuTorch version, or a refusal naming the package."""
    et = _import("executorch")
    version = getattr(et, "__version__", None)
    if version:
        return version
    meta = _import("importlib.metadata")
    try:
        return meta.version("executorch")
    except Exception as exc:
        raise QnnRefused(
            "torchnative qnn: executorch is importable but reports no version "
            f"({type(exc).__name__}). Refusing rather than reporting 'unknown', "
            "since every version-dependent statement in docs/devices/QNN.md is keyed to "
            "a number."
        ) from exc


def qnn_sdk_version():
    """The QNN SDK version *this ExecuTorch build pins*, read from its own script.

    Not transcribed. `executorch/backends/qualcomm/scripts/download_qnn_sdk.py`
    resolves `QNN_VERSION` from `qnn_config.sh` if that file shipped and falls
    back to a literal otherwise; reading the resulting module attribute gets
    whichever one is true for the installed wheel. Transcribing the number here
    would go stale the first time ExecuTorch bumps it, which is the mistake
    docs/graph/DECOMP.md §12.6 named for operator lists and this is the same mistake
    one size smaller.
    """
    mod = _import("executorch.backends.qualcomm.scripts.download_qnn_sdk")
    version = getattr(mod, "QNN_VERSION", None)
    url = getattr(mod, "QNN_ZIP_URL", None)
    if not version:
        raise QnnRefused(
            "torchnative qnn: download_qnn_sdk exposes no QNN_VERSION. The "
            "installed ExecuTorch does not say which SDK it was built against."
        )
    return version, url


def qnn_aot_refusal():
    """Why the QNN ahead-of-time lowering cannot run here, or `None` if it can.

    The check is an **import**, not a platform string comparison. A platform
    test would say "macOS is not on the supported list", which is true and is
    not the same claim; this says which module is missing, and it would go
    quiet by itself on a host where somebody had built it.
    """
    try:
        importlib.import_module("executorch")
    except Exception as exc:
        return (
            f"executorch is not installed ({type(exc).__name__}). The "
            "ahead-of-time half of this path is upstream ExecuTorch and there "
            "is nothing here to substitute for it."
        )
    try:
        importlib.import_module(
            "executorch.backends.qualcomm.python.PyQnnManagerAdaptor"
        )
    except Exception as exc:
        system, machine = host_platform()
        return (
            "executorch.backends.qualcomm.python.PyQnnManagerAdaptor is not "
            f"importable on this host ({system}/{machine}): "
            f"{type(exc).__name__}: {exc}. That module is a pybind extension "
            "built against $QNN_SDK_ROOT, and the Qualcomm AI Engine Direct "
            "SDK ships host libraries under lib/x86_64-linux-clang only -- "
            "ExecuTorch's own installer gates on is_linux_x86(). Every one of "
            "the QNN op builders imports it at module scope, so the lowering "
            "cannot start. docs/devices/QNN.md §3 records the measurement."
        )
    try:
        importlib.import_module(
            "executorch.backends.qualcomm.partition.qnn_partitioner"
        )
    except Exception as exc:
        return (
            "the QNN native adaptor imports but "
            "executorch.backends.qualcomm.partition.qnn_partitioner does not: "
            f"{type(exc).__name__}: {exc}"
        )
    return None


def qnn_aot_available():
    return qnn_aot_refusal() is None


def soc_targets():
    """`{name: (soc_model, htp_arch)}` read out of ExecuTorch's own tables.

    The authority is `QcomChipset` and `_soc_info_table` in
    `executorch/backends/qualcomm/serialization/qc_schema.py`. Those are plain
    Python enums with no SDK dependency, so this works on a host where the
    lowering does not -- which is the point: a refusal that can list the SoCs
    it would have accepted is more useful than one that cannot.

    docs/graph/DECOMP.md §12.2 is the rule being followed. Copying twenty-three
    chipset names into this file would make torchnative the authority on
    Qualcomm's part numbers, and it would be wrong the first time a part is
    added.
    """
    schema = _import("executorch.backends.qualcomm.serialization.qc_schema")
    chipset = schema.QcomChipset
    table = schema._soc_info_table
    out = {}
    for member in chipset:
        info = table.get(member)
        if info is None:
            continue
        out[member.name] = (int(member.value), int(info.htp_info.htp_arch))
    return out


def resolve_soc(name):
    """`(QcomChipset, soc_model, htp_arch)` for an SoC name, or a named refusal.

    Case-insensitive, because the value this is fed comes from the device --
    `ro.soc.model` on a Snapdragon 8 Gen 3 reads `SM8650`, but the property is
    OEM-populated and its case is not guaranteed.
    """
    schema = _import("executorch.backends.qualcomm.serialization.qc_schema")
    wanted = str(name).strip().upper()
    for member in schema.QcomChipset:
        if member.name.upper() == wanted:
            info = schema._soc_info_table.get(member)
            if info is None:
                raise QnnRefused(
                    f"torchnative qnn: {member.name} is in QcomChipset but has "
                    "no entry in _soc_info_table, so its HTP architecture is "
                    "unknown and no compile spec can be built for it."
                )
            return member, int(member.value), int(info.htp_info.htp_arch)
    known = ", ".join(sorted(soc_targets()))
    raise QnnRefused(
        f"torchnative qnn: {name!r} is not a SoC ExecuTorch's QNN backend "
        f"knows. Known: {known}. The name is read from the device's "
        "`ro.soc.model` property (docs/devices/QNN.md §5); if a real device reports "
        "one that is not on this list, the SDK and ExecuTorch both need to "
        "grow it and no lowering here can substitute."
    )


def probe():
    """Everything this host can say about its own QNN capability, measured.

    Returns a dict rather than raising, because the useful thing when the
    answer is no is the *whole* picture -- which half is missing.
    """
    system, machine = host_platform()
    out = {
        "system": system,
        "machine": machine,
        "executorch": None,
        "executorch_error": None,
        "qnn_sdk_version": None,
        "qnn_sdk_url": None,
        "qnn_aot_refusal": None,
        "qnn_aot_available": False,
        "soc_targets": 0,
        "runtime_backends": (),
        "qnn_backend_registered": False,
    }
    try:
        out["executorch"] = executorch_version()
    except QnnRefused as exc:
        out["executorch_error"] = str(exc)
        out["qnn_aot_refusal"] = qnn_aot_refusal()
        return out
    try:
        out["qnn_sdk_version"], out["qnn_sdk_url"] = qnn_sdk_version()
    except QnnRefused:
        pass
    out["qnn_aot_refusal"] = qnn_aot_refusal()
    out["qnn_aot_available"] = out["qnn_aot_refusal"] is None
    try:
        out["soc_targets"] = len(soc_targets())
    except QnnRefused:
        pass
    try:
        out["runtime_backends"] = runtime_backends()
    except QnnRefused:
        pass
    out["qnn_backend_registered"] = QNN_BACKEND_ID in out["runtime_backends"]
    return out


# --------------------------------------------------------------------------
# Runtime. This half is what ships; it loads a file and calls a method.
# --------------------------------------------------------------------------


def runtime_backends():
    """Backend ids registered in *this process's* ExecuTorch runtime.

    A runtime assertion, not a build-time one. `QnnBackend` appears here only
    where the runtime was linked against the QNN backend library, so its
    absence is the first thing to check when a QNN program will not load -- and
    its presence is a precondition for any claim about where a graph ran.
    """
    runtime = _import("executorch.runtime")
    registry = runtime.Runtime.get().backend_registry
    return tuple(registry.registered_backend_names)
