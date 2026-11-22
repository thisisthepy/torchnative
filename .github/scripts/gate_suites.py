"""Which gate suite runs on which CI runner -- and which stays local-only.

Issue #24. `.github/workflows/test.yml` runs `tests/run.sh` on
GitHub-hosted runners with `TORCHNATIVE_GATE_RUNNER` set to one of `RUNNERS`;
`run.sh` hands it to `suite_ledger.py --runner`, which reads this file.

**The rule this file exists for** (AGENTS.md §17.5): the suites run on CI plus
the suites left local-only must equal the full gate. So every `test_*.py` in
`tests/` (every function subfolder) is named in `SUITES` -- explicitly, not by a default --
and `check()` refuses a suite file that is missing here, or a name here that
is not a file. That refusal fires in three places:

  * `test_cigate.py`, which is itself a gate suite, so a round that adds a
    suite without classifying it goes red **locally**, before any push;
  * `suite_ledger.py --runner`, before a single suite runs on CI;
  * `gate_suites.py report`, which builds the PR summary.

A suite planned for a runner is run there whole. A suite **not** planned for a
runner is not executed there; the ledger prints

    SKIP <suite>: (whole suite) -- not run on CI runner '<runner>': <reason>

so it is counted in the ledger's SKIP column and named, the same shape as
`_skip.py`'s per-test lines. Inside a suite that does run, tests that need a
device the runner lacks skip by name through the suites' own `_skip` /
`vulkan_coverage` helpers, as they already do on a Mac without Vulkan.

How a suite earns `LOCAL_ONLY` here: it would either FAIL on a hosted runner
for a reason that is about the runner and not the code, or -- worse -- print
`ok` having tested nothing, because its skip is a hand-rolled `print` that the
ledger cannot see (the `_skip.py` docstring has the defect). The reasons are
in `WHY_NOT`, and the report copies them onto the PR.

What this classification cannot see: whether a suite planned for a runner
actually works there. Before #24 nobody had run this gate on Linux, so the
plan below is a static reading of the suites, not a measurement. The first CI
run is the measurement. A suite that fails there for a platform reason moves
to a narrower runner tuple **with a reason**; it is never deleted from
`SUITES`.

Usage:
    python .github/scripts/gate_suites.py check  [--pytests DIR]
    python .github/scripts/gate_suites.py list
    python .github/scripts/gate_suites.py report --log linux=PATH \
        --log macos=PATH1,PATH2,PATH3 [--out SUMMARY.md]

`--log macos=` takes the macOS gate's three shard logs (`SHARDS`), comma
separated or given as repeated entries, and reports them as one macOS column.
"""

import argparse
import hashlib
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
PYTESTS = REPO / "tests"

#: The CI runners a gate job can name in `TORCHNATIVE_GATE_RUNNER`.
RUNNERS = {
    "linux": "GitHub-hosted ubuntu-24.04 (x86_64). No GPU, no Apple "
             "frameworks, no Vulkan driver, no adb, no Intel NPU.",
    "macos": "GitHub-hosted macos-15 (arm64, a VM). Builds the Apple "
             "variant of the shim (Accelerate + Metal). No Neural Engine; "
             "GitHub documents MPS as unsupported in these VMs -- what Metal "
             "itself does there is printed by the job's probe step.",
    "windows": "GitHub-hosted windows-latest (x86_64). Builds the shim "
               "natively with MSVC (x86_64-pc-windows-msvc), so upstream's "
               "C++ runtime is MSVC's STL (issue #81). No GPU, no NPU.",
}

BOTH = ("linux", "macos")
#: The suites measured on the Windows MSVC runner. The first Windows gate
#: (issue #81, owner 2026-11-13: "msvc 신규 ci 만드는데 문제 있는거 아니면
#: 만들어") runs the tie-order suite and the import surface; a suite joins
#: once it has been seen green there, never by default.
ALL = ("linux", "macos", "windows")

#: A runner-wide reason for every suite not yet planned on that runner. A
#: suite-specific reason in `WHY_NOT` still wins. Only a runner added after
#: the partition was measured may have one, and it names the issue that
#: retires it.
WHY_NOT_ON = {
    "windows": (
        "not yet measured on the Windows MSVC runner (issue #81): the first "
        "Windows gate runs the tie-order and import suites only, and a suite "
        "joins the Windows plan once it has been seen green there"),
}
LOCAL_ONLY = ()

_ANE_SILENT = (
    "needs CoreML's MLComputePlan answering for the Apple Neural Engine "
    "(AGENTS.md §13.3), which no hosted runner has; and its fixture reports a "
    "missing coremltools with a hand-rolled print followed by `ok`, so on a "
    "runner it would print `ok` having run nothing. Local-only until its skips "
    "go through `_skip`, and even then it would only skip.")

_DEV_TOOLCHAIN = (
    "needs the release machine's toolchain, which a hosted runner does not "
    "have: the pinned interpreter at .caches/spike-venv (transformers 5.15.1, "
    "AGENTS.md §15.2), the rust cross targets and the target CPython "
    "distributions under tools/wheel.")

#: Every gate suite, and the runners it runs on. `()` means local-only.
SUITES = {
    "_support/test_shim.py": BOTH,
    "api/test_freethreaded.py": BOTH,
    "api/test_import.py": ALL,
    "api/test_kernel_bundle.py": BOTH,
    "api/test_threadstate.py": BOTH,
    "api/test_tnnamespace.py": BOTH,
    "api/test_torchnative_api.py": BOTH,
    "api/test_tntransformers.py": BOTH,
    "bindings/test_aliasinc.py": BOTH,
    "bindings/test_argform.py": BOTH,
    # Issue #423: every autogen .out of native_functions.yaml against upstream's
    # schema text (upstream in a separate process).
    "bindings/test_autogen_out.py": BOTH,
    "bindings/test_bfallback.py": BOTH,
    "bindings/test_bind2.py": BOTH,
    "bindings/test_bind3.py": BOTH,
    "bindings/test_bind4.py": BOTH,
    "bindings/test_bind5.py": BOTH,
    "bindings/test_bindings.py": BOTH,
    "bindings/test_bkeyset.py": BOTH,
    "bindings/test_convbackend.py": BOTH,
    "bindings/test_dispatch.py": BOTH,
    "bindings/test_dynamo_fills.py": BOTH,
    "bindings/test_dynamo_guards_diff.py": BOTH,
    "bindings/test_dynamo_guards_unit.py": BOTH,
    "bindings/test_dynamo_surface.py": BOTH,
    # Issue #372: the CompositeImplicitAutograd decomposition frame.
    "bindings/test_cia_frame.py": BOTH,
    "bindings/test_functional_ops.py": BOTH,
    "bindings/test_functional_tensor.py": BOTH,
    "bindings/test_prims_tags.py": BOTH,
    "bindings/test_pylibrary.py": BOTH,
    "bindings/test_metakey.py": BOTH,
    "bindings/test_numpy.py": BOTH,
    "bindings/test_methodspell.py": BOTH,
    # Issue #403: torch.isinf / Tensor.isinf against upstream.
    "bindings/test_isinf.py": BOTH,
    # Issue #415: torch.isnan / isposinf / isneginf against upstream.
    "bindings/test_isnan.py": BOTH,
    "bindings/test_jit_types.py": BOTH,
    "bindings/test_known_divergences.py": BOTH,
    # Issues #306, #380: packets with no `default` overload resolve by kind.
    "bindings/test_packet_overload.py": BOTH,
    "bindings/test_setitem.py": BOTH,
    "bindings/test_storagebuf.py": BOTH,
    "bindings/test_viewbase.py": BOTH,
    "bindings/test_wrapper_subclass.py": BOTH,
    "devices/coreml/test_ane_subgraph_components.py": LOCAL_ONLY,
    "devices/coreml/test_anedecode.py": LOCAL_ONLY,
    "devices/coreml/test_anegenerate.py": LOCAL_ONLY,
    "devices/coreml/test_anepath.py": BOTH,
    "devices/coreml/test_anesubgraph.py": LOCAL_ONLY,
    "devices/coreml/test_anetracer.py": LOCAL_ONLY,
    "devices/coreml/test_bf16ane.py": BOTH,
    "devices/coreml/test_coremlops.py": LOCAL_ONLY,
    "devices/coreml/test_emptyplan.py": BOTH,
    "devices/coreml/test_npmarshal.py": BOTH,
    "devices/coreml/test_npu2.py": BOTH,
    "devices/cuda/test_cuda.py": BOTH,
    "devices/mps/test_absmps.py": BOTH,
    "devices/mps/test_constset.py": BOTH,
    "devices/mps/test_gemmint.py": BOTH,
    "devices/mps/test_i8mps.py": BOTH,
    "devices/mps/test_intmps.py": BOTH,
    "devices/mps/test_metalcount.py": BOTH,
    "devices/mps/test_metalplace.py": BOTH,
    "devices/mps/test_mpsattn.py": BOTH,
    "devices/mps/test_mpsclone.py": BOTH,
    "devices/mps/test_mpsconst.py": BOTH,
    "devices/mps/test_mpsnarrow.py": BOTH,
    "devices/mps/test_mpsquant.py": BOTH,
    "devices/mps/test_mpsflash.py": BOTH,
    "devices/mps/test_mpsgen.py": BOTH,
    "devices/mps/test_mpscb.py": BOTH,
    "devices/mps/test_mpssetitem.py": BOTH,
    "devices/mps/test_mpsextreme.py": BOTH,
    "devices/mps/test_mpsfwd.py": BOTH,
    "devices/mps/test_mpsindex.py": BOTH,
    "devices/mps/test_mpsindexput.py": BOTH,
    "devices/mps/test_mpsdiv.py": BOTH,
    "devices/mps/test_mpsint32.py": BOTH,
    "devices/mps/test_mpsinplace.py": BOTH,
    "devices/mps/test_mpsrefuse.py": BOTH,
    "devices/mps/test_mpstrig.py": BOTH,
    "devices/mps/test_mpsresidency.py": BOTH,
    "devices/mps/test_paged_attention.py": BOTH,
    "devices/mps/test_viewdtype.py": BOTH,
    # Issue #423 (R5a): `.to(device.npu)` through ExecuTorch. Runs on macOS arm64,
    # where the runtime is built; on Linux the runtime tests skip by name and the
    # resolution refusal is checked.
    "devices/npu/test_etnpu.py": BOTH,
    "devices/npu/test_intelnpu.py": BOTH,
    "devices/npu/test_npublob.py": BOTH,
    "devices/npu/test_npudim.py": BOTH,
    "devices/npu/test_npufuse.py": BOTH,
    "devices/npu/test_npuvendor.py": BOTH,
    "devices/npu/test_npuwire.py": BOTH,
    "devices/npu/test_ovcache.py": BOTH,
    "devices/npu/test_ovpar.py": BOTH,
    "devices/qnn/test_qnn.py": BOTH,
    "devices/qnn/test_qnn_ops.py": BOTH,
    "devices/qnn/test_qnn_plan.py": BOTH,
    "devices/qnn/test_qnnlibs.py": BOTH,
    "devices/qnn/test_qnnprobe.py": BOTH,
    "devices/qnn/test_qnnprobe_ondevice.py": BOTH,
    "devices/test_devicens.py": BOTH,
    "devices/test_dtmdev.py": BOTH,
    "devices/test_dtypedev.py": BOTH,
    "devices/vulkan/test_vulkan4.py": BOTH,
    "devices/vulkan/test_vulkancov.py": BOTH,
    "distributed/test_asyncwork.py": BOTH,
    "distributed/test_collect2.py": BOTH,
    "distributed/test_fedserver.py": BOTH,
    "distributed/test_gloopin.py": BOTH,
    "distributed/test_launcher.py": BOTH,
    "distributed/test_newgroup.py": BOTH,
    "distributed/test_p2p.py": BOTH,
    "distributed/test_placement.py": BOTH,
    "distributed/test_ppinfer.py": BOTH,
    "distributed/test_secagg.py": BOTH,
    "distributed/test_server.py": BOTH,
    "distributed/test_swarm.py": BOTH,
    "distributed/test_threadranks.py": BOTH,
    "distributed/test_tpinfer.py": BOTH,
    "distributed/test_vfl.py": BOTH,
    "distributed/test_wireparity.py": BOTH,
    "export/test_compile_backend.py": BOTH,
    "export/test_compile_dynamic.py": BOTH,
    "export/test_compile_nopep523.py": BOTH,
    "export/test_compile_refuse.py": BOTH,
    "export/test_compile_qwen3_decode.py": BOTH,
    "export/test_compile_skip.py": BOTH,
    "export/test_compile_static_cache.py": BOTH,
    # Issue #315. macOS: the host shim is built with the `executorch` feature
    # there (install_shim.sh / run.sh), and a missing runtime is a FAIL. Linux:
    # the feature cannot exist on that target (torch_c/Cargo.toml), so the
    # runtime tests skip by name and the one that checks the platform rule
    # runs. Windows stays out by WHY_NOT_ON's runner-wide reason.
    "export/test_et_runtime.py": BOTH,
    "export/test_exir_shim.py": BOTH,
    # Issue #357: the linked flatc. Same platforms and reasons as
    # test_et_runtime.py: a FAIL on macOS when the binding is missing, skips by
    # name on Linux, where the runtime cannot be built.
    "export/test_et_flatc.py": BOTH,
    # Issue #423 (R5-S1): the Rust serializer. The program part is pure Rust and
    # runs on both; the XNNPACK payload needs the linked parser, so on Linux
    # those tests skip by name, as test_et_flatc.py does.
    "export/test_et_serialize.py": BOTH,
    # Issue #423 (R5-V): the vendored exir lowers to executorch's bytes. Portable
    # programs on both; the XNNPACK ones need _et_serialize_xnnpack, so on Linux
    # they must refuse by name.
    "export/test_et_vendored.py": BOTH,
    "export/test_export.py": BOTH,
    "export/test_export4.py": BOTH,
    "export/test_export5.py": BOTH,
    "export/test_export6.py": BOTH,
    "export/test_fusion_capture.py": BOTH,
    "export/test_fusion_patterns.py": BOTH,
    "export/test_functionalize_export.py": BOTH,
    # Issue #372: prints the CIA ops upstream's exports of the tiny SmolLM2 and
    # Qwen3 decompose (upstream only; the shim side is a table read).
    "export/test_cia_census.py": BOTH,
    "export/test_liftfresh.py": BOTH,
    "export/test_structseq.py": BOTH,
    "export/test_varmean.py": BOTH,
    "gate/test_benchguard_ab_upstream.py": BOTH,
    "gate/test_docwatch_probes.py": BOTH,
    "gate/test_gatelock.py": BOTH,
    "gate/test_ledger_parallel.py": BOTH,
    "gate/test_ledger_shards.py": BOTH,
    "gate/test_mpsbatch.py": BOTH,
    "gate/test_skipvis.py": BOTH,
    "gate/test_split_probe.py": BOTH,
    "gate/test_tmpleak.py": BOTH,
    "gate/test_toolguard_run_preflight.py": LOCAL_ONLY,
    "golden/test_x86pins.py": BOTH,
    "models/test_cbidentity.py": BOTH,
    "models/test_cbpath.py": BOTH,
    "models/test_gguf.py": BOTH,
    "models/test_kvcache.py": BOTH,
    "models/test_cbwalls.py": BOTH,
    "models/test_higgs.py": BOTH,
    "models/test_qwen3.py": BOTH,
    "models/test_stream.py": BOTH,
    "models/test_voice3.py": BOTH,
    "models/test_voice4.py": BOTH,
    "numerics/test_agree.py": BOTH,
    "numerics/test_agree2.py": BOTH,
    "numerics/test_int8.py": BOTH,
    "numerics/test_q4ref.py": BOTH,
    "numerics/test_q4kwriter.py": BOTH,
    "numerics/test_qgemm.py": BOTH,
    "numerics/test_promote0d.py": BOTH,
    "numerics/test_remeasure2.py": BOTH,
    "numerics/test_rngstate.py": BOTH,
    "ops/test_canine.py": BOTH,
    "ops/test_cmpscalar.py": BOTH,
    "ops/test_complex.py": BOTH,
    "ops/test_cplx2.py": BOTH,
    "ops/test_equal_allclose.py": BOTH,
    "ops/test_fft.py": BOTH,
    "ops/test_glu.py": BOTH,
    "ops/test_indexsel.py": BOTH,
    "ops/test_last7.py": BOTH,
    "ops/test_metaemb.py": BOTH,
    "ops/test_metafam.py": BOTH,
    "ops/test_metastride.py": BOTH,
    "ops/test_metal_argsort_dtype.py": BOTH,
    "ops/test_pad.py": BOTH,
    "ops/test_repeat.py": BOTH,
    "ops/test_rnn.py": BOTH,
    "ops/test_scatter.py": BOTH,
    "ops/test_signedzero.py": BOTH,
    # The decode-shaped float32 mm kernel (docs/perf/MM_DECODE.md); routing
    # is asserted through counters, the kernel is routed on the gemm-backed
    # Linux build and skips by name on the Accelerate build.
    "ops/test_skinny_gemm.py": BOTH,
    "ops/test_sdpa_decode.py": BOTH,
    "ops/test_sortties.py": ALL,
    "ops/test_strided.py": BOTH,
    "ops/test_tail1.py": BOTH,
    "ops/test_tail2.py": BOTH,
    "ops/test_tail3.py": BOTH,
    "ops/test_nanreduce_tri.py": BOTH,
    "ops/test_tail4.py": BOTH,
    "ops/test_asstrided_inplace.py": BOTH,
    "ops/test_elemstride.py": BOTH,
    "ops/test_empty_storage_slice.py": BOTH,
    "ops/test_unsafe_index.py": BOTH,
    "ops/test_viewstride.py": BOTH,
    "ops/test_vmap.py": BOTH,
    "release/test_benchonly.py": BOTH,
    "release/test_cigate.py": BOTH,
    "release/test_cipub.py": BOTH,
    "release/test_distquick.py": BOTH,
    "release/test_docrefs.py": BOTH,
    "release/test_executorch_vendor.py": BOTH,
    "release/test_exir_vendor.py": BOTH,
    "release/test_ftwheel.py": BOTH,
    "release/test_layout.py": BOTH,
    "release/test_platver_harnesses.py": BOTH,
    "release/test_publish.py": BOTH,
    "release/test_release.py": BOTH,
    "release/test_style.py": BOTH,
    "release/test_toolguard_wheel_staging.py": LOCAL_ONLY,
    "release/test_wheelmatrix.py": LOCAL_ONLY,
    "training/test_backward10.py": BOTH,
    "training/test_ddp.py": BOTH,
    "training/test_doublebackward.py": BOTH,
    "training/test_eagerlife.py": BOTH,
    "training/test_nllreduce.py": BOTH,
    "training/test_testtime.py": BOTH,
    "training/test_train.py": BOTH,
}

#: Why a suite is kept off a runner. Required for every suite whose tuple is
#: not every runner, and forbidden for one that runs everywhere.
WHY_NOT = {
    "devices/coreml/test_coremlops.py": _ANE_SILENT,
    "devices/coreml/test_anedecode.py": _ANE_SILENT,
    "devices/coreml/test_anesubgraph.py": _ANE_SILENT,
    "devices/coreml/test_anetracer.py": _ANE_SILENT,
    "devices/coreml/test_anegenerate.py": (
        "downloads SmolLM2-135M (270 MB) and compiles 15 CoreML programs; what it "
        "asserts is a Neural Engine placement (MLComputePlan), which no hosted "
        "runner has, so on a runner it could only skip."),
    "devices/coreml/test_ane_subgraph_components.py": (
        "imports coremltools at module scope, so without it the suite dies "
        "at import, and what it measures is subgraph placement on the Apple "
        "Neural Engine, which no hosted runner has. It also has no `_skip` "
        "runner of its own."),
    "gate/test_toolguard_run_preflight.py": _DEV_TOOLCHAIN + (
        " Measured on run 37078300696 (macos): 3 of 3 tests FAIL with "
        "`known-good interpreter missing: .../.caches/spike-venv/bin/python`."),
    "release/test_toolguard_wheel_staging.py": _DEV_TOOLCHAIN + (
        " Measured on run 37078300696 (macos): its one test FAILs with "
        "FileNotFoundError on the same interpreter."),
    "release/test_wheelmatrix.py": _DEV_TOOLCHAIN + (
        " Measured on run 37078300696 (macos): 20 of 22 ok; "
        "`test_every_target_names_a_rust_target_that_is_installed` needs the "
        "nine cross targets the release machine has, and "
        "`test_build_pys_own_self_test_passes` asserts `LINUX SELF-TEST: PASS`, "
        "which build.py prints only where the target CPython distribution "
        "exists (it skips loudly otherwise). The 20 lose CI coverage until "
        "those two take the runner's toolchain into account; follow-up."),
}


# ---------------------------------------------------------------------------
# The partition
# ---------------------------------------------------------------------------

def check(on_disk):
    """Problems with the manifest against the suite files present, by name."""
    problems = []
    disk = set(on_disk)
    named = set(SUITES)
    for name in sorted(disk - named):
        problems.append(
            f"{name} is a gate suite in neither the CI list nor the local-only "
            f"list -- classify it in .github/scripts/gate_suites.py SUITES")
    for name in sorted(named - disk):
        problems.append(
            f"{name} is named in .github/scripts/gate_suites.py but is not a file under "
            f"tests/ -- remove it or restore the suite")
    for name, runners in sorted(SUITES.items()):
        unknown = sorted(set(runners) - set(RUNNERS))
        if unknown:
            problems.append(f"{name} names unknown runner(s) {unknown}")
        for runner in sorted(set(RUNNERS) - set(runners)):
            if not why_not(name, runner).strip():
                problems.append(f"{name} is kept off {runner!r} with no reason in WHY_NOT")
        try:
            which = shard(name)
        except ValueError as exc:
            problems.append(str(exc))
        else:
            if which not in SHARDS:
                problems.append(f"{name} has shard {which!r}, not one of {SHARDS}")
    return problems


def why_not(name, runner):
    """Why `name` is not run on `runner`: its own reason, else the runner's."""
    return WHY_NOT.get(name, "") or WHY_NOT_ON.get(runner, "")


# Which suites may run side by side (issue #335). The ledger runs these
# `TORCHNATIVE_GATE_JOBS` at a time and every other suite afterwards, alone.
# A suite is parallel-safe only if it is listed here: an unlisted folder is
# serial, so a new folder cannot become parallel by accident.
#
# Listed: folders whose suites compute on the CPU in their own process and
# keep their files in directories they create (read 2026-11-15: no fixed port,
# no fixed temp name, no build, no write into the tree or the stage).
# Not listed, and why:
PARALLEL_FOLDERS = ("bindings/", "export/", "numerics/", "ops/", "training/")
PARALLEL_SUITES = ("_support/test_shim.py",)
SERIAL_WHY = {
    "api/": "thread-state and free-threading suites: their timing is the test",
    "devices/": "device suites: CPU contention turns CoreML MLComputePlan "
                "answers silent (AGENTS 14.3), and a device is one resource",
    "distributed/": "process groups and collectives between local processes",
    "gate/": "the gate's own machinery: stage locks, log directories, run.sh "
             "preflights",
    "golden/": "pins measured against one build of the artefact",
    "models/": "whole checkpoints: memory, and one shared Hugging Face cache",
    "release/": "reads and stages build outputs and the tracked tree",
}


# Lanes (issue #343, after bundle run 43659159846). A suite that must not run
# beside its own kind may still run beside everything else: Metal suites share
# one GPU, distributed suites start process groups, but neither reads what the
# other or a CPU suite does. Each lane runs its suites one at a time, and the
# lanes run at the same time as each other and as the side-by-side suites.
# Every other serial suite stays exclusive: it runs after all of that, with
# nothing beside it (CoreML answers go silent under load, AGENTS 14.3; model
# suites hold whole checkpoints in memory; the gate and release suites read the
# stage and the tree).
#
# Measured on the macOS runner (job log of run 43659159846, GATE PROGRESS
# lines): devices/mps suites 150 to 220 s each, about 1300 s in all, and the
# distributed suites 150 to 250 s each, about 1420 s, all run alone, out of
# 2963 s spent in the alone phase by the 72 suites that finished.
LANES = {
    "devices/mps/": "mps",
    "distributed/": "distributed",
}


def lane(name):
    """The lane suite `name` (relative to tests/) runs in, or None.

    Only meaningful for a suite `serial_reason` names: a parallel-safe suite
    has no lane, and a serial suite with no lane is exclusive.
    """
    if serial_reason(name) is None:
        return None
    for prefix, which in LANES.items():
        if name.startswith(prefix):
            return which
    return None


def serial_reason(name):
    """Why suite `name` (relative to tests/) must run alone, or None.

    None means the ledger may run it beside other parallel-safe suites.
    """
    if name in PARALLEL_SUITES or name.startswith(PARALLEL_FOLDERS):
        return None
    for prefix, why in SERIAL_WHY.items():
        if name.startswith(prefix):
            return why
    return "its folder is not listed as parallel-safe in gate_suites.py"


# Shards (leader decision 2026-11-15; no new workflow or dependency, the jobs of
# one run). The macOS gate job was one runner for 3 jobs of work, and the
# slowest job of run 43670318805: the side-by-side pool 84 suites in 1751 s
# (longest test_shim, 464 s), lane "mps" 28 suites in 1964 s, lane
# "distributed" 16 suites in 2359 s, and the 62 exclusive suites in 687 s, on
# top of about 5 minutes of setup per job and about 4 minutes of DOCWATCH. Three
# jobs of one run each take one of those piles, so the macOS wall clock is the
# longest pile and not their sum. Every suite belongs to exactly one shard:
# `tests/gate/test_ledger_shards.py` checks it on the real table, and the report
# refuses a suite that ran in no shard or in two.
SHARDS = (1, 2, 3)
#: The lane each shard owns. A lane with no entry is refused by name, so a new
#: lane in `LANES` cannot fall into a shard nobody chose.
SHARD_OF_LANE = {"mps": 1, "distributed": 2}


def shard(name):
    """The shard (one of `SHARDS`) that runs suite `name` (relative to tests/).

    2: the `distributed` lane. 3: every exclusive suite (serial, no lane).
    1: everything else, the side-by-side pool and the `mps` lane.
    """
    if serial_reason(name) is None:
        return 1
    which = lane(name)
    if which is None:
        return 3
    if which not in SHARD_OF_LANE:
        raise ValueError(
            f"{name} is in lane {which!r}, which no shard owns -- add it to "
            f"SHARD_OF_LANE in .github/scripts/gate_suites.py")
    return SHARD_OF_LANE[which]


def ci_suites():
    """Suites at least one CI runner executes."""
    return sorted(n for n, r in SUITES.items() if r)


def local_only():
    """Suites no CI runner executes: they are verified only by the local gate."""
    return sorted(n for n, r in SUITES.items() if not r)


def plan(runner, on_disk):
    """`(run, skipped, problems)` for one runner.

    `skipped` maps each suite this runner does not execute to the reason. An
    unknown runner raises: a typo in the workflow must not plan for nothing.
    """
    if runner not in RUNNERS:
        raise ValueError(
            f"unknown CI runner {runner!r}; .github/scripts/gate_suites.py knows "
            f"{sorted(RUNNERS)}")
    problems = check(on_disk)
    run, skipped = [], {}
    for name in sorted(on_disk):
        runners = SUITES.get(name)
        if runners is None:
            continue
        if runner in runners:
            run.append(name)
        else:
            skipped[name] = why_not(name, runner) or "no reason recorded"
    return run, skipped, problems


def skip_line(name, runner, why):
    """The ledger line for a suite not executed on `runner`."""
    base = name.rsplit("/", 1)[-1]
    stem = base[:-3] if base.endswith(".py") else base
    return f"SKIP {stem}: (whole suite) -- not run on CI runner '{runner}': {why}"


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

_HEADER = re.compile(r"^--- (test_\w+\.py) ---$")
_WHOLE = "(whole suite) -- not run on CI runner"


def _sections(text):
    """`{suite basename: its log lines}` read from one gate log."""
    sections, current = {}, None
    for line in text.splitlines():
        m = _HEADER.match(line)
        if m:
            current = m.group(1)
            sections[current] = []
            continue
        if line.startswith(("SUITE LEDGER:", "SUITE SHARD:", "CI PLAN:")):
            current = None
        if current is not None:
            sections[current].append(line)
    return sections


def classify_log(text, on_disk):
    """Per suite: counts and a verdict, read from one runner's gate log.

    PASS        ran, every test ok
    PARTIAL     ran, some tests ok and some skipped by name
    UNVERIFIED  ran, but every test skipped: it verified nothing here
    NOT RUN     not executed on this runner (local-only here)
    FAIL        a test failed
    EMPTY       ran and printed no ok/FAIL/SKIP line at all
    MISSING     the log has no section for it
    """
    sections = _sections(text)
    rows = {}
    for name in on_disk:
        body = sections.get(name.rsplit("/", 1)[-1])
        if body is None:
            rows[name] = {"verdict": "MISSING", "ok": 0, "fail": 0, "skip": 0}
            continue
        ok = sum(1 for ln in body if re.match(r"^ok\s", ln))
        fail = sum(1 for ln in body if re.match(r"^FAIL\s", ln))
        skips = [ln for ln in body if re.match(r"^SKIP\s", ln)]
        if any(_WHOLE in ln for ln in skips):
            verdict = "NOT RUN"
        elif fail:
            verdict = "FAIL"
        elif ok and skips:
            verdict = "PARTIAL"
        elif ok:
            verdict = "PASS"
        elif skips:
            verdict = "UNVERIFIED"
        else:
            verdict = "EMPTY"
        rows[name] = {"verdict": verdict, "ok": ok, "fail": fail,
                      "skip": len(skips), "skips": skips}
    return rows


def _grab(text, prefix):
    lines = [ln for ln in text.splitlines() if ln.startswith(prefix)]
    return lines[-1] if lines else None


# ---------------------------------------------------------------------------
# Shard logs (leader decision 2026-11-15)
# ---------------------------------------------------------------------------

#: `suite_ledger.py --shard` prints these. A suite another shard owns gets one
#: `SHARD <suite>: ...` line in its log, which is neither ok, FAIL nor SKIP, and
#: each shard ends its ledger with one `SUITE SHARD:` line naming what it owned.
_SHARD_LINE = re.compile(r"^SUITE SHARD: k=(\d+) n=(\d+) owned=(\d+) names=([0-9a-f]{64})$")
_SHARDED = re.compile(r"^SHARD \w+: \(whole suite\) -- runs in shard \d+ of \d+")
_LEDGER_LINE = re.compile(r"^SUITE LEDGER: suites=\d+/\d+ ok=(\d+) FAIL=(\d+) SKIP=(\d+)")


def names_digest(names):
    """The digest a shard prints over the sorted names of the suites it owns."""
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def _count(lines):
    return [sum(1 for ln in lines if re.match(pat, ln))
            for pat in (r"^ok\s", r"^FAIL\s", r"^SKIP\s")]


def merge_shards(logs, on_disk):
    """`(text, problems)`: the shard logs of one runner as one gate log.

    `logs` is a list of `(label, text or None)`. The merged text has every
    suite's section taken from the one shard that ran it, and a
    `SUITE LEDGER:` line whose totals are the sum of those sections, so it
    reads exactly as the single runner's log would. A shard log that is
    missing, a suite that no shard ran, a suite that two shards ran, or a suite
    that ran in a shard the manifest does not give it, is a problem by name.
    """
    problems, shards = [], {}
    for label, text in logs:
        if text is None:
            problems.append(f"shard log {label} is missing: nothing it owned is verified")
            continue
        lines = [ln for ln in text.splitlines() if _SHARD_LINE.match(ln)]
        if not lines:
            problems.append(f"{label} has no 'SUITE SHARD:' line: it is not a shard "
                            f"log, or the ledger stopped before printing it")
            continue
        k, n, owned, digest = _SHARD_LINE.match(lines[-1]).groups()
        k, n = int(k), int(n)
        if n != len(SHARDS):
            problems.append(f"{label} is shard {k}/{n} but the manifest has {len(SHARDS)} shards")
        elif k in shards:
            problems.append(f"{label} and {shards[k][0]} are both shard {k}/{n}")
        else:
            shards[k] = (label, text, _sections(text), int(owned), digest)
    absent = [k for k in SHARDS if k not in shards]
    for k in absent:
        problems.append(f"no log for shard {k}/{len(SHARDS)}: its suites are not verified")

    base = {rel.rsplit("/", 1)[-1]: rel for rel in on_disk}
    ran_in = {rel: [] for rel in on_disk}
    totals = [0, 0, 0]
    for k, (label, text, secs, owned, digest) in sorted(shards.items()):
        mine = [base[b] for b, body in secs.items()
                if b in base and not any(_SHARDED.match(ln) for ln in body)]
        for rel in mine:
            ran_in[rel].append(k)
        if owned != len(mine) or digest != names_digest(mine):
            problems.append(
                f"{label}: its SUITE SHARD line names {owned} suite(s) "
                f"(names={digest[:12]}) but its log holds {len(mine)} "
                f"(names={names_digest(mine)[:12]})")
        mine_counts = [sum(c) for c in zip(*(_count(secs[rel.rsplit('/', 1)[-1]])
                                              for rel in mine))] if mine else [0, 0, 0]
        raw = [ln for ln in text.splitlines() if _LEDGER_LINE.match(ln)]
        if not raw or "NOT RECONCILED" in raw[-1]:
            problems.append(f"{label}: no reconciled SUITE LEDGER line")
        else:
            said = [int(x) for x in _LEDGER_LINE.match(raw[-1]).groups()]
            if said != mine_counts:
                problems.append(f"{label}: its ledger says ok/FAIL/SKIP {said} "
                                f"but its suites' sections read {mine_counts}")
        if _grab(text, "EXIT=") is None:
            problems.append(f"{label}: no EXIT= line, the job did not finish")
        totals = [a + b for a, b in zip(totals, mine_counts)]

    out, covered = [], 0
    for rel in on_disk:
        k = ran_in[rel]
        out.append(f"--- {rel.rsplit('/', 1)[-1]} ---")
        if len(k) == 1:
            covered += 1
            out += shards[k[0]][2][rel.rsplit("/", 1)[-1]]
            try:
                want = shard(rel)
            except ValueError as exc:
                problems.append(str(exc))
            else:
                if want != k[0]:
                    problems.append(f"{rel} ran in shard {k[0]} but the manifest gives it shard {want}")
        elif not absent:
            problems.append(f"{rel} was executed by "
                            + (f"shards {k}" if k else "no shard")
                            + ": exactly one shard must run each suite")
    for prefix in ("CI PLAN:", "VULKAN COVERAGE:", "DOCWATCH:"):
        for k in sorted(shards):
            line = _grab(shards[k][1], prefix)
            if line:
                out.append(line)
    exits = [_grab(t[1], "EXIT=") for _, t in sorted(shards.items())]
    exit_line = next((e for e in exits if e and e != "EXIT=0"), None) or (
        "EXIT=0" if exits and all(exits) and not absent else "EXIT=?")
    out.append("SUITE LEDGER: suites=%d/%d ok=%d FAIL=%d SKIP=%d (combined from "
               "shards %s of %d)" % (covered, len(on_disk), totals[0], totals[1],
                                    totals[2], ",".join(map(str, sorted(shards))),
                                    len(SHARDS)))
    out.append(exit_line)
    return "\n".join(out) + "\n", problems


def render_report(logs, on_disk, merge_problems=()):
    """`(markdown, ok)`. `logs` maps runner -> log text, or None if absent.

    `merge_problems` (from `merge_shards`) make `ok` False and are listed.

    `ok` is False when the partition is broken or a runner's log is missing
    or lacks a suite: those are the ways a subset reads as the full gate.
    Test failures are the gate jobs' verdict, and are shown, not re-judged.
    """
    ok = True
    out = ["<!-- torchnative-gate-report -->", "## Gate on CI (issue #24)", ""]
    problems = check(on_disk)
    ci, local = ci_suites(), local_only()
    out.append(
        f"**Full gate: {len(on_disk)} suites** = {len(ci)} run on at least one "
        f"CI runner + {len(local)} local-only. The local gate "
        f"(`tests/run.sh` on the Mac) is still the only run "
        f"that covers all {len(on_disk)}.")
    out.append("")
    if problems:
        ok = False
        out.append("### PARTITION BROKEN -- a suite is in neither list")
        out += [f"- {p}" for p in problems]
        out.append("")
    if merge_problems:
        ok = False
        out.append("### SHARD MERGE FAILED -- the shard logs do not add up to one gate")
        out += [f"- FAIL {p}" for p in merge_problems]
        out.append("")

    all_rows = {}
    out.append("| runner | exit | ran | not run here | PASS | PARTIAL | "
               "UNVERIFIED (all skipped) | FAIL | missing/empty | ledger |")
    out.append("|---|---|---|---|---|---|---|---|---|---|")
    for runner in RUNNERS:
        text = logs.get(runner)
        if text is None:
            ok = False
            out.append(f"| {runner} | -- | no log | | | | | | | the gate job "
                       f"produced no log: nothing on this runner is verified |")
            continue
        rows = classify_log(text, on_disk)
        all_rows[runner] = rows
        n = {v: sum(1 for r in rows.values() if r["verdict"] == v)
             for v in ("PASS", "PARTIAL", "UNVERIFIED", "NOT RUN", "FAIL",
                       "EMPTY", "MISSING")}
        if n["MISSING"]:
            ok = False
        exit_line = _grab(text, "EXIT=") or "EXIT=?"
        ledger = _grab(text, "SUITE LEDGER:") or "no SUITE LEDGER line"
        ran = len(on_disk) - n["NOT RUN"] - n["MISSING"]
        out.append(
            f"| {runner} | {exit_line[5:]} | {ran} | {n['NOT RUN']} | "
            f"{n['PASS']} | {n['PARTIAL']} | {n['UNVERIFIED']} | {n['FAIL']} | "
            f"{n['MISSING'] + n['EMPTY']} | `{ledger[len('SUITE LEDGER: '):][:60]}` |")
    out.append("")
    for runner in all_rows:
        text = logs[runner]
        for prefix in ("CI PLAN:", "VULKAN COVERAGE:", "DOCWATCH:"):
            line = _grab(text, prefix)
            out.append(f"- **{runner}** `{line}`" if line else
                       f"- **{runner}** no `{prefix}` line (the gate stopped before it)")
    out.append("")

    out.append(f"### Local-only: verified by no CI runner ({len(local)})")
    out.append("")
    for name in local:
        out.append(f"- `{name}` -- {WHY_NOT.get(name, '')}")
    out.append("")

    attention = []
    for runner, rows in all_rows.items():
        for name, r in sorted(rows.items()):
            if r["verdict"] in ("FAIL", "UNVERIFIED", "EMPTY", "MISSING"):
                attention.append(f"| `{name}` | {runner} | {r['verdict']} | "
                                 f"{r['ok']} | {r['fail']} | {r['skip']} |")
    out.append("### Suites that verified nothing, or failed, on a runner")
    out.append("")
    if attention:
        out.append("| suite | runner | verdict | ok | FAIL | SKIP |")
        out.append("|---|---|---|---|---|---|")
        out += attention
    else:
        out.append("None.")
    out.append("")

    out.append("<details><summary>Every suite, every runner</summary>")
    out.append("")
    out.append("| suite | " + " | ".join(RUNNERS) + " |")
    out.append("|---|" + "---|" * len(RUNNERS))
    for name in on_disk:
        cells = []
        for runner in RUNNERS:
            r = all_rows.get(runner, {}).get(name)
            if r is None:
                cells.append("no log")
            else:
                cells.append(f"{r['verdict']} ({r['ok']} ok / {r['fail']} FAIL / "
                             f"{r['skip']} SKIP)")
        out.append(f"| `{name}` | " + " | ".join(cells) + " |")
    out.append("")
    out.append("</details>")
    out.append("")
    out.append(
        "A SKIP is unverified on that runner, never passed. UNVERIFIED means the "
        "suite ran and every one of its tests skipped by name.")
    return "\n".join(out) + "\n", ok


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _disk(pytests):
    root = pathlib.Path(pytests)
    return sorted(p.relative_to(root).as_posix() for p in root.rglob("test_*.py")
                  if "__pycache__" not in p.parts)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--pytests", default=str(PYTESTS))
    sub.add_parser("list")
    r = sub.add_parser("report")
    r.add_argument("--pytests", default=str(PYTESTS))
    r.add_argument("--log", action="append", default=[], metavar="RUNNER=PATH")
    r.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    if args.cmd == "check":
        problems = check(_disk(args.pytests))
        for p in problems:
            print("gate_suites: FAIL -- " + p)
        if not problems:
            print(f"gate_suites: {len(SUITES)} suites classified: "
                  f"{len(ci_suites())} on CI, {len(local_only())} local-only")
            side = [n for n in SUITES if serial_reason(n) is None]
            laned = [n for n in SUITES if lane(n) is not None]
            print(f"gate_suites: {len(side)} suites may run side by side, "
                  f"{len(laned)} in lanes ({', '.join(sorted(set(LANES.values())))}), "
                  f"{len(SUITES) - len(side) - len(laned)} exclusive")
            print("gate_suites: shards " + ", ".join(
                f"{k}: {sum(1 for n in SUITES if shard(n) == k)}" for k in SHARDS))
        return 1 if problems else 0

    if args.cmd == "list":
        for runner in RUNNERS:
            run = [n for n, rs in SUITES.items() if runner in rs]
            print(f"{runner}: {len(run)} suites")
        print(f"local-only ({len(local_only())}):")
        for n in local_only():
            print(f"  {n}")
        return 0

    paths = {runner: [] for runner in RUNNERS}
    for entry in args.log:
        runner, _, listed = entry.partition("=")
        if runner not in RUNNERS:
            print(f"gate_suites: unknown runner {runner!r} in --log", file=sys.stderr)
            return 2
        paths[runner] += [x for x in listed.split(",") if x]
    disk = _disk(args.pytests)
    logs, merge_problems = {}, []
    for runner, plist in paths.items():
        texts = []
        for x in plist:
            p = pathlib.Path(x)
            texts.append((x, p.read_text(errors="replace") if p.is_file() else None))
        sharded = len(texts) > 1 or any(t is not None and "\nSUITE SHARD: " in "\n" + t
                                        for _, t in texts)
        if sharded:
            text, problems = merge_shards(texts, disk)
            logs[runner] = text
            for p in problems:
                print(f"gate_suites: FAIL -- {runner}: {p}")
            merge_problems += [f"{runner}: {p}" for p in problems]
        else:
            logs[runner] = texts[0][1] if texts else None
    md, ok = render_report(logs, disk, merge_problems)
    if args.out:
        pathlib.Path(args.out).write_text(md)
    print(md)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
