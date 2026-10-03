"""What Metal and CoreML actually do on a GitHub-hosted macOS runner.

Issue #24 asks for a macOS gate job "for what can run there without a GPU",
and for the runner's limits to be stated from a measurement rather than
assumed. GitHub documents that hosted arm64 macOS runners are VMs under
Apple's Virtualization framework and that MPS is not supported there. That is
a statement about Metal Performance Shaders; it says nothing direct about
plain Metal compute (which candle's backend uses through its own kernels) or
about CoreML's MLComputePlan. This script asks each one at runtime and prints
one `PROBE <what>: <answer>` line per question.

It is a measurement, not a check: every question that cannot be answered is
printed as REFUSED/UNAVAILABLE with the reason, and the script exits 0. The
gate job's verdict comes from `run.sh`, not from here. AGENTS.md §16: "it ran
on the GPU" is asserted from a runtime answer -- a device name, a kernel's
read-back, `_metal_counters()` -- never inferred from a correct value.

    python tools/ci/macos_probe.py metal     # Metal device + one compute kernel (swift)
    python tools/ci/macos_probe.py shim      # the vendored shim: mps availability + counters
    python tools/ci/macos_probe.py coreml    # MLComputePlan for a one-op program (needs coremltools)
"""

import os
import platform
import subprocess
import sys
import tempfile

_SWIFT = r'''
import Foundation
import Metal
import MetalPerformanceShaders

guard let dev = MTLCreateSystemDefaultDevice() else {
    print("PROBE metal.device: NONE -- MTLCreateSystemDefaultDevice() returned nil")
    exit(0)
}
print("PROBE metal.device: \(dev.name)")
print("PROBE metal.mps_supports_device: \(MPSSupportsMTLDevice(dev))")
print("PROBE metal.families: apple7=\(dev.supportsFamily(.apple7)) apple8=\(dev.supportsFamily(.apple8)) mac2=\(dev.supportsFamily(.mac2))")
print("PROBE metal.recommended_working_set_bytes: \(dev.recommendedMaxWorkingSetSize)")
let src = "kernel void add1(device float* a [[buffer(0)]], uint i [[thread_position_in_grid]]) { a[i] = a[i] + 1.0; }"
do {
    let lib = try dev.makeLibrary(source: src, options: nil)
    let pso = try dev.makeComputePipelineState(function: lib.makeFunction(name: "add1")!)
    var data: [Float] = [1, 2, 3, 4]
    let buf = dev.makeBuffer(bytes: &data, length: 16, options: .storageModeShared)!
    let cb = dev.makeCommandQueue()!.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(buf, offset: 0, index: 0)
    enc.dispatchThreads(MTLSize(width: 4, height: 1, depth: 1),
                        threadsPerThreadgroup: MTLSize(width: 4, height: 1, depth: 1))
    enc.endEncoding()
    cb.commit()
    cb.waitUntilCompleted()
    let p = buf.contents().bindMemory(to: Float.self, capacity: 4)
    let got = [p[0], p[1], p[2], p[3]]
    let verdict = (got == [2, 3, 4, 5]) ? "AGREES" : "WRONG"
    print("PROBE metal.compute: \(verdict) -- status=\(cb.status.rawValue) error=\(String(describing: cb.error)) got=\(got) expected=[2.0, 3.0, 4.0, 5.0]")
} catch {
    print("PROBE metal.compute: REFUSED -- \(error)")
}
'''


def _metal():
    print(f"PROBE host: {platform.platform()} {platform.machine()}")
    try:
        hw = subprocess.run(["sysctl", "-n", "hw.model"], capture_output=True, text=True)
        print(f"PROBE host.hw_model: {hw.stdout.strip() or hw.stderr.strip()}")
        sp = subprocess.run(["system_profiler", "SPDisplaysDataType"],
                            capture_output=True, text=True)
        for line in sp.stdout.splitlines():
            if line.strip():
                print(f"PROBE displays: {line.rstrip()}")
    except OSError as e:
        print(f"PROBE host: UNAVAILABLE -- {e}")
    with tempfile.TemporaryDirectory(dir=os.environ.get("RUNNER_TEMP")) as d:
        path = os.path.join(d, "metal_probe.swift")
        with open(path, "w") as fh:
            fh.write(_SWIFT)
        try:
            proc = subprocess.run(["xcrun", "swift", path], capture_output=True, text=True)
        except OSError as e:
            print(f"PROBE metal: UNAVAILABLE -- cannot run swift: {e}")
            return 0
        sys.stdout.write(proc.stdout)
        if proc.returncode != 0:
            print(f"PROBE metal: REFUSED -- swift exited {proc.returncode}: "
                  f"{proc.stderr.strip()[-600:]}")
    return 0


def _shim():
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"PROBE shim: UNAVAILABLE -- import torch failed: {e}")
        return 0
    if not hasattr(torch._C, "_aten_implemented"):
        print("PROBE shim: WRONG TORCH -- this is upstream torch, not the shim "
              "(AGENTS.md §15.2); nothing below would be about this project")
        return 0
    C = torch._C
    print(f"PROBE shim.mps_is_built: {torch.backends.mps.is_built()}")
    print(f"PROBE shim.mps_is_available: {torch.backends.mps.is_available()}")
    before = C._metal_counters()
    print(f"PROBE shim.metal_counters.before: {before}")
    try:
        x = torch.ones(4, device="mps")
        y = (x + 1).cpu().tolist()
        after = C._metal_counters()
        print(f"PROBE shim.mps_add: got={y} expected=[2.0, 2.0, 2.0, 2.0]")
        print(f"PROBE shim.metal_counters.after: {after}")
        moved = {k: (after.get(k) or 0) - (before.get(k) or 0) for k in after}
        print(f"PROBE shim.metal_counters.delta: {moved} -- compute_encoders > 0 "
              "is the runtime evidence a kernel was launched on the device")
    except Exception as e:  # noqa: BLE001
        print(f"PROBE shim.mps_add: REFUSED -- {type(e).__name__}: "
              f"{str(e).splitlines()[0] if str(e) else ''}")
    return 0


def _coreml():
    try:
        import numpy as np
        import coremltools as ct
        from coremltools.converters.mil import Builder as mb
        from coremltools.models.compute_plan import MLComputePlan
    except Exception as e:  # noqa: BLE001
        print(f"PROBE coreml: UNAVAILABLE -- {type(e).__name__}: {e}")
        return 0
    print(f"PROBE coreml.coremltools: {ct.__version__}")

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 64, 64))])
    def prog(x):
        return mb.relu(x=mb.matmul(x=x, y=x))

    with tempfile.TemporaryDirectory(dir=os.environ.get("RUNNER_TEMP")) as d:
        try:
            model = ct.convert(prog, convert_to="mlprogram",
                               compute_units=ct.ComputeUnit.ALL)
            pkg = os.path.join(d, "probe.mlpackage")
            model.save(pkg)
            compiled = ct.utils.compile_model(pkg, os.path.join(d, "probe.mlmodelc"))
            print(f"PROBE coreml.compile: ok -- {os.path.basename(str(compiled))}")
        except Exception as e:  # noqa: BLE001
            print(f"PROBE coreml.compile: REFUSED -- {type(e).__name__}: {e}")
            return 0
        try:
            x = np.ones((1, 64, 64), dtype=np.float32)
            out = model.predict({"x": x})
            val = float(next(iter(out.values())).reshape(-1)[0])
            print(f"PROBE coreml.predict: got={val} expected=64.0")
        except Exception as e:  # noqa: BLE001
            print(f"PROBE coreml.predict: REFUSED -- {type(e).__name__}: {e}")
        for units in ("ALL", "CPU_AND_NE"):
            try:
                plan = MLComputePlan.load_from_path(
                    path=str(compiled), compute_units=getattr(ct.ComputeUnit, units))
                ops = plan.model_structure.program.functions["main"].block.operations
                for op in ops:
                    if op.operator_name == "const":
                        continue
                    usage = plan.get_compute_device_usage_for_mlprogram_operation(op)
                    if usage is None:
                        print(f"PROBE coreml.plan[{units}] {op.operator_name}: "
                              "preferred=unknown supported=[] (CoreML declined to answer)")
                        continue
                    pref = type(usage.preferred_compute_device).__name__
                    sup = sorted(type(s).__name__ for s in usage.supported_compute_devices)
                    print(f"PROBE coreml.plan[{units}] {op.operator_name}: "
                          f"preferred={pref} supported={sup}")
            except Exception as e:  # noqa: BLE001
                print(f"PROBE coreml.plan[{units}]: REFUSED -- {type(e).__name__}: {e}")
    return 0


def main(argv):
    what = argv[1] if len(argv) > 1 else ""
    table = {"metal": _metal, "shim": _shim, "coreml": _coreml}
    if what not in table:
        print(__doc__)
        return 2
    return table[what]()


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
