"""One portable ExecuTorch program and upstream's answers for it, as JSON on stdout.

Run as a script by `tests/export/test_et_runtime.py` (issue #315, step R1) in the
interpreter named by `TORCHNATIVE_EXIR_PYTHON` (executorch 1.4.1 over upstream
torch), never with the vendored tree on the path. It is a helper, not a suite: it
asserts nothing about torchnative.

What it reports:

    program      the `.pte` bytes of Linear(8, 4) + ReLU lowered by
                 `to_edge(...).to_executorch()` with no partitioner, base64, so
                 the bytes travel over the pipe and never touch a file
    operators    the program's operator table, `ns::name.overload`
    kernel_calls the operator of every KernelCall instruction, in order
    inputs       two inputs, as float.hex strings (exact)
    oracle       upstream's outputs for both inputs: from upstream's own runtime
                 (`executorch.extension.pybindings.portable_lib`) on these very
                 bytes when it loads in this interpreter, else from eager torch
                 (`oracle_kind` says which, and `oracle_error` why)
    eager32/64   eager upstream torch in float32 and float64, for the tolerance
                 (AGENTS.md section 16: the p90 of upstream's own f32-vs-f64
                 error, floored at 8 ulp)

The weights and inputs are built from `arange`, not from the RNG, and every value
is exactly representable in float32, so both sides start from the same bits.
"""

import base64
import hashlib
import json
import sys
import traceback

MARK = "ET_PROGRAM_JSON "


def _hex(values):
    return [float(v).hex() for v in values]


def _flat(t):
    return [float(v) for v in t.detach().reshape(-1).tolist()]


def main():
    report = {}
    import torch

    report["torch"] = torch.__version__
    report["shim"] = hasattr(torch._C, "_aten_implemented")
    report["torch_file"] = torch.__file__
    try:
        import executorch
        from executorch import exir
        from executorch.exir._serialize._program import deserialize_pte_binary
    except Exception as exc:  # noqa: BLE001 - the report names it
        report["error"] = f"import executorch: {type(exc).__name__}: {exc}"
        print(MARK + json.dumps(report))
        return 0
    try:
        from executorch.version import __version__ as et_version
    except Exception:  # noqa: BLE001
        et_version = getattr(executorch, "__version__", "unknown")
    report["executorch"] = et_version

    lin = torch.nn.Linear(8, 4)
    with torch.no_grad():
        lin.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32 - 0.5)
        lin.bias.copy_(torch.arange(4, dtype=torch.float32) / 4)
    model = torch.nn.Sequential(lin, torch.nn.ReLU()).eval()
    x1 = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16 - 0.5
    x2 = torch.arange(16, dtype=torch.float32).flip(0).reshape(2, 8) / 8 - 1.0
    inputs = [x1, x2]

    try:
        ep = torch.export.export(model, (x1,))
        buffer = exir.to_edge(ep).to_executorch().buffer
        assert isinstance(buffer, bytes), type(buffer)
    except Exception as exc:  # noqa: BLE001
        report["error"] = f"lowering: {type(exc).__name__}: {exc}"
        report["where"] = traceback.format_exc().splitlines()[-8:]
        print(MARK + json.dumps(report))
        return 0

    pte = deserialize_pte_binary(buffer)
    program = getattr(pte, "program", pte)
    plan = program.execution_plan[0]
    operators = [f"{op.name}.{op.overload}" if op.overload else op.name for op in plan.operators]
    kernel_calls = []
    for chain in plan.chains:
        for instruction in chain.instructions:
            args = instruction.instr_args
            if type(args).__name__ == "KernelCall":
                kernel_calls.append(operators[args.op_index])
    report.update(
        program=base64.b64encode(buffer).decode("ascii"),
        sha256=hashlib.sha256(buffer).hexdigest(),
        bytes=len(buffer),
        method=plan.name,
        operators=operators,
        kernel_calls=kernel_calls,
        delegates=[d.id for d in plan.delegates],
        inputs=[{"shape": list(x.shape), "values": _hex(_flat(x))} for x in inputs],
    )

    with torch.no_grad():
        report["eager32"] = [_hex(_flat(model(x))) for x in inputs]
        model64 = torch.nn.Sequential(torch.nn.Linear(8, 4), torch.nn.ReLU()).double().eval()
        model64[0].weight.copy_(lin.weight.double())
        model64[0].bias.copy_(lin.bias.double())
        report["eager64"] = [_hex(_flat(model64(x.double()))) for x in inputs]

    try:
        from executorch.extension.pybindings.portable_lib import (
            _load_for_executorch_from_buffer,
        )

        module = _load_for_executorch_from_buffer(buffer)
        report["oracle"] = [_hex(_flat(module.forward((x,))[0])) for x in inputs]
        report["oracle_kind"] = "upstream executorch portable_lib"
    except Exception as exc:  # noqa: BLE001 - recorded, and the eager oracle used instead
        report["oracle_error"] = f"{type(exc).__name__}: {exc}"[:800]
        report["oracle"] = report["eager32"]
        report["oracle_kind"] = "upstream eager torch float32"

    print(MARK + json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
