"""ExecuTorch programs and upstream's answers for them, as JSON on stdout.

Run as a script by `tests/export/test_et_runtime.py` (issue #315, step R1; issue
#337, step R2b) in the interpreter named by `TORCHNATIVE_EXIR_PYTHON` (executorch
1.4.1 over upstream torch), never with the vendored tree on the path. It is a
helper, not a suite: it asserts nothing about torchnative. What it does assert,
on the upstream side, is that the XNNPACK program really is delegated, so a test
cannot pass on a program that silently kept its portable operators.

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

    xnnpack      the same module lowered by `to_edge_transform_and_lower` with
                 `XnnpackPartitioner` (step R2b), with the same fields
                 (`program`, `operators`, `kernel_calls`, `delegates`, `oracle`),
                 plus `delegate_calls` (the backend id of every DelegateCall
                 instruction, in order) and `unregistered_program`: the same
                 program with every delegate id rewritten to
                 `unregistered_id` in the deserialized program and serialized
                 again, a backend no runtime registers. A failure in this
                 part is reported as `xnnpack.error` and leaves the portable
                 report intact.

The weights and inputs are built from `arange`, not from the RNG, and every value
is exactly representable in float32, so both sides start from the same bits.
"""

import base64
import hashlib
import json
import sys
import traceback

MARK = "ET_PROGRAM_JSON "

# A backend id no ExecuTorch runtime registers (the negative control of R2b).
UNREGISTERED_ID = "TorchnativeNoSuchBackend"
XNNPACK_ID = "XnnpackBackend"


def _hex(values):
    return [float(v).hex() for v in values]


def _flat(t):
    return [float(v) for v in t.detach().reshape(-1).tolist()]


def _describe(buffer, deserialize_pte_binary):
    """What the serialized program says: operators, calls and delegate ids."""
    pte = deserialize_pte_binary(buffer)
    program = getattr(pte, "program", pte)
    plan = program.execution_plan[0]
    operators = [f"{op.name}.{op.overload}" if op.overload else op.name for op in plan.operators]
    delegates = [d.id for d in plan.delegates]
    kernel_calls = []
    delegate_calls = []
    for chain in plan.chains:
        for instruction in chain.instructions:
            args = instruction.instr_args
            kind = type(args).__name__
            if kind == "KernelCall":
                kernel_calls.append(operators[args.op_index])
            elif kind == "DelegateCall":
                delegate_calls.append(delegates[args.delegate_index])
    return {
        "program": base64.b64encode(buffer).decode("ascii"),
        "sha256": hashlib.sha256(buffer).hexdigest(),
        "bytes": len(buffer),
        "method": plan.name,
        "operators": operators,
        "kernel_calls": kernel_calls,
        "delegates": delegates,
        "delegate_calls": delegate_calls,
    }


def _xnnpack(torch, exir, deserialize_pte_binary, model, inputs, eager32):
    """The XNNPACK-lowered program, upstream's answers and the negative control."""
    from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
    from executorch.exir._serialize._program import serialize_pte_binary

    ep = torch.export.export(model, (inputs[0],))
    lowered = exir.to_edge_transform_and_lower(ep, partitioner=[XnnpackPartitioner()])
    buffer = lowered.to_executorch().buffer
    assert isinstance(buffer, bytes), type(buffer)
    out = _describe(buffer, deserialize_pte_binary)

    # The assertions that keep a silently undelegated program out of the test:
    # the delegate table, the DelegateCall instructions and the absence of the
    # portable kernels Linear + ReLU would otherwise have used.
    assert out["delegates"] and set(out["delegates"]) == {XNNPACK_ID}, (
        f"the lowered program's delegate ids are {out['delegates']}, expected only {XNNPACK_ID}"
    )
    assert out["delegate_calls"] and set(out["delegate_calls"]) == {XNNPACK_ID}, (
        f"the lowered program's DelegateCall instructions go to {out['delegate_calls']}"
    )
    kept = [op for op in out["kernel_calls"] if any(k in op for k in ("addmm", "linear", "relu", "mm."))]
    assert not kept, f"the partitioner left Linear + ReLU operators as KernelCalls: {kept}"

    # The negative control: the same program, every delegate id renamed in the
    # deserialized program, serialized again. Read back to prove the rename is
    # in the bytes.
    pte = deserialize_pte_binary(buffer)
    renamed = 0
    for plan in pte.program.execution_plan:
        for delegate in plan.delegates:
            delegate.id = UNREGISTERED_ID
            renamed += 1
    assert renamed == len(out["delegates"]), (renamed, out["delegates"])
    fake = bytes(serialize_pte_binary(pte, extract_delegate_segments=True))
    back = _describe(fake, deserialize_pte_binary)
    assert back["delegates"] == [UNREGISTERED_ID] * renamed, back["delegates"]
    assert back["delegate_calls"] == [UNREGISTERED_ID] * len(out["delegate_calls"]), back["delegate_calls"]
    assert back["kernel_calls"] == out["kernel_calls"], (back["kernel_calls"], out["kernel_calls"])
    out["unregistered_program"] = base64.b64encode(fake).decode("ascii")
    out["unregistered_id"] = UNREGISTERED_ID
    out["unregistered_bytes"] = len(fake)

    try:
        from executorch.extension.pybindings.portable_lib import (
            _load_for_executorch_from_buffer,
        )

        module = _load_for_executorch_from_buffer(buffer)
        out["oracle"] = [_hex(_flat(module.forward((x,))[0])) for x in inputs]
        out["oracle_kind"] = "upstream executorch portable_lib with its XNNPACK backend"
    except Exception as exc:  # noqa: BLE001 - recorded, and the eager oracle used instead
        out["oracle_error"] = f"{type(exc).__name__}: {exc}"[:800]
        out["oracle"] = eager32
        out["oracle_kind"] = "upstream eager torch float32"
    return out


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

    report.update(_describe(buffer, deserialize_pte_binary))
    report["inputs"] = [{"shape": list(x.shape), "values": _hex(_flat(x))} for x in inputs]

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

    try:
        report["xnnpack"] = _xnnpack(torch, exir, deserialize_pte_binary, model, inputs, report["eager32"])
    except Exception as exc:  # noqa: BLE001 - the report names it; the portable part stands
        report["xnnpack"] = {
            "error": f"{type(exc).__name__}: {exc}"[:800],
            "where": traceback.format_exc().splitlines()[-8:],
        }

    print(MARK + json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
