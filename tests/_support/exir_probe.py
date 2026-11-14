"""One exir lowering, stage by stage, reported as JSON (issue #306, ExecuTorch step P3).

Run as a script by `tests/export/test_exir_shim.py`, twice: once with the vendored
tree on `PYTHONPATH` (torchnative's shim) and once without it (upstream torch, the
oracle). It is a helper, not a suite: it asserts nothing about agreement. It says
which torch it ran on, how far each stage got, and what the resulting programs
contain, and the suite compares the two reports.

Stages, each recorded as its result or as the named error that stopped it:

    import_exir     import executorch.exir
    import_xnnpack  import the XNNPACK partitioner
    export          torch.export.export of Linear(8, 4) + ReLU
    edge            to_edge(...): the edge dialect's operator targets
    portable        .to_executorch(): program bytes, operators, delegates
    xnnpack         to_edge_transform_and_lower(XnnpackPartitioner).to_executorch()

The program is `bytes` in this process. Nothing is written by this script; exir's
own serializers use temporary files that they delete (schema files, and the
delegate serializer's `flatc` scratch, which step P4 replaces).

The weights are filled from `arange`, not from the RNG, so both sides lower the
same numbers whatever their generators do.
"""

import hashlib
import json
import sys
import traceback

MARK = "EXIR_PROBE_JSON "


def _error(exc):
    tail = traceback.extract_tb(exc.__traceback__)[-6:]
    return {
        "error": f"{type(exc).__name__}: {exc}"[:1500],
        "where": [f"{f.filename.split('site-packages/')[-1]}:{f.lineno} {f.name}" for f in tail],
    }


def _program(buffer):
    from executorch.exir._serialize._program import deserialize_pte_binary

    assert isinstance(buffer, bytes), type(buffer)
    pte = deserialize_pte_binary(buffer)
    program = getattr(pte, "program", pte)
    plan = program.execution_plan[0]
    operators = [f"{op.name}.{op.overload}" if op.overload else op.name for op in plan.operators]
    return {
        "bytes": len(buffer),
        "sha256": hashlib.sha256(buffer).hexdigest(),
        "operators": operators,
        "delegates": [d.id for d in plan.delegates],
        "method": plan.name,
    }


def main():
    report = {"stages": {}}
    stages = report["stages"]

    import torch

    report["torch"] = torch.__version__
    report["shim"] = hasattr(torch._C, "_aten_implemented")
    report["torch_file"] = torch.__file__

    def run(name, needs, fn):
        missing = [n for n in needs if "error" in stages.get(n, {"error": "not run"})]
        if missing:
            stages[name] = {"error": f"not run: stage {missing[0]} did not complete", "where": []}
            return None
        try:
            value, record = fn()
        except Exception as exc:  # noqa: BLE001 - the report names it
            stages[name] = _error(exc)
            return None
        stages[name] = record
        return value

    def import_exir():
        import executorch.exir as exir

        return exir, {"ok": True}

    def import_xnnpack():
        from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

        return XnnpackPartitioner, {"ok": True}

    def module_and_input():
        lin = torch.nn.Linear(8, 4)
        with torch.no_grad():
            lin.weight.copy_(torch.arange(32, dtype=torch.float32).reshape(4, 8) / 32 - 0.5)
            lin.bias.copy_(torch.arange(4, dtype=torch.float32) / 4)
        model = torch.nn.Sequential(lin, torch.nn.ReLU()).eval()
        x = torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16 - 0.5
        return model, x

    def export():
        model, x = module_and_input()
        ep = torch.export.export(model, (x,))
        targets = [str(n.target) for n in ep.graph.nodes if n.op == "call_function"]
        return ep, {"targets": targets}

    exir = run("import_exir", [], import_exir)
    partitioner = run("import_xnnpack", [], import_xnnpack)
    ep = run("export", [], export)

    def edge():
        manager = exir.to_edge(ep)
        graph = manager.exported_program().graph
        # An edge target prints as "<EdgeOpOverload: aten.addmm.default>: schema = ...";
        # the name is what is compared, not the schema's spelling.
        targets = [
            str(n.target).split(">: schema")[0].lstrip("<")
            for n in graph.nodes
            if n.op == "call_function"
        ]
        return manager, {"targets": targets}

    manager = run("edge", ["import_exir", "export"], edge)

    def portable():
        buffer = manager.to_executorch().buffer
        return buffer, _program(buffer)

    run("portable", ["edge"], portable)

    def xnnpack():
        model, x = module_and_input()
        lowered = exir.to_edge_transform_and_lower(
            torch.export.export(model, (x,)), partitioner=[partitioner()]
        )
        buffer = lowered.to_executorch().buffer
        return buffer, _program(buffer)

    run("xnnpack", ["import_exir", "import_xnnpack", "export"], xnnpack)

    print(MARK + json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
