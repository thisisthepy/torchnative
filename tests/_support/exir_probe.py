"""One exir lowering, stage by stage, reported as JSON (issue #306, ExecuTorch step P3).

Run as a script by `tests/export/test_exir_shim.py`, twice: once with the vendored
tree on `PYTHONPATH` (torchnative's shim) and once without it (upstream torch, the
oracle). It is a helper, not a suite: it asserts nothing about agreement. It says
which torch it ran on, how far each stage got, and what the resulting programs
contain, and the suite compares the two reports.

Stages, each recorded as its result or as the named error that stopped it:

    import_exir     import executorch.exir
    import_xnnpack  import the XNNPACK partitioner
    edge_dtypes     the edge dialect's dtype table for `aten.linear.default`: which
                    arguments it holds (by `str(arg.type)`), and whether it accepts
                    float32 everywhere (issue #306 wall W8, CI run 43793269977)
    export          torch.export.export of Linear(8, 4) + ReLU
    edge            to_edge(...): the edge dialect's operator targets
    portable        .to_executorch(): program bytes, operators, delegates
    xnnpack         to_edge_transform_and_lower(XnnpackPartitioner).to_executorch()
    strict_export_interpolate
                    torch.export.export(F.interpolate bilinear, strict=True), the
                    capture the XNNPACK stage's first pass makes; report only

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


def _where(exc):
    tail = traceback.extract_tb(exc.__traceback__)[-6:]
    return [f"{f.filename.split('site-packages/')[-1]}:{f.lineno} {f.name}" for f in tail]


def _frames(exc):
    return [
        f"{f.filename.split('site-packages/')[-1]}:{f.lineno} {f.name}"
        for f in traceback.extract_tb(exc.__traceback__)
    ]


def _repeat(frames, head=20, max_period=64):
    """The repeating window at the end of a recursion's frames, or None.

    `frames` are one string per traceback frame, outermost first. The shortest
    window of at most `max_period` frames that repeats itself at least three
    times at the end is the cycle. Returned: the first `head` frames (stopping
    where the repetition starts), how many frames lie between those and the
    first repetition, the window once, its length and how many times it
    repeats. A traceback with no such window answers None.
    """
    n = len(frames)
    for period in range(1, min(max_period, n // 3) + 1):
        window = frames[n - period:]
        count = 1
        while n - (count + 1) * period >= 0 and (
            frames[n - (count + 1) * period: n - count * period] == window
        ):
            count += 1
        if count >= 3:
            start = n - count * period
            return {
                "head": frames[:min(head, start)],
                "skipped": max(0, start - head),
                "cycle": frames[start:start + period],
                "length": period,
                "count": count,
            }
    return None


def _record(exc):
    record = {"error": f"{type(exc).__name__}: {exc}"[:1500], "where": _where(exc)}
    if isinstance(exc, RecursionError):
        # The last six frames of a recursion are one fragment of its cycle and
        # name nothing. CI run after #407 (issue #306) stopped the XNNPACK
        # stage on "RecursionError: maximum recursion depth exceeded" with no
        # usable stack. The frames are recorded deduplicated: the head of the
        # traceback, then the repeating window once with its length and count.
        frames = _frames(exc)
        repeat = _repeat(frames)
        record["recursion"] = {"frames": len(frames), **(repeat or {"head": frames[:20], "cycle": None})}
    return record


def _error(exc):
    record = _record(exc)
    # exir's pass manager re-raises every pass failure as a bare `Exception`
    # naming the pass (`exir/pass_manager.py:234`, `raise ... from e`), so the
    # outer exception says where and never what. CI run 43802210944 stopped on
    # "An error occurred when running the 'ConvertToUpsampleBilinear2d' pass"
    # with the root exception lost. The chain is recorded, root last.
    causes, seen, cause = [], set(), exc.__cause__ or exc.__context__
    while cause is not None and id(cause) not in seen and len(causes) < 8:
        seen.add(id(cause))
        causes.append(_record(cause))
        cause = cause.__cause__ or cause.__context__
    if causes:
        record["causes"] = causes
    return record


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

    def edge_dtypes():
        # The verifier's own objects (`exir/verification/arg_validator.py` and
        # `exir/dialects/edge/_ops.py`), on the op the XNNPACK stage preserves.
        # `str(arg.type)` is what `exir/dialects/edge/spec/utils.py` decides
        # tensor-ness by; the table's keys are the names that survived it.
        from executorch.exir.dialects._ops import ops as exir_ops

        schema = torch.ops.aten.linear.default._schema
        edge_op = exir_ops.edge.aten.linear.default
        constraint = edge_op._schema.dtype_constraint
        f32 = torch.float32
        types = {"input": f32, "weight": f32, "bias": f32, "__ret_0": f32}
        return None, {
            "arg_types": [[a.name, str(a.type)] for a in list(schema.arguments) + list(schema.returns)],
            "table_names": sorted(constraint.type_constraint[0]),
            "float32_everywhere_valid": bool(constraint.validate(types)),
            "float32_in_allowed": f32 in constraint["input"],
        }

    run("edge_dtypes", ["import_exir"], edge_dtypes)

    def clone_memory_format():
        # Issue #306 (CI run 44423485167): `ConvertToSDPAPass` lowers an SDPA
        # graph containing `aten.clone.default(memory_format=
        # torch.contiguous_format)` after `permute_copy`, and
        # `MemoryFormatOpsPass` runs `copy.deepcopy(kwargs)` before handing
        # `memory_format` to `dim_order_utils.get_dim_order`.
        import copy
        from executorch.exir.dim_order_utils import get_dim_order

        class PermuteClone(torch.nn.Module):
            def forward(self, inp):
                return inp.permute(1, 0).clone(memory_format=torch.contiguous_format)

        exported = torch.export.export(PermuteClone(), (torch.zeros(2, 3),))
        node = next(
            n for n in exported.graph.nodes
            if n.op == "call_function" and "clone" in str(n.target)
        )
        mf = node.kwargs["memory_format"]
        cmf = dict(copy.deepcopy(node.kwargs))["memory_format"]
        return None, {
            "node_is": mf is torch.contiguous_format,
            "node_eq": mf == torch.contiguous_format,
            "node_in": mf in [torch.contiguous_format],
            "deepcopy_is": cmf is torch.contiguous_format,
            "deepcopy_eq": cmf == torch.contiguous_format,
            "deepcopy_in": cmf in [torch.contiguous_format],
            "dim_order": get_dim_order(cmf, 2),
        }

    run("clone_memory_format", ["import_exir"], clone_memory_format)
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
    # A complete report before the report-only stage below, flushed, so a
    # crash inside Dynamo cannot take the stages above with it; the suite reads
    # the last report line.
    print(MARK + json.dumps(report), flush=True)

    def strict_export_interpolate():
        # What the XNNPACK stage's first pass does before it looks at our
        # graph: `ConvertToUpsampleBilinear2d.call` builds its patterns with
        # `backends/xnnpack/partition/graphs/bilinear_2d.py`, which runs
        # `torch.export.export(..., strict=True)` (Dynamo's bytecode capture,
        # not the non-strict trace the stages above use) on an
        # `F.interpolate(mode="bilinear")` module and then `to_edge`. Run alone
        # so its outcome is recorded apart from the lowering (CI run
        # 43802210944). Report only: no test asserts on it yet.
        class Bilinear(torch.nn.Module):
            def forward(self, inp):
                return torch.nn.functional.interpolate(
                    inp, scale_factor=2, mode="bilinear", align_corners=False, antialias=False)

        sample = torch.arange(48, dtype=torch.float32).reshape(1, 3, 4, 4)
        strict = torch.export.export(Bilinear(), (sample,), strict=True)
        targets = [str(n.target) for n in strict.graph.nodes if n.op == "call_function"]
        return None, {"targets": targets}

    run("strict_export_interpolate", [], strict_export_interpolate)

    print(MARK + json.dumps(report))
    return 0


if __name__ == "__main__":
    sys.exit(main())
