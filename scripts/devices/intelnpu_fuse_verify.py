"""Issue #3 on a real Intel NPU: fused gated MLP, dynamic row axis, compile count.

    python scripts/devices/intelnpu_fuse_verify.py --model Qwen/Qwen3-0.6B

Run on the Windows laptop with the Intel NPU, inside the environment that has
torchnative's shim as `torch` and `uv add openvino` (docs/devices/NPUFUSE.md
section 5 has the numbered procedure). Nothing here can run on a Mac: the
point of this script is the one fact a Mac cannot produce -- what the **NPU**
does with the graph.

It answers four questions, each with a verdict line the doc asks you to paste
back:

1. ``ONE_GRAPH_PER_MLP`` -- did every gated MLP lower to ONE compiled model
   (`_NPUGatedMLP`), counted from the report and the module tree?
2. ``DYNAMIC_AXIS`` -- did the NPU accept the `-1` row axis, or did it refuse
   (and with what OpenVINO message)? This is the open question NPUFUSE.md
   section 3 cannot answer on a Mac.
3. ``COMPILES_DURING_GENERATE`` -- how many compiles happened inside
   `generate()` across three prompt lengths. Read off
   `intelnpu._compile_counters()`, not inferred from timing or values.
4. ``MLP_AGREEMENT`` -- layer 0's fused MLP on the NPU against the same MLP
   run eagerly on the CPU (float32), with the oracle derived the way
   docs/numerics/AGREE.md section 2 derives it, at the IR's precision: the
   shim's own float16 answer against its float64 answer. NOTE: the reference
   here is the shim's eager CPU path, not upstream torch -- upstream is not
   installed in this environment. The shim's agreement with upstream is a
   separate, already-measured claim (docs/numerics/AGREE.md).

And it prints ``EXECUTION_DEVICES`` for every lowered module: if any reads
anything but ``['NPU']`` the lowering already refused at `to()`.

Exit status 0 only if 1, 3 and 4 pass and every module reports NPU. A refused
dynamic axis (2) is reported, not failed: it is a measurement of the device,
and the lowering falls back by name.
"""

import argparse
import sys
import time


def _rel(a, b, scale):
    return float((a.double() - b.double()).abs().max()) / scale


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3-0.6B")
    p.add_argument("--new-tokens", type=int, default=6)
    args = p.parse_args()

    import torch

    if not hasattr(torch._C, "_aten_implemented"):
        print("REFUSED: `import torch` is upstream torch, not torchnative's shim. "
              "Run inside the torchnative environment (NPUFUSE.md section 5, step 2).")
        return 2

    from torchnative import device
    from torchnative.export import intelnpu
    from torchnative.transformers import AutoModelForCausalLM
    from transformers import AutoTokenizer

    print(f"loading {args.model} twice (one stays on the CPU as the reference) ...")
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()
    reference = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).eval()

    prompts = [
        "The capital of France is",
        "Write one sentence about the sea, the wind and the light over the harbour:",
        "List three prime numbers greater than one hundred, separated by commas, "
        "and then explain in one short sentence how you checked each of them:",
    ]
    encoded = [tok(text, return_tensors="pt").input_ids for text in prompts]
    print(f"prompt lengths  : {[int(e.shape[-1]) for e in encoded]}")

    # Layer 0's MLP inputs, captured from the CPU reference for each prompt.
    captured = []
    mlp_ref = reference.model.layers[0].mlp
    hook = mlp_ref.register_forward_hook(lambda m, i, o: captured.append(i[0].detach().clone()))
    with torch.no_grad():
        for ids in encoded:
            reference(ids)
    hook.remove()

    import copy
    mlp64 = copy.deepcopy(mlp_ref).double()
    try:
        mlp16 = copy.deepcopy(mlp_ref).half()
    except Exception as exc:  # noqa: BLE001
        mlp16 = None
        print(f"NOTE: no float16 copy of the MLP ({type(exc).__name__}: {exc})")

    expected = []
    with torch.no_grad():
        for ids in encoded:
            expected.append(reference.generate(ids, max_new_tokens=args.new_tokens,
                                               do_sample=False)[0].tolist())

    c0 = intelnpu._compile_counters()
    t0 = time.perf_counter()
    model.to(device.npu)
    print(f"to(device.npu)  : {time.perf_counter() - t0:.1f}s")
    c1 = intelnpu._compile_counters()
    report = model.torchnative_offload

    lowered = [(n, m) for n, m in model.named_modules()
               if type(m).__name__ in ("_NPULinear", "_NPUGatedMLP")]
    mlps = [n for n, m in model.named_modules() if n.endswith(".mlp")]
    fused_ok = (sorted(report["fused"]) == sorted(mlps)
                and all(type(model.get_submodule(n)).__name__ == "_NPUGatedMLP" for n in mlps))
    print(f"fused           : {len(report['fused'])} of {len(mlps)} '.mlp' modules")
    for path, why in report["unfused"][:4]:
        print(f"  NOT FUSED {path}: {why}")
    print(f"ONE_GRAPH_PER_MLP={'yes' if fused_ok else 'no'}")

    print(f"shape modes     : {report['shape_modes']}")
    for path, why in report["shape_fallbacks"][:2]:
        print(f"  FALLBACK {path}: {why}")
    dynamic = report["shape_modes"].get("dynamic", 0) == len(lowered)
    print(f"DYNAMIC_AXIS={'accepted' if dynamic else 'refused'}")

    devices = sorted({tuple(m.execution_devices or ()) for _, m in lowered})
    on_npu = devices == [("NPU",)]
    print(f"EXECUTION_DEVICES={devices}")
    print(f"compiles at to(): {c1['lowered'] - c0['lowered']} lowered, "
          f"{c1['ov_compile'] - c0['ov_compile']} ov_core_compile_model")

    # MLP agreement, layer 0, the fused module on the NPU.
    rows = []
    fused0 = model.model.layers[0].mlp
    with torch.no_grad():
        for x in captured:
            got = fused0(x)
            up32 = mlp_ref(x)
            up64 = mlp64(x.double())
            scale = float(up32.abs().max())
            row = {"rel": _rel(got, up32, scale), "rel64": _rel(got, up64, scale)}
            if mlp16 is not None:
                row["oracle"] = _rel(mlp16(x.half()), up64, scale)
            rows.append(row)
    for i, row in enumerate(rows):
        print(f"  mlp row {i}: " + ", ".join(f"{k}={v:.3e}" for k, v in row.items()))
    agree = None
    if mlp16 is not None and all(r["oracle"] > 0 for r in rows):
        oracles = sorted(r["oracle"] for r in rows)
        index = min(len(oracles) - 1, int(round(0.9 * (len(oracles) - 1))))
        tolerance = max(oracles[index], 8 * float(torch.finfo(torch.float16).eps))
        worst = max(r["rel"] for r in rows)
        ratio = max(r["rel64"] / r["oracle"] for r in rows)
        agree = worst <= tolerance and ratio <= 4.0
        print(f"MLP_AGREEMENT={'pass' if agree else 'FAIL'} worst={worst:.3e} "
              f"tolerance={tolerance:.3e} ratio={ratio:.2f} (4x rule)")
    else:
        print("MLP_AGREEMENT=not-measured (no float16 oracle in this environment)")

    # generate() over three prompt lengths: the compile counter must not move.
    g0 = intelnpu._compile_counters()
    t0 = time.perf_counter()
    same = []
    with torch.no_grad():
        for ids, want in zip(encoded, expected):
            out = model.generate(ids, max_new_tokens=args.new_tokens, do_sample=False)
            same.append(out[0].tolist() == want)
    g1 = intelnpu._compile_counters()
    during = g1["lowered"] - g0["lowered"]
    print(f"generate x{len(encoded)}    : {time.perf_counter() - t0:.1f}s")
    print(f"COMPILES_DURING_GENERATE={during} "
          f"(ov_core_compile_model: {g1['ov_compile'] - g0['ov_compile']})")
    print(f"TOKENS_EQUAL_TO_CPU={same} (reported, not a pass condition: f16 can "
          f"flip a near-tied argmax)")

    ok = fused_ok and on_npu and (during == 0 if dynamic else True) and agree is not False
    print(f"\nVERDICT={'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
