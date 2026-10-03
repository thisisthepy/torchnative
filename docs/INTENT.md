# Intent

Why torchnative exists, what it is for, and what it deliberately is not. This file is the
boundary for [`SPEC.md`](SPEC.md): **the spec may not go beyond it** (AGENTS.md rule 5).

Sources, in order of authority: the maintainer's design statement in
[`docs/design/DESIGN.md`](design/DESIGN.md) §0–§3 and §8, the project description in the former
`AGENTS.md` §11 and §20 (formerly `CLAUDE.md` §0 and §8), and the README's own positioning. Anything that
is this document's inference rather than a statement found there is marked
`> Inferred — confirm with the maintainer.`

---

## 1. The purpose

torchnative is an **on-device AI library**. Its goals, in the maintainer's words
(DESIGN.md §0):

1. cover **federated learning (FL)** and the whole of **test-time learning (TTL)** — which contains
   test-time adaptation (TTA) and test-time training (TTT) as nested cases, not siblings;
2. on top of that, provide **kernel optimisation across platforms** (flash-attention,
   flash-linear-attention and the like).

All of it rests on one premise: **a real PyTorch model has to actually run on the device.**

## 2. The premise — no façade

[PythonMultiplatform](https://github.com/thisisthepy/PythonMultiplatform) exists because a real
CPython on the device means real pip packages on the device. A façade that imitates the
`transformers` API would make that foundation pointless (DESIGN.md §1).

So the goal is not "something that looks like `transformers`" but **`import transformers`
succeeding on the device**, and the only obstacle is `torch`. The target users include vision
libraries (`test-time-adapters`: RT-DETR, YOLO11, Grounding DINO, ResNet) that no LLM-only engine
covers, so the answer has to be a general tensor layer.

## 3. The approach — replace `torch._C`, vendor the rest

PyTorch is mostly Python. The only native part is `torch._C` (ATen tensors, the dispatcher,
autograd), and PyTorch's own mobile build cannot produce it (`INTERN_BUILD_MOBILE` forces
`BUILD_PYTHON` off). Therefore:

- **Vendor upstream's Python tree unmodified and replace only `torch._C`** with a native Rust
  extension (DESIGN.md §2). The chase target shrinks from "the whole torch API" to "the ATen op
  set", which is far more stable.
- **The real `transformers`, the real `from_pretrained`, the real `generate`** run on top. Only the
  computation underneath is ours (AGENTS.md §11).
- **Correctness means agreeing with upstream PyTorch**, measured element-wise against upstream
  itself — "it runs" is not "it is a drop-in".
- **What is unsupported refuses by name.** A silent fallback or a plausible invented answer is
  a defect, not a partial feature (AGENTS.md §18).

## 4. What torchnative itself provides

torchnative does **not** replace torch; the torch layer is the *precondition*. On top of it,
torchnative's own surface is (DESIGN.md §2–§3):

| Area | Intent |
|---|---|
| `delta/` | **The core abstraction**: a weight delta over base weights whose *lifetime is part of its type*. TTA's adapted parameters and FL's local update are the same object with a different lifetime and destination. |
| `adapt/` | Adaptation that closes inside one device — stage 0 (no backward: e.g. normalisation statistics) and stage 1 (gradient: entropy, auxiliary tasks), separated **by type** so a backward-free build rejects gradient methods at import time. |
| `federated/` | A **layer above** `adapt/`, not a sibling: aggregation, communication, privacy. Using adaptation alone must not pull in the federated stack. Built on `torch.distributed`, because federated averaging *is* collective communication. |
| `kernels/` | A bundle resolver satisfying Hugging Face's `kernels` contract, with resolution **inverted from runtime download to build time** — iOS will not execute downloaded native code (DESIGN.md §8). |
| `api/` | Deployment, lifetime policy and device orchestration (`TorchNativeAPI`). |

## 5. Accelerators — fixed front end, swapped back end

**The front end is fixed; only the back end is swapped** (AGENTS.md §20):

```
front end (fixed)   real transformers · from_pretrained · generate
back end (swapped)  Apple → CoreML | Android → ExecuTorch / QNN | Windows → Intel NPU (OpenVINO)
```

The user keeps a real `nn.Module`; a `.pte` or vendor blob is an implementation detail behind a
module, never an object the user handles. Choosing a back end is about which reaches the vendor NPU
best, never about API shape. Quantisation follows the same module-replacement shape
(`docs/graph/QUANT2.md` §3).

## 6. Where it runs

The deployment target is an **embedded CPython 3.13+** inside Kotlin Multiplatform apps
(Android, iOS, desktop), via PythonMultiplatform (DESIGN.md §2). torchnative ships as **one
`cp313-abi3` wheel per platform** so one binary serves 3.13 and every later CPython.

> Inferred — confirm with the maintainer: Linux, Windows and WASM (Pyodide) wheels are in scope as
> *distribution* targets because the wheels exist and are published, but the intent statement
> (DESIGN.md) names the mobile/desktop embedding as the reason the project exists.

## 7. What torchnative is deliberately not

- **Not a reimplementation of PyTorch** and not a port of model architectures (contrast llama.cpp,
  MLC). Architectures should come for free once the operators exist.
- **Not an ahead-of-time export pipeline** as the user-facing model (contrast ExecuTorch/CoreML
  conversion). Graph capture exists only as the substrate NPU back ends consume.
- **Not `torch.compile`.** abi3 is chosen over PEP 523 frame evaluation; `torch.compile` is refused
  by name, permanently (`docs/graph/COMPILE.md`).
- **Not a façade** that imitates `transformers` (§2).
- **Not dependent on an installed upstream torch at runtime.** Upstream torch is a *comparison
  baseline* for tests only; the wheel *provides* `torch`.
- **Not a benchmark-scenario framework.** Delta lifetimes are driven by system events, not by the
  domain boundaries a benchmark hands you (DESIGN.md §3).
- **Not a wrapper that hides the model.** `model.to(device.npu)` returns the same `nn.Module`, so
  the model that trains and the model that runs on the NPU stay one object.

## 8. Development method

Intent → spec → test → code (AGENTS.md rule 5). Claims are graded *builds / reaches / agrees*
(AGENTS.md §16), and the documentation is a measured record that keeps its corrections visible.
