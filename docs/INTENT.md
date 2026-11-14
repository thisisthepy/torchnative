# Intent

Why torchnative exists, what it is for, and what it deliberately is not. This file is the
boundary for [`SPEC.md`](SPEC.md): **the spec may not go beyond it** (AGENTS.md rule 5).

Sources, in order of authority: the maintainer's design statement in
[`docs/design/DESIGN.md`](design/DESIGN.md) §0–§3 and §8, the project description in the former
`AGENTS.md` §11 and §20 (formerly `CLAUDE.md` §0 and §8), and the README's own positioning. Anything that
is this document's inference rather than a statement found there is marked
`> Inferred: confirm with the maintainer.`

---

## 1. The purpose

torchnative is an **on-device AI library**. Its goals, in the maintainer's words
(DESIGN.md §0):

1. cover **federated learning (FL)** and the whole of **test-time learning (TTL)**, which contains
   test-time adaptation (TTA) and test-time training (TTT) as nested cases, not siblings;
2. on top of that, provide **kernel optimisation across platforms** (flash-attention,
   flash-linear-attention and the like).

All of it rests on one premise: **a real PyTorch model has to actually run on the device.**

## 2. The premise: no façade

[PythonMultiplatform](https://github.com/thisisthepy/PythonMultiplatform) exists because a real
CPython on the device means real pip packages on the device. A façade that imitates the
`transformers` API would make that foundation pointless (DESIGN.md §1).

So the goal is not "something that looks like `transformers`" but **`import transformers`
succeeding on the device**, and the only obstacle is `torch`. The target users include vision
libraries (`test-time-adapters`: RT-DETR, YOLO11, Grounding DINO, ResNet) that no LLM-only engine
covers, so the answer has to be a general tensor layer.

## 3. The approach: replace `torch._C`, vendor the rest

PyTorch is mostly Python. The only native part is `torch._C` (ATen tensors, the dispatcher,
autograd), and PyTorch's own mobile build cannot produce it (`INTERN_BUILD_MOBILE` forces
`BUILD_PYTHON` off). Therefore:

- **Vendor upstream's Python tree unmodified and replace only `torch._C`** with a native Rust
  extension (DESIGN.md §2). The chase target shrinks from "the whole torch API" to "the ATen op
  set", which is far more stable.
- **The real `transformers`, the real `from_pretrained`, the real `generate`** run on top. Only the
  computation underneath is ours (AGENTS.md §11).
- **Correctness means agreeing with upstream PyTorch**, measured element-wise against upstream
  itself, "it runs" is not "it is a drop-in".
- **What is unsupported refuses by name.** A silent fallback or a plausible invented answer is
  a defect, not a partial feature (AGENTS.md §18).

## 4. What torchnative itself provides

torchnative does **not** replace torch; the torch layer is the *precondition*. On top of it,
torchnative's own surface is (DESIGN.md §2–§3):

| Area | Intent |
|---|---|
| `delta/` | **The core abstraction**: a weight delta over base weights whose *lifetime is part of its type*. TTA's adapted parameters and FL's local update are the same object with a different lifetime and destination. |
| `torch.nn.modules.test_time` and `torch.optim.test_time` | **Test-time learning, a time axis, not a paradigm package.** TTA/TTT is *when* learning happens, so it is not a sibling of `torch.federated` / `torch.swarm`, and optimizer-free adaptation (TTT-Linear, Titans) is out of scope. The **module layer** is `torch.nn.modules.test_time.TestTimeModuleMixin`, an opt-in mixin organised like `torch.nn.modules.lazy.LazyModuleMixin` (not exported at `torch.nn` top level): `online(mode=True)`, `offline()` and `online_parameters()` with test-time-adapters' semantics. The **rule layer** is `torch.optim.test_time`, whose base `TestTimeAdaptation` holds an optimizer the way `LRScheduler` does (it is not an optimizer) and states what to minimise, which parameters move, when to update and how to reset; its generic rules are named by objective type (`EntropyMinimization`, `NormStatistics`, `PseudoLabel`), and paper-named methods (Tent, EATA, SAR, CoTTA) stay in test-time-adapters. Gradient rules and the backward-free rule stay separable, so a backward-free configuration refuses the gradient rules at import. The implementation lives in `torchnative/nn/modules/test_time.py` and `torchnative/optim/test_time.py`; `torch/nn/modules/test_time.py` and `torch/optim/test_time.py` are add-hooks, as `torch/federated/` is. These names and this split were set by the leader by convention under the owner's delegation (owner, 2026-11-13: "컨벤션에 맞게 편성해"), replacing `torchnative.adapt`, which the owner did not agree to (below) and which is removed. Federated learning does not depend on this layer. |
| `torch.federated` | **Federated learning as a paradigm package beside `torch.distributed`** (owner, 2026-11-08: "연합학습과 군집학습 부분은 확정하고 문서 개정하고, 리드미 수정하고, 지금 리펙토링은 바로 들어가도록 해."). Same convention current PyTorch uses for new strategies such as FSDP: they live in their paradigm package, not in `torch.nn`. It holds the FL communication primitives built on `torch.distributed` (federated averaging *is* collective communication), `torch.federated.algorithms` (`FedAvg`, `FedAvgM`, `FedProx`, ...; FL algorithms keep the `Fed` prefix) and the module wrappers. Wrappers live in the paradigm package and their names start with the paradigm, the way "Distributed" starts `DistributedDataParallel`: `torch.federated.FederatedDataParallel` is **horizontal** FL (samples split, each participant holds the whole model; implemented). `torch.federated.FederatedFeatureParallel` is **vertical** FL (features split; **planned, not implemented**, no stub class). The implementation lives in `torchnative/federated/`; `torch/federated/` is the add-hook. `torch.nn.federated` (the owner's 2025 add-hook) is superseded and removed. FL does not depend on test-time adaptation (owner, 2026-11-08: "FL이랑 TTA가 왜 묶여. 풀어."). |
| `torch.swarm` | **Planned, not implemented** (same decision, 2026-11-08): the same structure for peer-to-peer learning without a server, with the wrapper `SwarmDataParallel`. No stub package exists. |
| `kernels/` | A bundle resolver satisfying Hugging Face's `kernels` contract, with resolution **inverted from runtime download to build time**, iOS will not execute downloaded native code (DESIGN.md §8). |
| `api/` | Deployment, lifetime policy and device orchestration (`TorchNativeAPI`). |

**Online/offline learning: why `adapt/` was replaced** (owner, 2026-11-08: "torchnative 현재
구현에는 내가 동의하지 않는다는거야"). What the owner has stated:
- `test-time-adapters` (the owner's TTA library) will use torchnative the way `transformers` uses
  `torch`: the online and offline **learning mechanism** moves into torchnative, and the **TTA
  methods** stay in `test-time-adapters`. TTA methods (`Tent`, normalisation statistics) therefore
  do not belong in torchnative.
- `offline` means **offline learning** and `online` means **online learning**, beside `train` and
  `eval` (as in `test-time-adapters`). The current `adapt.Adapted.offline()`, "exactly the wrapped
  model, nothing updated", contradicts that.
- The API shape was left open (an `Engine` as in `test-time-adapters`, or an `nn`-style mode). On
  2026-11-13 the owner delegated the naming ("컨벤션에 맞게 편성해"), and the leader set the two
  layers in the table above by convention: an `nn`-style mode on the module (`online()` /
  `offline()` beside `train()` / `eval()`) and an `LRScheduler`-shaped rule over an optimizer.
  `torchnative.adapt` (`wrap` / `Adapted`, where offline meant "nothing updated") is removed.

**TTA layering** (owner, 2026-11-08). The owner agreed that TTA is a **time axis** (when learning
happens), orthogonal to the topology axis (central, distributed, federated, swarm): "TTA 구현을 시점 축으로
보는 것에 동의해". So it is not a sibling package of `torch.federated` and `torch.swarm`. The layering follows
the conventions of torch itself; names are **not decided**.
- **Module layer: an opt-in mixin organised like `torch.nn.modules.lazy.LazyModuleMixin`.** It is
  implemented in `torch/nn/modules/<topic>.py` and exposed as `torch.nn.modules.<topic>.<Mixin>`, not at the
  `torch.nn` top level. A model opts in by inheritance; concrete layers, if any, go in `torch.nn` the way
  `LazyLinear` does. Owner: "OnlineModuleMixin으로 이름 확정하진 말고 컨벤션에 맞게 편성해" (do not fix
  the name `OnlineModuleMixin`; organise it to the convention).
- **Rule layer: a base class in `torch.optim.<name>`, like `torch.optim.lr_scheduler`** (the owner agreed).
  It holds an optimizer the way `LRScheduler` and `swa_utils` do; that is the justification, not "TTA is an
  optimizer". Following the convention it ships **general-purpose implementations defined by objective type**
  (normalisation statistics re-estimation, entropy minimisation, pseudo-label or consistency updates), while
  the paper-named methods (Tent, EATA, SAR, CoTTA) live in `test-time-adapters` and are built from them.
- **Scope: TTL = TTA + TTT.** Optimizer-free adaptation (TTT-Linear, Titans) is out of scope here; the owner
  implements it in theLiverLethe as `SelfAdaptation`, a type of self-attention.
- **Follow-up:** once this is implemented, `test-time-adapters`' `develop` branch gets a pull request to
  build on it (owner: "나중에 구현 다 되면 test-time-adapters도 develop 브랜치 수정해서 PR도 넣으면
  좋을거같아").

## 5. Accelerators: fixed front end, swapped back end

**The front end is fixed; only the back end is swapped** (AGENTS.md §20):

```
front end (fixed)   real transformers · from_pretrained · generate
back end (swapped)  Apple → CoreML | Android → undecided (ExecuTorch on device | QNN direct) | Windows → Intel NPU (OpenVINO)
```

Android NPU back end: undecided between ExecuTorch run on the device at `.to(device.npu)` time and Qualcomm QNN used directly at `.to(device.npu)`; both JIT, no per-model export. No offline or CI `.pte` pipeline (owner, 2026-11-07).

**Qualcomm QNN libraries are not shipped; they are found at run time** (owner, 2026-11-13, relayed by
the leader: "(a) 로 가자", choosing route (a) of the redistribution question). torchnative's wheel does not
contain `libQnnHtp*.so` or any other Qualcomm library. At run time torchnative looks for them on the device
or inside the app (an Android app adds `com.qualcomm.qti:qnn-runtime` from Maven Central in Gradle), and
refuses by name, naming the missing files and where it looked, when they are absent.

**For compiled devices, `model.to(device)` triggers `torch.compile`** (owner, 2026-11-08: "model.to를
하기 위해서 특정 디바이스들은 torch.compile이 트리거 되는 걸로 정의하자는거야"). Moving a module to an
NPU (and any other device whose back end compiles a graph) does not lower it eagerly; it wraps the
module in `torch.compile` with that device's back end. The graph is captured and compiled on the
first `forward` for a given input shape, and compiled again when the shape changes ("입력 크기가
바뀌면 재컴파일을 해야 한다는 측면에서도 논리적으로 그게 맞는거 같아"). **Sizes are symbolic where they vary**
(owner, 2026-11-13: "시퀀스 길이를 기호로 다루는거 좋은거 같아. 진행해. 동적 형태 구현해."): a dimension that
changes between calls, the sequence length first, becomes a symbol and compiles once for the range the guards
accept instead of once per size; a shape outside what the guards accept still recompiles. **On compiled devices
the KV cache is transformers' `StaticCache` by default** (owner, 2026-11-13, "그렇게 하자"): the maximum length is
preallocated and written at `cache_position`, set up by torchnative at `.to(device)` so user code is unchanged;
prefill compiles once with a symbolic length and decode stays static-shaped. A paged KV cache comes later, for
Gemstone's continuous batching (Gemstone #74). The front end cannot use
Dynamo's PEP 523 frame hook, which abi3 (the owner's decision) rules out; it enters Dynamo's bytecode
translator (`ConvertFrame`) without the PEP 523 hook, checking guards in pure Python
(`torch._C._dynamo.guards`) and wrapping resume functions so Python control flow (guards and graph
breaks) keeps working (owner, 2026-11-08: approved `torch.compile` without PEP 523; "파이썬 분기 처리 불가 >> 내가 봤을 때 이건 좋은 일이 아니야. 이럼 onnx같이 된다는거잖아"). A capture that only records
one traced path is not acceptable. `torch.export` is supported as well ("torch.export도 가능하게는 해주고"),
as a second way to get a graph, never as the path `.to()` depends on.

The user keeps a real `nn.Module`; a `.pte` or vendor blob is an implementation detail behind a
module, never an object the user handles. Choosing a back end is about which reaches the vendor NPU
best, never about API shape. Quantisation follows the same module-replacement shape
(`docs/graph/QUANT2.md` §3).

### Distributed inference and training across devices (planned)

**One model, several devices, through `torch.distributed`** (owner, 2026-11-08: "분산 추론을
torch.distributed와 호환되게 구현하고, 한 모델을 여러 디바이스에서 추론 혹은 학습 가능하게 구성해줘. 거기에
우리가 만든 npu, vulkan gpu 같은게 사용될 수 있도록 설계해줘."). The direction first appeared as a question
(owner, 2026-08-24: "torch.distributed를 약간 cpu, gpu, npu 연합 사용 기능으로 구현하고자 목표하면 어때?"),
which was not a decision; the 2026-11-08 statement is the approval of the goal.

What that approves is the goal: a `torch.distributed`-compatible way to run or train one model over
cpu, vulkan, mps and npu ranks, with `torch.federated` and `torch.swarm` using the same transport. The
design is `docs/distributed/BACKEND.md`. Decided afterwards the same day (2026-11-08; each decision and its
quote is recorded there, D1 to D14):

- **One backend named `torchnative`** (`init_process_group(backend="torchnative")`), following torch's
  convention that a backend is named after its communication library (gloo, nccl, mpi). The existing
  `local` backend is absorbed as its same-host transport and its name is removed (owner: "이게 더 좋긴
  한데"); callers change to `"torchnative"` with no alias.
- **Transports** `inproc`, `shm`, `tcp` and `ws`, chosen per rank pair (owner: "전송은 일단 니가 하자는대로
  구현하고"); a **swappable server** (owner: "서버 구현체를 갈아낄 수 있도록 설계해줘").
- **A rank is an execution unit bound to one device**, not necessarily an OS process (threads on
  Android and iOS, Web Workers on wasm, processes on desktop).
- **NPU ranks: inference first, training is in the plan** (owner: "학습도 계획안에 둘 것").
- A launcher inside torchnative: `torchnative.distributed.run` and `torchnative.distributed.launcher`.

The remaining questions were delegated ("나머지는 너의 결정에 따를게") and are marked as the leader's
decisions in BACKEND.md. The planned SPEC rows are S6.10 to S6.13. Anything BACKEND.md does not list as
decided (frame layout, topology, interface shapes) remains an agent proposal, **not confirmed**.

## 6. Where it runs

The deployment target is an **embedded CPython 3.13+** inside Kotlin Multiplatform apps
(Android, iOS, desktop), via PythonMultiplatform (DESIGN.md §2). torchnative ships as **one
`cp313-abi3` wheel per platform** so one binary serves 3.13 and every later CPython.

> Inferred, confirm with the maintainer: Linux, Windows and WASM (Pyodide) wheels are in scope as
> *distribution* targets because the wheels exist and are published, but the intent statement
> (DESIGN.md) names the mobile/desktop embedding as the reason the project exists.

## 7. What torchnative is deliberately not

- **Not a reimplementation of PyTorch** and not a port of model architectures (contrast llama.cpp,
  MLC). Architectures should come for free once the operators exist.
- **Not an ahead-of-time export pipeline** as the user-facing model (contrast ExecuTorch/CoreML
  conversion). Graph capture exists only as the substrate NPU back ends consume.
- **abi3 is the owner's decision** ("abi3 문제는 내가 동의한거 맞아", owner, 2026-11-07), so
  Dynamo's PEP 523 frame hook is out of reach. Refusing `torch.compile` permanently was an agent
  recommendation (`docs/graph/COMPILE.md`), **not confirmed by the owner**. The owner's direction:
  "torch.compile을 근데 수정하면 되는거 아냐? 난 애초에 우리가 device.npu 한 뒤 발생하는 현상이
  컴파일인거같은데" (2026-11-07): a `torch.compile` whose front end is torchnative's own capture
  instead of PEP 523 is open, not refused.
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
