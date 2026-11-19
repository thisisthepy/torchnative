# 0.1.0b9: release notes

**GGUF files run from their own blocks on cpu and mps, the two paths llama.cpp
is fastest at (CPU quantised prefill and mps attention) got their first real
kernels, and `torchnative.compile` now runs without PEP 523.** Prefill at 512
tokens on an M1, measured against llama.cpp on the same files
(`tests/bench/bench_llama_cpp.py`): SmolLM2-135M Q8_0 on mps went from 629 to
2038 tok/s, and on the CPU from 211 to 500 tok/s. llama.cpp is still 3x ahead on
mps and 3.4x ahead on the CPU for that cell.

The rest of the range is large: an ExecuTorch runtime built into `torch._C` for
the on-device NPU route, `torch.federated` and a test-time learning layer, a
round of shim fidelity work (views, layouts, overloads, functionalization),
and a rebuilt CI gate that runs by hand only.

Entries are taken from the 89 pull requests merged into develop since tag
`v0.1.0b8` (`git log --merges v0.1.0b8..HEAD`, #164 to #392). Bench numbers were
taken on a shared machine; each table row below says the load. Where a commit
says a step was built but not run, or run only by hand, that is repeated here.

---

## 1. Features added

### GGUF, quantised kernels and attention

- **GGUF blocks kept in the model (S2.5, #172, #176, #197).**
  `from_pretrained(..., gguf_file=..., keep_blocks=True)` and
  `quantization_config=TorchnativeConfig(...)` put the file's own per-layer
  blocks into `QuantizedLinear` leaves without decoding them.
  `LlamaTensorProcessor` reverses llama.cpp's q/k row permutation, on dense
  tensors and directly on block rows, so llama-architecture files (SmolLM2)
  load. `keep_blocks=True` alone keeps every block the file stores, an untied
  `lm_head` included, because that is the model llama.cpp runs from the file.
  #197 does the same for a tied `lm_head` (#186).
- **Fused attention on mps (S7.3, #181).** Calls with no explicit mask, causal
  only when query and key lengths agree, and an instantiated head dim run as
  one Metal kernel (candle-metal-kernels' `steel_attention` and
  `sdpa_vector`): one compute encoder per call, where the unfused path opened
  23 to 27. Other calls keep the unfused path. The fused path does not compute
  logsumexp; it returns a meta tensor that refuses to be read.
- **CPU quantised matmul (#178).** One parallel region per matmul instead of
  one per activation row, and a NEON four-column Q4_0 dot. Outputs are
  bit-identical to 0.1.0b8.
- **Preallocated KV cache (S7.4, #182).**
  `torchnative.kernels.kv_cache.PreallocatedCache` writes keys and values in
  place and hands attention a view, so decode issues no `torch.cat` for the
  cache. It is opt-in (`generate(past_key_values=...)`). Its output is
  bit-identical to `DynamicCache` on cpu and mps.
- **Kernels and orchestration (S7.1, S7.2, S7.3 paged attention; #164, #167,
  #168, #169; #18).** The kernel bundle resolver and the mobile kernel bundle
  selection, staging and `get_kernel` integration, `TorchNativeAPI` deployment
  and lifetime policy, and paged attention on cpu and mps.
- **Q4_K writer (#235, part of #27).** A port of llama.cpp's
  `quantize_row_q4_K_ref`, so the writer can be compared against the reference
  the b8 harness found it differed from.

### torch.compile without PEP 523

- **A hook-free `torchnative.compile` engine (#206, #208; #204, #207, #209,
  #210, #211).** The shim gains `torch._C._dynamo` reflection, slot constants
  and the `eval_frame` surface (#206), a pure-Python guards manager, the small
  `torch._C` fills Dynamo tracing needs, the `DelayedError` autograd node and
  compile-scoped FakeTensor leaf relaxation, and an engine that translates
  frames through `ConvertFrame` without `set_eval_frame` hooks. The OpenVINO
  and CoreML lowerings attach it. Spec rows S1.8 and S4.2 describe it (#203).
- **Dynamic shapes (#264; #262).** Fake tensors carry symbolic sizes beside a
  hint-sized meta tensor; a SymInt keeps its symbol through overload
  resolution. A tiny transformers Llama compiles 2 graphs for lengths 7 to 64
  (1 with `mark_dynamic`) and agrees with eager. An op with no symbolic rule,
  or a wrong rule, refuses by name and that frame compiles static. Size
  branches in the rules decide through the ShapeEnv, so each becomes a guard.
- **StaticCache by default on compiled modules (#265; #263).**
  `attach_to_module` (what `.to(<compiled device>)` calls) wraps `generate`: a
  call that picks no cache gets `cache_implementation="static"`. A tiny Llama
  compiles one prefill and one decode graph and greedy tokens equal eager.
- **Symbolic strides and unspecialized floats (#286, #288; #285, #287).**
  Symbolic strides for a view of a non-contiguous tensor, and upstream's
  `specialize_float=False`: a float argument is a graph input.
- **Refusals by name (#256, #275, #308; #255, #274, #303).** A failed capture,
  `backend='npu'` and `'inductor'`, and a root forward for which Dynamo gives no
  code each refuse by name instead of running eagerly in silence.
- **Qwen3 decode through `torchnative.compile` (#305, #304)**, with StaticCache
  and DynamicCache, as a test.

### ExecuTorch on the device (NPU route)

Direction recorded in #313 with the owner's words of 2026-11-08; the earlier
offline `.pte` pipeline is gone (see section 5).

- **Vendored runtime (#314; #312, #319).** A pinned, pruned ExecuTorch v1.4.1
  runtime core and portable kernels, flatbuffers headers v24.3.25, and a
  `--check` suite in the gate. #339 adds the XNNPACK backend runtime, the
  XNNPACK aarch64 subset, cpuinfo, pthreadpool, FXdiv and FP16 at ExecuTorch
  v1.4.1's pins.
- **`executorch_rt` crate and `torch._C._et_*` binding (#318, #339; #315, #337).**
  Feature `executorch`, macOS arm64. Run 43634472905 built and linked the
  runtime (237 kernels registered). The XNNPACK commit is marked "unbuilt" in
  its message.
- **Android build (#351; #350, #353).** `executorch_rt` for
  aarch64-linux-android with a CI cross-build and `.so` checks, and `_C` linked
  with 16 KB pages with or without the runtime. The commit leaves the device
  step to the leader.
- **Flatbuffers parser linked into `torch._C` (#361; #357)**, replacing exir's
  `flatc` subprocess.
- **QNN libraries are found at run time, never shipped (#261; #260, #217).**
  Owner, 2026-11-13: "(a) 로 가자". `device.npu` refuses by name
  (`QnnLibrariesMissing`) without them. The NPU probe reads `ro.soc.model`
  first and runs locally on a device; the commit records five runs on a Galaxy
  Z Flip 4 (SM8475).
- **Intel NPU (#166).** Bounded rows, a static one-row fallback, quiet probes,
  and an Embedding or RMSNorm the device refuses goes back to the CPU.

### Federated and test-time learning

- **`torch.federated` (#202; #201).** Owner decision 2026-11-08.
  `FederatedDataParallel` is a DDP-like `nn.Module` wrapper, with
  `torchnative.federated.algorithms` (FedAvg, FedAvgM, FedProx). It no longer
  depends on `torchnative.adapt`; `synchronize()` returns `None` like DDP.
- **Test-time learning layer (#248; #237).** `TestTimeModuleMixin`
  (`online()`/`offline()`) and `torch.optim.test_time.TestTimeAdaptation` with
  `EntropyMinimization`, `PseudoLabel` and `NormStatistics`. `torchnative.adapt`
  is removed. The names were set by the leader under the owner's delegation,
  2026-11-13.
- **Distributed (#218, #338; #229, #309).** The `torchnative` backend absorbs
  `local` (BACKEND.md step 1); object lists go over point-to-point; a bundle
  adds distributed, FL and swarm work (#309). #205 is a design draft for a
  heterogeneous backend, awaiting owner answers.

### Shim fidelity

- **Views and layouts.** View metadata as upstream's (`_base`, `_is_view`,
  creation meta, fake views) (#279; #272); a fresh result takes upstream's
  layout (elementwise stride order, `preserve_format`, flash outputs) (#387);
  `Tensor.as_strided_` in place (#384); full_like and ones_like on a meta input
  (#388, #306).
- **Dispatch.** Python `torch.library` kernels dispatch through `_aten_dispatch`
  (#239; #227); the torch-dispatch mode stack follows upstream's rules for
  infra modes (#326); `torch.Tensor._make_wrapper_subclass`, dispatching to
  `__torch_dispatch__` in upstream's order (#341); op packets pick upstream's
  first overload in registration order (#392; #386); JIT schema types compare
  by type (#328).
- **Functionalization and CompositeImplicit (#346, #368, #373; #329, #333,
  #372).** The `Repr::Functional` wrapper with its state queries, a
  schema-driven Functionalize kernel for non-mutating ops and views, and the
  CompositeImplicit decomposition frame with `aten.linear`. 42 CompositeImplicit
  ops are pinned as pending; the list may only shrink (#374).
- **Truthful fills (#324, #345).** `compiled_autograd` and anomaly-mode fills;
  anomaly mode only inside a FakeTensorMode, and backward refuses while it is on.
- **Python scalar types as `dtype`** in `tensor` and factory functions (#391; #268).
- **Autograd and aten.** `nll_loss` backward
  for `reduction='none'` and class weights (#250; #249); a backward leaves the
  graphs its root does not reach (#389; #358). CPU `index_copy_` on views is
  O(src) in place (#338; #254).

### Tooling and CI

- **Competitor harness (#284; #283).** ONNX Runtime and LiteRT, bench-only.
- **Windows MSVC runner in the gate (#280, part of #81).**
- **Gate structure (#336, #344, #349, #363, #367, #370, #355).** DOCWATCH runs
  each live probe once; parallel-safe suites run side by side with timings and a
  suite timeout; Metal and distributed suites run in lanes; macOS is three shard
  jobs of one run merged into one ledger; the six slow Metal suites are batched
  (comparison with the old mode is off by default); the runner's CPU is printed
  and golden comparisons run under `faulthandler`; macOS distributed suites are
  faster.
- **verify_published (#224; #214)** works with an iOS host CPython that has no
  `_ssl` or `_hashlib`; publish jobs use macos-15 with
  `MACOSX_DEPLOYMENT_TARGET=11.0` (#245; #244).
- **Release readiness (#188; #187).** `test-published-wheel` runs after
  `publish-pypi` on the released tag's version; Quick start adds transformers;
  CoreML without `libcoremlpython` refuses as `NpuUnresolved`;
  `torchnative.__version__`; README pins 0.1.0b9.

## 2. Defects fixed

- `module.to("mps")` left `QuantizedLinear` weights on the CPU, and the first
  forward panicked inside candle. The blocks now move byte for byte
  (`_C._quantized_to`), and a weight/activation device mismatch refuses by
  name (#176).
- Qwen3-0.6B-Q4_0 did not load with `keep_blocks=True`: the block carrier
  lacked `is_floating_point()`, and an untied block `lm_head` reached
  `nn.Parameter` (#176).
- `kernels` 0.17 and 0.18 validator APIs both work in `get_kernel` (#171); the
  gate pins `kernels==0.17.2` and tolerates a missing vendored shim (#170).
- Intel NPU: probe compiles are quiet (#166).
- `generate` through `attach_to_module` compiled nothing: every forward hit the
  catch-all and ran eagerly. Fixed with StaticCache by default (#265), along
  with `torch.tensor(dtype=int|float|bool)` being refused.
- A graph traced at length 20 for `x[:8]` was reused at length 5 and failed;
  size branches now become guards (#264).
- CUDA: `BackendDevice::synchronize` is called explicitly (#188).
- CPU `index_copy_` on views was not O(src) in place (#338; #254).
- Launcher: groups created during the stop grace are kept aborting (#365; #364).
- Windows and macOS gate: the macOS crate unit tests run with the interpreter's
  libpython inserted, after a run aborted at image load (#338; #302); the macOS
  Rust toolchain is pinned to 1.99.1 (#267; #266); a macOS gate TypeDict guard
  fix for `test_npufuse` (#271; #270); failures in `test_dynamo_guards_diff` and
  `test_shim` (#213; #212); `test_stream` q8_0 mps (#185; #184); `test_gloopin`
  names its precondition and SKIPs (#226; #132).
- `test_et_flatc` accepts both of the parser's refusal wordings (#361).
- The `#150` doublebackward pin has its root cause recorded: the x86 GEMM route
  (#342). It is still a pin.
- Two known divergences from upstream are pinned by name (#360; #358, #359).

## 3. Test added

- `test_compile_nopep523.py` (7 tests, #208), differential guard tests (#203),
  the Qwen3 decode compile test (#305), and symbolic-shape tests that were red
  before #264 and turn red again when their branch is reverted.
- ExecuTorch: the vendor `--check` suite (#314), `test_et_runtime` program tests
  with a venv step in the gate (#318), `test_et_flatc` (#361).
- Successors for the two adapt-only `test_shim` tests (#248), the CIA pending
  list pin (#374), and on-device `test_qnnprobe_ondevice.py` (#261).
- `tests/release/` suites for the gate, CI dispatch and dist-quick structure
  (#317, #320, #322, #356).

## 4. Measured but not implemented

- **Speed parity.** llama.cpp leads on every cell measured: 1.8 to 2.8x on mps
  decode, 3.0 to 3.5x on mps prefill, 3.3 to 4.6x on CPU prefill (cells measured
  at load 3 to 7; the Qwen3 Q4_0 CPU cells ran under load and are left out).
- **A decode gain from the preallocated cache.** At the bench's context
  (5-token prompt, 128 steps) the change was within run-to-run noise.
- **Q4_0 greedy text on the CPU.** SmolLM2-135M Q4_0 differs from llama.cpp's
  default CPU path. llama.cpp's own Metal path, and its CPU path with weight
  repacking or flash attention turned off, give torchnative's text. Qwen3-0.6B
  Q4_0 at 128 tokens differs at generated token 20, where the logit margin is
  0.084. Every Q8_0 cell and every mps cell matches llama.cpp.
- **ExecuTorch on a device.** The ExecuTorch runtime was built and linked on macOS
  arm64 (run 43634472905); the Android build is cross-built in CI and its device
  step was left to the leader. No claim is made here that the on-device
  `.to(device.npu)` route ran an ExecuTorch program on Android hardware.
- **The NNAPI device tests** did not run for this release; they wait on a
  connected device.
- **XNNPACK on the shim.** `test_xnnpack_lowering_on_the_shim_delegates_what_upstream_delegates`
  still stops in ExecuTorch's ConvertToSDPAPass (#306, open).

Carried over, not re-measured here:

- **`torch.compile` through the stock path.** The b8 notes called it a
  permanent refusal for the structural reason in
  [`../graph/COMPILE.md`](../graph/COMPILE.md). #194 marked that refusal
  unconfirmed and #208 added a PEP 523-free engine (section 1); that record
  was not rewritten or re-measured for this release.
- **82 of 297 architectures** were never numerically judged. That count is
  [`../architectures/ARCH100.md`](../architectures/ARCH100.md)'s, it measured
  reachability, and it was not re-measured for this release.
- Intel NPU never ran on hardware (#21).
- The Windows tie order (#81) still has no oracle result recorded in these notes.

## 5. Deleted

- **The offline per-model `.pte` QNN lowering pipeline.** The owner never
  approved it. #191 (#190) removes `test-qnn-lower.yml`, `qnn_lower.py`,
  `QNNCI.md` and `test_qnnci`; #278 (#277) removes `qnn.lower`,
  `qnn.lower_cpu_reference`, the `.pte` reader and loader (`PteArtefact`,
  `read_artefact`, `QnnModule`, `ExecuTorchModule`) and QNN.md's lowering
  sections.
- **`torchnative.adapt`** with `test_stage0.py` and `test_stagetype.py` (#248),
  replaced by the test-time layer and `torch.federated`.
- The public `Round` class of the federated module (#202), and the DOCWATCH that
  asserted `DistributedDataFederated` absent (#202).
- The macOS 14 mps CI job (#243).
- **`torch.compile` as a permanent refusal.** The b8 claim is withdrawn: #194
  marks it unconfirmed, #196 and #198 record that `model.to(<compiled device>)`
  triggers `torch.compile`, with the front end undecided.

## 6. Documentation corrected

- `docs/SPEC.md`: S2.5 and S7.3 state kept-block and fused-attention behaviour,
  S7.4 is new (#172, #181, #182); S1.8 and S4.2 describe compile without PEP 523
  (#203); S1.12 says a wrapper's `untyped_storage` answers as upstream.
- AGENTS and INTENT: what counts as the owner's decision (#192), an incident
  record for ExecuTorch and `.pte` (#193), clean up on merge and agents do not
  build (#195), INTENT 4 and README: `adapt/` was not agreed (#199).
- The Android NPU back end is ExecuTorch on the device, with the owner's words
  of 2026-11-08 (#313).
- `tests/bench/bench_llama_cpp.py` no longer expects SmolLM2's GGUF file to
  refuse; it gains the `gguf_blocks` mode, `--tn-modes` and `--kv-cache`.
- CI: development tests run only by hand, one run at a time, in one `dev-ci`
  group (#317, #320, #322, #356); docs-only pushes skip the gate (#243).
- FUNCTIONALIZATION.md cites SPEC S5.11 (#346).

## 7. Gate at this head

Not fully green, and this is stated plainly. The develop gate's known failure is
`tests/export/test_exir_shim.py::test_xnnpack_lowering_on_the_shim_delegates_what_upstream_delegates`,
which still stops in ExecuTorch's ConvertToSDPAPass (issue #306, open). The
latest develop-based run with everything else green on linux, macOS and Windows
is run 44454306548. In that run macOS shard 2/3 was cancelled without a runner;
that shard passed in run 44449037226. Nothing more is claimed about the gate.
