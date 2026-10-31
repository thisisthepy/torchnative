# Specification

What torchnative does, the behavioural contract. It must stay inside [`INTENT.md`](INTENT.md).
A behaviour change starts here, then a failing test, then code (AGENTS.md rule 5).

**How to read the status.** Each item is `implemented`, `partial` or `planned`.

- `implemented` means a test in this repository asserts the behaviour; the test is cited as
  `file::function`. Suites under `tests/` run in the gate
  (`bash tests/run.sh`, AGENTS.md §13), the package-level import tests
  (`tests/api/test_import.py`) among them. This document was written by reading those tests, **not** by running
  the gate; the last recorded gate result is the baseline in AGENTS.md §13.
- `partial` means some of the stated behaviour is asserted and the rest is named as missing.
- `planned` means no test asserts it yet. A roadmap claim never upgrades an item.
- Claim grades follow AGENTS.md §16: *builds* / *reaches* / *agrees*.

Test paths below are relative to `tests/` unless they start with another directory.

---

## S1. The `torch._C` replacement

| # | Behaviour | Status | Evidence |
|---|---|---|---|
| S1.1 | `import torch` resolves to upstream's vendored Python tree with **our** `torch._C`; probes can tell the shim from upstream via `torch._C._aten_implemented`. | implemented | `_support/test_shim.py::test_every_advertised_op_is_actually_dispatchable`; `api/test_tnnamespace.py::test_importing_torchnative_does_not_import_torch` |
| S1.2 | **Single door:** every operator reaches its kernel through `_aten_dispatch`; dispatch modes see the same operators as upstream, in the same order. | implemented | `bindings/test_dispatch.py::test_a_mode_sees_the_same_operators_as_upstream_in_the_same_order`, `::test_the_replaced_call_never_reaches_a_kernel`; `_support/test_shim.py::test_the_dispatch_table_matches_the_two_lists` |
| S1.3 | **Agreement with upstream:** each implemented ATen op is compared against upstream torch on value, shape and dtype (golden harness); the harness injects faults into its own comparators (`--self-test`). | implemented | `tests/golden/compare.py`, `tests/golden/cases.py` (run by the gate); DOCWATCH markers `golden_cases_failed eq 0`, `golden_pending eq 0` in `README.md` |
| S1.4 | **Refusal by name:** an unsupported op, argument form, dtype or device refuses with its own name and reason; it never falls back silently. | implemented | `_support/test_shim.py::test_ops_without_a_meta_kernel_name_themselves`; `devices/vulkan/test_vulkan4.py::test_an_op_that_is_not_taught_refuses_and_names_itself`; `bindings/test_argform.py::test_a_per_axis_differing_stride_is_still_refused_by_name` |
| S1.5 | The `meta` device carries shape and dtype without storage, with the dense kernels' promotion rules, enough for `from_pretrained` init paths that compute on meta. | implemented | `_support/test_shim.py::test_meta_tensors_carry_shape_and_dtype_and_no_data`, `::test_the_llama3_rope_init_runs_on_meta_end_to_end` |
| S1.6 | Seeded RNG (`uniform_`, `bernoulli_`, …) reproduces upstream's stream. | implemented | `_support/test_shim.py::test_uniform_matches_torchs_stream_bit_for_bit`, `::test_bernoulli_draws_in_double_for_every_dtype` |
| S1.7 | Built against CPython's limited API: one `cp313-abi3` binary loads on 3.13 and later CPythons. | implemented | `release/test_release.py::test_the_abi3_wheel_loads_on_later_cpythons` (skips by name when no newer `python3.N` is on `PATH`) |
| S1.8 | `torch.compile` refuses by name without breaking `transformers`; it is not a goal (INTENT §7). | implemented | `_support/test_shim.py::test_torch_compile_refuses_by_name_without_breaking_transformers` |

## S2. Running real models

| # | Behaviour | Status | Evidence |
|---|---|---|---|
| S2.1 | Real `transformers` models load through `from_pretrained` with bit-identical weights and forward to upstream's logits. | implemented | `_support/test_shim.py::test_from_pretrained_loads_the_real_weights_bit_for_bit_on_all_four_paths`, `::test_from_pretrained_forward_matches_upstream_logits`, `::test_a_real_transformers_llama_forward_matches_upstream` |
| S2.2 | `generate` (greedy) produces upstream's tokens. | implemented | `_support/test_shim.py::test_greedy_generate_matches_upstream_token_for_token`, `::test_the_llama3_rope_checkpoint_loads_and_generates_like_upstream` |
| S2.3 | `torch.load`, safetensors and `torch.save` round-trip with upstream (upstream reads what we write). | implemented | `_support/test_shim.py::test_ckpt_safetensors_two_readers_agree_with_torch_load_bit_for_bit`, `::test_save_upstream_reads_every_dtype_and_view_the_shim_wrote_bit_for_bit` |
| S2.4 | Architecture sweep: models that forward (*reaches*) and models that numerically agree with upstream at a derived tolerance (*agrees*). | partial | `numerics/test_agree.py::test_verdict_refuses_first_when_upstream_cannot_reproduce_itself` and `numerics/test_agree2.py::test_the_two_refusal_outcomes_cannot_be_deleted_into_passes` test the judging rules; the sweep itself (`arch_sweep.py`, `agree_sweep.py`) is a measurement, recorded in `docs/numerics/AGREE.md`, not re-run by the gate |
| S2.5 | **GGUF checkpoints** load through `torchnative.transformers.<Auto*>.from_pretrained(repo, gguf_file=...)`: upstream's GGUF config mapping, tensor-name tables and loader, with GGML blocks read by `torchnative.gguf` and dequantised by candle **bit-identically to llama.cpp's `gguf` package**; unsupported GGML types and tensor-rewriting architectures refuse by name. The loaded model is dense (blocks are not kept). docs/graph/GGUF.md. | partial | `models/test_gguf.py::test_from_pretrained_gguf_on_a_tiny_qwen3_written_by_gguf_py`, `::test_real_file_q8_0_matches_the_reference_reader`, `::test_every_unsupported_type_refuses_by_name` -- written and run by hand against the main checkout's shim (docs/graph/GGUF.md §6), **not yet through the gate**; the real-file `from_pretrained` test has not run |

## S3. Devices

| # | Behaviour | Status | Evidence |
|---|---|---|---|
| S3.1 | `torch.device` labels form a closed vocabulary; an invented label is refused. | implemented | `devices/cuda/test_cuda.py::test_cuda_is_a_constructible_label_even_with_no_cuda_in_the_build` |
| S3.2 | `mps` (candle Metal): ops whose kernel would read back to the host are refused by name, enumerable at runtime; softmax stays on device and a BERT encoder forwards and agrees. | implemented | `devices/mps/test_mpsrefuse.py::test_the_derivation_finds_the_ten_without_being_told_their_names`; `devices/mps/test_mpsattn.py::test_a_bert_encoder_forwards_on_mps_and_agrees_with_upstream`; `numerics/test_remeasure2.py::test_the_readme_host_readback_count_is_the_number_the_runtime_reports` |
| S3.3 | `vulkan`: a separate `Repr` arm; taught ops run in SPIR-V shaders, everything else refuses naming itself; dispatch counters prove execution. | implemented | `devices/vulkan/test_vulkan4.py::test_an_op_that_is_not_taught_refuses_and_names_itself`, `::test_the_exactly_rounded_ops_are_bit_identical_to_upstream`; `devices/vulkan/test_vulkancov.py::test_the_gate_summary_says_UNVERIFIED_when_nothing_ran` (a machine without a Vulkan loader reports the suites as unverified, not passed) |
| S3.4 | `cuda`: wired as a candle device behind `--cfg torch_c_cuda`; unavailability names one of `not_built`/`no_driver`/`no_device`/`wrong_arch`/`unclassified`. **Never compiled or run on a GPU.** | partial | `devices/cuda/test_cuda.py::test_asking_for_cuda_on_this_build_refuses_and_names_not_built`, `::test_this_file_never_claims_cuda_computes` |

## S4. The `torchnative` API

| # | Behaviour | Status | Evidence |
|---|---|---|---|
| S4.1 | `torchnative.device` offers `cpu`/`mps`/`vulkan`/`cuda`/`npu`; every availability answer names the probe behind it; `npu` resolves per host and never resolves to the CPU; eager and compiled devices differ by type. | implemented | `devices/test_devicens.py::test_the_namespace_has_the_decided_members`, `::test_npu_never_resolves_to_the_cpu`, `::test_eager_and_compiled_are_distinguished_by_type_not_by_convention` |
| S4.2 | `nn.Module.to(<torchnative device>)` keeps upstream semantics for every ordinary argument and returns the same module. | implemented | `devices/test_devicens.py::test_to_is_unchanged_for_every_ordinary_argument_form`, `::test_ordinary_calls_reach_upstream_with_identical_arguments` |
| S4.3 | `torchnative.transformers` exposes every `Auto*` class (enumerated, not hand-listed) subclassing upstream; unsupported arguments (`export=`, `load_in_4bit=`) refuse by name. | implemented | `api/test_tntransformers.py::test_the_family_is_enumerated_not_hand_listed`, `::test_export_refuses_by_name`, `::test_load_in_4bit_refuses_by_name` |
| S4.4 | `import torchnative` does not import `torch`; every subpackage is reachable. | implemented | `api/test_tnnamespace.py::test_importing_torchnative_does_not_import_torch`; `tests/api/test_import.py` |

## S5. Accelerator back ends (front end fixed, back end swapped)

| # | Behaviour | Status | Evidence |
|---|---|---|---|
| S5.1 | Graph capture through the single door, decomposition and refolding to a core op set (substrate for NPU delegates). | implemented | `export/test_export.py::test_capture_is_the_only_working_front_end_and_records_the_module_it_ran`; `_support/test_shim.py::test_decompose_refuses_by_name_what_it_cannot_lower` |
| S5.2 | **Apple / CoreML:** eligible leaves lower to CoreML; a float16 graph runs on the Neural Engine; every leaf not lowered is named and a partial offload warns. | implemented | `devices/coreml/test_coremlops.py::test_the_neural_engine_runs_the_conv_at_float16_and_cannot_at_float32`, `::test_every_leaf_not_lowered_is_named_and_a_partial_offload_warns` (macOS only; `MLComputePlan` is nondeterministic under load, AGENTS.md §13.3) |
| S5.3 | **Windows / Intel NPU:** `model.to(device.npu)` lowers eligible `nn.Linear` leaves (`_NPULinear`), fused gated MLPs (`_NPUGatedMLP`), `RMSNorm` (`_NPURMSNorm`) and `nn.Embedding` (`_NPUEmbedding`) through OpenVINO (`rmsnorm_ir`, `embedding_ir`, `mlp_ir`, `linear_ir`) and returns the same module; zero lowered leaves is a refusal. Dynamic row dimensions use a bounded range (`1..2048`) rather than unbounded `-1`; intentional dynamic-shape probe compilations install a no-op log callback, request `LOG_LEVEL=LOG_NONE` and mute file descriptors 1 and 2 for the call, while the refusal text is still raised and kept in the report (a normal compile stays audible); an Embedding or RMSNorm the device refuses at its eager compile is restored to the CPU module and named in `skipped`, so it never fails `to()` or `generate()`; when a device uses static shapes, the `batch=1` compiled model is reused across KV-cache decode steps and multi-row prefill without per-prompt-length recompilation. **The 0.1.0b8 Windows run (`SmolLM2-135M`, 151 modules, `fraction_moved` 0.8259) predates the bounded axis, `RMSNorm` and `Embedding`; none of the three has run on an Intel NPU. IR acceptance and numerics were checked only on OpenVINO's CPU plugin.** | partial | `devices/npu/test_npuwire.py::test_to_device_npu_returns_a_real_module_on_a_faked_intel_npu_host`, `::test_zero_leaves_lowered_is_a_refusal_and_not_a_success`; `devices/npu/test_npufuse.py::test_the_dynamic_axis_is_bounded_on_activations_and_never_on_weights`, `::test_a_device_that_refuses_the_dynamic_axis_reuses_the_static_one_row_model`, `::test_rmsnorm_and_embedding_lower_and_leave_only_rotary_on_the_cpu`, `::test_an_embedding_the_device_refuses_goes_back_to_the_cpu_and_to_still_succeeds`, `::test_probe_compile_suppresses_openvino_native_stderr_and_keeps_real_errors`; `test_intelnpu.py` |
| S5.4 | **Android / NNAPI:** the captured graph lowers to an NNAPI blob that executes through `ANeuralNetworksModel`. Only the CPU reference driver has run it, execution, not acceleration. | partial | `test_npu2.py` |
| S5.5 | **Android / QNN (ExecuTorch):** a submodule is delegated and the model still generates; the HTP ahead-of-time half refuses by name here. No claim that anything ran on an NPU. | partial | `devices/qnn/test_qnn.py::test_a_real_checkpoint_still_generates_with_a_submodule_delegated`, `::test_no_claim_is_made_that_anything_ran_on_an_npu` |
| S5.6 | Quantisation by **module replacement** (`torchnative.quant`, Q8_0/Q4_0/Q4K), reachable through a registered `HfQuantizer` so leaves are swapped before the weights land. | implemented | `_support/test_shim.py::test_the_quantizer_plugin_replaces_the_leaves_before_the_weights_land`, `::test_the_quantizer_plugin_and_quantize_produce_the_same_model` |

## S6. Training, adaptation and federation

| # | Behaviour | Status | Evidence |
|---|---|---|---|
| S6.1 | `loss.backward()` through upstream's own autograd path, with optimizer steps agreeing with upstream. Double backward (`create_graph=True`), `torch.autograd.Function` (custom `backward`, `save_for_backward`, `setup_context`, `once_differentiable`, non-differentiable outputs) and hooks (leaf and non-leaf `register_hook`, `Module.register_full_backward_hook`/`_pre_hook`) agree with upstream element-wise on training loops, against derived tolerances; `ctx.mark_dirty`, `register_post_accumulate_grad_hook`, the legacy `register_backward_hook`, and `create_graph` through an op whose first-order rule issues an op with no rule refuse by name (docs/training/BACKWARD10.md §5). `create_graph` through relu, gelu (exact and tanh), layer_norm and embedding has rules (#48, docs/training/BACKWARD10.md §5.1). | implemented | `_support/test_shim.py::test_a_real_training_loop_runs_through_loss_backward_and_agrees_with_upstream`, `::test_the_engine_refuses_several_roots_by_name_and_records_create_graph`; `test_train.py`; `test_backward10.py::test_a_wgan_gradient_penalty_trains_and_agrees_with_upstream`, `::test_custom_autograd_functions_train_and_agree_with_upstream`, `::test_a_custom_function_backward_is_what_runs_not_the_forward_ops`, `::test_tensor_and_module_hooks_train_and_agree_with_upstream`, `::test_the_hooks_this_round_did_not_build_refuse_by_name`; `test_doublebackward.py` (one test per op, plus `::test_a_gradient_penalty_and_hvp_through_embedding_linear_gelu_layernorm_agree`) |
| S6.2 | `torchnative.delta.Delta`: owns base and offset; a revert restores the base bit-for-bit; a delta is written and read back bit-for-bit; unsupported lifetimes name the check that would prove them. | implemented | `_support/test_shim.py::test_a_delta_reverts_the_base_weights_bit_for_bit`, `::test_a_delta_is_written_and_read_back_bit_for_bit`, `::test_a_delta_names_a_check_for_the_destination_it_cannot_reach` |
| S6.3 | `adapt` stage 1 (gradient): `Tent` lowers prediction entropy and agrees with upstream autograd; wrong sign and `lr=0` do not. | implemented | `_support/test_shim.py::test_tent_reduces_prediction_entropy_and_upstream_agrees`, `::test_the_wrong_sign_and_a_zero_step_do_not_reduce_entropy` |
| S6.4 | `adapt` stage 0 (no backward): normalisation statistics re-estimated as upstream computes them, and reverted. | implemented | `training/test_stage0.py::test_the_statistics_it_computes_are_the_statistics_upstream_computes`, `::test_revert_after_a_stage_0_run_restores_the_running_statistics` |
| S6.5 | Stage 0 and stage 1 separated **by type**, so a backward-free build rejects gradient methods at import time (INTENT §4). Stage 0 is `adapt.StatisticsMethod`; stage 1 is `adapt.gradient.GradientMethod` (`Tent`), reached lazily as `adapt.Tent`; `wrap` admits each stage only as its type. "Backward-free" is a **configuration**, `TORCHNATIVE_BACKWARD=off` at `torchnative.adapt` import (no build of this crate lacks the tape): under it `adapt.gradient` refuses to import, and any `Method` subclass declaring stage 1 or 2 is refused at class creation, each by name and reason; stage 0 still adapts and reverts without loading the stage-1 module. | implemented | `training/test_stagetype.py::test_stage_0_and_stage_1_are_distinct_types`, `::test_a_backward_free_build_refuses_the_stage_1_module_at_import_by_name`, `::test_a_stage_1_method_a_user_defines_is_refused_when_its_class_is_created`, `::test_stage_0_still_adapts_and_reverts_in_a_backward_free_build`, `::test_a_stage_declared_by_attribute_and_not_by_type_is_refused_at_wrap`, `::test_an_unrecognised_backward_setting_is_refused_by_name` |
| S6.6 | `torch.distributed` via `ProcessGroupLocal` over loopback TCP: eleven collectives and five reduce ops agree with upstream gloo at world 3 and 4; bitwise ops and point-to-point refuse by name. | implemented | `distributed/test_collect2.py::test_every_collective_matches_upstream_gloo_at_world_three_and_four`, `::test_point_to_point_still_refuses_by_name_and_was_not_weakened` |
| S6.7 | Federated averaging across processes equals the same average computed centrally; a world of one and unsupported round shapes refuse by name. Secure aggregation, differential privacy and participant selection are not implemented. | partial | `_support/test_shim.py::test_fedavg_over_two_processes_equals_the_same_average_computed_centrally`, `::test_federated_refuses_a_world_of_one_by_name_at_every_door`, `::test_multi_round_fedavg_equals_the_same_rounds_computed_centrally` |

## S7. Kernels and orchestration

| # | Behaviour | Status | Evidence |
|---|---|---|---|
| S7.1 | `torchnative.kernels`: a bundle resolver satisfying the HF `kernels` contract, resolving at build time on mobile. `target_profile` maps mobile targets (`android-arm64`, `ios-arm64`, `ios-arm64-sim`) to build-time resolution (`cpu` on Android; `metal` and `cpu` on iOS) and desktop targets (`macos-arm64`, `linux-x86_64`, `windows-x86_64`) to runtime Hub resolution, while refusing unsupported targets and backends (`vulkan`, `wasm`) by name. `select_mobile_variant` selects the preferred compatible variant for a mobile target at build time; `stage_bundle` packages selected local kernel variants and writes `bundle.json`; `BundleResolver` (and `BundleResolver.from_bundle`) implements `kernels.resolver.Resolver`, returning only local `LocalKernel` locations without network access; `get_kernel` loads a bundled kernel module end-to-end through `kernels.load.get_kernel_with_resolver` (and `use_bundle` patches `kernels.get_kernel` within its scope). | implemented | `api/test_kernel_bundle.py::test_bundle_resolver_uses_exact_dependency_and_preselected_path`, `::test_mobile_target_profiles_and_unsupported_backend_refusals`, `::test_build_time_selection_and_bundle_staging_for_mobile_targets`, `::test_bundle_resolver_loads_real_local_kernel_via_get_kernel_without_network`, `::test_use_bundle_routes_hf_kernels_get_kernel_through_the_bundle` |
| S7.2 | `torchnative.api.TorchNativeAPI`: deployment, lifetime policy, and device orchestration. Importing `torchnative.api` does not import `torch`. `deploy(model, ...)` orchestrates optional in-place quantisation (`torchnative.quant.quantize_`), target device placement/lowering (`model.to(resolved_device)`), and optional opening of parameter deltas (`Delta.over`) and buffer snapshots (`BufferSnapshot.over`), returning the exact same `nn.Module` (`deployed is model`) and recording `model.torchnative_deployment`. Lifetime policy methods (`record`, `revert`, `apply`, `persist`, `load_delta`) manage parameter deltas and buffer snapshots across system events bit-for-bit; `save_weights` and `load_weights` round-trip model state via safetensors; unsupported devices, quantisation formats, or arguments refuse by name. | implemented | `api/test_torchnative_api.py::test_import_torchnative_api_does_not_import_torch`, `::test_deploy_returns_same_nn_module_and_places_on_device`, `::test_deploy_with_quantization_replaces_linear_leaves_in_place`, `::test_lifetime_policy_delta_and_buffer_snapshot_record_revert_apply_persist`, `::test_weights_save_and_load_round_trip`, `::test_unsupported_device_quantize_or_arguments_refuse_by_name` |
| S7.3 | Paged attention kernel (`torchnative.kernels.paged_attention` / `gather_paged_kv`) on `cpu` and `mps`, plus flash-attention / flash-linear-attention kernels across platforms. `paged_attention` gathers paged `[num_blocks, block_size, num_kv_heads, head_dim]` KV blocks by `block_table`, supports MHA and GQA (`num_heads % num_kv_heads == 0`), single-query decode and multi-query prefill, `context_lens` masking, and `float32`/`float16`/`bfloat16`; agrees element-wise with upstream PyTorch's reference attention over the same gathered KV in a separate subprocess against a derived tolerance (`docs/numerics/AGREE.md`); on `mps`, `_C._metal_counters()` proves compute encoders ran and `host_downloads == 0` mid-dispatch; unsupported block sizes, head layouts, devices, or dtypes refuse by name. | partial | `devices/mps/test_paged_attention.py::test_paged_attention_agrees_with_upstream_reference_on_cpu_and_mps`, `::test_paged_attention_on_mps_opens_compute_encoders_and_downloads_nothing`, `::test_paged_attention_refuses_unsupported_block_or_head_layouts_by_name` |

## S8. Distribution

| # | Behaviour | Status | Evidence |
|---|---|---|---|
| S8.1 | Nine platform wheel targets, all `cp313-abi3`; Android x86_64 refuses by name; tags match what `packaging` generates. | implemented | `release/test_wheelmatrix.py::test_the_android_tags_are_what_packaging_generates`, `::test_the_readme_does_not_call_a_never_executed_target_measured` |
| S8.2 | The version lives only in `pyproject.toml`, is a pre-release, and the README never claims an unpublished version is on PyPI. | implemented | `release/test_release.py::test_pyproject_is_the_only_place_a_version_is_declared`, `::test_the_readme_does_not_claim_an_unpublished_version_is_on_pypi` |
| S8.3 | Release CI: a `v*` tag builds and publishes the wheels via PyPI Trusted Publishing; the tag must equal the version. | implemented | `test_cipub.py` (static checks over `.github/workflows/publish-pypi.yml` and `.github/scripts/check_tag_version.py`) |
| S8.4 | `main` carries a reduced, CI-generated layout reached only by a pull request from `release` (`.github/scripts/release/sync-release.sh`, run by `release-sync.yml`); that layout keeps `docs/guide/`, the Pages site. | implemented | `.github/scripts/release/test-sync-release.sh` passes; `release/test_publish.py::test_docs_guide_survives_the_main_only_layout`, `::test_the_guide_check_can_fail` |

---

## Outside intent: needs a decision

Behaviour that exists or is planned but that [`INTENT.md`](INTENT.md) does not clearly cover.

1. **WASM / Pyodide and Linux/Windows wheels.** Built and (for Linux/Windows) verified in CI, but
   INTENT only names the embedded mobile/desktop CPython as the reason the project exists.
2. **Vulkan as a hand-written `Repr` arm** (shaders, upload path). INTENT names CoreML, QNN/ExecuTorch
   and Intel NPU as the back ends; a general-purpose GPU back end written from scratch is not
   stated there.
3. **CUDA wiring and the CUDA wheel workflow.** Not an on-device target.
4. **`torchnative.transformers` `Auto*` mirror.** Consistent with "fixed front end", but INTENT §2
   also says not to build a façade; the maintainer should confirm that an import-path mirror
   subclassing upstream is not one.
