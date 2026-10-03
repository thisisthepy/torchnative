# Specification

What torchnative does — the behavioural contract. It must stay inside [`INTENT.md`](INTENT.md).
A behaviour change starts here, then a failing test, then code (AGENTS.md rule 5).

**How to read the status.** Each item is `implemented`, `partial` or `planned`.

- `implemented` means a test in this repository asserts the behaviour; the test is cited as
  `file::function`. Suites under `rust/torch_c/pytests/` run in the gate
  (`bash rust/torch_c/pytests/run.sh`, AGENTS.md §13); `torchnative/src/test/` holds the
  package-level import tests. This document was written by reading those tests, **not** by running
  the gate; the last recorded gate result is the baseline in AGENTS.md §13.
- `partial` means some of the stated behaviour is asserted and the rest is named as missing.
- `planned` means no test asserts it yet. A roadmap claim never upgrades an item.
- Claim grades follow AGENTS.md §16: *builds* / *reaches* / *agrees*.

Test paths below are relative to `rust/torch_c/pytests/` unless they start with another directory.

---

## S1. The `torch._C` replacement

| # | Behaviour | Status | Evidence |
|---|---|---|---|
| S1.1 | `import torch` resolves to upstream's vendored Python tree with **our** `torch._C`; probes can tell the shim from upstream via `torch._C._aten_implemented`. | implemented | `test_shim.py::test_every_advertised_op_is_actually_dispatchable`; `test_tnnamespace.py::test_importing_torchnative_does_not_import_torch` |
| S1.2 | **Single door:** every operator reaches its kernel through `_aten_dispatch`; dispatch modes see the same operators as upstream, in the same order. | implemented | `test_dispatch.py::test_a_mode_sees_the_same_operators_as_upstream_in_the_same_order`, `::test_the_replaced_call_never_reaches_a_kernel`; `test_shim.py::test_the_dispatch_table_matches_the_two_lists` |
| S1.3 | **Agreement with upstream:** each implemented ATen op is compared against upstream torch on value, shape and dtype (golden harness); the harness injects faults into its own comparators (`--self-test`). | implemented | `tools/golden/compare.py`, `tools/golden/cases.py` (run by the gate); DOCWATCH markers `golden_cases_failed eq 0`, `golden_pending eq 0` in `README.md` |
| S1.4 | **Refusal by name:** an unsupported op, argument form, dtype or device refuses with its own name and reason; it never falls back silently. | implemented | `test_shim.py::test_ops_without_a_meta_kernel_name_themselves`; `test_vulkan4.py::test_an_op_that_is_not_taught_refuses_and_names_itself`; `test_argform.py::test_a_per_axis_differing_stride_is_still_refused_by_name` |
| S1.5 | The `meta` device carries shape and dtype without storage, with the dense kernels' promotion rules, enough for `from_pretrained` init paths that compute on meta. | implemented | `test_shim.py::test_meta_tensors_carry_shape_and_dtype_and_no_data`, `::test_the_llama3_rope_init_runs_on_meta_end_to_end` |
| S1.6 | Seeded RNG (`uniform_`, `bernoulli_`, …) reproduces upstream's stream. | implemented | `test_shim.py::test_uniform_matches_torchs_stream_bit_for_bit`, `::test_bernoulli_draws_in_double_for_every_dtype` |
| S1.7 | Built against CPython's limited API: one `cp313-abi3` binary loads on 3.13 and later CPythons. | implemented | `test_release.py::test_the_abi3_wheel_loads_on_later_cpythons` (skips by name when no newer `python3.N` is on `PATH`) |
| S1.8 | `torch.compile` refuses by name without breaking `transformers`; it is not a goal (INTENT §7). | implemented | `test_shim.py::test_torch_compile_refuses_by_name_without_breaking_transformers` |

## S2. Running real models

| # | Behaviour | Status | Evidence |
|---|---|---|---|
| S2.1 | Real `transformers` models load through `from_pretrained` with bit-identical weights and forward to upstream's logits. | implemented | `test_shim.py::test_from_pretrained_loads_the_real_weights_bit_for_bit_on_all_four_paths`, `::test_from_pretrained_forward_matches_upstream_logits`, `::test_a_real_transformers_llama_forward_matches_upstream` |
| S2.2 | `generate` (greedy) produces upstream's tokens. | implemented | `test_shim.py::test_greedy_generate_matches_upstream_token_for_token`, `::test_the_llama3_rope_checkpoint_loads_and_generates_like_upstream` |
| S2.3 | `torch.load`, safetensors and `torch.save` round-trip with upstream (upstream reads what we write). | implemented | `test_shim.py::test_ckpt_safetensors_two_readers_agree_with_torch_load_bit_for_bit`, `::test_save_upstream_reads_every_dtype_and_view_the_shim_wrote_bit_for_bit` |
| S2.4 | Architecture sweep: models that forward (*reaches*) and models that numerically agree with upstream at a derived tolerance (*agrees*). | partial | `test_agree.py::test_verdict_refuses_first_when_upstream_cannot_reproduce_itself` and `test_agree2.py::test_the_two_refusal_outcomes_cannot_be_deleted_into_passes` test the judging rules; the sweep itself (`arch_sweep.py`, `agree_sweep.py`) is a measurement, recorded in `docs/numerics/AGREE.md`, not re-run by the gate |

## S3. Devices

| # | Behaviour | Status | Evidence |
|---|---|---|---|
| S3.1 | `torch.device` labels form a closed vocabulary; an invented label is refused. | implemented | `test_cuda.py::test_cuda_is_a_constructible_label_even_with_no_cuda_in_the_build` |
| S3.2 | `mps` (candle Metal): ops whose kernel would read back to the host are refused by name, enumerable at runtime; softmax stays on device and a BERT encoder forwards and agrees. | implemented | `test_mpsrefuse.py::test_the_derivation_finds_the_ten_without_being_told_their_names`; `test_mpsattn.py::test_a_bert_encoder_forwards_on_mps_and_agrees_with_upstream`; `test_remeasure2.py::test_the_readme_host_readback_count_is_the_number_the_runtime_reports` |
| S3.3 | `vulkan`: a separate `Repr` arm; taught ops run in SPIR-V shaders, everything else refuses naming itself; dispatch counters prove execution. | implemented | `test_vulkan4.py::test_an_op_that_is_not_taught_refuses_and_names_itself`, `::test_the_exactly_rounded_ops_are_bit_identical_to_upstream`; `test_vulkancov.py::test_the_gate_summary_says_UNVERIFIED_when_nothing_ran` (a machine without a Vulkan loader reports the suites as unverified, not passed) |
| S3.4 | `cuda`: wired as a candle device behind `--cfg torch_c_cuda`; unavailability names one of `not_built`/`no_driver`/`no_device`/`wrong_arch`/`unclassified`. **Never compiled or run on a GPU.** | partial | `test_cuda.py::test_asking_for_cuda_on_this_build_refuses_and_names_not_built`, `::test_this_file_never_claims_cuda_computes` |

## S4. The `torchnative` API

| # | Behaviour | Status | Evidence |
|---|---|---|---|
| S4.1 | `torchnative.device` offers `cpu`/`mps`/`vulkan`/`cuda`/`npu`; every availability answer names the probe behind it; `npu` resolves per host and never resolves to the CPU; eager and compiled devices differ by type. | implemented | `test_devicens.py::test_the_namespace_has_the_decided_members`, `::test_npu_never_resolves_to_the_cpu`, `::test_eager_and_compiled_are_distinguished_by_type_not_by_convention` |
| S4.2 | `nn.Module.to(<torchnative device>)` keeps upstream semantics for every ordinary argument and returns the same module. | implemented | `test_devicens.py::test_to_is_unchanged_for_every_ordinary_argument_form`, `::test_ordinary_calls_reach_upstream_with_identical_arguments` |
| S4.3 | `torchnative.transformers` exposes every `Auto*` class (enumerated, not hand-listed) subclassing upstream; unsupported arguments (`export=`, `load_in_4bit=`) refuse by name. | implemented | `test_tntransformers.py::test_the_family_is_enumerated_not_hand_listed`, `::test_export_refuses_by_name`, `::test_load_in_4bit_refuses_by_name` |
| S4.4 | `import torchnative` does not import `torch`; every subpackage is reachable. | implemented | `test_tnnamespace.py::test_importing_torchnative_does_not_import_torch`; `torchnative/src/test/test_import.py` |

## S5. Accelerator back ends (front end fixed, back end swapped)

| # | Behaviour | Status | Evidence |
|---|---|---|---|
| S5.1 | Graph capture through the single door, decomposition and refolding to a core op set (substrate for NPU delegates). | implemented | `test_export.py::test_capture_is_the_only_working_front_end_and_records_the_module_it_ran`; `test_shim.py::test_decompose_refuses_by_name_what_it_cannot_lower` |
| S5.2 | **Apple / CoreML:** eligible leaves lower to CoreML; a float16 graph runs on the Neural Engine; every leaf not lowered is named and a partial offload warns. | implemented | `test_coremlops.py::test_the_neural_engine_runs_the_conv_at_float16_and_cannot_at_float32`, `::test_every_leaf_not_lowered_is_named_and_a_partial_offload_warns` (macOS only; `MLComputePlan` is nondeterministic under load, AGENTS.md §13.3) |
| S5.3 | **Windows / Intel NPU:** `model.to(device.npu)` lowers eligible `nn.Linear` leaves through OpenVINO and returns the same module; zero lowered leaves is a refusal. **Tested against a faked probe and runtime; no Intel NPU has run it.** | partial | `test_npuwire.py::test_to_device_npu_returns_a_real_module_on_a_faked_intel_npu_host`, `::test_zero_leaves_lowered_is_a_refusal_and_not_a_success`; `test_intelnpu.py` |
| S5.4 | **Android / NNAPI:** the captured graph lowers to an NNAPI blob that executes through `ANeuralNetworksModel`. Only the CPU reference driver has run it — execution, not acceleration. | partial | `test_npu2.py` |
| S5.5 | **Android / QNN (ExecuTorch):** a submodule is delegated and the model still generates; the HTP ahead-of-time half refuses by name here. No claim that anything ran on an NPU. | partial | `test_qnn.py::test_a_real_checkpoint_still_generates_with_a_submodule_delegated`, `::test_no_claim_is_made_that_anything_ran_on_an_npu` |
| S5.6 | Quantisation by **module replacement** (`torchnative.quant`, Q8_0/Q4_0/Q4K), reachable through a registered `HfQuantizer` so leaves are swapped before the weights land. | implemented | `test_shim.py::test_the_quantizer_plugin_replaces_the_leaves_before_the_weights_land`, `::test_the_quantizer_plugin_and_quantize_produce_the_same_model` |

## S6. Training, adaptation and federation

| # | Behaviour | Status | Evidence |
|---|---|---|---|
| S6.1 | `loss.backward()` through upstream's own autograd path, with optimizer steps agreeing with upstream; double-backward, `autograd.Function` and hooks refuse by name. | partial | `test_shim.py::test_a_real_training_loop_runs_through_loss_backward_and_agrees_with_upstream`; `test_train.py` |
| S6.2 | `torchnative.delta.Delta`: owns base and offset; a revert restores the base bit-for-bit; a delta is written and read back bit-for-bit; unsupported lifetimes name the check that would prove them. | implemented | `test_shim.py::test_a_delta_reverts_the_base_weights_bit_for_bit`, `::test_a_delta_is_written_and_read_back_bit_for_bit`, `::test_a_delta_names_a_check_for_the_destination_it_cannot_reach` |
| S6.3 | `adapt` stage 1 (gradient): `Tent` lowers prediction entropy and agrees with upstream autograd; wrong sign and `lr=0` do not. | implemented | `test_shim.py::test_tent_reduces_prediction_entropy_and_upstream_agrees`, `::test_the_wrong_sign_and_a_zero_step_do_not_reduce_entropy` |
| S6.4 | `adapt` stage 0 (no backward): normalisation statistics re-estimated as upstream computes them, and reverted. | implemented | `test_stage0.py::test_the_statistics_it_computes_are_the_statistics_upstream_computes`, `::test_revert_after_a_stage_0_run_restores_the_running_statistics` |
| S6.5 | Stage 0 and stage 1 separated **by type**, so a backward-free build rejects gradient methods at import time (INTENT §4). Stage 0 is `adapt.StatisticsMethod`; stage 1 is `adapt.gradient.GradientMethod` (`Tent`), reached lazily as `adapt.Tent`; `wrap` admits each stage only as its type. "Backward-free" is a **configuration**, `TORCHNATIVE_BACKWARD=off` at `torchnative.adapt` import (no build of this crate lacks the tape): under it `adapt.gradient` refuses to import, and any `Method` subclass declaring stage 1 or 2 is refused at class creation, each by name and reason; stage 0 still adapts and reverts without loading the stage-1 module. | implemented | `test_stagetype.py::test_stage_0_and_stage_1_are_distinct_types`, `::test_a_backward_free_build_refuses_the_stage_1_module_at_import_by_name`, `::test_a_stage_1_method_a_user_defines_is_refused_when_its_class_is_created`, `::test_stage_0_still_adapts_and_reverts_in_a_backward_free_build`, `::test_a_stage_declared_by_attribute_and_not_by_type_is_refused_at_wrap`, `::test_an_unrecognised_backward_setting_is_refused_by_name` |
| S6.6 | `torch.distributed` via `ProcessGroupLocal` over loopback TCP: eleven collectives and five reduce ops agree with upstream gloo at world 3 and 4; bitwise ops and point-to-point refuse by name. | implemented | `test_collect2.py::test_every_collective_matches_upstream_gloo_at_world_three_and_four`, `::test_point_to_point_still_refuses_by_name_and_was_not_weakened` |
| S6.7 | Federated averaging across processes equals the same average computed centrally; a world of one and unsupported round shapes refuse by name. Secure aggregation, differential privacy and participant selection are not implemented. | partial | `test_shim.py::test_fedavg_over_two_processes_equals_the_same_average_computed_centrally`, `::test_federated_refuses_a_world_of_one_by_name_at_every_door`, `::test_multi_round_fedavg_equals_the_same_rounds_computed_centrally` |

## S7. Kernels and orchestration

| # | Behaviour | Status | Evidence |
|---|---|---|---|
| S7.1 | `torchnative.kernels`: a bundle resolver satisfying the HF `kernels` contract, resolving at build time on mobile. | planned | `torchnative/src/main/torchnative/kernels/__init__.py` is a docstring only; `torchnative/src/test/test_import.py` only imports it |
| S7.2 | `torchnative.api.TorchNativeAPI`: deployment, lifetime policy, device orchestration. | planned | skeleton class; no behavioural test |
| S7.3 | Flash-attention / flash-linear-attention kernels across platforms. | planned | no test |

## S8. Distribution

| # | Behaviour | Status | Evidence |
|---|---|---|---|
| S8.1 | Nine platform wheel targets, all `cp313-abi3`; Android x86_64 refuses by name; tags match what `packaging` generates. | implemented | `test_wheelmatrix.py::test_the_android_tags_are_what_packaging_generates`, `::test_the_readme_does_not_call_a_never_executed_target_measured` |
| S8.2 | The version lives only in `pyproject.toml`, is a pre-release, and the README never claims an unpublished version is on PyPI. | implemented | `test_release.py::test_pyproject_is_the_only_place_a_version_is_declared`, `::test_the_readme_does_not_claim_an_unpublished_version_is_on_pypi` |
| S8.3 | Release CI: a `v*` tag builds and publishes the wheels via PyPI Trusted Publishing; the tag must equal the version. | implemented | `test_cipub.py` (static checks over `.github/workflows/publish-pypi.yml` and `tools/ci/check_tag_version.py`) |
| S8.4 | `main` carries a reduced, CI-generated layout reached only by a pull request from `release` (`tools/release/sync-release.sh`). | partial | `tools/release/test-sync-release.sh` passes; `tools/release/publish_main.sh` (a different, local mechanism) is still present and tested by `test_publish.py` — see PROJECT.md "릴리스" |

---

## Outside intent — needs a decision

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
