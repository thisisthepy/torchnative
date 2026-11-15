# DDP: a Python reducer under upstream's names

Issue #247. `docs/distributed/BACKEND.md` D13, recorded there as "decided (leader), formerly Q7":
"DDP gets a Python `Reducer` under upstream's name, so user code runs unchanged." The goal is the
leader's; **every design choice below is an agent proposal, not confirmed.**

## 1. Grade: builds

Measured on 2026-11-13 in the worktree `.worktrees/tn-dist-ddp` (branch `feat/dist-ddp-reducer`):
`python -m py_compile` of `torchnative/python/torchnative/distributed/_reducer.py`,
`torchnative/python/torchnative/distributed/__init__.py` and `tests/training/test_ddp.py`, and
`compile()` of the probe sources inside the test. That is the **builds** grade of AGENTS.md §16 and
nothing more. No rank has run, no gradient has been compared, and the tests in §6 have not been seen
red or green. CI runs them; until it does, nothing in this document is *reaches* or *agrees*.

## 2. What upstream DDP calls, and what each name now is

`torch.nn.parallel.DistributedDataParallel` (`torchnative/python/torch/nn/parallel/distributed.py`,
the vendored 2.13.0 tree) is upstream Python and is not edited. On the default path (CPU module,
`find_unused_parameters=False`, no comm hook) it reaches these C++ names through `torch.distributed`,
which bound them from `torch._C._distributed_c10d` at import (`torch/distributed/__init__.py`):

| Name | Called from | Was | Now |
|---|---|---|---|
| `_verify_params_across_processes(pg, tensors, logger)` | `utils._verify_param_shape_across_processes`, constructor | refused | all-gathers each rank's parameter count, sizes and strides; raises `RuntimeError` unless every rank matches rank 0 |
| `_broadcast_coalesced(pg, tensors, buffer_size, src)` | `utils._sync_module_states`, constructor; buffer sync in `forward` | refused | flattens tensors into buckets of at most `buffer_size` bytes per dtype, `dist.broadcast(..., group_src=src)`, copies back on the other ranks |
| `_compute_bucket_assignment_by_size(tensors, limits, expect_sparse, tensor_indices, logger)` | `_ddp_init_helper` | refused | upstream's algorithm: one open bucket per (dtype, device), closed at the current limit, limits advance and the last repeats; sorted by smallest index unless `tensor_indices` is given |
| `Reducer(...)` (14 positional arguments) | `_ddp_init_helper` | a type whose members refused | the class in §3 |
| `Reducer.prepare_for_forward()` | `_pre_forward` | refused | counts iterations |
| `Reducer._rebuild_buckets()` | `_pre_forward` | refused | first checks the prior iteration finished; then, once, re-buckets in the first synced backward's ready order with `[first_bucket_bytes_cap, bucket_bytes_cap]` (or `bucket_size_limits_for_rebuilding`), takes rank 0's layout by two broadcasts, returns `True` |
| `Reducer.prepare_for_backward(outputs)` | `_post_forward` | refused | arms the hooks for this iteration |
| `Reducer.set_logger`, `_set_ddp_runtime_logging_sample_rate`, `_check_reducer_finalized`, `_remove_autograd_hooks` | various | refused | implemented |
| `Logger(reducer)`, `set_construction_data_and_log`, `set_runtime_stats_and_log`, `set_error_and_log`, `_get_ddp_logging_data`, `_set_comm_hook_name` | constructor, `_pre_forward`, `_log_and_throw`, `_get_ddp_logging_data` | refused | §5 |
| `_register_comm_hook`, `_register_builtin_comm_hook` | `register_comm_hook`, `_register_builtin_comm_hook` | refused ("world_size 1 has no peer") | refused by their own names, §4 |
| `GradBucket` | annotation check in `_check_comm_hook` | declared type | unchanged |

`torch._dynamo.utils.get_optimize_ddp_mode()` is read by the constructor; with dynamo's default
config it is not `"python_reducer"`, so the compiled path (which needs
`register_post_accumulate_grad_hook`) is not taken.

**Where they are installed.** `torchnative/distributed/_reducer.py`, `install()`, called from
`torchnative/distributed/__init__.py` after the backend registers. It sets each name on
`torch._C._distributed_c10d` and the re-exported ones on `torch.distributed`, the same object under
both. Agent proposal, chosen over `bootstrap.py` for two reasons: a reducer inside `bootstrap.py` is
a rebuild per change, and no DDP on this shim can reach a peer without `import torchnative.distributed`,
because that import registers `torchnative`, the only backend there is. Under upstream torch
`install()` does nothing: upstream's C++ reducer is real. `bootstrap.py`'s declared placeholders stay
what a process that never imported `torchnative.distributed` sees.

## 3. The reducer

**Arithmetic.** Upstream's default path multiplies each parameter's accumulated gradient by
`1 / world_size` while copying it into the bucket, all-reduces the bucket with `SUM`, and copies the
result back into `.grad`. The reducer does the same, and `ProcessGroupLocal` folds the ranks in
ascending order (`docs/distributed/COLLECT2.md`), so every rank ends with
`sum_r(grad_r * (1/W))`, bit-identical across ranks.

**Hook timing.** Upstream's reducer hooks `AccumulateGrad`, so it runs after `.grad` holds the
accumulated gradient. This shim refuses `register_post_accumulate_grad_hook` (SPEC S6.1), so the
reducer registers `Tensor.register_hook` on each parameter, which sees the *incoming* gradient
before accumulation. The shim's engine (`_ImperativeEngine.run_backward`, `bootstrap.py`) runs, for
each leaf in turn, its hooks and then its accumulation. The hook for parameter `p`:

1. computes `p.grad + incoming` (or `incoming`), the value accumulation is about to produce, and puts
   it, times `1/W`, into its bucket;
2. holds the caller's `p.grad` tensor and sets `p.grad = None`, so the engine's accumulation
   stores a fresh tensor and leaves the caller's alone;
3. launches, in bucket order, every bucket whose gradients are all in, as an `async_op=True`
   all-reduce;
4. in the hook of the **last** parameter: waits every bucket, copies each other parameter's reduced
   slice into its held tensor (so `.grad` keeps its identity, as upstream's in-place `copy_` keeps
   it), zeroes the last parameter's own `.grad` in place and returns its reduced slice, so the
   engine's `add_` leaves exactly the reduced value in the caller's tensor.

Step 2 is what turns the ordering assumption into a refusal: if the engine ever ran every hook before
any accumulation, a parameter's `.grad` would be `None` at step 4 and the reducer raises, naming the
two parameters and this section, instead of adding the incoming gradient on top of the reduced one.

**`no_sync()`.** DDP does not call `prepare_for_backward` inside it, the hooks are disarmed, and the
engine accumulates locally. The next synced backward's step 1 reads that accumulation as `p.grad`.

**A parameter with no gradient.** Its bucket never completes and finalize never runs, as upstream's.
The next forward's `_rebuild_buckets` restores each held `.grad` to the local accumulation and raises
upstream's "Expected to have finished reduction in the prior iteration" with the parameters' indices
and names.

**Hooks registered after DDP.** `prepare_for_backward` moves the reducer's hook to the end of each
parameter's hook dict, since upstream's runs after every tensor hook.

**Known divergence.** `torch.autograd.grad(...)` on a DDP output with hooks armed runs the reducer's
hooks too (upstream's `AccumulateGrad` hook does not fire for it). The parameters' `.grad` are set
aside and the iteration does not finish, so the next forward raises; it is not a silent wrong
gradient, and it is not upstream's behaviour.

## 4. What refuses by name

| Spelling | Where it refuses | Why |
|---|---|---|
| `find_unused_parameters=True` | `Reducer(...)` | needs a graph walk from the outputs and a used-parameter all-reduce each iteration |
| `gradient_as_bucket_view=True` | `Reducer(...)` | this reducer copies into `.grad`; `.grad` as a bucket view is not built |
| `static_graph=True`, `_set_static_graph()` | `Reducer._set_static_graph` | delays the first all-reduce to `Variable._execution_engine.queue_callback`, which this engine lacks |
| `register_comm_hook`, `_register_builtin_comm_hook` | `_register_comm_hook`, `_register_builtin_comm_hook` | hooks take a `GradBucket` and return a `torch.futures.Future`; this shim's `Work` has no future |
| `join()` | `Logger._set_uneven_input_join` | the join hook shadows collectives through `_get_zeros_like_grad_buckets` and `_run_comm_hook` |
| `mixed_precision=...` | its comm hook, then `_set_mixed_precision_param_dtype` | not built |
| sparse gradients, `skip_all_reduce_unused_params=True` | `Reducer(...)` | not built |
| any non-CPU parameter | `Reducer(...)` | `torchnative` registers `devices=["cpu"]` |
| the compiled path (`optimize_ddp="python_reducer"`) | `Reducer(...)` | needs `register_post_accumulate_grad_hook` |

## 5. Logger

Construction facts under upstream's keys: `module_name`, `device_ids`, `backend_name`, `dtypes`,
`bucket_sizes`, `output_device`, `broadcast_buffers`, `has_sync_bn`, `static_graph`, `world_size`,
`rank`, `bucket_cap_bytes`, `find_unused_parameters`, `gradient_as_bucket_view`,
`num_parameter_tensors`, `total_parameter_size_bytes`. At the sampling iterations (the first ten, then
every hundredth): `iteration`, and once after the rebuild `has_rebuilt_buckets`,
`rebuilt_bucket_sizes`, `rebuilt_per_bucket_size_limits`. **Timing stats are not built and are
absent rather than zero**, because an invented duration would be a wrong number.

## 6. Tests (`tests/training/test_ddp.py`) and what makes each fail

Three runs per world (2 and 3): the shim over `torchnative`, upstream DDP over gloo in float32, the
same in float64. The model is two `Linear`s and a `tanh`; data differ per rank; SGD, `lr=0.25`.
Tolerance: AGREE.md §2, both rules, 8-ulp floor, factor 4.

| Test | Fails if |
|---|---|
| `test_ddp_training_agrees_with_upstream_ddp_over_gloo` | any loss, gradient or parameter after 1, 2 or 3 steps, in five scenarios (same init, different init, `no_sync` then synced, `.grad` zeroed in place, one-byte buckets), leaves the tolerance: a wrong divisor, a sum instead of a mean, a wrong slice, a lost `no_sync` accumulation, a double-counted gradient, the wrong broadcast root |
| `test_the_reducer_in_use_is_this_project_s_and_upstream_s_on_the_oracle` | the reducer is not installed, the two namespaces hold different objects, or the oracle ran the shim |
| `test_every_rank_holds_rank_zero_s_parameters_after_construction` | `_broadcast_coalesced` is a no-op, has the wrong root or writes a copy; ranks' parameters differ after a step |
| `test_no_sync_accumulates_locally_and_the_next_step_reduces_the_sum` | the hook reduced inside `no_sync`, or reduced the incoming gradient instead of `.grad` plus it |
| `test_the_grad_tensor_keeps_its_identity_as_upstream_s_does` | a held `.grad` is not restored to the caller's tensor where upstream's stays the same object |
| `test_one_byte_buckets_give_the_same_numbers_as_one_bucket` | one bucket per parameter is not bit-identical to one bucket; the rebuild kept one bucket; rebuilt bucket sizes differ from upstream's as a multiset |
| `test_logging_data_agrees_with_upstream_on_what_it_records` | a construction or rebuild fact differs from upstream's |
| `test_an_unused_parameter_fails_the_next_iteration_as_upstream_does` | a parameter with no gradient is skipped silently or a partial bucket is reduced |
| `test_a_backend_allreduce_that_drops_its_result_makes_the_comparison_fail` | with `ProcessGroupLocal.allreduce` run on a copy in the child, the gradients still agree with upstream (the sabotage) |
| `test_what_is_not_built_refuses_by_name` | a refusal in §4 is accepted, or does not name itself |
| `test_the_tolerance_is_agree_s_and_not_a_free_parameter` | the floor or factor changed, or the comparison accepts a doubled gradient |

What these cannot find: a defect that only a larger model, a second dtype in one model, a buffer
(no scenario has one, so the forward buffer broadcast is unexercised), more than three ranks, or an
iteration past the tenth (the logging sample rate) would show.

## 7. Not verified

Nothing has run (§1). In particular: that `torch._dynamo.utils.get_optimize_ddp_mode()` answers on
the shim; that `p.grad` returns the stored object (the identity test reads it); that tensor-hook
dicts on parameters are ordered dicts (`_move_hooks_last` skips them otherwise); upstream's exact
logging keys, which the logging test reads from upstream rather than assuming, but requires at least
six of nine to exist.

<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/_reducer.py _mark_ready present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/_reducer.py _ensure_prior_reduction_finished present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/_reducer.py _compute_bucket_assignment_by_size present -->
