# W13: double backward, `torch.autograd.Function` and hooks (issue #10)

SPEC S6.1 said `loss.backward()` agrees with upstream and that double-backward,
`torch.autograd.Function` and hooks "refuse by name". This round measured each refusal before
designing anything, and **two of the three were not refusals at all**: they were silent wrong
answers. All three now agree with upstream element-wise on a training workload, in a separate
subprocess, against a tolerance derived per quantity (AGENTS.md §16).

Environment: worktree `work/train-autograd` on `develop` `652c7d7`, torch 2.13.0 as the oracle,
`.caches/spike-venv` of the main checkout. The machine was shared with other projects at load
averages between 16 and 168 throughout, so **no time is reported anywhere below**.

---

## 0. Answers, before the evidence

| question | answer |
|---|---|
| **Why did each one "refuse"?** | Three different causes, measured with a live probe against upstream (§1). `create_graph=True` really refused, in `run_backward`, because the eager backward ran under `NoGradGuard`. `autograd.Function` **did not refuse**: its forward ran with grad mode on, so the tape differentiated the forward's ops and never called `backward`. A leaf `register_hook` **did not refuse**: `_backward_hooks` was stored and never read. A non-leaf hook and a module full backward hook did refuse, at `grad_fn._register_hook_dict` and at upstream's own "please open an issue" |
| **Double backward?** | **Agrees.** A WGAN-GP critic trained three steps through `torch.autograd.grad(..., create_graph=True)`, and a Hessian-vector product. §4 |
| **`autograd.Function`?** | **Agrees.** A LoRA-style low-rank Function with a hand-written backward and `save_for_backward`, a gradient-reversal layer, and a straight-through estimator with `setup_context`, `@once_differentiable` and a non-differentiable output, in one network trained three steps. §2 |
| **Hooks?** | **Agree.** Leaf hooks (clipping, rescaling, removal by handle), a non-leaf hook installed from a forward hook, `register_full_backward_hook` and `register_full_backward_pre_hook`, trained three steps; every tensor a hook was handed and the order they fired in are compared too. §3 |
| **What still refuses by name?** | `ctx.mark_dirty`; a gradient-path `Function` inside a capture region; `register_post_accumulate_grad_hook`; the legacy `Module.register_backward_hook`; `create_graph` through `embedding`, and through any op whose first-order rule issues an op with no rule of its own (`relu`, `gelu`, `layer_norm` measured). §5 |
| **Defects found on the way?** | One more besides the two silent ones: a **seed tensor** for a non-scalar output was iterated as a sequence, so `y.backward(torch.ones_like(y))` with more than one row refused as "1 output and N gradients". §4.2 |

---

## 1. What each refusal actually was

A probe ran each case on the shim and on upstream, before any change. Reproduced here as measured.

| case | upstream 2.13.0 | shim before this round | kind |
|---|---|---|---|
| `torch.autograd.grad(y, x, create_graph=True)` then `.backward()` | `[36, 288, 972]` | `NotImplementedError: ... create_graph=True -- the eager backward runs under a NoGradGuard` | **refusal by name** |
| `Function` whose `forward` is `x.clone()`, `backward` is `-g` (gradient reversal) | `[-3, -4]`, `backward` called once, `grad_fn` `GRLBackward` | **`[3, 4]`**, `backward` called **zero** times, `grad_fn` `CloneBackward0` | **silent wrong answer** |
| same, `forward` `x.view_as(x)` | `-g` | `no derivative rule for aten.view_as.default` | a refusal, but naming the wrong thing |
| `Function` whose `forward` is `x*w*x` with a correct `backward` | agrees | agrees -- **by coincidence**: the forward's ops were differentiated instead | silent, correct only when `backward` is the true derivative |
| leaf `x.register_hook(lambda g: g * 10)` then `backward()` | `[20, 40, 60]` | **`[2, 4, 6]`** | **silent wrong answer** |
| leaf hook, through `torch.autograd.grad` | `[20, 40]` | **`[2, 4]`** | **silent wrong answer** |
| non-leaf `y.register_hook(...)` | `[80, 160, 240]` | `NotImplementedError: grad_fn._register_hook_dict` | refusal by name |
| `Module.register_full_backward_hook` | fires | `RuntimeError: Error while setting up backward hooks. Please open an issue` | upstream's message, not ours |
| `register_post_accumulate_grad_hook` | fires | `AttributeError: '_post_accumulate_grad_hooks'` | bare attribute error |

The causes, in the code:

* **`create_graph`** -- `_install_engine`'s `run_backward` raised before reaching the tape
  (`bootstrap.py`). The reason it gave was true: `capture.rs`'s `eager_backward` wrapped
  `tape::backward_in` in `NoGradGuard`, so nothing a rule dispatched was marked or recorded.
* **`autograd.Function`** -- `_FunctionBase.apply` was written for `bloom`'s forward-only
  `GeLUFunction` (docs/architectures/ARCH20.md §6) and ran `cls.forward(ctx, ...)` under whatever grad
  mode the caller had. Once docs/training/BACKWARD9.md opened `run_backward`, every op inside that forward was on the
  eager tape with a rule-bearing name, and `ctx.backward` had no node to be called from.
* **Hooks** -- `tensor.rs`'s `backward_hooks` slot says so in its own doc comment: "nothing ever
  fires a backward hook". `Tensor.register_hook` on a leaf only writes that slot, so it succeeded;
  on a non-leaf it also calls `grad_fn._register_hook_dict`, which the hollow `_GradFnNode` refuses.
  `nn.Module`'s full backward hooks are built by `torch/utils/hooks.py:BackwardHook` out of an
  `autograd.Function` (`BackwardHookFunction`) and `grad_fn.name()` / `grad_fn.register_hook`, so
  they could not work before `Function` did -- docs/training/BACKWARD9.md §6 had guessed "cheaper
  after `Function` than before it", and that was right.

The probe scripts asserted `hasattr(torch._C, "_aten_implemented")` on the shim side and its
absence on the upstream side, so neither half read the other's torch.

---

## 2. `torch.autograd.Function`

`_FunctionBase.apply` (`bootstrap.py`) is now upstream's `THPFunction_apply` in the shape the
eager tape can hold. When grad mode is on and a tensor argument requires grad:

1. `forward` runs **under no-grad**, as upstream's does, so its inner ops are not recorded and
   cannot be differentiated in place of `backward`.
2. The call is recorded as **one** tape node, `autograd.Function` (`tape.rs`'s `FUNCTION_OP`), by
   `torch._C._eager_record_function` (`capture.rs`). Its first argument is `ctx`, held as a
   literal -- this is the tape holding a *callback* rather than an op -- and its second is one
   entry per forward argument, the tensor or `None`.
3. Differentiable outputs get `from_op = "autograd.Function:<Name>Backward"`, and their `grad_fn`
   **is** `ctx`, as upstream's is. An input returned as-is becomes a view of it, as upstream's does
   -- `BackwardHookFunction.forward` returns its arguments, so this is not an edge case.
4. When the walk reaches the node, `tape.rs`'s `function_backward` binds the recorded tensor
   arguments to their `Ref`s and calls `ctx._shim_run_backward`, which is upstream's `PyNode::apply`
   plus `validate_outputs`: materialise missing output gradients as zeros unless
   `set_materialize_grads(False)`, run node pre-hooks, call `BackwardCFunction.apply` (which resolves
   `vjp` and `boxed_grads_call`), check the count with upstream's message, sum a broadcast gradient
   down to the input's shape, cast to the input's dtype, run node post-hooks.

Python's grad mode is set to the door's for that call. They differ inside a tape backward
(`NoGradGuard` touches only the door's mirror), and `@once_differentiable` keys on Python's.

`grad_fn`'s identity is kept in `_FUNCTION_NODES`, **weakly on both sides**. `ctx` is alive while
the tape holds it, which is the window `register_hook` is called in; a strong reference would be a
global root for the `ctx.to_save -> output -> ctx` cycle that `save_for_backward(output)` makes.
After the tape is freed, `grad_fn` falls back to the hollow node with upstream's class name. This is
a recorded divergence: upstream's `grad_fn` stays `ctx` for the life of the output.

Evidence: `test_backward10.py::test_custom_autograd_functions_train_and_agree_with_upstream`,
`::test_a_custom_function_backward_is_what_runs_not_the_forward_ops`,
`::test_a_custom_backward_is_validated_the_way_upstream_validates_it`.

---

## 3. Hooks

| hook | where it fires here | upstream's point |
|---|---|---|
| leaf `register_hook` | `run_backward`'s `_leaf_gradient`, on the summed gradient, before `.grad` accumulation **and** before `torch.autograd.grad` returns it | `AccumulateGrad`'s tensor pre-hooks; captured gradients are post-hook since 2.0 |
| non-leaf `register_hook` | `tape.rs`'s `run_tensor_hooks`, when the reverse walk reaches the node that produced the tensor. Every consumer has a higher index and has been walked, so the gradient is the total, once | the producing node's tensor pre-hooks |
| `Function` node `register_prehook` / `register_hook` | `_shim_run_backward`, before and after the user's `backward` | `PyNode` pre-/post-hooks |
| `Module.register_full_backward_hook` / `_pre_hook` | free: `BackwardHook` is built from the three rows above | same |

`_GradFnNode._register_hook_dict` is a method now, with no body: the tape reads `_backward_hooks`
off the very object an op produced, so there is nothing for the node to remember.

Evidence: `test_backward10.py::test_tensor_and_module_hooks_train_and_agree_with_upstream`.

---

## 4. Double backward

### 4.1 The design, and the shorter version that would have been wrong

`create_graph=True` runs the walk **on** the tape. `eager_backward` (`capture.rs`) takes a new
`create_graph` argument; it implies `retain_graph`, and `backward_in` runs under
`GradModeGuard::enter(true)` (`tensor.rs`) instead of `NoGradGuard`. Every op a rule dispatches is
then marked by the door and appended to the **live** eager tape.

That the tape is the live one, reached through `Recorder::duplicate_for_retained_backward`, is the
point. Its `known` map still names the forward's values, so a rule that reads a forward result --
`tanh`'s `1 - y*y` -- records a node *of* that result, and the second backward differentiates
through it. Had the backward taken the tape instead, the forward's values would be strangers to the
fresh tape that recorded the first backward's ops, would read as constants, and every second-order
term through them would be silently missing. §6's N5b is the mutation for this, and it has not been run yet.

`.grad` under `backward(create_graph=True)` is stored with its graph and added out of place after,
which is upstream's `AccumulateGrad` under grad mode.

### 4.2 The seed defect

`backward_in` read a caller's seed with `extract::<Vec<_>>()` first, and a tensor *is* a sequence:
a `[6, 1]` seed came back as six rows. So `y.backward(torch.ones_like(y))` and every
`grad_outputs=torch.ones_like(d)` with more than one row refused ("this trace has 1 output(s) and
backward() was given 3 gradient(s)", measured on `x * 2` over a `[3, 2]` leaf). It was found because
the WGAN-GP test is written the way gradient penalties are written. A tensor is now taken as one
seed before the sequence form is tried.

### 4.3 How far it reaches, measured

A one-hidden-layer critic, `torch.autograd.grad(d, x, ones, create_graph=True)`, then
`(||g||^2 - 1)^2` backward, first four elements of `W1.grad` against upstream:

| activation | result |
|---|---|
| `tanh`, `sigmoid`, `silu`, `exp`, `softmax`, `log_softmax` | agree to the printed 6 digits |
| `relu` | refused: `no derivative rule for aten.where.ScalarOther` (its first-order rule's mask-select) |
| `gelu` | refused: `no derivative rule for aten.erf.default` |
| `layer_norm` | refused: a gradient reached `native_layer_norm`'s returned mean |
| `embedding` (weight gradient) | refused by name: built by `index_put_`, an in-place write the tape does not record (§5) |

Each refusal names the op; none of them returns a number. A custom `Function` with a
differentiable `backward` double-differentiates (`x**3` gives `6x`, matching upstream); one marked
`@once_differentiable` refuses the second backward on both sides, with different wording.

Evidence: `test_backward10.py::test_a_wgan_gradient_penalty_trains_and_agrees_with_upstream`,
`::test_a_seeded_backward_from_a_non_scalar_output_agrees_with_upstream`,
`test_shim.py::test_the_engine_refuses_several_roots_by_name_and_records_create_graph`
(the old `create_graph` refusal test, inverted rather than deleted).

---

## 5. What still refuses, by name

| refused | measured reason |
|---|---|
| `ctx.mark_dirty` | An input written in place would have to become a non-leaf with this node as its `grad_fn`. This shim never rewrites the leafness of a tensor that already exists (`mark_from_op`'s last clause, docs/training/BACKWARD4.md §4.2). Before this round it was accepted without a word (measured: the `Dirty` case in `test_a_custom_function_refuses_mark_dirty_by_name` raised nothing) |
| a gradient-path `Function` inside `_capture_begin` | a trace replays aten ops through the door; a Python callback is not one |
| `register_post_accumulate_grad_hook` | upstream keeps the hook dict in a C slot that pickling skips. The only per-tensor storage a property can reach here is `__dict__`, which `torch.save` of a hooked `Parameter` would try to pickle. A Rust slot beside `backward_hooks` is the way; not built |
| legacy `Module.register_backward_hook` | calls `register_hook` on an aten op's `grad_fn`, the hollow node, which has no identity to hang a hook on. Deprecated upstream; `register_full_backward_hook` works |
| `create_graph=True` through `embedding`'s weight gradient | `index_put_` into a zero table is never recorded, so the gradient would come back without the graph a second backward needs |
| `create_graph=True` through `relu`, `gelu`, `layer_norm` | the ops their first-order rules issue have no rules of their own (§4.3). Each is a rule plus a float64 finite-difference case in `test_shim.py` |
| several root tensors, `GradientEdge`, `retain_grad` on non-leaves | unchanged from docs/training/BACKWARD9.md §6 |

---

## 6. What was checked by being switched off

Each mutation was applied to the source, built into the vendored tree through
`vendor/install_shim.sh`, the suite run, and the source restored. AGENTS.md §17.5.

| # | mutation | what went red |
|---|---|---|
| N1 | leaf hooks not run in `_leaf_gradient` | `test_tensor_and_module_hooks_...` (`first_grads` off by 0.39) |
| N2 | `run_tensor_hooks` returns the gradient untouched | `test_tensor_and_module_hooks_...` (hook call order: no `tanh` entries) |
| N3 | node post-hooks skipped | `test_tensor_and_module_hooks_...` (no `full`/`pre` entries) |
| N4 | `Function.apply` back to the old path (forward under grad mode, no node) | five tests: the isolated reversal, the training network, validation, `mark_dirty`, the hook network |
| N5 | Rust no longer forces `retain_graph` under `create_graph` | **nothing** -- upstream's Python already passes `retain_graph=create_graph`, so the line is redundant for every upstream caller. Not the mutation it was meant to be; N5b below is |
| N5b | `create_graph` *takes* the tape instead of duplicating it | **not yet run** -- needs a build, held |
| N6 | `create_graph` backward under `NoGradGuard` | the gradient-penalty test, the embedding refusal, `test_shim`'s inverted `create_graph` test |
| N7 | seed-tensor fix reverted | the seeded-backward test and the gradient-penalty test |
| N8 | `embedding`'s `create_graph` guard removed | `test_create_graph_through_a_rule_with_no_graph_refuses_by_name` |
| N9 | zero-materialisation skipped | `test_custom_autograd_functions_...` (`gpos` is `None` where upstream passes zeros) |
| N10 | a hook's return value ignored | `test_tensor_and_module_hooks_...` |
| N11 | broadcast gradient not summed to the input's shape | `test_a_custom_backward_is_validated_...` |
| N12 | gradient-count check removed | `test_a_custom_backward_is_validated_...` (a different message from upstream's) |
| N13 | node **pre**-hooks skipped | **nothing, in the first suite.** `BackwardHook` uses only post-hooks, so no test reached a pre-hook. `test_node_pre_and_post_hooks_on_a_custom_function_agree_with_upstream` was written for it and goes red on the N13 build (`seen_g` off by 2.33) |
| N14 | `grad_fn` falls back to the hollow node instead of `ctx` | **not yet run** -- needs a build, held |


---

## 7. The tolerances

Every bound below is derived: upstream runs the program again in float64, and the bound is the p90
of upstream's own float32-vs-float64 error relative to the quantity's largest magnitude, floored at
8 float32 ulp (docs/numerics/AGREE.md §2). Worst shim-vs-upstream difference / derived bound, as
printed by the passing run:

| test | quantity: worst / bound |
|---|---|
| gradient penalty | `g` 2.98e-08 / 1.96e-07 · `first_grads` 9.54e-07 / 4.78e-06 · `interp_grad` 5.96e-08 / 1.59e-07 · `losses` 1.43e-06 / 5.55e-06 · `params` 5.96e-08 / 6.02e-07 · `hvp_g` 1.19e-07 / 2.39e-06 · `hvp` 2.98e-08 / 3.43e-06 |
| `autograd.Function` network | `first_grads` 2.24e-08 / 5.90e-08 · `losses` 1.19e-07 / 1.69e-06 · `params` 2.98e-08 / 4.70e-07 |
| hooks network | `first_grads` 5.96e-08 / 5.61e-07 · `losses` 1.19e-07 / 1.03e-06 · `params` 2.98e-08 / 4.97e-07 · hook inputs <= 1.49e-08 against bounds >= 6.42e-08 · `grad_hooked` 0 / 1.91e-05 |


---

## 8. Reported by kind, not by count

AGENTS.md §17.3.

| kind | what |
|---|---|
| **feature added** | `torch.autograd.Function` on the gradient path: one tape node holding `ctx`, user `backward` called, `grad_fn` is `ctx`, upstream's validation (§2) |
| **feature added** | tensor hooks: leaf and non-leaf `register_hook`; node `register_hook`/`register_prehook`; `Module` full backward hooks and pre-hooks through them (§3) |
| **feature added** | `create_graph=True`: the backward recorded on the live tape (§4) |
| **defect fixed** | a custom `Function`'s `backward` was never called -- silent wrong gradient (§1) |
| **defect fixed** | leaf `register_hook` never fired -- silent wrong gradient (§1) |
| **defect fixed** | a non-scalar seed tensor iterated as a sequence (§4.2) |
| **defect fixed** | `ctx.mark_dirty` accepted silently; now refused by name |
| **tests added** | 10 in `test_backward10.py`; 1 inverted in `test_shim.py` (none deleted) |
| **documentation corrected** | SPEC S6.1, README, `docs/locale/README_ko.md`, `docs/platform/STATUS.md` (two places), a note on docs/training/BACKWARD9.md §6, `_install_engine`'s docstring |


---

## 9. Gates

Not yet run for this round's final tree: a build and gate hold was in force on the machine when
this was written. The new suite was run by hand against the build before the last nullification;
the final tree's gate is pending.


---

## 10. What this document does not establish

| # | not established | why |
|---|---|---|
| 1 | That a `transformers` model trains with any of these | every workload here is a hand-built `nn.Module`. `bloom`'s `GeLUFunction` now takes the differentiable path whenever its input requires grad, and nothing here trains `bloom` |
| 2 | Double backward through convolution, normalisation or attention | not measured beyond §4.3's activation table |
| 3 | `grad_fn` identity after the tape is freed | falls back to a hollow node by design (§2) |
| 4 | Any time or memory cost | the machine was loaded by other projects throughout |
| 5 | Thread safety of `_FUNCTION_NODES` | the eager tape is per thread and the registry is a plain dict; nothing here runs a backward on two threads |

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs function_backward present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs run_tensor_hooks present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tape.rs FUNCTION_OP present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/capture.rs eager_record_function present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/tensor.rs GradModeGuard present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _shim_run_backward present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _leaf_gradient present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/bootstrap.py _FUNCTION_NODES present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_backward10.py test_a_wgan_gradient_penalty_trains_and_agrees_with_upstream present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_backward10.py test_custom_autograd_functions_train_and_agree_with_upstream present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_backward10.py test_a_custom_function_backward_is_what_runs_not_the_forward_ops present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_backward10.py test_tensor_and_module_hooks_train_and_agree_with_upstream present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_backward10.py test_a_seeded_backward_from_a_non_scalar_output_agrees_with_upstream present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_backward10.py test_a_custom_backward_is_validated_the_way_upstream_validates_it present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_engine_refuses_several_roots_by_name_and_records_create_graph present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_shim.py test_the_engine_refuses_create_graph_and_several_roots_by_name absent -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_backward10.py test_node_pre_and_post_hooks_on_a_custom_function_agree_with_upstream present -->
