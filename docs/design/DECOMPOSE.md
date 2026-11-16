# CompositeImplicitAutograd decomposition on the shim (#372)

Status: the direction is a **leader decision 2026-11-16, agent proposal; not
owner-confirmed**: CompositeImplicitAutograd (CIA) decomposition of aten ops
comes before functionalization step 3 (docs/design/FUNCTIONALIZATION.md). The
leader's conditions: (1) the ops covered are those upstream's exports actually
decompose, measured, and everything else refuses by name; (2) upstream's own
Python decompositions (`torch._decomp`, `torch._refs`) are used where they
exist, own compositions only where they do not; (3) results are compared with
upstream's op sequence node for node; (4) the first PR is the frame plus the
Linear+ReLU path only. Everything else on this page, the design and the
choices in section 3 included, is agent proposal, not confirmed.

Written with the code, before anything ran on a built shim (agents compile
only). The upstream facts were measured on torch 2.13.0 (`.caches/spike-venv`)
on 2026-11-16 and are re-derived on every run by
`tests/bindings/test_cia_frame.py` and `tests/export/test_functionalize_export.py`.

## 1. What upstream does, measured

* `aten::linear` is CIA (`_dispatch_has_kernel_for_dispatch_key(name,
  CompositeImplicitAutograd)` True). `aten::t`, `permute` and `transpose.int`
  are CompositeExplicitAutograd, not CIA; `addmm`, `relu`, `mm` are neither.
  `matmul`, `reshape` and `to.dtype` are CIA.
* There is no Python decomposition of `linear`: not in
  `torch._decomp.decomposition_table`, no `py_kernels` entry at CIA, no
  `torch._refs.linear`. `t`'s is `torch._refs.t` (to `transpose`, then
  `permute`), used by Proxy's decomposition table, not by CIA.
* The CIA kernel of `linear`, logged by a `TorchDispatchMode` under
  `_op_dk(CompositeImplicitAutograd, ...)`; `aten.linear.default(...)` and
  `F.linear(...)` log the same:

  | input, bias | emitted ops |
  |---|---|
  | 2-D, any bias | `t, addmm` |
  | contiguous, `bias.dim() == 1` | `view(input, [rows, k]), t, addmm, view(out, [..., n])` |
  | contiguous, `bias.squeeze().dim() == 1` | `squeeze, view, t, addmm, view` (addmm gets the unsqueezed bias) |
  | contiguous, other bias | `squeeze, t, <matmul>, add.Tensor` |
  | non-contiguous, 1-D bias | `t, <matmul>, add.Tensor` |
  | no bias | `t, <matmul>` |

  `<matmul>` is `matmul`'s own CIA decomposition (`mm`; `view, mm,
  _unsafe_view`; `expand, view, bmm, _unsafe_view`; `mv`; `unsqueeze, mm,
  squeeze_.dim`; `dot`, by rank). A 0-D argument raises "both arguments to
  linear need to be at least 1D".
* Where a mode sees `linear` itself: under `torch.inference_mode()`, for an
  inference tensor, and under `_AutoDispatchBelowAutograd` (the autograd keys,
  where CIA sits above the `Python` key, are skipped; the backend key's CIA
  kernel runs after the mode). Plain eager, `no_grad`, and the Python
  dispatcher with nothing registered: the mode sees `t, addmm`.
* A mode on the **pre-dispatch** stack (`torch/_ops.py`,
  `_get_current_dispatch_mode_pre_dispatch`) also sees `linear` itself: the
  `PreDispatch` key is above the autograd keys. `make_fx(F.linear,
  pre_dispatch=True)` records `linear`; plain `make_fx` records `t, addmm`.
  The mode's own re-dispatch (with itself popped) reaches the autograd keys,
  and the CIA kernel decomposes there, below it.
* `torch.export.export(Linear(8, 4) + ReLU)` records `linear, relu`, and it
  does so through the pre-dispatch stack alone: its tracer is
  `make_fx(..., pre_dispatch=True)`, the pre-dispatch ProxyTorchDispatchMode
  receives `linear` and records it, and the FakeTensorMode under it receives
  `t, addmm`. Instrumented, none of `_override_composite_implicit_decomp`,
  `_split_decomp_table_to_cia_and_python_decomp`, `_collect_all_valid_cia_ops`,
  `_get_decomp_for_cia` and `_special_op_to_preserve_cia` is called during the
  export, and `linear`'s `py_kernels` stay empty. Its
  `run_decompositions()` gives `permute, addmm, relu`. During it
  `FunctionalTensorMode` receives `linear, t, addmm, relu` twice (metadata
  pass, then the trace) and `OpOverload.decompose` decomposes `linear` both
  times (through export's `_special_op_to_decompose_cia`, which calls
  `_op_dk(CompositeImplicitAutograd)`), while `t` and `relu` answer
  NotImplemented. Proxy then turns `t` into `transpose.int` and `permute`.
* `run_decompositions()` (not the export trace) preserves or decomposes a
  CIA op through `_override_composite_implicit_decomp`
  (torch/export/exported_program.py): for every op `_collect_all_valid_cia_ops`
  finds, which asks `_dispatch_has_kernel_for_dispatch_key(op, CIA)`, it
  registers Python kernels at the autograd keys (`autograd_not_implemented`,
  which calls the op below autograd), at CIA (the table's callable:
  `_special_op_to_decompose_cia` for `linear` in the default table, which
  calls `_op_dk(CompositeImplicitAutograd)`), and at the backend keys, and a
  fake rule that runs the original CIA through `_op_dk`. Those take effect
  under the Python dispatcher, which its re-trace enables.
* `torch._decomp.core_aten_decompositions()` is a `CustomDecompTable`: the
  post-autograd table (435 entries) plus `_get_decomp_for_cia(op)` for each of
  the 469 ops `_collect_all_valid_cia_ops_for_aten_namespace()` finds, 940 in
  all, `linear` among them. The shim's table grows by exactly the ops whose
  CIA answer is True here (`linear`: 419 to 420,
  `tests/_support/test_shim.py::test_decompose_gets_the_full_upstream_table_now`).
* `aten.t.default._op_dk(CompositeImplicitAutograd, x)` crashes the upstream
  process (exit 139): `_op_dk` at a key the op has no kernel for is not a
  decomposition upstream either.

## 2. The frame

Three pieces, all in `bootstrap.py` except the door's two calls in `aten.rs`:

* **`_CIA_TABLE`**: op key to body, with its source (`torchnative
  composition`, `upstream torch._decomp`, `upstream torch._refs`) and the
  upstream measurement it was checked against. `torch._C._shim_cia_table()`
  reads it back. The door's set of keys (`_shim_arm_cia`, `_shim_armed_cia`)
  is armed from the table at install and from nothing else.
* **The door's two phases** (`_shim_cia_dispatch`). `door`, right after the
  `torch.library` CIA phase and before the Functionalize kernel, the modes and
  the subclasses: the body runs unless upstream would not decompose there
  (a mode on the pre-dispatch stack, which receives the op first and runs the
  body when it re-dispatches; below autograd or in inference mode; under the
  Python dispatcher with an autograd kernel registered from Python, which is
  `run_decompositions` preserving the op),
  or the op has a native kernel and nothing could observe the decomposition
  (no mode, no functional tensor, no subclass argument), so eager stays as it
  was. `backend`, just before the native table: the body runs for an op that
  was handed down and that nothing answered, unless the op has a native
  kernel.
* **`_op_dk(CompositeImplicitAutograd, ...)`** (`_shim_cia_op_dk`): the body,
  or a refusal naming the op ("CompositeImplicit decomposition not
  implemented", issue #372; or "has no CompositeImplicitAutograd kernel
  upstream" for an op that is not CIA). This is what `OpOverload.decompose`,
  export's `_special_op_to_decompose_cia`, NestedTensor and DTensor call.

Every op a body emits goes back through `_aten_dispatch` from the top
(`_cia_emit`), so Functionalize, the modes and the subclasses see each one in
upstream's order. `torch._C._shim_cia_counters()` counts, per op, how often
its body ran through each entry (`door`, `backend`, `op_dk`), so a test can
show that the decomposition ran and not a kernel.

`torch._C._nn.linear` (`F.linear`) now enters the door as
`aten.linear.default` instead of composing `t` + `addmm` itself: before, no
mode ever saw it. That alone was not enough. The first CI run of #373 (run
43756286355) still recorded `t, addmm, relu`, because the door's CIA phase
ran before the mode stack was consulted, so the body decomposed `linear`
above export's pre-dispatch tracer. `_shim_cia_dispatch` now declines at the
door while a pre-dispatch mode is active (`pre_dispatch_mode` in
`_install_cia`), which is upstream's key order, not a special case for
export: aten.rs already consults that stack before the ordinary one for the
same reason (its own `pre_dispatch_mode`).

## 3. What `_dispatch_has_kernel_for_dispatch_key(op, CIA)` answers

True exactly when upstream answers True (the `native_functions.yaml` scan,
measured equal to upstream's 743 aten names) and the op has a body. Every
other aten op answers False, as before #372.

Upstream's answer (True) for an op without a body would send every caller
that decomposes to `_op_dk`, which refuses by name; that is the answer that
cannot hide a graph difference. It was not taken here because the callers are
not only graph builders. `FakeTensorMode` decomposes every CIA op it meets
that has no Python meta, including ops upstream's own Python decompositions
emit inside it (`aten.to.dtype` from `pw_cast_for_opmath` under `addmm`'s
decomposition, measured in the Linear+ReLU `run_decompositions` log). And 42
upstream-CIA overloads (counted against aten.rs's `IMPLEMENTED` and
`IMPLEMENTED_AWAITING_GOLDEN` lists) have a native kernel in this shim and reach the modes
as themselves today (`reshape`, `contiguous`, `matmul`, `view_as`, `rms_norm`,
`chunk`, `where.self`, `var`/`std`, `max_pool2d`, ...; aten.rs
`matmul_default` records that divergence). True would turn every fake or
traced call through them that works today into a refusal at once.

So the remaining silent class is named instead: `_shim_cia_table()["pending"]`
lists the upstream-CIA ops with a native kernel here and no body, and
`tests/bindings/test_cia_frame.py` pins it against upstream so it can only
shrink by a body landing. Switching to upstream's answer is one line
(`answers_true` in `_install_cia`) and is the leader's call.

## 4. Adding an op

1. Measure upstream (torch 2.13.0, the spike-venv oracle): is the op CIA
   (`_dispatch_has_kernel_for_dispatch_key`), and what a logging
   `TorchDispatchMode` sees under `op._op_dk(DispatchKey.CompositeImplicitAutograd,
   *args)` for each branch of its C++ kernel, with argument shapes,
   contiguity and keyword arguments.
2. Look for upstream's Python body: `op.py_kernels[CompositeImplicitAutograd]`,
   `torch._decomp.decomposition_table[op]`, `torch._refs.<name>`. If one
   exists and its logged sequence equals step 1's, register it with
   `_cia_upstream("aten.x.y", "upstream torch._decomp", "torch._decomp.decompositions",
   "<function>", checked="...")`. If it exists and differs, write a
   composition and say why in its docstring.
3. Otherwise write a composition next to `_cia_linear`:
   `@_cia_composition("aten.x.y", checked="...")`, schema argument names as
   parameters, every op emitted with `_cia_emit("aten.op.overload", ...)`,
   branches in upstream's order, and the measured table in the docstring.
4. Add the cases to `tests/bindings/test_cia_frame.py::CASES` (an expression
   is the argument tuple, or `{"args": (...), "kw": {...}}` for keyword
   arguments), and to `NO_KERNEL` the branches whose body reaches an op this
   shim has no kernel for. The table test fails for a covered op without cases.
5. If the op has a native kernel here it leaves `pending` by itself; eager
   calls with nothing observing keep the native kernel.
6. Nothing in `aten.rs` changes. `cargo check`, then CI.

## 5. Open, in order

* The pending view-like ops that remain (`view_as`, `reshape_as`, `unflatten`,
  `chunk`), which every LLM export goes through.
* `to.other` and `to.device` (CIA upstream, no body).
* The rest of the census `tests/export/test_cia_census.py` prints for the tiny
  SmolLM2 (Llama) and Qwen3 exports, once CI has run it. It stays on the tiny
  configs: no step of `.github/workflows/test.yml` or `.github/scripts/` fetches
  or caches `Qwen/Qwen3-0.6B` or `HuggingFaceTB/SmolLM2-135M` on the hosted
  runners, and `models/test_qwen3.py` and `models/test_stream.py` (which read
  `<repo>/.caches/hf-home` or `$HF_HOME` and skip by name when it is empty)
  therefore skip their real-model tests there, as far as the workflow files show (#377; no run log was read). The cache exists on the
  maintainer's Mac only, so a real-checkpoint census can be run there by hand,
  not in CI.
* Not changed, not measured: the `torch.library` CIA phase
  (`call_python_kernel` "autograd", issue #227) still runs before the
  pre-dispatch stack is consulted, so a custom op with a Python CIA kernel
  is decomposed under export's tracer, where upstream's key order suggests
  the tracer records it.
* Not done in the frame: an inference tensor used outside `inference_mode`
  (upstream's mode sees the op itself; the shim's tensors do not carry the
  property), the deferred-error `grad_fn` export's `autograd_not_implemented`
  attaches, and the Python dispatcher's nesting (the shim counts
  `_EnablePythonDispatcher` and `_DisablePythonDispatcher` guards and treats
  any open disable as off).

## 6. Covered ops (#372, #375, #377)

Measured on torch 2.13.0 (`.caches/spike-venv`), 2026-11-16, under a logging
`TorchDispatchMode` with `op._op_dk(CompositeImplicitAutograd, ...)`. A body is
a torchnative composition where upstream's Python decomposition emits a
different sequence on a measured branch; the same measurement checks each
body, run against upstream's own ops, over 345 cases (every case in
`tests/bindings/test_cia_frame.py::CASES` and `NO_KERNEL`; the shim itself has
not run them, agents compile only). The `to` and attention cases (#377) were
measured on 2026-11-16 as well; the `to` rows also record whether the result
is the first argument itself (`IDENTITY`).

Rule (leader decision 2026-11-16): equality with the op sequence upstream's
CompositeImplicit kernel emits, node for node, ranks above reusing upstream's
Python decomposition. Where upstream's Python body emits a different sequence
on any measured branch, the body here is a torchnative composition, and the
table says where the Python body differs.

| op | source | upstream Python body and where it differs |
|---|---|---|
| `aten.linear.default` | torchnative composition | none exists (section 1) |
| `aten.matmul.default` | torchnative composition | `torch._decomp.decompositions.matmul` equals the kernel on `mm` and the fold cases, but ends the bmm branch with `view` where the kernel logs `_unsafe_view`, and squeezes 1-D x 2-D with `squeeze.dim` where the kernel logs `squeeze_.dim` |
| `aten.reshape.default` | torchnative composition | `torch._refs.reshape` logs `prims.collapse_view` / `prims.reshape` |
| `aten.contiguous.default` | torchnative composition | `torch._refs.contiguous` equals the kernel except `preserve_format` on an already contiguous tensor (kernel returns `self`, ref raises) |
| `aten.narrow.default` | torchnative composition | `torch._refs.narrow` logs `as_strided` |
| `aten.softmax.int` | torchnative composition | `torch._refs.softmax` logs `prims.amax`, `exp`, `sum`, `div` |
| `aten.where.ScalarSelf` | torchnative composition | `torch._refs.where` (the decomposition table's entry) logs one `prims.where` with the scalar bare |
| `aten.__and__.Tensor` | torchnative composition | none exists |
| `aten.diff.default` | torchnative composition | none exists |
| `aten.to.dtype` | torchnative composition | none exists: not in the decomposition table, no `py_kernels` entry; `torch._refs.to` takes `Tensor.to`'s own argument forms and logs `prims.convert_element_type` where the kernel logs `_to_copy` |
| `aten.to.dtype_layout` | torchnative composition | same as `to.dtype` |
| `aten.scaled_dot_product_attention.default` | torchnative composition | none exists: not in the decomposition table, no `py_kernels` entry, no `torch._refs` function |
| `aten._scaled_dot_product_attention_math.default` | torchnative composition | none exists (same three places); CIA itself (measured: `_dispatch_has_kernel_for_dispatch_key` True), so a mode sees its body's ops, never the op |

The branches, as logged (`<x>` is another CIA op, emitted through the door):

* **matmul**: 1-D x 1-D `dot`; 2-D x 1-D `mv`; 1-D x 2-D `unsqueeze, mm,
  squeeze_.dim`; 2-D x 2-D `mm`. Otherwise `should_fold` (it emits
  `transpose.int` of `tensor2.mT` itself when `tensor2` is the larger, even
  when it answers False): never unless the larger operand has 3 or more dims
  and the other 1 or 2; then True when the smaller operand requires grad,
  False when `tensor1` is 2-D, True when the larger operand is empty, else
  True only when its leading n-1 dims are contiguous among themselves (the
  last dim's stride is not looked at). A fold is `[transpose.int,
  t,] <reshape>, mm|mv, _unsafe_view` and, when the right operand was the larger,
  `transpose.int, <contiguous>`. No fold: `expand, <reshape>` per operand (the
  vector right-hand side gets `unsqueeze(2)`), `bmm`, `squeeze.dim(-1)` for a
  vector, `_unsafe_view`. Two 3-D operands with different batch sizes, one of
  them 1, squeeze that batch first (`squeeze.dim`, regardless of
  `requires_grad` in 2.13.0) and recurse. A 0-D operand raises "both
  arguments to matmul need to be at least 1D, but they are 0D and 1D".
* **reshape**: `view.default(self, shape as given)` (a `-1` stays) when
  `computeStride` finds a view; otherwise `clone(memory_format=contiguous_format),
  _unsafe_view(clone, inferred shape)`. Errors from `infer_size_dv` first.
* **contiguous**: nothing when already contiguous in the format (including
  `preserve_format`); otherwise `clone(memory_format=contiguous_format)`, and
  `preserve_format` raises "preserve memory format is unsupported by the
  contiguous operator". A format whose rank does not fit raises first. This
  shim keeps no channels-last strides, so only `contiguous_format` is held
  to upstream's sequence.
* **narrow**: `slice.Tensor(self, dim as given, start wrapped, start + length)`
  after upstream's checks (0-D, length, dim, start range, end range).
* **softmax.int**: `_softmax(self, dim, False)`, after `_to_copy(dtype=,
  layout=strided, device=)` when `dtype` differs from the input's (`to.dtype`
  decomposes to that, so it is emitted directly). CUDA half input with
  `dtype=float32` is `_softmax(self, dim, True)` upstream, not measurable
  here, so the body refuses it by name.
* **where.ScalarSelf**: `scalar_tensor(scalar, dtype=promoted, layout=strided,
  device=other.device)`, `_to_copy(other, dtype=promoted)` when `other` differs,
  `where.self(condition, scalar tensor, other)`.
* **`__and__.Tensor`**: one `bitwise_and.Tensor`.
* **diff**: 0-D raises; negative `n` raises; with `prepend` or `append` the dim
  and their rank and sizes are checked, then `n == 0` is `zeros_like, copy_`
  of `self`; otherwise `cat(pieces, dim as given)` (no `dim` argument when it
  is 0) and `min(n, size)` rounds of `slice.Tensor` (start 1, then start 0),
  `sub.Tensor` (`logical_xor` for bool).
* **to.dtype / to.dtype_layout**: `self` itself, with nothing logged, when
  dtype, layout and device are None or equal, `copy` is False, and the memory
  format is None, `preserve_format` or the one `self` suggests (`pin_memory`
  and `non_blocking` take no part). Otherwise one `_to_copy`: for `to.dtype`
  `dtype=` always, `non_blocking=True` only when True, `memory_format=` only
  when given; for `to.dtype_layout` `dtype`, `layout`, `device`, `pin_memory`
  each when not None (`pin_memory=False` is forwarded), then `non_blocking` and
  `memory_format` as above. `copy` is never forwarded (`to.dtype(x, same dtype,
  copy=True)` logs `_to_copy(dtype=)`). A string `device` is parsed and
  forwarded as a `torch.device`. "The one `self` suggests" is
  `suggest_memory_format()` (not strict): channels-last for a 4-D tensor
  whose strides are like channels-last (5-D: channels-last-3d), contiguous for
  everything else, a 3-D tensor with channels-last-like strides and the
  ambiguous N111 strides included; it reads strides, not density (a sliced
  channels-last tensor still suggests channels-last), so the body transcribes
  `is_channels_last_strides_2d_s4` / `_3d_s5` and the cases build the tensor
  with `permute`. `to.other` and `to.device` are CIA upstream too and have no
  body here (not asked for, not measured).
* **scaled_dot_product_attention**: in order, `validate_sdpa_input` (same
  dtype, same device, every rank at least 2, mask dtype bool, float32 or the
  query's; upstream's messages, doubled spaces included); when any of the three
  has no elements `zeros(q.shape[:-1] + [Ev])`, `sum` of q, k, v with two
  `add.Tensor`, `mul.Scalar(0)` and `add.Tensor` (no backend is called); the
  backend choice (below; a 1-D mask raises `IndexError` there); a bool mask
  becomes additive (`scalar_tensor(-inf, dtype=, device=)` without a layout,
  then `where.ScalarSelf(mask, 0.0, that)`, which decomposes to
  `scalar_tensor(0.0, dtype=, layout=, device=)` and `where.self`); then flash
  or math. The flash call is `_scaled_dot_product_flash_attention_for_cpu(q, k,
  v[, dropout_p, is_causal], [attn_mask=], [scale=])`, positional `dropout_p`
  and `is_causal` only when `is_causal`, first result returned. Neither
  `_fused_sdp_choice` nor the math op is ever logged as itself.
* **Flash is chosen** (CPU, 2.13.0) when `dropout_p > 0` is False (a negative
  value stays on flash and the kernel refuses it), all three are 4-D, batch
  sizes are equal, head counts are equal (with `enable_gqa`: key and value
  heads equal and dividing the query's), the mask is absent or 2-D or 4-D,
  broadcastable onto (batch, heads, L, S) and does not require grad, the head
  dim (last size) is equal for all three, the last-dim strides are 1, and the
  dtype is float32, float64, float16 or bfloat16. Everything else is math
  (dropout, 3-D and 2-D and 5-D inputs, one key head without `enable_gqa`, a
  3-D or 5-D mask, a head dim that differs from the value's, a transposed head
  dim, int inputs which then fail inside the kernels).
* **math op** (`_scaled_dot_product_attention_math`, returns `(output,
  attention weights)`): half and bfloat16 inputs are widened with three
  `to.dtype` (`_to_copy(dtype=float32)`), `mul.Scalar(q, sqrt(scale))` (the
  default scale is `1 / sqrt(head dim)`, inf for a zero head dim; an explicit
  scale is `abs`ed and rooted, a negative one negates the query's factor
  only), `is_causal` raises when a mask is given ("Explicit attn_mask should not
  be set when is_causal=True", after the `mul.Scalar`), else `ones([L, S],
  dtype=bool, layout=, device=)`, `tril`, the additive mask; `enable_gqa`
  raises ("Number of heads in key and value must divide the number of heads in ",
  ending mid-sentence as upstream's does) or repeats key then value
  (`unsqueeze(heads dim + 1)`, `expand`, `clone(contiguous_format)`, `view`,
  nothing when the counts are equal); `transpose.int(key, -2, -1)`,
  `mul.Scalar(key_t, sqrt(abs(scale)))`, `matmul`, `add.Tensor` of the mask (a
  bool mask is not converted here: the attention op does it), `_safe_softmax(-1)`,
  dropout when `dropout_p > 0` (`torch.dropout(p, True)`'s own decomposition,
  `empty_like, bernoulli_.float(1 - p), div_.Scalar(1 - p), mul.Tensor`;
  there is no train or eval distinction at this level, only the number), `matmul`,
  and for half inputs two `to.dtype` back (output, then weights). `matmul`
  is the op above (decomposed through the door). With `dropout_mask` (testing
  only) it is `logical_not`, `masked_fill.Scalar(attn, not mask, 0.0)` and
  `mul.Scalar(value, 1 / (1 - p))`.

### 6.1 Walls that are not the frame's

* The `*_copy` variants of the views (#306, CI run 43793269977) are not CIA
  bodies: upstream registers them at `CompositeExplicitAutogradNonFunctional`,
  so they are kernels in the door (`aten.rs::VIEW_COPY_OPS`), nineteen of
  them. ExecuTorch's `to_edge` reaches them through
  `ReplaceViewOpsWithViewCopyOpsPass`, not through a decomposition
  (docs/design/FUNCTIONALIZATION.md section 5.1). Written and compiled, not run.

* `dot`, `mv`, `squeeze_.dim` and `logical_xor` have no kernel in this shim
  (aten.rs `IMPLEMENTED`), so matmul with a 1-D operand on the `dot`/`mv`/
  `squeeze_` branches, a fold with a vector, linear with a 1-D input and no
  bias, and bool `diff` refuse naming the op
  (`test_branches_without_a_shim_kernel_refuse_by_name`).
* Attention (#377) behaviour changes in plain eager, all of them upstream's
  own: `F.scaled_dot_product_attention` enters the door as
  `aten.scaled_dot_product_attention.default` (there is no native kernel for
  that name; the flash kernel is a different op) and the body runs, so a 3-D,
  2-D or 5-D query is no longer refused (it takes the math sequence), one key
  head without `enable_gqa` takes math instead of flash (upstream does), a
  3-D mask takes math, the error texts are upstream's, and the bool-mask
  `scalar_tensor` calls carry upstream's keyword arguments. The numbers the
  flash kernel returns for the cases that stay on flash are not touched.
  Per-call Python overhead of the checks is not measured.
* Not reproduced for attention: the extra `detach` upstream logs after the
  flash call when an input requires grad (autograd's saved output; this shim
  has no node), a value sequence longer than the key's (upstream's flash accepts
  it, measured; the shim's flash kernel is not asked to), the device-type
  message for mixed devices (measured: cpu and meta), nested tensors, and the
  runtime switch that disables flash. Mask-mismatch errors (a mask that does
  not broadcast) are raised by `add.Tensor`, whose message carries the op's
  name in this shim, so they are not cases.
* Not reproduced for `to`: `layout=sparse_coo` (upstream's `_to_copy` raises
  its own text, the shim's names the op), `channels_last` requested of
  `_to_copy` (the shim's kernel refuses it by name; the alias test and the
  suggested format are held to upstream with strides built by `permute`),
  `cpu:0` against a `cpu` tensor (not asked of the shim), and the copy's own
  strides under `preserve_format` (kernel, not body).
* Not measured, so not claimed: `softmax.int` on half and bfloat16 inputs
  (shim kernel support), complex `where.ScalarSelf` scalars, CUDA, symbolic
  sizes, and the public spellings (`torch.matmul`, `Tensor.reshape`, ...),
  which `PUBLIC` does not list for these ops.
* The linear exception the frame test used to pin (an undecomposed
  `aten.matmul.default` in a logging mode) is gone: matmul has a body, and the
  four linear cases that emitted it are in `CASES`.

### 6.2 Upstream's own sequence is compiler-dependent (CI run 43802210944)

Agent proposal, not confirmed. The bodies were measured on macOS (clang). In
CI run 43802210944 Linux upstream (the manylinux wheel, gcc) emitted 104 of the
777 `CASES` rows in a different order from macOS upstream, and macOS has none
of them:

* 58 "op names" rows, each the same multiset of op names, in three patterns:
  `_to_copy` of the attention weights before the second `matmul` rather than
  after it (31 rows, `_scaled_dot_product_attention_math` and the attention
  op through it), from `std::make_tuple(at::matmul(attn, value).to(t),
  attn.to(t))`; and `sum, sum, sum, add, add` rather than `sum, sum, add,
  sum, add` (27 rows, the zero-size attention path), from `(q.sum() +
  k.sum()) + v.sum()`.
* 46 `diff` rows (`call` and `op_dk` of every case) whose first `slice` is the
  second narrow (`start 0`) rather than the first (`start 1`), from
  `at::sub(at::narrow(self, dim, 1, n), at::narrow(self, dim, 0, n))`.

In every pattern the moved ops are sibling arguments of one C++ call: C++
leaves the evaluation order of function arguments unspecified, gcc evaluates
them right to left and clang left to right. Measured from the log rows, not
from upstream's C++ source, which is not in this tree. So the bodies stay as
they are, and `test_each_body_emits_upstreams_sequence` accepts, off macOS
only, a row that equals upstream's up to a reordering that respects the data
flow its probe records (tensor object identity: each step's producers; a step
that writes an argument is a barrier nothing crosses), and prints how many
rows needed it (`CIAFRAME on <platform>: N of M rows ...`). On macOS every row
must still equal upstream's exactly. The same test unifies C++ type-name
spellings in two raised messages off macOS (`long long` is clang's name for
`int64_t` on macOS; the Linux wheel's text differed and the log did not show
it). Not yet run.
