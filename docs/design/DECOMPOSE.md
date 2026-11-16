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
* **`to.dtype`** (no native kernel; Fake decomposes it inside `addmm`'s
  Python decomposition).
* The rest of the census `tests/export/test_cia_census.py` prints for the tiny
  SmolLM2 (Llama) and Qwen3 exports, once CI has run it.
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

## 6. Covered ops (#372, #375)

Measured on torch 2.13.0 (`.caches/spike-venv`), 2026-11-16, under a logging
`TorchDispatchMode` with `op._op_dk(CompositeImplicitAutograd, ...)`. A body is
a torchnative composition where upstream's Python decomposition emits a
different sequence on a measured branch; the same measurement checks each
body, run against upstream's own ops, over 168 cases (every case in
`tests/bindings/test_cia_frame.py::CASES` and `NO_KERNEL`; the shim itself has
not run them, agents compile only).

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

### 6.1 Walls that are not the frame's

* `dot`, `mv`, `squeeze_.dim` and `logical_xor` have no kernel in this shim
  (aten.rs `IMPLEMENTED`), so matmul with a 1-D operand on the `dot`/`mv`/
  `squeeze_` branches, a fold with a vector, linear with a 1-D input and no
  bias, and bool `diff` refuse naming the op
  (`test_branches_without_a_shim_kernel_refuse_by_name`).
* Not measured, so not claimed: `softmax.int` on half and bfloat16 inputs
  (shim kernel support), complex `where.ScalarSelf` scalars, CUDA, symbolic
  sizes, and the public spellings (`torch.matmul`, `Tensor.reshape`, ...),
  which `PUBLIC` does not list for these ops.
* The linear exception the frame test used to pin (an undecomposed
  `aten.matmul.default` in a logging mode) is gone: matmul has a body, and the
  four linear cases that emitted it are in `CASES`.
