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
* `torch.export.export(Linear(8, 4) + ReLU)` records `linear, relu`; its
  `run_decompositions()` gives `permute, addmm, relu`. During it
  `FunctionalTensorMode` receives `linear, t, addmm, relu` twice (metadata
  pass, then the trace) and `OpOverload.decompose` decomposes `linear` both
  times (through export's `_special_op_to_decompose_cia`, which calls
  `_op_dk(CompositeImplicitAutograd)`), while `t` and `relu` answer
  NotImplemented. Proxy then turns `t` into `transpose.int` and `permute`.
* Export preserves a CIA op only through `_override_composite_implicit_decomp`
  (torch/export/exported_program.py): for every op `_collect_all_valid_cia_ops`
  finds, which asks `_dispatch_has_kernel_for_dispatch_key(op, CIA)`, it
  registers Python kernels at the autograd keys (`autograd_not_implemented`,
  which calls the op below autograd), at CIA, and at the backend keys, and a
  fake rule that runs the original CIA through `_op_dk`. Those take effect
  under the Python dispatcher, which export's tracing enables.
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
  (below autograd or in inference mode; under the Python dispatcher with an
  autograd kernel registered from Python, which is export preserving the op),
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
`aten.linear.default` instead of composing `t` + `addmm` itself. That is why
the shim's export now records `linear` as upstream's does: before, no mode
ever saw it.

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
4. Add the cases to `tests/bindings/test_cia_frame.py::CASES` (and to
   `MATMUL_BRANCH`, or a group like it, for branches that emit another
   uncovered CIA op). The table test fails for a covered op without cases.
5. If the op has a native kernel here it leaves `pending` by itself; eager
   calls with nothing observing keep the native kernel.
6. Nothing in `aten.rs` changes. `cargo check`, then CI.

## 5. Open, in order

* **`matmul`** (CIA upstream, pending here; upstream's Python body is
  `torch._decomp.decompositions.matmul`, its `py_kernels` entry). Linear's
  no-bias and non-contiguous branches emit it, so a logging mode sees `matmul`
  where upstream's sees `mm`/`bmm`; `FunctionalTensorMode`, Proxy and Fake
  already decompose it through that `py_kernels` entry.
* **`reshape`** and the other pending view-like ops (`view_as`, `reshape_as`,
  `contiguous`, `unflatten`, `chunk`), which every LLM export goes through.
* **`to.dtype`** (no native kernel; Fake decomposes it inside `addmm`'s
  Python decomposition).
* The rest of the census `tests/export/test_cia_census.py` prints for the tiny
  SmolLM2 (Llama) and Qwen3 exports, once CI has run it.
* Not done in the frame: an inference tensor used outside `inference_mode`
  (upstream's mode sees the op itself; the shim's tensors do not carry the
  property), the deferred-error `grad_fn` export's `autograd_not_implemented`
  attaches, and the Python dispatcher's nesting (the shim counts
  `_EnablePythonDispatcher` and `_DisablePythonDispatcher` guards and treats
  any open disable as off).
