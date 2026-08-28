//! The aten dispatch choke point.
//!
//! DESIGN.md §6 rejects enumerating the op set up front and makes the shim its
//! own instrument instead: every aten call goes through one entry, and anything
//! missing raises with its own name in the message. Running a model then prints
//! the work queue in frequency order, which is better ordering information than
//! a list written in advance.
//!
//! That only holds if there is exactly one entrance. Hence no arithmetic on
//! `TensorBase` -- a convenience method would be a second door that the
//! instrument cannot see through.
//!
//! Overloads are part of the key (`aten.add.Tensor`, not `aten.add`). torch
//! resolves overloads before it reaches the kernel, and folding them together
//! here would make `add.Tensor` and `add.Scalar` -- different schemas -- look
//! like one implemented op.
use candle_core::{Device, Tensor};
use pyo3::prelude::*;
use pyo3::PyErr;
use pyo3::types::{PyDict, PyList, PyModule, PyTuple};
use pyo3::IntoPyObjectExt;

use crate::device::PyDevice;
use crate::dtype::{default_float, PyDtype, TorchDType};
use crate::err::{aten_not_implemented, candle_err, not_implemented};
use crate::tensor::PyTensorBase;

/// Every op with a real kernel behind it. Kept sorted; `_aten_implemented()`
/// hands it to Python so the vendored layer and the tests can ask rather than
/// keep their own copy of the list.
///
/// The `TensorBase` surface (docs/TENSORBASE.md) is interleaved here rather
/// than kept in a block of its own. `_aten_implemented()` means exactly one
/// thing -- "this op has a kernel *and* `tools/golden/cases.py` compares it
/// against upstream" -- and which Python spelling reaches an op is not part of
/// that meaning.
pub const IMPLEMENTED: &[&str] = &[
    "aten._local_scalar_dense.default",
    "aten._safe_softmax.default",
    "aten._scaled_dot_product_flash_attention_for_cpu.default",
    "aten._softmax.default",
    "aten._to_copy.default",
    "aten._unsafe_view.default",
    "aten.abs.default",
    "aten.add.Tensor",
    "aten.add_.Tensor",
    "aten.addmm.default",
    "aten.alias.default",
    "aten.any.default",
    "aten.any.dim",
    "aten.arange.default",
    "aten.arange.start",
    "aten.arange.start_step",
    "aten.argmax.default",
    "aten.baddbmm.default",
    "aten.bitwise_and.Scalar",
    "aten.bitwise_and.Tensor",
    "aten.bitwise_not.default",
    "aten.bitwise_or.Scalar",
    "aten.bitwise_or.Tensor",
    "aten.bmm.default",
    "aten.cat.default",
    "aten.ceil.default",
    "aten.clamp_.default",
    "aten.clone.default",
    "aten.convolution.default",
    "aten.copy_.default",
    "aten.cos.default",
    "aten.cumsum.default",
    "aten.detach.default",
    "aten.div.Tensor",
    "aten.div_.Tensor",
    "aten.embedding.default",
    "aten.empty.memory_format",
    "aten.empty_like.default",
    "aten.eq.Scalar",
    "aten.eq.Tensor",
    "aten.exp.default",
    "aten.expand.default",
    "aten.fill_.Scalar",
    "aten.fill_.Tensor",
    "aten.floor_divide.default",
    "aten.full.default",
    "aten.gather.default",
    "aten.ge.Scalar",
    "aten.gelu.default",
    "aten.gt.Scalar",
    "aten.gt.Tensor",
    "aten.histc.default",
    "aten.index.Tensor",
    "aten.index_put_.default",
    "aten.is_floating_point.default",
    "aten.isin.Tensor_Tensor",
    "aten.le.Scalar",
    "aten.le.Tensor",
    "aten.lift_fresh.default",
    "aten.lt.Scalar",
    "aten.lt.Tensor",
    "aten.masked_fill.Scalar",
    "aten.masked_fill_.Scalar",
    "aten.masked_select.default",
    "aten.max.default",
    "aten.max.dim",
    "aten.mean.default",
    "aten.mean.dim",
    "aten.min.default",
    "aten.mm.default",
    "aten.mul.Scalar",
    "aten.mul.Tensor",
    "aten.multinomial.default",
    "aten.native_layer_norm.default",
    "aten.ne.Scalar",
    "aten.ne.Tensor",
    "aten.neg.default",
    "aten.new_ones.default",
    "aten.normal_.default",
    "aten.ones.default",
    "aten.permute.default",
    "aten.pow.Scalar",
    "aten.pow.Tensor_Scalar",
    "aten.pow.Tensor_Tensor",
    "aten.randint.low",
    "aten.reciprocal.default",
    "aten.relu.default",
    "aten.relu_.default",
    "aten.rsqrt.default",
    "aten.rsub.Scalar",
    "aten.scalar_tensor.default",
    "aten.scatter.src",
    "aten.select.int",
    "aten.silu.default",
    "aten.sin.default",
    "aten.slice.Tensor",
    "aten.softplus.default",
    "aten.sort.default",
    "aten.split.Tensor",
    "aten.split_with_sizes.default",
    "aten.squeeze.dim",
    "aten.stack.default",
    "aten.sub.Tensor",
    "aten.sum.default",
    "aten.sum.dim_IntList",
    "aten.t.default",
    "aten.tanh.default",
    "aten.topk.default",
    "aten.transpose.int",
    "aten.unbind.int",
    "aten.uniform_.default",
    "aten.unsqueeze.default",
    "aten.view.default",
    "aten.view.dtype",
    "aten.where.ScalarOther",
    "aten.where.self",
    "aten.zero_.default",
    "aten.zeros_like.default",
];

/// Ops with a real kernel that `_aten_implemented()` does **not** advertise.
///
/// This is a reporting workaround, not a capability one, and it is the only
/// thing in this file that is neither implemented nor refused. The golden
/// harness treats "advertised in `_aten_implemented()` but absent from
/// `tools/golden/cases.py::CASE_BUILDERS`" as a hard failure, on purpose, so
/// that an op cannot be added without being compared against upstream. That
/// rule is right; this task simply was not allowed to edit the harness, and
/// the one op it produced that the harness has no builder for is parked here
/// rather than turned off.
///
/// `_aten_dispatch` reaches these exactly like any other op -- so
/// `torch.randint(10, (2,))` works -- but the coverage number stays honest in
/// the conservative direction (it under-reports rather than over-reports).
/// The fix is one case builder and one line move.
///
/// `aten.mul.Scalar` was the first to get that fix (docs/TAIL.md): a
/// re-measurement of `falcon` under `_aten_all_implemented()` found the
/// kernel already dispatching, so the remaining work was exactly the case
/// builder the comment above describes, plus this one line move.
pub const IMPLEMENTED_AWAITING_GOLDEN: &[&str] = &[
    "aten.add.Scalar",
    "aten.any.dims",
    "aten.contiguous.default",
    "aten.div.Scalar",
    "aten.masked_fill.Tensor",
    "aten.matmul.default",
    "aten.max.other",
    "aten.randint.default",
    "aten.reshape.default",
    "aten.sub.Scalar",
    "aten.zeros.default",
];

/// Everything `_aten_dispatch` answers, whether or not it is advertised: the
/// two lists above, unioned. A function rather than a third constant, so that
/// the smoke tests can check the dispatch table against it without anything
/// being able to read the union and report it as golden-compared coverage.
pub fn all_implemented() -> Vec<&'static str> {
    let mut out: Vec<&'static str> = IMPLEMENTED
        .iter()
        .chain(IMPLEMENTED_AWAITING_GOLDEN.iter())
        .copied()
        .collect();
    out.sort_unstable();
    out
}

// `set_default_dtype` arrived, so the default float dtype is no longer a
// constant here -- it is `dtype::default_float()`, a process-global. The
// dtype-inference rules below ("integral arguments give int64, anything else
// gives the default float") call it rather than reading a copy, which is the
// whole of what makes the setter load-bearing rather than decorative.

/// The single entrance. `torch.ops.aten.<op>.<overload>(...)` is expected to
/// land here once the Python layer is vendored.
#[pyfunction]
#[pyo3(name = "_aten_dispatch", signature = (op, *args, **kwargs))]
pub fn aten_dispatch(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    // One scan, two answers: that the tensor arguments agree about their
    // device, and *which* device that is. Making meta a second pass would have
    // paid for a scan twice at the hottest line in the crate; merged, meta
    // costs the dispatcher a discriminant test on a value the gate already had
    // in hand.
    let out = match check_devices_agree(op, args, kwargs)? {
        Some(Where::Meta) => meta_dispatch(py, op, args, kwargs)?,
        _ => aten_dispatch_inner(py, op, args, kwargs)?,
    };
    // One exit as well as one entrance: every tensor leaving the dispatcher
    // wears the registered Python tensor class (`tensor::promote`). Doing it
    // here rather than in each kernel means a kernel can keep returning the
    // native type and cannot forget.
    let out = crate::tensor::promote(py, out)?;
    // The graph capture hook (docs/CAPTURE.md). It is here rather than in
    // `aten_dispatch_inner` for two reasons: the meta path has to be recorded
    // too, and the identity the recorder registers has to be the object
    // *Python will hold*, which is `promote`'s result rather than the kernel's.
    //
    // When capture is off -- which is always, unless something asked for it --
    // this is one relaxed atomic load and a branch that is not taken. That is
    // the entire cost added to the single door, measured in docs/CAPTURE.md §7.
    if crate::capture::is_active() {
        crate::capture::record(py, op, args, kwargs, &out);
    }
    Ok(out)
}

/// Where a dispatched tensor argument lives, comparable without allocating.
///
/// Not `PyDevice`: a label costs a `String` per argument and that was measured
/// at +78 ns per dispatch (docs/DEVICE_ABS.md §6). Not `candle_core::Device`
/// either, because **a meta tensor has no candle handle** -- that is the whole
/// content of `Repr::Meta`. So the gate compares this, which is a discriminant
/// test in both arms, and builds a label only on the path that is about to
/// raise.
#[derive(Clone)]
enum Where {
    Dense(Device),
    Meta,
}

impl Where {
    fn of(tensor: &PyTensorBase) -> Self {
        match tensor.repr() {
            crate::tensor::Repr::Dense(inner) => Where::Dense(inner.device().clone()),
            crate::tensor::Repr::Meta { .. } => Where::Meta,
        }
    }

    fn label(&self) -> PyDevice {
        match self {
            Where::Dense(device) => PyDevice::from_candle(device),
            Where::Meta => PyDevice::meta(),
        }
    }
}

/// Refuse an op whose tensor arguments are not all on one device.
///
/// **Where this lives is the design decision, not whether it exists.**
/// Upstream has no such gate: the check is inside each kernel, which is why the
/// message it gives depends on which kernel you hit -- `a + m` says *"Expected
/// all tensors to be on the same device, but found at least two devices, mps:0
/// and cpu!"*, `torch.mm` says *"Tensor for argument #1 'mat1' is on CPU, but
/// expected it to be on GPU"*, and `torch.cat([cpu, mps])` **segfaults** the
/// process on torch 2.13.0 (measured on this host, docs/DEVICE_ABS.md §6). A
/// per-kernel check is a check every new kernel has to remember; putting it at
/// the single door means no kernel can forget it, and this shim has a single
/// door precisely so that things like this have somewhere to go.
///
/// The message is upstream's most common one, verbatim, so that code matching
/// on it keeps working.
///
/// **The rejecting half of this gate is now reachable, and `meta` is what
/// reached it.** Until there was a second device it could not fire at all --
/// `resolve()` refuses every non-CPU label, so every tensor was on the CPU and
/// the loop always found one device. That was recorded as an untested branch in
/// docs/DEVICE_ABS.md §10. `meta` needs no backend, so `cpu + meta` is an input
/// this build can actually construct, and docs/META.md §5 is the measurement
/// that it raises.
///
/// **The comparison is on candle's handles, not on reconstructed labels, and
/// that is a measured decision.** Building a `PyDevice` per tensor argument
/// costs a `String` allocation each, and it showed: an interleaved A/B of two
/// artefacts differing only in the call above measured **+78 ns per dispatch**
/// for `add.Tensor` (346 -> 424 ns) and **+86 ns** for `cat.default`
/// (392 -> 479 ns) -- +22% on the cheapest op this door has. Comparing `Device`
/// directly is an enum discriminant test; a label is built only on the failing
/// path, where an allocation is free next to raising. docs/DEVICE_ABS.md §6 has
/// the numbers from both versions. `Where` keeps that property while making
/// room for a tensor with no handle at all.
///
/// Returns the device the arguments agreed on, or `None` when there were no
/// tensor arguments at all (every factory call). The dispatcher needs that
/// answer anyway to route meta, so returning it here is one scan instead of two.
fn check_devices_agree(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Option<Where>> {
    let mut first: Option<Where> = None;

    for value in args.iter() {
        scan_for_device(op, &mut first, &value)?;
    }
    // **The keyword loop descends into sequences too, and it did not used to.**
    // That was a hole rather than an economy, and it was invisible for exactly
    // as long as the gate could not fire: `_torch_level_function` binds every
    // argument by name before dispatching, so a `torch.cat([a, b])` arrives
    // here as `kwargs["tensors"] = [a, b]` and **never touches the positional
    // loop at all**. With one device the two loops could not be told apart.
    // The first `torch.cat([cpu_tensor, meta_tensor])` after meta landed went
    // straight past the gate and died inside the kernel on
    // `Cannot copy out of meta tensor; no data!` -- the right refusal by
    // accident, from the wrong place, and it would have been a wrong *answer*
    // for two backends that could both actually compute. docs/META.md §5.
    if let Some(kwargs) = kwargs {
        for (_, value) in kwargs.iter() {
            scan_for_device(op, &mut first, &value)?;
        }
    }
    Ok(first)
}

/// One dispatched argument, and the sequences one level under it.
///
/// Tensor first, sequence second, and the order is the measurement: almost
/// every argument that is anything is a tensor, so trying the list and tuple
/// casts first paid for two failed type checks on the hot path. `cat`/`stack`
/// take `Tensor[]`, so the sequence branch cannot be dropped -- it is exactly
/// the ops most likely to mix devices that hide their tensors one level down.
///
/// A free function rather than a closure, and that too is a measurement: the
/// first version of this was a closure calling another closure, both capturing
/// `first` mutably, and the inner call did not inline. Taking `first` as a
/// parameter costs nothing and lets both collapse into the caller.
#[inline]
fn scan_for_device(
    op: &str,
    first: &mut Option<Where>,
    value: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if visit_for_device(op, first, value)? {
        return Ok(());
    }
    if let Ok(sequence) = value.cast::<PyList>() {
        for item in sequence.iter() {
            visit_for_device(op, first, &item)?;
        }
    } else if let Ok(sequence) = value.cast::<PyTuple>() {
        for item in sequence.iter() {
            visit_for_device(op, first, &item)?;
        }
    }
    Ok(())
}

/// One value. Returns whether it *was* a tensor, so the caller can skip the
/// sequence casts for the overwhelmingly common case.
///
/// The agreement test reads the representation in place instead of building a
/// `Where` per argument. Constructing one clones a `candle_core::Device`, and
/// the version that did so on every argument (rather than only on the first)
/// showed up in the A/B -- see docs/META.md §9. `Where` is built once, for the
/// first tensor, and again only on the path that is about to raise.
#[inline]
fn visit_for_device(
    op: &str,
    first: &mut Option<Where>,
    value: &Bound<'_, PyAny>,
) -> PyResult<bool> {
    let Ok(tensor) = value.cast::<PyTensorBase>() else {
        return Ok(false);
    };
    let borrowed = tensor.borrow();
    let agrees = match (&*first, borrowed.repr()) {
        (None, _) => {
            *first = Some(Where::of(&borrowed));
            return Ok(true);
        }
        (Some(Where::Dense(seen)), crate::tensor::Repr::Dense(inner)) => {
            seen.same_device(inner.device())
        }
        (Some(Where::Meta), crate::tensor::Repr::Meta { .. }) => true,
        _ => false,
    };
    if agrees {
        return Ok(true);
    }
    // `copy_` is the one op upstream lets cross devices, because transferring
    // *is* its definition (docs/DEVICE_ABS.md §3.4, measured: `cpu.copy_(mps)`
    // returns a cpu tensor). Checked only here, on the path that was about to
    // raise, so agreeing arguments pay nothing for it.
    if op == "aten.copy_.default" {
        return Ok(true);
    }
    let seen = first.as_ref().expect("a disagreement needs a first").label();
    Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
        "Expected all tensors to be on the same device, but found at least two \
         devices, {} and {}! ({op} in torch._C shim)",
        seen.__str__(),
        Where::of(&borrowed).label().__str__(),
    )))
}

/// The `meta` half of the dispatcher: ops whose inputs have no storage.
///
/// **Structured as a separate table rather than a branch inside each kernel,
/// because that is what upstream is.** torch has a `Meta` dispatch key with its
/// own registrations (`torch/_meta_registrations.py`); a meta kernel computes
/// shape and dtype and never touches bytes. Mirroring the shape here means the
/// dense kernels stay unaware of meta -- `PyTensorBase::tensor()` refuses them
/// if they ever try -- and it means the answer to "does this op work on meta?"
/// is a list rather than 96 separate readings.
///
/// **The list is short and honest about it.** Shape inference for the other
/// ninety-odd ops is a real body of work (upstream's file is thousands of
/// lines), it has no measured caller here yet, and guessing a shape rule is
/// exactly the kind of silent wrongness this shim refuses. So everything not
/// named below raises with its own name in the message, which is DESIGN.md §6's
/// instrument doing its job: run a model under `with torch.device("meta")` and
/// the work queue prints itself in frequency order.
///
/// `_aten_implemented()` is *not* extended by anything here. That constant
/// means "has a kernel and `tools/golden/cases.py` compares it against
/// upstream", and the golden harness compares values -- which a meta tensor by
/// definition has none of. Meta support is a property of ops already on the
/// list, so op coverage stays 96 and the evidence lives in
/// `pytests/test_shim.py` instead. docs/META.md §7.
fn meta_dispatch(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    match op {
        "aten._to_copy.default" => meta_to_copy(py, args, kwargs),
        "aten.copy_.default" => meta_copy_inplace(py, args, kwargs),
        // Metadata pass-throughs: upstream's kernels for these return a tensor
        // with the same shape and dtype and (for `detach`/`alias`) share
        // storage, which meta has none of. This shim's dense `detach`/`alias`
        // already copy rather than alias (docs/OPS4.md §8), so meta is not
        // losing an aliasing property it otherwise had.
        "aten.detach.default" | "aten.alias.default" | "aten.clone.default"
        | "aten.contiguous.default" | "aten.lift_fresh.default" => {
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            meta_result(py, input.dims().to_vec(), input.tag())
        }
        // **In-place initialisers: no-ops that return the receiver.** These are
        // not a convenience -- `nn.Linear.reset_parameters` runs
        // `init.kaiming_uniform_(self.weight)` in every `__init__`, so
        // `with torch.device("meta"): nn.Linear(4, 8)` stops on `uniform_`
        // before it can produce a single parameter, which is the exact call
        // `accelerate.init_empty_weights` is built around.
        //
        // "Write nothing and return self" is upstream's meta kernel for them
        // too, and it is the only honest answer: filling a tensor that holds no
        // bytes has no observable effect, and the shape it advertises is
        // unchanged by what would have been written. The refusal that matters
        // -- reading the values back -- is still `tolist`/`item`'s, one layer
        // out, so nothing here lets a caller mistake an uninitialised meta
        // parameter for an initialised one.
        //
        // `add_`/`mul_` are deliberately *not* here even though they are also
        // in-place: they broadcast, so their meta kernel has a shape rule to
        // get right, and a no-op would silently accept a shape upstream
        // rejects.
        "aten.uniform_.default"
        | "aten.normal_.default"
        | "aten.zero_.default"
        | "aten.fill_.Scalar" => {
            let receiver = tensor_receiver(op, args, kwargs)?;
            Ok(receiver.into_any().unbind())
        }
        // `aten::is_floating_point` reads the dtype tag, which meta carries in
        // full. Nothing about it needs storage.
        "aten.is_floating_point.default" => {
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            Ok(input.tag().is_floating_point().into_bound_py_any(py)?.unbind())
        }
        // `.item()`, `bool(t)`, `float(t)`. Upstream refuses this one with a
        // *different* message from the copy-out family, and the difference is
        // measured, not tidied: `torch.zeros(1, device="meta").item()` is
        // `RuntimeError: Tensor.item() cannot be called on meta tensors` while
        // `.tolist()` on the same tensor is `NotImplementedError: Cannot copy
        // out of meta tensor; no data!`.
        "aten._local_scalar_dense.default" => Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Tensor.item() cannot be called on meta tensors",
        )),
        // `new_ones` takes its shape from the argument and its device from the
        // input tensor, so on a meta input it is a meta factory.
        "aten.new_ones.default" => {
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            let size: Vec<usize> = required(op, args, kwargs, 1, "size")?.extract()?;
            let tag = dtype_arg(args, kwargs, 2, "dtype")?.unwrap_or(input.tag());
            match device_arg_or_label(args, kwargs, 4, "device", &PyDevice::meta())? {
                label if label.is_meta() => meta_result(py, size, tag),
                label => {
                    let device = label.resolve()?;
                    let storage = PyDtype::new(tag).storage(op)?;
                    let out = Tensor::ones(size, storage, &device)
                        .map_err(|e| candle_err(op, e))?;
                    finish(py, out, tag)
                }
            }
        }
        // `aten::empty_like` on a meta input -- the op that carries a model off
        // the meta device.
        //
        // `from_pretrained` builds the whole module tree under
        // `init_empty_weights`, so every parameter and buffer starts on meta.
        // Anything the checkpoint did not supply -- missing keys, and every
        // non-persistent buffer, which is never in a checkpoint by definition
        // -- is brought across by `torch.empty_like(param, device=...)` in
        // `transformers/modeling_utils.py:4763,4771`. Without this the load
        // reads the whole file and then stops one step before the model is
        // usable, which is exactly where docs/CKPT2.md §3 found it.
        //
        // Shape, dtype and device rules are the dense kernel's, restated with
        // the same helpers in the same argument order, for the reason
        // docs/E2E_REAL.md §6.1 gives: a meta kernel that promises a different
        // dtype than the dense one hands the caller an allocation the dense
        // kernel then refuses to compute into.
        //
        // Like the dense kernel, "empty" answers zeros. That is safe *here*
        // for a reason worth stating rather than assuming: every value this
        // produces is overwritten before it is read -- missing keys by
        // `_initialize_missing_keys`, non-persistent buffers by the module's
        // own initialisation -- and if one ever were not, the zeros would
        // reach the forward pass, where `pytests/test_shim.py` compares logits
        // against upstream.
        "aten.empty_like.default" => {
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            let tag = dtype_arg(args, kwargs, 1, "dtype")?.unwrap_or(input.tag());
            reject_unsupported(
                op,
                args,
                kwargs,
                &[(2, "layout"), (4, "pin_memory"), (5, "memory_format")],
            )?;
            let shape = input.dims().to_vec();
            match device_arg_or_label(args, kwargs, 3, "device", &PyDevice::meta())? {
                label if label.is_meta() => meta_result(py, shape, tag),
                label => {
                    let device = label.resolve()?;
                    let storage = PyDtype::new(tag).storage(op)?;
                    let out =
                        Tensor::zeros(shape, storage, &device).map_err(|e| candle_err(op, e))?;
                    finish(py, out, tag)
                }
            }
        }
        // `aten::div.Scalar` and `aten::mul.Scalar` -- shape is the input's,
        // dtype is `arith_tag`'s.
        //
        // Both are reached by `LlamaRotaryEmbedding.__init__`
        // (`transformers/models/llama/modeling_llama.py:108`), which
        // `from_pretrained` runs under `init_empty_weights`: the `/ dim`
        // directly, and the `*` that `torch/_tensor.py:1112` turns the leading
        // `1.0 /` into. There is no broadcasting to get right, a `Scalar`
        // overload having only one tensor, so the whole kernel is the
        // promotion -- and the promotion is `arith_tag`'s rather than a
        // restatement of it, so the dtype this advertises is by construction
        // the dtype the dense kernel would produce, refusals included.
        //
        // `add.Scalar` and `sub.Scalar` are the other two members of that
        // helper and are deliberately absent: nothing has reached them on
        // meta.
        "aten.div.Scalar" | "aten.mul.Scalar" => {
            let kind = if op == "aten.div.Scalar" {
                Arith::Div
            } else {
                Arith::Mul
            };
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            let other =
                scalar_arg(op, args, kwargs, 1, "other")?.ok_or_else(|| missing(op, "other"))?;
            let tag = arith_tag(op, kind, input.tag(), Some(!other.is_int()))?;
            meta_result(py, input.dims().to_vec(), tag)
        }
        // `aten::pow.Scalar(Scalar self, Tensor exponent)` -- the next link in
        // the same expression. The scalar is the *base* here, so the shape
        // comes from argument 1, and the dtype is torch's wrapped-number rule
        // (`pow_result_tag`): an integer base leaves an integral exponent
        // integral, a float base floats it.
        "aten.pow.Scalar" => {
            let base = scalar_arg(op, args, kwargs, 0, "self")?.ok_or_else(|| missing(op, "self"))?;
            let exponent = tensor_arg(op, args, kwargs, 1, "exponent")?;
            let tag = pow_result_tag(op, exponent.tag(), !base.is_int())?;
            meta_result(py, exponent.dims().to_vec(), tag)
        }
        // `aten::reciprocal` -- the last link. `torch/_tensor.py:1112` spells
        // `1.0 / t` as `t.reciprocal() * 1.0`, so this is what the rope
        // expression reaches rather than an `rdiv` op.
        //
        // Its dense counterpart is the `unary_float` family, whose rule is
        // "floating in, same out; anything else becomes the default float".
        // The other five members of that family (`cos`, `sin`, `tanh`, `exp`,
        // and `rsqrt`, which shares the rule from its own function) are
        // deliberately *not* listed here: nothing has reached them on meta, so
        // adding them would be a claim no test could have failed on.
        "aten.reciprocal.default" => {
            let input = tensor_arg(op, args, kwargs, 0, "self")?;
            let tag = if input.tag().is_floating_point() {
                input.tag()
            } else {
                default_float()
            };
            meta_result(py, input.dims().to_vec(), tag)
        }
        other => Err(not_implemented(format!(
            "torch._C shim has no meta kernel for {other}. A meta tensor holds shape and \
             dtype and no storage, so this op would have to infer its output shape without \
             computing -- which is a real kernel (upstream registers one in \
             torch/_meta_registrations.py), not a fallthrough. See docs/META.md §7 for the \
             list that is implemented."
        ))),
    }
}

/// A finished meta tensor: shape and dtype, no allocation.
///
/// The meta counterpart of `finish`, and deliberately not folded into it. The
/// two differ in exactly one interesting way -- `finish` has to ask
/// `PyDtype::storage()` whether candle can hold the dtype, and this does not,
/// because nothing is being held.
fn meta_result(py: Python<'_>, shape: Vec<usize>, tag: TorchDType) -> PyResult<Py<PyAny>> {
    Ok(PyTensorBase::meta(shape, tag)
        .into_pyobject(py)?
        .into_any()
        .unbind())
}

/// `aten::_to_copy` with a meta input: the dtype half works, the device half
/// only outwards-to-nowhere.
///
/// Three cases, all measured on torch 2.13.0:
///
/// | call | upstream |
/// |---|---|
/// | `meta.to(torch.float64)` | meta tensor, float64 |
/// | `meta.to("meta")` | the same object (`t.to("meta") is t`) |
/// | `meta.to("cpu")` / `meta.cpu()` | `NotImplementedError: Cannot copy out of meta tensor; no data!` |
///
/// The last one is the whole point of meta and the reason this is not a
/// fallthrough: a shim that quietly produced a zero-filled CPU tensor here
/// would turn "no weights were loaded" into "the weights are all zero", which
/// is the failure `docs/CKPT.md`'s `filled` guard exists to prevent one layer
/// down.
fn meta_to_copy(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten._to_copy.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = dtype_arg(args, kwargs, 1, "dtype")?.unwrap_or(input.tag());
    reject_unsupported(OP, args, kwargs, &[(2, "layout"), (4, "pin_memory")])?;
    reject_memory_format(OP, args, kwargs, 6)?;
    // `device=None` keeps the tensor where it is -- on meta. Same contract as
    // the dense path (docs/DEVICE_ABS.md §5.2), and here it is observable:
    // reading the absent argument as "the CPU" would make every `.float()` on
    // a meta parameter try to leave the meta device.
    let label = device_arg_or_label(args, kwargs, 3, "device", &PyDevice::meta())?;
    if !label.is_meta() {
        return Err(crate::tensor::no_data());
    }
    meta_result(py, input.dims().to_vec(), tag)
}

/// `aten::copy_` where one side is meta.
///
/// The gate above lets this op through with disagreeing devices because
/// transferring is its definition. What the two directions mean is not
/// symmetric, and both halves are measured:
///
/// * `meta.copy_(cpu)` **succeeds and does nothing.** The receiver stays meta.
///   `nn.Module.load_state_dict` without `assign=True` lands exactly here and
///   upstream warns about it in as many words -- *"copying from a non-meta
///   parameter in the checkpoint to a meta parameter in the current model,
///   which is a no-op"*.
/// * `cpu.copy_(meta)` **refuses**, with the copy-out message, because there is
///   nothing to read.
///
/// A shim that made the first case allocate would silently turn
/// `load_state_dict` into something that appeared to work; a shim that refused
/// it would stop a path upstream completes with a warning.
fn meta_copy_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.copy_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let source = tensor_arg(OP, args, kwargs, 1, "src")?;
    if !receiver.borrow().is_meta_repr() {
        // Receiver is dense, source is meta: reading the source is the copy.
        return Err(crate::tensor::no_data());
    }
    let _ = source;
    let _ = py;
    Ok(receiver.into_any().unbind())
}

fn aten_dispatch_inner(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    match op {
        "aten.add.Tensor" => add_tensor(py, args, kwargs),
        "aten.addmm.default" => addmm_default(py, args, kwargs),
        "aten.alias.default" => alias_default(py, args, kwargs),
        "aten.arange.default" => arange(py, args, kwargs, ArangeForm::End),
        "aten.arange.start" => arange(py, args, kwargs, ArangeForm::Start),
        "aten.arange.start_step" => arange(py, args, kwargs, ArangeForm::StartStep),
        "aten.argmax.default" => argmax_default(py, args, kwargs),
        "aten.bmm.default" => bmm_default(py, args, kwargs),
        "aten.cat.default" => cat_default(py, args, kwargs),
        "aten.stack.default" => stack_default(py, args, kwargs),
        "aten.scalar_tensor.default" => scalar_tensor_default(py, args, kwargs),
        "aten.embedding.default" => embedding_default(py, args, kwargs),
        "aten.empty.memory_format" => empty_memory_format(py, args, kwargs),
        "aten.full.default" => full_default(py, args, kwargs),
        "aten.is_floating_point.default" => is_floating_point_default(py, args, kwargs),
        "aten.isin.Tensor_Tensor" => isin_tensor_tensor(py, args, kwargs),
        "aten.lift_fresh.default" => lift_fresh_default(py, args, kwargs),
        "aten.mm.default" => mm_default(py, args, kwargs),
        "aten.ones.default" => ones_default(py, args, kwargs),
        "aten.pow.Scalar" => pow_scalar(py, args, kwargs),
        "aten.pow.Tensor_Scalar" => pow_tensor_scalar(py, args, kwargs),
        "aten.pow.Tensor_Tensor" => pow_tensor_tensor(py, args, kwargs),
        "aten.randint.default" => randint(py, args, kwargs, false),
        "aten.randint.low" => randint(py, args, kwargs, true),
        "aten.rsqrt.default" => rsqrt_default(py, args, kwargs),
        "aten.rsub.Scalar" => rsub_scalar(py, args, kwargs),

        // -- what upstream's `repr(tensor)` dispatches (docs/E2E_REAL.md) ----
        "aten.abs.default" => abs_default(py, args, kwargs),
        "aten.ceil.default" => ceil_default(py, args, kwargs),
        "aten.gt.Tensor" => compare_tensor(py, args, kwargs, "aten.gt.Tensor", Cmp::Gt),
        "aten.gt.Scalar" => compare_scalar(py, args, kwargs, "aten.gt.Scalar", Cmp::Gt),
        "aten.masked_select.default" => masked_select_default(py, args, kwargs),
        "aten.unbind.int" => unbind_int(py, args, kwargs),

        // -- attention (docs/OPS8.md) --------------------------------------
        "aten._scaled_dot_product_flash_attention_for_cpu.default" => {
            sdpa_flash_cpu(py, args, kwargs)
        }

        // -- the four docs/GPT2.md measured a 2-layer GPT-2 stopping on -----
        "aten.native_layer_norm.default" => native_layer_norm_default(py, args, kwargs),

        // -- the TensorBase surface (docs/TENSORBASE.md) -------------------
        "aten.add.Scalar" => arith_scalar(py, args, kwargs, "aten.add.Scalar", Arith::Add),
        "aten.sub.Tensor" => arith_tensor(py, args, kwargs, "aten.sub.Tensor", Arith::Sub),
        "aten.sub.Scalar" => arith_scalar(py, args, kwargs, "aten.sub.Scalar", Arith::Sub),
        "aten.mul.Tensor" => arith_tensor(py, args, kwargs, "aten.mul.Tensor", Arith::Mul),
        "aten.mul.Scalar" => arith_scalar(py, args, kwargs, "aten.mul.Scalar", Arith::Mul),
        "aten.div.Tensor" => arith_tensor(py, args, kwargs, "aten.div.Tensor", Arith::Div),
        "aten.div.Scalar" => arith_scalar(py, args, kwargs, "aten.div.Scalar", Arith::Div),
        "aten.matmul.default" => matmul_default(py, args, kwargs),

        "aten.eq.Tensor" => compare_tensor(py, args, kwargs, "aten.eq.Tensor", Cmp::Eq),
        "aten.eq.Scalar" => compare_scalar(py, args, kwargs, "aten.eq.Scalar", Cmp::Eq),
        "aten.ne.Tensor" => compare_tensor(py, args, kwargs, "aten.ne.Tensor", Cmp::Ne),
        "aten.ne.Scalar" => compare_scalar(py, args, kwargs, "aten.ne.Scalar", Cmp::Ne),
        "aten.lt.Tensor" => compare_tensor(py, args, kwargs, "aten.lt.Tensor", Cmp::Lt),
        "aten.lt.Scalar" => compare_scalar(py, args, kwargs, "aten.lt.Scalar", Cmp::Lt),
        "aten.le.Scalar" => compare_scalar(py, args, kwargs, "aten.le.Scalar", Cmp::Le),
        "aten.le.Tensor" => compare_tensor(py, args, kwargs, "aten.le.Tensor", Cmp::Le),

        "aten.bitwise_and.Tensor" => bitwise_binary(py, args, kwargs, "aten.bitwise_and.Tensor", Bitwise::And),
        "aten.bitwise_or.Tensor" => bitwise_binary(py, args, kwargs, "aten.bitwise_or.Tensor", Bitwise::Or),
        "aten.bitwise_and.Scalar" => bitwise_scalar(py, args, kwargs, "aten.bitwise_and.Scalar", Bitwise::And),
        "aten.bitwise_or.Scalar" => bitwise_scalar(py, args, kwargs, "aten.bitwise_or.Scalar", Bitwise::Or),
        "aten.bitwise_not.default" => bitwise_not_default(py, args, kwargs),

        "aten.cos.default" => unary_float(py, args, kwargs, "aten.cos.default", Unary::Cos),
        "aten.sin.default" => unary_float(py, args, kwargs, "aten.sin.default", Unary::Sin),
        "aten.reciprocal.default" => {
            unary_float(py, args, kwargs, "aten.reciprocal.default", Unary::Reciprocal)
        }
        "aten.tanh.default" => unary_float(py, args, kwargs, "aten.tanh.default", Unary::Tanh),
        "aten.neg.default" => neg_default(py, args, kwargs),
        "aten.silu.default" => silu_default(py, args, kwargs),
        "aten.relu.default" => relu_default(py, args, kwargs),
        "aten.relu_.default" => relu_inplace(py, args, kwargs),

        "aten.sum.default" => sum_or_mean(py, args, kwargs, "aten.sum.default", Reduce::Sum, false),
        "aten.sum.dim_IntList" => {
            sum_or_mean(py, args, kwargs, "aten.sum.dim_IntList", Reduce::Sum, true)
        }
        "aten.mean.default" => sum_or_mean(py, args, kwargs, "aten.mean.default", Reduce::Mean, false),
        "aten.mean.dim" => sum_or_mean(py, args, kwargs, "aten.mean.dim", Reduce::Mean, true),
        "aten.cumsum.default" => cumsum_default(py, args, kwargs),
        "aten.max.default" => extremum_default(py, args, kwargs, Extremum::Max),
        "aten.min.default" => extremum_default(py, args, kwargs, Extremum::Min),
        "aten.max.dim" => max_dim(py, args, kwargs),
        "aten.max.other" => max_other(py, args, kwargs),
        "aten.any.default" => any_default(py, args, kwargs),
        "aten.any.dim" => any_dim(py, args, kwargs, "aten.any.dim", false),
        "aten.any.dims" => any_dim(py, args, kwargs, "aten.any.dims", true),

        "aten.masked_fill.Scalar" => masked_fill(py, args, kwargs, "aten.masked_fill.Scalar"),
        "aten.masked_fill.Tensor" => masked_fill(py, args, kwargs, "aten.masked_fill.Tensor"),
        "aten.where.self" => where_self(py, args, kwargs),
        "aten.where.ScalarOther" => where_scalar_other(py, args, kwargs),

        "aten.expand.default" => expand_default(py, args, kwargs),
        "aten.reshape.default" => reshape_like(py, args, kwargs, "aten.reshape.default", "shape"),
        "aten.view.default" => reshape_like(py, args, kwargs, "aten.view.default", "size"),
        "aten.view.dtype" => view_dtype(py, args, kwargs),
        // Upstream's `_unsafe_view` differs from `view` only in what it
        // promises the autograd engine about aliasing -- the value is
        // `view`'s. There is no autograd here, so it is the same kernel, and
        // the key stays distinct because `reshape()`'s non-contiguous path
        // emits this one and not `view`.
        "aten._unsafe_view.default" => {
            reshape_like(py, args, kwargs, "aten._unsafe_view.default", "size")
        }
        "aten.transpose.int" => transpose_int(py, args, kwargs),
        "aten.permute.default" => permute_default(py, args, kwargs),
        "aten.t.default" => t_default(py, args, kwargs),
        "aten.unsqueeze.default" => unsqueeze_default(py, args, kwargs),
        "aten.squeeze.dim" => squeeze_dim(py, args, kwargs),
        "aten.split.Tensor" => split_tensor(py, args, kwargs),
        "aten.contiguous.default" => contiguous_default(py, args, kwargs),
        "aten.clone.default" => clone_default(py, args, kwargs),
        "aten.detach.default" => detach_default(py, args, kwargs),
        "aten._to_copy.default" => to_copy_default(py, args, kwargs),
        "aten.new_ones.default" => new_ones_default(py, args, kwargs),
        "aten.zeros.default" => zeros_or_ones(py, args, kwargs, "aten.zeros.default", false),
        "aten._local_scalar_dense.default" => local_scalar_dense(py, args, kwargs),

        "aten.select.int" => select_int(py, args, kwargs),
        "aten.slice.Tensor" => slice_tensor(py, args, kwargs),
        "aten.index.Tensor" => index_tensor(py, args, kwargs),

        "aten.fill_.Scalar" => fill_inplace(py, args, kwargs, "aten.fill_.Scalar"),
        "aten.fill_.Tensor" => fill_inplace(py, args, kwargs, "aten.fill_.Tensor"),
        "aten.zero_.default" => zero_inplace(py, args, kwargs),
        "aten.copy_.default" => copy_inplace(py, args, kwargs),
        "aten.uniform_.default" => uniform_inplace(py, args, kwargs),
        "aten.normal_.default" => normal_inplace(py, args, kwargs),

        // -- what widening past the Llama/GPT-2 family asks for (docs/ARCH.md) --
        "aten.gelu.default" => gelu_default(py, args, kwargs),
        "aten.gather.default" => gather_default(py, args, kwargs),

        // -- the eight `do_sample=True` stops on (docs/SAMPLING.md) ---------
        "aten._softmax.default" => softmax_default(py, args, kwargs),
        "aten.scatter.src" => scatter_src(py, args, kwargs),
        "aten.sort.default" => sort_default(py, args, kwargs),
        "aten.topk.default" => topk_default(py, args, kwargs),
        "aten.multinomial.default" => multinomial_default(py, args, kwargs),

        // -- falcon / bloom / gpt_bigcode (docs/TAIL.md) --------------------
        "aten._safe_softmax.default" => safe_softmax_default(py, args, kwargs),
        "aten.add_.Tensor" => add_inplace(py, args, kwargs),
        "aten.baddbmm.default" => baddbmm_default(py, args, kwargs),
        "aten.split_with_sizes.default" => split_with_sizes(py, args, kwargs),

        // -- mamba / mixtral (docs/OPS4.md) ---------------------------------
        "aten.exp.default" => unary_float(py, args, kwargs, "aten.exp.default", Unary::Exp),
        "aten.softplus.default" => softplus_default(py, args, kwargs),
        "aten.convolution.default" => convolution_default(py, args, kwargs),
        "aten.zeros_like.default" => zeros_or_empty_like(py, args, kwargs, "aten.zeros_like.default"),
        "aten.empty_like.default" => zeros_or_empty_like(py, args, kwargs, "aten.empty_like.default"),
        "aten.ge.Scalar" => compare_scalar(py, args, kwargs, "aten.ge.Scalar", Cmp::Ge),
        "aten.floor_divide.default" => floor_divide_default(py, args, kwargs),
        "aten.histc.default" => histc_default(py, args, kwargs),
        "aten.clamp_.default" => clamp_inplace_default(py, args, kwargs),
        "aten.div_.Tensor" => div_inplace_tensor(py, args, kwargs),
        "aten.masked_fill_.Scalar" => masked_fill_inplace(py, args, kwargs, "aten.masked_fill_.Scalar"),
        "aten.index_put_.default" => index_put_inplace(py, args, kwargs),

        other => Err(aten_not_implemented(other)),
    }
}

#[pyfunction]
#[pyo3(name = "_aten_implemented")]
pub fn aten_implemented() -> Vec<&'static str> {
    IMPLEMENTED.to_vec()
}

/// See `IMPLEMENTED_AWAITING_GOLDEN`. Separate function rather than a flag on
/// `_aten_implemented()` so that nothing can read the union by accident.
#[pyfunction]
#[pyo3(name = "_aten_implemented_awaiting_golden")]
pub fn aten_implemented_awaiting_golden() -> Vec<&'static str> {
    IMPLEMENTED_AWAITING_GOLDEN.to_vec()
}

/// Everything `_aten_dispatch` answers, as one sorted list. Exposed so the
/// smoke tests can check the dispatch table against the three constants rather
/// than keeping a fourth copy of the names.
#[pyfunction]
#[pyo3(name = "_aten_all_implemented")]
pub fn aten_all_implemented() -> Vec<&'static str> {
    all_implemented()
}

// ---------------------------------------------------------------------------
// Implemented ops
//
// Three ops, deliberately of three different *kinds* rather than three of the
// same kind -- a factory, an elementwise binary, and a matmul. Each exercises a
// different part of the floor, so the pattern is shown to generalise. The
// reasoning is written out in docs/TORCH_C.md.
// ---------------------------------------------------------------------------

/// `aten::full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None,
///             Layout? layout=None, Device? device=None, bool? pin_memory=None)`
///
/// The factory. Without one, every tensor has to enter through a back door that
/// the dispatcher cannot see, which would defeat the instrument above.
fn full_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.full.default";

    let size: Vec<usize> = required(OP, args, kwargs, 0, "size")?.extract()?;
    let fill = required(OP, args, kwargs, 1, "fill_value")?;

    // torch infers int64 from an integer fill value and the default float dtype
    // otherwise. A Python `bool` lands in this branch because `bool` subclasses
    // `int`; torch would give it `torch.bool`, which the shim has no dtype for
    // at all -- recorded in docs/TORCH_C.md rather than papered over.
    // `bool` subclasses `int` in Python, so the bool test has to come first.
    // torch gives `torch.full((2,), True)` dtype `torch.bool`; before the
    // dtype tag existed this branch was unreachable and the shim handed back
    // `int64` (docs/TORCH_C.md §2 recorded it as an open item).
    let fill_is_bool = fill.is_instance_of::<pyo3::types::PyBool>();
    let fill_is_int = fill.is_instance_of::<pyo3::types::PyInt>();
    let dtype = match optional(args, kwargs, 2, "dtype")? {
        Some(value) if !value.is_none() => value.extract::<PyDtype>()?.tag(),
        _ if fill_is_bool => TorchDType::Bool,
        _ if fill_is_int => TorchDType::Int64,
        // `default_float()`, not a literal `Float32`: this arm *is* the "and
        // the default float dtype otherwise" half of the rule the comment
        // above states, and leaving it a constant would have made `full` the
        // one factory that ignores `set_default_dtype`. Measured upstream
        // under a float64 default: `torch.full((2,), 1.0).dtype` is float64.
        _ => default_float(),
    };

    reject_unsupported(OP, args, kwargs, &[(3, "layout"), (5, "pin_memory")])?;
    let label = device_arg_or_label(args, kwargs, 4, "device", &PyDevice::cpu())?;

    // torch refuses a fill value the target dtype cannot hold. candle would
    // wrap (int) or saturate (float) instead, which is the silent divergence
    // the golden harness caught -- see `checked_convert` below.
    //
    // Checked *before* the meta branch on purpose: upstream's meta kernel runs
    // the same conversion check, so `torch.full((3,), 2**31, dtype=torch.int32,
    // device="meta")` raises there too. A meta tensor is a claim about what the
    // real call would have produced, and a claim that skipped the checks the
    // real call makes would be worth less than no claim.
    checked_convert(&fill, fill_is_int, dtype, size.iter().product())?;
    if label.is_meta() {
        return meta_result(py, size, dtype);
    }
    let device = label.resolve()?;
    let storage = PyDtype::new(dtype).storage(OP)?;

    if dtype == TorchDType::Bool {
        // Normalised on the way in, which is what makes the tag's invariant
        // hold by construction rather than by hope (BOOL.md §6.3).
        let truthy = fill.is_truthy()?;
        let tensor = Tensor::full(u8::from(truthy), size, &device)
            .map_err(|e| candle_err(OP, e))?;
        return Ok(PyTensorBase::boolean(tensor)?
            .into_pyobject(py)?
            .into_any()
            .unbind());
    }

    let tensor = if storage.is_int() {
        let value: i64 = fill.extract()?;
        Tensor::full(value, size, &device)
    } else {
        let value: f64 = fill.extract()?;
        Tensor::full(value, size, &device)
    }
    .and_then(|t| t.to_dtype(storage))
    .map_err(|e| candle_err(OP, e))?;

    Ok(PyTensorBase::new(tensor)?.into_pyobject(py)?.into_any().unbind())
}

/// `aten::add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1)`
///
/// The elementwise binary. This is where torch and candle disagree the most
/// cheaply observable way: torch broadcasts *and* promotes dtypes, candle
/// broadcasts but requires matching dtypes. Promotion is not implemented here;
/// a mismatch raises with both dtypes named rather than guessing, because a
/// wrong promotion is the silent numerical drift DESIGN.md §5 calls A's main
/// risk.
fn add_tensor(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.add.Tensor";

    let lhs = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rhs = tensor_arg(OP, args, kwargs, 1, "other")?;
    let alpha: f64 = match optional(args, kwargs, 2, "alpha")? {
        Some(value) if !value.is_none() => value.extract()?,
        _ => 1.0,
    };

    let tag = same_dtype(OP, &lhs, &rhs)?;
    // `bool + bool` is a logical or in torch, not an arithmetic sum
    // (BOOL.md §2.2). candle's `broadcast_add` would give 2 where both are
    // true, which is still truthy and therefore silently wrong downstream --
    // so this refuses rather than approximates.
    if tag == TorchDType::Bool {
        return Err(not_implemented(format!(
            "{OP}: torch.bool addition is logical or, not arithmetic, and is              not implemented in torch._C shim"
        )));
    }

    // Widened, added, and narrowed once -- `opmath_in`. `alpha` is applied
    // *after* the widening on purpose: upstream converts it to `opmath_t` and
    // multiplies there, so scaling in `bfloat16` first would round the
    // multiply and the add separately where torch rounds only the result.
    let storage = PyDtype::new(tag).storage(OP)?;
    let acc = opmath_in(storage);
    let lhs = lhs.tensor()?.to_dtype(acc).map_err(|e| candle_err(OP, e))?;
    let rhs = rhs.tensor()?.to_dtype(acc).map_err(|e| candle_err(OP, e))?;
    let rhs = scale_by_alpha(OP, &rhs, alpha, storage)?;
    let out = lhs
        .broadcast_add(&rhs)
        .and_then(|t| t.to_dtype(storage))
        .map_err(|e| candle_err(OP, e))?;

    Ok(PyTensorBase::new(out)?.into_pyobject(py)?.into_any().unbind())
}

/// The dtype torch accumulates a GEMM of this storage dtype in.
///
/// **`float16` and `bfloat16` GEMMs accumulate in `float32` upstream**
/// (`at::opmath_type<Half> == float`), and this is measured, not inferred:
/// `mm(half a, half b)` is *bitwise* equal to `half(mm(float(a), float(b)))`
/// at k = 4, 64 and 512, and the same holds for `bmm` and `addmm`.
///
/// candle has no such notion -- `Tensor::matmul` accumulates in the storage
/// dtype -- so routing a `float16` matmul straight at it computes a different
/// function, and the difference is not a rounding-order nicety: at k = 512
/// with unit-magnitude inputs, **15 of 64 outputs land outside this harness's
/// `float16` tolerance**, with a maximum absolute error of 0.078 against a
/// tolerance of 5e-3. It grows with the reduction depth (1/64 outputs already
/// wrong at k = 64), so a real model in `float16` drifts layer by layer.
///
/// That went unnoticed because every GEMM case in `tools/golden/cases.py` was
/// small enough for `float16` accumulation to be lossless; docs/GPT2.md §7
/// listed "the error at real layer sizes" as unmeasured, and this was in it.
/// The large-size cases added alongside this function are what found it.
///
/// The same widening incidentally gives `bfloat16` a matmul at all: candle has
/// no BF16 matmul kernel (`unsupported dtype BF16 for op matmul`), which
/// `mm`/`bmm`/`addmm` had been recording as a capability gap. Accumulating in
/// `float32` is not a workaround for that gap -- it is what upstream does --
/// so the gap closes as a side effect rather than being papered over.
///
/// The integral dtypes are deliberately *not* widened. candle having no
/// integral matmul is a real gap, and `float32` cannot hold an `int64` product
/// exactly; standing one in would answer a different question.
fn gemm_accumulate_in(storage: candle_core::DType) -> candle_core::DType {
    opmath_in(storage)
}

/// `at::opmath_type` -- the dtype torch *computes* in for a given storage
/// dtype. `float` for both reduced floats, the storage dtype for everything
/// else.
///
/// This is one rule, not a GEMM rule: torch widens `bfloat16` and `float16`
/// for **every** arithmetic kernel, computes there, and narrows back exactly
/// once, with round-to-nearest-even. `gemm_accumulate_in` above is this
/// function under the name the matmuls found it by; the elementwise ops and
/// the reductions need it for the same reason and were missing it.
///
/// **What that cost, measured.** candle's `bfloat16` add narrows by
/// truncating, and only on its vectorised path -- correct below 32 elements,
/// wrong at 32 and above, which is why every case in `tools/golden/cases.py`
/// (all at most 24 elements) passed. Truncation is *biased*: it moves every
/// rounded element toward zero instead of splitting them evenly, so the error
/// accumulates down a residual stream instead of cancelling. On
/// SmolLM2-135M's default `bfloat16` path that reached a maximum logit
/// difference of 11.75 against upstream and changed the generated text; with
/// the widening it is 0.0 and the tokens match. docs/BF16.md.
///
/// Widening is not "more accurate than torch" here -- it is what torch does.
/// `add` on two `bfloat16` values is `float(a) + float(b)` narrowed once
/// upstream too, so this reproduces the operation rather than improving it.
/// The integral dtypes are deliberately untouched: widening an `int64` to
/// `float32` would lose bits rather than keep them.
fn opmath_in(storage: candle_core::DType) -> candle_core::DType {
    match storage {
        candle_core::DType::F16 | candle_core::DType::BF16 => candle_core::DType::F32,
        other => other,
    }
}

/// `alpha * operand`, where `operand` is already widened to `opmath_in`.
///
/// `add` and `sub` are the only ops with an `alpha`, and it is **not** simply
/// a factor applied in the widened dtype. Two things about it were measured
/// against torch 2.13.0 over 20000 elements per alpha, because both are the
/// kind of thing that is invisible at `alpha=1` (the only value any caller in
/// this repository passes) and wrong everywhere else:
///
///   * **`alpha` is narrowed to the storage dtype first.** With
///     `alpha=0.3` on `bfloat16`, torch multiplies by `0.30078125`, not by
///     `0.3`. Keeping the `f64` the parser produced disagrees on 312/20000
///     elements.
///   * **On `bfloat16` the product is narrowed too, and on `float16` it is
///     not.** This asymmetry is measured, not derived, and it is not a
///     rounding accident: on `bfloat16`, narrowing the product agrees with
///     upstream on 0/20000 elements for every one of `alpha` in
///     {3, 2, 0.3, -1.5, 1.7}, while leaving it in `float` disagrees on up to
///     1607. On `float16` it is the other way round -- narrowing the product
///     disagrees on up to 1246 where leaving it in `float` disagrees on 0 or
///     1. The two reduced floats reach different kernels upstream.
///
/// The single residual `float16` disagreement (1/20000 at `alpha=1.7`) is a
/// double-rounding edge and is left as measured rather than papered over.
fn scale_by_alpha(
    op: &str,
    operand: &Tensor,
    alpha: f64,
    storage: candle_core::DType,
) -> PyResult<Tensor> {
    if alpha == 1.0 {
        return Ok(operand.clone());
    }
    let acc = opmath_in(storage);
    if acc == storage {
        return operand.affine(alpha, 0.0).map_err(|e| candle_err(op, e));
    }
    let narrowed = Tensor::full(alpha, (), operand.device())
        .and_then(|t| t.to_dtype(storage))
        .and_then(|t| t.to_dtype(candle_core::DType::F64))
        .and_then(|t| t.to_scalar::<f64>())
        .map_err(|e| candle_err(op, e))?;
    let scaled = operand
        .affine(narrowed, 0.0)
        .map_err(|e| candle_err(op, e))?;
    if storage == candle_core::DType::BF16 {
        return scaled
            .to_dtype(storage)
            .and_then(|t| t.to_dtype(acc))
            .map_err(|e| candle_err(op, e));
    }
    Ok(scaled)
}

/// `aten::mm(Tensor self, Tensor mat2)`
///
/// The matmul. Chosen over the other elementwise ops because it is the one op
/// that is *hot* -- `nn.Linear`, the only compute-heavy module in the ten
/// live Python modules of IMPORT_WALLS §5, becomes this. It is also the op
/// whose backend choice (naive gemm vs Accelerate vs a fused kernel from §8)
/// will be revisited, so the floor should already route through it.
fn mm_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.mm.default";

    let lhs = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rhs = tensor_arg(OP, args, kwargs, 1, "mat2")?;

    // torch's `mm` is strictly 2-D; `matmul`/`bmm` are separate ops with their
    // own overloads. candle's `matmul` accepts batched inputs, so accepting a
    // 3-D argument here would quietly implement a different op.
    if lhs.tensor()?.rank() != 2 || rhs.tensor()?.rank() != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{OP}: both arguments to mm need to be 2D, but they are {}D and {}D",
            lhs.tensor()?.rank(),
            rhs.tensor()?.rank()
        )));
    }
    let tag = same_dtype(OP, &lhs, &rhs)?;

    // Accumulate where torch accumulates -- see `gemm_accumulate_in`.
    let storage = PyDtype::new(tag).storage(OP)?;
    let acc = gemm_accumulate_in(storage);
    let rhs_inner = rhs.tensor()?;
    let out = lhs
        .tensor()?
        .to_dtype(acc)
        .and_then(|l| rhs_inner.to_dtype(acc).and_then(|r| l.matmul(&r)))
        .and_then(|p| p.to_dtype(storage))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `aten::bmm(Tensor self, Tensor mat2) -> Tensor`
///
/// **Not a one-line route into `matmul_default`.** The kernel underneath is
/// indeed the same candle call, and `matmul_default` already batches -- but it
/// batches by *broadcasting*, and `bmm` does not. Upstream refuses
/// `bmm((1,3,4), (2,4,5))` ("Expected size for first two dimensions of batch2
/// tensor to be: [1, 4] but got: [2, 4]") where `broadcast_matmul` happily
/// expands the batch of 1. Sending `aten.bmm.default` at `matmul_default`
/// would therefore implement a *different* op -- one that computes where
/// torch raises, which is the silent-divergence direction DESIGN.md §5 exists
/// to keep out.
///
/// So the shared part is the multiply and the distinct part is the contract:
/// both operands strictly 3-D, batch extents equal, no broadcasting.
fn bmm_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.bmm.default";

    let lhs = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rhs = tensor_arg(OP, args, kwargs, 1, "mat2")?;

    if lhs.tensor()?.rank() != 3 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "batch1 must be a 3D tensor",
        ));
    }
    if rhs.tensor()?.rank() != 3 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "batch2 must be a 3D tensor",
        ));
    }
    let tag = same_dtype(OP, &lhs, &rhs)?;

    // torch checks batch2's leading pair against batch1's (batch, k) and says
    // so in exactly these words. Reproduced rather than paraphrased: the
    // message is the work item a caller reads.
    let a = lhs.tensor()?.dims();
    let b = rhs.tensor()?.dims();
    if a[0] != b[0] || a[2] != b[1] {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Expected size for first two dimensions of batch2 tensor to be: \
             [{}, {}] but got: [{}, {}].",
            a[0], a[2], b[0], b[1]
        )));
    }

    let storage = PyDtype::new(tag).storage(OP)?;
    let acc = gemm_accumulate_in(storage);
    let rhs_inner = rhs.tensor()?;
    let out = lhs
        .tensor()?
        .to_dtype(acc)
        .and_then(|l| l.contiguous())
        .and_then(|l| {
            rhs_inner
                .to_dtype(acc)
                .and_then(|r| r.contiguous())
                .and_then(|r| l.matmul(&r))
        })
        .and_then(|p| p.to_dtype(storage))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `beta`/`alpha` for `addmm`, applied in the *result* dtype.
///
/// The truncation is upstream's, measured: `addmm` on `int64` operands with
/// `alpha=1.9` gives the same answer as `alpha=1`, and with `alpha=-1.9` the
/// same as `alpha=-1`. torch converts the `Scalar` to `scalar_t` before it
/// multiplies, so a fractional factor on an integral matmul is silently
/// truncated toward zero -- including `beta=0.5`, which rounds to `0` and
/// drops `self` entirely.
fn addmm_scale(
    op: &str,
    tensor: &Tensor,
    factor: Scalar,
    storage: candle_core::DType,
) -> PyResult<Tensor> {
    if storage.is_int() {
        let k = factor.as_i64();
        if k == 1 {
            return Ok(tensor.clone());
        }
        let scale = Tensor::full(k, (), tensor.device())
            .and_then(|t| t.to_dtype(storage))
            .map_err(|e| candle_err(op, e))?;
        tensor.broadcast_mul(&scale).map_err(|e| candle_err(op, e))
    } else {
        let k = factor.as_f64();
        if k == 1.0 {
            return Ok(tensor.clone());
        }
        tensor.affine(k, 0.0).map_err(|e| candle_err(op, e))
    }
}

/// `aten::addmm(Tensor self, Tensor mat1, Tensor mat2, *, Scalar beta=1,
///     Scalar alpha=1) -> Tensor`
///
/// `beta * self + alpha * (mat1 @ mat2)`, and the reason it exists as its own
/// op rather than as `mm` + `add` is docs/NN_SURFACE.md §5: `at::native::linear`
/// emits `addmm` for every `bias=True` branch, so a shim without it makes
/// `nn.Linear` take a path upstream would not take. `bootstrap.py` already
/// reads `_aten_all_implemented()` to decide, so landing this kernel is what
/// retires that patch -- nothing in the Python layer has to change.
///
/// Everything below is measured against torch 2.13.0, and three of the four
/// surprises would have been got wrong by inference:
///
///   * **`beta == 0` and `alpha == 0` are quick returns, not multiplications.**
///     `addmm(full(nan), m, n, beta=0)` gives a clean product, and
///     `addmm(b, m_with_inf, n, alpha=0)` gives a clean `b` -- so a literal
///     `0.0 * nan` would produce NaN where torch produces a number. Both
///     branches are therefore skipped, not scaled.
///   * **`self` is validated even when `beta == 0`.** A wrongly-shaped `self`
///     raises with `beta=0, alpha=0`, where nothing reads it. The expand check
///     runs unconditionally here for that reason.
///   * **no promotion, and the two checks are asymmetric.** `mat1` is compared
///     to `mat2` first ("mat1 and mat2 must have the same dtype"), then `self`
///     to `mat2` ("self and mat2 must have the same dtype") -- `mat2` is the
///     reference in both messages.
///   * `bool` and the wide unsigned dtypes have no `addmm_impl_cpu_` upstream.
///
/// The dtype gap this inherits is `mm`'s, not a new one: candle's `matmul` has
/// no integral kernel, so `int64`/`int32`/`int16`/`uint8`/`bfloat16` refuse
/// here exactly where `aten.mm.default` already refuses (docs/TORCH_C.md §2),
/// *except* when `alpha == 0` -- then no matmul happens and the answer comes
/// out. That asymmetry is deliberate: it is the quick return above, and
/// refusing it would be inventing a restriction torch does not have.
fn addmm_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.addmm.default";

    let bias = tensor_arg(OP, args, kwargs, 0, "self")?;
    let mat1 = tensor_arg(OP, args, kwargs, 1, "mat1")?;
    let mat2 = tensor_arg(OP, args, kwargs, 2, "mat2")?;

    if mat1.tensor()?.rank() != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "mat1 must be a matrix, got {}-D tensor",
            mat1.tensor()?.rank()
        )));
    }
    if mat2.tensor()?.rank() != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "mat2 must be a matrix, got {}-D tensor",
            mat2.tensor()?.rank()
        )));
    }
    if mat1.tag() != mat2.tag() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "mat1 and mat2 must have the same dtype, but got {} and {}",
            scalar_type_name(mat1.tag()),
            scalar_type_name(mat2.tag())
        )));
    }
    if bias.tag() != mat2.tag() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "self and mat2 must have the same dtype, but got {} and {}",
            scalar_type_name(bias.tag()),
            scalar_type_name(mat2.tag())
        )));
    }

    let tag = mat2.tag();
    if matches!(
        tag,
        TorchDType::Bool | TorchDType::UInt16 | TorchDType::UInt32 | TorchDType::UInt64
    ) {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"addmm_impl_cpu_\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }

    let a = mat1.tensor()?.dims().to_vec();
    let b = mat2.tensor()?.dims().to_vec();
    if a[1] != b[0] {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "mat1 and mat2 shapes cannot be multiplied ({}x{} and {}x{})",
            a[0], a[1], b[0], b[1]
        )));
    }
    let target = vec![a[0], b[1]];

    // `self` expanded to the product's shape, with torch's own two messages
    // for the two ways that fails. Done here rather than left to candle so the
    // caller reads the same sentence upstream would have given.
    let self_dims = bias.tensor()?.dims().to_vec();
    if self_dims.len() > target.len() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "expand(torch.{}Tensor{{{:?}}}, size={:?}): the number of sizes provided ({}) \
             must be greater or equal to the number of dimensions in the tensor ({})",
            scalar_type_name(tag),
            self_dims,
            target,
            target.len(),
            self_dims.len()
        )));
    }
    let offset = target.len() - self_dims.len();
    for (i, &extent) in self_dims.iter().enumerate() {
        let wanted = target[i + offset];
        if extent != wanted && extent != 1 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "The expanded size of the tensor ({wanted}) must match the existing size \
                 ({extent}) at non-singleton dimension {}.  Target sizes: {:?}.  \
                 Tensor sizes: {:?}",
                i + offset,
                target,
                self_dims
            )));
        }
    }

    let storage = PyDtype::new(tag).storage(OP)?;
    let beta = scalar_arg(OP, args, kwargs, 3, "beta")?.unwrap_or(Scalar::Int(1));
    let alpha = scalar_arg(OP, args, kwargs, 4, "alpha")?.unwrap_or(Scalar::Int(1));
    // Zero is decided in the *result* dtype, which is why `beta=0.5` on an
    // integral addmm counts as zero: it truncates to 0 before it multiplies.
    let (beta_zero, alpha_zero) = if storage.is_int() {
        (beta.as_i64() == 0, alpha.as_i64() == 0)
    } else {
        (beta.as_f64() == 0.0, alpha.as_f64() == 0.0)
    };

    // The whole body runs in the accumulation dtype and is narrowed once at the
    // end, which is upstream's shape: `addmm(half ...)` is bitwise equal to
    // `half(addmm(float ...))` (measured). Narrowing the product first and then
    // adding the bias in `float16` would round twice where torch rounds once.
    // See `gemm_accumulate_in`.
    let acc_dtype = gemm_accumulate_in(storage);
    let mut acc: Option<Tensor> = None;
    if !alpha_zero {
        let mat2_inner = mat2.tensor()?;
        let product = mat1
            .tensor()?
            .to_dtype(acc_dtype)
            .and_then(|l| l.contiguous())
            .and_then(|l| {
                mat2_inner
                    .to_dtype(acc_dtype)
                    .and_then(|r| r.contiguous())
                    .and_then(|r| l.matmul(&r))
            })
            .map_err(|e| candle_err(OP, e))?;
        acc = Some(addmm_scale(OP, &product, alpha, acc_dtype)?);
    }
    if !beta_zero {
        let expanded = bias
            .tensor()?
            .to_dtype(acc_dtype)
            .and_then(|t| t.broadcast_as(target.as_slice()))
            .and_then(|t| t.contiguous())
            .map_err(|e| candle_err(OP, e))?;
        let scaled = addmm_scale(OP, &expanded, beta, acc_dtype)?;
        acc = Some(match acc {
            Some(product) => product.add(&scaled).map_err(|e| candle_err(OP, e))?,
            None => scaled,
        });
    }
    let out = match acc {
        Some(tensor) => tensor.to_dtype(storage).map_err(|e| candle_err(OP, e))?,
        // Both factors zero: torch still answers with a correctly shaped,
        // correctly typed tensor of zeros rather than raising.
        None => Tensor::zeros(target.as_slice(), storage, mat1.tensor()?.device())
            .map_err(|e| candle_err(OP, e))?,
    };
    finish(py, out, tag)
}

/// `aten::baddbmm(Tensor self, Tensor batch1, Tensor batch2, *, Scalar beta=1,
///     Scalar alpha=1) -> Tensor`
///
/// `bmm`'s batching composed with `addmm`'s `beta * self + alpha * (batch1 @
/// batch2)`, needed to open `bloom` (docs/TAIL.md) -- its attention builds the
/// scaled QK^T scores with this one op rather than a separate scale-then-add.
///
/// The `beta=0` quick return is `addmm`'s, reused batched and re-measured to
/// still hold: `baddbmm(self, b1, b2, beta=0)` on a `nan`-filled `self` gives
/// a clean product (no `nan` leaks through the skipped add). `alpha=0` is
/// **not** the mirror-image quick return, despite the kernel used to claim
/// so and despite `addmm` itself skipping the multiply on `alpha=0` --
/// measured against torch 2.13.0 (docs/TAIL.md §2.1):
/// `baddbmm(zeros_self, inf_batch1, batch2, alpha=0)` comes back with `nan`
/// in it, so upstream still runs the real IEEE multiply and only then scales
/// by zero (`0 * inf == nan`, not skipped). This kernel now does the same:
/// the product is always computed and scaled through `addmm_scale`, and only
/// the `self` term is quick-returned on `beta=0`. One consequence: dtypes
/// candle has no matmul kernel for (`_MM_C_ERROR_DTYPES` in
/// `tools/golden/cases.py`) now raise on `alpha=0` too, where the old
/// unconditional skip used to dodge the missing kernel -- that dodge was
/// itself part of the bug, not a feature to preserve. Refusals are `addmm`'s
/// dtype-mismatch pair too (`batch1`/`batch2` compared first, `self`/`batch2`
/// second), plus `bmm`'s rank checks in place of `mm`'s, since the batch
/// dimension is what `bmm` added and `addmm` never had.
///
/// `self` broadcasts to the `(batch, n, p)` output the same way `addmm`'s
/// bias does -- rank up to 3, trailing dimensions either matching or `1` --
/// so a `(n, p)` or scalar `self` is as valid here as a fully-batched one,
/// matching `nn.Linear`-style bias broadcasting into a batched matmul.
fn baddbmm_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.baddbmm.default";

    let bias = tensor_arg(OP, args, kwargs, 0, "self")?;
    let batch1 = tensor_arg(OP, args, kwargs, 1, "batch1")?;
    let batch2 = tensor_arg(OP, args, kwargs, 2, "batch2")?;

    if batch1.tensor()?.rank() != 3 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "batch1 must be a 3D tensor",
        ));
    }
    if batch2.tensor()?.rank() != 3 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "batch2 must be a 3D tensor",
        ));
    }
    if batch1.tag() != batch2.tag() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "batch1 and batch2 must have the same dtype, but got {} and {}",
            scalar_type_name(batch1.tag()),
            scalar_type_name(batch2.tag())
        )));
    }
    if bias.tag() != batch2.tag() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "self and batch2 must have the same dtype, but got {} and {}",
            scalar_type_name(bias.tag()),
            scalar_type_name(batch2.tag())
        )));
    }

    let tag = batch2.tag();
    if matches!(
        tag,
        TorchDType::Bool | TorchDType::UInt16 | TorchDType::UInt32 | TorchDType::UInt64
    ) {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"baddbmm\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }

    let a = batch1.tensor()?.dims().to_vec();
    let b = batch2.tensor()?.dims().to_vec();
    if a[0] != b[0] || a[2] != b[1] {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Expected size for first two dimensions of batch2 tensor to be: \
             [{}, {}] but got: [{}, {}].",
            a[0], a[2], b[0], b[1]
        )));
    }
    let target = vec![a[0], a[1], b[2]];

    // Same expand check `addmm_default` runs on its bias, generalised to a
    // 3-D target.
    let self_dims = bias.tensor()?.dims().to_vec();
    if self_dims.len() > target.len() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "expand(torch.{}Tensor{{{:?}}}, size={:?}): the number of sizes provided ({}) \
             must be greater or equal to the number of dimensions in the tensor ({})",
            scalar_type_name(tag),
            self_dims,
            target,
            target.len(),
            self_dims.len()
        )));
    }
    let offset = target.len() - self_dims.len();
    for (i, &extent) in self_dims.iter().enumerate() {
        let wanted = target[i + offset];
        if extent != wanted && extent != 1 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "The expanded size of the tensor ({wanted}) must match the existing size \
                 ({extent}) at non-singleton dimension {}.  Target sizes: {:?}.  \
                 Tensor sizes: {:?}",
                i + offset,
                target,
                self_dims
            )));
        }
    }

    let storage = PyDtype::new(tag).storage(OP)?;
    let beta = scalar_arg(OP, args, kwargs, 3, "beta")?.unwrap_or(Scalar::Int(1));
    let alpha = scalar_arg(OP, args, kwargs, 4, "alpha")?.unwrap_or(Scalar::Int(1));
    // `alpha` has no quick return (see the kernel doc above) so only
    // `beta`'s zero-ness is decided ahead of time.
    let beta_zero = if storage.is_int() {
        beta.as_i64() == 0
    } else {
        beta.as_f64() == 0.0
    };

    // Same accumulation-dtype rule as `mm`/`bmm`/`addmm` -- see
    // `gemm_accumulate_in`.
    let acc_dtype = gemm_accumulate_in(storage);
    // Unlike `addmm`, `alpha == 0` is NOT a quick return here -- measured
    // against torch 2.13.0 (docs/TAIL.md §2.1): the multiply still runs and
    // its NaNs/Infs still leak through, only the *scale* is skipped. So the
    // product is always computed; `addmm_scale` folds the `alpha == 1` case
    // back down to a plain clone, same as it always did.
    //
    // The `?` on each `tensor()` is the meta change: a meta tensor has no bytes
    // to read, and the type says so rather than each kernel remembering to.
    let batch2_inner = batch2.tensor()?;
    let product = batch1
        .tensor()?
        .to_dtype(acc_dtype)
        .and_then(|l| l.contiguous())
        .and_then(|l| {
            batch2_inner
                .to_dtype(acc_dtype)
                .and_then(|r| r.contiguous())
                .and_then(|r| l.matmul(&r))
        })
        .map_err(|e| candle_err(OP, e))?;
    let mut acc: Option<Tensor> = Some(addmm_scale(OP, &product, alpha, acc_dtype)?);
    if !beta_zero {
        let expanded = bias
            .tensor()?
            .to_dtype(acc_dtype)
            .and_then(|t| t.broadcast_as(target.as_slice()))
            .and_then(|t| t.contiguous())
            .map_err(|e| candle_err(OP, e))?;
        let scaled = addmm_scale(OP, &expanded, beta, acc_dtype)?;
        acc = Some(match acc {
            Some(product) => product.add(&scaled).map_err(|e| candle_err(OP, e))?,
            None => scaled,
        });
    }
    let out = match acc {
        Some(tensor) => tensor.to_dtype(storage).map_err(|e| candle_err(OP, e))?,
        None => Tensor::zeros(target.as_slice(), storage, batch1.tensor()?.device())
            .map_err(|e| candle_err(OP, e))?,
    };
    finish(py, out, tag)
}

/// `aten::_scaled_dot_product_flash_attention_for_cpu(Tensor query, Tensor key,
///     Tensor value, float dropout_p=0., bool is_causal=False, *,
///     Tensor? attn_mask=None, float? scale=None) -> (Tensor, Tensor)`
///
/// The one fused op in this file, and the only one that answers with a pair of
/// tensors rather than one. Everything about it below was measured against
/// torch 2.13.0 rather than inferred, because the name says "flash attention"
/// and the observable contract is not what that name suggests:
///
///   * `is_causal` is **upper-left aligned**, not bottom-right: row `t`
///     attends keys `0..=t` even when the key sequence is longer than the
///     query one. Measured on a (q=2, kv=5) pair -- the bottom-right reading
///     disagrees on every element.
///   * `is_causal` and `attn_mask` **compose**. `F.scaled_dot_product_attention`
///     refuses to take both; this aten op accepts both and adds them, measured.
///   * the second result is `logsumexp` over the *masked, scaled* scores, so
///     it has to be computed after both masks land, not from the raw product.
///   * for `float16`/`bfloat16` inputs the output comes back in the input
///     dtype but the logsumexp comes back **`float32`** -- which is upstream
///     telling us the accumulation happens in float. This follows that:
///     reduced-precision inputs are widened to `f32` for the whole body and
///     only the output is narrowed again.
///
/// `dropout_p > 0` is refused here because upstream refuses it too ("Currently
/// do not support dropout > 0"), not because the shim lacks an RNG.
///
/// **Grouped-query attention lives here, not in the wrapper.** See
/// `repeat_kv_heads`.
///
/// Key and value are repeated to the query's head count *before* anything
/// else touches them, so `is_causal`, `attn_mask` and the logsumexp all see
/// the full head count. That order is not cosmetic: an `attn_mask` is indexed
/// by `(batch, head, row, col)` and upstream's has the *query* head count, so
/// repeating after masking would broadcast a per-head mask onto the wrong
/// heads.
fn sdpa_flash_cpu(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten._scaled_dot_product_flash_attention_for_cpu.default";

    let query = tensor_arg(OP, args, kwargs, 0, "query")?;
    let key = tensor_arg(OP, args, kwargs, 1, "key")?;
    let value = tensor_arg(OP, args, kwargs, 2, "value")?;
    let dropout_p = float_arg(args, kwargs, 3, "dropout_p", 0.0)?;
    let is_causal = bool_arg(args, kwargs, 4, "is_causal")?.unwrap_or(false);
    let attn_mask = match optional(args, kwargs, 5, "attn_mask")? {
        Some(value) if !value.is_none() => Some(value.extract::<PyTensorBase>()?),
        _ => None,
    };
    let scale = match optional(args, kwargs, 6, "scale")? {
        Some(value) if !value.is_none() => Some(value.extract::<f64>()?),
        _ => None,
    };

    same_dtype(OP, &query, &key)?;
    let tag = same_dtype(OP, &query, &value)?;
    if !tag.is_floating_point() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "scaled_dot_product_attention_flash_attention: Expected data type in \
             FP32, FP64, BF16, FP16, but got {} instead.",
            scalar_type_name(tag)
        )));
    }
    if dropout_p > 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "scaled_dot_product_attention_flash_attention: Currently do not support dropout > 0",
        ));
    }
    for operand in [&query, &key, &value] {
        if operand.tensor()?.rank() != 4 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "scaled_dot_product_attention_flash_attention: Accept only 4 dims inputs \
                 shape of {B, H, T, K}",
            ));
        }
    }
    if let Some(mask) = attn_mask.as_ref() {
        if mask.tag() != tag {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "scaled_dot_product_attention_flash_attention: Attention mask is the same \
                 data type as query",
            ));
        }
    }

    let storage = PyDtype::new(tag).storage(OP)?;
    // The widening upstream's `float32` logsumexp reports (see the doc comment).
    let acc = match storage {
        candle_core::DType::F16 | candle_core::DType::BF16 => candle_core::DType::F32,
        other => other,
    };
    let acc_tag = TorchDType::from_storage(acc).ok_or_else(|| {
        not_implemented(format!("{OP}: no torch dtype for the accumulate type {acc:?}"))
    })?;

    let widen = |t: &Tensor| t.to_dtype(acc).and_then(|t| t.contiguous());
    let q = widen(query.tensor()?).map_err(|e| candle_err(OP, e))?;
    let k = repeat_kv_heads(OP, &widen(key.tensor()?).map_err(|e| candle_err(OP, e))?, q.dims()[1])?;
    let v = repeat_kv_heads(
        OP,
        &widen(value.tensor()?).map_err(|e| candle_err(OP, e))?,
        q.dims()[1],
    )?;

    let head_dim = q.dims()[3];
    let scale = scale.unwrap_or_else(|| 1.0 / (head_dim as f64).sqrt());

    let mut scores = k
        .transpose(2, 3)
        .and_then(|kt| kt.contiguous())
        .and_then(|kt| q.matmul(&kt))
        .and_then(|s| s.affine(scale, 0.0))
        .map_err(|e| candle_err(OP, e))?;

    let (rows, cols) = {
        let dims = scores.dims();
        (dims[2], dims[3])
    };
    if is_causal {
        // Upper-left aligned, per the measurement above.
        let mut mask = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                mask.push(if c <= r { 0.0f64 } else { f64::NEG_INFINITY });
            }
        }
        let mask = Tensor::from_vec(mask, (rows, cols), scores.device())
            .and_then(|t| t.to_dtype(acc))
            .map_err(|e| candle_err(OP, e))?;
        scores = scores.broadcast_add(&mask).map_err(|e| candle_err(OP, e))?;
    }
    if let Some(mask) = attn_mask.as_ref() {
        let mask = mask
            .tensor()?
            .to_dtype(acc)
            .map_err(|e| candle_err(OP, e))?;
        scores = scores.broadcast_add(&mask).map_err(|e| candle_err(OP, e))?;
    }

    // Softmax written out: candle-core has no `softmax` (that lives in
    // candle-nn, which DESIGN.md §4 does not pull in). Shifting by the row
    // maximum first is not an optimisation -- without it a masked row's
    // `exp(-inf)` and a large logit's `exp(big)` land on the same NaN.
    let row_max = scores.max_keepdim(3).map_err(|e| candle_err(OP, e))?;
    let weights = scores
        .broadcast_sub(&row_max)
        .and_then(|s| s.exp())
        .map_err(|e| candle_err(OP, e))?;
    let row_sum = weights.sum_keepdim(3).map_err(|e| candle_err(OP, e))?;
    let out = weights
        .broadcast_div(&row_sum)
        .and_then(|p| p.contiguous())
        .and_then(|p| p.matmul(&v))
        .and_then(|o| o.to_dtype(storage))
        .map_err(|e| candle_err(OP, e))?;

    // logsumexp(x) = max(x) + log(sum(exp(x - max(x)))), on the same masked,
    // scaled scores the weights came from.
    let logsumexp = row_sum
        .log()
        .and_then(|l| l.broadcast_add(&row_max))
        .and_then(|l| l.squeeze(3))
        .map_err(|e| candle_err(OP, e))?;

    // Promoted element by element: `promote` at the dispatcher's exit does not
    // look inside a tuple, the same reason `max.dim` promotes its own pair.
    let pair = [
        crate::tensor::promote(py, finish(py, out, tag)?)?,
        crate::tensor::promote(py, finish(py, logsumexp, acc_tag)?)?,
    ];
    Ok(PyTuple::new(py, pair)?.into_any().unbind())
}

// ---------------------------------------------------------------------------
// The ops C_SURFACE.md measured a Llama forward + greedy `generate` actually
// calling. That document counted 13 of `_VariableFunctions`' 609 hoisted names
// (2.1%) as *called* rather than merely looked up, and this is that list.
//
// They are here because the overload resolver above now routes `torch.<op>` to
// them; before it existed the only reachable spelling was
// `torch.ops.aten.<op>.<overload>`, which no user-facing code writes.
//
// Every dtype rule below was measured against torch 2.13.0, not inferred. The
// ones that would have been got wrong by inference are called out at the site.
// ---------------------------------------------------------------------------

/// Which of the three `arange` overloads is being served. They differ only in
/// which of `start`/`step` the caller supplied, so one body covers all three
/// -- but the *key* stays distinct, because torch really does send
/// `arange(0, 5)` to `arange.start` and `arange(0, 5, 2)` to
/// `arange.start_step` (measured), and collapsing them would make the work
/// queue report an op that was never called.
#[derive(Clone, Copy)]
enum ArangeForm {
    End,
    Start,
    StartStep,
}

fn arange(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    form: ArangeForm,
) -> PyResult<Py<PyAny>> {
    let (op, start_at, end_at, step_at, options_at) = match form {
        ArangeForm::End => ("aten.arange.default", None, 0usize, None, 1usize),
        ArangeForm::Start => ("aten.arange.start", Some(0usize), 1, None, 2),
        ArangeForm::StartStep => ("aten.arange.start_step", Some(0usize), 1, Some(2usize), 3),
    };

    let start = match start_at {
        Some(index) => scalar_arg(op, args, kwargs, index, "start")?.unwrap_or(Scalar::Int(0)),
        None => Scalar::Int(0),
    };
    let end = scalar_arg(op, args, kwargs, end_at, "end")?
        .ok_or_else(|| missing(op, "end"))?;
    let step = match step_at {
        Some(index) => scalar_arg(op, args, kwargs, index, "step")?.unwrap_or(Scalar::Int(1)),
        None => Scalar::Int(1),
    };

    // torch: an integral start/end/step gives int64, anything else gives the
    // default float dtype. Not "the widest of the three" -- the categories are
    // what matter, so a single float argument floats the whole result.
    let integral = start.is_int() && end.is_int() && step.is_int();
    let dtype = dtype_arg(args, kwargs, options_at, "dtype")?
        .unwrap_or(if integral { TorchDType::Int64 } else { default_float() });
    reject_unsupported(
        op,
        args,
        kwargs,
        &[(options_at + 1, "layout"), (options_at + 3, "pin_memory")],
    )?;
    let label = device_arg_or_label(args, kwargs, options_at + 2, "device", &PyDevice::cpu())?;
    if label.is_meta() {
        // Length before dtype: `arange` is the one factory whose shape is
        // computed rather than given, and `arange_length` is the same
        // arithmetic the dense path below does -- shared, not restated, so the
        // two cannot disagree about how many elements the real call would have
        // produced.
        //
        // `arange_has_cpu_kernel` is deliberately *not* consulted here. It
        // reproduces an upstream CPU gap, and upstream's meta kernel does not
        // have that gap: `torch.arange(5, dtype=torch.uint16, device="meta")`
        // is a tensor while the CPU spelling raises. Measured.
        let n = arange_length(op, &start, &end, &step, integral)?;
        return meta_result(py, vec![n], dtype);
    }
    let device = label.resolve()?;
    let storage = PyDtype::new(dtype).storage(op)?;

    // torch has no `arange_cpu` kernel for these, and the golden harness
    // caught the shim computing an answer where torch refuses. Reproducing an
    // upstream *gap* rather than filling it is the same call docs/IMPORT_TORCH
    // §7 made for `full`'s numel==1 hole: the harness compares against torch,
    // so a shim that is more capable than torch diverges just as loudly as one
    // that is less capable, only in the other direction. Measured, not
    // reasoned: torch 2.13.0 refuses uint16/uint32/uint64, bool and the float8
    // family, and accepts everything else this shim can store.
    if !arange_has_cpu_kernel(dtype) {
        return Err(not_implemented(format!(
            "\"arange_cpu\" not implemented for '{}'",
            scalar_type_name(dtype)
        )));
    }

    // The zero-step and sign refusals live in `arange_length` -- one copy, so
    // the meta path cannot skip a check the dense path makes.
    let (s, d) = (start.as_f64(), step.as_f64());
    let n = arange_length(op, &start, &end, &step, integral)? as i64;
    let tensor = if integral {
        let (s, d) = (start.as_i64(), step.as_i64());
        // `s + i * d`, never an accumulator: candle's own `arange_step` adds
        // repeatedly, which drifts on floats and is the kind of divergence the
        // golden harness exists to catch.
        let values: Vec<i64> = (0..n).map(|i| s + i * d).collect();
        let len = values.len();
        Tensor::from_vec(values, len, &device)
    } else {
        let values: Vec<f64> = (0..n).map(|i| s + (i as f64) * d).collect();
        let len = values.len();
        Tensor::from_vec(values, len, &device)
    }
    .and_then(|t| t.to_dtype(storage))
    .map_err(|err| candle_err(op, err))?;

    finish(py, tensor, dtype)
}

/// How many elements `arange(start, end, step)` produces.
///
/// Split out of the kernel because the meta path needs the count without the
/// values, and a second copy of this arithmetic is exactly the kind of thing
/// that drifts: the integral branch is not `ceil((e-s)/d)` in floating point,
/// it is an integer ceiling-divide, and the two disagree at the boundaries the
/// golden harness already pins.
///
/// The sign and zero-step checks stay in the kernel: they are refusals, and a
/// meta call has to make the same ones before it can claim a shape.
fn arange_length(
    op: &str,
    start: &Scalar,
    end: &Scalar,
    step: &Scalar,
    integral: bool,
) -> PyResult<usize> {
    if step.as_f64() == 0.0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err("step must be nonzero"));
    }
    let (fs, fe, fd) = (start.as_f64(), end.as_f64(), step.as_f64());
    if (fd > 0.0 && fe < fs) || (fd < 0.0 && fe > fs) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "upper bound and lower bound inconsistent with step sign",
        ));
    }
    let _ = op;
    let n = if integral {
        let (s, e, d) = (start.as_i64(), end.as_i64(), step.as_i64());
        if d > 0 {
            (e - s + d - 1).div_euclid(d).max(0)
        } else {
            (s - e + (-d) - 1).div_euclid(-d).max(0)
        }
    } else {
        (((fe - fs) / fd).ceil()).max(0.0) as i64
    };
    Ok(n as usize)
}

/// `aten::ones(SymInt[] size, *, ScalarType? dtype=None, ...)` and
/// `aten::empty.memory_format(SymInt[] size, *, ...)`.
///
/// **`empty` returns zeros, and that is a divergence worth stating.** torch's
/// `empty` hands back whatever was in the allocation, and makes no promise
/// about it, so zeros satisfy the contract -- but they are not the *same*
/// bytes, and any test that reads an uninitialised tensor and compares against
/// torch is comparing noise. The shim is deterministic here where torch is
/// not, which is the safe direction but still a difference.
fn zeros_or_ones(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    one: bool,
) -> PyResult<Py<PyAny>> {
    let size: Vec<usize> = required(op, args, kwargs, 0, "size")?.extract()?;
    let dtype = dtype_arg(args, kwargs, 1, "dtype")?.unwrap_or(default_float());
    reject_unsupported(
        op,
        args,
        kwargs,
        &[(2, "layout"), (4, "pin_memory"), (5, "memory_format")],
    )?;
    let label = device_arg_or_label(args, kwargs, 3, "device", &PyDevice::cpu())?;
    // `meta` before `storage()`, and the order is the claim: a meta tensor
    // never has to be storable by candle, so `torch.empty(2,
    // dtype=torch.complex64, device="meta")` is representable here exactly as
    // it is upstream, while its CPU counterpart is refused by name.
    if label.is_meta() {
        return meta_result(py, size, dtype);
    }
    let device = label.resolve()?;
    let storage = PyDtype::new(dtype).storage(op)?;

    let tensor = if one {
        Tensor::ones(size, storage, &device)
    } else {
        Tensor::zeros(size, storage, &device)
    }
    .map_err(|err| candle_err(op, err))?;
    finish(py, tensor, dtype)
}

fn ones_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    zeros_or_ones(py, args, kwargs, "aten.ones.default", true)
}

fn empty_memory_format(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    zeros_or_ones(py, args, kwargs, "aten.empty.memory_format", false)
}

/// `aten::rsqrt(Tensor self) -> Tensor`
///
/// The dtype rule is torch's unary-float promotion: a floating input keeps its
/// own dtype (`float16` in, `float16` out -- *not* widened to float32), and an
/// integral or boolean input becomes the default float. Both halves measured.
fn rsqrt_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.rsqrt.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = if input.tag().is_floating_point() {
        input.tag()
    } else {
        default_float()
    };
    let storage = PyDtype::new(tag).storage(OP)?;
    let tensor = input
        .tensor()?
        .to_dtype(storage)
        .and_then(|t| t.sqrt())
        .and_then(|t| t.recip())
        .map_err(|err| candle_err(OP, err))?;
    finish(py, tensor, tag)
}

/// `aten::pow.Tensor_Scalar`, `.Tensor_Tensor` and `.Scalar`.
///
/// Not candle's `Tensor::pow`, which is `exp(exponent * log(base))` and so
/// returns NaN for every negative base -- `torch.pow(t, 2)` on a tensor with
/// negative entries is the RMSNorm path, so that would be wrong on the first
/// real model. `powf` (used for the float scalar case) does go through the
/// real `f64::powf` and is fine; the general case is computed here.
///
/// The dtype rule is torch's "wrapped number" promotion, measured: an integer
/// tensor with an integer exponent stays integral (`pow(int64, 2) -> int64`),
/// and a float on either side floats the result. A Python scalar never widens
/// a tensor of the same category.
fn pow_result_tag(op: &str, tensor: TorchDType, scalar_is_float: bool) -> PyResult<TorchDType> {
    if tensor == TorchDType::Bool {
        return Err(not_implemented(format!(
            "{op}: torch.bool operands are not implemented in torch._C shim -- \
             torch's own result category for a boolean pow has not been measured, \
             and guessing it is exactly the silent divergence this shim refuses"
        )));
    }
    Ok(if scalar_is_float && !tensor.is_floating_point() {
        default_float()
    } else {
        tensor
    })
}

fn pow_from_pairs(
    py: Python<'_>,
    op: &str,
    bases: PowSide,
    exponents: PowSide,
    shape: Vec<usize>,
    tag: TorchDType,
    device: &Device,
) -> PyResult<Py<PyAny>> {
    let storage = PyDtype::new(tag).storage(op)?;
    let tensor = if tag.is_floating_point() {
        let (b, e) = (bases.as_f64(), exponents.as_f64());
        let n = b.len().max(e.len());
        let values: Vec<f64> = (0..n)
            .map(|i| b[i % b.len()].powf(e[i % e.len()]))
            .collect();
        Tensor::from_vec(values, shape, device)
    } else {
        let (b, e) = (bases.as_i64(), exponents.as_i64());
        let n = b.len().max(e.len());
        let mut values = Vec::with_capacity(n);
        for i in 0..n {
            let exponent = e[i % e.len()];
            if exponent < 0 {
                // torch's message, verbatim.
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "Integers to negative integer powers are not allowed.",
                ));
            }
            // Wrapping, like torch's integer kernels: an int64 overflow there
            // wraps rather than raising, and refusing here would diverge in
            // the other direction.
            values.push(b[i % b.len()].wrapping_pow(exponent.min(u32::MAX as i64) as u32));
        }
        Tensor::from_vec(values, shape, device)
    }
    .and_then(|t| t.to_dtype(storage))
    .map_err(|err| candle_err(op, err))?;
    finish(py, tensor, tag)
}

/// The values of one side of a `pow`, already flattened to the broadcast shape
/// (or a single element, which the caller cycles).
enum PowSide {
    Floats(Vec<f64>),
    Ints(Vec<i64>),
}

impl PowSide {
    fn as_f64(&self) -> Vec<f64> {
        match self {
            PowSide::Floats(v) => v.clone(),
            PowSide::Ints(v) => v.iter().map(|&x| x as f64).collect(),
        }
    }

    fn as_i64(&self) -> Vec<i64> {
        match self {
            PowSide::Floats(v) => v.iter().map(|&x| x as i64).collect(),
            PowSide::Ints(v) => v.clone(),
        }
    }
}

fn pow_tensor_scalar(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.pow.Tensor_Scalar";
    let base = tensor_arg(OP, args, kwargs, 0, "self")?;
    let exponent = scalar_arg(OP, args, kwargs, 1, "exponent")?
        .ok_or_else(|| missing(OP, "exponent"))?;
    let tag = pow_result_tag(OP, base.tag(), !exponent.is_int())?;
    let shape = base.tensor()?.dims().to_vec();
    let bases = side_from_tensor(OP, base.tensor()?, tag)?;
    let exponents = side_from_scalar(&exponent, tag);
    pow_from_pairs(py, OP, bases, exponents, shape, tag, base.tensor()?.device())
}

fn pow_scalar(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.pow.Scalar";
    let base = scalar_arg(OP, args, kwargs, 0, "self")?.ok_or_else(|| missing(OP, "self"))?;
    let exponent = tensor_arg(OP, args, kwargs, 1, "exponent")?;
    let tag = pow_result_tag(OP, exponent.tag(), !base.is_int())?;
    let shape = exponent.tensor()?.dims().to_vec();
    let bases = side_from_scalar(&base, tag);
    let exponents = side_from_tensor(OP, exponent.tensor()?, tag)?;
    pow_from_pairs(py, OP, bases, exponents, shape, tag, exponent.tensor()?.device())
}

fn pow_tensor_tensor(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.pow.Tensor_Tensor";
    let base = tensor_arg(OP, args, kwargs, 0, "self")?;
    let exponent = tensor_arg(OP, args, kwargs, 1, "exponent")?;
    let tag = pow_result_tag(OP, same_dtype(OP, &base, &exponent)?, false)?;

    let shape = base
        .tensor()?
        .shape()
        .broadcast_shape_binary_op(exponent.tensor()?.shape(), "pow")
        .map_err(|err| candle_err(OP, err))?;
    let dims = shape.dims().to_vec();
    let broadcast = |t: &Tensor| -> PyResult<Tensor> {
        t.broadcast_as(shape.clone())
            .and_then(|t| t.contiguous())
            .map_err(|err| candle_err(OP, err))
    };
    let bases = side_from_tensor(OP, &broadcast(base.tensor()?)?, tag)?;
    let exponents = side_from_tensor(OP, &broadcast(exponent.tensor()?)?, tag)?;
    pow_from_pairs(py, OP, bases, exponents, dims, tag, base.tensor()?.device())
}

fn side_from_tensor(op: &str, tensor: &Tensor, tag: TorchDType) -> PyResult<PowSide> {
    let flat = tensor.flatten_all().map_err(|err| candle_err(op, err))?;
    if tag.is_floating_point() {
        Ok(PowSide::Floats(
            flat.to_dtype(candle_core::DType::F64)
                .and_then(|t| t.to_vec1::<f64>())
                .map_err(|err| candle_err(op, err))?,
        ))
    } else {
        Ok(PowSide::Ints(
            flat.to_dtype(candle_core::DType::I64)
                .and_then(|t| t.to_vec1::<i64>())
                .map_err(|err| candle_err(op, err))?,
        ))
    }
}

fn side_from_scalar(value: &Scalar, tag: TorchDType) -> PowSide {
    if tag.is_floating_point() {
        PowSide::Floats(vec![value.as_f64()])
    } else {
        PowSide::Ints(vec![value.as_i64()])
    }
}

/// `aten::cat(Tensor[] tensors, int dim=0) -> Tensor`
///
/// **A tensor of shape exactly `(0,)` is skipped, not concatenated.** This is
/// torch's "legacy empty" rule, and it is not a corner case anyone can avoid:
/// `transformers`' KV cache starts every layer as `torch.tensor([])` and grows
/// it with `torch.cat([self.keys, key_states], dim=-2)`
/// (`cache_utils.py:144`), so the *first* decoder step of every model is a cat
/// of a 1-D empty against a 4-D tensor. Without the rule this shim raised
/// `IndexError: Dimension out of range` and the forward pass stopped there
/// (docs/E2E_REAL.md).
///
/// The rule is narrow, and measured on 2.13.0 rather than inferred:
///
///   * only shape `(0,)` qualifies. `torch.ones(0, 5)` does **not** -- it
///     raises `Tensors must have same number of dimensions: got 2 and 4`,
///     so "empty" is not the test, `(0,)` is.
///   * a skipped entry takes no part in the rank check, the `dim` check or
///     the extent check -- `cat([tensor([]), ones(1,2,3,4)], dim=-2)` is
///     `(1, 2, 3, 4)`, while a *non*-empty 1-D entry in the same position
///     still raises `IndexError` on `dim=-2`.
///   * when every entry is skipped the result is `(0,)`, whatever `dim` was;
///     `cat([tensor([]), tensor([])], dim=5)` is `(0,)` rather than an
///     `IndexError`.
///
/// It does still take part in **dtype**: upstream promotes, so
/// `cat([int64 (0,), int32 (2,3)])` is `int64` rather than `int32`. This shim
/// refuses mixed dtypes here as it did before -- promotion is a separate gap
/// with its own refusal, and skipping the shape while silently dropping the
/// dtype would have been a third behaviour belonging to neither.
fn cat_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.cat.default";
    let tensors: Vec<PyTensorBase> = required(OP, args, kwargs, 0, "tensors")?.extract()?;
    if tensors.is_empty() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "torch.cat(): expected a non-empty list of Tensors",
        ));
    }
    let tag = tensors[0].tag();
    for other in &tensors[1..] {
        if other.tag() != tag {
            return Err(not_implemented(format!(
                "{OP}: dtype promotion not implemented in torch._C shim: {} vs {}",
                tag.name(),
                other.tag().name()
            )));
        }
    }

    // The legacy-empty partition, before anything reads a rank.
    let mut kept: Vec<&Tensor> = Vec::with_capacity(tensors.len());
    for t in &tensors {
        let inner = t.tensor()?;
        if inner.dims() != [0] {
            kept.push(inner);
        }
    }
    if kept.is_empty() {
        // Every entry was `(0,)`. Upstream hands back a `(0,)` of the same
        // dtype without ever looking at `dim`.
        let storage = PyDtype::new(tag).storage(OP)?;
        let out = Tensor::from_vec(Vec::<f64>::new(), 0usize, tensors[0].tensor()?.device())
            .and_then(|t| t.to_dtype(storage))
            .map_err(|err| candle_err(OP, err))?;
        return finish(py, out, tag);
    }

    let rank = kept[0].rank();
    let dim = normalise_dim(OP, dim_arg(args, kwargs, 1, "dim")?.unwrap_or(0), rank)?;
    let tensor = Tensor::cat(&kept, dim).map_err(|err| candle_err(OP, err))?;
    finish(py, tensor, tag)
}

/// `aten::stack(Tensor[] tensors, int dim=0) -> Tensor`
///
/// **Not `cat` with a different name.** `cat` joins along an existing axis and
/// lets the extents differ there; `stack` inserts a *new* axis and therefore
/// requires every entry to have exactly the same size, rank included. Measured:
/// `stack([zeros(2), zeros(2,1)])` raises `stack expects each tensor to be
/// equal size, but got [2] at entry 0 and [2, 1] at entry 1`, where the
/// corresponding `cat` is a legal call on a different axis. Routing this key at
/// `cat_default` would compute a different op.
///
/// Reached by GPT-J's rotary embedding, which is the only architecture of the
/// four this op was added for that calls it: `stack([x1, x2], dim=-1)` on a
/// pair of `(batch, seq, heads, dim/2)` slices, then a flatten. docs/ARCH.md
/// counts three more callers (`cohere`, `helium`, `mamba`).
///
/// Three rules were measured rather than inferred, and two of them differ from
/// `cat`:
///
///   * **`dim` runs to `rank`, not `rank - 1`.** The new axis can go after the
///     last existing one, so the legal range is `[-(rank+1), rank]` -- the same
///     widened range `unsqueeze` has and `cat` does not. `stack([a, b], 1)` on
///     two 1-D tensors is legal; `cat` at `dim=1` on the same pair is not.
///   * **an empty list is refused** (`stack expects a non-empty TensorList`),
///     because there is no shape to invent for the result. `cat` refuses too,
///     with different wording.
///   * **upstream promotes dtypes here** -- `stack([bool, int64])` gives
///     `int64` and `stack([int64, float32])` gives `float32`, both measured.
///     This shim refuses instead, the same way `cat_default` and `same_dtype`
///     do and for the reason written at `same_dtype`. The four architectures
///     this op was added for never mix (GPT-J stacks two `float32` halves of
///     one tensor), so the refusal costs nothing measured; the golden cases
///     record the promotion as `c_error` so it stays visible as a gap rather
///     than being forgotten.
///
/// Entries are made contiguous first. Upstream accepts a non-contiguous entry
/// and answers a contiguous result (measured on a transposed input), and
/// candle's `cat` -- which `Tensor::stack` reaches after unsqueezing -- wants
/// contiguous inputs.
fn stack_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.stack.default";
    let tensors: Vec<PyTensorBase> = required(OP, args, kwargs, 0, "tensors")?.extract()?;
    if tensors.is_empty() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "stack expects a non-empty TensorList",
        ));
    }
    let tag = tensors[0].tag();
    for other in &tensors[1..] {
        if other.tag() != tag {
            return Err(not_implemented(format!(
                "{OP}: dtype promotion not implemented in torch._C shim: {} vs {}",
                tag.name(),
                other.tag().name()
            )));
        }
    }
    let first = tensors[0].tensor()?.dims().to_vec();
    for (index, other) in tensors.iter().enumerate().skip(1) {
        if other.tensor()?.dims() != first.as_slice() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "stack expects each tensor to be equal size, but got {:?} at entry 0 \
                 and {:?} at entry {index}",
                first,
                other.tensor()?.dims()
            )));
        }
    }

    // The widened range: `normalise_dim` clamps to `rank.max(1)`, which is the
    // right rule for an *existing* axis and one short for a new one. A 0-D
    // entry stacks at `dim=0` only, and `stack([tensor(1.), tensor(2.)], 1)`
    // is an IndexError upstream -- so the extent is `rank + 1`, uniformly.
    let raw = dim_arg(args, kwargs, 1, "dim")?.unwrap_or(0);
    let extent = first.len() as isize + 1;
    let dim = if raw < 0 { raw + extent } else { raw };
    if dim < 0 || dim >= extent {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "{OP}: Dimension out of range (expected to be in range of [{}, {}], but got {raw})",
            -extent,
            extent - 1
        )));
    }

    let contiguous: Vec<Tensor> = tensors
        .iter()
        .map(|t| t.tensor()?.contiguous().map_err(|e| candle_err(OP, e)))
        .collect::<PyResult<_>>()?;
    let tensor = Tensor::stack(&contiguous, dim as usize).map_err(|e| candle_err(OP, e))?;
    finish(py, tensor, tag)
}

/// `aten::scalar_tensor(Scalar s, *, ScalarType? dtype=None, Layout? layout=None,
///     Device? device=None, bool? pin_memory=None) -> Tensor`
///
/// A 0-D tensor holding one number. It is how `falcon`, `gptj`, `bloom` and
/// `mpt` build their attention mask fill value -- all four call
/// `scalar_tensor(finfo(dtype).min, dtype=..., device=...)` and hand the result
/// straight to `where.self` (measured, docs/OPS4.md §1).
///
/// **The dtype rule is not `full`'s, and inferring it from `full` would be
/// wrong.** `full` reads the fill value's category (`full([], 3)` is `int64`,
/// `full([], True)` is `bool`); `scalar_tensor` ignores it entirely and always
/// answers the default float:
///
/// ```text
/// scalar_tensor(3)      -> float32   (full([], 3)     -> int64)
/// scalar_tensor(True)   -> float32   (full([], True)  -> bool)
/// scalar_tensor(1.5)    -> float32
/// ```
///
/// Measured on torch 2.13.0 over int, bool, float, `nan` and `inf`. A shim that
/// shared `full`'s inference would give `int64` where torch gives `float32`,
/// and the mask value would then be silently truncated.
///
/// The overflow rule *is* `full`'s, including its numel==1 hole, and that too
/// is measured rather than assumed: `scalar_tensor(1e6, float16)` is `inf`
/// while `scalar_tensor(1e300, float32)` raises, which is exactly what
/// `checked_convert` at `numel = 1` reproduces (only the reduced-precision
/// floats take the unchecked fast path). `scalar_tensor(-1, uint8)` is `255`
/// and `scalar_tensor(300, uint8)` raises -- the two's-complement wrap
/// allowance, same as `full`.
///
/// `layout=torch.strided` is accepted rather than refused. `reject_unsupported`
/// would turn it away, and the measured call sites pass it explicitly (`bloom`,
/// `mpt` and `gptj` all send `layout=torch.strided`), so refusing it would
/// block the four architectures this op exists to open on an argument that
/// names the only layout the shim has. Any other layout still refuses.
fn scalar_tensor_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.scalar_tensor.default";
    let raw = required(OP, args, kwargs, 0, "s")?;
    let value = scalar_arg(OP, args, kwargs, 0, "s")?.ok_or_else(|| missing(OP, "s"))?;
    let dtype = dtype_arg(args, kwargs, 1, "dtype")?.unwrap_or(default_float());
    reject_layout(OP, args, kwargs, 2)?;
    reject_unsupported(OP, args, kwargs, &[(4, "pin_memory")])?;
    let label = device_arg_or_label(args, kwargs, 3, "device", &PyDevice::cpu())?;

    // A 0-D tensor is one element, so the numel==1 branch of the upstream
    // check is the one that applies -- see the note above.
    if !raw.is_instance_of::<PyTensorBase>() {
        checked_convert(&raw, raw.is_instance_of::<pyo3::types::PyInt>(), dtype, 1)?;
    }
    if label.is_meta() {
        return meta_result(py, Vec::new(), dtype);
    }
    let device = label.resolve()?;

    if dtype == TorchDType::Bool {
        let truthy = u8::from(value.as_f64() != 0.0);
        let tensor = Tensor::full(truthy, (), &device).map_err(|e| candle_err(OP, e))?;
        return finish(py, tensor, dtype);
    }
    let storage = PyDtype::new(dtype).storage(OP)?;
    let tensor = if storage.is_int() {
        // Upstream truncates toward zero rather than rounding:
        // `scalar_tensor(-1.5, dtype=int64)` is `-1`, measured.
        Tensor::full(value.as_i64(), (), &device)
    } else {
        Tensor::full(value.as_f64(), (), &device)
    }
    .and_then(|t| t.to_dtype(storage))
    .map_err(|e| candle_err(OP, e))?;
    finish(py, tensor, dtype)
}

/// `aten::argmax(Tensor self, int? dim=None, bool keepdim=False) -> Tensor`
///
/// `dim=None` flattens first, and `keepdim=True` alongside it gives shape
/// `[1]` rather than `[]` -- measured, and not what "keep the reduced
/// dimension" suggests when there was no named dimension to keep.
///
/// The result is int64. candle's `argmax` yields `u32`, which would be a
/// visible dtype divergence on the very first `generate()` step.
fn argmax_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.argmax.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let dim = dim_arg(args, kwargs, 1, "dim")?;
    let keepdim = bool_arg(args, kwargs, 2, "keepdim")?.unwrap_or(false);

    let tensor = match dim {
        None => {
            let flat = input.tensor()?.flatten_all().map_err(|e| candle_err(OP, e))?;
            let reduced = flat.argmax(0).map_err(|e| candle_err(OP, e))?;
            if keepdim {
                reduced.reshape(1).map_err(|e| candle_err(OP, e))?
            } else {
                reduced
            }
        }
        Some(dim) => {
            let dim = normalise_dim(OP, dim, input.tensor()?.rank())?;
            if keepdim {
                input.tensor()?.argmax_keepdim(dim)
            } else {
                input.tensor()?.argmax(dim)
            }
            .map_err(|e| candle_err(OP, e))?
        }
    };
    let tensor = tensor
        .to_dtype(candle_core::DType::I64)
        .map_err(|e| candle_err(OP, e))?;
    finish(py, tensor, TorchDType::Int64)
}

/// `aten::embedding(Tensor weight, Tensor indices, SymInt padding_idx=-1,
///                  bool scale_grad_by_freq=False, bool sparse=False)`
///
/// candle's `Tensor::embedding` demands rank-1 indices; torch takes any shape
/// and appends the embedding dimension, which is the shape `transformers`
/// relies on (`[batch, seq]` in, `[batch, seq, hidden]` out).
///
/// The three trailing arguments are backward-only in torch -- the forward
/// result does not depend on any of them. `padding_idx` is therefore accepted
/// and ignored, which is what upstream's forward does; the other two are
/// refused, because switching on a gradient behaviour behind a shim with no
/// autograd would be claiming something.
fn embedding_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.embedding.default";
    let weight = tensor_arg(OP, args, kwargs, 0, "weight")?;
    let indices = tensor_arg(OP, args, kwargs, 1, "indices")?;
    for (index, name) in [(3, "scale_grad_by_freq"), (4, "sparse")] {
        if bool_arg(args, kwargs, index, name)?.unwrap_or(false) {
            return Err(not_implemented(format!(
                "{OP}: argument '{name}' only affects the backward pass, and there \
                 is no autograd in torch._C shim"
            )));
        }
    }

    if weight.tensor()?.rank() != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{OP}: weight must be 2D, got {}D",
            weight.tensor()?.rank()
        )));
    }
    let mut shape = indices.tensor()?.dims().to_vec();
    shape.push(weight.tensor()?.dims()[1]);

    let flat = indices
        .tensor()?
        .flatten_all()
        .and_then(|t| t.contiguous())
        .map_err(|e| candle_err(OP, e))?;
    let tensor = weight
        .tensor()?
        .index_select(&flat, 0)
        .and_then(|t| t.reshape(shape))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, tensor, weight.tag())
}

/// `aten::is_floating_point(Tensor self) -> bool`
///
/// The only op here that answers from the tag alone. Upstream's
/// `torch.is_floating_point` does not reach the dispatcher at all (measured
/// with a `TorchDispatchMode` logger: the call produces no aten record), but
/// this shim routes it through the one door anyway, so that the surface has a
/// single entrance rather than one entrance and one shortcut.
fn is_floating_point_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.is_floating_point.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    Ok(input.tag().is_floating_point().into_bound_py_any(py)?.unbind())
}

/// `aten::isin.Tensor_Tensor(Tensor elements, Tensor test_elements, *,
///                           bool assume_unique=False, bool invert=False)`
///
/// Result is `torch.bool` shaped like `elements`. `assume_unique` is a
/// performance hint with no effect on the answer, so it is accepted; `invert`
/// negates.
fn isin_tensor_tensor(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.isin.Tensor_Tensor";
    let elements = tensor_arg(OP, args, kwargs, 0, "elements")?;
    let test = tensor_arg(OP, args, kwargs, 1, "test_elements")?;
    let tag = same_dtype(OP, &elements, &test)?;
    let invert = bool_arg(args, kwargs, 3, "invert")?.unwrap_or(false);

    // Compared as f64 when either side is floating, as i64 otherwise. Equality
    // is exact in both, since the two operands share a dtype -- there is no
    // promotion step that could round one side onto the other.
    let (haystack, needles) = if tag.is_floating_point() {
        (
            side_from_tensor(OP, elements.tensor()?, tag)?.as_f64(),
            side_from_tensor(OP, test.tensor()?, tag)?.as_f64(),
        )
    } else {
        (
            side_from_tensor(OP, elements.tensor()?, tag)?
                .as_i64()
                .into_iter()
                .map(|v| v as f64)
                .collect(),
            side_from_tensor(OP, test.tensor()?, tag)?
                .as_i64()
                .into_iter()
                .map(|v| v as f64)
                .collect(),
        )
    };

    let bytes: Vec<u8> = haystack
        .iter()
        .map(|value| u8::from(needles.iter().any(|n| n == value) != invert))
        .collect();
    let tensor = Tensor::from_vec(bytes, elements.tensor()?.dims().to_vec(), elements.tensor()?.device())
        .map_err(|e| candle_err(OP, e))?;
    finish(py, tensor, TorchDType::Bool)
}

/// `aten::lift_fresh(Tensor(a) self) -> Tensor(a)`
///
/// Identity, and that is the whole op upstream too: it marks a constant as
/// having entered the graph. It is here because it is the *only* aten call
/// `torch.tensor([...])` makes -- measured -- so `torch.tensor` reaching the
/// dispatcher at all depends on it existing.
fn lift_fresh_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.lift_fresh.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    Ok(input.into_pyobject(py)?.into_any().unbind())
}

/// `aten::randint.low(SymInt low, SymInt high, SymInt[] size, *,
///                    ScalarType? dtype=4, ...)` and `aten::randint(...)`.
///
/// `dtype=4` in the schema is `ScalarType::Long`, so the default is int64
/// rather than the default float every other factory here uses.
///
/// **The generator is candle's, not torch's, so the *values* will not match a
/// seeded torch run.** There is no seed plumbing in this shim, and inventing
/// one that claims to reproduce torch's Philox stream would be a lie a test
/// could not see through. What is reproduced is the range, the shape and the
/// dtype.
fn randint(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    with_low: bool,
) -> PyResult<Py<PyAny>> {
    let (op, low, high_at, size_at) = if with_low {
        (
            "aten.randint.low",
            int_arg(args, kwargs, 0, "low")?.ok_or_else(|| missing("aten.randint.low", "low"))?,
            1usize,
            2usize,
        )
    } else {
        ("aten.randint.default", 0i64, 0usize, 1usize)
    };
    let high = int_arg(args, kwargs, high_at, "high")?.ok_or_else(|| missing(op, "high"))?;
    let size: Vec<usize> = required(op, args, kwargs, size_at, "size")?.extract()?;
    let options_at = size_at + 1;
    let dtype = dtype_arg(args, kwargs, options_at, "dtype")?.unwrap_or(TorchDType::Int64);
    reject_unsupported(
        op,
        args,
        kwargs,
        &[(options_at + 1, "layout"), (options_at + 3, "pin_memory")],
    )?;
    let label = device_arg_or_label(args, kwargs, options_at + 2, "device", &PyDevice::cpu())?;

    if high <= low {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "random_ expects 'from' to be less than 'to', but got from={low} >= to={high}"
        )));
    }
    // After the bound check, before the allocation: upstream's meta kernel
    // raises for `high <= low` too, so the meta answer is not a way around a
    // check the real call makes.
    if label.is_meta() {
        return meta_result(py, size, dtype);
    }
    let device = label.resolve()?;
    let storage = PyDtype::new(dtype).storage(op)?;
    let span = (high - low) as f64;
    let tensor = Tensor::rand(0f64, 1f64, size, &device)
        .and_then(|t| t.affine(span, low as f64))
        .and_then(|t| t.floor())
        // `rand` is half-open in principle but the affine can land exactly on
        // `high` after rounding; the clamp keeps the half-open contract that
        // callers actually rely on.
        .and_then(|t| t.clamp(low as f64, (high - 1) as f64))
        .and_then(|t| t.to_dtype(storage))
        .map_err(|e| candle_err(op, e))?;
    finish(py, tensor, dtype)
}

// ---------------------------------------------------------------------------
// The `TensorBase` surface
//
// docs/C_SURFACE.md §4 measured a small Llama forward plus greedy `generate`
// and found 50 of `TensorBase`'s 694 members actually used. These are the
// kernels behind that list. They are reached from `methods.json` through the
// same resolver `torch.<op>` uses and through the same single `_aten_dispatch`
// door -- there is still no arithmetic on the `TensorBase` type itself.
//
// Two rules carried over from the ops above, because they are what makes this
// shim worth having rather than fast:
//
//   * **No silent dtype promotion between two tensors.** `same_dtype` refuses
//     and names both. torch would promote; a wrong promotion is the silent
//     numerical drift DESIGN.md §5 calls candle's main risk, and a refusal is
//     a work item.
//
//     `mul.Tensor` and `bitwise_and.Tensor` are now the exceptions, and they
//     are *not* a relaxation of that rule -- they are two work items closed.
//     They promote through `promote_types`, whose table was measured cell by
//     cell against upstream and is pinned by the golden harness over every
//     storable pair. `add`/`sub`/`div`/`bitwise_or` still refuse, because no
//     measured caller needs them to promote and inventing the answer would be
//     exactly what the rule forbids. See `promote_operands` for the two
//     callers that made these two measured.
//   * **A Python scalar does not widen a tensor of the same category.** That
//     is torch's "wrapped number" rule, measured for `pow` in
//     docs/OVERLOAD.md §6.3 and re-measured here for the arithmetic ops:
//     `float_t * 2 -> float32`, `int64_t * 2 -> int64`, `int64_t * 2.0 ->
//     float32`. True division is the exception and always floats.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq)]
enum Arith {
    Add,
    Sub,
    Mul,
    Div,
}

/// The result dtype of an arithmetic op, given the tensor's dtype and (for the
/// `Scalar` overloads) whether the Python scalar was a float.
fn arith_tag(
    op: &str,
    kind: Arith,
    tensor: TorchDType,
    scalar_is_float: Option<bool>,
) -> PyResult<TorchDType> {
    // `bool * bool` is a logical and in torch, not an arithmetic product
    // (BOOL.md §2.2), and `bool + bool` is a logical or. candle would give 2
    // where both are true -- still truthy, therefore silently wrong
    // downstream. `add.Tensor` already refuses this; the rest follow.
    //
    // **`mul.Tensor` is the one exception, and it is an exception because the
    // two operations coincide rather than because it was convenient.** Under
    // the invariant BOOL.md §6.3 attaches to the `bool` tag -- the bytes are 0
    // or 1, enforced by `boolean()` being the only constructor that may attach
    // it -- an arithmetic product *is* the logical and: 1·1=1, 1·0=0, 0·0=0,
    // and no other pair of operands exists. So candle's `broadcast_mul`
    // computes torch's answer exactly, values and dtype both
    // (`tensor([T,F]) * tensor([T,T])` is `tensor([True, False])`,
    // `torch.bool`, measured on 2.13.0). That is not true of `+`, which would
    // give 2 and need clamping; `-` and `/` are refused by upstream itself.
    //
    // It is here because `torch.isfinite` needs it: upstream's own body is
    // `(self == self) * (self.abs() != inf)`, a multiply of two bool tensors,
    // and that is on the `print(tensor)` path (docs/E2E_REAL.md).
    //
    // The **scalar** overload stays refused, and that is a real difference
    // rather than caution: `bool_tensor * 2` promotes to `int64` upstream and
    // `bool_tensor * 1.5` to `float32`, so the scalar form is not a logical
    // and at all, and the promotion it needs is not implemented here.
    // `scalar_is_float.is_none()` is exactly "this is the Tensor overload".
    if tensor == TorchDType::Bool && !(kind == Arith::Mul && scalar_is_float.is_none()) {
        return Err(not_implemented(format!(
            "{op}: torch.bool operands are logical, not arithmetic, in torch \
             (BOOL.md §2.2) and are not implemented in torch._C shim"
        )));
    }
    let mut tag = tensor;
    if scalar_is_float == Some(true) && !tag.is_floating_point() {
        tag = default_float();
    }
    // torch's `/` is true division: it floats an integral pair rather than
    // truncating. `torch.tensor([1]) / torch.tensor([2])` is `0.5`, measured.
    if kind == Arith::Div && !tag.is_floating_point() {
        tag = default_float();
    }
    Ok(tag)
}

fn apply_arith(op: &str, kind: Arith, lhs: &Tensor, rhs: &Tensor) -> PyResult<Tensor> {
    match kind {
        Arith::Add => lhs.broadcast_add(rhs),
        Arith::Sub => lhs.broadcast_sub(rhs),
        Arith::Mul => lhs.broadcast_mul(rhs),
        Arith::Div => lhs.broadcast_div(rhs),
    }
    .map_err(|e| candle_err(op, e))
}

/// `alpha`, which only `add` and `sub` have. It is positional in the `Scalar`
/// schemas and keyword-only in the `Tensor` ones; `optional` covers both.
fn alpha_arg(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<f64> {
    Ok(scalar_arg(op, args, kwargs, 2, "alpha")?
        .map(|s| s.as_f64())
        .unwrap_or(1.0))
}

/// Where a dtype sits in torch's promotion lattice: the category, then the
/// width within it.
///
/// Only the dtypes `TorchDType::storage()` can hold appear; everything else
/// comes back `None` from `promote_types` below, which this shim would have
/// to refuse anyway since it cannot build the operand.
fn promotion_rank(dtype: TorchDType) -> Option<(u8, u8)> {
    use TorchDType::*;
    Some(match dtype {
        Bool => (0, 0),
        UInt8 => (1, 1),
        Int16 => (1, 2),
        Int32 => (1, 3),
        Int64 => (1, 4),
        // The two reduced floats are deliberately the SAME rank. They are not
        // ordered with respect to each other -- neither can hold the other --
        // and the tie is broken by escaping upwards, which is exactly the
        // `float16 x bfloat16 -> float32` cell.
        Float16 | BFloat16 => (2, 1),
        Float32 => (2, 2),
        Float64 => (2, 3),
        _ => return None,
    })
}

/// torch's `promote_types`, over the dtypes this shim can store.
///
/// Measured, not derived. The full 10x10 table over `{bool, uint8, uint32,
/// int16, int32, int64, float16, bfloat16, float32, float64}` was read off
/// `torch.promote_types` on 2.13.0 and separately checked cell by cell
/// against `aten.mul.Tensor`'s own result dtype -- they agree in every cell
/// where both are defined, which is why one function can serve the op.
/// Three cells break a plausible-looking shortcut:
///
/// ```text
/// int64   x float16   -> float16    an integral operand never widens a float
/// float16 x bfloat16  -> float32    two reduced floats promote OUT
/// uint8   x int16     -> int16      unsigned meets signed, no escape needed
/// ```
///
/// `uint32` (and torch's other unsigned types above `uint8`) has no promotion
/// with `bool` or a signed integer *upstream* -- "Promotion for uint16,
/// uint32, uint64 types is not supported" -- so `None` there is reproducing a
/// refusal, not admitting a gap. Against a float it does promote, and that
/// cell is answered.
fn promote_types(lhs: TorchDType, rhs: TorchDType) -> Option<TorchDType> {
    if lhs == rhs {
        return Some(lhs);
    }
    // Handled before the rank lookup, because `promotion_rank` has no entry
    // for these either and the two reasons must not be conflated: one is
    // upstream's refusal, the other would be this shim's.
    let unsigned_wide = |d: TorchDType| {
        matches!(
            d,
            TorchDType::UInt16 | TorchDType::UInt32 | TorchDType::UInt64
        )
    };
    for (a, b) in [(lhs, rhs), (rhs, lhs)] {
        if unsigned_wide(a) {
            return if b.is_floating_point() { Some(b) } else { None };
        }
    }

    let (lhs_cat, lhs_width) = promotion_rank(lhs)?;
    let (rhs_cat, rhs_width) = promotion_rank(rhs)?;
    if lhs_cat != rhs_cat {
        // A higher category wins outright, whatever the widths are:
        // `int64 x float16` is `float16`, not `float32`.
        return Some(if lhs_cat > rhs_cat { lhs } else { rhs });
    }
    match lhs_width.cmp(&rhs_width) {
        std::cmp::Ordering::Greater => Some(lhs),
        std::cmp::Ordering::Less => Some(rhs),
        // Equal rank but different dtypes: only `float16` with `bfloat16`
        // reaches here, and it escapes to `float32` rather than picking one.
        std::cmp::Ordering::Equal => Some(TorchDType::Float32),
    }
}

/// The dtype a promoting binary op computes in, with both operands named when
/// there is no answer.
///
/// **Two ops call this: `mul.Tensor` and `bitwise_and.Tensor`.** Everything
/// else -- `add`, `sub`, `div`, `bitwise_or` -- still goes through
/// `same_dtype` and still refuses. That split is the "no unmeasured
/// implementation" rule (docs/E2E_REAL.md §1.2) rather than an oversight:
/// these are the two ops a real `generate()` was measured stopping on, and
/// they were found one at a time, by running it.
///
/// `_prepare_attention_mask_for_generation` computes
///
/// ```text
/// attention_mask_from_padding * can_infer_attention_mask
///     + default_attention_mask * ~can_infer_attention_mask
/// ```
///
/// where each `*` has an `int64` left operand (`.long()`) and a 0-D `bool`
/// right one (`.any()`), while the `+` joining them is int64 with int64 --
/// which is why `mul` needed this and `add` did not. Then, once past that,
/// the sampling loop's own stopping condition
///
/// ```text
/// unfinished_sequences = unfinished_sequences & ~stopping_criteria(...)
/// ```
///
/// is `int64 & bool` (`generation/utils.py:2936`). `bitwise_and.Tensor` and
/// `bitwise_or.Tensor` were both re-measured against `torch.promote_types`
/// over the storable dtypes and agree with it in every cell, so the same
/// table serves them; only `bitwise_and` has a measured caller and only
/// `bitwise_and` is wired.
fn promote_operands(op: &str, lhs: &PyTensorBase, rhs: &PyTensorBase) -> PyResult<TorchDType> {
    if lhs.tag() == rhs.tag() {
        return Ok(lhs.tag());
    }
    promote_types(lhs.tag(), rhs.tag()).ok_or_else(|| {
        not_implemented(format!(
            "{op}: dtype promotion not implemented in torch._C shim: {} vs {}",
            lhs.tag().name(),
            rhs.tag().name()
        ))
    })
}

fn arith_tensor(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    kind: Arith,
) -> PyResult<Py<PyAny>> {
    let lhs = tensor_arg(op, args, kwargs, 0, "self")?;
    let rhs = tensor_arg(op, args, kwargs, 1, "other")?;
    let operand = if kind == Arith::Mul {
        promote_operands(op, &lhs, &rhs)?
    } else {
        same_dtype(op, &lhs, &rhs)?
    };
    let tag = arith_tag(op, kind, operand, None)?;
    let storage = PyDtype::new(tag).storage(op)?;

    // Computed in `opmath_in(storage)` and narrowed once at the end -- see
    // that function. `alpha` scales inside the widened dtype for the same
    // reason `add_tensor` does it there.
    let acc = opmath_in(storage);
    let left = lhs.tensor()?.to_dtype(acc).map_err(|e| candle_err(op, e))?;
    let right = rhs.tensor()?.to_dtype(acc).map_err(|e| candle_err(op, e))?;
    let alpha = alpha_arg(op, args, kwargs)?;
    let right = scale_by_alpha(op, &right, alpha, storage)?;
    let out = apply_arith(op, kind, &left, &right)?
        .to_dtype(storage)
        .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

fn arith_scalar(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    kind: Arith,
) -> PyResult<Py<PyAny>> {
    let lhs = tensor_arg(op, args, kwargs, 0, "self")?;
    let other =
        scalar_arg(op, args, kwargs, 1, "other")?.ok_or_else(|| missing(op, "other"))?;
    let tag = arith_tag(op, kind, lhs.tag(), Some(!other.is_int()))?;
    let storage = PyDtype::new(tag).storage(op)?;

    let acc = opmath_in(storage);
    let left = lhs.tensor()?.to_dtype(acc).map_err(|e| candle_err(op, e))?;
    let alpha = alpha_arg(op, args, kwargs)?;
    // A zero-dim tensor, which is what torch's own `Scalar` overloads become
    // one layer down (`wrapped_scalar_tensor`) -- a `TorchDispatchMode` logger
    // over `f * 2` reports `aten.mul.Tensor`, not `mul.Scalar`, for exactly
    // this reason. The key stays `mul.Scalar` here because that is what the
    // *parser* picked, and the parser is what this shim reproduces.
    //
    // Built at `acc`, not at `storage`: narrowing the scalar first would round
    // `0.3` to `bfloat16` and then round the result again, where torch rounds
    // once. `opmath_in` has the measurement.
    let right = if storage.is_int() {
        Tensor::full(other.as_i64() * (alpha as i64), (), left.device()).and_then(|t| t.to_dtype(acc))
    } else {
        // Narrowed to `storage` and widened back, not built at `acc`: torch's
        // promotion makes a python float beside a `bfloat16` tensor a
        // `bfloat16` operand (docs/GENERATE.md §3.2), so `x + 0.3` adds
        // `0.30078125`. Building the scalar at `float` would add `0.3`.
        Tensor::full(other.as_f64() * alpha, (), left.device())
            .and_then(|t| t.to_dtype(storage))
            .and_then(|t| t.to_dtype(acc))
    }
    .map_err(|e| candle_err(op, e))?;
    let out = apply_arith(op, kind, &left, &right)?
        .to_dtype(storage)
        .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// `aten::rsub.Scalar(Tensor self, Scalar other, Scalar alpha=1) -> Tensor`
///
/// `other - alpha * self`. The reversed operand order is the whole op: torch
/// reaches it for `scalar - tensor`, which a Llama forward does in mask
/// construction, and it is *not* `sub.Scalar` with the sign flipped -- `alpha`
/// scales `self`, the subtrahend, not the scalar.
///
/// Dtype follows `sub.Scalar`'s rule exactly (`arith_tag` with `Arith::Sub`),
/// including the refusal on `torch.bool`: upstream raises there too
/// ("Subtraction, the `-` operator, with a bool tensor is not supported"),
/// so both sides refuse rather than one of them inventing a number.
fn rsub_scalar(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.rsub.Scalar";
    let lhs = tensor_arg(OP, args, kwargs, 0, "self")?;
    let other = scalar_arg(OP, args, kwargs, 1, "other")?.ok_or_else(|| missing(OP, "other"))?;
    let tag = arith_tag(OP, Arith::Sub, lhs.tag(), Some(!other.is_int()))?;
    let storage = PyDtype::new(tag).storage(OP)?;

    let acc = opmath_in(storage);
    let right = lhs.tensor()?.to_dtype(acc).map_err(|e| candle_err(OP, e))?;
    let alpha = alpha_arg(OP, args, kwargs)?;
    let right = scale_by_alpha(OP, &right, alpha, storage)?;
    let left = if storage.is_int() {
        Tensor::full(other.as_i64(), (), right.device()).and_then(|t| t.to_dtype(acc))
    } else {
        Tensor::full(other.as_f64(), (), right.device())
            .and_then(|t| t.to_dtype(storage))
            .and_then(|t| t.to_dtype(acc))
    }
    .map_err(|e| candle_err(OP, e))?;
    let out = apply_arith(OP, Arith::Sub, &left, &right)?
        .to_dtype(storage)
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `aten::matmul(Tensor self, Tensor other) -> Tensor`
///
/// **Named where the parser names it, not where the dispatcher does.** torch's
/// `matmul` is `CompositeImplicitAutograd`: a `TorchDispatchMode` logger over
/// `a @ b` reports `mm.default` for a 2-D pair and
/// `expand/view/bmm/_unsafe_view` for a batched one, because the decomposition
/// runs below the parser. `THPVariable_matmul` picks `aten::matmul`, and that
/// is the key here. Recorded in docs/TENSORBASE.md as a difference in what the
/// work queue reports, not in what the call returns.
fn matmul_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.matmul.default";
    let lhs = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rhs = tensor_arg(OP, args, kwargs, 1, "other")?;
    let tag = same_dtype(OP, &lhs, &rhs)?;
    if lhs.tensor()?.rank() < 2 || rhs.tensor()?.rank() < 2 {
        // torch's 1-D rules prepend/append a dimension and remove it again.
        // Not measured as used, and guessing them is what this shim refuses.
        return Err(not_implemented(format!(
            "{OP}: matmul with a 1-D operand ({}D x {}D) is not implemented in \
             torch._C shim -- torch's vector rules were not measured",
            lhs.tensor()?.rank(),
            rhs.tensor()?.rank()
        )));
    }
    let storage = PyDtype::new(tag).storage(OP)?;
    let acc = gemm_accumulate_in(storage);
    let rhs_inner = rhs.tensor()?;
    let out = lhs
        .tensor()?
        .to_dtype(acc)
        .and_then(|l| l.contiguous())
        .and_then(|l| {
            rhs_inner
                .to_dtype(acc)
                .and_then(|r| r.contiguous())
                .and_then(|r| l.broadcast_matmul(&r))
        })
        .and_then(|p| p.to_dtype(storage))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

#[derive(Clone, Copy)]
enum Cmp {
    Eq,
    Ne,
    Lt,
    Le,
    Ge,
    Gt,
}

/// The comparison ops all answer `torch.bool`, and both operands are read in
/// one common representation so the comparison is exact: `f64` if either side
/// is floating, `i64` otherwise. There is no promotion step that could round
/// one side onto the other.
///
/// `le.Tensor` joined this family for `falcon`/`gptj`/`bloom`/`mpt`, which all
/// build their causal mask as `arange(...) <= arange(...)` on two `int64`
/// tensors (measured, docs/OPS4.md §1). It is a separate key from `le.Scalar`
/// -- different schema, different overload -- but the same kernel, exactly as
/// `lt.Tensor`/`lt.Scalar` already are.
fn compare_common(op: &str, tensor: &Tensor, floating: bool) -> PyResult<Tensor> {
    tensor
        .to_dtype(if floating {
            candle_core::DType::F64
        } else {
            candle_core::DType::I64
        })
        .map_err(|e| candle_err(op, e))
}

fn apply_cmp(op: &str, kind: Cmp, lhs: &Tensor, rhs: &Tensor) -> PyResult<Tensor> {
    match kind {
        Cmp::Eq => lhs.broadcast_eq(rhs),
        Cmp::Ne => lhs.broadcast_ne(rhs),
        Cmp::Lt => lhs.broadcast_lt(rhs),
        Cmp::Le => lhs.broadcast_le(rhs),
        Cmp::Ge => lhs.broadcast_ge(rhs),
        Cmp::Gt => lhs.broadcast_gt(rhs),
    }
    .map_err(|e| candle_err(op, e))
}

fn compare_tensor(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    kind: Cmp,
) -> PyResult<Py<PyAny>> {
    let lhs = tensor_arg(op, args, kwargs, 0, "self")?;
    let rhs = tensor_arg(op, args, kwargs, 1, "other")?;
    let tag = same_dtype(op, &lhs, &rhs)?;
    let floating = tag.is_floating_point();
    let left = compare_common(op, lhs.tensor()?, floating)?;
    let right = compare_common(op, rhs.tensor()?, floating)?;
    // candle's comparisons yield U8 with 0/1, which is exactly the invariant
    // `boolean()` asserts (BOOL.md §6.3).
    finish(py, apply_cmp(op, kind, &left, &right)?, TorchDType::Bool)
}

fn compare_scalar(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    kind: Cmp,
) -> PyResult<Py<PyAny>> {
    let lhs = tensor_arg(op, args, kwargs, 0, "self")?;
    let other =
        scalar_arg(op, args, kwargs, 1, "other")?.ok_or_else(|| missing(op, "other"))?;
    let floating = lhs.tag().is_floating_point() || !other.is_int();
    let left = compare_common(op, lhs.tensor()?, floating)?;
    let right = if floating {
        Tensor::full(other.as_f64(), (), left.device())
    } else {
        Tensor::full(other.as_i64(), (), left.device())
    }
    .map_err(|e| candle_err(op, e))?;
    finish(py, apply_cmp(op, kind, &left, &right)?, TorchDType::Bool)
}

#[derive(Clone, Copy)]
enum Bitwise {
    And,
    Or,
}

/// `bitwise_and` / `bitwise_or`, which are two different operations wearing one
/// name: logical on `torch.bool`, bit-level on the integer dtypes. That is the
/// distinction BOOL.md §3 measured and refused to collapse -- aliasing `bool`
/// onto `uint8` would make `~mask` a bit flip instead of a negation.
///
/// Computed element by element through `i64`, the same shape of implementation
/// `pow` and `isin` use above. candle has no bitwise kernels, and the ops that
/// reach here in a transformer are mask combinations, not hot arithmetic.
fn bitwise_binary(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    kind: Bitwise,
) -> PyResult<Py<PyAny>> {
    let lhs = tensor_arg(op, args, kwargs, 0, "self")?;
    let rhs = tensor_arg(op, args, kwargs, 1, "other")?;
    // `bitwise_and` promotes; `bitwise_or` does not, for the reason
    // `promote_operands` gives -- one has a measured caller and the other
    // does not. Both were measured to follow the SAME table, so when a
    // caller for `or` turns up this is a one-word change.
    let tag = if matches!(kind, Bitwise::And) {
        promote_operands(op, &lhs, &rhs)?
    } else {
        same_dtype(op, &lhs, &rhs)?
    };
    if tag.is_floating_point() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "\"bitwise_{}_cpu\" not implemented for '{}'",
            match kind {
                Bitwise::And => "and",
                Bitwise::Or => "or",
            },
            scalar_type_name(tag)
        )));
    }

    let shape = lhs
        .tensor()?
        .shape()
        .broadcast_shape_binary_op(rhs.tensor()?.shape(), "bitwise")
        .map_err(|e| candle_err(op, e))?;
    let dims = shape.dims().to_vec();
    let broadcast = |t: &Tensor| -> PyResult<Vec<i64>> {
        t.broadcast_as(shape.clone())
            .and_then(|t| t.contiguous())
            .and_then(|t| t.flatten_all())
            .and_then(|t| t.to_dtype(candle_core::DType::I64))
            .and_then(|t| t.to_vec1::<i64>())
            .map_err(|e| candle_err(op, e))
    };
    let (a, b) = (broadcast(lhs.tensor()?)?, broadcast(rhs.tensor()?)?);
    let values: Vec<i64> = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| match kind {
            Bitwise::And => x & y,
            Bitwise::Or => x | y,
        })
        .collect();

    if tag == TorchDType::Bool {
        let bytes: Vec<u8> = values.into_iter().map(|v| u8::from(v != 0)).collect();
        let out = Tensor::from_vec(bytes, dims, lhs.tensor()?.device())
            .map_err(|e| candle_err(op, e))?;
        return finish(py, out, tag);
    }
    let storage = PyDtype::new(tag).storage(op)?;
    let out = Tensor::from_vec(values, dims, lhs.tensor()?.device())
        .and_then(|t| t.to_dtype(storage))
        .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// `aten::bitwise_and.Scalar` / `aten::bitwise_or.Scalar`.
///
/// Unlike the arithmetic dunders, `x & 0b1010` really does keep the Python
/// number as a `Scalar` all the way down -- `TorchDispatchMode` reports
/// `bitwise_and.Scalar`, not `.Tensor` (measured; the same probe reports
/// `mul.Tensor` for `x * 2`). So this is a distinct key with a distinct
/// kernel, and the result keeps the tensor's dtype: a Python int does not
/// widen a `uint8` tensor.
fn bitwise_scalar(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    kind: Bitwise,
) -> PyResult<Py<PyAny>> {
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    let other =
        scalar_arg(op, args, kwargs, 1, "other")?.ok_or_else(|| missing(op, "other"))?;
    let tag = input.tag();
    if tag.is_floating_point() || !other.is_int() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "\"bitwise_{}_cpu\" not implemented for '{}'",
            match kind {
                Bitwise::And => "and",
                Bitwise::Or => "or",
            },
            scalar_type_name(tag)
        )));
    }
    let rhs = other.as_i64();
    let dims = input.tensor()?.dims().to_vec();
    let values: Vec<i64> = input
        .tensor()?
        .flatten_all()
        .and_then(|t| t.to_dtype(candle_core::DType::I64))
        .and_then(|t| t.to_vec1::<i64>())
        .map_err(|e| candle_err(op, e))?
        .into_iter()
        .map(|x| match kind {
            Bitwise::And => x & rhs,
            Bitwise::Or => x | rhs,
        })
        .collect();

    if tag == TorchDType::Bool {
        let bytes: Vec<u8> = values.into_iter().map(|v| u8::from(v != 0)).collect();
        let out = Tensor::from_vec(bytes, dims, input.tensor()?.device())
            .map_err(|e| candle_err(op, e))?;
        return finish(py, out, tag);
    }
    let storage = PyDtype::new(tag).storage(op)?;
    let out = Tensor::from_vec(values, dims, input.tensor()?.device())
        .and_then(|t| t.to_dtype(storage))
        .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// `aten::bitwise_not(Tensor self) -> Tensor`. Logical negation on `bool`,
/// two's-complement `!x` on the integers.
fn bitwise_not_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.bitwise_not.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = input.tag();
    if tag.is_floating_point() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "\"bitwise_not_cpu\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }
    let dims = input.tensor()?.dims().to_vec();
    let values: Vec<i64> = input
        .tensor()?
        .flatten_all()
        .and_then(|t| t.to_dtype(candle_core::DType::I64))
        .and_then(|t| t.to_vec1::<i64>())
        .map_err(|e| candle_err(OP, e))?;

    if tag == TorchDType::Bool {
        let bytes: Vec<u8> = values.into_iter().map(|v| u8::from(v == 0)).collect();
        let out = Tensor::from_vec(bytes, dims, input.tensor()?.device())
            .map_err(|e| candle_err(OP, e))?;
        return finish(py, out, tag);
    }
    let storage = PyDtype::new(tag).storage(OP)?;
    let out = Tensor::from_vec(values.into_iter().map(|v| !v).collect::<Vec<i64>>(), dims,
                               input.tensor()?.device())
        .and_then(|t| t.to_dtype(storage))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

#[derive(Clone, Copy)]
enum Unary {
    Cos,
    Sin,
    Reciprocal,
    Tanh,
    Exp,
}

/// `cos`, `sin`, `reciprocal`, `tanh`, `exp` -- torch's unary float promotion, the
/// same rule `rsqrt` above already implements: a floating input keeps its own
/// dtype (`float16` in, `float16` out, *not* widened), and an integral or
/// boolean input becomes the default float.
///
/// `tanh` belongs here rather than beside `silu` even though both are
/// activations, and the difference is measured, not stylistic: `silu` has no
/// integral CPU kernel upstream and raises, while `tanh(int64 tensor)` returns
/// `float32` -- so `tanh` follows the promoting rule and `silu` does not.
/// (`tanh(bool)` promotes too: `[True, False]` gives `[0.7615942, 0.0]`.)
///
/// `exp` joined this family for `mamba` (docs/OPS4.md), which computes
/// `A = -exp(A_log)` (`A_log` a plain `float32` parameter) -- measured
/// `torch.exp` on `int64`/`bool` promotes to `float32` exactly like `tanh`
/// does, and a `float16` input stays `float16`.
fn unary_float(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    kind: Unary,
) -> PyResult<Py<PyAny>> {
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    let tag = if input.tag().is_floating_point() {
        input.tag()
    } else {
        default_float()
    };
    let storage = PyDtype::new(tag).storage(op)?;
    let out = input
        .tensor()?
        .to_dtype(storage)
        .and_then(|t| match kind {
            Unary::Cos => t.cos(),
            Unary::Sin => t.sin(),
            Unary::Reciprocal => t.recip(),
            Unary::Tanh => t.tanh(),
            Unary::Exp => t.exp(),
        })
        .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// `aten::neg(Tensor self) -> Tensor`
///
/// **Not `unary_float`.** `neg` keeps the input dtype instead of promoting an
/// integral input to the default float -- `int64` in, `int64` out -- so it
/// cannot share that helper.
///
/// The integral path does not go through candle either. `candle_core`'s `neg`
/// is a `unary_op!`, and that macro's integer arms are `todo!()`: calling it on
/// an `i64` tensor **panics** rather than returning an error, which would take
/// the interpreter down instead of raising. So the integers are negated through
/// an `i64` round trip, the same shape `bitwise_not` already uses for the same
/// reason. `to_dtype` back to `u8` truncates, which is torch's answer too --
/// `neg(uint8 [1, 2, 0])` is `[255, 254, 0]`, measured.
///
/// Two refusals, both copied from upstream rather than invented: `bool` (torch
/// points at `~`/`logical_not()` instead) and the wide unsigned dtypes, which
/// have no `neg_cpu` kernel upstream at all.
fn neg_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.neg.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = input.tag();
    if tag == TorchDType::Bool {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Negation, the `-` operator, on a bool tensor is not supported. If you are \
             trying to invert a mask, use the `~` or `logical_not()` operator instead.",
        ));
    }
    if matches!(
        tag,
        TorchDType::UInt16 | TorchDType::UInt32 | TorchDType::UInt64
    ) {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"neg_cpu\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }

    let storage = PyDtype::new(tag).storage(OP)?;
    if tag.is_floating_point() {
        let out = input
            .tensor()?
            .to_dtype(storage)
            .and_then(|t| t.neg())
            .map_err(|e| candle_err(OP, e))?;
        return finish(py, out, tag);
    }

    let dims = input.tensor()?.dims().to_vec();
    let values: Vec<i64> = input
        .tensor()?
        .contiguous()
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_dtype(candle_core::DType::I64))
        .and_then(|t| t.to_vec1::<i64>())
        .map_err(|e| candle_err(OP, e))?;
    let out = Tensor::from_vec(
        values.into_iter().map(|v| v.wrapping_neg()).collect::<Vec<i64>>(),
        dims,
        input.tensor()?.device(),
    )
    .and_then(|t| t.to_dtype(storage))
    .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `aten::abs(Tensor self) -> Tensor`
///
/// The float path is candle's `abs`, which is IEEE `fabs`: `abs(-0.0)` is
/// `0.0`, `abs(-inf)` is `inf`, and `abs(nan)` is `nan` -- all three measured
/// against upstream 2.13.0 and all three agree.
///
/// **The integral path is `wrapping_abs`, not `abs`.** Upstream's answer for
/// the most negative element of a signed type is that element again:
/// `abs(int64 min)` is `int64 min`, measured. Rust's `i64::abs` panics on that
/// input in a debug build, so the round trip uses `wrapping_abs`, the same
/// shape `neg_default` above uses `wrapping_neg` for the same reason. The
/// width matters: an `int32` tensor wraps at `i32::MIN`, not at `i64::MIN`, so
/// the wrap is applied in the *storage* width before widening back.
///
/// It goes through `i64` rather than candle for the same reason `neg` does --
/// candle's `abs` is a `unary_op!` whose integer arms are `todo!()`, which
/// panics and takes the interpreter down instead of raising.
///
/// `uint8`/`uint32` are the identity, which is a fact rather than a special
/// case: no unsigned element is negative. `bool` is refused with upstream's
/// wording (`"abs_cpu" not implemented for 'Bool'`).
fn abs_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.abs.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = input.tag();
    if tag == TorchDType::Bool {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "\"abs_cpu\" not implemented for 'Bool'",
        ));
    }
    let storage = PyDtype::new(tag).storage(OP)?;
    if tag.is_floating_point() {
        let out = input.tensor()?.abs().map_err(|e| candle_err(OP, e))?;
        return finish(py, out, tag);
    }
    let dims = input.tensor()?.dims().to_vec();
    let values: Vec<i64> = input
        .tensor()?
        .contiguous()
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_dtype(candle_core::DType::I64))
        .and_then(|t| t.to_vec1::<i64>())
        .map_err(|e| candle_err(OP, e))?;
    let wrapped: Vec<i64> = values
        .into_iter()
        .map(|v| match storage {
            candle_core::DType::I16 => (v as i16).wrapping_abs() as i64,
            candle_core::DType::I32 => (v as i32).wrapping_abs() as i64,
            // Unsigned storages cannot hold a negative, so this is the
            // identity; `i64` is the only remaining signed width.
            _ => v.wrapping_abs(),
        })
        .collect();
    let out = Tensor::from_vec(wrapped, dims, input.tensor()?.device())
        .and_then(|t| t.to_dtype(storage))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `aten::ceil(Tensor self) -> Tensor`
///
/// Float: candle's `ceil`. `ceil(-0.5)` is `-0.0` and keeps its sign bit,
/// `ceil(inf)` is `inf`, `ceil(nan)` is `nan` -- measured against upstream,
/// and the `-0.0` one is why `repr` cares: `_tensor_str` compares
/// `value != torch.ceil(value)` to decide whether a float tensor prints in
/// integer mode, and `repr(tensor([-0.5]))` prints `-0.` upstream.
///
/// **Integral dtypes are the identity, not a refusal.** Measured:
/// `torch.arange(3).ceil()` is `tensor([0, 1, 2])`. Only `bool` refuses, with
/// upstream's own kernel name (`"ceil_vml_cpu" not implemented for 'Bool'`) --
/// note it is a different name from `abs`'s, because upstream reaches a
/// different kernel, and copying the wrong one would send a reader to the
/// wrong place.
fn ceil_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.ceil.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = input.tag();
    if tag == TorchDType::Bool {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(
            "\"ceil_vml_cpu\" not implemented for 'Bool'",
        ));
    }
    if !tag.is_floating_point() {
        // Already integral: upstream hands the tensor straight back.
        let out = input.tensor()?.clone();
        return finish(py, out, tag);
    }
    let out = input.tensor()?.ceil().map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `aten::masked_select(Tensor self, Tensor mask) -> Tensor`
///
/// Always 1-D, whatever the input rank -- it is "the elements where the mask
/// is true, in row-major order", and there is no shape that could describe
/// that in general.
///
/// Three measured behaviours it would have been easy to get wrong:
///
///   * **The mask must be `torch.bool`.** A `uint8` mask -- which reads like
///     a mask, and which old torch accepted -- raises
///     `masked_select: expected BoolTensor for mask` on 2.13.0, and so does an
///     `int64` one. Accepting them would silently treat `2` as true, which is
///     right for that value and wrong for the caller who meant a count.
///   * **Both sides broadcast**, not just the mask. `tensor([1., 2.])`
///     against a `(2, 2)` mask of `[[T,F],[T,T]]` gives `[1., 1., 2.]` --
///     the *self* tensor is the one that expanded.
///   * **An all-false mask is an empty tensor, not an error**, and it keeps
///     the input's dtype (`tensor([], dtype=torch.int64)`).
///
/// The selection is `index_select` over the flattened, broadcast input rather
/// than a per-element rebuild, so the input dtype is carried by candle instead
/// of being reconstructed -- which is what keeps `float16`/`bfloat16` exact
/// (an `f64` round trip would not be).
fn masked_select_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.masked_select.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let mask = tensor_arg(OP, args, kwargs, 1, "mask")?;
    if mask.tag() != TorchDType::Bool {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "masked_select: expected BoolTensor for mask",
        ));
    }
    let tag = input.tag();
    let shape = broadcast_shape(OP, input.tensor()?.dims(), mask.tensor()?.dims())?;

    let flat_self = input
        .tensor()?
        .broadcast_as(shape.as_slice())
        .and_then(|t| t.contiguous())
        .and_then(|t| t.flatten_all())
        .map_err(|e| candle_err(OP, e))?;
    let flat_mask: Vec<u8> = mask
        .tensor()?
        .broadcast_as(shape.as_slice())
        .and_then(|t| t.contiguous())
        .and_then(|t| t.flatten_all())
        .and_then(|t| t.to_dtype(candle_core::DType::U8))
        .and_then(|t| t.to_vec1::<u8>())
        .map_err(|e| candle_err(OP, e))?;

    let picked: Vec<u32> = flat_mask
        .iter()
        .enumerate()
        .filter(|(_, m)| **m != 0)
        .map(|(i, _)| i as u32)
        .collect();

    if picked.is_empty() {
        // `narrow` rather than an empty `index_select`: it needs no index
        // tensor at all and cannot depend on how candle handles a zero-length
        // one. The result is `(0,)` in the input's dtype, as upstream's is.
        let out = flat_self.narrow(0, 0, 0).map_err(|e| candle_err(OP, e))?;
        return finish(py, out, tag);
    }
    let count = picked.len();
    let index =
        Tensor::from_vec(picked, count, flat_self.device()).map_err(|e| candle_err(OP, e))?;
    let out = flat_self
        .index_select(&index, 0)
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// numpy/torch broadcast shape agreement, right-aligned: each pair of extents
/// must be equal or one of them 1, and the result takes the larger.
///
/// candle has `broadcast_as` but no public "what shape would these two
/// broadcast to?", and `masked_select` needs the answer before it can expand
/// either side. The refusal reproduces upstream's wording, which names both
/// extents and the axis -- a bare "shapes do not match" would make a broadcast
/// mistake much harder to place.
fn broadcast_shape(op: &str, lhs: &[usize], rhs: &[usize]) -> PyResult<Vec<usize>> {
    let rank = lhs.len().max(rhs.len());
    let mut out = vec![0usize; rank];
    for i in 0..rank {
        let a = if i < rank - lhs.len() {
            1
        } else {
            lhs[i - (rank - lhs.len())]
        };
        let b = if i < rank - rhs.len() {
            1
        } else {
            rhs[i - (rank - rhs.len())]
        };
        out[i] = if a == b {
            a
        } else if a == 1 {
            b
        } else if b == 1 {
            a
        } else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{op}: The size of tensor a ({a}) must match the size of tensor b ({b}) at \
                 non-singleton dimension {i}"
            )));
        };
    }
    Ok(out)
}

/// `aten::unbind.int(Tensor(a -> *) self, int dim=0) -> Tensor(a)[]`
///
/// Every slice along `dim`, with `dim` *removed* -- which is what separates it
/// from `split`, whose chunks keep the dimension with extent 1. `Tensor.
/// __iter__` is this op (`torch/_tensor.py:1215`), so `for row in tensor:` and
/// every `repr` of a tensor with more than one dimension goes through it.
///
/// Returns a `list`, not a tuple: `torch.ops.aten.unbind.int(x, 0)` on 2.13.0
/// hands back a `list` even though `Tensor.unbind` (the method binding, which
/// packs it) gives a tuple. The op-level spelling is what `_tensor_str.py:597`
/// uses, so that is the one reproduced here, matching `split.Tensor` above.
///
/// Two measured refusals rather than invented ones:
///
///   * a **0-d** input raises `IndexError: Dimension specified as 0 but tensor
///     has no dimensions` -- note `normalise_dim` would happily accept `dim=0`
///     for rank 0 (torch treats a scalar as 1-D *for indexing*), so this has
///     to be checked before it, not by it.
///   * an out-of-range `dim` raises `IndexError` naming the valid range, which
///     `normalise_dim` already spells the way upstream does.
///
/// An extent of 0 along `dim` is an **empty list**, not an error.
fn unbind_int(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.unbind.int";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let requested = dim_arg(args, kwargs, 1, "dim")?.unwrap_or(0);
    if rank == 0 {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "Dimension specified as {requested} but tensor has no dimensions"
        )));
    }
    let dim = normalise_dim(OP, requested, rank)?;
    let extent = input.tensor()?.dims()[dim];
    let tag = input.tag();

    let mut slices: Vec<Py<PyAny>> = Vec::with_capacity(extent);
    for i in 0..extent {
        let slice = input
            .tensor()?
            .narrow(dim, i, 1)
            .and_then(|t| t.squeeze(dim))
            .map_err(|e| candle_err(OP, e))?;
        slices.push(crate::tensor::promote(py, finish(py, slice, tag)?)?);
    }
    Ok(PyList::new(py, slices)?.into_any().unbind())
}

/// `aten::relu(Tensor self) -> Tensor`
///
/// Opened `opt`, `nemotron` and `persimmon` (docs/ARCH.md). The whole op is one
/// line of arithmetic and every interesting thing about it is in *which* line.
///
/// **`relu` is not `max(x, 0)`.** Two measured results rule that reading out:
///
/// ```text
/// relu([nan, inf, -inf, -0.0, 0.0]) == [nan, inf, 0.0, -0.0, 0.0]
/// signbit(relu(-0.0)) == True
/// ```
///
/// `nan` survives, so it is not a maximum against zero that would let either
/// operand win by comparison order; and `-0.0` comes back with its sign, so it
/// is not "clamp then normalise". Both fall out of `x < 0 ? 0 : x`, which is
/// what this computes: `-0.0 < 0` is false so the element passes through
/// untouched, and `nan < 0` is false for the same reason. A `max`-shaped
/// implementation would pass every ordinary test and differ on exactly these
/// two inputs -- the golden cases pin both.
///
/// **Unlike `silu`, the integral dtypes are not refused.** `relu` has an
/// integral CPU kernel upstream and `silu` does not, so the refusal that
/// belongs one function down does not belong here; only `bool` is refused, with
/// upstream's wording. On `uint8` it is the identity, which is correct rather
/// than a special case: no unsigned element is negative.
fn relu_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.relu.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = input.tag();
    if tag == TorchDType::Bool {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Boolean inputs not supported for relu",
        ));
    }
    let source = input.tensor()?.contiguous().map_err(|e| candle_err(OP, e))?;
    let zeros = source.zeros_like().map_err(|e| candle_err(OP, e))?;
    let out = source
        .lt(&zeros)
        .and_then(|negative| negative.where_cond(&zeros, &source))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `aten::silu(Tensor self) -> Tensor` -- `x * sigmoid(x)`, SwiGLU's activation.
///
/// Float only, and the refusal is upstream's: there is no `silu_cpu` for an
/// integral or boolean input, so an integer tensor raises here rather than
/// being promoted the way `cos`/`sin` promote theirs. (That difference is why
/// this is not another `Unary` variant.)
///
/// `float16`/`bfloat16` are computed in `f32` and narrowed once at the end.
/// candle's `silu` evaluates `v / (1 + exp(-v))` **in the input type**, which
/// rounds three times where upstream's vectorised CPU kernel rounds once, and
/// the shim's job is upstream's answer rather than candle's.
fn silu_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.silu.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = input.tag();
    if !tag.is_floating_point() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"silu_cpu\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }
    let storage = PyDtype::new(tag).storage(OP)?;
    let acc = match storage {
        candle_core::DType::F16 | candle_core::DType::BF16 => candle_core::DType::F32,
        other => other,
    };
    let out = input
        .tensor()?
        .to_dtype(acc)
        .and_then(|t| t.silu())
        .and_then(|t| t.to_dtype(storage))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `aten::gelu(Tensor self, *, str approximate="none") -> Tensor`
///
/// **The op has two different functions behind one name, and picking the wrong
/// one is not a rounding error.** `approximate="none"` is the exact
/// `x·Φ(x)` written with `erf`; `approximate="tanh"` is Hendrycks' cubic
/// approximation. Measured over `[-3, 3]` in `float32` they differ by up to
/// **4.12e-04** -- four orders of magnitude past this shim's `float32` golden
/// tolerance (`1e-5`), so a shim that silently answered with the other formula
/// would not be caught by "close enough", it would be caught by a wrong token.
///
/// The default is `"none"` (upstream's schema string, re-read from
/// `torch.ops.aten.gelu.default._schema` rather than remembered). Which
/// architectures ask for which was measured, not guessed:
///
/// | architecture | `approximate` |
/// |---|---|
/// | **Gemma / Gemma-2** | **`"tanh"`** (`gelu_pytorch_tanh`) |
/// | BERT · RoBERTa · ELECTRA · DistilBERT · DeBERTa-v2 | `"none"` |
/// | BART · Falcon · GPT-NeoX · GPT-BigCode · Starcoder2 · MPT · ViT | `"none"` |
///
/// GPT-2 is the interesting absentee: it *is* a tanh-gelu model, but HF spells
/// `gelu_new` in Python, so it reaches `aten.tanh.default` and never this op.
/// Both spellings therefore have to agree, which is why the tanh branch below
/// is composed rather than delegated (see next paragraph).
///
/// **Neither branch is candle's own `gelu`/`gelu_erf`, and the reason is
/// measured.** candle's `Tensor::gelu` factors the cubic as
/// `β·v·(1 + κ·v·v)` where upstream ATen writes `β·(v + κ·v³)`; algebraically
/// identical, not identical in `float32`, and the gap is 2.98e-08 on
/// `[-3, 3]`. Writing upstream's association here makes the `float32` and
/// `float64` tanh branch reproduce torch **bit for bit** instead of merely
/// within tolerance. (The exact branch does delegate to `gelu_erf`, which is
/// `libm::erff` and lands 4.47e-08 from torch's own kernel -- there is no
/// composition that closes that one, since the two `erf` implementations
/// differ, and 4.47e-08 is a quarter-ulp at magnitude 1.)
///
/// `float16`/`bfloat16` compute in `float32` and narrow once, which is not a
/// convenience: it is what upstream does (`at::opmath_type<Half> = float`),
/// and it was verified rather than assumed -- `gelu(half x)` is *bitwise*
/// equal to `half(gelu(float(x)))` for both approximations, at every probe
/// point. The same rule `silu_default` above already follows.
///
/// Integral and boolean inputs are refused, with upstream's own message. This
/// is `silu`'s side of the split, not `tanh`'s: there is no `GeluKernelImpl`
/// for `Long`, and a shim that reused the promoting unary helper would compute
/// where torch raises.
fn gelu_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.gelu.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;

    // `approximate` is keyword-only in the schema (`*` before it), and upstream
    // enforces that: `gelu(x, "tanh")` is a TypeError-shaped RuntimeError, not
    // a tanh gelu. Reproduced, because the natural way to write this helper
    // (`optional(args, kwargs, 1, ...)`) would accept the positional form and
    // quietly implement a laxer op.
    if args.len() > 1 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "aten::gelu() takes 1 positional argument(s) but {} was/were given.  \
             Declaration: aten::gelu(Tensor self, *, str approximate=\"none\") -> Tensor",
            args.len()
        )));
    }
    let approximate = match kwargs.and_then(|kw| kw.get_item("approximate").ok().flatten()) {
        Some(value) => {
            if value.is_none() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "aten::gelu() Expected a value of type 'str' for argument \
                     'approximate' but instead found type 'NoneType'.",
                ));
            }
            value.extract::<String>()?
        }
        None => "none".to_string(),
    };

    let tag = input.tag();
    if !tag.is_floating_point() || tag == TorchDType::Float8E4M3FN {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"GeluKernelImpl\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }

    // The string is validated before any arithmetic, so a typo raises on an
    // empty tensor too -- upstream refuses `gelu(zeros(0), approximate="TANH")`
    // and a shim that only checked inside a loop would answer `[]`.
    let use_tanh = match approximate.as_str() {
        "none" => false,
        "tanh" => true,
        _ => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "approximate argument must be either none or tanh.",
            ))
        }
    };

    let storage = PyDtype::new(tag).storage(OP)?;
    let acc = match storage {
        candle_core::DType::F16 | candle_core::DType::BF16 => candle_core::DType::F32,
        other => other,
    };
    let x = input.tensor()?.to_dtype(acc).map_err(|e| candle_err(OP, e))?;

    let out = if use_tanh {
        // `0.5 · v · (1 + tanh(β·(v + κ·v³)))`, ATen's association exactly.
        // `affine(mul, add)` narrows both constants to the tensor's dtype
        // before it multiplies, so β arrives as the `float32` nearest for a
        // `float32` tensor -- which is what upstream's `opmath_t` constant is.
        const BETA: f64 = std::f64::consts::FRAC_2_SQRT_PI * std::f64::consts::SQRT_2 * 0.5;
        const KAPPA: f64 = 0.044715;
        let cube = x
            .mul(&x)
            .and_then(|sq| sq.mul(&x))
            .map_err(|e| candle_err(OP, e))?;
        let inner = cube
            .affine(KAPPA, 0.0)
            .and_then(|scaled| x.add(&scaled))
            .and_then(|sum| sum.affine(BETA, 0.0))
            .map_err(|e| candle_err(OP, e))?;
        let half = x.affine(0.5, 0.0).map_err(|e| candle_err(OP, e))?;
        inner
            .tanh()
            .and_then(|t| t.affine(1.0, 1.0))
            .and_then(|t| half.mul(&t))
            .map_err(|e| candle_err(OP, e))?
    } else {
        // `(erf(v/√2) + 1) · 0.5 · v` -- candle's `gelu_erf`, `libm::erf(f)`.
        x.gelu_erf().map_err(|e| candle_err(OP, e))?
    };

    let out = out.to_dtype(storage).map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

#[derive(Clone, Copy, PartialEq)]
enum Reduce {
    Sum,
    Mean,
}

/// The dims a reduction runs over, normalised, plus whether the whole tensor
/// is being reduced.
fn reduce_dims(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    rank: usize,
) -> PyResult<Option<Vec<usize>>> {
    let value = match optional(args, kwargs, index, "dim")? {
        Some(value) if !value.is_none() => value,
        _ => return Ok(None),
    };
    let raw: Vec<isize> = match value.extract::<Vec<isize>>() {
        Ok(list) => list,
        Err(_) => vec![value.extract::<isize>()?],
    };
    if raw.is_empty() {
        // torch: an empty dim list reduces nothing but still runs, which is
        // not the same as `dim=None`.
        return Ok(Some(Vec::new()));
    }
    raw.into_iter()
        .map(|d| normalise_dim(op, d, rank))
        .collect::<PyResult<Vec<_>>>()
        .map(Some)
}

/// `sum` and `mean`, in their whole-tensor and per-dimension forms.
///
/// The dtype rules are torch's, measured: `sum` promotes every non-floating
/// input to `int64` (`bool_t.sum() -> int64`, `int32_t.sum() -> int64`) while
/// a floating input keeps its dtype, and `mean` refuses a non-floating input
/// outright rather than promoting it.
fn sum_or_mean(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    kind: Reduce,
    has_dim: bool,
) -> PyResult<Py<PyAny>> {
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let (dim_at, keepdim_at, dtype_at) = if has_dim { (1, 2, 3) } else { (99, 99, 1) };
    let dims = if has_dim {
        reduce_dims(op, args, kwargs, dim_at, rank)?
    } else {
        None
    };
    let keepdim = if has_dim {
        bool_arg(args, kwargs, keepdim_at, "keepdim")?.unwrap_or(false)
    } else {
        false
    };

    let natural = match kind {
        Reduce::Sum => {
            if input.tag().is_floating_point() {
                input.tag()
            } else {
                TorchDType::Int64
            }
        }
        Reduce::Mean => {
            if !input.tag().is_floating_point() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "mean(): could not infer output dtype. Input dtype must be either \
                     a floating point or complex dtype. Got: {}",
                    input.tag().name()
                )));
            }
            input.tag()
        }
    };
    let tag = dtype_arg(args, kwargs, dtype_at, "dtype")?.unwrap_or(natural);
    let storage = PyDtype::new(tag).storage(op)?;

    // Reduced in `acc_type<T>` -- `float` for both reduced floats -- and
    // narrowed once, which is what torch's reduction kernels do and what
    // `cumsum_default` below already says in its own doc comment. Reducing in
    // `bfloat16` instead rounds after every partial sum: measured, 168 of 200
    // rows of a 64-wide `bfloat16` sum differed from upstream. `opmath_in`.
    let acc = opmath_in(storage);
    let source = input
        .tensor()?
        .to_dtype(acc)
        .map_err(|e| candle_err(op, e))?;
    // torch: an empty `dim` list reduces *every* dimension (it is not the
    // same as reducing none), so it is equivalent to naming every axis.
    let dims = dims.map(|d| if d.is_empty() { (0..rank).collect() } else { d });
    let out = match dims {
        None => match kind {
            Reduce::Sum => source.sum_all(),
            Reduce::Mean => source.mean_all(),
        },
        Some(dims) => match (kind, keepdim) {
            (Reduce::Sum, true) => source.sum_keepdim(dims),
            (Reduce::Sum, false) => source.sum(dims),
            (Reduce::Mean, true) => source.mean_keepdim(dims),
            (Reduce::Mean, false) => source.mean(dims),
        },
    }
    .and_then(|t| t.to_dtype(storage))
    .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// `aten::cumsum(Tensor self, int dim, *, ScalarType? dtype=None)`. Same
/// integral-to-int64 promotion as `sum`.
fn cumsum_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.cumsum.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let dim = normalise_dim(
        OP,
        dim_arg(args, kwargs, 1, "dim")?.ok_or_else(|| missing(OP, "dim"))?,
        input.tensor()?.rank(),
    )?;
    let natural = if input.tag().is_floating_point() {
        input.tag()
    } else {
        TorchDType::Int64
    };
    let tag = dtype_arg(args, kwargs, 2, "dtype")?.unwrap_or(natural);
    let storage = PyDtype::new(tag).storage(OP)?;

    // Accumulated here rather than with `candle_core::Tensor::cumsum`, which
    // is a matmul against a triangular matrix and so only exists for the
    // dtypes candle's gemm covers -- the golden harness caught exactly that on
    // `int64` and `bfloat16` (`unsupported dtype I64 for op matmul`).
    //
    // Floating results accumulate in `f64`. torch's CPU kernel accumulates the
    // reduced-precision floats in `float` (`acc_type<BFloat16>`) and narrows
    // once at the end, so this is the same shape of computation with a wider
    // accumulator: it can differ from torch in the last bit of a long
    // `bfloat16` run, in the more-accurate direction. docs/TENSORBASE.md.
    let dims = input.tensor()?.dims().to_vec();
    let n = dims[dim];
    let inner: usize = dims[dim + 1..].iter().product();
    let outer: usize = dims[..dim].iter().product();

    let out = if storage.is_int() {
        let mut flat: Vec<i64> = input
            .tensor()?
            .flatten_all()
            .and_then(|t| t.to_dtype(candle_core::DType::I64))
            .and_then(|t| t.to_vec1::<i64>())
            .map_err(|e| candle_err(OP, e))?;
        for o in 0..outer {
            for k in 0..inner {
                let base = o * n * inner + k;
                for i in 1..n {
                    // Wrapping, like torch's integer kernels.
                    flat[base + i * inner] =
                        flat[base + i * inner].wrapping_add(flat[base + (i - 1) * inner]);
                }
            }
        }
        Tensor::from_vec(flat, dims, input.tensor()?.device())
    } else {
        let mut flat: Vec<f64> = input
            .tensor()?
            .flatten_all()
            .and_then(|t| t.to_dtype(candle_core::DType::F64))
            .and_then(|t| t.to_vec1::<f64>())
            .map_err(|e| candle_err(OP, e))?;
        for o in 0..outer {
            for k in 0..inner {
                let base = o * n * inner + k;
                for i in 1..n {
                    flat[base + i * inner] += flat[base + (i - 1) * inner];
                }
            }
        }
        Tensor::from_vec(flat, dims, input.tensor()?.device())
    }
    .and_then(|t| t.to_dtype(storage))
    .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

#[derive(Clone, Copy)]
enum Extremum {
    Max,
    Min,
}

/// `aten::max(Tensor self) -> Tensor` and `aten::min(Tensor self) -> Tensor` --
/// the whole-tensor forms, a zero-dim result in the input's own dtype.
///
/// **NaN propagates, and candle does not do that for us.** This started as
/// `max` alone, written as `flatten_all().max(0)`, and that was a wrong answer
/// nobody had asked the right question of: candle's reduction skips NaN, so
/// `max([3, nan, 1])` came back `3.0` where upstream 2.13.0 gives `nan`
/// (measured, both sides, docs/E2E_REAL.md). Torch's rule is the IEEE
/// *maximum*/*minimum* rule rather than `fmax`/`fmin` -- a NaN anywhere in the
/// input is the answer, because there is no ordering that would let a real
/// number beat it.
///
/// It went unnoticed because `max_default_cases` had no NaN case; the
/// `_pair_result_check` in the same harness has explicit NaN handling for
/// `sort`/`topk`, so the harness could always have caught this and simply was
/// never asked. It is caught now, on both ops.
///
/// The NaN test is one extra vectorised pass (`x != x`, summed), not a
/// per-element scan, and it only runs for floating dtypes -- an integer or
/// boolean tensor has no NaN to find, so those take candle's reduction
/// directly.
///
/// `min` is here rather than beside it as a copy because the two differ in one
/// token; upstream's empty-input refusal names the op, so that message is
/// built from the same `match`.
fn extremum_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    which: Extremum,
) -> PyResult<Py<PyAny>> {
    let (op, name) = match which {
        Extremum::Max => ("aten.max.default", "max"),
        Extremum::Min => ("aten.min.default", "min"),
    };
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    if input.tensor()?.elem_count() == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{name}(): Expected reduction dim to be specified for input.numel() == 0."
        )));
    }
    let tag = input.tag();
    let flat = input
        .tensor()?
        .flatten_all()
        .and_then(|t| t.contiguous())
        .map_err(|e| candle_err(op, e))?;

    if tag.is_floating_point() {
        let nan_count = flat
            .ne(&flat)
            .and_then(|m| m.to_dtype(candle_core::DType::I64))
            .and_then(|m| m.sum_all())
            .and_then(|s| s.to_scalar::<i64>())
            .map_err(|e| candle_err(op, e))?;
        if nan_count > 0 {
            let storage = PyDtype::new(tag).storage(op)?;
            let out = Tensor::full(f64::NAN, (), flat.device())
                .and_then(|t| t.to_dtype(storage))
                .map_err(|e| candle_err(op, e))?;
            return finish(py, out, tag);
        }
    }

    let out = match which {
        Extremum::Max => flat.max(0),
        Extremum::Min => flat.min(0),
    }
    .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// `aten::max.other(Tensor self, Tensor other)` -- elementwise, and upstream
/// decomposes it to `maximum` (measured).
fn max_other(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.max.other";
    let lhs = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rhs = tensor_arg(OP, args, kwargs, 1, "other")?;
    let tag = same_dtype(OP, &lhs, &rhs)?;
    let out = lhs
        .tensor()?
        .broadcast_maximum(rhs.tensor()?)
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// Grouped-query attention: the key/value head dimension repeated up to the
/// query's.
///
/// **This belongs in the aten op, and that was measured rather than
/// assumed.** A `TorchDispatchMode` over
/// `F.scaled_dot_product_attention(q, k, v, enable_gqa=True)` with
/// `q = (2, 9, 4, 8)` and `k = v = (2, 3, 4, 8)` reports exactly one op --
/// `aten._scaled_dot_product_flash_attention_for_cpu.default` -- with the key
/// and value *still* `(2, 3, 4, 8)`. Nothing repeats them on the way in, and
/// the aten op has no `enable_gqa` argument at all: calling it directly with
/// those mismatched shapes answers `(2, 9, 4, 8)` and agrees with the
/// `enable_gqa=True` result to `0.0`. So the flag is a validation switch in
/// the Python wrapper and the broadcast is the op's own behaviour.
///
/// **Which repetition** is the part that fails plausibly rather than loudly.
/// Measured three ways on those shapes:
///
/// ```text
/// repeat_interleave(3, dim=1)               0.0
/// transformers' repeat_kv (expand+reshape)  0.0
/// repeat(1, 3, 1, 1)  ("tile")              2.82
/// ```
///
/// Both correct spellings give query head `i` the key/value head `i / n_rep`;
/// tiling gives it `i % n_rep`. Tiling produces a same-shaped,
/// same-magnitude, entirely wrong answer -- the failure mode docs/ARCH.md
/// §5.1 records for `gelu`, where the logits look reasonable and are not.
/// This uses the `unsqueeze`/`expand`/`reshape` spelling, which is
/// `repeat_interleave` along one axis and is what transformers' own
/// `repeat_kv` does; it moves bytes and computes nothing, so there is no
/// rounding question attached to the choice.
///
/// **Non-divisible head counts are refused by name.** Upstream does not
/// refuse them -- it answers, deterministically, and part of the answer is
/// garbage. Measured per query head against `kv_head = q_head / (h_q /
/// h_kv)`:
///
/// ```text
/// h_q=9 h_kv=4   heads 0..7 agree to 0.0, head 8 differs by 0.93
/// h_q=9 h_kv=2   heads 0..7 agree to 0.0, head 8 differs by 2.28
/// h_q=6 h_kv=4   heads 0..3 agree to 0.0, head 4 by 0.78, head 5 by 2.38e+31
/// ```
///
/// That is this same rule for the first `h_kv * (h_q / h_kv)` heads and an
/// out-of-bounds read for the remainder -- 2.38e+31 is not a value an
/// attention output can take with unit-magnitude inputs. There is nothing
/// there to reproduce, so this refuses and says so. Every caller arriving
/// through `F.scaled_dot_product_attention` is stopped one layer earlier by
/// the divisibility check upstream also has.
fn repeat_kv_heads(op: &str, kv: &Tensor, query_heads: usize) -> PyResult<Tensor> {
    let dims = kv.dims().to_vec();
    let kv_heads = dims[1];
    if kv_heads == query_heads {
        return Ok(kv.clone());
    }
    if kv_heads == 0 || query_heads % kv_heads != 0 {
        return Err(not_implemented(format!(
            "{op}: grouped-query attention with {query_heads} query heads and {kv_heads} \
             key/value heads is not implemented in torch._C shim -- the head counts do not \
             divide. Upstream answers here, but its answer reads past the end of the \
             key/value tensor for the leftover heads (measured), so there is nothing to \
             reproduce"
        )));
    }
    let n_rep = query_heads / kv_heads;
    let (b, s, e) = (dims[0], dims[2], dims[3]);
    kv.unsqueeze(2)
        .and_then(|t| t.expand((b, kv_heads, n_rep, s, e)))
        .and_then(|t| t.reshape((b, query_heads, s, e)))
        .and_then(|t| t.contiguous())
        .map_err(|e| candle_err(op, e))
}

/// The `(values, indices)` pair `max.dim` returns.
///
/// Upstream's is a *structseq* from `torch.return_types`, built by `_C` and
/// re-exported by `torch/return_types.py`. This shim does not own that
/// machinery, so the pair is a `collections.namedtuple` with the same two
/// field names: index access and `.values`/`.indices` both work, and the type
/// is not `torch.return_types.max`. Recorded in docs/TENSORBASE.md.
static MAX_RESULT: std::sync::OnceLock<Py<PyAny>> = std::sync::OnceLock::new();

fn max_result_type(py: Python<'_>) -> PyResult<&'static Py<PyAny>> {
    if let Some(cached) = MAX_RESULT.get() {
        return Ok(cached);
    }
    let namedtuple = py
        .import("collections")?
        .getattr("namedtuple")?
        .call1(("max", ("values", "indices")))?
        .unbind();
    let _ = MAX_RESULT.set(namedtuple);
    Ok(MAX_RESULT.get().expect("just set"))
}

fn max_dim(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.max.dim";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let dim = normalise_dim(
        OP,
        dim_arg(args, kwargs, 1, "dim")?.ok_or_else(|| missing(OP, "dim"))?,
        rank,
    )?;
    let keepdim = bool_arg(args, kwargs, 2, "keepdim")?.unwrap_or(false);

    let (values, indices) = if keepdim {
        (
            input.tensor()?.max_keepdim(dim),
            input.tensor()?.argmax_keepdim(dim),
        )
    } else {
        (input.tensor()?.max(dim), input.tensor()?.argmax(dim))
    };
    let values = values.map_err(|e| candle_err(OP, e))?;
    // int64, like `argmax` above: candle yields u32, which would be a visible
    // dtype divergence the first time an index is used.
    let indices = indices
        .and_then(|t| t.to_dtype(candle_core::DType::I64))
        .map_err(|e| candle_err(OP, e))?;

    // Promoted here, not at the dispatcher's exit: the pair leaves inside a
    // namedtuple, which `promote` (rightly) does not look into.
    let pair = (
        crate::tensor::promote(py, finish(py, values, input.tag())?)?,
        crate::tensor::promote(py, finish(py, indices, TorchDType::Int64)?)?,
    );
    Ok(max_result_type(py)?.bind(py).call1(pair)?.unbind())
}

/// `any`, in all three of its forms. The result is `torch.bool` whatever the
/// input dtype was (measured: `int_t.any()` gives `torch.bool`).
fn any_from(op: &str, source: &Tensor) -> PyResult<Tensor> {
    // "is any element non-zero", read through a 0/1 byte mask so the result
    // satisfies `boolean()`'s invariant by construction.
    source
        .to_dtype(candle_core::DType::F64)
        .and_then(|t| t.ne(0f64))
        .map_err(|e| candle_err(op, e))
}

fn any_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.any.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    if input.tensor()?.elem_count() == 0 {
        let out = Tensor::zeros((), candle_core::DType::U8, input.tensor()?.device())
            .map_err(|e| candle_err(OP, e))?;
        return finish(py, out, TorchDType::Bool);
    }
    let out = any_from(OP, input.tensor()?)?
        .flatten_all()
        .and_then(|t| t.max(0))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, TorchDType::Bool)
}

fn any_dim(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    list_form: bool,
) -> PyResult<Py<PyAny>> {
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let dims = reduce_dims(op, args, kwargs, 1, rank)?;
    let keepdim = bool_arg(args, kwargs, 2, "keepdim")?.unwrap_or(false);
    let mask = any_from(op, input.tensor()?)?;

    let dims = match dims {
        Some(dims) => dims,
        None if list_form => (0..rank).collect(),
        None => return Err(missing(op, "dim")),
    };
    // "any" over a dimension is "max of the 0/1 mask over that dimension".
    let mut out = mask;
    for dim in dims.into_iter().rev() {
        out = if keepdim {
            out.max_keepdim(dim)
        } else {
            out.max(dim)
        }
        .map_err(|e| candle_err(op, e))?;
    }
    finish(py, out, TorchDType::Bool)
}

/// `aten::masked_fill.Scalar/.Tensor(Tensor self, Tensor mask, X value)`
///
/// The mask has to be `torch.bool`. torch refuses a `uint8` mask (it was
/// deprecated and then removed), and BOOL.md §3 lists that refusal as one of
/// the six guardrails that survive only because the tag is not aliased onto
/// `uint8` -- so this shim can keep it, and does.
fn masked_fill(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
) -> PyResult<Py<PyAny>> {
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    let mask = tensor_arg(op, args, kwargs, 1, "mask")?;
    if mask.tag() != TorchDType::Bool {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "masked_fill only supports boolean masks, but got mask with dtype {}",
            mask.tag().name()
        )));
    }
    let value = scalar_arg(op, args, kwargs, 2, "value")?.ok_or_else(|| missing(op, "value"))?;

    let tag = input.tag();
    let storage = PyDtype::new(tag).storage(op)?;
    let shape = input.tensor()?.shape().clone();
    let device = input.tensor()?.device();

    let condition = mask
        .tensor()?
        .broadcast_as(shape.clone())
        .and_then(|t| t.contiguous())
        .map_err(|e| candle_err(op, e))?;
    let filled = if storage.is_int() {
        Tensor::full(value.as_i64(), shape.clone(), device)
    } else {
        Tensor::full(value.as_f64(), shape.clone(), device)
    }
    .and_then(|t| t.to_dtype(storage))
    .map_err(|e| candle_err(op, e))?;
    let source = input
        .tensor()?
        .contiguous()
        .map_err(|e| candle_err(op, e))?;

    let out = condition
        .where_cond(&filled, &source)
        .map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// `aten::where.self(Tensor condition, Tensor self, Tensor other) -> Tensor`
///
/// The three-tensor select. `falcon`, `gptj`, `bloom` and `mpt` all reach it
/// the same way -- a `bool` causal mask and two **0-D** `float32` branches
/// (`scalar_tensor(finfo.min)` and `scalar_tensor(0.0)`) broadcast up to the
/// mask's shape (measured, docs/OPS4.md §1). That is why all three operands
/// broadcast here rather than only two: the branches carry no shape at all.
///
/// **The condition's dtype rule is not `masked_fill`'s.** `masked_fill`
/// refuses a `uint8` mask because upstream refuses one (BOOL.md §3 lists that
/// as a guardrail worth keeping). `where` *accepts* `uint8`, with a deprecation
/// warning, and refuses every other non-`bool` dtype:
///
/// ```text
/// where(bool  cond, ...)  -> computes
/// where(uint8 cond, ...)  -> computes, "where received a uint8 condition
///                            tensor. This behavior is deprecated..."
/// where(int64 cond, ...)  -> RuntimeError: where expected condition to be a
///                            boolean tensor, but got a tensor with dtype Long
/// ```
///
/// All four measured. Carrying `masked_fill`'s refusal over would have refused
/// a call upstream answers, which is the noisier direction but still a
/// divergence; carrying `uint8` acceptance into `masked_fill` would have been
/// the silent one. They are different ops and this shim keeps them different.
///
/// **Upstream promotes the two branches; this shim does not.** The full 9x9
/// table was measured and it is torch's ordinary promotion lattice -- notably
/// `where(float16, bfloat16) -> float32`, and an integral branch never widens a
/// floating one (`where(float16, int64) -> float16`). The shim refuses through
/// `same_dtype` for the reason written there, and the golden cases record the
/// promoting combinations as `c_error` so the gap stays visible. **No measured
/// call site mixes**: all four architectures pass two `float32` branches.
///
/// The unselected branch is never read for its value, only for its shape and
/// dtype -- `where(True, 1.0, nan)` is `1.0`, measured -- and `where_cond`
/// selects rather than blends, so that holds here too.
fn where_self(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.where.self";
    let condition = tensor_arg(OP, args, kwargs, 0, "condition")?;
    let lhs = tensor_arg(OP, args, kwargs, 1, "self")?;
    let rhs = tensor_arg(OP, args, kwargs, 2, "other")?;

    if condition.tag() != TorchDType::Bool && condition.tag() != TorchDType::UInt8 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "where expected condition to be a boolean tensor, but got a tensor with dtype {}",
            scalar_type_name(condition.tag())
        )));
    }
    let tag = same_dtype(OP, &lhs, &rhs)?;
    let out = where_select(OP, &condition, lhs.tensor()?, rhs.tensor()?, tag)?;
    finish(py, out, tag)
}

/// The select `where.self` and `where.ScalarOther` share.
///
/// Both branches arrive already agreed on `tag`; this does the three-way
/// broadcast, the condition's truthiness normalisation, and the selection.
/// Split out when `where.ScalarOther` landed rather than duplicated: the two
/// overloads differ only in where the second branch comes from, and a second
/// copy of the broadcast rule is a second place for it to drift.
fn where_select(
    op: &str,
    condition: &PyTensorBase,
    lhs: &Tensor,
    rhs: &Tensor,
    tag: TorchDType,
) -> PyResult<Tensor> {
    let storage = PyDtype::new(tag).storage(op)?;

    // torch broadcasts all three together, and the result shape is the join of
    // the three -- not the condition's. `where(tensor(True), ones(2,3),
    // zeros(3))` is `(2, 3)`, measured, where a condition-shaped answer would
    // be `()`.
    let rhs_shape = rhs.shape().clone();
    let shape = condition
        .tensor()?
        .shape()
        .broadcast_shape_binary_op(lhs.shape(), "where")
        .and_then(|s| s.broadcast_shape_binary_op(&rhs_shape, "where"))
        .map_err(|e| candle_err(op, e))?;

    let spread = |t: &Tensor| -> PyResult<Tensor> {
        t.broadcast_as(shape.clone())
            .and_then(|t| t.contiguous())
            .map_err(|e| candle_err(op, e))
    };
    // A `uint8` condition is truthiness, not a bit pattern: `where_cond` reads
    // "not zero", which is the same rule, but the tag is normalised to 0/1
    // first so the two dtypes take identical paths.
    let mask = spread(condition.tensor()?)?;
    let mask = if condition.tag() == TorchDType::Bool {
        mask
    } else {
        mask.ne(0u8).map_err(|e| candle_err(op, e))?
    };
    let on_true = spread(&lhs.to_dtype(storage).map_err(|e| candle_err(op, e))?)?;
    let on_false = spread(&rhs.to_dtype(storage).map_err(|e| candle_err(op, e))?)?;

    mask.where_cond(&on_true, &on_false)
        .map_err(|e| candle_err(op, e))
}

/// The dtype `where.ScalarOther` promotes to, given the tensor branch's dtype
/// and which Python type the scalar arrived as.
///
/// torch's "wrapped number" rule, and every cell measured on 2.13.0 rather
/// than derived from the general promotion lattice -- they are *not* the same
/// function. `promote_types(int64, float32)` is `float32` and so is
/// `where(cond, int64_t, 2.5)`, but that agreement is a coincidence of the
/// default dtype: `promote_types(float16, float32)` is `float32` while
/// `where(cond, float16_t, 2.5)` is `float16`, because a Python float is not
/// a `float32` tensor. A scalar names a *category*, not a width.
///
/// ```text
/// scalar         bool tensor   integral tensor   floating tensor
/// True/False     bool          the tensor's      the tensor's
/// 3              int64         the tensor's      the tensor's
/// 2.5            float32       float32           the tensor's
/// ```
///
/// The `bool` column is the one that needs the Python *type* and not the
/// value: `where(cond, bool_t, True)` is `bool` and `where(cond, bool_t, 1)`
/// is `int64`, measured. `Scalar` in this file folds `bool` into `Int` (it
/// says so, and every other op wants that), so this rule reads the raw
/// object instead of going through `Scalar`.
fn where_scalar_tag(tensor: TorchDType, scalar_is_bool: bool, scalar_is_int: bool) -> TorchDType {
    if scalar_is_bool {
        return tensor;
    }
    if scalar_is_int {
        return if tensor == TorchDType::Bool {
            TorchDType::Int64
        } else {
            tensor
        };
    }
    if tensor.is_floating_point() {
        tensor
    } else {
        default_float()
    }
}

/// `aten::where.ScalarOther(Tensor condition, Tensor self, Scalar other) -> Tensor`
///
/// `torch.where(mask, tensor, python_scalar)`. transformers' `eager_mask`
/// reaches it verbatim at `masking_utils.py:603`:
///
/// ```text
/// mask = torch.where(mask, torch.tensor(0.0, device=..., dtype=dtype), min_dtype)
/// ```
///
/// with `min_dtype = torch.finfo(dtype).min`. That single call was the whole
/// of what stood between this shim and a real pretrained model's *eager*
/// forward (docs/CKPT2.md §7.1), and the schema for it was already in
/// `overloads.json` -- only the kernel was missing, so the dispatcher was
/// resolving the call and then refusing it by name.
///
/// **What the overload does was measured, not read off the schema.** A
/// `TorchDispatchMode` over the call above reports two ops:
///
/// ```text
/// aten.scalar_tensor.default(-3.5, dtype=<the PROMOTED dtype>)
/// aten.where.self(cond, self, that)
/// ```
///
/// so upstream itself turns the scalar into a 0-D tensor *at the promoted
/// dtype* and then runs ordinary `where.self`. This does the same two steps,
/// through the same `checked_convert` `scalar_tensor` uses, rather than
/// growing a third select path -- which also means the overflow rule comes
/// out identical for free: `-1` into `uint8` wraps to 255 and is answered,
/// `300` does not fit and is refused, both measured on both sides.
///
/// **A 0-D tensor is not accepted here, and that is upstream's rule rather
/// than a shortcut.** `scalar_arg` takes a 0-D tensor anywhere a `Scalar` is
/// wanted, because torch does; this overload is the exception, measured:
///
/// ```text
/// aten::where() Expected a value of type 'number' for argument 'other'
///   but instead found type Tensor
/// ```
///
/// Answering it would compute where torch raises.
///
/// `where.ScalarSelf` and `where.Scalar` stay unimplemented. They are in the
/// same `overloads.json` entry and would each be a few lines, but no measured
/// caller reaches them -- the rule docs/E2E_REAL.md §1.2 sets, and the reason
/// this kernel exists at all is that a caller *was* measured reaching it.
fn where_scalar_other(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.where.ScalarOther";
    let condition = tensor_arg(OP, args, kwargs, 0, "condition")?;
    let lhs = tensor_arg(OP, args, kwargs, 1, "self")?;
    let raw = required(OP, args, kwargs, 2, "other")?;

    if condition.tag() != TorchDType::Bool && condition.tag() != TorchDType::UInt8 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "where expected condition to be a boolean tensor, but got a tensor with dtype {}",
            scalar_type_name(condition.tag())
        )));
    }
    if raw.is_instance_of::<PyTensorBase>() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "aten::where() Expected a value of type 'number' for argument 'other' \
             but instead found type Tensor",
        ));
    }

    // Order matters, as it does in `scalar_arg`: Python's `bool` is a subclass
    // of `int`, and here the two give different answers.
    let scalar_is_bool = raw.is_instance_of::<pyo3::types::PyBool>();
    let scalar_is_int = scalar_is_bool || raw.is_instance_of::<pyo3::types::PyInt>();
    let value = scalar_arg(OP, args, kwargs, 2, "other")?.ok_or_else(|| missing(OP, "other"))?;

    let tag = where_scalar_tag(lhs.tag(), scalar_is_bool, scalar_is_int);
    // The same check `scalar_tensor` runs, at the same numel: upstream builds
    // a one-element tensor here, so the numel==1 hole that check reproduces
    // applies to this call too.
    checked_convert(&raw, scalar_is_int, tag, 1)?;

    let device = lhs.tensor()?.device().clone();
    let other = if tag == TorchDType::Bool {
        Tensor::full(u8::from(value.as_f64() != 0.0), (), &device).map_err(|e| candle_err(OP, e))?
    } else {
        let storage = PyDtype::new(tag).storage(OP)?;
        if storage.is_int() {
            Tensor::full(value.as_i64(), (), &device)
        } else {
            Tensor::full(value.as_f64(), (), &device)
        }
        .and_then(|t| t.to_dtype(storage))
        .map_err(|e| candle_err(OP, e))?
    };

    let out = where_select(OP, &condition, lhs.tensor()?, &other, tag)?;
    finish(py, out, tag)
}

/// A shape argument with torch's placeholders resolved: `-1` in `reshape`
/// means "whatever is left", `-1` in `expand` means "keep this dimension".
fn resolve_shape(op: &str, requested: &[isize], numel: usize) -> PyResult<Vec<usize>> {
    let mut known: usize = 1;
    let mut wildcard: Option<usize> = None;
    for (i, &value) in requested.iter().enumerate() {
        if value == -1 {
            if wildcard.is_some() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "only one dimension can be inferred",
                ));
            }
            wildcard = Some(i);
        } else if value < 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{op}: invalid shape dimension {value}"
            )));
        } else {
            known *= value as usize;
        }
    }
    let mut out: Vec<usize> = requested
        .iter()
        .map(|&v| if v == -1 { 0 } else { v as usize })
        .collect();
    if let Some(index) = wildcard {
        if known == 0 || numel % known != 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "shape '{requested:?}' is invalid for input of size {numel}"
            )));
        }
        out[index] = numel / known;
    }
    Ok(out)
}

fn shape_arg(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Vec<isize>> {
    let value = required(op, args, kwargs, index, name)?;
    match value.extract::<Vec<isize>>() {
        Ok(list) => Ok(list),
        Err(_) => Ok(vec![value.extract::<isize>()?]),
    }
}

/// `aten::expand(Tensor(a) self, SymInt[] size, *, bool implicit=False)`
///
/// torch allows the requested size to have more dimensions than the tensor, in
/// which case the new ones are prepended, and `-1` means "keep whatever is
/// there". candle's `broadcast_as` has the same alignment-from-the-right rule
/// once the `-1`s are resolved.
fn expand_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.expand.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let requested = shape_arg(OP, args, kwargs, 1, "size")?;
    let dims = input.tensor()?.dims();
    if requested.len() < dims.len() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "expand(torch._C.TensorBase{dims:?}, size={requested:?}): the number of \
             sizes provided ({}) must be greater or equal to the number of \
             dimensions in the tensor ({})",
            requested.len(),
            dims.len()
        )));
    }
    let offset = requested.len() - dims.len();
    let mut target = Vec::with_capacity(requested.len());
    for (i, &value) in requested.iter().enumerate() {
        if value == -1 {
            if i < offset {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "expand: the expanded size of the tensor (-1) isn't allowed in a \
                     leading, non-existing dimension",
                ));
            }
            target.push(dims[i - offset]);
        } else if value < 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{OP}: invalid expand size {value}"
            )));
        } else {
            target.push(value as usize);
        }
    }
    let out = input
        .tensor()?
        .broadcast_as(target)
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, input.tag())
}

/// `aten::view.dtype(Tensor(a) self, ScalarType dtype) -> Tensor(a)` --
/// reinterpret the bytes, do not convert the numbers.
///
/// This is how a safetensors checkpoint becomes tensors. safetensors' `mmap`
/// backend hands torch a byte storage, makes a `uint8` tensor of it with
/// `torch.asarray`, and then spells the dtype with `.view(dtype)`; measured
/// with a `TorchDispatchMode` around `safe_open(...).get_tensor(k)`, the ops
/// are `empty` / `set_` / `view.dtype` / `view.default`. So this is not a
/// numeric op at all -- `1.0` viewed as `int32` is `1065353216`.
///
/// **The route is bytes out and bytes in**, `to_le_bytes` then
/// `from_le_bytes`, which is the same function the `torch.load` reader and
/// `torch.frombuffer` use. Anything that went through a numeric conversion
/// instead would round, and a checkpoint reader that rounds is worse than one
/// that refuses.
///
/// The three refusals are upstream's, measured on 2.13.0 including the C++
/// spelling of the dtype names (`dtype.rs::cpp_name`):
///
/// ```text
///   0-dim, different widths  self.dim() cannot be 0 to view Float as Byte ...
///   last dim indivisible     self.size(-1) must be divisible by 4 to view Byte as Float ...
///   last dim not packed      self.stride(-1) must be 1 to view Byte as Float ...
/// ```
///
/// The last one is why this kernel checks a stride even though it then makes a
/// contiguous copy. `stride(-1) == 1` is exactly the condition under which the
/// copy and upstream's genuine view agree: the reinterpretation only merges or
/// splits bytes *inside* the last dimension, so as long as that dimension is
/// packed, reading row-major and re-reading gives the same bytes in the same
/// places. Dropping the check would silently answer a shape upstream rejects.
fn view_dtype(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.view.dtype";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let want = dtype_arg(args, kwargs, 1, "dtype")?.ok_or_else(|| missing(OP, "dtype"))?;
    let have = input.tag();
    let (old, new) = (have.itemsize(), want.itemsize());
    let mut dims = input.dims().to_vec();

    if old != new {
        let (a, b) = (have.cpp_name(), want.cpp_name());
        let Some(&last) = dims.last() else {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "self.dim() cannot be 0 to view {a} as {b} (different element sizes)"
            )));
        };
        if input.tensor()?.stride().last() != Some(&1) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "self.stride(-1) must be 1 to view {a} as {b} (different element \
                 sizes), but got {}",
                input.tensor()?.stride().last().copied().unwrap_or(0)
            )));
        }
        if (last * old) % new != 0 {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "self.size(-1) must be divisible by {} to view {a} as {b} \
                 (different element sizes), but got {last}",
                new / old
            )));
        }
        *dims.last_mut().expect("checked non-empty above") = last * old / new;
    }

    let bytes = crate::tensor::to_le_bytes(OP, input.tensor()?)?;
    let wrapped = crate::tensor::from_le_bytes(OP, &bytes, &dims, want)?;
    crate::tensor::promote(py, wrapped.into_pyobject(py)?.into_any().unbind())
}

/// `reshape` and `view`. **They are the same kernel here and are not the same
/// op upstream**: `view` requires the existing strides to permit it and raises
/// otherwise, while `reshape` falls back to a copy. This shim copies in both
/// cases, so a `view` that upstream would reject succeeds here. That is a
/// divergence in the safe direction (the values are right either way) and it
/// is recorded in docs/TENSORBASE.md rather than papered over.
fn reshape_like(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
    name: &str,
) -> PyResult<Py<PyAny>> {
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    let requested = shape_arg(op, args, kwargs, 1, name)?;
    let target = resolve_shape(op, &requested, input.tensor()?.elem_count())?;
    let out = input
        .tensor()?
        .contiguous()
        .and_then(|t| t.reshape(target))
        .map_err(|e| candle_err(op, e))?;
    finish(py, out, input.tag())
}

fn transpose_int(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.transpose.int";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let dim0 = normalise_dim(
        OP,
        dim_arg(args, kwargs, 1, "dim0")?.ok_or_else(|| missing(OP, "dim0"))?,
        rank,
    )?;
    let dim1 = normalise_dim(
        OP,
        dim_arg(args, kwargs, 2, "dim1")?.ok_or_else(|| missing(OP, "dim1"))?,
        rank,
    )?;
    let out = input
        .tensor()?
        .transpose(dim0, dim1)
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, input.tag())
}

/// `aten::permute(Tensor(a) self, int[] dims) -> Tensor(a)`
///
/// The most-called op of the four this round opens: `falcon` sends every weight
/// through `permute([1, 0])` and all four send attention through
/// `permute([0, 2, 1, 3])` (measured, docs/OPS4.md §1).
///
/// **Upstream this is an alias, and this shim's is not.** Measured on torch
/// 2.13.0: `permute(x, [1, 0])` shares `x.data_ptr()`, comes back
/// non-contiguous with the strides swapped, and writing through it changes `x`.
/// candle's `permute` also shares storage (it clones the `Arc` and permutes the
/// layout), but that is not what makes an alias observable here -- the in-place
/// ops in this file never write into storage, they hand `replace_with` a new
/// tensor (see the "In-place ops" note). So a write through a permuted result
/// does not reach the base, and cannot, until that changes. **This is the same
/// unanswered question `slice.Tensor` and `split.Tensor` already carry**
/// (docs/GPT2.md §7); it is not answered here, it is measured and written down.
/// docs/OPS4.md §5 has the probe.
///
/// The refusals were read off torch rather than invented, and the first one
/// would not have been guessed -- it names a layout nobody asked for:
///
/// ```text
/// len(dims) != rank  ->  "permute(sparse_coo): number of dimensions in the
///                         tensor input does not match the length of the
///                         desired ordering of dimensions i.e. input.dim() = 2
///                         is not equal to len(dims) = 3"
/// duplicate entry    ->  "permute(): duplicate dims are not allowed."
/// out of range       ->  IndexError, torch's usual wording
/// ```
///
/// Negative entries are allowed and normalised (`permute(x, [-1, -2])` equals
/// `permute(x, [1, 0])` on a 2-D input, measured), and a 0-D tensor takes
/// `dims=[]` and comes back unchanged -- so the rank check has to be exact
/// rather than `rank.max(1)`, which is why `normalise_dim` is not used for the
/// length test.
fn permute_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.permute.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let requested = shape_arg(OP, args, kwargs, 1, "dims")?;
    let rank = input.tensor()?.rank();
    if requested.len() != rank {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "permute(sparse_coo): number of dimensions in the tensor input does not \
             match the length of the desired ordering of dimensions i.e. input.dim() \
             = {rank} is not equal to len(dims) = {}",
            requested.len()
        )));
    }
    if rank == 0 {
        return finish(py, input.tensor()?.clone(), input.tag());
    }

    let mut order = Vec::with_capacity(rank);
    for &value in &requested {
        let dim = normalise_dim(OP, value, rank)?;
        if order.contains(&dim) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "permute(): duplicate dims are not allowed.",
            ));
        }
        order.push(dim);
    }
    let out = input
        .tensor()?
        .permute(order)
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, input.tag())
}

/// `aten::t(Tensor(a) self) -> Tensor(a)`
///
/// `nn.Linear` reaches this on every projection (`x @ w.t()`), which is why a
/// Llama forward calls it more than anything else in this file.
///
/// Rank decides the behaviour and torch's rule is not "transpose the last two
/// dims": 0-D and 1-D come back **unchanged**, 2-D swaps, and 3-D or more is a
/// hard error rather than a batched transpose. Measured -- guessing it as
/// `transpose(-2, -1)` would silently compute on a 3-D input where upstream
/// raises.
fn t_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.t.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    if rank > 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "t() expects a tensor with <= 2 dimensions, but self is {rank}D"
        )));
    }
    let out = if rank == 2 {
        input
            .tensor()?
            .transpose(0, 1)
            .map_err(|e| candle_err(OP, e))?
    } else {
        input.tensor()?.clone()
    };
    finish(py, out, input.tag())
}

fn unsqueeze_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.unsqueeze.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    // `unsqueeze` is the one place the legal range is `[-(rank+1), rank]`:
    // the new dimension can go after the last existing one.
    let rank = input.tensor()?.rank();
    let raw = dim_arg(args, kwargs, 1, "dim")?.ok_or_else(|| missing(OP, "dim"))?;
    let extent = rank as isize + 1;
    let dim = if raw < 0 { raw + extent } else { raw };
    if dim < 0 || dim >= extent {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "{OP}: Dimension out of range (expected to be in range of [{}, {}], but got {raw})",
            -extent,
            extent - 1
        )));
    }
    let out = input
        .tensor()?
        .unsqueeze(dim as usize)
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, input.tag())
}

/// The memory format a call asked for, as its name. The instances are built in
/// `bootstrap.py` and carry `_shim_name`; there is no Rust type for them
/// because there is nothing behind `torch.contiguous_format` but a label.
fn memory_format_name(value: &Bound<'_, PyAny>) -> String {
    value
        .getattr("_shim_name")
        .and_then(|v| v.extract::<String>())
        .unwrap_or_else(|_| value.str().map(|s| s.to_string()).unwrap_or_default())
}

fn reject_memory_format(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
) -> PyResult<()> {
    if let Some(value) = optional(args, kwargs, index, "memory_format")? {
        if !value.is_none() {
            let name = memory_format_name(&value);
            // The only two that mean "leave the layout alone", which is all
            // this shim can honour -- it has no strided layouts to rearrange.
            if name != "contiguous_format" && name != "preserve_format" {
                return Err(not_implemented(format!(
                    "{op}: memory_format=torch.{name} is not implemented in torch._C shim"
                )));
            }
        }
    }
    Ok(())
}

/// `layout=`, for the factories that are actually *called* with one.
///
/// `reject_unsupported` refuses any non-`None` layout, which is the right
/// default -- a dropped `layout=torch.sparse_coo` is a wrong answer with no
/// trace. But `torch.strided` names the dense layout this shim already has, and
/// the measured `scalar_tensor` call sites in `bloom`, `mpt` and `gptj` pass it
/// explicitly. Refusing there would block those architectures on an argument
/// that asks for exactly what is being handed back. Everything else still
/// refuses, with the layout named.
///
/// Deliberately *not* applied to `full`/`ones`/`empty`: those already refuse
/// every layout and no measured call site passes one, so widening them here
/// would change behaviour nothing has asked to change.
fn reject_layout(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
) -> PyResult<()> {
    if let Some(value) = optional(args, kwargs, index, "layout")? {
        // `memory_format_name` reads `_shim_name`, which `bootstrap.py` puts on
        // every one of these label objects -- layouts, memory formats and
        // qschemes alike -- so it reads a layout as well as it reads a format.
        // Both spellings are accepted because both arrive: the shim's own label
        // answers `strided` from `_shim_name`, and a *real* `torch.strided`
        // (which the golden harness hands to both sides) has no `_shim_name` and
        // falls back to its `str()`, `torch.strided`.
        let name = memory_format_name(&value);
        if !value.is_none() && name != "strided" && name != "torch.strided" {
            return Err(not_implemented(format!(
                "{op}: argument 'layout' not implemented in torch._C shim (got {value})"
            )));
        }
    }
    Ok(())
}

fn contiguous_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.contiguous.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    reject_memory_format(OP, args, kwargs, 1)?;
    let out = input.tensor()?.contiguous().map_err(|e| candle_err(OP, e))?;
    finish(py, out, input.tag())
}

/// `aten::clone(Tensor self, *, MemoryFormat? memory_format=None)` -- a real
/// copy. candle's `Tensor::clone` is a refcount bump on shared storage;
/// `Tensor::copy` is the one that allocates, and `clone` has to allocate or
/// the in-place ops below would write through it.
fn clone_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.clone.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    reject_memory_format(OP, args, kwargs, 1)?;
    let out = input.tensor()?.copy().map_err(|e| candle_err(OP, e))?;
    finish(py, out, input.tag())
}

/// `aten::detach(Tensor(a) self) -> Tensor(a)`
///
/// Upstream returns a *view*: a new tensor sharing storage with autograd
/// history stripped. There is no autograd here, so the history half is a
/// no-op; the sharing half is not reproduced, because this shim's in-place ops
/// replace a wrapper's tensor rather than writing into storage
/// (`PyTensorBase::replace_with`). So `x.detach().fill_(0)` leaves `x` alone
/// here and does not upstream. Recorded in docs/TENSORBASE.md.
fn detach_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.detach.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    finish(py, input.tensor()?.clone(), input.tag())
}

/// `aten::alias(Tensor(a) self) -> Tensor(a)`
///
/// Upstream's cheapest op: a new tensor object over the same storage, with no
/// autograd stripping and no copy. The aliasing half is the half this shim does
/// not reproduce -- for the same reason `detach` above does not, and with the
/// same consequence, which is that a later in-place write through one of the
/// two will not be seen by the other.
///
/// It reaches a Llama forward through GQA's `expand`/`reshape` chain, where the
/// result is read and never written, so the divergence does not bite there. It
/// would bite a KV-cache write, and that is recorded rather than papered over.
fn alias_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.alias.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    finish(py, input.tensor()?.clone(), input.tag())
}

/// `aten::_to_copy(Tensor self, *, ScalarType? dtype=None, ...)`
///
/// The dtype conversion behind `.to()`, `.float()` and `.long()`. A `bool`
/// destination normalises through `!= 0` and leaves by way of `boolean()`, so
/// the 0/1 invariant holds by construction rather than by hope.
fn to_copy_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten._to_copy.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = dtype_arg(args, kwargs, 1, "dtype")?.unwrap_or(input.tag());
    reject_unsupported(OP, args, kwargs, &[(2, "layout"), (4, "pin_memory")])?;
    reject_memory_format(OP, args, kwargs, 6)?;
    // `device=None` keeps the input where it is. See `device_arg_or_label`.
    let label = device_arg_or_label(args, kwargs, 3, "device", &input.device_label())?;
    // **cpu -> meta is the only transfer this build can perform, and it is a
    // discard rather than a copy.** Upstream agrees --
    // `torch.zeros(2).to("meta")` is a meta tensor, measured -- and it is the
    // one direction that needs no bytes: going the other way is
    // `Cannot copy out of meta tensor; no data!`, which `meta_to_copy` raises.
    if label.is_meta() {
        return meta_result(py, input.dims().to_vec(), tag);
    }
    let device = label.resolve()?;

    if tag == TorchDType::Bool {
        let out = input
            .tensor()?
            .to_device(&device)
            .and_then(|t| t.to_dtype(candle_core::DType::F64))
            .and_then(|t| t.ne(0f64))
            .map_err(|e| candle_err(OP, e))?;
        return finish(py, out, tag);
    }
    let storage = PyDtype::new(tag).storage(OP)?;
    let out = input
        .tensor()?
        .to_device(&device)
        .and_then(|t| t.to_dtype(storage))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `aten::new_ones(Tensor self, SymInt[] size, *, ScalarType? dtype=None, ...)`
/// -- `ones`, with the dtype and device defaulted from an existing tensor
/// rather than from the global default.
fn new_ones_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.new_ones.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let size: Vec<usize> = required(OP, args, kwargs, 1, "size")?.extract()?;
    let tag = dtype_arg(args, kwargs, 2, "dtype")?.unwrap_or(input.tag());
    reject_unsupported(OP, args, kwargs, &[(3, "layout"), (5, "pin_memory")])?;
    let label = device_arg_or_label(args, kwargs, 4, "device", &input.device_label())?;
    if label.is_meta() {
        return meta_result(py, size, tag);
    }
    let device = label.resolve()?;
    let storage = PyDtype::new(tag).storage(OP)?;
    let out = Tensor::ones(size, storage, &device).map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `aten::_local_scalar_dense(Tensor self) -> Scalar`
///
/// The op behind `.item()` and `bool(t)`. Upstream reaches it for both -- a
/// `TorchDispatchMode` logger over `t.item()` records exactly this key, and
/// `aten::item` (which also exists) is never named.
fn local_scalar_dense(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten._local_scalar_dense.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    if input.tensor()?.elem_count() != 1 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "a Tensor with {} elements cannot be converted to Scalar",
            input.tensor()?.elem_count()
        )));
    }
    let flat = input
        .tensor()?
        .flatten_all()
        .map_err(|e| candle_err(OP, e))?;
    if input.tag() == TorchDType::Bool {
        let value = flat
            .to_vec1::<u8>()
            .map_err(|e| candle_err(OP, e))?[0];
        return Ok((value != 0).into_bound_py_any(py)?.unbind());
    }
    if input.tag().is_floating_point() {
        let value = flat
            .to_dtype(candle_core::DType::F64)
            .and_then(|t| t.to_vec1::<f64>())
            .map_err(|e| candle_err(OP, e))?[0];
        return Ok(value.into_bound_py_any(py)?.unbind());
    }
    let value = flat
        .to_dtype(candle_core::DType::I64)
        .and_then(|t| t.to_vec1::<i64>())
        .map_err(|e| candle_err(OP, e))?[0];
    Ok(value.into_bound_py_any(py)?.unbind())
}

/// torch's negative-index convention for a single position along `dim`.
fn normalise_index(op: &str, index: isize, extent: usize) -> PyResult<usize> {
    let signed = extent as isize;
    let resolved = if index < 0 { index + signed } else { index };
    if resolved < 0 || resolved >= signed {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "{op}: index {index} is out of bounds for dimension with size {extent}"
        )));
    }
    Ok(resolved as usize)
}

fn select_int(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.select.int";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    if rank == 0 {
        return Err(pyo3::exceptions::PyIndexError::new_err(
            "invalid index of a 0-dim tensor",
        ));
    }
    let dim = normalise_dim(
        OP,
        dim_arg(args, kwargs, 1, "dim")?.unwrap_or(0),
        rank,
    )?;
    let index = normalise_index(
        OP,
        int_arg(args, kwargs, 2, "index")?.ok_or_else(|| missing(OP, "index"))? as isize,
        input.tensor()?.dims()[dim],
    )?;
    let out = input
        .tensor()?
        .narrow(dim, index, 1)
        .and_then(|t| t.squeeze(dim))
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, input.tag())
}

/// `aten::slice.Tensor(Tensor(a) self, int dim=0, SymInt? start=None,
///                     SymInt? end=None, SymInt step=1)`
///
/// torch clamps rather than raising: an out-of-range bound gives an empty or
/// truncated result. Reproduced, because a raising slice would break the
/// `x[:seq_len]` idiom that mask construction is written with.
fn slice_tensor(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.slice.Tensor";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let dim = normalise_dim(OP, dim_arg(args, kwargs, 1, "dim")?.unwrap_or(0), rank)?;
    let extent = input.tensor()?.dims()[dim] as i64;
    let step = int_arg(args, kwargs, 4, "step")?.unwrap_or(1);
    if step <= 0 {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "step must be greater than zero, got {step}"
        )));
    }

    let clamp = |value: i64| -> i64 {
        let shifted = if value < 0 { value + extent } else { value };
        shifted.clamp(0, extent)
    };
    let start = clamp(int_arg(args, kwargs, 2, "start")?.unwrap_or(0));
    let end = match int_arg(args, kwargs, 3, "end")? {
        // `sys.maxsize` is how Python spells "to the end" in a slice.
        Some(value) if value >= extent => extent,
        Some(value) => clamp(value),
        None => extent,
    };
    let length = (end - start).max(0) as usize;

    let narrowed = input
        .tensor()?
        .narrow(dim, start as usize, length)
        .map_err(|e| candle_err(OP, e))?;
    let out = if step == 1 {
        narrowed
    } else {
        let picks: Vec<i64> = (0..length as i64).step_by(step as usize).collect();
        let count = picks.len();
        let index = Tensor::from_vec(picks, count, input.tensor()?.device())
            .map_err(|e| candle_err(OP, e))?;
        narrowed
            .contiguous()
            .and_then(|t| t.index_select(&index, dim))
            .map_err(|e| candle_err(OP, e))?
    };
    finish(py, out, input.tag())
}

/// A bool/uint8 mask of rank `k` at `axis`, as the `k` integer index tensors
/// torch replaces it with.
///
/// This is upstream's own move (`at::native::expandTensors`): where the
/// gather happens a mask is not a separate kind of index, it is the
/// coordinates of its true elements. Doing the same here is what lets masks
/// and integer indices mix in one call without a second code path, and the
/// `k` tensors it produces are adjacent and equal-length by construction, so
/// the ordinary placement rule below puts them in the right place.
///
/// Measured: `x[mask3d]` on `(2,3,4,5)` with three true elements gives
/// `(3,5)` -- exactly what three adjacent index tensors of shape `(3,)` at
/// axes 0,1,2 produce.
fn mask_to_indices(
    op: &str,
    mask: &Tensor,
    axis: usize,
    dims: &[usize],
) -> PyResult<Vec<(usize, Vec<i64>, Vec<usize>)>> {
    let mask_dims = mask.dims().to_vec();
    if mask_dims.is_empty() {
        // Upstream does not answer here either: it trips an internal assertion
        // ("ntensor >= 3 INTERNAL ASSERT FAILED", measured), which is a bug
        // rather than a rule. Refusing by name beats reproducing a crash or
        // inventing the semantics it would have had.
        return Err(not_implemented(format!(
            "{op}: a 0-dim boolean mask is not implemented in torch._C shim -- \
             upstream reaches an internal assertion on this input rather than \
             defining it"
        )));
    }
    if axis + mask_dims.len() > dims.len() || dims[axis..axis + mask_dims.len()] != mask_dims[..] {
        // torch names the *first mismatching dimension*, and names it twice --
        // once relative to the mask and once relative to the tensor.
        let at = (0..mask_dims.len())
            .find(|&i| dims.get(axis + i) != Some(&mask_dims[i]))
            .unwrap_or(0);
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "The shape of the mask {mask_dims:?} at index {at} does not match the \
             shape of the indexed tensor {dims:?} at index {}",
            axis + at
        )));
    }
    let bytes = mask
        .flatten_all()
        .and_then(|t| t.to_dtype(candle_core::DType::U8))
        .and_then(|t| t.to_vec1::<u8>())
        .map_err(|e| candle_err(op, e))?;
    let mut coords: Vec<Vec<i64>> = vec![Vec::new(); mask_dims.len()];
    for (flat, &byte) in bytes.iter().enumerate() {
        if byte == 0 {
            continue;
        }
        let mut rest = flat;
        for d in (0..mask_dims.len()).rev() {
            coords[d].push((rest % mask_dims[d]) as i64);
            rest /= mask_dims[d];
        }
    }
    let count = coords[0].len();
    Ok(coords
        .into_iter()
        .enumerate()
        .map(|(d, values)| (axis + d, values, vec![count]))
        .collect())
}

/// `aten::index.Tensor(Tensor self, Tensor?[] indices) -> Tensor`
///
/// Advanced ("fancy") indexing with any number of index tensors, integer or
/// boolean, in any positions. Every rule below was measured against torch
/// 2.13.0 rather than recalled, and three of them are what make this op easy
/// to get *plausibly* wrong -- right shape, wrong contents:
///
///   * **Where the result axes land.** If the indexed axes are *adjacent*,
///     the broadcast shape is spliced in at the first of them. If any
///     un-indexed axis *separates* them, the broadcast shape moves to the
///     **front** and the un-indexed axes follow in their original order.
///     Measured on `(2,3,4,5)` with index shapes `(2,1)` and `(3,)`:
///     `[i,None,j]` gives `(2,3,3,5)` (fronted) and `[None,i,j)` gives
///     `(2,2,3,5)` (spliced at axis 1). A kernel that always splices, or
///     always fronts, is right for half the calls and silently transposed
///     for the other half.
///   * **A `None` entry separates.** At this layer `None` does not mean "new
///     axis": torch has already unsqueezed `self` by the time the op is
///     called, and the `None` left in the list is an ordinary un-indexed
///     axis that breaks adjacency exactly like a `:`. Measured: the
///     `indices` lists for `x[i,None,j]` and `x[i,:,j]` are identical.
///   * **`uint8` is a mask, not an index.** Upstream accepts `long`, `int`,
///     `byte` and `bool`, and treats `byte` as a deprecated spelling of
///     `bool` -- `x[uint8_tensor]` gathers the *true* positions, not the
///     positions its values name. Reading it as an integer index is wrong
///     with a plausible shape, which is the failure mode this op invites.
///
/// A list shorter than the tensor's rank leaves the trailing axes alone;
/// upstream does not pad it out with `None`, and neither does this.
fn index_tensor(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.index.Tensor";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let raw = required(OP, args, kwargs, 1, "indices")?;
    let items: Vec<Bound<'_, PyAny>> = raw.extract()?;

    let dims = input.tensor()?.dims().to_vec();
    let rank = dims.len();
    if items.len() > rank {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "too many indices for tensor of dimension {rank} (got {})",
            items.len()
        )));
    }

    // (axis, values, shape), one entry per *indexed axis* -- masks having
    // already been expanded into one entry per axis they cover.
    let mut picks: Vec<(usize, Vec<i64>, Vec<usize>)> = Vec::new();
    let mut axis = 0usize;
    for item in items.iter() {
        if item.is_none() {
            axis += 1;
            continue;
        }
        let tensor = item.extract::<PyTensorBase>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "{OP}: indices must be tensors or None, got {}",
                item.get_type().name().map(|n| n.to_string()).unwrap_or_default()
            ))
        })?;
        match tensor.tag() {
            TorchDType::Int64 | TorchDType::Int32 => {
                let values = tensor
                    .tensor()?
                    .flatten_all()
                    .and_then(|t| t.to_dtype(candle_core::DType::I64))
                    .and_then(|t| t.to_vec1::<i64>())
                    .map_err(|e| candle_err(OP, e))?;
                picks.push((axis, values, tensor.tensor()?.dims().to_vec()));
                axis += 1;
            }
            TorchDType::Bool | TorchDType::UInt8 => {
                let expanded = mask_to_indices(OP, tensor.tensor()?, axis, &dims)?;
                axis += tensor.tensor()?.rank();
                picks.extend(expanded);
            }
            _ => {
                // Upstream's own wording, and it names four dtypes rather
                // than the one that was passed.
                return Err(pyo3::exceptions::PyIndexError::new_err(
                    "tensors used as indices must be long, int, byte or bool tensors",
                ));
            }
        }
    }

    if picks.is_empty() {
        // `x[None]`-only index lists never reach here (bootstrap.py handles
        // `None` with `unsqueeze`), and upstream trips an internal assertion
        // on an all-`None` list, so this stays the identity it always was.
        return finish(py, input.tensor()?.clone(), input.tag());
    }

    // The index tensors broadcast against each other, right-aligned.
    let broadcast_rank = picks.iter().map(|(_, _, shape)| shape.len()).max().unwrap_or(0);
    let mut broadcast: Vec<usize> = vec![1; broadcast_rank];
    for (_, _, shape) in &picks {
        let offset = broadcast_rank - shape.len();
        for (k, &extent) in shape.iter().enumerate() {
            let slot = &mut broadcast[offset + k];
            if *slot == extent || extent == 1 {
                continue;
            }
            if *slot == 1 {
                *slot = extent;
                continue;
            }
            let named = picks
                .iter()
                .map(|(_, _, s)| format!("{s:?}"))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "shape mismatch: indexing tensors could not be broadcast together \
                 with shapes {named}"
            )));
        }
    }

    // Each index tensor's stride *through the broadcast shape*: zero where it
    // is being stretched, its own row-major stride where it is not.
    let mut strides: Vec<Vec<usize>> = Vec::with_capacity(picks.len());
    for (_, _, shape) in &picks {
        let mut own = vec![1usize; shape.len()];
        for k in (0..shape.len().saturating_sub(1)).rev() {
            own[k] = own[k + 1] * shape[k + 1];
        }
        let offset = broadcast_rank - shape.len();
        let mut over = vec![0usize; broadcast_rank];
        for k in 0..shape.len() {
            over[offset + k] = if shape[k] == 1 { 0 } else { own[k] };
        }
        strides.push(over);
    }

    // Strides of the gathered subspace, in indexed-axis order.
    let indexed: Vec<usize> = picks.iter().map(|(axis, _, _)| *axis).collect();
    let mut sub = vec![1usize; picks.len()];
    for j in (0..picks.len().saturating_sub(1)).rev() {
        sub[j] = sub[j + 1] * dims[indexed[j + 1]];
    }

    let total: usize = broadcast.iter().product();
    let mut flat: Vec<i64> = Vec::with_capacity(total);
    let mut coords = vec![0usize; broadcast_rank];
    for linear in 0..total {
        let mut rest = linear;
        for k in (0..broadcast_rank).rev() {
            coords[k] = rest % broadcast[k];
            rest /= broadcast[k];
        }
        let mut offset = 0usize;
        for (j, (at, values, _)) in picks.iter().enumerate() {
            let mut position = 0usize;
            for k in 0..broadcast_rank {
                position += coords[k] * strides[j][k];
            }
            let raw = values[position];
            let extent = dims[*at] as i64;
            let resolved = if raw < 0 { raw + extent } else { raw };
            if resolved < 0 || resolved >= extent {
                // Upstream names the dimension being indexed and its size,
                // not the index tensor's own shape.
                return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                    "index {raw} is out of bounds for dimension {at} with size {extent}"
                )));
            }
            offset += (resolved as usize) * sub[j];
        }
        flat.push(offset as i64);
    }

    // Indexed axes to the front, gather along them as one flat axis, then put
    // the result axes where the placement rule says they go.
    let untouched: Vec<usize> = (0..rank).filter(|a| !indexed.contains(a)).collect();
    let mut order = indexed.clone();
    order.extend(untouched.iter().copied());
    let block: usize = indexed.iter().map(|&a| dims[a]).product();
    let trailing: Vec<usize> = untouched.iter().map(|&a| dims[a]).collect();
    let span: usize = trailing.iter().product();

    let index = Tensor::from_vec(flat, total, input.tensor()?.device())
        .map_err(|e| candle_err(OP, e))?;
    let mut shape: Vec<usize> = broadcast.clone();
    shape.extend(trailing.iter().copied());
    let mut out = input
        .tensor()?
        .permute(order)
        .and_then(|t| t.contiguous())
        .and_then(|t| t.reshape((block, span)))
        .and_then(|t| t.index_select(&index, 0))
        .and_then(|t| t.reshape(shape))
        .map_err(|e| candle_err(OP, e))?;

    // Adjacent indexed axes splice in place; separated ones stay at the front,
    // which is where `out` already has them.
    let adjacent = indexed.windows(2).all(|pair| pair[1] == pair[0] + 1);
    let at = indexed[0];
    if adjacent && at > 0 {
        let mut permutation: Vec<usize> = (0..at).map(|k| broadcast_rank + k).collect();
        permutation.extend(0..broadcast_rank);
        permutation.extend((at..trailing.len()).map(|k| broadcast_rank + k));
        out = out
            .permute(permutation)
            .and_then(|t| t.contiguous())
            .map_err(|e| candle_err(OP, e))?;
    }
    finish(py, out, input.tag())
}

// ---------------------------------------------------------------------------
// In-place ops
//
// These are the only ops that write. They take the *receiver object* rather
// than a copy of it, and hand a replacement to `PyTensorBase::replace_with`.
// docs/FROM_CONFIG.md §2.1 measured `fill_.Scalar` five times and
// `copy_.default` twice during `AutoModelForCausalLM.from_config`, so a shim
// without them cannot build a model at all.
//
// What they do *not* do is write into storage. See `replace_with`'s comment:
// aliases created by `detach()` or by a view do not see the write, and
// mutating through the same Python object does. The measured `from_config`
// path only ever mutates through the same object (`p.data.fill_(...)`, and
// `.data` returns `self` here).
// ---------------------------------------------------------------------------

/// The receiver of an in-place op, as the live Python object.
fn tensor_receiver<'py>(
    op: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyTensorBase>> {
    let value = required(op, args, kwargs, 0, "self")?;
    value.cast_into::<PyTensorBase>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{op}: argument 'self' must be a torch._C.TensorBase"
        ))
    })
}

/// `aten::fill_.Scalar/.Tensor(Tensor(a!) self, X value) -> Tensor(a!)`
///
/// Shape and dtype are the receiver's and do not change; only the values do.
fn fill_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
) -> PyResult<Py<PyAny>> {
    let receiver = tensor_receiver(op, args, kwargs)?;
    let raw = required(op, args, kwargs, 1, "value")?;
    let value = scalar_arg(op, args, kwargs, 1, "value")?.ok_or_else(|| missing(op, "value"))?;
    let (tag, shape, device, numel) = {
        let borrowed = receiver.borrow();
        (
            borrowed.tag(),
            borrowed.tensor()?.shape().clone(),
            borrowed.tensor()?.device().clone(),
            borrowed.tensor()?.elem_count(),
        )
    };
    // The same `c10::checked_convert` reproduction `full` uses, and the golden
    // harness caught its absence here the same way: `fill_(float16, 1e6)` gave
    // `inf` where torch raises, and `fill_(int32, 2**31)` wrapped to
    // `-2**31`. `fill_` is in fact where upstream's numel==1 hole lives (the
    // CPU fast path this check has to skip), so the rule is not merely
    // borrowed from `full` -- it is the same code path upstream.
    if !raw.is_instance_of::<PyTensorBase>() {
        checked_convert(&raw, raw.is_instance_of::<pyo3::types::PyInt>(), tag, numel)?;
    }

    let replacement = if tag == TorchDType::Bool {
        let truthy = u8::from(value.as_f64() != 0.0);
        PyTensorBase::boolean(
            Tensor::full(truthy, shape, &device).map_err(|e| candle_err(op, e))?,
        )?
    } else {
        let storage = PyDtype::new(tag).storage(op)?;
        let filled = if storage.is_int() {
            Tensor::full(value.as_i64(), shape, &device)
        } else {
            Tensor::full(value.as_f64(), shape, &device)
        }
        .and_then(|t| t.to_dtype(storage))
        .map_err(|e| candle_err(op, e))?;
        PyTensorBase::new(filled)?
    };
    receiver.borrow_mut().replace_with(replacement);
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// `aten::zero_(Tensor(a!) self) -> Tensor(a!)`
///
/// **Not a spelling of `fill_(0)`, even though it computes the same values.**
/// It is a separate overload upstream, and overloads are part of this shim's
/// key (see the module note), so folding it into `fill_inplace` would make
/// `_aten_implemented()` claim one op where two were asked for.
///
/// It is here because of where it fires, which was measured rather than
/// assumed: **`zero_` never appears in a forward pass.** Recording
/// `TorchDispatchMode` over construction and over inference separately,
/// `nn.LayerNorm(8)`'s constructor calls `empty.memory_format` ×2,
/// `fill_.Scalar` ×1 (the weight, to 1) and `zero_.default` ×1 (the bias),
/// and the forward calls neither. `nn.Linear` does not call it at all --
/// its `reset_parameters` uses `uniform_`. So `zero_` is not on the op-count
/// tail the architecture sweep measures; it is on the path *before* it, and a
/// model that cannot be constructed never reaches the tail at all. That is
/// why docs/GPT2.md saw `nn.LayerNorm` fail twice over: answering
/// `_C._get_cudnn_enabled` only moves the failure to this kernel.
///
/// Zero is representable exactly in every dtype this shim stores, so unlike
/// `fill_` there is no `checked_convert` here -- there is no value to overflow.
/// `bool` zeroes to `False` and a `nan`/`inf` element is overwritten like any
/// other (both measured).
fn zero_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.zero_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let (tag, shape, device) = {
        let borrowed = receiver.borrow();
        (
            borrowed.tag(),
            borrowed.tensor()?.shape().clone(),
            borrowed.tensor()?.device().clone(),
        )
    };
    let storage = PyDtype::new(tag).storage(OP)?;
    let zeros = Tensor::zeros(shape, storage, &device).map_err(|e| candle_err(OP, e))?;
    let replacement = if tag == TorchDType::Bool {
        PyTensorBase::boolean(zeros)?
    } else {
        PyTensorBase::new(zeros)?
    };
    receiver.borrow_mut().replace_with(replacement);
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// `aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False)`
///
/// The destination keeps its own shape and dtype; the source is broadcast and
/// cast into them. That asymmetry is torch's -- `int_t.copy_(float_t)` gives
/// an int tensor, measured -- and it is why this is not just an assignment.
fn copy_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.copy_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let source = tensor_arg(OP, args, kwargs, 1, "src")?;
    let (tag, shape) = {
        let borrowed = receiver.borrow();
        (borrowed.tag(), borrowed.tensor()?.shape().clone())
    };

    let widened = source
        .tensor()?
        .broadcast_as(shape)
        .and_then(|t| t.contiguous())
        .map_err(|e| candle_err(OP, e))?;
    let replacement = if tag == TorchDType::Bool {
        PyTensorBase::boolean(
            widened
                .to_dtype(candle_core::DType::F64)
                .and_then(|t| t.ne(0f64))
                .map_err(|e| candle_err(OP, e))?,
        )?
    } else {
        let storage = PyDtype::new(tag).storage(OP)?;
        PyTensorBase::new(widened.to_dtype(storage).map_err(|e| candle_err(OP, e))?)?
    };
    receiver.borrow_mut().replace_with(replacement);
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// `aten::add_.Tensor(Tensor(a!) self, Tensor other, *, Scalar alpha=1) -> Tensor(a!)`
///
/// The in-place sibling of `add.Tensor`, needed to open `falcon` (docs/TAIL.md)
/// -- its residual connections write `hidden_states += attn_output` rather
/// than rebinding the name, so the trace calls this overload, not `add.Tensor`.
///
/// **Aliasing is `replace_with`'s, the same as every other in-place op in this
/// file** (`fill_inplace`/`zero_inplace`/`copy_inplace` above): the receiver's
/// storage is swapped for a freshly computed tensor, not written through. An
/// alias taken *before* this call does not observe the update -- the same
/// limitation docs/OPS4.md recorded for `permute`/`t`/`transpose`/`slice`
/// (their `replace_with` never reaches the original storage either), now
/// extended to an arithmetic in-place op rather than only view-producing ones.
/// Fixing it is a `replace_with` redesign, out of this task's scope.
///
/// Two rules narrower than upstream, both borrowed rather than re-derived:
///
///   * `torch.bool` is refused, matching `add.Tensor`'s own refusal --
///     upstream's in-place bool add is a logical or (measured:
///     `tensor([True,False]).add_(tensor([True,True]))` gives
///     `[True, True]`), and this shim implements that arithmetic in neither
///     the out-of-place nor the in-place overload, so `add_` does not
///     silently acquire a capability `add.Tensor` lacks.
///   * `other` is cast into the receiver's dtype rather than promoted --
///     `copy_inplace`'s rule. Upstream additionally refuses some *safe*-
///     looking casts (measured: `int32.add_(float_tensor)` raises "result
///     type Float can't be cast to the desired output type Int") that this
///     shim accepts instead. Not hit by falcon/bloom/gpt_bigcode, whose
///     residual adds already agree in dtype, so left as a known gap.
fn add_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.add_.Tensor";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let other = tensor_arg(OP, args, kwargs, 1, "other")?;
    let alpha = alpha_arg(OP, args, kwargs)?;

    let (tag, shape) = {
        let borrowed = receiver.borrow();
        (borrowed.tag(), borrowed.tensor()?.shape().clone())
    };
    if tag == TorchDType::Bool {
        return Err(not_implemented(format!(
            "{OP}: torch.bool addition is logical or, not arithmetic, and is \
             not implemented in torch._C shim"
        )));
    }
    let storage = PyDtype::new(tag).storage(OP)?;
    // Same widening as `add.Tensor` -- the in-place spelling must not compute
    // a different function from the out-of-place one. See `opmath_in`.
    let acc = opmath_in(storage);
    let lhs = {
        let borrowed = receiver.borrow();
        borrowed
            .tensor()?
            .to_dtype(acc)
            .map_err(|e| candle_err(OP, e))?
    };
    let rhs = other
        .tensor()?
        .to_dtype(acc)
        .and_then(|t| t.broadcast_as(shape))
        .and_then(|t| t.contiguous())
        .map_err(|e| candle_err(OP, e))?;
    let rhs = scale_by_alpha(OP, &rhs, alpha, storage)?;
    let out = lhs
        .add(&rhs)
        .and_then(|t| t.to_dtype(storage))
        .map_err(|e| candle_err(OP, e))?;
    receiver.borrow_mut().replace_with(PyTensorBase::new(out)?);
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// `aten::relu_(Tensor(a!) self) -> Tensor(a!)`
///
/// The in-place sibling of `relu.default`, needed because `F.relu(x,
/// inplace=True)` traces to this overload specifically -- `torch.relu_` is a
/// genuinely separate op from `torch.relu`, not an alternate spelling of it
/// (measured: `torch.ops.aten.relu_.default` and `torch.ops.aten.relu.default`
/// are different `OpOverload` objects with different schemas,
/// `Tensor(a!) self` vs plain `Tensor self`), and `aten.rs` had a kernel for
/// neither name before this (docs/SPELLINGS.md §6.6 measured zero).
///
/// **Aliasing is `replace_with`'s, same limitation as `add_inplace`/
/// `copy_inplace`/every other in-place op in this file** (see `add_inplace`'s
/// doc comment and docs/OPS4.md §8): the receiver's storage is swapped for a
/// freshly computed tensor, not written through, so a view or alias taken
/// *before* this call does not observe the update. Upstream `relu_` really is
/// an alias-preserving in-place write (measured:
/// `y = x.view(-1); x.relu_(); y` shows the update through the view on real
/// torch) -- this shim does not reproduce that, and fixing it is the same
/// `replace_with` redesign `add_inplace` already flagged as out of scope. Not
/// attempted here either.
///
/// The value and the refusal are `relu.default`'s, reused rather than
/// re-derived: `torch.bool` raises with upstream's exact wording ("Boolean
/// inputs not supported for relu", measured on both the out-of-place and
/// in-place overloads), and every other dtype computes `x < 0 ? 0 : x`
/// element-wise, preserving `nan` and the sign of `-0.0` the same way
/// `relu_default`'s doc comment measured.
fn relu_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.relu_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let tag = receiver.borrow().tag();
    if tag == TorchDType::Bool {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Boolean inputs not supported for relu",
        ));
    }
    let source = {
        let borrowed = receiver.borrow();
        // `?` because `tensor()` now returns a Result -- a meta tensor has no
        // bytes to make contiguous. This kernel arrived on the other branch, so
        // the two changes never textually conflicted and the compiler is what
        // caught it.
        borrowed
            .tensor()?
            .contiguous()
            .map_err(|e| candle_err(OP, e))?
    };
    let zeros = source.zeros_like().map_err(|e| candle_err(OP, e))?;
    let out = source
        .lt(&zeros)
        .and_then(|negative| negative.where_cond(&zeros, &source))
        .map_err(|e| candle_err(OP, e))?;
    receiver.borrow_mut().replace_with(PyTensorBase::new(out)?);
    let _ = py;
    Ok(receiver.into_any().unbind())
}

// ---------------------------------------------------------------------------
// The two RNG ops
//
// docs/RNG.md is the standing decision behind these: candle's CPU backend
// refuses to be seeded at all, so its `rand_uniform`/`rand_normal` cannot be
// used here even in principle, and torch's own CPU generator is ported into
// `rng.rs` instead. What is left for this file is the part that depends on
// the *tensor* rather than on the stream -- which accumulate type the kernel
// runs in, which of `normal_`'s two paths a given size and layout takes, and
// where the narrowing cast happens relative to `uniform_`'s upper-bound clamp.
// Getting any of those wrong reproduces the stream perfectly and still
// produces different numbers.
// ---------------------------------------------------------------------------

/// The floating dtype an RNG op is allowed to fill, with its candle storage.
///
/// `AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, ...)` is the whole
/// permitted set for both kernels; an integer tensor reaches a different op
/// upstream (`random_`), so accepting one here would be implementing something
/// else under this name.
fn rng_float_dtype(op: &str, tag: TorchDType) -> PyResult<candle_core::DType> {
    match tag {
        TorchDType::Float64 => Ok(candle_core::DType::F64),
        TorchDType::Float32 => Ok(candle_core::DType::F32),
        TorchDType::Float16 => Ok(candle_core::DType::F16),
        TorchDType::BFloat16 => Ok(candle_core::DType::BF16),
        other => Err(not_implemented(format!(
            "{op}: not implemented in torch._C shim for torch.{} -- upstream \
             dispatches this op over floating dtypes only, and an integer \
             tensor reaches `random_`, a different op",
            other.name()
        ))),
    }
}

/// The `Generator? generator=None` tail both schemas carry.
///
/// There is exactly one generator here -- the process-wide default that
/// `torch.default_generator` names -- so a *different* generator is refused by
/// name rather than silently served from the default stream, which would make
/// `torch.Generator().manual_seed(0)` look like it worked while sharing state
/// with everything else. `None` is the common case and never even arrives:
/// the overload resolver drops arguments equal to their schema default.
fn generator_arg(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<()> {
    let Some(value) = optional(args, kwargs, index, name)? else {
        return Ok(());
    };
    if value.is_none() {
        return Ok(());
    }
    if value
        .getattr("_shim_is_default_generator")
        .is_ok_and(|flag| flag.is_truthy().unwrap_or(false))
    {
        return Ok(());
    }
    Err(not_implemented(format!(
        "{op}: only torch.default_generator is implemented in torch._C shim; \
         a separate torch.Generator has no state of its own here"
    )))
}

fn float_arg(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
    fallback: f64,
) -> PyResult<f64> {
    match optional(args, kwargs, index, name)? {
        Some(value) if !value.is_none() => value.extract::<f64>(),
        _ => Ok(fallback),
    }
}

/// Shape, device, dtype and layout of an in-place RNG op's receiver.
struct RngTarget {
    tag: TorchDType,
    storage: candle_core::DType,
    shape: candle_core::Shape,
    device: Device,
    numel: usize,
    contiguous: bool,
}

fn rng_target(op: &str, receiver: &Bound<'_, PyTensorBase>) -> PyResult<RngTarget> {
    let borrowed = receiver.borrow();
    let tag = borrowed.tag();
    Ok(RngTarget {
        tag,
        storage: rng_float_dtype(op, tag)?,
        shape: borrowed.tensor()?.shape().clone(),
        device: borrowed.tensor()?.device().clone(),
        numel: borrowed.tensor()?.elem_count(),
        contiguous: borrowed.tensor()?.is_contiguous(),
    })
}

/// Round a value through the storage dtype and back, so it can be compared in
/// the accumulate type. For `float32` this is the identity; for `float16` and
/// `bfloat16` it is the narrowing `static_cast<scalar_t>` that upstream's
/// clamp is written in terms of. candle's `to_dtype` rounds to nearest even,
/// which is what the C++ cast does.
fn narrow_roundtrip_f32(op: &str, value: f32, storage: candle_core::DType, device: &Device) -> PyResult<f32> {
    if storage == candle_core::DType::F32 {
        return Ok(value);
    }
    Tensor::from_vec(vec![value], 1, device)
        .and_then(|t| t.to_dtype(storage))
        .and_then(|t| t.to_dtype(candle_core::DType::F32))
        .and_then(|t| t.to_vec1::<f32>())
        .map(|values| values[0])
        .map_err(|e| candle_err(op, e))
}

/// `aten::uniform_(Tensor(a!) self, float from=0., float to=1., *,
///                 Generator? generator=None) -> Tensor(a!)`
///
/// This is the sixth wall on the way to `from_config` (docs/TENSORBASE.md §7):
/// `nn.init.kaiming_uniform_` ends in `tensor.uniform_(-bound, bound)`, so no
/// `nn.Linear` exists until it does.
///
/// Two things here are not the RNG and are still part of the answer.
///
/// *The accumulate type follows `opmath_type<scalar_t>`, not the dtype.* A
/// `float16` tensor draws **one** 32-bit word per element and transforms it in
/// float; only `float64` draws two. Reading the dtype instead would consume
/// the stream at the wrong rate and desynchronise everything after it.
///
/// *The upper bound is enforced after the cast, not before.* Upstream's
/// kernel ends `return value == to_scalar ? from_scalar : value;` -- because
/// narrowing a float that is just under `to` can round it *up to* `to`, and
/// `uniform_` promises a half-open range. On `float16` with `to=1.0` that is
/// roughly one draw in 4096, so a shim without the clamp passes casual
/// inspection and fails the golden harness's range check.
fn uniform_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.uniform_.default";

    let receiver = tensor_receiver(OP, args, kwargs)?;
    let from = float_arg(args, kwargs, 1, "from", 0.0)?;
    let to = float_arg(args, kwargs, 2, "to", 1.0)?;
    generator_arg(OP, args, kwargs, 3, "generator")?;
    let target = rng_target(OP, &receiver)?;

    // torch's own check, message included.
    if !(from <= to) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "uniform_ expects to return a [from, to) range, but found from={from} > to={to}"
        )));
    }

    let mut gen = crate::rng::default_generator();
    let replacement = if target.storage == candle_core::DType::F64 {
        let mut values = crate::rng::uniform_fill_f64(&mut gen, target.numel, from, to);
        for value in values.iter_mut() {
            if *value == to {
                *value = from;
            }
        }
        Tensor::from_vec(values, target.shape, &target.device).map_err(|e| candle_err(OP, e))?
    } else {
        let (from_f32, to_f32) = (from as f32, to as f32);
        let values = crate::rng::uniform_fill_f32(&mut gen, target.numel, from_f32, to_f32);
        // The cast first, then the comparison: `to_scalar` is
        // `static_cast<scalar_t>(to_)`, and both sides of `==` are in
        // `scalar_t`. Round-tripping through the storage dtype is how that
        // comparison is made without hand-rolling half-precision rounding.
        let narrowed = Tensor::from_vec(values, target.numel, &target.device)
            .and_then(|t| t.to_dtype(target.storage))
            .and_then(|t| t.to_dtype(candle_core::DType::F32))
            .and_then(|t| t.to_vec1::<f32>())
            .map_err(|e| candle_err(OP, e))?;
        let to_scalar = narrow_roundtrip_f32(OP, to_f32, target.storage, &target.device)?;
        let clamped: Vec<f32> = narrowed
            .into_iter()
            .map(|v| if v == to_scalar { from_f32 } else { v })
            .collect();
        Tensor::from_vec(clamped, target.shape, &target.device)
            .and_then(|t| t.to_dtype(target.storage))
            .map_err(|e| candle_err(OP, e))?
    };

    drop(gen);
    receiver.borrow_mut().replace_with(PyTensorBase::new(replacement)?);
    let _ = (py, target.tag);
    Ok(receiver.into_any().unbind())
}

/// `aten::normal_(Tensor(a!) self, float mean=0., float std=1., *,
///                Generator? generator=None) -> Tensor(a!)`
///
/// The op where the *shape of the kernel* is observable output. `normal_kernel`
/// branches on `size >= 16 && self.is_contiguous()`, and the two sides share
/// nothing:
///
///   * **Path B** (small or strided) runs Box-Muller one element at a time in
///     `double` -- for every dtype, `float16` included -- and leaves the
///     unused half of each pair *cached on the generator*, so an odd-sized
///     `normal_` changes what the next one returns.
///   * **Path A** fills the whole buffer with uniforms first and then applies
///     Box-Muller over it in blocks of 16, pairing element `j` with `j+8`.
///     When the size is not a multiple of 16 it steps back to `size - 16` and
///     redraws those sixteen *over values it already wrote*.
///
/// So `n=15` and `n=16` produce entirely different sequences from one seed,
/// and `n=17` differs from `n=16` in its first element too. docs/RNG.md §1.3
/// measured all three; the harness cases below them are the regression.
fn normal_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.normal_.default";

    let receiver = tensor_receiver(OP, args, kwargs)?;
    let mean = float_arg(args, kwargs, 1, "mean", 0.0)?;
    let std = float_arg(args, kwargs, 2, "std", 1.0)?;
    generator_arg(OP, args, kwargs, 3, "generator")?;
    let target = rng_target(OP, &receiver)?;

    if !(std >= 0.0) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "normal expects std >= 0.0, but found std {std}"
        )));
    }

    let mut gen = crate::rng::default_generator();
    let values_f64: Option<Vec<f64>>;
    let values_f32: Option<Vec<f32>>;

    if target.numel >= 16 && target.contiguous {
        match target.storage {
            candle_core::DType::F64 => {
                values_f64 = Some(crate::rng::normal_fill_f64(&mut gen, target.numel, mean, std));
                values_f32 = None;
            }
            candle_core::DType::F32 => {
                values_f64 = None;
                values_f32 = Some(crate::rng::normal_fill_f32(
                    &mut gen,
                    target.numel,
                    mean as f32,
                    std as f32,
                ));
            }
            // float16 / bfloat16 -- the stack-buffer branch.
            _ => {
                values_f64 = None;
                values_f32 = Some(crate::rng::normal_fill_reduced(
                    &mut gen,
                    target.numel,
                    mean as f32,
                    std as f32,
                ));
            }
        }
    } else {
        // Path B is `double` for every dtype; the narrowing happens once, at
        // the end, in `to_dtype`.
        values_f64 = Some(crate::rng::normal_serial(&mut gen, target.numel, mean, std));
        values_f32 = None;
    }
    drop(gen);

    let filled = match (values_f64, values_f32) {
        (Some(values), _) => Tensor::from_vec(values, target.shape, &target.device),
        (_, Some(values)) => Tensor::from_vec(values, target.shape, &target.device),
        _ => unreachable!("one of the two accumulate types is always produced"),
    }
    .and_then(|t| t.to_dtype(target.storage))
    .map_err(|e| candle_err(OP, e))?;

    receiver.borrow_mut().replace_with(PyTensorBase::new(filled)?);
    let _ = (py, target.tag);
    Ok(receiver.into_any().unbind())
}

// ---------------------------------------------------------------------------
// The eight ops `do_sample=True` stops on
//
// docs/GAP.md §4 predicted ten; the coordinating session re-measured a real
// transformers Llama against `_aten_implemented()` and found eight still
// missing. docs/SAMPLING.md records what each one turned out to be.
//
// Seven of the eight are ordinary kernels. `multinomial` is not: it is the only
// op in this file that *draws*, and a sampled token is only reproducible if it
// consumes torch's stream in torch's order. That made "read the upstream
// algorithm" a requirement rather than a nicety, and reading it produced two
// facts guessing would have missed -- see `multinomial_default`.
//
// One thing is deliberately *not* reproduced, and it is worth naming here
// rather than only at the call site: `topk`'s `sorted=False` order. Upstream
// leaves it unspecified and its CPU kernel returns a partition artefact
// (measured: `k=3` of an 8-element tensor answers `[7, 6, 0]` where `sorted=True`
// answers `[6, 7, 0]`). This shim always returns the sorted order, which is a
// legal answer to the same question; the harness compares those cases as
// multisets so it cannot accidentally pin an order upstream does not promise.
// ---------------------------------------------------------------------------

/// A tensor's elements read into the one representation that can hold them
/// exactly: `f64` for the floating dtypes (widening f32/f16/bf16 and back is
/// lossless), `i64` for everything else.
///
/// Reading floats as `i64` or `int64` as `f64` would both lose information the
/// ops below depend on -- `sort` on `int64` values beyond 2^53 would start
/// declaring ties that are not ties.
enum Flat {
    Float(Vec<f64>),
    Int(Vec<i64>),
}

impl Flat {
    fn empty_like(&self, n: usize) -> Flat {
        match self {
            Flat::Float(_) => Flat::Float(vec![0.0; n]),
            Flat::Int(_) => Flat::Int(vec![0; n]),
        }
    }
}

fn read_flat(op: &str, tensor: &Tensor, tag: TorchDType) -> PyResult<Flat> {
    let flat = tensor
        .contiguous()
        .and_then(|t| t.flatten_all())
        .map_err(|e| candle_err(op, e))?;
    if tag.is_floating_point() {
        Ok(Flat::Float(
            flat.to_dtype(candle_core::DType::F64)
                .and_then(|t| t.to_vec1::<f64>())
                .map_err(|e| candle_err(op, e))?,
        ))
    } else {
        Ok(Flat::Int(
            flat.to_dtype(candle_core::DType::I64)
                .and_then(|t| t.to_vec1::<i64>())
                .map_err(|e| candle_err(op, e))?,
        ))
    }
}

fn write_flat(
    op: &str,
    values: Flat,
    dims: Vec<usize>,
    device: &Device,
    tag: TorchDType,
) -> PyResult<Tensor> {
    let storage = PyDtype::new(tag).storage(op)?;
    match values {
        Flat::Float(v) => Tensor::from_vec(v, dims, device),
        Flat::Int(v) => Tensor::from_vec(v, dims, device),
    }
    .and_then(|t| t.to_dtype(storage))
    .map_err(|e| candle_err(op, e))
}

/// torch's ordering of floats, which is not `f64`'s.
///
/// IEEE says every comparison against NaN is false; torch sorts NaN as
/// *greatest*, so ascending puts it last and descending puts it first, and
/// `topk(largest=True)` picks it before `+inf` (all three measured). A
/// `partial_cmp().unwrap()` here would panic on the first NaN instead.
fn cmp_torch_f64(a: f64, b: f64) -> std::cmp::Ordering {
    use std::cmp::Ordering;
    match (a.is_nan(), b.is_nan()) {
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
        _ => a.partial_cmp(&b).unwrap_or(Ordering::Equal),
    }
}

/// The (values, indices) result `sort` and `topk` share, in the layout
/// upstream produces.
struct Ordered {
    values: Flat,
    indices: Vec<i64>,
    dims: Vec<usize>,
}

/// Reorder every lane along `dim`, keeping `keep` of each.
///
/// **The sort is stable, and for `sort` that is a measurement, not a
/// convenience.** Upstream's CPU `sort` keeps the original index order among
/// equal values in *both* directions: `[3,1,3,1,2,3]` descending answers
/// indices `[0,2,5,4,1,3]`, not the reverse of the ascending run, and an
/// 80-element all-ties tensor comes back as `0..79`. An unstable
/// `sort_unstable_by` here would be a silent behaviour change with no failing
/// test anywhere near it.
///
/// **`topk` is a different story and the difference is recorded rather than
/// hidden.** Upstream's `topk` is a partial selection, not a sort, and its tie
/// order is stable only sometimes: on the same `[3,1,3,1,2,3]`, `k=3` answers
/// `[0,2,5]` (stable, and this shim agrees) but `k=6` answers `[0,2,5,4,3,1]`
/// -- the two 1.0s come back reversed. Reproducing that would mean
/// transcribing the partition, and upstream promises nothing about it. This
/// shim answers `[0,2,5,4,1,3]` there. It matters for nothing measured: the
/// `top_k` warper reads only `values[..., -1]`, and `multinomial`'s
/// no-replacement path feeds `topk` continuous ratios where ties do not occur.
/// docs/SAMPLING.md §4 has the measurement; the golden cases keep `topk`'s
/// index comparison to tie-free inputs and compare the tied ones by value.
fn order_along(
    op: &str,
    input: &PyTensorBase,
    dim: usize,
    descending: bool,
    keep: Option<usize>,
) -> PyResult<Ordered> {
    let dims = input.tensor()?.dims().to_vec();
    // A 0-d tensor is one lane of one element; torch answers `sort`/`topk` on
    // one with a 0-d value and a 0-d index of 0, rather than refusing.
    let (outer, n, inner) = if dims.is_empty() {
        (1usize, 1usize, 1usize)
    } else {
        (
            dims[..dim].iter().product::<usize>(),
            dims[dim],
            dims[dim + 1..].iter().product::<usize>(),
        )
    };
    let keep = keep.unwrap_or(n);

    let source = read_flat(op, input.tensor()?, input.tag())?;
    let mut values = source.empty_like(outer * keep * inner);
    let mut indices = vec![0i64; outer * keep * inner];

    let mut order: Vec<usize> = Vec::with_capacity(n);
    for o in 0..outer {
        for i in 0..inner {
            let at = |j: usize| o * n * inner + j * inner + i;
            order.clear();
            order.extend(0..n);
            match &source {
                Flat::Float(v) => order.sort_by(|&a, &b| {
                    let (x, y) = (v[at(a)], v[at(b)]);
                    if descending {
                        cmp_torch_f64(y, x)
                    } else {
                        cmp_torch_f64(x, y)
                    }
                }),
                Flat::Int(v) => order.sort_by(|&a, &b| {
                    let (x, y) = (v[at(a)], v[at(b)]);
                    if descending {
                        y.cmp(&x)
                    } else {
                        x.cmp(&y)
                    }
                }),
            }
            for (slot, &j) in order.iter().take(keep).enumerate() {
                let dst = o * keep * inner + slot * inner + i;
                indices[dst] = j as i64;
                match (&source, &mut values) {
                    (Flat::Float(src), Flat::Float(out)) => out[dst] = src[at(j)],
                    (Flat::Int(src), Flat::Int(out)) => out[dst] = src[at(j)],
                    _ => unreachable!("values was allocated from source's own variant"),
                }
            }
        }
    }

    let mut out_dims = dims;
    if !out_dims.is_empty() {
        out_dims[dim] = keep;
    }
    Ok(Ordered {
        values,
        indices,
        dims: out_dims,
    })
}

static SORT_RESULT: std::sync::OnceLock<Py<PyAny>> = std::sync::OnceLock::new();
static TOPK_RESULT: std::sync::OnceLock<Py<PyAny>> = std::sync::OnceLock::new();

/// The `(values, indices)` namedtuple `sort` and `topk` answer with, built the
/// same way `max.dim`'s is and cached for the same reason.
fn values_indices_type(
    py: Python<'_>,
    cell: &'static std::sync::OnceLock<Py<PyAny>>,
    name: &str,
) -> PyResult<&'static Py<PyAny>> {
    if let Some(cached) = cell.get() {
        return Ok(cached);
    }
    let namedtuple = py
        .import("collections")?
        .getattr("namedtuple")?
        .call1((name, ("values", "indices")))?
        .unbind();
    let _ = cell.set(namedtuple);
    Ok(cell.get().expect("just set"))
}

fn finish_ordered(
    py: Python<'_>,
    op: &str,
    cell: &'static std::sync::OnceLock<Py<PyAny>>,
    name: &str,
    ordered: Ordered,
    tag: TorchDType,
    device: &Device,
) -> PyResult<Py<PyAny>> {
    let values = write_flat(op, ordered.values, ordered.dims.clone(), device, tag)?;
    let indices = Tensor::from_vec(ordered.indices, ordered.dims, device)
        .map_err(|e| candle_err(op, e))?;
    // Promoted here rather than at the dispatcher's exit, for the reason
    // `max.dim` gives: the pair leaves inside a namedtuple, which `promote`
    // does not look into.
    let pair = (
        crate::tensor::promote(py, finish(py, values, tag)?)?,
        crate::tensor::promote(py, finish(py, indices, TorchDType::Int64)?)?,
    );
    Ok(values_indices_type(py, cell, name)?
        .bind(py)
        .call1(pair)?
        .unbind())
}

/// `aten::sort(Tensor self, int dim=-1, bool descending=False)
///     -> (Tensor values, Tensor indices)`
///
/// Reached by `TopPLogitsWarper`, which sorts the whole vocabulary row before
/// taking a cumulative sum of the softmax over it.
fn sort_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.sort.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let dim = normalise_dim(OP, dim_arg(args, kwargs, 1, "dim")?.unwrap_or(-1), rank)?;
    let descending = bool_arg(args, kwargs, 2, "descending")?.unwrap_or(false);
    let ordered = order_along(OP, &input, dim, descending, None)?;
    let device = input.tensor()?.device().clone();
    finish_ordered(py, OP, &SORT_RESULT, "sort", ordered, input.tag(), &device)
}

/// `aten::topk(Tensor self, SymInt k, int dim=-1, bool largest=True,
///             bool sorted=True) -> (Tensor values, Tensor indices)`
///
/// Reached by `TopKLogitsWarper`, which only ever reads `values[..., -1]` --
/// the k-th largest logit, used as a threshold. The indices still have to be
/// right, because the same op is how `multinomial` picks more than one sample
/// without replacement.
fn topk_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.topk.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let k = int_arg(args, kwargs, 1, "k")?.ok_or_else(|| missing(OP, "k"))?;
    let dim = normalise_dim(OP, dim_arg(args, kwargs, 2, "dim")?.unwrap_or(-1), rank)?;
    let largest = bool_arg(args, kwargs, 3, "largest")?.unwrap_or(true);
    // Read and discarded: see the section note. `sorted=False` licenses an
    // unspecified order and this shim answers with the sorted one, which is
    // within that licence.
    let _sorted = bool_arg(args, kwargs, 4, "sorted")?.unwrap_or(true);

    let extent = if rank == 0 { 1 } else { input.tensor()?.dims()[dim] };
    if k < 0 || k as usize > extent {
        // torch's own wording, and torch's own conflation of "negative" with
        // "too large" -- `topk(x, -1)` gives this same message upstream.
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "selected index k out of range",
        ));
    }
    let ordered = order_along(OP, &input, dim, largest, Some(k as usize))?;
    let device = input.tensor()?.device().clone();
    finish_ordered(py, OP, &TOPK_RESULT, "topk", ordered, input.tag(), &device)
}

/// `aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)`
///
/// The one thing to get right is that a dimension whose size is not 1 is a
/// **no-op, not an error** -- `squeeze(1)` on a `(1,3,1,2)` tensor answers
/// `(1,3,1,2)` unchanged (measured). Refusing there would break the generation
/// loop's `next_tokens.squeeze(1)` the moment a batch had one row.
fn squeeze_dim(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.squeeze.dim";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    let dim = normalise_dim(
        OP,
        dim_arg(args, kwargs, 1, "dim")?.ok_or_else(|| missing(OP, "dim"))?,
        rank,
    )?;
    // A 0-d tensor has nothing to remove; `normalise_dim` already accepted
    // dim 0 and -1 on it, as torch does.
    if rank == 0 {
        return finish(py, input.tensor()?.clone(), input.tag());
    }
    // candle's `squeeze` carries the same "size != 1 is a no-op" rule and does
    // it without a contiguous copy, so a strided view survives.
    let out = input
        .tensor()?
        .squeeze(dim)
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, input.tag())
}

/// `aten::split.Tensor(Tensor(a -> *) self, SymInt split_size, int dim=0)
///     -> Tensor(a)[]`
///
/// The only op in this file that answers with a **list**, which is a fact about
/// the call and not a detail: GPT-2's attention is `c_attn(x).split(n, dim=2)`,
/// so the three-way unpack on the Python side is the op's whole purpose.
/// `promote` at the dispatcher's exit does not look inside a list, so each
/// chunk is promoted here -- the same reason `max.dim` promotes its own pair.
///
/// Measured against torch 2.13.0, including the parts that read like edge cases
/// and are not:
///
///   * the **last chunk is short**, not padded: `split(arange(10), 3)` is four
///     chunks sized 3, 3, 3, 1.
///   * a `split_size` larger than the dimension gives **one** chunk, the whole
///     tensor -- not an error.
///   * `split_size == 0` is an error *unless* the dimension is empty, and the
///     two refusals have different wording ("split_size can only be 0 if
///     dimension size is 0, but got dimension size of 10" vs "split expects
///     split_size be non-negative, but got split_size=-1").
///   * an **empty dimension** gives one empty chunk for any `split_size`,
///     including 0.
///   * a 0-d tensor raises rather than answering with itself.
///
/// **Not reproduced: aliasing.** Upstream's chunks are views -- writing to
/// `split(x, 3)[0][0]` changes `x`. candle's `narrow` gives a view too, but
/// whether a write through this shim's `TensorBase` reaches the source is the
/// same unanswered question `aten.slice.Tensor` already has, and this op does
/// not settle it.
fn split_tensor(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.split.Tensor";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    if rank == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "split expects at least a 1-dimensional tensor",
        ));
    }
    let split_size = int_arg(args, kwargs, 1, "split_size")?.ok_or_else(|| missing(OP, "split_size"))?;
    let dim = normalise_dim(OP, dim_arg(args, kwargs, 2, "dim")?.unwrap_or(0), rank)?;
    let extent = input.tensor()?.dims()[dim];

    if split_size < 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "split expects split_size be non-negative, but got split_size={split_size}"
        )));
    }
    let split_size = split_size as usize;
    if split_size == 0 && extent != 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "split_size can only be 0 if dimension size is 0, but got dimension size of {extent}"
        )));
    }

    let mut chunks: Vec<Py<PyAny>> = Vec::new();
    if extent == 0 {
        // One empty chunk, whatever the split size -- measured for both
        // `split_size == 0` and `split_size == 3`.
        chunks.push(crate::tensor::promote(
            py,
            finish(py, input.tensor()?.clone(), input.tag())?,
        )?);
    } else {
        let mut start = 0usize;
        while start < extent {
            let length = split_size.min(extent - start);
            let chunk = input
                .tensor()?
                .narrow(dim, start, length)
                .map_err(|e| candle_err(OP, e))?;
            chunks.push(crate::tensor::promote(py, finish(py, chunk, input.tag())?)?);
            start += length;
        }
    }
    Ok(PyList::new(py, chunks)?.into_any().unbind())
}

/// `aten::split_with_sizes(Tensor(a -> *) self, SymInt[] split_sizes, int
///     dim=0) -> Tensor(a)[]`
///
/// `split.Tensor` with the chunk sizes spelled out individually rather than
/// as one repeated size -- the spelling `gpt_bigcode` (docs/TAIL.md) reaches
/// for `c_attn(x).split((embed_dim, kv_dim, kv_dim), dim=2)`, an *uneven*
/// three-way unpack (query gets the full embedding width, key and value share
/// a narrower one under multi-query attention) that `split.Tensor`'s single
/// repeated size cannot express. `methods.json` already spells this op to a
/// single kernel key (docs/SPELLINGS.md §4); only the kernel was missing.
///
/// Measured against torch 2.13.0:
///
///   * sizes must be **non-negative** and **sum exactly** to the dimension's
///     extent -- unlike `split.Tensor`, there is no "last chunk is short"
///     leniency here, because the caller already spelled out every length.
///     Both refusals are reproduced (`split_with_sizes expects split_sizes
///     have only non-negative entries...` / `...expects split_sizes to sum
///     exactly to N...`).
///   * an **individual size of 0 is fine** even when the dimension itself is
///     not empty, as long as the sizes still sum correctly -- `split(arange(10),
///     [0, 10], 0)` gives an empty first chunk and the whole tensor as the
///     second, both measured.
///   * a **0-d tensor raises**, the same refusal `split.Tensor` gives and the
///     same wording ("split expects at least a 1-dimensional tensor") --
///     upstream's message does not distinguish the two overloads.
fn split_with_sizes(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.split_with_sizes.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let rank = input.tensor()?.rank();
    if rank == 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "split expects at least a 1-dimensional tensor",
        ));
    }
    let sizes_raw: Vec<i64> = required(OP, args, kwargs, 1, "split_sizes")?.extract()?;
    let dim = normalise_dim(OP, dim_arg(args, kwargs, 2, "dim")?.unwrap_or(0), rank)?;
    let extent = input.tensor()?.dims()[dim];

    if sizes_raw.iter().any(|&s| s < 0) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "split_with_sizes expects split_sizes have only non-negative entries, but got split_sizes={sizes_raw:?}"
        )));
    }
    let total: i64 = sizes_raw.iter().sum();
    if total as usize != extent {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "split_with_sizes expects split_sizes to sum exactly to {extent} (input tensor's \
             size at dimension {dim}), but got split_sizes={sizes_raw:?}"
        )));
    }

    let mut chunks: Vec<Py<PyAny>> = Vec::new();
    let mut start = 0usize;
    for &size in &sizes_raw {
        let length = size as usize;
        let chunk = input
            .tensor()?
            .narrow(dim, start, length)
            .map_err(|e| candle_err(OP, e))?;
        chunks.push(crate::tensor::promote(py, finish(py, chunk, input.tag())?)?);
        start += length;
    }
    Ok(PyList::new(py, chunks)?.into_any().unbind())
}

// ---------------------------------------------------------------------------
// mamba / mixtral -- the last two of the 20 measured architectures
// (docs/OPS4.md) with anything unimplemented. Traced with a real
// `TorchDispatchMode` over `transformers` 5.15.1 + torch 2.13.0 rather than
// read off a doc comment: docs/OPS4.md's own §0 note is that doc comments
// have been wrong about upstream three times before, so every rule below was
// re-measured, not copied from a kernel's docstring.
// ---------------------------------------------------------------------------

/// `aten::softplus(Tensor self, Scalar beta=1, Scalar threshold=20) -> Tensor`
///
/// `mamba`'s selective-scan `dt` (the discretisation step size) is
/// `softplus(dt_proj(x) + dt_bias)`, always `float32`, always the default
/// `beta`/`threshold` (measured: `softplus.default(float32(1,128,6))`, two
/// positional args absent).
///
/// **Not implemented for integral/boolean input** -- measured on real torch,
/// `softplus_cpu` raises `NotImplementedError` naming the dtype rather than
/// promoting the way `exp`/`tanh` do; softplus is refused, not widened.
///
/// The formula is upstream's numerically-stable split, not the naive
/// `log(1+exp(beta*x))/beta` a doc comment would suggest: writing `y =
/// beta*x`, `log(1+exp(y)) == max(y,0) + log(1+exp(-|y|))`, which never
/// overflows `exp` for large `y` and keeps `log`'s argument in `[1,2]` (never
/// the near-`1.0` region where `log(1+tiny)` loses precision). Above
/// `threshold`, upstream skips the formula entirely and returns `x` itself,
/// not an evaluation of it -- measured `softplus(20.1) == 20.1` exactly,
/// which the formula alone would not promise. Every measured call in `mamba`
/// stays well inside the default `threshold=20`, so that branch is exercised
/// by the golden cases, not by the model.
fn softplus_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.softplus.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    if !input.tag().is_floating_point() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"softplus_cpu\" not implemented for '{}'",
            scalar_type_name(input.tag())
        )));
    }
    let beta = scalar_arg(OP, args, kwargs, 1, "beta")?
        .map(Scalar::as_f64)
        .unwrap_or(1.0);
    let threshold = scalar_arg(OP, args, kwargs, 2, "threshold")?
        .map(Scalar::as_f64)
        .unwrap_or(20.0);

    let tag = input.tag();
    let x = input.tensor()?.clone();
    let y = x.affine(beta, 0.0).map_err(|e| candle_err(OP, e))?;
    let abs_y = y.abs().map_err(|e| candle_err(OP, e))?;
    // max(y, 0) == (y + |y|) / 2 -- avoids needing a `maximum` against a
    // freshly-built zero tensor.
    let positive = y
        .add(&abs_y)
        .and_then(|t| t.affine(0.5, 0.0))
        .map_err(|e| candle_err(OP, e))?;
    let log_term = abs_y
        .affine(-1.0, 0.0)
        .and_then(|t| t.exp())
        .and_then(|t| t.affine(1.0, 1.0))
        .and_then(|t| t.log())
        .map_err(|e| candle_err(OP, e))?;
    let full_formula = positive
        .add(&log_term)
        .and_then(|t| t.affine(1.0 / beta, 0.0))
        .map_err(|e| candle_err(OP, e))?;
    let over_threshold = y.gt(threshold).map_err(|e| candle_err(OP, e))?;
    let out = over_threshold
        .where_cond(&x, &full_formula)
        .map_err(|e| candle_err(OP, e))?;
    finish(py, out, tag)
}

/// `aten::convolution(Tensor input, Tensor weight, Tensor? bias, SymInt[]
///     stride, SymInt[] padding, SymInt[] dilation, bool transposed,
///     SymInt[] output_padding, SymInt groups) -> Tensor`
///
/// `mamba`'s short causal depthwise conv over the SSM's `(x, B, C)` sequence:
/// measured `convolution.default(float32(1,128,6), float32(128,1,4),
/// float32(128,), stride=[1], padding=[3], dilation=[1], transposed=False,
/// output_padding=[0], groups=128)` -- `groups == in_channels == out_channels`
/// (depthwise), `padding == kernel_size - 1` on *both* sides (the caller
/// slices `[..., :seq_len]` afterwards to keep it causal, which is the
/// model's job, not this kernel's).
///
/// Only that shape is implemented: **not transposed**, a zero
/// `output_padding` (irrelevant when not transposed, but checked so an
/// unmeasured combination fails loudly rather than being silently ignored),
/// a 3-D input (`(batch, channels, length)` -- the 1-D conv `nn.Conv1d`
/// lowers to), and floating dtypes only. `candle_core::Tensor::conv1d`
/// already accepts `groups`, and its symmetric `padding` argument is the
/// same convention `aten::convolution` uses for a non-transposed 1-D
/// convolution, so no reshaping trick is needed to reuse it.
///
/// Bias is `(out_channels,)` and is not something `conv1d` applies itself --
/// added afterwards, reshaped to `(1, out_channels, 1)` to broadcast over the
/// batch and length axes.
fn convolution_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.convolution.default";
    let input = tensor_arg(OP, args, kwargs, 0, "input")?;
    let weight = tensor_arg(OP, args, kwargs, 1, "weight")?;
    let bias = optional_tensor_arg(OP, args, kwargs, 2, "bias")?;
    let stride = shape_arg(OP, args, kwargs, 3, "stride")?;
    let padding = shape_arg(OP, args, kwargs, 4, "padding")?;
    let dilation = shape_arg(OP, args, kwargs, 5, "dilation")?;
    let transposed = bool_arg(args, kwargs, 6, "transposed")?.unwrap_or(false);
    let output_padding = shape_arg(OP, args, kwargs, 7, "output_padding")?;
    let groups = required(OP, args, kwargs, 8, "groups")?.extract::<i64>()?;

    if transposed {
        return Err(not_implemented(format!(
            "{OP}: transposed convolution not implemented in torch._C shim"
        )));
    }
    if output_padding.iter().any(|&v| v != 0) {
        return Err(not_implemented(format!(
            "{OP}: a non-zero output_padding is not implemented in torch._C shim \
             (only meaningful for transposed convolution, which is also not implemented)"
        )));
    }
    let rank = input.tensor()?.rank();
    if rank != 3 {
        return Err(not_implemented(format!(
            "{OP}: only 1-D convolution (3-D input, (batch, channels, length)) is \
             implemented in torch._C shim, got {rank}-D"
        )));
    }
    if stride.len() != 1 || padding.len() != 1 || dilation.len() != 1 {
        return Err(not_implemented(format!(
            "{OP}: only a single-element stride/padding/dilation (1-D convolution) is \
             implemented in torch._C shim"
        )));
    }
    if stride[0] <= 0 || padding[0] < 0 || dilation[0] <= 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{OP}: stride and dilation must be positive, padding must be non-negative"
        )));
    }
    if groups <= 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{OP}: groups must be a positive integer"
        )));
    }

    let tag = same_dtype(OP, &input, &weight)?;
    if !tag.is_floating_point() {
        return Err(not_implemented(format!(
            "{OP}: only floating-point convolution is implemented in torch._C shim, \
             got {}",
            scalar_type_name(tag)
        )));
    }
    let storage = PyDtype::new(tag).storage(OP)?;
    let x = input.tensor()?.to_dtype(storage).map_err(|e| candle_err(OP, e))?;
    let w = weight.tensor()?.to_dtype(storage).map_err(|e| candle_err(OP, e))?;
    let raw = x
        .conv1d(
            &w,
            padding[0] as usize,
            stride[0] as usize,
            dilation[0] as usize,
            groups as usize,
        )
        .map_err(|e| candle_err(OP, e))?;
    let out = match bias {
        Some(b) => {
            if b.tag() != tag {
                return Err(not_implemented(format!(
                    "{OP}: bias dtype must match input/weight dtype in torch._C shim"
                )));
            }
            let c_out = raw.dim(1).map_err(|e| candle_err(OP, e))?;
            let b_reshaped = b
                .tensor()?
                .to_dtype(storage)
                .and_then(|t| t.reshape((1, c_out, 1)))
                .map_err(|e| candle_err(OP, e))?;
            raw.broadcast_add(&b_reshaped).map_err(|e| candle_err(OP, e))?
        }
        None => raw,
    };
    finish(py, out, tag)
}

/// `aten::zeros_like`/`aten::empty_like(Tensor self, *, ScalarType? dtype=None,
///     Layout? layout=None, Device? device=None, bool? pin_memory=None,
///     MemoryFormat? memory_format=None) -> Tensor`
///
/// `mamba`'s selective-scan state is seeded with `zeros_like(...)`, and
/// `mixtral`'s grouped-MoE routing (`transformers`'
/// `integrations/moe.py::grouped_mm_experts_forward`) allocates
/// `torch.empty_like(perm)` purely to be overwritten in full two lines later
/// (`inv_perm[perm] = torch.arange(...)`) -- so, exactly like
/// `empty.memory_format` above, "empty" answers zeros here: the shim is
/// deterministic where upstream is not, and both measured call sites read
/// every element back before using it.
///
/// Structured like `new_ones_default`: the reference tensor supplies the
/// defaults (shape always, dtype/device unless overridden) a bare factory
/// would otherwise take from the process-wide default.
fn zeros_or_empty_like(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
) -> PyResult<Py<PyAny>> {
    let input = tensor_arg(op, args, kwargs, 0, "self")?;
    let tag = dtype_arg(args, kwargs, 1, "dtype")?.unwrap_or(input.tag());
    reject_unsupported(
        op,
        args,
        kwargs,
        &[(2, "layout"), (4, "pin_memory"), (5, "memory_format")],
    )?;
    let label = device_arg_or_label(args, kwargs, 3, "device", &input.device_label())?;
    let shape = input.tensor()?.dims().to_vec();
    if label.is_meta() {
        return meta_result(py, shape, tag);
    }
    let device = label.resolve()?;
    let storage = PyDtype::new(tag).storage(op)?;
    let out = Tensor::zeros(shape, storage, &device).map_err(|e| candle_err(op, e))?;
    finish(py, out, tag)
}

/// `aten::floor_divide(Tensor self, Tensor other) -> Tensor`
///
/// `mixtral`'s grouped-MoE routing recovers each selected token's row with
/// `hidden_states[perm // num_top_k]` -- `perm` an `int64` tensor, `//` a
/// **Python `int`**, not a tensor. Measured with a real `TorchDispatchMode`:
/// the dispatched call is `floor_divide.default(int64_tensor, 2)`, a bare
/// Python `int` reaching the `(Tensor, Tensor)` overload's `other` slot
/// rather than the `floor_divide.Scalar` overload the schema alternative
/// would suggest -- torch's own frontend picks `.default` here and leaves the
/// scalar-to-tensor conversion for the kernel, not the dispatcher. Both
/// shapes are accepted below.
///
/// **Floors toward negative infinity, matching Python's `//` -- not C's
/// truncation.** Measured on torch 2.13.0 with mixed-sign input: `[-7, -6,
/// -1, 0, 1, 6, 7] // 2 == [-4, -3, -1, 0, 0, 3, 3]` (not `[-3, -3, 0, 0, 0,
/// 3, 3]`, which is what truncating division would give for the negative
/// entries). Implemented as truncating division with the standard correction
/// (subtract one from the truncated quotient when the remainder is non-zero
/// and its sign disagrees with the divisor's), which reproduces the measured
/// table exactly.
///
/// **Dtype is preserved, not promoted** -- unlike `div.Tensor`'s true
/// division, `int64 // int64` stays `int64` (measured), so this does not
/// route through `arith_tag`.
///
/// **Division by zero on an integral dtype raises**, matching upstream's
/// measured `RuntimeError('ZeroDivisionError')` exactly (message and all) --
/// checked eagerly, before any division happens, so every element of a
/// zero-divisor call fails together the way upstream's does. Division by
/// zero on a floating dtype is not refused (`1.0 // 0.0 == inf`, `-1.0 //
/// 0.0 == -inf`, `0.0 // 0.0 == nan`, all measured) -- ordinary `f64`
/// division already answers this correctly, no special case needed.
fn floor_divide_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.floor_divide.default";
    let lhs = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = lhs.tag();
    if tag == TorchDType::Bool {
        return Err(not_implemented(format!(
            "{OP}: torch.bool operands are not implemented in torch._C shim"
        )));
    }
    let raw_other = required(OP, args, kwargs, 1, "other")?;
    let n = lhs.tensor()?.elem_count();
    let other_flat: Flat = if let Ok(other_tensor) = raw_other.extract::<PyTensorBase>() {
        if other_tensor.tag() != tag {
            return Err(not_implemented(format!(
                "{OP}: dtype promotion not implemented in torch._C shim: {} vs {}",
                tag.name(),
                other_tensor.tag().name()
            )));
        }
        let flat = read_flat(OP, other_tensor.tensor()?, tag)?;
        let count = other_tensor.tensor()?.elem_count();
        if count != n && count != 1 {
            return Err(not_implemented(format!(
                "{OP}: broadcasting other than a scalar or an exact shape match is not \
                 implemented in torch._C shim"
            )));
        }
        flat
    } else {
        let scalar = scalar_arg(OP, args, kwargs, 1, "other")?.ok_or_else(|| missing(OP, "other"))?;
        if tag.is_floating_point() {
            Flat::Float(vec![scalar.as_f64()])
        } else {
            Flat::Int(vec![scalar.as_i64()])
        }
    };

    let self_flat = read_flat(OP, lhs.tensor()?, tag)?;
    let out_flat = match (self_flat, other_flat) {
        (Flat::Float(a), Flat::Float(b)) => {
            let get = |i: usize| if b.len() == 1 { b[0] } else { b[i] };
            Flat::Float(a.iter().enumerate().map(|(i, &x)| (x / get(i)).floor()).collect())
        }
        (Flat::Int(a), Flat::Int(b)) => {
            let get = |i: usize| if b.len() == 1 { b[0] } else { b[i] };
            let mut out = Vec::with_capacity(a.len());
            for (i, &x) in a.iter().enumerate() {
                let d = get(i);
                if d == 0 {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err("ZeroDivisionError"));
                }
                let q = x / d;
                let r = x % d;
                out.push(if r != 0 && (r < 0) != (d < 0) { q - 1 } else { q });
            }
            Flat::Int(out)
        }
        _ => unreachable!("self and other share a dtype, checked above"),
    };

    let dims = lhs.tensor()?.dims().to_vec();
    let device = lhs.tensor()?.device().clone();
    let out = write_flat(OP, out_flat, dims, &device, tag)?;
    finish(py, out, tag)
}

/// `aten::histc(Tensor self, int bins=100, Scalar min=0, Scalar max=0) -> Tensor`
///
/// `mixtral`'s grouped-MoE routing counts tokens per expert with
/// `torch.histc(expert_ids_g.float(), bins=num_experts, min=0,
/// max=num_experts-1)` (`transformers`' `integrations/moe.py`, CPU path --
/// the comment there notes `histc` doesn't support integer dtypes on CPU,
/// which measured confirms: `NotImplementedError("histogram_cpu" not
/// implemented for 'Long')`, so only floating input is implemented here).
///
/// **`min == max` means "ignore both and use the data's own min/max"** --
/// measured on torch 2.13.0 with *non-zero* equal bounds too
/// (`histc(x, min=5, max=5)` on data ranging `[1,3]` bins over `[1,3]`, not
/// `[5,5]`), so the rule is "min equals max", not "both are zero". A
/// genuinely degenerate range (the data itself is constant) is a second,
/// nested case: measured `histc([2,2,2], bins=4, min=0, max=0)` answers
/// `[0,0,3,0]`, which is exactly what falling back to `[value-1, value+1]`
/// produces (bin width `0.5`, `2.0` lands in `[2.0,2.5)`, the third of four
/// bins) -- reproduced here as the same fallback applied twice rather than
/// as a separately-derived formula.
///
/// Range is inclusive on both ends (`x == max` counts in the last bin,
/// measured), and elements strictly outside `[min, max]` are dropped, not
/// clamped into an edge bin (measured: `-1` and `4` are absent from
/// `histc([..., -1, 4], bins=4, min=0, max=3)`'s counts).
///
/// `bins <= 0` and an explicit `min > max` are refused with upstream's exact
/// wording (both measured): `"bins must be > 0, but got {bins} for dimension
/// 0"` and `"torch.histc: max must be larger than min"`.
fn histc_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.histc.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let tag = input.tag();
    if !tag.is_floating_point() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"histogram_cpu\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }
    let bins = int_arg(args, kwargs, 1, "bins")?.unwrap_or(100);
    if bins <= 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "torch.histogram(): bins must be > 0, but got {bins} for dimension 0"
        )));
    }
    let min_arg = scalar_arg(OP, args, kwargs, 2, "min")?.map(Scalar::as_f64).unwrap_or(0.0);
    let max_arg = scalar_arg(OP, args, kwargs, 3, "max")?.map(Scalar::as_f64).unwrap_or(0.0);

    let values = read_flat(OP, input.tensor()?, tag)?;
    let values = match values {
        Flat::Float(v) => v,
        Flat::Int(_) => unreachable!("floating dtype checked above"),
    };

    let (mut lo, mut hi) = if min_arg == max_arg {
        let mut data_lo = f64::INFINITY;
        let mut data_hi = f64::NEG_INFINITY;
        for &v in &values {
            if v < data_lo {
                data_lo = v;
            }
            if v > data_hi {
                data_hi = v;
            }
        }
        if values.is_empty() {
            (0.0, 0.0)
        } else {
            (data_lo, data_hi)
        }
    } else {
        (min_arg, max_arg)
    };
    if lo > hi {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "torch.histc: max must be larger than min",
        ));
    }
    if lo == hi {
        lo -= 1.0;
        hi += 1.0;
    }

    let mut counts = vec![0.0f64; bins as usize];
    let width = (hi - lo) / bins as f64;
    for &v in &values {
        if v < lo || v > hi || v.is_nan() {
            continue;
        }
        let mut bin = ((v - lo) / width).floor() as i64;
        if bin >= bins {
            bin = bins - 1;
        }
        if bin < 0 {
            bin = 0;
        }
        counts[bin as usize] += 1.0;
    }

    let device = input.tensor()?.device().clone();
    let out = write_flat(OP, Flat::Float(counts), vec![bins as usize], &device, tag)?;
    finish(py, out, tag)
}

/// `aten::clamp_(Tensor(a!) self, Scalar? min=None, Scalar? max=None) -> Tensor(a!)`
///
/// `mixtral`'s grouped-MoE routing keeps a per-row expert-bias gather
/// in-bounds with `expert_ids_g.clamp_(max=self.num_experts - 1)` (`min`
/// absent) -- measured `clamp_.default(int64(12,), None, 3)`.
///
/// **The formula is `min(max(x, min_val), max_val)`, applied in that order
/// unconditionally** -- not "refuse if `min > max`". Measured:
/// `x.clamp_(min=8, max=2)` on `[1,5,10,-3]` gives `[2,2,2,2]` (every element
/// hits the floor first, then the ceiling clips it down again), which is
/// exactly what `candle_core::Tensor::clamp` already computes
/// (`maximum(min).minimum(max)`), so this reuses it rather than reproducing
/// it by hand. NaN propagates through both steps for the same reason
/// (Rust's `<`/`>` are false against NaN, so `maximum`/`minimum` return
/// whichever operand *is* NaN) -- measured `[nan,1.,-1.].clamp_(0,2) ==
/// [nan,1.,0.]`.
///
/// **A float bound against an integral receiver is refused outright,
/// regardless of the bound's actual value** -- measured `int32.clamp_(max=2.0)`
/// raises `"result type Float can't be cast to the desired output type
/// Int"` even though `2.0` is exactly representable as an int; torch does
/// not special-case exact values, so this does not either.
///
/// **Both bounds absent is refused, not a no-op** -- measured
/// `tensor.clamp_(None, None)` raises `"torch.clamp: At least one of 'min'
/// or 'max' must not be None"` rather than returning the receiver
/// unchanged, which "nothing to clamp against" would otherwise suggest.
fn clamp_inplace_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.clamp_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let min = scalar_arg(OP, args, kwargs, 1, "min")?;
    let max = scalar_arg(OP, args, kwargs, 2, "max")?;
    if min.is_none() && max.is_none() {
        // Measured, not a guess this shim could have gotten away with
        // skipping: `tensor.clamp_(None, None)` raises on real torch rather
        // than being an accepted no-op.
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "torch.clamp: At least one of 'min' or 'max' must not be None",
        ));
    }
    let tag = receiver.borrow().tag();
    if !tag.is_floating_point() {
        for bound in [min, max].into_iter().flatten() {
            if !bound.is_int() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "result type Float can't be cast to the desired output type {}",
                    scalar_type_name(tag)
                )));
            }
        }
    }
    let source = receiver.borrow().tensor()?.clone();
    let mut out = source;
    if let Some(bound) = min {
        out = if tag.is_floating_point() {
            out.maximum(bound.as_f64())
        } else {
            out.maximum(bound.as_i64())
        }
        .map_err(|e| candle_err(OP, e))?;
    }
    if let Some(bound) = max {
        out = if tag.is_floating_point() {
            out.minimum(bound.as_f64())
        } else {
            out.minimum(bound.as_i64())
        }
        .map_err(|e| candle_err(OP, e))?;
    }
    receiver.borrow_mut().replace_with(PyTensorBase::new(out)?);
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// `aten::div_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)`
///
/// `mixtral`'s router normalises top-k weights in place:
/// `top_k_weights.div_(top_k_weights.sum(dim=-1, keepdim=True))`, measured
/// `div_.Tensor(float32(6,2), float32(6,1))` -- `other` broadcasting *into*
/// the receiver's shape, never the other way around, which is in-place's
/// general rule (`add_inplace` above) and is exactly what this call needs
/// (`(6,1)` into `(6,2)`).
///
/// **True division, and the receiver's dtype cannot change to accommodate
/// it** -- unlike `div.Tensor`, which promotes an integral pair to
/// `float32` (`arith_tag`), the in-place form has nowhere to put a wider
/// result: measured `int64_tensor.div_(int64_tensor)` raises `"result type
/// Float can't be cast to the desired output type Long"`. So this refuses
/// non-floating receivers by name rather than silently truncating back to
/// int, which would be a wrong answer with no trace.
fn div_inplace_tensor(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.div_.Tensor";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let other = tensor_arg(OP, args, kwargs, 1, "other")?;

    let (tag, shape) = {
        let borrowed = receiver.borrow();
        (borrowed.tag(), borrowed.tensor()?.shape().clone())
    };
    if !tag.is_floating_point() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "result type Float can't be cast to the desired output type {}",
            scalar_type_name(tag)
        )));
    }
    let storage = PyDtype::new(tag).storage(OP)?;
    let lhs = {
        let borrowed = receiver.borrow();
        borrowed.tensor()?.to_dtype(storage).map_err(|e| candle_err(OP, e))?
    };
    let rhs = other
        .tensor()?
        .to_dtype(storage)
        .and_then(|t| t.broadcast_as(shape))
        .and_then(|t| t.contiguous())
        .map_err(|e| candle_err(OP, e))?;
    let out = lhs.broadcast_div(&rhs).map_err(|e| candle_err(OP, e))?;
    receiver.borrow_mut().replace_with(PyTensorBase::new(out)?);
    let _ = py;
    Ok(receiver.into_any().unbind())
}

/// `aten::masked_fill_.Scalar(Tensor(a!) self, Tensor mask, Scalar value) -> Tensor(a!)`
///
/// `mixtral`'s grouped-MoE routing zeroes sentinel rows twice, once each
/// side of the grouped matmul: `selected_hidden_states_g.masked_fill_(
/// sentinel_mask, 0.0)` and `weighted_out.masked_fill_(sentinel_mask, 0.0)`
/// (`transformers`' `integrations/moe.py`) -- measured
/// `masked_fill_.Scalar(float32(12,64), bool(12,1), 0.0)`, the mask
/// broadcasting from `(12,1)` into the receiver's `(12,64)`.
///
/// Not a new kernel: the value and every refusal are `masked_fill.Scalar`'s
/// (a `torch.bool` mask required, same as that op's doc comment measures),
/// computed once and written into the receiver via `replace_with` -- the
/// same "aliases created before this call do not see the write" limitation
/// `add_inplace`'s doc comment already states for every in-place op in this
/// file.
fn masked_fill_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    op: &str,
) -> PyResult<Py<PyAny>> {
    let receiver = tensor_receiver(op, args, kwargs)?;
    let result = masked_fill(py, args, kwargs, op)?;
    let replacement = result.extract::<PyTensorBase>(py)?;
    receiver.borrow_mut().replace_with(replacement);
    Ok(receiver.into_any().unbind())
}

/// `aten::index_put_(Tensor(a!) self, Tensor?[] indices, Tensor values, bool
///     accumulate=False) -> Tensor(a!)`
///
/// `mixtral`'s grouped-MoE routing builds the inverse of a sort permutation
/// with `inv_perm[perm] = torch.arange(perm.size(0))` (`transformers`'
/// `integrations/moe.py`), which lowers to `index_put_` with a single index
/// tensor -- measured `index_put_.default(int64(12,), [int64(12,)],
/// int64(12,), accumulate=False (default, absent))`.
///
/// **Restricted to exactly the shape measured**: one non-`None` index
/// tensor, `self`/`index`/`values` all rank 1 with matching element counts,
/// `accumulate=False`. That single-index, non-accumulating, 1-D case is
/// `self[index[i]] = values[i]` for each `i` -- which is `scatter.src` along
/// dimension 0 with `index` and `values` standing in for `scatter`'s `index`
/// and `src` (both already require `self`'s dtype and an int32/int64 index,
/// which is exactly what `index_put_`'s schema also demands here). Rather
/// than re-deriving that arithmetic, this builds the `(dim=0, index, src)`
/// call `scatter.src` already implements and writes the result back into the
/// receiver through `replace_with`. A wider index list, `accumulate=True`,
/// or non-1-D operands are refused by name: not measured, so not guessed at.
fn index_put_inplace(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.index_put_.default";
    let receiver = tensor_receiver(OP, args, kwargs)?;
    let raw_indices = required(OP, args, kwargs, 1, "indices")?;
    let items: Vec<Bound<'_, PyAny>> = raw_indices.extract()?;
    let values = tensor_arg(OP, args, kwargs, 2, "values")?;
    let accumulate = bool_arg(args, kwargs, 3, "accumulate")?.unwrap_or(false);
    if accumulate {
        return Err(not_implemented(format!(
            "{OP}: accumulate=True is not implemented in torch._C shim"
        )));
    }

    let mut chosen: Option<PyTensorBase> = None;
    for item in &items {
        if item.is_none() {
            continue;
        }
        let tensor = item.extract::<PyTensorBase>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "{OP}: indices must be tensors or None, got {}",
                item.get_type().name().map(|n| n.to_string()).unwrap_or_default()
            ))
        })?;
        if chosen.is_some() {
            return Err(not_implemented(format!(
                "{OP}: more than one index tensor is not implemented in torch._C shim"
            )));
        }
        chosen = Some(tensor);
    }
    let index = chosen.ok_or_else(|| {
        not_implemented(format!("{OP}: an all-None index list is not implemented in torch._C shim"))
    })?;

    let self_rank = receiver.borrow().tensor()?.rank();
    if self_rank != 1 || index.tensor()?.rank() != 1 || values.tensor()?.rank() != 1 {
        return Err(not_implemented(format!(
            "{OP}: only a 1-D self/index/values is implemented in torch._C shim"
        )));
    }

    // `scatter.src`'s own dim/dtype/index-dtype rules apply unchanged --
    // build the call it expects and let it do the work.
    let scatter_args = PyTuple::new(
        py,
        [
            receiver.clone().into_any(),
            0i64.into_pyobject(py)?.into_any(),
            index.into_pyobject(py)?.into_any(),
            values.into_pyobject(py)?.into_any(),
        ],
    )?;
    let result = scatter_src(py, &scatter_args, None)?;
    let replacement = result.extract::<PyTensorBase>(py)?;
    receiver.borrow_mut().replace_with(replacement);
    Ok(receiver.into_any().unbind())
}

/// `aten::native_layer_norm(Tensor input, SymInt[] normalized_shape,
///     Tensor? weight, Tensor? bias, float eps) -> (Tensor, Tensor, Tensor)`
///
/// GPT-2's normalisation, and structurally not Llama's: RMSNorm is
/// `mean.dim` + `rsqrt` + `mul` through the ordinary dispatcher, while
/// `LayerNorm` is **one fused op that returns three tensors** -- the output
/// plus the `mean` and `rstd` the backward pass would need. There is no
/// autograd here, but the two extra results are part of the schema, so they are
/// computed and returned rather than filled with zeros.
///
/// Measured against torch 2.13.0. The parts that inference gets wrong:
///
///   * **`mean`/`rstd` are not flat.** They keep the input's rank with the
///     normalised dimensions replaced by 1: `(2,3,4)` with
///     `normalized_shape=[4]` gives `(2,3,1)`, and with `[3,4]` gives
///     `(2,1,1)`.
///   * **the variance is biased** (divided by N, not N-1), and `eps` is added
///     to it *before* the reciprocal square root, not to the standard
///     deviation. A constant row therefore gives `rstd = 1/sqrt(eps)`
///     (`316.2278` at `eps=1e-5`), which is what pins the order.
///   * **`mean`/`rstd` follow the *parameter* dtype, not the input's.** A
///     `float16` input with `float16` (or absent) parameters gives `float16`
///     statistics; the same input with `float32` parameters gives `float32`
///     ones while the output stays `float16`. That is upstream's mixed-dtype
///     autocast path, and it is a *supported* combination, not an error.
///   * a **negative `eps` is not refused** -- it gives NaN, and this follows.
///
/// Refusals copied rather than invented: an integral or boolean input raises
/// `NotImplementedError` naming `LayerNormKernelImpl`; parameters that are
/// neither the input dtype nor the `float32` autocast partner raise
/// `mixed dtype (CPU): ...`, in two different wordings depending on whether the
/// input was `float64`.
///
/// **Not implemented: a zero-extent normalized dimension.** `normalized_shape=[0]`
/// makes upstream answer `mean=0, rstd=nan` -- a mean over no elements that is
/// zero on one side of the pair and NaN on the other. Reproducing an internal
/// inconsistency from one observation is guessing, so it is refused by name.
fn native_layer_norm_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.native_layer_norm.default";
    let input = tensor_arg(OP, args, kwargs, 0, "input")?;
    let normalized = shape_arg(OP, args, kwargs, 1, "normalized_shape")?;
    let weight = optional_tensor_arg(OP, args, kwargs, 2, "weight")?;
    let bias = optional_tensor_arg(OP, args, kwargs, 3, "bias")?;
    let eps = scalar_arg(OP, args, kwargs, 4, "eps")?
        .map(|s| s.as_f64())
        .ok_or_else(|| missing(OP, "eps"))?;

    if normalized.is_empty() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Expected normalized_shape to be at least 1-dimensional, i.e., containing at \
             least one element, but got normalized_shape = []",
        ));
    }

    let dims = input.tensor()?.dims().to_vec();
    let k = normalized.len();
    let suffix_matches = dims.len() >= k
        && dims[dims.len() - k..]
            .iter()
            .zip(normalized.iter())
            .all(|(&extent, &wanted)| wanted >= 0 && extent == wanted as usize);
    if !suffix_matches {
        let star: String = normalized.iter().map(|v| format!(", {v}")).collect();
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "Given normalized_shape={normalized:?}, expected input with shape [*{star}], \
             but got input of size{dims:?}"
        )));
    }
    let ns: Vec<usize> = normalized.iter().map(|&v| v as usize).collect();

    let tag = input.tag();
    if !tag.is_floating_point() {
        return Err(pyo3::exceptions::PyNotImplementedError::new_err(format!(
            "\"LayerNormKernelImpl\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }

    for (label, param) in [("weight", &weight), ("bias", &bias)] {
        if let Some(param) = param {
            if param.tensor()?.dims() != ns.as_slice() {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "Expected {label} to be of same shape as normalized_shape, but got \
                     {label} of shape {:?} and normalized_shape = {ns:?}",
                    param.tensor()?.dims()
                )));
            }
        }
    }

    // Upstream's rule, measured: the parameters agree with each other, and
    // they are either the input dtype or `float32` standing in front of a
    // reduced-precision input.
    let mixed_dtype = || {
        pyo3::exceptions::PyRuntimeError::new_err(if tag == TorchDType::Float64 {
            "mixed dtype (CPU): all inputs must share same datatype."
        } else {
            "mixed dtype (CPU): expect parameter to have scalar type of Float"
        })
    };
    let param_tag = match (&weight, &bias) {
        (Some(w), Some(b)) if w.tag() != b.tag() => return Err(mixed_dtype()),
        (Some(w), _) => Some(w.tag()),
        (None, Some(b)) => Some(b.tag()),
        (None, None) => None,
    };
    let mixed = match param_tag {
        None => false,
        Some(param) if param == tag => false,
        Some(TorchDType::Float32)
            if matches!(tag, TorchDType::Float16 | TorchDType::BFloat16) =>
        {
            true
        }
        Some(_) => return Err(mixed_dtype()),
    };

    let rows: usize = dims[..dims.len() - k].iter().product();
    let cols: usize = ns.iter().product();
    if cols == 0 {
        return Err(not_implemented(format!(
            "{OP}: a zero-extent normalized_shape ({ns:?}) is not implemented in torch._C \
             shim -- upstream answers mean=0 with rstd=nan there, and that pair was not \
             measured well enough to reproduce"
        )));
    }

    let storage = PyDtype::new(tag).storage(OP)?;
    // `opmath_type`: the reduced dtypes accumulate in `f32` and narrow once.
    let acc = match storage {
        candle_core::DType::F16 | candle_core::DType::BF16 => candle_core::DType::F32,
        other => other,
    };
    let stat_tag = if mixed { TorchDType::Float32 } else { tag };
    let stat_storage = PyDtype::new(stat_tag).storage(OP)?;
    let stat_dims: Vec<usize> = dims[..dims.len() - k]
        .iter()
        .copied()
        .chain(std::iter::repeat(1).take(k))
        .collect();
    let device = input.tensor()?.device().clone();

    // No rows to reduce over: every result is empty, and candle's reductions
    // have nothing to say about a zero-length axis.
    if rows == 0 {
        let empty = |shape: &[usize], dtype| {
            Tensor::zeros(shape, dtype, &device).map_err(|e| candle_err(OP, e))
        };
        let triple = [
            crate::tensor::promote(py, finish(py, empty(dims.as_slice(), storage)?, tag)?)?,
            crate::tensor::promote(
                py,
                finish(py, empty(stat_dims.as_slice(), stat_storage)?, stat_tag)?,
            )?,
            crate::tensor::promote(
                py,
                finish(py, empty(stat_dims.as_slice(), stat_storage)?, stat_tag)?,
            )?,
        ];
        return Ok(PyTuple::new(py, triple)?.into_any().unbind());
    }

    let flat = input
        .tensor()?
        .contiguous()
        .and_then(|t| t.to_dtype(acc))
        .and_then(|t| t.reshape((rows, cols)))
        .map_err(|e| candle_err(OP, e))?;
    let mean = flat.mean_keepdim(1).map_err(|e| candle_err(OP, e))?;
    let centred = flat.broadcast_sub(&mean).map_err(|e| candle_err(OP, e))?;
    let rstd = centred
        .sqr()
        .and_then(|t| t.mean_keepdim(1))
        // `var + eps` first, *then* rsqrt -- see the doc comment.
        .and_then(|t| t.affine(1.0, eps))
        .and_then(|t| t.sqrt())
        .and_then(|t| t.recip())
        .map_err(|e| candle_err(OP, e))?;

    let mut out = centred.broadcast_mul(&rstd).map_err(|e| candle_err(OP, e))?;
    if let Some(weight) = &weight {
        let row = weight
            .tensor()?
            .contiguous()
            .and_then(|t| t.to_dtype(acc))
            .and_then(|t| t.reshape((1, cols)))
            .map_err(|e| candle_err(OP, e))?;
        out = out.broadcast_mul(&row).map_err(|e| candle_err(OP, e))?;
    }
    if let Some(bias) = &bias {
        let row = bias
            .tensor()?
            .contiguous()
            .and_then(|t| t.to_dtype(acc))
            .and_then(|t| t.reshape((1, cols)))
            .map_err(|e| candle_err(OP, e))?;
        out = out.broadcast_add(&row).map_err(|e| candle_err(OP, e))?;
    }

    let out = out
        .to_dtype(storage)
        .and_then(|t| t.reshape(dims.as_slice()))
        .map_err(|e| candle_err(OP, e))?;
    let reshape_stat = |t: Tensor| {
        t.to_dtype(stat_storage)
            .and_then(|t| t.reshape(stat_dims.as_slice()))
            .map_err(|e| candle_err(OP, e))
    };
    let mean = reshape_stat(mean)?;
    let rstd = reshape_stat(rstd)?;

    // Promoted element by element: `promote` at the dispatcher's exit does not
    // look inside a tuple, the same reason `max.dim` promotes its own pair.
    let triple = [
        crate::tensor::promote(py, finish(py, out, tag)?)?,
        crate::tensor::promote(py, finish(py, mean, stat_tag)?)?,
        crate::tensor::promote(py, finish(py, rstd, stat_tag)?)?,
    ];
    Ok(PyTuple::new(py, triple)?.into_any().unbind())
}

/// `aten::_softmax(Tensor self, int dim, bool half_to_float) -> Tensor`
///
/// Two refusals are reproduced rather than papered over, both measured:
///
///   * `half_to_float=True` is **not supported on CPU at all**, for any dtype
///     -- `float16`, `bfloat16` and `float32` inputs all raise. It is a CUDA-only
///     fusion. A shim that quietly honoured it would return `float32` where
///     upstream raises.
///   * an integral input raises `NotImplementedError`, naming the kernel, not
///     a `RuntimeError`.
///
/// The accumulate type follows `opmath_type<scalar_t>` -- `float` for
/// `float16`/`bfloat16`/`float32`, `double` for `float64` -- so the reduced
/// dtypes compute in float and narrow once at the end, which is why they agree
/// with upstream to better than their own epsilon.
///
/// The max is subtracted before the exponential for the reason docs/OPS8.md §3
/// gives for the attention kernel: without it a masked `-inf` and a large logit
/// both come out NaN. With it, `exp(-inf - max)` is a clean zero. A row that is
/// *entirely* `-inf` still gives NaN on both sides, and a case pins that.
fn softmax_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten._softmax.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let dim_raw = dim_arg(args, kwargs, 1, "dim")?.ok_or_else(|| missing(OP, "dim"))?;
    let half_to_float = bool_arg(args, kwargs, 2, "half_to_float")?
        .ok_or_else(|| missing(OP, "half_to_float"))?;
    if half_to_float {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "softmax with half to float conversion is not supported on CPU",
        ));
    }
    let tag = input.tag();
    if !tag.is_floating_point() {
        return Err(not_implemented(format!(
            "\"softmax_lastdim_kernel_impl\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }
    let rank = input.tensor()?.rank();
    let dim = normalise_dim(OP, dim_raw, rank)?;

    let dims = input.tensor()?.dims().to_vec();
    let (outer, n, inner) = if dims.is_empty() {
        (1usize, 1usize, 1usize)
    } else {
        (
            dims[..dim].iter().product::<usize>(),
            dims[dim],
            dims[dim + 1..].iter().product::<usize>(),
        )
    };

    let source = match read_flat(OP, input.tensor()?, tag)? {
        Flat::Float(v) => v,
        Flat::Int(_) => unreachable!("the integral dtypes were refused above"),
    };
    let storage = PyDtype::new(tag).storage(OP)?;
    let double_acc = storage == candle_core::DType::F64;
    let out = softmax_body(&source, outer, n, inner, double_acc, false);

    let device = input.tensor()?.device().clone();
    let tensor = write_flat(OP, Flat::Float(out), dims, &device, tag)?;
    finish(py, tensor, tag)
}

/// The reduction shared by `_softmax.default` and `_safe_softmax.default`:
/// max-subtract, exponentiate, normalise, over the `(outer, n, inner)` view of
/// a dim-`dim` softmax (`outer`/`inner` are the product of the extents on
/// either side of `dim`; `n` is `dim`'s own extent).
///
/// `safe` is the only difference between the two ops, and it is applied where
/// the divergence is measured to live: a row whose max is `-inf` (every
/// element `-inf`, since `-inf` is softmax's own floor) computes `0` for
/// every element instead of running the usual exponential, which on that row
/// is a `NaN` produced by `-inf - (-inf)`. `safe=false` skips the check
/// entirely, so `_softmax.default`'s behaviour (a `NaN` row, matching
/// upstream, per this file's docs above) is untouched by this refactor.
fn softmax_body(source: &[f64], outer: usize, n: usize, inner: usize, double_acc: bool, safe: bool) -> Vec<f64> {
    let mut out = vec![0.0f64; source.len()];

    for o in 0..outer {
        for i in 0..inner {
            let at = |j: usize| o * n * inner + j * inner + i;
            if double_acc {
                let mut max = f64::NEG_INFINITY;
                for j in 0..n {
                    let v = source[at(j)];
                    if !(v <= max) {
                        max = v;
                    }
                }
                if safe && max == f64::NEG_INFINITY {
                    for j in 0..n {
                        out[at(j)] = 0.0;
                    }
                    continue;
                }
                let mut sum = 0.0f64;
                for j in 0..n {
                    let e = (source[at(j)] - max).exp();
                    out[at(j)] = e;
                    sum += e;
                }
                for j in 0..n {
                    out[at(j)] /= sum;
                }
            } else {
                let mut max = f32::NEG_INFINITY;
                for j in 0..n {
                    let v = source[at(j)] as f32;
                    if !(v <= max) {
                        max = v;
                    }
                }
                if safe && max == f32::NEG_INFINITY {
                    for j in 0..n {
                        out[at(j)] = 0.0;
                    }
                    continue;
                }
                let mut sum = 0.0f32;
                for j in 0..n {
                    let e = ((source[at(j)] as f32) - max).exp();
                    out[at(j)] = e as f64;
                    sum += e;
                }
                for j in 0..n {
                    out[at(j)] = ((out[at(j)] as f32) / sum) as f64;
                }
            }
        }
    }
    out
}

/// `aten::_safe_softmax(Tensor self, int dim, ScalarType? dtype=None) -> Tensor`
///
/// torch's own decomposition is the spec this reproduces
/// (`torch/_decomp/decompositions.py::safe_softmax`, `register_decomposition(aten._safe_softmax)`):
///
/// ```text
/// out = torch.softmax(self, dim=dim, dtype=dtype)
/// masked_rows = torch.all(self.eq(-inf), dim=dim, keepdim=True)
/// return torch.where(masked_rows, zeros, out)
/// ```
///
/// So the one place this disagrees with `_softmax.default` is a row that is
/// *entirely* `-inf`: measured on torch 2.13.0, plain `_softmax` answers `nan`
/// there (see that op's docs above) and `_safe_softmax` answers `0` for every
/// element of the row instead. That is exactly the shape of a fully-masked
/// attention row -- every key excluded by the causal mask plus padding -- and
/// `nan` there would poison every downstream matmul rather than staying
/// contained the way a `0` attention weight does.
///
/// `dtype`, when given, casts `self` **before** the integral/boolean refusal
/// below runs, not after -- measured: `_safe_softmax(int64_tensor, 0,
/// torch.float32)` succeeds. Composite `torch.softmax(self, dim, dtype)`
/// upcasts first and then calls the kernel in that dtype, and the refusal is
/// the kernel's, so a shim that checked the *original* dtype would reject a
/// call upstream accepts.
fn safe_softmax_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten._safe_softmax.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let dim_raw = dim_arg(args, kwargs, 1, "dim")?.ok_or_else(|| missing(OP, "dim"))?;
    let dtype = dtype_arg(args, kwargs, 2, "dtype")?;

    let (tensor, tag) = match dtype {
        Some(want) => {
            let storage = PyDtype::new(want).storage(OP)?;
            let cast = input
                .tensor()?
                .to_dtype(storage)
                .map_err(|e| candle_err(OP, e))?;
            (cast, want)
        }
        None => (input.tensor()?.clone(), input.tag()),
    };
    if !tag.is_floating_point() {
        return Err(not_implemented(format!(
            "\"softmax_lastdim_kernel_impl\" not implemented for '{}'",
            scalar_type_name(tag)
        )));
    }
    let rank = tensor.rank();
    let dim = normalise_dim(OP, dim_raw, rank)?;

    let dims = tensor.dims().to_vec();
    let (outer, n, inner) = if dims.is_empty() {
        (1usize, 1usize, 1usize)
    } else {
        (
            dims[..dim].iter().product::<usize>(),
            dims[dim],
            dims[dim + 1..].iter().product::<usize>(),
        )
    };

    let source = match read_flat(OP, &tensor, tag)? {
        Flat::Float(v) => v,
        Flat::Int(_) => unreachable!("the integral dtypes were refused above"),
    };
    let storage = PyDtype::new(tag).storage(OP)?;
    let double_acc = storage == candle_core::DType::F64;
    let out = softmax_body(&source, outer, n, inner, double_acc, true);

    let device = tensor.device().clone();
    let result = write_flat(OP, Flat::Float(out), dims, &device, tag)?;
    finish(py, result, tag)
}

/// Row-major strides for a contiguous shape, as element counts.
fn contiguous_strides(dims: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * dims[i + 1];
    }
    strides
}

/// `aten::scatter.src(Tensor self, int dim, Tensor index, Tensor src) -> Tensor`
///
/// Written out by hand rather than routed to `candle`'s `scatter`, because the
/// two do not agree on what a valid call is: candle requires
/// `index.dims() == src.dims()` and `self.dims() == src.dims()` off the scatter
/// axis, while torch requires only that `index` be **no larger than** either.
/// The generation loop uses exactly the shape candle rejects -- `scatter` of a
/// `(batch, k)` index into a `(batch, vocab)` row of logits -- so borrowing
/// candle's checks would refuse the call that matters.
///
/// Duplicate indices resolve to the *last* write in iteration order, matching
/// upstream (measured: scattering `[1,2,3]` at index `[0,0,0]` leaves 3).
fn scatter_src(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.scatter.src";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let dim_raw = dim_arg(args, kwargs, 1, "dim")?.ok_or_else(|| missing(OP, "dim"))?;
    let index = tensor_arg(OP, args, kwargs, 2, "index")?;
    let src = tensor_arg(OP, args, kwargs, 3, "src")?;

    let tag = input.tag();
    if src.tag() != tag {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "scatter(): Expected self.dtype to be equal to src.dtype",
        ));
    }
    // torch accepts both int32 and int64 here (measured -- an `int32` index is
    // *not* refused, unlike a `uint8` mask in `masked_fill`).
    if !matches!(index.tag(), TorchDType::Int64 | TorchDType::Int32) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "scatter(): Expected dtype int32 or int64 for index, got {}",
            index.tag().name()
        )));
    }

    let rank = input.tensor()?.rank();
    if rank == 0 {
        return Err(not_implemented(
            "aten.scatter.src: 0-dim self not implemented in torch._C shim",
        ));
    }
    let dim = normalise_dim(OP, dim_raw, rank)?;
    let self_dims = input.tensor()?.dims().to_vec();
    let idx_dims = index.tensor()?.dims().to_vec();
    let src_dims = src.tensor()?.dims().to_vec();
    if idx_dims.len() != rank {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Index tensor must have the same number of dimensions as self tensor",
        ));
    }
    if src_dims.len() != rank {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Index tensor must have the same number of dimensions as src tensor",
        ));
    }
    for d in 0..rank {
        if idx_dims[d] > src_dims[d] || (d != dim && idx_dims[d] > self_dims[d]) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Expected index {idx_dims:?} to be no larger than self {self_dims:?} apart \
                 from dimension {dim} and to be no larger size than src {src_dims:?}"
            )));
        }
    }

    let mut out = read_flat(OP, input.tensor()?, tag)?;
    let source = read_flat(OP, src.tensor()?, tag)?;
    let positions = match read_flat(OP, index.tensor()?, index.tag())? {
        Flat::Int(v) => v,
        Flat::Float(_) => unreachable!("the index dtype was checked above"),
    };

    let self_strides = contiguous_strides(&self_dims);
    let idx_strides = contiguous_strides(&idx_dims);
    let src_strides = contiguous_strides(&src_dims);
    let count: usize = idx_dims.iter().product();

    let mut coord = vec![0usize; rank];
    for _ in 0..count {
        let idx_off: usize = coord.iter().zip(&idx_strides).map(|(c, s)| c * s).sum();
        let target = positions[idx_off];
        if target < 0 || target as usize >= self_dims[dim] {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "index {target} is out of bounds for dimension {dim} with size {}",
                self_dims[dim]
            )));
        }
        let src_off: usize = coord.iter().zip(&src_strides).map(|(c, s)| c * s).sum();
        let self_off: usize = coord
            .iter()
            .enumerate()
            .map(|(d, c)| if d == dim { target as usize } else { *c } * self_strides[d])
            .sum();
        match (&source, &mut out) {
            (Flat::Float(s), Flat::Float(o)) => o[self_off] = s[src_off],
            (Flat::Int(s), Flat::Int(o)) => o[self_off] = s[src_off],
            _ => unreachable!("self and src share a dtype, checked above"),
        }
        for d in (0..rank).rev() {
            coord[d] += 1;
            if coord[d] < idx_dims[d] {
                break;
            }
            coord[d] = 0;
        }
    }

    let device = input.tensor()?.device().clone();
    let tensor = write_flat(OP, out, self_dims, &device, tag)?;
    finish(py, tensor, tag)
}

/// `aten::gather(Tensor self, int dim, Tensor index, *, bool sparse_grad=False) -> Tensor`
///
/// `scatter.src` read backwards, and written the same way for the same reason:
/// candle's `Tensor::gather` exists but does not agree with torch about what a
/// valid call is, so borrowing it would move the disagreement out of sight
/// rather than remove it. Three measured differences, each of which would be a
/// silent divergence:
///
/// 1. **Index dtype.** candle accepts `u8`/`u32`/`i64`; torch accepts exactly
///    `int32` and `int64` and refuses the rest by name. A `uint8` index is a
///    *mask* everywhere else in this shim, and candle would silently read it as
///    positions.
/// 2. **Out of range.** candle's `i64` path reaches `as_usize()` on a negative
///    index, so `-1` becomes a huge `usize` and the error names that number
///    instead of `-1`. torch says `index -1 is out of bounds for dimension 1
///    with size 3`, and that text is what a caller debugs against. There is no
///    negative-index convention here: unlike `select`/`slice`, `gather` does
///    **not** wrap (measured).
/// 3. **Contiguity.** candle refuses a non-contiguous `self` or `index`
///    outright (`RequiresContiguous`); torch gathers from a transposed tensor
///    without comment (measured on `arange(12).reshape(3,4).t()`), and BERT's
///    path arrives that way.
///
/// The rank rule is upstream's `ensure_nonempty_dim`, i.e. `max(rank, 1)` on
/// both sides, which is why a 0-d `self` accepts a 1-d index (`gather(tensor(7.),
/// 0, tensor([0,0]))` -> `[7., 7.]`) and a 1-d `self` accepts a 0-d index
/// (-> 0-d), but a 0-d `self` with a 2-d index is refused. Guessing "ranks must
/// be equal" would have refused two calls torch answers.
///
/// The output has the **index's** shape, not `self`'s: off-axis the index may
/// be *smaller* than `self` (and the extra rows are simply not read), and along
/// `dim` it may be *longer* (values repeat). Both measured.
///
/// `sparse_grad` is accepted and ignored -- it selects an autograd
/// representation, and there is no autograd here. Upstream's forward answer is
/// the same for either value (measured).
fn gather_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.gather.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let dim_raw = dim_arg(args, kwargs, 1, "dim")?.ok_or_else(|| missing(OP, "dim"))?;
    let index = tensor_arg(OP, args, kwargs, 2, "index")?;
    let _sparse_grad = bool_arg(args, kwargs, 3, "sparse_grad")?;

    if !matches!(index.tag(), TorchDType::Int64 | TorchDType::Int32) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "gather(): Expected dtype int32/int64 for index",
        ));
    }

    let tag = input.tag();
    // `ensure_nonempty_dim`/`ensure_nonempty_size`: a 0-d tensor counts as one
    // dimension of extent 1 for the purposes of both the rank check and the
    // bounds check. The *output* still gets the index's real shape.
    let self_dims: Vec<usize> = if input.tensor()?.rank() == 0 {
        vec![1]
    } else {
        input.tensor()?.dims().to_vec()
    };
    let idx_shape = index.tensor()?.dims().to_vec();
    let idx_dims: Vec<usize> = if idx_shape.is_empty() {
        vec![1]
    } else {
        idx_shape.clone()
    };
    if idx_dims.len() != self_dims.len() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "Index tensor must have the same number of dimensions as input tensor",
        ));
    }
    let dim = normalise_dim(OP, dim_raw, input.tensor()?.rank())?;
    for d in 0..self_dims.len() {
        if d != dim && idx_dims[d] > self_dims[d] {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Size does not match at dimension {d} expected index {idx_dims:?} \
                 to be no larger than self {self_dims:?} apart from dimension {dim}"
            )));
        }
    }

    let source = read_flat(OP, input.tensor()?, tag)?;
    let positions = match read_flat(OP, index.tensor()?, index.tag())? {
        Flat::Int(v) => v,
        Flat::Float(_) => unreachable!("the index dtype was checked above"),
    };

    let self_strides = contiguous_strides(&self_dims);
    let idx_strides = contiguous_strides(&idx_dims);
    let count: usize = idx_dims.iter().product();
    let rank = self_dims.len();

    let mut out = match &source {
        Flat::Float(_) => Flat::Float(vec![0.0f64; count]),
        Flat::Int(_) => Flat::Int(vec![0i64; count]),
    };
    let mut coord = vec![0usize; rank];
    for _ in 0..count {
        let idx_off: usize = coord.iter().zip(&idx_strides).map(|(c, s)| c * s).sum();
        let target = positions[idx_off];
        if target < 0 || target as usize >= self_dims[dim] {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "index {target} is out of bounds for dimension {dim} with size {}",
                self_dims[dim]
            )));
        }
        let self_off: usize = coord
            .iter()
            .enumerate()
            .map(|(d, c)| if d == dim { target as usize } else { *c } * self_strides[d])
            .sum();
        match (&source, &mut out) {
            (Flat::Float(s), Flat::Float(o)) => o[idx_off] = s[self_off],
            (Flat::Int(s), Flat::Int(o)) => o[idx_off] = s[self_off],
            _ => unreachable!("out was built from source's own variant"),
        }
        for d in (0..rank).rev() {
            coord[d] += 1;
            if coord[d] < idx_dims[d] {
                break;
            }
            coord[d] = 0;
        }
    }

    let device = input.tensor()?.device().clone();
    let tensor = write_flat(OP, out, idx_shape, &device, tag)?;
    finish(py, tensor, tag)
}

/// Round a vector through the storage dtype and back, in one candle pass.
///
/// `multinomial`'s fast path needs this twice, because upstream writes both
/// intermediates into tensors of the input's dtype: `q = empty_like(self)`
/// narrows the exponential draws, and `at::div_out(q, self, q)` narrows the
/// ratio. Doing the arithmetic in `f64` throughout and narrowing once at the
/// end would give a *different argmax* wherever two categories are within an
/// ulp of each other, which is precisely where a sampler's choice is decided.
fn narrow_through(
    op: &str,
    values: Vec<f64>,
    storage: candle_core::DType,
    device: &Device,
) -> PyResult<Vec<f64>> {
    if storage == candle_core::DType::F64 {
        return Ok(values);
    }
    let n = values.len();
    if n == 0 {
        return Ok(values);
    }
    Tensor::from_vec(values, n, device)
        .and_then(|t| t.to_dtype(storage))
        .and_then(|t| t.to_dtype(candle_core::DType::F64))
        .and_then(|t| t.to_vec1::<f64>())
        .map_err(|e| candle_err(op, e))
}

/// `aten::multinomial(Tensor self, SymInt num_samples, bool replacement=False,
///                    *, Generator? generator=None) -> Tensor`
///
/// The only op here that draws, and therefore the only one whose *answer*
/// depends on consuming torch's stream in torch's order. Two facts decide it,
/// and both were measured rather than assumed (docs/SAMPLING.md §2):
///
/// **1. There are two algorithms, and the branch is not the one the argument
/// name suggests.** `multinomial_out` takes a Gumbel-style fast path when
/// `!replacement` **or `num_samples == 1`** -- so the call
/// `GenerationMixin._sample` makes, `multinomial(probs, num_samples=1)`, takes
/// it even though `replacement` defaults to False and the docs describe the
/// cumulative-sum kernel. The fast path is
///
///     q = empty_like(self).exponential_(1);  q = self / q;  argmax(q, -1, keepdim)
///
/// (`topk` instead of `argmax` when more than one sample is wanted), which
/// consumes `2 * numel` 32-bit words regardless of `num_samples`. The
/// with-replacement kernel consumes `2 * n_dist * num_samples`. Counting words
/// on real torch is what distinguished them: `multinomial(ones(100), 1)`
/// advances the generator by 200 words, `multinomial(ones(100), 3,
/// replacement=True)` by 6.
///
/// **2. Every intermediate is narrowed to the input dtype.** See
/// `narrow_through`.
///
/// The with-replacement kernel's cumulative sum accumulates *in the input
/// dtype* (`scalar_t sum = 0; sum += val;`), sets the last bucket to exactly 1
/// before searching, and compares `cum_prob < uniform_sample` with the
/// left-biased binary search transcribed below. Both paths were checked against
/// upstream on a `(3, 11)` distribution over six seeds: 12/12 identical index
/// lists, no tolerance involved.
fn multinomial_default(
    py: Python<'_>,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "aten.multinomial.default";
    let input = tensor_arg(OP, args, kwargs, 0, "self")?;
    let num_samples =
        int_arg(args, kwargs, 1, "num_samples")?.ok_or_else(|| missing(OP, "num_samples"))?;
    let replacement = bool_arg(args, kwargs, 2, "replacement")?.unwrap_or(false);
    generator_arg(OP, args, kwargs, 3, "generator")?;

    let tag = input.tag();
    if !tag.is_floating_point() {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "multinomial only supports floating-point dtypes for input, got: {}",
            scalar_type_name(tag)
        )));
    }
    let rank = input.tensor()?.rank();
    if rank != 1 && rank != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "prob_dist must be 1 or 2 dim",
        ));
    }
    let dims = input.tensor()?.dims().to_vec();
    let n_categories = dims[rank - 1];
    let n_dist = if rank == 2 { dims[0] } else { 1 };
    if num_samples <= 0 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "cannot sample n_sample <= 0 samples",
        ));
    }
    let n_sample = num_samples as usize;
    if !replacement && n_sample > n_categories {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(
            "cannot sample n_sample > prob_dist.size(-1) samples without replacement",
        ));
    }

    let storage = PyDtype::new(tag).storage(OP)?;
    let device = input.tensor()?.device().clone();
    let probs = match read_flat(OP, input.tensor()?, tag)? {
        Flat::Float(v) => v,
        Flat::Int(_) => unreachable!("the integral dtypes were refused above"),
    };
    let out_dims: Vec<usize> = if rank == 2 {
        vec![n_dist, n_sample]
    } else {
        vec![n_sample]
    };

    let picks: Vec<i64> = if !replacement || n_sample == 1 {
        // Upstream's sanity checks, in upstream's order. `(max < INFINITY) &
        // (min >= 0)` is a single expression there; a NaN anywhere fails it
        // because every comparison against NaN is false.
        if probs.iter().any(|v| !v.is_finite() || *v < 0.0) {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "probability tensor contains either `inf`, `nan` or element < 0",
            ));
        }
        for row in probs.chunks(n_categories.max(1)) {
            if row.iter().sum::<f64>() == 0.0 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "invalid multinomial distribution (sum of probabilities <= 0)",
                ));
            }
        }

        let q = {
            let mut gen = crate::rng::default_generator();
            crate::rng::exponential_serial(&mut gen, probs.len(), 1.0)
        };
        let q = narrow_through(OP, q, storage, &device)?;
        // `at::div_out(q, self, q)` runs in `opmath_type<scalar_t>` -- float
        // for everything but float64 -- and stores back into `scalar_t`.
        let ratio: Vec<f64> = if storage == candle_core::DType::F64 {
            probs.iter().zip(&q).map(|(p, d)| p / d).collect()
        } else {
            probs
                .iter()
                .zip(&q)
                .map(|(p, d)| ((*p as f32) / (*d as f32)) as f64)
                .collect()
        };
        let ratio = narrow_through(OP, ratio, storage, &device)?;

        let mut picks = Vec::with_capacity(n_dist * n_sample);
        for i in 0..n_dist {
            let row = &ratio[i * n_categories..(i + 1) * n_categories];
            if n_sample == 1 {
                // `at::argmax_out` -- the *first* maximum wins.
                let mut best = 0usize;
                for j in 1..n_categories {
                    if cmp_torch_f64(row[j], row[best]) == std::cmp::Ordering::Greater {
                        best = j;
                    }
                }
                picks.push(best as i64);
            } else {
                // `at::topk_out(vals, result, q, n_sample)` -- largest first,
                // sorted, and stable among ties for the reason `order_along`
                // records.
                let mut order: Vec<usize> = (0..n_categories).collect();
                order.sort_by(|&a, &b| cmp_torch_f64(row[b], row[a]));
                picks.extend(order.into_iter().take(n_sample).map(|j| j as i64));
            }
        }
        picks
    } else {
        // `multinomial_with_replacement_apply`.
        //
        // **The cumulative distribution is *not* kept in the input's dtype**,
        // and that is the one thing here that had to be measured rather than
        // read off the shape of the rest of this file. A `bfloat16` input runs
        // the accumulation and the normalising division in `float`, with no
        // narrowing anywhere; carrying `scalar_t` through instead -- which is
        // what the surrounding code makes the natural guess -- put this shim on
        // a different bucket from upstream on 2 of 140 measured bf16 draws.
        //
        // The measurement is in docs/SAMPLING.md §5: 20,000 draws from a known
        // MT stream bracket every one of the eleven bucket boundaries, and the
        // brackets are ~2e-4 wide, which is a fifth of `bfloat16`'s spacing
        // there. A `bfloat16` cumulative distribution lands outside six of the
        // eleven; a `float` one lands inside all eleven, for `float16` too.
        //
        // So the accumulate type is `at::acc_type<scalar_t, false>` -- float
        // for float16/bfloat16/float32, double for float64 -- exactly as the
        // *other* kernels here use `opmath_type`, and the fast path above is
        // the odd one out precisely because it materialises real tensors of
        // `self`'s dtype.
        let double_acc = storage == candle_core::DType::F64;
        let add_acc = |a: f64, b: f64| -> f64 {
            if double_acc {
                a + b
            } else {
                ((a as f32) + (b as f32)) as f64
            }
        };
        let div_acc = |a: f64, b: f64| -> f64 {
            if double_acc {
                a / b
            } else {
                ((a as f32) / (b as f32)) as f64
            }
        };
        let mut picks = Vec::with_capacity(n_dist * n_sample);
        let mut gen = crate::rng::default_generator();
        for i in 0..n_dist {
            let row = &probs[i * n_categories..(i + 1) * n_categories];
            let mut cum = vec![0.0f64; n_categories];
            let mut sum = 0.0f64;
            for (j, &val) in row.iter().enumerate() {
                if val < 0.0 {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "invalid multinomial distribution (encountering probability entry < 0)",
                    ));
                }
                if !val.is_finite() {
                    return Err(pyo3::exceptions::PyRuntimeError::new_err(
                        "invalid multinomial distribution (encountering probability entry = infinity or NaN)",
                    ));
                }
                sum = add_acc(sum, val);
                cum[j] = sum;
            }
            if !(sum > 0.0) {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(
                    "invalid multinomial distribution (sum of probabilities <= 0)",
                ));
            }
            for slot in cum.iter_mut() {
                *slot = div_acc(*slot, sum);
            }
            // "Make sure the last cumulative distribution bucket sums to 1",
            // upstream's comment and upstream's assignment -- it is written
            // inside the sample loop there, which is the same thing done once.
            cum[n_categories - 1] = 1.0;
            for _ in 0..n_sample {
                let sample = crate::rng::uniform_sample_f64(&mut gen, 0.0, 1.0);
                let (mut left, mut right) = (0usize, n_categories);
                while right - left > 0 {
                    let mid = left + (right - left) / 2;
                    if cum[mid] < sample {
                        left = mid + 1;
                    } else {
                        right = mid;
                    }
                }
                picks.push(left as i64);
            }
        }
        drop(gen);
        picks
    };

    let out = Tensor::from_vec(picks, out_dims, &device).map_err(|e| candle_err(OP, e))?;
    finish(py, out, TorchDType::Int64)
}

// ---------------------------------------------------------------------------
// Argument plumbing
//
// aten schemas allow positional or keyword for everything before the `*`, and
// the vendored Python layer will use both spellings, so each argument is looked
// up by index and by name.
// ---------------------------------------------------------------------------

/// An aten `Scalar`, kept in the category it arrived as.
///
/// The category is not cosmetic: torch's dtype inference asks "was this an
/// integer?" in several of the ops above (`arange` picks int64 or the default
/// float from it, `pow` decides whether to float the result), and collapsing
/// everything to `f64` on the way in would throw away the only information
/// those rules read. Python `bool` lands in `Int` -- it subclasses `int` and
/// torch treats it as an integral scalar in these positions.
#[derive(Clone, Copy)]
enum Scalar {
    Int(i64),
    Float(f64),
}

impl Scalar {
    fn is_int(self) -> bool {
        matches!(self, Scalar::Int(_))
    }

    fn as_f64(self) -> f64 {
        match self {
            Scalar::Int(v) => v as f64,
            Scalar::Float(v) => v,
        }
    }

    fn as_i64(self) -> i64 {
        match self {
            Scalar::Int(v) => v,
            Scalar::Float(v) => v as i64,
        }
    }
}

fn missing(op: &str, name: &str) -> PyErr {
    pyo3::exceptions::PyTypeError::new_err(format!("{op}: missing required argument '{name}'"))
}

fn scalar_arg(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Option<Scalar>> {
    let value = match optional(args, kwargs, index, name)? {
        Some(value) if !value.is_none() => value,
        _ => return Ok(None),
    };
    // Order matters: `bool` is a subclass of `int` in Python.
    if value.is_instance_of::<pyo3::types::PyBool>() {
        return Ok(Some(Scalar::Int(i64::from(value.extract::<bool>()?))));
    }
    if value.is_instance_of::<pyo3::types::PyInt>() {
        return Ok(Some(Scalar::Int(value.extract()?)));
    }
    if value.is_instance_of::<pyo3::types::PyFloat>() {
        return Ok(Some(Scalar::Float(value.extract()?)));
    }
    // torch accepts a zero-dim tensor anywhere a `Scalar` is taken, and the
    // overload resolver in bootstrap.py binds one here for the same reason, so
    // refusing it at the kernel would make the two disagree.
    if let Ok(tensor) = value.extract::<PyTensorBase>() {
        if tensor.tensor()?.rank() != 0 {
            return Err(pyo3::exceptions::PyTypeError::new_err(format!(
                "{op}: argument '{name}' as a tensor must be zero-dim, got {}D",
                tensor.tensor()?.rank()
            )));
        }
        let as_f64 = tensor
            .tensor()?
            .to_dtype(candle_core::DType::F64)
            .and_then(|t| t.to_scalar::<f64>())
            .map_err(|err| candle_err(op, err))?;
        return Ok(Some(if tensor.tag().is_floating_point() {
            Scalar::Float(as_f64)
        } else {
            Scalar::Int(as_f64 as i64)
        }));
    }
    Err(pyo3::exceptions::PyTypeError::new_err(format!(
        "{op}: argument '{name}' must be a number, got {}",
        value.get_type().name().map(|n| n.to_string()).unwrap_or_default()
    )))
}

fn dtype_arg(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Option<TorchDType>> {
    match optional(args, kwargs, index, name)? {
        Some(value) if !value.is_none() => Ok(Some(value.extract::<PyDtype>()?.tag())),
        _ => Ok(None),
    }
}

fn int_arg(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Option<i64>> {
    match optional(args, kwargs, index, name)? {
        Some(value) if !value.is_none() => Ok(Some(value.extract()?)),
        _ => Ok(None),
    }
}

fn dim_arg(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Option<isize>> {
    match optional(args, kwargs, index, name)? {
        Some(value) if !value.is_none() => Ok(Some(value.extract()?)),
        _ => Ok(None),
    }
}

fn bool_arg(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Option<bool>> {
    match optional(args, kwargs, index, name)? {
        Some(value) if !value.is_none() => Ok(Some(value.extract()?)),
        _ => Ok(None),
    }
}

/// torch's negative-dimension convention, with torch's error message shape.
fn normalise_dim(op: &str, dim: isize, rank: usize) -> PyResult<usize> {
    // torch treats a zero-dim tensor as one-dimensional for indexing purposes.
    let extent = rank.max(1) as isize;
    let index = if dim < 0 { dim + extent } else { dim };
    if index < 0 || index >= extent {
        return Err(pyo3::exceptions::PyIndexError::new_err(format!(
            "{op}: Dimension out of range (expected to be in range of [{}, {}], but got {dim})",
            -extent,
            extent - 1
        )));
    }
    Ok(index as usize)
}

/// Wraps a finished tensor, routing the `torch.bool` tag through the one
/// constructor that is allowed to attach it (BOOL.md §6.3 item 1).
fn finish(py: Python<'_>, tensor: Tensor, tag: TorchDType) -> PyResult<Py<PyAny>> {
    let wrapped = if tag == TorchDType::Bool {
        PyTensorBase::boolean(tensor)?
    } else {
        PyTensorBase::new(tensor)?
    };
    Ok(wrapped.into_pyobject(py)?.into_any().unbind())
}

fn optional<'py>(
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    if index < args.len() {
        return Ok(Some(args.get_item(index)?));
    }
    match kwargs {
        Some(kwargs) => kwargs.get_item(name),
        None => Ok(None),
    }
}

fn required<'py>(
    op: &str,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Bound<'py, PyAny>> {
    optional(args, kwargs, index, name)?.ok_or_else(|| {
        pyo3::exceptions::PyTypeError::new_err(format!("{op}: missing required argument '{name}'"))
    })
}

fn tensor_arg(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<PyTensorBase> {
    let value = required(op, args, kwargs, index, name)?;
    // The wrapper, not the bare candle tensor: the torch dtype tag lives on
    // the wrapper and `torch.bool` is invisible in the candle dtype.
    value.extract::<PyTensorBase>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{op}: argument '{name}' must be a torch._C.TensorBase, got {}",
            value.get_type().name().map(|n| n.to_string()).unwrap_or_default()
        ))
    })
}

/// `tensor_arg` for a `Tensor?` slot. `None` and an absent argument are the
/// same answer -- `native_layer_norm(x, [4], None, None, eps)` is how a
/// `nn.LayerNorm(elementwise_affine=False)` arrives.
fn optional_tensor_arg(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Option<PyTensorBase>> {
    match optional(args, kwargs, index, name)? {
        Some(value) if !value.is_none() => Ok(Some(value.extract::<PyTensorBase>().map_err(|_| {
            pyo3::exceptions::PyTypeError::new_err(format!(
                "{op}: argument '{name}' must be a torch._C.TensorBase or None, got {}",
                value.get_type().name().map(|n| n.to_string()).unwrap_or_default()
            ))
        })?)),
        _ => Ok(None),
    }
}

/// A factory op's `device=` slot, as a **label**.
///
/// The label rather than a resolved handle, because `meta` has no handle to
/// resolve to: a factory has to see `meta` and build a storage-less tensor
/// instead of allocating one. `device_arg` below is this plus `resolve()`, for
/// the kernels that have no meta path and should fail loudly if handed one.
///
/// Absent means `fallback`. For most factories that is the CPU -- the
/// process-wide default device lives above this layer, in the torch-function
/// mode stack that `bootstrap.py` consults before a call ever reaches the
/// dispatcher (docs/META.md §8), which is where upstream puts it too.
fn device_arg_or_label(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
    fallback: &PyDevice,
) -> PyResult<PyDevice> {
    match optional(args, kwargs, index, name)? {
        // torch accepts a plain string, another `device`, or an integer
        // wherever a device is taken; `coerce` is the one place that knows
        // which of those are legal and what each means.
        Some(value) if !value.is_none() => PyDevice::coerce(&value),
        _ => Ok(fallback.clone()),
    }
}

/// Arguments the shim has no answer for. Ignoring them silently would make the
/// call look supported; a `layout=torch.sparse_coo` that is dropped on the floor
/// produces a wrong answer with no trace.
fn reject_unsupported(
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    fields: &[(usize, &str)],
) -> PyResult<()> {
    for (index, name) in fields {
        if let Some(value) = optional(args, kwargs, *index, name)? {
            if !value.is_none() {
                return Err(not_implemented(format!(
                    "{op}: argument '{name}' not implemented in torch._C shim (got {value})"
                )));
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Scalar -> dtype conversion, as torch does it
//
// `c10::checked_convert` refuses a scalar the destination dtype cannot hold.
// candle has no such check: an out-of-range integer wraps (two's complement)
// and an out-of-range float saturates to infinity, both without a word. The
// golden harness caught exactly that pair -- `full([3], 1e6, float16)` gave
// `inf` where torch raises, and `full([3], 2**31, int32)` gave `-2**31` where
// torch raises -- so the check is transcribed here rather than approximated.
//
// The rules below are measured against torch 2.13.0, not read off the C++.
// Two of them are counter-intuitive enough that guessing would have got them
// wrong, and both are load-bearing for cases the harness already passes:
//
//   * A negative value converted to an *unsigned* dtype is allowed to wrap,
//     as long as its magnitude fits. `full(-1, uint8) == 255` is legal in
//     torch; `full(-300, uint8)` is not. (c10 spells this "allow for negative
//     numbers to wrap using two's complement arithmetic".)
//   * Half / BFloat16 / Float8 skip the check entirely when the tensor has
//     exactly one element, and saturate silently instead. That is an upstream
//     inconsistency -- `fill_` takes a CPU numel==1 fast path whose conversion
//     is unchecked -- but `full([], 1e6, float16) == inf` while
//     `full([3], 1e6, float16)` raises, on real torch, so a shim that always
//     refuses would diverge from torch in the other direction.
// ---------------------------------------------------------------------------

/// The name torch puts in the message. These are C++ type spellings, so they
/// are not derivable from the torch dtype name -- each one was read off a real
/// `RuntimeError` from torch 2.13.0.
fn c10_name(dtype: TorchDType) -> &'static str {
    use TorchDType::*;
    match dtype {
        Float64 => "double",
        Float32 => "float",
        Float16 => "c10::Half",
        BFloat16 => "c10::BFloat16",
        Int64 => "int64_t",
        Int32 => "int",
        Int16 => "int16_t",
        Int8 => "int8_t",
        UInt8 => "uint8_t",
        UInt16 => "uint16_t",
        UInt32 => "uint32_t",
        UInt64 => "uint64_t",
        Bool => "bool",
        Float8E4M3FN => "c10::Float8_e4m3fn",
        Float8E5M2 => "c10::Float8_e5m2",
        other => other.name(),
    }
}

/// `c10::toString(ScalarType)` -- the spelling torch puts in a "not
/// implemented for '<X>'" message. A *third* naming of the same set, distinct
/// from both `TorchDType::name()` (`uint32`) and `c10_name` (`uint32_t`), and
/// like those it is not derivable: only the entries this shim can actually
/// reach are listed, each read off a real torch error.
fn scalar_type_name(dtype: TorchDType) -> &'static str {
    use TorchDType::*;
    match dtype {
        Float32 => "Float",
        Float64 => "Double",
        Float16 => "Half",
        BFloat16 => "BFloat16",
        UInt8 => "Byte",
        UInt16 => "UInt16",
        UInt32 => "UInt32",
        UInt64 => "UInt64",
        Int8 => "Char",
        Int16 => "Short",
        Int32 => "Int",
        Int64 => "Long",
        Bool => "Bool",
        Float8E4M3FN => "Float8_e4m3fn",
        other => other.name(),
    }
}

/// Does torch have an `arange_cpu` kernel for this dtype? Measured against
/// torch 2.13.0 over every dtype this shim can store.
fn arange_has_cpu_kernel(dtype: TorchDType) -> bool {
    use TorchDType::*;
    !matches!(
        dtype,
        UInt16
            | UInt32
            | UInt64
            | Bool
            | Float8E4M3FN
            | Float8E4M3FNUZ
            | Float8E5M2
            | Float8E5M2FNUZ
            | Float8E8M0FNU
            | Float4E2M1FNX2
    )
}

fn overflow(dtype: TorchDType) -> PyErr {
    // torch's own wording, with no shim prefix: this is torch semantics being
    // reproduced, not a shim limitation, and a caller matching on the message
    // should not have to know which of the two produced it.
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "value cannot be converted to type {} without overflow",
        c10_name(dtype)
    ))
}

/// Largest finite magnitude of a floating dtype; `None` for the others, which
/// are range-checked exactly rather than through `f64`.
fn float_max(dtype: TorchDType) -> Option<f64> {
    use TorchDType::*;
    Some(match dtype {
        Float64 => f64::MAX,
        Float32 => f32::MAX as f64,
        Float16 => 65504.0,
        BFloat16 => 3.3895313892515355e38,
        Float8E4M3FN | Float8E4M3FNUZ => 448.0,
        Float8E5M2 | Float8E5M2FNUZ => 57344.0,
        _ => return None,
    })
}

/// Inclusive integer range, as `(min, max)`. `min` is negative for the signed
/// types and zero for the unsigned ones -- the wrap allowance is applied by
/// the caller, not folded in here, so the two rules stay separable.
fn int_range(dtype: TorchDType) -> Option<(i64, i64)> {
    use TorchDType::*;
    Some(match dtype {
        Int64 => (i64::MIN, i64::MAX),
        Int32 => (i32::MIN as i64, i32::MAX as i64),
        Int16 => (i16::MIN as i64, i16::MAX as i64),
        Int8 => (i8::MIN as i64, i8::MAX as i64),
        UInt8 => (0, u8::MAX as i64),
        UInt16 => (0, u16::MAX as i64),
        UInt32 => (0, u32::MAX as i64),
        UInt64 => (0, i64::MAX),
        _ => return None,
    })
}

fn checked_convert(
    fill: &Bound<'_, PyAny>,
    fill_is_int: bool,
    dtype: TorchDType,
    numel: usize,
) -> PyResult<()> {
    // Every value is a valid `bool`; torch converts by truthiness and never
    // reports an overflow for it.
    if dtype == TorchDType::Bool {
        return Ok(());
    }

    // The upstream numel==1 hole, reproduced deliberately. Restricted to the
    // reduced-precision float types because torch's own fast path checks
    // float/double/int even at one element -- measured, not assumed.
    let unchecked_at_one = matches!(
        dtype,
        TorchDType::Float16
            | TorchDType::BFloat16
            | TorchDType::Float8E4M3FN
            | TorchDType::Float8E4M3FNUZ
            | TorchDType::Float8E5M2
            | TorchDType::Float8E5M2FNUZ
    );
    if numel == 1 && unchecked_at_one {
        return Ok(());
    }

    if fill_is_int {
        // A Python int too large for `i64` raises OverflowError here, which is
        // what torch does too ("int too big to convert") -- from the same
        // place, its own Python-to-Scalar conversion.
        let value: i64 = fill.extract()?;
        if let Some((min, max)) = int_range(dtype) {
            let fits = if value < 0 && min == 0 {
                // Two's-complement wrap, allowed for magnitude <= max.
                value.checked_neg().map(|m| m <= max).unwrap_or(false)
            } else {
                value >= min && value <= max
            };
            if !fits {
                return Err(overflow(dtype));
            }
        } else if let Some(max) = float_max(dtype) {
            if (value as f64).abs() > max {
                return Err(overflow(dtype));
            }
        }
    } else {
        let value: f64 = fill.extract()?;
        if let Some((min, max)) = int_range(dtype) {
            // Integer dtypes have neither infinity nor NaN, so both are
            // refused; a finite value must land inside the range.
            if value.is_nan() || value < min as f64 || value > max as f64 {
                return Err(overflow(dtype));
            }
        } else if let Some(max) = float_max(dtype) {
            // Infinity converts to infinity; only finite-but-too-large is an
            // overflow. NaN is fine -- the float types all have a quiet NaN.
            if value.is_finite() && value.abs() > max {
                return Err(overflow(dtype));
            }
        }
    }
    Ok(())
}

/// torch would promote here. The shim does not, and says so by name. Compares
/// the *torch* dtype, so `bool` and `uint8` are not accidentally the same
/// operand type just because candle stores both as `U8`.
fn same_dtype(op: &str, lhs: &PyTensorBase, rhs: &PyTensorBase) -> PyResult<TorchDType> {
    if lhs.tag() != rhs.tag() {
        return Err(not_implemented(format!(
            "{op}: dtype promotion not implemented in torch._C shim: {} vs {}",
            lhs.tag().name(),
            rhs.tag().name()
        )));
    }
    Ok(lhs.tag())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aten_dispatch, m)?)?;
    m.add_function(wrap_pyfunction!(aten_implemented, m)?)?;
    m.add_function(wrap_pyfunction!(aten_implemented_awaiting_golden, m)?)?;
    m.add_function(wrap_pyfunction!(aten_all_implemented, m)?)?;
    Ok(())
}
