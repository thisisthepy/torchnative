//! `Repr::Functional`: upstream's C++ `FunctionalTensorWrapper`, step 1 of 4
//! (issue #329).
//!
//! Direction: leader decision 2026-11-15, agent proposal; not owner-confirmed.
//! Functionalization on the shim is a real wrapper held as a `Repr` variant,
//! with a schema-driven `Functionalize` kernel at the one door in later steps,
//! and upstream's Python `FunctionalTensor` / `FunctionalTensorMode`
//! (`torch/_subclasses/functional_tensor.py`) running unchanged on top.
//! The design note, with the measurements behind it, is
//! `docs/design/FUNCTIONALIZATION.md`.
//!
//! **What step 1 is.** The wrapper exists, wraps and unwraps, and answers the
//! state queries (`torch._functionalize_has_data_mutation` and the rest) from
//! fields that the later steps write. It does not run a single operator: every
//! operator that reaches `_aten_dispatch` with a functional argument refuses by
//! name (`refuse_functional_args`), and `tensor()` refuses the variant, so a
//! kernel cannot compute on the wrapped value by forgetting to unwrap it. That
//! refusal is the property AGENTS.md section 11 asks of every non-`Dense` arm.
//!
//! **Why the state answers are truthful and not constants.** Upstream's fresh
//! wrapper answers `has_data_mutation False`, `mutation_counter 0`,
//! `are_all_mutations_hidden_from_autograd True` and so on (measured, torch
//! 2.13.0). Those are the answers of *counters that nothing has incremented*,
//! and here they are read from counters that nothing *can* increment yet,
//! because every path that would (an in-place op, `_functionalize_replace`,
//! `_functionalize_commit_update`, `_functionalize_mark_mutation_hidden_from_autograd`)
//! refuses by name until step 3. When step 3 lands, it writes these fields and
//! the queries follow without being touched.
//!
//! **What later steps add, and the fields reserved for them.**
//!
//! * Step 2 (non-mutating ops and views): the door's `Functionalize` kernel
//!   unwraps, runs the op on the values, and wraps each output in a fresh
//!   storage (`FunctionalWrapper::new`). A view op shares the input's
//!   `storage` `Arc` instead, sets `is_base` false, and records the view step;
//!   `is_multi_output_view` is set for `split`/`unbind`/`chunk` outputs and
//!   `is_symbolic` for a view step with symbolic inputs. The list of view steps
//!   (upstream's `view_metas_`) and the storage's base value (upstream's
//!   `base_`) are **not** declared yet: nothing in step 1 can read them, and a
//!   field nothing reads is a constant wearing a name. They join `is_base`,
//!   which they will replace (`is_base == view_steps.is_empty()`).
//! * Step 3 (mutation of a non-view tensor): an in-place op runs its functional
//!   variant, replaces `value`, bumps `storage.generation` and the counters in
//!   `FunctionalStorage`, and sets `has_metadata_mutation` for a metadata
//!   mutation (`t_`, `resize_`). A metadata mutation changes the wrapper's own
//!   layout, so step 3 replaces the whole `Arc<FunctionalWrapper>` (through
//!   `PyTensorBase::replace_with`) rather than mutating `shape` in place; that
//!   is why the layout fields are plain and immutable.
//! * Step 4 (mutation through views): a view wrapper whose `generation` lags
//!   the storage's replays the storage's updates in `_functionalize_sync`;
//!   until then, `sync` on a lagging wrapper refuses by name (unreachable in
//!   step 1, since nothing can make one lag).
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use pyo3::prelude::*;

use crate::device::PyDevice;
use crate::tensor::{PyTensorBase, Repr};

/// How many wrapper payloads are alive in the process.
///
/// The door's argument walk (`refuse_functional_args`) runs only while this is
/// non-zero, so a process that never wraps a tensor pays one relaxed load per
/// dispatch and nothing else, the same bargain `python_kernel_armed` makes.
/// Counted on the payload, not on the Python object: `promote` and
/// `Tensor._make_subclass` clone a `PyTensorBase`, and a clone shares the
/// payload's `Arc` rather than making a second wrapper.
static FUNCTIONAL_LIVE: AtomicUsize = AtomicUsize::new(0);

#[inline]
pub(crate) fn any_live() -> bool {
    FUNCTIONAL_LIVE.load(Ordering::Relaxed) != 0
}

/// Upstream's `FunctionalStorageImpl`: what a base and all of its views share.
///
/// Step 1 creates one per `torch._to_functional_tensor` and nothing else ever
/// shares one, because only a view op (step 2) does that.
pub struct FunctionalStorage {
    /// Upstream's `generation_`: bumped by every committed mutation (step 3).
    /// A wrapper whose own `generation` is behind this one is out of date and
    /// `_functionalize_sync` must replay (step 4).
    generation: AtomicU64,
    /// Upstream's `mutation_counter_`, read by
    /// `torch._functionalize_mutation_counter` and, through
    /// `has_data_mutation`, by AOT's metadata pass.
    mutation_counter: AtomicU64,
    /// Upstream's count of the mutations marked hidden from autograd.
    /// `are_all_mutations_hidden_from_autograd` compares it with
    /// `mutation_counter`; equal at zero is upstream's `True` for a fresh
    /// wrapper (measured).
    ///
    /// Measured on 2.13.0 for step 3, which owns the write: `add_` alone gives
    /// `False`; one `_functionalize_mark_mutation_hidden_from_autograd` before
    /// or after the `add_` gives `True`, and so do two marks with no mutation.
    /// That is not a plain equality of two counters, so step 3 must re-derive
    /// the rule before it writes here.
    mutations_hidden_from_autograd: AtomicU64,
    /// The same, for mutations made under `no_grad` or inference mode
    /// (`are_all_mutations_under_no_grad_or_inference_mode`; a fresh wrapper
    /// answers `True`, measured, and a `mul_` under `no_grad` keeps it `True`).
    mutations_under_no_grad_or_inference_mode: AtomicU64,
    /// Upstream's `inductor_storage_resized_counter_` (written by
    /// `resize_storage_bytes_` in step 3).
    inductor_storage_resized: AtomicU64,
}

impl FunctionalStorage {
    fn fresh() -> Self {
        Self {
            generation: AtomicU64::new(0),
            mutation_counter: AtomicU64::new(0),
            mutations_hidden_from_autograd: AtomicU64::new(0),
            mutations_under_no_grad_or_inference_mode: AtomicU64::new(0),
            inductor_storage_resized: AtomicU64::new(0),
        }
    }
}

/// Upstream's `FunctionalTensorWrapper`, minus the replay.
pub struct FunctionalWrapper {
    /// Upstream's `value_`: the wrapped tensor, as the Python object it was,
    /// because identity is observable: `torch._from_functional_tensor(
    /// torch._to_functional_tensor(t)) is t` is `True` upstream (measured), and
    /// a wrapped `FakeTensor` unwraps to that `FakeTensor`. A mutex because
    /// step 3 replaces it (`_functionalize_replace`, an in-place op).
    value: Mutex<Py<PyAny>>,
    /// The wrapper's own layout. Upstream's wrapper is a `TensorImpl` with its
    /// own sizes and strides, copied from `value_` at construction (and again
    /// by step 3 after a metadata mutation). Measured: wrapping `ones(2,3).t()`
    /// reports stride `(1, 3)`, wrapping `ones(2,3)[1:]` reports
    /// `storage_offset 3`, so the copy is exact rather than a contiguous
    /// re-derivation.
    pub(crate) shape: Vec<usize>,
    pub(crate) stride: Vec<usize>,
    pub(crate) storage_offset: usize,
    /// The wrapped value's device as Python reports it, not its `Repr`'s: a
    /// `FakeTensor` is `Repr::Meta` underneath and reports `cpu`, and its
    /// wrapper reports `cpu` and `is_meta False` upstream (measured).
    pub(crate) device: PyDevice,
    storage: Arc<FunctionalStorage>,
    /// The storage generation `value` reflects. Equal to the storage's for
    /// every wrapper step 1 can make.
    generation: AtomicU64,
    /// `torch._is_functional_tensor_base`: `True` for what
    /// `_to_functional_tensor` makes, `False` for a view of it (measured).
    /// Step 2 replaces it with the list of view steps.
    is_base: bool,
    /// `torch._functionalize_is_multi_output_view` (step 2 writes it).
    is_multi_output_view: bool,
    /// `torch._functionalize_is_symbolic` (step 2 writes it for a view step
    /// with symbolic inputs; the shim has no symbolic sizes today).
    is_symbolic: bool,
    /// `torch._functionalize_has_metadata_mutation` (step 3 writes it).
    has_metadata_mutation: AtomicBool,
    /// `torch._functionalize_was_storage_changed` and
    /// `torch._functionalize_storage_changed_counter`, written by
    /// `torch._functionalize_mark_storage_changed`, which is pure state and is
    /// implemented here (measured: it makes `was_storage_changed` `True`).
    was_storage_changed: AtomicBool,
    storage_changed_counter: AtomicU64,
}

impl FunctionalWrapper {
    pub(crate) fn new(
        value: Py<PyAny>,
        shape: Vec<usize>,
        stride: Vec<usize>,
        storage_offset: usize,
        device: PyDevice,
    ) -> Self {
        FUNCTIONAL_LIVE.fetch_add(1, Ordering::Relaxed);
        Self {
            value: Mutex::new(value),
            shape,
            stride,
            storage_offset,
            device,
            storage: Arc::new(FunctionalStorage::fresh()),
            generation: AtomicU64::new(0),
            is_base: true,
            is_multi_output_view: false,
            is_symbolic: false,
            has_metadata_mutation: AtomicBool::new(false),
            was_storage_changed: AtomicBool::new(false),
            storage_changed_counter: AtomicU64::new(0),
        }
    }

    pub(crate) fn value(&self, py: Python<'_>) -> Py<PyAny> {
        self.value
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
            .clone_ref(py)
    }

    fn is_current(&self) -> bool {
        self.generation.load(Ordering::SeqCst) == self.storage.generation.load(Ordering::SeqCst)
    }
}

impl Drop for FunctionalWrapper {
    fn drop(&mut self) {
        FUNCTIONAL_LIVE.fetch_sub(1, Ordering::Relaxed);
    }
}

/// `tensor()`'s refusal for `Repr::Functional`, and the door's.
///
/// Computing on the wrapped value would compile and would give the plain
/// answer, which is exactly the silent wrong answer: upstream records the op
/// in the functionalization state (an `add_` on a wrapper sets
/// `has_data_mutation` and leaves the wrapped tensor untouched, measured), and
/// a kernel that read through the wrapper would mutate the wrapped tensor and
/// record nothing.
pub fn no_functional_storage() -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err(
        "torch._C shim: this is a functional tensor (torch._to_functional_tensor); \
         its wrapped value is not storage a kernel may read, because an op on it \
         must be recorded by the Functionalize kernel, which is not implemented \
         yet (issue #329, steps 2 to 4). torch._from_functional_tensor(t) is the \
         way to the wrapped value.",
    )
}

/// Which follow-up of #329 implements an op on a functional argument.
fn step_for(op: &str) -> &'static str {
    // `aten.add_.Tensor` -> `add_`; an in-place or `out=` variant is a
    // mutation (step 3, and step 4 if the mutated tensor is a view); anything
    // else is step 2 (non-mutating ops and views).
    let base = op.split('.').nth(1).unwrap_or(op);
    let overload = op.split('.').nth(2).unwrap_or("");
    if base.ends_with('_') || overload == "out" || overload.starts_with("out_") {
        "#329 step 3 (mutation; step 4 for mutation through a view)"
    } else {
        "#329 step 2 (non-mutating ops and views)"
    }
}

/// The refusal at the door for an operator with a functional argument.
pub fn functional_op_refusal(op: &str) -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err(format!(
        "not implemented in torch._C shim: {op} on a functional tensor -- the \
         Functionalize kernel that unwraps, runs and re-wraps it is {}, not \
         implemented yet. Upstream functionalizes this op even with no \
         FunctionalTensorMode active (the Functionalize key is on the tensor); \
         running it on the wrapped value instead would drop it from the \
         functionalization record, so it is refused.",
        step_for(op)
    ))
}

fn is_functional(value: &Bound<'_, PyAny>) -> bool {
    match value.cast::<PyTensorBase>() {
        Ok(cell) => match cell.try_borrow() {
            Ok(t) => matches!(t.repr(), Repr::Functional(_)),
            // A tensor mutably borrowed at the door is not a state any caller
            // produces; answering "not functional" for it would be the silent
            // case this walk exists to prevent, so it counts as functional and
            // the op refuses.
            Err(_) => true,
        },
        Err(_) => false,
    }
}

fn any_functional_nested(value: &Bound<'_, PyAny>, depth: u8) -> bool {
    if is_functional(value) {
        return true;
    }
    if depth == 0 {
        return false;
    }
    if let Ok(seq) = value.cast::<pyo3::types::PyList>() {
        return seq.iter().any(|item| any_functional_nested(&item, depth - 1));
    }
    if let Ok(seq) = value.cast::<pyo3::types::PyTuple>() {
        return seq.iter().any(|item| any_functional_nested(&item, depth - 1));
    }
    false
}

/// The door's check, called by `aten_dispatch_entry` before the Python
/// kernels, the mode stack and the subclass path, because upstream's
/// `Functionalize` key sits above `Python`: a mode never sees a C++ wrapper
/// upstream, it sees what the Functionalize kernel unwrapped.
///
/// Unlike `scan_for_device`, the walk does not stop at the first non-tensor in
/// a list: `aten.index.Tensor`'s `indices` is `[None, f]`. Two levels deep,
/// which is every list-of-tensors position in the schemas.
pub fn refuse_functional_args(
    op: &str,
    args: &Bound<'_, pyo3::types::PyTuple>,
    kwargs: Option<&Bound<'_, pyo3::types::PyDict>>,
) -> PyResult<()> {
    let hit = args.iter().any(|item| any_functional_nested(&item, 2))
        || kwargs.is_some_and(|k| k.iter().any(|(_, v)| any_functional_nested(&v, 2)));
    if hit {
        return Err(functional_op_refusal(op));
    }
    Ok(())
}

fn not_functional(name: &str) -> PyErr {
    // Upstream's wording, measured on 2.13.0 for every query in this family
    // handed a plain tensor: `at::functionalization::impl::isFunctionalTensor(t)
    // INTERNAL ASSERT FAILED at ".../python_torch_functions_manual.cpp":<line>`.
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "at::functionalization::impl::isFunctionalTensor(t) INTERNAL ASSERT FAILED \
         (torch._C shim: torch.{name} expects a functional tensor, made by \
         torch._to_functional_tensor)"
    ))
}

fn payload(name: &str, t: &Bound<'_, PyAny>) -> PyResult<Arc<FunctionalWrapper>> {
    let cell = t.cast::<PyTensorBase>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{name}(): incompatible function arguments. The following argument \
             types are supported:\n    1. (arg0: torch.Tensor)"
        ))
    })?;
    let borrowed = cell.borrow();
    match borrowed.repr() {
        Repr::Functional(w) => Ok(Arc::clone(w)),
        Repr::Dense(_)
        | Repr::Meta { .. }
        | Repr::Quantized(_)
        | Repr::Vulkan(_)
        | Repr::Complex { .. } => Err(not_functional(name)),
    }
}

/// `torch._C._shim_to_functional(t)`: the Rust half of
/// `torch._to_functional_tensor`. The bootstrap checks the argument's type and
/// promotes the result to `torch.Tensor`.
#[pyfunction]
fn _shim_to_functional(py: Python<'_>, t: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let cell = t.cast::<PyTensorBase>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "_to_functional_tensor(): incompatible function arguments. The following \
             argument types are supported:\n    1. (arg0: torch.Tensor) -> torch.Tensor",
        )
    })?;
    // Read through Python, not the `Repr`: a `FakeTensor` overrides `device`.
    let device: PyDevice = t.getattr("device")?.extract()?;
    let wrapper = {
        let borrowed = cell.borrow();
        PyTensorBase::functional(py, &borrowed, t.clone().unbind(), device)?
    };
    let obj = Py::new(py, wrapper)?.into_any();
    crate::tensor::promote(py, obj)
}

/// `torch._C._shim_from_functional(f)`: upstream's `from_functional_tensor`
/// without a sync, which is upstream's behaviour too: after `f.add_(1)` on a
/// base, `_from_functional_tensor(f)` is the new value, but for a view of `f`
/// it is the stale value until `_functionalize_sync` (measured). In step 1
/// nothing can make a value stale.
#[pyfunction]
fn _shim_from_functional(py: Python<'_>, f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let w = payload("_from_functional_tensor", f).map_err(|e| {
        if e.is_instance_of::<pyo3::exceptions::PyRuntimeError>(py) {
            // Upstream's wording for a plain argument here is its own
            // (`!assert_functional INTERNAL ASSERT FAILED at
            // ".../FunctionalTensorWrapper.cpp"`, measured).
            pyo3::exceptions::PyRuntimeError::new_err(
                "!assert_functional INTERNAL ASSERT FAILED (torch._C shim: \
                 torch._from_functional_tensor expects a functional tensor)",
            )
        } else {
            e
        }
    })?;
    Ok(w.value(py))
}

/// `torch._C._shim_is_functional(t)`: `False` for anything that is not a
/// `TensorBase`, which the bootstrap turns into upstream's `TypeError` first.
#[pyfunction]
fn _shim_is_functional(t: &Bound<'_, PyAny>) -> bool {
    match t.cast::<PyTensorBase>() {
        Ok(cell) => matches!(cell.borrow().repr(), Repr::Functional(_)),
        Err(_) => false,
    }
}

/// How many wrapper payloads are alive, for tests of the door's gate.
#[pyfunction]
fn _shim_functional_live() -> usize {
    FUNCTIONAL_LIVE.load(Ordering::Relaxed)
}

/// `torch._C._shim_functional_query(name, f)`: one entry point for the state
/// family, so the bootstrap's upstream-named functions stay one line each.
///
/// Every name here is answered from a field. A name whose answer needs
/// machinery of steps 2 to 4 is not here, and the bootstrap refuses it by name.
#[pyfunction]
fn _shim_functional_query(py: Python<'_>, name: &str, f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    use pyo3::IntoPyObjectExt;
    let w = payload(name, f)?;
    let s = &w.storage;
    let bool_ = |b: bool| b.into_py_any(py);
    match name {
        "_is_functional_tensor_base" => bool_(w.is_base),
        "_functionalize_has_data_mutation" => bool_(s.generation.load(Ordering::SeqCst) > 0),
        "_functionalize_has_metadata_mutation" => bool_(w.has_metadata_mutation.load(Ordering::SeqCst)),
        "_functionalize_was_storage_changed" => bool_(w.was_storage_changed.load(Ordering::SeqCst)),
        "_functionalize_storage_changed_counter" => {
            w.storage_changed_counter.load(Ordering::SeqCst).into_py_any(py)
        }
        "_functionalize_mutation_counter" => s.mutation_counter.load(Ordering::SeqCst).into_py_any(py),
        "_functionalize_was_inductor_storage_resized" => {
            bool_(s.inductor_storage_resized.load(Ordering::SeqCst) > 0)
        }
        // Upstream's binding declares `-> bool` and answers `False` for a
        // fresh wrapper (measured); the counter is reported the same way.
        "_functionalize_inductor_storage_resized_counter" => {
            bool_(s.inductor_storage_resized.load(Ordering::SeqCst) > 0)
        }
        "_functionalize_are_all_mutations_hidden_from_autograd" => bool_(
            s.mutations_hidden_from_autograd.load(Ordering::SeqCst)
                == s.mutation_counter.load(Ordering::SeqCst),
        ),
        "_functionalize_are_all_mutations_under_no_grad_or_inference_mode" => bool_(
            s.mutations_under_no_grad_or_inference_mode.load(Ordering::SeqCst)
                == s.mutation_counter.load(Ordering::SeqCst),
        ),
        "_functionalize_is_multi_output_view" => bool_(w.is_multi_output_view),
        "_functionalize_is_symbolic" => bool_(w.is_symbolic),
        "_functionalize_mark_storage_changed" => {
            w.was_storage_changed.store(true, Ordering::SeqCst);
            w.storage_changed_counter.fetch_add(1, Ordering::SeqCst);
            Ok(py.None())
        }
        // A no-op exactly when the wrapper is current, which is every wrapper
        // step 1 can make. A lagging one needs step 4's replay.
        "_functionalize_sync" => {
            if w.is_current() {
                Ok(py.None())
            } else {
                Err(pyo3::exceptions::PyNotImplementedError::new_err(
                    "not implemented in torch._C shim: torch._functionalize_sync on a \
                     functional tensor whose storage has updates it has not seen -- \
                     replaying them is #329 step 4 (mutation through views)",
                ))
            }
        }
        other => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "torch._C shim internal error: _shim_functional_query has no query {other:?}"
        ))),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_shim_to_functional, m)?)?;
    m.add_function(wrap_pyfunction!(_shim_from_functional, m)?)?;
    m.add_function(wrap_pyfunction!(_shim_is_functional, m)?)?;
    m.add_function(wrap_pyfunction!(_shim_functional_live, m)?)?;
    m.add_function(wrap_pyfunction!(_shim_functional_query, m)?)?;
    Ok(())
}
