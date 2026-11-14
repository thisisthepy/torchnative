//! `Repr::Functional`: upstream's C++ `FunctionalTensorWrapper`, and the
//! `Functionalize` kernel at the door. Steps 1 and 2 of 4 (issues #329, #333).
//!
//! Direction: leader decision 2026-11-15, agent proposal; not owner-confirmed.
//! Functionalization on the shim is a real wrapper held as a `Repr` variant,
//! with one schema-driven `Functionalize` kernel at the one door
//! (`functionalize`, called by `aten::aten_dispatch_entry`), and upstream's
//! Python `FunctionalTensor` / `FunctionalTensorMode`
//! (`torch/_subclasses/functional_tensor.py`) running unchanged on top. The
//! design note, with the measurements behind it, is
//! `docs/design/FUNCTIONALIZATION.md`.
//!
//! **Step 1** made the wrapper, its unwrap, its own metadata and the state
//! queries. **Step 2** (this file's kernel) runs the non-mutating operators
//! and the view operators on wrappers, the way upstream's Functionalize
//! fallback (`functionalizeFallback`) and its generated view kernels do,
//! measured on torch 2.13.0 (`tests/bindings/test_functional_ops.py`):
//!
//! * classify the call from its **schema** (`bootstrap.py::_functionalize_plan`,
//!   cached here): an argument written to (`(a!)`) is a mutation (step 3,
//!   refused by name); a return aliasing an argument (`(a)`) is a view, a
//!   list return a multi-output view; upstream's metadata queries are
//!   answered from the wrapper's own layout; `lift`/`lift_fresh` wrap their
//!   plain argument; everything else is pure;
//! * unwrap every functional argument, inside lists too, after checking it is
//!   current (the `_functionalize_sync` upstream calls first);
//! * run the inner operator **back through the door** (`_aten_dispatch`, via
//!   `bootstrap.py::_functionalize_redispatch`) with Functionalize in the TLS
//!   exclude set, upstream's `AutoDispatchSkipFunctionalize`, so the modes and
//!   subclasses below Functionalize see it, in upstream's order. For a view
//!   with `reapply_views` False the inner op is its `*_copy` variant;
//! * wrap the tensor outputs if any argument was functional or there was no
//!   tensor argument at all (a factory op): a fresh `FunctionalStorage` for a
//!   pure op, the input's storage and a recorded `ViewStep` for a view.
//!
//! **What the later steps add.**
//!
//! * Step 3 (mutation of a non-view tensor): an in-place op runs its functional
//!   variant, replaces `value`, bumps `storage.generation` and the counters in
//!   `FunctionalStorage`, and sets `has_metadata_mutation` for a metadata
//!   mutation (`t_`, `resize_`). A metadata mutation changes the wrapper's own
//!   layout, so step 3 replaces the whole `Arc<FunctionalWrapper>` (through
//!   `PyTensorBase::replace_with`) rather than mutating `shape` in place; that
//!   is why the layout fields are plain and immutable.
//! * Step 4 (mutation through views): a view wrapper whose `generation` lags
//!   the storage's replays its `view_steps` from the storage's base in
//!   `_functionalize_sync`; until then, `sync` on a lagging wrapper refuses by
//!   name (unreachable in step 2, since nothing can make one lag). The
//!   storage's base value (upstream's `base_`) is not declared yet: nothing
//!   reads it before step 4, and a field nothing reads is a constant wearing a
//!   name.
use std::cell::Cell;
use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use pyo3::IntoPyObjectExt;

use crate::device::PyDevice;
use crate::tensor::{PyTensorBase, Repr};

/// How many wrapper payloads are alive in the process.
///
/// The door's argument walk runs only while this is non-zero (or while
/// Functionalize is in the TLS include set), so a process that never wraps a
/// tensor pays one relaxed load per dispatch and nothing else, the same
/// bargain `python_kernel_armed` makes. Counted on the payload, not on the
/// Python object: `promote` and `Tensor._make_subclass` clone a
/// `PyTensorBase`, and a clone shares the payload's `Arc` rather than making a
/// second wrapper.
static FUNCTIONAL_LIVE: AtomicUsize = AtomicUsize::new(0);

#[inline]
pub(crate) fn any_live() -> bool {
    FUNCTIONAL_LIVE.load(Ordering::Relaxed) != 0
}

// ---------------------------------------------------------------------------
// The TLS the kernel reads
// ---------------------------------------------------------------------------

const TLS_INCLUDED: u8 = 1;
const TLS_EXCLUDED: u8 = 2;
const TLS_REAPPLY: u8 = 4;

thread_local! {
    /// `Functionalize` in the TLS include set, in the exclude set, and
    /// `reapply_views`, mirrored from `bootstrap.py::_TLS` by its setters
    /// (`_shim_functionalize_tls`). Python's `threading.local` and this are
    /// both per OS thread, so the two agree on every thread.
    static TLS: Cell<u8> = const { Cell::new(0) };
    /// Depth of upstream's `AutoDispatchBelowADInplaceOrView` around the inner
    /// call of a kernel entered through the door: the outer call recorded the
    /// view (`tensor::note_view` on the wrapper), so the inner value is not an
    /// autograd view of its input (measured: `_from_functional_tensor(
    /// f.permute(1, 0))._base is None` with `reapply_views` True, while the same
    /// op through `_op_dk(Functionalize, ...)`, which skips autograd, leaves
    /// the value a view and the wrapper not one).
    static BELOW_AD_INPLACE_OR_VIEW: Cell<u32> = const { Cell::new(0) };
}

#[derive(Clone, Copy)]
pub(crate) struct Tls(u8);

impl Tls {
    fn included(self) -> bool {
        self.0 & TLS_INCLUDED != 0
    }
    fn excluded(self) -> bool {
        self.0 & TLS_EXCLUDED != 0
    }
    fn reapply_views(self) -> bool {
        self.0 & TLS_REAPPLY != 0
    }
    /// Upstream's key computation for a dispatch with no functional argument:
    /// `(tensor keys | included) - excluded`.
    pub(crate) fn active(self) -> bool {
        self.included() && !self.excluded()
    }
}

#[inline]
pub(crate) fn tls() -> Tls {
    Tls(TLS.with(Cell::get))
}

/// `bootstrap.py`'s `_TLS` setters call this whenever the include set, the
/// exclude set or `reapply_views` changes.
#[pyfunction]
fn _shim_functionalize_tls(included: bool, excluded: bool, reapply_views: bool) {
    let mut bits = 0;
    if included {
        bits |= TLS_INCLUDED;
    }
    if excluded {
        bits |= TLS_EXCLUDED;
    }
    if reapply_views {
        bits |= TLS_REAPPLY;
    }
    TLS.with(|c| c.set(bits));
}

/// Read by `tensor::note_view`: inside the inner call of a door-entered
/// kernel the view was already recorded on the wrapper.
pub(crate) fn below_ad_inplace_or_view() -> bool {
    BELOW_AD_INPLACE_OR_VIEW.with(Cell::get) > 0
}

struct BelowAdInplaceOrView;

impl BelowAdInplaceOrView {
    fn enter() -> Self {
        BELOW_AD_INPLACE_OR_VIEW.with(|c| c.set(c.get() + 1));
        Self
    }
}

impl Drop for BelowAdInplaceOrView {
    fn drop(&mut self) {
        BELOW_AD_INPLACE_OR_VIEW.with(|c| c.set(c.get().saturating_sub(1)));
    }
}

// ---------------------------------------------------------------------------
// Counters (AGENTS.md 13.1/16)
// ---------------------------------------------------------------------------

/// What the kernel did, counted where it did it. A value that agrees with
/// upstream cannot show which path computed it; these can. Process-wide,
/// monotonic, read by `torch._C._shim_functionalize_counters()`.
struct Counters {
    /// Calls whose functional arguments the kernel unwrapped or whose outputs
    /// it wrapped (the sum of `pure`, `view`, `factory` and `lift`).
    functionalized: AtomicU64,
    pure: AtomicU64,
    view: AtomicU64,
    factory: AtomicU64,
    lift: AtomicU64,
    /// Metadata queries answered from the wrapper's own layout.
    metadata: AtomicU64,
    /// The kernel ran (`_op_dk` at the key, or Functionalize TLS-active) with
    /// no functional argument and tensor arguments: redispatched, nothing
    /// wrapped, upstream's `should_wrap_outputs` False.
    passthrough: AtomicU64,
    /// Refused by name (a mutation, a missing `_copy` kernel, ...).
    refused: AtomicU64,
}

static COUNTERS: Counters = Counters {
    functionalized: AtomicU64::new(0),
    pure: AtomicU64::new(0),
    view: AtomicU64::new(0),
    factory: AtomicU64::new(0),
    lift: AtomicU64::new(0),
    metadata: AtomicU64::new(0),
    passthrough: AtomicU64::new(0),
    refused: AtomicU64::new(0),
};

fn bump(counter: &AtomicU64) {
    counter.fetch_add(1, Ordering::Relaxed);
}

fn refused(err: PyErr) -> PyErr {
    bump(&COUNTERS.refused);
    err
}

#[pyfunction]
fn _shim_functionalize_counters(py: Python<'_>) -> PyResult<Py<PyDict>> {
    let d = PyDict::new(py);
    let c = &COUNTERS;
    for (name, counter) in [
        ("functionalized", &c.functionalized),
        ("pure", &c.pure),
        ("view", &c.view),
        ("factory", &c.factory),
        ("lift", &c.lift),
        ("metadata", &c.metadata),
        ("passthrough", &c.passthrough),
        ("refused", &c.refused),
    ] {
        d.set_item(name, counter.load(Ordering::Relaxed))?;
    }
    Ok(d.unbind())
}

// ---------------------------------------------------------------------------
// The payload
// ---------------------------------------------------------------------------

/// Upstream's `FunctionalStorageImpl`: what a base and all of its views share.
pub struct FunctionalStorage {
    /// The identity `untyped_storage()` answers with (`_cdata`, which
    /// `StorageWeakRef` keys on): shared by a base and its views, distinct
    /// from every other storage, measured. Drawn from the meta storage
    /// counter so it cannot collide with a meta tensor's.
    id: usize,
    /// `untyped_storage().nbytes()`: the size of the storage the first value
    /// lived in (24 for a wrapper of `ones(2, 3)[1:]`, measured), which every
    /// view of it reports too.
    nbytes: usize,
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
    fn fresh(nbytes: usize) -> Self {
        Self {
            id: crate::tensor::next_meta_storage_id(),
            nbytes,
            generation: AtomicU64::new(0),
            mutation_counter: AtomicU64::new(0),
            mutations_hidden_from_autograd: AtomicU64::new(0),
            mutations_under_no_grad_or_inference_mode: AtomicU64::new(0),
            inductor_storage_resized: AtomicU64::new(0),
        }
    }

    pub(crate) fn id(&self) -> usize {
        self.id
    }
}

/// One view op that took a storage's base to this wrapper (upstream's
/// `ViewMeta`), recorded so steps 3 and 4 can replay it: the op as called,
/// its arguments with the viewed tensor replaced by `None` (the viewed tensor
/// is the storage's base or the previous step's output, so it is not kept),
/// the output index of a multi-output view, and the `reapply_views` it ran
/// with. Read back by `torch._C._shim_functional_view_steps(f)`.
pub struct ViewStep {
    op: String,
    args: Py<PyTuple>,
    kwargs: Option<Py<PyDict>>,
    out_index: Option<usize>,
    reapply_views: bool,
}

/// How a new wrapper relates to existing ones.
pub(crate) enum Origin {
    /// `_to_functional_tensor`, a pure op's output, a factory op's output: a
    /// storage of its own, no view steps (a functional base).
    Fresh,
    /// A view op's output: the input's storage and generation, the input's
    /// steps plus this one.
    View {
        storage: Arc<FunctionalStorage>,
        generation: u64,
        steps: Vec<Arc<ViewStep>>,
        multi_output: bool,
    },
    /// A fresh storage that is nonetheless not a base, upstream's
    /// `reshape` of a non-contiguous tensor (`clone` then `_unsafe_view`):
    /// the storage is the copy's, and the one step is the identity
    /// `_unsafe_view` of it (measured: `_is_functional_tensor_base` False,
    /// storage not shared with the input).
    FreshView { step: Arc<ViewStep> },
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
    /// re-derivation; a view's wrapper has its inner value's layout
    /// (`permute` with `reapply_views` True: stride `(1, 3)`; with it False,
    /// the `permute_copy` value's `(2, 1)`, measured).
    pub(crate) shape: Vec<usize>,
    pub(crate) stride: Vec<usize>,
    pub(crate) storage_offset: usize,
    /// The wrapped value's device as Python reports it, not its `Repr`'s: a
    /// `FakeTensor` is `Repr::Meta` underneath and reports `cpu`, and its
    /// wrapper reports `cpu` and `is_meta False` upstream (measured).
    pub(crate) device: PyDevice,
    /// The storage this wrapper uses. A mutex because
    /// `torch._functionalize_unsafe_set(dst, src)` swaps it and nothing else
    /// (measured: value, layout and base-ness stay; a later `dst.add_(1)`
    /// counts on `src`'s storage).
    storage: Mutex<Arc<FunctionalStorage>>,
    /// The storage generation `value` reflects. Equal to the storage's for
    /// every wrapper step 2 can make.
    generation: AtomicU64,
    /// The view steps from the storage's base to this wrapper. Empty for a
    /// base: `torch._is_functional_tensor_base` is `steps.is_empty()`
    /// (measured: `True` for what `_to_functional_tensor`, a pure op and a
    /// factory op make, `False` for every view, `detach` and `_unsafe_view`
    /// included).
    view_steps: Vec<Arc<ViewStep>>,
    /// `torch._functionalize_is_multi_output_view`: an output of a view op
    /// with a list return (`split`, `unbind`, `chunk`), measured.
    is_multi_output_view: bool,
    /// `torch._functionalize_is_symbolic`. A view step with symbolic inputs
    /// refuses by name in step 2 (the shim has no symbolic sizes), so this is
    /// never written `true`.
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
        value_storage_nbytes: usize,
        origin: Origin,
    ) -> Self {
        let (storage, generation, view_steps, is_multi_output_view) = match origin {
            Origin::Fresh => (Arc::new(FunctionalStorage::fresh(value_storage_nbytes)), 0, Vec::new(), false),
            Origin::View { storage, generation, steps, multi_output } => (storage, generation, steps, multi_output),
            Origin::FreshView { step } => {
                (Arc::new(FunctionalStorage::fresh(value_storage_nbytes)), 0, vec![step], false)
            }
        };
        FUNCTIONAL_LIVE.fetch_add(1, Ordering::Relaxed);
        Self {
            value: Mutex::new(value),
            shape,
            stride,
            storage_offset,
            device,
            storage: Mutex::new(storage),
            generation: AtomicU64::new(generation),
            view_steps,
            is_multi_output_view,
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

    pub(crate) fn storage(&self) -> Arc<FunctionalStorage> {
        Arc::clone(&self.storage.lock().unwrap_or_else(|poisoned| poisoned.into_inner()))
    }

    fn is_current(&self) -> bool {
        self.generation.load(Ordering::SeqCst) == self.storage().generation.load(Ordering::SeqCst)
    }
}

impl Drop for FunctionalWrapper {
    fn drop(&mut self) {
        FUNCTIONAL_LIVE.fetch_sub(1, Ordering::Relaxed);
    }
}

/// `tensor()`'s refusal for `Repr::Functional`.
///
/// Computing on the wrapped value would compile and would give the plain
/// answer, which is exactly the silent wrong answer: upstream records the op
/// in the functionalization state (an `add_` on a wrapper sets
/// `has_data_mutation` and leaves the wrapped tensor untouched, measured), and
/// a kernel that read through the wrapper would mutate the wrapped tensor and
/// record nothing. Every operator reaches a wrapper's value through the
/// Functionalize kernel at the door, which unwraps first; this is for the
/// readers that do not dispatch.
pub fn no_functional_storage() -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err(
        "torch._C shim: this is a functional tensor (torch._to_functional_tensor); \
         its wrapped value is not storage a kernel may read, because an op on it \
         must go through the Functionalize kernel at the door, which unwraps it \
         (#329). torch._from_functional_tensor(t) is the way to the wrapped value.",
    )
}

/// The refusal for an op that reaches a kernel with a functional argument by
/// a route that bypassed the Functionalize kernel (the Rust callers of
/// `aten_dispatch`: the tape, capture replay), and for a mutation, which is
/// step 3.
pub fn functional_op_refusal(op: &str) -> PyErr {
    let base = op.split('.').nth(1).unwrap_or(op);
    let overload = op.split('.').nth(2).unwrap_or("");
    if base.ends_with('_') || overload == "out" || overload.starts_with("out_") {
        return mutation_refusal(op);
    }
    pyo3::exceptions::PyNotImplementedError::new_err(format!(
        "not implemented in torch._C shim: {op} on a functional tensor reached a \
         kernel without passing the Functionalize kernel at the door (#329); a \
         kernel computing on the wrapped value would drop the op from the \
         functionalization record, so it is refused."
    ))
}

fn mutation_refusal(op: &str) -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err(format!(
        "not implemented in torch._C shim: {op} on a functional tensor -- its schema \
         writes to an argument, and functionalizing a mutation (run the functional \
         variant, replace the value, bump the storage generation, set the mutation \
         flags) is functionalization step 3 (mutation; step 4 for mutation through \
         a view), #329, not implemented yet. Upstream functionalizes this op even \
         with no FunctionalTensorMode active (the Functionalize key is on the \
         tensor); running it on the wrapped value instead would drop it from the \
         functionalization record, so it is refused."
    ))
}

fn is_functional(value: &Bound<'_, PyAny>) -> bool {
    match value.cast::<PyTensorBase>() {
        Ok(cell) => match cell.try_borrow() {
            Ok(t) => matches!(t.repr(), Repr::Functional(_)),
            // A tensor mutably borrowed at the door is not a state any caller
            // produces; answering "not functional" for it would be the silent
            // case this walk exists to prevent, so it counts as functional
            // and the kernel refuses when it tries to unwrap it.
            Err(_) => true,
        },
        Err(_) => false,
    }
}

/// Visit every argument and the elements of every list or tuple argument,
/// which is every tensor position in the aten schemas (`Tensor[]`,
/// `Tensor?[]`: `aten.index.Tensor`'s `indices` is `[None, f]`, so the walk
/// does not stop at a non-tensor).
fn any_leaf(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    test: &dyn Fn(&Bound<'_, PyAny>) -> bool,
) -> bool {
    let visit = |value: &Bound<'_, PyAny>| -> bool {
        if test(value) {
            return true;
        }
        if let Ok(seq) = value.cast::<PyList>() {
            return seq.iter().any(|item| test(&item));
        }
        if let Ok(seq) = value.cast::<PyTuple>() {
            return seq.iter().any(|item| test(&item));
        }
        false
    };
    args.iter().any(|item| visit(&item)) || kwargs.is_some_and(|k| k.iter().any(|(_, v)| visit(&v)))
}

fn has_functional(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> bool {
    any_leaf(args, kwargs, &|v| is_functional(v))
}

fn has_tensor(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> bool {
    any_leaf(args, kwargs, &|v| v.cast::<PyTensorBase>().is_ok())
}

fn has_symbolic(args: &Bound<'_, PyTuple>, kwargs: Option<&Bound<'_, PyDict>>) -> bool {
    any_leaf(args, kwargs, &|v| {
        v.get_type()
            .name()
            .map(|n| matches!(n.to_string().as_str(), "SymInt" | "SymFloat" | "SymBool"))
            .unwrap_or(false)
    })
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

fn try_payload(t: &Bound<'_, PyAny>) -> Option<Arc<FunctionalWrapper>> {
    let cell = t.cast::<PyTensorBase>().ok()?;
    let borrowed = cell.try_borrow().ok()?;
    match borrowed.repr() {
        Repr::Functional(w) => Some(Arc::clone(w)),
        Repr::Dense(_)
        | Repr::Meta { .. }
        | Repr::Quantized(_)
        | Repr::Vulkan(_)
        | Repr::Complex { .. } => None,
    }
}

/// Wrap `value` (a tensor object) in a new functional wrapper.
fn wrap_value(py: Python<'_>, value: &Bound<'_, PyAny>, origin: Origin) -> PyResult<Py<PyAny>> {
    let cell = value.cast::<PyTensorBase>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(
            "_to_functional_tensor(): incompatible function arguments. The following \
             argument types are supported:\n    1. (arg0: torch.Tensor) -> torch.Tensor",
        )
    })?;
    // Read through Python, not the `Repr`: a `FakeTensor` overrides `device`.
    let device: PyDevice = value.getattr("device")?.extract()?;
    let wrapper = {
        let borrowed = cell.borrow();
        PyTensorBase::functional(py, &borrowed, value.clone().unbind(), device, origin)?
    };
    let obj = Py::new(py, wrapper)?.into_any();
    crate::tensor::promote(py, obj)
}

/// `torch._C._shim_to_functional(t)`: the Rust half of
/// `torch._to_functional_tensor`. The bootstrap checks the argument's type and
/// promotes the result to `torch.Tensor`.
#[pyfunction]
fn _shim_to_functional(py: Python<'_>, t: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    wrap_value(py, t, Origin::Fresh)
}

/// `torch._C._shim_from_functional(f)`: upstream's `from_functional_tensor`
/// without a sync, which is upstream's behaviour too: after `f.add_(1)` on a
/// base, `_from_functional_tensor(f)` is the new value, but for a view of `f`
/// it is the stale value until `_functionalize_sync` (measured). In step 2
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
/// machinery of steps 3 and 4 is not here, and the bootstrap refuses it by name.
#[pyfunction]
fn _shim_functional_query(py: Python<'_>, name: &str, f: &Bound<'_, PyAny>) -> PyResult<Py<PyAny>> {
    let w = payload(name, f)?;
    let s = w.storage();
    let bool_ = |b: bool| b.into_py_any(py);
    match name {
        "_is_functional_tensor_base" => bool_(w.view_steps.is_empty()),
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
        // step 2 can make. A lagging one needs step 4's replay.
        "_functionalize_sync" => {
            if w.is_current() {
                Ok(py.None())
            } else {
                Err(lagging_refusal("torch._functionalize_sync"))
            }
        }
        other => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "torch._C shim internal error: _shim_functional_query has no query {other:?}"
        ))),
    }
}

fn lagging_refusal(what: &str) -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err(format!(
        "not implemented in torch._C shim: {what} on a functional tensor whose storage \
         has updates it has not seen -- replaying them is functionalization step 4 \
         (mutation through views), #329"
    ))
}

/// `torch._C._shim_functional_unsafe_set(dst, src)`: the Rust half of
/// `torch._functionalize_unsafe_set` for two functional tensors. Only the
/// storage moves (measured on 2.13.0: `dst`'s value, layout and base-ness are
/// unchanged and `StorageWeakRef(dst.untyped_storage())` equals `src`'s).
///
/// Upstream also accepts a plain tensor on either side (it swaps any
/// `TensorImpl`'s storage). A plain dense tensor's storage here is a candle
/// buffer, which cannot become a `FunctionalStorage` or adopt another
/// tensor's buffer under its own layout, so that case refuses by name. Its
/// caller upstream, `return_and_correct_aliasing`, passes `FunctionalTensor`s
/// (wrapper subclasses, plain from functionalization's point of view), which
/// this shim cannot make yet (`_make_wrapper_subclass`).
#[pyfunction]
fn _shim_functional_unsafe_set(dst: &Bound<'_, PyAny>, src: &Bound<'_, PyAny>) -> PyResult<()> {
    match (try_payload(dst), try_payload(src)) {
        (Some(d), Some(s)) => {
            let storage = s.storage();
            *d.storage.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = storage;
            Ok(())
        }
        _ => Err(crate::err::not_implemented(
            "not implemented in torch._C shim: torch._functionalize_unsafe_set with a \
             plain (non-functional) tensor on either side -- upstream swaps that \
             TensorImpl's storage, and a plain tensor's storage here is a candle buffer \
             that cannot take a functional storage, or another buffer under its own \
             layout. Two functional tensors are supported (#329).",
        )),
    }
}

/// `untyped_storage()` of a functional tensor (`tensor.rs` calls this).
pub(crate) fn untyped_storage(py: Python<'_>, w: &FunctionalWrapper) -> PyResult<Py<PyAny>> {
    let s = w.storage();
    crate::storage::functional(py, s.nbytes, s.id, w.device.clone())
}

/// `torch._C._shim_functional_view_steps(f)`: the recorded view steps, as
/// `[op, args, kwargs, out_index, reapply_views]` each, base first.
#[pyfunction]
fn _shim_functional_view_steps(py: Python<'_>, f: &Bound<'_, PyAny>) -> PyResult<Py<PyList>> {
    let w = payload("_shim_functional_view_steps", f)?;
    let items = PyList::empty(py);
    for step in &w.view_steps {
        let entry = PyList::empty(py);
        entry.append(&step.op)?;
        entry.append(step.args.bind(py))?;
        match &step.kwargs {
            Some(k) => entry.append(k.bind(py))?,
            None => entry.append(py.None())?,
        }
        entry.append(step.out_index)?;
        entry.append(step.reapply_views)?;
        items.append(entry)?;
    }
    Ok(items.unbind())
}

// ---------------------------------------------------------------------------
// The plan: what the schema says about an op
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
enum Kind {
    /// No argument written, no return aliased: unwrap, run, wrap fresh.
    Pure,
    /// A return aliases an argument (`Tensor(a)`): share its storage.
    View,
    /// An argument is written (`Tensor(a!)`, `out=`): step 3.
    Mutating,
    /// `lift`, `lift_fresh`, `lift_fresh_copy`: wrap a plain argument
    /// (upstream's own kernels in `FunctionalizeFallbackKernel.cpp`).
    Lift,
    /// Upstream's `FunctionalTensor.metadata_fns`: answered from the
    /// wrapper's own layout, nothing dispatched below (measured).
    Metadata,
}

#[derive(Clone)]
struct Plan {
    kind: Kind,
    /// The aliased argument of a view: its schema position and name.
    alias: Option<(usize, String)>,
    /// A view with a list return.
    multi_output: bool,
    /// The `*_copy` variant a view runs with `reapply_views` False, and
    /// whether the shim has a kernel for it.
    copy_op: Option<String>,
    copy_implemented: bool,
    /// For a view upstream never functionalizes directly (no `_copy` variant:
    /// it is CompositeImplicitAutograd and decomposes first), which the shim
    /// does not decompose: what its result is when the inner op returned a
    /// copy rather than a view of the value.
    on_copy: OnCopy,
}

/// See `Plan::on_copy`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OnCopy {
    /// A view with a `_copy` variant: upstream functionalizes it as a view
    /// whatever its value turns out to be, and so does this.
    NotChecked,
    /// Upstream decomposes it to a pure op when it copies (`to` to
    /// `_to_copy`, `contiguous` to `clone`): a fresh base.
    Fresh,
    /// Upstream decomposes it to `clone` then `_unsafe_view` when it copies
    /// (the `reshape` family, measured): a fresh storage that is not a base.
    UnsafeView,
}

static PLAN_RESOLVER: OnceLock<Py<PyAny>> = OnceLock::new();
static REDISPATCH: OnceLock<Py<PyAny>> = OnceLock::new();
static PLANS: Mutex<Option<HashMap<Box<str>, Plan>>> = Mutex::new(None);

/// `bootstrap.py::_install_functionalization` registers the schema reader and
/// the redispatcher, the way `tensor::set_view_alias_resolver` is registered.
#[pyfunction]
fn _shim_set_functionalize_hooks(resolver: Py<PyAny>, redispatch: Py<PyAny>) {
    let _ = PLAN_RESOLVER.set(resolver);
    let _ = REDISPATCH.set(redispatch);
}

fn plans() -> std::sync::MutexGuard<'static, Option<HashMap<Box<str>, Plan>>> {
    PLANS.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
}

fn not_installed() -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(
        "torch._C shim internal error: the Functionalize kernel ran before \
         bootstrap.py registered its hooks (_shim_set_functionalize_hooks)",
    )
}

/// The plan for `op`, from its schema, cached. The lock is never held across
/// the Python call (`tensor::view_alias_argument`'s reason).
fn plan_for(py: Python<'_>, op: &str) -> PyResult<Plan> {
    if let Some(hit) = plans().as_ref().and_then(|t| t.get(op).cloned()) {
        return Ok(hit);
    }
    let resolver = PLAN_RESOLVER.get().ok_or_else(not_installed)?;
    let answer = resolver.bind(py).call1((op,))?;
    if answer.is_none() {
        // A placeholder schema: there is nothing to classify the op by, and
        // guessing "pure" would wrap a view's output in a fresh storage.
        return Err(crate::err::not_implemented(format!(
            "not implemented in torch._C shim: {op} on a functional tensor -- the \
             Functionalize kernel classifies an op from its schema, and this op's \
             schema is not known here (#329)"
        )));
    }
    let (kind, alias_index, alias_name, multi_output, copy_op, copy_implemented, on_copy): (
        String,
        Option<usize>,
        Option<String>,
        bool,
        Option<String>,
        bool,
        Option<String>,
    ) = answer.extract()?;
    let on_copy = match on_copy.as_deref() {
        None => OnCopy::NotChecked,
        Some("fresh") => OnCopy::Fresh,
        Some("unsafe_view") => OnCopy::UnsafeView,
        Some(other) => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "torch._C shim internal error: _functionalize_plan({op:?}) answered on_copy {other:?}"
            )))
        }
    };
    let kind = match kind.as_str() {
        "pure" => Kind::Pure,
        "view" => Kind::View,
        "mutating" => Kind::Mutating,
        "lift" => Kind::Lift,
        "metadata" => Kind::Metadata,
        other => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "torch._C shim internal error: _functionalize_plan({op:?}) answered kind {other:?}"
            )))
        }
    };
    let plan = Plan {
        kind,
        alias: alias_index.zip(alias_name),
        multi_output,
        copy_op,
        copy_implemented,
        on_copy,
    };
    plans().get_or_insert_with(HashMap::new).insert(op.into(), plan.clone());
    Ok(plan)
}

// ---------------------------------------------------------------------------
// The kernel
// ---------------------------------------------------------------------------

/// How the kernel was reached.
#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) enum Entry {
    /// `aten_dispatch_entry`, with autograd (and `ADInplaceOrView`) above:
    /// the view is recorded on the wrapper by the door's `note_view`, and the
    /// inner call runs below `ADInplaceOrView`.
    Door,
    /// `OpOverload._op_dk(DispatchKey.Functionalize, ...)`: the kernel runs
    /// whatever the TLS says, as upstream's `callBoxedForDispatchKey` does.
    OpDk,
}

/// Run the inner op back through the door with Functionalize excluded.
fn redispatch(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    entry: Entry,
) -> PyResult<Py<PyAny>> {
    let hook = REDISPATCH.get().ok_or_else(not_installed)?;
    let _below = (entry == Entry::Door).then(BelowAdInplaceOrView::enter);
    Ok(hook.bind(py).call1((op, args, kwargs))?.unbind())
}

/// Replace every functional tensor (and every one inside a list or tuple) by
/// its value, after the currency check upstream's `sync` makes.
fn unwrap_one<'py>(py: Python<'py>, value: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    if let Some(w) = try_payload(value) {
        if !w.is_current() {
            return Err(lagging_refusal("an op"));
        }
        return Ok(w.value(py).into_bound(py));
    }
    if is_functional(value) {
        // Mutably borrowed while at the door: see `is_functional`.
        return Err(no_functional_storage());
    }
    if let Ok(seq) = value.cast::<PyList>() {
        if seq.iter().any(|item| is_functional(&item)) {
            let items = seq.iter().map(|item| unwrap_one(py, &item)).collect::<PyResult<Vec<_>>>()?;
            return Ok(PyList::new(py, items)?.into_any());
        }
    } else if let Ok(seq) = value.cast::<PyTuple>() {
        if seq.iter().any(|item| is_functional(&item)) {
            let items = seq.iter().map(|item| unwrap_one(py, &item)).collect::<PyResult<Vec<_>>>()?;
            return Ok(PyTuple::new(py, items)?.into_any());
        }
    }
    Ok(value.clone())
}

fn unwrap_all<'py>(
    py: Python<'py>,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<(Bound<'py, PyTuple>, Option<Bound<'py, PyDict>>)> {
    let new_args = args.iter().map(|a| unwrap_one(py, &a)).collect::<PyResult<Vec<_>>>()?;
    let new_kwargs = match kwargs {
        Some(k) => {
            let d = PyDict::new(py);
            for (key, value) in k.iter() {
                d.set_item(key, unwrap_one(py, &value)?)?;
            }
            Some(d)
        }
        None => None,
    };
    Ok((PyTuple::new(py, new_args)?, new_kwargs))
}

/// The aliased argument of a view, found by keyword or by position.
fn alias_arg<'py>(
    plan: &Plan,
    args: &Bound<'py, PyTuple>,
    kwargs: Option<&Bound<'py, PyDict>>,
) -> PyResult<Option<Bound<'py, PyAny>>> {
    let Some((index, name)) = &plan.alias else {
        return Ok(None);
    };
    if let Some(k) = kwargs {
        if let Some(value) = k.get_item(name.as_str())? {
            return Ok(Some(value));
        }
    }
    Ok(args.get_item(*index).ok())
}

/// The view step to record: the op as called, with the viewed argument
/// replaced by `None`.
fn view_step(
    py: Python<'_>,
    op: &str,
    plan: &Plan,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    out_index: Option<usize>,
    reapply_views: bool,
) -> PyResult<ViewStep> {
    let (index, name) = plan.alias.clone().unwrap_or((usize::MAX, String::new()));
    let by_keyword = kwargs.map(|k| k.contains(name.as_str())).transpose()?.unwrap_or(false);
    let items: Vec<Bound<'_, PyAny>> = args
        .iter()
        .enumerate()
        .map(|(i, a)| if i == index && !by_keyword { py.None().into_bound(py) } else { a })
        .collect();
    let kwargs = match kwargs {
        Some(k) => {
            let d = k.copy()?;
            if by_keyword {
                d.set_item(name.as_str(), py.None())?;
            }
            Some(d.unbind())
        }
        None => None,
    };
    Ok(ViewStep { op: op.to_owned(), args: PyTuple::new(py, items)?.unbind(), kwargs, out_index, reapply_views })
}

fn same_storage(a: &Bound<'_, PyAny>, b: &Bound<'_, PyAny>) -> bool {
    match (a.cast::<PyTensorBase>(), b.cast::<PyTensorBase>()) {
        (Ok(a), Ok(b)) => match (a.try_borrow(), b.try_borrow()) {
            (Ok(a), Ok(b)) => crate::tensor::share_storage(&a, &b),
            _ => false,
        },
        _ => false,
    }
}

/// Map `f` over the tensor outputs of `result` (a tensor, or a list or tuple
/// of them), keeping everything else and the container type.
fn map_outputs<'py>(
    py: Python<'py>,
    result: &Bound<'py, PyAny>,
    f: &mut dyn FnMut(usize, &Bound<'py, PyAny>) -> PyResult<Py<PyAny>>,
) -> PyResult<Py<PyAny>> {
    if result.cast::<PyTensorBase>().is_ok() {
        return f(0, result);
    }
    fn map_items<'py>(
        items: Vec<Bound<'py, PyAny>>,
        f: &mut dyn FnMut(usize, &Bound<'py, PyAny>) -> PyResult<Py<PyAny>>,
    ) -> PyResult<Vec<Py<PyAny>>> {
        items
            .iter()
            .enumerate()
            .map(|(i, item)| {
                if item.cast::<PyTensorBase>().is_ok() {
                    f(i, item)
                } else {
                    Ok(item.clone().unbind())
                }
            })
            .collect()
    }
    if let Ok(seq) = result.cast::<PyList>() {
        let items = map_items(seq.iter().collect(), f)?;
        return Ok(PyList::new(py, items)?.into_any().unbind());
    }
    if let Ok(seq) = result.cast::<PyTuple>() {
        let items = map_items(seq.iter().collect(), f)?;
        return Ok(PyTuple::new(py, items)?.into_any().unbind());
    }
    Ok(result.clone().unbind())
}

/// Upstream's metadata queries on a wrapper, answered from its own layout
/// (measured on 2.13.0: nothing is dispatched below Functionalize for them).
fn metadata_answer(
    py: Python<'_>,
    op: &str,
    w: &FunctionalWrapper,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let dim_arg = || -> PyResult<usize> {
        let raw: i64 = match kwargs.and_then(|k| k.get_item("dim").ok().flatten()) {
            Some(v) => v.extract()?,
            None => args.get_item(1)?.extract()?,
        };
        let rank = w.shape.len() as i64;
        let d = if raw < 0 { raw + rank } else { raw };
        if d < 0 || d >= rank.max(1) {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "Dimension out of range (expected to be in range of [{}, {}], but got {raw})",
                -rank.max(1),
                rank.max(1) - 1
            )));
        }
        Ok(d as usize)
    };
    match op {
        "aten.size.default" | "aten.sym_size.default" => w.shape.clone().into_py_any(py),
        "aten.sym_size.int" => w.shape.get(dim_arg()?).copied().unwrap_or(1).into_py_any(py),
        "aten.stride.default" | "aten.sym_stride.default" => w.stride.clone().into_py_any(py),
        "aten.sym_stride.int" => w.stride.get(dim_arg()?).copied().unwrap_or(1).into_py_any(py),
        "aten.dim.default" => w.shape.len().into_py_any(py),
        "aten.numel.default" | "aten.sym_numel.default" => w.shape.iter().product::<usize>().into_py_any(py),
        "aten.storage_offset.default" | "aten.sym_storage_offset.default" => w.storage_offset.into_py_any(py),
        "aten.is_contiguous.default" => crate::layout::is_contiguous(&w.shape, &w.stride).into_py_any(py),
        "aten.is_non_overlapping_and_dense.default" => {
            crate::layout::is_non_overlapping_and_dense(&w.shape, &w.stride).into_py_any(py)
        }
        other => Err(crate::err::not_implemented(format!(
            "not implemented in torch._C shim: {other} on a functional tensor -- a \
             metadata query the Functionalize kernel answers from the wrapper's own \
             layout, and this one (a memory-format question) is not answered yet (#329)"
        ))),
    }
}

/// The Functionalize kernel. `Ok(None)` means it does not apply (no
/// functional argument and Functionalize not TLS-active, at the door): the
/// door goes on to the modes and kernels as before.
pub(crate) fn functionalize(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    entry: Entry,
) -> PyResult<Option<Py<PyAny>>> {
    let tls = tls();
    let any_functional = any_live() && has_functional(args, kwargs);
    if entry == Entry::Door {
        if !any_functional && !tls.active() {
            return Ok(None);
        }
        if any_functional && tls.excluded() {
            // Upstream segfaults here (exit 139 on 2.13.0, measured in step
            // 1): a wrapper reaching a kernel with Functionalize excluded.
            return Err(refused(crate::err::not_implemented(format!(
                "not implemented in torch._C shim: {op} with a functional tensor argument \
                 while Functionalize is in the TLS exclude set -- the wrapper would reach a \
                 kernel unwrapped by nothing (upstream crashes here); unwrap it with \
                 torch._from_functional_tensor first (#329)"
            ))));
        }
    }
    let plan = plan_for(py, op).map_err(refused)?;
    if any_functional && has_symbolic(args, kwargs) {
        return Err(refused(crate::err::not_implemented(format!(
            "not implemented in torch._C shim: {op} on a functional tensor with symbolic \
             (SymInt) arguments -- the Functionalize kernel records view steps with \
             concrete sizes only (#329)"
        ))));
    }
    match plan.kind {
        Kind::Mutating if any_functional => return Err(refused(mutation_refusal(op))),
        Kind::Metadata if any_functional => {
            let first = args.get_item(0).ok().or_else(|| kwargs.and_then(|k| k.get_item("self").ok().flatten()));
            if let Some(w) = first.as_ref().and_then(try_payload) {
                bump(&COUNTERS.metadata);
                return metadata_answer(py, op, &w, args, kwargs).map(Some);
            }
        }
        Kind::Lift if any_functional => {
            // Upstream asserts `!isFunctionalTensor(self)` in these kernels.
            return Err(refused(crate::err::not_implemented(format!(
                "not implemented in torch._C shim: {op} of a tensor that is already \
                 functional (upstream asserts here: lifting wraps a plain tensor) (#329)"
            ))));
        }
        Kind::Lift => {
            let result = redispatch(py, op, args, kwargs, entry)?;
            let result = result.bind(py);
            let out = map_outputs(py, result, &mut |_, v| wrap_value(py, v, Origin::Fresh))?;
            bump(&COUNTERS.functionalized);
            bump(&COUNTERS.lift);
            return Ok(Some(out));
        }
        Kind::Mutating | Kind::Metadata | Kind::Pure | Kind::View => {}
    }
    let any_tensor = has_tensor(args, kwargs);
    // Upstream's `should_wrap_outputs`: a functional argument, or a factory
    // op (no tensor argument at all).
    let should_wrap = any_functional || !any_tensor;
    if !should_wrap || matches!(plan.kind, Kind::Mutating | Kind::Metadata) {
        // Plain arguments only: the generated kernels and the fallback both
        // redispatch with Functionalize skipped and wrap nothing.
        let out = redispatch(py, op, args, kwargs, entry)?;
        bump(&COUNTERS.passthrough);
        return Ok(Some(out));
    }
    let base = if plan.kind == Kind::View {
        alias_arg(&plan, args, kwargs)?.as_ref().and_then(try_payload)
    } else {
        None
    };
    let reapply = tls.reapply_views();
    let inner_op = match (&base, reapply) {
        (Some(_), false) => match (&plan.copy_op, plan.copy_implemented) {
            (Some(copy), true) => copy.clone(),
            (Some(copy), false) => {
                return Err(refused(crate::err::not_implemented(format!(
                    "not implemented in torch._C shim: functionalizing {op} on a functional \
                     tensor with reapply_views False runs {copy}, which has no kernel in \
                     torch._C shim (#329). FunctionalTensorMode sets reapply_views True, \
                     where the kernel runs {op} itself."
                ))))
            }
            (None, _) => {
                return Err(refused(crate::err::not_implemented(format!(
                    "not implemented in torch._C shim: functionalizing {op} on a functional \
                     tensor with reapply_views False needs a _copy variant, and {op} has \
                     none: upstream decomposes it (CompositeImplicitAutograd) before the \
                     Functionalize key, and this shim does not decompose it (#329). With \
                     reapply_views True the kernel runs {op} itself."
                ))))
            }
        },
        _ => op.to_owned(),
    };
    let (inner_args, inner_kwargs) = unwrap_all(py, args, kwargs)?;
    let base_value = base.as_ref().map(|b| b.value(py).into_bound(py));
    let base_object = if base.is_some() { alias_arg(&plan, args, kwargs)? } else { None };
    let result = redispatch(py, &inner_op, &inner_args, inner_kwargs.as_ref(), entry)?;
    let result = result.bind(py);
    let out = match &base {
        None => map_outputs(py, result, &mut |_, v| wrap_value(py, v, Origin::Fresh))?,
        Some(b) => {
            let storage = b.storage();
            let generation = b.generation.load(Ordering::SeqCst);
            map_outputs(py, result, &mut |i, v| {
                let out_index = plan.multi_output.then_some(i);
                if plan.on_copy != OnCopy::NotChecked {
                    if let (Some(bv), Some(bo)) = (base_value.as_ref(), base_object.as_ref()) {
                        // Upstream's decomposition returned `self` (`a.to(a.dtype)`,
                        // `contiguous()` of a contiguous tensor): the input
                        // wrapper itself, as the same object.
                        if v.is(bv) {
                            return Ok(bo.clone().unbind());
                        }
                    }
                }
                let copied = plan.on_copy != OnCopy::NotChecked
                    && reapply
                    && !base_value.as_ref().is_some_and(|bv| same_storage(v, bv));
                if copied && plan.on_copy == OnCopy::Fresh {
                    return wrap_value(py, v, Origin::Fresh);
                }
                if copied {
                    // `reshape` of a non-contiguous tensor: upstream's
                    // `clone` then `_unsafe_view`, a fresh storage whose one
                    // step is the identity `_unsafe_view` of the copy.
                    let shape = PyList::new(py, v.cast::<PyTensorBase>()?.borrow().dims().iter().copied())?;
                    let step = ViewStep {
                        op: "aten._unsafe_view.default".to_owned(),
                        args: PyTuple::new(py, [py.None().into_bound(py), shape.into_any()])?.unbind(),
                        kwargs: None,
                        out_index: None,
                        reapply_views: reapply,
                    };
                    return wrap_value(py, v, Origin::FreshView { step: Arc::new(step) });
                }
                let step = view_step(py, op, &plan, args, kwargs, out_index, reapply)?;
                let mut steps = b.view_steps.clone();
                steps.push(Arc::new(step));
                wrap_value(
                    py,
                    v,
                    Origin::View {
                        storage: Arc::clone(&storage),
                        generation,
                        steps,
                        multi_output: plan.multi_output,
                    },
                )
            })?
        }
    };
    bump(&COUNTERS.functionalized);
    bump(if base.is_some() {
        &COUNTERS.view
    } else if !any_tensor {
        &COUNTERS.factory
    } else {
        &COUNTERS.pure
    });
    Ok(Some(out))
}

/// `torch._C._shim_functionalize_op_dk(op, args, kwargs)`:
/// `OpOverload._op_dk(DispatchKey.Functionalize, ...)`, the call
/// `FunctionalTensorMode.__torch_dispatch__` makes.
#[pyfunction]
#[pyo3(signature = (op, args, kwargs=None))]
fn _shim_functionalize_op_dk(
    py: Python<'_>,
    op: &str,
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
) -> PyResult<Py<PyAny>> {
    let kwargs = kwargs.filter(|k| !k.is_empty());
    match functionalize(py, op, args, kwargs, Entry::OpDk)? {
        Some(out) => Ok(out),
        // `Entry::OpDk` always runs; `None` is only the door's answer.
        None => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "torch._C shim internal error: the Functionalize kernel declined {op} under _op_dk"
        ))),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_shim_to_functional, m)?)?;
    m.add_function(wrap_pyfunction!(_shim_from_functional, m)?)?;
    m.add_function(wrap_pyfunction!(_shim_is_functional, m)?)?;
    m.add_function(wrap_pyfunction!(_shim_functional_live, m)?)?;
    m.add_function(wrap_pyfunction!(_shim_functional_query, m)?)?;
    m.add_function(wrap_pyfunction!(_shim_functional_unsafe_set, m)?)?;
    m.add_function(wrap_pyfunction!(_shim_functional_view_steps, m)?)?;
    m.add_function(wrap_pyfunction!(_shim_functionalize_tls, m)?)?;
    m.add_function(wrap_pyfunction!(_shim_functionalize_counters, m)?)?;
    m.add_function(wrap_pyfunction!(_shim_set_functionalize_hooks, m)?)?;
    m.add_function(wrap_pyfunction!(_shim_functionalize_op_dk, m)?)?;
    Ok(())
}
