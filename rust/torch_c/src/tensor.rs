//! `torch._C.TensorBase` -- the candle tensor, seen from Python.
//!
//! The name is not arbitrary. The vendored Python tree's `torch/_tensor.py`
//! opens with `class Tensor(torch._C.TensorBase)`, and `torch._tensor` is one of
//! the ten modules IMPORT_WALLS §5 measured as actually executing during
//! inference. So `TensorBase` is the exact name the Python layer will subclass,
//! and getting it right now costs nothing while getting it wrong later means
//! patching a vendored file.
//!
//! What lives here is only the *identity* of a tensor -- shape, dtype, device.
//! Arithmetic does not: it goes through the aten dispatcher (`aten.rs`) so that
//! every operation passes the one choke point where an unimplemented op names
//! itself. A convenience method here would be a second, unmeasured entrance.
use candle_core::{DType, Tensor};
use pyo3::prelude::*;
use pyo3::types::{PyList, PyModule, PyTuple};
use pyo3::IntoPyObjectExt;

use crate::device::PyDevice;
use crate::dtype::{PyDtype, TorchDType};
use crate::err::{candle_err, not_implemented};

/// What a `TensorBase` is made of.
///
/// **The reason this is an enum is `meta`.** Upstream's meta tensor has shape,
/// dtype and stride and *no bytes* -- `torch.zeros(2, 3, device="meta")
/// .data_ptr()` is `0`, measured. candle has no such thing: every
/// `candle_core::Tensor` owns storage, and `Tensor::zeros` allocates. So a
/// meta tensor cannot be a candle tensor wearing a label; allocating and
/// calling it `meta` would invert the one property meta exists for.
/// docs/META.md §3.
///
/// The cost of the enum is paid once, at `tensor()`: it returns a `PyResult`
/// instead of a `&Tensor`, so **no kernel can read storage off a meta tensor
/// without handling the failure.** That is the point -- the refusal is
/// structural rather than a check each of the 96 kernels has to remember, which
/// is the same argument `check_devices_agree` makes for living at the door.
#[derive(Clone)]
pub enum Repr {
    Dense(Tensor),
    /// `meta`: shape and dtype, no storage.
    ///
    /// Stride is deliberately absent. `TensorBase` has no `.stride()` in this
    /// shim (dense tensors do not report one either), so modelling strides here
    /// would give meta a surface the dense side does not have. Recorded in
    /// docs/META.md §6 as a narrowing, since upstream's meta *does* carry
    /// stride (`torch.zeros(2,3,device="meta").t().stride()` is `(1, 3)`).
    ///
    /// The device label is not stored either, and that is measured rather than
    /// assumed: upstream normalises every meta index away --
    /// `torch.zeros(2, device="meta:7").device` is `device(type='meta')`, same
    /// as `meta:0` and bare `meta`. So there is exactly one meta device and the
    /// label is a constant. If a device kind ever arrives where the index
    /// *survives*, this is the field that has to appear, and
    /// docs/DEVICE_ABS.md §3.2 is the argument for it.
    Meta { shape: Vec<usize> },
}

#[pyclass(name = "TensorBase", module = "torch._C", subclass, from_py_object)]
pub struct PyTensorBase {
    inner: Repr,
    /// The torch-level dtype. Not derivable from `inner.dtype()`: `torch.bool`
    /// and `torch.uint8` share candle's `U8` storage and differ only here.
    /// BOOL.md §5-B is the decision; §6.3 is the invariant that comes with it
    /// (the bytes under a `bool` tag are 0 or 1), which is why `boolean()` is
    /// the only way to attach that tag.
    tag: TorchDType,
    /// **Inert.** Stored and reported, read by nothing. There is no autograd
    /// here (DESIGN.md §3 stage 0), so no graph node is ever created from it
    /// and `grad_fn` is always `None`. It exists because `from_config` calls
    /// `TensorBase.requires_grad_` before it calls anything else interesting,
    /// and the alternative was stopping there. `backward()` stays a raising
    /// stub so that code which really depends on the flag meaning something
    /// fails by name rather than silently getting nothing. Recorded as a
    /// papered-over item in docs/TENSORBASE.md, not as an implementation.
    requires_grad: bool,
    /// **Inert**, for the same reason as `requires_grad`: there is no autograd
    /// here, so nothing ever fires a backward hook. It is a real slot rather
    /// than a refusing property because `torch/_utils.py:246
    /// _rebuild_tensor_v2` *assigns* to it on every tensor of every
    /// `torch.load`, unconditionally and before anyone could have registered a
    /// hook -- so on this path the value written is always the empty
    /// `OrderedDict()` that `_rebuild_tensor_v2`'s own comment insists on
    /// ("we must give an EMPTY OrderedDict(), if you pass a None you'll run
    /// afoul #12219"). Refusing the assignment stopped `torch.load`; accepting
    /// it stores something nothing reads. Recorded in docs/CKPT.md §6 as
    /// papered over, not implemented.
    backward_hooks: Option<Py<PyAny>>,
}

/// Hand-written rather than derived: `backward_hooks` is a `Py<PyAny>`, and
/// incrementing a Python refcount needs the interpreter attached, which
/// `#[derive(Clone)]` has no way to ask for.
impl Clone for PyTensorBase {
    fn clone(&self) -> Self {
        Python::attach(|py| Self {
            inner: self.inner.clone(),
            tag: self.tag,
            requires_grad: self.requires_grad,
            backward_hooks: self.backward_hooks.as_ref().map(|h| h.clone_ref(py)),
        })
    }
}

/// The refusal every read of a meta tensor's bytes ends at.
///
/// Upstream's own wording, character for character, measured on torch 2.13.0:
/// `torch.zeros(2, device="meta").tolist()`, `.cpu()`, `.to("cpu")` and
/// `torch.zeros(2).copy_(meta)` all raise
/// `NotImplementedError: Cannot copy out of meta tensor; no data!`.
pub fn no_data() -> PyErr {
    pyo3::exceptions::PyNotImplementedError::new_err("Cannot copy out of meta tensor; no data!")
}

impl PyTensorBase {
    /// A tensor whose torch dtype is whatever candle is already storing.
    pub fn new(inner: Tensor) -> PyResult<Self> {
        let tag = TorchDType::from_storage(inner.dtype()).ok_or_else(|| {
            not_implemented(format!(
                "torch._C shim has no torch dtype for candle dtype: {}",
                inner.dtype().as_str()
            ))
        })?;
        Ok(Self {
            inner: Repr::Dense(inner),
            tag,
            requires_grad: false,
            backward_hooks: None,
        })
    }

    /// A tensor on the `meta` device: shape and dtype, no allocation.
    ///
    /// Unlike `new`/`boolean` there is no dtype narrowing to do -- `meta`
    /// carries the torch tag directly and never has to be storable by candle.
    /// That is a real widening over the dense side: `torch.empty(2,
    /// dtype=torch.complex64, device="meta")` is representable here while its
    /// CPU counterpart is not, which is also true upstream on a build without a
    /// kernel for a dtype. docs/META.md §6.
    pub fn meta(shape: Vec<usize>, tag: TorchDType) -> Self {
        Self {
            inner: Repr::Meta { shape },
            tag,
            requires_grad: false,
            backward_hooks: None,
        }
    }

    /// The single entrance for the `torch.bool` tag (BOOL.md §6.3 item 1).
    /// The caller is asserting the bytes are already normalised to 0/1;
    /// `BRAINWAVE_CHECK_BOOL=1` turns that assertion into a check.
    pub fn boolean(inner: Tensor) -> PyResult<Self> {
        if inner.dtype() != DType::U8 {
            return Err(not_implemented(format!(
                "torch._C shim: a torch.bool tensor stores as U8, got {}",
                inner.dtype().as_str()
            )));
        }
        if std::env::var("BRAINWAVE_CHECK_BOOL").is_ok_and(|v| v != "0") {
            let max = inner
                .flatten_all()
                .and_then(|t| t.max(0))
                .and_then(|t| t.to_scalar::<u8>())
                .map_err(|e| candle_err("bool invariant check", e))?;
            if max > 1 {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "torch._C shim: torch.bool invariant violated -- a byte under                      a bool tag is {max}, not 0 or 1 (BOOL.md §6.3)"
                )));
            }
        }
        Ok(Self {
            inner: Repr::Dense(inner),
            tag: TorchDType::Bool,
            requires_grad: false,
            backward_hooks: None,
        })
    }

    /// The storage. **Refuses on `meta`**, which has none.
    ///
    /// Every kernel in `aten.rs` reads its inputs through this, so the enum's
    /// one `?` is what makes "no kernel computes on a meta tensor" a property
    /// of the type rather than a rule 96 kernels have to remember. The door's
    /// meta gate (`aten.rs::check_meta`) refuses first and with a better
    /// message; this is the backstop under it, and it is the reason a kernel
    /// added tomorrow without being told about meta is safe.
    #[inline]
    pub fn tensor(&self) -> PyResult<&Tensor> {
        match &self.inner {
            Repr::Dense(tensor) => Ok(tensor),
            Repr::Meta { .. } => Err(no_data()),
        }
    }

    #[inline]
    pub fn repr(&self) -> &Repr {
        &self.inner
    }

    #[inline]
    pub fn is_meta_repr(&self) -> bool {
        matches!(self.inner, Repr::Meta { .. })
    }

    /// The shape, for either representation. This is the half of a tensor meta
    /// still has.
    #[inline]
    pub fn dims(&self) -> &[usize] {
        match &self.inner {
            Repr::Dense(tensor) => tensor.dims(),
            Repr::Meta { shape } => shape,
        }
    }

    #[inline]
    pub fn elem_count(&self) -> usize {
        match &self.inner {
            Repr::Dense(tensor) => tensor.elem_count(),
            Repr::Meta { shape } => shape.iter().product(),
        }
    }

    /// The device label, for either representation.
    ///
    /// For a dense tensor this is `PyDevice::from_candle`, with the lossy
    /// index reconstruction that documents. For a meta tensor there is no
    /// candle handle to reconstruct from and none is needed: `meta` is a
    /// constant, because upstream normalises every index off it (measured --
    /// `device="meta:7"` reports `device(type='meta')`).
    pub fn device_label(&self) -> PyDevice {
        match &self.inner {
            Repr::Dense(tensor) => PyDevice::from_candle(tensor.device()),
            Repr::Meta { .. } => PyDevice::meta(),
        }
    }

    pub fn tag(&self) -> TorchDType {
        self.tag
    }

    /// The write half of the in-place ops (`fill_`, `copy_`, `uniform_`,
    /// `normal_`).
    ///
    /// It takes a whole `PyTensorBase` rather than a bare candle tensor so
    /// that the replacement has already been through `new()` or `boolean()` --
    /// BOOL.md §6.3's rule that only one constructor may attach the `bool`
    /// tag survives a mutating API only if the mutation cannot bypass it.
    ///
    /// **This replaces the wrapper's tensor; it does not write into storage.**
    /// The difference is visible: upstream's `y = x.detach(); y.fill_(0)`
    /// writes through to `x` because the two share storage, and here it does
    /// not, because `y` is a different wrapper. Mutating through the *same*
    /// Python object (`p.data.fill_(0)`, since `.data` returns `self`) does
    /// behave like upstream. Recorded in docs/TENSORBASE.md.
    pub fn replace_with(&mut self, replacement: PyTensorBase) {
        self.inner = replacement.inner;
        self.tag = replacement.tag;
    }
}

/// A tensor from a little-endian payload, under a torch dtype tag.
///
/// The one place raw checkpoint bytes become a tensor. `torch.frombuffer` (the
/// safetensors path) and `TensorBase.set_` (the `torch.load` path) both come
/// here, so the two readers cannot disagree about a dtype.
///
/// Endianness is not checked, and cannot be from here: the caller has the
/// container's byte-order field (safetensors is little-endian by
/// specification; `torch.save` writes a `byteorder` record and
/// `torch/serialization.py` byteswaps before this point). Every target this
/// crate builds for is little-endian.
pub fn from_le_bytes(
    op: &str,
    bytes: &[u8],
    shape: &[usize],
    tag: TorchDType,
) -> PyResult<PyTensorBase> {
    let device = candle_core::Device::Cpu;

    // `torch.bool`: candle stores it as `U8`, and the 0/1 invariant
    // (BOOL.md §6.3) has to hold by construction. Raw checkpoint bytes under a
    // bool tag are not guaranteed normalised, so they are reduced with `!= 0`
    // -- which is exactly what torch guarantees a bool tensor *reads* as
    // (BOOL.md §2.6), so no value changes. Same reduction `_tensor_from_flat`
    // makes, for the same reason.
    if tag == TorchDType::Bool {
        let normalised: Vec<u8> = bytes.iter().map(|b| u8::from(*b != 0)).collect();
        let tensor =
            Tensor::from_vec(normalised, shape.to_vec(), &device).map_err(|e| candle_err(op, e))?;
        return PyTensorBase::boolean(tensor);
    }

    // `storage()` refuses by name for the dtypes candle cannot hold -- `int8`,
    // `uint16`, `uint64`, the complex family, and the fourteen sub-byte integer
    // tags. Upstream accepts several of them (measured: `torch.frombuffer` with
    // `torch.int8` and `torch.uint16` both return tensors), so this is a real
    // narrowing of the surface, and refusing loudly is the point. A checkpoint
    // in one of those dtypes stops here with the dtype in the message instead
    // of being reinterpreted as something else.
    let storage = tag.storage().ok_or_else(|| {
        not_implemented(format!(
            "{op}: dtype not storable by the candle backend in torch._C shim: torch.{}",
            tag.name()
        ))
    })?;
    let tensor =
        Tensor::from_raw_buffer(bytes, storage, shape, &device).map_err(|e| candle_err(op, e))?;
    PyTensorBase::new(tensor)
}

/// The little-endian payload of a tensor, row-major -- `from_le_bytes`'s inverse.
///
/// The pair is used by `aten::view.dtype`, which reinterprets one dtype's bytes
/// as another's. Going out through bytes and back in through `from_le_bytes`
/// rather than converting element-wise is what makes the reinterpretation
/// *exact*: `view` is a bit-level operation, and any route that passed through
/// a numeric type would round.
///
/// Written as the mirror of `from_le_bytes` on purpose, including the dtype
/// list -- the two are only usable together if they agree about which tags this
/// build can hold, and `storage()` is the single place that decides.
///
/// `torch.bool` is emitted as the 0/1 bytes candle stores, which is what
/// upstream's bool storage holds too (BOOL.md §2.6): measured, a `bool` tensor
/// viewed as `uint8` upstream gives exactly 0s and 1s.
///
/// **The half-width floats are refused here, and the asymmetry is deliberate.**
/// Getting at an `f16`/`bf16` tensor's bits through candle means naming the
/// `half` crate's types, and this crate does not depend on `half` -- it arrives
/// only as candle's own dependency, and CANDLE_DEPS.md is about not acquiring
/// dependencies by accident. The refusal costs nothing on the path that
/// exists: a checkpoint arrives as *bytes*, so `bf16` weights go
/// `uint8 -> bf16` through `from_le_bytes`, which reads a raw buffer and never
/// asks this function anything. Only the reverse direction --
/// `bf16_tensor.view(torch.uint8)` -- is closed, and nothing has reached it.
pub fn to_le_bytes(op: &str, tensor: &Tensor) -> PyResult<Vec<u8>> {
    let flat = tensor
        .flatten_all()
        .and_then(|t| t.contiguous())
        .map_err(|e| candle_err(op, e))?;
    macro_rules! pour {
        ($ty:ty) => {{
            let v = flat.to_vec1::<$ty>().map_err(|e| candle_err(op, e))?;
            let mut out = Vec::with_capacity(v.len() * std::mem::size_of::<$ty>());
            for x in v {
                out.extend_from_slice(&x.to_le_bytes());
            }
            out
        }};
    }
    Ok(match flat.dtype() {
        DType::U8 => flat.to_vec1::<u8>().map_err(|e| candle_err(op, e))?,
        DType::U32 => pour!(u32),
        DType::I16 => pour!(i16),
        DType::I32 => pour!(i32),
        DType::I64 => pour!(i64),
        DType::F32 => pour!(f32),
        DType::F64 => pour!(f64),
        other => {
            return Err(not_implemented(format!(
                "{op}: torch._C shim cannot read the raw bytes of candle dtype \
                 {other:?} -- reaching its bit pattern means naming the `half` \
                 crate's types, which this crate deliberately does not depend on \
                 (see tensor.rs::to_le_bytes). Reading a checkpoint *into* this \
                 dtype goes the other way, through `from_le_bytes`, and works"
            )))
        }
    })
}

/// The contiguous (row-major) stride for a shape, in elements.
fn contiguous_stride(size: &[usize]) -> Vec<i64> {
    let mut stride = vec![1i64; size.len()];
    for i in (0..size.len().saturating_sub(1)).rev() {
        stride[i] = stride[i + 1] * size[i + 1] as i64;
    }
    stride
}

/// Read a `(storage_offset, size, stride)` view out of a byte buffer into a
/// contiguous, row-major copy of it.
///
/// This is what makes `TensorBase.set_` correct for a *view*, and views are not
/// exotic in a checkpoint: `torch.save` records each tensor's stride and offset
/// as it finds them, so a `state_dict` that holds `w.t()`, or a slice of a
/// larger buffer, arrives here non-contiguous. A reader that took the first
/// `numel` elements instead would return the transpose's storage read in the
/// wrong order -- right dtype, right shape, wrong numbers, no error. Measured:
/// a `4x3` tensor saved as `base.t()` of a `3x4`.
///
/// The walk is over element indices rather than bytes so it is dtype-agnostic;
/// `itemsize` only decides how wide each copied element is. The contiguous case
/// is not special-cased into a memcpy because `from_le_bytes` is already the
/// hot path's cost and the gather is a single pass either way -- one code path
/// means the contiguous case cannot pass while the strided case rots.
fn gather_strided(
    op: &str,
    bytes: &[u8],
    storage_offset: usize,
    size: &[usize],
    stride: &[i64],
    itemsize: usize,
    numel: usize,
) -> PyResult<Vec<u8>> {
    let mut out = Vec::with_capacity(numel * itemsize);
    let mut index = vec![0usize; size.len()];

    for _ in 0..numel {
        let mut element = storage_offset;
        for (d, i) in index.iter().enumerate() {
            element += i * stride[d] as usize;
        }
        let start = element * itemsize;
        let end = start + itemsize;
        if end > bytes.len() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{op}: the view (offset {storage_offset}, size {size:?}, stride \
                 {stride:?}) reaches byte {end} of a storage holding {}",
                bytes.len()
            )));
        }
        out.extend_from_slice(&bytes[start..end]);

        // Odometer, last dimension fastest -- row-major, which is the order
        // `from_le_bytes` will read the result back in.
        for d in (0..size.len()).rev() {
            index[d] += 1;
            if index[d] < size[d] {
                break;
            }
            index[d] = 0;
        }
    }
    Ok(out)
}

/// The Python class an op result should wear.
///
/// Upstream, `_C` never hands back a bare `TensorBase`: `THPVariable_Wrap`
/// instantiates `THPVariableClass`, which C++ resolves to `torch._tensor.Tensor`
/// -- the Python subclass. The vendored tree depends on that in ways that have
/// no workaround, and `torch/nn/parameter.py:54` is the sharpest:
///
/// ```python
/// if type(data) is torch.Tensor or type(data) is Parameter:
///     return torch.Tensor._make_subclass(cls, data, requires_grad)
/// # otherwise: the custom-tensor path, which returns something that is not
/// # a Parameter, and `nn.Module.__setattr__` then classifies it as a plain
/// # attribute instead of registering it. A model built that way has no
/// # parameters at all.
/// ```
///
/// So the class is registered here, at the same moment upstream registers it
/// (`_initExtension`, after `from torch._tensor import Tensor` has run), and
/// `promote` puts every op result into it.
///
/// When nothing has registered a class -- which is exactly the standalone
/// `_C` that `tools/golden/loader.py` imports, with no `torch` package around
/// it -- `promote` is the identity and results stay `TensorBase`.
static TENSOR_CLASS: std::sync::OnceLock<Py<PyAny>> = std::sync::OnceLock::new();

#[pyfunction]
#[pyo3(name = "_set_tensor_class")]
pub fn set_tensor_class(cls: Py<PyAny>) {
    // First registration wins; `_initExtension` is idempotent upstream too.
    let _ = TENSOR_CLASS.set(cls);
}

/// Wrap a bare `TensorBase` in the registered Python tensor class.
///
/// Idempotent and narrow on purpose: anything that is already an instance of a
/// *subclass* (a `Tensor`, a `Parameter`) is returned untouched, so an
/// in-place op still hands back the object it mutated.
pub fn promote(py: Python<'_>, value: Py<PyAny>) -> PyResult<Py<PyAny>> {
    let cls = match TENSOR_CLASS.get() {
        Some(cls) => cls,
        None => return Ok(value),
    };
    let bound = value.bind(py);
    if bound.get_type().is(&py.get_type::<PyTensorBase>()) {
        return Ok(cls.bind(py).call1((bound,))?.unbind());
    }
    Ok(value)
}

#[pymethods]
impl PyTensorBase {
    /// `TensorBase(other)` -- the constructor `promote` calls.
    ///
    /// PyO3's generated `tp_new` allocates with the *subtype* it was called
    /// with, so `Tensor(base)` produces a `Tensor` and `Parameter(base)` a
    /// `Parameter`, each sharing the candle tensor. That is what makes
    /// `_make_subclass` a three-line Python function in `bootstrap.py` rather
    /// than another piece of native machinery.
    ///
    /// It takes a tensor and nothing else. Upstream's `torch.Tensor(2, 3)`
    /// (the legacy uninitialised-storage constructor) is a different function
    /// wearing the same name, is not used anywhere on the inference path, and
    /// is refused by name rather than guessed at.
    #[new]
    fn py_new(data: &Bound<'_, PyAny>) -> PyResult<Self> {
        data.extract::<Self>().map_err(|_| {
            not_implemented(format!(
                "torch._C shim: TensorBase(...) takes an existing tensor to re-wrap; \
                 upstream's legacy `torch.Tensor({})` storage constructor is not \
                 implemented",
                data.get_type().name().map(|n| n.to_string()).unwrap_or_default()
            ))
        })
    }

    /// torch returns `torch.Size`, itself a C-defined tuple subclass. The shim
    /// does not own that type yet, so this is a plain tuple -- structurally
    /// compatible, and the difference is recorded in docs/TORCH_C.md.
    #[getter]
    fn shape<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyTuple>> {
        PyTuple::new(py, self.dims())
    }

    #[getter]
    fn dtype(&self) -> PyDtype {
        PyDtype::new(self.tag)
    }

    #[getter]
    fn device(&self) -> PyDevice {
        self.device_label()
    }

    /// `tensor.is_meta`. Derived from the device rather than returned as a
    /// constant `False`, so it stays true to whatever `device()` reports if a
    /// meta-like device ever appears; today candle has three device kinds
    /// (`Cpu`, `Cuda`, `Metal`, see `PyDevice::from_candle`) and none of them
    /// is `meta`, so it always answers `False`.
    ///
    /// It is here because `Module.load_state_dict` reads it on every single
    /// parameter -- `torch/nn/modules/module.py:2449`, `if param.is_meta:`,
    /// before the shape check and before the copy. It was the only wall left on
    /// that path once the weights themselves could be read (docs/CKPT.md).
    /// A stub property raising by name stopped `load_state_dict` outright,
    /// which is the right behaviour for a hole and the wrong answer for a
    /// question the shim can answer.
    #[getter]
    fn is_meta(&self) -> bool {
        self.device_label().kind == "meta"
    }

    /// `tensor.is_cpu` / `tensor.is_cuda`. Same shape as `is_meta` and derived
    /// the same way, so all three agree with `device` by construction rather
    /// than by three separate constants that could drift apart.
    ///
    /// `is_cuda` in particular is read as a *guard* all over the vendored tree
    /// (`if t.is_cuda:` before a CUDA-only path), so answering it is what keeps
    /// those paths from being entered; a stub that raised turned a branch
    /// upstream never takes into a hard stop.
    #[getter]
    fn is_cpu(&self) -> bool {
        self.device_label().kind == "cpu"
    }

    #[getter]
    fn is_cuda(&self) -> bool {
        self.device_label().kind == "cuda"
    }

    /// `tensor.is_mps` / `is_xpu` / `is_maia`, the three remaining device
    /// predicates `torch/_tensor_str.py` reads before it prints anything.
    ///
    /// Same derivation as `is_cpu`/`is_cuda`/`is_meta` above, and here for the
    /// same reason those are: `_tensor_str.py:121-123` branches on all three
    /// (`tensor_totype` picks `float` over `double` on MPS and on Maia, and
    /// asks `xpu.get_device_properties(...).has_fp64` on XPU), and a raising
    /// stub made `print(tensor)` a hard stop on a line upstream skips.
    ///
    /// They answer `False` today because `PyDevice::from_candle` reports only
    /// `cpu`, `cuda` and `metal` -- but that is a fact about the device layer
    /// rather than a constant written here, which is the difference this
    /// spelling preserves. Note `mps` in particular: `torch.device("mps")` is
    /// a *constructible* label in this shim (there is a smoke test for it), so
    /// this predicate is one working backend away from answering `True` on its
    /// own.
    #[getter]
    fn is_mps(&self) -> bool {
        self.device_label().kind == "mps"
    }

    #[getter]
    fn is_xpu(&self) -> bool {
        self.device_label().kind == "xpu"
    }

    #[getter]
    fn is_maia(&self) -> bool {
        self.device_label().kind == "maia"
    }

    /// The five "which representation is this?" predicates, and `layout`.
    ///
    /// `torch/_tensor_str.py` reads every one of them before it prints a
    /// number: `is_nested` picks the `nested_tensor(` prefix, `is_sparse` and
    /// `layout` pick the COO and the compressed-sparse formatters,
    /// `is_quantized` adds the `quantization_scheme=` suffix, `_is_zerotensor`
    /// forces a `clone` and `is_neg` forces a `resolve_neg`.
    ///
    /// **They are an exhaustive `match` over `Repr`, not a `false`.** That is
    /// the whole of the argument for them, and it is deliberately structural:
    /// this shim has exactly two representations, candle's dense strided
    /// buffer and the shape-and-dtype-only `Meta` arm, and neither is nested,
    /// sparse, quantised, a `ZeroTensor` or a negative-bit view. Writing it as
    /// a match means a third arm cannot be added to `Repr` without the
    /// compiler asking what these six answer for it -- where a bare `false`
    /// would inherit silently, which is exactly the shape of the `is_mutable`
    /// accident in docs/DISTRIBUTED.md §8.1.
    ///
    /// The other half of the argument is in `pytests/test_shim.py`
    /// (`test_the_alternative_representations_have_no_constructors`): each of
    /// these representations has exactly one way into existence and every one
    /// of those ways refuses by name, so `False` is derivable from the
    /// constructor set rather than asserted. If any of them ever lands, that
    /// test fails and these stop being answerable this way.
    #[getter]
    fn is_nested(&self) -> bool {
        match self.inner {
            Repr::Dense(_) => false,
            Repr::Meta { .. } => false,
        }
    }

    #[getter]
    fn is_sparse(&self) -> bool {
        match self.inner {
            Repr::Dense(_) => false,
            Repr::Meta { .. } => false,
        }
    }

    #[getter]
    fn is_quantized(&self) -> bool {
        match self.inner {
            Repr::Dense(_) => false,
            Repr::Meta { .. } => false,
        }
    }

    /// A *method* upstream, not a property -- `_tensor_str.py:336` spells it
    /// `self._is_zerotensor()`. Getting that wrong is not a subtle failure
    /// (`'bool' object is not callable` is what a property gives), but it is
    /// the kind that is only visible by running the tree.
    fn _is_zerotensor(&self) -> bool {
        match self.inner {
            Repr::Dense(_) => false,
            Repr::Meta { .. } => false,
        }
    }

    /// Also a method (`_tensor_str.py:341`). The negative bit is set only by
    /// `torch._neg_view`; `neg()` materialises here exactly as it does
    /// upstream, so a negated tensor is a new buffer rather than a view
    /// wearing a flag.
    fn is_neg(&self) -> bool {
        match self.inner {
            Repr::Dense(_) => false,
            Repr::Meta { .. } => false,
        }
    }

    /// The name of `tensor.layout`, resolved to the `torch.layout` object in
    /// `bootstrap.py`.
    ///
    /// Split that way on purpose: the *fact* is about the representation and
    /// belongs beside the other five, while the `torch.strided` object is
    /// synthesised in Python (`_install_namespace_types`) and cannot be
    /// constructed here. `bootstrap.py` refuses by name for any string it does
    /// not recognise, so a new arm cannot leak through as `None`.
    fn _layout_name(&self) -> &'static str {
        match self.inner {
            // candle's tensors carry a stride and are dense; there is no
            // sparse or compressed storage anywhere in this build.
            Repr::Dense(_) => "strided",
            // Upstream's meta tensors are strided too -- `torch.zeros(2, 3,
            // device="meta").layout` is `torch.strided`, measured.
            Repr::Meta { .. } => "strided",
        }
    }

    // `tensor.get_device()` is *not* here, and the reason is a PyO3 collision
    // rather than a decision: `#[pymethods]` derives the slot name
    // `__pymethod_get_device__` from the `device` getter above and from a
    // method named `get_device` alike, and the crate is built without
    // `multiple-pymethods`, so the two cannot coexist in this block. It is
    // installed from `bootstrap.py` instead, where it is one line over the
    // `device` property this file already exposes.

    #[getter]
    fn ndim(&self) -> usize {
        self.dims().len()
    }

    fn dim(&self) -> usize {
        self.dims().len()
    }

    #[getter]
    fn _backward_hooks(&self) -> Option<&Py<PyAny>> {
        self.backward_hooks.as_ref()
    }

    #[setter]
    fn set__backward_hooks(&mut self, value: Option<Py<PyAny>>) {
        self.backward_hooks = value;
    }

    /// `tensor.element_size()` -- bytes per element, from the torch dtype tag
    /// and not from candle's, so `torch.bool` answers 1 rather than borrowing
    /// `uint8`'s answer by accident. (They agree; the point is that the tag is
    /// the authority, per BOOL.md §5-B.)
    fn element_size(&self) -> usize {
        self.tag.itemsize()
    }

    /// `tensor.set_(storage, storage_offset, size, stride)` -- **a copy, where
    /// upstream aliases.**
    ///
    /// The only caller that matters is `torch/_utils.py:198 _rebuild_tensor`,
    /// which every `torch.load` of every tensor goes through:
    ///
    /// ```python
    /// t = torch.empty((0,), dtype=storage.dtype, device=...)
    /// return t.set_(storage._untyped_storage, storage_offset, size, stride)
    /// ```
    ///
    /// Upstream this makes `t` a *view* of the storage. Here candle owns its
    /// memory, so the bytes are copied out and the tensor is independent of the
    /// storage afterwards. Two consequences, both refused rather than papered
    /// over, because either one produces silently wrong weights:
    ///
    /// 1. **The storage must already hold its payload.** See storage.rs -- the
    ///    legacy `torch.save` format fills storages *after* `_rebuild_tensor`
    ///    runs, and a copying `set_` there yields a checkpoint of zeros with no
    ///    error anywhere. Measured. So an unfilled storage is refused by name.
    ///
    /// 2. **The result is contiguous even when the view was not.** A saved
    ///    tensor may be a strided view of its storage -- `w.t()`, or a slice of
    ///    a larger buffer -- and `gather_strided` reads it in row-major order
    ///    into a fresh buffer. Upstream would have kept the view; here the copy
    ///    holds the same numbers with contiguous stride. That is visible to
    ///    anything that reads `.stride()`, and it is why `set_` cannot be used
    ///    to build an aliasing view on purpose.
    #[pyo3(signature = (source, storage_offset = 0, size = None, stride = None))]
    fn set_<'py>(
        slf: &Bound<'py, Self>,
        source: &Bound<'py, PyAny>,
        storage_offset: usize,
        size: Option<Vec<usize>>,
        stride: Option<Vec<i64>>,
    ) -> PyResult<Bound<'py, Self>> {
        const OP: &str = "TensorBase.set_";

        let storage: PyRef<'_, crate::storage::PyStorageBase> =
            source.extract().map_err(|_| {
                let got = source
                    .get_type()
                    .name()
                    .map(|n| n.to_string())
                    .unwrap_or_else(|_| "?".to_string());
                not_implemented(format!(
                    "{OP}: expected a torch.UntypedStorage, got {got} -- the \
                     no-argument and tensor-argument spellings of set_ are not \
                     implemented in this shim"
                ))
            })?;

        let size = size.ok_or_else(|| {
            not_implemented(format!(
                "{OP}(storage) without an explicit size is not implemented -- it \
                 would mean adopting the storage's whole extent, which is the \
                 aliasing behaviour this shim does not have"
            ))
        })?;

        let tag = slf.borrow().tag;
        let itemsize = tag.itemsize();
        let numel: usize = size.iter().product();

        if numel > 0 && !storage.is_filled() {
            return Err(not_implemented(format!(
                "{OP}: the storage has never been filled. This shim's set_ copies \
                 out of the storage instead of aliasing it, so a tensor built \
                 from an empty storage would be silently zero. The caller must \
                 deliver the bytes before set_, not after (see storage.rs and \
                 docs/CKPT.md §4)."
            )));
        }

        let stride = stride.unwrap_or_else(|| contiguous_stride(&size));
        if stride.len() != size.len() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{OP}: size {size:?} has {} dimensions but stride {stride:?} has {}",
                size.len(),
                stride.len()
            )));
        }
        if stride.iter().any(|s| *s < 0) {
            return Err(not_implemented(format!(
                "{OP}: negative stride in {stride:?}. torch itself does not produce \
                 these, so this is refused rather than guessed at."
            )));
        }

        let bytes = storage.bytes();
        let gathered = gather_strided(
            OP,
            bytes,
            storage_offset,
            &size,
            &stride,
            itemsize,
            numel,
        )?;

        let replacement = from_le_bytes(OP, &gathered, &size, tag)?;
        drop(storage);
        slf.borrow_mut().replace_with(replacement);
        Ok(slf.clone())
    }

    /// `tensor.data = other`, the write half of the `.data` property that
    /// `bootstrap.py` installs.
    ///
    /// **This is the last wall on `nn.Module.to(device)`.** `Module._apply`
    /// converts a parameter, asks `_has_compatible_shallow_copy_type` whether
    /// the result can take the old one's place, and on `True` -- which is what
    /// upstream answers for two dense tensors -- assigns
    /// `param.data = param_applied` (`torch/nn/modules/module.py:995`). With no
    /// setter that assignment is an `AttributeError`, so every `.to()`,
    /// `.cpu()`, `.float()` and `.half()` on a module died there.
    ///
    /// It is `replace_with`, which means it inherits that method's recorded
    /// divergence: the wrapper starts pointing at a different candle tensor
    /// rather than the storage being rewritten, so a view taken *before* the
    /// assignment does not follow it. Upstream's `.data =` swaps the TensorImpl
    /// too, so pre-existing views do not follow there either -- for this
    /// spelling the two agree, and docs/OPS4.md §8's open aliasing question is
    /// about writes through views, which this is not. docs/DEVICE_ABS.md §4.
    ///
    /// `requires_grad` is deliberately left alone: upstream's `.data =` does
    /// not touch it, and `_apply` relies on that to keep a `Parameter` a
    /// parameter.
    fn _shim_set_data(slf: &Bound<'_, Self>, value: PyTensorBase) {
        slf.borrow_mut().replace_with(value);
    }

    fn numel(&self) -> usize {
        PyTensorBase::elem_count(self)
    }

    #[pyo3(signature = (dim = None))]
    fn size<'py>(&self, py: Python<'py>, dim: Option<isize>) -> PyResult<Bound<'py, PyAny>> {
        match dim {
            None => Ok(self.shape(py)?.into_any()),
            Some(dim) => {
                let rank = self.dims().len() as isize;
                let index = if dim < 0 { dim + rank } else { dim };
                if index < 0 || index >= rank {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "Dimension out of range (expected to be in range of [{}, {}], but got {dim})",
                        -rank,
                        rank - 1
                    )));
                }
                self.dims()[index as usize].into_bound_py_any(py)
            }
        }
    }

    /// `tensor.is_contiguous()`.
    ///
    /// A meta tensor answers `True` unconditionally, and that is a narrowing
    /// rather than an answer: upstream tracks stride on meta and
    /// `torch.zeros(2,3,device="meta").t().is_contiguous()` is `False`. Here
    /// `Repr::Meta` carries no stride (see its comment) and no meta kernel can
    /// produce a transposed one, so every meta tensor this shim can make *is*
    /// contiguous. It stops being true the day a meta `t`/`permute` kernel
    /// lands, and that kernel is the thing that has to add the stride field.
    fn is_contiguous(&self) -> bool {
        match &self.inner {
            Repr::Dense(tensor) => tensor.is_contiguous(),
            Repr::Meta { .. } => true,
        }
    }

    /// See the field comment: stored, reported, read by nothing.
    #[getter]
    fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    #[setter]
    fn set_requires_grad(&mut self, value: bool) {
        self.requires_grad = value;
    }

    /// Nested Python lists, as `torch.Tensor.tolist` gives. This is the only
    /// way to read values out at the moment, so tests can compare numbers
    /// against real torch without any further surface being built first.
    ///
    /// On `meta` it raises upstream's own `NotImplementedError: Cannot copy
    /// out of meta tensor; no data!` -- the same refusal, from the same place,
    /// as `.cpu()` and `.to("cpu")`, because it is the same question.
    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let flat = flat_objects(py, self.tensor()?, self.tag)?;
        nest(py, &flat, self.dims())
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorBase(shape={:?}, dtype={}, device={})",
            self.dims(),
            self.tag.name(),
            self.device_label().__str__()
        )
    }
}

/// Flattens to Python scalars. Every float type is read through `f64` and every
/// integer type through `i64`, so a dtype candle can hold but this shim has not
/// taught itself to read fails by name instead of returning garbage.
fn flat_objects(py: Python<'_>, tensor: &Tensor, tag: TorchDType) -> PyResult<Vec<Py<PyAny>>> {
    let flat = tensor.flatten_all().map_err(|e| candle_err("tolist", e))?;
    let dtype = tensor.dtype();
    if tag == TorchDType::Bool {
        // torch's `tolist` on a bool tensor yields Python `bool`s, not 0/1
        // ints. Reading `!= 0` rather than the raw byte is also what torch
        // guarantees (BOOL.md §2.6) -- the tag promises the read, not the byte.
        let values = flat
            .to_vec1::<u8>()
            .map_err(|e| candle_err("tolist", e))?;
        return values
            .into_iter()
            .map(|v| (v != 0).into_py_any(py))
            .collect::<PyResult<Vec<_>>>();
    }
    if dtype.is_float() {
        let values = flat
            .to_dtype(DType::F64)
            .and_then(|t| t.to_vec1::<f64>())
            .map_err(|e| candle_err("tolist", e))?;
        values
            .into_iter()
            .map(|v| v.into_py_any(py))
            .collect::<PyResult<Vec<_>>>()
    } else if dtype.is_int() {
        let values = flat
            .to_dtype(DType::I64)
            .and_then(|t| t.to_vec1::<i64>())
            .map_err(|e| candle_err("tolist", e))?;
        values
            .into_iter()
            .map(|v| v.into_py_any(py))
            .collect::<PyResult<Vec<_>>>()
    } else {
        Err(not_implemented(format!(
            "tolist not implemented in torch._C shim for dtype: {}",
            dtype.as_str()
        )))
    }
}

fn nest(py: Python<'_>, flat: &[Py<PyAny>], dims: &[usize]) -> PyResult<Py<PyAny>> {
    match dims.split_first() {
        // 0-d tensor: torch's tolist returns the bare scalar.
        None => Ok(flat[0].clone_ref(py)),
        Some((&outer, rest)) => {
            let stride: usize = rest.iter().product();
            let mut items = Vec::with_capacity(outer);
            for i in 0..outer {
                items.push(nest(py, &flat[i * stride..(i + 1) * stride], rest)?);
            }
            Ok(PyList::new(py, items)?.into_any().unbind())
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTensorBase>()?;
    m.add_function(wrap_pyfunction!(set_tensor_class, m)?)?;
    Ok(())
}
