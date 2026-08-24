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

#[pyclass(name = "TensorBase", module = "torch._C", subclass, from_py_object)]
#[derive(Clone)]
pub struct PyTensorBase {
    inner: Tensor,
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
            inner,
            tag,
            requires_grad: false,
        })
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
            inner,
            tag: TorchDType::Bool,
            requires_grad: false,
        })
    }

    pub fn tensor(&self) -> &Tensor {
        &self.inner
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
        PyTuple::new(py, self.inner.dims())
    }

    #[getter]
    fn dtype(&self) -> PyDtype {
        PyDtype::new(self.tag)
    }

    #[getter]
    fn device(&self) -> PyDevice {
        PyDevice::from_candle(self.inner.device())
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
        PyDevice::from_candle(self.inner.device()).kind == "meta"
    }

    #[getter]
    fn ndim(&self) -> usize {
        self.inner.rank()
    }

    fn dim(&self) -> usize {
        self.inner.rank()
    }

    fn numel(&self) -> usize {
        self.inner.elem_count()
    }

    #[pyo3(signature = (dim = None))]
    fn size<'py>(&self, py: Python<'py>, dim: Option<isize>) -> PyResult<Bound<'py, PyAny>> {
        match dim {
            None => Ok(self.shape(py)?.into_any()),
            Some(dim) => {
                let rank = self.inner.rank() as isize;
                let index = if dim < 0 { dim + rank } else { dim };
                if index < 0 || index >= rank {
                    return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                        "Dimension out of range (expected to be in range of [{}, {}], but got {dim})",
                        -rank,
                        rank - 1
                    )));
                }
                self.inner.dims()[index as usize].into_bound_py_any(py)
            }
        }
    }

    fn is_contiguous(&self) -> bool {
        self.inner.is_contiguous()
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
    fn tolist(&self, py: Python<'_>) -> PyResult<Py<PyAny>> {
        let flat = flat_objects(py, &self.inner, self.tag)?;
        nest(py, &flat, self.inner.dims())
    }

    fn __repr__(&self) -> String {
        format!(
            "TensorBase(shape={:?}, dtype={}, device={})",
            self.inner.dims(),
            self.tag.name(),
            PyDevice::from_candle(self.inner.device()).__str__()
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
