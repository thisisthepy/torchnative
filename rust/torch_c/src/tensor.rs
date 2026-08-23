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
        Ok(Self { inner, tag })
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
        })
    }

    pub fn tensor(&self) -> &Tensor {
        &self.inner
    }

    pub fn tag(&self) -> TorchDType {
        self.tag
    }
}

#[pymethods]
impl PyTensorBase {
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
    Ok(())
}
