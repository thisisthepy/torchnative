//! `torch._C.dtype`.
//!
//! In real PyTorch `torch.float32` is not a Python-level constant -- it is an
//! instance of a type defined in `_C`, and the vendored Python tree re-exports
//! it (`torch/__init__.py`). So the shim has to own the type, not just a name.
//!
//! The mapping is not a bijection. candle's `DType` is smaller than torch's in
//! some places (no `bool`, no signed 8-bit, no complex) and larger in others
//! (MX float formats torch has no name for). Only the pairs that mean the same
//! thing are listed here; everything else has to fail loudly rather than be
//! aliased onto a near neighbour, because a silently wrong dtype is exactly the
//! "수치 불일치가 조용히 번짐" risk DESIGN.md §5 names as A's main hazard.
use candle_core::DType;
use pyo3::prelude::*;
use pyo3::types::PyModule;

/// `torch.float32` and friends.
#[pyclass(name = "dtype", module = "torch._C", frozen, from_py_object)]
#[derive(Clone, Copy, PartialEq, Eq)]
pub struct PyDtype {
    inner: DType,
}

impl PyDtype {
    pub fn new(inner: DType) -> Self {
        Self { inner }
    }

    pub fn dtype(&self) -> DType {
        self.inner
    }

    /// The `torch.*` spelling. `None` when candle has a type torch does not
    /// name -- those must never be handed to Python under a borrowed name.
    pub fn torch_name(dtype: DType) -> Option<&'static str> {
        Some(match dtype {
            DType::F64 => "float64",
            DType::F32 => "float32",
            DType::F16 => "float16",
            DType::BF16 => "bfloat16",
            DType::I64 => "int64",
            DType::I32 => "int32",
            DType::I16 => "int16",
            DType::U8 => "uint8",
            DType::U32 => "uint32",
            DType::F8E4M3 => "float8_e4m3fn",
            _ => return None,
        })
    }
}

#[pymethods]
impl PyDtype {
    fn __repr__(&self) -> PyResult<String> {
        match Self::torch_name(self.inner) {
            Some(name) => Ok(format!("torch.{name}")),
            // Reachable only if a candle op produces an MX/F4 type; naming it
            // by its candle spelling is honest, inventing a torch name is not.
            None => Ok(format!("torch._C.dtype(candle:{})", self.inner.as_str())),
        }
    }

    fn __str__(&self) -> PyResult<String> {
        self.__repr__()
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.extract::<PyDtype>() {
            Ok(other) => self.inner == other.inner,
            Err(_) => false,
        }
    }

    fn __hash__(&self) -> u64 {
        self.inner as u64
    }

    /// torch spells this as a property on `torch.dtype`.
    #[getter]
    fn is_floating_point(&self) -> bool {
        self.inner.is_float()
    }

    #[getter]
    fn is_signed(&self) -> bool {
        !matches!(self.inner, DType::U8 | DType::U32)
    }

    #[getter]
    fn itemsize(&self) -> usize {
        self.inner.size_in_bytes()
    }
}

/// Registers the type and the module-level dtype instances.
pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDtype>()?;
    for dtype in [
        DType::F64,
        DType::F32,
        DType::F16,
        DType::BF16,
        DType::I64,
        DType::I32,
        DType::I16,
        DType::U8,
        DType::U32,
        DType::F8E4M3,
    ] {
        let name = PyDtype::torch_name(dtype).expect("listed dtypes all have torch names");
        m.add(name, PyDtype::new(dtype))?;
    }
    Ok(())
}
