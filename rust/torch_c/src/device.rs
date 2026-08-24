//! `torch._C.device`.
//!
//! Deliberately *not* a wrapper around `candle_core::Device`. In torch,
//! `torch.device("cuda")` is constructible on a CPU-only build -- it is a label,
//! and only using it fails. candle's `Device` is the opposite: the enum variant
//! carries a live handle, so it cannot represent a device this build has no
//! backend for. Storing the label and resolving it on use keeps torch's
//! semantics and makes the failure land where torch puts it.
use candle_core::Device;
use pyo3::prelude::*;
use pyo3::types::PyModule;

use crate::err::not_implemented;

#[pyclass(name = "device", module = "torch._C", frozen, from_py_object)]
#[derive(Clone)]
pub struct PyDevice {
    #[pyo3(get, name = "type")]
    pub kind: String,
    #[pyo3(get)]
    pub index: Option<i64>,
}

impl PyDevice {
    pub fn cpu() -> Self {
        Self {
            kind: "cpu".to_string(),
            index: None,
        }
    }

    /// The backend this label resolves to. Only CPU exists today; Metal and
    /// CUDA are feature-gated off in `Cargo.toml` on purpose (device builds
    /// must not link them), so asking for one is a loud failure, not a silent
    /// fallback to CPU.
    pub fn resolve(&self) -> PyResult<Device> {
        match self.kind.as_str() {
            "cpu" => Ok(Device::Cpu),
            other => Err(not_implemented(format!(
                "device not available in torch._C shim: {other}"
            ))),
        }
    }

    pub fn from_candle(device: &Device) -> Self {
        match device {
            Device::Cpu => Self::cpu(),
            Device::Cuda(_) => Self {
                kind: "cuda".to_string(),
                index: Some(0),
            },
            Device::Metal(_) => Self {
                kind: "mps".to_string(),
                index: Some(0),
            },
        }
    }
}

#[pymethods]
impl PyDevice {
    /// `torch.device("cpu")` / `torch.device("cuda", 0)`.
    #[new]
    #[pyo3(signature = (device, index = None))]
    pub fn new(device: &str, index: Option<i64>) -> PyResult<Self> {
        let (kind, parsed_index) = match device.split_once(':') {
            Some((kind, idx)) => {
                let idx: i64 = idx.parse().map_err(|_| {
                    pyo3::exceptions::PyRuntimeError::new_err(format!(
                        "Invalid device string: '{device}'"
                    ))
                })?;
                (kind.to_string(), Some(idx))
            }
            None => (device.to_string(), None),
        };
        if index.is_some() && parsed_index.is_some() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "type (string) must not include an index because index was passed explicitly",
            ));
        }
        Ok(Self {
            kind,
            index: index.or(parsed_index),
        })
    }

    fn __repr__(&self) -> String {
        match self.index {
            Some(index) => format!("device(type='{}', index={})", self.kind, index),
            None => format!("device(type='{}')", self.kind),
        }
    }

    pub fn __str__(&self) -> String {
        match self.index {
            Some(index) => format!("{}:{}", self.kind, index),
            None => self.kind.clone(),
        }
    }

    fn __eq__(&self, other: &Bound<'_, PyAny>) -> bool {
        match other.extract::<PyDevice>() {
            Ok(other) => self.kind == other.kind && self.index == other.index,
            Err(_) => false,
        }
    }

    fn __hash__(&self) -> u64 {
        let mut hash = 0u64;
        for byte in self.kind.as_bytes() {
            hash = hash.wrapping_mul(31).wrapping_add(*byte as u64);
        }
        hash.wrapping_mul(31)
            .wrapping_add(self.index.unwrap_or(-1) as u64)
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyDevice>()?;
    Ok(())
}
