//! `torch._C` -- the one native piece of PyTorch this project replaces.
//!
//! DESIGN.md §2: PyTorch is mostly Python. `nn/modules/*.py`, `nn/functional.py`,
//! `_tensor.py`, `_decomp/` are all vendored from upstream under BSD; the only
//! part that is native, and therefore the only part that has to be rebuilt, is
//! `_C` -- ATen tensors, the dispatcher, autograd. §5 settled how: candle under
//! a PyO3 adapter (option A), because option B's mobile CMake path force-sets
//! `BUILD_PYTHON=OFF` and so structurally cannot produce a `torch._C` at all.
//!
//! Layout:
//!
//! | module | what |
//! |---|---|
//! | `tensor` | `TensorBase` -- identity of a tensor: shape, dtype, device |
//! | `dtype` | `torch.float32` and friends, as `_C`-owned instances |
//! | `device` | `torch.device`, a label rather than a live backend handle |
//! | `aten` | the single dispatch entrance, and the ops behind it |
//! | `err` | the message shapes; §6's discovery mechanism lives on these |
//!
//! This is a floor, not a coverage effort. Three ops are implemented. Everything
//! else raises with its own name, so running a model produces the work queue by
//! itself, in frequency order (§6). Details in docs/TORCH_C.md.
// The module must be named `_C` -- that is the name Python imports. rustc's
// snake-case lint has no opinion worth honouring here.
#![allow(non_snake_case)]

use candle_core::Tensor;
use pyo3::prelude::*;
use pyo3::types::PyModule;

mod aten;
mod device;
mod dtype;
mod err;
mod tensor;

use crate::device::PyDevice;
use crate::dtype::PyDtype;
use crate::err::candle_err;
use crate::tensor::PyTensorBase;

/// Scaffolding, not torch. There is no aten op that takes a Python list of
/// numbers -- `torch.tensor(...)` is a Python-layer factory that lowers to
/// `lift_fresh`/`_to_copy`, and `lift_fresh` is one of the two ops CORE_ATEN §0
/// found to be neither Core ATen nor covered by the decomposition table. Until
/// that is decided, tests need *some* way to get real data in, so this exists
/// with a leading underscore and no aten name, and is expected to be deleted
/// rather than promoted.
#[pyfunction]
#[pyo3(signature = (values, shape, dtype = None, device = None))]
fn _tensor_from_flat(
    py: Python<'_>,
    values: Vec<f64>,
    shape: Vec<usize>,
    dtype: Option<PyDtype>,
    device: Option<PyDevice>,
) -> PyResult<Py<PyAny>> {
    let expected: usize = shape.iter().product();
    if values.len() != expected {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "_tensor_from_flat: shape {shape:?} needs {expected} values, got {}",
            values.len()
        )));
    }
    let device = device.unwrap_or_else(PyDevice::cpu).resolve()?;
    let target = dtype.map(|d| d.dtype()).unwrap_or(candle_core::DType::F32);
    let tensor = Tensor::from_vec(values, shape, &device)
        .and_then(|t| t.to_dtype(target))
        .map_err(|e| candle_err("_tensor_from_flat", e))?;
    Ok(PyTensorBase::new(tensor)
        .into_pyobject(py)?
        .into_any()
        .unbind())
}

/// The triple this artefact was built for. Three targets are cross-compiled and
/// the results are indistinguishable once renamed to `_C.so`, so the build
/// records it here rather than leaving it to be guessed from a file path.
#[pyfunction]
fn _shim_target() -> &'static str {
    env!("TORCH_C_TARGET")
}

#[pymodule]
fn _C(m: &Bound<'_, PyModule>) -> PyResult<()> {
    dtype::register(m)?;
    device::register(m)?;
    tensor::register(m)?;
    aten::register(m)?;
    m.add_function(wrap_pyfunction!(_tensor_from_flat, m)?)?;
    m.add_function(wrap_pyfunction!(_shim_target, m)?)?;
    Ok(())
}
