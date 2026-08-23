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
mod info;
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
    // BOOL.md §6.3 lists this function as one of the two ways the `torch.bool`
    // invariant could be broken quietly, since arbitrary values come in here.
    // It refuses the tag instead: this is scaffolding due for deletion, and it
    // should not be the thing that teaches the shim to lie about booleans.
    if dtype.is_some_and(|d| d.tag() == crate::dtype::TorchDType::Bool) {
        return Err(crate::err::not_implemented(
            "_tensor_from_flat: torch.bool is not accepted here -- a bool tensor \
             must come from an op that guarantees 0/1 bytes (BOOL.md §6.3)",
        ));
    }
    let target = dtype
        .map(|d| d.storage("_tensor_from_flat"))
        .transpose()?
        .unwrap_or(candle_core::DType::F32);
    let tensor = Tensor::from_vec(values, shape, &device)
        .and_then(|t| t.to_dtype(target))
        .map_err(|e| candle_err("_tensor_from_flat", e))?;
    Ok(PyTensorBase::new(tensor)?
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

/// The name surface the vendored tree expects `_C` to present, extracted from
/// the tree's own `.pyi` stubs by `vendor/gen_surface.py` and compiled in so
/// the artefact needs nothing on disk at runtime. See `bootstrap.py`.
const SURFACE: &str = include_str!("surface.json");

/// Everything that is a name rather than a behaviour is built in Python, from
/// `SURFACE`, at module init.
///
/// Why Python and not Rust: what has to be built is ~1,700 Python callables,
/// ~200 heap types with chosen metaclasses, and 27 entries in `sys.modules`.
/// All of that is dynamic Python object construction either way; doing it in
/// Rust would be the same operations spelled through `Bound<'_, PyAny>`, at
/// several times the length, with no more type safety, and it would still be
/// executing at exactly this moment. Keeping it in one readable file also
/// keeps the *reason* each name exists next to the name.
///
/// What stays in Rust is everything with behaviour: dtypes, devices, tensors,
/// and the aten dispatcher. `bootstrap.py` never computes anything -- it wires
/// names to the one door in `aten.rs`.
fn run_bootstrap(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    let code = std::ffi::CString::new(include_str!("bootstrap.py"))?;
    let boot = PyModule::from_code(
        py,
        code.as_c_str(),
        c"torch_c_bootstrap.py",
        c"_torch_c_bootstrap",
    )?;
    boot.getattr("install")?.call1((m, SURFACE))?;
    Ok(())
}

#[pymodule]
fn _C(m: &Bound<'_, PyModule>) -> PyResult<()> {
    dtype::register(m)?;
    device::register(m)?;
    info::register(m)?;
    tensor::register(m)?;
    aten::register(m)?;
    m.add_function(wrap_pyfunction!(_tensor_from_flat, m)?)?;
    m.add_function(wrap_pyfunction!(_shim_target, m)?)?;
    run_bootstrap(m)?;
    Ok(())
}
