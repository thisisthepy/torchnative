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
use candle_core::{DType, Device, Tensor};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule, PyTuple};

use crate::device::PyDevice;
use crate::dtype::PyDtype;
use crate::err::{aten_not_implemented, candle_err, not_implemented};
use crate::tensor::PyTensorBase;

/// Every op with a real kernel behind it. Kept sorted; `_aten_implemented()`
/// hands it to Python so the vendored layer and the tests can ask rather than
/// keep their own copy of the list.
pub const IMPLEMENTED: &[&str] = &["aten.add.Tensor", "aten.full.default", "aten.mm.default"];

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
    match op {
        "aten.add.Tensor" => add_tensor(py, args, kwargs),
        "aten.full.default" => full_default(py, args, kwargs),
        "aten.mm.default" => mm_default(py, args, kwargs),
        other => Err(aten_not_implemented(other)),
    }
}

#[pyfunction]
#[pyo3(name = "_aten_implemented")]
pub fn aten_implemented() -> Vec<&'static str> {
    IMPLEMENTED.to_vec()
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
    let fill_is_int = fill.is_instance_of::<pyo3::types::PyInt>();
    let dtype = match optional(args, kwargs, 2, "dtype")? {
        Some(value) if !value.is_none() => value.extract::<PyDtype>()?.dtype(),
        _ => {
            if fill_is_int {
                DType::I64
            } else {
                DType::F32
            }
        }
    };

    reject_unsupported(OP, args, kwargs, &[(3, "layout"), (5, "pin_memory")])?;
    let device = device_arg(args, kwargs, 4, "device")?;

    let tensor = if dtype.is_int() {
        let value: i64 = fill.extract()?;
        Tensor::full(value, size, &device)
    } else {
        let value: f64 = fill.extract()?;
        Tensor::full(value, size, &device)
    }
    .and_then(|t| t.to_dtype(dtype))
    .map_err(|e| candle_err(OP, e))?;

    Ok(PyTensorBase::new(tensor).into_pyobject(py)?.into_any().unbind())
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

    same_dtype(OP, &lhs, &rhs)?;

    let rhs = if alpha == 1.0 {
        rhs
    } else {
        rhs.affine(alpha, 0.0).map_err(|e| candle_err(OP, e))?
    };
    let out = lhs.broadcast_add(&rhs).map_err(|e| candle_err(OP, e))?;

    Ok(PyTensorBase::new(out).into_pyobject(py)?.into_any().unbind())
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
    if lhs.rank() != 2 || rhs.rank() != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{OP}: both arguments to mm need to be 2D, but they are {}D and {}D",
            lhs.rank(),
            rhs.rank()
        )));
    }
    same_dtype(OP, &lhs, &rhs)?;

    let out = lhs.matmul(&rhs).map_err(|e| candle_err(OP, e))?;
    Ok(PyTensorBase::new(out).into_pyobject(py)?.into_any().unbind())
}

// ---------------------------------------------------------------------------
// Argument plumbing
//
// aten schemas allow positional or keyword for everything before the `*`, and
// the vendored Python layer will use both spellings, so each argument is looked
// up by index and by name.
// ---------------------------------------------------------------------------

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
) -> PyResult<Tensor> {
    let value = required(op, args, kwargs, index, name)?;
    let base = value.extract::<PyTensorBase>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{op}: argument '{name}' must be a torch._C.TensorBase, got {}",
            value.get_type().name().map(|n| n.to_string()).unwrap_or_default()
        ))
    })?;
    Ok(base.tensor().clone())
}

fn device_arg(
    args: &Bound<'_, PyTuple>,
    kwargs: Option<&Bound<'_, PyDict>>,
    index: usize,
    name: &str,
) -> PyResult<Device> {
    match optional(args, kwargs, index, name)? {
        Some(value) if !value.is_none() => {
            let device = match value.extract::<PyDevice>() {
                Ok(device) => device,
                // torch accepts a plain string wherever a device is taken.
                Err(_) => PyDevice::new(&value.extract::<String>()?, None)?,
            };
            device.resolve()
        }
        _ => Ok(Device::Cpu),
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

/// torch would promote here. The shim does not, and says so by name.
fn same_dtype(op: &str, lhs: &Tensor, rhs: &Tensor) -> PyResult<()> {
    if lhs.dtype() != rhs.dtype() {
        return Err(not_implemented(format!(
            "{op}: dtype promotion not implemented in torch._C shim: {} vs {}",
            lhs.dtype().as_str(),
            rhs.dtype().as_str()
        )));
    }
    Ok(())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aten_dispatch, m)?)?;
    m.add_function(wrap_pyfunction!(aten_implemented, m)?)?;
    Ok(())
}
