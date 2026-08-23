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
use candle_core::{Device, Tensor};
use pyo3::prelude::*;
use pyo3::PyErr;
use pyo3::types::{PyDict, PyModule, PyTuple};

use crate::device::PyDevice;
use crate::dtype::{PyDtype, TorchDType};
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
    // `bool` subclasses `int` in Python, so the bool test has to come first.
    // torch gives `torch.full((2,), True)` dtype `torch.bool`; before the
    // dtype tag existed this branch was unreachable and the shim handed back
    // `int64` (docs/TORCH_C.md §2 recorded it as an open item).
    let fill_is_bool = fill.is_instance_of::<pyo3::types::PyBool>();
    let fill_is_int = fill.is_instance_of::<pyo3::types::PyInt>();
    let dtype = match optional(args, kwargs, 2, "dtype")? {
        Some(value) if !value.is_none() => value.extract::<PyDtype>()?.tag(),
        _ if fill_is_bool => TorchDType::Bool,
        _ if fill_is_int => TorchDType::Int64,
        _ => TorchDType::Float32,
    };

    reject_unsupported(OP, args, kwargs, &[(3, "layout"), (5, "pin_memory")])?;
    let device = device_arg(args, kwargs, 4, "device")?;

    // torch refuses a fill value the target dtype cannot hold. candle would
    // wrap (int) or saturate (float) instead, which is the silent divergence
    // the golden harness caught -- see `checked_convert` below.
    checked_convert(&fill, fill_is_int, dtype, size.iter().product())?;
    let storage = PyDtype::new(dtype).storage(OP)?;

    if dtype == TorchDType::Bool {
        // Normalised on the way in, which is what makes the tag's invariant
        // hold by construction rather than by hope (BOOL.md §6.3).
        let truthy = fill.is_truthy()?;
        let tensor = Tensor::full(u8::from(truthy), size, &device)
            .map_err(|e| candle_err(OP, e))?;
        return Ok(PyTensorBase::boolean(tensor)?
            .into_pyobject(py)?
            .into_any()
            .unbind());
    }

    let tensor = if storage.is_int() {
        let value: i64 = fill.extract()?;
        Tensor::full(value, size, &device)
    } else {
        let value: f64 = fill.extract()?;
        Tensor::full(value, size, &device)
    }
    .and_then(|t| t.to_dtype(storage))
    .map_err(|e| candle_err(OP, e))?;

    Ok(PyTensorBase::new(tensor)?.into_pyobject(py)?.into_any().unbind())
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

    let tag = same_dtype(OP, &lhs, &rhs)?;
    // `bool + bool` is a logical or in torch, not an arithmetic sum
    // (BOOL.md §2.2). candle's `broadcast_add` would give 2 where both are
    // true, which is still truthy and therefore silently wrong downstream --
    // so this refuses rather than approximates.
    if tag == TorchDType::Bool {
        return Err(not_implemented(format!(
            "{OP}: torch.bool addition is logical or, not arithmetic, and is              not implemented in torch._C shim"
        )));
    }

    let (lhs, rhs) = (lhs.tensor().clone(), rhs.tensor().clone());
    let rhs = if alpha == 1.0 {
        rhs
    } else {
        rhs.affine(alpha, 0.0).map_err(|e| candle_err(OP, e))?
    };
    let out = lhs.broadcast_add(&rhs).map_err(|e| candle_err(OP, e))?;

    Ok(PyTensorBase::new(out)?.into_pyobject(py)?.into_any().unbind())
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
    if lhs.tensor().rank() != 2 || rhs.tensor().rank() != 2 {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{OP}: both arguments to mm need to be 2D, but they are {}D and {}D",
            lhs.tensor().rank(),
            rhs.tensor().rank()
        )));
    }
    same_dtype(OP, &lhs, &rhs)?;

    let out = lhs
        .tensor()
        .matmul(rhs.tensor())
        .map_err(|e| candle_err(OP, e))?;
    Ok(PyTensorBase::new(out)?.into_pyobject(py)?.into_any().unbind())
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
) -> PyResult<PyTensorBase> {
    let value = required(op, args, kwargs, index, name)?;
    // The wrapper, not the bare candle tensor: the torch dtype tag lives on
    // the wrapper and `torch.bool` is invisible in the candle dtype.
    value.extract::<PyTensorBase>().map_err(|_| {
        pyo3::exceptions::PyTypeError::new_err(format!(
            "{op}: argument '{name}' must be a torch._C.TensorBase, got {}",
            value.get_type().name().map(|n| n.to_string()).unwrap_or_default()
        ))
    })
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

// ---------------------------------------------------------------------------
// Scalar -> dtype conversion, as torch does it
//
// `c10::checked_convert` refuses a scalar the destination dtype cannot hold.
// candle has no such check: an out-of-range integer wraps (two's complement)
// and an out-of-range float saturates to infinity, both without a word. The
// golden harness caught exactly that pair -- `full([3], 1e6, float16)` gave
// `inf` where torch raises, and `full([3], 2**31, int32)` gave `-2**31` where
// torch raises -- so the check is transcribed here rather than approximated.
//
// The rules below are measured against torch 2.13.0, not read off the C++.
// Two of them are counter-intuitive enough that guessing would have got them
// wrong, and both are load-bearing for cases the harness already passes:
//
//   * A negative value converted to an *unsigned* dtype is allowed to wrap,
//     as long as its magnitude fits. `full(-1, uint8) == 255` is legal in
//     torch; `full(-300, uint8)` is not. (c10 spells this "allow for negative
//     numbers to wrap using two's complement arithmetic".)
//   * Half / BFloat16 / Float8 skip the check entirely when the tensor has
//     exactly one element, and saturate silently instead. That is an upstream
//     inconsistency -- `fill_` takes a CPU numel==1 fast path whose conversion
//     is unchecked -- but `full([], 1e6, float16) == inf` while
//     `full([3], 1e6, float16)` raises, on real torch, so a shim that always
//     refuses would diverge from torch in the other direction.
// ---------------------------------------------------------------------------

/// The name torch puts in the message. These are C++ type spellings, so they
/// are not derivable from the torch dtype name -- each one was read off a real
/// `RuntimeError` from torch 2.13.0.
fn c10_name(dtype: TorchDType) -> &'static str {
    use TorchDType::*;
    match dtype {
        Float64 => "double",
        Float32 => "float",
        Float16 => "c10::Half",
        BFloat16 => "c10::BFloat16",
        Int64 => "int64_t",
        Int32 => "int",
        Int16 => "int16_t",
        Int8 => "int8_t",
        UInt8 => "uint8_t",
        UInt16 => "uint16_t",
        UInt32 => "uint32_t",
        UInt64 => "uint64_t",
        Bool => "bool",
        Float8E4M3FN => "c10::Float8_e4m3fn",
        Float8E5M2 => "c10::Float8_e5m2",
        other => other.name(),
    }
}

fn overflow(dtype: TorchDType) -> PyErr {
    // torch's own wording, with no shim prefix: this is torch semantics being
    // reproduced, not a shim limitation, and a caller matching on the message
    // should not have to know which of the two produced it.
    pyo3::exceptions::PyRuntimeError::new_err(format!(
        "value cannot be converted to type {} without overflow",
        c10_name(dtype)
    ))
}

/// Largest finite magnitude of a floating dtype; `None` for the others, which
/// are range-checked exactly rather than through `f64`.
fn float_max(dtype: TorchDType) -> Option<f64> {
    use TorchDType::*;
    Some(match dtype {
        Float64 => f64::MAX,
        Float32 => f32::MAX as f64,
        Float16 => 65504.0,
        BFloat16 => 3.3895313892515355e38,
        Float8E4M3FN | Float8E4M3FNUZ => 448.0,
        Float8E5M2 | Float8E5M2FNUZ => 57344.0,
        _ => return None,
    })
}

/// Inclusive integer range, as `(min, max)`. `min` is negative for the signed
/// types and zero for the unsigned ones -- the wrap allowance is applied by
/// the caller, not folded in here, so the two rules stay separable.
fn int_range(dtype: TorchDType) -> Option<(i64, i64)> {
    use TorchDType::*;
    Some(match dtype {
        Int64 => (i64::MIN, i64::MAX),
        Int32 => (i32::MIN as i64, i32::MAX as i64),
        Int16 => (i16::MIN as i64, i16::MAX as i64),
        Int8 => (i8::MIN as i64, i8::MAX as i64),
        UInt8 => (0, u8::MAX as i64),
        UInt16 => (0, u16::MAX as i64),
        UInt32 => (0, u32::MAX as i64),
        UInt64 => (0, i64::MAX),
        _ => return None,
    })
}

fn checked_convert(
    fill: &Bound<'_, PyAny>,
    fill_is_int: bool,
    dtype: TorchDType,
    numel: usize,
) -> PyResult<()> {
    // Every value is a valid `bool`; torch converts by truthiness and never
    // reports an overflow for it.
    if dtype == TorchDType::Bool {
        return Ok(());
    }

    // The upstream numel==1 hole, reproduced deliberately. Restricted to the
    // reduced-precision float types because torch's own fast path checks
    // float/double/int even at one element -- measured, not assumed.
    let unchecked_at_one = matches!(
        dtype,
        TorchDType::Float16
            | TorchDType::BFloat16
            | TorchDType::Float8E4M3FN
            | TorchDType::Float8E4M3FNUZ
            | TorchDType::Float8E5M2
            | TorchDType::Float8E5M2FNUZ
    );
    if numel == 1 && unchecked_at_one {
        return Ok(());
    }

    if fill_is_int {
        // A Python int too large for `i64` raises OverflowError here, which is
        // what torch does too ("int too big to convert") -- from the same
        // place, its own Python-to-Scalar conversion.
        let value: i64 = fill.extract()?;
        if let Some((min, max)) = int_range(dtype) {
            let fits = if value < 0 && min == 0 {
                // Two's-complement wrap, allowed for magnitude <= max.
                value.checked_neg().map(|m| m <= max).unwrap_or(false)
            } else {
                value >= min && value <= max
            };
            if !fits {
                return Err(overflow(dtype));
            }
        } else if let Some(max) = float_max(dtype) {
            if (value as f64).abs() > max {
                return Err(overflow(dtype));
            }
        }
    } else {
        let value: f64 = fill.extract()?;
        if let Some((min, max)) = int_range(dtype) {
            // Integer dtypes have neither infinity nor NaN, so both are
            // refused; a finite value must land inside the range.
            if value.is_nan() || value < min as f64 || value > max as f64 {
                return Err(overflow(dtype));
            }
        } else if let Some(max) = float_max(dtype) {
            // Infinity converts to infinity; only finite-but-too-large is an
            // overflow. NaN is fine -- the float types all have a quiet NaN.
            if value.is_finite() && value.abs() > max {
                return Err(overflow(dtype));
            }
        }
    }
    Ok(())
}

/// torch would promote here. The shim does not, and says so by name. Compares
/// the *torch* dtype, so `bool` and `uint8` are not accidentally the same
/// operand type just because candle stores both as `U8`.
fn same_dtype(op: &str, lhs: &PyTensorBase, rhs: &PyTensorBase) -> PyResult<TorchDType> {
    if lhs.tag() != rhs.tag() {
        return Err(not_implemented(format!(
            "{op}: dtype promotion not implemented in torch._C shim: {} vs {}",
            lhs.tag().name(),
            rhs.tag().name()
        )));
    }
    Ok(lhs.tag())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(aten_dispatch, m)?)?;
    m.add_function(wrap_pyfunction!(aten_implemented, m)?)?;
    Ok(())
}
