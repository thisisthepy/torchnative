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
mod rng;
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
    // invariant could be broken quietly, since arbitrary `f64`s come in here.
    // It used to refuse the tag outright. It now *normalises* instead, for the
    // reason docs/OVERLOAD.md §6.7 gave when `_tensor_new_from_data` was added:
    // the invariant is kept by construction, not by hope, as long as every byte
    // under a `bool` tag goes through `PyTensorBase::boolean` after being
    // reduced to 0/1. `!= 0` is that reduction, and it is also what torch
    // guarantees a bool tensor *reads* as (BOOL.md §2.6).
    //
    // The refusal had become load-bearing in the wrong direction:
    // `tools/golden/build.py` builds every operand through this function, so a
    // mask could not be built at all, and `masked_fill` and `index.Tensor`
    // -- both of which take a `torch.bool` argument -- could not be compared
    // against upstream. An op that cannot be golden-checked is worse than a
    // constructor that normalises.
    if dtype.is_some_and(|d| d.tag() == crate::dtype::TorchDType::Bool) {
        let bytes: Vec<u8> = values.iter().map(|v| u8::from(*v != 0.0)).collect();
        let tensor = Tensor::from_vec(bytes, shape, &device)
            .map_err(|e| candle_err("_tensor_from_flat", e))?;
        return crate::tensor::promote(
            py,
            PyTensorBase::boolean(tensor)?.into_pyobject(py)?.into_any().unbind(),
        );
    }
    let target = dtype
        .map(|d| d.storage("_tensor_from_flat"))
        .transpose()?
        .unwrap_or(candle_core::DType::F32);
    let tensor = Tensor::from_vec(values, shape, &device)
        .and_then(|t| t.to_dtype(target))
        .map_err(|e| candle_err("_tensor_from_flat", e))?;
    crate::tensor::promote(
        py,
        PyTensorBase::new(tensor)?.into_pyobject(py)?.into_any().unbind(),
    )
}

/// One element of a nested Python sequence handed to `torch.tensor(...)`,
/// kept in the category it arrived as -- that is what decides the dtype.
enum Leaf {
    Bool(bool),
    Int(i64),
    Float(f64),
}

fn sequence_items<'py>(value: &Bound<'py, PyAny>) -> Option<Vec<Bound<'py, PyAny>>> {
    if let Ok(list) = value.cast::<pyo3::types::PyList>() {
        return Some(list.iter().collect());
    }
    if let Ok(tuple) = value.cast::<pyo3::types::PyTuple>() {
        return Some(tuple.iter().collect());
    }
    None
}

fn ragged() -> PyErr {
    pyo3::exceptions::PyValueError::new_err(
        "torch._C shim: torch.tensor() got a ragged nested sequence",
    )
}

fn walk_data(
    value: &Bound<'_, PyAny>,
    depth: usize,
    shape: &mut Vec<usize>,
    leaf_depth: &mut Option<usize>,
    out: &mut Vec<Leaf>,
) -> PyResult<()> {
    if let Some(items) = sequence_items(value) {
        if shape.len() == depth {
            shape.push(items.len());
        } else if shape[depth] != items.len() {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected sequence of length {} at dim {} (got {})",
                shape[depth],
                depth,
                items.len()
            )));
        }
        for item in items {
            walk_data(&item, depth + 1, shape, leaf_depth, out)?;
        }
        return Ok(());
    }
    // Every leaf has to sit at the same depth. Counting elements is not
    // enough: `[[1], 2]` walks to shape `[2, 1]` and produces exactly the two
    // values that shape asks for, so the count check passes and the answer is
    // wrong.
    match leaf_depth {
        Some(seen) if *seen != depth => return Err(ragged()),
        _ => *leaf_depth = Some(depth),
    }
    // `bool` before `int`: it is a subclass, and the difference is the whole
    // point -- `torch.tensor([True])` is `torch.bool`, `torch.tensor([1])` is
    // `torch.int64`.
    if value.is_instance_of::<pyo3::types::PyBool>() {
        out.push(Leaf::Bool(value.extract()?));
    } else if value.is_instance_of::<pyo3::types::PyInt>() {
        out.push(Leaf::Int(value.extract()?));
    } else if value.is_instance_of::<pyo3::types::PyFloat>() {
        out.push(Leaf::Float(value.extract()?));
    } else {
        return Err(pyo3::exceptions::PyTypeError::new_err(format!(
            "torch._C shim: torch.tensor() takes nested sequences of Python \
             bool/int/float, and got {}",
            value.get_type().name().map(|n| n.to_string()).unwrap_or_default()
        )));
    }
    Ok(())
}

/// The storage half of `torch.tensor(...)`.
///
/// Upstream's `torch.tensor` is not an aten op either: it is
/// `THPVariable_tensor` -> `internal_new_from_data`, a `_C` function the
/// dispatcher never sees, and the only aten record a real
/// `torch.tensor([1, 2])` produces is `aten.lift_fresh.default` (measured with
/// a `TorchDispatchMode` logger). So this mirrors upstream's shape rather than
/// inventing an `aten::tensor` call -- `bootstrap.py` builds the data here and
/// then passes the result through `lift_fresh`, which *is* dispatched.
///
/// Distinct from `_tensor_from_flat`, which stays what it is: scaffolding due
/// for deletion that takes an already-flat `f64` list and refuses `torch.bool`
/// outright (BOOL.md §6.3). This one has to accept booleans, because
/// `torch.tensor([True, False])` is how a mask is written -- and it does so
/// through `PyTensorBase::boolean`, the one tagging constructor, so the 0/1
/// invariant holds by construction here too.
#[pyfunction]
#[pyo3(signature = (data, dtype = None, device = None))]
fn _tensor_new_from_data(
    py: Python<'_>,
    data: &Bound<'_, PyAny>,
    dtype: Option<PyDtype>,
    device: Option<PyDevice>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "torch.tensor";

    let mut shape: Vec<usize> = Vec::new();
    let mut leaves: Vec<Leaf> = Vec::new();
    let mut leaf_depth: Option<usize> = None;
    walk_data(data, 0, &mut shape, &mut leaf_depth, &mut leaves)?;
    // An empty list has no leaves and a legal shape of `[0]`; anything with
    // leaves must have them exactly at the bottom of the shape it declared.
    let expected: usize = shape.iter().product();
    if leaf_depth.is_some_and(|d| d != shape.len()) || leaves.len() != expected {
        return Err(ragged());
    }

    // torch's inference, in category order: all booleans give `torch.bool`,
    // all integers give `int64`, anything with a float in it gives the default
    // float dtype. An empty list gives the default float dtype.
    let all_bool = !leaves.is_empty() && leaves.iter().all(|l| matches!(l, Leaf::Bool(_)));
    let any_float = leaves.iter().any(|l| matches!(l, Leaf::Float(_)));
    let inferred = if all_bool {
        crate::dtype::TorchDType::Bool
    } else if any_float || leaves.is_empty() {
        crate::aten::DEFAULT_FLOAT
    } else {
        crate::dtype::TorchDType::Int64
    };
    let tag = dtype.map(|d| d.tag()).unwrap_or(inferred);
    let storage = PyDtype::new(tag).storage(OP)?;
    let device = device.unwrap_or_else(PyDevice::cpu).resolve()?;

    let tensor = if tag == crate::dtype::TorchDType::Bool {
        let bytes: Vec<u8> = leaves
            .iter()
            .map(|leaf| {
                u8::from(match leaf {
                    Leaf::Bool(v) => *v,
                    Leaf::Int(v) => *v != 0,
                    Leaf::Float(v) => *v != 0.0,
                })
            })
            .collect();
        Tensor::from_vec(bytes, shape, &device)
    } else if storage.is_int() {
        let values: Vec<i64> = leaves
            .iter()
            .map(|leaf| match leaf {
                Leaf::Bool(v) => i64::from(*v),
                Leaf::Int(v) => *v,
                Leaf::Float(v) => *v as i64,
            })
            .collect();
        Tensor::from_vec(values, shape, &device).and_then(|t| t.to_dtype(storage))
    } else {
        let values: Vec<f64> = leaves
            .iter()
            .map(|leaf| match leaf {
                Leaf::Bool(v) => f64::from(u8::from(*v)),
                Leaf::Int(v) => *v as f64,
                Leaf::Float(v) => *v,
            })
            .collect();
        Tensor::from_vec(values, shape, &device).and_then(|t| t.to_dtype(storage))
    }
    .map_err(|e| candle_err(OP, e))?;

    let wrapped = if tag == crate::dtype::TorchDType::Bool {
        PyTensorBase::boolean(tensor)?
    } else {
        PyTensorBase::new(tensor)?
    };
    crate::tensor::promote(py, wrapped.into_pyobject(py)?.into_any().unbind())
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

/// The signature list `torch.<op>(...)` resolves against, per op, in order.
/// Unlike `SURFACE` this is not generated from the vendored tree -- the tree
/// carries aten overload *names* and Python-level signatures but nothing that
/// joins them (docs/OVERLOAD.md §2) -- so it is transcribed and checked by
/// `pytests/verify_schemas.py` against an installed upstream torch. Compiled
/// in the same way: nothing is read from disk at runtime.
const OVERLOADS: &str = include_str!("overloads.json");

/// The same, for `tensor.<method>(...)`. A separate table because upstream has
/// a separate binding: `THPVariable_mul` (a `TensorBase` method) and the
/// `_VariableFunctions` entry are different C functions with different
/// signature lists, which is why docs/C_SURFACE.md counted the two surfaces
/// apart -- 50 `TensorBase` members called against 13 hoisted functions.
/// Checked by the same `pytests/verify_schemas.py`.
const METHODS: &str = include_str!("methods.json");

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
    boot.getattr("install")?.call1((m, SURFACE, OVERLOADS, METHODS))?;
    Ok(())
}

#[pymodule]
fn _C(m: &Bound<'_, PyModule>) -> PyResult<()> {
    dtype::register(m)?;
    device::register(m)?;
    info::register(m)?;
    tensor::register(m)?;
    aten::register(m)?;
    rng::register(m)?;
    m.add_function(wrap_pyfunction!(_tensor_from_flat, m)?)?;
    m.add_function(wrap_pyfunction!(_tensor_new_from_data, m)?)?;
    m.add_function(wrap_pyfunction!(_shim_target, m)?)?;
    run_bootstrap(m)?;
    Ok(())
}
