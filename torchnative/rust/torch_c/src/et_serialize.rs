//! `torch._C._et_serialize_pte` and `torch._C._et_dataclass_json`: exir's
//! serializer step in Rust (issue #423, step R5-S1; owner 2026-11-21, "Rust로
//! 재구현"). The XNNPACK payload, which also needs the linked flatbuffers
//! parser, is `torch._C._et_serialize_xnnpack` in `et.rs`.
//!
//! The Python face is `torchnative/python/torchnative/export/et_serialize.py`,
//! which keeps exir's signatures and hooks them into exir.

use crate::et_pte::{self, NamedEntry, PteOptions};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

fn alignment(name: &str, v: Option<i64>) -> PyResult<Option<usize>> {
    match v {
        None => Ok(None),
        Some(x) if x <= 0 => Err(PyValueError::new_err(format!("Bad {name} {x}"))),
        Some(x) => Ok(Some(x as usize)),
    }
}

/// `torch._C._et_serialize_pte(program, mutable_data=None, named_buffers=None,
/// named_entries=None, *, extract_delegate_segments=False,
/// segment_alignment=128, constant_tensor_alignment=None,
/// delegate_alignment=None) -> bytes`.
///
/// `program` is exir's `Program` dataclass; `mutable_data` the storages of
/// `PTEFile.mutable_data` (bytes); `named_buffers` the
/// `NamedDataStoreOutput.buffers` and `named_entries` its `pte_data` as
/// `(key, buffer_index, alignment)` in dict order. The bytes are
/// `bytes(serialize_pte_binary(...))` of executorch 1.4.1.
#[pyfunction]
#[pyo3(name = "_et_serialize_pte", signature = (program, mutable_data = None, named_buffers = None, named_entries = None, *, extract_delegate_segments = false, segment_alignment = 128, constant_tensor_alignment = None, delegate_alignment = None))]
#[allow(clippy::too_many_arguments)]
fn et_serialize_pte(
    py: Python<'_>,
    program: &Bound<'_, PyAny>,
    mutable_data: Option<Vec<Bound<'_, PyAny>>>,
    named_buffers: Option<Vec<Bound<'_, PyAny>>>,
    named_entries: Option<Vec<(String, usize, Option<usize>)>>,
    extract_delegate_segments: bool,
    segment_alignment: i64,
    constant_tensor_alignment: Option<i64>,
    delegate_alignment: Option<i64>,
) -> PyResult<Py<PyBytes>> {
    if segment_alignment <= 0 {
        return Err(PyValueError::new_err(format!("Bad segment_alignment {segment_alignment}")));
    }
    let opt = PteOptions {
        extract_delegate_segments,
        segment_alignment: segment_alignment as usize,
        constant_tensor_alignment: alignment("constant_tensor_alignment", constant_tensor_alignment)?,
        delegate_alignment: alignment("delegate_alignment", delegate_alignment)?,
    };
    let mutable = match mutable_data {
        None => None,
        Some(v) => Some(v.iter().map(et_pte::coerce_bytes).collect::<PyResult<Vec<_>>>()?),
    };
    let named = match (named_buffers, named_entries) {
        (None, None) => None,
        (Some(b), Some(e)) => {
            let buffers = b.iter().map(et_pte::coerce_bytes).collect::<PyResult<Vec<_>>>()?;
            let entries = e
                .into_iter()
                .map(|(key, buffer_index, alignment)| NamedEntry { key, buffer_index, alignment: alignment.unwrap_or(1) })
                .collect();
            Some((buffers, entries))
        }
        _ => return Err(PyValueError::new_err("named_buffers and named_entries go together")),
    };
    let out = et_pte::serialize_pte(program, mutable, named, &opt)?;
    Ok(PyBytes::new(py, &out).unbind())
}

/// `torch._C._et_dataclass_json(obj) -> str`: the text of
/// `json.dumps(obj, cls=exir._serialize._dataclass._DataclassEncoder)`.
#[pyfunction]
#[pyo3(name = "_et_dataclass_json")]
fn et_dataclass_json(py: Python<'_>, obj: &Bound<'_, PyAny>) -> PyResult<String> {
    crate::et_json::dataclass_json(py, obj)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(et_serialize_pte, m)?)?;
    m.add_function(wrap_pyfunction!(et_dataclass_json, m)?)?;
    Ok(())
}
