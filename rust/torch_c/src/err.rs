//! Error surface.
//!
//! DESIGN.md §6 -- "발견은 shim 이 스스로 한다". Everything the shim cannot do
//! must name itself in the exception text, because that text is the work queue.
//! A generic failure here costs a debugging session; a named one costs nothing.
use pyo3::exceptions::{PyNotImplementedError, PyRuntimeError};
use pyo3::PyErr;

/// The one message shape §6 specifies. Used for aten ops and, by extension,
/// for the other things the shim has not built yet (dtypes, devices, promotion).
pub fn not_implemented(message: impl Into<String>) -> PyErr {
    PyNotImplementedError::new_err(message.into())
}

/// The exact string DESIGN.md §6 asks for. Kept as its own function so the
/// wording cannot drift between call sites -- tooling that mines traces for the
/// next op to implement matches on it.
pub fn aten_not_implemented(op: &str) -> PyErr {
    not_implemented(format!(
        "aten op not implemented in torch._C shim: {op}"
    ))
}

/// candle's failures are torch's `RuntimeError`s. The `candle:` prefix stays so
/// a numerical-semantics mismatch is never mistaken for a torch-side error.
pub fn candle_err(op: &str, e: candle_core::Error) -> PyErr {
    PyRuntimeError::new_err(format!("{op}: candle: {e}"))
}
