//! Spike for `torch._C`.
//!
//! Only enough to prove the cross-build links against each target's CPython.
//! See docs/RUST_CROSSBUILD.md -- Android links a shared library out of
//! prefix/lib, iOS links Python.framework, and those are not the same wiring.
use pyo3::prelude::*;

#[pyfunction]
fn _spike_version() -> &'static str {
    "torch._C spike"
}

#[pymodule]
fn _C(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_spike_version, m)?)?;
    Ok(())
}
