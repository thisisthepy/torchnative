//! `torch._C._et_*`: the ExecuTorch runtime bound into this extension
//! (issue #315, step R1; SPEC S5.9).
//!
//! The runtime is the ATen-free ExecuTorch v1.4.1 core plus the whole
//! portable kernel library, compiled by the `executorch_rt` crate and linked
//! here behind the cargo feature `executorch`, on aarch64-apple-darwin only
//! (`cfg(torch_c_executorch)`, set by `build.rs`). Everything else builds
//! exactly as before, and there `_et_runtime_info()` is the one entry point
//! and says why the runtime is absent.
//!
//! This is not an operator, so it does not go through `_aten_dispatch`: a
//! program is a whole graph the runtime executes with its own kernels. The
//! door it does respect is the tensor one: inputs are read only from
//! `Repr::Dense` CPU tensors through `PyTensorBase::tensor()` (every other
//! representation refuses there, by name), and outputs come back through
//! `PyTensorBase::new` / `PyTensorBase::boolean`, the constructors every
//! dense tensor passes through.
//!
//! Copies: one in (into the method's planned arena, inside the shim) and one
//! out (from the arena into a fresh candle tensor). Both happen with the GIL
//! released, together with `execute`.
//!
//! GIL and locking: the method's `Mutex` is only ever taken inside
//! `py.detach`, and released before the closure returns, so a thread waiting
//! for the lock never holds the GIL a running execution would need. Nothing
//! inside the closure touches a Python object.

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyModule};

#[cfg(torch_c_executorch)]
pub use enabled::register;

#[cfg(not(torch_c_executorch))]
pub use disabled::register;

/// Why the runtime is not in this artefact, for `_et_runtime_info()`.
#[cfg(not(torch_c_executorch))]
fn absent_reason() -> String {
    let target = env!("TORCH_C_TARGET");
    if cfg!(feature = "executorch") {
        format!(
            "torch._C was built with the `executorch` feature for {target}, where step R1 \
             (issue #315) does not build the runtime; only aarch64-apple-darwin does"
        )
    } else {
        format!(
            "torch._C was built without the cargo feature `executorch` (target {target}); \
             scripts/vendor/install_shim.sh and tests/run.sh enable it on macOS arm64"
        )
    }
}

#[cfg(not(torch_c_executorch))]
mod disabled {
    use super::*;

    /// `torch._C._et_runtime_info()` on a build without the runtime.
    #[pyfunction]
    #[pyo3(name = "_et_runtime_info")]
    fn et_runtime_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("available", false)?;
        d.set_item("reason", super::absent_reason())?;
        d.set_item("target", env!("TORCH_C_TARGET"))?;
        Ok(d.unbind())
    }

    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(et_runtime_info, m)?)?;
        Ok(())
    }
}

#[cfg(torch_c_executorch)]
mod enabled {
    use super::*;
    use crate::dtype::TorchDType;
    use crate::tensor::{promote, PyTensorBase};
    use candle_core::{CpuStorage, DType, Device, Storage, Tensor};
    use executorch_rt as rt;
    use pyo3::sync::PyOnceLock;
    use pyo3::types::{PyList, PyType};
    use std::ffi::CString;
    use std::sync::{Arc, Mutex};

    // -------------------------------------------------------- exceptions

    /// `torch._C.EtError` (a `RuntimeError`) and one subclass per runtime
    /// code, `torch._C.Et<Name>`, built once at module init.
    static ET_ERROR: PyOnceLock<Py<PyType>> = PyOnceLock::new();
    static ET_ERRORS: PyOnceLock<Vec<(i32, Py<PyType>)>> = PyOnceLock::new();

    fn make_exceptions(m: &Bound<'_, PyModule>) -> PyResult<()> {
        let py = m.py();
        let base = PyErr::new_type(
            py,
            c"torch._C.EtError",
            Some(c"A call into the ExecuTorch runtime failed; the subclass names the runtime's error code."),
            Some(&py.get_type::<pyo3::exceptions::PyRuntimeError>()),
            None,
        )?;
        m.add("EtError", base.clone_ref(py))?;
        let mut subclasses = Vec::with_capacity(rt::ERROR_NAMES.len());
        for (code, name) in rt::ERROR_NAMES {
            let qualified = CString::new(format!("torch._C.Et{name}"))?;
            let ty = PyErr::new_type(py, qualified.as_c_str(), None, Some(base.bind(py)), None)?;
            m.add(format!("Et{name}").as_str(), ty.clone_ref(py))?;
            subclasses.push((*code, ty));
        }
        // A second module init in one process (subinterpreters are not
        // supported by this crate) would keep the first set; either is fine.
        let _ = ET_ERROR.set(py, base);
        let _ = ET_ERRORS.set(py, subclasses);
        Ok(())
    }

    fn raise(py: Python<'_>, code: i32, message: String) -> PyErr {
        let ty = ET_ERRORS
            .get(py)
            .and_then(|all| all.iter().find(|(c, _)| *c == code).map(|(_, t)| t))
            .or_else(|| ET_ERROR.get(py));
        match ty {
            Some(ty) => PyErr::from_type(ty.bind(py).clone(), message),
            None => pyo3::exceptions::PyRuntimeError::new_err(message),
        }
    }

    fn runtime_err(py: Python<'_>, e: rt::Error) -> PyErr {
        raise(py, e.code, e.to_string())
    }

    const INVALID_ARGUMENT: i32 = 0x12;
    const NOT_SUPPORTED: i32 = 0x10;

    /// What can go wrong with the GIL released; turned into a Python
    /// exception after reattaching.
    enum Fail {
        Runtime(rt::Error),
        Refused(i32, String),
    }

    impl From<rt::Error> for Fail {
        fn from(e: rt::Error) -> Self {
            Fail::Runtime(e)
        }
    }

    fn into_py(py: Python<'_>, f: Fail) -> PyErr {
        match f {
            Fail::Runtime(e) => runtime_err(py, e),
            Fail::Refused(code, msg) => raise(py, code, msg),
        }
    }

    // ------------------------------------------------------------- dtypes

    /// torchnative's dtype -> ExecuTorch `ScalarType` (c10 numbering,
    /// runtime/core/portable_type/scalar_type.h). Only the tags candle can
    /// hold; anything else refuses by name rather than being reinterpreted.
    fn et_scalar_type(tag: TorchDType) -> Option<i8> {
        Some(match tag {
            TorchDType::UInt8 => 0,
            TorchDType::Int8 => 1,
            TorchDType::Int16 => 2,
            TorchDType::Int32 => 3,
            TorchDType::Int64 => 4,
            TorchDType::Float16 => 5,
            TorchDType::Float32 => 6,
            TorchDType::Float64 => 7,
            TorchDType::Bool => 11,
            TorchDType::BFloat16 => 15,
            TorchDType::UInt32 => 28,
            _ => return None,
        })
    }

    /// The inverse, for outputs: (torch name, candle storage, is bool).
    fn from_et_scalar_type(code: i8) -> Option<(&'static str, DType, bool)> {
        Some(match code {
            0 => ("uint8", DType::U8, false),
            1 => ("int8", DType::I8, false),
            2 => ("int16", DType::I16, false),
            3 => ("int32", DType::I32, false),
            4 => ("int64", DType::I64, false),
            5 => ("float16", DType::F16, false),
            6 => ("float32", DType::F32, false),
            7 => ("float64", DType::F64, false),
            11 => ("bool", DType::U8, true),
            15 => ("bfloat16", DType::BF16, false),
            28 => ("uint32", DType::U32, false),
            _ => return None,
        })
    }

    fn et_dtype_name(code: i8) -> String {
        match from_et_scalar_type(code) {
            Some((name, _, _)) => name.to_string(),
            None => format!("executorch ScalarType {code}"),
        }
    }

    /// The bytes of a contiguous CPU tensor, without copying.
    fn host_bytes<'a>(cpu: &'a CpuStorage, start: usize, count: usize) -> Option<&'a [u8]> {
        fn view<T>(v: &[T], start: usize, count: usize) -> Option<&[u8]> {
            let end = start.checked_add(count)?;
            let s = v.get(start..end)?;
            // SAFETY: reinterpreting an initialized slice of plain numbers as
            // bytes; the length is exactly its size in bytes.
            Some(unsafe { std::slice::from_raw_parts(s.as_ptr().cast::<u8>(), std::mem::size_of_val(s)) })
        }
        match cpu {
            CpuStorage::U8(v) => view(v, start, count),
            CpuStorage::U32(v) => view(v, start, count),
            CpuStorage::I16(v) => view(v, start, count),
            CpuStorage::I8(v) => view(v, start, count),
            CpuStorage::I32(v) => view(v, start, count),
            CpuStorage::I64(v) => view(v, start, count),
            CpuStorage::BF16(v) => view(v, start, count),
            CpuStorage::F16(v) => view(v, start, count),
            CpuStorage::F32(v) => view(v, start, count),
            CpuStorage::F64(v) => view(v, start, count),
            // float8 and candle's MX formats have no ExecuTorch counterpart the
            // binding passes; their tags already refused in et_scalar_type.
            _ => None,
        }
    }

    // --------------------------------------------------------------- info

    /// `torch._C._et_runtime_info()`: what is compiled in, read from the
    /// runtime itself (kernel and backend registries), not from a list.
    #[pyfunction]
    #[pyo3(name = "_et_runtime_info")]
    fn et_runtime_info(py: Python<'_>) -> PyResult<Py<PyDict>> {
        let d = PyDict::new(py);
        d.set_item("available", true)?;
        d.set_item("version", rt::UPSTREAM_VERSION)?;
        d.set_item("commit", rt::UPSTREAM_COMMIT)?;
        d.set_item("target", env!("TORCH_C_TARGET"))?;
        d.set_item("pal", rt::PAL)?;
        d.set_item("kernels", rt::registered_kernels().len())?;
        d.set_item("backends", rt::registered_backends())?;
        d.set_item("event_tracer", rt::event_tracer_enabled())?;
        d.set_item("program_verification", rt::program_verification_enabled())?;
        d.set_item("threadpool", false)?;
        Ok(d.unbind())
    }

    /// `torch._C._et_registered_kernels()`: the operator registry's names, in
    /// registration order. Empty would mean the static registration objects
    /// were dropped by the linker (executorch_rt's build.rs, whole-archive).
    #[pyfunction]
    #[pyo3(name = "_et_registered_kernels")]
    fn et_registered_kernels() -> Vec<String> {
        rt::registered_kernels()
    }

    #[pyfunction]
    #[pyo3(name = "_et_registered_backends")]
    fn et_registered_backends() -> Vec<String> {
        rt::registered_backends()
    }

    // ------------------------------------------------------------ program

    #[pyclass(name = "EtProgram", module = "torch._C", frozen)]
    pub struct PyEtProgram {
        inner: rt::Program,
    }

    /// `torch._C._et_load(buf)`: copies `buf` once into an aligned buffer the
    /// program owns, and loads it with full verification
    /// (`Program::Verification::InternalConsistency`). A truncated or garbage
    /// buffer raises `EtInvalidProgram`.
    #[pyfunction]
    #[pyo3(name = "_et_load")]
    fn et_load(py: Python<'_>, buf: &[u8]) -> PyResult<PyEtProgram> {
        let loaded = py.detach(|| rt::Program::load(buf));
        loaded.map(|inner| PyEtProgram { inner }).map_err(|e| runtime_err(py, e))
    }

    fn slot_dict<'py>(py: Python<'py>, s: &rt::SlotMeta) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        let tag = match s.tag {
            0 => "None",
            1 => "Tensor",
            2 => "String",
            3 => "Double",
            4 => "Int",
            5 => "Bool",
            _ => "Other",
        };
        d.set_item("tag", tag)?;
        if s.tag == 1 {
            d.set_item("dtype", et_dtype_name(s.scalar_type))?;
            d.set_item("sizes", s.sizes.clone())?;
            d.set_item("dim_order", s.dim_order.clone())?;
            d.set_item("nbytes", s.nbytes)?;
            d.set_item("memory_planned", s.memory_planned)?;
        }
        Ok(d)
    }

    #[pymethods]
    impl PyEtProgram {
        fn method_names(&self) -> Vec<String> {
            self.inner.method_names()
        }

        /// What the program declares. `backends` is a declaration, never
        /// evidence that a backend ran (AGENTS.md section 16).
        fn method_meta(&self, py: Python<'_>, name: &str) -> PyResult<Py<PyDict>> {
            let meta = self.inner.method_meta(name).map_err(|e| runtime_err(py, e))?;
            let d = PyDict::new(py);
            d.set_item("name", &meta.name)?;
            let inputs = PyList::empty(py);
            for s in &meta.inputs {
                inputs.append(slot_dict(py, s)?)?;
            }
            let outputs = PyList::empty(py);
            for s in &meta.outputs {
                outputs.append(slot_dict(py, s)?)?;
            }
            d.set_item("inputs", inputs)?;
            d.set_item("outputs", outputs)?;
            d.set_item("planned_buffers", meta.planned_buffers.clone())?;
            d.set_item("backends", meta.backends.clone())?;
            d.set_item("num_instructions", meta.num_instructions)?;
            Ok(d.unbind())
        }

        /// `Program::load_method`: arenas sized from `MethodMeta`, the
        /// counting tracer attached, `Method::init` run (a kernel missing
        /// from the registry raises `EtOperatorMissing` here).
        fn load_method(&self, py: Python<'_>, name: &str) -> PyResult<PyEtMethod> {
            let meta = self.inner.method_meta(name).map_err(|e| runtime_err(py, e))?;
            let program = &self.inner;
            let method = py.detach(|| program.load_method(name)).map_err(|e| runtime_err(py, e))?;
            Ok(PyEtMethod { inner: Arc::new(Mutex::new(method)), meta })
        }

        #[getter]
        fn nbytes(&self) -> usize {
            self.inner.nbytes()
        }

        fn __repr__(&self) -> String {
            format!(
                "<torch._C.EtProgram {} bytes, methods {:?}>",
                self.inner.nbytes(),
                self.inner.method_names()
            )
        }
    }

    // ------------------------------------------------------------- method

    #[pyclass(name = "EtMethod", module = "torch._C", frozen)]
    pub struct PyEtMethod {
        inner: Arc<Mutex<rt::Method>>,
        meta: rt::MethodMeta,
    }

    /// An output copied out of the arena, with the GIL released.
    enum Out {
        Tensor(Tensor, bool),
        Int(i64),
        Double(f64),
        Bool(bool),
        None,
    }

    fn lock(m: &Mutex<rt::Method>) -> std::sync::MutexGuard<'_, rt::Method> {
        m.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    fn copy_out(index: usize, value: rt::Value<'_>) -> Result<Out, Fail> {
        Ok(match value {
            rt::Value::Tensor(view) => {
                let identity = view.dim_order.iter().enumerate().all(|(i, d)| *d as usize == i);
                if !identity {
                    return Err(Fail::Refused(
                        NOT_SUPPORTED,
                        format!(
                            "EtMethod.execute: output {index} has dim order {:?}; the binding \
                             returns contiguous (identity dim order) tensors only",
                            view.dim_order
                        ),
                    ));
                }
                let (name, dtype, is_bool) = from_et_scalar_type(view.scalar_type).ok_or_else(|| {
                    Fail::Refused(
                        NOT_SUPPORTED,
                        format!(
                            "EtMethod.execute: output {index} is {}, which torch._C shim cannot hold",
                            et_dtype_name(view.scalar_type)
                        ),
                    )
                })?;
                let shape: Vec<usize> = view.sizes.iter().map(|s| *s as usize).collect();
                let tensor = if is_bool {
                    // ExecuTorch's bool is C++ bool (0/1); reduce anyway so the
                    // shim's bool invariant holds by construction.
                    let bytes: Vec<u8> = view.data.iter().map(|b| u8::from(*b != 0)).collect();
                    Tensor::from_vec(bytes, shape.clone(), &Device::Cpu)
                } else {
                    Tensor::from_raw_buffer(view.data, dtype, &shape, &Device::Cpu)
                }
                .map_err(|e| {
                    Fail::Refused(
                        NOT_SUPPORTED,
                        format!("EtMethod.execute: output {index} ({name}, {shape:?}): candle: {e}"),
                    )
                })?;
                Out::Tensor(tensor, is_bool)
            }
            rt::Value::Int(v) => Out::Int(v),
            rt::Value::Double(v) => Out::Double(v),
            rt::Value::Bool(v) => Out::Bool(v),
            rt::Value::None => Out::None,
        })
    }

    #[pymethods]
    impl PyEtMethod {
        /// Run the method on `inputs` (CPU tensors, in program order) and
        /// return its outputs as new tensors. One copy in, one copy out, with
        /// the GIL released for both and for the execution.
        fn execute(&self, py: Python<'_>, inputs: Vec<Bound<'_, PyAny>>) -> PyResult<Vec<Py<PyAny>>> {
            let want = self.meta.inputs.len();
            if inputs.len() != want {
                return Err(raise(
                    py,
                    INVALID_ARGUMENT,
                    format!(
                        "EtMethod.execute: method {:?} takes {want} inputs, got {}",
                        self.meta.name,
                        inputs.len()
                    ),
                ));
            }
            // With the GIL: read each input's dense candle tensor (an Arc
            // clone, no data copied) and its torch dtype.
            let mut staged: Vec<(i8, Vec<i64>, Tensor)> = Vec::with_capacity(inputs.len());
            for (i, obj) in inputs.iter().enumerate() {
                let base = obj.extract::<PyRef<'_, PyTensorBase>>().map_err(|_| {
                    raise(
                        py,
                        INVALID_ARGUMENT,
                        format!(
                            "EtMethod.execute: input {i} is a {}, not a tensor",
                            obj.get_type().name().map(|n| n.to_string()).unwrap_or_default()
                        ),
                    )
                })?;
                let tag = base.tag();
                let code = et_scalar_type(tag).ok_or_else(|| {
                    raise(
                        py,
                        NOT_SUPPORTED,
                        format!(
                            "EtMethod.execute: input {i} is torch.{}, which the ExecuTorch binding \
                             does not pass",
                            tag.name()
                        ),
                    )
                })?;
                // Meta, quantized, Vulkan and complex tensors refuse here, by name.
                let tensor = base.tensor()?.clone();
                if !tensor.device().is_cpu() {
                    return Err(raise(
                        py,
                        NOT_SUPPORTED,
                        format!(
                            "EtMethod.execute: input {i} is on {:?}; the ExecuTorch runtime reads \
                             host memory, move it with .cpu() first",
                            tensor.device().location()
                        ),
                    ));
                }
                let sizes: Vec<i64> = tensor.dims().iter().map(|d| *d as i64).collect();
                staged.push((code, sizes, tensor));
            }

            let method = &self.inner;
            let result: Result<Vec<Out>, Fail> = py.detach(move || {
                let mut m = lock(method);
                for (i, (code, sizes, tensor)) in staged.iter().enumerate() {
                    let tensor = tensor.contiguous().map_err(|e| {
                        Fail::Refused(NOT_SUPPORTED, format!("EtMethod.execute: input {i}: candle: {e}"))
                    })?;
                    let (guard, layout) = tensor.storage_and_layout();
                    let Storage::Cpu(cpu) = &*guard else {
                        return Err(Fail::Refused(
                            NOT_SUPPORTED,
                            format!("EtMethod.execute: input {i} has no host storage"),
                        ));
                    };
                    let bytes = host_bytes(cpu, layout.start_offset(), tensor.elem_count()).ok_or_else(|| {
                        Fail::Refused(
                            NOT_SUPPORTED,
                            format!("EtMethod.execute: input {i}: cannot read its {:?} storage", tensor.dtype()),
                        )
                    })?;
                    m.set_input(i, *code, sizes, bytes)?;
                }
                m.execute()?;
                let mut outs = Vec::with_capacity(m.num_outputs());
                for i in 0..m.num_outputs() {
                    let value = m.output(i)?;
                    outs.push(copy_out(i, value)?);
                }
                Ok(outs)
            });
            let outs = result.map_err(|f| into_py(py, f))?;

            let mut objects = Vec::with_capacity(outs.len());
            for out in outs {
                objects.push(match out {
                    Out::Tensor(t, true) => {
                        promote(py, PyTensorBase::boolean(t)?.into_pyobject(py)?.into_any().unbind())?
                    }
                    Out::Tensor(t, false) => {
                        promote(py, PyTensorBase::new(t)?.into_pyobject(py)?.into_any().unbind())?
                    }
                    Out::Int(v) => v.into_pyobject(py)?.into_any().unbind(),
                    Out::Double(v) => v.into_pyobject(py)?.into_any().unbind(),
                    Out::Bool(v) => pyo3::types::PyBool::new(py, v).to_owned().into_any().unbind(),
                    Out::None => py.None(),
                });
            }
            Ok(objects)
        }

        /// Runtime evidence of what ran: `operator_calls` counts each kernel
        /// by the name its generated wrapper reports when it is entered
        /// (`addmm.out`), `scopes` every profiling scope the runtime opened
        /// (`OPERATOR_CALL` per kernel instruction), `delegate_ops` what a
        /// delegate's own profiler reported (empty in R1), `executions` the
        /// successful `execute` calls. Since load or `reset_counters()`.
        fn counters(&self, py: Python<'_>) -> PyResult<Py<PyDict>> {
            let method = &self.inner;
            let (executions, ops, scopes, delegate) = py.detach(|| {
                let m = lock(method);
                (
                    m.executions(),
                    m.counters(rt::Counter::OperatorCalls),
                    m.counters(rt::Counter::Scopes),
                    m.counters(rt::Counter::DelegateOps),
                )
            });
            let d = PyDict::new(py);
            d.set_item("executions", executions)?;
            for (key, list) in [("operator_calls", ops), ("scopes", scopes), ("delegate_ops", delegate)] {
                let sub = PyDict::new(py);
                for (name, count) in list {
                    sub.set_item(name, count)?;
                }
                d.set_item(key, sub)?;
            }
            Ok(d.unbind())
        }

        fn reset_counters(&self, py: Python<'_>) {
            let method = &self.inner;
            py.detach(|| lock(method).reset_counters());
        }

        #[getter]
        fn name(&self) -> String {
            self.meta.name.clone()
        }

        #[getter]
        fn num_inputs(&self) -> usize {
            self.meta.inputs.len()
        }

        #[getter]
        fn num_outputs(&self) -> usize {
            self.meta.outputs.len()
        }

        fn __repr__(&self) -> String {
            format!(
                "<torch._C.EtMethod {:?}: {} inputs, {} outputs>",
                self.meta.name,
                self.meta.inputs.len(),
                self.meta.outputs.len()
            )
        }
    }

    pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
        rt::init();
        make_exceptions(m)?;
        m.add_class::<PyEtProgram>()?;
        m.add_class::<PyEtMethod>()?;
        m.add_function(wrap_pyfunction!(et_runtime_info, m)?)?;
        m.add_function(wrap_pyfunction!(et_registered_kernels, m)?)?;
        m.add_function(wrap_pyfunction!(et_registered_backends, m)?)?;
        m.add_function(wrap_pyfunction!(et_load, m)?)?;
        Ok(())
    }
}
