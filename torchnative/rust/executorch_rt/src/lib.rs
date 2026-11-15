//! Safe Rust over torchnative's ExecuTorch runtime shim (issue #315, step R1).
//!
//! `build.rs` compiles the vendored ExecuTorch v1.4.1 runtime, the prim ops,
//! the whole portable kernel library and `shim/tn_et.cpp` into one static
//! archive. This file is the only Rust that touches the C surface
//! (`shim/tn_et.h`); `torch_c` sees the types below and nothing else.
//!
//! # Lifetimes
//!
//! ```text
//! AlignedBytes (the program, copied once)  outlives  tn_et_program  outlives  every tn_et_method
//! ```
//!
//! * The runtime's `BufferDataLoader` does not copy: `Program`, every
//!   `Method` and every delegate read the program bytes in place, for as long
//!   as the `Program` lives. So [`Program::load`] copies the caller's bytes
//!   once into an allocation aligned to 64 bytes (the runtime requires
//!   `alignof(max_align_t)`; a Python `bytes` object guarantees neither that
//!   nor immutability across the GIL), and that allocation lives in the same
//!   reference-counted `ProgramInner` as the C++ `Program`, whose `Drop`
//!   frees the C++ object first and the bytes after (field drop order).
//! * A [`Method`] holds an `Arc` of that `ProgramInner`, so a method keeps
//!   its program (and therefore the bytes) alive even after the Python
//!   `EtProgram` is gone. The method's arenas (one per memory-planned buffer,
//!   sized from `MethodMeta`), its allocators and its tracer live in the C++
//!   handle and are freed in `tn_et_method_free`, after the `Method` itself.
//! * Output views borrow the `Method` (`&self`), so the borrow checker
//!   enforces the runtime's rule that an output is overwritten by the next
//!   `set_input`/`execute` (both take `&mut self`).
//!
//! # Threads
//!
//! * `ProgramInner` is `Send + Sync`: after load the C++ `Program` is only
//!   read, and every call into it still goes through `ProgramInner::lock`, so
//!   the `Sync` claim does not rest on reading upstream's code for hidden
//!   mutation.
//! * [`Method`] is `Send` and not `Sync`: an ExecuTorch `Method` is not
//!   reentrant, and every mutating call takes `&mut self`. The C++ handle has
//!   no thread affinity (no thread-local state is kept between calls; the
//!   shim's log buffer is per thread but is read right after the call that
//!   filled it, on the same thread). `torch_c` wraps it in a `Mutex`.
//! * The kernel registry is process-global, filled by static initializers
//!   when the extension is loaded, and only read afterwards.
//!
//! # What cannot be prevented
//!
//! ExecuTorch has no exceptions; an internal invariant failure is an
//! `ET_CHECK`, which calls `et_pal_abort` and ends the process. The shim
//! prevents the reachable ones it knows of by validating first (program
//! size and `InternalConsistency` verification at load, input dtype, rank,
//! shape and byte count before `set_input`, output index before
//! `get_output`), and on any other abort prints the runtime's last log lines
//! to stderr before the process dies. A kernel that `ET_CHECK`s on its own
//! inputs inside `execute` is not preventable from here.

use std::alloc::{self, Layout};
use std::ffi::{c_char, CStr, CString};
use std::ptr::NonNull;
use std::sync::{Arc, Mutex, Once};

/// The upstream tag and commit the vendored tree is pinned to (read from
/// `MANIFEST.tsv` by `build.rs`).
pub const UPSTREAM_VERSION: &str = env!("TN_ET_UPSTREAM_VERSION");
pub const UPSTREAM_COMMIT: &str = env!("TN_ET_UPSTREAM_COMMIT");

/// The PAL compiled in (`runtime/platform/default/posix.cpp`), with the log
/// and abort hooks replaced by the shim through `register_pal()`.
pub const PAL: &str = "posix";

mod ffi {
    use std::ffi::c_char;

    #[repr(C)]
    pub struct tn_et_program {
        _private: [u8; 0],
    }

    #[repr(C)]
    pub struct tn_et_method {
        _private: [u8; 0],
    }

    #[repr(C)]
    #[derive(Default)]
    pub struct tn_et_method_info {
        pub num_inputs: usize,
        pub num_outputs: usize,
        pub num_planned_buffers: usize,
        pub num_backends: usize,
        pub num_instructions: usize,
    }

    #[repr(C)]
    pub struct tn_et_tensor_info {
        pub tag: i32,
        pub scalar_type: i8,
        pub memory_planned: u8,
        pub ndim: usize,
        pub sizes: *const i32,
        pub dim_order: *const u8,
        pub nbytes: usize,
    }

    #[repr(C)]
    pub struct tn_et_value {
        pub tag: i32,
        pub scalar_type: i8,
        pub ndim: usize,
        pub sizes: *const i32,
        pub dim_order: *const u8,
        pub data: *const std::ffi::c_void,
        pub nbytes: usize,
        pub int_value: i64,
        pub double_value: f64,
        pub bool_value: u8,
    }

    extern "C" {
        pub fn tn_et_init() -> i32;
        pub fn tn_et_take_log(buf: *mut c_char, cap: usize) -> usize;
        pub fn tn_et_event_tracer_enabled() -> i32;
        pub fn tn_et_program_verification_enabled() -> i32;
        pub fn tn_et_kernel_count() -> usize;
        pub fn tn_et_kernel_name(index: usize) -> *const c_char;
        pub fn tn_et_backend_count() -> usize;
        pub fn tn_et_backend_name(index: usize) -> *const c_char;

        pub fn tn_et_program_load(data: *const u8, len: usize, out: *mut *mut tn_et_program) -> i32;
        pub fn tn_et_program_free(program: *mut tn_et_program);
        pub fn tn_et_program_num_methods(program: *const tn_et_program) -> usize;
        pub fn tn_et_program_method_name(program: *const tn_et_program, index: usize) -> *const c_char;
        pub fn tn_et_method_meta(
            program: *const tn_et_program,
            name: *const c_char,
            out: *mut tn_et_method_info,
        ) -> i32;
        pub fn tn_et_method_input_meta(
            program: *const tn_et_program,
            name: *const c_char,
            index: usize,
            out: *mut tn_et_tensor_info,
        ) -> i32;
        pub fn tn_et_method_output_meta(
            program: *const tn_et_program,
            name: *const c_char,
            index: usize,
            out: *mut tn_et_tensor_info,
        ) -> i32;
        pub fn tn_et_method_planned_buffer_size(
            program: *const tn_et_program,
            name: *const c_char,
            index: usize,
            out: *mut i64,
        ) -> i32;
        pub fn tn_et_method_backend_name(
            program: *const tn_et_program,
            name: *const c_char,
            index: usize,
            out: *mut *const c_char,
        ) -> i32;

        pub fn tn_et_method_load(
            program: *const tn_et_program,
            name: *const c_char,
            out: *mut *mut tn_et_method,
        ) -> i32;
        pub fn tn_et_method_free(method: *mut tn_et_method);
        pub fn tn_et_method_num_inputs(method: *const tn_et_method) -> usize;
        pub fn tn_et_method_num_outputs(method: *const tn_et_method) -> usize;
        pub fn tn_et_method_set_input(
            method: *mut tn_et_method,
            index: usize,
            scalar_type: i8,
            sizes: *const i64,
            ndim: usize,
            data: *const std::ffi::c_void,
            nbytes: usize,
        ) -> i32;
        pub fn tn_et_method_execute(method: *mut tn_et_method) -> i32;
        pub fn tn_et_method_output(method: *const tn_et_method, index: usize, out: *mut tn_et_value) -> i32;
        pub fn tn_et_method_executions(method: *const tn_et_method) -> u64;
        pub fn tn_et_method_counter_count(method: *const tn_et_method, family: i32) -> usize;
        pub fn tn_et_method_counter_at(
            method: *const tn_et_method,
            family: i32,
            index: usize,
            name: *mut *const c_char,
            count: *mut u64,
        ) -> i32;
        pub fn tn_et_method_reset_counters(method: *mut tn_et_method);
    }
}

// ------------------------------------------------------------------ errors

/// `executorch::runtime::Error` (runtime/core/error.h), by code. The names
/// are upstream's enumerator names; `torch_c` prefixes them with `Et` for the
/// Python exception classes.
pub const ERROR_NAMES: &[(i32, &str)] = &[
    (0x01, "Internal"),
    (0x02, "InvalidState"),
    (0x03, "EndOfMethod"),
    (0x04, "AlreadyLoaded"),
    (0x10, "NotSupported"),
    (0x11, "NotImplemented"),
    (0x12, "InvalidArgument"),
    (0x13, "InvalidType"),
    (0x14, "OperatorMissing"),
    (0x15, "RegistrationExceedingMaxKernels"),
    (0x16, "RegistrationAlreadyRegistered"),
    (0x20, "NotFound"),
    (0x21, "MemoryAllocationFailed"),
    (0x22, "AccessFailed"),
    (0x23, "InvalidProgram"),
    (0x24, "InvalidExternalData"),
    (0x25, "OutOfResources"),
    (0x30, "DelegateInvalidCompatibility"),
    (0x31, "DelegateMemoryAllocationFailed"),
    (0x32, "DelegateInvalidHandle"),
];

/// The enumerator name for a code, or `None` for a code this table does not
/// know (which would mean the vendored runtime grew a code).
pub fn error_name(code: i32) -> Option<&'static str> {
    ERROR_NAMES.iter().find(|(c, _)| *c == code).map(|(_, n)| *n)
}

/// A failed runtime call: the runtime's code, what was being done, and the
/// runtime's own log lines from that call (where it names the reason).
#[derive(Debug, Clone)]
pub struct Error {
    pub code: i32,
    pub during: String,
    pub log: String,
}

impl Error {
    pub fn name(&self) -> &'static str {
        error_name(self.code).unwrap_or("Unknown")
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "ExecuTorch {} (0x{:02x}) during {}", self.name(), self.code, self.during)?;
        let log = self.log.trim_end();
        if !log.is_empty() {
            write!(f, "; runtime log:\n{log}")?;
        }
        Ok(())
    }
}

impl std::error::Error for Error {}

fn take_log() -> String {
    let mut buf = vec![0u8; 8192];
    // SAFETY: buf is writable for buf.len() bytes; the shim NUL-terminates.
    let n = unsafe { ffi::tn_et_take_log(buf.as_mut_ptr().cast(), buf.len()) };
    let n = n.min(buf.len() - 1);
    buf.truncate(n);
    String::from_utf8_lossy(&buf).into_owned()
}

/// Turns a status into a `Result`, collecting the runtime's log on failure.
/// Must be called on the thread that made the call (the log is per thread).
fn check(status: i32, during: impl FnOnce() -> String) -> Result<(), Error> {
    if status == 0 {
        return Ok(());
    }
    Err(Error { code: status, during: during(), log: take_log() })
}

fn local(code: i32, during: String, why: String) -> Error {
    Error { code, during, log: why }
}

const INVALID_ARGUMENT: i32 = 0x12;
const NOT_SUPPORTED: i32 = 0x10;

// ------------------------------------------------------------- process-wide

static INIT: Once = Once::new();

/// Installs the shim's PAL hooks and runs `runtime_init()`, once.
pub fn init() {
    INIT.call_once(|| {
        // SAFETY: no arguments; idempotent on the C++ side as well.
        unsafe {
            ffi::tn_et_init();
        }
    });
}

fn owned(ptr: *const c_char) -> Option<String> {
    if ptr.is_null() {
        return None;
    }
    // SAFETY: the shim returns NUL-terminated strings owned by the runtime
    // (kernel and backend names are static; method names live in the
    // program, which the caller holds).
    Some(unsafe { CStr::from_ptr(ptr) }.to_string_lossy().into_owned())
}

/// Every kernel in the runtime's registry, in registration order. Filled by
/// static initializers when the extension is loaded; an empty list means the
/// archive's registration objects were dropped by the linker.
pub fn registered_kernels() -> Vec<String> {
    init();
    // SAFETY: plain reads of the global registry.
    let n = unsafe { ffi::tn_et_kernel_count() };
    (0..n).filter_map(|i| owned(unsafe { ffi::tn_et_kernel_name(i) })).collect()
}

/// Every registered backend (none in R1: no delegate is compiled).
pub fn registered_backends() -> Vec<String> {
    init();
    // SAFETY: plain reads of the global backend registry.
    let n = unsafe { ffi::tn_et_backend_count() };
    (0..n).filter_map(|i| owned(unsafe { ffi::tn_et_backend_name(i) })).collect()
}

pub fn event_tracer_enabled() -> bool {
    // SAFETY: constant.
    unsafe { ffi::tn_et_event_tracer_enabled() != 0 }
}

pub fn program_verification_enabled() -> bool {
    // SAFETY: constant.
    unsafe { ffi::tn_et_program_verification_enabled() != 0 }
}

// ------------------------------------------------------------------ bytes

/// The one copy of the program bytes, aligned for the runtime.
struct AlignedBytes {
    ptr: NonNull<u8>,
    len: usize,
    layout: Layout,
}

/// The runtime requires `alignof(max_align_t)` (16 on arm64); 64 also covers
/// the cache-line alignment exir gives constant data.
const PROGRAM_ALIGNMENT: usize = 64;

impl AlignedBytes {
    fn copy_of(bytes: &[u8]) -> Option<Self> {
        let layout = Layout::from_size_align(bytes.len().max(1), PROGRAM_ALIGNMENT).ok()?;
        // SAFETY: layout has non-zero size.
        let ptr = NonNull::new(unsafe { alloc::alloc(layout) })?;
        // SAFETY: ptr is valid for layout.size() >= bytes.len() bytes and does
        // not overlap bytes (fresh allocation).
        unsafe { std::ptr::copy_nonoverlapping(bytes.as_ptr(), ptr.as_ptr(), bytes.len()) };
        Some(Self { ptr, len: bytes.len(), layout })
    }
}

impl Drop for AlignedBytes {
    fn drop(&mut self) {
        // SAFETY: allocated with exactly this layout in copy_of.
        unsafe { alloc::dealloc(self.ptr.as_ptr(), self.layout) }
    }
}

// ---------------------------------------------------------------- program

struct ProgramInner {
    raw: NonNull<ffi::tn_et_program>,
    /// Serializes every call into the C++ Program (see the module docs).
    lock: Mutex<()>,
    /// Declared after `raw`, so it is dropped after `Drop::drop` has freed the
    /// C++ Program that reads it.
    bytes: AlignedBytes,
}

// SAFETY: the C++ Program has no thread affinity; every access goes through
// `lock` except `Drop`, which has exclusive access by construction. The bytes
// are never written after `copy_of`.
unsafe impl Send for ProgramInner {}
unsafe impl Sync for ProgramInner {}

impl Drop for ProgramInner {
    fn drop(&mut self) {
        // SAFETY: raw came from tn_et_program_load and is freed exactly once;
        // no Method is alive (each holds an Arc of self).
        unsafe { ffi::tn_et_program_free(self.raw.as_ptr()) }
    }
}

impl ProgramInner {
    fn guard(&self) -> std::sync::MutexGuard<'_, ()> {
        self.lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

/// A loaded ExecuTorch program. Cheap to clone (shared).
#[derive(Clone)]
pub struct Program {
    inner: Arc<ProgramInner>,
}

/// One input or output slot of a method, as the program declares it.
#[derive(Debug, Clone)]
pub struct SlotMeta {
    /// `executorch::runtime::Tag` (1 = Tensor, 3 = Double, 4 = Int, 5 = Bool).
    pub tag: i32,
    pub scalar_type: i8,
    pub sizes: Vec<i32>,
    pub dim_order: Vec<u8>,
    pub nbytes: usize,
    pub memory_planned: bool,
}

/// What `MethodMeta` says about a method. `backends` is what the program
/// declares, never evidence that a backend ran (AGENTS.md section 16).
#[derive(Debug, Clone)]
pub struct MethodMeta {
    pub name: String,
    pub inputs: Vec<SlotMeta>,
    pub outputs: Vec<SlotMeta>,
    pub planned_buffers: Vec<i64>,
    pub backends: Vec<String>,
    pub num_instructions: usize,
}

fn cstring(name: &str, during: &str) -> Result<CString, Error> {
    CString::new(name).map_err(|_| {
        local(INVALID_ARGUMENT, during.to_string(), format!("method name {name:?} contains a NUL byte"))
    })
}

fn slot(info: &ffi::tn_et_tensor_info) -> SlotMeta {
    let (sizes, dim_order) = if info.tag == 1 && info.ndim > 0 {
        // SAFETY: for a tensor slot the shim points both arrays into the
        // program's flatbuffer, ndim entries each, alive while the program is.
        unsafe {
            (
                std::slice::from_raw_parts(info.sizes, info.ndim).to_vec(),
                std::slice::from_raw_parts(info.dim_order, info.ndim).to_vec(),
            )
        }
    } else {
        (Vec::new(), Vec::new())
    };
    SlotMeta {
        tag: info.tag,
        scalar_type: info.scalar_type,
        sizes,
        dim_order,
        nbytes: info.nbytes,
        memory_planned: info.memory_planned != 0,
    }
}

impl Program {
    /// Copies `bytes` once into an aligned, owned buffer and loads it with
    /// `Program::Verification::InternalConsistency`. A truncated or garbage
    /// buffer is `InvalidProgram`.
    pub fn load(bytes: &[u8]) -> Result<Program, Error> {
        init();
        let owned = AlignedBytes::copy_of(bytes).ok_or_else(|| {
            local(0x21, "Program::load".into(), format!("cannot allocate {} bytes for the program", bytes.len()))
        })?;
        let mut raw: *mut ffi::tn_et_program = std::ptr::null_mut();
        // SAFETY: owned.ptr is valid for owned.len bytes and outlives the
        // program (it is moved into the same ProgramInner below).
        let status = unsafe { ffi::tn_et_program_load(owned.ptr.as_ptr(), owned.len, &mut raw) };
        check(status, || format!("Program::load of {} bytes", bytes.len()))?;
        let raw = NonNull::new(raw).ok_or_else(|| {
            local(0x01, "Program::load".into(), "the shim reported success and returned no program".into())
        })?;
        Ok(Program { inner: Arc::new(ProgramInner { raw, lock: Mutex::new(()), bytes: owned }) })
    }

    /// The program's size in bytes (the owned copy).
    pub fn nbytes(&self) -> usize {
        self.inner.bytes.len
    }

    pub fn method_names(&self) -> Vec<String> {
        let _g = self.inner.guard();
        let raw = self.inner.raw.as_ptr();
        // SAFETY: raw is a live program; calls serialized by the guard.
        let n = unsafe { ffi::tn_et_program_num_methods(raw) };
        (0..n).filter_map(|i| owned(unsafe { ffi::tn_et_program_method_name(raw, i) })).collect()
    }

    pub fn method_meta(&self, name: &str) -> Result<MethodMeta, Error> {
        let during = format!("method_meta({name:?})");
        let cname = cstring(name, &during)?;
        let _g = self.inner.guard();
        let raw = self.inner.raw.as_ptr();
        let mut info = ffi::tn_et_method_info::default();
        // SAFETY: raw is live, cname is NUL-terminated, info is writable.
        check(unsafe { ffi::tn_et_method_meta(raw, cname.as_ptr(), &mut info) }, || during.clone())?;
        let empty = || ffi::tn_et_tensor_info {
            tag: 0,
            scalar_type: 0,
            memory_planned: 0,
            ndim: 0,
            sizes: std::ptr::null(),
            dim_order: std::ptr::null(),
            nbytes: 0,
        };
        let mut inputs = Vec::with_capacity(info.num_inputs);
        for i in 0..info.num_inputs {
            let mut t = empty();
            // SAFETY: as above.
            check(unsafe { ffi::tn_et_method_input_meta(raw, cname.as_ptr(), i, &mut t) }, || {
                format!("{during}: input {i}")
            })?;
            inputs.push(slot(&t));
        }
        let mut outputs = Vec::with_capacity(info.num_outputs);
        for i in 0..info.num_outputs {
            let mut t = empty();
            // SAFETY: as above.
            check(unsafe { ffi::tn_et_method_output_meta(raw, cname.as_ptr(), i, &mut t) }, || {
                format!("{during}: output {i}")
            })?;
            outputs.push(slot(&t));
        }
        let mut planned_buffers = Vec::with_capacity(info.num_planned_buffers);
        for i in 0..info.num_planned_buffers {
            let mut size = 0i64;
            // SAFETY: as above.
            check(
                unsafe { ffi::tn_et_method_planned_buffer_size(raw, cname.as_ptr(), i, &mut size) },
                || format!("{during}: planned buffer {i}"),
            )?;
            planned_buffers.push(size);
        }
        let mut backends = Vec::with_capacity(info.num_backends);
        for i in 0..info.num_backends {
            let mut ptr: *const c_char = std::ptr::null();
            // SAFETY: as above.
            check(unsafe { ffi::tn_et_method_backend_name(raw, cname.as_ptr(), i, &mut ptr) }, || {
                format!("{during}: backend {i}")
            })?;
            backends.extend(owned(ptr));
        }
        Ok(MethodMeta {
            name: name.to_string(),
            inputs,
            outputs,
            planned_buffers,
            backends,
            num_instructions: info.num_instructions,
        })
    }

    /// `Program::load_method`: allocates the method's planned arenas from its
    /// `MethodMeta`, attaches the counting tracer, and runs `Method::init`
    /// (kernel resolution: a program naming a kernel the registry lacks
    /// fails here with `OperatorMissing`).
    pub fn load_method(&self, name: &str) -> Result<Method, Error> {
        let during = format!("load_method({name:?})");
        let cname = cstring(name, &during)?;
        let _g = self.inner.guard();
        let mut raw: *mut ffi::tn_et_method = std::ptr::null_mut();
        // SAFETY: the program is live and stays live through the Arc the
        // Method takes below.
        let status = unsafe { ffi::tn_et_method_load(self.inner.raw.as_ptr(), cname.as_ptr(), &mut raw) };
        check(status, || during.clone())?;
        let raw = NonNull::new(raw)
            .ok_or_else(|| local(0x01, during.clone(), "the shim reported success and returned no method".into()))?;
        Ok(Method { raw, name: name.to_string(), _program: Arc::clone(&self.inner) })
    }
}

// ----------------------------------------------------------------- method

/// A loaded method, ready to execute. See the module docs for lifetimes and
/// threads.
pub struct Method {
    raw: NonNull<ffi::tn_et_method>,
    name: String,
    /// Keeps the C++ Program and its bytes alive; dropped after `Drop::drop`
    /// has freed the method.
    _program: Arc<ProgramInner>,
}

// SAFETY: the C++ method handle has no thread affinity; `Method` is not
// `Sync`, and every call that mutates it takes `&mut self`.
unsafe impl Send for Method {}

impl Drop for Method {
    fn drop(&mut self) {
        // SAFETY: raw came from tn_et_method_load and is freed exactly once,
        // while _program still holds the Program alive.
        unsafe { ffi::tn_et_method_free(self.raw.as_ptr()) }
    }
}

/// Counter families (`TN_ET_COUNT_*` in the shim).
#[derive(Debug, Clone, Copy)]
pub enum Counter {
    /// Per kernel name (`addmm.out`), from the `native_call_<op>` profiling
    /// scope the generated kernel wrapper opens around the kernel call.
    OperatorCalls = 0,
    /// Every profiling scope the runtime opened (`OPERATOR_CALL` per kernel
    /// instruction, `DELEGATE_CALL` per delegate instruction, ...).
    Scopes = 1,
    /// Operator names a delegate's own profiler reported (R2).
    DelegateOps = 2,
}

/// An output, borrowed from the method's arena.
pub enum Value<'a> {
    Tensor(TensorView<'a>),
    Int(i64),
    Double(f64),
    Bool(bool),
    None,
}

pub struct TensorView<'a> {
    pub scalar_type: i8,
    pub sizes: Vec<i64>,
    pub dim_order: Vec<u8>,
    pub data: &'a [u8],
}

impl Method {
    pub fn name(&self) -> &str {
        &self.name
    }

    pub fn num_inputs(&self) -> usize {
        // SAFETY: live handle.
        unsafe { ffi::tn_et_method_num_inputs(self.raw.as_ptr()) }
    }

    pub fn num_outputs(&self) -> usize {
        // SAFETY: live handle.
        unsafe { ffi::tn_et_method_num_outputs(self.raw.as_ptr()) }
    }

    /// One copy: `data` is copied into the planned arena (or into a buffer
    /// the method owns, for an unplanned input) and not referenced after the
    /// call. The shim checks dtype, rank, sizes and byte count against the
    /// program before the runtime sees anything.
    pub fn set_input(&mut self, index: usize, scalar_type: i8, sizes: &[i64], data: &[u8]) -> Result<(), Error> {
        // SAFETY: sizes and data are valid for their lengths for the duration
        // of the call; the shim copies and keeps neither.
        let status = unsafe {
            ffi::tn_et_method_set_input(
                self.raw.as_ptr(),
                index,
                scalar_type,
                sizes.as_ptr(),
                sizes.len(),
                data.as_ptr().cast(),
                data.len(),
            )
        };
        check(status, || format!("{}: set_input({index})", self.name))
    }

    pub fn execute(&mut self) -> Result<(), Error> {
        // SAFETY: live handle, exclusive access through &mut self.
        let status = unsafe { ffi::tn_et_method_execute(self.raw.as_ptr()) };
        check(status, || format!("{}: execute", self.name))
    }

    pub fn output(&self, index: usize) -> Result<Value<'_>, Error> {
        let mut v = ffi::tn_et_value {
            tag: 0,
            scalar_type: 0,
            ndim: 0,
            sizes: std::ptr::null(),
            dim_order: std::ptr::null(),
            data: std::ptr::null(),
            nbytes: 0,
            int_value: 0,
            double_value: 0.0,
            bool_value: 0,
        };
        // SAFETY: live handle; v is writable.
        let status = unsafe { ffi::tn_et_method_output(self.raw.as_ptr(), index, &mut v) };
        check(status, || format!("{}: output({index})", self.name))?;
        Ok(match v.tag {
            1 => {
                // SAFETY: for a tensor output the shim points at the output
                // tensor's sizes, dim order and data in the method's arena,
                // valid until the next &mut call, which the returned borrow
                // of self prevents.
                let (sizes, dim_order, data) = unsafe {
                    let sizes = if v.ndim > 0 {
                        std::slice::from_raw_parts(v.sizes, v.ndim).iter().map(|s| *s as i64).collect()
                    } else {
                        Vec::new()
                    };
                    let dim_order =
                        if v.ndim > 0 { std::slice::from_raw_parts(v.dim_order, v.ndim).to_vec() } else { Vec::new() };
                    let data: &[u8] = if v.nbytes > 0 && !v.data.is_null() {
                        std::slice::from_raw_parts(v.data.cast(), v.nbytes)
                    } else {
                        &[]
                    };
                    (sizes, dim_order, data)
                };
                Value::Tensor(TensorView { scalar_type: v.scalar_type, sizes, dim_order, data })
            }
            4 => Value::Int(v.int_value),
            3 => Value::Double(v.double_value),
            5 => Value::Bool(v.bool_value != 0),
            0 => Value::None,
            other => {
                return Err(local(
                    NOT_SUPPORTED,
                    format!("{}: output({index})", self.name),
                    format!("output tag {other} is not returned by the binding"),
                ))
            }
        })
    }

    /// Successful executions since load or the last `reset_counters`.
    pub fn executions(&self) -> u64 {
        // SAFETY: live handle.
        unsafe { ffi::tn_et_method_executions(self.raw.as_ptr()) }
    }

    pub fn counters(&self, family: Counter) -> Vec<(String, u64)> {
        let raw = self.raw.as_ptr();
        // SAFETY: live handle; names are copied before any other call.
        let n = unsafe { ffi::tn_et_method_counter_count(raw, family as i32) };
        let mut out = Vec::with_capacity(n);
        for i in 0..n {
            let mut name: *const c_char = std::ptr::null();
            let mut count = 0u64;
            // SAFETY: as above.
            if unsafe { ffi::tn_et_method_counter_at(raw, family as i32, i, &mut name, &mut count) } == 0 {
                if let Some(name) = owned(name) {
                    out.push((name, count));
                }
            }
        }
        out
    }

    pub fn reset_counters(&mut self) {
        // SAFETY: live handle, exclusive access.
        unsafe { ffi::tn_et_method_reset_counters(self.raw.as_ptr()) }
    }
}
