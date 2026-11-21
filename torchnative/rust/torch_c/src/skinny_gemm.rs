//! The decode-shaped `float32` matmul on the CPU: a few activation rows against
//! a transposed weight, streamed once and split across the thread pool.
//!
//! **What this is for.** One decode step of a transformer is `F.linear` per
//! projection, which `bootstrap.py::linear` lowers to `t, view, mm,
//! _unsafe_view` (upstream's own no-bias N-D branch). The `mm` that reaches
//! `gemm_multiply` is therefore `(rows, k) @ t(weight)` with `rows` = batch x
//! 1 token: for SmolLM2-135M, 211 calls per step with `rows` of 1 or 4, `k` of
//! 576 or 1536 and `n` of 192, 576, 1536 or 49152. Gemstone's breakdown on an
//! AMD EPYC 7763 runner (issue #74, run 44792524320) put 90.8 ms of a 122.9 ms
//! torchnative step in `aten.mm` against 76.8 ms for upstream's whole step.
//!
//! **Why the general path is slow at these shapes.** On every non-Apple target
//! candle hands the multiply to the `gemm` crate (`cpu_backend/mod.rs`,
//! `cfg(not(any(feature = "mkl", feature = "accelerate")))`), and that path has
//! three properties that only matter when `rows` is tiny (read from
//! `gemm-common-0.19.0/src/gemm.rs`, `gemm-f32-0.19.0/src/microkernel.rs`):
//!
//! 1. `rows == 1` goes to `gemv::mixed_gemv_rowmajor`, which takes no
//!    `Parallelism` and runs on the calling thread alone.
//! 2. `rows > 1` goes to the blocked GEMM, which threads only when
//!    `m * n_chunk * k_chunk` reaches the threading threshold. `lib.rs` sets
//!    that to 4 M multiply-adds (docs/perf/PERF_ANDROID.md §4, measured on
//!    square, compute-bound shapes). Every decode projection of SmolLM2-135M
//!    at batch 4 is below it (`4 * 576 * 1536 = 3.5 M`), so 210 of the 211
//!    calls run on one thread; only `lm_head` (113 M) is split. The cost model
//!    is arithmetic, but a `(4, 576) @ (576, 1536)` call is a 3.5 MB weight
//!    stream with 4 multiply-adds per weight element: it is bound by memory
//!    bandwidth, and one core's share of it.
//! 3. The `fma` microkernel tile is `MR = 16` rows by up to `NR = 6` columns
//!    (`MR = 64` with AVX-512). With 4 rows, 12 of the 16 row lanes of every
//!    fused multiply-add are padding, and the `lhs` is pre-packed into that
//!    tile on every call.
//!
//! What the general path does *not* do, checked rather than assumed: it does
//! not copy the transposed weight (`rhs_rs == 1` is passed straight through,
//! and `do_pack_rhs` is false below `128 * MR` rows on x86), and the output
//! allocation is `rows * n` floats. So the two things to fix are threading and
//! lane use, and this module fixes both inside `gemm_multiply`, behind the
//! same `Tensor -> Tensor` contract the other kernels see.
//!
//! **The arithmetic, stated so a test can pin it.** For every output element
//! `(i, j)`, the `k` products `lhs[i, t] * rhs[t, j]` are accumulated into
//! `LANES` partial sums, product `t` into partial sum `t mod LANES`, in
//! increasing `t`, each step one fused multiply-add where the target has one
//! (`fma` on x86_64, always on aarch64) and a separate multiply and add
//! otherwise; the result is the partial sums added in lane order `0, 1, ...,
//! LANES - 1`. Three consequences a test checks:
//!
//! * an element's value depends on its row and column only, so an `m`-row
//!   call equals the `m` one-row calls bit for bit
//!   (`tests/ops/test_skinny_gemm.py`);
//! * the column split across threads changes which thread computes an
//!   element, never how, so the thread count cannot move a bit, and the
//!   instruction set cannot either (below);
//! * it is *not* `gemm`'s summation order, so against upstream (and against
//!   the path it replaces) the agreement is the golden harness's scale-aware
//!   GEMM bound, not bit equality. That is the grade upstream itself meets at
//!   these depths (`tests/golden/cases.py`, `_gemm_scale_check`, and the x86
//!   pins on the float32 matmul cases). Toy depths stay on the old path:
//!   `MIN_DEPTH` keeps every bit-exact golden case (`k` of 3 and 4) where it
//!   was.
//!
//! **What takes this path.** Both operands rank 2, `float32`, on the CPU,
//! `1 <= rows <= MAX_ROWS`, `k >= MIN_DEPTH`, the `lhs` rows contiguous
//! (column stride 1) and the `rhs` *columns* contiguous (row stride 1), which
//! is exactly the layout `t(weight)` has and a plain row-major `rhs` does not.
//! Everything else falls through to candle unchanged. The path is compiled on
//! every target and *routed* only where candle's matmul is the `gemm` crate:
//! on an Apple build with Accelerate, `sgemm` already threads and already
//! matches upstream at these shapes (docs/perf/LINEAR.md §5.1, 0.94 to 1.05x),
//! and this module has no measurement there, so it stays out of the way.
//!
//! **Instruction sets.** The kernel body is one generic function compiled for
//! three targets and chosen per call from `is_x86_feature_detected` (cached by
//! the standard library): baseline (separate multiply and add), `avx2,fma`,
//! and `avx512f,avx512dq` (x86_64). The AVX-512 body keeps *the same
//! arithmetic* as the others, so its bits equal theirs: instead of 16 partial
//! sums per element (which would pair lanes `l` and `l + 8` and change the
//! rounding) a 512-bit register holds the 8 partial sums of *two columns*, one
//! per 256-bit half, and four columns are in flight per step (two registers
//! per row). That halves the multiply-add instructions, gives a single-row
//! call two independent dependency chains instead of one, and reuses each
//! broadcast `lhs` load across two column pairs. `TORCHNATIVE_SKINNY_GEMM_SIMD`
//! (`avx2`, `baseline`, `avx512`, default automatic) forces a lower set for an
//! A/B on the same binary; asking for a set the CPU lacks falls back to the
//! best one it has, and the counters report what ran.
//!
//! **Threading.** A persistent pool of this module's own (the workers spin
//! briefly, then park, like candle's `BarrierPool`, and a call wakes only the
//! workers it uses), not candle's pool: candle sizes its pool with
//! `num_cpus::get_physical()`, which on Linux sums the `cpu cores` lines of
//! `/proc/cpuinfo` and knows nothing of cgroup quotas or affinity masks; a
//! virtual machine that reports fewer `cpu cores` than it has vCPUs is
//! under-threaded by that number, and this pool is a separate one so that
//! the quantised matmul's pool is left alone. The pool size (`threads`):
//! `TORCHNATIVE_SKINNY_GEMM_THREADS` if set, else `CANDLE_NUM_THREADS` if set
//! (the one knob candle documents; `1` means no split), else on Linux the
//! larger of candle's physical-core answer and
//! `std::thread::available_parallelism()` (which honours affinity and cgroup
//! quotas), elsewhere candle's answer alone (on Apple and Android the
//! available-parallelism count includes efficiency cores, which a streaming
//! split should not wait for). A call is split only when the weight it
//! streams is at least `TORCHNATIVE_SKINNY_GEMM_THREAD_BYTES` (default
//! `DEFAULT_THREAD_BYTES`), and then over at most one thread per
//! `TORCHNATIVE_SKINNY_GEMM_SHARE_BYTES` (default `DEFAULT_SHARE_BYTES`) of
//! weight, so a 442 KB projection does not wake every core of a 64-core
//! machine for a few microseconds of work each. Workers own disjoint column
//! ranges (a multiple of four columns, the AVX-512 group) and the calling
//! thread is the last. Both byte defaults are readings of the wake cost
//! against streaming time, **not measurements**; the environment variables
//! exist so the CI bench (`tests/bench/bench_skinny_mm.py`) can sweep them
//! without a rebuild, as `BW_GEMM_THREADING_THRESHOLD` does for `gemm`.
//!
//! **Counters, because a faster answer is not evidence that this ran**
//! (AGENTS.md §16). `_C._skinny_gemm_counters()` reports whether the path is
//! built and routed, how many calls it took, how many of those were split
//! across the pool, and the instruction set the kernel was dispatched to; the
//! suite asserts on the counts, not on timings. `TORCHNATIVE_SKINNY_GEMM=0`
//! turns the routing off for an A/B against the general path.

use candle_core::{CpuStorage, Device, DType, Storage, Tensor};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, AtomicUsize, Ordering};

/// The most activation rows this kernel takes. Eight 8-lane accumulators is
/// the register budget of one AVX2 tile; above this the general GEMM's tile
/// is no longer mostly padding and it is the better kernel.
pub const MAX_ROWS: usize = 8;

/// The least reduction depth this kernel takes. Below it a GEMM is a handful
/// of flops where the general path is already at parity, and it is where the
/// golden harness's bit-exact matmul cases live (`k` of 3 and 4): those must
/// keep computing exactly what they compute today.
pub const MIN_DEPTH: usize = 32;

/// Partial sums per output element. Eight is one AVX2 register of `f32`,
/// two NEON registers, and is also what fixes the summation order stated in
/// the module docs, so it is part of the arithmetic and not a tuning knob.
const LANES: usize = 8;

/// The default weight size, in bytes streamed, from which a call is split
/// across the pool. `k * n * 4`: 256 KiB is below SmolLM2-135M's smallest
/// projection (`k/v_proj`, 192 x 576 x 4 = 442 KB), so every call of that
/// model is split. Not measured; see the module docs.
pub const DEFAULT_THREAD_BYTES: usize = 256 * 1024;

/// The default weight size, in bytes, one thread of a split call is given at
/// least. A call uses `min(threads, bytes / share)` participants, never fewer
/// than two once it is split. Not measured; see the module docs.
pub const DEFAULT_SHARE_BYTES: usize = 128 * 1024;

static ENABLED: AtomicBool = AtomicBool::new(true);
static THREAD_BYTES: AtomicUsize = AtomicUsize::new(DEFAULT_THREAD_BYTES);
static SHARE_BYTES: AtomicUsize = AtomicUsize::new(DEFAULT_SHARE_BYTES);
static CALLS: AtomicU64 = AtomicU64::new(0);
static THREADED_CALLS: AtomicU64 = AtomicU64::new(0);
/// 0 automatic, 1 baseline, 2 avx2, 3 avx512: a ceiling, never a promise.
static SIMD_CEILING: AtomicU8 = AtomicU8::new(0);
/// 0 means automatic (see `configured_threads`).
static THREADS_OVERRIDE: AtomicUsize = AtomicUsize::new(0);

fn env_usize(name: &str) -> Option<usize> {
    std::env::var(name).ok().and_then(|raw| raw.trim().parse::<usize>().ok())
}

/// Read the environment switches once, at module initialisation, so no
/// per-call `getenv` sits on the 211-calls-per-step path.
pub fn apply_env() {
    if let Ok(raw) = std::env::var("TORCHNATIVE_SKINNY_GEMM") {
        ENABLED.store(!matches!(raw.as_str(), "" | "0"), Ordering::Relaxed);
    }
    if let Some(value) = env_usize("TORCHNATIVE_SKINNY_GEMM_THREAD_BYTES") {
        THREAD_BYTES.store(value, Ordering::Relaxed);
    }
    if let Some(value) = env_usize("TORCHNATIVE_SKINNY_GEMM_SHARE_BYTES") {
        SHARE_BYTES.store(value, Ordering::Relaxed);
    }
    if let Some(value) = env_usize("TORCHNATIVE_SKINNY_GEMM_THREADS") {
        THREADS_OVERRIDE.store(value, Ordering::Relaxed);
    }
    if let Ok(raw) = std::env::var("TORCHNATIVE_SKINNY_GEMM_SIMD") {
        let ceiling = match raw.trim().to_ascii_lowercase().as_str() {
            "baseline" | "portable" => 1,
            "avx2" => 2,
            "avx512" => 3,
            _ => 0,
        };
        SIMD_CEILING.store(ceiling, Ordering::Relaxed);
    }
}

/// Is candle's CPU matmul the `gemm` crate on this build? Accelerate and MKL
/// builds keep their BLAS for every shape (module docs). Answered through
/// candle's own feature report rather than a target cfg, so the Cargo table
/// that chooses the BLAS stays the single statement of that choice.
pub fn routed() -> bool {
    !candle_core::utils::has_accelerate() && !candle_core::utils::has_mkl()
}

/// The kernel body a call is handed to, best first.
#[derive(Clone, Copy, PartialEq, Eq)]
#[allow(dead_code)] // each target constructs only its own variants
enum Isa {
    Avx512,
    Avx2Fma,
    /// x86 without `avx2` + `fma`, or any target with no dedicated body here:
    /// separate multiply and add.
    Baseline,
    /// aarch64: NEON with fused multiply-add, from the compiler's own
    /// vectorisation of the portable body.
    Neon,
}

/// Which body this machine and `TORCHNATIVE_SKINNY_GEMM_SIMD` select. The
/// feature probes are atomic loads of the standard library's cache.
#[inline]
fn isa() -> Isa {
    #[cfg(target_arch = "x86_64")]
    {
        let ceiling = SIMD_CEILING.load(Ordering::Relaxed);
        if (ceiling == 0 || ceiling >= 3)
            && is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("fma")
        {
            return Isa::Avx512;
        }
        if (ceiling == 0 || ceiling >= 2) && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Isa::Avx2Fma;
        }
        return Isa::Baseline;
    }
    #[cfg(target_arch = "x86")]
    {
        let ceiling = SIMD_CEILING.load(Ordering::Relaxed);
        if (ceiling == 0 || ceiling >= 2) && is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Isa::Avx2Fma;
        }
        return Isa::Baseline;
    }
    #[cfg(target_arch = "aarch64")]
    {
        return Isa::Neon;
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    {
        Isa::Baseline
    }
}

/// Which kernel body `dispatch_rows` hands a call to on this machine.
fn simd_name() -> &'static str {
    match isa() {
        Isa::Avx512 => "avx512f+dq+fma",
        Isa::Avx2Fma => "avx2+fma",
        Isa::Neon => "neon+fma",
        Isa::Baseline => {
            if cfg!(any(target_arch = "x86_64", target_arch = "x86")) {
                "x86 baseline (separate multiply and add)"
            } else {
                "portable (separate multiply and add)"
            }
        }
    }
}

/// The widest body this CPU could run, ignoring `TORCHNATIVE_SKINNY_GEMM_SIMD`.
fn simd_best_available() -> &'static str {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512dq")
            && is_x86_feature_detected!("fma")
        {
            return "avx512f+dq+fma";
        }
    }
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return "avx2+fma";
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return "neon+fma";
    }
    #[allow(unreachable_code)]
    "baseline"
}

/// How many threads (callers included) a split call may use, and where the
/// number came from. Resolved once; the policy is in the module docs.
fn configured_threads() -> (usize, &'static str) {
    static CACHED: std::sync::OnceLock<(usize, &'static str)> = std::sync::OnceLock::new();
    *CACHED.get_or_init(|| {
        let forced = THREADS_OVERRIDE.load(Ordering::Relaxed);
        if forced > 0 {
            return (forced, "TORCHNATIVE_SKINNY_GEMM_THREADS");
        }
        if let Some(n) = env_usize("CANDLE_NUM_THREADS").filter(|&n| n > 0) {
            return (n, "CANDLE_NUM_THREADS");
        }
        let physical = candle_core::utils::get_num_threads().max(1);
        #[cfg(target_os = "linux")]
        {
            if let Ok(avail) = std::thread::available_parallelism() {
                if avail.get() > physical {
                    return (avail.get(), "available_parallelism");
                }
            }
        }
        (physical, "physical cores")
    })
}

/// A persistent pool for the column split. Flat fan-out: the caller signals the
/// workers it needs, runs the last slice itself, and waits for them. Workers
/// spin briefly after a job (consecutive projections of one step arrive
/// within microseconds) and then park.
mod pool {
    use std::any::Any;
    use std::cell::UnsafeCell;
    use std::hint::spin_loop;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex, OnceLock};
    use std::thread::Thread;

    /// One worker's mailbox, on its own cache line.
    #[repr(C, align(64))]
    struct Slot {
        go: AtomicUsize,
        done: AtomicUsize,
    }

    struct Work {
        trampoline: unsafe fn(*const (), usize),
        data: *const (),
    }

    struct Inner {
        slots: Box<[Slot]>,
        work: UnsafeCell<Work>,
        panic_payload: Mutex<Option<Box<dyn Any + Send>>>,
    }

    // SAFETY: `work` is written only by the caller holding `call_lock`, before
    // it `Release`-stores the generation into a slot, and read by a worker only
    // after its `Acquire` load of that generation; the caller does not write it
    // again until every signalled worker has `Release`-stored `done`.
    unsafe impl Sync for Inner {}
    unsafe impl Send for Inner {}

    pub struct Pool {
        inner: Arc<Inner>,
        handles: Vec<Thread>,
        call_lock: Mutex<usize>,
    }

    unsafe fn call_trampoline<F: Fn(usize)>(data: *const (), tid: usize) {
        (*(data as *const F))(tid);
    }

    const SPIN_LIMIT: u32 = 10_000;

    impl Pool {
        /// `workers` threads besides the caller. A thread that cannot be
        /// spawned ends the list there: the pool is then smaller, not absent.
        fn new(workers: usize) -> Pool {
            let inner = Arc::new(Inner {
                slots: (0..workers)
                    .map(|_| Slot { go: AtomicUsize::new(0), done: AtomicUsize::new(0) })
                    .collect::<Vec<_>>()
                    .into_boxed_slice(),
                work: UnsafeCell::new(Work { trampoline: |_, _| {}, data: std::ptr::null() }),
                panic_payload: Mutex::new(None),
            });
            let mut handles = Vec::with_capacity(workers);
            for tid in 0..workers {
                let shared = Arc::clone(&inner);
                let spawned = std::thread::Builder::new()
                    .name(format!("tn-skinny-{tid}"))
                    .spawn(move || {
                        let slot = &shared.slots[tid];
                        let mut gen = 0usize;
                        loop {
                            let mut spins = 0u32;
                            loop {
                                let g = slot.go.load(Ordering::Acquire);
                                if g != gen {
                                    gen = g;
                                    break;
                                }
                                if spins < SPIN_LIMIT {
                                    spins += 1;
                                    spin_loop();
                                } else {
                                    // An `unpark` before `park` leaves a token, so no wake is lost.
                                    std::thread::park();
                                    spins = 0;
                                }
                            }
                            let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                                // SAFETY: see `unsafe impl Sync for Inner`.
                                let work = unsafe { &*shared.work.get() };
                                unsafe { (work.trampoline)(work.data, tid) };
                            }));
                            if let Err(payload) = outcome {
                                let mut slot_payload = shared.panic_payload.lock().unwrap_or_else(|e| e.into_inner());
                                if slot_payload.is_none() {
                                    *slot_payload = Some(payload);
                                }
                            }
                            slot.done.store(gen, Ordering::Release);
                        }
                    });
                match spawned {
                    Ok(handle) => handles.push(handle.thread().clone()),
                    Err(_) => break,
                }
            }
            Pool { inner, handles, call_lock: Mutex::new(0) }
        }

        /// Workers this pool really has.
        pub fn workers(&self) -> usize {
            self.handles.len()
        }

        /// Run `f(tid)` for `tid` in `0..=active` (`active` workers, the
        /// caller as `tid == active`), returning when all have finished. A
        /// panic on any thread is re-raised here after every worker is done.
        pub fn execute<F: Fn(usize) + Sync>(&self, active: usize, f: F) {
            let active = active.min(self.handles.len());
            if active == 0 {
                f(0);
                return;
            }
            let mut guard = self.call_lock.lock().unwrap_or_else(|e| e.into_inner());
            let gen = guard.wrapping_add(1);
            *guard = gen;
            // SAFETY: no worker reads `work` between jobs (see `Inner`).
            unsafe {
                let work = &mut *self.inner.work.get();
                work.trampoline = call_trampoline::<F>;
                work.data = &f as *const F as *const ();
            }
            for w in 0..active {
                self.inner.slots[w].go.store(gen, Ordering::Release);
                self.handles[w].unpark();
            }
            let _wait = Wait { inner: &self.inner, active, gen };
            f(active);
        }
    }

    /// Blocks until the signalled workers are done, also while unwinding, so
    /// the closure they borrow outlives them.
    struct Wait<'a> {
        inner: &'a Inner,
        active: usize,
        gen: usize,
    }

    impl Drop for Wait<'_> {
        fn drop(&mut self) {
            for w in 0..self.active {
                let mut spins = 0u32;
                while self.inner.slots[w].done.load(Ordering::Acquire) != self.gen {
                    spins = spins.saturating_add(1);
                    if spins < 1 << 14 {
                        spin_loop();
                    } else {
                        std::thread::yield_now();
                    }
                }
            }
            let payload = self.inner.panic_payload.lock().unwrap_or_else(|e| e.into_inner()).take();
            if !std::thread::panicking() {
                if let Some(p) = payload {
                    std::panic::resume_unwind(p);
                }
            }
        }
    }

    static POOL: OnceLock<Pool> = OnceLock::new();

    /// The process-wide pool, built on first use with `participants - 1` workers.
    pub fn get(participants: usize) -> &'static Pool {
        POOL.get_or_init(|| Pool::new(participants.saturating_sub(1)))
    }
}

/// `(rows, k) @ (k, n)` through this kernel, or `None` when the operands are
/// not a shape or layout it claims, in which case the caller continues to
/// candle exactly as before. Every `None` below is a fall-through, never an
/// error: a shape this kernel does not take is not a shape that is wrong.
pub fn try_mm(lhs: &Tensor, rhs: &Tensor) -> Option<candle_core::Result<Tensor>> {
    if !ENABLED.load(Ordering::Relaxed) || !routed() {
        return None;
    }
    if lhs.dtype() != DType::F32 || rhs.dtype() != DType::F32 {
        return None;
    }
    if !matches!(lhs.device(), Device::Cpu) || !matches!(rhs.device(), Device::Cpu) {
        return None;
    }
    let (m, k) = lhs.dims2().ok()?;
    let (k2, n) = rhs.dims2().ok()?;
    if k != k2 || m == 0 || n == 0 || m > MAX_ROWS || k < MIN_DEPTH {
        return None;
    }
    let lhs_stride = lhs.layout().stride();
    let rhs_stride = rhs.layout().stride();
    if lhs_stride[1] != 1 || rhs_stride[0] != 1 {
        return None;
    }
    let (lhs_rs, rhs_cs) = (lhs_stride[0], rhs_stride[1]);

    let (lhs_guard, lhs_layout) = lhs.storage_and_layout();
    let (rhs_guard, rhs_layout) = rhs.storage_and_layout();
    let a = match &*lhs_guard {
        Storage::Cpu(CpuStorage::F32(v)) => v.get(lhs_layout.start_offset()..)?,
        _ => return None,
    };
    let b = match &*rhs_guard {
        Storage::Cpu(CpuStorage::F32(v)) => v.get(rhs_layout.start_offset()..)?,
        _ => return None,
    };
    // The slices every row and column read must exist. candle's layouts are
    // in bounds by construction; this is the check that makes the reads below
    // safe code rather than a promise.
    if a.len() < (m - 1) * lhs_rs + k || b.len() < (n - 1) * rhs_cs + k {
        return None;
    }

    let mut dst = vec![0f32; m * n];
    let bytes = k.saturating_mul(n).saturating_mul(std::mem::size_of::<f32>());
    let (threads, _) = configured_threads();
    // Participants of this call: the configured threads, at most one per
    // `SHARE_BYTES` of weight, at most one per column. Fewer than two is no split.
    let share = SHARE_BYTES.load(Ordering::Relaxed);
    let by_bytes = if share == 0 { usize::MAX } else { (bytes / share).max(1) };
    let mut participants = threads.min(by_bytes).min(n);
    let mut pool = None;
    if participants > 1 && bytes >= THREAD_BYTES.load(Ordering::Relaxed) {
        // The pool may have fewer workers than asked for if a spawn failed.
        let p = pool::get(threads);
        participants = participants.min(p.workers() + 1);
        if participants > 1 {
            pool = Some(p);
        }
    }

    CALLS.fetch_add(1, Ordering::Relaxed);
    let dst_ptr = dst.as_mut_ptr() as usize;
    let isa = isa();
    match pool {
        None => dispatch_rows(isa, m, k, n, a, lhs_rs, b, rhs_cs, dst_ptr, 0, n),
        Some(pool) => {
            THREADED_CALLS.fetch_add(1, Ordering::Relaxed);
            // Whole groups of four columns per thread (the AVX-512 body's group).
            let cols_per = n.div_ceil(participants).next_multiple_of(4);
            pool.execute(participants - 1, |tid| {
                let start = tid * cols_per;
                if start >= n {
                    return;
                }
                let end = n.min(start + cols_per);
                // SAFETY of the writes inside: worker `tid` owns columns
                // `start..end` of every row of `dst` and no other worker writes
                // them; `dst` outlives `execute`, which joins every worker before
                // it returns; nothing reads `dst` until then.
                dispatch_rows(isa, m, k, n, a, lhs_rs, b, rhs_cs, dst_ptr, start, end);
            });
        }
    }
    Some(Tensor::from_vec(dst, (m, n), &Device::Cpu))
}

/// Pick the row-count instantiation and the instruction set.
///
/// The row count is a const generic so the accumulators are a fixed-size
/// array the compiler keeps in registers; the match is the whole cost of that
/// choice. The x86_64 body is compiled three times (AVX-512, `avx2,fma`, and
/// the baseline the wheel is built for, because `f32::mul_add` without the
/// `fma` feature is a library call); `isa` has already checked that the CPU
/// has what the chosen body enables.
#[allow(clippy::too_many_arguments)]
fn dispatch_rows(
    isa: Isa,
    m: usize,
    k: usize,
    n: usize,
    lhs: &[f32],
    lhs_rs: usize,
    rhs: &[f32],
    rhs_cs: usize,
    dst: usize,
    col_start: usize,
    col_end: usize,
) {
    macro_rules! rows {
        ($M:literal) => {{
            match isa {
                #[cfg(target_arch = "x86_64")]
                // SAFETY: `isa()` returned `Avx512` only after detecting
                // `avx512f`, `avx512dq` and `fma` on this CPU.
                Isa::Avx512 => unsafe {
                    columns_avx512::<$M>(k, n, lhs, lhs_rs, rhs, rhs_cs, dst, col_start, col_end)
                },
                #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
                // SAFETY: `isa()` returned `Avx2Fma` only after detecting both.
                Isa::Avx2Fma => unsafe {
                    columns_avx2_fma::<$M>(k, n, lhs, lhs_rs, rhs, rhs_cs, dst, col_start, col_end)
                },
                Isa::Neon => columns::<$M, true>(k, n, lhs, lhs_rs, rhs, rhs_cs, dst, col_start, col_end),
                _ => columns::<$M, false>(k, n, lhs, lhs_rs, rhs, rhs_cs, dst, col_start, col_end),
            }
        }};
    }
    match m {
        1 => rows!(1),
        2 => rows!(2),
        3 => rows!(3),
        4 => rows!(4),
        5 => rows!(5),
        6 => rows!(6),
        7 => rows!(7),
        8 => rows!(8),
        _ => unreachable!("skinny_gemm: try_mm admits 1..={MAX_ROWS} rows, got {m}"),
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn columns_avx2_fma<const M: usize>(
    k: usize,
    n: usize,
    lhs: &[f32],
    lhs_rs: usize,
    rhs: &[f32],
    rhs_cs: usize,
    dst: usize,
    col_start: usize,
    col_end: usize,
) {
    columns::<M, true>(k, n, lhs, lhs_rs, rhs, rhs_cs, dst, col_start, col_end)
}

/// The AVX-512 body. Same arithmetic as `columns` (module docs): a 512-bit
/// register carries the 8 partial sums of two columns, the low 256 bits
/// column `a` and the high 256 bits column `b`, so lane `l` of each half
/// still receives products `t` with `t mod 8 == l` in increasing `t`, one
/// fused multiply-add each, and the reduction below adds lanes `0..8` in
/// order. Four columns per step (two registers per row): a `lhs` load is
/// broadcast to both halves and shared by two column pairs. A range that does
/// not end on a multiple of four repeats its last column in the unused
/// positions and does not store them.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f,avx512dq,avx2,fma")]
#[allow(clippy::too_many_arguments)]
unsafe fn columns_avx512<const M: usize>(
    k: usize,
    n: usize,
    lhs: &[f32],
    lhs_rs: usize,
    rhs: &[f32],
    rhs_cs: usize,
    dst: usize,
    col_start: usize,
    col_end: usize,
) {
    use std::arch::x86_64::*;
    if col_start >= col_end {
        return;
    }
    let rows: [&[f32]; M] = std::array::from_fn(|i| &lhs[i * lhs_rs..i * lhs_rs + k]);
    let full = k / LANES * LANES;
    let last = col_end - 1;
    let out = dst as *mut f32;
    let mut j0 = col_start;
    while j0 < col_end {
        let cj: [usize; 4] = std::array::from_fn(|q| (j0 + q).min(last));
        let cols: [&[f32]; 4] = std::array::from_fn(|q| &rhs[cj[q] * rhs_cs..cj[q] * rhs_cs + k]);
        let mut acc = [[_mm512_setzero_ps(); 2]; M];
        let mut t = 0;
        while t < full {
            // SAFETY: `t + LANES <= full <= k` and every slice holds `k` floats.
            let (c0, c1) = unsafe {
                let pair = |a: &[f32], b: &[f32]| {
                    _mm512_insertf32x8::<1>(
                        _mm512_castps256_ps512(_mm256_loadu_ps(a.as_ptr().add(t))),
                        _mm256_loadu_ps(b.as_ptr().add(t)),
                    )
                };
                (pair(cols[0], cols[1]), pair(cols[2], cols[3]))
            };
            for i in 0..M {
                // SAFETY: as above.
                let a = unsafe { _mm512_broadcast_f32x8(_mm256_loadu_ps(rows[i].as_ptr().add(t))) };
                acc[i][0] = _mm512_fmadd_ps(a, c0, acc[i][0]);
                acc[i][1] = _mm512_fmadd_ps(a, c1, acc[i][1]);
            }
            t += LANES;
        }
        for i in 0..M {
            for p in 0..2 {
                let mut lanes = [0f32; 2 * LANES];
                // SAFETY: `lanes` holds 16 floats.
                unsafe { _mm512_storeu_ps(lanes.as_mut_ptr(), acc[i][p]) };
                let (ca, cb) = (cols[2 * p], cols[2 * p + 1]);
                // The tail keeps the rule "product t goes to lane t mod LANES".
                for (l, t) in (full..k).enumerate() {
                    lanes[l] = rows[i][t].mul_add(ca[t], lanes[l]);
                    lanes[LANES + l] = rows[i][t].mul_add(cb[t], lanes[LANES + l]);
                }
                for half in 0..2 {
                    let j = j0 + 2 * p + half;
                    if j >= col_end {
                        continue;
                    }
                    let mut s = lanes[half * LANES];
                    for l in 1..LANES {
                        s += lanes[half * LANES + l];
                    }
                    // SAFETY: `dst` is the `m * n` float buffer `try_mm`
                    // allocated, `i < M == m` and `j < col_end <= n`; the
                    // caller guarantees no other thread writes column `j`.
                    unsafe { *out.add(i * n + j) = s };
                }
            }
        }
        j0 += 4;
    }
}

/// One multiply-add step of the stated arithmetic: fused where `FMA`, else a
/// multiply rounded once and an add rounded once, which is what `gemm`'s
/// own fallback kernels do on a machine without the instruction.
#[inline(always)]
fn madd<const FMA: bool>(a: f32, b: f32, c: f32) -> f32 {
    if FMA {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// Columns `col_start..col_end` of the product, every row, in the order the
/// module docs state. `#[inline(always)]` so that the `avx2,fma` wrapper
/// compiles this body with those features on and the lane loops become
/// vector instructions; the body itself is safe code over slices.
#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn columns<const M: usize, const FMA: bool>(
    k: usize,
    n: usize,
    lhs: &[f32],
    lhs_rs: usize,
    rhs: &[f32],
    rhs_cs: usize,
    dst: usize,
    col_start: usize,
    col_end: usize,
) {
    let rows: [&[f32]; M] = std::array::from_fn(|i| &lhs[i * lhs_rs..i * lhs_rs + k]);
    let full = k / LANES * LANES;
    for j in col_start..col_end {
        let col = &rhs[j * rhs_cs..j * rhs_cs + k];
        let mut acc = [[0f32; LANES]; M];
        let mut t = 0;
        while t < full {
            let r: &[f32; LANES] = col[t..t + LANES].try_into().unwrap();
            for i in 0..M {
                let a: &[f32; LANES] = rows[i][t..t + LANES].try_into().unwrap();
                for l in 0..LANES {
                    acc[i][l] = madd::<FMA>(a[l], r[l], acc[i][l]);
                }
            }
            t += LANES;
        }
        // The tail keeps the rule "product t goes to lane t mod LANES".
        for (l, t) in (full..k).enumerate() {
            for i in 0..M {
                acc[i][l] = madd::<FMA>(rows[i][t], col[t], acc[i][l]);
            }
        }
        for i in 0..M {
            let mut s = acc[i][0];
            for l in 1..LANES {
                s += acc[i][l];
            }
            // SAFETY: `dst` is the `m * n` float buffer `try_mm` allocated and
            // `i * n + j < m * n` (`i < M == m`, `j < col_end <= n`); the
            // caller guarantees no other thread writes column `j`.
            unsafe { *(dst as *mut f32).add(i * n + j) = s };
        }
    }
}

/// `_C._skinny_gemm_counters()` -- the runtime answer to "did the decode
/// matmul take this kernel, and was it split?" Process-wide, never reset:
/// a test takes a bracket. `routed` is false on a BLAS build, where `calls`
/// stays 0 by construction; `threads` is the most participants (callers
/// included) a split call uses, `threads_source` says which rule chose it, and
/// `simd` is the kernel body a call is handed to now.
#[pyfunction]
#[pyo3(name = "_skinny_gemm_counters")]
fn skinny_gemm_counters(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("built", true)?;
    d.set_item("routed", routed())?;
    d.set_item("enabled", ENABLED.load(Ordering::Relaxed))?;
    d.set_item("calls", CALLS.load(Ordering::Relaxed))?;
    d.set_item("threaded_calls", THREADED_CALLS.load(Ordering::Relaxed))?;
    let (threads, source) = configured_threads();
    d.set_item("threads", threads)?;
    d.set_item("threads_source", source)?;
    d.set_item("thread_bytes", THREAD_BYTES.load(Ordering::Relaxed))?;
    d.set_item("share_bytes", SHARE_BYTES.load(Ordering::Relaxed))?;
    d.set_item("max_rows", MAX_ROWS)?;
    d.set_item("min_depth", MIN_DEPTH)?;
    d.set_item("lanes", LANES)?;
    d.set_item("simd", simd_name())?;
    d.set_item("simd_best_available", simd_best_available())?;
    Ok(d.into_any().unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(skinny_gemm_counters, m)?)?;
    Ok(())
}
