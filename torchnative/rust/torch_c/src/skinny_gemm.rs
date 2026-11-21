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
//!   element, never how, so the thread count cannot move a bit;
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
//! **Threading.** candle's persistent `BarrierPool` (`utils::barrier_pool`),
//! the pool the quantised matmul already uses for the same column-split
//! tiling (issue #177, `k_quants.rs`). Workers own disjoint column ranges of
//! the output and the calling thread is the last worker. A call is split only
//! when the weight it streams is at least `TORCHNATIVE_SKINNY_GEMM_THREAD_BYTES`
//! (default `DEFAULT_THREAD_BYTES`): below that the barrier costs more than
//! the stream. That default is a reading of the wake cost against a few tens
//! of microseconds of streaming, **not a measurement**; the environment
//! variable exists so the CI bench (`tests/bench/bench_skinny_mm.py`) can
//! sweep it without a rebuild, as `BW_GEMM_THREADING_THRESHOLD` does for `gemm`.
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
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

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

static ENABLED: AtomicBool = AtomicBool::new(true);
static THREAD_BYTES: AtomicUsize = AtomicUsize::new(DEFAULT_THREAD_BYTES);
static CALLS: AtomicU64 = AtomicU64::new(0);
static THREADED_CALLS: AtomicU64 = AtomicU64::new(0);

/// Read the two environment switches once, at module initialisation, so no
/// per-call `getenv` sits on the 211-calls-per-step path.
pub fn apply_env() {
    if let Ok(raw) = std::env::var("TORCHNATIVE_SKINNY_GEMM") {
        ENABLED.store(!matches!(raw.as_str(), "" | "0"), Ordering::Relaxed);
    }
    if let Some(value) = std::env::var("TORCHNATIVE_SKINNY_GEMM_THREAD_BYTES")
        .ok()
        .and_then(|raw| raw.parse::<usize>().ok())
    {
        THREAD_BYTES.store(value, Ordering::Relaxed);
    }
}

/// Is candle's CPU matmul the `gemm` crate on this build? Accelerate and MKL
/// builds keep their BLAS for every shape (module docs). Answered through
/// candle's own feature report rather than a target cfg, so the Cargo table
/// that chooses the BLAS stays the single statement of that choice.
pub fn routed() -> bool {
    !candle_core::utils::has_accelerate() && !candle_core::utils::has_mkl()
}

/// Which kernel body `dispatch_rows` hands a call to on this machine.
fn simd_name() -> &'static str {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return "avx2+fma";
        }
        return "x86 baseline (separate multiply and add)";
    }
    #[cfg(target_arch = "aarch64")]
    {
        "neon+fma"
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    {
        "portable (separate multiply and add)"
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
    let pool = candle_core::utils::barrier_pool();
    let participants = pool.n_workers() + 1;
    let split = participants > 1 && n >= participants && bytes >= THREAD_BYTES.load(Ordering::Relaxed);

    CALLS.fetch_add(1, Ordering::Relaxed);
    if !split {
        let dst_ptr = dst.as_mut_ptr() as usize;
        dispatch_rows(m, k, n, a, lhs_rs, b, rhs_cs, dst_ptr, 0, n);
    } else {
        THREADED_CALLS.fetch_add(1, Ordering::Relaxed);
        let cols_per = n.div_ceil(participants);
        let dst_ptr = dst.as_mut_ptr() as usize;
        pool.execute(|tid| {
            let start = tid * cols_per;
            if start >= n {
                return;
            }
            let end = n.min(start + cols_per);
            // SAFETY of the writes inside: worker `tid` owns columns
            // `start..end` of every row of `dst` and no other worker writes
            // them; `dst` outlives `execute`, which joins every worker before
            // it returns; nothing reads `dst` until then.
            dispatch_rows(m, k, n, a, lhs_rs, b, rhs_cs, dst_ptr, start, end);
        });
    }
    Some(Tensor::from_vec(dst, (m, n), &Device::Cpu))
}

/// Pick the row-count instantiation and the instruction set.
///
/// The row count is a const generic so the `M x LANES` accumulator is a
/// fixed-size array the compiler keeps in registers; the match is the whole
/// cost of that choice. The x86_64 body is compiled twice, once with
/// `avx2,fma` enabled for the machine that has them (detected once, cached by
/// the standard library) and once at the baseline the wheel is built for,
/// because `f32::mul_add` without the `fma` feature is a library call.
#[allow(clippy::too_many_arguments)]
fn dispatch_rows(
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
            #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
            {
                if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
                    // SAFETY: the target features the function enables were
                    // just detected on this CPU.
                    unsafe {
                        columns_avx2_fma::<$M>(k, n, lhs, lhs_rs, rhs, rhs_cs, dst, col_start, col_end)
                    }
                } else {
                    columns::<$M, false>(k, n, lhs, lhs_rs, rhs, rhs_cs, dst, col_start, col_end)
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                columns::<$M, true>(k, n, lhs, lhs_rs, rhs, rhs_cs, dst, col_start, col_end)
            }
            #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
            {
                columns::<$M, false>(k, n, lhs, lhs_rs, rhs, rhs_cs, dst, col_start, col_end)
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
/// stays 0 by construction; `threads` is the pool's participant count
/// (candle's `CANDLE_NUM_THREADS`, physical cores by default) and is what the
/// column split divides by.
#[pyfunction]
#[pyo3(name = "_skinny_gemm_counters")]
fn skinny_gemm_counters(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("built", true)?;
    d.set_item("routed", routed())?;
    d.set_item("enabled", ENABLED.load(Ordering::Relaxed))?;
    d.set_item("calls", CALLS.load(Ordering::Relaxed))?;
    d.set_item("threaded_calls", THREADED_CALLS.load(Ordering::Relaxed))?;
    d.set_item("threads", candle_core::utils::barrier_pool().n_workers() + 1)?;
    d.set_item("thread_bytes", THREAD_BYTES.load(Ordering::Relaxed))?;
    d.set_item("max_rows", MAX_ROWS)?;
    d.set_item("min_depth", MIN_DEPTH)?;
    d.set_item("lanes", LANES)?;
    d.set_item("simd", simd_name())?;
    Ok(d.into_any().unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(skinny_gemm_counters, m)?)?;
    Ok(())
}
