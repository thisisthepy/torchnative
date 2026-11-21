//! Decode-shaped scaled dot-product attention on the CPU: a few query rows
//! per head against the whole key/value cache, one fused pass per row.
//!
//! **What this is for.** One decode step of a transformer calls
//! `aten._scaled_dot_product_flash_attention_for_cpu` once per layer with a
//! query of one token per sequence (`q_len` 1, or the packed token count of a
//! continuous batch) against a cache of hundreds of keys. Gemstone's
//! breakdown on an Intel Xeon 8573C runner (issue #74, run 44874740472,
//! SmolLM2-135M `float32`, torchnative 0.1.0b11) put 6.4 to 6.7 ms of a 75 ms
//! batch-4 decode step in those 30 calls: about 220 us per call for about
//! 340 thousand multiply-adds of work, which is overhead, not arithmetic.
//!
//! **Where the 220 us go.** `aten.rs::sdpa_flash_cpu`'s default path spells
//! the attention out in candle ops: widen and `contiguous` each operand
//! (three copies when the query is the transposed `(batch, len, heads, dim)`
//! view it always is), repeat the key/value heads for grouped-query attention
//! (two more copies), transpose and copy the key again, a batched matmul
//! (one `gemm` call per batch x head pair), `affine`, the mask's
//! `broadcast_add`, `amax`, `broadcast_sub`, `exp`, `sum`, `broadcast_div`,
//! a second batched matmul, then `log`, `broadcast_add` and `squeeze` for the
//! logsumexp: about fifteen allocations and some seventy `gemm` entries per
//! call at batch 4, each sized for a (1 x 64) by (64 x 300) product.
//!
//! **What this module does instead.** For every `(batch, head, query row)` it
//! reads the query row and the cache rows in place (any strides on the batch,
//! head and row axes, so neither the transposed query nor a cache slice is
//! copied; grouped-query heads are an index mapping, never a repeat), forms
//! the `kv_len` scores, the row maximum, the exponentials and their sum, and
//! accumulates the normalised weights against the value rows into the output
//! row, writing the logsumexp beside it. Rows are independent, so a call is
//! split across candle's persistent `BarrierPool` (`utils::barrier_pool`, the
//! pool the quantised matmul and `sort` tile over) by rows when the work is
//! at least `TORCHNATIVE_SDPA_DECODE_THREAD_MADDS` multiply-adds.
//!
//! **The arithmetic, stated so a test can pin it.** Per row:
//!
//! 1. `score[j] = (q . k[j]) * scale + 0.0` for each key `j`, the dot product
//!    over the head dimension accumulated into `LANES` partial sums, product
//!    `t` into partial sum `t mod LANES`, each step a fused multiply-add where
//!    the target has one (`avx2,fma` on x86_64 detected at run time, always on
//!    aarch64) and a separate multiply and add otherwise, the partial sums
//!    added in lane order; `scale` as `float32`; `+ 0.0` is the `affine` of the
//!    general path (it turns a negative zero positive). Then `+ mask[b, h, i, j]`
//!    when there is a mask.
//! 2. `max` over the scores (NaN if any score is NaN, as `amax` answers),
//!    `w[j] = exp(score[j] - max)` with the standard library's `f32::exp`,
//!    `sum` of the `w` in key order, `p[j] = w[j] / sum`.
//! 3. `out[d] = sum over j in key order of p[j] * v[j][d]`, one fused
//!    multiply-add per term where the target has one; `logsumexp = ln(sum) + max`.
//!
//! Consequences a test checks (`tests/ops/test_sdpa_decode.py`): a row's
//! result depends on its own query, its head's cache and its mask row only,
//! so a batch-`b` call equals the `b` batch-1 calls bit for bit and a
//! `q_len`-`l` call equals the `l` one-row calls; the row split decides which
//! thread computes a row, never how, so the thread count cannot move a bit;
//! grouped-query attention equals the same call with the key/value heads
//! repeated. Against the general path, and against upstream, the agreement is
//! a bound (the general path sums the products in `gemm`'s order and the
//! weighted sum in a second GEMM), the same grade the golden harness holds the
//! general path to (`tests/golden/cases.py::_sdpa_pair_check`, 1e-5 for
//! `float32`). A fully masked row answers NaN as the general path does.
//!
//! **What takes this path.** Query, key and value rank 4, `float32`, on the
//! CPU, with the head dimension contiguous (stride 1), `1 <= q_len <=
//! MAX_QUERY_ROWS`, key length and head dimension at least 1, the query head
//! count a multiple of the key/value head count, `is_causal` false (at decode
//! the model passes an explicit mask, and the shim's causal rule is the
//! upper-left-aligned one that a one-row query against a long cache never
//! wants), no dropout, and a mask that is `None` or a rank-4 `float32` CPU
//! tensor whose every dimension is 1 or the full extent with its last
//! dimension contiguous. Everything else continues to the general path
//! unchanged. Routed only where candle's matmul is the `gemm` crate (every
//! non-Apple build): on an Accelerate build the general path's matmuls are
//! `sgemm` and `tests/_support/test_shim.py` holds that path to a measured
//! 1e-6 of upstream's Accelerate-backed answer, a bound this kernel was not
//! measured against. The reference kernel (`crate::flash`), when switched on,
//! keeps its path.
//!
//! **Counters, because a faster answer is not evidence this ran** (AGENTS.md
//! §16): `_C._sdpa_decode_counters()` reports whether the path is built and
//! routed, how many calls took it and how many were split; the suite asserts
//! on counts, not timings. `TORCHNATIVE_SDPA_DECODE=0` turns the routing off
//! for an A/B against the general path on the same binary.

use candle_core::{CpuStorage, Device, DType, Storage, Tensor};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// The most query rows per head this kernel takes. Decode is 1 per
/// sequence, a packed continuous batch a handful; above this the general
/// path's blocked GEMMs are the better shape.
pub const MAX_QUERY_ROWS: usize = 8;

/// Partial sums of the dot product over the head dimension: one AVX2
/// register of `f32`, two NEON registers. Part of the stated arithmetic.
const LANES: usize = 8;

/// Multiply-adds (`2 * rows * kv_len * head_dim`) from which a call is split
/// across the pool. SmolLM2-135M at batch 4 against 74 keys is 340 thousand;
/// one sequence against a 20-key cache is 23 thousand. Not measured; the
/// environment variable exists so the bench can sweep it without a rebuild.
pub const DEFAULT_THREAD_MADDS: usize = 1 << 16;

static ENABLED: AtomicBool = AtomicBool::new(true);
static THREAD_MADDS: AtomicUsize = AtomicUsize::new(DEFAULT_THREAD_MADDS);
static CALLS: AtomicU64 = AtomicU64::new(0);
static THREADED_CALLS: AtomicU64 = AtomicU64::new(0);

/// Read the switches once, at module initialisation.
pub fn apply_env() {
    if let Ok(raw) = std::env::var("TORCHNATIVE_SDPA_DECODE") {
        ENABLED.store(!matches!(raw.as_str(), "" | "0"), Ordering::Relaxed);
    }
    if let Some(value) = std::env::var("TORCHNATIVE_SDPA_DECODE_THREAD_MADDS")
        .ok()
        .and_then(|raw| raw.trim().parse::<usize>().ok())
    {
        THREAD_MADDS.store(value, Ordering::Relaxed);
    }
}

/// Is this kernel routed on this build? See the module docs.
pub fn routed() -> bool {
    !candle_core::utils::has_accelerate() && !candle_core::utils::has_mkl()
}

/// A rank-4 `float32` operand read in place: `data[b * s0 + h * s1 + r * s2 + d]`.
struct Operand<'a> {
    data: &'a [f32],
    s0: usize,
    s1: usize,
    s2: usize,
}

/// The mask, read in place with a zero stride on every broadcast axis.
struct Mask<'a> {
    data: &'a [f32],
    s: [usize; 4],
}

struct Problem<'a> {
    heads: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    n_rep: usize,
    q: Operand<'a>,
    k: Operand<'a>,
    v: Operand<'a>,
    mask: Option<Mask<'a>>,
    scale: f32,
    /// `rows * head_dim` floats, row `r` at `r * head_dim`.
    out: usize,
    /// `rows` floats.
    lse: usize,
}

/// The scaled dot-product attention of `q` against `k` and `v` through this
/// kernel, or `None` when the operands are not a shape or layout it claims,
/// in which case the caller continues to the general path exactly as before.
/// Every `None` is a fall-through, never an error.
pub fn try_attend(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
    is_causal: bool,
) -> Option<candle_core::Result<(Tensor, Tensor)>> {
    if !ENABLED.load(Ordering::Relaxed) || !routed() || is_causal || !scale.is_finite() {
        return None;
    }
    for t in [q, k, v].into_iter().chain(mask) {
        if t.dtype() != DType::F32 || !matches!(t.device(), Device::Cpu) || t.rank() != 4 {
            return None;
        }
    }
    let (b, h, l, d) = q.dims4().ok()?;
    let (kb, hk, s, kd) = k.dims4().ok()?;
    let (vb, vhk, vs, vd) = v.dims4().ok()?;
    if b == 0 || h == 0 || l == 0 || d == 0 || s == 0 || hk == 0 {
        return None;
    }
    if l > MAX_QUERY_ROWS || kb != b || vb != b || vhk != hk || vs != s || kd != d || vd != d || h % hk != 0 {
        return None;
    }
    let (qs, ks, vs_) = (q.layout().stride(), k.layout().stride(), v.layout().stride());
    if qs[3] != 1 || ks[3] != 1 || vs_[3] != 1 {
        return None;
    }
    let mask_strides = match mask {
        None => None,
        Some(m) => {
            let dims = m.dims();
            let st = m.layout().stride();
            let full = [b, h, l, s];
            let mut strides = [0usize; 4];
            for axis in 0..4 {
                if dims[axis] == 1 {
                    strides[axis] = 0;
                } else if dims[axis] == full[axis] {
                    strides[axis] = st[axis];
                } else {
                    return None;
                }
            }
            if dims[3] != 1 && strides[3] != 1 {
                return None;
            }
            Some(strides)
        }
    };

    // The storage read guards live in this frame until the kernel has run;
    // the slices borrow them and are read only.
    let (q_guard, q_layout) = q.storage_and_layout();
    let (k_guard, k_layout) = k.storage_and_layout();
    let (v_guard, v_layout) = v.storage_and_layout();
    let mask_held = mask.map(|m| m.storage_and_layout());
    let q_data = f32_slice(&q_guard, q_layout.start_offset())?;
    let k_data = f32_slice(&k_guard, k_layout.start_offset())?;
    let v_data = f32_slice(&v_guard, v_layout.start_offset())?;
    let mask_data = match &mask_held {
        None => None,
        Some((guard, layout)) => Some(f32_slice(guard, layout.start_offset())?),
    };
    // Every element a row reads must exist. candle's layouts are in bounds by
    // construction; this check is what makes the reads below safe code.
    let extent = |len: usize, st: &[usize], dims: [usize; 4]| -> bool {
        len >= (dims[0] - 1) * st[0] + (dims[1] - 1) * st[1] + (dims[2] - 1) * st[2] + dims[3]
    };
    if !extent(q_data.len(), qs, [b, h, l, d])
        || !extent(k_data.len(), ks, [b, hk, s, d])
        || !extent(v_data.len(), vs_, [b, hk, s, d])
    {
        return None;
    }
    let mask = match (mask_data, mask_strides) {
        (Some(data), Some(st)) => {
            let need = (b - 1) * st[0] + (h - 1) * st[1] + (l - 1) * st[2] + (s - 1) * st[3] + 1;
            if data.len() < need {
                return None;
            }
            Some(Mask { data, s: st })
        }
        _ => None,
    };

    let rows = b * h * l;
    let mut out = vec![0f32; rows * d];
    let mut lse = vec![0f32; rows];
    let problem = Problem {
        heads: h,
        q_len: l,
        kv_len: s,
        head_dim: d,
        n_rep: h / hk,
        q: Operand { data: q_data, s0: qs[0], s1: qs[1], s2: qs[2] },
        k: Operand { data: k_data, s0: ks[0], s1: ks[1], s2: ks[2] },
        v: Operand { data: v_data, s0: vs_[0], s1: vs_[1], s2: vs_[2] },
        mask,
        scale: scale as f32,
        out: out.as_mut_ptr() as usize,
        lse: lse.as_mut_ptr() as usize,
    };

    CALLS.fetch_add(1, Ordering::Relaxed);
    let madds = rows.saturating_mul(s).saturating_mul(d).saturating_mul(2);
    let pool = candle_core::utils::barrier_pool();
    let participants = pool.n_workers() + 1;
    if participants > 1 && rows >= 2 && madds >= THREAD_MADDS.load(Ordering::Relaxed) {
        THREADED_CALLS.fetch_add(1, Ordering::Relaxed);
        let rows_per = rows.div_ceil(participants);
        pool.execute(|tid| {
            let start = tid * rows_per;
            if start < rows {
                // SAFETY of the writes inside: thread `tid` owns rows
                // `start..end` of `out` and `lse` and no other thread writes
                // them; both buffers outlive `execute`, which joins every
                // worker before it returns; nothing reads them until then.
                dispatch(&problem, start, rows.min(start + rows_per));
            }
        });
    } else {
        dispatch(&problem, 0, rows);
    }
    Some(
        Tensor::from_vec(out, (b, h, l, d), &Device::Cpu)
            .and_then(|out| Tensor::from_vec(lse, (b, h, l), &Device::Cpu).map(|lse| (out, lse))),
    )
}

/// The `float32` CPU storage from `start`, or `None` for any other storage.
fn f32_slice(storage: &Storage, start: usize) -> Option<&[f32]> {
    match storage {
        Storage::Cpu(CpuStorage::F32(data)) => data.get(start..),
        _ => None,
    }
}

/// Which body this machine runs, for the counters.
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

/// Rows `row_start..row_end` through the body the CPU has.
fn dispatch(p: &Problem<'_>, row_start: usize, row_end: usize) {
    #[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
    {
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            // SAFETY: both features were just detected on this CPU.
            unsafe { rows_avx2_fma(p, row_start, row_end) };
        } else {
            rows::<false>(p, row_start, row_end);
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        rows::<true>(p, row_start, row_end);
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "x86", target_arch = "aarch64")))]
    {
        rows::<false>(p, row_start, row_end);
    }
}

#[cfg(any(target_arch = "x86_64", target_arch = "x86"))]
#[target_feature(enable = "avx2,fma")]
unsafe fn rows_avx2_fma(p: &Problem<'_>, row_start: usize, row_end: usize) {
    rows::<true>(p, row_start, row_end)
}

/// One step of the stated arithmetic: fused where `FMA`, else a multiply
/// rounded once and an add rounded once.
#[inline(always)]
fn madd<const FMA: bool>(a: f32, b: f32, c: f32) -> f32 {
    if FMA {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// `q . k` over the head dimension: `LANES` partial sums, product `t` into
/// sum `t mod LANES`, added in lane order.
#[inline(always)]
fn dot<const FMA: bool>(q: &[f32], k: &[f32]) -> f32 {
    let n = q.len();
    let full = n / LANES * LANES;
    let mut acc = [0f32; LANES];
    let mut t = 0;
    while t < full {
        let a: &[f32; LANES] = q[t..t + LANES].try_into().unwrap();
        let r: &[f32; LANES] = k[t..t + LANES].try_into().unwrap();
        for l in 0..LANES {
            acc[l] = madd::<FMA>(a[l], r[l], acc[l]);
        }
        t += LANES;
    }
    for (l, t) in (full..n).enumerate() {
        acc[l] = madd::<FMA>(q[t], k[t], acc[l]);
    }
    let mut s = acc[0];
    for l in 1..LANES {
        s += acc[l];
    }
    s
}

/// Rows `row_start..row_end` of the problem, in the order the module docs
/// state. `#[inline(always)]` so the `avx2,fma` wrapper compiles this body
/// with those features on; the body is safe code over slices except for the
/// two owned-row writes.
#[inline(always)]
fn rows<const FMA: bool>(p: &Problem<'_>, row_start: usize, row_end: usize) {
    let (l, s, d) = (p.q_len, p.kv_len, p.head_dim);
    let mut scores: Vec<f32> = Vec::with_capacity(s);
    let out = p.out as *mut f32;
    let lse = p.lse as *mut f32;
    for r in row_start..row_end {
        let bi = r / (p.heads * l);
        let rem = r % (p.heads * l);
        let (hi, i) = (rem / l, rem % l);
        let kh = hi / p.n_rep;
        let q_row = &p.q.data[bi * p.q.s0 + hi * p.q.s1 + i * p.q.s2..][..d];
        let k_base = bi * p.k.s0 + kh * p.k.s1;
        let v_base = bi * p.v.s0 + kh * p.v.s1;

        // 1. the scores
        scores.clear();
        let mut max = f32::NEG_INFINITY;
        let mut any_nan = false;
        for j in 0..s {
            let k_row = &p.k.data[k_base + j * p.k.s2..][..d];
            let mut score = dot::<FMA>(q_row, k_row) * p.scale + 0.0;
            if let Some(m) = &p.mask {
                score += m.data[bi * m.s[0] + hi * m.s[1] + i * m.s[2] + j * m.s[3]];
            }
            if score.is_nan() {
                any_nan = true;
            } else if score > max {
                max = score;
            }
            scores.push(score);
        }
        if any_nan {
            max = f32::NAN;
        }

        // 2. the weights and their sum
        let mut sum = 0f32;
        for score in scores.iter_mut() {
            let w = (*score - max).exp();
            *score = w;
            sum += w;
        }

        // 3. the weighted sum of the value rows, and the logsumexp
        // SAFETY: row `r` of `out` (`d` floats at `r * d`) and element `r` of
        // `lse` belong to this call's `row_start..row_end`, which no other
        // thread writes; both buffers hold `rows * d` and `rows` floats and
        // outlive this function (see `try_attend`).
        let out_row = unsafe { std::slice::from_raw_parts_mut(out.add(r * d), d) };
        for (j, &w) in scores.iter().enumerate() {
            let pj = w / sum;
            let v_row = &p.v.data[v_base + j * p.v.s2..][..d];
            for t in 0..d {
                out_row[t] = madd::<FMA>(pj, v_row[t], out_row[t]);
            }
        }
        unsafe { *lse.add(r) = sum.ln() + max };
    }
}

/// `_C._sdpa_decode_counters()`: the runtime answer to "did the decode
/// attention take this kernel, and was it split?" Process-wide, never reset:
/// a test takes a bracket. `routed` is false on a BLAS build, where `calls`
/// stays 0 by construction.
#[pyfunction]
#[pyo3(name = "_sdpa_decode_counters")]
fn sdpa_decode_counters(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("built", true)?;
    d.set_item("routed", routed())?;
    d.set_item("enabled", ENABLED.load(Ordering::Relaxed))?;
    d.set_item("calls", CALLS.load(Ordering::Relaxed))?;
    d.set_item("threaded_calls", THREADED_CALLS.load(Ordering::Relaxed))?;
    d.set_item("threads", candle_core::utils::barrier_pool().n_workers() + 1)?;
    d.set_item("thread_madds", THREAD_MADDS.load(Ordering::Relaxed))?;
    d.set_item("max_query_rows", MAX_QUERY_ROWS)?;
    d.set_item("lanes", LANES)?;
    d.set_item("simd", simd_name())?;
    Ok(d.into_any().unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(sdpa_decode_counters, m)?)?;
    Ok(())
}
