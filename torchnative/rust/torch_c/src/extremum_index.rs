//! One pass for the CPU index reductions: the `int64` index that `argmax`,
//! `argmin`, the whole-tensor `max` and `min` and the index half of `max.dim`
//! and `min.dim` report, for a `float32` or `float64` CPU tensor reduced
//! along its innermost contiguous axis.
//!
//! **The evidence.** Gemstone's batching breakdown on the Xeon 8573C runner
//! (run 44874740472, SmolLM2-135M, batch 4, torchnative 0.1.0b11; the
//! transformers continuous-batching manager with `sdpa_paged`): the one
//! `aten.argmax` over the `[4, 49152]` logits cost 1.24 ms of the 82.8 ms
//! decode step, while upstream's did not reach the table's tenth row at
//! 0.48 ms. `first_extremum_index` (aten.rs) composed the answer from candle
//! ops: `max_keepdim`, `broadcast_eq`, `argmax_keepdim` over the byte mask
//! and `to_dtype`, then for the NaN rule `ne` (`x != x`), `max_keepdim`,
//! `argmax_keepdim`, `to_dtype` and `where_cond`: about seven passes over
//! the slice and five allocations, each pass a scalar loop in candle's
//! `ReduceIndex`. That composition exists for Metal, whose reduction seeds
//! its fold with the finite lowest value and so cannot be asked for the
//! index directly (issue #29); on the CPU the slice can simply be read.
//!
//! **What this does.** The same answer in one pass over each slice, read in
//! place: the index of the first NaN if the slice holds one, else the index
//! of the first element no later element beats (`>` for the maximum, `<`
//! for the minimum, so ties keep the lower index; an all-`-inf` slice
//! answers 0; `-0.0` and `0.0` compare equal, so the first of them wins,
//! which is upstream's rule and `tests/ops/test_signedzero.py`'s). The
//! decision is made on the layout and dtype, never on the values, so a call
//! is either whole on this path or whole on the composition, and the
//! composition stays for every other case (another device, another dtype,
//! a reduced axis that is not the innermost contiguous one): it is the
//! reference `tests/ops/test_extremum_index.py` holds this path to, index
//! for index.
//!
//! `_C._extremum_index_counters()` reports how many calls each path took.

use candle_core::{CpuStorage, Device, Storage, Tensor};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use std::sync::atomic::{AtomicU64, Ordering};

static ONE_PASS_CALLS: AtomicU64 = AtomicU64::new(0);
static COMPOSED_CALLS: AtomicU64 = AtomicU64::new(0);

/// Counts a call the candle-op composition answered.
pub fn note_composed() {
    COMPOSED_CALLS.fetch_add(1, Ordering::Relaxed);
}

/// The `int64` index tensor with `dim` kept at extent 1 when `source` is a
/// `float32` or `float64` CPU tensor whose axis `dim` is the innermost
/// contiguous one (every later axis of extent 1, stride 1 on `dim`, a
/// contiguous layout); `None` for anything else. `max` picks the maximum,
/// otherwise the minimum.
pub fn try_index(source: &Tensor, dim: usize, max: bool) -> Option<candle_core::Result<Tensor>> {
    if !matches!(source.device(), Device::Cpu) || dim >= source.rank() {
        return None;
    }
    let dims = source.dims();
    let n = dims[dim];
    let total = source.elem_count();
    if n == 0 || total == 0 {
        return None;
    }
    if dims[dim + 1..].iter().any(|&d| d != 1) {
        return None;
    }
    let layout = source.layout();
    if !layout.is_contiguous() || (n > 1 && layout.stride()[dim] != 1) {
        return None;
    }
    let start = layout.start_offset();
    let out: Vec<i64> = {
        let (guard, _layout) = source.storage_and_layout();
        match &*guard {
            Storage::Cpu(CpuStorage::F32(data)) => scan(data.get(start..start + total)?, n, max),
            Storage::Cpu(CpuStorage::F64(data)) => scan(data.get(start..start + total)?, n, max),
            _ => return None,
        }
    };
    ONE_PASS_CALLS.fetch_add(1, Ordering::Relaxed);
    let mut shape = dims.to_vec();
    shape[dim] = 1;
    Some(Tensor::from_vec(out, shape, &Device::Cpu))
}

/// Each slice of `n` elements in turn: the first NaN's index, else the first
/// index no later element beats.
fn scan<T: Copy + PartialOrd>(data: &[T], n: usize, max: bool) -> Vec<i64> {
    data.chunks_exact(n)
        .map(|row| {
            let mut best = row[0];
            if best != best {
                return 0;
            }
            let mut at = 0usize;
            for (i, &s) in row.iter().enumerate().skip(1) {
                if s != s {
                    return i as i64;
                }
                let better = if max { s > best } else { s < best };
                if better {
                    best = s;
                    at = i;
                }
            }
            at as i64
        })
        .collect()
}

/// `one_pass_calls` and `composed_calls` since the module loaded.
#[pyfunction]
#[pyo3(name = "_extremum_index_counters")]
fn extremum_index_counters(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let d = PyDict::new(py);
    d.set_item("one_pass_calls", ONE_PASS_CALLS.load(Ordering::Relaxed))?;
    d.set_item("composed_calls", COMPOSED_CALLS.load(Ordering::Relaxed))?;
    Ok(d.into_any().unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(extremum_index_counters, m)?)?;
    Ok(())
}
