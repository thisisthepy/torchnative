//! Block quantisation -- candle's `QTensor`, seen from Python.
//!
//! **Why this is a module and not eight more ops in `aten.rs`.** There is no
//! aten operator this implements. Upstream's quantisation surface
//! (`torch.quantize_per_tensor`, `aten::_int_mm`, `aten::_weight_int8pack_mm`,
//! `aten::_dyn_quant_matmul_4bit`) names per-tensor-affine `int8` and KleidiAI
//! 4-bit packs, and docs/numerics/DTYPE.md §6.1 measured what the non-KleidiAI half of
//! that actually is: the pack casts every `uint8` nibble to `float32` and
//! concatenates, so it *grows* the weight. Wearing those names over a GGML
//! k-quant would be claiming a contract this does not honour. So the entrances
//! carry a leading underscore and no aten name, exactly as `_tensor_from_flat`
//! does, and docs/graph/QUANT2.md §7 records what it would take to earn the names.
//!
//! **What this buys and what it does not.** It replaces a leaf -- one
//! `nn.Linear` at a time, from Python, at run time. It does not fuse, does not
//! see a graph, and cannot reach anything a module boundary hides. That
//! ceiling is structural and is the reason docs/graph/DECOMP.md's path is not made
//! redundant by this one.
//!
//! The verification axis is docs/graph/QUANT2.md §2 and it is not a tolerance: the
//! Q8_0 and Q4_0 quantisers are reimplemented from the format in
//! `tests/_support/ggml_ref.py` and the blob compared **byte for byte**, and the
//! dequantiser is reimplemented for those two plus Q4K and the reconstruction
//! compared **bit for bit** as `float32`. What is left to a bound is only the
//! part that is genuinely lossy, and that bound has a floor as well as a
//! ceiling so that "no quantisation happened" fails it.
use std::borrow::Cow;
use std::sync::Arc;

use candle_core::quantized::{GgmlDType, QMatMul, QStorage, QTensor};
use candle_core::{Device, Module};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyModule};

use crate::dtype::TorchDType;
use crate::err::{candle_err, not_implemented};
use crate::tensor::PyTensorBase;

/// The spelling of a `GgmlDType` on the Python side.
///
/// GGML's own lower-case names, because the blob these produce is GGML's blob:
/// a `q4_k` tensor written out here has the byte layout `llama.cpp` reads, and
/// calling it something else would hide that. The `_k` suffix is spelled with
/// the underscore GGUF uses in its type names (`Q4_K`), not candle's Rust
/// identifier (`Q4K`).
pub fn format_name(dtype: GgmlDType) -> &'static str {
    match dtype {
        GgmlDType::F32 => "f32",
        GgmlDType::F16 => "f16",
        GgmlDType::BF16 => "bf16",
        GgmlDType::Q4_0 => "q4_0",
        GgmlDType::Q4_1 => "q4_1",
        GgmlDType::Q5_0 => "q5_0",
        GgmlDType::Q5_1 => "q5_1",
        GgmlDType::Q8_0 => "q8_0",
        GgmlDType::Q8_1 => "q8_1",
        GgmlDType::Q2K => "q2_k",
        GgmlDType::Q3K => "q3_k",
        GgmlDType::Q4K => "q4_k",
        GgmlDType::Q5K => "q5_k",
        GgmlDType::Q6K => "q6_k",
        GgmlDType::Q8K => "q8_k",
    }
}

/// Every format this build can hold, in the order `_quantized_formats()`
/// reports them. Derived from the `GgmlDType` enum rather than written twice:
/// a candle upgrade that adds a format makes `format_name` fail to compile,
/// and this list follows from it.
const ALL_FORMATS: [GgmlDType; 15] = [
    GgmlDType::F32,
    GgmlDType::F16,
    GgmlDType::BF16,
    GgmlDType::Q4_0,
    GgmlDType::Q4_1,
    GgmlDType::Q5_0,
    GgmlDType::Q5_1,
    GgmlDType::Q8_0,
    GgmlDType::Q8_1,
    GgmlDType::Q2K,
    GgmlDType::Q3K,
    GgmlDType::Q4K,
    GgmlDType::Q5K,
    GgmlDType::Q6K,
    GgmlDType::Q8K,
];

/// The inverse, refusing by name. A typo in a format string is the kind of
/// thing that would otherwise quietly pick a default and produce a model that
/// is merely worse, so there is no default.
fn format_from_name(op: &str, name: &str) -> PyResult<GgmlDType> {
    ALL_FORMATS
        .into_iter()
        .find(|d| format_name(*d) == name)
        .ok_or_else(|| {
            not_implemented(format!(
                "{op}: unknown quantisation format {name:?}. This build has: {}",
                ALL_FORMATS
                    .into_iter()
                    .map(format_name)
                    .collect::<Vec<_>>()
                    .join(", ")
            ))
        })
}

/// `_C._quantized_formats()` -- what `_quantize` will accept.
#[pyfunction]
fn _quantized_formats() -> Vec<&'static str> {
    ALL_FORMATS.into_iter().map(format_name).collect()
}

/// `_C._quantize(t, format)` -- a dense tensor to a block-quantised one.
///
/// The result's dtype tag is `float32` and not a `q*` tag; the argument for
/// that is on `PyTensorBase::quantized`.
///
/// **`Q8_1` and `Q8_K` are refused here even though candle can hold them.**
/// They are activation-side formats: `vec_dot` consumes them as the *right*
/// operand and no `GgmlType::VecDotType` names them as a weight, so a weight
/// quantised to either reaches `QMatMul` and fails inside candle with a
/// message about block types. Refusing at the door names the reason instead.
/// docs/graph/QUANT2.md §6.
#[pyfunction]
#[pyo3(name = "_quantize")]
fn quantize(py: Python<'_>, tensor: PyTensorBase, format: &str) -> PyResult<Py<PyAny>> {
    const OP: &str = "torch._C._quantize";
    let dtype = format_from_name(OP, format)?;
    if matches!(dtype, GgmlDType::Q8_1 | GgmlDType::Q8K) {
        return Err(not_implemented(format!(
            "{OP}: {format} is an activation-side block format -- candle quantises \
             the *input* to it inside vec_dot and no weight format names it as a \
             VecDotType, so a weight stored this way has no matmul. Use q8_0."
        )));
    }
    let src = tensor.tensor()?;
    // Checked here rather than left to candle's `check_shape`, which raises a
    // `RuntimeError` phrased as an internal invariant. This is the wall a
    // caller actually hits -- SmolLM2-135M is 576 wide and no 256-element
    // k-quant can hold a 576-column weight (docs/graph/QUANT2.md §5.2) -- so it
    // refuses by name, at the door, saying which multiple it wanted. The
    // module replacement in `torchnative/quant` groups its skips by this
    // message, so the phrasing is what makes a new wall visible as a new line
    // rather than as a count that got smaller.
    let block = dtype.block_size();
    match src.dims().last() {
        Some(&last) if last.is_multiple_of(block) => {}
        Some(&last) => {
            return Err(not_implemented(format!(
                "{OP}: {format} stores {block} elements per block, so it must have \
                 their last dim divisible by block size -- got {last}, which is not \
                 a multiple of {block} (shape {:?})",
                src.dims()
            )))
        }
        None => {
            return Err(not_implemented(format!(
                "{OP}: a 0-d tensor cannot be quantised"
            )))
        }
    }
    // candle's `QTensor::quantize` reads the whole storage buffer, ignoring
    // the layout, so any view of a larger buffer (`rows[s:s+n]`, even with
    // `s == 0`) panicked in `copy_from_slice` once the buffer was bigger than
    // the view. The layout alone cannot say whether the buffer is larger, so
    // the input is always copied into storage of its own first.
    let owned = src.force_contiguous().map_err(|e| candle_err(OP, e))?;
    let src = &owned;
    let q = QTensor::quantize(src, dtype).map_err(|e| candle_err(OP, e))?;
    let wrapped = PyTensorBase::quantized(Arc::new(q), TorchDType::Float32);
    crate::tensor::promote(py, wrapped.into_pyobject(py)?.into_any().unbind())
}

/// `_C._dequantize(t)` -- back to a dense `float32` tensor.
///
/// This is the round trip the verification axis is built on: it is
/// deterministic, it needs no model, and `tests/_support/ggml_ref.py` reimplements it
/// from the format so the comparison is against an independent derivation
/// rather than against candle restating itself.
#[pyfunction]
#[pyo3(name = "_dequantize")]
fn dequantize(py: Python<'_>, tensor: PyTensorBase) -> PyResult<Py<PyAny>> {
    const OP: &str = "torch._C._dequantize";
    let q = tensor.qtensor(OP)?;
    let device = q.device();
    let dense = q.dequantize(&device).map_err(|e| candle_err(OP, e))?;
    let wrapped = PyTensorBase::new(dense)?;
    crate::tensor::promote(py, wrapped.into_pyobject(py)?.into_any().unbind())
}

/// `_C._quantized_format(t)` -- `"q4_k"` and friends.
#[pyfunction]
#[pyo3(name = "_quantized_format")]
fn quantized_format(tensor: PyTensorBase) -> PyResult<&'static str> {
    Ok(format_name(
        tensor.qtensor("torch._C._quantized_format")?.dtype(),
    ))
}

/// `_C._quantized_nbytes(t)` -- what the weight actually costs.
///
/// The question `element_size()` refuses, answered. `numel()` divided by this
/// is the compression ratio, and it is the only honest way to state one: the
/// tag says `float32` and the blob is nothing of the kind.
#[pyfunction]
#[pyo3(name = "_quantized_nbytes")]
fn quantized_nbytes(tensor: PyTensorBase) -> PyResult<usize> {
    Ok(tensor
        .qtensor("torch._C._quantized_nbytes")?
        .storage_size_in_bytes())
}

/// `_C._quantized_blob(t)` -- the raw GGML bytes.
///
/// **This exists for the verification axis and nothing else.** It is what lets
/// `tests/_support/ggml_ref.py` compare candle's quantiser against an independent
/// reimplementation *byte for byte* rather than through a tolerance on the
/// reconstruction, which is the difference between checking the format and
/// checking that two lossy things are near each other. docs/graph/QUANT2.md §2.1.
#[pyfunction]
#[pyo3(name = "_quantized_blob")]
fn quantized_blob<'py>(py: Python<'py>, tensor: PyTensorBase) -> PyResult<Bound<'py, PyBytes>> {
    const OP: &str = "torch._C._quantized_blob";
    let q = tensor.qtensor(OP)?;
    let data = q.data().map_err(|e| candle_err(OP, e))?;
    Ok(PyBytes::new(py, &data))
}

/// `_C._quantized_from_blob(bytes, shape, format)` -- the other direction.
///
/// The half of the axis that checks the *reader*: hand candle a blob this
/// process did not produce (a reference one, or a deliberately corrupted one)
/// and see what it reconstructs. Without it, `dequantize(quantize(x))` could
/// only ever compare candle to itself.
#[pyfunction]
#[pyo3(name = "_quantized_from_blob")]
fn quantized_from_blob(
    py: Python<'_>,
    data: &[u8],
    shape: Vec<usize>,
    format: &str,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "torch._C._quantized_from_blob";
    let dtype = format_from_name(OP, format)?;
    let elem_count: usize = shape.iter().product();
    let blocks = elem_count / dtype.block_size();
    let want = blocks * dtype.type_size();
    if !elem_count.is_multiple_of(dtype.block_size()) || data.len() != want {
        return Err(pyo3::exceptions::PyValueError::new_err(format!(
            "{OP}: {format} stores {} elements in {} bytes, so a {shape:?} tensor \
             needs {want} bytes; got {}",
            dtype.block_size(),
            dtype.type_size(),
            data.len()
        )));
    }
    let storage = QStorage::from_data(Cow::Borrowed(data), &Device::Cpu, dtype)
        .map_err(|e| candle_err(OP, e))?;
    let q = QTensor::new(storage, shape).map_err(|e| candle_err(OP, e))?;
    let wrapped = PyTensorBase::quantized(Arc::new(q), TorchDType::Float32);
    crate::tensor::promote(py, wrapped.into_pyobject(py)?.into_any().unbind())
}

/// `_C._quantized_linear(input, weight, bias)` -- `input @ weight.T + bias`,
/// with `weight` block-quantised.
///
/// The shape contract is `nn.Linear`'s, so the module replacement in
/// `torchnative/quant` is a swap and not a transpose: `weight` is
/// `(out_features, in_features)` exactly as `nn.Linear.weight` is, and
/// candle's `QMatMul` reads its `QTensor` as already transposed, which is the
/// same convention.
///
/// **`format="f32"` is the exactness control and it is load-bearing.**
/// `QMatMul::from_arc` dequantises F32/F16/BF16 up front and takes a plain
/// `matmul`, so this call with an `f32` "quantised" weight has to be *bit
/// identical* to `torch.nn.functional.linear`. That separates two claims which
/// a tolerance would blur: whether the wiring (transpose, bias, batch dims,
/// contiguity) is right, and whether the quantisation is close. The first is
/// checked exactly; only the second is checked with a bound.
///
/// The activation must be `float32`. `QMatMul::forward` accepts `f32` and
/// `f16` only and this shim's reduced-precision path is `bfloat16`
/// (docs/numerics/DTYPE.md §6.2), so widening would have to happen somewhere; doing it
/// silently here would hide a cost from whoever is measuring. Refused by name
/// until there is a measurement to put next to it.
#[pyfunction]
#[pyo3(name = "_quantized_linear")]
#[pyo3(signature = (input, weight, bias = None))]
fn quantized_linear(
    py: Python<'_>,
    input: PyTensorBase,
    weight: PyTensorBase,
    bias: Option<PyTensorBase>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "torch._C._quantized_linear";

    let q = weight.qtensor(OP)?;
    let x = input.tensor()?;
    if input.tag() != TorchDType::Float32 {
        return Err(not_implemented(format!(
            "{OP}: the activation must be float32, got torch.{}. candle's QMatMul \
             takes f32 or f16 only, and widening here would spend time the caller \
             cannot see. docs/numerics/DTYPE.md §6.2.",
            input.tag().name()
        )));
    }
    // candle's `QMatMul::forward` dispatches on the *activation's* storage and
    // then `unreachable!()`s when the weight's storage is another backend
    // (`Cannot call metal matmul on non metal QTensor`), which reaches Python
    // as a `PanicException`. Refused here by name instead, in upstream's
    // wording for the same mistake on a dense `linear`. Issue #175.
    if !q.device().same_device(x.device()) {
        return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "{OP}: Expected all tensors to be on the same device, but found at \
             least two devices, {} (the input) and {} (the quantised weight). \
             Move the module with `.to(device)`; QuantizedLinear moves its blocks \
             byte for byte (torch._C._quantized_to).",
            crate::device::PyDevice::from_candle(x.device()).__str__(),
            crate::device::PyDevice::from_candle(&q.device()).__str__(),
        )));
    }
    let (out_features, in_features) = q.shape().dims2().map_err(|e| candle_err(OP, e))?;
    match x.dims().last() {
        Some(&k) if k == in_features => {}
        _ => {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{OP}: mat1 and mat2 shapes cannot be multiplied ({:?} and \
                 {in_features}x{out_features})",
                x.dims()
            )))
        }
    }
    // `QMatMul::cpu_fwd` refuses a non-contiguous input by name rather than
    // reading it wrongly, which is the right refusal but not a useful one for
    // a caller who sliced a batch. Upstream's `linear` accepts any layout, so
    // the copy happens here and is recorded rather than pushed onto the user.
    let x = if x.is_contiguous() {
        x.clone()
    } else {
        x.contiguous().map_err(|e| candle_err(OP, e))?
    };

    // Rebuilt per call, and that is cheap on purpose: `from_arc` is an `Arc`
    // clone and a match for every quantised format. It is *not* cheap for
    // F32/F16/BF16, where it dequantises -- but those are the control path,
    // not the shipping one.
    let mm = QMatMul::from_arc(Arc::clone(q)).map_err(|e| candle_err(OP, e))?;
    let mut out = mm.forward(&x).map_err(|e| candle_err(OP, e))?;

    if let Some(bias) = bias.as_ref() {
        let b = bias.tensor()?;
        if bias.tag() != TorchDType::Float32 {
            return Err(not_implemented(format!(
                "{OP}: the bias must be float32, got torch.{}",
                bias.tag().name()
            )));
        }
        if b.dims() != [out_features] {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                "{OP}: bias shape {:?} does not match out_features {out_features}",
                b.dims()
            )));
        }
        out = out.broadcast_add(b).map_err(|e| candle_err(OP, e))?;
    }

    let wrapped = PyTensorBase::new(out)?;
    crate::tensor::promote(py, wrapped.into_pyobject(py)?.into_any().unbind())
}

/// `_C._quantized_to(t, device)` -- the same blocks, on another device.
///
/// **Byte for byte, never re-quantised.** The blob is read off the source
/// device (`QTensor::data`, a blit on Metal) and handed unchanged to
/// `QStorage::from_data` on the target, so a GGUF file's own blocks
/// (`keep_blocks=True`, S2.5) are still the file's blocks after
/// `model.to("mps")`. Dequantising and quantising again would recompute every
/// block's scale and change the bytes, which is exactly what keeping the blocks
/// exists to avoid. Same device in, the input itself out (no copy), the way a
/// dense `tensor.to(device)` returns `self`.
///
/// `QuantizedLinear._apply` calls this, because its `qweight` is a plain
/// attribute that `nn.Module._apply` does not visit (issue #175).
#[pyfunction]
#[pyo3(name = "_quantized_to")]
fn quantized_to(
    py: Python<'_>,
    tensor: Bound<'_, PyAny>,
    device: Bound<'_, PyAny>,
) -> PyResult<Py<PyAny>> {
    const OP: &str = "torch._C._quantized_to";
    let base: PyTensorBase = tensor.extract()?;
    let q = base.qtensor(OP)?;
    let target = crate::device::PyDevice::coerce(&device)?.resolve()?;
    if q.device().same_device(&target) {
        return Ok(tensor.unbind());
    }
    let dtype = q.dtype();
    let data = q.data().map_err(|e| candle_err(OP, e))?;
    let storage =
        QStorage::from_data(data, &target, dtype).map_err(|e| candle_err(OP, e))?;
    let moved = QTensor::new(storage, q.shape().clone()).map_err(|e| candle_err(OP, e))?;
    let wrapped = PyTensorBase::quantized(Arc::new(moved), base.tag());
    crate::tensor::promote(py, wrapped.into_pyobject(py)?.into_any().unbind())
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(quantized_to, m)?)?;
    m.add_function(wrap_pyfunction!(_quantized_formats, m)?)?;
    m.add_function(wrap_pyfunction!(quantize, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize, m)?)?;
    m.add_function(wrap_pyfunction!(quantized_format, m)?)?;
    m.add_function(wrap_pyfunction!(quantized_nbytes, m)?)?;
    m.add_function(wrap_pyfunction!(quantized_blob, m)?)?;
    m.add_function(wrap_pyfunction!(quantized_from_blob, m)?)?;
    m.add_function(wrap_pyfunction!(quantized_linear, m)?)?;
    Ok(())
}

#[cfg(test)]
mod q4k_writer_tests {
    use candle_core::quantized::{GgmlDType, QTensor};
    use candle_core::{Device, Tensor};

    /// Issue #27: candle's Q4K writer is llama.cpp's `quantize_row_q4_K_ref`
    /// (ggml/src/ggml-quants.c at 3de52ac, lines 1467-1535, with
    /// `make_qkx2_quants` lines 808-887 and the ties-to-even `nearest_int`
    /// lines 630-635), byte for byte.
    ///
    /// The expected bytes were produced by the scalar transcription in
    /// `tests/numerics/test_q4kwriter.py` and agree with `quantize_row_q4_K_ref`
    /// exported by a compiled llama.cpp (`libggml-base.dylib`, read through
    /// ctypes). candle 0.11.0's old `make_qkx1_quants(15, 5)` writer differs on
    /// all four super-blocks. The inputs are exact in `f32`, so no RNG or libm
    /// is involved:
    ///
    ///   0. `((i * 37) % 61 - 30) / 16`, mixed signs;
    ///   1. all `-0.5`: llama.cpp keeps `the_min = 0.5` (dmin != 0, d == 0);
    ///   2. sub-block 0 is `(i % 7 - 3) / 4096` against `/ 8` elsewhere, so its
    ///      6-bit scale rounds to 0 and `if (!d) continue;` keeps the search's
    ///      levels (nonzero nibbles under a zero scale);
    ///   3. `(i * 11 + i / 32) % 33`, where rounding an exact .5 away from zero
    ///      instead of to even changes the bytes.
    ///
    /// The Python suite covers Gaussian and heavy-tailed data through `_C`;
    /// this one runs in `cargo test` with no Python, so a writer regression is
    /// red even when the extension is not imported.
    #[test]
    fn q4k_writer_is_llama_cpp_quantize_row_q4_k_ref() {
        let mut x = Vec::with_capacity(4 * 256);
        for i in 0..256usize {
            x.push((((i * 37) % 61) as f32 - 30.0) / 16.0);
        }
        x.extend(std::iter::repeat_n(-0.5f32, 256));
        for i in 0..256usize {
            x.push(if i < 32 {
                ((i % 7) as f32 - 3.0) / 4096.0
            } else {
                (((i * 13) % 29) as f32 - 14.0) / 8.0
            });
        }
        for i in 0..256usize {
            x.push(((i * 11 + i / 32) % 33) as f32);
        }
        let t = Tensor::from_vec(x, (4, 256), &Device::Cpu).unwrap();
        let q = QTensor::quantize(&t, GgmlDType::Q4K).unwrap();
        let got = q.data().unwrap();
        let want: String = [
            "e51b8d27fffffffefdfffdfeefffefec6009a34dd6701aa44de7711ab45ee7812bb55ef8822bc56ff8923cc66f09933c",
            "3dd6701aa44de7711ab45ee7812bb55ef8822bc56ff8923cc66f09933cd670091aa44de7711ab45ee8812bb55ef8822b",
            "c56ff9923cc66f09933cd6700aa34dd7e7811ab45ef8812bc55ef8922bc56ff9923cd66f09a33cd6700aa34de7711ab4",
            "0000102000000000fffffffff0f0f0f00000000000000000000000000000000000000000000000000000000000000000",
            "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
            "000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000",
            "0f1c5a27c0ffffffc0ffffffffffecff50c245a72a9c1f70e265c74abc3f901285e76adc5fb032a5278a0c7fd052c547",
            "1a72e961c74eb63c941b83e961d85eb63da52b830a72d850c74da52c941a72e9b63c941b83e961d85eb63da52b830a72",
            "d850c74da52c941a72e961c74eb63c9461d85fb63da52c830a72d950c74ea52c941b72e961c74eb63d941b83ea61d85f",
            "78285027ededeef4005400d7052506fff0178ef0178ef0178ef0178ef0178ef0178ef0178ef0178ef0178ef0178ef017",
            "288fe1288fe1288fe1288fe1288fe1288fe1288fe1288fe1288fe1288fe1288f9ef2389ef2389ef2389ef2389ef2389e",
            "f2389ef2389ef2389ef2389ef2389ef2e3499fe3499fe3499fe3499fe3499fe3499fe3499fe3499fe3499fe3499fe349",
        ]
        .concat();
        let got_hex: String = got.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(got.len(), 4 * 144);
        for b in 0..4 {
            let (g, w) = (
                &got_hex[b * 288..(b + 1) * 288],
                &want[b * 288..(b + 1) * 288],
            );
            assert_eq!(g, w, "super-block {b} is not llama.cpp's Q4_K bytes");
        }
    }
}
