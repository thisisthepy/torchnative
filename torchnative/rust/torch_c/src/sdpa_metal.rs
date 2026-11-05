//! Fused scaled-dot-product attention on Metal (S7.3, issue #179).
//!
//! `aten._scaled_dot_product_flash_attention_for_cpu` is the op every
//! `F.scaled_dot_product_attention` reaches here, on every device
//! (`bootstrap.py`'s `scaled_dot_product_attention`). On `mps` its default
//! branch spelled attention out in candle tensor ops: a `q k^T` matmul, a
//! scale-and-mask pass, a row max, `exp`, a row sum, a divide and a `p v`
//! matmul, each its own Metal kernel with its own `[B, H, L, S]` buffer.
//! This is one kernel instead: candle-metal-kernels' MLX-derived
//! `steel_attention` (prefill, `L > 1`) and `sdpa_vector` (decode, `L == 1`),
//! with an online softmax and no score matrix in device memory.
//!
//! **What it computes is upstream's attention to within a tolerance, not
//! bit for bit.** The online softmax rescales partial sums block by block, so
//! the order of additions differs from upstream's blocked CPU kernel and from
//! the unfused branch. `crate::flash` remains the bit-exact reference.
//!
//! **Where it is used, and why only there** (`eligible`):
//!
//! * no explicit mask. The kernels take a mask, but transformers passes
//!   `attn_mask=None, is_causal=True` on an unpadded prefill and
//!   `attn_mask=None` on a decode step; padded batches keep the unfused path.
//! * `is_causal` only when `L == S`. The kernel's causal mask is bottom-right
//!   aligned (`ql_off = kl - ql` in `call_sdpa_full`); upstream's is top-left.
//!   They are the same mask exactly when the query and key lengths agree.
//! * decode (`L == 1`) only when not causal: a top-left causal mask with one
//!   query attends to the first key alone, which `sdpa_vector` cannot express.
//! * head dims the kernels are instantiated for, and `float32`/`float16`/
//!   `bfloat16`, with `head_dim` equal for q, k and v.
//!
//! **logsumexp is not computed.** The fused kernels do not produce it, and no
//! forward in this repository reads it (`tape.rs`'s `sdpa_backward` recomputes
//! from q, k and v). It is returned as a meta tensor of the right shape and
//! dtype, so reading its values refuses by name instead of returning numbers
//! nobody computed.

use candle_core::{CpuStorage, CustomOp3, DType, Layout, MetalStorage, Result, Shape, Tensor};
use candle_metal_kernels::kernels::sdpa::{call_sdpa_full, call_sdpa_vector, SdpaDType};

const FULL_HEAD_DIMS: [usize; 7] = [32, 64, 72, 80, 96, 128, 256];
const VECTOR_HEAD_DIMS: [usize; 6] = [32, 64, 96, 128, 256, 512];

/// Whether the fused kernel computes exactly this call's attention.
/// `q`/`k`/`v` are `[B, H, L, D]`, `[B, Hkv, S, D]`, `[B, Hkv, S, Dv]`.
pub fn eligible(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    has_mask: bool,
    is_causal: bool,
) -> bool {
    if has_mask || !q.device().is_metal() {
        return false;
    }
    if !matches!(q.dtype(), DType::F32 | DType::F16 | DType::BF16)
        || k.dtype() != q.dtype()
        || v.dtype() != q.dtype()
    {
        return false;
    }
    let (qd, kd, vd) = (q.dims(), k.dims(), v.dims());
    if qd.len() != 4 || kd.len() != 4 || vd.len() != 4 {
        return false;
    }
    let (b, h, l, d) = (qd[0], qd[1], qd[2], qd[3]);
    let (hkv, s) = (kd[1], kd[2]);
    if kd[0] != b || vd[0] != b || vd[1] != hkv || vd[2] != s || kd[3] != d || vd[3] != d {
        return false;
    }
    if hkv == 0 || h % hkv != 0 || l == 0 || s == 0 {
        return false;
    }
    if l == 1 {
        !is_causal && VECTOR_HEAD_DIMS.contains(&d)
    } else {
        (!is_causal || l == s) && FULL_HEAD_DIMS.contains(&d)
    }
}

/// `softmax(scale * q k^T [+ causal]) v`, fused. Call only when `eligible`.
pub fn attend(q: &Tensor, k: &Tensor, v: &Tensor, scale: f32, is_causal: bool) -> Result<Tensor> {
    // `steel_attention` reads q, k and v through their batch/head/row strides
    // and needs only a unit last stride. `sdpa_vector` reads q densely and
    // k/v as `(batch * kv_heads + head) * head_stride + row * D`, so it needs
    // rows dense and the batch stride to be `kv_heads * head_stride`. A
    // narrowed preallocated KV cache (`torchnative.kernels.kv_cache`, issue
    // #180) has exactly that layout, so it is read in place, not copied.
    let decode = q.dims()[2] == 1;
    let q = if decode { q.contiguous()? } else { readable(q, false)? };
    let k = readable(k, decode)?;
    let v = readable(v, decode)?;
    q.apply_op3_no_bwd(&k, &v, &FusedSdpa { scale, is_causal })
}

fn readable(t: &Tensor, vector: bool) -> Result<Tensor> {
    let st = t.stride();
    let d = t.dims();
    let ok = st[3] == 1 && (!vector || (st[2] == d[3] && (d[0] == 1 || st[0] == d[1] * st[1])));
    if ok {
        Ok(t.clone())
    } else {
        t.contiguous()
    }
}

struct FusedSdpa {
    scale: f32,
    is_causal: bool,
}

impl CustomOp3 for FusedSdpa {
    fn name(&self) -> &'static str {
        "torchnative-metal-sdpa"
    }

    fn cpu_fwd(
        &self,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
        _: &CpuStorage,
        _: &Layout,
    ) -> Result<(CpuStorage, Shape)> {
        candle_core::bail!("torchnative-metal-sdpa: Metal only; `eligible` keeps CPU tensors away")
    }

    fn metal_fwd(
        &self,
        q: &MetalStorage,
        q_l: &Layout,
        k: &MetalStorage,
        k_l: &Layout,
        v: &MetalStorage,
        v_l: &Layout,
    ) -> Result<(MetalStorage, Shape)> {
        use candle_core::backend::BackendStorage;

        let device = q.device();
        let itype = match q.dtype() {
            DType::F32 => SdpaDType::F32,
            DType::F16 => SdpaDType::F16,
            DType::BF16 => SdpaDType::BF16,
            other => candle_core::bail!("torchnative-metal-sdpa: unsupported dtype {other:?}"),
        };
        let out_shape = q_l.shape().clone();
        let elem_count = out_shape.elem_count();
        let output = device.new_buffer(elem_count, q.dtype(), "torchnative-sdpa-out")?;
        let size = q.dtype().size_in_bytes();
        let encoder = device.command_encoder()?;
        let q_dims = q_l.dims();
        if q_dims[2] == 1 {
            call_sdpa_vector(
                device.metal_device(),
                &encoder,
                device.kernels(),
                q_l.start_offset() * size,
                q_dims,
                q.buffer(),
                k_l.start_offset() * size,
                k_l.dims(),
                k_l.stride(),
                k.buffer(),
                v_l.start_offset() * size,
                v_l.stride(),
                v.buffer(),
                &output,
                self.scale,
                1.0,
                itype,
            )
            .map_err(candle_core::MetalError::from)?;
        } else {
            let o_strides = Layout::contiguous(out_shape.clone()).stride().to_vec();
            call_sdpa_full(
                device.metal_device(),
                &encoder,
                device.kernels(),
                q_l.start_offset() * size,
                q_dims,
                q_l.stride(),
                q.buffer(),
                k_l.start_offset() * size,
                k_l.dims(),
                k_l.stride(),
                k.buffer(),
                v_l.start_offset() * size,
                v.buffer(),
                v_l.stride(),
                None,
                None,
                None,
                &output,
                &o_strides,
                self.scale,
                self.is_causal,
                itype,
            )
            .map_err(candle_core::MetalError::from)?;
        }
        Ok((
            MetalStorage::new(output, device.clone(), elem_count, q.dtype()),
            out_shape,
        ))
    }
}
