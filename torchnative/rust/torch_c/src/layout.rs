//! Stride arithmetic for tensors whose layout is metadata: `Repr::Meta`.
//!
//! Every function here is a port of one upstream rule, named at its head, and
//! nothing here reads a byte. They exist because a meta tensor now **stores**
//! its stride and storage offset (docs/graph/STRIDE.md), so the meta kernels
//! have to compute the layout upstream would have produced, not just the shape.
//!
//! Concrete integers only. Upstream's versions are written against `SymInt`
//! and wrap every comparison in `guard_or_false`/`guard_or_true`; for concrete
//! values each of those is the plain comparison, and the one place that is not
//! obviously so (`ge` in `logical_to_physical_perm`) is argued at its site.
//!
//! None of these is trusted because it reads like upstream:
//! `tests/ops/test_metastride.py` compares the layouts they produce against
//! upstream torch, run in a separate process, case by case.

/// The row-major stride for a shape. `make_contiguous_strides_for`, and
/// `TensorImpl`'s default: a zero extent contributes `max(extent, 1)`, so
/// `empty(0, 3).stride()` is `(3, 1)` rather than `(0, 1)`.
pub fn contiguous(shape: &[usize]) -> Vec<usize> {
    let mut stride = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        stride[i] = stride[i + 1] * shape[i + 1].max(1);
    }
    stride
}

/// `TensorImpl::compute_contiguous`: fewer than two elements is contiguous;
/// otherwise strides must nest right to left, skipping extent-1 axes, whose
/// stride is unobservable.
pub fn is_contiguous(shape: &[usize], stride: &[usize]) -> bool {
    if shape.iter().product::<usize>() < 2 {
        return true;
    }
    nests(shape.iter().zip(stride).rev().map(|(&x, &s)| (x, s)))
}

/// `(extent, stride)` pairs, innermost first, nest into one another.
fn nests(pairs: impl Iterator<Item = (usize, usize)>) -> bool {
    let mut expected = 1usize;
    for (extent, stride) in pairs {
        if extent == 1 {
            continue;
        }
        if stride != expected {
            return false;
        }
        expected *= extent;
    }
    true
}

/// `compute_channels_last_contiguous_2d` / `_3d`: the same nesting, visiting
/// the axes in NHWC (NDHWC) order. Only a rank-4 tensor can be
/// `channels_last` and only a rank-5 one `channels_last_3d`; every other rank
/// answers `false`, as upstream's `switch` does.
pub fn is_channels_last(shape: &[usize], stride: &[usize], rank: usize) -> bool {
    let Some(order) = channels_last_order(rank) else {
        return false;
    };
    if shape.len() != rank {
        return false;
    }
    nests(order.iter().map(|&d| (shape[d], stride[d])))
}

/// The NHWC (NDHWC) axis order, or `None` for a rank with no channels-last
/// format.
fn channels_last_order(rank: usize) -> Option<&'static [usize]> {
    match rank {
        4 => Some(&[1, 3, 2, 0]),
        5 => Some(&[1, 4, 3, 2, 0]),
        _ => None,
    }
}

/// `is_channels_last_strides_2d_s4` / `_3d_s5`: the stride *ordering*
/// `suggest_memory_format` reads, which is looser than channels-last
/// contiguity and has upstream's two ambiguity fallbacks -- a zero channel
/// stride, and an `N111` whose batch stride equals its channel stride, both
/// answer "not channels-last".
pub fn strides_like_channels_last(shape: &[usize], stride: &[usize]) -> bool {
    let Some(order) = channels_last_order(shape.len()) else {
        return false;
    };
    if stride[1] == 0 {
        return false;
    }
    let mut min = 0usize;
    for &d in order {
        if shape[d] == 0 || stride[d] < min {
            return false;
        }
        if d == 0 && min == stride[1] {
            return false;
        }
        min = stride[d];
        if shape[d] > 1 {
            min *= shape[d];
        }
    }
    true
}

/// `get_channels_last_strides_2d` / `_3d`: the channels-last stride for a
/// shape. No `max(extent, 1)` here, unlike `contiguous` -- upstream's two
/// functions differ in exactly that, deliberately ("consistent with eager").
pub fn channels_last(shape: &[usize]) -> Option<Vec<usize>> {
    let order = channels_last_order(shape.len())?;
    let mut stride = vec![0usize; shape.len()];
    let mut acc = 1usize;
    for &d in order {
        stride[d] = acc;
        acc *= shape[d];
    }
    Some(stride)
}

/// `_prims_common.is_non_overlapping_and_dense_or_false`: some permutation of
/// the axes makes the layout contiguous.
///
/// Upstream sorts `(extent, stride)` pairs with `K.__lt__`, which for
/// non-zero strides is exactly `a.stride < b.stride`. For a zero stride it is
/// not a strict weak order (`K(3) < K(0)` and `K(0) < K(3)` both hold), but it
/// cannot change the answer: an axis with stride 0 and extent > 1 fails the
/// nesting check wherever it lands, and one with extent 1 is skipped wherever
/// it lands. Ties among equal non-zero strides are likewise order-free -- two
/// extents > 1 cannot both nest at one stride. So a stable sort by stride is
/// the rule.
pub fn is_non_overlapping_and_dense(shape: &[usize], stride: &[usize]) -> bool {
    if shape.iter().product::<usize>() < 2 {
        return true;
    }
    if shape.len() == 1 {
        return stride[0] == 1;
    }
    let mut pairs: Vec<(usize, usize)> = shape.iter().copied().zip(stride.iter().copied()).collect();
    pairs.sort_by_key(|&(_, s)| s);
    nests(pairs.into_iter())
}

/// The output stride of a `preserve_format` copy (`clone`, `_to_copy`):
/// the input's own stride if it is non-overlapping and dense, otherwise the
/// elementwise permutation. This is the Python function's shortcut that
/// `elementwise_stride` deliberately lacks, and the two answer differently
/// for an empty transposed tensor -- `clone` keeps `(1, 3)`, measured.
pub fn preserve_format_stride(shape: &[usize], stride: &[usize]) -> Vec<usize> {
    if shape.len() > 1 && is_non_overlapping_and_dense(shape, stride) {
        return stride.to_vec();
    }
    elementwise_stride(shape, &[stride.to_vec()])
}

/// The failures `collapse_view` can report, each in upstream's type.
pub enum CollapseError {
    /// `validate_idx`'s `AssertionError`.
    OutOfBounds { idx: usize, rank: usize },
    /// `torch._check_value`'s `ValueError`.
    Backwards { start: usize, end: usize },
    /// `torch._check`'s `RuntimeError`: the axes do not nest.
    NoSuchView,
}

/// `torch._prims._collapse_view_helper` with `must_be_valid` set: the shape
/// and stride of axes `start..=end` merged into one, or why not.
pub fn collapse_view(
    shape: &[usize],
    stride: &[usize],
    start: usize,
    end: usize,
) -> Result<(Vec<usize>, Vec<usize>), CollapseError> {
    let rank = shape.len().max(1);
    for idx in [start, end] {
        if !(idx < rank || idx == 0) {
            return Err(CollapseError::OutOfBounds { idx, rank });
        }
    }
    if end < start {
        return Err(CollapseError::Backwards { start, end });
    }
    let zero_dim = shape.is_empty();
    let (shape, stride): (Vec<usize>, Vec<usize>) = if zero_dim {
        (vec![1], vec![1])
    } else {
        (shape.to_vec(), stride.to_vec())
    };
    if zero_dim || end == start {
        return Ok((shape, stride));
    }
    let numel: usize = shape.iter().product();
    if numel != 0 {
        for idx in (start..end).rev() {
            let nested = shape[idx] == 1
                || shape[idx + 1] == 1
                || stride[idx] == stride[idx + 1] * shape[idx + 1];
            if !nested {
                return Err(CollapseError::NoSuchView);
            }
        }
    }
    let mut merged_stride = stride[end];
    for idx in (start..end).rev() {
        if shape[idx] != 1 {
            merged_stride = merged_stride.min(stride[idx]);
        }
    }
    let mut length = shape[end];
    if length != 0 {
        for idx in (start..end).rev() {
            if shape[idx] == 0 {
                length = 0;
                merged_stride = 0;
                break;
            }
            length *= shape[idx];
        }
    } else {
        merged_stride = 0;
    }
    let mut new_shape = shape[..start].to_vec();
    new_shape.push(length);
    new_shape.extend_from_slice(&shape[end + 1..]);
    let mut new_stride = stride[..start].to_vec();
    new_stride.push(merged_stride);
    new_stride.extend_from_slice(&stride[end + 1..]);
    // "when the input has no elements it's restrided as if it were contiguous"
    if numel == 0 {
        new_stride = contiguous(&new_shape);
    }
    Ok((new_shape, new_stride))
}

/// `computeStorageNbytes`: the bytes a layout addresses, `0` if it has no
/// elements. This is the size upstream gives the storage `empty_strided`
/// allocates, and it is smaller than `numel * itemsize` for an overlapping or
/// expanded layout.
pub fn storage_nbytes(shape: &[usize], stride: &[usize], offset: usize, itemsize: usize) -> usize {
    if shape.iter().any(|&x| x == 0) {
        return 0;
    }
    let last: usize = shape.iter().zip(stride).map(|(&x, &s)| (x - 1) * s).sum();
    (offset + last + 1) * itemsize
}

/// `torch::computeStride` (`TH`'s `THTensor_compute_stride`): the stride a
/// `view` to `new_shape` would have, or `None` where no such view exists.
///
/// The rule walks the old shape right to left in *chunks* of axes whose
/// strides nest, and a view is legal iff every new axis falls inside one
/// chunk. A contiguous input is one chunk, which is why `view` of a
/// contiguous tensor never refuses on layout.
pub fn view_stride(old_shape: &[usize], old_stride: &[usize], new_shape: &[usize]) -> Option<Vec<usize>> {
    if old_shape.is_empty() {
        return Some(vec![1; new_shape.len()]);
    }
    let numel: usize = old_shape.iter().product();
    if numel == 0 && old_shape == new_shape {
        return Some(old_stride.to_vec());
    }
    let mut new_stride = vec![0usize; new_shape.len()];
    if numel == 0 {
        for d in (0..new_shape.len()).rev() {
            new_stride[d] = if d + 1 == new_shape.len() {
                1
            } else {
                new_shape[d + 1].max(1) * new_stride[d + 1]
            };
        }
        return Some(new_stride);
    }
    let mut view_d = new_shape.len() as isize - 1;
    let mut chunk_base = *old_stride.last().expect("rank > 0");
    let mut tensor_numel = 1usize;
    let mut view_numel = 1usize;
    for tensor_d in (0..old_shape.len()).rev() {
        tensor_numel *= old_shape[tensor_d];
        let chunk_ends = tensor_d == 0
            || (old_shape[tensor_d - 1] != 1
                && old_stride[tensor_d - 1] != tensor_numel * chunk_base);
        if chunk_ends {
            while view_d >= 0 && (view_numel < tensor_numel || new_shape[view_d as usize] == 1) {
                new_stride[view_d as usize] = view_numel * chunk_base;
                view_numel *= new_shape[view_d as usize];
                view_d -= 1;
            }
            if view_numel != tensor_numel {
                return None;
            }
            if tensor_d > 0 {
                chunk_base = old_stride[tensor_d - 1];
                tensor_numel = 1;
                view_numel = 1;
            }
        }
    }
    if view_d != -1 {
        return None;
    }
    Some(new_stride)
}

/// `inferExpandGeometry`'s stride half: an axis that grows from extent 1 gets
/// stride 0, a kept axis keeps its stride, and a *new* leading axis of extent
/// 1 gets the stride a contiguous tensor would give it (it is not a zero,
/// which is what a rule that treated all new axes alike would say).
///
/// `target` is the resolved shape; `-1` sentinels are the caller's to resolve.
pub fn expand_stride(shape: &[usize], stride: &[usize], target: &[usize]) -> Vec<usize> {
    let ndim = target.len();
    let offset = ndim - shape.len();
    let mut out = vec![0usize; ndim];
    for i in (0..ndim).rev() {
        let (size, mut s) = if i >= offset {
            (shape[i - offset], stride[i - offset])
        } else if i + 1 == ndim {
            (1, 1)
        } else {
            (1, target[i + 1] * out[i + 1])
        };
        if size != target[i] {
            s = 0;
        }
        out[i] = s;
    }
    out
}

/// `inferUnsqueezeGeometry`'s stride for the inserted axis.
pub fn unsqueeze_stride(shape: &[usize], stride: &[usize], dim: usize) -> usize {
    if dim >= shape.len() {
        1
    } else {
        shape[dim] * stride[dim]
    }
}

/// The output stride of an elementwise op, or of a `preserve_format` copy:
/// `_prims_common.compute_elementwise_output_strides`.
///
/// `operands` are the tensor operands' strides, **already expanded** to
/// `shape` (`expand_stride`), in argument order. Measured before it was
/// written: on fifteen layouts, every pointwise and `preserve_format` meta
/// kernel upstream answers exactly this, including the layouts where it
/// differs from "keep the stride if dense, else contiguous".
///
/// **There is no single-operand shortcut**, although the Python function of
/// that name has one ("keep the stride if non-overlapping and dense"). The
/// meta kernels do not go through it: they go through `refs.empty_like`,
/// which applies the permutation directly, and the two differ on an empty
/// tensor -- a transposed `(3, 0)` keeps `(1, 3)` under the shortcut and
/// becomes `(1, 1)` under the kernels, measured. Where the input is dense
/// the permutation reproduces its stride anyway.
pub fn elementwise_stride(shape: &[usize], operands: &[Vec<usize>]) -> Vec<usize> {
    let ndim = shape.len();
    if operands.is_empty() || ndim == 0 {
        return Vec::new();
    }
    if ndim == 1 {
        return vec![1];
    }
    let perm = logical_to_physical_perm(shape, operands);
    let physical: Vec<usize> = perm.iter().map(|&d| shape[d]).collect();
    let physical_stride = contiguous(&physical);
    let mut out = vec![0usize; ndim];
    for (i, &d) in perm.iter().enumerate() {
        out[d] = physical_stride[i];
    }
    out
}

/// A copy of `t` in a storage of its own, laid out with `stride` at offset 0.
///
/// `stride` must be a dense permutation of `t`'s shape (what
/// `elementwise_stride` and `preserve_format_stride` answer for a tensor with
/// elements): every element lands at exactly one storage position and the
/// storage holds exactly `t.elem_count()` elements. The values are written in
/// that order, so `stride()` describes the storage: the axes are permuted
/// into decreasing-stride order, that view is copied row-major, and the copy
/// is presented through `Tensor::view_with_layout` with `t`'s shape and
/// `stride`. Only an extent-1 axis can tie with another, and where it lands
/// does not move an element.
///
/// `aten.rs::copy_in_format` (`clone`, same-dtype `_to_copy`) and the
/// elementwise results of issue #383 are built here. A layout that is not a
/// dense permutation is refused by `view_with_layout` (it addresses elements
/// the copy does not have) or answers wrong strides, so the callers pass only
/// those two functions' answers.
pub fn copy_into_stride(t: &candle_core::Tensor, stride: &[usize]) -> candle_core::Result<candle_core::Tensor> {
    let dims = t.dims().to_vec();
    if stride.len() != dims.len() {
        candle_core::bail!(
            "copy_into_stride: shape {dims:?} has rank {} but {} strides were given",
            dims.len(),
            stride.len()
        );
    }
    let mut order: Vec<usize> = (0..dims.len()).collect();
    order.sort_by(|&a, &b| stride[b].cmp(&stride[a]));
    let physical = if order.iter().enumerate().all(|(i, &d)| i == d) {
        t.force_contiguous()?
    } else {
        t.permute(order)?.force_contiguous()?
    };
    if physical.dims() == dims.as_slice() && physical.layout().stride() == stride {
        return Ok(physical);
    }
    physical.view_with_layout(candle_core::Layout::new(
        candle_core::Shape::from(dims),
        stride.to_vec(),
        0,
    ))
}

/// The stride upstream's CPU flash attention gives its `logsumexp`
/// (`(batch, heads, q_len)`): a `(batch, q_len, heads)` buffer transposed,
/// `(q_len * heads, 1, heads)`, for every query layout (measured on 2.13.0,
/// and what the meta arm answers from `_meta_registrations.py`).
pub fn flash_logsumexp_stride(shape: &[usize]) -> Vec<usize> {
    match shape {
        [_, heads, q_len] => vec![q_len * heads, 1, *heads],
        _ => contiguous(shape),
    }
}

/// `_prims_common.compute_elementwise_output_logical_to_physical_perm`.
fn logical_to_physical_perm(shape: &[usize], operands: &[Vec<usize>]) -> Vec<usize> {
    let ndim = shape.len();
    let contiguous_all = operands.iter().all(|s| is_contiguous(shape, s));
    let channels_last_all = operands.iter().all(|s| is_channels_last(shape, s, 4));
    if contiguous_all && !channels_last_all {
        return (0..ndim).collect();
    }
    if channels_last_all && !contiguous_all {
        let mut perm = vec![0];
        perm.extend(2..ndim);
        perm.push(1);
        return perm;
    }
    // `ge(a, b)` for sizes and strides known to be >= 0: upstream's
    // `b == 0 or (a != 0 and (a >= b or a % b == 0))`, and `a % b == 0` with
    // `a, b > 0` already implies `a >= b`.
    let ge = |a: usize, b: usize| b == 0 || (a != 0 && a >= b);
    let should_swap = |a: usize, b: usize| -> i8 {
        for stride in operands {
            let (sa, sb) = (stride[a], stride[b]);
            if sa == 0 || sb == 0 {
                continue;
            }
            if sa == sb {
                if ge(shape[b], shape[a]) {
                    continue;
                }
                return 1;
            }
            if ge(sb, sa) {
                return -1;
            }
            if ge(sa, sb) {
                return 1;
            }
        }
        0
    };
    // Back-to-front insertion sort that stops at the first "keep", exactly as
    // upstream's does -- an ambiguous (`0`) comparison moves past, it does not
    // stop the scan.
    let mut perm: Vec<usize> = (0..ndim).rev().collect();
    for i in 1..ndim {
        let mut dim1 = i;
        for dim0 in (0..i).rev() {
            match should_swap(perm[dim0], perm[dim1]) {
                c if c > 0 => {
                    perm.swap(dim0, dim1);
                    dim1 = dim0;
                }
                c if c < 0 => break,
                _ => {}
            }
        }
    }
    perm.reverse();
    perm
}

#[cfg(test)]
mod tests {
    use super::*;

    // Hand-checkable anchors only. The coverage is the Python differential
    // test against upstream; these pin the arithmetic so a refactor that
    // breaks it fails in `cargo test` before a model has to find it.

    #[test]
    fn a_transpose_is_neither_contiguous_nor_viewable_as_a_vector() {
        assert!(!is_contiguous(&[4, 3], &[1, 4]));
        assert_eq!(view_stride(&[4, 3], &[1, 4], &[12]), None);
        assert_eq!(view_stride(&[4, 3], &[1, 4], &[4, 3, 1]), Some(vec![1, 4, 4]));
    }

    #[test]
    fn expanding_a_singleton_gives_stride_zero_and_a_new_unit_axis_does_not() {
        assert_eq!(expand_stride(&[3, 1], &[1, 1], &[3, 4]), vec![1, 0]);
        assert_eq!(expand_stride(&[3], &[1], &[1, 3]), vec![3, 1]);
        assert_eq!(expand_stride(&[3], &[1], &[2, 3]), vec![0, 1]);
    }

    #[test]
    fn elementwise_output_keeps_a_dense_permutation_and_repacks_a_gappy_one() {
        assert_eq!(elementwise_stride(&[4, 3], &[vec![1, 4]]), vec![1, 4]);
        // `x[:, ::2]` of a contiguous (3, 8): not dense, output contiguous
        assert_eq!(elementwise_stride(&[3, 4], &[vec![8, 2]]), vec![4, 1]);
    }

    #[test]
    fn a_channels_last_stride_ordering_is_suggested_and_the_ambiguous_ones_are_not() {
        assert!(strides_like_channels_last(&[2, 3, 4, 5], &[60, 1, 15, 3]));
        assert!(!strides_like_channels_last(&[2, 3, 4, 5], &[60, 20, 5, 1]));
        // N111 with identical strides: upstream falls back to NCHW
        assert!(!strides_like_channels_last(&[2, 1, 1, 1], &[1, 1, 1, 1]));
        assert_eq!(channels_last(&[2, 3, 4, 5]), Some(vec![60, 1, 15, 3]));
        assert_eq!(channels_last(&[2, 3, 4]), None);
    }

    #[test]
    fn channels_last_is_read_off_the_stride() {
        assert!(is_channels_last(&[2, 3, 4, 5], &[60, 1, 15, 3], 4));
        assert!(!is_channels_last(&[2, 3, 4, 5], &[60, 20, 5, 1], 4));
        assert!(!is_channels_last(&[2, 3, 4, 5], &[60, 1, 15, 3], 5));
    }

    /// The candle fork's `Tensor::view_with_layout` (issue #380): a strided
    /// view shares the storage, and a layout reaching past the elements the
    /// receiver addresses is refused by name rather than read out of bounds.
    #[test]
    fn view_with_layout_shares_the_storage_and_refuses_a_layout_out_of_bounds() {
        use candle_core::{Device, Layout, Shape, Tensor};
        let base = Tensor::arange(0f32, 24f32, &Device::Cpu).unwrap().reshape((2, 3, 4)).unwrap();
        let t = base.transpose(0, 1).unwrap();
        let stride = view_stride(t.dims(), t.layout().stride(), &[3, 2, 2, 2]).unwrap();
        assert_eq!(stride, vec![4, 12, 2, 1]);
        let v = t
            .view_with_layout(Layout::new(Shape::from(vec![3, 2, 2, 2]), stride, 0))
            .unwrap();
        assert_eq!(crate::storage::storage_identity(&v), crate::storage::storage_identity(&base));
        assert_eq!(v.layout().stride(), &[4, 12, 2, 1]);
        assert_eq!(
            v.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            t.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        // One element past the receiver's span: offset 1 with the same strides.
        let past = t.view_with_layout(Layout::new(Shape::from(vec![3, 2, 4]), vec![4, 12, 1], 1));
        let msg = past.unwrap_err().to_string();
        assert!(msg.contains("view_with_layout"), "{msg}");
        // A stride that reaches further than the receiver does.
        assert!(t
            .view_with_layout(Layout::new(Shape::from(vec![3, 2, 4]), vec![8, 12, 1], 0))
            .is_err());
        // Shape and strides of different rank.
        assert!(t.view_with_layout(Layout::new(Shape::from(vec![24]), vec![1, 1], 0)).is_err());
        // An empty layout addresses nothing and is accepted.
        assert!(t.view_with_layout(Layout::new(Shape::from(vec![0, 5]), vec![9, 9], 99)).is_ok());
        // A narrowed receiver bounds its views by its own span, not the storage's.
        let row = base.narrow(0, 1, 1).unwrap();
        assert!(row.view_with_layout(Layout::new(Shape::from(vec![12]), vec![1], 12)).is_ok());
        assert!(row.view_with_layout(Layout::new(Shape::from(vec![12]), vec![1], 0)).is_err());
    }

    /// The candle fork's `Tensor::view_with_layout_in_storage` (issues #306,
    /// #380), the constructor `aten.as_strided_.default` restrides with: the
    /// bound is the STORAGE, not what the receiver addresses. A narrowed
    /// receiver grows back over the whole buffer, the result shares the buffer
    /// (a write to the base is read through it), one element past the storage
    /// is refused naming the constructor, an empty layout is accepted at any
    /// offset, and a rank mismatch is refused.
    ///
    /// Red when the bound is the receiver's span (the `grown` line), when the
    /// result copies instead of sharing the `Arc` (`storage_identity` and the
    /// written value), and when the end check is off by one (`past`).
    #[test]
    fn view_with_layout_in_storage_is_bounded_by_the_storage_not_the_receiver() {
        use candle_core::{Device, Layout, Shape, Tensor};
        let base = Tensor::arange(0f32, 12f32, &Device::Cpu).unwrap();
        let row = base.narrow(0, 2, 2).unwrap();
        // `w = arange(12.)[2:4]; w.as_strided_((10,), (1,), 2)` upstream.
        let grown = row
            .view_with_layout_in_storage(Layout::new(Shape::from(vec![10]), vec![1], 2))
            .unwrap();
        assert_eq!(crate::storage::storage_identity(&grown), crate::storage::storage_identity(&base));
        assert_eq!(
            grown.to_vec1::<f32>().unwrap(),
            vec![2., 3., 4., 5., 6., 7., 8., 9., 10., 11.]
        );
        // The same layout through `view_with_layout` is refused: the receiver
        // addresses two elements, so the two bounds differ here.
        assert!(row.view_with_layout(Layout::new(Shape::from(vec![10]), vec![1], 2)).is_err());
        // A transposed (3, 2) over offset 1, upstream's `as_strided_((3, 2), (1, 3), 1)`.
        let t = base
            .view_with_layout_in_storage(Layout::new(Shape::from(vec![3, 2]), vec![1, 3], 1))
            .unwrap();
        assert_eq!(
            t.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            vec![1., 4., 2., 5., 3., 6.]
        );
        // The last element of the storage is in bounds, one past it is not.
        assert!(base
            .view_with_layout_in_storage(Layout::new(Shape::from(vec![2]), vec![1], 10))
            .is_ok());
        let past = base.view_with_layout_in_storage(Layout::new(Shape::from(vec![2]), vec![1], 11));
        let msg = past.unwrap_err().to_string();
        assert!(msg.contains("view_with_layout_in_storage"), "{msg}");
        assert!(base
            .view_with_layout_in_storage(Layout::new(Shape::from(vec![4, 4]), vec![4, 1], 0))
            .is_err());
        // No elements: accepted at any offset, as upstream's `as_strided_((0, 3), (3, 1), 100)`.
        assert!(base
            .view_with_layout_in_storage(Layout::new(Shape::from(vec![0, 3]), vec![3, 1], 100))
            .is_ok());
        assert!(base
            .view_with_layout_in_storage(Layout::new(Shape::from(vec![12]), vec![1, 1], 0))
            .is_err());
        // Shared, not copied: a write to the base is read through the grown view.
        let ones = Tensor::ones(12, candle_core::DType::F32, &Device::Cpu).unwrap();
        base.slice_set(&ones, 0, 0).unwrap();
        assert_eq!(grown.to_vec1::<f32>().unwrap(), vec![1f32; 10]);
    }

    /// Issue #383: a fresh result re-laid in its operands' permutation holds
    /// its values in that order, in a storage of its own. Red when the copy
    /// only relabels the strides over row-major data (the values read back
    /// through the layout differ), when it shares the source's storage, and
    /// when the physical order is sorted the wrong way (`view_with_layout`
    /// refuses or the values move).
    #[test]
    fn copy_into_stride_writes_the_values_in_the_order_the_stride_describes() {
        use candle_core::{Device, Tensor};
        // Row-major (2, 3, 4) values, to be laid out as a permute(2, 0, 1)
        // result: stride (1, 8, 2).
        let rowmajor = Tensor::arange(0f32, 24f32, &Device::Cpu).unwrap().reshape((2, 3, 4)).unwrap();
        let out = copy_into_stride(&rowmajor, &[1, 8, 2]).unwrap();
        assert_eq!(out.dims(), &[2, 3, 4]);
        assert_eq!(out.layout().stride(), &[1, 8, 2]);
        assert_eq!(out.layout().start_offset(), 0);
        assert_ne!(crate::storage::storage_identity(&out), crate::storage::storage_identity(&rowmajor));
        assert_eq!(
            out.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            rowmajor.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        // The storage itself is in the permuted order: element (i, j, k) at
        // i + 8 j + 2 k. Read it back as a contiguous (3, 4, 2) view.
        let storage = out
            .view_with_layout(candle_core::Layout::new(candle_core::Shape::from(vec![3, 4, 2]), vec![8, 2, 1], 0))
            .unwrap();
        let expect = rowmajor.permute((1, 2, 0)).unwrap().contiguous().unwrap();
        assert_eq!(
            storage.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
            expect.flatten_all().unwrap().to_vec1::<f32>().unwrap()
        );
        // The row-major stride is the plain copy.
        let same = copy_into_stride(&rowmajor, &[12, 4, 1]).unwrap();
        assert_eq!(same.layout().stride(), &[12, 4, 1]);
        assert!(copy_into_stride(&rowmajor, &[1, 1]).is_err());
        // The flash `logsumexp` layout.
        assert_eq!(flash_logsumexp_stride(&[2, 4, 5]), vec![20, 1, 4]);
        assert_eq!(flash_logsumexp_stride(&[2, 1, 5]), vec![5, 1, 1]);
    }

    #[test]
    fn storage_nbytes_is_the_addressed_extent_not_the_element_count() {
        assert_eq!(storage_nbytes(&[3, 4], &[1, 1], 0, 4), 24);
        assert_eq!(storage_nbytes(&[0, 4], &[9, 1], 5, 4), 0);
        assert_eq!(storage_nbytes(&[2, 3], &[1, 2], 0, 4), 24);
    }
}
