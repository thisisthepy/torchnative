//! `a * b + c` the way upstream's compiler emitted it (issue #300).
//!
//! Several kernels here spell a fused multiply-add where upstream's source
//! writes `a * b + c`, because the compiler that built upstream contracts that
//! expression into one FMA and the result differs from two roundings in the
//! last bit (`rng.rs`'s `uniform_real`, `aten.rs`'s `chbevl` and the weighted
//! `nll_loss` accumulation; each records its own measurement). That is a fact
//! about the **build**, not the source:
//!
//! * macOS arm64 and Android (clang, `-ffp-contract=on`) and the manylinux
//!   wheel (GCC 13, `-ffp-contract=fast`) contract. The golden cases agree
//!   bit for bit there with the fused form (gate, every runner but Windows).
//! * The win_amd64 wheel is built with MSVC, which (read from these measurements,
//!   not from its build flags) does not contract under its
//!   default `/fp:precise`. The first Windows gate runs measured the fused form
//!   1 ulp away from upstream on `uniform_`, `i0`, `kaiser_window` and the
//!   weighted `nll_loss_forward` (run 43531608269, 24 + 19 + 10 + 6 cases).
//!
//! So the choice follows the target's C++ toolchain, as `msvc_sort.rs` does for
//! `std::sort`. Where upstream writes `std::fma` or a vector `fmadd` intrinsic
//! the FMA is in the source and every compiler keeps it; those sites call
//! `mul_add` directly and must not use this.

/// Whether upstream's build for this target contracts `a * b + c`.
pub(crate) const UPSTREAM_CONTRACTS: bool = !cfg!(target_env = "msvc");

/// `a * b + c` as upstream's compiled `float` code computes it.
#[inline]
pub(crate) fn madd_f32(a: f32, b: f32, c: f32) -> f32 {
    if UPSTREAM_CONTRACTS {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}

/// `a * b + c` as upstream's compiled `double` code computes it.
#[inline]
pub(crate) fn madd_f64(a: f64, b: f64, c: f64) -> f64 {
    if UPSTREAM_CONTRACTS {
        a.mul_add(b, c)
    } else {
        a * b + c
    }
}
