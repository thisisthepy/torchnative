//! torch's CPU random number generator, ported rather than approximated.
//!
//! docs/RNG.md settled the question this file answers. candle's CPU backend
//! *refuses* to be seeded -- `set_seed` and `get_current_seed` are both
//! `bail!` -- so "use candle's RNG but line the seeds up" was never an option,
//! and `rand_distr`'s Ziggurat normal consumes a data-dependent number of
//! words, so even a shared bit stream would drift out of alignment. The only
//! way to make `torch.manual_seed(0)` mean the same thing here as upstream is
//! to move torch's generator over. That is what this is.
//!
//! Three layers have to match, and only the first is a library problem:
//!
//!   1. the engine -- MT19937 with Knuth's `init_genrand` seeding and the
//!      `--left` draw order (`ATen/core/MT19937RNGEngine.h`);
//!   2. the transformation -- a *low-bit mask* times a power-of-two divisor,
//!      one 32-bit word for float and two (**high word first**) for double
//!      (`ATen/core/TransformationHelper.h:85`);
//!   3. **the shape of the kernel** (`ATen/native/cpu/DistributionTemplates.h`).
//!
//! Layer 3 is the one no RNG crate could supply, and the one that looks like a
//! numerical bug when it is missed. `normal_` takes an entirely different path
//! -- different consumption, different values -- depending on whether the
//! tensor has 16 or more contiguous elements, and when the size is not a
//! multiple of 16 the kernel *redraws the last 16 elements over the top of
//! values it already wrote*. Same seed, `n=15` and `n=16` share not one value.
//! docs/RNG.md §1.3 measured all of that; this file transcribes it.
//!
//! Everything here is deliberately arithmetic-for-arithmetic with the C++,
//! down to `2.0f * c10::pi<double>` being evaluated in double and only then
//! narrowed. Where upstream writes `std::fma`, this writes `mul_add`.
use std::sync::{Mutex, MutexGuard, OnceLock};

// ---------------------------------------------------------------------------
// Layer 1 -- the engine
// ---------------------------------------------------------------------------

const MERSENNE_STATE_N: usize = 624;
const MERSENNE_STATE_M: usize = 397;
const MATRIX_A: u32 = 0x9908_b0df;
const UMASK: u32 = 0x8000_0000;
const LMASK: u32 = 0x7fff_ffff;

/// `at::mt19937_engine`.
///
/// The field names are upstream's because the legacy `get_state()` blob is
/// literally this struct (docs/RNG.md §1.1), so keeping them aligned is what
/// makes that format implementable later without re-deriving anything.
pub struct Mt19937 {
    seed: u64,
    left: i32,
    #[allow(dead_code)]
    seeded: bool,
    next: usize,
    state: [u32; MERSENNE_STATE_N],
}

impl Mt19937 {
    pub fn new(seed: u64) -> Self {
        let mut engine = Self {
            seed: 0,
            left: 1,
            seeded: false,
            next: 0,
            state: [0; MERSENNE_STATE_N],
        };
        engine.init_with_uint32(seed);
        engine
    }

    fn init_with_uint32(&mut self, seed: u64) {
        self.seed = seed;
        self.seeded = true;
        self.state[0] = (seed & 0xffff_ffff) as u32;
        for j in 1..MERSENNE_STATE_N {
            let prev = self.state[j - 1];
            self.state[j] = 1_812_433_253u32
                .wrapping_mul(prev ^ (prev >> 30))
                .wrapping_add(j as u32);
        }
        // `left_ = 1`, not `MERSENNE_STATE_N`. Combined with the pre-decrement
        // in `next_u32`, this makes a twist run *before the first draw* --
        // docs/RNG.md §1.1 records getting this backwards first, which
        // produces a plausible-looking stream that matches numpy and not
        // torch.
        self.left = 1;
        self.next = 0;
    }

    #[inline]
    fn mix_bits(u: u32, v: u32) -> u32 {
        (u & UMASK) | (v & LMASK)
    }

    #[inline]
    fn twist(u: u32, v: u32) -> u32 {
        (Self::mix_bits(u, v) >> 1) ^ (if v & 1 != 0 { MATRIX_A } else { 0 })
    }

    fn next_state(&mut self) {
        self.left = MERSENNE_STATE_N as i32;
        self.next = 0;

        let mut p = 0usize;
        // `for (int j = N - M + 1; --j; p++)` -- the pre-decrement makes this
        // run N-M times, not N-M+1.
        for _ in 0..(MERSENNE_STATE_N - MERSENNE_STATE_M) {
            self.state[p] =
                self.state[p + MERSENNE_STATE_M] ^ Self::twist(self.state[p], self.state[p + 1]);
            p += 1;
        }
        // `for (int j = M; --j; p++)` -- M-1 times. `p[M-N]` is a *negative*
        // offset in C; here p >= N-M, so `p + M - N` stays in range.
        for _ in 0..(MERSENNE_STATE_M - 1) {
            self.state[p] = self.state[p + MERSENNE_STATE_M - MERSENNE_STATE_N]
                ^ Self::twist(self.state[p], self.state[p + 1]);
            p += 1;
        }
        self.state[p] = self.state[p + MERSENNE_STATE_M - MERSENNE_STATE_N]
            ^ Self::twist(self.state[p], self.state[0]);
    }

    #[inline]
    pub fn next_u32(&mut self) -> u32 {
        self.left -= 1;
        if self.left == 0 {
            self.next_state();
        }
        let mut y = self.state[self.next];
        self.next += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^= y >> 18;
        y
    }
}

/// `at::CPUGeneratorImpl`.
///
/// The two `next_*_normal_sample` slots are not an optimisation detail that
/// can be dropped: Box-Muller produces samples in pairs, upstream returns one
/// and *keeps the other on the generator*, and the kept one survives across
/// calls. A `normal_()` on an odd-sized tensor therefore changes what the next
/// `normal_()` returns. Seeding clears them, exactly as
/// `set_current_seed` does.
pub struct CpuGenerator {
    engine: Mt19937,
    next_double_normal: Option<f64>,
    /// Unused today and kept anyway: it is the float half of upstream's pair
    /// cache. Nothing in the CPU kernels reaches it, because `normal_kernel`'s
    /// scalar path instantiates `normal_distribution<double>` regardless of
    /// the tensor's dtype (docs/RNG.md §1.3) -- but that is a fact about the
    /// *kernel*, not about the generator, and a future op that asks for a
    /// float normal would need this slot to stay in step with upstream.
    #[allow(dead_code)]
    next_float_normal: Option<f32>,
}

impl CpuGenerator {
    pub fn new(seed: u64) -> Self {
        Self {
            engine: Mt19937::new(seed),
            next_double_normal: None,
            next_float_normal: None,
        }
    }

    pub fn set_current_seed(&mut self, seed: u64) {
        self.engine = Mt19937::new(seed);
        self.next_double_normal = None;
        self.next_float_normal = None;
    }

    pub fn current_seed(&self) -> u64 {
        self.engine.seed
    }

    /// `CPUGeneratorImpl::random()`
    #[inline]
    pub fn random(&mut self) -> u32 {
        self.engine.next_u32()
    }

    /// `CPUGeneratorImpl::random64()` -- **high word drawn first.**
    #[inline]
    pub fn random64(&mut self) -> u64 {
        let hi = self.engine.next_u32() as u64;
        let lo = self.engine.next_u32() as u64;
        (hi << 32) | lo
    }
}

/// The process-wide CPU generator behind `torch.default_generator`.
///
/// Upstream seeds it non-deterministically at startup
/// (`getDefaultCPUGenerator()` -> `getNonDeterministicRandom()`), so a program
/// that wants reproducibility has to say `torch.manual_seed(...)`. Matching
/// that matters: seeding it to a constant here would make this shim *more*
/// deterministic than torch, and code that silently depends on the difference
/// would then behave differently on real torch.
static DEFAULT_GENERATOR: OnceLock<Mutex<CpuGenerator>> = OnceLock::new();

fn nondeterministic_seed() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x1234_5678_9abc_def0);
    let pid = std::process::id() as u64;
    // A heap address, for the case where the clock is coarse.
    let boxed = Box::new(0u8);
    let addr = (&*boxed as *const u8) as u64;
    nanos ^ (pid << 32) ^ addr.rotate_left(17)
}

/// The lock upstream's kernels take (`std::lock_guard lock(generator->mutex_)`).
/// The GIL already serialises callers here; this keeps the shape of the
/// upstream kernel rather than relying on that.
pub fn default_generator() -> MutexGuard<'static, CpuGenerator> {
    let cell = DEFAULT_GENERATOR.get_or_init(|| Mutex::new(CpuGenerator::new(nondeterministic_seed())));
    match cell.lock() {
        Ok(guard) => guard,
        // A panic while holding the RNG lock would leave the stream in an
        // unknown position; carrying on with it is still better than aborting
        // the interpreter, and nothing here can panic while the state is
        // half-written.
        Err(poisoned) => poisoned.into_inner(),
    }
}

// ---------------------------------------------------------------------------
// Layer 2 -- the transformations (`ATen/core/TransformationHelper.h`)
// ---------------------------------------------------------------------------

/// `transformation::uniform_real<float>`.
///
/// The mask is on the *low* bits. docs/RNG.md §1.2 measured the obvious
/// alternative (`val >> 8`) against real torch and it disagreed on every seed.
///
/// The affine step is `mul_add`, not `*` then `+`. Upstream's source says
/// `x * (to - from) + from`, but clang contracts that into a single fused
/// multiply-add by default and torch is built that way, so the *written*
/// expression is a 1-ulp-wrong transcription of the *compiled* one. This
/// hides completely on the ranges anyone tries first -- `(0,1)`, `(-1,1)`,
/// `(-0.5,0.5)` all have a power-of-two width, which makes the multiply exact
/// and the two forms identical. It shows up on `(2.0, 7.5)`, where ~9.5% of
/// draws came out 1 ulp low before this was `mul_add`.
#[inline]
fn uniform_real_f32(val: u32, from: f32, to: f32) -> f32 {
    const MASK: u32 = (1u32 << 24) - 1;
    const DIVISOR: f32 = 1.0 / ((1u32 << 24) as f32);
    let x = (val & MASK) as f32 * DIVISOR;
    x.mul_add(to - from, from)
}

/// `transformation::uniform_real<double>`. Same contraction as the float form.
#[inline]
fn uniform_real_f64(val: u64, from: f64, to: f64) -> f64 {
    const MASK: u64 = (1u64 << 53) - 1;
    const DIVISOR: f64 = 1.0 / ((1u64 << 53) as f64);
    let x = (val & MASK) as f64 * DIVISOR;
    x.mul_add(to - from, from)
}

// ---------------------------------------------------------------------------
// Layer 3 -- the kernels (`ATen/native/cpu/DistributionTemplates.h`)
// ---------------------------------------------------------------------------

/// `uniform_kernel`'s draw, in the accumulate type. Producing the whole vector
/// in one call keeps the generator locked for the length of the fill, which is
/// what upstream's single `lock_guard` does.
///
/// The caller applies the narrowing cast and the upper-bound clamp, because
/// both are defined in terms of the *storage* type and only the caller knows
/// it.
pub fn uniform_fill_f32(gen: &mut CpuGenerator, size: usize, from: f32, to: f32) -> Vec<f32> {
    (0..size)
        .map(|_| uniform_real_f32(gen.random(), from, to))
        .collect()
}

pub fn uniform_fill_f64(gen: &mut CpuGenerator, size: usize, from: f64, to: f64) -> Vec<f64> {
    (0..size)
        .map(|_| uniform_real_f64(gen.random64(), from, to))
        .collect()
}

/// `NormalFill16<float>` -- the scalar primary template.
///
/// Not the AVX2/VSX specialisation: neither is compiled on aarch64, so this is
/// what upstream actually runs on the hosts this shim is built for. On a
/// machine where the vector specialisation *is* live the last bits could
/// differ, since `sincos256_ps`/`log256_ps` are not libm -- recorded as
/// unmeasured in docs/RNG.md §6.
///
/// Two transcription traps here, both invisible in the output until they are
/// wrong by a few ulp:
///   * `u1 = 1 - data[j]` maps `[0,1)` to `(0,1]` so the log never sees zero;
///     it is *not* `data[j]`.
///   * `theta` is computed in **double** (`2.0f * c10::pi<double>` promotes)
///     and narrowed once, rather than being a float multiply throughout.
#[inline]
fn normal_fill_16_f32(data: &mut [f32], mean: f32, std: f32) {
    for j in 0..8 {
        let u1 = 1.0f32 - data[j];
        let u2 = data[j + 8];
        let radius = (-2.0f32 * u1.ln()).sqrt();
        let theta = (2.0f64 * std::f64::consts::PI * (u2 as f64)) as f32;
        data[j] = (radius * theta.cos()).mul_add(std, mean);
        data[j + 8] = (radius * theta.sin()).mul_add(std, mean);
    }
}

/// `NormalFill16<double>`.
#[inline]
fn normal_fill_16_f64(data: &mut [f64], mean: f64, std: f64) {
    for j in 0..8 {
        let u1 = 1.0f64 - data[j];
        let u2 = data[j + 8];
        let radius = (-2.0f64 * u1.ln()).sqrt();
        let theta = 2.0f64 * std::f64::consts::PI * u2;
        data[j] = (radius * theta.cos()).mul_add(std, mean);
        data[j + 8] = (radius * theta.sin()).mul_add(std, mean);
    }
}

/// `normal_fill<float>` -- path A, the `scalar_t == opmath_t` branch.
///
/// The uniforms go straight into the output buffer and Box-Muller runs over it
/// in place, 16 at a time. When the size is not a multiple of 16 the kernel
/// then steps back to `size - 16` and draws **sixteen fresh uniforms over
/// values it already computed**, so elements near the end of a `n=17` tensor
/// are not the ones a `n=16` tensor would have had, and neither are elements
/// 1..15. That overlap is the single most surprising thing in this file.
pub fn normal_fill_f32(gen: &mut CpuGenerator, size: usize, mean: f32, std: f32) -> Vec<f32> {
    let mut data: Vec<f32> = (0..size)
        .map(|_| uniform_real_f32(gen.random(), 0.0, 1.0))
        .collect();
    let mut i: usize = 0;
    while (i as i64) < size as i64 - 15 {
        normal_fill_16_f32(&mut data[i..i + 16], mean, std);
        i += 16;
    }
    if size % 16 != 0 {
        let offset = size - 16;
        for j in 0..16 {
            data[offset + j] = uniform_real_f32(gen.random(), 0.0, 1.0);
        }
        normal_fill_16_f32(&mut data[offset..offset + 16], mean, std);
    }
    data
}

/// `normal_fill<double>`.
pub fn normal_fill_f64(gen: &mut CpuGenerator, size: usize, mean: f64, std: f64) -> Vec<f64> {
    let mut data: Vec<f64> = (0..size)
        .map(|_| uniform_real_f64(gen.random64(), 0.0, 1.0))
        .collect();
    let mut i: usize = 0;
    while (i as i64) < size as i64 - 15 {
        normal_fill_16_f64(&mut data[i..i + 16], mean, std);
        i += 16;
    }
    if size % 16 != 0 {
        let offset = size - 16;
        for j in 0..16 {
            data[offset + j] = uniform_real_f64(gen.random64(), 0.0, 1.0);
        }
        normal_fill_16_f64(&mut data[offset..offset + 16], mean, std);
    }
    data
}

/// `normal_fill<BFloat16/Half>` -- path A's *other* branch.
///
/// Reduced-precision dtypes do not get a buffer the size of the tensor; they
/// get a sixteen-slot stack buffer that is refilled per block. The consumption
/// pattern is the same, but the tail case draws its sixteen uniforms without
/// the preceding whole-buffer fill, so the stream position at the tail differs
/// from the float path. Returned in `opmath_t` (float); the caller narrows.
pub fn normal_fill_reduced(gen: &mut CpuGenerator, size: usize, mean: f32, std: f32) -> Vec<f32> {
    let mut out: Vec<f32> = vec![0.0; size];
    let mut buf = [0.0f32; 16];
    let mut i: usize = 0;
    while (i as i64) < size as i64 - 15 {
        for slot in buf.iter_mut() {
            *slot = uniform_real_f32(gen.random(), 0.0, 1.0);
        }
        normal_fill_16_f32(&mut buf, mean, std);
        out[i..i + 16].copy_from_slice(&buf);
        i += 16;
    }
    if size % 16 != 0 {
        let offset = size - 16;
        for slot in buf.iter_mut() {
            *slot = uniform_real_f32(gen.random(), 0.0, 1.0);
        }
        normal_fill_16_f32(&mut buf, mean, std);
        out[offset..offset + 16].copy_from_slice(&buf);
    }
    out
}

/// `at::normal_distribution<double>::operator()` -- path B.
///
/// `normal_kernel`'s scalar branch instantiates this with `T = double` for
/// *every* dtype, so a `float16` tensor of 5 elements consumes 64-bit uniforms
/// and rounds at the very end. Assuming the accumulate type followed the
/// tensor dtype was the first thing docs/RNG.md §1.3 got wrong, and it
/// disagreed with torch on all eighteen combinations it tried.
#[inline]
fn normal_sample_f64(gen: &mut CpuGenerator, mean: f64, stdv: f64) -> f64 {
    // `transformation::normal` is `val * std + mean`, contracted by clang into
    // an fma for the same reason `uniform_real`'s affine step is.
    if let Some(cached) = gen.next_double_normal.take() {
        return cached.mul_add(stdv, mean);
    }
    let u1 = uniform_real_f64(gen.random64(), 0.0, 1.0);
    let u2 = uniform_real_f64(gen.random64(), 0.0, 1.0);
    // `log1p(-u2)`, not `log(1 - u2)`: they differ in the last bits.
    let r = (-2.0f64 * (-u2).ln_1p()).sqrt();
    let theta = 2.0f64 * std::f64::consts::PI * u1;
    let sample = r * theta.sin();
    gen.next_double_normal = Some(sample);
    let ret = r * theta.cos();
    ret.mul_add(stdv, mean)
}

/// Path B for a whole tensor: `size < 16`, or any non-contiguous tensor.
pub fn normal_serial(gen: &mut CpuGenerator, size: usize, mean: f64, stdv: f64) -> Vec<f64> {
    (0..size).map(|_| normal_sample_f64(gen, mean, stdv)).collect()
}

// ---------------------------------------------------------------------------
// Seeding, as `torch.default_generator` sees it
// ---------------------------------------------------------------------------

/// `torch.manual_seed`'s remap of a negative seed, from its own docstring:
/// "Negative inputs are remapped to positive values with the formula
/// `0xffff_ffff_ffff_ffff + seed`".
pub fn normalise_seed(seed: i128) -> Option<u64> {
    let value = if seed < 0 {
        (u64::MAX as i128).checked_add(seed)?
    } else {
        seed
    };
    u64::try_from(value).ok()
}

pub fn register(m: &pyo3::Bound<'_, pyo3::types::PyModule>) -> pyo3::PyResult<()> {
    use pyo3::prelude::*;

    /// Behind `torch.default_generator.manual_seed(...)`; see
    /// `bootstrap.py::_install_default_generator`.
    #[pyfunction]
    #[pyo3(name = "_shim_manual_seed")]
    fn shim_manual_seed(seed: i128) -> PyResult<u64> {
        let seed = normalise_seed(seed).ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "Overflow when unpacking long: {seed} is outside the inclusive range \
                 [-0x8000_0000_0000_0000, 0xffff_ffff_ffff_ffff]"
            ))
        })?;
        let mut gen = default_generator();
        gen.set_current_seed(seed);
        Ok(seed)
    }

    /// `torch.initial_seed()`.
    #[pyfunction]
    #[pyo3(name = "_shim_initial_seed")]
    fn shim_initial_seed() -> u64 {
        default_generator().current_seed()
    }

    /// `torch.seed()` -- reseed non-deterministically and report the seed used.
    #[pyfunction]
    #[pyo3(name = "_shim_reseed")]
    fn shim_reseed() -> u64 {
        let seed = nondeterministic_seed();
        default_generator().set_current_seed(seed);
        seed
    }

    m.add_function(wrap_pyfunction!(shim_manual_seed, m)?)?;
    m.add_function(wrap_pyfunction!(shim_initial_seed, m)?)?;
    m.add_function(wrap_pyfunction!(shim_reseed, m)?)?;
    Ok(())
}
