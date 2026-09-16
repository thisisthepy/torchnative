# `torch.int8`: what the storage half alone would unlock, and what it costs

Round date 2026-09-06. Branch `work/int8`, on develop `9f4557e`. Host Apple M1,
CPython 3.13, upstream torch 2.13.0, `candle-core` 0.11.0.

**This is a sizing document.** It answers three questions with measurements rather
than argument, and it deliberately does not touch the quantisation path.

> **Read `docs/graph/QUANT.md` §2.1, `docs/graph/QUANT2.md` §3 and `docs/numerics/DTYPE.md` §6 first.**
> They close `torch.int8` and their four reasons are still correct as stated.
> Three of the four are about **speed**; only the first is about **storage**, and
> this document separates them.

---

## 0. Answers first

| question | answer | where |
|---|---|---|
| Does a later `candle-core` have `I8`? | **No.** 0.11.0 is `max_version` on crates.io, and `main` at `ddf1b879` (2026-09-04) has no `I8` in `DType` either. A version bump is not available. | §1 |
| What was the vendored candle patched for? | **One `cfg` line**, gating `quantized::tokenizer` behind the `tokenizers` feature. Nothing to do with dtypes. | §1.2 |
| What does adding `I8` to candle actually take? | **+252 net lines across 11 files**, CPU backend only. It compiles under both `--no-default-features` and `--features accelerate`, candle's own suite is **0 failed**, and an `i8` tensor round-trips through construction, add/mul, casts, `sum`, `max`, `t`, `narrow`, `cat` and comparison. | §2 |
| What does the shim side take? | **~6 lines.** `TorchDType::Int8` already exists and is already in five of the six per-tag metadata tables. | §3 |
| How much op surface would that unlock? | §4 | §4 |

---

## 1. Upstream candle, and the vendored copy

### 1.1 There is no later release, and `main` has not added it

    $ cargo search candle-core
    candle-core = "0.11.0"

    crates.io api:  max_version 0.11.0   newest_version 0.11.0   updated 2026-06-26

`https://raw.githubusercontent.com/huggingface/candle/main/candle-core/src/dtype.rs`
fetched 2026-09-06 (`main` at `ddf1b879dc3a`, committed 2026-09-04) declares

    U8, U32, I16, I32, I64, BF16, F16, F32, F64, F8E4M3, F6E2M3, F6E3M2, F4, F8E8M0

and `grep -w I8` over that file returns **nothing**. So the cheap answer — bump the
pin — does not exist. Whatever this costs, it costs a fork.

<!-- DOCWATCH: symbol-in-file rust/torch_c/Cargo.toml '[patch.crates-io]' present -->

### 1.2 The fork, and where it lives

`docs/numerics/FLOAT8C.md` recorded that this crate's `Cargo.toml` has **no `[patch]` section**.
**That is no longer true.** This crate forks `candle-core` 0.11.0 to add `DType::I8`. Upstream
has no `I8` as of `ddf1b879dc3a`, so the fork is the only road; it can be dropped the day
upstream `candle-core` gains `I8` and the pin moves to that release.

**Where it lives (2026-09-15).** The fork was first landed as
`candle-core = { path = "/Volumes/macMini/caches/candle-vendor/..." }` — an absolute path on one
machine, outside the repository. cargo treats a missing `[patch]` path as an error, so every
other checkout, CI and the wheel builds failed at resolution. It is now:

| | |
|---|---|
| `rust/torch_c/Cargo.toml` | `[patch.crates-io] candle-core = { path = "../../vendor/candle-core" }` |
| `vendor/candle-core/` | **committed.** The published crate plus the patch, 113 files, 1.9 MB |
| `vendor/int8-candle-0.11.0-cpu.patch` | the only place the fork is edited |
| `vendor/vendor_candle.sh` | regenerates the tree; `--check` rebuilds it in a temp dir and diffs |

The base is the **published** `candle-core-0.11.0.crate`, pinned by sha256
`5ecb2450…6706` — the checksum crates.io's index records, i.e. the one develop's `Cargo.lock`
carried. Its `.cargo_vcs_info.json` names candle commit `31f35b1`, the commit
`docs/design/CANDLE_DEPS.md` §8 cloned. Published rather than cloned because `cargo package`
normalises `Cargo.toml`: the crate stands alone, where the repository's `candle-core` inherits
from a 24 MB workspace. That size was CANDLE_DEPS.md §9's reason not to vendor; the published
crate is 1.9 MB.

**Committed, not gitignored, deliberately.** `[patch]` is resolved before any build script runs,
so a per-machine tree would need the vendoring step in front of every cargo invocation — two
`vendor/*.sh`, `run.sh`, a dozen cross builds across two CI workflows, `cargo ndk`, the device
scripts — and the first one missed would reproduce this defect. Committed, a fresh clone builds
with plain `cargo build`. The cost is a copy that could drift from its two inputs, and
`rust/torch_c/pytests/test_int8.py` runs `--check` in the gate, including a test that a
drifted copy and a wrong crate are **refused**.

**The patch carried in `vendor/` was not the fork that was built.** Applied to the published
crate it failed one hunk of `dtype.rs` (the hand-built tree held a `dtype.rs.rej`), and it had no
`metal_backend/mod.rs` hunks at all — the four `UnsupportedDTypeForOp` refusals that keep an
`mps` user from a silent fallback existed only in the machine-local tree. The patch was
regenerated from that tree: 12 files, all under `src/`. On `mps` an `int8` tensor refuses with
`candle: unsupported dtype I8 for op to_dtype` — the candle spelling, not `int8`.

**Portability, tested rather than asserted.** A copy of the checkout's tracked and untracked
files at a different path, with a fresh `CARGO_HOME` and target directory, under
`sandbox-exec` denying reads of the old absolute path, the original worktree and
`~/.cargo/registry`:

    pre-fix Cargo.toml + Cargo.lock    cargo build --release          exit 101
        failed to read `/Volumes/macMini/caches/candle-vendor/.../candle-core/Cargo.toml`
    vendor_candle.sh --check           (no cache: fetched static.crates.io, sha256 verified)  exit 0
    fixed Cargo.toml + Cargo.lock      cargo build --release --locked exit 0
        Compiling candle-core v0.11.0 (<copy>/vendor/candle-core); 200 crates downloaded,
        no candle-core crate among them

and `test_int8.py` passed 10/10 against the artefact that sandboxed build produced. Not
tested: a non-macOS host, and the CUDA backend, which the patch does not touch and which
nothing here can compile (§5 item 4 still stands for CUDA).

**The wheel build refused the fork, and so would it have refused the absolute path.**
`tools/wheel/build.py`'s freshness check reads cargo's dep-info and treated any input outside
`rust/torch_c` as "built from a different checkout" — and a path dependency's sources are in
that dep-info. The gate's `test_toolguard_wheel_staging.py` went red on it
(`the build read 52 input(s) from outside .../rust/torch_c, e.g. .../vendor/candle-core/src/accelerate.rs`).
The rule is now the crate plus each `[patch]` `path` that resolves **inside the repository**:
an input elsewhere in the repository, or a `[patch]` at an outside absolute path, is still
foreign. `build.py --self-test` carries both as cases, and the fork case was run red before
the rule changed.

**The tokenizer gate is not carried.** The machine-local tree was also the one CANDLE_DEPS.md §8
made to drop `tokenizers` from the graph — the diff below. That is a separate decision §9
declined; the fork carries only `I8`, and `Cargo.lock` matches develop's except that
`candle-core` has no registry `source`.

<!-- DOCWATCH: symbol-in-file rust/torch_c/Cargo.toml '"../../vendor/candle-core"' present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/Cargo.toml 'candle-vendor' absent -->
<!-- DOCWATCH: symbol-in-file vendor/vendor_candle.sh 5ecb245093b0f791b89d3420c3df9c6d49c60ab63ba54db896bf8a3baf486706 present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_int8.py test_the_committed_fork_is_the_pinned_crate_plus_the_patch present -->

What follows is the history of that machine-local tree. `docs/numerics/FLOAT8C.md` did not say
**what its patch was**. Diffed against the crates.io
package, the vendored `candle-core` differs in exactly two files, and only one of them
is source:

```diff
--- registry/candle-core-0.11.0/src/quantized/mod.rs
+++ candle-vendor/candle-0.11.0-patched/candle-core/src/quantized/mod.rs
@@ -14,7 +14,7 @@
-#[cfg(not(target_arch = "wasm32"))]
+#[cfg(all(not(target_arch = "wasm32"), feature = "tokenizers"))]
 pub mod tokenizer;
```

The other differing file is `Cargo.toml`, and it differs because the vendored tree is a
**git checkout of the workspace** (`version.workspace = true`) rather than the
normalised package crates.io publishes.

    So the earlier vendoring was a build-dependency workaround -- keeping
    `quantized::tokenizer` from dragging in `tokenizers` -- and not a dtype fork.
    It is no precedent either for or against this one, and it was not touched.

---

## 2. Adding `I8` to `candle-core`: measured, not estimated

Method: copy the crates.io package to a scratch directory this worktree alone uses
(`/Volumes/macMini/caches/int8-candle-probe`, **not** the shared
`candle-vendor`), add the variant, and let the compiler enumerate the damage.

### 2.1 The variant alone: 20 errors

Adding `I8` to `DType` and nothing else:

    cargo build --release --no-default-features   ->  20 errors, 22 distinct sites

across `dtype.rs`, `cpu_backend/mod.rs`, `scalar.rs`, `safetensors.rs`, `npy.rs`,
`display.rs`, `convert.rs`. Every one is a non-exhaustive `match` — candle's `DType`
matches are written without wildcards, so **nothing is silently skipped**. That is the
same property `docs/numerics/DTYPE.md` §6.3 noted for this shim's own `Repr` enum, and it holds
on candle's side too.

### 2.2 The full thread: +252 lines, 11 files, and it builds

`I16` is the right template — it is the narrow signed integer candle already carries,
and it has no vector kernel, so its arms are the shape an `I8` arm takes. It appears
**133 times as `DType::I16`/`CpuStorage::I16` and 73 times as the `i16` type**, over 15
files; of those, `cuda_backend/*` (27), `metal_backend/*` (6) and
`quantized/tokenizer.rs` (1) are out of scope for a CPU-only fork.

Cloning each `I16` line and each `I16` match arm to an `I8` one, then fixing what
cloning cannot do, reaches a clean build. The residue after mechanical cloning was
**nine errors**, and they are the interesting part — they are what "add a dtype" costs
beyond bookkeeping:

| residual error | what it is |
|---|---|
| `i8: VecOps` not satisfied (×2) | `cpu/kernels.rs` needs an `impl VecOps for i8` (`min`/`max`/`sum`/`dot`) |
| `no associated function \`i8\` for type parameter B` (×2) | `op.rs`'s `UnaryOpT` and `BinaryOpT` traits need an `fn i8` method, which means **every op impl in the file** — 14 blocks — needs an arm |
| `B::i16_vec` / `B::i16_scalar_vec` type mismatch (×2) | the vectorised binary path names its per-dtype helpers with an underscore, so a word-boundary rename misses them |
| `f.write_i16::<LittleEndian>` (×1) | `byteorder`'s `write_i8` takes **no** endianness parameter, a one-byte type having no endianness |
| `vec![0i16; n]` in `CpuStorage::I8` (×1) | the zero-fill literal is typed |
| `(&CpuStorage::I16(_), DType::I8)` and `(&CpuStorage::I8(_), DType::I16)` not covered (×1) | the cast table is **N×N**; the two cells where source and target are both new cannot be produced by cloning either row |

Final size:

```
  265 lines added, 13 removed, across 11 files
  cpu_backend/mod.rs 140   op.rs 50   dtype.rs 15   cpu/kernels.rs 10
  npy.rs 7   display.rs 7   convert.rs 6   scalar.rs 5   safetensors.rs 5
  cpu_backend/utils.rs 5   sort.rs 3
```

### 2.3 It builds both ways, and candle's own suite stays green

| gate | result |
|---|---|
| `cargo build --release --no-default-features` | **EXIT=0**, 0 errors |
| `cargo build --release --features accelerate` (what ships on Apple) | **EXIT=0**, 0 errors |
| `cargo test --release --features accelerate` (candle's own suite) | **EXIT=0**, 0 failed |

### 2.4 And an `i8` tensor actually works

A probe test in the patched crate (`tests/i8_probe.rs`) passes:

```
Tensor::from_slice(&[-1i8, 2, -3, 4], (2,2))   dtype == I8, round-trips
t + t, t * t                                    exact, in i8
t.to_dtype(F32), .to_dtype(I64), and F32 -> I8  exact both ways
t.sum_all(), t.max(1), t.t(), t.narrow(), cat   all correct
t.gt(zeros)                                     -> u8 mask, correct
t.matmul(&t)                                    is_err() == true
DType::I8.size_in_bytes()                       == 1
```

The `matmul` line is the point `docs/graph/QUANT2.md` §3 makes, confirmed rather than
assumed: **the storage exists and the fast kernel does not.** `I8` buys the container.

### 2.5 What this sizing did **not** establish

- **The +252 lines are a mechanical clone, not a shippable patch.** The build emits
  **9 `unreachable pattern` warnings** — e.g. `npy.rs` gets `"h" | "i1" => DType::I8`
  after `"h" | "i2" => DType::I16` already matched `"h"`, and `dtype.rs`'s
  `is_int()`/`is_float()` or-patterns get a duplicated line. A real patch has to go
  through each of those by hand. Call it a day's work on top, not zero.
- **The `i8` arms are `i8` arithmetic, so they wrap.** `t * t` on `[-1,2,-3,4]` gave
  `[1,4,9,16]`; nothing in this probe drove an overflow, and upstream's own overflow
  behaviour for `int8` was not compared.
- **CUDA and Metal were not touched.** 27 and 6 `I16` sites respectively, plus kernel
  sources in `candle-kernels`/`candle-metal-kernels`, which this sizing did not open.
- **No timings.** Five other agents were running (`load average 21.49` at the start),
  so this round produced no performance numbers and makes no performance claim.

---

## 3. The shim side is ~6 lines, because `Int8` is already there

This is the surprise of the round and it cuts the other way from §2: **almost nothing
in `torch_c` has to change.** `TorchDType::Int8` has existed since `dtype.rs` stopped
wrapping `candle_core::DType` — the tag, the name, the abbreviation, `is_signed`,
`itemsize`, `in_all_dtypes` are all already correct for it, and it is already an entry
in five of the six per-tag metadata tables:

| table | `Int8` present today? |
|---|---|
| `aten.rs::randint_representable` | yes — `("signed char", i8::MIN, i8::MAX)` |
| `aten.rs::c10_name` | yes — `"int8_t"` |
| `aten.rs::scalar_type_name` | yes — `"Char"` |
| `aten.rs::int_range` | yes — `(i8::MIN, i8::MAX)` |
| `info.rs::iinfo_row` | yes — `(8, i8::MIN, i8::MAX)` |
| `aten.rs::promotion_rank` | **no** — its doc comment says why: "only the dtypes `TorchDType::storage()` can hold appear" |

`DType::I16`, the closest analogue for a new narrow integer, appears in only **six**
places in the whole crate:

```
dtype.rs:269    Int16 => DType::I16,          (storage)
dtype.rs:290    DType::I16 => Int16,          (from_storage)
tensor.rs:740   DType::I16 => pour!(I16, i16) (flat_storage)
tensor.rs:972   DType::I16 => pour!(i16)      (to_vec)
aten.rs:6972    DType::I16 => wrapping_abs    (abs)
aten.rs:12049   DType::I16 => wrapping_abs    (abs, second site)
```

So the shim-side change is those six plus one `promotion_rank` row: **seven lines**.
Every table that would otherwise need auditing was already written to name `Int8`,
because the tag has always existed — the dtype has always been *nameable* and only
ever un-*storable*.

    The cost is not distributed between candle and the shim.
    It is ~252 lines in candle and ~7 in this crate.

---

## 4. What `I8` alone would unlock: 135 of 203 ops

### 4.1 Method, and what it cannot see

`uint8` is already storable, so **its arm in every kernel is what an `int8` arm would
look like** — same width class, same absence of a vector kernel, same integer
promotion family. The probe is therefore three calls per op:

| column | what it establishes |
|---|---|
| `upstream(int8)` | whether upstream computes `int8` at all — the oracle |
| `upstream(uint8)` | whether the op is narrow-integer-generic upstream, or `int8`-specific |
| `shim(uint8)` | whether **this build's** kernel path handles a narrow integer |

An op that is `OK` in all three is one an `I8` arm would unlock: upstream computes it,
it is not signedness-specific, and this build's own code already runs it for the
neighbouring dtype.

Each call runs in its own subprocess under a 25 s watchdog, killed by process group —
the same precaution `docs/numerics/FLOAT8B.md` §1 needed, kept because it costs nothing. **No
op hung.**

Arguments come from two recipes: a generic one synthesised from each op's `_schema`
(shape chosen by trying `(2,)`, `(2,3)`, `(1,3,4,4)`, `(1,)` against upstream in
`float32` and keeping the first that returns), and a hand-written table for 58 ops the
generic one could not reach. Between them **169 of 203** ops are judged.

> **A shim/upstream difference the recipe tripped over, worth recording.** The generic
> recipe reads `str(arg.type)` from the schema. On the shim that string carries the
> **alias annotation** — `Tensor(a!)`, `Tensor(a)`, `Tensor(a -> *)` — where upstream
> renders plain `Tensor`. The first run therefore produced 37 spurious "no recipe"
> rows on the shim side only, all of them in-place or view ops. Stripping `(a!)` before
> the type match fixed it. This is a real divergence in how the two spell a schema, and
> anything that dispatches on `str(arg.type)` will hit it.

### 4.2 The result

| | ops | meaning |
|---|---|---|
| **`I8` would unlock** | **135** | upstream computes `int8`, upstream computes `uint8`, this build computes `uint8` |
| must keep refusing | **28** | upstream refuses `int8` by name; this build already refuses `uint8` |
| refuses for a reason `int8` did not cause | **5** | `mm`, `bmm`, `addmm`, `matmul`, `_local_scalar_dense` — §4.4 |
| signedness-specific | **1** | `aten.hardtanh.default`: upstream computes `int8` and **refuses** `uint8` ("cannot do hardtanh on an unsigned type with negative limits"), so the `uint8` proxy says nothing about it |
| unjudged | **34** | neither recipe reached them; listed in §4.5 |

### 4.3 The 135

```
_to_copy _unsafe_view abs abs_ add.Scalar add.Tensor add_.Tensor alias all all.dim
all.dims amax any any.dim arange.start_step argmax bernoulli_.float bitwise_and.Scalar
bitwise_and.Tensor bitwise_not bitwise_or.Tensor cat ceil ceil_ clamp clamp_min clone
constant_pad_nd copy_ cos cumsum detach div.Tensor div_.Scalar embedding empty
empty_like eq.Scalar eq.Tensor erf exp expand expm1 fill_.Scalar flip floor_divide
floor_divide.Scalar full full_like gather ge.Scalar ge.Tensor greater.Tensor gt.Scalar
gt.Tensor index.Tensor is_floating_point isin le.Scalar le.Tensor lift_fresh log log2
lt.Scalar lt.Tensor masked_fill.Scalar masked_select max max.dim max.other min min.dim
min.other mul.Scalar mul.Tensor mul_.Tensor ne.Scalar ne.Tensor neg neg_ new_ones
new_zeros nonzero one_hot ones ones_like permute pow.Tensor_Scalar pow.Tensor_Tensor
randperm reciprocal relu relu_ remainder.Scalar remainder.Tensor repeat reshape roll
rsqrt rsub.Scalar scalar_tensor scatter.src select sigmoid sign sin slice sort split
split_with_sizes sqrt squeeze squeeze.dim squeeze.dims stack sub.Scalar sub.Tensor
sub_.Tensor sum sum.dim_IntList t tanh topk transpose tril triu unbind unsqueeze
view view.dtype where.ScalarOther where.default where.self zero_ zeros_like
```

**This is the case the quantisation path does not cover, and it is most of the
surface.** Not one of these wants a fast int8 matmul. `bitwise_and`/`bitwise_or`/
`bitwise_not` are masks; `gather`/`scatter`/`index`/`embedding`/`nonzero`/
`masked_select` are indexing; `view.dtype`, `_to_copy` and `copy_` are what
`t.to(torch.int8)` and an `int8` checkpoint tensor go through.

### 4.4 The five that would still refuse, and why they are not new

| op | shim refusal on **`uint8`**, today |
|---|---|
| `aten.mm.default` | `candle: unsupported dtype U8 for op matmul` |
| `aten.bmm.default` | same |
| `aten.addmm.default` | same |
| `aten.matmul.default` | a 1-D×1-D shape limit, unrelated to dtype |
| `aten._local_scalar_dense.default` | probe artefact — a 2-element tensor cannot become a scalar |

The first three are **`docs/graph/QUANT2.md` §3's point 3, measured**. candle has no integer
matmul at all, for `u8` or `i8` alike, and §2.4's probe confirms it from the other side
(`t.matmul(&t).is_err() == true` in the patched crate). So an `int8` tensor would
construct and then refuse `mm` — where upstream computes it.

    Whether that is "a half-storable dtype" is the decision this document does not
    make. What it can say is that it is **the same door `uint8` already stands at**,
    with a named message, and that it is 3 ops out of 169 judged rather than ten
    scattered interior failures.

### 4.5 The 34 this round could not judge

```
_grouped_mm adaptive_avg_pool1d adaptive_avg_pool2d add_.Scalar arange arange.start
avg_pool2d baddbmm bitwise_or.Scalar clamp_ clamp_min_ convolution div.Scalar_mode
div.Tensor_mode fill_.Tensor greater.Scalar index_put_ linspace masked_fill_.Scalar
max_pool2d mul_.Scalar multinomial native_batch_norm native_dropout native_group_norm
native_layer_norm nll_loss_forward norm.ScalarOpt_dim pow.Scalar randint randint.low
sub_.Scalar upsample_bilinear2d where.ScalarSelf
```

Neither recipe produced a call these accept. Several are in-place or `Scalar` twins of
ops already in the 135 (`add_.Scalar`, `mul_.Scalar`, `clamp_`, `bitwise_or.Scalar`,
`greater.Scalar`) and would almost certainly join it; several others are float-only
(`native_layer_norm`, `avg_pool2d`, `upsample_bilinear2d`) and would almost certainly
join the 28. **Neither guess is measured, so neither is counted.**

---

## 5. Verdict: sized, not landed

The round's own criterion was "if it turns out small and self-contained, land it".
It is small. It is **not self-contained**, and that is the whole of why this document
ends here rather than in a diff.

1. **It requires forking a pinned crates.io dependency into the shipped build.**
   §1.1 removes the cheap alternative: there is no later `candle-core`, and `main`
   has not added `I8` either. So landing means a `[patch]` section, and `Cargo.toml`
   deliberately has none today (`docs/numerics/FLOAT8C.md`, re-verified §1.2). That reaches
   every platform this crate builds for — Android, iOS device and simulator, wasm,
   and every wheel target — and this round verified **host CPU only**.
2. **The +252 lines are a clone, not a patch.** Nine `unreachable pattern` warnings
   (§2.5) say so out loud. Landing them as-is would put dead arms into a vendored
   dependency, which is the sort of thing that reads as intentional a year later.
3. **`mm`/`bmm`/`addmm` would diverge from upstream** (§4.4). Three of 169 judged,
   at a named door, but a divergence to be decided on rather than assumed.
4. **CUDA and Metal were not touched** and 33 `I16` sites there are unexamined.

> **Superseded 2026-09-15.** The fork has landed (§1.2), and item 1's objection is answered
> by committing the published crate under `vendor/candle-core` rather than by an absolute
> path. The patch named in the next paragraph moved to `vendor/`, and the copy that was
> carried **did not apply** — it has been regenerated from the tree that was actually built.
> Items 3 and 4 are unchanged.

What is landed instead is the evidence: `vendor/int8-candle-0.11.0-cpu.patch` is the
exact +252-line diff that compiles under `--no-default-features` and
`--features accelerate`, keeps candle's own suite at 0 failed, and passes the
functional probe in §2.4. Applying it is a `[patch.crates-io]` entry and the seven
lines of §3 away from `torch.tensor([1], dtype=torch.int8)` working.

## 6. What this document could not establish

- **Any performance number.** Five other agents were running throughout
  (`load average 21.49` at the start, `7.65` at the lowest); per `docs/graph/QUANT.md` §1
  and CLAUDE.md's rule on measurement isolation, no timing was taken and none is
  claimed.
- **Whether the 135 would be *bit-identical* to upstream.** The probe establishes
  that upstream computes and that this build computes the neighbouring dtype. It does
  **not** compare values — that is the golden suite's job and it cannot run on a
  dtype that does not construct.
- **`int8` overflow behaviour.** §2.4 exercised no overflow, and upstream's wrapping
  semantics for `int8` were not compared against candle's. **Closed 2026-09-15** by
  `rust/torch_c/pytests/test_int8.py`, exact against upstream in the same interpreter:
  ordinary arithmetic, the wrap-around edges (`127 + 1`, `-(-128)`, `abs(-128)`, scalar
  forms, and `sum` widening to `int64`), casts into `int8` from six dtypes and out of it to
  eight, and `int8 × uint8 → int16` in both orders. Each was broken on purpose and went red:
  saturating `i8` arithmetic in the fork (edges), a clamping `int64 → int8` cast (casts in),
  and the `Int8`/`UInt8` rule removed from `promote_types` (promotion). Grade **agrees** for
  those cases only; the 135 of §4.3 are still unjudged.

<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_int8.py test_int8_wraps_at_the_edges_exactly_where_upstream_wraps present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_int8.py test_int8_with_uint8_promotes_to_int16_in_both_orders present -->
- **The 34 of §4.5**, and **CUDA/Metal**.
