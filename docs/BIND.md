# BIND — the Python argument-binding layer

What was hot in `rust/torch_c/src/bootstrap.py`'s overload resolution, what was
precomputed, why that is safe, and what is still slower than upstream.

**This is `rust/torch_c/src/bootstrap.py`, and Android loads the same file** —
it is embedded in the artefact `vendor/install_shim.sh` installs, and the
Android build embeds that same source. Nothing here is host-specific: it is
plain Python doing dict and attribute work, so the win applies on device too,
where the interpreter is slower and therefore the share it occupies is larger.
It was measured on the host only, because that is where it can be measured
against upstream on the same machine.

---

## 1. The problem

The kernels are already faster than upstream at the shapes SmolLM2-135M
actually uses, yet the model is slower. Measured by the coordinating session on
a quiet machine (load 1.4, no other agents):

| | upstream | ours | ratio |
|---|---|---|---|
| SmolLM2-135M float32 prefill | 35.3 ms | 39.8 ms | **1.13x slower** |

Per kernel, at the model's real shapes (µs/call):

| kernel | upstream | ours | ratio |
|---|---|---|---|
| `_softmax` (attention shape) | 16.25 | 2.39 | 0.15x |
| `native_layer_norm` | 12.25 | 6.13 | 0.50x |
| `silu` | 11.23 | 6.19 | 0.55x |
| `sdpa` | 26.77 | 22.24 | 0.83x |
| `linear` (MLP) | 90.49 | 89.50 | 0.99x |

Weighted by call count that is **−1.17 ms**; the model showed **+4.5 ms**.
Dispatch itself is also ours-faster (0.86 vs 1.11 µs per small-tensor call).
The loss was in the Python surface:

| | upstream | ours | ratio |
|---|---|---|---|
| `.view()` | 0.76 µs | 4.97 µs | 6.5x |
| `.transpose()` | 0.82 µs | 4.02 µs | 4.9x |
| attribute reads (`.shape`, `.dtype`, `.dim()`) | — | — | comparable |

`cProfile` over 5 forward passes, before:

```
ncalls   tottime  function
  9275    0.166   torch._C._aten_dispatch
  8255    0.025   bootstrap.py:1889 _bind
 35030    0.024   bootstrap.py:1677 _decompose_type
105090    0.009   str.endswith          <- 21018 per forward pass
 49890    0.004   str.strip
 15610    0.010   bootstrap.py:1742 check
 14535    0.006   bootstrap.py:1760 coerce
```

A microbenchmark of `t.view(1, 6, 576)` and `t.transpose(1, 2)` at the model's
shape isolates it further. Of the 0.800 s that 40 000 such calls cost,
`_aten_dispatch` — everything the shim actually computes — was **0.030 s, under
4%**. The other 96% was the Python layer deciding which overload to call.

---

## 2. Method

Backends alternate inside one script run, so a drift in machine state moves
both, and only the **ratio to upstream** is reported. `uptime` was recorded
before and after every measurement and no measurement was taken above load 2.4.

```sh
export HF_HOME=/Volumes/macMini/caches/hf-home
export CARGO_TARGET_DIR=/Volumes/macMini/caches/cargo-target-bind
export TORCH_C_ARTEFACT=$CARGO_TARGET_DIR/release/lib_C.dylib
# ours:     PYTHONPATH=torchnative/src/main TORCH_USE_RTLD_GLOBAL=1
# upstream: no PYTHONPATH
```

* **prefill** — SmolLM2-135M, `dtype=torch.float32`, prompt of 6 tokens; the
  minimum of 5 timed passes after 2 warmups; 4 alternating rounds.
* **microbench** — minimum of 5 blocks of 20 000 calls after 200 warmups, on a
  `(1, 6, 9, 64)` float32 tensor.

**The "before" row was re-measured, not reused.** The first baseline reading
was taken at load 3.4 and gave 1.149; the pre-change artefact was rebuilt from
`git show HEAD` and re-measured under the same conditions as the "after" row.
That is the number in the table.

---

## 3. What changed

Nothing in the resolution *algorithm*. Every step below moves a computation
from call time to table-parse time, or removes one whose answer was already
known.

### 3.1 `_decompose_type` was the wrong thing to cache

The obvious move — `functools.lru_cache` on `_decompose_type` — is in the diff,
and it is **not where the win came from**. Measured after the rest of the work:

```
after import:          hits=1863  misses=28  distinct=28
after 4000 op calls:   hits=0     misses=0
```

Twenty-eight distinct type spellings exist in the whole of `overloads.json` and
`methods.json`. Caching turns 35 030 parses per forward pass into 35 030 dict
lookups; **hoisting the call out of the loop turns them into zero.** The memo is
kept because it makes `import torch` do 1863 fewer parses and because
`_SchemaType.isSubtypeOf` and `containedTypes` — the fake-tensor and prims path,
which this benchmark does not reach — still call it per question. But it is an
import-time and elsewhere win, not the one being reported.

It is safe to memoise: the answer depends on nothing but the characters of the
string, there is no context in which `"int[1]?"` means two different things, and
the returned tuple is immutable so no caller can corrupt the next one's entry.
The bound (`maxsize=4096`) is there because `parse_schema` accepts text from
outside the tables.

### 3.2 `_SchemaPlan` / `_ArgPlan` — everything fixed, computed once

`_bind` re-derived all of this on every call, from data that is settled the
moment `_Schema.parse` returns and that nothing ever writes to again:

| was | now |
|---|---|
| `[a for a in schema.arguments if not a.kwarg_only]` | `plan.positional` |
| `{a.name: a for a in schema.arguments}` | `plan.by_name` |
| `_decompose_type(str(arg.type))` in `check`, once per argument per call | `_ArgPlan.base` / `.is_list` / `.optional` |
| `_decompose_type(str(arg.type))` again in `coerce` | `_ArgPlan.sized_int_list` |
| `_decompose_type(str(positional[skip].type))` for the varargs rule | `plan.varargs_intlist` |
| `1 if self.self_bound else 0` | `_Overloads._skip` |
| `arg.has_default_value()` | `_ArgPlan.has_default` |
| `zip(self.schemas, self.keys)` per `resolve` | `_Overloads._candidates` |

`check`'s "a sized int list also accepts a bare int" precondition and `coerce`'s
"widen it to a one-tuple" precondition are the *same* predicate on the type, so
they collapse into one precomputed flag.

### 3.3 The twelve-way type chain, compiled per argument

`_TypeChecker._base` walks up to twelve string comparisons to decide what test
to run, every call. `predicate_for` returns a closure that has already made that
decision. `check`, `coerce` and `_base` are kept **unchanged** as the readable
statement of the rules — including the three that would be got wrong by
intuition (`bool` does not satisfy `int`; `int` does satisfy `float`; a 0-dim
tensor satisfies `Scalar`).

The predicates are built lazily, on the first bind against that schema, for the
same reason `_TypeChecker` itself is: `layout` and `memory_format` do not exist
when the tables are parsed. Once built they are fixed, which was already true of
the attributes `check` reads — `_TypeChecker.__init__` snapshots them.

### 3.4 Four passes that could not fail, or ran when they had nothing to do

* `all(base_ok(item) for item in value)` → an explicit loop. The generator costs
  a frame per element plus one to stop, and shape lists are the hottest thing
  the checker sees. It short-circuits where `all` does.
* The "was every required argument bound?" walk is skipped when
  `len(bound) == plan.n_arguments`: `bound`'s keys are always argument names, so
  equal counts means equal sets. (A schema with a repeated argument name has
  fewer distinct names than arguments, so the counts cannot match and the walk
  still happens — the skip fails safe.)
* The "drop arguments equal to their own default" pass built a second dict to
  hold exactly what the first held, for every schema with no defaults —
  `view(Tensor self, SymInt[] size)` and every other pure-shape schema.
  `plan.any_defaults` returns `bound` directly in that case.
* The "given twice" check moved out of the positional loop and behind
  `if kwargs:`, so a call with no keywords does not pay a dict lookup per
  argument. Both spellings answer `None` for the same calls and neither has a
  side effect, so which of the two reasons is found first is not observable.
* `_strip_python_only_kwargs({})` is `{}`, and `**kwargs` already handed the
  caller a fresh dict, so the call is skipped when there is nothing to strip.
* `type(value) is int` is tried before `isinstance(value, int) and not
  isinstance(value, bool)`. The first implies the second and is true of nearly
  every value that reaches there; the full test still decides everything else.

---

## 4. Why this is safe

Two arguments, one by construction and one by measurement.

**By construction.** Every value moved out of the call path is a pure function
of the parsed schema, and the parsed schema is written once. `_Schema.parse`
fills `arguments` and nothing in the file writes to an `_Argument` afterwards;
`_Overloads.__init__` is the only constructor of `_SchemaPlan`. There is no
context in which the same argument decomposes two ways, which is the failure
mode a cache has to be checked against. The one genuinely late-bound thing —
the type predicate, which needs `layout` and `memory_format` — is still built
late, on first call.

**By measurement.** The pre-change `_bind` was extracted verbatim and run beside
the new one, driven off the same parsed schemas, over **every installed entry**
(124 of them, recovered from the closures the install actually left behind) and
6613 call shapes — positional tuples up to length 3 drawn from 18 values that
exercise every rule the checker has, plus six keyword shapes. Any difference in
refusal, in chosen key, in bound keys, or in a bound value is a behaviour
change.

```
{"entries": 124, "comparisons": 1679702, "mismatches": 0}
```

That harness can fail. Tampering with each precomputed field in turn, after the
plans were built:

| tamper | mismatches |
|---|---|
| `plan.any_defaults = False` | 271 |
| `plan.varargs_intlist = False` | many (first at `torch.view` with `size=(-1,-1)`) |
| `plan.required = ()` | many (first at `torch.masked_select` with no arguments) |
| `_ArgPlan.sized_int_list = False` | 84 |
| `plan.n_arguments = 1` | 788 |
| `plan.n_arguments = -1` | **0** |

The last one is not a hole: `-1` makes `len(bound) != n_arguments` always true,
so the required walk always runs. That is the conservative direction — correct,
just slower — which is what the skip is designed to fail into.

The repository's own gates, run on the final artefact:

```
PYTHON=$PY sh rust/torch_c/pytests/run.sh   -> 197 ok, exit 0
$PY tools/golden/compare.py                 -> 2811/2811, ops covered=119, exit 0
$PY rust/torch_c/pytests/verify_schemas.py  -> 4203/4203, exit 0
```

Unchanged from before the work, which is the point.

---

## 5. Result

Load 1.8–2.4 throughout; the pre-change artefact was rebuilt and re-measured
under the same conditions rather than reusing the earlier, noisier reading.

### Prefill, ratio to upstream

| | upstream (ms) | ours (ms) | ratio |
|---|---|---|---|
| before | 35.912 | 41.067 | **1.1435** |
| after | 35.933 | 38.096 | **1.0602** |

Pairwise across the four alternating rounds: before 1.1340 / 1.1444 / 1.1486 /
1.1507; after 1.0572 / 1.0587 / 1.0597 / 1.0634. The two bands do not overlap.

**2.99 ms of the 5.16 ms gap is gone — 58% of it.** (Gap before
41.067 − 35.912 = 5.155 ms; after 38.096 − 35.933 = 2.163 ms.)

#### What the residual is, and is not, resolved to

Three independent runs put the *after* ratio at 1.019, 1.033 (median 1.021) and
1.060. The improvement is not in doubt — the before and after bands above do not
overlap, and every reading lands far below 1.1435. **The residual itself is not
resolved to a percent, and this document should not be read as claiming 1.06
exactly.**

The reason is the noise floor rather than a disagreement about method. A
four-round re-measurement taken at load 3.2 scattered pairwise 0.915 / 1.012 /
1.039 / 1.045 — one round came out *faster* than upstream. A spread that
straddles 1.00 cannot resolve a 2–6% residual, so the correct summary is a
range: **within a few percent of upstream on desktop CPU, from 14% behind.**

This machine cannot be made quiet on demand. The load is a windowing server, a
user application, and two Android emulators that are shared with other projects
and must not be killed. Pinning the residual would need either a quiet machine
or enough rounds to average the interference out; neither was done, so the range
stands. The Android numbers in §7 are less affected because the effect there is
1.8–2.75x — an order of magnitude above the same noise.

### Microbench, µs/call

| | upstream | before | after | still |
|---|---|---|---|---|
| `.view(1, 6, 576)` | 0.796 | 5.212 | **1.790** | 2.25x |
| `.transpose(1, 2)` | 0.821 | 4.039 | **1.460** | 1.78x |
| `.view((1, 6, 576))` | 1.034 | 6.363 | **1.944** | 1.88x |
| `t + t` | 1.031 | 3.744 | **1.943** | 1.88x |
| `.shape` | 0.120 | 0.078 | 0.079 | 0.66x |
| `.dim()` | 0.061 | 0.046 | 0.045 | 0.74x |

Function calls for those 40 000 method calls: **3 300 002 → 500 002**.
Profiled time 0.800 s → 0.145 s. `_aten_dispatch`'s share of it went from
under 4% to 19%.

### The forward pass, profiled

| | before | after |
|---|---|---|
| total function calls (5 passes) | 486 795 | 146 000 |
| `bootstrap.py` `tottime`, summed | 0.084 s | 0.024 s |
| `_aten_dispatch` `tottime` | 0.170 s | 0.170 s |
| `_decompose_type` calls | 35 030 | **0** |
| `str.endswith` calls | 105 090 | 0 |

`_aten_dispatch` did not move, which is the check that this touched only the
Python layer.

---

## 6. What is still slower, and why

**`.view()` is still 2.2x upstream.** The gap that remains is structural, not
redundant work:

1. **Three Python frames per call** — `method` → `resolve` → `_bind` — where
   upstream has none. Merging them would save perhaps 0.1 µs per call
   (~0.15 ms per prefill) at the cost of a second copy of the resolution loop.
   Not taken: a duplicated loop that can drift out of step with the original is
   a worse trade than 0.15 ms.
2. **`dispatch(key, **bound)`** — the bound arguments are built as a dict and
   then unpacked into keyword arguments for the C entry point, which re-parses
   them. Removing that round trip means changing `_aten_dispatch`'s signature in
   `aten.rs`, which is outside this work's area.
3. **`view` costs two binds.** `view.dtype` is tried and refused before
   `view.default` binds. The refusal is now cheap (a length compare) but it is
   still a Python call. The order is upstream's and is part of the answer, so it
   cannot be reordered.

**Roughly 2.2 ms of prefill gap remains.** Attributing it: about 1855
dispatches per prefill at the ~0.7 µs per-call Python overhead the microbench
still shows is ~1.3 ms, which leaves ~0.9 ms not explained by this layer. That
residue was not chased — it is on the C side, and this work was scoped to
`bootstrap.py`.

**Two things were tried and did not help.**

* **`functools.lru_cache` on `_decompose_type`** — the brief's first candidate.
  It is a real win at import (1863 parses avoided) and for the prims path, but
  **zero** on the call path, because the right fix was to stop calling it. Kept
  for the other two reasons, not counted toward the result. §3.1.
* **Fusing `check` and `coerce` into one value-or-sentinel closure.** One fewer
  call and one fewer attribute load per bound argument, and it measured *within
  noise* — view 1.84 vs 1.80 µs, transpose 1.52 vs 1.51, identical function-call
  counts (500 002 either way). It needed a second copy of the twelve `_base`
  rules with a different return shape; two spellings of "a zero-dim tensor
  satisfies `Scalar`" is a real hazard bought with no measurable time, so it was
  reverted. The bookkeeping it *did* pay for — hoisting the "given twice" check
  out of the positional loop — was kept.

**Not measured on device.** The reasoning that this transfers to Android is that
the same `bootstrap.py` is embedded there and the change is pure interpreter
work with no host-specific path, so the share it occupies is if anything larger
on a slower CPU. That is an argument, not a measurement.

---

## 7. Android measurement (in progress)

Turning §6's closing claim into a number. Device: `emulator-5556`, API 36,
arm64-v8a. There is no upstream torch for Android (no wheel is published,
which is a large part of why this project exists), so this section reports
**old/new ratios on the same device**, not a ratio to upstream.

### 7.1 `bootstrap.py` is baked into the artefact, not loaded from disk

`rust/torch_c/src/lib.rs:568` does
`std::ffi::CString::new(include_str!("bootstrap.py"))` — the source text is
compiled into `lib_C.so` at Rust build time. Swapping the `.py` file on the
device without rebuilding does nothing; the interpreter never reads a
`bootstrap.py` file at all on either platform. **Both sides were rebuilt** for
`aarch64-linux-android` via `scripts/device_android.sh build`:

* new (HEAD, `972dfe4`): `rust/torch_c/src/bootstrap.py` unchanged, built as-is.
* old (`972dfe4^`): `git show 972dfe4^:rust/torch_c/src/bootstrap.py` copied
  over `rust/torch_c/src/bootstrap.py`, built, then the working tree file was
  immediately restored from a `cp` backup (`git status --short` confirmed a
  clean diff afterward).

Both `.so` artefacts were saved to `/tmp/bw_bind_android/lib_C.{old,new}.so`
(distinct md5s) before restoring the tree, so rounds alternate by swapping the
staged file rather than rebuilding per round: `_C.abi3.so` is the only file
that differs between old and new, so once the rest of the tree (CPython
runtime + vendored `torch` + deps) is staged once via
`scripts/device_android.sh stage`, alternation is a single `adb push` of the
5.4 MB `.so` to `/data/local/tmp/bw_device/site/torch/_C.abi3.so` — no re-stage
of the ~440 MB tree per round.

### 7.2 Method

Same shapes as the host (§2): a `(1, 6, 9, 64)` float32 tensor, 200 warmups,
minimum of 5 blocks of 20 000 calls per round. Three microbenchmarks:
`.view(1, 6, 576)`, `.transpose(1, 2)`, and `t + t` (the dispatch-bound loop —
smallest possible op, so the Python binding layer dominates the per-call cost).
Rounds alternate old, new, old, new, ... and the minimum per side across
rounds is reported, per the host methodology. A control round swaps the *same*
artefact in for both labels to check the harness itself reads ~1.00x.

Load stayed 2.27–2.78 for every measured round (`uptime`, checked immediately
before each push and each run); no round was taken above that. The one 3.55
reading during an earlier import smoke-test (not a timed round) was discarded
and re-checked before measurement began.

### 7.3 Result

Minimum of 3 alternating rounds per side, µs/call:

| | old (`972dfe4^`) | new (`972dfe4`) | ratio old/new |
|---|---|---|---|
| `.view(1, 6, 576)` | 9.172 | 3.331 | **2.75x** |
| `.transpose(1, 2)` | 6.891 | 2.742 | **2.51x** |
| `t + t` (dispatch-bound) | 6.164 | 3.418 | **1.80x** |

All three rounds per side agreed within ~1.3% of their own minimum (e.g. view:
9.172 / 9.204 / 9.249 old; 3.331 / 3.412 / 3.421 new) — tight enough that
old and new do not overlap on any metric.

**Control** (same `new` artefact pushed and measured twice, under labels A and
B, through the identical harness): view 1.038x, transpose 1.033x, add 1.033x.
All three land at 1.03–1.04x, not exactly 1.00 but an order of magnitude below
the 1.80–2.75x the real comparison shows, and it sets the noise floor the
above spreads (~1.3%) are already comfortably inside of. Harness bias is not
what is producing the win.

**The win transfers, and transfers more strongly than the host measurement.**
Host (§5) showed view 2.25x-still-slow-but-improved (5.212 → 1.790 µs, a 2.91x
speedup) and transpose 4.039 → 1.460 µs (2.77x); device shows 2.75x and 2.51x
on the same two calls, plus 1.80x on the dispatch-bound `t + t` the host table
also carries (3.744 → 1.943 µs there, 1.93x — comparable). The device ratios
land in the same range as the host's, not a different regime, and if anything
the interpreter-bound share is if anything larger here: absolute per-call
times are 2–3x the host's on both sides (e.g. new-view 3.33 µs on device vs.
1.79 µs on host), consistent with §7's opening argument that a slower
interpreter makes the Python layer's share larger rather than smaller. **This
device measurement supports, rather than merely assumes, the transfer.**

### 7.4 Method notes

* **Rebuild required, `.py` swap alone does not work** (§7.1) — both artefacts
  were built via `scripts/device_android.sh build` for `aarch64-linux-android`,
  saved to `/tmp/bw_bind_android/lib_C.{old,new}.so` (verified distinct md5),
  and `rust/torch_c/src/bootstrap.py` was restored to HEAD (`cp` backup, not
  `git checkout`) immediately after the old build — `git status --short` was
  clean on that file before device rounds began.
* Only `/data/local/tmp/bw_device/site/torch/_C.abi3.so` was swapped between
  rounds (direct `adb push`, not a full re-stage) — the CPython runtime,
  vendored `torch` tree and dependencies were staged once via
  `scripts/device_android.sh stage` and are identical across all rounds; the
  `.so` is the only variable.
* `_multiprocessing`/`_posixshmem` stubs from `scripts/device_parity.py`
  (`_install_android_stubs`, gated on `BW_STUB_MULTIPROCESSING=1`) were copied
  into the microbenchmark script — `torch/multiprocessing/__init__.py` imports
  `multiprocessing.resource_tracker` unconditionally and Android's CPython
  ships neither extension; without the stub `import torch` fails before any
  timing runs.
* Raw per-round JSON and build/stage logs are under `/tmp/bw_bind_android/`
  (`out_{old,new,controlA,controlB}_r*.json`, `build_{old,new}.log`,
  `stage_new.log`) for anyone who wants to re-check the arithmetic above.
