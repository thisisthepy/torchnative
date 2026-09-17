# The wall nine of ten shared was the `Python` dispatch key, not a result type

`docs/graph/VARMEAN.md` §4 localised `torch.export`'s stop to one line and named
it a result *type*:

```
  File "torch/fx/experimental/proxy_tensor.py", line 714, in extract_val
    return val.__class__([extract_val(x) for x in val])
TypeError: return_types_native_layer_norm.__new__() missing 2 required
           positional arguments: 'out1' and 'out2'
```

and §4.2 named the next measurement — `_cached_dispatch_impl`'s cache — as this
round's first task. That measurement was made. It moves the cause **two levels
below** where the traceback points, and the repair is in neither of the two
places the question was framed around.

Read these four things first, because they decide how the rest should be taken:

* **The count moved: 0 of 10 → 7 of 10**, under the replay-and-agree bar, and
  the seven also agree with *upstream's* replay cross-side. §5 reports it.
  Three rounds in a row reported no movement and were right each time; this one
  is not that report.
* **There is no structseq anywhere in this story.** `torch.return_types
  .native_layer_norm` does not exist on upstream either (§1). Upstream's own
  fake dispatch returns the identical `_out_wrapper` NamedTuple this shim
  returned. The divergence is what happens to that object *after* the mode
  answers.
* **Two defects, one wall.** The `Python` dispatch key was not carried on the
  tensor (§2), and the dispatcher did not re-box a mode's answer through the
  schema (§3). Either one alone leaves the wall standing; §4.4 shows that as a
  nullification.
* **Neither of `VARMEAN` §4.1's rejected explanations was re-derived, and
  neither is overturned.** The `_dispatch_has_kernel_for_dispatch_key(...,
  "Meta") == False` gap is still real, still not this wall, and still
  unlanded — this round did not touch it. §6.

---

## 1. What upstream actually returns, measured on both sides

The one-line divergence `VARMEAN` §4 recorded is reproduced exactly:

```python
with FakeTensorMode(): torch.ops.aten.native_layer_norm.default(x, [6], w, b, 1e-5)
#   upstream:   <class 'tuple'>
#   this shim:  return_types_native_layer_norm
```

Two further measurements change what it means.

**Upstream has no `torch.return_types.native_layer_norm`.** Asking for it is an
`AttributeError` on both sides. So the reading that upstream returns a
C-level `PyStructSequence` which accepts a single iterable, and that this shim
should build one, was never available: there is no structseq to build.

**Upstream's `_dispatch_impl` returns the same NamedTuple this shim does.**
Instrumenting `FakeTensorMode._dispatch_impl` directly:

```
                        upstream                          this shim
_dispatch_impl      return_types_native_layer_norm    return_types_native_layer_norm
op(...) 1st call    tuple                             return_types_native_layer_norm
```

The object is identical where it is produced and different where it is
received. Everything below is about the two things that happen in between.

---

## 2. Defect one: the `Python` dispatch key lived nowhere

`rust/torch_c/src/aten.rs`'s door consulted the **mode stack** and nothing
else — `any_dispatch_mode_active` reads
`torch.utils._python_dispatch._is_in_torch_dispatch_mode`, and when that is
false the call goes straight to the dense path.

That is not the whole of upstream's dispatcher. A tensor subclass that
overrides `__torch_dispatch__` is dispatched to **by virtue of being in the
arguments**, with no mode entered anywhere: `DispatchKey::Python` is set on the
tensor's own key set. It is why `torch/_tensor.py:457` computes a `types` tuple
at all, and this tree already had the function that computes it
(`overriding_types`) — used only to *describe* the call to a mode, never to
route one.

The consequence, measured with every mode popped (which is precisely the state
`FakeTensorMode.dispatch` leaves before running a `fake_impls` handler, and
therefore the state a `_refs` body runs in):

```
                    upstream                     this shim, before
fx - fx             FakeTensor  device=cpu       Tensor  device=meta
torch.rsqrt(fx)     FakeTensor  device=cpu       Tensor  device=meta
aten.add.Tensor     FakeTensor  device=cpu       Tensor  device=meta
aten.var_mean.dim   FakeTensor  device=cpu       Tensor  device=meta
```

**It was not `native_layer_norm`-shaped and never was.** Subtraction was
broken in this state. `native_layer_norm` was simply the first op in the ten
architectures' path that (a) runs its ref body under an empty stack and (b)
returns more than one value.

### 2.1 The chain from there to `extract_val`

Each link measured, not reasoned:

1. `_refs.native_layer_norm`'s body computes on `FakeTensor` inputs with an
   empty mode stack, and — without the key — every intermediate came back a
   bare `meta` tensor. All three outputs were `Tensor device=meta`.
2. `_make_cache_entry` requires every element of a tuple result to be a
   `FakeTensor`, and raises `_BypassDispatchCache("non-FakeTensor output")`
   when one is not. It did, on every call.
3. A bypass writes a **negative** cache entry under that key, so the op is
   never cached for those arguments again.
4. `_output_from_cache_entry` — whose last line is `return tuple(outputs)`,
   and which is the *only* reason a plain `tuple` was ever observed coming out
   of upstream's cache — was therefore never reached.

Step 4 is why the first reading of the divergence pointed at the cache. It is
real, and it is not the mechanism either: §3 shows the plain tuple upstream's
*caller* sees on the very first, uncached call.

### 2.2 What the repair is, and what makes it narrow

`subclass_dispatch_target` picks the first argument whose type is a `Tensor`
subclass overriding `__torch_dispatch__`; `dispatch_through_subclass` hands it
the call. Two properties are load-bearing and each has a test:

* **`NotImplemented` is an answer, not an error.** A subclass returns it when
  it does not recognise another subclass in the arguments
  (`fake_tensor.py:1047` says so); the fall-through is the dense path, which is
  what this door did before the key existed.
* **`no_dispatch()` suppresses it.** `in_kernel_invocation_manager` runs the
  real meta kernel inside `torch._C._DisableTorchDispatch()` *with the
  `FakeTensor`s it is computing from still in hand*. A key that ignored that
  guard re-enters `FakeTensor.__torch_dispatch__` from inside the call
  `FakeTensor.__torch_dispatch__` made. The guard is this tree's existing
  `_shim_dispatch_suppressed` counter, read before the subclass is consulted so
  that the ordinary path pays nothing for it.

The ordinary path — no mode, no subclass — costs one `PyObject_TypeCheck` per
argument: a plain `Tensor` is skipped on a pointer compare against
`torch.Tensor` before any attribute is touched, and a non-tensor fails the
instance check in C. The Python call that reads the suppression guard happens
only once a subclass has actually been found.

### 2.3 Upstream segfaults when asked this question, so it is not the oracle

The `no_dispatch()` half cannot be compared against upstream. On torch 2.13.0
in this venv, with the mode stack popped, the **first** operator applied to a
`FakeTensor` inside `torch._C._DisableTorchDispatch()` takes SIGSEGV — exit
139, before any output is flushed:

```python
with pd._disable_current_modes():
    with torch._C._DisableTorchDispatch():
        fx - fx          # SIGSEGV
```

So `test_no_dispatch_still_suppresses_subclass_dispatch` compares the shim
against *itself across the guard* — inside it the subclass must not be
consulted, outside it the same op on the same tensor must be — which is the
property that matters and needs no second side. It is written that way rather
than skipped, and the crash is recorded here rather than worked around,
because a test that quietly dropped the case would be CLAUDE.md §5.5's shape.

---

## 3. Defect two: `OpOverload.__call__` re-boxes, and this shim did not

With §2 repaired, all three outputs are `FakeTensor`s, the cache entry is made,
and a *repeat* dispatch returns a plain `tuple`. **The export still stopped**,
because export dispatches `native_layer_norm` exactly **once** — measured, on
both sides — and on that one call `_cached_dispatch_impl` returns the
NamedTuple on both sides too.

Upstream's caller nevertheless sees a plain tuple. The step in between is
`OpOverload.__call__`: upstream does not hand back the object
`__torch_dispatch__` returned. It converts that object to IValues **per the
schema** and boxes the IValues back out.

Measured directly, with no `FakeTensor` involved — an ordinary
`TorchDispatchMode` that deliberately returns the wrong box:

```python
class Renaming(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        ...
        return NAMED(*r)          # a namedtuple

with Renaming():
    r = torch.ops.aten.native_layer_norm.default(x, [6], w, b, 1e-5)

#   upstream:   tuple            <- the mode did not get to choose
#   this shim:  NAMED            <- the object, passed straight through
```

So `proxy_tensor.py:714`'s `val.__class__([...])` — a reconstruction every
plain `tuple` survives — raised here and nowhere upstream. The `_out_wrapper`
NamedTuple never escapes upstream's dispatcher at all.

### 3.1 The rule, both halves measured

Not read off upstream's source — driven through the probe above, one op per
row:

| schema | upstream re-boxes to |
|---|---|
| more than one return (`native_layer_norm`, `max.dim`, `min.dim`, `sort`, `topk`, `var_mean.correction`) | `tuple` |
| one return of list type (`split.Tensor`, `unbind.int`) | `list` |
| one Tensor return (`relu`) | unchanged |

`reshape_to_schema` implements exactly that, on both the mode path and the
subclass path. A result that is *already* a plain `tuple` or a plain `list` is
returned untouched on a pointer compare, without the schema being read — which
is every op whose mode did not build a named result.

---

## 4. The design decision, and why it is neither of the two offered

The question was whether to fix **this op** or **the way this shim spells
multi-value returns**. The measurement says it is neither, and the third answer
is better than both:

* **Per-op** would have taught `native_layer_norm` to return a structseq. That
  makes it agree with upstream by a mechanism upstream does not use — there is
  no structseq (§1) — and leaves `native_batch_norm`, whose result class is the
  same `_out_wrapper` NamedTuple, to raise the identical `TypeError` under a
  different name. Measured before choosing: **six** result classes rejected
  `extract_val`'s reconstruction, not one.
* **Respelling every multi-value return** is the surface change the brief
  warned needs its own evidence — and it would have been wrong in the same
  direction, because the shim's `max`/`min`/`sort`/`topk` classes are not what
  upstream's dispatcher hands back either.
* **Re-boxing in the dispatcher** is one rule, in one place, and it is
  upstream's own. It fixes all six at once and the `split`/`unbind` list rule
  with them, and it changes no operator's implementation.

### 4.1 How many of the ten, and how many ops, were actually involved

Measured before choosing, as the brief required:

* **Nine of the ten** stopped at this wall; the tenth
  (`aimv2_vision_model`) stopped at `FakeTensorDeviceMismatchError`.
* **Six** result classes reject `extract_val`'s reconstruction:
  `native_layer_norm`, `native_batch_norm`, `max.dim`, `min.dim`, `sort`,
  `topk`. `var_mean` and `native_group_norm` already returned plain tuples and
  are the reason the wall surfaced under one name rather than eight.
* The §2 defect is not counted in ops at all: `fx - fx` was wrong.

### 4.2 The tenth architecture came along

`aimv2_vision_model` was `docs/graph/STRIDE.md` §6's deferred
`FakeTensorDeviceMismatchError`, and `VARMEAN` §3 left it deferred. It now
exports, replays and agrees. That is consistent with §2 being its cause too —
bare `meta` outputs leaking a device into an operation whose other argument
carried `cpu` is exactly the shape of that error — but **this round did not
measure that it was the cause**, only that the architecture passes. Stated as
an observation, not a claim.

### 4.3 What this round deliberately did NOT do

* **No structseq was built**, for §1's reason.
* **No operator's implementation changed.** The diff is the dispatcher door and
  tests.
* **The per-op `Meta` dispatch-key predicate of `VARMEAN` §4.1 was not
  landed.** It remains a real gap — 408 of upstream's 2083 aten overloads
  answer `False` there, so the blanket answer is wrong — and it remains not
  this wall. It was not re-derived and not touched.
* **The vendored tree was not modified.** `_refs/__init__.py` and
  `fake_tensor.py` are upstream's, script-generated, and the repair would have
  been invisible the next time they are regenerated.

### 4.4 Nullification: 5 attempted, 0 uncaught

Each written into the source, rebuilt (`touch`ed first — `EXPORT6` §4.1's
stale-mtime trap), installed, run, and reverted.

| | nullification | red |
|---|---|---|
| N1 | the whole repair absent (the round's starting state) | 5 tests |
| N2 | the `no_dispatch()` guard is not read | 6 |
| N3 | `reshape_to_schema` never re-boxes | 2 |
| N4 | single returns are re-boxed too (the `relu` control) | 1 |
| N5 | a list-typed single return is boxed as a `tuple` | 1 |

**N3 is the one worth keeping.** With §2's repair in place and §3's removed,
five of the seven tests stay green — FakeTensors are produced, the cache entry
is made, the repeat dispatch is a plain tuple — and the **export test is red**.
That is the round's own evidence that the two defects are independent and that
neither alone clears the wall. A round that had stopped at §2 would have had
six green tests and no movement in the count.

N2 reddening *six* tests rather than one is not over-reach: without the guard,
`in_kernel_invocation_manager` recurses, so every path that reaches a meta
kernel fails.

---

## 5. The numbers, under the bar

`export_sweep.py --limit 40`, both sides, load average 3.5 (checked with
`uptime`; the other worktree's round was idle).

```
                                     before   after
architectures swept                      40      40
upstream exported+replayed+agreed        10      10
this shim, of those ten                   0       7     <- THE BAR
```

**0 of 10 → 7 of 10.** Not "export() returned": each of the seven returned an
`ExportedProgram`, replayed it, and agreed element-wise with its own eager
module — and separately with *upstream's* replay:

```
cross-side: shim replay vs upstream replay, 7 comparable
            agreeing within 1e-05: 7/7
            worst:  aimv2_vision_model 1.0e-06   altclip 1.0e-06   albert 1.0e-06
                    beit 1.0e-06   bert 1.0e-06
                    bert-generation 0.0   camembert 0.0
```

Ten is still the ceiling and still upstream's: the other thirty are refused by
upstream itself in this sweep. Nobody should read more into the number.

All forty, export-stage failures only, by first wall:

| wall | `VARMEAN` §3 | after |
|---|---:|---:|
| `return_types_native_layer_norm.__new__()` | 14 | **0** |
| `FakeTensorDeviceMismatchError` | 5 | **0** |
| `aten.lift_fresh_copy.default` | 7 | 9 |

Where the ten stop now:

| architecture | before | after |
|---|---|---|
| `albert`, `bert`, `bert-generation`, `camembert` | §4's wall | **agrees** |
| `altclip`, `beit` | §4's wall | **agrees** |
| `aimv2_vision_model` | `FakeTensorDeviceMismatchError` | **agrees** |
| `big_bird`, `blip`, `canine` | §4's wall | `aten.lift_fresh_copy.default` |

**The three that remain all stop at one place, and it is a wall `VARMEAN` §3
already had on the board** at 7 of the forty — it was never behind the layer-norm
wall, it was beside it. It is the next round's first task, and it is a named
operator rather than a dispatcher question.

### 5.1 The four-line reproducer

`VARMEAN` §4's module, verbatim in shape, is the export test and it now clears
the full bar rather than the export stage:

```
EXPORT: exported+replayed+agreed, worst relative 5.158e-08;
        2 distinct call targets in the graph
```

The tolerance is derived, `docs/numerics/AGREE.md` §2's method — the p90 of
upstream's own float32-vs-float64 relative error on these very outputs, floored
at 8 ulp. There is no constant in the test for anyone to widen.

The call-target assertion is there for `docs/graph/COMPILE.md`'s reason: an
`ExportedProgram` holding no operators would replay and agree trivially.

---

## 5.2 The gate

Two full runs, both clean and identical:

```
run 1   GATE_EXIT=0   suites 90/90   ok=1736   FAIL=0   VULKAN 33 ran / 0 skipped
run 2   GATE_EXIT=0   suites 90/90   ok=1736   FAIL=0   VULKAN 33 ran / 0 skipped

        golden self-test  PASS -- 26 comparators x 11 fault modes,
                          0 problems, 0 comparators never exercised
        DOCWATCH          PASS -- 1288/1288
```

Against the `develop` baseline of 1729 ok / 0 FAIL / 89 suites / DOCWATCH 1281
/ golden 11606 cases, ops=307, every delta accounts for itself:

| | baseline | here | why |
|---|---:|---:|---|
| ok | 1729 | 1736 | **+7** = this round's 7 new tests |
| FAIL | 0 | 0 | |
| suites | 89 | 90 | **+1** = `test_structseq.py`; the ledger discovers `test_*.py` by glob and accounted for all 90 |
| DOCWATCH | 1281 | 1288 | **+7** = this round's 7 markers |
| golden ops | 307 | 307 | unchanged -- **no operator was added or altered** |
| golden cases | 11606 | 11606 | unchanged, same reason |

**The two unchanged golden rows are the load-bearing ones.** This round is a
dispatcher change, and a dispatcher change that moved a single kernel's numbers
would show up there. It did not: 11606/11606 pass, 0 failed, 0 pending,
ops=307, exactly as before.

Between the runs, `docs/graph/VARMEAN.md`'s forward note was added; run 2
includes it and the DOCWATCH count is unchanged because a prose note carries no
markers.

---

## 6. This round, separated

**Features added** — capability the shim did not have:

1. The `Python` dispatch key: an op with a tensor-subclass argument reaches
   that subclass's `__torch_dispatch__` with no mode entered (§2).
2. Schema re-boxing of a mode's or subclass's answer, both halves of upstream's
   rule (§3).

**Defects fixed**: both of the above are also defects — `fx - fx` on a
`FakeTensor` with the mode popped returned a bare `meta` tensor, and a mode was
allowed to choose the result class. They are listed once, as features, because
the surface they add and the defect they close are the same code.

**Tests added**: 7, all in `rust/torch_c/pytests/test_structseq.py`, each
comparing against upstream torch's own answer in a second subprocess. One of
them (`test_no_dispatch_still_suppresses_subclass_dispatch`) compares the shim
against itself instead, because upstream segfaults on the question — §2.3.

**Tests corrected**: none.

**Docs corrected**: none. `VARMEAN` §4's localisation is not wrong about where
the traceback is; it is incomplete about why, and this file is the continuation
its own §4.2 asked for rather than a correction to it.

**Docs**: this file.

**Removed**: nothing.

**Measured and deliberately NOT written**: §4.3's list — no structseq, no
operator changes, no `Meta` dispatch-key predicate, no vendored-tree edits.

<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs subclass_dispatch_target present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs dispatch_through_subclass present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/src/aten.rs reshape_to_schema present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_structseq.py test_a_fake_tensor_argument_reaches_its_subclass_with_every_mode_popped present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_structseq.py test_no_dispatch_still_suppresses_subclass_dispatch present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_structseq.py test_the_dispatcher_reboxes_a_modes_answer_into_the_schemas_shape present -->
<!-- DOCWATCH: symbol-in-file rust/torch_c/pytests/test_structseq.py test_a_four_line_layer_norm_module_exports_replays_and_agrees present -->
