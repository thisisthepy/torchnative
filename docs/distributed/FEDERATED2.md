# FEDERATED2: rounds > 1: multi-round FedAvg across two operating-system processes

> **Names since 2026-11-08 (owner decision, docs/INTENT.md §4).** `torchnative.nn.federated` is now `torchnative.federated` (`torch.federated`), `Engine` is `torchnative.federated.FederatedDataParallel`, and `FedAvg`, `FedAvgM` and `FedProx` live in `torchnative.federated.algorithms`. The text below is the record of its round and keeps the names that round used.

> **Local training since 2026-11-08 (owner: "FL이랑 TTA가 왜 묶여. 풀어.").** `FederatedDataParallel` no longer runs the local step through `torchnative.adapt`. It is a `torch.nn.Module` wrapper like `DistributedDataParallel`: the caller trains with any torch optimizer in an ordinary loop, then calls `synchronize(weight=...)` once per round. `participate(...)`, `rounds=`, `method=`, `lr=`, `aggregator=` and `Round.steps` / `Round.history` are gone; the measurements below were taken on the earlier adapt-driven local step and are kept as the record of that run.

`docs/distributed/FEDERATED.md` landed one round of `FedAvg` between two OS processes,
with `torch.equal` acceptance against the same average computed centrally.
This document is the second of those, and it records what makes N rounds
work (the `re_snapshot` lifecycle) and what would go wrong without it.

Measured 2026-09-03, host `darwin/arm64`, CPython 3.13, upstream torch 2.13.0
(`/Volumes/macMini/caches/spike-venv`).

---

## 1. What "rounds > 1" needed, and why it stopped

`docs/distributed/FEDERATED.md` §4 named the missing piece:

> Rounds > 1 (needs re-opening the delta over the aggregated weights,
> `Adapted.online()` never re-snapshots)

The snapshot lifecycle before this round:

```
online()        if _delta is None: snapshot base; create optimizer
                else: just set _online = True
step(...)       forward + capture + backward + optimizer.step + delta.record
revert()        delta.revert (restore from base copy)
```

`online()` is idempotent: it takes the snapshot exactly once, at construction.
That is correct for single-round adaptation where `revert()` restores the
original model. But for multi-round federation, after round 1 installs the
aggregate, the model holds `base + aggregate_round1`, and calling `online()`
again does **not** update the base, so round 2's delta is measured from the
original model and re-sends round 1's cumulative movement. The model diverges
by compounding.

### 1.1 The fix: `Delta.re_snapshot(model)`

A new method on `Delta`:

```python
def re_snapshot(self, model):
    """Replace the base with the current live weights and zero the value."""
    params = dict(model.named_parameters())
    self.base = {n: params[n].detach().clone() for n in self.base}
    self.value = None
    return self
```

Called by `FederatedDataParallel.synchronize` at the end of every round:

```python
delta.record(self.module)         # the caller's local training, as an offset
table = delta.publish(...)        # FedAvg aggregate
delta.value = dict(table)
delta.apply(self.module)          # install base + aggregate
delta.re_snapshot(self.module)    # next round measures from HERE
```

After `re_snapshot`:
- `delta.base` holds the aggregated weights (the model's current state)
- `delta.value` is `None` (the zero delta)
- The next call to `step()` will record movement from the aggregated weights

### 1.2 What is NOT reset

The **optimizer state** is not reset between rounds. Momentum and variance from
the previous round carry into the next. Whether that matters depends on the
optimizer and the task; it is named here rather than silently decided either way
(DESIGN.md §6).

---

## 2. The acceptance measurement: N rounds, distributed == central

Two OS processes, same setup as `docs/distributed/FEDERATED.md` §2: `TCPStore` over
`tcp://127.0.0.1:<port>`, `init_process_group(backend="local", ...)`,
deterministic `TinyLM` with the same base on both ranks, different local data
and different weights (3 and 7).

Three rounds, three adaptation steps each.

*[Results will be filled in once the test suite passes.]*

### 2.1 Distributed against central, per round, element for element

At every round *k*:
- Both ranks start from the same base (verified by JSON equality of the base)
- Each rank produces a local delta from that base
- The weighted average `(3·d0 + 7·d1) / 10` is computed centrally from those
  deltas
- `torch.equal(distributed, central)`: exact, not a tolerance

The critical property: the base at round *k+1* is `base_k + aggregate_k`,
which is the aggregated model from round *k*. Without `re_snapshot`, this would
be the original base at every round.

### 2.2 The stale-base control

The same three rounds are also run *without* `re_snapshot`. In this path:
- The delta always measures from the original base
- Round 2's delta includes round 1's movement plus round 2's local adaptation
- The model diverges by compounding

The control asserts that the stale-base final weights **differ** from the
correct ones, and by more than float noise (>1e-3). This is the proof that
`re_snapshot` matters: without it, the test goes red.

---

## 3. The `FederatedDataParallel` road

```python
fdp = federated.FederatedDataParallel(model, algorithm=federated.algorithms.FedAvg(),
                                      parameters=["norm1.weight", "norm2.weight"])
optimizer = torch.optim.SGD(fdp.federated_parameters().values(), lr=4.0)
rounds = []
for _ in range(3):                         # one round per iteration
    for _ in range(3):                     # the caller's own local loop
        optimizer.zero_grad()
        loss_fn(fdp(input_ids=ids)).backward()
        optimizer.step()
    fdp.synchronize(weight=n_local_samples)
    rounds.append(fdp._get_federated_logging_data())
```

`synchronize` returns None, as DDP methods do; per-round diagnostics are the dict `fdp._get_federated_logging_data()`. The caller's loop is the round loop. Both ranks end holding
the same weights after all rounds, which is verified in the test.

The wrapper's final weights are checked against the low-level road's final
weights: they must agree, because `FederatedDataParallel` is a *use* of `Delta.publish` and
`Delta.re_snapshot` rather than a reimplementation.

---

## 4. What was scoped out, by name

| | why it refuses rather than approximates |
|---|---|
| `Engine(select=...)` | Unchanged from `docs/distributed/FEDERATED.md` §4 |
| `Engine(allow_missing=...)` | Unchanged |
| `FedAvg()` with no `weight` | Unchanged |
| secure aggregation, differential privacy, compression | Not offered at all |
| aggregators other than `FedAvg` | Unchanged |
| non-CPU devices | Unchanged |
| optimizer reset between rounds | Named in the docstring; the optimizer state carries forward. A `reset_optimizer=True` flag that zeroes momentum would be a flag on this loop, not a new mechanism, but it changes the arithmetic and nobody has measured the difference |

---


> **Correction (2026-09-06, `docs/distributed/FEDERATED3.md`).** Three rows of the table above are no longer true, and are left in place rather than edited away for the reason `docs/distributed/FEDERATED.md` §6 gives. `Engine(select=...)` no longer refuses at construction: `federated.cohort` agrees the participant set across the ranks and only a *proper subset* refuses. `Engine(allow_missing=...)` has become `on_missing=`, whose `'refuse'` policy is implemented, a lost peer raises `federated.RankDropped` and the round is undone. And there are now two aggregators besides `FedAvg`: `FedAvgM` and `FedProx`. Secure aggregation and differential privacy are still not offered, and now refuse by name.

## 5. What was changed

| file | change |
|---|---|
| `torchnative/python/torchnative/delta/__init__.py` | `+re_snapshot(model)` method |
| `torchnative/python/torchnative/federated/__init__.py` | `Engine.__init__` accepts `rounds=N`; `participate` loops N rounds with re_snapshot; returns `list[Round]` |
| `tests/_support/test_shim.py` | 3 new tests, updated worker and assertions |

---

## 6. Regression

*[To be filled in with gate results.]*

<!-- DOCWATCH: count smoke_ok ge 370 -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/delta/__init__.py re_snapshot present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/federated/__init__.py FederatedDataParallel present -->
<!-- DOCWATCH: symbol-in-file tests/_support/test_shim.py test_multi_round_fedavg_equals_the_same_rounds_computed_centrally present -->
<!-- DOCWATCH: symbol-in-file tests/_support/test_shim.py test_multi_round_engine_leaves_both_ranks_holding_the_same_weights present -->
<!-- DOCWATCH: symbol-in-file tests/_support/test_shim.py test_a_stale_base_makes_multi_round_deltas_cumulative present -->
