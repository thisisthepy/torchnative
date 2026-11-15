# FEDSERVER: federated aggregation on the server

Issue #282. BACKEND.md §9 step 15, D14 and D4; SPEC S6.7. Written 2026-11-13.

**Grade: builds.** Every file below passed `python -m py_compile` and nothing else. No suite, gate or probe ran in
the round that wrote it (AGENTS.md rule 10), so nothing here is *reaches* or *agrees* until CI runs
`tests/distributed/test_fedserver.py` (AGENTS.md §16). Read every "equals" below as "is written to equal". The tests
were written in the same round as the code and were not observed red (they cannot be run here); the first CI run is
both their red-or-green and the implementation's.

Provenance. Decided: the server is swappable behind an interface that also carries federated server-side aggregation
(owner, 2026-11-08, BACKEND.md D4: "서버 구현체를 갈아낄 수 있도록 설계해줘"); server-style aggregation through that
interface is a follow-up built with this backend (leader, D14). Everything else in this document is an **agent
proposal, not confirmed**: the option `aggregation='server'` and its default, the route, the server operations
`fed_gather` and `fed_fold` and their message layout, the ledger, the choice to keep FedAvgM's velocity on the ranks,
and every refusal's wording.

## 1. The question and the answer

D14's premise: `torch.federated` aggregated with `all_reduce` among the participants and no server, which by the
confirmed taxonomy is swarm-shaped. The question this round answers is how a `FederatedDataParallel` round reaches the
coordinator of BACKEND.md §4.5 instead, without changing the caller's loop.

```python
import datetime
import torch.distributed as dist
import torchnative.distributed
from torchnative import federated
from torchnative.federated import algorithms

dist.init_process_group(backend="torchnative", init_method="tnserver://127.0.0.1:29500",
                        rank=rank, world_size=3, timeout=datetime.timedelta(seconds=120))
fdp = federated.FederatedDataParallel(model, algorithm=algorithms.FedAvg(), aggregation="server")
for batch in local_batches:          # unchanged
    ...
fdp.synchronize(weight=n_local_samples)   # unchanged
```

`aggregation='all_reduce'` stays the default, so every existing caller and test keeps its path.

## 2. Design (agent proposal, not confirmed)

| Piece | Where | What it does |
|---|---|---|
| Option | `federated/__init__.py` `FederatedDataParallel(aggregation=...)`, `AGGREGATION` | `'all_reduce'` (default) or `'server'`; anything else is a `ValueError` naming both |
| Route | `federated/__init__.py` `_ServerRoute`, `_via_server` | one per `synchronize`; finds the server from the group's own store and names the round's keys |
| Digest checks | `federated/__init__.py` `agree` | inside a route, the schema, base and cohort digests are gathered through the server (`fed_gather`), not with `all_gather` |
| Weight sum | `federated/__init__.py` `_total_weight` | inside a route, folded on the server (`fed_fold` of an empty table) |
| Delta sum | `federated/algorithms.py` `FedAvg.aggregate` | inside a route, the scaled tables go to the server in one `fed_fold`; the division is the same expression as on the `all_reduce` path |
| Server fold | `distributed/backend/server.py` `_FedCore`, `_fold_compute`, `_gather_compute` | waits for every rank of the round, folds in ascending rank order, answers every rank with the same bytes |
| Wire | `server.py` `fed_request`, `_Listener._fed` | a JSON header (metadata only) followed by `wire.py` tensor frames, raw little-endian bytes, both ways |
| Ledger | `server.py` `Server.fed_ledger` | every completed or abandoned operation: key, ranks folded, weights, total, SHA-256 of the result |
| Receipts | `_get_federated_logging_data()["server_receipts"]` | what the server answered for each fold of the round; tests match them to the ledger |

<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/federated/__init__.py _ServerRoute present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/federated/__init__.py _refuse_for_server present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/backend/server.py _fold_compute present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/backend/server.py fed_ledger present -->

### 2.1 Why an option and not automatic selection

Selecting the server automatically whenever the group came from `tnserver://` would change, silently, where the
aggregate of every existing `tnserver://` caller is computed. An explicit option keeps that a visible choice, and the
choice refuses by name when it cannot be served (§4) instead of falling back to `all_reduce`, which would be the silent
fallback AGENTS.md §11 designs against. The algorithm does not choose either: FedAvg's arithmetic is the same on both
paths, so where it runs belongs to the wrapper that owns the round.

### 2.2 How the round reaches the server

The backend keeps the `PrefixStore` that `init_process_group` handed it. Under it, for a group initialised through
`tnserver://`, is the server's client (`ServerStore`), which holds the server's address and the group's timeout. The
route walks `underlying_store` down to it; a group whose store is anything else (a `TCPStore` from `tcp://`, a
`HashStore`) refuses by name.

Keys are `tn_fed/<group token>/route<n>/op<k>`. The group token is the one the mesh's rank 0 publishes at rendezvous,
so two groups on one server never share a round. `n` counts the routes this process opened for the group: every rank
opens one per `synchronize`, at the top, before anything can differ between ranks, so the n-th round names the same
keys everywhere and a round that one rank abandons locally does not shift the next round's keys. `k` counts the
operations inside the route, in program order.

Inside a route, a round is: the cohort digest if `select=` is set (`fed_gather`), the base digest (`fed_gather`), the
schema digest (`fed_gather`), the weighted deltas (`fed_fold`), the weights (`fed_fold` of an empty table). Nothing
of the round goes over the participants' own sockets; the test counts `dist.all_reduce` and `dist.all_gather` and
requires zero. The group's star and mesh are still built by `init_process_group`, because the backend builds them for
every group; they are idle during a server round.

### 2.3 Why the aggregate is the same bits

The `all_reduce` path computes, per parameter, `scaled_k = d_k * w_k` in the table's dtype, sums the `scaled_k` on
`ProcessGroupLocal`'s hub in ascending rank order with `Tensor.add_` in that dtype, `((s_0 + s_1) + s_2) + ...`, and
each rank divides by `torch.tensor(total, dtype)`, where `total` is the binary64 sum of the weights in the same order.

The server path computes the same `scaled_k` with the same expression, sends their bytes, and the server folds them in
ascending rank order with `array('f')` (float32) or `array('d')` (float64): each addition is done in binary64 and stored
back into the element's width. For float32 that is exactly the binary32 sum, because 53 >= 2 * 24 + 2 (the
double-rounding bound: rounding the exact sum to binary64 and then to binary32 equals rounding it to binary32 directly).
The weights are summed in binary64 in the same order. The rank then divides with the same expression as the `all_reduce`
path. So the comparison is bit equality, and the tolerance AGENTS.md §16 would derive is zero. The test also checks that
its data are order-sensitive (ascending and descending folds differ on some element), so equality pins the order rather
than passing by commutativity.

float16 and bfloat16 would need their own rounding after each addition and are refused by name (§4); a float64 fold
is Python's own arithmetic.

### 2.4 FedAvgM and FedProx

`FedAvgM` forms its mean through `FedAvg.aggregate`, so the mean is the server's; its velocity update
`v = beta * v + mean` stays on each rank, as on the `all_reduce` path, where every rank already holds the same mean and
applies the same deterministic update. Keeping the velocity on the server would make the server stateful across rounds
and would need its own binary32 multiply-add; it is not built, and nothing claims it is. `FedProx`'s difference is the
local proximal hook; its server step is FedAvg's, so it reaches the server the same way.

### 2.5 A participant that does not hand in

`on_missing='refuse'` is the only policy served. The server completes a round only when every rank of the world has
handed in. At the request's timeout (the group's, which `init_process_group` sets on the store) it answers every rank
that did hand in with the ranks that did not, records the round as abandoned in the ledger, and computes **no partial
aggregate**. Each rank raises `RankDropped` naming the missing ranks, and `FederatedDataParallel` undoes the round
(`delta.revert`), as it does for a lost peer on the `all_reduce` path. A rank that arrives after the verdict gets the
same verdict. A rank that submits twice, or tables that do not line up, refuse the whole round for every rank.
`ServerAggregationError` names a server that refused or could not be reached; the round is undone on it too.

## 3. Wire

Each federated request is one TCP connection to the listener that already serves the store. The header is the store
protocol's 4-byte length plus JSON, with metadata only (`op`, `key`, `rank`, `world`, `timeout`, `weight` as
`float.hex`, `names`, `frames`). The tensors follow as `wire.py` frames (`TNP2` header, raw little-endian payload). The
reply is a JSON header (`ok`, `ranks`, `names`, `total_weight` as `float.hex`, `sha256`, `frames`) followed by the
result's frames. No tensor value is ever JSON or base64. One request may carry at most 1 GiB of frames
(`_MAX_FOLD_BYTES`), and the listener's reader refuses past it before allocating. The server loads `wire.py` beside
itself when it runs as a script, so the standalone server still imports no torch.

## 4. Refused by name

| What | Where it refuses | Why |
|---|---|---|
| `aggregation=` anything but `'all_reduce'`, `'server'` | constructor, `ValueError` | |
| `aggregation='server'` on a group without a server (`tcp://`, `env://`, `file://`) | first `synchronize`, before any collective | the participants' `all_reduce` is a different topology and is not substituted |
| an algorithm other than `FedAvg`, `FedAvgM`, `FedProx` (subclasses included) | constructor | the server's fold is a weighted sum; another algorithm may mask, clip, add noise or take a median |
| `on_missing='average_arrived'` | constructor | the survivor set would have to be the server's verdict, carried with the aggregate; not built |
| `secure_aggregation=` | constructor | the server reads every weighted delta in the clear |
| `differential_privacy=` | constructor | no clipping, noise or accountant on the server's fold |
| a float16 or bfloat16 table | rank side before sending, and the server's fold | no per-addition rounding for those widths |
| a request past 1 GiB of frames | rank side, and the listener | nobody has run a round that large through it |

Secure aggregation and differential privacy from other branches meet the constructor's refusal when merged: any
algorithm type not in the three, and either flag, refuses with `aggregation='server'` named.

## 5. Tests written (not run)

`tests/distributed/test_fedserver.py`, registered in `.github/scripts/gate_suites.py` for both runners. Each test's
docstring names the sabotage it catches.

<!-- DOCWATCH: symbol-in-file tests/distributed/test_fedserver.py test_server_aggregation_equals_all_reduce_aggregation_bit_for_bit_through_the_in_process_server present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_fedserver.py test_server_aggregation_equals_all_reduce_aggregation_bit_for_bit_through_a_standalone_server_process present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_fedserver.py test_the_server_really_aggregated_every_server_round_by_its_own_ledger present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_fedserver.py test_a_participant_that_never_hands_in_is_named_and_the_round_is_undone present -->

| Test | Asserts |
|---|---|
| `test_the_server_folds_in_ascending_rank_order_and_every_rank_gets_the_same_bits` | no torch: three threads fold `[1, 0.5]`, `[2^-24, 0.25]`, `[2^-24, 0.125]`; ascending gives exactly `[1.0, 0.875]` (worked by hand; descending would give `1 + 2^-23`); total 15 |
| `test_the_server_gathers_every_ranks_value_in_rank_order` | no torch: `[5, 6, 7]` to every rank |
| `test_the_server_refuses_by_name_and_folds_nothing_for_a_bad_round` | no torch: a missing rank (named, after the timeout), a rank twice, different tables, float16; the ledger holds exactly the one good fold and one abandoned round |
| `test_server_aggregation_equals_all_reduce_aggregation_bit_for_bit_through_the_in_process_server` | world 3, `tnserver://` hosted by rank 0: FedAvg, FedAvgM, FedProx, three rounds each, from the same weights on the same data; the deltas handed in are equal, and the aggregate, the installed parameters and the total weight are equal bit for bit, on every rank |
| `test_server_aggregation_equals_all_reduce_aggregation_bit_for_bit_through_a_standalone_server_process` | the same through a `StandaloneServer` the test started |
| `test_the_data_pin_the_fold_order_and_the_server_matches_a_central_ascending_fold` | the data are order-sensitive; FedAvg's server aggregate equals the ascending fold divided by the total, computed in plain Python binary32 from what the ranks reported |
| `test_the_server_really_aggregated_every_server_round_by_its_own_ledger` | every receipt is a ledger fold over ranks `[0, 1, 2]` with matching SHA-256, weights and total; the delta fold's SHA-256 equals the hash of the central ascending fold; 18 folds for 9 server rounds and none unaccounted |
| `test_a_server_round_sends_nothing_over_the_participants_collectives` | `dist.all_reduce` and `dist.all_gather` called 0 times in a server round, more than 0 in an `all_reduce` round |
| `test_a_participant_that_never_hands_in_is_named_and_the_round_is_undone` | rank 2 never calls `synchronize`: ranks 0 and 1 raise `RankDropped` naming rank 2 within 0.5 to 4 times the 10 s timeout, their parameters are back at the base byte for byte, and the ledger has no fold and one abandoned round over `[0, 1]` |
| `test_server_aggregation_refuses_what_it_does_not_serve_by_name` | the constructor refusals of §4 and float16 |
| `test_server_aggregation_on_a_group_without_a_server_refuses_by_name` | world 2 from `tcp://`: refused naming `tnserver://` |
| `test_every_rank_really_ran_the_shim_and_the_standalone_server_was_another_process` | shim on every rank; the server's pid is none of theirs |

### 5.1 Nullifications planned (not performed: nothing may run in this round)

| Mutant | Test that must go red |
|---|---|
| `_fold_compute` skips the last rank's table | both bit-equality tests (aggregate differs from round 1), the ledger test (`ranks` and SHA-256) |
| `_fold_compute` folds in descending order | both bit-equality tests and the central-fold test (the data are checked to be order-sensitive), the unit fold test (element 0 becomes `1 + 2^-23`) |
| `_fold_compute` accumulates in binary64 and rounds once at the end | the unit fold test (element 0) and, on these data, the bit-equality tests |
| `FedAvg.aggregate` ignores the route (computes on the ranks with `all_reduce`) | the collectives-count test (non-zero) and the ledger test (no delta folds) |
| `agree` ignores the route | the collectives-count test (`all_gather` non-zero) |
| `_FedCore.join` folds over whoever arrived at the timeout | the missing-participant test (a fold over `[0, 1]` in the ledger; no `RankDropped`) and the unit refusal test |
| `_refuse_for_server` removed | the refusal test (subclass and custom algorithm accepted) |
| `_ServerRoute` falls back to `all_reduce` without a server | the no-server test |

A nullification that is not caught is a finding; CI is where these are first tried.

## 6. Known gaps

1. Graded *builds*: not run anywhere. The bit-equality argument of §2.3 assumes the shim's float32 `add_` and `/` on
   CPU are IEEE-correct per element; if a run shows otherwise, the comparison would need a derived tolerance, and that
   would be a finding, not a widened bound.
2. The fold is pure Python: one addition per element per rank, run under the federated lock. Fine for the test's
   models; not measured on a real model, and the 1 GiB request limit is a refusal, not a measurement.
3. `on_missing='average_arrived'`, a server-held FedAvgM velocity, float16 and bfloat16, secure aggregation and
   differential privacy are not built with server aggregation (§4).
4. The participants still form the star and the mesh at `init_process_group`; a server round does not use them, but
   a participant that cannot reach its peers cannot join a group at all. Participants that reach only the server need
   a backend that does not build the mesh, which is not built.
5. The federated operations are not authenticated: the pre-shared key (SERVER.md §4) covers mesh connections only,
   and a party that reaches the server's port can submit to a round whose key it knows or guesses. The key is not
   secret (it contains the group token from the store). A rank that submits twice refuses the round; a stranger who
   submits first as some rank would be folded in. Authenticating federated requests is not built.
6. The cohort path (`select=`) with `aggregation='server'` is written (its digest goes through `fed_gather`) and is
   not tested by this suite.
