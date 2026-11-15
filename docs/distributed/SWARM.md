# SWARM: `torch.swarm.SwarmDataParallel`, gossip averaging over point-to-point without a server

Issue #291. SPEC S6.9. Written 2026-11-13.

**Grade: builds.** Every file below passed `python3 -m py_compile` and nothing else. No suite, gate or probe
ran in the round that wrote it (AGENTS.md rule 10: agents do not run suites; the gate runs in CI). Nothing here
is *reaches* or *agrees* until CI runs `tests/distributed/test_swarm.py` (AGENTS.md §16). Read every "does"
below as "is written to".

## 0. Provenance

| Statement | Source |
|---|---|
| `torch.swarm` is a paradigm package for peer-to-peer learning without a server, beside `torch.federated`, with the wrapper `SwarmDataParallel`, "the same structure as `torch.federated`" | owner, 2026-11-08 (docs/INTENT.md §4: "연합학습과 군집학습 부분은 확정하고 문서 개정하고, 리드미 수정하고, 지금 리펙토링은 바로 들어가도록 해.") |
| `torch.federated` and `torch.swarm` use the same transport as `torch.distributed` | owner, 2026-11-08 (docs/INTENT.md §5, the goal statement) |
| Everything else in this document | **agent proposal, not confirmed**: the synchronous gossip round, the ring default, Metropolis-Hastings weights, the agreement check, the tags, `torch.swarm.algorithms` and `GossipAvg`, the argument names, `NeighbourDropped`, the refusals |

The owner named the package and the wrapper, and said "without a server". The topology and the mixing rule were
not discussed; they are open for the owner to change.

## 1. What was built

| Piece | Where | What it does |
|---|---|---|
| Wrapper | `torchnative/python/torchnative/swarm/__init__.py` `SwarmDataParallel` | a `torch.nn.Module` wrapper used like `FederatedDataParallel`: the caller trains locally with any optimizer, then calls `synchronize()` once per round |
| Topology | same file, `resolve_topology`, `TOPOLOGIES` | `"ring"` (default), `"complete"`, or one neighbour list per rank (a list of lists or a `{rank: [ranks]}` mapping), validated |
| Algorithm | `torchnative/python/torchnative/swarm/algorithms.py` `GossipAvg`, `metropolis_hastings` | the mixing matrix (Metropolis-Hastings by default, or an explicit matrix that is validated) and the fold |
| Failure | `NeighbourDropped` | a neighbour that does not deliver; the round is not applied |
| Add-hook | `torchnative/python/torch/swarm/__init__.py`, `.../torch/swarm/algorithms.py` | `torch.swarm` and `torch.swarm.algorithms`, one `import *` line each, as `torch.federated` |
| Package list | `torchnative/python/torchnative/__init__.py` | `swarm` added to `__all__` (lazy subpackage) |

<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/swarm/__init__.py SwarmDataParallel present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/swarm/__init__.py resolve_topology present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/swarm/__init__.py NeighbourDropped present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/swarm/algorithms.py GossipAvg present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/swarm/algorithms.py metropolis_hastings present -->

```python
from torch.swarm import SwarmDataParallel
from torch.swarm.algorithms import GossipAvg

sdp = SwarmDataParallel(model, algorithm=GossipAvg(), topology="ring")
optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
for batch in local_batches:
    optimizer.zero_grad()
    loss_fn(sdp(batch)).backward()
    optimizer.step()
sdp.synchronize()
info = sdp._get_swarm_logging_data()
```

## 2. One round

Rank `i` holds parameters `x_i` (the swarm parameters flattened to one vector). One `synchronize()`:

1. **Agreement check, point to point.** `i` sends each neighbour three `int64`s: a digest of the neighbour lists
   and the mixing matrix (each weight as `float.hex`), a digest of the parameter schema (names, shapes, dtype),
   and the round number. A neighbour whose three differ is refused by name (`ValueError`), before any parameter
   is sent. On a connected graph, every adjacent pair agreeing implies every pair agrees, so the check is exact
   and needs no collective.
2. **Parameters, point to point.** `i` sends `x_i` to each neighbour with `torch.distributed.isend` and receives
   each neighbour's with `irecv`, then waits for all of them. Nothing goes to a rank that is not a neighbour.
3. **Mix.** `x_i <- sum_j W[i][j] * x_j` over `{i}` and its neighbours, with each term computed as
   `float64(x_j) * W[i][j]`, the terms added **in ascending rank order** in float64, and the sum rounded once to
   the parameters' dtype. The fixed order is what lets the same round computed elsewhere be equal bit for bit.
4. Install, record the diagnostics, re-snapshot (for `local_norm` next round), and increment the round.

There is no `all_reduce`, no hub and no server. After a round the ranks still differ: each holds its own mix.
Repeated rounds approach the global mean, with the disagreement contracting by `lambda_2`, the largest modulus
among the mixing matrix's eigenvalues other than the one at 1.

Tags: `0x535701` for the check and `0x535702` for the parameters. User traffic on the same tags in the same group
would interleave with them.

### 2.1 Why these choices (agent proposals)

* **Synchronous rounds.** The user loop then has the shape of `FederatedDataParallel`'s (local training, then one
  call per round), and a round is a deterministic function of what each rank held, which a test can recompute.
  Asynchronous gossip changes both and is not offered.
* **Parameters, not deltas.** `FederatedDataParallel` mixes deltas from a common base because every rank installs
  the same aggregate. After a gossip round the ranks hold different weights, so there is no common base, and
  mixing offsets from different bases is not the same operation as mixing parameters.
* **Ring as the default.** Degree two whatever the world size: each rank sends two parameter messages a round. At
  world 2 the ring is one edge and one round is the exact mean; at world 3 it is the complete graph.
* **Metropolis-Hastings weights.** `W[i][j] = 1 / (1 + max(d_i, d_j))` on an edge and
  `W[i][i] = 1 - fsum(row i off the diagonal)`. Each rank needs only its own degree and its neighbours', the
  matrix is symmetric by construction (so doubly stochastic, so the global mean is preserved), and the self
  weight is at least `1 / (1 + d_i) > 0` (so on a connected graph the chain is aperiodic).
* **`torch.swarm.algorithms` exists, small.** It mirrors `torch.federated.algorithms` and separates the two things
  that vary independently: the topology and exchange (the wrapper) and the mixing rule (the algorithm). The known
  next rules (push-sum for directed graphs and weighted means, compressed gossip, gradient tracking) change the
  rule and keep the exchange. Unlike `FedAvg.aggregate`, `GossipAvg.aggregate` is not a collective: it is a pure
  function of what arrived.
* **No refusal of a round without training.** `FederatedDataParallel` refuses a zero delta because it would count
  at full weight in the one aggregate every rank installs. In gossip a rank's contribution is its parameters, so
  rounds with no training between them are consensus averaging, which is what the convergence test runs.

## 3. What is refused by name

| Case | Where | Refusal |
|---|---|---|
| no process group | `synchronize`, `neighbours`, `mixing_matrix` | `RuntimeError`, naming `init_process_group` |
| a world of one | same | `NotImplementedError`: the round is the identity, which no test can tell from a broken mix |
| unknown topology name, wrong number of lists, a self loop, a rank out of range, a duplicate | `resolve_topology` | `ValueError` naming the rank |
| a directed (asymmetric) neighbour list | same | `ValueError`: needs push-sum, not built |
| a disconnected topology | same | `ValueError` naming the components: each would approach its own mean |
| a mixing matrix of the wrong size, negative, asymmetric, a row not summing to one, a zero self weight, a weight on a non-edge, a zero weight on an edge, an unknown rule | `GossipAvg` | `ValueError` naming the entry |
| `weight=` | `GossipAvg.resolve_weight` | `TypeError`: a weighted mean without a server needs push-sum |
| mixed dtypes, integer parameters, parameters off the CPU, no parameters, an unknown parameter name | `SwarmDataParallel` | named refusals |
| `secure_aggregation=`, `differential_privacy=` | `SwarmDataParallel()` | `NotImplementedError`, as in `torch.federated` |
| a neighbour that does not deliver (connection closed, timeout) | `synchronize` | `NeighbourDropped` naming the neighbour; the parameters are left as they were |
| neighbours that disagree about the topology, the matrix, the schema or the round | `synchronize` | `ValueError` naming both ranks and both digests; nothing mixed |

The row-sum tolerance is derived, not chosen: each entry is a typed decimal stored as the nearest double, off by
at most `2**-53` of a number at most 1, and `math.fsum` rounds once, so a row of `n` entries is within
`n * 2**-53` of what was meant.

A zero self weight is refused instead of checking the spectrum: on a bipartite graph it makes the mix periodic
(two ranks swapping forever, eigenvalue -1). A positive diagonal on a connected graph rules that out without an
eigen solver in the library.

## 4. Tests written (not run)

`tests/distributed/test_swarm.py`, registered in `.github/scripts/gate_suites.py` for both runners. Process
worlds use `test_collect2.py`'s `_c2_spawn`. The shim children run at world 2 and 4, upstream torch with gloo at
world 2 and 4, and one shim child at world 1.

<!-- DOCWATCH: symbol-in-file tests/distributed/test_swarm.py test_one_gossip_round_equals_the_same_round_computed_centrally_bit_for_bit present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_swarm.py test_repeated_rounds_approach_the_global_mean_at_the_rate_of_the_second_eigenvalue present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_swarm.py test_it_was_peer_to_peer_no_collective_and_messages_only_to_neighbours present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_swarm.py test_swarm_agrees_with_upstream_gloo_running_the_same_gossip_by_hand present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_swarm.py test_the_central_oracle_can_tell_a_dropped_neighbour_a_wrong_weight_and_an_all_reduce_apart present -->

| Test | Asserts |
|---|---|
| `test_one_gossip_round_equals_the_same_round_computed_centrally_bit_for_bit` | ring at world 2 and 4: each rank's result equals the fold in pure Python floats (no torch), rounded with `struct`; the matrix is this file's own Metropolis-Hastings |
| `test_at_world_two_one_round_is_the_exact_mean_on_both_ranks` | one edge, weight 1/2: both ranks hold `f32(a/2 + b/2)` |
| `test_a_round_after_real_local_training_equals_the_central_mix_of_what_each_rank_trained` | forward, backward, SGD step, `synchronize()`: the mix of the trained weights; `local_norm` within a derived bound |
| `test_the_mixing_matrix_is_metropolis_hastings_computed_independently` | `mixing_matrix()` and `metropolis_hastings(path)` against the test's own formula |
| `test_the_logging_data_reports_the_round` | the diagnostics dict |
| `test_repeated_rounds_approach_the_global_mean_at_the_rate_of_the_second_eigenvalue` | 8 rounds, no training: `d_{k+1} <= lambda_2 d_k + eps_k`, the global mean held within the same slack, `lambda_2` from a Jacobi eigen solver checked against the closed forms below; on ring-4 also the lower bound |
| `test_it_was_peer_to_peer_no_collective_and_messages_only_to_neighbours` | per round: zero calls of `ProcessGroupLocal._star_exchange` (the road every collective of the backend takes) and of `torch.distributed`'s collectives; exactly one check and one parameter message to and from each neighbour, none to anyone else |
| `test_swarm_agrees_with_upstream_gloo_running_the_same_gossip_by_hand` | `SwarmDataParallel` on the shim equals the same round written with `isend`/`irecv` on upstream gloo and on the shim, for the ring round, 8 ring rounds and 8 path rounds |
| `test_the_central_oracle_can_tell_a_dropped_neighbour_a_wrong_weight_and_an_all_reduce_apart` | the oracle under each of the three mutants of §6 predicts values the observed round does not have |
| `test_a_world_of_one_and_a_missing_process_group_refuse_by_name` | §3 |
| `test_topologies_that_cannot_converge_or_cannot_be_exchanged_refuse_by_name` | §3; the disconnected case also through `synchronize` at world 4, with zero messages sent |
| `test_mixing_matrices_that_are_not_doubly_stochastic_refuse_by_name` | §3 |
| `test_weight_secure_aggregation_privacy_and_bad_parameters_refuse_by_name` | §3; `weight=` also at world 4, with zero messages sent |
| `test_neighbours_that_disagree_about_the_topology_refuse_before_mixing` | ranks 0 and 1 with Metropolis-Hastings, 2 and 3 with another valid matrix on the same ring: all four refuse, parameters unchanged |
| `test_a_neighbour_that_vanishes_is_named_and_the_round_is_not_applied` | world 2, rank 1 exits: rank 0 raises `NeighbourDropped` with `missing == (1,)`, parameters and diagnostics unchanged |
| `test_torch_swarm_is_the_add_hook_for_torchnative_swarm` | `torch.swarm.SwarmDataParallel is torchnative.swarm.SwarmDataParallel`, from the tracked add-hook file |
| `test_every_rank_really_ran_the_shim_and_the_oracle_really_ran_upstream` | backend names per rank |

`tests/release/test_cipub.py::test_the_pinned_add_hook_count_is_the_number_of_tracked_add_hook_modules` is new as
well (§5).

### 4.1 The contraction, stated

The expected second-largest eigenvalue moduli of the Metropolis-Hastings matrices the test runs, in closed form:

| Topology | `W` | `lambda_2` |
|---|---|---|
| ring, world 2 | `[[1/2, 1/2], [1/2, 1/2]]` | 0: one round is the mean |
| ring, world 4 | self 1/3, edges 1/3 | 1/3, and every non-trivial eigenvalue has modulus 1/3 (1/3, 1/3, -1/3), so the ratio `d_{k+1}/d_k` is 1/3 exactly up to rounding |
| path 0-1-2-3, world 4 | `(I - L/3)` with `L` the path Laplacian | `(1 + sqrt 2)/3 = 0.8047`; the others 1/3 and `(1 - sqrt 2)/3 = -0.138` |

The floating-point slack `eps_k` is derived: a round rounds each element once to float32 (relative `2**-24`) after
a float64 fold of at most `deg + 1` terms (`2**-53` each), so it adds at most
`(2**-24 + (deg + 1) * 2**-53) * ||X_{k+1}||_F` to the disagreement. These numbers are a statement of what the test
asserts; none has been measured.

## 5. The add-hook count

The stamp `add_hooks=N` counts the `.py` files of the add-hook packages. With `torch.swarm` it is 4. Changed:

| File | Change |
|---|---|
| `scripts/vendor/vendor_torch.sh` | `--exclude '/swarm/'` in the rsync (otherwise `--delete` removes the tracked add-hook), and the stamp counts `torch/federated` and `torch/swarm` |
| `.github/workflows/test.yml` | `add_hooks=2` to `add_hooks=4` |
| `.github/workflows/publish-pypi.yml` | `add_hooks=2` to `add_hooks=4` |
| `.gitignore` | negations for `torchnative/python/torch/swarm/__init__.py` and `algorithms.py` |
| `tests/release/test_cipub.py` | a new test reading all of the above against `git ls-files torchnative/python/torch` |
| `docs/platform/PUBLISH_CI.md` | the test count (38 to 39) and a note; the 2026 measurement `add_hooks=2` stays as recorded |

The two add-hook files have to be added to the index (`git add`) before CI, because the stamp test counts
`git ls-files`, and the workflows count what the checkout holds.

## 6. Nullifications planned (not performed)

AGENTS.md §17.5. None of these was run (rule 10). Each names the test that should go red.

| Mutation in `torchnative/swarm/` | Test that should go red |
|---|---|
| `GossipAvg.aggregate` skips the highest neighbour's term | `test_one_gossip_round_equals_the_same_round_computed_centrally_bit_for_bit`, `test_swarm_agrees_with_upstream_gloo_running_the_same_gossip_by_hand`, and the mean-drift assertion of the convergence test (the row no longer sums to one) |
| `metropolis_hastings` uses `1 / (1 + d_i)` instead of `1 / (1 + max(d_i, d_j))` | ring rounds stay green (the ring is regular); `test_the_mixing_matrix_is_metropolis_hastings_computed_independently` (path), the path half of the gloo test, and the mean drift on the path (the matrix is no longer symmetric) |
| `synchronize` replaced by an `all_reduce` mean (a host twin of the right shape) | `test_it_was_peer_to_peer_no_collective_and_messages_only_to_neighbours` (`star > 0`) and the central and gloo equality at world 4 (gossip is not the global mean) |
| the agreement check skipped | `test_neighbours_that_disagree_about_the_topology_refuse_before_mixing` |
| `_is_gone` returns False | `test_a_neighbour_that_vanishes_is_named_and_the_round_is_not_applied` (the raw transport error, not `NeighbourDropped`) |

`test_the_central_oracle_can_tell_a_dropped_neighbour_a_wrong_weight_and_an_all_reduce_apart` is the part of the
first three that can be shown without mutating the module: it applies each mutant to the inputs the ranks reported
and asserts the prediction differs from what was observed. **What it cannot show** is that the module, mutated,
would actually produce the mutant's numbers; only the mutation runs above show that.

## 7. Known gaps and risks, read off source, not measured

1. **The shim's float64 scalar multiply.** Bit equality assumes `float64_tensor * python_float` is one IEEE
   double multiply on the shim, as it is upstream. If the shim routes it through float32 or a fused multiply-add,
   the central and gloo comparisons go red, which is the intended outcome, and the fix is in the kernel.
2. **A neighbour that stalls without closing** is reported after the group timeout
   (`init_process_group(timeout=...)`), not at once.
3. **After a `NeighbourDropped` or a disagreement**, sends already queued to other neighbours stay in their
   queues; a later round on the same group starts misaligned. The agreement check (the round number) then refuses
   rather than mixing, but recovery (re-forming the group) is not built.
4. **Not built:** weighted means (push-sum), directed or time-varying topologies, asynchronous gossip, compression,
   device parameters, mixed dtypes, `new_group` sub-swarms (point-to-point on a sub-group is not tested).
5. Each rank's parameters reach its neighbours in the clear. That is inherent to gossip and is why
   `secure_aggregation=` refuses.
