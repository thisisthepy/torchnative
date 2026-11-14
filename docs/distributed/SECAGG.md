# SECAGG: secure aggregation by pairwise masking in `torch.federated`

Issue #253 (part of #20). Written 2026-11-13. Stacked on PR #230 (binary wire, transports, point-to-point on a
full mesh) and PR #218 (backend `torchnative`).

**Grade: builds.** Every file below passed `python -m py_compile` and nothing else. No suite, gate or probe ran
in the round that wrote it (AGENTS.md rule 10: agents do not run suites; the gate runs in CI). Nothing here is
*reaches* or *agrees* until CI runs `tests/distributed/test_secagg.py` (AGENTS.md §16). Read every "does" below
as "is written to".

Provenance. The algorithm is the leader's decision, 2026-11-13: "secure aggregation via pairwise masking
(Bonawitz-style) is approved as the algorithm". No cryptographic dependency was approved, so the code uses the
standard library only (`secrets`, `hashlib`, `hmac`, `struct`). Everything else in this document is an **agent
proposal, not confirmed**: the wrapper shape and its name, the fixed-point scale and modulus, the PRG and its
domain string, the seed tag, the wire shape, the logging keys, and the choice to mask the total weight.

## 1. Why it is needed here

Collectives in this tree still run over `ProcessGroupLocal`'s star: every leaf sends its payload to rank 0, which
folds them in rank order and sends one result back (FEDERATED4.md §1). So the hub reads every rank's contribution
in the clear. Plain `FedAvg` therefore hands rank 0 every client's weighted delta and every client's sample count.
Masking makes the hub's view a set of uniformly distributed residues whose sum is the only thing that decodes.

## 2. What was built

| Piece | Where | What it does |
|---|---|---|
| Fixed point and PRG | `torchnative/python/torchnative/federated/_secagg.py` | `modulus_bits`, `per_rank_bound`, `quantise`, `mask_words` (HMAC-SHA-256 counter mode), `net_mask`, `centred`. Pure Python ints, no torch |
| The wrapper | `torchnative/python/torchnative/federated/algorithms.py` `SecAgg` | `SecAgg(FedAvg(), fraction_bits=32)`: the round in §3 |
| Wiring | `torchnative/python/torchnative/federated/__init__.py` | `FederatedDataParallel` reads the masked total weight from `SecAgg.last_total_weight` instead of its own unmasked weight collective, merges `SecAgg.logging_data()` into `_get_federated_logging_data()`, refuses `on_missing='average_arrived'` with a secure algorithm, and `secure_aggregation=` refuses and points at `SecAgg` |
| Tests | `tests/distributed/test_secagg.py` | §6 |

<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/federated/algorithms.py SecAgg present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/federated/_secagg.py mask_words present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/federated/_secagg.py modulus_bits present -->

**Why a wrapper, not a flag.** Masking replaces exactly one thing `FedAvg` does: how `sum(w_k d_k)` and
`sum(w_k)` are formed. An algorithm object is where this package keeps that, `Delta.publish(aggregator=...)`
reaches it on the low-level road (a flag on `FederatedDataParallel` would not), and it composes the way `FedAvgM`
composes `FedAvg`. Two spellings of one mechanism would be one too many, so the flag stays a refusal.

```python
from torchnative import federated
from torchnative.federated import algorithms

fdp = federated.FederatedDataParallel(model, algorithm=algorithms.SecAgg(algorithms.FedAvg()))
... local training ...
fdp.synchronize(weight=n_local_samples)
```

## 3. One round

1. `agree` on the parameter schema (as FedAvg does) and on `(round number, fraction_bits)`, because a pair that
   disagreed about either would produce masks that do not cancel: a wrong aggregate nothing downstream notices.
2. Each rank quantises every element: `q = round(fl64(w * d) * 2**f)`, and its weight: `q_w = round(w * 2**f)`.
3. One int64 `all_reduce` of a 0/1 flag: if any rank holds a value past the bound (§4) or a nan/inf, every rank
   refuses by name. This flag is the one thing that travels in the clear: the hub learns how many ranks
   overflowed.
4. Seeds. For every pair `i < j`, rank `i` draws 32 bytes with `secrets.token_bytes` and `dist.isend`s them to `j`
   on tag `0x5EC0A6` over the default group's direct mesh link. A seed passes through rank 0 only when `i` or `j`
   is 0. Seeds are fresh every round, local to the call, cleared after the masks are drawn, and never logged.
5. Masks. Rank `r` adds `PRG(s_rj)` for every `j > r` and subtracts `PRG(s_ir)` for every `i < r`, mod `2**k`.
   `PRG` is HMAC-SHA-256 keyed by the seed, in counter mode over
   `DOMAIN || 0x00 || round (u64) || len(label) (u32) || label || chunk (u64)`, four 64-bit words per block, each
   reduced to its low `k` bits (uniform because the modulus is a power of two). Labels are `b"p:" + name` per
   parameter and `b"w"` for the weight, so they cannot collide.
6. The masked residues go through the existing `all_reduce(SUM)` as int64, one collective per label. Every
   pair's block appears once with each sign, so `sum(masked) mod 2**k == sum(q) mod 2**k` exactly.
7. Decode once: `S = centred(sum)`, `S_w = centred(sum_w)`, and the mean is `S / S_w` as one correctly rounded
   Python int division in float64 (the `2**f` cancels), then cast to the table's dtype.

**The total weight is masked too.** It is the local sample count, which a client may not want the hub to learn,
and masking it costs one element. It is exact whenever every weight is an integer (any multiple of `2**-f`).

## 4. The bound, and the quantisation error

**Overflow.** The hub folds `world` residues, each in `[0, 2**k)`, in an int64 tensor. Their sum is below
`world * 2**k`, which is at most `2**63 - 1` whenever `world < 2**(63 - k)`. So

    k = 63 - world.bit_length()

is the largest modulus the accumulator cannot overflow. The decoded sum is the centred residue of `sum(q) mod
2**k`, which equals `sum(q)` exactly when `|sum(q)| <= 2**(k-1) - 1`; that holds for any inputs when every rank
keeps `|q| <= B = (2**(k-1) - 1) // world`. So the per-rank bound is

    |w * d| <= B / 2**f

and a value past it is refused by name on every rank, never wrapped. At world 3 and `f = 32`: `k = 61`,
`B = (2**60 - 1) // 3`, and the bound is about 8.9e7. `SecAgg` refuses a world so large that `B < 2**f`.

**Quantisation error.** Against the exact mean `m = sum(w_k d_k) / T` with `T` the integer total weight:

| Step | Error per element |
|---|---|
| `q = round(fl64(w d) 2**f)` summed over the world | `Q = (world 2**-(f+1) + 2**-53 sum|w_k d_k|) / T` |
| `fl64(S / S_w)` | `2**-53 (|m| + Q)` |
| cast to float32 | `2**-24 (|m| + Q)(1 + 2**-53)` |
| install `fl32(base + a)` | `2**-24 (|base| + |m| + B_a)` |

`B_a` is the sum of the first three rows, `B_inst = B_a +` the fourth. The test's tolerance is exactly these terms,
evaluated in `fractions.Fraction` against an exact reference, valid for normal float32 results.

## 5. Threat model

- **Protected:** an honest-but-curious aggregator. Rank 0, the hub, follows the protocol and reads everything the
  star carries; it sees masked residues, their sum, and one overflow count. It does not see any individual
  update or weight. The sum is public to every rank, as in FedAvg, so at world 3 the hub learns the sum of the
  other two ranks' updates.
- **Not protected: the links.** Anyone who can read a pair's link learns that pair's seed. The links are not
  encrypted. The opt-in pre-shared key of D10 (PR #238) authenticates peers; it does not hide traffic.
- **Not protected: collusion.** Ranks that pool their seeds can unmask others; `world - 1` colluders learn the
  last rank's update from the sum.
- **Not protected: a malicious aggregator.** A hub that alters results, drops or reorders contributions, or lies
  about who is present is not defended against.
- `agree()`'s 56-bit digest detects accidents, not adversaries (unchanged from FEDERATED4.md §7).

## 6. Tests (`tests/distributed/test_secagg.py`), and what makes each fail

All run at world 3 in real processes through `test_shim._fed4_spawn`. The children instrument
`ProcessGroupLocal._recv_json` on rank 0 (what the hub reads) and `ProcessGroupTorchnative.send`/`recv` (which link
each seed crossed) before `init_process_group`.

| Test | Asserts | Fails if |
|---|---|---|
| `test_secagg_world_three_installs_fedavg_within_the_derived_quantisation_bound` | installed model and aggregate within §4's bound of the exact `(3 d0 + 7 d1 + 2 d2)/12`; all ranks bit for bit equal; the unweighted mean falls outside the bound somewhere (control); logging has `secure_aggregation`, `fraction_bits`, `scale`, `modulus_bits`, `per_rank_bound` as derived | masks do not cancel, scale wrong, a weight ignored, a rank lost, ranks decode differently |
| `test_secagg_hub_never_reads_a_plain_update` | each leaf's frame at the hub equals what it sent and differs element by element from its plain residues; every rank's own contribution differs too; no float reaches the hub in a secure round | masking skipped. Sabotage in the same children: `Unmasked` (net mask zero) makes the hub's frames equal the plain ones, and plain FedAvg puts floats on the hub, so both assertions are shown able to fail |
| `test_secagg_masks_cancel_exactly_in_the_integer_sum` | `sum(masked) == sum(plain) mod 2**61` per element, equal to every rank's all_reduce result; no pair of ranks cancels alone | a pair's ends disagree on seed, round, label, chunk or sign; the hub's int64 fold wraps or rounds through float |
| `test_secagg_seeds_cross_only_the_direct_pair_links` | rank `r` sends one 32-byte frame to each `j > r` and receives one from each `i < r`; rank 0 receives none; pair (1,2)'s seed is in no hub frame (bytes or int64 words), fresh per round, not in logging or `SecAgg` state | seeds relayed through a collective, reused, or kept |
| `test_secagg_with_a_proper_subset_cohort_masks_the_abstention` | cohort `[0, 1]`: the mean over 0 and 1 within the bound; rank 2's plain is zero, its masked is not; total weight 10 | an abstention weighted or left unmasked |
| `test_secagg_refuses_by_name_what_it_cannot_encode_or_combine` | overflow and nan refuse on every rank quoting the derived bound; `tolerate_missing`, `average_arrived`, the flag, `SecAgg(FedAvgM)`, `SecAgg(FedProx)`, `SecAgg(SecAgg)`, `fraction_bits=0` refuse; the group survives | a rank wraps, only the offender refuses, a combination is accepted |
| `test_secagg_a_rank_lost_after_the_seeds_aborts_and_undoes_the_round` | rank 2 exits between seeds and the masked sum; ranks 0 and 1 raise `RankDropped` naming SecAgg and "recovery is not built"; models back at the base | survivors decode noise, a bare socket error, training left in place |

What these cannot find: a leak through a side channel the star does not carry (timing, message sizes), a seed
relayed through the hub in a transformed encoding, and anything an off-machine reader of the links learns.

## 7. Not built, by name

- **Dropout recovery.** Bonawitz's Shamir shares of each seed and the unmasking round are not built. A rank
  missing after the seeds aborts the round with `RankDropped`; `FederatedDataParallel(on_missing='refuse')` then
  undoes it. `on_missing='average_arrived'` with `SecAgg` is refused at construction, and `SecAgg` refuses inside
  `tolerate_missing()`. A peer that stays connected but silent during the seed exchange waits for the group's
  point-to-point timeout (30 minutes by default) before failing.
- **Key agreement and encrypted links.** Seeds are sent, not agreed by Diffie-Hellman, and the links are plain.
- **Combinations.** Only `FedAvg` is wrapped. `FedAvgM`, `FedProx` and `SecAgg` itself refuse by name. DPFedAvg
  (PR #228) is not in this tree; combining it means adding the noise before quantisation, and it is not built.
- **Performance.** Quantisation, masking and decoding are Python loops over elements. Nothing was measured.

## 8. Expected merge conflict

PR #228 rewords the `secure_aggregation=` refusal in `torchnative/python/torchnative/federated/__init__.py` and the
module docstring's table row beside it, and adds `algorithms.DPFedAvg`. This change rewrites the same refusal (it
now points at `SecAgg`) and the same table row, and updates the two `test_shim.py` assertions on that refusal's
text. The merge keeps this text for `secure_aggregation=` and #228's for `differential_privacy=`.
