# FEDERATED5: client-level differential privacy (`DPFedAvg`)

Issue #20 (SPEC S6.7) named three things `torch.federated` did not do: secure
aggregation, differential privacy and participant selection. This round builds
the differential-privacy part as client-level DP in the DP-FedAvg shape
(McMahan, Ramage, Talwar and Zhang, 2018). **Every design choice below is an
agent proposal, not confirmed by the owner.** The owner's direction this rests
on is the package itself (owner, 2026-11-08, docs/INTENT.md §4); the owner
has not ruled on DP, its shape or its threat model.

**Grade: builds.** The code compiles (`python -m py_compile`) and the tests
are written; nothing was run in this round (AGENTS.md §10: tests run in CI).
No number in this document was measured, and no test was observed red or
green.

| | |
|---|---|
| per-client L2 clip of the whole delta | **builds**: `algorithms.DPFedAvg(clip_norm=C, ...)` |
| Gaussian noise N(0, (sigma*C)^2) added once to the sum, same on every rank | **builds**: `noise_multiplier=sigma`, `seed=` |
| diagnostics in `_get_federated_logging_data()` | **builds**: clip_norm, noise_multiplier, noise_std, clipped, pre_clip_norm, noise_seed, dp_round, threat_model, accountant |
| privacy accounting (epsilon) | **not built**: no accountant |
| sample-count weights under DP | **refused by name** |
| `threat_model='local'` (protection against peers) | **refused by name** |
| example-level DP (per-example clipping) | **not built** |
| secure aggregation | **refused by name**, planned route: issue #219 |
| participant selection without attendance (`new_group`) | **pending**, issue #221 |

---

## 1. The shape: an algorithm, not a flag (agent proposal)

```python
from torchnative import federated
from torchnative.federated import algorithms

fdp = federated.FederatedDataParallel(
    model, algorithm=algorithms.DPFedAvg(clip_norm=1.0, noise_multiplier=1.1))
# ... the caller's own local loop ...
fdp.synchronize()                       # no weight=: see section 3
info = fdp._get_federated_logging_data()
```

`DPFedAvg` sits beside `FedAvg`, `FedAvgM` and `FedProx` rather than behind a
`FederatedDataParallel(differential_privacy=...)` keyword, for three reasons:

1. The clip and the noise sit *inside* the aggregation: the noise goes on the
   sum, after the `all_reduce` and before the division. That is the
   algorithm's code.
2. The weight contract is the algorithm's `resolve_weight`, which
   `FederatedDataParallel` already asks.
3. As an algorithm it also serves the low-level road,
   `Delta.publish(aggregator=DPFedAvg(...))`.

<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/federated/algorithms.py DPFedAvg present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/federated/algorithms.py _dp_gaussians present -->

`FederatedDataParallel(differential_privacy=...)` keeps refusing, now by
pointing at `algorithms.DPFedAvg`. It also says why an `(epsilon, delta)`
target is not accepted: there is no accountant.

## 2. The arithmetic

For each rank *k* with delta `d_k` over every federated parameter:

```
norm_k     = sqrt(fsum(x*x for every element x of d_k))      (float64)
clip(d_k)  = d_k * float32(C / norm_k)   if norm_k > C,  else d_k unchanged
aggregate  = (sum_k clip(d_k) + noise) / n
noise      = float32(sigma*C * z),  z ~ N(0, 1) per element, from the agreed seed
```

`n` is the number of participating ranks: every participant weighs 1.0, and a
rank outside the `select=` cohort contributes `ABSTAIN` (weight 0, a zero
table). The installed model is `base + aggregate`.

- **One norm over the whole delta.** Clipping each tensor to C separately
  would bound the whole delta by `C * sqrt(#tensors)`, which is not the
  sensitivity the noise is calibrated to.
- **The norm is exact and platform-independent.** For float32, float16 and
  bfloat16 every square is exact in float64 (at most 48 significant bits), and
  `math.fsum` rounds the exact sum once. So the norm does not depend on a
  summation order, a kernel or a device, and the tests recompute the clip
  centrally on upstream torch and compare bit for bit.
- **Rounding.** `d * float32(C / norm)` is one rounded multiply per element.
  The clipped norm can exceed C by a few float32 ulp (relative ~1e-7). This is
  not corrected.
- **sigma = 0 and C = inf is `FedAvg(weighted=False)`**, bit for bit: the same
  collectives, the same scaling spelling, no multiply on an unclipped rank, no
  noise tensor added.

## 3. Weights and sensitivity

The guarantee rests on the L2 sensitivity of the sum. With weight 1 per client,
adding or removing one client moves the sum by at most C, and the noise
standard deviation is `sigma * C`.

With sample-count weights `w_k`, the sensitivity becomes `max_k w_k * C`.
That is unbounded and data-dependent. The sum of the weights (the divisor)
would also be a private number on the wire with no noise. So
`DPFedAvg(weighted=True)` refuses by name (`NotImplementedError`), and a
`weight=` passed to `aggregate` or `synchronize` refuses by name
(`TypeError`). DP-FedAvg's bounded-weight variant, `w_k = min(n_k / W, 1)`,
is not built.

Under *replace-one* adjacency the sensitivity is 2C instead of C. This class
states the add/remove reading and computes neither.

## 4. The seed, and the threat model it serves

The noise must be identical on every rank, or the ranks install different
models. Two ways to get that were weighed. Both leave every participant
knowing the noise:

- draw it on one rank and broadcast it (a full-size tensor on the wire, and
  every receiver then knows it);
- draw it on every rank from a shared seed (chosen: one `int64` element on the
  wire).

`seed=None` (the default): rank 0 draws a 56-bit seed with `secrets` each
round and shares it in one `int64` `all_gather`. `seed=<int>`: the noise of
round *r* is a function of `(seed, r)`, for tests and replays. In both cases
the seed and the round index are agreed by digest (`federated.agree`) before
any noise is drawn. A rank whose seed or round counter differs refuses by
name.

**The consequence, stated plainly.** Every participating rank knows the noise
and can subtract it to recover the exact noiseless sum. Without secure
aggregation, the transport's hub (rank 0 of the star) also sees every clipped
delta. So this is **central DP with a trusted set of participants**. It
protects a client's contribution against an observer of the *released model*:
whoever later receives the weights, or a party that only sees the installed
model. It does not protect a client from its peers or from the aggregator.
That threat model (local DP, or distributed DP with noise shares under secure
aggregation) is `threat_model='local'`, and it refuses by name, citing
issue #219. A fixed `seed=` is weaker still: anyone who knows the integer can
subtract the noise from the released model.

## 5. The noise construction

Standard library only, so a test can regenerate it from the seed:

- key = `"torchnative.federated.DPFedAvg\0<seed>\0<round>\0<name>\0"` (UTF-8,
  decimal integers);
- block *i* = SHA-256(key + *i* as 8 bytes big-endian);
- each 32-byte block holds two 16-byte pairs: `u1 = ((a >> 11) + 1) * 2^-53`
  in (0, 1] and `u2 = (b >> 11) * 2^-53` in [0, 1);
- Box-Muller in float64 gives two normals per pair; each is scaled by
  `sigma * C` and cast once to the parameter's dtype.

The construction is keyed by parameter name and round, so a round never
reuses the previous round's draw, and one parameter's noise does not depend
on table order.

**Known weaknesses, not mitigated:**

- `math.log`, `cos` and `sin` come from the platform's libm, which need not
  round correctly. Two platforms can draw different last bits from one seed.
  The noise is therefore digested and agreed before it is added, and a
  disagreement refuses by name instead of installing two models. This refusal
  was **never produced**: both test ranks run on one host.
- This is not a floating-point-hardened sampler. Mironov (2012) shows that
  the low bits of a naively sampled floating-point Gaussian can leak.
- Generation is Python-speed, roughly one SHA-256 call per four elements.
  It was not measured, and a large model will feel it.

## 6. What is reported

`DPFedAvg.round_report()` is merged into
`FederatedDataParallel._get_federated_logging_data()`:

| key | meaning |
|---|---|
| `clip_norm` | C |
| `noise_multiplier` | sigma |
| `noise_std` | `sigma * C` (0.0 when sigma is 0) |
| `clipped` | whether **this rank's** delta was scaled |
| `pre_clip_norm` | this rank's exact delta norm before the clip |
| `noise_seed` | `'fixed'`, `'drawn'`, or None when sigma is 0. Never the seed itself |
| `dp_round` | the round index the noise was keyed by |
| `threat_model` | `'central'` |
| `accountant` | None: **no accountant** |

No epsilon is computed or reported. With full participation, an exact
accountant using only the standard library is possible (Gaussian DP:
`mu = sqrt(T) / sigma`, converted to (epsilon, delta) with `math.erfc`).
It is not built, because a `select=` cohort is subsampling, which it does not
cover. A follow-up has to decide that (agent proposal).

## 7. Tests (written, not run)

All are in `tests/_support/test_shim.py`. Each uses one two-process world on
backend `torchnative` (`_fed5_dp_round`, through `_fed3_spawn`), and every
expected value is recomputed centrally on upstream torch.

| test | what would make it fail |
|---|---|
| `test_dp_fedavg_clips_a_delta_above_the_clip_norm_and_leaves_one_below_it_unchanged` | clip skipped, the rank under C scaled, clip per parameter, a norm other than the exact one, noise drawn at sigma 0 |
| `test_dp_fedavg_adds_one_agreed_noise_draw_to_the_sum_and_every_rank_installs_the_same_model` | noise not added, per-rank noise, the wrong scale, noise added to the mean, the same noise every round, a seed that does not change the draw, missing logging keys |
| `test_dp_fedavg_with_no_noise_and_no_clip_is_fedavg_bit_for_bit` | any arithmetic difference from `FedAvg(weighted=False)` outside the clip and the noise |
| `test_dp_fedavg_draws_a_fresh_seed_on_rank_0_when_none_is_given` | each rank drawing its own seed, a seed reused across rounds, the drawn seed written into the report |
| `test_dp_fedavg_refuses_by_name_what_it_cannot_bound` | any of the refusals accepted (invalid C or sigma, inf C with noise, weights, `threat_model='local'`, bad seeds, ranks holding different seeds, the two keywords), or the group not surviving them |

<!-- DOCWATCH: symbol-in-file tests/_support/test_shim.py test_dp_fedavg_adds_one_agreed_noise_draw_to_the_sum_and_every_rank_installs_the_same_model present -->
<!-- DOCWATCH: symbol-in-file tests/_support/test_shim.py test_dp_fedavg_refuses_by_name_what_it_cannot_bound present -->

`test_federated_refuses_secure_aggregation_and_differential_privacy_by_name` changed
with this round. The `differential_privacy=` refusal now has to name
`algorithms.DPFedAvg` and must no longer say "round of its own".

**What these tests cannot find.** Anything about the strength of the guarantee.
They show the mechanism is the stated function of its inputs. They do not show
that a given sigma is private enough, because nothing computes epsilon. The
libm-disagreement refusal is never exercised.

## 8. Secure aggregation and selection, unchanged in substance

- `secure_aggregation=` still refuses. Its reason no longer stops at "the
  transport refuses p2p". It names the planned route, pairwise masking over
  the full-mesh point-to-point links of issue #219, pending a leader decision.
- Participant selection is as FEDERATED4 §5 left it: `cohort` agrees the
  participant set and serves a proper subset by attendance and abstention. Not
  reaching the unselected ranks at all needs `new_group` on the backend
  (issue #221).
