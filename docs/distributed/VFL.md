# VFL: `torch.federated.FederatedFeatureParallel`, vertical federated learning

Issue #293. SPEC S6.8. Written 2026-11-13.

**Grade: builds.** Every file below passed `python -m py_compile` and nothing else. No suite, gate or probe ran
in the round that wrote it (AGENTS.md rule 10: agents do not run suites; the gate runs in CI). Nothing here is
*reaches* or *agrees* until CI runs `tests/distributed/test_vfl.py` (AGENTS.md §16). Read every "does" below as
"is written to".

**Provenance.** The name `torch.federated.FederatedFeatureParallel` and its place in `torch.federated` are the
owner's (2026-11-08, INTENT §4: "연합학습과 군집학습 부분은 확정하고 문서 개정하고, 리드미 수정하고, 지금
리펙토링은 바로 들어가도록 해."). **Everything else in this document is an agent proposal, not confirmed**: the
split-learning shape, the constructor's arguments, `backward()`, concatenation as the merge, the message
protocol and its tags, the sample-id digest, the refusals and their wording.

## 1. What was built

| Piece | Where | What it does |
|---|---|---|
| Wrapper | `torchnative/python/torchnative/federated/feature_parallel.py` `FederatedFeatureParallel` | wraps this rank's bottom module; on the label holder also the top module |
| Refusal type | same file, `SamplesMisaligned` | a `ValueError` raised on every rank of a step whose batches differ |
| Export | `torchnative/python/torchnative/federated/__init__.py` `__all__` | `torch.federated.FederatedFeatureParallel` through the existing add-hook `torchnative/python/torch/federated/__init__.py` (unchanged: it star-imports `__all__`) |
| Suite | `tests/distributed/test_vfl.py`, registered in `.github/scripts/gate_suites.py` for both runners | section 5 |

<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/federated/feature_parallel.py FederatedFeatureParallel present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/federated/feature_parallel.py SamplesMisaligned present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/federated/__init__.py FederatedFeatureParallel present -->

## 2. The shape (agent proposal, not confirmed)

Features are split by column: every rank holds some columns of the same samples. Every rank wraps a **bottom
module** over its own columns, whose output is the rank's **embedding**, a CPU `(batch, width)` float32 or float64
tensor. One rank, the **label holder** ("active party"), also holds the **top module** and the labels. The other
ranks are **passive parties**.

The loop stays a torch loop, the way `FederatedDataParallel` keeps DDP's:

```python
ffp = torch.federated.FederatedFeatureParallel(
    bottom, label_holder=0, top=top if rank == 0 else None)
optimizer = torch.optim.SGD(ffp.parameters(), lr=0.1)
for ids, x_local, y in loader:              # y exists on the label holder only
    optimizer.zero_grad()
    out = ffp(x_local, sample_ids=ids)
    if ffp.is_label_holder:
        loss_fn(out, y).backward()          # an ordinary backward
    else:
        ffp.backward()                      # receives this rank's gradient
    optimizer.step()
```

**The label holder's loop is an ordinary loop.** `forward` returns `top(cat(embeddings))`; the caller computes
the loss from it and the labels and calls `loss.backward()`. Each received embedding is a leaf with a gradient
hook, so each passive party's gradient leaves the moment the backward reaches it. `ffp.backward(loss)` is the
same backward followed by a check that every passive party's gradient was sent; with a plain `loss.backward()`
the same check runs at the next `forward`.

**A passive party's loop differs in one line.** It has no labels, so it has no loss: `forward` returns its
embedding (still attached to its bottom module's graph), and `ffp.backward()` receives that embedding's gradient
from the label holder and runs `embedding.backward(gradient)`, which fills the bottom module's `.grad` exactly as
a `loss.backward()` would. `ffp.backward(loss)` with `loss=None` on passive parties lets one loop body serve
every rank.

Each rank steps its own optimizer over its own parameters; nothing about the optimizer crosses ranks.

### 2.1 Concatenate, not sum (agent proposal)

The label holder concatenates the embeddings along dim 1 **in rank order**, its own at its rank's position, so
the top module's input width is the sum of the widths. Reasons: widths may differ between parties (they do in the
tests: 2, 3 and 2); the top module can weight parties differently, which a sum erases; the gradient to each party
is a slice of the concatenation's gradient, so no party receives anything computed from another party's slice.
Summing would require one width everywhere and would hand every party the same gradient. `merge="sum"` refuses by
name rather than being offered untested.

### 2.2 Why a hook on the label holder

The alternative was a mandatory `ffp.backward(loss)` that sends `.grad` of each received leaf after the backward.
The hook keeps the label holder's loop unchanged (`loss.backward()`), and the cost is that "the backward never
reached an embedding" is detected one call later. That case is not left to hang: the label holder sends status 0
to every party still waiting, and both sides refuse by name (section 4).

## 3. One step on the wire

All point-to-point (S6.12, `docs/distributed/P2P.md`), tagged `tag_base + kind` where `tag_base` moves by 8 per
instance so frames of an instance that refused halfway are never matched by the next one.

| Order | Message | Direction | Shape |
|---|---|---|---|
| 1 | header | passive to label holder | `(6,)` int64: step, batch, width, dtype code, grad mode, sample digest |
| 2 | verdict | label holder to passive | `(5,)` int64: step, ok, holder's digest, bad-rank mask, reason bits |
| 3 | embedding | passive to label holder | `(batch, width)`, only if the verdict was ok |
| 4 | status | label holder to passive | `(2,)` int64: step, ok (0 if the backward never reached this embedding) |
| 5 | gradient | label holder to passive | `(batch, width)`, only if the status was ok |

Under `torch.no_grad()` on every rank, messages 4 and 5 are not sent and no `backward` is called.

Construction runs one collective: `federated.agree` over a digest of (world, label holder, instance index), so two
ranks that disagree about the label holder refuse by name instead of waiting on each other for the group timeout.

**What never leaves its rank:** the raw features (they enter only the local bottom module), the labels (they are
never handed to this class) and the sample ids (only a 56-bit SHA-256 digest crosses). The suite asserts this from
the backend, not from this table: section 5.

### 3.1 Batch alignment

Feeding the same samples in the same order on every rank is the caller's job, and getting it wrong fails silently:
the top module trains on rows that do not belong together and the loss still falls. So every step the label
holder compares each header's digest and batch size with its own **before any embedding is sent**. The digest is
over the ids in order, so the same samples in another order are a mismatch (the test reverses one rank's batch). On
a mismatch every rank raises `SamplesMisaligned` naming the ranks that differ; the step index has moved on every
rank alike, so the next aligned step proceeds. The digest detects an accident, not an adversary: 56 bits, and
nothing stops a party that wants to collide it.

## 4. Refused by name

| What | Where | Why |
|---|---|---|
| no process group | constructor | there is nobody to exchange with |
| a world of one | constructor (`_require_world`) | one party is not vertical FL; the joined model would be the only model |
| no `label_holder=`, or not a rank | constructor | there is no loss and nowhere to send an embedding |
| `top=` missing on the label holder, or given on a passive party | constructor | only the label holder runs the top module |
| `group=` | constructor | point-to-point inside a sub-group needs `new_group`/`split_group` (issue #221) |
| `merge=` other than `"cat"` | constructor | section 2.1 |
| `secure_aggregation=`, `differential_privacy=`, `encryption=` | constructor | each needs a protocol (homomorphic encryption or MPC of the first top layer; clipping, calibrated noise and an accountant; a key exchange and an authenticated cipher in the transport), not a flag |
| world above 62 | constructor | the verdict's bad-rank mask is one int64 |
| no `sample_ids=`, non-integer ids, wrong count | `forward` | alignment cannot be checked |
| a non-CPU embedding | `forward` | staging vulkan, mps or npu tensors through host memory (S6.10) is not built for point-to-point |
| an embedding that is not 2-D float32/float64, or that does not require grad in grad mode | `forward` | |
| misaligned batches, a step-index, grad-mode or dtype disagreement | `forward`, every rank | section 3.1 |
| a passive party's next `forward` before its `backward` | `forward` | it would train without the gradient already sent |
| a loss on a passive party; no loss on the label holder; `backward` with no step pending | `backward` | |
| a backward that never reached an embedding | label holder's next `forward` or `backward(loss)`, and that party's `backward` | the party would otherwise wait out the group timeout |
| a second gradient for one embedding in one step (`retain_graph=True`) | the hook | the party could not tell a sum from one gradient |

A refusal raised locally after the step index moved (for example a wrong embedding shape on one rank only) leaves
the other ranks waiting on that rank's header until the group timeout. Named here because it is not refused.

## 5. Tests written (not run)

`tests/distributed/test_vfl.py`. The oracles, because upstream torch has no vertical-FL module:

* **(a) the joined model**, the same bottoms, top and SGD trained in one process on the joined features with
  ordinary autograd. On the shim the split run must be **bit-equal**: it performs the same operations in the same
  order on bit-identical operands and only inserts copies (the wire is bit-exact). The `tanh` after each bottom
  `Linear` is load-bearing: its backward writes a fresh contiguous tensor, so the bottom's backward sees the same
  layout whether its gradient came from a slice of `cat`'s backward or from a received copy. On upstream torch it
  is held to the tolerance derived from upstream's own float32-vs-float64 run of the same training, floored at 8
  ulp (`test_collect2.c2_assert_matches`, AGENTS.md §16).
* **(b) upstream torch and gloo doing the exchange by hand** with `dist.send`/`dist.recv`, held to gloo's own
  float32-vs-float64 tolerance; and checked bit for bit against upstream's joined model, so neither oracle is
  trusted alone.

Layouts: two parties (feature widths 3 and 5, embedding widths 2 and 3, label holder rank 0) and three (4 more
columns for a third party with width 2, label holder **rank 1**, so rank order is not "holder first"). Batch 4,
three SGD steps, cross entropy over three classes. Inputs are multiples of a power of two, exact in both precisions.

<!-- DOCWATCH: symbol-in-file tests/distributed/test_vfl.py test_split_training_equals_the_joined_model_bit_for_bit_at_two_and_three_parties present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_vfl.py test_split_training_agrees_with_upstream_gloo_doing_the_exchange_by_hand present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_vfl.py test_only_embedding_and_gradient_shaped_tensors_cross_ranks present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_vfl.py test_a_misaligned_batch_refuses_by_name_on_every_rank_and_the_next_step_recovers present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_vfl.py test_nullified_zero_gradient_to_one_party_is_caught_by_the_joined_comparison present -->

| Test | Asserts |
|---|---|
| `test_split_training_equals_the_joined_model_bit_for_bit_at_two_and_three_parties` | loss, every gradient, every updated value, three steps, both layouts; the count of compared quantities is pinned |
| `test_the_label_holder_is_not_rank_zero_at_three_parties_and_its_slot_is_its_rank` | the widths the label holder concatenated, in rank order |
| `test_split_training_agrees_with_the_upstream_joined_model_within_its_derived_tolerance` | oracle (a) on upstream |
| `test_the_gloo_exchange_by_hand_is_the_upstream_joined_model_bit_for_bit` | oracle (b) against oracle (a) |
| `test_split_training_agrees_with_upstream_gloo_doing_the_exchange_by_hand` | oracle (b) |
| `test_only_embedding_and_gradient_shaped_tensors_cross_ranks` | every frame `transport.Mesh.send` was handed while training, decoded: exactly section 3's messages, in order, at those shapes and dtypes; nothing between passive parties; the embedding frame's bytes are the embedding's; no collective while training and exactly one `allgather` at construction (every collective method of `ProcessGroupTorchnative` is wrapped) |
| `test_no_raw_feature_label_or_sample_id_byte_string_is_in_any_payload` | each step's feature batch, every feature row, the labels and the ids, searched for byte for byte in every payload |
| `test_a_misaligned_batch_refuses_by_name_on_every_rank_and_the_next_step_recovers` | one rank reverses its batch: `SamplesMisaligned` on every rank naming that rank, then an aligned step trains |
| `test_a_rank_in_no_grad_mode_refuses_the_step_on_every_rank` | grad-mode disagreement |
| `test_a_loss_that_reaches_no_embedding_refuses_on_both_sides_instead_of_hanging` | status 0, both sides refuse, the next step trains |
| `test_local_misuse_refuses_by_name_before_anything_is_sent` | ids, `backward` misuse, a `meta` embedding |
| `test_the_constructor_refuses_by_name_what_is_not_built` | section 4's constructor rows, then a fresh instance still trains |
| `test_no_process_group_and_a_world_of_one_refuse_by_name` | |
| `test_nullified_zero_gradient_to_one_party_is_caught_by_the_joined_comparison` | nullification 1, below |
| `test_nullified_batch_digest_lets_a_misaligned_batch_through` | nullification 2, below |
| `test_the_add_hook_exports_the_same_class` | `torch.federated.FederatedFeatureParallel is torchnative.federated.FederatedFeatureParallel` |
| `test_every_rank_really_ran_the_shim_and_the_oracles_really_ran_upstream` | |

**Nullifications (AGENTS.md §17.5), run inside the suite.** (1) `_send_gradient` is patched to send zeros to the
last rank. The bit-equality comparison must report every gradient and value of that rank's bottom module at every
step, and nothing on any other rank at step 0 (from step 1 the stale bottom feeds the top a different embedding,
so propagation is not asserted either way). (2) `_batch_digest` is replaced by a constant, and the reversed batch
must then be **accepted**, so the misalignment refusal is shown to be the digest's. Neither has been run.

## 6. Known gaps and risks, read off source, not measured

1. Nothing has run. The first measurement is CI's; bit equality in section 5 (a) is an argument from the
   operation sequence, not an observation. If CI shows a last-bit difference, look first at tensor layout in the
   backward of the first op after the embedding.
2. A refusal raised on one rank only after the step index moved leaves the others waiting until the group timeout
   (section 4, last paragraph).
3. The label holder sees every embedding in the clear, and every passive party sees its embedding's gradient,
   which is known to leak information about labels (label leakage in split learning). Nothing here mitigates it;
   `secure_aggregation=` and `differential_privacy=` refuse rather than pretend.
4. One label holder, one top module, one merge. Multiple label holders, a passive party with its own top, and
   asynchronous or stale embeddings are not built and are not refused by a dedicated door: they have no argument.
5. `sample_ids` alignment is per step; nothing checks that the parties used the same sample universe (set
   intersection, private set intersection) before training.
