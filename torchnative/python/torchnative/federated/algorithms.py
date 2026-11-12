"""Federated aggregation algorithms: ``FedAvg``, ``FedAvgM``, ``FedProx``, ``SecAgg``.

``torch.federated.algorithms``. FL algorithms keep the ``Fed`` prefix. They are
built on the communication primitives of :mod:`torchnative.federated`
(``agree``, ``cohort``, ``ABSTAIN``, the wire guard and the collectives); the
module wrapper that drives them is
:class:`torchnative.federated.FederatedDataParallel`.

Moved, not rewritten, from ``torchnative.nn.federated`` (owner decision
2026-11-08, see docs/INTENT.md section 4).
"""

from __future__ import annotations

import torch

# The primitives are defined in the package ``__init__`` before it imports this
# module, so these names exist when this line runs.
from . import (ABSTAIN, RankDropped, _all_reduce, _arrival, _check_wire,
               _collective, _dist, _require_world, _total_weight, agree,
               cohort, digest)
from . import _secagg

__all__ = ["FedAvg", "FedAvgM", "FedProx", "SecAgg"]


class FedAvg:
    """Federated averaging: the weighted mean of the ranks' deltas.

    ``sum(w_k * d_k) / sum(w_k)`` over the ranks of a process group, element by
    element, computed with ``all_reduce(SUM)`` -- which is the identity McMahan
    et al. (2017) write for FedAvg and also, literally, what the collective
    does. Every rank ends holding the same table.

    **Weighting is explicit or refused.** ``FedAvg()`` requires a ``weight`` at
    every call. ``FedAvg(weighted=False)`` takes none and gives every rank 1.0.
    There is no default sample count and none is inferred from a batch: an
    aggregator that silently weighted every client equally when the caller
    meant to weight by data volume produces a different model and reports
    success, which docs/design/DESIGN.md §6 puts below a refusal.
    """

    def __init__(self, weighted=True):
        self.weighted = bool(weighted)

    def __repr__(self):
        return "FedAvg(weighted=%r)" % (self.weighted,)

    def resolve_weight(self, weight):
        """The weight this aggregator will use, or a refusal. Not a collective.

        Split out so that :class:`FederatedDataParallel` reports the same number the average
        was computed with, rather than a second guess at it.
        """
        if weight is ABSTAIN:
            # A rank outside the cohort. It attends the collectives -- it has
            # to, the others are waiting on it -- and contributes nothing.
            return 0.0
        if self.weighted:
            if weight is None:
                raise TypeError(
                    "torchnative.federated.algorithms.FedAvg.aggregate: weight= is "
                    "required. FedAvg weights each client by how much data it "
                    "trained on, and there is no safe guess -- weighting two "
                    "clients equally when one holds ten times the data is a "
                    "different model, silently.\n"
                    "Pass weight=<local sample count>, or say so with "
                    "FedAvg(weighted=False)."
                )
            w = float(weight)
            if w != w or w in (float("inf"), float("-inf")) or not w > 0.0:
                raise ValueError(
                    "torchnative.federated.algorithms.FedAvg: weight=%r. A weight has "
                    "to be finite and positive; a rank weighted 0 contributes "
                    "nothing while still counting as having participated"
                    % (weight,)
                )
            return w
        if weight is not None:
            raise TypeError(
                "torchnative.federated.algorithms.FedAvg(weighted=False) was given "
                "weight=%r. Unweighted means every rank counts 1.0, so a "
                "weight here would be accepted and ignored" % (weight,)
            )
        return 1.0

    def aggregate(self, table, weight=None, group=None):
        """The weighted average of ``table`` across the group. Returns a table.

        ``table`` is ``{name: tensor}`` -- what ``Delta.value`` holds and what
        ``Delta.load`` returns. Not a ``Delta``: which model an averaged offset
        belongs on is the caller's knowledge, the reasoning ``Delta.load``'s
        docstring gives.

        **A world of one is refused.** At ``world_size = 1`` this function *is*
        ``table``; returning it would make every test of this class pass no
        matter what the arithmetic below said, including no arithmetic at all.
        A federated aggregator that cannot fail is not evidence of anything, so
        the degenerate case is named instead of served.
        """
        world, rank = _require_world(group, "FedAvg.aggregate")
        if not table:
            raise ValueError(
                "torchnative.federated.algorithms.FedAvg: an empty table. A round that "
                "contributes no parameters would complete and change nothing"
            )
        w = self.resolve_weight(weight)

        for name, t in table.items():
            if not isinstance(t, torch.Tensor):
                raise TypeError(
                    "torchnative.federated.algorithms.FedAvg: %r is %r, not a Tensor"
                    % (name, type(t).__name__)
                )
            if not t.dtype.is_floating_point:
                raise NotImplementedError(
                    "torchnative.federated.algorithms.FedAvg: %r is %s. A delta holds "
                    "floating parameters; averaging an integer table would "
                    "round every intermediate, and no caller here means that"
                    % (name, t.dtype)
                )

        # The two ranks must be averaging the same thing. `all_reduce` sums
        # element-wise whatever it is handed, so a disagreement about which
        # parameters are covered -- or their order, shape or dtype -- comes
        # back as a number rather than as an error.
        agree(digest(table, values=False), group,
              "which parameters this round covers")

        total = _total_weight(w, group)
        if not total > 0.0:
            raise ValueError(
                "torchnative.federated.algorithms.FedAvg: the ranks' weights sum to "
                "%r, so there is no average. Every rank that reported either "
                "abstained (federated.ABSTAIN) or weighed nothing -- a round "
                "over an empty cohort would otherwise divide by zero and "
                "return a table of the right names full of nan" % (total,)
            )

        out = {}
        for name in sorted(table):
            t = table[name]
            # Scaled by a 0-dim tensor of the table's own dtype rather than by
            # a Python float: tensor-tensor arithmetic has one dtype and one
            # rounding, where a Python scalar leaves the width of the
            # intermediate to a promotion rule. What is compared against the
            # centrally computed average has to be reproducible by anyone
            # writing the same expression, and this is the spelling that is.
            scaled = (t.detach() * torch.tensor(w, dtype=t.dtype)).clone()
            _check_wire(scaled, name)
            _all_reduce(scaled, group,
                        "the sum of the ranks' deltas for %r" % (name,))
            out[name] = scaled / torch.tensor(total, dtype=t.dtype)
        return out


class FedAvgM:
    """FedAvg with server momentum: ``v <- beta v + avg``, and the round applies ``eta v``.

    Hsu, Qi and Brown (2019).  The client half is unchanged -- every rank still
    contributes ``w_k^local - w_global`` and the group still forms the weighted
    mean of those.  What differs is what the *server* does with that mean: it
    accumulates it into a velocity that carries across rounds, so a direction
    every round agrees on is amplified and one that reverses is damped.

    **Why this is a real second aggregator and not FedAvg with a knob.** At
    ``momentum=0`` it is FedAvg, bit for bit, and that is asserted rather than
    claimed.  Above zero its output at round *k* depends on rounds *1..k-1*,
    which no FedAvg does -- so a single round cannot distinguish them and the
    test does not try: it runs three and checks each against
    ``v_k = beta v_(k-1) + mean_k`` computed centrally.

    **Every rank holds the same velocity without communicating it.** The mean
    is already identical on both ranks (``all_reduce`` leaves the same bits
    everywhere, docs/distributed/FEDERATED.md §2.1), and the update applied to it is
    deterministic, so the states cannot drift.  This is checked rather than
    argued: the test asserts the two ranks' outputs are equal at every round.
    Nothing here reduces the velocity itself -- a server state that needed a
    collective to stay in step would be a different design and would have to
    say so.

    The state is keyed on parameter name and refuses a schema change between
    rounds: a momentum buffer silently restarted at zero for a parameter that
    was renamed is the FedAvg answer reported as the FedAvgM one.
    """

    def __init__(self, momentum=0.9, server_lr=1.0, weighted=True):
        m = float(momentum)
        if m != m or m in (float("inf"), float("-inf")) or not 0.0 <= m < 1.0:
            raise ValueError(
                "torchnative.federated.algorithms.FedAvgM: momentum=%r. Has to be "
                "finite and in [0, 1). At 1 the velocity never decays and the "
                "server diverges; below 0 it alternates sign every round, and "
                "neither is a thing a caller means" % (momentum,)
            )
        lr = float(server_lr)
        if lr != lr or lr in (float("inf"), float("-inf")) or not lr > 0.0:
            raise ValueError(
                "torchnative.federated.algorithms.FedAvgM: server_lr=%r. Has to be "
                "finite and positive; at 0 every round installs the zero "
                "update and the model never moves while the round reports "
                "success" % (server_lr,)
            )
        self.momentum = m
        self.server_lr = lr
        self._avg = FedAvg(weighted=weighted)
        self.velocity = {}
        self.rounds_seen = 0

    @property
    def weighted(self):
        return self._avg.weighted

    def __repr__(self):
        return "FedAvgM(momentum=%r, server_lr=%r, weighted=%r)" % (
            self.momentum, self.server_lr, self.weighted)

    def resolve_weight(self, weight):
        """Delegated, so the FederatedDataParallel reports the number the mean was formed with."""
        return self._avg.resolve_weight(weight)

    def aggregate(self, table, weight=None, group=None):
        """The weighted mean, accumulated into the velocity. Returns a table.

        Every refusal :meth:`FedAvg.aggregate` makes is made first, because
        the mean is formed by that method and not by a copy of it.
        """
        mean = self._avg.aggregate(table, weight=weight, group=group)
        if self.velocity and set(self.velocity) != set(mean):
            raise ValueError(
                "torchnative.federated.algorithms.FedAvgM: this round covers %s and "
                "the velocity was built over %s. A momentum buffer that "
                "restarted at zero for a renamed parameter would return the "
                "FedAvg answer and call it FedAvgM"
                % (sorted(mean), sorted(self.velocity))
            )
        out = {}
        for name in sorted(mean):
            m = mean[name]
            beta = torch.tensor(self.momentum, dtype=m.dtype)
            eta = torch.tensor(self.server_lr, dtype=m.dtype)
            previous = self.velocity.get(name)
            v = m.clone() if previous is None else (previous * beta + m)
            self.velocity[name] = v
            out[name] = v * eta
        self.rounds_seen += 1
        return out


class FedProx:
    """FedAvg's server step with FedProx's *local* objective.

    Li et al. (2020).  The proximal term is ``mu/2 * ||w - w_global||^2`` added
    to what each client minimises, which on the gradient is
    ``mu * (w - w_global)`` -- and that is the whole of the difference.  **The
    aggregation is FedAvg's, unchanged**, which is the fact that makes a
    "FedProx aggregator" a trap: a class that only overrode ``aggregate``
    would compute FedAvg and be indistinguishable from it, at every world size,
    for ever.

    So this refuses to aggregate unless :meth:`attach` has installed the term
    on a real local loop.  :class:`FederatedDataParallel` calls ``attach`` from
    its constructor; the low-level ``Delta.publish`` road does not, and there
    the refusal is the honest answer rather than a silent FedAvg.

    **The mechanism is a gradient hook.**  ``attach(fdp)`` registers a
    ``Tensor.register_hook`` on every federated parameter; during the user's own
    ``loss.backward()`` the hook returns ``grad + mu * (w - w_base)``, with
    ``w_base`` read from ``fdp.base`` at call time so a round that re-snapshots
    gets the new global weights without re-attaching.  The optimizer is the
    user's and is untouched.

    **What the term does, in a shape a test can catch.**  At the first local
    step ``w == w_global``, so the term is exactly zero and the step is
    FedAvg's, bit for bit.  From the second step on it pulls back toward the
    round's starting weights, so the delta is *smaller* than the unregularised
    one -- both halves are asserted, and the first one is what shows the term
    is the stated function of ``w - w_global`` rather than a constant nudge.
    """

    def __init__(self, mu=0.01, weighted=True):
        m = float(mu)
        if m != m or m in (float("inf"), float("-inf")) or not m > 0.0:
            raise ValueError(
                "torchnative.federated.algorithms.FedProx: mu=%r. Has to be finite and "
                "positive -- FedProx at mu=0 *is* FedAvg, and a caller who "
                "means FedAvg should say FedAvg rather than reach it through a "
                "parameter that reads like a tuning knob" % (mu,)
            )
        self.mu = m
        self._avg = FedAvg(weighted=weighted)
        self._attached = None
        self._handles = []
        #: How many times the term was added to a gradient since the last
        #: aggregate. A round whose backward never ran the hook trained
        #: without the term, however it was wired.
        self.applied = 0

    @property
    def weighted(self):
        return self._avg.weighted

    def __repr__(self):
        return "FedProx(mu=%r, weighted=%r)" % (self.mu, self.weighted)

    def resolve_weight(self, weight):
        return self._avg.resolve_weight(weight)

    def attach(self, fdp):
        """Install the proximal term on ``fdp``'s federated parameters. Idempotent."""
        if self._attached is fdp:
            return self
        if self._attached is not None:
            raise RuntimeError(
                "torchnative.federated.algorithms.FedProx: this algorithm is already "
                "attached to another FederatedDataParallel. The proximal term is "
                "defined against one round's w_global; sharing an instance "
                "across two local loops would pull each toward the other's base"
            )
        mu = self.mu

        def make(name, param):
            def proximal(grad):
                base = fdp.base.get(name)
                if base is None:
                    return grad
                self.applied += 1
                scale = torch.tensor(mu, dtype=grad.dtype)
                return grad + (param.detach() - base) * scale
            return proximal

        for name, param in fdp.federated_parameters().items():
            self._handles.append(param.register_hook(make(name, param)))
        self._attached = fdp
        return self

    def detach(self):
        """Remove the term again. Here so that attaching is reversible in a test."""
        for handle in self._handles:
            handle.remove()
        self._handles = []
        self._attached = None
        return self

    def aggregate(self, table, weight=None, group=None):
        """FedAvg's weighted mean -- but only if the local term was installed and ran.

        A FedProx that never attached is FedAvg wearing a different name, and it
        would report success at every world size. docs/design/DESIGN.md §6.
        A rank outside the cohort (``ABSTAIN``) trains nothing, so it is not
        asked whether the term ran.
        """
        if self._attached is None:
            raise RuntimeError(
                "torchnative.federated.algorithms.FedProx: the proximal term was never "
                "installed, so this would compute FedAvg's weighted mean and "
                "report it as FedProx. The difference between the two is "
                "entirely in the local objective -- mu*(w - w_global) on the "
                "gradient -- and the server step is identical, so nothing "
                "downstream could tell them apart.\n"
                "Check: federated.FederatedDataParallel(model, "
                "algorithm=FedProx(mu=...)) attaches it, or call "
                "FedProx.attach(fdp) before the local steps."
            )
        if weight is not ABSTAIN and self.applied == 0:
            raise RuntimeError(
                "torchnative.federated.algorithms.FedProx: the proximal term "
                "is attached but its gradient hook did not run since the last "
                "round, so the local training was plain FedAvg's. Train through "
                "loss.backward() on the wrapped module's federated parameters "
                "before synchronize()."
            )
        self.applied = 0
        return self._avg.aggregate(table, weight=weight, group=group)


#: The point-to-point tag the pairwise seeds travel on. Any fixed value works
#: (frames are matched per ``(src, tag)`` in posting order); this one is not
#: used by anything else in the tree. Agent proposal, not confirmed.
_SEED_TAG = 0x5EC0A6

#: The PRG label of the masked total weight. Parameter labels are
#: ``b"p:" + name``, so no parameter name can produce this one.
_WEIGHT_LABEL = b"w"


def _param_label(name):
    return b"p:" + name.encode("utf-8")


class SecAgg:
    """Secure aggregation by pairwise masking (Bonawitz et al. 2017), around ``FedAvg``.

    Provenance. The algorithm is the leader's decision (2026-11-13: "secure
    aggregation via pairwise masking (Bonawitz-style) is approved as the
    algorithm"), with no cryptographic dependency approved, so this is stdlib
    only. The wrapper shape, the fixed-point scale, the generator, the wire
    shape and this API are an **agent proposal, not confirmed**
    (docs/distributed/SECAGG.md). Graded *builds*: nothing ran in the round
    that wrote it.

        fdp = federated.FederatedDataParallel(
            model, algorithm=algorithms.SecAgg(algorithms.FedAvg()))

    **Why an algorithm wrapper and not a flag.** The masking replaces exactly
    one thing FedAvg does, how ``sum(w_k d_k)`` and ``sum(w_k)`` are formed,
    and keeps every refusal FedAvg makes about weights. An algorithm object is
    where the package already keeps that (``Delta.publish(aggregator=...)``
    reaches it on the low-level road too, which a flag on
    :class:`FederatedDataParallel` would not), and it composes the way
    ``FedAvgM`` composes ``FedAvg``. So ``FederatedDataParallel
    (secure_aggregation=True)`` stays a refusal that points here: two spellings
    of one mechanism would be one too many.

    **One round.** Over the default process group of backend ``torchnative``:

    1. Every pair ``i < j`` agrees a fresh 32-byte seed over its *direct* p2p
       link: ``i`` draws it with ``secrets`` and ``dist.isend`` s it to ``j``.
       A seed passes through rank 0 only when ``i`` or ``j`` is 0.
    2. Each rank quantises ``w * d`` to ``q = round(w * d * 2**f)``, then adds
       ``PRG(s_ij)`` for every ``j > i`` and subtracts ``PRG(s_ji)`` for every
       ``j < i``, mod ``2**k``. The PRG is HMAC-SHA-256 in counter mode over
       (seed, round, label, chunk): ``_secagg.mask_words``.
    3. The masked residues go through the existing ``all_reduce(SUM)`` as
       int64. The hub (rank 0, the star's centre) sees only masked values.
       Every pair's mask appears once with each sign, so the integer sum mod
       ``2**k`` is ``sum(q)`` exactly, and is decoded to a float once.

    **The bound** (``_secagg`` module docstring): ``k = 63 -
    world.bit_length()`` so the hub's int64 sum cannot overflow, and each rank
    must keep ``|w * d| <= B / 2**f`` with ``B = (2**(k-1) - 1) // world`` so
    the centred decode is unique. A value past it, or a nan or inf, is refused
    by name on every rank (one int64 count travels in the clear for that: the
    hub learns *how many* ranks overflowed, nothing about by how much).

    **Quantisation error.** Per element, ``|sum(q)/2**f - sum(w_k d_k)| <=
    world * 2**-(f+1) + 2**-53 * sum|w_k d_k|``; the decoded mean is then one
    correctly rounded float64 division ``sum(q) / sum(q_w)`` and one cast to the
    table's dtype. ``tests/distributed/test_secagg.py`` derives its tolerance
    from exactly these terms.

    **The total weight is masked too.** It is the local sample count, which is
    itself something a client may not want the hub to learn, and masking it
    costs one more element. The divisor is the decoded ``sum(q_w) / 2**f``,
    exact when every weight is an integer (or any multiple of ``2**-f``).
    :class:`FederatedDataParallel` reads it from :attr:`last_total_weight`
    rather than running its own unmasked weight collective.

    **Threat model.** An honest-but-curious aggregator (rank 0, the hub) does
    not see any individual update or weight, only masked residues and their
    sum. Not protected: anyone who can read a pair's link learns that pair's
    seed (the links are not encrypted; the opt-in pre-shared key of D10 /
    PR #238 authenticates, it does not hide); ranks that collude can unmask
    the others (``world - 1`` colluders learn the last one's update from the
    sum); a malicious aggregator that alters results or lies about who is
    present is not defended against. The sum itself is public to every rank,
    as it is in FedAvg.

    **Dropout aborts the round.** A rank that leaves after the seeds were
    agreed leaves its pairs' masks uncancelled, and Bonawitz's recovery
    (Shamir shares of each seed, an unmasking round) **is not built**. So a
    missing rank raises :class:`RankDropped` naming secure aggregation, and
    ``FederatedDataParallel(on_missing='average_arrived')`` with this
    algorithm is refused at construction.

    **Selection.** ``cohort()`` / ``ABSTAIN`` works unchanged: a rank outside
    the cohort contributes a masked zero at a masked weight of zero, so the hub
    cannot tell an abstention from a contribution by its values.

    **Combinations.** Only ``FedAvg`` is wrapped. ``FedAvgM`` (its server
    momentum would be fine, it acts on the mean), ``FedProx`` (its local hook
    would need forwarding) and ``SecAgg`` itself are refused by name until a
    round tests them. DPFedAvg (PR #228) is not in this tree; combining it
    means adding the noise before quantisation, and is not built here.
    """

    #: Read by :class:`FederatedDataParallel`: the round's total weight is
    #: :attr:`last_total_weight`, not a separate unmasked collective.
    secure = True

    def __init__(self, inner=None, fraction_bits=_secagg.DEFAULT_FRACTION_BITS):
        if inner is None:
            inner = FedAvg()
        if isinstance(inner, SecAgg):
            raise NotImplementedError(
                "torchnative.federated.algorithms.SecAgg(SecAgg(...)): masking "
                "twice is the same sum with twice the seeds, and nothing was "
                "built or tested for it. Wrap FedAvg once.")
        if not isinstance(inner, FedAvg):
            raise NotImplementedError(
                "torchnative.federated.algorithms.SecAgg(%r): only FedAvg is "
                "wrapped. Secure aggregation replaces how sum(w_k d_k) and "
                "sum(w_k) are formed, and for %s that wiring is not built: "
                "FedAvgM's server momentum and FedProx's local gradient hook "
                "would each need forwarding, and DPFedAvg's noise would have "
                "to be added before quantisation. Refused rather than "
                "silently masking a different algorithm's inputs.\n"
                "Check: algorithms.SecAgg(algorithms.FedAvg(weighted=...))."
                % (inner, type(inner).__name__))
        f = fraction_bits
        if isinstance(f, bool) or not isinstance(f, int) or not 1 <= f <= 52:
            raise ValueError(
                "torchnative.federated.algorithms.SecAgg: fraction_bits=%r. An "
                "integer in [1, 52]: below 1 there is no fraction at all, and "
                "above 52 the float64 product w * d has no bits left for the "
                "scale to keep" % (fraction_bits,))
        self.inner = inner
        self.fraction_bits = f
        #: Rounds this instance has masked; part of every mask's input, and
        #: agreed across the ranks before it is used.
        self.rounds = 0
        #: The decoded total weight of the last round (``sum(q_w) / 2**f``).
        self.last_total_weight = None
        self._last_world = None

    def __repr__(self):
        return "SecAgg(%r, fraction_bits=%d)" % (self.inner, self.fraction_bits)

    @property
    def weighted(self):
        return self.inner.weighted

    def resolve_weight(self, weight):
        """Delegated, so every refusal FedAvg makes about weights is made here."""
        return self.inner.resolve_weight(weight)

    def layout(self, world):
        """``(k, B)`` for ``world``, or a refusal if ``f`` leaves no integer range."""
        k = _secagg.modulus_bits(world)
        bound = _secagg.per_rank_bound(world, k)
        if bound < (1 << self.fraction_bits):
            raise NotImplementedError(
                "torchnative.federated.algorithms.SecAgg: world_size %d with "
                "fraction_bits=%d. The int64 accumulator leaves k = %d modulus "
                "bits, so one rank may contribute |q| <= %d, which is less "
                "than 1.0 at this scale. Lower fraction_bits."
                % (world, self.fraction_bits, k, bound))
        return k, bound

    def logging_data(self):
        """What :meth:`FederatedDataParallel._get_federated_logging_data` adds. Never a seed."""
        out = {"secure_aggregation": True,
               "fraction_bits": self.fraction_bits,
               "scale": float(1 << self.fraction_bits),
               "secure_rounds": self.rounds}
        if self._last_world is not None:
            k, bound = self.layout(self._last_world)
            out["modulus_bits"] = k
            out["per_rank_bound"] = bound / float(1 << self.fraction_bits)
        return out

    # -- the round ---------------------------------------------------------

    def aggregate(self, table, weight=None, group=None):
        """The weighted mean of ``table`` across the group, masked. Returns a table."""
        world, rank = _require_world(group, "SecAgg.aggregate")
        if _arrival() is not None:
            raise NotImplementedError(
                "torchnative.federated.algorithms.SecAgg inside "
                "tolerate_missing() (on_missing='average_arrived'). A rank that "
                "drops after the seeds were agreed leaves its pairs' masks in "
                "the sum, and removing them needs Bonawitz's recovery: Shamir "
                "shares of every seed and an unmasking round. Recovery is not "
                "built, so a partial secure sum is refused rather than "
                "returned as noise.")
        if not table:
            raise ValueError(
                "torchnative.federated.algorithms.SecAgg: an empty table. A "
                "round that contributes no parameters would complete and "
                "change nothing")
        for name, t in table.items():
            if not isinstance(t, torch.Tensor):
                raise TypeError(
                    "torchnative.federated.algorithms.SecAgg: %r is %r, not a "
                    "Tensor" % (name, type(t).__name__))
            if not t.dtype.is_floating_point:
                raise NotImplementedError(
                    "torchnative.federated.algorithms.SecAgg: %r is %s. A delta "
                    "holds floating parameters" % (name, t.dtype))
        w = self.resolve_weight(weight)
        k, bound = self.layout(world)
        f = self.fraction_bits

        # The same premise FedAvg checks, plus the two numbers every mask and
        # every encoding depend on. A pair that disagreed about the round or
        # the scale would sum masks that do not cancel: a wrong aggregate
        # that no later step notices.
        agree(digest(table, values=False), group,
              "which parameters this round covers")
        round_index = self.rounds
        agree(_secagg.round_tag(round_index, f), group,
              "the secure-aggregation round number and fraction_bits")
        self.rounds += 1
        self._last_world = world

        names = sorted(table)
        plain = {}
        shapes = {}
        worst = 0.0
        status = "ok"
        for name in names:
            t = table[name].detach()
            shapes[name] = (tuple(t.shape), t.dtype)
            ints, top, st = _secagg.quantise(
                t.reshape(-1).tolist(), w, f, bound)
            plain[_param_label(name)] = ints
            worst = max(worst, top)
            if st != "ok" and status != "nonfinite":
                status = st
        ints, top, st = _secagg.quantise([1.0], w, f, bound)
        plain[_WEIGHT_LABEL] = ints
        worst = max(worst, top)
        if st != "ok" and status != "nonfinite":
            status = st
        self._refuse_out_of_range(status, worst, world, rank, k, bound, group)

        seeds = self._exchange_seeds(group, world, rank)
        try:
            labels = [_param_label(n) for n in names] + [_WEIGHT_LABEL]
            masked = {}
            for label in labels:
                masked[label] = self._contribute(
                    label, plain[label], seeds, rank, round_index, k)
        finally:
            # The seeds are this round's and nobody else's: not kept on the
            # instance, not logged, not returned.
            seeds.clear()
        sums = self._masked_sum(masked, labels, group, world)

        s_w = _secagg.centred(sums[_WEIGHT_LABEL][0], k)
        total = s_w / float(1 << f)
        self.last_total_weight = total
        if not s_w > 0:
            raise ValueError(
                "torchnative.federated.algorithms.SecAgg: the ranks' masked "
                "weights sum to %r, so there is no average. Every rank either "
                "abstained (federated.ABSTAIN) or weighed nothing" % (total,))

        out = {}
        for name in names:
            shape, dtype = shapes[name]
            # sum(q) / sum(q_w) is int / int: one correctly rounded float64
            # division of the exact quotient, and the 2**f cancels.
            values = [_secagg.centred(s, k) / s_w
                      for s in sums[_param_label(name)]]
            out[name] = torch.tensor(values, dtype=torch.float64).to(
                dtype).reshape(shape)
        return out

    def _refuse_out_of_range(self, status, worst, world, rank, k, bound,
                             group):
        """Refuse on every rank if any rank's value has no encoding. One count in the clear."""
        flag = torch.tensor([0 if status == "ok" else 1], dtype=torch.int64)
        _all_reduce(flag, group,
                    "how many ranks hold a value past the secure-aggregation "
                    "bound")
        count = int(flag[0].item())
        if count == 0:
            return
        limit = bound / float(1 << self.fraction_bits)
        raise ValueError(
            "torchnative.federated.algorithms.SecAgg: %d of %d rank(s) hold a "
            "weighted update past the fixed-point bound, so the masked sum "
            "would wrap mod 2**%d and decode to a wrong number that looks "
            "right. Refused on every rank. The bound is |w * d| <= %r "
            "(B / 2**f with B = (2**(k-1) - 1) // world = %d, k = %d, "
            "f = %d, world = %d). Rank %d: %s.\n"
            "Check: scale the update down, or lower fraction_bits to buy "
            "range at the cost of resolution."
            % (count, world, k, limit, bound, k, self.fraction_bits, world,
               rank,
               "a nan or inf, which has no fixed-point encoding"
               if status == "nonfinite" else
               "largest |w * d| = %r" % (worst,) if status == "overflow" else
               "within the bound (another rank is not)"))

    def _exchange_seeds(self, group, world, rank):
        """``{peer: seed}``: one fresh 32-byte seed per pair, over the direct link.

        Rank ``i`` draws the seed of every pair ``(i, j)`` with ``j > i`` and
        sends it to ``j``; it receives the seed of every pair ``(h, i)`` with
        ``h < i`` from ``h``. Sends are posted first and waited last, so no
        rank blocks on a receive while a peer blocks on its own.
        """
        import secrets

        dist = _dist()

        def run():
            seeds = {}
            works = []
            for peer in range(rank + 1, world):
                seed = secrets.token_bytes(_secagg.SEED_BYTES)
                seeds[peer] = seed
                works.append(dist.isend(
                    torch.tensor(list(seed), dtype=torch.uint8), dst=peer,
                    group=group, tag=_SEED_TAG))
            for peer in range(rank):
                holder = torch.zeros(_secagg.SEED_BYTES, dtype=torch.uint8)
                dist.recv(holder, src=peer, group=group, tag=_SEED_TAG)
                seeds[peer] = bytes(int(b) for b in holder.tolist())
            for work in works:
                work.wait()
            return seeds

        try:
            return _collective(run, group, "the secure-aggregation seed exchange")
        except RankDropped as exc:
            raise self._dropped(exc, "the seed exchange")

    def _contribute(self, label, plain, seeds, rank, round_index, k):
        """This rank's masked residues for ``label``: ``(q + net mask) mod 2**k``."""
        mask = self._net_mask(label, len(plain), seeds, rank, round_index, k)
        return _secagg.add_mod(plain, mask, k)

    def _net_mask(self, label, n, seeds, rank, round_index, k):
        return _secagg.net_mask(label, n, seeds, rank, round_index, k)

    def _masked_sum(self, masked, labels, group, world):
        """``{label: [sum of residues mod 2**k]}`` through the existing all_reduce(SUM)."""
        k = _secagg.modulus_bits(world)
        modulus = 1 << k
        sums = {}
        try:
            for label in labels:
                t = torch.tensor(masked[label], dtype=torch.int64)
                _check_wire(t, label.decode("utf-8", "replace"))
                _all_reduce(t, group,
                            "the masked sum for %r" % (label.decode(
                                "utf-8", "replace"),))
                sums[label] = [int(v) % modulus for v in t.tolist()]
        except RankDropped as exc:
            raise self._dropped(exc, "the masked sum")
        return sums

    @staticmethod
    def _dropped(exc, during):
        return RankDropped(
            "torchnative.federated.algorithms.SecAgg: a rank did not report "
            "during %s, so this secure round has no aggregate. Its pairs' "
            "masks are in the other ranks' contributions and only its seeds "
            "remove them: Bonawitz's dropout recovery (Shamir shares of every "
            "seed, an unmasking round) is not built. The round is aborted, "
            "not decoded into noise.\n(%s)" % (during, exc),
            rank=exc.rank, missing=exc.missing, during=during)
