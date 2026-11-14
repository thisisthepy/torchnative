"""Federated aggregation algorithms: ``FedAvg``, ``FedAvgM``, ``FedProx``, ``DPFedAvg``.

``torch.federated.algorithms``. FL algorithms keep the ``Fed`` prefix. They are
built on the communication primitives of :mod:`torchnative.federated`
(``agree``, ``cohort``, ``ABSTAIN``, the wire guard and the collectives); the
module wrapper that drives them is
:class:`torchnative.federated.FederatedDataParallel`.

Moved, not rewritten, from ``torchnative.nn.federated`` (owner decision
2026-11-08, see docs/INTENT.md section 4).
"""

from __future__ import annotations

import hashlib
import math

import torch

# The primitives are defined in the package ``__init__`` before it imports this
# module, so these names exist when this line runs.
from . import (ABSTAIN, RankDropped, _all_gather, _all_reduce, _check_wire,
               _require_world, _total_weight, agree, cohort, digest)

__all__ = ["FedAvg", "FedAvgM", "FedProx", "DPFedAvg"]


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


# -- client-level differential privacy ---------------------------------------
#
# Agent proposal, not confirmed by the owner (issue #20, SPEC S6.7). The shape,
# the noise construction and the threat model below are this round's choices
# and are recorded as such in docs/distributed/FEDERATED5.md.

#: The domain separator of the noise construction. Part of the contract: two
#: ranks, or a rank and a test, that hash a different prefix draw different
#: noise.
_DP_KEY = "torchnative.federated.DPFedAvg"


def _dp_gaussians(seed, round_index, name, count):
    """``count`` standard normal floats (Python ``float``, float64), from the seed.

    The construction, stated so that anyone can regenerate it with the
    standard library alone:

    * key = ``"torchnative.federated.DPFedAvg\\0<seed>\\0<round>\\0<name>\\0"``,
      UTF-8, with ``seed`` and ``round`` in decimal;
    * block ``i`` = SHA-256(key + ``i`` as 8 bytes big-endian), ``i = 0, 1, ...``;
    * each 32-byte block holds two 16-byte pairs; in a pair the first 8 bytes
      give ``u1 = ((a >> 11) + 1) * 2**-53`` in (0, 1] and the second
      ``u2 = (b >> 11) * 2**-53`` in [0, 1), both big-endian;
    * each pair gives ``sqrt(-2 ln u1) * cos(2 pi u2)`` and
      ``sqrt(-2 ln u1) * sin(2 pi u2)`` (Box-Muller), in that order.

    Keyed by parameter name and round, so the noise of one parameter does not
    depend on the order the table is walked in, and a round never reuses the
    previous round's draw.

    **What this is not.** Not a constant-time or floating-point-hardened
    sampler: Mironov (2012) shows that the low bits of a naively sampled
    floating-point Gaussian can leak, and nothing here mitigates that. And
    ``math.log``/``cos``/``sin`` come from the platform's libm, which is not
    required to round correctly, so two platforms can disagree in the last bit.
    :meth:`DPFedAvg.aggregate` digests the noise across the ranks and refuses
    a disagreement by name rather than installing two different models.
    """
    key = ("%s\x00%d\x00%d\x00%s\x00" % (_DP_KEY, int(seed), int(round_index), name)
           ).encode("utf-8")
    out = []
    counter = 0
    while len(out) < count:
        block = hashlib.sha256(key + counter.to_bytes(8, "big")).digest()
        counter += 1
        for i in (0, 16):
            a = int.from_bytes(block[i:i + 8], "big") >> 11
            b = int.from_bytes(block[i + 8:i + 16], "big") >> 11
            u1 = (a + 1) * 2.0 ** -53
            u2 = b * 2.0 ** -53
            r = math.sqrt(-2.0 * math.log(u1))
            theta = 2.0 * math.pi * u2
            out.append(r * math.cos(theta))
            out.append(r * math.sin(theta))
    return out[:count]


def _dp_exact_norm(table):
    """The L2 norm of the whole table, the same number on every platform.

    ``sqrt(fsum(x*x))`` in float64 over the elements' values. For float32,
    float16 and bfloat16 each square is exact (at most 2 * 24 significant
    bits, inside float64's 53) and ``math.fsum`` rounds the exact sum once, so
    the result does not depend on a summation order, a kernel, or the device.
    That is what lets a central recomputation compare the clip exactly.
    For float64 tables the squares round, deterministically.
    """
    return math.sqrt(math.fsum(x * x for t in table.values()
                               for x in t.detach().reshape(-1).tolist()))


def _dp_real(value, what):
    """``value`` as a float, refusing ``bool`` and anything that is not a real number."""
    import numbers

    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(
            "torchnative.federated.algorithms.DPFedAvg: %s=%r is %s, not a "
            "real number" % (what, value, type(value).__name__)
        )
    return float(value)


class DPFedAvg:
    """FedAvg with client-level differential privacy (the DP-FedAvg shape).

    **Agent proposal, not confirmed by the owner** (issue #20; SPEC S6.7;
    docs/distributed/FEDERATED5.md). McMahan, Ramage, Talwar and Zhang (2018),
    "Learning differentially private recurrent language models", with full
    participation or a :func:`torchnative.federated.cohort`:

        clip(d_k)  = d_k * min(1, C / ||d_k||)      (one norm over every federated parameter)
        aggregate  = (sum_k clip(d_k) + N(0, (sigma*C)^2 I)) / n

    where ``n`` is the number of participating ranks. The noise is added
    **once, to the sum**, and every rank adds the **same** noise, so every rank
    installs the same model.

        fdp = FederatedDataParallel(model, algorithm=DPFedAvg(
            clip_norm=1.0, noise_multiplier=1.1))
        ...
        fdp.synchronize()          # no weight=: see "Weights" below

    **Why an algorithm and not a ``FederatedDataParallel`` flag.** The clip and
    the noise sit *inside* the aggregation, between the ``all_reduce`` of the
    sum and the division by the divisor, which is the algorithm's code; the
    weight contract (below) is the algorithm's ``resolve_weight``; and as an
    algorithm it serves the low-level road too
    (``Delta.publish(aggregator=DPFedAvg(...))``), as ``FedAvgM`` and
    ``FedProx`` do. ``FederatedDataParallel(differential_privacy=...)``
    refuses by name and points here.

    **Weights.** Every participating rank weighs exactly 1.0, and a ``weight=``
    is refused. The guarantee rests on the L2 sensitivity of the sum: with
    weight 1 per client, adding or removing one client moves the sum by at
    most ``C``, and the noise is calibrated to that. A sample-count weight
    ``w_k`` makes the sensitivity ``max_k w_k * C``, which is unbounded and
    data-dependent, and the sum of the weights (the divisor) would itself be
    a private quantity on the wire with no noise. DP-FedAvg's bounded-weight
    variant (``w_k = min(n_k / W, 1)``) is not built; ``weighted=True``
    refuses by name. Under *replace-one* adjacency the sensitivity is ``2C``,
    so the same ``sigma`` buys half the noise-to-sensitivity ratio; this class
    states the add/remove reading and does not compute either.

    **The seed and the threat model.** ``seed=None`` (the default): rank 0
    draws a 56-bit seed with :mod:`secrets` every round and shares it with the
    group in one ``int64`` ``all_gather``. ``seed=<int>``: the noise of round
    ``r`` is a function of ``(seed, r)``, reproducible, for tests and replays.
    Either way the seed and the round index are agreed by digest before the
    noise is drawn, and the noise itself is digested and agreed before it is
    added, so ranks that would install different models refuse instead.

    Every participating rank therefore **knows the noise**, and can subtract
    it to recover the exact noiseless sum; and without secure aggregation the
    hub of the transport sees every clipped delta anyway. So this is
    **central DP** with a trusted set of participants: it protects a client's
    contribution against an observer of the *released model* (whoever later
    receives the weights, or a server that only sees the installed model), and
    nothing else. It is not local DP and not distributed DP: it does not
    protect a client from its peers or from the aggregator. That threat model
    is ``threat_model='local'``, and it refuses by name. A fixed ``seed=`` is
    weaker still: anyone who knows the integer can subtract the noise from the
    released model, so it belongs in tests, not in a release.

    **No accountant.** No epsilon is computed or reported. A guarantee is a
    number at a stated granularity, and this class only produces the mechanism
    (clip, Gaussian noise, the noise multiplier); turning ``sigma``, the number
    of rounds and the cohort sampling into an ``(epsilon, delta)`` is an
    accountant's job and none is built. Client-level only: there is no
    per-example clipping (this stack's backward produces one gradient per
    parameter over the batch), so this says nothing about one *example* that
    it does not say about the client holding it.

    Per round, :meth:`round_report` (merged into
    ``FederatedDataParallel._get_federated_logging_data()``) gives
    ``clip_norm``, ``noise_multiplier``, ``noise_std``, ``clipped`` (whether
    *this rank's* delta was scaled), ``pre_clip_norm`` (its exact norm before
    the clip), ``noise_seed`` (``'fixed'``, ``'drawn'`` or None when sigma is
    0; never the seed itself), ``dp_round``, ``threat_model`` and
    ``accountant`` (None: there is none).

    **Rounding.** The clipped delta is ``d * float32(C / ||d||)``, one
    correctly rounded multiply per element, so its norm can exceed ``C`` by a
    few float32 ulp (relative ~1e-7). Not corrected.

    ``noise_multiplier=0`` is clipping without privacy, and ``clip_norm=inf``
    with it is ``FedAvg(weighted=False)``, bit for bit.
    """

    THREAT_MODELS = ("central",)

    def __init__(self, *, clip_norm, noise_multiplier, seed=None,
                 weighted=False, threat_model="central"):
        c = _dp_real(clip_norm, "clip_norm")
        if c != c or not c > 0.0:
            raise ValueError(
                "torchnative.federated.algorithms.DPFedAvg: clip_norm=%r. The "
                "clip norm C is the sensitivity the noise is calibrated to, so "
                "it has to be a positive number: at 0 every delta is clipped "
                "to nothing and the round installs pure noise" % (clip_norm,)
            )
        s = _dp_real(noise_multiplier, "noise_multiplier")
        if s != s or s in (float("inf"), float("-inf")) or s < 0.0:
            raise ValueError(
                "torchnative.federated.algorithms.DPFedAvg: noise_multiplier=%r. "
                "sigma has to be finite and at least 0 (0 is clipping without "
                "privacy)" % (noise_multiplier,)
            )
        if c == float("inf") and s > 0.0:
            raise ValueError(
                "torchnative.federated.algorithms.DPFedAvg: clip_norm=inf with "
                "noise_multiplier=%r. The noise std is sigma * C, so this is "
                "infinite noise; an unclipped round has an unbounded "
                "sensitivity and no noise can be calibrated to it. "
                "clip_norm=inf is accepted only with noise_multiplier=0, where "
                "it is FedAvg(weighted=False)" % (noise_multiplier,)
            )
        if weighted:
            raise NotImplementedError(
                "torchnative.federated.algorithms.DPFedAvg(weighted=True) is not "
                "implemented. The noise is calibrated to the L2 sensitivity of "
                "the sum, which is C when every client weighs 1. A sample-count "
                "weight w_k makes it max_k w_k * C, unbounded and "
                "data-dependent, and the sum of the weights would be a private "
                "number sent with no noise. DP-FedAvg's bounded weights "
                "(w_k = min(n_k / W, 1)) are not built.\n"
                "Check: DPFedAvg(..., weighted=False) and synchronize() with "
                "no weight="
            )
        if threat_model == "local":
            raise NotImplementedError(
                "torchnative.federated.algorithms.DPFedAvg: threat_model='local' "
                "is not implemented. The noise is drawn from a seed every "
                "participating rank knows, so any peer can subtract the noise "
                "and recover the exact sum, and without secure aggregation the "
                "transport's hub sees every clipped delta. Protection against "
                "peers needs per-client noise (local DP) or noise shares under "
                "secure aggregation (distributed DP); secure aggregation by "
                "pairwise masking over the full-mesh point-to-point links "
                "(issue #219) is the planned route, pending a leader decision.\n"
                "threat_model='central' is what this class serves: protection "
                "of a client's contribution against an observer of the "
                "released model."
            )
        if threat_model not in self.THREAT_MODELS:
            raise ValueError(
                "torchnative.federated.algorithms.DPFedAvg: threat_model=%r. "
                "One of %s ('local' is named and refused)"
                % (threat_model, list(self.THREAT_MODELS))
            )
        if seed is not None:
            if isinstance(seed, bool) or not isinstance(seed, int):
                raise TypeError(
                    "torchnative.federated.algorithms.DPFedAvg: seed=%r is %s. "
                    "It is an int every rank passes identically, or None to "
                    "have rank 0 draw one per round" % (seed, type(seed).__name__)
                )
            if not 0 <= seed < 2 ** 63:
                raise ValueError(
                    "torchnative.federated.algorithms.DPFedAvg: seed=%r. It has "
                    "to be in [0, 2**63)" % (seed,)
                )
        self.clip_norm = c
        self.noise_multiplier = s
        self.seed = seed
        self.threat_model = threat_model
        self.rounds_seen = 0
        self._report = {}

    @property
    def weighted(self):
        return False

    @property
    def noise_std(self):
        """``sigma * C``: the standard deviation of the noise added to the sum."""
        return self.noise_multiplier * self.clip_norm if self.noise_multiplier > 0.0 else 0.0

    def __repr__(self):
        return "DPFedAvg(clip_norm=%r, noise_multiplier=%r, seed=%s)" % (
            self.clip_norm, self.noise_multiplier,
            "drawn" if self.seed is None else "fixed")

    def resolve_weight(self, weight):
        """1.0 for a participant, 0.0 for ``ABSTAIN``, and a refusal for a number."""
        if weight is ABSTAIN:
            return 0.0
        if weight is not None:
            raise TypeError(
                "torchnative.federated.algorithms.DPFedAvg: weight=%r. Every "
                "participant weighs 1.0, because the noise is calibrated to the "
                "sensitivity of the sum, and a sample-count weight w_k raises "
                "it to w_k * C (and puts the private sample count on the wire "
                "without noise). Call synchronize() / aggregate() with no "
                "weight=" % (weight,)
            )
        return 1.0

    def round_report(self):
        """What the last :meth:`aggregate` did, as a plain dict (a copy). Empty before it."""
        return dict(self._report)

    def _seed_digest(self, seed, round_index):
        h = hashlib.sha256(("%s\x00seed\x00%d\x00%d" % (_DP_KEY, seed, round_index))
                           .encode("utf-8"))
        return int.from_bytes(h.digest()[:7], "big")

    def _round_seed(self, group, rank, world, round_index):
        """The seed of this round, the same on every rank, or a refusal."""
        if self.seed is not None:
            seed = self.seed
        else:
            import secrets

            mine = secrets.randbits(56) if rank == 0 else -1
            holders = [torch.full((1,), -1, dtype=torch.int64) for _ in range(world)]
            _all_gather(holders, torch.tensor([mine], dtype=torch.int64), group,
                        "DPFedAvg: the noise seed rank 0 drew for round %d"
                        % (round_index,))
            seed = mine if rank == 0 else int(holders[0][0].item())
            if seed < 0:
                # Only reachable under `tolerate_missing`: rank 0 left. A
                # RankDropped, so FederatedDataParallel undoes the round.
                raise RankDropped(
                    "torchnative.federated.algorithms.DPFedAvg: rank 0 did not "
                    "report the noise seed for round %d, so there is no noise "
                    "every rank agrees on, and a round without it would install "
                    "different models or no privacy" % (round_index,),
                    rank=rank, missing=(0,),
                    during="DPFedAvg: the noise seed for round %d" % (round_index,))
        # Also catches two ranks at different round indices, whose noise would
        # differ although their seeds agree.
        agree(self._seed_digest(seed, round_index), group,
              "the DPFedAvg noise seed and round (rank %d is at round %d)"
              % (rank, round_index))
        return seed

    def _noise(self, name, like, seed, round_index):
        """The noise for parameter ``name`` this round: ``sigma*C * N(0, 1)``, shaped like ``like``."""
        std = self.noise_std
        zs = _dp_gaussians(seed, round_index, name, like.numel())
        return torch.tensor([z * std for z in zs], dtype=like.dtype).reshape(like.shape)

    def aggregate(self, table, weight=None, group=None):
        """Clip this rank's delta, sum over the group, add the agreed noise, divide.

        Returns a table. Every refusal ``FedAvg.aggregate`` makes is made here
        too, in the same order, and a world of one is refused for the same
        reason.
        """
        world, rank = _require_world(group, "DPFedAvg.aggregate")
        if not table:
            raise ValueError(
                "torchnative.federated.algorithms.DPFedAvg: an empty table. A "
                "round that contributes no parameters would complete and "
                "change nothing"
            )
        w = self.resolve_weight(weight)
        for name, t in table.items():
            if not isinstance(t, torch.Tensor):
                raise TypeError(
                    "torchnative.federated.algorithms.DPFedAvg: %r is %r, not a "
                    "Tensor" % (name, type(t).__name__)
                )
            if not t.dtype.is_floating_point:
                raise NotImplementedError(
                    "torchnative.federated.algorithms.DPFedAvg: %r is %s. A "
                    "delta holds floating parameters" % (name, t.dtype)
                )

        # The clip is this rank's alone and needs no collective: one norm over
        # every federated parameter, as DP-FedAvg defines it. Clipping each
        # tensor separately would bound each part by C and the whole by
        # C * sqrt(#tensors), which is not the sensitivity the noise assumes.
        pre_clip = _dp_exact_norm(table)
        clipped = pre_clip > self.clip_norm

        agree(digest(table, values=False), group,
              "which parameters this round covers")

        round_index = self.rounds_seen
        noise = {}
        if self.noise_multiplier > 0.0:
            seed = self._round_seed(group, rank, world, round_index)
            for name in sorted(table):
                noise[name] = self._noise(name, table[name], seed, round_index)
            # libm is not required to round correctly, so two platforms can
            # draw different last bits from one seed. Agreed before it is
            # added: two ranks installing different models is the failure.
            agree(digest(noise), group,
                  "the DPFedAvg noise for round %d (drawn from the agreed seed; "
                  "a difference here is the platforms' libm disagreeing)"
                  % (round_index,))

        total = _total_weight(w, group)
        if not total > 0.0:
            raise ValueError(
                "torchnative.federated.algorithms.DPFedAvg: the ranks' weights "
                "sum to %r, so there is no average: every rank abstained"
                % (total,)
            )

        out = {}
        for name in sorted(table):
            t = table[name].detach()
            if clipped:
                t = t * torch.tensor(self.clip_norm / pre_clip, dtype=t.dtype)
            # The same spelling FedAvg uses, so that sigma = 0 and C = inf is
            # FedAvg(weighted=False) bit for bit.
            scaled = (t * torch.tensor(w, dtype=t.dtype)).clone()
            _check_wire(scaled, name)
            _all_reduce(scaled, group,
                        "the sum of the ranks' clipped deltas for %r" % (name,))
            if noise:
                scaled = scaled + noise[name]
            out[name] = scaled / torch.tensor(total, dtype=t.dtype)

        self._report = {
            "clip_norm": self.clip_norm,
            "noise_multiplier": self.noise_multiplier,
            "noise_std": self.noise_std,
            "clipped": bool(clipped),
            "pre_clip_norm": pre_clip,
            "noise_seed": (None if not noise
                           else "drawn" if self.seed is None else "fixed"),
            "dp_round": round_index,
            "threat_model": self.threat_model,
            "accountant": None,
        }
        self.rounds_seen += 1
        return out
