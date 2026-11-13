"""Federated aggregation algorithms: ``FedAvg``, ``FedAvgM``, ``FedProx``.

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
from . import (ABSTAIN, _all_reduce, _check_wire, _require_world, _route,
               _total_weight, agree, cohort, digest)

__all__ = ["FedAvg", "FedAvgM", "FedProx"]


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

        route = _route()
        if route is not None:
            # aggregation='server' (BACKEND.md D14): the same scaled tables go
            # to the server in one request, the server folds them and the
            # weights in ascending rank order (the hub's order on the
            # all_reduce path) and every rank divides exactly as below.
            scaled = {name: (table[name].detach()
                             * torch.tensor(w, dtype=table[name].dtype)).clone()
                      for name in sorted(table)}
            sums, total = route.fold(
                scaled, w, "the server's fold of the ranks' weighted deltas")
            self._refuse_empty_total(total)
            return {name: sums[name] / torch.tensor(total, dtype=table[name].dtype)
                    for name in sorted(table)}

        total = _total_weight(w, group)
        self._refuse_empty_total(total)

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

    @staticmethod
    def _refuse_empty_total(total):
        if not total > 0.0:
            raise ValueError(
                "torchnative.federated.algorithms.FedAvg: the ranks' weights sum to "
                "%r, so there is no average. Every rank that reported either "
                "abstained (federated.ABSTAIN) or weighed nothing -- a round "
                "over an empty cohort would otherwise divide by zero and "
                "return a table of the right names full of nan" % (total,)
            )


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
