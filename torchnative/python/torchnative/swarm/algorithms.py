"""Swarm (gossip) algorithms: ``GossipAvg``.

``torch.swarm.algorithms``. The names are an **agent proposal, not
confirmed** (issue #291; the owner confirmed the package name ``torch.swarm``
and the wrapper name ``SwarmDataParallel`` on 2026-11-08, nothing below them).

Why a separate module, as :mod:`torchnative.federated.algorithms` is: the
wrapper :class:`torchnative.swarm.SwarmDataParallel` owns *who talks to whom*
(the topology) and the exchange over point-to-point; an algorithm owns *what a
rank does with what its neighbours sent* (the mixing rule and the fold). Those
vary independently. Gossip averaging is the first; push-sum (directed graphs,
weighted averages), compressed gossip (CHOCO) and gradient tracking are the
known next ones, and each changes the rule without changing the exchange.

Unlike ``FedAvg.aggregate``, :meth:`GossipAvg.aggregate` is **not** a
collective. In gossip there is nothing every rank agrees on after a round
(each rank holds its own mix), so the communication is the wrapper's, done
once per neighbour, and the arithmetic here is a pure function of what arrived.
That is what lets a test compute the same round centrally.
"""

from __future__ import annotations

import math

import torch

__all__ = ["GossipAvg", "metropolis_hastings"]


def _refuse_value(msg):
    raise ValueError("torchnative.swarm.algorithms.GossipAvg: " + msg)


def metropolis_hastings(neighbours):
    """The Metropolis-Hastings mixing matrix of an undirected graph.

    ``neighbours[i]`` is the sorted tuple of rank ``i``'s neighbours. For an
    edge ``(i, j)``: ``W[i][j] = 1 / (1 + max(d_i, d_j))``; off the graph
    ``W[i][j] = 0``; and ``W[i][i] = 1 - fsum(row i off the diagonal)``.

    Symmetric by construction (the edge weight reads both degrees the same
    way), so with rows summing to one it is doubly stochastic, which is what
    makes repeated rounds preserve the global mean and approach it. The self
    weight is at least ``1 / (1 + d_i) > 0``, so on a connected graph the
    chain is aperiodic and every eigenvalue but the one at 1 has modulus below
    1 (Xiao and Boyd 2004 use the same weights for the same reason: they need
    nothing but each rank's and its neighbour's degree).

    Returns a tuple of tuples of Python floats.
    """
    world = len(neighbours)
    degree = [len(n) for n in neighbours]
    rows = []
    for i in range(world):
        row = [0.0] * world
        for j in neighbours[i]:
            row[j] = 1.0 / (1.0 + max(degree[i], degree[j]))
        row[i] = 1.0 - math.fsum(row[j] for j in neighbours[i])
        rows.append(tuple(row))
    return tuple(rows)


class GossipAvg:
    """Gossip averaging: ``x_i <- sum_j W[i][j] x_j`` over ``{i} + neighbours(i)``.

    One synchronous round of decentralised averaging (the averaging step of
    D-PSGD, Lian et al. 2017): every rank replaces its parameters with a convex
    combination of its own and its neighbours'. ``W`` is the mixing matrix.

    ``mixing="metropolis_hastings"`` (the default) derives ``W`` from the
    topology (:func:`metropolis_hastings`). ``mixing=<matrix>`` takes an
    explicit ``world x world`` matrix, which is **validated, not trusted**:
    symmetric, non-negative, rows summing to one (so doubly stochastic), a
    positive self weight on every rank, and an off-diagonal support equal to
    the topology's edge set. Anything else refuses by name, because each of
    those failures produces a number rather than an error: a row that does not
    sum to one scales the model, an asymmetric matrix moves the global mean, a
    weight on a pair that is not an edge needs a message nobody sends.

    **Unweighted only.** A doubly stochastic mix approaches the *uniform* mean
    of the ranks. A sample-count weighted mean needs ratio consensus (push-sum:
    mix ``w_i x_i`` and ``w_i`` with the same matrix and keep the mass across
    rounds), which is not built; so ``weight=`` refuses by name rather than
    being accepted and ignored.

    **The fold is fixed**: the terms ``float64(x_j) * W[i][j]`` are summed in
    ascending rank order over ``{i} + neighbours(i)``, in float64, and the sum
    is rounded once to the parameters' dtype. A fixed order is what makes the
    same round computed elsewhere (centrally, or on upstream torch with gloo)
    equal bit for bit rather than within a tolerance.
    """

    MIXING = ("metropolis_hastings",)

    def __init__(self, mixing="metropolis_hastings"):
        if isinstance(mixing, str):
            if mixing not in self.MIXING:
                _refuse_value(
                    "mixing=%r is not a known rule. Known: %s, or an explicit "
                    "world x world matrix" % (mixing, list(self.MIXING)))
            self.mixing = mixing
            return
        try:
            rows = tuple(tuple(float(v) for v in row) for row in mixing)
        except TypeError:
            raise TypeError(
                "torchnative.swarm.algorithms.GossipAvg: mixing=%r is neither "
                "a rule name nor a matrix (a sequence of rows of numbers)"
                % (mixing,)) from None
        self.mixing = rows

    def __repr__(self):
        if isinstance(self.mixing, str):
            return "GossipAvg(mixing=%r)" % (self.mixing,)
        return "GossipAvg(mixing=<%dx%d matrix>)" % (
            len(self.mixing), len(self.mixing))

    def resolve_weight(self, weight):
        """The weight this rank mixes with, or a refusal. Always 1.0."""
        if weight is not None:
            raise TypeError(
                "torchnative.swarm.algorithms.GossipAvg: weight=%r. Gossip "
                "with a doubly stochastic mixing matrix approaches the uniform "
                "mean of the ranks, so a per-rank weight would be accepted and "
                "ignored. A sample-count weighted mean without a server needs "
                "ratio consensus (push-sum: mix w_i * x_i and w_i with the "
                "same matrix and carry the mass across rounds), which is not "
                "built.\nCheck: synchronize() with no weight." % (weight,))
        return 1.0

    def mixing_matrix(self, neighbours):
        """``W`` for this topology, validated. A tuple of tuples of floats."""
        world = len(neighbours)
        if isinstance(self.mixing, str):
            return metropolis_hastings(neighbours)
        rows = self.mixing
        if len(rows) != world or any(len(r) != world for r in rows):
            _refuse_value(
                "the mixing matrix is %dx%s and the world is %d. It needs one "
                "row and one column per rank"
                % (len(rows), sorted({len(r) for r in rows}), world))
        for i in range(world):
            for j in range(world):
                v = rows[i][j]
                if not math.isfinite(v) or v < 0.0:
                    _refuse_value(
                        "W[%d][%d] = %r. Every weight has to be finite and "
                        "non-negative; a negative weight makes the mix a "
                        "non-convex combination that can grow without bound"
                        % (i, j, v))
                if rows[i][j] != rows[j][i]:
                    _refuse_value(
                        "W[%d][%d] = %r and W[%d][%d] = %r. The matrix has to "
                        "be symmetric: an asymmetric row-stochastic mix moves "
                        "the global mean every round, so repeated rounds "
                        "would approach some other point and report success"
                        % (i, j, rows[i][j], j, i, rows[j][i]))
        # The bound is derived, not chosen: each entry is a decimal the caller
        # typed, carried as the nearest double, so it is off by at most half an
        # ulp of a number <= 1 (2**-53), and `fsum` adds no rounding of its
        # own beyond the final one. A row of `world` entries is therefore
        # within world * 2**-53 of what the caller meant.
        bound = world * 2.0 ** -53
        for i in range(world):
            total = math.fsum(rows[i])
            if abs(total - 1.0) > bound:
                _refuse_value(
                    "row %d sums to %r, not 1 (within %r, the rounding of "
                    "%d typed entries). A row that does not sum to one scales "
                    "rank %d's model every round" % (i, total, bound, world, i))
            if not rows[i][i] > 0.0:
                _refuse_value(
                    "W[%d][%d] = %r. A zero self weight is refused: on a "
                    "bipartite graph it makes the mix periodic (two ranks "
                    "swapping forever, eigenvalue -1) and the rounds never "
                    "approach the mean. A positive diagonal on a connected "
                    "graph rules that out without an eigen solver"
                    % (i, i, rows[i][i]))
            for j in range(world):
                if j == i:
                    continue
                edge = j in neighbours[i]
                if edge and rows[i][j] == 0.0:
                    _refuse_value(
                        "W[%d][%d] = 0 but %d and %d are neighbours in the "
                        "topology. The message would be sent and ignored; "
                        "drop the edge from the topology instead"
                        % (i, j, i, j))
                if not edge and rows[i][j] != 0.0:
                    _refuse_value(
                        "W[%d][%d] = %r but %d and %d are not neighbours in "
                        "the topology, so nothing would carry rank %d's "
                        "parameters to rank %d. Add the edge, or zero the "
                        "weight" % (i, j, rows[i][j], i, j, j, i))
        return rows

    def aggregate(self, rank, own, received, row):
        """One rank's mix. Not a collective: a pure function of what arrived.

        ``own`` is this rank's flat parameters, ``received`` is
        ``{neighbour: flat tensor}``, ``row`` is ``W[rank]``. Returns the
        float64 sum, in ascending rank order (see the class docstring).
        """
        acc = None
        for j in sorted([rank] + list(received)):
            x = own if j == rank else received[j]
            term = x.to(torch.float64) * row[j]
            acc = term if acc is None else acc + term
        return acc
