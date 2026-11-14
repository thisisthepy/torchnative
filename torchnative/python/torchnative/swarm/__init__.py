"""``torch.swarm``: peer-to-peer learning without a server (swarm learning).

Owner decision, 2026-11-08 (docs/INTENT.md section 4): "연합학습과 군집학습
부분은 확정하고 문서 개정하고, 리드미 수정하고, 지금 리펙토링은 바로 들어가도록
해." The package ``torch.swarm`` and the wrapper ``SwarmDataParallel`` are the
owner's names; "the same structure as ``torch.federated``" is the owner's
shape. **Everything below those names is an agent proposal, not confirmed**
(issue #291): the synchronous gossip round, the default ring, the
Metropolis-Hastings weights, the agreement check, the tags, the argument names
and :mod:`torchnative.swarm.algorithms`. The design is
docs/distributed/SWARM.md.

Used the way :class:`torchnative.federated.FederatedDataParallel` is used:
wrap the module, train locally with any torch optimizer in your own loop, and
call ``synchronize`` once per round.

    from torchnative import swarm
    from torchnative.swarm import algorithms

    sdp = swarm.SwarmDataParallel(model, algorithm=algorithms.GossipAvg(),
                                  topology="ring")
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    for batch in local_batches:
        optimizer.zero_grad()
        loss_fn(sdp(batch)).backward()
        optimizer.step()
    sdp.synchronize()
    info = sdp._get_swarm_logging_data()

**What a round is.** Each rank sends its parameters to its neighbours, and
only to them, with ``torch.distributed.isend``; receives theirs with
``irecv``; and replaces its own with ``sum_j W[i][j] x_j`` over itself and its
neighbours (:class:`torchnative.swarm.algorithms.GossipAvg`). There is no
``all_reduce``, no hub and no server: the federated package's aggregation is a
collective every rank ends agreeing on, and this is the opposite, a local mix
after which ranks still differ, approaching the global mean over rounds at the
rate of the mixing matrix's second-largest eigenvalue modulus.

**Before the parameters, an agreement check, also point to point.** Each rank
sends its neighbours three int64s (a digest of the mixing matrix, a digest of
the parameter schema, and the round number) and refuses if any neighbour's
differ. On a connected graph, every adjacent pair agreeing implies every pair
agrees, so this is an exact check with no collective. Without it, two ranks
built with different topologies or models would mix incomparable vectors and
return a number.

**What is refused by name**, rather than approximated:

======================================  ====================================
a world of one                          refused: a rank with no neighbours
                                        mixes with nobody and the round is
                                        the identity, which no test can tell
                                        from a broken mix
a disconnected topology                 refused: each component approaches
                                        its own mean and the swarm never
                                        agrees
an asymmetric (directed) neighbour      refused: needs push-sum, not built
list, a self loop, a rank out of
range, a duplicate
a mixing matrix that is not             refused (see ``GossipAvg``)
symmetric, doubly stochastic, with a
positive diagonal and the topology's
support
``weight=``                             refused: a weighted mean without a
                                        server needs push-sum
a neighbour that does not arrive        :class:`NeighbourDropped`, naming it;
                                        the round is not applied and the
                                        parameters are left as they were
mixed dtypes, integer parameters,       refused by name
parameters off the CPU
asynchronous gossip, time-varying       not offered: no argument here takes
topologies, compression                 them
secure aggregation, differential        ``secure_aggregation=`` and
privacy                                 ``differential_privacy=`` refuse by
                                        name, as in ``torch.federated``
======================================  ====================================

Grade: *builds* (AGENTS.md section 16). Nothing here has run; the first run is
CI's, of ``tests/distributed/test_swarm.py``.
"""

from __future__ import annotations

import hashlib
import struct

import torch

from .algorithms import GossipAvg

__all__ = ["SwarmDataParallel", "NeighbourDropped", "TOPOLOGIES",
           "resolve_topology", "components"]

#: The point-to-point tags of the two messages of a round. An agent proposal;
#: user traffic on these tags in the same group would interleave with them.
CHECK_TAG = 0x5357_01
PARAMS_TAG = 0x5357_02


def _dist():
    """``torch.distributed``, with the ``torchnative`` backend registered."""
    import torch.distributed as dist

    import torchnative.distributed  # noqa: F401  (registers backend="torchnative")

    return dist


def _group(group, who):
    dist = _dist()
    if not dist.is_initialized():
        raise RuntimeError(
            "torchnative.swarm: %s has no process group, so there are no "
            "neighbours to exchange with.\n"
            "Check: import torchnative.distributed; "
            "torch.distributed.init_process_group(backend='torchnative', "
            "init_method='tcp://127.0.0.1:<port>', rank=<r>, "
            "world_size=<n >= 2>)" % (who,))
    return dist.get_world_size(group), dist.get_rank(group)


def _require_world(group, who):
    world, rank = _group(group, who)
    if world < 2:
        raise NotImplementedError(
            "torchnative.swarm: a world of one, reached through %s. A rank "
            "alone has no neighbours, so a gossip round would mix its "
            "parameters with nothing and return them unchanged: the identity, "
            "which a test at this size cannot tell from a mix that ignores "
            "its neighbours, drops one, or uses the wrong weights. The "
            "degenerate case is named instead of served.\n"
            "Check: init_process_group(..., world_size=2) or more." % (who,))
    return world, rank


class NeighbourDropped(RuntimeError):
    """A neighbour did not deliver its message for this round.

    Raised *instead of* a mix, never alongside one: the parameters are left
    exactly as they were before ``synchronize``. Mixing with the neighbours
    that did arrive would need a different row of the matrix (the missing
    weight has to go somewhere), and any choice there is a matrix nobody
    validated. ``rank`` is this rank, ``missing`` the neighbour, ``during``
    the message that noticed.
    """

    def __init__(self, message, rank=None, missing=(), during=None):
        super().__init__(message)
        self.rank = rank
        self.missing = tuple(missing)
        self.during = during


def _ring(world):
    if world == 2:
        return ((1,), (0,))
    return tuple(tuple(sorted({(r - 1) % world, (r + 1) % world}))
                 for r in range(world))


def _complete(world):
    return tuple(tuple(j for j in range(world) if j != r) for r in range(world))


#: Named topologies. ``ring`` is the default: degree two, so each rank sends
#: two messages a round whatever the world size.
TOPOLOGIES = {"ring": _ring, "complete": _complete}


def components(neighbours):
    """The connected components of an undirected graph, each a sorted list."""
    seen = set()
    out = []
    for start in range(len(neighbours)):
        if start in seen:
            continue
        stack, comp = [start], []
        seen.add(start)
        while stack:
            r = stack.pop()
            comp.append(r)
            for n in neighbours[r]:
                if n not in seen:
                    seen.add(n)
                    stack.append(n)
        out.append(sorted(comp))
    return out


def resolve_topology(topology, world):
    """``neighbours``: a tuple, per rank, of its sorted neighbour ranks.

    ``topology`` is a name in :data:`TOPOLOGIES` or one neighbour list per
    rank (a sequence of ``world`` sequences, or a mapping ``rank -> ranks``).
    A user list is validated: one entry per rank, no self loop, no rank out of
    range, no duplicate, symmetric (``j`` in ``i``'s list iff ``i`` in
    ``j``'s), and connected. Each failure refuses by name.
    """
    who = "torchnative.swarm.resolve_topology"
    if isinstance(topology, str):
        if topology not in TOPOLOGIES:
            raise ValueError(
                "%s: topology=%r is not a known topology. Known: %s, or one "
                "neighbour list per rank" % (who, topology, sorted(TOPOLOGIES)))
        neighbours = TOPOLOGIES[topology](world)
    else:
        if isinstance(topology, dict):
            if sorted(topology) != list(range(world)):
                raise ValueError(
                    "%s: the neighbour mapping has keys %s; a world of %d "
                    "needs exactly the ranks %s"
                    % (who, sorted(topology), world, list(range(world))))
            lists = [topology[r] for r in range(world)]
        else:
            try:
                lists = list(topology)
            except TypeError:
                raise TypeError(
                    "%s: topology=%r is neither a name nor one neighbour list "
                    "per rank" % (who, topology)) from None
        if len(lists) != world:
            raise ValueError(
                "%s: %d neighbour lists for a world of %d. Every rank needs "
                "one, even a rank that is only listed by others"
                % (who, len(lists), world))
        out = []
        for r, entry in enumerate(lists):
            try:
                ranks = [int(v) for v in entry]
            except (TypeError, ValueError):
                raise TypeError(
                    "%s: the neighbour list of rank %d is %r, not a sequence "
                    "of ranks" % (who, r, entry)) from None
            if len(set(ranks)) != len(ranks):
                raise ValueError(
                    "%s: rank %d lists a neighbour twice (%r). One edge is one "
                    "message; a duplicate would be a second weight on the "
                    "same pair" % (who, r, ranks))
            for n in ranks:
                if n == r:
                    raise ValueError(
                        "%s: rank %d lists itself. The self weight is the "
                        "diagonal of the mixing matrix, not an edge"
                        % (who, r))
                if not 0 <= n < world:
                    raise ValueError(
                        "%s: rank %d lists rank %d, which is not in a world "
                        "of %d, so that neighbour cannot arrive"
                        % (who, r, n, world))
            out.append(tuple(sorted(ranks)))
        for r, ranks in enumerate(out):
            for n in ranks:
                if r not in out[n]:
                    raise ValueError(
                        "%s: rank %d lists rank %d, and rank %d does not list "
                        "rank %d. A directed edge needs push-sum (a "
                        "column-stochastic mix with a carried mass), which is "
                        "not built; symmetric gossip needs both ends to "
                        "exchange" % (who, r, n, n, r))
        neighbours = tuple(out)
    comps = components(neighbours)
    if len(comps) > 1:
        raise ValueError(
            "torchnative.swarm: the topology is disconnected, components %s. "
            "Gossip on a disconnected graph approaches one mean per "
            "component and the swarm never agrees, while each round still "
            "reports success. Add an edge between the components."
            % (comps,))
    return neighbours


def _digest(text):
    """A signed 64-bit digest of ``text``. ``hashlib``: compared across processes."""
    raw = hashlib.blake2b(text.encode("utf-8"), digest_size=8).digest()
    return struct.unpack("<q", raw)[0]


def _is_gone(exc):
    """Whether an exception from a point-to-point wait means the peer is gone."""
    if isinstance(exc, (OSError, EOFError)):
        return True
    text = str(exc).lower()
    return ("connection closed" in text or "did not complete within" in text
            or "timed out" in text or "connection reset" in text)


class SwarmDataParallel(torch.nn.Module):
    """Wrap a module for gossip averaging with its neighbours, without a server.

        sdp = swarm.SwarmDataParallel(model, topology="ring")
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
        for batch in local_batches:           # your own loop, as with DDP
            optimizer.zero_grad()
            loss_fn(sdp(batch)).backward()
            optimizer.step()
        sdp.synchronize()

    ``forward`` calls the wrapped module (``.module``). This class owns the
    round, not the training: it does not own the optimizer and does not count
    steps.

    ``topology=`` is ``"ring"`` (default), ``"complete"``, or one neighbour
    list per rank (:func:`resolve_topology`). It is resolved at the first
    ``synchronize``, when the world size is known. ``algorithm=`` defaults to
    :class:`~torchnative.swarm.algorithms.GossipAvg` with Metropolis-Hastings
    weights. ``parameters=`` names the swarm parameters (default: every
    parameter with ``requires_grad``); they must share one floating dtype and
    live on the CPU.

    **``synchronize(weight=None)`` is one gossip round** and returns None, as
    DDP's methods do. Afterwards :meth:`_get_swarm_logging_data` holds:
    ``rank``, ``world``, ``round`` (0 for the first), ``neighbours``,
    ``mixing_row`` (``{rank: weight}`` over itself and its neighbours),
    ``local_norm`` (how far local training moved the parameters since the
    last round), ``mix_norm`` (how far the mix moved them), ``sent_to`` and
    ``received_from``.

    **A round with no local training is not refused**, unlike
    ``FederatedDataParallel``. There, a zero delta is counted at full weight
    in an average every rank installs. Here a rank's contribution *is* its
    parameters, so repeated rounds with no training in between are consensus
    averaging, a meaningful operation, and the one the convergence test runs.

    **Optimizer state** is the caller's and carries across rounds; the mix
    changes the parameters under it, as DDP's averaging does not (named here
    rather than silently decided).
    """

    def __init__(self, module, algorithm=None, topology="ring", group=None,
                 parameters=None, secure_aggregation=False,
                 differential_privacy=None):
        super().__init__()
        if secure_aggregation:
            raise NotImplementedError(
                "torchnative.swarm.SwarmDataParallel: secure_aggregation= is "
                "not implemented. In gossip every neighbour receives this "
                "rank's parameters in the clear, which is the opposite of "
                "secure aggregation's promise; hiding them needs pairwise "
                "masks that cancel in the mix, a key agreement per edge, and "
                "a plan for a neighbour that drops after the masks are "
                "drawn. None of it exists.")
        if differential_privacy is not None:
            raise NotImplementedError(
                "torchnative.swarm.SwarmDataParallel: differential_privacy= "
                "is not implemented. A guarantee is an (epsilon, delta) at a "
                "stated granularity, and producing one needs per-example "
                "clipping inside the local step, calibrated noise and an "
                "accountant over the rounds and the graph. Adding noise "
                "without those would make a worse model and an absent "
                "guarantee, both reported as success.")
        self.module = module
        self.algorithm = algorithm if algorithm is not None else GossipAvg()
        self.topology = topology
        self.group = group
        named = dict(module.named_parameters())
        if parameters is None:
            names = [n for n, p in module.named_parameters() if p.requires_grad]
        else:
            names = list(parameters)
            unknown = [n for n in names if n not in named]
            if unknown:
                raise ValueError(
                    "torchnative.swarm.SwarmDataParallel: parameters=%r names "
                    "%s, which the module does not have" % (names, unknown))
            if len(set(names)) != len(names):
                raise ValueError(
                    "torchnative.swarm.SwarmDataParallel: parameters=%r names "
                    "one parameter twice" % (names,))
        if not names:
            raise ValueError(
                "torchnative.swarm.SwarmDataParallel: no parameters to mix. A "
                "round over nothing would complete and change nothing")
        self._names = tuple(names)
        self._plan_cache = None
        self._round = 0
        self._swarm_logging_data = {}
        self._snapshot = self._flatten("SwarmDataParallel()")[0].to(
            torch.float64)

    def __repr__(self):
        return "<SwarmDataParallel algorithm=%r topology=%r over %d parameters>" % (
            self.algorithm, self.topology, len(self._names))

    def forward(self, *args, **kwargs):
        """The wrapped module's forward, unchanged (as ``DistributedDataParallel``)."""
        return self.module(*args, **kwargs)

    @property
    def covers(self):
        """The swarm parameter names, in the order they are flattened."""
        return self._names

    def _params(self):
        named = dict(self.module.named_parameters())
        return [named[n] for n in self._names]

    def _flatten(self, who):
        """``(flat, dtype)``: the swarm parameters as one contiguous 1-d tensor."""
        params = self._params()
        dtypes = sorted({str(p.dtype) for p in params})
        if len(dtypes) != 1:
            raise NotImplementedError(
                "torchnative.swarm: %s over parameters of %d dtypes %s. A "
                "round sends one message per neighbour, which carries one "
                "dtype; one message per dtype is not built" % (
                    who, len(dtypes), dtypes))
        dtype = params[0].dtype
        if not dtype.is_floating_point:
            raise NotImplementedError(
                "torchnative.swarm: %s over %s parameters. A mix is a convex "
                "combination; rounding every one of them to an integer is not "
                "what any caller means" % (who, dtype))
        devices = sorted({str(p.device) for p in params})
        if devices != ["cpu"]:
            raise NotImplementedError(
                "torchnative.swarm: %s over parameters on %s. Staging device "
                "parameters through host memory, with counted uploads and "
                "downloads, is not built for swarm rounds" % (who, devices))
        flat = torch.cat([p.detach().reshape(-1) for p in params])
        return flat, dtype

    def _plan(self, world):
        """``(neighbours, W)`` for ``world``, resolved and validated once."""
        if self._plan_cache is None or self._plan_cache[0] != world:
            neighbours = resolve_topology(self.topology, world)
            matrix = self.algorithm.mixing_matrix(neighbours)
            self._plan_cache = (world, neighbours, matrix)
        return self._plan_cache[1], self._plan_cache[2]

    def neighbours(self):
        """This rank's neighbours (needs the process group)."""
        world, rank = _require_world(self.group, "SwarmDataParallel.neighbours")
        return self._plan(world)[0][rank]

    def mixing_matrix(self):
        """The whole mixing matrix ``W`` (needs the process group)."""
        world, _ = _require_world(self.group, "SwarmDataParallel.mixing_matrix")
        return self._plan(world)[1]

    def _global(self, peer):
        if self.group is None:
            return peer
        return _dist().get_global_rank(self.group, peer)

    def _exchange(self, tensor, peers, tag, what, rank, fill):
        """Send ``tensor`` to every peer and receive one from each.

        ``isend`` to all, then ``irecv`` from all, then wait: no ordering by
        rank is needed, because neither backend's ``isend`` waits for the
        receiver. A slot is pre-filled with ``fill`` so a buffer the transport
        never wrote is visible rather than read as a value.
        """
        dist = _dist()
        sends = [(p, dist.isend(tensor, dst=self._global(p), group=self.group,
                                tag=tag)) for p in peers]
        bufs = {p: torch.full(tensor.shape, fill, dtype=tensor.dtype)
                for p in peers}
        recvs = [(p, dist.irecv(bufs[p], src=self._global(p), group=self.group,
                                tag=tag)) for p in peers]
        for p, work in recvs + sends:
            try:
                work.wait()
            except Exception as exc:  # noqa: BLE001  (translated or re-raised)
                if not _is_gone(exc):
                    raise
                raise NeighbourDropped(
                    "torchnative.swarm: neighbour %d did not deliver %s to "
                    "rank %d, so this round has no mix and the parameters are "
                    "left as they were. Mixing with the neighbours that did "
                    "arrive would need a different row of the mixing matrix "
                    "(the missing weight has to go somewhere), which nobody "
                    "validated.\nCause: %s: %s"
                    % (p, what, rank, type(exc).__name__, exc),
                    rank=rank, missing=(p,), during=what) from exc
        return bufs

    def synchronize(self, weight=None):
        """One gossip round. Returns None; diagnostics in :meth:`_get_swarm_logging_data`."""
        world, rank = _require_world(self.group, "SwarmDataParallel.synchronize")
        self.algorithm.resolve_weight(weight)
        neighbours, matrix = self._plan(world)
        peers = neighbours[rank]
        flat, dtype = self._flatten("SwarmDataParallel.synchronize")
        params = self._params()
        own64 = flat.to(torch.float64)
        moved = own64 - self._snapshot
        local_norm = float((moved * moved).sum().item()) ** 0.5

        topo_digest = _digest(repr((neighbours, tuple(
            tuple(v.hex() for v in row) for row in matrix))))
        schema_digest = _digest(repr([(n, tuple(p.shape)) for n, p in
                                      zip(self._names, params)] + [str(dtype)]))
        mine = (topo_digest, schema_digest, self._round)
        check = torch.tensor(list(mine), dtype=torch.int64)
        got = self._exchange(check, peers, CHECK_TAG, "the agreement check",
                             rank, -1)
        for p in peers:
            theirs = tuple(int(v) for v in got[p].tolist())
            if theirs != mine:
                what = [name for name, a, b in zip(
                    ("the topology and mixing matrix", "the parameter schema",
                     "the round number"), mine, theirs) if a != b]
                raise ValueError(
                    "torchnative.swarm: rank %d and its neighbour %d disagree "
                    "about %s (rank %d holds %r, rank %d holds %r). Mixing "
                    "would combine incomparable vectors and return a number; "
                    "nothing has been sent but this check, and the parameters "
                    "are unchanged." % (rank, p, " and ".join(what), rank,
                                        mine, p, theirs))

        received = self._exchange(flat, peers, PARAMS_TAG,
                                  "its parameters", rank, float("nan"))
        row = matrix[rank]
        mixed = self.algorithm.aggregate(rank, flat, received, row)

        offset = 0
        with torch.no_grad():
            for p in params:
                n = p.numel()
                p.data.copy_(mixed[offset:offset + n].reshape(p.shape).to(dtype))
                offset += n
        after = self._flatten("SwarmDataParallel.synchronize")[0].to(torch.float64)
        shift = after - own64
        self._swarm_logging_data = {
            "rank": rank, "world": world, "round": self._round,
            "neighbours": tuple(peers),
            "mixing_row": {j: row[j] for j in sorted([rank] + list(peers))},
            "local_norm": local_norm,
            "mix_norm": float((shift * shift).sum().item()) ** 0.5,
            "sent_to": tuple(peers), "received_from": tuple(sorted(received)),
        }
        self._round += 1
        self._snapshot = after

    def _get_swarm_logging_data(self):
        """What the last :meth:`synchronize` did, as a plain dict (a copy).

        Empty before the first ``synchronize``, like DDP's
        ``_get_ddp_logging_data``.
        """
        return dict(self._swarm_logging_data)
