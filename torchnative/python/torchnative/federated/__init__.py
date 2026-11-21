"""``torch.federated``: federated learning, a paradigm package beside ``torch.distributed``.

Owner decision, 2026-11-08: "연합학습과 군집학습 부분은 확정하고 문서 개정하고,
리드미 수정하고, 지금 리펙토링은 바로 들어가도록 해." The package holds the
communication primitives built on ``torch.distributed`` (this module), the
algorithms (:mod:`torchnative.federated.algorithms`: ``FedAvg``, ``FedAvgM``,
``FedProx``, ``DPFedAvg``) and the module wrapper :class:`FederatedDataParallel`, named the way
``DistributedDataParallel`` is: the paradigm first. ``FederatedDataParallel`` is
horizontal federated learning (samples split, every participant holds the whole
model). :class:`FederatedFeatureParallel` (vertical federated learning,
features split; :mod:`torchnative.federated.feature_parallel`, issue #293) is
built and graded *builds* until CI runs ``tests/distributed/test_vfl.py``;
``torch.swarm`` (``SwarmDataParallel``, peer to peer without a server) is
:mod:`torchnative.swarm` (issue #291).

Used the way ``DistributedDataParallel`` is used: wrap the module, train locally
with any torch optimizer in your own loop, and call ``synchronize`` when the
local round is done.

    from torchnative import federated
    from torchnative.federated import algorithms

    fdp = federated.FederatedDataParallel(model, algorithm=algorithms.FedAvg())
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    for batch in local_batches:
        optimizer.zero_grad()
        loss_fn(fdp(batch)).backward()
        optimizer.step()
    fdp.synchronize(weight=n_local_samples)
    info = fdp._get_federated_logging_data()   # per-round diagnostics, as DDP has _get_ddp_logging_data

**What this is built on, and why it is one round.** DESIGN.md §11.1 stacks
rounds/selection/aggregation above ``torch.distributed``, and aggregation *is*
collective communication -- so ``FedAvg`` is a weighted ``all_reduce`` and
not a private channel. The transport under it (docs/distributed/TRANSPORT.md) implements
``world_size`` 2 with ``ReduceOp.SUM``; everything past that refuses by name,
and so does everything here that would need it.

**The refusal that matters most is at one rank.** ``FedAvg`` over a world of
one is the identity function: it returns the delta it was handed, and no test
at that size can distinguish a correct weighted average from an aggregator that
ignores its weights, drops a peer, or does nothing at all. So a world of one is
refused rather than served -- see :meth:`torchnative.federated.algorithms.FedAvg.aggregate`. This is the one
place in this package where "it works at world_size 1" would be a lie told by a
green test rather than by a sentence.

**Two premises are checked rather than assumed**, because both fail silently:

``the ranks cover the same parameters``
    An ``all_reduce`` sums element-wise whatever it is handed. Two ranks whose
    deltas cover different names, in a different order, or at different shapes
    would produce a number rather than an error. :func:`agree` refuses first.

``the ranks started from the same weights``
    FedAvg averages *offsets from a common base*. If the bases differ the
    average is over incomparable quantities, and again nothing raises.
    :meth:`torchnative.delta.Delta.publish` refuses first.

Both are answered with one ``int64`` ``all_reduce`` of a digest, which costs a
collective of one element and is exact -- ``h0 + h1 == 2 * h0`` if and only if
``h1 == h0``, which is why the check is written for a world of exactly two and
says so.

**What is not here**, named rather than approximated:

======================================  ====================================
more than one round                     built -- call ``synchronize`` once per
                                        round; it re-snapshots the base
                                        (docs/distributed/FEDERATED2.md)
participant selection                   *half* built. :func:`cohort` agrees
                                        the participant set across the ranks,
                                        which nothing downstream would notice
                                        going wrong; a **proper subset**
                                        refuses, because it needs a sub-group
                                        and a world larger than two
a rank that does not arrive             ``on_missing='refuse'`` is
                                        implemented and is the default: the
                                        collective raises
                                        :class:`RankDropped`, naming the rank,
                                        and the round is *undone* rather than
                                        left half-applied.
                                        ``on_missing='average_arrived'``
                                        refuses
aggregators other than FedAvg           ``algorithms.FedAvgM`` (server
                                        momentum) and ``algorithms.FedProx``
                                        (a proximal term on the *local*
                                        objective) are built.
                                        FedAdam, SCAFFOLD: none
secure aggregation                      ``algorithms.SecAgg`` (pairwise
                                        masking, issue #253): the hub sees
                                        only masked integers. No dropout
                                        recovery; ``secure_aggregation=``
                                        refuses and points at it
client-level differential privacy       built (agent proposal, not
                                        confirmed): ``algorithms.DPFedAvg``,
                                        clip + one agreed Gaussian draw on
                                        the sum, central DP only, no
                                        accountant (no epsilon).
                                        ``differential_privacy=`` refuses and
                                        points there
                                        (docs/distributed/FEDERATED5.md)
aggregation on a server                 built for FedAvg, FedAvgM and
                                        FedProx: ``FederatedDataParallel(
                                        aggregation='server')`` on a group
                                        initialised through ``tnserver://``
                                        folds the round on the coordinator
                                        (BACKEND.md D14,
                                        docs/distributed/FEDSERVER.md); the
                                        default stays ``'all_reduce'``
compression                             not offered: no surface here takes
                                        a key or a codec
vertical federated learning             built, graded *builds*:
                                        ``FederatedFeatureParallel``
                                        (docs/distributed/VFL.md)
peer to peer without a server           :mod:`torchnative.swarm`
                                        (``SwarmDataParallel``, gossip
                                        over point-to-point; issue #291)
======================================  ====================================

docs/distributed/FEDERATED.md records the measurements and what each refusal was weighed
against.
"""

from __future__ import annotations

import contextlib
import hashlib
import threading

import torch

__all__ = ["FederatedDataParallel", "FederatedFeatureParallel",
           "SamplesMisaligned", "digest", "agree", "cohort",
           "RankDropped", "ServerAggregationError", "ABSTAIN",
           "tolerate_missing"]


# The wire is JSON over a socket (docs/distributed/TRANSPORT.md §2). This guard existed
# because both ranks used to `sendall` before either received, so a payload
# larger than the kernel's socket buffer blocked both of them until the 30 s
# timeout: 2,836,968 B completed in 0.17 s and 4,053,011 B failed after 30 s,
# and the wall was not even a constant -- the same 8.1 MB went through in
# 0.23 s if smaller collectives had run first and grown the buffers.
#
# **That transport defect is fixed**: `ProcessGroupLocal.allreduce` now orders
# the exchange by rank, so one side is always draining. 3,000,000 floats
# complete in 0.60 s where 4 MB used to hang. The guard is kept, far above the
# old wall, as a *refusal* rather than a limit -- there is no longer a known
# size that fails, and a number here that nothing measured would be a claim
# this file cannot support. It is the size past which nobody has run one.
# 15 MB: what 3,000,000 `ones` floats actually encode to, which is the
# largest payload measured through the fixed transport (0.60 s). Not a
# rounded-up guess -- raise it with another measurement.
_WIRE_SAFE_BYTES = 15_000_000
_WIRE_COLD_OK_BYTES = 2_836_968
_WIRE_COLD_DEADLOCK_BYTES = 4_053_011

# An upper bound on the JSON length of one float element including its ", "
# separator: `repr` of a float64 holding a float32 value is at most 24
# characters ("-1.1754943508222875e-38"). Used so the common case costs no
# serialisation -- the exact length is measured only when this bound is passed.
_MAX_ELEMENT_BYTES = 26


def _dist():
    """``torch.distributed``, with the ``torchnative`` backend registered.

    Imported here rather than at module import so that reading
    ``torchnative.federated`` does not register a backend as a side effect.
    ``torchnative.distributed`` is what registers it, and its ``register()`` is
    idempotent by construction.
    """
    import torch.distributed as dist

    import torchnative.distributed  # noqa: F401  -- registers backend="torchnative"

    return dist


def _group(group):
    """``(world_size, rank)`` for ``group``, refusing an uninitialised world.

    A caller who never called ``init_process_group`` gets the check that would
    have made this work, rather than ``Default process group has not been
    initialized`` from four frames down.
    """
    dist = _dist()
    if not dist.is_initialized():
        raise RuntimeError(
            "torchnative.federated: there is no process group, so there is "
            "nobody to aggregate with.\n"
            "Check: import torchnative.distributed; "
            "torch.distributed.init_process_group(backend='torchnative', "
            "init_method='tcp://127.0.0.1:<port>', rank=<0 or 1>, "
            "world_size=2)  -- docs/distributed/TRANSPORT.md"
        )
    return dist.get_world_size(group), dist.get_rank(group)


def _require_world(group, who, minimum=2):
    """``(world_size, rank)``, refusing a world too small to prove anything.

    One place, because the *interesting* refusal -- a world of one -- has to
    read the same wherever a caller hits it. It was written twice first, and
    the second copy (in ``agree``) shadowed the first: at ``world_size = 1``
    ``Delta.publish`` reached the base-agreement check before the aggregator
    and reported "written for a world of exactly two", which is true and is
    not the point.

    ``minimum`` is 2 for aggregation itself and larger for the things that
    two ranks cannot show: a **proper subset** cohort and a **survivor set**
    of two or more both need three (docs/distributed/FEDERATED4.md), because at two ranks
    each of them is a world of one and FedAvg over one delta is that delta.
    """
    world, rank = _group(group)
    if world == 1:
        raise NotImplementedError(
            "torchnative.federated: a world of one, reached through %s. "
            "Averaging a single client's delta with itself is the identity "
            "function, so this would return what it was handed and prove "
            "nothing about the aggregation -- a test at this size passes "
            "whether the weights are honoured, ignored, or never read. "
            "FedAvg is not defined here as 'the average of one'; the "
            "degenerate case is named instead of served.\n"
            "Check: torch.distributed.get_world_size() >= 2, reached through "
            "init_process_group(backend='torchnative', init_method='tcp://...', "
            "world_size=2) -- docs/distributed/TRANSPORT.md" % (who,)
        )
    if world < minimum:
        raise NotImplementedError(
            "torchnative.federated: world_size %d, reached through %s, "
            "which needs at least %d. The transport carries any world "
            "(docs/distributed/FEDERATED4.md), but %d ranks are not enough to show what "
            "this does: whatever it selects or whatever survives is a world "
            "of one, and FedAvg over one delta is that delta -- so a test of "
            "it would pass with no aggregation at all.\n"
            "Check: init_process_group(..., world_size=%d)"
            % (world, who, minimum, world, minimum)
        )
    return world, rank


#: The spelling ``Delta.publish`` calls. It no longer means "exactly two" --
#: it means "at least two", which is what it always checked for.
_require_two = _require_world


class _Arrival:
    """Who reported, across the collectives of one round.

    ``FederatedDataParallel(on_missing='average_arrived')`` divides by the weights of the
    ranks that arrived, and that is only a number if **every** collective in
    the round agreed about which ranks those were. The survivor set is the
    hub's verdict and travels with the data
    (``ProcessGroupLocal.allreduce_partial``), so the survivors hold the same
    one -- but a rank that leaves *between* two collectives of the same round
    changes it, and a divisor that changed halfway is not the divisor of any
    average. So it is recorded once and compared after that.
    """

    def __init__(self, world):
        self.world = world
        self.missing = None

    @property
    def survivors(self):
        if self.missing is None:
            return tuple(range(self.world))
        return tuple(r for r in range(self.world) if r not in self.missing)

    def record(self, missing, what):
        missing = tuple(sorted(int(r) for r in missing))
        if self.missing is None:
            self.missing = missing
            return
        if missing != self.missing:
            raise RankDropped(
                "torchnative.federated: the survivor set changed inside "
                "one round -- %s were missing, and %s are missing during %s. "
                "A weighted mean whose divisor changed halfway is not the "
                "mean of anything, so this refuses rather than reporting the "
                "last one."
                % (list(self.missing), list(missing), what),
                missing=missing, during=what,
            )


_ARRIVAL = threading.local()


@contextlib.contextmanager
def tolerate_missing(group=None):
    """Let the collectives inside complete over whoever arrived.

    Yields an :class:`_Arrival`. Outside this block every collective in this
    module is all-or-nothing and a lost rank raises :class:`RankDropped`;
    inside it, the collectives use the shim's ``*_partial`` spellings, which
    return the hub's survivor set alongside the data.

    This is a *mechanism*. The policy -- and the minimum number of survivors
    the caller is willing to divide by -- is ``FederatedDataParallel(on_missing=...,
    min_participants=...)``, because "however many arrived" is exactly the
    divisor nobody chose that :class:`RankDropped` exists to refuse.
    """
    world, _ = _group(group)
    previous = getattr(_ARRIVAL, "current", None)
    arrival = _Arrival(world)
    _ARRIVAL.current = arrival
    try:
        yield arrival
    finally:
        _ARRIVAL.current = previous


def _arrival():
    return getattr(_ARRIVAL, "current", None)


# -- server-side aggregation (BACKEND.md D14, issue #282) --------------------
#
# Agent proposal, not confirmed (docs/distributed/FEDSERVER.md): the option
# name `aggregation='server'`, the route below, and the server's `fed_gather`
# and `fed_fold` operations. The decision it serves is D14 (leader): "Server-
# style aggregation through the server interface (D4) is a follow-up built
# with this backend."


class ServerAggregationError(RuntimeError):
    """The server refused a round, or could not be reached. The message says which.

    Raised *instead of* an aggregate, like :class:`RankDropped`, and
    ``FederatedDataParallel`` undoes the round on it the same way.
    """


_ROUTE = threading.local()

#: How many server routes this process opened per group token. Every rank
#: opens them in the same program order, so the n-th route of a group names
#: the same server round on every rank.
_ROUTES_OPENED = {}


def _route():
    return getattr(_ROUTE, "current", None)


class _ServerRoute:
    """One synchronize's worth of server operations for the calling rank.

    Built from the group's own store: the backend keeps the ``PrefixStore``
    ``init_process_group`` handed it, and under it, for a group initialised
    through ``tnserver://``, is the server's client (``ServerStore``), which
    knows the server's address. A group whose store is anything else has no
    server to aggregate on, and that is refused by name rather than served by
    the participants' own ``all_reduce``.

    Keys are ``tn_fed/<group token>/route<n>/op<k>``: the group token is the
    one the mesh's rank 0 published at rendezvous (unique per group, so two
    groups on one server never share a round), ``n`` counts the routes this
    process opened for the group and ``k`` the operations inside this one.
    """

    def __init__(self, group, who):
        world, rank = _group(group)
        dist = _dist()
        from torchnative.distributed.backend import server as _server
        from torchnative.distributed.backend import transport as _transport

        pg = group if group is not None else dist.distributed_c10d._get_default_group()
        sole = getattr(pg, "_sole_backend", None)
        backend = sole(who) if sole is not None else None
        store = getattr(backend, "_store", None)
        inner = store
        while inner is not None and hasattr(inner, "underlying_store"):
            inner = inner.underlying_store
        if store is None or not isinstance(inner, _server._ServerStoreMethods):
            raise NotImplementedError(
                "torchnative.federated: %s with aggregation='server', but this "
                "group's store is %s, not a torchnative server's. Server-side "
                "aggregation runs on the coordinator the group rendezvoused "
                "through, and only a group initialised through tnserver:// has "
                "one; the participants' own all_reduce is a different "
                "topology (BACKEND.md D14), so it is not substituted.\n"
                "Check: init_process_group(backend='torchnative', "
                "init_method='tnserver://host:port') (or "
                "'?server=external' for a StandaloneServer), or "
                "aggregation='all_reduce'."
                % (who, type(inner).__name__ if inner is not None else None))
        self.host, self.port = inner.host, inner.port
        self.timeout_s = float(inner._timeout_s)
        token = store.get(_transport._KEY_GROUP)
        token = token.decode("utf-8") if isinstance(token, bytes) else str(token)
        n = _ROUTES_OPENED.get(token, 0)
        _ROUTES_OPENED[token] = n + 1
        self.prefix = "tn_fed/%s/route%d" % (token, n)
        self.rank, self.world = rank, world
        self.ops = 0
        #: What the server answered for each fold of this route, for
        #: ``_get_federated_logging_data()["server_receipts"]``.
        self.receipts = []

    def _key(self):
        key = "%s/op%d" % (self.prefix, self.ops)
        self.ops += 1
        return key

    def _ask(self, req, frames, what):
        from torchnative.distributed.backend import server as _server

        try:
            reply, out = _server.fed_request(self.host, self.port, req, frames,
                                             wait_s=self.timeout_s)
        except (OSError, ConnectionError, ValueError, RuntimeError) as exc:
            raise ServerAggregationError(
                "torchnative.federated: rank %d lost the server at %s:%d "
                "during %s (%s: %s), so this round has no aggregate"
                % (self.rank, self.host, self.port, what, type(exc).__name__,
                   exc)) from exc
        if reply.get("ok"):
            return reply, out
        if reply.get("kind") == "missing":
            missing = tuple(int(r) for r in reply.get("missing", ()))
            raise RankDropped(
                "torchnative.federated: rank(s) %s did not reach the server "
                "at %s:%d during %s within the group's timeout of %.1f s, so "
                "this round has no aggregate. The server aggregates only when "
                "every rank of the round has handed in, and it computed no "
                "partial average: dividing by whoever arrived would divide by "
                "a number nobody chose and report success "
                "(docs/design/DESIGN.md §6). "
                "on_missing='average_arrived' with aggregation='server' is "
                "not built (docs/distributed/FEDSERVER.md)."
                % (",".join(str(r) for r in missing) or "?", self.host,
                   self.port, what, float(reply.get("seconds", self.timeout_s))),
                rank=self.rank, missing=missing, during=what)
        raise ServerAggregationError(
            "torchnative.federated: the server at %s:%d refused %s: %s"
            % (self.host, self.port, what, reply.get("msg")))

    def gather_int(self, value, what):
        """Every rank's ``value``, in rank order, through the server."""
        reply, _ = self._ask({"op": "fed_gather", "key": self._key(),
                              "rank": self.rank, "world": self.world,
                              "timeout": self.timeout_s, "value": int(value)},
                             (), what)
        return [int(v) for v in reply["values"]]

    def fold(self, table, weight, what):
        """``(sum of the tables, sum of the weights)`` over the round, folded on the server.

        ``table`` is ``{name: tensor}``, already scaled by the caller; the
        names go in sorted order. Both sums are the server's, in ascending rank
        order. An empty table folds only the weights.
        """
        from torchnative.distributed.backend import server as _server
        from torchnative.distributed.backend import wire

        names = sorted(table)
        frames = []
        size = 0
        for i, name in enumerate(names):
            t = table[name]
            dtype = wire.dtype_name(t)
            if dtype not in _server.FOLD_DTYPES:
                raise NotImplementedError(
                    "torchnative.federated: %r is torch.%s, and the server's "
                    "fold (aggregation='server') adds only %s. A float16 or "
                    "bfloat16 sum needs its own per-addition rounding to equal "
                    "the all_reduce path's, which the server does not have; "
                    "it is refused rather than summed in another width.\n"
                    "Check: federate float32 parameters, or "
                    "aggregation='all_reduce'."
                    % (name, dtype, ", ".join("torch." + d
                                              for d in _server.FOLD_DTYPES)))
            frame = wire.encode_tensor(t, i)
            size += len(frame)
            frames.append(frame)
        if size > _server._MAX_FOLD_BYTES:
            raise NotImplementedError(
                "torchnative.federated: one server fold of %d bytes, past the "
                "server's %d-byte limit for one request; nobody has run a "
                "round that large through it" % (size, _server._MAX_FOLD_BYTES))
        key = self._key()
        reply, out = self._ask({"op": "fed_fold", "key": key,
                                "rank": self.rank, "world": self.world,
                                "timeout": self.timeout_s,
                                "weight": float(weight).hex(), "names": names},
                               frames, what)
        if list(reply.get("names", ())) != names or len(out) != len(names):
            raise ServerAggregationError(
                "torchnative.federated: the server answered %r with %d tables "
                "named %s for the %d named %s"
                % (key, len(out), reply.get("names"), len(names), names))
        sums = {}
        for name, frame in zip(names, out):
            t = table[name]
            got = wire.frame_to_tensor(frame)
            if got.dtype != t.dtype or tuple(got.shape) != tuple(t.shape):
                raise ServerAggregationError(
                    "torchnative.federated: the server returned %r as %s %s "
                    "for a %s %s" % (name, got.dtype, tuple(got.shape),
                                     t.dtype, tuple(t.shape)))
            sums[name] = got if t.device.type == "cpu" else got.to(t.device)
        total = float.fromhex(reply["total_weight"])
        self.receipts.append({"key": key, "ranks": list(reply.get("ranks", ())),
                              "names": names, "sha256": reply.get("sha256"),
                              "total_weight": reply["total_weight"]})
        return sums, total


@contextlib.contextmanager
def _via_server(group=None, who="federated aggregation"):
    """Route this block's aggregation through the group's server. Yields the route.

    Inside it :func:`agree` gathers its digests through the server and
    ``FedAvg.aggregate`` / ``_total_weight`` fold on the server instead of
    running ``all_reduce`` among the participants. ``FederatedDataParallel(
    aggregation='server')`` opens one per ``synchronize``.
    """
    route = _ServerRoute(group, who)
    previous = _route()
    _ROUTE.current = route
    try:
        yield route
    finally:
        _ROUTE.current = previous


def _backend(group, what):
    """The shim's backend object for ``group``, or a refusal naming why.

    ``allreduce_partial`` has no upstream spelling: upstream's allreduce either
    completes over the world or is an error. So this reaches past
    ``torch.distributed``'s API to the backend the shim registered, and says so
    when that backend is not there.
    """
    dist = _dist()
    pg = group if group is not None else dist.distributed_c10d._get_default_group()
    sole = getattr(pg, "_sole_backend", None)
    if sole is None:
        raise NotImplementedError(
            "torchnative.federated: %r is not this shim's process group, "
            "and %s has no upstream spelling -- a collective that completes "
            "over whoever arrived is not an allreduce, so no backend but this "
            "one offers it (docs/distributed/FEDERATED4.md)" % (pg, what)
        )
    return sole(what)


def _sum_opts():
    dist = _dist()

    class _Opts:
        pass

    opts = _Opts()
    opts.reduceOp = dist.ReduceOp.SUM
    return opts


def _all_reduce(tensor, group, what):
    """``all_reduce(SUM)``, tolerant or not depending on the context."""
    dist = _dist()
    arrival = _arrival()
    if arrival is None:
        return _collective(
            lambda: dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=group),
            group, what)

    def run():
        work, missing = _backend(group, "allreduce_partial").allreduce_partial(
            [tensor], _sum_opts())
        arrival.record(missing, what)
        return work

    return _collective(run, group, what)


def _all_gather(holders, tensor, group, what):
    """``all_gather``, tolerant or not depending on the context."""
    dist = _dist()
    arrival = _arrival()
    if arrival is None:
        return _collective(
            lambda: dist.all_gather(holders, tensor, group=group), group, what)

    def run():
        work, missing = _backend(group, "allgather_partial").allgather_partial(
            [holders], [tensor])
        arrival.record(missing, what)
        return work

    return _collective(run, group, what)


class RankDropped(RuntimeError):
    """A rank that was selected for this round did not report.

    Raised *instead of* an aggregate, never alongside one.  The failure this
    names is the one docs/design/DESIGN.md §6 puts below a refusal: an aggregator that
    divides by however many ranks arrived returns a number, and the round
    reports success while the model it produced is a weighted mean over a
    cohort nobody chose.  ``sum(w_k d_k) / sum_{arrived}(w_k)`` is not the
    quantity FedAvg names, and nothing downstream can tell the difference.

    ``rank`` is this process's, ``missing`` the peer(s) that did not report,
    and ``during`` the collective that noticed.
    """

    def __init__(self, message, rank=None, missing=(), during=None):
        super().__init__(message)
        self.rank = rank
        self.missing = tuple(missing)
        self.during = during


def _collective(fn, group, what):
    """Run one collective, turning a lost peer into :class:`RankDropped`.

    The transport is a socket (docs/distributed/TRANSPORT.md §2), so a rank that exits
    mid-round surfaces as ``RuntimeError('connection closed')`` from
    ``_recv_all``, as a ``socket.timeout`` after 30 s, or as ``BrokenPipeError``
    on the send.  All three mean the same thing to this layer and none of them
    says so: they name the socket rather than the round.

    Translating them is the *mechanism*; the policy is in
    :class:`FederatedDataParallel`'s ``on_missing=``.  What is guaranteed here is that no
    partial average exists to be returned -- the exception replaces the value
    rather than accompanying it, because ``all_reduce`` either completes over
    every rank or does not complete.
    """
    import struct

    dist = _dist()
    try:
        return fn()
    except RankDropped:
        raise
    except (OSError, EOFError, struct.error) as exc:
        pass
    except RuntimeError as exc:
        if "connection closed" not in str(exc):
            raise
    try:
        world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        missing = tuple(r for r in range(world) if r != rank)
    except Exception:  # the group itself is gone
        world, rank, missing = -1, -1, ()
    raise RankDropped(
        "torchnative.federated: rank %s did not report during %s, so this "
        "round has no aggregate. The collective over %s rank(s) did not "
        "complete, and no partial average was produced -- averaging over "
        "whichever ranks arrived would divide by a number nobody chose and "
        "would report success (docs/design/DESIGN.md §6).\n"
        "Policy: FederatedDataParallel(on_missing='refuse') is the default and is this. "
        "FederatedDataParallel(on_missing='average_arrived', min_participants=k) divides by "
        "the survivors instead, and needs k >= 2 and a world of at least "
        "three -- at two ranks the survivor set is one and FedAvg over one "
        "delta is that delta (docs/distributed/FEDERATED4.md)."
        % (",".join(str(r) for r in missing) or "?", what, world),
        rank=rank, missing=missing, during=what,
    )


def digest(table, values=True):
    """A deterministic integer over a ``{name: tensor}`` table.

    Over the *schema* -- names, shapes, dtypes, in sorted name order -- and,
    with ``values=True``, over the bytes as well.

    ``hashlib`` rather than ``hash()``: this number is compared across
    processes, and CPython salts ``hash`` of a string per interpreter, so the
    built-in would disagree between two ranks holding identical tables. That
    failure looks exactly like the disagreement this exists to detect.

    Truncated to 56 bits so ``value * world_size`` stays exact in the ``int64``
    tensor :func:`agree` reduces it in.

    The bytes come from ``Delta._bytes``, the same little-endian encoding
    ``Delta.persist`` writes -- so two ranks agree here exactly when a byte
    comparison of what they would persist agrees.
    """
    from torchnative.delta import Delta

    h = hashlib.sha256()
    for name in sorted(table):
        t = table[name]
        h.update(name.encode("utf-8"))
        h.update(b"\x00")
        h.update(repr(tuple(t.shape)).encode("utf-8"))
        h.update(b"\x00")
        h.update(str(t.dtype).encode("utf-8"))
        h.update(b"\x00")
        if values:
            h.update(Delta._bytes(t)[1])
        h.update(b"\x01")
    return int.from_bytes(h.digest()[:7], "big")


def agree(value, group=None, what="this value"):
    """Refuse unless every rank passed the same integer. Returns the sum.

    One ``int64`` ``all_gather``, and the comparison is **element by element
    against every rank's digest**, not against a total.

    It used to be one ``all_reduce(SUM)`` and the test ``total == value *
    world``, which is an equality test only because the world had exactly two
    members: ``h0 + h1 == 2 * h0`` iff ``h0 == h1``.  At three it accepts
    ``(h - 1, h, h + 1)`` -- three ranks that disagree about the schema, the
    base or the cohort, summing to exactly what agreement would have summed
    to.  docs/distributed/FEDERATED3.md §5 named that and refused the larger world rather
    than weaken here; ``ProcessGroupLocal.allgather`` at ``world_size N``
    (docs/distributed/FEDERATED4.md) is what settles it, so the refusal is gone and the
    check is stronger rather than wider.

    Inside :func:`tolerate_missing` the gather is the partial spelling and the
    ranks that did not report are not compared -- there is nothing of theirs
    to compare.
    """
    world, rank = _require_world(group, "federated.agree")
    route = _route()
    if route is not None:
        # aggregation='server': the digests go through the server too, so a
        # server round sends nothing over the participants' own sockets.
        gathered = route.gather_int(int(value), "federated.agree(%s)" % (what,))
        seen = {r: gathered[r] for r in range(world)}
    else:
        probe = torch.tensor([int(value)], dtype=torch.int64)
        # Filled with a value no digest can take, so a slot the transport
        # never wrote is visible as one rather than read as a zero that agrees.
        holders = [torch.full((1,), -1, dtype=torch.int64) for _ in range(world)]
        _all_gather(holders, probe, group,
                    "federated.agree(%s)" % (what,))
        arrival = _arrival()
        missing = set(arrival.missing or ()) if arrival is not None else set()
        seen = {r: int(holders[r][0].item())
                for r in range(world) if r not in missing}
    seen[rank] = int(value)
    odd = sorted(r for r, v in seen.items() if v != int(value))
    if odd:
        raise ValueError(
            "torchnative.federated: the ranks disagree about %s. Rank %d "
            "holds digest %d; the %d ranks that reported hold %s, and rank(s) "
            "%s differ from this one.\n"
            "Gathered rank by rank rather than summed: a sum-and-compare is "
            "an equality test only at two ranks, and at three it accepts "
            "(h-1, h, h+1). An all_reduce would have averaged them anyway and "
            "returned a number, which is why this is checked instead of "
            "assumed."
            % (what, rank, int(value), len(seen),
               [seen[r] for r in sorted(seen)], odd)
        )
    return int(value) * len(seen)


def _check_wire(tensor, name):
    """Refuse a tensor too large for the transport to carry, by name.

    See the ``_WIRE_*`` constants: this is a limit of ``ProcessGroupLocal``'s
    send-before-receive, not of aggregation, and the message says so because
    the fix belongs there.
    """
    if tensor.numel() * _MAX_ELEMENT_BYTES + 2 <= _WIRE_SAFE_BYTES:
        return
    import json

    size = len(json.dumps(tensor.tolist()).encode("utf-8"))
    if size <= _WIRE_SAFE_BYTES:
        return
    raise NotImplementedError(
        "torchnative.federated: %r is %d elements and %d bytes on the wire, "
        "and nobody has run a round that large. The bound is %d.\n"
        "This is a refusal, not a measured wall. There used to be one: both "
        "ranks called sendall before either received, so a payload past the "
        "socket buffer blocked both until the 30 s timeout -- %d B completed "
        "and %d B did not, and the wall moved with how much traffic came "
        "before it. ProcessGroupLocal.allreduce now orders the exchange by "
        "rank, and 3,000,000 floats complete in 0.60 s. Raise this bound with "
        "a measurement rather than a guess."
        % (name, tensor.numel(), size, _WIRE_SAFE_BYTES,
           _WIRE_COLD_OK_BYTES, _WIRE_COLD_DEADLOCK_BYTES)
    )


def _total_weight(weight, group=None):
    """``sum(weight)`` over the group, as a float. One 1-element collective.

    Under ``aggregation='server'`` the server folds the weights instead, in
    the same ascending rank order and in binary64, so the number is the same.
    """
    route = _route()
    if route is not None:
        _, total = route.fold({}, float(weight), "the sum of the ranks' weights")
        return total
    total = torch.tensor([float(weight)], dtype=torch.float64)
    _all_reduce(total, group, "the sum of the ranks' weights")
    return float(total[0].item())


class _Abstain:
    """The weight of a rank that is in the world but not in this round's cohort.

    Not ``0.0``, which every aggregator here refuses: a rank weighted zero
    contributes nothing while still counting as having participated, and that
    is a caller mistake worth refusing. This is the same arithmetic said on
    purpose -- ``ABSTAIN`` contributes a zero table at weight zero, so the
    divisor is the cohort's weight and the aggregate is the cohort's mean.

    It exists because a proper subset of the ranks still has to *attend* every
    collective: the transport is a star and a rank that skipped one would hang
    the ones that did not. Attending without contributing is what selection
    means here, and this is the value that says so.
    """

    __slots__ = ()

    def __repr__(self):
        return "federated.ABSTAIN"

    def __bool__(self):
        return False


#: See :class:`_Abstain`. Passed as ``weight=`` by :class:`FederatedDataParallel` for a rank
#: outside the round's cohort.
ABSTAIN = _Abstain()


def cohort(select, group=None, who="federated.cohort"):
    """The ranks this round runs over, agreed across the group. Returns a tuple.

    ``select`` is a sequence of ranks or a callable taking the world size and
    returning one.  Every rank evaluates it and the results are compared over
    the same digest collective the schema and base use, because **selection
    fails silently in exactly the way they do**: two ranks that disagree about
    who is participating still complete every collective and still produce a
    weighted mean, over a cohort neither of them chose.  A rule as ordinary as
    "sample half the clients at random" disagrees whenever the ranks seed
    differently, which is the default.

    **A proper subset is now served, and not with ``new_group``.**  Every rank
    still attends every collective -- the transport is a star and a rank that
    walked away would hang the ones that did not -- and the ranks outside the
    cohort contribute ``ABSTAIN``: a zero table at weight zero.  The divisor is
    then the cohort's weight and the result is the cohort's weighted mean,
    exactly.  What ``torch.distributed.new_group`` would add on top is that the
    unselected ranks are not *reached* at all, which is a property of the wire
    and not of the arithmetic; it is named in docs/distributed/FEDERATED4.md as still
    missing, because a client that is asleep cannot attend a barrier.

    A cohort of **one** still refuses, at any world size, and so does any
    proper subset in a world of two: FedAvg over one delta is that delta, and
    a test of it would pass with no aggregation at all.
    """
    world, rank = _require_world(group, who)
    proposal = select(world) if callable(select) else select
    try:
        ranks = tuple(sorted({int(r) for r in proposal}))
    except (TypeError, ValueError):
        raise TypeError(
            "torchnative.federated.cohort: select= produced %r, which is "
            "not a sequence of rank numbers" % (proposal,)
        )
    if not ranks:
        raise ValueError(
            "torchnative.federated.cohort: the empty cohort. A round with "
            "no participants would complete, aggregate nothing and report "
            "success"
        )
    bad = [r for r in ranks if not 0 <= r < world]
    if bad:
        raise ValueError(
            "torchnative.federated.cohort: rank(s) %s are not in a world of "
            "%d" % (bad, world)
        )

    # Agreed before it is acted on, and before the local epochs: two ranks
    # holding different cohorts is not detectable downstream.
    agree(digest({"cohort": torch.tensor(ranks, dtype=torch.float64)}),
          group,
          "which ranks this round selected (rank %d proposed %s)"
          % (rank, list(ranks)))

    # One refusal, not two. There was a second here -- "a proper subset needs
    # a world of at least three" -- and it was **unreachable**: in a world of
    # two the only proper subsets have one member, so the check above had
    # already refused every input that could reach it. A refusal that cannot
    # fire is not a refusal, so its reason was folded into the one that does
    # (docs/distributed/FEDERATED4.md §5).
    if len(ranks) < 2:
        raise NotImplementedError(
            "torchnative.federated.cohort: %s of a world of %d is a cohort "
            "of %d. FedAvg over one delta is that delta -- the identity this "
            "whole package refuses to serve -- so a round over it would "
            "return the selected rank's own weights and report success, and a "
            "test of it would pass with no aggregation at all.\n"
            "In a world of two this is the only shape a proper subset can "
            "have, which is why a proper subset needs a world of at least "
            "three. The transport carries one now (docs/distributed/FEDERATED4.md), so "
            "what refuses here is the arithmetic and no longer the wire: a "
            "subset of a world of three is served, and aggregates over the "
            "subset.\n"
            "Check: len(select(world)) >= 2, in a world of at least three if "
            "the cohort is not the whole world."
            % (list(ranks), world, len(ranks))
        )
    # A rank outside the cohort is neither an error nor excused from the
    # collectives: it attends and abstains. See `_Abstain`.
    return ranks


from . import algorithms  # noqa: E402  (after the primitives it imports)
from .algorithms import FedAvg, FedProx  # noqa: E402,F401  (FederatedDataParallel's default and its FedProx hook)


class FederatedDataParallel(torch.nn.Module):
    """Wrap a module for federated averaging, the way DDP wraps it for gradients.

        fdp = federated.FederatedDataParallel(model, algorithm=algorithms.FedAvg())
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)

        for batch in local_batches:           # your own loop, as with DDP
            optimizer.zero_grad()
            loss_fn(fdp(batch)).backward()
            optimizer.step()
        fdp.synchronize(weight=len(local_dataset))

    ``forward`` calls the wrapped module (``.module``), so the local half of a
    round is ordinary training with any torch optimizer: forward,
    ``loss.backward()``, ``optimizer.step()``. This class does not run it, does
    not own the optimizer and does not know what the loss is. It owns the
    *round*: the snapshot the local training started from, and what happens when
    the caller says the local training is done.

    ``parameters=`` names which parameters are federated (default: every
    parameter of the module with ``requires_grad``). Only those are in the
    delta, on the wire and in the average.

    **``synchronize(weight=None)`` is one federated round.** The local delta is
    the federated parameters now minus the base snapshot
    (:class:`torchnative.delta.Delta`), it is published through the algorithm,
    ``base + aggregate`` is installed, and the base is **re-snapshotted** to the
    installed weights, so that round *k+1* measures only the new local training
    and not the movement since round 1. Call it once per round; there is no
    ``rounds=``, the caller's loop is the round loop.

    **Optimizer state.** The optimizer is the caller's and is not reset between
    rounds: momentum and variance from the previous round carry into the next.
    Whether that matters depends on the optimizer and the task; it is named
    here rather than silently decided (DESIGN.md §6).

    **A round with no local training is refused.** If the local delta is
    exactly zero, ``synchronize`` raises before any collective: the round would
    contribute the zero delta and still be counted, at full weight, in the
    average. (A rank outside the ``select=`` cohort is the deliberate exception
    and contributes zero at weight zero.)

    **``synchronize`` returns None**, as DDP's methods do. What the round did is
    in a plain dict, :meth:`_get_federated_logging_data` (the name mirrors DDP's
    ``_get_ddp_logging_data``): ``rank``, ``world``, ``weight`` (this rank's),
    ``total_weight`` (the group's), ``share`` (``weight / total_weight``, exactly
    0.5 when two ranks were unweighted), ``covers``, ``local_norm``,
    ``aggregate_norm``, ``cohort`` (the ranks the round ran over),
    ``participated`` (False for a rank the cohort left out) and ``missing`` (the
    ranks that did not report, non-empty only under ``'average_arrived'``),
    and ``secure_aggregation`` (True under ``algorithms.SecAgg``, which adds
    its fixed-point scale and bits).

    **``steps`` is not reported.** The number of optimizer steps taken between
    two ``synchronize`` calls is the caller's loop's, not something a wrapper
    around ``forward`` can count honestly (a step is ``optimizer.step()``, which
    this class never sees), so the dict has no ``steps`` key.

    **``aggregation='server'``** (BACKEND.md D14; agent proposal, not
    confirmed) aggregates the round on the coordinator the group rendezvoused
    through instead of with an ``all_reduce`` among the participants: the
    group must come from ``init_process_group(init_method='tnserver://...')``.
    Each rank sends its weighted delta to the server as raw bytes, the server
    folds the deltas and the weights in ascending rank order (the order the
    ``all_reduce`` hub uses, so the aggregate is the same bits) and every rank
    receives the one result; the digest checks of the round go through the
    server too. The caller's loop does not change. FedAvg, FedAvgM and FedProx
    are served; any other algorithm, ``on_missing='average_arrived'``,
    ``secure_aggregation=`` and ``differential_privacy=`` refuse by name with
    it. docs/distributed/FEDSERVER.md.
    """

    #: The dropout policies this wrapper knows. **Both are implemented**:
    #: `'average_arrived'` is served with `min_participants=k` and refuses only
    #: below a world of three (docs/distributed/FEDERATED4.md section 6).
    ON_MISSING = ("refuse", "average_arrived")

    #: Where a round is aggregated. ``'all_reduce'`` (the default, and what
    #: this class always did) sums the weighted deltas with an ``all_reduce``
    #: among the participants. ``'server'`` sends them to the coordinator the
    #: group rendezvoused through (``tnserver://``), which folds them and
    #: sends one aggregate back (BACKEND.md D14; the option's name and shape
    #: are an agent proposal, not confirmed, docs/distributed/FEDSERVER.md).
    AGGREGATION = ("all_reduce", "server")

    def __init__(self, module, algorithm=None, group=None, parameters=None,
                 select=None, on_missing="refuse", min_participants=None,
                 secure_aggregation=False, differential_privacy=None,
                 aggregation="all_reduce"):
        super().__init__()
        if aggregation not in self.AGGREGATION:
            raise ValueError(
                "torchnative.federated.FederatedDataParallel: aggregation=%r. "
                "One of %s" % (aggregation, list(self.AGGREGATION)))
        if aggregation == "server":
            self._refuse_for_server(algorithm, on_missing, secure_aggregation,
                                    differential_privacy)
        # `select=` is evaluated in `synchronize`, not here: it needs the
        # world size, and the cohort has to be *agreed across the ranks*, which
        # is a collective. What it cannot do is a proper subset -- see
        # `cohort()`, which refuses that by name and says what it would take.
        if on_missing not in self.ON_MISSING:
            raise ValueError(
                "torchnative.federated.FederatedDataParallel: on_missing=%r. One of %s"
                % (on_missing, list(self.ON_MISSING))
            )
        # `min_participants` is what turns "however many arrived" into a
        # number the caller chose. Without it this policy is exactly the
        # defect `RankDropped` exists to refuse -- a weighted mean over a
        # cohort decided by a socket close, reported as success.
        if on_missing == "average_arrived":
            if min_participants is None:
                raise TypeError(
                    "torchnative.federated.FederatedDataParallel: "
                    "on_missing='average_arrived' requires "
                    "min_participants=k. Averaging over whoever arrived, with "
                    "no floor, makes the divisor a number nobody chose: the "
                    "round returns a weighted mean over a cohort decided by a "
                    "socket close and reports success (docs/design/DESIGN.md §6). "
                    "The floor is the caller saying how few ranks an aggregate "
                    "may still be built from.\n"
                    "Check: min_participants=2 or more."
                )
            if not isinstance(min_participants, int) or min_participants < 2:
                raise ValueError(
                    "torchnative.federated.FederatedDataParallel: min_participants=%r. It "
                    "has to be an integer of at least 2 -- a survivor set of "
                    "one is a world of one, where FedAvg returns the "
                    "survivor's own delta and a round over it would report "
                    "success having aggregated nothing (docs/distributed/FEDERATED3.md "
                    "§4.1 measured that identity to 6e-8)."
                    % (min_participants,)
                )
        elif min_participants is not None:
            raise TypeError(
                "torchnative.federated.FederatedDataParallel: min_participants=%r with "
                "on_missing=%r. A floor on the survivor set only means "
                "something under 'average_arrived'; with 'refuse' the round "
                "needs every rank and this would be accepted and ignored"
                % (min_participants, on_missing)
            )
        if (getattr(algorithm, "secure", False)
                and on_missing == "average_arrived"):
            raise NotImplementedError(
                "torchnative.federated.FederatedDataParallel: "
                "on_missing='average_arrived' with %r. A rank that drops after "
                "the pairwise seeds were agreed leaves its masks in the "
                "survivors' sum, and removing them needs Bonawitz's recovery "
                "(Shamir shares of every seed, an unmasking round), which is "
                "not built. Secure rounds use on_missing='refuse': a missing "
                "rank aborts the round and it is undone." % (algorithm,)
            )
        if secure_aggregation:
            # Built since issue #253, as an algorithm rather than a flag (see
            # algorithms.SecAgg for why one spelling).
            raise NotImplementedError(
                "torchnative.federated.FederatedDataParallel: secure_aggregation= "
                "is not a flag. Secure aggregation by pairwise masking is an "
                "algorithm wrapper: FederatedDataParallel(model, "
                "algorithm=algorithms.SecAgg(algorithms.FedAvg())). A flag "
                "beside algorithm= would be a second spelling of the same "
                "mechanism, and one of the two would end up meaning something "
                "the other does not."
            )
        if differential_privacy is not None:
            # Client-level DP is an algorithm, not a flag: the clip and the
            # noise sit between the sum and the division, which is the
            # algorithm's code (docs/distributed/FEDERATED5.md, agent proposal).
            raise NotImplementedError(
                "torchnative.federated.FederatedDataParallel: differential_privacy=%r "
                "is not a keyword this wrapper serves. Client-level DP (the "
                "DP-FedAvg mechanism: per-client L2 clip of the whole delta, "
                "Gaussian noise added once to the sum) is "
                "algorithm=algorithms.DPFedAvg(clip_norm=C, noise_multiplier=sigma), "
                "because the clip and the noise live inside the aggregation. "
                "An (epsilon, delta) target is not accepted: turning one into "
                "sigma needs an accountant, and none is built. Example-level "
                "DP needs per-example gradient clipping inside the local step, "
                "and this stack's backward produces one gradient per parameter "
                "over the batch, not per example." % (differential_privacy,)
            )

        from torchnative.delta import Delta

        self.module = module
        self.algorithm = algorithm if algorithm is not None else FedAvg()
        self.group = group
        self.aggregation = aggregation
        self._federated_logging_data = {}
        self.select = select
        self.on_missing = on_missing
        self.min_participants = min_participants
        if parameters is None:
            names = [n for n, p in module.named_parameters() if p.requires_grad]
        else:
            names = list(parameters)
        # `Delta.over` checks the names against the module and refuses an empty
        # set, so a typo or a module with nothing trainable is not a round that
        # covers less than it claims.
        self._delta = Delta.over(module, names)
        # An algorithm whose difference from FedAvg lives in the *local*
        # objective gets to install it here. FedProx is the one; it refuses to
        # aggregate at all if this never happened, so the wiring cannot be
        # forgotten silently.
        if hasattr(self.algorithm, "attach"):
            self.algorithm.attach(self)

    @staticmethod
    def _refuse_for_server(algorithm, on_missing, secure_aggregation,
                           differential_privacy):
        """What ``aggregation='server'`` does not serve, refused by name, before anything runs."""
        from .algorithms import FedAvgM
        served = (FedAvg, FedAvgM, FedProx)
        if algorithm is not None and type(algorithm) not in served:
            raise NotImplementedError(
                "torchnative.federated.FederatedDataParallel: algorithm=%r "
                "with aggregation='server'. The server's fold is a weighted "
                "sum in ascending rank order, which is exactly how %s "
                "aggregate, and nothing else is known to aggregate that way: "
                "a %s may mask, clip, add noise or take a median, and running "
                "it through a weighted sum would compute something else and "
                "report success. Subclasses are not assumed to keep the "
                "parent's aggregation either.\n"
                "Check: algorithm=FedAvg(), FedAvgM() or FedProx(), or "
                "aggregation='all_reduce'."
                % (algorithm, ", ".join(c.__name__ for c in served),
                   type(algorithm).__name__))
        if on_missing == "average_arrived":
            raise NotImplementedError(
                "torchnative.federated.FederatedDataParallel: "
                "on_missing='average_arrived' with aggregation='server' is not "
                "built. The survivor set would have to be the server's verdict, "
                "carried to every survivor with the aggregate, the way the "
                "hub's is on the all_reduce path (docs/distributed/FEDERATED4.md "
                "§6); the server's fold refuses a round that is missing a rank "
                "instead (on_missing='refuse').\n"
                "Check: on_missing='refuse', or aggregation='all_reduce'.")
        if secure_aggregation:
            raise NotImplementedError(
                "torchnative.federated.FederatedDataParallel: "
                "secure_aggregation= with aggregation='server' is not built. "
                "The server's fold reads every participant's weighted delta in "
                "the clear, which is the opposite of what masked aggregation "
                "promises; it would need the masking, key agreement and "
                "unmasking rounds on the server's side, and none exists here.")
        if differential_privacy is not None:
            raise NotImplementedError(
                "torchnative.federated.FederatedDataParallel: "
                "differential_privacy= with aggregation='server' is not built. "
                "Neither the clipping a DP guarantee needs nor server-side "
                "noise and an accountant exist on the server's fold, and a fold "
                "without them gives no (epsilon, delta) at all.")

    def __repr__(self):
        return "<FederatedDataParallel algorithm=%r over %d parameters>" % (
            self.algorithm, len(self._delta))

    def forward(self, *args, **kwargs):
        """The wrapped module's forward, unchanged (as ``DistributedDataParallel``)."""
        return self.module(*args, **kwargs)

    # -- what an algorithm may read ----------------------------------------

    @property
    def covers(self):
        """The federated parameter names, in the order the delta was opened."""
        return self._delta.covers

    @property
    def base(self):
        """``{name: tensor}``: the weights this round started from (``w_global``).

        Live: it is replaced at every :meth:`synchronize`, so an algorithm that
        reads it at call time follows the rounds without re-attaching.
        """
        return self._delta.base

    def federated_parameters(self):
        """``{name: Parameter}`` of the federated parameters of the wrapped module."""
        params = dict(self.module.named_parameters())
        return {n: params[n] for n in self._delta.covers}

    # -- the round ---------------------------------------------------------

    def synchronize(self, weight=None):
        """One federated round. Returns None.

        Per-round diagnostics are read afterwards from
        :meth:`_get_federated_logging_data`.

        Publishes ``current parameters - base`` through the algorithm, installs
        ``base + aggregate``, and re-snapshots the base. ``weight`` is this
        rank's (for FedAvg, its local sample count); whether one is required is
        the algorithm's call, and ``FedAvg()`` refuses a missing one.
        """
        delta = self._delta
        # Every check runs before the local delta is read: a world of one, a
        # missing weight or a disagreement about the cohort should refuse
        # before anything is published.
        world, rank = _require_world(self.group, "FederatedDataParallel.synchronize")
        if self.on_missing == "average_arrived":
            # Checked here rather than in `__init__`: the world is not known
            # until there is a group, and this is the first moment there is.
            if world < 3:
                raise NotImplementedError(
                    "torchnative.federated.FederatedDataParallel: "
                    "on_missing='average_arrived' in a world of %d. One rank "
                    "leaving leaves %d, and FedAvg over one delta is that "
                    "delta -- the partial average would be the survivor's own "
                    "weights (docs/distributed/FEDERATED3.md §4.1 measured that identity "
                    "to 6e-8), so this policy cannot be shown to do anything "
                    "here.\n"
                    "Check: init_process_group(..., world_size=3) or more."
                    % (world, world - 1))
            if self.min_participants > world:
                raise ValueError(
                    "torchnative.federated.FederatedDataParallel: min_participants=%d in "
                    "a world of %d, so no round could ever meet it"
                    % (self.min_participants, world))
        if self.aggregation == "server":
            # One route per round, opened before anything can differ between
            # the ranks, so the n-th round names the same server keys on every
            # rank. The user's loop is unchanged: synchronize(weight=...).
            with _via_server(self.group, "FederatedDataParallel.synchronize") as route:
                self._round(world, rank, weight, route)
        else:
            self._round(world, rank, weight, None)

    def _round(self, world, rank, weight, route):
        """The round itself, after the checks that need only the world.

        ``route`` is the server route under ``aggregation='server'`` (every
        collective inside goes through it) and None otherwise.
        """
        delta = self._delta
        # Agreed before the delta is read, for the same reason the world size
        # is: a cohort the ranks disagree about should refuse first.
        ranks = (tuple(range(world)) if self.select is None
                 else cohort(self.select, self.group, "FederatedDataParallel.synchronize"))
        # A rank the cohort left out still attends every collective and
        # contributes nothing -- see `_Abstain`. Its weight is resolved from
        # ABSTAIN rather than from what the caller passed, so a caller who
        # passed a sample count does not have it counted in a round it was not
        # selected for.
        participating = rank in ranks
        contributed = weight if participating else ABSTAIN
        resolved = (self.algorithm.resolve_weight(contributed)
                    if hasattr(self.algorithm, "resolve_weight")
                    else (1.0 if contributed is None else 0.0
                          if contributed is ABSTAIN else float(contributed)))

        if participating:
            delta.record(self.module)
            if delta.max_abs() == 0.0:
                delta.zero()
                raise ValueError(
                    "torchnative.federated.FederatedDataParallel.synchronize: "
                    "the local delta is exactly zero, so no local training "
                    "happened since the last synchronize (or since "
                    "construction). A round with no local training would "
                    "contribute the zero delta and still be counted, at full "
                    "weight, in the average.\n"
                    "Check: forward, loss.backward(), optimizer.step() ran on "
                    "the wrapped module's federated parameters before this "
                    "call."
                )
        else:
            # Not selected: contributes the zero offset at weight zero. It
            # has to go on the wire: the ranks that were picked are inside the
            # same collectives. Anything this rank trained locally is
            # discarded when base + aggregate is installed below.
            delta.value = {n: torch.zeros_like(t)
                           for n, t in delta.base.items()}

        local_norm = delta.norm()
        missing = ()
        try:
            if self.on_missing == "average_arrived":
                with tolerate_missing(self.group) as arrival:
                    table = delta.publish(group=self.group, weight=contributed,
                                          aggregator=self.algorithm)
                    total = _total_weight(resolved, self.group)
                    missing = tuple(arrival.missing or ())
                    survivors = len(arrival.survivors)
                if survivors < self.min_participants:
                    raise RankDropped(
                        "torchnative.federated: rank(s) %s did not report, "
                        "leaving %d of %d, and this FederatedDataParallel was built with "
                        "min_participants=%d. The aggregate is refused rather "
                        "than divided by whoever was left: below the floor the "
                        "caller chose, the 'partial average' stops being an "
                        "average of anything they asked for."
                        % (list(missing), survivors, world,
                           self.min_participants),
                        rank=rank, missing=missing, during="the round's "
                        "collectives")
            else:
                table = delta.publish(group=self.group, weight=contributed,
                                      aggregator=self.algorithm)
                if getattr(self.algorithm, "secure", False):
                    # Its total weight was masked with the deltas; a second,
                    # unmasked weight collective would hand the hub every
                    # rank's sample count (algorithms.SecAgg).
                    total = self.algorithm.last_total_weight
                else:
                    total = _total_weight(resolved, self.group)
        except (RankDropped, ServerAggregationError):
            # Policy `on_missing='refuse'`: the round is *undone*. The local
            # training has already moved the model, and leaving it there would
            # keep an update no other rank has -- the ranks would silently
            # stop holding the same weights, which is the one property that
            # makes this federated learning rather than two devices training
            # alone. `revert` restores the base byte for byte, and the base is
            # the last aggregate every rank agreed on.
            delta.revert(self.module)
            delta.zero()
            raise

        # Install the aggregate: base + aggregated_delta.
        delta.value = dict(table)
        delta.apply(self.module)
        self._federated_logging_data = {
            "rank": rank, "world": world, "weight": resolved,
            "total_weight": total, "share": resolved / total,
            "covers": tuple(delta.covers), "local_norm": local_norm,
            "aggregate_norm": delta.norm(), "cohort": tuple(ranks),
            "participated": bool(participating), "missing": tuple(missing),
            "aggregation": self.aggregation,
        }
        secure = getattr(self.algorithm, "logging_data", None)
        self._federated_logging_data.update(
            secure() if secure is not None else {"secure_aggregation": False})
        # An algorithm with per-round diagnostics of its own (DPFedAvg: the
        # clip norm, the noise multiplier, whether this rank was clipped and
        # its pre-clip norm) reports them here, under its own keys.
        report = getattr(self.algorithm, "round_report", None)
        if callable(report):
            self._federated_logging_data.update(report())
        if route is not None:
            # Which server folds produced this round, as the server answered
            # them: a test matches these against the server's own ledger.
            self._federated_logging_data["server"] = "%s:%d" % (route.host,
                                                                route.port)
            self._federated_logging_data["server_receipts"] = [
                dict(r) for r in route.receipts]
        # Re-snapshot: the next round's delta must be measured against the
        # aggregated weights, not against the original base.
        delta.re_snapshot(self.module)

    def _get_federated_logging_data(self):
        """What the last :meth:`synchronize` did, as a plain dict (a copy).

        Empty before the first ``synchronize``, like DDP's
        ``_get_ddp_logging_data``. Keys: rank, world, weight, total_weight,
        share, covers, local_norm, aggregate_norm, cohort, participated,
        missing, secure_aggregation, aggregation; with ``algorithms.SecAgg`` also
        fraction_bits, scale, modulus_bits, per_rank_bound and secure_rounds
        (never a seed); with ``algorithm=algorithms.DPFedAvg(...)`` also
        clip_norm, noise_multiplier, noise_std, clipped, pre_clip_norm,
        noise_seed, dp_round, threat_model and accountant (always None:
        there is none); and under ``aggregation='server'`` also server
        (``host:port``) and server_receipts (one dict per server fold of the
        round: key, ranks, names, sha256, total_weight, as the server answered).
        """
        return dict(self._federated_logging_data)


# Vertical federated learning (issue #293). Imported last: it uses `agree`,
# `_require_world` and `_dist` from this module.
from .feature_parallel import FederatedFeatureParallel, SamplesMisaligned  # noqa: E402
