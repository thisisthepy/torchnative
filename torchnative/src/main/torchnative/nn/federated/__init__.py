"""Federated learning.

A layer above the single-device adaptation methods, not a sibling of them: the
local step is the same mechanism, with aggregation, transport and privacy on
top. Its dependencies stay behind the ``federated`` extra so that using
adaptation alone does not pull in the aggregation stack. See DESIGN.md §3.

    from torchnative.nn import federated

    engine = federated.Engine(model, method=adapt.Tent(), lr=1e-3,
                              aggregator=federated.FedAvg())
    report = engine.participate(batches, weight=n_local_samples)

**What this is built on, and why it is one round.** DESIGN.md §11.1 stacks
rounds/selection/aggregation above ``torch.distributed``, and aggregation *is*
collective communication -- so ``FedAvg`` here is a weighted ``all_reduce`` and
not a private channel. The transport under it (docs/TRANSPORT.md) implements
``world_size`` 2 with ``ReduceOp.SUM``; everything past that refuses by name,
and so does everything here that would need it.

**The refusal that matters most is at one rank.** ``FedAvg`` over a world of
one is the identity function: it returns the delta it was handed, and no test
at that size can distinguish a correct weighted average from an aggregator that
ignores its weights, drops a peer, or does nothing at all. So a world of one is
refused rather than served -- see :meth:`FedAvg.aggregate`. This is the one
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
more than one round                     :class:`Engine` refuses, and names
                                        what it would take
participant selection                   ``Engine(select=...)`` refuses
a rank that does not arrive             ``Engine(allow_missing=...)`` refuses;
                                        the round raises rather than averaging
                                        what did arrive
secure aggregation, differential
privacy, compression                    not offered at all -- no surface here
                                        takes a key, an epsilon or a codec
aggregators other than FedAvg           FedProx, FedAdam, SCAFFOLD: none
======================================  ====================================

docs/FEDERATED.md records the measurements and what each refusal was weighed
against.
"""

from __future__ import annotations

import hashlib

import torch

__all__ = ["FedAvg", "Engine", "Round", "digest", "agree"]


# The wire is JSON over a socket (docs/TRANSPORT.md §2). This guard existed
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
    """``torch.distributed``, with the ``local`` backend registered.

    Imported here rather than at module import so that reading
    ``torchnative.nn.federated`` does not register a backend as a side effect.
    ``torchnative.distributed`` is what registers it, and its ``register()`` is
    idempotent by construction.
    """
    import torch.distributed as dist

    import torchnative.distributed  # noqa: F401  -- registers backend="local"

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
            "torchnative.nn.federated: there is no process group, so there is "
            "nobody to aggregate with.\n"
            "Check: import torchnative.distributed; "
            "torch.distributed.init_process_group(backend='local', "
            "init_method='tcp://127.0.0.1:<port>', rank=<0 or 1>, "
            "world_size=2)  -- docs/TRANSPORT.md"
        )
    return dist.get_world_size(group), dist.get_rank(group)


def _require_two(group, who):
    """``(world_size, rank)``, refusing a world that is not exactly two.

    One place, because the *interesting* refusal -- a world of one -- has to
    read the same wherever a caller hits it. It was written twice first, and
    the second copy (in ``agree``) shadowed the first: at ``world_size = 1``
    ``Delta.publish`` reached the base-agreement check before the aggregator
    and reported "written for a world of exactly two", which is true and is
    not the point.
    """
    world, rank = _group(group)
    if world == 1:
        raise NotImplementedError(
            "torchnative.nn.federated: a world of one, reached through %s. "
            "Averaging a single client's delta with itself is the identity "
            "function, so this would return what it was handed and prove "
            "nothing about the aggregation -- a test at this size passes "
            "whether the weights are honoured, ignored, or never read. "
            "FedAvg is not defined here as 'the average of one'; the "
            "degenerate case is named instead of served.\n"
            "Check: torch.distributed.get_world_size() == 2, reached through "
            "init_process_group(backend='local', init_method='tcp://...', "
            "world_size=2) -- docs/TRANSPORT.md" % (who,)
        )
    if world != 2:
        raise NotImplementedError(
            "torchnative.nn.federated: world_size %d, reached through %s. The "
            "transport under this implements 2 (docs/TRANSPORT.md §3), and the "
            "agreement check in agree() is exact only for 2 -- at three it "
            "would accept (h-1, h, h+1). An all_gather settles it for any "
            "world, and allgather refuses above world_size 1."
            % (world, who)
        )
    return world, rank


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

    One ``int64`` ``all_reduce``. The test is ``total == value * world_size``,
    an *equality* test only because the world has exactly two members:
    ``h0 + h1 == 2 * h0`` iff ``h0 == h1``. At three or more it would accept
    ``(h-1, h, h+1)``, so this refuses a larger world rather than quietly
    weakening -- and the transport refuses one first.

    ``SUM`` because it is the one reduction the transport implements
    (docs/TRANSPORT.md §3), which is also why this is a sum-and-compare rather
    than a ``MIN``/``MAX`` bracket.
    """
    dist = _dist()
    world, rank = _require_two(group, "federated.agree")
    probe = torch.tensor([int(value)], dtype=torch.int64)
    dist.all_reduce(probe, op=dist.ReduceOp.SUM, group=group)
    total = int(probe[0].item())
    if total != int(value) * world:
        raise ValueError(
            "torchnative.nn.federated: the ranks disagree about %s. Rank %d "
            "holds digest %d; the %d ranks sum to %d, and would sum to %d if "
            "they agreed.\n"
            "An all_reduce would have averaged them anyway and returned a "
            "number, which is why this is checked instead of assumed."
            % (what, rank, int(value), world, total, int(value) * world)
        )
    return total


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
        "torchnative.nn.federated: %r is %d elements and %d bytes on the wire, "
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
    """``sum(weight)`` over the group, as a float. One 1-element collective."""
    dist = _dist()
    total = torch.tensor([float(weight)], dtype=torch.float64)
    dist.all_reduce(total, op=dist.ReduceOp.SUM, group=group)
    return float(total[0].item())


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
    success, which docs/DESIGN.md §6 puts below a refusal.
    """

    def __init__(self, weighted=True):
        self.weighted = bool(weighted)

    def __repr__(self):
        return "FedAvg(weighted=%r)" % (self.weighted,)

    def resolve_weight(self, weight):
        """The weight this aggregator will use, or a refusal. Not a collective.

        Split out so that :class:`Engine` reports the same number the average
        was computed with, rather than a second guess at it.
        """
        if self.weighted:
            if weight is None:
                raise TypeError(
                    "torchnative.nn.federated.FedAvg.aggregate: weight= is "
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
                    "torchnative.nn.federated.FedAvg: weight=%r. A weight has "
                    "to be finite and positive; a rank weighted 0 contributes "
                    "nothing while still counting as having participated"
                    % (weight,)
                )
            return w
        if weight is not None:
            raise TypeError(
                "torchnative.nn.federated.FedAvg(weighted=False) was given "
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
        dist = _dist()
        world, rank = _require_two(group, "FedAvg.aggregate")
        if not table:
            raise ValueError(
                "torchnative.nn.federated.FedAvg: an empty table. A round that "
                "contributes no parameters would complete and change nothing"
            )
        w = self.resolve_weight(weight)

        for name, t in table.items():
            if not isinstance(t, torch.Tensor):
                raise TypeError(
                    "torchnative.nn.federated.FedAvg: %r is %r, not a Tensor"
                    % (name, type(t).__name__)
                )
            if not t.dtype.is_floating_point:
                raise NotImplementedError(
                    "torchnative.nn.federated.FedAvg: %r is %s. A delta holds "
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
            dist.all_reduce(scaled, op=dist.ReduceOp.SUM, group=group)
            out[name] = scaled / torch.tensor(total, dtype=t.dtype)
        return out


class Round:
    """What one round did, as an object rather than as a printed line.

    Held so a caller can assert on the round instead of on the model that came
    out of it: ``weight`` is this rank's, ``total_weight`` the group's, and
    ``share`` the fraction of the aggregate this rank contributed -- the number
    that is exactly 0.5 when the round was unweighted.
    """

    def __init__(self, rank, world, weight, total_weight, steps, history,
                 covers, local_norm, aggregate_norm):
        self.rank = rank
        self.world = world
        self.weight = weight
        self.total_weight = total_weight
        self.steps = steps
        self.history = list(history)
        self.covers = tuple(covers)
        self.local_norm = local_norm
        self.aggregate_norm = aggregate_norm

    @property
    def share(self):
        return self.weight / self.total_weight

    def __repr__(self):
        return ("<Round rank %d/%d, %d step(s) over %d parameters, "
                "weight %g of %g, |local|=%.4g |aggregate|=%.4g>"
                % (self.rank, self.world, self.steps, len(self.covers),
                   self.weight, self.total_weight,
                   self.local_norm, self.aggregate_norm))


class Engine:
    """One federated round: adapt locally, contribute a delta, take the average.

        engine = federated.Engine(model, method=adapt.Tent(), lr=1e-3,
                                  aggregator=federated.FedAvg())
        report = engine.participate(batches, weight=len(local_dataset))

    The local half is ``torchnative.adapt`` unchanged -- ``wrap`` a method
    round the model, go ``online``, take a step per batch -- so the delta this
    contributes is produced by the same machinery a device uses when it is not
    federated at all. That is DESIGN.md §3's claim made operational: the
    federated destination is a *destination* for a delta, not a second kind of
    delta.

    **``participate`` takes the local data.** README §2 sketches it with no
    arguments; it does not have that shape here, because a round with no
    arguments would have to invent either the batches or the sample count, and
    both change the result. The rest of the sketch stands.

    **This is one round.** ``rounds`` exists so that asking for more is refused
    with the reason rather than silently accepted -- see :meth:`__init__`.
    """

    def __init__(self, model, method=None, aggregator=None, rounds=1,
                 group=None, lr=1e-3, optimizer=None, select=None,
                 allow_missing=False, **optimizer_kwargs):
        if method is None:
            raise TypeError(
                "torchnative.nn.federated.Engine: method= is required. The "
                "local half of a round is an adaptation method, and "
                "torchnative.adapt.wrap refuses a default for the same reason: "
                "the choice decides which parameters move and what is minimised"
            )
        if select is not None:
            raise NotImplementedError(
                "torchnative.nn.federated.Engine: select= is participant "
                "selection, and it is not implemented. Choosing a subset of "
                "clients needs a world larger than the two this transport "
                "carries and a sub-group to run the collective over -- "
                "torch.distributed.new_group is what would build it, and the "
                "backend refuses above world_size 2 (docs/TRANSPORT.md §3).\n"
                "At two ranks every selection rule that is not 'both' leaves "
                "one, and FedAvg refuses a world of one."
            )
        if allow_missing:
            raise NotImplementedError(
                "torchnative.nn.federated.Engine: allow_missing= is dropout "
                "handling, and it is not implemented. Averaging over whichever "
                "ranks arrived makes the divisor a number nobody chose, and the "
                "round reports success -- docs/DESIGN.md §6 puts that below a "
                "refusal.\n"
                "As built, a rank that does not arrive makes the collective "
                "raise (the socket times out after 30 s); it never produces a "
                "partial average. What is missing is a *policy*, not a "
                "mechanism to notice."
            )
        if rounds != 1:
            raise NotImplementedError(
                "torchnative.nn.federated.Engine: rounds=%r. One round is "
                "implemented.\n"
                "Round two is not a loop around round one: the delta each rank "
                "contributes has to be opened over the *aggregated* weights, "
                "and Adapted.online() deliberately never re-snapshots its base "
                "-- so every later round would measure from the original model "
                "and contribute the accumulated offset again. The check that "
                "would make a second round provable is already here "
                "(Delta.publish verifies the ranks share a base); what is "
                "missing is re-opening the delta per round, and a test that "
                "the two ranks stay bit-identical across the boundary."
                % (rounds,)
            )
        from torchnative import adapt

        self.model = model
        self.method = method
        self.aggregator = aggregator if aggregator is not None else FedAvg()
        self.rounds = rounds
        self.group = group
        self.adapted = adapt.wrap(model, method=method, optimizer=optimizer,
                                  lr=lr, **optimizer_kwargs)
        self._participated = False

    def __repr__(self):
        return "<Engine method=%s aggregator=%r rounds=%d>" % (
            type(self.method).__name__, self.aggregator, self.rounds)

    def participate(self, batches, weight=None, epochs=1):
        """Run the local epochs, contribute the delta, take the average.

        ``batches`` is an iterable of what the model is called with: a mapping
        goes as keyword arguments, a tuple as positional, anything else as one
        positional argument. One adaptation step per batch, ``epochs`` passes.

        Returns a :class:`Round`. The model is left holding
        ``base + aggregate`` -- the *averaged* update, not this rank's -- which
        is the point, and is also why the two ranks end bit-identical.
        """
        if self._participated:
            raise RuntimeError(
                "torchnative.nn.federated.Engine.participate: called twice, and "
                "this engine is one round (see rounds= in __init__). A second "
                "call would open no new delta and would contribute this rank's "
                "first-round offset again"
            )
        batches = list(batches)
        if not batches:
            raise ValueError(
                "torchnative.nn.federated.Engine.participate: no batches. A "
                "round with no local data contributes the zero delta and would "
                "still be counted, at full weight, in the average"
            )
        if epochs < 1:
            raise ValueError(
                "torchnative.nn.federated.Engine.participate: epochs=%r"
                % (epochs,)
            )

        # Both checks run before the local epochs rather than after: a world of
        # one, or a missing weight, should refuse before the model has been
        # moved -- not after a round of training that then cannot be
        # contributed and cannot be undone without a revert the caller did not
        # ask for.
        world, rank = _require_two(self.group, "Engine.participate")
        # Resolved before the local epochs, not after: a missing or nonsensical
        # weight should refuse before the model has been moved, not after a
        # round of training that then cannot be contributed.
        resolved = (self.aggregator.resolve_weight(weight)
                    if hasattr(self.aggregator, "resolve_weight")
                    else (1.0 if weight is None else float(weight)))

        self.adapted.online()
        steps = 0
        for _ in range(epochs):
            for batch in batches:
                if isinstance(batch, dict):
                    self.adapted.step(**batch)
                elif isinstance(batch, tuple):
                    self.adapted.step(*batch)
                else:
                    self.adapted.step(batch)
                steps += 1

        delta = self.adapted.adapted
        local_norm = delta.norm()
        table = delta.publish(group=self.group, weight=weight,
                              aggregator=self.aggregator)
        total = _total_weight(resolved, self.group)

        # The model keeps the base it started the round from and takes the
        # aggregate on top of it. `Delta.apply` is what writes `base + value`,
        # so the averaged offset is installed through the existing path rather
        # than through a second one that could disagree with it.
        delta.value = dict(table)
        delta.apply(self.model)
        self._participated = True

        return Round(
            rank=rank, world=world, weight=resolved, total_weight=total,
            steps=steps, history=self.adapted.history, covers=delta.covers,
            local_norm=local_norm, aggregate_norm=delta.norm(),
        )
