"""Weight deltas over base weights.

Every adaptation method produces one of these; they differ only in lifetime
and destination. The boundary is ``model(x)`` -- state a model manages inside
its own forward (fast weights, caches) is the model's, not ours.

    from torchnative.delta import Delta

    d = Delta.over(model, names)     # snapshot the base of exactly those names
    ...                              # something writes into the parameters
    d.record(model)                  # the offset from base is now an object
    d.revert(model)                  # the base is back, bit for bit
    d.apply(model)                   # the offset is back on top of it

**Why the base is held per delta rather than per model.** DESIGN.md §9 item 5
records that ``ttadapters`` keeps a full weight copy in ``base_state`` so that
``AdaptationEngine.reset()`` has something to restore. A delta narrows that: it
holds the base of the parameters *it covers*, which for a method that adapts
normalisation affine parameters is four orders of magnitude smaller than the
model. docs/ADAPT.md §4 has both numbers on a real checkpoint.

**Why a base copy at all, rather than subtracting the offset back off.**
Because ``(w + d) - d`` is not ``w`` in floating point, and "reverted" has to
mean the base weights are the bytes they were. Subtraction is offered as
:meth:`Delta.revert_by_subtraction` precisely so the two can be compared --
docs/ADAPT.md §5 measures how far apart they land on a real checkpoint. If they
were equal the copy would be waste; they are not.

**Lifetime is the axis DESIGN.md §3 deliberately left unnamed.** It withdrew
one invented set of names and one borrowed set, and said the names get chosen
once a real integration shows the usage. What a lifetime policy has to answer
is three questions, and this type answers them by *running* rather than by
carrying a label:

======================================================  ========================
can this delta be discarded, and at what cost           :meth:`Delta.revert`,
                                                        :attr:`Delta.nbytes`
does it survive a process restart                       :meth:`Delta.persist`
can it leave the device                                 :meth:`Delta.publish`
======================================================  ========================

Two of the three refuse today. They refuse with a check the reader can run,
not with a claim about the world, because a refusal that names a fact goes
stale the day the fact changes and this repository has paid for that six times.
"""

from __future__ import annotations

__all__ = ["Delta"]


class Delta:
    """The offset of a set of named parameters from the base they were taken over.

    A delta owns two dictionaries keyed on parameter name: ``base``, the values
    the covered parameters held when the delta was opened, and ``value``, the
    offset recorded on top of them. ``value`` is ``None`` until
    :meth:`record` -- an unrecorded delta is the zero delta and costs one copy
    of the covered parameters rather than two.

    It does *not* hold the model. A delta that needed the object it came from
    could not be the thing federated averaging sends, and DESIGN.md §3's whole
    claim is that those two are the same object with different destinations.
    """

    def __init__(self, base, value=None):
        self.base = dict(base)
        self.value = None if value is None else dict(value)

    # -- construction ------------------------------------------------------

    @classmethod
    def over(cls, model, names):
        """Open a delta over the named parameters of ``model``.

        ``names`` is checked against the model rather than trusted: a name that
        does not resolve is a typo that would otherwise produce a delta silently
        covering less than it claims, which is the failure mode this whole file
        is arranged against.
        """
        params = dict(model.named_parameters())
        names = list(names)
        missing = [n for n in names if n not in params]
        if missing:
            raise KeyError(
                "torchnative.delta: %d of %d names are not parameters of this "
                "model, first is %r" % (len(missing), len(names), missing[0])
            )
        if not names:
            raise ValueError(
                "torchnative.delta: a delta over no parameters. A method that "
                "selects nothing adapts nothing, and would report every step as "
                "having run"
            )
        return cls({n: params[n].detach().clone() for n in names})

    # -- what it covers ----------------------------------------------------

    @property
    def covers(self):
        """The parameter names this delta is over, in the order it was opened."""
        return tuple(self.base)

    def __len__(self):
        return len(self.base)

    @property
    def nbytes(self):
        """``(base_bytes, value_bytes)`` -- what holding this delta costs.

        Counted through ``element_size()`` rather than assumed from the dtype,
        for the reason ``torchnative.quant.storage_bytes`` gives.
        """
        base = sum(t.numel() * t.element_size() for t in self.base.values())
        val = 0 if self.value is None else sum(
            t.numel() * t.element_size() for t in self.value.values()
        )
        return base, val

    def norm(self):
        """The L2 norm of the whole delta, as one number.

        Zero for a delta that has not been recorded, and zero is the honest
        answer there: an unrecorded delta is the zero offset.
        """
        if self.value is None:
            return 0.0
        total = 0.0
        for t in self.value.values():
            total += float((t * t).sum().item())
        return total ** 0.5

    def max_abs(self):
        """The largest single component of the delta."""
        if self.value is None:
            return 0.0
        return max(float(t.abs().max().item()) for t in self.value.values())

    # -- the three operations ---------------------------------------------

    def record(self, model):
        """Read the covered parameters and store their offset from base.

        Called after something has written into the parameters -- an optimiser
        step, a hand-applied update -- to turn "the model is different now" into
        an object that can be kept, measured, reverted and (one day) sent.
        """
        params = dict(model.named_parameters())
        self.value = {}
        for name, base in self.base.items():
            self.value[name] = (params[name].detach() - base).clone()
        return self

    def revert(self, model):
        """Put the base weights back, byte for byte.

        A copy, not a subtraction. See the module docstring, and
        :meth:`revert_by_subtraction` for the comparison that makes the choice
        a measurement rather than an assertion.

        The delta itself is *kept*. Reverting is not discarding: the whole point
        of holding the offset separately is that it can go back on with
        :meth:`apply`.
        """
        params = dict(model.named_parameters())
        for name, base in self.base.items():
            params[name].data.copy_(base)
        return self

    def apply(self, model):
        """Put ``base + value`` into the covered parameters.

        A no-op for an unrecorded delta, because an unrecorded delta is zero.

        This is *not* guaranteed to reproduce the weights :meth:`record` read:
        ``base + (w - base)`` is not ``w`` in floating point. docs/ADAPT.md §5
        measures the gap; it is what "keeping a delta" costs against "keeping
        the weights".
        """
        if self.value is None:
            return self
        params = dict(model.named_parameters())
        for name, base in self.base.items():
            params[name].data.copy_(base + self.value[name])
        return self

    def revert_by_subtraction(self, model):
        """Undo the delta by subtracting it instead of restoring the base.

        Here so that "the base copy is necessary" is a number rather than an
        argument. It is not what :meth:`revert` does.
        """
        if self.value is None:
            return self
        params = dict(model.named_parameters())
        for name in self.base:
            p = params[name]
            p.data.copy_(p.detach() - self.value[name])
        return self

    def zero(self):
        """Discard the offset, keeping the base. The delta is the zero delta again."""
        self.value = None
        return self

    # -- the two destinations that do not exist yet ------------------------

    def persist(self, path):
        """Write this delta somewhere it survives the process. Refuses today.

        The refusal names a check rather than a state of the world, because a
        refusal that names a fact goes stale silently.
        """
        raise NotImplementedError(
            "torchnative.delta: a delta cannot outlive this process yet -- "
            "writing one needs a tensor serialiser.\n"
            "Check: torch.save({'w': next(iter(delta.value.values()))}, path). "
            "If that returns instead of raising, serialisation has landed and "
            "this refusal is stale; write the bytes and delete it."
        )

    def publish(self, group=None):
        """Send this delta to an aggregator. Refuses today.

        This is the federated destination -- DESIGN.md §3's second axis, the one
        place where adaptation state leaves the device.
        """
        raise NotImplementedError(
            "torchnative.delta: a delta cannot leave this device yet -- "
            "aggregation needs a process group with more than one rank.\n"
            "Check: torch.distributed.init_process_group(backend='local', "
            "rank=0, world_size=2, store=torch.distributed.HashStore()). "
            "If that returns instead of refusing, a world of two exists and "
            "this refusal is stale (docs/DISTRIBUTED.md §5)."
        )

    def __repr__(self):
        base, val = self.nbytes
        state = "zero" if self.value is None else "|d|=%.4g" % self.norm()
        return "<Delta over %d parameters, %s, base %d B, value %d B>" % (
            len(self.base), state, base, val,
        )
