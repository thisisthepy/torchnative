"""The collectives of the ``torchnative`` backend, on the binary wire and the mesh.

BACKEND.md §9 step 3 (issue #232). Until this module, the collectives of
``ProcessGroupTorchnative`` were ``ProcessGroupLocal``'s, run over that class's
star of loopback sockets with length-prefixed JSON of ``tolist()``. Here they
run over the full mesh ``transport.Mesh`` already builds for point-to-point,
in ``wire`` frames of raw little-endian bytes. **No JSON** (BACKEND.md §4.2).

Everything below is an **agent proposal, not confirmed** (AGENTS.md §7): the
shape of the exchange, the frame kind, the sequence-number tags, the verdict
frame, the choice of which ops go through the hub. Graded *builds*: nothing in
this file has run (AGENTS.md §16); CI runs ``tests/distributed/test_wireparity.py``.

**The shape is still a star through rank 0, with the ascending-rank fold.**
A ring is not built in this step, and that is deliberate: the hub folding in
ascending rank order is what makes the answer bit-equal to
``ProcessGroupLocal``'s (BACKEND.md §4.2 "Determinism": a ring changes the
bracketing, so it could only be held to a tolerance), and it is what gives
federated partial attendance a single verdict that every survivor receives
(``allreduce_partial``/``allgather_partial``, FEDERATED4.md §6). What moved is
the wire under it: each leaf reaches the hub over the leaf's own mesh
connection to rank 0, in binary frames, and the hub answers each leaf with
only what that leaf needs (its own chunk for ``scatter`` and
``reduce_scatter``, its own column for ``alltoall``, nothing but the verdict
for a non-root ``reduce`` or ``gather``), where the star sent everybody
everything.

**The selecting ops go through the hub too** (``broadcast``, ``scatter``,
``gather``, ``all_gather``, ``all_to_all``), although the mesh would let the
root send to each rank directly. One exchange shape keeps one deadlock argument
(every leaf sends, then reads; the hub reads from everyone, then writes; the
mesh's reader threads drain every connection into mailboxes regardless, so a
leaf blocked in a send is only waiting for bytes a reader is already taking),
one failure semantics (a rank that is gone is named in the hub's verdict, on
every survivor, for every collective), and one sequence of collectives per
group. A leaf whose payload the hub would discard (a non-root rank of
``broadcast`` or ``scatter``, every rank of ``barrier``) sends a header saying
"zero tensors" instead of its tensor, which ``ProcessGroupLocal`` could not do
because its fold read every leaf's JSON.

**One exchange**, ``CollectiveWire.exchange``, tagged with the group's
collective sequence number (one counter per group, advanced once per exchange;
collectives run one at a time per group, so it advances identically on every
rank). Every frame is ``wire.KIND_COLLECTIVE``, which the mesh routes to a
mailbox of its own (``transport.Mailbox.collective``), so no user tag can
collide with a collective and no collective frame can match a user ``recv``.

    leaf -> hub   int64 [k]                        then k tensor frames
    hub  -> leaf  int64 [status, n, *missing]      then n tensor frames

``status`` is 0, or 1 when the hub's fold raised; then one ``uint8`` frame
carries the hub's error text, so the leaves raise naming it instead of
waiting out the group timeout (``ProcessGroupLocal``'s leaves waited 30 s for
a socket timeout in that case). ``missing`` is the hub's verdict, the same on
every survivor.

**Arithmetic identity with ``ProcessGroupLocal``**, argued from the source:

* ``ProcessGroupLocal`` folds ``torch.tensor(payload, dtype=template.dtype)``
  where ``payload`` is ``t.tolist()``. For every dtype the wire carries,
  ``tolist`` gives Python values that represent the tensor's elements exactly
  (a float32, float16 or bfloat16 value is exactly a float64; an int64 is a
  Python int), and ``torch.tensor(..., dtype)`` converts them back to the same
  bits. The binary path decodes the same elements from their own bytes. So
  both folds start from bit-identical tensors of the template's dtype.
* ``SUM``/``PRODUCT``: both do ``acc = <rank r0's tensor>`` then
  ``getattr(acc, step)(<next>)`` in ascending rank order, with ``step`` read
  from the same table (``ProcessGroupLocal._FOLD_STEP``), in the template's own
  dtype. ``AVG``: both then do ``(acc / len(contributions)).to(template.dtype)``.
  Same operations, same order, same inputs: same bits.
* ``MIN``/``MAX``: ``ProcessGroupLocal`` folds with Python's ``min(a, b)`` /
  ``max(a, b)``, which return ``b`` exactly when ``b < a`` / ``b > a`` and
  ``a`` otherwise. Here it is ``torch.where(b < a, b, a)`` /
  ``torch.where(b > a, b, a)`` elementwise, the same rule, so NaN and signed
  zero are selected the same way (a comparison with NaN is False on both).
  ``bool`` is compared through ``uint8``, where ``False < True`` as in Python.
* The result: ``ProcessGroupLocal`` sends ``acc.tolist()`` as JSON (``repr`` of a
  float round-trips exactly) and each rank does ``t.copy_(torch.tensor(result,
  dtype=t.dtype))``; here each rank does ``t.copy_(<decoded bytes of acc>)``.
  The same value lands in ``t``'s dtype by the same conversion.

What this argument does not cover, and the test does: anything the reading
missed. ``tests/distributed/test_wireparity.py`` runs ``ProcessGroupLocal``
itself as the oracle on the same inputs and compares bit for bit.

Deliberate differences from ``ProcessGroupLocal``, none of them in a value:

* ``_reduce_scatter_base`` and ``alltoall_base`` call this module's synchronous
  bodies, not the async-wrapped public methods. ``ProcessGroupLocal``'s call
  ``self.reduce_scatter(...)`` / ``self.alltoall(...)`` from inside a body that
  is already running on the async worker, with the caller's ``asyncOp=True``
  options; the wrapper then queues a second job behind the running one and
  returns at once, so the outer body copies zeros (read from source, not run:
  an ``async_op=True`` ``all_to_all_single`` or ``reduce_scatter_tensor`` on the
  oracle would return the zeros of ``outs``/an unreduced buffer).
* A peer is "gone" when its connection closed (``wire.PeerGone``, which says
  "connection closed") or when it sent nothing within the **group's** timeout.
  ``ProcessGroupLocal`` used a fixed 30 s socket timeout (BACKEND.md §4.2
  "Timeouts" asks for the group's).
* The fold runs on the template's device, as ``ProcessGroupLocal``'s does;
  BACKEND.md §4.3's host-side fold for device tensors is step 7.
"""

from __future__ import annotations

import struct
import threading
import time
import types

from torch._C._distributed_c10d import ProcessGroupLocal, Work

from . import wire

__all__ = ["CollectiveWire", "BODIES", "async_collective",
           "allreduce_partial", "allgather_partial", "reduce_values"]

#: The hub's status in a verdict frame.
_OK = 0
_HUB_FAILED = 1


def _refuse_c10d(what, why):
    """``ProcessGroupLocal``'s ``refuse``, letter for letter (bootstrap.py
    ``_install_distributed_c10d``), so a refusal raised here reads exactly as the
    oracle's does and the parity test can compare the two texts."""
    raise NotImplementedError("torch._C._distributed_c10d.%s: %s" % (what, why))


# -- frames ----------------------------------------------------------------------

def _ints(tag, values):
    values = [int(v) for v in values]
    return wire.encode(wire.KIND_COLLECTIVE, tag,
                       struct.pack("<%dq" % len(values), *values),
                       dtype="int64", shape=(len(values),))


def _read_ints(frame, src, what, at_least):
    if (frame.dtype != "int64" or len(frame.shape) != 1
            or frame.shape[0] < at_least):
        raise wire.WireError(
            "torchnative.distributed.backend: the %s header from rank %d is "
            "torch.%s of shape %s, not int64 of at least %d elements; the two "
            "ranks are out of step in their collectives"
            % (what, src, frame.dtype, list(frame.shape), at_least))
    return list(struct.unpack("<%dq" % frame.shape[0], frame.payload))


def _frame_bytes(frame):
    return wire.HEADER_SIZE + 8 * len(frame.shape) + len(frame.payload)


def _through_wire(tensor, tag):
    """The hub's own contribution, encoded and decoded like a leaf's.

    Not needed for the bits (a clone would hold the same ones), but it makes the
    hub's tensor the same kind of object as every other contribution: a fresh,
    contiguous CPU tensor decoded from the wire.
    """
    return wire.frame_to_tensor(wire.decode(wire.encode_collective(tensor, tag)))


class _Gone(Exception):
    """A peer that closed its connection or sent nothing within the timeout."""

    def __init__(self, peer, message):
        super().__init__(message)
        self.peer = peer


class _FrameWait:
    """One posted receive in the collectives' mailbox (``Mailbox.post`` shape)."""

    def __init__(self, peer, tag):
        self._peer = peer
        self._tag = tag
        self._done = threading.Event()
        self._frame = None
        self._exception = None

    def _fulfil(self, frame):
        self._frame = frame
        self._done.set()

    def _fail(self, exc):
        if self._done.is_set():
            return
        self._exception = exc
        self._done.set()


class CollectiveWire:
    """One group's collective exchanges over its mesh. Built by the backend."""

    def __init__(self, mesh, rank, size, timeout_s):
        self.mesh = mesh
        self.rank = rank
        self.size = size
        self.timeout_s = float(timeout_s)
        self._seq = 0
        self._lock = threading.Lock()
        self._counters = {}

    # -- counters --------------------------------------------------------------
    def counters(self):
        """``{what: {exchanges, frames_sent, bytes_sent, frames_received,
        bytes_received}}``: what this rank put on and took off the mesh, per
        collective. For tests: a collective that reached the mesh has frames
        here; one served any other way does not."""
        with self._lock:
            return {what: dict(c) for what, c in self._counters.items()}

    def _bump(self, what, **deltas):
        with self._lock:
            c = self._counters.setdefault(what, {
                "exchanges": 0, "frames_sent": 0, "bytes_sent": 0,
                "frames_received": 0, "bytes_received": 0})
            for key, value in deltas.items():
                c[key] += value

    # -- frames ----------------------------------------------------------------
    def _send(self, what, dst, tag, data):
        work = self.mesh.send(dst, tag, data)
        self._bump(what, frames_sent=1, bytes_sent=len(data))
        return work

    def _recv(self, what, src, tag, deadline):
        box = self.mesh.mailbox.collective
        waiter = _FrameWait(src, tag)
        box.post(waiter)
        if not waiter._done.wait(max(deadline - time.monotonic(), 0.0)):
            box.cancel(waiter)
            if not waiter._done.is_set():
                raise _Gone(src,
                            "torchnative.distributed.backend: rank %d sent "
                            "nothing for this %s (collective %d) within the "
                            "group's timeout of %.1f s"
                            % (src, what, tag, self.timeout_s))
        exc = waiter._exception
        if exc is not None:
            if isinstance(exc, wire.PeerGone):
                raise _Gone(src, str(exc))
            raise exc
        frame = waiter._frame
        self._bump(what, frames_received=1, bytes_received=_frame_bytes(frame))
        return frame

    # -- one exchange ------------------------------------------------------------
    def exchange(self, what, mine, fold, tolerate):
        """One collective. Returns ``(tensors for this rank, missing)``.

        ``mine`` is this rank's list of tensors (possibly empty). ``fold(
        contributions, present)`` runs **on the hub only**; ``contributions``
        maps each rank that arrived to its decoded tensors and ``present`` is
        those ranks in ascending order; it returns ``{rank: [tensors]}`` for
        every present rank. With ``tolerate=False`` and anybody missing, the
        fold is skipped: the caller raises naming the missing ranks, on every
        survivor, because the verdict travels to all of them. When the hub
        itself is gone, a leaf gets ``(None, every other rank)``, as
        ``ProcessGroupLocal._star_exchange`` returns.
        """
        with self._lock:
            self._seq += 1
            tag = self._seq
        self._bump(what, exchanges=1)
        deadline = time.monotonic() + self.timeout_s
        if self.rank == 0:
            return self._hub(what, tag, mine, fold, tolerate, deadline)
        return self._leaf(what, tag, mine, deadline)

    def _hub(self, what, tag, mine, fold, tolerate, deadline):
        contributions = {0: [_through_wire(t, tag) for t in mine]}
        missing = []
        for peer in range(1, self.size):
            try:
                (count,) = _read_ints(self._recv(what, peer, tag, deadline),
                                      peer, what, 1)[:1]
                contributions[peer] = [
                    wire.frame_to_tensor(self._recv(what, peer, tag, deadline))
                    for _ in range(count)]
            except _Gone:
                contributions.pop(peer, None)
                missing.append(peer)
        present = [r for r in range(self.size) if r not in missing]

        failure = None
        replies = {r: [] for r in present}
        if tolerate or not missing:
            try:
                replies = fold(contributions, present)
            except Exception as exc:  # noqa: BLE001 -- told to every leaf, then raised
                failure = exc

        encoded = {}

        def encode(tensor):
            key = id(tensor)
            if key not in encoded:
                encoded[key] = (tensor, wire.encode_collective(tensor, tag))
            return encoded[key][1]

        sends = []
        for peer in present:
            if peer == 0:
                continue
            if failure is not None:
                text = ("%s: %s" % (type(failure).__name__, failure)).encode(
                    "utf-8", "replace")
                frames = [_ints(tag, [_HUB_FAILED, 1] + missing),
                          wire.encode(wire.KIND_COLLECTIVE, tag, text,
                                      dtype="uint8", shape=(len(text),))]
            else:
                out = replies[peer]
                frames = ([_ints(tag, [_OK, len(out)] + missing)]
                          + [encode(t) for t in out])
            for data in frames:
                sends.append((peer, self._send(what, peer, tag, data)))

        late = []
        for peer, work in sends:
            if peer in late:
                continue
            try:
                work.wait(max(deadline - time.monotonic(), 1.0))
            except Exception:  # noqa: BLE001 -- a send that did not land
                late.append(peer)
        if failure is not None:
            raise failure
        if late:
            # ProcessGroupLocal._star_exchange's refusal, letter for letter.
            raise RuntimeError(
                "connection closed: rank(s) %s contributed to this "
                "collective and then went away before the result "
                "reached them, so the survivors do not agree about "
                "who was in it" % (late,)
            )
        return replies[0], tuple(missing)

    def _leaf(self, what, tag, mine, deadline):
        everyone_else = tuple(r for r in range(self.size) if r != self.rank)
        try:
            sends = [self._send(what, 0, tag, _ints(tag, [len(mine)]))]
            for tensor in mine:
                sends.append(self._send(what, 0, tag,
                                        wire.encode_collective(tensor, tag)))
            verdict = _read_ints(self._recv(what, 0, tag, deadline), 0, what, 2)
            status, count, missing = verdict[0], verdict[1], tuple(verdict[2:])
            frames = [self._recv(what, 0, tag, deadline) for _ in range(count)]
            for work in sends:
                try:
                    work.wait(max(deadline - time.monotonic(), 1.0))
                except Exception as exc:  # noqa: BLE001
                    raise _Gone(0, str(exc))
        except _Gone:
            # The hub is what every rank reaches the others through, so losing
            # it is losing all of them at once (ProcessGroupLocal's rule).
            return None, everyone_else
        if status == _HUB_FAILED:
            text = (frames[0].payload.decode("utf-8", "replace") if frames
                    else "the hub sent no reason")
            raise RuntimeError(
                "torchnative.distributed.backend: rank 0, the hub of this %s, "
                "failed while folding it, so no rank holds a result: %s"
                % (what, text))
        if status != _OK:
            raise wire.WireError(
                "torchnative.distributed.backend: the hub's verdict for this %s "
                "has status %d, which this end does not know" % (what, status))
        return [wire.frame_to_tensor(f) for f in frames], missing


def _exchange(pg, what, mine, fold, tolerate=False):
    """The one place every collective reaches the wire.

    The drain lives here and nowhere else, as it lived in
    ``ProcessGroupLocal._star_exchange``: a synchronous collective issued from
    the caller's thread while asynchronous ones are still queued waits for them
    first, so the wire order is the issue order (ASYNCWORK.md §4, §8: one choke
    point, so deleting it is one thing a test can see).
    """
    pg._drain_async()
    return pg._collective_wire.exchange(what, mine, fold, tolerate)


# -- the arithmetic ------------------------------------------------------------------

def _as_template(value, template):
    out = value.to(dtype=template.dtype)
    if str(out.device) != str(template.device):
        out = out.to(template.device)
    return out.clone()


def _pick(acc, value, name):
    """Python's ``min(acc, value)`` / ``max(acc, value)``, elementwise."""
    import torch
    a, b = acc, value
    if a.dtype == torch.bool:
        a, b = a.to(torch.uint8), b.to(torch.uint8)
    mask = (b < a) if name == "MIN" else (b > a)
    return torch.where(mask, value, acc)


def reduce_values(values, template, kind):
    """Fold ``values`` (one tensor per present rank, ascending rank order).

    ``ProcessGroupLocal._reduce_fold``'s arithmetic on tensors instead of on
    ``tolist()`` payloads; the module docstring argues the identity.
    """
    name = getattr(kind, "name", "SUM") if kind is not None else "SUM"
    acc = _as_template(values[0], template)
    if name in ("MIN", "MAX"):
        for value in values[1:]:
            acc = _pick(acc, _as_template(value, template), name)
        return acc
    step = ProcessGroupLocal._FOLD_STEP.get(name, "add_")
    for value in values[1:]:
        getattr(acc, step)(_as_template(value, template))
    if name == "AVG":
        acc = (acc / len(values)).to(template.dtype)
    return acc


def _fold_reduce(template, kind, root=None):
    """Every present rank (or only ``root``) gets the fold of tensor 0."""
    def fold(contributions, present):
        acc = reduce_values([contributions[r][0] for r in present], template,
                            kind)
        return {r: ([acc] if root is None or r == root else [])
                for r in present}
    return fold


def _fold_pick_root(root, what):
    """``ProcessGroupLocal._pick_fold``: the root's tensors, to every rank."""
    def fold(contributions, present):
        if root not in contributions:
            raise RuntimeError(
                "connection closed: rank %d is the root of this %s and did "
                "not arrive, so there is nothing to send" % (root, what))
        chosen = list(contributions[root])
        return {r: chosen for r in present}
    return fold


def _fold_gather(root=None):
    """Every present rank's tensor 0, in rank order, to everyone (or ``root``)."""
    def fold(contributions, present):
        everything = [contributions[r][0] for r in present]
        return {r: (everything if root is None or r == root else [])
                for r in present}
    return fold


def _fold_scatter(root):
    """Chunk ``r`` of the root's list, to rank ``r`` only."""
    def fold(contributions, present):
        chunks = contributions[root]
        return {r: [chunks[r]] for r in present}
    return fold


def _fold_reduce_scatter(template, kind):
    """Chunk ``r`` folded across ranks, to rank ``r`` only."""
    def fold(contributions, present):
        return {r: [reduce_values([contributions[p][r] for p in present],
                                  template, kind)]
                for r in present}
    return fold


def _fold_transpose(size):
    """Rank ``src``'s chunk ``r`` becomes rank ``r``'s slot ``src``."""
    def fold(contributions, present):
        return {r: [contributions[src][r] for src in range(size)]
                for r in present}
    return fold


def _fold_nothing(contributions, present):
    return {r: [] for r in present}


# -- the collectives ---------------------------------------------------------------
#
# Each body takes the backend as `pg`. At world 1 each one is the oracle's own
# body (`ProcessGroupLocal._sync_<name>`), which never touched the star there.
# Above one, the validation is the oracle's, through its own helpers
# (`_check_reduce_op`, `_reduce_kind`, `_check_root_n`) where it has them and
# letter for letter where it was inline; the parity test compares the refusal
# texts of both.

def _gone(gone, what, phrase):
    raise RuntimeError("connection closed: rank(s) %s %s this %s"
                       % (list(gone), phrase, what))


def _allreduce(pg, tensors, opts, tolerate):
    pg._check_reduce_op(opts, "allreduce")
    kind = pg._reduce_kind(opts, "allreduce")
    missing = ()
    what = "allreduce_partial" if tolerate else "allreduce"
    for t in tensors:
        result, gone = _exchange(pg, what, [t], _fold_reduce(t, kind), tolerate)
        if gone and not tolerate:
            _gone(gone, "allreduce", "did not contribute to")
        missing = gone
        if result:
            t.copy_(result[0])
    return Work(tensors), missing


def allreduce(pg, tensors, opts=None):
    if pg._size == 1:
        return ProcessGroupLocal._sync_allreduce(pg, tensors, opts)
    work, _ = _allreduce(pg, tensors, opts, tolerate=False)
    return work


def allreduce_partial(pg, tensors, opts=None):
    """``ProcessGroupLocal.allreduce_partial`` on the wire: ``(Work, missing)``.

    The survivor set is the hub's verdict, carried to every survivor in the
    verdict frame of the same exchange as the data. Not async-wrapped, as in
    the oracle (ASYNCWORK.md §7).
    """
    if pg._size == 1:
        return ProcessGroupLocal.allreduce_partial(pg, tensors, opts)
    return _allreduce(pg, tensors, opts, tolerate=True)


def reduce(pg, tensors, opts=None):
    if pg._size == 1:
        return ProcessGroupLocal._sync_reduce(pg, tensors, opts)
    pg._check_reduce_op(opts, "reduce")
    root = pg._check_root_n(opts, "reduce")
    kind = pg._reduce_kind(opts, "reduce")
    for t in tensors:
        result, gone = _exchange(pg, "reduce", [t],
                                 _fold_reduce(t, kind, root=root))
        if gone:
            _gone(gone, "reduce", "did not contribute to")
        # Only the root's buffer is written; every other one is left as it
        # was, which is ProcessGroupLocal's documented choice (COLLECT2.md §4).
        if pg._rank == root and result:
            t.copy_(result[0])
    return Work(tensors)


def broadcast(pg, tensors, opts=None):
    if pg._size == 1:
        return ProcessGroupLocal._sync_broadcast(pg, tensors, opts)
    root = pg._check_root_n(opts, "broadcast")
    for t in tensors:
        mine = [t] if pg._rank == root else []
        result, gone = _exchange(pg, "broadcast", mine,
                                 _fold_pick_root(root, "broadcast"))
        if gone:
            _gone(gone, "broadcast", "never reached")
        if result:
            t.copy_(result[0])
    return Work(tensors)


def _allgather(pg, source, tolerate, what, root=None):
    """``([tensor or None per rank], missing)``, ordered by rank."""
    result, gone = _exchange(pg, what, [source], _fold_gather(root), tolerate)
    if gone and not tolerate:
        # ProcessGroupLocal._allgather says "allgather" for every caller.
        _gone(gone, "allgather", "did not contribute to")
    if result is None:
        return None, gone
    present = [r for r in range(pg._size) if r not in gone]
    slots = [None] * pg._size
    for r, tensor in zip(present, result):
        slots[r] = tensor
    return slots, gone


def allgather(pg, output_tensors, input_tensors, opts=None):
    if pg._size == 1:
        return ProcessGroupLocal._sync_allgather(pg, output_tensors,
                                                 input_tensors, opts)
    for outputs, source in zip(output_tensors, input_tensors):
        if len(outputs) != pg._size:
            raise ValueError(
                "torch._C._distributed_c10d.ProcessGroupLocal.allgather"
                ": output list has %d slots for a world "
                "of size %d" % (len(outputs), pg._size))
        payloads, _ = _allgather(pg, source, False, "allgather")
        for rank, payload in enumerate(payloads):
            outputs[rank].copy_(payload)
    return Work([t for group in output_tensors for t in group])


def allgather_partial(pg, output_tensors, input_tensors, opts=None):
    """``ProcessGroupLocal.allgather_partial`` on the wire: ``(Work, missing)``.

    The slots of a rank that did not report are left as the caller passed them,
    as in the oracle.
    """
    if pg._size == 1:
        return ProcessGroupLocal.allgather_partial(pg, output_tensors,
                                                   input_tensors, opts)
    missing = ()
    for outputs, source in zip(output_tensors, input_tensors):
        if len(outputs) != pg._size:
            raise ValueError(
                "torch._C._distributed_c10d.ProcessGroupLocal."
                "allgather_partial: output list has %d "
                "slots for a world of size %d" % (len(outputs), pg._size))
        payloads, missing = _allgather(pg, source, True, "allgather_partial")
        if payloads is None:
            continue
        for rank, payload in enumerate(payloads):
            if payload is None:
                continue
            outputs[rank].copy_(payload)
    return Work([t for group in output_tensors for t in group]), missing


def _allgather_base(pg, output, input, opts=None):
    if pg._size == 1:
        return ProcessGroupLocal._sync__allgather_base(pg, output, input, opts)
    import torch
    payloads, _ = _allgather(pg, input, False, "_allgather_base")
    flat = torch.cat([p.to(dtype=output.dtype).reshape(-1) for p in payloads])
    output.copy_(flat.reshape(output.shape))
    return Work([output])


def gather(pg, output_tensors, input_tensors, opts=None):
    if pg._size == 1:
        return ProcessGroupLocal._sync_gather(pg, output_tensors,
                                              input_tensors, opts)
    root = pg._check_root_n(opts, "gather")
    for index, source in enumerate(input_tensors):
        payloads, _ = _allgather(pg, source, False, "gather", root=root)
        if pg._rank != root:
            continue
        outputs = output_tensors[index] if index < len(output_tensors) else []
        if len(outputs) != pg._size:
            raise ValueError(
                "torch._C._distributed_c10d.ProcessGroupLocal.gather: "
                "the root's output list has %d slots for "
                "a world of size %d" % (len(outputs), pg._size))
        for rank, payload in enumerate(payloads):
            outputs[rank].copy_(payload)
    return Work([t for group in output_tensors for t in group])


def scatter(pg, output_tensors, input_tensors, opts=None):
    if pg._size == 1:
        return ProcessGroupLocal._sync_scatter(pg, output_tensors,
                                               input_tensors, opts)
    root = pg._check_root_n(opts, "scatter")
    for index, output in enumerate(output_tensors):
        if pg._rank == root:
            sources = input_tensors[index] if index < len(input_tensors) else []
            if len(sources) != pg._size:
                raise ValueError(
                    "torch._C._distributed_c10d.ProcessGroupLocal."
                    "scatter: the root's input list has %d "
                    "chunks for a world of size %d" % (len(sources), pg._size))
            mine = list(sources)
        else:
            mine = []
        result, gone = _exchange(pg, "scatter", mine, _fold_scatter(root))
        if gone:
            _gone(gone, "scatter", "never reached")
        if result:
            output.copy_(result[0])
    return Work(list(output_tensors))


def _reduce_scatter(pg, output_tensors, input_tensors, opts, what):
    kind = pg._reduce_kind(opts, "reduce_scatter")
    for output, sources in zip(output_tensors, input_tensors):
        if len(sources) != pg._size:
            raise ValueError(
                "torch._C._distributed_c10d.ProcessGroupLocal."
                "reduce_scatter: the input list has %d "
                "chunks for a world of size %d" % (len(sources), pg._size))
        result, gone = _exchange(pg, what, list(sources),
                                 _fold_reduce_scatter(output, kind))
        if gone:
            _gone(gone, "reduce_scatter", "did not contribute to")
        if result:
            output.copy_(result[0])
    return Work(list(output_tensors))


def reduce_scatter(pg, output_tensors, input_tensors, opts=None):
    if pg._size == 1:
        return ProcessGroupLocal._sync_reduce_scatter(pg, output_tensors,
                                                      input_tensors, opts)
    pg._check_reduce_op(opts, "reduce_scatter")
    return _reduce_scatter(pg, output_tensors, input_tensors, opts,
                           "reduce_scatter")


def _reduce_scatter_base(pg, output, input, opts=None):
    if pg._size == 1:
        return ProcessGroupLocal._sync__reduce_scatter_base(pg, output, input,
                                                            opts)
    pg._check_reduce_op(opts, "_reduce_scatter_base")
    flat = input.reshape(-1)
    total = flat.shape[0]
    width = output.reshape(-1).shape[0]
    if total != width * pg._size:
        raise ValueError(
            "torch._C._distributed_c10d.ProcessGroupLocal."
            "reduce_scatter_tensor: the input holds %d elements, "
            "which is not %d chunks of the output's %d"
            % (total, pg._size, width))
    chunks = [flat[j * width:(j + 1) * width] for j in range(pg._size)]
    # The oracle reaches this through `self.reduce_scatter`, which checks the
    # op again under that name; the same check, so the same refusal text.
    pg._check_reduce_op(opts, "reduce_scatter")
    _reduce_scatter(pg, [output], [chunks], opts, "_reduce_scatter_base")
    return Work([output])


def _alltoall(pg, output_tensors, input_tensors, what):
    if len(input_tensors) != pg._size or len(output_tensors) != pg._size:
        raise ValueError(
            "torch._C._distributed_c10d.ProcessGroupLocal.alltoall: "
            "%d inputs and %d "
            "outputs in a world of size %d"
            % (len(input_tensors), len(output_tensors), pg._size))
    result, gone = _exchange(pg, what, list(input_tensors),
                             _fold_transpose(pg._size))
    if gone:
        _gone(gone, "alltoall", "never reached")
    if result:
        for src in range(pg._size):
            output_tensors[src].copy_(result[src])
    return Work(list(output_tensors))


def alltoall(pg, output_tensors, input_tensors, opts=None):
    if pg._size == 1:
        return ProcessGroupLocal._sync_alltoall(pg, output_tensors,
                                                input_tensors, opts)
    return _alltoall(pg, output_tensors, input_tensors, "alltoall")


def alltoall_base(pg, output, input, output_split_sizes=None,
                  input_split_sizes=None, opts=None):
    if pg._size == 1:
        return ProcessGroupLocal._sync_alltoall_base(
            pg, output, input, output_split_sizes, input_split_sizes, opts)
    for name, splits in (("output_split_sizes", output_split_sizes),
                         ("input_split_sizes", input_split_sizes)):
        if splits is None or len(splits) == 0:
            continue
        sizes = list(splits)
        if len(set(sizes)) != 1:
            _refuse_c10d(
                "ProcessGroupLocal.all_to_all_single with uneven "
                "%s at world_size %d" % (name, pg._size),
                "%s=%s is not one width repeated. An uneven "
                "all-to-all needs every rank to know every other "
                "rank's split vector before it can place its own "
                "chunk, which is an exchange this backend does not do. "
                "Equal splits are implemented (docs/distributed/COLLECT2.md)"
                % (name, sizes),
            )
    flat_in = input.reshape(-1)
    total = flat_in.shape[0]
    if total % pg._size:
        raise ValueError(
            "torch._C._distributed_c10d.ProcessGroupLocal."
            "all_to_all_single: %d input elements do not divide "
            "into %d equal chunks" % (total, pg._size))
    width = total // pg._size
    out_total = output.reshape(-1).shape[0]
    if out_total != width * pg._size:
        raise ValueError(
            "torch._C._distributed_c10d.ProcessGroupLocal."
            "all_to_all_single: the output holds %d elements "
            "where %d are being scattered into it"
            % (out_total, width * pg._size))
    import torch
    ins = [flat_in[j * width:(j + 1) * width] for j in range(pg._size)]
    outs = [torch.zeros(width, dtype=output.dtype, device=output.device)
            for _ in range(pg._size)]
    # The synchronous body, not `pg.alltoall`: see the module docstring.
    _alltoall(pg, outs, ins, "alltoall_base")
    output.copy_(torch.cat(outs).reshape(output.shape))
    return Work([output])


def barrier(pg, opts=None):
    if pg._size == 1:
        return ProcessGroupLocal._sync_barrier(pg, opts)
    _, gone = _exchange(pg, "barrier", [], _fold_nothing)
    if gone:
        _gone(gone, "barrier", "never reached")
    return Work()


#: The collectives `ProcessGroupLocal` wraps for `async_op=True`
#: (bootstrap.py `_ASYNC_COLLECTIVES`), with this module's body for each. The
#: two `*_partial` spellings are absent there and here (ASYNCWORK.md §7).
BODIES = {
    "allreduce": allreduce,
    "reduce": reduce,
    "broadcast": broadcast,
    "allgather": allgather,
    "_allgather_base": _allgather_base,
    "gather": gather,
    "scatter": scatter,
    "reduce_scatter": reduce_scatter,
    "_reduce_scatter_base": _reduce_scatter_base,
    "alltoall": alltoall,
    "alltoall_base": alltoall_base,
    "barrier": barrier,
}


# -- async_op=True ---------------------------------------------------------------------

def async_collective(name, body):
    """``body`` wrapped exactly as ``ProcessGroupLocal`` wraps its own ``name``.

    ``ProcessGroupLocal`` applies ``_make_async_collective`` to its own methods
    when its class is created, so a subclass's override is not wrapped. That
    factory is a local of ``_install_distributed_c10d`` and is not exported. So
    instead of writing a second wrapper, this builds a new function from **the
    same code object** as the oracle's wrapper for ``name``, with the same
    closure cells (``AsyncWork``, ``_tree_clone``, ``_async_opts``, and the
    output index the oracle chose for ``name``) except ``body``, which becomes
    this module's. The staging clone, the publication rule, the single worker
    thread and the "no drain in the wrapper" comment are therefore the
    oracle's, not a copy of them. bootstrap.py is not edited.

    If the oracle's wrapper ever stops having that shape, this refuses by name
    at import rather than leaving ``async_op=True`` silently synchronous.
    """
    template = ProcessGroupLocal.__dict__.get(name)
    code = getattr(template, "__code__", None)
    closure = getattr(template, "__closure__", None)
    if code is None or closure is None:
        _refuse_c10d(
            "ProcessGroupLocal.%s" % name,
            "torchnative.distributed.backend.collectives.async_collective "
            "expected the oracle's async wrapper here (bootstrap.py "
            "_make_async_collective) and found %r, so async_op=True cannot be "
            "given the same staging-clone semantics" % (template,))
    cells = dict(zip(code.co_freevars, closure))
    if "body" not in cells or "out_index" not in cells:
        _refuse_c10d(
            "ProcessGroupLocal.%s" % name,
            "its async wrapper closes over %s, not over `body` and "
            "`out_index`; bootstrap.py _make_async_collective changed shape "
            "and torchnative.distributed.backend.collectives.async_collective "
            "must change with it" % (sorted(cells),))
    cells["body"] = types.CellType(body)
    fn = types.FunctionType(code, template.__globals__, name,
                            template.__defaults__,
                            tuple(cells[v] for v in code.co_freevars))
    fn.__kwdefaults__ = template.__kwdefaults__
    fn.__qualname__ = "ProcessGroupTorchnative." + name
    fn.__doc__ = body.__doc__
    return fn
