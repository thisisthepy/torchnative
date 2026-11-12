"""The ``torchnative`` distributed backend: collectives plus point-to-point.

BACKEND.md §4.1 puts this backend in a pure Python package rather than in
``bootstrap.py`` (D8: Python first), so it can change without a rebuild. The
layout (``wire``, ``transport``, ``collectives``, this module) is an **agent
proposal, not confirmed**.

What ``ProcessGroupTorchnative`` is, at BACKEND.md §9 steps 2, 3 and 4:

* It **subclasses** ``torch._C._distributed_c10d.ProcessGroupLocal`` for that
  class's validation helpers, its async worker and its world-of-one bodies,
  and keeps ``ProcessGroupLocal`` itself untouched as the oracle.
* Above one rank it builds a **full mesh** (``transport.Mesh``): one
  connection per rank pair, ``inproc`` / ``shm`` / ``tcp`` chosen per pair,
  binary frames (``wire``). It does **not** build ``ProcessGroupLocal``'s star:
  ``_rendezvous`` is overridden to do nothing, and ``_star_exchange`` refuses
  by name, so a collective body that was not moved cannot silently run on a
  hub with no sockets.
* The collectives (``allreduce``, ``reduce``, ``broadcast``, ``allgather``,
  ``_allgather_base``, ``gather``, ``scatter``, ``reduce_scatter``,
  ``_reduce_scatter_base``, ``alltoall``, ``alltoall_base``, ``barrier``, and
  the shim-only ``allreduce_partial``/``allgather_partial``) run on the mesh
  (``collectives``, step 3): still a star through rank 0 with the
  ascending-rank fold, so the values are ``ProcessGroupLocal``'s bit for bit,
  but every byte is a binary frame on the leaf's mesh connection to rank 0.
  The twelve that ``ProcessGroupLocal`` makes ``async_op=True`` capable are
  wrapped here with that class's own wrapper code
  (``collectives.async_collective``), because a subclass's overrides are not
  wrapped by the base class's class-creation loop.
* ``send`` returns a ``Work`` that completes when a sender thread has written
  the frame, so ``dist.isend`` is asynchronous as gloo's is; ``dist.send`` is
  upstream's ``isend(...).wait()``. ``recv`` returns a ``Work`` matched by
  ``(src, tag)`` in posting order, written into the caller's tensor at
  ``wait()``. ``batch_isend_irecv`` takes upstream's non-coalescing branch
  (``supports_coalescing`` is False), which issues each op as an
  ``isend``/``irecv`` and returns their works; nothing in that branch needs
  ``_start_coalescing``/``_end_coalescing``. ``send_object_list`` and
  ``recv_object_list`` are upstream functions that pickle into a ``uint8``
  tensor and call ``send``/``recv``, so they reach this class through the same
  two methods.
* ``recv_anysource`` stays refused by name (BACKEND.md §4.4: no consumer).
* ``name()`` is ``"torchnative"``.

Nothing here has run: graded *builds* (AGENTS.md §16).
"""

from __future__ import annotations

from torch._C._distributed_c10d import ProcessGroupLocal

from . import collectives, transport, wire

#: upstream's default group timeout (``_DEFAULT_PG_TIMEOUT``, 30 minutes),
#: used only when the creator is handed none.
_DEFAULT_TIMEOUT_S = 1800.0

__all__ = ["ProcessGroupTorchnative", "transport_of", "wire", "transport",
           "collectives"]


def _refuse(what, why):
    raise NotImplementedError(
        "torchnative.distributed.backend.ProcessGroupTorchnative.%s: %s"
        % (what, why))


class ProcessGroupTorchnative(ProcessGroupLocal):
    """``ProcessGroupLocal``'s collective semantics on a binary mesh, plus point-to-point."""

    def __init__(self, rank=0, size=1, store=None, timeout=None,
                 transport_kind=None, bind_addr=None):
        super().__init__(rank, size, store)
        self._p2p_timeout = transport._seconds(timeout, _DEFAULT_TIMEOUT_S)
        if timeout is not None:
            self._set_default_timeout(timeout)
        self._mesh = transport.Mesh(rank, size, store, self._p2p_timeout,
                                    force=transport_kind,
                                    bind_addr=bind_addr)
        self._collective_wire = collectives.CollectiveWire(
            self._mesh, rank, size, self._p2p_timeout)

    # -- the star is not built ------------------------------------------------
    def _rendezvous(self, rank, size, store):
        """``ProcessGroupLocal`` builds its star here; this backend does not.

        The collectives run on the mesh (``collectives``), so the star's
        ``size - 1`` loopback sockets and its ``pg_local_port`` store key would
        be built only to sit unused.
        """
        return None

    def _star_exchange(self, payload, fold, tolerate=False):
        _refuse("_star_exchange",
                "the JSON star of ProcessGroupLocal is not built on this "
                "backend; every collective runs on the binary wire "
                "(torchnative.distributed.backend.collectives). Reaching this "
                "means a ProcessGroupLocal collective body ran that this class "
                "should have overridden")

    def _collective_counters(self):
        """Per collective, what this rank put on and took off the mesh. For tests."""
        return self._collective_wire.counters()

    def name(self):
        return "torchnative"

    # -- introspection -------------------------------------------------------
    def transport_of(self, peer):
        """``"inproc"``, ``"shm"`` or ``"tcp"``: the transport this rank uses to ``peer``."""
        return self._mesh.kind_of(peer)

    # -- point to point ------------------------------------------------------
    def _one_tensor(self, tensors, what):
        if not isinstance(tensors, (list, tuple)) or len(tensors) != 1:
            count = len(tensors) if isinstance(tensors, (list, tuple)) else None
            _refuse(what,
                    "takes a list of exactly one tensor, as gloo's does; got "
                    "%s" % ("%d tensors" % count if count is not None
                            else type(tensors).__name__))
        return tensors[0]

    def _check_peer(self, peer, what):
        if (isinstance(peer, bool) or not isinstance(peer, int)
                or not 0 <= peer < self._size):
            _refuse("%s at world_size %d" % (what, self._size),
                    "rank %r is not in this group of size %d, so there is no "
                    "such peer" % (peer, self._size))

    def send(self, tensors, dst_rank, tag=0):
        tensor = self._one_tensor(tensors, "send")
        self._check_peer(dst_rank, "send")
        # Encoded now, on the caller's thread: the frame is a snapshot, so a
        # caller who reuses the tensor after `isend` returns cannot change
        # what is sent (upstream calls that undefined; here it is defined).
        data = wire.encode_tensor(tensor, tag)
        return self._mesh.send(dst_rank, tag, data, tensor)

    def recv(self, tensors, src_rank, tag=0):
        tensor = self._one_tensor(tensors, "recv")
        self._check_peer(src_rank, "recv")
        wire._check_tag(tag)
        return self._mesh.recv(src_rank, tag, tensor)

    def recv_anysource(self, tensors, tag=0):
        _refuse("recv_anysource",
                "receiving from whichever rank sends first is not built. It "
                "has no consumer (docs/distributed/BACKEND.md §4.4: pipeline "
                "stages always name their peer), and matching a frame to the "
                "first of N queues is a different contract from the "
                "per-(src, tag) FIFO send/recv keep. Pass src=")

    # -- collectives on the binary wire --------------------------------------
    allreduce_partial = collectives.allreduce_partial
    allgather_partial = collectives.allgather_partial

    # -- lifetime ------------------------------------------------------------
    def shutdown(self):
        # The async worker first: the collectives still queued on it run on
        # the mesh, so the mesh closes after they have.
        result = super().shutdown()
        self._mesh.close()
        return result

    def abort(self):
        self._mesh.close()
        return super().abort()


# The collectives, and `async_op=True` for the twelve the oracle makes async.
# `_sync_<name>` mirrors the oracle's naming for the unwrapped body.
for _name, _body in collectives.BODIES.items():
    setattr(ProcessGroupTorchnative, "_sync_" + _name, _body)
    setattr(ProcessGroupTorchnative, _name,
            collectives.async_collective(_name, _body))
del _name, _body


def transport_of(peer, group=None):
    """The transport the calling rank uses to ``peer`` (a rank of ``group``).

    For tests and diagnostics: ``"inproc"``, ``"shm"`` or ``"tcp"``. ``group``
    defaults to the default process group.
    """
    import torch
    import torch.distributed as dist
    pg = group if group is not None else dist.distributed_c10d._get_default_group()
    backend = pg._get_backend(torch.device("cpu"))
    probe = getattr(backend, "transport_of", None)
    if probe is None:
        raise TypeError(
            "torchnative.distributed.backend.transport_of: the group's backend "
            "is %s, not ProcessGroupTorchnative" % type(backend).__name__)
    return probe(peer)
