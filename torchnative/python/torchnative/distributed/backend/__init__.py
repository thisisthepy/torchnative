"""The ``torchnative`` distributed backend: collectives plus point-to-point.

BACKEND.md §4.1 puts this backend in a pure Python package rather than in
``bootstrap.py`` (D8: Python first), so it can change without a rebuild. The
layout (``wire``, ``transport``, this module) is an **agent proposal, not
confirmed**.

What ``ProcessGroupTorchnative`` is, at BACKEND.md §9 steps 2 and 4:

* It **subclasses** ``torch._C._distributed_c10d.ProcessGroupLocal``, so the
  eleven collectives, ``async_op=True``, and federated partial attendance are
  exactly the code ``tests/distributed/test_collect2.py`` and
  ``test_asyncwork.py`` hold against gloo. They still run over the star and the
  JSON wire until step 3 moves them onto the mesh.
* In addition, above one rank it builds a **full mesh** (``transport.Mesh``):
  one connection per rank pair, ``inproc`` / ``shm`` / ``tcp`` chosen per pair,
  binary frames (``wire``). ``send`` and ``recv`` run on it.
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
* BACKEND.md §9 step 5 (``server.py``): before any rendezvous a non-loopback
  bind without ``TORCHNATIVE_DIST_ALLOW_LAN=1`` refuses, then every rank
  writes its arrival to the store and waits for the others at the group's
  timeout, naming a rank that never came; the mesh's connections run the
  pre-shared key handshake when ``TORCHNATIVE_DIST_KEY`` (or ``key=``) is set.

Nothing here has run: graded *builds* (AGENTS.md §16).
"""

from __future__ import annotations

import os

from torch._C._distributed_c10d import ProcessGroupLocal

from . import server, transport, wire

#: upstream's default group timeout (``_DEFAULT_PG_TIMEOUT``, 30 minutes),
#: used only when the creator is handed none.
_DEFAULT_TIMEOUT_S = 1800.0

__all__ = ["ProcessGroupTorchnative", "transport_of", "wire", "transport",
           "server"]


def _refuse(what, why):
    raise NotImplementedError(
        "torchnative.distributed.backend.ProcessGroupTorchnative.%s: %s"
        % (what, why))


class ProcessGroupTorchnative(ProcessGroupLocal):
    """``ProcessGroupLocal``'s collectives, plus point-to-point over a mesh."""

    def __init__(self, rank=0, size=1, store=None, timeout=None,
                 transport_kind=None, bind_addr=None, key=None):
        p2p_timeout = transport._seconds(timeout, _DEFAULT_TIMEOUT_S)
        if size > 1 and store is not None:
            # BACKEND.md §9 step 5 (agent proposal, see server.py). Both before
            # the collectives' star rendezvous, whose accept waits a fixed
            # 300 s and cannot say who is missing: a LAN bind without the
            # opt-in refuses before anything binds, and a rank that never
            # arrives is named on every rank that did, at the group's timeout.
            server.check_bind(
                bind_addr or os.environ.get(transport.ENV_BIND)
                or transport.LOOPBACK, "rank %d's mesh listener" % rank)
            server.resolve_key(key)  # an empty key refuses before rendezvous
            server.rendezvous_barrier(store, rank, size, p2p_timeout)
        super().__init__(rank, size, store)
        self._p2p_timeout = p2p_timeout
        if timeout is not None:
            self._set_default_timeout(timeout)
        try:
            self._mesh = transport.Mesh(rank, size, store, self._p2p_timeout,
                                        force=transport_kind,
                                        bind_addr=bind_addr, key=key)
        except BaseException:
            self._close_star()
            raise

    def _close_star(self):
        """Close the collectives' sockets when the mesh could not be built."""
        for sock in list(getattr(self, "_peers", {}).values()):
            try:
                sock.close()
            except OSError:
                pass
        hub = getattr(self, "_hub", None)
        if hub is not None:
            try:
                hub.close()
            except OSError:
                pass

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

    # -- lifetime ------------------------------------------------------------
    def shutdown(self):
        self._mesh.close()
        return super().shutdown()

    def abort(self):
        self._mesh.close()
        return super().abort()


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
