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
* ``record_traffic()`` (off until asked) appends one ``(direction, peer,
  tag, dtype, shape)`` record per ``send`` and per posted ``recv``, so a test
  can read from the backend what crossed ranks instead of trusting the caller
  (the pipeline runner, ``docs/distributed/PPINFER.md`` section 4). It records
  descriptors, not payloads, and only after it is asked: a long run that never
  asks pays one ``is None`` per message.

Nothing here has run: graded *builds* (AGENTS.md §16).
"""

from __future__ import annotations

from torch._C._distributed_c10d import ProcessGroupLocal

from . import transport, wire

#: upstream's default group timeout (``_DEFAULT_PG_TIMEOUT``, 30 minutes),
#: used only when the creator is handed none.
_DEFAULT_TIMEOUT_S = 1800.0

__all__ = ["ProcessGroupTorchnative", "transport_of", "record_traffic", "wire",
           "transport"]


def _refuse(what, why):
    raise NotImplementedError(
        "torchnative.distributed.backend.ProcessGroupTorchnative.%s: %s"
        % (what, why))


class ProcessGroupTorchnative(ProcessGroupLocal):
    """``ProcessGroupLocal``'s collectives, plus point-to-point over a mesh."""

    def __init__(self, rank=0, size=1, store=None, timeout=None,
                 transport_kind=None, bind_addr=None):
        super().__init__(rank, size, store)
        #: None until `record_traffic()` is called; then the list it returned.
        self._traffic = None
        self._p2p_timeout = transport._seconds(timeout, _DEFAULT_TIMEOUT_S)
        if timeout is not None:
            self._set_default_timeout(timeout)
        try:
            self._mesh = transport.Mesh(rank, size, store, self._p2p_timeout,
                                        force=transport_kind,
                                        bind_addr=bind_addr)
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

    def record_traffic(self):
        """Start recording point-to-point descriptors; return the live list.

        Each ``send`` appends ``("send", dst, tag, dtype, shape)`` and each
        posted ``recv`` appends ``("recv", src, tag, dtype, shape)``, with
        ``dtype`` as ``str(tensor.dtype)`` and ``shape`` a tuple of ints. A
        second call returns the same list. Collectives are not recorded: they
        still run on ``ProcessGroupLocal``'s star.
        """
        if self._traffic is None:
            self._traffic = []
        return self._traffic

    def _note(self, direction, peer, tag, tensor):
        log = self._traffic
        if log is not None:
            log.append((direction, int(peer), int(tag), str(tensor.dtype),
                        tuple(int(d) for d in tensor.shape)))

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
        self._note("send", dst_rank, tag, tensor)
        return self._mesh.send(dst_rank, tag, data, tensor)

    def recv(self, tensors, src_rank, tag=0):
        tensor = self._one_tensor(tensors, "recv")
        self._check_peer(src_rank, "recv")
        wire._check_tag(tag)
        self._note("recv", src_rank, tag, tensor)
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


def record_traffic(group=None):
    """Start recording the calling rank's point-to-point descriptors.

    Returns the live list ``ProcessGroupTorchnative.record_traffic`` appends
    to. ``group`` defaults to the default process group. A group whose backend
    is not this one refuses by name: there is nothing of ours to read there.
    """
    import torch
    import torch.distributed as dist
    pg = group if group is not None else dist.distributed_c10d._get_default_group()
    backend = pg._get_backend(torch.device("cpu"))
    probe = getattr(backend, "record_traffic", None)
    if probe is None:
        raise TypeError(
            "torchnative.distributed.backend.record_traffic: the group's backend "
            "is %s, not ProcessGroupTorchnative" % type(backend).__name__)
    return probe()
