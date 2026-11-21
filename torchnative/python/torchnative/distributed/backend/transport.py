"""Connections, the three transports, and the full mesh between ranks.

BACKEND.md §4.2 and D3 (owner, 2026-11-08: "전송은 일단 니가 하자는대로
구현하고"): transports split intra-node versus inter-node, chosen **per rank
pair** automatically, overridable by an option or an environment variable.
D9 (leader): the same-host transport is a Unix socket first, shared memory
later. Everything below that is not one of those sentences is an **agent
proposal, not confirmed**: the class shapes, the environment variable names,
the store keys, the greeting, and the topology (a full mesh, BACKEND.md §11).

Three layers here, matching BACKEND.md §4.5's second layer:

``Connection``
    One per rank pair. ``send_frame(bytes)``, ``recv_frame() -> wire.Frame``,
    ``close()``, and a ``kind`` attribute that is ``"inproc"``, ``"shm"`` or
    ``"tcp"``. ``SocketConnection`` serves both socket transports (the frame
    decoder loops on ``recv``, so a frame split over any number of reads
    decodes the same); ``InprocConnection`` is a pair of queues.

``InprocTransport`` / ``UnixTransport`` / ``TcpTransport``
    The listening halves. ``inproc`` registers an inbox in a per-process
    registry keyed by ``(group token, rank)``, so two groups in one process
    never meet. ``shm`` (D9) is an ``AF_UNIX`` stream socket whose path is
    unique, under ``tempfile.gettempdir()``, and unlinked as soon as the mesh
    is built. ``tcp`` binds the address given by the ``bind_addr`` option or
    ``TORCHNATIVE_DIST_BIND_ADDR``; with neither it binds loopback, the default
    D10 decided. ``TCP_NODELAY`` on both ends.

``Mesh``
    Every rank publishes what it listens on in the Store under ``tn_p2p/``
    (distinct from ``ProcessGroupLocal``'s ``pg_local_port``), reads every
    other rank's record, and picks a kind per pair: same process ``inproc``,
    same host ``shm``, otherwise ``tcp``. The lower rank of a pair connects to
    the higher one and greets with its rank and the group token; the higher
    one answers, and with a pre-shared key (``TORCHNATIVE_DIST_KEY``, D10)
    both prove it by HMAC over each other's nonce before either trusts the
    link (``_connect_handshake`` / ``_accept_handshake``; ``server.py`` holds
    the key rules and the LAN opt-in). One reader
    thread per connection demultiplexes frames into per-``(src, tag)`` queues,
    which is what makes tags match and out-of-order arrival work; one sender
    thread per connection serialises sends, which is what makes ``isend``
    asynchronous. A peer that disappears fails every receive waiting on it
    with a message naming that rank (AGENTS.md §18).

``TORCHNATIVE_DIST_TRANSPORT=inproc|shm|tcp`` forces one kind for every pair
where that kind is possible. Forcing a slower kind than the natural one works
(``tcp`` between two processes of one host; ``shm`` between two threads); a
kind the pair cannot have (``inproc`` across processes, ``shm`` across hosts)
refuses by name instead of falling back. ``ws`` (D3) is not built and refuses
by name.

The Store records are JSON. "No JSON" (BACKEND.md §4.2) is about the data
wire; the records are a few hundred bytes exchanged once at rendezvous.

Timeouts are the group's timeout (``init_process_group(timeout=...)``), never a
constant (BACKEND.md §4.2 "Timeouts", §5.3).

Graded *builds* only: nothing in this file has been run (AGENTS.md §16).
"""

from __future__ import annotations

import collections
import json
import os
import queue
import socket
import tempfile
import threading
import time
import uuid

from . import server as _server
from . import wire

#: The loopback default (BACKEND.md D10). The only address literal here.
LOOPBACK = "127.0.0.1"

#: Agent proposal: the override variable (in the style of NCCL_P2P_DISABLE).
ENV_TRANSPORT = "TORCHNATIVE_DIST_TRANSPORT"
#: Agent proposal: the opt-in LAN bind address (D10: loopback by default).
ENV_BIND = "TORCHNATIVE_DIST_BIND_ADDR"
#: How long a rank whose mesh failed keeps its listeners answering, so lower
#: ranks still mid-handshake get its by-name refusal (Mesh._linger).
LINGER_S = 5.0

#: Fastest first. "Slower or equal" in the override rule reads this order.
KINDS = ("inproc", "shm", "tcp")

STORE_PREFIX = "tn_p2p/"
_KEY_GROUP = STORE_PREFIX + "group"
_KEY_ADDR = STORE_PREFIX + "addr/%d"

_PROCESS_UUID = uuid.uuid4().hex


def _refuse(what, why):
    raise NotImplementedError(
        "torchnative.distributed.backend.%s: %s" % (what, why))


def process_token():
    """Unique per process, and different in a forked child (the pid is in it)."""
    return "%s-%d" % (_PROCESS_UUID, os.getpid())


def host_id():
    """What two ranks compare to decide they share a host.

    The host name, plus ``/etc/machine-id`` where it exists (Linux), so two
    containers or two machines that happen to share a name are told apart
    where the OS gives a way to. Agent proposal; on macOS it is the host name
    alone.
    """
    name = socket.gethostname()
    try:
        with open("/etc/machine-id") as handle:
            machine = handle.read().strip()
    except OSError:
        machine = ""
    return name + ("/" + machine if machine else "")


def _is_loopback(host):
    return host == "localhost" or host == "::1" or str(host).startswith("127.")


# -- connections ---------------------------------------------------------------

class Connection:
    """One link to one peer. Subclasses set ``kind``."""

    kind = None

    def __init__(self, peer=None):
        self.peer = peer

    def send_frame(self, data):
        raise NotImplementedError

    def recv_frame(self):
        raise NotImplementedError

    def close(self):
        return None


class SocketConnection(Connection):
    """A stream socket (``AF_UNIX`` for ``shm``, ``AF_INET`` for ``tcp``).

    ``sock`` needs only ``recv``, ``sendall`` and ``close``, so a test can hand
    it an object that returns one byte per ``recv`` and exercise the real
    decoding loop.
    """

    def __init__(self, sock, kind, peer=None):
        super().__init__(peer)
        self.kind = kind
        self._sock = sock
        self._send_lock = threading.Lock()
        self._closed = False

    def send_frame(self, data):
        with self._send_lock:
            self._sock.sendall(data)

    def recv_frame(self):
        return wire.read_frame(self._sock.recv, self.peer)

    def settimeout(self, seconds):
        setter = getattr(self._sock, "settimeout", None)
        if setter is not None:
            setter(seconds)

    def close(self):
        if self._closed:
            return
        self._closed = True
        # `shutdown` first: it wakes a reader blocked in `recv` on this socket
        # in another thread, which `close` alone does not do everywhere.
        try:
            self._sock.shutdown(socket.SHUT_RDWR)
        except (OSError, AttributeError):
            pass
        try:
            self._sock.close()
        except OSError:
            pass


_EOF = object()


class _InprocLink:
    """The state two ends of one in-process link share."""

    def __init__(self):
        self.lock = threading.Lock()
        self.closed = set()  # which ends (0 or 1) have closed


class InprocConnection(Connection):
    """Ranks in one process: frames are ``bytes`` on a queue.

    The bytes are real frames, encoded and decoded exactly as on a socket, so
    the in-process path cannot drift from the wire format the other two
    transports carry.
    """

    kind = "inproc"

    def __init__(self, inbox, outbox, link, end, peer=None):
        super().__init__(peer)
        self._in = inbox
        self._out = outbox
        self._link = link
        self._end = end

    @classmethod
    def pair(cls):
        a_to_b, b_to_a = queue.Queue(), queue.Queue()
        link = _InprocLink()
        return (cls(b_to_a, a_to_b, link, 0), cls(a_to_b, b_to_a, link, 1))

    def send_frame(self, data):
        with self._link.lock:
            if self._end in self._link.closed:
                raise RuntimeError(
                    "torchnative.distributed.backend: send on a closed "
                    "in-process link to rank %s" % (self.peer,))
            if (1 - self._end) in self._link.closed:
                raise wire.PeerGone(
                    "torchnative.distributed.backend: rank %s is gone: "
                    "connection closed (its in-process end was closed)"
                    % (self.peer,))
            self._out.put(bytes(data))

    def recv_frame(self, timeout=None):
        try:
            item = self._in.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(
                "torchnative.distributed.backend: rank %s sent nothing within "
                "%.1f s" % (self.peer, timeout)) from None
        if item is _EOF:
            self._in.put(_EOF)  # every later read sees the end too
            raise wire.PeerGone(
                "torchnative.distributed.backend: rank %s is gone: connection "
                "closed at a frame boundary" % (self.peer,))
        return wire.decode(item, self.peer)

    def close(self):
        with self._link.lock:
            if self._end in self._link.closed:
                return
            self._link.closed.add(self._end)
        self._out.put(_EOF)
        self._in.put(_EOF)  # wake this end's own reader


# -- listening halves ------------------------------------------------------------

def _accept_loop(listener, kind, on_conn, stop, greet_timeout):
    """Accept until ``stop`` is set. ``on_conn`` reads the greeting."""
    while not stop.is_set():
        try:
            sock, _addr = listener.accept()
        except socket.timeout:
            continue
        except OSError:
            return
        sock.settimeout(None)
        if kind == "tcp":
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn = SocketConnection(sock, kind)
        conn.settimeout(greet_timeout)
        on_conn(conn)


class InprocTransport:
    kind = "inproc"
    _registry = {}
    _registry_lock = threading.Lock()

    def __init__(self, group_token, rank, on_conn):
        self._key = (group_token, rank)
        self._inbox = queue.Queue()
        with self._registry_lock:
            if self._key in self._registry:
                raise RuntimeError(
                    "torchnative.distributed.backend: rank %d of group %s is "
                    "already registered in this process" % (rank, group_token))
            self._registry[self._key] = self._inbox
        self._thread = threading.Thread(
            target=self._serve, args=(on_conn,), daemon=True,
            name="torchnative-p2p-inproc-accept-r%d" % rank)
        self._thread.start()

    def _serve(self, on_conn):
        while True:
            conn = self._inbox.get()
            if conn is None:
                return
            on_conn(conn)

    def address(self):
        return self._key[0]

    @classmethod
    def connect(cls, group_token, peer):
        with cls._registry_lock:
            inbox = cls._registry.get((group_token, peer))
        if inbox is None:
            raise RuntimeError(
                "torchnative.distributed.backend: rank %d of group %s has no "
                "in-process listener in this process" % (peer, group_token))
        mine, theirs = InprocConnection.pair()
        inbox.put(theirs)
        return mine

    def close(self):
        with self._registry_lock:
            if self._registry.get(self._key) is self._inbox:
                del self._registry[self._key]
        self._inbox.put(None)


class UnixTransport:
    """``shm``, first form (D9): an ``AF_UNIX`` stream socket."""

    kind = "shm"

    def __init__(self, rank, on_conn, greet_timeout):
        self.path = os.path.join(
            tempfile.gettempdir(),
            "tnp2p-%s-r%d.sock" % (uuid.uuid4().hex[:12], rank))
        self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            self._sock.bind(self.path)
            self._sock.listen(128)
            self._sock.settimeout(0.2)
        except BaseException:
            self._sock.close()
            self._unlink()
            raise
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=_accept_loop,
            args=(self._sock, "shm", on_conn, self._stop, greet_timeout),
            daemon=True, name="torchnative-p2p-shm-accept-r%d" % rank)
        self._thread.start()

    def address(self):
        return self.path

    @staticmethod
    def connect(path, timeout):
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            sock.settimeout(timeout)
            sock.connect(path)
            sock.settimeout(None)
        except BaseException:
            sock.close()
            raise
        return SocketConnection(sock, "shm")

    def _unlink(self):
        try:
            os.unlink(self.path)
        except FileNotFoundError:
            pass

    def close(self):
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass
        self._unlink()


class TcpTransport:
    """``tcp``: any bind address, ``TCP_NODELAY``, loopback only by default."""

    kind = "tcp"

    def __init__(self, rank, on_conn, greet_timeout, bind_addr):
        family = socket.AF_INET6 if ":" in bind_addr else socket.AF_INET
        self._sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((bind_addr, 0))
            self._sock.listen(128)
            self._sock.settimeout(0.2)
        except BaseException:
            self._sock.close()
            raise
        port = self._sock.getsockname()[1]
        if bind_addr in ("", "0.0.0.0", "::"):
            # A wildcard bind is reachable at whatever this host is called.
            advertised = socket.gethostname()
        else:
            advertised = bind_addr
        self._address = [advertised, port]
        self._stop = threading.Event()
        self._thread = threading.Thread(
            target=_accept_loop,
            args=(self._sock, "tcp", on_conn, self._stop, greet_timeout),
            daemon=True, name="torchnative-p2p-tcp-accept-r%d" % rank)
        self._thread.start()

    def address(self):
        return list(self._address)

    @staticmethod
    def connect(address, timeout):
        host, port = address
        sock = socket.create_connection((host, int(port)), timeout=timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.settimeout(None)
        return SocketConnection(sock, "tcp")

    def close(self):
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass


# -- selection ---------------------------------------------------------------------

def forced_kind(value=None):
    """The override, validated. ``None`` means "choose per pair"."""
    if value is None:
        value = os.environ.get(ENV_TRANSPORT)
    if value is None or not str(value).strip():
        return None
    kind = str(value).strip().lower()
    if kind == "ws":
        _refuse("transport",
                "%s=ws: the WebSocket transport (BACKEND.md D3) is not built "
                "yet; it is step 13 of BACKEND.md §9" % ENV_TRANSPORT)
    if kind not in KINDS:
        raise ValueError(
            "torchnative.distributed.backend: %s=%r is not a transport; "
            "choose one of %s" % (ENV_TRANSPORT, value, ", ".join(KINDS)))
    return kind


def choose(mine, theirs, force=None):
    """The kind for one pair, from the two ranks' Store records.

    Symmetric in ``mine`` and ``theirs``, so both ends of a pair reach the
    same answer from the same records without talking.
    """
    a, b = mine["rank"], theirs["rank"]
    same_process = mine["process"] == theirs["process"]
    same_host = mine["host"] == theirs["host"]
    shm_ok = bool(same_host and mine.get("shm") and theirs.get("shm"))
    lan_ok = (not _is_loopback(mine["tcp"][0])
              and not _is_loopback(theirs["tcp"][0]))
    tcp_ok = same_host or lan_ok
    possible = {"inproc": same_process, "shm": shm_ok, "tcp": tcp_ok}
    if force is None:
        for kind in KINDS:
            if possible[kind]:
                return kind
        _refuse("Mesh",
                "rank %d (host %r) and rank %d (host %r) are on different "
                "hosts and at least one of them listens on loopback only "
                "(%r, %r). A LAN bind is an explicit opt-in (BACKEND.md D10): "
                "set %s to an address the other host can reach"
                % (a, mine["host"], b, theirs["host"], mine["tcp"][0],
                   theirs["tcp"][0], ENV_BIND))
    if not possible[force]:
        why = {
            "inproc": "inproc joins ranks of one process, and these are two "
                      "processes (pid %s and pid %s)"
                      % (mine.get("pid"), theirs.get("pid")),
            "shm": "shm joins ranks of one host, and these are hosts %r and "
                   "%r (or one of them has no Unix socket)"
                   % (mine["host"], theirs["host"]),
            "tcp": "the two hosts differ and one listens on loopback only; "
                   "set %s (BACKEND.md D10)" % ENV_BIND,
        }[force]
        _refuse("Mesh",
                "%s=%s is impossible for the pair rank %d / rank %d: %s. "
                "Refused rather than silently using another transport"
                % (ENV_TRANSPORT, force, a, b, why))
    return force


def _store_get(store, key, deadline, rank, what):
    """A blocking ``get`` on any of the shim's stores.

    ``TCPStore.get`` already blocks until the key exists. ``HashStore.get``
    (used when ranks are threads of one process) raises on an absent key, so it
    is polled until the group's deadline.
    """
    while True:
        try:
            return store.get(key)
        except RuntimeError:
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "torchnative.distributed.backend: rank %d gave up waiting "
                    "for %s (store key %r) at the group's timeout"
                    % (rank, what, key))
            time.sleep(0.01)


# -- works ------------------------------------------------------------------------

def _seconds(timeout, default):
    if timeout is None:
        return default
    seconds = (timeout.total_seconds() if hasattr(timeout, "total_seconds")
               else float(timeout))
    return default if seconds <= 0 else seconds


def _work_base():
    from torch._C._distributed_c10d import Work
    return Work


class _P2PWork(_work_base()):
    """A point-to-point operation in flight. ``wait()`` blocks until it is done.

    Upstream's contract for a ``Work``: ``wait()`` returns True or raises the
    operation's error; ``is_completed()`` is a poll. A timeout names the peer,
    the tag and the number of seconds, which is the group's timeout unless the
    caller passes one.
    """

    def __init__(self, peer, tag, what, timeout_s):
        super().__init__()
        self._peer = peer
        self._tag = tag
        self._what = what
        self._timeout_s = timeout_s
        self._done = threading.Event()
        self._exception = None

    def _fail(self, exc):
        if self._done.is_set():
            return
        self._exception = exc
        self._done.set()

    def _finish(self):
        return None

    def _on_timeout(self):
        return None

    def is_completed(self):
        if not self._done.is_set():
            return False
        self._finish()
        return True

    def is_success(self):
        return self._done.is_set() and self._exception is None

    def exception(self):
        return self._exception

    def wait(self, timeout=None):
        seconds = _seconds(timeout, self._timeout_s)
        if not self._done.wait(seconds):
            self._on_timeout()
            if not self._done.is_set():
                raise RuntimeError(
                    "torchnative.distributed.backend: %s rank %d (tag %d) did "
                    "not complete within %.1f s" % (
                        self._what, self._peer, self._tag, seconds))
        self._finish()
        if self._exception is not None:
            raise self._exception
        return True

    def synchronize(self):
        self.wait()
        return None

    def source_rank(self):
        return self._peer

    def _source_rank(self):
        return self._peer


class SendWork(_P2PWork):
    """Done when the frame has been handed to the transport in full."""

    def __init__(self, dst, tag, timeout_s, tensor=None):
        super().__init__(dst, tag, "send to", timeout_s)
        self._tensor = tensor

    def _sent(self):
        self._done.set()

    def result(self):
        self.wait()
        return [] if self._tensor is None else [self._tensor]


class RecvWork(_P2PWork):
    """Owns the matched frame until a synchronisation point writes it.

    The same ownership rule as ``AsyncWork`` (ASYNCWORK.md §3): the caller's
    tensor is written at ``wait()`` or at an ``is_completed()`` that returns
    True, never underneath them by the reader thread. A caller who reads the
    tensor before either sees what it held before the receive was posted.
    """

    def __init__(self, dest, src, tag, timeout_s, mailbox):
        super().__init__(src, tag, "receive from", timeout_s)
        self._dest = dest
        self._mailbox = mailbox
        self._frame = None
        self._published = False
        self._publish_lock = threading.Lock()

    def _fulfil(self, frame):
        self._frame = frame
        self._done.set()

    def _on_timeout(self):
        self._mailbox.cancel(self)

    def _finish(self):
        with self._publish_lock:
            if self._published:
                return
            self._published = True
            if self._exception is not None or self._frame is None:
                return
            try:
                self._write(self._frame)
            except Exception as exc:  # noqa: BLE001 -- handed to wait()
                self._exception = exc

    def _write(self, frame):
        import torch
        dest = self._dest
        want = wire.dtype_name(dest)
        if frame.kind != wire.KIND_TENSOR:
            raise RuntimeError(
                "torchnative.distributed.backend: receive from rank %d tag %d "
                "matched a frame of kind %d, not a tensor"
                % (self._peer, self._tag, frame.kind))
        if frame.dtype != want or frame.numel() != dest.numel():
            raise RuntimeError(
                "torchnative.distributed.backend: receive from rank %d tag %d: "
                "rank %d sent torch.%s of shape %s and this rank posted "
                "torch.%s of shape %s. Both ends of a point-to-point pair "
                "must agree on dtype and element count; nothing was written"
                % (self._peer, self._tag, self._peer, frame.dtype,
                   list(frame.shape), want, list(dest.shape)))
        value = wire.frame_to_tensor(frame).reshape(tuple(dest.shape))
        with torch.no_grad():
            dest.copy_(value)

    def result(self):
        self.wait()
        return [self._dest]


# -- demultiplexing -----------------------------------------------------------------

class Mailbox:
    """Frames by ``(src, tag)``, matched to receives in the order both arrived.

    A frame that arrives with nobody waiting is queued; a receive posted with
    no frame queued waits. Both queues are FIFO per ``(src, tag)``, which is
    upstream's ordering rule for one pair, one direction and one tag
    (``batch_isend_irecv``'s docstring). Different tags never block each
    other, so a sender may send tag 2 then tag 1 and the receiver take tag 1
    first.

    ``collective`` is a second mailbox of the same shape for frames of kind
    ``wire.KIND_COLLECTIVE`` (BACKEND.md §9 step 3, issue #232). ``deliver``
    routes by kind, so the collectives' frames and the user's point-to-point
    frames are matched in two separate spaces and a user tag can never collide
    with a collective's sequence number. A lost peer and a shutdown reach both.
    """

    def __init__(self, rank, _route=True):
        self.rank = rank
        self._lock = threading.Lock()
        self._frames = {}
        self._waiters = {}
        self._gone = {}
        self._closed = None
        #: the collectives' mailbox; ``None`` on that mailbox itself.
        self.collective = Mailbox(rank, _route=False) if _route else None

    def deliver(self, src, frame):
        if self.collective is not None and frame.kind == wire.KIND_COLLECTIVE:
            self.collective.deliver(src, frame)
            return
        key = (src, frame.tag)
        with self._lock:
            waiters = self._waiters.get(key)
            if waiters:
                work = waiters.popleft()
                if not waiters:
                    del self._waiters[key]
                work._fulfil(frame)
                return
            self._frames.setdefault(key, collections.deque()).append(frame)

    def post(self, work):
        key = (work._peer, work._tag)
        with self._lock:
            frames = self._frames.get(key)
            if frames:
                frame = frames.popleft()
                if not frames:
                    del self._frames[key]
                work._fulfil(frame)
                return
            if self._closed is not None:
                work._fail(RuntimeError(self._closed))
                return
            gone = self._gone.get(work._peer)
            if gone is not None:
                work._fail(gone[0](gone[1]))
                return
            self._waiters.setdefault(key, collections.deque()).append(work)

    def cancel(self, work):
        key = (work._peer, work._tag)
        with self._lock:
            waiters = self._waiters.get(key)
            if waiters and work in waiters:
                waiters.remove(work)
                if not waiters:
                    del self._waiters[key]

    def peer_gone(self, src, exc_type, message):
        if self.collective is not None:
            self.collective.peer_gone(src, exc_type, message)
        with self._lock:
            if src in self._gone:
                return
            self._gone[src] = (exc_type, message)
            for key in [k for k in self._waiters if k[0] == src]:
                for work in self._waiters.pop(key):
                    work._fail(exc_type(message))

    def is_gone(self, src):
        with self._lock:
            return src in self._gone

    def close(self, message):
        if self.collective is not None:
            self.collective.close(message)
        with self._lock:
            self._closed = message
            waiting = [w for q in self._waiters.values() for w in q]
            self._waiters.clear()
        for work in waiting:
            work._fail(RuntimeError(message))


# -- the greeting and the pre-shared key (D10) ---------------------------------------
#
# Agent proposal. Every frame below is a control frame whose tag is the
# sender's rank and whose payload is a small JSON object with a "t" field.
# The lower rank of a pair (the connector, C) dials the higher (the acceptor,
# A):
#
#   C -> A  hello      token, auth (does C hold a key), nonce_c
#   A -> C  accept     (neither holds a key)
#        or challenge  nonce_a, proof_a = HMAC(key, "accept"  | token | C | A | nonce_c)
#        or refuse     why (exactly one side holds a key, or a bad hello)
#   C -> A  proof      proof_c = HMAC(key, "connect" | token | C | A | nonce_a)
#        or refuse     why (proof_a did not verify)
#   A -> C  ok | refuse
#
# Each side proves the key over the *other* side's fresh nonce, so a recorded
# exchange does not verify again; the label differs per direction, so one
# side's proof cannot be reflected back as the other's. The key never leaves
# the process: only nonces and HMAC digests are sent, and nothing about the key
# is written to the store. This authenticates the link; it does not encrypt it
# (BACKEND.md §8).

class _Stranger(Exception):
    """A connection that did not greet as a rank of this group."""


class _HandshakeFailed(Exception):
    """A rank of this group failed the key check; ``error`` is what to raise."""

    def __init__(self, peer, error):
        super().__init__(str(error))
        self.peer = peer
        self.error = error


def _hs_send(conn, rank, **msg):
    conn.send_frame(wire.encode_control(rank, json.dumps(msg).encode("utf-8")))


def _hs_recv(conn, seconds):
    seconds = max(0.1, float(seconds))
    if isinstance(conn, InprocConnection):
        frame = conn.recv_frame(timeout=seconds)
    else:
        if isinstance(conn, SocketConnection):
            conn.settimeout(seconds)
        frame = conn.recv_frame()
    if frame.kind != wire.KIND_CONTROL:
        raise ValueError("a frame of kind %d where a greeting belongs" % frame.kind)
    msg = json.loads(frame.payload.decode("utf-8"))
    if not isinstance(msg, dict):
        raise ValueError("a greeting that is not an object")
    return frame.tag, msg


def _hex(data):
    return bytes(data).hex()


def _unhex(text):
    return bytes.fromhex(str(text))


def _key_refusal(rank, peer, why):
    return _server.AuthRefused(
        "torchnative.distributed.backend: rank %d refused its connection with "
        "rank %d: %s. Every rank must hold the same pre-shared key (%s), or "
        "none must. The key authenticates; it does not encrypt"
        % (rank, peer, why, _server.ENV_KEY))


def _connect_handshake(conn, rank, peer, token, key, deadline):
    """The connector's half. Returns once ``peer`` is trusted, or raises naming it."""
    def left():
        return max(1.0, deadline - time.monotonic())

    def recv():
        try:
            tag, msg = _hs_recv(conn, left())
        except (wire.PeerGone, wire.WireError, OSError, ValueError) as exc:
            raise RuntimeError(
                "torchnative.distributed.backend: rank %d's greeting to rank "
                "%d got no valid answer (%s: %s)"
                % (rank, peer, type(exc).__name__, exc)) from exc
        if tag != peer:
            raise RuntimeError(
                "torchnative.distributed.backend: rank %d dialled rank %d and "
                "was answered as rank %d" % (rank, peer, tag))
        if msg.get("t") == "refuse":
            raise _key_refusal(rank, peer, "rank %d refused it: %s"
                               % (peer, msg.get("why")))
        return msg

    nonce_c = _server.new_nonce()
    try:
        _hs_send(conn, rank, t="hello", v=1, token=token, auth=key is not None,
                 nonce=_hex(nonce_c))
    except OSError as exc:
        raise RuntimeError(
            "torchnative.distributed.backend: rank %d's greeting to rank %d "
            "could not be sent; rank %d closed the connection before reading "
            "it (%s: %s)" % (rank, peer, peer, type(exc).__name__, exc)) from exc
    msg = recv()
    kind = msg.get("t")
    if key is None:
        if kind == "accept":
            return
        _hs_send(conn, rank, t="refuse", why="rank %d holds no key" % rank)
        raise _key_refusal(rank, peer, "rank %d holds a key and this rank "
                                       "holds none" % peer)
    if kind != "challenge":
        _hs_send(conn, rank, t="refuse", why="rank %d holds a key" % rank)
        raise _key_refusal(rank, peer, "this rank holds a key and rank %d "
                                       "answered without one" % peer)
    try:
        nonce_a = _unhex(msg["nonce"])
        proof_a = _unhex(msg["proof"])
    except (KeyError, ValueError):
        _hs_send(conn, rank, t="refuse", why="a malformed challenge")
        raise _key_refusal(rank, peer, "rank %d sent a malformed challenge"
                           % peer) from None
    want = _server.proof(key, b"accept", token, rank, peer, nonce_c)
    if not _server.verify(want, proof_a):
        _hs_send(conn, rank, t="refuse",
                 why="rank %d's proof of the key does not verify on rank %d"
                     % (peer, rank))
        raise _key_refusal(rank, peer, "rank %d's proof of the pre-shared key "
                                       "does not verify; the two ranks hold "
                                       "different keys" % peer)
    _hs_send(conn, rank, t="proof",
             proof=_hex(_server.proof(key, b"connect", token, rank, peer,
                                      nonce_a)))
    msg = recv()
    if msg.get("t") != "ok":
        raise _key_refusal(rank, peer, "rank %d answered %r where ok belongs"
                           % (peer, msg.get("t")))


def _accept_handshake(conn, rank, token, key, seconds):
    """The acceptor's half. Returns the peer's rank, or raises ``_Stranger`` / ``_HandshakeFailed``."""
    try:
        peer, hello = _hs_recv(conn, seconds)
    except Exception:  # noqa: BLE001 -- garbage, a scanner, a timeout
        raise _Stranger() from None
    if (hello.get("t") != "hello" or hello.get("token") != token
            or not isinstance(peer, int) or not 0 <= peer < rank):
        raise _Stranger()

    def fail(why, tell=None):
        if tell is not None:
            try:
                _hs_send(conn, rank, t="refuse", why=tell)
            except Exception:  # noqa: BLE001 -- it may be gone already
                pass
        return _HandshakeFailed(peer, _key_refusal(rank, peer, why))

    theirs = bool(hello.get("auth"))
    if key is None:
        if theirs:
            raise fail("rank %d holds a key and this rank holds none" % peer,
                       tell="rank %d holds no key" % rank)
        _hs_send(conn, rank, t="accept", v=1)
        return peer
    if not theirs:
        raise fail("this rank holds a key and rank %d holds none" % peer,
                   tell="rank %d holds a key" % rank)
    try:
        nonce_c = _unhex(hello["nonce"])
    except (KeyError, ValueError):
        raise fail("rank %d sent a malformed hello" % peer,
                   tell="a malformed hello") from None
    nonce_a = _server.new_nonce()
    _hs_send(conn, rank, t="challenge", v=1, nonce=_hex(nonce_a),
             proof=_hex(_server.proof(key, b"accept", token, peer, rank,
                                      nonce_c)))
    try:
        tag, msg = _hs_recv(conn, seconds)
    except Exception as exc:  # noqa: BLE001
        raise _HandshakeFailed(peer, RuntimeError(
            "torchnative.distributed.backend: rank %d sent no proof of the "
            "pre-shared key to rank %d (%s: %s)"
            % (peer, rank, type(exc).__name__, exc))) from None
    if tag != peer:
        raise fail("a proof arrived signed as rank %r" % (tag,),
                   tell="a proof from the wrong rank")
    if msg.get("t") == "refuse":
        raise fail("rank %d refused it: %s" % (peer, msg.get("why")))
    if msg.get("t") != "proof":
        raise fail("rank %d sent %r where a proof belongs" % (peer, msg.get("t")),
                   tell="no proof")
    try:
        proof_c = _unhex(msg["proof"])
    except (KeyError, ValueError):
        raise fail("rank %d sent a malformed proof" % peer,
                   tell="a malformed proof") from None
    want = _server.proof(key, b"connect", token, peer, rank, nonce_a)
    if not _server.verify(want, proof_c):
        raise fail("rank %d's proof of the pre-shared key does not verify; the "
                   "two ranks hold different keys" % peer,
                   tell="rank %d's proof of the key does not verify on rank %d"
                        % (peer, rank))
    _hs_send(conn, rank, t="ok")
    return peer


# -- the mesh ----------------------------------------------------------------------

class Mesh:
    """One connection per rank pair, built at construction, closed by ``close()``."""

    def __init__(self, rank, size, store, timeout_s, force=None, bind_addr=None,
                 key=None):
        self.rank = rank
        self.size = size
        self.timeout_s = float(timeout_s)
        self.force = forced_kind(force)
        self.mailbox = Mailbox(rank)
        self.unix_path = None
        self._conns = {}
        self._kinds = {}
        self._senders = {}
        self._readers = {}
        self._closed = False
        self._close_lock = threading.Lock()
        #: D10: the pre-shared key, or None. Held only here; it is never put
        #: on the wire or into the store, only HMACs over nonces are.
        self._key = _server.resolve_key(key)
        self.authenticated = self._key is not None
        if size > 1:
            # Before any socket or store traffic: a LAN bind without the
            # opt-in refuses here, whether or not the address exists.
            bind = bind_addr or os.environ.get(ENV_BIND) or LOOPBACK
            _server.check_bind(bind, "rank %d's mesh listener" % rank)
            if store is None:
                _refuse("Mesh",
                        "a world of %d needs a Store to exchange addresses "
                        "through, and none was passed" % size)
            self._build(store, bind_addr)

    # -- building --------------------------------------------------------------
    def _build(self, store, bind_addr):
        rank, size = self.rank, self.size
        deadline = time.monotonic() + self.timeout_s
        if rank == 0:
            token = uuid.uuid4().hex
            store.set(_KEY_GROUP, token.encode("utf-8"))
        else:
            token = _store_get(store, _KEY_GROUP, deadline, rank,
                               "the group token rank 0 publishes").decode("utf-8")
        arrivals = queue.Queue()
        greet_timeout = min(self.timeout_s, 60.0)
        heard = set()

        def on_conn(conn):
            # Runs on a listener's thread with a connection nobody has named
            # yet. One that does not greet as a rank of this group (another
            # group, a port scanner) is a stranger and is dropped, not
            # trusted. One that greets as a rank of this group but fails the
            # pre-shared key check is a refusal, raised on the build thread.
            try:
                peer = _accept_handshake(conn, rank, token, self._key,
                                         greet_timeout)
            except _Stranger:
                conn.close()
                return
            except _HandshakeFailed as exc:
                conn.close()
                arrivals.put((exc.peer, exc.error))
                return
            if isinstance(conn, SocketConnection):
                conn.settimeout(None)
            conn.peer = peer
            arrivals.put((peer, conn))

        bind = bind_addr or os.environ.get(ENV_BIND) or LOOPBACK
        listeners = []
        try:
            listeners.append(InprocTransport(token, rank, on_conn))
            unix = None
            if hasattr(socket, "AF_UNIX"):
                try:
                    unix = UnixTransport(rank, on_conn, greet_timeout)
                except OSError:
                    unix = None  # e.g. a temp path longer than sun_path
                else:
                    listeners.append(unix)
                    self.unix_path = unix.path
            tcp = TcpTransport(rank, on_conn, greet_timeout, bind)
            listeners.append(tcp)

            mine = {"rank": rank, "host": host_id(), "process": process_token(),
                    "pid": os.getpid(), "inproc": token,
                    "shm": unix.address() if unix is not None else None,
                    "tcp": tcp.address()}
            store.set(_KEY_ADDR % rank, json.dumps(mine).encode("utf-8"))
            records = {}
            for peer in range(size):
                if peer == rank:
                    continue
                raw = _store_get(store, _KEY_ADDR % peer, deadline, rank,
                                 "rank %d's addresses" % peer)
                records[peer] = json.loads(raw.decode("utf-8"))
            for peer, theirs in records.items():
                self._kinds[peer] = choose(mine, theirs, self.force)

            for peer in range(rank + 1, size):
                conn = self._connect(self._kinds[peer], records[peer], peer,
                                     deadline)
                conn.peer = peer
                try:
                    _connect_handshake(conn, rank, peer, token, self._key,
                                       deadline)
                except BaseException:
                    conn.close()
                    raise
                if isinstance(conn, SocketConnection):
                    conn.settimeout(None)
                self._conns[peer] = conn

            while len(self._conns) < size - 1:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    missing = [p for p in range(rank) if p not in self._conns]
                    raise RuntimeError(
                        "torchnative.distributed.backend: rank %d heard from "
                        "every rank except %s within the group's timeout of "
                        "%.1f s" % (rank, missing, self.timeout_s))
                try:
                    peer, conn = arrivals.get(timeout=min(remaining, 1.0))
                except queue.Empty:
                    continue
                heard.add(peer)
                if isinstance(conn, BaseException):
                    raise conn
                if peer in self._conns:
                    conn.close()
                    raise RuntimeError(
                        "torchnative.distributed.backend: rank %d connected to "
                        "rank %d twice" % (peer, rank))
                if conn.kind != self._kinds[peer]:
                    conn.close()
                    _refuse("Mesh",
                            "rank %d connected to rank %d over %s, but rank %d "
                            "chose %s for that pair. The two ranks disagree "
                            "about %s, which must be the same on every rank"
                            % (peer, rank, conn.kind, rank, self._kinds[peer],
                               ENV_TRANSPORT))
                self._conns[peer] = conn
        except BaseException as exc:
            for conn in self._conns.values():
                conn.close()
            self._conns.clear()
            if isinstance(exc, Exception):
                self._linger(arrivals, heard, deadline)
            raise
        finally:
            for listener in listeners:
                listener.close()

        for peer, conn in self._conns.items():
            outbox = queue.Queue()
            sender = threading.Thread(
                target=self._send_loop, args=(peer, conn, outbox), daemon=True,
                name="torchnative-p2p-r%d-send-to%d" % (rank, peer))
            reader = threading.Thread(
                target=self._read_loop, args=(peer, conn), daemon=True,
                name="torchnative-p2p-r%d-read-from%d" % (rank, peer))
            self._senders[peer] = (outbox, sender)
            self._readers[peer] = reader
            sender.start()
            reader.start()

    def _linger(self, arrivals, heard, deadline):
        """Keep the listeners answering for a moment after this rank failed.

        A rank that fails (say rank 2 refused its key) would otherwise close
        its listeners at once, and a lower rank whose connection is still in
        the backlog gets a bare broken pipe instead of this rank's by-name
        refusal (Linux gate, run 43302327796). The handshakes run on the
        listener threads, so waiting here lets each lower rank that has not
        been heard from reach its own verdict. Bounded by ``LINGER_S`` and by
        the group's deadline, so a rank that timed out waits no longer.
        """
        end = min(time.monotonic() + LINGER_S, deadline)
        waiting = set(range(self.rank)) - heard
        while waiting:
            remaining = end - time.monotonic()
            if remaining <= 0:
                return
            try:
                peer, conn = arrivals.get(timeout=min(remaining, 0.2))
            except queue.Empty:
                continue
            waiting.discard(peer)
            if not isinstance(conn, BaseException):
                conn.close()

    def _connect(self, kind, theirs, peer, deadline):
        timeout = max(1.0, deadline - time.monotonic())
        try:
            if kind == "inproc":
                return InprocTransport.connect(theirs["inproc"], peer)
            if kind == "shm":
                return UnixTransport.connect(theirs["shm"], timeout)
            return TcpTransport.connect(theirs["tcp"], timeout)
        except OSError as exc:
            raise RuntimeError(
                "torchnative.distributed.backend: rank %d could not reach rank "
                "%d over %s (%r): %s: %s"
                % (self.rank, peer, kind, theirs.get(kind), type(exc).__name__,
                   exc)) from exc

    # -- threads ---------------------------------------------------------------
    def _read_loop(self, peer, conn):
        while True:
            try:
                frame = conn.recv_frame()
            except BaseException as exc:  # noqa: BLE001
                if self._closed:
                    return
                if isinstance(exc, wire.WireError):
                    self.mailbox.peer_gone(peer, wire.WireError, str(exc))
                elif isinstance(exc, wire.PeerGone):
                    self.mailbox.peer_gone(peer, wire.PeerGone, str(exc))
                else:
                    self.mailbox.peer_gone(
                        peer, wire.PeerGone,
                        "torchnative.distributed.backend: rank %d is gone: "
                        "connection closed (%s: %s)"
                        % (peer, type(exc).__name__, exc))
                return
            if frame.kind == wire.KIND_CONTROL:
                continue
            self.mailbox.deliver(peer, frame)

    def _send_loop(self, peer, conn, outbox):
        while True:
            item = outbox.get()
            if item is None:
                return
            work, data = item
            try:
                conn.send_frame(data)
            except BaseException as exc:  # noqa: BLE001
                work._fail(wire.PeerGone(
                    "torchnative.distributed.backend: send to rank %d failed, "
                    "connection closed (%s: %s)"
                    % (peer, type(exc).__name__, exc)))
            else:
                work._sent()

    # -- the operations the backend calls ----------------------------------------
    def kind_of(self, peer):
        """``"inproc"``, ``"shm"`` or ``"tcp"``: the transport to ``peer``."""
        if peer == self.rank:
            raise ValueError(
                "torchnative.distributed.backend: rank %d is this rank; no "
                "transport joins a rank to itself" % peer)
        kind = self._kinds.get(peer)
        if kind is None:
            raise ValueError(
                "torchnative.distributed.backend: rank %r is not in this group "
                "of size %d" % (peer, self.size))
        return kind

    def send(self, dst, tag, data, tensor=None):
        """Queue one encoded frame for ``dst``. Returns a ``SendWork``."""
        work = SendWork(dst, tag, self.timeout_s, tensor)
        if dst == self.rank:
            # upstream's isend allows src == dst; the frame goes straight
            # into this rank's own mailbox, decoded exactly as from a peer.
            self.mailbox.deliver(self.rank, wire.decode(data, self.rank))
            work._sent()
            return work
        if self._closed:
            work._fail(RuntimeError(
                "torchnative.distributed.backend: send to rank %d after the "
                "group was shut down" % dst))
            return work
        self._senders[dst][0].put((work, data))
        return work

    def recv(self, src, tag, dest):
        """Post a receive of one tensor from ``src`` with ``tag``. Returns a ``RecvWork``."""
        work = RecvWork(dest, src, tag, self.timeout_s, self.mailbox)
        self.mailbox.post(work)
        return work

    def close(self):
        """Flush queued sends, close every connection, fail waiting receives. Idempotent."""
        with self._close_lock:
            if self._closed:
                return
            self._closed = True
        for outbox, _thread in self._senders.values():
            outbox.put(None)
        for _outbox, thread in self._senders.values():
            thread.join(timeout=min(self.timeout_s, 30.0))
        for conn in self._conns.values():
            conn.close()
        self.mailbox.close(
            "torchnative.distributed.backend: the group was shut down while "
            "this receive was waiting")
