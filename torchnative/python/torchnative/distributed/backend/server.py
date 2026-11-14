"""The swappable server (coordinator), the LAN opt-in and the pre-shared key.

BACKEND.md D4 (owner, 2026-11-08: "서버 구현체를 갈아낄 수 있도록 설계해줘"):
the coordinator that ranks rendezvous through and exchange addresses through is
swappable behind an interface. D10 (leader): loopback by default, plus an opt-in
pre-shared key using stdlib ``hmac``. Those two sentences are the decisions;
everything below is an **agent proposal, not confirmed**: the class names
(``Server``, ``InProcessServer``, ``StandaloneServer``), the ``tnserver://``
scheme and its ``server=`` query, the store protocol, the environment variable
names (``TORCHNATIVE_DIST_KEY``, ``TORCHNATIVE_DIST_ALLOW_LAN``) and the
handshake message layout.

What a server is here
    A key/value store that every rank of a group can reach, with blocking
    ``get``/``wait`` that **time out** at the store's timeout (the group's
    timeout, which ``init_process_group`` hands it with ``set_timeout``).
    ``make_store(rank, world_size, timeout)`` returns a client that is a
    ``torch._C._distributed_c10d.Store``, so ``PrefixStore``, the backend
    creator and the mesh's address exchange use it unchanged. It is reached
    through ``init_process_group(init_method="tnserver://host:port")``, a
    handler registered with upstream's ``register_rendezvous_handler``: no new
    door, and ``env://``, ``tcp://`` and ``file://`` keep working.

``InProcessServer``
    The store served from a daemon thread of the calling process. With
    ``tnserver://host:port`` (the same as ``?server=rank0``) rank 0 hosts one
    and every rank connects to it, the way ``tcp://`` has rank 0 host a
    ``TCPStore``.

``StandaloneServer``
    The same server in its own Python process: ``python -m
    torchnative.distributed.backend.server --host H --port P`` or
    ``python path/to/server.py ...``. The module imports only the standard
    library at its top, so the server process needs no torch. Ranks reach it
    with ``tnserver://H:P?server=external``, and no rank hosts anything.
    ``?server=external`` also reaches an ``InProcessServer`` the caller
    started itself; ``Server.url()`` returns that form.

The PythonMultiplatform app-host server and a remote server (D4) are not built.

Federated server-side aggregation (BACKEND.md D14, §9 step 15; issue #282)
    Both implementations also serve two federated operations, through the
    same listener: ``fed_gather`` (every rank of a round hands in one integer
    and gets every rank's back, in rank order) and ``fed_fold`` (every rank
    hands in a weight and a table of tensors and gets back the sum of the
    weights and the element-wise sum of the tables, folded in **ascending rank
    order**, ``((x_0 + x_1) + x_2) + ...``, which is the order
    ``ProcessGroupLocal``'s hub folds an ``all_reduce`` in). The request header
    is the store protocol's length-prefixed JSON and carries only metadata;
    the tensors follow it as ``wire.py`` frames, raw little-endian bytes, and
    the reply's tensors come back the same way. The fold uses only the
    standard library (``array``), so the standalone server still needs no
    torch. Every completed operation is entered in a ledger
    (``Server.fed_ledger()``) with the ranks folded and a SHA-256 of the
    result's bytes, which is the server-side evidence a test asserts on. A
    round that is missing a rank at the timeout produces no aggregate for
    anybody and names the missing ranks. Everything in this paragraph is an
    **agent proposal, not confirmed** (docs/distributed/FEDSERVER.md).

Security (D10, BACKEND.md §4.2 "Security" and §8)
    * Every listener this backend opens binds loopback unless told otherwise.
      A non-loopback address refuses **before any socket is bound** unless
      ``TORCHNATIVE_DIST_ALLOW_LAN=1`` acknowledges that hosts on the network
      can reach it (``check_bind``).
    * ``TORCHNATIVE_DIST_KEY`` (or a ``key=`` argument) turns on a
      challenge-response on every **mesh** connection (``transport.Mesh``):
      both ends send a fresh ``secrets`` nonce and prove knowledge of the key
      with HMAC-SHA-256 over the other end's nonce, compared with
      ``hmac.compare_digest``. The key itself never goes on the wire and never
      into the store. A wrong key, or a key on one side only, refuses by name on
      both sides, naming the peer rank.
    * **This authenticates; it does not encrypt.** Frames after the handshake
      are plaintext, and a party that can read the network reads the tensors.
    * **The store is not authenticated.** It holds the group token and each
      rank's listening addresses, none of which is secret; a forged address
      record leads a rank to a listener that cannot pass the handshake. A party
      that can reach the store can still disrupt rendezvous (a denial of
      service), which the key does not prevent.
    * The collectives' star (``ProcessGroupLocal``) binds and connects
      ``127.0.0.1`` only and is not covered by the key; it never leaves the
      host. When the collectives move onto the mesh (BACKEND.md §9 step 3) they
      inherit the handshake.

Graded *builds* only: nothing here has been run (AGENTS.md §16).
"""

from __future__ import annotations

import array
import base64
import hashlib
import hmac
import json
import os
import secrets
import socket
import struct
import subprocess
import sys
import threading
import time

#: Agent proposal: the pre-shared key (D10).
ENV_KEY = "TORCHNATIVE_DIST_KEY"
#: Agent proposal: the explicit acknowledgement a non-loopback bind needs (D10).
ENV_ALLOW_LAN = "TORCHNATIVE_DIST_ALLOW_LAN"
#: Agent proposal: the rendezvous scheme.
SCHEME = "tnserver"

LOOPBACK = "127.0.0.1"

#: Store keys of the arrival barrier (agent proposal).
KEY_ARRIVED = "tn_server/arrived/%d"

#: A store request or reply larger than this is a stranger, not a rank.
_MAX_MESSAGE = 64 << 20

#: The line a standalone server prints once it listens.
_READY = "TNSERVER-LISTENING"

__all__ = [
    "Server", "InProcessServer", "StandaloneServer", "connect_store",
    "check_bind", "resolve_key", "proof", "rendezvous_barrier",
    "register_rendezvous_handler", "BindRefused", "AuthRefused",
    "RendezvousTimeout", "ENV_KEY", "ENV_ALLOW_LAN", "SCHEME",
    "fed_request", "FOLD_DTYPES",
]


class BindRefused(RuntimeError):
    """A non-loopback bind without ``TORCHNATIVE_DIST_ALLOW_LAN=1``."""


class AuthRefused(RuntimeError):
    """The pre-shared key handshake failed. The message names the peer rank."""


class RendezvousTimeout(RuntimeError):
    """A rank did not arrive within the group's timeout. Names that rank."""


# -- D10: the bind rule and the key ------------------------------------------------

def is_loopback(host):
    host = str(host)
    return host in ("localhost", "::1") or host.startswith("127.")


def lan_allowed():
    return os.environ.get(ENV_ALLOW_LAN, "").strip() == "1"


def check_bind(host, what):
    """Refuse a non-loopback ``host`` unless the LAN opt-in is set.

    Called before the socket is created, so the refusal cannot depend on
    whether the address exists on this machine. A host *name* other than
    ``localhost`` counts as non-loopback even if it resolves to 127.x: the
    rule reads what was asked for, not what DNS says today.
    """
    if is_loopback(host) or lan_allowed():
        return
    raise BindRefused(
        "torchnative.distributed: %s would bind %r, which is not a loopback "
        "address. A non-loopback bind is an explicit opt-in (BACKEND.md D10): "
        "set %s=1 to accept that hosts on the network can reach this "
        "listener, and set %s on every rank to require a pre-shared key. The "
        "key authenticates connections; nothing is encrypted"
        % (what, host, ENV_ALLOW_LAN, ENV_KEY))


def resolve_key(key=None):
    """The pre-shared key as bytes, or None when authentication is off.

    ``key`` wins over ``TORCHNATIVE_DIST_KEY``. A key that is set but empty
    refuses: an empty HMAC key is a configuration error, not "no key".
    """
    source = "the key= argument (which overrides %s)" % ENV_KEY
    if key is None:
        key = os.environ.get(ENV_KEY)
        source = ENV_KEY
        if key is None:
            return None
    if isinstance(key, str):
        key = key.encode("utf-8")
    key = bytes(key)
    if not key:
        raise ValueError(
            "torchnative.distributed: %s is set but empty. Unset it to run "
            "without a pre-shared key, or give every rank the same non-empty "
            "key" % source)
    return key


def proof(key, label, token, connector, acceptor, nonce):
    """HMAC-SHA-256 binding the key to one direction of one handshake.

    ``label`` is ``b"connect"`` or ``b"accept"``, so one side's proof can
    never be replayed as the other's; the group token and both ranks bind it
    to this pair of this group; ``nonce`` is the *other* side's fresh random
    challenge, so a recorded proof does not verify in a later handshake.
    """
    if isinstance(token, str):
        token = token.encode("utf-8")
    message = b"|".join([b"tn-mesh-auth-v1", label, token,
                         str(int(connector)).encode(), str(int(acceptor)).encode(),
                         nonce])
    return hmac.new(key, message, hashlib.sha256).digest()


def new_nonce():
    return secrets.token_bytes(32)


def verify(expected, got):
    return hmac.compare_digest(expected, got)


# -- the store protocol -------------------------------------------------------------
#
# Length-prefixed JSON (4-byte big-endian length), one request and one reply per
# connection. Values are base64. Rendezvous exchanges a few hundred bytes per
# rank, so text here does not contradict the binary data wire of BACKEND.md §4.2.

def _send_msg(sock, obj):
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("!I", len(data)) + data)


def _recv_exact(sock, n):
    chunks, got = [], 0
    while got < n:
        piece = sock.recv(n - got)
        if not piece:
            raise ConnectionError("connection closed after %d of %d bytes"
                                  % (got, n))
        chunks.append(piece)
        got += len(piece)
    return b"".join(chunks)


def _recv_msg(sock):
    (length,) = struct.unpack("!I", _recv_exact(sock, 4))
    if length > _MAX_MESSAGE:
        raise ConnectionError("message of %d bytes is over the %d-byte limit"
                              % (length, _MAX_MESSAGE))
    return json.loads(_recv_exact(sock, length).decode("utf-8"))


def raw_request(host, port, req, timeout=10.0):
    """One request to a server, without a ``Store`` (no torch needed). Returns the reply."""
    with socket.create_connection((host, int(port)), timeout=timeout) as sock:
        _send_msg(sock, req)
        return _recv_msg(sock)


def _b64(data):
    return base64.b64encode(bytes(data)).decode("ascii")


def _unb64(text):
    return base64.b64decode(text.encode("ascii"))


class _StoreCore:
    """The dictionary and the condition every server implementation serves."""

    def __init__(self):
        self._data = {}
        self._cond = threading.Condition()

    def _wait_for(self, keys, timeout):
        deadline = None if timeout is None else time.monotonic() + float(timeout)
        while True:
            missing = [k for k in keys if k not in self._data]
            if not missing:
                return []
            if deadline is None:
                self._cond.wait()
                continue
            left = deadline - time.monotonic()
            if left <= 0:
                return missing
            self._cond.wait(left)

    def handle(self, req):
        op = req.get("op")
        with self._cond:
            if op == "set":
                self._data[req["key"]] = _unb64(req["value"])
                self._cond.notify_all()
                return {"ok": True}
            if op == "get":
                missing = self._wait_for([req["key"]], req.get("timeout"))
                if missing:
                    return {"ok": False, "kind": "timeout", "missing": missing}
                return {"ok": True, "value": _b64(self._data[req["key"]])}
            if op == "wait":
                missing = self._wait_for(list(req["keys"]), req.get("timeout"))
                if missing:
                    return {"ok": False, "kind": "timeout", "missing": missing}
                return {"ok": True}
            if op == "check":
                return {"ok": True,
                        "value": all(k in self._data for k in req["keys"])}
            if op == "add":
                total = int(self._data.get(req["key"], b"0")) + int(req["amount"])
                self._data[req["key"]] = str(total).encode("ascii")
                self._cond.notify_all()
                return {"ok": True, "value": total}
            if op == "compare_set":
                key = req["key"]
                expected = _unb64(req["expected"])
                desired = _unb64(req["desired"])
                current = self._data.get(key)
                if current is None:
                    if expected == b"":
                        self._data[key] = desired
                        self._cond.notify_all()
                        return {"ok": True, "value": _b64(desired)}
                    return {"ok": True, "value": _b64(expected)}
                if current == expected:
                    self._data[key] = desired
                    self._cond.notify_all()
                    return {"ok": True, "value": _b64(desired)}
                return {"ok": True, "value": _b64(current)}
            if op == "delete_key":
                return {"ok": True,
                        "value": self._data.pop(req["key"], None) is not None}
            if op == "num_keys":
                return {"ok": True, "value": len(self._data)}
            if op == "list_keys":
                return {"ok": True, "value": sorted(self._data)}
            if op == "ping":
                return {"ok": True, "value": "pong"}
        return {"ok": False, "kind": "error", "msg": "unknown op %r" % (op,)}

    def snapshot(self):
        """A copy of every key and value. For tests and diagnostics."""
        with self._cond:
            return dict(self._data)


# -- federated server-side aggregation (D14; agent proposal) -------------------------

_WIRE = []


def _wire():
    """``wire.py``, the backend's binary frame, without importing torch.

    The relative import works when this file is the package module. When it
    runs as a script (``StandaloneServer`` starts this file directly) or is
    loaded by path, there is no package, so ``wire.py`` is loaded from beside
    this file. ``wire.py`` imports only ``struct`` and ``sys`` at its top.
    """
    if _WIRE:
        return _WIRE[0]
    try:
        from . import wire as module  # type: ignore[no-redef]
    except ImportError:
        import importlib.util
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "wire.py")
        spec = importlib.util.spec_from_file_location("_tn_server_wire", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    _WIRE.append(module)
    return module


#: The dtypes the server folds, with their ``array`` type codes. Python's float
#: is IEEE binary64, and one ``+`` of two binary32 values computed in binary64
#: and rounded back to binary32 is exactly the binary32 sum, because 53 >= 2*24
#: + 2 (the double-rounding bound). So ``array('f')`` folds float32 bit for bit
#: like ``Tensor.add_`` does, with no numpy. float16 and bfloat16 would need
#: their own rounding code and are refused by name.
FOLD_DTYPES = {"float32": "f", "float64": "d"}

#: The federated operations the listener serves besides the store's.
_FED_OPS = ("fed_gather", "fed_fold", "fed_ledger")

#: The most tensor bytes one ``fed_fold`` request may carry. A request past it
#: is a stranger or a model this server was never measured with.
_MAX_FOLD_BYTES = 1 << 30

#: How many closed rounds the server remembers, so a rank that arrives after
#: the verdict gets that verdict instead of opening the round again.
_FED_TOMBSTONES = 4096


class _FedRefused(Exception):
    """A federated round the server will not aggregate. The text says why."""


class _FedRound:
    __slots__ = ("kind", "world", "contrib", "result", "error", "served")

    def __init__(self, kind, world):
        self.kind = kind
        self.world = world
        self.contrib = {}
        self.result = None
        self.error = None
        self.served = set()


def _as_array(code, payload):
    out = array.array(code)
    out.frombytes(bytes(payload))
    if sys.byteorder != "little":
        out.byteswap()
    return out


def _le_bytes(values):
    if sys.byteorder != "little":
        values = array.array(values.typecode, values)
        values.byteswap()
    return values.tobytes()


def _gather_compute(key, world, contrib):
    ranks = list(range(world))
    reply = {"ok": True, "values": [contrib[r] for r in ranks], "ranks": ranks}
    entry = {"key": key, "kind": "gather", "ranks": ranks, "world": world}
    return reply, [], entry


def _fold_compute(key, world, contrib):
    """The weighted sum of one round, folded in ascending rank order.

    The order is the contract (``ProcessGroupLocal``'s class docstring): float
    addition is not associative, and the hub of the ``all_reduce`` path folds
    ``((x_0 + x_1) + x_2) + ...``. Doing the same here is what lets a test
    hold the two paths to bit equality rather than to a tolerance.
    """
    wire = _wire()
    ranks = list(range(world))
    schema = [(n, d, tuple(s)) for n, d, s, _ in contrib[0]["tensors"]]
    for r in ranks[1:]:
        other = [(n, d, tuple(s)) for n, d, s, _ in contrib[r]["tensors"]]
        if other != schema:
            raise _FedRefused(
                "rank 0 and rank %d submitted different tables to %r: %s "
                "versus %s. The server sums element by element and would "
                "return a number for tables that do not line up, so the round "
                "has no aggregate" % (r, key, schema, other))
    total = contrib[0]["weight"]
    for r in ranks[1:]:
        total = total + contrib[r]["weight"]
    frames = []
    digest = hashlib.sha256()
    elements = 0
    for i, (name, dtype, shape) in enumerate(schema):
        code = FOLD_DTYPES.get(dtype)
        if code is None:
            raise _FedRefused(
                "%r is torch.%s, and the server folds only %s. A float16 or "
                "bfloat16 sum needs its own rounding step, which this fold "
                "does not have; adding in float64 and rounding once at the "
                "end would be a different number from the all_reduce path's"
                % (name, dtype, ", ".join("torch." + d for d in FOLD_DTYPES)))
        acc = _as_array(code, contrib[0]["tensors"][i][3])
        for r in ranks[1:]:
            nxt = _as_array(code, contrib[r]["tensors"][i][3])
            # One rounding per addition, in the element's own width: storing
            # into array(code) rounds the binary64 sum back to it.
            acc = array.array(code, [a + b for a, b in zip(acc, nxt)])
        payload = _le_bytes(acc)
        digest.update(payload)
        elements += len(acc)
        frames.append(wire.encode(wire.KIND_TENSOR, i, payload, dtype=dtype,
                                  shape=shape))
    names = [n for n, _, _ in schema]
    reply = {"ok": True, "total_weight": total.hex(), "ranks": ranks,
             "names": names, "sha256": digest.hexdigest()}
    entry = {"key": key, "kind": "fold", "ranks": ranks, "world": world,
             "names": names, "elements": elements,
             "weights": [contrib[r]["weight"].hex() for r in ranks],
             "total_weight": total.hex(), "sha256": digest.hexdigest()}
    return reply, frames, entry


class _FedCore:
    """The rounds in flight and the ledger of the ones the server completed.

    A round is a key every participating rank names. It completes when every
    rank of its ``world`` has handed in; then one thread computes the result
    (in ascending rank order) and every rank gets the same reply. A round still
    missing a rank at the request's timeout completes **for nobody**: each rank
    that did hand in is told which ranks did not, and no partial aggregate is
    ever computed. That is ``FederatedDataParallel(on_missing='refuse')``.
    """

    def __init__(self):
        self._cond = threading.Condition()
        self._rounds = {}
        self._closed = {}
        self._closed_order = []
        self.ledger = []

    def ledger_copy(self):
        with self._cond:
            return [dict(e) for e in self.ledger]

    def _retire(self, key, rnd):
        self._rounds.pop(key, None)
        if rnd.error is not None:
            verdict = dict(rnd.error)
        else:
            verdict = {"ok": False, "kind": "refused",
                       "msg": "round %r is already complete; a second "
                              "submission to it is not a participant of it"
                              % (key,)}
        self._closed[key] = verdict
        self._closed_order.append(key)
        while len(self._closed_order) > _FED_TOMBSTONES:
            self._closed.pop(self._closed_order.pop(0), None)

    def join(self, kind, key, rank, world, data, timeout, compute):
        """Hand in ``data`` for ``rank``; wait for the round. Returns ``(reply, frames)``."""
        deadline = time.monotonic() + float(timeout)
        with self._cond:
            if key in self._closed:
                return dict(self._closed[key]), []
            rnd = self._rounds.get(key)
            if rnd is None:
                rnd = self._rounds[key] = _FedRound(kind, world)
            problem = None
            if rnd.kind != kind or rnd.world != world:
                problem = ("rank %d sent a %s for a world of %d to %r, which "
                           "is a %s for a world of %d"
                           % (rank, kind, world, key, rnd.kind, rnd.world))
            elif not 0 <= rank < world:
                problem = "rank %d is not in a world of %d" % (rank, world)
            elif rank in rnd.contrib:
                problem = "rank %d submitted twice to %r" % (rank, key)
            if problem is not None:
                # One participant's mistake is the round's: nobody gets an
                # aggregate that might be missing or doubling a contribution.
                if rnd.result is None and rnd.error is None:
                    rnd.error = {"ok": False, "kind": "refused", "msg": problem}
                    self._cond.notify_all()
                return {"ok": False, "kind": "refused", "msg": problem}, []
            if rnd.result is None and rnd.error is None:
                rnd.contrib[rank] = data
                if len(rnd.contrib) == world:
                    try:
                        reply, frames, entry = compute(key, world, rnd.contrib)
                        rnd.result = (reply, frames)
                        self.ledger.append(entry)
                    except _FedRefused as exc:
                        rnd.error = {"ok": False, "kind": "refused",
                                     "msg": str(exc)}
                    self._cond.notify_all()
            while rnd.result is None and rnd.error is None:
                left = deadline - time.monotonic()
                if left <= 0:
                    missing = [r for r in range(world) if r not in rnd.contrib]
                    rnd.error = {"ok": False, "kind": "missing",
                                 "missing": missing, "seconds": float(timeout)}
                    self.ledger.append({"key": key, "kind": kind + "-abandoned",
                                        "ranks": sorted(rnd.contrib),
                                        "missing": missing, "world": world})
                    self._cond.notify_all()
                    break
                self._cond.wait(left)
            if rank in rnd.contrib:
                rnd.served.add(rank)
            if rnd.error is not None:
                reply, frames = dict(rnd.error), []
            else:
                reply, frames = dict(rnd.result[0]), list(rnd.result[1])
            if rnd.served >= set(rnd.contrib):
                self._retire(key, rnd)
            return reply, frames

    def handle(self, req, frames):
        """One ``fed_gather`` or ``fed_fold`` request. Returns ``(reply, frames)``."""
        op = req.get("op")
        try:
            key = str(req["key"])
            rank = int(req["rank"])
            world = int(req["world"])
            timeout = float(req["timeout"])
        except (KeyError, TypeError, ValueError) as exc:
            return {"ok": False, "kind": "refused",
                    "msg": "%s request without %s" % (op, exc)}, []
        if world < 2:
            return {"ok": False, "kind": "refused",
                    "msg": "a federated round over a world of %d aggregates "
                           "nothing" % (world,)}, []
        if op == "fed_gather":
            return self.join("gather", key, rank, world, int(req.get("value")),
                             timeout, _gather_compute)
        names = list(req.get("names") or [])
        if len(names) != len(frames):
            return {"ok": False, "kind": "refused",
                    "msg": "%d names and %d tensor frames" % (len(names),
                                                             len(frames))}, []
        tensors = []
        wire = _wire()
        for name, frame in zip(names, frames):
            if frame.kind != wire.KIND_TENSOR:
                return {"ok": False, "kind": "refused",
                        "msg": "%r did not arrive as a tensor frame" % (name,)}, []
            tensors.append((str(name), frame.dtype, tuple(frame.shape),
                            frame.payload))
        try:
            weight = float.fromhex(str(req["weight"]))
        except (KeyError, ValueError) as exc:
            return {"ok": False, "kind": "refused",
                    "msg": "fed_fold without a weight (%s)" % (exc,)}, []
        return self.join("fold", key, rank, world,
                         {"weight": weight, "tensors": tensors}, timeout,
                         _fold_compute)


def _capped_recv(sock, limit):
    """``sock.recv`` that refuses to read more than ``limit`` bytes in total."""
    used = [0]

    def recv(n):
        if used[0] + n > limit:
            raise ConnectionError("a federated request over %d bytes" % limit)
        data = sock.recv(min(n, 1 << 20))
        used[0] += len(data)
        return data
    return recv


def fed_request(host, port, req, frames=(), wait_s=None):
    """One federated request to the server at ``host:port``. Returns ``(reply, frames)``.

    The header ``req`` goes as the store protocol's JSON message and holds no
    tensor data; each of ``frames`` (``wire.py`` frames, ``bytes``) follows it
    raw. The reply is a JSON header followed by ``reply["frames"]`` frames,
    returned decoded (``wire.Frame``). ``wait_s`` is how long the server may
    hold the request while the round fills.
    """
    wire = _wire()
    req = dict(req)
    req["frames"] = len(frames)
    with socket.create_connection((host, int(port)), timeout=30.0) as sock:
        sock.settimeout(None if wait_s is None else float(wait_s) + 30.0)
        _send_msg(sock, req)
        for frame in frames:
            sock.sendall(frame)
        reply = _recv_msg(sock)
        out = [wire.read_frame(sock.recv, peer="the server")
               for _ in range(int(reply.get("frames", 0) or 0))]
    return reply, out


class _Listener:
    """A TCP listener that serves ``_StoreCore`` requests, one thread per connection."""

    def __init__(self, host, port, what):
        check_bind(host, what)
        family = socket.AF_INET6 if ":" in str(host) else socket.AF_INET
        self.core = _StoreCore()
        self.fed = _FedCore()
        self._sock = socket.socket(family, socket.SOCK_STREAM)
        try:
            self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self._sock.bind((host, int(port)))
            self._sock.listen(128)
            self._sock.settimeout(0.2)
        except BaseException:
            self._sock.close()
            raise
        self.host = host
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True,
                                        name="torchnative-server-accept")
        self._thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                conn, _addr = self._sock.accept()
            except socket.timeout:
                continue
            except OSError:
                return
            threading.Thread(target=self._one, args=(conn,), daemon=True,
                             name="torchnative-server-client").start()

    def _one(self, conn):
        try:
            # A client sends its request at once; one that does not within a
            # minute is a stranger and does not keep a thread forever.
            conn.settimeout(60.0)
            req = _recv_msg(conn)
            op = req.get("op") if isinstance(req, dict) else None
            if op in _FED_OPS:
                self._fed(conn, req)
                return
            conn.settimeout(None)
            reply = self.core.handle(req)
            _send_msg(conn, reply)
        except Exception:  # noqa: BLE001 -- a stranger or a client that left
            pass
        finally:
            try:
                conn.close()
            except OSError:
                pass

    def _fed(self, conn, req):
        """A federated request: its frames are read under the same 60 s limit."""
        if req["op"] == "fed_ledger":
            conn.settimeout(None)
            _send_msg(conn, {"ok": True, "value": self.fed.ledger_copy(),
                             "frames": 0})
            return
        wire = _wire()
        count = int(req.get("frames", 0) or 0)
        if not 0 <= count <= 65536:
            _send_msg(conn, {"ok": False, "kind": "refused", "frames": 0,
                             "msg": "%d frames in one request" % count})
            return
        recv = _capped_recv(conn, _MAX_FOLD_BYTES)
        frames = [wire.read_frame(recv, peer="rank %s" % req.get("rank"))
                  for _ in range(count)]
        conn.settimeout(None)
        reply, out = self.fed.handle(req, frames)
        reply["frames"] = len(out)
        _send_msg(conn, reply)
        for frame in out:
            conn.sendall(frame)

    def close(self):
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass


# -- the client ---------------------------------------------------------------------

def _seconds(timeout, default=300.0):
    if timeout is None:
        return default
    seconds = (timeout.total_seconds() if hasattr(timeout, "total_seconds")
               else float(timeout))
    return default if seconds <= 0 else seconds


class _ServerStoreMethods:
    """A ``Store`` whose every call is one request to a server at ``host:port``.

    ``get`` and ``wait`` block on the **server** until the key exists or the
    store's timeout passes, then raise ``RuntimeError`` naming the keys and the
    seconds. Mixed in front of the shim's ``Store`` by ``_store_class`` so the
    result passes as a ``torch._C._distributed_c10d.Store``.
    """

    def _tn_init(self, host, port, timeout):
        self.host = host
        self.port = int(port)
        self._timeout_s = _seconds(timeout)

    # The shim's Store keeps a timedelta; keep both views in step.
    def set_timeout(self, timeout):
        self._timeout_s = _seconds(timeout)
        try:
            super().set_timeout(timeout)
        except Exception:  # noqa: BLE001
            pass

    @property
    def timeout(self):
        import datetime
        return datetime.timedelta(seconds=self._timeout_s)

    def _request(self, req, wait_s=None):
        deadline = time.monotonic() + self._timeout_s
        last = None
        while True:
            left = deadline - time.monotonic()
            try:
                sock = socket.create_connection(
                    (self.host, self.port), timeout=max(0.5, min(left, 10.0)))
            except OSError as exc:
                # The server may not be listening yet (rank 0 hosts it and
                # may start after its peers): retry until the store's timeout.
                last = exc
                if time.monotonic() >= deadline:
                    raise RuntimeError(
                        "torchnative.distributed.backend.server: no server "
                        "answered at %s:%d within the store's timeout of "
                        "%.1f s (%s: %s)" % (self.host, self.port,
                                             self._timeout_s,
                                             type(last).__name__, last))
                time.sleep(0.05)
                continue
            try:
                sock.settimeout(None if wait_s is None else wait_s + 30.0)
                _send_msg(sock, req)
                return _recv_msg(sock)
            except (OSError, ConnectionError, ValueError) as exc:
                raise RuntimeError(
                    "torchnative.distributed.backend.server: the server at "
                    "%s:%d dropped the %s request (%s: %s)"
                    % (self.host, self.port, req.get("op"),
                       type(exc).__name__, exc)) from exc
            finally:
                sock.close()

    def _ok(self, reply, what):
        if reply.get("ok"):
            return reply
        if reply.get("kind") == "timeout":
            raise RuntimeError(
                "torchnative.distributed.backend.server: %s timed out after "
                "%.1f s (the store's timeout, which init_process_group sets to "
                "the group's); still missing: %s"
                % (what, self._timeout_s, reply.get("missing")))
        raise RuntimeError("torchnative.distributed.backend.server: %s failed: "
                           "%s" % (what, reply.get("msg")))

    @staticmethod
    def _bytes(value):
        return value.encode("utf-8") if isinstance(value, str) else bytes(value)

    def set(self, key, value):
        self._ok(self._request({"op": "set", "key": key,
                                "value": _b64(self._bytes(value))}),
                 "set(%r)" % key)

    def get(self, key):
        reply = self._ok(self._request(
            {"op": "get", "key": key, "timeout": self._timeout_s},
            wait_s=self._timeout_s), "get(%r)" % key)
        return _unb64(reply["value"])

    def wait(self, keys, timeout=None):
        seconds = _seconds(timeout, self._timeout_s)
        self._ok(self._request({"op": "wait", "keys": list(keys),
                                "timeout": seconds}, wait_s=seconds),
                 "wait(%r)" % (list(keys),))

    def check(self, keys):
        return bool(self._ok(self._request({"op": "check", "keys": list(keys)}),
                             "check")["value"])

    def add(self, key, value):
        return int(self._ok(self._request({"op": "add", "key": key,
                                           "amount": int(value)}),
                            "add(%r)" % key)["value"])

    def compare_set(self, key, expected_value, desired_value):
        reply = self._ok(self._request(
            {"op": "compare_set", "key": key,
             "expected": _b64(self._bytes(expected_value)),
             "desired": _b64(self._bytes(desired_value))}), "compare_set(%r)" % key)
        return _unb64(reply["value"])

    def delete_key(self, key):
        return bool(self._ok(self._request({"op": "delete_key", "key": key}),
                             "delete_key(%r)" % key)["value"])

    def multi_get(self, keys):
        return [self.get(k) for k in keys]

    def multi_set(self, keys, values):
        for key, value in zip(keys, values):
            self.set(key, value)

    def num_keys(self):
        return int(self._ok(self._request({"op": "num_keys"}), "num_keys")["value"])

    def list_keys(self):
        return list(self._ok(self._request({"op": "list_keys"}), "list_keys")["value"])

    def _no_queue(self, what):
        raise NotImplementedError(
            "torchnative.distributed.backend.server.ServerStore.%s: the server "
            "store has no queues yet; nothing in this backend uses them" % what)

    def queue_push(self, key, value):
        self._no_queue("queue_push")

    def queue_pop(self, key, block=True):
        self._no_queue("queue_pop")

    def queue_len(self, key):
        self._no_queue("queue_len")


_STORE_CLASS = []


def _store_class():
    """``ServerStore``: the methods above in front of the shim's ``Store``.

    Built on first use, not at import, so this module stays importable by a
    standalone server process that has no torch.
    """
    if not _STORE_CLASS:
        try:
            from torch._C._distributed_c10d import Store as base
        except Exception:  # noqa: BLE001 -- no torch: a plain client
            base = object
        _STORE_CLASS.append(type("ServerStore", (_ServerStoreMethods, base), {
            "__doc__": _ServerStoreMethods.__doc__,
            "__module__": __name__,
        }))
    return _STORE_CLASS[0]


def connect_store(host, port, timeout=None):
    """A store client for the server at ``host:port``."""
    store = _store_class()()
    store._tn_init(host, port, timeout)
    return store


# -- the interface and its two implementations ---------------------------------------

class Server:
    """The coordinator interface (BACKEND.md §4.5 layer 3; agent proposal).

    ``start()`` returns ``self`` once ranks can reach it; ``stop()`` is
    idempotent; ``address()`` is ``(host, port)``; ``url()`` is what to pass
    as ``init_method``; ``make_store(rank, world_size, timeout)`` is a client
    for one rank. Usable as a context manager.
    """

    kind = None

    def start(self):
        raise NotImplementedError

    def stop(self):
        raise NotImplementedError

    def address(self):
        raise NotImplementedError

    def url(self):
        host, port = self.address()
        if ":" in str(host):
            host = "[%s]" % host
        # Already running: no rank may host another one on the same port.
        return "%s://%s:%d?server=external" % (SCHEME, host, port)

    def make_store(self, rank, world_size, timeout=None):
        if not 0 <= int(rank) < int(world_size):
            raise ValueError(
                "torchnative.distributed.backend.server: rank %r is not in a "
                "world of size %r" % (rank, world_size))
        host, port = self.address()
        return connect_store(host, port, timeout)

    def fed_ledger(self):
        """Every federated operation this server completed or abandoned, oldest first.

        Each entry is a dict: ``key``, ``kind`` (``gather``, ``fold``, or
        ``<kind>-abandoned`` for a round that was missing a rank at its
        timeout), ``ranks`` (the ranks folded, in fold order), ``world``, and
        for a fold ``names``, ``elements``, ``weights`` and ``total_weight``
        (``float.hex``) and ``sha256`` (of the result's bytes). Read over the
        socket, so it is the server's own record for both implementations.
        """
        host, port = self.address()
        return raw_request(host, port, {"op": "fed_ledger"})["value"]

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()
        return False


class InProcessServer(Server):
    """The store served from a daemon thread of this process."""

    kind = "inprocess"

    def __init__(self, host=LOOPBACK, port=0):
        self._host = host
        self._port = int(port)
        self._listener = None

    def start(self):
        if self._listener is None:
            self._listener = _Listener(self._host, self._port,
                                       "the in-process server")
        return self

    def stop(self):
        if self._listener is not None:
            self._listener.close()
            self._listener = None

    def address(self):
        if self._listener is None:
            raise RuntimeError(
                "torchnative.distributed.backend.server.InProcessServer: not "
                "started")
        return (self._host, self._listener.port)

    def snapshot(self):
        """Every key and value the server holds. For tests and diagnostics."""
        return self._listener.core.snapshot()


class StandaloneServer(Server):
    """The same server as a separate Python process.

    ``start()`` runs this file with ``python`` (``sys.executable`` unless
    given) and waits for the line it prints once it listens. The child exits
    when ``stop()`` is called or when this process goes away (it watches its
    standard input for end of file), so a crashed parent leaves no server.
    """

    kind = "standalone"

    def __init__(self, host=LOOPBACK, port=0, python=None, start_timeout=60.0):
        self._host = host
        self._port = int(port)
        self._python = python or sys.executable
        self._start_timeout = float(start_timeout)
        self._proc = None
        self._bound = None
        self._lines = []

    def command(self):
        return [self._python, os.path.abspath(__file__), "--host",
                str(self._host), "--port", str(self._port), "--exit-with-stdin"]

    def start(self):
        if self._proc is not None:
            return self
        # Refuse here too, so a LAN bind without the opt-in is named in the
        # parent rather than only in a child's output.
        check_bind(self._host, "the standalone server")
        self._proc = subprocess.Popen(
            self.command(), stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT)
        ready = threading.Event()

        def drain():
            for raw in self._proc.stdout:
                line = raw.decode("utf-8", "replace").rstrip("\n")
                self._lines.append(line)
                parts = line.split()
                if len(parts) == 3 and parts[0] == _READY:
                    self._bound = (parts[1], int(parts[2]))
                    ready.set()
            ready.set()

        threading.Thread(target=drain, daemon=True,
                         name="torchnative-standalone-server-out").start()
        if not ready.wait(self._start_timeout) or self._bound is None:
            self.stop()
            raise RuntimeError(
                "torchnative.distributed.backend.server.StandaloneServer: the "
                "server process did not report a listening port within %.1f s. "
                "Its output:\n%s" % (self._start_timeout,
                                     "\n".join(self._lines[-50:])))
        return self

    def stop(self):
        proc, self._proc = self._proc, None
        if proc is None:
            return
        try:
            proc.stdin.close()
        except OSError:
            pass
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def pid(self):
        return None if self._proc is None else self._proc.pid

    def address(self):
        if self._bound is None:
            raise RuntimeError(
                "torchnative.distributed.backend.server.StandaloneServer: not "
                "started")
        return (self._host, self._bound[1])


# -- rendezvous ----------------------------------------------------------------------

#: The in-process servers rank 0 hosts through ``tnserver://``, kept alive for
#: the life of the process.
_HOSTED = []


def rendezvous_barrier(store, rank, size, timeout_s):
    """Every rank of the group writes its arrival and waits for every other one.

    Run first in the backend's constructor, so a rank that never arrives is
    named on every rank that did, within the group's timeout, instead of
    leaving them blocked inside a later step that cannot say who is missing.
    """
    if size <= 1:
        return
    deadline = time.monotonic() + float(timeout_s)
    store.set(KEY_ARRIVED % rank, b"1")
    for peer in range(size):
        if peer == rank:
            continue
        last = None
        while True:
            try:
                store.get(KEY_ARRIVED % peer)
                break
            except RuntimeError as exc:
                # The server's stores raise at their timeout; a HashStore
                # raises at once on an absent key and is polled.
                last = exc
                if time.monotonic() >= deadline:
                    raise RendezvousTimeout(
                        "torchnative.distributed.backend: rank %d waited %.1f "
                        "s (the group's timeout) at rendezvous for rank %d, "
                        "which never arrived. Ranks after %d were not checked. "
                        "Last store error: %s"
                        % (rank, float(timeout_s), peer, peer, last)) from None
                time.sleep(0.02)


def _query(query):
    out = {}
    for pair in filter(None, query.split("&")):
        name, _, value = pair.partition("=")
        out[name] = value
    return out


def _tnserver_handler(url, timeout=None, **kwargs):
    """``tnserver://host:port?rank=R&world_size=W[&server=rank0|external]``.

    ``rank0`` (the default): rank 0 starts an ``InProcessServer`` on
    ``host:port``. ``external``: a server is already running there (a
    ``StandaloneServer``, or an ``InProcessServer`` the caller started).
    """
    from urllib.parse import urlparse

    result = urlparse(url)
    if result.port is None or result.hostname is None:
        raise ValueError(
            "Error initializing torch.distributed using tnserver:// "
            "rendezvous: %r needs host:port" % (url,))
    query = _query(result.query)
    for name in ("rank", "world_size"):
        if name not in query:
            raise ValueError(
                "Error initializing torch.distributed using tnserver:// "
                "rendezvous: %s parameter missing" % name)
    rank, world_size = int(query["rank"]), int(query["world_size"])
    mode = query.get("server", "rank0")
    if mode == "rank0":
        if rank == 0:
            _HOSTED.append(InProcessServer(result.hostname, result.port).start())
    elif mode != "external":
        raise ValueError(
            "Error initializing torch.distributed using tnserver:// "
            "rendezvous: server=%r is not one of rank0, external" % mode)
    store = connect_store(result.hostname, result.port, timeout)
    yield (store, rank, world_size)
    raise RuntimeError("Unable to perform re-rendezvous using tnserver:// method")


def register_rendezvous_handler():
    """Register ``tnserver://`` with upstream, once. Returns the scheme."""
    # The module, not the attribute: `torch.distributed.rendezvous` is the
    # *function* `rendezvous`, which `torch.distributed` re-exports over the
    # submodule's name, so `from torch.distributed import rendezvous` (and
    # `import torch.distributed.rendezvous as m`) hand back that function.
    # Gate run 43290520768 failed every distributed suite on exactly that.
    import importlib
    rdzv = importlib.import_module("torch.distributed.rendezvous")
    if SCHEME not in rdzv._rendezvous_handlers:
        rdzv.register_rendezvous_handler(SCHEME, _tnserver_handler)
    return SCHEME


# -- the standalone process ----------------------------------------------------------

def main(argv=None):
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m torchnative.distributed.backend.server",
        description="Serve a torchnative rendezvous store (tnserver://).")
    parser.add_argument("--host", default=LOOPBACK)
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--exit-with-stdin", action="store_true",
                        help="exit when standard input reaches end of file")
    args = parser.parse_args(argv)
    try:
        listener = _Listener(args.host, args.port, "the standalone server")
    except BindRefused as exc:
        print("BindRefused: %s" % exc, flush=True)
        return 2
    print("%s %s %d" % (_READY, args.host, listener.port), flush=True)
    stop = threading.Event()
    if args.exit_with_stdin:
        def watch():
            try:
                while sys.stdin.buffer.read(4096):
                    pass
            except (OSError, ValueError):
                pass
            stop.set()
        threading.Thread(target=watch, daemon=True).start()
    try:
        while not stop.wait(1.0):
            pass
    except KeyboardInterrupt:
        pass
    listener.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
