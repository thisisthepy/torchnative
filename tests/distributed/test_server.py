"""The swappable server, the LAN opt-in and the pre-shared key (issue #233).

BACKEND.md §9 step 5, D4 and D10; SPEC S6.11. `backend/server.py` holds the
`Server` interface with `InProcessServer` and `StandaloneServer`, the
`tnserver://` rendezvous handler, the bind rule and the key rules;
`backend/transport.py` runs the key handshake on every mesh connection. All
interface and variable names are agent proposals, not confirmed.

Three kinds of test:

* **handshake, in one process** (a shim subprocess): the two halves of the
  mesh handshake over an in-process pair, with the right key, a wrong key, a
  key on one side, and a replayed proof; every byte either side sent is
  recorded and searched for the key. The bind rule against an address that
  need not exist, before and after the opt-in.
* **process worlds**: a world of three through each server implementation,
  ring `send`/`recv` and `all_reduce` against upstream gloo given the same
  inputs; then the right key, a wrong key, a key on one side only, and a rank
  that never arrives.
* each test says what sabotage it would catch.

The process harness is `test_collect2.py`'s `_c2_spawn`, imported. Written
before anything was run (AGENTS.md rule 10); the first run is CI's. A
two-machine run is a device test for the coordinating session and is not
claimed here.
"""

import functools
import importlib.util
import json
import os
import sys

import gloo_loopback

from test_collect2 import _c2_spawn, c2_exc, c2_ok
from test_p2p import _solo, _with_env


SRV_WORLD = 3
SRV_TIMEOUT_S = 10

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SERVER_PY = os.path.join(_REPO, "torchnative", "python", "torchnative",
                          "distributed", "backend", "server.py")

#: Every variable this suite sets in a child; unset unless a test gives it.
_SRV_ENV = ("TORCHNATIVE_DIST_KEY", "TORCHNATIVE_DIST_ALLOW_LAN",
            "TORCHNATIVE_DIST_BIND_ADDR", "TORCHNATIVE_DIST_TRANSPORT",
            "SRV_MODE", "SRV_URL", "SRV_ABSENT", "SRV_TIMEOUT", "SRV_PROBES",
            "SRV_KEY_0", "SRV_KEY_1", "SRV_KEY_2")


def _server_module():
    """`server.py` loaded by path: stdlib only at import, so no torch here."""
    spec = importlib.util.spec_from_file_location("tn_server_under_test",
                                                  _SERVER_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ---------------------------------------------------------------------------
# the handshake and the bind rule, in one shim process
# ---------------------------------------------------------------------------

_UNIT_SRC = r'''
import json, os, sys, threading
import torch
assert hasattr(torch._C, "_aten_implemented"), (
    "this subprocess loaded upstream torch, not the shim")
for name in ("TORCHNATIVE_DIST_KEY", "TORCHNATIVE_DIST_ALLOW_LAN",
             "TORCHNATIVE_DIST_BIND_ADDR", "TORCHNATIVE_DIST_TRANSPORT"):
    os.environ.pop(name, None)
from torch._C._distributed_c10d import HashStore
from torchnative.distributed.backend import server, transport, wire

out = {}

def probe(key, fn):
    try:
        out[key] = {"ok": fn()}
    except Exception as exc:
        out[key] = {"exc": "%s: %s" % (type(exc).__name__, exc)}

TOKEN = "group-token-for-the-unit-probe"
KEY = b"the-right-key-0123456789"
OTHER = b"a-different-key-9876543210"

def recording(conn):
    """Keep every frame `conn` sends, by wrapping its `send_frame` in place.

    In place, because `_hs_recv` dispatches on the connection's type, so the
    handshake must be handed the real `InprocConnection`.
    """
    sent = []
    original = conn.send_frame
    def send_frame(data):
        sent.append(bytes(data))
        original(data)
    conn.send_frame = send_frame
    return sent

def run_pair(key_c, key_a, c=0, a=1):
    """Connector rank c, acceptor rank a. Both outcomes and both sides' bytes."""
    import time
    end_c, end_a = transport.InprocConnection.pair()
    end_c.peer, end_a.peer = a, c
    sent_c, sent_a = recording(end_c), recording(end_a)
    accept = {}
    def acceptor():
        try:
            accept["peer"] = transport._accept_handshake(end_a, a, TOKEN, key_a, 10.0)
        except transport._HandshakeFailed as exc:
            accept["exc"] = "%s: %s" % (type(exc.error).__name__, exc.error)
        except transport._Stranger:
            accept["exc"] = "stranger"
    th = threading.Thread(target=acceptor)
    th.start()
    connect = {}
    try:
        transport._connect_handshake(end_c, c, a, TOKEN, key_c,
                                     time.monotonic() + 10.0)
        connect["ok"] = True
    except Exception as exc:
        connect["exc"] = "%s: %s" % (type(exc).__name__, exc)
    th.join(30)
    return {"connect": connect, "accept": accept, "sent_c": sent_c,
            "sent_a": sent_a}

def summary(res, key):
    wire_bytes = b"".join(res["sent_c"] + res["sent_a"])
    secret = key or b"\x00no-key\x00"
    return {"connect": res["connect"], "accept": res["accept"],
            "frames": len(res["sent_c"]) + len(res["sent_a"]),
            "key_on_wire": secret in wire_bytes,
            "key_hex_on_wire": secret.hex().encode() in wire_bytes}

probe("hs_right_key", lambda: summary(run_pair(KEY, KEY), KEY))
probe("hs_no_key", lambda: summary(run_pair(None, None), None))
probe("hs_wrong_key", lambda: summary(run_pair(KEY, OTHER), KEY))
probe("hs_key_on_connector_only", lambda: summary(run_pair(KEY, None), KEY))
probe("hs_key_on_acceptor_only", lambda: summary(run_pair(None, KEY), KEY))

def replay():
    # Session 1 succeeds and is recorded. Session 2 sends session 1's hello
    # and proof again, verbatim, to a fresh acceptor holding the right key.
    first = run_pair(KEY, KEY)
    hello, proof_frame = first["sent_c"][0], first["sent_c"][1]
    end_c, end_a = transport.InprocConnection.pair()
    end_c.peer, end_a.peer = 1, 0
    accept = {}
    def acceptor():
        try:
            accept["peer"] = transport._accept_handshake(end_a, 1, TOKEN, KEY, 10.0)
        except transport._HandshakeFailed as exc:
            accept["exc"] = "%s: %s" % (type(exc.error).__name__, exc.error)
    th = threading.Thread(target=acceptor)
    th.start()
    end_c.send_frame(hello)
    challenge = json.loads(end_c.recv_frame(timeout=10).payload.decode())
    end_c.send_frame(proof_frame)
    answer = json.loads(end_c.recv_frame(timeout=10).payload.decode())
    th.join(30)
    return {"first_ok": first["connect"].get("ok") is True
                        and first["accept"].get("peer") == 0,
            "challenge": challenge.get("t"), "answer": answer.get("t"),
            "accept": accept}
probe("hs_replay", replay)

# -- the bind rule ---------------------------------------------------------
# 203.0.113.7 is TEST-NET-3 (RFC 5737): no machine has it, so binding it fails
# with OSError. A refusal that names the opt-in therefore proves the check ran
# before the bind; with the opt-in set, the OSError proves the check let it by.
LAN = "203.0.113.7"

probe("bind_loopback", lambda: server.check_bind("127.0.0.1", "probe") is None)
probe("bind_localhost", lambda: server.check_bind("localhost", "probe") is None)
probe("bind_lan", lambda: server.check_bind(LAN, "probe"))
probe("bind_wildcard", lambda: server.check_bind("0.0.0.0", "probe"))

def mesh_lan(allow):
    if allow:
        os.environ["TORCHNATIVE_DIST_ALLOW_LAN"] = "1"
    store = HashStore()
    try:
        transport.Mesh(0, 2, store, 5.0, bind_addr=LAN)
        return {"built": True}
    except Exception as exc:
        return {"exc": "%s: %s" % (type(exc).__name__, exc),
                "type": type(exc).__name__,
                "is_bind_error": isinstance(exc, OSError),
                "store_keys_written": store.num_keys()}
    finally:
        os.environ.pop("TORCHNATIVE_DIST_ALLOW_LAN", None)
probe("mesh_lan_refused", lambda: mesh_lan(False))
probe("mesh_lan_opted_in", lambda: mesh_lan(True))

def server_lan(cls, allow):
    if allow:
        os.environ["TORCHNATIVE_DIST_ALLOW_LAN"] = "1"
    try:
        srv = cls(LAN, 0).start()
        srv.stop()
        return {"started": True}
    except Exception as exc:
        return {"exc": "%s: %s" % (type(exc).__name__, exc),
                "type": type(exc).__name__,
                "is_bind_error": isinstance(exc, OSError)}
    finally:
        os.environ.pop("TORCHNATIVE_DIST_ALLOW_LAN", None)
probe("inprocess_lan_refused", lambda: server_lan(server.InProcessServer, False))
probe("inprocess_lan_opted_in", lambda: server_lan(server.InProcessServer, True))
probe("standalone_lan_refused", lambda: server_lan(server.StandaloneServer, False))

probe("empty_key", lambda: server.resolve_key(""))

json.dump(out, open(sys.argv[1], "w"))
'''


@functools.cache
def srv_unit():
    return _solo(_UNIT_SRC, "server-unit")


def test_the_handshake_passes_with_the_same_key_and_with_no_key():
    """Both halves in one process: the same key on both, then no key on either.

    With a key it is four frames (hello, challenge, proof, ok); without, two
    (hello, accept). A count that differs means the key path was not taken.
    """
    report = srv_unit()
    for name, frames in (("hs_right_key", 4), ("hs_no_key", 2)):
        got = c2_ok(report, name)
        assert got["connect"] == {"ok": True}, (name, got)
        assert got["accept"] == {"peer": 0}, (name, got)
        assert got["frames"] == frames, (name, got)


def test_a_wrong_key_is_refused_by_name_by_both_halves_in_process():
    """Sabotage: skip `verify` in either half and that half accepts the wrong key."""
    got = c2_ok(srv_unit(), "hs_wrong_key")
    assert "exc" in got["connect"], got
    assert got["connect"]["exc"].startswith("AuthRefused:"), got
    assert "rank 1" in got["connect"]["exc"], got
    assert "different keys" in got["connect"]["exc"], got
    assert "exc" in got["accept"], got
    assert got["accept"]["exc"].startswith("AuthRefused:"), got
    assert "rank 0" in got["accept"]["exc"], got
    assert "TORCHNATIVE_DIST_KEY" in got["accept"]["exc"], got


def test_a_key_on_one_side_only_is_refused_by_name_by_both_halves_in_process():
    """Both directions: the connector holds the key, then the acceptor does.

    Sabotage: treat "the peer sent no key" as "accept without auth" and the
    side holding the key would trust an unauthenticated peer; here it must
    refuse instead.
    """
    for name in ("hs_key_on_connector_only", "hs_key_on_acceptor_only"):
        got = c2_ok(srv_unit(), name)
        assert "exc" in got["connect"], (name, got)
        assert got["connect"]["exc"].startswith("AuthRefused:"), (name, got)
        assert "rank 1" in got["connect"]["exc"], (name, got)
        assert "exc" in got["accept"], (name, got)
        assert got["accept"]["exc"].startswith("AuthRefused:"), (name, got)
        assert "rank 0" in got["accept"]["exc"], (name, got)


def test_the_key_never_crosses_the_wire_and_a_replayed_proof_does_not_verify():
    """Every byte both halves sent is searched for the key, raw and hex.

    The replay sends a recorded session's hello and proof to a fresh acceptor.
    Sabotage: drop the nonce from `server.proof` and the replayed proof
    verifies (the answer would be "ok").
    """
    report = srv_unit()
    for name in ("hs_right_key", "hs_wrong_key", "hs_key_on_connector_only",
                 "hs_key_on_acceptor_only"):
        got = c2_ok(report, name)
        assert got["frames"] >= 2, (name, got)
        assert got["key_on_wire"] is False, (name, "the key itself was sent")
        assert got["key_hex_on_wire"] is False, (name, "the key was sent as hex")
    rep = c2_ok(report, "hs_replay")
    assert rep["first_ok"] is True, rep
    assert rep["challenge"] == "challenge", rep
    assert rep["answer"] == "refuse", rep
    assert "does not verify" in rep["accept"]["exc"], rep


def test_a_lan_bind_without_the_opt_in_refuses_before_anything_binds():
    """203.0.113.7 exists nowhere: a named refusal without the opt-in, an OSError with it.

    The OSError on the opted-in run is what proves the refusal came first: if
    the check were moved after `bind`, the run without the opt-in would also
    end in OSError. Sabotage: delete `check_bind` and `mesh_lan_refused`
    becomes an OSError.
    """
    report = srv_unit()
    assert c2_ok(report, "bind_loopback") is True
    assert c2_ok(report, "bind_localhost") is True
    for name in ("bind_lan", "bind_wildcard"):
        msg = c2_exc(report, name)
        assert msg.startswith("BindRefused:"), (name, msg)
        assert "TORCHNATIVE_DIST_ALLOW_LAN=1" in msg, (name, msg)
        assert "not encrypted" in msg or "nothing is encrypted" in msg, (name, msg)
    got = c2_ok(report, "mesh_lan_refused")
    assert got["type"] == "BindRefused", got
    assert "203.0.113.7" in got["exc"] and "rank 0" in got["exc"], got
    assert got["store_keys_written"] == 0, (
        "the mesh wrote to the store before refusing the bind", got)
    assert got["is_bind_error"] is False, got
    for name in ("mesh_lan_opted_in", "inprocess_lan_opted_in"):
        got = c2_ok(report, name)
        assert "exc" in got, (name, "203.0.113.7 was bound", got)
        assert got["type"] != "BindRefused", (name, got)
        assert got["is_bind_error"] is True, (name, got)
    got = c2_ok(report, "inprocess_lan_refused")
    assert got["type"] == "BindRefused", got
    got = c2_ok(report, "standalone_lan_refused")
    assert got["type"] == "BindRefused", got


def test_an_empty_key_refuses_by_name():
    msg = c2_exc(srv_unit(), "empty_key")
    assert msg.startswith("ValueError:"), msg
    assert "TORCHNATIVE_DIST_KEY" in msg and "empty" in msg, msg


# ---------------------------------------------------------------------------
# process worlds
# ---------------------------------------------------------------------------
#
# One probe source, shared by the shim and gloo. The float inputs are
# multiples of 2**-6 in [-2, 2), so a sum of three is exact in float32 in any
# order: the comparison with gloo is equality, and a ring-versus-star order
# difference cannot hide in it (an integer sum is compared too).

_SRV_PROBE_SRC = r'''
SRV_SEED = 20261113

def srv_ints(rank, n, tag):
    state = (SRV_SEED + rank * 7919 + sum(ord(c) * 31 for c in tag)) % 2147483647
    out = []
    for _ in range(n):
        state = (state * 1103515245 + 12345) % 2147483648
        out.append((state >> 11) % 1000 - 500)
    return out

def srv_floats(rank, n, tag):
    return [(v % 256 - 128) / 64.0 for v in srv_ints(rank, n, tag)]

def srv_run(torch, dist, rank, world):
    out = {}

    def probe(key, fn):
        try:
            out[key] = {"ok": fn()}
        except Exception as exc:
            out[key] = {"exc": "%s: %s" % (type(exc).__name__, exc)}

    nxt, prv = (rank + 1) % world, (rank - 1) % world

    def ring(dtype, values, tag):
        buf = torch.zeros(len(values), dtype=dtype)
        src = torch.tensor(values, dtype=dtype)
        if rank % 2 == 0:
            dist.send(src, nxt, tag=tag)
            dist.recv(buf, prv, tag=tag)
        else:
            dist.recv(buf, prv, tag=tag)
            dist.send(src, nxt, tag=tag)
        return buf.tolist()
    probe("ring_float32", lambda: ring(torch.float32, srv_floats(rank, 6, "ringf"), 7))
    probe("ring_int64", lambda: ring(torch.int64, srv_ints(rank, 6, "ringi"), 8))

    def allreduce(dtype, values):
        t = torch.tensor(values, dtype=dtype)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return t.tolist()
    probe("allreduce_float32", lambda: allreduce(torch.float32, srv_floats(rank, 8, "arf")))
    probe("allreduce_int64", lambda: allreduce(torch.int64, srv_ints(rank, 8, "ari")))
    return out
'''

SRV_VALUE_PROBES = ("ring_float32", "ring_int64", "allreduce_float32",
                    "allreduce_int64")


_SRV_SHIM_SRC = (
    "import datetime, json, os, sys, time\n"
    "import torch\n"
    'assert hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded upstream torch, not the shim")\n'
    "import torch.distributed as dist\n"
    "import torchnative.distributed  # noqa: F401 -- backend and tnserver://\n"
    + _SRV_PROBE_SRC
    + r'''
rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
key = os.environ.get("SRV_KEY_%d" % rank)
if key is None:
    os.environ.pop("TORCHNATIVE_DIST_KEY", None)
else:
    os.environ["TORCHNATIVE_DIST_KEY"] = key
if os.environ.get("SRV_ABSENT") == str(rank):
    # The rank that never arrives: it reports and leaves before rendezvous.
    with open(dest, "w") as handle:
        json.dump({"absent": True}, handle)
    sys.exit(0)
mode = os.environ["SRV_MODE"]
if mode == "rank0":
    url = "tnserver://127.0.0.1:%d" % port
elif mode == "external":
    url = os.environ["SRV_URL"]
elif mode == "tcp":
    url = "tcp://127.0.0.1:%d" % port
else:
    raise SystemExit("unknown SRV_MODE %r" % mode)
timeout = float(os.environ.get("SRV_TIMEOUT", "120"))
t0 = time.time()
try:
    dist.init_process_group(backend="torchnative", init_method=url,
                            rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=timeout))
except Exception as exc:
    with open(dest, "w") as handle:
        json.dump({"init_exc": "%s: %s" % (type(exc).__name__, exc),
                   "seconds": time.time() - t0}, handle)
    # Whoever hosts the server keeps it up while the peers reach their own
    # verdicts, so a peer reports the refusal under test and not a lost store.
    time.sleep(3.0)
    sys.exit(0)
from torchnative.distributed.backend import server, transport_of
pg = dist.distributed_c10d._get_default_group()
backend = pg._get_backend(torch.device("cpu"))
out = srv_run(torch, dist, rank, world) if os.environ.get("SRV_PROBES") == "1" else {}
meta = {"shim": True, "rank": dist.get_rank(), "world": dist.get_world_size(),
        "backend": dist.get_backend(), "pid": os.getpid(),
        "kinds": {str(p): transport_of(p) for p in range(world) if p != rank},
        "authenticated": bool(backend._mesh.authenticated)}
if mode == "rank0" and rank == 0:
    snap = server._HOSTED[0].snapshot()
    meta["hosted_keys"] = sorted(snap)
    secret = (key or "").encode()
    meta["key_in_store"] = bool(secret) and any(
        secret in v or secret.hex().encode() in v for v in snap.values())
out["_meta"] = meta
with open(dest, "w") as handle:
    json.dump(out, handle)
# Rank 0 may host the store: leave only after every peer is done with it.
dist.barrier()
time.sleep(1.0)
'''
)


_SRV_GLOO_SRC = (
    gloo_loopback.child_pin_source()
    + "import json, os, sys, time\n"
    "import torch\n"
    'assert not hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded the shim, not upstream torch")\n'
    "import torch.distributed as dist\n"
    + _SRV_PROBE_SRC
    + r'''
rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(port)
dist.init_process_group(backend="gloo", rank=rank, world_size=world)
out = srv_run(torch, dist, rank, world)
out["_meta"] = {"shim": False, "rank": dist.get_rank(),
                "world": dist.get_world_size(), "backend": "gloo"}
with open(dest, "w") as handle:
    json.dump(out, handle)
dist.destroy_process_group()
'''
)


def _srv_world(world, what, timeout=600, **env):
    full = {name: None for name in _SRV_ENV}
    full.update(env)
    return _with_env(full, lambda: _c2_spawn(_SRV_SHIM_SRC, world, "float32",
                                             True, what, timeout=timeout))


@functools.cache
def srv_gloo():
    return _c2_spawn(_SRV_GLOO_SRC, SRV_WORLD, "float32", False, "srv-gloo3")


@functools.cache
def srv_inprocess():
    return _srv_world(SRV_WORLD, "srv-inprocess", SRV_MODE="rank0", SRV_PROBES="1")


@functools.cache
def srv_standalone():
    """A world of three through a separate server process the test itself started."""
    mod = _server_module()
    srv = mod.StandaloneServer("127.0.0.1", 0).start()
    try:
        host, port = srv.address()
        reports = _srv_world(SRV_WORLD, "srv-standalone", SRV_MODE="external",
                             SRV_URL=srv.url(), SRV_PROBES="1")
        keys = mod.raw_request(host, port, {"op": "list_keys"})["value"]
        return {"reports": reports, "server_pid": srv.pid(), "keys": keys,
                "url": srv.url()}
    finally:
        srv.stop()


@functools.cache
def srv_right_key():
    k = "shared-key-for-the-right-key-world"
    return _srv_world(SRV_WORLD, "srv-rightkey", SRV_MODE="rank0", SRV_PROBES="1",
                      SRV_KEY_0=k, SRV_KEY_1=k, SRV_KEY_2=k)


@functools.cache
def srv_wrong_key():
    return _srv_world(2, "srv-wrongkey", timeout=180, SRV_MODE="rank0",
                      SRV_TIMEOUT="60", SRV_KEY_0="alpha-key", SRV_KEY_1="beta-key")


@functools.cache
def srv_one_sided_key():
    return _srv_world(SRV_WORLD, "srv-onesided", timeout=180, SRV_MODE="rank0",
                      SRV_TIMEOUT="60", SRV_KEY_1="only-rank-one-holds-this")


@functools.cache
def srv_absent(mode):
    return _srv_world(SRV_WORLD, "srv-absent-%s" % mode, timeout=180,
                      SRV_MODE=mode, SRV_ABSENT="2", SRV_TIMEOUT=str(SRV_TIMEOUT_S))


def _agrees_with_gloo(reports, label):
    gloo = srv_gloo()
    seen = 0
    for rank in range(SRV_WORLD):
        meta = reports[rank]["_meta"]
        assert (meta["shim"], meta["rank"], meta["world"], meta["backend"]) == (
            True, rank, SRV_WORLD, "torchnative"), (label, meta)
        for key in SRV_VALUE_PROBES:
            ours, theirs = c2_ok(reports[rank], key), c2_ok(gloo[rank], key)
            assert ours == theirs, (
                "%s: %s at rank %d differs from upstream gloo.\n  shim: %r\n"
                "  gloo: %r" % (label, key, rank, ours, theirs))
            seen += 1
    assert seen == SRV_WORLD * len(SRV_VALUE_PROBES), seen


def test_a_world_of_three_agrees_with_gloo_through_the_in_process_server():
    """`tnserver://` with rank 0 hosting an `InProcessServer`: ring and all_reduce equal gloo's.

    Sabotage: a store client whose `get` returned a wrong value would break
    the mesh's address exchange and the star's port, and the world would not
    form; one that dropped `set` would time out naming a rank.
    """
    reports = srv_inprocess()
    _agrees_with_gloo(reports, "in-process server")
    hosted = reports[0]["_meta"]["hosted_keys"]
    assert any(k.endswith("tn_server/arrived/2") for k in hosted), hosted
    assert any(k.endswith("tn_p2p/addr/2") for k in hosted), hosted
    for rank in range(SRV_WORLD):
        assert reports[rank]["_meta"]["authenticated"] is False, reports[rank]["_meta"]


def test_the_same_world_agrees_with_gloo_through_a_standalone_server_process():
    """The server is a separate process; the ranks reach it with `?server=external`.

    The server's own key list proves the ranks rendezvoused through it, and its
    pid is none of theirs. Sabotage: if `external` fell back to rank 0 hosting,
    rank 0 would fail to bind the port the standalone server holds.
    """
    got = srv_standalone()
    assert got["url"].startswith("tnserver://127.0.0.1:"), got["url"]
    assert got["url"].endswith("?server=external"), got["url"]
    reports = got["reports"]
    _agrees_with_gloo(reports, "standalone server")
    pids = {reports[r]["_meta"]["pid"] for r in range(SRV_WORLD)}
    assert got["server_pid"] not in pids, (got["server_pid"], pids)
    for rank in range(SRV_WORLD):
        assert any(k.endswith("tn_server/arrived/%d" % rank) for k in got["keys"]), (
            rank, got["keys"])
        assert any(k.endswith("tn_p2p/addr/%d" % rank) for k in got["keys"]), (
            rank, got["keys"])


def test_with_the_right_key_the_world_works_and_every_rank_authenticated():
    """Same key everywhere: values equal gloo's, each mesh reports `authenticated`, the store never saw the key.

    Sabotage: if the backend ignored `TORCHNATIVE_DIST_KEY`, the values would
    still agree but `authenticated` would be False.
    """
    reports = srv_right_key()
    _agrees_with_gloo(reports, "right key")
    for rank in range(SRV_WORLD):
        assert reports[rank]["_meta"]["authenticated"] is True, reports[rank]["_meta"]
    assert reports[0]["_meta"]["key_in_store"] is False, reports[0]["_meta"]


def test_a_wrong_key_refuses_by_name_on_both_sides():
    """Rank 0 holds one key, rank 1 another: each refuses, naming the other.

    It must observe a refusal, not merely the absence of success: skipping the
    HMAC check would let this world form, and both ranks would then report
    `_meta` instead of `init_exc`.
    """
    reports = srv_wrong_key()
    for rank, peer in ((0, 1), (1, 0)):
        msg = reports[rank].get("init_exc")
        assert msg is not None, ("rank %d formed a world with a wrong key" % rank,
                                 reports[rank])
        assert msg.startswith("AuthRefused:"), msg
        assert "rank %d" % peer in msg, msg
        assert "TORCHNATIVE_DIST_KEY" in msg, msg
        assert "does not verify" in msg, msg
        assert reports[rank]["seconds"] < 60, reports[rank]


def test_a_key_on_one_side_only_refuses_by_name_on_both_sides():
    """Only rank 1 holds a key. Ranks 0 and 2 refuse naming rank 1; rank 1 names one of them."""
    reports = srv_one_sided_key()
    for rank in (0, 2):
        msg = reports[rank].get("init_exc")
        assert msg is not None, reports[rank]
        assert msg.startswith("AuthRefused:"), msg
        assert "rank 1" in msg and "TORCHNATIVE_DIST_KEY" in msg, msg
    msg = reports[1].get("init_exc")
    assert msg is not None, reports[1]
    assert msg.startswith("AuthRefused:"), msg
    assert "rank 0" in msg or "rank 2" in msg, msg


def test_a_rank_that_never_arrives_is_named_within_the_timeout():
    """Rank 2 never calls init_process_group; ranks 0 and 1 name it at the group's timeout.

    Through `tnserver://` (the server's store) and through `tcp://` (the shim's
    `TCPStore`, whose `get` blocked forever before this step: without that fix
    the `tcp` world hangs until the harness's 180 s deadline and this fails).
    The lower bound shows the wait was the timeout, not an early guess.
    """
    for mode in ("rank0", "tcp"):
        reports = srv_absent(mode)
        assert reports[2] == {"absent": True}, (mode, reports[2])
        for rank in (0, 1):
            msg = reports[rank].get("init_exc")
            assert msg is not None, (mode, rank, reports[rank])
            assert msg.startswith("RendezvousTimeout:"), (mode, msg)
            assert "rank 2" in msg and "never arrived" in msg, (mode, msg)
            seconds = reports[rank]["seconds"]
            assert SRV_TIMEOUT_S * 0.5 <= seconds <= SRV_TIMEOUT_S * 4, (
                mode, rank, seconds)


def test_every_rank_really_ran_the_shim_and_the_oracle_really_ran_upstream():
    for rank in range(SRV_WORLD):
        meta = srv_gloo()[rank]["_meta"]
        assert (meta["shim"], meta["backend"]) == (False, "gloo"), meta
        meta = srv_inprocess()[rank]["_meta"]
        assert (meta["shim"], meta["backend"]) == (True, "torchnative"), meta
        assert meta["kinds"] == {str(p): "shm" for p in range(SRV_WORLD) if p != rank}, meta


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
