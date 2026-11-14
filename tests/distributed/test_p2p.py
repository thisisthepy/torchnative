"""Point-to-point on the `torchnative` backend, against upstream gloo (issue #219).

BACKEND.md §9 steps 2 and 4, SPEC S6.10 and S6.12. Three layers, three kinds
of test:

* **wire** (`backend/wire.py`): every dtype the frame carries round-trips bit
  for bit, checked against bytes this file builds with `struct` itself (an
  oracle independent of the export path, so an export that was wrong in a
  self-consistent way still fails); a frame fed one byte per `recv` decodes;
  truncation, a flipped magic byte and a wrong version raise naming the peer.
* **transports** (`backend/transport.py`): a loopback pair per transport, built
  as two threads of one shim process over a `HashStore`, and a forced
  override that changes the transport actually used; two groups in one process
  do not meet.
* **p2p differential**: real process worlds of 2 and 4, the shim against
  upstream gloo given the same inputs. Point-to-point only moves bytes, so the
  comparison is equality, with no tolerance.

The process harness is `test_collect2.py`'s (`_c2_spawn`, imported, not
copied): children on an ephemeral port, output to files, every rank's state on
a timeout, every child killed on every path.

Written before the implementation was run anywhere (AGENTS.md rule 10: agents
do not run suites); the first run is CI's.
"""

import functools
import json
import os
import subprocess
import sys
import tempfile

import gloo_loopback

from test_collect2 import _c2_spawn, c2_exc, c2_ok
from test_shim import _CKPT_VENDOR_DIR


P2P_WORLDS = (2, 4)


def _shim_env():
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    env.pop("TORCHNATIVE_DIST_TRANSPORT", None)
    env.pop("TORCHNATIVE_DIST_BIND_ADDR", None)
    return env


def _solo(source, what, timeout=600):
    """Run `source` once in a shim subprocess; it writes JSON to argv[1]."""
    import atexit
    import shutil
    tmp = tempfile.mkdtemp(prefix="p2p-%s-" % what)
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    dest = os.path.join(tmp, "out.json")
    proc = subprocess.run([sys.executable, "-c", source, dest],
                          capture_output=True, text=True, env=_shim_env(),
                          timeout=timeout)
    if proc.returncode != 0 or not os.path.exists(dest):
        raise RuntimeError(
            "p2p/%s: the shim subprocess exited %d\n--- stdout ---\n%s\n"
            "--- stderr ---\n%s" % (what, proc.returncode, proc.stdout[-4000:],
                                    proc.stderr[-6000:]))
    with open(dest) as handle:
        return json.load(handle)


_PROBE_HEAD = r'''
import json, os, struct, sys, threading, time
import torch
assert hasattr(torch._C, "_aten_implemented"), (
    "this subprocess loaded upstream torch, not the shim")
os.environ.pop("TORCHNATIVE_DIST_TRANSPORT", None)
from torchnative.distributed.backend import transport, wire

out = {}

def probe(key, fn):
    try:
        out[key] = {"ok": fn()}
    except Exception as exc:
        out[key] = {"exc": "%s: %s" % (type(exc).__name__, exc)}
'''


# ---------------------------------------------------------------------------
# wire
# ---------------------------------------------------------------------------

_WIRE_SRC = _PROBE_HEAD + r'''
FLOATS = [0.0, -0.0, 1.0, -1.5, 0.15625, 1024.0, -3.75,
          float("inf"), float("-inf")]
CASES = {
    "float32": ("f", FLOATS),
    "float64": ("d", FLOATS),
    "float16": ("e", FLOATS),
    "bfloat16": (None, FLOATS),
    "int8": ("b", [-128, 127, 0, -1, 5, -6, 7, 100, -100]),
    "int16": ("h", [-32768, 32767, 0, -1, 300, -301, 7, 12345, -12345]),
    "int32": ("i", [-2 ** 31, 2 ** 31 - 1, 0, -1, 70000, -70001, 7,
                    123456789, -123456789]),
    "int64": ("q", [-2 ** 63, 2 ** 63 - 1, 0, -1, 2 ** 40 + 3,
                    -(2 ** 40) - 5, 7, 9, -9]),
    "uint8": ("B", [0, 255, 1, 128, 127, 200, 3, 4, 5]),
    "bool": ("?", [True, False, True, True, False, False, True, False, True]),
}

def oracle(name, fmt, values):
    """The little-endian bytes, built by `struct` and nothing else."""
    if name == "bfloat16":
        raw = struct.pack("<%df" % len(values), *values)
        return b"".join(raw[i + 2:i + 4] for i in range(0, len(raw), 4))
    return struct.pack("<%d%s" % (len(values), fmt), *values)

for name, (fmt, values) in CASES.items():
    def rt(name=name, fmt=fmt, values=values):
        t = torch.tensor(values, dtype=getattr(torch, name)).reshape(3, 3)
        data = wire.encode_tensor(t, -7)
        frame = wire.decode(data, 1)
        back = wire.frame_to_tensor(frame)
        want = oracle(name, fmt, values)
        return {"dtype": str(back.dtype), "shape": list(back.shape),
                "tag": frame.tag, "kind": frame.kind,
                "payload_is_oracle": frame.payload == want,
                "export_is_oracle": wire.tensor_bytes(t)[2] == want,
                "back_export_is_oracle": wire.tensor_bytes(back)[2] == want,
                "values_equal": back.tolist() == t.tolist(),
                "path": wire.export_path(t)}
    probe("rt_" + name, rt)

def transposed():
    vals = [float(v) for v in range(12)]
    t = torch.tensor(vals, dtype=torch.float32).reshape(3, 4).t()
    rows = [[vals[r * 4 + c] for r in range(3)] for c in range(4)]
    flat = [v for row in rows for v in row]
    frame = wire.decode(wire.encode_tensor(t, 0), 1)
    return {"shape": list(frame.shape),
            "payload_is_oracle": frame.payload == struct.pack("<12f", *flat),
            "values": wire.frame_to_tensor(frame).tolist(), "rows": rows}
probe("rt_transposed", transposed)

def empty():
    frame = wire.decode(wire.encode_tensor(torch.zeros(0, 3), 4), 1)
    back = wire.frame_to_tensor(frame)
    return {"shape": list(back.shape), "payload": len(frame.payload)}
probe("rt_empty", empty)

def scalar():
    frame = wire.decode(wire.encode_tensor(torch.tensor(2.5), 4), 1)
    back = wire.frame_to_tensor(frame)
    return {"shape": list(back.shape), "value": back.item()}
probe("rt_scalar", scalar)

SAMPLE = wire.encode_tensor(
    torch.tensor([1.25, -2.5, 3.0, 4.75], dtype=torch.float32).reshape(2, 2), 9)

class OneByteSock:
    """A socket whose every `recv` returns at most one byte."""
    def __init__(self, data):
        self.data = bytes(data)
        self.pos = 0
        self.calls = 0
        self.sent = b""
    def recv(self, n):
        self.calls += 1
        chunk = self.data[self.pos:self.pos + 1]
        self.pos += len(chunk)
        return chunk
    def sendall(self, data):
        self.sent += bytes(data)
    def close(self):
        pass

def one_byte():
    sock = OneByteSock(SAMPLE)
    conn = transport.SocketConnection(sock, "tcp", peer=2)
    frame = conn.recv_frame()
    conn.send_frame(SAMPLE)
    return {"calls": sock.calls, "frame_bytes": len(SAMPLE),
            "tag": frame.tag, "values": wire.frame_to_tensor(frame).tolist(),
            "sent_back_intact": sock.sent == SAMPLE}
probe("one_byte_per_recv", one_byte)

def via_conn(data):
    return transport.SocketConnection(OneByteSock(data), "shm", peer=2).recv_frame()

probe("truncated_payload", lambda: repr(via_conn(SAMPLE[:-3])))
probe("truncated_header", lambda: repr(via_conn(SAMPLE[:10])))
probe("truncated_shape", lambda: repr(via_conn(SAMPLE[:28])))
probe("clean_eof", lambda: repr(via_conn(b"")))

def flipped(at):
    data = bytearray(SAMPLE)
    data[at] ^= 0xFF
    return bytes(data)
probe("bad_magic", lambda: repr(via_conn(flipped(0))))
probe("bad_version", lambda: repr(via_conn(flipped(4))))
probe("bad_magic_decode", lambda: repr(wire.decode(flipped(1), 3)))
probe("intact_control", lambda: list(via_conn(SAMPLE).shape))

probe("complex_refused",
      lambda: len(wire.encode_tensor(torch.zeros(2, dtype=torch.complex64), 0)))

json.dump(out, open(sys.argv[1], "w"))
'''


@functools.cache
def p2p_wire():
    return _solo(_WIRE_SRC, "wire")


P2P_WIRE_DTYPES = ("float32", "float64", "float16", "bfloat16", "int8",
                   "int16", "int32", "int64", "uint8", "bool")


def test_every_dtype_round_trips_bit_for_bit_against_struct_built_bytes():
    """Ten dtypes, each exact in both directions, measured against `struct`.

    Three byte comparisons per dtype and each catches something different:
    the frame's payload (what crossed the wire), the export of the original
    tensor, and the export of the decoded tensor (so a decoder that reordered
    or rounded is caught even if the encoder were right). `-0.0` and the
    infinities are in the floats because a path through a numeric conversion
    is where a sign bit or an infinity goes missing.
    """
    report = p2p_wire()
    seen = 0
    for name in P2P_WIRE_DTYPES:
        got = c2_ok(report, "rt_" + name)
        assert got["dtype"] == "torch." + name, (name, got)
        assert got["shape"] == [3, 3], (name, got)
        assert got["tag"] == -7, (name, got)
        assert got["kind"] == 1, (name, got)
        for key in ("payload_is_oracle", "export_is_oracle",
                    "back_export_is_oracle", "values_equal"):
            assert got[key] is True, (
                "torch.%s: %s is False; the wire is not bit-exact for this "
                "dtype (path %s)" % (name, key, got["path"]))
        seen += 1
    assert seen == len(P2P_WIRE_DTYPES) == 10, seen


def test_a_non_contiguous_an_empty_and_a_zero_dim_tensor_round_trip():
    report = p2p_wire()
    t = c2_ok(report, "rt_transposed")
    assert t["shape"] == [4, 3], t
    assert t["payload_is_oracle"] is True, t
    assert t["values"] == t["rows"], t
    e = c2_ok(report, "rt_empty")
    assert e == {"shape": [0, 3], "payload": 0}, e
    s = c2_ok(report, "rt_scalar")
    assert s == {"shape": [], "value": 2.5}, s


def test_a_frame_split_into_one_byte_reads_decodes_whole():
    """`recv` is allowed to return one byte; the decoder must loop until a frame is whole."""
    got = c2_ok(p2p_wire(), "one_byte_per_recv")
    # At least one call per byte: the fake never returned more than one.
    assert got["calls"] >= got["frame_bytes"], got
    assert got["tag"] == 9, got
    assert got["values"] == [[1.25, -2.5], [3.0, 4.75]], got
    assert got["sent_back_intact"] is True, got


def test_a_truncated_frame_raises_naming_the_peer():
    report = p2p_wire()
    for key in ("truncated_payload", "truncated_header", "truncated_shape"):
        msg = c2_exc(report, key)
        assert msg.startswith("WireError:"), (key, msg)
        assert "rank 2" in msg, (key, msg)
        assert "truncated" in msg, (key, msg)
    # The stream ending *between* frames is a vanished peer, not corruption.
    msg = c2_exc(report, "clean_eof")
    assert msg.startswith("PeerGone:"), msg
    assert "rank 2" in msg and "connection closed" in msg, msg


def test_a_flipped_header_byte_is_a_named_error_not_a_garbled_tensor():
    """The sabotage of step 2: corrupt the header and the decoder must say so."""
    report = p2p_wire()
    msg = c2_exc(report, "bad_magic")
    assert msg.startswith("WireError:"), msg
    assert "rank 2" in msg and "magic" in msg, msg
    msg = c2_exc(report, "bad_magic_decode")
    assert "rank 3" in msg and "magic" in msg, msg
    msg = c2_exc(report, "bad_version")
    assert "rank 2" in msg and "version" in msg, msg
    # The same frame, unflipped, decodes: the failures above are the flip.
    assert c2_ok(report, "intact_control") == [2, 2]


def test_a_dtype_without_a_wire_code_refuses_by_name():
    msg = c2_exc(p2p_wire(), "complex_refused")
    assert msg.startswith("NotImplementedError:"), msg
    assert "complex64" in msg, msg


# ---------------------------------------------------------------------------
# transports, as threads of one process
# ---------------------------------------------------------------------------

_TRANSPORT_SRC = _PROBE_HEAD + r'''
from torch._C._distributed_c10d import HashStore

def mesh_pair(force, seed):
    store = HashStore()
    res = {}
    def run(rank):
        try:
            m = transport.Mesh(rank, 2, store, 60.0, force=force)
            try:
                res[rank] = {"kind": m.kind_of(1 - rank),
                             "unix_path_left": bool(
                                 m.unix_path and os.path.exists(m.unix_path))}
                dest = torch.zeros(6, dtype=torch.int64)
                if rank == 0:
                    t = torch.arange(6, dtype=torch.int64) * seed
                    m.send(1, 5, wire.encode_tensor(t, 5)).wait()
                    m.recv(1, 6, dest).wait()
                else:
                    m.recv(0, 5, dest).wait()
                    m.send(0, 6, wire.encode_tensor(dest + 1, 6)).wait()
                res[rank]["got"] = dest.tolist()
            finally:
                m.close()
        except Exception as exc:
            res[rank] = {"exc": "%s: %s" % (type(exc).__name__, exc)}
    threads = [threading.Thread(target=run, args=(r,)) for r in (0, 1)]
    for th in threads:
        th.start()
    for th in threads:
        th.join(120)
    return {str(r): res.get(r, {"exc": "thread did not finish"}) for r in (0, 1)}

probe("pair_default", lambda: mesh_pair(None, 3))
probe("pair_shm", lambda: mesh_pair("shm", 5))
probe("pair_tcp", lambda: mesh_pair("tcp", 7))

def two_groups():
    got = {}
    def one(name, seed):
        got[name] = mesh_pair(None, seed)
    ths = [threading.Thread(target=one, args=("a", 11)),
           threading.Thread(target=one, args=("b", 13))]
    for th in ths:
        th.start()
    for th in ths:
        th.join(180)
    return got
probe("two_groups", two_groups)

probe("force_ws", lambda: transport.forced_kind("ws"))
probe("force_bogus", lambda: transport.forced_kind("carrier-pigeon"))

json.dump(out, open(sys.argv[1], "w"))
'''


@functools.cache
def p2p_transports():
    return _solo(_TRANSPORT_SRC, "transports")


def _pair_ok(pair, kind, seed):
    for rank in ("0", "1"):
        assert "exc" not in pair[rank], (kind, rank, pair[rank])
        assert pair[rank]["kind"] == kind, (kind, rank, pair[rank])
        assert pair[rank]["unix_path_left"] is False, (
            "the shm socket path is still on disk after the mesh was built",
            pair[rank])
    # rank 1 received rank 0's tensor; rank 0 received it back plus one.
    assert pair["1"]["got"] == [seed * i for i in range(6)], pair
    assert pair["0"]["got"] == [seed * i + 1 for i in range(6)], pair


def test_a_loopback_pair_works_over_each_transport_and_reports_which():
    """inproc (the default for two threads), then shm and tcp forced."""
    report = p2p_transports()
    _pair_ok(c2_ok(report, "pair_default"), "inproc", 3)
    _pair_ok(c2_ok(report, "pair_shm"), "shm", 5)
    _pair_ok(c2_ok(report, "pair_tcp"), "tcp", 7)


def test_two_groups_in_one_process_do_not_meet():
    """The in-process registry is keyed by group, so two concurrent groups stay apart."""
    got = c2_ok(p2p_transports(), "two_groups")
    _pair_ok(got["a"], "inproc", 11)
    _pair_ok(got["b"], "inproc", 13)


def test_an_override_that_names_no_built_transport_refuses():
    report = p2p_transports()
    msg = c2_exc(report, "force_ws")
    assert msg.startswith("NotImplementedError:"), msg
    assert "ws" in msg and "TORCHNATIVE_DIST_TRANSPORT" in msg, msg
    msg = c2_exc(report, "force_bogus")
    assert msg.startswith("ValueError:"), msg
    assert "carrier-pigeon" in msg, msg


# ---------------------------------------------------------------------------
# point to point in process worlds, against upstream gloo
# ---------------------------------------------------------------------------
#
# One probe source, shared verbatim by both sides, as in test_collect2.py: the
# claim is that the two were given the same inputs. Pairings are written so
# that a *blocking* gloo `send` cannot deadlock: the sync ring alternates even
# and odd ranks, and everything else uses `isend`.

_P2P_PROBE_SRC = r'''
P2P_SEED = 20261113

def p2p_ints(rank, n, tag):
    state = (P2P_SEED + rank * 104729 + sum(ord(c) * 17 for c in tag)) % 2147483647
    out = []
    for _ in range(n):
        state = (state * 1103515245 + 12345) % 2147483648
        out.append((state >> 11) % 1000 - 500)
    return out

def p2p_floats(rank, n, tag):
    # Multiples of 2**-6 in [-2, 2): exact in float16 and bfloat16 too.
    return [(v % 256 - 128) / 64.0 for v in p2p_ints(rank, n, tag)]

def p2p_values(name, rank, n, tag):
    if name == "bool":
        return [v % 2 == 1 for v in p2p_ints(rank, n, tag)]
    if name == "uint8":
        return [v % 256 for v in p2p_ints(rank, n, tag)]
    if name == "int8":
        return [v % 256 - 128 for v in p2p_ints(rank, n, tag)]
    if name.startswith("int"):
        return p2p_ints(rank, n, tag)
    return p2p_floats(rank, n, tag)

P2P_DTYPES = ("float32", "float64", "float16", "bfloat16", "int8", "int16",
              "int32", "int64", "uint8", "bool")

def p2p_run(torch, dist, rank, world, shim):
    out = {}

    def probe(key, fn):
        try:
            out[key] = {"ok": fn()}
        except Exception as exc:
            out[key] = {"exc": "%s: %s" % (type(exc).__name__, exc)}

    def mk(name, r, n, tag):
        return torch.tensor(p2p_values(name, r, n, tag), dtype=getattr(torch, name))

    nxt, prv = (rank + 1) % world, (rank - 1) % world

    # -- ring, blocking send and recv, alternating so gloo cannot deadlock --
    for name in ("float32", "int64"):
        def ring_sync(name=name):
            buf = torch.zeros(5, dtype=getattr(torch, name))
            src = mk(name, rank, 5, "sync" + name)
            if rank % 2 == 0:
                dist.send(src, nxt, tag=11)
                dist.recv(buf, prv, tag=11)
            else:
                dist.recv(buf, prv, tag=11)
                dist.send(src, nxt, tag=11)
            return buf.tolist()
        probe("ring_sync_" + name, ring_sync)

    # -- ring, isend then a blocking recv, every dtype -------------------------
    for i, name in enumerate(P2P_DTYPES):
        def ring(name=name, tag=100 + i):
            buf = torch.zeros(5, dtype=getattr(torch, name))
            work = dist.isend(mk(name, rank, 5, "ring" + name), nxt, tag=tag)
            dist.recv(buf, prv, tag=tag)
            work.wait()
            return buf.tolist()
        probe("ring_" + name, ring)

    # -- a pair that is not neighbours: 0<->2 and 1<->3 at world 4 ----------
    def xor2():
        if world < 4:
            return None
        partner = rank ^ 2
        buf = torch.zeros(4, dtype=torch.float32)
        src = mk("float32", rank, 4, "xor2")
        if rank < partner:
            dist.send(src, partner, tag=21)
            dist.recv(buf, partner, tag=21)
        else:
            dist.recv(buf, partner, tag=21)
            dist.send(src, partner, tag=21)
        return buf.tolist()
    probe("pair_xor2", xor2)

    # -- tags: rank 0 sends tag 2 then tag 1; rank 1 receives tag 1 first ---
    def tags():
        if rank == 0:
            a = dist.isend(mk("int64", 0, 3, "tagA"), 1, tag=2)
            time.sleep(0.2)
            b = dist.isend(mk("int64", 0, 3, "tagB"), 1, tag=1)
            a.wait()
            b.wait()
            return "sent"
        if rank == 1:
            time.sleep(0.5)  # both frames are queued before either is taken
            first = torch.zeros(3, dtype=torch.int64)
            second = torch.zeros(3, dtype=torch.int64)
            dist.recv(first, 0, tag=1)
            dist.recv(second, 0, tag=2)
            return [first.tolist(), second.tolist()]
        return None
    probe("tags_out_of_order", tags)

    # -- isend / irecv, both asynchronous, then wait --------------------------
    for name in ("float32", "int32"):
        def async_ring(name=name):
            buf = torch.zeros(6, dtype=getattr(torch, name))
            r = dist.irecv(buf, prv, tag=31)
            s = dist.isend(mk(name, rank, 6, "async" + name), nxt, tag=31)
            s.wait()
            r.wait()
            return buf.tolist()
        probe("isend_irecv_" + name, async_ring)

    # -- an irecv is not complete before its sender has sent ----------------
    def pending():
        if rank == 0:
            buf = torch.zeros(3, dtype=torch.float32)
            work = dist.irecv(buf, 1, tag=41)
            before = buf.tolist()
            completed = bool(work.is_completed())
            waited = bool(work.wait())
            return {"completed_before_wait": completed, "before": before,
                    "wait_returned": waited, "after": buf.tolist()}
        if rank == 1:
            time.sleep(0.5)
            dist.send(mk("float32", 1, 3, "pending"), 0, tag=41)
            return "sent"
        return None
    probe("irecv_pending", pending)

    # -- batch_isend_irecv, a ring --------------------------------------------
    def batch():
        buf = torch.zeros(4, dtype=torch.float32)
        ops = [dist.P2POp(dist.isend, mk("float32", rank, 4, "batch"), nxt, tag=51),
               dist.P2POp(dist.irecv, buf, prv, tag=51)]
        reqs = dist.batch_isend_irecv(ops)
        for req in reqs:
            req.wait()
        return {"values": buf.tolist(), "requests": len(reqs)}
    probe("batch_ring", batch)

    # Object lists are not probed: the shim lacks `ByteStorage._from_buffer`,
    # so the sender fails before sending and the receiver would wait out the
    # group timeout, stalling every later probe (issue #229).

    if shim:
        # Upstream gloo would block on these; they are asserted on the shim only.
        def anysource():
            dist.recv(torch.zeros(2))
            return "ACCEPTED"
        probe("recv_anysource", anysource)

        def bad_dtype():
            # A pair that disagrees about the dtype refuses with both named
            # and writes nothing.
            if rank == 0:
                dist.send(torch.ones(3, dtype=torch.int64), 1, tag=61)
                return "sent"
            if rank == 1:
                buf = torch.full((3,), 7.0)
                try:
                    dist.recv(buf, 0, tag=61)
                except Exception as exc:
                    return {"exc": "%s: %s" % (type(exc).__name__, exc),
                            "buf": buf.tolist()}
                return {"exc": None, "buf": buf.tolist()}
            return None
        probe("dtype_mismatch", bad_dtype)

        # The last probe. The highest rank leaves (see the source below,
        # which exits it after writing its report); every other rank is left
        # waiting on a receive from it that no send will ever satisfy.
        def vanished():
            if rank == world - 1:
                return "left"
            t0 = time.time()
            try:
                dist.recv(torch.zeros(2), world - 1, tag=999)
            except Exception as exc:
                return {"exc": "%s: %s" % (type(exc).__name__, exc),
                        "seconds": time.time() - t0}
            return {"exc": None}
        probe("peer_vanished", vanished)
    return out
'''


_P2P_SHIM_SRC = (
    "import datetime, json, os, sys, time\n"
    "import torch\n"
    'assert hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded upstream torch, not the shim")\n'
    "import torch.distributed as dist\n"
    "import torchnative.distributed  # noqa: F401 -- registers backend='torchnative'\n"
    + _P2P_PROBE_SRC
    + r'''
rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
try:
    dist.init_process_group(backend="torchnative",
                            init_method="tcp://127.0.0.1:%d" % port,
                            rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=120))
except Exception as exc:
    with open(dest, "w") as handle:
        json.dump({"init_exc": "%s: %s" % (type(exc).__name__, exc)}, handle)
    # Rank 0 hosts the TCPStore. Leaving at once could take the store away
    # while a peer is still reading the records it needs to reach its *own*
    # verdict, and that peer would then report a refused connection instead
    # of the refusal under test.
    time.sleep(3.0)
    sys.exit(0)
from torchnative.distributed.backend import transport_of
kinds = {str(p): transport_of(p) for p in range(world) if p != rank}
if os.environ.get("P2P_INIT_ONLY") == "1":
    out = {}
else:
    out = p2p_run(torch, dist, rank, world, True)
out["_meta"] = {"shim": True, "rank": dist.get_rank(),
                "world": dist.get_world_size(),
                "backend": dist.get_backend(), "kinds": kinds}
with open(dest, "w") as handle:
    json.dump(out, handle)
if out.get("peer_vanished", {}).get("ok") == "left":
    # The departure the survivors' last probe waits on: no goodbye, no
    # atexit, the sockets closed by the kernel.
    os._exit(0)
'''
)


_P2P_GLOO_SRC = (
    gloo_loopback.child_pin_source()
    + "import json, os, sys, time\n"
    "import torch\n"
    'assert not hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded the shim, not upstream torch")\n'
    "import torch.distributed as dist\n"
    + _P2P_PROBE_SRC
    + r'''
rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(port)
dist.init_process_group(backend="gloo", rank=rank, world_size=world)
out = p2p_run(torch, dist, rank, world, False)
out["_meta"] = {"shim": False, "rank": dist.get_rank(),
                "world": dist.get_world_size(), "backend": "gloo"}
with open(dest, "w") as handle:
    json.dump(out, handle)
dist.destroy_process_group()
'''
)


def _with_env(env, fn):
    """Run `fn` with `env` applied to os.environ; a value of None unsets. Restores on exit."""
    previous = {key: os.environ.get(key) for key in env}
    for key, value in env.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value
    try:
        return fn()
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _shim_world(world, what, **env):
    """A shim world; the three variables this suite reads are unset unless given."""
    full = {"TORCHNATIVE_DIST_TRANSPORT": None, "TORCHNATIVE_DIST_BIND_ADDR": None,
            "P2P_INIT_ONLY": None}
    full.update(env)
    return _with_env(full, lambda: _c2_spawn(_P2P_SHIM_SRC, world, "float32",
                                             True, what))


@functools.cache
def p2p_shim(world):
    return _shim_world(world, "p2p-shim%d" % world)


@functools.cache
def p2p_gloo(world):
    return _c2_spawn(_P2P_GLOO_SRC, world, "float32", False, "p2p-gloo%d" % world)


@functools.cache
def p2p_shim_forced_tcp():
    return _shim_world(2, "p2p-shim-tcp", TORCHNATIVE_DIST_TRANSPORT="tcp")


@functools.cache
def p2p_shim_forced_inproc():
    return _shim_world(2, "p2p-shim-inproc", TORCHNATIVE_DIST_TRANSPORT="inproc",
                       P2P_INIT_ONLY="1")


def _helpers():
    """The probe's pure-Python input generators, for oracles in this process."""
    namespace = {}
    exec(compile(_P2P_PROBE_SRC, "<p2p probe>", "exec"), namespace)
    return namespace


#: Every probe whose value both sides produce. Equality, no tolerance.
P2P_VALUE_PROBES = (
    ("ring_sync_float32", "ring_sync_int64")
    + tuple("ring_" + n for n in ("float32", "float64", "float16", "bfloat16",
                                   "int8", "int16", "int32", "int64", "uint8",
                                   "bool"))
    + ("pair_xor2", "tags_out_of_order", "isend_irecv_float32",
       "isend_irecv_int32", "batch_ring")
)


def test_point_to_point_agrees_with_gloo_bit_for_bit_at_world_two_and_four():
    """Ring, non-neighbour pairs, tags, isend/irecv, batch: equal to gloo, element for element.

    Point-to-point combines no numbers, so there is no tolerance to derive:
    the shim's received bytes either are the sender's or they are wrong.
    """
    seen = 0
    for world in P2P_WORLDS:
        shim, gloo = p2p_shim(world), p2p_gloo(world)
        for rank in range(world):
            for key in P2P_VALUE_PROBES:
                ours = c2_ok(shim[rank], key)
                theirs = c2_ok(gloo[rank], key)
                assert ours == theirs, (
                    "%s at world %d rank %d differs from upstream gloo.\n"
                    "  shim: %r\n  gloo: %r" % (key, world, rank, ours, theirs))
                seen += 1
    assert seen == len(P2P_VALUE_PROBES) * sum(P2P_WORLDS), seen


def test_what_arrived_is_what_the_peer_sent_not_only_what_gloo_got():
    """The same results against the inputs themselves, so the oracle is not only gloo.

    If gloo and the shim were both wrong the same way (they share the probe),
    the differential above would pass; this rebuilds each sender's values from
    the generator and checks the receiver holds them.
    """
    h = _helpers()
    for world in P2P_WORLDS:
        shim = p2p_shim(world)
        for rank in range(world):
            prv = (rank - 1) % world
            for name in h["P2P_DTYPES"]:
                got = c2_ok(shim[rank], "ring_" + name)
                assert got == h["p2p_values"](name, prv, 5, "ring" + name), (
                    world, rank, name, got)
            assert c2_ok(shim[rank], "ring_sync_float32") == h["p2p_values"](
                "float32", prv, 5, "syncfloat32"), (world, rank)
            assert c2_ok(shim[rank], "batch_ring")["values"] == h["p2p_values"](
                "float32", prv, 4, "batch"), (world, rank)
            assert c2_ok(shim[rank], "batch_ring")["requests"] == 2, (world, rank)
        if world == 4:
            for rank in range(4):
                assert c2_ok(shim[rank], "pair_xor2") == h["p2p_values"](
                    "float32", rank ^ 2, 4, "xor2"), (
                    "rank %d did not receive from its non-neighbour rank %d"
                    % (rank, rank ^ 2))


def test_tags_match_even_when_the_sender_sent_them_in_the_other_order():
    """Rank 0 sends tag 2 then tag 1; rank 1 takes tag 1 first and gets tag 1's tensor."""
    h = _helpers()
    for world in P2P_WORLDS:
        first, second = c2_ok(p2p_shim(world)[1], "tags_out_of_order")
        assert first == h["p2p_values"]("int64", 0, 3, "tagB"), (world, first)
        assert second == h["p2p_values"]("int64", 0, 3, "tagA"), (world, second)
        assert first != second, "the two payloads must differ for this to test anything"


def test_irecv_is_genuinely_pending_until_the_sender_sends_and_owns_its_buffer():
    """Both backends report an irecv incomplete while its sender is still asleep.

    And on the shim the buffer read before `wait()` is exactly what it held
    before the receive (AsyncWork's ownership rule): published at a
    synchronisation point, never underneath the caller.
    """
    h = _helpers()
    for world in P2P_WORLDS:
        ours = c2_ok(p2p_shim(world)[0], "irecv_pending")
        theirs = c2_ok(p2p_gloo(world)[0], "irecv_pending")
        assert theirs["completed_before_wait"] is False, (
            "upstream gloo reported a finished irecv before its sender woke; "
            "the other side of this comparison is gone", theirs)
        assert ours["completed_before_wait"] is False, (
            "the shim's irecv was complete before its sender sent", ours)
        assert ours["before"] == [0.0, 0.0, 0.0], ours
        assert ours["wait_returned"] is True and theirs["wait_returned"] is True
        want = h["p2p_values"]("float32", 1, 3, "pending")
        assert ours["after"] == theirs["after"] == want, (ours, theirs, want)


def test_recv_anysource_still_refuses_by_name():
    """It has no consumer (BACKEND.md §4.4) and stays refused now that send/recv work."""
    for world in P2P_WORLDS:
        for rank in range(world):
            msg = c2_exc(p2p_shim(world)[rank], "recv_anysource")
            assert "NotImplementedError" in msg, msg
            assert "recv_anysource" in msg, msg


def test_a_dtype_disagreement_refuses_naming_both_and_writes_nothing():
    for world in P2P_WORLDS:
        got = c2_ok(p2p_shim(world)[1], "dtype_mismatch")
        assert got["exc"] is not None, got
        assert "torch.int64" in got["exc"] and "torch.float32" in got["exc"], got
        assert "rank 0" in got["exc"], got
        assert got["buf"] == [7.0, 7.0, 7.0], got


def test_a_peer_that_vanishes_is_named_by_the_recv_waiting_on_it():
    """The highest rank exits; each survivor's pending recv raises naming it, well before the timeout."""
    for world in P2P_WORLDS:
        shim = p2p_shim(world)
        assert c2_ok(shim[world - 1], "peer_vanished") == "left"
        for rank in range(world - 1):
            got = c2_ok(shim[rank], "peer_vanished")
            assert got["exc"] is not None, (world, rank, got)
            assert "rank %d" % (world - 1) in got["exc"], got
            assert "connection closed" in got["exc"], got
            # The group timeout is 120 s; a vanished peer is noticed at EOF.
            assert got["seconds"] < 60, got


def test_two_processes_on_one_host_default_to_shm():
    for world in P2P_WORLDS:
        for rank in range(world):
            kinds = p2p_shim(world)[rank]["_meta"]["kinds"]
            assert kinds == {str(p): "shm" for p in range(world) if p != rank}, (
                world, rank, kinds)


def test_forcing_tcp_changes_the_transport_used_and_not_the_values():
    forced, gloo = p2p_shim_forced_tcp(), p2p_gloo(2)
    for rank in range(2):
        assert forced[rank]["_meta"]["kinds"] == {str(1 - rank): "tcp"}, forced[rank]["_meta"]
        for key in P2P_VALUE_PROBES:
            assert c2_ok(forced[rank], key) == c2_ok(gloo[rank], key), (key, rank)


def test_forcing_inproc_across_processes_refuses_by_name():
    """An override the pair cannot have is refused, never quietly replaced by another transport."""
    for rank in range(2):
        msg = p2p_shim_forced_inproc()[rank].get("init_exc")
        assert msg is not None, p2p_shim_forced_inproc()[rank]
        assert msg.startswith("NotImplementedError:"), msg
        assert "TORCHNATIVE_DIST_TRANSPORT=inproc" in msg, msg
        assert "two processes" in msg, msg


def test_every_rank_really_ran_the_shim_and_the_oracle_really_ran_upstream():
    for world in P2P_WORLDS:
        for rank in range(world):
            meta = p2p_shim(world)[rank]["_meta"]
            assert (meta["shim"], meta["rank"], meta["world"], meta["backend"]) == (
                True, rank, world, "torchnative"), meta
            meta = p2p_gloo(world)[rank]["_meta"]
            assert (meta["shim"], meta["rank"], meta["world"], meta["backend"]) == (
                False, rank, world, "gloo"), meta


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
