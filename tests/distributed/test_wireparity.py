"""The collectives on the binary wire, against `ProcessGroupLocal` and gloo (issue #232).

BACKEND.md §9 step 3, SPEC S6.10. The `torchnative` backend's collectives
moved off `ProcessGroupLocal`'s star of JSON sockets onto the full mesh's
binary frames (`torchnative/python/torchnative/distributed/backend/collectives.py`).
The shape is still a star through rank 0 with the ascending-rank fold, so the
claim is **bit equality with `ProcessGroupLocal`**, not a tolerance, and this
file holds it with two oracles:

* **`ProcessGroupLocal` itself**, run as a second world in the same shim on the
  same inputs (registered under a test-only backend name, `tnoracle`, whose
  creator returns a plain `ProcessGroupLocal`). Every probe's value, and every
  refusal's text, must be equal, not close.
* **upstream gloo**, through `test_collect2`'s harness and its derived
  tolerances (imported, not copied), so the binary path is not only equal to
  something that was itself once compared with gloo.

In every binary-wire child the JSON star is sabotaged before the group is
built: `ProcessGroupLocal._star_exchange`, `ProcessGroupTorchnative._star_exchange`
and `ProcessGroupLocal._recv_json` are replaced by a function that records the
call and raises. A collective that still went through the star would fail its
probe there and leave a record in `_meta["star_calls"]`.

Written before the implementation was run anywhere (AGENTS.md rule 10: agents
do not run suites). The red run was not observed locally: before
`collectives.py` existed, every binary child's probes would have raised from
the sabotaged star (the collectives were `ProcessGroupLocal`'s) and
`_collective_counters` did not exist. The first run is CI's.
"""

import functools

from test_collect2 import (
    C2_VALUE_PROBES,
    C2_WORLDS,
    _C2_PROBE_SRC,
    _c2_spawn,
    c2_assert_matches,
    c2_exc,
    c2_gloo32,
    c2_gloo64,
    c2_ok,
)


# ---------------------------------------------------------------------------
# Child sources
# ---------------------------------------------------------------------------

_WP_HEAD = (
    "import datetime, json, os, sys, time\n"
    "import torch\n"
    'assert hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded upstream torch, not the shim")\n'
    "import torch.distributed as dist\n"
    "import torchnative.distributed  # noqa: F401 -- registers backend='torchnative'\n"
    "from torch._C._distributed_c10d import ProcessGroupLocal\n"
)

#: Installed before the group exists, in the binary children only.
_WP_SABOTAGE = r'''
from torchnative.distributed.backend import ProcessGroupTorchnative, collectives
STAR = []

def _json_star(self, *args, **kwargs):
    STAR.append(type(self).__name__)
    raise AssertionError(
        "wireparity: the JSON star ran on %s; the collectives were supposed "
        "to be on the binary wire" % type(self).__name__)

ProcessGroupLocal._star_exchange = _json_star
ProcessGroupTorchnative._star_exchange = _json_star
ProcessGroupLocal._recv_json = _json_star
'''

#: Registers the oracle: a plain ProcessGroupLocal, star and JSON and all.
_WP_ORACLE_REGISTER = r'''
def _oracle(opts, backend_options=None):
    return ProcessGroupLocal(getattr(opts, "group_rank", 0),
                             getattr(opts, "group_size", 1),
                             getattr(opts, "store", None))

dist.Backend.register_backend("tnoracle", _oracle, extended_api=True,
                              devices=["cpu"])
'''

#: Probes beyond `c2_run`: more dtypes, and one allreduce whose bytes are known.
_WP_EXTRA_SRC = r'''
WP_DTYPES = ("float64", "float16", "bfloat16", "int32", "int8", "uint8")

def wp_values(rank, name, tag, n=5):
    if name.startswith("float") or name.startswith("bfloat"):
        return c2_values(rank, n, tag)
    ints = c2_ints(rank, n, tag)
    if name == "uint8":
        return [abs(v) % 7 for v in ints]
    return [(v % 7) - 3 for v in ints]

def wp_run(torch, dist, rank, world, be):
    out = {}

    def probe(key, fn):
        try:
            out[key] = {"ok": fn()}
        except Exception as exc:
            out[key] = {"exc": "%s: %s" % (type(exc).__name__, exc)}

    def allreduce(name, op):
        t = torch.tensor(wp_values(rank, name, "dt%s%s" % (name, op)),
                         dtype=getattr(torch, name))
        dist.all_reduce(t, op=getattr(dist.ReduceOp, op))
        return t.tolist()

    def all_gather(name):
        dt = getattr(torch, name)
        holders = [torch.zeros(5, dtype=dt) for _ in range(world)]
        dist.all_gather(holders, torch.tensor(
            wp_values(rank, name, "dtag" + name), dtype=dt))
        return [h.tolist() for h in holders]

    def reduce_scatter(name):
        dt = getattr(torch, name)
        result = torch.zeros(2, dtype=dt)
        chunks = [torch.tensor(wp_values(rank, name, "dtrs%s%d" % (name, j), 2),
                               dtype=dt) for j in range(world)]
        dist.reduce_scatter(result, chunks, op=dist.ReduceOp.SUM)
        return result.tolist()

    for name in WP_DTYPES:
        floating = name.startswith("float") or name.startswith("bfloat")
        ops = (("SUM", "MIN", "MAX", "PRODUCT", "AVG") if floating
               else ("SUM", "MIN", "MAX", "PRODUCT"))
        for op in ops:
            probe("dt_%s_allreduce_%s" % (name, op),
                  lambda name=name, op=op: allreduce(name, op))
        probe("dt_%s_all_gather" % name, lambda name=name: all_gather(name))
        probe("dt_%s_reduce_scatter" % name,
              lambda name=name: reduce_scatter(name))

    for op in ("MIN", "MAX"):
        def bools(op=op):
            t = torch.tensor([bool((v + rank) % 2)
                              for v in c2_ints(rank, 6, "bool" + op)],
                             dtype=torch.bool)
            dist.all_reduce(t, op=getattr(dist.ReduceOp, op))
            return t.tolist()
        probe("dt_bool_allreduce_" + op, bools)

    # The allreduce whose bytes are known in advance: 1000 float32 elements
    # are 4000 payload bytes. JSON of the same numbers is about five times
    # that, so an exact count says the bytes went out raw.
    def known_bytes():
        counters = getattr(be, "_collective_counters", None)
        before = counters().get("allreduce", {}) if counters else {}
        t = torch.tensor(c2_values(rank, 1000, "bytes"), dtype=torch.float32)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        if not counters:
            return {"result": t.tolist(), "delta": None}
        after = counters().get("allreduce", {})
        return {"result": t.tolist(),
                "delta": {k: after.get(k, 0) - before.get(k, 0)
                          for k in after}}
    probe("known_bytes", known_bytes)
    return out
'''

_WP_ARGS = r'''
rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
DT = getattr(torch, sys.argv[5])
'''

_WP_BINARY_SRC = (
    _WP_HEAD + _WP_SABOTAGE + _C2_PROBE_SRC + _WP_EXTRA_SRC + _WP_ARGS
    + r'''
dist.init_process_group(backend="torchnative",
                        init_method="tcp://127.0.0.1:%d" % port,
                        rank=rank, world_size=world)
be = dist.distributed_c10d._get_default_group()._get_backend(torch.device("cpu"))
out = c2_run(torch, dist, rank, world, DT, True)
out.update(wp_run(torch, dist, rank, world, be))
names = sorted(collectives.BODIES)
out["_meta"] = {
    "rank": dist.get_rank(), "world": dist.get_world_size(),
    "backend_class": type(be).__name__,
    "star_calls": list(STAR),
    "star_built": bool(be._peers) or be._hub is not None,
    "counters": be._collective_counters(),
    # The oracle's own async wrapper code, around this module's bodies.
    "wrapped": {n: getattr(type(be), n).__code__
                is ProcessGroupLocal.__dict__[n].__code__ for n in names},
    "bodies_are_ours": {n: getattr(type(be), "_sync_" + n)
                        is collectives.BODIES[n] for n in names},
}
with open(dest, "w") as handle:
    json.dump(out, handle)
'''
)

_WP_ORACLE_SRC = (
    _WP_HEAD + _WP_ORACLE_REGISTER + _C2_PROBE_SRC + _WP_EXTRA_SRC + _WP_ARGS
    + r'''
dist.init_process_group(backend="tnoracle",
                        init_method="tcp://127.0.0.1:%d" % port,
                        rank=rank, world_size=world)
be = dist.distributed_c10d._get_default_group()._get_backend(torch.device("cpu"))
assert type(be) is ProcessGroupLocal, type(be)
out = c2_run(torch, dist, rank, world, DT, True)
out.update(wp_run(torch, dist, rank, world, be))
out["_meta"] = {"rank": dist.get_rank(), "world": dist.get_world_size(),
                "backend_class": type(be).__name__}
with open(dest, "w") as handle:
    json.dump(out, handle)
'''
)

#: Partial attendance: the last rank leaves a live group without a goodbye.
_WP_PARTIAL_BODY = r'''
live = torch.tensor([float(rank) + 1.0])
dist.all_reduce(live, op=dist.ReduceOp.SUM)
out = {"live": live.tolist()}
if rank == world - 1:
    out["_meta"] = {"rank": rank, "left": True}
    with open(dest, "w") as handle:
        json.dump(out, handle)
    os._exit(0)

be = dist.distributed_c10d._get_default_group()._get_backend(torch.device("cpu"))

class _Opts:
    pass

def opts(op):
    o = _Opts()
    o.reduceOp = getattr(dist.ReduceOp, op)
    return o

def probe(key, fn):
    try:
        out[key] = {"ok": fn()}
    except Exception as exc:
        out[key] = {"exc": "%s: %s" % (type(exc).__name__, exc)}

def partial_reduce(op):
    # One tag for every op, so SUM and AVG reduce the same numbers and the
    # test can check AVG's divisor against SUM directly.
    t = torch.tensor(c2_values(rank, 5, "partial"), dtype=torch.float32)
    _work, missing = be.allreduce_partial([t], opts(op))
    return {"value": t.tolist(), "missing": list(missing)}

for op in ("SUM", "AVG", "PRODUCT", "MAX"):
    probe("partial_" + op, lambda op=op: partial_reduce(op))

def partial_gather():
    holders = [torch.full((3,), -7.0) for _ in range(world)]
    src = torch.tensor(c2_values(rank, 3, "pgather"), dtype=torch.float32)
    _work, missing = be.allgather_partial([holders], [src])
    return {"slots": [h.tolist() for h in holders], "missing": list(missing)}
probe("partial_gather", partial_gather)

def strict():
    dist.all_reduce(torch.ones(2), op=dist.ReduceOp.SUM)
    return "ACCEPTED"
probe("strict_after_drop", strict)
'''

_WP_PARTIAL_BINARY_SRC = (
    _WP_HEAD + _WP_SABOTAGE + _C2_PROBE_SRC + _WP_ARGS
    + r'''
dist.init_process_group(backend="torchnative",
                        init_method="tcp://127.0.0.1:%d" % port,
                        rank=rank, world_size=world,
                        timeout=datetime.timedelta(seconds=120))
''' + _WP_PARTIAL_BODY + r'''
out["_meta"] = {"rank": rank, "backend_class": type(be).__name__,
                "star_calls": list(STAR),
                "counters": be._collective_counters()}
with open(dest, "w") as handle:
    json.dump(out, handle)
'''
)

_WP_PARTIAL_ORACLE_SRC = (
    _WP_HEAD + _WP_ORACLE_REGISTER + _C2_PROBE_SRC + _WP_ARGS
    + r'''
dist.init_process_group(backend="tnoracle",
                        init_method="tcp://127.0.0.1:%d" % port,
                        rank=rank, world_size=world,
                        timeout=datetime.timedelta(seconds=120))
''' + _WP_PARTIAL_BODY + r'''
out["_meta"] = {"rank": rank, "backend_class": type(be).__name__}
with open(dest, "w") as handle:
    json.dump(out, handle)
'''
)

#: The mailbox routing, in one shim process: no world, no sockets.
_WP_ROUTING_SRC = (
    _WP_HEAD + r'''
from torchnative.distributed.backend import transport, wire
dest = sys.argv[4]
box = transport.Mailbox(0)
t = torch.tensor([1.0, 2.0, 3.0])
user = wire.decode(wire.encode_tensor(t, 5), 1)
coll = wire.decode(wire.encode_collective(t, 5), 1)
box.deliver(1, coll)
after_collective = {"user_keys": sorted(map(list, box._frames)),
                    "collective_keys": sorted(map(list, box.collective._frames))}
box.deliver(1, user)
after_user = {"user_keys": sorted(map(list, box._frames)),
              "collective_keys": sorted(map(list, box.collective._frames))}
box.peer_gone(2, wire.PeerGone, "rank 2 is gone: connection closed")
out = {"after_collective": after_collective, "after_user": after_user,
       "collective_kind": coll.kind, "user_kind": user.kind,
       "collective_values": wire.frame_to_tensor(coll).tolist(),
       "gone_reaches_collective": box.collective.is_gone(2),
       "gone_reaches_user": box.is_gone(2)}
with open(dest, "w") as handle:
    json.dump(out, handle)
'''
)


# ---------------------------------------------------------------------------
# Worlds
# ---------------------------------------------------------------------------

#: Every world the bit-equality claim is held at. float32 at both sizes of
#: `C2_WORLDS`, and bfloat16 at three, because bfloat16 leaves the tensor
#: through a different export path (`wire._bf16_fast`) and the JSON path
#: never had one.
WP_RUNS = tuple((world, "float32") for world in C2_WORLDS) + ((3, "bfloat16"),)

#: The partial-attendance world: three survivors, so `AVG` over the survivors
#: and over the world are different divisors.
WP_PARTIAL_WORLD = 4

#: Every collective this backend moved, by the name its exchange counts under.
WP_COLLECTIVES = ("allreduce", "reduce", "broadcast", "allgather",
                  "_allgather_base", "gather", "scatter", "reduce_scatter",
                  "_reduce_scatter_base", "alltoall", "alltoall_base",
                  "barrier")


@functools.cache
def wp_binary(world, dtype):
    return _c2_spawn(_WP_BINARY_SRC, world, dtype, True,
                     "wp-bin%d-%s" % (world, dtype))


@functools.cache
def wp_oracle(world, dtype):
    return _c2_spawn(_WP_ORACLE_SRC, world, dtype, True,
                     "wp-oracle%d-%s" % (world, dtype))


@functools.cache
def wp_partial(which):
    source = {"binary": _WP_PARTIAL_BINARY_SRC,
              "oracle": _WP_PARTIAL_ORACLE_SRC}[which]
    return _c2_spawn(source, WP_PARTIAL_WORLD, "float32", True,
                     "wp-partial-%s" % which, timeout=600)


#: Keys whose values legitimately differ run to run; compared apart.
_WP_TIMED = ("barrier_blocks", "async_allreduce", "_meta")


def _wp_equal(world, dtype, rank, key, ours, theirs):
    assert ours == theirs, (
        "%s at world %d (%s) rank %d: the binary wire gave %r and "
        "ProcessGroupLocal gave %r. The two fold the same tensors in the same "
        "order, so any difference, in a value or in a refusal's text, is a "
        "defect of the binary path" % (key, world, dtype, rank, ours, theirs))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_every_collective_on_the_binary_wire_equals_process_group_local_bit_for_bit():
    """Every `c2_run` probe, both worlds, equal as JSON values, not close.

    What would make it fail: a fold in another order or another dtype, an AVG
    that divides differently, a MIN/MAX that selects NaN or a signed zero
    differently from Python's `min`/`max`, a byte lost or reordered on the
    wire, a dtype conversion on the way back, a refusal whose text drifted from
    the oracle's (the texts are compared too), or a probe that raises on one
    side only. The value probes must have succeeded on the oracle, so two
    identical failures cannot pass as agreement.
    """
    compared = 0
    for world, dtype in WP_RUNS:
        ours_all, theirs_all = wp_binary(world, dtype), wp_oracle(world, dtype)
        for rank in range(world):
            ours, theirs = ours_all[rank], theirs_all[rank]
            assert ours["_meta"]["backend_class"] == "ProcessGroupTorchnative", ours["_meta"]
            assert theirs["_meta"]["backend_class"] == "ProcessGroupLocal", theirs["_meta"]
            assert set(ours) == set(theirs), sorted(set(ours) ^ set(theirs))
            for key in C2_VALUE_PROBES:
                c2_ok(theirs, key)
            for key in sorted(theirs):
                if key in _WP_TIMED or key.startswith("dt_") or key == "known_bytes":
                    continue
                _wp_equal(world, dtype, rank, key, ours[key], theirs[key])
                compared += 1

            # barrier: the same rank slept, and nobody left before it woke.
            b_ours = c2_ok(ours, "barrier_blocks")
            b_theirs = c2_ok(theirs, "barrier_blocks")
            assert b_ours["slept"] == b_theirs["slept"], (b_ours, b_theirs)
            assert b_ours["elapsed"] >= 0.9, (world, rank, b_ours)

            # async: the same answer, and the early read is the input.
            a_ours = c2_ok(ours, "async_allreduce")
            a_theirs = c2_ok(theirs, "async_allreduce")
            for field in ("source", "before_wait", "after_wait", "wait_returned"):
                _wp_equal(world, dtype, rank, "async_allreduce." + field,
                          a_ours[field], a_theirs[field])
            assert a_ours["before_wait"] == a_ours["source"], a_ours
            if rank == 0:
                # Rank 0 provably arrived first (the others sleep 0.5 s), so
                # the collective cannot have finished when it polled.
                assert a_ours["completed_before_wait"] is False, a_ours
                assert a_theirs["completed_before_wait"] is False, a_theirs
    # Every non-timed c2 key, at every rank of every run. Stated so that a
    # probe dropped from both sides is a failure, not a smaller green number.
    assert compared >= len(C2_VALUE_PROBES) * sum(w for w, _ in WP_RUNS), compared


def test_every_dtype_on_the_binary_wire_equals_process_group_local_bit_for_bit():
    """float64, float16, bfloat16, int32, int8, uint8 and bool, every reduce op.

    What would make it fail: a dtype the wire exports wrongly (bfloat16 takes a
    separate path), a MIN/MAX on bool or a small integer type that compares
    differently from Python's, a PRODUCT that wraps differently, a
    `reduce_scatter` that converts a chunk to the wrong dtype before folding.
    Each key must have succeeded on the oracle, so a dtype that both sides
    refuse identically fails here instead of agreeing vacuously.
    """
    seen = 0
    for world, dtype in WP_RUNS:
        ours_all, theirs_all = wp_binary(world, dtype), wp_oracle(world, dtype)
        for rank in range(world):
            keys = sorted(k for k in theirs_all[rank] if k.startswith("dt_"))
            # 3 float dtypes x (5 ops + 2) + 3 integer dtypes x (4 ops + 2)
            # + 2 bool ops.
            assert len(keys) == 41, keys
            for key in keys:
                theirs = c2_ok(theirs_all[rank], key)
                ours = c2_ok(ours_all[rank], key)
                _wp_equal(world, dtype, rank, key, ours, theirs)
                seen += 1
            ours = c2_ok(ours_all[rank], "known_bytes")["result"]
            theirs = c2_ok(theirs_all[rank], "known_bytes")["result"]
            _wp_equal(world, dtype, rank, "known_bytes", ours, theirs)
    assert seen > 0, seen


def test_the_binary_wire_agrees_with_gloo_with_the_json_star_disabled():
    """`test_collect2`'s comparison with gloo, re-run on the sabotaged children.

    `test_collect2` runs on `backend="torchnative"` too, so it now measures the
    binary path as well; but nothing there says which path ran. Here the star
    raises, so agreement can only have come from the binary wire.

    What would make it fail: any value outside upstream's own derived
    tolerance (bit equality where upstream's float32 and float64 agree), and
    the bitwise refusals losing their name.
    """
    seen = 0
    for world in C2_WORLDS:
        ours_all = wp_binary(world, "float32")
        up32, up64 = c2_gloo32(world), c2_gloo64(world)
        for rank in range(world):
            for key in C2_VALUE_PROBES:
                c2_assert_matches(key, world, rank,
                                  c2_ok(ours_all[rank], key),
                                  c2_ok(up32[rank], key),
                                  c2_ok(up64[rank], key))
                seen += 1
            for opname in ("BAND", "BOR", "BXOR"):
                msg = c2_exc(ours_all[rank], "allreduce_" + opname)
                assert "ProcessGroupLocal.allreduce" in msg, msg
                assert "ReduceOp.%s" % opname in msg, msg
    assert seen == len(C2_VALUE_PROBES) * sum(C2_WORLDS), seen


def test_every_collective_put_binary_frames_on_the_mesh_and_none_used_the_json_star():
    """The sabotage check: the bytes went over the mesh, raw, and not as JSON.

    Three things, each with its own failure:

    * `star_calls` is empty on every rank and the star's sockets were never
      built. Fails if any collective body still reaches
      `ProcessGroupLocal._star_exchange` or `_recv_json`.
    * every one of the twelve collectives has exchanges and frames in both
      directions in this rank's mesh counters. Fails if one is served without
      the mesh (a local copy, a forgotten override, a path the counter does not
      see).
    * one float32 allreduce of 1000 elements costs a leaf exactly 4072 bytes
      out (a 40-byte header frame plus 24 + 8 + 4000) and 4080 in (a 48-byte
      verdict plus the same tensor frame), and the hub the mirror image per
      leaf. JSON of those numbers is about 20 000 bytes; any text encoding, an
      extra frame, or a second copy of the tensor fails the equality.
    """
    for world, dtype in WP_RUNS:
        for rank, report in enumerate(wp_binary(world, dtype)):
            meta = report["_meta"]
            assert meta["star_calls"] == [], (world, rank, meta["star_calls"])
            assert meta["star_built"] is False, (world, rank)
            counters = meta["counters"]
            for what in WP_COLLECTIVES:
                c = counters.get(what)
                assert c is not None, (
                    "%s at world %d rank %d never reached the mesh; counters "
                    "hold only %s" % (what, world, rank, sorted(counters)))
                assert c["exchanges"] > 0, (what, c)
                assert c["frames_sent"] > 0 and c["bytes_sent"] > 0, (what, c)
                assert c["frames_received"] > 0 and c["bytes_received"] > 0, (
                    what, c)
            delta = c2_ok(report, "known_bytes")["delta"]
            tensor_frame = 24 + 8 + 4000
            if rank == 0:
                leaves = world - 1
                want = {"exchanges": 1,
                        "frames_sent": 2 * leaves,
                        "bytes_sent": (48 + tensor_frame) * leaves,
                        "frames_received": 2 * leaves,
                        "bytes_received": (40 + tensor_frame) * leaves}
            else:
                want = {"exchanges": 1, "frames_sent": 2,
                        "bytes_sent": 40 + tensor_frame,
                        "frames_received": 2,
                        "bytes_received": 48 + tensor_frame}
            assert delta == want, (world, dtype, rank, delta, want)


def test_async_op_runs_the_oracles_own_wrapper_around_the_binary_bodies():
    """`async_op=True` is the oracle's wrapper code, with this backend's bodies.

    What would make it fail: an override left unwrapped (the base class wraps
    only its own methods when it is created), so `async_op=True` ran inline;
    a hand-written second wrapper (its code object is not the oracle's); or
    the wrapper bound to `ProcessGroupLocal`'s JSON bodies instead of the
    binary ones. The behaviour half (the early read is the input, rank 0's
    poll is False) is in the first test and in `test_asyncwork.py`, which runs
    on this backend.
    """
    for world, dtype in WP_RUNS:
        for rank, report in enumerate(wp_binary(world, dtype)):
            meta = report["_meta"]
            assert sorted(meta["wrapped"]) == sorted(WP_COLLECTIVES), meta["wrapped"]
            assert all(meta["wrapped"].values()), meta["wrapped"]
            assert all(meta["bodies_are_ours"].values()), meta["bodies_are_ours"]


def test_partial_attendance_on_the_binary_wire_equals_process_group_local_when_a_rank_dies():
    """The last of four ranks leaves a live group; the survivors agree with the oracle.

    `allreduce_partial` (SUM, AVG, PRODUCT, MAX) and `allgather_partial` over
    the three survivors, then a strict `all_reduce`, in a binary world and in a
    `ProcessGroupLocal` world given the same inputs.

    What would make it fail: a survivor set that is not `[3]` on every
    survivor (the hub's verdict not travelling, or a leaf deciding for
    itself); AVG dividing by the world rather than the survivors; a dead
    rank's slot written rather than left at the caller's -7.0; a strict
    collective that returns instead of raising, or raises without naming rank
    3 and "connection closed" (which `torchnative.federated` reads as a lost
    rank); and any value differing from the oracle's by a bit.
    """
    ours_all, theirs_all = wp_partial("binary"), wp_partial("oracle")
    world = WP_PARTIAL_WORLD
    assert ours_all[world - 1]["_meta"]["left"] is True
    assert theirs_all[world - 1]["_meta"]["left"] is True
    for rank in range(world - 1):
        ours, theirs = ours_all[rank], theirs_all[rank]
        assert ours["_meta"]["star_calls"] == [], ours["_meta"]
        assert ours["_meta"]["backend_class"] == "ProcessGroupTorchnative"
        assert theirs["_meta"]["backend_class"] == "ProcessGroupLocal"
        assert ours["live"] == theirs["live"] == [float(sum(range(1, world + 1)))]
        for key in ("partial_SUM", "partial_AVG", "partial_PRODUCT",
                    "partial_MAX", "partial_gather"):
            got, want = c2_ok(ours, key), c2_ok(theirs, key)
            assert got["missing"] == [world - 1], (rank, key, got)
            _wp_equal(world, "float32", rank, key, got, want)
        total = c2_ok(ours, "partial_SUM")["value"]
        mean = c2_ok(ours, "partial_AVG")["value"]
        assert len(total) == len(mean) == 5
        slots = c2_ok(ours, "partial_gather")["slots"]
        assert slots[world - 1] == [-7.0, -7.0, -7.0], slots
        assert all(s != [-7.0, -7.0, -7.0] for s in slots[:world - 1]), slots
        msg = c2_exc(ours, "strict_after_drop")
        assert "connection closed" in msg and "[%d]" % (world - 1) in msg, msg
        _wp_equal(world, "float32", rank, "strict_after_drop", msg,
                  c2_exc(theirs, "strict_after_drop"))
    # AVG divided by the three survivors, not by the world of four: checked
    # against the binary run's own SUM of the same inputs, so the divisor is
    # held without trusting either oracle.
    survivors = world - 1
    for rank in range(survivors):
        total = c2_ok(ours_all[rank], "partial_SUM")["value"]
        mean = c2_ok(ours_all[rank], "partial_AVG")["value"]
        for t, m in zip(total, mean):
            assert abs(t / survivors - m) <= 1e-6 * max(abs(m), 1.0), (
                rank, t, m)
        assert any(abs(t / world - m) > 1e-4 for t, m in zip(total, mean)), (
            "AVG at rank %d equals SUM / %d as well as SUM / %d, so these "
            "inputs cannot tell the two divisors apart" % (rank, world,
                                                           survivors))


def test_a_collective_frame_never_matches_a_user_receive():
    """The mesh routes `KIND_COLLECTIVE` to a mailbox of its own.

    A user's tag may be any signed 64-bit value, so collective frames are told
    apart by kind, not by a tag range. What would make it fail: the routing in
    `transport.Mailbox.deliver` removed (the collective frame would land in the
    user mailbox under `(1, 5)`, where a `recv(src=1, tag=5)` would take it),
    or a lost peer not reaching the collectives' mailbox (a collective waiting
    on a dead rank would then wait out the group timeout instead of naming it).
    """
    import json
    import os
    import subprocess
    import sys
    import tempfile
    from test_shim import _CKPT_VENDOR_DIR

    with tempfile.TemporaryDirectory(prefix="wp-routing-") as tmp:
        dest = os.path.join(tmp, "out.json")
        env = dict(os.environ)
        env["PYTHONPATH"] = _CKPT_VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
        proc = subprocess.run(
            [sys.executable, "-c", _WP_ROUTING_SRC, "0", "0", "1", dest,
             "float32"],
            capture_output=True, text=True, env=env, timeout=300)
        assert proc.returncode == 0 and os.path.exists(dest), (
            proc.returncode, proc.stdout[-3000:], proc.stderr[-6000:])
        with open(dest) as handle:
            out = json.load(handle)
    assert out["collective_kind"] == 4 and out["user_kind"] == 1, out
    assert out["after_collective"] == {"user_keys": [],
                                       "collective_keys": [[1, 5]]}, out
    assert out["after_user"] == {"user_keys": [[1, 5]],
                                 "collective_keys": [[1, 5]]}, out
    assert out["collective_values"] == [1.0, 2.0, 3.0], out
    assert out["gone_reaches_collective"] is True, out
    assert out["gone_reaches_user"] is True, out


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
