"""Federated aggregation on the server instead of all_reduce (issue #282).

BACKEND.md D14 (leader) and §9 step 15; SPEC S6.7. `FederatedDataParallel(
aggregation='server')` sends each rank's weighted delta to the coordinator the
group rendezvoused through (`tnserver://`, `backend/server.py`), which folds the
deltas and the weights in ascending rank order and sends one aggregate back.
The option's name, the route and the server's `fed_gather` / `fed_fold`
operations are agent proposals, not confirmed (docs/distributed/FEDSERVER.md).

Four kinds of test:

* **the server alone**, in this process with no torch: a fold whose answer
  depends on the order, worked out by hand; a gather; and the refusals the
  server makes (a missing rank, a rank twice, tables that do not line up,
  float16).
* **agreement**: in one world of three, the same three rounds of FedAvg,
  FedAvgM and FedProx run once with `aggregation='all_reduce'` and once with
  `aggregation='server'` from the same weights on the same data. The two
  aggregates are compared **bit for bit**, against an `InProcessServer` and a
  `StandaloneServer`. Bit equality is the right comparison and not a
  tolerance: both paths fold the same scaled float32 tables in the same
  ascending rank order, one addition and one rounding at a time, and divide
  with the same expression (AGENTS.md §16: a tolerance is derived, and here the
  derivation gives zero). The data are checked to be order-sensitive, so a
  fold in any other order would not pass.
* **the server really aggregated**, from the server's own ledger: every fold
  the ranks were answered with is in it, over ranks [0, 1, 2], with a SHA-256
  that matches the ascending fold of the deltas the ranks handed in; and the
  participants' own `all_reduce` / `all_gather` ran zero times in a server
  round (and more than zero in an all_reduce round, so the counter counts).
* **refusals by name**: a rank that never hands in, a group without a server,
  and every combination `aggregation='server'` does not serve.

The process harness is `test_collect2.py`'s `_c2_spawn`, imported. Written
before anything was run (AGENTS.md rule 10); the first run is CI's.
"""

import functools
import hashlib
import importlib.util
import os
import struct
import threading

from test_collect2 import _c2_spawn, c2_exc, c2_ok
from test_p2p import _with_env


FS_WORLD = 3
FS_ROUNDS = 3
FS_ALGORITHMS = ("FedAvg", "FedAvgM", "FedProx")
FS_TIMEOUT_S = 10
FS_WEIGHTS = {0: 3.0, 1: 7.0, 2: 5.0}

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SERVER_PY = os.path.join(_REPO, "torchnative", "python", "torchnative",
                          "distributed", "backend", "server.py")

#: Every variable this suite sets in a child; unset unless a test gives it.
_FS_ENV = ("TORCHNATIVE_DIST_KEY", "TORCHNATIVE_DIST_ALLOW_LAN",
           "TORCHNATIVE_DIST_BIND_ADDR", "TORCHNATIVE_DIST_TRANSPORT",
           "FS_MODE", "FS_URL", "FS_SCENARIO", "FS_TIMEOUT")


def _server_module():
    """`server.py` loaded by path: stdlib only at import, so no torch here."""
    spec = importlib.util.spec_from_file_location("tn_fedserver_under_test",
                                                  _SERVER_PY)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _f32(x):
    """`x` rounded to binary32, as a Python float."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _floats(hexed):
    raw = bytes.fromhex(hexed)
    return list(struct.unpack("<%df" % (len(raw) // 4), raw))


def _hex(values):
    return struct.pack("<%df" % len(values), *values).hex()


# ---------------------------------------------------------------------------
# the server alone, no torch
# ---------------------------------------------------------------------------

def _submit_all(mod, port, op, key, world, per_rank, timeout=10.0):
    """Every rank in `per_rank` submits at once, from its own thread. Returns {rank: result}."""
    wire = mod._wire()
    results = {}

    def one(rank, body):
        req = {"op": op, "key": key, "rank": rank, "world": world,
               "timeout": timeout}
        frames = []
        if op == "fed_gather":
            req["value"] = body
        else:
            weight, tensors, dtype = body
            req["weight"] = float(weight).hex()
            req["names"] = [name for name, _ in tensors]
            for i, (name, values) in enumerate(tensors):
                fmt = {"float32": "f", "float16": "e", "float64": "d"}[dtype]
                payload = struct.pack("<%d%s" % (len(values), fmt), *values)
                frames.append(wire.encode(wire.KIND_TENSOR, i, payload,
                                          dtype=dtype, shape=(len(values),)))
        try:
            reply, out = mod.fed_request("127.0.0.1", port, req, frames,
                                         wait_s=timeout)
            results.setdefault(rank, []).append(
                {"reply": reply, "payloads": [f.payload.hex() for f in out]})
        except Exception as exc:  # noqa: BLE001
            results.setdefault(rank, []).append(
                {"exc": "%s: %s" % (type(exc).__name__, exc)})

    threads = [threading.Thread(target=one, args=(rank, body))
               for rank, body in per_rank]
    for th in threads:
        th.start()
    for th in threads:
        th.join(timeout + 30.0)
    return results


@functools.cache
def fs_unit():
    """The server's federated operations over a real socket, from threads of this process."""
    mod = _server_module()
    srv = mod.InProcessServer("127.0.0.1", 0).start()
    try:
        host, port = srv.address()
        out = {}
        # Element 0 is the order witness, worked out by hand: in float32,
        # (1 + 2^-24) + 2^-24 rounds to 1 twice (ties to even), while
        # (2^-24 + 2^-24) + 1 is 1 + 2^-23, which is representable. Ascending
        # rank order must give exactly 1.0. Element 1 is exact in any order.
        tiny = 2.0 ** -24
        out["fold"] = _submit_all(mod, port, "fed_fold", "unit/fold", 3, [
            (0, (3.0, [("a", [1.0, 0.5])], "float32")),
            (1, (7.0, [("a", [tiny, 0.25])], "float32")),
            (2, (5.0, [("a", [tiny, 0.125])], "float32")),
        ])
        out["gather"] = _submit_all(mod, port, "fed_gather", "unit/gather", 3,
                                    [(0, 5), (1, 6), (2, 7)])
        out["missing"] = _submit_all(mod, port, "fed_fold", "unit/missing", 3, [
            (0, (1.0, [("a", [1.0])], "float32")),
            (1, (1.0, [("a", [2.0])], "float32")),
        ], timeout=1.5)
        out["twice"] = _submit_all(mod, port, "fed_fold", "unit/twice", 2, [
            (0, (1.0, [("a", [1.0])], "float32")),
            (0, (1.0, [("a", [2.0])], "float32")),
        ], timeout=5.0)
        out["mismatch"] = _submit_all(mod, port, "fed_fold", "unit/mismatch", 2, [
            (0, (1.0, [("a", [1.0])], "float32")),
            (1, (1.0, [("b", [2.0])], "float32")),
        ])
        out["half"] = _submit_all(mod, port, "fed_fold", "unit/half", 2, [
            (0, (1.0, [("h", [1.0])], "float16")),
            (1, (1.0, [("h", [2.0])], "float16")),
        ])
        out["ledger"] = srv.fed_ledger()
        return out
    finally:
        srv.stop()


def _only(results, rank):
    entries = results[rank]
    assert len(entries) == 1, (rank, entries)
    assert "exc" not in entries[0], (rank, entries[0])
    return entries[0]


def test_the_server_folds_in_ascending_rank_order_and_every_rank_gets_the_same_bits():
    """Hand-worked order witness: ascending gives 1.0 exactly, descending would give 1 + 2^-23.

    Sabotage: fold in arrival order or descending order and element 0 comes
    back as 1 + 2^-23 on some runs or on every run; fold in float64 and round
    once at the end and it comes back as 1 + 2^-23 too (1 + 2^-23 is the
    nearest float32 to 1 + 2^-23 exactly).
    """
    got = fs_unit()["fold"]
    expected = _hex([1.0, 0.875])
    for rank in range(3):
        entry = _only(got, rank)
        reply = entry["reply"]
        assert reply["ok"] is True, reply
        assert reply["ranks"] == [0, 1, 2], reply
        assert entry["payloads"] == [expected], (rank, entry["payloads"], expected)
        assert float.fromhex(reply["total_weight"]) == 15.0, reply
    shas = {_only(got, r)["reply"]["sha256"] for r in range(3)}
    assert shas == {hashlib.sha256(bytes.fromhex(expected)).hexdigest()}, shas


def test_the_server_gathers_every_ranks_value_in_rank_order():
    got = fs_unit()["gather"]
    for rank in range(3):
        reply = _only(got, rank)["reply"]
        assert reply["ok"] is True and reply["values"] == [5, 6, 7], reply


def test_the_server_refuses_by_name_and_folds_nothing_for_a_bad_round():
    """A missing rank, a rank twice, tables that do not line up, float16: no fold, a named verdict.

    Sabotage: average over the ranks that arrived and `missing` returns an
    aggregate of ranks 0 and 1; accept a duplicate and `twice` returns a fold
    with rank 0 counted twice.
    """
    unit = fs_unit()
    for rank in (0, 1):
        reply = _only(unit["missing"], rank)["reply"]
        assert reply == {"ok": False, "kind": "missing", "missing": [2],
                         "seconds": 1.5, "frames": 0}, reply
    twice = unit["twice"][0]
    assert len(twice) == 2, twice
    for entry in twice:
        assert entry["reply"]["ok"] is False, entry
        assert entry["reply"]["kind"] == "refused", entry
        assert "submitted twice" in entry["reply"]["msg"], entry
    for rank in (0, 1):
        reply = _only(unit["mismatch"], rank)["reply"]
        assert reply["ok"] is False and "different tables" in reply["msg"], reply
        reply = _only(unit["half"], rank)["reply"]
        assert reply["ok"] is False and "torch.float16" in reply["msg"], reply
    ledger = unit["ledger"]
    folds = [e for e in ledger if e["kind"] == "fold"]
    assert [e["key"] for e in folds] == ["unit/fold"], ledger
    assert folds[0]["ranks"] == [0, 1, 2], folds
    abandoned = [e for e in ledger if e["kind"] == "fold-abandoned"]
    assert [(e["key"], e["missing"]) for e in abandoned] == [("unit/missing", [2])], ledger


# ---------------------------------------------------------------------------
# process worlds
# ---------------------------------------------------------------------------

_FS_SHIM_SRC = r'''
import datetime, json, os, sys, time
import torch
assert hasattr(torch._C, "_aten_implemented"), (
    "this subprocess loaded upstream torch, not the shim")
import torch.nn as nn
import torch.distributed as dist
import torchnative.distributed  # noqa: F401  (registers the backend and tnserver://)
from torchnative import federated
from torchnative.federated import algorithms
from torchnative.distributed.backend import server, wire

rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
for name in ("TORCHNATIVE_DIST_KEY", "TORCHNATIVE_DIST_TRANSPORT",
             "TORCHNATIVE_DIST_BIND_ADDR"):
    os.environ.pop(name, None)
mode, scenario = os.environ["FS_MODE"], os.environ["FS_SCENARIO"]
timeout = float(os.environ.get("FS_TIMEOUT", "120"))
if mode == "rank0":
    url = "tnserver://127.0.0.1:%d" % port
elif mode == "external":
    url = os.environ["FS_URL"]
elif mode == "tcp":
    url = "tcp://127.0.0.1:%d" % port
else:
    raise SystemExit("unknown FS_MODE %r" % mode)
dist.init_process_group(backend="torchnative", init_method=url, rank=rank,
                        world_size=world,
                        timeout=datetime.timedelta(seconds=timeout))

SEED = 20261113
WEIGHT = {0: 3.0, 1: 7.0, 2: 5.0}[rank]

def ints(key, n, tag):
    state = (SEED + key * 7919 + sum(ord(c) * 31 for c in tag)) % 2147483647
    out = []
    for _ in range(n):
        state = (state * 1103515245 + 12345) % 2147483648
        out.append((state >> 11) % 2000 - 1000)
    return out

def floats(key, n, tag):
    # Not dyadic: sums of these round, so the fold order is observable.
    return [v / 613.0 for v in ints(key, n, tag)]

def build():
    m = nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Linear(3, 2))
    with torch.no_grad():
        for i, (_, p) in enumerate(m.named_parameters()):
            p.copy_(torch.tensor(floats(100 + i, p.numel(), "init"),
                                 dtype=torch.float32).reshape(tuple(p.shape)))
    return m

def local_train(fdp, opt, rnd, steps=3):
    x = torch.tensor(floats(rank, 20, "x%d" % rnd), dtype=torch.float32).reshape(5, 4)
    y = torch.tensor(floats(rank, 10, "y%d" % rnd), dtype=torch.float32).reshape(5, 2)
    for _ in range(steps):
        opt.zero_grad()
        loss = ((fdp(x) - y) ** 2).mean()
        loss.backward()
        opt.step()

def hx(t):
    return wire.tensor_bytes(t.detach().contiguous())[2].hex()

out = {}

def probe(key, fn):
    try:
        out[key] = {"ok": fn()}
    except Exception as exc:
        out[key] = {"exc": "%s: %s" % (type(exc).__name__, exc)}

# Counts the participants' own collectives. `torchnative.federated` looks
# `dist.all_reduce` / `dist.all_gather` up at call time, so these see every one.
COUNT = {"all_reduce": 0, "all_gather": 0}
_orig_all_reduce, _orig_all_gather = dist.all_reduce, dist.all_gather
def _counted_all_reduce(*a, **k):
    COUNT["all_reduce"] += 1
    return _orig_all_reduce(*a, **k)
def _counted_all_gather(*a, **k):
    COUNT["all_gather"] += 1
    return _orig_all_gather(*a, **k)
dist.all_reduce, dist.all_gather = _counted_all_reduce, _counted_all_gather

def make(name):
    if name == "FedAvg":
        return algorithms.FedAvg()
    if name == "FedAvgM":
        return algorithms.FedAvgM(momentum=0.9)
    return algorithms.FedProx(mu=0.05)

def recording(alg, log):
    """Keep what the algorithm was handed and what it returned, on the instance.

    On the instance rather than by subclassing, so the type is still the one
    aggregation='server' serves.
    """
    inner = alg.aggregate
    def aggregate(table, weight=None, group=None):
        result = inner(table, weight=weight, group=group)
        log.append({"local": {n: hx(t) for n, t in table.items()},
                    "weight": weight,
                    "aggregate": {n: hx(t) for n, t in result.items()}})
        return result
    alg.aggregate = aggregate
    return alg

def run(alg_name, aggregation):
    m = build()
    log = []
    alg = recording(make(alg_name), log)
    fdp = federated.FederatedDataParallel(m, algorithm=alg, aggregation=aggregation)
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    rounds = []
    for rnd in range(3):
        local_train(fdp, opt, rnd)
        before = dict(COUNT)
        ret = fdp.synchronize(weight=WEIGHT)
        after = dict(COUNT)
        info = fdp._get_federated_logging_data()
        rounds.append({
            "returned": repr(ret),
            "installed": {n: hx(p) for n, p in m.named_parameters()},
            "collectives": {k: after[k] - before[k] for k in COUNT},
            "aggregation": info["aggregation"],
            "receipts": info.get("server_receipts", []),
            "server": info.get("server"),
            "total_weight": float(info["total_weight"]).hex(),
        })
    assert len(log) == len(rounds), (len(log), len(rounds))
    for entry, record in zip(rounds, log):
        entry.update(record)
    return rounds

if scenario == "main":
    for alg_name in ("FedAvg", "FedAvgM", "FedProx"):
        for aggregation in ("all_reduce", "server"):
            probe("%s/%s" % (alg_name, aggregation),
                  lambda a=alg_name, g=aggregation: run(a, g))

    out["default_aggregation"] = federated.FederatedDataParallel(build()).aggregation
    probe("bad_aggregation", lambda: federated.FederatedDataParallel(
        build(), aggregation="ring"))
    probe("server_average_arrived", lambda: federated.FederatedDataParallel(
        build(), aggregation="server", on_missing="average_arrived",
        min_participants=2))
    probe("server_secure", lambda: federated.FederatedDataParallel(
        build(), aggregation="server", secure_aggregation=True))
    probe("server_dp", lambda: federated.FederatedDataParallel(
        build(), aggregation="server", differential_privacy=1.0))

    class MedianOfMeans(algorithms.FedAvg):
        pass
    probe("server_subclass", lambda: federated.FederatedDataParallel(
        build(), algorithm=MedianOfMeans(), aggregation="server"))

    class Custom:
        def aggregate(self, table, weight=None, group=None):
            return table
    probe("server_custom", lambda: federated.FederatedDataParallel(
        build(), algorithm=Custom(), aggregation="server"))

    def half():
        with federated._via_server(None, "the float16 probe"):
            return algorithms.FedAvg().aggregate(
                {"h": torch.ones(3, dtype=torch.float16)}, weight=WEIGHT)
    probe("server_float16", half)
    if mode == "rank0" and rank == 0:
        out["ledger"] = server._HOSTED[0].fed_ledger()

elif scenario == "missing":
    m = build()
    fdp = federated.FederatedDataParallel(m, algorithm=algorithms.FedAvg(),
                                          aggregation="server")
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    out["base"] = {n: hx(t) for n, t in fdp.base.items()}
    local_train(fdp, opt, 0)
    out["trained"] = {n: hx(p) for n, p in m.named_parameters()}
    if rank == 2:
        # The participant that never hands in: it trained, and it never calls
        # synchronize. It stays alive past the others' timeout.
        out["absent"] = True
        out["_meta"] = {"shim": True, "rank": rank, "pid": os.getpid()}
        with open(dest, "w") as handle:
            json.dump(out, handle)
        time.sleep(timeout * 1.5 + 4.0)
        sys.exit(0)
    t0 = time.time()
    try:
        fdp.synchronize(weight=WEIGHT)
        out["sync"] = {"ok": True}
    except Exception as exc:
        out["sync"] = {"exc": "%s: %s" % (type(exc).__name__, exc),
                       "type": type(exc).__name__,
                       "missing": list(getattr(exc, "missing", ())),
                       "seconds": time.time() - t0}
    out["after"] = {n: hx(p) for n, p in m.named_parameters()}
    if rank == 0:
        out["ledger"] = server._HOSTED[0].fed_ledger()

elif scenario == "noserver":
    m = build()
    fdp = federated.FederatedDataParallel(m, algorithm=algorithms.FedAvg(),
                                          aggregation="server")
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    local_train(fdp, opt, 0)
    probe("noserver", lambda: fdp.synchronize(weight=WEIGHT))

out["_meta"] = {"shim": True, "rank": dist.get_rank(),
                "world": dist.get_world_size(), "backend": dist.get_backend(),
                "pid": os.getpid()}
with open(dest, "w") as handle:
    json.dump(out, handle)
if scenario == "missing":
    # Rank 0 hosts the server: keep it up while rank 1 reads its verdict.
    if rank == 0:
        time.sleep(5.0)
    sys.exit(0)
dist.barrier()
time.sleep(1.0)
'''


def _fs_world(world, what, timeout=600, **env):
    full = {name: None for name in _FS_ENV}
    full.update(env)
    return _with_env(full, lambda: _c2_spawn(_FS_SHIM_SRC, world, "float32",
                                             True, what, timeout=timeout))


@functools.cache
def fs_inprocess():
    reports = _fs_world(FS_WORLD, "fs-inprocess", FS_MODE="rank0",
                        FS_SCENARIO="main")
    return {"reports": reports, "ledger": reports[0].get("ledger"),
            "server_pid": None}


@functools.cache
def fs_standalone():
    """A world of three through a separate server process the test itself started."""
    mod = _server_module()
    srv = mod.StandaloneServer("127.0.0.1", 0).start()
    try:
        reports = _fs_world(FS_WORLD, "fs-standalone", FS_MODE="external",
                            FS_URL=srv.url(), FS_SCENARIO="main")
        return {"reports": reports, "ledger": srv.fed_ledger(),
                "server_pid": srv.pid()}
    finally:
        srv.stop()


@functools.cache
def fs_missing():
    return _fs_world(FS_WORLD, "fs-missing", timeout=240, FS_MODE="rank0",
                     FS_SCENARIO="missing", FS_TIMEOUT=str(FS_TIMEOUT_S))


@functools.cache
def fs_noserver():
    return _fs_world(2, "fs-noserver", timeout=240, FS_MODE="tcp",
                     FS_SCENARIO="noserver", FS_TIMEOUT="60")


def _runs(world):
    """{(algorithm, aggregation, rank): [round, ...]} from one main world."""
    out = {}
    for rank in range(FS_WORLD):
        meta = world["reports"][rank]["_meta"]
        assert (meta["shim"], meta["rank"], meta["world"], meta["backend"]) == (
            True, rank, FS_WORLD, "torchnative"), meta
        for alg in FS_ALGORITHMS:
            for agg in ("all_reduce", "server"):
                rounds = c2_ok(world["reports"][rank], "%s/%s" % (alg, agg))
                assert len(rounds) == FS_ROUNDS, (alg, agg, rank, len(rounds))
                out[(alg, agg, rank)] = rounds
    return out


def _agrees_bit_for_bit(world, label):
    runs = _runs(world)
    compared = 0
    for alg in FS_ALGORITHMS:
        for rank in range(FS_WORLD):
            ours, theirs = runs[(alg, "server", rank)], runs[(alg, "all_reduce", rank)]
            for rnd in range(FS_ROUNDS):
                a, b = ours[rnd], theirs[rnd]
                assert (a["aggregation"], b["aggregation"]) == ("server", "all_reduce"), (
                    label, alg, rank, rnd)
                assert a["returned"] == "None", (label, alg, a["returned"])
                # Same inputs: otherwise equal outputs would mean nothing.
                assert a["local"] == b["local"], (
                    "%s: %s rank %d round %d: the two paths were handed different "
                    "deltas, so the comparison would not be of the aggregation"
                    % (label, alg, rank, rnd))
                assert a["aggregate"] == b["aggregate"], (
                    "%s: %s rank %d round %d: the server's aggregate differs from "
                    "the all_reduce aggregate.\n  server:     %r\n  all_reduce: %r"
                    % (label, alg, rank, rnd, a["aggregate"], b["aggregate"]))
                assert a["installed"] == b["installed"], (label, alg, rank, rnd)
                assert a["total_weight"] == b["total_weight"], (label, alg, rank, rnd)
                assert float.fromhex(a["total_weight"]) == sum(FS_WEIGHTS.values()), a
                compared += 1
        # Every rank holds the same aggregate.
        for rnd in range(FS_ROUNDS):
            seen = {repr(sorted(runs[(alg, "server", r)][rnd]["aggregate"].items()))
                    for r in range(FS_WORLD)}
            assert len(seen) == 1, (label, alg, rnd, "ranks disagree")
    assert compared == len(FS_ALGORITHMS) * FS_WORLD * FS_ROUNDS, compared


def test_server_aggregation_equals_all_reduce_aggregation_bit_for_bit_through_the_in_process_server():
    """FedAvg, FedAvgM and FedProx, three rounds each, at world 3: the same bits either way.

    Sabotage: drop one rank's delta in `_fold_compute` (or divide by a total
    missing its weight) and the aggregates differ from round 1 on; fold in
    descending order and they differ wherever the data are order-sensitive,
    which `test_the_data_pin_the_fold_order_and_the_server_matches_a_central_ascending_fold`
    shows they are.
    """
    _agrees_bit_for_bit(fs_inprocess(), "in-process server")


def test_server_aggregation_equals_all_reduce_aggregation_bit_for_bit_through_a_standalone_server_process():
    """The same, with the server in its own process (no torch there): the fold is the stdlib one."""
    _agrees_bit_for_bit(fs_standalone(), "standalone server")


def _central(rounds_by_rank, rnd):
    """Ascending and descending folds of f32(w * d) per element, from the ranks' own inputs."""
    names = sorted(rounds_by_rank[0][rnd]["local"])
    asc, desc = {}, {}
    for name in names:
        scaled = []
        for rank in range(FS_WORLD):
            entry = rounds_by_rank[rank][rnd]
            assert entry["weight"] == FS_WEIGHTS[rank], entry["weight"]
            scaled.append([_f32(v * entry["weight"])
                           for v in _floats(entry["local"][name])])
        a = scaled[0]
        for s in scaled[1:]:
            a = [_f32(x + y) for x, y in zip(a, s)]
        d = scaled[-1]
        for s in reversed(scaled[:-1]):
            d = [_f32(x + y) for x, y in zip(d, s)]
        asc[name], desc[name] = a, d
    return names, asc, desc


def _central_checks(world, label):
    runs = _runs(world)
    total = _f32(sum(FS_WEIGHTS[r] for r in range(FS_WORLD)))
    order_sensitive = 0
    for alg in FS_ALGORITHMS:
        by_rank = [runs[(alg, "server", r)] for r in range(FS_WORLD)]
        for rnd in range(FS_ROUNDS):
            names, asc, desc = _central(by_rank, rnd)
            for name in names:
                order_sensitive += sum(1 for x, y in zip(asc[name], desc[name])
                                       if x != y)
            if alg == "FedAvg":
                for rank in range(FS_WORLD):
                    got = by_rank[rank][rnd]["aggregate"]
                    for name in names:
                        want = _hex([_f32(v / total) for v in asc[name]])
                        assert got[name] == want, (
                            "%s: FedAvg rank %d round %d %r is not the central "
                            "ascending-rank fold divided by the total weight"
                            % (label, rank, rnd, name))
    return order_sensitive


def test_the_data_pin_the_fold_order_and_the_server_matches_a_central_ascending_fold():
    """The data are order-sensitive, and FedAvg's server aggregate is the central ascending fold.

    The central fold is computed here, in plain Python binary32 arithmetic,
    from the deltas and weights the ranks reported. If ascending and
    descending folds agreed on every element, bit equality with the
    all_reduce path would not pin the order and this test says so instead of
    passing.
    """
    for world, label in ((fs_inprocess(), "in-process"), (fs_standalone(), "standalone")):
        sensitive = _central_checks(world, label)
        assert sensitive > 0, (
            "%s: on these data the ascending and descending folds agree on every "
            "element, so equality with the all_reduce path cannot show the "
            "order; change the data" % label)


def _ledger_checks(world, label):
    runs = _runs(world)
    ledger = world["ledger"]
    assert ledger is not None, label
    folds = {e["key"]: e for e in ledger if e["kind"] == "fold"}
    assert len(folds) == sum(1 for e in ledger if e["kind"] == "fold"), "duplicate keys"
    assert not [e for e in ledger if e["kind"].endswith("-abandoned")], ledger
    accounted = set()
    for alg in FS_ALGORITHMS:
        by_rank = [runs[(alg, "server", r)] for r in range(FS_WORLD)]
        for rnd in range(FS_ROUNDS):
            receipts = [by_rank[r][rnd]["receipts"] for r in range(FS_WORLD)]
            assert receipts[0] == receipts[1] == receipts[2], (label, alg, rnd)
            assert len(receipts[0]) == 2, (label, alg, rnd, receipts[0])
            delta_fold = [r for r in receipts[0] if r["names"]]
            weight_fold = [r for r in receipts[0] if not r["names"]]
            assert len(delta_fold) == 1 and len(weight_fold) == 1, receipts[0]
            names, asc, _ = _central(by_rank, rnd)
            want_sha = hashlib.sha256(
                b"".join(bytes.fromhex(_hex(asc[n])) for n in names)).hexdigest()
            for receipt in receipts[0]:
                entry = folds.get(receipt["key"])
                assert entry is not None, (
                    "%s: %s round %d was answered with fold %r, which the server "
                    "never recorded" % (label, alg, rnd, receipt["key"]))
                assert entry["ranks"] == [0, 1, 2], entry
                assert entry["world"] == FS_WORLD, entry
                assert entry["sha256"] == receipt["sha256"], (entry, receipt)
                assert entry["total_weight"] == receipt["total_weight"], (entry, receipt)
                assert entry["weights"] == [FS_WEIGHTS[r].hex() for r in range(FS_WORLD)], entry
                accounted.add(receipt["key"])
            entry = folds[delta_fold[0]["key"]]
            assert entry["names"] == names, (entry, names)
            assert entry["sha256"] == want_sha, (
                "%s: %s round %d: the server's fold is not the ascending fold of "
                "the deltas the ranks handed in" % (label, alg, rnd))
            # No server receipts on the all_reduce path.
            for r in range(FS_WORLD):
                assert runs[(alg, "all_reduce", r)][rnd]["receipts"] == [], (alg, r)
    assert set(folds) == accounted, (
        "%s: the server folded rounds nobody was answered with: %s"
        % (label, sorted(set(folds) - accounted)))
    assert len(accounted) == 2 * len(FS_ALGORITHMS) * FS_ROUNDS, len(accounted)


def test_the_server_really_aggregated_every_server_round_by_its_own_ledger():
    """Server-side evidence, not equal numbers: every fold a rank was answered with is in the server's ledger.

    Each ledger entry names the ranks folded ([0, 1, 2], in that order), the
    weights, and a SHA-256 of the sums, which must equal the hash of the
    ascending fold of the deltas the ranks reported. 18 folds for 9 server
    rounds (the deltas and the weights), none for the all_reduce rounds.
    Sabotage: compute the aggregate on the ranks and use the server only as
    a relay, and the ledger has no fold entries; drop rank 2 in the fold and
    `ranks` and the SHA-256 both change.
    """
    for world, label in ((fs_inprocess(), "in-process"), (fs_standalone(), "standalone")):
        _ledger_checks(world, label)


def test_a_server_round_sends_nothing_over_the_participants_collectives():
    """`dist.all_reduce` and `dist.all_gather` counted: zero in a server round, more than zero otherwise.

    The non-zero half shows the counter counts. Sabotage: leave `agree` or
    `_total_weight` on the all_reduce path inside a server round and the
    server-round count is no longer zero.
    """
    for world, label in ((fs_inprocess(), "in-process"), (fs_standalone(), "standalone")):
        runs = _runs(world)
        for (alg, agg, rank), rounds in runs.items():
            for rnd, entry in enumerate(rounds):
                calls = entry["collectives"]
                if agg == "server":
                    assert calls == {"all_reduce": 0, "all_gather": 0}, (
                        label, alg, rank, rnd, calls)
                    assert entry["server"] is not None, entry
                else:
                    assert calls["all_reduce"] > 0 and calls["all_gather"] > 0, (
                        label, alg, rank, rnd, calls)


def test_a_participant_that_never_hands_in_is_named_and_the_round_is_undone():
    """Rank 2 trains and never calls synchronize: ranks 0 and 1 raise RankDropped naming it.

    Within the group's timeout (10 s), with no aggregate computed anywhere:
    the server's ledger holds no fold and one abandoned round naming rank 2,
    and both ranks' parameters are back at the round's base, byte for byte
    (on_missing='refuse'). Sabotage: let the server average whoever arrived
    and the ledger gains a fold over [0, 1] and the ranks install it.
    """
    reports = fs_missing()
    assert reports[2].get("absent") is True, reports[2]
    for rank in (0, 1):
        got = reports[rank]
        sync = got["sync"]
        assert "exc" in sync, ("rank %d aggregated without rank 2" % rank, sync)
        assert sync["type"] == "RankDropped", sync
        assert sync["missing"] == [2], sync
        assert "rank(s) 2" in sync["exc"] and "server" in sync["exc"], sync
        assert FS_TIMEOUT_S * 0.5 <= sync["seconds"] <= FS_TIMEOUT_S * 4, sync
        assert got["trained"] != got["base"], "the local training did not move the model"
        assert got["after"] == got["base"], (
            "rank %d kept its local update after the round was refused" % rank)
    ledger = reports[0]["ledger"]
    assert not [e for e in ledger if e["kind"] == "fold"], ledger
    abandoned = [e for e in ledger if e["kind"].endswith("-abandoned")]
    assert len(abandoned) == 1 and abandoned[0]["missing"] == [2], ledger
    assert abandoned[0]["ranks"] == [0, 1], abandoned


def test_server_aggregation_refuses_what_it_does_not_serve_by_name():
    """Every combination `aggregation='server'` does not build refuses, naming itself.

    Sabotage: drop `_refuse_for_server` and the subclass, custom algorithm,
    secure-aggregation and DP probes are accepted (the last two then refuse
    with the general message, which does not name server aggregation).
    """
    report = fs_inprocess()["reports"][0]
    assert report["default_aggregation"] == "all_reduce", report["default_aggregation"]
    msg = c2_exc(report, "bad_aggregation")
    assert msg.startswith("ValueError:") and "'ring'" in msg and "server" in msg, msg
    msg = c2_exc(report, "server_average_arrived")
    assert msg.startswith("NotImplementedError:"), msg
    assert "average_arrived" in msg and "aggregation='server'" in msg, msg
    for key, word in (("server_secure", "secure_aggregation"),
                      ("server_dp", "differential_privacy")):
        msg = c2_exc(report, key)
        assert msg.startswith("NotImplementedError:"), (key, msg)
        assert word in msg and "aggregation='server'" in msg, (key, msg)
    for key, name in (("server_subclass", "MedianOfMeans"), ("server_custom", "Custom")):
        msg = c2_exc(report, key)
        assert msg.startswith("NotImplementedError:"), (key, msg)
        assert name in msg and "aggregation='server'" in msg, (key, msg)
    for rank in range(FS_WORLD):
        msg = c2_exc(fs_inprocess()["reports"][rank], "server_float16")
        assert "float16" in msg and "aggregation='server'" in msg, msg


def test_server_aggregation_on_a_group_without_a_server_refuses_by_name():
    """A group from tcp:// has no server: refused naming tnserver://, not served by all_reduce.

    Sabotage: fall back to the all_reduce path when no server is found and
    `noserver` returns None instead of refusing.
    """
    reports = fs_noserver()
    for rank in range(2):
        msg = c2_exc(reports[rank], "noserver")
        assert msg.startswith("NotImplementedError:"), msg
        assert "tnserver://" in msg and "aggregation='server'" in msg, msg


def test_every_rank_really_ran_the_shim_and_the_standalone_server_was_another_process():
    world = fs_standalone()
    pids = {world["reports"][r]["_meta"]["pid"] for r in range(FS_WORLD)}
    assert world["server_pid"] is not None and world["server_pid"] not in pids, (
        world["server_pid"], pids)
    for rank in range(FS_WORLD):
        meta = fs_inprocess()["reports"][rank]["_meta"]
        assert (meta["shim"], meta["backend"]) == (True, "torchnative"), meta


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
