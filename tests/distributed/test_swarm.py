"""`torch.swarm.SwarmDataParallel`: gossip without a server, against oracles (issue #291).

SPEC S6.9, docs/distributed/SWARM.md. The round under test: each rank sends its
parameters to its neighbours only (point to point), and replaces them with
`sum_j W[i][j] x_j` over itself and its neighbours, W the Metropolis-Hastings
matrix of the topology (ring by default). Four oracles, each independent of
the code under test in a different way:

* **central**: the same round computed in this process in pure Python floats
  (IEEE double, the fold in ascending rank order, rounded once to float32 with
  `struct`). No torch, no torchnative. The module folds in a fixed order in
  float64, so the comparison is equality, with no tolerance.
* **upstream gloo**: the same round written by hand with `isend`/`irecv` on
  upstream torch and gloo, in separate processes (`test_collect2.py`'s
  harness), and the same by-hand round on the shim. Equality.
* **spectral**: repeated rounds without training approach the global mean, and
  the disagreement contracts by the mixing matrix's second-largest eigenvalue
  modulus, computed here by a Jacobi eigen solver. The bound's floating-point
  slack is derived from one float32 rounding per element per round, not chosen.
* **traffic**: counters wrapped around the backend's `send`/`recv` and around
  `ProcessGroupLocal._star_exchange` (the one road every collective of the
  backend takes) show zero collectives and messages to neighbours only.

`test_the_central_oracle_can_tell_a_dropped_neighbour_a_wrong_weight_and_an_all_reduce_apart`
is the nullification evidence this file can carry without mutating the module:
it shows the oracle the other tests compare against would disagree with each
of the three planned mutants on the very values observed.

Written before the implementation was run anywhere (AGENTS.md rule 10: agents
do not run suites); the first run is CI's.
"""

import functools
import math
import struct

import gloo_loopback

from test_collect2 import _c2_spawn, c2_exc, c2_ok


SW_WORLDS = (2, 4)
SW_ROUNDS = 8
#: Duplicated from torchnative/swarm/__init__.py on purpose: two copies that
#: must agree, so neither can be edited alone.
SW_CHECK_TAG = 0x5357_01
SW_PARAMS_TAG = 0x5357_02
SW_NUMEL = 15  # Linear(4, 3): 12 weights and 3 biases
SW_PATH4 = [[1], [0, 2], [1, 3], [2]]


# ---------------------------------------------------------------------------
# The probe, shared verbatim by the shim and the gloo children
# ---------------------------------------------------------------------------

_SW_PROBE_SRC = r'''
import math

SW_SEED = 20261113
SW_ROUNDS = 8
SW_PATH4 = [[1], [0, 2], [1, 3], [2]]
# Ring at world 4 with a different (valid) matrix: self weight 1/2, edges 1/4.
SW_Q4 = [[0.5, 0.25, 0.0, 0.25], [0.25, 0.5, 0.25, 0.0],
         [0.0, 0.25, 0.5, 0.25], [0.25, 0.0, 0.25, 0.5]]

def sw_ints(rank, n, tag):
    state = (SW_SEED + rank * 104729 + sum(ord(c) * 17 for c in tag)) % 2147483647
    out = []
    for _ in range(n):
        state = (state * 1103515245 + 12345) % 2147483648
        out.append((state >> 11) % 1000 - 500)
    return out

def sw_floats(rank, n, tag):
    # Multiples of 2**-6 in [-2, 2): exact in float32.
    return [(v % 256 - 128) / 64.0 for v in sw_ints(rank, n, tag)]

def sw_ring(world):
    if world == 2:
        return [[1], [0]]
    return [sorted({(r - 1) % world, (r + 1) % world}) for r in range(world)]

def sw_mh(neighbours):
    """Metropolis-Hastings, written here and not imported."""
    deg = [len(n) for n in neighbours]
    rows = []
    for i in range(len(neighbours)):
        row = [0.0] * len(neighbours)
        for j in neighbours[i]:
            row[j] = 1.0 / (1.0 + max(deg[i], deg[j]))
        row[i] = 1.0 - math.fsum(row[j] for j in neighbours[i])
        rows.append(row)
    return rows

def sw_by_hand(torch, dist, rank, flat, neighbours, row, tag):
    """One gossip round with nothing but torch.distributed point to point."""
    peers = neighbours[rank]
    sends = [dist.isend(flat, dst=p, tag=tag) for p in peers]
    bufs = {p: torch.zeros_like(flat) for p in peers}
    recvs = [dist.irecv(bufs[p], src=p, tag=tag) for p in peers]
    for work in recvs + sends:
        work.wait()
    acc = None
    for j in sorted([rank] + list(peers)):
        x = flat if j == rank else bufs[j]
        term = x.to(torch.float64) * row[j]
        acc = term if acc is None else acc + term
    return acc.to(flat.dtype)

def sw_rounds_by_hand(torch, dist, rank, world, neighbours, tag, seed_tag):
    W = sw_mh(neighbours)
    flat = torch.tensor(sw_floats(rank, 15, seed_tag), dtype=torch.float32)
    rounds = [flat.tolist()]
    for _ in range(SW_ROUNDS):
        flat = sw_by_hand(torch, dist, rank, flat, neighbours, W[rank], tag)
        rounds.append(flat.tolist())
    return rounds

def sw_probe_fn(out):
    def probe(key, fn):
        try:
            out[key] = {"ok": fn()}
        except Exception as exc:
            out[key] = {"exc": "%s: %s" % (type(exc).__name__, exc)}
    return probe
'''


# ---------------------------------------------------------------------------
# The shim child: SwarmDataParallel, with traffic counters
# ---------------------------------------------------------------------------

_SW_SHIM_RUN = r'''
from torchnative.distributed.backend import ProcessGroupTorchnative as _PGT

SW_COUNT = {"star": 0, "api": 0, "send": [], "recv": []}

_orig_star = _PGT._star_exchange
def _counted_star(self, *a, **k):
    SW_COUNT["star"] += 1
    return _orig_star(self, *a, **k)
_PGT._star_exchange = _counted_star

_orig_send = _PGT.send
def _counted_send(self, tensors, dst_rank, tag=0):
    SW_COUNT["send"].append([int(dst_rank), int(tag), int(tensors[0].numel())])
    return _orig_send(self, tensors, dst_rank, tag)
_PGT.send = _counted_send

_orig_recv = _PGT.recv
def _counted_recv(self, tensors, src_rank, tag=0):
    SW_COUNT["recv"].append([int(src_rank), int(tag), int(tensors[0].numel())])
    return _orig_recv(self, tensors, src_rank, tag)
_PGT.recv = _counted_recv

for _name in ("all_reduce", "all_gather", "all_gather_into_tensor", "broadcast",
              "reduce", "reduce_scatter", "reduce_scatter_tensor", "all_to_all",
              "all_to_all_single", "barrier", "monitored_barrier", "gather",
              "scatter", "all_gather_object", "broadcast_object_list",
              "gather_object", "scatter_object_list"):
    _fn = getattr(dist, _name, None)
    if _fn is None:
        continue
    def _wrap(fn=_fn):
        def inner(*a, **k):
            SW_COUNT["api"] += 1
            return fn(*a, **k)
        return inner
    setattr(dist, _name, _wrap())

def counted(fn):
    s0, a0 = SW_COUNT["star"], SW_COUNT["api"]
    n0, r0 = len(SW_COUNT["send"]), len(SW_COUNT["recv"])
    fn()
    return {"star": SW_COUNT["star"] - s0, "api": SW_COUNT["api"] - a0,
            "send": SW_COUNT["send"][n0:], "recv": SW_COUNT["recv"][r0:]}

def sw_model(torch, rank, tag):
    m = torch.nn.Linear(4, 3)
    vals = sw_floats(rank, 15, tag)
    with torch.no_grad():
        m.weight.copy_(torch.tensor(vals[:12], dtype=torch.float32).reshape(3, 4))
        m.bias.copy_(torch.tensor(vals[12:], dtype=torch.float32))
    return m

def sw_flat(m):
    return ([float(v) for v in m.weight.detach().reshape(-1).tolist()]
            + [float(v) for v in m.bias.detach().tolist()])

def jsonable(log):
    out = {}
    for k, v in log.items():
        if isinstance(v, dict):
            v = {str(a): b for a, b in v.items()}
        elif isinstance(v, tuple):
            v = list(v)
        out[k] = v
    return out

def sw_shim_run(torch, dist, rank, world):
    import torchnative.swarm as swarm
    from torchnative.swarm import algorithms
    out = {}
    probe = sw_probe_fn(out)

    def addhook():
        import torch.swarm
        import torch.swarm.algorithms
        return {"sdp": torch.swarm.SwarmDataParallel is swarm.SwarmDataParallel,
                "gossip": torch.swarm.algorithms.GossipAvg is algorithms.GossipAvg,
                "dropped": torch.swarm.NeighbourDropped is swarm.NeighbourDropped,
                "file": torch.swarm.__file__}
    probe("addhook", addhook)

    def round_ring():
        m = sw_model(torch, rank, "init")
        sdp = swarm.SwarmDataParallel(m)
        empty = sdp._get_swarm_logging_data()
        before = sw_flat(m)
        traffic = counted(sdp.synchronize)
        return {"before": before, "after": sw_flat(m), "traffic": traffic,
                "matrix": [list(r) for r in sdp.mixing_matrix()],
                "neighbours": list(sdp.neighbours()),
                "log": jsonable(sdp._get_swarm_logging_data()),
                "log_before": empty}
    probe("round_ring", round_ring)

    probe("by_hand_ring", lambda: sw_by_hand(
        torch, dist, rank, torch.tensor(sw_floats(rank, 15, "init"),
                                        dtype=torch.float32),
        sw_ring(world), sw_mh(sw_ring(world))[rank], 77).tolist())

    def trained():
        m = sw_model(torch, rank, "train")
        sdp = swarm.SwarmDataParallel(m)
        opt = torch.optim.SGD(m.parameters(), lr=0.125)
        x = torch.tensor(sw_floats(rank, 8, "x"), dtype=torch.float32).reshape(2, 4)
        y = torch.tensor(sw_floats(rank, 6, "y"), dtype=torch.float32).reshape(2, 3)
        opt.zero_grad()
        loss = ((sdp(x) - y) ** 2).mean()
        loss.backward()
        opt.step()
        before = sw_flat(m)
        sdp.synchronize()
        log = sdp._get_swarm_logging_data()
        return {"init": sw_floats(rank, 15, "train"), "before": before,
                "after": sw_flat(m), "local_norm": log["local_norm"],
                "mix_norm": log["mix_norm"]}
    probe("trained", trained)

    def rounds(topology, tag):
        m = sw_model(torch, rank, tag)
        sdp = swarm.SwarmDataParallel(m, topology=topology)
        seen = [sw_flat(m)]
        for _ in range(SW_ROUNDS):
            sdp.synchronize()
            seen.append(sw_flat(m))
        return seen
    probe("cons_ring", lambda: rounds("ring", "cons"))
    if world == 4:
        probe("cons_path", lambda: rounds(SW_PATH4, "path"))
        probe("by_hand_path", lambda: sw_rounds_by_hand(
            torch, dist, rank, world, SW_PATH4, 79, "path"))
    probe("by_hand_cons", lambda: sw_rounds_by_hand(
        torch, dist, rank, world, sw_ring(world), 78, "cons"))

    if world == 4:
        def disagree():
            alg = (algorithms.GossipAvg() if rank < 2
                   else algorithms.GossipAvg(mixing=SW_Q4))
            m = sw_model(torch, rank, "dis")
            sdp = swarm.SwarmDataParallel(m, algorithm=alg)
            before = sw_flat(m)
            try:
                sdp.synchronize()
            except Exception as exc:
                return {"exc": "%s: %s" % (type(exc).__name__, exc),
                        "unchanged": sw_flat(m) == before}
            return {"exc": None}
        probe("disagree", disagree)

        def no_traffic(fn):
            # A refusal that is raised before any message: nothing was sent.
            traffic = {}
            def run():
                try:
                    fn()
                except Exception as exc:
                    traffic["exc"] = "%s: %s" % (type(exc).__name__, exc)
            t = counted(run)
            traffic.update({"sends": len(t["send"]), "star": t["star"]})
            return traffic
        probe("refuse_weight", lambda: no_traffic(lambda: swarm.SwarmDataParallel(
            sw_model(torch, rank, "w")).synchronize(weight=10)))
        probe("refuse_disconnected", lambda: no_traffic(
            lambda: swarm.SwarmDataParallel(
                sw_model(torch, rank, "d"),
                topology=[[1], [0], [3], [2]]).synchronize()))

    if world == 2:
        # Last: the highest rank leaves, so nothing may follow it.
        def vanish():
            m = sw_model(torch, rank, "gone")
            sdp = swarm.SwarmDataParallel(m)
            if rank == world - 1:
                return "left"
            before = sw_flat(m)
            try:
                sdp.synchronize()
            except Exception as exc:
                return {"type": type(exc).__name__, "msg": str(exc),
                        "missing": list(getattr(exc, "missing", ())),
                        "unchanged": sw_flat(m) == before,
                        "log_empty": sdp._get_swarm_logging_data() == {}}
            return {"type": None}
        probe("vanish", vanish)
    return out
'''

_SW_SHIM_SRC = (
    "import datetime, json, os, sys, time\n"
    "import torch\n"
    'assert hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded upstream torch, not the shim")\n'
    "import torch.distributed as dist\n"
    "import torchnative.distributed  # noqa: F401 -- registers backend='torchnative'\n"
    + _SW_PROBE_SRC
    + _SW_SHIM_RUN
    + r'''
rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
dist.init_process_group(backend="torchnative",
                        init_method="tcp://127.0.0.1:%d" % port,
                        rank=rank, world_size=world,
                        timeout=datetime.timedelta(seconds=120))
out = sw_shim_run(torch, dist, rank, world)
out["_meta"] = {"shim": True, "rank": dist.get_rank(),
                "world": dist.get_world_size(), "backend": dist.get_backend()}
with open(dest, "w") as handle:
    json.dump(out, handle)
if out.get("vanish", {}).get("ok") == "left":
    os._exit(0)
'''
)


_SW_GLOO_SRC = (
    gloo_loopback.child_pin_source()
    + "import json, os, sys\n"
    "import torch\n"
    'assert not hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded the shim, not upstream torch")\n'
    "import torch.distributed as dist\n"
    + _SW_PROBE_SRC
    + r'''
rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(port)
dist.init_process_group(backend="gloo", rank=rank, world_size=world)
out = {}
probe = sw_probe_fn(out)
probe("by_hand_ring", lambda: sw_by_hand(
    torch, dist, rank, torch.tensor(sw_floats(rank, 15, "init"),
                                    dtype=torch.float32),
    sw_ring(world), sw_mh(sw_ring(world))[rank], 77).tolist())
probe("by_hand_cons", lambda: sw_rounds_by_hand(
    torch, dist, rank, world, sw_ring(world), 78, "cons"))
if world == 4:
    probe("by_hand_path", lambda: sw_rounds_by_hand(
        torch, dist, rank, world, SW_PATH4, 79, "path"))
out["_meta"] = {"shim": False, "rank": dist.get_rank(),
                "world": dist.get_world_size(), "backend": "gloo"}
with open(dest, "w") as handle:
    json.dump(out, handle)
dist.destroy_process_group()
'''
)


# ---------------------------------------------------------------------------
# The solo shim child: refusals that need no peer, and a world of one
# ---------------------------------------------------------------------------

_SW_SOLO_SRC = (
    "import datetime, json, sys\n"
    "import torch\n"
    'assert hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded upstream torch, not the shim")\n'
    "import torch.distributed as dist\n"
    "import torchnative.distributed  # noqa: F401\n"
    + _SW_PROBE_SRC
    + r'''
import torchnative.swarm as swarm
from torchnative.swarm import algorithms

rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
out = {}
probe = sw_probe_fn(out)

def lin():
    return torch.nn.Linear(4, 3)

class Mixed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Parameter(torch.zeros(2))
        self.b = torch.nn.Parameter(torch.zeros(2, dtype=torch.float64))

probe("before_init", lambda: swarm.SwarmDataParallel(lin()).synchronize())
probe("secure", lambda: repr(swarm.SwarmDataParallel(lin(), secure_aggregation=True)))
probe("dp", lambda: repr(swarm.SwarmDataParallel(lin(), differential_privacy=1.0)))
probe("no_params", lambda: repr(swarm.SwarmDataParallel(lin(), parameters=[])))
probe("unknown_param", lambda: repr(swarm.SwarmDataParallel(lin(), parameters=["nope"])))
probe("mixed_dtype", lambda: repr(swarm.SwarmDataParallel(Mixed())))

def topo(t, world=4):
    return [list(n) for n in swarm.resolve_topology(t, world)]
probe("ring4", lambda: topo("ring"))
probe("ring2", lambda: topo("ring", 2))
probe("complete3", lambda: topo("complete", 3))
probe("path4", lambda: topo(SW_PATH4))
probe("path4_mapping", lambda: topo({0: [1], 1: [0, 2], 2: [1, 3], 3: [2]}))
probe("t_unknown", lambda: topo("torus"))
probe("t_count", lambda: topo([[1], [0]]))
probe("t_self", lambda: topo([[0, 1], [0, 2], [1, 3], [2]]))
probe("t_range", lambda: topo([[1, 9], [0, 2], [1, 3], [2]]))
probe("t_dup", lambda: topo([[1, 1], [0, 2], [1, 3], [2]]))
probe("t_directed", lambda: topo([[1], [2], [3], [0]]))
probe("t_disconnected", lambda: topo([[1], [0], [3], [2]]))

RING4 = swarm.resolve_topology("ring", 4)
def mix(m):
    return [list(r) for r in algorithms.GossipAvg(mixing=m).mixing_matrix(RING4)]
probe("m_ok", lambda: mix(SW_Q4))
# Row 0 still sums to one; W[0][1] = 0.3 against W[1][0] = 0.25.
probe("m_asym", lambda: mix([[0.5, 0.3, 0.0, 0.2]] + [list(r) for r in SW_Q4[1:]]))
probe("m_rowsum", lambda: mix([[0.6 if i == j else v for j, v in enumerate(r)]
                               for i, r in enumerate(SW_Q4)]))
probe("m_support", lambda: mix([[0.4, 0.25, 0.1, 0.25], [0.25, 0.5, 0.25, 0.0],
                                [0.1, 0.25, 0.4, 0.25], [0.25, 0.0, 0.25, 0.5]]))
probe("m_zero_diag", lambda: mix([[0.0, 0.5, 0.0, 0.5], [0.5, 0.0, 0.5, 0.0],
                                  [0.0, 0.5, 0.0, 0.5], [0.5, 0.0, 0.5, 0.0]]))
probe("m_negative", lambda: mix([[1.5, -0.25, 0.0, -0.25], [-0.25, 1.5, -0.25, 0.0],
                                 [0.0, -0.25, 1.5, -0.25], [-0.25, 0.0, -0.25, 1.5]]))
probe("m_edge_zero", lambda: mix([[0.75, 0.0, 0.0, 0.25], [0.0, 0.75, 0.25, 0.0],
                                  [0.0, 0.25, 0.5, 0.25], [0.25, 0.0, 0.25, 0.5]]))
probe("m_size", lambda: mix([[0.5, 0.5], [0.5, 0.5]]))
probe("m_unknown_rule", lambda: repr(algorithms.GossipAvg(mixing="sinkhorn")))
probe("weight", lambda: algorithms.GossipAvg().resolve_weight(3))
probe("no_weight", lambda: algorithms.GossipAvg().resolve_weight(None))
probe("mh_path4", lambda: [list(r) for r in algorithms.metropolis_hastings(
    swarm.resolve_topology(SW_PATH4, 4))])

dist.init_process_group(backend="torchnative",
                        init_method="tcp://127.0.0.1:%d" % port,
                        rank=rank, world_size=world,
                        timeout=datetime.timedelta(seconds=60))
probe("one_sync", lambda: swarm.SwarmDataParallel(lin()).synchronize())
probe("one_neighbours", lambda: swarm.SwarmDataParallel(lin()).neighbours())
probe("one_matrix", lambda: swarm.SwarmDataParallel(lin()).mixing_matrix())
out["_meta"] = {"shim": True, "world": dist.get_world_size(),
                "backend": dist.get_backend()}
with open(dest, "w") as handle:
    json.dump(out, handle)
'''
)


@functools.cache
def sw_shim(world):
    return _c2_spawn(_SW_SHIM_SRC, world, "float32", True, "swarm-shim%d" % world)


@functools.cache
def sw_gloo(world):
    return _c2_spawn(_SW_GLOO_SRC, world, "float32", False, "swarm-gloo%d" % world)


@functools.cache
def sw_solo():
    return _c2_spawn(_SW_SOLO_SRC, 1, "float32", True, "swarm-solo")[0]


@functools.cache
def _helpers():
    """The probe's pure-Python generators and topology, for oracles here."""
    namespace = {}
    exec(compile(_SW_PROBE_SRC, "<swarm probe>", "exec"), namespace)
    return namespace


# ---------------------------------------------------------------------------
# Oracles, in this process, without torch
# ---------------------------------------------------------------------------

def _f32(value):
    """``value`` rounded to the nearest float32, by `struct` alone."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def _central_round(xs, neighbours, matrix, drop_last=False):
    """One gossip round over every rank: the module's fold, in Python floats.

    ``drop_last`` is the first mutant: each rank ignores its highest
    neighbour's message (the term is left out, nothing else changes).
    """
    out = []
    for i in range(len(xs)):
        terms = sorted([i] + list(neighbours[i]))
        if drop_last and neighbours[i]:
            terms.remove(max(neighbours[i]))
        row = []
        for e in range(len(xs[i])):
            acc = None
            for j in terms:
                term = xs[j][e] * matrix[i][j]
                acc = term if acc is None else acc + term
            row.append(_f32(acc))
        out.append(row)
    return out


def _wrong_weights(neighbours):
    """The second mutant: edge weight 1/(1 + d_i) (one end's degree), not MH."""
    world = len(neighbours)
    rows = []
    for i in range(world):
        row = [0.0] * world
        for j in neighbours[i]:
            row[j] = 1.0 / (1.0 + len(neighbours[i]))
        row[i] = 1.0 - math.fsum(row[j] for j in neighbours[i])
        rows.append(row)
    return rows


def _all_reduce_mean(xs):
    """The third mutant: a global average, as an all_reduce would give."""
    n = len(xs)
    return [[_f32(math.fsum(xs[r][e] for r in range(n)) / n)
             for e in range(len(xs[0]))] for _ in range(n)]


def _jacobi_eigenvalues(matrix, sweeps=100):
    """Eigenvalues of a small symmetric matrix by cyclic Jacobi rotations."""
    a = [list(map(float, row)) for row in matrix]
    n = len(a)
    for _ in range(sweeps):
        off = math.fsum(a[i][j] ** 2 for i in range(n) for j in range(n) if i != j)
        if off < 1e-30:
            break
        for p in range(n):
            for q in range(p + 1, n):
                if abs(a[p][q]) < 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q])
                t = (1.0 if theta >= 0 else -1.0) / (abs(theta) + math.sqrt(theta * theta + 1.0))
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
    return sorted(a[i][i] for i in range(n))


def _second_modulus(matrix):
    """``max |lambda|`` over every eigenvalue but the one at 1 (the consensus)."""
    eig = _jacobi_eigenvalues(matrix)
    one = min(range(len(eig)), key=lambda k: abs(eig[k] - 1.0))
    assert abs(eig[one] - 1.0) < 1e-12, (
        "the mixing matrix has no eigenvalue at 1, so it is not stochastic: %r"
        % (eig,))
    rest = [abs(v) for k, v in enumerate(eig) if k != one]
    return max(rest), rest


def _disagreement(xs):
    """Frobenius distance of the ranks' vectors from their mean, and the mean."""
    n = len(xs)
    mean = [math.fsum(xs[r][e] for r in range(n)) / n for e in range(len(xs[0]))]
    d = math.sqrt(math.fsum((xs[r][e] - mean[e]) ** 2
                            for r in range(n) for e in range(len(mean))))
    return d, mean


def _frobenius(xs):
    return math.sqrt(math.fsum(v * v for row in xs for v in row))


def _ranks(report, key):
    return [c2_ok(report[r], key) for r in range(len(report))]


# ---------------------------------------------------------------------------
# One round
# ---------------------------------------------------------------------------

def test_one_gossip_round_equals_the_same_round_computed_centrally_bit_for_bit():
    """The ring round at world 2 and 4 equals the pure-Python fold, element for element.

    The inputs are each rank's own parameters, reported by that rank before
    the round, so the oracle is computed on exactly what the ranks held. The
    mixing matrix comes from this file's `sw_mh`, not from the module.
    """
    h = _helpers()
    for world in SW_WORLDS:
        got = _ranks(sw_shim(world), "round_ring")
        xs = [g["before"] for g in got]
        for r in range(world):
            assert xs[r] == h["sw_floats"](r, SW_NUMEL, "init"), (world, r)
        neighbours = h["sw_ring"](world)
        want = _central_round(xs, neighbours, h["sw_mh"](neighbours))
        for r in range(world):
            assert got[r]["after"] == want[r], (
                "world %d rank %d: SwarmDataParallel's round differs from the "
                "central fold.\n  ours:    %r\n  central: %r"
                % (world, r, got[r]["after"], want[r]))
            assert got[r]["after"] != got[r]["before"], (world, r)


def test_at_world_two_one_round_is_the_exact_mean_on_both_ranks():
    """Ring at two ranks is one edge with weight 1/2: one round is the mean, bit-identical."""
    got = _ranks(sw_shim(2), "round_ring")
    a, b = got[0]["before"], got[1]["before"]
    want = [_f32(x * 0.5 + y * 0.5) for x, y in zip(a, b)]
    assert got[0]["after"] == got[1]["after"] == want, got


def test_a_round_after_real_local_training_equals_the_central_mix_of_what_each_rank_trained():
    """The user loop: forward, backward, SGD step, `synchronize()`. The mix is of the trained weights.

    `local_norm` is checked against the distance this file computes between
    the reported initial and trained weights. Its bound is derived: the
    float64 sum of 15 squares carries at most 16 roundings of 2**-53 each.
    """
    h = _helpers()
    for world in SW_WORLDS:
        got = _ranks(sw_shim(world), "trained")
        xs = [g["before"] for g in got]
        neighbours = h["sw_ring"](world)
        want = _central_round(xs, neighbours, h["sw_mh"](neighbours))
        for r in range(world):
            assert xs[r] != got[r]["init"], (
                "world %d rank %d: the SGD step did not move the parameters, "
                "so this test would not be testing a trained round" % (world, r))
            assert got[r]["after"] == want[r], (world, r, got[r]["after"], want[r])
            moved = math.sqrt(math.fsum((x - y) ** 2 for x, y in
                                        zip(xs[r], got[r]["init"])))
            bound = moved * (SW_NUMEL + 1) * 2.0 ** -53
            assert abs(got[r]["local_norm"] - moved) <= bound, (
                world, r, got[r]["local_norm"], moved)
            assert got[r]["mix_norm"] > 0.0, (world, r)


def test_the_mixing_matrix_is_metropolis_hastings_computed_independently():
    h = _helpers()
    for world in SW_WORLDS:
        for r, got in enumerate(_ranks(sw_shim(world), "round_ring")):
            neighbours = h["sw_ring"](world)
            assert got["matrix"] == h["sw_mh"](neighbours), (world, r, got["matrix"])
            assert got["neighbours"] == neighbours[r], (world, r)
    solo = sw_solo()
    assert c2_ok(solo, "mh_path4") == h["sw_mh"](SW_PATH4), c2_ok(solo, "mh_path4")


def test_the_logging_data_reports_the_round():
    h = _helpers()
    for world in SW_WORLDS:
        neighbours = h["sw_ring"](world)
        matrix = h["sw_mh"](neighbours)
        for r, got in enumerate(_ranks(sw_shim(world), "round_ring")):
            log = got["log"]
            assert got["log_before"] == {}, got["log_before"]
            assert (log["rank"], log["world"], log["round"]) == (r, world, 0), log
            assert log["neighbours"] == log["sent_to"] == log["received_from"] \
                == neighbours[r], log
            assert log["mixing_row"] == {
                str(j): matrix[r][j] for j in sorted([r] + neighbours[r])}, log
            assert log["local_norm"] == 0.0, log   # no training before this round
            assert log["mix_norm"] > 0.0, log


# ---------------------------------------------------------------------------
# Repeated rounds approach the global mean, at the spectral rate
# ---------------------------------------------------------------------------

def _check_contraction(world, rounds, matrix, label, exact_rate):
    """`d_{k+1} <= lam d_k + eps_k`, the mean held within the same slack.

    ``eps_k`` is derived: each round rounds every element once to float32
    (relative error <= 2**-24) after a float64 fold of at most ``deg + 1``
    terms (each <= 2**-53 relative), so the round adds at most
    ``(2**-24 + (deg + 1) * 2**-53) * ||X_{k+1}||_F`` to the disagreement;
    the deviation from the mean is a projection, which does not enlarge it.
    """
    lam, rest = _second_modulus(matrix)
    deg = max(sum(1 for j in range(world) if j != i and matrix[i][j] != 0.0)
              for i in range(world))
    unit = 2.0 ** -24 + (deg + 1) * 2.0 ** -53
    xs = [[rounds[r][k] for r in range(world)] for k in range(SW_ROUNDS + 1)]
    d0, mean0 = _disagreement(xs[0])
    assert d0 > 0.0, (label, "the ranks started equal; nothing to contract")
    bound = d0
    ratios = []
    drift = 0.0
    for k in range(SW_ROUNDS):
        dk, _ = _disagreement(xs[k])
        dn, mean = _disagreement(xs[k + 1])
        eps = unit * _frobenius(xs[k + 1])
        bound = lam * bound + eps
        ratios.append(dn / dk if dk else 0.0)
        assert dn <= bound, (
            "%s: after round %d the disagreement is %r, above the spectral "
            "bound %r (lambda_2 = %r). Measured ratios: %r"
            % (label, k + 1, dn, bound, lam, ratios))
        if exact_rate:
            # Every non-trivial eigenvalue has the same modulus, so W - J is
            # lam times an orthogonal map on the deviation: the ratio is lam.
            assert dn >= lam * dk - eps, (label, k + 1, dn, lam * dk, ratios)
        drift += eps
        for e, (a, b) in enumerate(zip(mean, mean0)):
            assert abs(a - b) <= drift, (
                "%s: round %d moved the global mean of element %d from %r to "
                "%r; a doubly stochastic mix preserves it (slack %r)"
                % (label, k + 1, e, b, a, drift))
    final, _ = _disagreement(xs[SW_ROUNDS])
    return lam, rest, ratios, d0, final


def test_repeated_rounds_approach_the_global_mean_at_the_rate_of_the_second_eigenvalue():
    """Ring at 2 (lambda 0) and 4 (lambda 1/3), path at 4 (lambda (1 + sqrt 2)/3).

    The expected moduli are written here as numbers and compared with the
    Jacobi solver's answer, so a solver that went wrong would be caught too.
    """
    h = _helpers()
    expected = {("ring", 2): 0.0, ("ring", 4): 1.0 / 3.0,
                ("path", 4): (1.0 + math.sqrt(2.0)) / 3.0}
    cases = [(2, "cons_ring", h["sw_ring"](2), ("ring", 2)),
             (4, "cons_ring", h["sw_ring"](4), ("ring", 4)),
             (4, "cons_path", SW_PATH4, ("path", 4))]
    for world, key, neighbours, name in cases:
        matrix = h["sw_mh"](neighbours)
        rounds = _ranks(sw_shim(world), key)
        lam, rest, ratios, d0, final = _check_contraction(
            world, rounds, matrix, "%s-%d" % name,
            exact_rate=max(rest) - min(rest) < 1e-12 and max(rest) > 0.0)
        assert abs(lam - expected[name]) < 1e-12, (name, lam, rest)
        assert final < 0.25 * d0, (name, d0, final, ratios)


# ---------------------------------------------------------------------------
# Peer to peer, not a collective
# ---------------------------------------------------------------------------

def test_it_was_peer_to_peer_no_collective_and_messages_only_to_neighbours():
    """Per round: zero collectives on the backend, two messages to each neighbour, none elsewhere.

    `star` counts `ProcessGroupLocal._star_exchange`, which every collective of
    the backend (all_reduce, all_gather, broadcast, barrier, ...) runs through;
    `api` counts calls of the collective functions of `torch.distributed`.
    """
    h = _helpers()
    for world in SW_WORLDS:
        neighbours = h["sw_ring"](world)
        for r, got in enumerate(_ranks(sw_shim(world), "round_ring")):
            t = got["traffic"]
            assert t["star"] == 0 and t["api"] == 0, (world, r, t)
            want = sorted([p, SW_CHECK_TAG, 3] for p in neighbours[r]) + sorted(
                [p, SW_PARAMS_TAG, SW_NUMEL] for p in neighbours[r])
            assert sorted(t["send"]) == sorted(want), (world, r, t["send"])
            assert sorted(t["recv"]) == sorted(want), (world, r, t["recv"])
            assert len(t["send"]) > 0, (world, r)
            others = set(range(world)) - set(neighbours[r]) - {r}
            assert not {s[0] for s in t["send"]} & others, (world, r, t)


# ---------------------------------------------------------------------------
# Against upstream torch and gloo
# ---------------------------------------------------------------------------

def test_swarm_agrees_with_upstream_gloo_running_the_same_gossip_by_hand():
    """SwarmDataParallel on the shim == the by-hand round on gloo == the by-hand round on the shim."""
    for world in SW_WORLDS:
        shim, gloo = sw_shim(world), sw_gloo(world)
        for r in range(world):
            ours = c2_ok(shim[r], "round_ring")["after"]
            assert ours == c2_ok(gloo[r], "by_hand_ring"), (world, r)
            assert ours == c2_ok(shim[r], "by_hand_ring"), (world, r)
            cons = c2_ok(shim[r], "cons_ring")
            assert cons == c2_ok(gloo[r], "by_hand_cons"), (world, r)
            assert cons == c2_ok(shim[r], "by_hand_cons"), (world, r)
            if world == 4:
                path = c2_ok(shim[r], "cons_path")
                assert path == c2_ok(gloo[r], "by_hand_path"), (world, r)
                assert path == c2_ok(shim[r], "by_hand_path"), (world, r)


def test_every_rank_really_ran_the_shim_and_the_oracle_really_ran_upstream():
    for world in SW_WORLDS:
        for r in range(world):
            meta = sw_shim(world)[r]["_meta"]
            assert (meta["shim"], meta["rank"], meta["world"], meta["backend"]) == (
                True, r, world, "torchnative"), meta
            meta = sw_gloo(world)[r]["_meta"]
            assert (meta["shim"], meta["rank"], meta["world"], meta["backend"]) == (
                False, r, world, "gloo"), meta


# ---------------------------------------------------------------------------
# The oracle can fail: the three planned mutants, predicted
# ---------------------------------------------------------------------------

def test_the_central_oracle_can_tell_a_dropped_neighbour_a_wrong_weight_and_an_all_reduce_apart():
    """AGENTS.md section 17.5, for the oracle itself.

    The module is not mutated here (agents do not run suites; the mutation
    runs are listed in SWARM.md section 6). What this shows is that each
    mutant, applied to the very inputs the ranks reported, predicts values
    the observed round does not have. So the equality tests above would go
    red on each: dropping a neighbour's message, a wrong edge weight, and a
    host-side all_reduce in place of gossip.
    """
    h = _helpers()
    world = 4
    got = _ranks(sw_shim(world), "round_ring")
    xs = [g["before"] for g in got]
    observed = [g["after"] for g in got]
    ring = h["sw_ring"](world)
    assert _central_round(xs, ring, h["sw_mh"](ring), drop_last=True) != observed
    assert _all_reduce_mean(xs) != observed
    # The ring is regular, so a one-ended degree gives the same weights there:
    # the wrong weight is tested where degrees differ, on the path.
    path = _ranks(sw_shim(world), "cons_path")
    p0 = [path[r][0] for r in range(world)]
    p1 = [path[r][1] for r in range(world)]
    assert _wrong_weights(SW_PATH4) != h["sw_mh"](SW_PATH4)
    assert _central_round(p0, SW_PATH4, _wrong_weights(SW_PATH4)) != p1
    assert _central_round(p0, SW_PATH4, h["sw_mh"](SW_PATH4)) == p1


# ---------------------------------------------------------------------------
# Refusals
# ---------------------------------------------------------------------------

def test_a_world_of_one_and_a_missing_process_group_refuse_by_name():
    solo = sw_solo()
    msg = c2_exc(solo, "before_init")
    assert msg.startswith("RuntimeError") and "no process group" in msg, msg
    for key in ("one_sync", "one_neighbours", "one_matrix"):
        msg = c2_exc(solo, key)
        assert msg.startswith("NotImplementedError"), (key, msg)
        assert "a world of one" in msg and "identity" in msg, (key, msg)
    assert solo["_meta"]["world"] == 1, solo["_meta"]


def test_topologies_that_cannot_converge_or_cannot_be_exchanged_refuse_by_name():
    solo = sw_solo()
    assert c2_ok(solo, "ring4") == [[1, 3], [0, 2], [1, 3], [0, 2]]
    assert c2_ok(solo, "ring2") == [[1], [0]]
    assert c2_ok(solo, "complete3") == [[1, 2], [0, 2], [0, 1]]
    assert c2_ok(solo, "path4") == c2_ok(solo, "path4_mapping") == SW_PATH4
    expect = {
        "t_unknown": ("ValueError", "not a known topology"),
        "t_count": ("ValueError", "2 neighbour lists for a world of 4"),
        "t_self": ("ValueError", "lists itself"),
        "t_range": ("ValueError", "not in a world of 4"),
        "t_dup": ("ValueError", "twice"),
        "t_directed": ("ValueError", "push-sum"),
        "t_disconnected": ("ValueError", "disconnected"),
    }
    for key, (kind, phrase) in expect.items():
        msg = c2_exc(solo, key)
        assert msg.startswith(kind) and phrase in msg, (key, msg)
    for r in range(4):
        got = c2_ok(sw_shim(4)[r], "refuse_disconnected")
        assert "disconnected" in got["exc"] and "[[0, 1], [2, 3]]" in got["exc"], got
        assert got["sends"] == 0 and got["star"] == 0, got


def test_mixing_matrices_that_are_not_doubly_stochastic_refuse_by_name():
    solo = sw_solo()
    h = _helpers()
    assert c2_ok(solo, "m_ok") == h["SW_Q4"]
    expect = {
        "m_asym": "symmetric",
        "m_rowsum": "sums to",
        "m_support": "not neighbours",
        "m_zero_diag": "zero self weight",
        "m_negative": "non-negative",
        "m_edge_zero": "are neighbours",
        "m_size": "one row and one column per rank",
        "m_unknown_rule": "not a known rule",
    }
    for key, phrase in expect.items():
        msg = c2_exc(solo, key)
        assert msg.startswith("ValueError") and phrase in msg, (key, msg)


def test_weight_secure_aggregation_privacy_and_bad_parameters_refuse_by_name():
    solo = sw_solo()
    msg = c2_exc(solo, "weight")
    assert msg.startswith("TypeError") and "push-sum" in msg, msg
    assert c2_ok(solo, "no_weight") == 1.0
    for key, kind, phrase in (
            ("secure", "NotImplementedError", "secure_aggregation"),
            ("dp", "NotImplementedError", "differential_privacy"),
            ("no_params", "ValueError", "no parameters to mix"),
            ("unknown_param", "ValueError", "does not have"),
            ("mixed_dtype", "NotImplementedError", "2 dtypes")):
        msg = c2_exc(solo, key)
        assert msg.startswith(kind) and phrase in msg, (key, msg)
    for r in range(4):
        got = c2_ok(sw_shim(4)[r], "refuse_weight")
        assert got["exc"].startswith("TypeError") and "push-sum" in got["exc"], got
        assert got["sends"] == 0 and got["star"] == 0, got


def test_neighbours_that_disagree_about_the_topology_refuse_before_mixing():
    """Ranks 0 and 1 use Metropolis-Hastings, 2 and 3 a different valid matrix on the same ring.

    Every rank has one neighbour of the other kind, so every rank refuses,
    after the agreement check and before any parameter is sent.
    """
    for r in range(4):
        got = c2_ok(sw_shim(4)[r], "disagree")
        assert got["exc"] is not None, (r, "a disagreement was mixed")
        assert got["exc"].startswith("ValueError"), got
        assert "the topology and mixing matrix" in got["exc"], got
        assert got["unchanged"] is True, got


def test_a_neighbour_that_vanishes_is_named_and_the_round_is_not_applied():
    got = _ranks(sw_shim(2)[:1], "vanish")[0]
    assert c2_ok(sw_shim(2)[1], "vanish") == "left"
    assert got["type"] == "NeighbourDropped", got
    assert got["missing"] == [1], got
    assert "neighbour 1" in got["msg"], got
    assert got["unchanged"] is True and got["log_empty"] is True, got


def test_torch_swarm_is_the_add_hook_for_torchnative_swarm():
    for world in SW_WORLDS:
        for r in range(world):
            got = c2_ok(sw_shim(world)[r], "addhook")
            assert got["sdp"] and got["gossip"] and got["dropped"], got
            assert got["file"].replace("\\", "/").endswith(
                "torchnative/python/torch/swarm/__init__.py"), got


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
