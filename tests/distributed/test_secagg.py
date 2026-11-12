"""Secure aggregation by pairwise masking, `torch.federated.algorithms.SecAgg` (issue #253).

Provenance: the algorithm is the leader's (2026-11-13: "secure aggregation via
pairwise masking (Bonawitz-style) is approved as the algorithm"); everything
else (the wrapper, the scale, the PRG, the wire shape) is an agent proposal,
not confirmed. docs/distributed/SECAGG.md.

Every test runs on real processes: world 3, three `subprocess.Popen`s of the
shim that share nothing but sockets, through `test_shim._fed4_spawn` (imported,
not copied). The children instrument themselves before `init_process_group`:

* `ProcessGroupLocal._recv_json` on rank 0 records every frame a leaf hands
  the hub. That is literally what the honest-but-curious aggregator reads, so
  the privacy test asserts on the hub's view rather than on what the code
  meant to send.
* `ProcessGroupTorchnative.send` / `recv` record every point-to-point frame,
  so the seed test can say which link each seed crossed.
* A `SecAgg` subclass records each rank's plain quantised residues and its
  masked contribution (never a seed).

The reference side is exact: `fractions.Fraction` over the float32 deltas the
ranks dumped, so the tolerance below contains only the implementation's own
rounding terms, each derived in `_bound_terms`. Nothing is chosen by eye.

Written before the implementation was run anywhere (AGENTS.md rule 10: agents
do not run suites); the first run is CI's. Graded *builds* until then.
"""

import functools
import json
import os
import struct
from fractions import Fraction

import _skip

from test_shim import (_ADAPT_MODEL_SRC, _CKPT_VENDOR_SHIM, _FED_LOCAL_SRC,
                       _fed4_spawn)

WORLD = 3
WEIGHTS = {0: 3, 1: 7, 2: 2}
NORMS = ["norm1.weight", "norm2.weight"]
F = 32                                   # SecAgg's default fraction_bits
K = 63 - WORLD.bit_length()              # 61: the int64 accumulator's modulus
M = 1 << K
B = ((1 << (K - 1)) - 1) // WORLD        # per-rank |q| bound
SEED_TAG = 0x5EC0A6


def _shim_present():
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        _skip.skip("no vendored shim at %s (scripts/vendor/install_shim.sh)"
                   % (_CKPT_VENDOR_SHIM,))
        return False
    return True


# ---------------------------------------------------------------------------
# the children
# ---------------------------------------------------------------------------

_HEAD = (
    "import json, os, sys\n"
    "import torch\n"
    "import torch.nn as nn\n"
    + _ADAPT_MODEL_SRC
    + _FED_LOCAL_SRC
    + r'''
out = {"shim": hasattr(torch._C, "_aten_implemented")}
print("shim" if out["shim"] else "upstream", file=sys.stderr, flush=True)
assert out["shim"], "this subprocess loaded upstream torch, not the shim"

import torch.distributed as dist
from torch._C._distributed_c10d import ProcessGroupLocal
import torchnative.distributed  # noqa: F401 -- registers backend="torchnative"
from torchnative.distributed.backend import ProcessGroupTorchnative
from torchnative import federated
from torchnative.federated import algorithms

rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
WEIGHT = {0: 3.0, 1: 7.0, 2: 2.0}[rank]
LOCAL_IDS = {0: [[3, 7, 1, 19, 5]],
             1: [[11, 2, 23, 0, 14]],
             2: [[6, 9, 13, 4, 21]]}[rank]
ids = torch.tensor(LOCAL_IDS)

# --- instrumentation, installed before the group exists -------------------
PHASE = ["setup"]
HUB = []   # rank 0 only: [phase, leaf rank, payload] for every frame a leaf sent the hub
P2P = []   # [phase, "send" | "recv", peer, tag, bytes sent or None]

_recv_json = ProcessGroupLocal._recv_json
def _hub_recv_json(self, sock):
    payload = _recv_json(self, sock)
    if self._rank == 0:
        peer = next((r for r, s in self._peers.items() if s is sock), None)
        HUB.append([PHASE[0], peer, payload])
    return payload
ProcessGroupLocal._recv_json = _hub_recv_json

_send = ProcessGroupTorchnative.send
def _logged_send(self, tensors, dst_rank, tag=0):
    P2P.append([PHASE[0], "send", dst_rank, tag,
                [int(v) for v in tensors[0].reshape(-1).tolist()]])
    return _send(self, tensors, dst_rank, tag)
ProcessGroupTorchnative.send = _logged_send

_recv = ProcessGroupTorchnative.recv
def _logged_recv(self, tensors, src_rank, tag=0):
    P2P.append([PHASE[0], "recv", src_rank, tag, None])
    return _recv(self, tensors, src_rank, tag)
ProcessGroupTorchnative.recv = _logged_recv

dist.init_process_group(backend="torchnative",
                        init_method="tcp://127.0.0.1:%d" % port,
                        rank=rank, world_size=world)
out["rank"], out["world"] = dist.get_rank(), dist.get_world_size()
out["backend"] = dist.get_backend()


class Recording(algorithms.SecAgg):
    """SecAgg that keeps what each rank handed the wire, and its plain twin."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.log = []

    def aggregate(self, table, weight=None, group=None):
        self.cur = {"table": {n: t.tolist() for n, t in table.items()},
                    "plain": {}, "masked": {}, "order": []}
        result = super().aggregate(table, weight=weight, group=group)
        self.cur["aggregate"] = {n: t.tolist() for n, t in result.items()}
        self.log.append(self.cur)
        return result

    def _exchange_seeds(self, group, world, rank):
        prev = PHASE[0]
        PHASE[0] = prev + ":seeds"
        try:
            return super()._exchange_seeds(group, world, rank)
        finally:
            PHASE[0] = prev

    def _contribute(self, label, plain, seeds, rank, round_index, k):
        sent = super()._contribute(label, plain, seeds, rank, round_index, k)
        key = label.decode("utf-8")
        self.cur["plain"][key] = [q % (1 << k) for q in plain]
        self.cur["masked"][key] = list(sent)
        self.cur["order"].append(key)
        self.cur["k"] = k
        return sent

    def _masked_sum(self, masked, labels, group, world):
        prev = PHASE[0]
        PHASE[0] = prev + ":masked"
        try:
            sums = super()._masked_sum(masked, labels, group, world)
        finally:
            PHASE[0] = prev
        self.cur["sums"] = {l.decode("utf-8"): v for l, v in sums.items()}
        return sums


class Unmasked(Recording):
    """The sabotage: every net mask is zero, so the hub reads plain residues."""

    def _net_mask(self, label, n, seeds, rank, round_index, k):
        return [0] * n


def params(model):
    named = dict(model.named_parameters())
    return {n: named[n].tolist() for n in NORMS}


def held(algo):
    """The algorithm's own state, minus what the test subclass recorded."""
    return repr({k: v for k, v in vars(algo).items() if k not in ("log", "cur")})


def run_round(phase, algo, select=None):
    PHASE[0] = phase
    model = build()
    fdp = federated.FederatedDataParallel(model, algorithm=algo,
                                          parameters=NORMS, select=select)
    base = params(model)
    if select is None or rank in select(world):
        local_train(fdp, sgd_for(model), ids, 3)
    fdp.synchronize(weight=WEIGHT)
    rep = fdp._get_federated_logging_data()
    PHASE[0] = "idle"
    got = {"base": base, "after": params(model), "rep": rep}
    if hasattr(algo, "log"):
        got["log"] = algo.log[-1]
        got["held"] = held(algo)
    return got

def refuses(key, fn):
    PHASE[0] = key
    try:
        fn()
    except Exception as e:
        out[key] = "%s: %s" % (type(e).__name__, str(e))
    else:
        out[key] = "ACCEPTED"
    PHASE[0] = "idle"
'''
)

_ROUND_SRC = _HEAD + r'''
out["secure"] = run_round("secure", Recording())
out["control"] = run_round("control", Unmasked())
out["cohort"] = run_round("cohort", Recording(), select=lambda _w: [0, 1])
# Plain FedAvg through the same instrumentation: the control for "the hub
# sees no float in a secure round", which must fail here.
out["plain"] = run_round("plain", algorithms.FedAvg())

# --- refusals, each on a live group ---------------------------------------
refuses("overflow", lambda: algorithms.SecAgg().aggregate(
    {"x": torch.tensor([0.5, 1e30 if rank == 1 else 0.25])}, weight=1.0))
refuses("nonfinite", lambda: algorithms.SecAgg().aggregate(
    {"x": torch.tensor([0.5, float("nan") if rank == 2 else 0.25])},
    weight=1.0))

def inside_tolerate():
    with federated.tolerate_missing():
        algorithms.SecAgg().aggregate({"x": torch.ones(2)}, weight=1.0)
refuses("tolerate", inside_tolerate)
refuses("average_arrived", lambda: federated.FederatedDataParallel(
    build(), algorithm=algorithms.SecAgg(), parameters=NORMS,
    on_missing="average_arrived", min_participants=2))
refuses("flag", lambda: federated.FederatedDataParallel(
    build(), parameters=NORMS, secure_aggregation=True))
refuses("wrap_fedavgm", lambda: algorithms.SecAgg(algorithms.FedAvgM()))
refuses("wrap_fedprox", lambda: algorithms.SecAgg(algorithms.FedProx()))
refuses("wrap_secagg", lambda: algorithms.SecAgg(algorithms.SecAgg()))
refuses("fraction_bits", lambda: algorithms.SecAgg(fraction_bits=0))

out["again"] = federated.agree(4242, what="a probe every rank shares")
out["hub"] = HUB
out["p2p"] = P2P
with open(dest, "w") as handle:
    json.dump(out, handle)
'''

_DROP_SRC = _HEAD + r'''
class DiesBeforeTheSum(algorithms.SecAgg):
    """Rank 2 leaves after the seeds were agreed, before the masked sum."""

    def _masked_sum(self, masked, labels, group, world):
        if rank == 2:
            out["died_after_seeds"] = True
            with open(dest, "w") as handle:
                json.dump(out, handle)
            os._exit(0)   # no goodbye: a device that lost power
        return super()._masked_sum(masked, labels, group, world)

PHASE[0] = "drop"
model = build()
fdp = federated.FederatedDataParallel(model, algorithm=DiesBeforeTheSum(),
                                      parameters=NORMS)
out["base"] = params(model)
local_train(fdp, sgd_for(model), ids, 3)
try:
    fdp.synchronize(weight=WEIGHT)
except Exception as e:
    out["dropped"] = "%s: %s" % (type(e).__name__, str(e))
    out["is_rank_dropped"] = isinstance(e, federated.RankDropped)
else:
    out["dropped"] = "ACCEPTED"
    out["is_rank_dropped"] = False
out["after"] = params(model)
with open(dest, "w") as handle:
    json.dump(out, handle)
'''


@functools.cache
def _round():
    return _fed4_spawn(_ROUND_SRC, world=WORLD, what="secagg-round")


@functools.cache
def _drop():
    return _fed4_spawn(_DROP_SRC, world=WORLD, what="secagg-drop")


# ---------------------------------------------------------------------------
# the reference and its bound
# ---------------------------------------------------------------------------

def _bound_terms(wd_abs_sum, m_abs, base_abs, total, world=WORLD, f=F):
    """``(B_a, B_inst)``: the derived per-element bounds, as exact fractions.

    Against the exact mean ``m = sum(w_k d_k) / T`` (``T`` the integer total
    weight, exact in fixed point because every weight is an integer):

    * quantisation: each rank's ``q = round(fl64(w d) * 2**f)`` is off by at
      most ``2**-(f+1)`` (the rounding) plus ``2**-53 |w d|`` (the float64
      product), so ``|S / 2**f - sum(w d)| <= world 2**-(f+1) + 2**-53
      sum|w d|`` and, dividing by ``T``, ``Q``;
    * decode: ``fl64(S / S_w)`` is one correctly rounded division,
      ``<= 2**-53 (|m| + Q)``;
    * cast to float32: ``<= 2**-24 (|m| + Q)(1 + 2**-53)``;
    * install: ``fl32(base + a)`` adds ``<= 2**-24 (|base| + |m| + B_a)``.

    Valid for normal float32 results; the deltas here are of order 1e-3 and
    nowhere near the subnormal range.
    """
    two = Fraction(2)
    q = (world * two ** -(f + 1) + two ** -53 * wd_abs_sum) / total
    b_a = q + (two ** -53 + two ** -24 * (1 + two ** -53)) * (m_abs + q)
    b_inst = b_a + two ** -24 * (base_abs + m_abs + b_a)
    return b_a, b_inst


def _check_round(reports, phase, ranks):
    """Assert one secure round's installed model against the exact FedAvg mean."""
    logs = [r[phase]["log"] for r in reports]
    total = sum(WEIGHTS[r] for r in ranks)
    for name in NORMS:
        bases = [r[phase]["base"][name] for r in reports]
        assert bases[0] == bases[1] == bases[2], (phase, name)
        deltas = [logs[r]["table"][name] for r in range(WORLD)]
        afters = [r[phase]["after"][name] for r in reports]
        aggs = [logs[r]["aggregate"][name] for r in range(WORLD)]
        # Every rank installs the same model and the same aggregate, bit for bit.
        assert afters[0] == afters[1] == afters[2], (phase, name)
        assert aggs[0] == aggs[1] == aggs[2], (phase, name)
        # Premise: the contributing deltas differ, so the mean is no operand.
        for i in ranks:
            for j in ranks:
                if i < j:
                    assert deltas[i] != deltas[j], (phase, name, i, j)
        for e, base in enumerate(bases[0]):
            wd = [Fraction(WEIGHTS[r]) * Fraction(deltas[r][e]) for r in ranks]
            m = sum(wd) / total
            b_a, b_inst = _bound_terms(sum(abs(x) for x in wd), abs(m),
                                       abs(Fraction(base)), total)
            gap_a = abs(Fraction(aggs[0][e]) - m)
            assert gap_a <= b_a, (phase, name, e, float(gap_a), float(b_a))
            gap = abs(Fraction(afters[0][e]) - (Fraction(base) + m))
            assert gap <= b_inst, (phase, name, e, float(gap), float(b_inst))
        # The aggregate is not any single contributor's own delta.
        for r in ranks:
            assert aggs[0] != deltas[r], (phase, name, r)
    return logs


def _ints_only(payload):
    """True iff every leaf of a JSON payload is an int (no float anywhere)."""
    if isinstance(payload, list):
        return all(_ints_only(p) for p in payload)
    if isinstance(payload, dict):
        return all(_ints_only(p) for p in payload.values())
    return isinstance(payload, int) and not isinstance(payload, bool)


def _flat_ints(payload):
    if isinstance(payload, list):
        for p in payload:
            yield from _flat_ints(p)
    elif isinstance(payload, dict):
        for p in payload.values():
            yield from _flat_ints(p)
    elif isinstance(payload, int) and not isinstance(payload, bool):
        yield payload


def _contains_run(seq, run):
    n = len(run)
    return any(seq[i:i + n] == run for i in range(len(seq) - n + 1))


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

def test_secagg_world_three_installs_fedavg_within_the_derived_quantisation_bound():
    """Masked FedAvg over three processes equals ``(3 d0 + 7 d1 + 2 d2) / 12``.

    The reference is exact (``Fraction`` over the deltas each rank handed
    ``aggregate``), and the tolerance is ``_bound_terms``: quantisation at
    ``f = 32`` and ``world = 3``, then the float64 decode, the float32 cast and
    the float32 install, nothing else. Every rank must install the same bits.

    Fails if: the masks do not cancel (the decode is then off by up to
    ``2**60 / 2**32``); the scale is applied twice or not at all; a weight is
    ignored (the control below); a rank's contribution is lost; two ranks
    decode differently; the round number or seeds differ between a pair.
    """
    if not _shim_present():
        return
    reports = _round()
    assert [r["rank"] for r in reports] == [0, 1, 2], reports
    assert all(r["world"] == WORLD and r["shim"] for r in reports), reports
    assert all(r["backend"] == "torchnative" for r in reports), reports
    logs = _check_round(reports, "secure", (0, 1, 2))
    assert all(log["k"] == K for log in logs), [log["k"] for log in logs]

    # Control: the unweighted mean is outside the bound somewhere, so the
    # bound is tight enough to see the weights.
    missed = False
    for name in NORMS:
        deltas = [logs[r]["table"][name] for r in range(WORLD)]
        agg = logs[0]["aggregate"][name]
        for e in range(len(agg)):
            wd = [Fraction(WEIGHTS[r]) * Fraction(deltas[r][e]) for r in range(3)]
            m = sum(wd) / 12
            b_a, _ = _bound_terms(sum(abs(x) for x in wd), abs(m), Fraction(0), 12)
            unweighted = sum(Fraction(deltas[r][e]) for r in range(3)) / 3
            if abs(unweighted - Fraction(agg[e])) > b_a:
                missed = True
    assert missed, "the unweighted mean fits the bound everywhere: no test"

    # Logging: secure, with the scale and bits; the total weight was decoded
    # from the masked sum and is exact for integer weights.
    for r, report in enumerate(reports):
        rep = report["secure"]["rep"]
        assert rep["secure_aggregation"] is True, rep
        assert rep["fraction_bits"] == F and rep["scale"] == float(1 << F), rep
        assert rep["modulus_bits"] == K, rep
        assert rep["per_rank_bound"] == B / float(1 << F), rep
        assert rep["total_weight"] == 12.0, rep
        assert rep["weight"] == float(WEIGHTS[r]), rep
        assert report["plain"]["rep"]["secure_aggregation"] is False, report["plain"]["rep"]


def test_secagg_hub_never_reads_a_plain_update():
    """What rank 0 reads off the star in the masked sum is never a plain residue.

    The hub's view is recorded in ``ProcessGroupLocal._recv_json`` on rank 0.
    For each leaf (1, 2) and each label (both parameters and the weight), the
    frame the hub read must equal what that leaf says it contributed (so the
    instrument is reading the right frames) and must differ **element by
    element** from the leaf's plain quantised residues. Every rank's own
    contribution, the hub's included, differs from its plain residues too.
    And no frame the hub reads in a secure round holds a float at all, which
    rules out a plain FedAvg path (or ``_total_weight``) running beside it.

    Sabotage, run in the same children: ``Unmasked`` zeroes every net mask,
    and there the hub's frames **equal** the plain residues, so the
    differ-assertion would fail; plain FedAvg puts floats on the hub, so the
    ints-only assertion would fail. Cannot find: a leak through a side channel
    the star does not carry (timing, the logging dict) or through p2p.
    """
    if not _shim_present():
        return
    reports = _round()
    hub = reports[0]["hub"]
    for phase, masked in (("secure", True), ("cohort", True), ("control", False)):
        frames = [(peer, p) for ph, peer, p in hub if ph == phase + ":masked"]
        for leaf in (1, 2):
            log = reports[leaf][phase]["log"]
            mine = [p for peer, p in frames if peer == leaf]
            assert len(mine) == len(log["order"]) == len(NORMS) + 1, (
                phase, leaf, len(mine), log["order"])
            for label, payload in zip(log["order"], mine):
                assert payload == log["masked"][label], (phase, leaf, label)
                plain = log["plain"][label]
                if masked:
                    same = [i for i, (a, b) in enumerate(zip(payload, plain))
                            if a == b]
                    assert not same, (phase, leaf, label, same)
                else:
                    assert payload == plain, (phase, leaf, label)
        if masked:
            for r in range(WORLD):
                log = reports[r][phase]["log"]
                for label in log["order"]:
                    pairs = zip(log["masked"][label], log["plain"][label])
                    assert all(a != b for a, b in pairs), (phase, r, label)

    secure_frames = [p for ph, _peer, p in hub
                     if ph.split(":")[0] in ("secure", "cohort", "control")]
    assert secure_frames, "the hub recorded nothing: the instrument is broken"
    assert all(_ints_only(p) for p in secure_frames), \
        "a float reached the hub in a secure round"
    plain_frames = [p for ph, _peer, p in hub if ph == "plain"]
    assert any(not _ints_only(p) for p in plain_frames), \
        "plain FedAvg put no float on the hub, so the check above cannot fail"


def test_secagg_masks_cancel_exactly_in_the_integer_sum():
    """``sum(masked) mod 2**k == sum(plain) mod 2**k``, exactly, per element.

    And that number is what every rank's all_reduce returned. The control:
    no proper subset of the ranks cancels (two ranks' masked sum differs from
    their plain sum), so cancellation needs every pair's both ends.

    Fails if: a pair's two ends used different seeds, round numbers, labels or
    chunk indices; a sign is wrong (``j > i`` adds, ``j < i`` subtracts); the
    modulus differs between ranks; the hub's int64 fold wrapped or rounded
    through a float (``2**61``-sized residues are past float64's 53 bits).
    """
    if not _shim_present():
        return
    reports = _round()
    for phase in ("secure", "cohort"):
        logs = [r[phase]["log"] for r in reports]
        for label in logs[0]["order"]:
            masked = [logs[r]["masked"][label] for r in range(WORLD)]
            plain = [logs[r]["plain"][label] for r in range(WORLD)]
            n = len(masked[0])
            total_m = [sum(masked[r][e] for r in range(WORLD)) % M for e in range(n)]
            total_p = [sum(plain[r][e] for r in range(WORLD)) % M for e in range(n)]
            assert total_m == total_p, (phase, label)
            for r in range(WORLD):
                assert logs[r]["sums"][label] == total_m, (phase, r, label)
            for i, j in ((0, 1), (0, 2), (1, 2)):
                part_m = [(masked[i][e] + masked[j][e]) % M for e in range(n)]
                part_p = [(plain[i][e] + plain[j][e]) % M for e in range(n)]
                assert part_m != part_p, (phase, label, i, j)


def test_secagg_seeds_cross_only_the_direct_pair_links():
    """Each seed goes ``i -> j`` on the pair's own link and never reaches the hub.

    Per round, rank ``r`` sends exactly one 32-byte ``uint8`` frame on the seed
    tag to each ``j > r`` and receives one from each ``i < r``; rank 0 receives
    none; nothing else uses point-to-point. The seed of pair (1, 2), read from
    rank 1's send, appears in no frame rank 0 read in any phase (neither as
    its 32 bytes nor as its four int64 words), is fresh per round, and is not
    in the logging dict or the algorithm's state on either end.

    Fails if: seeds were relayed through a collective (the hub would hold
    them, and the send counts would be zero); a seed was reused across rounds;
    ``_get_federated_logging_data`` or ``SecAgg`` kept one. Cannot find: a seed
    relayed in a transformed encoding through the hub, or a link that is read
    by someone off the machine (the links are not encrypted: the threat model
    says so).
    """
    if not _shim_present():
        return
    reports = _round()
    phases = ("secure", "control", "cohort")
    s12 = {}
    for r, report in enumerate(reports):
        for ph, kind, _peer, _tag, _b in report["p2p"]:
            assert ph.endswith(":seeds"), (r, ph, kind)
        for phase in phases:
            sends = [(peer, tag, b) for ph, kind, peer, tag, b in report["p2p"]
                     if ph == phase + ":seeds" and kind == "send"]
            recvs = [(peer, tag) for ph, kind, peer, tag, _b in report["p2p"]
                     if ph == phase + ":seeds" and kind == "recv"]
            assert sorted(p for p, _t, _b in sends) == list(range(r + 1, WORLD)), \
                (r, phase, sends)
            assert all(t == SEED_TAG and len(b) == 32 for _p, t, b in sends), \
                (r, phase)
            assert sorted(p for p, _t in recvs) == list(range(r)), (r, phase, recvs)
            if r == 1:
                s12[phase] = next(b for p, _t, b in sends if p == 2)
    assert not [k for ph, k, *_ in reports[0]["p2p"] if k == "recv"], \
        "rank 0 received a point-to-point frame"
    assert len({bytes(s) for s in s12.values()}) == len(phases), \
        "the seed of pair (1, 2) was reused across rounds"

    hub_ints = [list(_flat_ints(p)) for _ph, _peer, p in reports[0]["hub"]]
    for phase, seed in s12.items():
        words = list(struct.unpack("<4q", bytes(seed)))
        for seq in hub_ints:
            assert not _contains_run(seq, seed), (phase, "seed bytes at the hub")
            assert not _contains_run(seq, words), (phase, "seed words at the hub")
        for r in (1, 2):
            text = json.dumps(reports[r][phase]["rep"]) + reports[r][phase]["held"]
            assert bytes(seed).hex() not in text, (phase, r)
            assert repr(bytes(seed)) not in text, (phase, r)


def test_secagg_with_a_proper_subset_cohort_masks_the_abstention():
    """Cohort ``[0, 1]`` of three: the mean is over 0 and 1, rank 2 abstains masked.

    Same exact reference and derived bound as the full round, over the
    selected ranks (``T = 10``). Rank 2's plain residues are all zero (an
    abstention is a zero table at weight zero) and the hub still reads only
    masked values from it (the hub test covers this phase too).

    Fails if: an abstention is weighted, masks are skipped for a zero
    contribution, or the masked total weight is wrong (it must be 10).
    """
    if not _shim_present():
        return
    reports = _round()
    _check_round(reports, "cohort", (0, 1))
    log2 = reports[2]["cohort"]["log"]
    for label in log2["order"]:
        assert all(v == 0 for v in log2["plain"][label]), label
        assert all(v != 0 for v in log2["masked"][label]), label
    for report in reports:
        rep = report["cohort"]["rep"]
        assert rep["total_weight"] == 10.0 and rep["cohort"] == [0, 1], rep
        assert rep["secure_aggregation"] is True, rep


def test_secagg_refuses_by_name_what_it_cannot_encode_or_combine():
    """Overflow, a non-finite value and the unsupported combinations refuse by name.

    The overflow refusal fires on **every** rank (one int64 count in the clear
    makes it symmetric) and quotes the bound ``B / 2**f`` the module derives,
    recomputed here from ``world`` and ``f``. Fails if: a rank wraps instead of
    refusing, only the offending rank refuses (the others would hang or
    decode garbage), the bound in the message is not the derived one, or any
    of the combinations is silently accepted. The group must still be usable
    afterwards.
    """
    if not _shim_present():
        return
    reports = _round()
    limit = repr(B / float(1 << F))
    for r, report in enumerate(reports):
        msg = report["overflow"]
        assert msg.startswith("ValueError:"), (r, msg)
        assert "past the fixed-point bound" in msg and limit in msg, (r, msg)
        assert "1 of 3 rank(s)" in msg, (r, msg)
        if r == 1:
            assert "largest |w * d| = 1e+30" in msg, msg

        nan = report["nonfinite"]
        assert nan.startswith("ValueError:") and "1 of 3 rank(s)" in nan, (r, nan)
        if r == 2:
            assert "nan or inf" in nan, nan

        tol = report["tolerate"]
        assert tol.startswith("NotImplementedError:"), tol
        assert "Recovery is not built" in tol, tol

        aa = report["average_arrived"]
        assert aa.startswith("NotImplementedError:"), aa
        assert "average_arrived" in aa and "not built" in aa, aa

        flag = report["flag"]
        assert flag.startswith("NotImplementedError:"), flag
        assert "not a flag" in flag and "algorithms.SecAgg" in flag, flag

        for key, who in (("wrap_fedavgm", "FedAvgM"), ("wrap_fedprox", "FedProx")):
            msg = report[key]
            assert msg.startswith("NotImplementedError:"), (key, msg)
            assert "only FedAvg is wrapped" in msg and who in msg, (key, msg)
        assert report["wrap_secagg"].startswith("NotImplementedError:"), report
        assert "masking twice" in report["wrap_secagg"], report["wrap_secagg"]
        assert report["fraction_bits"].startswith("ValueError:"), report

        assert report["again"] == 4242 * WORLD, report["again"]


def test_secagg_a_rank_lost_after_the_seeds_aborts_and_undoes_the_round():
    """Rank 2 exits between the seed exchange and the masked sum.

    Ranks 0 and 1 must raise ``RankDropped`` naming secure aggregation and
    saying recovery is not built, and their models must be back at the base,
    byte for byte (``on_missing='refuse'`` undoes the round). Fails if: the
    survivors decode a sum with rank 2's masks in it and install noise; the
    error is a bare socket error rather than the by-name refusal; the local
    training is left in place.
    """
    if not _shim_present():
        return
    reports = _drop()
    assert reports[2].get("died_after_seeds") is True, reports[2]
    for r in (0, 1):
        msg = reports[r]["dropped"]
        assert msg.startswith("RankDropped:"), (r, msg)
        assert "SecAgg" in msg and "recovery" in msg and "is not built" in msg, msg
        assert reports[r]["is_rank_dropped"] is True, reports[r]
        assert reports[r]["after"] == reports[r]["base"], (
            r, "the round was not undone")


def _main():
    failures = _skip.run_tests(
        [(name, fn) for name, fn in sorted(globals().items())
         if name.startswith("test_")],
        suite="test_secagg",
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
