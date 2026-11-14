"""Ranks as threads of one process (issue #281, BACKEND.md D5 and §9 step 6, SPEC S6.10/S6.11).

Step 6's deciding test, from BACKEND.md §9: "A world of 4 as 4 threads of one
process agrees with gloo; mixed world (2 threads + 1 remote process) passes;
on a GIL build the test is `UNVERIFIED`, not a pass".

What is run, every side in its own subprocess:

* **a world of 4 as 4 threads of one shim process.** Each thread calls the
  unchanged upstream `dist.init_process_group(backend="torchnative",
  init_method="tcp://...", rank=r, world_size=4)` after
  `torchnative.distributed.threads.install()`, then runs the very probe
  sources `test_collect2.py` (eleven collectives) and `test_p2p.py`
  (point-to-point) run in processes. The answers are compared with upstream
  gloo run as 4 processes (the cached oracles of those suites), at the
  tolerance `test_collect2.py` derives, and with the shim's own 4-process
  world bit for bit (its star folds in rank order, so a thread world must give
  the same bits).
* **a mixed world of 3**: ranks 0 and 1 are threads of one shim process, rank
  2 is a separate shim process. Same probes, against gloo at world 3.
* **the thread launcher**: `elastic_launch(LaunchConfig(start_method=
  "thread"), entry)(...)`, results per rank, a failing rank named, a rank
  stuck outside `torch.distributed` named as not stopped.

The two agreement tests print `SKIP ... UNVERIFIED` on an interpreter with
the GIL (measured in the child, after the shim was imported): their values are
asserted first and can fail, but the claim of step 6 is ranks running in
parallel on free-threaded 3.15t, which a GIL build cannot show.

Nullification kept in the suite: `test_without_the_thread_local_world_two_
rank_threads_collide` runs the same threads without `install()` and asserts
they collide, so `install()` is shown to be what makes the other tests pass.

Written before the implementation was run anywhere (AGENTS.md rule 10: agents
do not run suites); the first run is CI's. Nothing here is *agrees* until then.
"""

import functools
import json
import os
import subprocess
import sys
import tempfile
import time

import _skip

from test_collect2 import (C2_VALUE_PROBES, _C2_PROBE_SRC, _c2_free_port,
                           c2_assert_matches, c2_gloo32, c2_gloo64, c2_ok,
                           c2_shim)
from test_p2p import P2P_VALUE_PROBES, _P2P_PROBE_SRC, p2p_gloo
from test_shim import _CKPT_VENDOR_DIR


def _shim_env(extra=None):
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    for key in ("TORCHNATIVE_DIST_TRANSPORT", "TORCHNATIVE_DIST_BIND_ADDR",
                "TR_NO_INSTALL", "TR_PROBES", "TR_JOIN_S", "RANK", "WORLD_SIZE",
                "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    env.update(extra or {})
    return env


# ---------------------------------------------------------------------------
# the child: some ranks of a world, as threads (or one rank on the main thread)
# ---------------------------------------------------------------------------
#
# argv: port world ranks(comma separated) mode(threads|main) dest
# TR_PROBES=full runs the collect2 and p2p probes; =init only initialises,
# reports and runs a two-op smoke. TR_NO_INSTALL=1 skips install() (the
# nullification control). TR_JOIN_S bounds the wait for the rank threads; a
# thread still running after it is reported in `unfinished`, not waited for.

_TR_SRC = (
    "import datetime, json, os, sys, sysconfig, threading, time, traceback\n"
    "import torch\n"
    'assert hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded upstream torch, not the shim")\n'
    "import torch.distributed as dist\n"
    "import torchnative.distributed  # noqa: F401 -- registers backend='torchnative'\n"
    "from torchnative.distributed import threads as tnthreads\n"
    "from torchnative.distributed.backend import transport, transport_of\n"
    + _C2_PROBE_SRC
    + _P2P_PROBE_SRC
    + r'''
port, world, dest = int(sys.argv[1]), int(sys.argv[2]), sys.argv[5]
ranks = [int(r) for r in sys.argv[3].split(",")]
mode = sys.argv[4]
probes = os.environ.get("TR_PROBES", "full")
no_install = os.environ.get("TR_NO_INSTALL") == "1"
c10d_c = torch._C._distributed_c10d

top = {"mode": mode, "pid": os.getpid(), "ranks": {}}
if mode == "threads" and not no_install:
    tnthreads.install()
top["installed"] = tnthreads.installed()

gate = threading.Barrier(len(ranks))


def field(rep, key, fn):
    try:
        rep[key] = fn()
    except Exception as exc:
        rep[key] = {"exc": "%s: %s" % (type(exc).__name__, exc)}


def rank_main(rank):
    rep = {"thread": threading.current_thread().name}
    top["ranks"][str(rank)] = rep
    try:
        dist.init_process_group(
            backend="torchnative", init_method="tcp://127.0.0.1:%d" % port,
            rank=rank, world_size=world,
            timeout=datetime.timedelta(seconds=180))
    except Exception as exc:
        rep["init_exc"] = "%s: %s" % (type(exc).__name__, exc)
        return
    field(rep, "rank", dist.get_rank)
    field(rep, "world", dist.get_world_size)
    field(rep, "backend", dist.get_backend)
    field(rep, "kinds", lambda: {str(p): transport_of(p)
                                 for p in range(world) if p != rank})
    if no_install:
        # The control: the groups collided, so collectives would read another
        # thread's group. Report what each thread sees and stop here.
        return
    pg = dist.group.WORLD
    field(rep, "group_name", lambda: pg.group_name)
    field(rep, "isolation", lambda: bool(c10d_c._get_thread_isolation_mode()))
    field(rep, "resolves_own",
          lambda: c10d_c._resolve_process_group(pg.group_name) is pg)

    def smoke():
        t = torch.tensor([float(rank + 1)])
        dist.all_reduce(t)
        buf = torch.zeros(2, dtype=torch.int64)
        nxt, prv = (rank + 1) % world, (rank - 1) % world
        work = dist.isend(torch.tensor([rank, 7 * rank], dtype=torch.int64), nxt)
        dist.recv(buf, prv)
        work.wait()
        return {"all_reduce": t.tolist(), "ring": buf.tolist()}
    field(rep, "smoke", smoke)
    if probes == "full":
        out = c2_run(torch, dist, rank, world, torch.float32, False)
        out.update(p2p_run(torch, dist, rank, world, False))
        rep["probes"] = out
    dist.barrier()

    # Per-thread teardown: the first local rank destroys its world while the
    # others still hold theirs; they must be untouched by it.
    first = rank == ranks[0]
    if first:
        dist.destroy_process_group()
    gate.wait(120)
    if not first:
        field(rep, "survives_peer_destroy", lambda: {
            "initialized": dist.is_initialized(),
            "resolves_own": c10d_c._resolve_process_group(pg.group_name) is pg,
            "rank": dist.get_rank()})
        dist.destroy_process_group()
    rep["initialized_after_destroy"] = dist.is_initialized()


def run(rank):
    try:
        rank_main(rank)
    except BaseException:
        top["ranks"].setdefault(str(rank), {})["thread_exc"] = traceback.format_exc()


if mode == "main":
    assert len(ranks) == 1, ranks
    run(ranks[0])
else:
    pool = []
    for rank in ranks:
        th = threading.Thread(target=run, args=(rank,), daemon=True,
                              name="tr-rank-%d" % rank)
        th.start()
        pool.append(th)
        if rank == 0:
            time.sleep(0.4)  # rank 0's thread hosts the TCPStore
    deadline = time.monotonic() + float(os.environ.get("TR_JOIN_S", "900"))
    for th in pool:
        th.join(max(0.0, deadline - time.monotonic()))
    top["unfinished"] = [th.name for th in pool if th.is_alive()]

if not no_install:
    # Every transport thread is gone once every rank has destroyed its group.
    deadline = time.monotonic() + 15
    def lingering():
        return sorted(th.name for th in threading.enumerate()
                      if th.name.startswith(("torchnative-p2p",
                                             "ProcessGroupLocal-async")))
    while lingering() and time.monotonic() < deadline:
        time.sleep(0.1)
    top["lingering"] = lingering()
    top["inproc_registry"] = len(transport.InprocTransport._registry)
top["gil_enabled"] = tnthreads.gil_enabled()
top["free_threaded_build"] = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))
with open(dest, "w") as handle:
    json.dump(top, handle)
# A collided thread (the control) may still be blocked in a rendezvous that
# can never complete; it is a daemon, and the report is written.
os._exit(0)
'''
)


def _tr_spawn(specs, what, extra_env=None, timeout=1200, python=None):
    """Run one child per `(ranks, mode)` in `specs`; return their reports in order.

    `python` is the interpreter (default: this one), for the free-threaded run.

    Output goes to files; every child is killed on every path; a timeout or a
    non-zero exit names every child's state and keeps what it said.
    """
    import atexit
    import shutil
    tmp = tempfile.mkdtemp(prefix="threadranks-%s-" % what)
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    port = _c2_free_port()
    world = sum(len(ranks) for ranks, _ in specs)
    env = _shim_env(extra_env)
    procs, streams = [], []
    python = python or sys.executable

    def states():
        lines = []
        for i, proc in enumerate(procs):
            lines.append("--- child %d %r: returncode %r ---" % (
                i, specs[i], proc.poll()))
            for ext in ("out", "err"):
                path = os.path.join(tmp, "child-%d.%s" % (i, ext))
                try:
                    with open(path, "rb") as handle:
                        lines.append(handle.read()[-4000:].decode(errors="replace"))
                except OSError:
                    pass
        return "\n".join(lines)

    try:
        for i, (ranks, mode) in enumerate(specs):
            out = open(os.path.join(tmp, "child-%d.out" % i), "wb")
            err = open(os.path.join(tmp, "child-%d.err" % i), "wb")
            streams += [out, err]
            procs.append(subprocess.Popen(
                [python, "-c", _TR_SRC, str(port), str(world),
                 ",".join(str(r) for r in ranks), mode,
                 os.path.join(tmp, "child-%d.json" % i)],
                stdout=out, stderr=err, env=env))
            if 0 in ranks:
                time.sleep(0.6)  # the child holding rank 0 binds the store
        deadline = time.monotonic() + timeout
        for proc in procs:
            try:
                proc.wait(timeout=max(0.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                for stream in streams:
                    stream.flush()
                raise RuntimeError("threadranks/%s: not every child finished "
                                   "within %ds\n%s" % (what, timeout, states()))
        for stream in streams:
            stream.flush()
        reports = []
        for i, proc in enumerate(procs):
            if proc.returncode != 0:
                raise RuntimeError("threadranks/%s: child %d exited %d\n%s"
                                   % (what, i, proc.returncode, states()))
            with open(os.path.join(tmp, "child-%d.json" % i)) as handle:
                reports.append(json.load(handle))
        return reports
    finally:
        for proc in procs:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        for stream in streams:
            stream.close()


def _by_rank(reports):
    """{global rank: that rank's report}, with every rank's child attached."""
    out = {}
    for child in reports:
        for rank, rep in child["ranks"].items():
            rep = dict(rep)
            rep["_child"] = {k: v for k, v in child.items() if k != "ranks"}
            out[int(rank)] = rep
    return out


def _ran(rep, rank):
    for key in ("init_exc", "thread_exc"):
        assert key not in rep, "rank %d: %s" % (rank, rep[key])
    return rep


@functools.cache
def tr_four_threads():
    return _tr_spawn([([0, 1, 2, 3], "threads")], "four-threads")


@functools.cache
def tr_four_threads_free_threaded():
    """The same world on a 3.15t interpreter, when the issue #51 inputs are given.

    `TORCHNATIVE_FT_PYTHON` and `TORCHNATIVE_FT_STAGE` are the inputs
    `tests/api/test_freethreaded.py` reads. The stage goes first on the path so
    its free-threaded `torch._C` is the one imported; the checkout's package
    directory follows for `torchnative`.
    """
    py = os.environ.get("TORCHNATIVE_FT_PYTHON")
    stage = os.environ.get("TORCHNATIVE_FT_STAGE")
    if not py or not stage:
        raise _skip.Skip("TORCHNATIVE_FT_PYTHON / TORCHNATIVE_FT_STAGE not set "
                         "(issue #51); the free-threaded run of step 6 was not made")
    return _tr_spawn([([0, 1, 2, 3], "threads")], "four-threads-ft",
                     {"PYTHONPATH": os.pathsep.join([stage, _CKPT_VENDOR_DIR]),
                      "PYTHON_GIL": "0"}, python=py)


@functools.cache
def tr_mixed():
    return _tr_spawn([([0, 1], "threads"), ([2], "main")], "mixed")


@functools.cache
def tr_threads_forced_shm():
    return _tr_spawn([([0, 1], "threads")], "forced-shm",
                     {"TORCHNATIVE_DIST_TRANSPORT": "shm", "TR_PROBES": "init"})


@functools.cache
def tr_no_install():
    return _tr_spawn([([0, 1], "threads")], "no-install",
                     {"TR_NO_INSTALL": "1", "TR_PROBES": "init",
                      "TR_JOIN_S": "90"}, timeout=600)


def _unverified_if_gil(reports, what):
    """Step 6's claim is parallel ranks; a GIL interpreter took turns. Say so by name."""
    gil = [c["gil_enabled"] for c in reports]
    assert all(isinstance(g, bool) for g in gil), gil
    if any(gil):
        _skip.skip(
            "UNVERIFIED: %s, values asserted and equal, but the interpreter "
            "ran with the GIL (sys._is_gil_enabled() was True after the shim "
            "was imported; free-threaded build: %s), so the rank threads took "
            "turns. BACKEND.md §9 step 6 claims parallel ranks on "
            "free-threaded 3.15t, which this run cannot show"
            % (what, [c["free_threaded_build"] for c in reports]))


def _assert_collect2_and_p2p_agree(by_rank, world):
    up32, up64, gloo = c2_gloo32(world), c2_gloo64(world), p2p_gloo(world)
    seen = 0
    for rank in range(world):
        probes = _ran(by_rank[rank], rank)["probes"]
        for key in C2_VALUE_PROBES:
            c2_assert_matches(key, world, rank, c2_ok(probes, key),
                              c2_ok(up32[rank], key), c2_ok(up64[rank], key))
            seen += 1
        for key in P2P_VALUE_PROBES:
            ours, theirs = c2_ok(probes, key), c2_ok(gloo[rank], key)
            assert ours == theirs, (
                "%s at world %d rank %d (threads) differs from upstream gloo.\n"
                "  shim: %r\n  gloo: %r" % (key, world, rank, ours, theirs))
            seen += 1
    assert seen == (len(C2_VALUE_PROBES) + len(P2P_VALUE_PROBES)) * world, seen


# ---------------------------------------------------------------------------
# a world of 4 as 4 threads
# ---------------------------------------------------------------------------

def test_four_rank_threads_of_one_process_agree_with_gloo():
    """Eleven collectives and point-to-point, 4 threads of one process vs 4 gloo processes.

    Fails if any value differs beyond test_collect2's derived tolerance (zero
    for every collective that only moves numbers), if a p2p value is not bit
    equal, if a probe is missing, or if the four ranks were not one process.
    Then, on a GIL interpreter, reports UNVERIFIED instead of ok.
    """
    reports = tr_four_threads()
    assert len(reports) == 1 and reports[0]["mode"] == "threads", reports
    assert reports[0]["installed"] is True, reports[0]
    assert reports[0].get("unfinished") == [], reports[0].get("unfinished")
    by_rank = _by_rank(reports)
    assert sorted(by_rank) == [0, 1, 2, 3], sorted(by_rank)
    assert len({rep["thread"] for rep in by_rank.values()}) == 4
    _assert_collect2_and_p2p_agree(by_rank, 4)
    _unverified_if_gil(reports, "a world of 4 as 4 threads agrees with gloo")


def test_four_rank_threads_on_free_threaded_python_run_without_the_gil_and_agree():
    """Step 6's claim itself: the same world of 4 threads on 3.15t, GIL off, values equal gloo's.

    SKIP by name without the issue #51 inputs. With them, fails if the GIL
    was on in the child after the shim was imported (the claim would be
    unshown), if the interpreter is not a free-threaded build, or on any value.
    """
    reports = tr_four_threads_free_threaded()
    child = reports[0]
    assert child["free_threaded_build"] is True, child["free_threaded_build"]
    assert child["gil_enabled"] is False, (
        "the GIL was enabled after importing the shim on a free-threaded "
        "build; the rank threads took turns")
    assert child.get("unfinished") == [], child.get("unfinished")
    _assert_collect2_and_p2p_agree(_by_rank(reports), 4)


def test_four_rank_threads_give_the_bits_four_processes_give():
    """The star folds in rank order, so threads and processes must agree bit for bit.

    An oracle with no tolerance at all: the shim's own 4-process world
    (test_collect2's cached run). Fails on any differing bit, which a thread
    world that mixed up two ranks' contributions would show at once.
    """
    by_rank, procs = _by_rank(tr_four_threads()), c2_shim(4)
    for rank in range(4):
        probes = _ran(by_rank[rank], rank)["probes"]
        for key in C2_VALUE_PROBES:
            assert c2_ok(probes, key) == c2_ok(procs[rank], key), (
                key, rank, c2_ok(probes, key), c2_ok(procs[rank], key))


def test_rank_threads_pair_over_inproc_as_the_backend_reports():
    """`transport_of` (read from the backend's mesh, not inferred) is inproc for every pair."""
    by_rank = _by_rank(tr_four_threads())
    for rank in range(4):
        kinds = _ran(by_rank[rank], rank)["kinds"]
        assert kinds == {str(p): "inproc" for p in range(4) if p != rank}, (
            rank, kinds)


def test_each_rank_thread_has_its_own_default_group_and_registry():
    """Four default groups all named the same, each thread resolving its own.

    Fails if `get_rank()` in a thread is not that thread's rank (the shared
    world), if the shim's registry handed one thread another's group (thread
    isolation off), or if the first thread's destroy_process_group took the
    others' groups with it (upstream calls `_unregister_all_process_groups`).
    """
    by_rank = _by_rank(tr_four_threads())
    names = set()
    for rank in range(4):
        rep = _ran(by_rank[rank], rank)
        assert rep["rank"] == rank and rep["world"] == 4, rep
        assert rep["backend"] == "torchnative", rep["backend"]
        assert rep["isolation"] is True, rep["isolation"]
        assert rep["resolves_own"] is True, (rank, rep["resolves_own"])
        names.add(rep["group_name"])
        assert rep["smoke"] == {"all_reduce": [10.0],
                                "ring": [(rank - 1) % 4, 7 * ((rank - 1) % 4)]}, (
            rank, rep["smoke"])
        assert rep["initialized_after_destroy"] is False, rep
        if rank != 0:
            assert rep["survives_peer_destroy"] == {
                "initialized": True, "resolves_own": True, "rank": rank}, (
                rank, rep["survives_peer_destroy"])
    # One name for four groups: the collision thread isolation exists for.
    assert len(names) == 1, names


def test_destroying_every_rank_thread_leaves_no_transport_thread_or_listener():
    """After every thread's destroy_process_group: no mesh or async thread, no inproc listener."""
    child = tr_four_threads()[0]
    assert child["lingering"] == [], child["lingering"]
    assert child["inproc_registry"] == 0, child["inproc_registry"]


def test_without_the_thread_local_world_two_rank_threads_collide():
    """The nullification, kept: the same two threads without `install()`.

    Upstream's process-global world cannot hold two default groups. Read
    off source, the collision takes one of three forms: one thread's
    init_process_group refuses ("initialize the default process group
    twice"); or the shared `group_count` names the two groups "0" and "1", so
    their Store prefixes differ and the rendezvous never meets (a thread still
    running after 90 s); or both finish and read one default group, so
    `get_rank()` is not {0, 1}. If none happens, `install()` is not what makes
    the other tests in this file pass, and they prove less than claimed.
    """
    child = tr_no_install()[0]
    assert child["installed"] is False, child
    reps = child["ranks"]
    assert sorted(reps) == ["0", "1"], reps
    refused = [r.get("init_exc", "") for r in reps.values()]
    seen = {r.get("rank") for r in reps.values()
            if "init_exc" not in r and "rank" in r}
    collided = (any("twice" in msg for msg in refused)
                or child["unfinished"] != [] or seen != {0, 1})
    assert collided, ("two rank threads initialized without the thread-local "
                      "world and each saw its own rank", reps)


def test_forcing_shm_between_rank_threads_changes_the_reported_transport():
    """TORCHNATIVE_DIST_TRANSPORT=shm on two threads: the backend reports shm and still works.

    So the `inproc` asserted above is read from the backend's choice, not a
    constant: forcing another kind changes the answer.
    """
    by_rank = _by_rank(tr_threads_forced_shm())
    for rank in range(2):
        rep = _ran(by_rank[rank], rank)
        assert rep["kinds"] == {str(1 - rank): "shm"}, rep["kinds"]
        assert rep["smoke"] == {"all_reduce": [3.0],
                                "ring": [1 - rank, 7 * (1 - rank)]}, rep["smoke"]


# ---------------------------------------------------------------------------
# a mixed world: 2 threads of one process + 1 separate process
# ---------------------------------------------------------------------------

def test_a_mixed_world_of_two_threads_and_one_process_agrees_with_gloo():
    """Ranks 0 and 1 threads of one process, rank 2 another process; world 3 vs gloo.

    Fails as the four-thread test does, and also if the two thread ranks are
    not in one process or rank 2 shares it. UNVERIFIED on a GIL interpreter.
    """
    reports = tr_mixed()
    assert [c["mode"] for c in reports] == ["threads", "main"], reports
    assert reports[0]["pid"] != reports[1]["pid"], reports
    assert reports[0]["installed"] is True and reports[1]["installed"] is False
    by_rank = _by_rank(reports)
    assert sorted(by_rank) == [0, 1, 2], sorted(by_rank)
    _assert_collect2_and_p2p_agree(by_rank, 3)
    _unverified_if_gil(reports, "a mixed world of 2 threads and 1 process "
                       "agrees with gloo")


def test_a_mixed_world_pairs_threads_over_inproc_and_the_process_over_shm():
    by_rank = _by_rank(tr_mixed())
    want = {0: {"1": "inproc", "2": "shm"}, 1: {"0": "inproc", "2": "shm"},
            2: {"0": "shm", "1": "shm"}}
    for rank in range(3):
        assert _ran(by_rank[rank], rank)["kinds"] == want[rank], (
            rank, by_rank[rank]["kinds"])


# ---------------------------------------------------------------------------
# the thread launcher
# ---------------------------------------------------------------------------

_LAUNCH_SRC = r'''
import json, os, sys, threading, time
import torch
assert hasattr(torch._C, "_aten_implemented"), (
    "this subprocess loaded upstream torch, not the shim")
import torch.distributed as dist
from torchnative.distributed import launcher
from torchnative.distributed.launcher import api as launcher_api
from torchnative.distributed.backend import transport_of

out = {}

def probe(key, fn):
    try:
        out[key] = {"ok": fn()}
    except Exception as exc:
        out[key] = {"exc": "%s: %s" % (type(exc).__name__, exc)}

ENV_KEYS = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
out["env_before"] = {k: os.environ.get(k) for k in ENV_KEYS}

def entry(scale):
    dist.init_process_group(backend="torchnative")   # env://, unchanged
    r, w = dist.get_rank(), dist.get_world_size()
    t = torch.tensor([float(r + 1), float(r * scale)])
    dist.all_reduce(t)
    return {"rank": r, "world": w, "sum": t.tolist(),
            "kinds": {str(p): transport_of(p) for p in range(w) if p != r},
            "info": launcher.rank_info(),
            "thread": threading.current_thread().name,
            "env": {k: os.environ.get(k) for k in ENV_KEYS}}

def launch(n, scale):
    cfg = launcher.LaunchConfig(nproc_per_node=n, start_method="thread")
    got = launcher.elastic_launch(cfg, entry)(scale)
    return {str(k): v for k, v in got.items()}

probe("launch4", lambda: launch(4, 10))
# Twice in one process: the first launch's groups, registry entries and
# transport threads must be gone, or this one collides with them.
probe("relaunch2", lambda: launch(2, 3))

def entry_fails(_):
    dist.init_process_group(backend="torchnative")
    if dist.get_rank() == 1:
        raise ValueError("boom from rank 1")
    dist.barrier()          # rank 1 never arrives
    return "passed a barrier rank 1 never reached"

def failing(entrypoint, n):
    t0 = time.time()
    cfg = launcher.LaunchConfig(nproc_per_node=n, start_method="thread")
    try:
        launcher.elastic_launch(cfg, entrypoint)(None)
    except launcher.ChildFailedError as error:
        return {"failures": {str(k): v for k, v in error.failures.items()},
                "unstopped": error.unstopped,
                "after": sorted(error.after_failure),
                "message": str(error), "seconds": time.time() - t0}
    return "NO ERROR"

probe("fails", lambda: failing(entry_fails, 3))

time.sleep(1.0)
out["lingering"] = sorted(th.name for th in threading.enumerate()
                          if th.name.startswith(("torchnative-rank",
                                                 "torchnative-p2p")))

def entry_stuck(_):
    dist.init_process_group(backend="torchnative")
    r = dist.get_rank()
    if r == 1:
        raise ValueError("boom from rank 1")
    if r == 0:
        time.sleep(40)      # outside torch.distributed: nothing can wake it
        return "slept"
    dist.barrier()
    return "passed"

launcher_api.THREAD_STOP_GRACE = 3.0
probe("stuck", lambda: failing(entry_stuck, 3))

with open(sys.argv[1], "w") as handle:
    json.dump(out, handle)
os._exit(0)   # the stuck rank 0 thread is a daemon; do not wait for it
'''


@functools.cache
def tr_launcher():
    import atexit
    import shutil
    tmp = tempfile.mkdtemp(prefix="threadranks-launch-")
    atexit.register(shutil.rmtree, tmp, ignore_errors=True)
    dest = os.path.join(tmp, "out.json")
    proc = subprocess.run([sys.executable, "-c", _LAUNCH_SRC, dest],
                          capture_output=True, text=True, env=_shim_env(),
                          timeout=900)
    if proc.returncode != 0 or not os.path.exists(dest):
        raise RuntimeError(
            "threadranks/launcher: the shim subprocess exited %d\n--- stdout "
            "---\n%s\n--- stderr ---\n%s" % (proc.returncode, proc.stdout[-4000:],
                                             proc.stderr[-6000:]))
    with open(dest) as handle:
        return json.load(handle)


def test_the_thread_launcher_starts_rank_threads_and_returns_each_result():
    """`elastic_launch(LaunchConfig(start_method="thread"), entry)(10)` at 4, then again at 2.

    Fails if a rank is missing or holds another's value, if the all-reduce
    sum is wrong (10 and 60 at world 4; 3 and 3 at world 2), if any pair is
    not inproc, if the entrypoint did not run on a launcher thread, if
    `rank_info()` is not the variables process mode would set, or if the
    launcher wrote RANK and the rest into the process environment.
    """
    report = tr_launcher()
    env_before = report["env_before"]
    for key, world, sums in (("launch4", 4, [10.0, 60.0]),
                             ("relaunch2", 2, [3.0, 3.0])):
        got = c2_ok(report, key)
        assert sorted(got) == [str(r) for r in range(world)], (key, got)
        for r in range(world):
            res = got[str(r)]
            assert res["rank"] == r and res["world"] == world, (key, res)
            assert res["sum"] == sums, (key, r, res["sum"])
            assert res["kinds"] == {str(p): "inproc" for p in range(world)
                                    if p != r}, (key, r, res["kinds"])
            assert res["thread"] == "torchnative-rank-%d" % r, res["thread"]
            info = res["info"]
            assert (info["RANK"], info["LOCAL_RANK"], info["WORLD_SIZE"]) == (
                str(r), str(r), str(world)), info
            assert res["env"] == env_before, (
                "thread mode must not write the per-process environment",
                res["env"], env_before)


def test_a_failing_rank_thread_is_named_and_its_peers_are_woken():
    """Rank 1 raises; ranks 0 and 2 are blocked in a barrier with it.

    Fails if the launcher returns, if the failure is not rank 1's own
    traceback, if a woken peer is reported as the failure or as unstopped,
    if the peers had to wait out a socket timeout (30 s) to stop, or if a
    rank or transport thread outlives the launch.
    """
    report = tr_launcher()
    got = c2_ok(report, "fails")
    assert got != "NO ERROR", got
    assert list(got["failures"]) == ["1"], got
    assert "boom from rank 1" in got["failures"]["1"], got["failures"]
    assert "rank 1 failed" in got["message"], got["message"]
    assert got["unstopped"] == [], got
    assert set(got["after"]) <= {0, 2}, got
    assert got["seconds"] < 25, got["seconds"]
    assert report["lingering"] == [], report["lingering"]


def test_a_rank_thread_that_cannot_be_woken_is_named_as_not_stopped():
    """Rank 1 raises while rank 0 sleeps outside torch.distributed; a thread cannot be killed.

    Fails unless the error names rank 1 as the failure and rank 0 as still
    running after the grace (3 s here), and returns well before rank 0's 40 s.
    """
    got = c2_ok(tr_launcher(), "stuck")
    assert got != "NO ERROR", got
    assert list(got["failures"]) == ["1"], got
    assert got["unstopped"] == [0], got
    assert "rank thread(s) 0 did not return" in got["message"], got["message"]
    assert got["seconds"] < 30, got["seconds"]


def _main():
    return _skip.run_tests(
        [(name, fn) for name, fn in sorted(globals().items())
         if name.startswith("test_")],
        suite="test_threadranks",
    )


if __name__ == "__main__":
    raise SystemExit(1 if _main() else 0)
