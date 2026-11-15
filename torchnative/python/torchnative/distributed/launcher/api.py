"""Programmatic launcher: `LaunchConfig` and `elastic_launch` (BACKEND.md D7, S6.11).

Shaped like `torch.distributed.launcher.api` so an app that already calls
`elastic_launch(config, entrypoint)(*args)` keeps its call. The name is
upstream's; the behaviour is not elastic. This is a small, self-contained
process launcher: one `multiprocessing` spawn-context process per local rank,
each result carried back through its own pipe. Nothing of upstream's elastic
agent is imported.

What is refused, by name (BACKEND.md §8 and D5):

* `min_nodes != max_nodes` and `max_restarts > 0`: fault-tolerant relaunch and
  elastic membership are out of scope.
* Start methods other than `"spawn"` and `"thread"`.

`start_method="thread"` (BACKEND.md D5, §9 steps 6 and 14, issue #281) runs
each local rank as a **thread of this process** instead (an agent proposal,
not confirmed, for the API shape: upstream's field, a new value). The world
is made per thread first (`torchnative.distributed.threads.install`), the
launcher hosts the Store when it is node 0, and each rank thread's `env://`
is bound to its rank (`threads.bind_env`), so an entrypoint that calls
`dist.init_process_group(backend="torchnative")` with no arguments works
unchanged. `os.environ` is one per process, so in thread mode the launcher
sets **no** variables; `rank_info()` returns what they would have held. The
entrypoint need not be picklable and its return value is passed back as is.
If a rank thread raises, the launcher aborts every rank thread's groups (which
closes their sockets, so a peer blocked in a collective with the failed rank
raises instead of waiting out a timeout), waits `THREAD_STOP_GRACE` seconds,
and raises `ChildFailedError` naming the rank and, separately, every rank
thread that had still not returned (a thread cannot be killed; a rank blocked
outside `torch.distributed` keeps running as a daemon thread).

Option behaviour beyond torchrun's (`node_rank` on the config, the refusal
wording, the termination grace period, everything about thread mode above)
is an agent proposal, not confirmed.
"""

import dataclasses
import multiprocessing
import multiprocessing.connection
import os
import queue
import signal
import socket
import sys
import threading
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

#: What torchrun uses when no endpoint is given.
DEFAULT_MASTER_ADDR = "127.0.0.1"
DEFAULT_MASTER_PORT = 29500

#: Seconds a child gets between SIGTERM and SIGKILL.
TERMINATE_GRACE = 5.0

#: Thread mode: seconds the other rank threads get to return after the
#: launcher aborted their groups because one rank failed. Agent proposal.
THREAD_STOP_GRACE = 10.0

#: The start methods `LaunchConfig` accepts.
START_METHODS = ("spawn", "thread")


class LauncherRefused(NotImplementedError):
    """A launcher feature that is out of scope, named in the message."""


class ChildFailedError(RuntimeError):
    """One or more ranks failed. `failures` maps global rank to a description.

    In thread mode `unstopped` lists the global ranks whose threads had still
    not returned `THREAD_STOP_GRACE` seconds after the launcher aborted their
    groups; it is empty for processes, which are killed.
    """

    def __init__(self, failures: Dict[int, str], unstopped=()):
        self.failures = dict(failures)
        self.unstopped = sorted(unstopped)
        lines = ["rank %d failed: %s" % (r, m) for r, m in sorted(failures.items())]
        if self.unstopped:
            lines.append(
                "rank thread(s) %s did not return within %.0f s after their "
                "groups were aborted; a thread cannot be killed, so they keep "
                "running as daemon threads"
                % (", ".join(str(r) for r in self.unstopped), THREAD_STOP_GRACE))
        super().__init__("; ".join(lines))


def refuse_elastic(min_nodes: int, max_nodes: int, max_restarts: int) -> None:
    if min_nodes != max_nodes:
        raise LauncherRefused(
            "torchnative launcher: elastic membership (min_nodes=%d != max_nodes=%d) "
            "is out of scope (docs/distributed/BACKEND.md §8); pass one fixed node count"
            % (min_nodes, max_nodes))
    if max_restarts != 0:
        raise LauncherRefused(
            "torchnative launcher: max_restarts=%d needs fault-tolerant relaunch "
            "(elastic), which is out of scope (docs/distributed/BACKEND.md §8); "
            "only max_restarts=0 is supported" % max_restarts)


def parse_endpoint(endpoint: str) -> Tuple[str, int]:
    """`host:port` (or `[v6]:port`) to (host, port)."""
    host, sep, port = endpoint.rpartition(":")
    if not sep or not host or not port.isdigit():
        raise ValueError("rendezvous endpoint %r is not host:port" % endpoint)
    return host.strip("[]"), int(port)


def free_port(host: str = "127.0.0.1") -> int:
    """An ephemeral port, bound and released here (the caller races for it)."""
    sock = socket.socket()
    try:
        sock.bind((host, 0))
        return sock.getsockname()[1]
    finally:
        sock.close()


def rank_env(*, node_rank: int, local_rank: int, nnodes: int, nproc_per_node: int,
             master_addr: str, master_port: int) -> Dict[str, str]:
    """The variables a rank sees: those torchrun sets for `env://`, no elastic ones."""
    return {
        "RANK": str(node_rank * nproc_per_node + local_rank),
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(nnodes * nproc_per_node),
        "LOCAL_WORLD_SIZE": str(nproc_per_node),
        "GROUP_RANK": str(node_rank),
        "MASTER_ADDR": master_addr,
        "MASTER_PORT": str(master_port),
    }


def terminate_all(procs, grace: float = TERMINATE_GRACE) -> None:
    """SIGTERM then SIGKILL every still-running `multiprocessing.Process`."""
    live = [p for p in procs if p.is_alive()]
    for p in live:
        try:
            p.terminate()
        except OSError:
            pass
    end = time.monotonic() + grace
    for p in live:
        p.join(max(0.0, end - time.monotonic()))
    for p in live:
        if p.is_alive():
            p.kill()
            p.join()


@dataclasses.dataclass
class LaunchConfig:
    """Mirrors the fields of upstream's `LaunchConfig` that mean something here."""

    min_nodes: int = 1
    max_nodes: int = 1
    nproc_per_node: int = 1
    run_id: str = ""
    rdzv_endpoint: str = ""
    max_restarts: int = 0
    start_method: str = "spawn"
    node_rank: int = 0

    def __post_init__(self):
        refuse_elastic(self.min_nodes, self.max_nodes, self.max_restarts)
        if self.start_method not in START_METHODS:
            raise LauncherRefused(
                "torchnative launcher: start_method=%r is not supported; only "
                "'spawn' (one process per rank) and 'thread' (ranks as threads "
                "of this process, BACKEND.md D5)" % (self.start_method,))
        if self.nproc_per_node < 1 or self.max_nodes < 1:
            raise ValueError("nproc_per_node and nodes must be at least 1")
        if not 0 <= self.node_rank < self.max_nodes:
            raise ValueError("node_rank %d outside [0, %d)" % (self.node_rank, self.max_nodes))
        if self.max_nodes > 1 and not self.rdzv_endpoint:
            raise ValueError("more than one node needs rdzv_endpoint host:port")


def _worker(entrypoint, args, env, conn):
    os.environ.update(env)
    try:
        value = entrypoint(*args)
        try:
            conn.send(("ok", value))
        except Exception as error:  # noqa: BLE001  unpicklable result
            conn.send(("err", "result not picklable: %r" % (error,)))
    except BaseException:  # noqa: BLE001
        conn.send(("err", traceback.format_exc()))
        conn.close()
        raise SystemExit(1)
    conn.close()


class elastic_launch:  # noqa: N801  upstream's spelling
    """`elastic_launch(config, entrypoint)(*args)` -> `{local_rank: return value}`."""

    def __init__(self, config: LaunchConfig, entrypoint: Callable[..., Any]):
        if not callable(entrypoint):
            raise LauncherRefused(
                "torchnative launcher: entrypoint must be a picklable callable; a command "
                "string is run by `python -m torchnative.distributed.run`")
        self._config = config
        self._entrypoint = entrypoint

    def __call__(self, *args) -> Dict[int, Any]:
        cfg = self._config
        if cfg.start_method == "thread":
            return _launch_threads(cfg, self._entrypoint, args)
        if cfg.rdzv_endpoint:
            host, port = parse_endpoint(cfg.rdzv_endpoint)
        else:
            host, port = DEFAULT_MASTER_ADDR, free_port()
        ctx = multiprocessing.get_context("spawn")
        procs: List[Any] = []
        conns: Dict[Any, int] = {}
        results: Dict[int, Any] = {}
        failures: Dict[int, str] = {}
        base = cfg.node_rank * cfg.nproc_per_node
        try:
            for local in range(cfg.nproc_per_node):
                env = rank_env(node_rank=cfg.node_rank, local_rank=local,
                               nnodes=cfg.max_nodes, nproc_per_node=cfg.nproc_per_node,
                               master_addr=host, master_port=port)
                parent, child = ctx.Pipe(duplex=False)
                p = ctx.Process(target=_worker, args=(self._entrypoint, args, env, child))
                p.start()
                child.close()
                procs.append(p)
                conns[parent] = local
            alive = {p.sentinel: (i, p) for i, p in enumerate(procs)}
            while conns or alive:
                ready = multiprocessing.connection.wait(list(conns) + list(alive))
                for obj in ready:
                    if obj in conns:
                        local = conns[obj]
                        try:
                            kind, payload = obj.recv()
                        except EOFError:
                            del conns[obj]
                            continue
                        if kind == "ok":
                            results[local] = payload
                        else:
                            failures[base + local] = payload
                    elif obj in alive:
                        local, p = alive.pop(obj)
                        p.join()
                        if p.exitcode != 0 and (base + local) not in failures and local not in results:
                            failures[base + local] = "exited with code %s" % p.exitcode
                if failures:
                    break
            if failures:
                terminate_all(procs)
                raise ChildFailedError(failures)
            return {local: results[local] for local in sorted(results)}
        finally:
            terminate_all(procs)
            for c in conns:
                c.close()




# -- thread mode (BACKEND.md D5, issue #281) ---------------------------------------

_rank_tls = threading.local()


def rank_info() -> Optional[Dict[str, str]]:
    """In a rank thread, the variables process mode would have set; else None.

    Thread mode sets no environment variables (one `os.environ` per process),
    so an entrypoint that wants its `LOCAL_RANK` reads it here, or calls
    `dist.get_rank()` after `init_process_group`.
    """
    info = getattr(_rank_tls, "info", None)
    return None if info is None else dict(info)


def _thread_store(cfg, host, port, world_size):
    """The one Store every local rank thread shares, made before any thread starts.

    Node 0 hosts it (a `TCPStore` master on the rendezvous endpoint), so a
    rank in another process or on another machine can join the same world;
    no rank thread races to bind it. Other nodes are clients of node 0's.
    A world of one has no peer to reach, and the shim's `TCPStore` refuses
    that, so it gets a `HashStore`.
    """
    import datetime
    from torch.distributed import HashStore, TCPStore
    if world_size == 1:
        return HashStore()
    return TCPStore(host, port, world_size, cfg.node_rank == 0,
                    timeout=datetime.timedelta(minutes=30))


def _abort_world(world):
    for pg in list(getattr(world, "pg_map", {}) or {}):
        try:
            pg.abort()
        except Exception:  # noqa: BLE001  best effort: waking peers
            pass


def _launch_threads(cfg, entrypoint, args) -> Dict[int, Any]:
    import torch.distributed as dist
    from torchnative.distributed import threads
    import torchnative.distributed  # noqa: F401  registers backend="torchnative"

    threads.install()
    if cfg.rdzv_endpoint:
        host, port = parse_endpoint(cfg.rdzv_endpoint)
    else:
        host, port = DEFAULT_MASTER_ADDR, free_port()
    world_size = cfg.max_nodes * cfg.nproc_per_node
    base = cfg.node_rank * cfg.nproc_per_node
    store = _thread_store(cfg, host, port, world_size)

    done: "queue.Queue" = queue.Queue()
    worlds: Dict[int, Any] = {}
    lock = threading.Lock()
    failed = threading.Event()  # set by the first rank thread that raises

    def rank_main(local):
        rank = base + local
        info = rank_env(node_rank=cfg.node_rank, local_rank=local,
                        nnodes=cfg.max_nodes, nproc_per_node=cfg.nproc_per_node,
                        master_addr=host, master_port=port)
        _rank_tls.info = info
        outcome = None
        try:
            threads.bind_env(rank, world_size, store, local_rank=local)
            with lock:
                worlds[rank] = threads.current_world()
            outcome = ("ok", entrypoint(*args), False)
        except BaseException:  # noqa: BLE001  reported, never swallowed
            text = traceback.format_exc()
            with lock:
                # The first rank to raise is the failure; a rank that raises
                # after it is (most likely) its consequence, a peer whose
                # collective lost that rank, and is reported as such.
                first = not failed.is_set()
                failed.set()
            outcome = ("err", text, first)
        finally:
            try:
                if dist.is_initialized():
                    dist.destroy_process_group()
            except Exception as error:  # noqa: BLE001
                if outcome is None or outcome[0] == "ok":
                    with lock:
                        first = not failed.is_set()
                        failed.set()
                    outcome = ("err", "destroy_process_group failed: %r"
                               % (error,), first)
            threads.unbind_env()
            _rank_tls.info = None
            done.put((rank, outcome))

    def abort_all():
        with lock:
            for world in worlds.values():
                _abort_world(world)

    hook = sys.excepthook  # init_process_group wraps it once per rank thread
    results: Dict[int, Any] = {}
    failures: Dict[int, str] = {}
    later: Dict[int, str] = {}
    pending = set(range(base, base + cfg.nproc_per_node))
    clean = False
    try:
        for local in range(cfg.nproc_per_node):
            threading.Thread(target=rank_main, args=(local,), daemon=True,
                             name="torchnative-rank-%d" % (base + local)).start()
        deadline = None
        while pending:
            timeout = None if deadline is None else max(0.0, deadline - time.monotonic())
            try:
                rank, (kind, payload, first) = done.get(timeout=timeout)
            except queue.Empty:
                break  # the grace ran out; what is still pending did not stop
            pending.discard(rank)
            if kind == "ok":
                results[rank - base] = payload
            elif first:
                failures[rank] = payload
            else:
                later[rank] = payload
            if failures and deadline is None:
                # Wake every rank thread blocked in a collective with the
                # failed rank: aborting a group closes its sockets.
                abort_all()
                deadline = time.monotonic() + THREAD_STOP_GRACE
        if failures:
            error = ChildFailedError(failures, unstopped=pending)
            error.after_failure = dict(later)
            raise error
        clean = True
        return {local: results[local] for local in sorted(results)}
    except BaseException:
        if not clean:
            abort_all()
        raise
    finally:
        sys.excepthook = hook
