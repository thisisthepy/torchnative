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
* `start_method="thread"`: ranks as threads of the app process (D5) need the
  `inproc` transport (issue #219) first. It refuses rather than quietly
  running processes.

Option behaviour beyond torchrun's (`node_rank` on the config, the refusal
wording, the termination grace period) is an agent proposal, not confirmed.
"""

import dataclasses
import multiprocessing
import multiprocessing.connection
import os
import signal
import socket
import time
import traceback
from typing import Any, Callable, Dict, List, Optional, Tuple

#: What torchrun uses when no endpoint is given.
DEFAULT_MASTER_ADDR = "127.0.0.1"
DEFAULT_MASTER_PORT = 29500

#: Seconds a child gets between SIGTERM and SIGKILL.
TERMINATE_GRACE = 5.0


class LauncherRefused(NotImplementedError):
    """A launcher feature that is out of scope, named in the message."""


class ChildFailedError(RuntimeError):
    """One or more ranks failed. `failures` maps global rank to a description."""

    def __init__(self, failures: Dict[int, str]):
        self.failures = dict(failures)
        lines = ["rank %d failed: %s" % (r, m) for r, m in sorted(failures.items())]
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
        if self.start_method == "thread":
            raise LauncherRefused(
                "torchnative launcher: start_method='thread' (ranks as threads of the "
                "app process, BACKEND.md D5) needs the inproc transport, issue #219, "
                "which is not built; use start_method='spawn'")
        if self.start_method != "spawn":
            raise LauncherRefused(
                "torchnative launcher: start_method=%r is not supported; only 'spawn'"
                % (self.start_method,))
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


