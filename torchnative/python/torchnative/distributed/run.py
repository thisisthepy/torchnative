"""`python -m torchnative.distributed.run`: a torchrun-shaped launcher (D7, S6.11).

::

    uv run python -m torchnative.distributed.run --nproc-per-node 3 script.py args...
    uv run python -m torchnative.distributed.run --standalone --nproc-per-node 2 -m pkg.mod

Starts `nproc_per_node` child processes with the variables `init_process_group`
reads for `env://` (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `LOCAL_WORLD_SIZE`,
`GROUP_RANK`, `MASTER_ADDR`, `MASTER_PORT`) and waits for them. It does not
pretend to be elastic: no `TORCHELASTIC_*` variables, no restarts. When any
rank exits non-zero, the children this launcher started are terminated and it
exits non-zero, naming the failing rank(s) and exit codes. SIGINT and SIGTERM
are forwarded to the children it started.

Refused by name (docs/distributed/BACKEND.md §8): `--nnodes a:b` with a != b
and `--max-restarts` above 0, both elastic.

The master address is passed through unchanged. A non-loopback address is not
secured here: authentication is BACKEND.md D10 (step 5), not this launcher.
The default endpoint is 127.0.0.1:29500 as in torchrun; `--standalone` picks a
free loopback port instead. Behaviour beyond torchrun's is an agent proposal,
not confirmed.
"""

import argparse
import os
import signal
import subprocess
import sys
import time
from typing import List, Optional

from torchnative.distributed.launcher.api import (
    DEFAULT_MASTER_ADDR, DEFAULT_MASTER_PORT, TERMINATE_GRACE, LauncherRefused,
    free_port, parse_endpoint, rank_env, refuse_elastic)

_PROG = "torchnative.distributed.run"


def _nnodes(text: str) -> int:
    lo, sep, hi = text.partition(":")
    try:
        low = int(lo)
        high = int(hi) if sep else low
    except ValueError:
        raise argparse.ArgumentTypeError("--nnodes takes an integer, got %r" % text)
    refuse_elastic(low, high, 0)  # raises LauncherRefused for a:b
    return low


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m " + _PROG, add_help=True)
    p.add_argument("--nnodes", type=_nnodes, default=1)
    p.add_argument("--nproc-per-node", "--nproc_per_node", dest="nproc_per_node",
                   type=int, default=1)
    p.add_argument("--node-rank", "--node_rank", dest="node_rank", type=int, default=0)
    p.add_argument("--master-addr", "--master_addr", dest="master_addr", default=None)
    p.add_argument("--master-port", "--master_port", dest="master_port", type=int,
                   default=None)
    p.add_argument("--rdzv-endpoint", "--rdzv_endpoint", dest="rdzv_endpoint", default=None)
    p.add_argument("--standalone", action="store_true")
    p.add_argument("--max-restarts", "--max_restarts", dest="max_restarts", type=int,
                   default=0)
    p.add_argument("-m", "--module", dest="module", action="store_true",
                   help="treat the target as a module name, like python -m")
    p.add_argument("training_script")
    p.add_argument("training_script_args", nargs=argparse.REMAINDER)
    return p


def resolve_endpoint(ns) -> tuple:
    if ns.standalone:
        return "127.0.0.1", free_port()
    if ns.rdzv_endpoint:
        return parse_endpoint(ns.rdzv_endpoint)
    host = ns.master_addr or DEFAULT_MASTER_ADDR
    port = ns.master_port if ns.master_port is not None else DEFAULT_MASTER_PORT
    return host, port


def _signal_group(proc, sig) -> None:
    try:
        if hasattr(os, "killpg"):
            os.killpg(proc.pid, sig)
        else:
            proc.send_signal(sig)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _stop(procs, grace: float = TERMINATE_GRACE) -> None:
    for p in procs:
        if p.poll() is None:
            _signal_group(p, signal.SIGTERM)
    end = time.monotonic() + grace
    for p in procs:
        try:
            p.wait(max(0.0, end - time.monotonic()))
        except subprocess.TimeoutExpired:
            pass
    for p in procs:
        if p.poll() is None:
            _signal_group(p, signal.SIGKILL)
            p.wait()


def run(ns) -> int:
    refuse_elastic(ns.nnodes, ns.nnodes, ns.max_restarts)
    if ns.nproc_per_node < 1 or ns.nnodes < 1:
        raise SystemExit("%s: nnodes and nproc-per-node must be at least 1" % _PROG)
    if not 0 <= ns.node_rank < ns.nnodes:
        raise SystemExit("%s: --node-rank %d outside [0, %d)" % (_PROG, ns.node_rank, ns.nnodes))
    if ns.nnodes > 1 and not (ns.rdzv_endpoint or ns.master_addr):
        raise SystemExit("%s: more than one node needs --rdzv-endpoint or --master-addr" % _PROG)
    host, port = resolve_endpoint(ns)

    cmd = [sys.executable, "-u"]
    cmd += ["-m", ns.training_script] if ns.module else [ns.training_script]
    cmd += ns.training_script_args

    procs: List[subprocess.Popen] = []
    got: List[int] = []
    old = {}

    def on_signal(signum, _frame):
        got.append(signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        old[sig] = signal.signal(sig, on_signal)
    try:
        for local in range(ns.nproc_per_node):
            env = dict(os.environ)
            env.update(rank_env(node_rank=ns.node_rank, local_rank=local,
                                nnodes=ns.nnodes, nproc_per_node=ns.nproc_per_node,
                                master_addr=host, master_port=port))
            procs.append(subprocess.Popen(cmd, env=env, start_new_session=True))
        base = ns.node_rank * ns.nproc_per_node
        failed = {}
        while True:
            if got:
                _stop(procs)
                sys.stderr.write("%s: received signal %d, stopped %d rank(s)\n"
                                 % (_PROG, got[0], len(procs)))
                return 128 + got[0]
            codes = [p.poll() for p in procs]
            failed = {base + i: c for i, c in enumerate(codes) if c not in (None, 0)}
            if failed or all(c == 0 for c in codes):
                break
            time.sleep(0.05)
        if failed:
            _stop(procs)
            pids = ", ".join("rank %d pid %d" % (base + i, p.pid) for i, p in enumerate(procs))
            sys.stderr.write(
                "%s: failed: %s; remaining ranks terminated (children: %s)\n"
                % (_PROG, ", ".join("rank %d exit code %d" % (r, c)
                                    for r, c in sorted(failed.items())), pids))
            return 1
        return 0
    finally:
        _stop(procs)
        for sig, handler in old.items():
            signal.signal(sig, handler)


def main(argv: Optional[List[str]] = None) -> int:
    try:
        ns = build_parser().parse_args(argv)
        return run(ns)
    except LauncherRefused as refusal:
        sys.stderr.write("%s: refused: %s\n" % (_PROG, refusal))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
