"""The launcher (BACKEND.md step 14, D7, SPEC S6.11), against torchrun's contract and gloo.

`python -m torchnative.distributed.run` starts a world of 3 that runs real
collectives over `backend="torchnative"` with `env://`; the same child script
runs under upstream torch with gloo (the oracle, no PYTHONPATH, never the shim)
and the answers are compared. Refusals are by name: elastic (`--nnodes a:b`,
`--max-restarts 1`) and `start_method="thread"` (D5, needs the inproc transport,
issue #219).

Grade: these are written to be run by CI; nothing here was run when written.

Every test says in its docstring what would make it fail.
"""

import gloo_loopback
import launcher_entries
import json
import os
import signal
import subprocess
import sys
import tempfile
import time

from test_collect2 import _c2_free_port
from test_shim import _CKPT_VENDOR_DIR

_WORLD = 3

_ENV_KEYS = ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE",
             "GROUP_RANK", "MASTER_ADDR", "MASTER_PORT")

# The child. TN_BACKEND picks the side: "torchnative" runs on the shim and
# asserts it is the shim; "gloo" asserts it is upstream. Integers only, so
# every collective is exact and the comparison is bit equality.
_CHILD = r'''
import json, os
import torch
if os.environ["TN_BACKEND"] == "torchnative":
    assert hasattr(torch._C, "_aten_implemented"), "upstream torch, not the shim"
    import torchnative.distributed  # noqa: F401
else:
    assert not hasattr(torch._C, "_aten_implemented"), "the shim, not upstream"
import torch.distributed as dist

dist.init_process_group(backend=os.environ["TN_BACKEND"])  # env://
rank, world = dist.get_rank(), dist.get_world_size()
ar = torch.tensor([float(rank + 1), float(rank * 2), 0.5])
dist.all_reduce(ar)
ag = [torch.zeros(2) for _ in range(world)]
dist.all_gather(ag, torch.tensor([float(rank), float(rank * rank)]))
bc = torch.tensor([float(rank + 10), 7.0]) if rank == 1 else torch.zeros(2)
dist.broadcast(bc, src=1)
out = {"env": {k: os.environ.get(k) for k in %r},
       "all_reduce": ar.tolist(), "all_gather": [t.tolist() for t in ag],
       "broadcast": bc.tolist(), "backend": dist.get_backend()}
with open(os.path.join(os.environ["OUT_DIR"], "rank-%%d.json" %% rank), "w") as f:
    json.dump(out, f)
dist.destroy_process_group()
''' % (_ENV_KEYS,)

_FAIL_CHILD = r'''
import os, sys, time
rank = int(os.environ["RANK"])
with open(os.path.join(os.environ["OUT_DIR"], "pid-%d" % rank), "w") as f:
    f.write(str(os.getpid()))
if rank == 1:
    # Exit only once the peers have written their pids, so the test can read
    # every pid on a slow machine (a fixed sleep raced them).
    out = os.environ["OUT_DIR"]
    deadline = time.monotonic() + 50
    while time.monotonic() < deadline and not all(
            os.path.exists(os.path.join(out, "pid-%d" % r)) and
            os.path.getsize(os.path.join(out, "pid-%d" % r)) for r in (0, 2)):
        time.sleep(0.05)
    sys.exit(7)
time.sleep(120)   # the peers: would outlive the test if the launcher did not stop them
'''


def _launcher_env(out_dir, backend="torchnative"):
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    env["OUT_DIR"] = out_dir
    env["TN_BACKEND"] = backend
    return env


def _run_cli(args, env, timeout=300):
    return subprocess.run([sys.executable, "-m", "torchnative.distributed.run"] + args,
                          env=env, capture_output=True, text=True, timeout=timeout)


def _read_ranks(out_dir, world=_WORLD):
    reports = []
    for rank in range(world):
        with open(os.path.join(out_dir, "rank-%d.json" % rank)) as f:
            reports.append(json.load(f))
    return reports


def _oracle(script, out_dir, world=_WORLD, timeout=300):
    """The same script, one process per rank, upstream torch + gloo, env set by hand."""
    port = _c2_free_port()
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)          # the oracle, never the shim
    env["OUT_DIR"] = out_dir
    env["TN_BACKEND"] = "gloo"
    src = gloo_loopback.child_pin_source() + open(script).read()
    procs = []
    try:
        for rank in range(world):
            e = dict(env, RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(world),
                     LOCAL_WORLD_SIZE=str(world), GROUP_RANK="0",
                     MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
            procs.append(subprocess.Popen([sys.executable, "-c", src], env=e,
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE))
        deadline = time.monotonic() + timeout
        for rank, p in enumerate(procs):
            _, err = p.communicate(timeout=max(1.0, deadline - time.monotonic()))
            assert p.returncode == 0, "oracle rank %d exited %d: %s" % (
                rank, p.returncode, err.decode(errors="replace")[-2000:])
    finally:
        for p in procs:
            if p.poll() is None:
                p.kill()
                p.wait()


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def test_the_cli_runs_a_world_of_three_that_agrees_with_gloo():
    """Runs the child under the CLI (torchnative) and by hand (gloo); compares.

    Fails if the CLI starts fewer or more than 3 ranks (a missing rank-N.json),
    if the children do not rendezvous over env://, if any collective differs
    from gloo's, or if the world used a backend other than torchnative (the
    reported backend is asserted, and the child asserts it loaded the shim).
    """
    with tempfile.TemporaryDirectory(prefix="launcher-") as tmp:
        script = os.path.join(tmp, "child.py")
        with open(script, "w") as f:
            f.write(_CHILD)
        tn_dir, gl_dir = os.path.join(tmp, "tn"), os.path.join(tmp, "gloo")
        os.mkdir(tn_dir)
        os.mkdir(gl_dir)
        res = _run_cli(["--nproc-per-node", str(_WORLD), "--standalone", script],
                       _launcher_env(tn_dir))
        assert res.returncode == 0, (res.returncode, res.stdout, res.stderr)
        _oracle(script, gl_dir)
        tn, gl = _read_ranks(tn_dir), _read_ranks(gl_dir)
        for rank in range(_WORLD):
            assert tn[rank]["backend"] == "torchnative", tn[rank]["backend"]
            for key in ("all_reduce", "all_gather", "broadcast"):
                assert tn[rank][key] == gl[rank][key], (rank, key, tn[rank][key], gl[rank][key])
        # The expected values, so "both wrong the same way" cannot pass.
        assert tn[0]["all_reduce"] == [6.0, 6.0, 1.5], tn[0]["all_reduce"]
        assert tn[2]["broadcast"] == [11.0, 7.0], tn[2]["broadcast"]


def test_each_rank_sees_exactly_the_environment_torchrun_sets():
    """Fails if any of the seven variables is unset, mis-numbered, or if
    `--master-addr/--master-port` are not passed through (the explicit port is
    checked), or if TORCHELASTIC_RUN_ID appears (it would claim elasticity)."""
    script_src = ("import os, json\n"
                  "keys = %r\n"
                  "d = {k: os.environ.get(k) for k in keys}\n"
                  "d['elastic'] = os.environ.get('TORCHELASTIC_RUN_ID')\n"
                  "open(os.path.join(os.environ['OUT_DIR'], 'rank-' + os.environ['RANK'] + '.json'), 'w').write(json.dumps(d))\n"
                  % (_ENV_KEYS,))
    with tempfile.TemporaryDirectory(prefix="launcher-env-") as tmp:
        script = os.path.join(tmp, "envdump.py")
        with open(script, "w") as f:
            f.write(script_src)
        port = _c2_free_port()
        env = dict(os.environ)
        env.pop("TORCHELASTIC_RUN_ID", None)
        env.update(OUT_DIR=tmp, PYTHONPATH=_CKPT_VENDOR_DIR, TORCH_USE_RTLD_GLOBAL="1")
        res = _run_cli(["--nproc_per_node", "3", "--master-addr", "127.0.0.1",
                        "--master-port", str(port), script], env)
        assert res.returncode == 0, (res.stdout, res.stderr)
        for rank, seen in enumerate(_read_ranks(tmp)):
            assert seen == {"RANK": str(rank), "LOCAL_RANK": str(rank), "WORLD_SIZE": "3",
                            "LOCAL_WORLD_SIZE": "3", "GROUP_RANK": "0",
                            "MASTER_ADDR": "127.0.0.1", "MASTER_PORT": str(port),
                            "elastic": None}, (rank, seen)


def test_a_failing_rank_fails_the_launcher_names_the_rank_and_leaves_no_child():
    """Rank 1 exits 7 while ranks 0 and 2 would sleep 120 s. Fails if the
    launcher exits 0, does not name `rank 1` and `exit code 7`, waits out the
    sleepers (the elapsed-time bound), or leaves any reported pid alive."""
    with tempfile.TemporaryDirectory(prefix="launcher-fail-") as tmp:
        script = os.path.join(tmp, "fail.py")
        with open(script, "w") as f:
            f.write(_FAIL_CHILD)
        env = dict(os.environ, OUT_DIR=tmp, PYTHONPATH=_CKPT_VENDOR_DIR,
                   TORCH_USE_RTLD_GLOBAL="1")
        started = time.monotonic()
        res = _run_cli(["--nproc-per-node", "3", "--standalone", script], env, timeout=100)
        elapsed = time.monotonic() - started
        assert res.returncode != 0, (res.stdout, res.stderr)
        assert elapsed < 60, elapsed
        assert "rank 1 exit code 7" in res.stderr, res.stderr
        pids = [int(open(os.path.join(tmp, "pid-%d" % r)).read()) for r in range(3)]
        assert all(str("pid %d" % p) in res.stderr for p in pids), (pids, res.stderr)
        deadline = time.monotonic() + 10
        while any(_alive(p) for p in pids) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not any(_alive(p) for p in pids), [p for p in pids if _alive(p)]


def test_sigterm_to_the_launcher_stops_the_children_it_started():
    """Fails if SIGTERM is not forwarded: the sleeping ranks would stay alive
    (checked by pid) or the launcher would not exit within the bound."""
    with tempfile.TemporaryDirectory(prefix="launcher-sig-") as tmp:
        script = os.path.join(tmp, "sleeper.py")
        with open(script, "w") as f:
            f.write("import os, time\n"
                    "open(os.path.join(os.environ['OUT_DIR'], 'pid-' + os.environ['RANK']), 'w')"
                    ".write(str(os.getpid()))\ntime.sleep(120)\n")
        env = dict(os.environ, OUT_DIR=tmp, PYTHONPATH=_CKPT_VENDOR_DIR,
                   TORCH_USE_RTLD_GLOBAL="1")
        proc = subprocess.Popen([sys.executable, "-m", "torchnative.distributed.run",
                                 "--nproc-per-node", "2", "--standalone", script],
                                env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            files = [os.path.join(tmp, "pid-%d" % r) for r in range(2)]
            deadline = time.monotonic() + 60
            while (not all(os.path.exists(p) and os.path.getsize(p) for p in files)
                   and time.monotonic() < deadline):
                time.sleep(0.1)
            pids = [int(open(p).read()) for p in files]
            proc.send_signal(signal.SIGTERM)
            proc.wait(timeout=30)
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        assert proc.returncode == 128 + signal.SIGTERM, proc.returncode
        deadline = time.monotonic() + 10
        while any(_alive(p) for p in pids) and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not any(_alive(p) for p in pids), pids


def test_elastic_options_refuse_by_name():
    """`--max-restarts 1` and `--nnodes 1:2` must exit 2 naming `elastic`; a
    launcher that accepted them (and ran the script, which would create the
    marker file) fails here."""
    with tempfile.TemporaryDirectory(prefix="launcher-refuse-") as tmp:
        marker = os.path.join(tmp, "ran")
        script = os.path.join(tmp, "s.py")
        with open(script, "w") as f:
            f.write("open(%r, 'w').write('x')\n" % marker)
        env = dict(os.environ, PYTHONPATH=_CKPT_VENDOR_DIR, TORCH_USE_RTLD_GLOBAL="1")
        for opts, needle in ((["--max-restarts", "1"], "max_restarts=1"),
                             (["--nnodes", "1:2"], "min_nodes=1 != max_nodes=2")):
            res = _run_cli(opts + [script], env, timeout=100)
            assert res.returncode == 2, (opts, res.returncode, res.stderr)
            assert "refused" in res.stderr and "elastic" in res.stderr, res.stderr
            assert needle in res.stderr, (needle, res.stderr)
            assert not os.path.exists(marker), "the script ran despite the refusal"


def test_elastic_launch_returns_each_ranks_return_value():
    """Fails if a rank is missing from the dict, a value is another rank's, or
    the env the entrypoint sees is not its own."""
    sys.path.insert(0, _CKPT_VENDOR_DIR)
    from torchnative.distributed.launcher import LaunchConfig, elastic_launch
    cfg = LaunchConfig(min_nodes=1, max_nodes=1, nproc_per_node=3, run_id="t",
                       rdzv_endpoint="127.0.0.1:%d" % _c2_free_port())
    out = elastic_launch(cfg, launcher_entries.entry)(10)
    assert sorted(out) == [0, 1, 2], out
    for r in range(3):
        assert out[r] == {"rank": r, "local": r, "world": 3, "value": 10 * r}, out


def test_elastic_launch_raises_naming_the_failed_rank_and_stops_the_rest():
    """Fails if a child exception is swallowed, the rank is not named, or the
    sleeping peer is waited out (the bound)."""
    sys.path.insert(0, _CKPT_VENDOR_DIR)
    from torchnative.distributed.launcher import (ChildFailedError, LaunchConfig,
                                                  elastic_launch)
    cfg = LaunchConfig(nproc_per_node=2)
    started = time.monotonic()
    try:
        elastic_launch(cfg, launcher_entries.entry_fails)(None)
    except ChildFailedError as error:
        assert list(error.failures) == [1], error.failures
        assert "boom from rank 1" in error.failures[1], error.failures
    else:
        raise AssertionError("a failing rank did not raise")
    assert time.monotonic() - started < 45


def test_the_api_refuses_elastic_and_thread_ranks_by_name():
    """Fails if `LaunchConfig` accepts min!=max nodes, max_restarts>0, or
    start_method='thread' (which must cite D5 and issue #219), or if the
    refusal is any other type than the named LauncherRefused."""
    sys.path.insert(0, _CKPT_VENDOR_DIR)
    from torchnative.distributed.launcher import LaunchConfig, LauncherRefused
    cases = ((dict(min_nodes=1, max_nodes=2, rdzv_endpoint="h:1"), ("elastic", "min_nodes=1")),
             (dict(max_restarts=1), ("elastic", "max_restarts=1")),
             (dict(start_method="thread"), ("D5", "#219", "thread")))
    for kwargs, needles in cases:
        try:
            LaunchConfig(**kwargs)
        except LauncherRefused as error:
            text = str(error)
            assert all(n in text for n in needles), (kwargs, text)
        else:
            raise AssertionError("accepted %r" % (kwargs,))


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
