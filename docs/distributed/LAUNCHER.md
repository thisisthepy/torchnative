# Launcher: what was built for BACKEND.md step 14

Grade: **builds**. The code was written and byte-compiled; nothing was run. CI runs
`tests/distributed/test_launcher.py` (SPEC S6.11, issue #220). Design source: BACKEND.md D7, §5.5, §8.

## What exists

| Piece | File | Behaviour |
|---|---|---|
| CLI | `torchnative/python/torchnative/distributed/run.py` | `uv run python -m torchnative.distributed.run [opts] script.py args...`, or `-m module`. Starts `nproc_per_node` child processes with `RANK`, `LOCAL_RANK`, `WORLD_SIZE`, `LOCAL_WORLD_SIZE`, `GROUP_RANK`, `MASTER_ADDR`, `MASTER_PORT`. |
| API | `torchnative/python/torchnative/distributed/launcher/api.py` | `LaunchConfig` and `elastic_launch(config, entrypoint)(*args)` returning `{local_rank: value}`; ranks are `multiprocessing` spawn processes, results come back over one pipe per rank. |
| API, thread mode | same file, `LaunchConfig(start_method="thread")` | one thread per local rank in this process (issue #281, D5); design, failure rules and tests in `docs/distributed/THREADRANKS.md` §4. Graded builds. |

Options: `--nnodes` (int), `--nproc-per-node`/`--nproc_per_node`, `--node-rank`, `--master-addr`, `--master-port`,
`--rdzv-endpoint host:port`, `--standalone` (a free loopback port), `--max-restarts` (0 only), `-m`.

## Behaviour

- If a rank exits non-zero, the launcher terminates the children it started (SIGTERM, then SIGKILL after 5 s), prints the
  failing rank(s), exit codes and every child pid, and exits 1. The API raises `ChildFailedError` with the same facts.
- SIGINT and SIGTERM to the CLI are forwarded to the children it started; the exit code is 128 plus the signal.
- The default endpoint is `127.0.0.1:29500`, as in torchrun. A non-loopback master address is passed through
  unsecured: authentication is D10 (step 5).

## Refused by name (BACKEND.md §8, D5)

- `--nnodes a:b` with a != b, `min_nodes != max_nodes`: elastic membership.
- `--max-restarts` above 0, `max_restarts > 0`: fault-tolerant relaunch.
- Start methods other than `spawn` and `thread`. (`thread` used to refuse here, pending the `inproc` transport of
  issue #219; issue #281 built it.)
- A non-callable `entrypoint`: a command string belongs to the CLI.

## Not built

Web Worker ranks (step 13, D5), multi-node rendezvous beyond passing the endpoint through,
`TORCHELASTIC_*` variables, `launch_agent`.

## Provenance

The existence and location of the launcher is D7 (leader, owner left the location open). Option behaviour beyond
torchrun's (the `node_rank` field on `LaunchConfig`, the refusal wording, the 5 s termination grace, process-group
kill of children) is an agent proposal, not confirmed.
