# THREADRANKS: ranks as threads of one process

Issue #281. BACKEND.md D5, §5.1, §9 step 6, and the thread half of step 14. Written 2026-11-13.

**Grade: builds.** Every file below passed `python3 -m py_compile` and nothing else. No suite, gate or probe ran
in the round that wrote it (AGENTS.md rule 10: agents do not run suites; CI runs them). `bootstrap.py` changed, so
the extension must be rebuilt before any of it is measured (AGENTS.md §15.2). Read every "works" below as "is
written to".

Provenance. That a rank can be a thread, with several ranks per app on free-threaded 3.15t, `inproc` between them
and `tcp` to other machines, is D5 (leader, 2026-11-08, from the owner's transport and server answers). That the
programmatic launcher starts rank threads is D7 and step 14 (leader). Everything else in this document is an
**agent proposal, not confirmed**: the thread-local world as the mechanism, the module and function names, the
`env://` binding, `start_method="thread"` as the launcher's spelling, the failure and grace rules.

## 1. The problem

`torch.distributed` keeps the default group and every group table in one process-global object,
`distributed_c10d._world`. Its `_World` properties even read module globals (`global _pg_map`), so a second
`_World()` shares them. N threads that each call `init_process_group` therefore collide. Read off source, not
measured, the collision has three forms:

1. one thread finds `GroupMember.WORLD` set and refuses ("trying to initialize the default process group twice!");
2. both start before either finishes; the shared `group_count` names their groups `"0"` and `"1"`, their Store
   prefixes differ, and the star rendezvous never meets (`TCPStore.get` has no timeout);
3. both finish, the last `_update_default_pg` wins, and both threads read one group: wrong ranks, silently.

The shim's group registry (`torch._C._distributed_c10d._register_process_group`) and its global rank are
process-global too, and with a per-thread world every rank's default group is named `"0"`, so the registry would
hand three of four threads another thread's group, and one thread's `destroy_process_group` (which calls
`_unregister_all_process_groups`) would clear everyone's.

## 2. The decision: a thread-local world (agent proposal, not confirmed)

| Option | `dist.all_reduce(t)` in a rank thread | Cost |
|---|---|---|
| **Thread-local world** (chosen) | works unchanged: the default group is the thread's | replaces `distributed_c10d._world`, which upstream documents as an "experimental extension point to override it"; needs the registry per thread too |
| Explicit per-thread groups | does not: with no `group=` it reads the one process-global default group | every call site passes `group=`; `transformers` tensor parallelism reads the default group, so unmodified upstream code would not run |

The chosen form is upstream's own, from its threaded test harness: `multi_threaded_pg.py`'s `ThreadLocalWorld`, and
`common_distributed.py`'s `_set_thread_isolation_mode(True)` around threaded tests.

<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/threads.py install present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/threads.py _ThreadLocalWorld present -->
<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/src/bootstrap.py _set_thread_isolation_mode present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/launcher/api.py _launch_threads present -->

| Piece | Where | What it does |
|---|---|---|
| Thread-local world | `torchnative/python/torchnative/distributed/threads.py` `install()` | replaces `_world` with `_ThreadLocalWorld`, whose every attribute is the calling thread's `_RankWorld` (a `_World` subclass with its own tables). Idempotent. |
| Thread isolation | `bootstrap.py`, `_set_thread_isolation_mode` / `_get_thread_isolation_mode` | upstream's names (`GroupRegistry.hpp`). While on, the group registry and the global rank are per thread. Off, the per-process dict is used exactly as before. `install()` turns it on. |
| `env://` per thread | `threads.bind_env` | a rank thread's `env://` answers from a binding (rank, world size, Store) instead of `os.environ`, which is one per process. A thread with no binding goes to upstream's handler unchanged. |
| Star closed on shutdown | `backend/__init__.py` `ProcessGroupTorchnative.shutdown`/`abort` | also close the collectives' star sockets (`shutdown(SHUT_RDWR)` then `close`). A process closes them by exiting; a thread does not exit. |
| Thread launcher | `launcher/api.py`, `LaunchConfig(start_method="thread")` | one thread per local rank; see §4. |

Nothing chooses a transport. Two rank threads publish the same process token, so `transport.choose` picks
`inproc` for them; a rank in another process of the host gets `shm`, another host `tcp`. The `inproc` registry is
keyed by (group token, rank), and the token is per group, so two worlds in one process do not meet.

**Collectives still run on the star over loopback TCP**, between threads too. `inproc` carries point-to-point
only until BACKEND.md §9 step 3 moves the collectives onto the mesh. `transport_of` reports the mesh's kind, which
is the p2p transport.

## 3. Refused or named, not supported

| What | What happens |
|---|---|
| `install()` while the process world holds a default group | `ThreadRanksRefused` naming the rank and size: per-thread now would hide that group from the thread that made it |
| `install()` after something else replaced `_world` (for example upstream's `ThreadLocalWorld`) | `ThreadRanksRefused` naming the type: two overrides of one extension point cannot both hold |
| `install()` on an extension built before this change | `ThreadRanksRefused` naming `_set_thread_isolation_mode` (the old build's placeholder refuses) |
| `uninstall()` while a thread still holds a default group | `ThreadRanksRefused` naming the threads |
| A thread started *by* a rank thread | gets an empty world of its own; a collective there fails with upstream's "Default process group has not been initialized" |
| `init_process_group(rank=k)` in a launcher rank thread bound to another rank | `ThreadRanksRefused` naming both ranks |
| Reading `os.environ["RANK"]` in thread mode | not set; `launcher.rank_info()` returns what it would hold |
| Parallel ranks on a GIL build | values are right, ranks take turns; the agreement tests say `UNVERIFIED` by name |

## 4. The thread launcher (agent proposal, not confirmed)

`elastic_launch(LaunchConfig(nproc_per_node=4, start_method="thread"), entry)(*args)` returns
`{local_rank: entry's return value}`. In order:

1. `threads.install()`.
2. The launcher makes the one Store every local rank shares, **before** any thread starts: on node 0 a `TCPStore`
   master on the rendezvous endpoint (so a rank in another process or host can join the same world, and no rank
   thread races to bind it), on other nodes a client of it, at world 1 a `HashStore` (the shim's `TCPStore` refuses
   world 1).
3. One daemon thread per local rank, named `torchnative-rank-<global rank>`. Each binds `env://` to its rank and
   calls the entrypoint, so `dist.init_process_group(backend="torchnative")` with no arguments works. No
   environment variable is written. When the entrypoint returns or raises, the thread destroys any group it still
   holds.
4. The first rank thread to raise is the failure. The launcher then aborts every rank thread's groups (closing
   their sockets, so a peer blocked in a collective with the failed rank raises "connection closed" at once rather
   than at the 30 s socket timeout), waits `THREAD_STOP_GRACE` (10 s), and raises `ChildFailedError` with
   `failures` holding the first failure, `after_failure` the ranks that raised after it, and `unstopped` the
   rank threads still running. A thread cannot be killed, so a rank blocked outside `torch.distributed` keeps
   running as a daemon thread, and the error says so.
5. `sys.excepthook` is restored (every `init_process_group` wraps it once).

The entrypoint need not be picklable; its return value is passed back as is. Elastic options refuse as in process
mode.

## 5. Tests written (not run)

`tests/distributed/test_threadranks.py`, registered in `.github/scripts/gate_suites.py` for both runners. The
collective and p2p probes are `test_collect2.py`'s and `test_p2p.py`'s sources, run unchanged inside the threads;
the oracles are those suites' cached gloo runs in separate processes.

<!-- DOCWATCH: symbol-in-file tests/distributed/test_threadranks.py test_four_rank_threads_of_one_process_agree_with_gloo present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_threadranks.py test_without_the_thread_local_world_two_rank_threads_collide present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_threadranks.py test_a_mixed_world_of_two_threads_and_one_process_agrees_with_gloo present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_threadranks.py test_a_failing_rank_thread_is_named_and_its_peers_are_woken present -->

| Test | Asserts |
|---|---|
| `test_four_rank_threads_of_one_process_agree_with_gloo` | 24 collective probes at the derived tolerance and 17 p2p probes bit for bit, 4 threads vs 4 gloo processes; then `UNVERIFIED` on a GIL interpreter |
| `test_four_rank_threads_on_free_threaded_python_run_without_the_gil_and_agree` | the same world under `TORCHNATIVE_FT_PYTHON` with `TORCHNATIVE_FT_STAGE` (the issue #51 inputs of `tests/api/test_freethreaded.py`): free-threaded build, GIL still off after importing the shim, values equal gloo's. SKIP by name without the inputs. This is the only test that can show step 6's claim |
| `test_four_rank_threads_give_the_bits_four_processes_give` | the same 24 collectives equal the shim's own 4-process world bit for bit (rank-order fold) |
| `test_rank_threads_pair_over_inproc_as_the_backend_reports` | `transport_of` is `inproc` for all 12 directed pairs |
| `test_each_rank_thread_has_its_own_default_group_and_registry` | own rank, isolation on, the registry resolves each thread's own group, one name for four groups, the first thread's destroy leaves the others intact |
| `test_destroying_every_rank_thread_leaves_no_transport_thread_or_listener` | no `torchnative-p2p*` or async thread alive, no `inproc` listener registered |
| `test_without_the_thread_local_world_two_rank_threads_collide` | the nullification kept in the suite: without `install()` the threads collide in one of the three forms of §1 |
| `test_forcing_shm_between_rank_threads_changes_the_reported_transport` | the override reaches `transport_of`, so `inproc` above is the backend's choice, not a constant |
| `test_a_mixed_world_of_two_threads_and_one_process_agrees_with_gloo` | ranks 0, 1 threads of one process and rank 2 another process, world 3 vs gloo; `UNVERIFIED` on a GIL interpreter |
| `test_a_mixed_world_pairs_threads_over_inproc_and_the_process_over_shm` | 0-1 `inproc`, 0-2 and 1-2 `shm` |
| `test_the_thread_launcher_starts_rank_threads_and_returns_each_result` | world 4 then world 2 in one process; per-rank values, `inproc`, thread names, `rank_info()`, environment untouched |
| `test_a_failing_rank_thread_is_named_and_its_peers_are_woken` | rank 1 raises, peers blocked in a barrier are woken in under 25 s, only rank 1 is the failure, no thread left |
| `test_a_rank_thread_that_cannot_be_woken_is_named_as_not_stopped` | rank 0 asleep outside `torch.distributed` is named in `unstopped` |

`test_launcher.py::test_the_api_refuses_elastic_and_thread_ranks_by_name` became
`test_the_api_refuses_elastic_and_unknown_start_methods_by_name`: `"thread"` is accepted, `"forkserver"` refuses.

### 5.1 Nullifications planned for CI (not performed: no build ran here)

| Mutant | Test that should go red |
|---|---|
| `install()` skipped | kept in the suite as `test_without_the_thread_local_world_two_rank_threads_collide`'s control, and every thread-world test would fail at init |
| `transport.choose`: `same_process = False` | `test_rank_threads_pair_over_inproc_as_the_backend_reports`, the mixed-world pairs test, and the launcher test (`shm` instead of `inproc`) |
| `bootstrap.py` `_groups()` always returns the per-process dict | `test_each_rank_thread_has_its_own_default_group_and_registry` (`resolves_own` False for three of four threads) |
| `ProcessGroupTorchnative.abort` without `_close_star` | `test_a_rank_thread_that_cannot_be_woken_is_named_as_not_stopped`: rank 2, blocked on a sleeping hub, waits out the 30 s socket timeout, so the launcher returns late or names rank 2 unstopped |

## 6. Known gaps and risks, read off source, not measured

1. **Free-threaded 3.15t has not been run for this.** The gate's wheel is abi3, which cannot load on 3.15t; issue
   #51 builds a second, non-abi3 extension that declares `gil_used = false` (`lib.rs`), and only the test fed
   that build (`TORCHNATIVE_FT_PYTHON`, `TORCHNATIVE_FT_STAGE`) can show parallel ranks. The tests read
   `sys._is_gil_enabled()` in the child after importing the shim, because an extension that did not declare itself
   safe re-enables the GIL at import. ASYNCWORK.md §4's single-writer assumptions inside `ProcessGroupLocal` are per
   group, and each rank thread has its own group; shared shim state outside `torch.distributed` is what
   `tests/api/test_freethreaded.py`'s hammer covers, and it was not re-read for this round.
2. Upstream's `_install_threaded_pg` also turns off autograd's multithreading; this does not, because the shim's
   autograd engine is not upstream's C++ engine. Unexamined.
3. The shim's `TCPStore` has no close: the launcher's master Store keeps its port and server thread for the life of
   the process, one per launch.
4. Collectives between rank threads still cross loopback TCP (§2).
5. A rank thread whose entrypoint never returns keeps its groups and sockets; `uninstall()` then refuses.
