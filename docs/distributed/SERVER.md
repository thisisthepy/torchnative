# SERVER: the swappable coordinator, the LAN opt-in and the pre-shared key

Issue #233. BACKEND.md §9 step 5, D4 and D10. Written 2026-11-13.

**Grade: builds.** Every file below passed `python -m py_compile` and nothing else. No suite, gate or probe
ran in the round that wrote it (AGENTS.md rule 10). Nothing here is *reaches* or *agrees* until CI runs
`tests/distributed/test_server.py` (AGENTS.md §16). Read every "works" below as "is written to". A run across two
machines is a device test for the coordinating session (BACKEND.md §9 step 5) and is not claimed.

Provenance. Decided: the server is swappable behind an interface, with an in-process server and a standalone
Python server process among its implementations (owner, 2026-11-08, BACKEND.md D4: "서버 구현체를 갈아낄 수 있도록
설계해줘"); loopback by default plus an opt-in pre-shared key with stdlib `hmac` (leader, D10). Everything else in
this document is an **agent proposal, not confirmed**: the class names, the `tnserver://` scheme and its `server=`
query, the store protocol, the variable names `TORCHNATIVE_DIST_KEY` and `TORCHNATIVE_DIST_ALLOW_LAN`, the
arrival barrier and its store keys, and the handshake's message layout.

## 1. What was built

| Piece | Where | What it does |
|---|---|---|
| Interface | `torchnative/python/torchnative/distributed/backend/server.py` `Server` | `start()`, `stop()`, `address()`, `url()`, `make_store(rank, world_size, timeout)`; a context manager |
| In-process server | `server.py` `InProcessServer` | the store served from daemon threads of the calling process |
| Standalone server | `server.py` `StandaloneServer`, `main` | the same server in its own process: `python -m torchnative.distributed.backend.server --host H --port P`, or the file run directly. Its top-level imports are the standard library only, so it needs no torch |
| Client | `server.py` `connect_store` | a `torch._C._distributed_c10d.Store` subclass; `get` and `wait` block on the server and time out at the store's timeout |
| Rendezvous | `server.py` `register_rendezvous_handler`, called from `torchnative/distributed/__init__.py` | `tnserver://` through upstream's `register_rendezvous_handler`; `env://`, `tcp://`, `file://` are untouched |
| Arrival barrier | `server.py` `rendezvous_barrier`, called first in `ProcessGroupTorchnative.__init__` | names a rank that never arrives, at the group's timeout |
| Bind rule | `server.py` `check_bind` | a non-loopback bind refuses before any socket exists, unless `TORCHNATIVE_DIST_ALLOW_LAN=1` |
| Key handshake | `transport.py` `_connect_handshake`, `_accept_handshake`; `server.py` `proof`, `resolve_key` | HMAC-SHA-256 challenge-response on every mesh connection when a key is set |
| `TCPStore` timeouts | `torchnative/rust/torch_c/src/bootstrap.py` `TCPStore` | `get` and `wait` honour the store's timeout instead of blocking forever |

<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/backend/server.py InProcessServer present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/backend/server.py StandaloneServer present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/backend/server.py check_bind present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/backend/server.py rendezvous_barrier present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/backend/transport.py _accept_handshake present -->

## 2. Using it

```python
import datetime
import torch.distributed as dist
import torchnative.distributed  # registers backend="torchnative" and tnserver://

# Rank 0 hosts an InProcessServer on 127.0.0.1:29500; every rank connects to it.
dist.init_process_group(backend="torchnative", init_method="tnserver://127.0.0.1:29500",
                        rank=rank, world_size=3, timeout=datetime.timedelta(seconds=120))
```

```sh
# A standalone server process; no rank hosts anything.
uv run python -m torchnative.distributed.backend.server --host 127.0.0.1 --port 29500
# each rank: init_method="tnserver://127.0.0.1:29500?server=external"
```

`server=rank0` is the default. `server=external` reaches any server already running, which is also what
`Server.url()` returns for an `InProcessServer` or a `StandaloneServer` the caller started.

The PythonMultiplatform app-host server and a remote server (D4) are not built. Federated server-side aggregation
through this interface (D14) is a follow-up and is not built.

## 3. Timeouts

The client's `get` and `wait` send the store's timeout to the server, which waits on a condition until the key
exists or the time runs out, then answers "timeout" and the client raises naming the missing keys and the seconds.
`init_process_group` sets the store's timeout to the group's (`store.set_timeout(timeout)`), so the store, the
arrival barrier and the mesh all wait the group's timeout. The shim's `TCPStore` had `get` and `wait` with no
timeout at all (P2P.md §6 item 3); its server and client got the same rule in `bootstrap.py` (a short edit to
`get`, `wait` and the constructor; the extension must be rebuilt, AGENTS.md §15.2).

`ProcessGroupLocal`'s star still waits a fixed 300 s in `accept` and 30 s per socket. The arrival barrier runs
before it, so a rank that never arrives is named before the star can wait on it. Moving the collectives onto the
mesh (BACKEND.md §9 step 3) removes the star.

## 4. Security (D10)

* **Loopback by default.** The mesh's TCP listener, the in-process server and the standalone server bind
  `127.0.0.1` unless given an address. A non-loopback address (`0.0.0.0` included; any host name other than
  `localhost` counts as non-loopback, because the rule reads what was asked for) refuses with `BindRefused` naming
  `TORCHNATIVE_DIST_ALLOW_LAN=1`, before the socket is created.
* **Opt-in pre-shared key.** With `TORCHNATIVE_DIST_KEY` (or `key=`) set, each mesh connection runs:

  ```
  C (lower rank) -> A   hello      token, auth, nonce_c
  A -> C                challenge  nonce_a, HMAC(key, "accept"  | token | C | A | nonce_c)
  C -> A                proof      HMAC(key, "connect" | token | C | A | nonce_a)
  A -> C                ok
  ```

  Nonces are 32 bytes from `secrets`; digests are compared with `hmac.compare_digest`. Each side proves the key
  over the other's fresh nonce, so a recorded exchange does not verify again, and the label differs per direction
  so a proof cannot be reflected. Without a key on either side the exchange is `hello`, `accept`. A wrong key, or a
  key on one side only, ends in `refuse` and raises `AuthRefused` on both sides, each naming the peer rank. An empty
  key refuses by name.
* **The key never leaves the process.** Only nonces and digests go on the wire; nothing about the key goes into the
  store.
* **This authenticates; it does not encrypt.** Every frame after the handshake is plaintext (BACKEND.md §8).
* **Not covered**, read off source:
  1. The store is not authenticated. It holds the group token and listening addresses, none of them secret; a
     forged address leads a rank to a listener that cannot pass the handshake. Someone who reaches the store can
     still disrupt rendezvous.
  2. The collectives' star (`ProcessGroupLocal`) binds and dials `127.0.0.1` only, so it never leaves the host,
     and it is not covered by the key. Collectives therefore do not yet run across machines at all; step 3 moves
     them onto the authenticated mesh.
  3. A connection that greets with the right group token and a bad key aborts the build on the rank it reached.
     That is the refusal the tests ask for, and it also means a party on the network who knows the token can stop
     a rendezvous.

## 5. Tests written (not run)

`tests/distributed/test_server.py`, registered in `.github/scripts/gate_suites.py` for both runners. Each test
states in its docstring what sabotage it catches.

<!-- DOCWATCH: symbol-in-file tests/distributed/test_server.py test_the_same_world_agrees_with_gloo_through_a_standalone_server_process present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_server.py test_a_wrong_key_refuses_by_name_on_both_sides present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_server.py test_a_rank_that_never_arrives_is_named_within_the_timeout present -->

| Test | Asserts |
|---|---|
| `test_the_handshake_passes_with_the_same_key_and_with_no_key` | both halves in one process; four frames with a key, two without |
| `test_a_wrong_key_is_refused_by_name_by_both_halves_in_process` | `AuthRefused` on each half, naming the other rank |
| `test_a_key_on_one_side_only_is_refused_by_name_by_both_halves_in_process` | the key on the connector only, then on the acceptor only |
| `test_the_key_never_crosses_the_wire_and_a_replayed_proof_does_not_verify` | every byte sent is searched for the key, raw and hex; a recorded hello and proof replayed to a fresh acceptor is refused |
| `test_a_lan_bind_without_the_opt_in_refuses_before_anything_binds` | `203.0.113.7` (TEST-NET-3): `BindRefused` without the opt-in and nothing written to the store; an `OSError` from `bind` with it, which proves the check ran first; the same for both servers |
| `test_an_empty_key_refuses_by_name` | `TORCHNATIVE_DIST_KEY=""` |
| `test_a_world_of_three_agrees_with_gloo_through_the_in_process_server` | ring `send`/`recv` and `all_reduce` (float32 and int64) equal gloo's; the hosted store holds every rank's arrival and address |
| `test_the_same_world_agrees_with_gloo_through_a_standalone_server_process` | the same, through a server process the test started; its key list holds every rank's records and its pid is no rank's |
| `test_with_the_right_key_the_world_works_and_every_rank_authenticated` | values equal gloo's, every mesh reports `authenticated`, the key is in no store value |
| `test_a_wrong_key_refuses_by_name_on_both_sides` | world 2 with two keys: each rank's `init_process_group` raises naming the other |
| `test_a_key_on_one_side_only_refuses_by_name_on_both_sides` | world 3, only rank 1 holds a key: ranks 0 and 2 name rank 1, rank 1 names rank 0 or 2 |
| `test_a_rank_that_never_arrives_is_named_within_the_timeout` | rank 2 never calls `init_process_group`; ranks 0 and 1 raise `RendezvousTimeout` naming it, between 0.5 and 4 times the 10 s timeout, through `tnserver://` and through `tcp://` (the `TCPStore` fix) |
| `test_every_rank_really_ran_the_shim_and_the_oracle_really_ran_upstream` | the shim on one side, gloo on the other, `shm` between the shim's ranks |

The float inputs are multiples of 2^-6 in [-2, 2), so a sum over three ranks is exact in float32 in any order:
equality with gloo is the right comparison, and the star's fold order versus gloo's cannot differ on them.

## 6. Known gaps

1. Everything in section 4's "not covered".
2. The store protocol is one request per TCP connection (length-prefixed JSON). Rendezvous makes a few dozen
   requests per rank; it has not been timed.
3. `queue_push`/`queue_pop`/`queue_len` refuse by name on the server's store; nothing in this backend uses them.
4. The arrival barrier names the first missing rank it finds and says the ranks after it were not checked.
5. The backend constructor's ordering edit (barrier before the star) lives in `backend/__init__.py`, which the
   step 3 round also edits; the merge must keep the barrier ahead of any rendezvous.
