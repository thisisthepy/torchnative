# P2P: the binary wire, the transports and point-to-point on the `torchnative` backend

Issue #219. BACKEND.md §9 steps 2 and 4. Written 2026-11-13.

**Grade: builds.** Every file below passed `python -m py_compile` and nothing else. No suite, gate or probe
ran in the round that wrote it (AGENTS.md rule 10: agents do not run suites; the gate runs in CI). Nothing here
is *reaches* or *agrees* until CI runs `tests/distributed/test_p2p.py` (AGENTS.md §16). Read every "works" below
as "is written to".

Provenance. The transport split and per-pair selection are decided (owner, 2026-11-08, BACKEND.md D3: "전송은
일단 니가 하자는대로 구현하고"); Unix socket first for the same host is D9 (leader); Python first is D8 (leader);
loopback by default is D10 (leader). Everything else in this document is an **agent proposal, not confirmed**:
the frame layout, the dtype codes, the environment variable names, the store keys, the greeting, the full-mesh
topology, the class and module names.

## 1. What was built

| Piece | Where | What it does |
|---|---|---|
| Binary frame | `torchnative/python/torchnative/distributed/backend/wire.py` | 24-byte header (magic `TNP2`, version, kind, dtype code, ndim, signed 64-bit tag, payload length), then the shape, then raw little-endian bytes. No JSON |
| Transports | `.../backend/transport.py` | `InprocConnection` (queues), `SocketConnection` over `AF_UNIX` (`shm`) or TCP (`tcp`, `TCP_NODELAY`); listeners `InprocTransport`, `UnixTransport`, `TcpTransport` |
| Selection | `transport.choose` | per pair: same process `inproc`, same host `shm`, else `tcp`; `TORCHNATIVE_DIST_TRANSPORT` forces one kind where the pair can have it and refuses by name where it cannot |
| Mesh | `transport.Mesh` | one connection per pair, a reader thread per connection demultiplexing into per-`(src, tag)` FIFO queues, a sender thread per connection |
| Backend | `.../backend/__init__.py` `ProcessGroupTorchnative` | subclasses `ProcessGroupLocal` (collectives unchanged), adds `send`/`recv`, keeps `recv_anysource` refused, `transport_of(peer)` for tests |
| Registration | `torchnative/python/torchnative/distributed/__init__.py` | the `torchnative` creator now returns `ProcessGroupTorchnative` and passes the group timeout |

<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/backend/wire.py read_frame present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/backend/wire.py encode_tensor present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/backend/transport.py Mesh present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/backend/transport.py choose present -->
<!-- DOCWATCH: symbol-in-file torchnative/python/torchnative/distributed/backend/__init__.py ProcessGroupTorchnative present -->

## 2. The frame

```
offset  size  field
0       4     magic            b"TNP2"
4       1     version          1
5       1     kind             1 tensor, 2 object bytes, 3 control
6       1     dtype code       0 for a frame that is not a tensor
7       1     ndim             0..255
8       8     tag              signed 64-bit
16      8     payload length   unsigned 64-bit, bytes
24      8*n   shape            n = ndim extents, unsigned 64-bit
...           payload          raw little-endian element bytes, row-major
```

Dtypes with a code: `float32`, `float64`, `float16`, `bfloat16`, `int8`, `int16`, `int32`, `int64`, `uint8`,
`bool`. Anything else (complex, `uint32`, `float8_e4m3fn`) refuses by name. Upstream's `isend`/`irecv` turn a
complex tensor into its real view before the backend sees it, so complex never reaches the wire through
`torch.distributed`.

A decoder error names the peer rank: a stream that ends inside a frame is `WireError ... truncated`; a bad magic
or version is `WireError ... magic` / `... version`; a stream that ends between frames is `PeerGone ... connection
closed`.

### 2.1 Bytes out of a tensor

Step 0 of BACKEND.md had left "a cheap exact bytes export" unmeasured. Reading the shim found one:
`TensorBase._shim_numpy_parts` (tensor.rs), the hook `Tensor.numpy()` uses, returns the address of the CPU
element buffer, and `ctypes.string_at` copies `numel * itemsize` bytes from it. That is the **fast** path, one
copy and no per-element Python. `bfloat16` is refused by that hook, so it goes out through `float32` (exact: a
bfloat16 value is a float32 with its low 16 bits zero) and the high half of each word is taken by slicing. The
**slow** path, `struct.pack` of `tolist()`, runs only if the hook is missing or the layout is unusual; it is exact
for finite values and infinities but does not preserve NaN payload bits. `wire.export_path(t)` says which path a
tensor takes. Into a tensor: `torch.frombuffer`. Neither path has been timed (D8: Rust only where measured slow).

## 3. Transports and selection

Each rank publishes, under store keys prefixed `tn_p2p/` (distinct from `ProcessGroupLocal`'s `pg_local_port`), a
record of its host id, a per-process token (a random id plus the pid, so a forked child differs), its Unix socket
path and its TCP address. Rank 0 also publishes a per-group token; the in-process registry is keyed by it, so two
groups in one process do not meet.

| Pair | Natural kind | Forcible kinds |
|---|---|---|
| two threads of one process | `inproc` | `inproc`, `shm`, `tcp` |
| two processes of one host | `shm` | `shm`, `tcp` |
| two hosts | `tcp` (needs a non-loopback bind on both) | `tcp` |

`TORCHNATIVE_DIST_TRANSPORT=inproc|shm|tcp` forces a kind; a kind the pair cannot have refuses by name (for
example `inproc` between two processes) rather than falling back. `ws` (D3) is not built and refuses by name.

The `shm` socket lives in `tempfile.gettempdir()` under a unique name and is unlinked as soon as the mesh is
built. The `tcp` listener binds `TORCHNATIVE_DIST_BIND_ADDR` (or the `bind_addr` option) and otherwise
`127.0.0.1`, the loopback default of D10. The brief for this round suggested deriving the bind from `MASTER_ADDR`;
that was not done, because `MASTER_ADDR` names the master's address and binding it on any other host fails. Two
hosts that both listen on loopback refuse by name, naming the variable to set.

The host id is the host name plus `/etc/machine-id` where that file exists (Linux). On macOS it is the host name
alone, so two machines with one name would be treated as one host and their `shm` connect would fail (it names
the peer and the path).

## 4. Point-to-point semantics

* `send(tensors, dst, tag)` encodes the frame on the caller's thread (a snapshot, so reusing the tensor after
  `isend` returns cannot change what is sent) and queues it for that peer's sender thread. The `Work` completes
  when the frame has been written. `dist.isend` returns it; `dist.send` is upstream's `isend(...).wait()`.
* `recv(tensors, src, tag)` posts a receive matched by `(src, tag)` in posting order. The matched frame is
  written into the caller's tensor at `wait()` or at an `is_completed()` that returns True, the ownership rule
  of `AsyncWork` (ASYNCWORK.md §3). A dtype or element-count disagreement refuses naming both sides and writes
  nothing.
* Different tags never block each other: a sender may send tag 2 then tag 1, and the receiver may take tag 1
  first.
* `batch_isend_irecv` takes upstream's non-coalescing branch (`supports_coalescing` is False), which issues each
  op as an `isend`/`irecv`. The coalescing hooks are not needed for it.
* `send_object_list`/`recv_object_list` are upstream functions that pickle into a `uint8` tensor and call
  `send`/`recv`; they reach this backend only through those two methods.
* A self-send (`isend` to one's own rank, which upstream allows) is delivered into the rank's own queue.
* `recv_anysource` refuses by name: it has no consumer (BACKEND.md §4.4).
* A peer whose connection closes fails every receive waiting on it, and every later receive from it, with a
  message naming the rank and containing "connection closed". A receive with no matching send waits for the
  group timeout (`init_process_group(timeout=...)`), then raises naming the peer, the tag and the seconds.
* `ProcessGroup.shutdown`/`abort` now forward to the registered backends (a two-method edit in `bootstrap.py`,
  section 6), so `destroy_process_group` flushes queued sends and closes the mesh. All threads are daemons.

The collectives are untouched: they still run on `ProcessGroupLocal`'s star and JSON wire (step 3 moves them).

## 5. Tests written (not run)

`tests/distributed/test_p2p.py`, registered in `.github/scripts/gate_suites.py` for both runners:

<!-- DOCWATCH: symbol-in-file tests/distributed/test_p2p.py test_every_dtype_round_trips_bit_for_bit_against_struct_built_bytes present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_p2p.py test_a_flipped_header_byte_is_a_named_error_not_a_garbled_tensor present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_p2p.py test_point_to_point_agrees_with_gloo_bit_for_bit_at_world_two_and_four present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_p2p.py test_recv_anysource_still_refuses_by_name present -->

| Test | Asserts |
|---|---|
| `test_every_dtype_round_trips_bit_for_bit_against_struct_built_bytes` | ten dtypes; frame payload, original export and decoded export each equal bytes built by `struct` alone |
| `test_a_non_contiguous_an_empty_and_a_zero_dim_tensor_round_trip` | a transposed view, a `(0, 3)` tensor and a 0-d tensor |
| `test_a_frame_split_into_one_byte_reads_decodes_whole` | a fake socket returning one byte per `recv` through the real `SocketConnection` |
| `test_a_truncated_frame_raises_naming_the_peer` | truncation in header, shape and payload; a clean EOF is `PeerGone` |
| `test_a_flipped_header_byte_is_a_named_error_not_a_garbled_tensor` | the sabotage: magic and version flips raise named errors; the unflipped frame decodes |
| `test_a_dtype_without_a_wire_code_refuses_by_name` | `complex64` |
| `test_a_loopback_pair_works_over_each_transport_and_reports_which` | two threads: default `inproc`, forced `shm`, forced `tcp`; values both ways; the socket path is unlinked |
| `test_two_groups_in_one_process_do_not_meet` | two concurrent in-process groups each get their own payload |
| `test_an_override_that_names_no_built_transport_refuses` | `ws` and a bogus name |
| `test_point_to_point_agrees_with_gloo_bit_for_bit_at_world_two_and_four` | sync ring, isend ring over ten dtypes, 0<->2 and 1<->3 at world 4, tags out of order, isend/irecv, batch ring: equal to gloo |
| `test_what_arrived_is_what_the_peer_sent_not_only_what_gloo_got` | the same values against the sender's generator, so gloo is not the only oracle |
| `test_tags_match_even_when_the_sender_sent_them_in_the_other_order` | tag 1 is taken first and holds tag 1's tensor |
| `test_irecv_is_genuinely_pending_until_the_sender_sends_and_owns_its_buffer` | incomplete before the sender wakes, on both backends; the shim's buffer is untouched before `wait()` |
| `test_recv_anysource_still_refuses_by_name` | the one p2p refusal kept |
| `test_a_dtype_disagreement_refuses_naming_both_and_writes_nothing` | int64 sent, float32 posted |
| `test_a_peer_that_vanishes_is_named_by_the_recv_waiting_on_it` | the highest rank `os._exit`s; survivors' receive names it within 60 s (timeout 120 s) |
| `test_two_processes_on_one_host_default_to_shm` | `transport_of` at world 2 and 4 |
| `test_forcing_tcp_changes_the_transport_used_and_not_the_values` | the override reaches `transport_of`, and the values still equal gloo's |
| `test_forcing_inproc_across_processes_refuses_by_name` | `init_process_group` refuses, naming the variable |

`test_collect2.py::test_point_to_point_still_refuses_by_name_and_was_not_weakened` is retired, with its
probes: an unmatched `recv` would now wait for the timeout rather than refuse. Its surviving claim
(`recv_anysource` refuses) is in the new suite. The federated probe in `tests/_support/test_shim.py` that
asserted `send` refuses now asserts `recv_anysource` refuses.

## 6. Known gaps and risks, read off source, not measured

1. **Object lists are not tested and not claimed (issue #229).** Upstream's `_object_to_tensor` calls
   `torch.ByteStorage._from_buffer`, and no `from_buffer` was found in the shim's Rust storage class or in
   `bootstrap.py`, so the sender would fail before sending and the receiver would wait out the group timeout.
   The probe was removed rather than left to stall every later probe in its world; the fix is in the storage
   layer, not here.
2. **Receiving into a complex tensor.** Upstream `irecv` receives into `torch.view_as_real(t)`. In this shim a
   complex tensor is two real buffers, so that view is likely a copy and the caller's tensor would not be written.
   Untested; refused nowhere yet.
3. `TCPStore.get` blocks with no timeout, so a rank that never publishes its record holds the others in
   rendezvous until `ProcessGroupLocal`'s star rendezvous (300 s) fails first.
4. NaN payload bits survive only the fast export path (section 2.1).
5. `bootstrap.py` changed (`ProcessGroup.shutdown`/`abort` forward to backends), so the extension must be rebuilt
   before this is measured (AGENTS.md §15.2).
