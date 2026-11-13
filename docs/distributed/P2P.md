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
  `send`/`recv`; they reach this backend only through those two methods. Their storage path is section 7
  (issue #229).
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
| `test_object_lists_agree_with_gloo_at_world_two_and_four` | preflight, `send_object_list`/`recv_object_list` between non-neighbours, a list of one, `broadcast_object_list`: equal to gloo (section 7) |
| `test_object_lists_hold_what_the_sender_pickled` | the same against the sender's own objects |
| `test_all_gather_object_and_gather_object_refuse_naming_resize_` | both refuse on every rank naming `resize_`; gloo's answers asserted too |
| `test_every_object_rank_really_ran_its_backend` | the object worlds' `_meta` |

`test_collect2.py::test_point_to_point_still_refuses_by_name_and_was_not_weakened` is retired, with its
probes: an unmatched `recv` would now wait for the timeout rather than refuse. Its surviving claim
(`recv_anysource` refuses) is in the new suite. The federated probe in `tests/_support/test_shim.py` that
asserted `send` refuses now asserts `recv_anysource` refuses.

## 6. Known gaps and risks, read off source, not measured

1. **Object lists are written to work and are graded *builds* (issue #229, section 7).** The storage path
   they lacked is built; `all_gather_object`, `gather_object` and `scatter_object_list` still stop at
   `Tensor.resize_`, refused by name. `scatter_object_list` refuses on its source rank only, so its other
   ranks wait out the group timeout.
2. **Receiving into a complex tensor.** Upstream `irecv` receives into `torch.view_as_real(t)`. In this shim a
   complex tensor is two real buffers, so that view is likely a copy and the caller's tensor would not be written.
   Untested; refused nowhere yet.
3. `TCPStore.get` blocks with no timeout, so a rank that never publishes its record holds the others in
   rendezvous until `ProcessGroupLocal`'s star rendezvous (300 s) fails first.
4. NaN payload bits survive only the fast export path (section 2.1).
5. `bootstrap.py` changed (`ProcessGroup.shutdown`/`abort` forward to backends), so the extension must be rebuilt
   before this is measured (AGENTS.md §15.2).

## 7. Object lists (issue #229)

Written 2026-11-13. **Grade: builds.** `python -m py_compile` passed on every file named here; no suite ran
(AGENTS.md rule 10). The from_buffer argument logic was also run, extracted from `bootstrap.py` with a stand-in
storage class, against upstream's `UntypedStorage.from_buffer` on 46 cases and agreed on all 46. That is a check
of the Python logic alone, not of the shim: nothing here is *reaches* or *agrees* until CI runs
`tests/bindings/test_storagebuf.py` and `tests/distributed/test_p2p.py`. Everything below is an **agent
proposal, not confirmed**.

The calls upstream makes, and what answers each:

| Upstream call (`distributed_c10d.py`) | Answered by |
|---|---|
| `torch.ByteStorage._from_buffer(pickled)` | the vendored `TypedStorage._from_buffer`, unchanged, which calls the next row |
| `torch.UntypedStorage.from_buffer(buf, dtype=torch.uint8)` | new: `StorageBase.from_buffer` in `bootstrap.py` (`_install_storage_from_buffer`). Slices and byte-swaps the buffer and fills the storage through `StorageBase._shim_fill`, so the `filled` invariant of storage.rs holds. `byte_order`, `count`, `offset` and `dtype` follow upstream, including complex swapped per component and bool normalised to 0/1; the error texts are upstream's |
| `torch.ByteTensor(byte_storage)` | new: the legacy constructors' storage form (`_legacy_tensor_from_storage`), `empty([0]).set_(storage, 0, [n], [1])`. Typed storages must carry the class's dtype; untyped ones give `nbytes // itemsize` elements |
| `torch.LongTensor([n])`, `.to(device)`, `torch.cat`, `torch.sum(...).item()`, slicing by tensor bounds | already present |
| `_get_object_coll_device(group)` reads `group._device_types` | new: `ProcessGroup._device_types`, the registered device types as `torch.device`s |
| `tensor.cpu().numpy().tobytes()[:size]` | already present (`Tensor.numpy`, `Tensor.__index__`) |

<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/src/bootstrap.py _install_storage_from_buffer present -->
<!-- DOCWATCH: symbol-in-file torchnative/rust/torch_c/src/bootstrap.py _legacy_tensor_from_storage present -->
<!-- DOCWATCH: symbol-in-file tests/bindings/test_storagebuf.py test_bytes_in_are_bytes_out_for_empty_small_large_and_non_ascii_pickles present -->
<!-- DOCWATCH: symbol-in-file tests/bindings/test_storagebuf.py test_a_from_buffer_that_delivers_zeros_is_caught present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_p2p.py test_object_lists_agree_with_gloo_at_world_two_and_four present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_p2p.py test_all_gather_object_and_gather_object_refuse_naming_resize_ present -->

Divergence, recorded rather than hidden: upstream's `ByteTensor(storage)` shares the storage; here `set_`'s
storage form copies (storage.rs), so writing into the tensor does not change the storage. The object helpers
never write into it.

Refused by name:

* `from_buffer` for float8, `complex32` and the bit and sub-byte dtypes: `NotImplementedError` naming the
  dtype. Upstream accepts some of them; its argument rules for them were not measured. Upstream itself rejects
  `uint16`/`uint32`/`uint64` and the quantized dtypes (`RuntimeError: Unknown type: UInt32`), and so does this.
* `Tensor.resize_` (the existing catch-all refusal), which `all_gather_object`, `gather_object` and
  `scatter_object_list` call. The first two reach it on every rank after the same collective, so no rank is
  left waiting.

Tests. `tests/bindings/test_storagebuf.py` (new, registered `BOTH` in `.github/scripts/gate_suites.py`) runs
one child against upstream and one against the shim and compares: round trips for empty, one-byte, small, all
256 byte values, a 3 MiB payload and three non-ASCII pickles, each also checked against the payload itself; the
`_object_to_tensor`/`_tensor_to_object` pair; byte order per dtype; `count`/`offset`; the legacy constructor
from typed and untyped storages; 23 refusals with upstream's type and text. The object-list probes in
`test_p2p.py` run in worlds of their own with a 30 s group timeout, behind a communication-free preflight and
an all-reduced failure flag after each probe, so a failure is recorded on every rank instead of stalling the
probes after it.

Red before green. On the main checkout's shim (built 2026-11-05, before this round),
`torch.UntypedStorage.from_buffer(b"abc", dtype=torch.uint8)` and `torch.ByteStorage._from_buffer(b"abc")` both
raised a bare `NotImplementedError`, and `torch.ByteTensor(typed_storage)` refused with "building from data,
`torch.Tensor(TypedStorage)`, is a third form and is not implemented". Measured by a probe script, not by a suite.

Nullification. `test_a_from_buffer_that_delivers_zeros_is_caught` replaces `StorageBase.from_buffer` in the
shim child with one that fills zeros of the right length, and asserts that the round-trip digests then differ
from upstream's for every payload that is not already all zeros, and match again once it is restored. Not yet
run.

