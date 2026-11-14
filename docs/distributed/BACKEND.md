# BACKEND: the `torchnative` distributed backend, one model across a phone's NPU, a PC's Vulkan GPU and a Mac

Status of this document: **design, no code.** Written 2026-11-08 from a read of the tree, not from a
run. The vendored `torch` tree in the main checkout would not import in this session (`libtorch_global_deps.dylib`
absent, the tree is not built), so **no probe ran** and every statement about what imports is a
reading of source, graded *reads*, never *reaches* (AGENTS.md §16). Where this document says "unmeasured", step 0
of the plan measures it.

Direction, in the owner's words (2026-11-08, verbatim):

> "분산 추론을 torch.distributed와 호환되게 구현하고, 한 모델을 여러 디바이스에서 추론 혹은 학습 가능하게 구성해줘. 거기에 우리가 만든 npu, vulkan gpu 같은게 사용될 수 있도록 설계해줘."

and earlier (2026-08-24):

> "torch.distributed를 약간 cpu, gpu, npu 연합 사용 기능으로 구현하고자 목표하면 어때?"

The 2026-08-24 line is a question and was never answered as a decision (AGENTS.md §7). The 2026-11-08 line
approves the goal (distributed inference and training of one model across devices, compatible with
`torch.distributed`, usable with our npu and vulkan). It does **not** pick a transport, a topology, a
backend name or a dependency. Every such choice below is marked **agent proposal, not confirmed**, unless it
appears in the next section, which records what the owner decided afterwards.

### Decisions of 2026-11-08 (owner, and the leader under the owner's delegation)

Marking used in this document: **decided (owner)** carries the owner's verbatim words; **decided (leader)** was
settled by the leader after the owner wrote "나머지는 너의 결정에 따를게" (the remaining questions follow your
decision); everything else is an **agent proposal, not confirmed**.

| # | Decision | Status |
|---|---|---|
| D1 | **One backend named `torchnative`**: `init_process_group(backend="torchnative")`. In torch a backend is named after its communication library (gloo, nccl, mpi), so the name follows that convention. "Heterogeneous" is a property of the devices and the cluster, not of a backend, so no backend is called `hetero`. | decided (owner) for absorbing `local` as a transport: "이게 더 좋긴 한데"; name and convention reasoning from the leader |
| D2 | **`local` is absorbed.** `ProcessGroupLocal`'s code becomes this backend's same-host transport and keeps its gloo-agreement tests. The name `local` is removed and every `backend="local"` caller changes to `"torchnative"` when this is implemented. No alias is kept (AGENTS.md: rules over backward compatibility). | decided (owner, same quote as D1) |
| D3 | **Transports are split intra-node versus inter-node**, as NCCL (P2P, SHM, NET, CollNet) and gloo (tcp, uv, ibverbs) split them. Transports: `inproc` (ranks in one process, memory queues), `shm` (same host: Unix socket first, shared memory later), `tcp` (LAN), `ws` (WebSocket for the browser, through the server). The transport is chosen **per rank pair** automatically (same process: `inproc`; same host: `shm`; otherwise `tcp`) and can be overridden by an option or an environment variable, as `NCCL_P2P_DISABLE` and `GLOO_SOCKET_IFNAME` do. | decided (owner): "전송은 일단 니가 하자는대로 구현하고" |
| D4 | **The server implementation is swappable** (owner: "서버 구현체를 갈아낄 수 있도록 설계해줘"). Three layers: collective algorithms (transport-agnostic), a transport interface, and a **server (coordinator) interface** for rendezvous and address exchange plus federated server-side aggregation. Implementations: an in-process server, a standalone Python server process, the PythonMultiplatform app-host server, and later a remote server. It is built on torch's `Store` and rendezvous extension points. | decided (owner) |
| D5 | **A rank is an execution unit bound to one device, not necessarily an OS process.** Desktop: one process per device, as torchrun does. Android and iOS: several ranks as threads of the app process (free-threaded 3.15t, `inproc` between them, `tcp` to other machines; torch itself has a threaded process group in its test utilities). wasm: one Web Worker per rank (SharedArrayBuffer and messages inside the tab; toolchain #81 already enables cross-origin isolation; WebSocket to other machines through the server, because browsers have no raw TCP). | decided (leader), from the owner's transport and server answers above |
| D6 | **NPU ranks: inference first, and training is in the plan** (owner: "학습도 계획안에 둘 것"). Training goes with compiling backward through the `torch.compile` path. | decided (owner) |
| D7 | **Launcher inside torchnative** (the owner left the location to the leader): `torchnative.distributed.run` (a CLI like torchrun, `python -m torchnative.distributed.run`, torchrun-compatible options) and `torchnative.distributed.launcher` (a programmatic API like `torch.distributed.launcher.api`, used by apps: PythonMultiplatform hosts and browsers start rank threads or workers). | decided (leader) |
| D8 | Python first, then Rust only where measured slow. | decided (leader), formerly Q2 |
| D9 | Same-host transport: Unix socket first (shared memory later, D3). | decided (leader), formerly Q3 |
| D10 | LAN security: loopback by default plus an opt-in pre-shared key using stdlib `hmac` (no new dependency). | decided (leader), formerly Q4 |
| D11 | Uneven devices are balanced with pipeline parallelism first; uneven tensor parallelism is listed, not scheduled. | decided (leader), formerly Q5 |
| D12 | A pipeline rank may load only its layer subset: a new loader option under `torchnative.transformers`. | decided (leader), formerly Q6 |
| D13 | DDP gets a Python `Reducer` under upstream's name, so user code runs unchanged. | decided (leader), formerly Q7 |
| D14 | **Federated server-side aggregation.** Today `torch.federated` aggregates with `all_reduce` and no server, which by the confirmed taxonomy looks like swarm. Server-style aggregation through the server interface (D4) is a follow-up built with this backend. | decided (leader) |

Where the text below still says `hetero` or `local` as a backend name, read D1 and D2: the backend is
`torchnative`, and `local`'s code is its same-host transport.

---

## 0. Answer first

| | |
|---|---|
| What exists | A working `torch.distributed` backend, `local` (to be absorbed, D2): **eleven collectives and five reduce ops agree with upstream gloo at world 3 and 4**, `async_op=True` is real, over **loopback** TCP in a star with the hub at rank 0 (COLLECT2.md §3, ASYNCWORK.md). Everything is CPU tensors, JSON on the wire. |
| What is missing for the owner's goal | (1) **point-to-point** (`send`/`recv`), which pipeline parallelism needs, refuses by name today; (2) **`new_group`**, which an N-D `DeviceMesh` needs: this cell used to say it "refuses", and a reading on 2026-11-13 found no refusal and no `new_group` code in `bootstrap.py` at all; upstream's `new_group` builds each subgroup through this backend's creator under its own `PrefixStore`, so it is expected to work and is **unmeasured** until `tests/distributed/test_newgroup.py` runs (NEWGROUP.md); (3) a **LAN** transport, the data sockets are hard-wired to `127.0.0.1` (`bootstrap.py:16959`, `:16984`); (4) a **binary wire**, JSON of `tolist()` is not a transport for activations; (5) **non-CPU tensors** in any collective (registered `devices=["cpu"]`, `distributed/__init__.py:94`); (6) **`_c10d_functional` kernels**, which DTensor and compiled TP call, have a schema and no kernel (`bootstrap.py:8654-8660`); (7) the C++ **DDP `Reducer`**, refused (DISTRIBUTED.md §8 row 6). |
| What does not need building | Real transformers tensor parallelism (`tp_plan`), as installed (5.15.1), uses **plain `dist.all_reduce`/`all_gather`/`reduce_scatter`** on `device_mesh.get_group()`, not DTensor (`transformers/integrations/tensor_parallel.py:445,482,551`). It needs only the collectives of this backend plus a `DeviceMesh`. |
| The design in one line | One backend, **`torchnative`** (D1), that registers through `Backend.register_backend`, carries raw bytes over transports chosen per rank pair (`inproc`, `shm`, `tcp`, `ws`, D3), stages every device tensor through host memory, and treats **every collective as a graph-break boundary** so an NPU compiles the per-rank local module between collectives and never the collective. `local`'s code becomes its same-host transport and keeps the gloo-agreement tests (D2); the server (rendezvous, address exchange, federated aggregation) is swappable (D4). |
| The honest limit | Over Wi-Fi, tensor parallelism of a decoder is **latency-bound** (two all-reduces per layer per token). Pipeline parallelism sends one activation per stage boundary per token. The design offers both and says which one a link can afford (§7). This is reasoning, not measurement. |

---

## 1. What torchnative has today (read, with citations)

### 1.1 The backend

| Fact | Source |
|---|---|
| Registered as `local` with `Backend.register_backend(..., extended_api=True, devices=["cpu"])` | `torchnative/python/torchnative/distributed/__init__.py:81-96` |
| The collectives are `torch._C._distributed_c10d.ProcessGroupLocal`, a Python class inside `bootstrap.py` (baked into the extension with `include_str!`, AGENTS.md §15.2: edit it and you must rebuild) | `bootstrap.py:16892` |
| Topology: **star**, hub = rank 0, leaves talk only to the hub. The hub folds in **ascending rank order**, so a reduction is a property of the ranks, not of arrival order | `bootstrap.py:16892-16915`, `:17105-17135` |
| Rendezvous: rank 0 binds `("127.0.0.1", 0)`, publishes `pg_local_port` in the store; leaves `connect(("127.0.0.1", port))` | `bootstrap.py:16957-16985` |
| Wire: length-prefixed **JSON** of `t.tolist()` (`!I` length, then UTF-8) | `bootstrap.py:17007-17014`, `:17350`, `:17636`; TRANSPORT.md §2 |
| Timeouts: rendezvous 300 s, per-socket 30 s | `bootstrap.py:16889-16890` |
| Collectives that run and agree with gloo at world 3 and 4: `broadcast` (any root), `all_gather`, `all_gather_into_tensor`, `gather`, `scatter`, `reduce`, `reduce_scatter`, `reduce_scatter_tensor`, `all_to_all`, `all_to_all_single` (equal splits), `barrier`; reduce ops `SUM`, `MIN`, `MAX`, `PRODUCT`, `AVG` | COLLECT2.md §3; `tests/distributed/test_collect2.py` |
| `async_op=True` is real: one worker thread per group, staging clone owned by the `Work`, output published at `wait()` | ASYNCWORK.md §2-§4; `bootstrap.py:17960-18000` |
| Refuses by name: `send`, `recv`, `recv_anysource` (a star has no leaf-to-leaf route), `BAND/BOR/BXOR`, `PREMUL_SUM`, uneven `all_to_all_single`, `*_coalesced` reduce-scatter, secure aggregation, DP. (`new_group` was listed here; the cited lines are `send`/`recv`, and no `new_group` refusal exists anywhere in `bootstrap.py`: NEWGROUP.md §1) | `bootstrap.py:17907-17924`; COLLECT2.md §3; ASYNCWORK.md §9 |
| Store: `TCPStore` is a JSON-over-TCP key/value server in a daemon thread on the master; it binds `host_name`, so it can already sit on a LAN address. At world 1 it refuses. Its wire carries `set`, `get`, `wait` and (since NEWGROUP.md §4) `add`; `check`, `compare_set`, `delete_key`, `num_keys`, `list_keys` and the queue methods refuse by name instead of answering from a local dict | `bootstrap.py:16175-16215` |
| World sizes measured: 1, 2, 3, 4. Nothing larger, nothing off loopback, nothing on a non-CPU tensor | TRANSPORT.md, COLLECT2.md, `distributed/__init__.py:88-94` |

One defect to carry forward, not fix here: `send`/`recv` refusal text says the star has no route and "a relay
the hub can read is not the primitive secure aggregation needs" (`bootstrap.py:17910-17914`). That argument
stands for secure aggregation. It is **not** an argument against pipeline parallelism, whose activations the hub
may see. The new backend needs a direct route between any two ranks (§4.2), which removes the objection rather
than weakening the refusal.

### 1.2 The layers above

`torchnative/nn/federated/` exists; `FederatedDataParallel` and `torch.swarm` are **not in the tree** (grep of
`*.py` and `*.md` found no definition). The federated layer is documented as sitting on `torch.distributed`
(`distributed/__init__.py:3-10`, FEDERATED*.md). Whatever transport this document builds is its transport too
(owner context 2026-08-24, quoted above).

---

## 2. What the vendored tree needs (read, `torchnative/python/torch/distributed`)

Nothing here ran. Import status is the one thing that cannot be read off source, so it is a probe in step 0.

| Piece | What it needs from the shim | State (read) |
|---|---|---|
| `init_process_group(backend="torchnative")` | `Backend.register_backend` with a creator, a `Store`, `ProcessGroup`, `Backend`, `Work` | Works for `local` today, same path (DISTRIBUTED.md §5) |
| `torch.distributed.device_mesh` | opaque-type registration of `ProcessGroup` (`device_mesh.py:72`, `_register_distributed_opaque_types`), `torch.<device_type>` module as device handle (`device_mesh.py:143-150`), `new_group` for any non-trivial mesh | Imports (`bootstrap.py:16623,16733`, test_shim `:7128,:7152`). A **1-D mesh over the whole world** reuses the default group (`device_mesh.py:453-496`) and needs no `new_group`. N-D meshes need `new_group` or `split_group` (`:564,:586`). This cell said **refused today**; that was never measured, and a reading (NEWGROUP.md §2-§3) finds that the CPU path takes `new_group` (`split_group` only when the default group has a bound device, which a CPU group cannot have), and that `new_group` reaches this backend's creator with the subgroup's rank and size. Unmeasured until `tests/distributed/test_newgroup.py` runs |
| `torch.distributed.tensor` (DTensor) | `_make_wrapper_subclass` and `__torch_dispatch__` on a real tensor subclass, `_c10d_functional` **kernels** for every redistribute, `shard_dim_alltoall` | Schemas known, **no kernel**: `_c10d_functional::*` and `_dtensor::shard_dim_alltoall` are listed in `_NON_ATEN_SCHEMA_TEXT` (`bootstrap.py:8655-8689`), and calling one "has no kernel for the `_c10d_functional` namespace and refuses by name" (`:8664-8666`). DISTRIBUTED.md §8 row 8: "DTensor/FSDP **use** untried; import only" |
| `tensor.parallel.parallelize_module` (`ColwiseParallel`, `RowwiseParallel`, `SequenceParallel`) | DTensor, above | Blocked by DTensor |
| `_functional_collectives` (`all_reduce`, `all_gather_tensor`, `reduce_scatter_tensor`, `all_to_all_single`, `permute_tensor`) | the same `_c10d_functional` ops plus `wait_tensor` (`_functional_collectives.py:142,155,179,209,279,583`) | Same wall |
| `pipelining` (`PipelineStage`, `ScheduleGPipe`, `Schedule1F1B`) | `dist.isend/irecv`, `dist.send/recv` for metadata, `batch_isend_irecv`, `send_object_list` (`pipelining/stage.py:507,587,1135,1723,1782`) | Blocked: p2p refuses |
| `fsdp.fully_shard` (FSDP2) | DTensor, `all_gather_into_tensor`, `reduce_scatter_tensor`, a device-handle with streams and events (`_fully_shard/`) | Imports at least as far as `_fsdp_param_group` (OPS8.md:269, SURFACE_HONESTY.md:187); **use** untried. Stream/event semantics on a shim device handle: unmeasured |
| `fsdp.FullyShardedDataParallel` (FSDP1) | `_flat_param`, `ProcessGroup` collectives, `torch.cuda` assumptions | Imports (DISTRIBUTED.md §0); use untried |
| `nn.parallel.DistributedDataParallel` | C++ `Reducer`, `Logger`, `GradBucket`, `_broadcast_coalesced` | **Refused by name** (DISTRIBUTED.md §8 row 6) |
| `transformers` `tp_plan="auto"` / `DistributedConfig(tp_size=N)` | See §2.1 | The shortest road |

### 2.1 `transformers` tensor parallelism, as installed

`.caches/spike-venv/.../transformers/integrations/tensor_parallel.py` (1620 lines, transformers 5.15.1):

- `initialize_tensor_parallelism` (`:52-90`): with no `device_mesh` it asks `torch._C._get_accelerator().type`.
  **The shim answers `cpu` unconditionally** (`bootstrap.py:15165`), so the default path would build a CPU mesh and
  never see a vulkan or npu device. It also raises "Tensor parallelism is not supported on MPS devices" for `mps`
  (`:67`), unreachable today for the same reason. For a non-CPU device it reads `LOCAL_RANK` and calls
  `torch.<device_type>.set_device`. **Consequence: pass `device_mesh=` explicitly.** The `else` branch (`:81-90`)
  takes a mesh with a `"tp"` dimension and builds `torch.device(f"{mesh.device_type}:{LOCAL_RANK}")`.
- The layers (`ColwiseParallel :681`, `RowwiseParallel :838`, `EmbeddingParallel :952`, packed variants, MoE experts)
  shard weights while loading (`get_tensor_shard :301`) and wrap the module with `register_forward_pre_hook` /
  `register_forward_hook` (`distribute_module :603-617`). The communication is
  `dist.all_reduce(x, group=device_mesh.get_group())` (`:445,:456`), `dist.all_gather` (`:482,:528,:569`) and
  `dist.reduce_scatter` (`:551`), plus `dist.all_gather` when saving (`:1394`).
- Source comment `:1522`: DTensor "does not work well with `.view`", which is why transformers keeps plain tensors.

So the inference path for a real HF model needs **three collectives and a mesh**, on plain tensors. That is the
reason the design puts the plain-collective path first and DTensor second (§6).

---

## 3. Devices: what a collective on one has to do

| Device | What it is in torchnative | Tensor destination? | Source |
|---|---|---|---|
| `cpu` | candle CPU, eager | yes | `torchnative/device/__init__.py` |
| `mps` | candle Metal, eager, `torch.device("mps")` | yes | `__init__.py:1-60`; S3.2 |
| `vulkan` | a separate `Repr` arm, eager, taught ops run in SPIR-V, the rest refuse naming themselves; dispatch and `host_uploads`/`host_downloads` counters exist | yes (`EagerDevice`) | S3.3; VULKAN11.md |
| `cuda` | candle device behind a cfg, never run | yes, unrun | S3.4 |
| `npu` | `CompiledDevice`: **not a tensor destination**, `.torch_device` refuses by name (`EagerUseRefused`). `model.to(device.npu)` lowers/compiles the module; per owner (2026-11-08) it triggers `torch.compile` with that device's back end | **no** | `device/__init__.py`; `_module_to.py`; INTENT §5; S4.2 |

Three consequences, all read from the above, none measured:

1. **A tensor never lives on an NPU.** Weights and activations that feed an NPU graph are host tensors. So a
   collective on "an npu rank" is a collective on **host** tensors, and `devices=["cpu"]` is already the correct
   registration for it. The NPU does not need a `Backend` device registration; it needs the module around it to be
   compiled per rank (§5).
2. **Vulkan and mps tensors do live on the device.** A collective on one must **stage**: device-to-host
   (`.cpu()`), send raw bytes, receive into a host buffer, host-to-device. This is the same hop gloo makes for a
   tensor it cannot address. Cost is visible: VULKAN11.md's `host_downloads` counter is the instrument, and
   a collective that stages shows up as `downloads == 1` per call. Upload and download cost on Vulkan and mps is
   **unmeasured here**.
3. The vulkan device refuses most ops (taught ones only). An activation that crosses a rank boundary must land on
   a device that can run the next op, or the next op refuses by name. That is a **placement** constraint, handled in
   §5, not a transport one.

---

## 4. The backend

### 4.1 Registration (name decided, D1; layout an agent proposal, not confirmed)

`torchnative/python/torchnative/distributed/backend/`, a **pure Python package** (D8: Python first), not more code in `bootstrap.py`.
Reason: `bootstrap.py` is baked into the binary at build time (AGENTS.md §15.2), so every edit costs a rebuild
and every agent runs the wrong binary if it forgets. A Python `Backend` subclass registered with
`Backend.register_backend("torchnative", creator, extended_api=True, devices=["cpu", "vulkan", "mps"])` is the same
public door `local` uses (`distributed/__init__.py:81-96`) and can be iterated and tested without a rebuild.

`devices=` lists only devices whose tensors the backend **stages**. `npu` is absent from it for the reason in §3.1.
`cuda` is absent until it has run once.

`local` is **not** left as a second backend (D2): its code becomes the same-host transport of this backend, its
gloo-agreement tests (`tests/distributed/test_collect2.py` and the async tests) move over unchanged in what they
assert, and the name `local` is removed. Every `backend="local"` caller is changed to `"torchnative"` when this is
implemented, with no alias. The backend is tested against gloo (§9). Because the code is absorbed, the JSON-wire
`local` oracle is retired once the binary wire reaches parity in step 3; until then the old class keeps running
the parity comparison.

### 4.2 Transport

| | Proposal | Why |
|---|---|---|
| Wire | Binary frame: fixed header (magic, op, sequence number, dtype, ndim, shape, payload length) then raw little-endian bytes. **No JSON.** | JSON of `tolist()` costs roughly an order of magnitude more bytes than the tensor and a Python loop per element. A decoder all-reduce moves a hidden-size activation per layer; that must not be text. Size ratio is arithmetic, not measured |
| Getting bytes out of and into a tensor | `torch.frombuffer` exists in the shim (`bootstrap.py:4573`); the **export** half (a tensor's bytes without `tolist`) is **unmeasured** (`untyped_storage()` exists, `:14437`; the shim's tensors copy rather than alias, `:4741-4749`). Step 0 measures it | The whole design rests on a cheap, exact bytes path; if it is not there it is the first thing to add |
| LAN (`tcp`) | TCP, `TCP_NODELAY`, any bind address (`--bind`/`MASTER_ADDR`-derived), no `127.0.0.1` constant | `bootstrap.py:16959,16984` |
| Transports | `inproc` (ranks in one process: memory queues), `shm` (same host: Unix socket first, shared memory later), `tcp` (LAN), `ws` (WebSocket for the browser, through the server). **Decided (owner), D3.** Intra-node versus inter-node, as NCCL and gloo split them | The same split is why one backend suffices: the pair of ranks, not the backend, decides the transport |
| Selection | Per rank pair, automatic: same process, `inproc`; same host, `shm`; otherwise `tcp`. Overridable by an option or an environment variable in the style of `NCCL_P2P_DISABLE` and `GLOO_SOCKET_IFNAME` (**decided, D3**; the variable names are an agent proposal) | A phone app with several rank threads and a PC across the LAN is one world with mixed transports |
| Server | A swappable coordinator for rendezvous, address exchange and federated server-side aggregation (**decided, D4**, §4.5) | The transport never assumes who the server is |
| Topology | **Full mesh of persistent connections** (each pair of ranks has one socket), with **rank 0's store key** used only for address exchange. Collectives are built from p2p: ring for `all_gather`/`reduce_scatter`/`all_reduce` when payload is large, star-through-a-chosen-root for tiny payloads | A star gives no leaf-to-leaf route, which is what refuses `send`/`recv` today. A full mesh gives p2p for pipeline stages at no extra concept. For world sizes of a handful of devices, N(N-1)/2 sockets is nothing |
| Determinism | Keep the **ascending-rank fold** contract of `local` (`bootstrap.py:16907-16915`) for `SUM`/`PRODUCT`/`AVG`: a ring reduces in a ring order that is a fixed function of the ranks, documented and tested bit-for-bit against the same expression written centrally, and `local`'s star order is the oracle for small worlds | ASYNCWORK/COLLECT2 §2: arrival-order sums are how a lost rank hides. A ring changes the bracketing from `local`'s, so it is **not** bit-equal to `local`; it is held to gloo's own ring-vs-star tolerance derived per AGREE.md §2. State this in the test, do not hide it |
| Timeouts | Per-op timeout taken from the process group `timeout` argument; no hard-coded 30 s | `_PG_SOCKET_TIMEOUT = 30.0` (`bootstrap.py:16890`) is shorter than an NPU's first-forward compile (NPUPAR.md §1: 504 driver compiles before the second token). §5.3 |
| Async | One worker per peer link (not one per group): `isend`/`irecv` overlap with compute; collectives keep `local`'s "one at a time per group" ordering rule for wire order | ASYNCWORK.md §4 |
| Failure | A peer that dies mid-collective raises on the survivors with the rank named; no silent partial results | AGENTS.md §18 |
| Security | Loopback by default; a non-loopback bind needs an explicit opt-in naming that fact, plus an opt-in pre-shared key using stdlib `hmac` (no new dependency). **Decided (leader), D10.** Plain TCP on a LAN is otherwise unauthenticated | Encryption stays out of scope (§8) |

### 4.3 Staging device tensors

```
send(t):  host = t if t.device is cpu else t.cpu()      # one D2H, counted
          frame(host) -> socket
recv(t):  host = empty(like t, device=cpu); socket -> host
          t.copy_(host)                                  # one H2D if t is on device
```

Rules:

- The staging buffer is owned by the `Work` handle (ASYNCWORK.md §2), so a caller reading its output before `wait()`
  sees the pre-collective contents, deterministically.
- A tensor whose device cannot do the copy refuses **naming the device and the op**, never a silent CPU round trip
  that hides the cost. (Vulkan refuses unlisted ops, S3.3; mps has named readback refusals, S3.2.)
- Each collective on a device tensor reports `host_downloads`/`host_uploads` through the existing counters, so a
  test can assert "exactly one D2H and one H2D per all-reduce".
- `float64` on `mps` refuses by name already (matrix.md §3.1); the backend does not paper over it.
- dtype on the wire is the tensor's own dtype; `bfloat16` and `float16` are sent as raw 2-byte elements. Reduction
  of `float16`/`bfloat16` happens in **float32 on the host and is rounded once**, as gloo does. Exact tolerance is
  derived the AGREE.md way, not guessed.

### 4.4 Collective coverage required, by consumer

| Consumer | Needs | Today | New backend |
|---|---|---|---|
| TP inference (`transformers` tp_plan) | `all_reduce`, `all_gather`, `reduce_scatter`, `broadcast` (load), `barrier` | **have** (CPU, JSON, loopback) | binary, LAN, device staging |
| TP with sequence parallel, MoE expert parallel | `all_to_all_single` (uneven splits too) | equal splits only | uneven splits |
| PP | `send`, `recv`, `isend`, `irecv`, `batch_isend_irecv`, `send_object_list`/`recv_object_list` (shape metadata) | **refused** | **new** |
| FSDP (1 and 2) | `all_gather_into_tensor`, `reduce_scatter_tensor`, `broadcast`, `all_reduce` (HSDP) | have | coalesced forms, stream-free async |
| DDP | `broadcast` (init sync), `all_reduce` per gradient bucket, `barrier` | collectives have; **Reducer refused** | Python `Reducer` (§6.3) |
| N-D `DeviceMesh` (tp x dp, pp x tp) | `new_group`, `split_group` | `new_group`: no refusal exists, expected to work by reading, unmeasured (NEWGROUP.md); `split_group`: refused by upstream's own Python on CPU, gloo included | **new**: subgroups over the mesh's persistent connections |
| Federated / swarm | `all_reduce` weighted, partial attendance (`allreduce_partial`) | have | kept; partial-attendance semantics ported |
| Not offered | `recv_anysource` is kept refused (no consumer); bitwise reduce ops stay refused; `PREMUL_SUM` stays refused until FSDP2 needs it |

### 4.5 Three layers and a swappable server (decided, D4; interfaces are agent proposals)

1. **Collective algorithms**: ring, star-through-a-root, the ascending-rank fold. They see only `send`/`recv` on a
   peer and know nothing about sockets.
2. **Transport interface**: `inproc`, `shm`, `tcp`, `ws` (§4.2). One connection per rank pair, chosen per pair.
3. **Server (coordinator) interface**: rendezvous, address exchange and, for `torch.federated`, server-side
   aggregation. Implementations: an in-process server, a standalone Python server process, the PythonMultiplatform
   app-host server, and later a remote server. It is built on torch's `Store` and rendezvous extension points
   (`torch.distributed.rendezvous` handlers, `Store`), so `init_process_group` needs no new door.

Federated aggregation today is `all_reduce` without a server. By the confirmed taxonomy that is swarm-shaped.
Server-style aggregation through this interface is a follow-up built with the backend (D14).

---

## 5. Placing a model on heterogeneous ranks

### 5.1 A rank is an execution unit bound to one device, rank to device

**Decided (leader, D5):** a rank is an execution unit bound to one device, not necessarily an OS process.

| Platform | Rank is | Between ranks of the same app | To other machines |
|---|---|---|---|
| Desktop | one process per device, as torchrun does | `shm` on the same host | `tcp` |
| Android, iOS | a thread of the app process; several ranks per app (free-threaded 3.15t) | `inproc` | `tcp` |
| wasm (browser) | one Web Worker per rank | SharedArrayBuffer and messages inside the tab (toolchain #81 already enables cross-origin isolation) | `ws` through the server, since browsers have no raw TCP |

torch itself has a threaded process group in its test utilities, so ranks as threads are not foreign to the API.
Ranks sharing a process share the GIL unless the interpreter is free-threaded, which is why the mobile case is tied
to 3.15t; shim single-writer assumptions (ASYNCWORK.md §4) must be re-read for it before it is claimed to work.

The mapping rank to device is a plain table the caller (or `torchnative.api`, or the launcher of §5.5) builds and
the backend does not interpret:

```python
# agent proposal, not confirmed: the shape, not an API
mesh = init_device_mesh("cpu", (3,), mesh_dim_names=("tp",))   # collectives always run on host tensors
# rank 0 (phone):  model.to(torchnative.device.npu)
# rank 1 (PC):     model.to("vulkan")
# rank 2 (Mac):    model.to("mps")
```

The mesh's `device_type` is **`cpu`** for transport purposes. The device a rank *computes* on is separate and
per-rank. `DeviceMesh` upstream assumes one device type for the whole mesh and reads
`torch.<device_type>` for `device_count()` (`device_mesh.py:120-150`). Using `"cpu"` for the mesh sidesteps that
assumption instead of fighting it. This is a deliberate reading of `device_mesh.py`, **unmeasured**.

### 5.2 Shard first, then compile each rank's local module

Owner decision (INTENT §5): `model.to(device.npu)` triggers `torch.compile` with that device's back end, compiles on
first forward per input shape, recompiles on shape change. For a sharded model the order that works with this is:

1. Build the model, **shard weights while loading** (transformers `get_tensor_shard`, or `parallelize_module`):
   each rank holds `1/N` of the matrices.
2. Install the communication as **module hooks outside the compiled region** (transformers does exactly this:
   forward pre/post hooks, `tensor_parallel.py:603-617`).
3. `model.to(device)` last. The compiled graph is the rank's local layer body, whose input is a plain local tensor.

Why collectives must not be inside the compiled graph: an NPU takes whole subgraphs (AGENTS.md §20); no NPU
back end here has a collective op, and a collective is a network call. With hooks outside, Dynamo's capture sees
`Linear -> Linear` on local shards and the all-reduce sits between graphs, which is a graph break by construction,
not by accident. The DTensor path puts collectives *inside* the traced region (`_c10d_functional` ops in the
graph), which an NPU back end cannot lower: **DTensor on an NPU rank is refused by name** unless the back end
learns the op (it is not on the list; §8).

Shape implication: a per-rank shard changes the weight shape, so a graph compiled for the full model is wrong for a
shard. Compile after sharding, never before, and key the compile cache by shard shape. Recompile on batch/sequence
change is already the decided behaviour.

### 5.3 First-forward compile versus a collective timeout

A rank that compiles for minutes at its first forward is silent to its peers; a 30 s socket timeout kills the
world (`bootstrap.py:16890`). Design rule: **compile eagerly at `to()`**, which NPUPAR.md §3 already ships for the
Intel path ("eager compilation, the same serial work, moved to where the user asked for it"), then run a
**barrier with a long timeout** before the first real collective. Where eager compile is not available for a back
end, the group timeout must be raised by the caller; the backend does not guess.

### 5.4 Uneven devices

**Decided (leader, D11): pipeline parallelism is the uneven-device tool; uneven tensor parallelism is listed, not scheduled.**
Weights are not split evenly by default. Proposal: the sharding plan accepts **weights per rank** (e.g. 5:3:2),
chosen from a capability record (free memory, measured matmul throughput) the caller can also pass by hand.
Upstream `ColwiseParallel` splits evenly, transformers' `get_tensor_shard` splits evenly: uneven TP means either
a patch point in our own `parallelize` wrapper or padding. **Agent proposal: pipeline parallelism is the uneven-device
tool** (give each stage as many layers as its memory and speed justify) and TP stays even across
similar devices. Uneven TP is listed but not scheduled (D11).

### 5.5 Launcher (decided, D7)

Inside torchnative, nothing at the repository root:

- `torchnative.distributed.run`: a CLI like torchrun, `python -m torchnative.distributed.run`, with
  torchrun-compatible options (nnodes, nproc_per_node, rdzv endpoint, and so on).
- `torchnative.distributed.launcher`: a programmatic API like `torch.distributed.launcher.api`, used by apps. A
  PythonMultiplatform host starts rank threads; a browser starts rank workers.

Fault-tolerant relaunch (elastic) stays out of scope (§8).

---

## 6. The paths

### 6.1 Inference by tensor parallelism (`transformers`, first)

`from_pretrained(..., distributed_config=DistributedConfig(tp_size=N), device_mesh=mesh)` or `tp_plan` on a
`from_config` model. Needs: backend collectives (have, plus binary/LAN), a 1-D `DeviceMesh` (reuses the default
group, no `new_group`), `torch.distributed.is_initialized()` (read at `modeling_utils.py:4993`). Unmeasured: whether
`DeviceMesh` construction and `get_group()` work on the shim with a `torchnative` default group. The collectives run
under `torch.autograd.Function`s (`tensor_parallel.py:431-570`), so **TP training on plain tensors comes with the
same code** (`backward` issues the matching collective); it is graded separately.

### 6.2 Inference by pipeline parallelism (devices with too little memory)

Two routes:

- **Upstream `torch.distributed.pipelining`** (`PipelineStage`, schedules): needs p2p, `send_object_list`,
  `batch_isend_irecv`, and `_unflatten`/`_IR` machinery that uses `torch.export` (EXPORT*.md say export capture
  works in part). Unmeasured. Lowest conceptual cost, highest risk.
- **A thin torchnative stage runner** over `isend`/`irecv`: split `model.layers` by index ranges, rank r runs its
  slice, hidden states cross the boundary as one tensor per token step, KV cache stays on the rank that owns the
  layer. This is the minimum for autoregressive generation across devices and has no export dependency.
  **Agent proposal; both are built from the same p2p primitive, so the primitive comes first and the choice can wait.**

A real HF model split by layer ranges needs `from_pretrained` to load only a layer subset per rank (memory is the
point of PP). **Decided (leader, D12):** a rank may load only its layer subset through a new loader option under `torchnative.transformers`. Whether the installed loader can skip non-local layers without `device_map` (which would make the option thin) is unmeasured.

### 6.3 Training

- **TP training:** §6.1, autograd functions already pair forward and backward collectives.
- **Data parallel across devices:** all-reduce of gradients. The upstream way is DDP's `Reducer`, refused today.
  **Decided (leader, D13): a Python gradient-bucketing reducer under upstream's name**, so user code runs unchanged (`register_post_accumulate_grad_hook` is *refused* by
  name in this shim, S6.1, so the reducer uses `Tensor.register_hook` on leaves, which agrees with upstream per S6.1),
  the refusal of the C++ `Reducer` is replaced, not worked around).
- **FSDP2 `fully_shard`:** pure Python on top of DTensor; blocked by DTensor (§2). FSDP1 imports; use untried.
- **`FederatedDataParallel`:** its collectives are `all_reduce` with weights and partial attendance, which `local`
  already has; it inherits the backend unchanged. Server-style aggregation through the server interface is a
  follow-up (D14).
- **Optimizer step location:** each rank steps the parameters it owns; the hetero point is that a vulkan rank and an
  npu rank hold different dtypes/precisions. Mixed precision across ranks is the caller's, collectives require the
  **same dtype on both ends** and refuse otherwise.
- **Backward on an NPU rank: in the plan, after inference** (**decided (owner), D6**: "학습도 계획안에 둘 것").
  Training goes with compiling backward through the `torch.compile` path. Today a compiled-device module's
  `backward()` works only because `to(npu)` lowers leaves in place and wraps nothing (`_module_to.py` docstring), so
  this is new work, scheduled as step 12.

### 6.4 DTensor tier (second, after the plain tier is green)

Required, in order: (a) `_make_wrapper_subclass` and `__torch_dispatch__` on a real subclass (DISPATCH3.md covers
`FakeTensorMode`, not necessarily a user subclass; **unmeasured**); (b) kernels for the `_c10d_functional` ops
(`all_reduce`, `all_gather_into_tensor`, `reduce_scatter_tensor`, `broadcast`, `all_to_all_single`, `wait_tensor`,
`isend`, `irecv`), each a thin Python function calling the process group, `wait_tensor` joining the `Work`; how a
Python-defined kernel is registered for a namespace that the single Rust door refuses (`bootstrap.py:8664`) is the
design question, answered in step 0; (c) `_dtensor::shard_dim_alltoall`. When this lands, `parallelize_module`,
`distribute_tensor`, `DTensor.redistribute` and FSDP2 follow with differential tests against upstream.

---

## 7. What links can afford (reasoning, unmeasured)

A decoder layer under TP does two all-reduces per token per layer (attention output, MLP output). For a 32-layer model
that is 64 all-reduces per generated token, each a round trip. At a LAN round trip of about 1 ms that is 64 ms of
pure latency; over Wi-Fi at 3-10 ms, 0.2 to 0.6 s per token, before any bytes. Pipeline parallelism pays one
transfer per stage boundary per token. Therefore: **TP is for ranks on one host or one wired switch; PP is the
default across a phone and a PC.** The backend exposes per-op latency and bytes counters so this stops being an
estimate; step 17 measures it. These numbers are my arithmetic, not a measurement, and the owner should not cite them.

---

## 8. Out of scope, refused by name

Each refuses naming itself and the reason (AGENTS.md §18):

| Not built | Reason |
|---|---|
| `torch.distributed.rpc`, `torch.distributed.elastic`, `torchrun` fault-tolerant relaunch | No consumer in the owner's request; the launcher of §5.5 is enough. `run.py`/`launch.py` imports are not promised |
| NCCL, UCC, MPI, Gloo-as-ours backends | Absent by design (the one backend is `torchnative`, D1) (`bootstrap.py` docstring of `_install_distributed_c10d`): their availability flags must stay False |
| DTensor ops inside an NPU-compiled graph | No NPU back end can lower a collective (§5.2) |
| `recv_anysource`, bitwise reduces, `PREMUL_SUM` | No consumer; stay refused as today |
| Secure aggregation, differential privacy | Unchanged (FEDERATED4.md §7) |
| `cuda` in `devices=` | Never run on a GPU (S3.4) |
| Encryption on the wire | Not built. Authentication is the opt-in pre-shared key of D10; non-loopback binds need an explicit acknowledgement (§4.2) |
| Sub-interpreters as the way to get parallel ranks | Same-process ranks are **in scope** as threads (D5, free-threaded 3.15t, `inproc` transport), not as sub-interpreters (shim single-writer assumptions, ASYNCWORK.md §4) |
| Other distributed-inference engines as dependencies | They are comparison baselines to benchmark against later, not parts of this design |
| Cross-device mixed dtype in one collective | Refused with both dtypes named |

---

## 9. Plan, with sizes and the test that decides each step

Sizes are agent estimates in working sessions (S = under a day, M = 1-3 days, L = a week), not measurements.
Every step starts with tests written from the SPEC rows, observed red (AGENTS.md rule 5), and agents do not run
builds (AGENTS.md rule 10): the gate runs in CI or in the coordinating session.

| # | Step | Size | Test (differential against upstream gloo wherever it can be) |
|---|---|---|---|
| 0 | **Probe, no code.** On a built tree: which of `device_mesh`, `tensor.parallel`, `pipelining`, `fsdp`, `_functional_collectives` import and construct at world 1; whether `torch.frombuffer`-style raw export/import is cheap and exact; whether `_make_wrapper_subclass`/`__torch_dispatch__` works on a user subclass; how a Python kernel could be registered under `_c10d_functional`; cost of `.cpu()`/upload on vulkan and mps; whether torch's threaded process group utilities import (D5) | S | A transcript recorded in this doc's §10, graded *reaches* only for what ran |
| 1 | **Absorb `local` (D2)**: register `backend="torchnative"`, move `ProcessGroupLocal`'s code in as the same-host transport behind a `Transport` interface, keep its gloo-agreement tests, remove the name `local` and change every `backend="local"` caller (no alias) | M | The existing `test_collect2.py` and async tests, with the backend name changed, still agree with gloo at world 3 and 4; `grep` finds no `backend="local"`; sabotage: break the transport and the agreement tests must fail |
| 2 | **Binary frame and the transports**: `inproc`, `shm` (Unix socket), `tcp` (any bind address, `TCP_NODELAY`, no `127.0.0.1` literal), per-pair selection and its override. Round-trip every dtype, `bfloat16` included, bit-exact; a frame split across many `recv`s; a truncated frame raises naming the peer | M | Loopback pairs per transport; sabotage: flip a header byte, expect a named error; a forced override changes the transport actually used |
| 3 | **Backend parity on the binary wire**: the eleven collectives, five reduce ops, async, at world 3 and 4 | M | `tests/distributed/test_torchnative_parity.py`: same harness as `test_collect2.py`, compared with **gloo**; bit equality for the selecting ops, derived tolerance for reductions (AGREE.md §2) |
| 4 | **Point-to-point**: `send`, `recv`, `isend`, `irecv`, `batch_isend_irecv`, `send_object_list`/`recv_object_list` over the full mesh | M | Differential vs gloo p2p at world 2 and 4 (ring and non-neighbour pairs, tag matching, out-of-order arrival) |
| 5 | **Server interface (D4)**: the coordinator for rendezvous and address exchange, with the in-process and standalone-process implementations behind it, on torch's `Store` and rendezvous extension points; per-op timeouts from the group; a killed rank names itself on the survivors; the opt-in pre-shared key (D10) | M | The same world-of-3 test runs against both server implementations and passes; a wrong key is refused by name; a real two-machine run is a **device test the coordinating session runs itself** (AGENTS.md §10), never claimed from CI |
| 6 | **Rank as thread for mobile (D5)**: several ranks in one process on `inproc`, `tcp` to other machines, on free-threaded 3.15t | M | A world of 4 as 4 threads of one process agrees with gloo; mixed world (2 threads + 1 remote process) passes; on a GIL build the test is `UNVERIFIED`, not a pass |
| 7 | **Device staging**: vulkan and mps tensors through all collectives with counted H2D/D2H; refusals name the device and op; `devices=` set accordingly | M | vs gloo on the CPU copy of the same data; counter test "one download, one upload per all-reduce"; nullify the staging and the counter test must fail. Needs a Vulkan loader (`TORCHNATIVE_REQUIRE_VULKAN=1` so a missing loader is `UNVERIFIED`, not a pass) |
| 8 | **`new_group` / `split_group`** over persistent connections, and an N-D `DeviceMesh` | M | `init_device_mesh("cpu", (2,2), mesh_dim_names=("dp","tp"))` at world 4: each `get_group()` all-reduces over the right ranks, compared with gloo on the same mesh. Tests written: `tests/distributed/test_newgroup.py` (issue #221), over today's star; not yet run (NEWGROUP.md) |
| 9 | **TP inference with a real `transformers` model**: `tp_size=2` then 3 on loopback CPU, explicit `device_mesh`, logits against the single-process model and against upstream torch with gloo TP | L | The existing real-model agreement harness (E2E_REAL.md); tolerance derived; world 1 control must pass unchanged. Then the same with one rank `to("vulkan")` |
| 10 | **PP inference**: p2p stage runner (route B) generating tokens across 2 and 3 ranks; KV cache stays per stage; the layer-subset loader option (D12) | L | Greedy tokens equal the single-process run, bit for bit; one rank on vulkan/npu where the device exists |
| 11 | **NPU rank, inference**: shard, then `to(device.npu)`, collectives between compiled bodies; eager compile at `to()`, long-timeout barrier | M | On a host with an NPU back end (Intel NPU on Windows is the only one with a recorded run, S5.3, which itself is "partial"): a device test run by the coordinating session. Without an NPU: only the **ordering and timeout logic** is tested, against a faked compile delay, and the report says so |
| 12 | **Training**: TP training step vs upstream gloo (loss and grads); the Python `Reducer` under DDP's name (D13); `FederatedDataParallel` rides on the same group; **NPU training** by compiling backward through the `torch.compile` path (D6) | L | One optimiser step equals upstream's within a derived tolerance, per S6.1's method; DDP user code runs unchanged; NPU training is a device test of the same kind as step 11 |
| 13 | **Wasm worker transport (D5)**: one Web Worker per rank, SharedArrayBuffer and messages inside the tab, `ws` to other machines through the server | M | A world of 2 workers in a browser test agrees with the same computation on `inproc`; `ws` needs the toolchain #81 cross-origin isolation; a browser test run by the coordinating session |
| 14 | **Launcher (D7)**: `torchnative.distributed.run` and `torchnative.distributed.launcher` | M | `python -m torchnative.distributed.run` starts a world of 3 with torchrun-compatible options and its collectives agree with gloo; the programmatic API starts rank threads from a Python host |
| 15 | **Federated server-side aggregation (D14)** through the server interface, a follow-up | M | Server aggregation equals the all_reduce aggregation of S6.7 on the same inputs |
| 16 | **DTensor tier** (§6.4): `_c10d_functional` kernels, `shard_dim_alltoall`, then `parallelize_module` and FSDP2 | L | Each redistribute vs upstream; `fully_shard` step vs upstream at world 2 |
| 17 | **Capability-weighted placement** and uneven PP, plus a measured latency/bytes table that replaces §7's arithmetic | M | Placement is a pure function, unit tested; the table is a measurement record |

Steps 1-8 are the backend. Steps 9-10 deliver the owner's inference goal. Steps 11-12 deliver npu and training.
Steps 13-15 are the platform reach (wasm, launcher, server aggregation). Step 16 is the upstream-DTensor-compatible
tier and is last because the plain tier already gives a real model on N devices.

---

## 10. Measurements still to record

This section is intentionally empty. Step 0 fills it. Nothing above is to be quoted as measured.

Steps 2 and 4 were written on 2026-11-13 (issue #219) and are recorded in `docs/distributed/P2P.md`, graded
*builds* only: nothing ran. Rows of §1.1 that say `send`/`recv` refuse describe the tree before that round.

Step 10 (pipeline-parallel generation and the D12 loader option) was written on 2026-11-13 (issue #295) and is
recorded in `docs/distributed/PPINFER.md`, graded *builds* only: nothing ran. It answers §6.2's open question by
reading: the installed loader skips non-local layers without `device_map` once the model it builds has only
those layers (PPINFER.md §2). Step 10's device rank (vulkan or npu) is not claimed.

---

## 11. Questions for the owner: all answered

The nine questions of the first draft were answered on 2026-11-08, each now a decision in the table near the top.

| Old question | Answer | Where |
|---|---|---|
| Q1 backend name and the fate of `local` | One backend `torchnative`; `local` absorbed as the same-host transport (owner: "이게 더 좋긴 한데") | D1, D2 |
| Q2 Python or Rust | Python first, Rust where measured slow | D8 |
| Q3 same-host transport | Unix socket first, shared memory later | D3, D9 |
| Q4 network security | Loopback by default, opt-in pre-shared key with stdlib `hmac` | D10 |
| Q5 uneven tensor parallelism | Pipeline parallelism first | D11 |
| Q6 layer-subset loading | New loader option | D12 |
| Q7 DDP | Python `Reducer` under upstream's name | D13 |
| Q8 NPU training | In the plan, after inference (owner: "학습도 계획안에 둘 것") | D6 |
| Q9 launcher location | Inside torchnative (owner left it to the leader) | D7 |

Still **agent proposals, not confirmed**: the binary frame layout, the full-mesh topology and ring determinism
contract, the module layout of §4.1, the environment-variable names of D3, the interface shapes of §4.5, the
pipeline stage runner (§6.2), the DTensor tier order (§6.4).

No new dependency is introduced by this design. Shared memory beyond the stdlib, or TLS, would be a dependency
question and returns to the owner.

---

## 12. Reproducing this reading

```sh
cd torchnative/.worktrees/dist-infer-design
grep -n "ProcessGroupLocal\|_PG_.*TIMEOUT\|127.0.0.1" torchnative/rust/torch_c/src/bootstrap.py
sed -n 52,90p .caches/spike-venv/lib/python3.13/site-packages/transformers/integrations/tensor_parallel.py
```

(the second path is relative to the main checkout, `.caches/` is not in a worktree). Nothing was run against the shim.
