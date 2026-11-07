# HETERO: one model across a phone's NPU, a PC's Vulkan GPU and a Mac, through `torch.distributed`

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
backend name or a dependency. Every such choice below is marked **agent proposal, not confirmed**.

---

## 0. Answer first

| | |
|---|---|
| What exists | A working `torch.distributed` backend, `local`: **eleven collectives and five reduce ops agree with upstream gloo at world 3 and 4**, `async_op=True` is real, over **loopback** TCP in a star with the hub at rank 0 (COLLECT2.md §3, ASYNCWORK.md). Everything is CPU tensors, JSON on the wire. |
| What is missing for the owner's goal | (1) **point-to-point** (`send`/`recv`), which pipeline parallelism needs, refuses by name today; (2) **`new_group`**, which an N-D `DeviceMesh` needs, refuses; (3) a **LAN** transport, the data sockets are hard-wired to `127.0.0.1` (`bootstrap.py:16959`, `:16984`); (4) a **binary wire**, JSON of `tolist()` is not a transport for activations; (5) **non-CPU tensors** in any collective (registered `devices=["cpu"]`, `distributed/__init__.py:94`); (6) **`_c10d_functional` kernels**, which DTensor and compiled TP call, have a schema and no kernel (`bootstrap.py:8654-8660`); (7) the C++ **DDP `Reducer`**, refused (DISTRIBUTED.md §8 row 6). |
| What does not need building | Real transformers tensor parallelism (`tp_plan`), as installed (5.15.1), uses **plain `dist.all_reduce`/`all_gather`/`reduce_scatter`** on `device_mesh.get_group()`, not DTensor (`transformers/integrations/tensor_parallel.py:445,482,551`). It needs only the collectives of this backend plus a `DeviceMesh`. |
| The design in one line | A new transport-parameterised backend, **`hetero`** (name: proposal), that registers through `Backend.register_backend`, carries raw bytes over TCP or a Unix socket, stages every device tensor through host memory, and treats **every collective as a graph-break boundary** so an NPU compiles the per-rank local module between collectives and never the collective. `local` stays as the JSON reference oracle. |
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
| Refuses by name: `send`, `recv`, `recv_anysource` (a star has no leaf-to-leaf route), `BAND/BOR/BXOR`, `PREMUL_SUM`, uneven `all_to_all_single`, `*_coalesced` reduce-scatter, `new_group`, secure aggregation, DP | `bootstrap.py:17907-17924`; COLLECT2.md §3; ASYNCWORK.md §9 |
| Store: `TCPStore` is a JSON-over-TCP key/value server in a daemon thread on the master; it binds `host_name`, so it can already sit on a LAN address. At world 1 it refuses | `bootstrap.py:16175-16215` |
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
| `init_process_group(backend="hetero")` | `Backend.register_backend` with a creator, a `Store`, `ProcessGroup`, `Backend`, `Work` | Works for `local` today, same path (DISTRIBUTED.md §5) |
| `torch.distributed.device_mesh` | opaque-type registration of `ProcessGroup` (`device_mesh.py:72`, `_register_distributed_opaque_types`), `torch.<device_type>` module as device handle (`device_mesh.py:143-150`), `new_group` for any non-trivial mesh | Imports (`bootstrap.py:16623,16733`, test_shim `:7128,:7152`). A **1-D mesh over the whole world** reuses the default group (`device_mesh.py:453-496`) and needs no `new_group`. N-D meshes need `new_group` or `split_group` (`:564,:586`): **refused today** |
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

### 4.1 Registration (agent proposal, not confirmed)

`torchnative/python/torchnative/distributed/hetero/`, a **pure Python package**, not more code in `bootstrap.py`.
Reason: `bootstrap.py` is baked into the binary at build time (AGENTS.md §15.2), so every edit costs a rebuild
and every agent runs the wrong binary if it forgets. A Python `Backend` subclass registered with
`Backend.register_backend("hetero", creator, extended_api=True, devices=["cpu", "vulkan", "mps"])` is the same
public door `local` uses (`distributed/__init__.py:81-96`) and can be iterated and tested without a rebuild.

`devices=` lists only devices whose tensors the backend **stages**. `npu` is absent from it for the reason in §3.1.
`cuda` is absent until it has run once.

`local` is left as it is: it is the reference that agrees with gloo, and the new backend is tested **against it
and against gloo** (§9).

### 4.2 Transport

| | Proposal | Why |
|---|---|---|
| Wire | Binary frame: fixed header (magic, op, sequence number, dtype, ndim, shape, payload length) then raw little-endian bytes. **No JSON.** | JSON of `tolist()` costs roughly an order of magnitude more bytes than the tensor and a Python loop per element. A decoder all-reduce moves a hidden-size activation per layer; that must not be text. Size ratio is arithmetic, not measured |
| Getting bytes out of and into a tensor | `torch.frombuffer` exists in the shim (`bootstrap.py:4573`); the **export** half (a tensor's bytes without `tolist`) is **unmeasured** (`untyped_storage()` exists, `:14437`; the shim's tensors copy rather than alias, `:4741-4749`). Step 0 measures it | The whole design rests on a cheap, exact bytes path; if it is not there it is the first thing to add |
| LAN | TCP, `TCP_NODELAY`, any bind address (`--bind`/`MASTER_ADDR`-derived), no `127.0.0.1` constant | `bootstrap.py:16959,16984` |
| Same host | Unix domain socket when both ends share a host (optional), otherwise loopback TCP | Question Q3 |
| Topology | **Full mesh of persistent connections** (each pair of ranks has one socket), with **rank 0's store key** used only for address exchange. Collectives are built from p2p: ring for `all_gather`/`reduce_scatter`/`all_reduce` when payload is large, star-through-a-chosen-root for tiny payloads | A star gives no leaf-to-leaf route, which is what refuses `send`/`recv` today. A full mesh gives p2p for pipeline stages at no extra concept. For world sizes of a handful of devices, N(N-1)/2 sockets is nothing |
| Determinism | Keep the **ascending-rank fold** contract of `local` (`bootstrap.py:16907-16915`) for `SUM`/`PRODUCT`/`AVG`: a ring reduces in a ring order that is a fixed function of the ranks, documented and tested bit-for-bit against the same expression written centrally, and `local`'s star order is the oracle for small worlds | ASYNCWORK/COLLECT2 §2: arrival-order sums are how a lost rank hides. A ring changes the bracketing from `local`'s, so it is **not** bit-equal to `local`; it is held to gloo's own ring-vs-star tolerance derived per AGREE.md §2. State this in the test, do not hide it |
| Timeouts | Per-op timeout taken from the process group `timeout` argument; no hard-coded 30 s | `_PG_SOCKET_TIMEOUT = 30.0` (`bootstrap.py:16890`) is shorter than an NPU's first-forward compile (NPUPAR.md §1: 504 driver compiles before the second token). §5.3 |
| Async | One worker per peer link (not one per group): `isend`/`irecv` overlap with compute; collectives keep `local`'s "one at a time per group" ordering rule for wire order | ASYNCWORK.md §4 |
| Failure | A peer that dies mid-collective raises on the survivors with the rank named; no silent partial results | AGENTS.md §18 |
| Security | **None by default.** Plain TCP on a LAN is unauthenticated. The backend refuses to bind a non-loopback address unless the caller passes an explicit opt-in naming that fact | Agent proposal; a pre-shared key is a dependency-free option (`hmac`, stdlib). Question Q4 |

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
| N-D `DeviceMesh` (tp x dp, pp x tp) | `new_group`, `split_group` | refused | **new**: subgroups over the mesh's persistent connections |
| Federated / swarm | `all_reduce` weighted, partial attendance (`allreduce_partial`) | have | kept; partial-attendance semantics ported |
| Not offered | `recv_anysource` is kept refused (no consumer); bitwise reduce ops stay refused; `PREMUL_SUM` stays refused until FSDP2 needs it |

---

## 5. Placing a model on heterogeneous ranks

### 5.1 One process per device, rank to device

A rank is a process. **One process per device** (phone NPU, PC Vulkan, Mac mps) is the general case; **one process
per host driving several local devices** is a later optimisation. The mapping rank to device is a plain table the
caller (or `torchnative.api`) builds and the backend does not interpret:

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

Weights are not split evenly by default. Proposal: the sharding plan accepts **weights per rank** (e.g. 5:3:2),
chosen from a capability record (free memory, measured matmul throughput) the caller can also pass by hand.
Upstream `ColwiseParallel` splits evenly, transformers' `get_tensor_shard` splits evenly: uneven TP means either
a patch point in our own `parallelize` wrapper or padding. **Agent proposal: pipeline parallelism is the uneven-device
tool** (give each stage as many layers as its memory and speed justify) and TP stays even across
similar devices. Uneven TP is listed but not scheduled (Q5).

---

## 6. The paths

### 6.1 Inference by tensor parallelism (`transformers`, first)

`from_pretrained(..., distributed_config=DistributedConfig(tp_size=N), device_mesh=mesh)` or `tp_plan` on a
`from_config` model. Needs: backend collectives (have, plus binary/LAN), a 1-D `DeviceMesh` (reuses the default
group, no `new_group`), `torch.distributed.is_initialized()` (read at `modeling_utils.py:4993`). Unmeasured: whether
`DeviceMesh` construction and `get_group()` work on the shim with a `hetero` default group. The collectives run
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
point of PP). Whether the installed loader can skip non-local layers without `device_map`, or needs a
`device_map`-style plan per rank, is unmeasured (Q6).

### 6.3 Training

- **TP training:** §6.1, autograd functions already pair forward and backward collectives.
- **Data parallel across devices:** all-reduce of gradients. The upstream way is DDP's `Reducer`, refused today.
  **Agent proposal: a Python gradient-bucketing reducer** (`register_post_accumulate_grad_hook` is *refused* by
  name in this shim, S6.1, so the reducer uses `Tensor.register_hook` on leaves, which agrees with upstream per S6.1),
  exposed under `nn.parallel.DistributedDataParallel`'s name only if it is a drop-in; otherwise under its own name
  and DDP stays refused by name. Q7.
- **FSDP2 `fully_shard`:** pure Python on top of DTensor; blocked by DTensor (§2). FSDP1 imports; use untried.
- **`FederatedDataParallel`:** owner-named, not in the tree. Its collectives are `all_reduce` with weights and
  partial attendance, which `local` already has; it inherits the backend unchanged.
- **Optimizer step location:** each rank steps the parameters it owns; the hetero point is that a vulkan rank and an
  npu rank hold different dtypes/precisions. Mixed precision across ranks is the caller's, collectives require the
  **same dtype on both ends** and refuse otherwise.
- **Backward on an NPU rank** is a question this document does not answer: a compiled-device module's `backward()`
  works today only because `to(npu)` lowers leaves in place and wraps nothing (`_module_to.py` docstring). Whether
  training through a compiled NPU graph is in scope is the owner's (Q8).

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
estimate; step 7 measures it. These numbers are my arithmetic, not a measurement, and the owner should not cite them.

---

## 8. Out of scope, refused by name

Each refuses naming itself and the reason (AGENTS.md §18):

| Not built | Reason |
|---|---|
| `torch.distributed.rpc`, `torch.distributed.elastic`, `torchrun` fault-tolerant relaunch | No consumer in the owner's request; a rendezvous script of our own (`torchnative` launcher) is enough. `run.py`/`launch.py` imports are not promised |
| NCCL, UCC, MPI, Gloo-as-ours backends | Absent by design (`bootstrap.py` docstring of `_install_distributed_c10d`): their availability flags must stay False |
| DTensor ops inside an NPU-compiled graph | No NPU back end can lower a collective (§5.2) |
| `recv_anysource`, bitwise reduces, `PREMUL_SUM` | No consumer; stay refused as today |
| Secure aggregation, differential privacy | Unchanged (FEDERATED4.md §7) |
| `cuda` in `devices=` | Never run on a GPU (S3.4) |
| Authentication and encryption on the wire | Not built; the backend refuses non-loopback binds without an explicit acknowledgement (§4.2) |
| Sub-interpreter or in-process multi-device "ranks" | One process per rank; same-process ranks would share the GIL and the shim's single-writer assumptions (ASYNCWORK.md §4) |
| Other distributed-inference engines as dependencies | They are comparison baselines to benchmark against later, not parts of this design |
| Cross-device mixed dtype in one collective | Refused with both dtypes named |

---

## 9. Plan, with sizes and the test that decides each step

Sizes are agent estimates in working sessions (S = under a day, M = 1-3 days, L = a week), not measurements.
Every step starts with tests written from the SPEC rows, observed red (AGENTS.md rule 5), and agents do not run
builds (AGENTS.md rule 10): the gate runs in CI or in the coordinating session.

| # | Step | Size | Test (differential against upstream gloo wherever it can be) |
|---|---|---|---|
| 0 | **Probe, no code.** On a built tree: which of `device_mesh`, `tensor.parallel`, `pipelining`, `fsdp`, `_functional_collectives` import and construct at world 1; whether `torch.frombuffer`-style raw export/import is cheap and exact; whether `_make_wrapper_subclass`/`__torch_dispatch__` works on a user subclass; how a Python kernel could be registered under `_c10d_functional`; cost of `.cpu()`/upload on vulkan and mps | S | A transcript recorded in this doc's §10, graded *reaches* only for what ran |
| 1 | **Binary frame and a `Transport` interface**, TCP, any bind address, `TCP_NODELAY`, no `127.0.0.1` literal. Unit-level: round-trip every dtype, `bfloat16` included, bit-exact; a frame split across many `recv`s; a truncated frame raises naming the peer | M | In-process loopback pair; sabotage: flip a header byte, expect a named error. No collective yet |
| 2 | **`hetero` backend v0 = parity with `local`** on the binary wire: the eleven collectives, five reduce ops, async, at world 3 and 4 on loopback | M | `tests/distributed/test_hetero_parity.py`: same harness as `test_collect2.py`, compared with **gloo** and with **`local`**; bit equality for the selecting ops, derived tolerance for reductions (AGREE.md §2) |
| 3 | **Point-to-point**: `send`, `recv`, `isend`, `irecv`, `batch_isend_irecv`, `send_object_list`/`recv_object_list` over the full mesh | M | Differential vs gloo p2p at world 2 and 4 (ring and non-neighbour pairs, tag matching, out-of-order arrival); `local` still refuses and its refusal test is untouched |
| 4 | **LAN and rendezvous**: store on a non-loopback address; two hosts (or two network namespaces/interfaces on one) run a world of 3; per-op timeouts from the group; a killed rank names itself on the survivors | M | Two-interface test on one host; a real two-machine run is a **device test the coordinating session runs itself** (AGENTS.md §10), reported as such, never claimed from CI |
| 5 | **Device staging**: vulkan and mps tensors through all collectives with counted H2D/D2H; refusals name the device and op; `devices=` set accordingly | M | vs gloo on the CPU copy of the same data; counter test "one download, one upload per all-reduce"; nullify the staging and the counter test must fail. Needs a Vulkan loader (`TORCHNATIVE_REQUIRE_VULKAN=1` so a missing loader is `UNVERIFIED`, not a pass) |
| 6 | **`new_group` / `split_group`** over persistent connections, and an N-D `DeviceMesh` | M | `init_device_mesh("cpu", (2,2), mesh_dim_names=("dp","tp"))` at world 4: each `get_group()` all-reduces over the right ranks, compared with gloo on the same mesh |
| 7 | **TP inference with a real `transformers` model**: `tp_size=2` then 3 on loopback CPU, explicit `device_mesh`, logits against the single-process model and against upstream torch with gloo TP | L | The existing real-model agreement harness (E2E_REAL.md); tolerance derived; world 1 control must pass unchanged. Then the same with one rank `to("vulkan")` |
| 8 | **PP inference**: p2p stage runner (route B) generating tokens across 2 and 3 ranks; KV cache stays per stage | L | Greedy tokens equal the single-process run, bit for bit; one rank on vulkan/npu where the device exists |
| 9 | **NPU rank**: shard, then `to(device.npu)`, collectives between compiled bodies; eager compile at `to()`, long-timeout barrier | M | On a host with an NPU back end (Intel NPU on Windows is the only one with a recorded run, S5.3, which itself is "partial"): a device test run by the coordinating session. Without an NPU: only the **ordering and timeout logic** is tested, against a faked compile delay, and the report says so |
| 10 | **Training**: TP training step vs upstream gloo (loss and grads); Python gradient reducer for data parallel; `FederatedDataParallel` rides on the same group | L | One optimiser step equals upstream's within a derived tolerance, per S6.1's method; DDP name stays refused unless the reducer is a drop-in |
| 11 | **DTensor tier** (§6.4): `_c10d_functional` kernels, `shard_dim_alltoall`, then `parallelize_module` and FSDP2 | L | Each redistribute vs upstream; `fully_shard` step vs upstream at world 2 |
| 12 | **Capability-weighted placement** and uneven PP, plus a measured latency/bytes table that replaces §7's arithmetic | M | Placement is a pure function, unit tested; the table is a measurement record |

Steps 1-6 are the backend. Steps 7-8 deliver the owner's inference goal. Steps 9-10 deliver npu and training. Step 11 is
the upstream-DTensor-compatible tier and is last because the plain tier already gives a real model on N devices.

---

## 10. Measurements still to record

This section is intentionally empty. Step 0 fills it. Nothing above is to be quoted as measured.

---

## 11. Questions for the owner

Only real trade-offs. Each carries the owner's constraint behind it.

- **Q1. Backend name and the fate of `local`.** The goal says `torch.distributed`-compatible, and `local` is the
  oracle that agrees with gloo. Option A: new backend `hetero`, `local` untouched as the reference (proposed).
  Option B: grow `local` itself (one name, but the reference stops being a reference and every change needs a
  rebuild because it lives in `bootstrap.py`).
- **Q2. Python backend versus Rust.** The proposal is a pure Python package for iteration speed and no rebuild.
  If per-token latency on a phone matters more than iteration, the hot path (framing, ring reduce) would later move to
  `torch_c`. Do you want Python first and measure, or Rust from the start?
- **Q3. Same-host transport.** Unix domain socket and loopback TCP (stdlib, proposed), or shared memory for ranks
  on one machine (faster, more platform-specific, none on Android/iOS app sandboxes without extra work)?
- **Q4. Network security.** LAN TCP is unauthenticated. Option A: loopback-only by default plus explicit opt-in
  for LAN and a pre-shared key using stdlib `hmac` (no dependency, proposed). Option B: no key at all. Option C: TLS.
  A phone and a PC on a home network is the case, so this is a real choice.
- **Q5. Uneven tensor parallelism.** Splitting matrices 5:3:2 means leaving the upstream and transformers even-split
  code paths (our own wrapper or padding). Or: TP only across similar devices, PP for uneven ones (proposed). Which
  matters more to you?
- **Q6. Loading a layer subset for PP.** Real memory savings need each rank to load only its layers. Is it acceptable
  to add a torchnative-side loader option for that (new API surface under `torchnative.transformers`), or must it work
  through stock `from_pretrained` arguments only?
- **Q7. `DistributedDataParallel`.** Drop-in Python `Reducer` under upstream's name, or a differently-named
  torchnative class with upstream's DDP staying refused by name?
- **Q8. Training through a compiled NPU graph.** `to(npu)` compiles the module; is training (backward) on an NPU rank
  in scope, or are NPU ranks inference-only while training ranks are cpu/vulkan/mps?
- **Q9. Where do the planned launch tools live?** A launcher that starts one rank per device across machines would be a
  new entry point. Options: a module inside `torchnative/` (no root change, proposed), or none (the caller starts
  processes and sets `MASTER_ADDR`). Nothing at the repository root either way.

No new dependency is introduced by this design. If Q4 resolves to TLS or Q3 to shared memory beyond the stdlib,
that is a dependency question and returns to you.

---

## 12. Reproducing this reading

```sh
cd torchnative/.worktrees/dist-infer-design
grep -n "ProcessGroupLocal\|_PG_.*TIMEOUT\|127.0.0.1" torchnative/rust/torch_c/src/bootstrap.py
sed -n 52,90p .caches/spike-venv/lib/python3.13/site-packages/transformers/integrations/tensor_parallel.py
```

(the second path is relative to the main checkout, `.caches/` is not in a worktree). Nothing was run against the shim.
