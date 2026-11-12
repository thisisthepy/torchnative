# TPINFER: tensor-parallel inference of a real `transformers` model on `torchnative`

BACKEND.md §9 step 9, issue #242, SPEC S6.13. Written 2026-11-13 on `feat/dist-tp-inference`, stacked on
PR #218 (backend `local` renamed to `torchnative`; collectives are `ProcessGroupLocal`'s star over loopback,
CPU tensors).

**Grade: builds.** Nothing in this document ran. The test file compiles (`python -m py_compile`, and its three
child-process sources compile as Python); the gate in CI is the first run. Every "works" below is a reading of
source, cited by file and line, and every claim about values waits for that run. Agents do not build or run
suites here (AGENTS.md §10).

---

## 0. Answer first

| Question | Answer (read, not run) |
|---|---|
| Does the documented user line `from_pretrained(path, distributed_config=DistributedConfig(tp_size=N), device_mesh=mesh)` work? | **No, on upstream either.** It gets through `initialize_tensor_parallelism` and then refuses in `check_and_set_device_map` because a `device_map` requires `accelerate`, which is in neither `.caches/spike-venv` nor `.github/scripts/gate-requirements.txt` (§1.1). |
| Is that a shim wall? | No. It is a missing dependency. Adding `accelerate` is a dependency decision, so it is an **open question for the owner** (AGENTS.md §7), not done here. |
| Can transformers' tensor parallelism run without it? | Yes, by reading. The hooks (`apply_tensor_parallelism`) and the per-style shard functions (`shard_tensor`) are plain functions that need no `accelerate`. The test drives them in `from_pretrained`'s order, with about 15 lines of glue (§2). |
| Shim changes needed, by reading | **None found.** No `bootstrap.py` edit, no edit under `torchnative/python/torchnative/`. Every piece the path needs is either upstream Python in the vendored tree or already present in the shim (§1.2-§1.5). If CI shows a wall, it is one of the unverified reads in §4. |
| Tests | `tests/distributed/test_tpinfer.py`, 7 tests, registered `BOTH` in `.github/scripts/gate_suites.py` (§3). |

---

## 1. What was read

### 1.1 The user line refuses for want of `accelerate`, on both sides

transformers 5.15.1 (`.caches/spike-venv/.../transformers/`):

1. `tp_plan="auto"` is not a `from_pretrained` keyword in this version. The path is `distributed_config=`
   (`modeling_utils.py:4121-4168`), handled by `DistributedMixin.prepare_distribute_model`
   (`distributed/mixin.py:151-186`).
2. `prepare_distribute_model` checks `tp_size * fsdp_size == world_size` and calls
   `initialize_tensor_parallelism(tp_plan="auto", tp_size, device_mesh, device_map)`
   (`integrations/tensor_parallel.py:52-90`). With an explicit mesh it takes the `else` branch: reads
   `device_mesh.ndim`, then `device_map = torch.device(f"{device_mesh.device_type}:{int(os.environ['LOCAL_RANK'])}")`
   (`:88`). So `LOCAL_RANK` must be set, and `torch._C._get_accelerator()` (`:65`) is never asked.
3. `modeling_utils.py:4181` then calls `check_and_set_device_map(device_map)`, which turns the device into
   `{"": device}` and refuses at `integrations/accelerate.py:134-138` when `accelerate` is not importable:
   "Using a `device_map`, `tp_plan`, `torch.device` context manager or setting `torch.set_default_device(device)`
   requires `accelerate`."

Nothing in steps 1-3 is torch-specific, so upstream torch over gloo stops at the same line. The test holds the shim
to **the same message, word for word** (`test_the_users_from_pretrained_line_reaches_the_same_wall_on_both_sides`),
which is a real check: a shim that failed in the mesh or in `initialize_tensor_parallelism` would raise a different
message. If `accelerate` is installed on both sides, the same test instead requires the line to run and agree.

<!-- DOCWATCH: symbol-in-file tests/distributed/test_tpinfer.py test_the_users_from_pretrained_line_reaches_the_same_wall_on_both_sides present -->

### 1.2 `shard_and_distribute_module` passes an integer as the device

`integrations/tensor_parallel.py:1544`: `tp_layer.shard_tensor(param, ..., device=rank)` with `rank` an `int`.
On the shim `torch.device(<int>)` refuses by name, because a bare integer means "index n of the current accelerator"
and this build has none (`torchnative/rust/torch_c/src/device.rs:206-219`). This function is reached from
`_move_missing_keys_from_meta_to_device` (`modeling_utils.py:4761-4764`) only for keys missing from the checkpoint,
and the checkpoint here has every key. The checkpoint-loading path uses `device_map[""]` instead
(`core_model_loading.py:1683-1690`), which is `cpu:LOCAL_RANK`; the shim resolves any `cpu:N` to its one CPU, as
upstream does (`device.rs`, `PyDevice::resolve`). The test passes `device="cpu"` for the same reason.

### 1.3 The order is hooks first, then shards

`maybe_distribute_model` (`distributed/mixin.py:188-212`) runs `apply_tensor_parallelism` before any weight is
loaded, and `add_tensor_parallel_hooks_to_module` calls `validate_module` (`tensor_parallel.py:1495`). The loader
then builds `ALL_PARALLEL_STYLES[style].__class__(...)` (a fresh instance, so `ColwiseParallel` with the default
`gather_output=False`) and its `update_module_attributes` shrinks `lm_head.out_features`. Done in the other order,
`colwise_gather_output`'s check `out_features % world == 0` (`:692-700`) would see 32 at tp 3 and refuse. The test
follows transformers' order.

### 1.4 The mesh, on the shim

Upstream `torch/distributed/device_mesh.py` (vendored, read in the main checkout's built tree):

- `init_device_mesh("cpu", (N,), mesh_dim_names=("tp",))` builds a rank map with `torch.arange(..., dtype=torch.int)`
  under `with torch.device("cpu")` (`:1581-1582`), then `DeviceMesh.__init__` (`:212-343`).
- `_setup_world_group_and_device` (`:438-496`): the process group is initialised; `torch.cpu.is_initialized()` is
  always `True` (`torch/cpu/__init__.py:247-251`), so no device is set.
- `_init_one_process_group` (`:499-540`): a 1-D mesh over the whole world with no backend override returns the
  **default group's** name, because `torch.cuda.is_available()` is false. No `new_group`.
- `_resolve_process_group(name)`: the default group was registered by upstream's `_new_process_group_helper`
  (`distributed_c10d.py:6860`), and the shim keeps that registry (`bootstrap.py:18123-18145`).
- `get_local_rank` (`:1169-1211`) is `get_rank(group)`; `mesh` (`:428-431`) is `as_strided` and `reshape` over an
  int tensor, then `(mesh == rank).nonzero()` (`:345-372`).

Unverified by reading: that `as_strided`, `nonzero` and `eq` take the int32 rank map on the shim. These are the
first things CI's run would name if they fail (§4).

### 1.5 The collectives, on the shim

transformers keeps plain tensors (BACKEND.md §2.1). Under `torch.no_grad()` the shim's `_FunctionBase.apply` runs
`forward` directly (`bootstrap.py:13112-13211`, the `not differentiable` branch), so `_AllReduceForward` and
`_AllGather` (`tensor_parallel.py:449-495`) reach `dist.all_reduce(x, group=mesh.get_group())` and
`dist.all_gather(list, x, group=...)` on the default group. `ProcessGroupLocal._allreduce` and `allgather`
(`bootstrap.py:17357-17377`, `:17485-17507`) take any shape through `tolist()` and write the result back with
`copy_`, in place, as the hook expects. World 2 is a valid star (`bootstrap.py:16930-16956` only refuses
`size < 1`). Per forward of a 2-layer Qwen3 the plan issues exactly **4 `all_reduce`** (`o_proj`, `down_proj` per
layer) and **1 `all_gather`** (`lm_head`, `colwise_gather_output`); `q_norm`/`k_norm`
(`replicated_with_grad_allreduce`) only register a backward hook.

---

## 2. The test's path, and what is ours in it

Per rank, in a child process (`tests/distributed/test_tpinfer.py`, `_TP_PROBE_SRC`, one source shared verbatim by the
shim children and the upstream children):

1. `init_process_group("torchnative", init_method="tcp://127.0.0.1:<ephemeral>")` (upstream: `"gloo"`, pinned to
   loopback by `tests/_support/gloo_loopback.py`), `LOCAL_RANK`/`RANK`/`WORLD_SIZE` set.
2. `init_device_mesh("cpu", (N,), mesh_dim_names=("tp",))`, explicitly (BACKEND.md §2.1).
3. The user line, recorded (§1.1).
4. `from_pretrained(path, dtype=...)` (full weights), then `apply_tensor_parallelism(model, None,
   DistributedConfig(tp_size=N), mesh)`, then for every parameter the plan names:
   `ALL_PARALLEL_STYLES[style].__class__(device_mesh=mesh, rank=mesh.get_local_rank(), empty_param=p)
   .shard_tensor(p, device="cpu", dtype=p.dtype)`, set back as a `Parameter`, then `update_module_attributes`.
   **The loop is ours** (about 15 lines); every function it calls is transformers'.
5. One clean forward, counting `all_reduce`/`all_gather` calls; one sabotaged forward (§3).

The model: Qwen3 (the owner's test family, 2026-10-03), `vocab 96, hidden 48, intermediate 60, 2 layers, 12 heads
over 6 kv heads, head_dim 8`, untied. Every sharded dimension divides by 2 and 3. Written once by an upstream child
with `torch.manual_seed(0)` and every parameter drawn from `N(0, 0.3)`, saved to a temporary checkpoint that all
sides load. It is built by `from_config`, so CI needs no weights from the Hub.

---

## 3. The tests, and what would make each fail

| Test | Asserts | Fails if |
|---|---|---|
| `test_world_one_control_single_process_shim_agrees_with_upstream` | single-process shim logits within the bound derived from upstream single f32 vs f64 (AGREE.md §2), constants pinned (8 ulp, 4x) | the shim's Qwen3 forward drifts; anyone widens the floor or factor |
| `test_tp2_..._agrees_with_single_process_and_with_upstream_gloo_tp`, `test_tp3_...` | per rank: shim TP logits within the bound derived from upstream's own f32-vs-f64 **TP** error, against the single-process shim and against upstream TP over gloo; premise that upstream's TP is within that bound of upstream single; shim ranks bit-identical (one fold on the hub, FEDERATED4.md §2) | a collective returns a wrong sum or gather; a shard is cut on the wrong dim; a rank holds different bytes |
| `test_the_weights_are_sharded_and_the_collectives_run_on_every_rank` | mesh facts (local rank, size, `cpu`, `("tp",)`, default group reused); 19 sharded parameters with the exact full and shard shapes; 4 `all_reduce` + 1 `all_gather` per forward; shim's shard table equals upstream's | hooks not installed or weights left whole (a replicated model agrees by value and fails here by count) |
| `test_dropping_one_ranks_all_reduce_result_leaves_the_bound` | the last rank's `dist.all_reduce` runs on a clone and discards the result: its logits must leave the bound, on the shim and on upstream; and the shim's sabotaged logits agree with upstream's sabotaged logits | the collective's result is not on the path (the agreement tests could not see a missing collective) |
| `test_the_users_from_pretrained_line_reaches_the_same_wall_on_both_sides` | §1.1 | the shim stops earlier, or later, than upstream on the documented line |
| `test_every_child_ran_the_torch_it_claims` | every shim child loaded the shim, every oracle child upstream, checked from outside | a child silently imported the other torch (AGENTS.md §15.2) |

Why the sabotage drops the result instead of skipping the call: in the star, a rank that does not send leaves its
peers blocked in the exchange, and the harness would report a timeout, which says nothing about values.

<!-- DOCWATCH: symbol-in-file tests/distributed/test_tpinfer.py test_dropping_one_ranks_all_reduce_result_leaves_the_bound present -->
<!-- DOCWATCH: symbol-in-file tests/distributed/test_tpinfer.py test_the_weights_are_sharded_and_the_collectives_run_on_every_rank present -->

Why the folder is `tests/distributed/` and not `tests/models/`: the suite reuses `test_collect2.py`'s harness
(`_c2_spawn`) and gloo oracle by import, and a suite's own folder is `sys.path[0]` with only `tests/_support/` on
PYTHONPATH (`tests/run.sh:378`), so the harness is importable only from here.

What these tests cannot see (AGENTS.md §17.4): uneven shards, packed and MoE styles, `embedding_rowwise`, a model
whose forward reads the global head count instead of the local one, the TP backward (step 12), any device other
than CPU, and the `from_pretrained` loader's own sharding path (it is behind the `accelerate` wall).

---

## 4. What CI must show before any of this is more than *builds*

1. `test_tpinfer` in the ledger with 7 ok on both runners, and the ok count rising by 7 over the base branch.
2. The shim children get past `device_mesh` (§1.4: `as_strided`/`nonzero`/`eq` on int32, `torch.device("cpu")` as a
   context manager). A failure there names the stage `device_mesh` and its traceback.
3. The user-line test is green with the `accelerate` message on both sides.
4. Then a nullification run by the coordinating session: make `ProcessGroupLocal._allreduce` return without
   `copy_` (through `scripts/vendor/install_shim.sh`, AGENTS.md §17.5) and confirm the tp 2 and tp 3 tests go red.
   Not done here, because nothing here was run.

Open for the owner: whether to add `accelerate` to the gate's requirements so the documented
`from_pretrained(..., distributed_config=..., device_mesh=...)` line runs (agent proposal, not confirmed). Without it,
the line is transformers' refusal on both sides, and the sharding path above is the one that runs.
