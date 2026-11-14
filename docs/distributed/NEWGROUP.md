# NEWGROUP: `new_group`, `split_group` and an N-D `DeviceMesh` on backend `torchnative`

Issue #221, `docs/distributed/BACKEND.md` §9 step 8. Written 2026-11-13 by an agent round.

**Grade: reads.** Nothing in this record ran. Every statement below is a trace through source
(the vendored `torchnative/python/torch/distributed/`, which is upstream 2.13.0 and byte-identical
in `distributed_c10d.py` and `device_mesh.py` to the oracle venv's copy, and `bootstrap.py`). The
only things checked mechanically are that `bootstrap.py` and `tests/distributed/test_newgroup.py`
compile (`python -m py_compile`): **builds**, not a claim about computation (AGENTS.md §16). The
tests that decide every claim here are written and run in CI, not here.

## 0. Answer first

| Question | Answer (read) |
|---|---|
| Does `new_group` refuse on `torchnative`? | **No, and it never did.** No `new_group` or `split_group` code exists in `bootstrap.py`. The documents that said "refuses" cited `bootstrap.py:17907-17924`, which is `send`/`recv`/`recv_anysource` (§1). |
| Does it work? | Expected to, with no code change: upstream's `new_group` builds each subgroup by calling this backend's creator with the subgroup's rank and size and a store prefixed by the group's name, so each subgroup gets its own star (§2). **Unmeasured.** |
| Can two subgroups collide on the rendezvous key `pg_local_port`? | No, by reading: the key a subgroup's star uses is `default_pg/<group_name>/cpu/pg_local_port`, and group names are distinct per group (§4.1). Upstream does the prefixing; no fix was needed. |
| N-D `DeviceMesh` on CPU | Takes `new_group`, never `split_group`, because the CPU default group has no bound device (§3). Its tensor bookkeeping needs `arange`, `as_strided`, `eq`, `nonzero` and indexing on int tensors, which the shim lists but which this round did not run. |
| `split_group` | Refused **by upstream's own Python** on CPU, before any backend is consulted, on gloo too (§5). |
| Defect found | `TCPStore.add` (and `check`, `compare_set`, `delete_key`, `num_keys`, `list_keys`, the queue methods) were inherited from the one-process `Store` and answered from the calling process's own dict. `TORCH_DIST_INIT_BARRIER=1` counts `new_group` arrivals with `add`, so that barrier could never complete. Fixed in `bootstrap.py` (§4.2); **needs a rebuild** to take effect (AGENTS.md §15.2). |

## 1. Where "`new_group` refuses" came from

The sentence appeared in `BACKEND.md` §0, §1.1, §2 and §4.4, in `SPEC.md` S6.12, in `COLLECT2.md`
§3's table and §9, and in `ASYNCWORK.md` §9. The citation in `BACKEND.md` §1.1 was
`bootstrap.py:17907-17924`, which is the three point-to-point refusals. A search of `bootstrap.py`
for `new_group` and `split_group` finds nothing; `ProcessGroup` (`bootstrap.py:16811`) has no
`new_group` method, and no `__getattr__` that could supply one. So the behaviour was simply
unmeasured, and the reading below is what upstream does when it reaches this backend.

This is the shape AGENTS.md §18 names: a promised refusal that nobody had checked. Here the
direction is the benign one (the thing may work rather than silently fail), but a reader of the old
text would not have looked either way.

## 2. `new_group`, step by step

Line numbers are in the vendored `distributed_c10d.py` unless stated.

| Step | Upstream does | What the shim provides |
|---|---|---|
| `new_group` (`:5745`) | If `_use_torchcomms_enabled()` (`:5836`), delegates to `split_group`; otherwise `_new_group_with_tag` | TorchComms is off unless `TORCH_DISTRIBUTED_USE_TORCHCOMMS=1`; nothing in the shim sets it |
| `_new_group_with_tag` (`:5943`) | Reads `default_pg.bound_device_id` (None on CPU), `_world.pg_map[default_pg]` for the backend name and the default store, validates and sorts `ranks`, computes `group_rank` (None for a non-member) | `ProcessGroup.bound_device_id = None` (`bootstrap.py:16811` onward) |
| group name (`:6026`, `_process_group_name` `:5486`) | `str(_world.group_count)`, incremented on **every** rank, member or not, so names stay in step when all ranks call in the same order (upstream's documented requirement) | Pure Python, nothing of ours |
| delegation (`:6032`) | `if hasattr(default_pg, "new_group")`, call it instead | The shim's `ProcessGroup` has no `new_group`, so this branch is not taken |
| `_new_process_group_helper` (`:1984`) | Non-member: returns `(GroupMember.NON_GROUP_MEMBER, None)` (`:2058`) before touching any backend. Member: `PrefixStore(f"{group_name}/", store)` (`:2060`), `ProcessGroup(prefix_store, group_rank, group_size)`, `_set_default_backend(CUSTOM)`, then per device `PrefixStore(f"{device}/", prefix_store)` (`:2101`) and, for a plugin with `extended_api=True`, `creator_fn(dist_backend_opts, backend_options)` (`:2259`) with `group_rank`, `group_size`, `store`, `group_id`, `global_ranks_in_group` | `ProcessGroup(store, rank, size, options)`, `_set_default_backend`, `_register_backend`, `_set_group_name`, `_set_group_desc`, `group_name`; `PrefixStore` (`bootstrap.py:16349`); `torchnative.distributed._create_backend` reads `group_rank`, `group_size`, `store` and builds `ProcessGroupLocal(rank, size, store)` |
| the subgroup's backend | `ProcessGroupLocal.__init__` runs `_rendezvous` (`bootstrap.py:17012`) for `size > 1`: group rank 0 binds an ephemeral loopback port and publishes it under `pg_local_port` in the store it was handed; the others wait on that key and connect | A **new star per subgroup**, hub = the subgroup's rank 0 (global rank 1 for `[1, 3]`). Separate sockets, separate async worker thread |
| registration | `_register_pg_in_world` (`:6822`) fills `pg_map`, `pg_names`, tags, and calls `_register_process_group(group_name, pg)`; `_world.pg_group_ranks[pg] = {global: group}` (`:6071`) | `_register_process_group` / `_resolve_process_group` (`bootstrap.py`, the module registry) |
| optional barrier (`:6075`) | Only if `TORCH_DIST_INIT_BARRIER=1`: `_store_based_barrier` on the **default** store over the whole world, counting with `store.add` | See §4.2: this is the defect |

After that, collectives are upstream's Python: `get_rank(group)` and `get_world_size(group)` read
`_world.pg_group_ranks`; a non-member gets -1 and every collective on that group is a warning and a
return (`_rank_not_in_group`); `broadcast(src=...)` and `broadcast(group_src=...)` both become
`opts.rootRank = <group rank>` through `_canonicalize_group_rank`, and a `src` outside the group is
refused by `get_group_rank` (`:1111`) before any backend sees it. The backend receives group-relative
ranks only, and `ProcessGroupLocal` validates `rootRank` against its own (group) size.

**Rendezvous order cannot deadlock**, by reading: every rank creates groups in the same order, a
non-member returns before any socket work, and a group's rendezvous waits only for its own members.
The earliest group not yet built always has every member able to reach it, so by induction each
completes. A 2 x 2 mesh traced by hand (`[0,2]`, `[1,3]`, `[0,1]`, `[2,3]`) has no cycle.

**Interleaving cannot cross traffic**, by reading: WORLD and each subgroup are separate
`ProcessGroupLocal` instances with separate sockets, so a frame sent on one star is never read from
another. The async worker thread is per instance too.

## 3. `DeviceMesh` on CPU

`init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))` (`device_mesh.py:1498`) builds
`torch.arange(4, dtype=torch.int)` under `with torch.device("cpu")` and constructs a `DeviceMesh`.

- `_setup_world_group_and_device` (`device_mesh.py:438`): the device handle is `torch.cpu`, whose
  `is_initialized()` is upstream Python, so no device is set.
- `_init_one_process_group` (`device_mesh.py:499`): a dimension that spans the whole world reuses the
  default group. Otherwise it calls `split_group` only if the default group has a
  `bound_device_id` or TorchComms is on, **and** an accelerator is available (`:552`); on CPU neither
  holds, so it loops over the dimension's rank lists calling `new_group` (`:586`), keeping the one
  that contains this rank. That is exactly §2, four times for a 2 x 2 mesh.
- Bookkeeping uses `as_strided` (`_mesh_layout.py:400`), `==`, `nonzero`, indexing and `tolist` on
  int tensors; `mesh["dp"]` imports `torch.fx.experimental.proxy_tensor` and
  `torch._subclasses.fake_tensor`. The ops are in the shim's tables; that they run here is
  **unmeasured**.
- `get_group` resolves the stored group name through `_resolve_process_group`, which the shim
  provides.

So no mesh-specific code is needed in the shim, by reading.

## 4. The store

### 4.1 No key collision between subgroups

`init_process_group` wraps the rendezvous store as `PrefixStore("default_pg", store)` (`:1897`), and
that is the store `_world.pg_map` hands to `new_group`. The helper adds `f"{group_name}/"` and then
`"cpu/"`. The shim's `PrefixStore._key` (`bootstrap.py:16362`) joins with exactly one `/`, so the key
a subgroup's hub writes is `default_pg/<group_name>/cpu/pg_local_port`: `default_pg/0/...` for
WORLD, `default_pg/1/...` and `default_pg/2/...` for the first two `new_group`s. Group names are
distinct (§2), so the keys are. Upstream prefixes; no change was needed.

### 4.2 `TCPStore.add` and the methods that were not on the wire

`TCPStore` (`bootstrap.py:16183`) sends `set`, `get` and `wait` to the server on the master. Every
other method was inherited from `Store`, the one-process dict. Across processes that is a wrong
answer with no error: four ranks each calling `add(key, 1)` read 1. `_store_based_barrier` (`:1027`)
counts arrivals that way and then waits for a `:last_worker` key that only the rank seeing the full
count sets, so with `TORCH_DIST_INIT_BARRIER=1` both `init_process_group` and every `new_group`
would wait until their timeout.

Fix (`bootstrap.py`, needs a rebuild):

- `add` goes over the wire: a new server command, applied under the server's lock on the master, so
  the count is the world's (`bootstrap.py:16262`, `:16310`).
- `check`, `compare_set`, `delete_key`, `num_keys`, `list_keys`, `queue_push`, `queue_pop` and
  `queue_len` refuse by name (`NotImplementedError`, naming `TCPStore.<method>` and this record)
  instead of answering from the local dict (`bootstrap.py:16318`). Nothing on the `new_group` path
  calls them; upstream's Python under `torch/distributed/*.py` calls only `add` on a store
  (`_store_based_barrier`). `HashStore`, `FileStore` and `PrefixStore` over a `HashStore` keep the
  dict behaviour, which is correct for one process (`test_shim.py`'s store test is unaffected).
- `multi_get` / `multi_set` stay inherited: they loop over `get` / `set`, which are on the wire.

## 5. `split_group`

`split_group` (`:5517`) raises `RuntimeError("No device associated with the default pg, not safe to
split any process groups")` (`:5573`) whenever the default group's `bound_device_id` is unset and
TorchComms is off. `init_process_group` refuses a CPU `device_id` ("must be an accelerator with an
index", `:2016`). So on CPU, gloo included, `split_group` is refused by upstream's Python before any
backend is consulted, and the shim inherits the same words. Making it work would need a device-bound
default group and a backend with `supports_splitting`; neither is in scope for a CPU star.

## 6. Tests written, and what each would catch

`tests/distributed/test_newgroup.py`, world 4, shim against upstream gloo in float32 and float64
(tolerance derived per AGREE.md §2; harness imported from `test_collect2.py`). Each would go red on:

| Test | Red when |
|---|---|
| `test_new_group_membership_and_rank_translation_match_gloo` | a non-member is handed anything but `NON_GROUP_MEMBER`, a rank map or group name differs from gloo's |
| `test_subgroup_collectives_match_gloo` | a subgroup reduces or gathers over the wrong ranks, or its hub is assumed to be global rank 0 |
| `test_broadcast_takes_a_group_relative_or_a_global_src_and_refuses_a_foreign_one` | `group_src` and `src` disagree, the root is not group rank 1, or a foreign `src` is accepted |
| `test_a_non_member_gets_nothing_and_its_buffer_is_untouched` | a collective on a group the rank is not in runs anyway (for example on WORLD) |
| `test_interleaved_collectives_on_subgroups_and_world_do_not_cross` | a frame is read from the wrong star; the `10 ** rank` sums name the members |
| `test_device_mesh_2x2_dp_and_tp_groups_cover_exactly_the_right_ranks` | any mesh coordinate, dp/tp group, rank, size, value or interleaving differs from gloo or from `[[0, 1], [2, 3]]` |
| `test_split_group_refuses_on_both_sides_with_upstreams_words` | the shim splits, or refuses with different words from gloo |
| `test_the_default_store_add_counts_every_rank` | `TCPStore.add` counts locally (the pre-fix shim reads 1, not 4) |
| `test_tcp_store_methods_not_on_the_wire_refuse_by_name` | `check` / `compare_set` answer from a local dict again |
| `test_new_group_under_the_init_barrier_finishes_and_agrees` | the store barrier hangs (the pre-fix shim waits until the 240 s spawn deadline) |
| `test_every_rank_really_ran_the_side_it_claims` | a worker loaded the wrong torch or backend |

Expected red before the fix, by reading: the two `add` tests and the `check`/`compare_set` test.
Expected green with no fix: everything else, if §2 and §3 are right. Nullifications worth running in
CI: make `PrefixStore._key` return `key` unchanged (the subgroup leaves then read WORLD's stale port;
membership and value tests must go red); make `_create_backend` ignore `group_size` and pass the
world size (rendezvous hangs or values differ).

## 7. What this record cannot see

- Whether any of it runs. Every row above is *reads*.
- LAN, non-CPU tensors, `use_local_synchronization=True`, `new_subgroups`, TorchComms, a mesh larger
  than 2 x 2, and more than one machine.
- Upstream's C++ `PrefixStore` joins with `/` unconditionally, so its keys differ from the shim's in
  slash count. Both are internally consistent; nothing compares keys across the two.
- The shim's per-socket timeout (30 s) applies to subgroups as to WORLD: a member that reaches a
  subgroup collective more than 30 s after its hub is reported as gone. Not new, not tested here.

## 8. Reproducing this reading

```sh
grep -n "new_group\|split_group" torchnative/rust/torch_c/src/bootstrap.py      # nothing
sed -n 5745,6100p torchnative/python/torch/distributed/distributed_c10d.py      # new_group path
sed -n 1984,2330p torchnative/python/torch/distributed/distributed_c10d.py      # helper, PrefixStore
sed -n 499,605p torchnative/python/torch/distributed/device_mesh.py             # mesh groups
```
