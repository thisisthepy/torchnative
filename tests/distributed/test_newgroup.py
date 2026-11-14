"""`new_group`, `split_group` and an N-D `DeviceMesh` on backend `torchnative`, against gloo.

BACKEND.md section 9 step 8 (issue #221). `docs/distributed/BACKEND.md` said `new_group`
"refuses" on this backend. A reading of `bootstrap.py` found no `new_group` or
`split_group` code at all, so that sentence was a claim about behaviour nobody
had measured. `docs/distributed/NEWGROUP.md` is the reading: upstream's
`new_group` (`distributed_c10d.py:5745`, then `_new_group_with_tag` and
`_new_process_group_helper`) builds a fresh `ProcessGroup` per subgroup, wraps
the default store in `PrefixStore(f"{group_name}/")` and then
`PrefixStore("cpu/")`, and calls this backend's creator with the subgroup's own
rank and size. So each subgroup gets its own star, rendezvoused under its own
key, with no code of ours involved. This file is the measurement that decides
whether that reading is right.

Every value is compared against **upstream's own `torch.distributed` over gloo**
at world 4, on the same inputs (the LCG of `test_collect2.py`, imported, not
copied), at a tolerance derived from upstream's own float32-vs-float64 error
on the same collective (`docs/numerics/AGREE.md` section 2). Collectives that
only move numbers are held to bit equality, because that derivation gives zero
for them.

The checks that do not need the oracle are stated without it as well: a value
of `10 ** rank` summed over a group is a number that names the group's members
exactly (101 is ranks 0 and 2, 1010 is ranks 1 and 3, 1111 is the world), so
a collective that crossed into another group's traffic cannot produce the
right digits by accident.

World 4 only. A 2 x 2 mesh needs exactly four ranks, and the subgroup cases are
written over the same four so that the two halves of the file read one world.

What this file cannot find: anything about a LAN (every socket is loopback),
non-CPU tensors, `use_local_synchronization=True`, and `new_group` on more
than one machine. It also cannot see whether the shim's group names match
upstream's under `TORCH_DISTRIBUTED_USE_TORCHCOMMS=1`, which neither side sets.
"""

import functools

import gloo_loopback

from test_collect2 import (
    _C2_PROBE_SRC,
    _c2_spawn,
    c2_assert_matches,
    c2_exc,
    c2_ok,
)


#: The world every test here runs at. A 2 x 2 mesh fixes it.
NG_WORLD = 4

#: The two disjoint subgroups `new_group` builds, by global rank.
NG_G02 = [0, 2]
NG_G13 = [1, 3]


# ---------------------------------------------------------------------------
# The probes, one source string shared by both sides (test_collect2's reason:
# two copies of a probe are two chances for the inputs to drift). It is
# appended to `_C2_PROBE_SRC`, so `c2_values` is the same function on both.
# ---------------------------------------------------------------------------

_NG_PROBE_SRC = r'''
def ng_run(torch, dist, rank, world, DT, shim):
    out = {}

    def probe(key, fn):
        try:
            out[key] = {"ok": fn()}
        except Exception as exc:
            out[key] = {"exc": "%s: %s" % (type(exc).__name__, exc)}

    def ft(values):
        return torch.tensor(values, dtype=DT)

    # Every rank enters both calls, in the same order: upstream requires it,
    # and the group names ("1", "2") come from a counter that only stays in
    # step if it does.
    g02 = dist.new_group([0, 2])
    g13 = dist.new_group([1, 3])
    mine, other = (g02, g13) if rank in (0, 2) else (g13, g02)
    members = [0, 2] if rank in (0, 2) else [1, 3]
    foreign = 1 if rank in (0, 2) else 0

    NGM = dist.GroupMember.NON_GROUP_MEMBER

    def membership():
        return {
            "g02_is_non_member": g02 is NGM,
            "g13_is_non_member": g13 is NGM,
            "rank_g02": dist.get_rank(g02),
            "size_g02": dist.get_world_size(g02),
            "rank_g13": dist.get_rank(g13),
            "size_g13": dist.get_world_size(g13),
            "ranks_mine": dist.get_process_group_ranks(mine),
            "global_of_group_rank": [dist.get_global_rank(mine, i)
                                     for i in range(2)],
            "group_rank_of_self": dist.get_group_rank(mine, rank),
            "group_name_mine": mine.group_name,
        }
    probe("membership", membership)

    # -- subgroup collectives -------------------------------------------
    for opname in ("SUM", "MAX"):
        def ar(opname=opname):
            t = ft(c2_values(rank, 6, "ng_sub_allreduce"))
            dist.all_reduce(t, op=getattr(dist.ReduceOp, opname), group=mine)
            return t.tolist()
        probe("sub_allreduce_" + opname, ar)

    def ag():
        holders = [torch.zeros(4, dtype=DT) for _ in range(2)]
        dist.all_gather(holders, ft(c2_values(rank, 4, "ng_sub_allgather")),
                        group=mine)
        return [h.tolist() for h in holders]
    probe("sub_all_gather", ag)

    # The root is group rank 1, i.e. global rank 2 or 3: never the global hub
    # and never the subgroup's hub, so "the hub's value wins" cannot pass.
    def bc_group_src():
        t = ft(c2_values(rank, 5, "ng_sub_bcast"))
        dist.broadcast(t, group_src=1, group=mine)
        return t.tolist()
    probe("sub_broadcast_group_src", bc_group_src)

    def bc_global_src():
        t = ft(c2_values(rank, 5, "ng_sub_bcast"))
        dist.broadcast(t, src=members[1], group=mine)
        return t.tolist()
    probe("sub_broadcast_global_src", bc_global_src)

    def bc_foreign_src():
        dist.broadcast(ft([0.0]), src=foreign, group=mine)
        return "ACCEPTED"
    probe("sub_broadcast_foreign_src", bc_foreign_src)

    # -- a group this rank is not in --------------------------------------
    def non_member():
        source = [float(rank) + 0.5] * 3
        t = ft(source)
        returned = dist.all_reduce(t, group=other)
        return {"returned": repr(returned), "input": source,
                "after": t.tolist()}
    probe("non_member_allreduce", non_member)

    # -- interleaving: subgroup, world, subgroup, ... ----------------------
    def interleave():
        seq = []
        for step in range(3):
            a = ft([float(10 ** rank) * (step + 1)])
            dist.all_reduce(a, group=mine)
            seq.append(["sub", a.tolist()])
            w = ft([float(10 ** rank) * (step + 1)])
            dist.all_reduce(w)
            seq.append(["world", w.tolist()])
        return seq
    probe("interleave", interleave)

    # An async collective on the subgroup is still in flight while a
    # synchronous one runs on WORLD.
    def interleave_async():
        a = ft([float(10 ** rank)])
        work = dist.all_reduce(a, group=mine, async_op=True)
        w = ft([float(10 ** rank)])
        dist.all_reduce(w)
        work.wait()
        return {"sub": a.tolist(), "world": w.tolist()}
    probe("interleave_async", interleave_async)

    # -- the default store's add, across ranks ------------------------------
    def store_add():
        store = dist.distributed_c10d._get_default_store()
        store.add("ng_counter", 1)
        dist.barrier()
        total = store.add("ng_counter", 0)
        dist.barrier()
        return total
    probe("store_add", store_add)

    if shim:
      def store_check():
        dist.distributed_c10d._get_default_store().check(["ng_counter"])
        return "ACCEPTED"
      probe("store_check", store_check)

      def store_compare_set():
        dist.distributed_c10d._get_default_store().compare_set(
            "ng_counter", "4", "5")
        return "ACCEPTED"
      probe("store_compare_set", store_compare_set)

    # -- split_group --------------------------------------------------------
    def split():
        dist.distributed_c10d.split_group(split_ranks=[[0, 2], [1, 3]])
        return "ACCEPTED"
    probe("split_group", split)

    # -- an N-D DeviceMesh ---------------------------------------------------
    state = {}

    def mesh_layout():
        from torch.distributed.device_mesh import init_device_mesh
        m = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
        dp, tp = m["dp"].get_group(), m["tp"].get_group()
        state.update(mesh=m, dp=dp, tp=tp)
        return {
            "mesh": m.mesh.tolist(),
            "coordinate": list(m.get_coordinate()),
            "dp_mesh": m["dp"].mesh.tolist(),
            "tp_mesh": m["tp"].mesh.tolist(),
            "dp_ranks": dist.get_process_group_ranks(dp),
            "tp_ranks": dist.get_process_group_ranks(tp),
            "dp_rank": dist.get_rank(dp), "dp_size": dist.get_world_size(dp),
            "tp_rank": dist.get_rank(tp), "tp_size": dist.get_world_size(tp),
            "dp_by_dim_name": dist.get_process_group_ranks(m.get_group("dp")),
            "tp_by_dim_index": dist.get_process_group_ranks(m.get_group(1)),
            "dp_name": dp.group_name, "tp_name": tp.group_name,
        }
    probe("mesh_layout", mesh_layout)

    for dim in ("dp", "tp"):
        def mar(dim=dim):
            t = ft(c2_values(rank, 6, "ng_mesh_" + dim))
            dist.all_reduce(t, group=state[dim])
            return t.tolist()
        probe("mesh_%s_allreduce" % dim, mar)

        def mag(dim=dim):
            holders = [torch.zeros(3, dtype=DT) for _ in range(2)]
            dist.all_gather(holders, ft(c2_values(rank, 3, "ng_mag_" + dim)),
                            group=state[dim])
            return [h.tolist() for h in holders]
        probe("mesh_%s_all_gather" % dim, mag)

    # Every rank is in one dp group and one tp group; alternating them with
    # WORLD is the case where one process holds three stars at once.
    def mesh_interleave():
        seq = []
        for step in range(2):
            for which in ("dp", "tp", "world"):
                t = ft([float(10 ** rank) * (step + 1)])
                if which == "world":
                    dist.all_reduce(t)
                else:
                    dist.all_reduce(t, group=state[which])
                seq.append([which, t.tolist()])
        return seq
    probe("mesh_interleave", mesh_interleave)

    return out
'''


def _ng_shim_src(probe_call):
    return (
        "import json, os, sys, time\n"
        "import torch\n"
        'assert hasattr(torch._C, "_aten_implemented"), (\n'
        '    "this subprocess loaded upstream torch, not the shim")\n'
        "import torch.distributed as dist\n"
        "import torchnative.distributed  # noqa: F401 (registers 'torchnative')\n"
        + _C2_PROBE_SRC + _NG_PROBE_SRC + _NG_BARRIER_SRC
        + r'''
rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
DT = getattr(torch, sys.argv[5])
''' + probe_call[0] + r'''
dist.init_process_group(backend="torchnative",
                        init_method="tcp://127.0.0.1:%d" % port,
                        rank=rank, world_size=world)
out = ''' + probe_call[1] + r'''(torch, dist, rank, world, DT, True)
out["_meta"] = {"shim": True, "rank": dist.get_rank(),
                "world": dist.get_world_size(),
                "backend": dist.get_backend()}
with open(dest, "w") as handle:
    json.dump(out, handle)
'''
    )


def _ng_gloo_src(probe_call):
    return (
        gloo_loopback.child_pin_source()
        + "import json, os, sys, time\n"
        "import torch\n"
        'assert not hasattr(torch._C, "_aten_implemented"), (\n'
        '    "this subprocess loaded the shim, not upstream torch")\n'
        "import torch.distributed as dist\n"
        + _C2_PROBE_SRC + _NG_PROBE_SRC + _NG_BARRIER_SRC
        + r'''
rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
DT = getattr(torch, sys.argv[5])
''' + probe_call[0] + r'''
os.environ["MASTER_ADDR"] = "127.0.0.1"
os.environ["MASTER_PORT"] = str(port)
dist.init_process_group(backend="gloo", rank=rank, world_size=world)
out = ''' + probe_call[1] + r'''(torch, dist, rank, world, DT, False)
out["_meta"] = {"shim": False, "rank": dist.get_rank(),
                "world": dist.get_world_size(),
                "backend": dist.get_backend()}
with open(dest, "w") as handle:
    json.dump(out, handle)
dist.destroy_process_group()
'''
    )


# The store-based barrier `new_group` runs after it builds a group, switched on
# by an environment variable upstream documents (`distributed_c10d.py:1412`).
# It counts arrivals with `store.add`, which is why it has its own world.
_NG_BARRIER_SRC = r'''
def ng_barrier_run(torch, dist, rank, world, DT, shim):
    out = {}
    try:
        g02 = dist.new_group([0, 2])
        g13 = dist.new_group([1, 3])
        mine = g02 if rank in (0, 2) else g13
        a = torch.tensor([float(10 ** rank)], dtype=DT)
        dist.all_reduce(a, group=mine)
        w = torch.tensor([float(10 ** rank)], dtype=DT)
        dist.all_reduce(w)
        out["barrier_run"] = {"ok": {"sub": a.tolist(), "world": w.tolist()}}
    except Exception as exc:
        out["barrier_run"] = {"exc": "%s: %s" % (type(exc).__name__, exc)}
    return out
'''

_NG_MAIN = ("", "ng_run")
_NG_BARRIER = ('os.environ["TORCH_DIST_INIT_BARRIER"] = "1"', "ng_barrier_run")


@functools.cache
def ng_shim():
    return _c2_spawn(_ng_shim_src(_NG_MAIN), NG_WORLD, "float32", True,
                     "ng-shim")


@functools.cache
def ng_gloo32():
    return _c2_spawn(_ng_gloo_src(_NG_MAIN), NG_WORLD, "float32", False,
                     "ng-gloo32")


@functools.cache
def ng_gloo64():
    """Where the tolerance comes from, not a result."""
    return _c2_spawn(_ng_gloo_src(_NG_MAIN), NG_WORLD, "float64", False,
                     "ng-gloo64")


@functools.cache
def ng_barrier_shim():
    # A hang is what this world exists to catch (the barrier used to wait for
    # a count no rank could see), so its deadline is short.
    return _c2_spawn(_ng_shim_src(_NG_BARRIER), NG_WORLD, "float32", True,
                     "ng-barrier-shim", timeout=240)


@functools.cache
def ng_barrier_gloo():
    return _c2_spawn(_ng_gloo_src(_NG_BARRIER), NG_WORLD, "float32", False,
                     "ng-barrier-gloo", timeout=240)


#: `c2_values`, the inputs' generator, evaluated in this process from the very
#: source the ranks run, so an expectation can be computed rather than copied.
_NG_LCG = {}
exec(_C2_PROBE_SRC, _NG_LCG)


def _group_sum(ranks, scale=1):
    return float(sum(10 ** r for r in ranks) * scale)


def _members(rank):
    return NG_G02 if rank in NG_G02 else NG_G13


# ---------------------------------------------------------------------------
# new_group
# ---------------------------------------------------------------------------

def test_new_group_membership_and_rank_translation_match_gloo():
    """`get_rank(group)`, `get_world_size(group)` and the rank maps, on both sides.

    Stated explicitly as well as against gloo: a non-member is handed
    `GroupMember.NON_GROUP_MEMBER` itself and reads -1 for its rank and size in
    that group, and group rank `i` of `[1, 3]` is global rank `[1, 3][i]`.
    The group names must also agree, because `_resolve_process_group` and
    every `DeviceMesh` lookup go through them.
    """
    shim, gloo = ng_shim(), ng_gloo32()
    for rank in range(NG_WORLD):
        ours = c2_ok(shim[rank], "membership")
        theirs = c2_ok(gloo[rank], "membership")
        assert ours == theirs, (rank, ours, theirs)

        in02 = rank in NG_G02
        assert ours["g02_is_non_member"] is (not in02), (rank, ours)
        assert ours["g13_is_non_member"] is in02, (rank, ours)
        assert ours["rank_g02"] == (NG_G02.index(rank) if in02 else -1), ours
        assert ours["size_g02"] == (2 if in02 else -1), ours
        assert ours["rank_g13"] == (-1 if in02 else NG_G13.index(rank)), ours
        assert ours["size_g13"] == (-1 if in02 else 2), ours
        assert ours["ranks_mine"] == _members(rank), ours
        assert ours["global_of_group_rank"] == _members(rank), ours
        assert ours["group_rank_of_self"] == _members(rank).index(rank), ours
        assert ours["group_name_mine"] == ("1" if in02 else "2"), ours


#: The subgroup collectives compared value for value against gloo.
NG_SUB_VALUE_PROBES = (
    "sub_allreduce_SUM", "sub_allreduce_MAX", "sub_all_gather",
    "sub_broadcast_group_src", "sub_broadcast_global_src",
)


def test_subgroup_collectives_match_gloo():
    """all_reduce, all_gather and broadcast on `new_group([0, 2])` and `([1, 3])`.

    One assertion per (probe, rank), with the tolerance derived from
    upstream's own float32-vs-float64 run of that collective. A subgroup that
    reduced over the world, or over the wrong pair, gives different numbers
    and fails here; so does one whose hub is assumed to be global rank 0
    (the `[1, 3]` hub is global rank 1).
    """
    shim, up32, up64 = ng_shim(), ng_gloo32(), ng_gloo64()
    seen = 0
    for rank in range(NG_WORLD):
        for key in NG_SUB_VALUE_PROBES:
            c2_assert_matches(key, NG_WORLD, rank, c2_ok(shim[rank], key),
                              c2_ok(up32[rank], key), c2_ok(up64[rank], key))
            seen += 1
    assert seen == len(NG_SUB_VALUE_PROBES) * NG_WORLD, seen

    # And the gathered slots are the members' own inputs, in group-rank
    # order, computed here from the same LCG rather than read off either side.
    # The inputs are multiples of 2**-12 in [-2, 2), so float32 holds them
    # exactly and Python floats compare equal to what the ranks sent.
    for rank in range(NG_WORLD):
        gathered = c2_ok(shim[rank], "sub_all_gather")
        expected = [_NG_LCG["c2_values"](peer, 4, "ng_sub_allgather")
                    for peer in _members(rank)]
        assert gathered == expected, (rank, gathered, expected)


def test_broadcast_takes_a_group_relative_or_a_global_src_and_refuses_a_foreign_one():
    """`group_src=1` and `src=<its global rank>` name the same root.

    Both must deliver group rank 1's input to both members. A root that is
    not in the group is refused by upstream's own rank translation
    (`get_group_rank`, `distributed_c10d.py:1111`) before any backend sees it,
    on both sides, with the same words.
    """
    shim, gloo = ng_shim(), ng_gloo32()
    for rank in range(NG_WORLD):
        by_group = c2_ok(shim[rank], "sub_broadcast_group_src")
        by_global = c2_ok(shim[rank], "sub_broadcast_global_src")
        assert by_group == by_global, (rank, by_group, by_global)
        root = _members(rank)[1]
        assert by_group == _NG_LCG["c2_values"](root, 5, "ng_sub_bcast"), (
            rank, by_group)
        assert by_group == c2_ok(gloo[rank], "sub_broadcast_group_src"), rank

        ours = c2_exc(shim[rank], "sub_broadcast_foreign_src")
        theirs = c2_exc(gloo[rank], "sub_broadcast_foreign_src")
        foreign = 1 if rank in NG_G02 else 0
        prefix = "ValueError: Global rank %d is not part of group" % foreign
        assert ours.startswith(prefix), ours
        assert theirs.startswith(prefix), theirs


def test_a_non_member_gets_nothing_and_its_buffer_is_untouched():
    """A collective on a group this rank is not in is a warning and a no-op.

    Upstream returns None without touching the tensor (`_rank_not_in_group`).
    The shim must do the same and must not, for instance, run the collective
    on WORLD instead, which would change the buffer.
    """
    shim, gloo = ng_shim(), ng_gloo32()
    for rank in range(NG_WORLD):
        ours = c2_ok(shim[rank], "non_member_allreduce")
        theirs = c2_ok(gloo[rank], "non_member_allreduce")
        assert ours == theirs, (rank, ours, theirs)
        assert ours["returned"] == "None", ours
        assert ours["after"] == ours["input"], ours


def test_interleaved_collectives_on_subgroups_and_world_do_not_cross():
    """Subgroup, WORLD, subgroup, ... with values that name their group.

    `10 ** rank` summed over a group spells its members (101, 1010, 1111), so
    a frame read from the wrong star gives the wrong digits. The async case
    keeps a subgroup collective in flight across a synchronous WORLD one.
    """
    shim, gloo = ng_shim(), ng_gloo32()
    for rank in range(NG_WORLD):
        seq = c2_ok(shim[rank], "interleave")
        expected = []
        for step in range(3):
            expected.append(["sub", [_group_sum(_members(rank), step + 1)]])
            expected.append(["world", [_group_sum(range(NG_WORLD), step + 1)]])
        assert seq == expected, (rank, seq, expected)
        assert seq == c2_ok(gloo[rank], "interleave"), rank

        both = c2_ok(shim[rank], "interleave_async")
        assert both == {"sub": [_group_sum(_members(rank))],
                        "world": [_group_sum(range(NG_WORLD))]}, (rank, both)
        assert both == c2_ok(gloo[rank], "interleave_async"), rank


# ---------------------------------------------------------------------------
# DeviceMesh
# ---------------------------------------------------------------------------

def test_device_mesh_2x2_dp_and_tp_groups_cover_exactly_the_right_ranks():
    """`init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))` at world 4.

    The mesh is `[[0, 1], [2, 3]]`. "dp" runs down a column, so rank r's dp
    group is `[r % 2, r % 2 + 2]`; "tp" runs along a row, so it is
    `[2 * (r // 2), 2 * (r // 2) + 1]`. Stated here, compared with gloo, and
    then exercised: each group's all_reduce and all_gather must equal gloo's
    on the same mesh, and the interleaved identity sums must name the group.
    """
    shim, up32, up64 = ng_shim(), ng_gloo32(), ng_gloo64()
    for rank in range(NG_WORLD):
        ours = c2_ok(shim[rank], "mesh_layout")
        theirs = c2_ok(up32[rank], "mesh_layout")
        assert ours == theirs, (rank, ours, theirs)

        dp = [rank % 2, rank % 2 + 2]
        tp = [2 * (rank // 2), 2 * (rank // 2) + 1]
        assert ours["mesh"] == [[0, 1], [2, 3]], ours
        assert ours["coordinate"] == [rank // 2, rank % 2], ours
        assert ours["dp_mesh"] == dp and ours["dp_ranks"] == dp, ours
        assert ours["tp_mesh"] == tp and ours["tp_ranks"] == tp, ours
        assert ours["dp_by_dim_name"] == dp, ours
        assert ours["tp_by_dim_index"] == tp, ours
        assert (ours["dp_rank"], ours["dp_size"]) == (rank // 2, 2), ours
        assert (ours["tp_rank"], ours["tp_size"]) == (rank % 2, 2), ours

        for key in ("mesh_dp_allreduce", "mesh_tp_allreduce",
                    "mesh_dp_all_gather", "mesh_tp_all_gather"):
            c2_assert_matches(key, NG_WORLD, rank, c2_ok(shim[rank], key),
                              c2_ok(up32[rank], key), c2_ok(up64[rank], key))

        seq = c2_ok(shim[rank], "mesh_interleave")
        expected = []
        for step in range(2):
            expected.append(["dp", [_group_sum(dp, step + 1)]])
            expected.append(["tp", [_group_sum(tp, step + 1)]])
            expected.append(["world", [_group_sum(range(NG_WORLD), step + 1)]])
        assert seq == expected, (rank, seq, expected)
        assert seq == c2_ok(up32[rank], "mesh_interleave"), rank


# ---------------------------------------------------------------------------
# split_group
# ---------------------------------------------------------------------------

def test_split_group_refuses_on_both_sides_with_upstreams_words():
    """`split_group` needs a default group bound to a device; neither side has one.

    `split_group` (`distributed_c10d.py:5517`) refuses before it reaches any
    backend when `default_pg.bound_device_id` is None and TorchComms is off,
    and `init_process_group` will not bind `cpu` (it requires "an accelerator
    with an index"). So on CPU gloo refuses it too, and the shim must refuse
    it with the same message rather than invent a split.
    """
    shim, gloo = ng_shim(), ng_gloo32()
    for rank in range(NG_WORLD):
        ours = c2_exc(shim[rank], "split_group")
        theirs = c2_exc(gloo[rank], "split_group")
        assert ours == theirs, (rank, ours, theirs)
        assert ours.startswith("RuntimeError: No device associated with the "
                               "default pg, not safe to split"), ours


# ---------------------------------------------------------------------------
# The store underneath every group
# ---------------------------------------------------------------------------

def test_the_default_store_add_counts_every_rank():
    """Four ranks each `add(key, 1)`, a barrier, then `add(key, 0)` reads 4.

    `TCPStore.add` used to be inherited from the one-process `Store`, so each
    rank counted in its own dict and read 1, with no error. That is the
    counter `_store_based_barrier` uses after `new_group`.
    """
    shim, gloo = ng_shim(), ng_gloo32()
    for rank in range(NG_WORLD):
        assert c2_ok(gloo[rank], "store_add") == NG_WORLD, rank
        assert c2_ok(shim[rank], "store_add") == NG_WORLD, (
            "rank %d read %r from a counter every rank added 1 to"
            % (rank, c2_ok(shim[rank], "store_add")))


def test_tcp_store_methods_not_on_the_wire_refuse_by_name():
    """`check` and `compare_set` refuse rather than answer from a local dict."""
    shim = ng_shim()
    for rank in range(NG_WORLD):
        for key, name in (("store_check", "TCPStore.check"),
                          ("store_compare_set", "TCPStore.compare_set")):
            msg = c2_exc(shim[rank], key)
            assert "NotImplementedError" in msg, msg
            assert name in msg, msg
            assert "docs/distributed/NEWGROUP.md" in msg, msg


def test_new_group_under_the_init_barrier_finishes_and_agrees():
    """`TORCH_DIST_INIT_BARRIER=1`: init and both `new_group`s barrier on the store.

    Before `TCPStore.add` went over the wire this world could not finish: no
    rank's count ever reached 4, so every rank waited for a "last worker" key
    nobody set, until the spawn's deadline.
    """
    shim, gloo = ng_barrier_shim(), ng_barrier_gloo()
    for rank in range(NG_WORLD):
        ours = c2_ok(shim[rank], "barrier_run")
        assert ours == {"sub": [_group_sum(_members(rank))],
                        "world": [_group_sum(range(NG_WORLD))]}, (rank, ours)
        assert ours == c2_ok(gloo[rank], "barrier_run"), rank


def test_every_rank_really_ran_the_side_it_claims():
    """The premise of every comparison above, checked from the outside."""
    for rank in range(NG_WORLD):
        for spawn in (ng_shim, ng_barrier_shim):
            assert spawn()[rank]["_meta"] == {
                "shim": True, "rank": rank, "world": NG_WORLD,
                "backend": "torchnative"}, spawn()[rank]["_meta"]
        for spawn in (ng_gloo32, ng_gloo64, ng_barrier_gloo):
            assert spawn()[rank]["_meta"] == {
                "shim": False, "rank": rank, "world": NG_WORLD,
                "backend": "gloo"}, spawn()[rank]["_meta"]


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
