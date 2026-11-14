"""DDP's gradient reducer, in Python, under upstream's names.

Issue #247, `docs/distributed/BACKEND.md` D13 (leader, 2026-11-08: "DDP gets a
Python `Reducer` under upstream's name, so user code runs unchanged"). The
design is `docs/distributed/DDP.md`; what follows is the part a reader of this
file needs.

`torch.nn.parallel.DistributedDataParallel` is upstream Python, and it runs
unchanged. What it calls underneath is C++ upstream --
`torch._C._distributed_c10d.Reducer`, `Logger`, `_broadcast_coalesced`,
`_compute_bucket_assignment_by_size`, `_verify_params_across_processes`,
`_register_comm_hook`, `_register_builtin_comm_hook` -- and in this shim those
names were declared and refused. `install()` puts the implementations below in
their place, in `torch._C._distributed_c10d` **and** in `torch.distributed`,
which bound its copies when it was imported (`torch/distributed/__init__.py`);
one object, two names, so `dist.Reducer is torch._C._distributed_c10d.Reducer`.

**Why here and not in `bootstrap.py`.** `bootstrap.py` is baked into the
extension, so every change to a reducer living there is a rebuild. And nothing
is lost by installing it here: the only backend this shim has is `torchnative`,
which `import torchnative.distributed` registers, so no DDP on this shim reaches
a peer without this module having run. Under upstream torch nothing is
installed (`install()` checks for the shim), because upstream's C++ reducer is
real. Agent proposal, not confirmed.

**The gradient, exactly.** Upstream's default path (no comm hook) multiplies
each parameter's *accumulated* `.grad` by `1 / world_size` while copying it into
the bucket, all-reduces the bucket with `SUM`, and copies the bucket back into
`.grad`. This does the same arithmetic, and the backend folds ranks in
ascending order (`ProcessGroupLocal`), so the result is
`sum_r(grad_r * (1/W))` in a fixed bracketing.

**The hook, and why it hands `.grad` to the engine and takes it back.**
Upstream's reducer hooks `AccumulateGrad` and so runs *after* `.grad` holds the
accumulated value. This shim refuses `register_post_accumulate_grad_hook`
(SPEC S6.1), so the reducer uses `Tensor.register_hook` on each parameter, which
runs on the incoming gradient *before* accumulation. The shim's engine
(`_ImperativeEngine.run_backward` in `bootstrap.py`) runs, for each leaf in
turn, that leaf's hooks and then its accumulation. So in the hook for parameter
`p`:

  1. the accumulated value is computed here, `p.grad + incoming` (or
     `incoming`), which is the number `AccumulateGrad` is about to make;
  2. it is scaled into the bucket;
  3. `p.grad` is set to `None` and the caller's tensor is held, so the engine's
     accumulation stores a fresh tensor and does not touch the caller's;
  4. when the **last** parameter's hook runs, every bucket is complete: the
     remaining all-reduces are launched, all are waited, and every other
     parameter's reduced slice is copied into the held tensor (identity kept,
     as upstream's in-place `copy_` keeps it). For the last parameter, whose
     accumulation has not happened yet, `.grad` is zeroed in place and the
     reduced slice is *returned*, so the engine's `add_` leaves exactly the
     reduced value in the same tensor.

Step 3 is what makes the engine's ordering checkable rather than assumed: if a
future engine ran every hook before any accumulation, a parameter's `.grad`
would still be `None` at step 4, and that **refuses by name** instead of adding
the incoming gradient a second time on top of the reduced one.

What is not built refuses by name: `find_unused_parameters=True`,
`gradient_as_bucket_view=True`, `static_graph=True`, sparse gradients, non-CPU
or mixed-device parameters, communication hooks, `join()`, mixed precision and
the compiled (`python_reducer`) path. See `docs/distributed/DDP.md` §4.
"""

from __future__ import annotations

import functools
import os
import weakref

#: Upstream's default runtime-logging sample rate (`kDDPRuntimeLoggingSampleRate`).
_DEFAULT_SAMPLE_RATE = 100

#: Where the refusals point.
_DOC = "docs/distributed/DDP.md"


def _refuse(what, why):
    raise NotImplementedError(
        f"not implemented in torch._C shim: torch._C._distributed_c10d.{what} -- "
        f"{why} ({_DOC})")


def _torch():
    import torch
    return torch


def _dist():
    import torch.distributed as dist
    return dist


def _nbytes(tensor):
    return tensor.numel() * tensor.element_size()


# ---------------------------------------------------------------------------
# Free functions
# ---------------------------------------------------------------------------

def _compute_bucket_assignment_by_size(tensors, bucket_size_limits,
                                       expect_sparse_gradient=(),
                                       tensor_indices=(), logger=None):
    """Upstream's bucketing, `reducer.cpp` `compute_bucket_assignment_by_size`.

    Walk the tensors in the order given; keep one open bucket per
    (dtype, device); add each tensor to its open bucket and close the bucket
    once its byte size reaches the current limit, after which that key moves to
    the next limit in `bucket_size_limits` (the last one repeats). A tensor
    expected to produce a sparse gradient gets a bucket of its own. With
    `tensor_indices` empty the indices are positions and the buckets are sorted
    by their smallest index; with it given (the rebuild, in gradient-ready
    order) the indices are `tensor_indices[i]` and the order is kept.

    Returns `(bucket_indices, per_bucket_size_limits)`.
    """
    tensors = list(tensors)
    limits = [int(v) for v in bucket_size_limits]
    if not limits:
        raise ValueError("_compute_bucket_assignment_by_size: "
                         "bucket_size_limits must not be empty")
    expect_sparse = list(expect_sparse_gradient or ())
    indices = list(tensor_indices or ())
    if expect_sparse and len(expect_sparse) != len(tensors) and not indices:
        raise ValueError("expect_sparse_gradient must be empty or have one "
                         "entry per tensor")
    result = []
    open_buckets = {}
    limit_at = {}
    for i, tensor in enumerate(tensors):
        index = indices[i] if indices else i
        if expect_sparse and expect_sparse[index]:
            result.append(([index], 0))
            continue
        key = (str(tensor.dtype), str(tensor.device))
        bucket = open_buckets.setdefault(key, {"indices": [], "size": 0,
                                               "limit": 0})
        bucket["indices"].append(index)
        bucket["size"] += _nbytes(tensor)
        at = limit_at.setdefault(key, 0)
        bucket["limit"] = limits[at]
        if bucket["size"] >= limits[at]:
            result.append((bucket["indices"], bucket["limit"]))
            open_buckets[key] = {"indices": [], "size": 0, "limit": 0}
            if at + 1 < len(limits):
                limit_at[key] = at + 1
    for bucket in open_buckets.values():
        if bucket["indices"]:
            result.append((bucket["indices"], bucket["limit"]))
    if not indices:
        result.sort(key=lambda entry: min(entry[0]))
    return [list(b) for b, _ in result], [int(s) for _, s in result]


def _broadcast_coalesced(process_group, tensors, buffer_size, src=0):
    """Every tensor takes `src`'s value, in place, a bucket at a time.

    Upstream flattens the tensors into buckets of at most `buffer_size` bytes
    per dtype, broadcasts each flat buffer, and copies it back. Same here,
    through `torch.distributed.broadcast` on `process_group`, so it goes
    through whichever backend that group registered. `src` is a rank of the
    group. A broadcast moves numbers and rounds none, so every rank ends
    bit-identical to `src`.
    """
    torch = _torch()
    dist = _dist()
    tensors = list(tensors)
    if not tensors or process_group.size() == 1:
        return
    buckets, _ = _compute_bucket_assignment_by_size(tensors, [int(buffer_size)])
    is_src = process_group.rank() == src
    with torch.no_grad():
        for bucket in buckets:
            members = [tensors[i] for i in bucket]
            flat = torch.cat([t.reshape(-1) for t in members])
            dist.broadcast(flat, group=process_group, group_src=src)
            if is_src:
                continue
            offset = 0
            for t in members:
                n = t.numel()
                t.copy_(flat[offset:offset + n].reshape(t.shape))
                offset += n


def _verify_params_across_processes(process_group, tensors, logger=None):
    """Same number of parameters, same sizes and strides, on every rank.

    Upstream all-gathers each rank's parameter metadata and compares it with
    rank 0's before DDP broadcasts anything, because a broadcast into a model
    of another shape is a wrong answer, not an error. Same here, with the
    metadata as one int64 vector per rank.
    """
    torch = _torch()
    dist = _dist()
    if process_group.size() == 1:
        return
    meta = [len(tensors)]
    for t in tensors:
        meta += [t.dim()] + list(t.shape) + list(t.stride())
    world = process_group.size()
    length = torch.tensor([len(meta)], dtype=torch.int64)
    lengths = [torch.zeros(1, dtype=torch.int64) for _ in range(world)]
    dist.all_gather(lengths, length, group=process_group)
    longest = max(int(v.item()) for v in lengths)
    padded = torch.tensor(meta + [-1] * (longest - len(meta)), dtype=torch.int64)
    everyone = [torch.zeros(longest, dtype=torch.int64) for _ in range(world)]
    dist.all_gather(everyone, padded, group=process_group)
    rows = [row.tolist() for row in everyone]
    reference = rows[0]
    for rank, row in enumerate(rows):
        if row[0] != reference[0]:
            message = (f"DDP expects same model across all ranks, but Rank "
                       f"{rank} has {row[0]} params, while rank 0 has "
                       f"inconsistent {reference[0]} params.")
            if logger is not None:
                logger.set_error_and_log(message)
            raise RuntimeError(message)
    for rank, row in enumerate(rows):
        if row != reference:
            message = (f"DDP expects same model across all ranks, but the "
                       f"parameter sizes or strides on rank {rank} do not "
                       f"match rank 0's: {row} vs {reference}")
            if logger is not None:
                logger.set_error_and_log(message)
            raise RuntimeError(message)


def _register_comm_hook(reducer, state, comm_hook):
    _refuse("_register_comm_hook",
            "DDP communication hooks are not built: a hook takes a "
            "dist.GradBucket and returns a torch.futures.Future, and this "
            "shim's Work has no Future. The default path (no hook) is the "
            "built one, and it is upstream's allreduce-then-average")


def _register_builtin_comm_hook(reducer, comm_hook_type):
    _refuse("_register_builtin_comm_hook",
            f"built-in communication hook {comm_hook_type!r} is not built; "
            "the default path (no hook) is")


# ---------------------------------------------------------------------------
# Logger
# ---------------------------------------------------------------------------

class DDPLoggingData:
    """What `Logger._get_ddp_logging_data()` returns: two maps, as upstream's."""

    def __init__(self, strs_map=None, ints_map=None):
        self.strs_map = dict(strs_map or {})
        self.ints_map = dict(ints_map or {})

    def __repr__(self):
        return f"DDPLoggingData(strs_map={self.strs_map}, ints_map={self.ints_map})"


class Logger:
    """`torch._C._distributed_c10d.Logger`: the facts DDP records about itself.

    Construction-time facts (module, world, bucket capacity, parameter count
    and bytes, bucket sizes) and the rebuild facts (`has_rebuilt_buckets`,
    `rebuilt_bucket_sizes`) are recorded under upstream's keys, at upstream's
    moments. Timing stats (`avg_forward_compute_time` and the rest) are not
    built and are absent rather than zero: an invented duration would be a
    wrong number.
    """

    def __init__(self, reducer):
        self._reducer = reducer
        self._data = DDPLoggingData()

    def _bucket_sizes(self):
        return ", ".join(str(b.nbytes) for b in self._reducer._buckets)

    def _bucket_limits(self):
        return ", ".join(str(v) for v in self._reducer._per_bucket_size_limits)

    def set_construction_data_and_log(self, module_name, device_ids,
                                      output_device, broadcast_buffers,
                                      has_sync_bn, static_graph):
        reducer = self._reducer
        strs, ints = self._data.strs_map, self._data.ints_map
        params = reducer._params
        strs["module_name"] = str(module_name)
        strs["device_ids"] = ", ".join(str(d) for d in device_ids)
        strs["backend_name"] = str(_dist().get_backend(reducer._process_group))
        strs["dtypes"] = ", ".join(sorted({str(p.dtype).replace("torch.", "")
                                           for p in params}))
        strs["bucket_sizes"] = self._bucket_sizes()
        strs["initial_bucket_size_limits"] = self._bucket_limits()
        ints["output_device"] = int(output_device)
        ints["broadcast_buffers"] = int(bool(broadcast_buffers))
        ints["has_sync_bn"] = int(bool(has_sync_bn))
        ints["static_graph"] = int(bool(static_graph))
        ints["world_size"] = reducer._process_group.size()
        ints["rank"] = reducer._process_group.rank()
        ints["bucket_cap_bytes"] = int(reducer._bucket_bytes_cap)
        ints["find_unused_parameters"] = int(reducer._find_unused_parameters)
        ints["gradient_as_bucket_view"] = int(reducer._gradient_as_bucket_view)
        ints["is_multi_device_module"] = 0
        ints["num_parameter_tensors"] = len(params)
        ints["total_parameter_size_bytes"] = sum(_nbytes(p) for p in params)

    def set_runtime_stats_and_log(self):
        """Upstream records at the sampling iterations only, and the rebuild
        facts once, the first sampled iteration after the rebuild."""
        reducer = self._reducer
        if not reducer._should_collect_runtime_stats():
            return
        ints, strs = self._data.ints_map, self._data.strs_map
        ints["iteration"] = reducer._num_iterations
        if ints.get("has_rebuilt_buckets", 0) != int(reducer._has_rebuilt):
            ints["has_rebuilt_buckets"] = int(reducer._has_rebuilt)
            strs["rebuilt_bucket_sizes"] = self._bucket_sizes()
            strs["rebuilt_per_bucket_size_limits"] = self._bucket_limits()

    def set_error_and_log(self, error):
        self._data.strs_map["error"] = str(error)

    def _get_ddp_logging_data(self):
        return DDPLoggingData(self._data.strs_map, self._data.ints_map)

    def _set_comm_hook_name(self, name):
        self._data.strs_map["comm_hook"] = str(name)

    def _set_static_graph(self):
        self._data.ints_map["static_graph"] = 1

    def _set_uneven_input_join(self):
        _refuse("Logger._set_uneven_input_join",
                "DistributedDataParallel.join() is not built: its join hook "
                "shadows the reducer's collectives through "
                "Reducer._get_zeros_like_grad_buckets and "
                "Reducer._run_comm_hook, which need GradBucket and Future")


# ---------------------------------------------------------------------------
# Reducer
# ---------------------------------------------------------------------------

class _Bucket:
    """One flat all-reduce: which parameters, where each sits, what is pending."""

    __slots__ = ("indices", "offsets", "lengths", "nbytes", "contribs",
                 "pending", "flat", "work")

    def __init__(self, indices, params):
        self.indices = list(indices)
        self.offsets, self.lengths = [], []
        offset = 0
        for i in self.indices:
            n = params[i].numel()
            self.offsets.append(offset)
            self.lengths.append(n)
            offset += n
        self.nbytes = sum(_nbytes(params[i]) for i in self.indices)
        self.reset()

    def reset(self):
        self.contribs = [None] * len(self.indices)
        self.pending = len(self.indices)
        self.flat = None
        self.work = None


def _reducer_hook(reducer_ref, index, grad):
    """The tensor hook on parameter `index`. Holds the reducer weakly, so a
    dropped DDP leaves hooks that do nothing rather than a reducer kept alive
    by its own parameters."""
    reducer = reducer_ref()
    if reducer is None:
        return None
    return reducer._mark_ready(index, grad)


class Reducer:
    """`torch._C._distributed_c10d.Reducer`, the default (no comm hook) path.

    The constructor's arguments are upstream's, positionally, as
    `DistributedDataParallel._ddp_init_helper` passes them. See the module
    docstring for the hook design.
    """

    def __init__(self, params, bucket_indices, per_bucket_size_limits,
                 process_group, expect_sparse_gradients=(),
                 bucket_bytes_cap=25 * 1024 * 1024,
                 find_unused_parameters=False, gradient_as_bucket_view=False,
                 param_to_name_mapping=None,
                 first_bucket_bytes_cap=1024 * 1024,
                 skip_all_reduce_unused_params=False,
                 use_python_reducer=False,
                 bucket_size_limits_for_rebuilding=(),
                 batched_grad_copy=False):
        params = list(params)
        if find_unused_parameters:
            _refuse("Reducer(find_unused_parameters=True)",
                    "finding unused parameters means walking the autograd "
                    "graph from the forward's outputs every iteration and "
                    "all-reducing a used-parameter map; not built. Every "
                    "parameter must receive a gradient each synced iteration")
        if gradient_as_bucket_view:
            _refuse("Reducer(gradient_as_bucket_view=True)",
                    "this reducer copies the reduced bucket back into each "
                    ".grad; making .grad a view into the bucket is not built")
        if use_python_reducer:
            _refuse("Reducer(use_python_reducer=True)",
                    "the compiled-DDP path registers "
                    "register_post_accumulate_grad_hook, which this shim "
                    "refuses (SPEC S6.1)")
        if any(expect_sparse_gradients or ()):
            _refuse("Reducer with sparse gradients",
                    "an embedding with sparse=True gets a bucket of its own "
                    "and a sparse all-reduce; not built")
        if skip_all_reduce_unused_params:
            _refuse("Reducer(skip_all_reduce_unused_params=True)",
                    "it only means something with find_unused_parameters, "
                    "which is not built")
        if not params:
            raise RuntimeError("Reducer: no parameters to reduce")
        devices = {str(p.device) for p in params}
        if devices != {"cpu"}:
            _refuse("Reducer on device(s) %s" % sorted(devices),
                    "backend 'torchnative' registers CPU only and no "
                    "collective has run on another device; a module split "
                    "across devices is a second refusal behind that one")
        self._params = params
        self._process_group = process_group
        self._expect_sparse = list(expect_sparse_gradients or ())
        self._bucket_bytes_cap = int(bucket_bytes_cap)
        self._first_bucket_bytes_cap = int(first_bucket_bytes_cap)
        self._find_unused_parameters = bool(find_unused_parameters)
        self._gradient_as_bucket_view = bool(gradient_as_bucket_view)
        self._names = dict(param_to_name_mapping or {})
        self._limits_for_rebuilding = [int(v) for v in
                                       (bucket_size_limits_for_rebuilding or ())]
        self._per_bucket_size_limits = [int(v) for v in per_bucket_size_limits]
        # Upstream's default path multiplies by 1/div_factor, the world size at
        # construction, while copying each gradient into its bucket.
        self._scale = 1.0 / process_group.size()
        self._logger = None
        self._sample_rate = _DEFAULT_SAMPLE_RATE
        self._num_iterations = 0
        self._has_rebuilt = False
        self._rebuilt_order = []
        self._expect_hooks = False
        self._require_finalize = False
        self._marked = set()
        self._held = {}
        self._next_bucket = 0
        self._buckets = []
        self._locator = {}
        self._initialize_buckets(bucket_indices)
        self._handles = []
        ref = weakref.ref(self)
        for index, p in enumerate(params):
            self._handles.append(
                p.register_hook(functools.partial(_reducer_hook, ref, index)))

    # -- buckets -----------------------------------------------------------
    def _initialize_buckets(self, bucket_indices):
        seen = []
        buckets = []
        for indices in bucket_indices:
            indices = [int(i) for i in indices]
            if not indices:
                raise RuntimeError("Reducer: an empty bucket")
            dtypes = {self._params[i].dtype for i in indices}
            if len(dtypes) != 1:
                raise RuntimeError("Reducer: all parameters in a bucket must "
                                   "have the same dtype, got %s" % sorted(
                                       str(d) for d in dtypes))
            seen += indices
            buckets.append(_Bucket(indices, self._params))
        if sorted(seen) != list(range(len(self._params))):
            raise RuntimeError(
                "Reducer: the bucket indices must name every parameter exactly "
                "once, got %s for %d parameters" % (bucket_indices,
                                                    len(self._params)))
        self._buckets = buckets
        self._locator = {}
        for b, bucket in enumerate(buckets):
            for j, index in enumerate(bucket.indices):
                self._locator[index] = (b, j)

    def _sync_bucket_indices(self, bucket_indices):
        """Rank 0's bucket layout, on every rank, as upstream's
        `sync_bucket_indices` makes it: two broadcasts, the count and then
        the lengths followed by the indices."""
        torch = _torch()
        dist = _dist()
        pg = self._process_group
        if pg.size() == 1:
            return bucket_indices
        count = torch.tensor([len(bucket_indices)], dtype=torch.int64)
        dist.broadcast(count, group=pg, group_src=0)
        nb = int(count.item())
        n = len(self._params)
        if pg.rank() == 0:
            payload = ([len(b) for b in bucket_indices]
                       + [i for b in bucket_indices for i in b])
        else:
            payload = [0] * (nb + n)
        flat = torch.tensor(payload, dtype=torch.int64)
        dist.broadcast(flat, group=pg, group_src=0)
        values = [int(v) for v in flat.tolist()]
        lengths, rest = values[:nb], values[nb:]
        out, at = [], 0
        for length in lengths:
            out.append(rest[at:at + length])
            at += length
        return out

    def _launch_ready_buckets(self):
        """All-reduce, in bucket order, every bucket whose gradients are all in.

        In order and not as each completes, as upstream's reducer does, so
        every rank issues the same collectives in the same sequence whatever
        order its hooks fired in.
        """
        torch = _torch()
        dist = _dist()
        while (self._next_bucket < len(self._buckets)
               and self._buckets[self._next_bucket].pending == 0):
            bucket = self._buckets[self._next_bucket]
            bucket.flat = torch.cat(bucket.contribs)
            bucket.contribs = [None] * len(bucket.indices)
            bucket.work = dist.all_reduce(bucket.flat, group=self._process_group,
                                          async_op=True)
            self._next_bucket += 1

    # -- the hook ----------------------------------------------------------
    def _name(self, index):
        return self._names.get(index, "index %d" % index)

    def _mark_ready(self, index, grad):
        if not self._expect_hooks:
            # `no_sync()`, or a backward after this iteration's finalize:
            # plain local accumulation, as upstream's reducer ignores its hook.
            return None
        torch = _torch()
        if index in self._marked:
            raise RuntimeError(
                "Expected to mark a variable ready only once. This error is "
                "caused by one of the following reasons: 1) Use of a module "
                "parameter outside the `forward` function. 2) Reused "
                "parameters in multiple reentrant backward passes. Parameter "
                "at %s has been marked as ready twice." % self._name(index))
        self._marked.add(index)
        self._require_finalize = True
        if not self._has_rebuilt:
            self._rebuilt_order.append(index)
        p = self._params[index]
        b, j = self._locator[index]
        bucket = self._buckets[b]
        with torch.no_grad():
            old = p.grad
            accumulated = grad if old is None else old + grad
            bucket.contribs[j] = (accumulated * self._scale).contiguous().reshape(-1)
        bucket.pending -= 1
        last = len(self._marked) == len(self._params)
        if not last:
            self._held[index] = old
            p.grad = None
        self._launch_ready_buckets()
        if not last:
            return None
        return self._finalize(index, old)

    def _finalize(self, last_index, last_old):
        """Wait for every bucket, then write each reduced slice into `.grad`."""
        torch = _torch()
        for bucket in self._buckets:
            if bucket.work is None:
                raise RuntimeError(
                    "Reducer: every gradient arrived but a bucket was never "
                    "launched; this is a defect in the reducer")
            bucket.work.wait()
        returned = None
        with torch.no_grad():
            for bucket in self._buckets:
                for j, index in enumerate(bucket.indices):
                    p = self._params[index]
                    off, n = bucket.offsets[j], bucket.lengths[j]
                    reduced = bucket.flat[off:off + n].reshape(p.shape)
                    if index == last_index:
                        returned = reduced.clone()
                        continue
                    current = p.grad
                    if current is None:
                        raise RuntimeError(
                            "not implemented in torch._C shim: the DDP reducer "
                            "expected the engine to have accumulated parameter "
                            "%s before parameter %s's hook ran, and it has not; "
                            "the engine no longer runs each leaf's hooks then "
                            "its accumulation in turn, which this reducer is "
                            "built on (%s §3). Refusing rather than adding the "
                            "gradient twice" % (self._name(index),
                                                self._name(last_index), _DOC))
                    held = self._held.pop(index, None)
                    target = held if held is not None else current
                    target.copy_(reduced)
                    if target is not current:
                        p.grad = target
            if last_old is not None:
                # The engine will `add_` the returned value into this tensor:
                # zero it so the sum is exactly the reduced gradient, in the
                # caller's own tensor. (0 + x is x for every x but -0.0.)
                last_old.zero_()
        self._held.clear()
        self._expect_hooks = False
        self._require_finalize = False
        for bucket in self._buckets:
            bucket.reset()
        self._next_bucket = 0
        return returned

    def _ensure_prior_reduction_finished(self):
        if not self._require_finalize:
            return
        torch = _torch()
        missing = [i for i in range(len(self._params)) if i not in self._marked]
        # Give back what the engine left: the hook handed `.grad` to the engine
        # and held the caller's tensor, and the finalize that would have
        # restored it never came. Restore the local accumulation, which is
        # upstream's state at this point too.
        with torch.no_grad():
            for index, held in list(self._held.items()):
                p = self._params[index]
                if held is not None:
                    if p.grad is not None:
                        held.add_(p.grad)
                    p.grad = held
        self._held.clear()
        rank = self._process_group.rank()
        names = ", ".join(self._name(i) for i in missing)
        raise RuntimeError(
            "Expected to have finished reduction in the prior iteration before "
            "starting a new one. This error indicates that your module has "
            "parameters that were not used in producing loss. "
            "find_unused_parameters=True is not built in this shim (%s), so "
            "every parameter DDP wraps has to receive a gradient each synced "
            "iteration. Parameter indices which did not receive grad for rank "
            "%d: %s. Parameters which did not receive grad for rank %d: %s"
            % (_DOC, rank, " ".join(str(i) for i in missing), rank, names))

    # -- what DDP calls -----------------------------------------------------
    def set_logger(self, logger):
        self._logger = weakref.ref(logger)

    def prepare_for_forward(self):
        self._num_iterations += 1

    def prepare_for_backward(self, outputs=()):
        self._ensure_prior_reduction_finished()
        self._expect_hooks = True
        self._marked.clear()
        self._held.clear()
        self._next_bucket = 0
        for bucket in self._buckets:
            bucket.reset()
        if not self._has_rebuilt:
            self._rebuilt_order = []
        self._move_hooks_last()

    def _move_hooks_last(self):
        """Run after every tensor hook the caller registered since.

        Upstream's reducer runs after `AccumulateGrad`, which is after every
        tensor hook on the parameter; a hook registered after DDP was built
        would otherwise run after this one and change a gradient this one has
        already reduced.
        """
        for p, handle in zip(self._params, self._handles):
            hooks = getattr(p, "_backward_hooks", None)
            if hooks is not None and handle.id in hooks and hasattr(
                    hooks, "move_to_end"):
                hooks.move_to_end(handle.id)

    def _rebuild_buckets(self):
        """Re-bucket once, in the order the first synced backward readied the
        gradients, with the capacities DDP asked for. Returns whether it did."""
        self._ensure_prior_reduction_finished()
        if self._has_rebuilt or len(self._rebuilt_order) != len(self._params):
            return False
        limits = (list(self._limits_for_rebuilding)
                  or [self._first_bucket_bytes_cap, self._bucket_bytes_cap])
        order = list(self._rebuilt_order)
        last_small = os.environ.get("DDP_SET_LAST_BUCKET_CAP") == "1"
        if last_small:
            order.reverse()
        indices, sizes = _compute_bucket_assignment_by_size(
            [self._params[i] for i in order], limits, self._expect_sparse, order)
        if last_small:
            indices.reverse()
            sizes.reverse()
        indices = self._sync_bucket_indices(indices)
        self._has_rebuilt = True
        self._rebuilt_order = []
        self._per_bucket_size_limits = sizes
        self._initialize_buckets(indices)
        return True

    def _should_collect_runtime_stats(self):
        n = self._num_iterations
        return n > 0 and (n <= 10 or n % self._sample_rate == 0)

    def _set_ddp_runtime_logging_sample_rate(self, sample_rate):
        self._sample_rate = int(sample_rate)

    def _check_reducer_finalized(self):
        self._ensure_prior_reduction_finished()

    def _remove_autograd_hooks(self):
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __del__(self):
        try:
            self._remove_autograd_hooks()
        except Exception:  # noqa: BLE001 -- interpreter teardown
            pass

    # -- refused: each names what it would need -----------------------------
    def _set_static_graph(self):
        _refuse("Reducer._set_static_graph",
                "static_graph=True delays the first iteration's all-reduce to "
                "a callback queued with Variable._execution_engine."
                "queue_callback, which this shim's engine does not have")

    def _delay_all_reduce(self):
        self._set_static_graph()

    def _get_local_used_map(self):
        _refuse("Reducer._get_local_used_map",
                "the used-parameter map belongs to find_unused_parameters "
                "and join(), neither built")

    def _get_zeros_like_grad_buckets(self):
        _refuse("Reducer._get_zeros_like_grad_buckets",
                "it returns GradBuckets for join()'s shadow collectives; "
                "not built")

    def _run_comm_hook(self, grad_bucket):
        _refuse("Reducer._run_comm_hook", "communication hooks are not built")

    def _set_forward_pass_work_handle(self, work, divide_by_initial_world_size):
        _refuse("Reducer._set_forward_pass_work_handle", "join() is not built")

    def _push_all_rebuilt_params(self):
        _refuse("Reducer._push_all_rebuilt_params", "join() is not built")

    def _install_post_backward_futures(self, futures):
        _refuse("Reducer._install_post_backward_futures",
                "buffer communication hooks need torch.futures; not built")

    def _set_mixed_precision_param_dtype(self, dtype):
        _refuse("Reducer._set_mixed_precision_param_dtype",
                "DDP mixed_precision is not built")

    def _autograd_hook(self, index):
        _refuse("Reducer._autograd_hook",
                "the mixed-precision path fires the reducer by index; not built")

    def _set_optimizer_in_backward(self):
        _refuse("Reducer._set_optimizer_in_backward",
                "optimizer-in-backward rides on a communication hook; not built")

    def _set_sparse_metadata(self, global_unique_ids):
        _refuse("Reducer._set_sparse_metadata", "sparse gradients are not built")

    def _reset_state(self):
        _refuse("Reducer._reset_state",
                "DDP._update_process_group is not built")

    def _update_process_group(self, new_process_group):
        _refuse("Reducer._update_process_group",
                "re-binding DDP to another process group is not built")


# ---------------------------------------------------------------------------
# Installation
# ---------------------------------------------------------------------------

#: Installed in `torch._C._distributed_c10d` and, where `torch.distributed`
#: re-exports the name, there too.
_C10D_NAMES = {
    "Reducer": Reducer,
    "Logger": Logger,
    "DDPLoggingData": DDPLoggingData,
    "_broadcast_coalesced": _broadcast_coalesced,
    "_compute_bucket_assignment_by_size": _compute_bucket_assignment_by_size,
    "_verify_params_across_processes": _verify_params_across_processes,
    "_register_comm_hook": _register_comm_hook,
    "_register_builtin_comm_hook": _register_builtin_comm_hook,
}

#: The subset `torch/distributed/__init__.py` binds from `_distributed_c10d`.
_DIST_NAMES = ("Reducer", "Logger", "_broadcast_coalesced",
               "_compute_bucket_assignment_by_size",
               "_verify_params_across_processes", "_register_comm_hook",
               "_register_builtin_comm_hook")


def install():
    """Put the reducer under upstream's names. Idempotent; a no-op on upstream.

    Returns the names installed, so a caller can see it happened.
    """
    torch = _torch()
    if not hasattr(torch._C, "_aten_implemented"):
        # Upstream torch: its C++ reducer is real, and replacing it would be
        # this module overriding the thing it exists to agree with.
        return ()
    dist = _dist()
    c10d = torch._C._distributed_c10d
    for name, value in _C10D_NAMES.items():
        setattr(c10d, name, value)
    for name in _DIST_NAMES:
        setattr(dist, name, _C10D_NAMES[name])
    return tuple(_C10D_NAMES)
