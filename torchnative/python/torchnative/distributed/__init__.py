"""The `torchnative` backend -- this project's `torch.distributed` transport.

Named after this project, as torch names a backend after its communication
library (gloo, nccl, mpi): ``init_process_group(backend="torchnative")``
(docs/distributed/BACKEND.md D1, SPEC S6.10). It absorbed the backend that
used to be registered as ``local`` (D2, owner 2026-11-08: "이게 더 좋긴 한데");
that name was removed with no alias, so ``backend="local"`` now fails the way
any unregistered name does.

DESIGN.md §11.1 puts four layers in a stack::

    torchnative.federated      rounds, client selection, aggregation
      └ torch.distributed      ProcessGroup, collectives
          └ backends           ours, registered with register_backend
              └ devices        CPU, Metal, Vulkan, NPU

This module is the third layer's registration. The backend it registers is
``torchnative.distributed.backend.ProcessGroupTorchnative`` (BACKEND.md §4.1,
issue #219), a Python subclass of ``torch._C._distributed_c10d.ProcessGroupLocal``:
it keeps that class's collective semantics (the ascending-rank fold through
rank 0, the refusals, ``async_op=True``) but runs them, and point-to-point, over
a full mesh of per-pair transports in binary frames (``backend/collectives.py``,
``backend/transport.py``, ``backend/wire.py``; issue #232). ``ProcessGroupLocal``
itself, with its star of JSON sockets, stays in ``_C`` as the oracle. What has to happen here is the
*registration*, because ``Backend.register_backend`` is an API of
``torch.distributed.distributed_c10d`` and ``_C`` is imported before that file
exists.

Why here rather than in the vendored tree: DESIGN.md §1 forbids a facade, and
IMPORT_TORCH.md records that the vendored tree is not edited. Registering from
``torchnative`` uses the extension point upstream published for exactly this.

**What this backend is, and is not.** This paragraph used to say it was "the
honest world_size-1 case, not a simulation of a larger one", and that it has
not been for some time --- it is corrected here rather than deleted, because a
module docstring understating its own module is the direction a reader cannot
check. ``ProcessGroupLocal`` now runs a **real world of three or more** over
loopback TCP in a star with the hub at rank 0, folding contributions in
ascending rank order so the answer is a property of the ranks and not of who
arrived first. **Eleven collectives** run and agree with upstream gloo at world
3 and 4 --- ``broadcast``, ``all_gather``, ``all_gather_into_tensor``,
``gather``, ``scatter``, ``reduce``, ``reduce_scatter``,
``reduce_scatter_tensor``, ``all_to_all``, ``all_to_all_single``, ``barrier``
--- with ``SUM``/``MIN``/``MAX``/``PRODUCT``/``AVG``
(``docs/distributed/COLLECT2.md``, ``docs/distributed/TRANSPORT.md``). The
world of one still behaves as described --- reductions are the identity,
``broadcast`` and ``barrier`` are no-ops that are *true* rather than
convenient --- it is simply no longer the only world on offer.

What still refuses by name: the bitwise reduce ops ``BAND``/``BOR``/``BXOR``,
``PREMUL_SUM``, and ``recv_anysource``. ``send``/``recv`` used to refuse too,
because the star has no route between two non-hub ranks
(``docs/distributed/TRANSPORT.md`` section 3); the mesh gives every pair its
own route, so they are implemented (SPEC S6.12, ``docs/distributed/P2P.md``).

It is deliberately not ``fake``: upstream's own ``FakeProcessGroup`` docstring
says it "would produce wrong results for every collective", and a wrong result
is worse than a refusal.
"""

from __future__ import annotations

import torch.distributed as dist


#: The name to pass as ``backend=`` to ``torch.distributed.init_process_group``.
BACKEND_NAME = "torchnative"


def _create_backend(dist_backend_opts, backend_options=None):
    """``creator_fn`` for ``Backend.register_backend``.

    Called with ``extended_api=True``, so the first argument is a
    ``_DistributedBackendOptions`` carrying the group's rank and size rather
    than the four positional arguments the narrow API passes.
    """
    from torchnative.distributed.backend import ProcessGroupTorchnative

    rank = getattr(dist_backend_opts, "group_rank", 0)
    size = getattr(dist_backend_opts, "group_size", 1)
    store = getattr(dist_backend_opts, "store", None)
    # The group's own timeout, so a point-to-point wait is bounded by what the
    # caller passed to init_process_group and not by a constant (BACKEND.md
    # §4.2 "Timeouts").
    timeout = getattr(dist_backend_opts, "timeout", None)
    return ProcessGroupTorchnative(rank, size, store, timeout=timeout)


def register() -> str:
    """Register the ``torchnative`` backend, once. Returns its name.

    Idempotent because importing this module calls it and a caller may call it
    again; ``register_backend`` raises on a duplicate name.
    """
    if BACKEND_NAME.upper() not in dist.Backend._plugins:
        dist.Backend.register_backend(
            BACKEND_NAME,
            _create_backend,
            extended_api=True,
            # CPU only. The comment here used to justify that by saying
            # DESIGN.md 11.1's fourth layer -- Metal, Vulkan, NPU -- "is not
            # built", and two of those three now are (see the README's
            # Accelerators row). The restriction stands on its own ground
            # instead: no collective in ProcessGroupLocal has been run on a
            # non-CPU tensor, and claiming a device here would route one onto
            # a device nothing has exercised.
            devices=["cpu"],
        )
    # `init_method="tnserver://host:port"` reaches the swappable server
    # (BACKEND.md D4, §9 step 5) through upstream's own rendezvous extension
    # point; env://, tcp:// and file:// are untouched.
    from torchnative.distributed.backend import server
    server.register_rendezvous_handler()
    return BACKEND_NAME


register()
