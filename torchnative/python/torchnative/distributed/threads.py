"""Ranks as threads of one process (BACKEND.md D5, §9 step 6, issue #281).

D5 (leader, 2026-11-08): "A rank is an execution unit bound to one device, not
necessarily an OS process ... Android and iOS: several ranks as threads of the
app process (free-threaded 3.15t, ``inproc`` between them, ``tcp`` to other
machines)". Everything below that is not that sentence is an **agent
proposal, not confirmed**: the module name, the function names, the choice of
a thread-local world, and the ``env://`` binding the thread launcher uses.

The problem. ``torch.distributed`` keeps its default group and every group
table in one process-global object, ``distributed_c10d._world``. N threads
that each call ``init_process_group`` collide: the second raises upstream's
"trying to initialize the default process group twice!". The shim's group
registry (``torch._C._distributed_c10d._register_process_group``) and its
global rank are process-global too, and every rank's default group is named
``"0"``.

The answer here is upstream's own, from its threaded test harness
(``torch/testing/_internal/distributed/multi_threaded_pg.py``,
``ThreadLocalWorld``; ``common_distributed.py``,
``_set_thread_isolation_mode(True)``):

* ``install()`` replaces ``distributed_c10d._world`` with a world that is
  **per thread** (upstream documents ``_world`` as an "experimental extension
  point to override it"), and turns on the shim's thread isolation mode, so
  the group registry and the global rank are per thread as well.
* After that, a thread that calls ``init_process_group`` gets a default group
  of its own, and ``dist.all_reduce(t)`` in that thread uses it. Upstream user
  code inside a rank thread is unchanged.
* Between two rank threads of one process the ``torchnative`` backend's mesh
  picks ``inproc`` by itself (``transport.choose``: the two Store records carry
  the same process token); to a rank in another process it picks ``shm`` (same
  host) or ``tcp``. Nothing in this module chooses a transport.

The alternative not taken: explicit per-thread groups (each thread builds a
``ProcessGroup`` and passes ``group=`` to every call). It needs no global
patch, but ``dist.all_reduce(t)`` with no ``group=`` would still read the one
process-global default group, so upstream code would have to change; D5's
"several ranks per app" would then not run unmodified ``transformers``
tensor parallelism, which reads the default group.

What this does **not** do, each refused or named:

* ``install()`` refuses while the process world already has a default group
  (it would hide that group from the thread that made it).
* ``install()`` refuses when ``_world`` was already replaced by something else
  (for example upstream's own ``ThreadLocalWorld``): two overrides of one
  extension point cannot both hold.
* A thread started *by* a rank thread does not inherit its rank's world. It
  has an empty world of its own, so a collective there fails with upstream's
  "Default process group has not been initialized", which names it.
* ``os.environ`` is one per process, so a rank thread cannot have its own
  ``RANK``. The thread launcher therefore binds ``env://`` per thread
  (``bind_env``) instead of setting variables; code that reads
  ``os.environ["RANK"]`` itself does not see a rank in thread mode.
* On an interpreter with the GIL the threads take turns. Values are still
  right; parallel execution is not, and the tests say ``UNVERIFIED`` by name
  for that claim (BACKEND.md §9 step 6).

Graded *builds*: nothing in this file has been run (AGENTS.md §16).
"""

from __future__ import annotations

import threading
import weakref

__all__ = ["install", "installed", "uninstall", "current_world", "gil_enabled",
           "bind_env", "unbind_env", "bound_env", "ThreadRanksRefused"]


class ThreadRanksRefused(NotImplementedError):
    """Something about ranks as threads that is not supported, named in the message."""


def _refuse(what, why):
    raise ThreadRanksRefused(
        "torchnative.distributed.threads.%s: %s" % (what, why))


def gil_enabled():
    """True when this interpreter runs Python threads one at a time.

    ``sys._is_gil_enabled`` (3.13+) answers after imports, which matters:
    a free-threaded interpreter re-enables the GIL when it imports an
    extension that did not declare it safe without one, and the shim is such
    an extension until proven otherwise. Before 3.13 there is always a GIL.
    """
    import sys
    probe = getattr(sys, "_is_gil_enabled", None)
    return True if probe is None else bool(probe())


# -- the per-thread world ---------------------------------------------------------

def _c10d():
    import torch.distributed.distributed_c10d as c10d
    return c10d


_lock = threading.Lock()
_state = {"original": None, "installed": None}
#: Every per-thread world ever made, for `uninstall`'s check. Weak, so a
#: finished thread's world goes away with its thread-local slot.
_worlds = weakref.WeakSet()


def _world_class():
    c10d = _c10d()

    class _RankWorld(c10d._World):
        """One thread's group tables.

        Upstream's ``_World`` keeps ``pg_map``, ``pg_names``, ``group_count``
        and the rest in **module globals** of ``distributed_c10d`` (its
        properties read them with ``global``), so a second ``_World()`` would
        share them. This subclass keeps its own, as upstream's ``WorldData``
        does; ``pg_config_info`` is inherited and reads them through ``self``.
        """

        def __init__(self):
            super().__init__()
            self._tn_pg_map = {}
            self._tn_pg_names = {}
            self._tn_pg_group_ranks = {}
            self._tn_pg_backend_config = {}
            self._tn_group_count = 0
            self._tn_tags_to_pg = {}
            self._tn_pg_to_tag = {}
            self.thread_name = threading.current_thread().name

        @property
        def pg_map(self):
            return self._tn_pg_map

        @property
        def pg_names(self):
            return self._tn_pg_names

        @property
        def pg_group_ranks(self):
            return self._tn_pg_group_ranks

        @property
        def pg_backend_config(self):
            return self._tn_pg_backend_config

        @property
        def group_count(self):
            return self._tn_group_count

        @group_count.setter
        def group_count(self, value):
            self._tn_group_count = value

        @property
        def tags_to_pg(self):
            return self._tn_tags_to_pg

        @property
        def pg_to_tag(self):
            return self._tn_pg_to_tag

    return _RankWorld


class _ThreadLocalWorld:
    """Stands where ``distributed_c10d._world`` stood; every attribute is the calling thread's.

    ``distributed_c10d`` reads and writes ``_world.<name>`` at call time
    (``global _world``), and ``GroupMember.WORLD`` is a property over
    ``_world.default_pg``, so replacing the module attribute is enough for
    every function there to see the calling thread's tables.
    """

    def __init__(self, world_class):
        object.__setattr__(self, "_tn_class", world_class)
        object.__setattr__(self, "_tn_local", threading.local())

    def _tn_get(self):
        local = object.__getattribute__(self, "_tn_local")
        world = getattr(local, "world", None)
        if world is None:
            world = object.__getattribute__(self, "_tn_class")()
            local.world = world
            _worlds.add(world)
        return world

    def __getattr__(self, name):
        return getattr(self._tn_get(), name)

    def __setattr__(self, name, value):
        setattr(self._tn_get(), name, value)

    def __repr__(self):
        return "<torchnative.distributed.threads world of thread %r>" % (
            threading.current_thread().name,)


def installed():
    """True when ``distributed_c10d._world`` is this module's per-thread world."""
    world = _state["installed"]
    return world is not None and _c10d()._world is world


def install():
    """Make ``torch.distributed``'s world per thread. Idempotent. Returns the world object.

    Call it before starting rank threads (the thread launcher calls it).
    Refuses, by name, when the process world already holds a default group,
    when ``_world`` was replaced by something else, or when the extension has
    no thread isolation mode (an extension built before issue #281).
    """
    import torch
    c10d = _c10d()
    with _lock:
        if installed():
            return _state["installed"]
        current = c10d._world
        if type(current) is not c10d._World:
            _refuse("install",
                    "torch.distributed.distributed_c10d._world is already a %s, "
                    "not the process world. Something else (for example "
                    "upstream's multi_threaded_pg.ThreadLocalWorld) replaced "
                    "it, and two replacements of one extension point cannot "
                    "both hold" % type(current).__name__)
        if current.default_pg is not None:
            _refuse("install",
                    "this process already initialized a default process group "
                    "(rank %d of %d) on its process-wide world. Making the "
                    "world per thread now would hide that group from the "
                    "thread that made it; call destroy_process_group() first, "
                    "or start the rank threads before any init_process_group"
                    % (current.default_pg.rank(), current.default_pg.size()))
        c10d_c = torch._C._distributed_c10d
        try:
            c10d_c._set_thread_isolation_mode(True)
        except NotImplementedError as exc:
            _refuse("install",
                    "torch._C._distributed_c10d._set_thread_isolation_mode is "
                    "not in this build of the extension (%s). It was added with "
                    "issue #281; rebuild the extension" % exc)
        world = _ThreadLocalWorld(_world_class())
        _state["original"] = current
        _state["installed"] = world
        c10d._world = world
        return world


def uninstall():
    """Put the process world back. Refuses while any thread still holds a default group."""
    import torch
    c10d = _c10d()
    with _lock:
        if not installed():
            return
        holding = [w.thread_name for w in list(_worlds)
                   if w.default_pg is not None]
        if holding:
            _refuse("uninstall",
                    "rank threads %s still hold a default process group; each "
                    "must call destroy_process_group() first" % sorted(holding))
        c10d._world = _state["original"]
        _state["installed"] = None
        _state["original"] = None
        torch._C._distributed_c10d._set_thread_isolation_mode(False)


def current_world():
    """The calling thread's world object (its group tables). Requires ``install()``."""
    if not installed():
        _refuse("current_world", "install() has not been called, so there is "
                "one world for the whole process, not one per thread")
    return _state["installed"]._tn_get()


# -- env:// for rank threads ----------------------------------------------------------
#
# Agent proposal, not confirmed. `init_process_group()` with no arguments is
# `env://`, which reads RANK, WORLD_SIZE, MASTER_ADDR and MASTER_PORT from
# `os.environ`, one per process. The thread launcher cannot set four different
# RANKs, so it binds a rank to the thread instead and this handler answers
# `env://` from the binding. A thread with no binding is handed to upstream's
# own handler, unchanged.

_env_tls = threading.local()
_env_state = {"original": None}


def bound_env():
    """The calling thread's binding (a dict), or None."""
    return getattr(_env_tls, "binding", None)


def _env_handler(url, timeout=None, **kwargs):
    binding = bound_env()
    original = _env_state["original"]
    if binding is None:
        if timeout is None:
            return original(url, **kwargs)
        return original(url, timeout=timeout, **kwargs)
    return _bound_rendezvous(url, binding)


def _bound_rendezvous(url, binding):
    from urllib.parse import urlparse
    query = dict(pair.split("=", 1) for pair in urlparse(url).query.split("&")
                 if "=" in pair)
    rank = int(query.get("rank", binding["rank"]))
    world_size = int(query.get("world_size", binding["world_size"]))
    if rank != binding["rank"] or world_size != binding["world_size"]:
        _refuse("env://",
                "this rank thread was started as rank %d of %d and "
                "init_process_group asked for rank %d of %d; a thread's rank "
                "is fixed by the launcher that started it"
                % (binding["rank"], binding["world_size"], rank, world_size))
    yield (binding["store"], rank, world_size)
    raise RuntimeError("Unable to perform re-rendezvous using env:// method")


def _install_env_handler():
    from torch.distributed import rendezvous as rdzv
    with _lock:
        handlers = rdzv._rendezvous_handlers
        current = handlers.get("env")
        if current is _env_handler:
            return
        if current is None:
            _refuse("bind_env", "torch.distributed.rendezvous has no env:// "
                    "handler to stand in front of")
        _env_state["original"] = current
        handlers["env"] = _env_handler


def bind_env(rank, world_size, store, **extra):
    """Bind ``env://`` in the calling thread to ``rank`` of ``world_size`` over ``store``.

    Used by the thread launcher. ``extra`` (``local_rank``, ``master_addr``,
    ...) is kept on the binding for ``torchnative.distributed.launcher.rank_info``.
    """
    _install_env_handler()
    binding = dict(extra)
    binding.update(rank=int(rank), world_size=int(world_size), store=store)
    _env_tls.binding = binding
    return binding


def unbind_env():
    _env_tls.binding = None
