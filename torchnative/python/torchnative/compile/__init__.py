"""Hook-free Dynamo compilation engine without PEP 523.

Issue #211, INTENT §5, §7, SPEC S1.8, S4.2:
Runs Dynamo frame conversion and bytecode tracing without PEP 523 set_eval_frame hooks:
1. `Engine`: wraps ConvertFrame with FakeFrame and pure-Python guards evaluation.
2. `Monitor`: sys.monitoring (PY_START, PEP 669) + __code__ trampoline for nested non-inlined functions.
3. `compile`: top-level entry point mirroring torch.compile without PEP 523.
4. `attach_to_module`: in-place forward wrapper for nn.Module after accelerator lowering.
   A root forward that fails to capture, or gets no code (the recompile limit), refuses
   by name (#255, #303, SPEC S1.4); `Engine.skips` records why each frame got no code.
"""

import builtins
import contextlib
import functools
import gc
import inspect
import sys
import types

__all__ = [
    "CompileFallbackWarning",
    "CompileRefused",
    "Engine",
    "FakeFrame",
    "Monitor",
    "SkipRecord",
    "attach_to_module",
    "compile",
]

_shim_initialized = False
_active_engines = []
_all_engines = []
_orig_install_resume_function_global = None
_eval_frame_wired = False


def _ensure_shim_ready():
    """Ensure torch._C Dynamo surfaces, guards, and fills are installed."""
    global _shim_initialized
    if _shim_initialized:
        return
    import torch

    if hasattr(torch._C, "_aten_implemented"):
        from .. import _dynamo_fills, _dynamo_guards, _dynamo_surface

        _dynamo_surface.install(torch._C)
        _dynamo_guards.install(torch._C)
        _dynamo_fills.install(torch._C)
        # Symbolic sizes through the shim's fake tensors (#262): the bridge has to be in
        # place before Dynamo makes its first fake tensor with a SymInt size.
        from .. import _dynamo_symbolic

        _dynamo_symbolic.install()

    import torch._dynamo.config as dc

    # Upstream's defaults (#262, owner 2026-11-13): the first call compiles static;
    # a size that changes on a later call is made symbolic and compiled once for the
    # range the guards accept. `mark_dynamic` makes a dimension symbolic from the start.
    # A Python float argument is lifted into the graph instead of recompiling per
    # value (`specialize_float = False`, #287).
    dc.automatic_dynamic_shapes = True
    dc.assume_static_by_default = True
    dc.specialize_float = False
    _shim_initialized = True


def _symbolic_refusal(exc):
    """The `SymbolicShapeUnsupported` behind a Dynamo error, if that is what it was."""
    from .._dynamo_symbolic import SymbolicShapeUnsupported

    seen = set()
    e = exc
    while e is not None and id(e) not in seen:
        if isinstance(e, SymbolicShapeUnsupported):
            return e
        seen.add(id(e))
        e = e.__cause__ or e.__context__
    return None


class ViewWithoutBase(RuntimeError):
    """An argument is a view whose base the shim cannot return, and it aliases another."""


def _arg_tensors(loc):
    """Tensors among the frame's arguments, one container level deep."""
    import torch

    out = []
    for v in loc.values():
        if isinstance(v, torch.Tensor):
            out.append(v)
        elif isinstance(v, (list, tuple)):
            out.extend(x for x in v if isinstance(x, torch.Tensor))
        elif isinstance(v, dict):
            out.extend(x for x in v.values() if isinstance(x, torch.Tensor))
    return out


def _baseless_views(loc):
    """Argument views whose `_base` the shim refuses, after checking they alias nothing.

    Since #272 a view the shim tracks answers its real `_base` and is not listed
    here. What remains are views it cannot track (one built by `_make_subclass`,
    such as `nn.Parameter(a[:, 1:])`): `_is_view()` answers from the layout and
    `_base` refuses by name. Dynamo asks for the base only to model aliasing between the view
    and other inputs. A view whose storage no other argument shares has nothing to
    model, so it is traced as the tensor it is; one that does share refuses by name.
    """
    tensors = _arg_tensors(loc)
    views = []
    for t in tensors:
        try:
            if not t._is_view():
                continue
            t._base
        except NotImplementedError:
            views.append(t)
    if not views:
        return []

    def storage(t):
        try:
            return t.untyped_storage().data_ptr()
        except Exception:
            return None

    for v in views:
        sv = storage(v)
        for t in tensors:
            if t is not v and sv is not None and storage(t) == sv:
                raise ViewWithoutBase(
                    "torchnative.compile: an argument is a view whose base the shim cannot "
                    "return, and another argument shares its storage; the aliasing between "
                    "them cannot be traced"
                )
    return views


@contextlib.contextmanager
def _as_plain_tensors(views):
    """For the translation only: `_is_view()` answers False on the given tensors.

    Their fakes copy the layout (offset, strides), so the shim's layout rule would call
    each fake a view again and `meta_utils.assert_metadata_eq` would see the two
    disagree. Fakes have no base either, so while such views are being traced a fake
    whose `_base` the shim refuses answers False as well.
    """
    if not views:
        yield
        return
    from torch._subclasses.fake_tensor import FakeTensor

    for v in views:
        v.__dict__["_is_view"] = _not_a_view
    had = "_is_view" in FakeTensor.__dict__
    prev = FakeTensor.__dict__.get("_is_view")
    inner = getattr(FakeTensor, "_is_view")

    def _is_view(self):
        if not inner(self):
            return False
        try:
            self._base
        except NotImplementedError:
            return False
        return True

    FakeTensor._is_view = _is_view
    try:
        yield
    finally:
        for v in views:
            v.__dict__.pop("_is_view", None)
        if had:
            FakeTensor._is_view = prev
        else:
            del FakeTensor._is_view


def _not_a_view():
    return False


def _marked_dynamic(loc):
    """Does any argument tensor carry `torch._dynamo.mark_dynamic` dimensions?"""
    import torch

    for v in loc.values():
        if isinstance(v, torch.Tensor) and getattr(v, "_dynamo_dynamic_indices", None):
            return True
    return False


def _ensure_resume_patch():
    """Patch OutputGraph.install_resume_function_global once globally to wrap resume functions."""
    global _orig_install_resume_function_global
    if _orig_install_resume_function_global is not None:
        return
    import torch._dynamo.output_graph as og

    _orig_install_resume_function_global = og.OutputGraph.install_resume_function_global

    def patched(self_og, name, code, f_globals):
        _orig_install_resume_function_global(self_og, name, code, f_globals)
        val = self_og.global_scope[name]
        eng = _active_engines[-1] if _active_engines else (_all_engines[-1] if _all_engines else None)
        if eng is not None:
            if code.co_freevars:
                factory = val

                def wrapped_factory(closure):
                    return eng.wrap(factory(closure))

                self_og.global_scope[name] = wrapped_factory
            else:
                self_og.global_scope[name] = eng.wrap(val)
            eng.stats["resume_wrapped"] += 1

    og.OutputGraph.install_resume_function_global = patched


def _ensure_eval_frame_wired():
    """Wire eval_frame cache query and reset functions to active engine entries."""
    global _eval_frame_wired
    if _eval_frame_wired:
        return
    import torch._C._dynamo.eval_frame as ef

    tef = sys.modules.get("torch._dynamo.eval_frame")

    def region(code, rid=0):
        for eng in reversed(_active_engines):
            if code in eng.entries:
                return [
                    _CacheEntry(gm, getattr(nf, "__code__", code))
                    for gm, nf in eng.entries.get(code, ())
                ]
        for eng in reversed(_all_engines):
            if code in eng.entries:
                return [
                    _CacheEntry(gm, getattr(nf, "__code__", code))
                    for gm, nf in eng.entries.get(code, ())
                ]
        return []

    def total(code):
        for eng in reversed(_active_engines):
            if code in eng.entries:
                return len(eng.entries.get(code, ()))
        for eng in reversed(_all_engines):
            if code in eng.entries:
                return len(eng.entries.get(code, ()))
        return 0

    def debug_list(code):
        return region(code, 0)

    def reset(code):
        # Upstream's reset_code drops the code's extra state: its cache entries and
        # the frame execution strategy a no-code answer left on it (#303).
        for eng in reversed(_all_engines):
            eng.entries.pop(code, None)
            eng.skip_codes.discard(code)
            eng.skip_reasons.pop(code, None)

    orig_strategy = getattr(ef, "set_code_exec_strategy", None)

    def set_strategy(code, strategy):
        action = getattr(strategy, "cur_action", strategy)
        if action == 1 or getattr(action, "name", None) == "SKIP":
            for eng in reversed(_all_engines):
                eng.skip_codes.add(code)
                eng.skip_reasons.setdefault(
                    code,
                    SkipRecord(
                        getattr(code, "co_qualname", code.co_name),
                        reason="the code was marked SKIP through set_code_exec_strategy "
                        "(for example torch._dynamo.skip_code)",
                        action="SKIP",
                    ),
                )
        if orig_strategy is not None:
            try:
                orig_strategy(code, strategy)
            except Exception:
                pass

    ef._get_cache_entries_for_region = region
    ef._get_total_cache_entry_count = total
    ef._debug_get_cache_entry_list = debug_list
    ef.reset_code = reset
    ef.set_code_exec_strategy = set_strategy

    if tef is not None:
        tef._get_cache_entries_for_region = region
        tef._get_total_cache_entry_count = total
        tef._debug_get_cache_entry_list = debug_list
        tef.reset_code = reset
        tef.set_code_exec_strategy = set_strategy

    _eval_frame_wired = True


class FakeFrame:
    """Stand-in for torch._C._dynamo.eval_frame._PyInterpreterFrame."""

    def __init__(self, code, g, loc, b, closure):
        self.f_code = code
        self.f_globals = g
        self.f_locals = loc
        self.f_builtins = b
        self.closure = closure
        self.f_lasti = 0
        self.f_lineno = code.co_firstlineno
        self.f_back = None


def _eager_backend(gm, example_inputs):
    """Run the captured graph as it is: upstream's `backend="eager"`."""
    return gm.forward


#: Backend names that torchnative cannot honour, with the reason (#274). Each
#: used to map to `_eager_backend`, so the graph ran eagerly on the host under a
#: name that said otherwise.
_REFUSED_BACKENDS = {
    "npu": (
        "there is no NPU graph compiler. The NPU is reached through "
        "`model.to(torchnative.device.npu)`, which lowers the device leaves and "
        "runs them on the NPU"
    ),
    "inductor": "the torch._C shim has no Inductor",
}


def _resolve_backend(backend):
    """Turn a `backend=` argument into a compiler function, or refuse by name.

    `None` and `"eager"` run the graph eagerly, which is upstream's meaning of
    `"eager"`. A name in `_REFUSED_BACKENDS` refuses. Any other string resolves
    through upstream's `lookup_backend`, as `torch.compile` does, so an unknown
    name raises upstream's `InvalidBackend`. A callable is used as given.
    """
    if backend is None or backend == "eager":
        return _eager_backend
    if isinstance(backend, str):
        why = _REFUSED_BACKENDS.get(backend)
        if why is not None:
            raise NotImplementedError(
                f"torchnative.compile: backend={backend!r} is refused: {why}. "
                f"Running the graph eagerly under this name would be a silent "
                f"fallback (SPEC S1.4, #274). Use backend='eager' to run it "
                f"eagerly on purpose."
            )
        from torch._dynamo.backends.registry import lookup_backend

        return lookup_backend(backend)
    return backend


#: The gb_type upstream's `_compile` gives the recompile-limit `Unsupported`
#: (`convert_frame.py`, `raise_unimplemented_cache_limit_exceeded`).
_RECOMPILE_LIMIT_GB_TYPE = "Dynamo recompile limit exceeded"


class SkipRecord:
    """Why Dynamo produced no code for a frame (#303).

    `function` is the frame's qualified name. `reason` is Dynamo's own
    `ConvertFrameReturn.skip_reason`. `action` is the frame action it returned
    (`"SKIP"`, or `"RUN_ONLY"` for the recompile limit). `cause` is the
    suppressed exception, as `"<type>: <first line>"`, when the no-code answer
    came from one (`ConvertFrame` swallows `Unsupported` and, under
    `suppress_errors`, every capture error). For the recompile limit,
    `limit_type` is the config name (`"recompile_limit"` or
    `"accumulated_recompile_limit"`) and `limit` its value.
    """

    __slots__ = ("function", "reason", "action", "apply_to_code", "cause", "limit_type", "limit")

    def __init__(self, function, reason=None, action=None, apply_to_code=True, cause=None,
                 limit_type=None, limit=None):
        self.function = function
        self.reason = reason
        self.action = action
        self.apply_to_code = apply_to_code
        self.cause = cause
        self.limit_type = limit_type
        self.limit = limit

    @property
    def why(self):
        if self.limit_type is not None:
            return (
                f"Dynamo hit torch._dynamo.config.{self.limit_type} ({self.limit}), "
                f"so the frame runs only the graphs it already compiled and "
                f"otherwise eagerly (FrameAction.RUN_ONLY)"
            )
        text = self.reason or "Dynamo returned no code"
        if self.action == "RUN_ONLY":
            text += " (FrameAction.RUN_ONLY, which upstream sets for the recompile limit)"
        if self.cause:
            text += f" ({self.cause})"
        return text

    def __str__(self):
        return f"{self.function}: {self.why}"

    def __repr__(self):
        return f"SkipRecord({str(self)!r})"


def _skip_record(fn, ret, cause):
    """Build a `SkipRecord` from what `convert_frame` returned and the cause it swallowed."""
    import re

    strategy = getattr(ret, "frame_exec_strategy", None)
    cur = getattr(strategy, "cur_action", None)
    action = getattr(cur, "name", None) or (None if cur is None else str(cur))
    text = None
    limit_type = limit = None
    if cause is not None:
        full = str(cause)
        first = full.strip().splitlines()[0] if full.strip() else ""
        text = f"{type(cause).__name__}: {first}" if first else type(cause).__name__
        if getattr(cause, "gb_type", None) == _RECOMPILE_LIMIT_GB_TYPE or action == "RUN_ONLY":
            m = re.search(r"Limit type: (\w+)", full)
            if m:
                limit_type = m.group(1)
                v = re.search(r"currently set to (\d+)", full)
                if v:
                    limit = int(v.group(1))
                else:
                    import torch._dynamo.config as dc

                    limit = getattr(dc, limit_type, None)
    return SkipRecord(
        getattr(fn, "__qualname__", getattr(fn, "__name__", repr(fn))),
        reason=getattr(ret, "skip_reason", None),
        action=action,
        apply_to_code=bool(getattr(ret, "apply_to_code", True)),
        cause=text,
        limit_type=limit_type,
        limit=limit,
    )


class _CauseRecorder:
    """Stands in for `ConvertFrame._inner_convert` to see what it raised (#303).

    `ConvertFrame.__call__` catches the inner converter's exception and, for a
    soft failure (`Unsupported`, which is what the recompile limit raises) or
    under `suppress_errors`, returns a `ConvertFrameReturn` with no code and a
    generic `skip_reason`. The exception is the only place upstream names the
    real reason (and the limit's name and value), so this records it and
    re-raises it unchanged. Every other attribute reads through to the inner
    converter (`ConvertFrame` reads `_inner_convert._box`).
    """

    def __init__(self, inner):
        self._inner = inner
        self.last = None

    def __call__(self, *args, **kwargs):
        try:
            return self._inner(*args, **kwargs)
        except Exception as exc:
            self.last = exc
            raise

    def __getattr__(self, name):
        return getattr(self.__dict__["_inner"], name)


class _CacheEntry:
    """Holds guard_manager and code for eval_frame cache entry queries."""

    def __init__(self, guard_manager, code):
        self.guard_manager = guard_manager
        self.code = code
        self.compile_id = None
        self.trace_annotation = ""


class Engine:
    """Hook-free Dynamo execution engine executing ConvertFrame without set_eval_frame."""

    def __init__(self, backend=None):
        _ensure_shim_ready()
        from torch._dynamo.convert_frame import convert_frame
        from torch._dynamo.hooks import Hooks

        backend = _resolve_backend(backend)

        self.backend = backend
        self.hooks = Hooks()
        self.cb = convert_frame(backend, self.hooks)
        self._causes = None
        if hasattr(self.cb, "_inner_convert"):
            self._causes = _CauseRecorder(self.cb._inner_convert)
            self.cb._inner_convert = self._causes
        self._last_cause = None
        self.entries = {}  # code -> [(guard_manager, newfn)]
        self.frame_state = {}  # code -> frame_state dict
        # Codes Dynamo answered with SKIP or RUN_ONLY for all future frames
        # (`ConvertFrameReturn.apply_to_code`), as upstream's eval_frame keeps them:
        # a cached graph still serves its guards, anything else runs eagerly.
        self.skip_codes = set()
        # Why each frame produced no code (#303), in order; `skip_reasons` keeps the
        # latest record per skipped code so a later call can name it.
        self.skips = []
        self.skip_reasons = {}
        # The no-code check of an attached module's root forward; claimed by the
        # first `run` after `root_frame` sets it, so nested frames never see it.
        self._root_check = None
        self.stats = dict(
            translate=0, hit=0, skip=0, fail=0, resume_wrapped=0, dynamic_refused=0
        )
        # Each entry names the op and the reason a symbolic trace refused (#262); the
        # frame was then compiled with static shapes.
        self.dynamic_refusals = []
        self.static_codes = set()
        self.known = set()
        self.swaps = []  # (fn, orig_code, tramp_code)
        self.translating = 0
        self.debug = False

        _all_engines.append(self)
        _ensure_eval_frame_wired()
        _ensure_resume_patch()

    @contextlib.contextmanager
    def root_frame(self, check):
        """Run the next `run` call as a root frame whose "no code" goes to `check`.

        `check(record)` receives the `SkipRecord` when the root frame gets no
        code (a fresh translation, or a code already skipped and missing every
        cached guard). It may raise; if it returns, the frame runs eagerly.
        The first `run` claims it on entry, so the frames that run inside the
        root (resume functions after a graph break, trampolined helpers, the
        eager run itself) are ordinary frames and never reach `check`.
        """
        self._root_check = check
        try:
            yield
        finally:
            self._root_check = None

    @contextlib.contextmanager
    def unswapped(self):
        """While Dynamo traces, user functions must show their real code (not the trampoline)."""
        for fn, oc, tc in self.swaps:
            try:
                fn.__code__ = oc
            except Exception:
                pass
        self.translating += 1
        try:
            yield
        finally:
            self.translating -= 1
            for fn, oc, tc in self.swaps:
                try:
                    fn.__code__ = tc
                except Exception:
                    pass

    @staticmethod
    def _locals(fn, args, kwargs):
        if inspect.ismethod(fn):
            return Engine._locals(fn.__func__, (fn.__self__,) + tuple(args), kwargs)
        code = getattr(fn, "__code__", None)
        if code is None:
            return {}
        n = code.co_argcount + code.co_kwonlyargcount
        names = code.co_varnames[:n]
        loc = {}
        defaults = getattr(fn, "__defaults__", None) or ()
        for i, a in enumerate(args[: code.co_argcount]):
            loc[names[i]] = a
        idx = n
        if code.co_flags & 0x04:  # CO_VARARGS (*args)
            loc[code.co_varnames[idx]] = tuple(args[code.co_argcount :])
            idx += 1
        kw_rest = {}
        for k, v in kwargs.items():
            if k in names:
                loc[k] = v
            elif code.co_flags & 0x08:  # CO_VARKEYWORDS (**kwargs)
                kw_rest[k] = v
            else:
                loc[k] = v
        if code.co_flags & 0x08:
            loc[code.co_varnames[idx]] = kw_rest
        nd = len(defaults)
        for i, nm in enumerate(names[: code.co_argcount]):
            if nm not in loc:
                loc[nm] = defaults[i - (code.co_argcount - nd)]
        kwdefaults = getattr(fn, "__kwdefaults__", None) or {}
        for nm in names[code.co_argcount :]:
            if nm not in loc and nm in kwdefaults:
                loc[nm] = kwdefaults[nm]
        closure = getattr(fn, "__closure__", None) or ()
        for nm, cell in zip(code.co_freevars, closure):
            loc[nm] = cell.cell_contents
        return loc

    def wrap(self, fn):
        """Wrap callable to check guard cache and execute compiled code or translate."""
        eng = self
        code = getattr(fn, "__code__", None)
        if code is not None:
            self.known.add(code)

        def runner(*args, **kwargs):
            return eng.run(fn, args, kwargs)

        runner.__wrapped__ = fn
        runner._tn_orig = fn
        return runner

    def run(self, fn, args, kwargs):
        root, self._root_check = self._root_check, None  # only this frame is the root
        code = getattr(fn, "__code__", None)
        if code is None:
            return fn(*args, **kwargs)

        self.known.add(code)  # entered through the engine: never trampoline it
        loc = self._locals(fn, args, kwargs)
        ents = self.entries.get(code, ())

        for gm, newfn in ents:
            if self.swaps:
                with self.unswapped():  # guards ID_MATCH G[...].__code__ of inlined helpers
                    ok = gm.check(loc)
            else:
                ok = gm.check(loc)
            if ok:
                self.stats["hit"] += 1
                return newfn(*args, **kwargs)

        if code in self.skip_codes:
            self.stats["skip"] += 1
            if root is not None:
                record = self.skip_reasons.get(code) or SkipRecord(
                    fn.__qualname__, reason="the code is marked SKIP (set_code_exec_strategy)",
                    action="SKIP",
                )
                root(record)
            return fn(*args, **kwargs)

        if self.debug and self.entries.get(code):
            for gm, _ in self.entries[code][:2]:
                print(
                    "[miss]",
                    fn.__name__,
                    str(gm.check_verbose(loc).verbose_code_parts)[-300:],
                    file=sys.__stderr__,
                )

        from .. import _dynamo_fills

        frame = FakeFrame(
            code, fn.__globals__, loc, builtins.__dict__, fn.__closure__ or ()
        )
        fs = self.frame_state.setdefault(code, {})
        self.stats["translate"] += 1

        _active_engines.append(self)
        try:
            try:
                ret = self._translate(frame, fs, static=code in self.static_codes)
            except Exception as exc:
                refusal = _symbolic_refusal(exc)
                if refusal is None or code in self.static_codes:
                    raise
                # A symbolic trace refused by name. A dimension the user marked dynamic
                # cannot be quietly made static: that refusal is theirs to see.
                if _marked_dynamic(loc):
                    raise refusal from exc
                self.stats["dynamic_refused"] += 1
                self.dynamic_refusals.append(f"{fn.__qualname__}: {refusal}")
                self.static_codes.add(code)
                ret = self._translate(frame, fs, static=True)
        finally:
            if _active_engines and _active_engines[-1] is self:
                _active_engines.pop()

        cause, self._last_cause = self._last_cause, None
        gc_res = getattr(ret, "guarded_code", None)
        if gc_res is None:
            self.stats["skip"] += 1
            record = _skip_record(fn, ret, cause)
            self.skips.append(record)
            # Upstream's eval_frame applies the returned strategy to every later
            # frame of this code when `apply_to_code` is set (the default
            # `ConvertFrameReturn` is SKIP; the recompile limit is RUN_ONLY): the
            # frame is not translated again. Cached graphs keep serving their guards.
            if record.apply_to_code and record.action in ("SKIP", "RUN_ONLY"):
                self.skip_codes.add(code)
                self.skip_reasons[code] = record
            if root is not None:
                root(record)
            return fn(*args, **kwargs)

        newfn = types.FunctionType(
            gc_res.code,
            fn.__globals__,
            fn.__name__,
            fn.__defaults__,
            fn.__closure__,
        )
        newfn.__kwdefaults__ = fn.__kwdefaults__
        self.entries.setdefault(code, []).insert(0, (gc_res.guard_manager, newfn))
        return newfn(*args, **kwargs)

    def _translate(self, frame, fs, static=False):
        from .. import _dynamo_fills

        rec = self._causes
        saved = rec.last if rec is not None else None
        if rec is not None:
            rec.last = None
        try:
            with contextlib.ExitStack() as stack:
                stack.enter_context(_as_plain_tensors(_baseless_views(frame.f_locals)))
                if static:
                    import torch._dynamo.config as dc

                    stack.enter_context(dc.patch(automatic_dynamic_shapes=False))
                stack.enter_context(self.unswapped())
                stack.enter_context(_dynamo_fills.compile_leaf_relaxation())
                ret = self.cb(frame, None, self.hooks, fs)
            # What the inner converter raised for this frame and `ConvertFrame`
            # swallowed; read by `run` straight after (#303).
            self._last_cause = rec.last if rec is not None else None
            return ret
        finally:
            if rec is not None:
                rec.last = saved

    def compile_method(self, model):
        """Compile an nn.Module's forward method."""
        fwd = type(model).forward

        def call(*args, **kwargs):
            return self.run(fwd, (model,) + args, kwargs)

        return call


class Monitor:
    """sys.monitoring (PY_START, PEP 669) + __code__ trampoline.

    Documented limits:
    1. Methods on arbitrary non-FunctionType callables are not trampolined.
    2. C-dispatched callables (builtin_function_or_method) are not trampolined.
    3. Generators, coroutines, and async generators (co_flags & (0x20 | 0x80 | 0x100 | 0x200))
       are not trampolined and run eagerly.
    4. Code objects with no live types.FunctionType referrer in gc.get_referrers(code)
       are not trampolined and run eagerly.
    """

    def __init__(self, engine, roots):
        self.e = engine
        self.roots = tuple(roots)
        self.events = 0
        self.swapped = []
        self.nofunc = 0
        self.tool = sys.monitoring.OPTIMIZER_ID
        sys.monitoring.use_tool_id(self.tool, "tn-dynamo")
        sys.monitoring.register_callback(
            self.tool, sys.monitoring.events.PY_START, self.on_start
        )
        sys.monitoring.set_events(self.tool, sys.monitoring.events.PY_START)

    def stop(self):
        try:
            sys.monitoring.set_events(self.tool, 0)
            sys.monitoring.register_callback(
                self.tool, sys.monitoring.events.PY_START, None
            )
            sys.monitoring.free_tool_id(self.tool)
        except Exception:
            pass
        for fn, oc, tc in self.e.swaps:
            try:
                fn.__code__ = oc
            except Exception:
                pass

    def on_start(self, code, offset):
        from torch._dynamo.convert_frame import output_codes

        self.events += 1
        dis = sys.monitoring.DISABLE
        if not code.co_filename.startswith(self.roots):
            return dis
        if self.e.translating:
            return None  # keep event enabled, retry on a later call
        if code in self.e.known or code in output_codes or code in self.e.entries:
            return dis
        if code.co_flags & (0x20 | 0x80 | 0x100 | 0x200):  # generator/coroutine/async gen
            return dis
        self.e.known.add(code)
        fns = [
            o
            for o in gc.get_referrers(code)
            if isinstance(o, types.FunctionType) and o.__code__ is code
        ]
        if not fns:
            self.nofunc += 1
            return dis
        for fn in fns:
            self.swap(fn)
        return dis

    def swap(self, fn):
        eng = self.e
        orig = types.FunctionType(
            fn.__code__,
            fn.__globals__,
            fn.__name__,
            fn.__defaults__,
            fn.__closure__,
        )
        orig.__kwdefaults__ = fn.__kwdefaults__
        key = f"__tn_run_{len(self.swapped)}_{fn.__name__}"

        def run(a, k):
            return eng.run(orig, a, k)

        fn.__globals__[key] = run
        free = fn.__code__.co_freevars
        src = (
            "def mk("
            + ",".join(free)
            + "):\n    def t(*a, **k):\n"
            + "".join(f"        {n}\n" for n in free)
            + f"        return {key}(a, k)\n    return t\n"
        )
        ns = {}
        exec(src, {}, ns)
        tramp = ns["mk"](*[None] * len(free))
        eng.swaps.append((fn, fn.__code__, tramp.__code__))
        fn.__code__ = tramp.__code__
        self.swapped.append(fn.__qualname__)


class CompileRefused(RuntimeError):
    """A capture failure on an attached module, refused by name (#255, SPEC S1.4).

    The message names the forward that failed, the compiled device's resolution
    when there is one, and the cause, which is also chained as `__cause__`.
    """


class CompileFallbackWarning(UserWarning):
    """`torch._dynamo.config.suppress_errors` turned a capture failure into an
    eager run. Named so a caller can filter or escalate exactly this."""


def _is_capture_failure(exc):
    """The exception classes the engine treats as "Dynamo could not capture this".

    The same set the attached forward caught before #255 (class names containing
    `Dynamo`, `Unsupported` or `Internal`); what changed is what happens next.
    """
    name = type(exc).__name__
    return "Dynamo" in name or "Unsupported" in name or "Internal" in name


def _describe_target(model, orig_fwd_fn, resolution):
    where = f"{type(model).__name__}.{getattr(orig_fwd_fn, '__name__', 'forward')}"
    if resolution is not None:
        backend = getattr(resolution, "backend", None)
        unit = getattr(resolution, "unit", None)
        target = (
            f" for {unit} ({backend})" if unit or backend else f" for {resolution!r}"
        )
    else:
        target = ""
    return where, target


def _describe_failure(model, orig_fwd_fn, resolution, exc):
    where, target = _describe_target(model, orig_fwd_fn, resolution)
    return where, (
        f"torchnative.compile: capturing {where}{target} failed with "
        f"{type(exc).__name__}: {exc}"
    )


def _describe_no_code(model, orig_fwd_fn, resolution, record):
    where, target = _describe_target(model, orig_fwd_fn, resolution)
    return where, (
        f"torchnative.compile: Dynamo produced no code for {where}{target}: "
        f"{record.why}"
    )


#: Dynamo's skip reason for a frame that traced no ops at all
#: (`symbolic_convert.py`, `_return` raising `SkipFrame`). For an attached root
#: forward this is the one no-code answer that is not a fallback: the forward
#: does no tensor work, so its eager run puts nothing on the host that the
#: device would have run. The other legitimate skip, "no tensor-related objects
#: found in frame", cannot reach a root forward: its `self` is an `nn.Module`,
#: which `has_tensor_in_frame` counts.
_NOTHING_TO_CAPTURE = "No ops traced for the FX graph"


def _suppress_errors():
    try:
        import torch._dynamo.config as _dynamo_config

        return bool(getattr(_dynamo_config, "suppress_errors", False))
    except Exception:  # noqa: BLE001
        return False


#: The words every device leaf's Dynamo graph break carries, so a reader of
#: `torch._dynamo.utils.counters["graph_break"]` (or `TORCH_LOGS=graph_breaks`)
#: can tell this break from an accidental one.
_DEVICE_LEAF_REASON = "torchnative device leaf"


def _device_leaf_forward(forward, *, unit):
    """`forward` of a lowered device leaf, made opaque to Dynamo capture (#255).

    A lowered leaf (`export.intelnpu._NPULinear` and its subclasses,
    `export.coreml._CoreMLLinear` / `_CoreMLConv2d` / the decoder group) is the
    device's own execution. Its forward is host-side marshalling into a vendor
    runtime: `.tolist()` and `struct.pack` of the ids, the activation as f16
    bytes, a ctypes or CoreML call, the result unpacked into a new tensor.
    None of that is a tensor program Dynamo can or should trace, so tracing
    into it fails (`.tolist()` iterates a fake tensor) or captures nothing
    useful.

    The documented upstream mechanism for "this callable runs as itself, do
    not trace into it" is `torch.compiler.disable` (`torch._dynamo.disable`):
    Dynamo graph-breaks at the call, the call runs eagerly, and capture resumes
    after it. That is what this applies, with a reason naming the leaf's unit.

    This is not a silent fallback. The eager call IS the device kernel: the
    leaf runs on its device exactly as it would without compile, and its own
    refusals and counters (`_compile_counters()`, the CoreML compute plan) are
    untouched. What Dynamo skips is only the host-side glue around the device
    call. The graph break is recorded by name in Dynamo's own counters
    (`_DEVICE_LEAF_REASON` and `unit` are in its message), and a real capture
    failure anywhere else still raises `CompileRefused`.

    Without `torch._dynamo` there is no capture to be opaque to, so `forward`
    is returned unchanged.
    """
    try:
        import torch._dynamo as _dynamo
    except ImportError:
        return forward
    wrapped = _dynamo.disable(
        forward,
        reason=(
            f"{_DEVICE_LEAF_REASON} ({unit}): its forward is the device's own "
            f"execution, host-side marshalling into the vendor runtime, so it "
            f"runs as itself and is not traced"
        ),
    )
    wrapped._torchnative_device_leaf = unit
    return wrapped


def _caller_chose_a_cache(model, kwargs):
    """Did this `generate` call pick its own KV cache (or no cache)?"""
    if kwargs.get("past_key_values") is not None or kwargs.get("cache_params") is not None:
        return True
    if "cache_implementation" in kwargs or kwargs.get("use_cache") is False:
        return True
    gc = kwargs.get("generation_config")
    for cfg in (gc, getattr(model, "generation_config", None)):
        if cfg is not None and (
            getattr(cfg, "cache_implementation", None) is not None
            or getattr(cfg, "use_cache", True) is False
        ):
            return True
    return False


def _install_static_cache_default(model):
    """`generate` on a compiled module uses transformers' `StaticCache` by default.

    Owner, 2026-11-13 ("그렇게 하자", INTENT §5, #263): the KV cache on compiled devices is
    `StaticCache`, preallocated and written at `cache_position`, set up here so user
    code is unchanged. A `DynamicCache` grows by one position per decode step, which is
    a new shape (and a recompile) per step; a preallocated cache keeps decode shapes
    fixed. A call that picks its own cache (`past_key_values`, `cache_implementation`,
    `use_cache=False`, or a generation config that sets one) is passed through untouched.
    Returns "static" when installed, None when the module has no `generate`.
    """
    gen = getattr(model, "generate", None)
    if gen is None or not callable(gen) or getattr(gen, "_tn_static_cache", False):
        return "static" if getattr(gen, "_tn_static_cache", False) else None
    if getattr(model, "generation_config", None) is None:
        return None

    @functools.wraps(gen)
    def generate(*args, **kwargs):
        if not _caller_chose_a_cache(model, kwargs):
            kwargs["cache_implementation"] = "static"
        return gen(*args, **kwargs)

    generate._tn_static_cache = True
    model.generate = generate
    return "static"


def attach_to_module(model, *, backend=None, resolution=None, graph_runs_on=None, **options):
    """Attach torchnative hook-free compile engine to lowered nn.Module.

    A capture failure is never answered by a silent eager run (#255, SPEC S1.4).
    By default it raises `CompileRefused`. With
    `torch._dynamo.config.suppress_errors = True`, upstream's opt-in for the
    same trade, it warns (`CompileFallbackWarning`) naming the forward, the
    device and the cause, and runs the original forward. Either way
    `stats["fail"]` counts it. Graph breaks are not failures: Dynamo resumes
    them inside the engine.

    The same holds when the root forward gets no code without an exception
    (#303): past the recompile limit (RUN_ONLY), or after Dynamo swallowed a
    frame-level `Unsupported` or, under `suppress_errors`, any capture error.
    The refusal names the reason the engine recorded (`Engine.skips`). Only
    the root frame is checked (`Engine.root_frame`): a nested frame Dynamo
    skips, such as a resume function with nothing left to trace, is recorded
    and runs as upstream runs it. A root forward that traces no ops at all
    (`_NOTHING_TO_CAPTURE`) is recorded and not refused.

    `graph_runs_on` is recorded on `model.torchnative_compile` so that the
    record names where the captured graph executes; `.to(device.npu)` says
    the host, because the graph between the lowered leaves runs eagerly
    there (#274).
    """
    _ensure_shim_ready()
    if hasattr(model, "torchnative_compile") and model.torchnative_compile.get("compiled"):
        return model

    if isinstance(backend, Engine):
        eng = backend
    else:
        eng = Engine(backend)

    orig_fwd_fn = type(model).forward

    def no_code(record):
        # The root forward got no code (#303): the recompile limit, a frame-level
        # `Unsupported` Dynamo swallowed, or an error `suppress_errors` swallowed.
        if _NOTHING_TO_CAPTURE in (record.reason or ""):
            return
        eng.stats["fail"] += 1
        _where, message = _describe_no_code(model, orig_fwd_fn, resolution, record)
        if not _suppress_errors():
            raise CompileRefused(
                f"{message}. Refusing rather than running it eagerly: an "
                f"uncompiled forward on a compiled device is a silent fallback "
                f"(SPEC S1.4). Set torch._dynamo.config.suppress_errors = True "
                f"to run it eagerly with a warning instead."
            )
        import warnings

        warnings.warn(
            f"{message}; running it eagerly because "
            f"torch._dynamo.config.suppress_errors is set.",
            CompileFallbackWarning,
            stacklevel=4,
        )

    @functools.wraps(orig_fwd_fn)
    def compiled_forward(*args, **kwargs):
        try:
            with eng.root_frame(no_code):
                return eng.run(orig_fwd_fn, (model,) + args, kwargs)
        except Exception as exc:
            if not _is_capture_failure(exc):
                raise
            eng.stats["fail"] += 1
            _where, message = _describe_failure(model, orig_fwd_fn, resolution, exc)
            if not _suppress_errors():
                raise CompileRefused(
                    f"{message}. Refusing rather than running it eagerly: an "
                    f"uncaptured forward on a compiled device is a silent "
                    f"fallback (SPEC S1.4). Set "
                    f"torch._dynamo.config.suppress_errors = True to run it "
                    f"eagerly with a warning instead."
                ) from exc
            import warnings

            warnings.warn(
                f"{message}; running it eagerly because "
                f"torch._dynamo.config.suppress_errors is set.",
                CompileFallbackWarning,
                stacklevel=2,
            )
            return orig_fwd_fn(model, *args, **kwargs)

    model.torchnative_compile = {
        "backend": backend,
        "graph_runs_on": graph_runs_on,
        "engine": eng,
        "compiled": True,
        "resolution": resolution,
        "kv_cache": _install_static_cache_default(model),
    }
    model.forward = compiled_forward
    return model


def compile(model_or_fn=None, *, backend="eager", monitor_roots=None, **options):
    """High-level entry point mirroring torch.compile without PEP 523."""
    if model_or_fn is None:
        return lambda fn: compile(
            fn, backend=backend, monitor_roots=monitor_roots, **options
        )

    _ensure_shim_ready()
    import torch.nn as nn

    if isinstance(model_or_fn, nn.Module):
        attach_to_module(
            model_or_fn, backend=backend, monitor_roots=monitor_roots, **options
        )
        return model_or_fn

    eng = Engine(backend)
    if monitor_roots:
        Monitor(eng, monitor_roots)

    if callable(model_or_fn):
        return eng.wrap(model_or_fn)

    raise TypeError(f"torchnative.compile: unsupported target {type(model_or_fn)}")
