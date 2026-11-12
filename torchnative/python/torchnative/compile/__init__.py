"""Hook-free Dynamo compilation engine without PEP 523.

Issue #211, INTENT §5, §7, SPEC S1.8, S4.2:
Runs Dynamo frame conversion and bytecode tracing without PEP 523 set_eval_frame hooks:
1. `Engine`: wraps ConvertFrame with FakeFrame and pure-Python guards evaluation.
2. `Monitor`: sys.monitoring (PY_START, PEP 669) + __code__ trampoline for nested non-inlined functions.
3. `compile`: top-level entry point mirroring torch.compile without PEP 523.
4. `attach_to_module`: in-place forward wrapper for nn.Module after accelerator lowering.
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

    import torch._dynamo.config as dc

    dc.automatic_dynamic_shapes = False
    dc.assume_static_by_default = True
    dc.specialize_float = True
    _shim_initialized = True


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
        for eng in reversed(_all_engines):
            eng.entries.pop(code, None)

    orig_strategy = getattr(ef, "set_code_exec_strategy", None)

    def set_strategy(code, strategy):
        action = getattr(strategy, "cur_action", strategy)
        if action == 1 or getattr(action, "name", None) == "SKIP":
            for eng in reversed(_all_engines):
                eng.skip_codes.add(code)
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

        if backend is None or backend in ("eager", "inductor", "npu"):
            backend = lambda gm, ex: gm.forward

        self.backend = backend
        self.hooks = Hooks()
        self.cb = convert_frame(backend, self.hooks)
        self.entries = {}  # code -> [(guard_manager, newfn)]
        self.frame_state = {}  # code -> frame_state dict
        self.skip_codes = set()
        self.stats = dict(translate=0, hit=0, skip=0, fail=0, resume_wrapped=0)
        self.known = set()
        self.swaps = []  # (fn, orig_code, tramp_code)
        self.translating = 0
        self.debug = False

        _all_engines.append(self)
        _ensure_eval_frame_wired()
        _ensure_resume_patch()

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
            with self.unswapped():
                with _dynamo_fills.compile_leaf_relaxation():
                    ret = self.cb(frame, None, self.hooks, fs)
        finally:
            if _active_engines and _active_engines[-1] is self:
                _active_engines.pop()

        gc_res = getattr(ret, "guarded_code", None)
        if gc_res is None:
            self.stats["skip"] += 1
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


def _describe_failure(model, orig_fwd_fn, resolution, exc):
    where = f"{type(model).__name__}.{getattr(orig_fwd_fn, '__name__', 'forward')}"
    if resolution is not None:
        backend = getattr(resolution, "backend", None)
        unit = getattr(resolution, "unit", None)
        target = (
            f" for {unit} ({backend})" if unit or backend else f" for {resolution!r}"
        )
    else:
        target = ""
    return where, (
        f"torchnative.compile: capturing {where}{target} failed with "
        f"{type(exc).__name__}: {exc}"
    )


def attach_to_module(model, *, backend=None, resolution=None, **options):
    """Attach torchnative hook-free compile engine to lowered nn.Module.

    A capture failure is never answered by a silent eager run (#255, SPEC S1.4).
    By default it raises `CompileRefused`. With
    `torch._dynamo.config.suppress_errors = True`, upstream's opt-in for the
    same trade, it warns (`CompileFallbackWarning`) naming the forward, the
    device and the cause, and runs the original forward. Either way
    `stats["fail"]` counts it. Graph breaks are not failures: Dynamo resumes
    them inside the engine.
    """
    _ensure_shim_ready()
    if hasattr(model, "torchnative_compile") and model.torchnative_compile.get("compiled"):
        return model

    if isinstance(backend, Engine):
        eng = backend
    else:
        eng = Engine(backend)

    orig_fwd_fn = type(model).forward

    @functools.wraps(orig_fwd_fn)
    def compiled_forward(*args, **kwargs):
        try:
            return eng.run(orig_fwd_fn, (model,) + args, kwargs)
        except Exception as exc:
            if not _is_capture_failure(exc):
                raise
            eng.stats["fail"] += 1
            _where, message = _describe_failure(model, orig_fwd_fn, resolution, exc)
            try:
                import torch._dynamo.config as _dynamo_config

                suppress = bool(getattr(_dynamo_config, "suppress_errors", False))
            except Exception:  # noqa: BLE001
                suppress = False
            if not suppress:
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
        "engine": eng,
        "compiled": True,
        "resolution": resolution,
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
