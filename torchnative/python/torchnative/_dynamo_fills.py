"""Small fills on torch._C and torch._C.TensorBase for Dynamo tracing.

Issue #209, #210, INTENT §5, SPEC S1.8:
Provides fills required during TorchDynamo frame analysis and tracing:
- Default mobile CPU allocator inspection and stubs
- Functorch interpreter stack query stub
- Autograd saved tensors hooks and tracing toggle
- MTIA device count query stub
- Thread count query (torch._C.get_num_threads and torch.get_num_threads)
- DisableTorchFunction context manager alias
- DispatchKeySet query for tensors (_dispatch_keys)
- TensorBase members (__delitem__, _fix_weakref, nbytes, itemsize, _version,
  identity-only references for torch.overrides, WeakTensorKeyDictionary support)
- __name__ property/attribute fallback on _Unimplemented placeholders
- DelayedError autograd refusal on C._functions
- compile_leaf_relaxation context manager for FakeTensor is_leaf relaxation
- torch._C._dynamo.compiled_autograd (#306): the state answers "off", and
  turning it on refuses by name
- anomaly mode (#306, #330): accepted with check_nan=False while a
  FakeTensorMode is active, and backward refuses by name while it is on

NOTE: This module must NOT import torch at top level (test_import.py and
test_tnnamespace.py require that torchnative modules do not eagerly pull in
torch at module import time).
"""

import contextlib
import os
import sys
import types


def install(C, put=None) -> None:
    """Install Dynamo small fills onto torch._C module and TensorBase."""
    def _set(owner, name, value, qual=None):
        if put is not None:
            try:
                put(owner, name, value, qual)
                return
            except Exception:
                pass
        setattr(owner, name, value)

    # 1. Unimplemented __name__ support
    unimpl_cls = None
    b = sys.modules.get("_torch_c_bootstrap")
    if b is not None and hasattr(b, "_Unimplemented"):
        unimpl_cls = getattr(b, "_Unimplemented")
    if unimpl_cls is None:
        for attr_name in ("_mtia_getDeviceCount", "get_num_threads", "_functorch"):
            attr = getattr(C, attr_name, None)
            if attr is not None and type(attr).__name__ == "_Unimplemented":
                unimpl_cls = type(attr)
                break
    if unimpl_cls is not None:
        def _unimpl_getattr(self, name):
            if name == "__name__":
                return self._qualname.rsplit(".", 1)[-1]
            raise AttributeError(f"{self!r} has no attribute {name!r}")
        unimpl_cls.__getattr__ = _unimpl_getattr

    # 2. Mobile CPU allocator trio
    _set(C, "_is_default_mobile_cpu_allocator_set", lambda: False, "_is_default_mobile_cpu_allocator_set")
    _set(C, "_set_default_mobile_cpu_allocator", lambda: None, "_set_default_mobile_cpu_allocator")
    _set(C, "_unset_default_mobile_cpu_allocator", lambda: None, "_unset_default_mobile_cpu_allocator")

    # 3. Functorch interpreter stack
    functorch = getattr(C, "_functorch", None)
    if functorch is None or type(functorch).__name__ == "_Unimplemented":
        prefix = getattr(C, "__name__", "torch._C")
        functorch = types.ModuleType(f"{prefix}._functorch")
        functorch.__path__ = []
        setattr(C, "_functorch", functorch)
        sys.modules[f"{prefix}._functorch"] = functorch
    _set(functorch, "get_interpreter_stack", lambda: None, "_functorch.get_interpreter_stack")

    # 4. Autograd saved tensors hooks and tracing toggle
    autograd = getattr(C, "_autograd", None)
    if autograd is None or type(autograd).__name__ == "_Unimplemented":
        prefix = getattr(C, "__name__", "torch._C")
        autograd = types.ModuleType(f"{prefix}._autograd")
        autograd.__path__ = []
        setattr(C, "_autograd", autograd)
        sys.modules[f"{prefix}._autograd"] = autograd

    _set(autograd, "_top_saved_tensors_default_hooks", lambda ignore_is_tracing=False: None, "_autograd._top_saved_tensors_default_hooks")

    tracing_cell = [False]

    def _saved_tensors_hooks_set_tracing(v):
        prior = tracing_cell[0]
        tracing_cell[0] = bool(v)
        return prior

    _saved_tensors_hooks_set_tracing.__name__ = (
        _saved_tensors_hooks_set_tracing.__qualname__
    ) = "_saved_tensors_hooks_set_tracing"
    _set(autograd, "_saved_tensors_hooks_set_tracing", _saved_tensors_hooks_set_tracing, "_autograd._saved_tensors_hooks_set_tracing")

    # 5. MTIA device count
    _set(C, "_mtia_getDeviceCount", lambda: 0, "_mtia_getDeviceCount")

    # 6. Thread count query
    def get_num_threads():
        return os.cpu_count() or 1

    get_num_threads.__name__ = get_num_threads.__qualname__ = "get_num_threads"
    _set(C, "get_num_threads", get_num_threads, "get_num_threads")

    varfns = getattr(C, "_VariableFunctions", None)
    if varfns is not None:
        _set(varfns, "get_num_threads", get_num_threads, "_VariableFunctions.get_num_threads")

    if hasattr(C, "__all__") and isinstance(C.__all__, list):
        if "get_num_threads" not in C.__all__:
            C.__all__.append("get_num_threads")

    if "torch" in sys.modules:
        setattr(sys.modules["torch"], "get_num_threads", get_num_threads)

    # 7. DisableTorchFunction
    dtf = getattr(C, "DisableTorchFunctionSubclass", None)
    if dtf is None or type(dtf).__name__ == "_Unimplemented":
        class _StubDisableTorchFunction:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

        dtf = _StubDisableTorchFunction
    _set(C, "DisableTorchFunction", dtf, "DisableTorchFunction")

    # 8. Dispatch keys query
    DK = getattr(C, "DispatchKey", None)
    DS = getattr(C, "DispatchKeySet", None)
    dev_key = {
        "cpu": ("CPU", "AutogradCPU"),
        "cuda": ("CUDA", "AutogradCUDA"),
        "mps": ("MPS", "AutogradMPS"),
        "meta": ("Meta", "AutogradMeta"),
    }

    def _dispatch_keys(t):
        main, ag = dev_key.get(t.device.type, ("CPU", "AutogradCPU"))
        return DS(getattr(DK, main)) | DS(getattr(DK, "ADInplaceOrView")) | DS(getattr(DK, ag))

    _dispatch_keys.__name__ = _dispatch_keys.__qualname__ = "_dispatch_keys"
    _set(C, "_dispatch_keys", _dispatch_keys, "_dispatch_keys")

    # 9. TensorBase members
    TB = getattr(C, "TensorBase", None)
    if TB is not None:
        def __delitem__(self, key):
            raise TypeError("'Tensor' object doesn't support item deletion")

        __delitem__.__name__ = __delitem__.__qualname__ = "__delitem__"
        setattr(TB, "__delitem__", __delitem__)

        setattr(TB, "_fix_weakref", lambda self: None)

        def _make_raising_func(n):
            def f(self, *a, **k):
                raise NotImplementedError(n)

            f.__name__ = f.__qualname__ = n
            return f

        for n in ("_rev_view_func_unsafe", "_use_count", "_view_func"):
            setattr(TB, n, _make_raising_func(n))

        def _make_raising_prop(n):
            def g(self, n=n):
                raise NotImplementedError(n)

            g.__name__ = g.__qualname__ = n
            return property(g)

        for n in ("_python_dispatch", "name", "volatile"):
            setattr(TB, n, _make_raising_prop(n))

        setattr(TB, "_version", property(lambda self: 0))
        setattr(TB, "nbytes", property(lambda self: self.numel() * self.element_size()))
        setattr(TB, "itemsize", property(lambda self: self.element_size()))

    # 10. DelayedError on C._functions
    _delayed_fn_cls = [None]

    def _get_delayed_fn_cls():
        if _delayed_fn_cls[0] is None:
            import torch

            class _DelayedErrorFn(torch.autograd.Function):
                @staticmethod
                def forward(ctx, x, msg):
                    ctx.msg = msg
                    # Upstream's DelayedError returns `var.tensor_data()`: the
                    # same storage, and not a view. `aten.alias` is a view op
                    # by schema, so since #272 its result answers
                    # `_is_view()` True, and meta_utils' assert_metadata_eq
                    # then refuses every fake non-leaf input (#279's gate,
                    # run 43362953251). `detach` shares storage and is not a
                    # view, as upstream's answer is.
                    return x.detach()

                @staticmethod
                def backward(ctx, grad_out):
                    raise RuntimeError(ctx.msg)

            _delayed_fn_cls[0] = _DelayedErrorFn
        return _delayed_fn_cls[0]

    class DelayedError:
        def __init__(self, msg, num_inputs):
            self.msg = msg
            self.num_inputs = num_inputs

        def __call__(self, *inputs):
            fn_cls = _get_delayed_fn_cls()
            if len(inputs) == 1:
                return fn_cls.apply(inputs[0], self.msg)
            return tuple(fn_cls.apply(x, self.msg) for x in inputs)

    DelayedError.__name__ = DelayedError.__qualname__ = "DelayedError"

    funcs = getattr(C, "_functions", None)
    if funcs is None or type(funcs).__name__ == "_Unimplemented":
        prefix = getattr(C, "__name__", "torch._C")
        funcs = types.ModuleType(f"{prefix}._functions")
        funcs.__path__ = []
        setattr(C, "_functions", funcs)
        sys.modules[f"{prefix}._functions"] = funcs

    _set(funcs, "DelayedError", DelayedError, "_functions.DelayedError")
    prefix = getattr(C, "__name__", "torch._C")
    mod_funcs = sys.modules.get(f"{prefix}._functions")
    if mod_funcs is not None and mod_funcs is not funcs:
        setattr(mod_funcs, "DelayedError", DelayedError)

    # 11. torch._C._dynamo.compiled_autograd (#306)
    _install_compiled_autograd(C, _set)

    # 12. Anomaly mode (#306)
    _install_anomaly_mode(C, _set)

    # 7. CUDA tensor types on torch.cuda for Dynamo user_defined._in_graph_classes
    torch_mod = sys.modules.get("torch")
    if torch_mod is not None:
        cuda_mod = getattr(torch_mod, "cuda", None)
        if cuda_mod is not None:
            tensor_names = [
                "FloatTensor",
                "DoubleTensor",
                "HalfTensor",
                "BFloat16Tensor",
                "ByteTensor",
                "CharTensor",
                "IntTensor",
                "ShortTensor",
                "LongTensor",
            ]
            for tname in tensor_names:
                if not hasattr(cuda_mod, tname):
                    base_tensor = getattr(
                        torch_mod, tname, getattr(torch_mod, "Tensor", object)
                    )
                    try:
                        cuda_cls = type(
                            tname, (base_tensor,), {"__module__": "torch.cuda"}
                        )
                    except TypeError:
                        cuda_cls = base_tensor
                    setattr(cuda_mod, tname, cuda_cls)


def _named(fn, name):
    fn.__name__ = fn.__qualname__ = name
    return fn


def _install_compiled_autograd(C, _set) -> None:
    """`torch._C._dynamo.compiled_autograd`: compiled autograd is always off here.

    Issue #306. `torch/_functorch/aot_autograd.py:1416` enters
    `torch/_dynamo/compiled_autograd.py:1689` `_disable()`, which calls
    `set_autograd_compiler(None, False)` and restores the pair it returned.
    That is the re-trace `torch.export`'s `run_decompositions` takes, so exir's
    `to_edge` stops there while the module is the catch-all's `_Unimplemented`
    (CI run 43610884249). `torch/_dynamo/__init__.py`'s `reset()` adds
    `set_verbose_logger(None)` and `clear_cache()`.

    Upstream's compiled autograd hands the C++ autograd engine's backward graph
    to a Python compiler. This build has no C++ autograd engine: backward is
    torchnative's own reverse walk over a captured trace
    (docs/training/AUTOGRAD.md). So the state is always "no compiler", the
    cache is always empty, and installing a compiler refuses by name rather
    than being accepted and never consulted.

    Upstream answers, measured in torch 2.14 on 2026-11-15:
    `set_autograd_compiler(None, d)` returns the prior `(compiler, dynamic)`
    and stores `d` even with no compiler; `is_cache_empty()` is True in a
    fresh process; `set_verbose_logger(x)` returns True for any `x`; both
    `set_autograd_compiler` arguments are required.
    """
    prefix = getattr(C, "__name__", "torch._C")
    dynamo = getattr(C, "_dynamo", None)
    if dynamo is None or type(dynamo).__name__ == "_Unimplemented":
        dynamo = types.ModuleType(f"{prefix}._dynamo")
        dynamo.__path__ = []
        setattr(C, "_dynamo", dynamo)
        sys.modules[f"{prefix}._dynamo"] = dynamo

    # A real module, registered, and without the bootstrap's catch-all:
    # upstream's module has exactly these five names and answers
    # AttributeError for any other. Registering it in `sys.modules` keeps a
    # later `import torch._C._dynamo.compiled_autograd` from building a second
    # (catch-all) module through `_SubmoduleFinder` and rebinding the
    # attribute over this one, the hazard `_install_dynamo_bool` describes
    # for `guards`.
    modname = f"{prefix}._dynamo.compiled_autograd"
    ca = sys.modules.get(modname)
    if ca is None or not isinstance(ca, types.ModuleType):
        ca = types.ModuleType(modname)
        sys.modules[modname] = ca
    setattr(dynamo, "compiled_autograd", ca)

    # Upstream keeps the dynamic flag in a C++ global next to the compiler
    # pointer; it is stored whether or not a compiler is installed.
    dynamic_cell = [False]
    logger_cell = [None]

    def set_autograd_compiler(autograd_compiler, dynamic, /):
        if autograd_compiler is not None:
            raise NotImplementedError(
                "torch._C._dynamo.compiled_autograd.set_autograd_compiler: "
                "compiled autograd is not available in this build. It captures "
                "the C++ autograd engine's backward graph, and torchnative has "
                "no C++ autograd engine: backward is its own reverse walk over "
                "a captured trace (docs/training/AUTOGRAD.md). Only None "
                "(compiled autograd off, which it always is here) is accepted."
            )
        prior = (None, dynamic_cell[0])
        dynamic_cell[0] = bool(dynamic)
        return prior

    def clear_cache():
        # Nothing is ever compiled, so there is nothing to clear.
        return None

    def is_cache_empty():
        return True

    def set_verbose_logger(fn, /):
        # Stored, as upstream stores it. It is never called because compiled
        # autograd never runs here, which is also what upstream does with a
        # logger while no compiler is installed.
        logger_cell[0] = fn
        return True

    def call_cpp_tensor_pre_hooks(idx, grad, /):
        raise NotImplementedError(
            "torch._C._dynamo.compiled_autograd.call_cpp_tensor_pre_hooks: "
            "only a compiled autograd graph calls this, and this build never "
            "builds one (set_autograd_compiler refuses a compiler)."
        )

    for fn in (set_autograd_compiler, clear_cache, is_cache_empty,
               set_verbose_logger, call_cpp_tensor_pre_hooks):
        _named(fn, fn.__name__)
        _set(ca, fn.__name__, fn, f"_dynamo.compiled_autograd.{fn.__name__}")


def _fake_mode_active(C) -> bool:
    """Is the FAKE infra slot of the torch-dispatch mode stack set?

    The same query upstream's `torch._subclasses.fake_tensor` and
    `torch.utils._python_dispatch._detect_infra_mode` make, answered by the
    slot table `bootstrap.py`'s `_install_mode_stack` keeps. While a mode's own
    `__torch_dispatch__` runs, upstream (and the shim) unset its slot, so this
    answers False there; nothing in upstream sets anomaly mode from inside a
    handler.
    """
    key_enum = getattr(C, "_TorchDispatchModeKey", None)
    get_mode = getattr(C, "_get_dispatch_mode", None)
    if key_enum is None or get_mode is None:
        return False
    return get_mode(key_enum.FAKE) is not None


def _install_anomaly_mode(C, _set) -> None:
    """`torch.is_anomaly_enabled`, its two siblings, and the backward guard.

    Issues #306 and #330. `torch/_functorch/_aot_autograd/
    graph_capture_wrappers.py:1529` re-traces a `GraphModule` inside
    `torch.autograd.detect_anomaly(check_nan=False)`, which exir's `to_edge`
    reaches through `run_decompositions`. `torch/autograd/anomaly_mode.py`
    reads `is_anomaly_enabled()` and `is_anomaly_check_nan_enabled()` in
    `__init__`, calls `set_anomaly_enabled(True, check_nan)` on enter and
    `set_anomaly_enabled(prev, prev_check_nan)` on exit.

    The rule (leader decision 2026-11-15, agent proposal; not owner-confirmed):

    * `set_anomaly_enabled(True, check_nan=False)` is accepted only while a
      FakeTensorMode is active (the FAKE infra slot is set). Measured on
      upstream 2.13.0: both re-traces `run_decompositions` makes of a
      Linear + ReLU program call it with the FAKE slot set, and nothing else
      in upstream's Python reads the flag.
    * `check_nan=True` refuses by name in any context: torchnative's backward
      (docs/training/AUTOGRAD.md) checks no backward output for NaN.
    * Outside a FakeTensorMode, turning it on refuses by name.
    * Turning it off always works and stores `check_nan`, so a context
      manager's exit restores `(prev, prev_check_nan)` exactly.
    * While it is on, `_ImperativeEngine.run_backward` refuses by name (see
      `_guard_backward`). Upstream's engine records each node's forward
      traceback under anomaly mode and reports it when a backward fails; the
      shim records none, so a backward under the mode would silently break
      that promise.

    **If the FakeTensorMode exits while anomaly mode is still on**, the mode
    stays on. Upstream's flag is a process global with no tie to fake mode
    (measured on 2.13.0: `with FakeTensorMode(): set_anomaly_enabled(True,
    False)` leaves `(True, False)` after the block), and in upstream's own
    nesting the anomaly block is the inner one: `detect_anomaly` is entered
    inside `functional_call`, which runs under the tracer's FakeTensorMode, and
    its exit (`set_anomaly_enabled(False, True)`) is measured before the fake
    mode's exit. So only a caller that sets the flag by hand can reach this
    state. Resetting it behind that caller would make the queries disagree
    with what the caller set and with upstream; instead the queries stay
    truthful and every backward keeps refusing by name until the caller turns
    the mode off. Turning it on again outside the fake mode still refuses.

    `grad_fn.metadata` keeps refusing by name (`bootstrap.py`
    `_GradFnNode.__getattr__`). Upstream puts `metadata["traceback_"]` on
    every node created under the mode; its one reader,
    `logging_utils.setup_stacktrace_preservation_hooks`, runs only on the
    joint (training) trace and after `functional_call` has left the anomaly
    block, and it is not called by an eval `run_decompositions` of Linear +
    ReLU (measured on 2.13.0: zero calls). An empty dict would claim that no
    traceback was recorded, which under anomaly mode is false.
    """
    state = {"enabled": False, "check_nan": True}

    def is_anomaly_enabled():
        return state["enabled"]

    def is_anomaly_check_nan_enabled():
        return state["check_nan"]

    def set_anomaly_enabled(enabled, check_nan=True):
        if not enabled:
            state["enabled"] = False
            state["check_nan"] = bool(check_nan)
            return None
        if check_nan:
            raise NotImplementedError(
                "torch.set_anomaly_enabled(True, check_nan=True): anomaly "
                "detection with check_nan is not available in this build. "
                "Upstream's autograd engine then checks every backward output "
                "for NaN, and torchnative's backward "
                "(docs/training/AUTOGRAD.md) checks nothing, so the mode would "
                "promise a check that never runs. Only check_nan=False, inside "
                "a FakeTensorMode (AOT autograd's re-trace of a GraphModule, "
                "torch/_functorch/_aot_autograd/graph_capture_wrappers.py), "
                "is accepted."
            )
        if not _fake_mode_active(C):
            raise NotImplementedError(
                "torch.set_anomaly_enabled(True, check_nan=False) outside a "
                "FakeTensorMode: anomaly detection is not available in this "
                "build. Upstream's autograd engine records each node's forward "
                "traceback and reports it when a backward fails; torchnative's "
                "backward (docs/training/AUTOGRAD.md) records none, so on real "
                "tensors the mode would change nothing. It is accepted only "
                "while a FakeTensorMode is active, the tracing-only use AOT "
                "autograd's re-trace of a GraphModule makes "
                "(torch/_functorch/_aot_autograd/graph_capture_wrappers.py), "
                "and backward refuses while it is on."
            )
        state["enabled"] = True
        state["check_nan"] = False
        return None

    fns = (is_anomaly_enabled, is_anomaly_check_nan_enabled, set_anomaly_enabled)
    torch_mod = sys.modules.get("torch")
    for fn in fns:
        _named(fn, fn.__name__)
        _set(C, fn.__name__, fn, fn.__name__)
        # `torch.<name>` is bound by `from torch._C import *`, which runs after
        # the bootstrap; a re-install after `import torch` has to rebind it too.
        if torch_mod is not None and torch_mod is not C:
            setattr(torch_mod, fn.__name__, fn)

    _guard_backward(C, state)


def _guard_backward(C, state) -> None:
    """Refuse every backward by name while anomaly mode is on (#330).

    `_ImperativeEngine.run_backward` (installed by `bootstrap.py`
    `_install_engine`) is the one choke point: `Tensor.backward()` and
    `torch.autograd.backward` reach it through
    `torch/autograd/graph.py::_engine_run_backward`, and so does
    `torch.autograd.grad` (docs/training/BACKWARD2.md section 1.3 measured
    that it is one wall). The guard runs before the engine touches the tape,
    so a refused backward leaves the graph as it was and the same backward
    succeeds once the mode is off.

    Installing twice (the bootstrap installs these fills, and a probe may
    install them again) wraps the unguarded engine, not the previous guard,
    and the guard reads the state of the latest install.
    """
    engine = getattr(C, "_ImperativeEngine", None)
    if engine is None:
        return
    current = getattr(engine, "run_backward", None)
    if current is None:
        return
    unguarded = getattr(current, "_tn_unguarded", current)

    def run_backward(self, *args, **kwargs):
        if state["enabled"]:
            raise NotImplementedError(
                "_ImperativeEngine.run_backward while anomaly mode is on: "
                "torchnative's backward cannot honour anomaly mode. Upstream's "
                "engine records each node's forward traceback (grad_fn."
                "metadata['traceback_']) and reports it when a backward fails; "
                "this build records none and would report nothing. Anomaly "
                "mode is accepted only for tracing inside a FakeTensorMode; "
                "turn it off (torch.set_anomaly_enabled(False)) before "
                "Tensor.backward() or torch.autograd.grad()."
            )
        return unguarded(self, *args, **kwargs)

    run_backward.__name__ = "run_backward"
    run_backward.__qualname__ = getattr(
        unguarded, "__qualname__", "_ImperativeEngine.run_backward"
    )
    run_backward._tn_unguarded = unguarded
    engine.run_backward = run_backward


_relaxation_depth = 0
_orig_assert_metadata_eq = None


@contextlib.contextmanager
def compile_leaf_relaxation():
    """Relax FakeTensor metadata equality check on is_leaf during compilation.

    Restores original assert_metadata_eq upon exiting the outermost context.
    """
    global _relaxation_depth, _orig_assert_metadata_eq
    import torch._subclasses.meta_utils as mu

    if _relaxation_depth == 0:
        _orig_assert_metadata_eq = mu.assert_metadata_eq
        orig = _orig_assert_metadata_eq

        def _relaxed_assert_metadata_eq(assert_eq, m1, m2, *args, **kwargs):
            kwargs["skip_leaf"] = True
            return orig(assert_eq, m1, m2, *args, **kwargs)

        mu.assert_metadata_eq = _relaxed_assert_metadata_eq

    _relaxation_depth += 1
    try:
        yield
    finally:
        _relaxation_depth -= 1
        if _relaxation_depth == 0:
            if _orig_assert_metadata_eq is not None:
                mu.assert_metadata_eq = _orig_assert_metadata_eq
                _orig_assert_metadata_eq = None
