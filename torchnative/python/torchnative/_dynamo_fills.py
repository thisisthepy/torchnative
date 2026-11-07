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
                    return torch.ops.aten.alias.default(x)

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
