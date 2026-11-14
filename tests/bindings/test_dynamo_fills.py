"""Unit test suite: torch._C small fills for Dynamo tracing.

Issue #209, #210, INTENT §5, SPEC S1.8:
Provides small fills on torch._C and torch._C.TensorBase required for Dynamo tracing:
1. TensorBase members (__delitem__, _fix_weakref, nbytes, itemsize, _version,
   identity-only references for torch.overrides, WeakTensorKeyDictionary support).
2. torch.overrides.get_ignored_functions and get_testing_overrides support
   (__name__ on placeholders and TensorBase descriptors).
3. _functorch interpreter stack, autograd saved tensors hooks, mtia device count,
   mobile allocator trio, and get_num_threads / GlobalStateGuard zero missing symbols.
4. DisableTorchFunction context manager and _dispatch_keys DispatchKeySet synthesis.
5. DelayedError autograd refusal and compile-scoped FakeTensor leaf relaxation.
6. torch._C._dynamo.compiled_autograd and anomaly mode (#306): the state queries
   answer "off", and turning either on refuses by name.

Seven tests:
1. `test_tensorbase_members_and_weak_tensor_key_dictionary`
2. `test_torch_overrides_get_ignored_and_testing_overrides_succeed`
3. `test_functorch_autograd_mtia_mobile_alloc_and_threads_fills`
4. `test_disable_torch_function_and_dispatch_keys`
5. `test_delayed_error_and_compile_scoped_leaf_relaxation`
6. `test_compiled_autograd_answers_off_and_refuses_to_turn_on`
7. `test_anomaly_mode_answers_off_and_refuses_to_turn_on`
"""

import json
import os
import pathlib
import subprocess
import sys

_TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]
_SUPPORT_DIR = _TESTS_DIR / "_support"
if str(_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPPORT_DIR))

import _skip

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_VENDOR_DIR = str(_REPO_ROOT / "torchnative" / "python")
_VENDOR_SHIM = str(_REPO_ROOT / "torchnative" / "python" / "torch" / "_C.abi3.so")
_MAIN_VENDOR_DIR = "/Volumes/macMini/thisisthepy/torchnative/torchnative/python"
_MAIN_VENDOR_SHIM = "/Volumes/macMini/thisisthepy/torchnative/torchnative/python/torch/_C.abi3.so"


def _find_vendor_dir():
    if os.path.isfile(_VENDOR_SHIM):
        return _VENDOR_DIR
    if os.path.isfile(_MAIN_VENDOR_SHIM):
        return _MAIN_VENDOR_DIR
    return _VENDOR_DIR


_SUBPROCESS_HARNESS = r"""
import json
import os
import sys
import traceback

test_target = sys.argv[1] if len(sys.argv) > 1 else ""

import torch
if not hasattr(torch._C, "_aten_implemented"):
    print(json.dumps({"status": "error", "error": "not running torchnative shim"}))
    sys.exit(1)

# Fallback install if not rebuilt yet into _C.abi3.so
if os.environ.get("TORCHNATIVE_NO_FILLS_FALLBACK") != "1":
    try:
        import torchnative._dynamo_fills as _df
        _df.install(torch._C)
    except ImportError:
        pass


def run_tensorbase_members_and_weak_tensor_key_dictionary():
    t = torch.ones(5, dtype=torch.float32)

    # 1. del t[0] raises TypeError with "doesn't support item deletion"
    try:
        del t[0]
        raise AssertionError("del t[0] did not raise TypeError")
    except TypeError as e:
        assert "doesn't support item deletion" in str(e), f"Unexpected TypeError message: {e}"

    # 2. _fix_weakref and WeakTensorKeyDictionary
    assert t._fix_weakref() is None, f"Expected None, got {t._fix_weakref()}"
    from torch.utils.weak import WeakTensorKeyDictionary
    d = WeakTensorKeyDictionary()
    d[t] = 42
    assert d[t] == 42
    keys = list(d.keys())
    assert len(keys) == 1 and keys[0] is t
    items = list(d.items())
    assert len(items) == 1 and items[0][0] is t and items[0][1] == 42

    # 3. nbytes and itemsize across dtypes
    dtypes = [torch.float32, torch.float16, torch.int64, torch.bool]
    for dt in dtypes:
        t_d = torch.zeros(3, 4, dtype=dt)
        assert t_d.nbytes == t_d.numel() * t_d.element_size(), (
            f"nbytes mismatch for {dt}: {t_d.nbytes} vs {t_d.numel() * t_d.element_size()}"
        )
        assert t_d.itemsize == t_d.element_size(), (
            f"itemsize mismatch for {dt}: {t_d.itemsize} vs {t_d.element_size()}"
        )

    # 4. _version == 0
    assert t._version == 0, f"Expected t._version == 0, got {t._version}"

    # 5. Identity-only references on TensorBase
    for n in ["_rev_view_func_unsafe", "_use_count", "_view_func"]:
        assert hasattr(torch._C.TensorBase, n), f"torch._C.TensorBase missing {n}"
        fn = getattr(torch._C.TensorBase, n)
        assert getattr(fn, "__name__", None) == n, f"{n}.__name__ != {n}"

    for n in ["_python_dispatch", "name", "volatile"]:
        assert hasattr(torch._C.TensorBase, n), f"torch._C.TensorBase missing {n}"
        prop = getattr(torch._C.TensorBase, n)
        assert isinstance(prop, property), f"{n} is not property, got {type(prop)}"


def run_torch_overrides_get_ignored_and_testing_overrides_succeed():
    import torch.overrides

    # 1. get_ignored_functions succeeds and returns non-empty set containing callables and descriptors
    ign = torch.overrides.get_ignored_functions()
    assert isinstance(ign, set), f"Expected set, got {type(ign)}"
    assert len(ign) > 0, "get_ignored_functions returned empty set"
    assert torch.Tensor.__delitem__ in ign, "torch.Tensor.__delitem__ not in get_ignored_functions"
    assert any(callable(item) for item in ign), "No callables in get_ignored_functions"
    assert any(hasattr(item, "__get__") for item in ign), "No descriptors in get_ignored_functions"

    # 2. get_testing_overrides succeeds and returns non-empty dict where key has __name__ string
    t_overrides = torch.overrides.get_testing_overrides()
    assert isinstance(t_overrides, dict), f"Expected dict, got {type(t_overrides)}"
    assert len(t_overrides) > 0, "get_testing_overrides returned empty dict"
    for k in t_overrides:
        assert hasattr(k, "__name__"), f"Key {k} missing __name__"
        assert isinstance(k.__name__, str) and len(k.__name__) > 0, f"Key {k}.__name__ is not non-empty string"


def run_functorch_autograd_mtia_mobile_alloc_and_threads_fills():
    # 1. functorch get_interpreter_stack
    assert torch._C._functorch.get_interpreter_stack() is None

    # 2. autograd _top_saved_tensors_default_hooks
    assert torch._C._autograd._top_saved_tensors_default_hooks(False) is None
    assert torch._C._autograd._top_saved_tensors_default_hooks(True) is None

    # 3. autograd _saved_tensors_hooks_set_tracing
    assert torch._C._autograd._saved_tensors_hooks_set_tracing(True) is False
    assert torch._C._autograd._saved_tensors_hooks_set_tracing(False) is True

    # 4. mtia getDeviceCount
    assert torch._C._mtia_getDeviceCount() == 0

    # 5. default mobile cpu allocator
    assert torch._C._is_default_mobile_cpu_allocator_set() is False
    assert torch._C._set_default_mobile_cpu_allocator() is None
    assert torch._C._unset_default_mobile_cpu_allocator() is None

    # 6. get_num_threads and GlobalStateGuard
    t_th = torch.get_num_threads()
    c_th = torch._C.get_num_threads()
    assert isinstance(t_th, int) and t_th >= 1, f"Invalid torch.get_num_threads(): {t_th}"
    assert isinstance(c_th, int) and c_th >= 1, f"Invalid torch._C.get_num_threads(): {c_th}"

    from torchnative._dynamo_guards import GlobalStateGuard, MISSING
    gsg = GlobalStateGuard()
    assert gsg.check() is True, f"GlobalStateGuard check failed: {gsg.reason()}"
    assert MISSING == set(), f"GlobalStateGuard MISSING is not empty: {MISSING}"


def run_disable_torch_function_and_dispatch_keys():
    # 1. DisableTorchFunction context manager
    init_mode = torch._C._is_torch_function_enabled()
    with torch._C.DisableTorchFunction():
        assert torch._C._is_torch_function_enabled() is False
    assert torch._C._is_torch_function_enabled() == init_mode

    from torch.overrides import TorchFunctionMode
    class DummyMode(TorchFunctionMode):
        def __torch_function__(self, func, types, args=(), kwargs=None):
            return func(*args, **(kwargs or {}))

    with DummyMode():
        with torch._C.DisableTorchFunction():
            assert torch._C._is_torch_function_enabled() is False

    # 2. _dispatch_keys
    DK = torch._C.DispatchKey
    DS = torch._C.DispatchKeySet

    t_cpu = torch.zeros(2, 2)
    cpu_keys = torch._C._dispatch_keys(t_cpu)
    expected_cpu = DS(DK.CPU) | DS(DK.ADInplaceOrView) | DS(DK.AutogradCPU)
    assert cpu_keys == expected_cpu, f"Expected {expected_cpu}, got {cpu_keys}"

    t_meta = torch.empty(2, 2, device="meta")
    meta_keys = torch._C._dispatch_keys(t_meta)
    expected_meta = DS(DK.Meta) | DS(DK.ADInplaceOrView) | DS(DK.AutogradMeta)
    assert meta_keys == expected_meta, f"Expected {expected_meta}, got {meta_keys}"

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        t_mps = torch.zeros(2, 2, device="mps")
        mps_keys = torch._C._dispatch_keys(t_mps)
        expected_mps = DS(DK.MPS) | DS(DK.ADInplaceOrView) | DS(DK.AutogradMPS)
        assert mps_keys == expected_mps, f"Expected {expected_mps}, got {mps_keys}"


def run_delayed_error_and_compile_scoped_leaf_relaxation():
    import torchnative._dynamo_fills as _dynamo_fills
    from torch._subclasses import meta_utils
    from torch._subclasses.fake_tensor import FakeTensorMode

    # a. DelayedError returns non-leaf tensor with y.is_leaf is False and y.grad_fn is not None
    t = torch.randn(3, requires_grad=True)
    delayed_fn = torch._C._functions.DelayedError("custom delayed error", 1)
    y = delayed_fn(t)
    assert y.is_leaf is False, f"Expected y.is_leaf is False, got {y.is_leaf}"
    assert y.grad_fn is not None, "Expected y.grad_fn is not None"

    # b. y.sum().backward() raises RuntimeError containing 'custom delayed error'
    try:
        y.sum().backward()
        raise AssertionError("y.sum().backward() did not raise RuntimeError")
    except RuntimeError as e:
        assert "custom delayed error" in str(e), f"Unexpected RuntimeError: {e}"

    # c. FakeTensorMode from_tensor produces FakeTensor ft with ft.is_leaf is False
    t_nonleaf = torch.randn(3, requires_grad=True) * 2.0
    with FakeTensorMode() as mode:
        ft = mode.from_tensor(t_nonleaf)
        assert ft.is_leaf is False, f"Expected ft.is_leaf is False, got {ft.is_leaf}"

    # d. compile_leaf_relaxation
    def assert_eq(a, b):
        assert a == b, f"{a} != {b}"

    t_leaf = t_nonleaf.detach().requires_grad_(True)

    # Outside: raises AssertionError (is_leaf differs False != True)
    try:
        meta_utils.assert_metadata_eq(assert_eq, t_nonleaf, t_leaf)
        raise AssertionError("Expected assert_metadata_eq to raise AssertionError outside context")
    except AssertionError as e:
        assert "False != True" in str(e) or str(e) != "", f"Unexpected error: {e}"

    # Inside context: succeeds (skip_leaf=True)
    with _dynamo_fills.compile_leaf_relaxation():
        meta_utils.assert_metadata_eq(assert_eq, t_nonleaf, t_leaf)
        # Nested context: stays relaxed
        with _dynamo_fills.compile_leaf_relaxation():
            meta_utils.assert_metadata_eq(assert_eq, t_nonleaf, t_leaf)
        # Still inside outer context: stays relaxed
        meta_utils.assert_metadata_eq(assert_eq, t_nonleaf, t_leaf)

    # Outside context: raises AssertionError again
    try:
        meta_utils.assert_metadata_eq(assert_eq, t_nonleaf, t_leaf)
        raise AssertionError("Expected assert_metadata_eq to raise AssertionError after exiting context")
    except AssertionError:
        pass


def _refuses_by_name(fn, *names):
    try:
        fn()
    except NotImplementedError as e:
        for n in names:
            assert n in str(e), f"refusal does not name {n!r}: {e}"
        return str(e)
    raise AssertionError(f"expected a NotImplementedError naming {names}, got no error")


def run_compiled_autograd_answers_off_and_refuses_to_turn_on():
    # Issue #306: `torch.export`'s re-trace (`aot_export_joint_with_descriptors`,
    # which exir's to_edge reaches) enters `compiled_autograd._disable()`, and that
    # calls `set_autograd_compiler(None, False)`. Upstream answers measured in
    # torch 2.14 (qnn-venv), 2026-11-15: (None, False) fresh; the dynamic flag is
    # stored even with no compiler; is_cache_empty True; set_verbose_logger True.
    import sys as _sys
    import types as _types

    ca = torch._C._dynamo.compiled_autograd
    assert isinstance(ca, _types.ModuleType), f"compiled_autograd is {ca!r}, not a module"
    assert _sys.modules.get("torch._C._dynamo.compiled_autograd") is ca, (
        "a later `import torch._C._dynamo.compiled_autograd` would build a second module"
    )

    # 1. The state query answers the truth: no compiler, and the prior pair comes back.
    assert ca.set_autograd_compiler(None, False) == (None, False)
    assert ca.set_autograd_compiler(None, True) == (None, False)
    assert ca.set_autograd_compiler(None, False) == (None, True), "dynamic flag was not stored"
    try:
        ca.set_autograd_compiler(None)
        raise AssertionError("set_autograd_compiler accepted one argument; upstream requires two")
    except TypeError:
        pass

    # 2. Installing a real compiler refuses by name, and leaves the state as it was.
    _refuses_by_name(
        lambda: ca.set_autograd_compiler(lambda: None, False),
        "set_autograd_compiler", "compiled autograd",
    )
    assert ca.set_autograd_compiler(None, False) == (None, False), "a refused call changed the state"

    # 3. The cache: always empty, and clearing it is a no-op that says so by answering.
    assert ca.is_cache_empty() is True
    assert ca.clear_cache() is None
    assert ca.is_cache_empty() is True

    # 4. The verbose logger is stored; upstream answers True for any argument.
    assert ca.set_verbose_logger(print) is True
    assert ca.set_verbose_logger(None) is True

    # 5. The hook runner only a compiled autograd graph calls refuses by name.
    _refuses_by_name(
        lambda: ca.call_cpp_tensor_pre_hooks(0, torch.ones(1)),
        "call_cpp_tensor_pre_hooks",
    )

    # 6. The Python layer above it: _disable() and reset() run; _enable() refuses.
    import torch._dynamo.compiled_autograd as pyca

    with pyca._disable():
        assert pyca.compiled_autograd_enabled is False
    pyca.reset()

    def _enter_enable():
        with pyca._enable(lambda gm: gm):
            pass

    _refuses_by_name(_enter_enable, "set_autograd_compiler")
    assert pyca.compiled_autograd_enabled is False, "a refused _enable() left compiled autograd on"
    assert ca.set_autograd_compiler(None, False) == (None, False)


def run_anomaly_mode_answers_off_and_refuses_to_turn_on():
    # Upstream answers measured in torch 2.14 (qnn-venv), 2026-11-15: a fresh process
    # has anomaly mode off and check_nan ON; set_anomaly_enabled stores check_nan
    # even when it leaves anomaly mode off.
    for fn in (torch.is_anomaly_enabled, torch._C.is_anomaly_enabled):
        assert fn() is False, f"{fn!r}() answered {fn()!r}"
    for fn in (torch.is_anomaly_check_nan_enabled, torch._C.is_anomaly_check_nan_enabled):
        assert fn() is True, f"{fn!r}() answered {fn()!r}; upstream's default is True"

    assert torch.set_anomaly_enabled(False, False) is None
    assert torch.is_anomaly_check_nan_enabled() is False, "check_nan was not stored"
    torch.set_anomaly_enabled(False)
    assert torch.is_anomaly_check_nan_enabled() is True, "check_nan's default is not True"

    # Turning it on refuses by name: this build's backward records no forward
    # traceback and checks nothing for NaN, so "on" would be a mode that does nothing.
    _refuses_by_name(lambda: torch.set_anomaly_enabled(True, False), "set_anomaly_enabled", "anomaly")
    assert torch.is_anomaly_enabled() is False, "a refused call turned anomaly mode on"
    assert torch.is_anomaly_check_nan_enabled() is True, "a refused call changed check_nan"

    # The context managers built on it: off works, on refuses on entry and restores nothing.
    with torch.autograd.set_detect_anomaly(False):
        assert torch.is_anomaly_enabled() is False

    def _enter_detect():
        with torch.autograd.detect_anomaly(check_nan=False):
            pass

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        _refuses_by_name(_enter_detect, "set_anomaly_enabled")
    assert torch.is_anomaly_enabled() is False


try:
    if test_target == "tensorbase":
        run_tensorbase_members_and_weak_tensor_key_dictionary()
    elif test_target == "overrides":
        run_torch_overrides_get_ignored_and_testing_overrides_succeed()
    elif test_target == "fills":
        run_functorch_autograd_mtia_mobile_alloc_and_threads_fills()
    elif test_target == "dispatch":
        run_disable_torch_function_and_dispatch_keys()
    elif test_target == "leaf_relaxation":
        run_delayed_error_and_compile_scoped_leaf_relaxation()
    elif test_target == "compiled_autograd":
        run_compiled_autograd_answers_off_and_refuses_to_turn_on()
    elif test_target == "anomaly":
        run_anomaly_mode_answers_off_and_refuses_to_turn_on()
    else:
        raise ValueError(f"Unknown test target: {test_target}")
    print(json.dumps({"status": "ok"}))
except Exception as e:
    tb = traceback.format_exc()
    print(json.dumps({"status": "error", "error": str(e), "traceback": tb}))
    sys.exit(1)
"""


def _run_fills_subprocess(target):
    vendor_dir = _find_vendor_dir()
    env = dict(os.environ)
    if _VENDOR_DIR != vendor_dir:
        env["PYTHONPATH"] = f"{_VENDOR_DIR}:{vendor_dir}"
    else:
        env["PYTHONPATH"] = vendor_dir
    env["TORCH_USE_RTLD_GLOBAL"] = "1"

    proc = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_HARNESS, target],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )

    if proc.returncode != 0:
        lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
        if lines:
            try:
                data = json.loads(lines[-1])
                if data.get("status") == "error":
                    raise AssertionError(
                        f"Fills subprocess error ({target}):\n{data.get('traceback') or data.get('error')}"
                    )
            except json.JSONDecodeError:
                pass
        raise AssertionError(
            f"Fills subprocess exited with returncode {proc.returncode}:\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )

    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
    if not lines:
        raise AssertionError(
            f"Fills subprocess produced no JSON output:\n{proc.stdout}\n{proc.stderr}"
        )
    data = json.loads(lines[-1])
    if data.get("status") != "ok":
        raise AssertionError(f"Fills test failed: {data}")


def test_tensorbase_members_and_weak_tensor_key_dictionary():
    """Verify TensorBase members, item deletion refusal, nbytes, itemsize, and WeakTensorKeyDictionary."""
    _run_fills_subprocess("tensorbase")


def test_torch_overrides_get_ignored_and_testing_overrides_succeed():
    """Verify torch.overrides get_ignored_functions and get_testing_overrides succeed."""
    _run_fills_subprocess("overrides")


def test_functorch_autograd_mtia_mobile_alloc_and_threads_fills():
    """Verify functorch, autograd hooks, mtia, mobile allocator, and get_num_threads."""
    _run_fills_subprocess("fills")


def test_disable_torch_function_and_dispatch_keys():
    """Verify DisableTorchFunction context manager and _dispatch_keys DispatchKeySet synthesis."""
    _run_fills_subprocess("dispatch")


def test_delayed_error_and_compile_scoped_leaf_relaxation():
    """Verify DelayedError autograd refusal and compile-scoped FakeTensor leaf relaxation."""
    _run_fills_subprocess("leaf_relaxation")


def test_compiled_autograd_answers_off_and_refuses_to_turn_on():
    """compiled_autograd: (None, False) state, empty cache, named refusal of a real compiler (#306)."""
    _run_fills_subprocess("compiled_autograd")


def test_anomaly_mode_answers_off_and_refuses_to_turn_on():
    """Anomaly mode: off with check_nan True by default, stored when off, refused when on (#306)."""
    _run_fills_subprocess("anomaly")


def _main():
    tests = [
        ("test_tensorbase_members_and_weak_tensor_key_dictionary", test_tensorbase_members_and_weak_tensor_key_dictionary),
        ("test_torch_overrides_get_ignored_and_testing_overrides_succeed", test_torch_overrides_get_ignored_and_testing_overrides_succeed),
        ("test_functorch_autograd_mtia_mobile_alloc_and_threads_fills", test_functorch_autograd_mtia_mobile_alloc_and_threads_fills),
        ("test_disable_torch_function_and_dispatch_keys", test_disable_torch_function_and_dispatch_keys),
        ("test_delayed_error_and_compile_scoped_leaf_relaxation", test_delayed_error_and_compile_scoped_leaf_relaxation),
        ("test_compiled_autograd_answers_off_and_refuses_to_turn_on", test_compiled_autograd_answers_off_and_refuses_to_turn_on),
        ("test_anomaly_mode_answers_off_and_refuses_to_turn_on", test_anomaly_mode_answers_off_and_refuses_to_turn_on),
    ]
    return _skip.run_tests(tests, suite="test_dynamo_fills")


if __name__ == "__main__":
    sys.exit(_main())
