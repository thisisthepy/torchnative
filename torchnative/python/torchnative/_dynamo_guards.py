"""Pure-Python stand-in for torch._C._dynamo.guards (abi3-compatible, no CPython internals).

Issue #207, SPEC S1.8:
Pure-Python guard manager matching upstream PyTorch's C++ guards across all
leaf guards, accessors, DictGuardManager, relational guards, and epilogue lambdas.

NOTE: This module must NOT import torch at top level (test_import.py and
test_tnnamespace.py require that torchnative modules do not eagerly pull in
torch at module import time).
"""

import enum
import sys
import types
import weakref

_SENTINEL = object()
MISSING = set()  # Names this implementation needed and did not find


class GuardDebugInfo:
    def __init__(self, result, verbose_code_parts, num_guards_executed, user_stack=None):
        self.result = result
        self.verbose_code_parts = verbose_code_parts
        self.num_guards_executed = num_guards_executed
        self.user_stack = user_stack


class GlobalStateGuard:
    def __init__(self):
        self._st = self._snapshot()

    @staticmethod
    def _snapshot():
        import torch
        def g(name, f):
            try:
                return f()
            except NotImplementedError:
                MISSING.add(name)
                return None
        d = dict(
            grad_mode=g("is_grad_enabled", torch.is_grad_enabled),
            torch_function=g("_is_torch_function_enabled", torch._C._is_torch_function_enabled),
            deterministic=g("are_deterministic_algorithms_enabled", torch.are_deterministic_algorithms_enabled),
            warn_only=g("is_deterministic_algorithms_warn_only_enabled", torch.is_deterministic_algorithms_warn_only_enabled),
            num_threads=g("get_num_threads", torch.get_num_threads),
            default_dtype=g("get_default_dtype", torch.get_default_dtype),
            autocast=(g("is_autocast_enabled(cpu)", lambda: torch.is_autocast_enabled("cpu")),
                      g("get_autocast_dtype", lambda: torch.get_autocast_dtype("cpu")),
                      g("is_autocast_cache_enabled", torch.is_autocast_cache_enabled)),
        )
        return d

    def check(self):
        return self._st == self._snapshot()

    def reason(self):
        cur = self._snapshot()
        return " ".join(k for k in self._st if self._st[k] != cur[k])


class LeafGuard:
    def __init__(self, root=None, verbose_code_parts=None, user_stack=None):
        self._root = root
        self._vcp = list(verbose_code_parts or [])
        self._user_stack = user_stack

    def verbose_code_parts(self):
        return self._vcp

    def check(self, value):
        return self.check_nopybind(value)

    def check_verbose(self, value):
        try:
            ok = self.check_nopybind(value)
        except Exception as e:  # LAMBDA_GUARD reports the exception text
            return GuardDebugInfo(False, [f"{type(e).__name__}: {e}"], 0)
        if ok:
            return GuardDebugInfo(True, [], 0)
        return GuardDebugInfo(False, self._vcp, 0, self._user_stack)


class RelationalGuard(LeafGuard):
    def reset_state(self):
        pass


def _eq(a, b, false_on_error=True):
    try:
        return bool(a == b)
    except Exception:
        if false_on_error:
            return False
        raise


class LAMBDA_GUARD(LeafGuard):
    def __init__(self, root, fn, vcp, us):
        super().__init__(root, vcp, us)
        self._fn = fn

    def check_nopybind(self, v):
        try:
            return bool(self._fn(v))
        except Exception:
            return False

    def check_verbose(self, value):
        try:
            ok = self._fn(value)
        except Exception as e:
            return GuardDebugInfo(False, [str(e)], 0)
        if ok:
            return GuardDebugInfo(True, [], 0)
        return GuardDebugInfo(False, self._vcp, 0, self._user_stack)


class TYPE_MATCH(LeafGuard):
    def __init__(self, root, type_id, vcp, us):
        super().__init__(root, vcp, us)
        self._t = type_id

    def check_nopybind(self, v):
        return id(type(v)) == self._t


class FAKE_SCRIPT_TYPE_MATCH(LeafGuard):
    def __init__(self, root, cls, type_id, vcp, us):
        super().__init__(root, vcp, us)
        self._t = type_id

    def check_nopybind(self, v):
        return id(type(v)) == self._t


class ID_MATCH(LeafGuard):
    def __init__(self, root, id_val, vcp, us):
        super().__init__(root, vcp, us)
        self._id = id_val

    def check_nopybind(self, v):
        return id(v) == self._id


class NONE_MATCH(LeafGuard):
    def check_nopybind(self, v):
        return v is None


class NOT_NONE(LeafGuard):
    def check_nopybind(self, v):
        return v is not None


class TRUE_MATCH(LeafGuard):
    def check_nopybind(self, v):
        return v is True


class FALSE_MATCH(LeafGuard):
    def check_nopybind(self, v):
        return v is False


class EQUALS_MATCH(LeafGuard):
    def __init__(self, root, value, vcp, us):
        super().__init__(root, vcp, us)
        self._v = value
        self._t = type(value)

    def check_nopybind(self, v):
        if v is self._v:
            return True
        if type(v) is not self._t:
            return False
        return _eq(v, self._v)


class RANGE_ITERATOR_MATCH(LeafGuard):
    def __init__(self, root, start, stop, step, type_id, vcp, us):
        super().__init__(root, vcp, us)
        self._r = range(start, stop, step)
        self._t = type_id

    def check_nopybind(self, v):
        if id(type(v)) != self._t:
            return False
        _, (r,), idx = v.__reduce__()
        if idx is None:
            cur_start = r.start
        else:
            cur_start = r.start + r.step * idx
        return (cur_start, r.stop, r.step) == (self._r.start, self._r.stop, self._r.step)


class TUPLE_ITERATOR_LEN(LeafGuard):
    def __init__(self, root, length, type_id, vcp, us):
        super().__init__(root, vcp, us)
        self._n = length
        self._t = type_id

    def check_nopybind(self, v):
        return id(type(v)) == self._t and v.__length_hint__() == self._n


class LENGTH_CHECK(LeafGuard):
    def __init__(self, root, value, vcp, us):
        super().__init__(root, vcp, us)
        self._n = value

    def check_nopybind(self, v):
        try:
            return len(v) == self._n
        except Exception:
            return False


class DICT_LENGTH(LENGTH_CHECK):
    def check_nopybind(self, v):
        return isinstance(v, dict) and dict.__len__(v) == self._n


class MAPPING_KEYS_MATCH(LeafGuard):
    def __init__(self, root, value, vcp, us):
        super().__init__(root, vcp, us)
        self._keys = list(value.keys())

    def check_nopybind(self, v):
        return _eq(list(v.keys()), self._keys, False)


class DEFAULT_DEVICE(LeafGuard):
    def __init__(self, root, vcp, us):
        super().__init__(root, vcp, us)
        import torch.utils._device as d
        self._mod = d
        self._dev = d.CURRENT_DEVICE

    def check_nopybind(self, v):
        d = self._mod.CURRENT_DEVICE
        return d is self._dev or _eq(d, self._dev)


class GLOBAL_STATE(LeafGuard):
    def __init__(self, root, initial_state, vcp, us):
        super().__init__(root, vcp, us)
        self._g = initial_state

    def check_nopybind(self, v):
        return self._g.check()


class NO_HASATTR(LeafGuard):
    def __init__(self, root, attr, vcp, us):
        super().__init__(root, vcp, us)
        self._a = attr

    def check_nopybind(self, v):
        return not hasattr(v, self._a)


class DICT_CONTAINS(LeafGuard):
    def __init__(self, root, contains, key, vcp, us):
        super().__init__(root, vcp, us)
        self._c = contains
        self._k = key

    def check_nopybind(self, v):
        try:
            return (self._k in v) == self._c
        except Exception:
            return False


class SET_CONTAINS(DICT_CONTAINS):
    pass


class FLOAT_IS_NAN(LeafGuard):
    def check_nopybind(self, v):
        return isinstance(v, float) and v != v


class COMPLEX_IS_NAN(LeafGuard):
    def check_nopybind(self, v):
        return isinstance(v, complex) and (v.real != v.real or v.imag != v.imag)


class DUAL_LEVEL_MATCH(LeafGuard):
    def __init__(self, root, level, vcp, us):
        super().__init__(root, vcp, us)
        self._l = level

    def check_nopybind(self, v):
        import torch.autograd.forward_ad as fwAD
        return fwAD._current_level == self._l


class OBJECT_ALIASING(RelationalGuard):
    def __init__(self, root, vcp, us):
        super().__init__(root, vcp, us)
        self.reset_state()

    def reset_state(self):
        self._first = _SENTINEL

    def check_nopybind(self, v):
        if self._first is _SENTINEL:
            self._first = v
            return True
        return self._first is v


class NO_TENSOR_ALIASING(RelationalGuard):
    def __init__(self, root, names, vcp, us):
        super().__init__(root, vcp, us)
        self._names = names
        self.reset_state()

    def reset_state(self):
        self._seen = set()

    def check_nopybind(self, v):
        k = id(v)
        if k in self._seen:
            return False
        self._seen.add(k)
        return True


class STORAGE_OVERLAPPING(RelationalGuard):
    def __init__(self, root, overlapping, checker, vcp, us):
        super().__init__(root, vcp, us)
        self._ov = overlapping
        self._ck = checker

    def reset_state(self):
        self._ck.reset()

    def check_nopybind(self, v):
        return self._ck.add(self._ov, v)


class _StorageOverlapChecker:
    def __init__(self, n_ov, n_non):
        self.n = (n_ov, n_non)
        self.reset()

    def reset(self):
        self.ov = []
        self.non = []

    def add(self, overlapping, t):
        (self.ov if overlapping else self.non).append(t)
        if len(self.ov) == self.n[0] and len(self.non) == self.n[1]:
            allt = self.ov + self.non
            idx = compute_overlapping_tensors(allt, False)
            return len(idx) == len(self.ov) and all(i in idx for i in range(len(self.ov)))
        return True


def compute_overlapping_tensors(tensors, symbolic=True):
    def disjoint(x, y):
        if x is y:
            return False
        if x.numel() == 0 or y.numel() == 0:
            return True
        if x.storage_offset() > y.storage_offset():
            return disjoint(y, x)
        if x.is_contiguous() and y.is_contiguous():
            return not (x.storage_offset() + x.numel() > y.storage_offset())
        x_last = x.storage_offset() + sum((x.size(i) - 1) * x.stride(i) for i in range(x.dim()))
        if x_last < y.storage_offset():
            return True
        return False
    out = set()
    for i, ti in enumerate(tensors):
        for j in range(i):
            if not disjoint(ti, tensors[j]):
                out.add(i)
                out.add(j)
    return out


class SYMBOLIC_SHAPE_GUARD(RelationalGuard):
    pass


class DIMENSION_DYNAMIC_MARKING_GUARD(LeafGuard):
    def __init__(self, root, expected, absent, dependent, vcp, us):
        super().__init__(root, vcp, us)
        self._exp = dict(expected)
        self._abs = list(absent)
        self._dep = dict(dependent)
        self._all_absent = not self._exp and not self._dep

    def check_nopybind(self, v):
        M = "_dynamo_has_dynamic_marking"
        if self._all_absent:
            return not hasattr(v, M)
        if not hasattr(v, M):
            return True
        for a, exp in self._exp.items():
            try:
                act = getattr(v, a)
            except AttributeError:
                continue
            if not set(act).issubset(exp):
                return False
        for a in self._abs:
            if hasattr(v, a):
                return False
        for a, (exp, gate) in self._dep.items():
            if not hasattr(v, gate):
                continue
            try:
                act = getattr(v, a)
            except AttributeError:
                if exp is None:
                    continue
                return False
            if not _eq(act, exp, False):
                return False
        return True


class DICT_VERSION(LeafGuard):
    """Emulated by a (keys, value ids) snapshot."""
    def __init__(self, root, value, vcp, us):
        super().__init__(root, vcp, us)
        self._snap = dict_version(value)

    def check_nopybind(self, v):
        return isinstance(v, dict) and dict_version(v) == self._snap


def dict_version(d):
    return tuple((k, id(x)) for k, x in dict.items(d))


class TORCH_FUNCTION_MODE_STACK(LeafGuard):
    def __init__(self, root, initial_stack, vcp, us):
        super().__init__(root, vcp, us)
        self._ref = [type(m) for m in initial_stack]

    def check_nopybind(self, v):
        from torch.overrides import _get_current_function_mode_stack
        st = _get_current_function_mode_stack()
        return len(st) == len(self._ref) and all(type(m) is t for m, t in zip(st, self._ref))


class DISPATCH_KEY_SET_MATCH(LeafGuard):
    def __init__(self, root, value, vcp, us):
        super().__init__(root, vcp, us)
        self._v = value

    def check_nopybind(self, v):
        return v == self._v


def _tls_dispatch_state():
    try:
        import torch
        return (torch._C._dispatch_tls_local_include_set(), torch._C._dispatch_tls_local_exclude_set())
    except Exception:
        return None


class TENSOR_MATCH(LeafGuard):
    def __init__(self, root, value, sizes, strides, name, vcp, us, ptype, dispatch_keys):
        super().__init__(root, vcp, us)
        self._name = name
        self._pt = ptype
        self._dtype = value.dtype
        self._dev = value.device
        self._rg = value.requires_grad
        self._layout = value.layout
        self._sizes = list(sizes) if sizes else list(value.size())
        self._strides = list(strides) if strides else list(value.stride())
        self._dim = len(self._sizes)
        self._keys = None
        try:
            import torch
            self._keys = torch._C._dispatch_keys(value)
        except Exception:
            pass

    def _fail(self, v):
        if type(v) is not self._pt:
            return f"expected type of '{self._name}' to be {self._pt}, but found {type(v)}"
        if v.dtype != self._dtype:
            return f"tensor '{self._name}' dtype mismatch. expected {self._dtype}, actual {v.dtype}"
        if v.device != self._dev:
            return f"tensor '{self._name}' device mismatch"
        if v.requires_grad != self._rg:
            return f"tensor '{self._name}' requires_grad mismatch. expected requires_grad={self._rg}"
        if v.layout != self._layout:
            return f"tensor '{self._name}' layout mismatch"
        if v.dim() != self._dim:
            return f"tensor '{self._name}' rank mismatch. expected {self._dim}, actual {v.dim()}"
        for i, k in enumerate(self._sizes):
            if k is not None and v.size(i) != k:
                return f"tensor '{self._name}' size mismatch at index {i}. expected {k}, actual {v.size(i)}"
        import torch
        if v.layout == torch.strided:
            for i, k in enumerate(self._strides):
                if k is not None and v.stride(i) != k:
                    return f"tensor '{self._name}' stride mismatch at index {i}. expected {k}, actual {v.stride(i)}"
        return None

    def check_nopybind(self, v):
        return self._fail(v) is None

    def check_verbose(self, v):
        r = self._fail(v)
        return GuardDebugInfo(r is None, [r] if r else [], 1 if r is None else 0)


# ---------------------------------------------------------------- accessors
class GuardAccessor:
    def __init__(self, root, key, source, example_value, enum_):
        self._key = key
        self._source = source
        self._mgr = _make_manager(root, source, example_value, enum_)

    def get_guard_manager(self):
        return self._mgr

    def get_source(self):
        return self._source

    def matches_key(self, key):
        try:
            return bool(self._key == key)
        except Exception:
            return False

    def repr(self):
        return type(self).__name__

    def get(self, v):
        raise NotImplementedError


def _func(v):
    return v.__func__ if isinstance(v, types.MethodType) else v


class GetAttrGuardAccessor(GuardAccessor):
    def get_attr_name(self):
        return self._key

    def get(self, v):
        return getattr(v, self._key)

    def repr(self):
        return f"GetAttrGuardAccessor({self._key})"


class GenericGetAttrGuardAccessor(GuardAccessor):
    def get(self, v):
        return object.__getattribute__(v, self._key)


class GetGenericDictGuardAccessor(GuardAccessor):
    def get(self, v):
        return object.__getattribute__(v, "__dict__")


class GetItemGuardAccessor(GuardAccessor):
    def get(self, v):
        return v[self._key]


class FrameLocalsGuardAccessor(GuardAccessor):
    def __init__(self, root, key, source, example_value, enum_):
        super().__init__(root, key[0], source, example_value, enum_)

    def get(self, v):
        return v[self._key]


class DictGetItemGuardAccessor(GuardAccessor):
    def get(self, v):
        # `type_dict_manager` hands over `type.__dict__`, a mappingproxy: upstream's C++
        # reads the type's real `tp_dict` there, and the proxy is its read-only view.
        if type(v) is types.MappingProxyType:
            return v[self._key]
        return dict.__getitem__(v, self._key)


class ListGetItemGuardAccessor(GuardAccessor):
    def get(self, v):
        if not isinstance(v, list):
            raise TypeError
        return list.__getitem__(v, self._key)


class SetGetItemGuardAccessor(GuardAccessor):
    def get(self, v):
        return list(v)[self._key]


class TupleGetItemGuardAccessor(GuardAccessor):
    def get(self, v):
        if not isinstance(v, tuple):
            raise TypeError
        return tuple.__getitem__(v, self._key)


class TensorPropertyGuardAccessor(GuardAccessor):
    prop = "size"

    def get(self, v):
        import torch
        if not isinstance(v, torch.Tensor):
            raise TypeError
        if self.prop == "storage_offset":
            return v.storage_offset()
        if v.dim() <= self._key:
            raise IndexError
        return v.size(self._key) if self.prop == "size" else v.stride(self._key)


class IndexedGuardAccessor(GuardAccessor):
    def get(self, v):
        return (self._key, v)


class GradGuardAccessor(GuardAccessor):
    def get(self, v):
        return v.grad


class FuncDefaultsGuardAccessor(GuardAccessor):
    def get(self, v):
        return _func(v).__defaults__


class FuncKwDefaultsGuardAccessor(GuardAccessor):
    def get(self, v):
        return _func(v).__kwdefaults__


class GlobalsGuardAccessor(GuardAccessor):
    def get(self, v):
        return self._key


class TypeGuardAccessor(GuardAccessor):
    def get(self, v):
        return type(v)


class TypeDictGuardAccessor(GuardAccessor):
    def get(self, v):
        return v.__dict__


class TypeMROGuardAccessor(GuardAccessor):
    def get(self, v):
        return v.__mro__


class TupleIteratorGetItemAccessor(GuardAccessor):
    def get(self, v):
        _, (tup,), idx = v.__reduce__()
        return tup[idx + self._key]


class GlobalWeakRefGuardAccessor(GuardAccessor):
    def get(self, v):
        w = dict.__getitem__(v, self._key)
        if not isinstance(w, weakref.ReferenceType):
            raise TypeError
        return w()


class WeakRefCallGuardAccessor(GuardAccessor):
    def get(self, v):
        if not isinstance(v, weakref.ReferenceType):
            raise TypeError
        return v()


class CodeGuardAccessor(GuardAccessor):
    def get(self, v):
        return _func(v).__code__


class ClosureGuardAccessor(GuardAccessor):
    def get(self, v):
        return _func(v).__closure__


class CallFunctionNoArgsGuardAccessor(GuardAccessor):
    def get(self, v):
        if not callable(v):
            raise TypeError
        return v()


class PythonLambdaGuardAccessor(GuardAccessor):
    def get(self, v):
        return self._key(v)

    def repr(self):
        return "PythonLambdaGuardAccessor"


class _SizeAcc(TensorPropertyGuardAccessor):
    prop = "size"


class _StrideAcc(TensorPropertyGuardAccessor):
    prop = "stride"


class _OffAcc(TensorPropertyGuardAccessor):
    prop = "storage_offset"


# ---------------------------------------------------------------- managers
class GuardManager:
    def __init__(self, root, source, example_value=None):
        self._root = root if root is not None else self
        self._source = source
        self._leaf = []
        self._acc = []
        self._fail = 0
        self._is_dict = isinstance(example_value, dict)
        self._type = type(example_value)
        self._immutable = example_value is None or isinstance(example_value, (int, float, bool, str, types.CodeType))
        self._tag_safe = False
        self._tag_safe_root = False
        self._obj_alias = False

    def get_root(self):
        return self._root

    def get_source(self):
        return self._source

    def fail_count(self):
        return self._fail

    def get_child_managers(self):
        return [a.get_guard_manager() for a in self._acc]

    def get_accessors(self):
        return list(self._acc)

    def get_leaf_guards(self):
        return list(self._leaf)

    def type_of_guarded_value(self):
        return str(self._type)

    def get_type_of_guarded_value(self):
        return self._type

    def is_guarded_value_immutable(self):
        return self._immutable

    def is_tag_safe(self):
        return self._tag_safe

    def is_tag_safe_root(self):
        return self._tag_safe_root

    def has_no_accessors(self):
        return not self._acc

    def has_object_aliasing_guard(self):
        return self._obj_alias

    def mark_tag_safe(self):
        self._tag_safe = True

    def mark_tag_safe_root(self):
        self._tag_safe_root = True

    def repr(self):
        return type(self).__name__

    def _child(self, cls, key, source, ev, en):
        for a in self._acc:
            if a.matches_key(key) and a.get_source() == source and type(a) is cls:
                return a.get_guard_manager()
        a = cls(self._root, key, source, ev, en)
        self._acc.append(a)
        return a.get_guard_manager()

    # Accessor creators
    def globals_dict_manager(self, f_globals, source, ev, en):
        return self._child(GlobalsGuardAccessor, f_globals, source, ev, en)

    def framelocals_manager(self, key, source, ev, en):
        return self._child(FrameLocalsGuardAccessor, key, source, ev, en)

    def dict_getitem_manager(self, key, source, ev, en):
        return self._child(DictGetItemGuardAccessor, key, source, ev, en)

    def grad_manager(self, source, ev, en):
        return self._child(GradGuardAccessor, "grad", source, ev, en)

    def generic_getattr_manager(self, attr, source, ev, en):
        return self._child(GenericGetAttrGuardAccessor, attr, source, ev, en)

    def getitem_manager(self, key, source, ev, en):
        return self._child(GetItemGuardAccessor, key, source, ev, en)

    def get_generic_dict_manager(self, source, ev, en):
        return self._child(GetGenericDictGuardAccessor, "__dict__", source, ev, en)

    def list_getitem_manager(self, key, source, ev, en):
        return self._child(ListGetItemGuardAccessor, key, source, ev, en)

    def tuple_getitem_manager(self, key, source, ev, en):
        return self._child(TupleGetItemGuardAccessor, key, source, ev, en)

    def set_getitem_manager(self, index, source, ev, en):
        return self._child(SetGetItemGuardAccessor, index, source, ev, en)

    def func_defaults_manager(self, source, ev, en):
        return self._child(FuncDefaultsGuardAccessor, "__defaults__", source, ev, en)

    def func_kwdefaults_manager(self, source, ev, en):
        return self._child(FuncKwDefaultsGuardAccessor, "__kwdefaults__", source, ev, en)

    def tuple_iterator_getitem_manager(self, index, source, ev, en):
        return self._child(TupleIteratorGetItemAccessor, index, source, ev, en)

    def weakref_call_manager(self, source, ev, en):
        return self._child(WeakRefCallGuardAccessor, "__call__", source, ev, en)

    def call_function_no_args_manager(self, source, ev, en):
        return self._child(CallFunctionNoArgsGuardAccessor, "__call__", source, ev, en)

    def global_weakref_manager(self, global_name, source, ev, en):
        return self._child(GlobalWeakRefGuardAccessor, global_name, source, ev, en)

    def type_manager(self, source, ev, en):
        return self._child(TypeGuardAccessor, "__type__", source, ev, en)

    def getattr_manager(self, attr, source, ev, en):
        return self._child(GetAttrGuardAccessor, attr, source, ev, en)

    def tensor_property_size_manager(self, idx, source, ev, en):
        return self._child(_SizeAcc, idx, source, ev, en)

    def tensor_property_shape_manager(self, idx, source, ev, en):
        return self._child(_StrideAcc, idx, source, ev, en)

    def tensor_property_storage_offset_manager(self, idx, source, ev, en):
        return self._child(_OffAcc, idx, source, ev, en)

    def indexed_manager(self, idx, source, ev, en):
        return self._child(IndexedGuardAccessor, idx, source, ev, en)

    def lambda_manager(self, fn, source, ev, en):
        return self._child(PythonLambdaGuardAccessor, fn, source, ev, en)

    def type_dict_manager(self, source, ev, en):
        return self._child(TypeDictGuardAccessor, "__dict__", source, ev, en)

    def type_mro_manager(self, source, ev, en):
        return self._child(TypeMROGuardAccessor, "__mro__", source, ev, en)

    def code_manager(self, source, ev, en):
        return self._child(CodeGuardAccessor, "__code__", source, ev, en)

    def closure_manager(self, source, ev, en):
        return self._child(ClosureGuardAccessor, "__closure__", source, ev, en)

    # Leaf guards
    def _add(self, g):
        self._leaf.append(g)

    def add_lambda_guard(self, fn, vcp, us):
        self._add(LAMBDA_GUARD(self._root, fn, vcp, us))

    def add_id_match_guard(self, v, vcp, us):
        self._add(ID_MATCH(self._root, v, vcp, us))

    def add_equals_match_guard(self, v, vcp, us):
        self._add(EQUALS_MATCH(self._root, v, vcp, us))

    def add_global_state_guard(self, st, vcp, us):
        self._add(GLOBAL_STATE(self._root, st, vcp, us))

    def add_torch_function_mode_stack_guard(self, st, vcp, us):
        self._add(TORCH_FUNCTION_MODE_STACK(self._root, st, vcp, us))

    def add_mapping_keys_guard(self, v, vcp, us):
        self._add(MAPPING_KEYS_MATCH(self._root, v, vcp, us))

    def add_dict_length_check_guard(self, v, vcp, us):
        self._add(DICT_LENGTH(self._root, v, vcp, us))

    def add_length_check_guard(self, v, vcp, us):
        self._add(LENGTH_CHECK(self._root, v, vcp, us))

    def add_true_match_guard(self, vcp, us):
        self._add(TRUE_MATCH(self._root, vcp, us))

    def add_false_match_guard(self, vcp, us):
        self._add(FALSE_MATCH(self._root, vcp, us))

    def add_none_match_guard(self, vcp, us):
        self._add(NONE_MATCH(self._root, vcp, us))

    def add_not_none_guard(self, vcp, us):
        self._add(NOT_NONE(self._root, vcp, us))

    def add_dispatch_key_set_guard(self, k, vcp, us):
        self._add(DISPATCH_KEY_SET_MATCH(self._root, k, vcp, us))

    def add_tensor_match_guard(self, value, sizes, strides, name, vcp, us, ptype, dk):
        self._add(TENSOR_MATCH(self._root, value, sizes, strides, name, vcp, us, ptype, dk))

    def add_dimension_marking_guard(self, e, a, d, vcp, us):
        self._add(DIMENSION_DYNAMIC_MARKING_GUARD(self._root, e, a, d, vcp, us))

    def add_no_hasattr_guard(self, n, vcp, us):
        self._add(NO_HASATTR(self._root, n, vcp, us))

    def add_dict_contains_guard(self, c, k, vcp, us):
        self._add(DICT_CONTAINS(self._root, c, k, vcp, us))

    def add_type_match_guard(self, v, vcp, us):
        self._add(TYPE_MATCH(self._root, v, vcp, us))

    def add_fake_script_type_match_guard(self, cls, tid, vcp, us):
        self._add(FAKE_SCRIPT_TYPE_MATCH(self._root, cls, tid, vcp, us))

    def add_dict_version_guard(self, v, vcp, us):
        self._add(DICT_VERSION(self._root, v, vcp, us))

    def add_set_contains_guard(self, c, i, vcp, us):
        self._add(SET_CONTAINS(self._root, c, i, vcp, us))

    def add_dual_level_match_guard(self, l, vcp, us):
        self._add(DUAL_LEVEL_MATCH(self._root, l, vcp, us))

    def add_float_is_nan_guard(self, vcp, us):
        self._add(FLOAT_IS_NAN(self._root, vcp, us))

    def add_complex_is_nan_guard(self, vcp, us):
        self._add(COMPLEX_IS_NAN(self._root, vcp, us))

    def add_tuple_iterator_length_guard(self, n, tid, vcp, us):
        self._add(TUPLE_ITERATOR_LEN(self._root, n, tid, vcp, us))

    def add_range_iterator_match_guard(self, a, b, c, tid, vcp, us):
        self._add(RANGE_ITERATOR_MATCH(self._root, a, b, c, tid, vcp, us))

    def add_default_device_guard(self, vcp, us):
        self._add(DEFAULT_DEVICE(self._root, vcp, us))

    # Evaluation
    def _check_leaf(self, v):
        for g in self._leaf:
            if not g.check_nopybind(v):
                self._fail += 1
                return False
        return True

    def _acc_value(self, a, v):
        try:
            return True, a.get(v)
        except Exception:
            return False, None

    def _check_acc(self, v):
        for a in self._acc:
            ok, x = self._acc_value(a, v)
            if not ok or not a.get_guard_manager()._check(x):
                self._fail += 1
                return False
        return True

    def _check(self, v):
        return self._check_leaf(v) and self._check_acc(v)

    def _verbose(self, v):
        n = 0
        for g in self._leaf:
            r = g.check_verbose(v)
            n += 1
            if not r.result:
                return GuardDebugInfo(False, r.verbose_code_parts, n, r.user_stack)
        for a in self._acc:
            ok, x = self._acc_value(a, v)
            if not ok:
                return GuardDebugInfo(False, [f"getattr/getitem failed on source {a.get_source()}"], n)
            r = a.get_guard_manager()._verbose(x)
            n += r.num_guards_executed
            if not r.result:
                return GuardDebugInfo(False, r.verbose_code_parts, n, r.user_stack)
        return GuardDebugInfo(True, [], n)

    def check(self, value):
        return self._check(value)

    def check_verbose(self, value):
        return self._verbose(value)

    def _clone_into(self, cloned_root, to, filt):
        for g in self._leaf:
            to._leaf.append(g)
        for a in self._acc:
            m = a.get_guard_manager()._clone(cloned_root, filt)
            if m is not None:
                na = object.__new__(type(a))
                na.__dict__.update(a.__dict__)
                na._mgr = m
                to._acc.append(na)

    def _clone(self, cloned_root, filt):
        if not filt(self):
            return None
        m = type(self).__new__(type(self))
        m.__dict__.update(self.__dict__)
        m._root = cloned_root
        m._leaf = []
        m._acc = []
        self._clone_into(cloned_root, m, filt)
        return m


class RootGuardManager(GuardManager):
    def __init__(self):
        super().__init__(None, "L")
        self._epilogue = []
        self._compile_id = None

    def get_epilogue_lambda_guards(self):
        return list(self._epilogue)

    def add_epilogue_lambda_guard(self, guard, vcp, us):
        self._epilogue.append(guard if isinstance(guard, LeafGuard) else LAMBDA_GUARD(self, guard, vcp, us))

    def attach_compile_id(self, cid):
        self._compile_id = cid

    def _reset_relational(self):
        def walk(m):
            for g in m._leaf:
                if isinstance(g, RelationalGuard):
                    g.reset_state()
            for c in m.get_child_managers():
                walk(c)
            if isinstance(m, DictGuardManager):
                for k, v in m._kv.values():
                    for x in (k, v):
                        if x is not None:
                            walk(x)
        walk(self)

    def _eval(self, value, verbose):
        td = None
        if "torch" in sys.modules:
            torch_mod = sys.modules["torch"]
            C = getattr(torch_mod, "_C", None)
            if C is not None and hasattr(C, "DisableTorchFunction"):
                dtf = getattr(C, "DisableTorchFunction")
                if type(dtf).__name__ != "_Unimplemented":
                    try:
                        td = dtf()
                    except (NotImplementedError, TypeError):
                        td = None
        try:
            if td is not None and hasattr(td, "__enter__"):
                td.__enter__()
            try:
                if verbose:
                    r = GuardManager._verbose(self, value)
                    if r.result:
                        for g in self._epilogue:
                            rr = g.check_verbose(value)
                            if not rr.result:
                                return rr
                    return r
                if not GuardManager._check(self, value):
                    return False
                return all(g.check_nopybind(value) for g in self._epilogue)
            finally:
                if td is not None and hasattr(td, "__exit__"):
                    td.__exit__(None, None, None)
        finally:
            self._reset_relational()

    def check(self, value):
        return self._eval(value, False)

    def check_verbose(self, value):
        return self._eval(value, True)

    def _check(self, v):
        return self._eval(v, False)

    def _verbose(self, v):
        return self._eval(v, True)

    def clone_manager(self, filt):
        if not filt(self):
            return None
        r = RootGuardManager()
        r._epilogue = list(self._epilogue)
        self._clone_into(r, r, filt)
        return r


class DictGuardManager(GuardManager):
    def __init__(self, root, source, example_value):
        super().__init__(root, source, example_value)
        self._size = len(example_value)
        self._expected_type = type(example_value)
        self._exact = type(example_value) is dict
        self._kv = {}

    def _idx(self, i):
        return self._kv.setdefault(i, [None, None])

    def get_key_manager(self, index, source, ev, en):
        e = self._idx(index)
        if e[0] is None:
            e[0] = _make_manager(self._root, source, ev, en)
        return e[0]

    def get_value_manager(self, index, source, ev, en):
        e = self._idx(index)
        if e[1] is None:
            e[1] = _make_manager(self._root, source, ev, en)
        return e[1]

    def get_key_value_managers(self):
        return {i: (k, v) for i, (k, v) in self._kv.items()}

    def get_child_managers(self):
        return super().get_child_managers()

    def _check(self, v):
        if type(v) is not self._expected_type or dict.__len__(v) != self._size:
            self._fail += 1
            return False
        if self._size == 0:
            return True
        if not GuardManager._check(self, v):
            self._fail += 1
            return False
        items = list(dict.items(v))
        for i in sorted(self._kv):
            k, val = items[i]
            km, vm = self._kv[i]
            if km is not None and not km._check(k):
                self._fail += 1
                return False
            if vm is not None and not vm._check(val):
                self._fail += 1
                return False
        return True

    def _verbose(self, v):
        if type(v) is not self._expected_type:
            return GuardDebugInfo(False, [f"dict type mismatch: expected {self._expected_type}, actual {type(v)}"], 0)
        if dict.__len__(v) != self._size:
            return GuardDebugInfo(False, [f"dict length mismatch: expected {self._size}, actual {dict.__len__(v)}"], 0)
        if self._size == 0:
            return GuardDebugInfo(True, [], 0)
        r = GuardManager._verbose(self, v)
        if not r.result:
            return r
        items = list(dict.items(v))
        for i in sorted(self._kv):
            k, val = items[i]
            km, vm = self._kv[i]
            if km is not None:
                kr = km._verbose(k)
                if not kr.result:
                    return kr
            if vm is not None:
                vr = vm._verbose(val)
                if not vr.result:
                    return vr
        return GuardDebugInfo(True, [], r.num_guards_executed)

    def _clone(self, cloned_root, filt):
        if not filt(self):
            return None
        m = type(self).__new__(type(self))
        m.__dict__.update(self.__dict__)
        m._root = cloned_root
        m._leaf = []
        m._acc = []
        m._kv = {}
        self._clone_into(cloned_root, m, filt)
        for i, (k, v) in self._kv.items():
            m._kv[i] = [k._clone(cloned_root, filt) if k else None, v._clone(cloned_root, filt) if v else None]
        return m


class GuardManagerType(enum.Enum):
    GUARD_MANAGER = 1
    DICT_GUARD_MANAGER = 2


def _make_manager(root, source, example_value, enum_):
    if isinstance(example_value, dict):
        name = getattr(enum_, "name", "")
        if name == "DICT_GUARD_MANAGER":
            return DictGuardManager(root, source, example_value)
    return GuardManager(root, source, example_value)


# ---------------------------------------------------------------- free functions
def check_obj_id(obj, expected):
    return id(obj) == expected


def check_type_id(obj, expected):
    return id(type(obj)) == expected


def install_object_aliasing_guard(x, y, vcp, us):
    g = OBJECT_ALIASING(x.get_root(), vcp, us)
    x._add(g)
    y._add(g)
    x._obj_alias = y._obj_alias = True


def install_no_tensor_aliasing_guard(mgrs, names, vcp, us):
    g = NO_TENSOR_ALIASING(mgrs[0].get_root(), names, vcp, us)
    for m in mgrs:
        m._add(g)


def install_storage_overlapping_guard(ov, non, vcp, us):
    ck = _StorageOverlapChecker(len(ov), len(non))
    g1 = STORAGE_OVERLAPPING(ov[0].get_root() if ov else non[0].get_root(), True, ck, vcp, us)
    g2 = STORAGE_OVERLAPPING(g1._root, False, ck, vcp, us)
    for m in ov:
        m._add(g1)
    for m in non:
        m._add(g2)


def install_symbolic_shape_guard(*a, **k):
    raise NotImplementedError(
        "C-function-pointer shape guards are not available; set enable_cpp_symbolic_shape_guards=False"
    )


def profile_guard_manager(gm, f_locals, n):
    import time
    t = time.perf_counter()
    for _ in range(n):
        gm.check(f_locals)
    return (time.perf_counter() - t) / n


_mode_state = {"value": False}


def set_is_in_mode_without_ignore_compile_internals(v):
    _mode_state["value"] = bool(v)


def is_in_mode_without_ignore_compile_internals():
    return _mode_state["value"]


class TensorGuards:
    def __init__(self, *a, **k):
        raise NotImplementedError("TensorGuards (legacy)")


def _bind_pyi_keywords():
    """Make the manager/leaf-guard methods accept the keyword names declared in guards.pyi."""
    import ast
    import functools
    import os

    candidates = []
    if "torch" in sys.modules:
        torch_mod = sys.modules["torch"]
        if hasattr(torch_mod, "__file__") and torch_mod.__file__:
            candidates.append(os.path.join(os.path.dirname(torch_mod.__file__), "_C", "_dynamo", "guards.pyi"))
            candidates.append(os.path.join(os.path.dirname(torch_mod.__file__), "..", "torch", "_C", "_dynamo", "guards.pyi"))
    repo_cand = "/Volumes/macMini/thisisthepy/torchnative/torchnative/python/torch/_C/_dynamo/guards.pyi"
    candidates.append(repo_cand)

    path = None
    for cand in candidates:
        if os.path.isfile(cand):
            path = cand
            break

    if path is None:
        return

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        tree = ast.parse(content)
    except Exception:
        return

    for cls_node in (n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in ("GuardManager", "RootGuardManager", "DictGuardManager")):
        target = {"GuardManager": GuardManager, "RootGuardManager": RootGuardManager, "DictGuardManager": DictGuardManager}[cls_node.name]
        for fn in (n for n in cls_node.body if isinstance(n, ast.FunctionDef)):
            impl = target.__dict__.get(fn.name)
            if impl is None or getattr(impl, "_pyi_bound", False):
                continue
            names = [a.arg for a in fn.args.args][1:]
            def make(orig_impl, arg_names):
                @functools.wraps(orig_impl)
                def w(self, *args, **kw):
                    if kw:
                        args = list(args)
                        for n in arg_names[len(args):]:
                            if n in kw:
                                args.append(kw.pop(n))
                            else:
                                break
                        if kw:
                            raise TypeError(f"{orig_impl.__name__}() got unexpected keyword arguments {sorted(kw)}")
                    return orig_impl(self, *args)
                w._pyi_bound = True
                return w
            setattr(target, fn.name, make(impl, names))


def install(C, put=None):
    """Populate torch._C._dynamo.guards with all guard classes, accessors, and functions."""
    prefix = C.__name__
    dynamo = getattr(C, "_dynamo", None)
    if dynamo is None or type(dynamo).__name__ == "_Unimplemented":
        dynamo = types.ModuleType(f"{prefix}._dynamo")
        dynamo.__path__ = []
        setattr(C, "_dynamo", dynamo)
    sys.modules[f"{prefix}._dynamo"] = dynamo

    guards = getattr(dynamo, "guards", None)
    if guards is None or type(guards).__name__ == "_Unimplemented":
        guards = types.ModuleType(f"{prefix}._dynamo.guards")
        guards.__path__ = []
        setattr(dynamo, "guards", guards)
    sys.modules[f"{prefix}._dynamo.guards"] = guards

    def _set(target, name, val, full_name):
        if put is not None:
            put(target, name, val, full_name)
        else:
            setattr(target, name, val)

    # Free functions and state
    _set(guards, "set_is_in_mode_without_ignore_compile_internals",
         set_is_in_mode_without_ignore_compile_internals,
         "_dynamo.guards.set_is_in_mode_without_ignore_compile_internals")
    _set(guards, "is_in_mode_without_ignore_compile_internals",
         is_in_mode_without_ignore_compile_internals,
         "_dynamo.guards.is_in_mode_without_ignore_compile_internals")

    _set(guards, "check_obj_id", check_obj_id, "_dynamo.guards.check_obj_id")
    _set(guards, "check_type_id", check_type_id, "_dynamo.guards.check_type_id")
    _set(guards, "dict_version", dict_version, "_dynamo.guards.dict_version")
    _set(guards, "install_object_aliasing_guard", install_object_aliasing_guard, "_dynamo.guards.install_object_aliasing_guard")
    _set(guards, "install_no_tensor_aliasing_guard", install_no_tensor_aliasing_guard, "_dynamo.guards.install_no_tensor_aliasing_guard")
    _set(guards, "install_storage_overlapping_guard", install_storage_overlapping_guard, "_dynamo.guards.install_storage_overlapping_guard")
    _set(guards, "compute_overlapping_tensors", compute_overlapping_tensors, "_dynamo.guards.compute_overlapping_tensors")
    _set(guards, "install_symbolic_shape_guard", install_symbolic_shape_guard, "_dynamo.guards.install_symbolic_shape_guard")
    _set(guards, "profile_guard_manager", profile_guard_manager, "_dynamo.guards.profile_guard_manager")

    # Classes
    classes = [
        ("GuardDebugInfo", GuardDebugInfo),
        ("GlobalStateGuard", GlobalStateGuard),
        ("LeafGuard", LeafGuard),
        ("RelationalGuard", RelationalGuard),
        ("LAMBDA_GUARD", LAMBDA_GUARD),
        ("TYPE_MATCH", TYPE_MATCH),
        ("FAKE_SCRIPT_TYPE_MATCH", FAKE_SCRIPT_TYPE_MATCH),
        ("ID_MATCH", ID_MATCH),
        ("NONE_MATCH", NONE_MATCH),
        ("NOT_NONE", NOT_NONE),
        ("TRUE_MATCH", TRUE_MATCH),
        ("FALSE_MATCH", FALSE_MATCH),
        ("EQUALS_MATCH", EQUALS_MATCH),
        ("RANGE_ITERATOR_MATCH", RANGE_ITERATOR_MATCH),
        ("TUPLE_ITERATOR_LEN", TUPLE_ITERATOR_LEN),
        ("LENGTH_CHECK", LENGTH_CHECK),
        ("DICT_LENGTH", DICT_LENGTH),
        ("MAPPING_KEYS_MATCH", MAPPING_KEYS_MATCH),
        ("DEFAULT_DEVICE", DEFAULT_DEVICE),
        ("GLOBAL_STATE", GLOBAL_STATE),
        ("NO_HASATTR", NO_HASATTR),
        ("DICT_CONTAINS", DICT_CONTAINS),
        ("SET_CONTAINS", SET_CONTAINS),
        ("FLOAT_IS_NAN", FLOAT_IS_NAN),
        ("COMPLEX_IS_NAN", COMPLEX_IS_NAN),
        ("DUAL_LEVEL_MATCH", DUAL_LEVEL_MATCH),
        ("OBJECT_ALIASING", OBJECT_ALIASING),
        ("NO_TENSOR_ALIASING", NO_TENSOR_ALIASING),
        ("STORAGE_OVERLAPPING", STORAGE_OVERLAPPING),
        ("SYMBOLIC_SHAPE_GUARD", SYMBOLIC_SHAPE_GUARD),
        ("DIMENSION_DYNAMIC_MARKING_GUARD", DIMENSION_DYNAMIC_MARKING_GUARD),
        ("DICT_VERSION", DICT_VERSION),
        ("TORCH_FUNCTION_MODE_STACK", TORCH_FUNCTION_MODE_STACK),
        ("DISPATCH_KEY_SET_MATCH", DISPATCH_KEY_SET_MATCH),
        ("TENSOR_MATCH", TENSOR_MATCH),
        ("GuardAccessor", GuardAccessor),
        ("GetAttrGuardAccessor", GetAttrGuardAccessor),
        ("GenericGetAttrGuardAccessor", GenericGetAttrGuardAccessor),
        ("GetGenericDictGuardAccessor", GetGenericDictGuardAccessor),
        ("GetItemGuardAccessor", GetItemGuardAccessor),
        ("FrameLocalsGuardAccessor", FrameLocalsGuardAccessor),
        ("DictGetItemGuardAccessor", DictGetItemGuardAccessor),
        ("ListGetItemGuardAccessor", ListGetItemGuardAccessor),
        ("SetGetItemGuardAccessor", SetGetItemGuardAccessor),
        ("TupleGetItemGuardAccessor", TupleGetItemGuardAccessor),
        ("TensorPropertyGuardAccessor", TensorPropertyGuardAccessor),
        ("IndexedGuardAccessor", IndexedGuardAccessor),
        ("GradGuardAccessor", GradGuardAccessor),
        ("FuncDefaultsGuardAccessor", FuncDefaultsGuardAccessor),
        ("FuncKwDefaultsGuardAccessor", FuncKwDefaultsGuardAccessor),
        ("GlobalsGuardAccessor", GlobalsGuardAccessor),
        ("TypeGuardAccessor", TypeGuardAccessor),
        ("TypeDictGuardAccessor", TypeDictGuardAccessor),
        ("TypeMROGuardAccessor", TypeMROGuardAccessor),
        ("TupleIteratorGetItemAccessor", TupleIteratorGetItemAccessor),
        ("GlobalWeakRefGuardAccessor", GlobalWeakRefGuardAccessor),
        ("WeakRefCallGuardAccessor", WeakRefCallGuardAccessor),
        ("CodeGuardAccessor", CodeGuardAccessor),
        ("ClosureGuardAccessor", ClosureGuardAccessor),
        ("CallFunctionNoArgsGuardAccessor", CallFunctionNoArgsGuardAccessor),
        ("PythonLambdaGuardAccessor", PythonLambdaGuardAccessor),
        ("GuardManager", GuardManager),
        ("RootGuardManager", RootGuardManager),
        ("DictGuardManager", DictGuardManager),
        ("GuardManagerType", GuardManagerType),
        ("TensorGuards", TensorGuards),
    ]

    for name, cls in classes:
        _set(guards, name, cls, f"_dynamo.guards.{name}")

    _bind_pyi_keywords()
