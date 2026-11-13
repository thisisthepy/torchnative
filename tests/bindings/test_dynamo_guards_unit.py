"""Unit test suite: torch._C._dynamo.guards per-leaf and per-accessor semantics.

Issue #207, SPEC S1.8:
Pure-Python `torch._C._dynamo.guards` implements GuardManager, RootGuardManager,
DictGuardManager, every leaf guard kind, accessor kind, relational guard,
epilogue lambdas, and clone_manager.

Three test suites:
1. `test_every_leaf_guard_kind_passes_and_fails_with_verbose_diagnostics`:
   Tests every leaf guard kind individually on pass, fail, and check_verbose.
2. `test_every_accessor_kind_navigates_and_guards_child_values`:
   Tests every accessor kind individually, accessor deduplication, and failure diagnostics.
3. `test_dict_guard_manager_relational_guards_epilogue_and_clone`:
   Tests DictGuardManager, relational guards with reset_state across check calls,
   epilogue lambdas, clone_manager with filter predicates, free functions,
   and guards.pyi keyword argument binding.
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

gm = getattr(getattr(sys.modules.get("torch", None), "_C", None), "_dynamo", None)
gm = getattr(gm, "guards", None)

import torch
gm = getattr(getattr(torch._C, "_dynamo", None), "guards", None)
has_root = gm is not None and hasattr(gm, "RootGuardManager")
check_fn = getattr(gm.RootGuardManager, "check", None) if has_root else None
has_check = check_fn is not None and type(check_fn).__name__ != "_Unimplemented"

if not has_check and os.environ.get("TORCHNATIVE_NO_GUARDS_FALLBACK") != "1":
    try:
        import torchnative._dynamo_guards as _dg
        _dg.install(torch._C)
        gm = getattr(getattr(torch._C, "_dynamo", None), "guards", None)
    except ImportError:
        pass

if hasattr(torch._C, "DisableTorchFunction"):
    dtf = getattr(torch._C, "DisableTorchFunction")
    if type(dtf).__name__ == "_Unimplemented":
        class _StubDisableTorchFunction:
            def __enter__(self): pass
            def __exit__(self, *args): pass
        torch._C.DisableTorchFunction = _StubDisableTorchFunction
    else:
        try:
            dtf()
        except (NotImplementedError, TypeError):
            class _StubDisableTorchFunction:
                def __enter__(self): pass
                def __exit__(self, *args): pass
            torch._C.DisableTorchFunction = _StubDisableTorchFunction

has_root = gm is not None and hasattr(gm, "RootGuardManager")
check_fn = getattr(gm.RootGuardManager, "check", None) if has_root else None
has_check = check_fn is not None and type(check_fn).__name__ != "_Unimplemented"

if not has_check:
    print(json.dumps({
        "status": "unimplemented",
        "refusal": "not implemented in torch._C shim: _dynamo.guards.RootGuardManager"
    }))
    sys.exit(0)

# Import necessary dependencies for unit testing
import types
import weakref
from torch.overrides import TorchFunctionMode
import torch.autograd.forward_ad as fwAD


def run_leaf_tests():
    # 1. TYPE_MATCH
    r_type = gm.RootGuardManager()
    r_type.add_type_match_guard(id(int), ["type must be int"], None)
    assert r_type.check(10) is True
    assert r_type.check(10.5) is False
    assert r_type.check_verbose(10).result is True
    dbg_type = r_type.check_verbose(10.5)
    assert dbg_type.result is False
    assert len(dbg_type.verbose_code_parts) > 0

    # 2. ID_MATCH
    target_obj = object()
    r_id = gm.RootGuardManager()
    r_id.add_id_match_guard(id(target_obj), ["id must match target_obj"], None)
    assert r_id.check(target_obj) is True
    assert r_id.check(object()) is False
    assert r_id.check_verbose(target_obj).result is True
    assert r_id.check_verbose(object()).result is False

    # 3. EQUALS_MATCH
    r_eq = gm.RootGuardManager()
    r_eq.add_equals_match_guard(42, ["val must equal 42"], None)
    assert r_eq.check(42) is True
    assert r_eq.check(42.0) is False  # Type mismatch
    assert r_eq.check(43) is False    # Value mismatch
    assert r_eq.check_verbose(42).result is True
    assert r_eq.check_verbose(43).result is False

    # 4. LENGTH_CHECK
    r_len = gm.RootGuardManager()
    r_len.add_length_check_guard(3, ["len must be 3"], None)
    assert r_len.check([1, 2, 3]) is True
    assert r_len.check([1, 2]) is False
    assert r_len.check(123) is False

    # 5. DICT_LENGTH
    r_dlen = gm.RootGuardManager()
    r_dlen.add_dict_length_check_guard(2, ["dict len must be 2"], None)
    assert r_dlen.check({"a": 1, "b": 2}) is True
    assert r_dlen.check({"a": 1}) is False
    assert r_dlen.check([1, 2]) is False

    # 6. MAPPING_KEYS_MATCH
    r_keys = gm.RootGuardManager()
    r_keys.add_mapping_keys_guard({"x": 1, "y": 2}, ["keys must match"], None)
    assert r_keys.check({"x": 10, "y": 20}) is True
    assert r_keys.check({"y": 20, "x": 10}) is False
    assert r_keys.check({"x": 1}) is False

    # 7. DEFAULT_DEVICE
    r_dev = gm.RootGuardManager()
    r_dev.add_default_device_guard(["default device"], None)
    assert r_dev.check(None) is True

    # 8. GLOBAL_STATE
    st = gm.GlobalStateGuard()
    r_gs = gm.RootGuardManager()
    r_gs.add_global_state_guard(st, ["global state"], None)
    assert r_gs.check(None) is True
    orig_grad = torch.is_grad_enabled()
    try:
        torch.set_grad_enabled(not orig_grad)
        assert r_gs.check(None) is False
    finally:
        torch.set_grad_enabled(orig_grad)

    # 9. TORCH_FUNCTION_MODE_STACK
    r_mode = gm.RootGuardManager()
    r_mode.add_torch_function_mode_stack_guard([], ["mode stack"], None)
    assert r_mode.check(None) is True
    class DummyMode(TorchFunctionMode): pass
    with DummyMode():
        assert r_mode.check(None) is False

    # 10. DISPATCH_KEY_SET_MATCH
    ks = torch._C.DispatchKeySet(torch._C.DispatchKey.CPU)
    r_ks = gm.RootGuardManager()
    r_ks.add_dispatch_key_set_guard(ks, ["keyset"], None)
    assert r_ks.check(ks) is True
    ks_diff = torch._C.DispatchKeySet(torch._C.DispatchKey.CUDA)
    assert r_ks.check(ks_diff) is False

    # 11. TENSOR_MATCH
    t = torch.randn(2, 3)
    r_tm = gm.RootGuardManager()
    r_tm.add_tensor_match_guard(t, [2, 3], list(t.stride()), "x", ["tensor match"], None, type(t), None)
    assert r_tm.check(t) is True
    # dtype mismatch
    t_f64 = t.to(torch.float64)
    assert r_tm.check(t_f64) is False
    dbg_dtype = r_tm.check_verbose(t_f64)
    assert dbg_dtype.result is False
    assert any("dtype mismatch" in p for p in dbg_dtype.verbose_code_parts)
    # requires_grad mismatch
    t_rg = t.clone().requires_grad_(True)
    assert r_tm.check(t_rg) is False
    dbg_rg = r_tm.check_verbose(t_rg)
    assert any("requires_grad mismatch" in p for p in dbg_rg.verbose_code_parts)
    # rank mismatch
    t_rank = t.unsqueeze(0)
    assert r_tm.check(t_rank) is False
    dbg_rank = r_tm.check_verbose(t_rank)
    assert any("rank mismatch" in p for p in dbg_rank.verbose_code_parts)
    # size mismatch
    t_size = torch.randn(2, 4)
    assert r_tm.check(t_size) is False
    dbg_size = r_tm.check_verbose(t_size)
    assert any("size mismatch" in p for p in dbg_size.verbose_code_parts)
    # stride mismatch
    t_stride = torch.randn(3, 2).t()
    if t_stride.stride() != t.stride():
        assert r_tm.check(t_stride) is False
        dbg_stride = r_tm.check_verbose(t_stride)
        assert any("stride mismatch" in p for p in dbg_stride.verbose_code_parts)

    # 12. DIMENSION_DYNAMIC_MARKING_GUARD
    r_ddm = gm.RootGuardManager()
    r_ddm.add_dimension_marking_guard({}, ["_forbidden_dyn"], {}, ["dim marking"], None)
    class DynObj: pass
    d_obj = DynObj()
    assert r_ddm.check(d_obj) is True
    d_obj._dynamo_has_dynamic_marking = True
    d_obj._forbidden_dyn = True
    assert r_ddm.check(d_obj) is False

    # 13. NO_HASATTR
    r_nha = gm.RootGuardManager()
    r_nha.add_no_hasattr_guard("forbidden_attr", ["no hasattr"], None)
    class AttrObj: pass
    a_obj = AttrObj()
    assert r_nha.check(a_obj) is True
    a_obj.forbidden_attr = 10
    assert r_nha.check(a_obj) is False

    # 14. DICT_CONTAINS
    r_dc = gm.RootGuardManager()
    r_dc.add_dict_contains_guard(True, "expected_key", ["dict contains"], None)
    assert r_dc.check({"expected_key": 1}) is True
    assert r_dc.check({}) is False

    # 15. SET_CONTAINS
    r_sc = gm.RootGuardManager()
    r_sc.add_set_contains_guard(True, "item", ["set contains"], None)
    assert r_sc.check({"item", "other"}) is True
    assert r_sc.check({"other"}) is False

    # 16. FAKE_SCRIPT_TYPE_MATCH
    class ScriptCls: pass
    r_fstm = gm.RootGuardManager()
    r_fstm.add_fake_script_type_match_guard(ScriptCls, id(ScriptCls), ["fake script type"], None)
    assert r_fstm.check(ScriptCls()) is True
    assert r_fstm.check(object()) is False

    # 17. DICT_VERSION
    d_ver = {"k1": 1, "k2": 2}
    r_dv = gm.RootGuardManager()
    r_dv.add_dict_version_guard(d_ver, ["dict version"], None)
    assert r_dv.check(d_ver) is True
    d_ver_mod = dict(d_ver)
    d_ver_mod["k3"] = 3
    assert r_dv.check(d_ver_mod) is False

    # 18. DUAL_LEVEL_MATCH
    r_dl = gm.RootGuardManager()
    r_dl.add_dual_level_match_guard(fwAD._current_level, ["dual level"], None)
    assert r_dl.check(None) is True

    # 19. FLOAT_IS_NAN
    r_fnan = gm.RootGuardManager()
    r_fnan.add_float_is_nan_guard(["float is nan"], None)
    assert r_fnan.check(float("nan")) is True
    assert r_fnan.check(1.0) is False

    # 20. COMPLEX_IS_NAN
    r_cnan = gm.RootGuardManager()
    r_cnan.add_complex_is_nan_guard(["complex is nan"], None)
    assert r_cnan.check(complex(float("nan"), 0.0)) is True
    assert r_cnan.check(complex(1.0, 1.0)) is False

    # 21. TUPLE_ITERATOR_LEN
    tup_ex = (1, 2, 3)
    it_tup = iter(tup_ex)
    r_ti = gm.RootGuardManager()
    r_ti.add_tuple_iterator_length_guard(3, id(type(it_tup)), ["tuple iterator len"], None)
    assert r_ti.check(it_tup) is True
    next(it_tup)
    assert r_ti.check(it_tup) is False

    # 22. RANGE_ITERATOR_MATCH
    # 22a. Active range iterator
    it_act = iter(range(0, 10, 2))
    next(it_act)  # advance to 2
    r_rim = gm.RootGuardManager()
    r_rim.add_range_iterator_match_guard(2, 10, 2, id(type(it_act)), ["range iter active"], None)
    assert r_rim.check(it_act) is True
    next(it_act)  # advance to 4
    assert r_rim.check(it_act) is False

    # 22b. Empty range iterator
    it_empty = iter(range(0, 0, 1))
    r_empty = gm.RootGuardManager()
    r_empty.add_range_iterator_match_guard(0, 0, 1, id(type(it_empty)), ["range iter empty"], None)
    assert r_empty.check(it_empty) is True

    # 22c. Exhausted range iterator in Python 3.13 (where __reduce__()[2] is None!)
    it_exhaust = iter(range(3))
    list(it_exhaust)  # fully consume
    r_exh = gm.RootGuardManager()
    r_exh.add_range_iterator_match_guard(3, 3, 1, id(type(it_exhaust)), ["range iter exhausted"], None)
    assert r_exh.check(it_exhaust) is True

    # 23. TRUE_MATCH
    r_tm = gm.RootGuardManager()
    r_tm.add_true_match_guard(["true match"], None)
    assert r_tm.check(True) is True
    assert r_tm.check(False) is False
    assert r_tm.check(1) is False

    # 24. FALSE_MATCH
    r_fm = gm.RootGuardManager()
    r_fm.add_false_match_guard(["false match"], None)
    assert r_fm.check(False) is True
    assert r_fm.check(True) is False
    assert r_fm.check(0) is False

    # 25. NONE_MATCH
    r_nm = gm.RootGuardManager()
    r_nm.add_none_match_guard(["none match"], None)
    assert r_nm.check(None) is True
    assert r_nm.check(0) is False
    assert r_nm.check(False) is False

    # 26. NOT_NONE
    r_nn = gm.RootGuardManager()
    r_nn.add_not_none_guard(["not none"], None)
    assert r_nn.check(0) is True
    assert r_nn.check("") is True
    assert r_nn.check(False) is True
    assert r_nn.check(None) is False

    # 27. LAMBDA_GUARD
    # Boolean pass/fail
    r_lg = gm.RootGuardManager()
    r_lg.add_lambda_guard(lambda x: x > 0, ["val must be > 0"], None)
    assert r_lg.check(5) is True
    assert r_lg.check(-1) is False
    dbg_lg = r_lg.check_verbose(-1)
    assert dbg_lg.result is False
    assert any("val must be > 0" in p for p in dbg_lg.verbose_code_parts)

    # Exception reporting in check_verbose
    r_lg_ex = gm.RootGuardManager()
    def raising_predicate(x):
        raise ValueError("deliberate failure in lambda guard")
    r_lg_ex.add_lambda_guard(raising_predicate, ["raising lambda"], None)
    assert r_lg_ex.check(10) is False
    dbg_lg_ex = r_lg_ex.check_verbose(10)
    assert dbg_lg_ex.result is False
    assert any("deliberate failure in lambda guard" in p for p in dbg_lg_ex.verbose_code_parts)


def run_accessor_tests():
    root = gm.RootGuardManager()
    GM = gm.GuardManagerType.GUARD_MANAGER

    # Helper target object
    class Target:
        attr = 100
        foo = 200
        def __init__(self):
            self.inst_dict_item = 300

    t_inst = Target()

    # 1. dict_getitem_manager
    m_dict = root.dict_getitem_manager("a", "L['a']", 1, GM)
    m_dict.add_equals_match_guard(1, [], None)

    # 2. framelocals_manager
    m_fl = root.framelocals_manager(("fl_key", 0), "L['fl_key']", 2, GM)
    m_fl.add_equals_match_guard(2, [], None)

    # 3. globals_dict_manager
    glob_dict = {"g_val": 42}
    m_glob = root.globals_dict_manager(glob_dict, "G", glob_dict, GM)
    m_g_item = m_glob.dict_getitem_manager("g_val", "G['g_val']", 42, GM)
    m_g_item.add_equals_match_guard(42, [], None)

    # 4. getattr_manager
    m_ga = root.getattr_manager("attr", "obj.attr", 100, GM)
    m_ga.add_equals_match_guard(100, [], None)

    # 5. generic_getattr_manager
    m_gga = root.generic_getattr_manager("foo", "obj.foo", 200, GM)
    m_gga.add_equals_match_guard(200, [], None)

    # 6. get_generic_dict_manager
    m_dict_acc = root.get_generic_dict_manager("obj.__dict__", t_inst.__dict__, GM)
    m_dict_item = m_dict_acc.dict_getitem_manager("inst_dict_item", "obj.__dict__['inst_dict_item']", 300, GM)
    m_dict_item.add_equals_match_guard(300, [], None)

    # 7. getitem_manager
    m_gi = root.getitem_manager("sub", "map['sub']", 9, GM)
    m_gi.add_equals_match_guard(9, [], None)

    # 8. list_getitem_manager
    m_lgi = root.list_getitem_manager(0, "lst[0]", 55, GM)
    m_lgi.add_equals_match_guard(55, [], None)

    # 9. tuple_getitem_manager
    m_tgi = root.tuple_getitem_manager(0, "tup[0]", 77, GM)
    m_tgi.add_equals_match_guard(77, [], None)

    # 10. set_getitem_manager
    m_sgi = root.set_getitem_manager(0, "list(s)[0]", "alpha", GM)
    m_sgi.add_equals_match_guard("alpha", [], None)

    # 11. func_defaults_manager
    def sample_func(x, y=10): return x + y
    m_fd = root.func_defaults_manager("fn.__defaults__", sample_func.__defaults__, GM)
    m_fd.add_equals_match_guard((10,), [], None)

    # 12. func_kwdefaults_manager
    def sample_kw_func(x, *, z=20): return x + z
    m_fkd = root.func_kwdefaults_manager("fn.__kwdefaults__", sample_kw_func.__kwdefaults__, GM)
    m_fkd.add_equals_match_guard({"z": 20}, [], None)

    # 13. tuple_iterator_getitem_manager
    it_tup = iter((101, 102))
    m_titer = root.tuple_iterator_getitem_manager(0, "tup_iter[0]", 101, GM)
    m_titer.add_equals_match_guard(101, [], None)

    # 14. weakref_call_manager
    ref_obj = Target()
    wref = weakref.ref(ref_obj)
    m_wc = root.weakref_call_manager("wref()", ref_obj, GM)
    m_wc.add_type_match_guard(id(Target), [], None)

    # 15. call_function_no_args_manager
    m_cf = root.call_function_no_args_manager("fn0()", 999, GM)
    m_cf.add_equals_match_guard(999, [], None)

    # 16. global_weakref_manager
    glob_wref_dict = {"ref_entry": wref}
    m_gw = root.global_weakref_manager("ref_entry", "G_W['ref_entry']()", ref_obj, GM)
    m_gw.add_type_match_guard(id(Target), [], None)

    # 17. type_manager
    m_tm = root.type_manager("type(obj)", Target, GM)
    m_tm.add_id_match_guard(id(Target), [], None)

    # 18. type_dict_manager
    m_td = root.type_dict_manager("Target.__dict__", Target.__dict__, GM)
    assert m_td is not None

    # 19. type_mro_manager
    m_tmro = root.type_mro_manager("Target.__mro__", Target.__mro__, GM)
    m_tmro.add_id_match_guard(id(Target.__mro__), [], None)

    # 20. code_manager
    m_code = root.code_manager("fn.__code__", sample_func.__code__, GM)
    m_code.add_id_match_guard(id(sample_func.__code__), [], None)

    # 21. closure_manager
    def outer():
        cl_val = 1234
        def inner(): return cl_val
        return inner
    inner_fn = outer()
    m_clo = root.closure_manager("inner.__closure__", inner_fn.__closure__, GM)
    m_clo.add_length_check_guard(1, [], None)

    # 22. tensor_property_size_manager
    t_prop = torch.randn(4, 8)
    m_sz = root.tensor_property_size_manager(0, "t.size(0)", 4, GM)
    m_sz.add_equals_match_guard(4, [], None)

    # 23. tensor_property_shape_manager (strides)
    m_str = root.tensor_property_shape_manager(0, "t.stride(0)", t_prop.stride(0), GM)
    m_str.add_equals_match_guard(t_prop.stride(0), [], None)

    # 24. tensor_property_storage_offset_manager
    m_off = root.tensor_property_storage_offset_manager(0, "t.storage_offset()", 0, GM)
    m_off.add_equals_match_guard(0, [], None)

    # 25. indexed_manager
    m_idx = root.indexed_manager(1, "indexed[1]", (1, "val"), GM)
    m_idx.add_equals_match_guard((1, "val"), [], None)

    # 26. lambda_manager
    m_lam = root.lambda_manager(lambda s: s.strip().upper(), "s.strip().upper()", "HELLO", GM)
    m_lam.add_equals_match_guard("HELLO", [], None)

    # 27. grad_manager
    t_grad = torch.randn(2, requires_grad=True)
    out_g = (t_grad * 2).sum()
    out_g.backward()
    m_grad = root.grad_manager("t.grad", t_grad.grad, GM)
    m_grad.add_type_match_guard(id(torch.Tensor), [], None)

    # Test Deduplication: calling _child/accessor creator with same key/source/cls returns existing manager
    child_dup1 = root.dict_getitem_manager("dedup_key", "L['dedup_key']", 1, GM)
    child_dup2 = root.dict_getitem_manager("dedup_key", "L['dedup_key']", 1, GM)
    assert child_dup1 is child_dup2, "Accessor creation must deduplicate identical accessors"

    # Test Accessor failure in check and check_verbose
    r_fail_test = gm.RootGuardManager()
    m_missing = r_fail_test.dict_getitem_manager("absent_key", "L['absent_key']", 1, GM)
    m_missing.add_equals_match_guard(1, [], None)
    assert r_fail_test.check({}) is False
    dbg_acc_fail = r_fail_test.check_verbose({})
    assert dbg_acc_fail.result is False
    assert any("getattr/getitem failed" in p for p in dbg_acc_fail.verbose_code_parts)


def run_dict_relational_clone_tests():
    root = gm.RootGuardManager()
    GM = gm.GuardManagerType.GUARD_MANAGER
    DGM = gm.GuardManagerType.DICT_GUARD_MANAGER

    # 1. DictGuardManager
    sample_dict = {"alpha": 10, "beta": 20}
    dict_mgr = root.dict_getitem_manager("d", "L['d']", sample_dict, DGM)
    assert type(dict_mgr).__name__ == "DictGuardManager"
    k0 = dict_mgr.get_key_manager(0, "d.keys()[0]", "alpha", GM)
    v0 = dict_mgr.get_value_manager(0, "d['alpha']", 10, GM)
    kv_map = dict_mgr.get_key_value_managers()
    assert 0 in kv_map and kv_map[0][0] is k0 and kv_map[0][1] is v0
    k0.add_equals_match_guard("alpha", [], None)
    v0.add_equals_match_guard(10, [], None)

    assert root.check({"d": {"alpha": 10, "beta": 20}}) is True
    assert root.check({"d": {"alpha": 99, "beta": 20}}) is False  # Val mismatch
    assert root.check({"d": {"alpha": 10}}) is False                # Size mismatch
    assert root.check({"d": [10, 20]}) is False                     # Type mismatch

    # 2. install_object_aliasing_guard and reset_state across calls
    r_alias = gm.RootGuardManager()
    m_a = r_alias.dict_getitem_manager("a", "L['a']", None, GM)
    m_b = r_alias.dict_getitem_manager("b", "L['b']", None, GM)
    gm.install_object_aliasing_guard(m_a, m_b, ["a must alias b"], None)
    shared_obj = object()
    assert r_alias.check({"a": shared_obj, "b": shared_obj}) is True
    assert r_alias.check({"a": object(), "b": object()}) is False
    # Verify state reset
    assert r_alias.check({"a": shared_obj, "b": shared_obj}) is True

    # 3. install_no_tensor_aliasing_guard and reset_state across calls
    r_nta = gm.RootGuardManager()
    m_t1 = r_nta.dict_getitem_manager("t1", "L['t1']", None, GM)
    m_t2 = r_nta.dict_getitem_manager("t2", "L['t2']", None, GM)
    gm.install_no_tensor_aliasing_guard([m_t1, m_t2], ["t1", "t2"], ["no alias"], None)
    t1 = torch.randn(3)
    t2 = torch.randn(3)
    assert r_nta.check({"t1": t1, "t2": t2}) is True
    assert r_nta.check({"t1": t1, "t2": t1}) is False
    # Verify state reset
    assert r_nta.check({"t1": t1, "t2": t2}) is True

    # 4. install_storage_overlapping_guard and compute_overlapping_tensors
    base_t = torch.randn(10)
    sub1 = base_t[:5]
    sub2 = base_t[2:7]
    disjoint_t = base_t[7:]
    assert gm.compute_overlapping_tensors([sub1, sub2], False) == {0, 1}
    assert gm.compute_overlapping_tensors([sub1, disjoint_t], False) == set()

    r_so = gm.RootGuardManager()
    m_s1 = r_so.dict_getitem_manager("s1", "L['s1']", None, GM)
    m_s2 = r_so.dict_getitem_manager("s2", "L['s2']", None, GM)
    gm.install_storage_overlapping_guard([m_s1, m_s2], [], ["storage overlap"], None)
    assert r_so.check({"s1": sub1, "s2": sub2}) is True
    assert r_so.check({"s1": sub1, "s2": disjoint_t}) is False

    # 5. Epilogue lambda guards
    r_epi = gm.RootGuardManager()
    assert r_epi.get_epilogue_lambda_guards() == []
    r_epi.add_epilogue_lambda_guard(lambda env: env.get("flag") is True, ["flag must be True"], None)
    assert len(r_epi.get_epilogue_lambda_guards()) == 1
    assert r_epi.check({"flag": True}) is True
    assert r_epi.check({"flag": False}) is False

    # 6. clone_manager with filter predicates
    r_orig = gm.RootGuardManager()
    m_orig = r_orig.dict_getitem_manager("val", "L['val']", 7, GM)
    m_orig.add_equals_match_guard(7, [], None)
    cloned_all = r_orig.clone_manager(lambda m: True)
    assert cloned_all is not None and cloned_all is not r_orig
    assert cloned_all.check({"val": 7}) is True
    assert cloned_all.check({"val": 8}) is False
    cloned_none = r_orig.clone_manager(lambda m: False)
    assert cloned_none is None

    # 7. Free functions
    assert gm.check_obj_id(shared_obj, id(shared_obj)) is True
    assert gm.check_obj_id(shared_obj, id(object())) is False
    assert gm.check_type_id(10, id(int)) is True
    assert gm.check_type_id(10, id(float)) is False
    ver = gm.dict_version({"k": 1})
    assert isinstance(ver, tuple)
    assert ver == gm.dict_version({"k": 1})

    # 8. Keyword-argument binding from guards.pyi
    r_kw = gm.RootGuardManager()
    child_kw = r_kw.dict_getitem_manager(
        key="param",
        source="L['param']",
        example_value=123,
        guard_manager_enum=GM,
    )
    child_kw.add_equals_match_guard(
        equals_val=123,
        verbose_code_parts=["param must be 123"],
        user_stack=None,
    )
    assert r_kw.check({"param": 123}) is True
    assert r_kw.check({"param": 999}) is False


def run_type_dict_tests():
    # Issue #255: `type(self.config).__mro__[1].__dict__['attribute_map']`, the
    # chain Dynamo builds when it reads a class attribute through an instance
    # (`self.config.attribute_map` on a transformers PretrainedConfig). Dynamo
    # names the TypeDictSource `dict(X.__dict__)` because the C++
    # TypeDictGuardAccessor hands its child `type->tp_dict`, a real dict, not
    # the mappingproxy `X.__dict__` returns; the child is a
    # DictGetItemGuardAccessor, which is `PyDict_GetItem` and so requires a
    # dict. A guard built on a frame must pass on that same frame.
    GM = gm.GuardManagerType.GUARD_MANAGER

    class Base:
        attribute_map = {"hidden": "hidden_size"}

    class Config(Base):
        pass

    class Owner:
        def __init__(self):
            self.config = Config()

    owner = Owner()
    frame = {"self": owner}
    mapping = Base.__dict__["attribute_map"]

    root = gm.RootGuardManager()
    m_self = root.framelocals_manager(("self", 0), "L['self']", owner, GM)
    m_cfg = m_self.getattr_manager("config", "L['self'].config", owner.config, GM)
    m_type = m_cfg.type_manager("type(L['self'].config)", Config, GM)
    m_mro = m_type.type_mro_manager("type(L['self'].config).__mro__", Config.__mro__, GM)
    m_base = m_mro.tuple_getitem_manager(1, "type(L['self'].config).__mro__[1]", Base, GM)
    m_base.add_id_match_guard(id(Base), ["___check_obj_id(Base)"], None)
    src = "dict(type(L['self'].config).__mro__[1].__dict__)"
    m_td = m_base.type_dict_manager(src, dict(Base.__dict__), GM)
    # What the accessor hands its child is a dict, as tp_dict is in C++.
    m_td.add_type_match_guard(id(dict), ["type(tp_dict) is dict"], None)
    m_am = m_td.dict_getitem_manager("attribute_map", src + "['attribute_map']", mapping, GM)
    m_am.add_id_match_guard(id(mapping), ["___check_obj_id(attribute_map)"], None)

    assert root.check(frame) is True, "a type-dict guard fails on the frame it was built from"
    dbg = root.check_verbose(frame)
    assert dbg.result is True, dbg.verbose_code_parts

    # It still guards: rebinding the class attribute fails it, by the id guard
    # on the item, not by the accessor.
    Base.attribute_map = {"hidden": "other"}
    try:
        assert root.check(frame) is False, "rebinding the class attribute did not fail the guard"
        dbg = root.check_verbose(frame)
        assert dbg.result is False
        assert any("attribute_map" in p for p in dbg.verbose_code_parts), dbg.verbose_code_parts
        assert not any("getattr/getitem failed" in p for p in dbg.verbose_code_parts), dbg.verbose_code_parts
    finally:
        Base.attribute_map = mapping
    assert root.check(frame) is True

    # Removing the attribute fails the accessor itself (a KeyError in C++).
    del Base.attribute_map
    try:
        assert root.check(frame) is False
    finally:
        Base.attribute_map = mapping

    # A DictGuardManager over the type dict sees the same dict.
    root2 = gm.RootGuardManager()
    DGM = gm.GuardManagerType.DICT_GUARD_MANAGER
    m_td2 = root2.type_dict_manager("dict(Base.__dict__)", dict(Base.__dict__), DGM)
    assert type(m_td2).__name__ == "DictGuardManager", type(m_td2)
    assert root2.check(Base) is True, "a DictGuardManager on a type dict fails on its own type"

    # Only a type has a tp_dict: anything else fails the accessor.
    assert root2.check(Owner()) is False


if test_target == "leaf":
    run_leaf_tests()
elif test_target == "accessor":
    run_accessor_tests()
elif test_target == "dict_relational":
    run_dict_relational_clone_tests()
elif test_target == "type_dict":
    run_type_dict_tests()
else:
    raise ValueError(f"Unknown test target: {test_target}")

print(json.dumps({"status": "ok"}))
"""


def _run_unit_subprocess(target):
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
        raise AssertionError(
            f"Unit test probe exited with returncode {proc.returncode}:\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )

    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
    if not lines:
        raise AssertionError(
            f"Unit test probe produced no JSON output:\n{proc.stdout}\n{proc.stderr}"
        )
    data = json.loads(lines[-1])
    status = data.get("status")
    if status == "unimplemented":
        raise AssertionError(f"Shim dynamo guards not implemented: {data}")
    if status != "ok":
        raise AssertionError(f"Unit test failed: {data}")


def test_every_leaf_guard_kind_passes_and_fails_with_verbose_diagnostics():
    """Verify all 27 leaf guard kinds pass, fail, and report verbose diagnostics."""
    _run_unit_subprocess("leaf")


def test_every_accessor_kind_navigates_and_guards_child_values():
    """Verify all 27 accessor kinds navigate child values, deduplicate, and report diagnostics."""
    _run_unit_subprocess("accessor")


def test_dict_guard_manager_relational_guards_epilogue_and_clone():
    """Verify DictGuardManager, relational guards, epilogue lambdas, clone_manager, and pyi kwargs."""
    _run_unit_subprocess("dict_relational")


def test_a_type_dict_guard_passes_on_the_frame_it_was_built_from():
    """`dict(type(x).__mro__[1].__dict__)['k']` hands its child a real dict, as C++ tp_dict does (#255)."""
    _run_unit_subprocess("type_dict")


def _main():
    tests = [
        ("test_every_leaf_guard_kind_passes_and_fails_with_verbose_diagnostics", test_every_leaf_guard_kind_passes_and_fails_with_verbose_diagnostics),
        ("test_every_accessor_kind_navigates_and_guards_child_values", test_every_accessor_kind_navigates_and_guards_child_values),
        ("test_dict_guard_manager_relational_guards_epilogue_and_clone", test_dict_guard_manager_relational_guards_epilogue_and_clone),
        ("test_a_type_dict_guard_passes_on_the_frame_it_was_built_from", test_a_type_dict_guard_passes_on_the_frame_it_was_built_from),
    ]
    return _skip.run_tests(tests, suite="test_dynamo_guards_unit")


if __name__ == "__main__":
    sys.exit(_main())
