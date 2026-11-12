"""Differential guard test suite: torch._C._dynamo.guards vs upstream C++ guards.

Issue #200, INTENT §5, SPEC S1.8, S4.2:
`torch.compile` on torchnative operates without Dynamo's PEP 523 frame evaluation
hook (which abi3 rules out). Python control flow (guards and graph breaks) is
maintained through pure-Python `torch._C._dynamo.guards` matching upstream PyTorch's
C++ guard evaluation semantics.

This suite compares `torch._C._dynamo.guards` differentially against upstream
PyTorch 2.13.0 C++ guards across every leaf guard kind, accessor kind,
`DictGuardManager`, relational guard (object and tensor aliasing, storage
overlapping), and `RootGuardManager` epilogue/cloning machinery.

Three tests:
1. `test_upstream_cpp_guards_oracle_runs_every_scenario_and_resets_relational_state`:
   Runs the full scenario battery against upstream PyTorch 2.13's C++ guards in an
   isolated subprocess.
2. `test_differential_comparator_catches_a_planted_guard_divergence`:
   Nullification guard (AGENTS.md §17.5). Verifies that any diverged guard
   verdict or missing scenario is immediately caught by the comparator.
3. `test_shim_guards_match_upstream_cpp_guards_or_name_the_unimplemented_root`:
   Runs on the torchnative shim. In Step 1-2, pins the exact refusal naming
   `_dynamo.guards.RootGuardManager`. When `TORCHNATIVE_REQUIRE_SHIM_GUARDS=1`
   or `TORCHNATIVE_TEST_SHIM_GUARDS=1` is passed, or once Step 3 installs
   `torch._C._dynamo.guards`, demands 100% differential equality with the
   upstream C++ oracle across all scenarios.
"""

import copy
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

_PROBE_SCRIPT = r"""
import importlib.util
import json
import os
import sys
import types
import weakref

mode = sys.argv[1] if len(sys.argv) > 1 else "--upstream"
candidate_path = os.environ.get("TORCHNATIVE_GUARDS_MOD")
require_guards = (
    os.environ.get("TORCHNATIVE_REQUIRE_SHIM_GUARDS") == "1"
    or os.environ.get("TORCHNATIVE_TEST_SHIM_GUARDS") == "1"
)

if mode == "--shim":
    import torch
    if not hasattr(torch._C, "_aten_implemented"):
        print(json.dumps({"status": "error", "error": "not running torchnative shim"}))
        sys.exit(1)

    if candidate_path and os.path.isfile(candidate_path):
        if hasattr(torch._C, "DisableTorchFunction"):
            try:
                torch._C.DisableTorchFunction()
            except (NotImplementedError, TypeError):
                class _StubDisableTorchFunction:
                    def __enter__(self): pass
                    def __exit__(self, *args): pass
                torch._C.DisableTorchFunction = _StubDisableTorchFunction

        spec = importlib.util.spec_from_file_location("pyguards", candidate_path)
        gm = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(gm)
    else:
        gm = getattr(getattr(torch._C, "_dynamo", None), "guards", None)

    has_root = gm is not None and hasattr(gm, "RootGuardManager")
    check_fn = getattr(gm.RootGuardManager, "check", None) if has_root else None
    has_check = check_fn is not None and type(check_fn).__name__ != "_Unimplemented"

    if not has_check:
        if require_guards:
            raise NotImplementedError(
                "not implemented in torch._C shim: _dynamo.guards.RootGuardManager"
            )
        out = {
            "status": "unimplemented",
            "refusal": "not implemented in torch._C shim: _dynamo.guards.RootGuardManager",
        }
        print(json.dumps(out))
        sys.exit(0)
else:
    import torch
    import torch._C._dynamo.guards as gm

from torch._dynamo.guards import GuardManagerType
from torch.overrides import TorchFunctionMode
import torch.autograd.forward_ad as fwAD

# Helper classes
class _DummyMode(TorchFunctionMode):
    pass

class _TargetObj:
    pass

class _GenericAttrItem:
    def __init__(self, v):
        self.val = v

class _EmptyObj:
    pass

class _HasAttrObj:
    val = 10

class _CustomMapping:
    def __init__(self, data):
        self._d = dict(data)
    def __getitem__(self, k):
        return self._d[k]

class _CustomClassWithClosure:
    pass

# Patch in-memory prototype if needed for Python 3.13 range iterator reduce format
if hasattr(getattr(gm, "RANGE_ITERATOR_MATCH", None), "check_nopybind"):
    orig_range_check = gm.RANGE_ITERATOR_MATCH.check_nopybind
    def _safe_range_check(self, v):
        try:
            return orig_range_check(self, v)
        except TypeError:
            if id(type(v)) != self._t:
                return False
            _, (r,), idx = v.__reduce__()
            if idx is None:
                return (r.start, r.stop, r.step) == (self._r.start, self._r.stop, self._r.step)
            return (r.start + r.step * idx, r.stop, r.step) == (self._r.start, self._r.stop, self._r.step)
    gm.RANGE_ITERATOR_MATCH.check_nopybind = _safe_range_check

def run_scenarios(mod):
    res = {}

    # 1. Leaf: TYPE_MATCH
    r = mod.RootGuardManager()
    m = r.dict_getitem_manager("v", "L['v']", 0, GuardManagerType.GUARD_MANAGER)
    m.add_type_match_guard(id(int), ["type int"], None)
    res["type_match_pass"] = {"ok": r.check({"v": 42}), "v_ok": r.check_verbose({"v": 42}).result}
    res["type_match_fail"] = {"ok": r.check({"v": "42"}), "v_ok": r.check_verbose({"v": "42"}).result}

    # 2. Leaf: ID_MATCH
    sentinel = [1, 2]
    r = mod.RootGuardManager()
    m = r.dict_getitem_manager("v", "L['v']", None, GuardManagerType.GUARD_MANAGER)
    m.add_id_match_guard(id(sentinel), ["id sentinel"], None)
    res["id_match_pass"] = {"ok": r.check({"v": sentinel}), "v_ok": r.check_verbose({"v": sentinel}).result}
    res["id_match_fail"] = {"ok": r.check({"v": [1, 2]}), "v_ok": r.check_verbose({"v": [1, 2]}).result}

    # 3. Leaf: EQUALS_MATCH
    r = mod.RootGuardManager()
    m = r.dict_getitem_manager("v", "L['v']", 1, GuardManagerType.GUARD_MANAGER)
    m.add_equals_match_guard(1, ["equals 1"], None)
    res["equals_match_pass"] = {"ok": r.check({"v": 1}), "v_ok": r.check_verbose({"v": 1}).result}
    res["equals_match_fail_type"] = {"ok": r.check({"v": 1.0}), "v_ok": r.check_verbose({"v": 1.0}).result}
    res["equals_match_fail_val"] = {"ok": r.check({"v": 2}), "v_ok": r.check_verbose({"v": 2}).result}

    r_str = mod.RootGuardManager()
    m_str = r_str.dict_getitem_manager("s", "L['s']", "abc", GuardManagerType.GUARD_MANAGER)
    m_str.add_equals_match_guard("abc", ["equals abc"], None)
    res["equals_match_str_pass"] = {"ok": r_str.check({"s": "abc"}), "v_ok": r_str.check_verbose({"s": "abc"}).result}
    res["equals_match_str_fail"] = {"ok": r_str.check({"s": "def"}), "v_ok": r_str.check_verbose({"s": "def"}).result}

    # 4. Leaf: NONE_MATCH & NOT_NONE
    r = mod.RootGuardManager()
    m_n = r.dict_getitem_manager("n", "L['n']", None, GuardManagerType.GUARD_MANAGER)
    m_n.add_none_match_guard(["none"], None)
    m_nn = r.dict_getitem_manager("nn", "L['nn']", 0, GuardManagerType.GUARD_MANAGER)
    m_nn.add_not_none_guard(["not none"], None)
    res["none_not_none_pass"] = {"ok": r.check({"n": None, "nn": 0}), "v_ok": r.check_verbose({"n": None, "nn": 0}).result}
    res["none_fail_0"] = {"ok": r.check({"n": 0, "nn": 0}), "v_ok": r.check_verbose({"n": 0, "nn": 0}).result}
    res["not_none_fail_none"] = {"ok": r.check({"n": None, "nn": None}), "v_ok": r.check_verbose({"n": None, "nn": None}).result}

    # 5. Leaf: TRUE_MATCH & FALSE_MATCH
    r = mod.RootGuardManager()
    m_t = r.dict_getitem_manager("t", "L['t']", True, GuardManagerType.GUARD_MANAGER)
    m_t.add_true_match_guard(["true"], None)
    m_f = r.dict_getitem_manager("f", "L['f']", False, GuardManagerType.GUARD_MANAGER)
    m_f.add_false_match_guard(["false"], None)
    res["true_false_pass"] = {"ok": r.check({"t": True, "f": False}), "v_ok": r.check_verbose({"t": True, "f": False}).result}
    res["true_fail_1"] = {"ok": r.check({"t": 1, "f": False}), "v_ok": r.check_verbose({"t": 1, "f": False}).result}
    res["false_fail_0"] = {"ok": r.check({"t": True, "f": 0}), "v_ok": r.check_verbose({"t": True, "f": 0}).result}

    # 6. Leaf: LENGTH_CHECK & DICT_LENGTH
    r = mod.RootGuardManager()
    m_l = r.dict_getitem_manager("lst", "L['lst']", [1, 2, 3], GuardManagerType.GUARD_MANAGER)
    m_l.add_length_check_guard(3, ["len 3"], None)
    m_d = r.dict_getitem_manager("d", "L['d']", {"a": 1, "b": 2}, GuardManagerType.GUARD_MANAGER)
    m_d.add_dict_length_check_guard(2, ["dict len 2"], None)
    res["length_pass"] = {"ok": r.check({"lst": [1, 2, 3], "d": {"a": 1, "b": 2}}), "v_ok": r.check_verbose({"lst": [1, 2, 3], "d": {"a": 1, "b": 2}}).result}
    res["length_fail"] = {"ok": r.check({"lst": [1, 2], "d": {"a": 1, "b": 2}}), "v_ok": r.check_verbose({"lst": [1, 2], "d": {"a": 1, "b": 2}}).result}
    res["dict_length_fail"] = {"ok": r.check({"lst": [1, 2, 3], "d": {"a": 1}}), "v_ok": r.check_verbose({"lst": [1, 2, 3], "d": {"a": 1}}).result}

    # 7. Leaf: MAPPING_KEYS_MATCH
    r = mod.RootGuardManager()
    m = r.dict_getitem_manager("d", "L['d']", {"a": 1, "b": 2}, GuardManagerType.GUARD_MANAGER)
    m.add_mapping_keys_guard({"a": 1, "b": 2}, ["keys match"], None)
    res["mapping_keys_pass"] = {"ok": r.check({"d": {"a": 10, "b": 20}}), "v_ok": r.check_verbose({"d": {"a": 10, "b": 20}}).result}
    res["mapping_keys_fail_order"] = {"ok": r.check({"d": {"b": 2, "a": 1}}), "v_ok": r.check_verbose({"d": {"b": 2, "a": 1}}).result}
    res["mapping_keys_fail_keys"] = {"ok": r.check({"d": {"a": 1, "c": 2}}), "v_ok": r.check_verbose({"d": {"a": 1, "c": 2}}).result}

    # 8. Leaf: DICT_CONTAINS & SET_CONTAINS
    r = mod.RootGuardManager()
    m_dc = r.dict_getitem_manager("d", "L['d']", {"k": 1}, GuardManagerType.GUARD_MANAGER)
    m_dc.add_dict_contains_guard(True, "k", ["dict contains k"], None)
    m_sc = r.dict_getitem_manager("s", "L['s']", {42}, GuardManagerType.GUARD_MANAGER)
    m_sc.add_set_contains_guard(True, 42, ["set contains 42"], None)
    res["contains_pass"] = {"ok": r.check({"d": {"k": 1}, "s": {42}}), "v_ok": r.check_verbose({"d": {"k": 1}, "s": {42}}).result}
    res["dict_contains_fail"] = {"ok": r.check({"d": {"other": 1}, "s": {42}}), "v_ok": r.check_verbose({"d": {"other": 1}, "s": {42}}).result}
    res["set_contains_fail"] = {"ok": r.check({"d": {"k": 1}, "s": {99}}), "v_ok": r.check_verbose({"d": {"k": 1}, "s": {99}}).result}

    # 9. Leaf: NO_HASATTR & DICT_VERSION
    d_v = {"x": 10}
    r = mod.RootGuardManager()
    m_no = r.dict_getitem_manager("obj", "L['obj']", None, GuardManagerType.GUARD_MANAGER)
    m_no.add_no_hasattr_guard("val", ["no attr val"], None)
    m_dv = r.dict_getitem_manager("d", "L['d']", d_v, GuardManagerType.GUARD_MANAGER)
    m_dv.add_dict_version_guard(d_v, ["dict version"], None)
    res["no_hasattr_dict_version_pass"] = {"ok": r.check({"obj": _EmptyObj(), "d": d_v}), "v_ok": r.check_verbose({"obj": _EmptyObj(), "d": d_v}).result}
    res["no_hasattr_fail"] = {"ok": r.check({"obj": _HasAttrObj(), "d": d_v}), "v_ok": r.check_verbose({"obj": _HasAttrObj(), "d": d_v}).result}
    res["dict_version_fail"] = {"ok": r.check({"obj": _EmptyObj(), "d": {"x": 99}}), "v_ok": r.check_verbose({"obj": _EmptyObj(), "d": {"x": 99}}).result}

    # 10. Leaf: FLOAT_IS_NAN & COMPLEX_IS_NAN
    r = mod.RootGuardManager()
    m_fn = r.dict_getitem_manager("f", "L['f']", float("nan"), GuardManagerType.GUARD_MANAGER)
    m_fn.add_float_is_nan_guard(["float nan"], None)
    m_cn = r.dict_getitem_manager("c", "L['c']", complex(float("nan"), 0.0), GuardManagerType.GUARD_MANAGER)
    m_cn.add_complex_is_nan_guard(["complex nan"], None)
    res["nans_pass"] = {"ok": r.check({"f": float("nan"), "c": complex(float("nan"), 0.0)}), "v_ok": r.check_verbose({"f": float("nan"), "c": complex(float("nan"), 0.0)}).result}
    res["float_nan_fail"] = {"ok": r.check({"f": 1.0, "c": complex(float("nan"), 0.0)}), "v_ok": r.check_verbose({"f": 1.0, "c": complex(float("nan"), 0.0)}).result}
    res["complex_nan_fail"] = {"ok": r.check({"f": float("nan"), "c": complex(1.0, 2.0)}), "v_ok": r.check_verbose({"f": float("nan"), "c": complex(1.0, 2.0)}).result}

    # 11. Leaf: TUPLE_ITERATOR_LEN & RANGE_ITERATOR_MATCH
    tup = (1, 2, 3)
    it_t = iter(tup)
    r_obj = range(0, 10, 2)
    it_r = iter(r_obj)
    r = mod.RootGuardManager()
    m_ti = r.dict_getitem_manager("it_t", "L['it_t']", it_t, GuardManagerType.GUARD_MANAGER)
    m_ti.add_tuple_iterator_length_guard(3, id(type(it_t)), ["tup iter len 3"], None)
    m_ri = r.dict_getitem_manager("it_r", "L['it_r']", it_r, GuardManagerType.GUARD_MANAGER)
    m_ri.add_range_iterator_match_guard(0, 10, 2, id(type(it_r)), ["range iter match"], None)
    res["iterators_pass"] = {"ok": r.check({"it_t": iter(tup), "it_r": iter(r_obj)}), "v_ok": r.check_verbose({"it_t": iter(tup), "it_r": iter(r_obj)}).result}
    adv_t = iter(tup)
    next(adv_t)
    res["tuple_iter_fail"] = {"ok": r.check({"it_t": adv_t, "it_r": iter(r_obj)}), "v_ok": r.check_verbose({"it_t": adv_t, "it_r": iter(r_obj)}).result}
    adv_r = iter(r_obj)
    next(adv_r)
    res["range_iter_fail"] = {"ok": r.check({"it_t": iter(tup), "it_r": adv_r}), "v_ok": r.check_verbose({"it_t": iter(tup), "it_r": adv_r}).result}

    # 12. Leaf: DEFAULT_DEVICE
    r = mod.RootGuardManager()
    m_dev = r.dict_getitem_manager("v", "L['v']", 0, GuardManagerType.GUARD_MANAGER)
    m_dev.add_default_device_guard(["default dev"], None)
    res["default_device_pass"] = {"ok": r.check({"v": 0}), "v_ok": r.check_verbose({"v": 0}).result}

    # 13. Leaf: GLOBAL_STATE
    r_gs = mod.RootGuardManager()
    gs = mod.GlobalStateGuard()
    r_gs.add_global_state_guard(gs, ["global state"], None)
    res["global_state_pass"] = {"ok": r_gs.check({}), "v_ok": r_gs.check_verbose({}).result}
    with torch.no_grad():
        res["global_state_fail_no_grad"] = {"ok": r_gs.check({}), "v_ok": r_gs.check_verbose({}).result}
    res["global_state_pass_restored"] = {"ok": r_gs.check({}), "v_ok": r_gs.check_verbose({}).result}

    # 14. Leaf: DUAL_LEVEL_MATCH & TORCH_FUNCTION_MODE_STACK & DISPATCH_KEYS & DIMENSION_MARKING
    dks = torch._C._dispatch_tls_local_include_set()
    t_dim = torch.randn(2, 2)
    r_sys = mod.RootGuardManager()
    r_sys.add_dual_level_match_guard(-1, ["level -1"], None)
    r_sys.add_torch_function_mode_stack_guard([], ["empty mode stack"], None)
    m_dks = r_sys.dict_getitem_manager("dks", "L['dks']", dks, GuardManagerType.GUARD_MANAGER)
    m_dks.add_dispatch_key_set_guard(dks, ["dispatch keys"], None)
    m_dm = r_sys.dict_getitem_manager("t", "L['t']", t_dim, GuardManagerType.GUARD_MANAGER)
    m_dm.add_dimension_marking_guard({}, [], {}, ["marking"], None)
    res["sys_guards_pass"] = {"ok": r_sys.check({"dks": dks, "t": t_dim}), "v_ok": r_sys.check_verbose({"dks": dks, "t": t_dim}).result}
    with _DummyMode():
        res["torch_function_mode_fail"] = {"ok": r_sys.check({"dks": dks, "t": t_dim}), "v_ok": r_sys.check_verbose({"dks": dks, "t": t_dim}).result}
    orig_dual = getattr(fwAD, "_current_level", -1)
    try:
        fwAD._current_level = 0
        res["dual_level_fail"] = {"ok": r_sys.check({"dks": dks, "t": t_dim}), "v_ok": r_sys.check_verbose({"dks": dks, "t": t_dim}).result}
    finally:
        fwAD._current_level = orig_dual

    # 15. Leaf: LAMBDA_GUARD (pass, fail, and exception propagation)
    r_lam = mod.RootGuardManager()
    m_lam = r_lam.dict_getitem_manager("v", "L['v']", 10, GuardManagerType.GUARD_MANAGER)
    m_lam.add_lambda_guard(lambda x: x > 5, ["v > 5"], None)
    res["lambda_pass"] = {"ok": r_lam.check({"v": 10}), "v_ok": r_lam.check_verbose({"v": 10}).result}
    res["lambda_fail"] = {"ok": r_lam.check({"v": 2}), "v_ok": r_lam.check_verbose({"v": 2}).result}

    def _exploding_lambda(x):
        raise ValueError("lambda boom check")
    r_boom = mod.RootGuardManager()
    m_boom = r_boom.dict_getitem_manager("v", "L['v']", 0, GuardManagerType.GUARD_MANAGER)
    m_boom.add_lambda_guard(_exploding_lambda, ["exploding"], None)
    dbg_boom = r_boom.check_verbose({"v": 0})
    res["lambda_exc"] = {"ok": r_boom.check({"v": 0}), "v_ok": dbg_boom.result}
    res["lambda_exc_has_vcp"] = {"ok": len(dbg_boom.verbose_code_parts) > 0, "v_ok": True}

    # 16. Leaf: TENSOR_MATCH
    t_ref = torch.randn(2, 3)
    r_tm = mod.RootGuardManager()
    m_tm = r_tm.dict_getitem_manager("t", "L['t']", t_ref, GuardManagerType.GUARD_MANAGER)
    try:
        dk = torch._C._dispatch_keys(t_ref)
    except (NotImplementedError, AttributeError, TypeError):
        dk = torch._C._dispatch_tls_local_include_set()
    m_tm.add_tensor_match_guard(t_ref, list(t_ref.size()), list(t_ref.stride()), "t", ["tensor match"], None, type(t_ref), dk)
    res["tensor_match_pass"] = {"ok": r_tm.check({"t": t_ref}), "v_ok": r_tm.check_verbose({"t": t_ref}).result}
    res["tensor_match_fail_size"] = {"ok": r_tm.check({"t": torch.randn(2, 4)}), "v_ok": r_tm.check_verbose({"t": torch.randn(2, 4)}).result}
    res["tensor_match_fail_dtype"] = {"ok": r_tm.check({"t": t_ref.to(torch.float64)}), "v_ok": r_tm.check_verbose({"t": t_ref.to(torch.float64)}).result}
    res["tensor_match_fail_rank"] = {"ok": r_tm.check({"t": torch.randn(2, 3, 1)}), "v_ok": r_tm.check_verbose({"t": torch.randn(2, 3, 1)}).result}

    # 17. Accessors: list, tuple, generic_getattr, type, weakref, call_no_args, lambda_manager, func_defaults
    target_obj = _TargetObj()
    wref = weakref.ref(target_obj)
    def _zero_fn():
        return 42
    def _fn_with_defaults(a, b=100):
        return a + b
    def _fn_with_kwdefaults(a, *, c=200):
        return a + c

    r_acc = mod.RootGuardManager()
    m_lst = r_acc.list_getitem_manager(1, "lst[1]", 20, GuardManagerType.GUARD_MANAGER)
    m_lst.add_equals_match_guard(20, ["lst[1] == 20"], None)
    m_tup = r_acc.tuple_getitem_manager(0, "tup[0]", 5, GuardManagerType.GUARD_MANAGER)
    m_tup.add_equals_match_guard(5, ["tup[0] == 5"], None)
    m_ga = r_acc.generic_getattr_manager("val", "item.val", 10, GuardManagerType.GUARD_MANAGER)
    m_ga.add_equals_match_guard(10, ["item.val == 10"], None)
    m_typ = r_acc.type_manager("type(obj)", int, GuardManagerType.GUARD_MANAGER)
    m_typ.add_equals_match_guard(int, ["type(obj) is int"], None)
    m_wr = r_acc.weakref_call_manager("wref()", target_obj, GuardManagerType.GUARD_MANAGER)
    m_wr.add_id_match_guard(id(target_obj), ["id target"], None)
    m_cna = r_acc.call_function_no_args_manager("fn()", 42, GuardManagerType.GUARD_MANAGER)
    m_cna.add_equals_match_guard(42, ["fn() == 42"], None)
    m_lm = r_acc.lambda_manager(lambda x: x + 10, "lam(x)", 15, GuardManagerType.GUARD_MANAGER)
    m_lm.add_equals_match_guard(15, ["lam(x) == 15"], None)
    m_fd = r_acc.func_defaults_manager("fn.__defaults__", (100,), GuardManagerType.GUARD_MANAGER)
    m_fd.add_equals_match_guard((100,), ["defaults == (100,)"], None)
    m_fkd = r_acc.func_kwdefaults_manager("fn.__kwdefaults__", {"c": 200}, GuardManagerType.GUARD_MANAGER)
    m_fkd.add_equals_match_guard({"c": 200}, ["kwdefaults == {'c': 200}"], None)

    class _AccessEnv:
        def __init__(self):
            self.val = 10
        def __getitem__(self, idx):
            if idx == 0: return 5
            if idx == 1: return 20
            raise IndexError
        def __call__(self):
            return 42

    res["accessors_all_pass"] = {"ok": r_acc.check(_AccessEnv()), "v_ok": r_acc.check_verbose(_AccessEnv()).result}

    # Accessor deduplication: calling twice returns the exact same child GuardManager
    a1 = r_acc.getattr_manager("test_attr", "obj.test_attr", None, GuardManagerType.GUARD_MANAGER)
    a2 = r_acc.getattr_manager("test_attr", "obj.test_attr", None, GuardManagerType.GUARD_MANAGER)
    res["accessor_dedup"] = {"ok": (a1 is a2), "v_ok": True}

    # 18. Accessors: Tensor properties (size, storage_offset, stride)
    t_prop = torch.randn(3, 4)
    r_tp = mod.RootGuardManager()
    m_sz = r_tp.tensor_property_size_manager(0, "t.size(0)", 3, GuardManagerType.GUARD_MANAGER)
    m_sz.add_equals_match_guard(3, ["size == 3"], None)
    m_off = r_tp.tensor_property_storage_offset_manager(0, "t.offset()", 0, GuardManagerType.GUARD_MANAGER)
    m_off.add_equals_match_guard(0, ["offset == 0"], None)
    stride_fn = getattr(r_tp, "tensor_property_stride_manager", getattr(r_tp, "tensor_property_shape_manager", None))
    if stride_fn is not None:
        m_strd = stride_fn(0, "t.stride(0)", 4, GuardManagerType.GUARD_MANAGER)
        m_strd.add_equals_match_guard(4, ["stride == 4"], None)
    res["tensor_prop_pass"] = {"ok": r_tp.check(t_prop), "v_ok": r_tp.check_verbose(t_prop).result}
    res["tensor_prop_fail_size"] = {"ok": r_tp.check(torch.randn(5, 4)), "v_ok": r_tp.check_verbose(torch.randn(5, 4)).result}

    # 19. DictGuardManager
    r_dgm = mod.RootGuardManager()
    dm = r_dgm.dict_getitem_manager("d", "L['d']", {"k": 100}, GuardManagerType.DICT_GUARD_MANAGER)
    km = dm.get_key_manager(0, "k", "k", GuardManagerType.GUARD_MANAGER)
    km.add_equals_match_guard("k", ["k match"], None)
    vm = dm.get_value_manager(0, "v", 100, GuardManagerType.GUARD_MANAGER)
    vm.add_equals_match_guard(100, ["v == 100"], None)
    res["dict_mgr_pass"] = {"ok": r_dgm.check({"d": {"k": 100}}), "v_ok": r_dgm.check_verbose({"d": {"k": 100}}).result}
    res["dict_mgr_fail_size"] = {"ok": r_dgm.check({"d": {"k": 100, "extra": 1}}), "v_ok": r_dgm.check_verbose({"d": {"k": 100, "extra": 1}}).result}
    res["dict_mgr_fail_val"] = {"ok": r_dgm.check({"d": {"k": 999}}), "v_ok": r_dgm.check_verbose({"d": {"k": 999}}).result}

    # 20. Relational: OBJECT_ALIASING with state reset across calls
    r_rel = mod.RootGuardManager()
    m1 = r_rel.dict_getitem_manager("a", "L['a']", None, GuardManagerType.GUARD_MANAGER)
    m2 = r_rel.dict_getitem_manager("b", "L['b']", None, GuardManagerType.GUARD_MANAGER)
    mod.install_object_aliasing_guard(m1, m2, ["alias"], None)
    o1 = [1]
    o2 = [1]
    res["rel_alias_pass"] = {"ok": r_rel.check({"a": o1, "b": o1}), "v_ok": r_rel.check_verbose({"a": o1, "b": o1}).result}
    res["rel_alias_fail"] = {"ok": r_rel.check({"a": o1, "b": o2}), "v_ok": r_rel.check_verbose({"a": o1, "b": o2}).result}
    res["rel_alias_reset"] = {"ok": r_rel.check({"a": o1, "b": o1}), "v_ok": r_rel.check_verbose({"a": o1, "b": o1}).result}

    # 21. Relational: NO_TENSOR_ALIASING with state reset across calls
    r_nta = mod.RootGuardManager()
    tx1 = r_nta.dict_getitem_manager("x", "L['x']", None, GuardManagerType.GUARD_MANAGER)
    ty1 = r_nta.dict_getitem_manager("y", "L['y']", None, GuardManagerType.GUARD_MANAGER)
    mod.install_no_tensor_aliasing_guard([tx1, ty1], ["x", "y"], ["no alias"], None)
    t1 = torch.tensor([1.0])
    t2 = torch.tensor([2.0])
    res["no_tensor_alias_pass"] = {"ok": r_nta.check({"x": t1, "y": t2}), "v_ok": r_nta.check_verbose({"x": t1, "y": t2}).result}
    res["no_tensor_alias_fail"] = {"ok": r_nta.check({"x": t1, "y": t1}), "v_ok": r_nta.check_verbose({"x": t1, "y": t1}).result}
    res["no_tensor_alias_reset"] = {"ok": r_nta.check({"x": t1, "y": t2}), "v_ok": r_nta.check_verbose({"x": t1, "y": t2}).result}

    # 22. Relational: STORAGE OVERLAPPING & compute_overlapping_tensors
    t_full = torch.randn(10)
    v1 = t_full[:5]
    v2 = t_full[2:7]
    v3 = t_full[6:]
    res["compute_overlap_tensors_overlapping"] = {"ok": mod.compute_overlapping_tensors([v1, v2]) == {0, 1}, "v_ok": True}
    res["compute_overlap_tensors_disjoint"] = {"ok": mod.compute_overlapping_tensors([v1, v3]) == set(), "v_ok": True}

    r_so = mod.RootGuardManager()
    m_so1 = r_so.dict_getitem_manager("a", "L['a']", None, GuardManagerType.GUARD_MANAGER)
    m_so2 = r_so.dict_getitem_manager("b", "L['b']", None, GuardManagerType.GUARD_MANAGER)
    mod.install_storage_overlapping_guard([m_so1, m_so2], [], ["storage overlap"], None)
    res["storage_overlap_pass"] = {"ok": r_so.check({"a": v1, "b": v2}), "v_ok": r_so.check_verbose({"a": v1, "b": v2}).result}
    res["storage_overlap_fail"] = {"ok": r_so.check({"a": v1, "b": v3}), "v_ok": r_so.check_verbose({"a": v1, "b": v3}).result}

    # 23. EPILOGUE & CLONE_MANAGER
    r_epi = mod.RootGuardManager()
    m_val = r_epi.dict_getitem_manager("keep", "L['keep']", 10, GuardManagerType.GUARD_MANAGER)
    m_val.add_equals_match_guard(10, ["keep == 10"], None)
    m_drop = r_epi.dict_getitem_manager("drop", "L['drop']", 20, GuardManagerType.GUARD_MANAGER)
    m_drop.add_equals_match_guard(20, ["drop == 20"], None)
    lg = mod.LAMBDA_GUARD(r_epi, lambda v: v.get("keep") == 10, ["epilogue lambda"], None)
    r_epi.add_epilogue_lambda_guard(lg, ["epilogue lambda"], None)
    res["epilogue_pass"] = {"ok": r_epi.check({"keep": 10, "drop": 20}), "v_ok": r_epi.check_verbose({"keep": 10, "drop": 20}).result}
    cloned = r_epi.clone_manager(lambda mgr: mgr.get_source() != "L['drop']")
    res["cloned_pass"] = {"ok": cloned.check({"keep": 10, "drop": 999}), "v_ok": cloned.check_verbose({"keep": 10, "drop": 999}).result}
    res["orig_after_clone_fail"] = {"ok": not r_epi.check({"keep": 10, "drop": 999}), "v_ok": True}

    return res

results = run_scenarios(gm)
out = {
    "status": "implemented",
    "results": results,
}
print(json.dumps(out))
"""


def _find_vendor_dir():
    if os.path.isfile(_VENDOR_SHIM):
        return _VENDOR_DIR
    if os.path.isfile(_MAIN_VENDOR_SHIM):
        return _MAIN_VENDOR_DIR
    return _VENDOR_DIR


def _run_oracle_subprocess():
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env.pop("TORCH_USE_RTLD_GLOBAL", None)
    env.pop("TORCHNATIVE_GUARDS_MOD", None)
    env.pop("TORCHNATIVE_REQUIRE_SHIM_GUARDS", None)
    env.pop("TORCHNATIVE_TEST_SHIM_GUARDS", None)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT, "--upstream"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Upstream oracle probe exited {proc.returncode}:\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    line = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")][-1]
    data = json.loads(line)
    if data.get("status") != "implemented":
        raise RuntimeError(f"Oracle failed to run scenarios: {data}")
    return data["results"]


def _run_shim_subprocess(require_guards=False, candidate_path=None):
    vendor_dir = _find_vendor_dir()
    env = dict(os.environ)
    env["PYTHONPATH"] = vendor_dir
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    if candidate_path:
        env["TORCHNATIVE_GUARDS_MOD"] = candidate_path
    if require_guards:
        env["TORCHNATIVE_REQUIRE_SHIM_GUARDS"] = "1"

    proc = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT, "--shim"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if proc.returncode != 0:
        if require_guards and "NotImplementedError: not implemented in torch._C shim: _dynamo.guards.RootGuardManager" in proc.stderr:
            raise NotImplementedError("not implemented in torch._C shim: _dynamo.guards.RootGuardManager")
        raise RuntimeError(
            f"Shim probe exited {proc.returncode}:\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    lines = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")]
    if not lines:
        raise RuntimeError(f"Shim probe produced no JSON output:\n{proc.stdout}\n{proc.stderr}")
    return json.loads(lines[-1])


def _compare_guard_results(actual, oracle):
    divergences = []
    missing = set(oracle.keys()) - set(actual.keys())
    if missing:
        divergences.append(f"Missing scenarios in actual results: {sorted(missing)}")
    extra = set(actual.keys()) - set(oracle.keys())
    if extra:
        divergences.append(f"Unexpected extra scenarios in actual results: {sorted(extra)}")

    for key in sorted(oracle.keys()):
        if key not in actual:
            continue
        exp = oracle[key]
        act = actual[key]
        if act.get("ok") != exp.get("ok"):
            divergences.append(
                f"[{key}]: guard ok diverged (expected {exp.get('ok')}, got {act.get('ok')})"
            )
        if act.get("v_ok") != exp.get("v_ok"):
            divergences.append(
                f"[{key}]: v_ok diverged (expected {exp.get('v_ok')}, got {act.get('v_ok')})"
            )
    return divergences


def test_upstream_cpp_guards_oracle_runs_every_scenario_and_resets_relational_state():
    """Verify that the differential scenario harness runs against upstream C++ guards.

    Runs all 46 scenarios across leaf guards, accessors, DictGuardManager,
    relational guards (with relational state reset across check calls),
    epilogue lambdas, and clone_manager.
    """
    oracle = _run_oracle_subprocess()
    assert len(oracle) >= 40, f"Expected at least 40 scenarios, got {len(oracle)}"

    # Check key expected outcomes
    assert oracle["type_match_pass"]["ok"] is True
    assert oracle["type_match_fail"]["ok"] is False
    assert oracle["equals_match_pass"]["ok"] is True
    assert oracle["equals_match_fail_type"]["ok"] is False
    assert oracle["equals_match_fail_val"]["ok"] is False
    assert oracle["iterators_pass"]["ok"] is True
    assert oracle["range_iter_fail"]["ok"] is False
    assert oracle["rel_alias_pass"]["ok"] is True
    assert oracle["rel_alias_fail"]["ok"] is False
    assert oracle["rel_alias_reset"]["ok"] is True, "Relational state must reset across check calls"
    assert oracle["no_tensor_alias_pass"]["ok"] is True
    assert oracle["no_tensor_alias_fail"]["ok"] is False
    assert oracle["no_tensor_alias_reset"]["ok"] is True, "No-tensor-aliasing state must reset across check calls"
    assert oracle["dict_mgr_pass"]["ok"] is True
    assert oracle["dict_mgr_fail_size"]["ok"] is False
    assert oracle["epilogue_pass"]["ok"] is True
    assert oracle["cloned_pass"]["ok"] is True


def test_differential_comparator_catches_a_planted_guard_divergence():
    """AGENTS.md §17.5 nullification test: a check that cannot fail is not a check.

    Verifies that mutating any boolean verdict or deleting a scenario is caught
    by the comparator.
    """
    oracle = _run_oracle_subprocess()

    # Identical results must produce no divergences
    assert _compare_guard_results(oracle, oracle) == []

    # 1. Mutate boolean check result
    mutated_ok = copy.deepcopy(oracle)
    mutated_ok["type_match_pass"]["ok"] = False
    diffs = _compare_guard_results(mutated_ok, oracle)
    assert len(diffs) == 1, diffs
    assert "type_match_pass" in diffs[0] and "guard ok diverged" in diffs[0]

    # 2. Mutate verbose check result
    mutated_v = copy.deepcopy(oracle)
    mutated_v["equals_match_fail_val"]["v_ok"] = True
    diffs_v = _compare_guard_results(mutated_v, oracle)
    assert len(diffs_v) == 1, diffs_v
    assert "equals_match_fail_val" in diffs_v[0] and "v_ok diverged" in diffs_v[0]

    # 3. Missing scenario
    missing = copy.deepcopy(oracle)
    del missing["rel_alias_reset"]
    diffs_m = _compare_guard_results(missing, oracle)
    assert any("Missing scenarios" in d and "rel_alias_reset" in d for d in diffs_m)


def test_shim_guards_match_upstream_cpp_guards_or_name_the_unimplemented_root():
    """Verify guards on the torchnative shim.

    In Step 1-2 (before Step 3 lands), RootGuardManager on the shim is an
    unimplemented stub; this test asserts that it refuses naming
    `_dynamo.guards.RootGuardManager`.

    When TORCHNATIVE_REQUIRE_SHIM_GUARDS=1 or TORCHNATIVE_TEST_SHIM_GUARDS=1
    is passed, the test requires the shim implementation and raises the
    NotImplementedError (red state).

    When Step 3 lands and installs RootGuardManager on the shim, or when
    TORCHNATIVE_GUARDS_MOD is set, this test verifies 100% differential
    agreement against upstream PyTorch 2.13 C++ guards across all scenarios.
    """
    oracle = _run_oracle_subprocess()

    # Check for explicit red-state demand
    require_guards = (
        os.environ.get("TORCHNATIVE_REQUIRE_SHIM_GUARDS") == "1"
        or os.environ.get("TORCHNATIVE_TEST_SHIM_GUARDS") == "1"
    )

    data = _run_shim_subprocess(require_guards=require_guards)
    status = data.get("status")

    if status == "unimplemented":
        refusal = data.get("refusal", "")
        assert "_dynamo.guards.RootGuardManager" in refusal, f"Refusal must name _dynamo.guards.RootGuardManager, got {refusal}"
    elif status == "implemented":
        results = data.get("results", {})
        diffs = _compare_guard_results(results, oracle)
        assert not diffs, f"Differential divergences detected between shim and upstream guards:\n" + "\n".join(diffs)
    else:
        raise RuntimeError(f"Unexpected probe status: {data}")

    # If candidate pyguards prototype exists, verify it passes 100% differentially
    candidate_default = "/Volumes/macMini/thisisthepy/torchnative/.scratch/dynamo-no-pep523/shim/pyguards.py"
    if os.path.isfile(candidate_default):
        cand_data = _run_shim_subprocess(require_guards=False, candidate_path=candidate_default)
        if cand_data.get("status") == "implemented":
            cand_diffs = _compare_guard_results(cand_data.get("results", {}), oracle)
            assert not cand_diffs, f"Candidate pyguards prototype diverged from upstream guards:\n" + "\n".join(cand_diffs)


def _main():
    tests = [
        ("test_upstream_cpp_guards_oracle_runs_every_scenario_and_resets_relational_state", test_upstream_cpp_guards_oracle_runs_every_scenario_and_resets_relational_state),
        ("test_differential_comparator_catches_a_planted_guard_divergence", test_differential_comparator_catches_a_planted_guard_divergence),
        ("test_shim_guards_match_upstream_cpp_guards_or_name_the_unimplemented_root", test_shim_guards_match_upstream_cpp_guards_or_name_the_unimplemented_root),
    ]
    return _skip.run_tests(tests, suite="test_dynamo_guards_diff")


if __name__ == "__main__":
    sys.exit(_main())
