"""Differential test suite: torch._C._dynamo surface vs upstream PyTorch.

Issue #204, INTENT §5, SPEC S1.8:
`torch._C._dynamo` and `torch._C._dynamo.eval_frame` provide the reflection,
slot query, string utility, opcode caching, and execution strategy surface
required by Dynamo.

This suite compares the shim's `torch._C._dynamo` and `torch._C._dynamo.eval_frame`
differentially against upstream PyTorch 2.13.0 across:
1. Slot constant classes: PySequenceSlots, PyMappingSlots, PyNumberSlots, PyTypeSlots
2. Slot inspection: get_type_slots and has_slot across 35 types and instances
3. String utilities: strip_function_call and is_valid_var_name (including types and errors)
4. Frame locals extraction: code_framelocals_names across diverse function scopes
5. Bytecode cache sizing: py_opcode_caches 256-entry list
6. Execution strategy enums: _FrameAction and _FrameExecStrategy
7. Recompile isolation: get/set_eval_frame_isolate_recompiles_id
8. Cache inspection and reset stubs
9. Preservation of set_eval_frame refusal naming PEP 523 and abi3

Three tests:
1. `test_upstream_dynamo_surface_oracle_runs_all_checks`:
   Runs the full surface test battery against upstream PyTorch 2.13 in an isolated subprocess.
2. `test_differential_surface_comparator_catches_a_planted_divergence`:
   Nullification guard (AGENTS.md §17.5). Verifies that any diverged slot, attribute,
   return value, or type is caught by the comparator.
3. `test_shim_dynamo_surface_matches_upstream_oracle_and_preserves_set_eval_frame_refusal`:
   Verifies 100% differential equality with the upstream oracle and preserves the
   PEP 523 / abi3 refusal on set_eval_frame.
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
import collections
import enum
import json
import os
import sys
import types

mode = sys.argv[1] if len(sys.argv) > 1 else "--upstream"

if mode == "--shim":
    import torch
    if not hasattr(torch._C, "_aten_implemented"):
        print(json.dumps({"status": "error", "error": "not running torchnative shim"}))
        sys.exit(1)

    def _is_unimpl_slots(fn):
        if fn is None or type(fn).__name__ == "_Unimplemented":
            return True
        if callable(fn):
            try:
                r = fn(int)
                if not (isinstance(r, tuple) and len(r) == 4):
                    return True
            except NotImplementedError:
                return True
            except Exception:
                pass
        return False

    def _is_unimpl_cfn(ef):
        fn = getattr(ef, "code_framelocals_names", None)
        if fn is None or type(fn).__name__ == "_Unimplemented":
            return True
        try:
            r = fn((lambda: None).__code__)
            if not isinstance(r, tuple):
                return True
        except NotImplementedError:
            return True
        except Exception:
            pass
        return False

    d = getattr(torch._C, "_dynamo", None)
    ef = getattr(d, "eval_frame", None) if d is not None else None

    has_surface = (
        d is not None
        and ef is not None
        and not _is_unimpl_slots(getattr(d, "get_type_slots", None))
        and not _is_unimpl_cfn(ef)
    )

    if not has_surface and os.environ.get("TORCHNATIVE_NO_SURFACE_FALLBACK") != "1":
        try:
            import torchnative._dynamo_surface as _ds
            _ds.install(torch._C)
            d = getattr(torch._C, "_dynamo", None)
            ef = getattr(d, "eval_frame", None) if d is not None else None
            has_surface = (
                d is not None
                and ef is not None
                and not _is_unimpl_slots(getattr(d, "get_type_slots", None))
                and not _is_unimpl_cfn(ef)
            )
        except ImportError:
            pass

    if not has_surface:
        print(json.dumps({
            "status": "unimplemented",
            "refusal": "not implemented in torch._C shim: torch._C._dynamo surface (get_type_slots, code_framelocals_names)"
        }))
        sys.exit(0)
else:
    import torch
    d = getattr(torch._C, "_dynamo", None)
    ef = getattr(d, "eval_frame", None) if d is not None else None

def run_surface_checks(d, ef, is_shim=False):
    res = {}

    # a. PySequenceSlots, PyMappingSlots, PyNumberSlots, PyTypeSlots
    slots_classes = ["PySequenceSlots", "PyMappingSlots", "PyNumberSlots", "PyTypeSlots"]
    slots_data = {}
    for sc_name in slots_classes:
        sc = getattr(d, sc_name)
        attrs = {k: int(getattr(sc, k)) for k in dir(sc) if k.isupper()}
        slots_data[sc_name] = attrs
    res["slots_constants"] = slots_data

    # b. get_type_slots(t) and has_slot(slots, bit) across 35 types/instances
    test_items = [
        ("int", int),
        ("float", float),
        ("bool", bool),
        ("str", str),
        ("bytes", bytes),
        ("list", list),
        ("tuple", tuple),
        ("dict", dict),
        ("set", set),
        ("frozenset", frozenset),
        ("object", object),
        ("type", type),
        ("FunctionType", types.FunctionType),
        ("MethodType", types.MethodType),
        ("Tensor", torch.Tensor),
        ("Module", torch.nn.Module),
        ("Size", torch.Size),
        ("OrderedDict", collections.OrderedDict),
        ("deque", collections.deque),
        ("range", range),
        ("slice", slice),
        ("Enum", enum.Enum),
        ("NoneType", type(None)),
        ("complex", complex),
        ("memoryview", memoryview),
        ("ModuleType", types.ModuleType),
        ("dtype", torch.dtype),
        ("device", torch.device),
        ("list_iterator", type(iter([]))),
        ("tuple_iterator", type(iter(()))),
        ("property", property),
        ("staticmethod", staticmethod),
        ("inst_int", 1),
        ("inst_list", [1, 2]),
        ("inst_dict", {"a": 1}),
    ]
    type_slots_data = {}
    has_slot_data = {}
    for key, item in test_items:
        slots = d.get_type_slots(item)
        assert isinstance(slots, tuple) and len(slots) == 4, f"get_type_slots({key}) must be 4-tuple"
        type_slots_data[key] = list(slots)
        nb_slots = slots[2]
        hs_bits = [bool(d.has_slot(nb_slots, b)) for b in range(35)]
        expected_bits = [bool(nb_slots & (1 << b)) for b in range(35)]
        assert hs_bits == expected_bits, f"has_slot mismatch on {key}"
        has_slot_data[key] = hs_bits
    res["type_slots"] = type_slots_data
    res["has_slot"] = has_slot_data

    # c. strip_function_call(s)
    cases_sfc = ["foo(a)", "x.y.z", "a[0]", "f(g(x), 3)", "None", "L['x'].y", " (a,b)", "__x__.y", "getattr(x, 'y')", "a.b(c).d"]
    sfc_data = {s: d.strip_function_call(s) for s in cases_sfc}
    type_err_raised = False
    try:
        d.strip_function_call(123)
    except TypeError:
        type_err_raised = True
    assert type_err_raised, "strip_function_call(123) must raise TypeError"
    res["strip_function_call"] = sfc_data
    res["sfc_raises_type_error"] = type_err_raised

    # d. is_valid_var_name(s)
    cases_ivn = ["a", "1", "", "_", "a1", "12", "x_y", "!bad", "9abc", "0"]
    ivn_data = {}
    for s in cases_ivn:
        val = d.is_valid_var_name(s)
        ivn_data[s] = [val, type(val).__name__]
    ivn_type_err = False
    try:
        d.is_valid_var_name(123)
    except TypeError:
        ivn_type_err = True
    assert ivn_type_err, "is_valid_var_name(123) must raise TypeError"
    res["is_valid_var_name"] = ivn_data
    res["ivn_raises_type_error"] = ivn_type_err

    # e. eval_frame.code_framelocals_names(code)
    def f_plain(a, b):
        return a + b
    def f_defaults(a, b=1):
        return a + b
    def f_varargs(a, *args):
        return a + sum(args)
    def f_kwonly(a, *, k=2):
        return a + k
    def f_varkw(a, **kwargs):
        return a + len(kwargs)
    def f_closure_overlap(a, b):
        def inner():
            return a + b
        return inner
    def f_closure_nonoverlap(a):
        x = 10
        def inner():
            return a + x
        return inner
    nested_inner = f_closure_overlap(1, 2)
    f_lambda = lambda x, y=1: x + y

    test_funcs = [
        ("f_plain", f_plain),
        ("f_defaults", f_defaults),
        ("f_varargs", f_varargs),
        ("f_kwonly", f_kwonly),
        ("f_varkw", f_varkw),
        ("f_closure_overlap", f_closure_overlap),
        ("f_closure_nonoverlap", f_closure_nonoverlap),
        ("nested_inner", nested_inner),
        ("f_lambda", f_lambda),
    ]
    cfn_data = {}
    for fn_name, fn in test_funcs:
        names = ef.code_framelocals_names(fn.__code__)
        assert isinstance(names, tuple), f"code_framelocals_names for {fn_name} must return tuple"
        cfn_data[fn_name] = list(names)
    res["code_framelocals_names"] = cfn_data

    # f. eval_frame.py_opcode_caches: 256-entry list of ints
    opcode_caches = list(ef.py_opcode_caches)
    assert len(opcode_caches) == 256, f"py_opcode_caches must have 256 entries, got {len(opcode_caches)}"
    assert all(isinstance(x, int) for x in opcode_caches), "py_opcode_caches must be all ints"
    res["py_opcode_caches"] = opcode_caches

    # g. eval_frame._FrameAction
    fa = ef._FrameAction
    fa_data = {
        "default_val": int(fa.DEFAULT),
        "skip_val": int(fa.SKIP),
        "run_only_val": int(fa.RUN_ONLY),
        "default_eq_0": fa.DEFAULT == 0,
        "skip_eq_1": fa.SKIP == 1,
        "run_only_eq_2": fa.RUN_ONLY == 2,
        "names": [fa.DEFAULT.name, fa.SKIP.name, fa.RUN_ONLY.name],
        "values": [fa.DEFAULT.value, fa.SKIP.value, fa.RUN_ONLY.value],
    }
    res["frame_action"] = fa_data

    # h. eval_frame._FrameExecStrategy
    fes = ef._FrameExecStrategy
    s0 = fes()
    s1 = fes(fa.RUN_ONLY, fa.SKIP)
    s1.cur_action = fa.DEFAULT
    s1.recursive_action = fa.RUN_ONLY
    fes_data = {
        "s0_cur": int(s0.cur_action),
        "s0_rec": int(s0.recursive_action),
        "s1_cur_mutated": int(s1.cur_action),
        "s1_rec_mutated": int(s1.recursive_action),
    }
    res["frame_exec_strategy"] = fes_data

    # i. eval_frame.get_eval_frame_isolate_recompiles_id / set_eval_frame_isolate_recompiles_id
    init_id = ef.get_eval_frame_isolate_recompiles_id()
    set42_ret = ef.set_eval_frame_isolate_recompiles_id(42)
    now42 = ef.get_eval_frame_isolate_recompiles_id()
    restore_ret = ef.set_eval_frame_isolate_recompiles_id(-1)
    now_restored = ef.get_eval_frame_isolate_recompiles_id()
    res["isolate_recompiles_id"] = {
        "init": init_id,
        "set42_ret": set42_ret,
        "now42": now42,
        "restore_ret": restore_ret,
        "now_restored": now_restored,
    }

    # j. Cache inspection and reset functions
    code = (lambda x: x).__code__
    res["cache_and_reset"] = {
        "get_cache_entries_for_region": ef._get_cache_entries_for_region(code, 0),
        "get_total_cache_entry_count": ef._get_total_cache_entry_count(code),
        "debug_get_cache_entry_list": ef._debug_get_cache_entry_list(code),
        "reset_code": ef.reset_code(code),
        "get_eval_frame_callback": ef.get_eval_frame_callback(),
    }

    # k. set_eval_frame
    none_ret = ef.set_eval_frame(None)
    hook_refused = False
    refusal_msg = ""
    try:
        ef.set_eval_frame(lambda *a: None)
    except NotImplementedError as e:
        hook_refused = True
        refusal_msg = str(e)
    finally:
        ef.set_eval_frame(None)
    res["set_eval_frame"] = {
        "none_ret": none_ret,
        "hook_refused": hook_refused,
        "refusal_msg": refusal_msg,
    }

    return res

results = run_surface_checks(d, ef, is_shim=(mode == "--shim"))
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
    env.pop("TORCHNATIVE_NO_SURFACE_FALLBACK", None)
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
        raise RuntimeError(f"Oracle failed to run surface checks: {data}")
    return data["results"]


def _run_shim_subprocess(no_fallback=False):
    vendor_dir = _find_vendor_dir()
    env = dict(os.environ)
    worktree_python_dir = str(_REPO_ROOT / "torchnative" / "python")
    if worktree_python_dir != vendor_dir:
        env["PYTHONPATH"] = f"{worktree_python_dir}:{vendor_dir}"
    else:
        env["PYTHONPATH"] = vendor_dir
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    if no_fallback:
        env["TORCHNATIVE_NO_SURFACE_FALLBACK"] = "1"

    proc = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT, "--shim"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"Shim probe exited {proc.returncode}:\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    line = [ln for ln in proc.stdout.strip().splitlines() if ln.startswith("{")][-1]
    return json.loads(line)


def _compare_surface_results(shim, oracle):
    """Compare shim surface results against upstream oracle results.

    Returns list of difference strings. If empty, surface is 100% identical.
    """
    diffs = []

    # 1. slots_constants
    s_slots = shim.get("slots_constants", {})
    o_slots = oracle.get("slots_constants", {})
    for sc_name in ["PySequenceSlots", "PyMappingSlots", "PyNumberSlots", "PyTypeSlots"]:
        s_cls = s_slots.get(sc_name, {})
        o_cls = o_slots.get(sc_name, {})
        if s_cls != o_cls:
            diffs.append(f"slots_constants mismatch on {sc_name}: shim={s_cls} != oracle={o_cls}")

    # 2. type_slots
    s_ts = shim.get("type_slots", {})
    o_ts = oracle.get("type_slots", {})
    for k in o_ts:
        if s_ts.get(k) != o_ts.get(k):
            diffs.append(f"type_slots mismatch on {k}: shim={s_ts.get(k)} != oracle={o_ts.get(k)}")

    # 3. has_slot
    s_hs = shim.get("has_slot", {})
    o_hs = oracle.get("has_slot", {})
    for k in o_hs:
        if s_hs.get(k) != o_hs.get(k):
            diffs.append(f"has_slot mismatch on {k}: shim={s_hs.get(k)} != oracle={o_hs.get(k)}")

    # 4. strip_function_call
    s_sfc = shim.get("strip_function_call", {})
    o_sfc = oracle.get("strip_function_call", {})
    for k in o_sfc:
        if s_sfc.get(k) != o_sfc.get(k):
            diffs.append(f"strip_function_call mismatch on {k!r}: shim={s_sfc.get(k)!r} != oracle={o_sfc.get(k)!r}")
    if shim.get("sfc_raises_type_error") != oracle.get("sfc_raises_type_error"):
        diffs.append("strip_function_call TypeError flag mismatch")

    # 5. is_valid_var_name
    s_ivn = shim.get("is_valid_var_name", {})
    o_ivn = oracle.get("is_valid_var_name", {})
    for k in o_ivn:
        if s_ivn.get(k) != o_ivn.get(k):
            diffs.append(f"is_valid_var_name mismatch on {k!r}: shim={s_ivn.get(k)} != oracle={o_ivn.get(k)}")
    if shim.get("ivn_raises_type_error") != oracle.get("ivn_raises_type_error"):
        diffs.append("is_valid_var_name TypeError flag mismatch")

    # 6. code_framelocals_names
    s_cfn = shim.get("code_framelocals_names", {})
    o_cfn = oracle.get("code_framelocals_names", {})
    for k in o_cfn:
        if s_cfn.get(k) != o_cfn.get(k):
            diffs.append(f"code_framelocals_names mismatch on {k}: shim={s_cfn.get(k)} != oracle={o_cfn.get(k)}")

    # 7. py_opcode_caches
    s_poc = shim.get("py_opcode_caches", [])
    o_poc = oracle.get("py_opcode_caches", [])
    if s_poc != o_poc:
        diffs.append(f"py_opcode_caches mismatch: shim len={len(s_poc)}, oracle len={len(o_poc)}")

    # 8. frame_action
    s_fa = shim.get("frame_action", {})
    o_fa = oracle.get("frame_action", {})
    for k in ["default_val", "skip_val", "run_only_val", "default_eq_0", "skip_eq_1", "run_only_eq_2", "names", "values"]:
        if s_fa.get(k) != o_fa.get(k):
            diffs.append(f"frame_action mismatch on {k}: shim={s_fa.get(k)} != oracle={o_fa.get(k)}")

    # 9. frame_exec_strategy
    s_fes = shim.get("frame_exec_strategy", {})
    o_fes = oracle.get("frame_exec_strategy", {})
    for k in ["s0_cur", "s0_rec", "s1_cur_mutated", "s1_rec_mutated"]:
        if s_fes.get(k) != o_fes.get(k):
            diffs.append(f"frame_exec_strategy mismatch on {k}: shim={s_fes.get(k)} != oracle={o_fes.get(k)}")

    # 10. isolate_recompiles_id
    s_iri = shim.get("isolate_recompiles_id", {})
    o_iri = oracle.get("isolate_recompiles_id", {})
    for k in ["init", "set42_ret", "now42", "restore_ret", "now_restored"]:
        if s_iri.get(k) != o_iri.get(k):
            diffs.append(f"isolate_recompiles_id mismatch on {k}: shim={s_iri.get(k)} != oracle={o_iri.get(k)}")

    # 11. cache_and_reset
    s_cr = shim.get("cache_and_reset", {})
    o_cr = oracle.get("cache_and_reset", {})
    for k in ["get_cache_entries_for_region", "get_total_cache_entry_count", "debug_get_cache_entry_list", "reset_code", "get_eval_frame_callback"]:
        if s_cr.get(k) != o_cr.get(k):
            diffs.append(f"cache_and_reset mismatch on {k}: shim={s_cr.get(k)} != oracle={o_cr.get(k)}")

    return diffs


def test_upstream_dynamo_surface_oracle_runs_all_checks():
    """Verify upstream oracle probe completes cleanly and produces all surface entries."""
    oracle = _run_oracle_subprocess()
    assert "slots_constants" in oracle, "missing slots_constants"
    assert "type_slots" in oracle, "missing type_slots"
    assert "has_slot" in oracle, "missing has_slot"
    assert "strip_function_call" in oracle, "missing strip_function_call"
    assert "is_valid_var_name" in oracle, "missing is_valid_var_name"
    assert "code_framelocals_names" in oracle, "missing code_framelocals_names"
    assert "py_opcode_caches" in oracle, "missing py_opcode_caches"
    assert "frame_action" in oracle, "missing frame_action"
    assert "frame_exec_strategy" in oracle, "missing frame_exec_strategy"
    assert "isolate_recompiles_id" in oracle, "missing isolate_recompiles_id"
    assert "cache_and_reset" in oracle, "missing cache_and_reset"
    assert len(oracle["slots_constants"]["PySequenceSlots"]) == 8
    assert len(oracle["slots_constants"]["PyMappingSlots"]) == 3
    assert len(oracle["slots_constants"]["PyNumberSlots"]) == 35
    assert len(oracle["slots_constants"]["PyTypeSlots"]) == 11
    assert len(oracle["py_opcode_caches"]) == 256


def test_differential_surface_comparator_catches_a_planted_divergence():
    """Nullification guard (AGENTS.md §17.5).

    Verifies that any diverged slot constant, bit, string return, type, opcode
    cache entry, or strategy action is caught by the differential comparator.
    """
    oracle = _run_oracle_subprocess()
    assert not _compare_surface_results(oracle, oracle), "Self-comparison must produce zero diffs"

    # 1. Plant slots_constants divergence
    mut = copy.deepcopy(oracle)
    mut["slots_constants"]["PySequenceSlots"]["SQ_LENGTH"] = 99
    assert any("slots_constants" in d for d in _compare_surface_results(mut, oracle)), "Failed to catch slots_constants diff"

    # 2. Plant type_slots divergence
    mut = copy.deepcopy(oracle)
    mut["type_slots"]["int"][0] = 999999
    assert any("type_slots" in d for d in _compare_surface_results(mut, oracle)), "Failed to catch type_slots diff"

    # 3. Plant has_slot divergence
    mut = copy.deepcopy(oracle)
    mut["has_slot"]["int"][0] = not mut["has_slot"]["int"][0]
    assert any("has_slot" in d for d in _compare_surface_results(mut, oracle)), "Failed to catch has_slot diff"

    # 4. Plant strip_function_call divergence
    mut = copy.deepcopy(oracle)
    mut["strip_function_call"]["foo(a)"] = "wrong"
    assert any("strip_function_call" in d for d in _compare_surface_results(mut, oracle)), "Failed to catch strip_function_call diff"

    # 5. Plant is_valid_var_name value and type divergence
    mut = copy.deepcopy(oracle)
    mut["is_valid_var_name"]["1"] = [True, "bool"]  # Upstream is [2, "int"]
    assert any("is_valid_var_name" in d for d in _compare_surface_results(mut, oracle)), "Failed to catch is_valid_var_name diff"

    # 6. Plant code_framelocals_names divergence
    mut = copy.deepcopy(oracle)
    mut["code_framelocals_names"]["f_plain"] = ["b", "a"]
    assert any("code_framelocals_names" in d for d in _compare_surface_results(mut, oracle)), "Failed to catch code_framelocals_names diff"

    # 7. Plant py_opcode_caches divergence
    mut = copy.deepcopy(oracle)
    mut["py_opcode_caches"][0] = 999
    assert any("py_opcode_caches" in d for d in _compare_surface_results(mut, oracle)), "Failed to catch py_opcode_caches diff"

    # 8. Plant frame_action divergence
    mut = copy.deepcopy(oracle)
    mut["frame_action"]["default_val"] = 99
    assert any("frame_action" in d for d in _compare_surface_results(mut, oracle)), "Failed to catch frame_action diff"

    # 9. Plant frame_exec_strategy divergence
    mut = copy.deepcopy(oracle)
    mut["frame_exec_strategy"]["s0_cur"] = 99
    assert any("frame_exec_strategy" in d for d in _compare_surface_results(mut, oracle)), "Failed to catch frame_exec_strategy diff"

    # 10. Plant isolate_recompiles_id divergence
    mut = copy.deepcopy(oracle)
    mut["isolate_recompiles_id"]["init"] = 0
    assert any("isolate_recompiles_id" in d for d in _compare_surface_results(mut, oracle)), "Failed to catch isolate_recompiles_id diff"

    # 11. Plant cache_and_reset divergence
    mut = copy.deepcopy(oracle)
    mut["cache_and_reset"]["get_total_cache_entry_count"] = 1
    assert any("cache_and_reset" in d for d in _compare_surface_results(mut, oracle)), "Failed to catch cache_and_reset diff"


def test_shim_dynamo_surface_matches_upstream_oracle_and_preserves_set_eval_frame_refusal():
    """Verify shim dynamo surface matches upstream oracle and preserves set_eval_frame refusal.

    In Phase 1 (before _dynamo_surface is installed), the shim probe reports
    status="unimplemented" and this test fails (red state).
    Once installed, this test demands 100% differential equality with the
    upstream PyTorch oracle and asserts that set_eval_frame refuses non-None
    callbacks by name, preserving PEP 523 and abi3 messaging.
    """
    oracle = _run_oracle_subprocess()
    data = _run_shim_subprocess()
    status = data.get("status")

    if status != "implemented":
        refusal = data.get("refusal", str(data))
        raise AssertionError(f"Shim dynamo surface is not implemented: {refusal}")

    results = data.get("results", {})
    diffs = _compare_surface_results(results, oracle)
    assert not diffs, (
        "Differential divergences detected between shim and upstream dynamo surface:\n"
        + "\n".join(diffs)
    )

    # Verify set_eval_frame refusal preserves PEP 523 / abi3 messaging
    ef_info = results.get("set_eval_frame", {})
    assert ef_info.get("none_ret") is None, f"set_eval_frame(None) must return None, got {ef_info.get('none_ret')}"
    assert ef_info.get("hook_refused") is True, "set_eval_frame(callback) must refuse non-None hook"
    msg = ef_info.get("refusal_msg", "")
    assert "PEP 523" in msg, f"Refusal message must name PEP 523, got {msg!r}"
    assert "abi3" in msg, f"Refusal message must name abi3, got {msg!r}"


def _main():
    tests = [
        ("test_upstream_dynamo_surface_oracle_runs_all_checks", test_upstream_dynamo_surface_oracle_runs_all_checks),
        ("test_differential_surface_comparator_catches_a_planted_divergence", test_differential_surface_comparator_catches_a_planted_divergence),
        ("test_shim_dynamo_surface_matches_upstream_oracle_and_preserves_set_eval_frame_refusal", test_shim_dynamo_surface_matches_upstream_oracle_and_preserves_set_eval_frame_refusal),
    ]
    return _skip.run_tests(tests, suite="test_dynamo_surface")


if __name__ == "__main__":
    sys.exit(_main())
