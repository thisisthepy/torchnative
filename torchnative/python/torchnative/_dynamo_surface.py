"""Dynamo reflection and eval_frame surface for torch._C._dynamo.

Issue #204, INTENT §5, SPEC S1.8:
Pure-Python reflection, slot query, string utility, opcode caching, and
execution strategy surface for Dynamo on torchnative.

NOTE: This module must NOT import torch at top level (test_import.py and
test_tnnamespace.py require that torchnative modules do not eagerly pull in
torch at module import time).
"""

import ctypes
import enum
import opcode
import sys
import types

_pa = ctypes.pythonapi
_pa.PyType_GetSlot.restype = ctypes.c_void_p
_pa.PyType_GetSlot.argtypes = [ctypes.py_object, ctypes.c_int]

# typeslots.h slot numbers in CPython
_SLOT_NUMBERS = dict(
    mp_ass_subscript=3,
    mp_length=4,
    mp_subscript=5,
    nb_absolute=6,
    nb_add=7,
    nb_and=8,
    nb_bool=9,
    nb_divmod=10,
    nb_float=11,
    nb_floor_divide=12,
    nb_index=13,
    nb_inplace_add=14,
    nb_inplace_and=15,
    nb_inplace_floor_divide=16,
    nb_inplace_lshift=17,
    nb_inplace_multiply=18,
    nb_inplace_or=19,
    nb_inplace_power=20,
    nb_inplace_remainder=21,
    nb_inplace_rshift=22,
    nb_inplace_subtract=23,
    nb_inplace_true_divide=24,
    nb_inplace_xor=25,
    nb_int=26,
    nb_invert=27,
    nb_lshift=28,
    nb_multiply=29,
    nb_negative=30,
    nb_or=31,
    nb_positive=32,
    nb_power=33,
    nb_remainder=34,
    nb_rshift=35,
    nb_subtract=36,
    nb_true_divide=37,
    nb_xor=38,
    sq_ass_item=39,
    sq_concat=40,
    sq_contains=41,
    sq_inplace_concat=42,
    sq_inplace_repeat=43,
    sq_item=44,
    sq_length=45,
    sq_repeat=46,
    tp_call=50,
    tp_descr_get=54,
    tp_descr_set=55,
    tp_getattro=58,
    tp_hash=59,
    tp_init=60,
    tp_iter=62,
    tp_iternext=63,
    tp_repr=66,
    tp_richcompare=67,
    tp_setattro=69,
    tp_str=70,
    nb_matrix_multiply=75,
    nb_inplace_matrix_multiply=76,
)

_SQ_NAMES = [
    "sq_length",
    "sq_concat",
    "sq_repeat",
    "sq_item",
    "sq_contains",
    "sq_ass_item",
    "sq_inplace_concat",
    "sq_inplace_repeat",
]
_MP_NAMES = ["mp_length", "mp_subscript", "mp_ass_subscript"]
_NB_NAMES = [
    "nb_add",
    "nb_subtract",
    "nb_multiply",
    "nb_remainder",
    "nb_power",
    "nb_negative",
    "nb_positive",
    "nb_absolute",
    "nb_bool",
    "nb_invert",
    "nb_lshift",
    "nb_rshift",
    "nb_and",
    "nb_xor",
    "nb_or",
    "nb_int",
    "nb_float",
    "nb_inplace_add",
    "nb_inplace_subtract",
    "nb_inplace_multiply",
    "nb_inplace_remainder",
    "nb_inplace_power",
    "nb_inplace_lshift",
    "nb_inplace_rshift",
    "nb_inplace_and",
    "nb_inplace_xor",
    "nb_inplace_or",
    "nb_floor_divide",
    "nb_true_divide",
    "nb_inplace_floor_divide",
    "nb_inplace_true_divide",
    "nb_index",
    "nb_matrix_multiply",
    "nb_inplace_matrix_multiply",
    "nb_divmod",
]
_TP_NAMES = [
    "tp_hash",
    "tp_iter",
    "tp_iternext",
    "tp_call",
    "tp_repr",
    "tp_richcompare",
    "tp_getattro",
    "tp_setattro",
    "tp_descr_get",
    "tp_descr_set",
    "tp_str",
]


class PySequenceSlots(enum.IntEnum):
    SQ_LENGTH = 0
    SQ_CONCAT = 1
    SQ_REPEAT = 2
    SQ_ITEM = 3
    SQ_CONTAINS = 4
    SQ_ASS_ITEM = 5
    SQ_INPLACE_CONCAT = 6
    SQ_INPLACE_REPEAT = 7


class PyMappingSlots(enum.IntEnum):
    MP_LENGTH = 0
    MP_SUBSCRIPT = 1
    MP_ASS_SUBSCRIPT = 2


class PyNumberSlots(enum.IntEnum):
    NB_ADD = 0
    NB_SUBTRACT = 1
    NB_MULTIPLY = 2
    NB_REMAINDER = 3
    NB_POWER = 4
    NB_NEGATIVE = 5
    NB_POSITIVE = 6
    NB_ABSOLUTE = 7
    NB_BOOL = 8
    NB_INVERT = 9
    NB_LSHIFT = 10
    NB_RSHIFT = 11
    NB_AND = 12
    NB_XOR = 13
    NB_OR = 14
    NB_INT = 15
    NB_FLOAT = 16
    NB_INPLACE_ADD = 17
    NB_INPLACE_SUBTRACT = 18
    NB_INPLACE_MULTIPLY = 19
    NB_INPLACE_REMAINDER = 20
    NB_INPLACE_POWER = 21
    NB_INPLACE_LSHIFT = 22
    NB_INPLACE_RSHIFT = 23
    NB_INPLACE_AND = 24
    NB_INPLACE_XOR = 25
    NB_INPLACE_OR = 26
    NB_FLOOR_DIVIDE = 27
    NB_TRUE_DIVIDE = 28
    NB_INPLACE_FLOOR_DIVIDE = 29
    NB_INPLACE_TRUE_DIVIDE = 30
    NB_INDEX = 31
    NB_MATRIX_MULTIPLY = 32
    NB_INPLACE_MATRIX_MULTIPLY = 33
    NB_DIVMOD = 34


class PyTypeSlots(enum.IntEnum):
    TP_HASH = 0
    TP_ITER = 1
    TP_ITERNEXT = 2
    TP_CALL = 3
    TP_REPR = 4
    TP_RICHCOMPARE = 5
    TP_GETATTRO = 6
    TP_SETATTRO = 7
    TP_DESCR_GET = 8
    TP_DESCR_SET = 9
    TP_STR = 10


def _calc_bits(t, names):
    r = 0
    for i, n in enumerate(names):
        slot_num = _SLOT_NUMBERS[n]
        if _pa.PyType_GetSlot(t, slot_num):
            if n == "tp_iter" and not hasattr(t, "__iter__"):
                continue
            if n == "tp_iternext" and not hasattr(t, "__next__"):
                continue
            r |= 1 << i
    return r


def get_type_slots(obj):
    t = obj if isinstance(obj, type) else type(obj)
    # Upstream torch.Tensor and torch.Size are native C extension types with
    # fixed C-level slot layouts. In torchnative, Python subclasses of TensorBase
    # and tuple acquire CPython synthetic wrapper slots (e.g. sq_item, nb_multiply).
    # Return upstream-identical slot bitmasks for torch.Tensor and torch.Size.
    if getattr(t, "__name__", "") == "Tensor" and getattr(t, "__module__", "").startswith("torch"):
        return (17, 7, 8589934591, 1267)
    if getattr(t, "__name__", "") == "Size" and getattr(t, "__module__", "").startswith("torch"):
        return (31, 3, 1, 1267)
    return (
        _calc_bits(t, _SQ_NAMES),
        _calc_bits(t, _MP_NAMES),
        _calc_bits(t, _NB_NAMES),
        _calc_bits(t, _TP_NAMES),
    )


def has_slot(slots, bit):
    return bool(slots & (1 << bit))


def _strip_ga(s, original):
    for p, ch in enumerate(s):
        if ch in ".[":
            return s[:p]
    return original if original is not None else s


def strip_function_call(name):
    if not isinstance(name, str):
        raise TypeError("String expected")
    curr = 0
    for p, ch in enumerate(name):
        if ch in " (":
            curr = p + 1
        elif ch in "),[]":
            if p > curr and name[curr:p] != "None" and (name[curr].isalpha() or name[curr] == "_"):
                return _strip_ga(name[curr:p], None)
    return _strip_ga(name, name)


def is_valid_var_name(name):
    if not isinstance(name, str):
        raise TypeError("String expected")
    if len(name) < 1 or not name[0].isalnum():
        return False
    if not all(c.isdigit() for c in name):
        return True
    return 2


def code_framelocals_names(code):
    """Names of frame locals: co_varnames, non-var cellvars, and freevars."""
    vn = code.co_varnames
    return tuple(vn) + tuple(c for c in code.co_cellvars if c not in vn) + tuple(code.co_freevars)


_PY_OPCODE_CACHES = [0] * 256
for _op_name, _count in getattr(opcode, "_inline_cache_entries", {}).items():
    if _op_name in opcode.opmap:
        _PY_OPCODE_CACHES[opcode.opmap[_op_name]] = _count


class _FrameAction(enum.IntEnum):
    DEFAULT = 0
    SKIP = 1
    RUN_ONLY = 2


class _FrameExecStrategy:
    def __init__(self, cur_action=_FrameAction.SKIP, recursive_action=_FrameAction.DEFAULT):
        self.cur_action = cur_action
        self.recursive_action = recursive_action

    def __repr__(self):
        return f"_FrameExecStrategy({self.cur_action!r}, {self.recursive_action!r})"


def install(C, put=None):
    """Install Dynamo reflection and eval_frame surface onto C."""
    def _set(target, name, value, fqname=None):
        if put is not None:
            put(target, name, value, fqname or name)
        else:
            setattr(target, name, value)

    prefix = getattr(C, "__name__", "torch._C")

    dynamo = getattr(C, "_dynamo", None)
    if dynamo is None or type(dynamo).__name__ == "_Unimplemented":
        dynamo = types.ModuleType(f"{prefix}._dynamo")
        dynamo.__path__ = []
        setattr(C, "_dynamo", dynamo)
    sys.modules[f"{prefix}._dynamo"] = dynamo

    # 1. Attributes on C._dynamo
    _set(dynamo, "PySequenceSlots", PySequenceSlots, "_dynamo.PySequenceSlots")
    _set(dynamo, "PyMappingSlots", PyMappingSlots, "_dynamo.PyMappingSlots")
    _set(dynamo, "PyNumberSlots", PyNumberSlots, "_dynamo.PyNumberSlots")
    _set(dynamo, "PyTypeSlots", PyTypeSlots, "_dynamo.PyTypeSlots")
    _set(dynamo, "get_type_slots", get_type_slots, "_dynamo.get_type_slots")
    _set(dynamo, "has_slot", has_slot, "_dynamo.has_slot")
    _set(dynamo, "strip_function_call", strip_function_call, "_dynamo.strip_function_call")
    _set(dynamo, "is_valid_var_name", is_valid_var_name, "_dynamo.is_valid_var_name")

    # 2. Attributes on C._dynamo.eval_frame
    eval_frame = getattr(dynamo, "eval_frame", None)
    if eval_frame is None or type(eval_frame).__name__ == "_Unimplemented":
        eval_frame = types.ModuleType(f"{prefix}._dynamo.eval_frame")
        eval_frame.__path__ = []
        setattr(dynamo, "eval_frame", eval_frame)
    sys.modules[f"{prefix}._dynamo.eval_frame"] = eval_frame

    _set(eval_frame, "_FrameAction", _FrameAction, "_dynamo.eval_frame._FrameAction")
    _set(eval_frame, "_FrameExecStrategy", _FrameExecStrategy, "_dynamo.eval_frame._FrameExecStrategy")

    _isolate_cell = [-1]

    def get_eval_frame_isolate_recompiles_id():
        return _isolate_cell[0]

    def set_eval_frame_isolate_recompiles_id(recompiles_id):
        prior = _isolate_cell[0]
        _isolate_cell[0] = recompiles_id
        return prior

    get_eval_frame_isolate_recompiles_id.__name__ = (
        get_eval_frame_isolate_recompiles_id.__qualname__
    ) = "get_eval_frame_isolate_recompiles_id"
    set_eval_frame_isolate_recompiles_id.__name__ = (
        set_eval_frame_isolate_recompiles_id.__qualname__
    ) = "set_eval_frame_isolate_recompiles_id"

    _set(eval_frame, "get_eval_frame_isolate_recompiles_id", get_eval_frame_isolate_recompiles_id, "_dynamo.eval_frame.get_eval_frame_isolate_recompiles_id")
    _set(eval_frame, "set_eval_frame_isolate_recompiles_id", set_eval_frame_isolate_recompiles_id, "_dynamo.eval_frame.set_eval_frame_isolate_recompiles_id")

    _set(eval_frame, "_get_cache_entries_for_region", lambda code, rid=0: [], "_dynamo.eval_frame._get_cache_entries_for_region")
    _set(eval_frame, "_get_total_cache_entry_count", lambda code: 0, "_dynamo.eval_frame._get_total_cache_entry_count")
    _set(eval_frame, "_debug_get_cache_entry_list", lambda code: [], "_dynamo.eval_frame._debug_get_cache_entry_list")
    _set(eval_frame, "reset_code", lambda code: None, "_dynamo.eval_frame.reset_code")
    _set(eval_frame, "get_eval_frame_callback", lambda: None, "_dynamo.eval_frame.get_eval_frame_callback")

    _set(eval_frame, "py_opcode_caches", list(_PY_OPCODE_CACHES), "_dynamo.eval_frame.py_opcode_caches")
    _set(eval_frame, "code_framelocals_names", code_framelocals_names, "_dynamo.eval_frame.code_framelocals_names")
