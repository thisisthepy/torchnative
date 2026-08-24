"""Smoke tests for the built `torch._C` shim.

Linking is not proof. RUST_CROSSBUILD.md §0.5 checks that each target produces
an artefact with the right shape; this file checks that the host artefact,
renamed to `_C.so`, actually imports and computes.

Run it against a built extension:

    ./pytests/run.sh                      # builds, renames, runs
    PYTHONPATH=<dir with _C.so> python3 -m pytest pytests/test_shim.py

Written with plain asserts so it runs under pytest or on its own, without
adding a test dependency to a package that has none.
"""

import _C


# --- identity ---------------------------------------------------------------


def test_module_loads():
    assert _C._shim_target()
    assert isinstance(_C._aten_implemented(), list)


def test_dtype_is_a_type_owned_by_c():
    # torch.float32 is an instance of a C-defined type, not a Python constant.
    assert isinstance(_C.float32, _C.dtype)
    assert repr(_C.float32) == "torch.float32"
    assert _C.float32 == _C.float32
    assert _C.float32 != _C.int64
    assert {_C.float32, _C.float32} == {_C.float32}
    assert _C.float32.is_floating_point
    assert not _C.int64.is_floating_point
    assert _C.float32.itemsize == 4


def test_device_is_a_label_not_a_backend():
    cpu = _C.device("cpu")
    assert cpu.type == "cpu"
    assert cpu.index is None
    assert repr(cpu) == "device(type='cpu')"
    assert _C.device("cuda:0").type == "cuda"
    assert _C.device("cuda:0").index == 0
    # Constructible for a backend this build has no kernels for -- as in torch.
    assert _C.device("cuda") != _C.device("cpu")


def test_tensor_exposes_shape_dtype_device():
    t = _C._aten_dispatch("aten.full.default", [2, 3], 1.5)
    assert isinstance(t, _C.TensorBase)
    assert t.shape == (2, 3)
    assert t.dtype == _C.float32
    assert t.device == _C.device("cpu")
    assert t.ndim == 2
    assert t.numel() == 6
    assert t.size(0) == 2
    assert t.size(-1) == 3
    assert t.tolist() == [[1.5, 1.5, 1.5], [1.5, 1.5, 1.5]]


def test_tensor_base_is_subclassable():
    # torch/_tensor.py does exactly this: class Tensor(torch._C.TensorBase).
    class Tensor(_C.TensorBase):
        pass

    assert issubclass(Tensor, _C.TensorBase)


# --- discovery (DESIGN.md §6) ----------------------------------------------


def test_unimplemented_op_names_itself():
    # `aten.embedding.default` used to stand here and now has a kernel, which
    # is the right failure mode for this test -- it goes red when the op it
    # samples stops being a sample. `relu` is the example docs/TORCH_C.md §1
    # uses and is on no current work list.
    try:
        _C._aten_dispatch("aten.relu.default")
    except NotImplementedError as e:
        assert str(e) == "aten op not implemented in torch._C shim: aten.relu.default"
    else:
        raise AssertionError("an unimplemented op must raise")


def test_every_advertised_op_is_actually_dispatchable():
    # A name in the list that falls through to the fallback would make the
    # instrument lie about what is covered.
    for op in _C._aten_implemented() + _C._aten_implemented_awaiting_golden():
        try:
            _C._aten_dispatch(op)
        except NotImplementedError as e:  # pragma: no cover - regression guard
            raise AssertionError(f"{op} is advertised but not dispatchable: {e}")
        except TypeError:
            pass  # missing arguments: reached the kernel, which is the point


def test_the_two_op_lists_are_disjoint():
    # `_aten_implemented_awaiting_golden()` under-reports on purpose (it is
    # kept out of the harness's coverage count until a case builder exists).
    # An op appearing in both would make it over-report instead, which is the
    # direction docs/TORCH_C.md §1 says the instrument must never take.
    advertised = set(_C._aten_implemented())
    parked = set(_C._aten_implemented_awaiting_golden())
    assert not (advertised & parked)


# --- overload resolution (docs/OVERLOAD.md) ---------------------------------
#
# `torch.<op>` lives on `_C._VariableFunctions`; `torch/__init__.py` hoists it
# onto the `torch` namespace at import. These tests reach it at the source, so
# they run against the bare artefact with no vendored tree present.


def _vf(name):
    return getattr(_C._VariableFunctions, name)


def _resolved_key(name, *args, **kwargs):
    """Which aten key does this call land on?

    Asked by making the call fail at the dispatcher, which is the only place
    that knows -- an assertion on the *message* would be asserting on wording.
    So instead the call is aimed at an op whose kernel exists and the answer is
    read from the result, except here where the point is the key itself: an
    unimplemented overload names itself, and that name is the answer.
    """
    try:
        _vf(name)(*args, **kwargs)
    except NotImplementedError as e:
        text = str(e)
        marker = "aten op not implemented in torch._C shim: "
        assert marker in text, text
        return text.split(marker, 1)[1].strip()
    raise AssertionError(f"torch.{name} was expected to reach an unimplemented op")


def test_overload_resolution_picks_the_same_key_torch_picks():
    # Each of these was measured on real torch 2.13.0 with a TorchDispatchMode
    # logger (docs/OVERLOAD.md §3). The two arange rows are the ones a
    # plausible reading of the schemas gets wrong: `start_step` has a default
    # for `step`, so it would happily swallow the two-argument call.
    t = _C._aten_dispatch("aten.full.default", [2], 1.0)
    assert _vf("arange")(5).dtype == _C.int64
    assert _vf("arange")(0.0, 5).dtype == _C.float32
    assert _vf("ones")(2, 3).shape == (2, 3)
    assert _vf("full")([2], True).dtype == _C.bool
    # `out=` selects the `.out` variant, which has no kernel -- so the key it
    # names is the proof it was selected.
    assert _resolved_key("full", [2], 1.0, out=t) == "aten.full.out"
    assert _resolved_key("arange", 5, out=t) == "aten.arange.out"
    assert _resolved_key("mm", t, t, out=t) == "aten.mm.out"


def test_overload_resolution_refuses_rather_than_guessing():
    # `torch.full` has two positional arguments, so the varargs int-list rule
    # does not apply and `full(2, 3)` is not `full([2], 3)`.
    try:
        _vf("full")(2, 3)
    except TypeError as e:
        assert "no matching overload" in str(e)
        # The candidates are listed, because the work item is "which signature
        # did I mean", not "something went wrong".
        assert "aten::full" in str(e)
    else:
        raise AssertionError("full(2, 3) must not resolve")

    # An op with no table entry keeps the old refusal rather than guessing
    # `.default`.
    try:
        _vf("relu")(1)
    except NotImplementedError as e:
        assert "no table entry" in str(e)
    else:
        raise AssertionError("an op with no table entry must refuse")


def test_varargs_intlist_rule_matches_torchs_precondition():
    # torch allows `ones(2, 3)` only because `ones`'s single positional
    # argument is an int list. The same spelling on a two-positional signature
    # is an error, which the previous test covers from the other side.
    assert _vf("ones")(2, 3).shape == (2, 3)
    assert _vf("ones")((2, 3)).shape == (2, 3)
    assert _vf("ones")(2).shape == (2,)


def test_requires_grad_is_refused_not_ignored():
    try:
        _vf("ones")(2, requires_grad=True)
    except NotImplementedError as e:
        assert "autograd" in str(e)
    else:
        raise AssertionError("requires_grad=True must not be silently dropped")
    # ... and the falsy spelling, which the vendored tree passes constantly,
    # goes through.
    assert _vf("ones")(2, requires_grad=False).shape == (2,)


def test_torch_tensor_builds_data_and_goes_through_lift_fresh():
    tensor = _vf("tensor")
    assert tensor([1, 2]).dtype == _C.int64
    assert tensor([1.0, 2]).dtype == _C.float32
    # The category test is the reason this is not just "is it a number":
    # `bool` subclasses `int`, and a bool list is a mask.
    assert tensor([True, False]).dtype == _C.bool
    assert tensor([True, False]).tolist() == [True, False]
    assert tensor(5).shape == ()
    assert tensor([[1, 2], [3, 4]]).shape == (2, 2)
    try:
        tensor([[1], 2])
    except ValueError:
        pass
    else:
        raise AssertionError("a ragged nested sequence must be refused")


def test_the_overload_table_is_inspectable():
    # Same reason as `_shim_off_switches`: which keys a `torch.<op>` call can
    # reach should be answerable by asking, not by reading the table back out
    # of the artefact.
    table = _C._shim_overloads
    assert table["full"] == ["aten.full.out", "aten.full.default"]
    assert "aten.arange.start" in table["arange"]
    # Every key the table can produce is either implemented or refuses by
    # name; nothing in between.
    for keys in table.values():
        for key in keys:
            try:
                _C._aten_dispatch(key)
            except (NotImplementedError, TypeError):
                pass


# --- implemented ops --------------------------------------------------------


def test_full_infers_dtype_from_the_fill_value():
    assert _C._aten_dispatch("aten.full.default", [2], 1.0).dtype == _C.float32
    assert _C._aten_dispatch("aten.full.default", [2], 1).dtype == _C.int64
    explicit = _C._aten_dispatch("aten.full.default", [2], 1, dtype=_C.float64)
    assert explicit.dtype == _C.float64
    assert explicit.tolist() == [1.0, 1.0]


def test_full_rejects_arguments_it_does_not_honour():
    try:
        _C._aten_dispatch("aten.full.default", [2], 1.0, None, "strided")
    except NotImplementedError as e:
        assert "layout" in str(e)
    else:
        raise AssertionError("a dropped argument must not look supported")


def test_add_broadcasts_and_applies_alpha():
    a = _C._tensor_from_flat([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = _C._tensor_from_flat([10.0, 20.0], [1, 2])
    assert _C._aten_dispatch("aten.add.Tensor", a, b).tolist() == [
        [11.0, 22.0],
        [13.0, 24.0],
    ]
    assert _C._aten_dispatch("aten.add.Tensor", a, b, alpha=2.0).tolist() == [
        [21.0, 42.0],
        [23.0, 44.0],
    ]


def test_add_refuses_to_guess_a_promotion():
    a = _C._tensor_from_flat([1.0], [1], dtype=_C.float32)
    b = _C._tensor_from_flat([1.0], [1], dtype=_C.float64)
    try:
        _C._aten_dispatch("aten.add.Tensor", a, b)
    except NotImplementedError as e:
        assert "promotion" in str(e)
        # torch spellings, not candle's `f32`/`f64`. The shim owns the dtype
        # tag now (BOOL.md §5-B), and candle's spelling cannot tell
        # `torch.bool` from `torch.uint8` -- both are `u8` down there. A
        # message that cannot name the difference cannot report the bug the
        # whole distinction exists to catch.
        assert "float32" in str(e) and "float64" in str(e)
    else:
        raise AssertionError("dtype promotion must not be silently invented")


def test_mm_matches_torch():
    # torch.mm(torch.tensor([[1., 2.], [3., 4.]]), torch.tensor([[5., 6.], [7., 8.]]))
    # -> [[19., 22.], [43., 50.]]
    a = _C._tensor_from_flat([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = _C._tensor_from_flat([5.0, 6.0, 7.0, 8.0], [2, 2])
    assert _C._aten_dispatch("aten.mm.default", a, b).tolist() == [
        [19.0, 22.0],
        [43.0, 50.0],
    ]


def test_mm_is_2d_only():
    a = _C._tensor_from_flat([1.0] * 8, [2, 2, 2])
    try:
        _C._aten_dispatch("aten.mm.default", a, a)
    except RuntimeError as e:
        assert "2D" in str(e)
    else:
        raise AssertionError("candle's batched matmul must not stand in for mm")


# --- the dtype tag (BOOL.md option B) ---------------------------------------


def test_dtype_aliases_are_one_object():
    # torch guarantees `torch.float is torch.float32`, and the vendored tree
    # uses dtypes as dict keys often enough that "equal but not identical"
    # would be a difference worth not having.
    assert _C.float is _C.float32
    assert _C.long is _C.int64
    assert _C.cfloat is _C.complex64


def test_bool_is_not_uint8():
    # BOOL.md's entire argument in one assertion. candle stores both as U8;
    # aliasing them would make `bool + bool` an arithmetic sum instead of a
    # logical or and would switch off six of torch's own guardrails (§7).
    assert _C.bool != _C.uint8
    assert _C.bool.itemsize == _C.uint8.itemsize == 1
    assert _C.bool.abbr == "b8" and _C.uint8.abbr == "u8"


def test_full_gives_a_bool_tensor_for_a_bool_fill():
    # torch.full((2,), True).dtype is torch.bool. `bool` subclasses `int` in
    # Python, so before the tag existed this fell into the integer branch and
    # produced int64 -- docs/TORCH_C.md §2 recorded it as unfixable then.
    t = _C._aten_dispatch("aten.full.default", [2], True)
    assert t.dtype == _C.bool
    # and it reads back as Python bools, not as 0/1
    assert t.tolist() == [True, True]
    assert _C._aten_dispatch("aten.full.default", [2], False).tolist() == [False, False]


def test_bool_addition_refuses_rather_than_summing():
    # `bool + bool` is a logical or in torch (BOOL.md §2.2). candle would give
    # 2, which is still truthy and therefore silently wrong downstream.
    t = _C._aten_dispatch("aten.full.default", [2], True)
    try:
        _C._aten_dispatch("aten.add.Tensor", t, t)
    except NotImplementedError as e:
        assert "logical or" in str(e)
    else:
        raise AssertionError("bool addition must not be candle's arithmetic sum")


def test_dtypes_candle_cannot_store_are_named_but_refused():
    # `_C` names all 33 torch dtypes because the vendored tree needs them;
    # candle can store ten. The other twenty-three must fail by name rather
    # than land on a near neighbour.
    assert _C.complex64.itemsize == 8
    assert not _C.complex64._has_storage
    assert _C.float32._has_storage
    try:
        _C._aten_dispatch("aten.full.default", [2], 1.0, dtype=_C.complex64)
    except NotImplementedError as e:
        assert "complex64" in str(e)
    else:
        raise AssertionError("a dtype candle cannot store must not be substituted")


# --- torch's overflow rule for `full` ---------------------------------------


def test_full_refuses_a_fill_the_dtype_cannot_hold():
    # Found by tools/golden/compare.py: torch raises here and the shim used to
    # saturate to inf / wrap to int32 min.
    for dtype, fill in ((_C.float16, 1e6), (_C.int32, 2**31), (_C.uint8, 300)):
        try:
            _C._aten_dispatch("aten.full.default", [3], fill, dtype=dtype)
        except RuntimeError as e:
            assert "without overflow" in str(e)
        else:
            raise AssertionError(f"{dtype} must refuse a fill of {fill}")


def test_full_wraps_a_negative_into_an_unsigned_dtype():
    # c10 allows two's-complement wrap for negatives whose magnitude fits, so
    # `full(-1, uint8)` is 255 in torch and must be 255 here.
    assert _C._aten_dispatch("aten.full.default", [2], -1, dtype=_C.uint8).tolist() == [
        255,
        255,
    ]


def test_full_reproduces_torchs_one_element_hole():
    # Measured on torch 2.13.0: `fill_` takes a CPU numel==1 fast path whose
    # conversion is unchecked, but only for the reduced-precision float types.
    # So `full([], 1e6, float16)` is inf while `full([3], 1e6, float16)`
    # raises. Matching torch matters more than being tidy -- the golden harness
    # compares against torch, and a shim that always refused would diverge in
    # the other direction.
    assert _C._aten_dispatch(
        "aten.full.default", [], 1e6, dtype=_C.float16
    ).tolist() == float("inf")
    try:
        _C._aten_dispatch("aten.full.default", [], 2**31, dtype=_C.int32)
    except RuntimeError:
        pass
    else:
        raise AssertionError("the one-element hole is float16/bfloat16 only")


# --- the name surface -------------------------------------------------------


def test_the_import_surface_is_present():
    # docs/IMPORT_TORCH.md: `import torch` needs these before it finishes.
    assert len([n for n in dir(_C.TensorBase) if not n.startswith("__")]) > 500
    assert len(dir(_C._VariableFunctions)) > 900
    assert isinstance(_C._jit_tree_views.SourceRangeFactory, type)
    # instantiable with arguments -- torch/_sources.py:87 subclasses it and
    # calls super().__init__ with four arguments while nn.functional imports
    _C._jit_tree_views.SourceRangeFactory("", None, 0, 0)


def test_variable_functions_members_do_not_bind():
    # torch/__init__.py assigns `__module__` on every harvested member, which
    # is read-only on a bound method. Upstream gets away with it because these
    # are builtin functions; ours are instance attributes for the same reason.
    fn = _C._VariableFunctions.add
    fn.__module__ = "torch"
    assert fn.__module__ == "torch"


def test_a_placeholder_refuses_a_truth_test():
    # The whole point of `_Unimplemented` is that it is not a chameleon. A
    # truth test is a question, and answering it is what a chameleon does.
    #
    # This is not hypothetical. `_has_cudnn` was one of these, and
    # `torch/backends/cudnn/__init__.py:231` -- a class body, so it runs during
    # `import torch` -- is `if is_available():` over `return
    # torch._C._has_cudnn`. A truthy placeholder took the "cuDNN is here"
    # branch and bound `CudnnModule.benchmark_limit` to a pair of placeholders,
    # where upstream without cuDNN leaves it `None`.
    #
    # The name sampled here is undeclared by the stubs (it reaches the surface
    # only through the tree text scan), so the shim genuinely does not know
    # whether it is a flag or a function. "I cannot answer that" is the true
    # state, and raising is how it is said.
    assert isinstance(_C._is_cow_tensor, type(_C._is_alias_of))
    try:
        bool(_C._is_cow_tensor)
    except NotImplementedError as e:
        assert "_is_cow_tensor" in str(e)
    else:
        raise AssertionError("a placeholder must not answer a truth test")


def test_every_build_flag_the_stubs_declare_answers_with_a_real_bool():
    # `surface.json` marks the module-level names the stubs annotate `_bool`.
    # Each is a build-configuration flag: it has exactly two possible answers
    # and both change behaviour, so there is no honest placeholder for one.
    # Nothing here checks *which* bool -- that is a fact about this build and
    # lives in `_BUILD_FLAGS` with its reason. What is checked is that the
    # question is answered in kind, which is the property a truthy placeholder
    # silently broke for `_has_cudnn`.
    import json
    import os

    here = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(here, os.pardir, "src", "surface.json")) as fh:
        surface = json.load(fh)
    declared = [n for n, kind in surface["module"].items() if kind == "bool"]
    assert len(declared) >= 14, declared
    off = set(_C._shim_off_switches)
    for name in declared:
        if name in off:
            assert not hasattr(_C, name), f"{name} is an off-switch and must be absent"
            continue
        value = getattr(_C, name)
        assert type(value) is bool, f"{name} answers {type(value).__name__}, not bool"


def test_off_switches_stay_off():
    # VENDOR.md wall 11: the vendored tree turns subsystems off by asking
    # whether `_C` has a name. A module-level catch-all would answer yes to
    # all of them at once.
    assert not hasattr(_C, "_c10d_init")
    assert not hasattr(_C, "_cuda_getDeviceCount")
    assert not hasattr(_C, "_rpc_init")
    assert _C._shim_off_switches, "the off-switch list must be inspectable"


def test_op_registry_routes_to_the_one_door():
    # `torch.ops.aten.<op>.<overload>` is built out of these three, and all
    # three have to lead to `_aten_dispatch` or DESIGN.md §6's instrument goes
    # blind.
    op, overloads = _C._jit_get_operation("aten::mm")
    assert callable(op) and overloads
    op, op_dk, tags = _C._get_operation_overload("aten::relu", "default")
    try:
        op()
    except NotImplementedError as e:
        assert "aten.relu.default" in str(e)
    else:
        raise AssertionError("an unimplemented op must name itself")
    # ... and a name that is not an op is refused, so `getattr(ns, x, None)`
    # keeps working
    assert _C._get_operation_overload("aten::__deepcopy__", "") is None


def test_parsed_schema_really_reads_the_schema():
    # A stub with an empty argument list would answer "no mutable arguments"
    # to every question, which is a wrong answer rather than a missing one.
    schema = _C.parse_schema("aten::add_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")
    assert schema.name == "aten::add_" and schema.overload_name == "Tensor"
    assert [a.name for a in schema.arguments] == ["self", "other"]
    assert schema.arguments[0].alias_info is not None
    assert schema.arguments[0].alias_info.is_write
    assert schema.arguments[1].alias_info is None
    assert schema.is_mutable()
    functional = _C.parse_schema("aten::mm(Tensor self, Tensor mat2) -> Tensor")
    assert not functional.is_mutable()
    kwonly = _C.parse_schema(
        "aten::full(SymInt[] size, Scalar fill_value, *, ScalarType? dtype=None) -> Tensor"
    )
    assert [a.kwarg_only for a in kwonly.arguments] == [False, False, True]
    assert kwonly.arguments[2].default_value == "None"


def test_finfo_and_iinfo_report_torchs_numbers():
    assert _C.finfo(_C.float32).eps == 1.1920928955078125e-07
    assert _C.finfo(_C.bfloat16).resolution == 0.01
    # torch reports float32's smallest normal for bfloat16, not bfloat16's own
    assert _C.finfo(_C.bfloat16).tiny == _C.finfo(_C.float32).tiny
    assert _C.iinfo(_C.int32).max == 2147483647
    try:
        _C.iinfo(_C.bool)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("torch refuses iinfo(bool); so must the shim")


# --- TensorBase methods (docs/TENSORBASE.md) --------------------------------
#
# These run against the bare artefact, with no vendored tree: `TensorBase` is
# the class the methods are installed on, so nothing here needs `torch.Tensor`
# to exist. That is also why every result below is a `TensorBase` and not a
# `Tensor` -- `_set_tensor_class` is called by `_initExtension`, which only a
# real `import torch` reaches.


def _t(values, shape, dtype=None):
    return _C._tensor_from_flat(values, shape, dtype)


def test_tensor_methods_reach_the_one_door():
    # Every method in the table resolves to an aten key and goes through
    # `_aten_dispatch`; none of them computes on the `TensorBase` type itself.
    assert isinstance(_C._shim_methods, dict)
    assert _C._shim_methods["__mul__"] == ["aten.mul.Tensor", "aten.mul.Scalar"]
    # `item`, `to`, `float`, `__bool__` and `__getitem__` are deliberately
    # *not* in the table -- upstream's binding for each is not a plain overload
    # set, so they are written out in `bootstrap.py` (docs/TENSORBASE.md §3).
    # They still end at `_aten_dispatch`; they just do not resolve first.
    for python_level in ("item", "to", "float", "__bool__", "__getitem__"):
        assert python_level not in _C._shim_methods
        assert callable(getattr(_C.TensorBase, python_level))
    # `sum` has no dim in its first signature, so `x.sum()` cannot bind the
    # second one -- the order is not what decides it, the arity is.
    assert _C._shim_methods["sum"] == ["aten.sum.default", "aten.sum.dim_IntList"]


def test_method_overload_resolution_picks_by_argument_type():
    x = _t([1.0, 2.0, 3.0, 4.0], [2, 2])
    y = _t([5.0, 6.0, 7.0, 8.0], [2, 2])
    assert (x * y).tolist() == [[5.0, 12.0], [21.0, 32.0]]
    # A Python scalar picks the `Scalar` overload at the *parser* level, which
    # is what this shim reproduces -- upstream's dispatcher then records
    # `mul.Tensor` one layer down (docs/TENSORBASE.md).
    assert (x * 2).tolist() == [[2.0, 4.0], [6.0, 8.0]]
    assert (x * 2.5).tolist() == [[2.5, 5.0], [7.5, 10.0]]


def test_a_scalar_does_not_widen_a_tensor_of_the_same_category():
    # torch's "wrapped number" rule, the same one `pow` follows.
    ints = _t([1.0, 2.0, 3.0, 4.0], [2, 2], _C.int64)
    assert (ints * 3).dtype == _C.int64
    assert (ints * 3.0).dtype == _C.float32
    # ...except true division, which always floats.
    assert (ints / ints).dtype == _C.float32


def test_comparisons_answer_bool_and_only_bool():
    x = _t([1.0, 2.0, 3.0, 4.0], [2, 2])
    for result in (x == 3.0, x != 3.0, x < 3.0, x == x):
        assert result.dtype == _C.bool
    assert (x < 3.0).tolist() == [[True, True], [False, False]]


def test_varargs_rule_applies_after_self_is_bound():
    # `x.view(2, 2)` has to mean `view([2, 2])`. The precondition for torch's
    # varargs int-list rule is "the signature has exactly one positional
    # argument", which is only true of `view` once `self` is out of the count.
    x = _t([1.0, 2.0, 3.0, 4.0], [4])
    assert x.view(2, 2).shape == (2, 2)
    assert x.view([2, 2]).shape == (2, 2)
    assert x.reshape(-1, 1).shape == (4, 1)


def test_a_bare_int_binds_a_sized_int_list():
    # `sum.dim_IntList(Tensor self, int[1]? dim, ...)`. Demanding a real
    # sequence would make `x.sum(0)` report "no matching overload" -- a wrong
    # answer in the shape of a right one.
    x = _t([1.0, 2.0, 3.0, 4.0], [2, 2])
    assert x.sum(0).tolist() == [4.0, 6.0]
    assert x.sum([0]).tolist() == [4.0, 6.0]
    assert x.sum(0, True).shape == (1, 2)


def test_sum_promotes_integral_inputs_and_mean_refuses_them():
    ints = _t([1.0, 2.0, 3.0, 4.0], [2, 2], _C.int32)
    assert ints.sum().dtype == _C.int64
    try:
        ints.mean()
    except RuntimeError:
        pass
    else:
        raise AssertionError("torch refuses mean() on an integral tensor")


def test_getitem_decomposes_into_aten_calls():
    x = _t([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])
    assert x[0].tolist() == [1.0, 2.0]
    assert x[-1].tolist() == [5.0, 6.0]
    assert x[1, 0].tolist() == 3.0
    assert x[:, 1].tolist() == [2.0, 4.0, 6.0]
    assert x[None].shape == (1, 3, 2)
    assert x[:, None].shape == (3, 1, 2)
    assert x[0:2].tolist() == [[1.0, 2.0], [3.0, 4.0]]
    assert x[..., 0].tolist() == [1.0, 3.0, 5.0]
    mask = _t([1.0, 0.0, 1.0], [3], _C.bool)
    assert x[mask].tolist() == [[1.0, 2.0], [5.0, 6.0]]


def test_getitem_refuses_mixing_a_tensor_with_a_slice():
    x = _t([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])
    mask = _t([1.0, 0.0, 1.0], [3], _C.bool)
    try:
        x[mask, 0:1]
    except NotImplementedError:
        pass
    else:
        raise AssertionError("mixed basic/advanced indexing is not implemented")


def test_in_place_ops_mutate_the_receiver():
    x = _t([1.0, 2.0, 3.0, 4.0], [2, 2])
    assert x.fill_(7.0) is x
    assert x.tolist() == [[7.0, 7.0], [7.0, 7.0]]
    src = _t([9.0, 9.0, 9.0, 9.0], [2, 2])
    assert x.copy_(src) is x
    assert x.tolist() == [[9.0, 9.0], [9.0, 9.0]]
    # `copy_` keeps the destination's dtype, not the source's -- torch's
    # asymmetry, measured.
    ints = _t([0.0, 0.0], [2], _C.int64)
    ints.copy_(_t([1.5, 2.5], [2]))
    assert ints.dtype == _C.int64 and ints.tolist() == [1, 2]


def test_fill_refuses_a_value_the_dtype_cannot_hold():
    # The same `checked_convert` rule `full` follows, and the same upstream
    # numel==1 hole: `fill_` *is* where that hole lives.
    small = _t([0.0, 0.0, 0.0, 0.0], [2, 2], _C.int32)
    try:
        small.fill_(2 ** 31)
    except RuntimeError:
        pass
    else:
        raise AssertionError("torch refuses an int32 fill of 2**31")


def test_item_leaves_the_tensor_world_with_torchs_own_message():
    assert _t([2.5], [1]).item() == 2.5
    assert _t([1.0], [1], _C.int64).item() == 1
    assert _t([1.0], [1], _C.bool).item() is True
    assert bool(_t([0.0], [1])) is False
    try:
        _t([1.0, 2.0], [2]).item()
    except RuntimeError as e:
        assert "cannot be converted to Scalar" in str(e)
    else:
        raise AssertionError("item() needs exactly one element")


def test_to_returns_self_when_nothing_changes():
    # Measured upstream: `f.to(torch.float32)` on a float32 tensor produces no
    # aten record at all, and `f.float()` likewise.
    x = _t([1.0, 2.0], [2])
    assert x.to(_C.float32) is x
    assert x.float() is x
    assert x.to(_C.float64).dtype == _C.float64
    assert x.long().dtype == _C.int64


def test_bitwise_is_logical_on_bool_and_bitwise_on_ints():
    # BOOL.md §3: this is one of the six guardrails that survive only because
    # `bool` is not aliased onto `uint8`.
    mask = _t([1.0, 0.0], [2], _C.bool)
    other = _t([1.0, 1.0], [2], _C.bool)
    assert (mask & other).tolist() == [True, False]
    assert (mask | other).tolist() == [True, True]
    assert (~mask).tolist() == [False, True]
    ints = _t([12.0, 10.0], [2], _C.int64)
    assert (ints & 10).tolist() == [8, 10]
    assert (~ints).tolist() == [-13, -11]


def test_masked_fill_refuses_a_non_bool_mask():
    x = _t([1.0, 2.0], [2])
    try:
        x.masked_fill(_t([1.0, 0.0], [2], _C.uint8), 0.0)
    except RuntimeError as e:
        assert "boolean masks" in str(e)
    else:
        raise AssertionError("torch refuses a uint8 mask")


def test_operator_dunders_decline_rather_than_raise():
    # `NotImplemented`, so Python can fall back. The vendored tree compares
    # tensors against strings and against None in several places.
    x = _t([1.0, 2.0], [2])
    assert (x == "cpu") is False
    assert (x == None) is False  # noqa: E711


def test_the_rng_ops_have_kernels_now():
    # This used to be `test_the_rng_ops_are_the_wall_and_they_name_themselves`,
    # asserting that both ops resolved to the right aten key and then stopped
    # (docs/TENSORBASE.md §7, wall 6). `rng.rs` is that wall coming down: the
    # assertion is inverted rather than deleted, so the file still records
    # that these two are the pair `from_config` needed.
    x = _t([0.0] * 6, [2, 3])
    assert x.uniform_(-1.0, 1.0) is x
    assert all(-1.0 <= v < 1.0 for row in x.tolist() for v in row)
    assert x.normal_(0.0, 0.02) is x
    assert "aten.uniform_.default" in _C._aten_implemented()
    assert "aten.normal_.default" in _C._aten_implemented()


def test_the_same_seed_gives_the_same_stream():
    # The point of porting torch's generator rather than using candle's: a
    # seed has to mean something. candle's CPU backend refuses `set_seed`
    # outright (docs/RNG.md §2.1), so this test is unimplementable on top of
    # it -- not merely failing, unimplementable.
    _C._shim_manual_seed(1234)
    first = _t([0.0] * 5, [5])
    first.uniform_(0.0, 1.0)
    _C._shim_manual_seed(1234)
    second = _t([0.0] * 5, [5])
    second.uniform_(0.0, 1.0)
    assert first.tolist() == second.tolist()
    assert _C._shim_initial_seed() == 1234


def test_uniform_matches_torchs_stream_bit_for_bit():
    # Values lifted from real torch 2.13.0 on this host:
    #     torch.manual_seed(0); torch.empty(4).uniform_()
    # Not a range check -- `uniform_` is integer masking and one fused
    # multiply-add, so upstream's exact bits are reproducible, and anything
    # weaker would not notice a transformation that is subtly wrong.
    _C._shim_manual_seed(0)
    x = _t([0.0] * 4, [4])
    x.uniform_()
    assert x.tolist() == [
        0.49625658988952637,
        0.7682217955589294,
        0.08847743272781372,
        0.13203048706054688,
    ]


def test_normal_takes_a_different_path_at_sixteen_elements():
    # The trap docs/RNG.md §1.3 measured and §5 item 2 asked to be cased:
    # `normal_kernel` branches on `size >= 16 && is_contiguous()`, so one seed
    # gives two unrelated sequences either side of the boundary, and a size
    # that is not a multiple of 16 redraws its last 16 elements over values it
    # already wrote. Reproducing Box-Muller correctly and missing this looks
    # like a numerical bug.
    def draw(n):
        _C._shim_manual_seed(0)
        t = _t([0.0] * n, [n])
        t.normal_()
        return t.tolist()

    fifteen, sixteen, seventeen = draw(15), draw(16), draw(17)
    # Upstream's heads, measured. Path B, then path A, then path A with the
    # tail redraw.
    assert round(fifteen[0], 4) == 1.541
    assert round(sixteen[0], 4) == -1.1258
    assert round(seventeen[0], 4) == -1.1258
    assert round(seventeen[1], 4) == -1.6959 != round(sixteen[1], 4)
    assert fifteen[:4] != sixteen[:4]


def test_normal_caches_the_other_half_of_the_pair_on_the_generator():
    # Box-Muller yields two samples; upstream returns one and keeps the other
    # *on the generator*, so an odd-sized draw changes the next call's output.
    # Dropping the cache would still give a correct Gaussian and the wrong
    # numbers.
    _C._shim_manual_seed(7)
    first = _t([0.0] * 5, [5])
    first.normal_()
    second = _t([0.0] * 5, [5])
    second.normal_()
    # If the cache were dropped between calls, the second draw would restart
    # from a fresh Box-Muller pair rather than consuming the held one.
    assert second.tolist() != first.tolist()
    _C._shim_manual_seed(7)
    again = _t([0.0] * 5, [5])
    again.normal_()
    assert again.tolist() == first.tolist()


def test_uniform_refuses_a_generator_it_does_not_own():
    # There is one generator here. A `torch.Generator()` of one's own has no
    # state, and serving it from the default stream would look like it worked.
    other = _C.Generator()
    x = _t([0.0, 0.0], [2])
    try:
        x.uniform_(0.0, 1.0, generator=other)
    except NotImplementedError as e:
        assert "torch.default_generator" in str(e)
    else:
        raise AssertionError("a foreign generator was accepted")
    # The default one is fine, named explicitly.
    assert x.uniform_(0.0, 1.0, generator=_C.default_generator) is x


def test_rng_ops_refuse_integer_tensors():
    # `AT_DISPATCH_FLOATING_TYPES_AND2(kHalf, kBFloat16, ...)` is the whole
    # dispatch set; an int tensor reaches `random_` upstream, a different op.
    x = _t([0, 0], [2], _C.int64)
    for call in (lambda: x.uniform_(), lambda: x.normal_()):
        try:
            call()
        except NotImplementedError as e:
            assert "int64" in str(e)
        else:
            raise AssertionError("an integer tensor was filled")


def test_grad_mode_state_round_trips():
    # 84 calls during `from_config` (docs/FROM_CONFIG.md §2.2). The flag is
    # real; what it would govern is not.
    assert _C.is_grad_enabled() is True
    _C._set_grad_enabled(False)
    assert _C.is_grad_enabled() is False
    _C._set_grad_enabled(True)
    assert _C.is_grad_enabled() is True
    assert _C._VariableFunctions.is_grad_enabled is _C.is_grad_enabled


def test_make_subclass_produces_the_subclass():
    # This is what `nn.Parameter` needs: `_make_subclass(cls, data, rg)` has to
    # return an instance of `cls`, or `Module.__setattr__` files the result as
    # a plain attribute and the model ends up with no parameters.
    class Param(_C.TensorBase):
        pass

    data = _t([1.0, 2.0], [2])
    made = _C.TensorBase._make_subclass(Param, data, True)
    assert type(made) is Param
    assert made.requires_grad is True
    assert made.tolist() == [1.0, 2.0]
    assert data.requires_grad is False


def test_the_dispatch_table_matches_the_two_lists():
    # An op that answers but is in neither list would be invisible to the
    # golden harness *and* to the work queue.
    listed = set(_C._aten_all_implemented())
    assert listed == set(_C._aten_implemented()) | set(
        _C._aten_implemented_awaiting_golden()
    )
    for key in sorted(listed):
        try:
            _C._aten_dispatch(key)
        except NotImplementedError as e:
            raise AssertionError(f"{key} is listed but not dispatched: {e}") from None
        except Exception:
            pass  # missing arguments -- the key itself resolved


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    print(f"\ntarget={_C._shim_target()} implemented={_C._aten_implemented()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
