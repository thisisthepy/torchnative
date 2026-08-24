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
