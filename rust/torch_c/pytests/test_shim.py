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

import math

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


# --- the device layer (docs/DEVICE_ABS.md) ----------------------------------


def test_device_label_is_validated_against_a_closed_vocabulary():
    """A label that accepts anything is not a label.

    Each refusal below is upstream's, measured on torch 2.13.0 and reproduced
    with the same exception type; before this the shim accepted all of them and
    only failed later, at `resolve()`, naming a device type nobody had asked
    for. The list matters beyond typos: it is the vocabulary
    `torch.distributed`'s backend registration keys off (DESIGN.md §11.1), so
    it has to be upstream's exact list rather than a superset.
    """
    for bad in ("nosuchdevice", "CPU", " cpu", "", "cuda:-1", "cuda:x"):
        try:
            _C.device(bad)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"torch.device({bad!r}) must be refused")

    # ... and every accepted one really is accepted, including the two a future
    # accelerator would have to arrive as.
    for good in ("cpu", "cuda", "mps", "meta", "vulkan", "xpu", "privateuseone"):
        assert _C.device(good).type == good


def test_device_accepts_every_spelling_torch_normalises_through():
    # `torch.device(x)` is idempotent upstream and the vendored tree relies on
    # it: `_parse_to`, `Module.to` and every `device=` keyword normalise by
    # calling it on whatever arrived. Refusing a `device` there breaks the
    # normalisation everything else assumes.
    assert _C.device(_C.device("cuda:1")) == _C.device("cuda", 1)
    assert _C.device(type="cuda", index=1) == _C.device("cuda:1")
    assert _C.device(device="cuda:1") == _C.device("cuda:1")

    # An index cannot arrive twice, and a bare integer has no device type to
    # attach to on a build with no accelerator (upstream reads it as an index
    # of `torch.accelerator.current_accelerator()`).
    for call in (lambda: _C.device("cuda:0", 1),
                 lambda: _C.device("cuda", -1),
                 lambda: _C.device(0)):
        try:
            call()
        except RuntimeError:
            pass
        else:
            raise AssertionError("must be refused")


def test_device_is_picklable():
    # `torch.save`/`torch.load` and every `copy.deepcopy` of a config carry
    # devices through pickle. Upstream's shape, measured:
    # `torch.device('cuda', 1).__reduce__()` is `(torch.device, ('cuda', 1))`,
    # and the index-less form drops the second element rather than passing
    # `None` -- which matters, because `device('cpu', None)` is not the same
    # call as `device('cpu')` once `index` is parsed positionally.
    #
    # The `__reduce__` *shape* is what is checked here rather than a
    # `pickle.dumps` round trip, and the reason is a property of this harness,
    # not of the code: this file imports the artefact as the top-level module
    # `_C`, so `torch._C` is not in `sys.modules` and pickle cannot resolve the
    # class by name ("it's not the same object as torch._C.device"). The round
    # trip is exercised where the name does resolve -- in the vendored-tree
    # subprocess, `test_device_road_through_the_vendored_tree`.
    for spelling, expected in (("cpu", ("cpu",)),
                               ("cpu:0", ("cpu", 0)),
                               ("cuda:1", ("cuda", 1)),
                               ("meta", ("meta",))):
        factory, args = _C.device(spelling).__reduce__()
        assert factory is _C.device, spelling
        assert args == expected, (spelling, args)
        assert factory(*args) == _C.device(spelling), spelling


def test_indexed_and_bare_labels_are_unequal_but_name_one_device():
    # Two different relations, and the shim needs both. `cpu` != `cpu:0`
    # upstream (measured, hashes differ too), yet a tensor made with either
    # reports plain `cpu` and the two interoperate. Equality is a property of
    # the label; the mixed-device check needs the other one.
    assert _C.device("cpu") != _C.device("cpu:0")
    assert hash(_C.device("cpu")) != hash(_C.device("cpu:0"))
    made = _C._aten_dispatch("aten.full.default", [2], 1.0, device="cpu:0")
    assert made.device == _C.device("cpu")


def test_tensor_reports_its_device_through_every_spelling():
    t = _C._aten_dispatch("aten.full.default", [2, 3], 1.5)
    assert t.device == _C.device("cpu")
    assert t.is_cpu
    assert not t.is_cuda
    assert not t.is_meta
    # `-1` is not an error code: it is how torch spells "this device kind is
    # not indexed" (measured: `torch.zeros(2).get_device()` is `-1`).
    assert t.get_device() == -1
    assert t.cpu() is t
    assert t.is_floating_point()
    assert not t.is_complex()


def test_to_copy_with_no_device_keeps_the_tensor_where_it_is():
    """`device=None` means "stay", not "go to the CPU".

    Unobservable while there is one device -- which is why it survived to be
    found by reading rather than by a failing test -- and wrong the day there
    are two. Pinned here as the *contract*; the only observable it has today is
    that a dtype-only `_to_copy` does not lose the device on the way through.
    """
    t = _C._aten_dispatch("aten.full.default", [3], 2.0)
    cast = _C._aten_dispatch("aten._to_copy.default", t, _C.float64)
    assert cast.dtype == _C.float64
    assert cast.device == t.device


def test_unavailable_device_fails_where_torch_fails_it():
    # The label constructs; only using it raises. That is the whole point of
    # storing a label instead of a live handle.
    cuda = _C.device("cuda")
    assert cuda.type == "cuda"
    try:
        _C._aten_dispatch("aten.full.default", [2], 1.0, device=cuda)
    except NotImplementedError as e:
        assert "cuda" in str(e)
    else:
        raise AssertionError("an unavailable device must raise at use")


def test_data_setter_replaces_the_tensor_behind_a_parameter():
    # `nn.Module._apply` ends every `.to()`/`.cpu()`/`.float()` with
    # `param.data = param_applied`. Without a setter that assignment is an
    # AttributeError, which is where the whole module-side device road died.
    t = _C._aten_dispatch("aten.full.default", [2], 1.0)
    t.requires_grad = True
    replacement = _C._aten_dispatch("aten.full.default", [3], 7.0, dtype=_C.float64)
    t.data = replacement
    assert t.shape == (3,)
    assert t.dtype == _C.float64
    assert t.tolist() == [7.0, 7.0, 7.0]
    # Upstream's `.data =` does not touch `requires_grad`, and `_apply` relies
    # on that to keep a Parameter a parameter.
    assert t.requires_grad
    try:
        t.data = 5
    except TypeError:
        pass
    else:
        raise AssertionError("Tensor.data must only accept a TensorBase")


def test_parse_to_matches_the_real_parser_not_the_dynamo_polyfill():
    """`torch._C._nn._parse_to` -- the single entrance for `nn.Module.to`.

    Every shape below was measured against torch 2.13.0's own parser. Three of
    them are places where the vendored tree's own reimplementation
    (`torch/_dynamo/polyfills/torch_c_nn.py`) and the real binding disagree:
    the polyfill takes one positional where the real one takes three, accepts
    `memory_format` positionally where the real one is keyword-only, and says
    nothing about `copy`, which the real one rejects outright.
    """
    parse = _C._nn._parse_to
    assert parse() == (None, None, False, None)
    assert parse(None) == (None, None, False, None)
    assert parse("cpu") == (_C.device("cpu"), None, False, None)
    assert parse(_C.device("cpu")) == (_C.device("cpu"), None, False, None)
    assert parse(_C.float16) == (None, _C.float16, False, None)
    assert parse("cpu", _C.float16) == (_C.device("cpu"), _C.float16, False, None)
    assert parse("cpu", _C.float16, True) == (_C.device("cpu"), _C.float16, True, None)
    assert parse(dtype=_C.float16) == (None, _C.float16, False, None)
    assert parse(device="cpu", dtype=None, non_blocking=False, memory_format=None) == (
        _C.device("cpu"), None, False, None,
    )
    # `to(other)` takes both the device and the dtype from the tensor.
    other = _C._aten_dispatch("aten.full.default", [1], 1.0, dtype=_C.float64)
    assert parse(other) == (_C.device("cpu"), _C.float64, False, None)
    # Measured: `RuntimeError: .to() does not accept copy argument`, whether it
    # arrives as the second boolean positional or by name.
    for call in (lambda: parse("cpu", _C.float32, False, True),
                 lambda: parse(_C.float32, False, True),
                 lambda: parse("cpu", copy=True)):
        try:
            call()
        except RuntimeError as e:
            assert "copy" in str(e)
        else:
            raise AssertionError("_parse_to must refuse a copy argument")


def test_shallow_copy_compatibility_answers_for_dense_tensors_only():
    # `Module._apply` decides in-place-vs-replace with this. Upstream answers
    # `True` for two dense tensors of different dtype, of different device, and
    # for a Parameter against a Tensor (all measured); the only `False` is for
    # a subclass with its own impl, which this shim cannot produce -- so
    # anything that is not a TensorBase is refused rather than guessed.
    ask = _C._VariableFunctions._has_compatible_shallow_copy_type
    a = _C._aten_dispatch("aten.full.default", [2], 1.0)
    b = _C._aten_dispatch("aten.full.default", [3], 1.0, dtype=_C.float64)
    assert ask(a, b)
    try:
        ask(a, 5)
    except NotImplementedError:
        pass
    else:
        raise AssertionError("a non-tensor must be refused, not guessed")


def test_the_two_accelerator_questions_get_two_different_answers():
    """They are read by two callers with opposite `None` handling.

    `torch.get_device_module()` does `torch._C._get_accelerator().type` with no
    guard (`torch/__init__.py:2978`), and its own docstring says the
    no-accelerator answer is the CPU device. `torch.accelerator.
    current_accelerator()` does `if (acc := torch._C._accelerator_getAccelerator())
    is not None` (`torch/accelerator/__init__.py:128`) and returns `None` when
    it fires. Answering both the same way breaks one of them.
    """
    assert _C._get_accelerator() == _C.device("cpu")
    assert _C._accelerator_getAccelerator() is None
    # Upstream returns the *string* 'cpu' here, not a device.
    assert _C._get_default_device() == "cpu"
    assert _C._mps_is_available() is False


def test_mixed_device_gate_lets_agreeing_tensors_through():
    """The passing half of the gate. The refusing half is the next test.

    `_aten_dispatch` refuses an op whose tensor arguments disagree about their
    device (`check_devices_agree` in aten.rs). This half -- agreeing tensors
    still dispatch, through a plain argument and through the `Tensor[]` a
    top-level-only scan would miss -- is the one that runs on every call.

    `_shim_same_device` is the *label*-level version of the comparison -- what a
    device-carrying tensor would need (docs/DEVICE_ABS.md §3.2). It has no other
    caller today, because the gate reads candle's handles directly: building a
    label per argument cost a measured 78 ns per dispatch. It is pinned here
    because it is deliberately *not* `==`; `cpu` and `cpu:0` are unequal labels
    that name one device.
    """
    same = _C._shim_same_device
    d = _C.device
    assert same(d("cpu"), d("cpu:0"))
    assert same(d("cpu:0"), d("cpu"))
    assert same(d("cpu"), d("cuda:1")) is False
    assert same(d("cuda:0"), d("cuda:1")) is False
    assert same(d("cuda"), d("cuda:1"))
    assert same(d("cuda:0"), d("cuda:0"))
    assert same(d("mps"), d("meta")) is False

    # The positive half of the gate does run on every dispatch: an op whose
    # tensors agree must still go through, including through the `Tensor[]`
    # argument that `cat` takes (the traversal a top-level-only scan misses).
    a = _C._aten_dispatch("aten.full.default", [2], 1.0)
    b = _C._aten_dispatch("aten.full.default", [2], 2.0)
    assert _C._aten_dispatch("aten.add.Tensor", a, b).tolist() == [3.0, 3.0]
    assert _C._aten_dispatch("aten.cat.default", [a, b]).tolist() == [1.0, 1.0, 2.0, 2.0]


def test_generator_reports_a_device():
    # Every Generator this build can make is a CPU generator, because
    # `PyDevice::resolve` refuses every other label -- so there is no second
    # value for a per-instance attribute to hold. This stops being true the day
    # a second backend lands.
    assert _C.default_generator.device == _C.device("cpu")
    assert _C.Generator().device == _C.device("cpu")


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
    # `embedding` stood here, then `relu`, and both gained kernels -- which is
    # the right failure mode for this test: it goes red when the op it samples
    # stops being a sample. Both were plausible sample choices *because* they
    # were plausible implementation targets, which is exactly what makes them
    # bad ones. `ormqr` multiplies by a Householder product from a LAPACK
    # factorisation; nothing in the architecture tails of docs/ARCH.md or
    # docs/OPS4.md reaches for it, and on-device inference has no reason to.
    try:
        _C._aten_dispatch("aten.ormqr.default")
    except NotImplementedError as e:
        assert str(e) == "aten op not implemented in torch._C shim: aten.ormqr.default"
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
    # `.default`. `flatten` has no kernel (docs/SPELLINGS.md) and so no table
    # entry either -- `relu` used to be this example, until docs/SPELLINGS.md
    # §6 gave it one.
    try:
        _vf("flatten")(1)
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
    assert not hasattr(_C, "_cuda_getDeviceCount")
    assert not hasattr(_C, "_rpc_init")
    assert _C._shim_off_switches, "the off-switch list must be inspectable"

    # `_c10d_init` used to be on this list and is now answered -- see
    # `ANSWERED_PROBES` in bootstrap.py and docs/DISTRIBUTED.md. The switch is
    # inverted rather than removed: a subsystem that is built has to say so,
    # and the list must not still be claiming it is off.
    assert hasattr(_C, "_c10d_init")
    assert _C._c10d_init() is True
    assert "_c10d_init" not in _C._shim_off_switches


def test_op_registry_routes_to_the_one_door():
    # `torch.ops.aten.<op>.<overload>` is built out of these three, and all
    # three have to lead to `_aten_dispatch` or DESIGN.md §6's instrument goes
    # blind.
    op, overloads = _C._jit_get_operation("aten::mm")
    assert callable(op) and overloads
    # `ormqr` for the same reason as test_unimplemented_op_names_itself above.
    op, op_dk, tags = _C._get_operation_overload("aten::ormqr", "default")
    try:
        op()
    except NotImplementedError as e:
        assert "aten.ormqr.default" in str(e)
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
    # A property, as upstream's is -- see
    # `test_schema_is_mutable_is_a_property_not_a_call` for why that matters.
    assert schema.is_mutable
    functional = _C.parse_schema("aten::mm(Tensor self, Tensor mat2) -> Tensor")
    assert not functional.is_mutable
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


# --- against real upstream torch, live in the same process (docs/E2E.md) ---
#
# Every measurement below was a one-off until now: it lived in a probe script
# under a `caches/` directory that is not committed and evaporates with the
# next `git worktree` cleanup, so nobody would notice if a future change to
# `aten.rs` or `rng.rs` broke the property "the shim's tokens match upstream
# torch's tokens". These tests pin that property into the suite that runs on
# every change.
#
# The approach: `import torch` here is the *real* upstream package (nothing
# on `sys.path` shadows it -- this file never adds `vendor/` to the path),
# and `_C` (imported at the top of this file) is the shim, loaded standalone
# as a module literally named `_C`, never as `torch._C`. The two do not
# collide in one process -- confirmed by running the model-comparison probes
# this section is built from directly, both here and previously as throwaway
# scripts under `caches/bw-sample-probe/`. docs/E2E.md records that check and
# why it is not the two-process split this file's callers expected going in.
#
# This makes the tests below option (b) from that discussion -- live against
# upstream, not frozen constants -- but without the subprocess: same process,
# so no serialization boundary and no per-test interpreter startup cost. The
# honest cost is the same as (b) would have had: upstream `torch` has to be
# importable. It usually is not (this file's own docstring promises no test
# dependency on a package that has none), so every test in this section
# no-ops rather than fails when `torch` is missing, and docs/E2E.md says so.
try:
    import torch as _upstream_torch
except ImportError:  # pragma: no cover - most interpreters running this file
    _upstream_torch = None


def _e2e_det(n, seed):
    """A linear congruential generator, not a real init scheme -- the point
    is that both backends receive bit-identical weights without sharing an
    RNG, not that the numbers are good ones to train with."""
    out, state = [], seed
    for _ in range(n):
        state = (state * 1103515245 + 12345) % 2147483648
        out.append(round(((state / 2147483648.0) * 2.0 - 1.0) * 0.2, 6))
    return out


_E2E_H, _E2E_HEADS, _E2E_INTER, _E2E_VOCAB, _E2E_LAYERS = 64, 2, 128, 100, 2
_E2E_HD = _E2E_H // _E2E_HEADS


def _e2e_weights():
    H, INTER, VOCAB, LAYERS = _E2E_H, _E2E_INTER, _E2E_VOCAB, _E2E_LAYERS
    w = {}
    s = 1
    for name, n in [("embed", VOCAB * H), ("lm_head", VOCAB * H), ("final_norm", H)]:
        w[name] = _e2e_det(n, s)
        s += 1
    for layer in range(LAYERS):
        for name, n in [
            ("q", H * H), ("k", H * H), ("v", H * H), ("o", H * H),
            ("gate", INTER * H), ("up", INTER * H), ("down", H * INTER),
            ("in_norm", H), ("post_norm", H),
        ]:
            w[f"{layer}.{name}"] = _e2e_det(n, s)
            s += 1
    # RMSNorm weights start near 1.0, not near 0.0 -- a norm layer initialized
    # like a linear layer zeroes its own output.
    norm_keys = ("final_norm",) + tuple(
        f"{layer}.{n}" for layer in range(LAYERS) for n in ("in_norm", "post_norm")
    )
    for k in norm_keys:
        w[k] = [1.0 + v for v in w[k]]
    return w


class _E2EBackend:
    """Runs the same op sequence against either the shim (`_C`) or real
    upstream `torch.ops.aten`, so the model-building code below is written
    once and executed through both."""

    def __init__(self, kind):
        assert kind in ("shim", "upstream")
        self.kind = kind

    def t(self, flat, shape, dtype="float32"):
        if self.kind == "upstream":
            dt = getattr(_upstream_torch, dtype)
            return _upstream_torch.tensor(list(flat), dtype=dt).reshape(list(shape))
        return _C._tensor_from_flat(list(flat), list(shape), dtype=getattr(_C, dtype))

    def op(self, name, *args, **kw):
        if self.kind == "upstream":
            ns, o, ov = name.split(".")
            return getattr(getattr(getattr(_upstream_torch.ops, ns), o), ov)(*args, **kw)
        return _C._aten_dispatch(name, *args, **kw)

    def seed(self, s):
        if self.kind == "upstream":
            _upstream_torch.manual_seed(s)
        else:
            _C._shim_manual_seed(s)


def _e2e_build(b, w):
    H = _E2E_H
    return {
        k: b.t(
            v,
            (len(v) // H, H) if k in ("embed", "lm_head") else
            (_E2E_INTER, H) if k.endswith(("gate", "up")) else
            (H, _E2E_INTER) if k.endswith("down") else
            (H, H) if k.split(".")[-1] in ("q", "k", "v", "o") else (H,),
        )
        for k, v in w.items()
    }


def _e2e_rms_norm(b, x, w, eps=1e-6):
    sq = b.op("aten.mul.Tensor", x, x)
    mean = b.op("aten.mean.dim", sq, [-1], True)
    var = b.op("aten.add.Tensor", mean, b.t([eps], ()))
    inv = b.op("aten.rsqrt.default", var)
    return b.op("aten.mul.Tensor", b.op("aten.mul.Tensor", x, inv), w)


def _e2e_linear(b, x, w):
    """`nn.Linear` without bias: `x @ w.t()`, decomposed the way
    docs/NN_SURFACE.md §5 measured upstream actually calling it."""
    dims = list(x.shape)
    flat = b.op("aten.view.default", x, [-1, dims[-1]])
    out = b.op("aten.mm.default", flat, b.op("aten.t.default", w))
    return b.op("aten.view.default", out, dims[:-1] + [int(out.shape[-1])])


def _e2e_rope_tables(b, seq, hd, base=10000.0):
    cos, sin = [], []
    for p in range(seq):
        row_c, row_s = [], []
        for i in range(hd // 2):
            inv = 1.0 / (base ** (2 * i / hd))
            row_c.append(math.cos(p * inv))
            row_s.append(math.sin(p * inv))
        cos.append(row_c + row_c)
        sin.append(row_s + row_s)
    flat_c = [v for r in cos for v in r]
    flat_s = [v for r in sin for v in r]
    return b.t(flat_c, (1, 1, seq, hd)), b.t(flat_s, (1, 1, seq, hd))


def _e2e_rotate_half(b, x):
    hd = int(x.shape[-1])
    x1 = b.op("aten.slice.Tensor", x, 3, 0, hd // 2, 1)
    x2 = b.op("aten.slice.Tensor", x, 3, hd // 2, hd, 1)
    return b.op("aten.cat.default", [b.op("aten.neg.default", x2), x1], 3)


def _e2e_apply_rope(b, x, cos, sin):
    return b.op(
        "aten.add.Tensor",
        b.op("aten.mul.Tensor", x, cos),
        b.op("aten.mul.Tensor", _e2e_rotate_half(b, x), sin),
    )


def _e2e_forward(b, w, ids, seq):
    """A 2-layer Llama-shaped decoder -- RMSNorm, RoPE, flash `sdpa`, SwiGLU
    MLP -- op-for-op the sequence transformers' `LlamaModel` produces
    (docs/NN_SURFACE.md §5-6)."""
    H, HD, HEADS = _E2E_H, _E2E_HD, _E2E_HEADS
    idx = b.t(ids, (len(ids),), "int64")
    h = b.op("aten.embedding.default", w["embed"], idx)
    h = b.op("aten.view.default", h, [1, seq, H])
    cos, sin = _e2e_rope_tables(b, seq, HD)
    for layer in range(_E2E_LAYERS):
        resid = h
        x = _e2e_rms_norm(b, h, w[f"{layer}.in_norm"])
        q = _e2e_linear(b, x, w[f"{layer}.q"])
        k = _e2e_linear(b, x, w[f"{layer}.k"])
        v = _e2e_linear(b, x, w[f"{layer}.v"])

        def heads(t):
            t = b.op("aten.view.default", t, [1, seq, HEADS, HD])
            return b.op("aten.transpose.int", t, 1, 2)

        q, k, v = heads(q), heads(k), heads(v)
        q = _e2e_apply_rope(b, q, cos, sin)
        k = _e2e_apply_rope(b, k, cos, sin)
        att = b.op(
            "aten._scaled_dot_product_flash_attention_for_cpu.default", q, k, v, 0.0, True
        )[0]
        att = b.op("aten.transpose.int", att, 1, 2)
        contig = b.op("aten.contiguous.default", att) if b.kind == "shim" else att.contiguous()
        att = b.op("aten._unsafe_view.default", contig, [1, seq, H])
        h = b.op("aten.add.Tensor", resid, _e2e_linear(b, att, w[f"{layer}.o"]))
        resid = h
        x = _e2e_rms_norm(b, h, w[f"{layer}.post_norm"])
        gate = b.op("aten.silu.default", _e2e_linear(b, x, w[f"{layer}.gate"]))
        up = _e2e_linear(b, x, w[f"{layer}.up"])
        h = b.op(
            "aten.add.Tensor",
            resid,
            _e2e_linear(b, b.op("aten.mul.Tensor", gate, up), w[f"{layer}.down"]),
        )
    h = _e2e_rms_norm(b, h, w["final_norm"])
    return _e2e_linear(b, h, w["lm_head"])


def _e2e_flatten(x):
    if isinstance(x, list):
        out = []
        for v in x:
            out.extend(_e2e_flatten(v))
        return out
    return [x]


# A token match is necessary, not sufficient: docs/ARCH.md §5.1 measured a
# case (Gemma, wrong `gelu` approximation) where greedy decoding produced the
# *same* tokens at three weight scales while the logits behind them differed
# by 5.87e-04 -- 379x the 1.55e-06 the correct kernel gave on the same model.
# A token-only suite would have shipped that bug green. So every model-forward
# comparison below checks logits too, not just the tokens they produced.
#
# `_E2E_LOGIT_ATOL` is the line between "normal float32 rounding" and "wrong
# math", picked from measurements, not invented:
#
#   this file's own aten-level 2-layer decoder ................ up to 5.2e-06
#     (greedy, 4 steps: ~2.3e-06; do_sample, 6 steps x 12 configs: ~5.2e-06)
#   torch.nn-assembled 2-layer decoder (docs/NN_SURFACE.md §7) ... rel 5.8e-07
#   aten-level 2-layer Llama (docs/SAMPLING.md §3) ................ 2.3e-09
#   GPT-2, 2 layers (docs/GPT2.md) ................................. 4.1e-08
#   Gemma, 2 layers (docs/ARCH.md §5) .............................. 1.55e-06
#   BERT, 2 layers (docs/ARCH.md §5): hidden 1.43e-06, pooled ...... 9.39e-07
#   -------------------------------------------------------------------------
#   Gemma with the wrong gelu approximation (docs/ARCH.md §5.1) .... 5.87e-04
#
# Every normal measurement across five different architectures and two
# measurement methods (this file's own live comparison, and the aten-level
# transcriptions docs/ARCH.md and friends built independently) lands at or
# below 5.2e-06. The one case known to be *wrong* lands at 5.87e-04 -- roughly
# 113x the worst normal figure. `_E2E_LOGIT_ATOL = 1e-5` sits in that gap: the
# golden harness's own float32 bound (tools/golden/dtypes.py
# `TOLERANCES["float32"]`, atol=rtol=1e-5), not a number invented for this
# file, and close to the geometric mean of the two clusters
# (sqrt(5.2e-06 * 5.87e-04) ~= 5.5e-05, same order of magnitude). It leaves
# ~2x margin above the worst normal figure measured in this file and ~59x
# margin below the one documented wrong-math figure -- see the do_sample test
# below for a direct check that this bound actually catches an error of that
# size, and docs/E2E.md for how it was confirmed.
_E2E_LOGIT_ATOL = 1e-5


def test_two_layer_llama_greedy_matches_upstream_token_for_token():
    # Regression-pins the claim measured in docs/NN_SURFACE.md §7 and
    # docs/SAMPLING.md §3's aten-level model: an aten-level 2-layer decoder,
    # decoded greedily, produces the exact same tokens as real torch.
    if _upstream_torch is None:
        return  # no upstream torch in this interpreter -- see docs/E2E.md
    ids, steps = [7, 42, 3, 88], 4
    results = {}
    for kind in ("upstream", "shim"):
        b = _E2EBackend(kind)
        w = _e2e_build(b, _e2e_weights())
        out = list(ids)
        last_logits = None
        for _ in range(steps):
            logits = _e2e_forward(b, w, out, len(out))
            last = b.op("aten.slice.Tensor", logits, 1, len(out) - 1, len(out), 1)
            nxt = b.op("aten.argmax.default", last, -1, False)
            v = nxt.tolist()
            while isinstance(v, list):
                v = v[0]
            out.append(int(v))
            last_logits = logits
        results[kind] = (out, _e2e_flatten(last_logits.tolist()))
    (t_out, t_last), (c_out, c_last) = results["upstream"], results["shim"]
    assert t_out == c_out, (t_out, c_out)
    # Measured max gap here today: ~2.3e-06 -- float32 rounding from 2 layers
    # x ~12 matmuls each, well inside _E2E_LOGIT_ATOL. See the comment above
    # this test for where the bound comes from and how wide the margin is on
    # both sides.
    max_diff = max(abs(a - b_) for a, b_ in zip(t_last, c_last))
    assert max_diff < _E2E_LOGIT_ATOL, max_diff


_E2E_FILTER = float("-inf")


def _e2e_top_k_warp(b, scores, top_k):
    """`transformers.TopKLogitsWarper`, transcribed op for op."""
    kth = b.op("aten.topk.default", scores, top_k, -1, True, True)[0]
    kth = b.op("aten.slice.Tensor", kth, 1, top_k - 1, top_k, 1)
    remove = b.op("aten.lt.Tensor", scores, kth)
    return b.op("aten.masked_fill.Scalar", scores, remove, _E2E_FILTER)


def _e2e_top_p_warp(b, scores, top_p):
    """`transformers.TopPLogitsWarper`, transcribed op for op."""
    sorted_logits, sorted_indices = b.op("aten.sort.default", scores, -1, False)
    probs = b.op("aten._softmax.default", sorted_logits, -1, False)
    cum = b.op("aten.cumsum.default", probs, -1)
    remove = b.op("aten.le.Scalar", cum, 1.0 - top_p)
    n = int(scores.shape[-1])
    rows = int(scores.shape[0])
    keep_idx = b.t([n - 1] * rows, (rows, 1), "int64")
    false_src = b.op("aten.le.Scalar", b.t([1.0] * rows, (rows, 1)), 0.0)
    remove = b.op("aten.scatter.src", remove, 1, keep_idx, false_src)
    remove = b.op("aten.scatter.src", remove, 1, sorted_indices, remove)
    return b.op("aten.masked_fill.Scalar", scores, remove, _E2E_FILTER)


def _e2e_sample_step(b, logits, seq, temperature, top_k, top_p, seed, vocab):
    last = b.op("aten.slice.Tensor", logits, 1, seq - 1, seq, 1)
    last = b.op("aten.view.default", last, [1, vocab])
    scores = b.op("aten.div.Tensor", last, b.t([temperature], ()))
    scores = _e2e_top_k_warp(b, scores, top_k)
    scores = _e2e_top_p_warp(b, scores, top_p)
    probs = b.op("aten._softmax.default", scores, -1, False)
    if seed is not None:
        b.seed(seed)
    tok = b.op("aten.multinomial.default", probs, 1, False)
    return b.op("aten.squeeze.dim", tok, 1)


def _e2e_generate(kind, ids, steps, temperature, top_k, top_p, seed, reseed_each_step):
    """Returns `(tokens, raw_logits)`, where `raw_logits[step]` is the flat
    last-position logit vector *before* temperature/top-k/top-p (the same
    quantity docs/ARCH.md §5.1 showed a token-only check cannot tell apart
    from a wrong kernel). Callers that only need tokens can ignore the second
    element; `test_do_sample_matches_upstream_across_configs_and_reseed_modes`
    below uses both."""
    b = _E2EBackend(kind)
    w = _e2e_build(b, _e2e_weights())
    out = list(ids)
    raw_logits = []
    if not reseed_each_step:
        b.seed(seed)
    for step in range(steps):
        logits = _e2e_forward(b, w, out, len(out))
        last = b.op("aten.slice.Tensor", logits, 1, len(out) - 1, len(out), 1)
        raw_logits.append(_e2e_flatten(last.tolist()))
        tok = _e2e_sample_step(
            b, logits, len(out), temperature, top_k, top_p,
            seed + step if reseed_each_step else None, _E2E_VOCAB,
        )
        out.append(int(tok.tolist()[0]))
    return out, raw_logits


def _e2e_max_logit_diff(t_logits, c_logits):
    return max(
        abs(a - b_)
        for t_step, c_step in zip(t_logits, c_logits)
        for a, b_ in zip(t_step, c_step)
    )


def test_do_sample_matches_upstream_across_configs_and_reseed_modes():
    # Regression-pins docs/SAMPLING.md §3's judgment call: 15 configurations x
    # 6 greedy-sampled tokens = 90 tokens, all matching upstream. Nine configs
    # reseed before every draw (same starting point -> same value); six seed
    # once and let the stream run across all 6 steps, which is the stronger
    # claim -- it only matches if both sides consume the same *number* of
    # random words per step, not merely the same value from a fresh seed.
    #
    # Tokens are necessary but not sufficient (see the comment above
    # `_E2E_LOGIT_ATOL`), so every configuration below also compares the raw
    # logits `_e2e_generate` now returns alongside the tokens.
    if _upstream_torch is None:
        return  # no upstream torch in this interpreter -- see docs/E2E.md
    ids, steps = [7, 42, 3, 88], 6
    checked = 0
    max_diff = 0.0
    for temperature, top_k, top_p in ((1.0, 50, 0.95), (0.7, 20, 0.9), (1.3, 100, 1.0)):
        for seed in (0, 1, 1234):
            t_out, t_logits = _e2e_generate("upstream", ids, steps, temperature, top_k, top_p, seed, True)
            c_out, c_logits = _e2e_generate("shim", ids, steps, temperature, top_k, top_p, seed, True)
            assert t_out == c_out, ("reseed", temperature, top_k, top_p, seed, t_out, c_out)
            diff = _e2e_max_logit_diff(t_logits, c_logits)
            assert diff < _E2E_LOGIT_ATOL, ("reseed", temperature, top_k, top_p, seed, diff)
            max_diff = max(max_diff, diff)
            checked += steps
    for temperature, top_k, top_p in ((1.0, 50, 0.95), (0.7, 20, 0.9)):
        for seed in (0, 1, 1234):
            t_out, t_logits = _e2e_generate("upstream", ids, steps, temperature, top_k, top_p, seed, False)
            c_out, c_logits = _e2e_generate("shim", ids, steps, temperature, top_k, top_p, seed, False)
            assert t_out == c_out, ("running", temperature, top_k, top_p, seed, t_out, c_out)
            diff = _e2e_max_logit_diff(t_logits, c_logits)
            assert diff < _E2E_LOGIT_ATOL, ("running", temperature, top_k, top_p, seed, diff)
            max_diff = max(max_diff, diff)
            checked += steps
    assert checked == 90, checked
    # Measured max gap across all 15 configurations today: ~5.2e-06, inside
    # _E2E_LOGIT_ATOL with room to spare -- see the comment above that
    # constant for the full range and where the bound comes from.
    assert max_diff < _E2E_LOGIT_ATOL, max_diff


def test_multinomial_matches_upstream_through_a_second_draw():
    # docs/SAMPLING.md §2: `multinomial` takes one of two algorithms
    # (Gumbel-style argmax/topk, or cumsum + binary search) depending on
    # `!replacement or n_sample == 1`, and the fast path consumes a fixed word
    # count regardless of `n_sample`/`replacement`. A single draw matching by
    # coincidence would not show that; a *second* draw right after, with no
    # reseed in between, only matches if the first draw consumed exactly the
    # same number of generator words on both sides.
    if _upstream_torch is None:
        return  # no upstream torch in this interpreter -- see docs/E2E.md
    cases = [(5, 1, False), (5, 3, True), (8, 1, True), (100, 1, False), (100, 3, True)]
    for n_cat, n_sample, replacement in cases:
        flat = [round(((i * 2654435761 + 1) % 1000) / 1000.0 + 0.01, 6) for i in range(n_cat)]
        for seed in (0, 1, 1234):
            t_probs = _upstream_torch.tensor(flat, dtype=_upstream_torch.float32)
            c_probs = _C._tensor_from_flat(flat, [n_cat], dtype=_C.float32)
            _upstream_torch.manual_seed(seed)
            t1 = _upstream_torch.ops.aten.multinomial.default(t_probs, n_sample, replacement)
            _C._shim_manual_seed(seed)
            c1 = _C._aten_dispatch("aten.multinomial.default", c_probs, n_sample, replacement)
            t2 = _upstream_torch.ops.aten.multinomial.default(t_probs, n_sample, replacement)
            c2 = _C._aten_dispatch("aten.multinomial.default", c_probs, n_sample, replacement)
            assert t1.tolist() == c1.tolist(), (n_cat, n_sample, replacement, seed, "draw1")
            assert t2.tolist() == c2.tolist(), (n_cat, n_sample, replacement, seed, "draw2")


# --- checkpoint round trip: torch.load / safetensors read what upstream
# wrote, and the `filled` guard refuses to fabricate zeros (docs/CKPT.md) ----
#
# docs/CKPT.md measured this by hand under /Volumes/macMini/caches/ckpt-probe/
# (not committed, evaporates with the worktree) and said so itself: "이 중
# 아무것도 회귀로부터 보호되지 않는다." This section pins those measurements.
#
# Unlike the `_E2EBackend` section above, this cannot use the "one process,
# two names" trick (`_C` standalone + `import torch` for upstream) --
# `torch.load`/`nn.Module`/`state_dict` live in *pure-Python* torch
# (`torch/serialization.py`, `torch/nn/modules.py`, ...), and reaching them
# through the shim means importing the *vendored* `torch` package, which has
# `torch/_C.abi3.so` replaced by the shim (vendor/install_shim.sh). That
# package is named `torch`, same as upstream -- the two cannot both be the
# `torch` module in one interpreter. It would also mean `dlopen`-ing the
# shim's native library a second time from a second path in a process that
# already loaded it once as standalone `_C`, which nothing here has measured
# as safe. So this uses a *second interpreter*, exactly the two-script recipe
# docs/CKPT.md §7 already validated (`make_ckpt.py` then `verify.py`): a
# subprocess with `torchnative/src/main` on `PYTHONPATH` gets the vendored
# `torch` (shim-backed); this process keeps plain upstream `torch`
# (`_upstream_torch` above, already guarded for its absence).
#
# The subprocess needs `torchnative/src/main/torch/_C.abi3.so` to exist,
# which `vendor/install_shim.sh` places there -- a *different* build step
# than the one `pytests/run.sh` runs for the standalone `_C` every other test
# in this file uses. So the guard below checks for that file too, not just
# `_upstream_torch is None`, and skips the same way (silently, no pytest.skip
# -- see docs/E2E.md's reasoning, which applies here unchanged) when it is
# missing.
import functools
import json
import os
import subprocess
import sys
import tempfile

_CKPT_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_CKPT_VENDOR_DIR = os.path.join(_CKPT_REPO_ROOT, "torchnative", "src", "main")
_CKPT_VENDOR_SHIM = os.path.join(_CKPT_VENDOR_DIR, "torch", "_C.abi3.so")

_CKPT_V, _CKPT_H, _CKPT_F = 32, 16, 32  # vocab, hidden, ffn -- same shape as caches/ckpt-probe/make_ckpt.py's Tiny
_CKPT_IDS = [3, 7, 1, 19]


def _ckpt_det(n, seed, lo=-1000, hi=1000):
    """Deterministic, RNG-free weights -- same generator as
    caches/ckpt-probe/make_ckpt.py's and make_hard.py's `det()`, and the same
    idea as `_e2e_det` above: checkpoint content must not depend on which
    RNG (upstream's or the shim's) happens to run."""
    return [((seed * 1103515245 + i * 12345) % (hi - lo) + lo) / 4000.0 for i in range(n)]


def _ckpt_shim_available():
    return _upstream_torch is not None and os.path.isfile(_CKPT_VENDOR_SHIM)


# Runs in a *fresh* interpreter, with the vendored (shim-backed) `torch` on
# PYTHONPATH. Reads the checkpoint this file's process (upstream) wrote,
# forward-passes it through the same architecture, and reports everything as
# one JSON object on stdout -- one subprocess launch buys every check below,
# instead of one launch per property (each launch pays a real `import torch`
# + `import safetensors`, not free -- see the wall-time note at the bottom of
# this section).
_CKPT_SHIM_SCRIPT = r"""
import json, os, struct, sys
import torch
import torch.nn as nn

cfg = json.load(sys.stdin)
d = cfg["dir"]
V, H, F = cfg["vocab"], cfg["hidden"], cfg["ffn"]
ids_list, ref_logits, ref_keys, hard_meta = cfg["ids"], cfg["logits"], cfg["keys"], cfg["hard_meta"]
result = {}


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(V, H)
        self.norm = nn.LayerNorm(H)
        self.gate = nn.Linear(H, F, bias=False)
        self.up = nn.Linear(H, F, bias=False)
        self.down = nn.Linear(F, H, bias=True)
        self.final_norm = nn.LayerNorm(H)
        self.lm_head = nn.Linear(H, V, bias=False)

    def forward(self, ids):
        h = self.embed(ids)
        x = self.norm(h)
        h = h + self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))
        return self.lm_head(self.final_norm(h))


def logit_diff(sd):
    m = Tiny()
    ids = torch.tensor(ids_list, dtype=torch.int64)
    before = max(abs(a - b) for a, b in zip(m(ids).reshape(-1).tolist(), ref_logits))
    m.load_state_dict(sd)
    after = max(abs(a - b) for a, b in zip(m(ids).reshape(-1).tolist(), ref_logits))
    return before, after


sd_true = torch.load(os.path.join(d, "tiny.pt"), weights_only=True)
result["keys_match"] = sorted(sd_true) == sorted(ref_keys)
result["shapes_dtypes_match"] = all(
    list(sd_true[k].shape) == ref_keys[k][0] and str(sd_true[k].dtype) == ref_keys[k][1] for k in ref_keys
)
b, a = logit_diff(sd_true)
result["random_init_diff"] = b
result["zip_weights_only_true_diff"] = a

sd_false = torch.load(os.path.join(d, "tiny.pt"), weights_only=False)
_, a2 = logit_diff(sd_false)
result["zip_weights_only_false_diff"] = a2

from safetensors.torch import load as sload, load_file

raw = open(os.path.join(d, "tiny.safetensors"), "rb").read()
sd_st = sload(raw)
_, a3 = logit_diff(sd_st)
result["safetensors_bytes_diff"] = a3

sd_pr = load_file(os.path.join(d, "tiny.safetensors"), backend="pread")
_, a4 = logit_diff(sd_pr)
result["safetensors_pread_diff"] = a4

result["reader_agreement_worst"] = max(
    max(abs(x - y) for x, y in zip(sd_true[k].reshape(-1).tolist(), sd_st[k].reshape(-1).tolist()))
    for k in sd_true
)

# Negative control: a checkpoint that was NOT perturbed would be worthless
# evidence -- this shows a one-float bump actually moves the logits.
hlen = struct.unpack("<Q", raw[:8])[0]
header = json.loads(raw[8 : 8 + hlen])
name, info = next((k, v) for k, v in header.items() if k != "__metadata__" and v["data_offsets"][0] == 0)
off = 8 + hlen + info["data_offsets"][0]
orig = struct.unpack("<f", raw[off : off + 4])[0]
bumped = bytearray(raw)
bumped[off : off + 4] = struct.pack("<f", orig + 1.0)
_, ap = logit_diff(sload(bytes(bumped)))
result["negative_control_diff"] = ap

try:
    torch.load(os.path.join(d, "tiny_legacy.pt"), weights_only=True)
    result["legacy_refused"] = False
except NotImplementedError as e:
    result["legacy_refused"] = True
    result["legacy_error"] = str(e)

# safetensors' *default* backend. docs/CKPT.md §3.1 recorded this refusing --
# it goes through `UntypedStorage.from_file`, which did not exist -- and
# docs/CKPT2.md §2 is the round that implemented it. What is recorded now is
# not "it works" but "it agrees": a third reader of the same file, reaching the
# bytes by a third route (whole-file storage, sliced, `asarray`, `view.dtype`),
# has to land on the same numbers as the two docs/CKPT.md §1 already compared.
try:
    sd_mm = load_file(os.path.join(d, "tiny.safetensors"), backend="mmap")
    result["mmap_backend"] = "OK"
    result["mmap_vs_pread_worst"] = max(
        max(abs(x - y) for x, y in zip(
            sd_st[k].reshape(-1).tolist(), sd_mm[k].reshape(-1).tolist()))
        for k in sd_st
    )
    _, result["mmap_logit_diff"] = logit_diff(sd_mm)
except BaseException as e:
    result["mmap_backend"] = "%s: %s" % (type(e).__name__, str(e)[:200])

s = torch.UntypedStorage(16)
t = torch.empty((0,), dtype=torch.float32)
try:
    t.set_(s, 0, [4], [1])
    result["unfilled_refused"] = False
except NotImplementedError as e:
    result["unfilled_refused"] = True
    result["unfilled_error"] = str(e)

s._shim_fill(struct.pack("<4f", 0.0, 1.0, 2.0, 3.0))
t.set_(s, 0, [2, 2], [2, 1])
result["contiguous_read"] = t.reshape(-1).tolist()

t2 = torch.empty((0,), dtype=torch.float32)
t2.set_(s, 0, [2, 2], [1, 2])
result["transposed_read"] = t2.reshape(-1).tolist()

t3 = torch.empty((0,), dtype=torch.float32)
t3.set_(s, 2, [2], [1])
result["offset_read"] = t3.reshape(-1).tolist()

t4 = torch.empty((0,), dtype=torch.float32)
try:
    t4.set_(s, 0, [9], [1])
    result["past_end_refused"] = False
except RuntimeError as e:
    result["past_end_refused"] = True
    result["past_end_error"] = str(e)

t5 = torch.empty((0,), dtype=torch.float32)
try:
    t5.set_(s, 0, [2, 2], [-1, 1])
    result["negative_stride_refused"] = False
except NotImplementedError as e:
    result["negative_stride_refused"] = True
    result["negative_stride_error"] = str(e)

sd_hard = torch.load(os.path.join(d, "hard.pt"), weights_only=True)
hard_results = {}
for key, want in hard_meta.items():
    got = sd_hard[key]
    vals = got.float().reshape(-1).tolist()
    worst_k = max((abs(x - y) for x, y in zip(vals, want["values"])), default=0.0)
    hard_results[key] = {
        "dtype_ok": str(got.dtype) == want["dtype"],
        "shape_ok": list(got.shape) == want["shape"],
        "worst": worst_k,
    }
result["hard"] = hard_results
result["tied_equal"] = sd_hard["tied_a"].reshape(-1).tolist() == sd_hard["tied_b"].reshape(-1).tolist()

sys.stdout.write(json.dumps(result))
"""


@functools.lru_cache(maxsize=None)
def _ckpt_fixture():
    """Builds a checkpoint with upstream torch (in this process), then reads
    it back with the shim (in a subprocess -- see the section comment above)
    and runs the identical forward pass on both sides. `lru_cache` computes
    this once no matter how many of the five `test_ckpt_*` functions below
    call it -- if it raises, `lru_cache` does not cache the exception, so
    each caller re-runs it and independently fails (rather than one caller
    absorbing the error and the rest silently reporting nothing)."""
    torch = _upstream_torch
    nn = torch.nn

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(_CKPT_V, _CKPT_H)
            self.norm = nn.LayerNorm(_CKPT_H)
            self.gate = nn.Linear(_CKPT_H, _CKPT_F, bias=False)
            self.up = nn.Linear(_CKPT_H, _CKPT_F, bias=False)
            self.down = nn.Linear(_CKPT_F, _CKPT_H, bias=True)
            self.final_norm = nn.LayerNorm(_CKPT_H)
            self.lm_head = nn.Linear(_CKPT_H, _CKPT_V, bias=False)

        def forward(self, ids):
            h = self.embed(ids)
            x = self.norm(h)
            h = h + self.down(torch.nn.functional.silu(self.gate(x)) * self.up(x))
            return self.lm_head(self.final_norm(h))

    m = Tiny()
    with torch.no_grad():
        for i, (name, p) in enumerate(sorted(m.state_dict().items())):
            flat = _ckpt_det(p.numel(), i + 1)
            m.state_dict()[name].copy_(torch.tensor(flat, dtype=torch.float32).reshape(p.shape))
    m.eval()
    ids_t = torch.tensor(_CKPT_IDS, dtype=torch.int64)
    with torch.no_grad():
        logits = m(ids_t)
    sd = m.state_dict()

    tmpdir = tempfile.mkdtemp(prefix="ckpt-harness-")
    torch.save(sd, os.path.join(tmpdir, "tiny.pt"))
    torch.save(sd, os.path.join(tmpdir, "tiny_legacy.pt"), _use_new_zipfile_serialization=False)
    from safetensors.torch import save_file

    save_file({k: v.contiguous() for k, v in sd.items()}, os.path.join(tmpdir, "tiny.safetensors"))
    keys = {k: [list(v.shape), str(v.dtype)] for k, v in sd.items()}

    # The 14-case "hard" checkpoint: dtypes and views a real checkpoint has
    # that Tiny above does not (docs/CKPT.md §5-6).
    hard = {}
    hard["w_f32"] = torch.tensor(_ckpt_det(24, 101), dtype=torch.float32).reshape(4, 6)
    hard["w_f16"] = hard["w_f32"].half()
    hard["w_bf16"] = hard["w_f32"].bfloat16()
    hard["w_f64"] = torch.tensor(_ckpt_det(12, 104), dtype=torch.float64).reshape(3, 4)
    hard["buf_i64"] = torch.arange(10, dtype=torch.int64)
    hard["buf_i32"] = torch.arange(10, dtype=torch.int32)
    hard["buf_bool"] = torch.arange(10) % 3 == 0
    hard["scalar"] = torch.tensor(1.25)
    hard["empty"] = torch.zeros(0, dtype=torch.float32)
    hard["rank3"] = torch.tensor(_ckpt_det(24, 105), dtype=torch.float32).reshape(2, 3, 4)
    tied = torch.tensor(_ckpt_det(20, 106), dtype=torch.float32).reshape(4, 5)
    hard["tied_a"], hard["tied_b"] = tied, tied
    base = torch.tensor(_ckpt_det(12, 107), dtype=torch.float32).reshape(3, 4)
    hard["transposed"] = base.t()
    big = torch.tensor(_ckpt_det(20, 108), dtype=torch.float32)
    hard["slice_offset"] = big[5:13]
    torch.save(hard, os.path.join(tmpdir, "hard.pt"))
    hard_meta = {
        k: {"shape": list(v.shape), "dtype": str(v.dtype), "values": v.float().reshape(-1).tolist()}
        for k, v in hard.items()
    }

    cfg = {
        "dir": tmpdir,
        "vocab": _CKPT_V, "hidden": _CKPT_H, "ffn": _CKPT_F,
        "ids": _CKPT_IDS,
        "logits": logits.reshape(-1).tolist(),
        "keys": keys,
        "hard_meta": hard_meta,
    }

    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    # Not a workaround -- upstream ships this switch for builds without
    # libtorch_global_deps, which is exactly this one (docs/CKPT.md §7,
    # VENDOR.md:181).
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _CKPT_SHIM_SCRIPT],
        input=json.dumps(cfg),
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"checkpoint subprocess (shim side) exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise RuntimeError(f"checkpoint subprocess produced non-JSON stdout: {proc.stdout!r}") from e


def test_ckpt_torch_load_zip_round_trip_matches_upstream_within_measured_tolerance():
    # docs/CKPT.md §1: torch.load (zip, both weights_only=True and False)
    # measured 2.98e-08 today, inside the "normal float32 rounding" range
    # (2.3e-09~5.2e-06) that table itself states -- the same range
    # `_E2E_LOGIT_ATOL` above is grounded in, so this reuses that constant
    # rather than inventing a second one for the same evidence.
    if not _ckpt_shim_available():
        return  # no upstream torch, or vendor shim not installed -- see docs/CKPT.md
    r = _ckpt_fixture()
    assert r["keys_match"]
    assert r["shapes_dtypes_match"]
    # Negative control: a freshly initialized model must NOT already sit at
    # the target logits, or "small diff after loading" would prove nothing.
    assert r["random_init_diff"] > 1e-3, r["random_init_diff"]
    assert r["zip_weights_only_true_diff"] < _E2E_LOGIT_ATOL, r["zip_weights_only_true_diff"]
    assert r["zip_weights_only_false_diff"] < _E2E_LOGIT_ATOL, r["zip_weights_only_false_diff"]


def test_ckpt_safetensors_two_readers_agree_with_torch_load_bit_for_bit():
    # docs/CKPT.md §1, §3.1: both safetensors backends load correctly, and
    # agree with torch.load's reading of the *same weights* to exactly 0.0 --
    # not an approximation, because both go through the shared `from_le_bytes`
    # (docs/CKPT.md §6) rather than two independent dtype/bool paths.
    if not _ckpt_shim_available():
        return
    r = _ckpt_fixture()
    assert r["safetensors_bytes_diff"] < _E2E_LOGIT_ATOL, r["safetensors_bytes_diff"]
    assert r["safetensors_pread_diff"] < _E2E_LOGIT_ATOL, r["safetensors_pread_diff"]
    assert r["reader_agreement_worst"] == 0.0, r["reader_agreement_worst"]
    # Negative control on the safetensors payload itself (not just the model):
    # a one-float bump must move the logits, or "bit exact" would be checking
    # two readers that both ignore their input.
    assert r["negative_control_diff"] > 1e-3, r["negative_control_diff"]


def test_ckpt_legacy_format_is_refused_by_name_and_the_mmap_backend_agrees():
    """The two halves of this used to be one claim, and are now opposites.

    docs/CKPT.md §3.3 listed legacy `torch.load` and safetensors' default mmap
    backend together, as the two paths that refused. They were never the same
    kind of thing, and docs/CKPT2.md §2 separated them:

      * **legacy stays refused, and must.** Its container fills the storage
        *after* `set_`, and this shim's `set_` copies, so an implementation
        that did not refuse would return a state dict of `0.0` with no error
        (docs/CKPT.md §4). The refusal is the `filled` guard firing through
        `_rebuild_tensor` -> `set_`, so this also checks the guard is what is
        doing it, by message.
      * **mmap is implemented, so the assertion becomes agreement.** It reads
        the same file by a third route, and the bar is bit-for-bit equality
        with the `pread` backend -- the same `== 0.0` §1 holds the other two
        readers to. "It returned tensors" would pass on garbage.
    """
    if not _ckpt_shim_available():
        return
    r = _ckpt_fixture()
    assert r["legacy_refused"], "legacy (non-zip) torch.load did NOT refuse -- check for zeros!"
    assert "never been filled" in r["legacy_error"], r["legacy_error"]
    assert r["mmap_backend"] == "OK", r["mmap_backend"]
    assert r["mmap_vs_pread_worst"] == 0.0, r["mmap_vs_pread_worst"]
    assert r["mmap_logit_diff"] < _E2E_LOGIT_ATOL, r["mmap_logit_diff"]


def test_ckpt_filled_guard_refuses_set_on_unfilled_storage_then_gathers_strided_views():
    # docs/CKPT.md §4: the single most important safety property in this
    # work. Without it, `torch.load` of a legacy-format checkpoint succeeds,
    # `load_state_dict` reports "All keys matched successfully", and every
    # loaded weight is silently 0.0 -- no exception anywhere (docs/CKPT.md §4
    # reproduces that exact failure). `set_` on a storage that was never
    # filled must refuse by name; once filled, `set_` must gather strided
    # views correctly (docs/CKPT.md §5: contiguous, transposed, and a nonzero
    # storage_offset), and must still refuse bounds violations and negative
    # strides rather than guess.
    if not _ckpt_shim_available():
        return
    r = _ckpt_fixture()
    assert r["unfilled_refused"], "set_ on an unfilled storage did NOT refuse -- IT RETURNED ZEROS"
    assert "never been filled" in r["unfilled_error"], r["unfilled_error"]
    assert r["contiguous_read"] == [0.0, 1.0, 2.0, 3.0], r["contiguous_read"]
    assert r["transposed_read"] == [0.0, 2.0, 1.0, 3.0], r["transposed_read"]
    assert r["offset_read"] == [2.0, 3.0], r["offset_read"]
    assert r["past_end_refused"], "set_ past the end of the storage did NOT refuse"
    assert "storage" in r["past_end_error"], r["past_end_error"]
    assert r["negative_stride_refused"], "set_ with a negative stride did NOT refuse"
    assert "egative stride" in r["negative_stride_error"], r["negative_stride_error"]


def test_ckpt_fourteen_hard_dtypes_and_views_round_trip_bit_exact():
    # docs/CKPT.md §5: f16/bf16/f64/i64/i32/bool/scalar/empty/rank-3, tied
    # weights, a transposed view, and a nonzero storage_offset slice -- all
    # loaded from a checkpoint upstream wrote. docs/CKPT.md's own bar for
    # these is bit-exact ("키마다 비트 일치"), not a tolerance: float16 and
    # bfloat16 round-trip exactly through float32, so a nonzero diff here is
    # a real regression, not rounding.
    if not _ckpt_shim_available():
        return
    r = _ckpt_fixture()
    hard = r["hard"]
    expected_keys = {
        "w_f32", "w_f16", "w_bf16", "w_f64", "buf_i64", "buf_i32", "buf_bool",
        "scalar", "empty", "rank3", "tied_a", "tied_b", "transposed", "slice_offset",
    }
    assert set(hard) == expected_keys, set(hard) ^ expected_keys
    for key, got in hard.items():
        assert got["dtype_ok"], (key, "dtype mismatch")
        assert got["shape_ok"], (key, "shape mismatch")
        assert got["worst"] == 0.0, (key, got["worst"])
    # Weight tying is preserved in value (not identity -- docs/CKPT.md §5's
    # one recorded gap, which this does not re-litigate).
    assert r["tied_equal"]


# ---------------------------------------------------------------------------
# The device road, end to end through the vendored tree (docs/DEVICE_ABS.md §5)
# ---------------------------------------------------------------------------
#
# Everything above tests `_C` in isolation, which is where the pieces live but
# not where they are used. `nn.Module.to("cpu")` is four of them in a row --
# `_parse_to`, `Tensor.to`, `_has_compatible_shallow_copy_type`, `Tensor.data =`
# -- and it was dead on each of them in turn, one wall at a time. Only the
# vendored tree can prove the chain, so this runs in a subprocess with the
# shim-backed `torch` on PYTHONPATH, the same way the checkpoint tests do.

_DEVICE_ROAD_SCRIPT = r"""
import json, pickle, sys
import torch
import torch.nn as nn

out = {}
# Two Linears and no activation on purpose: `nn.ReLU`'s forward goes through
# `torch.relu`, which has no overload-table entry, and this test is about the
# device road rather than about that hole. `nn.Linear` forward is the one
# docs/DEVICE.md measured as bit-identical to upstream.
m = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
before = [id(p) for p in m.parameters()]

out["to_str_returns_self"] = m.to("cpu") is m
out["to_device_returns_self"] = m.to(torch.device("cpu")) is m
out["cpu_returns_self"] = m.cpu() is m
out["to_dtype_returns_self"] = m.to(torch.float32) is m
out["to_device_and_dtype"] = m.to("cpu", torch.float32) is m
out["float_returns_self"] = m.float() is m
# `_apply` on the set-data path keeps parameter object identity, which is what
# upstream's `param.data = ...` branch is for. Replacing them instead would
# silently break anything holding a reference (an optimizer, a weight tie).
out["param_identity_preserved"] = [id(p) for p in m.parameters()] == before
out["param_device"] = str(next(m.parameters()).device)

# A real cast has to actually cast, or "returns self" would be satisfied by
# doing nothing at all.
m.double()
out["double_dtype"] = str(next(m.parameters()).dtype)
m.float()
out["back_to_float_dtype"] = str(next(m.parameters()).dtype)

# ... and the model still computes afterwards.
x = torch.zeros(3, 4)
out["forward_shape"] = list(m(x).shape)

# Tensor-side spellings.
t = torch.zeros(2, 3)
out["t_device"] = str(t.device)
out["t_is_cpu"] = t.is_cpu
out["t_is_cuda"] = t.is_cuda
out["t_get_device"] = t.get_device()
out["t_cpu_is_self"] = t.cpu() is t
out["t_to_cpu_is_self"] = t.to("cpu") is t
out["t_to_cpu0_is_self"] = t.to("cpu:0") is t
out["t_to_cpu0_device"] = str(t.to("cpu:0").device)
out["t_to_copy_is_not_self"] = t.to("cpu", copy=True) is not t
out["t_to_dtype_device"] = str(t.to(torch.device("cpu"), torch.float64).dtype)

# The label round-trips through pickle only when `torch._C` is importable by
# name, which is exactly the situation here and not in the bare-`_C` harness.
out["pickle_round_trip"] = [
    str(pickle.loads(pickle.dumps(torch.device(s))))
    for s in ("cpu", "cpu:0", "cuda:1", "meta")
]

# The two accelerator questions, through their real callers.
out["get_device_module_none"] = torch.get_device_module().__name__
out["get_device_module_cpu"] = torch.get_device_module("cpu").__name__
out["current_accelerator"] = repr(torch.accelerator.current_accelerator())
out["accelerator_count"] = torch.accelerator.device_count()
out["accelerator_available"] = torch.accelerator.is_available()
out["default_device"] = repr(torch.get_default_device())
out["cuda_available"] = torch.cuda.is_available()
out["mps_available"] = torch.backends.mps.is_available()
out["generator_device"] = str(torch.default_generator.device)

# An unavailable device is refused at use, by name, wherever it is asked for.
for name, call in (
    ("module_to_cuda", lambda: m.to("cuda")),
    ("tensor_to_cuda", lambda: t.to("cuda")),
    ("tensor_cuda_method", lambda: t.cuda()),
    ("factory_cuda", lambda: torch.zeros(2, device="cuda")),
):
    try:
        call()
    except NotImplementedError as e:
        out[name] = "refused:" + ("cuda" if "cuda" in str(e) else "meta" if "meta" in str(e) else "?")
    except BaseException as e:
        out[name] = f"{type(e).__name__}: {e}"
    else:
        out[name] = "ACCEPTED"

# A typo is refused at construction, not at use.
try:
    torch.device("cuad")
except RuntimeError:
    out["typo_refused"] = True
except BaseException as e:
    out["typo_refused"] = f"{type(e).__name__}"
else:
    out["typo_refused"] = False

json.dump(out, sys.stdout)
"""


def _device_road_fixture():
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    proc = subprocess.run(
        [sys.executable, "-c", _DEVICE_ROAD_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"device-road subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def test_device_road_through_the_vendored_tree():
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return  # vendor tree not installed -- see vendor/install_shim.sh
    r = _device_road_fixture()

    # `nn.Module.to(...)` in every spelling `Module.to`'s docstring offers.
    for key in ("to_str_returns_self", "to_device_returns_self", "cpu_returns_self",
                "to_dtype_returns_self", "to_device_and_dtype", "float_returns_self",
                "param_identity_preserved"):
        assert r[key] is True, key
    assert r["param_device"] == "cpu"
    # The cast is real, not a no-op that trivially satisfies "returns self".
    assert r["double_dtype"] == "torch.float64", r["double_dtype"]
    assert r["back_to_float_dtype"] == "torch.float32", r["back_to_float_dtype"]
    assert r["forward_shape"] == [3, 2], r["forward_shape"]

    # Tensor-side. `.to()` to where you already are is an alias, and a real
    # `copy=True` is not -- measured on torch 2.13.0, both ways.
    assert r["t_device"] == "cpu"
    assert r["t_is_cpu"] is True and r["t_is_cuda"] is False
    assert r["t_get_device"] == -1
    for key in ("t_cpu_is_self", "t_to_cpu_is_self", "t_to_copy_is_not_self"):
        assert r[key] is True, key
    # `cpu:0` is *not* the label a plain-cpu tensor wears, so `.to("cpu:0")`
    # copies even though it lands on the same device -- and the copy then
    # reports plain `cpu`, because the tensor's label comes from the backend it
    # ended up on rather than from the string that was asked for. Both halves
    # measured on torch 2.13.0, and both are the reason `same_physical_device`
    # exists next to `__eq__`.
    assert r["t_to_cpu0_is_self"] is False, r["t_to_cpu0_is_self"]
    assert r["t_to_cpu0_device"] == "cpu", r["t_to_cpu0_device"]
    assert r["t_to_dtype_device"] == "torch.float64"

    assert r["pickle_round_trip"] == ["cpu", "cpu:0", "cuda:1", "meta"]

    # The two accelerator questions, and the callers that disagree about None.
    assert r["get_device_module_none"] == "torch.cpu", r["get_device_module_none"]
    assert r["get_device_module_cpu"] == "torch.cpu"
    assert r["current_accelerator"] == "None", r["current_accelerator"]
    assert r["accelerator_count"] == 0
    assert r["accelerator_available"] is False
    assert r["default_device"] == "device(type='cpu')", r["default_device"]
    assert r["cuda_available"] is False
    assert r["mps_available"] is False
    assert r["generator_device"] == "cpu"

    # An unavailable device is refused at *use*; a typo is refused at
    # *construction*. Those are the two halves of "a device is a label".
    assert r["module_to_cuda"] == "refused:cuda", r["module_to_cuda"]
    assert r["tensor_to_cuda"] == "refused:cuda", r["tensor_to_cuda"]
    assert r["tensor_cuda_method"] == "refused:cuda", r["tensor_cuda_method"]
    assert r["factory_cuda"] == "refused:cuda", r["factory_cuda"]
    assert r["typo_refused"] is True, r["typo_refused"]

    # `meta` used to be in the loop above, refused alongside `cuda`. It is not
    # the same kind of thing and now it does not behave like one: `cuda` is a
    # backend this build does not link, `meta` is a device that needs no
    # backend at all. `t.to("meta")` returns a tensor here now, and the
    # measurement that it is the *right* tensor lives in
    # `test_meta_tensors_carry_shape_and_dtype_and_no_data`. docs/META.md.


# --- the meta device (docs/META.md) -----------------------------------------
#
# `meta` is the second device this build has, and the first one that needed no
# backend. That is what it is *for* here: every claim docs/DEVICE_ABS.md had to
# argue about "when there are two devices" is testable now, and the tests below
# are that argument turned into assertions.


def test_meta_tensors_carry_shape_and_dtype_and_no_data():
    """The whole of what a meta tensor is, and the whole of what it is not.

    Every expected value here was measured on upstream torch 2.13.0 first
    (docs/META.md §2) -- including the two *different* refusals, which are not
    tidied into one: `.tolist()` is
    `NotImplementedError: Cannot copy out of meta tensor; no data!` and
    `.item()` is `RuntimeError: Tensor.item() cannot be called on meta
    tensors`. A shim that unified them would be reporting a shape upstream
    does not have.
    """
    d = _C._aten_dispatch
    meta = _C.device("meta")

    t = d("aten.empty.memory_format", [2, 3], _C.float32, device=meta)
    assert tuple(t.shape) == (2, 3)
    assert t.dtype == _C.float32
    assert t.device == meta
    assert t.is_meta is True
    assert t.is_cpu is False
    assert t.is_cuda is False
    assert t.numel() == 6
    assert t.dim() == 2
    assert t.element_size() == 4
    assert t.size(0) == 2 and t.size(-1) == 3

    # No bytes, and it says so rather than producing any.
    try:
        t.tolist()
    except NotImplementedError as e:
        assert str(e) == "Cannot copy out of meta tensor; no data!", str(e)
    else:
        raise AssertionError("tolist() on a meta tensor returned data")

    one = d("aten.empty.memory_format", [1], _C.float32, device=meta)
    try:
        d("aten._local_scalar_dense.default", one)
    except RuntimeError as e:
        assert str(e) == "Tensor.item() cannot be called on meta tensors", str(e)
    else:
        raise AssertionError("item() on a meta tensor returned a scalar")


def test_meta_drops_the_device_index_where_cpu_does_too():
    """Measured, not tidied: upstream normalises every meta index away.

    `torch.zeros(2, device="meta:7").device` is `device(type='meta')`, exactly
    as `device="cpu:3"` reports plain `cpu`. That is why `Repr::Meta` stores no
    label -- there is only one meta device, so the label is a constant, and
    `PyDevice::from_candle`'s hardcoded index (docs/DEVICE_ABS.md §3.2) is not
    the thing that answers for it.

    The *label* `meta:7` still exists and is still unequal to bare `meta`;
    it is tensors that forget the index, not labels.
    """
    d = _C._aten_dispatch
    for spelling in ("meta", "meta:0", "meta:7"):
        t = d("aten.empty.memory_format", [2], device=_C.device(spelling))
        assert t.device == _C.device("meta"), (spelling, t.device)
    assert _C.device("meta:7") != _C.device("meta")
    assert _C._shim_same_device(_C.device("meta"), _C.device("meta:7"))


def test_the_gate_refuses_a_mixed_device_op_and_finds_it_in_a_sequence():
    """**The half of `check_devices_agree` that had never run.**

    docs/DEVICE_ABS.md §10 recorded it as untested and unreachable: with `cpu`
    the only resolvable label, an input that disagreed could not be built. It
    can now, and running it found a real hole -- the keyword loop did not
    descend into sequences, and every torch-level call arrives with its
    arguments bound by *name*, so `cat([cpu, meta])` went straight past the
    gate. docs/META.md §5.

    Both argument shapes are pinned here for that reason: a plain tensor
    argument and a `Tensor[]`, each positionally *and* by keyword.
    """
    d = _C._aten_dispatch
    cpu = d("aten.full.default", [2], 1.0)
    meta = d("aten.empty.memory_format", [2], _C.float32, device=_C.device("meta"))

    def refuses(*args, **kwargs):
        try:
            d(*args, **kwargs)
        except RuntimeError as e:
            assert "at least two devices" in str(e), str(e)
            return True
        except NotImplementedError as e:  # pragma: no cover - the bug this pins
            raise AssertionError(
                "the gate did not fire; the call reached a kernel and died "
                f"there instead: {e}"
            )
        return False

    assert refuses("aten.add.Tensor", cpu, meta)
    assert refuses("aten.add.Tensor", meta, cpu)
    assert refuses("aten.add.Tensor", self=cpu, other=meta)
    assert refuses("aten.cat.default", [cpu, meta])
    assert refuses("aten.cat.default", (cpu, meta))
    assert refuses("aten.cat.default", tensors=[cpu, meta])
    assert refuses("aten.cat.default", tensors=(meta, cpu))

    # And the passing half still passes, on both devices.
    assert d("aten.add.Tensor", cpu, cpu).tolist() == [2.0, 2.0]
    assert d("aten.detach.default", meta).is_meta is True

    # `copy_` is the exception upstream carves out, because transferring is
    # the definition of the op. The direction decides what happens:
    # meta <- cpu is a no-op that keeps the receiver on meta (upstream warns
    # about exactly this in `load_state_dict`), cpu <- meta has nothing to read.
    assert d("aten.copy_.default", meta, cpu).is_meta is True
    try:
        d("aten.copy_.default", cpu, meta)
    except NotImplementedError as e:
        assert str(e) == "Cannot copy out of meta tensor; no data!", str(e)
    else:
        raise AssertionError("cpu.copy_(meta) read bytes that do not exist")


def test_meta_transfers_go_one_way_only():
    """`cpu -> meta` discards; `meta -> cpu` refuses. Both are upstream's.

    This is the property `meta` exists for, and the one a shim could most
    easily get wrong in the expensive direction: materialising zeros for
    `meta.to("cpu")` would turn "these weights were never loaded" into "these
    weights are zero", which is the failure `docs/CKPT.md`'s `filled` guard
    exists to stop one layer down.
    """
    d = _C._aten_dispatch
    meta = _C.device("meta")
    cpu = _C.device("cpu")

    dense = d("aten.full.default", [2], 1.0)
    moved = d("aten._to_copy.default", dense, device=meta)
    assert moved.is_meta is True and tuple(moved.shape) == (2,)
    assert dense.tolist() == [1.0, 1.0]  # the source is untouched

    # dtype changes stay on meta, including when `device=` is absent -- absent
    # means "stay where you are", not "go to the CPU" (docs/DEVICE_ABS.md §5.2,
    # observable for the first time now that there are two devices).
    cast = d("aten._to_copy.default", moved, _C.float64)
    assert cast.is_meta is True and cast.dtype == _C.float64

    for kwargs in ({"device": cpu}, {"device": cpu, "dtype": _C.float64}):
        try:
            d("aten._to_copy.default", moved, **kwargs)
        except NotImplementedError as e:
            assert str(e) == "Cannot copy out of meta tensor; no data!", str(e)
        else:
            raise AssertionError(f"meta -> cpu succeeded for {kwargs}")


def test_meta_arith_scalar_promotes_by_the_same_rule_the_dense_kernel_uses():
    """`from_pretrained` builds the rotary embedding under a meta context.

    `transformers/models/llama/modeling_llama.py:108` computes

        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float) / dim))

    inside `LlamaRotaryEmbedding.__init__`, which `from_pretrained` runs under
    `init_empty_weights`. Two of its links are `Scalar`-overload arithmetic on
    a meta tensor -- the `/ dim`, and the `* other` that
    `torch/_tensor.py:1112` turns `1.0 / t` into.

    The dtype is not "the input's". True division always floats; multiplication
    follows torch's wrapped-number rule, where a Python `float` floats an
    integral tensor and a Python `int` does not. Both rules already exist in
    `arith_tag`, and the meta kernels call it rather than restating it -- a
    meta kernel that promised a dtype the dense kernel would not produce is
    worse than none, because the shape and dtype it hands back are what the
    caller then allocates against.

    Measured on torch 2.13.0 (`torch.empty(..., device="meta") <op> s`):

        float32 / 2      float32      float32 * 2      float32
        float32 / 2.0    float32      float32 * 2.0    float32
        float64 / 2      float64      int64   * 2      int64
        float16 / 2      float16      int64   * 2.0    float32
        int64   / 2      float32      bool    * 2      int64
        int64   / 2.0    float32      bool    * 2.0    float32
        int32   / 2      float32
        bool    / 2      float32

    and under `set_default_dtype(torch.float64)` every row that floats an
    integral operand becomes float64 -- the same coupling the dense path has.
    """
    d = _C._aten_dispatch
    meta = _C.device("meta")

    def empty(shape, dtype):
        return d("aten.empty.memory_format", shape, dtype, device=meta)

    for op, dtype, scalar, want in (
        ("aten.div.Scalar", _C.float32, 2, _C.float32),
        ("aten.div.Scalar", _C.float32, 2.0, _C.float32),
        ("aten.div.Scalar", _C.float64, 2, _C.float64),
        ("aten.div.Scalar", _C.float16, 2, _C.float16),
        ("aten.div.Scalar", _C.int64, 2, _C.float32),
        ("aten.div.Scalar", _C.int64, 2.0, _C.float32),
        ("aten.div.Scalar", _C.int32, 2, _C.float32),
        ("aten.mul.Scalar", _C.float32, 2, _C.float32),
        ("aten.mul.Scalar", _C.float32, 2.0, _C.float32),
        ("aten.mul.Scalar", _C.float64, 2.0, _C.float64),
        # The row that separates the two rules: `*` keeps an integral tensor
        # integral under an `int` scalar, `/` does not.
        ("aten.mul.Scalar", _C.int64, 2, _C.int64),
        ("aten.mul.Scalar", _C.int64, 2.0, _C.float32),
        ("aten.mul.Scalar", _C.int32, 2, _C.int32),
    ):
        out = d(op, empty([2, 3], dtype), scalar)
        assert out.is_meta is True, (op, dtype, scalar)
        assert tuple(out.shape) == (2, 3), (op, dtype, scalar, tuple(out.shape))
        assert out.dtype == want, (op, dtype, scalar, out.dtype)

    # The promotion reads the default dtype, so it moves with it. This is the
    # same cell `set_default_dtype` writes; if a meta kernel had hardcoded
    # float32 this would be the assertion that said so.
    try:
        _C._set_default_dtype(_C.float64)
        assert d("aten.div.Scalar", empty([4], _C.int64), 2).dtype == _C.float64
        assert d("aten.div.Scalar", empty([4], _C.float32), 2).dtype == _C.float32
        assert d("aten.mul.Scalar", empty([4], _C.int64), 2.0).dtype == _C.float64
        assert d("aten.mul.Scalar", empty([4], _C.int64), 2).dtype == _C.int64
    finally:
        _C._set_default_dtype(_C.float32)

    # `torch.bool` is the one row above that is *not* reproduced: the dense
    # kernel refuses a boolean operand by name (BOOL.md §2.2) and the meta
    # kernel shares that rule rather than advertising a computation this build
    # would then refuse to perform. Upstream answers float32 and int64 for
    # these two; that divergence is the dense one, and this asserts it stays a
    # single divergence rather than becoming two answers.
    for op in ("aten.div.Scalar", "aten.mul.Scalar"):
        try:
            d(op, empty([2], _C.bool), 2)
        except NotImplementedError as e:
            assert "torch.bool" in str(e), (op, str(e))
        else:
            raise AssertionError(f"meta {op} accepted a bool the dense path refuses")


def test_meta_pow_scalar_keeps_the_wrapped_number_rule():
    """The next link in the same `LlamaRotaryEmbedding.__init__` expression.

    `base ** (...)` with a Python `base` is `aten::pow.Scalar(Scalar self,
    Tensor exponent)` -- the scalar is the *base* and the tensor is the
    exponent, which is the opposite of `pow.Tensor_Scalar` and the reason the
    two cannot share a kernel.

    The rule is torch's wrapped-number rule, not "widest wins": an `int` base
    does not float an integral exponent, a `float` base does. Measured on
    2.13.0 over `s ** torch.empty(..., device="meta")`:

        2   ** float32  float32     2   ** int64  int64
        2.0 ** float32  float32     2.0 ** int64  float32
        2   ** float16  float16     2   ** int32  int32
        2   ** float64  float64     2.0 ** int32  float32

    `pow_result_tag` is that rule and the meta kernel calls it, so the two
    cannot drift apart.
    """
    d = _C._aten_dispatch
    meta = _C.device("meta")

    def empty(shape, dtype):
        return d("aten.empty.memory_format", shape, dtype, device=meta)

    for base, dtype, want in (
        (2, _C.float32, _C.float32),
        (2.0, _C.float32, _C.float32),
        (2, _C.float64, _C.float64),
        (2, _C.float16, _C.float16),
        (2, _C.int64, _C.int64),
        (2.0, _C.int64, _C.float32),
        (2, _C.int32, _C.int32),
        (2.0, _C.int32, _C.float32),
    ):
        out = d("aten.pow.Scalar", base, empty([2, 3], dtype))
        assert out.is_meta is True, (base, dtype)
        assert tuple(out.shape) == (2, 3), (base, dtype, tuple(out.shape))
        assert out.dtype == want, (base, dtype, out.dtype)

    # Only the float-base-over-integral-exponent row reads the default dtype,
    # which is exactly the row upstream floats. If the kernel had hardcoded
    # float32 the first of these would say so; if it had floated everything the
    # second would.
    try:
        _C._set_default_dtype(_C.float64)
        assert d("aten.pow.Scalar", 2.0, empty([4], _C.int64)).dtype == _C.float64
        assert d("aten.pow.Scalar", 2, empty([4], _C.int64)).dtype == _C.int64
    finally:
        _C._set_default_dtype(_C.float32)

    # As with `div.Scalar`: upstream answers `int64` for `2 ** bool_meta`, the
    # dense kernel refuses a boolean operand by name, and the meta kernel
    # refuses with it rather than promising a second answer.
    try:
        d("aten.pow.Scalar", 2, empty([2], _C.bool))
    except NotImplementedError as e:
        assert "torch.bool" in str(e), str(e)
    else:
        raise AssertionError("meta pow.Scalar accepted a bool the dense path refuses")


def test_meta_reciprocal_floats_an_integral_input():
    """The last link: `1.0 / t` is `t.reciprocal() * 1.0` in the vendored tree.

    `torch/_tensor.py:1112` spells `__rdiv__` that way, so
    `LlamaRotaryEmbedding.__init__` reaches `aten::reciprocal` and then
    `aten::mul.Scalar` rather than an `rdiv` op.

    `reciprocal` is in the `unary_float` family: a floating input keeps its
    dtype, anything else becomes the default float. Measured on 2.13.0 over
    `torch.empty(..., device="meta").reciprocal()` -- float32/float64/float16/
    bfloat16 are preserved, int64/int32 give float32, and under
    `set_default_dtype(torch.float64)` the integral rows give float64 while the
    float32 row stays float32.

    Only `reciprocal` is added here. `cos`/`sin`/`tanh`/`exp`/`rsqrt` share the
    dense helper and would share this rule, but nothing has asked for them on
    meta -- a meta kernel nobody reached is a claim nobody checked.
    """
    d = _C._aten_dispatch
    meta = _C.device("meta")

    def empty(shape, dtype):
        return d("aten.empty.memory_format", shape, dtype, device=meta)

    for dtype, want in (
        (_C.float32, _C.float32),
        (_C.float64, _C.float64),
        (_C.float16, _C.float16),
        (_C.bfloat16, _C.bfloat16),
        (_C.int64, _C.float32),
        (_C.int32, _C.float32),
    ):
        out = d("aten.reciprocal.default", empty([2, 3], dtype))
        assert out.is_meta is True, dtype
        assert tuple(out.shape) == (2, 3), (dtype, tuple(out.shape))
        assert out.dtype == want, (dtype, out.dtype)

    try:
        _C._set_default_dtype(_C.float64)
        assert d("aten.reciprocal.default", empty([4], _C.int64)).dtype == _C.float64
        assert d("aten.reciprocal.default", empty([4], _C.float32)).dtype == _C.float32
    finally:
        _C._set_default_dtype(_C.float32)

    # The links either side of it, in the order the rope expression runs them.
    # Asserting the chain rather than the link is what catches a kernel that is
    # right in isolation and wrong in place -- `arange` already answered on
    # meta, and `div.Scalar`/`pow.Scalar` were the two walls before this one.
    chain = d("aten.arange.start_step", 0, 8, 2, _C.float32, device=meta)
    assert tuple(chain.shape) == (4,), tuple(chain.shape)
    chain = d("aten.div.Scalar", chain, 8)
    chain = d("aten.pow.Scalar", 10000.0, chain)
    chain = d("aten.mul.Scalar", d("aten.reciprocal.default", chain), 1.0)
    assert chain.is_meta is True
    assert tuple(chain.shape) == (4,), tuple(chain.shape)
    assert chain.dtype == _C.float32, chain.dtype


def test_ops_without_a_meta_kernel_name_themselves():
    """DESIGN.md §6's instrument, pointed at the meta device.

    Shape inference for an op is a real kernel (upstream keeps thousands of
    lines of them in `torch/_meta_registrations.py`), so the ops that have one
    here are a short list and everything else refuses **with its own name in
    the message**. That is the difference between a recorded boundary and a
    hole: running a model under `with torch.device("meta")` prints the work
    queue in frequency order rather than producing a wrong shape.
    """
    d = _C._aten_dispatch
    meta = _C.device("meta")
    a = d("aten.empty.memory_format", [2, 3], _C.float32, device=meta)
    b = d("aten.empty.memory_format", [2, 3], _C.float32, device=meta)

    for op, args in (
        ("aten.add.Tensor", (a, b)),
        ("aten.mm.default", (a, a)),
        ("aten.view.default", (a, [3, 2])),
        ("aten.slice.Tensor", (a, 0, 0, 1)),
        ("aten.sum.default", (a,)),
    ):
        try:
            d(op, *args)
        except NotImplementedError as e:
            assert op in str(e), (op, str(e))
            assert "no meta kernel" in str(e), (op, str(e))
        else:
            raise AssertionError(f"{op} answered on meta without a meta kernel")

    # `_aten_implemented()` is untouched by any of this: it means "has a kernel
    # *and* tools/golden/cases.py compares it against upstream", and a meta
    # tensor has no values to compare. Meta support is a property of ops
    # already on that list. docs/META.md §7.
    assert "aten.add.Tensor" in _C._aten_implemented()


def test_the_initialisers_a_module_constructor_runs_are_no_ops_on_meta():
    """`nn.Linear.__init__` ends in `kaiming_uniform_`, which is `uniform_`.

    Without these, `with torch.device("meta"): nn.Linear(4, 8)` stops before it
    can produce a single parameter -- which is the entire call
    `accelerate.init_empty_weights` is built around. Writing nothing into a
    tensor that holds no bytes is upstream's meta kernel for them too; the
    refusal that matters (reading the values back) is still `tolist`'s.
    """
    d = _C._aten_dispatch
    t = d("aten.empty.memory_format", [2, 3], _C.float32, device=_C.device("meta"))
    for op, args in (
        ("aten.uniform_.default", (t, 0.0, 1.0)),
        ("aten.normal_.default", (t, 0.0, 1.0)),
        ("aten.zero_.default", (t,)),
        ("aten.fill_.Scalar", (t, 3.0)),
    ):
        out = d(op, *args)
        assert out is t, op
        assert out.is_meta is True and tuple(out.shape) == (2, 3), op


# `with torch.device(...)` needs the *vendored* tree: `torch.device.__enter__`
# builds a `torch.utils._device.DeviceContext`, which lives in Python, and the
# factories it rewrites are `torch._C._VariableFunctions` members reached
# through `torch`. None of that exists around the standalone `_C` the tests
# above import, so this one runs in the same subprocess shape as
# `test_device_road_through_the_vendored_tree`.
_META_ROAD_SCRIPT = r"""
import json, sys
import torch
import torch.nn as nn

out = {}

# -- the mode stack is real, and balanced -----------------------------------
out["stack_before"] = torch._C._len_torch_function_stack()
with torch.device("meta"):
    out["stack_inside"] = torch._C._len_torch_function_stack()
    out["zeros_device"] = str(torch.zeros(2).device)
    out["tensor_device"] = str(torch.tensor([1.0, 2.0]).device)
    out["empty_device"] = str(torch.empty(2, 3).device)
    # An explicit device beats the context, which is what upstream's
    # `kwargs.get("device") is None` guard means.
    out["explicit_wins"] = str(torch.zeros(2, device="cpu").device)
    out["default_inside"] = repr(torch.get_default_device())
out["stack_after"] = torch._C._len_torch_function_stack()
out["zeros_after"] = str(torch.zeros(2).device)

# `__enter__` returns the *device*, not the mode (measured on upstream).
with torch.device("meta") as handle:
    out["enter_returns"] = repr(handle)

# Nested: the inner block wins inside, the outer one is restored after.
with torch.device("meta"):
    with torch.device("cpu"):
        out["nested_inner"] = str(torch.zeros(2).device)
    out["nested_outer"] = str(torch.zeros(2).device)

# -- set_default_device is the same mechanism, held open ---------------------
torch.set_default_device("meta")
out["default_device_set"] = str(torch.zeros(2).device)
out["default_device_read"] = repr(torch.get_default_device())
torch.set_default_device(None)
out["default_device_cleared"] = str(torch.zeros(2).device)

# -- `accelerate.init_empty_weights`, which is what all of it is for ---------
with torch.device("meta"):
    model = nn.Sequential(nn.Linear(4, 8), nn.Linear(8, 2))
out["empty_params"] = [
    [name, list(p.shape), str(p.device), type(p).__name__]
    for name, p in model.named_parameters()
]
out["empty_state_dict"] = {
    k: [str(v.device), list(v.shape)] for k, v in model.state_dict().items()
}

# ... and the weights land on it afterwards, and it computes.
model.load_state_dict(
    {
        "0.weight": torch.ones(8, 4), "0.bias": torch.zeros(8),
        "1.weight": torch.ones(2, 8), "1.bias": torch.zeros(2),
    },
    assign=True,
)
out["after_load_device"] = str(next(model.parameters()).device)
out["after_load_forward"] = model(torch.ones(1, 4)).tolist()

# -- tensor-side spellings through the vendored `Tensor` --------------------
t = torch.zeros(2, 3)
out["to_meta_device"] = str(t.to("meta").device)
out["to_meta_is_not_self"] = t.to("meta") is not t
m = t.to("meta")
out["meta_to_meta_is_self"] = m.to("meta") is m
out["meta_is_meta"] = m.is_meta
try:
    m.cpu()
except NotImplementedError as e:
    out["meta_cpu"] = str(e)
else:
    out["meta_cpu"] = "ACCEPTED"
try:
    torch.zeros(2) + torch.zeros(2, device="meta")
except RuntimeError as e:
    out["mixed_add"] = str(e)
else:
    out["mixed_add"] = "ACCEPTED"

json.dump(out, sys.stdout)
"""


def _meta_road_fixture():
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    proc = subprocess.run(
        [sys.executable, "-c", _META_ROAD_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"meta-road subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def test_meta_road_through_the_vendored_tree():
    """`with torch.device("meta")` end to end, and the reason it had to be.

    docs/DEVICE_ABS.md §7.2 refused to build the mode stack on its own, because
    a stack nothing consults makes `with torch.device("meta"): torch.zeros(2)`
    return a **CPU** tensor with the block appearing to work. Every assertion
    below is the other half of that: the stack exists *and* the factories
    consult it.

    Each expected value was measured on upstream torch 2.13.0 first
    (docs/META.md §2/§8), including the ones that are easy to guess wrong --
    `__enter__` returns the device rather than the mode, and an explicit
    `device=` beats the context.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return  # vendor tree not installed -- see vendor/install_shim.sh
    r = _meta_road_fixture()

    # The stack is real and it balances. A leak here would leave every later
    # factory call routing through a dead mode.
    assert r["stack_before"] == 0
    assert r["stack_inside"] == 1
    assert r["stack_after"] == 0

    # Factories consult it -- the half that would otherwise be silent.
    assert r["zeros_device"] == "meta", r["zeros_device"]
    assert r["tensor_device"] == "meta", r["tensor_device"]
    assert r["empty_device"] == "meta", r["empty_device"]
    assert r["explicit_wins"] == "cpu", r["explicit_wins"]
    assert r["default_inside"] == "device(type='meta')", r["default_inside"]
    assert r["zeros_after"] == "cpu", r["zeros_after"]
    assert r["enter_returns"] == "device(type='meta')", r["enter_returns"]
    assert r["nested_inner"] == "cpu", r["nested_inner"]
    assert r["nested_outer"] == "meta", r["nested_outer"]

    # `set_default_device` is `DeviceContext` held open, and it has to be
    # closeable again.
    assert r["default_device_set"] == "meta", r["default_device_set"]
    assert r["default_device_read"] == "device(type='meta')"
    assert r["default_device_cleared"] == "cpu", r["default_device_cleared"]

    # `init_empty_weights`: a whole module with shapes, dtypes and `Parameter`
    # identity, and not one byte allocated for its weights.
    assert r["empty_params"] == [
        ["0.weight", [8, 4], "meta", "Parameter"],
        ["0.bias", [8], "meta", "Parameter"],
        ["1.weight", [2, 8], "meta", "Parameter"],
        ["1.bias", [2], "meta", "Parameter"],
    ], r["empty_params"]
    assert r["empty_state_dict"]["0.weight"] == ["meta", [8, 4]]

    # ... then the real weights arrive and it computes. The number is upstream's
    # -- the same script on torch 2.13.0 gives [[32.0, 32.0]] (docs/META.md §8).
    assert r["after_load_device"] == "cpu", r["after_load_device"]
    assert r["after_load_forward"] == [[32.0, 32.0]], r["after_load_forward"]

    # One way only, through the vendored `Tensor.to`.
    assert r["to_meta_device"] == "meta"
    assert r["to_meta_is_not_self"] is True
    assert r["meta_to_meta_is_self"] is True  # upstream: `t.to("meta") is t`
    assert r["meta_is_meta"] is True
    assert r["meta_cpu"] == "Cannot copy out of meta tensor; no data!", r["meta_cpu"]
    assert "at least two devices" in r["mixed_add"], r["mixed_add"]


# --- the capture layer (docs/CAPTURE.md) ------------------------------------
#
# DESIGN.md §11.1 named the reason this exists: an NPU is not an eager device,
# it is an executor that takes a whole graph. The single door is what makes
# recording one cheap -- every op goes through `_aten_dispatch`, so the record
# is taken in one place rather than in 97 kernels.
#
# The proof these tests are after is *not* "the recorder produced a list". It
# is that the recorded graph, replayed with different inputs and detached from
# the Python that produced it, computes the same function as eager -- because
# that is exactly what a delegate does with it. Everything else here is the
# other half: the conditions under which that equality is allowed to be
# claimed (the guards), and the refusals that keep it from being claimed when
# it does not hold.


def _flat(t):
    """A tensor's values as a flat list, for bit-exact comparison."""

    def walk(v, into):
        if isinstance(v, list):
            for item in v:
                walk(item, into)
        else:
            into.append(v)

    out = []
    walk(t.tolist(), out)
    return out


def test_capture_is_off_until_it_is_asked_for():
    """The door stays a door. Nothing is recorded unless recording is on."""
    assert _C._capture_active() is False
    assert _C._capture_reason() is None
    # And the ordinary path still answers while capture is off.
    t = _C._aten_dispatch("aten.full.default", [2], 1.0)
    assert t.tolist() == [1.0, 1.0]
    assert _C._capture_active() is False


def test_capture_records_ops_in_order_with_input_and_output_metadata():
    d = _C._aten_dispatch
    a = _C._tensor_new_from_data([[1.0, 2.0], [3.0, 4.0]])
    b = _C._tensor_new_from_data([[5.0, 6.0], [7.0, 8.0]])
    bias = _C._tensor_new_from_data([1.0, -100.0])

    _C._capture_begin([a, b])
    assert _C._capture_active() is True
    x = d("aten.mm.default", a, b)
    y = d("aten.add.Tensor", x, bias)
    z = d("aten.relu.default", y)
    trace = _C._capture_end(z)
    assert _C._capture_active() is False

    assert len(trace) == 3
    assert [n["op"] for n in trace.nodes] == [
        "aten.mm.default",
        "aten.add.Tensor",
        "aten.relu.default",
    ]

    # Values are references, not copies: an argument is either a placeholder,
    # a constant, or the output of an earlier node. That is the whole content
    # of "straight-line segment".
    v = _C._capture_value
    assert trace.nodes[0]["args"] == [v("input", 0), v("input", 1)]
    assert trace.nodes[1]["args"] == [v("node", 0), v("const", 0)]
    assert trace.nodes[2]["args"] == [v("node", 1)]
    assert trace.outputs == [v("node", 2)]

    # Each node carries the shape and dtype of what it produced ...
    for node in trace.nodes:
        assert node["outputs"] == [
            {"shape": [2, 2], "dtype": "torch.float32", "device": "cpu"}
        ], node

    # ... and the placeholders carry the shape, dtype and device of what came
    # in. `bias` was never declared an input, so it is a constant -- which is
    # the split `ExportedProgram.graph_signature` makes between user inputs and
    # lifted parameters.
    assert trace.guards == [
        {"index": 0, "shape": [2, 2], "dtype": "torch.float32", "device": "cpu"},
        {"index": 1, "shape": [2, 2], "dtype": "torch.float32", "device": "cpu"},
    ]
    assert trace.constants == [
        {"index": 0, "shape": [2], "dtype": "torch.float32", "device": "cpu"}
    ]


def test_capture_replay_matches_eager_bit_for_bit():
    """The proof. Same graph, different inputs, same answer as eager.

    Not "close": equal. Replay re-enters the same door with the same op names
    and the same non-tensor arguments, so any difference at all would mean the
    record had lost something -- an argument, an order, a dtype.
    """
    d = _C._aten_dispatch
    a = _C._tensor_new_from_data([[1.0, 2.0], [3.0, 4.0]])
    b = _C._tensor_new_from_data([[5.0, 6.0], [7.0, 8.0]])
    bias = _C._tensor_new_from_data([1.0, -100.0])

    def eager(p, q):
        x = d("aten.mm.default", p, q)
        y = d("aten.add.Tensor", x, bias)
        y = d("aten.relu.default", y)
        y = d("aten.sum.dim_IntList", y, [1])
        return d("aten.mul.Scalar", y, 2.5)

    _C._capture_begin([a, b])
    recorded = eager(a, b)
    trace = _C._capture_end(recorded)

    # Same inputs first: the record has to reproduce what it just watched.
    (again,) = trace.replay([a, b])
    assert _flat(again) == _flat(recorded)

    # Then the point of it -- inputs it has never seen.
    for pv, qv in (
        ([[0.5, -1.5], [2.0, 9.0]], [[1.25, 0.0], [-3.0, 4.0]]),
        ([[1e-8, 1e8], [-7.0, 0.0]], [[3.5, -2.5], [0.125, 6.0]]),
    ):
        p = _C._tensor_new_from_data(pv)
        q = _C._tensor_new_from_data(qv)
        (replayed,) = trace.replay([p, q])
        assert _flat(replayed) == _flat(eager(p, q)), (pv, qv)


def test_capture_replay_carries_keyword_and_literal_arguments():
    """Non-tensor arguments are burned into the graph, and that is a guard too.

    A trace is only a function of its *tensor* inputs. Every dim, every scalar,
    every dtype seen at record time is a constant afterwards -- so a `dim=1`
    trace replayed on a program that meant `dim=0` is not a near miss, it is a
    different function. The record keeps them verbatim so that no later reader
    has to reconstruct them.
    """
    d = _C._aten_dispatch
    a = _C._tensor_new_from_data([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    _C._capture_begin([a])
    s = d("aten.sum.dim_IntList", a, [1], keepdim=True)
    c = d("aten.cat.default", [s, s], dim=1)
    out = d("aten._to_copy.default", c, _C.float64)
    trace = _C._capture_end(out)

    assert trace.nodes[0]["args"][1] == [1]
    assert trace.nodes[0]["kwargs"] == {"keepdim": True}
    assert trace.nodes[1]["kwargs"] == {"dim": 1}
    assert trace.nodes[2]["args"][1] == _C.float64
    assert trace.nodes[2]["outputs"][0]["dtype"] == "torch.float64"

    b = _C._tensor_new_from_data([[9.0, 8.0, 7.0], [0.5, 0.25, 0.125]])
    (replayed,) = trace.replay([b])
    expect = d(
        "aten._to_copy.default",
        d("aten.cat.default", [d("aten.sum.dim_IntList", b, [1], keepdim=True)] * 2, dim=1),
        _C.float64,
    )
    assert _flat(replayed) == _flat(expect)
    assert replayed.dtype == _C.float64


def test_capture_replay_handles_ops_that_return_more_than_one_tensor():
    d = _C._aten_dispatch
    a = _C._tensor_new_from_data([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    w = _C._tensor_new_from_data([1.0, 1.0])
    bias = _C._tensor_new_from_data([0.0, 0.0])

    _C._capture_begin([a])
    ln = d("aten.native_layer_norm.default", a, [2], w, bias, 1e-5)
    out = d("aten.mul.Scalar", ln[0], 3.0)
    trace = _C._capture_end(out)

    # Three results, and the record knows which one was consumed.
    assert len(trace.nodes[0]["outputs"]) == 3
    assert trace.nodes[1]["args"][0] == _C._capture_value("node", 0, 0)

    b = _C._tensor_new_from_data([[-1.0, 4.0], [0.0, 0.25], [7.0, 7.5]])
    (replayed,) = trace.replay([b])
    expect = d("aten.mul.Scalar", d("aten.native_layer_norm.default", b, [2], w, bias, 1e-5)[0], 3.0)
    assert _flat(replayed) == _flat(expect)


def test_capture_replay_returns_every_declared_output():
    d = _C._aten_dispatch
    a = _C._tensor_new_from_data([[1.0, 2.0], [3.0, 4.0]])

    _C._capture_begin([a])
    lo = d("aten.mul.Scalar", a, 2.0)
    hi = d("aten.add.Scalar", a, 10.0)
    trace = _C._capture_end([lo, hi])

    assert len(trace.outputs) == 2
    b = _C._tensor_new_from_data([[0.5, 0.5], [0.5, 0.5]])
    got_lo, got_hi = trace.replay([b])
    assert _flat(got_lo) == _flat(d("aten.mul.Scalar", b, 2.0))
    assert _flat(got_hi) == _flat(d("aten.add.Scalar", b, 10.0))


# --- the guards -------------------------------------------------------------


def _one_op_trace():
    d = _C._aten_dispatch
    a = _C._tensor_new_from_data([[1.0, 2.0], [3.0, 4.0]])
    _C._capture_begin([a])
    out = d("aten.mul.Scalar", a, 2.0)
    return _C._capture_end(out)


def test_capture_guard_refuses_a_different_shape():
    """A capture without guards is a capture that is quietly wrong.

    The record holds concrete shapes -- every intermediate shape in it was
    computed from the shapes that came in. Replaying with a different one does
    not merely risk a wrong answer, it makes every recorded output shape a lie.
    Dynamic shapes are out of scope here (docs/CAPTURE.md §4), and this is what
    "out of scope" has to mean: refused by name, not attempted.
    """
    trace = _one_op_trace()
    wrong = _C._tensor_new_from_data([[1.0, 2.0, 3.0]])
    try:
        trace.replay([wrong])
    except RuntimeError as e:
        assert "shape" in str(e), str(e)
        assert "[2, 2]" in str(e) and "[1, 3]" in str(e), str(e)
        assert "input 0" in str(e), str(e)
    else:
        raise AssertionError("replay accepted a differently shaped input")


def test_capture_guard_refuses_a_different_dtype():
    trace = _one_op_trace()
    wrong = _C._aten_dispatch(
        "aten._to_copy.default",
        _C._tensor_new_from_data([[1.0, 2.0], [3.0, 4.0]]),
        _C.float64,
    )
    try:
        trace.replay([wrong])
    except RuntimeError as e:
        assert "dtype" in str(e), str(e)
        assert "torch.float32" in str(e) and "torch.float64" in str(e), str(e)
    else:
        raise AssertionError("replay accepted a differently typed input")


def test_capture_guard_refuses_a_different_device():
    trace = _one_op_trace()
    wrong = _C._aten_dispatch(
        "aten.empty.memory_format", [2, 2], _C.float32, device=_C.device("meta")
    )
    try:
        trace.replay([wrong])
    except RuntimeError as e:
        assert "device" in str(e), str(e)
        assert "meta" in str(e) and "cpu" in str(e), str(e)
    else:
        raise AssertionError("replay accepted an input on another device")


def test_capture_guard_refuses_the_wrong_number_of_inputs():
    trace = _one_op_trace()
    a = _C._tensor_new_from_data([[1.0, 2.0], [3.0, 4.0]])
    for bad in ([], [a, a]):
        try:
            trace.replay(bad)
        except RuntimeError as e:
            assert "1 input" in str(e), str(e)
        else:
            raise AssertionError(f"replay accepted {len(bad)} inputs for a 1-input trace")


def test_capture_guard_refuses_a_non_tensor_input():
    trace = _one_op_trace()
    try:
        trace.replay([3.0])
    except TypeError as e:
        assert "input 0" in str(e), str(e)
    else:
        raise AssertionError("replay accepted a float where a tensor was recorded")


# --- the refusals -----------------------------------------------------------


def _capture_refusal(body, inputs):
    """Run `body` under recording and return the reason capture gave up.

    Poisoning rather than raising at the op: capture is an observation, and an
    observation that changes the program it observes is not one. The model runs
    to completion either way; what fails is the *claim* that it was captured.
    """
    _C._capture_begin(inputs)
    body()
    reason = _C._capture_reason()
    try:
        _C._capture_end(None)
    except NotImplementedError as e:
        assert reason is not None and reason in str(e), (reason, str(e))
        assert _C._capture_active() is False
        return str(e)
    raise AssertionError("capture claimed a trace it should have refused")


def test_capture_refuses_reading_a_tensor_value_onto_the_host():
    """`.item()` and `bool(t)` are where a graph stops being a graph.

    DESIGN.md §6 lists branching on a tensor value as one of the things the
    static scan hunts for, and this is the runtime half of the same rule: the
    value is not in the record, so a Python `if` taken on it is a decision the
    replay cannot know it made. The recorded straight line would be *one arm*
    of a branch, replayed unconditionally.
    """
    d = _C._aten_dispatch
    a = _C._tensor_new_from_data([1.0, 2.0])
    got = _capture_refusal(
        lambda: d("aten._local_scalar_dense.default", d("aten.sum.default", a)),
        [a],
    )
    assert "aten._local_scalar_dense.default" in got, got
    # It ran. Capture gave up; the program did not.
    assert d("aten._local_scalar_dense.default", d("aten.sum.default", a)) == 3.0


def test_capture_refuses_in_place_ops():
    """Mutation is what makes aliasing observable, so refusing it removes both.

    In-place aliasing is out of scope (docs/CAPTURE.md §4). Rather than model
    it, capture refuses every mutating overload -- and with no mutation in the
    segment, whether two recorded values share storage cannot be observed. The
    trace is single-assignment by construction rather than by hope.
    """
    d = _C._aten_dispatch
    a = _C._tensor_new_from_data([1.0, 2.0])
    b = _C._tensor_new_from_data([3.0, 4.0])
    for op, args in (
        ("aten.add_.Tensor", (a, b)),
        ("aten.relu_.default", (a,)),
        ("aten.fill_.Scalar", (a, 1.0)),
        ("aten.zero_.default", (a,)),
        ("aten.copy_.default", (a, b)),
    ):
        got = _capture_refusal(lambda op=op, args=args: d(op, *args), [a, b])
        assert op in got, (op, got)
        assert "in place" in got, (op, got)


def test_capture_refuses_ops_that_draw_random_numbers():
    """A replay that does not equal eager cannot be checked against eager.

    Not a claim that the graph is invalid -- it is a claim that this layer has
    no story yet for seeding a delegate, and a trace whose replay differs from
    eager for a legitimate reason would hide one that differs for a bad one.
    docs/CAPTURE.md §4.
    """
    d = _C._aten_dispatch
    probs = _C._tensor_new_from_data([0.25, 0.75])
    got = _capture_refusal(lambda: d("aten.multinomial.default", probs, 1), [probs])
    assert "aten.multinomial.default" in got, got
    assert "random" in got, got

    got = _capture_refusal(lambda: d("aten.randint.default", 10, [2]), [probs])
    assert "aten.randint.default" in got, got


def test_capture_refuses_an_op_that_returns_something_other_than_tensors():
    """The line is metadata versus data.

    `is_floating_point` reads only the dtype, and the dtype is guarded, so its
    answer is fixed for every input the trace admits -- it is recorded and its
    result burned in. `_local_scalar_dense` reads the *bytes*, which the guards
    say nothing about. Only the second is a refusal, and the reason is written
    down here so the allowlist cannot grow by accident.
    """
    d = _C._aten_dispatch
    a = _C._tensor_new_from_data([1.0, 2.0])

    _C._capture_begin([a])
    assert d("aten.is_floating_point.default", a) is True
    out = d("aten.mul.Scalar", a, 2.0)
    trace = _C._capture_end(out)
    assert [n["op"] for n in trace.nodes] == [
        "aten.is_floating_point.default",
        "aten.mul.Scalar",
    ]
    assert trace.nodes[0]["outputs"] == [None]  # recorded, not tracked


def test_capture_refuses_to_nest_or_to_replay_while_recording():
    a = _C._tensor_new_from_data([1.0, 2.0])
    trace = _one_op_trace()

    _C._capture_begin([a])
    try:
        try:
            _C._capture_begin([a])
        except RuntimeError as e:
            assert "already" in str(e), str(e)
        else:
            raise AssertionError("capture nested")

        try:
            trace.replay([_C._tensor_new_from_data([[1.0, 2.0], [3.0, 4.0]])])
        except RuntimeError as e:
            assert "while recording" in str(e), str(e)
        else:
            raise AssertionError("replayed inside a recording")
    finally:
        _C._capture_abandon()
    assert _C._capture_active() is False


def test_capture_refuses_to_end_without_beginning():
    assert _C._capture_active() is False
    for call in (lambda: _C._capture_end(None), _C._capture_abandon):
        try:
            call()
        except RuntimeError as e:
            assert "not recording" in str(e), str(e)
        else:
            raise AssertionError("capture answered while not recording")


def test_capture_refuses_an_output_it_never_saw():
    d = _C._aten_dispatch
    a = _C._tensor_new_from_data([1.0, 2.0])
    stranger = _C._tensor_new_from_data([9.0])
    _C._capture_begin([a])
    d("aten.mul.Scalar", a, 2.0)
    try:
        _C._capture_end(stranger)
    except RuntimeError as e:
        assert "output 0" in str(e), str(e)
    else:
        raise AssertionError("capture accepted an output produced outside the trace")
    assert _C._capture_active() is False


def test_capture_abandon_leaves_nothing_behind():
    d = _C._aten_dispatch
    a = _C._tensor_new_from_data([1.0, 2.0])
    _C._capture_begin([a])
    d("aten.mul.Scalar", a, 2.0)
    _C._capture_abandon()
    assert _C._capture_active() is False
    assert _C._capture_reason() is None
    # And the next recording starts from zero rather than from the abandoned one.
    _C._capture_begin([a])
    d("aten.add.Scalar", a, 1.0)
    trace = _C._capture_end(d("aten.add.Scalar", a, 1.0))
    assert len(trace) == 2


# --- the shape of the record ------------------------------------------------


def test_capture_graph_is_shaped_like_an_exported_program():
    """docs/CAPTURE.md §5: is this structure one that can become Edge dialect?

    The judgement is recorded as a test rather than only as prose, because the
    answer is a claim about *this* data structure and it is cheap to pin. FX's
    graph is placeholders, `get_attr` constants, `call_function` nodes whose
    args are references or literals, and one `output`. That is what `graph()`
    emits, with the op named by its aten overload -- the same key
    `torch.ops.aten.<op>.<overload>` resolves, which is what an Edge lowering
    needs to look each node up.
    """
    d = _C._aten_dispatch
    a = _C._tensor_new_from_data([[1.0, 2.0], [3.0, 4.0]])
    w = _C._tensor_new_from_data([[1.0, 0.0], [0.0, 1.0]])
    _C._capture_begin([a])
    out = d("aten.mm.default", a, w)
    graph = _C._capture_end(out).graph()

    assert sorted(graph) == ["constants", "nodes", "outputs", "placeholders"]
    assert graph["placeholders"] == [
        {"index": 0, "shape": [2, 2], "dtype": "torch.float32", "device": "cpu"}
    ]
    assert graph["constants"] == [
        {"index": 0, "shape": [2, 2], "dtype": "torch.float32", "device": "cpu"}
    ]
    assert graph["nodes"][0]["op"] == "aten.mm.default"
    assert graph["outputs"] == [_C._capture_value("node", 0, 0)]

    # Every op name in the record is one the dispatcher answers, so nothing in
    # a trace is un-lowerable for want of a key.
    known = set(_C._aten_all_implemented())
    for node in graph["nodes"]:
        assert node["op"] in known, node["op"]


def test_capture_value_reads_like_an_fx_node():
    v = _C._capture_value
    assert repr(v("input", 0)) == "%in0"
    assert repr(v("const", 2)) == "%c2"
    assert repr(v("node", 3)) == "%3"
    assert repr(v("node", 3, 1)) == "%3#1"
    assert v("node", 3) == v("node", 3, 0)
    assert v("node", 3) != v("node", 3, 1)
    assert v("input", 0) != v("const", 0)
    assert len({v("node", 3), v("node", 3, 0)}) == 1


# `nn.Module` forward passes need the *vendored* tree: the module layer, the
# functional layer and `Tensor.__matmul__` are all Python, and only the ops
# they lower to reach `_C`. So the end-to-end proof runs in the same subprocess
# shape as the checkpoint and device roads above.
_CAPTURE_ROAD_SCRIPT = r"""
import json, sys
import torch
import torch.nn as nn

torch.manual_seed(0)
out = {}

model = nn.Sequential(nn.Linear(4, 8), nn.ReLU(), nn.Linear(8, 3))
model.eval()

x = torch.ones(2, 4)
eager_first = model(x)

torch._C._capture_begin([x])
y = model(x)
trace = torch._C._capture_end(y)

out["ops"] = [n["op"] for n in trace.nodes]
out["n_placeholders"] = len(trace.guards)
out["n_constants"] = len(trace.constants)
out["guard"] = trace.guards[0]
out["recorded"] = y.reshape(-1).tolist()
out["eager_first"] = eager_first.reshape(-1).tolist()

# Replay with the same input, then with inputs the trace has never seen, and
# compare each against eager on the same input.
(same,) = trace.replay([x])
out["replay_same"] = same.reshape(-1).tolist()

pairs = []
for scale in (0.5, -2.0, 7.25):
    z = torch.ones(2, 4) * scale
    (replayed,) = trace.replay([z])
    pairs.append([replayed.reshape(-1).tolist(), model(z).reshape(-1).tolist()])
out["pairs"] = pairs

# The guard is what makes any of that a claim rather than a hope.
try:
    trace.replay([torch.ones(3, 4)])
except RuntimeError as e:
    out["wrong_batch"] = str(e)
else:
    out["wrong_batch"] = "ACCEPTED"

# A branch on a tensor value inside the traced region is refused by name.
torch._C._capture_begin([x])
h = model(x)
if h.sum().item() > -1e30:
    h = h * 2
out["branch_reason"] = torch._C._capture_reason()
try:
    torch._C._capture_end(h)
except NotImplementedError as e:
    out["branch_refusal"] = str(e)
else:
    out["branch_refusal"] = "ACCEPTED"
out["active_after"] = torch._C._capture_active()

json.dump(out, sys.stdout)
"""


def _capture_road_fixture():
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    proc = subprocess.run(
        [sys.executable, "-c", _CAPTURE_ROAD_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=180,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"capture-road subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def test_capture_road_through_the_vendored_tree():
    """A real `nn.Module` forward, captured and replayed, against eager.

    This is the assertion the rest of the file exists to reach. The tests above
    drive `_aten_dispatch` by hand, which proves the recorder but not that a
    model *reaches* it in a capturable shape -- two `nn.Linear` layers and a
    `ReLU` do, and the ops that come out are the record.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return
    r = _capture_road_fixture()

    # A Linear is an addmm; the record says so rather than being trusted to.
    assert r["ops"].count("aten.addmm.default") == 2, r["ops"]
    assert "aten.relu.default" in r["ops"], r["ops"]
    # Weights and biases were never declared inputs, so they are constants --
    # four of them, which is `graph_signature`'s lifted-parameter half.
    assert r["n_placeholders"] == 1, r["n_placeholders"]
    assert r["n_constants"] == 4, r["n_constants"]
    assert r["guard"] == {
        "index": 0, "shape": [2, 4], "dtype": "torch.float32", "device": "cpu",
    }, r["guard"]

    # Recording changed nothing about the answer.
    assert r["recorded"] == r["eager_first"], (r["recorded"], r["eager_first"])
    # Replay reproduces it, bit for bit ...
    assert r["replay_same"] == r["recorded"], (r["replay_same"], r["recorded"])
    # ... and agrees with eager on inputs it never saw.
    for replayed, eager in r["pairs"]:
        assert replayed == eager, (replayed, eager)

    # The batch dimension is *not* free. Nothing in this layer makes it so, and
    # a capture that pretended otherwise would be wrong on the first model with
    # a shape-dependent constant in it.
    assert "shape" in r["wrong_batch"], r["wrong_batch"]
    assert "ACCEPTED" not in r["wrong_batch"], r["wrong_batch"]

    # `.item()` behind a Python `if` is refused by name, and the refusal comes
    # at the end rather than in the middle -- the model still ran.
    assert "aten._local_scalar_dense.default" in r["branch_reason"], r["branch_reason"]
    assert "aten._local_scalar_dense.default" in r["branch_refusal"], r["branch_refusal"]
    assert r["active_after"] is False


# --- torch.distributed at world_size 1 (docs/DISTRIBUTED.md) -----------------
#
# `torch.distributed` was off, because `_c10d_init` was one of the names the
# surface deliberately omits (bootstrap.py's "Deliberate omissions"). Turning it
# on is not a one-line switch: `torch/distributed/__init__.py` reaches straight
# into `torch._C._distributed_c10d` and what it needs there is *structure* --
# real enums iterated with `__members__`, nested types, subclassable bases --
# not names. docs/SURFACE_HONESTY.md §2.4 measured that a catch-all answering
# every question still could not finish `import torch`.
#
# The tests below are grouped by what they hold down:
#   * the four `_C` defects the walk uncovered, none of them distributed;
#   * the shape of `_distributed_c10d` itself;
#   * the road through the vendored tree, in a subprocess.


def test_schema_is_mutable_is_a_property_not_a_call():
    """`is_mutable` is a property upstream and was a method here.

    `torch/_library/utils.py:104` is `if schema.is_mutable:` and there are
    fifteen more `._schema.is_mutable` sites in the vendored tree, every one of
    them an attribute read. A bound method is truthy, so the shim answered
    "mutable" for *every* schema and `is_functional_schema` was False
    everywhere -- the always-true predicate this repository has been bitten by
    before. It never showed up because the one test that covered it used the
    shim's own spelling, `is_mutable()`, which works either way.

    `_is_view_op` stays a method: `torch/distributed/tensor/_dispatch.py:569`
    calls it with parentheses.
    """
    mutating = _C.parse_schema("aten::add_.Tensor(Tensor(a!) self, Tensor other) -> Tensor(a!)")
    functional = _C.parse_schema("aten::mm(Tensor self, Tensor mat2) -> Tensor")
    # `is True` / `is False`, not truthiness -- truthiness is what hid this.
    assert mutating.is_mutable is True
    assert functional.is_mutable is False
    assert callable(functional._is_view_op)


def test_the_c10d_functional_namespace_has_real_schemas():
    """`_get_schema` answered every op with an empty argument list.

    `torch/distributed/_functional_collectives.py:637` registers an autograd
    formula for `_c10d_functional::wait_tensor`, and `register_autograd`
    refuses unless the schema is *functional* -- which needs arguments and
    returns to actually be there. These ops live in C++ upstream, so there is
    nowhere else for them to come from.
    """
    wait = _C._get_schema("_c10d_functional::wait_tensor", "")
    assert [a.name for a in wait.arguments] == ["tensor"], wait
    assert len(wait.returns) == 1, wait
    assert wait.is_mutable is False

    # The in-place variant is mutable, so the table is not just "everything
    # is functional".
    inplace = _C._get_schema("_c10d_functional::all_reduce_", "")
    assert inplace.is_mutable is True, inplace

    # And an op that is genuinely absent still refuses rather than inventing
    # an empty schema that would answer "functional" to the same question.
    for qualname in ("_c10d_functional::all_reduce",
                     "_c10d_functional_autograd::all_to_all_single",
                     "_dtensor::shard_dim_alltoall"):
        s = _C._get_schema(qualname, "")
        assert s.arguments, qualname
        assert s.returns, qualname


def test_make_subclass_does_not_advertise_a_signature_the_polyfill_rejects():
    """Upstream's `_make_subclass` is a C builtin with no readable signature.

    `torch/_dynamo/decorators.py:966` wraps `inspect.signature(original_fn)` in
    `except ValueError: pass`, so upstream never runs the comparison at all.
    Ours is a Python function, so the comparison ran and rejected it -- the
    polyfill spells the flag `requires_grad` while the real (measured) keyword
    upstream accepts is `require_grad`, which is also what
    `torch/_C/__init__.pyi:2389` declares.

    Matching the polyfill's spelling would have made the shim accept a keyword
    upstream rejects and reject the one it accepts, so the signature is
    withheld instead -- which is the same amount of information upstream gives.
    """
    import inspect

    sig = inspect.signature(_C.TensorBase._make_subclass)
    kinds = [p.kind for p in sig.parameters.values()]
    assert kinds == [inspect.Parameter.VAR_POSITIONAL,
                     inspect.Parameter.VAR_KEYWORD], sig

    # Withholding the signature must not have changed what it accepts.
    x = _C._VariableFunctions.zeros([2])
    made = _C.TensorBase._make_subclass(_C.TensorBase, x, require_grad=True)
    assert made.requires_grad is True
    assert _C.TensorBase._make_subclass(_C.TensorBase, x).requires_grad is False


def test_default_dtype_answers_with_the_dtype_the_dispatcher_actually_uses():
    """`torch.get_default_dtype()` had no overload-table entry and refused.

    `torch/distributed/_shard/sharded_tensor/metadata.py:20` calls it in a
    dataclass field default, at import time. It is not an aten op -- upstream
    binds it straight onto `_C` -- so overload resolution was never going to
    find it.

    The value is not a free choice: `dtype.rs`'s `default_float()` is what
    factory functions infer, so this asks the dispatcher rather than asserting
    a constant.

    This test asserts the *resting* value. That it can be moved, and that
    moving it moves the inference rules, is
    `test_set_default_dtype_moves_every_rule_that_reads_the_default`.
    """
    assert _C.get_default_dtype() is _C.float32
    # `==`, not `is`. Upstream interns dtypes so `torch.zeros(1).dtype is
    # torch.float32` holds; here a tensor's `.dtype` builds a fresh instance,
    # so the two are equal and not identical. That is a real difference and it
    # is *not* what this test is about -- it predates this change and is
    # recorded here rather than asserted away.
    assert _C._VariableFunctions.zeros([1]).dtype == _C.get_default_dtype()


# The dtypes upstream's `set_default_dtype` accepts, and what each of the two
# refusal branches says. Measured on torch 2.13.0 across every `torch.dtype`
# the module exposes (45 of them); the split is exact, not sampled.
#
#   accepted                       float32 float64 float16 bfloat16
#   TypeError, "only floating-      every non-floating-point tag, the
#     point types are supported       complex and quantised ones included
#     as the default type"
#   TypeError, "couldn't find      the six float8/float4 tags -- they pass
#     storage object <X>Storage"      upstream's floating-point gate and then
#                                     fail its storage-class lookup
#   TypeError, "invalid dtype      anything that is not a `torch.dtype` at
#     object: only floating-point     all, `None` included
#     types are supported as the
#     default type"
_DEFAULT_DTYPE_ACCEPTS = ("float32", "float64", "float16", "bfloat16")
_DEFAULT_DTYPE_NOT_FLOAT = ("int64", "bool", "complex64", "quint8", "bits16", "uint4")
_DEFAULT_DTYPE_NO_STORAGE = (
    ("float8_e4m3fn", "Float8_e4m3fnStorage"),
    ("float8_e5m2", "Float8_e5m2Storage"),
    ("float4_e2m1fn_x2", "Float4_e2m1fn_x2Storage"),
)


def test_set_default_dtype_moves_every_rule_that_reads_the_default():
    """A setter that leaves `torch.ones(3).dtype` alone is worse than a refusal.

    docs/DISTRIBUTED.md §3.4 refused `set_default_dtype` by name because the
    value lived in a Rust `const` and a setter cannot reach one. `transformers`
    ends the argument: `modeling_utils.py:239` (`local_torch_dtype`, entered
    from `from_pretrained` at line 4304) calls `torch.set_default_dtype(dtype)`
    unconditionally, so `from_pretrained` cannot start without it.

    The point of this test is that the global is *load-bearing*. Every site
    listed below reads "the default float dtype" in the shim today, and each is
    asserted to follow -- if any one of them kept its own copy, the setter
    would be a lie in exactly the way the refusal was not.
    """
    assert _C.get_default_dtype() is _C.float32
    try:
        _C._set_default_dtype(_C.float64)
        assert _C.get_default_dtype() is _C.float64

        f64 = _C.float64
        ints = _C._VariableFunctions.zeros([2], dtype=_C.int64)
        # Each entry names the site in the shim that reads the default.
        followers = {
            # aten.rs `empty_like_family` -- ones/zeros/empty
            "ones": _C._VariableFunctions.ones([3]).dtype,
            "zeros": _C._VariableFunctions.zeros([3]).dtype,
            "empty": _C._VariableFunctions.empty([3]).dtype,
            # aten.rs `arange` -- a non-integral endpoint floats the result
            "arange": _C._VariableFunctions.arange(0.0, 3.0).dtype,
            # lib.rs `_tensor_new_from_data` -- `torch.tensor`
            "tensor": _C._tensor_new_from_data([1.0]).dtype,
            # ... including the empty-list case, which has no float in it
            "tensor([])": _C._tensor_new_from_data([]).dtype,
            # aten.rs `full` -- a float fill value
            "full": _C._VariableFunctions.full([2], 1.0).dtype,
            # aten.rs `scalar_tensor`
            "scalar_tensor": _C._aten_dispatch("aten.scalar_tensor.default", 1.5).dtype,
            # aten.rs `rsqrt_default` -- integral in, default float out
            "rsqrt": _C._aten_dispatch("aten.rsqrt.default", ints).dtype,
            # aten.rs `unary_float` -- same rule, shared by cos/sin/exp/tanh
            "cos": _C._aten_dispatch("aten.cos.default", ints).dtype,
            # aten.rs `pow_result_tag` -- integral base, float exponent
            "pow": _C._aten_dispatch("aten.pow.Tensor_Scalar", ints, 2.0).dtype,
            # aten.rs `arith_result_tag`, both halves: a float scalar against
            # an integral tensor, and true division of an integral pair
            "mul.Scalar": _C._aten_dispatch("aten.mul.Scalar", ints, 1.5).dtype,
            "div.Tensor": _C._aten_dispatch("aten.div.Tensor", ints, ints).dtype,
        }
        stayed = {k: str(v) for k, v in followers.items() if v != f64}
        assert not stayed, stayed
        # info.rs `PyFinfo::new` -- `torch.finfo()` with no argument reports the
        # default dtype, and its comment used to cite this setter's absence.
        assert _C.finfo().dtype == "float64", _C.finfo()

        # It moves back, and it moves to the other two upstream accepts.
        for name in _DEFAULT_DTYPE_ACCEPTS:
            want = getattr(_C, name)
            _C._set_default_dtype(want)
            assert _C.get_default_dtype() is want, name
            assert _C._VariableFunctions.ones([1]).dtype == want, name

        # -- what it refuses, in upstream's words ------------------------
        for name in _DEFAULT_DTYPE_NOT_FLOAT:
            try:
                _C._set_default_dtype(getattr(_C, name))
            except TypeError as e:
                assert str(e) == (
                    "only floating-point types are supported as the default type"
                ), (name, str(e))
            else:
                raise AssertionError(f"set_default_dtype accepted torch.{name}")

        # These three pass the floating-point gate upstream and still refuse,
        # on a storage-class lookup. `float8_e4m3fn` is the one candle *can*
        # store, so accepting it would have been a divergence nothing else
        # would have caught.
        for name, storage in _DEFAULT_DTYPE_NO_STORAGE:
            try:
                _C._set_default_dtype(getattr(_C, name))
            except TypeError as e:
                assert str(e) == f"couldn't find storage object {storage}", (
                    name, str(e))
            else:
                raise AssertionError(f"set_default_dtype accepted torch.{name}")

        for bad in (None, 3, "float64", _C.device("cpu")):
            try:
                _C._set_default_dtype(bad)
            except TypeError as e:
                assert str(e) == (
                    "invalid dtype object: only floating-point types are "
                    "supported as the default type"
                ), (bad, str(e))
            else:
                raise AssertionError(f"set_default_dtype accepted {bad!r}")

        # A refused call must not have moved anything.
        assert _C.get_default_dtype() is _C.bfloat16
    finally:
        _C._set_default_dtype(_C.float32)
    assert _C.get_default_dtype() is _C.float32
    assert _C._VariableFunctions.ones([1]).dtype == _C.float32


def test_wait_counter_guard_is_a_context_manager():
    """Every `@_exception_logger` c10d entry point opens one.

    `torch/distributed/c10d_logger.py:96` wraps each public collective in
    `with _WaitCounter(...).guard():`. Nothing here measures anything; the
    requirement is only that the block can be entered.
    """
    with _C._monitor._WaitCounter("pytorch.wait_counter.test").guard():
        pass


def test_reduce_op_is_a_real_enum_that_can_be_walked():
    """`distributed_c10d.py:560` runs `reduce_op = _reduce_op()` at module scope,
    whose `__init__` iterates `ReduceOp.RedOpType.__members__.items()`.

    docs/SURFACE_HONESTY.md §2.4 stopped exactly here: a placeholder that
    answers every attribute still has no `__members__` to walk.
    """
    c10d = _C._distributed_c10d
    members = c10d.ReduceOp.RedOpType.__members__
    assert set(members) == {
        "SUM", "AVG", "PRODUCT", "MIN", "MAX",
        "BAND", "BOR", "BXOR", "PREMUL_SUM", "UNUSED",
    }, sorted(members)
    # Used as a default argument value in seven `def`s, so it is read at import.
    assert c10d.ReduceOp.SUM is members["SUM"]
    assert c10d.ReduceOp(c10d.ReduceOp.SUM) == c10d.ReduceOp.SUM


def test_backend_type_has_the_seven_members_the_table_names():
    """`distributed_c10d.py:320-328` builds `backend_type_map` in a class body,
    naming seven members. Seven, not "at least seven": `FAKE` maps to `CUSTOM`
    rather than to a member of its own.
    """
    bt = _C._distributed_c10d.ProcessGroup.BackendType
    assert set(bt.__members__) == {
        "UNDEFINED", "GLOO", "NCCL", "UCC", "MPI", "XCCL", "CUSTOM",
    }, sorted(bt.__members__)


def test_the_c10d_types_are_real_types():
    """Three separate structural demands, all at import time.

    `class _IllegalWork(Work)` (distributed_c10d.py:2809) needs a subclassable
    base. `_export_c_types()` (line 202) assigns `__module__` on sixteen of
    them. `device_mesh.py:72` evaluates `C10dBackend.Options | None`, which is
    a `TypeError` unless `Options` is a genuine nested type.
    """
    c10d = _C._distributed_c10d

    class _Sub(c10d.Work):
        pass

    assert _Sub().is_completed() is True

    for name in ("AllreduceCoalescedOptions", "AllreduceOptions", "AllToAllOptions",
                 "BarrierOptions", "BroadcastOptions", "GatherOptions", "PrefixStore",
                 "ProcessGroup", "ReduceOp", "ReduceOptions", "ReduceScatterOptions",
                 "ScatterOptions", "Store", "DebugLevel", "Work"):
        obj = getattr(c10d, name)
        obj.__module__ = "torch.distributed.distributed_c10d"
        assert obj.__module__ == "torch.distributed.distributed_c10d", name
    c10d.get_debug_level.__module__ = "torch.distributed.distributed_c10d"

    assert isinstance(c10d.Backend.Options, type)
    assert c10d.Backend.Options | None is not None


def test_process_group_can_take_the_opaque_metaclass():
    """`device_mesh._register_distributed_opaque_types` demands that
    `ProcessGroup` carry `torch._opaque_base.OpaqueBaseMeta`, which upstream's
    pybind class does (measured on 2.13.0).

    `_C` cannot import the tree at its own import time, so the binding is late
    -- the same shape as `_C._set_generator_metaclass`, pulled rather than
    pushed, and `_c10d_init()` is where the tree asks for it. What this test
    holds down is the precondition: `__class__` assignment refuses when the old
    metaclass is `type`, because `type` is a static type.
    """
    pg = _C._distributed_c10d.ProcessGroup
    assert type(pg) is not type, "ProcessGroup's metaclass must be a heap type"
    assert issubclass(type(pg), type)

    class _Meta(type):
        pass

    original = type(pg)
    try:
        pg.__class__ = _Meta
    finally:
        pg.__class__ = original


def test_the_store_is_a_real_key_value_store():
    """At world_size 1 the store is not a stand-in for anything -- this process
    is the only writer and the only reader, so a dict *is* the rendezvous.
    """
    c10d = _C._distributed_c10d
    s = c10d.HashStore()
    s.set("k", "v")
    assert s.get("k") == b"v"
    assert s.add("counter", 3) == 3
    assert s.add("counter", 4) == 7
    assert s.check(["k"]) is True and s.check(["nope"]) is False
    assert s.num_keys() == 2
    assert s.delete_key("k") is True and s.check(["k"]) is False

    p = c10d.PrefixStore("pfx", s)
    p.set("a", "1")
    assert p.get("a") == b"1"
    assert s.get("pfx/a") == b"1"


def test_the_store_refuses_to_wait_for_a_rank_that_cannot_exist():
    """A silent no-op `wait` is the failure mode this repository named in
    docs/CKPT.md: it looks like it worked. At world_size 1 nobody else can ever
    set the key, so waiting is not "not yet", it is "never".
    """
    s = _C._distributed_c10d.HashStore()
    s.set("here", "1")
    s.wait(["here"])  # present: returns
    try:
        s.wait(["never"])
    except RuntimeError as e:
        assert "never" in str(e), e
    else:
        raise AssertionError("Store.wait silently returned for an absent key")


def test_transports_that_need_a_peer_refuse_by_name():
    """DESIGN.md §6. `TCPStore` cannot be honest at world_size 1 -- there is
    nothing at the other end of the socket -- so it says so instead of
    pretending to connect.
    """
    try:
        _C._distributed_c10d.TCPStore("127.0.0.1", 29500, 1, True)
    except NotImplementedError as e:
        assert "TCPStore" in str(e), e
    else:
        raise AssertionError("TCPStore pretended to connect")


# --- the road through the vendored tree (docs/DISTRIBUTED.md) ---------------
#
# Everything above drives `_C` directly. What this section holds down is the
# thing the work was for: that the *tree* gets somewhere it could not get
# before. It runs in a subprocess with `torchnative/src/main` on PYTHONPATH,
# the same two-interpreter shape as the checkpoint, device, meta and capture
# roads above and for the same reason (see that section's comment).

_DIST_ROAD_SCRIPT = r"""
import json, sys, traceback

out = {}

def step(name, code):
    try:
        exec(code, globals())
    except BaseException:
        out[name] = "FAILED: " + traceback.format_exc(limit=0).strip().splitlines()[-1]
    else:
        out[name] = "OK"

import torch
import torch.distributed as dist
import torch.distributed.distributed_c10d as c10d

out["is_available"] = dist.is_available()
out["has_Store"] = hasattr(dist, "Store")
# The wire backends must stay *absent*, so the tree's own availability flags
# come out False and it refuses by name rather than reaching for a hollow
# object. docs/SURFACE_HONESTY.md 2.4's regression lives here.
out["gloo_available"] = c10d._GLOO_AVAILABLE
out["nccl_available"] = c10d._NCCL_AVAILABLE
out["mpi_available"] = c10d._MPI_AVAILABLE
out["ucc_available"] = c10d._UCC_AVAILABLE
try:
    dist.init_process_group(backend="gloo", rank=0, world_size=1,
                            store=dist.HashStore())
except RuntimeError as e:
    out["gloo_refusal"] = str(e)
else:
    out["gloo_refusal"] = "ACCEPTED"

step("fsdp", "import torch.distributed.fsdp")
step("transformers", "import transformers")
step("AutoModelForCausalLM", "from transformers import AutoModelForCausalLM")
step("LlamaForCausalLM",
     "from transformers.models.llama.modeling_llama import LlamaForCausalLM")

import torchnative.distributed as tnd
out["backend_name"] = tnd.BACKEND_NAME
out["registered"] = tnd.BACKEND_NAME.upper() in dist.Backend._plugins

dist.init_process_group(backend=tnd.BACKEND_NAME, rank=0, world_size=1,
                        store=dist.HashStore())
out["rank"] = dist.get_rank()
out["world_size"] = dist.get_world_size()
out["backend"] = dist.get_backend()

def record(name, fn):
    try:
        out[name] = fn()
    except NotImplementedError as e:
        out[name] = "REFUSED: " + str(e).split(":")[0]
    except BaseException as e:
        out[name] = "%s: %s" % (type(e).__name__, e)

def src():
    return torch.tensor([1.0, -2.0, 3.5])

for op_name in ("SUM", "AVG", "PRODUCT", "MIN", "MAX"):
    def run(op_name=op_name):
        x = src()
        dist.all_reduce(x, op=getattr(dist.ReduceOp, op_name))
        return x.tolist()
    record("all_reduce_" + op_name, run)

record("all_reduce_PREMUL_SUM", lambda: (
    lambda x: (dist.all_reduce(x, op=dist.ReduceOp.PREMUL_SUM), x.tolist())[1])(src()))
record("broadcast", lambda: (
    lambda x: (dist.broadcast(x, src=0), x.tolist())[1])(src()))
record("reduce", lambda: (
    lambda x: (dist.reduce(x, dst=0), x.tolist())[1])(src()))
record("barrier", lambda: dist.barrier())

def all_gather():
    buf = [torch.zeros(3)]
    dist.all_gather(buf, src())
    return [t.tolist() for t in buf]
record("all_gather", all_gather)

def all_gather_single():
    buf = torch.zeros(3)
    dist.all_gather_single(buf, src())
    return buf.tolist()
record("all_gather_single", all_gather_single)

def gather():
    buf = [torch.zeros(3)]
    dist.gather(src(), buf, dst=0)
    return [t.tolist() for t in buf]
record("gather", gather)

def scatter():
    buf = torch.zeros(3)
    dist.scatter(buf, [src()], src=0)
    return buf.tolist()
record("scatter", scatter)

def reduce_scatter():
    buf = torch.zeros(3)
    dist.reduce_scatter(buf, [src()])
    return buf.tolist()
record("reduce_scatter", reduce_scatter)

def reduce_scatter_single():
    buf = torch.zeros(3)
    dist.reduce_scatter_single(buf, src())
    return buf.tolist()
record("reduce_scatter_single", reduce_scatter_single)

def all_to_all_single():
    buf = torch.zeros(3)
    dist.all_to_all_single(buf, src())
    return buf.tolist()
record("all_to_all_single", all_to_all_single)

record("send_to_1", lambda: dist.send(src(), dst=1))
record("recv_from_1", lambda: dist.recv(src(), src=1))

# A store that is asked to wait on a key nobody can ever write.
try:
    s = dist.HashStore()
    s.wait(["nobody-will-write-this"])
except RuntimeError as e:
    out["store_wait"] = "REFUSED"
else:
    out["store_wait"] = "RETURNED"

json.dump(out, sys.stdout)
"""


@functools.cache
def _dist_road_fixture():
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    proc = subprocess.run(
        [sys.executable, "-c", _DIST_ROAD_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"distributed-road subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def test_distributed_is_on_and_the_wire_backends_are_still_off():
    """Both halves of the switch, because only having one is the bug.

    docs/SURFACE_HONESTY.md §2.4 turned `_c10d_init` on once before and
    `import torch` stopped. It gets on the road now -- and the second half
    matters just as much: `ProcessGroupGloo` and its siblings must stay
    *absent*, or `distributed_c10d.py:220` sets `_GLOO_AVAILABLE = True` and
    `init_process_group(backend="gloo")` reaches for something hollow instead
    of getting the tree's own "doesn't have Gloo built in".

    That is not hypothetical: the first version of this work synthesised them,
    twice -- once through the module catch-all and once through
    `_SubmoduleFinder`, which answered the same name as an empty *module* after
    the attribute lookup had correctly refused.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return  # vendor tree not installed -- see vendor/install_shim.sh
    r = _dist_road_fixture()
    assert r["is_available"] is True
    assert r["has_Store"] is True
    for flag in ("gloo_available", "nccl_available", "mpi_available", "ucc_available"):
        assert r[flag] is False, (flag, r[flag])
    assert "doesn't have Gloo built in" in r["gloo_refusal"], r["gloo_refusal"]


def test_the_road_reaches_transformers():
    """The thing this was for.

    `import transformers` on its own never needed torch (IMPORT_WALLS.md), so
    it is not the measurement -- `AutoModelForCausalLM` is, because that is
    what pulls `GenerationMixin` and with it `torch._dynamo`, `fsdp`, DTensor
    and the functional collectives. DESIGN.md §11.1 recorded this as the wall
    that stopped step 1.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    r = _dist_road_fixture()
    for step_name in ("fsdp", "transformers", "AutoModelForCausalLM",
                      "LlamaForCausalLM"):
        assert r[step_name] == "OK", (step_name, r[step_name])


def test_world_size_one_collectives_agree_with_upstream_gloo():
    """The values are not asserted from first principles -- they were measured.

    The same script was run against upstream torch 2.13.0 with `backend="gloo"`
    at `world_size=1`, and every value-producing collective below came back
    identical. The three that differ are the three that *fail on both sides*:
    upstream fails PREMUL_SUM with a pybind argument error and send/recv with
    `IndexError: vector`, and this build refuses all three by name.

    `[1.0, -2.0, 3.5]` throughout is the input, so this is a statement that a
    world of one changes nothing -- which is exactly the claim, and it is
    checked rather than assumed. A collective that silently did nothing at all
    would produce the same answer for the reductions and `[0, 0, 0]` for the
    gather/scatter family, which is why those are in the list.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    r = _dist_road_fixture()
    assert r["registered"] is True
    assert (r["rank"], r["world_size"], r["backend"]) == (0, 1, "local"), r

    unchanged = [1.0, -2.0, 3.5]
    for key in ("all_reduce_SUM", "all_reduce_AVG", "all_reduce_PRODUCT",
                "all_reduce_MIN", "all_reduce_MAX", "broadcast", "reduce"):
        assert r[key] == unchanged, (key, r[key])
    # These moved data into a zeroed buffer, so agreeing is not the same as
    # doing nothing.
    for key in ("all_gather_single", "scatter", "reduce_scatter",
                "reduce_scatter_single", "all_to_all_single"):
        assert r[key] == unchanged, (key, r[key])
    for key in ("all_gather", "gather"):
        assert r[key] == [unchanged], (key, r[key])
    assert r["barrier"] is None


def test_what_needs_a_peer_refuses_by_name():
    """DESIGN.md §6, and the specific accident docs/CKPT.md recorded.

    `send`/`recv` cannot be made true by any amount of local work, and a
    `recv` that quietly did nothing would leave the caller reading an unwritten
    buffer. `PREMUL_SUM` is the subtler one: every other reduction is the
    identity at world_size 1, and PREMUL_SUM is not -- it is `factor * x` --
    so treating it like the others would be a wrong answer wearing the shape of
    a right one.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    r = _dist_road_fixture()
    assert r["send_to_1"].startswith("REFUSED"), r["send_to_1"]
    assert "send" in r["send_to_1"], r["send_to_1"]
    assert r["recv_from_1"].startswith("REFUSED"), r["recv_from_1"]
    assert "recv" in r["recv_from_1"], r["recv_from_1"]
    assert r["all_reduce_PREMUL_SUM"].startswith("REFUSED"), r["all_reduce_PREMUL_SUM"]
    assert "PREMUL_SUM" in r["all_reduce_PREMUL_SUM"], r["all_reduce_PREMUL_SUM"]
    assert r["store_wait"] == "REFUSED", r["store_wait"]


# --- printing a tensor (docs/E2E_REAL.md) -----------------------------------
#
# `print(tensor)` was the one thing docs/WHEEL.md §5 recorded the built wheel
# could not do, and it was not a packaging fault: `torch/_tensor_str.py` walks
# a surface this shim had holes in. The spec for closing it is not a guess --
# it is a `TorchDispatchMode` logger placed *inside* `_str`'s
# `_disable_current_modes()` guard on upstream torch 2.13.0, over fourteen
# tensors chosen to reach every branch of `_str_intern` (empty, 0-d, integral,
# bool, summarised, sci-mode, all-zero, inf/nan, requires_grad). Upstream's
# `repr` dispatches exactly 21 aten ops across those; fifteen already had
# kernels here and six did not:
#
#     aten.abs.default   aten.ceil.default    aten.gt.Scalar
#     aten.min.default   aten.unbind.int      aten.masked_select.default
#
# `torch.isfinite` is *not* on that list, and that is the measurement paying
# for itself: upstream's `isfinite` is `CompositeImplicitAutograd`, so the
# logger sees it as `eq.Tensor`/`abs`/`ne.Scalar`/`bitwise_and.Tensor` and it
# belongs in `_install_composites`, not in the kernel table. Naming it a
# kernel would have invented a work item upstream does not have -- the same
# mistake `overloads.json`'s note about `layer_norm` describes.


_REPR_CASES = [
    ("2d float", "torch.ones(3, 2) * 4.0"),
    ("0d", "torch.ones(()) * 1.5"),
    ("1d empty", "torch.ones(0)"),
    ("2d empty", "torch.ones(0, 3)"),
    ("int64", "torch.arange(0, 5)"),
    ("bool", "torch.arange(0, 4).ne(2)"),
    ("summarised", "torch.arange(0, 2000).to(torch.float32)"),
    ("3d", "torch.ones(2, 3, 4)"),
    ("fractional", "torch.arange(0, 6).to(torch.float32) / 7.0 - 0.4"),
    ("sci-mode high", "torch.arange(0, 4).to(torch.float32) * 1.0e9"),
    ("sci-mode low", "torch.arange(1, 5).to(torch.float32) * 1.0e-7"),
    ("all zeros", "torch.zeros(3, 3)"),
    ("inf and nan", "torch.tensor([float('inf'), float('nan'), -float('inf'), 1.0])"),
    ("requires_grad", "(torch.ones(2, 2) * 2.0).requires_grad_(True)"),
    ("wide int64", "torch.arange(0, 3) * 123456789"),
    ("negative int64", "torch.arange(0, 5) - 2"),
    ("mm result", "torch.ops.aten.mm.default(torch.ones(3, 4), torch.ones(4, 2))"),
]

_REPR_ROAD_SCRIPT = r"""
import json, sys, traceback
import torch

CASES = %r
out = {}
for label, expr in CASES:
    try:
        out[label] = repr(eval(expr))
    except BaseException:
        out[label] = "RAISED: " + traceback.format_exc(limit=0).strip().splitlines()[-1]
json.dump(out, sys.stdout)
""" % (_REPR_CASES,)


@functools.lru_cache(maxsize=1)
def _repr_road_fixture():
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    proc = subprocess.run(
        [sys.executable, "-c", _REPR_ROAD_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"repr-road subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def test_repr_matches_upstream_character_for_character():
    """Working and correct are different things.

    The comparison is the *string*, not "did it come back without raising".
    Tensor formatting is where a shim looks finished and is not: the value
    can be right while the width, the precision, the `...` elision, the
    `dtype=` suffix or the line breaks are wrong, and every one of those is a
    thing a user reads. So the expected side is upstream torch 2.13.0 in this
    interpreter, the actual side is the vendored tree in a subprocess, and
    they must be equal with `==`.

    The case list is the one the dispatch trace was taken over, so each entry
    is here because it reaches a branch: `_tensor_str` elides above
    `PRINT_OPTS.threshold`, switches to scientific notation on the
    max/min ratio, prints `dtype=` only when it is not the default, prints
    `size=` only for an empty tensor whose shape is not `(0,)`, and appends
    `requires_grad=True` from the autograd surface rather than from the
    formatter.
    """
    if not _ckpt_shim_available():
        return
    got = _repr_road_fixture()
    mismatches = []
    for label, expr in _REPR_CASES:
        expected = repr(eval(expr, {"torch": _upstream_torch}))
        actual = got[label]
        if actual != expected:
            mismatches.append(f"[{label}] {expr}\n  upstream: {expected!r}\n  shim:     {actual!r}")
    if mismatches:
        raise AssertionError(
            "repr() differs from upstream torch 2.13.0 in "
            f"{len(mismatches)}/{len(_REPR_CASES)} cases:\n" + "\n".join(mismatches)
        )


def test_the_alternative_representations_have_no_constructors():
    """Why `is_sparse` and its family are allowed to be one answer.

    Six of the predicates `repr` reads -- `is_nested`, `is_sparse`,
    `is_quantized`, `_is_zerotensor`, `is_neg` and `_is_functional_tensor` --
    ask "which representation is this tensor?", and this build has exactly
    one: candle's dense strided buffer, or the shape-and-dtype-only `Meta`
    arm. So they answer `False` today, and CLAUDE.md §5.5 is right to be
    suspicious of that: a predicate that cannot say anything else is not a
    predicate.

    What makes it a fact rather than a constant is *this* test. Each of these
    representations has exactly one way into existence, and every one of those
    ways refuses by name. If any of them ever lands, this test fails, and the
    predicate that quietly said `False` becomes a lie that something noticed.
    That is the difference between an invariant and an assumption -- the
    `is_mutable` accident in docs/DISTRIBUTED.md §8.1 is what an unguarded one
    looks like.

    `layout` is in the same family and is checked here for the same reason:
    it reports `torch.strided`, and the sparse layouts have no constructor.
    """
    if not _ckpt_shim_available():
        return
    r = _repr_predicate_fixture()
    for maker in (
        "torch.sparse_coo_tensor",
        "torch.sparse_csr_tensor",
        "torch._efficientzerotensor",
        "torch._neg_view",
        "torch._to_functional_tensor",
        "torch.quantize_per_tensor",
        "torch.nested.nested_tensor",
        "torch._nested_tensor_from_tensor_list",
        "torch.Tensor.to_sparse",
        "torch._C._functorch._wrap_for_grad",
        "torch._C._functorch._add_batch_dim",
        "torch._C._functorch._vmap_increment_nesting",
    ):
        assert r["makers"][maker].startswith("REFUSED"), (maker, r["makers"][maker])


def test_the_repr_predicates_are_derived_not_asserted():
    """Each of the eleven, and where its answer comes from.

    Three groups, three different kinds of grounds:

    * **Device.** `is_mps`, `is_xpu` and `is_maia` are `device.type == ...`,
      the same derivation `is_cpu`/`is_cuda`/`is_meta` already use in
      tensor.rs. They are not constants at all -- `torch.zeros(2,
      device="meta").is_meta` is `True` on this build -- so the test asserts
      agreement with `.device`, which is what would break if one drifted.

    * **Representation.** `is_nested`/`is_sparse`/`is_quantized`/
      `_is_zerotensor`/`is_neg`/`layout` are an exhaustive `match` over the
      shim's `Repr` enum, so adding an arm to it fails the build rather than
      inheriting a `False`. The constructor guard above is the other half.

    * **A real, empty stack.** `is_functorch_wrapped_tensor` is not a constant
      here either: it is `maybe_get_level(t) != -1`, and `maybe_get_level`
      reads the functorch dynamic layer stack, whose depth is 0 because
      everything that pushes onto it refuses. Upstream agrees on both halves
      -- measured on 2.13.0, `maybe_get_level(plain) == -1` outside `vmap`
      and `1` inside it.
    """
    if not _ckpt_shim_available():
        return
    r = _repr_predicate_fixture()
    # Device-derived: the three new ones agree with `.device`, and the meta
    # tensor proves the derivation is live rather than a fixed False.
    assert r["cpu_device_type"] == "cpu"
    assert r["cpu_is_mps"] is False
    assert r["cpu_is_xpu"] is False
    assert r["cpu_is_maia"] is False
    assert r["cpu_is_cpu"] is True
    assert r["meta_is_meta"] is True, "is_meta must follow the device, not a constant"
    assert r["meta_is_cpu"] is False
    # Representation-derived.
    assert r["is_nested"] is False
    assert r["is_sparse"] is False
    assert r["is_quantized"] is False
    assert r["is_zerotensor"] is False
    assert r["is_neg"] is False
    assert r["layout"] == "torch.strided"
    assert r["is_functional"] is False
    # The functorch stack is real and empty, and the predicate reads it.
    assert r["functorch_stack_depth"] == 0
    assert r["maybe_get_level"] == -1
    assert r["is_functorch_wrapped"] is False


_REPR_PREDICATE_SCRIPT = r"""
import json, sys, traceback
import torch

out = {}
t = torch.ones(2, 2)
m = torch.zeros(2, 2, device="meta")

out["cpu_device_type"] = t.device.type
out["cpu_is_mps"] = t.is_mps
out["cpu_is_xpu"] = t.is_xpu
out["cpu_is_maia"] = t.is_maia
out["cpu_is_cpu"] = t.is_cpu
out["meta_is_meta"] = m.is_meta
out["meta_is_cpu"] = m.is_cpu
out["is_nested"] = t.is_nested
out["is_sparse"] = t.is_sparse
out["is_quantized"] = t.is_quantized
out["is_zerotensor"] = t._is_zerotensor()
out["is_neg"] = t.is_neg()
out["layout"] = str(t.layout)
out["is_functional"] = torch._is_functional_tensor(t)
out["functorch_stack_depth"] = torch._C._functorch.get_dynamic_layer_stack_depth()
out["maybe_get_level"] = torch._C._functorch.maybe_get_level(t)
out["is_functorch_wrapped"] = torch._C._functorch.is_functorch_wrapped_tensor(t)

makers = {}
def maker(name, fn):
    try:
        fn()
    except NotImplementedError as e:
        makers[name] = "REFUSED: " + str(e)[:100]
    except BaseException as e:
        makers[name] = "%s: %s" % (type(e).__name__, str(e)[:100])
    else:
        makers[name] = "MADE"

idx = torch.arange(0, 2).reshape(1, 2)
maker("torch.sparse_coo_tensor", lambda: torch.sparse_coo_tensor(idx, torch.ones(2)))
maker("torch.sparse_csr_tensor",
      lambda: torch.sparse_csr_tensor(torch.arange(0, 3), torch.arange(0, 2), torch.ones(2)))
maker("torch._efficientzerotensor", lambda: torch._efficientzerotensor((2, 2)))
maker("torch._neg_view", lambda: torch._neg_view(t))
maker("torch._to_functional_tensor", lambda: torch._to_functional_tensor(t))
maker("torch.quantize_per_tensor", lambda: torch.quantize_per_tensor(t, 0.1, 0, torch.qint8))
maker("torch.nested.nested_tensor", lambda: torch.nested.nested_tensor([t]))
maker("torch._nested_tensor_from_tensor_list",
      lambda: torch._nested_tensor_from_tensor_list([t]))
maker("torch.Tensor.to_sparse", lambda: t.to_sparse())
maker("torch._C._functorch._wrap_for_grad", lambda: torch._C._functorch._wrap_for_grad(t, 0))
maker("torch._C._functorch._add_batch_dim", lambda: torch._C._functorch._add_batch_dim(t, 0, 1))
maker("torch._C._functorch._vmap_increment_nesting",
      lambda: torch._C._functorch._vmap_increment_nesting(1, "error"))
out["makers"] = makers
json.dump(out, sys.stdout)
"""


@functools.lru_cache(maxsize=1)
def _repr_predicate_fixture():
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _REPR_PREDICATE_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"repr-predicate subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def test_the_six_repr_kernels_dispatch():
    """The kernels themselves, at the `_C` door.

    `tools/golden/cases.py` is what checks these against upstream value by
    value; what this holds down is that the dispatcher *reaches* them under
    the exact keys the trace named, because a kernel registered under a key
    nothing resolves to is invisible to both.
    """
    for op in (
        "aten.abs.default",
        "aten.ceil.default",
        "aten.gt.Scalar",
        "aten.gt.Tensor",
        "aten.masked_select.default",
        "aten.min.default",
        "aten.unbind.int",
    ):
        assert op in _C._aten_all_implemented(), op

    x = _C._tensor_from_flat([-2.0, -0.5, 0.0, 3.25], [4], _C.float32)
    assert _C._aten_dispatch("aten.abs.default", x).tolist() == [2.0, 0.5, 0.0, 3.25]
    assert _C._aten_dispatch("aten.ceil.default", x).tolist() == [-2.0, -0.0, 0.0, 4.0]
    assert _C._aten_dispatch("aten.gt.Scalar", x, 0.0).tolist() == [False, False, False, True]
    assert _C._aten_dispatch("aten.min.default", x).tolist() == -2.0

    mask = _C._tensor_from_flat([1, 0, 0, 1], [4], _C.bool)
    assert _C._aten_dispatch("aten.masked_select.default", x, mask).tolist() == [-2.0, 3.25]

    y = _C._tensor_from_flat([1.0, 2.0, 3.0, 4.0], [2, 2], _C.float32)
    rows = _C._aten_dispatch("aten.unbind.int", y, 0)
    assert [r.tolist() for r in rows] == [[1.0, 2.0], [3.0, 4.0]]
    cols = _C._aten_dispatch("aten.unbind.int", y, 1)
    assert [c.tolist() for c in cols] == [[1.0, 3.0], [2.0, 4.0]]

    z = _C._tensor_from_flat([1.0, 5.0], [2], _C.float32)
    w = _C._tensor_from_flat([3.0, 3.0], [2], _C.float32)
    assert _C._aten_dispatch("aten.gt.Tensor", z, w).tolist() == [False, True]


def test_cat_skips_a_tensor_of_shape_zero_and_only_that_shape():
    """torch's "legacy empty" rule, and the line it is drawn at.

    `transformers`' KV cache is `torch.tensor([])` until the first decoder
    step and then `torch.cat([self.keys, key_states], dim=-2)`
    (`cache_utils.py:144`), so the very first attention layer of every model
    concatenates a 1-D empty against a 4-D tensor. This shim raised
    `IndexError` there and the forward pass stopped -- docs/E2E_REAL.md.

    The four assertions below are the rule's *edges*, measured on 2.13.0,
    because "skip empty tensors" is the plausible over-generalisation and it
    is wrong: `torch.ones(0, 5)` is empty and is **not** skipped.
    """
    empty = _C._tensor_from_flat([], [0], _C.float32)
    body = _C._tensor_from_flat([float(i) for i in range(24)], [1, 2, 3, 4], _C.float32)

    joined = _C._aten_dispatch("aten.cat.default", [empty, body], -2)
    assert tuple(joined.shape) == (1, 2, 3, 4), joined.shape
    assert joined.tolist() == body.tolist()

    # Every entry skipped: `(0,)` back, and `dim` is never looked at.
    both = _C._aten_dispatch("aten.cat.default", [empty, empty], 5)
    assert tuple(both.shape) == (0,), both.shape

    # A *non-empty* 1-D entry does not get the exemption.
    one_d = _C._tensor_from_flat([1.0, 2.0], [2], _C.float32)
    try:
        _C._aten_dispatch("aten.cat.default", [one_d, body], -2)
    except IndexError:
        pass
    else:
        raise AssertionError("cat must not skip a non-empty 1-D tensor")

    # Empty but not `(0,)`: upstream refuses on the rank mismatch, so this
    # must not be skipped either.
    wide_empty = _C._tensor_from_flat([], [0, 5], _C.float32)
    try:
        _C._aten_dispatch("aten.cat.default", [wide_empty, body], -2)
    except (RuntimeError, IndexError):
        pass
    else:
        raise AssertionError("cat must not skip an empty tensor of shape (0, 5)")


def test_autocast_is_off_and_cannot_be_turned_on():
    """`torch._C.is_autocast_enabled` -- the wall docs/DISTRIBUTED.md §7 named.

    `transformers/utils/generic.py:250` opens `maybe_autocast` with
    `if torch.is_autocast_enabled(device_type) or enabled:`, and
    `modeling_llama.py:121` wraps the rotary embedding in it. So the first
    forward pass of the first real `transformers` model this project ever
    built stopped on this name.

    **The answer is not a constant, and the refusal beside it is why.**
    Autocast is a dispatch key that casts an op's inputs to a lower precision;
    this shim has no such key, so a `True` here would promise casting that
    never happens and hand back `float32` results claiming to be `bfloat16`
    ones. `is_autocast_enabled` therefore reads a real flag, and
    `set_autocast_enabled(device_type, True)` **refuses by name** -- the flag
    cannot be raised, so the read is derived rather than asserted, the same
    shape as the functorch stack in `_install_repr_surface`.

    Setting it to `False` is accepted, because that is already true.
    """
    if not _ckpt_shim_available():
        return
    r = _forward_road_fixture()
    assert r["autocast_cpu"] is False
    assert r["autocast_enable_refused"].startswith("REFUSED"), r["autocast_enable_refused"]
    assert "autocast" in r["autocast_enable_refused"], r["autocast_enable_refused"]
    assert r["autocast_disable_ok"] == "OK", r["autocast_disable_ok"]
    # ... and upstream agrees about the value outside an autocast block.
    assert _upstream_torch._C.is_autocast_enabled("cpu") is False


def test_is_tracing_agrees_with_the_tracing_state():
    """Two names, one fact.

    `torch/jit/_trace.py:1269` asks `_is_tracing()`; `torch/_tensor.py:1186`
    asks `_get_tracing_state()`. Upstream's `_is_tracing()` is
    "is there a tracing state?", and `_get_tracing_state` was already
    answering `None` here (see `_DISCOVERED_RETURNS`). So this is derived from
    that one rather than being a second constant that could drift away from
    it -- which is the whole content of the assertion below.
    """
    if not _ckpt_shim_available():
        return
    r = _forward_road_fixture()
    assert r["tracing_state"] is None
    assert r["is_tracing"] is False
    assert r["is_tracing"] == (r["tracing_state"] is not None)
    assert _upstream_torch._C._is_tracing() is False


def test_a_real_transformers_llama_forward_matches_upstream():
    """The thing this round was for.

    Every model comparison in this file until now used a decoder **this file
    transcribes op by op** (`_e2e_build`/`_e2e_forward`). That proves the
    kernels and proves nothing about the tree: a transcription can agree with
    upstream while `transformers`' own `LlamaForCausalLM` cannot run at all,
    and until this round it could not -- docs/DISTRIBUTED.md §8 item 1 lists
    the forward pass as untried.

    This one builds the real thing on both sides:
    `AutoModelForCausalLM.from_config(LlamaConfig(...))`, the same
    `transformers` 5.15.1 in both interpreters, differing only in which
    `torch` is underneath. Weights are pushed in through `load_state_dict`
    from an RNG-free generator, so neither side depends on the other's
    random stream, and both `state_dict` key order and shapes come from
    `transformers` rather than from this file.

    Tokens *and* logits, for the reason the comment above
    `_E2E_LOGIT_ATOL` gives: docs/ARCH.md §5.1 measured a real bug that
    produced identical greedy tokens with logits 379x further apart than the
    correct kernel's.

    **`_E2E_LOGIT_ATOL` is the wrong bound for this test, and the one used
    below was measured rather than chosen.** That constant is 1e-5
    *absolute*, sized for the 64-wide, 100-vocab decoder the rest of this file
    compares. This model is 16-wide with a 32-token vocabulary and its logits
    sit around 0.05, so 1e-5 is roughly 20% of a logit: the assertion could
    not fail. Measured, by scaling `aten.silu.default`'s output on the shim
    side only and re-running this comparison (docs/E2E_REAL.md):

        clean                        2.24e-08
        silu x 1.0001 (0.01% high)   1.42e-07    under 1e-5 -- not caught
        silu x 1.001  (0.1%  high)   1.44e-06    under 1e-5 -- not caught

    So the bound here is 5e-7: 22x above the clean measurement, and it does
    catch the 0.1% error the file-wide constant lets through. It does **not**
    catch the 0.01% one, and that is said rather than hidden -- a model this
    small dilutes a single activation error, and the answer to that is a
    bigger model, not a tighter number.
    """
    if not _ckpt_shim_available() or _upstream_transformers is None:
        return
    r = _forward_road_fixture()
    assert r["forward"] == "OK", r["forward"]
    expected = _upstream_llama_logits()
    assert r["logits_shape"] == expected["shape"], (r["logits_shape"], expected["shape"])
    assert r["argmax"] == expected["argmax"], (r["argmax"], expected["argmax"])
    max_diff = max(abs(a - b) for a, b in zip(r["logits"], expected["logits"]))
    assert max_diff < _REAL_LLAMA_ATOL, max_diff
    # And the logits are printable, which is the other half of this round.
    assert r["logits_repr"].startswith("tensor("), r["logits_repr"][:80]


# See the docstring above for the three measurements this comes from.
_REAL_LLAMA_ATOL = 5e-7


try:
    import transformers as _upstream_transformers
except Exception:  # pragma: no cover - transformers is not a test dependency
    _upstream_transformers = None


# The config is small on purpose -- big enough to exercise attention, rotary
# embeddings, the MLP and the LM head, small enough that `load_state_dict`
# from a Python list is not the thing being measured.
_LLAMA_CFG = dict(
    vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
    num_attention_heads=2, num_key_value_heads=2, max_position_embeddings=32,
    tie_word_embeddings=False,
)
_LLAMA_IDS = [3, 7, 1, 19]

# Shared by both interpreters, as source text: the two sides must fill the
# weights by *the same procedure*, not by two transcriptions of one idea.
_LLAMA_FILL = r'''
def _fill(model, torch):
    sd = model.state_dict()
    new = {}
    for i, key in enumerate(sorted(sd)):
        ref = sd[key]
        n = 1
        for d in ref.shape:
            n *= int(d)
        state = (i + 1) * 7919
        vals = []
        for _ in range(n):
            state = (state * 1103515245 + 12345) % 2147483648
            vals.append(round(((state / 2147483648.0) * 2.0 - 1.0) * 0.2, 6))
        t = torch.tensor(vals, dtype=torch.float32)
        new[key] = t.reshape(list(int(d) for d in ref.shape)) if len(ref.shape) != 1 else t
    model.load_state_dict(new)
    return model
'''


@functools.lru_cache(maxsize=1)
def _upstream_llama_logits():
    """The expected side, in this interpreter, on upstream torch."""
    torch = _upstream_torch
    from transformers import AutoModelForCausalLM
    from transformers.models.llama.configuration_llama import LlamaConfig

    ns = {}
    exec(_LLAMA_FILL, ns)
    model = AutoModelForCausalLM.from_config(LlamaConfig(**_LLAMA_CFG))
    model.eval()
    ns["_fill"](model, torch)
    with torch.no_grad():
        logits = model(torch.tensor([_LLAMA_IDS])).logits
    flat = _e2e_flatten(logits.tolist())
    return {
        "shape": list(int(d) for d in logits.shape),
        "logits": flat,
        "argmax": [int(x) for x in logits[0].argmax(-1).tolist()],
    }


_FORWARD_ROAD_SCRIPT = r"""
import json, sys, traceback
import torch

out = {}

out["autocast_cpu"] = torch._C.is_autocast_enabled("cpu")
try:
    torch._C.set_autocast_enabled("cpu", True)
except NotImplementedError as e:
    out["autocast_enable_refused"] = "REFUSED: " + str(e)[:160]
except BaseException as e:
    out["autocast_enable_refused"] = "%s: %s" % (type(e).__name__, str(e)[:120])
else:
    out["autocast_enable_refused"] = "ACCEPTED"
try:
    torch._C.set_autocast_enabled("cpu", False)
except BaseException as e:
    out["autocast_disable_ok"] = "%s: %s" % (type(e).__name__, str(e)[:120])
else:
    out["autocast_disable_ok"] = "OK"

out["tracing_state"] = torch._C._get_tracing_state()
out["is_tracing"] = torch._C._is_tracing()

FILL = __FILL__
CFG = __CFG__
IDS = __IDS__
try:
    from transformers import AutoModelForCausalLM
    from transformers.models.llama.configuration_llama import LlamaConfig
    ns = {}
    exec(FILL, ns)
    model = AutoModelForCausalLM.from_config(LlamaConfig(**CFG))
    model.eval()
    ns["_fill"](model, torch)
    out["model_class"] = type(model).__name__
    with torch.no_grad():
        logits = model(torch.tensor([IDS])).logits
except BaseException:
    out["forward"] = "FAILED: " + traceback.format_exc(limit=4)
else:
    out["forward"] = "OK"
    out["logits_shape"] = [int(d) for d in logits.shape]
    def flat(v):
        if isinstance(v, list):
            r = []
            for e in v:
                r.extend(flat(e))
            return r
        return [v]
    out["logits"] = flat(logits.tolist())
    out["argmax"] = [int(x) for x in logits[0].argmax(-1).tolist()]
    out["logits_repr"] = repr(logits)
json.dump(out, sys.stdout)
""".replace("__FILL__", repr(_LLAMA_FILL)).replace(
    "__CFG__", repr(_LLAMA_CFG)).replace("__IDS__", repr(_LLAMA_IDS))


@functools.lru_cache(maxsize=1)
def _forward_road_fixture():
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _FORWARD_ROAD_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"forward-road subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def test_min_and_max_refuse_an_empty_reduction_the_same_way():
    """`min.default` is `max.default`'s mirror, including the refusal.

    Upstream raises `RuntimeError: min(): Expected reduction dim to be
    specified for input.numel() == 0.` -- the identity element is what a
    whole-tensor min of nothing would need, and there isn't one. The shim
    already reproduced this for `max`; the point of asserting both here is
    that the new kernel is the mirror rather than a fresh guess.
    """
    empty = _C._tensor_from_flat([], [0], _C.float32)
    for op in ("aten.min.default", "aten.max.default"):
        try:
            _C._aten_dispatch(op, empty)
        except RuntimeError as e:
            assert "numel() == 0" in str(e), (op, str(e))
        else:
            raise AssertionError(f"{op} on an empty tensor must raise")


# --- the decomposition pass (docs/DECOMP.md) ---------------------------------
#
# `_capture_end` records ATen; ExecuTorch's Edge dialect is defined over Core
# ATen. docs/CAPTURE.md §5 measured that gap and found it is already open in
# the smallest example there. `torchnative.export.decompose` closes it by
# running upstream's own decomposition rules -- which means the tests need the
# vendored tree, the same subprocess shape the capture road above uses.

_DECOMP_ROAD_SCRIPT = r"""
import json, sys
import torch
from torchnative.export import (
    DecompositionRefused,
    core_ops,
    decompose,
    decomposition_table,
    decomposition_table_source,
    is_core,
    non_core_ops,
)
from torchnative.export.decompose import _native_functions_path

out = {}

# -- what Core ATen is, and that the scanner reading it is not lying ----------
out["n_core"] = len(core_ops())
out["t_is_core"] = is_core("aten.t.default")
out["addmm_is_core"] = is_core("aten.addmm.default")
try:
    import yaml
except ImportError:
    out["yaml_diff"] = None
else:
    loader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)
    with open(_native_functions_path()) as handle:
        entries = yaml.load(handle, Loader=loader)
    reference = set()
    for entry in entries:
        tags = entry.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        if "core" not in tags:
            continue
        signature = entry["func"].split("(", 1)[0]
        if "." in signature:
            name, overload = signature.split(".", 1)
        else:
            name, overload = signature, "default"
        reference.add("aten.%s.%s" % (name, overload))
    out["yaml_diff"] = sorted(reference ^ set(core_ops()))

# -- which upstream table answered, and why not the fuller one ----------------
source, reason = decomposition_table_source()
out["table_source"] = source
out["table_reason"] = reason
out["table_size"] = len(decomposition_table())

# -- the proof: capture, lower, replay, compare ------------------------------
#
# Written with `torch.ops.aten.*` spellings rather than `torch.stack` /
# `torch.split` because neither has an entry in the shim's `torch.<fn>`
# overload table yet (docs/DECOMP.md §4). The recorded region is identical
# either way -- capture records at the dispatcher, and both spellings arrive
# there -- so this costs the test nothing and keeps it about decomposition.
def program(x, w):
    h = torch.ops.aten.stack.default([x, x * 2.0])   # not Core ATen
    h = h.view(4, 4)
    y = torch.mm(h, w)
    y = torch.relu(y)
    lo, hi = torch.ops.aten.split.Tensor(y, 2)       # not Core ATen
    return lo + hi

x = torch.ones(2, 4)
w = torch.ones(4, 3) * 0.5

torch._C._capture_begin([x])
captured_out = program(x, w)
out["capture_reason"] = torch._C._capture_reason()
trace = torch._C._capture_end(captured_out)

out["ops_before"] = [n["op"] for n in trace.nodes]
out["non_core_before"] = non_core_ops(out["ops_before"])
out["n_constants_before"] = len(trace.constants)

lowered = decompose(trace)
out["ops_after"] = lowered.ops
out["non_core_after"] = non_core_ops(lowered.ops)
out["n_nodes"] = [len(trace), len(lowered)]
out["n_constants_after"] = len(lowered.constants)
out["repr"] = repr(lowered)
out["graph_keys"] = sorted(lowered.graph())
out["guards"] = lowered.guards

pairs = []
for scale in (1.0, 0.5, -2.0, 7.25):
    z = torch.ones(2, 4) * scale
    (lowered_result,) = lowered.replay([z])
    (captured_result,) = trace.replay([z])
    pairs.append([
        lowered_result.reshape(-1).tolist(),
        captured_result.reshape(-1).tolist(),
        program(z, w).reshape(-1).tolist(),
    ])
out["pairs"] = pairs

# The guard survives the rewrite: a lowered trace is no more general than the
# one it was lowered from.
try:
    lowered.replay([torch.ones(3, 4)])
except RuntimeError as error:
    out["wrong_shape"] = str(error)
else:
    out["wrong_shape"] = "ACCEPTED"


d = torch._C._aten_dispatch


def refusal(tensor, call):
    # The tensor is built *before* recording opens, on purpose: anything
    # allocated inside the region is an op in the record too, and a one-op
    # trace is what these three cases are about.
    torch._C._capture_begin([tensor])
    produced = call(tensor)
    trace = torch._C._capture_end(produced)
    try:
        decompose(trace)
    except DecompositionRefused as error:
        return str(error)
    return "ACCEPTED"


# No rule at all in the table this build can reach.
out["refuse_no_rule"] = refusal(
    torch.ones(3, 4), lambda t: d("aten.transpose.int", t, 0, 1)
)
# A rule exists, and running it hits something the shim does not have.
out["refuse_unrunnable"] = refusal(
    torch.ones(4, 8), lambda t: d("aten.t.default", t)
)
# A rule exists, runs, and produces a result the recording disagrees with.
# `aten.baddbmm.default`'s decomposition multiplies by the Python floats
# `beta`/`alpha`, which promotes float32 to float64 here (docs/DECOMP.md
# §6.2 -- a scalar-promotion divergence, unfixed and out of scope for the
# `sum` fix below).
_bb_c, _bb_a, _bb_b = torch.ones(2, 3, 5), torch.ones(2, 3, 4), torch.ones(2, 4, 5)
torch._C._capture_begin([_bb_c, _bb_a, _bb_b])
_bb_produced = d("aten.baddbmm.default", _bb_c, _bb_a, _bb_b)
_bb_trace = torch._C._capture_end(_bb_produced)
try:
    decompose(_bb_trace)
except DecompositionRefused as error:
    out["refuse_disagrees"] = str(error)
else:
    out["refuse_disagrees"] = "ACCEPTED"

# `aten.sum.default` used to land here too: upstream's rule rewrites it to
# `sum(x, dim=[], dtype=None)`, and this shim's `aten.sum.dim_IntList` used
# to return the input unchanged for an empty `dim` list instead of reducing
# every dimension, so the pass caught the divergence and refused. The kernel
# is fixed now (an empty `dim` list expands to every axis, matching
# upstream), so this proves the fix by lowering the recording instead of
# refusing it.
_sum_tensor = torch.ones(3, 4)
torch._C._capture_begin([_sum_tensor])
_sum_produced = d("aten.sum.default", _sum_tensor)
_sum_trace = torch._C._capture_end(_sum_produced)
_sum_lowered = decompose(_sum_trace)
out["sum_default_ops_after"] = _sum_lowered.ops
(_sum_replayed,) = _sum_lowered.replay([_sum_tensor])
out["sum_default_replayed"] = _sum_replayed.reshape(-1).tolist()
# The divergence that used to live underneath the refusal, stated directly.
out["sum_all_dims"] = list(d("aten.sum.dim_IntList", torch.ones(3, 4), []).shape)
out["sum_all_dims_keepdim"] = list(
    d("aten.sum.dim_IntList", torch.ones(3, 4), [], keepdim=True).shape
)

json.dump(out, sys.stdout)
"""


@functools.lru_cache(maxsize=None)
def _decomp_road_fixture():
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    proc = subprocess.run(
        [sys.executable, "-c", _DECOMP_ROAD_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"decompose-road subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def test_decompose_reads_core_aten_out_of_the_vendored_tree():
    """The Core ATen set is upstream's data, not a list transcribed here.

    `torchgen/packaged/ATen/native/native_functions.yaml` carries a
    `tags: core` annotation per entry, and it is the same file upstream
    generates `torch.Tag.core` from. Reading it is what makes "is this op
    core" answerable in a build whose `_C` is a Python shim -- in this build
    `torch.ops.aten.<op>.<ov>.tags` is empty for every op, so the tag route
    would classify the whole program as non-core.

    The scan is a line scanner rather than a YAML parse, because `pyyaml` is
    not a declared dependency of this distribution. This test diffs the scan
    against a real YAML parse so that a format change upstream is a failure
    here rather than a quietly shorter list.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return
    r = _decomp_road_fixture()
    assert r["n_core"] == 193, r["n_core"]
    assert r["addmm_is_core"] is True
    # The op docs/CAPTURE.md §5 named: the smallest model already emits one
    # that Edge will not take.
    assert r["t_is_core"] is False
    if r["yaml_diff"] is not None:
        assert r["yaml_diff"] == [], r["yaml_diff"]


def test_decompose_says_which_upstream_table_it_could_get():
    """A silent fallback to a smaller table is a silent loss of coverage.

    `core_aten_decompositions()` is the table this pass wants. It cannot be
    built here: its constructor enumerates every CompositeImplicitAutograd
    registration through `torch._C._dispatch_get_registrations_for_dispatch_key`,
    and this `_C` has no C++ dispatcher to enumerate. So the pass falls back to
    `_core_aten_decompositions_post_autograd()` and *reports that it did*.

    Pinned as a test rather than a comment because the day the shim answers
    that query, this assertion is what tells us the fuller table arrived.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    r = _decomp_road_fixture()
    assert r["table_source"] == "_core_aten_decompositions_post_autograd", r["table_source"]
    assert "_dispatch_get_registrations_for_dispatch_key" in (r["table_reason"] or ""), r
    assert r["table_size"] == 224, r["table_size"]


def test_decompose_lowers_a_trace_to_core_aten():
    """The pass does what it says: non-core in, Core ATen out.

    Node count going *up* is the point rather than an incidental -- a
    decomposition replaces one op with several, and a pass that left the count
    alone would not have done anything. `aten.stack.default` becomes
    `cat` + `view` and `aten.split.Tensor` becomes `split_with_sizes`, both
    upstream's rules, neither written here.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    r = _decomp_road_fixture()
    assert r["capture_reason"] is None, r["capture_reason"]
    assert r["non_core_before"] == ["aten.split.Tensor", "aten.stack.default"], r
    assert r["non_core_after"] == [], r["non_core_after"]
    assert r["n_nodes"] == [7, 8], r["n_nodes"]
    assert "aten.cat.default" in r["ops_after"], r["ops_after"]
    assert "aten.split_with_sizes.default" in r["ops_after"], r["ops_after"]
    # The record keeps its shape across the rewrite: same reader, same keys.
    assert r["graph_keys"] == ["constants", "nodes", "outputs", "placeholders"]
    assert r["guards"] == [
        {"index": 0, "shape": [2, 4], "dtype": "torch.float32", "device": "cpu"}
    ], r["guards"]
    assert r["repr"].startswith("<DecomposedTrace 8 nodes, 1 inputs"), r["repr"]


def test_decomposed_replay_matches_eager_bit_for_bit():
    """The judgement. Lowering must not change the answer.

    A decomposition is the same mathematics in a different order, and a
    different order of floating-point operations is entitled to a different
    last bit. So this is a measurement, not an assumption: on this trace, over
    four inputs of which three the trace never saw, the lowered graph, the
    captured graph and eager all produce **identical** lists. If that ever
    stops being true the honest response is to record the size of the
    difference here (docs/DEVICE.md §5 prefers a named exception to a
    tolerance), not to widen the comparison.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    r = _decomp_road_fixture()
    assert len(r["pairs"]) == 4
    for lowered, captured, eager in r["pairs"]:
        assert lowered == eager, (lowered, eager)
        assert captured == eager, (captured, eager)
    # And the guard is still a guard: lowering does not generalise a trace.
    assert r["wrong_shape"] != "ACCEPTED"
    assert "[2, 4]" in r["wrong_shape"], r["wrong_shape"]


def test_decompose_refuses_by_name_what_it_cannot_lower():
    """Three different ways to fail, and each one says the op it failed on.

    Passing a non-core op through quietly is the expensive failure: ExecuTorch
    rejects the program later, with nothing pointing at which op or which pass
    was responsible. So each refusal names the op *and* which of the three
    walls it hit -- no rule, the rule will not run here, or the rule disagreed
    with the recording.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    r = _decomp_road_fixture()

    # 1. Nothing in the reachable table has a rule for it.
    assert r["refuse_no_rule"] != "ACCEPTED"
    assert "aten.transpose.int" in r["refuse_no_rule"], r["refuse_no_rule"]
    assert "no rule" in r["refuse_no_rule"], r["refuse_no_rule"]

    # 2. A rule exists and running it reaches something the shim lacks. The
    #    refusal carries the underlying reason, so the gap is findable.
    assert r["refuse_unrunnable"] != "ACCEPTED"
    assert "aten.t.default" in r["refuse_unrunnable"], r["refuse_unrunnable"]
    assert "torch.transpose" in r["refuse_unrunnable"], r["refuse_unrunnable"]

    # 3. A rule exists, runs, and produces a result the recording disagrees
    #    with -- `aten.baddbmm.default`'s decomposition promotes float32 to
    #    float64 (docs/DECOMP.md §6.2, unfixed). `aten.sum.default` used to
    #    be this example; it moved once the kernel bug it caught was fixed
    #    (test_decompose_lowers_sum_default_now_that_the_kernel_agrees).
    assert r["refuse_disagrees"] != "ACCEPTED"
    assert "aten.baddbmm.default" in r["refuse_disagrees"], r["refuse_disagrees"]
    assert "torch.float64" in r["refuse_disagrees"], r["refuse_disagrees"]
    assert "torch.float32" in r["refuse_disagrees"], r["refuse_disagrees"]


def test_decompose_lowers_sum_default_now_that_the_kernel_agrees():
    """The check that found a real bug rather than a missing feature -- and
    the regression pin for the fix.

    `aten.sum.default` has a rule, and it rewrites `sum(x)` to
    `sum(x, dim=[], dtype=None)`. An empty `dim` list means *reduce every
    dimension*: upstream returns a scalar 12.0, shape `[]`, for a 3x4 of
    ones. This shim's `aten.sum.dim_IntList` used to return the input
    unchanged for that argument (shape `[3, 4]`), so the decompose pass
    caught the divergence between the sub-trace it ran and the recording,
    and refused rather than emit a lowered graph that silently disagreed
    with eager.

    The kernel is fixed (`rust/torch_c/src/aten.rs::sum_or_mean` now expands
    an empty `dim` list to every axis before reducing), so the rule and the
    recording agree and the pass lowers the trace instead of refusing it.
    Both are asserted directly: `sum.dim_IntList([])` itself, and that
    `decompose` accepts a one-op `sum.default` trace and replays it to the
    same scalar. If the kernel regresses to returning the input unchanged,
    the shape assertions here go red before anyone reaches for the refusal
    path again.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    r = _decomp_road_fixture()
    # The divergence itself, fixed: empty `dim` reduces every axis.
    assert r["sum_all_dims"] == [], r["sum_all_dims"]
    # `keepdim=True` with an empty `dim` list keeps every axis, size 1 each,
    # rather than reducing to a 0-d scalar.
    assert r["sum_all_dims_keepdim"] == [1, 1], r["sum_all_dims_keepdim"]
    # The rule and the recording now agree, so `sum.default` lowers cleanly
    # to `sum.dim_IntList` instead of being refused.
    assert r["sum_default_ops_after"] == ["aten.sum.dim_IntList"], r[
        "sum_default_ops_after"
    ]
    assert r["sum_default_replayed"] == [12.0], r["sum_default_replayed"]


def test_capture_trace_hands_out_the_constants_it_burned_in():
    """A rewrite needs the weights, and metadata is not the weights.

    `CaptureTrace.constants` is shape/dtype/device -- enough to *read* a
    trace, not enough to build a second one that computes the same thing.
    `constant_values` is the other half, and it hands out the same references
    replay uses rather than copies: two records of one region must not drift
    apart from each other or from the module they came from.
    """
    d = _C._aten_dispatch
    a = _C._tensor_new_from_data([[1.0, 2.0], [3.0, 4.0]])
    w = _C._tensor_new_from_data([[1.0, 0.0], [0.0, 1.0]])
    _C._capture_begin([a])
    trace = _C._capture_end(d("aten.mm.default", a, w))

    assert len(trace.constant_values) == len(trace.constants) == 1
    held = trace.constant_values[0]
    assert held is w
    assert list(held.shape) == trace.constants[0]["shape"]


# --- `from_pretrained` with real weights (docs/CKPT2.md) ---------------------
#
# docs/E2E_REAL.md §6.2 left `from_pretrained` stopped at
# `torch.UntypedStorage.from_file`: the model was built, the weights were not
# read. These tests are the other side of that line. They are deliberately not
# "did it load" tests -- docs/CKPT.md §4 measured a load path that reported
# `<All keys matched successfully>` with every weight at `0.0` and no exception
# anywhere, so "it loaded" is exactly the claim that failure makes.
#
# The judgement here is values, at two depths:
#
#   * every loaded parameter is compared to upstream's, **bit for bit** -- this
#     is what catches zeros, and also what catches an mmap offset that is off
#     by a record header and hands back the neighbouring tensor's bytes;
#   * the forward pass of the loaded model is compared to upstream's logits.
#
# and with a negative control, because a comparison that cannot fail is not a
# comparison: the same model with its weights left at initialisation must be
# *far* from the truth logits. If that control ever comes out close, the
# comparison above is measuring nothing.

_FROM_PRETRAINED_SCRIPT = r"""
import json, os, sys, traceback
import torch

ST = sys.argv[1]      # safetensors checkpoint directory
BIN = sys.argv[2]     # .bin checkpoint directory
PAYLOAD = sys.argv[3] # a plain file of known bytes, for from_file itself
IDS = json.loads(sys.argv[4])

out = {}


def flat(v):
    if isinstance(v, list):
        r = []
        for e in v:
            r.extend(flat(e))
        return r
    return [v]


# --- `UntypedStorage.from_file` on its own, against the same file the
# --- expected side measured upstream.
n = os.path.getsize(PAYLOAD)
ff = {}


def probe(key, fn):
    try:
        ff[key] = fn()
    except BaseException as e:
        ff[key] = "%s: %s" % (type(e).__name__, str(e)[:200])


probe("nbytes_full", lambda: torch.UntypedStorage.from_file(PAYLOAD, False, n).nbytes())
probe("nbytes_16", lambda: torch.UntypedStorage.from_file(PAYLOAD, False, 16).nbytes())
probe("nbytes_0", lambda: torch.UntypedStorage.from_file(PAYLOAD, False, 0).nbytes())
probe("kwargs", lambda: torch.UntypedStorage.from_file(PAYLOAD, shared=False, nbytes=8).nbytes())
probe("default_nbytes", lambda: torch.UntypedStorage.from_file(PAYLOAD, False).nbytes())
probe("first8", lambda: [torch.UntypedStorage.from_file(PAYLOAD, False, n)[i] for i in range(8)])
probe("slice_16_24", lambda: [torch.UntypedStorage.from_file(PAYLOAD, False, n)[16:24][i] for i in range(8)])
probe("slice_len_0", lambda: len(torch.UntypedStorage.from_file(PAYLOAD, False, n)[0:0]))
probe("slice_clamped", lambda: len(torch.UntypedStorage.from_file(PAYLOAD, False, n)[n - 4:n + 8]))
probe("slice_negative", lambda: len(torch.UntypedStorage.from_file(PAYLOAD, False, n)[-8:]))
probe("slice_offset_ptr",
      lambda: torch.UntypedStorage.from_file(PAYLOAD, False, n)[16:24].data_ptr()
              - torch.UntypedStorage.from_file(PAYLOAD, False, n).data_ptr())
probe("filename", lambda: torch.UntypedStorage.from_file(PAYLOAD, False, n).filename)
probe("element_size", lambda: torch.UntypedStorage.from_file(PAYLOAD, False, n).element_size())
probe("device", lambda: str(torch.UntypedStorage.from_file(PAYLOAD, False, n).device))
probe("too_big", lambda: torch.UntypedStorage.from_file(PAYLOAD, False, n + 1).nbytes())
probe("missing", lambda: torch.UntypedStorage.from_file(PAYLOAD + ".nope", False, 8).nbytes())
probe("step_2", lambda: len(torch.UntypedStorage.from_file(PAYLOAD, False, n)[::2]))
probe("shared_true", lambda: torch.UntypedStorage.from_file(PAYLOAD, True, n).nbytes())
out["from_file"] = ff

# --- `torch.load(mmap=True)` must agree with `mmap=False`, which docs/CKPT.md
# --- already proved correct. This is the cross-check that an offset error
# --- cannot survive: the two readers reach the payload by different routes.
try:
    ck = os.path.join(BIN, "pytorch_model.bin")
    a = torch.load(ck, mmap=True, weights_only=True)
    b = torch.load(ck, mmap=False, weights_only=True)
    worst = 0.0
    for k in sorted(b):
        va, vb = flat(a[k].tolist()), flat(b[k].tolist())
        if len(va) != len(vb):
            raise AssertionError("shape drift on " + k)
        for x, y in zip(va, vb):
            worst = max(worst, abs(float(x) - float(y)))
except BaseException:
    out["mmap_vs_read"] = "FAILED: " + traceback.format_exc(limit=4)
else:
    out["mmap_vs_read"] = "OK"
    out["mmap_vs_read_worst"] = worst
    out["mmap_vs_read_keys"] = sorted(b)

# --- `from_pretrained`, four ways.
from transformers import AutoModelForCausalLM


def load(tag, path, **kw):
    try:
        model = AutoModelForCausalLM.from_pretrained(path, **kw)
        model.eval()
        with torch.no_grad():
            logits = model(torch.tensor([IDS])).logits
        sd = {k: flat(v.tolist()) for k, v in sorted(model.state_dict().items())}
    except BaseException:
        out[tag] = "FAILED: " + traceback.format_exc(limit=6)
    else:
        out[tag] = "OK"
        out[tag + "_shape"] = [int(d) for d in logits.shape]
        out[tag + "_logits"] = flat(logits.tolist())
        out[tag + "_argmax"] = [int(x) for x in logits[0].argmax(-1).tolist()]
        out[tag + "_state_dict"] = sd


load("st_mmap", ST)
load("st_nommap", ST, disable_mmap=True)
load("bin_mmap", BIN)
load("bin_nommap", BIN, disable_mmap=True)

# The three checkpoint *shapes* docs/E2E_REAL.md §6.2 listed as unmeasured,
# plus bfloat16, which is what real checkpoints are actually stored in.
for tag, sub in (("tied", "tied"), ("shard", "shard"), ("bf16", "bf16"),
                 ("meta", "meta")):
    load("hard_" + tag, os.path.join(sys.argv[5], sub))

# --- the negative control: the same architecture, weights never loaded.
try:
    from transformers.models.llama.configuration_llama import LlamaConfig
    cfg = AutoModelForCausalLM.from_pretrained(ST).config if False else None
    import json as _json
    with open(os.path.join(ST, "config.json")) as fh:
        raw = _json.load(fh)
    fresh = AutoModelForCausalLM.from_config(LlamaConfig(**{
        k: raw[k] for k in (
            "vocab_size", "hidden_size", "intermediate_size", "num_hidden_layers",
            "num_attention_heads", "num_key_value_heads", "max_position_embeddings",
            "tie_word_embeddings") if k in raw}))
    fresh.eval()
    with torch.no_grad():
        out["unloaded_logits"] = flat(fresh(torch.tensor([IDS])).logits.tolist())
except BaseException:
    out["unloaded_logits"] = "FAILED: " + traceback.format_exc(limit=4)

json.dump(out, sys.stdout)
"""


_FROM_FILE_PAYLOAD = bytes(range(256)) * 4


@functools.lru_cache(maxsize=1)
def _from_pretrained_fixture():
    """Write real checkpoints with upstream torch; read them with the shim.

    Two interpreters for docs/CKPT.md §8.2's reason -- `from_pretrained` lives
    in pure-Python `transformers` on top of pure-Python `torch`, so the shim
    has to be `torch` by name, and a process has only one of those.

    The expected side is computed here, on upstream torch, from the same
    `_LLAMA_FILL` source text the rest of this file uses; nothing below is a
    number copied out of a previous run.
    """
    torch = _upstream_torch
    import shutil
    import tempfile

    from transformers import AutoModelForCausalLM
    from transformers.models.llama.configuration_llama import LlamaConfig

    root = tempfile.mkdtemp(prefix="from-pretrained-")
    st = os.path.join(root, "st")
    binned = os.path.join(root, "bin")
    payload = os.path.join(root, "payload.bin")
    with open(payload, "wb") as fh:
        fh.write(_FROM_FILE_PAYLOAD)

    ns = {}
    exec(_LLAMA_FILL, ns)
    model = AutoModelForCausalLM.from_config(LlamaConfig(**_LLAMA_CFG))
    model.eval()
    ns["_fill"](model, torch)
    with torch.no_grad():
        logits = model(torch.tensor([_LLAMA_IDS])).logits

    model.save_pretrained(st, safe_serialization=True)
    # transformers 5 writes safetensors whatever `safe_serialization` says, so
    # the `.bin` container -- the one that reaches `torch.load(mmap=True)` --
    # is written directly.
    os.makedirs(binned, exist_ok=True)
    for name in ("config.json", "generation_config.json"):
        src = os.path.join(st, name)
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(binned, name))
    torch.save(model.state_dict(), os.path.join(binned, "pytorch_model.bin"))

    expected = {
        "logits": _e2e_flatten(logits.tolist()),
        "shape": [int(d) for d in logits.shape],
        "argmax": [int(x) for x in logits[0].argmax(-1).tolist()],
        "state_dict": {
            k: _e2e_flatten(v.tolist()) for k, v in sorted(model.state_dict().items())
        },
    }

    # --- the checkpoint shapes docs/E2E_REAL.md §6.2 left unmeasured ---------
    #
    # Each is a *container* property, not a numeric one, and each has its own
    # way of going quietly wrong:
    #
    #   tied   one storage, two state-dict keys. safetensors refuses to write
    #          the duplicate and records it in the header instead, so a reader
    #          that ignores that lands a model missing its lm_head.
    #   shard  the weights are spread over N files behind an index. A reader
    #          that stops after the first file gets a model that is mostly
    #          freshly initialised -- and reports no error.
    #   bf16   what real checkpoints are stored in, and the one case where the
    #          bytes are not float32.
    #   meta   `nn.Module.state_dict()` attaches a `_metadata` attribute that
    #          `torch.save` pickles alongside the tensors.
    hard = os.path.join(root, "hard")
    for tag, kw, extra in (
        ("tied", dict(tie_word_embeddings=True), {}),
        ("shard", dict(num_hidden_layers=3), dict(max_shard_size="6KB")),
        ("bf16", {}, {}),
        ("meta", {}, {}),
    ):
        conf = dict(_LLAMA_CFG)
        conf.update(kw)
        m = AutoModelForCausalLM.from_config(LlamaConfig(**conf))
        m.eval()
        ns["_fill"](m, torch)
        if tag == "bf16":
            m = m.to(torch.bfloat16)
        with torch.no_grad():
            lg = m(torch.tensor([_LLAMA_IDS])).logits
        expected["hard_" + tag] = {
            "logits": _e2e_flatten(lg.float().tolist()),
            "argmax": [int(x) for x in lg[0].argmax(-1).tolist()],
            "state_dict": {
                k: _e2e_flatten(v.float().tolist())
                for k, v in sorted(m.state_dict().items())
            },
        }
        d = os.path.join(hard, tag)
        if tag == "meta":
            # `save_pretrained` always writes safetensors on transformers 5,
            # and safetensors has nowhere to put `_metadata`. The `.bin`
            # container does, so this one is written by hand.
            os.makedirs(d, exist_ok=True)
            shutil.copy(os.path.join(st, "config.json"), os.path.join(d, "config.json"))
            state = m.state_dict()
            assert hasattr(state, "_metadata"), "state_dict() should carry _metadata"
            torch.save(state, os.path.join(d, "pytorch_model.bin"))
        else:
            m.save_pretrained(d, **extra)
    assert len([n for n in os.listdir(os.path.join(hard, "shard"))
                if n.endswith(".safetensors")]) > 1, "the shard case must shard"

    # The same `from_file` probes the subprocess runs, answered by upstream.
    n = len(_FROM_FILE_PAYLOAD)
    ff = {}

    def probe(key, fn):
        try:
            ff[key] = fn()
        except BaseException as e:  # noqa: BLE001
            ff[key] = "%s: %s" % (type(e).__name__, str(e)[:200])

    U = torch.UntypedStorage
    probe("nbytes_full", lambda: U.from_file(payload, False, n).nbytes())
    probe("nbytes_16", lambda: U.from_file(payload, False, 16).nbytes())
    probe("nbytes_0", lambda: U.from_file(payload, False, 0).nbytes())
    probe("kwargs", lambda: U.from_file(payload, shared=False, nbytes=8).nbytes())
    probe("default_nbytes", lambda: U.from_file(payload, False).nbytes())
    probe("first8", lambda: [U.from_file(payload, False, n)[i] for i in range(8)])
    probe("slice_16_24", lambda: [U.from_file(payload, False, n)[16:24][i] for i in range(8)])
    probe("slice_len_0", lambda: len(U.from_file(payload, False, n)[0:0]))
    probe("slice_clamped", lambda: len(U.from_file(payload, False, n)[n - 4:n + 8]))
    probe("slice_negative", lambda: len(U.from_file(payload, False, n)[-8:]))
    probe("slice_offset_ptr",
          lambda: U.from_file(payload, False, n)[16:24].data_ptr()
                  - U.from_file(payload, False, n).data_ptr())
    probe("filename", lambda: U.from_file(payload, False, n).filename)
    probe("element_size", lambda: U.from_file(payload, False, n).element_size())
    probe("device", lambda: str(U.from_file(payload, False, n).device))
    expected["from_file"] = ff

    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _FROM_PRETRAINED_SCRIPT, st, binned, payload,
         json.dumps(_LLAMA_IDS), hard],
        capture_output=True,
        text=True,
        env=env,
        timeout=900,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"from_pretrained subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return expected, json.loads(proc.stdout)


def _worst_state_dict_drift(expected, got):
    """Bit-for-bit, per parameter. Returns (worst, name-of-worst).

    Not a tolerance: the shim reads the same little-endian float32 bytes
    upstream wrote, so anything other than `0.0` here means the bytes that
    arrived were not the bytes on disk.
    """
    assert sorted(expected) == sorted(got), (sorted(expected), sorted(got))
    worst, where = 0.0, None
    for k in sorted(expected):
        a, b = expected[k], got[k]
        assert len(a) == len(b), (k, len(a), len(b))
        for x, y in zip(a, b):
            d = abs(float(x) - float(y))
            if d > worst:
                worst, where = d, k
    return worst, where


def test_from_file_answers_what_upstream_answers_for_a_private_mapping():
    """`from_file(shared=False)` is a private mapping, and this shim copies.

    docs/CKPT2.md §2 has the measurement that makes copying the right answer
    rather than a shortcut: upstream's default is `MAP_PRIVATE`, writes through
    such a mapping never reach the file and are invisible to a second mapping,
    so a read of the same bytes is observationally the same object. What is
    asserted here is the observable surface, against upstream's own answers for
    the same file -- sizes, contents, slicing, the offset arithmetic that
    `torch/serialization.py` slices storages with, and the two errors.
    """
    if not _ckpt_shim_available() or _upstream_transformers is None:
        return
    expected, got = _from_pretrained_fixture()
    ff = got["from_file"]
    for key, want in expected["from_file"].items():
        assert ff[key] == want, (key, ff[key], want)
    # The two refusals upstream raises are `RuntimeError`, and the shim must
    # raise them too rather than hand back a short buffer or an empty one.
    assert ff["too_big"].startswith("RuntimeError"), ff["too_big"]
    assert ff["missing"].startswith("RuntimeError"), ff["missing"]
    assert ff["step_2"].startswith("RuntimeError"), ff["step_2"]
    # `shared=True` is the one thing here that cannot be copied: it means
    # writes go back to the file. docs/CKPT2.md §2.1. Refused, by name.
    assert isinstance(ff["shared_true"], str) and "shared" in ff["shared_true"], (
        ff["shared_true"])


def test_mmap_and_read_paths_of_torch_load_agree_on_every_tensor():
    """Two routes to the same bytes, and they have to meet.

    `mmap=False` goes through `zipfile.read`, which docs/CKPT.md validated.
    `mmap=True` goes through `from_file` plus `get_record_offset` -- byte
    arithmetic over the archive's local file headers. An offset that is wrong
    by a header does not raise: it silently returns the neighbouring tensor.
    Comparing the two routes is the check that arithmetic cannot dodge.
    """
    if not _ckpt_shim_available() or _upstream_transformers is None:
        return
    _, got = _from_pretrained_fixture()
    assert got["mmap_vs_read"] == "OK", got["mmap_vs_read"]
    assert got["mmap_vs_read_worst"] == 0.0, got["mmap_vs_read_worst"]
    assert len(got["mmap_vs_read_keys"]) > 0


def test_from_pretrained_loads_the_real_weights_bit_for_bit_on_all_four_paths():
    """The weights that arrive are the weights on disk -- not zeros, not neighbours.

    Four routes reach the checkpoint and all four are asserted, because they
    are genuinely different code: safetensors' mmap backend and its byte
    backend, and `torch.load` with mmap on and off. docs/E2E_REAL.md §6.2 had
    three of the four stopped at the same name.
    """
    if not _ckpt_shim_available() or _upstream_transformers is None:
        return
    expected, got = _from_pretrained_fixture()
    for tag in ("st_mmap", "st_nommap", "bin_mmap", "bin_nommap"):
        assert got[tag] == "OK", (tag, got[tag])
        worst, where = _worst_state_dict_drift(
            expected["state_dict"], got[tag + "_state_dict"])
        assert worst == 0.0, (tag, where, worst)


def test_from_pretrained_forward_matches_upstream_logits():
    """Loading is not the claim; computing the same answer is.

    The bound is `_REAL_LLAMA_ATOL`, for the reason its own comment gives --
    this is the same model at the same width, so the same sensitivity limit
    applies and the file default would be ~20% of a logit.
    """
    if not _ckpt_shim_available() or _upstream_transformers is None:
        return
    expected, got = _from_pretrained_fixture()
    for tag in ("st_mmap", "st_nommap", "bin_mmap", "bin_nommap"):
        assert got[tag] == "OK", (tag, got[tag])
        assert got[tag + "_shape"] == expected["shape"], (tag, got[tag + "_shape"])
        assert got[tag + "_argmax"] == expected["argmax"], (
            tag, got[tag + "_argmax"], expected["argmax"])
        diff = max(abs(a - b) for a, b in zip(got[tag + "_logits"], expected["logits"]))
        assert diff < _REAL_LLAMA_ATOL, (tag, diff)


def test_the_four_hard_checkpoint_shapes_load_with_the_right_weights():
    """Shared tensors, shards, bfloat16, and a `_metadata` state dict.

    docs/E2E_REAL.md §6.2 named the first, second and fourth of these as
    unmeasured; docs/CKPT2.md §6 is the measurement. The bar is the same as the
    plain case -- every parameter bit-for-bit against upstream's -- because
    every one of these fails *quietly*: a reader that ignores safetensors'
    shared-tensor header loses `lm_head` and initialises it fresh, and a reader
    that stops at the first shard keeps only the layers in it. Neither raises.

    **bfloat16 is asserted on weights only, and that is the honest bar.** Its
    logits are compared separately and loosely below, because this build
    upcasts `bf16` to `f32` inside several kernels (`aten.rs`), so it computes
    the forward pass in *more* precision than upstream and the two are expected
    to differ by roughly bf16's own resolution. The load is exact regardless,
    which is what this test is about.
    """
    if not _ckpt_shim_available() or _upstream_transformers is None:
        return
    expected, got = _from_pretrained_fixture()
    for tag in ("tied", "shard", "bf16", "meta"):
        key = "hard_" + tag
        assert got[key] == "OK", (tag, got[key])
        worst, where = _worst_state_dict_drift(
            expected[key]["state_dict"], got[key + "_state_dict"])
        assert worst == 0.0, (tag, where, worst)
        assert got[key + "_argmax"] == expected[key]["argmax"], (
            tag, got[key + "_argmax"], expected[key]["argmax"])
        diff = max(abs(a - b)
                   for a, b in zip(got[key + "_logits"], expected[key]["logits"]))
        # bf16's ulp near 1.0 is ~0.0078; the measured divergence is 0.042 on
        # logits whose scale is ~0.4, which is the accumulation difference
        # described above rather than a load error. The other three are held to
        # the same bound as the plain case.
        bound = 0.1 if tag == "bf16" else _REAL_LLAMA_ATOL
        assert diff < bound, (tag, diff)


def test_an_unloaded_model_is_far_from_the_truth_logits():
    """The negative control for the three tests above.

    If a model that never read the checkpoint landed within `_REAL_LLAMA_ATOL`
    of the truth, then those tests would pass whether or not any weight was
    ever read, and docs/CKPT.md §4's failure -- a full state dict of `0.0` --
    would sail through. The threshold is deliberately coarse: what is being
    established is that loading moves the answer at all.
    """
    if not _ckpt_shim_available() or _upstream_transformers is None:
        return
    expected, got = _from_pretrained_fixture()
    unloaded = got["unloaded_logits"]
    assert not isinstance(unloaded, str), unloaded
    diff = max(abs(a - b) for a, b in zip(unloaded, expected["logits"]))
    assert diff > 1e-3, diff


# ---------------------------------------------------------------------------
# Schema text (docs/SCHEMA.md)
# ---------------------------------------------------------------------------
#
# `_get_schema` used to answer every aten op with `_Schema(qualname, overload)`
# -- no arguments, no returns. Every predicate reading it was therefore a
# constant. docs/DISTRIBUTED.md §8.1 caught `is_mutable`, which had just been
# fixed from always-true to always-*false* and was still wrong for the seven
# in-place ops among the implemented set. It is not one predicate: 66 sites in
# the vendored tree read `._schema.arguments`, 38 read `.returns`, 18 read
# `.is_mutable` and 2 call `._is_view_op()`, and all of them were reading an
# empty schema.
#
# These tests run in the vendored tree, because that is where the schema text
# comes from: `torchgen/packaged/ATen/native/native_functions.yaml`, which ships
# in the wheel beside `torch/`. The bare `_C` on this file's PYTHONPATH has no
# such sibling, so it can only be asked the questions that need no table.

_SCHEMA_ROAD_SCRIPT = r"""
import json, sys
import torch

request = json.load(sys.stdin)
out = {}
out["source"] = torch._C._shim_schema_source()

ops = {}
for key in torch._C._aten_implemented():
    namespace, _, rest = key.partition(".")
    name, _, overload = rest.rpartition(".")
    schema = torch._C._get_schema(f"{namespace}::{name}",
                                  "" if overload == "default" else overload)
    entry = {"text": str(schema), "placeholder": schema.is_placeholder}
    entry["is_mutable"] = None if schema.is_placeholder else schema.is_mutable
    # The route the op actually takes, not a second lookup: `torch.ops` is what
    # every reader in the tree goes through, and if `_get_schema` were right
    # while `torch.ops.aten.<op>.<ov>._schema` were not, nothing would be fixed.
    packet = getattr(torch.ops.aten, name)
    entry["via_ops"] = str(getattr(packet, overload)._schema)
    ops[key] = entry
out["ops"] = ops

# Every (op, overload) the transcribed tables carry, whether or not it is
# implemented. These strings are `str(...._schema)` from upstream 2.13.0, so
# they are an in-repo oracle for the normalisation -- no upstream torch needed.
table = {}
for qualname, overload in request["table_keys"]:
    schema = torch._C._get_schema(qualname, overload)
    table[f"{qualname}|{overload}"] = {
        "text": str(schema), "placeholder": schema.is_placeholder,
        "from": torch._C._shim_schema_provenance(qualname, overload),
    }
out["table"] = table

# An op nothing can have text for. It answers, and the answer is counted.
absent = torch._C._get_schema("aten::not_an_operator_this_repo_will_ever_have", "")
out["absent_placeholder"] = absent.is_placeholder
out["absent_text"] = str(absent)
out["absent_is_mutable"] = absent.is_mutable
out["absent_is_view_op"] = absent._is_view_op()
out["placeholders_listed"] = torch._C._shim_placeholder_schemas()
out["unanswered"] = torch._C._shim_unanswered_predicates()

# The registry, not the schema: an in-place variant the file does not declare
# is not an operator, and a name the file simply omits still is.
registry = {}
for name in ("convolution_", "mm_", "add_", "quantized_lstm", "zero", "relu_"):
    try:
        registry[name] = sorted(getattr(torch.ops.aten, name).overloads())
    except AttributeError as error:
        registry[name] = f"AttributeError: {error}"
out["registry"] = registry

json.dump(out, sys.stdout)
"""


def _schema_table_keys():
    """`(qualname, overload)` for every entry of the two transcribed tables.

    Keyed off the schema string rather than the table key, for the reason
    `verify_schemas.py:_aten_name` gives: `methods.json`'s key is the Python
    method name (`__mul__`, `item`) and not the aten op.
    """
    keys = {}
    for filename in ("overloads.json", "methods.json"):
        path = os.path.join(_CKPT_REPO_ROOT, "rust", "torch_c", "src", filename)
        with open(path, encoding="utf-8") as fh:
            table = json.load(fh)
        for name, schemas in table.items():
            if name.startswith("_README"):
                continue
            for text in schemas:
                head = text.split("(", 1)[0].strip()
                _, _, rest = head.rpartition("::")
                op, _, overload = rest.partition(".")
                keys[(f"aten::{op}", overload)] = text
    return keys


@functools.lru_cache(maxsize=None)
def _schema_road_fixture():
    keys = _schema_table_keys()
    request = json.dumps({"table_keys": [list(k) for k in sorted(keys)]})
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    proc = subprocess.run(
        [sys.executable, "-c", _SCHEMA_ROAD_SCRIPT],
        input=request,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"schema-road subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return keys, json.loads(proc.stdout)


#: The mutating ops among the implemented set, as upstream 2.13.0 answers for
#: the same overloads (`verify_schemas.py` re-derives this against real torch;
#: it is written out here so the check needs no upstream install).
#:
#: docs/DISTRIBUTED.md §8.1 named seven -- `add_ copy_ fill_ normal_ relu_
#: uniform_ zero_ ` -- against the 97 ops implemented when it was written. The
#: set is 117 now and five more of them mutate, which is the other half of that
#: section's point: the wrong direction was "does not mutate", so growing the
#: op set grew the silent lie rather than exposing it.
#:
#: Listed as full keys rather than derived from the trailing underscore. A
#: name rule would agree with a name-based implementation for the wrong reason,
#: and it would also get `index_put_` right while getting `fill_.Tensor` and
#: `fill_.Scalar` -- which differ only by overload -- for free, hiding whether
#: the overload was resolved at all.
_EXPECTED_MUTABLE = (
    "aten.add_.Tensor",
    "aten.clamp_.default",
    "aten.copy_.default",
    "aten.div_.Tensor",
    "aten.fill_.Scalar",
    "aten.fill_.Tensor",
    "aten.index_put_.default",
    "aten.masked_fill_.Scalar",
    "aten.normal_.default",
    "aten.relu_.default",
    "aten.uniform_.default",
    "aten.zero_.default",
)

#: The seven docs/DISTRIBUTED.md §8.1 named, as the judgement it set.
_SECTION_8_1_MUTABLE = ("add_", "copy_", "fill_", "normal_", "relu_", "uniform_",
                        "zero_")


def _native_functions_keys():
    """`(op, overload)` for every `- func:` in the vendored yaml.

    Read here, in the test, rather than asked of the shim: the point of the
    checks below is to compare what the shim answers against what the file
    says, and asking the shim for both sides would make the comparison vacuous.
    """
    path = os.path.join(_CKPT_VENDOR_DIR, "torchgen", "packaged", "ATen",
                        "native", "native_functions.yaml")
    keys = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.startswith("- func:"):
                continue
            text = line[len("- func:"):].strip()
            name, _, overload = text.split("(", 1)[0].partition(".")
            keys[(f"aten::{name}", overload)] = text
    return keys


def test_every_implemented_op_has_schema_text():
    """The 117 implemented ops, and how many of them are still placeholders.

    A placeholder is not a missing answer, it is a wrong one: `arguments` is
    empty, so `any(a.alias_info.is_write for a in schema.arguments)` is False
    and `is_functional_schema` is True, for every op. This asserts the count is
    zero rather than "small", because the source (`native_functions.yaml`,
    vendored) carries all 2584 aten entries and there is no reason for a
    shortfall to be tolerated silently.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        print("   (skipped: vendored tree has no _C.abi3.so)")
        return
    _, report = _schema_road_fixture()
    assert report["source"].endswith("native_functions.yaml"), report["source"]
    placeholders = sorted(k for k, v in report["ops"].items() if v["placeholder"])
    assert placeholders == [], placeholders
    assert len(report["ops"]) == 117, len(report["ops"])
    for key, entry in sorted(report["ops"].items()):
        assert entry["text"] != f"{key}(...) -> ...", key
        assert entry["text"].startswith("aten::"), (key, entry["text"])
        # The route the tree takes has to carry the same text.
        assert entry["via_ops"] == entry["text"], (key, entry)


def test_the_seven_in_place_ops_say_that_they_mutate():
    """docs/DISTRIBUTED.md §8.1's judgement, in one assertion.

    `add_.Tensor` is `aten::add_.Tensor(Tensor(a!) self, ...)` -- the `!` on
    `self`'s alias annotation is the whole content of `is_mutable`, and it can
    only be read if the argument list was really parsed.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    _, report = _schema_road_fixture()
    mutable = sorted(k for k, v in report["ops"].items() if v["is_mutable"])
    assert mutable == sorted(_EXPECTED_MUTABLE), mutable
    # §8.1's own list, checked as it was written -- by op name, so that the
    # judgement survives the expected set above being re-measured.
    for name in _SECTION_8_1_MUTABLE:
        hits = [k for k in mutable if k.split(".")[1] == name]
        assert hits, name
    # And the functional sibling is not mutable, so the predicate is not
    # answering "yes" to everything sharing a stem.
    assert report["ops"]["aten.add.Tensor"]["is_mutable"] is False
    assert report["ops"]["aten.add_.Tensor"]["is_mutable"] is True


def test_is_mutable_is_not_constant_over_the_implemented_ops():
    """The failure this work exists to remove, stated as its own check.

    `is_mutable` has been wrong in both directions -- always True while it was
    a method (a bound method is truthy), then always False once it was a
    property reading an empty argument list. Both times the shape of the defect
    was the same: a predicate that cannot take two values. This asserts the
    partition is non-trivial in *both* directions, which is the only thing that
    fails for either version.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    _, report = _schema_road_fixture()
    values = [v["is_mutable"] for v in report["ops"].values()]
    assert values.count(True) == 12, values.count(True)
    assert values.count(False) == 105, values.count(False)
    assert None not in values


def test_schema_text_survives_the_round_trip_through_the_transcribed_tables():
    """The normalisation, checked against an oracle that is already in the repo.

    `native_functions.yaml` is not quite what upstream prints: it spells float
    defaults `0`/`1.0` where upstream prints `0.`/`1.`, quotes strings with
    `'`, writes `ScalarType? dtype=long` where upstream writes `=4`, and leaves
    `SymInt[2] stride=1` unexpanded. So the yaml text has to be re-printed the
    way upstream's `FunctionSchema` printer does, and a re-printer is a place to
    be quietly wrong.

    `overloads.json` and `methods.json` are the oracle: every string in them is
    `str(torch.ops.aten.<op>.<ov>._schema)` transcribed from upstream 2.13.0,
    and `verify_schemas.py` keeps them honest. 173 distinct overloads, 7 of
    which exercise a normalisation rule -- if the re-printer drops one, those
    seven stop matching. This needs no upstream torch.

    The provenance assertion is not decoration. In the first working version
    `_get_schema` consulted the tables *before* the file, so these 173 lookups
    were answered by the oracle itself and the comparison was the oracle
    against itself: deleting the float printer entirely left this test green
    (measured). 169 of the 173 have to come from the file for the comparison to
    mean anything, and the four that cannot -- `.out` variants torchgen
    generates and the file does not declare -- are named.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    keys, report = _schema_road_fixture()
    declared = _native_functions_keys()
    mismatched = []
    shadowed = []
    for (qualname, overload), expected in sorted(keys.items()):
        got = report["table"][f"{qualname}|{overload}"]
        if got["placeholder"] or got["text"] != expected:
            mismatched.append((qualname, overload, expected, got["text"]))
        if (qualname, overload) in declared and got["from"] != "native_functions.yaml":
            shadowed.append((qualname, overload, got["from"]))
    assert mismatched == [], mismatched[:5]
    assert shadowed == [], shadowed[:5]
    assert len(keys) == 173, len(keys)
    from_tables = sorted(
        k for k in keys
        if report["table"][f"{k[0]}|{k[1]}"]["from"] == "tables"
    )
    assert from_tables == [
        ("aten::div", "Scalar_mode_out"),
        ("aten::div", "Scalar_out"),
        ("aten::embedding", "out"),
        ("aten::empty_like", "out"),
    ], from_tables


def test_a_predicate_answered_without_text_is_counted_as_such():
    """"I do not know" has to be distinguishable from "no" -- and countable.

    Raising from `is_mutable` was tried first, because a refusal is the version
    of "I do not know" that cannot be mistaken for an answer. It does not
    survive contact with the tree: with the refusal in, a full run reads
    `is_mutable` on 102 ops with no text, and `import transformers` stops on
    the first (`aten::convolution_`, then `aten::_native_batch_norm_legit_
    functional`). 84 of those 102 are names upstream has no operator for at
    all -- `torch/distributed/tensor/_ops/autogen.py` synthesises `<base>_` and
    `<base>_functional` and probes them, and `torch/_ops.py` asks every packet
    for a `default` overload -- so upstream answers with AttributeError and the
    caller's guard reaches the same branch that `False` reaches here.

    So the placeholder answers, and the answer is recorded. This test pins the
    receipt, not the value: the pair is in `_shim_unanswered_predicates()`, so
    the set can be diffed instead of rediscovered by instrumenting the build.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    _, report = _schema_road_fixture()
    absent = "aten::not_an_operator_this_repo_will_ever_have"
    assert report["absent_placeholder"] is True
    assert report["absent_text"] == f"{absent}(...) -> ...", report["absent_text"]
    assert report["absent_is_mutable"] is False
    assert report["absent_is_view_op"] is False
    assert absent in [entry[0] for entry in report["placeholders_listed"]], \
        report["placeholders_listed"]
    unanswered = {tuple(entry) for entry in report["unanswered"]}
    assert (absent, "is_mutable") in unanswered, sorted(unanswered)[:10]
    assert (absent, "_is_view_op()") in unanswered, sorted(unanswered)[:10]


def test_no_operator_upstream_has_is_answered_from_an_empty_schema():
    """The property the transcribed table exists to hold.

    A predicate answered without text is only harmless while the op it is about
    does not exist. Every op that *does* exist and gets asked has to have text,
    or the answer is a claim about a real operator -- which is what
    `aten::native_dropout_backward.out` was: mutable upstream, False here,
    silently, because `torchgen` generates its schema at build time and no data
    file in this tree carries it.

    Stated as an equivalence over the recorded set rather than as a list of
    ops. `verify_schemas.py --unanswered` is the half that needs a real torch:
    it takes this same set and asks upstream whether each op exists. Here, with
    no upstream available, what is checkable is that every recorded op is one
    the shim itself has no text for by *either* route -- the file or the
    transcribed tables -- so a regression that stops consulting one of them
    shows up as a longer list.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    _, report = _schema_road_fixture()
    keys = _native_functions_keys()
    table = _schema_table_keys()
    leaked = []
    for spelling, _predicate in report["unanswered"]:
        qualname, _, overload = spelling.partition(".")
        if not qualname.startswith("aten::"):
            continue
        if (qualname, overload) in keys or (qualname, overload) in table:
            leaked.append(spelling)
    assert leaked == [], sorted(set(leaked))[:10]
    # `_fused_adam` and the sixteen beside it are in `_GENERATED_ATEN_SCHEMA_TEXT`
    # now, so they must *not* be in the recorded set at all.
    for gone in ("aten::native_dropout_backward.out",
                 "aten::_native_batch_norm_legit_functional",
                 "aten::_fused_adam",
                 "aten::add"):
        assert gone not in [s for s, _ in report["unanswered"]], gone


def test_every_remaining_placeholder_is_an_op_the_source_really_lacks():
    """Placeholder must mean "absent from the source", not "lookup missed".

    `_shim_placeholder_schemas()` is the readable form of the gap, the same
    shape as `_shim_registrations` and `_shim_overloads`. Asserting the list is
    empty would be false -- `import torch` asks about 345 ops this build has no
    text for, most of them `prims::` (defined through `Library.define` in
    `torch/_prims`) and aten *packet* names with no default overload
    (`aten::add` is `add.Tensor`/`add.Scalar` upstream and nothing else). So
    what is asserted is the implication: every aten placeholder is a key
    `native_functions.yaml` genuinely does not carry.

    This is the check that fails for the defect that actually happened. The
    first working version of this table keyed the default overload as `""` and
    was asked for `"default"` by `torch/_library/effects.py:55`, so 201 aten
    ops came back empty from a file that has every one of them. Both this test
    and `test_every_implemented_op_has_schema_text` were green at the time --
    the implemented 117 arrive spelled `""` -- and only the equivalence below
    names the ones that did not.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    _, report = _schema_road_fixture()
    keys = _native_functions_keys()
    table = _schema_table_keys()
    wrongly_placeheld = []
    for qualname, overload in report["placeholders_listed"]:
        if not qualname.startswith("aten::"):
            continue  # another namespace; the yaml says nothing about those
        if (qualname, overload) in keys or (qualname, overload) in table:
            wrongly_placeheld.append((qualname, overload))
    assert wrongly_placeheld == [], wrongly_placeheld[:10]
    # And the list is not empty for a trivial reason -- if nothing were ever
    # asked, the implication above would hold vacuously.
    assert len(report["placeholders_listed"]) > 100, len(report["placeholders_listed"])


def test_an_in_place_variant_the_file_does_not_declare_is_not_an_operator():
    """The registry stops inventing `<base>_`, and stops only that.

    `torch/distributed/tensor/_ops/autogen.py:244` builds `f"{base_name}_"` and
    asks the packet whether it is mutable, to discover whether an in-place
    variant exists. Upstream answers AttributeError -- there is no
    `aten::convolution_` -- and this shim answered with an operator carrying an
    empty schema, so the question was answered by the absence of arguments
    rather than by the operator. The schema layer cannot fix that; the packet
    has to not exist.

    `native_functions.yaml` is a sound oracle for exactly this question and not
    for the general one. Measured on 2.13.0: of the 1348 names of the form
    `<yaml base>_` the file does not declare, upstream registers none. Of the
    aten names the file lacks *in general* there are 176, and `quantized_lstm`
    is one -- `torch/__init__.py:2395` reads it unconditionally, so a rule
    keyed on "absent from the file" stops `import torch` there. It did; that is
    why the rule is the narrow one, and why `quantized_lstm` is asserted here
    beside `convolution_`.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    _, report = _schema_road_fixture()
    registry = report["registry"]
    for absent in ("convolution_", "mm_"):
        assert isinstance(registry[absent], str), (absent, registry[absent])
        assert "no attribute" in registry[absent], (absent, registry[absent])
    for present in ("add_", "relu_", "quantized_lstm", "zero"):
        assert not isinstance(registry[present], str), (present, registry[present])


def test_the_bare_shim_says_why_it_has_no_schema_table():
    """No `torchgen` beside it, and it says so rather than answering False.

    This file's `_C` is loaded from a staging directory with no vendored tree,
    which is exactly the shape of a build where the data file went missing. The
    contract is that the shim reports the absence -- `_shim_schema_source()`
    returns the reason instead of a path -- and that predicates over the
    schemas it consequently lacks refuse.
    """
    source = _C._shim_schema_source()
    assert isinstance(source, str) and source, source
    if source.endswith("native_functions.yaml"):
        return  # a tree is on the path after all; the vendored tests cover it
    assert "native_functions.yaml" in source, source
    schema = _C._get_schema("aten::add_", "Tensor")
    assert schema.is_placeholder is True
    assert str(schema) == "aten::add_.Tensor(...) -> ..."
    assert schema.is_mutable is False  # no text, so no writer to find
    assert ("aten::add_.Tensor", "is_mutable") in _C._shim_unanswered_predicates()
    # The transcribed tables still answer without the file, so the shim is not
    # uniformly blind here -- `_c10d_functional` and the generated aten
    # schemas are compiled into the artefact.
    wired = _C._get_schema("_c10d_functional::all_reduce_", "")
    assert wired.is_placeholder is False
    assert wired.is_mutable is True


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
