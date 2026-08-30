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
    # `.default`. `cumprod` has no kernel and so no table entry either --
    # `relu` used to be this example until docs/SPELLINGS.md §6 gave it one,
    # and `flatten` was it until docs/ARCH20.md §5 gave `cohere` a composite.
    #
    # The example keeps having to move, and that is the point rather than an
    # annoyance: it can only be a name that is *still* unreachable, so choosing
    # one is choosing a claim that gets falsified the day someone implements
    # it. `cumprod` is `cumsum`'s sibling and this shim has `cumsum`, which
    # makes it exactly the kind of name a future round is likely to reach for.
    try:
        _vf("cumprod")(1)
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


# --- _grouped_mm (docs/GROUPED_MM.md) ---------------------------------------
#
# `tools/golden/cases.py` compares this op against upstream on 64 cases; what
# these add is the part that is worth being able to check *without* upstream
# torch installed -- the offset convention, and the refusals, stated as
# hand-checkable arithmetic.
#
# Every shape here is a multiple of four in its last two dimensions, and that
# is not a coincidence: upstream's CPU kernel refuses operands whose
# last-two-dimension strides are not a multiple of 16 bytes, so a float32 case
# with an inner extent of 3 is a refusal, not a test.

_GROUPED_A = [1.0, 2.0, 3.0, 4.0,
              5.0, 6.0, 7.0, 8.0,
              9.0, 10.0, 11.0, 12.0,
              13.0, 14.0, 15.0, 16.0]
# Two experts: the identity, and twice the identity. Chosen so the answer is
# readable -- a row multiplied by expert 0 comes back unchanged, and a row
# multiplied by expert 1 comes back doubled, so *which expert saw which row* is
# visible in the output rather than inferred from it.
_GROUPED_B = [1.0, 0.0, 0.0, 0.0,
              0.0, 1.0, 0.0, 0.0,
              0.0, 0.0, 1.0, 0.0,
              0.0, 0.0, 0.0, 1.0,
              2.0, 0.0, 0.0, 0.0,
              0.0, 2.0, 0.0, 0.0,
              0.0, 0.0, 2.0, 0.0,
              0.0, 0.0, 0.0, 2.0]


def _grouped_operands():
    a = _C._tensor_from_flat(_GROUPED_A, [4, 4])
    b = _C._tensor_from_flat(_GROUPED_B, [2, 4, 4])
    return a, b


def test_grouped_mm_reads_offsets_as_cumulative_ends_not_lengths():
    # offs=[1, 3] over four rows. Read as cumulative ends -- which is what it
    # is -- expert 0 takes row 0, expert 1 takes rows 1 and 2, and row 3 is
    # written by nobody. Read as *lengths*, expert 1 would take rows 1..3 and
    # row 3 would come back doubled instead of empty. That single row is the
    # whole difference between the two readings.
    a, b = _grouped_operands()
    offs = _C._tensor_from_flat([1, 3], [2], dtype=_C.int32)
    out = _C._aten_dispatch("aten._grouped_mm.default", a, b, offs).tolist()
    assert out[0] == [1.0, 2.0, 3.0, 4.0], out[0]
    assert out[1] == [10.0, 12.0, 14.0, 16.0], out[1]
    assert out[2] == [18.0, 20.0, 22.0, 24.0], out[2]
    # Upstream leaves this row uninitialised and `transformers` masks it rather
    # than reading it; this shim fills it with zeros so the answer is at least
    # deterministic. Nothing in the golden suite compares it -- see
    # docs/GROUPED_MM.md §2.3.
    assert out[3] == [0.0, 0.0, 0.0, 0.0], out[3]


def test_grouped_mm_gives_an_empty_group_no_rows():
    # A repeated offset is an expert that routed no tokens, which is the normal
    # state of most experts on a short prompt. offs=[0, 4] means expert 0 got
    # nothing and expert 1 got everything, so every row comes back doubled.
    a, b = _grouped_operands()
    offs = _C._tensor_from_flat([0, 4], [2], dtype=_C.int32)
    out = _C._aten_dispatch("aten._grouped_mm.default", a, b, offs).tolist()
    assert out == [[2.0, 4.0, 6.0, 8.0],
                   [10.0, 12.0, 14.0, 16.0],
                   [18.0, 20.0, 22.0, 24.0],
                   [26.0, 28.0, 30.0, 32.0]], out


def test_grouped_mm_partitions_the_contraction_when_both_operands_are_2d():
    # The layout whose output rank goes *up*: (M,K) x (K,N) with offsets over
    # K gives (G,M,N), one matrix per group, because the groups do not share an
    # output. [1,2,3,4] . [1,1,1,1] split as K=[0,2) and [2,4) is 3 and 7.
    a = _C._tensor_from_flat([1.0, 2.0, 3.0, 4.0], [1, 4])
    b = _C._tensor_from_flat([1.0] * 16, [4, 4])
    offs = _C._tensor_from_flat([2, 4], [2], dtype=_C.int32)
    out = _C._aten_dispatch("aten._grouped_mm.default", a, b, offs)
    assert list(out.shape) == [2, 1, 4], list(out.shape)
    assert out.tolist() == [[[3.0, 3.0, 3.0, 3.0]], [[7.0, 7.0, 7.0, 7.0]]], out.tolist()


def test_grouped_mm_is_bmm_when_both_operands_are_3d_and_refuses_offsets_there():
    a = _C._tensor_from_flat(_GROUPED_A, [1, 4, 4])
    b = _C._tensor_from_flat(_GROUPED_B[16:], [1, 4, 4])  # twice the identity
    out = _C._aten_dispatch("aten._grouped_mm.default", a, b)
    assert out.tolist() == [[[2.0, 4.0, 6.0, 8.0],
                             [10.0, 12.0, 14.0, 16.0],
                             [18.0, 20.0, 22.0, 24.0],
                             [26.0, 28.0, 30.0, 32.0]]], out.tolist()
    offs = _C._tensor_from_flat([4], [1], dtype=_C.int32)
    try:
        _C._aten_dispatch("aten._grouped_mm.default", a, b, offs)
    except RuntimeError as e:
        assert "no offset if both matrices are 3d" in str(e), str(e)
    else:
        raise AssertionError("two 3-D operands must not accept offsets")


def test_grouped_mm_requires_int32_offsets():
    a, b = _grouped_operands()
    offs = _C._tensor_from_flat([1, 4], [2], dtype=_C.int64)
    try:
        _C._aten_dispatch("aten._grouped_mm.default", a, b, offs)
    except RuntimeError as e:
        assert "Offsets have to be int32" in str(e), str(e)
    else:
        raise AssertionError("int64 offsets must be refused, as upstream refuses them")


def test_grouped_mm_refuses_bias_and_a_foreign_out_dtype():
    # Both are in the schema and neither is implemented upstream. Computing
    # either one here would answer a question torch declines to answer.
    a, b = _grouped_operands()
    offs = _C._tensor_from_flat([1, 4], [2], dtype=_C.int32)
    bias = _C._tensor_from_flat([0.0] * 4, [4])
    try:
        _C._aten_dispatch("aten._grouped_mm.default", a, b, offs, bias=bias)
    except RuntimeError as e:
        assert "Bias not supported yet" in str(e), str(e)
    else:
        raise AssertionError("bias is unimplemented upstream and must be refused, not computed")
    try:
        _C._aten_dispatch("aten._grouped_mm.default", a, b, offs, out_dtype=_C.float16)
    except RuntimeError as e:
        assert "output dtype must match" in str(e), str(e)
    else:
        raise AssertionError("out_dtype other than mat_a's is refused upstream")


def test_grouped_mm_refuses_operands_upstreams_cpu_kernel_cannot_align():
    # candle would multiply these happily. The refusal exists because
    # upstream's CPU kernel has it -- 16 bytes is 4 float32 elements, and a
    # contraction of 3 is not a multiple of that. docs/GROUPED_MM.md §2.2.
    a = _C._tensor_from_flat([1.0] * 12, [4, 3])
    b = _C._tensor_from_flat([1.0] * 24, [2, 3, 4])
    offs = _C._tensor_from_flat([1, 4], [2], dtype=_C.int32)
    try:
        _C._aten_dispatch("aten._grouped_mm.default", a, b, offs)
    except RuntimeError as e:
        assert "16 bytes" in str(e), str(e)
    else:
        raise AssertionError("computing where upstream raises is silent divergence")


def test_grouped_mm_refuses_a_dtype_the_kernel_has_no_gemm_for():
    # float64 is the interesting one: this shim HAS a float64 matmul and
    # `mm`/`bmm`/`addmm` all use it, so the refusal is fidelity to upstream's
    # kernel rather than a missing capability.
    a = _C._tensor_from_flat([1.0] * 16, [4, 4], dtype=_C.float64)
    b = _C._tensor_from_flat([1.0] * 32, [2, 4, 4], dtype=_C.float64)
    offs = _C._tensor_from_flat([1, 4], [2], dtype=_C.int32)
    try:
        _C._aten_dispatch("aten._grouped_mm.default", a, b, offs)
    except RuntimeError as e:
        assert "Float32, BFloat16 or Float16" in str(e), str(e)
    else:
        raise AssertionError("float64 is refused upstream even though candle can multiply it")


def test_grouped_mm_resolves_from_the_torch_level_name():
    """...and now it does. The predicate that stopped it, and what it cost.

    `torch._grouped_mm` is what `torch.nn.functional.grouped_mm` calls, and
    that is the route `transformers`' MoE layer takes on CPU. `overloads.json`
    carried the entry and the kernel was reachable through
    `torch.ops.aten._grouped_mm.default`, but the *name* refused, because
    `bootstrap.py`'s table constructor dropped it:

        overloads = {... for name, schemas in json.loads(overloads_json).items()
                     if not name.startswith("_")}      # the table's embedded README

    The comment said README and the predicate said every underscore-prefixed
    key, so an aten op whose name began with `_` could not have a `torch.<op>`
    binding at all. `methods.json`'s sibling comprehension six lines below had
    always spelled the same intent correctly as `startswith("_README")`;
    narrowing this one to match was the whole fix. docs/GROUPED_MM.md §6.1.

    This test asserted the broken state until the fix landed, so that it could
    not land silently. It now asserts the fixed state, and asserts the *scope*
    of the widening as well: which underscore-prefixed keys the wider predicate
    admits, enumerated rather than assumed harmless. It said "if a second
    underscore-prefixed op is added to `overloads.json` later, that assertion
    is where it announces itself" -- and one was:

    `_safe_softmax` (docs/TRIL.md §2). The leading underscore is why it went
    missing in the first place. It reads as private, so docs/ARCH20.md §9's
    inventory filed it under "no such public function upstream" without
    checking; `hasattr(torch, '_safe_softmax')` is `True` on 2.13.0, it fires
    `aten._safe_softmax.default` as a leaf op, and this shim has had that
    kernel golden-compared since docs/SDPA.md. Two refusals in
    `scaled_dot_product_attention` meanwhile named it as a kernel that did not
    exist -- see `test_the_two_stale_sdpa_refusals_no_longer_claim_a_missing_kernel`.
    """
    fn = getattr(_C._VariableFunctions, "_grouped_mm")
    assert fn is not None
    assert _C._shim_overloads["_grouped_mm"] == ["aten._grouped_mm.default"], (
        _C._shim_overloads["_grouped_mm"]
    )
    # The whole difference the widened predicate makes, enumerated rather
    # than assumed harmless: `_README` is still excluded (it is a list of
    # prose, not schemas, and admitting it would fail to parse), and these
    # are the underscore-prefixed keys that reach a `torch.<name>` because of
    # it. Both have kernels; neither would resolve under the old predicate.
    admitted = sorted(n for n in _C._shim_overloads if n.startswith("_"))
    assert admitted == ["_grouped_mm", "_safe_softmax"], admitted
    assert _C._shim_overloads["_safe_softmax"] == ["aten._safe_softmax.default"], (
        _C._shim_overloads["_safe_softmax"]
    )

    result = fn(
        _C._tensor_from_flat([1.0] * 16, [4, 4]),
        _C._tensor_from_flat([1.0] * 32, [2, 4, 4]),
        offs=_C._tensor_from_flat([1, 4], [2], dtype=_C.int32),
    )
    # The name and the key reach the same kernel, so they must agree exactly.
    direct = _C._aten_dispatch(
        "aten._grouped_mm.default",
        _C._tensor_from_flat([1.0] * 16, [4, 4]),
        _C._tensor_from_flat([1.0] * 32, [2, 4, 4]),
        _C._tensor_from_flat([1, 4], [2], dtype=_C.int32),
    )
    assert list(result.shape) == list(direct.shape) == [4, 4]
    assert result.tolist() == direct.tolist(), (result.tolist(), direct.tolist())
    # Keyword spelling too -- `torch/nn/functional.py:7139` passes all three
    # of `offs`, `bias` and `out_dtype`, two of them `None`.
    kwargs_form = fn(
        _C._tensor_from_flat([1.0] * 16, [4, 4]),
        _C._tensor_from_flat([1.0] * 32, [2, 4, 4]),
        offs=_C._tensor_from_flat([1, 4], [2], dtype=_C.int32),
        bias=None,
        out_dtype=None,
    )
    assert kwargs_form.tolist() == direct.tolist()


def test_the_mixtral_member_names_reach_the_kernels_that_were_already_there():
    """Seven `TensorBase` members, none of them a new operator.

    docs/GROUPED_MM.md §6.4 measured what stopped Mixtral from *executing*
    after `_grouped_mm` took it to zero missing operators, and every item was
    a name with a kernel already in `_aten_implemented()`. Five are
    `methods.json` entries (`__idiv__`, `__ge__`, `clamp_`, `masked_fill_`,
    `div_`, plus `ge` for symmetry with `le`/`lt`/`gt`); `chunk` and
    `__setitem__` are Python-level, for the reasons their bootstrap.py
    docstrings give.

    Each is asserted through the *member*, not through `_aten_dispatch`: the
    dispatch keys were reachable all along, so a test that called them would
    have passed before the fix and proved nothing.
    """
    def t(flat, shape, dtype=None):
        kw = {} if dtype is None else {"dtype": dtype}
        return _C._tensor_from_flat(list(flat), list(shape), **kw)

    # `torch/_tensor.py:1115` assigns `Tensor.__itruediv__ = TensorBase.__idiv__`.
    x = t([4.0, 8.0], [2])
    x.__idiv__(t([2.0, 2.0], [2]))
    assert x.tolist() == [2.0, 4.0], x.tolist()
    # In place, so the receiver is the thing that changed.
    y = t([4.0, 8.0], [2])
    assert y.div_(t([2.0, 4.0], [2])).tolist() == [2.0, 2.0]
    assert y.tolist() == [2.0, 2.0], y.tolist()

    # `__le__`/`__gt__`/`__lt__` all existed; only `__ge__` was absent.
    g = t([1.0, 3.0, 5.0], [3])
    assert g.__ge__(3).tolist() == [False, True, True]
    assert g.ge(3).tolist() == [False, True, True]
    assert (g >= 3).tolist() == [False, True, True]
    # ...and the Tensor overload behind those same three spellings, which had
    # no kernel when §6.4 was written: the name bound and `_aten_dispatch`
    # refused. `>=` and `>` differ only on the equal element, so the middle
    # entry is what says this reached `Cmp::Ge` rather than `Cmp::Gt`.
    other = t([3.0, 3.0, 3.0], [3])
    assert (g >= other).tolist() == [False, True, True]
    assert g.ge(other).tolist() == [False, True, True]
    assert g.__ge__(other).tolist() == [False, True, True]
    assert (g > other).tolist() == [False, False, True]
    assert "aten.ge.Tensor" in _C._aten_implemented()

    # `expert_ids_g.clamp_(max=...)` -- max only, min absent.
    c = t([1.0, 5.0, 10.0, -3.0], [4])
    assert c.clamp_(max=6).tolist() == [1.0, 5.0, 6.0, -3.0]
    assert c.tolist() == [1.0, 5.0, 6.0, -3.0], c.tolist()

    m = t([1.0, 2.0, 3.0, 4.0], [4])
    mask = t([True, False, True, False], [4], dtype=_C.bool)
    assert m.masked_fill_(mask, 0.0).tolist() == [0.0, 2.0, 0.0, 4.0]
    assert m.tolist() == [0.0, 2.0, 0.0, 4.0], m.tolist()

    # `chunk` is upstream's composite, not an even division: `chunks` is an
    # upper bound on how many pieces come back.
    ten = _C._aten_dispatch("aten.arange.default", 10)
    assert [c.tolist() for c in ten.chunk(3)] == [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9]]
    assert [c.tolist() for c in ten.chunk(4)] == [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9]]
    assert [len(c.tolist()) for c in ten.chunk(5)] == [2, 2, 2, 2, 2]
    three = _C._aten_dispatch("aten.arange.default", 3)
    assert len(three.chunk(7)) == 3, "chunks is an upper bound, not a promise"
    assert isinstance(ten.chunk(3), tuple), "upstream's THPVariable_chunk returns a tuple"

    # `inv_perm[perm] = torch.arange(perm.size(0))` -- the missing half of
    # `__getitem__`, and the one shape of it this shim can do correctly.
    s = t([0.0] * 5, [5])
    s[t([0, 2, 4], [3], dtype=_C.int64)] = t([7.0, 8.0, 9.0], [3])
    assert s.tolist() == [7.0, 0.0, 8.0, 0.0, 9.0], s.tolist()
    whole = t([0.0] * 3, [3])
    whole[:] = t([1.0, 2.0, 3.0], [3])
    assert whole.tolist() == [1.0, 2.0, 3.0], whole.tolist()
    fill = t([0.0] * 3, [3])
    fill[...] = 4.0
    assert fill.tolist() == [4.0, 4.0, 4.0], fill.tolist()

    for name in ("div_", "__idiv__", "ge", "__ge__", "clamp_", "masked_fill_"):
        assert name in _C._shim_methods, name
    # ...and the two that are deliberately not table entries.
    assert "chunk" not in _C._shim_methods
    assert "__setitem__" not in _C._shim_methods


def test_setitem_writes_the_basic_index_through_to_the_base():
    """`x[0] = v`, and the one spelling of it that still refuses.

    **This test used to assert the opposite** -- it was written as the signal
    for docs/VIEWS.md §4, asserting through a probe that a write via
    `select.int` did *not* reach the base, so that it would go red the day
    views became mutable. It did, and this is the other side of it.

    Every assertion reads the ORIGINAL name after the write and never the
    value the op returned. Every in-place op returns `self`, so a test that
    reads the return value cannot tell a write-through from a
    write-into-a-fresh-buffer; that is exactly how the old behaviour survived
    3037 green golden cases.

    The refusal that is left is a slice with `step != 1`, and it is not a
    view problem in the same sense: `slice.Tensor` above step 1 reaches its
    result through `index_select`, which materialises, so there is no shared
    buffer to write into. docs/VIEWS.md §6.4.
    """
    # The probe the old version of this test asserted the negative of.
    x = _C._tensor_from_flat([0.0] * 5, [5])
    view = _C._aten_dispatch("aten.select.int", x, 0, 1)
    _C._aten_dispatch("aten.copy_.default", view, _C._tensor_from_flat([3.0], []))
    assert x.tolist() == [0.0, 3.0, 0.0, 0.0, 0.0], x.tolist()

    sliced = _C._aten_dispatch("aten.slice.Tensor", x, 0, 1, 3, 1)
    _C._aten_dispatch("aten.copy_.default", sliced, _C._tensor_from_flat([1.0, 2.0], [2]))
    assert x.tolist() == [0.0, 1.0, 2.0, 0.0, 0.0], x.tolist()

    # ...and through the subscript, where there is no return value at all.
    # A (3,4) grid so that a column is a genuinely strided destination.
    grid = lambda: _C._tensor_from_flat([float(v) for v in range(1, 13)], [3, 4])

    a = grid()
    a[0] = 3.0
    assert a.tolist() == [[3.0] * 4, [5.0, 6.0, 7.0, 8.0], [9.0, 10.0, 11.0, 12.0]], a.tolist()

    b = grid()
    b[:, 1] = 7.0
    assert [row[1] for row in b.tolist()] == [7.0, 7.0, 7.0], b.tolist()
    assert [row[0] for row in b.tolist()] == [1.0, 5.0, 9.0], (
        "the strided write spilled into a neighbouring column"
    )

    c = grid()
    c[1, 2] = 99.0
    assert c.tolist()[1][2] == 99.0 and c.tolist()[1][1] == 6.0, c.tolist()

    d = grid()
    d[1:3] = 0.0
    assert d.tolist() == [[1.0, 2.0, 3.0, 4.0], [0.0] * 4, [0.0] * 4], d.tolist()

    e = grid()
    e[-1] = -1.0
    assert e.tolist()[2] == [-1.0] * 4 and e.tolist()[0] == [1.0, 2.0, 3.0, 4.0], e.tolist()

    # A view of a view: the write has to travel two narrowings back to the base.
    f = grid()
    row = f[1]
    row[2] = 42.0
    assert f.tolist()[1][2] == 42.0, f.tolist()

    # A step above 1 still refuses, by name, and names the reason.
    g = grid()
    try:
        g[0:3:2] = 0.0
    except NotImplementedError as error:
        assert "step-2" in str(error), str(error)
    else:
        raise AssertionError("a step-2 slice write was silently accepted")
    assert g.tolist()[0] == [1.0, 2.0, 3.0, 4.0], "the refused write happened anyway"


def test_which_ops_share_storage_with_their_input_and_which_do_not():
    """The aliasing table, asserted op by op, in both directions.

    Before docs/VIEWS.md §6 this could not have been written: no in-place op
    reached storage, so every one of these answers was `independent` whatever
    the op did. Making one write go through makes **all twenty-eight of them
    correctness questions at once**, which is the reason this table is a test
    and not a paragraph.

    The instrument is the only one that can answer it from outside candle
    (`same_storage` is `pub(crate)`): write into the result and read the
    input. The expectations here were measured against torch 2.13.0 with the
    same script run twice, and 26 of the 28 agree -- the two that do not have
    their own test above.

    **The `independent` half is not the boring half.** `_to_copy` with nothing
    to convert used to alias, because candle's `to_device`/`fast_to` return
    `self.clone()` on a no-op and a candle clone is an `Arc` clone. Deleting
    the fix left the entire golden suite green, because every other case reads
    the op's *result* and the result was correct -- only the input can fail.
    That defect is a corruption rather than a lost write, which makes this
    direction the more important one.
    """
    def base():
        return _C._tensor_from_flat([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])

    def wrote_through(build):
        x = base()
        before = x.tolist()
        result = build(x)
        _C._aten_dispatch("aten.fill_.Scalar", result, 0.0)
        return x.tolist() != before

    d = _C._aten_dispatch

    shares = {
        "aten.alias.default": lambda x: d("aten.alias.default", x),
        "aten.detach.default": lambda x: d("aten.detach.default", x),
        "aten.contiguous.default": lambda x: d("aten.contiguous.default", x),
        "aten.expand.default": lambda x: d("aten.expand.default", x, [2, 3]),
        "aten.permute.default": lambda x: d("aten.permute.default", x, [1, 0]),
        "aten.select.int": lambda x: d("aten.select.int", x, 0, 1),
        "aten.select.int (dim 1)": lambda x: d("aten.select.int", x, 1, 0),
        "aten.slice.Tensor (step 1)": lambda x: d("aten.slice.Tensor", x, 0, 0, 1, 1),
        "aten.squeeze.dim": lambda x: d("aten.squeeze.dim", d("aten.unsqueeze.default", x, 0), 0),
        "aten.t.default": lambda x: d("aten.t.default", x),
        "aten.transpose.int": lambda x: d("aten.transpose.int", x, 0, 1),
        "aten.unsqueeze.default": lambda x: d("aten.unsqueeze.default", x, 0),
        "aten.view.default": lambda x: d("aten.view.default", x, [6]),
        "aten._unsafe_view.default": lambda x: d("aten._unsafe_view.default", x, [6]),
        "aten.lift_fresh.default": lambda x: d("aten.lift_fresh.default", x),
        "aten.split.Tensor[0]": lambda x: d("aten.split.Tensor", x, 1)[0],
        "aten.split_with_sizes.default[0]": lambda x: d("aten.split_with_sizes.default", x, [1, 1])[0],
        "aten.unbind.int[0]": lambda x: d("aten.unbind.int", x, 0)[0],
    }
    independent = {
        "aten.clone.default": lambda x: d("aten.clone.default", x),
        "aten.contiguous.default (of a strided input)":
            lambda x: d("aten.contiguous.default", d("aten.t.default", x)),
        # The one that had no test and needed one -- see the docstring.
        "aten._to_copy.default (nothing to convert)":
            lambda x: d("aten._to_copy.default", x),
        "aten._to_copy.default (dtype change)":
            lambda x: d("aten._to_copy.default", x, _C.float64),
        "aten.zeros_like.default": lambda x: d("aten.zeros_like.default", x),
        "aten.empty_like.default": lambda x: d("aten.empty_like.default", x),
        "aten.neg.default": lambda x: d("aten.neg.default", x),
        "aten.abs.default": lambda x: d("aten.abs.default", x),
        "aten.relu.default": lambda x: d("aten.relu.default", x),
        "aten.masked_fill.Scalar": lambda x: d(
            "aten.masked_fill.Scalar", x,
            _C._tensor_from_flat([1, 0, 1, 0, 1, 0], [2, 3], dtype=_C.bool), 9.0),
        # `repeat` materialises, always -- and the all-ones row is the one
        # that matters. candle's `Tensor::repeat` skips every repeat that is
        # not `> 1`, so `repeat([1, 1])` returns `self.clone()`, and a candle
        # clone is an `Arc` clone. That is the `_to_copy` defect this table's
        # docstring describes, in a new op: correct values, corrupted input.
        "aten.repeat.default (tiling)": lambda x: d("aten.repeat.default", x, [2, 3]),
        "aten.repeat.default (all ones)": lambda x: d("aten.repeat.default", x, [1, 1]),
        "aten.repeat.default (rank raised)": lambda x: d("aten.repeat.default", x, [1, 1, 1]),
    }

    for name, build in sorted(shares.items()):
        assert wrote_through(build), (
            f"{name} no longer shares storage with its input -- upstream's does, so a "
            f"write through it is now silently lost"
        )
    for name, build in sorted(independent.items()):
        assert not wrote_through(build), (
            f"{name} shares storage with its input -- upstream's does not, so a write "
            f"through the result now CORRUPTS the input"
        )


def test_the_two_aliasing_relationships_that_still_diverge_are_pinned():
    """The write-lost divergences, asserted so they cannot be forgotten.

    Every other view-producing op in this shim shares storage with its input
    the way upstream does, and since docs/VIEWS.md §6 a write through any of
    them reaches the base. Two do not, and both are properties of candle's
    storage model rather than oversights:

      * **`slice.Tensor` with `step > 1`** goes through `index_select`, which
        copies. A stepped view would need a `Layout` whose stride is `step`
        over the input's storage, and the only public pairing of a storage
        with a layout is `Tensor::from_storage` -- contiguous-only, and
        taking a `Storage` that `Tensor::storage()` (`pub(crate)`) will not
        hand over.
      * **`view.dtype`** reinterprets bytes, and candle's `Layout` counts
        elements of a storage whose dtype is fixed. There is no reinterpreting
        layout to build at any visibility.

    Both have `expect="diverge"` golden cases as well, which compare the base
    against upstream and would fail if either silently started agreeing. This
    asserts the shim's half on its own, so the gap is recorded even when the
    golden harness is not being run.
    """
    x = _C._tensor_from_flat([1.0, 2.0, 3.0, 4.0], [4])
    stepped = _C._aten_dispatch("aten.slice.Tensor", x, 0, 0, 4, 2)
    _C._aten_dispatch("aten.fill_.Scalar", stepped, 0.0)
    assert stepped.tolist() == [0.0, 0.0], stepped.tolist()
    assert x.tolist() == [1.0, 2.0, 3.0, 4.0], (
        "a step-2 slice is now a view -- update docs/VIEWS.md §6.4, the diverge "
        "case in slice_cases, and __setitem__'s step refusal"
    )

    y = _C._tensor_from_flat([1, 2, 3, 4], [4], dtype=_C.int32)
    reinterpreted = _C._aten_dispatch("aten.view.dtype", y, _C.float32)
    _C._aten_dispatch("aten.fill_.Scalar", reinterpreted, 0.0)
    assert y.tolist() == [1, 2, 3, 4], (
        "view.dtype is now a view -- update docs/VIEWS.md §6.4 and the diverge "
        "case in view_dtype_cases"
    )


def test_every_in_place_op_writes_through_a_strided_view():
    """All twelve in-place keys, each through a **non-contiguous** view.

    A column of a `(3,4)` tensor has stride 4, so a write through it exercises
    `tensor.rs::write_strided`'s odometer rather than its `copy_from_slice`
    fast path -- and every assertion reads the base, never the return value.

    The two RNG ops are here for *reachability*, not for values: their streams
    are compared against upstream in the golden suite, and what this pins is
    that a draw through a view lands in the base's column and leaves the
    neighbouring columns alone. `zeros` cannot be the check, since a draw may
    legitimately produce one, so the assertion is on the columns that must
    NOT have moved.
    """
    def grid(dtype=None):
        kw = {} if dtype is None else {"dtype": dtype}
        return _C._tensor_from_flat([float(v) for v in range(1, 13)], [3, 4], **kw)

    def column(t, k):
        return [row[k] for row in t.tolist()]

    def untouched(t, moved):
        for k in range(4):
            if k == moved:
                continue
            assert column(t, k) == [1.0 + k, 5.0 + k, 9.0 + k], (
                f"the write through column {moved} disturbed column {k}: {t.tolist()}"
            )

    def view(t, k=1):
        return _C._aten_dispatch("aten.select.int", t, 1, k)

    # 1. fill_.Scalar
    a = grid()
    _C._aten_dispatch("aten.fill_.Scalar", view(a), 7.0)
    assert column(a, 1) == [7.0, 7.0, 7.0], a.tolist()
    untouched(a, 1)

    # 2. fill_.Tensor
    b = grid()
    _C._aten_dispatch("aten.fill_.Tensor", view(b), _C._tensor_from_flat([7.0], []))
    assert column(b, 1) == [7.0, 7.0, 7.0], b.tolist()
    untouched(b, 1)

    # 3. zero_
    c = grid()
    _C._aten_dispatch("aten.zero_.default", view(c))
    assert column(c, 1) == [0.0, 0.0, 0.0], c.tolist()
    untouched(c, 1)

    # 4. copy_
    d = grid()
    _C._aten_dispatch(
        "aten.copy_.default", view(d), _C._tensor_from_flat([-1.0, -2.0, -3.0], [3])
    )
    assert column(d, 1) == [-1.0, -2.0, -3.0], d.tolist()
    untouched(d, 1)

    # 5. add_.Tensor
    e = grid()
    _C._aten_dispatch(
        "aten.add_.Tensor", view(e), _C._tensor_from_flat([10.0, 10.0, 10.0], [3])
    )
    assert column(e, 1) == [12.0, 16.0, 20.0], e.tolist()
    untouched(e, 1)

    # 6. relu_
    f = _C._tensor_from_flat([float(v) for v in range(-6, 6)], [3, 4])
    _C._aten_dispatch("aten.relu_.default", view(f))
    assert column(f, 1) == [0.0, 0.0, 3.0], f.tolist()
    assert column(f, 0) == [-6.0, -2.0, 2.0], f.tolist()

    # 7. clamp_
    g = grid()
    _C._aten_dispatch("aten.clamp_.default", view(g), 3.0, 8.0)
    assert column(g, 1) == [3.0, 6.0, 8.0], g.tolist()
    untouched(g, 1)

    # 8. div_.Tensor
    h = grid()
    _C._aten_dispatch(
        "aten.div_.Tensor", view(h), _C._tensor_from_flat([2.0, 2.0, 2.0], [3])
    )
    assert column(h, 1) == [1.0, 3.0, 5.0], h.tolist()
    untouched(h, 1)

    # 9. masked_fill_.Scalar
    i = grid()
    _C._aten_dispatch(
        "aten.masked_fill_.Scalar", view(i),
        _C._tensor_from_flat([1, 0, 1], [3], dtype=_C.bool), -1.0,
    )
    assert column(i, 1) == [-1.0, 6.0, -1.0], i.tolist()
    untouched(i, 1)

    # 10. index_put_
    j = grid()
    _C._aten_dispatch(
        "aten.index_put_.default", view(j),
        [_C._tensor_from_flat([0, 2], [2], dtype=_C.int64)],
        _C._tensor_from_flat([-5.0, -6.0], [2]), False,
    )
    assert column(j, 1) == [-5.0, 6.0, -6.0], j.tolist()
    untouched(j, 1)

    # 11-12. uniform_ and normal_: reachability, plus the neighbours.
    for key, args in (
        ("aten.uniform_.default", (0.0, 1.0)),
        ("aten.normal_.default", (0.0, 1.0)),
    ):
        k = grid()
        _C._aten_dispatch(key, view(k), *args)
        assert column(k, 1) != [2.0, 6.0, 10.0], f"{key} wrote nothing: {k.tolist()}"
        untouched(k, 1)


def test_an_expanded_destination_follows_upstream_op_by_op():
    """Writing into a tensor that addresses one element twice.

    `expand` gives an axis stride 0, so several logical positions share one
    storage element. Upstream's answer is **a table, not a rule** (measured on
    torch 2.13.0): `fill_.Scalar`, `zero_`, `masked_fill_` and `index_put_`
    write; `fill_.Tensor`, `copy_`, `add_`, `relu_`, `clamp_`, `div_`,
    `uniform_` and `normal_` raise `unsupported operation: more than one
    element of the written-to tensor refers to a single memory location`.

    The `fill_` pair is the sharpest: two overloads of one kernel, one
    permitted and one refused, which is not derivable from anything and had to
    be measured. Before write-through the question could not arise at all --
    every in-place op replaced the wrapper, so writing "through" an expanded
    tensor wrote through nothing.
    """
    def expanded():
        base = _C._tensor_from_flat([1.0, 2.0], [1, 2])
        return base, _C._aten_dispatch("aten.expand.default", base, [3, 2])

    base, wide = expanded()
    _C._aten_dispatch("aten.fill_.Scalar", wide, 7.0)
    assert base.tolist() == [[7.0, 7.0]], base.tolist()
    assert wide.tolist() == [[7.0, 7.0]] * 3, wide.tolist()

    base, wide = expanded()
    _C._aten_dispatch("aten.zero_.default", wide)
    assert base.tolist() == [[0.0, 0.0]], base.tolist()

    refusing = [
        ("aten.fill_.Tensor", lambda w: (w, _C._tensor_from_flat([1.0], []))),
        ("aten.copy_.default", lambda w: (w, _C._tensor_from_flat([0.0] * 6, [3, 2]))),
        ("aten.add_.Tensor", lambda w: (w, _C._tensor_from_flat([1.0] * 6, [3, 2]))),
        ("aten.relu_.default", lambda w: (w,)),
        ("aten.clamp_.default", lambda w: (w, 0.0, 1.0)),
        ("aten.div_.Tensor", lambda w: (w, _C._tensor_from_flat([2.0] * 6, [3, 2]))),
        ("aten.uniform_.default", lambda w: (w, 0.0, 1.0)),
        ("aten.normal_.default", lambda w: (w, 0.0, 1.0)),
    ]
    for key, build in refusing:
        base, wide = expanded()
        try:
            _C._aten_dispatch(key, *build(wide))
        except RuntimeError as error:
            assert "more than one element of the written-to tensor" in str(error), str(error)
        else:
            raise AssertionError(f"{key} wrote into an expanded destination")
        assert base.tolist() == [[1.0, 2.0]], (
            f"{key} raised and wrote anyway: {base.tolist()}"
        )


def test_index_put_takes_a_mask_a_matrix_and_a_number():
    """The three `index_put_` gaps docs/GROUPED_MM.md §6.4 recorded, closed.

    All three were the same cause: the kernel delegated to `scatter.src`,
    which wants an int32/int64 index and index/src/self all of one rank. So a
    bool mask refused on dtype, a receiver of rank above 1 refused on rank,
    and `x[t] = 5` refused because the number lifts to a 0-d tensor. None of
    them was a missing operator and none of them was a missing name -- the
    walk in `bootstrap.py` emitted `aten.index_put_.default` with the right
    arguments the whole time.

    Every expectation below is upstream's, measured on torch 2.13.0. The
    golden suite diffs the values against upstream directly (43 cases on this
    key); this pins the shapes that were *refused* so a regression to the
    `scatter` delegation fails here by name rather than only by value.
    """
    def t(flat, shape, dtype=None):
        kw = {} if dtype is None else {"dtype": dtype}
        return _C._tensor_from_flat(list(flat), list(shape), **kw)

    # 1. A bool mask selects positions, which is a different operation from an
    #    integer index rather than a cast of one.
    m = t([0.0] * 4, [4])
    m[t([1, 0, 1, 0], [4], dtype=_C.bool)] = t([1.0, 2.0], [2])
    assert m.tolist() == [1.0, 0.0, 2.0, 0.0], m.tolist()
    # A uint8 mask is upstream's deprecated spelling of the same thing, and it
    # is emphatically NOT the integer index [1,0,1,0].
    u = t([0.0] * 4, [4])
    u[t([1, 0, 1, 0], [4], dtype=_C.uint8)] = t([1.0, 2.0], [2])
    assert u.tolist() == [1.0, 0.0, 2.0, 0.0], u.tolist()
    # The mask's RANK decides how many axes it eats: the same values as a
    # (2,3) mask write three elements and as a (2,) mask write a whole row.
    two_d = t([0.0] * 6, [2, 3])
    two_d[t([1, 0, 1, 0, 1, 0], [2, 3], dtype=_C.bool)] = t([1.0, 2.0, 3.0], [3])
    assert two_d.tolist() == [[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]], two_d.tolist()
    row = t([0.0] * 6, [2, 3])
    row[t([1, 0], [2], dtype=_C.bool)] = t([1.0, 2.0, 3.0], [3])
    assert row.tolist() == [[1.0, 2.0, 3.0], [0.0, 0.0, 0.0]], row.tolist()

    # 2. A receiver of rank above 1 with a 1-D index -- `x[idx] = v` on a
    #    matrix, which is the common case.
    mat = t([0.0] * 6, [3, 2])
    mat[t([0, 2], [2], dtype=_C.int64)] = t([1.0, 2.0, 3.0, 4.0], [2, 2])
    assert mat.tolist() == [[1.0, 2.0], [0.0, 0.0], [3.0, 4.0]], mat.tolist()
    # `values` broadcasts right-aligned, and the two broadcasts of the same
    # two numbers give different answers -- (2,) fills across, (2,1) down.
    across = t([0.0] * 6, [3, 2])
    across[t([0, 2], [2], dtype=_C.int64)] = t([1.0, 2.0], [2])
    assert across.tolist() == [[1.0, 2.0], [0.0, 0.0], [1.0, 2.0]], across.tolist()
    down = t([0.0] * 6, [3, 2])
    down[t([0, 2], [2], dtype=_C.int64)] = t([1.0, 2.0], [2, 1])
    assert down.tolist() == [[1.0, 1.0], [0.0, 0.0], [2.0, 2.0]], down.tolist()

    # 3. `x[t] = 5`: a Python number, lifted to a 0-d tensor and broadcast.
    #    The lift follows the RECEIVER's dtype, not the number's Python type
    #    (measured), and `index_put_` refuses a dtype mismatch -- so getting
    #    the lift wrong refuses a write upstream performs.
    num = t([0.0] * 6, [3, 2])
    num[t([0, 2], [2], dtype=_C.int64)] = 5
    assert num.tolist() == [[5.0, 5.0], [0.0, 0.0], [5.0, 5.0]], num.tolist()
    trunc = t([0, 0, 0, 0], [4], dtype=_C.int64)
    trunc[t([0, 2], [2], dtype=_C.int64)] = 5.9
    assert trunc.tolist() == [5, 0, 5, 0], trunc.tolist()
    assert trunc.dtype == _C.int64
    # The pair that tells the two lift rules apart at the *kernel* level: `2`
    # inferred from the Python type is int64, and `index_put_` refuses a
    # dtype mismatch, so the old rule turned this write into a refusal.
    truthy = t([0, 0, 0], [3], dtype=_C.bool)
    truthy[t([0], [1], dtype=_C.int64)] = 2
    assert truthy.tolist() == [True, False, False], truthy.tolist()
    assert truthy.dtype == _C.bool
    # The other caller of the same lift, which reaches `fill_` and not
    # `index_put_`. It does NOT discriminate between the two lift rules --
    # `fill_` takes its dtype from the receiver either way -- and is here
    # because it is part of `__setitem__`'s measured behaviour.
    filled = t([0, 0, 0], [3], dtype=_C.int64)
    filled[:] = 3.0
    assert filled.tolist() == [3, 3, 3] and filled.dtype == _C.int64, filled.tolist()

    # `x[:, t] = v` -- the `[None, t]` index list, writing columns not rows.
    cols = t([0.0] * 6, [2, 3])
    cols[:, t([0, 2], [2], dtype=_C.int64)] = 5.0
    assert cols.tolist() == [[5.0, 0.0, 5.0], [5.0, 0.0, 5.0]], cols.tolist()

    # Negative indices wrap. `scatter` had no rule for this and refused them.
    neg = t([0.0] * 5, [5])
    neg[t([-1, -2], [2], dtype=_C.int64)] = t([1.0, 2.0], [2])
    assert neg.tolist() == [0.0, 0.0, 0.0, 2.0, 1.0], neg.tolist()

    # ...and out of range still refuses, in upstream's wording.
    try:
        t([0.0] * 5, [5])[t([9], [1], dtype=_C.int64)] = 1.0
    except IndexError as error:
        assert "index 9 is out of bounds for dimension 0 with size 5" in str(error), str(error)
    else:
        raise AssertionError("an out-of-range index must not be clamped")

    # An empty index and an all-false mask write nothing rather than refusing
    # or zeroing. The receiver is non-zero so both mistakes are visible.
    empty = t([1.0, 2.0, 3.0], [3])
    empty[t([], [0], dtype=_C.int64)] = t([], [0])
    assert empty.tolist() == [1.0, 2.0, 3.0], empty.tolist()

    # Still refused, and still by name: two index tensors, and accumulate.
    for label, call in (
        ("two index tensors", lambda: _C._aten_dispatch(
            "aten.index_put_.default",
            t([0.0] * 4, [2, 2]),
            [t([0], [1], dtype=_C.int64), t([1], [1], dtype=_C.int64)],
            t([1.0], []),
            False,
        )),
        ("accumulate=True", lambda: _C._aten_dispatch(
            "aten.index_put_.default",
            t([0.0] * 3, [3]), [t([0], [1], dtype=_C.int64)], t([1.0], [1]), True,
        )),
    ):
        try:
            call()
        except NotImplementedError as error:
            assert "torch._C shim" in str(error), str(error)
        else:
            raise AssertionError(f"{label} must still refuse by name")


def test_index_put_writes_into_the_receiver_and_not_into_a_copy():
    """The mutation, read back through the binding that was passed in.

    `index_put_` returns `self`, so every check that reads its return value
    passes just as well against a kernel that built a fresh tensor and handed
    it back. This reads the ORIGINAL name afterwards and throws the return
    value away, which is the only shape that can fail that way.

    The third block used to assert the *limitation* that came with
    `replace_with`: a second wrapper made before the call did not see the
    write. docs/VIEWS.md §6 removed it, and the assertion is inverted rather
    than deleted -- `detach` shares storage upstream too, so seeing the write
    is now the agreeing answer and a regression to wrapper-swapping fails
    here.
    """
    x = _C._tensor_from_flat([0.0] * 5, [5])
    returned = _C._aten_dispatch(
        "aten.index_put_.default",
        x,
        [_C._tensor_from_flat([0, 2, 4], [3], dtype=_C.int64)],
        _C._tensor_from_flat([7.0, 8.0, 9.0], [3]),
        False,
    )
    assert x.tolist() == [7.0, 0.0, 8.0, 0.0, 9.0], x.tolist()
    assert returned is x, "index_put_ returns self, as its schema says"

    # Through the subscript, where the return value does not exist at all.
    y = _C._tensor_from_flat([0.0] * 6, [3, 2])
    y[_C._tensor_from_flat([0, 2], [2], dtype=_C.int64)] = 5
    assert y.tolist() == [[5.0, 5.0], [0.0, 0.0], [5.0, 5.0]], y.tolist()

    # And through an alias, which is upstream's behaviour: `detach` shares
    # storage, so a write to either name is visible through both.
    z = _C._tensor_from_flat([0.0] * 4, [4])
    alias = _C._aten_dispatch("aten.detach.default", z)
    z[_C._tensor_from_flat([0], [1], dtype=_C.int64)] = 1.0
    assert z.tolist() == [1.0, 0.0, 0.0, 0.0], z.tolist()
    assert alias.tolist() == [1.0, 0.0, 0.0, 0.0], (
        "the alias did not see the write -- index_put_ has gone back to replacing "
        "the wrapper instead of writing through the layout (docs/VIEWS.md §6)"
    )
    # ...and the other way round, which is the direction that fails if only
    # the receiver's own wrapper is being written.
    _C._aten_dispatch("aten.fill_.Scalar", alias, 5.0)
    assert z.tolist() == [5.0] * 4, z.tolist()


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


# --- torch.randn / torch.rand and their siblings (docs/RANDOM.md) ----------
#
# `randn`/`rand` are not `overloads.json` entries: there is no `aten::randn`
# or `aten::rand` kernel in `aten.rs`, only `aten.empty.memory_format` +
# `aten.uniform_.default`/`aten.normal_.default`. Real torch's own C++ body
# for these factories is the same composition (measured: seeded
# `torch.randn(4, 4)` is bit-identical to seeded
# `torch.empty(4, 4).normal_(0., 1.)`, and the same holds for `rand`/
# `uniform_`), so these are Python-level compositions in `_install_composites`
# -- the same shape as `dropout`/`layer_norm`/`isfinite` above -- rather than
# a table entry that would name a kernel `aten.rs` does not have.


def test_randn_and_rand_are_wired_rather_than_refused():
    # The defect this section exists for: both used to raise
    # "torch._C shim: ... overload resolution has no table entry".
    x = _C._VariableFunctions.randn(4, 4)
    assert x.shape == (4, 4)
    assert x.dtype == _C.get_default_dtype()
    y = _C._VariableFunctions.rand(2, 2)
    assert y.shape == (2, 2)
    assert y.dtype == _C.get_default_dtype()
    assert all(0.0 <= v < 1.0 for row in y.tolist() for v in row)


def test_randn_accepts_a_size_tuple_or_list_as_well_as_varargs():
    # Upstream accepts both `torch.randn(4, 4)` and `torch.randn((4, 4))` --
    # and they have to draw identically, not just produce the same shape.
    _C._shim_manual_seed(0)
    a = _C._VariableFunctions.randn(4, 4)
    _C._shim_manual_seed(0)
    b = _C._VariableFunctions.randn((4, 4))
    assert a.tolist() == b.tolist()
    _C._shim_manual_seed(0)
    c = _C._VariableFunctions.randn([4, 4])
    assert a.tolist() == c.tolist()
    _C._shim_manual_seed(0)
    d = _C._VariableFunctions.randn(4)
    assert d.shape == (4,)


def test_rand_like_and_randn_like_take_shape_and_dtype_from_the_input():
    base = _C._VariableFunctions.empty([3, 2])
    r = _C._VariableFunctions.rand_like(base)
    assert r.shape == (3, 2)
    assert r.dtype == base.dtype
    n = _C._VariableFunctions.randn_like(base)
    assert n.shape == (3, 2)
    assert n.dtype == base.dtype


def test_randn_refuses_an_integer_dtype_by_naming_it():
    # The composite falls through to `normal_`, which already refuses integer
    # tensors by name (`test_rng_ops_refuse_integer_tensors` above). This
    # checks the composite does not swallow that refusal or convert it into a
    # silently-wrong result.
    try:
        _C._VariableFunctions.randn(2, 2, dtype=_C.int64)
    except NotImplementedError as e:
        assert "int64" in str(e)
    else:
        raise AssertionError("torch.randn(dtype=torch.int64) must be refused")


def test_randn_refuses_requires_grad_true_by_name():
    try:
        _C._VariableFunctions.randn(2, 2, requires_grad=True)
    except NotImplementedError as e:
        assert "requires_grad" in str(e)
    else:
        raise AssertionError("torch.randn(requires_grad=True) must be refused")


def test_randn_and_rand_refuse_out_by_name_rather_than_silently_ignoring_it():
    # Upstream resizes `out` to the requested shape --
    # `torch.randn(4, 4, out=torch.empty(2, 2))` returns a 4x4 tensor -- which
    # needs a resize kernel this shim does not have (no `aten::empty.out`, no
    # generic `resize_`). Refusing by name is the honest answer; computing
    # into the wrong shape, or silently ignoring `out` and returning a tensor
    # the caller's `out` variable does not point to, would both be worse.
    dest = _C._VariableFunctions.empty([2, 2])
    for call in (
        lambda: _C._VariableFunctions.randn(2, 2, out=dest),
        lambda: _C._VariableFunctions.rand(2, 2, out=dest),
    ):
        try:
            call()
        except NotImplementedError as e:
            assert "out" in str(e)
        else:
            raise AssertionError("torch.randn(out=...) must be refused")


def test_randn_generator_kwarg_reaches_the_same_stream_as_manual_seed():
    _C._shim_manual_seed(5)
    a = _C._VariableFunctions.randn(3)
    _C._shim_manual_seed(5)
    b = _C._VariableFunctions.randn(3, generator=_C.default_generator)
    assert a.tolist() == b.tolist()
    other = _C.Generator()
    try:
        _C._VariableFunctions.randn(3, generator=other)
    except NotImplementedError as e:
        assert "torch.default_generator" in str(e)
    else:
        raise AssertionError("a foreign generator was accepted")


def test_normal_size_overload_matches_manual_composition():
    _C._shim_manual_seed(0)
    a = _C._VariableFunctions.normal(0.0, 1.0, size=[2, 2])
    _C._shim_manual_seed(0)
    b = _C._VariableFunctions.empty([2, 2])
    b.normal_(0.0, 1.0)
    assert a.tolist() == b.tolist()


def test_normal_accepts_tensor_mean_and_or_std():
    mean_t = _t([1.0, 2.0, 3.0], [3])
    std_t = _t([1.0, 1.0, 1.0], [3])
    a = _C._VariableFunctions.normal(mean_t, 1.0)
    assert a.shape == (3,)
    b = _C._VariableFunctions.normal(0.5, std_t)
    assert b.shape == (3,)
    c = _C._VariableFunctions.normal(mean_t, std_t)
    assert c.shape == (3,)


def test_normal_requires_size_when_both_mean_and_std_are_plain_numbers():
    try:
        _C._VariableFunctions.normal(0.0, 1.0)
    except TypeError:
        pass
    else:
        raise AssertionError(
            "torch.normal(float, float) with no size must be refused"
        )


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


def test_randn_matches_upstreams_stream_bit_for_bit():
    # `torch.randn` composes `empty` + `normal_` in bootstrap.py; this checks
    # the composition draws the same number of words in the same order as
    # upstream's own `aten::randn.default` kernel does, not just that the
    # distribution looks right.
    if _upstream_torch is None:
        return  # no upstream torch in this interpreter -- see docs/E2E.md
    for seed in (0, 1, 1234):
        _upstream_torch.manual_seed(seed)
        want = _upstream_torch.randn(4, 4)
        _C._shim_manual_seed(seed)
        got = _C._VariableFunctions.randn(4, 4)
        assert got.tolist() == want.tolist(), seed
        # Two draws with no reseed in between, to catch a composition that
        # consumes a different word count than upstream's fused kernel would.
        want2 = _upstream_torch.randn(3)
        got2 = _C._VariableFunctions.randn(3)
        assert got2.tolist() == want2.tolist(), (seed, "second draw")


def test_rand_matches_upstreams_stream_bit_for_bit():
    if _upstream_torch is None:
        return
    for seed in (0, 1, 1234):
        _upstream_torch.manual_seed(seed)
        want = _upstream_torch.rand(2, 3)
        _C._shim_manual_seed(seed)
        got = _C._VariableFunctions.rand(2, 3)
        assert got.tolist() == want.tolist(), seed


def test_rand_like_and_randn_like_match_upstreams_stream_bit_for_bit():
    if _upstream_torch is None:
        return
    flat = [float(i) for i in range(6)]
    for seed in (0, 7):
        t_base = _upstream_torch.tensor(flat).reshape(2, 3)
        c_base = _C._tensor_from_flat(flat, [2, 3], dtype=_C.float32)
        _upstream_torch.manual_seed(seed)
        want_r = _upstream_torch.rand_like(t_base)
        _C._shim_manual_seed(seed)
        got_r = _C._VariableFunctions.rand_like(c_base)
        assert got_r.tolist() == want_r.tolist(), (seed, "rand_like")
        _upstream_torch.manual_seed(seed)
        want_n = _upstream_torch.randn_like(t_base)
        _C._shim_manual_seed(seed)
        got_n = _C._VariableFunctions.randn_like(c_base)
        assert got_n.tolist() == want_n.tolist(), (seed, "randn_like")


def test_normal_matches_upstreams_stream_bit_for_bit_across_overloads():
    if _upstream_torch is None:
        return
    for seed in (0, 7):
        _upstream_torch.manual_seed(seed)
        want = _upstream_torch.normal(0.0, 1.0, size=(2, 2))
        _C._shim_manual_seed(seed)
        got = _C._VariableFunctions.normal(0.0, 1.0, size=[2, 2])
        assert got.tolist() == want.tolist(), (seed, "float_float")

        mean_flat = [1.0, 2.0, 3.0, 4.0]
        t_mean = _upstream_torch.tensor(mean_flat).reshape(2, 2)
        c_mean = _C._tensor_from_flat(mean_flat, [2, 2], dtype=_C.float32)
        _upstream_torch.manual_seed(seed)
        want = _upstream_torch.normal(t_mean, 2.0)
        _C._shim_manual_seed(seed)
        got = _C._VariableFunctions.normal(c_mean, 2.0)
        assert got.tolist() == want.tolist(), (seed, "Tensor_float")

        _upstream_torch.manual_seed(seed)
        want = _upstream_torch.normal(0.5, t_mean)  # positive values -> a valid std
        _C._shim_manual_seed(seed)
        got = _C._VariableFunctions.normal(0.5, c_mean)
        assert got.tolist() == want.tolist(), (seed, "float_Tensor")

        std_flat = [1.0, 1.0, 1.0, 1.0]
        t_std = _upstream_torch.tensor(std_flat).reshape(2, 2)
        c_std = _C._tensor_from_flat(std_flat, [2, 2], dtype=_C.float32)
        _upstream_torch.manual_seed(seed)
        want = _upstream_torch.normal(t_mean, t_std)
        _C._shim_manual_seed(seed)
        got = _C._VariableFunctions.normal(c_mean, c_std)
        assert got.tolist() == want.tolist(), (seed, "Tensor_Tensor")

        # Broadcasting mean and std against each other, not just matching
        # shapes -- the standard-normal draw has to be sized at the
        # *broadcast* shape, not either operand's own shape (measured against
        # upstream 2.13.0; see docs/RANDOM.md).
        bflat_mean, bflat_std = [0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]
        t_bmean = _upstream_torch.tensor(bflat_mean).reshape(3, 1)
        t_bstd = _upstream_torch.tensor(bflat_std).reshape(1, 4)
        c_bmean = _C._tensor_from_flat(bflat_mean, [3, 1], dtype=_C.float32)
        c_bstd = _C._tensor_from_flat(bflat_std, [1, 4], dtype=_C.float32)
        _upstream_torch.manual_seed(seed)
        want = _upstream_torch.normal(t_bmean, t_bstd)
        _C._shim_manual_seed(seed)
        got = _C._VariableFunctions.normal(c_bmean, c_bstd)
        assert got.tolist() == want.tolist(), (seed, "Tensor_Tensor broadcast")


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
import struct
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

    `cos`/`sin`/`tanh`/`exp`/`log`/`expm1`/`rsqrt` joined it later and share
    the rule through `unary_float_tag`; they are covered by
    `test_meta_unary_promotions_are_the_dense_families_own` below.
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


# --- the elementwise meta family (docs/META.md §7.1) -------------------------
#
# Every test below checks **shape and dtype separately and explicitly**. That
# is not belt-and-braces: a meta kernel's entire output is those two fields, so
# a test that reads only `.shape` cannot fail on a dtype fault, and the dtype
# half is the half that is not guessable from the input -- `gt` answers `bool`
# whatever went in, `div` on two integers answers a float, `where` takes its
# dtype from the values and its shape from all three operands.
#
# The oracle is upstream torch on meta tensors. Every row was read off
# `torch.ops.aten.<op>(torch.zeros(..., device="meta"), ...)` on 2.13.0.


def _meta_empty(shape, dtype):
    return _C._aten_dispatch(
        "aten.empty.memory_format", shape, dtype, device=_C.device("meta")
    )


def test_meta_comparisons_answer_bool_whatever_went_in():
    """The family the user's `from_pretrained` report stopped on.

    `transformers/modeling_rope_utils.py:655` opens `_compute_llama3_parameters`
    with `torch.where(wavelen > low_freq_wavelen, ...)`, and `>` on a meta
    tensor is `aten.gt.Scalar`. `from_pretrained` initialises weights on the
    meta device, so the whole rope computation runs there; SmolLM2 only worked
    because it has no `rope_scaling` and its default rope init does no
    comparisons.

    **The dtype is the point.** All twelve keys answer `torch.bool` regardless
    of the operand dtype -- measured on 2.13.0 across
    `{float32, float16, bfloat16, float64, int64, int32, int16, uint8, bool}`
    and both integer and float scalars. A kernel that returned the input dtype
    would still give the right shape, and every shape-only assertion would
    still pass, which is why the dtype is asserted on its own line below.

    The `Tensor` overloads add the broadcast. `(1,3)` against `(2,1)` is
    `(2,3)` and `(2,3)` against a 0-dim is `(2,3)` -- the 0-dim row being the
    one a "shape is the left operand's" shortcut gets right by accident.
    """
    d = _C._aten_dispatch
    SCALAR = [
        "aten.eq.Scalar", "aten.ne.Scalar", "aten.lt.Scalar",
        "aten.le.Scalar", "aten.ge.Scalar", "aten.gt.Scalar",
    ]
    TENSOR = [op.replace(".Scalar", ".Tensor") for op in SCALAR]
    DTYPES = [_C.float32, _C.float16, _C.bfloat16, _C.float64,
              _C.int64, _C.int32, _C.int16, _C.uint8, _C.bool]

    for op in SCALAR:
        for dtype in DTYPES:
            for scalar in (1, 1.5, True):
                out = d(op, _meta_empty([2, 3], dtype), scalar)
                assert out.is_meta is True, (op, dtype, scalar)
                assert tuple(out.shape) == (2, 3), (op, dtype, scalar, tuple(out.shape))
                assert out.dtype == _C.bool, (op, dtype, scalar, out.dtype)
        # 0-dim in, 0-dim out -- and still bool.
        out = d(op, _meta_empty([], _C.float32), 1.0)
        assert tuple(out.shape) == () and out.dtype == _C.bool, op

    for op in TENSOR:
        for dtype in DTYPES:
            for lhs, rhs, want in (
                ([2, 3], [2, 3], (2, 3)),
                ([1, 3], [2, 1], (2, 3)),
                ([2, 3], [], (2, 3)),
                ([], [2, 3], (2, 3)),
                ([], [], ()),
                ([4, 1], [1, 5], (4, 5)),
                ([2, 3], [3], (2, 3)),
                ([0, 3], [1, 3], (0, 3)),
            ):
                out = d(op, _meta_empty(lhs, dtype), _meta_empty(rhs, dtype))
                assert tuple(out.shape) == want, (op, dtype, lhs, rhs, tuple(out.shape))
                assert out.dtype == _C.bool, (op, dtype, lhs, rhs, out.dtype)

    # A shape that cannot broadcast is still an error on meta -- upstream's
    # message, naming both extents and the axis.
    try:
        d("aten.gt.Tensor", _meta_empty([2, 3], _C.float32), _meta_empty([4, 5], _C.float32))
    except RuntimeError as e:
        assert "must match the size" in str(e), str(e)
    else:
        raise AssertionError("meta gt.Tensor broadcast a pair that does not broadcast")

    # And the dense kernel's dtype refusal is the meta kernel's. Upstream
    # promotes here; this shim does not, on either device, and the meta path
    # must not be the one that quietly starts (docs/E2E_REAL.md §6.1).
    try:
        d("aten.gt.Tensor", _meta_empty([2], _C.float32), _meta_empty([2], _C.int64))
    except NotImplementedError as e:
        assert "promotion" in str(e), str(e)
    else:
        raise AssertionError("meta gt.Tensor promoted where the dense kernel refuses")


def test_meta_elementwise_arithmetic_broadcasts_and_promotes_like_the_dense_kernel():
    """`add`/`sub`/`mul`/`div`, both overloads, on meta.

    Two rules, and neither is "the input's dtype":

      * **`div` floats an integral pair.** `int64 / int64` is `float32`,
        measured, because torch's `/` is true division. A kernel that passed
        the input dtype through would be right for the three other members and
        wrong only here.
      * **`mul.Tensor` promotes its operands; the other three require them
        equal.** That asymmetry is the dense kernel's (`promote_operands` vs
        `same_dtype`), and the meta arm calls the same two functions rather
        than restating either.

    Both are `arith_tag`'s, which is why the two paths cannot drift: the meta
    kernel calls it, so the dtype it advertises is by construction the dtype
    the dense kernel would produce.
    """
    d = _C._aten_dispatch
    SHAPES = [([2, 3], [2, 3], (2, 3)), ([1, 3], [2, 1], (2, 3)),
              ([2, 3], [], (2, 3)), ([], [2, 3], (2, 3)), ([], [], ()),
              ([4, 1], [1, 5], (4, 5)), ([2, 3], [3], (2, 3)),
              ([0, 3], [1, 3], (0, 3))]

    for op in ("aten.add.Tensor", "aten.sub.Tensor", "aten.mul.Tensor", "aten.div.Tensor"):
        floats = op == "aten.div.Tensor"
        for dtype, want in ((_C.float32, _C.float32), (_C.float16, _C.float16),
                            (_C.bfloat16, _C.bfloat16), (_C.float64, _C.float64),
                            (_C.int64, _C.float32 if floats else _C.int64),
                            (_C.int32, _C.float32 if floats else _C.int32),
                            (_C.uint8, _C.float32 if floats else _C.uint8)):
            for lhs, rhs, shape in SHAPES:
                out = d(op, _meta_empty(lhs, dtype), _meta_empty(rhs, dtype))
                assert out.is_meta is True, (op, dtype)
                assert tuple(out.shape) == shape, (op, dtype, lhs, rhs, tuple(out.shape))
                assert out.dtype == want, (op, dtype, lhs, rhs, out.dtype)

    # `mul.Tensor` is the one member that promotes a mixed pair, and the three
    # others refuse it. Both halves, so that a meta kernel which promoted
    # everything and one which refused everything both fail.
    out = d("aten.mul.Tensor", _meta_empty([2], _C.float32), _meta_empty([2], _C.int64))
    assert out.dtype == _C.float32, out.dtype
    out = d("aten.mul.Tensor", _meta_empty([2], _C.float16), _meta_empty([2], _C.bfloat16))
    assert out.dtype == _C.float32, ("float16 x bfloat16 escapes upwards", out.dtype)
    for op in ("aten.add.Tensor", "aten.sub.Tensor", "aten.div.Tensor"):
        try:
            d(op, _meta_empty([2], _C.float32), _meta_empty([2], _C.int64))
        except NotImplementedError as e:
            assert "promotion" in str(e), (op, str(e))
        else:
            raise AssertionError(f"meta {op} promoted where the dense kernel refuses")

    # The `Scalar` overloads, including the two that were absent until the
    # `Tensor` ones landed. `int64 * 2` stays `int64`; `int64 * 2.0` floats.
    for op in ("aten.add.Scalar", "aten.sub.Scalar", "aten.mul.Scalar",
               "aten.div.Scalar", "aten.rsub.Scalar"):
        floats = op == "aten.div.Scalar"
        for dtype, scalar, want in (
            (_C.float32, 2, _C.float32), (_C.float32, 2.0, _C.float32),
            (_C.float16, 2, _C.float16), (_C.float64, 2, _C.float64),
            (_C.int64, 2, _C.float32 if floats else _C.int64),
            (_C.int64, 2.0, _C.float32),
            (_C.int32, 2, _C.float32 if floats else _C.int32),
        ):
            out = d(op, _meta_empty([2, 3], dtype), scalar)
            assert tuple(out.shape) == (2, 3), (op, dtype, scalar, tuple(out.shape))
            assert out.dtype == want, (op, dtype, scalar, out.dtype)

    # The promotion reads `set_default_dtype`, as the dense one does. If a meta
    # kernel had hardcoded float32 this is the line that says so.
    try:
        _C._set_default_dtype(_C.float64)
        assert d("aten.div.Tensor", _meta_empty([2], _C.int64),
                 _meta_empty([2], _C.int64)).dtype == _C.float64
        assert d("aten.mul.Tensor", _meta_empty([2], _C.int64),
                 _meta_empty([2], _C.int64)).dtype == _C.int64
    finally:
        _C._set_default_dtype(_C.float32)


def test_meta_where_broadcasts_three_operands_and_takes_dtype_from_the_values():
    """`where.self` is the one op here with three operands, and both of its
    rules are the ones that are easy to get wrong.

    **Shape is the join of all three, not the condition's.** Measured on
    2.13.0: `where(meta_bool(2,1), meta_f32(1,3), meta_f32(3))` is `(2,3)`
    where a condition-shaped answer would be `(2,1)`, and
    `where(meta_bool(), f32(2,3), f32(3))` is `(2,3)` where it would be `()`.

    **Dtype comes from the two value operands, not the condition.** The
    condition is `bool` and the answer is not.

    `where.ScalarOther` is the same join with the third operand a Python
    scalar, so it broadcasts two and takes `where_scalar_tag`'s wrapped-number
    rule: a `bool` scalar leaves the tensor's dtype alone, an `int` one lifts
    only a boolean tensor, a `float` one floats an integral tensor.
    """
    d = _C._aten_dispatch

    for dtype in (_C.float32, _C.float16, _C.bfloat16, _C.float64,
                  _C.int64, _C.int32, _C.uint8, _C.bool):
        out = d("aten.where.self", _meta_empty([2, 3], _C.bool),
                _meta_empty([2, 3], dtype), _meta_empty([2, 3], dtype))
        assert out.is_meta is True, dtype
        assert tuple(out.shape) == (2, 3), (dtype, tuple(out.shape))
        assert out.dtype == dtype, ("dtype came from the condition", dtype, out.dtype)

    for cond, lhs, rhs, want in (
        ([2, 1], [1, 3], [3], (2, 3)),
        ([], [2, 3], [3], (2, 3)),
        ([2, 3], [], [], (2, 3)),
        ([1, 1, 3], [2, 1], [4, 1, 1], (4, 2, 3)),
        ([2, 3], [2, 3], [2, 3], (2, 3)),
    ):
        out = d("aten.where.self", _meta_empty(cond, _C.bool),
                _meta_empty(lhs, _C.float32), _meta_empty(rhs, _C.float32))
        assert tuple(out.shape) == want, (cond, lhs, rhs, tuple(out.shape))
        assert out.dtype == _C.float32, (cond, lhs, rhs, out.dtype)

    # The condition's dtype is one of the few dense checks a meta tensor can
    # still run in full, since it is carried and never read for values.
    try:
        d("aten.where.self", _meta_empty([2, 3], _C.float32),
          _meta_empty([2, 3], _C.float32), _meta_empty([2, 3], _C.float32))
    except RuntimeError as e:
        assert "boolean tensor" in str(e), str(e)
    else:
        raise AssertionError("meta where.self accepted a float condition")

    # And the value operands must agree, as they must on the dense path.
    try:
        d("aten.where.self", _meta_empty([2], _C.bool),
          _meta_empty([2], _C.float32), _meta_empty([2], _C.int64))
    except NotImplementedError as e:
        assert "promotion" in str(e), str(e)
    else:
        raise AssertionError("meta where.self promoted where the dense kernel refuses")

    for dtype, scalar, want in (
        (_C.float32, 0.0, _C.float32), (_C.float32, 0, _C.float32),
        (_C.int64, 0, _C.int64), (_C.int64, 0.0, _C.float32),
        (_C.bool, 0, _C.int64), (_C.bool, True, _C.bool),
        (_C.float16, 0.0, _C.float16),
    ):
        out = d("aten.where.ScalarOther", _meta_empty([2, 3], _C.bool),
                _meta_empty([2, 3], dtype), scalar)
        assert tuple(out.shape) == (2, 3), (dtype, scalar, tuple(out.shape))
        assert out.dtype == want, (dtype, scalar, out.dtype)
    for cond, lhs, want in (([2, 1], [1, 3], (2, 3)), ([], [2, 3], (2, 3)),
                            ([2, 3], [], (2, 3))):
        out = d("aten.where.ScalarOther", _meta_empty(cond, _C.bool),
                _meta_empty(lhs, _C.float32), 0.0)
        assert tuple(out.shape) == want, (cond, lhs, tuple(out.shape))


def test_meta_unary_promotions_are_the_dense_families_own():
    """Three different unary rules, and the test exists because they differ.

      * **`unary_float`** (`cos`, `sin`, `tanh`, `exp`, `log`, `expm1`,
        `rsqrt`, `reciprocal`): floating in, the *same* floating dtype out --
        `float16` stays `float16` and is not widened -- and anything else
        becomes the default float.
      * **`neg`**: keeps the input dtype, so `int64` in is `int64` out. It is
        the counter-example to the family above, and it carries two refusals
        (`bool`, and the wide unsigned dtypes) that the meta path reproduces.
      * **`bitwise_not`**: keeps the input dtype and refuses floats.

    A single "shape and dtype pass through" implementation would satisfy `neg`
    and `bitwise_not` and be wrong for every integral input to the first group;
    a single "promote to float" one would be wrong the other way. Both
    directions are asserted.
    """
    d = _C._aten_dispatch
    PROMOTING = ("aten.cos.default", "aten.sin.default", "aten.tanh.default",
                 "aten.exp.default", "aten.log.default", "aten.expm1.default",
                 "aten.rsqrt.default", "aten.reciprocal.default")

    for op in PROMOTING:
        for dtype, want in ((_C.float32, _C.float32), (_C.float16, _C.float16),
                            (_C.bfloat16, _C.bfloat16), (_C.float64, _C.float64),
                            (_C.int64, _C.float32), (_C.int32, _C.float32),
                            (_C.int16, _C.float32), (_C.uint8, _C.float32),
                            (_C.bool, _C.float32)):
            for shape in ([2, 3], [], [0, 4], [2, 3, 4]):
                out = d(op, _meta_empty(shape, dtype))
                assert out.is_meta is True, (op, dtype)
                assert tuple(out.shape) == tuple(shape), (op, dtype, shape)
                assert out.dtype == want, (op, dtype, out.dtype)

    for dtype in (_C.float32, _C.float16, _C.bfloat16, _C.float64,
                  _C.int64, _C.int32, _C.int16, _C.uint8):
        for shape in ([2, 3], [], [2, 3, 4]):
            out = d("aten.neg.default", _meta_empty(shape, dtype))
            assert tuple(out.shape) == tuple(shape), (dtype, shape)
            assert out.dtype == dtype, ("neg promoted", dtype, out.dtype)
    try:
        d("aten.neg.default", _meta_empty([2], _C.bool))
    except RuntimeError as e:
        assert "bool tensor is not supported" in str(e), str(e)
    else:
        raise AssertionError("meta neg accepted a bool the dense path refuses")

    for dtype in (_C.int64, _C.int32, _C.int16, _C.uint8, _C.bool):
        out = d("aten.bitwise_not.default", _meta_empty([2, 3], dtype))
        assert tuple(out.shape) == (2, 3) and out.dtype == dtype, (dtype, out.dtype)
    try:
        d("aten.bitwise_not.default", _meta_empty([2], _C.float32))
    except RuntimeError as e:
        assert "bitwise_not_cpu" in str(e), str(e)
    else:
        raise AssertionError("meta bitwise_not accepted a float")

    # The promoting group reads the default dtype and `neg` does not.
    try:
        _C._set_default_dtype(_C.float64)
        assert d("aten.exp.default", _meta_empty([2], _C.int64)).dtype == _C.float64
        assert d("aten.exp.default", _meta_empty([2], _C.float32)).dtype == _C.float32
        assert d("aten.neg.default", _meta_empty([2], _C.int64)).dtype == _C.int64
    finally:
        _C._set_default_dtype(_C.float32)


def test_meta_clamp_and_pow_share_the_dense_kernels_dtype_ladders():
    """The two ops here whose dtype rule is a ladder rather than a one-liner.

    `clamp` out of place **promotes** where `clamp_` refuses, which is the row
    the golden cases had to correct once. Measured on meta, 2.13.0:

        clamp(int32,  0,     5)      int32
        clamp(int32,  None,  2.0)    float32
        clamp(uint8,  None,  2)      uint8
        clamp(float16,None,  2.0)    float16    a float never widens a float tensor
        clamp(bool,   0,     5)      int64
        clamp(bool,   0.0,   1.0)    float32
        clamp(bool,   False, True)   refused    a bool scalar does not lift a bool tensor

    `pow.Tensor_Scalar` is the wrapped-number rule: an integral tensor with an
    integer exponent stays integral, a float on either side floats it.
    `pow.Tensor_Tensor` promotes its operands first, so it hands the promotion
    -- not an operand's dtype -- to the same function.

    Both meta arms call the dense kernels' own `clamp_result_tag` and
    `pow_result_tag`, so a change to either rule moves both paths together.
    """
    d = _C._aten_dispatch

    for dtype, mn, mx, want in (
        (_C.int32, 0, 5, _C.int32),
        (_C.int32, None, 2.0, _C.float32),
        (_C.uint8, None, 2, _C.uint8),
        (_C.uint8, None, 2.0, _C.float32),
        (_C.float16, None, 2.0, _C.float16),
        (_C.float32, 0.0, None, _C.float32),
        (_C.bool, 0, 5, _C.int64),
        (_C.bool, 0.0, 1.0, _C.float32),
        (_C.int64, 0, 5, _C.int64),
    ):
        out = d("aten.clamp.default", _meta_empty([2, 3], dtype), mn, mx)
        assert out.is_meta is True, (dtype, mn, mx)
        assert tuple(out.shape) == (2, 3), (dtype, mn, mx, tuple(out.shape))
        assert out.dtype == want, (dtype, mn, mx, out.dtype)
    try:
        d("aten.clamp.default", _meta_empty([2], _C.bool), False, True)
    except NotImplementedError as e:
        assert "clamp_scalar_cpu" in str(e), str(e)
    else:
        raise AssertionError("meta clamp accepted bool bounds on a bool tensor")
    try:
        d("aten.clamp.default", _meta_empty([2], _C.float32), None, None)
    except RuntimeError as e:
        assert "must not be None" in str(e), str(e)
    else:
        raise AssertionError("meta clamp with no bounds was a no-op")

    for dtype, exponent, want in (
        (_C.float32, 2, _C.float32), (_C.float32, 2.0, _C.float32),
        (_C.float16, 2, _C.float16), (_C.float64, 2, _C.float64),
        (_C.int64, 2, _C.int64), (_C.int64, 2.0, _C.float32),
        (_C.int32, 2, _C.int32), (_C.int32, 2.0, _C.float32),
    ):
        out = d("aten.pow.Tensor_Scalar", _meta_empty([2, 3], dtype), exponent)
        assert tuple(out.shape) == (2, 3), (dtype, exponent, tuple(out.shape))
        assert out.dtype == want, (dtype, exponent, out.dtype)

    for lhs, rhs, want in (
        (_C.float32, _C.float32, _C.float32),
        (_C.float32, _C.int32, _C.float32),
        (_C.int64, _C.int32, _C.int64),
        (_C.float16, _C.bfloat16, _C.float32),
        (_C.float16, _C.float16, _C.float16),
    ):
        out = d("aten.pow.Tensor_Tensor", _meta_empty([2, 3], lhs), _meta_empty([2, 3], rhs))
        assert tuple(out.shape) == (2, 3), (lhs, rhs, tuple(out.shape))
        assert out.dtype == want, (lhs, rhs, out.dtype)
    out = d("aten.pow.Tensor_Tensor", _meta_empty([1, 3], _C.float32),
            _meta_empty([2, 1], _C.float32))
    assert tuple(out.shape) == (2, 3), tuple(out.shape)


def test_meta_shape_kernels_drop_expand_and_keep_the_triangle():
    """The three shape kernels the twenty-architecture meta sweep asked for.

    Each was the *next* wall after the one before it, re-measured rather than
    planned (ARCH20.md §0.2's "a wall is not one wall"): `select.int` for five
    architectures, then `tril.default` for `gpt_bigcode`, then
    `expand.default` for `bert`.

      * `select.int` **removes** the dimension, which is what separates it from
        `slice`. `(2,3,4)` selected on dim 1 is `(2,4)`.
      * `tril`/`triu` change *which values are zero* and nothing else, so on
        meta the whole kernel is the rank refusal.
      * `expand` prepends new dimensions, resolves `-1` against the existing
        extent, and refuses to stretch a non-singleton one -- including a
        **zero** extent, which is not singleton for this rule.

    All rows measured against upstream on meta tensors, refusals included.
    """
    d = _C._aten_dispatch

    for dtype in (_C.float32, _C.int64, _C.bool):
        for shape, dim, index, want in (
            ([2, 3], 0, 0, (3,)), ([2, 3], 1, 2, (2,)), ([2, 3], -1, -1, (2,)),
            ([2, 3, 4], 1, 0, (2, 4)), ([5], 0, 4, ()), ([2, 1, 3], 1, 0, (2, 3)),
        ):
            out = d("aten.select.int", _meta_empty(shape, dtype), dim, index)
            assert out.is_meta is True, (shape, dim, index)
            assert tuple(out.shape) == want, (shape, dim, index, tuple(out.shape))
            assert out.dtype == dtype, (shape, dim, index, out.dtype)
    for shape, dim, index, exc in (
        ([2, 3], 0, 2, IndexError), ([2, 3], 0, -3, IndexError),
        ([2, 3], 2, 0, IndexError), ([], 0, 0, IndexError),
        ([0, 3], 0, 0, IndexError),
    ):
        try:
            d("aten.select.int", _meta_empty(shape, _C.float32), dim, index)
        except exc:
            pass
        else:
            raise AssertionError(f"meta select.int accepted {shape} dim={dim} index={index}")

    for op in ("aten.tril.default", "aten.triu.default"):
        for dtype in (_C.float32, _C.int64, _C.bool):
            for shape in ([2, 3], [3, 3], [2, 3, 4]):
                for diagonal in (0, 1, -1, 100, -100):
                    out = d(op, _meta_empty(shape, dtype), diagonal)
                    assert tuple(out.shape) == tuple(shape), (op, shape, diagonal)
                    assert out.dtype == dtype, (op, shape, diagonal, out.dtype)
        for shape in ([5], []):
            try:
                d(op, _meta_empty(shape, _C.float32), 0)
            except RuntimeError as e:
                assert "at least 2 dimensions" in str(e), str(e)
            else:
                raise AssertionError(f"meta {op} accepted rank {len(shape)}")

    for dtype in (_C.float32, _C.int64):
        for base, size, want in (
            ([2, 1, 3], [2, 5, 3], (2, 5, 3)),
            ([2, 1, 3], [4, 2, 5, 3], (4, 2, 5, 3)),
            ([2, 1, 3], [2, 4, -1], (2, 4, 3)),
            ([2, 1, 3], [-1, 4, 3], (2, 4, 3)),
            ([2, 1, 3], [-1, -1, -1], (2, 1, 3)),
            ([1], [3], (3,)),
            ([], [2, 3], (2, 3)),
            ([2, 3], [2, 3], (2, 3)),
            ([2, 1], [-1, 3], (2, 3)),
            # A singleton expands to **zero**, and that is the asymmetric
            # half of the zero-extent rule: `(1,3) -> (0,3)` is allowed and
            # `(0,3) -> (2,3)` is not (below). Measured on 2.13.0.
            ([1, 3], [0, 3], (0, 3)),
        ):
            out = d("aten.expand.default", _meta_empty(base, dtype), size)
            assert tuple(out.shape) == want, (base, size, tuple(out.shape))
            assert out.dtype == dtype, (base, size, out.dtype)
    for base, size, needle in (
        ([3], [2, 4], "must match the existing size"),
        ([0, 3], [2, 3], "must match the existing size"),
        ([2, 1, 3], [2, 4], "must be greater or equal"),
        # The `-1` here is *not* in a leading position (index 3, offset 1), so
        # it resolves to the existing extent 3 and the refusal comes from
        # dimension 1 instead. Upstream reports the requested sizes with the
        # sentinel still in them, which is what this asserts.
        ([2, 1, 3], [2, 4, 3, -1], "Target sizes: [2, 4, 3, -1]"),
        # A **zero** extent is not singleton for this rule, which is the case
        # a `have <= 1` written for `have != 1` would silently accept.
        ([0, 3], [2, 3], "must match the existing size"),
        # A leading `-1` has no existing extent to resolve against.
        ([2, 3], [-1, 2, 3], "leading, non-existing dimension"),
    ):
        try:
            d("aten.expand.default", _meta_empty(base, _C.float32), size)
        except RuntimeError as e:
            assert needle in str(e), (base, size, str(e))
        else:
            raise AssertionError(f"meta expand accepted {base} -> {size}")


def test_the_llama3_rope_init_runs_on_meta_end_to_end():
    """The user's report, reduced to the expression that failed.

    `transformers/modeling_rope_utils.py::_compute_llama3_parameters`, run
    op by op on meta tensors. This is the whole of what `from_pretrained`
    could not do: the rope init succeeded on `cpu` and failed on `meta`, and
    `from_pretrained` initialises weights on the meta device.

    Asserted **against the same expression run on cpu**, which is the oracle
    that does not need a network: the meta answer must have the shape and dtype
    the dense answer has. A meta kernel that returned the right shape and the
    wrong dtype passes a shape-only test and fails this one.
    """
    d = _C._aten_dispatch

    def rope(device):
        kw = {"device": device} if device is not None else {}
        # inv_freq = 1.0 / (base ** (arange(0, dim, 2, float32) / dim))
        inv_freq = d("aten.arange.start_step", 0, 16, 2, _C.float32, **kw)
        inv_freq = d("aten.div.Scalar", inv_freq, 16)
        inv_freq = d("aten.pow.Scalar", 10000.0, inv_freq)
        inv_freq = d("aten.mul.Scalar", d("aten.reciprocal.default", inv_freq), 1.0)
        # wavelen = 2 * pi / inv_freq
        wavelen = d("aten.mul.Scalar", d("aten.reciprocal.default", inv_freq), 6.283185307179586)
        # inv_freq_llama = where(wavelen > low, inv_freq / factor, inv_freq)
        high = d("aten.gt.Scalar", wavelen, 32.0)
        llama = d("aten.where.self", high, d("aten.div.Scalar", inv_freq, 8.0), inv_freq)
        # smooth = (old / wavelen - low_factor) / (high_factor - low_factor)
        smooth = d("aten.mul.Scalar", d("aten.reciprocal.default", wavelen), 32.0)
        smooth = d("aten.sub.Scalar", smooth, 1.0)
        smooth = d("aten.div.Scalar", smooth, 3.0)
        # smoothed = (1 - smooth) * llama / factor + smooth * llama
        smoothed = d("aten.add.Tensor",
                     d("aten.div.Scalar",
                       d("aten.mul.Tensor", d("aten.rsub.Scalar", smooth, 1.0), llama), 8.0),
                     d("aten.mul.Tensor", smooth, llama))
        # is_medium = ~(wavelen < high) * ~(wavelen > low)
        medium = d("aten.mul.Tensor",
                   d("aten.bitwise_not.default", d("aten.lt.Scalar", wavelen, 8.0)),
                   d("aten.bitwise_not.default", high))
        return d("aten.where.self", medium, smoothed, llama)

    on_meta = rope(_C.device("meta"))
    on_cpu = rope(None)
    assert on_meta.is_meta is True and on_cpu.is_meta is False
    assert tuple(on_meta.shape) == tuple(on_cpu.shape), (
        tuple(on_meta.shape), tuple(on_cpu.shape))
    assert on_meta.dtype == on_cpu.dtype, (on_meta.dtype, on_cpu.dtype)
    assert tuple(on_meta.shape) == (8,), tuple(on_meta.shape)
    assert on_meta.dtype == _C.float32, on_meta.dtype

    # The intermediate the report named. `wavelen > low_freq_wavelen` is
    # `aten.gt.Scalar`, and it is `bool` on both devices -- the dtype that,
    # had the meta kernel guessed "the input's", would have made `where`
    # refuse one layer later with a message naming the wrong op.
    assert d("aten.gt.Scalar", _meta_empty([8], _C.float32), 32.0).dtype == _C.bool


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

    # `add.Tensor` used to head this list and is now implemented
    # (docs/META.md §7.1). It was replaced rather than the test deleted: the
    # boundary moved, it did not disappear, and the reductions, the
    # contractions and the remaining views are still behind it.
    for op, args in (
        ("aten.mm.default", (a, a)),
        ("aten.bmm.default", (a, a)),
        ("aten.view.default", (a, [3, 2])),
        ("aten.reshape.default", (a, [6])),
        ("aten.slice.Tensor", (a, 0, 0, 1)),
        ("aten.sum.default", (a,)),
        ("aten.mean.dim", (a, [1])),
        ("aten.cat.default", ([a, b], 0)),
        ("aten.t.default", (a,)),
        ("aten.permute.default", (a, [1, 0])),
    ):
        try:
            d(op, *args)
        except NotImplementedError as e:
            assert op in str(e), (op, str(e))
            assert "no meta kernel" in str(e), (op, str(e))
        else:
            raise AssertionError(f"{op} answered on meta without a meta kernel")

    # And the other direction, so that this test cannot pass by the meta table
    # being *empty*: the ops §7.1 did implement must not name themselves.
    cond = d("aten.empty.memory_format", [2, 3], _C.bool, device=meta)
    for op, args in (
        ("aten.gt.Scalar", (a, 1.0)),
        ("aten.where.self", (cond, a, b)),
        ("aten.div.Tensor", (a, b)),
        ("aten.select.int", (a, 0, 0)),
        ("aten.expand.default", (a, [2, 3])),
        ("aten.tril.default", (a, 0)),
    ):
        out = d(op, *args)
        assert out.is_meta is True, op

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
    ask "which representation is this tensor?". Five of them answer `False`
    for every tensor this build can make, and CLAUDE.md §5.5 is right to be
    suspicious of that: a predicate that cannot say anything else is not a
    predicate.

    What makes it a fact rather than a constant is *this* test. Each of those
    five representations has exactly one way into existence, and every one of
    those ways refuses by name. If any of them ever lands, this test fails,
    and the predicate that quietly said `False` becomes a lie that something
    noticed. That is the difference between an invariant and an assumption --
    the `is_mutable` accident in docs/DISTRIBUTED.md §8.1 is what an unguarded
    one looks like.

    **The sixth is no longer in that group.** `Repr::Quantized` landed, so
    `is_quantized` has a constructor behind it and answers `True` for a
    block-quantised weight -- `test_is_quantized_is_no_longer_a_constant`.
    `torch.quantize_per_tensor` is still checked here and still refuses,
    because that is a different representation: upstream's per-tensor-affine
    `int8` with a scale and a zero point, not a GGML block format. The way in
    is `torch._C._quantize`, and it does not wear an aten name (quant.rs).

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
      That mechanism has since fired for real: the `Repr::Quantized` arm made
      the compiler ask all six, and `is_quantized` answered differently
      (tensor.rs). The tensor in *this* fixture is dense, so the five `False`
      answers below are still the right ones for it.

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


# --- grouped-query attention and greedy generate (docs/GENERATE.md) ---------
#
# docs/CKPT2.md §7.1 and §8 left three kernels between this shim and a real
# pretrained model actually producing text. All three are exercised here, on
# a model `transformers` builds rather than one this file transcribes:
#
#   1. `sdpa(enable_gqa=True)`     the SDPA path, when the KV heads are fewer
#                                  than the query heads
#   2. `aten.where.ScalarOther`    the EAGER path's mask, masking_utils.py:603
#   3. `aten.mul.Tensor(int64, bool)`   `generate()`'s attention-mask inference
#
# The config below differs from `_LLAMA_CFG` in one field --
# `num_key_value_heads=2` against `num_attention_heads=4` -- and that one
# field is what routes the forward through (1). `_LLAMA_CFG` has them equal,
# which is why every Llama comparison before this round missed the whole
# grouped path (docs/CKPT2.md §7.1 says so explicitly).
#
# **Tokens are checked AND logits are checked.** docs/ARCH.md §5.1 is the
# reason: a wrong `gelu` approximation once produced identical greedy tokens
# with logits 379x further apart than the correct kernel's. A token-only
# assertion would have passed it. The bound is the same `_REAL_LLAMA_ATOL`
# the ungrouped forward uses, and it was checked to be a live bound rather
# than inherited on faith -- see the docstring on the GQA test.

_GQA_CFG = dict(
    vocab_size=32, hidden_size=16, intermediate_size=32, num_hidden_layers=2,
    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=32,
    tie_word_embeddings=False,
)
_GQA_IDS = [3, 7, 1, 19]
_GQA_NEW_TOKENS = 8


@functools.lru_cache(maxsize=4)
def _upstream_gqa(attn):
    """The expected side: upstream torch, in this interpreter."""
    torch = _upstream_torch
    from transformers import AutoModelForCausalLM
    from transformers.models.llama.configuration_llama import LlamaConfig

    ns = {}
    exec(_LLAMA_FILL, ns)
    model = AutoModelForCausalLM.from_config(
        LlamaConfig(**_GQA_CFG), attn_implementation=attn
    )
    model.eval()
    ns["_fill"](model, torch)
    ids = torch.tensor([_GQA_IDS])
    with torch.no_grad():
        logits = model(ids).logits
        generated = model.generate(
            ids, max_new_tokens=_GQA_NEW_TOKENS, do_sample=False,
            use_cache=False, pad_token_id=0,
        )
    return {
        "shape": [int(d) for d in logits.shape],
        "logits": _e2e_flatten(logits.tolist()),
        "argmax": [int(x) for x in logits[0].argmax(-1).tolist()],
        "generated": [int(x) for x in generated[0].tolist()],
    }


_GQA_ROAD_SCRIPT = r"""
import json, sys, traceback
import torch

out = {}
FILL = __FILL__
CFG = __CFG__
IDS = __IDS__
NEW = __NEW__
ATTN = sys.argv[1]
try:
    from transformers import AutoModelForCausalLM
    from transformers.models.llama.configuration_llama import LlamaConfig
    ns = {}
    exec(FILL, ns)
    model = AutoModelForCausalLM.from_config(
        LlamaConfig(**CFG), attn_implementation=ATTN
    )
    model.eval()
    ns["_fill"](model, torch)
    ids = torch.tensor([IDS])
    with torch.no_grad():
        logits = model(ids).logits
except BaseException:
    out["forward"] = "FAILED: " + traceback.format_exc(limit=4)
else:
    out["forward"] = "OK"
    out["shape"] = [int(d) for d in logits.shape]
    def flat(v):
        if isinstance(v, list):
            r = []
            for e in v:
                r.extend(flat(e))
            return r
        return [v]
    out["logits"] = flat(logits.tolist())
    out["argmax"] = [int(x) for x in logits[0].argmax(-1).tolist()]

# Deliberately a SECOND try block: `generate` is behind a different set of
# kernels than the forward (docs/CKPT2.md §8 item 2), and one failing must
# not hide the other's result. docs/CKPT2.md §3 used the same shape of probe
# for the four checkpoint paths and it is why that round could report which
# wall each path stopped at.
try:
    generated = model.generate(
        torch.tensor([IDS]), max_new_tokens=NEW, do_sample=False,
        use_cache=False, pad_token_id=0,
    )
except BaseException:
    out["generate"] = "FAILED: " + traceback.format_exc(limit=4)
else:
    out["generate"] = "OK"
    out["generated"] = [int(x) for x in generated[0].tolist()]
json.dump(out, sys.stdout)
""".replace("__FILL__", repr(_LLAMA_FILL)).replace(
    "__CFG__", repr(_GQA_CFG)).replace("__IDS__", repr(_GQA_IDS)).replace(
    "__NEW__", repr(_GQA_NEW_TOKENS))


@functools.lru_cache(maxsize=4)
def _gqa_road_fixture(attn):
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _GQA_ROAD_SCRIPT, attn],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"gqa-road subprocess ({attn}) exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def test_grouped_query_attention_forward_matches_upstream_on_both_paths():
    """9 query heads to 3 KV heads is what a real pretrained model looks like.

    `num_key_value_heads < num_attention_heads` is the ordinary case in the
    pretrained checkpoints this project targets -- SmolLM2-135M is 9 and 3 --
    and it is the case every Llama comparison in this file missed, because
    `_LLAMA_CFG` sets the two equal.

    Both attention implementations are checked because they fail
    *differently*, and each one hides the other's wall:

        sdpa    routes into `F.scaled_dot_product_attention(enable_gqa=True)`,
                which hands the UNREPEATED key and value to the aten flash op
                and expects the op to broadcast the head dimension. Measured
                with a `TorchDispatchMode`: one op, key still (B, 3, S, K).
        eager   never touches sdpa and instead builds an additive mask with
                `torch.where(mask, tensor(0.0), finfo.min)` --
                `aten.where.ScalarOther`.

    The two must also agree with EACH OTHER on this shim, which is a check
    upstream passes for free and a shim need not: they share no kernel on the
    attention path, so agreeing to within float32 noise is evidence that
    neither one is quietly computing a different attention.

    **The logit bound is live here, not inherited.** `_REAL_LLAMA_ATOL` was
    measured for the ungrouped model in
    `test_a_real_transformers_llama_forward_matches_upstream`. It is reused
    because this model has the same width and vocabulary, and it was checked
    to still bite: repeating the KV heads by tiling (`repeat`) instead of
    `repeat_interleave` -- the plausible wrong spelling, which gives query
    head `i` the KV head `i % n` instead of `i // n` -- moves these logits by
    far more than 5e-7. That measurement is in docs/GENERATE.md; the golden
    harness pins the same distinction at the kernel.
    """
    if not _ckpt_shim_available() or _upstream_transformers is None:
        return
    seen = {}
    for attn in ("sdpa", "eager"):
        got = _gqa_road_fixture(attn)
        assert got["forward"] == "OK", (attn, got["forward"])
        want = _upstream_gqa(attn)
        assert got["shape"] == want["shape"], (attn, got["shape"], want["shape"])
        assert got["argmax"] == want["argmax"], (attn, got["argmax"], want["argmax"])
        worst = max(abs(a - b) for a, b in zip(got["logits"], want["logits"]))
        assert worst < _REAL_LLAMA_ATOL, (attn, worst)
        seen[attn] = got["logits"]

    # The two implementations of the same attention, on this shim, against
    # each other. Upstream's own two agree to 2e-7 on this model (measured),
    # so the bound is that plus room for the shim's own reassociation.
    cross = max(abs(a - b) for a, b in zip(seen["sdpa"], seen["eager"]))
    assert cross < 1e-5, cross


def test_greedy_generate_matches_upstream_token_for_token():
    """`generate()`, which is the thing docs/CKPT2.md §8 item 2 left open.

    `do_sample=False` on purpose: greedy decoding has no RNG in it, so a
    token mismatch is a kernel disagreement and nothing else. With sampling
    the two sides would have to share a random stream to be comparable at
    all, and docs/SAMPLING.md already covers that surface separately.

    The wall this opens is neither of the attention ones. Before reaching any
    layer, `_prepare_attention_mask_for_generation` computes

        attention_mask_from_padding * can_infer_attention_mask
            + default_attention_mask * ~can_infer_attention_mask

    whose multiplies are `int64` by a 0-D `bool` -- `aten.mul.Tensor` with two
    different dtypes, which this shim refused for every op until now.

    **Tokens alone would not be enough and are not what is asserted.** The
    forward test above pins the logits; this one pins the sequence those
    logits decode to, on both attention implementations. docs/ARCH.md §5.1 is
    the case that makes the distinction non-theoretical.
    """
    if not _ckpt_shim_available() or _upstream_transformers is None:
        return
    for attn in _GENERATE_PATHS:
        got = _gqa_road_fixture(attn)
        assert got["generate"] == "OK", (attn, got["generate"])
        want = _upstream_gqa(attn)
        assert got["generated"] == want["generated"], (
            attn, got["generated"], want["generated"]
        )
        # The prompt has to still be there, and something has to follow it:
        # `generate` returns prompt + continuation, and a shim that produced
        # nothing at all would still satisfy the equality above if the
        # expected side also produced nothing.
        assert got["generated"][:len(_GQA_IDS)] == _GQA_IDS, got["generated"]
        new = len(got["generated"]) - len(_GQA_IDS)
        # Not `== _GQA_NEW_TOKENS`: `max_new_tokens` is a ceiling, and this
        # model's stopping criteria fire first -- upstream returns 7 new
        # tokens for a ceiling of 8, measured. Asserting the ceiling would
        # have been asserting something upstream does not do.
        assert 1 < new <= _GQA_NEW_TOKENS, (attn, got["generated"])


# Which attention implementations reach the end of `generate()`.
#
# `eager` joined `sdpa` here when `aten.index.Tensor` learned multi-tensor
# advanced indexing (docs/BF16.md §5). The wall it used to be behind was
# reached from the eager mask builder, not from any attention kernel, which
# is why the *forward* had always worked on both while `generate` worked on
# only one. `test_eager_generate_stops_at_index_tensor_and_says_so` used to
# pin that refusal by name and was deleted rather than relaxed: the thing it
# described no longer happens, and a test kept alive by weakening its
# assertion stops being evidence of anything.
_GENERATE_PATHS = ("sdpa", "eager")


def test_eager_generate_reaches_the_end_and_uses_multi_tensor_indexing():
    """`generate(attn_implementation="eager")` completes, and for the reason
    claimed.

    The completion alone is checked by
    `test_greedy_generate_matches_upstream_token_for_token`, which now runs
    both paths. What this adds is the *why*: it calls the op that used to be
    the wall, with the shape the eager mask builder calls it with (measured:
    `self` bool `(1, kv)`, indices `(1,1,1,1)` and `(1,1,1,kv)`, both int64),
    so that "eager generates now" cannot be satisfied by `generate` quietly
    taking some other route.
    """
    if not _ckpt_shim_available() or _upstream_transformers is None:
        return
    got = _gqa_road_fixture("eager")
    assert got["forward"] == "OK", got["forward"]
    assert got["generate"] == "OK", got["generate"][:400]

    # The call the mask builder makes, at the shape it makes it.
    kv = 4
    mask = _C._tensor_from_flat([1, 1, 0, 1], [1, kv], dtype=_C.bool)
    batch = _C._tensor_from_flat([0], [1, 1, 1, 1], dtype=_C.int64)
    keys = _C._tensor_from_flat(list(range(kv)), [1, 1, 1, kv], dtype=_C.int64)
    out = _C._aten_dispatch("aten.index.Tensor", mask, [batch, keys])
    assert out.tolist() == [[[[True, True, False, True]]]], out.tolist()

    if _upstream_torch is not None:
        want = _upstream_torch.ops.aten.index.Tensor(
            _upstream_torch.tensor([[True, True, False, True]]),
            [
                _upstream_torch.tensor([0]).reshape(1, 1, 1, 1),
                _upstream_torch.arange(kv).reshape(1, 1, 1, kv),
            ],
        )
        assert out.tolist() == want.tolist(), (out.tolist(), want.tolist())


def test_the_two_grouped_attention_walls_are_refused_by_name_not_by_shape():
    """What the kernel does when it cannot do the right thing.

    Both refusals were measured on upstream first and neither is a guess:

      * `enable_gqa=False` with mismatched heads is an upstream error, and it
        comes from the broadcast rather than from a head check: "The size of
        tensor a (4) must match the size of tensor b (2) at non-singleton
        dimension 1". The shim must not answer here just because its kernel
        now knows how to repeat heads -- the flag is what asks for it.
      * `enable_gqa=True` with head counts that do not divide is an upstream
        error too: "Number of heads in key and value must divide the number
        of heads in query".

    The aten op underneath has neither check -- it answers both, and for the
    non-divisible case part of the answer is an out-of-bounds read (measured:
    one head came back at 2.4e+31 with unit-magnitude inputs). So the shim's
    aten kernel refuses non-divisible counts by name, and this test asserts
    the refusal names the op rather than surfacing as a candle shape error,
    which is the difference between a work item and a mystery.
    """
    def q(h):
        return _C._tensor_from_flat(
            _e2e_det(1 * h * 3 * 4, 11), [1, h, 3, 4], _C.float32
        )

    # Divisible: the kernel answers.
    out = _C._aten_dispatch(
        "aten._scaled_dot_product_flash_attention_for_cpu.default",
        q(4), q(2), q(2), 0.0, False,
    )
    assert tuple(out[0].shape) == (1, 4, 3, 4), out[0].shape

    # Non-divisible, both directions: refused, by name.
    for h_q, h_kv in [(4, 3), (3, 4), (9, 2)]:
        try:
            _C._aten_dispatch(
                "aten._scaled_dot_product_flash_attention_for_cpu.default",
                q(h_q), q(h_kv), q(h_kv), 0.0, False,
            )
        except NotImplementedError as e:
            text = str(e)
            assert "_scaled_dot_product_flash_attention_for_cpu" in text, text
            assert str(h_q) in text and str(h_kv) in text, text
        else:
            raise AssertionError(f"h_q={h_q} h_kv={h_kv} must be refused")


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


# -- the op docs/CAPTURE.md §5 named ------------------------------------------
#
# `aten.t.default` takes two rounds: its own rule emits `aten.transpose.int`,
# whose rule emits `aten.permute.default`, which is Core ATen. Both rounds were
# blocked until this build could see them -- the second one twice over, first
# because `_jit_get_operation` reported `["default"]` for every packet so
# upstream's `transpose` rule sat on a key that does not exist, and then
# because the recording spells its arguments with the *schema's* names and
# upstream's rule does not use them.
_t_tensor = torch.ones(4, 8)
torch._C._capture_begin([_t_tensor])
_t_trace = torch._C._capture_end(d("aten.t.default", _t_tensor))
_t_lowered = decompose(_t_trace)
out["t_ops_before"] = [n["op"] for n in _t_trace.nodes]
out["t_ops_after"] = _t_lowered.ops
(_t_replayed,) = _t_lowered.replay([_t_tensor])
(_t_captured,) = _t_trace.replay([_t_tensor])
out["t_shape"] = list(_t_replayed.shape)
out["t_bits"] = [
    _t_replayed.reshape(-1).tolist(),
    _t_captured.reshape(-1).tolist(),
    d("aten.t.default", _t_tensor).reshape(-1).tolist(),
]

# The same two-round road, entered at the second round.
torch._C._capture_begin([_t_tensor])
_tr_trace = torch._C._capture_end(torch.transpose(_t_tensor, 0, 1))
out["transpose_kwargs"] = sorted(_tr_trace.nodes[0]["kwargs"])
out["transpose_ops_after"] = decompose(_tr_trace).ops

# Arithmetic, not data movement: `matmul` reaches `mm`, and `isin` reaches
# three ops that actually compute. §5 of docs/DECOMP.md is about whether the
# answer survives, so these are replayed against eager below.
_mm_a, _mm_b = torch.ones(3, 4) * 0.5, torch.ones(4, 5) * 0.25
torch._C._capture_begin([_mm_a, _mm_b])
_mm_trace = torch._C._capture_end(d("aten.matmul.default", _mm_a, _mm_b))
_mm_lowered = decompose(_mm_trace)
out["matmul_ops_after"] = _mm_lowered.ops
(_mm_replayed,) = _mm_lowered.replay([_mm_a, _mm_b])
out["matmul_bits"] = [
    _mm_replayed.reshape(-1).tolist(),
    d("aten.matmul.default", _mm_a, _mm_b).reshape(-1).tolist(),
]

_isin_a = d("aten.arange.start_step", 0, 6, 1)
_isin_b = d("aten.arange.start_step", 2, 4, 1)
torch._C._capture_begin([_isin_a, _isin_b])
_isin_trace = torch._C._capture_end(d("aten.isin.Tensor_Tensor", _isin_a, _isin_b))
_isin_lowered = decompose(_isin_trace)
out["isin_ops_after"] = _isin_lowered.ops
(_isin_replayed,) = _isin_lowered.replay([_isin_a, _isin_b])
out["isin_bits"] = [
    _isin_replayed.reshape(-1).tolist(),
    d("aten.isin.Tensor_Tensor", _isin_a, _isin_b).reshape(-1).tolist(),
]

# What the CompositeImplicitAutograd half of the table is now worth: the four
# alias keys the file is authoritative for, and the one this unblocked.
out["cia_registrations"] = len(
    torch._C._dispatch_get_registrations_for_dispatch_key("CompositeImplicitAutograd")
)
try:
    torch._C._dispatch_get_registrations_for_dispatch_key("CPU")
except NotImplementedError as error:
    out["cia_backend_key"] = str(error)
else:
    out["cia_backend_key"] = "ANSWERED"

# The whole of docs/CAPTURE.md §5's example, end to end. Upstream's answer
# for the same module (quoted in docs/DECOMP.md §7) is
#   aten.permute.default, aten.addmm.default, aten.relu.default,
#   aten.permute.default, aten.addmm.default
# and that is what this has to produce, op for op.
import torch.nn as _nn

torch.manual_seed(0)
_model = _nn.Sequential(_nn.Linear(4, 8), _nn.ReLU(), _nn.Linear(8, 3))
_model_x = torch.ones(2, 4)
torch._C._capture_begin([_model_x])
_model_y = _model(_model_x)
out["model_reason"] = torch._C._capture_reason()
_model_trace = torch._C._capture_end(_model_y)
out["model_ops_before"] = [n["op"] for n in _model_trace.nodes]
out["model_non_core_before"] = non_core_ops(out["model_ops_before"])
_model_lowered = decompose(_model_trace)
out["model_ops_after"] = _model_lowered.ops
out["model_non_core_after"] = non_core_ops(_model_lowered.ops)
out["model_pairs"] = []
for _scale in (1.0, 0.5, -2.0, 7.25):
    _z = torch.ones(2, 4) * _scale
    (_lowered_result,) = _model_lowered.replay([_z])
    (_captured_result,) = _model_trace.replay([_z])
    out["model_pairs"].append([
        _lowered_result.reshape(-1).tolist(),
        _captured_result.reshape(-1).tolist(),
        _model(_z).reshape(-1).tolist(),
    ])

# `OpOverload.tags`, which used to be `[]` for everything (docs/DECOMP.md §2).
out["tags_addmm"] = [t.name for t in torch.ops.aten.addmm.default.tags]
out["tags_dropout"] = sorted(t.name for t in torch.ops.aten.dropout.default.tags)
out["unknown_tags"] = torch._C._shim_unknown_tags()
_tagged_core = set()
for _key in torch._C._aten_all_implemented():
    _ns, _name, _ov = _key.split(".")
    if torch.Tag.core in getattr(getattr(torch.ops, _ns), _name).__getattr__(_ov).tags:
        _tagged_core.add(_key)
out["tag_core_vs_file"] = sorted(
    _tagged_core ^ {k for k in torch._C._aten_all_implemented() if is_core(k)}
)
out["tag_core_count"] = len(_tagged_core)

# The packet collapse itself, at the level it happened.
out["overloads_transpose"] = torch.ops.aten.transpose.overloads()
out["overloads_rsub"] = torch.ops.aten.rsub.overloads()
out["overloads_relu"] = torch.ops.aten.relu.overloads()
import torch._decomp as _D

out["registry"] = len(_D.global_decomposition_table["post_autograd"])
out["registry_default"] = sum(
    1 for k in _D.global_decomposition_table["post_autograd"]
    if str(k).endswith(".default")
)


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


# No rule at all -- not in the reachable table and not upstream's either.
# `aten.transpose.int` used to stand here, and it does not any more: its rule
# was always in the tree, keyed on an overload that did not exist
# (docs/DECOMP.md §3). `aten.reshape.default` is the honest replacement --
# upstream has no Python rule for it at all, so it is the wall itself rather
# than a wall this build put up.
out["refuse_no_rule"] = refusal(
    torch.ones(3, 4), lambda t: d("aten.reshape.default", t, [4, 3])
)
# A rule exists, and running it hits something the shim does not have.
# `aten.t.default` used to stand here too and now lowers; `aten.zeros_like`
# is blocked one layer down, on `torch.full_like` having no overload entry.
out["refuse_unrunnable"] = refusal(
    torch.ones(4, 8), lambda t: d("aten.zeros_like.default", t)
)
# Wall 3 -- "a rule exists, runs, and produces a result the recording
# disagrees with" -- has no example any more, and this is the census that
# says so rather than an absence nobody looked for.
#
# `aten.baddbmm.default` was the example: its decomposition multiplies by
# the Python floats `beta`/`alpha` and came out float64 for float32 inputs.
# The cause is `TensorBase.dtype` returning a fresh object per read, so
# `a.dtype is b.dtype` was false and upstream's `get_higher_dtype` fell past
# its `if a is b: return a` guard into the `ordered_datatypes` table
# (docs/BIND.md §9). Fixed, so `baddbmm` lowers -- recorded below as its own
# case, the way `sum.default` was when the kernel bug it caught was fixed.
def verdict(inputs, call):
    torch._C._capture_begin(list(inputs))
    produced = call()
    trace = torch._C._capture_end(produced)
    try:
        decompose(trace)
    except DecompositionRefused as error:
        text = str(error)
        if "has no rule for it" in text:
            return "NO_RULE"
        if "raised" in text:
            return "RAISED_INSIDE"
        return "DISAGREES"
    return "LOWERED"


# Every non-core implemented op that lowers today -- which is the whole
# population a disagreement could come from, because an op that refuses at
# wall 1 or wall 2 never reaches the meta comparison at all. Taken from
# `pytests/decomp_sweep.py`'s LOWERED column, one entry per op.
_c1, _a1, _b1 = torch.ones(2, 3, 5), torch.ones(2, 3, 4), torch.ones(2, 4, 5)
_x1, _y1 = torch.ones(3, 4), torch.ones(3, 4) * 2.0
_m1, _m2 = torch.ones(3, 4), torch.ones(4, 2)
_i1, _i2 = torch.ones(3, 4), torch.ones(2)
_census = [
    ("aten._unsafe_view.default", [_x1], lambda: d("aten._unsafe_view.default", _x1, [4, 3])),
    ("aten.baddbmm.default", [_c1, _a1, _b1], lambda: d("aten.baddbmm.default", _c1, _a1, _b1)),
    ("aten.detach.default", [_x1], lambda: d("aten.detach.default", _x1)),
    ("aten.isin.Tensor_Tensor", [_i1, _i2], lambda: d("aten.isin.Tensor_Tensor", _i1, _i2)),
    ("aten.matmul.default", [_m1, _m2], lambda: d("aten.matmul.default", _m1, _m2)),
    ("aten.split.Tensor", [_x1], lambda: d("aten.split.Tensor", _x1, 2, 0)),
    ("aten.stack.default", [_x1, _y1], lambda: d("aten.stack.default", [_x1, _y1], 0)),
    ("aten.sum.default", [_x1], lambda: d("aten.sum.default", _x1)),
    ("aten.t.default", [_x1], lambda: d("aten.t.default", _x1)),
    ("aten.transpose.int", [_x1], lambda: d("aten.transpose.int", _x1, 0, 1)),
]
out["wall3_census"] = {name: verdict(inputs, call) for name, inputs, call in _census}

# A census that reports "none" is worth nothing unless it *can* report one.
# Force the meta comparison to disagree and check the same probe changes its
# answer, then put it back and check it changes back. Without this the
# wall-3 assertion in the test would pass just as happily against a census
# that had quietly stopped looking.
_decompose_module = sys.modules[decompose.__module__]
_real_meta_matches = _decompose_module._meta_matches
_decompose_module._meta_matches = lambda want, got: False
out["wall3_control_forced"] = verdict(
    [_c1, _a1, _b1], lambda: d("aten.baddbmm.default", _c1, _a1, _b1)
)
_decompose_module._meta_matches = _real_meta_matches
out["wall3_control_restored"] = verdict(
    [_c1, _a1, _b1], lambda: d("aten.baddbmm.default", _c1, _a1, _b1)
)

# `baddbmm` as a lowers-correctly case -- the `sum.default` treatment.
torch._C._capture_begin([_c1, _a1, _b1])
_bb_produced = d("aten.baddbmm.default", _c1, _a1, _b1)
_bb_trace = torch._C._capture_end(_bb_produced)
_bb_lowered = decompose(_bb_trace)
out["baddbmm_ops_after"] = _bb_lowered.ops
(_bb_replayed,) = _bb_lowered.replay([_c1, _a1, _b1])
out["baddbmm_replayed_dtype"] = str(_bb_replayed.dtype)
out["baddbmm_replayed_shape"] = list(_bb_replayed.shape)
out["baddbmm_replayed"] = _bb_replayed.reshape(-1).tolist()
out["baddbmm_eager"] = _bb_produced.reshape(-1).tolist()
# The identity contract the fix restored, asserted where the bug showed up
# and not only where it was diagnosed.
out["dtype_is_singleton"] = _x1.dtype is torch.float32
out["dtype_is_self"] = _x1.dtype is _x1.dtype

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


def test_decompose_gets_the_full_upstream_table_now():
    """The fuller table arrived, and this is the assertion that said so.

    `core_aten_decompositions()` is the table this pass wants, and it used to
    raise: its constructor enumerates every CompositeImplicitAutograd
    registration through `torch._C._dispatch_get_registrations_for_dispatch_key`
    and this `_C` has no C++ dispatcher. The shim answers that query now, out of
    `native_functions.yaml` -- 743 aten names against upstream's 744, the one
    difference being a TorchScript builtin the file does not carry
    (docs/DECOMP.md §3).

    Backend keys are *not* answered, and that is asserted here rather than
    left as a comment: the same file lists which ops upstream's C++ registers
    under `CPU`, and answering with it would claim about 1500 kernels this
    build does not have.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    r = _decomp_road_fixture()
    assert r["table_source"] == "core_aten_decompositions", r["table_source"]
    assert r["table_reason"] is None, r["table_reason"]
    # 415 until `zeros_like.out` reached the registry -- the same one entry
    # `test_a_packet_reports_the_overloads_the_file_declares` counts, seen from
    # the decomposition table's side. `cia_registrations` reads
    # `native_functions.yaml` directly and so does not move with the tables.
    # 417 with `ones_like.out`, which is the same one entry again, from the
    # decomposition table's side: `overloads.json` now carries
    # `aten::ones_like.out`, the yaml does not declare it, so
    # `register_decomposition(aten.ones_like)` has a schema to resolve and
    # reaches one more overload. `cia_registrations` still does not move,
    # because it reads `native_functions.yaml` directly.
    assert r["table_size"] == 417, r["table_size"]
    assert r["cia_registrations"] == 743, r["cia_registrations"]
    assert r["cia_backend_key"] != "ANSWERED"
    assert "backend key" in r["cia_backend_key"], r["cia_backend_key"]


def test_a_packet_reports_the_overloads_the_file_declares():
    """`_jit_get_operation` answered `["default"]` for every packet.

    That is one line of `bootstrap.py` and it cost the decomposition registry
    half its entries. `torch/_decomp/__init__.py:82` expands a packet-level
    `@register_decomposition(aten.transpose)` by walking
    `packet.op_overloads()`, so upstream's rule for `transpose` was registered
    against `aten.transpose.default` -- an overload that does not exist in any
    torch. The rule was in the vendored tree the whole time, under a name
    nothing would look up.

    The numbers are the measurement: the registry held 592 entries of which
    525 ended in `.default`, against upstream's 1097 and 456. Reading the
    overload names out of `native_functions.yaml` puts it at 1004/461.

    `relu` is here as the control. It really does have exactly one, unnamed,
    overload, so `["default"]` was the right answer for it -- which is why the
    bug was invisible to anything that only looked at simple ops.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    r = _decomp_road_fixture()
    # `transpose` is the one docs/DECOMP.md §3 measured: upstream's packet has
    # exactly one overload and it is *not* `default`, which is what made
    # `["default"]` a wrong answer rather than a lossy one.
    assert r["overloads_transpose"] == ["int"], r["overloads_transpose"]
    assert r["overloads_rsub"] == ["Tensor", "Scalar"], r["overloads_rsub"]
    assert r["overloads_relu"] == ["default"], r["overloads_relu"]
    # 1005 until docs/ARCH20.md added a `zeros_like` entry to
    # `overloads.json`. It moved for exactly the reason `floor_divide.Scalar_out`
    # moved it before (see `test_schema_text_survives_the_round_trip...`): the
    # table now carries `aten::zeros_like.out`, which the yaml does not declare,
    # so `register_decomposition(aten.zeros_like)` has a schema to resolve and
    # reaches one more overload. `registry_default` does not move, because the
    # new one is `.out` and not `.default` -- which is the check that this is
    # the same mechanism and not a regression of the `["default"]` bug.
    # 1007 with `ones_like`, for the third time by the same mechanism (see
    # `zeros_like` above and `floor_divide.Scalar_out` before it):
    # `overloads.json` carries `aten::ones_like.out`, the yaml does not declare
    # it, so the packet resolves one more overload. `registry_default` is again
    # unchanged, because the new one is `.out` -- which is the check that this
    # is that mechanism and not a regression of the `["default"]` bug.
    #
    # `detach` went into `overloads.json` in the same change and moved NOTHING
    # here: `aten::detach` has no `.out` variant, so there is no undeclared
    # overload for a packet to reach.
    assert r["registry"] == 1007, r["registry"]
    assert r["registry_default"] == 461, r["registry_default"]


def test_core_ops_and_op_tags_agree():
    """`OpOverload.tags` answered `[]` for every op, and two things read it.

    docs/DECOMP.md §2 measured the first: `torch.Tag.core in op.tags` was False
    for all 120 implemented ops, so a Core ATen classifier built the obvious
    way answers "nothing is core" and refuses whole programs. `core_ops()`
    routed around it by reading `native_functions.yaml` itself, which was right
    for that module and did nothing for the tree's own readers.

    The second is `torch/_decomp/__init__.py:57`, which reads
    `maybe_aliasing_or_mutating` to decide whether an op may be preserved
    rather than decomposed. With `[]` that question was always False, and
    `_collect_all_valid_cia_ops` collected `aten.dropout.default` and
    `aten.unsafe_chunk.default`, which upstream excludes for exactly that tag.

    Tags come from the file now. The two classifiers are diffed against each
    other here rather than each being asserted alone -- one source read twice
    would agree with itself, and these are two different scans of it reaching
    two different consumers.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    r = _decomp_road_fixture()
    # `pt2_compliant_tag` is on every aten entry and is not written on any of
    # them: `torchgen/model.py:756` adds it while parsing. Three tags work that
    # way (`out` and `inplace` are the others) and leaving them out put 0 of
    # 118 implemented ops in agreement with upstream.
    assert r["tags_addmm"] == ["core", "pt2_compliant_tag"], r["tags_addmm"]
    assert r["tags_dropout"] == [
        "maybe_aliasing_or_mutating",
        "nondeterministic_seeded",
        "pt2_compliant_tag",
    ], r["tags_dropout"]
    # A tag name the file uses that `_C.Tag` has no member for would be dropped
    # silently; this is the count of that happening.
    assert r["unknown_tags"] == [], r["unknown_tags"]
    assert r["tag_core_vs_file"] == [], r["tag_core_vs_file"]
    # 77 until `aten.ge.Tensor` got a kernel (docs/VIEWS.md §1). This counts
    # *implemented* ops that upstream tags `core`, so it moves whenever a
    # core-tagged op is implemented -- and `ge.Tensor` is core upstream
    # exactly as its `le`/`lt`/`gt` siblings already counted here are.
    # 78 until docs/ARCH20.md, which implemented four more ops that upstream
    # tags `core`: `clamp.default`, `constant_pad_nd.default`, `expm1.default`
    # and `log.default`. The seven in-place kernels the same round added
    # (`sub_`, `mul_`, `neg_`, `exp_` and the `Scalar` overloads) are NOT core
    # upstream and do not appear here -- checked, not assumed, which is what
    # makes 82 the right number rather than 85.
    # 83 with `aten.amax.default` (docs/SEQLEN.md §7). Read off
    # `torch.ops.aten.amax.default.tags` -- `['core', 'pt2_compliant_tag',
    # 'reduction']` -- rather than inferred from its neighbours, because
    # `max.dim` sitting next to it in this shim is *not* core.
    #
    # 84 with docs/TRIL.md, and the *one* is the point: that round put five new
    # keys into `_aten_implemented()` and only `min.dim` is core. Every one was
    # read off its own `.tags` rather than inferred from a sibling, which is
    # what stops this from being 88:
    #
    #     min.dim      ['core', 'pt2_compliant_tag', 'reduction']   <- counted
    #     min.other    ['pointwise', 'pt2_compliant_tag']
    #     max.other    ['pointwise', 'pt2_compliant_tag']           (promoted, not new)
    #     tril.default ['pt2_compliant_tag']
    #     triu.default ['pt2_compliant_tag']
    #
    # `min.dim` is core while `max.dim` -- the same overload of the mirror op,
    # implemented by the same function -- is not. There is no rule to derive
    # that from; it is upstream's table and it has to be read.
    #
    # 85 with docs/KERNELS26.md's `sqrt`, read off
    # `torch.ops.aten.sqrt.default.tags` -- `['core', 'pointwise',
    # 'pt2_compliant_tag']` -- and not inferred from `rsqrt`, which happens to
    # carry the same three. This number moves once per core-tagged kernel that
    # round adds; the kernels it adds that are NOT core upstream (checked, one
    # at a time, on each op's own `.tags`) do not appear here.
    #
    # 86 with `repeat`, whose tags are `['core', 'pt2_compliant_tag']` -- core
    # but *not* `pointwise`, which is the check that these are being read off
    # each op rather than copied from the last one added.
    #
    # 88 with `remainder.Scalar` and `remainder.Tensor`, **both** of which are
    # `['core', 'pointwise', 'pt2_compliant_tag']`. +2 rather than +1 because
    # this counts overloads, not ops -- and two overloads of one op do not have
    # to agree (`min.dim` is core while `max.dim` is not, above), so both were
    # read rather than one inferred from the other.
    # 90 with `div.Tensor_mode` and `div.Scalar_mode`, both
    # `['core', 'pointwise', 'pt2_compliant_tag']` -- read off each overload's
    # own `.tags` rather than copied from `div.Tensor` beside them, which
    # happens to carry the same three. +2 for the same reason `remainder` was
    # +2: this counts overloads.
    assert r["tag_core_count"] == 90, r["tag_core_count"]


def test_decompose_lowers_the_op_capture_md_named():
    """`aten.t.default` -- the op the smallest model in docs/CAPTURE.md emits.

    Two rounds, and upstream's own answer for the same model:
    `t` -> `transpose.int` -> `permute`, which is Core ATen. Both rules are
    upstream's; nothing about the road is written in this repository.

    Three walls stood between here and this and all three were somewhere else:
    `_jit_get_operation` collapsing every packet onto `.default` (so the
    `transpose` rule was unreachable), `overloads.json` having no `transpose`
    or `permute` entry (so neither rule could *run*), and the pass calling a
    rule with the recording's kwargs, whose names are the aten schema's and
    not the rule's (`TypeError: transpose() got an unexpected keyword argument
    'self'`).

    The last one is asserted directly, through the recorded kwarg names, so
    that a regression that goes back to forwarding them fails here and not
    only through the two-round road above.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    r = _decomp_road_fixture()
    assert r["t_ops_before"] == ["aten.t.default"], r["t_ops_before"]
    assert r["t_ops_after"] == ["aten.permute.default"], r["t_ops_after"]
    assert r["t_shape"] == [8, 4], r["t_shape"]
    lowered, captured, eager = r["t_bits"]
    assert lowered == eager, (lowered, eager)
    assert captured == eager, (captured, eager)
    # The recording really does spell its operands with the schema's names --
    # this is the input the pass has to re-order, not a hypothetical.
    assert r["transpose_kwargs"] == ["dim0", "dim1", "self"], r["transpose_kwargs"]
    assert r["transpose_ops_after"] == ["aten.permute.default"], r[
        "transpose_ops_after"
    ]


def test_the_smallest_model_lowers_the_way_upstream_lowers_it():
    """docs/CAPTURE.md §5's example, whole, against upstream's own answer.

    That section chose `nn.Sequential(Linear, ReLU, Linear)` because it is the
    smallest thing anyone would export and it already emits `aten.t.default`,
    which Edge will not take. docs/DECOMP.md §7 then asked upstream what it
    does with the same module:

        ep.run_decompositions(core_aten_decompositions())
        # -> aten.permute.default, aten.addmm.default, aten.relu.default,
        #    aten.permute.default, aten.addmm.default

    This asserts that list, op for op and in order. It is the only test here
    that is about a *module* rather than a hand-built trace, and it is the one
    that says the chain from capture to Core ATen is connected rather than
    connected in the places that were measured.

    Three inputs the trace never saw are replayed as well, because a lowering
    that only agrees on the recorded input would agree for the wrong reason.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    r = _decomp_road_fixture()
    assert r["model_reason"] is None, r["model_reason"]
    assert r["model_ops_before"] == [
        "aten.t.default",
        "aten.addmm.default",
        "aten.relu.default",
        "aten.t.default",
        "aten.addmm.default",
    ], r["model_ops_before"]
    assert r["model_non_core_before"] == ["aten.t.default"], r[
        "model_non_core_before"
    ]
    assert r["model_ops_after"] == [
        "aten.permute.default",
        "aten.addmm.default",
        "aten.relu.default",
        "aten.permute.default",
        "aten.addmm.default",
    ], r["model_ops_after"]
    assert r["model_non_core_after"] == [], r["model_non_core_after"]
    assert len(r["model_pairs"]) == 4
    for lowered, captured, eager in r["model_pairs"]:
        assert lowered == eager, (lowered, eager)
        assert captured == eager, (captured, eager)


def test_the_newly_opened_decompositions_still_agree_with_eager():
    """§5 again, on rules that are not pure data movement.

    The bit-for-bit result in `test_decomposed_replay_matches_eager_bit_for_
    bit` is evidence about *those* rules: `stack`->`cat`+`view`,
    `split`->`split_with_sizes`, `_unsafe_view`->`view` all move bytes and do
    no arithmetic, so there was nothing for the last bit to disagree about.

    `matmul`->`mm` and `isin`->`view`+`eq`+`any` are the first newly-reachable
    rules that compute. They agree exactly too, and that is measured here
    rather than assumed -- if a future rule does diverge, the reply is to
    record how far (docs/DEVICE.md §5), not to widen a comparison.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    r = _decomp_road_fixture()
    assert r["matmul_ops_after"] == ["aten.mm.default"], r["matmul_ops_after"]
    lowered, eager = r["matmul_bits"]
    assert lowered == eager, (lowered, eager)
    assert r["isin_ops_after"] == [
        "aten.view.default",
        "aten.eq.Tensor",
        "aten.any.dims",
    ], r["isin_ops_after"]
    lowered, eager = r["isin_bits"]
    assert lowered == eager, (lowered, eager)


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

    **Walls 1 and 2 have examples. Wall 3 currently has none**, and that is
    asserted here as a fact rather than papered over with an invented one.
    Its last example was `aten.baddbmm.default`, whose decomposition promoted
    float32 to float64; the cause was `TensorBase.dtype` handing back a fresh
    object per read (docs/BIND.md §9), it is fixed, and `baddbmm` now lowers
    (`test_decompose_lowers_baddbmm_default_now_that_the_dtype_is_a_singleton`).
    That is the same thing that happened to `aten.sum.default` one round
    earlier, and to `aten.t.default` before it.

    The wall itself is not gone -- `decompose` still compares every
    decomposition's meta against the recording and still refuses on a
    mismatch. What is gone is an op that trips it. So this asserts the census
    (every op that reaches the comparison agrees) together with a **positive
    control**: with the comparison forced to disagree, the same probe reports
    `DISAGREES`. Without that control, "no op disagrees" would pass equally
    well if the check had stopped running, which is the failure mode
    docs/DECOMP.md §7.2 now records.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    r = _decomp_road_fixture()

    # 1. Nothing has a rule for it -- upstream included. `aten.transpose.int`
    #    used to be this example and it lowers now: its rule existed all along
    #    and the packet collapse hid it (docs/DECOMP.md §3). The replacement is
    #    an op upstream really has no Python rule for.
    assert r["refuse_no_rule"] != "ACCEPTED"
    assert "aten.reshape.default" in r["refuse_no_rule"], r["refuse_no_rule"]
    assert "no rule" in r["refuse_no_rule"], r["refuse_no_rule"]

    # 2. A rule exists and running it reaches something the shim lacks. The
    #    refusal carries the underlying reason, so the gap is findable.
    assert r["refuse_unrunnable"] != "ACCEPTED"
    assert "aten.zeros_like.default" in r["refuse_unrunnable"], r["refuse_unrunnable"]
    assert "torch.full_like" in r["refuse_unrunnable"], r["refuse_unrunnable"]

    # 3. A rule exists, runs, and produces a result the recording disagrees
    #    with. **No op does this any more.** Every non-core op that reaches
    #    the meta comparison agrees with the recording, so the census is all
    #    LOWERED and there is nothing to name here. Asserted per op rather
    #    than as a count, so a *new* disagreement names itself instead of
    #    only moving a number.
    census = r["wall3_census"]
    assert census, census
    disagreeing = sorted(k for k, v in census.items() if v == "DISAGREES")
    assert disagreeing == [], (
        "wall 3 has an example again -- promote it: make this the named case "
        f"and give the census a lowers-correctly entry instead. {disagreeing}"
    )
    # Every entry must actually have reached the comparison; an op that
    # started refusing earlier would drop out of the population silently.
    assert sorted(census) == sorted(
        [
            "aten._unsafe_view.default",
            "aten.baddbmm.default",
            "aten.detach.default",
            "aten.isin.Tensor_Tensor",
            "aten.matmul.default",
            "aten.split.Tensor",
            "aten.stack.default",
            "aten.sum.default",
            "aten.t.default",
            "aten.transpose.int",
        ]
    ), sorted(census)
    assert set(census.values()) == {"LOWERED"}, census

    # The positive control: the census can still report a disagreement.
    assert r["wall3_control_forced"] == "DISAGREES", r["wall3_control_forced"]
    assert r["wall3_control_restored"] == "LOWERED", r["wall3_control_restored"]


def test_decompose_lowers_baddbmm_default_now_that_the_dtype_is_a_singleton():
    """Wall 3's last example, turned into proof that its cause is fixed.

    `aten.baddbmm.default`'s upstream rule is `@pw_cast_for_opmath`-decorated,
    so it runs `elementwise_dtypes` -> `get_higher_dtype`, which opens with

        if a is b: return a

    before it consults `ordered_datatypes`. `TensorBase.dtype` used to build a
    fresh `PyDtype` on every read, so two float32 operands were never `is`
    each other, both fell past that guard, and the pair came out **float64**.
    The decomposition then produced a float64 result where the recording had
    float32, `decompose` caught the divergence, and refused -- correctly, on a
    real bug. docs/BIND.md §9 has the diagnosis; docs/DECOMP.md §7.2 had
    carried it as "cause unknown" until then.

    `dtype` is interned now, so the rule and the recording agree and the trace
    lowers to `bmm` + `add`. All three links are asserted, so a regression
    says which one broke: the identity contract itself, the lowered op list,
    and that replaying the lowered graph reproduces eager's values in
    float32.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return
    r = _decomp_road_fixture()
    # The cause, at the level it lives: `t.dtype` is the module-level object.
    assert r["dtype_is_singleton"] is True
    assert r["dtype_is_self"] is True
    # The rule and the recording agree, so it lowers instead of refusing.
    assert r["baddbmm_ops_after"] == ["aten.bmm.default", "aten.add.Tensor"], r[
        "baddbmm_ops_after"
    ]
    # ...and the result is float32, which is the divergence stated directly.
    assert r["baddbmm_replayed_dtype"] == "torch.float32", r["baddbmm_replayed_dtype"]
    assert r["baddbmm_replayed_shape"] == [2, 3, 5], r["baddbmm_replayed_shape"]
    # `baddbmm(ones(2,3,5), ones(2,3,4), ones(2,4,5))` is 1 + 4 elementwise.
    assert r["baddbmm_replayed"] == [5.0] * 30, r["baddbmm_replayed"][:8]
    assert r["baddbmm_replayed"] == r["baddbmm_eager"], r["baddbmm_eager"][:8]


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
                 ("meta", "meta"), ("rope3", "rope3")):
    load("hard_" + tag, os.path.join(sys.argv[5], sub))

# --- and the user's line verbatim, on the llama3-rope checkpoint: not just
# --- `from_pretrained` but the `generate` that follows it.
try:
    m = AutoModelForCausalLM.from_pretrained(os.path.join(sys.argv[5], "rope3"))
    m.eval()
    with torch.no_grad():
        out["rope3_generate"] = m.generate(
            torch.tensor([IDS]), max_new_tokens=8, do_sample=False).tolist()
except BaseException:
    out["rope3_generate"] = "FAILED: " + traceback.format_exc(limit=6)

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
    #   rope3  `rope_scaling={"rope_type": "llama3", ...}`, which is what
    #          `meta-llama/Llama-3.2-1B` -- README.md's own headline example --
    #          has and what a plain `LlamaConfig` does not. It is a *meta*
    #          case, not a container one: `from_pretrained` builds the module
    #          tree under `init_empty_weights`, so
    #          `_compute_llama3_parameters` runs on meta tensors, and its
    #          first line is `torch.where(wavelen > low_freq_wavelen, ...)`.
    #          Published 0.0.5a0 stopped there with "no meta kernel for
    #          aten.gt.Scalar" (docs/META.md §7). The plain case cannot see
    #          this: with no `rope_scaling` the default rope init does no
    #          comparisons at all, which is why every other row here passed
    #          while the flagship model did not load.
    hard = os.path.join(root, "hard")
    for tag, kw, extra in (
        ("tied", dict(tie_word_embeddings=True), {}),
        ("shard", dict(num_hidden_layers=3), dict(max_shard_size="6KB")),
        ("bf16", {}, {}),
        ("meta", {}, {}),
        ("rope3", dict(rope_theta=10000.0, rope_scaling={
            "rope_type": "llama3", "factor": 8.0, "low_freq_factor": 1.0,
            "high_freq_factor": 4.0, "original_max_position_embeddings": 16,
        }), {}),
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
        if tag == "rope3":
            # The oracle for the user's line. Greedy, so it is a token
            # sequence and not a tolerance.
            with torch.no_grad():
                expected["rope3_generate"] = m.generate(
                    torch.tensor([_LLAMA_IDS]), max_new_tokens=8,
                    do_sample=False).tolist()
            # The rope config must actually have survived into the saved
            # checkpoint, or this whole row degenerates into a second copy of
            # the plain case. transformers 5 renames the key, so both
            # spellings are accepted -- what is asserted is `llama3`.
            params = getattr(m.config, "rope_parameters", None) or getattr(
                m.config, "rope_scaling", None)
            assert params and params.get("rope_type") == "llama3", params
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


def test_the_llama3_rope_checkpoint_loads_and_generates_like_upstream():
    """**The user's line**, on a checkpoint that has `rope_type: llama3`.

        model = AutoModelForCausalLM.from_pretrained(...)
        out = model.generate(**tok(...), max_new_tokens=...)

    Published 0.0.5a0 could not do the first of those for any llama3-rope
    model -- including `meta-llama/Llama-3.2-1B`, which README.md's own
    headline example loads. `from_pretrained` builds the module tree under
    `init_empty_weights`, so `_compute_llama3_parameters` runs on the meta
    device, and its first line is a comparison
    (`transformers/modeling_rope_utils.py:655`). docs/META.md §7.

    `meta-llama/Llama-3.2-1B` itself is gated on the Hub and is not in this
    machine's cache, so the checkpoint here is built by upstream torch with
    the same `rope_scaling` dict and read back through the same
    `AutoModelForCausalLM.from_pretrained` -- the real path, not a hand-rolled
    meta init. The fixture asserts that the `llama3` rope type survived into
    the saved config, so this cannot quietly become a second copy of the
    plain case.

    Weights are compared bit-for-bit, logits to the same bound as the plain
    case, and `generate` as an exact token sequence -- greedy decoding turns
    the acceptance into an equality rather than a tolerance.
    """
    if not _ckpt_shim_available() or _upstream_transformers is None:
        return
    expected, got = _from_pretrained_fixture()
    assert got["hard_rope3"] == "OK", got["hard_rope3"]
    worst, where = _worst_state_dict_drift(
        expected["hard_rope3"]["state_dict"], got["hard_rope3_state_dict"])
    assert worst == 0.0, (where, worst)
    diff = max(abs(a - b) for a, b in
               zip(got["hard_rope3_logits"], expected["hard_rope3"]["logits"]))
    assert diff < _REAL_LLAMA_ATOL, diff
    assert got["rope3_generate"] == expected["rope3_generate"], (
        got["rope3_generate"], expected["rope3_generate"])


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
    "aten.add_.Scalar",
    "aten.add_.Tensor",
    "aten.clamp_.default",
    "aten.copy_.default",
    "aten.div_.Tensor",
    "aten.exp_.default",
    "aten.fill_.Scalar",
    "aten.fill_.Tensor",
    "aten.index_put_.default",
    "aten.masked_fill_.Scalar",
    "aten.mul_.Scalar",
    "aten.mul_.Tensor",
    "aten.neg_.default",
    "aten.normal_.default",
    "aten.relu_.default",
    "aten.sub_.Scalar",
    "aten.sub_.Tensor",
    "aten.uniform_.default",
    "aten.zero_.default",
)

#: The seven docs/DISTRIBUTED.md §8.1 named, as the judgement it set.
#: docs/ARCH20.md §8 added `sub_`, `mul_`, `neg_` and `exp_` to the *set* above;
#: this tuple stays the seven §8.1 named, because it is a record of that
#: judgement rather than a second copy of the implemented list.
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
    # Derived from `_aten_implemented()`, not pinned to a number. A literal
    # here says "117 operators" where what is meant is "every one of them", so
    # it reddens on the next operator added and teaches the reader to bump the
    # constant -- which is how a coverage assertion decays into a change
    # detector. It did exactly that on the merge that brought the 118th.
    assert len(report["ops"]) == len(_C._aten_implemented()), (
        len(report["ops"]),
        len(_C._aten_implemented()),
    )
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
    # What this test is for is that the predicate can take both values -- the
    # defect it was written against answered one of them for everything. Pinning
    # the two counts instead turns it into a change detector that reddens
    # whenever the operator set moves, which it did on the merge that brought
    # the 118th. Which operators are mutable is asserted by name in
    # `test_the_seven_in_place_ops_say_that_they_mutate`, where a wrong answer
    # is legible; here only the shape of the answer matters.
    assert values.count(True) > 0, values.count(True)
    assert values.count(False) > 0, values.count(False)
    assert values.count(True) + values.count(False) == len(values)
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
    and `verify_schemas.py` keeps them honest. 176 distinct overloads, 7 of
    which exercise a normalisation rule -- if the re-printer drops one, those
    seven stop matching. This needs no upstream torch.

    173 until docs/DECOMP.md's `transpose`/`permute`/`sub` entries arrived;
    three of those five schema strings were already reachable through
    `methods.json`, so the union grew by `aten::permute` and `aten::sub.out`.
    175 until docs/GROUPED_MM.md added `aten::_grouped_mm` and, with it, the
    `floor_divide`/`cumsum`/`histc` entries Mixtral's Python surface needs --
    seven more schema strings. One of the seven is a *fifth* table-only entry:
    `floor_divide.Scalar_out` is torchgen-generated and the yaml does not
    declare it, which is why `from_tables` below grew rather than staying at
    four. That entry is also why the decomposition registry gained one
    (`test_decompose_gets_the_full_upstream_table_now`): with a schema to
    resolve, `register_decomposition(aten.floor_divide)` now reaches it.

    183 until the seven `TensorBase` members Mixtral needs went into
    `methods.json` (docs/GROUPED_MM.md §6.4): `div_` with its four overloads,
    `ge` with two, `masked_fill_` with two and `clamp_` with two -- ten new
    schema strings, all of them declared in the yaml, so `from_tables` is
    unchanged at five. `__idiv__` and `__ge__` add none: they are second
    spellings of `div_.Tensor`/`div_.Scalar` and `ge.Tensor`/`ge.Scalar`, and
    this count is over distinct `(qualname, overload)` pairs, not table keys.

    The provenance assertion is not decoration. In the first working version
    `_get_schema` consulted the tables *before* the file, so these 173 lookups
    were answered by the oracle itself and the comparison was the oracle
    against itself: deleting the float printer entirely left this test green
    (measured). All but four have to come from the file for the comparison to
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
    # 193 until docs/ARCH20.md, which added 22 distinct `(qualname, overload)`
    # pairs across the two tables: the six spellings whose kernels already
    # existed (`exp`, `stack`, `zeros_like`, plus their `.out` siblings), the
    # three new kernels (`log`, `expm1`, `constant_pad_nd`), `clamp`, and the
    # in-place family. The `__iadd__`/`__isub__`/`__imul__` entries add none --
    # they are second spellings of `add_`/`sub_`/`mul_`, and this counts
    # distinct schema identities, not table keys.
    #
    # 217 until docs/SPELLINGS.md's 22-name `overloads.json` batch (this
    # round's fix for docs/ARCH20.md §9's inventory of kernels with no
    # `torch.<name>`). +2, not +22: every schema this round put in
    # `overloads.json` was already a distinct identity somewhere in
    # `methods.json` -- `abs`, `cos`, `sin`, `reciprocal`, `clone`, `clamp`
    # (both overloads), the six comparisons (both overloads each), `max`
    # (all three), `min` (all three), `mul` (both), `reshape`, `unbind`,
    # `bitwise_and`/`bitwise_or` (both overloads each) and `bitwise_not` --
    # because the whole point of this round was giving an *existing* member's
    # kernel a second, function-shaped door, not a new kernel. Only
    # `scalar_tensor` and `convolution` are new identities: neither has a
    # `Tensor` receiver, so neither was ever a candidate for `methods.json`.
    #
    # 220 with docs/TRIL.md's `amax`, `tril` and `triu`. **+3, not +6**, and
    # that is the check rather than an aside: each of the three went into
    # *both* tables in the same change, and this counts distinct
    # `(qualname, overload)` pairs rather than table keys -- so a `torch.<name>`
    # and a `Tensor.<name>` naming the same schema are one identity, exactly as
    # `__iadd__`/`add_` were. Getting +6 here would mean the two tables had
    # transcribed the same op differently.
    #
    # 221 with `_safe_softmax`, which is `overloads.json`-only: upstream has
    # `torch._safe_softmax` and no `Tensor._safe_softmax`, so it adds one
    # identity rather than two.
    #
    # 223 with docs/KERNELS26.md's `sqrt`. **+2, not +3**, for the same reason
    # `amax`/`tril`/`triu` were +3 rather than +6: `sqrt` went into both
    # tables, and `overloads.json`'s `aten::sqrt|default` and
    # `methods.json`'s are one identity. The second is `aten::sqrt|out`, which
    # `overloads.json` carries with no kernel behind it so that
    # `torch.sqrt(x, out=y)` refuses by the right name -- exactly as
    # `rsqrt.out` already did.
    #
    # 224 with `repeat`. **+1**: it is `methods.json`-only, because upstream
    # has no `torch.repeat` at all (`hasattr(torch, "repeat")` is False on
    # 2.13.0 -- there is `Tensor.repeat` and the unrelated
    # `torch.repeat_interleave`), and `aten::repeat` has no `.out` variant
    # this shim carries.
    #
    # 228 with `remainder`. **+4**: `overloads.json` carries all four of
    # `Scalar_out`, `Tensor_out`, `Tensor` and `Scalar` (the two `.out` forms
    # with no kernel, so `torch.remainder(x, y, out=z)` refuses by the right
    # name), and `methods.json`'s `remainder` and `__mod__` add none -- both
    # name the same `Tensor`/`Scalar` pair, and `__mod__` is a second spelling
    # exactly as `__iadd__` is of `add_`.
    #
    # 230 with `ones_like` and `detach`. **+2, not +3**, and which of the two
    # contributed is the check: `ones_like` is in neither table before this
    # change, so its `.out` and default schemas are two new identities, while
    # `detach` was **already in `methods.json`** -- putting it in
    # `overloads.json` gives `torch.detach` a door onto a schema
    # `Tensor.detach` already named, and a second spelling of one schema is one
    # identity, exactly as `__mod__`/`remainder` and `__iadd__`/`add_` are.
    # Getting +3 here would mean the two tables had transcribed `detach`
    # differently.
    assert len(keys) == 230, len(keys)
    from_tables = sorted(
        k for k in keys
        if report["table"][f"{k[0]}|{k[1]}"]["from"] == "tables"
    )
    # `zeros_like.out` is the sixth table-only entry (docs/ARCH20.md): like the
    # five before it, torchgen generates it and the yaml does not declare it.
    # Every other schema this round added IS declared in the yaml, so the list
    # grew by one rather than by twenty-two -- which is the check that the new
    # entries are being answered by the file and not by the oracle itself.
    #
    # `ones_like.out` is the seventh, and it is the same generated-not-declared
    # shape as `zeros_like.out` beside it -- which is why it, and not
    # `ones_like` itself, is what moved the decomposition registry by one.
    # `detach` adds nothing here: it is declared in the yaml.
    assert from_tables == [
        ("aten::div", "Scalar_mode_out"),
        ("aten::div", "Scalar_out"),
        ("aten::embedding", "out"),
        ("aten::empty_like", "out"),
        ("aten::floor_divide", "Scalar_out"),
        ("aten::ones_like", "out"),
        ("aten::zeros_like", "out"),
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


# --- reduced-float rounding -------------------------------------------------
#
# torch computes reduced-float arithmetic in `at::opmath_type` -- `float` for
# both `bfloat16` and `float16` -- and narrows back **once**, at the end, with
# round-to-nearest-even. A kernel that instead computes in the storage dtype,
# or that narrows by truncating, lands within one ulp of the right answer on
# every element, and is therefore invisible to every tolerance-based check in
# this repository: the golden harness compares with
# `tools/golden/dtypes.py::TOLERANCES`, and one bfloat16 ulp is far inside it.
#
# It is not invisible in a model, because the error does not cancel. A biased
# narrowing pushes every rounded element the same way, so 30 residual layers
# accumulate it instead of averaging it out. docs/BF16.md measures exactly
# that: the shim's `add` truncated, the divergence reached an O(1) logit
# difference, and SmolLM2-135M produced different text from upstream on the
# **default** dtype path.
#
# So these tests assert **exact** agreement. That is the only bound that can
# tell "one ulp because addition is hard" apart from "one ulp every time, in
# the same direction". They are also the reason the fix is checkable at all:
# a tolerance here would pass both the right kernel and the wrong one.
_REDUCED_FLOATS = ("bfloat16", "float16")


def _rounding_operands(n):
    """Two deterministic operand lists whose exact sums land on bfloat16
    rounding boundaries often enough to be decisive.

    Not "nice" numbers: 0.5 + 0.25 rounds the same way under every rule, so a
    probe built from tidy constants cannot fail. These come from the same LCG
    the end-to-end tests use, at two magnitudes, so that roughly a quarter of
    the pairs have a tie or near-tie in the discarded bits.
    """
    a = _e2e_det(n, 20260828)
    b = [v * 0.03125 for v in _e2e_det(n, 7654321)]
    return a, b


def _reduced_float_cases(b, dtype, n=256):
    """The op calls this section checks, built on one backend."""
    x = b.t(_rounding_operands(n)[0], (n,), dtype)
    y = b.t(_rounding_operands(n)[1], (n,), dtype)
    grid = b.t(_rounding_operands(n)[0], (16, 16), dtype)
    return {
        # The op the model actually rides on: every residual join and the
        # `(q*cos) + (rotate_half(q)*sin)` of rotary embedding is this one.
        "add.Tensor": lambda: b.op("aten.add.Tensor", x, y),
        "add.Tensor alpha": lambda: b.op("aten.add.Tensor", x, y, alpha=3.0),
        "add.Scalar": lambda: b.op("aten.add.Scalar", x, 0.3),
        "sub.Tensor": lambda: b.op("aten.sub.Tensor", x, y),
        "mul.Tensor": lambda: b.op("aten.mul.Tensor", x, y),
        "div.Tensor": lambda: b.op("aten.div.Tensor", x, y),
        # Reductions accumulate in `acc_type<T>` upstream, which is `float`
        # for both reduced floats -- the same rule `cumsum_default` already
        # states in its doc comment.
        "sum.dim_IntList": lambda: b.op("aten.sum.dim_IntList", grid, [1], False),
        "mean.dim": lambda: b.op("aten.mean.dim", grid, [1], False),
    }


def _flat_values(result):
    out = []
    stack = [result.tolist()]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


def test_reduced_float_arithmetic_narrows_exactly_like_upstream():
    """Every reduced-float op agrees with upstream to the last bit.

    Exact, not close. See the section comment for why a tolerance here would
    be a check that cannot fail.
    """
    if _upstream_torch is None:
        return  # no upstream torch in this interpreter -- see docs/E2E.md
    for dtype in _REDUCED_FLOATS:
        shim = _reduced_float_cases(_E2EBackend("shim"), dtype)
        upstream = _reduced_float_cases(_E2EBackend("upstream"), dtype)
        for name in upstream:
            want = _flat_values(upstream[name]())
            got = _flat_values(shim[name]())
            assert len(got) == len(want), (dtype, name, len(got), len(want))
            wrong = [i for i, (g, w) in enumerate(zip(got, want)) if g != w]
            assert not wrong, (
                f"{dtype} {name}: {len(wrong)}/{len(want)} elements differ from "
                f"upstream; first at {wrong[0]} "
                f"(shim {got[wrong[0]]!r} vs upstream {want[wrong[0]]!r})"
            )


def test_reduced_float_operands_survive_construction():
    """The premise of the test above: both sides start from the same bits.

    Without this, a narrowing bug in `_tensor_from_flat` would show up as a
    difference in every op downstream and be misread as an arithmetic bug.
    """
    if _upstream_torch is None:
        return
    for dtype in _REDUCED_FLOATS:
        for flat in _rounding_operands(256):
            got = _flat_values(_E2EBackend("shim").t(flat, (256,), dtype))
            want = _flat_values(_E2EBackend("upstream").t(flat, (256,), dtype))
            assert got == want, dtype


def test_reduced_float_narrowing_is_round_to_nearest_even_not_truncation():
    """Names the wrong rule, so the test fails if the shim adopts it.

    Each pair below is exactly halfway between two representable bfloat16
    values, which is where round-to-nearest-even and truncation disagree by
    construction. `truncated` is what this shim returned before docs/BF16.md
    -- it is asserted *against*, so that a regression to the old behaviour
    fails here with the rule named rather than as a drift somewhere in a
    model.

    Upstream is still consulted for the expected value; the literals are a
    second opinion, not the source of truth (`tools/golden/cases.py`'s `_pair`
    note is the standing rule).

    **The length matters and is the reason this went unseen.** Measured: the
    same four pairs narrow correctly at 31 elements and incorrectly at 32, so
    the wrong rule lived on a vectorised path that only ran on tensors bigger
    than any case in `tools/golden/cases.py`. Both lengths are asserted here;
    testing only the short one is testing the path that was never broken.
    """
    # (a, b, round-to-nearest-even sum, truncated sum)
    ties = [
        (-0.69140625, -0.228515625, -0.921875, -0.91796875),
        (0.189453125, 0.030761719, 0.220703125, 0.2197265625),
        (-0.48046875, 0.020019531, -0.4609375, -0.458984375),
        (0.85546875, -0.000652313, 0.85546875, 0.8515625),
    ]

    def sums(backend, reps):
        rows = ties * reps
        n = len(rows)
        return _flat_values(
            backend.op(
                "aten.add.Tensor",
                backend.t([r[0] for r in rows], (n,), "bfloat16"),
                backend.t([r[1] for r in rows], (n,), "bfloat16"),
            )
        )

    shim = _E2EBackend("shim")
    for reps in (1, 16):  # 4 elements (scalar path) and 64 (vectorised path)
        rows = ties * reps
        got = sums(shim, reps)
        assert got == [r[2] for r in rows], (reps, got[:8])
        assert got != [r[3] for r in rows], (
            f"shim truncates instead of rounding to nearest even at "
            f"{len(rows)} elements"
        )
        if _upstream_torch is not None:
            assert got == sums(_E2EBackend("upstream"), reps), reps


# --- the fused reduced-float kernels (docs/DTYPE.md) ------------------------
#
# `rust/torch_c/src/reduced.rs` replaced two things the ops above used to do
# through candle: the `{float16,bfloat16} <-> float32` conversions, and the
# three-pass shape of widen / compute / narrow. Both are claimed to compute the
# *same function* as before, only faster, and that claim is what the tests
# below are for -- the speed is measured elsewhere, in docs/DTYPE.md.
#
# The interesting failure is not "wrong everywhere". A vectorised kernel that
# handles eight elements per iteration with a scalar tail has three regions and
# two boundaries, and a fault in the tail is invisible at any length that is a
# multiple of eight. docs/BF16.md §2.3 records this repository losing a whole
# class of defect to exactly that -- the wrong rounding rule lived on a path no
# case in the golden harness was long enough to reach. So the lengths here
# straddle the vector width deliberately rather than being round.

# 7 is tail-only, 8 is body-only, 9..15 are body plus tail, and 1000/1001 are
# long enough that a kernel carrying per-block state has room to lose it.
_REDUCED_LENGTHS = (1, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 1000, 1001)


def _f32_from_bits(bits):
    """A float32 by its bit pattern, widened to the Python float that carries
    it. The payload survives the round trip -- checked on this host -- for
    everything except a signalling NaN, which the hardware quiets."""
    import struct

    return struct.unpack("<f", struct.pack("<I", bits))[0]


def test_fused_reduced_float_arithmetic_matches_upstream_at_every_vector_boundary():
    """add/sub/mul/div in both reduced floats, exactly, at 13 lengths.

    All four ops rather than just `add` because the fused kernel is one
    function with the arithmetic as a parameter: a fault in the widening or the
    narrowing shows up in all four, and a fault in the dispatch shows up in
    exactly one. Telling those apart is worth the extra assertions.
    """
    if _upstream_torch is None:
        return
    shim, up = _E2EBackend("shim"), _E2EBackend("upstream")
    for dtype in _REDUCED_FLOATS:
        for n in _REDUCED_LENGTHS:
            a, b = _rounding_operands(n)
            # Nothing near zero in the divisor: division by zero is a different
            # question from rounding and the golden cases already fix it.
            b = [v if abs(v) > 1e-3 else 0.5 for v in b]
            for op in ("add.Tensor", "sub.Tensor", "mul.Tensor", "div.Tensor"):
                got = _flat_values(
                    shim.op("aten." + op, shim.t(a, (n,), dtype), shim.t(b, (n,), dtype))
                )
                want = _flat_values(
                    up.op("aten." + op, up.t(a, (n,), dtype), up.t(b, (n,), dtype))
                )
                wrong = [i for i, (g, w) in enumerate(zip(got, want)) if g != w]
                assert not wrong, (
                    f"{dtype} {op} n={n}: {len(wrong)}/{n} differ, first at "
                    f"{wrong[0]} (shim {got[wrong[0]]!r} vs upstream "
                    f"{want[wrong[0]]!r})"
                )


def test_fused_reduced_float_arithmetic_matches_upstream_when_an_operand_broadcasts():
    """The rotary shape: a `(1, k)` table against an `(m, k)` activation.

    The fused kernel materialises the broadcast operand rather than declining
    it, so this is the arm where it could produce a correctly-shaped wrong
    answer by pairing the wrong elements. A shape check would pass that.
    """
    if _upstream_torch is None:
        return
    shim, up = _E2EBackend("shim"), _E2EBackend("upstream")
    for dtype in _REDUCED_FLOATS:
        for (m, k) in ((3, 40), (5, 8), (2, 1001)):
            a, b = _rounding_operands(m * k)
            row = b[:k]
            for op in ("add.Tensor", "mul.Tensor"):
                got = _flat_values(
                    shim.op("aten." + op, shim.t(a, (m, k), dtype), shim.t(row, (1, k), dtype))
                )
                want = _flat_values(
                    up.op("aten." + op, up.t(a, (m, k), dtype), up.t(row, (1, k), dtype))
                )
                assert got == want, (dtype, op, m, k)


def test_reduced_float_arithmetic_matches_upstream_on_non_contiguous_operands():
    """A transposed operand does not reach the fused kernel, and must not have
    to. This pins that the fallback still agrees.

    A transpose rather than a slice, because it leaves the layout
    non-contiguous without changing the element count -- so a fast path that
    ignored strides would read the right *number* of elements in the wrong
    order and produce a wrong answer of the right shape.
    """
    if _upstream_torch is None:
        return
    shim, up = _E2EBackend("shim"), _E2EBackend("upstream")
    for dtype in _REDUCED_FLOATS:
        a, b = _rounding_operands(12 * 7)
        xs = shim.t(a, (12, 7), dtype)
        ys = shim.op("aten.t.default", shim.t(b, (7, 12), dtype))
        xu = up.t(a, (12, 7), dtype)
        yu = up.op("aten.t.default", up.t(b, (7, 12), dtype))
        for op, swap in (("add.Tensor", False), ("mul.Tensor", True)):
            l, r = (ys, xs) if swap else (xs, ys)
            lu, ru = (yu, xu) if swap else (xu, yu)
            got = _flat_values(shim.op("aten." + op, l, r))
            want = _flat_values(up.op("aten." + op, lu, ru))
            assert got == want, (dtype, op)


def test_reduced_float_conversion_carries_the_values_no_shift_would():
    """`_to_copy` both ways, over the values a plausible converter gets wrong.

    The `bfloat16` narrowing in `reduced.rs` is integer arithmetic with a select
    for NaN, and the select is not decoration: without it the rounding add turns
    the signalling NaN `0x7f80_0001` into `0x7f80`, which reads back as
    **infinity**. Subnormals, signed zero and the overflow edge are here for the
    same reason -- they are where a converter that looks right stops agreeing.
    """
    if _upstream_torch is None:
        return
    specials = [
        float("inf"), float("-inf"), 0.0, -0.0, 1.0, -1.0,
        6.103515625e-05,        # smallest normal float16
        5.960464477539063e-08,  # smallest subnormal float16
        1e-45,                  # float32 subnormal: underflows both
        65504.0, -65504.0,      # largest finite float16
        65536.0,                # overflows float16 to inf, ordinary in bfloat16
        3.3895313892515355e+38, # near the float32 top, finite in bfloat16
        0.30000001192092896,
        -0.69140625,
    ]
    # A length that is not a multiple of eight, so the scalar tail converts too.
    flat = specials + [i * 0.007 - 0.5 for i in range(13)]
    n = len(flat)
    shim, up = _E2EBackend("shim"), _E2EBackend("upstream")
    for dtype in _REDUCED_FLOATS:
        got = _flat_values(shim.t(flat, (n,), dtype))
        want = _flat_values(up.t(flat, (n,), dtype))
        assert repr(got) == repr(want), (dtype, "narrowing", got, want)
        back_got = _flat_values(
            shim.op("aten._to_copy.default", shim.t(flat, (n,), dtype), dtype=_C.float32)
        )
        back_want = _flat_values(
            up.op(
                "aten._to_copy.default",
                up.t(flat, (n,), dtype),
                dtype=_upstream_torch.float32,
            )
        )
        assert repr(back_got) == repr(back_want), (dtype, "widening")
        # NaN separately: it does not compare equal to itself, so the checks
        # above would pass whatever came back for it. The payloads are the
        # point. `0x7fff_ffff` is a NaN whose mantissa is all ones, and the
        # rounding add alone carries into the exponent and turns it into
        # `0x8000` -- **negative zero**, silently, for a value that was not a
        # number. The quiet default `0x7fc0_0000` does *not* expose that, so a
        # NaN probe built from `float("nan")` passes either way; measured.
        nans = [
            _f32_from_bits(0x7FFFFFFF),
            _f32_from_bits(0xFFFFFFFF),
            _f32_from_bits(0x7F800001),  # signalling; the host quiets it
            float("nan"),
        ] * 3  # 12 elements, so the vectorised body runs and the tail does not
        narrowed = shim.op(
            "aten._to_copy.default",
            shim.t(nans, (12,), "float32"),
            dtype=getattr(_C, dtype),
        )
        for where, side in (
            ("narrowed", _flat_values(narrowed)),
            (
                "widened",
                _flat_values(shim.op("aten._to_copy.default", narrowed, dtype=_C.float32)),
            ),
        ):
            bad = [(i, v) for i, v in enumerate(side) if v == v]
            assert not bad, f"{dtype} {where}: NaN came back as {bad}"

# --- the flash-attention kernel, bit for bit --------------------------------
#
# `aten::_scaled_dot_product_flash_attention_for_cpu` is a *blocked* kernel
# with an online softmax, and for `bfloat16`/`float16` inputs every step of it
# happens in portable code -- upstream reaches its own kernel rather than a
# BLAS. `rust/torch_c/src/flash.rs` reproduces that arrangement, and
# docs/SDPA.md is the measurement. These tests assert the consequence: **exact**
# agreement, with no tolerance.
#
# A tolerance here would not be a check. docs/SDPA.md §3 has the numbers: the
# formulation this replaced sat inside `tools/golden/dtypes.py`'s bfloat16
# tolerance (6e-2) on every element while disagreeing with upstream on 32% of
# them, and the golden harness passed it for months. One bfloat16 ulp is
# 1/256 of the value; the tolerance is fifteen times that.
#
# **That formulation is still the default, so these tests turn the kernel on.**
# It costs 20x (docs/SDPA.md §12), so `sdpa` reaches it only when asked. Every
# test below therefore runs inside `_sdpa_reference()`, which flips
# `_C._shim_sdpa_reference` and flips it back -- the switch is process-global,
# and leaking it would make the six hundred cases that follow this section slow
# for no reason. `test_sdpa_reference_switch_is_off_by_default_and_restores`
# is what keeps that wiring honest: without it, a switch stuck on would make
# every test here pass while costing the default path 20x, and a switch stuck
# off would make them all fail loudly, which is the harmless direction.
#
# The shapes below are not arbitrary. Each crosses a boundary inside the
# kernel that a naive reading does not have:
#
#   (26, 26)   SmolLM2-135M's own prefill, 9 query heads over 3 KV heads
#   (33, 33)   one past the 32-row query block, so a second block runs
#   (70, 70)   three query blocks, and 70 % 8 == 6 -- the mask's vector body
#              stops at 64, not 68 (see the second test)
#   (40, 600)  past the 512-column key split, so the online rescale runs
_SDPA_OP = "aten._scaled_dot_product_flash_attention_for_cpu.default"

_SDPA_SHAPES = [
    # (batch, query heads, kv heads, q_len, kv_len, head_dim)
    (1, 2, 2, 8, 8, 16),
    (1, 9, 3, 26, 26, 64),
    (1, 2, 2, 33, 33, 8),
    (1, 2, 2, 70, 70, 8),
    (1, 1, 1, 40, 600, 8),
    (1, 1, 1, 2, 5, 4),
]


class _sdpa_reference:
    """Turns the exact kernel on for the block, and puts it back afterwards.

    Restores the *previous* value rather than `False`, so that running the
    suite under `BW_SDPA_REFERENCE=1` -- which is how the kernel gets exercised
    end to end, past this section -- is not silently undone here.
    """

    def __enter__(self):
        self.was = _C._shim_sdpa_reference(True)
        return self

    def __exit__(self, *exc):
        _C._shim_sdpa_reference(self.was)
        return False


def _sdpa_det(n, seed):
    """`n` values of the form k/128.

    Every one is exactly representable in bfloat16, float16, float32 and
    float64 alike, so `_tensor_from_flat` and `torch.tensor` cannot disagree
    about the *inputs* and be misread as a disagreement about the kernel.
    `test_reduced_float_operands_survive_construction` makes the same premise
    explicit for the section above.
    """
    out, state = [], seed * 2654435761 % 2147483647
    for _ in range(n):
        state = (state * 1103515245 + 12345) % 2147483648
        out.append((state % 321 - 160) / 128.0)
    return out


def _sdpa_call(b, batch, heads, kv_heads, q_len, kv_len, head_dim, dtype, causal, mask):
    q = b.t(_sdpa_det(batch * heads * q_len * head_dim, 1),
            (batch, heads, q_len, head_dim), dtype)
    k = b.t(_sdpa_det(batch * kv_heads * kv_len * head_dim, 2),
            (batch, kv_heads, kv_len, head_dim), dtype)
    v = b.t(_sdpa_det(batch * kv_heads * kv_len * head_dim, 3),
            (batch, kv_heads, kv_len, head_dim), dtype)
    kw = {}
    if mask:
        flat = _sdpa_det(batch * heads * q_len * kv_len, 4)
        # A wholly masked-out column, which is the shape a padding mask has and
        # the reason the kernel writes zeros instead of `exp(-inf - -inf)`.
        for i in range(0, len(flat), 7):
            flat[i] = float("-inf")
        kw["attn_mask"] = b.t(flat, (batch, heads, q_len, kv_len), dtype)
    # The one funnel all three tests below go through, so the switch is flipped
    # here rather than in each of them. It wraps the *call* and nothing else:
    # building the tensors does not read it, and the upstream backend does not
    # have it. `b.kind == "upstream"` still enters the block, harmlessly -- the
    # alternative is a branch that makes the two backends take different code
    # here, which is the one thing this helper exists to avoid.
    with _sdpa_reference():
        return b.op(_SDPA_OP, q, k, v, 0.0, causal, **kw)


def test_amax_at_a_real_score_row_width_agrees_with_upstream_exactly():
    """`amax` at the row width attention actually produces, not a toy one.

    The golden suite's rows are 3 to 40 elements wide. That covers the
    accumulator lanes and the remainder, and it cannot cover the regime this
    kernel was written for: a 512-wide row is 32 full chunks, and a bug in the
    lane-combining step that a 40-element row happens to survive (because the
    maximum sits in a lane the short combine reaches first) does not survive
    32 of them. So this walks the maximum through every one of the 512
    positions and demands the answer exactly -- a maximum has no arithmetic in
    it, so `==` is the right comparison and a tolerance would be hiding
    something.

    docs/SEQLEN.md §7 is why this shape and not another: `[1, 9, S, S]` with
    `S=512` is what a SmolLM2-135M prefill hands its softmax, and it is the
    shape at which candle's index-tracking reduction measured 56x upstream.
    """
    n = 512
    for at in list(range(0, n, 37)) + [n - 1, 0, 1]:
        flat = [float((i * 7) % 101) - 50.0 for i in range(n)]
        flat[at] = 1234.5
        row = _C._tensor_from_flat(flat, [1, n], dtype=_C.float32)
        got = _C._aten_dispatch("aten.amax.default", row, [1], False)
        assert got.tolist() == [1234.5], f"maximum at position {at}: {got.tolist()}"
        assert got.dtype == _C.float32
        assert list(got.shape) == [1]
    # And the two-dimensional form, so the reduction runs 9 rows at a time the
    # way SDPA drives it rather than once.
    rows = 9
    flat = []
    for r in range(rows):
        flat.extend(float((i * 13 + r) % 199) - 99.0 for i in range(n))
        flat[r * n + (r * 41) % n] = 500.0 + r
    t = _C._tensor_from_flat(flat, [rows, n], dtype=_C.float32)
    got = _C._aten_dispatch("aten.amax.default", t, [1], True)
    assert list(got.shape) == [rows, 1]
    assert got.tolist() == [[500.0 + r] for r in range(rows)], got.tolist()


def test_amax_propagates_nan_where_candles_own_reduction_drops_it():
    """The divergence this kernel exists to *not* inherit.

    `aten.max.default` answered `3.0` for `max([3, nan, 1])` before
    docs/E2E_REAL.md, because candle's reduction compares with `x < y` and
    every comparison against a NaN is false, so a NaN that is not the first
    element is skipped. `max.other` had the same hole in its second operand
    (docs/SPELLINGS.md). Two ops, one predicate, so the third op to use that
    predicate would have had it too.

    `amax` does not use it. This checks the NaN survives from a position past
    the first accumulator chunk -- the position a kernel gets *wrong* by
    accident, since a NaN in element 0 seeds every lane and is preserved even
    by a wrong reduction.
    """
    n = 200
    for at in (0, 1, 17, 63, 199):
        flat = [float(i) for i in range(n)]
        flat[at] = float("nan")
        t = _C._tensor_from_flat(flat, [1, n], dtype=_C.float32)
        got = _C._aten_dispatch("aten.amax.default", t, [1], False).tolist()[0]
        assert math.isnan(got), f"a NaN at position {at} of {n} came back as {got}"
    # This used to assert that `aten.max.dim` on the same data still answered a
    # *number*, as the live proof that candle's predicate had not changed. It
    # answers NaN now, and the assertion did its job: it failed the moment
    # docs/TRIL.md §3 fixed `max.dim`, which is exactly the notification it was
    # written to give.
    #
    # **The question that failure asks -- "is amax's own NaN pass now
    # redundant?" -- is answered no, and the reason is where the callers are.**
    # `max.dim`'s correction lives in `aten.rs`, above the aten dispatch
    # boundary. `sdpa_flash_cpu` calls `crate::tensor::amax_keepdim` *directly*
    # in Rust and never crosses that boundary, so an aten-level correction does
    # not reach it. candle's own reduction is unchanged and still drops the
    # NaN; that statement now lives where it can be made against candle
    # directly rather than through an aten op that has stopped exhibiting it --
    # `tensor.rs::candle_drops_the_nan_this_kernel_keeps`, a `cargo test`.
    flat = [float(i) for i in range(n)]
    flat[17] = float("nan")
    t = _C._tensor_from_flat(flat, [1, n], dtype=_C.float32)
    pair = _C._aten_dispatch("aten.max.dim", t, 1, False)
    values, indices = pair[0].tolist()[0], pair[1].tolist()[0]
    assert math.isnan(values), (
        f"aten.max.dim dropped a NaN again -- docs/TRIL.md §3 fixed this; got {values}"
    )
    assert indices == 17, (
        f"aten.max.dim must report the index of the first NaN, not of the "
        f"maximum among the rest; got {indices}"
    )


def test_a_fully_masked_attention_row_reduces_to_negative_infinity():
    """Every element `-inf` is a real shape, not a pathological one.

    It is what a causal mask plus padding produces, and it is the row where a
    reduction seeded with a neutral element rather than with the data answers
    something finite. `-inf` is the right answer and it is what upstream gives
    (measured). The softmax that follows then produces NaN for that row, which
    is `_softmax`'s documented behaviour and `_safe_softmax`'s zero -- neither
    of which this op is allowed to pre-empt by answering something else here.
    """
    for n in (1, 5, 16, 17, 512):
        t = _C._tensor_from_flat([float("-inf")] * n, [1, n], dtype=_C.float32)
        got = _C._aten_dispatch("aten.amax.default", t, [1], False).tolist()[0]
        assert got == float("-inf"), f"n={n}: an all -inf row gave {got}"
    # One finite element among 511 `-inf` still wins, from the last position.
    flat = [float("-inf")] * 512
    flat[511] = -12.5
    t = _C._tensor_from_flat(flat, [1, 512], dtype=_C.float32)
    assert _C._aten_dispatch("aten.amax.default", t, [1], False).tolist() == [-12.5]


def test_amax_now_has_both_python_spellings_and_they_reach_the_kernel():
    """`torch.amax` and `Tensor.amax` resolve, and give the kernel's answer.

    **This test previously asserted the opposite**, by name:
    `test_amax_has_no_python_spelling_yet_and_says_so_by_name`. The kernel
    landed in docs/SEQLEN.md §7 and the two Python names did not, so that round
    wrote down the absence as an executable claim -- "when the table entry
    lands, this test fails, which is the notification wanted". It did fail, on
    the first run after the entries went into `src/overloads.json` and
    `src/methods.json` (docs/TRIL.md §2), and this is the update it asked for.

    That mechanism is the whole reason the gap did not go another round
    unnoticed. The golden harness dispatches by key and is structurally blind
    to a missing name -- `aten.amax.default` had been compared against upstream
    since the day it landed, with 120 cases, while both Python spellings
    raised. A kernel-level suite cannot see this class of gap; only a test
    written *through the name* can.

    So this checks three routes to one kernel and requires them to agree:
    `torch.ops.aten.amax.default`, `torch.amax`, and `Tensor.amax`.
    """
    a = _C._tensor_from_flat([1.0, 5.0, 2.0, 9.0], [2, 2], dtype=_C.float32)
    assert _C._aten_dispatch("aten.amax.default", a, [1], False).tolist() == [5.0, 9.0]
    if not _ckpt_shim_available():
        return
    got = _amax_spelling_fixture()
    assert got["aten_key"] == [5.0, 9.0], got
    for label in ("torch.amax", "Tensor.amax"):
        answer = got[label]
        assert answer == [5.0, 9.0], (
            f"{label} did not reach the kernel: {answer!r} -- the entries are in "
            f"overloads.json / methods.json, see docs/TRIL.md §2"
        )
    # The arguments the entry has to carry, not just the two-positional form:
    # `keepdim`, a negative `dim`, the schema's `dim=[]` default (which reduces
    # *everything*, unlike `sum`'s reading of an empty list), and keyword
    # spellings. A table entry with the wrong signature resolves the simple
    # call and fails these.
    assert got["keepdim"] == [[5.0], [9.0]], got["keepdim"]
    assert got["negative_dim"] == [5.0, 9.0], got["negative_dim"]
    assert got["no_dim"] == 9.0, got["no_dim"]
    assert got["by_keyword"] == [5.0, 9.0], got["by_keyword"]
    assert got["member_keyword"] == [[5.0], [9.0]], got["member_keyword"]


_AMAX_SPELLING_SCRIPT = r"""
import json, sys
import torch

out = {}
t = torch.tensor([[1.0, 5.0], [2.0, 9.0]])
out["aten_key"] = torch.ops.aten.amax.default(t, [1], False).tolist()
for label, call in (("torch.amax", lambda x: torch.amax(x, 1)),
                    ("Tensor.amax", lambda x: x.amax(1)),
                    ("keepdim", lambda x: torch.amax(x, 1, True)),
                    ("negative_dim", lambda x: torch.amax(x, -1)),
                    ("no_dim", lambda x: torch.amax(x)),
                    ("by_keyword", lambda x: torch.amax(x, dim=1, keepdim=False)),
                    ("member_keyword", lambda x: x.amax(dim=1, keepdim=True))):
    try:
        out[label] = call(t).tolist()
    except NotImplementedError as e:
        out[label] = "NotImplementedError: %s" % e
json.dump(out, sys.stdout)
"""


@functools.lru_cache(maxsize=1)
def _amax_spelling_fixture():
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _AMAX_SPELLING_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"amax-spelling subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def test_sdpa_reduced_float_matches_upstream_to_the_last_bit():
    """Both halves of the pair, every shape, no tolerance.

    The logsumexp half is checked as carefully as the output half and is not
    redundant: it comes back in `float32` even for a `bfloat16` call, so it is
    the only place a one-ulp disagreement in the row sum is *visible* --
    narrowing the output to bfloat16 hides it. The defect docs/SDPA.md §4.4
    describes showed up in exactly that asymmetry and nowhere else.
    """
    if _upstream_torch is None:
        return  # no upstream torch in this interpreter -- see docs/E2E.md
    shim, upstream = _E2EBackend("shim"), _E2EBackend("upstream")
    for dtype in _REDUCED_FLOATS:
        for shape in _SDPA_SHAPES:
            for causal in (False, True):
                for mask in (False, True):
                    args = (*shape, dtype, causal, mask)
                    got_pair = _sdpa_call(shim, *args)
                    want_pair = _sdpa_call(upstream, *args)
                    for half, label in ((0, "output"), (1, "logsumexp")):
                        got = _flat_values(got_pair[half])
                        want = _flat_values(want_pair[half])
                        assert len(got) == len(want), (dtype, shape, label)
                        wrong = [i for i, (g, w) in enumerate(zip(got, want)) if g != w]
                        assert not wrong, (
                            f"{dtype} {shape} causal={causal} mask={mask} {label}: "
                            f"{len(wrong)}/{len(want)} elements differ from upstream; "
                            f"first at {wrong[0]} (shim {got[wrong[0]]!r} vs "
                            f"upstream {want[wrong[0]]!r})"
                        )


def test_sdpa_mask_body_strides_by_the_mask_dtype_not_the_accumulator():
    """Names the wrong rule, so a regression to it fails here with the cause.

    Upstream fuses `qk * scale + mask` in a loop that strides by
    `Vectorized<mask_t>::size()`, and `mask_t` is the *input* dtype. For a
    reduced float that is eight lanes, not the accumulator's four, so a
    70-column block leaves six columns to the scalar remainder -- and the
    remainder is one C statement, hence fused, where the body is not.

    Reading that stride as four is a difference of two columns out of seventy.
    It moved **one element in 226136** (docs/SDPA.md §4.4), in the logsumexp
    and not the output, because the shift cancels between the row maximum and
    the log of the row sum. Nothing shaped like a tolerance can see it, and
    the general test above only sees it because it uses none.
    """
    if _upstream_torch is None:
        return
    shape = (1, 2, 2, 70, 70, 32)
    assert shape[4] % 8 == 6, "the point of this shape is the six-column remainder"
    for dtype in _REDUCED_FLOATS:
        got = _flat_values(_sdpa_call(_E2EBackend("shim"), *shape, dtype, False, True)[1])
        want = _flat_values(
            _sdpa_call(_E2EBackend("upstream"), *shape, dtype, False, True)[1]
        )
        wrong = [i for i, (g, w) in enumerate(zip(got, want)) if g != w]
        assert not wrong, (
            f"{dtype} logsumexp differs on {len(wrong)}/{len(want)} rows; the mask "
            f"body is probably striding by four lanes instead of eight "
            f"(first row {wrong[0]}: shim {got[wrong[0]]!r} vs upstream "
            f"{want[wrong[0]]!r})"
        )


def test_sdpa_wide_float_is_close_but_not_promised_exact():
    """The other side of the claim, pinned so it cannot quietly become one.

    For `float32` upstream hands both matrix products to the platform BLAS
    (Accelerate here), whose summation order is not portable, so this kernel
    is *not* bit-identical there and docs/SDPA.md §5 says so. The bound below
    is the measured worst case over the same shapes, not a guess; asserting a
    bound rather than equality is what keeps the two claims apart.
    """
    if _upstream_torch is None:
        return
    shim, upstream = _E2EBackend("shim"), _E2EBackend("upstream")
    bounds = {"float32": 1e-6, "float64": 1e-14}
    for dtype, bound in bounds.items():
        worst = 0.0
        for shape in _SDPA_SHAPES:
            for causal in (False, True):
                args = (*shape, dtype, causal, False)
                got_pair = _sdpa_call(shim, *args)
                want_pair = _sdpa_call(upstream, *args)
                for half in (0, 1):
                    for g, w in zip(_flat_values(got_pair[half]), _flat_values(want_pair[half])):
                        if g == w or math.isinf(g) or math.isinf(w):
                            continue
                        worst = max(worst, abs(g - w))
        assert worst <= bound, f"{dtype} sdpa drifted to {worst:.4g}, over {bound:.4g}"


def test_sdpa_reference_switch_is_off_by_default_and_restores():
    """The switch is the claim; this is the test that can fail if it breaks.

    Three separate things, because three separate ways of getting it wrong all
    end with the suite green:

      * **off by default.** If the switch defaulted on, every test above would
        pass while every forward pass in the library paid 20x. Nothing else
        here would notice -- the two paths agree to within a tolerance, which
        is precisely why this cost went unnoticed for as long as it did.
      * **the two paths actually differ.** A switch wired to nothing also makes
        the tests above pass, because they would then be measuring the path
        that is already bit-exact... or not measuring anything. So this asks
        for a shape where the default answer and the reference answer are
        *different bits*, and fails if they are the same. That shape is not
        exotic: it is the first entry of `_SDPA_SHAPES`, where 93 of 256
        bfloat16 elements move.
      * **it restores.** `_sdpa_call` flips the switch on every call, so a
        context manager that failed to put it back would leave the rest of
        this file -- and, under `run.sh`, everything after it -- on the slow
        path.
    """
    # `BW_SDPA_REFERENCE` is the env spelling of the same switch, so a suite run
    # under it has legitimately changed the default. Everything else here still
    # holds; only the claim about the default is that run's to make.
    asked_by_env = os.environ.get("BW_SDPA_REFERENCE") not in (None, "", "0")
    resting = _C._shim_sdpa_reference()
    assert resting is asked_by_env, (
        f"the reference kernel is {'on' if resting else 'off'} at rest with "
        f"BW_SDPA_REFERENCE={os.environ.get('BW_SDPA_REFERENCE')!r}; it costs 20x "
        "(docs/SDPA.md §12) and must be reached only when asked for"
    )

    shim = _E2EBackend("shim")
    q = shim.t(_sdpa_det(2 * 8 * 16, 1), (1, 2, 8, 16), "bfloat16")
    k = shim.t(_sdpa_det(2 * 8 * 16, 2), (1, 2, 8, 16), "bfloat16")
    v = shim.t(_sdpa_det(2 * 8 * 16, 3), (1, 2, 8, 16), "bfloat16")

    def run():
        return _flat_values(_C._aten_dispatch(_SDPA_OP, q, k, v, 0.0, False)[0])

    _C._shim_sdpa_reference(False)
    fast = run()
    with _sdpa_reference() as ctx:
        assert ctx.was is False
        assert _C._shim_sdpa_reference() is True
        exact = run()
    assert _C._shim_sdpa_reference() is False, "the switch did not come back off"
    _C._shim_sdpa_reference(resting)

    assert len(fast) == len(exact)
    differing = [i for i, (a, b) in enumerate(zip(fast, exact)) if a != b]
    assert differing, (
        "the default and reference paths gave identical bits, so the switch is "
        "not selecting anything -- the tests above are then vacuous"
    )


# --- block quantisation (docs/QUANT2.md) ------------------------------------
#
# **The section that had to build its own judge.** Everything above decides by
# bit equality against upstream, and quantisation cannot be decided that way:
# it is lossy on purpose, so "differs from upstream" is its normal state and a
# tolerance loose enough to accept a correct Q4K weight is loose enough to
# accept a broken one. docs/QUANT2.md §2 is the argument; these are the checks.
#
# The axis has three layers and only the third has a tolerance in it:
#
#   1. **The bytes.** `pytests/ggml_ref.py` reimplements the Q8_0 and Q4_0
#      quantisers from the format and the blob is compared byte for byte.
#   2. **The reconstruction.** The same file reimplements the dequantisers for
#      those two plus Q4K, and the `float32` output is compared bit for bit.
#   3. **The loss.** What is genuinely lossy is held to a bound *derived from
#      the format*, with a floor as well as a ceiling -- because a "quantiser"
#      that returned its input would pass any ceiling.
#
# And the matmul is checked exactly, which is the part that looked impossible:
# on operands the format represents without loss, `_quantized_linear` through a
# Q8_0 weight is **bit identical** to a dense `linear`. See
# `test_a_quantised_matmul_is_exact_on_operands_the_format_holds_exactly`.

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ggml_ref  # noqa: E402


_QUANT_SEED = 20260828


def _lcg(seed):
    """A generator this file owns, so the fixtures do not depend on the shim's
    RNG being right -- these tests are about quantisation, and borrowing
    `_shim_manual_seed` would couple a failure here to a failure there."""
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    return nxt


def _gauss_flat(n, seed=_QUANT_SEED):
    """Box-Muller over the LCG. Gaussian rather than uniform because it is the
    input quantisation is *worst* at -- docs/QUANT.md §7 measured 7.5% relative
    RMS on random Gaussian weights and called it an upper bound for that
    reason."""
    nxt = _lcg(seed)
    out = []
    while len(out) < n:
        u1 = max(nxt(), 1e-12)
        u2 = nxt()
        r = math.sqrt(-2.0 * math.log(u1))
        out.append(r * math.cos(2 * math.pi * u2))
        out.append(r * math.sin(2 * math.pi * u2))
    return out[:n]


def _exactly_representable(rows, cols, amax=127, seed=_QUANT_SEED):
    """Values Q8_0 stores with **no** loss, which is what makes the matmul
    check exact rather than approximate.

    Two conditions, and both are needed:

      * every entry is a whole number, so `round(x/d)` is `x/d`;
      * every 32-element block has absmax exactly `amax = 127`, so the scale is
        `127/127 = 1.0` -- which is representable in `f16`, so storing it costs
        nothing either.

    Then `dequantize(quantize(w)) == w` bit for bit, the dot products are sums
    of products of integers below 127, and every partial sum stays under
    `2**24` where `float32` is exact on integers. So **every** accumulation
    order gives the same bits, and the comparison does not depend on which BLAS
    ran or how it blocked the loop.
    """
    nxt = _lcg(seed)
    out = []
    for _ in range(rows):
        row = [float(int(nxt() * (2 * amax - 2)) - (amax - 1)) for _ in range(cols)]
        for b in range(cols // 32):
            # One entry per block pinned to the extreme, so the scale is 1.0.
            row[b * 32 + int(nxt() * 31)] = float(amax if nxt() < 0.5 else -amax)
        out.append(row)
    return out


def _quant_fixture(rows, cols, fmt, seed=_QUANT_SEED):
    flat = _gauss_flat(rows * cols, seed)
    dense = _C._tensor_from_flat(flat, [rows, cols], dtype=_C.float32)
    q = _C._quantize(dense, fmt)
    return flat, dense, q


def test_quantised_blobs_match_an_independent_reimplementation_byte_for_byte():
    """Layer 1: the bytes, not the numbers.

    `ggml_ref.quantize_q8_0` and `quantize_q4_0` are written from the format --
    block size, scale derivation, rounding mode, nibble interleave -- and the
    result is compared to `_C._quantized_blob()` with `==` on `bytes`. There is
    no tolerance to widen and no shape of near-miss that passes.

    This is the strongest statement available about a lossy operation: it does
    not say the answer is close, it says **the function is the same function**.
    Two of the three easy mistakes it catches are in Q4_0 alone -- the scale is
    `max / -8` and keeps the sign of the largest-magnitude element, and element
    `j` and element `j + 16` share a byte rather than `j` and `j + 1`.

    What it cannot catch is in `ggml_ref`'s own docstring: a misreading shared
    by both implementations.
    """
    for fmt in sorted(ggml_ref.QUANTIZERS):
        for rows, cols in ((4, 32), (3, 256), (2, 512), (7, 64)):
            flat, _, q = _quant_fixture(rows, cols, fmt)
            got = _C._quantized_blob(q)
            want = ggml_ref.QUANTIZERS[fmt](flat)
            assert len(got) == len(want), (fmt, rows, cols, len(got), len(want))
            assert got == want, (
                fmt,
                (rows, cols),
                [i for i, (a, b) in enumerate(zip(got, want)) if a != b][:8],
            )


def test_quantised_blob_sizes_are_the_format_and_not_a_coincidence():
    """`_quantized_nbytes` against the format's own arithmetic.

    `blocks * type_size`, with `type_size` written out in `ggml_ref.TYPE_SIZE`
    from the struct definition rather than read back from candle. It is the
    check that a format silently changing block layout under a version bump
    fails loudly here instead of shifting every number in QUANT2.md by a few
    percent.
    """
    for fmt, type_size in sorted(ggml_ref.TYPE_SIZE.items()):
        block = ggml_ref.BLOCK_SIZE[fmt]
        rows, cols = 3, block * 2
        _, _, q = _quant_fixture(rows, cols, fmt)
        want = (rows * cols // block) * type_size
        assert _C._quantized_nbytes(q) == want, (fmt, _C._quantized_nbytes(q), want)
        assert len(_C._quantized_blob(q)) == want, fmt
        # And it is actually smaller than the float32 it replaced, which is the
        # entire reason for the exercise. `f32` as a "format" is the control
        # and is excluded by not being in TYPE_SIZE.
        assert want < rows * cols * 4, fmt


def test_dequantisation_matches_the_reference_reconstruction_bit_for_bit():
    """Layer 2: the reader, over a blob neither side is free to choose.

    The blob comes from `_C._quantized_blob`, so both implementations
    reconstruct *the same bytes* and the comparison isolates the reader from
    the writer. Q4K is here even though its quantiser is not: the k-quant
    *writer* is an iterative least-squares search and transcribing a search
    would not produce an independent answer, but the *reader* is a pure format
    -- two f16 super-scales, eight 6-bit sub-scales, eight 6-bit minima packed
    across twelve bytes -- and that is transcribed and checked.

    Bit for bit, through `repr`, so a `-0.0` that came back as `0.0` fails.
    """
    for fmt in ("q8_0", "q4_0", "q4_k"):
        block = ggml_ref.BLOCK_SIZE[fmt]
        for rows, cols in ((3, block), (2, block * 3), (5, block * 2)):
            _, _, q = _quant_fixture(rows, cols, fmt)
            blob = _C._quantized_blob(q)
            got = _flat_values(_C._dequantize(q))
            want = ggml_ref.DEQUANTIZERS[fmt](blob, rows * cols)
            assert len(got) == len(want), (fmt, len(got), len(want))
            assert repr(got) == repr(want), (
                fmt,
                (rows, cols),
                [(i, a, b) for i, (a, b) in enumerate(zip(got, want)) if a != b][:6],
            )


def test_the_quantisation_axis_fails_when_the_reference_is_perturbed():
    """**The check on the checks.** CLAUDE.md §5.5: a verification that cannot
    fail is not a verification.

    Six faults, each shaped like a mistake somebody would actually make while
    writing a GGML codec, are injected into the *reference* and the two layers
    above are re-run. Every one must be caught. The golden harness's
    `--self-test` makes exactly this argument for its comparators; this is the
    same argument for this axis.

    The faults are chosen so that a naive "close enough" comparison would let
    at least four of them through: truncation instead of rounding moves each
    quant by less than one step, dropping the `f16` storage of the scale moves
    the reconstruction by a part in 2048, and off-by-one on the nibble
    interleave leaves the *set* of reconstructed values unchanged. Only exact
    comparison catches those.
    """
    flat = _gauss_flat(256)
    dense = _C._tensor_from_flat(flat, [1, 256], dtype=_C.float32)

    q8 = _C._quantize(dense, "q8_0")
    blob8 = _C._quantized_blob(q8)
    q4 = _C._quantize(dense, "q4_0")
    blob4 = _C._quantized_blob(q4)
    deq8 = _flat_values(_C._dequantize(q8))

    caught = []

    # 1. Truncation where the format rounds. Python's int() on a positive
    #    value is exactly this mistake.
    bad = bytearray()
    for i in range(0, 256, 32):
        block = [ggml_ref.f32(v) for v in flat[i : i + 32]]
        amax = max(abs(v) for v in block)
        d = ggml_ref.f32(amax / 127.0)
        inv = ggml_ref.f32(1.0 / d) if d else 0.0
        bad += ggml_ref.f16_bytes(d)
        for v in block:
            bad.append(ggml_ref.as_i8(int(ggml_ref.f32(v * inv))) & 0xFF)
    caught.append(("q8_0 truncates instead of rounding", bytes(bad) != blob8))

    # 2. Banker's rounding where the format rounds half away from zero. This
    #    is what Python's `round()` does, and it is the one a reviewer would
    #    not see.
    #
    #    It needs its own fixture, and that is the point rather than an
    #    inconvenience: the fault only shows on a tie, and no tie occurred
    #    anywhere in the Gaussian block above (checked -- the injection came
    #    out byte-identical). A fault that the fixture cannot reach is a fault
    #    the axis is not testing, so the tie is constructed instead of hoped
    #    for. Absmax exactly 127 makes the scale exactly 1.0, so a half-integer
    #    entry lands exactly on the boundary: Rust rounds 0.5 to 1, Python's
    #    `round` rounds it to 0.
    ties = []
    for j in range(32):
        ties.append(127.0 if j == 0 else float(j - 16) + 0.5)
    tie_dense = _C._tensor_from_flat(ties, [1, 32], dtype=_C.float32)
    tie_blob = _C._quantized_blob(_C._quantize(tie_dense, "q8_0"))
    assert ggml_ref.quantize_q8_0(ties) == tie_blob, "the reference misses the tie fixture"
    bad = bytearray()
    d = ggml_ref.f32(127.0 / 127.0)
    inv = ggml_ref.f32(1.0 / d)
    bad += ggml_ref.f16_bytes(d)
    for v in ties:
        bad.append(ggml_ref.as_i8(float(round(ggml_ref.f32(v * inv)))) & 0xFF)
    caught.append(("q8_0 uses banker's rounding", bytes(bad) != tie_blob))

    # 3. The scale kept in f32 rather than narrowed to f16. A part in 2048 --
    #    invisible to any tolerance anyone would write.
    bad = bytearray()
    for i in range(0, 256, 32):
        block = [ggml_ref.f32(v) for v in flat[i : i + 32]]
        amax = max(abs(v) for v in block)
        d = ggml_ref.f32(amax / 127.0)
        inv = ggml_ref.f32(1.0 / d) if d else 0.0
        bad += ggml_ref.f16_bytes(d)
        for v in block:
            bad.append(ggml_ref.as_i8(ggml_ref.rust_round(ggml_ref.f32(v * inv))) & 0xFF)
    unnarrowed = []
    off = 0
    for i in range(8):
        block = [ggml_ref.f32(v) for v in flat[i * 32 : (i + 1) * 32]]
        d = ggml_ref.f32(max(abs(v) for v in block) / 127.0)  # f32, not f16
        qs = struct.unpack("<32b", bytes(bad)[off + 2 : off + 34])
        unnarrowed.extend(ggml_ref.f32(qv * d) for qv in qs)
        off += 34
    caught.append(("q8_0 scale kept in f32", repr(unnarrowed) != repr(deq8)))

    # 4. Q4_0's nibble interleave read as adjacent pairs.
    wrong = [0.0] * 256
    off = 0
    for i in range(8):
        d = ggml_ref.f16_from_bytes(blob4[off : off + 2])
        qs = blob4[off + 2 : off + 18]
        for j in range(16):
            wrong[i * 32 + 2 * j] = ggml_ref.f32(((qs[j] & 0x0F) - 8) * d)
            wrong[i * 32 + 2 * j + 1] = ggml_ref.f32(((qs[j] >> 4) - 8) * d)
        off += 18
    caught.append(
        ("q4_0 read as adjacent pairs", repr(wrong) != repr(ggml_ref.dequantize_q4_0(blob4, 256)))
    )

    # 5. Q4_0's scale taken as `amax / 8` rather than `max / -8` -- the sign
    #    dropped. Half the blocks come out negated.
    bad = bytearray()
    for i in range(0, 256, 32):
        block = [ggml_ref.f32(v) for v in flat[i : i + 32]]
        amax = max(abs(v) for v in block)
        d = ggml_ref.f32(amax / -8.0)
        inv = ggml_ref.f32(1.0 / d) if d else 0.0
        bad += ggml_ref.f16_bytes(d)
        for j in range(16):
            x0 = ggml_ref.f32(block[j] * inv)
            x1 = ggml_ref.f32(block[16 + j] * inv)
            bad.append(
                min(15, ggml_ref.as_u8(ggml_ref.f32(x0 + 8.5)))
                | (min(15, ggml_ref.as_u8(ggml_ref.f32(x1 + 8.5))) << 4)
            )
    caught.append(("q4_0 scale loses the sign of max", bytes(bad) != blob4))

    # 6. Q4K's 6-bit scale unpack done with the simple branch throughout --
    #    correct for the first four sub-blocks and wrong for the last four.
    q4k = _C._quantize(dense, "q4_k")
    blobk = _C._quantized_blob(q4k)
    ref_k = ggml_ref.dequantize_q4_k(blobk, 256)
    d = ggml_ref.f16_from_bytes(blobk[0:2])
    dmin = ggml_ref.f16_from_bytes(blobk[2:4])
    scales = blobk[4:16]
    qs = blobk[16:144]
    wrong = []
    is_ = 0
    for j in range(0, 256, 64):
        q = qs[j // 2 : j // 2 + 32]
        sc, m = scales[is_ % 4] & 63, scales[is_ % 4 + 4] & 63
        d1, m1 = ggml_ref.f32(d * sc), ggml_ref.f32(dmin * m)
        sc, m = scales[(is_ + 1) % 4] & 63, scales[(is_ + 1) % 4 + 4] & 63
        d2, m2 = ggml_ref.f32(d * sc), ggml_ref.f32(dmin * m)
        wrong.extend(ggml_ref.f32(ggml_ref.f32(d1 * (b & 0xF)) - m1) for b in q)
        wrong.extend(ggml_ref.f32(ggml_ref.f32(d2 * (b >> 4)) - m2) for b in q)
        is_ += 2
    caught.append(("q4_k 6-bit scale unpack without the split branch", repr(wrong) != repr(ref_k)))

    missed = [name for name, did in caught if not did]
    assert not missed, f"the axis did not notice: {missed}"


def test_a_quantised_matmul_is_exact_on_operands_the_format_holds_exactly():
    """**The one that makes this landable.** A lossy kernel, checked with `==`.

    Quantised arithmetic is not approximate *everywhere*. Restrict both
    operands to what Q8_0 stores without loss -- whole numbers, each 32-element
    block reaching absmax exactly 127 so the scale is 1.0 -- and the entire
    pipeline becomes integer arithmetic that `float32` carries exactly:
    `dequantize(quantize(x)) == x`, every product is an integer below 16129,
    every partial sum stays under `2**24`. So the result cannot depend on the
    accumulation order, and `_quantized_linear` through a Q8_0 weight must be
    **bit identical** to a dense `linear` over the same numbers.

    That converts the whole quantised matmul path -- the weight blocks, the
    activation quantisation candle does inside `vec_dot`, the block dot
    product, the scale multiply, the accumulation across blocks, the bias --
    into a bit comparison. `k` runs to 256, past the point where the NEON
    kernels take over from the scalar fallback, so the vector path is what is
    being checked and not the reference one.

    It is not a claim that quantisation is lossless. It is the claim that
    everything *except* the rounding is right, which is the half a tolerance
    can never separate out.
    """
    for k in (32, 64, 256, 512):
        for m in (1, 2, 8):
            for n in (1, 3, 5):
                w_rows = _exactly_representable(m, k, seed=_QUANT_SEED + k + m)
                x_rows = _exactly_representable(n, k, seed=_QUANT_SEED + k + n + 1)
                w_flat = [v for row in w_rows for v in row]
                x_flat = [v for row in x_rows for v in row]
                w = _C._tensor_from_flat(w_flat, [m, k], dtype=_C.float32)
                x = _C._tensor_from_flat(x_flat, [n, k], dtype=_C.float32)
                q = _C._quantize(w, "q8_0")

                # The weight itself survives, which is the premise.
                assert repr(_flat_values(_C._dequantize(q))) == repr(w_flat), (k, m)

                got = _flat_values(_C._quantized_linear(x, q, None))
                want = [
                    float(sum(x_rows[i][t] * w_rows[j][t] for t in range(k)))
                    for i in range(n)
                    for j in range(m)
                ]
                assert repr(got) == repr(want), (k, m, n, got[:4], want[:4])


def test_the_f32_control_path_is_a_dense_linear_to_the_bit():
    """The wiring, separated from the rounding.

    `GgmlDType::F32` is a "quantised" format that stores `float32` unchanged,
    and `QMatMul::from_arc` dequantises it up front and takes an ordinary
    `matmul`. So `_quantized_linear` with `format="f32"` has to reproduce a
    dense `linear` exactly, and what it is checking is everything that is not
    arithmetic: the transpose convention (`weight` is
    `(out_features, in_features)`, as `nn.Linear` stores it), the bias
    broadcast, the batch dimensions, the contiguity copy.

    Integer-valued operands again, and for the same reason as above rather than
    for convenience: on **random** operands this path differs from `addmm` by
    up to 2 ULP, because `QMatMul` passes the weight to BLAS as a transposed
    view where `addmm` passes it differently and the two reassociate the sum.
    Measured, 20 of 35 elements, max 2.9e-06 -- real, and not a defect on
    either side. On integer operands reassociation cannot change the answer, so
    the comparison is exact and still catches every wiring fault, which move
    numbers by far more than an ULP.
    """
    for k, m in ((32, 4), (64, 3), (256, 2)):
        w_rows = _exactly_representable(m, k, amax=64, seed=_QUANT_SEED + k)
        x_rows = _exactly_representable(3, k, amax=64, seed=_QUANT_SEED + k + 9)
        w_flat = [v for row in w_rows for v in row]
        x_flat = [v for row in x_rows for v in row]
        w = _C._tensor_from_flat(w_flat, [m, k], dtype=_C.float32)
        x = _C._tensor_from_flat(x_flat, [3, k], dtype=_C.float32)
        b_flat = [float(i - m // 2) for i in range(m)]
        b = _C._tensor_from_flat(b_flat, [m], dtype=_C.float32)
        qf = _C._quantize(w, "f32")

        for bias, bias_flat in ((None, [0.0] * m), (b, b_flat)):
            got = _flat_values(_C._quantized_linear(x, qf, bias))
            want = [
                float(sum(x_rows[i][t] * w_rows[j][t] for t in range(k)) + bias_flat[j])
                for i in range(3)
                for j in range(m)
            ]
            assert repr(got) == repr(want), (k, m, bias is not None)

        # And the batch rank the module replacement actually feeds it: a
        # transformer hands `linear` a `(batch, seq, hidden)` activation.
        x3 = _C._tensor_from_flat(x_flat, [1, 3, k], dtype=_C.float32)
        got3 = _flat_values(_C._quantized_linear(x3, qf, b))
        want3 = [
            float(sum(x_rows[i][t] * w_rows[j][t] for t in range(k)) + b_flat[j])
            for i in range(3)
            for j in range(m)
        ]
        assert repr(got3) == repr(want3), (k, m, "rank 3")


def test_the_round_trip_loss_is_under_the_formats_bound_and_over_zero():
    """Layer 3, the only one with a number in it -- and it has two.

    The ceiling is derived, not observed: Q8_0 reconstructs `round(x/d) * d`
    with `d = absmax/127`, so no element can move by more than half a step.
    Q4_0's step is eight times coarser and its bound is eight times looser.
    `ggml_ref.round_trip_bound` computes both per block.

    **The floor is the half that can fail for the interesting reason.** A
    ceiling alone is passed perfectly by a "quantiser" that returns its input,
    which is exactly the failure mode a compression claim has to rule out -- so
    the error is also required to be *non-zero*, and the formats are required
    to come out in the order the bit widths predict (`q8_0` strictly better
    than `q4_0`, `q4_k` strictly better than `q4_0` at the same 4 bits, which
    is the entire reason k-quants exist).
    """
    rows, cols = 8, 512
    flat = _gauss_flat(rows * cols)
    dense = _C._tensor_from_flat(flat, [rows, cols], dtype=_C.float32)

    rms = {}
    for fmt in ("q8_0", "q4_0", "q4_k", "q6_k"):
        back = _flat_values(_C._dequantize(_C._quantize(dense, fmt)))
        err = [ggml_ref.f32(a) - b for a, b in zip(flat, back)]
        assert any(e != 0.0 for e in err), f"{fmt} changed nothing -- that is not quantisation"
        rms[fmt] = math.sqrt(sum(e * e for e in err) / len(err)) / math.sqrt(
            sum(ggml_ref.f32(v) ** 2 for v in flat) / len(flat)
        )
        if fmt in ("q8_0", "q4_0"):
            block = ggml_ref.BLOCK_SIZE[fmt]
            for start in range(0, len(flat), block):
                chunk = flat[start : start + block]
                bound = ggml_ref.round_trip_bound(fmt, chunk)
                worst = max(abs(e) for e in err[start : start + block])
                assert worst <= bound, (fmt, start, worst, bound)

    assert rms["q8_0"] < rms["q4_k"] < rms["q4_0"], rms
    assert rms["q6_k"] < rms["q4_k"], rms
    # And the ordering is not marginal: 8 bits should be several times better
    # than 4, or something is scaling wrong rather than rounding.
    assert rms["q4_0"] > 4 * rms["q8_0"], rms


def test_a_quantised_tensor_refuses_every_dense_kernel_by_name():
    """The third `Repr` arm earns its keep the way the second one did.

    `tensor()` returns `Err` for `Repr::Quantized` exactly as it does for
    `Repr::Meta`, so the 96 kernels behind the dispatcher refuse without any of
    them having been told about quantisation. That is the property the enum
    exists for (tensor.rs, `Repr`), and this is the test that it holds rather
    than that somebody remembered to write a check.

    Each refusal is checked to *name* the format, not merely to raise: a
    `NotImplementedError` that says "meta tensor" about a Q4K weight would be
    the same defect one layer down.
    """
    _, dense, q = _quant_fixture(4, 64, "q8_0")
    probes = [
        ("aten.mm.default", (q, dense)),
        ("aten.mm.default", (dense, q)),
        ("aten.add.Tensor", (q, dense)),
        ("aten.mul.Tensor", (q, q)),
        ("aten.sum.default", (q,)),
        ("aten.t.default", (q,)),
        ("aten._to_copy.default", (q,)),
    ]
    for name, args in probes:
        try:
            _C._aten_dispatch(name, *args)
        except NotImplementedError as exc:
            assert "q8_0" in str(exc) or "block-quantised" in str(exc), (name, str(exc))
        except Exception as exc:  # noqa: BLE001
            # An op that refuses earlier for its own reason (an unimplemented
            # overload) is not a failure of this property, but it must not
            # have *computed* something.
            assert "not implemented" in str(exc).lower(), (name, type(exc).__name__, str(exc))
        else:
            raise AssertionError(f"{name} computed on a quantised tensor")

    # `tolist` is the direct read and it refuses too, which is why `print()` of
    # a quantised weight cannot produce numbers that are not there.
    try:
        q.tolist()
    except NotImplementedError as exc:
        assert "block-quantised" in str(exc), str(exc)
    else:
        raise AssertionError("tolist read a quantised tensor")


def test_element_size_refuses_rather_than_guessing_on_a_quantised_tensor():
    """`numel() * element_size()` is how upstream code sizes a buffer.

    The dtype tag on a quantised tensor is `float32` -- it is what
    `_dequantize` produces -- so answering from the tag would say 4 bytes per
    element for a Q4K weight that stores 0.5625, wrong by 7.1x in the direction
    that flatters a memory budget. It refuses and names the arithmetic
    (`type_size` per `block_size`), and `_quantized_nbytes` answers the
    question that has an answer.
    """
    _, dense, q = _quant_fixture(4, 256, "q4_k")
    assert dense.element_size() == 4
    try:
        q.element_size()
    except NotImplementedError as exc:
        assert "144 bytes per 256 elements" in str(exc), str(exc)
        assert "_quantized_nbytes" in str(exc), str(exc)
    else:
        raise AssertionError("element_size answered for a quantised tensor")
    assert _C._quantized_nbytes(q) == 4 * 144
    assert q.numel() == 1024
    # The compression this actually buys, stated the only honest way.
    assert q.numel() * 4 / _C._quantized_nbytes(q) > 7.0


def test_is_quantized_is_no_longer_a_constant():
    """docs/QUANT2.md §4, and the reason the six predicates were a `match`.

    tensor.rs wrote `is_nested`/`is_sparse`/`is_quantized`/`_is_zerotensor`/
    `is_neg`/`layout` as an exhaustive match over `Repr` so that a third arm
    could not inherit a `False` silently, and recorded that
    `test_the_alternative_representations_have_no_constructors` was the other
    half of the argument -- `False` was derivable because nothing could build
    the representation.

    A third arm landed. Five of the six still answer `False`; this one does
    not, and now has a constructor behind it. The predicate that was
    suspicious for being unfalsifiable is now falsifiable.
    """
    _, dense, q = _quant_fixture(2, 64, "q8_0")
    assert dense.is_quantized is False
    assert q.is_quantized is True
    # The other five did answer, and answered `False`.
    for name in ("is_nested", "is_sparse"):
        assert getattr(q, name) is False, name
    assert q._is_zerotensor() is False
    assert q.is_neg() is False
    assert q._layout_name() == "strided"
    # `qscheme` exists so that a reader who believes `is_quantized` gets told
    # what kind of quantised, rather than an AttributeError from _tensor_str.
    for t in (dense, q):
        try:
            t.qscheme()
        except (NotImplementedError, RuntimeError) as exc:
            assert "qscheme" in str(exc), str(exc)
        else:
            raise AssertionError("qscheme answered")


def test_a_format_that_cannot_hold_a_shape_refuses_at_the_door_by_name():
    """The wall SmolLM2 walks into, checked here so it is not a surprise there.

    A GGML block must be filled, so a format with a 256-element block cannot
    store a weight whose last dimension is 576 -- and 576 is SmolLM2-135M's
    hidden size, not an exotic number. Every k-quant is therefore unavailable
    for that model and the 32-element formats are not (docs/QUANT2.md §5.2).

    Refused in `quant.rs` rather than left to candle, so the message names the
    format and the multiple it wanted; `torchnative.quant` groups its skips by
    that text, which is how a wall shows up as a line in the report.
    """
    w = _C._tensor_from_flat(_gauss_flat(576 * 4), [4, 576], dtype=_C.float32)
    for fmt in ("q2_k", "q3_k", "q4_k", "q5_k", "q6_k"):
        try:
            _C._quantize(w, fmt)
        except NotImplementedError as exc:
            assert "divisible by block size" in str(exc), (fmt, str(exc))
            assert "576" in str(exc) and "256" in str(exc), (fmt, str(exc))
        else:
            raise AssertionError(f"{fmt} accepted a 576-column weight")
    for fmt in ("q4_0", "q8_0", "q5_0"):
        assert _C._quantized_format(_C._quantize(w, fmt)) == fmt

    # The activation-side formats refuse for a different reason and say so.
    for fmt in ("q8_1", "q8_k"):
        try:
            _C._quantize(_C._tensor_from_flat(_gauss_flat(256), [1, 256], dtype=_C.float32), fmt)
        except NotImplementedError as exc:
            assert "activation-side" in str(exc), (fmt, str(exc))
        else:
            raise AssertionError(f"{fmt} accepted a weight")

    # And an unknown name lists what there is instead of picking a default.
    try:
        _C._quantize(w, "q4_K")
    except NotImplementedError as exc:
        assert "unknown quantisation format" in str(exc), str(exc)
    else:
        raise AssertionError("a misspelled format was accepted")


def test_a_blob_round_trips_through_python_unchanged():
    """`_quantized_blob` and `_quantized_from_blob` are inverses.

    Needed for the axis rather than for its own sake: without a way in,
    `dequantize(quantize(x))` could only compare candle to itself, and the
    reference dequantisers could never be driven by a blob candle did not
    write. It is also what a GGUF writer would be built on, which is the
    reason it is a pair and not a one-way debug hook.

    The size check is part of the contract: a blob of the wrong length for the
    shape is a `ValueError` naming both, not a reinterpretation of whatever
    bytes arrived.
    """
    for fmt in ("q8_0", "q4_0", "q4_k"):
        block = ggml_ref.BLOCK_SIZE[fmt]
        _, _, q = _quant_fixture(3, block * 2, fmt)
        blob = _C._quantized_blob(q)
        back = _C._quantized_from_blob(blob, [3, block * 2], fmt)
        assert _C._quantized_format(back) == fmt
        assert _C._quantized_blob(back) == blob
        assert repr(_flat_values(_C._dequantize(back))) == repr(_flat_values(_C._dequantize(q)))
        try:
            _C._quantized_from_blob(blob[:-1], [3, block * 2], fmt)
        except ValueError as exc:
            assert "bytes" in str(exc), str(exc)
        else:
            raise AssertionError("a short blob was accepted")


def test_a_quantised_activation_is_refused_and_a_reduced_float_one_too():
    """Two refusals that would otherwise be silent widenings.

    `QMatMul::forward` takes `f32` or `f16` and this shim's reduced-precision
    path is `bfloat16` (docs/DTYPE.md §6.2), so a `bfloat16` activation has to
    be widened by somebody. Doing it inside `_quantized_linear` would hide the
    conversion cost from whoever is measuring -- the exact mistake docs/DTYPE.md
    §2 spent a document unpicking -- so it refuses and says why.

    And the weight argument must be quantised: handing `_quantized_linear` a
    dense weight would otherwise work through the F32 arm and quietly measure
    nothing.
    """
    _, dense, q = _quant_fixture(4, 64, "q8_0")
    x32 = _C._tensor_from_flat(_gauss_flat(2 * 64), [2, 64], dtype=_C.float32)
    xbf = _C._aten_dispatch("aten._to_copy.default", x32, dtype=_C.bfloat16)
    try:
        _C._quantized_linear(xbf, q, None)
    except NotImplementedError as exc:
        assert "bfloat16" in str(exc) and "float32" in str(exc), str(exc)
    else:
        raise AssertionError("a bfloat16 activation was accepted")

    try:
        _C._quantized_linear(x32, dense, None)
    except NotImplementedError as exc:
        assert "block-quantised" in str(exc), str(exc)
    else:
        raise AssertionError("a dense weight was accepted as a quantised one")

    # A shape mismatch is upstream's message, not a candle one.
    bad = _C._tensor_from_flat(_gauss_flat(2 * 32), [2, 32], dtype=_C.float32)
    try:
        _C._quantized_linear(bad, q, None)
    except RuntimeError as exc:
        assert "shapes cannot be multiplied" in str(exc), str(exc)
    else:
        raise AssertionError("a mismatched activation was accepted")


# --- docs/ARCH20.md: names that had kernels and no way in --------------------
#
# Every one of these is a *reachability* check, and that is deliberate: the
# golden harness already compares the kernels against upstream, and it did so
# for weeks while `x += y` raised `NotImplementedError` and `torch.stack`
# refused. A kernel case cannot see a missing name. These can, and each one
# fails if exactly one line is deleted from `methods.json`, `overloads.json`
# or `bootstrap.py`.


def _arch20_tensor(flat, shape=None, dtype=None):
    shape = list(shape if shape is not None else [len(flat)])
    return _C._tensor_from_flat(list(flat), shape, dtype=dtype or _C.float32)


def test_the_in_place_arithmetic_members_are_bound():
    """`add_`, `sub_`, `mul_`, `neg_`, `exp_`, `relu_` and the three `__i*__`.

    `aten.add_.Tensor` had had a kernel since docs/TAIL.md and
    `aten.relu_.default` since docs/KERNELS.md; neither was reachable from
    Python. The value assertions are deliberately weak -- the golden harness
    owns correctness -- and the *binding* is what is asserted."""
    x = _arch20_tensor([1.0, 2.0, 3.0])
    y = _arch20_tensor([10.0, 20.0, 30.0])
    assert x.add_(y).tolist() == [11.0, 22.0, 33.0]
    assert _arch20_tensor([1.0, 2.0]).sub_(_arch20_tensor([0.5, 0.5])).tolist() == [0.5, 1.5]
    assert _arch20_tensor([2.0, 3.0]).mul_(_arch20_tensor([3.0, 3.0])).tolist() == [6.0, 9.0]
    assert _arch20_tensor([1.0, -2.0]).neg_().tolist() == [-1.0, 2.0]
    assert _arch20_tensor([0.0]).exp_().tolist() == [1.0]
    assert _arch20_tensor([-1.0, 2.0]).relu_().tolist() == [0.0, 2.0]

    # The augmented-assignment spellings, which are a *different*
    # `methods.json` key from the named members and were the ones `falcon`
    # needed. `a += b` rebinds the name to whatever `__iadd__` returns, so the
    # check below would pass even against a non-mutating `__iadd__` -- which is
    # why the base check follows it.
    a = _arch20_tensor([1.0, 2.0])
    a += _arch20_tensor([1.0, 1.0])
    assert a.tolist() == [2.0, 3.0]
    b = _arch20_tensor([4.0, 6.0])
    b -= _arch20_tensor([1.0, 1.0])
    assert b.tolist() == [3.0, 5.0]
    c = _arch20_tensor([4.0, 6.0])
    c *= _arch20_tensor([2.0, 2.0])
    assert c.tolist() == [8.0, 12.0]

    # The scalar overloads, which need their own kernels because the parser
    # binds a `Scalar` signature (docs/ARCH20.md §8.4).
    assert _arch20_tensor([1.0, 2.0]).add_(1.0).tolist() == [2.0, 3.0]
    assert _arch20_tensor([1.0, 2.0]).sub_(1.0).tolist() == [0.0, 1.0]
    assert _arch20_tensor([1.0, 2.0]).mul_(3.0).tolist() == [3.0, 6.0]


def test_the_in_place_members_write_through_to_the_base():
    """The half a return-value assertion cannot reach (docs/VIEWS.md §6).

    Every in-place op returns `self`, so `t.add_(1); assert t == expected`
    passes against a kernel that computed into a fresh buffer and handed it
    back. These narrow a view, mutate the view, and read the **base**."""
    for label, mutate, expected in [
        ("add_", lambda v: v.add_(_arch20_tensor([100.0, 100.0])), [1.0, 2.0, 103.0, 104.0]),
        ("sub_", lambda v: v.sub_(_arch20_tensor([1.0, 1.0])), [1.0, 2.0, 2.0, 3.0]),
        ("mul_", lambda v: v.mul_(_arch20_tensor([10.0, 10.0])), [1.0, 2.0, 30.0, 40.0]),
        ("neg_", lambda v: v.neg_(), [1.0, 2.0, -3.0, -4.0]),
        ("relu_", lambda v: v.relu_(), [1.0, 2.0, 3.0, 4.0]),
        ("add_ scalar", lambda v: v.add_(1.0), [1.0, 2.0, 4.0, 5.0]),
        ("__iadd__", lambda v: v.__iadd__(_arch20_tensor([1.0, 1.0])), [1.0, 2.0, 4.0, 5.0]),
    ]:
        base = _arch20_tensor([1.0, 2.0, 3.0, 4.0], (2, 2))
        mutate(base[1])
        assert base.tolist() == [expected[:2], expected[2:]], (label, base.tolist())


def test_the_in_place_ops_refuse_a_cast_upstream_refuses(): 
    """`inplace_cast_check` -- docs/ARCH20.md §8.3.

    This shim used to *compute* a truncated answer for the first of these,
    which is the silent-divergence direction. Upstream raises for all four."""
    for label, call in [
        ("int32.add_(float32)",
         lambda: _arch20_tensor([1, 2], dtype=_C.int32).add_(_arch20_tensor([1.5, 2.5]))),
        ("int32.mul_(2.5)",
         lambda: _arch20_tensor([1, 2], dtype=_C.int32).mul_(2.5)),
        ("int64.exp_()",
         lambda: _arch20_tensor([1, 2], dtype=_C.int64).exp_()),
        ("bool.neg_()",
         lambda: _arch20_tensor([1, 0], dtype=_C.bool).neg_()),
    ]:
        try:
            call()
        except RuntimeError as e:
            assert "cast" in str(e) or "not supported" in str(e), (label, str(e))
        else:
            raise AssertionError(f"{label} must refuse")
    # ...and the safe direction still computes, so the check is not a blanket
    # refusal on any dtype difference.
    assert _arch20_tensor([1.0, 2.0]).add_(
        _arch20_tensor([1, 1], dtype=_C.int32)
    ).tolist() == [2.0, 3.0]


def test_the_spellings_whose_kernels_already_existed_now_resolve():
    """Six names that dispatched to implemented, golden-compared kernels and
    refused at the Python surface (docs/ARCH20.md §0.3)."""
    vf = _C._VariableFunctions
    a = _arch20_tensor([1.0, 2.0])
    assert vf.stack([a, a]).shape == (2, 2)
    assert vf.exp(_arch20_tensor([0.0])).tolist() == [1.0]
    assert vf.zeros_like(a).tolist() == [0.0, 0.0]
    assert _C._nn.softplus(_arch20_tensor([0.0])).tolist()[0] > 0.0
    # conv1d: the composite fills in `transposed=False` and `output_padding`.
    x = _arch20_tensor([float(v) for v in range(8)], (1, 2, 4))
    w = _arch20_tensor([1.0, 0.0, 0.0, 1.0], (2, 1, 2))
    assert vf.conv1d(x, w, None, 1, 0, 1, 2).shape == (1, 2, 3)
    assert vf.flatten(_arch20_tensor([1.0, 2.0, 3.0, 4.0], (2, 2))).shape == (4,)


def test_the_three_composites_that_opened_persimmon_and_cohere():
    vf = _C._VariableFunctions
    # `square` is `pow(x, 2)` with an INTEGER exponent, which is what keeps an
    # integral tensor integral.
    assert vf.square(_arch20_tensor([2.0, 3.0])).tolist() == [4.0, 9.0]
    squared = vf.square(_arch20_tensor([2, 3], dtype=_C.int64))
    assert squared.dtype == _C.int64, squared.dtype
    assert squared.tolist() == [4, 9]
    # `repeat_interleave`, both dims, because the two answers differ and a
    # wrong unsqueeze axis passes one of them.
    m = _arch20_tensor([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], (2, 3))
    assert vf.repeat_interleave(m, 2, dim=-1).tolist() == [
        [0.0, 0.0, 1.0, 1.0, 2.0, 2.0], [3.0, 3.0, 4.0, 4.0, 5.0, 5.0]
    ]
    assert vf.repeat_interleave(m, 2, dim=0).tolist() == [
        [0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [3.0, 4.0, 5.0]
    ]
    assert vf.repeat_interleave(m, 2).shape == (12,)
    # The tensor-`repeats` overload is refused BY NAME, not approximated.
    try:
        vf.repeat_interleave(m, _arch20_tensor([1, 2, 3], dtype=_C.int64), dim=1)
    except NotImplementedError as e:
        assert "repeat_interleave.Tensor" in str(e), str(e)
    else:
        raise AssertionError("a tensor `repeats` must refuse")
    # `flatten`'s 0-d arm is a reshape to [1], not a no-op.
    assert _arch20_tensor([5.0], ()).flatten().shape == (1,)


def test_a_list_index_lifts_into_an_index_tensor():
    """`falcon`'s `fused_qkv[..., [-2], :]` (docs/ARCH20.md §7)."""
    x = _arch20_tensor([float(v) for v in range(24)], (2, 3, 4))
    assert x[..., [-2], :].shape == (2, 1, 4)
    assert x[..., [-2], :].tolist() == [[[4.0, 5.0, 6.0, 7.0]], [[16.0, 17.0, 18.0, 19.0]]]
    # A tuple item lifts the same way a list one does.
    assert x[..., (0, 1), :].shape == (2, 2, 4)
    # A bare list at top level is ONE index tensor...
    assert x[[0, 1]].shape == (2, 3, 4)
    # ...unless it contains a slice, in which case upstream reads it as a
    # tuple of indices (`treatSequenceAsTuple`). Both arms, because picking
    # either one alone passes half the cases.
    assert x[[slice(None)]].shape == (2, 3, 4)
    assert x[[[0, 1]]].shape == (2, 3, 4)
    # And the write side takes the same route.
    y = _arch20_tensor([0.0, 0.0, 0.0, 0.0], (2, 2))
    y[[0]] = _arch20_tensor([7.0, 8.0])
    assert y.tolist() == [[7.0, 8.0], [0.0, 0.0]]


def test_the_determinism_flags_are_state_cells_with_upstreams_defaults():
    """`bert`'s wall: `F.pad` reads this on every call (docs/ARCH20.md §2)."""
    assert _C._get_deterministic_algorithms() is False
    assert _C._get_deterministic_algorithms_warn_only() is False
    # The one that defaults True -- a blanket "determinism starts off" would
    # have got exactly this cell wrong.
    assert _C._get_deterministic_fill_uninitialized_memory() is True
    assert _C._get_cudnn_deterministic() is False
    assert _C._get_mkldnn_deterministic() is False
    try:
        _C._set_deterministic_algorithms(True, warn_only=True)
        assert _C._get_deterministic_algorithms() is True
        assert _C._get_deterministic_algorithms_warn_only() is True
        # `torch/__init__.py:1585` calls it with one positional argument.
        _C._set_deterministic_algorithms(False)
        assert _C._get_deterministic_algorithms() is False
        assert _C._get_deterministic_algorithms_warn_only() is False
    finally:
        _C._set_deterministic_algorithms(False)


def test_pad_is_wired_for_constant_mode_and_refuses_the_other_three():
    padded = _C._nn.pad(_arch20_tensor([1.0, 2.0]), (0, 3), "constant", 0)
    assert padded.tolist() == [1.0, 2.0, 0.0, 0.0, 0.0]
    # `pad` is read LAST-dim-first and in pairs; the two dims get different
    # pads here so a front-to-back reading produces a different shape.
    grid = _arch20_tensor([float(v) for v in range(6)], (2, 3))
    assert _C._nn.pad(grid, (1, 1, 2, 0), "constant", 7.0).shape == (4, 5)
    # A negative entry crops, and crop-and-pad happen on the same axis.
    assert _C._aten_dispatch("aten.constant_pad_nd.default", grid, [-1, 2]).tolist() == [
        [1.0, 2.0, 0.0, 0.0], [4.0, 5.0, 0.0, 0.0]
    ]
    for mode in ("reflect", "replicate", "circular"):
        try:
            _C._nn.pad(_arch20_tensor([1.0, 2.0]), (1, 1), mode, 0)
        except NotImplementedError as e:
            assert mode in str(e), str(e)
        else:
            raise AssertionError(f"pad(mode={mode!r}) must refuse")


def test_autograd_function_apply_runs_the_forward():
    """`bloom` calls `autograd.Function.apply` on a FORWARD (§6.3).

    `_FunctionBase.apply` is exercised here without the vendored tree, by
    standing in the two things the tree's metaclass provides: a
    `_backward_cls` and a `forward`."""
    assert _C._are_functorch_transforms_active() is False
    probe = _arch20_tensor([1.0, 2.0])
    assert _C._functorch.unwrap_if_dead(probe) is probe

    seen = {}

    class Ctx(_C._FunctionBase):
        def save_for_backward(self, *tensors):
            seen["saved"] = tensors

    class Combined:
        _backward_cls = Ctx

        @staticmethod
        def forward(ctx, value):
            ctx.save_for_backward(value)
            seen["needs"] = ctx.needs_input_grad
            return value.add_(_arch20_tensor([1.0, 1.0]))

    out = _C._FunctionBase.apply.__func__(Combined, probe)
    assert out.tolist() == [2.0, 3.0]
    assert seen["saved"] == (probe,)
    # Read-only upstream (a `getset_descriptor`), so it is a property here and
    # the forward has to be able to read it.
    assert seen["needs"] == (False,)


def test_clamp_out_of_place_promotes_where_clamp_underscore_refuses():
    """The dtype rule that is NOT shared between the two (docs/ARCH20.md §4)."""
    promoted = _arch20_tensor([1, 5, 10], dtype=_C.int32).clamp(max=2.0)
    assert promoted.dtype == _C.float32, promoted.dtype
    assert promoted.tolist() == [1.0, 2.0, 2.0]
    # The in-place sibling refuses the identical call.
    try:
        _arch20_tensor([1, 5, 10], dtype=_C.int32).clamp_(max=2.0)
    except RuntimeError as e:
        assert "cast" in str(e), str(e)
    else:
        raise AssertionError("clamp_ must refuse a float bound on an int receiver")
    # ...and the receiver of the out-of-place form is untouched.
    receiver = _arch20_tensor([1.0, 5.0, 10.0])
    receiver.clamp(max=2.0)
    assert receiver.tolist() == [1.0, 5.0, 10.0]


def test_expm1_is_not_exp_minus_one():
    """The one case that separates the two (docs/ARCH20.md §4).

    Upstream float64 `expm1(1e-8)` is `1.0000000050000001e-08`;
    `exp(1e-8) - 1` is `9.99999993922529e-09`, wrong from the ninth digit. A
    subtraction-based kernel passes every other expm1 case and fails this."""
    tiny = _C._tensor_from_flat([1e-8], [1], dtype=_C.float64)
    got = _C._VariableFunctions.expm1(tiny).tolist()[0]
    assert abs(got - 1.0000000050000001e-08) < 1e-24, got
    naive = _C._VariableFunctions.exp(tiny).tolist()[0] - 1.0
    assert abs(naive - got) > 1e-17, (naive, got)


def test_pow_promotes_and_refuses_where_upstream_does():
    """`bloom`'s `torch.pow(float32_base, int32_powers)` (docs/ARCH20.md §6)."""
    base = _arch20_tensor([2.0, 3.0])
    powers = _arch20_tensor([2, 3], dtype=_C.int32)
    out = _C._aten_dispatch("aten.pow.Tensor_Tensor", base, powers)
    assert out.dtype == _C.float32, out.dtype
    assert out.tolist() == [4.0, 27.0]
    # Same-rank reduced floats escape upwards; a same-dtype pair does not.
    f16 = _C._tensor_from_flat([2.0], [1], dtype=_C.float16)
    bf16 = _C._tensor_from_flat([3.0], [1], dtype=_C.bfloat16)
    assert _C._aten_dispatch("aten.pow.Tensor_Tensor", f16, bf16).dtype == _C.float32
    assert _C._aten_dispatch("aten.pow.Tensor_Tensor", f16, f16).dtype == _C.float16
    # bool ** bool is where upstream raises, so this does too.
    mask = _arch20_tensor([1, 0], dtype=_C.bool)
    try:
        _C._aten_dispatch("aten.pow.Tensor_Tensor", mask, mask)
    except NotImplementedError as e:
        assert "bool" in str(e), str(e)
    else:
        raise AssertionError("bool ** bool must refuse")
    # A negative integer exponent: `powi`, not a refusal -- only the
    # Tensor_Scalar overload refuses.
    b = _arch20_tensor([2, 1, -1, 0], dtype=_C.int64)
    e = _arch20_tensor([-1, -1, -1, -1], dtype=_C.int64)
    assert _C._aten_dispatch("aten.pow.Tensor_Tensor", b, e).tolist() == [0, 1, -1, 0]
    try:
        _C._aten_dispatch("aten.pow.Tensor_Scalar", b, -1)
    except RuntimeError as e:
        assert "negative integer powers" in str(e), str(e)
    else:
        raise AssertionError("pow.Tensor_Scalar with a negative int must refuse")


# ---------------------------------------------------------------------------
# The spelling road (docs/SPELLINGS.md, docs/ARCH20.md §9's 25-name inventory)
# ---------------------------------------------------------------------------
#
# Every case above this reaches its kernel one of two ways: `_C._aten_dispatch`
# directly, or (in the e2e sections) a caller several layers removed from the
# table this round edited. Neither proves the thing docs/ARCH20.md §9 found
# missing for 22 names: that `torch.<name>(...)` -- the literal spelling a
# model's Python source writes -- resolves through `overloads.json` /
# `methods.json` to the *same* kernel. `tools/golden/cases.py` cannot prove
# this either (docs/SPELLINGS.md): its `c_module` is the bare `_C` extension,
# loaded standalone, with no `overloads.json`/`methods.json` resolver
# installed on it at all -- so a golden case for e.g. `aten.abs.default` was
# passing for as long as that kernel existed, regardless of whether
# `torch.abs` had ever been wired to reach it. This is the gap that let a
# gpt2-shaped case: kernel golden-compared for weeks, `torch.<name>` still
# `NotImplementedError`.
#
# So this section runs in a subprocess with the shim-backed `torch` on
# `PYTHONPATH` (the same shape `_DEVICE_ROAD_SCRIPT` above uses) and calls
# every one of the 22 names the way a model actually would: as
# `torch.<name>(...)`, and as `tensor.<name>(...)` wherever a member exists.
# Expected values are computed in plain Python (`math.cos`, `==`, `&`, ...)
# rather than by importing upstream torch a second time, so this section
# still runs on an interpreter with no torch installed at all, matching this
# file's own docstring promise.
#
# The 3 names docs/ARCH20.md §9 listed that do NOT get an entry here --
# `gelu`, `silu`, `softplus` -- are deliberately absent from this road too:
# measured (`hasattr(torch, "gelu")` on real torch 2.13.0) there never was a
# bare `torch.gelu` upstream, only `torch.nn.functional.gelu` /
# `torch._C._nn.gelu`, and both of those already work in this shim
# (`bootstrap.py`'s `_install_nn`) -- there is no gap to close for them, so no
# `overloads.json` entry and nothing to prove here either.

_SPELL_ROAD_SCRIPT = r"""
import json, math, sys
import torch

out = {}

def rec(key, value_fn):
    try:
        out[key] = value_fn()
    except Exception as e:
        out[key] = f"ERROR:{type(e).__name__}:{e}"

x = torch.tensor([-1.0, 2.0, -3.0, 4.0])
y = torch.tensor([1.0, 1.0, -3.0, 5.0])

# --- unary, free function + member ------------------------------------------
rec("abs_fn", lambda: torch.abs(x).tolist())
rec("abs_member", lambda: x.abs().tolist())
rec("cos_fn", lambda: torch.cos(x).tolist())
rec("cos_member", lambda: x.cos().tolist())
rec("sin_fn", lambda: torch.sin(x).tolist())
rec("sin_member", lambda: x.sin().tolist())
rec("reciprocal_fn", lambda: torch.reciprocal(torch.tensor([1.0, 2.0, 4.0])).tolist())
rec("reciprocal_member", lambda: torch.tensor([1.0, 2.0, 4.0]).reciprocal().tolist())

# clone: a real copy, not an alias -- mutate the clone, check the original.
def _clone_check():
    src = x.clone()
    cloned = torch.clone(src)
    cloned.fill_(0.0)
    return [src.tolist(), cloned.tolist()]
rec("clone_fn_independent", _clone_check)
def _clone_member_check():
    src = x.clone()
    cloned = src.clone()
    cloned.fill_(0.0)
    return [src.tolist(), cloned.tolist()]
rec("clone_member_independent", _clone_member_check)

# --- clamp: both/min-only/max-only (kernel), neither (kernel-less) ---------
rec("clamp_both_fn", lambda: torch.clamp(x, min=-2.0, max=2.0).tolist())
rec("clamp_min_fn", lambda: torch.clamp(x, min=-2.0).tolist())
rec("clamp_max_fn", lambda: torch.clamp(x, max=2.0).tolist())
rec("clamp_both_member", lambda: x.clamp(min=-2.0, max=2.0).tolist())
try:
    torch.clamp(x)
    out["clamp_neither_fn"] = "ACCEPTED"
except NotImplementedError as e:
    out["clamp_neither_fn"] = f"refused:{e}"

# --- comparisons: all six, free function + member, plus mixed-dtype -------
for name in ("eq", "ne", "lt", "le", "gt", "ge"):
    rec(f"{name}_fn", lambda name=name: getattr(torch, name)(x, y).tolist())
    rec(f"{name}_member", lambda name=name: getattr(x, name)(y).tolist())
try:
    torch.eq(torch.tensor([1, 2, 3], dtype=torch.int32), torch.tensor([1.0, 3.0, 2.0]))
    out["eq_mixed_dtype"] = "ACCEPTED"
except NotImplementedError as e:
    out["eq_mixed_dtype"] = f"refused:{e}"

# --- max/min: whole-tensor, two-tensor (max only), dim-reduce (max only) --
rec("max_whole_fn", lambda: torch.max(x).item())
rec("max_whole_member", lambda: x.max().item())
rec("max_other_fn", lambda: torch.max(x, y).tolist())
rec("max_other_member", lambda: x.max(y).tolist())
m2d = torch.tensor([[1.0, 5.0, 2.0], [9.0, 0.0, 3.0]])
def _max_dim():
    vals, idx = torch.max(m2d, 1)
    return [vals.tolist(), idx.tolist()]
rec("max_dim_fn", _max_dim)
def _max_dim_member():
    vals, idx = m2d.max(1)
    return [vals.tolist(), idx.tolist()]
rec("max_dim_member", _max_dim_member)
rec("min_whole_fn", lambda: torch.min(x).item())
rec("min_whole_member", lambda: x.min().item())
# `min.other` and `min.dim` were recorded here as *refusals* while their
# spelling-table entries existed with no kernel behind them (docs/SPELLINGS.md
# §7.2, deliberately, so the refusal would name the right op). docs/TRIL.md §3
# implemented both; these now compute, and this is where that shows.
rec("min_other_fn", lambda: torch.min(x, y).tolist())
rec("min_other_member", lambda: x.min(y).tolist())
def _min_dim():
    vals, idx = torch.min(m2d, 1)
    return [vals.tolist(), idx.tolist()]
rec("min_dim_fn", _min_dim)
def _min_dim_member():
    vals, idx = m2d.min(1)
    return [vals.tolist(), idx.tolist()]
rec("min_dim_member", _min_dim_member)

# --- tril / triu / amax / softmax: docs/TRIL.md's four new names -----------
tri = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
rec("tril_fn", lambda: torch.tril(tri).tolist())
rec("tril_member", lambda: tri.tril().tolist())
rec("tril_diag_fn", lambda: torch.tril(tri, 1).tolist())
rec("tril_diag_member", lambda: tri.tril(diagonal=-1).tolist())
rec("triu_fn", lambda: torch.triu(tri).tolist())
rec("triu_member", lambda: tri.triu().tolist())
rec("triu_diag_fn", lambda: torch.triu(tri, 1).tolist())
# GPT-BigCode's own call, verbatim: a bool causal-mask buffer.
rec("tril_bool_fn", lambda: torch.tril(torch.ones((3, 3), dtype=torch.bool)).tolist())
rec("tril_bool_dtype", lambda: str(torch.tril(torch.ones((3, 3), dtype=torch.bool)).dtype))
rec("amax_fn", lambda: torch.amax(m2d, 1).tolist())
rec("amax_member", lambda: m2d.amax(1).tolist())
rec("softmax_fn", lambda: torch.softmax(m2d, dim=1).tolist())
rec("softmax_member", lambda: m2d.softmax(1).tolist())
rec("softmax_dtype_fn", lambda: str(torch.softmax(m2d, 1, torch.float64).dtype))

# --- mul: Tensor-Tensor and Tensor-Scalar, free function + member ---------
rec("mul_tensor_fn", lambda: torch.mul(x, y).tolist())
rec("mul_scalar_fn", lambda: torch.mul(x, 2.0).tolist())
rec("mul_tensor_member", lambda: x.mul(y).tolist())

# --- reshape: contiguous, inferred dim, non-contiguous (copy arm) ---------
rec("reshape_fn", lambda: torch.reshape(x, (2, 2)).tolist())
rec("reshape_member", lambda: x.reshape((2, 2)).tolist())
def _reshape_noncontig():
    base = torch.arange(12.0).reshape(3, 4).t()
    return torch.reshape(base, (12,)).tolist()
rec("reshape_noncontig_fn", _reshape_noncontig)

# --- unbind: default dim and explicit dim, free function + member ---------
m2 = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
rec("unbind_default_fn", lambda: [t.tolist() for t in torch.unbind(m2)])
rec("unbind_dim1_fn", lambda: [t.tolist() for t in torch.unbind(m2, dim=1)])
rec("unbind_member", lambda: [t.tolist() for t in m2.unbind()])

# --- bitwise family: Tensor + Scalar, free function + member --------------
bi = torch.tensor([1, 3, 5], dtype=torch.int64)
bj = torch.tensor([3, 3, 4], dtype=torch.int64)
rec("bitwise_and_fn", lambda: torch.bitwise_and(bi, bj).tolist())
rec("bitwise_and_scalar_fn", lambda: torch.bitwise_and(bi, 3).tolist())
rec("bitwise_and_member", lambda: bi.bitwise_and(bj).tolist())
rec("bitwise_or_fn", lambda: torch.bitwise_or(bi, bj).tolist())
rec("bitwise_or_member", lambda: bi.bitwise_or(bj).tolist())
rec("bitwise_not_fn", lambda: torch.bitwise_not(bi).tolist())
rec("bitwise_not_member", lambda: bi.bitwise_not().tolist())
bbool = torch.tensor([True, False, True])
rec("bitwise_not_bool_fn", lambda: torch.bitwise_not(bbool).tolist())

# --- scalar_tensor: no member (a factory, no receiver) ---------------------
rec("scalar_tensor_fn", lambda: torch.scalar_tensor(3.5).item())
rec("scalar_tensor_dtype_fn", lambda: str(torch.scalar_tensor(3.5).dtype))

# --- convolution: 1-D, no member, checked against the raw kernel path -----
def _convolution_matches_raw_dispatch():
    w = torch.zeros(2, 3, 3)
    w[0, 0, 1] = 1.0  # picks out one input channel, identity-ish kernel
    inp = torch.arange(24.0).reshape(1, 3, 8)
    via_spelling = torch.convolution(inp, w, None, [1], [0], [1], False, [0], 1)
    via_raw = torch._C._aten_dispatch(
        "aten.convolution.default", inp, w, None, [1], [0], [1], False, [0], 1
    )
    return [via_spelling.tolist() == via_raw.tolist(), list(via_spelling.shape)]
rec("convolution_fn_matches_raw", _convolution_matches_raw_dispatch)

# --- spelling reaches the same kernel `_aten_dispatch` does, directly -----
# (the "reaches the right kernel" half of docs/SPELLINGS.md's split, for a
# sample of the 22 rather than all of them -- a resolver bug that picked a
# *different but coincidentally value-compatible* kernel would not be caught
# by the value checks above alone.)
rec("abs_matches_raw", lambda: torch.abs(x).tolist() == torch._C._aten_dispatch("aten.abs.default", x).tolist())
rec("eq_matches_raw", lambda: torch.eq(x, y).tolist() == torch._C._aten_dispatch("aten.eq.Tensor", x, y).tolist())
rec("max_other_matches_raw", lambda: torch.max(x, y).tolist() == torch._C._aten_dispatch("aten.max.other", x, y).tolist())
rec("reshape_matches_raw", lambda: torch.reshape(x, (2, 2)).tolist() == torch._C._aten_dispatch("aten.reshape.default", x, [2, 2]).tolist())
rec("bitwise_and_matches_raw", lambda: torch.bitwise_and(bi, bj).tolist() == torch._C._aten_dispatch("aten.bitwise_and.Tensor", bi, bj).tolist())

json.dump(out, sys.stdout)
"""


def _spell_road_fixture():
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    proc = subprocess.run(
        [sys.executable, "-c", _SPELL_ROAD_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"spelling-road subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def test_spelling_road_through_the_vendored_tree():
    """docs/ARCH20.md §9's 22 fixable names, each through `torch.<name>(...)`
    (and `tensor.<name>(...)` where a member exists), in a real `import torch`
    against the shim -- not `_C._aten_dispatch` with the op string typed by
    the test author, which is exactly what let these 22 sit unreachable while
    every one of their kernels passed kernel-level golden comparison."""
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return  # vendor tree not installed -- see vendor/install_shim.sh
    out = _spell_road_fixture()

    def eq(key, expected):
        got = out.get(key, "<missing>")
        assert got == expected, f"{key}: expected {expected!r}, got {got!r}"

    def close(key, expected, tol=1e-4):
        got = out.get(key, "<missing>")
        assert isinstance(got, list) and len(got) == len(expected), f"{key}: got {got!r}"
        for g, e in zip(got, expected):
            assert abs(g - e) < tol, f"{key}: expected {expected!r}, got {got!r}"

    close("abs_fn", [1.0, 2.0, 3.0, 4.0])
    close("abs_member", [1.0, 2.0, 3.0, 4.0])
    close("cos_fn", [math.cos(v) for v in [-1.0, 2.0, -3.0, 4.0]])
    close("cos_member", [math.cos(v) for v in [-1.0, 2.0, -3.0, 4.0]])
    close("sin_fn", [math.sin(v) for v in [-1.0, 2.0, -3.0, 4.0]])
    close("sin_member", [math.sin(v) for v in [-1.0, 2.0, -3.0, 4.0]])
    close("reciprocal_fn", [1.0, 0.5, 0.25])
    close("reciprocal_member", [1.0, 0.5, 0.25])

    eq("clone_fn_independent", [[-1.0, 2.0, -3.0, 4.0], [0.0, 0.0, 0.0, 0.0]])
    eq("clone_member_independent", [[-1.0, 2.0, -3.0, 4.0], [0.0, 0.0, 0.0, 0.0]])

    close("clamp_both_fn", [-1.0, 2.0, -2.0, 2.0])
    close("clamp_min_fn", [-1.0, 2.0, -2.0, 4.0])
    close("clamp_max_fn", [-1.0, 2.0, -3.0, 2.0])
    close("clamp_both_member", [-1.0, 2.0, -2.0, 2.0])
    got = out.get("clamp_neither_fn", "")
    assert got.startswith("refused:") and "clamp.Tensor" in got, got

    x, y = [-1.0, 2.0, -3.0, 4.0], [1.0, 1.0, -3.0, 5.0]
    eq("eq_fn", [a == b for a, b in zip(x, y)])
    eq("eq_member", [a == b for a, b in zip(x, y)])
    eq("ne_fn", [a != b for a, b in zip(x, y)])
    eq("ne_member", [a != b for a, b in zip(x, y)])
    eq("lt_fn", [a < b for a, b in zip(x, y)])
    eq("lt_member", [a < b for a, b in zip(x, y)])
    eq("le_fn", [a <= b for a, b in zip(x, y)])
    eq("le_member", [a <= b for a, b in zip(x, y)])
    eq("gt_fn", [a > b for a, b in zip(x, y)])
    eq("gt_member", [a > b for a, b in zip(x, y)])
    eq("ge_fn", [a >= b for a, b in zip(x, y)])
    eq("ge_member", [a >= b for a, b in zip(x, y)])
    got = out.get("eq_mixed_dtype", "")
    assert got.startswith("refused:"), got  # documented gap, not this round's to close

    eq("max_whole_fn", 4.0)
    eq("max_whole_member", 4.0)
    close("max_other_fn", [1.0, 2.0, -3.0, 5.0])
    close("max_other_member", [1.0, 2.0, -3.0, 5.0])
    eq("max_dim_fn", [[5.0, 9.0], [1, 0]])
    eq("max_dim_member", [[5.0, 9.0], [1, 0]])
    eq("min_whole_fn", -3.0)
    eq("min_whole_member", -3.0)
    # Both of these asserted a *refusal* until docs/TRIL.md §3 gave them
    # kernels. x = [-1, 2, -3, 4], y = [1, 1, -3, 5].
    close("min_other_fn", [-1.0, 1.0, -3.0, 4.0])
    close("min_other_member", [-1.0, 1.0, -3.0, 4.0])
    eq("min_dim_fn", [[1.0, 0.0], [0, 1]])
    eq("min_dim_member", [[1.0, 0.0], [0, 1]])

    # docs/TRIL.md's new names, every one through the Python spelling and the
    # member -- the route the golden harness cannot see.
    eq("tril_fn", [[1.0, 0.0, 0.0], [4.0, 5.0, 0.0], [7.0, 8.0, 9.0]])
    eq("tril_member", [[1.0, 0.0, 0.0], [4.0, 5.0, 0.0], [7.0, 8.0, 9.0]])
    eq("tril_diag_fn", [[1.0, 2.0, 0.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    eq("tril_diag_member", [[0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [7.0, 8.0, 0.0]])
    eq("triu_fn", [[1.0, 2.0, 3.0], [0.0, 5.0, 6.0], [0.0, 0.0, 9.0]])
    eq("triu_member", [[1.0, 2.0, 3.0], [0.0, 5.0, 6.0], [0.0, 0.0, 9.0]])
    eq("triu_diag_fn", [[0.0, 2.0, 3.0], [0.0, 0.0, 6.0], [0.0, 0.0, 0.0]])
    eq("tril_bool_fn", [[True, False, False], [True, True, False], [True, True, True]])
    eq("tril_bool_dtype", "torch.bool")
    eq("amax_fn", [5.0, 9.0])
    eq("amax_member", [5.0, 9.0])
    # softmax rows: [1, 5, 2] and [9, 0, 3], each normalised along dim 1.
    for key in ("softmax_fn", "softmax_member"):
        rows = out.get(key, "<missing>")
        assert isinstance(rows, list) and len(rows) == 2, f"{key}: {rows!r}"
        for row, src in zip(rows, ([1.0, 5.0, 2.0], [9.0, 0.0, 3.0])):
            assert abs(sum(row) - 1.0) < 1e-5, f"{key}: row does not sum to 1: {row!r}"
            denom = sum(math.exp(v - max(src)) for v in src)
            for got_v, s in zip(row, src):
                want = math.exp(s - max(src)) / denom
                assert abs(got_v - want) < 1e-5, f"{key}: {row!r} vs {src!r}"
    eq("softmax_dtype_fn", "torch.float64")

    close("mul_tensor_fn", [-1.0, 2.0, 9.0, 20.0])
    close("mul_scalar_fn", [-2.0, 4.0, -6.0, 8.0])
    close("mul_tensor_member", [-1.0, 2.0, 9.0, 20.0])

    eq("reshape_fn", [[-1.0, 2.0], [-3.0, 4.0]])
    eq("reshape_member", [[-1.0, 2.0], [-3.0, 4.0]])
    eq("reshape_noncontig_fn", [0.0, 4.0, 8.0, 1.0, 5.0, 9.0, 2.0, 6.0, 10.0, 3.0, 7.0, 11.0])

    eq("unbind_default_fn", [[1.0, 2.0], [3.0, 4.0]])
    eq("unbind_dim1_fn", [[1.0, 3.0], [2.0, 4.0]])
    eq("unbind_member", [[1.0, 2.0], [3.0, 4.0]])

    eq("bitwise_and_fn", [1, 3, 4])
    eq("bitwise_and_scalar_fn", [1, 3, 1])
    eq("bitwise_and_member", [1, 3, 4])
    eq("bitwise_or_fn", [3, 3, 5])
    eq("bitwise_or_member", [3, 3, 5])
    eq("bitwise_not_fn", [-2, -4, -6])
    eq("bitwise_not_member", [-2, -4, -6])
    eq("bitwise_not_bool_fn", [False, True, False])

    eq("scalar_tensor_fn", 3.5)
    eq("scalar_tensor_dtype_fn", "torch.float32")

    eq("convolution_fn_matches_raw", [True, [1, 2, 6]])

    eq("abs_matches_raw", True)
    eq("eq_matches_raw", True)
    eq("max_other_matches_raw", True)
    eq("reshape_matches_raw", True)
    eq("bitwise_and_matches_raw", True)


# --- torch.jit.script defaults to unavailable, upstream's own way ----------
#
# docs/TORCHSCRIPT.md is the investigation. Short version: there is no
# TorchScript frontend here, `bootstrap.py` now sets `PYTORCH_JIT=0` by
# `os.environ.setdefault` before `torch.jit._state` is first imported, and
# upstream's own `torch/jit/_script.py` (`if not _enabled: return obj`,
# unmodified in the vendored tree) is what turns `torch.jit.script` and
# `torch.jit.script_method` into identity in that mode.
#
# These three tests go through the *Python-facing* path (call
# `torch.jit.script`, read back what it returns / whether an import raises),
# not `_C._aten_dispatch` or a dispatch-key table -- the brief's own warning
# that "golden compares by dispatch key and is structurally blind to a
# missing spelling" applies to this fix exactly as much as to a kernel: a
# passing dispatch-key case would prove nothing about whether the decorator
# a real model file uses actually degrades.
#
# All three run in a subprocess with the vendored (shim-backed) `torch` on
# `PYTHONPATH`, the same shape every other *-road test in this file uses,
# because there is no way to have both the vendored `torch` and this
# process's own `_C`-only import coexist as one `torch` module.

_JIT_DEFAULT_SCRIPT = r"""
import json, sys
import torch
out = {}
out["env_pytorch_jit"] = __import__("os").environ.get("PYTORCH_JIT")
import torch.jit._state as st
out["jit_enabled"] = bool(st._enabled)

def f(x):
    return x + 1

scripted = torch.jit.script(f)
out["script_is_identity"] = scripted is f
out["script_result"] = scripted(3)

class M(torch.nn.Module):
    def forward(self, x):
        return x + 1

    __constants__ = []

import warnings
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    stub = torch.jit.script_method(M.forward)
out["script_method_is_identity"] = stub is M.forward

json.dump(out, sys.stdout)
"""


@functools.lru_cache(maxsize=1)
def _jit_default_fixture():
    env = dict(os.environ)
    env.pop("PYTORCH_JIT", None)  # the point of this test is the *default*
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _JIT_DEFAULT_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"jit-default subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def test_torch_jit_script_defaults_to_returning_the_original_function():
    if not _ckpt_shim_available():
        return
    r = _jit_default_fixture()
    assert r["env_pytorch_jit"] == "0", r["env_pytorch_jit"]
    assert r["jit_enabled"] is False, r
    assert r["script_is_identity"] is True, r
    assert r["script_result"] == 4, r
    assert r["script_method_is_identity"] is True, r


_GPT_BIGCODE_IMPORT_SCRIPT = r"""
import json, sys, traceback
import torch
out = {}
try:
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.for_model(
        "gpt_bigcode", hidden_size=32, num_hidden_layers=1,
        num_attention_heads=4, num_key_value_heads=2, vocab_size=100,
        max_position_embeddings=64, intermediate_size=64,
    )
    from transformers.models.gpt_bigcode.modeling_gpt_bigcode import (
        GPTBigCodeForCausalLM,
    )
except BaseException:
    out["import_result"] = "FAILED: " + traceback.format_exc(limit=6)
else:
    out["import_result"] = "OK"
    out["class_name"] = GPTBigCodeForCausalLM.__name__
json.dump(out, sys.stdout)
"""


@functools.lru_cache(maxsize=1)
def _gpt_bigcode_import_fixture():
    env = dict(os.environ)
    env.pop("PYTORCH_JIT", None)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _GPT_BIGCODE_IMPORT_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"gpt-bigcode-import subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def test_gpt_bigcode_imports_now_that_scripting_defaults_to_unavailable():
    """The module-scope `@torch.jit.script` at
    `modeling_gpt_bigcode.py:54` used to raise `NotImplementedError:
    SourceRangeFactory.make_range` at import time (docs/TORCHSCRIPT.md).
    With scripting off by default, it degrades to a plain function and the
    module imports. This does not mean the architecture forwards -- it
    still needs `aten.tril` (docs/TORCHSCRIPT.md §6), which is not this
    file's territory -- only that the TorchScript wall specifically is gone.
    """
    if not _ckpt_shim_available() or _upstream_transformers is None:
        return
    r = _gpt_bigcode_import_fixture()
    assert r["import_result"] == "OK", r["import_result"]
    assert r["class_name"] == "GPTBigCodeForCausalLM", r


_JIT_EXPLICIT_ENABLE_SCRIPT = r"""
import json, sys, traceback
import torch
out = {}
try:
    @torch.jit.script
    def f(x: int) -> int:
        return x + 1
except BaseException as e:
    out["result"] = "REFUSED: " + type(e).__name__ + ": " + str(e)[:200]
else:
    out["result"] = "SCRIPTED"
json.dump(out, sys.stdout)
"""


@functools.lru_cache(maxsize=1)
def _jit_explicit_enable_fixture():
    env = dict(os.environ)
    env["PYTORCH_JIT"] = "1"  # explicit override must NOT be clobbered
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _JIT_EXPLICIT_ENABLE_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"jit-explicit-enable subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def test_an_explicit_pytorch_jit_1_is_not_clobbered_by_the_default():
    """`os.environ.setdefault` in `bootstrap.py` must not override a caller
    who explicitly asked for real scripting -- they still get the honest
    refusal (naming `SourceRangeFactory.make_range`, not a silent identity),
    which is the other half of the correctness bar this fix has to clear."""
    if not _ckpt_shim_available():
        return
    r = _jit_explicit_enable_fixture()
    assert r["result"].startswith("REFUSED:"), r
    assert "make_range" in r["result"], r


def test_the_whole_max_min_family_agrees_on_one_nan_rule():
    """Six ops, one predicate, and this is the fourth time it has been repaired.

    candle folds every reduction and every elementwise comparison with
    `|x, y| x < y`. Comparison against a NaN is false, so a NaN the accumulator
    does not *start* on is skipped. That single fact has produced four separate
    wrong answers in this repository, found four separate times:

        max.default / min.default   docs/E2E_REAL.md   value dropped
        max.other                   docs/SPELLINGS.md  second operand's NaN dropped
        amax                        docs/SEQLEN.md     avoided by construction, not repaired
        max.dim, argmax             docs/TRIL.md       value AND index dropped

    Written as one table over all of them rather than as six tests, because
    what keeps going wrong is not any one kernel -- it is that a new member of
    the family gets written against candle's primitive and inherits the fault
    silently. A seventh op added to this table with no NaN handling fails here.

    **Every case puts the NaN somewhere other than position 0 as well as at
    it.** A NaN in element 0 seeds candle's accumulator and survives even a
    kernel that does nothing about NaN, so a suite of `at=0` cases passes under
    the bug -- the hole docs/SEQLEN.md §7.12 found in `amax`'s own first test.
    """
    nan = float("nan")

    def t(flat, shape=None):
        return _C._tensor_from_flat(flat, shape or [len(flat)], dtype=_C.float32)

    for at, where in ((0, "first"), (1, "middle"), (3, "last")):
        flat = [1.0, 5.0, 2.0, 9.0]
        flat[at] = nan

        # Whole-tensor reductions: the value is NaN.
        for op in ("aten.max.default", "aten.min.default"):
            got = _C._aten_dispatch(op, t(flat)).tolist()
            assert math.isnan(got), f"{op} at={at} ({where}) gave {got}"

        # amax: value only, no index.
        got = _C._aten_dispatch("aten.amax.default", t(flat), [0], False).tolist()
        assert math.isnan(got), f"amax at={at} ({where}) gave {got}"

        # The dim reductions: BOTH halves of the pair. The index is the first
        # NaN's position (measured upstream), not the position of the largest
        # non-NaN -- and that distinction is the whole reason `max.dim` could
        # not simply be routed through `amax`, which has no index to give.
        for op in ("aten.max.dim", "aten.min.dim"):
            pair = _C._aten_dispatch(op, t(flat), 0, False)
            values, indices = pair[0].tolist(), pair[1].tolist()
            assert math.isnan(values), f"{op} at={at} ({where}) values={values}"
            assert indices == at, f"{op} at={at} ({where}) indices={indices}, want {at}"
            # ...and the same through `.values` / `.indices`, which is how a
            # caller actually spells it.
            assert math.isnan(pair.values.tolist()), op
            assert pair.indices.tolist() == at, op

        # argmax: index only, and the same index.
        got = _C._aten_dispatch("aten.argmax.default", t(flat), 0, False).tolist()
        assert got == at, f"argmax at={at} ({where}) gave {got}, want {at}"

        # The elementwise pair, with the NaN in each operand in turn. The
        # asymmetry is the thing: candle's `broadcast_maximum` propagates a NaN
        # in the *first* operand for free (nothing displaces an accumulator
        # that already holds one) and drops one that is only in the second, so
        # a suite testing only the first operand passes under the bug.
        clean = [1.0, 5.0, 2.0, 9.0]
        for op in ("aten.max.other", "aten.min.other"):
            for label, a, b in (("nan in self", flat, clean), ("nan in other", clean, flat)):
                got = _C._aten_dispatch(op, t(a), t(b)).tolist()
                assert math.isnan(got[at]), f"{op} {label} at={at}: {got}"
                # ...and only that lane. A correction that NaNs the whole
                # tensor passes every isnan check above and fails here.
                for i, v in enumerate(got):
                    if i != at:
                        assert not math.isnan(v), f"{op} {label} at={at} spread to {i}: {got}"

    # Two NaNs: the earlier index wins, for every op that returns one. This is
    # what separates "report the first NaN" from "report the last".
    two = [1.0, nan, 2.0, nan]
    for op in ("aten.max.dim", "aten.min.dim"):
        assert _C._aten_dispatch(op, t(two), 0, False)[1].tolist() == 1, op
    assert _C._aten_dispatch("aten.argmax.default", t(two), 0, False).tolist() == 1

    # And the boundary the correction must NOT cross: `-inf` is ordered, not
    # NaN. A repair keyed on "not finite" rather than on `x != x` passes every
    # case above and turns a fully masked attention row into NaN here.
    ninf = float("-inf")
    row = [ninf, ninf, -2.0, ninf]
    assert _C._aten_dispatch("aten.amax.default", t(row), [0], False).tolist() == -2.0
    pair = _C._aten_dispatch("aten.max.dim", t(row), 0, False)
    assert pair[0].tolist() == -2.0 and pair[1].tolist() == 2, pair
    assert _C._aten_dispatch("aten.max.default", t([ninf] * 4)).tolist() == ninf
    got = _C._aten_dispatch("aten.max.other", t([ninf, 1.0]), t([1.0, ninf])).tolist()
    assert got == [1.0, 1.0], got

    # A NaN in one slice and not the other: the correction is per-slice.
    rows = _C._tensor_from_flat([1.0, nan, 2.0, 4.0, 9.0, 3.0], [2, 3], dtype=_C.float32)
    pair = _C._aten_dispatch("aten.max.dim", rows, 1, False)
    values, indices = pair[0].tolist(), pair[1].tolist()
    assert math.isnan(values[0]) and values[1] == 9.0, values
    assert indices == [1, 1], indices


def test_the_two_stale_sdpa_refusals_no_longer_claim_a_missing_kernel():
    """A refusal that names a kernel must be re-derived, not just re-worded.

    `scaled_dot_product_attention` refused two inputs -- `dropout_p != 0` and a
    non-4-D query -- and both messages said `aten._safe_softmax.default` had no
    kernel. It has had one, golden-compared, since docs/SDPA.md. The
    architecture that stayed blocked for weeks in docs/TORCHSCRIPT.md was
    blocked by exactly this shape of mistake, and the bool-mask branch of this
    same function already carries a note about being the first instance of it.

    So this asserts the *claim*, not the wording: for every kernel a refusal
    names as present, it must be in `_aten_implemented()`; for every one it
    names as absent, it must not be. A refusal message that drifts out of
    agreement with the artefact fails here rather than sitting unread.
    """
    if not _ckpt_shim_available():
        return
    r = _sdpa_refusal_fixture()
    implemented = set(_C._aten_implemented()) | set(_C._aten_all_implemented())

    for label in ("dropout", "three_d"):
        message = r[label]
        assert message.startswith("NotImplementedError"), f"{label}: {message}"
        assert "math backend" in message, f"{label}: {message}"
        # Nothing in either message may say a kernel is missing when it is not.
        assert "_safe_softmax.default; it has no kernel" not in message, message
        assert "aten._safe_softmax.default, " not in message, message

    # The positive claims, checked against the artefact.
    assert "aten._safe_softmax.default is " in r["dropout"], r["dropout"]
    assert "aten._safe_softmax.default" in implemented
    assert "_safe_softmax" in r["three_d"], r["three_d"]
    for op in ("aten.mul.Scalar", "aten.expand.default", "aten.view.default", "aten.bmm.default"):
        assert op in implemented, f"{op} named as implemented by the 3-D refusal, but is not"
    # ...and the negative ones.
    for op in ("aten.bernoulli_.float", "aten.div_.Scalar"):
        assert op not in implemented, (
            f"{op} now has a kernel -- the dropout refusal names it as missing "
            f"and has gone stale again"
        )

    # The gap the stale text hid: `torch._safe_softmax` is a real upstream name
    # (`hasattr(torch, '_safe_softmax')` is True on 2.13.0) for a leaf op this
    # shim implements, and it refused with "no table entry" the whole time.
    # docs/ARCH20.md §9 filed it under "no such public function upstream",
    # which is how a name nothing calls stops correcting the text about it.
    assert r["safe_softmax_spelling"] is not None, r
    rows = r["safe_softmax_spelling"]
    for row in rows:
        assert abs(sum(row) - 1.0) < 1e-5, rows
    # The fully-masked row is what distinguishes this op from `_softmax`: it
    # answers zeros where `_softmax` answers NaN. That difference is the entire
    # reason the SDPA math fallback refuses instead of substituting.
    assert r["safe_softmax_masked_row"] == [0.0, 0.0, 0.0], r["safe_softmax_masked_row"]


_SDPA_REFUSAL_SCRIPT = r"""
import json, sys
import torch

out = {}
q = torch.randn(1, 2, 3, 4)
try:
    torch.nn.functional.scaled_dot_product_attention(q, q, q, dropout_p=0.5)
    out["dropout"] = "ACCEPTED"
except NotImplementedError as e:
    out["dropout"] = "NotImplementedError: %s" % e
q3 = torch.randn(2, 3, 4)
try:
    torch.nn.functional.scaled_dot_product_attention(q3, q3, q3)
    out["three_d"] = "ACCEPTED"
except NotImplementedError as e:
    out["three_d"] = "NotImplementedError: %s" % e

try:
    out["safe_softmax_spelling"] = torch._safe_softmax(
        torch.tensor([[1.0, 5.0, 2.0], [9.0, 0.0, 3.0]]), 1).tolist()
except NotImplementedError as e:
    out["safe_softmax_spelling"] = None
try:
    ninf = float("-inf")
    out["safe_softmax_masked_row"] = torch._safe_softmax(
        torch.tensor([[ninf, ninf, ninf]]), 1).tolist()[0]
except NotImplementedError as e:
    out["safe_softmax_masked_row"] = None
json.dump(out, sys.stdout)
"""


@functools.lru_cache(maxsize=1)
def _sdpa_refusal_fixture():
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", _SDPA_REFUSAL_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"sdpa-refusal subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def test_max_min_dim_return_types_are_named_for_their_own_op():
    """`min.dim`'s pair must not print as `max(...)`.

    Upstream returns a structseq whose type is `torch.return_types.min`; this
    shim returns a `collections.namedtuple` (docs/TENSORBASE.md says why), and
    the one thing that has to survive the substitution is the *name*, because
    it is what `repr()` shows and what a traceback shows. Sharing one cached
    namedtuple between the two overloads -- the obvious economy, since the
    field names are identical -- would make every `min` result claim to be a
    `max`.
    """
    a = _C._tensor_from_flat([1.0, 5.0, 2.0, 9.0], [2, 2], dtype=_C.float32)
    hi = _C._aten_dispatch("aten.max.dim", a, 1, False)
    lo = _C._aten_dispatch("aten.min.dim", a, 1, False)
    assert type(hi).__name__ == "max", type(hi).__name__
    assert type(lo).__name__ == "min", type(lo).__name__
    assert type(hi) is not type(lo)
    assert hi._fields == ("values", "indices") == lo._fields
    assert hi.values.tolist() == [5.0, 9.0] and hi.indices.tolist() == [1, 1]
    assert lo.values.tolist() == [1.0, 2.0] and lo.indices.tolist() == [0, 0]
    assert "min(" in repr(lo), repr(lo)


def test_tril_and_triu_zero_by_selecting_not_by_multiplying():
    """The mistake that passes every case built from small integers.

    A 0/1 mask of the input's dtype and a broadcast multiply is the obvious
    implementation of "zero one side of the diagonal", and `nan * 0` is `nan`
    while `inf * 0` is `nan` too. Upstream zeroes those positions like any
    other -- measured, `tril([[1, nan], [inf, -inf]])` is `[[1, 0], [inf,
    -inf]]` -- so a multiply turns a masked-out `-inf` into a NaN and every
    test written with `ones` and `arange` goes on passing.

    Also pins the sign convention in both directions, which is the other thing
    here that fails silently: a swapped `tril`/`triu` produces the same shape,
    the same dtype and the same magnitudes.
    """
    nan, pinf, ninf = float("nan"), float("inf"), float("-inf")
    m = _C._tensor_from_flat(
        [1.0, nan, pinf, ninf, nan, 2.0, ninf, pinf, 0.0], [3, 3], dtype=_C.float32
    )

    def cell(t, i, j):
        return t.tolist()[i][j]

    lower = _C._aten_dispatch("aten.tril.default", m, 0)
    # Above the diagonal: zeroed, whatever was there.
    for i, j in ((0, 1), (0, 2), (1, 2)):
        assert cell(lower, i, j) == 0.0, f"tril kept ({i},{j}): {cell(lower, i, j)}"
    # On and below: untouched, NaN and infinities included.
    assert cell(lower, 0, 0) == 1.0
    assert math.isnan(cell(lower, 1, 1))
    assert cell(lower, 1, 0) == ninf and cell(lower, 2, 0) == ninf
    assert cell(lower, 2, 1) == pinf

    upper = _C._aten_dispatch("aten.triu.default", m, 0)
    for i, j in ((1, 0), (2, 0), (2, 1)):
        assert cell(upper, i, j) == 0.0, f"triu kept ({i},{j}): {cell(upper, i, j)}"
    assert math.isnan(cell(upper, 0, 1))
    assert cell(upper, 0, 2) == pinf and cell(upper, 1, 2) == 2.0

    # The sign convention, on data where every position is distinguishable.
    x = _C._tensor_from_flat([float(v) for v in range(1, 10)], [3, 3], dtype=_C.float32)
    assert _C._aten_dispatch("aten.tril.default", x, 0).tolist() == [
        [1.0, 0.0, 0.0], [4.0, 5.0, 0.0], [7.0, 8.0, 9.0]]
    assert _C._aten_dispatch("aten.tril.default", x, 1).tolist() == [
        [1.0, 2.0, 0.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]
    assert _C._aten_dispatch("aten.tril.default", x, -1).tolist() == [
        [0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [7.0, 8.0, 0.0]]
    assert _C._aten_dispatch("aten.triu.default", x, 0).tolist() == [
        [1.0, 2.0, 3.0], [0.0, 5.0, 6.0], [0.0, 0.0, 9.0]]
    assert _C._aten_dispatch("aten.triu.default", x, 1).tolist() == [
        [0.0, 2.0, 3.0], [0.0, 0.0, 6.0], [0.0, 0.0, 0.0]]
    assert _C._aten_dispatch("aten.triu.default", x, -1).tolist() == [
        [1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [0.0, 8.0, 9.0]]
    # Unbounded in both directions -- not clamped, not an index.
    assert _C._aten_dispatch("aten.tril.default", x, 100).tolist() == x.tolist()
    assert _C._aten_dispatch("aten.tril.default", x, -100).tolist() == [[0.0] * 3] * 3
    assert _C._aten_dispatch("aten.triu.default", x, 100).tolist() == [[0.0] * 3] * 3
    assert _C._aten_dispatch("aten.triu.default", x, -100).tolist() == x.tolist()

    # A transposed (non-contiguous) receiver is masked by position in the
    # *matrix*, not by position in memory.
    #
    # **This assertion cannot currently fail, and saying so is the point.**
    # Deleting the kernel's `.contiguous()` was injected as a fault and every
    # gate stayed green (docs/TRIL.md §5, fault 3) -- candle's `WCond` already
    # falls back to `strided_index()` for a non-contiguous operand. It is here
    # as coverage of the shape, not as a check of the normalisation; if candle
    # ever loses that fallback this is where it shows, and until then nobody
    # should read a green run as evidence the `.contiguous()` is doing work.
    xt = _C._aten_dispatch("aten.t.default", x)
    assert _C._aten_dispatch("aten.tril.default", xt, 0).tolist() == [
        [1.0, 0.0, 0.0], [2.0, 5.0, 0.0], [3.0, 6.0, 9.0]]

    # Rank < 2 refuses with upstream's wording, on both names.
    for op, name in (("aten.tril.default", "tril"), ("aten.triu.default", "triu")):
        for flat, shape in (([1.0], []), ([1.0, 2.0], [2])):
            small = _C._tensor_from_flat(flat, shape, dtype=_C.float32)
            try:
                _C._aten_dispatch(op, small, 0)
            except RuntimeError as e:
                assert "at least 2 dimensions" in str(e), str(e)
                assert str(e).startswith(name), str(e)
            else:
                raise AssertionError(f"{op} accepted a rank-{len(shape)} input")


# --- docs/KERNELS26.md: the kernels that stopped six architectures ----------
#
# Same split as the `_SPELL_ROAD_SCRIPT` section above, and for the same
# reason: `tools/golden/compare.py` reaches a kernel by its *dispatch key*, so
# it is structurally blind to a kernel that exists and has no `torch.<name>`
# or `tensor.<name>` route into it. Every kernel this round adds gets both --
# the raw key here (fast, in-process, against the bare artefact) and the two
# Python spellings in `_KERNELS26_ROAD_SCRIPT` (a real `import torch` through
# the vendored tree).
#
# Expected values are computed in plain Python (`math.sqrt`, `%`) rather than
# by importing upstream torch, so this file still runs on an interpreter with
# no torch installed, matching the module docstring.


def test_sqrt_is_a_leaf_kernel_with_ieee_domain_and_sign():
    """`aten.sqrt.default` -- the kernel that stopped `deberta` and
    `deberta_v2` (docs/ARCH26.md §1), both of which reach `torch.sqrt` before
    any weight multiplies.

    Three things are asserted that a `pow(x, 0.5)` composite would get wrong,
    and they are the reason this is a kernel rather than a `bootstrap.py`
    one-liner:

      * `sqrt(-0.0)` is `-0.0`. IEEE-754 keeps the sign of zero. Checked on
        the sign bit via `math.copysign`, because `-0.0 == 0.0` is true and no
        value comparison can see it.
      * `sqrt(-inf)` is NaN, and `sqrt(+inf)` is `+inf`. candle's own
        `Tensor::pow` is `exp(e * log(b))`, which answers NaN for the second
        of those.
      * the dtype rule is `unary_float`'s -- a float keeps its own width
        (`float16` in, `float16` out, not widened) and anything else becomes
        the default float.
    """
    d = _C._aten_dispatch

    got = d("aten.sqrt.default", _t([1.0, 4.0, 9.0, 16.0], [2, 2])).tolist()
    assert got == [[1.0, 2.0], [3.0, 4.0]], got
    got = d("aten.sqrt.default", _t([2.0, 3.0], [2])).tolist()
    for g, src in zip(got, (2.0, 3.0)):
        assert abs(g - math.sqrt(src)) < 1e-6, (got, src)

    # The domain, all four corners.
    domain = d(
        "aten.sqrt.default",
        _t([-1.0, float("inf"), float("-inf"), float("nan")], [4]),
    ).tolist()
    assert math.isnan(domain[0]), domain
    assert domain[1] == float("inf"), domain
    assert math.isnan(domain[2]), domain  # -inf -> NaN, NOT -inf
    assert math.isnan(domain[3]), domain

    # The signed zero, on its sign bit.
    for src, want_negative in ((-0.0, True), (0.0, False)):
        z = d("aten.sqrt.default", _t([src], [1])).tolist()[0]
        assert z == 0.0, (src, z)
        assert (math.copysign(1.0, z) < 0) is want_negative, (src, z)

    # The dtype rule, both halves.
    for dtype in (_C.float32, _C.float64, _C.float16, _C.bfloat16):
        out = d("aten.sqrt.default", _t([4.0], [1], dtype))
        assert out.dtype == dtype, (dtype, out.dtype)
    for dtype in (_C.int64, _C.int32, _C.int16, _C.uint8, _C.bool):
        out = d("aten.sqrt.default", _t([1], [1], dtype))
        assert out.dtype == _C.float32, (dtype, out.dtype)
    try:
        _C._set_default_dtype(_C.float64)
        assert d("aten.sqrt.default", _t([4], [1], _C.int64)).dtype == _C.float64
        assert d("aten.sqrt.default", _t([4.0], [1], _C.float32)).dtype == _C.float32
    finally:
        _C._set_default_dtype(_C.float32)

    # 0-d and empty both answer rather than refusing.
    assert d("aten.sqrt.default", _t([9.0], [])).item() == 3.0
    assert list(d("aten.sqrt.default", _t([], [0])).shape) == [0]

    # meta agrees with the dense kernel about shape and dtype.
    meta = _C.device("meta")
    for dtype, want in ((_C.float16, _C.float16), (_C.int64, _C.float32)):
        out = d(
            "aten.sqrt.default",
            d("aten.empty.memory_format", [2, 3], dtype, device=meta),
        )
        assert out.is_meta is True, dtype
        assert tuple(out.shape) == (2, 3), (dtype, tuple(out.shape))
        assert out.dtype == want, (dtype, out.dtype)


def test_sqrt_is_reachable_by_name_not_only_by_dispatch_key():
    """The half of the surface `tools/golden/compare.py` cannot see.

    `deberta`'s wall was `torch.sqrt(...)` -- the *spelling* -- and a kernel
    with no `overloads.json` entry would leave that wall exactly where it was
    while every golden case passed. Both tables are asserted here; the
    end-to-end route through a real `import torch` is in
    `test_kernels26_road_through_the_vendored_tree`.
    """
    assert _C._shim_overloads["sqrt"] == [
        "aten.sqrt.out",
        "aten.sqrt.default",
    ], _C._shim_overloads.get("sqrt")
    assert _C._shim_methods["sqrt"] == ["aten.sqrt.default"], _C._shim_methods.get("sqrt")
    # `repeat` is a member and NOT a free function: measured on upstream
    # 2.13.0, `hasattr(torch, "repeat")` is False -- there is only
    # `Tensor.repeat` and the unrelated `torch.repeat_interleave`. Adding a
    # `torch.repeat` here would invent a name upstream does not have, so the
    # absence is asserted rather than left to chance.
    assert _C._shim_methods["repeat"] == ["aten.repeat.default"], _C._shim_methods.get("repeat")
    assert "repeat" not in _C._shim_overloads, "upstream has no torch.repeat"
    # `remainder` is both a free function and a member, and `__mod__` is a
    # third spelling of the same two keys. The order matters: the resolver
    # picks `.Tensor` when handed a tensor and `.Scalar` otherwise, so a table
    # that listed only one of them would silently answer the wrong overload
    # for half the calls `x % y` can make.
    for name in ("remainder", "__mod__"):
        assert _C._shim_methods[name] == [
            "aten.remainder.Tensor",
            "aten.remainder.Scalar",
        ], (name, _C._shim_methods.get(name))
    assert _C._shim_overloads["remainder"] == [
        "aten.remainder.Scalar_out",
        "aten.remainder.Tensor_out",
        "aten.remainder.Tensor",
        "aten.remainder.Scalar",
    ], _C._shim_overloads.get("remainder")
    # `__rmod__` is deliberately absent: `3 % x` is `aten.remainder.Scalar_Tensor`,
    # a distinct overload with its own promotion rule, and it is not implemented
    # here. Asserted so that adding `__rmod__` without the kernel -- which would
    # make `3 % x` refuse by a name that resolves to nothing -- fails here.
    assert "__rmod__" not in _C._shim_methods
    # `div`'s four overloads were already in both tables before the two
    # `_mode` kernels existed -- only the kernels were missing, which is why
    # the sweep's wall was an `aten op not implemented` rather than an
    # overload-resolution failure. The ORDER is what makes `torch.div(a, b)`
    # and `torch.div(a, b, rounding_mode=...)` land on different keys: the
    # `_mode` entries come first and are skipped when the keyword is absent,
    # because `str? rounding_mode` is keyword-only *and has no default* in
    # native_functions.yaml. Reordering these so `div.Tensor` came first would
    # make every `rounding_mode=` call silently answer true division.
    assert _C._shim_methods["div"] == [
        "aten.div.Tensor_mode",
        "aten.div.Scalar_mode",
        "aten.div.Tensor",
        "aten.div.Scalar",
    ], _C._shim_methods.get("div")
    for key in ("aten.div.Tensor_mode", "aten.div.Scalar_mode"):
        assert key in _C._aten_all_implemented(), key
    # `.out` is in the table with no kernel behind it, exactly as `rsqrt.out`
    # is, so that `torch.sqrt(x, out=y)` refuses *by the right name* instead of
    # falling through to "no table entry for this op".
    try:
        _C._aten_dispatch("aten.sqrt.out", _t([1.0], [1]), out=_t([0.0], [1]))
    except NotImplementedError as e:
        assert "aten.sqrt.out" in str(e), str(e)
    else:
        raise AssertionError("aten.sqrt.out has no kernel and must say so")


def test_repeat_tiles_rather_than_broadcasts():
    """`aten.repeat.default` -- the kernel `sqrt` uncovered, and the one
    docs/ARCH26.md §8 found recurring across four of the six architectures.

    `repeat` is tiling and `expand` is broadcasting; the assertions here are
    the places the two are confusable, plus the two places candle's own
    `Tensor::repeat` disagrees with upstream:

      * a repeat of **0** must produce an empty dimension. candle's loop is
        `if repeat > 1 { cat }`, so it treats 0 as 1 and returns the input.
      * `len(repeats) < rank` must **refuse**. candle silently uses the
        tensor as-is and then concatenates along the wrong axes.
    """
    d = _C._aten_dispatch

    # 1-D, same rank.
    assert d("aten.repeat.default", _t([1.0, 2.0, 3.0], [3]), [2]).tolist() == [
        1.0, 2.0, 3.0, 1.0, 2.0, 3.0]
    # 1-D raised to 2-D: the LAST repeat multiplies the existing dimension.
    out = d("aten.repeat.default", _t([1.0, 2.0, 3.0], [3]), [2, 3])
    assert list(out.shape) == [2, 9], list(out.shape)
    assert out.tolist() == [[1.0, 2.0, 3.0] * 3] * 2, out.tolist()
    # 2-D, same rank -- rows tile down, columns tile across.
    out = d("aten.repeat.default", _t([1.0, 2.0, 3.0, 4.0], [2, 2]), [2, 3])
    assert out.tolist() == [
        [1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
        [3.0, 4.0, 3.0, 4.0, 3.0, 4.0],
    ] * 2, out.tolist()
    # 0-D.
    assert d("aten.repeat.default", _t([5.0], []), [3]).tolist() == [5.0, 5.0, 5.0]
    assert list(d("aten.repeat.default", _t([5.0], []), []).shape) == []

    # A repeat of zero is an empty dimension, not a no-op.
    assert list(d("aten.repeat.default", _t([1.0, 2.0, 3.0], [3]), [0]).shape) == [0]
    assert list(d("aten.repeat.default", _t([1.0, 2.0, 3.0, 4.0], [2, 2]), [0, 2]).shape) == [0, 4]
    assert list(d("aten.repeat.default", _t([1.0, 2.0, 3.0, 4.0], [2, 2]), [2, 0]).shape) == [4, 0]

    # dtype passes through untouched, `bool` included -- `repeat` is data
    # movement and does not promote.
    for dtype in (_C.float32, _C.float64, _C.float16, _C.bfloat16,
                  _C.int64, _C.int32, _C.int16, _C.uint8, _C.bool):
        out = d("aten.repeat.default", _t([1, 0], [2], dtype), [2])
        assert out.dtype == dtype, (dtype, out.dtype)
        assert list(out.shape) == [4], (dtype, list(out.shape))

    # A non-contiguous input is tiled by its logical order, not its storage.
    strided = d("aten.t.default", d("aten.view.default", _t(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [6]), [2, 3]))
    assert strided.tolist() == [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]
    assert d("aten.repeat.default", strided, [2, 1]).tolist() == [
        [0.0, 3.0], [1.0, 4.0], [2.0, 5.0], [0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]

    # Upstream's two refusals, with upstream's own wording.
    try:
        d("aten.repeat.default", _t([1.0, 2.0, 3.0, 4.0], [2, 2]), [2])
    except RuntimeError as e:
        assert "can not be smaller than number of dimensions" in str(e), str(e)
    else:
        raise AssertionError("fewer repeats than dimensions must refuse")
    for repeats, want in (([-1, 2], "-2"), ([2, -1], "-2")):
        try:
            d("aten.repeat.default", _t([1.0, 2.0, 3.0, 4.0], [2, 2]), repeats)
        except RuntimeError as e:
            assert "negative dimension" in str(e), str(e)
            assert want in str(e), (repeats, str(e))
        else:
            raise AssertionError(f"repeat({repeats}) must refuse")


def test_remainder_follows_the_sign_of_the_divisor_not_the_dividend():
    """`aten.remainder.{Scalar,Tensor}` -- `sam3_video`'s wall
    (docs/ARCH26.md §5), reached through `TensorBase.__mod__` inside a ViT
    rotary embedding's `__init__`.

    **This is the op where `fmod` is the wrong answer.** `remainder` takes the
    sign of the *divisor*; `fmod` (and C's `%`, and Rust's) takes the sign of
    the *dividend*. The two agree in exactly half the sign quadrants, so a
    case set with positive operands -- or with only one sign varying -- cannot
    tell them apart. Both halves are asserted: where the conventions must
    differ, and where they must agree.

    Three corners beyond the sign rule, each measured on upstream 2.13.0:

      * `remainder(-0.0, 3.0)` is `-0.0`, not `+0.0`. Python's own
        `-0.0 % 3.0` **is** `+0.0`, so this is a case where "spell it the way
        Python spells it" is wrong, and `-0.0 == 0.0` hides it.
      * a float divisor of `0.0` gives NaN; an **integral** one raises
        `RuntimeError('ZeroDivisionError')`. Same op, different category,
        different kind of answer.
      * `remainder(int64_min, -1)` is `0`. Rust's `%` panics on that pair.
    """
    d = _C._aten_dispatch

    # The four sign quadrants, written out rather than derived, so a shared
    # helper cannot make both sides wrong in the same way.
    quadrants = {
        (7, 3): 1, (7, -3): -2, (-7, 3): 2, (-7, -3): -1,
        (5, 2): 1, (5, -2): -1, (-5, 2): 1, (-5, -2): -1,
        (6, 3): 0, (6, -3): 0, (-6, 3): 0, (-6, -3): 0,
    }
    for (a, b), want in quadrants.items():
        got = d("aten.remainder.Scalar", _t([float(a)], [1]), float(b)).tolist()[0]
        assert got == float(want), (a, b, got, want)
        got_i = d("aten.remainder.Scalar", _t([a], [1], _C.int64), b).tolist()[0]
        assert got_i == want, (a, b, got_i, want)
        got_t = d(
            "aten.remainder.Tensor", _t([float(a)], [1]), _t([float(b)], [1])
        ).tolist()[0]
        assert got_t == float(want), (a, b, got_t, want)

    # Opposite signs: `remainder` and `fmod` MUST disagree. `fmod` is computed
    # here by `math.fmod`, not taken from the shim.
    for a, b in ((7, -3), (-7, 3), (5, -2), (-5, 2)):
        rem = d("aten.remainder.Scalar", _t([float(a)], [1]), float(b)).tolist()[0]
        assert rem != math.fmod(a, b), (
            f"remainder({a}, {b}) == fmod({a}, {b}) == {rem} -- the kernel is "
            "following the sign of the dividend"
        )
        assert (rem < 0) == (b < 0), (a, b, rem)
    # Same signs: they MUST agree. The other half of the same claim.
    for a, b in ((7, 3), (-7, -3), (5, 2), (-5, -2)):
        rem = d("aten.remainder.Scalar", _t([float(a)], [1]), float(b)).tolist()[0]
        assert rem == math.fmod(a, b), (a, b, rem, math.fmod(a, b))

    # The signed zero, on its sign bit.
    z = d("aten.remainder.Scalar", _t([-0.0], [1]), 3.0).tolist()[0]
    assert z == 0.0 and math.copysign(1.0, z) < 0, (
        f"remainder(-0.0, 3.0) must be -0.0, got {z!r} with sign "
        f"{math.copysign(1.0, z)} -- Python's own -0.0 % 3.0 is +0.0"
    )
    z = d("aten.remainder.Scalar", _t([0.0], [1]), 3.0).tolist()[0]
    assert z == 0.0 and math.copysign(1.0, z) > 0, z

    # Non-finite operands, both sides.
    inf, nan = float("inf"), float("nan")
    assert math.isnan(d("aten.remainder.Scalar", _t([inf], [1]), 3.0).tolist()[0])
    assert math.isnan(d("aten.remainder.Scalar", _t([nan], [1]), 3.0).tolist()[0])
    assert d("aten.remainder.Scalar", _t([5.0], [1]), inf).tolist()[0] == 5.0
    assert d("aten.remainder.Scalar", _t([5.0], [1]), -inf).tolist()[0] == -inf
    assert d("aten.remainder.Scalar", _t([-5.0], [1]), inf).tolist()[0] == inf

    # Division by zero: NaN for floats, a raise for integers.
    assert math.isnan(d("aten.remainder.Scalar", _t([5.0], [1]), 0.0).tolist()[0])
    try:
        d("aten.remainder.Scalar", _t([5], [1], _C.int64), 0)
    except RuntimeError as e:
        assert "ZeroDivisionError" in str(e), str(e)
    else:
        raise AssertionError("integral remainder by zero must raise")

    # The overflow pair Rust's `%` panics on.
    assert d("aten.remainder.Scalar", _t([-(2 ** 63)], [1], _C.int64), -1).tolist()[0] == 0

    # A scalar is narrowed INTO the tensor's dtype before the arithmetic:
    # `uint8(200) % -3` is `200`, because -3 becomes 253 and 200 % 253 is 200.
    assert d("aten.remainder.Scalar", _t([200], [1], _C.uint8), -3).tolist()[0] == 200

    # dtype: the wrapped-number rule on `Scalar`, `promote_types` on `Tensor`.
    assert d("aten.remainder.Scalar", _t([5], [1], _C.int64), 3).dtype == _C.int64
    assert d("aten.remainder.Scalar", _t([5], [1], _C.int64), 3.0).dtype == _C.float32
    assert d("aten.remainder.Scalar", _t([5.0], [1], _C.float16), 3).dtype == _C.float16
    assert d(
        "aten.remainder.Tensor", _t([5], [1], _C.int32), _t([3.0], [1], _C.float32)
    ).dtype == _C.float32

    # Broadcasting -- the sign correction has to run after it, not before.
    got = d("aten.remainder.Tensor", _t([7.0, 8.0], [2, 1]), _t([3.0, -3.0], [2])).tolist()
    assert got == [[1.0, -2.0], [2.0, -1.0]], got

    # Bool: upstream's own refusal on `Tensor`, this shim's on `Scalar`.
    bt = _t([1, 0], [2], _C.bool)
    try:
        d("aten.remainder.Tensor", bt, bt)
    except NotImplementedError as e:
        assert "remainder_cpu" in str(e) and "Bool" in str(e), str(e)
    else:
        raise AssertionError("remainder of two bool tensors must refuse, as upstream does")
    try:
        d("aten.remainder.Scalar", bt, 2)
    except NotImplementedError as e:
        assert "fast-path ladder" in str(e), str(e)
    else:
        raise AssertionError("the bool-with-scalar gap must refuse, not invent a dtype")


def test_the_legacy_tensor_size_constructor_allocates_and_the_data_form_refuses():
    """`torch.Tensor(n)` -- `sew_d`'s wall (docs/ARCH26.md §4).

    Three forms wear the same name upstream and this asserts all three,
    because the whole risk in implementing one of them is answering a
    *different* one by accident:

        TensorBase(existing)   re-wrap, sharing the candle tensor
        TensorBase(2, 3)       uninitialised storage of that shape
        TensorBase([3, 4])     build from data -- a (2,) tensor, NOT a (3, 4)
                               empty one. Still refused; asserted refused.

    The third is the trap: `[3, 4]` looks like a size list and is not one.
    A constructor that accepted it as a shape would silently produce a
    `(3, 4)` tensor of zeros where upstream produces `tensor([3., 4.])`.
    """
    # The re-wrap form still shares, which is what `_make_subclass` depends on.
    base = _t([1.0, 2.0, 3.0], [3])
    rewrapped = _C.TensorBase(base)
    _C._aten_dispatch("aten.fill_.Scalar", rewrapped, 9.0)
    assert base.tolist() == [9.0, 9.0, 9.0], (
        "TensorBase(existing) must re-wrap, not copy -- `_make_subclass` relies on it"
    )

    # The size form, one dimension and several.
    assert list(_C.TensorBase(3).shape) == [3]
    assert list(_C.TensorBase(3, 4).shape) == [3, 4]
    assert list(_C.TensorBase(2, 3, 4).shape) == [2, 3, 4]
    # Zero arguments is `(0,)`, not `()` -- measured on upstream.
    assert list(_C.TensorBase().shape) == [0]
    assert list(_C.TensorBase(0).shape) == [0]

    # dtype is the default float, read at call time so it moves with the
    # setter exactly as upstream's does.
    assert _C.TensorBase(3).dtype == _C.float32
    try:
        _C._set_default_dtype(_C.float64)
        assert _C.TensorBase(3).dtype == _C.float64
    finally:
        _C._set_default_dtype(_C.float32)

    # A negative size refuses with upstream's wording.
    try:
        _C.TensorBase(-1)
    except RuntimeError as e:
        assert "negative dimension" in str(e), str(e)
    else:
        raise AssertionError("TensorBase(-1) must refuse")

    # The data form is a DIFFERENT function and must not be answered as a
    # shape. Refusing is the documented decision; answering `(3, 4)` would be
    # the silent wrong answer.
    try:
        _C.TensorBase([3, 4])
    except NotImplementedError as e:
        assert "third form" in str(e), str(e)
    else:
        raise AssertionError(
            "TensorBase([3, 4]) is torch's build-from-data form (a (2,) tensor of "
            "3.0 and 4.0), not a size list -- answering it as a shape would be worse "
            "than refusing"
        )

    # `sew_d`'s line verbatim, which is the reason this exists at all.
    spec_embed = _C.TensorBase(32)
    _C._aten_dispatch("aten.uniform_.default", spec_embed, 0.0, 1.0)
    assert list(spec_embed.shape) == [32]
    assert all(0.0 <= v < 1.0 for v in spec_embed.tolist()), spec_embed.tolist()

    # **Two calls must return independent storage.** Nothing above can see this:
    # the shape, dtype and refusal assertions all pass for a constructor that
    # handed out one shared buffer, and so does the `.uniform_()` check, because
    # a shared buffer is still filled with values in range. The values
    # themselves are the one thing that cannot be compared against upstream here
    # (upstream's are uninitialised), so *independence* is what stands in for
    # them -- the property a caller actually depends on when it writes into the
    # result. `masked_spec_embed` is constructed once per model, but two models
    # in one process must not share it.
    first, second = _C.TensorBase(4), _C.TensorBase(4)
    _C._aten_dispatch("aten.fill_.Scalar", first, 1.0)
    _C._aten_dispatch("aten.fill_.Scalar", second, 2.0)
    assert first.tolist() == [1.0] * 4, first.tolist()
    # Checked by writing into one and reading the other rather than by
    # comparing `data_ptr()`, which `TensorBase` does not expose at this level.
    assert second.tolist() == [2.0] * 4, (
        "two TensorBase(n) calls must not share storage", second.tolist())
    assert first.tolist() == [1.0] * 4, (
        "the second call must not have overwritten the first", first.tolist())


def test_set_from_a_tensor_aliases_where_set_from_a_storage_copies():
    """`aten.set_.source_Tensor` -- the wall `vits` AND `sew_d` both stopped
    on (docs/ARCH26.md §2), reached through
    `torch.nn.utils.parametrizations.weight_norm`.

    **The two forms of `set_` in this shim have opposite aliasing behaviour,
    and that is not an inconsistency.** The storage form copies, because
    candle owns its memory and a `Storage` is bytes held separately (see
    `set_`'s own doc comment, and docs/CKPT.md §4). The tensor form aliases,
    because `Repr::Dense` *is* a candle tensor and a candle clone is an `Arc`
    clone of the same storage -- so it gets upstream's semantics for free.
    Both directions are asserted here, next to each other, because "one of
    these copies and the other does not" is the kind of claim that rots.
    """
    d = _C._aten_dispatch

    # a.set_(b): a adopts b's shape, and the two share storage afterwards.
    a = _t([0.0, 0.0, 0.0], [3])
    b = _t([0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [2, 3])
    returned = a.set_(b)
    assert returned is a, "set_ is in place and returns the receiver"
    assert list(a.shape) == [2, 3], list(a.shape)
    assert a.tolist() == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    # The aliasing half: a write into the source is visible through the target.
    d("aten.fill_.Scalar", b, 9.0)
    assert a.tolist() == [[9.0, 9.0, 9.0], [9.0, 9.0, 9.0]], (
        "set_(tensor) must alias, as upstream's does -- got a copy"
    )

    # A non-contiguous source keeps its layout rather than being flattened.
    strided = d("aten.t.default", d("aten.view.default", _t(
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0], [6]), [2, 3]))
    target = _t([0.0], [1])
    target.set_(strided)
    assert list(target.shape) == [3, 2], list(target.shape)
    assert target.tolist() == [[0.0, 3.0], [1.0, 4.0], [2.0, 5.0]]

    # set_() with no arguments empties in place, keeping the dtype.
    e = _t([1.0, 2.0, 3.0, 4.0], [4], _C.float64)
    e.set_()
    assert list(e.shape) == [0], list(e.shape)
    assert e.dtype == _C.float64, e.dtype

    # A dtype mismatch refuses, with upstream's own C++ type names -- which
    # are a FOURTH spelling of the dtype set (`long long`, not `int64_t` and
    # not `Long`), so this asserts the exact string rather than a substring.
    try:
        _t([0.0, 0.0], [2], _C.float32).set_(_t([1, 2, 3], [3], _C.int64))
    except RuntimeError as err:
        assert str(err) == (
            "Could not set tensor of type long long to a tensor of type float"
        ), str(err)
    else:
        raise AssertionError("set_ must refuse a dtype mismatch, as upstream does")

    # The four-argument tensor form is a distinct overload
    # (`source_Tensor_storage_offset`) and refuses by that name rather than
    # silently ignoring the extra arguments.
    try:
        _t([0.0], [1]).set_(_t([1.0, 2.0], [2]), 0, [2], [1])
    except NotImplementedError as err:
        assert "source_Tensor_storage_offset" in str(err), str(err)
    else:
        raise AssertionError("the storage-offset tensor overload must refuse by name")

    # The storage form is unchanged and still copies -- exercised through the
    # vendored tree by `test_checkpoint_road_...`, whose `unfilled_refused`
    # probe reaches this same method with an `UntypedStorage`. What is checked
    # here is only that the tensor arm did not swallow it: a storage argument
    # must still reach the storage path and refuse for the storage path's
    # reason, not be mistaken for a tensor.
    try:
        _t([], [0]).set_(object(), 0, [4], [1])
    except NotImplementedError as err:
        assert "UntypedStorage or a tensor" in str(err), str(err)
    else:
        raise AssertionError("a non-tensor, non-storage source must refuse")


# The end-to-end half: a real `import torch` through the vendored tree, calling
# each new name the way a model does. Grows one block per kernel this round.

_KERNELS26_ROAD_SCRIPT = r"""
import json, math, sys
import torch

out = {}

def rec(key, value_fn):
    try:
        out[key] = value_fn()
    except Exception as e:
        out[key] = f"ERROR:{type(e).__name__}:{e}"

# --- sqrt: free function, member, and the two DeBERTa expressions ----------
sq = torch.tensor([1.0, 4.0, 9.0, 16.0])
rec("sqrt_fn", lambda: torch.sqrt(sq).tolist())
rec("sqrt_member", lambda: sq.sqrt().tolist())
rec("sqrt_neg_is_nan", lambda: math.isnan(torch.sqrt(torch.tensor([-1.0])).item()))
rec("sqrt_neg_zero_sign",
    lambda: math.copysign(1.0, torch.sqrt(torch.tensor([-0.0])).item()))
rec("sqrt_int_dtype", lambda: str(torch.sqrt(torch.tensor([4], dtype=torch.int64)).dtype))
rec("sqrt_matches_raw",
    lambda: torch.sqrt(sq).tolist()
    == torch._C._aten_dispatch("aten.sqrt.default", sq).tolist())

# `deberta_v2`'s `scaled_size_sqrt` verbatim: the attention temperature, a
# 0-d tensor built from a Python int and a scale factor.
rec("deberta_scaled_size_sqrt",
    lambda: torch.sqrt(torch.tensor(64, dtype=torch.float) * 3).item())
# `DebertaLayerNorm.forward` verbatim, on a toy row.
def _deberta_layer_norm():
    h = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    mean = h.mean(-1, keepdim=True)
    variance = (h - mean).pow(2).mean(-1, keepdim=True)
    return ((h - mean) / torch.sqrt(variance + 1e-7)).tolist()
rec("deberta_layer_norm", _deberta_layer_norm)

# --- repeat: member only (upstream has no `torch.repeat`), varargs and list --
rp = torch.tensor([1.0, 2.0, 3.0])
rec("repeat_member_varargs", lambda: rp.repeat(2, 3).tolist())
rec("repeat_member_list", lambda: rp.repeat([2, 3]).tolist())
rec("repeat_member_single", lambda: rp.repeat(2).tolist())
rec("repeat_zero_shape", lambda: list(rp.repeat(0).shape))
rec("repeat_no_free_function", lambda: hasattr(torch, "repeat"))
rec("repeat_matches_raw",
    lambda: rp.repeat(2, 3).tolist()
    == torch._C._aten_dispatch("aten.repeat.default", rp, [2, 3]).tolist())
# `repeat` must copy: writing into the result must not reach the source.
def _repeat_is_a_copy():
    src = torch.tensor([1.0, 2.0])
    r = src.repeat(1)
    r.fill_(0.0)
    return src.tolist()
rec("repeat_is_a_copy", _repeat_is_a_copy)
# `build_relative_position`'s shape, the reason `deberta` needs this at all:
# a (1, q, k) relative-position grid tiled to the batch.
rec("deberta_relative_position_repeat",
    lambda: list(torch.arange(6).reshape(1, 2, 3).repeat(4, 1, 1).shape))

# --- convolution 2-D: through torch.conv2d and nn.Conv2d --------------------
# `torch.conv2d` was already wired (ARCH26.md §7) over a kernel that refused
# 4-D input, so this is the first time that spelling reaches anything.
import torch.nn as nn
def _conv2d_road():
    x = torch.arange(2 * 3 * 8 * 8, dtype=torch.float32).reshape(2, 3, 8, 8) * 0.01
    w = torch.arange(4 * 3 * 4 * 4, dtype=torch.float32).reshape(4, 3, 4, 4) * 0.02
    b = torch.tensor([0.1, -0.2, 0.3, -0.4])
    # Dinov2's patch embedding shape: 4x4 kernel, stride 4, no padding.
    out = torch.conv2d(x, w, b, 2 if False else [4, 4], [0, 0], [1, 1], 1)
    return [list(out.shape), round(float(out.sum()), 4)]
rec("conv2d_patch_embed", _conv2d_road)
def _conv2d_matches_raw():
    x = torch.arange(1 * 2 * 5 * 5, dtype=torch.float32).reshape(1, 2, 5, 5)
    w = torch.arange(3 * 2 * 3 * 3, dtype=torch.float32).reshape(3, 2, 3, 3)
    a = torch.conv2d(x, w, None, [1, 1], [1, 1], [1, 1], 1)
    b = torch._C._aten_dispatch("aten.convolution.default", x, w, None,
                                [1, 1], [1, 1], [1, 1], False, [0, 0], 1)
    return a.tolist() == b.tolist()
rec("conv2d_matches_raw", _conv2d_matches_raw)
def _nn_conv2d():
    # The route a vision backbone actually takes: nn.Conv2d.forward, which is
    # F.conv2d, which is torch.conv2d, which is aten.convolution.default.
    m = nn.Conv2d(3, 4, kernel_size=4, stride=4, bias=True)
    with torch.no_grad():
        m.weight.fill_(0.01)
        m.bias.fill_(0.5)
        out = m(torch.ones(2, 3, 8, 8))
    # each output element = 3*4*4 ones * 0.01 + 0.5
    return [list(out.shape), round(float(out.reshape(-1)[0]), 6)]
rec("nn_conv2d_forward", _nn_conv2d)
def _conv2d_asymmetric_refused():
    x = torch.ones(1, 2, 7, 7); w = torch.ones(2, 2, 3, 3)
    try:
        torch.conv2d(x, w, None, [2, 1], [0, 0], [1, 1], 1)
        return "ACCEPTED"
    except NotImplementedError as e:
        return "refused" if "asymmetric" in str(e) else "refused:" + str(e)[:80]
rec("conv2d_asymmetric_refused", _conv2d_asymmetric_refused)

# --- remainder: free function, member, and the `%` operator -----------------
rm = torch.tensor([7.0, -7.0, 7.0, -7.0])
rd = torch.tensor([3.0, 3.0, -3.0, -3.0])
rec("remainder_fn", lambda: torch.remainder(rm, rd).tolist())
rec("remainder_member", lambda: rm.remainder(rd).tolist())
rec("remainder_operator", lambda: (rm % rd).tolist())
rec("remainder_operator_scalar", lambda: (rm % -3.0).tolist())
rec("remainder_int_operator", lambda: (torch.tensor([7, -7]) % 3).tolist())
# `fmod` has no kernel here (docs/KERNELS26.md §6 leaves it named), so the
# comparison is against Python's own `math.fmod` -- the same C function
# upstream's `fmod` kernel calls, and it keeps this file's promise of not
# needing a second torch to compute an expectation.
rec("remainder_vs_fmod",
    lambda: [torch.remainder(rm, rd).tolist(),
             [math.fmod(a, b) for a, b in zip(rm.tolist(), rd.tolist())]])
rec("remainder_neg_zero_sign",
    lambda: math.copysign(1.0, (torch.tensor([-0.0]) % 3.0).item()))
rec("remainder_matches_raw",
    lambda: (rm % rd).tolist()
    == torch._C._aten_dispatch("aten.remainder.Tensor", rm, rd).tolist())
# `Sam3ViTRotaryEmbedding.__init__` verbatim: a flattened index grid taken
# modulo the row length, which is what `sam3_video` stopped on.
def _sam3_rotary_positions():
    end_x, end_y = 4, 3
    flat = torch.arange(end_x * end_y)
    return (flat % end_x).tolist()
rec("sam3_rotary_x_positions", _sam3_rotary_positions)

# --- div rounding modes: free function, member, and both overloads ----------
dv = torch.tensor([7, 7, -7, -7, 6, -6])
dd = torch.tensor([3, -3, 3, -3, 3, 3])
rec("div_floor_fn", lambda: torch.div(dv, dd, rounding_mode="floor").tolist())
rec("div_trunc_fn", lambda: torch.div(dv, dd, rounding_mode="trunc").tolist())
rec("div_floor_member", lambda: dv.div(dd, rounding_mode="floor").tolist())
rec("div_trunc_member", lambda: dv.div(dd, rounding_mode="trunc").tolist())
# The Scalar_mode overload: a bare Python int in the divisor slot.
rec("div_floor_scalar", lambda: torch.div(dv, 3, rounding_mode="floor").tolist())
rec("div_trunc_scalar", lambda: torch.div(dv, 3, rounding_mode="trunc").tolist())
rec("div_floor_scalar_member", lambda: dv.div(3, rounding_mode="floor").tolist())
# dtype: the two rounding modes PRESERVE and rounding_mode=None PROMOTES.
rec("div_floor_dtype", lambda: str(torch.div(dv, dd, rounding_mode="floor").dtype))
rec("div_none_dtype", lambda: str(torch.div(dv, dd, rounding_mode=None).dtype))
rec("div_plain_dtype", lambda: str(torch.div(dv, dd).dtype))
# `rounding_mode=None` must equal the plain `div` -- it delegates to it.
rec("div_none_equals_plain",
    lambda: torch.div(dv, dd, rounding_mode=None).tolist() == torch.div(dv, dd).tolist())
# Which overload each spelling actually resolved to.
rec("div_scalar_matches_raw",
    lambda: torch.div(dv, 3, rounding_mode="floor").tolist()
    == torch._C._aten_dispatch("aten.div.Scalar_mode", dv, 3, rounding_mode="floor").tolist())
rec("div_tensor_matches_raw",
    lambda: torch.div(dv, dd, rounding_mode="floor").tolist()
    == torch._C._aten_dispatch("aten.div.Tensor_mode", dv, dd, rounding_mode="floor").tolist())
# floor and trunc must DISAGREE where the signs differ and the division is
# inexact, and AGREE where it is exact. Computed here, not taken from the shim.
rec("div_floor_trunc_disagree",
    lambda: [f != t for f, t in zip(torch.div(dv, dd, rounding_mode="floor").tolist(),
                                    torch.div(dv, dd, rounding_mode="trunc").tolist())])
# The float corner that kills `floor(a / b)`.
rec("div_floor_inf_over_finite",
    lambda: math.isnan(torch.div(torch.tensor([float("inf")]), torch.tensor([3.0]),
                                 rounding_mode="floor").item()))
rec("div_trunc_inf_over_finite",
    lambda: math.isinf(torch.div(torch.tensor([float("inf")]), torch.tensor([3.0]),
                                 rounding_mode="trunc").item()))
rec("div_floor_five_over_neg_inf",
    lambda: torch.div(torch.tensor([5.0]), torch.tensor([float("-inf")]),
                      rounding_mode="floor").item())
rec("div_floor_five_over_zero",
    lambda: math.isinf(torch.div(torch.tensor([5.0]), torch.tensor([0.0]),
                                 rounding_mode="floor").item()))
rec("div_floor_neg_zero_sign",
    lambda: math.copysign(1.0, torch.div(torch.tensor([-0.0]), torch.tensor([3.0]),
                                         rounding_mode="floor").item()))
# Integral division by zero raises under a rounding mode and NOT under None.
def _div_zero_int():
    try:
        torch.div(torch.tensor([5]), 0, rounding_mode="floor")
        return "ACCEPTED"
    except RuntimeError as e:
        return "raised:" + str(e)
rec("div_int_zero_raises", _div_zero_int)
rec("div_int_zero_none_is_inf",
    lambda: math.isinf(torch.div(torch.tensor([5]), 0, rounding_mode=None).item()))
# An unrecognised mode is refused by upstream's own wording.
def _div_bad_mode():
    try:
        torch.div(dv, dd, rounding_mode="ceil")
        return "ACCEPTED"
    except RuntimeError as e:
        return str(e)
rec("div_bad_mode", _div_bad_mode)
# The scalar narrows into the result dtype BEFORE dividing.
rec("div_uint8_narrows",
    lambda: torch.div(torch.tensor([200], dtype=torch.uint8), -3,
                      rounding_mode="floor").tolist())
# `Sam3ViTRotaryEmbedding.__init__` verbatim -- BOTH axes, which is the pair of
# lines `sam3_video` stopped on. `%` is `remainder`, `//`-shaped `torch.div`
# with rounding_mode="floor" is this kernel.
def _sam3_rotary_both_axes():
    end_x, end_y = 4, 3
    flat = torch.arange(end_x * end_y)
    x_positions = flat % end_x
    y_positions = torch.div(flat, end_x, rounding_mode="floor")
    return [x_positions.tolist(), y_positions.tolist()]
rec("sam3_rotary_both_axes", _sam3_rotary_both_axes)

# --- conv_transpose2d: the free function, nn.ConvTranspose2d, and the layout --
def _convt_layout_by_shape():
    # in=3, out=5, kernel 2x4. `out_channels` is weight.shape[1]; if the layout
    # were (out, in, ...) this call would raise instead of answering (2,5,6,10).
    x = torch.arange(2 * 3 * 5 * 7, dtype=torch.float32).reshape(2, 3, 5, 7) * 0.01
    w = torch.arange(3 * 5 * 2 * 4, dtype=torch.float32).reshape(3, 5, 2, 4) * 0.02
    return list(torch.conv_transpose2d(x, w, None, 1, 0, 0, 1, 1).shape)
rec("convt_layout_by_shape", _convt_layout_by_shape)
def _convt_wrong_layout_refused():
    # The same call with the weight transposed: 5 in-channels against a 3-channel
    # input. Upstream raises here and so must this.
    x = torch.ones(2, 3, 5, 7); w = torch.ones(5, 3, 2, 4)
    try:
        torch.conv_transpose2d(x, w)
        return "ACCEPTED"
    except RuntimeError:
        return "refused"
rec("convt_wrong_layout_refused", _convt_wrong_layout_refused)
def _convt_swap_changes_the_answer():
    # Equal channels and a square kernel -- the shape cannot tell the two
    # arrangements apart, so this asserts that the VALUES do. If this ever
    # comes back True the kernel is ignoring the weight's axis order, which is
    # precisely the failure `zoedepth`'s own square call cannot show.
    x = torch.arange(1 * 2 * 3 * 3, dtype=torch.float64).reshape(1, 2, 3, 3)
    # Built from index arithmetic on plain lists rather than with `.transpose`
    # and `.flip`: `TensorBase.flip` is not implemented in this shim, and a
    # rearrangement done with ops under test could not be trusted to be the
    # rearrangement it claims anyway.
    base = [[[[((i * 2 + o) * 3 + p) * 3 + q for q in range(3)] for p in range(3)]
             for o in range(2)] for i in range(2)]
    swapped = [[[[base[o][i][p][q] for q in range(3)] for p in range(3)]
                for o in range(2)] for i in range(2)]
    flip = [[[[base[i][o][2 - p][2 - q] for q in range(3)] for p in range(3)]
             for o in range(2)] for i in range(2)]
    w = torch.tensor(base, dtype=torch.float64)
    a = torch.conv_transpose2d(x, w)
    b = torch.conv_transpose2d(x, torch.tensor(swapped, dtype=torch.float64))
    flipped = torch.conv_transpose2d(x, torch.tensor(flip, dtype=torch.float64))
    return {
        "same_shape": list(a.shape) == list(b.shape) == list(flipped.shape),
        "swap_equal": a.tolist() == b.tolist(),
        "flip_equal": a.tolist() == flipped.tolist(),
        # The spatial flip leaves the SUM identical, so a checksum test cannot
        # see it. Recorded so that claim is checked rather than asserted.
        "flip_same_sum": abs(float(a.sum()) - float(flipped.sum())) < 1e-6,
        "swap_same_sum": abs(float(a.sum()) - float(b.sum())) < 1e-6,
    }
rec("convt_swap_changes_the_answer", _convt_swap_changes_the_answer)
def _nn_conv_transpose2d():
    # The route zoedepth takes: nn.ConvTranspose2d.forward -> F.conv_transpose2d
    # -> torch.conv_transpose2d -> aten.convolution.default(transposed=True).
    m = nn.ConvTranspose2d(2, 3, kernel_size=2, stride=2, padding=0)
    with torch.no_grad():
        m.weight.fill_(0.5)
        m.bias.fill_(0.25)
        o = m(torch.ones(1, 2, 3, 3))
    # stride == kernel, so the windows tile: each output element is
    # 2 in-channels * 0.5 + 0.25.
    return [list(o.shape), round(float(o.reshape(-1)[0]), 6)]
rec("nn_conv_transpose2d_forward", _nn_conv_transpose2d)
rec("convt_matches_raw",
    lambda: torch.conv_transpose2d(
        torch.ones(1, 2, 3, 3), torch.ones(2, 3, 2, 2), None, 2, 0, 0, 1, 1).tolist()
    == torch._C._aten_dispatch(
        "aten.convolution.default", torch.ones(1, 2, 3, 3), torch.ones(2, 3, 2, 2),
        None, [2, 2], [0, 0], [1, 1], True, [0, 0], 1).tolist())
# `F.conv_transpose2d` IS `torch.conv_transpose2d` upstream -- asserted, not
# assumed, because the whole spelling rests on it.
import torch.nn.functional as _F
rec("convt_F_is_torch", lambda: _F.conv_transpose2d is torch.conv_transpose2d)
# The signature has `groups` BEFORE `dilation`, unlike conv2d. Called
# positionally with groups=1 and dilation=2: a transcription of conv2d's order
# would read the 7th positional as `dilation` and the 8th as `groups`, giving
# dilation=1 and groups=2 -- a different shape, and in fact a refusal.
rec("convt_positional_signature",
    lambda: list(torch.conv_transpose2d(
        torch.ones(1, 2, 4, 4), torch.ones(2, 3, 3, 3), None, 2, 1, 1, 1, 2).shape))
def _convt_groups_refused():
    try:
        torch.conv_transpose2d(torch.ones(1, 4, 4, 4), torch.ones(4, 2, 3, 3),
                               None, 1, 0, 0, 2, 1)
        return "ACCEPTED"
    except NotImplementedError as e:
        return "refused" if "groups" in str(e) else "refused:" + str(e)[:80]
rec("convt_groups_refused", _convt_groups_refused)
def _convt_outpad_bound():
    try:
        torch.conv_transpose2d(torch.ones(1, 2, 4, 4), torch.ones(2, 2, 3, 3),
                               None, 1, 0, 1, 1, 1)
        return "ACCEPTED"
    except RuntimeError as e:
        return str(e)
rec("convt_outpad_bound", _convt_outpad_bound)

# --- weight_norm: three pieces, and the route that hid all of them -----------
wn_v = torch.tensor([[3.0, 4.0], [0.0, 5.0]])
rec("ned_dim0", lambda: torch.norm_except_dim(wn_v, 2, 0).tolist())
rec("ned_dim1", lambda: torch.norm_except_dim(wn_v, 2, 1).tolist())
rec("ned_dim_minus1", lambda: torch.norm_except_dim(wn_v, 2, -1).tolist())
rec("ned_dim0_shape", lambda: list(torch.norm_except_dim(wn_v, 2, 0).shape))
rec("ned_dim1_shape", lambda: list(torch.norm_except_dim(wn_v, 2, 1).shape))
rec("ned_dim_minus1_shape", lambda: list(torch.norm_except_dim(wn_v, 2, -1).shape))
# `torch._weight_norm` returns ONE tensor; `_weight_norm_interface` returns the
# pair. Both spellings, so the `[0]` in the composite is pinned.
wn_g = torch.tensor([[2.0], [3.0]])
rec("weight_norm_fn", lambda: torch._weight_norm(wn_v, wn_g, 0).tolist())
rec("weight_norm_interface_pair",
    lambda: [t.tolist() for t in torch._C._aten_dispatch(
        "aten._weight_norm_interface.default", wn_v, wn_g, 0)])
rec("weight_norm_matches_interface",
    lambda: torch._weight_norm(wn_v, wn_g, 0).tolist()
    == torch._C._aten_dispatch(
        "aten._weight_norm_interface.default", wn_v, wn_g, 0)[0].tolist())
# norm.ScalarOpt_dim by its own key, across the p family.
nrm = torch.tensor([[3.0, -4.0], [0.0, 1.0]])
rec("norm_p_family", lambda: {
    str(p): torch._C._aten_dispatch("aten.norm.ScalarOpt_dim", nrm, p, [1], False).tolist()
    for p in (2, 1, 0, None)})
rec("norm_p_inf", lambda: [
    torch._C._aten_dispatch("aten.norm.ScalarOpt_dim", nrm, float("inf"), [1], False).tolist(),
    torch._C._aten_dispatch("aten.norm.ScalarOpt_dim", nrm, float("-inf"), [1], False).tolist()])
rec("norm_empty_dim_reduces_all",
    lambda: torch._C._aten_dispatch("aten.norm.ScalarOpt_dim", nrm, 2, [], False).tolist())

# THE ROUTE: nn.utils.parametrizations.weight_norm, which is what vits and
# sew_d use. A missing kernel on this path surfaced as
# `TypeError: _WeightNorm.forward() missing 1 required positional argument`
# 200 frames away, because ParametrizationList.__init__ has
# `except NotImplementedError: pass` around right_inverse.
def _weight_norm_module(dim):
    m = nn.Conv1d(2, 3, 3, bias=False)
    with torch.no_grad():
        m.weight.fill_(0.5)
    wn = nn.utils.parametrizations.weight_norm(m, name="weight", dim=dim)
    y = wn(torch.ones(1, 2, 6))
    return [list(y.shape), round(float(y.reshape(-1)[0]), 5)]
rec("weight_norm_module_dim0", lambda: _weight_norm_module(0))
rec("weight_norm_module_dim2", lambda: _weight_norm_module(2))
# The parametrization really is installed (not silently skipped), and the
# reconstructed weight equals the original -- right_inverse followed by forward
# is the identity when the kernels are present. That is the assertion the
# swallowed NotImplementedError defeated.
def _weight_norm_roundtrip():
    m = nn.Conv1d(2, 3, 3, bias=False)
    with torch.no_grad():
        m.weight.copy_(torch.arange(18, dtype=torch.float32).reshape(3, 2, 3) * 0.1 - 0.5)
    before = m.weight.tolist()
    wn = nn.utils.parametrizations.weight_norm(m, name="weight", dim=0)
    after = wn.weight.tolist()
    flat_b = [v for a in before for b in a for v in b]
    flat_a = [v for a in after for b in a for v in b]
    return {
        "parametrized": bool(getattr(wn, "parametrizations", None) is not None
                             and "weight" in getattr(wn, "parametrizations", {})),
        "roundtrip_max_abs_diff": max(abs(x - y) for x, y in zip(flat_b, flat_a)),
        "has_g_and_v": sorted(
            n for n, _ in wn.parametrizations.weight.named_parameters()),
    }
rec("weight_norm_roundtrip", _weight_norm_roundtrip)

# --- the tail: four name gaps over kernels that already existed --------------
ov_a, ov_b = torch.arange(3.0), torch.arange(1.0, 5.0)
rec("outer_fn", lambda: torch.outer(ov_a, ov_b).tolist())
rec("outer_member", lambda: ov_a.outer(ov_b).tolist())
rec("outer_shape", lambda: list(torch.outer(ov_a, ov_b).shape))
rec("outer_int_dtype",
    lambda: str(torch.outer(torch.arange(3), torch.arange(4)).dtype))
rec("outer_mixed_dtype",
    lambda: str(torch.outer(torch.arange(3), torch.arange(4.0)).dtype))
def _outer_rank_refused():
    try:
        torch.outer(torch.ones(2, 2), ov_b)
        return "ACCEPTED"
    except RuntimeError as e:
        return str(e)
rec("outer_rank_refused", _outer_rank_refused)

# `tile` is NOT `repeat`: too few dims are left-padded rather than refused.
tl = torch.arange(6).reshape(2, 3)
rec("tile_too_few_dims", lambda: list(tl.tile((2,)).shape))
rec("tile_varargs", lambda: list(tl.tile(2, 2).shape))
rec("tile_more_dims", lambda: list(tl.tile((2, 1, 1)).shape))
rec("tile_free_fn", lambda: list(torch.tile(tl, (2,)).shape))
rec("tile_values", lambda: tl.tile((2,)).tolist())
def _repeat_refuses_too_few():
    # The difference that makes `tile` a separate function: `repeat` needs at
    # least as many dims as the rank, `tile` pads.
    try:
        tl.repeat(2)
        return "ACCEPTED"
    except Exception as e:
        return type(e).__name__
rec("repeat_refuses_too_few", _repeat_refuses_too_few)

rec("ones_like_values", lambda: torch.ones_like(torch.zeros(2, 3)).tolist())
rec("ones_like_dtype",
    lambda: str(torch.ones_like(torch.zeros(2, 2, dtype=torch.int64)).dtype))
rec("detach_fn", lambda: torch.detach(torch.arange(3.0)).tolist())
# `detach` returns a view that shares storage upstream; asserted through a
# write, because that is the property callers depend on.
def _detach_shares():
    base = torch.ones(3)
    d = torch.detach(base)
    d.fill_(5.0)
    return base.tolist()
rec("detach_shares_storage", _detach_shares)

json.dump(out, sys.stdout)
"""


def _kernels26_road_fixture():
    env = dict(os.environ)
    env["PYTHONPATH"] = _CKPT_VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    proc = subprocess.run(
        [sys.executable, "-c", _KERNELS26_ROAD_SCRIPT],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"kernels26 road subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout)


def test_kernels26_road_through_the_vendored_tree():
    """Every kernel docs/KERNELS26.md adds, reached as a model reaches it.

    Not `_C._aten_dispatch("aten.sqrt.default", ...)` with the key typed by the
    test author -- that is what the section above does, and it is exactly the
    check that cannot fail on a missing `overloads.json` entry. This one goes
    `torch.sqrt(x)` and `x.sqrt()` through a real `import torch`.
    """
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        return  # vendor tree not installed -- see vendor/install_shim.sh
    out = _kernels26_road_fixture()

    def eq(key, expected):
        got = out.get(key, "<missing>")
        assert got == expected, f"{key}: expected {expected!r}, got {got!r}"

    def close(key, expected, tol=1e-5):
        got = out.get(key, "<missing>")
        assert isinstance(got, (int, float)), f"{key}: got {got!r}"
        assert abs(got - expected) < tol, f"{key}: expected {expected!r}, got {got!r}"

    # --- sqrt ---------------------------------------------------------------
    eq("sqrt_fn", [1.0, 2.0, 3.0, 4.0])
    eq("sqrt_member", [1.0, 2.0, 3.0, 4.0])
    eq("sqrt_neg_is_nan", True)
    eq("sqrt_neg_zero_sign", -1.0)  # sqrt(-0.0) is -0.0, not +0.0
    eq("sqrt_int_dtype", "torch.float32")
    eq("sqrt_matches_raw", True)
    close("deberta_scaled_size_sqrt", math.sqrt(64.0 * 3))
    rows = out.get("deberta_layer_norm", "<missing>")
    assert isinstance(rows, list) and len(rows) == 1, rows
    mean = 2.5
    var = sum((v - mean) ** 2 for v in (1.0, 2.0, 3.0, 4.0)) / 4
    want = [(v - mean) / math.sqrt(var + 1e-7) for v in (1.0, 2.0, 3.0, 4.0)]
    for got_v, want_v in zip(rows[0], want):
        assert abs(got_v - want_v) < 1e-5, (rows[0], want)

    # --- repeat -------------------------------------------------------------
    eq("repeat_member_varargs", [[1.0, 2.0, 3.0] * 3] * 2)
    eq("repeat_member_list", [[1.0, 2.0, 3.0] * 3] * 2)
    eq("repeat_member_single", [1.0, 2.0, 3.0, 1.0, 2.0, 3.0])
    eq("repeat_zero_shape", [0])
    eq("repeat_no_free_function", False)  # upstream has no torch.repeat
    eq("repeat_matches_raw", True)
    eq("repeat_is_a_copy", [1.0, 2.0])
    eq("deberta_relative_position_repeat", [4, 2, 3])

    # --- remainder ----------------------------------------------------------
    # (7, -7) against (3, -3): the four sign quadrants in one call.
    eq("remainder_fn", [1.0, 2.0, -2.0, -1.0])
    eq("remainder_member", [1.0, 2.0, -2.0, -1.0])
    eq("remainder_operator", [1.0, 2.0, -2.0, -1.0])
    eq("remainder_operator_scalar", [-2.0, -1.0, -2.0, -1.0])
    eq("remainder_int_operator", [1, 2])
    # The pair that separates the two conventions, side by side.
    eq("remainder_vs_fmod", [[1.0, 2.0, -2.0, -1.0], [1.0, -1.0, 1.0, -1.0]])
    eq("remainder_neg_zero_sign", -1.0)
    eq("remainder_matches_raw", True)
    eq("sam3_rotary_x_positions", [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3])

    # --- convolution 2-D ----------------------------------------------------
    shape, _total = out.get("conv2d_patch_embed", ["<missing>", None])
    assert shape == [2, 4, 2, 2], out.get("conv2d_patch_embed")
    eq("conv2d_matches_raw", True)
    # 3*4*4 = 48 ones times 0.01, plus a bias of 0.5.
    eq("nn_conv2d_forward", [[2, 4, 2, 2], round(48 * 0.01 + 0.5, 6)])
    eq("conv2d_asymmetric_refused", "refused")

    # --- div rounding modes -------------------------------------------------
    # (7, 7, -7, -7, 6, -6) over (3, -3, 3, -3, 3, 3). The last two are the
    # EXACT pairs, where floor and trunc agree even though the signs differ --
    # so a case set built only from them could not tell the modes apart.
    eq("div_floor_fn", [2, -3, -3, 2, 2, -2])
    eq("div_trunc_fn", [2, -2, -2, 2, 2, -2])
    eq("div_floor_member", [2, -3, -3, 2, 2, -2])
    eq("div_trunc_member", [2, -2, -2, 2, 2, -2])
    eq("div_floor_scalar", [2, 2, -3, -3, 2, -2])
    eq("div_trunc_scalar", [2, 2, -2, -2, 2, -2])
    eq("div_floor_scalar_member", [2, 2, -3, -3, 2, -2])
    # The modes disagree on exactly the opposite-sign INEXACT pairs: indices
    # 1 and 2. Index 5 is `-6 / 3`, opposite signs but exact, and they agree.
    eq("div_floor_trunc_disagree", [False, True, True, False, False, False])
    # dtype is how `None` is told apart from the other two.
    eq("div_floor_dtype", "torch.int64")
    eq("div_none_dtype", "torch.float32")
    eq("div_plain_dtype", "torch.float32")
    eq("div_none_equals_plain", True)
    eq("div_scalar_matches_raw", True)
    eq("div_tensor_matches_raw", True)
    # The corners that a `floor(a / b)` implementation gets wrong.
    eq("div_floor_inf_over_finite", True)   # nan, NOT inf
    eq("div_trunc_inf_over_finite", True)   # inf under trunc, though
    eq("div_floor_five_over_neg_inf", -1.0)  # not -0.0
    eq("div_floor_five_over_zero", True)     # inf: the b == 0 early return
    eq("div_floor_neg_zero_sign", -1.0)      # -0.0 keeps its sign bit
    got = out.get("div_int_zero_raises", "<missing>")
    assert got.startswith("raised:") and "ZeroDivisionError" in got, got
    eq("div_int_zero_none_is_inf", True)     # ... and None does not raise
    eq(
        "div_bad_mode",
        "div expected rounding_mode to be one of None, 'trunc', or 'floor' "
        "but found 'ceil'",
    )
    # `-3` becomes `253` in uint8 before the division, so this is 0, not -66.
    eq("div_uint8_narrows", [0])
    # Both axes of the rotary grid `sam3_video` stopped on.
    eq(
        "sam3_rotary_both_axes",
        [[0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3], [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]],
    )

    # --- conv_transpose2d ---------------------------------------------------
    # The weight is (in, out/groups, kH, kW) -- the OPPOSITE of the forward
    # convolution's (out, in/groups, kH, kW). in=3, out=5, kernel 2x4, so
    # out_channels=5 comes from weight.shape[1] and no two axes are
    # interchangeable.
    eq("convt_layout_by_shape", [2, 5, 6, 10])
    eq("convt_wrong_layout_refused", "refused")
    # `zoedepth`'s own call is ConvTranspose2d(c, c, kernel_size=f, stride=f) --
    # equal channels and a square kernel -- so its shape cannot reveal the
    # layout at all. These three assertions are what stands in for that:
    # the arrangements have the same shape, and different values.
    swap = out.get("convt_swap_changes_the_answer", "<missing>")
    assert swap.get("same_shape") is True, swap
    assert swap.get("swap_equal") is False, (
        "swapping the weight's first two axes must change the answer; it did "
        "not, so the kernel is ignoring the axis order", swap)
    assert swap.get("flip_equal") is False, (
        "flipping the kernel spatially must change the answer", swap)
    # ...and the flip keeps the SUM identical, which is why none of this can be
    # checked with a checksum.
    assert swap.get("flip_same_sum") is True, swap
    assert swap.get("swap_same_sum") is False, swap
    # 2 in-channels * 0.5 + 0.25 bias, tiled because stride == kernel_size.
    eq("nn_conv_transpose2d_forward", [[1, 3, 6, 6], round(2 * 0.5 + 0.25, 6)])
    eq("convt_matches_raw", True)
    eq("convt_F_is_torch", True)
    # (input, weight, bias, stride=2, padding=1, output_padding=1, groups=1,
    # dilation=2) -- `groups` is the 7th positional and `dilation` the 8th,
    # which is NOT conv2d's order. Reading them the other way round gives
    # dilation=1, groups=2, which is a different shape and in fact a refusal.
    eq("convt_positional_signature", [1, 3, 10, 10])
    eq("convt_groups_refused", "refused")
    eq(
        "convt_outpad_bound",
        "output padding must be smaller than either stride or dilation, but got "
        "output_padding_height: 1 output_padding_width: 1",
    )

    # --- weight_norm's three pieces -----------------------------------------
    # v = [[3, 4], [0, 5]]. `norm_except_dim` KEEPS `dim` and reduces the rest,
    # keepdim -- so dim=0 gives column norms shaped (2,1) and dim=1 gives row
    # norms shaped (1,2). Those two are different numbers as well as different
    # shapes, which is what makes the axis choice checkable.
    eq("ned_dim0", [[5.0], [5.0]])                    # sqrt(9+16), sqrt(0+25)
    eq("ned_dim0_shape", [2, 1])
    eq("ned_dim1_shape", [1, 2])
    rows = out.get("ned_dim1", "<missing>")
    assert abs(rows[0][0] - 3.0) < 1e-5 and abs(rows[0][1] - math.sqrt(41)) < 1e-5, rows
    # dim=-1 is NOT "the last axis" here: it is upstream's "no axis is exempt"
    # spelling, and gives the whole-tensor norm as a 0-d result.
    eq("ned_dim_minus1_shape", [])
    close("ned_dim_minus1", math.sqrt(50))
    # `torch._weight_norm` returns one tensor; the interface returns the pair.
    eq("weight_norm_fn", [[1.2000000476837158, 1.600000023841858], [0.0, 3.0]])
    eq(
        "weight_norm_interface_pair",
        [[[1.2000000476837158, 1.600000023841858], [0.0, 3.0]], [[5.0], [5.0]]],
    )
    eq("weight_norm_matches_interface", True)
    # The `p` family: five of these are different functions, and
    # `norm_except_dim` only ever passes 2 -- so nothing on the weight_norm path
    # would exercise the rest.
    eq("norm_p_family", {"2": [5.0, 1.0], "1": [7.0, 1.0],
                         "0": [2.0, 1.0], "None": [5.0, 1.0]})
    eq("norm_p_inf", [[4.0, 1.0], [3.0, 0.0]])        # max |x|, then min |x|
    close("norm_empty_dim_reduces_all", math.sqrt(26))  # [] means EVERY axis
    # The real route, which is what vits and sew_d take. The weight is filled
    # with 0.5, and weight_norm reproduces it exactly at construction, so a
    # 2-channel 3-tap convolution of ones is 2*3*0.5 = 3.0 -- for BOTH dims.
    eq("weight_norm_module_dim0", [[1, 3, 4], 3.0])
    eq("weight_norm_module_dim2", [[1, 3, 4], 3.0])
    # ...and the parametrization is genuinely installed rather than silently
    # skipped. `ParametrizationList.__init__` swallows NotImplementedError
    # around `right_inverse`, so a missing kernel here leaves a half-built
    # parametrization that fails 200 frames later with a TypeError naming no
    # kernel at all. These three assertions are what that defeated.
    rt = out.get("weight_norm_roundtrip", "<missing>")
    assert rt.get("parametrized") is True, rt
    assert rt.get("has_g_and_v") == ["original0", "original1"], (
        "weight_norm must install BOTH the magnitude and the direction "
        "parameter; one of them means right_inverse was skipped", rt)
    assert rt.get("roundtrip_max_abs_diff") < 1e-5, (
        "right_inverse followed by forward must reproduce the original weight", rt)

    # --- the tail: four name gaps over kernels that already existed ---------
    # `outer` fires only view+mul, so what is checked is the SHAPE convention
    # (rows from self, columns from vec2) and the promotion it inherits.
    eq("outer_fn", [[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    eq("outer_member", [[0.0, 0.0, 0.0, 0.0], [1.0, 2.0, 3.0, 4.0], [2.0, 4.0, 6.0, 8.0]])
    eq("outer_shape", [3, 4])            # (len(self), len(vec2)), not the reverse
    eq("outer_int_dtype", "torch.int64")     # inherited from mul, not restated
    eq("outer_mixed_dtype", "torch.float32")
    eq("outer_rank_refused", "outer: Expected 1-D argument self, but got 2-D")
    # `tile` is NOT `repeat`. Too FEW dims are left-padded here and refused
    # there, which is the whole reason it is a separate function -- and the
    # padding is on the LEFT, so a (2,3) tiled by 2 is (2,6), not (4,3).
    eq("tile_too_few_dims", [2, 6])
    eq("repeat_refuses_too_few", "RuntimeError")
    eq("tile_varargs", [4, 6])
    eq("tile_more_dims", [2, 2, 3])      # too many: extra dims lead, as repeat
    eq("tile_free_fn", [2, 6])
    eq("tile_values", [[0, 1, 2, 0, 1, 2], [3, 4, 5, 3, 4, 5]])
    # `ones_like`'s values are defined, unlike its zeros/empty siblings'.
    eq("ones_like_values", [[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    eq("ones_like_dtype", "torch.int64")
    # `detach` was already a kernel with no name; it must still be a view.
    eq("detach_fn", [0.0, 1.0, 2.0])
    eq("detach_shares_storage", [5.0, 5.0, 5.0])


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
