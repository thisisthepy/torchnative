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
    try:
        _C._aten_dispatch("aten.embedding.default")
    except NotImplementedError as e:
        assert str(e) == (
            "aten op not implemented in torch._C shim: aten.embedding.default"
        )
    else:
        raise AssertionError("an unimplemented op must raise")


def test_every_advertised_op_is_actually_dispatchable():
    # A name in the list that falls through to the fallback would make the
    # instrument lie about what is covered.
    for op in _C._aten_implemented():
        try:
            _C._aten_dispatch(op)
        except NotImplementedError as e:  # pragma: no cover - regression guard
            raise AssertionError(f"{op} is advertised but not dispatchable: {e}")
        except TypeError:
            pass  # missing arguments: reached the kernel, which is the point


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
        assert "f32" in str(e) and "f64" in str(e)
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
