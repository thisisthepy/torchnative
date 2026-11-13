"""`TensorBase._base` answers the base tensor, as upstream does (#255).

PR #256 made a capture failure on a compiled device refuse by name. Gate run
43333868410 then showed two `tests/devices/npu/test_npufuse.py` tests refusing
at `TensorBase._base`: Dynamo's `builder.py` (`wrap_tensor`,
`_automatic_dynamic`) reads `t._base` for every graph input whose `_is_view()`
is True, and transformers' `generate()` hands `LlamaModel.forward` a slice of
its id buffer. The shim's `_base` refused because `PyTensorBase` kept no base.

Every expected value below was measured on upstream torch 2.13.0 (the
`spike-venv` wheel), not chosen:

    a = torch.zeros(3, 4)
    a._is_view() False, a._base None
    a[:, 1:]._base is a          a[:, 1:].t()._base is a   (the root, not the slice)
    a.view(-1)._base is a        a.split(2)[0]._base is a  a.unbind()[0]._base is a
    (a + 1)._base None           a.clone()._base None      a.detach()._base None
    r = zeros(3, requires_grad=True): r[1:]._base is r, under no_grad too
    c = a[1:]; c.add_(1): c._base is a still
    a view keeps its base alive (weakref to the base survives `del base`)

`test_a_cycle_through_a_view_is_collected` asks for more than upstream: its
docstring gives the measurement (upstream leaks that cycle) and the reason.

The one case this suite pins to the shim's answer rather than upstream's is
`test_a_view_the_shim_cannot_track_still_refuses_by_name`: upstream answers
`_base is None` for `nn.Parameter(a[:, 1:])` because `_make_subclass` detaches,
but the shim cannot tell that tensor apart from a view whose base it lost (its
layout proves it aliases part of a buffer), so it refuses by name instead of
guessing. That docstring says so again.

Nullification this suite is meant to catch:

* removing the base tracking at the dispatch door turns every `_base is ...`
  test red (the full-coverage case `test_a_full_coverage_view_answers_its_base`
  goes red even with the old layout detector, because layout cannot prove it);
* storing the immediate input instead of collapsing to the root turns
  `test_a_view_of_a_view_answers_the_root` red;
* tracking `detach` (whose schema aliases) turns
  `test_ops_that_are_not_views_answer_none` red;
* holding the base weakly turns `test_a_view_keeps_its_base_alive` red;
  holding it without a GC traverse turns `test_a_cycle_through_a_view_is_collected` red;
* answering None (rather than refusing) for an untracked view turns
  `test_a_view_the_shim_cannot_track_still_refuses_by_name` red;
* storing CreationMeta as a constant, or not propagating it, turns
  `test_creation_meta_answers_upstreams_values` red; leaving
  `_dispatch_tls_set_dispatch_key_excluded` a stub or a no-op turns
  `test_the_excluded_dispatch_key_set_is_written` red;
* dropping the view hook on the door's mode/subclass returns turns
  `test_a_fake_view_answers_its_fake_base` red;
* dropping the base tracking, or any of the autograd view metadata Dynamo reads
  after `_base` (`torch._C._autograd._get_creation_meta`,
  `_set_creation_meta`, `torch._C._dispatch_tls_set_dispatch_key_excluded`, and
  `_base` on a FakeTensor view), turns
  `test_a_compiled_module_whose_input_is_a_slice_captures_without_refusal` red
  with CompileRefused naming the missing piece.
"""

import gc
import os
import pathlib
import sys
import weakref

_TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]
_SUPPORT_DIR = _TESTS_DIR / "_support"
if str(_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPPORT_DIR))

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_VENDOR_DIR = str(_REPO_ROOT / "torchnative" / "python")
if _VENDOR_DIR not in sys.path:
    sys.path.insert(0, _VENDOR_DIR)
os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import _skip  # noqa: E402


def _torch():
    if not os.path.isfile(os.path.join(_VENDOR_DIR, "torch", "__init__.py")):
        return None
    import torch

    assert hasattr(torch._C, "_aten_implemented"), "upstream torch, not the shim"
    return torch


def _fresh(torch):
    # `torch.zeros` is a factory, so this is a root: no view op produced it.
    # `torch.arange(12.).reshape(3, 4)` would NOT be: upstream answers
    # `_is_view()` True for it (reshape of a contiguous tensor is a view).
    return torch.zeros(3, 4)


def test_a_fresh_tensor_has_no_base():
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    a = _fresh(torch)
    assert a._is_view() is False, a._is_view()
    assert a._base is None, a._base
    print("ok   viewbase: a factory tensor is not a view and its _base is None")


def test_a_slice_answers_its_base():
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    a = _fresh(torch)
    b = a[:, 1:]
    assert b._is_view() is True, b._is_view()
    assert b._base is a, (type(b._base), b._base is None)
    print("ok   viewbase: a[:, 1:]._base is a")


def test_a_view_of_a_view_answers_the_root():
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    a = _fresh(torch)
    b = a[:, 1:]
    c = b.t()
    assert c._is_view() is True
    assert c._base is a, "the base of a view of a view is the root, never the middle view"
    assert c._base is not b
    d = c.unsqueeze(0)[0]
    assert d._base is a, "a three-op chain still collapses to the root"
    print("ok   viewbase: a[:, 1:].t()._base is a, and longer chains collapse too")


def test_a_full_coverage_view_answers_its_base():
    """`a.view(-1)` covers the whole buffer contiguously from offset 0, so no
    layout signal can prove it is a view. The old detector answered
    `_is_view()` False here, the one wrong answer its docstring admitted."""
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    a = _fresh(torch)
    v = a.view(-1)
    assert v._is_view() is True, "a.view(-1) is a view upstream"
    assert v._base is a
    r = a.reshape(12)
    assert r._is_view() is True and r._base is a, "reshape of a contiguous tensor is a view"
    print("ok   viewbase: a.view(-1)._base is a and a.reshape(12)._base is a")


def test_multi_output_views_each_answer_the_root():
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    a = _fresh(torch)
    for name, parts in (("split", a.split(2)), ("unbind", a.unbind()), ("chunk", a.chunk(2))):
        for i, part in enumerate(parts):
            assert part._is_view() is True, (name, i)
            assert part._base is a, (name, i)
    print("ok   viewbase: every output of split/unbind/chunk answers the root")


def test_ops_that_are_not_views_answer_none():
    """`detach` aliases by schema (`Tensor(a) -> Tensor(a)`) but upstream makes
    it a non-differentiable alias: `_is_view()` False, `_base` None."""
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    a = _fresh(torch)
    for name, t in (("add", a + 1), ("clone", a.clone()), ("detach", a.detach()),
                    ("mul", a * 2)):
        assert t._is_view() is False, name
        assert t._base is None, name
    print("ok   viewbase: add/clone/detach/mul results are not views")


def test_is_view_agrees_with_base_wherever_base_answers():
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    a = _fresh(torch)
    tensors = [a, a[:, 1:], a.t(), a.view(-1), a[0], a.unsqueeze(0), a + 1,
               a.clone(), a[:, 1:].t(), a.expand(2, 3, 4), a.permute(1, 0)]
    for i, t in enumerate(tensors):
        assert t._is_view() is (t._base is not None), (i, t._is_view())
    print("ok   viewbase: _is_view() agrees with `_base is not None` on 11 tensors")


def test_delayed_error_output_is_not_a_view():
    """`torch._C._functions.DelayedError` is how meta_utils fakes the autograd
    history of a non-leaf input (`_backward_error`). Upstream returns
    `var.tensor_data()`: same storage, not a view, so the fake agrees with the
    real tensor's `_is_view()` False. Returning `aten.alias` (a view op by
    schema) broke every fake non-leaf input once views were tracked (#279's
    gate, run 43362953251: test_dynamo_fills, test_compile_nopep523).

    Nullification: make the fill return `torch.ops.aten.alias.default(x)`
    again and this test turns red."""
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    leaf = torch.ones(2, 3, requires_grad=True)
    nonleaf = leaf * 2
    err = torch._C._functions.DelayedError("boom 279", 1)
    out = err(nonleaf)
    assert out._is_view() is False, out._is_view()
    assert out._base is None, out._base
    assert out.data_ptr() == nonleaf.data_ptr(), "DelayedError copied instead of sharing storage"
    assert out.requires_grad and out.grad_fn is not None, (out.requires_grad, out.grad_fn)
    try:
        out.sum().backward()
    except RuntimeError as exc:
        assert "boom 279" in str(exc), exc
    else:
        raise AssertionError("backward through DelayedError did not raise")
    print("ok   viewbase: DelayedError's output shares storage, is not a view, and refuses backward")


def test_a_dense_permuted_tensor_no_view_op_made_is_not_a_view():
    """`clone` of a transpose keeps the transposed strides (upstream's
    preserve_format), covers its whole storage, and no view op made it.
    Upstream answers `_is_view()` False and `_base` None. The shim's layout rule
    used to count any non-contiguous stride as proof of a view, so it answered
    True and `_base` refused; Dynamo then could not wrap a `DynamicCache`
    layer (#256's gate, test_compile_static_cache). Measured on upstream torch
    2.13.0: clone(x.transpose(1, 2)) strides (96, 8, 16, 1), _is_view False.

    Nullification: put `!is_contiguous_inner()` back in
    `tensor.rs::layout_says_view` and this test turns red."""
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    x = torch.randn(1, 6, 2, 8).transpose(1, 2)
    assert x._is_view() and x._base is not None
    c = x.clone()
    assert not c.is_contiguous(), c.stride()
    assert c._is_view() is False, c.stride()
    assert c._base is None, c._base
    # A layout with a gap still proves an alias, tracked or not.
    p = torch.nn.Parameter(torch.zeros(3, 4)[:, ::2], requires_grad=False)
    assert p._is_view() is True, p.stride()
    print("ok   viewbase: a dense permuted clone is not a view; a strided gap still is")


def test_in_place_ops_keep_the_base():
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    a = _fresh(torch)
    c = a[1:]
    c.add_(1)
    assert c._base is a, "an in-place op on a view leaves it a view of the same base"
    assert float(a[1, 0]) == 1.0, "and the write reached the base"
    assert a._base is None, "the base did not become a view by being written through"
    print("ok   viewbase: an in-place write through a view keeps c._base is a")


def test_a_requires_grad_leaf_answers_its_views():
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    r = torch.zeros(3, requires_grad=True)
    assert r[1:]._base is r
    with torch.no_grad():
        assert r[1:]._base is r, "view tracking does not depend on grad mode"
    print("ok   viewbase: views of a requires_grad leaf answer it, with and without grad")


def test_a_view_keeps_its_base_alive():
    """Upstream's view holds a strong reference to its base (measured: a
    weakref to the base survives `del base` while the view lives). And it must
    not keep it alive after the view is gone, or every slice leaks its root."""
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    a = _fresh(torch)
    w = weakref.ref(a)
    v = a[:, 1:]
    del a
    gc.collect()
    assert w() is not None, "the base died while a view of it was alive"
    assert v._base is w()
    del v
    gc.collect()
    assert w() is None, "the base outlived its last view: the view leaked it"
    print("ok   viewbase: a view keeps its base alive, and releases it when it dies")


def test_a_cycle_through_a_view_is_collected():
    """`a.cache = a[1:]` is a cycle: a -> a.__dict__ -> view -> (base slot) -> a.
    Only a GC traverse of the base slot lets the collector see it.

    **Upstream does not pass this** (measured on 2.13.0: the weakref is still
    alive after `gc.collect()`), because its base reference lives in C++
    autograd metadata the collector cannot see. This is the one test here that
    asks for more than upstream gives. It is kept because the shim's slot is a
    Python reference it can expose to the collector for free, and a leak of the
    whole root buffer per such cycle is the cost of not doing so. If the
    coordinating session decides parity with upstream's leak is preferred,
    delete this test rather than inverting it.
    """
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    a = _fresh(torch)
    w = weakref.ref(a)
    a.cache = a[1:]
    assert a.cache._base is a
    del a
    gc.collect()
    assert w() is None, "a cycle through a view's base slot was not collected"
    print("ok   viewbase: a cycle through a view's base slot is collected")


def test_a_view_the_shim_cannot_track_still_refuses_by_name():
    """`nn.Parameter(a[:, 1:])` goes through `Tensor._make_subclass`, which
    builds a new wrapper with no record of the slice. Its layout still proves
    it aliases part of a buffer (non-zero storage offset).

    Upstream answers `_base is None` here (`_make_subclass` detaches). The
    shim cannot tell this tensor apart from a view whose base it lost, so it
    keeps refusing by name: a `None` there would be read as "not a view" one
    line after `_is_view()` said it is. A view taken of it inherits the same
    unknown root and refuses too, rather than naming the parameter as its base.
    """
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    a = _fresh(torch)
    p = torch.nn.Parameter(a[:, 1:], requires_grad=False)
    assert p._is_view() is True, "the layout proves an alias, so _is_view() says so"
    for name, t in (("parameter", p), ("view of it", p[:, 1:])):
        try:
            got = t._base
        except NotImplementedError as exc:
            assert "TensorBase._base" in str(exc), (name, str(exc))
        else:
            raise AssertionError(f"{name}: _base answered {got!r} for a view the shim cannot track")
    print("ok   viewbase: a view the shim cannot track refuses _base by name")


def test_creation_meta_answers_upstreams_values():
    """`torch._C._autograd._get_creation_meta`, measured on 2.13.0:
    `DEFAULT` for an ordinary view, `NO_GRAD_MODE` under no_grad,
    `MULTI_OUTPUT_NODE` for `split`/`unbind`, `NO_GRAD_MODE` carried to a view
    of a no_grad view made with grad on, and upstream's own `RuntimeError` on a
    tensor that is not a view. `_set_creation_meta` writes what `_get` reads."""
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    A = torch._C._autograd
    CM = A.CreationMeta
    get = A._get_creation_meta
    a = _fresh(torch)
    assert get(a[:, 1:]) == CM.DEFAULT
    assert get(a.split(2)[0]) == CM.MULTI_OUTPUT_NODE
    assert get(a.unbind()[0]) == CM.MULTI_OUTPUT_NODE
    with torch.no_grad():
        quiet = a[1:]
        assert get(a.split(2)[0]) == CM.NO_GRAD_MODE
    assert get(quiet) == CM.NO_GRAD_MODE
    assert get(quiet[1:]) == CM.NO_GRAD_MODE, "propagate_creation_meta keeps the earlier value"
    b = a[1:]
    A._set_creation_meta(b, CM.NO_GRAD_MODE)
    assert get(b) == CM.NO_GRAD_MODE
    for name, call in (("get", lambda: get(a)), ("set", lambda: A._set_creation_meta(a, CM.DEFAULT))):
        try:
            call()
        except RuntimeError as exc:
            assert "meta != nullptr" in str(exc), (name, str(exc))
        else:
            raise AssertionError(f"_{name}_creation_meta on a tensor that is not a view answered")
    print("ok   viewbase: CreationMeta get/set answer upstream's values and refusals")


def test_the_excluded_dispatch_key_set_is_written():
    """`meta_utils.py` re-enables `ADInplaceOrView` around `view_from_base` and
    restores what it read; the restore only restores if write and read agree."""
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    C = torch._C
    key = C.DispatchKey.ADInplaceOrView
    was = C._dispatch_tls_is_dispatch_key_excluded(key)
    try:
        C._dispatch_tls_set_dispatch_key_excluded(key, True)
        assert C._dispatch_tls_is_dispatch_key_excluded(key) is True
        assert C._dispatch_tls_local_exclude_set().has(key)
        C._dispatch_tls_set_dispatch_key_excluded(key, False)
        assert C._dispatch_tls_is_dispatch_key_excluded(key) is False
    finally:
        C._dispatch_tls_set_dispatch_key_excluded(key, was)
    assert C._dispatch_tls_is_dispatch_key_excluded(key) is was
    print("ok   viewbase: _dispatch_tls_set_dispatch_key_excluded writes what the reader reads")


def test_a_fake_view_answers_its_fake_base():
    """A view op under `FakeTensorMode` returns through the door's mode branch,
    not a kernel. Upstream (2.13.0): `f[:, 1:]._base is f`, and
    `mode.from_tensor(a[:, 1:])` is a fake view whose `_base` is the fake of
    `a`, with `CreationMeta.DEFAULT`."""
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    from torch._subclasses.fake_tensor import FakeTensorMode

    get = torch._C._autograd._get_creation_meta
    a = _fresh(torch)
    x = a[:, 1:]
    with FakeTensorMode() as mode:
        f = mode.from_tensor(a)
        fv = f[:, 1:]
        assert fv._is_view() is True and fv._base is f
        fx = mode.from_tensor(x)
        assert fx._is_view() is True
        assert fx._base is f, "the fake of a view is a view of the fake of its base"
        assert get(fx) == torch._C._autograd.CreationMeta.DEFAULT
        assert (tuple(fx.shape), fx.stride(), fx.storage_offset()) == ((3, 3), (4, 1), 1)
    print("ok   viewbase: fake views answer their fake base, including from_tensor of a view")


def test_a_compiled_module_whose_input_is_a_slice_captures_without_refusal():
    """The capture the NPU tests needed, on a module small enough to read.

    Dynamo wraps the sliced graph input into a FakeTensor: it reads `x._base`
    (`builder.py` `wrap_tensor` and `_automatic_dynamic`), fake-ifies the base,
    rebuilds the view from it (`meta_utils.py` `view_from_base`) and checks the
    fake view against the real one (`assert_metadata_eq`). Asserted: no
    CompileRefused, no counted failure, a graph reached the backend, and the
    answer is the eager one.
    """
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    import torch.nn as nn
    import torchnative.compile as C

    class Small(nn.Module):
        def __init__(self):
            super().__init__()
            torch.manual_seed(0)
            self.lin = nn.Linear(5, 3)

        def forward(self, x):
            return self.lin(x) * 2

    model = Small()
    buf = torch.tensor([[float(i) for i in range(6)], [float(i) for i in range(6, 12)]])
    x = buf[:, 1:]
    assert x._base is buf, "precondition: the input is a tracked view"
    expected = type(model).forward(model, x)

    graphs = []

    def recording(gm, example_inputs):
        graphs.append(gm)
        return gm.forward

    eng = C.Engine(recording)
    C.attach_to_module(model, backend=eng)
    got = model(x)  # a CompileRefused here is the failure this test exists for

    assert eng.stats["fail"] == 0, eng.stats
    assert graphs, "nothing was captured"
    assert torch.equal(got, expected), (got, expected)
    print("ok   viewbase: a module whose input is a slice compiles without refusal")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        _skip._state["current"] = name
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            if name in _skip._state["skipped"]:
                print(f"SKIP test_viewbase: {name} -- {_skip._state['skipped'][name]}")
        _skip._state["current"] = None
    raise SystemExit(1 if failures else 0)
