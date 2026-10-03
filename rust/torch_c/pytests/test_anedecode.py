"""A decode step is batch 1, and batch 1 is where the Neural Engine was lost.

`docs/platform/RELEASE_0_1_0b4.md` §3 recorded the defect: at batch 1 every
`Linear` in SmolLM2-135M is CPU-*preferred*, and only at batch 128 do four of
the five shapes become NeuralEngine. The unit is in the *supported* column
throughout, so this was never a missing capability -- CoreML weighs the work
against the dispatch and picks the CPU.

`docs/graph/ANEDECODE.md` is this round's measurement of why, and it changes
the shape of the problem in two ways:

* **The form matters and `linear` cannot win.** A rank-2 `ios16.linear` at
  batch 1 is CPU-preferred at *every* output width tried, up to 49152, and
  stays CPU-preferred in a program holding 64 of them. A 1x1 `ios16.conv` over
  a rank-4 `(1, C, 1, 1)` tensor -- Apple's `ml-ane-transformers` layout --
  computes the identical dot products and does cross to the unit. So the
  negative control here is load-bearing: without it, "conv reached the ANE"
  would be indistinguishable from "the program got big enough".

* **The unit of the decision is the program, not the operation.** The
  threshold is roughly 4.7M weights *in the compiled program*, and this
  project compiles one program per leaf, so no single projection can ever
  clear it. That is an architectural fact about the lowering, not about the
  hardware, and §4 of the document is where it is stated.

What is implemented from that is the half a per-leaf lowering can deliver:
`_CoreMLLinear` emits the conv form at batch 1, which moves `lm_head` -- the
one shape large enough on its own -- from CPU to NeuralEngine, and leaves the
other four where they were. The four are not fixed here and the document says
so.

Every assertion below is on `MLComputePlan`'s `preferred` column. `supported`
is not evidence: it was already NeuralEngine for all five shapes while all
five ran on the CPU, which is the whole reason §3 could be written.
"""

import json
import os
import subprocess
import sys

from test_shim import _CKPT_VENDOR_SHIM, _STDOUT_GUARD, _npu_fixture


_ANEDECODE_SCRIPT = r"""
import io
import json
import os
import sys

@STDOUT_GUARD@

out = {}
try:
    import coremltools as ct
    from coremltools.converters.mil import Builder as mb
    out["coremltools"] = ct.__version__
except Exception as error:
    out["coremltools"] = None
    out["import_error"] = f"{type(error).__name__}: {error}"
    print(json.dumps(out), file=_stdout, flush=True)
    raise SystemExit(0)

import numpy as np
import torch

from torchnative.export import coreml as C

UNITS = ct.ComputeUnit.ALL


def _convert(program, units=UNITS):
    return ct.convert(
        program, convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_precision=ct.precision.FLOAT16, compute_units=units)


def _preferred(program, units=UNITS):
    rows = C.computes(C.compute_plan(_convert(program, units), compute_units=units))
    return [(r["op"], r["preferred"], r["supported"]) for r in rows]


def _linear_program(i, o, batch):
    w = (np.random.randn(o, i) * 0.02).astype(np.float32)
    b = np.zeros(o, np.float32)

    @mb.program(input_specs=[mb.TensorSpec(shape=(batch, i))])
    def program(x):
        return mb.linear(x=x, weight=w, bias=b)
    return program


def _conv_chain_program(count, i=576, o=576):
    pairs = [((np.random.randn(o, i, 1, 1) * 0.02).astype(np.float32),
              np.zeros(o, np.float32)) for _ in range(count)]

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, i, 1, 1))])
    def program(x):
        for w, b in pairs:
            x = mb.conv(x=x, weight=w, bias=b, strides=[1, 1],
                        pad_type="custom", pad=[0, 0, 0, 0],
                        dilations=[1, 1], groups=1)
        return x
    return program


def _linear_chain_program(count, i=576, o=576):
    pairs = [((np.random.randn(o, i) * 0.02).astype(np.float32),
              np.zeros(o, np.float32)) for _ in range(count)]

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, i))])
    def program(x):
        for w, b in pairs:
            x = mb.linear(x=x, weight=w, bias=b)
        return x
    return program


# SmolLM2-135M's decoder-layer projections: q, k, v, o, gate, up, down.
LAYER = [(576, 576), (576, 192), (576, 192), (576, 576),
         (576, 1536), (576, 1536), (1536, 576)]


def _decode_layers_program(count):
    blocks = [[((np.random.randn(o, i, 1, 1) * 0.02).astype(np.float32),
                np.zeros(o, np.float32)) for (i, o) in LAYER]
              for _ in range(count)]

    def conv(x, wb):
        w, b = wb
        return mb.conv(x=x, weight=w, bias=b, strides=[1, 1],
                       pad_type="custom", pad=[0, 0, 0, 0],
                       dilations=[1, 1], groups=1)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, 576, 1, 1))])
    def program(x):
        for blk in blocks:
            q = conv(x, blk[0])
            conv(x, blk[1])
            conv(x, blk[2])
            x = mb.add(x=x, y=conv(q, blk[3]))
            g = mb.silu(x=conv(x, blk[4]))
            u = conv(x, blk[5])
            x = mb.add(x=x, y=conv(mb.mul(x=g, y=u), blk[6]))
        return x
    return program


torch.manual_seed(0)

# -- the leaf, through the public class, at the five real shapes -----------
SHAPES = [(576, 192), (576, 576), (576, 1536), (1536, 576), (576, 49152)]
leaf = {}
for (i, o) in SHAPES:
    for batch in (1, 128):
        layer = torch.nn.Linear(i, o)
        cell = C._CoreMLLinear.from_torch(layer, precision="float16")
        cell._compile_for(batch, probe=True)
        rows = C.computes(cell._report["plans"][-1]["rows"])
        leaf[f"{i}x{o}x{batch}"] = {
            "preferred": sorted({r["preferred"] for r in rows}),
            "supported": sorted({d for r in rows for d in r["supported"]}),
            "ops": sorted({r["op"] for r in rows}),
        }
out["leaf"] = leaf

# -- the negative control: rank-2 linear, batch 1, every width ------------
out["linear_widths"] = {
    str(o): [p for _, p, _ in _preferred(_linear_program(576, o, 1))]
    for o in (192, 576, 1536, 4096, 8192, 16384, 49152)
}
out["linear_chain"] = {
    str(n): sorted({p for _, p, _ in _preferred(_linear_chain_program(n))})
    for n in (1, 16, 64)
}
out["conv_chain"] = {
    str(n): sorted({p for _, p, _ in _preferred(_conv_chain_program(n))})
    for n in (1, 8, 16)
}

# -- a whole decode step's worth of projections in one program ------------
out["decode_layers"] = {
    str(n): sorted({p for _, p, _ in _preferred(_decode_layers_program(n))})
    for n in (1, 2)
}

# -- can any compute_units setting force the unit at batch 1? -------------
forcing = {}
for name in ("ALL", "CPU_AND_NE"):
    units = getattr(ct.ComputeUnit, name)
    rows = _preferred(_linear_program(576, 576, 1), units)
    forcing[name] = {"preferred": [p for _, p, _ in rows],
                     "supported": sorted({d for _, _, s in rows for d in s})}
out["forcing"] = forcing

# -- does a flexible (symbolic) dimension change the decision? ------------
from coremltools.converters.mil.mil import get_new_symbol

flex = {}
w_head = (np.random.randn(49152, 576) * 0.02).astype(np.float32)
b_head = np.zeros(49152, np.float32)
wc_head = w_head.reshape(49152, 576, 1, 1)


def _head_conv(x):
    return mb.conv(x=x, weight=wc_head, bias=b_head, strides=[1, 1],
                   pad_type="custom", pad=[0, 0, 0, 0],
                   dilations=[1, 1], groups=1)


def _head_linear(x):
    return mb.linear(x=x, weight=w_head, bias=b_head)


for label, shape, fn in (
    ("conv_static", (1, 576, 1, 1), _head_conv),
    ("conv_symbolic", (1, 576, 1, get_new_symbol()), _head_conv),
    ("linear_static", (1, 576), _head_linear),
    ("linear_symbolic", (get_new_symbol(), 576), _head_linear),
):
    program = mb.program(input_specs=[mb.TensorSpec(shape=shape)])(fn)
    flex[label] = sorted({p for _, p, _ in _preferred(program)})
try:
    from coremltools.converters.mil.input_types import RangeDim
    mb.TensorSpec(shape=(1, 576, 1, RangeDim(1, 128)))
    flex["rangedim_accepted"] = True
except Exception as error:
    flex["rangedim_accepted"] = f"{type(error).__name__}"
out["flexible"] = flex

# -- what the rewrite costs numerically -----------------------------------
rng = np.random.default_rng(0)
agree = {}
for (i, o) in ((576, 576), (576, 49152)):
    w = (rng.standard_normal((o, i)) * 0.02).astype(np.float32)
    b = (rng.standard_normal(o) * 0.01).astype(np.float32)
    x = rng.standard_normal((1, i)).astype(np.float32)
    reference = (x @ w.T + b).reshape(-1)
    scale = float(np.abs(reference).max())

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, i))])
    def lin(x):
        return mb.linear(x=x, weight=w, bias=b)

    wc = w.reshape(o, i, 1, 1)

    @mb.program(input_specs=[mb.TensorSpec(shape=(1, i, 1, 1))])
    def cnv(x):
        return mb.conv(x=x, weight=wc, bias=b, strides=[1, 1],
                       pad_type="custom", pad=[0, 0, 0, 0],
                       dilations=[1, 1], groups=1)

    feeds = {}
    ml = _convert(lin)
    fl = C._feed_buffer(feeds, "l", (1, i))
    fl[...] = x
    got_l = np.asarray(list(C._predict(
        ml, {ml.get_spec().description.input[0].name: fl}).values())[0]
    ).reshape(-1)
    mc = _convert(cnv)
    fc = C._feed_buffer(feeds, "c", (1, i, 1, 1))
    fc[...] = x.reshape(1, i, 1, 1)
    got_c = np.asarray(list(C._predict(
        mc, {mc.get_spec().description.input[0].name: fc}).values())[0]
    ).reshape(-1)
    agree[f"{i}x{o}"] = {
        "linear_vs_float32": float(np.abs(got_l - reference).max() / scale),
        "conv_vs_float32": float(np.abs(got_c - reference).max() / scale),
        "conv_vs_linear": float(np.abs(got_c - got_l).max() / scale),
        "same_argmax": bool(got_c.argmax() == got_l.argmax()
                            == reference.argmax()),
    }
out["agreement"] = agree

print(json.dumps(out), file=_stdout, flush=True)
"""


_CACHE = {}


def _fixture_or_skip():
    """The measurement, run once, or `None` with a named reason."""
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        print("skip anedecode: the vendored shim build is absent")
        return None
    if "r" not in _CACHE:
        _CACHE["r"] = _npu_fixture(
            _ANEDECODE_SCRIPT.replace("@STDOUT_GUARD@", _STDOUT_GUARD))
    result = _CACHE["r"]
    if result.get("coremltools") is None:
        print(f"skip anedecode: coremltools absent "
              f"({result.get('import_error')})")
        return None
    return result


#: The float16 grade `docs/platform/RELEASE_0_1_0b4.md` §3 already named --
#: 1.5e-03 -- and **not** widened here. The conv rewrite has to live inside the
#: number that was already published, or it is buying the unit with accuracy.
FLOAT16_GRADE = 1.5e-03


def test_lm_head_reaches_the_neural_engine_at_batch_one():
    """The one shape a per-leaf program can carry to the unit on its own.

    576->49152 is 28.3M weights, which clears the ~4.7M threshold
    `docs/graph/ANEDECODE.md` §3 measures, so it does not need the rest of the
    model for company. It is also the shape §3 of the release notes singled out
    as CPU at batch 128 *as well* -- and the reason turns out not to be that it
    is unusually hard for the unit, but that `linear` is the wrong form at any
    size. In the conv form it is the *easiest* of the five.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    got = r["leaf"]["576x49152x1"]
    assert got["preferred"] == ["NeuralEngine"], got


def test_the_four_projection_shapes_are_still_cpu_at_batch_one():
    """Stated as a test so it cannot quietly be believed to have been fixed.

    576->192, 576->576, 576->1536 and 1536->576 hold between 0.11M and 0.88M
    weights each -- an order of magnitude under the threshold -- and one
    program per leaf gives them nothing to be scheduled with. This round does
    not move them, and a test that asserts the failure is how the next round
    finds out the day it changes.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    for shape in ("576x192", "576x576", "576x1536", "1536x576"):
        got = r["leaf"][f"{shape}x1"]
        assert got["preferred"] == ["CPU"], (shape, got)
        # ...and `supported` still says NeuralEngine, which is exactly why
        # `supported` is not evidence of anything.
        assert "NeuralEngine" in got["supported"], (shape, got)


def test_the_rewrite_is_confined_to_batch_one():
    """Above batch 1 the conv form is *worse*, so the leaf must not use it.

    A `(1, 576, 1, S)` conv stays CPU-preferred out to S=128, while a rank-2
    `linear` at batch 128 is NeuralEngine -- the prefill story §3 does report
    as working. Applying the rewrite unconditionally would trade a measured win
    for a measured loss, so the four shapes that work at 128 are checked here
    rather than assumed.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    for shape in ("576x192", "576x576", "576x1536", "1536x576"):
        got = r["leaf"][f"{shape}x128"]
        assert got["preferred"] == ["NeuralEngine"], (shape, got)


def test_rank_two_linear_never_reaches_the_unit_at_batch_one():
    """The negative control that makes the conv form load-bearing.

    If `linear` also crossed once it got big enough, the win above would be a
    size effect wearing a layout's clothes. It does not: CPU at every width out
    to 49152, and CPU in a program holding 64 of them (21M weights, four times
    the threshold a chain of convs crosses at).
    """
    r = _fixture_or_skip()
    if r is None:
        return
    for width, preferred in r["linear_widths"].items():
        assert set(preferred) == {"CPU"}, (width, preferred)
    for count, preferred in r["linear_chain"].items():
        assert preferred == ["CPU"], (count, preferred)


def test_the_scheduling_unit_is_the_program_and_not_the_operation():
    """Identical convs, identical shapes -- and the answer follows the count.

    Eight 576->576 convs in one program are CPU-preferred and sixteen are
    NeuralEngine-preferred, with no operation changed. So "which unit runs this
    leaf" has no answer that is a property of the leaf, and a lowering that
    emits one program per leaf has already decided it.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    chain = r["conv_chain"]
    assert chain["1"] == ["CPU"], chain
    assert chain["8"] == ["CPU"], chain
    assert chain["16"] == ["NeuralEngine"], chain


def test_a_two_layer_decode_step_is_neural_engine_preferred_throughout():
    """The result this round exists to record, and it is not a leaf result.

    Two SmolLM2 decoder layers' projections as 1x1 convs, at batch 1, with the
    `silu`, `mul` and `add` a decode step really contains: **every operation**
    is NeuralEngine-preferred. One layer is not -- 3.54M weights is under the
    threshold.

    That also withdraws a negative from `RELEASE_0_1_0b4.md` §3, which recorded
    `silu`-class elementwise leaves as never ANE-*preferred*. They were
    measured as one-operation programs, which hold no weights at all and so can
    never clear the threshold; in a program that does clear it they are
    preferred on the unit like everything else.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    assert r["decode_layers"]["1"] == ["CPU"], r["decode_layers"]
    assert r["decode_layers"]["2"] == ["NeuralEngine"], r["decode_layers"]


def test_no_compute_units_setting_forces_the_unit_at_batch_one():
    """A negative result, recorded because it is the first thing to try.

    `CPU_AND_NE` removes the GPU from the *supported* column and leaves
    `preferred` exactly where `ALL` left it. `MLComputePlan` has no forcing
    knob: `computeUnits` bounds the choice, it does not make it.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    forcing = r["forcing"]
    assert forcing["ALL"]["preferred"] == forcing["CPU_AND_NE"]["preferred"], forcing
    assert set(forcing["ALL"]["preferred"]) == {"CPU"}, forcing
    assert "GPU" in forcing["ALL"]["supported"], forcing
    assert "GPU" not in forcing["CPU_AND_NE"]["supported"], forcing


def test_the_conv_form_costs_no_accuracy_against_the_form_it_replaces():
    """Bit-identical at 576->576, and inside the published grade at lm_head.

    The tolerance is `RELEASE_0_1_0b4.md` §3's own 1.5e-03 float16 number and
    is not widened: a rewrite that reached the unit by spending accuracy would
    be a different trade and would have to be named as one. At lm_head the two
    forms differ by 1.3e-03 *because they run on different units*, and there
    the conv form is the closer of the two to float32 -- so the difference is
    not the rewrite degrading the answer.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    small = r["agreement"]["576x576"]
    assert small["conv_vs_linear"] == 0.0, small
    head = r["agreement"]["576x49152"]
    assert head["conv_vs_linear"] <= FLOAT16_GRADE, head
    assert head["conv_vs_float32"] <= head["linear_vs_float32"], head
    for shape, got in r["agreement"].items():
        assert got["same_argmax"], (shape, got)


def test_a_flexible_input_dimension_does_not_change_the_decision():
    """Measured, because "we already use the favourable case" is an argument.

    Pinning a static batch-1 shape is the obvious lever to try and it turns out
    not to be one: a symbolic dimension neither costs the conv form the unit
    nor buys the linear form it. `mb.TensorSpec` also rejects `RangeDim`
    outright -- that is a frontend `ct.TensorType` concept and never reaches a
    program built through `mb.program` -- so "enumerated shapes" is not a
    setting this lowering could offer even if it wanted to.
    """
    r = _fixture_or_skip()
    if r is None:
        return
    flex = r["flexible"]
    assert flex["conv_static"] == ["NeuralEngine"], flex
    assert flex["conv_symbolic"] == ["NeuralEngine"], flex
    assert flex["linear_static"] == ["CPU"], flex
    assert flex["linear_symbolic"] == ["CPU"], flex
    assert flex["rangedim_accepted"] == "ValueError", flex


def test_the_stdout_guard_carries_its_own_imports():
    """The defect that made this file's nine tests report as a subprocess exit.

    Every fixture in this repository that talks to CoreML splices
    `test_shim._STDOUT_GUARD` into a script handed to a fresh interpreter,
    because the ANE compiler writes to fd 1 directly and corrupts the JSON
    line the parent parses. The guard used to *require* the splicing script to
    have done `import io, os, sys` first, and that requirement was recorded
    only in a comment. This file imported `json`, `os` and `sys`; the guard
    reached `sys.stdout = io.StringIO()` and raised `NameError`, so all nine
    tests below failed identically with `npu subprocess exited 1` -- a message
    about the exit status, naming nothing.

    So the guard imports what it uses. This runs it against a script that
    imports *nothing*, which is the case the old form could not survive: if
    the imports are taken back out of `_STDOUT_GUARD`, this goes red with the
    same `NameError`, in one test instead of nine, and says which module.
    """
    script = _STDOUT_GUARD + 'print("guarded", file=_stdout, flush=True)\n'
    proc = subprocess.run([sys.executable, "-c", script],
                          capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, (proc.returncode, proc.stderr)
    assert proc.stdout.strip() == "guarded", (proc.stdout, proc.stderr)


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
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
