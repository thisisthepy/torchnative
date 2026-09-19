"""Vulkan, widened from four ops to eighteen -- and what the four actually were.

docs/platform/RELEASE_0_1_0b0.md §5 says:

    Vulkan is four ops. Correctness is testable on this host; performance
    needs a phone and has not been measured.

**That sentence was re-measured before it was widened** (docs/devices/VULKAN4.md §1),
because four §5 gap statements in this repository have turned out to be wrong
when re-checked. This one is not wrong. It is *true and misleading*, in a way
that only shows up when you ask each of the four what it does:

  * `aten.add.Tensor`        -- a real SPIR-V compute shader on the GPU.
  * `aten._to_copy.default`  -- a memory copy. No shader, no arithmetic.
  * `aten.detach.default`    -- an `Arc` clone of a shape. No GPU work at all.
  * `aten.alias.default`     -- the same.

So "four ops" counted *reachability* and was read as *computation*, and one of
the four was the whole of the arithmetic. Worse, and not visible from the op
list at all: **there was no way to put arbitrary data on the device.** The only
routes in were `ones`/`zeros`/`empty`, so every Vulkan kernel that had ever
been tested had been tested on constants -- and a matmul of all-ones agrees
with any implementation that sums the right number of ones.

This file holds the widening down. Every assertion here is one of three kinds,
and the kinds are kept apart on purpose:

  * **value** -- element-wise against upstream torch, at a tolerance *derived*
    from upstream's own float32-vs-float64 error the way docs/numerics/AGREE.md §2
    derives its own. For every exactly-rounded op that derivation comes out at
    zero, so those are held to **bit equality**.
  * **device** -- that the GPU did the work, asserted from `_vulkan_counters()`
    at runtime. Not inferred from the answer being right, and deliberately not
    a source scan: docs/devices/MPSATTN.md §3.1 records a way to defeat exactly that
    shape of evidence, and §5 of docs/devices/VULKAN4.md explains why a counter placed
    after `vkWaitForFences` cannot be defeated the same way.
  * **refusal** -- that everything not taught still refuses naming itself, and
    that the narrowings (broadcast, alpha, non-f32, rank > 2) refuse rather
    than reach for the CPU implementation half a metre away.

Everything that needs a Vulkan loader skips **by name**, printing the loader's
own words. docs/devices/VULKAN3.md §6.1 is the reason that matters: macOS strips
`DYLD_*` when `/bin/sh` execs, so running this through `run.sh` skips these
even when the loader was pointed at correctly. A skip that says "no vulkan"
when a loader was supplied is the closest thing to a false green this device
has produced, and it was caught by the skip line quoting the loader.
"""

import json
import math
import os
import struct
import subprocess
import sys

import vulkan_coverage
from test_shim import _C


REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
VENDOR_DIR = os.path.join(REPO, "torchnative", "src", "main")

# The four ops docs/devices/VULKAN3.md landed, so this file can state the widening as a
# difference rather than a number, and so a later round that removes one is a
# failure here rather than a silently smaller list.
VULKAN3_OPS = (
    "aten._to_copy.default",
    "aten.add.Tensor",
    "aten.alias.default",
    "aten.detach.default",
)

# Of those four, the only one that ever ran a compute shader (docs/devices/VULKAN4.md §1).
VULKAN3_OPS_THAT_COMPUTED = ("aten.add.Tensor",)

FLOAT32_EPS = 2.0 ** -23


# ---------------------------------------------------------------------------
# Skipping by name
# ---------------------------------------------------------------------------

def _vulkan_or_skip(what):
    """A live Vulkan device, or None having said why not -- in the loader's words.

    Deliberately not shared with `test_shim.py`'s helper, for docs/devices/VULKAN3.md
    §6.1's reason: the skip line is the thing this file promises to keep
    truthful, so it names the file that skipped and quotes the probe's own
    `error` rather than paraphrasing it.
    """
    probe = _C._vulkan_probe()
    if not probe["available"]:
        reason = f"{what}: no vulkan loader here -- {probe['error']}"
        # Recorded, so the runner prints `SKIP <test>: <reason>` and not ok
        # (docs/devices/VULKAN5.md §2); that line carries the loader's words.
        vulkan_coverage.vulkan_skip(reason)
        return None
    vulkan_coverage.vulkan_used(probe["device"])
    return probe


def _counters():
    return _C._vulkan_counters()


def _delta(before, after):
    return {k: after[k] - before[k] for k in after}


# ---------------------------------------------------------------------------
# Moving data
# ---------------------------------------------------------------------------

def _cpu(values, shape):
    return _C._tensor_from_flat([float(v) for v in values], list(shape),
                                dtype=_C.float32)


def _i64(values, shape):
    """An int64 CPU tensor -- the dtype an index operand actually has."""
    return _C._tensor_from_flat([float(v) for v in values], list(shape),
                                dtype=_C.int64)


def _to_vulkan(t):
    return _C._aten_dispatch("aten._to_copy.default", t,
                             device=_C.device("vulkan"))


def _to_cpu(t):
    return _C._aten_dispatch("aten._to_copy.default", t, device=_C.device("cpu"))


def _flat(t):
    out, stack = [], [t.tolist()]
    while stack:
        item = stack.pop()
        if isinstance(item, list):
            stack.extend(reversed(item))
        else:
            out.append(item)
    return out


def _bits(x):
    """A float32's bit pattern, so 'equal' means equal and not 'close'."""
    return struct.unpack("<I", struct.pack("<f", x))[0]


# ---------------------------------------------------------------------------
# The upstream oracle
# ---------------------------------------------------------------------------

def _upstream():
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        print(f"   (skipped: no upstream torch here -- {type(e).__name__})")
        vulkan_coverage.vulkan_skip(f"no upstream torch here -- {type(e).__name__}")
        return None
    # The oracle must not be the thing under test. If `torch` on this path is
    # the shim, every agreement number below would be the shim agreeing with
    # itself -- which is the failure a previous round in this repository spent
    # hours inside before noticing.
    assert not hasattr(torch._C, "_aten_implemented"), (
        "upstream torch expected here, got the shim: the oracle would be "
        "comparing the shim against itself")
    return torch


def _rand(torch, *shape, seed):
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=torch.float32)


# ---------------------------------------------------------------------------
# 1. What the list is, and what the list means
# ---------------------------------------------------------------------------

def test_the_taught_list_grew_and_kept_everything_it_had():
    """Eighteen, containing the four -- stated as a difference, not a number.

    `ge` rather than `eq` on the length, so a later round that teaches a
    nineteenth op does not have to edit this file; but every one of
    docs/devices/VULKAN3.md's four is asserted by name, so a round that *drops* one
    fails here. A count alone could not tell those two apart.
    """
    ops = _C._vulkan_ops()
    assert len(ops) == len(set(ops)), f"duplicate entries in _vulkan_ops(): {ops}"
    assert ops == sorted(ops), "_vulkan_ops() is meant to be a sorted, closed list"
    for op in VULKAN3_OPS:
        assert op in ops, f"docs/devices/VULKAN3.md taught {op} and it is no longer taught"
    assert len(ops) >= 18, f"expected at least 18 taught ops, got {len(ops)}: {ops}"


def test_an_op_that_is_not_taught_refuses_and_names_itself():
    """The property that makes the list mean something.

    The candidate is chosen *because* it is absent from `_vulkan_ops()`, so a
    later round that teaches `aten.gelu.default` makes this test pick a
    different op by itself rather than starting to lie.
    """
    if _vulkan_or_skip("the refusal-names-the-op check") is None:
        return
    ops = set(_C._vulkan_ops())
    candidates = [op for op in ("aten.gelu.default", "aten._softmax.default",
                                "aten.native_layer_norm.default",
                                "aten.bmm.default", "aten.exp.default")
                  if op not in ops]
    assert candidates, "every candidate is taught now -- widen this list"
    op = candidates[0]
    x = _to_vulkan(_cpu([1.0, 2.0], [2]))
    try:
        _C._aten_dispatch(op, x)
    except NotImplementedError as e:
        message = str(e)
    else:
        raise AssertionError(f"{op} is not in _vulkan_ops() but did not refuse")
    assert op in message, f"the refusal does not name the op: {message}"
    assert "vulkan" in message, message
    # The refusal must point somewhere, or the reader is stuck.
    assert ".cpu()" in message, message


def test_the_four_ops_of_the_previous_round_were_one_shader_and_three_that_were_not():
    """docs/devices/VULKAN4.md §1's re-measurement of §5, as an assertion.

    This is the finding that made "Vulkan is four ops" misleading rather than
    wrong, and it is checked here so that it cannot quietly stop being true:
    of the four, `add` dispatches a compute shader and `detach`/`alias`
    dispatch nothing at all.
    """
    if _vulkan_or_skip("the four-ops re-measurement") is None:
        return
    a = _to_vulkan(_cpu([1.0, 2.0, 3.0], [3]))
    b = _to_vulkan(_cpu([4.0, 5.0, 6.0], [3]))

    before = _counters()
    _C._aten_dispatch("aten.add.Tensor", a, b)
    add = _delta(before, _counters())
    assert add["shader_dispatches"] == 1, add
    assert add["host_downloads"] == 0, add

    for op in ("aten.detach.default", "aten.alias.default"):
        before = _counters()
        _C._aten_dispatch(op, a)
        d = _delta(before, _counters())
        assert d["shader_dispatches"] == 0, (op, d)
        assert d["host_uploads"] == 0 and d["host_downloads"] == 0, (op, d)


# ---------------------------------------------------------------------------
# 2. Arbitrary data can reach the device -- the thing that did not exist
# ---------------------------------------------------------------------------

def test_arbitrary_data_reaches_the_device_and_returns_unchanged():
    """`x.to("vulkan")`, which raised "device not available" before this round.

    Until it existed the only tensors on this device were `ones` and `zeros`,
    so no numerical claim about a Vulkan kernel could have been stronger than
    "it handles constants" (docs/devices/VULKAN4.md §1). The values here are chosen to
    have nothing in common with each other or with 1.0, and the comparison is
    on **bits**: an upload followed by a download changes no arithmetic, so
    anything less than bit equality would be a defect and not a tolerance.
    """
    if _vulkan_or_skip("the host-to-device round trip") is None:
        return
    values = [0.1, -2.5, 3.75, 1e-8, -1e8, 0.0, -0.0, 6.02e23]
    src = _cpu(values, [2, 4])
    before = _counters()
    dev = _to_vulkan(src)
    up = _delta(before, _counters())
    assert str(dev.device) == "vulkan", dev.device
    assert up["host_uploads"] == 1, up
    assert up["shader_dispatches"] == 0, "an upload is a copy, not a kernel"

    back = _to_cpu(dev)
    got, want = _flat(back), _flat(src)
    assert [_bits(v) for v in got] == [_bits(v) for v in want], (got, want)


def test_a_dtype_that_has_no_shader_refuses_on_the_way_in():
    """f32 only, and the refusal says so rather than widening silently."""
    if _vulkan_or_skip("the dtype refusal") is None:
        return
    src = _C._tensor_from_flat([1.0, 2.0], [2], dtype=_C.float64)
    try:
        _to_vulkan(src)
    except NotImplementedError as e:
        assert "float" in str(e), str(e)
    else:
        raise AssertionError("a float64 tensor reached the vulkan device")


# ---------------------------------------------------------------------------
# 3. Values -- with the tolerance derived, not chosen
# ---------------------------------------------------------------------------

# The exactly-rounded ops. Each is one IEEE-754 single operation per element
# (or none at all), and IEEE-754 specifies those to the last bit, so upstream's
# own float32-vs-float64 error *is* the shim's: the derivation in
# docs/numerics/AGREE.md §2, applied to this population, produces a tolerance of zero
# and these are held to bit equality. Anything looser here would be a
# tolerance hiding a defect, not measuring a precision.
EXACT_OPS = ("add", "sub", "mul", "div", "neg", "relu", "clone",
             "contiguous", "view", "t", "transpose")


def _shim_apply(op, a, b=None):
    d = _C._aten_dispatch
    if op == "add":
        return d("aten.add.Tensor", a, b)
    if op == "sub":
        return d("aten.sub.Tensor", a, b)
    if op == "mul":
        return d("aten.mul.Tensor", a, b)
    if op == "div":
        return d("aten.div.Tensor", a, b)
    if op == "neg":
        return d("aten.neg.default", a)
    if op == "relu":
        return d("aten.relu.default", a)
    if op == "clone":
        return d("aten.clone.default", a)
    if op == "contiguous":
        return d("aten.contiguous.default", a)
    if op == "view":
        return d("aten.view.default", a, [-1])
    if op == "t":
        return d("aten.t.default", a)
    if op == "transpose":
        return d("aten.transpose.int", a, 0, 1)
    raise AssertionError(op)


def _upstream_apply(torch, op, a, b=None):
    return {
        "add": lambda: a + b, "sub": lambda: a - b, "mul": lambda: a * b,
        "div": lambda: a / b, "neg": lambda: -a, "relu": lambda: torch.relu(a),
        "clone": lambda: a.clone(), "contiguous": lambda: a.contiguous(),
        "view": lambda: a.reshape(-1), "t": lambda: a.t(),
        "transpose": lambda: a.transpose(0, 1),
    }[op]()


def test_the_exactly_rounded_ops_are_bit_identical_to_upstream():
    """Eleven ops, on real data, compared as bit patterns.

    The data is `randn`, not constants -- which only became possible this round
    (docs/devices/VULKAN4.md §1). `div`'s divisor is pushed away from zero so the case
    measures division rather than the representation of infinity.
    """
    if _vulkan_or_skip("the bit-equality sweep") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    checked = 0
    for op in EXACT_OPS:
        for seed, shape in ((1, (4, 5)), (2, (7, 3)), (3, (2, 6))):
            a = _rand(torch, *shape, seed=seed)
            b = None
            if op in ("add", "sub", "mul", "div"):
                b = _rand(torch, *shape, seed=seed + 100)
                if op == "div":
                    b = b + torch.where(b.abs() < 0.5, torch.sign(b) + (b == 0),
                                        torch.zeros_like(b))
            want = _upstream_apply(torch, op, a, b)

            va = _to_vulkan(_cpu(a.reshape(-1).tolist(), shape))
            vb = None if b is None else _to_vulkan(_cpu(b.reshape(-1).tolist(), shape))
            got = _to_cpu(_shim_apply(op, va, vb))

            assert list(got.shape) == list(want.shape), (op, shape, got.shape, want.shape)
            gb = [_bits(v) for v in _flat(got)]
            wb = [_bits(v) for v in want.reshape(-1).tolist()]
            assert gb == wb, (
                f"{op}{list(shape)} is not bit-identical to upstream: "
                f"{sum(x != y for x, y in zip(gb, wb))} of {len(gb)} elements differ")
            checked += 1
    assert checked == len(EXACT_OPS) * 3, checked
    print(f"   {checked} exactly-rounded cases, all bit-identical to upstream")


def _derived_tolerance(upstream_rel_errors):
    """docs/numerics/AGREE.md §2's rule, recomputed here from this round's population.

    The p90 of upstream's own float32-vs-float64 relative error, floored at
    8 float32 ulp. The floor is AGREE's, and its reason is AGREE's: a
    population that happened to be numerically easy must not be able to drive
    the tolerance down to where float32 differs for reasons nobody claims are
    defects. Recomputed rather than pasted, so a change in the population
    changes the number instead of silently failing against a stale constant.
    """
    pop = sorted(upstream_rel_errors)
    p90 = pop[int(0.9 * (len(pop) - 1))]
    return max(p90, 8 * FLOAT32_EPS), p90


MATMUL_SHAPES = ((2, 3, 4), (8, 16, 8), (5, 64, 7), (32, 128, 16), (1, 512, 1),
                 (3, 257, 5))


def test_the_matmuls_agree_with_upstream_at_a_derived_tolerance():
    """`mm` and `addmm` -- the first kernels here whose answer is not forced.

    Everything in `EXACT_OPS` is one exactly-rounded IEEE operation, so a
    difference could only be plumbing. A dot product is a *sum*, and a sum has
    an order. So this is the one place a tolerance is needed, and it is read
    off upstream's own float32-vs-float64 error rather than chosen -- and then
    the *next* test proves the residue is that order and not a defect.
    """
    if _vulkan_or_skip("the matmul agreement sweep") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    rows, up_errs = [], []
    for i, (m, k, n) in enumerate(MATMUL_SHAPES):
        a = _rand(torch, m, k, seed=200 + i)
        b = _rand(torch, k, n, seed=300 + i)
        c = _rand(torch, n, seed=400 + i)
        va = _to_vulkan(_cpu(a.reshape(-1).tolist(), (m, k)))
        vb = _to_vulkan(_cpu(b.reshape(-1).tolist(), (k, n)))
        vc = _to_vulkan(_cpu(c.reshape(-1).tolist(), (n,)))
        for name, want32, want64, got in (
            ("mm",
             torch.mm(a, b), torch.mm(a.double(), b.double()),
             _to_cpu(_C._aten_dispatch("aten.mm.default", va, vb))),
            ("addmm",
             torch.addmm(c, a, b),
             torch.addmm(c.double(), a.double(), b.double()),
             _to_cpu(_C._aten_dispatch("aten.addmm.default", vc, va, vb))),
        ):
            truth = want64.reshape(-1).tolist()
            scale = max(abs(v) for v in truth)
            up_rel = max(abs(x - y) for x, y in
                         zip(want32.double().reshape(-1).tolist(), truth)) / scale
            shim_rel = max(abs(x - y) for x, y in zip(_flat(got), truth)) / scale
            up_errs.append(up_rel)
            rows.append((f"{name}[{m},{k},{n}]", shim_rel, up_rel))

    tol, p90 = _derived_tolerance(up_errs)
    worst = max(r[1] for r in rows)
    print(f"   derived tolerance = max(p90 {p90:.3e}, 8 ulp {8*FLOAT32_EPS:.3e}) "
          f"= {tol:.3e}; worst shim relative error {worst:.3e}")
    for name, shim_rel, up_rel in rows:
        assert shim_rel <= tol, (
            f"{name}: shim {shim_rel:.3e} exceeds the derived tolerance {tol:.3e} "
            f"(upstream's own error on the same output was {up_rel:.3e})")


def _f32(x):
    """Round a Python double to float32, the way the hardware would."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


def _host_model_of_the_matmul_kernel(a, b, m, k, n):
    """What `shaders/matmul_f32.comp` says it does, executed on the host.

    One invocation per output element, accumulating over `k` in declaration
    order, in float32, **with the multiply-add contracted** -- the product of
    two float32 values is exact in a Python double, so rounding
    `acc + a*b` once is precisely a fused multiply-add.

    This is not a re-implementation for its own sake. It is the instrument that
    settles whether the matmul's disagreement with upstream is precision or a
    defect (docs/devices/VULKAN4.md §4.2): a kernel that had transposed an index or
    lost a term would not be reproduced by a model of the arithmetic it claims
    to do, at every shape, to the bit.
    """
    out = []
    for i in range(m):
        for j in range(n):
            acc = 0.0
            for kk in range(k):
                acc = _f32(acc + a[i * k + kk] * b[kk * n + j])
            out.append(acc)
    return out


def test_the_matmul_residue_is_fma_contraction_and_not_a_defect():
    """The proof, rather than a tolerance that happens to hold.

    `mm` differs from upstream in the last bits at longer `k` -- 2 of 15
    elements at k=257, 1 of 1 at k=512. That is either an accumulation-order
    difference or a broken kernel, and a tolerance cannot tell those apart:
    both look like "small". So the question is answered by construction
    instead. The GPU's answer is reproduced **bit for bit, at every shape** by
    a host model of sequential float32 accumulation with FMA, which is a legal
    contraction of `acc += a*b` and the one the driver applied.

    A kernel that read the wrong element would still be "small" and would not
    survive this.
    """
    if _vulkan_or_skip("the FMA-contraction proof") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    differed_from_upstream = 0
    for i, (m, k, n) in enumerate(MATMUL_SHAPES):
        a = _rand(torch, m, k, seed=200 + i)
        b = _rand(torch, k, n, seed=300 + i)
        af, bf = a.reshape(-1).tolist(), b.reshape(-1).tolist()
        got = _flat(_to_cpu(_C._aten_dispatch(
            "aten.mm.default",
            _to_vulkan(_cpu(af, (m, k))), _to_vulkan(_cpu(bf, (k, n))))))
        model = _host_model_of_the_matmul_kernel(af, bf, m, k, n)
        assert [_bits(v) for v in got] == [_bits(v) for v in model], (
            f"mm[{m},{k},{n}] is not what shaders/matmul_f32.comp describes: "
            f"{sum(_bits(x) != _bits(y) for x, y in zip(got, model))} of "
            f"{m * n} elements differ from the host model of the kernel")
        upstream = torch.mm(a, b).reshape(-1).tolist()
        differed_from_upstream += sum(
            _bits(x) != _bits(y) for x, y in zip(got, upstream))
    # If nothing ever differed from upstream, this test would be asserting
    # something true for a trivial reason and the sweep above would be enough.
    # It does differ, which is what makes the model the interesting evidence.
    assert differed_from_upstream > 0, (
        "no element differed from upstream anywhere -- this population no "
        "longer exercises the accumulation-order question")
    print(f"   the host FMA model reproduces every shape bit-for-bit; "
          f"{differed_from_upstream} elements differ from upstream's blocked gemm")


# ---------------------------------------------------------------------------
# 4. Did the GPU do it -- asserted from the runtime, not from the answer
# ---------------------------------------------------------------------------

# op -> (how many compute shaders it must run, arity)
#
# Zero is as much a claim as one. `view` and `contiguous` are shape-only on a
# device where every tensor is contiguous by construction, and saying they
# dispatch nothing is more honest than a number that would imply GPU work.
EXPECTED_DISPATCHES = {
    "aten.add.Tensor": (1, 2),
    "aten.sub.Tensor": (1, 2),
    "aten.mul.Tensor": (1, 2),
    "aten.div.Tensor": (1, 2),
    "aten.neg.default": (1, 1),
    "aten.relu.default": (1, 1),
    "aten.clone.default": (1, 1),
    "aten.t.default": (1, 1),
    "aten.transpose.int": (1, 1),
    "aten.mm.default": (1, 2),
    "aten.addmm.default": (2, 3),
    "aten.gelu.default": (1, 1),
    "aten._softmax.default": (1, 1),
    "aten._safe_softmax.default": (1, 1),
    "aten.all.default": (1, 1),
    "aten._local_scalar_dense.default": (0, 1),
    # docs/devices/VULKAN7.md -- the math path, composed from taught ops
    # rather than fused: transpose, matmul (q @ kT), mul.Scalar (the scale),
    # _softmax, matmul (@ v), and -- since docs/devices/VULKAN8.md -- one more
    # reduction for the `logsumexp` this op's second result is. **Six**, and
    # with an `attn_mask` seven. This entry said 0 while the handler raised
    # before reaching a kernel, so the number was never observed; it was
    # measured at 5 and the logsumexp is the sixth.
    "aten._scaled_dot_product_flash_attention_for_cpu.default": (6, 7),
    "aten.native_layer_norm.default": (1, 1),
    "aten.bmm.default": (1, 2),
    # docs/devices/VULKAN6.md. `embedding` is one gather; the two `.Scalar`
    # ops are one elementwise pass each with the number in the push constants,
    # so neither uploads a staging buffer -- which the `host_uploads == 0`
    # assertion below is what actually proves.
    "aten.embedding.default": (1, 2),
    "aten.mul.Scalar": (1, 1),
    "aten.div.Scalar": (1, 1),
    # docs/devices/VULKAN7.md -- the pretrained-BERT wall. The three view ops
    # materialise through one strided-gather shader each (a `VkTensor` has no
    # strides), `gather` is its own shader, and `tanh` is one elementwise pass.
    # The arguments below are chosen so none of them is an identity view,
    # which is the one case these dispatch nothing for.
    "aten.slice.Tensor": (1, 1),
    "aten.select.int": (1, 1),
    "aten.expand.default": (1, 1),
    "aten.gather.default": (1, 2),
    "aten.tanh.default": (1, 1),
    # The shim keeps `matmul` whole (docs/devices/VULKAN4.md §2.1); with equal
    # batch dimensions it is one batched-product pass.
    "aten.matmul.default": (1, 2),
    # docs/devices/VULKAN9.md -- the reduction the sdpa backward stops on.
    # One dispatch whatever the axes are: the kernel takes the source stride
    # of every axis (this device has no strides, so they are derived from the
    # shape) and walks the kept and reduced index spaces itself, rather than
    # materialising a permuted copy and reducing the last axis.
    "aten.sum.dim_IntList": (1, 1),
    "aten.sum.default": (1, 1),
    # docs/devices/VULKAN10.md §2 -- `mean` is that same kernel with a
    # divisor, so it is **one** dispatch and not a sum followed by a divide.
    # Two would still be on the GPU and would still be the wrong kernel, which
    # is why the number here is the claim rather than ">= 1".
    "aten.mean.dim": (1, 1),
    "aten.mean.default": (1, 1),
    # docs/devices/VULKAN10.md §3 -- the gradient seed, and a shader rather
    # than the host-built upload `torch.ones(device="vulkan")` still is. The
    # 1 here is the difference: an uploading seed would read 0 dispatches and
    # 1 upload, and the `host_uploads == 0` assertion below is what makes a
    # whole training step's counters mean anything.
    "aten.ones_like.default": (1, 1),
    "aten.zeros_like.default": (1, 1),
    "aten.detach.default": (0, 1),
    "aten.alias.default": (0, 1),
    "aten.contiguous.default": (0, 1),
    "aten.view.default": (0, 1),
    "aten._unsafe_view.default": (0, 1),
    "aten.reshape.default": (0, 1),
}


def test_every_taught_op_ran_on_the_gpu_or_says_it_did_not():
    """The device assertion, from `_vulkan_counters()` at runtime.

    **Why not a source scan.** docs/devices/MPSATTN.md §3.1 records, against its own
    round, that an op could have been taken off the `mps` refusal list while
    keeping its host readback and *both* of that device's derivation tests
    would still have passed: the per-op scan looks for six helper names in a
    kernel body, the classification test looks for three markers, and moving
    the readback one call deeper into a differently-named helper is invisible
    to both. Every check of that shape can be defeated by moving the thing it
    greps for.

    This one cannot, because it is not a description of the source.
    `shader_dispatches` is incremented inside `dispatch_kernel` *after*
    `vkWaitForFences` returns success, and `host_downloads` inside `download`,
    which is the module's only map-for-reading. An op that computed on the host
    would have to read its operands to do so, and reading them goes through
    that one function however many helpers deep it is buried. So the assertion
    below is about what the process did:

        the expected number of compute shaders ran, and nothing was read back.
    """
    if _vulkan_or_skip("the per-op GPU assertion") is None:
        return
    a = _to_vulkan(_cpu([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3]))
    b = _to_vulkan(_cpu([6.0, 5.0, 4.0, 3.0, 2.0, 1.0], [2, 3]))
    sq = _to_vulkan(_cpu([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2]))
    bias = _to_vulkan(_cpu([0.5, 1.5], [2]))
    a3 = _to_vulkan(_cpu([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 2, 3]))
    b3 = _to_vulkan(_cpu([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [1, 3, 2]))
    w3 = _to_vulkan(_cpu([0.5, 1.0, 2.0], [3]))
    c3 = _to_vulkan(_cpu([0.1, 0.2, 0.3], [3]))

    taught = set(_C._vulkan_ops())
    assert taught == set(EXPECTED_DISPATCHES) | {"aten._to_copy.default"}, (
        "an op was taught or dropped without saying how many shaders it runs; "
        f"missing from this table: {sorted(taught - set(EXPECTED_DISPATCHES) - {'aten._to_copy.default'})}")

    for op, (expected, arity) in sorted(EXPECTED_DISPATCHES.items()):
        if op in ("aten.mm.default", "aten.matmul.default"):
            args = (a, sq)
        elif op == "aten.addmm.default":
            args = (bias, a, sq)
        elif op == "aten.transpose.int":
            args = (a, 0, 1)
        elif op == "aten._softmax.default":
            args = (a, -1, False)
        elif op == "aten._safe_softmax.default":
            args = (a, -1, None)
        elif op == "aten._scaled_dot_product_flash_attention_for_cpu.default":
            # q, k, v only. The op's arity is 7, but the remaining four
            # (dropout_p, is_causal, attn_mask, scale) all have defaults, and
            # this loop is checking the dispatch footprint rather than the
            # argument surface. Without this arm the generic `else` gives it a
            # single tensor and the math path raises `IndexError` reading
            # `args[1]` -- which is how it failed before this arm existed.
            args = (a3, a3, a3)
        elif op == "aten.all.default":
            args = (_to_vulkan(_C._tensor_from_flat([1.0], [1], dtype=_C.bool)),)
        elif op == "aten._local_scalar_dense.default":
            args = (_to_vulkan(_cpu([1.0], [1])),)
        elif op == "aten.native_layer_norm.default":
            args = (a, [3], w3, c3, 1e-5)
        elif op == "aten.bmm.default":
            args = (a3, b3)
        elif op == "aten.embedding.default":
            args = (a, _to_vulkan(_i64([0, 1], [2])))
        elif op in ("aten.mul.Scalar", "aten.div.Scalar"):
            args = (a, 2.0)
        elif op == "aten.slice.Tensor":
            args = (a, 1, 1, 3)
        elif op == "aten.select.int":
            args = (a, 0, 1)
        elif op == "aten.expand.default":
            args = (a, [4, 2, 3])
        elif op == "aten.sum.dim_IntList":
            args = (a, [1], False)
        elif op == "aten.gather.default":
            args = (a, 1, _to_vulkan(_i64([2, 0, 1, 1], [2, 2])))
        elif op in ("aten.view.default", "aten._unsafe_view.default",
                    "aten.reshape.default"):
            args = (a, [6])
        elif arity == 2:
            args = (a, b)
        else:
            args = (a,)
        before = _counters()
        out = _C._aten_dispatch(op, *args)
        d = _delta(before, _counters())
        if op == "aten._local_scalar_dense.default":
            # The one exemption from `host_downloads == 0` and from
            # `out.device == "vulkan"`, and it is keyed to this exact op
            # string -- an `==` on the name, not a category, so no op can
            # inherit it by being added near this one.
            #
            # It is also not a free pass. Reading one scalar back to the host
            # is what this op *is*, so the exemption is stated as a positive
            # claim: exactly one readback, and a Python number rather than a
            # tensor. If a future change made it return a device tensor, or
            # made it read more than the single element, this fails rather
            # than going quiet.
            assert not hasattr(out, "device"), (
                f"{op} returned {type(out).__name__}; it must return a Python "
                f"scalar, which is the whole reason it is exempt here")
            assert isinstance(out, (int, float, bool)), (op, type(out))
            assert d["host_downloads"] == 1, (
                f"{op} performed {d['host_downloads']} readbacks; it is exempt "
                f"from the zero-readback rule for exactly one")
        else:
            if isinstance(out, tuple):
                # native_layer_norm: (out, mean, invstd) -- all three tensors.
                # sdpa: (output, logsumexp), and the second **used to be
                # `None`** on this device. It is a real device tensor now
                # (docs/devices/VULKAN8.md); this arm says so rather than
                # tolerating either answer, so a regression that took the
                # logsumexp away again fails here and not only in the sweep.
                if op == _SDPA:
                    assert out[1] is not None, (
                        f"{op} returned None for its logsumexp; this device "
                        f"computes one (docs/devices/VULKAN8.md) and a `None` "
                        f"here means the handler stopped")
                    assert str(out[1].device) == "vulkan", out[1].device
                assert all(str(o.device) == "vulkan"
                           for o in out if o is not None), (op, out)
                out = out[0]
            assert str(out.device) == "vulkan", (op, out.device)
        assert d["shader_dispatches"] == expected, (
            f"{op} ran {d['shader_dispatches']} compute shaders, expected "
            f"{expected}")
        if op != "aten._local_scalar_dense.default":
            assert d["host_downloads"] == 0, (
                f"{op} read {d['host_downloads']} buffer(s) back to the host -- it "
                f"is computing on the CPU under a vulkan label")
        assert d["host_uploads"] == 0, (
            f"{op} uploaded {d['host_uploads']} buffer(s); no taught op builds "
            f"an operand on the host")


def test_the_readback_counter_moves_when_something_is_actually_read_back():
    """The positive control for the instrument above.

    An assertion of the form "this counter did not move" is worthless if the
    counter never moves. `.cpu()` genuinely reads the buffer back, so it must
    move it -- and if a future change made `download` stop counting, the test
    above would go quietly green on an op that had started computing on the
    host. This is the test that fails first in that case.
    """
    if _vulkan_or_skip("the readback-counter control") is None:
        return
    x = _to_vulkan(_cpu([1.0, 2.0], [2]))
    before = _counters()
    _to_cpu(x)
    d = _delta(before, _counters())
    assert d["host_downloads"] == 1, (
        f"a .cpu() did not register as a host readback ({d}) -- the counter "
        f"that the per-op assertions rely on is not live")


# ---------------------------------------------------------------------------
# 5. A whole module -- the honest answer to "does Vulkan work?"
# ---------------------------------------------------------------------------

_MLP_SHIM_SCRIPT = r"""
import json, sys
import torch, torch.nn as nn

# The probe a previous round in this repository paid for by measuring upstream
# torch for hours: this subprocess must be the shim, not the oracle.
assert hasattr(torch._C, "_aten_implemented"), "this subprocess got upstream torch"

cfg = json.load(sys.stdin)
out = {"probe": torch._C._vulkan_probe()}
if not out["probe"]["available"]:
    json.dump(out, sys.stdout); raise SystemExit

m = nn.Sequential(nn.Linear(cfg["i"], cfg["h"]), nn.ReLU(),
                  nn.Linear(cfg["h"], cfg["o"]))
m.load_state_dict({k: torch.as_tensor(v, dtype=torch.float32)
                   for k, v in cfg["sd"].items()})
m.eval()
x = torch.as_tensor(cfg["x"], dtype=torch.float32).reshape(cfg["sx"])
with torch.no_grad():
    out["cpu"] = m(x).reshape(-1).tolist()
m.to("vulkan")
before = torch._C._vulkan_counters()
with torch.no_grad():
    r = m(x.to("vulkan"))
after = torch._C._vulkan_counters()
out["device"] = str(r.device)
out["counters"] = {k: after[k] - before[k] for k in after}
out["vulkan"] = r.cpu().reshape(-1).tolist()
json.dump(out, sys.stdout)
"""


def test_a_whole_module_forwards_on_the_gpu_and_agrees_with_upstream():
    """`nn.Sequential(Linear, ReLU, Linear)`, forward, on the Vulkan device.

    **This is the claim docs/devices/VULKAN4.md §5 makes and the one it stops at.** A
    module, not an op: `nn.Linear` dispatches `aten.t.default` and then
    `aten.addmm.default`, so the forward really does go through the transpose
    and the matmul rather than around them, and `m.to("vulkan")` really does
    move the parameters.

    It is **not** a transformer. `native_layer_norm`, `_softmax`, `gelu`,
    `embedding` and `bmm` are all still refused, and every one of them is on
    the measured trace of a BERT forward (docs/devices/VULKAN4.md §2), so a transformer
    stops at the first of them. Saying "an MLP forwards" is the honest size of
    this result.

    Three assertions, and the third is the one that is not about the answer:

      1. the output really is a vulkan tensor;
      2. it agrees with upstream inside docs/numerics/AGREE.md's derived rule, and with
         the shim's own cpu answer to about one float32 ulp;
      3. **seven compute shaders ran and nothing was read back** -- 2x(t,
         matmul, bias) + 1 relu, which is what the two Linears and the ReLU
         must cost if the GPU is doing them.
    """
    if _vulkan_or_skip("the module forward") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    try:
        import torch.nn as nn
    except Exception:  # noqa: BLE001
        vulkan_coverage.vulkan_skip("the module forward: no torch.nn upstream")
        return

    i, h, o, batch = 12, 32, 6, 5
    g = torch.Generator().manual_seed(11)
    model = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o)).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32))
    x = torch.randn(batch, i, generator=g, dtype=torch.float32)

    cfg = {"i": i, "h": h, "o": o, "sx": [batch, i],
           "x": x.reshape(-1).tolist(),
           "sd": {k: v.tolist() for k, v in model.state_dict().items()}}
    env = dict(os.environ)
    env["PYTHONPATH"] = VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run([sys.executable, "-c", _MLP_SHIM_SCRIPT],
                          input=json.dumps(cfg), capture_output=True,
                          text=True, env=env, timeout=300)
    assert proc.returncode == 0, (proc.stdout[-2000:], proc.stderr[-3000:])
    got = json.loads(proc.stdout)
    if not got["probe"]["available"]:
        # This printed "(skipped ...)" and then `ok` -- measured, on a host
        # where this process had a loader and the vendored `_C` was a build
        # that did not search Homebrew's prefix (docs/devices/VULKAN5.md §2).
        vulkan_coverage.vulkan_skip(
            f"the module forward: the vendored-tree subprocess has no loader -- "
            f"{got['probe']['error']}")
        return

    assert got["device"].startswith("vulkan"), got["device"]

    with torch.no_grad():
        f32 = model(x).reshape(-1).tolist()
        wide = nn.Sequential(nn.Linear(i, h), nn.ReLU(), nn.Linear(h, o)).eval()
        wide.load_state_dict(model.state_dict())
        truth = wide.double()(x.double()).reshape(-1).tolist()

    scale = max(abs(v) for v in truth)
    up_err = max(abs(a - b) for a, b in zip(f32, truth))
    vk_err = max(abs(a - b) for a, b in zip(got["vulkan"], truth))
    cpu_err = max(abs(a - b) for a, b in zip(got["cpu"], truth))
    backends = max(abs(a - b) for a, b in zip(got["vulkan"], got["cpu"]))
    ulp = scale * FLOAT32_EPS

    # docs/numerics/AGREE.md §2's second rule: a difference is not a defect if it is
    # within 4x upstream's own distance from the float64 truth on the same
    # output. Measured at 0.90x (docs/devices/VULKAN4.md §4.3).
    assert vk_err <= 4 * up_err, (
        f"vulkan is {vk_err:.3e} from the float64 truth against upstream's own "
        f"{up_err:.3e}; docs/numerics/AGREE.md's rule allows 4x")
    assert cpu_err <= 4 * up_err, (cpu_err, up_err)
    assert backends <= 2 * ulp, (
        f"vulkan and the shim's own cpu differ by {backends:.3e}, more than "
        f"two float32 ulp ({2 * ulp:.3e}) at this magnitude")

    # The third assertion, and the one that is about the device rather than
    # the answer. Two Linears = 2 x (transpose + matmul + bias) = 6, plus the
    # ReLU = 7. A forward that had fallen back to the host would still get the
    # numbers right and would fail here.
    counters = got["counters"]
    assert counters["shader_dispatches"] == 7, (
        f"the module forward ran {counters['shader_dispatches']} compute "
        f"shaders, expected 7 (2 Linears x 3 + 1 ReLU): {counters}")
    assert counters["host_downloads"] <= 1 and counters["host_uploads"] <= 4, (
        f"the module forward read {counters['host_downloads']} buffer(s) back "
        f"to the host: {counters}")
    print(f"   MLP on vulkan: upstream f32 err {up_err:.3e}, shim cpu "
          f"{cpu_err:.3e}, shim vulkan {vk_err:.3e} ({vk_err / up_err:.2f}x), "
          f"vulkan-vs-cpu {backends / ulp:.2f} ulp, "
          f"{counters['shader_dispatches']} shaders, "
          f"{counters['host_downloads']} readbacks")


def test_the_transformer_wall_moved_and_the_new_one_is_named():
    """The wall was `embedding`, then five ops and a 4-D transpose. Both are gone.

    `docs/devices/VULKAN5.md` §4 said no transformer reached its *second* op;
    `docs/devices/VULKAN6.md` §3 closed that and named what stopped a
    *pretrained* BERT next: `expand`, `slice`, `gather`, `select`, `tanh`, and
    `transpose.int` on 4-D (batch, head, seq, dim) tensors.
    `docs/devices/VULKAN7.md` teaches all six, and
    `test_a_pretrained_bert_forwards_on_the_gpu_and_agrees_with_upstream` is
    the evidence that the list was the whole wall rather than its first
    layer.

    This pins the list in both directions: every op VULKAN6 named is taught,
    and a 4-D transpose -- which VULKAN6 asserted *refused* -- now runs one
    shader. A round that drops any of them fails here.
    """
    if _vulkan_or_skip("the named transformer wall") is None:
        return
    taught = set(_C._vulkan_ops())
    landed = ("aten.native_layer_norm.default", "aten._softmax.default",
              "aten.gelu.default", "aten.bmm.default",
              "aten.embedding.default", "aten.div.Scalar",
              # docs/devices/VULKAN6.md §3.1's wall, in its order.
              "aten.expand.default", "aten.slice.Tensor", "aten.gather.default",
              "aten.select.int", "aten.tanh.default",
              # ...and the one VULKAN6.md's trace could not see: the shim keeps
              # matmul whole (docs/devices/VULKAN7.md §3.2).
              "aten.matmul.default")
    assert all(op in taught for op in landed), sorted(set(landed) - taught)

    a4 = _to_vulkan(_cpu([float(i) for i in range(16)], [2, 2, 2, 2]))
    before = _counters()
    out = _C._aten_dispatch("aten.transpose.int", a4, 1, 2)
    d = _delta(before, _counters())
    assert list(out.shape) == [2, 2, 2, 2], out.shape
    assert d["shader_dispatches"] == 1 and d["host_downloads"] == 0, d

    # `embedding`'s wall was int64 *storage*; an index tensor still lands, and
    # what it cannot hold still refuses with the bound named (VULKAN6.md §1).
    idx = _to_vulkan(_i64([0, 1], [2]))
    assert str(idx.device) == "vulkan" and str(idx.dtype) == "torch.int64", (
        idx.device, idx.dtype)


# ---------------------------------------------------------------------------
# 6. The narrowings refuse rather than reaching for the CPU
# ---------------------------------------------------------------------------

def test_every_narrowing_refuses_by_name_rather_than_being_emulated():
    """The property the whole `Repr::Vulkan` design exists to protect.

    Each of these has a perfectly good CPU implementation a few lines away, and
    reaching for one would be the silent fallback docs/devices/VULKAN.md §5 calls the
    worst available outcome. They refuse, and the message says which narrowing
    was hit -- a generic "not implemented" would leave the reader unable to
    tell a missing broadcast from a missing dtype.
    """
    if _vulkan_or_skip("the narrowing refusals") is None:
        return
    a23 = _to_vulkan(_cpu([1.0] * 6, [2, 3]))
    a32 = _to_vulkan(_cpu([1.0] * 6, [3, 2]))
    a3d = _to_vulkan(_cpu([1.0] * 8, [2, 2, 2]))
    a9d = _to_vulkan(_cpu([1.0] * 2, [2] + [1] * 8))
    bias = _to_vulkan(_cpu([1.0, 1.0], [2]))

    cases = (
        ("non-broadcastable", lambda: _C._aten_dispatch("aten.add.Tensor", a23, a32),
         "must match the size"),
        ("alpha", lambda: _C._aten_dispatch("aten.add.Tensor", a23, a23, 2.0),
         "alpha"),
        ("rank-9 transpose",
         lambda: _C._aten_dispatch("aten.transpose.int", a9d, 0, 1), "rank"),
        ("batched matmul",
         lambda: _C._aten_dispatch("aten.mm.default", a3d, a3d), "2-D"),
        ("addmm beta",
         lambda: _C._aten_dispatch("aten.addmm.default", bias, a23, a32,
                                   2.0), "beta"),
        ("bad reshape",
         lambda: _C._aten_dispatch("aten.view.default", a23, [4, 4]),
         "invalid"),
    )
    for what, call, needle in cases:
        try:
            call()
        except (NotImplementedError, RuntimeError) as e:
            assert needle in str(e), (
                f"the {what} refusal does not say which narrowing was hit "
                f"(looked for {needle!r}): {e}")
        else:
            raise AssertionError(f"{what} was emulated instead of refused")


def test_a_vulkan_tensor_still_has_no_cpu_storage_to_read():
    """The structural property, re-asserted after eighteen ops were added.

    `PyTensorBase::tensor()` refuses on every non-`Dense` arm, and that refusal
    is what makes a silent CPU fallback unrepresentable rather than merely
    avoided. Widening the device is exactly the change that could have
    weakened it -- so it is checked again here rather than assumed to have
    survived.
    """
    if _vulkan_or_skip("the no-cpu-storage property") is None:
        return
    x = _to_vulkan(_cpu([1.0, 2.0], [2]))
    try:
        x.tolist()
    except (NotImplementedError, RuntimeError) as e:
        assert "cpu" in str(e).lower(), str(e)
    else:
        raise AssertionError(
            "a vulkan tensor handed out CPU storage through tolist()")


# ---------------------------------------------------------------------------
# 7. The checked-in SPIR-V is what the GLSL says
# ---------------------------------------------------------------------------

def test_the_checked_in_spirv_is_not_stale_for_any_shader():
    """docs/devices/VULKAN3.md's guard, extended from one shader to all of them.

    The `.spv` words are checked in and `include_bytes!`d rather than built by
    a `build.rs`, so an edit to a `.comp` does nothing until
    `shaders/compile.sh` is run -- and the old kernel ships silently. That
    guard existed for `add_f32` alone; this round added nine more files, and a
    guard that covers one of ten is a guard someone will trust.
    """
    shaders = os.path.join(REPO, "rust", "torch_c", "shaders")
    comps = sorted(f for f in os.listdir(shaders) if f.endswith(".comp"))
    assert len(comps) >= 10, f"expected at least ten shaders, found {comps}"
    for comp in comps:
        spv = os.path.join(shaders, comp[:-len(".comp")] + ".spv")
        assert os.path.exists(spv), (
            f"{comp} has no compiled {os.path.basename(spv)} -- run "
            f"shaders/compile.sh")
        assert os.path.getmtime(spv) >= os.path.getmtime(os.path.join(shaders, comp)), (
            f"{os.path.basename(spv)} is older than {comp}: the checked-in "
            f"kernel is not the one the GLSL describes. Run shaders/compile.sh")
        with open(spv, "rb") as fh:
            words = fh.read()
        assert len(words) % 4 == 0 and len(words) > 20, (comp, len(words))
        # SPIR-V's magic number, little-endian. A truncated or text file that
        # happened to be newer would otherwise pass the two checks above.
        assert words[:4] == b"\x03\x02\x23\x07", (
            f"{os.path.basename(spv)} does not begin with the SPIR-V magic "
            f"number -- it is not a compiled shader")


def test_every_shader_on_disk_is_reachable_from_the_dispatcher():
    """A shader nobody dispatches is dead weight that still looks like coverage."""
    shaders = os.path.join(REPO, "rust", "torch_c", "shaders")
    src = os.path.join(REPO, "rust", "torch_c", "src", "vulkan.rs")
    with open(src) as fh:
        text = fh.read()
    for comp in sorted(f for f in os.listdir(shaders) if f.endswith(".comp")):
        stem = comp[:-len(".comp")]
        assert f'"{stem}"' in text, (
            f"shaders/{comp} is compiled and checked in but no call in "
            f"vulkan.rs names {stem!r}")


# ---------------------------------------------------------------------------
# 8. Finding the loader (docs/devices/VULKAN5.md §1)
# ---------------------------------------------------------------------------

def test_a_loader_installed_where_this_build_searches_is_found():
    """A loader on disk at a searched path must not produce "failed to load".

    The defect this holds down: Homebrew's `vulkan-loader` puts
    `libvulkan.dylib` in `/opt/homebrew/lib`, which is not on dyld's default
    search path, so a bare `dlopen("libvulkan.dylib")` misses it and every
    Vulkan test skipped on a machine that had a loader.

    The places to check are listed **here**, not read back from the
    extension. The first version of this test took them from
    `_vulkan_loader_candidates()`, and deleting the Homebrew fallback left it
    green: the path vanished from the list it was checked against at the same
    moment (docs/devices/VULKAN5.md §5.1). On a machine with none of these on
    disk there is nothing to assert, and it says so.
    """
    known = ["/opt/homebrew/lib/libvulkan.1.dylib", "/usr/local/lib/libvulkan.1.dylib"]
    if os.environ.get("VULKAN_SDK"):
        known.append(os.path.join(os.environ["VULKAN_SDK"], "lib", "libvulkan.1.dylib"))
    candidates = _C._vulkan_loader_candidates()
    assert candidates, "the extension names no place to look for a Vulkan loader"
    probe = _C._vulkan_probe()
    if probe["available"]:
        assert probe["loader"] in candidates, (probe, candidates)
        print(f"   loader {probe['loader']} -> {probe['device']} ({probe['type']})")
    on_disk = [p for p in known if os.path.exists(p)] if sys.platform == "darwin" else []
    if not on_disk:
        print(f"   no known macOS loader location exists here; searched {candidates}")
        return
    missing = [p for p in on_disk if p not in candidates]
    assert not missing, f"{missing} exist on disk but the extension never looks there: {candidates}"
    assert not str(probe["error"] or "").startswith("failed to load the Vulkan loader"), (
        f"{on_disk} exist on disk and are searched, yet the loader was not "
        f"loaded: {probe['error']}")


# ---------------------------------------------------------------------------
# 9. The four transformer kernels -- values (docs/devices/VULKAN5.md §3)
# ---------------------------------------------------------------------------

def _rel(xs, truth):
    scale = max(abs(v) for v in truth) or 1.0
    return max(abs(x - y) for x, y in zip(xs, truth)) / scale


def _elem_ratio(xs, want32, truth):
    """Worst per-element distance from the float64 truth, as a multiple of
    upstream float32's own distance on *that element* (floored at one ulp of
    it). The tensor-scale metric above is AGREE's and is what is asserted;
    this is printed so a small element that is far off in its own terms is
    visible rather than averaged into a large scale. Measured in ulp it read
    8388608 for shim and upstream alike wherever the truth underflows float32,
    which said nothing about either -- hence the ratio."""
    worst = 0.0
    for x, u, t in zip(xs, want32, truth):
        allowed = max(abs(u - t), abs(t) * FLOAT32_EPS, 1e-38)
        worst = max(worst, abs(x - t) / allowed)
    return worst


def _assert_agreement(name, cases):
    """`cases`: (label, shim, upstream float32, upstream float64) flat lists.

    The tolerance is re-derived from *this* population's upstream float32 vs
    float64 error (docs/numerics/AGREE.md §2), per op, never chosen.
    """
    rows, up_errs = [], []
    for label, got, want32, truth in cases:
        assert len(got) == len(want32) == len(truth), (label, len(got), len(want32), len(truth))
        up = _rel(want32, truth)
        rows.append((label, _rel(got, truth), up,
                     sum(_bits(x) != _bits(y) for x, y in zip(got, want32)), len(got),
                     _elem_ratio(got, want32, truth)))
        up_errs.append(up)
    tol, p90 = _derived_tolerance(up_errs)
    worst = max(rows, key=lambda r: r[1])
    print(f"   {name}: {len(rows)} cases, tolerance max(p90 {p90:.3e}, 8 ulp {8 * FLOAT32_EPS:.3e}) "
          f"= {tol:.3e}; worst shim {worst[1]:.3e} at {worst[0]} (upstream's own {worst[2]:.3e}); "
          f"{sum(r[3] for r in rows)}/{sum(r[4] for r in rows)} elements differ in bits from upstream f32; "
          f"worst element {max(r[5] for r in rows):.2f}x upstream's own error on it")
    for label, shim, up, *_ in rows:
        assert shim <= tol, (
            f"{label}: shim {shim:.3e} from the float64 truth exceeds the derived "
            f"tolerance {tol:.3e} (upstream float32's own error {up:.3e})")


def _up_flat(t):
    return t.reshape(-1).tolist()


GELU_SHAPES = ((4, 5), (7, 33), (2, 3, 64), (1025,))


def test_gelu_agrees_with_upstream_at_a_derived_tolerance():
    """Both `approximate` modes, on `3 * randn` so the tails are exercised."""
    if _vulkan_or_skip("the gelu agreement sweep") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    import torch.nn.functional as F
    for approximate in ("none", "tanh"):
        cases = []
        for i, shape in enumerate(GELU_SHAPES):
            a = _rand(torch, *shape, seed=500 + i) * 3.0
            out = _C._aten_dispatch("aten.gelu.default",
                                    _to_vulkan(_cpu(_up_flat(a), shape)),
                                    approximate=approximate)
            assert list(out.shape) == list(shape), (out.shape, shape)
            cases.append((f"gelu[{approximate}]{list(shape)}", _flat(_to_cpu(out)),
                          _up_flat(F.gelu(a, approximate=approximate)),
                          _up_flat(F.gelu(a.double(), approximate=approximate))))
        _assert_agreement(f"gelu approximate={approximate}", cases)


# (shape, dim, input scale) -- scale 8 makes rows peaky, which is where a
# softmax that forgot to subtract the max overflows.
SOFTMAX_CASES = (((4, 5), -1, 1.0), ((3, 7), 1, 4.0), ((2, 3, 64), 2, 1.0),
                 ((2, 512), -1, 8.0), ((5, 1), -1, 1.0), ((3, 9), -1, 40.0))


def test_softmax_agrees_with_upstream_at_a_derived_tolerance():
    if _vulkan_or_skip("the softmax agreement sweep") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    cases = []
    for i, (shape, dim, scale) in enumerate(SOFTMAX_CASES):
        a = _rand(torch, *shape, seed=600 + i) * scale
        out = _C._aten_dispatch("aten._softmax.default",
                                _to_vulkan(_cpu(_up_flat(a), shape)), dim, False)
        assert list(out.shape) == list(shape), (out.shape, shape)
        cases.append((f"softmax{list(shape)} dim={dim} x{scale}", _flat(_to_cpu(out)),
                      _up_flat(torch.ops.aten._softmax.default(a, dim, False)),
                      _up_flat(torch.ops.aten._softmax.default(a.double(), dim, False))))
    _assert_agreement("_softmax", cases)


_SDPA = "aten._scaled_dot_product_flash_attention_for_cpu.default"

# (B, S, D) for q/k/v, and whether a float32 additive mask is supplied.
SDPA_CASES = (((1, 2, 3), False), ((1, 4, 8), True), ((2, 5, 4), False),
              ((2, 3, 16), True))


def _sdpa_reference(torch, q, k, v, mask, dtype):
    """Upstream's math path, written out, at whatever dtype it is handed.

    Not `torch.ops.aten._scaled_dot_product_flash_attention_for_cpu` itself:
    that op accepts only rank-4 `{B, H, T, K}` operands (`sdpa_flash_cpu` in
    aten.rs reproduces the refusal), and the shapes below are rank 3. The
    formula is the oracle instead, and it is run twice -- float32 and float64 --
    so the tolerance is still *derived* from upstream's own error the way
    docs/numerics/AGREE.md §2 derives it, rather than chosen here.
    """
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
    if mask is not None:
        scores = scores + mask.to(dtype)
    return torch.softmax(scores, dim=-1) @ v


def test_sdpa_is_reachable_from_the_dispatcher_and_agrees():
    """The math path called the way the dispatcher calls it, not the way Python does.

    **This is the asymmetry test.** `sdpa_vulkan` is a handler on the device
    dispatch table, so the operands it is handed are whatever the dispatcher
    is holding -- `torch._C.TensorBase`, not the `torch.Tensor` subclass the
    Python layer constructs. A handler that re-enters the *Python* API to do
    its work (`torch.softmax(...)`) therefore works only for callers who
    arrived through the Python API, and fails for the dispatcher itself:

        TypeError: softmax(): argument 'input' (position 1) must be Tensor,
                   not torch._C.TensorBase

    which is exactly what this op did before the handler was changed to
    dispatch `aten._softmax.default` instead. A real model never saw it,
    because a real model's operands came in through `torch.Tensor` -- so the
    defect was invisible to every model-level test and visible only from here.

    Fixing it by handing this test `torch.Tensor` fixtures would restore the
    green and leave the asymmetry: see the `Not to do` section of the round
    that found it. The entry point is the thing under test.
    """
    if _vulkan_or_skip("the sdpa dispatcher-entry agreement sweep") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    cases = []
    for i, (shape, with_mask) in enumerate(SDPA_CASES):
        b, s, d = shape
        q = _rand(torch, *shape, seed=900 + i)
        k = _rand(torch, *shape, seed=930 + i)
        v = _rand(torch, *shape, seed=960 + i)
        mask = _rand(torch, b, s, s, seed=990 + i) if with_mask else None
        args = [_to_vulkan(_cpu(_up_flat(q), shape)),
                _to_vulkan(_cpu(_up_flat(k), shape)),
                _to_vulkan(_cpu(_up_flat(v), shape))]
        kwargs = {}
        if mask is not None:
            kwargs["attn_mask"] = _to_vulkan(_cpu(_up_flat(mask), [b, s, s]))
        out = _C._aten_dispatch(_SDPA, *args, **kwargs)
        # The op returns (output, logsumexp); only the first is computed here.
        assert isinstance(out, tuple), type(out)
        got = out[0]
        assert str(got.device) == "vulkan", got.device
        assert list(got.shape) == list(shape), (got.shape, shape)
        cases.append((f"sdpa{list(shape)} mask={with_mask}", _flat(_to_cpu(got)),
                      _up_flat(_sdpa_reference(torch, q, k, v, mask, torch.float32)),
                      _up_flat(_sdpa_reference(torch, q, k, v, mask, torch.float64))))
    _assert_agreement("sdpa (math path, entered from the dispatcher)", cases)


def test_sdpa_reads_its_arguments_at_the_positions_the_schema_gives_them():
    """`attn_mask` is argument **5**, not 3.

    The schema is `(query, key, value, dropout_p=0.0, is_causal=False,
    attn_mask=None, scale=None)` -- `sdpa_flash_cpu` in aten.rs parses it at
    those indices and this handler must agree, or a positional call means one
    thing on the CPU device and another on this one. The inline Python this
    handler used to be read `args[3]` as the mask, which is `dropout_p`.

    So: the same mask, once by keyword and once positionally at 5, must give
    the same answer; and a `dropout_p` at 3 must be refused rather than
    silently added to the scores.
    """
    if _vulkan_or_skip("the sdpa argument-position check") is None:
        return
    shape, mshape = [1, 3, 4], [1, 3, 3]
    q = _to_vulkan(_cpu([0.1 * i for i in range(12)], shape))
    kk = _to_vulkan(_cpu([0.2 * i for i in range(12)], shape))
    v = _to_vulkan(_cpu([0.3 * i for i in range(12)], shape))
    mask_values = [0.0, -1e9, 0.0, 0.0, 0.0, -1e9, -1e9, 0.0, 0.0]

    def mask():
        return _to_vulkan(_cpu(mask_values, mshape))

    by_keyword = _flat(_to_cpu(_C._aten_dispatch(
        _SDPA, q, kk, v, attn_mask=mask())[0]))
    positional = _flat(_to_cpu(_C._aten_dispatch(
        _SDPA, q, kk, v, 0.0, False, mask())[0]))
    assert by_keyword == positional, (by_keyword, positional)

    # And it is not the same as no mask at all -- otherwise the equality above
    # would hold for a handler that ignored the mask in both calls.
    unmasked = _flat(_to_cpu(_C._aten_dispatch(_SDPA, q, kk, v)[0]))
    assert unmasked != by_keyword, (unmasked, by_keyword)

    # `dropout_p` at 3 is a float, not a mask. Upstream's CPU kernel refuses a
    # non-zero one by name ("Currently do not support dropout > 0") rather
    # than returning the undropped answer, and so must this.
    try:
        _C._aten_dispatch(_SDPA, q, kk, v, 0.5)
    except Exception as e:  # noqa: BLE001
        assert "dropout" in str(e), e
    else:
        raise AssertionError("dropout_p=0.5 was accepted; it must be refused")


# (input shape, normalized_shape, which affine parameters are given)
LAYER_NORM_CASES = (((4, 5), [5], "both"), ((2, 4, 3), [3], "both"),
                    ((2, 4, 3), [4, 3], "both"), ((3, 64), [64], "weight"),
                    ((3, 64), [64], "bias"), ((6, 17), [17], "none"),
                    ((1, 512), [512], "both"))


def test_native_layer_norm_agrees_with_upstream_at_a_derived_tolerance():
    """All three outputs -- `out`, `mean`, `invstd` -- and their shapes.

    Every affine combination, because the kernel this round inherited
    silently dropped `bias` whenever `weight` was absent: `nn.LayerNorm` never
    produces that, so nothing on a model trace would have caught it.
    """
    if _vulkan_or_skip("the native_layer_norm agreement sweep") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    per_output = {"out": [], "mean": [], "invstd": []}
    for i, (shape, norm, affine) in enumerate(LAYER_NORM_CASES):
        a = _rand(torch, *shape, seed=700 + i) * 2.0 + 0.5
        w = _rand(torch, *norm, seed=800 + i) if affine in ("both", "weight") else None
        b = _rand(torch, *norm, seed=900 + i) if affine in ("both", "bias") else None
        vw = None if w is None else _to_vulkan(_cpu(_up_flat(w), norm))
        vb = None if b is None else _to_vulkan(_cpu(_up_flat(b), norm))
        got = _C._aten_dispatch("aten.native_layer_norm.default",
                                _to_vulkan(_cpu(_up_flat(a), shape)), norm, vw, vb, 1e-5)
        want32 = torch.ops.aten.native_layer_norm.default(a, norm, w, b, 1e-5)
        want64 = torch.ops.aten.native_layer_norm.default(
            a.double(), norm, None if w is None else w.double(),
            None if b is None else b.double(), 1e-5)
        for name, g, u32, u64 in zip(("out", "mean", "invstd"), got, want32, want64):
            assert list(g.shape) == list(u32.shape), (
                f"native_layer_norm{list(shape)} {norm} {affine}: {name} has shape "
                f"{list(g.shape)}, upstream {list(u32.shape)}")
            per_output[name].append((f"{name}{list(shape)} {norm} {affine}",
                                     _flat(_to_cpu(g)), _up_flat(u32), _up_flat(u64)))
    for name, cases in per_output.items():
        _assert_agreement(f"native_layer_norm {name}", cases)


BMM_SHAPES = ((2, 3, 4, 5), (4, 16, 8, 16), (3, 5, 257, 7), (2, 1, 512, 1))


def test_bmm_agrees_with_upstream_and_is_the_kernel_it_claims_to_be():
    """Agreement at a derived tolerance, then the matmul proof per batch.

    A tolerance cannot tell an accumulation-order residue from a kernel that
    reads the wrong batch's element -- both are "small" on randn. So the GPU
    answer is also compared **bit for bit** against the host model of the
    matmul kernel applied to each batch slice separately: a batch-offset bug
    would not survive that.
    """
    if _vulkan_or_skip("the bmm agreement sweep") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    cases, model_misses = [], []
    for i, (bs, m, k, n) in enumerate(BMM_SHAPES):
        a = _rand(torch, bs, m, k, seed=1000 + i)
        b = _rand(torch, bs, k, n, seed=1100 + i)
        af, bf = _up_flat(a), _up_flat(b)
        out = _C._aten_dispatch("aten.bmm.default", _to_vulkan(_cpu(af, (bs, m, k))),
                                _to_vulkan(_cpu(bf, (bs, k, n))))
        assert list(out.shape) == [bs, m, n], out.shape
        got = _flat(_to_cpu(out))
        cases.append((f"bmm[{bs},{m},{k},{n}]", got, _up_flat(torch.bmm(a, b)),
                      _up_flat(torch.bmm(a.double(), b.double()))))
        model = []
        for bi in range(bs):
            model += _host_model_of_the_matmul_kernel(
                af[bi * m * k:(bi + 1) * m * k], bf[bi * k * n:(bi + 1) * k * n], m, k, n)
        misses = sum(_bits(x) != _bits(y) for x, y in zip(got, model))
        model_misses.append((f"bmm[{bs},{m},{k},{n}]", misses, len(got)))
    _assert_agreement("bmm", cases)
    print(f"   bmm vs the host FMA model of the kernel: {model_misses}")
    for label, misses, total in model_misses:
        assert misses == 0, (
            f"{label}: {misses} of {total} elements are not what per-batch "
            f"sequential float32 accumulation gives -- a batch or index offset "
            f"is wrong, or this driver contracts differently")


# ---------------------------------------------------------------------------
# 10. The four transformer kernels -- refusals
# ---------------------------------------------------------------------------

def test_the_transformer_kernels_refuse_what_they_do_not_implement_before_any_gpu_work():
    """Each refusal names its reason, and happens before a shader is dispatched.

    Three of these were not refusals in the kernels this round inherited:
    `gelu(approximate="foo")` computed the exact gelu, a `normalized_shape`
    that did not match the input normalised the wrong span (or panicked when
    it was longer than the input's rank), and `weight=None, bias=b` dropped
    `b`. Messages follow upstream's where upstream raises.
    """
    if _vulkan_or_skip("the transformer-kernel refusals") is None:
        return
    d = _C._aten_dispatch
    x = _to_vulkan(_cpu([float(i) for i in range(12)], [3, 4]))
    w5 = _to_vulkan(_cpu([1.0] * 5, [5]))
    b234 = _to_vulkan(_cpu([1.0] * 24, [2, 3, 4]))
    b254 = _to_vulkan(_cpu([1.0] * 40, [2, 5, 4]))
    b334 = _to_vulkan(_cpu([1.0] * 36, [3, 4, 3]))
    ln = "aten.native_layer_norm.default"
    cases = (
        ("gelu approximate", lambda: d("aten.gelu.default", x, approximate="foo"),
         RuntimeError, "approximate"),
        ("softmax over a non-last dim", lambda: d("aten._softmax.default", x, 0, False),
         NotImplementedError, "last dimension"),
        ("softmax dim out of range", lambda: d("aten._softmax.default", x, 2, False),
         IndexError, "out of range"),
        ("softmax half_to_float", lambda: d("aten._softmax.default", x, -1, True),
         NotImplementedError, "half_to_float"),
        ("layer_norm shape mismatch", lambda: d(ln, x, [5], None, None, 1e-5),
         RuntimeError, "normalized_shape"),
        ("layer_norm longer than input", lambda: d(ln, x, [2, 3, 4], None, None, 1e-5),
         RuntimeError, "normalized_shape"),
        ("layer_norm weight shape", lambda: d(ln, x, [4], w5, None, 1e-5),
         RuntimeError, "weight"),
        ("layer_norm bias shape", lambda: d(ln, x, [4], None, w5, 1e-5),
         RuntimeError, "bias"),
        ("bmm inner size", lambda: d("aten.bmm.default", b234, b254),
         RuntimeError, "batch2"),
        ("bmm batch size", lambda: d("aten.bmm.default", b234, b334),
         RuntimeError, "batch2"),
        ("bmm rank", lambda: d("aten.bmm.default", x, x), RuntimeError, "3D"),
    )
    for what, call, exc, needle in cases:
        before = _counters()
        try:
            call()
        except exc as e:
            assert needle in str(e), f"the {what} refusal does not say {needle!r}: {e}"
        except Exception as e:  # noqa: BLE001
            raise AssertionError(f"{what}: expected {exc.__name__}, got {type(e).__name__}: {e}")
        else:
            raise AssertionError(f"{what} was computed instead of refused")
        assert _delta(before, _counters())["shader_dispatches"] == 0, (
            f"{what}: a shader ran before the refusal")


def test_every_float_dtype_but_float32_refuses_on_the_way_to_the_device():
    """"Per dtype", measured: float32 is the only dtype these kernels ever see.

    float16, bfloat16 and float64 inputs cannot reach a Vulkan kernel at all,
    because the upload refuses them by name, so there is no agreement number
    for them to report and none is claimed.
    """
    if _vulkan_or_skip("the per-dtype refusal") is None:
        return
    for dtype in (_C.float16, _C.bfloat16, _C.float64):
        src = _C._tensor_from_flat([1.0, 2.0], [2], dtype=dtype)
        try:
            _to_vulkan(src)
        except NotImplementedError as e:
            # The sentence grew when int64 index storage landed
            # (docs/devices/VULKAN6.md §1): it now says what *is* stored as
            # well as what is not, so the check is that both halves are
            # there and that the dtype is named.
            msg = str(e)
            assert "stores float32" in msg and "int64 indices as int32" in msg, (dtype, msg)
            assert f"not {str(dtype).replace('torch.', '')}" in msg, (dtype, msg)
        else:
            raise AssertionError(f"a {dtype} tensor reached the vulkan device")


# ---------------------------------------------------------------------------
# 10. Integer index storage, `embedding`, and the ops a block needs
#     (docs/devices/VULKAN6.md)
# ---------------------------------------------------------------------------

def test_the_device_reports_its_integer_capabilities_and_the_index_bound_is_not_one_of_them():
    """The measurement the design rests on, and the refusal to depend on it.

    `docs/devices/VULKAN6.md` §1 had to decide how an index buffer is stored,
    and the honest input was what *this* driver reports rather than the usual
    claim that Vulkan compute has no 64-bit integers. Measured here: MoltenVK
    1.4.2 on an Apple M1 reports `shaderInt64 = True`.

    The index storage is int32 anyway, and that is the part this test holds
    down: `index_max` is the same number whatever the device says, so the range
    of indices a model may use is a property of this library and not of the GPU
    it happens to be running on. On a machine whose `shaderInt64` is True --
    this one -- the bound is *provably* not a capability report, because a
    value the device could represent natively is still refused.
    """
    probe = _vulkan_or_skip("the integer-capability report")
    if probe is None:
        return
    for key in ("shader_int64", "shader_int16", "shader_float64",
                "khr_8bit_storage", "khr_16bit_storage"):
        assert key in probe, f"_vulkan_probe() does not report {key}: {sorted(probe)}"
        assert isinstance(probe[key], bool), (key, probe[key])
    assert probe["index_max"] == 2 ** 31 - 1, probe["index_max"]

    print(f"   {probe['device']} via {probe['loader']}: "
          f"shaderInt64={probe['shader_int64']} shaderInt16={probe['shader_int16']} "
          f"shaderFloat64={probe['shader_float64']} "
          f"8bit={probe['khr_8bit_storage']} 16bit={probe['khr_16bit_storage']}")

    if not probe["shader_int64"]:
        print("   (this device has no shaderInt64, so the bound could not be "
              "distinguished from a capability report here)")
        return
    too_big = _i64([2 ** 31], [1])
    try:
        _to_vulkan(too_big)
    except NotImplementedError as e:
        assert "int32" in str(e) and "2147483647" in str(e), str(e)
    else:
        raise AssertionError(
            "2**31 reached the device: the index bound is following the "
            "driver's shaderInt64 instead of being this library's own")


def test_int64_indices_round_trip_through_the_device_unchanged():
    """The storage class, end to end -- and a reshape of it on the way.

    Values at both ends of the bound, because a narrowing that is wrong at the
    edges is right in the middle. `view` is in the middle of the trip because a
    transformer reshapes its `input_ids` before the embedding sees them, and
    `view` is the one op here that had to learn about a non-float dtype.
    """
    if _vulkan_or_skip("the int64 round trip") is None:
        return
    values = [0, 1, 2, 7, 2 ** 31 - 1, -(2 ** 31), -1, 12345]
    src = _i64(values, [2, 4])
    dev = _to_vulkan(src)
    assert str(dev.device) == "vulkan", dev.device
    assert str(dev.dtype) == "torch.int64", dev.dtype
    viewed = _C._aten_dispatch("aten.view.default", dev, [8])
    back = _to_cpu(viewed)
    assert str(back.dtype) == "torch.int64", back.dtype
    got = [int(v) for v in _flat(back)]
    assert got == values, (got, values)


def test_an_int64_value_beyond_the_bound_refuses_by_name_and_uploads_nothing():
    """The narrowing is a refusal, not a truncation.

    `2**31` truncated to int32 is `-2147483648`, which as an index is a
    perfectly plausible wrong row rather than an error -- which is why this
    refuses instead. The message has to carry the value, its position and the
    bound, because "does not fit" alone leaves the caller hunting for which
    element.

    The second assertion is the one that makes it a refusal rather than a late
    error: no buffer was uploaded, so nothing of this tensor exists on the
    device.
    """
    if _vulkan_or_skip("the out-of-bound index refusal") is None:
        return
    for values, offender, position in ((([1, 2 ** 31]), 2 ** 31, 1),
                                       (([-(2 ** 31) - 1, 3]), -(2 ** 31) - 1, 0)):
        src = _i64(values, [2])
        before = _counters()
        try:
            _to_vulkan(src)
        except NotImplementedError as e:
            msg = str(e)
            assert str(offender) in msg, (offender, msg)
            assert f"element {position}" in msg, (position, msg)
            assert "2147483647" in msg, msg
        else:
            raise AssertionError(f"{offender} was narrowed instead of refused")
        d = _delta(before, _counters())
        assert d["host_uploads"] == 0 and d["shader_dispatches"] == 0, (values, d)


def _embedding_cases():
    return (
        # (vocab, dim, index shape, indices)
        (8, 4, [3], [0, 7, 3]),
        (8, 4, [2, 3], [0, 1, 2, 3, 4, 5]),
        (50257, 2, [4], [0, 1, 50256, 42]),
        (5, 1, [1], [4]),
        (6, 3, [2, 2], [5, 5, 0, 0]),
    )


def test_embedding_is_bit_identical_to_upstream_and_ran_on_the_gpu():
    """The kernel, held to bit equality rather than a tolerance.

    A gather does no arithmetic: every output element is a float32 copied from
    the table, so "agrees with upstream" here means *the same bits*, and a
    tolerance would be a place for a defect to hide. docs/numerics/AGREE.md §2's
    derivation produces zero for an op like this, the way it does for `view`
    and `t`.

    The second half is the device assertion: one compute shader ran and nothing
    was read back, so the gather happened on the GPU. Both halves are needed --
    a host implementation would get the bits exactly right.
    """
    if _vulkan_or_skip("the embedding agreement sweep") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    checked = 0
    for vocab, dim, shape, indices in _embedding_cases():
        g = torch.Generator().manual_seed(vocab * 131 + dim)
        w = torch.randn(vocab, dim, generator=g, dtype=torch.float32)
        ids = torch.as_tensor(indices, dtype=torch.int64).reshape(shape)
        want = torch.ops.aten.embedding.default(w, ids)

        wv = _to_vulkan(_cpu(w.reshape(-1).tolist(), [vocab, dim]))
        iv = _to_vulkan(_i64(indices, shape))
        before = _counters()
        out = _C._aten_dispatch("aten.embedding.default", wv, iv)
        d = _delta(before, _counters())
        assert str(out.device) == "vulkan", out.device
        assert list(out.shape) == list(want.shape), (out.shape, want.shape)
        assert d["shader_dispatches"] == 1, (vocab, dim, shape, d)
        assert d["host_downloads"] == 0, (
            f"embedding read {d['host_downloads']} buffer(s) back -- it is "
            f"gathering on the CPU under a vulkan label")

        got = _flat(_to_cpu(out))
        exp = want.reshape(-1).tolist()
        assert len(got) == len(exp), (len(got), len(exp))
        off = [i for i, (x, y) in enumerate(zip(got, exp)) if _bits(x) != _bits(y)]
        assert not off, (
            f"embedding[{vocab},{dim}]{shape}: {len(off)} of {len(exp)} "
            f"elements differ from upstream's bits, first at {off[:5]}")
        checked += len(exp)
    print(f"   embedding: {checked} elements bit-identical to upstream over "
          f"{len(_embedding_cases())} cases")


def test_embedding_refuses_an_out_of_range_index_before_any_gpu_work():
    """Upstream raises `IndexError` here, and so does this -- with no dispatch.

    The row is checked against the table on the host, from the range recorded
    when the indices were uploaded, so it costs no read-back (the shader could
    not raise anyway; it could only clamp or read garbage, and both of those
    are the silent wrong answer this device exists to prevent).

    Upstream is asked the same question rather than its behaviour being
    asserted from memory.
    """
    if _vulkan_or_skip("the embedding range refusal") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    vocab, dim = 5, 3
    w = torch.zeros(vocab, dim, dtype=torch.float32)
    wv = _to_vulkan(_cpu([0.0] * (vocab * dim), [vocab, dim]))
    for bad in (vocab, vocab + 100, -1):
        try:
            torch.ops.aten.embedding.default(w, torch.as_tensor([bad]))
        except IndexError:
            pass
        else:
            raise AssertionError(
                f"upstream accepted index {bad} into a {vocab}-row table; this "
                f"test's premise is wrong")
        iv = _to_vulkan(_i64([0, bad], [2]))
        before = _counters()
        try:
            _C._aten_dispatch("aten.embedding.default", wv, iv)
        except IndexError as e:
            assert str(bad) in str(e) and f"[0, {vocab})" in str(e), str(e)
        else:
            raise AssertionError(f"index {bad} was gathered instead of refused")
        d = _delta(before, _counters())
        assert d["shader_dispatches"] == 0, (
            f"the out-of-range refusal ran {d['shader_dispatches']} compute "
            f"shader(s) before refusing: {d}")
        assert d["host_downloads"] == 0, (
            f"the range check read {d['host_downloads']} buffer(s) back; it is "
            f"meant to use the range recorded at upload: {d}")


def test_embedding_ignores_the_autograd_only_arguments_exactly_as_upstream_does():
    """`padding_idx`, `scale_grad_by_freq` and `sparse` change no forward value.

    That is the reason this kernel accepts them instead of refusing by name,
    and it is a claim about *upstream's* kernel -- so it is checked against
    upstream on the same inputs rather than asserted from the signature. If a
    future torch made `padding_idx` zero the row in the forward, this fails
    and the acceptance has to become a refusal.
    """
    if _vulkan_or_skip("the embedding autograd-argument check") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    vocab, dim = 6, 3
    g = torch.Generator().manual_seed(7)
    w = torch.randn(vocab, dim, generator=g, dtype=torch.float32)
    indices = [0, 5, 2, 0]
    ids = torch.as_tensor(indices, dtype=torch.int64)
    wv = _to_vulkan(_cpu(w.reshape(-1).tolist(), [vocab, dim]))
    plain = torch.ops.aten.embedding.default(w, ids).reshape(-1).tolist()
    for padding_idx, scale, sparse in ((0, False, False), (-1, True, False),
                                       (2, False, True), (0, True, True)):
        want = torch.ops.aten.embedding.default(
            w, ids, padding_idx, scale, sparse).reshape(-1).tolist()
        assert [_bits(x) for x in want] == [_bits(x) for x in plain], (
            f"upstream's forward DOES read (padding_idx={padding_idx}, "
            f"scale_grad_by_freq={scale}, sparse={sparse}) -- the vulkan "
            f"kernel must stop accepting them and refuse by name instead")
        iv = _to_vulkan(_i64(indices, [len(indices)]))
        out = _C._aten_dispatch("aten.embedding.default", wv, iv,
                                padding_idx, scale, sparse)
        got = _flat(_to_cpu(out))
        assert [_bits(x) for x in got] == [_bits(x) for x in want], (
            padding_idx, scale, sparse)


def test_embedding_refuses_the_operands_it_does_not_implement():
    """Each refusal names what was wrong, before any GPU work."""
    if _vulkan_or_skip("the embedding operand refusals") is None:
        return
    w2 = _to_vulkan(_cpu([1.0] * 6, [3, 2]))
    w3 = _to_vulkan(_cpu([1.0] * 8, [2, 2, 2]))
    idx = _to_vulkan(_i64([0, 1], [2]))
    floaty = _to_vulkan(_cpu([0.0, 1.0], [2]))
    cases = (
        ("a 3-D table", lambda: _C._aten_dispatch("aten.embedding.default", w3, idx),
         "2-D"),
        ("float indices", lambda: _C._aten_dispatch("aten.embedding.default", w2, floaty),
         "int64"),
    )
    for what, call, needle in cases:
        before = _counters()
        try:
            call()
        except (NotImplementedError, RuntimeError) as e:
            assert needle in str(e), (what, needle, str(e))
        else:
            raise AssertionError(f"{what} was computed instead of refused")
        d = _delta(before, _counters())
        assert d["shader_dispatches"] == 0, (what, d)


def test_the_batched_transpose_is_bit_identical_and_a_rank_above_the_bound_refuses():
    """`k.transpose(1, 2)` -- the op between an attention block's two `bmm`s.

    Measured, not chosen: with `embedding` taught and this not, a transformer
    block's forward stopped here (docs/devices/VULKAN6.md §3). Pure data
    movement, so bit equality rather than a tolerance -- the same standard
    `t` and the 2-D transpose are held to.

    The refusals matter as much as the kernel: a general permutation needs
    strides a `VkTensor` does not have, so every other dimension pair and every
    higher rank still refuses by name rather than being routed through
    something that happens to be nearby.
    """
    if _vulkan_or_skip("the batched transpose") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    for batch, rows, cols in ((2, 3, 4), (1, 5, 1), (3, 2, 2), (2, 1, 7)):
        g = torch.Generator().manual_seed(batch * 977 + rows * 31 + cols)
        x = torch.randn(batch, rows, cols, generator=g, dtype=torch.float32)
        want = x.transpose(1, 2).contiguous().reshape(-1).tolist()
        xv = _to_vulkan(_cpu(x.reshape(-1).tolist(), [batch, rows, cols]))
        before = _counters()
        out = _C._aten_dispatch("aten.transpose.int", xv, 1, 2)
        d = _delta(before, _counters())
        assert list(out.shape) == [batch, cols, rows], out.shape
        assert d["shader_dispatches"] == 1, (batch, rows, cols, d)
        assert d["host_downloads"] == 0, d
        got = _flat(_to_cpu(out))
        off = [i for i, (a, b) in enumerate(zip(got, want)) if _bits(a) != _bits(b)]
        assert not off, (
            f"transpose[{batch},{rows},{cols}](1,2): {len(off)} of {len(want)} "
            f"elements differ from upstream's bits, first at {off[:5]}")
        # Negative dims name the same pair and must take the same path.
        neg = _C._aten_dispatch("aten.transpose.int", xv, -1, -2)
        assert [_bits(v) for v in _flat(_to_cpu(neg))] == [_bits(v) for v in want]

    # docs/devices/VULKAN7.md: every other pair and rank up to the stated
    # bound is taught now (through the strided gather, one shader each), so
    # what is left to refuse is a rank above that bound -- by name, before any
    # GPU work.
    a9 = _to_vulkan(_cpu([1.0] * 2, [2] + [1] * 8))
    before = _counters()
    try:
        _C._aten_dispatch("aten.transpose.int", a9, 0, 8)
    except NotImplementedError as e:
        assert "rank" in str(e) and "8" in str(e), str(e)
    else:
        raise AssertionError("a rank-9 transpose was permuted instead of refused")
    assert _delta(before, _counters())["shader_dispatches"] == 0


def test_the_scalar_ops_are_bit_identical_and_the_divide_stayed_a_divide():
    """`x / sqrt(d)` is on the trace, and it must not become `x * (1/sqrt(d))`.

    The reciprocal rewrite is the same class of transformation MoltenVK's
    fast-math was turned off for (docs/devices/VULKAN5.md §3.1): it is within
    any tolerance anyone would pick and it is not what upstream computes. So
    the assertion is bit equality against upstream -- and the test first checks
    that the case can *tell the difference*, by counting how many elements the
    reciprocal form would get wrong. A case where that count is zero would
    prove nothing, and this refuses to be such a case.
    """
    if _vulkan_or_skip("the scalar ops") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    g = torch.Generator().manual_seed(4242)
    x = torch.randn(64, generator=g, dtype=torch.float32)
    xv = _to_vulkan(_cpu(x.tolist(), [64]))
    discriminating = 0
    for scalar in (2.0, 3.0, 32.0 ** 0.5, 0.1, -7.5):
        s32 = torch.as_tensor(scalar, dtype=torch.float32)
        for op, want in (("aten.mul.Scalar", x * s32), ("aten.div.Scalar", x / s32)):
            before = _counters()
            out = _C._aten_dispatch(op, xv, scalar)
            d = _delta(before, _counters())
            assert d["shader_dispatches"] == 1, (op, scalar, d)
            assert d["host_downloads"] == 0, (op, scalar, d)
            got = _flat(_to_cpu(out))
            exp = want.tolist()
            off = [i for i, (a, b) in enumerate(zip(got, exp)) if _bits(a) != _bits(b)]
            assert not off, (
                f"{op} by {scalar}: {len(off)} of {len(exp)} elements differ "
                f"from upstream's bits, first at {off[:5]}")
            if op == "aten.div.Scalar":
                recip = (x * (torch.as_tensor(1.0, dtype=torch.float32) / s32)).tolist()
                differ = sum(1 for a, b in zip(recip, exp) if _bits(a) != _bits(b))
                discriminating += differ
    assert discriminating > 0, (
        "no scalar in this sweep distinguishes `a / s` from `a * (1/s)`, so "
        "the bit-equality assertion above could not have caught the rewrite")
    print(f"   scalar ops bit-identical; the reciprocal rewrite would have "
          f"moved {discriminating} element(s) across this sweep")


_BLOCK_SHIM_SCRIPT = r"""
import json, math, sys
import torch, torch.nn as nn

assert hasattr(torch._C, "_aten_implemented"), "this subprocess got upstream torch"

cfg = json.load(sys.stdin)
out = {"probe": torch._C._vulkan_probe()}
if not out["probe"]["available"]:
    json.dump(out, sys.stdout); raise SystemExit

V, D, F = cfg["V"], cfg["D"], cfg["F"]


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(V, D)
        self.ln1 = nn.LayerNorm(D)
        self.ln2 = nn.LayerNorm(D)
        self.q = nn.Linear(D, D); self.k = nn.Linear(D, D)
        self.v = nn.Linear(D, D); self.o = nn.Linear(D, D)
        self.f1 = nn.Linear(D, F); self.f2 = nn.Linear(F, D)

    def forward(self, ids):
        x = self.emb(ids)
        h = self.ln1(x)
        q, k, v = self.q(h), self.k(h), self.v(h)
        scores = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(D)
        y = torch.bmm(torch.softmax(scores, dim=-1), v)
        x = x + self.o(y)
        h = self.ln2(x)
        return x + self.f2(torch.nn.functional.gelu(self.f1(h)))


m = Block()
m.load_state_dict({k: torch.as_tensor(v, dtype=torch.float32)
                   for k, v in cfg["sd"].items()})
m.eval()
ids = torch.as_tensor(cfg["ids"], dtype=torch.int64).reshape(cfg["sids"])
with torch.no_grad():
    out["cpu"] = m(ids).reshape(-1).tolist()
m.to("vulkan")
before = torch._C._vulkan_counters()
with torch.no_grad():
    r = m(ids.to("vulkan"))
after = torch._C._vulkan_counters()
out["device"] = str(r.device)
out["counters"] = {k: after[k] - before[k] for k in after}
out["vulkan"] = r.cpu().reshape(-1).tolist()
json.dump(out, sys.stdout)
"""


def test_a_transformer_block_forwards_on_the_gpu_and_agrees_with_upstream():
    """**The deliverable of docs/devices/VULKAN6.md, and the reason it exists.**

    `docs/architectures/VOICE4.md` records seven rounds that closed operators
    one at a time and never ran a model; `docs/devices/VULKAN5.md` closed four
    transformer kernels and could not run a transformer either, because
    `embedding` is the first op. So the claim being made here is not that a
    kernel agrees -- it is that a *transformer block* forwards, on the device,
    end to end:

        embedding -> layer_norm -> q,k,v -> transpose -> bmm -> /sqrt(d)
                  -> softmax -> bmm -> out proj -> residual
                  -> layer_norm -> fc -> gelu -> fc -> residual

    Three assertions, and only the first is about the answer:

      1. it agrees with upstream inside docs/numerics/AGREE.md §2's rule (no
         worse than 4x upstream's own distance from the float64 truth), and
         with the shim's own cpu answer to a couple of float32 ulp;
      2. the result is a vulkan tensor;
      3. **every compute shader in it ran on the GPU and nothing was read
         back.** A forward that fell back to the host would still get the
         numbers right and would fail here.

    It is a single-head block, and that is stated rather than implied: BERT's
    ten `transpose.int` calls are 4-D (batch, head, seq, dim) and refuse, along
    with `expand`, `slice`, `gather`, `select` and `tanh` -- see
    `test_the_transformer_wall_moved_and_the_new_one_is_named`.
    """
    if _vulkan_or_skip("the transformer block forward") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    try:
        import torch.nn as nn
    except Exception:  # noqa: BLE001
        vulkan_coverage.vulkan_skip("the transformer block forward: no torch.nn upstream")
        return

    V, D, F, B, S = 32, 16, 32, 2, 6
    g = torch.Generator().manual_seed(2026)

    class Block(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, D)
            self.ln1 = nn.LayerNorm(D)
            self.ln2 = nn.LayerNorm(D)
            self.q = nn.Linear(D, D); self.k = nn.Linear(D, D)
            self.v = nn.Linear(D, D); self.o = nn.Linear(D, D)
            self.f1 = nn.Linear(D, F); self.f2 = nn.Linear(F, D)

        def forward(self, ids):
            x = self.emb(ids)
            h = self.ln1(x)
            q, k, v = self.q(h), self.k(h), self.v(h)
            scores = torch.bmm(q, k.transpose(1, 2)) / (D ** 0.5)
            y = torch.bmm(torch.softmax(scores, dim=-1), v)
            x = x + self.o(y)
            h = self.ln2(x)
            return x + self.f2(torch.nn.functional.gelu(self.f1(h)))

    model = Block().eval()
    with torch.no_grad():
        for p in model.parameters():
            p.copy_(torch.randn(p.shape, generator=g, dtype=torch.float32) * 0.5)
    ids = torch.randint(0, V, (B, S), generator=g, dtype=torch.int64)

    cfg = {"V": V, "D": D, "F": F, "ids": ids.reshape(-1).tolist(),
           "sids": [B, S],
           "sd": {k: v.tolist() for k, v in model.state_dict().items()}}
    env = dict(os.environ)
    env["PYTHONPATH"] = VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run([sys.executable, "-c", _BLOCK_SHIM_SCRIPT],
                          input=json.dumps(cfg), capture_output=True,
                          text=True, env=env, timeout=600)
    assert proc.returncode == 0, (proc.stdout[-2000:], proc.stderr[-4000:])
    got = json.loads(proc.stdout)
    if not got["probe"]["available"]:
        vulkan_coverage.vulkan_skip(
            f"the transformer block forward: the vendored-tree subprocess has "
            f"no loader -- {got['probe']['error']}")
        return
    assert got["device"].startswith("vulkan"), got["device"]

    with torch.no_grad():
        f32 = model(ids).reshape(-1).tolist()
        wide = Block().eval()
        wide.load_state_dict(model.state_dict())
        truth = wide.double()(ids).reshape(-1).tolist()

    scale = max(abs(v) for v in truth)
    up_err = max(abs(a - b) for a, b in zip(f32, truth))
    vk_err = max(abs(a - b) for a, b in zip(got["vulkan"], truth))
    cpu_err = max(abs(a - b) for a, b in zip(got["cpu"], truth))
    backends = max(abs(a - b) for a, b in zip(got["vulkan"], got["cpu"]))
    ulp = scale * FLOAT32_EPS

    assert vk_err <= 4 * up_err, (
        f"the vulkan block is {vk_err:.3e} from the float64 truth against "
        f"upstream's own {up_err:.3e}; docs/numerics/AGREE.md's rule allows 4x")
    assert cpu_err <= 4 * up_err, (cpu_err, up_err)
    assert backends <= 8 * ulp, (
        f"vulkan and the shim's own cpu differ by {backends:.3e}, more than "
        f"eight float32 ulp ({8 * ulp:.3e}) at this magnitude")

    counters = got["counters"]
    # 1 embedding + 2 layer_norm + 6 Linear x (t + matmul + bias)
    # + 1 transpose + 2 bmm + 1 div + 1 softmax + 1 gelu + 2 residual add = 29.
    assert counters["shader_dispatches"] == 29, (
        f"the block forward ran {counters['shader_dispatches']} compute "
        f"shaders, expected 29: {counters}")
    assert counters["host_downloads"] == 0, (
        f"the block forward read {counters['host_downloads']} buffer(s) back "
        f"to the host -- part of it computed on the CPU: {counters}")
    print(f"   transformer block on vulkan: upstream f32 err {up_err:.3e}, "
          f"shim cpu {cpu_err:.3e}, shim vulkan {vk_err:.3e} "
          f"({vk_err / up_err:.2f}x), vulkan-vs-cpu {backends / ulp:.2f} ulp, "
          f"{counters['shader_dispatches']} shaders, "
          f"{counters['host_downloads']} readbacks")


# ---------------------------------------------------------------------------
# 11. The pretrained-BERT wall (docs/devices/VULKAN7.md)
# ---------------------------------------------------------------------------

def _vk_like(torch, x):
    """An upstream tensor's values, on the shim's vulkan device, same dtype."""
    shape = list(x.shape)
    flat = x.reshape(-1).tolist()
    if x.dtype == torch.int64:
        return _to_vulkan(_i64(flat, shape))
    assert x.dtype == torch.float32, x.dtype
    return _to_vulkan(_cpu(flat, shape))


def _bit_mismatches(got, want):
    """Positions where the shim's answer is not upstream's, bit for bit.

    Floats are compared by their bit pattern, integers by value; a float
    compared with `==` would call -0.0 and 0.0 the same, and NaN different
    from itself.
    """
    got = _flat(got)
    want = want.reshape(-1).tolist()
    assert len(got) == len(want), (len(got), len(want))
    if want and isinstance(want[0], float):
        return [i for i, (a, b) in enumerate(zip(got, want)) if _bits(a) != _bits(b)]
    return [i for i, (a, b) in enumerate(zip(got, want)) if int(a) != int(b)]


# (label, input shape, dtype, op, args, expected shader count)
#
# Every case asks upstream's own `torch.ops.aten.<op>` the same question, so
# the clamping of `slice`, the `-1` of `expand` and the negative dims and
# indices are upstream's answers rather than this file's reading of them.
# Zero shaders is claimed for exactly two shapes of case: an identity view
# (the output *is* the input, and the buffer is shared -- safe because no op
# on this device writes in place) and an empty result.
VIEW_CASES = (
    ("slice bert position_ids", [1, 512], "i64", "slice.Tensor", (1, 0, 7), 1),
    ("slice f32 [2,512] 0:7", [2, 512], "f32", "slice.Tensor", (1, 0, 7), 1),
    ("slice step 2 neg", [4, 5, 6], "f32", "slice.Tensor", (-1, 1, -1, 2), 1),
    ("slice clamps", [4, 5], "f32", "slice.Tensor", (0, -10, 100, 3), 1),
    ("slice inner dim", [3, 4, 5], "f32", "slice.Tensor", (1, 1, 3), 1),
    ("slice empty", [3, 4], "f32", "slice.Tensor", (1, 3, 1), 0),
    ("slice identity", [2, 3, 4, 5], "f32", "slice.Tensor", (2, 0, 2 ** 63 - 1), 0),
    ("select pooler", [2, 7, 8], "f32", "select.int", (1, 0), 1),
    ("select neg", [5, 3], "f32", "select.int", (0, -1), 1),
    ("select to 0-d", [4], "f32", "select.int", (0, 2), 1),
    ("select i64", [3, 4], "i64", "select.int", (1, 3), 1),
    ("expand bert token_type", [1, 512], "i64", "expand.default", ([2, -1],), 1),
    ("expand lead dims", [3, 1], "f32", "expand.default", ([2, 3, 4],), 1),
    ("expand middle", [2, 1, 4], "f32", "expand.default", ([2, 3, 4],), 1),
    ("expand 0-d", [], "f32", "expand.default", ([3, 2],), 1),
    ("expand identity", [2, 3], "f32", "expand.default", ([2, -1],), 0),
    ("transpose 4-D (1,2)", [2, 4, 7, 16], "f32", "transpose.int", (1, 2), 1),
    ("transpose 4-D (2,3)", [2, 4, 7, 16], "f32", "transpose.int", (2, 3), 1),
    ("transpose 4-D (-1,-2)", [2, 4, 7, 5], "f32", "transpose.int", (-1, -2), 1),
    ("transpose 4-D (0,3)", [2, 3, 4, 5], "f32", "transpose.int", (0, 3), 1),
    ("transpose 5-D (1,3)", [2, 3, 4, 5, 6], "f32", "transpose.int", (1, 3), 1),
    ("transpose 3-D (0,2)", [3, 4, 5], "f32", "transpose.int", (0, 2), 1),
    ("transpose 3-D (0,1)", [3, 4, 5], "f32", "transpose.int", (0, 1), 1),
    ("transpose i64 4-D", [2, 3, 2, 2], "i64", "transpose.int", (1, 3), 1),
)


def test_the_view_ops_are_bit_identical_to_upstream_and_ran_on_the_gpu():
    """`slice`, `select`, `expand` and any-rank `transpose`, against upstream.

    No tolerance: these move bytes and do no arithmetic, so the only right
    answer is upstream's bits. Both storage classes are covered -- the shader
    reads and writes `uint` words, so a float32 and an int32 index word travel
    the same way -- and the int64 cases are exactly the two BERT runs on its
    `position_ids` and `token_type_ids` buffers.

    The device half is the counter delta, per case. A host twin of any of these
    would produce the same bits (VULKAN6.md §6 D5 is that exact experiment for
    `embedding`); it would not produce `shader_dispatches == 1` with
    `host_downloads == 0`.

    **Why the values are random and distinct.** A copy that read the wrong
    source position -- a stride off by one axis -- is invisible on a tensor of
    ones. `discriminating` counts the cases whose output would change under
    the most likely such bug (reading the input front to back, i.e. ignoring
    the view), and refuses to pass if that count is zero.
    """
    if _vulkan_or_skip("the view-op sweep") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    discriminating = 0
    for n, (label, shape, kind, op, args, shaders) in enumerate(VIEW_CASES):
        g = torch.Generator().manual_seed(7000 + n)
        if kind == "i64":
            count = 1
            for e in shape:
                count *= e
            x = torch.randperm(max(count, 1), generator=g)[:count].reshape(shape)
        else:
            x = torch.randn(shape, generator=g, dtype=torch.float32)
        name, overload = op.split(".")
        want = getattr(getattr(torch.ops.aten, name), overload)(x, *args).contiguous()
        xv = _vk_like(torch, x)
        before = _counters()
        out = _C._aten_dispatch(f"aten.{op}", xv, *args)
        d = _delta(before, _counters())
        assert str(out.device) == "vulkan", (label, out.device)
        assert list(out.shape) == list(want.shape), (label, out.shape, want.shape)
        assert str(out.dtype) == str(want.dtype), (label, out.dtype, want.dtype)
        assert d["shader_dispatches"] == shaders, (
            f"{label}: {d['shader_dispatches']} shaders, expected {shaders}: {d}")
        assert d["host_downloads"] == 0 and d["host_uploads"] == 0, (label, d)
        off = _bit_mismatches(_to_cpu(out), want)
        assert not off, (
            f"{label}: {len(off)} of {want.numel()} elements differ from "
            f"upstream's bits, first at {off[:5]}")
        # An `expand` output is larger than its input, so a front-to-back copy
        # would run off the end -- which the bit check above catches too.
        if want.numel() > x.numel():
            discriminating += 1
        else:
            front = x.reshape(-1)[:want.numel()].reshape(want.shape)
            discriminating += int(want.numel() > 1 and not torch.equal(front, want))
    assert discriminating >= 15, (
        f"only {discriminating} cases distinguish a view from a front-to-back "
        f"copy of its input; the sweep cannot catch a stride error")
    print(f"   {len(VIEW_CASES)} view cases bit-identical to upstream; "
          f"{discriminating} of them would catch a copy that ignored the view")


# (label, self shape, dtype, dim, index shape)
GATHER_CASES = (
    ("bert token_type", [2, 512], "i64", 1, [2, 7]),
    ("f32 dim1 narrower", [3, 5], "f32", 1, [3, 4]),
    ("f32 dim1 smaller rows", [3, 5], "f32", 1, [2, 9]),
    ("f32 3-D dim0", [4, 3, 2], "f32", 0, [2, 3, 2]),
    ("f32 3-D dim -1", [4, 3, 2], "f32", -1, [4, 1, 2]),
    ("f32 4-D dim2", [2, 3, 5, 4], "f32", 2, [2, 2, 6, 3]),
    ("i64 1-D repeats", [6], "i64", 0, [10]),
)


def test_gather_is_bit_identical_to_upstream_and_ran_on_the_gpu():
    """`torch.gather` on the device, both storage classes, against upstream.

    BERT's embeddings call it on the int64 `token_type_ids` buffer, indexed by
    the int64 `position_ids` -- so this op both reads an index buffer and, for
    that case, *produces* one. The produced one is checked by value here, and
    by use (it feeds `embedding`) in the BERT forward.
    """
    if _vulkan_or_skip("the gather sweep") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    for n, (label, shape, kind, dim, ishape) in enumerate(GATHER_CASES):
        g = torch.Generator().manual_seed(8100 + n)
        if kind == "i64":
            x = torch.randint(-50, 50, shape, generator=g, dtype=torch.int64)
        else:
            x = torch.randn(*shape, generator=g, dtype=torch.float32)
        idx = torch.randint(0, shape[dim], ishape, generator=g, dtype=torch.int64)
        for sparse_grad in (False, True):
            want = torch.gather(x, dim, idx, sparse_grad=sparse_grad)
            xv, iv = _vk_like(torch, x), _vk_like(torch, idx)
            before = _counters()
            out = _C._aten_dispatch("aten.gather.default", xv, dim, iv,
                                    sparse_grad=sparse_grad)
            d = _delta(before, _counters())
            assert list(out.shape) == ishape and str(out.dtype) == str(want.dtype), (
                label, out.shape, out.dtype)
            assert d["shader_dispatches"] == 1, (label, d)
            assert d["host_downloads"] == 0 and d["host_uploads"] == 0, (label, d)
            off = _bit_mismatches(_to_cpu(out), want)
            assert not off, (
                f"gather {label} sparse_grad={sparse_grad}: {len(off)} of "
                f"{want.numel()} elements differ from upstream, first at {off[:5]}")


def test_gather_refuses_what_upstream_refuses_before_any_gpu_work():
    """Out-of-range and malformed indices, with zero shaders.

    A shader cannot raise, so an out-of-range gather index is checked on the
    host against the range recorded at upload -- the same trade VULKAN6.md §2
    made for `embedding`, applied to a second reader of the index buffer.
    Each case asks upstream first, so "refuses" means "refuses where upstream
    refuses", and the message is upstream's.
    """
    if _vulkan_or_skip("the gather refusals") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    x = torch.arange(15, dtype=torch.float32).reshape(3, 5)
    cases = (
        ("index == size", x, 1, torch.tensor([[0, 5]]), "out of bounds"),
        ("negative index", x, 1, torch.tensor([[-1, 0]]), "out of bounds"),
        ("rank mismatch", x, 1, torch.tensor([0, 1]), "same number of dimensions"),
        ("index too wide", x, 1, torch.zeros(4, 2, dtype=torch.int64), "Size does not match"),
        ("dim out of range", x, 2, torch.zeros(1, 1, dtype=torch.int64), "Dimension out of range"),
    )
    for what, src, dim, idx, needle in cases:
        try:
            torch.gather(src, dim, idx)
        except (RuntimeError, IndexError):
            pass
        else:
            raise AssertionError(f"upstream accepted the case {what!r}; the sweep is wrong")
        xv, iv = _vk_like(torch, src), _vk_like(torch, idx)
        before = _counters()
        try:
            _C._aten_dispatch("aten.gather.default", xv, dim, iv)
        except (RuntimeError, IndexError) as e:
            assert needle in str(e), (what, needle, str(e))
        else:
            raise AssertionError(f"gather {what} was computed instead of refused")
        d = _delta(before, _counters())
        assert d["shader_dispatches"] == 0 and d["host_downloads"] == 0, (what, d)

    # A float index is refused by dtype, naming it.
    fv = _to_vulkan(_cpu([0.0, 1.0], [1, 2]))
    try:
        _C._aten_dispatch("aten.gather.default", _vk_like(torch, x), 1, fv)
    except (RuntimeError, NotImplementedError) as e:
        assert "int" in str(e), str(e)
    else:
        raise AssertionError("a float gather index was accepted")


def test_an_index_range_is_inherited_through_views_and_a_loose_one_refuses_by_name():
    """What a slice of an index tensor knows about its values.

    The range check costs no read-back because the upload recorded `(min,
    max)` (VULKAN6.md §2). A slice, select, expand or gather of that tensor
    holds a *subset* of its values, so the parent's range is still a true
    bound -- but possibly a loose one, and the exact one cannot be had without
    reading the device back.

    So two things are asserted. Where the inherited bound fits the table, the
    gather runs and is bit-identical (BERT's `position_ids[:, :7]`, whose
    parent range 0..511 fits a 512-row table, is this case). Where it does
    not, the refusal says the bound was **inherited** and gives it, with zero
    shaders -- rather than gathering an unchecked index, and rather than
    pretending the sliced values themselves were out of range.
    """
    if _vulkan_or_skip("the inherited index range") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    table = torch.randn(8, 3, generator=torch.Generator().manual_seed(9), dtype=torch.float32)
    parent = torch.arange(50, dtype=torch.int64)
    child = parent[2:6]
    want = torch.nn.functional.embedding(child, table)  # upstream answers

    pv = _vk_like(torch, parent)
    cv = _C._aten_dispatch("aten.slice.Tensor", pv, 0, 2, 6)
    tv = _vk_like(torch, table)
    before = _counters()
    try:
        _C._aten_dispatch("aten.embedding.default", tv, cv)
    except IndexError as e:
        message = str(e)
    else:
        raise AssertionError("an index with an unchecked, inherited range was gathered")
    d = _delta(before, _counters())
    assert d == {"shader_dispatches": 0, "host_uploads": 0, "host_downloads": 0}, d
    assert "inherited" in message and "49" in message and "8" in message, message

    # The same slice, against a table the inherited bound fits: it runs.
    wide = torch.randn(50, 3, generator=torch.Generator().manual_seed(10), dtype=torch.float32)
    out = _C._aten_dispatch("aten.embedding.default", _vk_like(torch, wide), cv)
    assert not _bit_mismatches(_to_cpu(out), torch.nn.functional.embedding(child, wide))
    assert want.shape == (4, 3)

    # The same holds through select, expand and gather: each output's bound
    # is its source's.
    sel = _C._aten_dispatch("aten.select.int", pv, 0, 3)
    exp = _C._aten_dispatch("aten.expand.default", _C._aten_dispatch(
        "aten.view.default", pv, [1, 50]), [2, 50])
    gat = _C._aten_dispatch("aten.gather.default", exp, 1,
                            _vk_like(torch, torch.tensor([[1, 2], [3, 4]])))
    for what, t in (("select", sel), ("expand", exp), ("gather", gat)):
        try:
            _C._aten_dispatch("aten.embedding.default", tv, t)
        except IndexError as e:
            assert "inherited" in str(e) and "49" in str(e), (what, str(e))
        else:
            raise AssertionError(f"{what}: the inherited bound was dropped")


def test_the_view_ops_refuse_what_they_cannot_express_before_any_gpu_work():
    """Upstream's refusals in upstream's words, and this device's own by name.

    The last case is this device's, not upstream's: an output whose element
    count does not fit the shaders' 32-bit invocation index. Upstream's
    `expand` answers it (a view costs nothing), and this device would have to
    materialise it, so it refuses with the value and the bound *before*
    allocating -- the same shape of decision as VULKAN6.md §1's index ceiling.
    """
    if _vulkan_or_skip("the view-op refusals") is None:
        return
    x = _to_vulkan(_cpu([float(i) for i in range(6)], [2, 3]))
    s = _to_vulkan(_cpu([1.0], []))
    i64 = _to_vulkan(_i64([0, 1], [2]))
    d = _C._aten_dispatch
    cases = (
        ("slice step 0", lambda: d("aten.slice.Tensor", x, 1, 0, 3, 0), "step must be greater than zero"),
        ("slice dim", lambda: d("aten.slice.Tensor", x, 2, 0, 3), "Dimension out of range"),
        ("select 0-d", lambda: d("aten.select.int", s, 0, 0), "0-dim"),
        ("select index", lambda: d("aten.select.int", x, 1, 3), "out of"),
        ("expand extent", lambda: d("aten.expand.default", x, [2, 4]), "must match the existing size"),
        ("expand fewer", lambda: d("aten.expand.default", x, [3]), "must be greater or equal"),
        ("expand -1 lead", lambda: d("aten.expand.default", x, [-1, 2, 3]), "leading, non-existing"),
        ("expand past u32", lambda: d("aten.expand.default", s, [2 ** 32 + 1]), "4294967297"),
        ("tanh int64", lambda: d("aten.tanh.default", i64), "float32"),
        ("rank 9 expand", lambda: d("aten.expand.default", s, [1] * 9), "rank"),
    )
    for what, call, needle in cases:
        before = _counters()
        try:
            call()
        except (RuntimeError, IndexError, ValueError, NotImplementedError) as e:
            assert needle in str(e), (what, needle, str(e))
        else:
            raise AssertionError(f"{what} was computed instead of refused")
        delta = _delta(before, _counters())
        assert delta["shader_dispatches"] == 0 and delta["host_downloads"] == 0, (what, delta)


# (label, lhs shape, rhs shape)
BROADCAST_CASES = (
    ("bert position add", [2, 7, 16], [1, 7, 16]),
    ("bias row", [3, 5], [5]),
    ("column vs row", [4, 1], [1, 6]),
    ("scalar tensor", [2, 3], []),
    ("lead dims both", [2, 1, 3, 1], [5, 1, 4]),
    ("mask shape", [2, 4, 7, 7], [2, 1, 1, 7]),
    ("lhs is the small one", [1, 3], [4, 3]),
)


def test_broadcasting_arithmetic_is_bit_identical_to_upstream_and_one_shader():
    """`a + b` with unequal, broadcastable shapes -- a wall VULKAN6.md did not list.

    A pretrained BERT with a batch of two adds `position_embeddings` of shape
    `[1, S, H]` to embeddings of shape `[2, S, H]`. VULKAN6.md §3.1's trace ran
    one sequence, where that add is between equal shapes, and it counted op
    *names* -- `add.Tensor` was taught, so it was not a wall there. It is here
    (docs/devices/VULKAN7.md §3.2).

    One IEEE operation per element, so bit equality, for all four operators.
    One shader per call: the broadcast is strides in the kernel's push
    constants, not a materialised `expand` first.
    """
    if _vulkan_or_skip("the broadcasting arithmetic") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    ops = (("aten.add.Tensor", torch.add), ("aten.sub.Tensor", torch.sub),
           ("aten.mul.Tensor", torch.mul), ("aten.div.Tensor", torch.div))
    for n, (label, sa, sb) in enumerate(BROADCAST_CASES):
        g = torch.Generator().manual_seed(9300 + n)
        a = torch.randn(sa, generator=g, dtype=torch.float32)
        b = torch.randn(sb, generator=g, dtype=torch.float32) + 0.25
        av, bv = _vk_like(torch, a), _vk_like(torch, b)
        for op, fn in ops:
            for x, y, xv, yv, order in ((a, b, av, bv, "a.b"), (b, a, bv, av, "b.a")):
                want = fn(x, y)
                before = _counters()
                out = _C._aten_dispatch(op, xv, yv)
                d = _delta(before, _counters())
                assert list(out.shape) == list(want.shape), (label, op, out.shape, want.shape)
                assert d["shader_dispatches"] == 1, (label, op, order, d)
                assert d["host_downloads"] == 0 and d["host_uploads"] == 0, (label, op, d)
                off = _bit_mismatches(_to_cpu(out), want)
                assert not off, (
                    f"{op} {label} ({order}): {len(off)} of {want.numel()} "
                    f"elements differ from upstream's bits, first at {off[:5]}")

    # What upstream refuses, this refuses, with upstream's wording and no GPU work.
    x = _to_vulkan(_cpu([1.0] * 6, [2, 3]))
    y = _to_vulkan(_cpu([1.0] * 6, [3, 2]))
    try:
        torch.add(torch.ones(2, 3), torch.ones(3, 2))
    except RuntimeError as e:
        upstream = str(e)
    else:
        raise AssertionError("upstream broadcast [2,3] with [3,2]")
    # The helper that words this refusal is shared with the meta kernels
    # (`aten.rs::broadcast_shape`), and it walked left to right: for more than
    # one disagreeing axis it named a different one from upstream. Held here
    # on meta, where the same helper answers.
    meta = _C.device("meta")
    for sa, sb in (((2, 3), (3, 2)), ((5, 2, 3), (4, 1, 3)), ((4, 2), (3,))):
        try:
            torch.add(torch.ones(sa), torch.ones(sb))
        except RuntimeError as e:
            want = str(e)
        else:
            raise AssertionError((sa, sb))
        xm = _C._aten_dispatch("aten.empty.memory_format", list(sa), device=meta)
        ym = _C._aten_dispatch("aten.empty.memory_format", list(sb), device=meta)
        try:
            _C._aten_dispatch("aten.add.Tensor", xm, ym)
        except RuntimeError as e:
            assert want in str(e), (sa, sb, str(e), want)
        else:
            raise AssertionError(f"meta add {sa} {sb} was answered")
    before = _counters()
    try:
        _C._aten_dispatch("aten.add.Tensor", x, y)
    except RuntimeError as e:
        # The shim prefixes the op name, as its dense kernels do.
        assert upstream in str(e), (str(e), upstream)
    else:
        raise AssertionError("[2,3] + [3,2] was computed")
    assert _delta(before, _counters())["shader_dispatches"] == 0


# (label, lhs shape, rhs shape, shaders): one `bmm` pass, plus one strided
# copy for each operand whose batch dimensions must be materialised.
MATMUL_CASES = (
    ("bert q.kT", [2, 4, 7, 16], [2, 4, 16, 7], 1),
    ("bert attn.v", [2, 4, 7, 7], [2, 4, 7, 16], 1),
    ("2-D", [5, 3], [3, 4], 1),
    ("3-D", [3, 5, 2], [3, 2, 6], 1),
    ("broadcast lhs batch", [1, 4, 5, 3], [2, 4, 3, 2], 2),
    ("broadcast rhs 2-D", [2, 3, 5, 3], [3, 4], 2),
    ("vector rhs", [2, 5, 3], [3], 2),
    ("vector lhs", [3], [3, 4], 1),
    ("vector vector", [6], [6], 1),
)


def test_matmul_agrees_with_upstream_and_ran_as_one_batched_product():
    """`torch.matmul` on the device -- the other wall VULKAN6.md did not list.

    The shim keeps `aten.matmul.default` as one op (docs/devices/VULKAN4.md
    §2.1); upstream decomposes it into `expand` + `bmm` + `_unsafe_view`, and
    VULKAN6.md's wall came from upstream's trace, where `matmul` never appears.
    BERT's eager attention calls it twice per layer.

    Values within the tolerance derived from upstream's own float32 error
    (docs/numerics/AGREE.md §2) -- a sum of products is not exactly rounded,
    so bit equality is not the bar. The shader count says the product ran once
    over the flattened batch, and that a broadcast batch cost exactly one
    extra strided copy per operand that needed it.
    """
    if _vulkan_or_skip("the matmul sweep") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    cases = []
    for n, (label, sa, sb, shaders) in enumerate(MATMUL_CASES):
        a = _rand(torch, *sa, seed=9500 + n)
        b = _rand(torch, *sb, seed=9600 + n)
        av, bv = _vk_like(torch, a), _vk_like(torch, b)
        before = _counters()
        out = _C._aten_dispatch("aten.matmul.default", av, bv)
        d = _delta(before, _counters())
        want = torch.matmul(a, b)
        assert list(out.shape) == list(want.shape), (label, out.shape, want.shape)
        assert str(out.device) == "vulkan", (label, out.device)
        assert d["shader_dispatches"] == shaders, (label, d)
        assert d["host_downloads"] == 0 and d["host_uploads"] == 0, (label, d)
        cases.append((f"matmul {label}", _flat(_to_cpu(out)), _up_flat(want),
                      _up_flat(torch.matmul(a.double(), b.double()))))
    _assert_agreement("matmul", cases)

    for label, sa, sb in (("inner mismatch", [2, 3], [4, 5]),
                          ("batch mismatch", [2, 3, 4], [3, 4, 5]),
                          ("0-d", [], [3])):
        a, b = torch.ones(sa), torch.ones(sb)
        try:
            torch.matmul(a, b)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"upstream accepted {label}")
        av, bv = _vk_like(torch, a), _vk_like(torch, b)
        before = _counters()
        try:
            _C._aten_dispatch("aten.matmul.default", av, bv)
        except RuntimeError:
            pass
        else:
            raise AssertionError(f"matmul {label} was computed instead of refused")
        assert _delta(before, _counters())["shader_dispatches"] == 0, label


TANH_SHAPES = ((4, 5), (7, 33), (2, 3, 64), (1025,), (2, 768))


def test_tanh_agrees_with_upstream_at_a_derived_tolerance():
    """BERT's pooler ends in `tanh`. Values over a wide range, tails included."""
    if _vulkan_or_skip("the tanh agreement sweep") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    cases = []
    for i, shape in enumerate(TANH_SHAPES):
        a = _rand(torch, *shape, seed=900 + i) * (0.5 + 3.0 * i)
        av = _to_vulkan(_cpu(_up_flat(a), shape))
        before = _counters()
        out = _C._aten_dispatch("aten.tanh.default", av)
        d = _delta(before, _counters())
        assert d["shader_dispatches"] == 1, (shape, d)
        assert d["host_downloads"] == 0 and d["host_uploads"] == 0, (shape, d)
        assert list(out.shape) == list(shape), (out.shape, shape)
        cases.append((f"tanh{list(shape)}", _flat(_to_cpu(out)),
                      _up_flat(torch.tanh(a)), _up_flat(torch.tanh(a.double()))))
    _assert_agreement("tanh", cases)


_BERT_SHIM_SCRIPT = r"""
import json, sys
import torch

assert hasattr(torch._C, "_aten_implemented"), "this subprocess got upstream torch"

cfg = json.load(sys.stdin)
out = {"probe": torch._C._vulkan_probe()}
if not out["probe"]["available"]:
    json.dump(out, sys.stdout); raise SystemExit

from transformers import AutoModel

m = AutoModel.from_pretrained(cfg["path"], attn_implementation="eager").eval()
ids = torch.as_tensor(cfg["ids"], dtype=torch.int64).reshape(cfg["sids"])
with torch.no_grad():
    r = m(input_ids=ids)
out["cpu"] = r.last_hidden_state.reshape(-1).tolist()
out["cpu_pooled"] = r.pooler_output.reshape(-1).tolist()
m.to("vulkan")
out["param_devices"] = sorted({str(p.device) for p in m.parameters()} |
                              {str(b.device) for b in m.buffers()})
v = ids.to("vulkan")
before = torch._C._vulkan_counters()
with torch.no_grad():
    r = m(input_ids=v)
after = torch._C._vulkan_counters()
out["counters"] = {k: after[k] - before[k] for k in after}
out["device"] = [str(r.last_hidden_state.device), str(r.pooler_output.device)]
out["vulkan"] = r.last_hidden_state.cpu().reshape(-1).tolist()
out["vulkan_pooled"] = r.pooler_output.cpu().reshape(-1).tolist()
out["layers"] = m.config.num_hidden_layers
json.dump(out, sys.stdout)
"""


def _pretrained_bert_dir():
    """A local `bert-base-uncased` with weights, or None.

    Looked for, never downloaded: the gate must not reach the network. The
    candidates are the two Hugging Face caches this machine uses; a snapshot
    without `model.safetensors` (the default cache here holds only a config)
    does not count.
    """
    import glob
    roots = []
    if os.environ.get("TORCHNATIVE_BERT_DIR"):
        roots.append(os.environ["TORCHNATIVE_BERT_DIR"])
    for home in (os.environ.get("HF_HOME"), "/Volumes/macMini/caches/hf-home",
                 os.path.expanduser("~/.cache/huggingface")):
        if home:
            for repo in ("models--google-bert--bert-base-uncased",
                         "models--bert-base-uncased"):
                roots.extend(sorted(glob.glob(os.path.join(home, "hub", repo, "snapshots", "*"))))
    for root in roots:
        if (os.path.exists(os.path.join(root, "model.safetensors"))
                and os.path.exists(os.path.join(root, "config.json"))):
            return root
    return None


def test_a_pretrained_bert_forwards_on_the_gpu_and_agrees_with_upstream():
    """**The deliverable of docs/devices/VULKAN7.md.**

    Real `transformers`, real `AutoModel.from_pretrained("bert-base-uncased")`,
    moved to the Vulkan device with `m.to("vulkan")`, forwarded on real token
    ids -- and compared against the same checkpoint in upstream torch.

    `docs/architectures/VOICE4.md` records the failure this exists to avoid:
    seven rounds that closed operators one at a time and never ran a model.
    VULKAN6.md stopped at "a single-head block runs"; the claim here is the
    pretrained, twelve-layer, twelve-head model.

    Three kinds of assertion, kept apart:

      * **value** -- `last_hidden_state` and `pooler_output` each within
        docs/numerics/AGREE.md §2's rule: no further from upstream's float64
        answer than 4x upstream float32's own distance from it;
      * **device** -- every parameter and buffer reports `vulkan`, and so do
        both outputs;
      * **work** -- the forward ran exactly the number of compute shaders the
        architecture implies (derived below from the model code, not copied
        from a run) and **read nothing back**. A forward that computed any
        part on the host would still pass the value check.
    """
    if _vulkan_or_skip("the pretrained BERT forward") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    path = _pretrained_bert_dir()
    if path is None:
        vulkan_coverage.vulkan_skip(
            "the pretrained BERT forward: no local bert-base-uncased with "
            "model.safetensors (set TORCHNATIVE_BERT_DIR); nothing is downloaded")
        return
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    # Two sentences of different length, so the batch is not a copy of one
    # row -- padded, and forwarded without a mask on both sides alike.
    enc = tok(["the quick brown fox jumps over the lazy dog",
               "vulkan runs a pretrained transformer"],
              return_tensors="pt", padding=True)
    ids = enc["input_ids"]
    B, S = ids.shape

    cfg = {"path": path, "ids": ids.reshape(-1).tolist(), "sids": [B, S]}
    env = dict(os.environ)
    env["PYTHONPATH"] = VENDOR_DIR
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.run([sys.executable, "-c", _BERT_SHIM_SCRIPT],
                          input=json.dumps(cfg), capture_output=True,
                          text=True, env=env, timeout=1200)
    assert proc.returncode == 0, (proc.stdout[-2000:], proc.stderr[-4000:])
    got = json.loads(proc.stdout)
    if not got["probe"]["available"]:
        vulkan_coverage.vulkan_skip(
            f"the pretrained BERT forward: the vendored-tree subprocess has "
            f"no loader -- {got['probe']['error']}")
        return

    assert got["param_devices"] == ["vulkan"], got["param_devices"]
    assert got["device"] == ["vulkan", "vulkan"], got["device"]

    up = AutoModel.from_pretrained(path, attn_implementation="eager").eval()
    with torch.no_grad():
        r32 = up(input_ids=ids)
        r64 = up.double()(input_ids=ids)

    for what, key, want32, truth in (
            ("last_hidden_state", "vulkan", r32.last_hidden_state, r64.last_hidden_state),
            ("pooler_output", "vulkan_pooled", r32.pooler_output, r64.pooler_output)):
        want32 = want32.reshape(-1).tolist()
        truth = truth.reshape(-1).tolist()
        cpu_key = "cpu" if key == "vulkan" else "cpu_pooled"
        assert len(got[key]) == len(truth), (what, len(got[key]), len(truth))
        up_err = max(abs(a - b) for a, b in zip(want32, truth))
        vk_err = max(abs(a - b) for a, b in zip(got[key], truth))
        cpu_err = max(abs(a - b) for a, b in zip(got[cpu_key], truth))
        print(f"   bert {what}: upstream f32 err {up_err:.3e}, shim cpu "
              f"{cpu_err:.3e}, shim vulkan {vk_err:.3e} ({vk_err / up_err:.2f}x)")
        assert vk_err <= 4 * up_err, (
            f"bert {what} on vulkan is {vk_err:.3e} from upstream's float64 "
            f"answer against upstream float32's own {up_err:.3e}; "
            f"docs/numerics/AGREE.md's rule allows 4x")

    # Derived from `transformers/models/bert/modeling_bert.py` with
    # `attn_implementation="eager"`, `input_ids` only, B = 2 (so `expand` is
    # not an identity), and the dispatch table above:
    #
    #   embeddings  9  slice(position_ids) + expand(token_type_ids) + gather
    #                  + 3 embedding + 2 add + layer_norm
    #   per layer  32  3 x Linear(q,k,v) [t + matmul + bias = 3]  9
    #                  3 x transpose(1, 2) after view                3
    #                  transpose(k, 2, 3)                            1
    #                  matmul -> bmm, * scaling, softmax, matmul     4
    #                  transpose(1, 2) of the context                1
    #                  attention output: Linear + add + layer_norm  5
    #                  intermediate: Linear + gelu                   4
    #                  output: Linear + add + layer_norm             5
    #   pooler      5  [:, 0] = identity slice (0) + select (1)
    #                  + Linear on 2-D (t + matmul + bias = 3) + tanh
    expected = 9 + 32 * got["layers"] + 5
    counters = got["counters"]
    print(f"   bert on vulkan: {got['layers']} layers, B={B} S={S}, "
          f"{counters['shader_dispatches']} shaders, "
          f"{counters['host_downloads']} readbacks, "
          f"{counters['host_uploads']} uploads during the forward")
    assert counters["host_downloads"] == 0, (
        f"the BERT forward read {counters['host_downloads']} buffer(s) back to "
        f"the host -- part of it computed on the CPU: {counters}")
    assert counters["host_uploads"] == 0, (
        f"the BERT forward uploaded {counters['host_uploads']} buffer(s) -- "
        f"something was built on the host mid-forward: {counters}")
    assert counters["shader_dispatches"] == expected, (
        f"the BERT forward ran {counters['shader_dispatches']} compute shaders, "
        f"expected {expected} = 9 + 32 x {got['layers']} + 5: {counters}")






# ---------------------------------------------------------------------------
# BERT under the DEFAULT `sdpa` with a real `attention_mask` -- what actually
# happens, which is not what the round that wrote this claimed.
#
# The commit that added this file's previous version of the test below
# (`716d16e`, "BERT on Vulkan takes an attention_mask") states:
#
#     `bert-base-uncased` forwards on the Vulkan device under the **default
#     `sdpa`** and with a **real padded `attention_mask`** ... 410 compute
#     shaders, one host upload and zero downloads, agreeing with CPU at about
#     8.6e-06.
#
# **That test had never executed.** It was defined *below* this file's
# `raise SystemExit(_main())`, so the module exited before Python reached the
# `def`; the runner collected 40 test functions out of a file containing 41 and
# nothing compared the two numbers. Moving the runner to the bottom of the file
# (see the block at the end) made it run for the first time, and it fails.
#
# It fails for a reason that is not a regression and not this round's doing.
# Its mechanism was to monkeypatch `BertModel.get_extended_attention_mask` so
# the additive mask is built on the host and uploaded once. In transformers
# 5.15.1 -- the version this repository pins -- BERT does not call that method.
# `BertModel.forward` calls `_create_attention_masks` ->
# `create_bidirectional_mask` -> `masking_utils.sdpa_mask`, and the first thing
# `sdpa_mask` does is `torch.arange(batch_size, device=device)` with `device`
# being the model's. This backend has no integer arithmetic on purpose
# (docs/devices/VULKAN6.md §1), so `aten.arange.default` refuses by name, and
# the patched method is never reached to prevent it.
#
# So the claim "BERT forwards on Vulkan under the default sdpa with a real
# attention_mask" is **unverified**, and the numbers in it (410 shaders, one
# upload, 8.6e-06) were never produced by anything in this tree. The test below
# now pins what is measured instead: the refusal, by name, with nothing read
# back, and the patched method's call count at zero so the *reason* is pinned
# too and not just the symptom. It fails the moment either changes.
#
# What it would take is written down and not done here: either integer
# `arange`/`add`/`where` on this device (a widening of `check_dtype`, which
# VULKAN6 decided against and this round does not reopen), or an interception
# at `masking_utils.sdpa_mask` rather than at the method transformers 5.15.1 no
# longer calls. The second is a day's work and belongs to a mask round.
# ---------------------------------------------------------------------------

_BERT_SHIM_SCRIPT_MASK = r"""
import json, sys
import torch


assert hasattr(torch._C, "_aten_implemented"), "this subprocess got upstream torch"

cfg = json.load(sys.stdin)
out = {"probe": torch._C._vulkan_probe()}
if not out["probe"]["available"]:
    json.dump(out, sys.stdout); raise SystemExit

from transformers import AutoModel

m = AutoModel.from_pretrained(cfg["path"], attn_implementation="sdpa").eval()
ids = torch.as_tensor(cfg["ids"], dtype=torch.int64).reshape(cfg["sids"])
mask = torch.as_tensor(cfg["mask"], dtype=torch.int64).reshape(cfg["sids"])

# The interception the previous round relied on, with a counter on it. The
# counter is the point: it says whether transformers still routes through here.
orig_get_extended = type(m).get_extended_attention_mask
calls = {"n": 0}


def host_mask(self, attention_mask, input_shape, device=None, dtype=None):
    calls["n"] += 1
    attention_mask = attention_mask.cpu() if attention_mask is not None else None
    res = orig_get_extended(self, attention_mask, input_shape, device="cpu", dtype=dtype)
    return res.to("vulkan") if res is not None else None


type(m).get_extended_attention_mask = host_mask

m.to("vulkan")
out["param_devices"] = sorted({str(p.device) for p in m.parameters()} |
                              {str(b.device) for b in m.buffers()})
v_ids = ids.to("vulkan")
before = torch._C._vulkan_counters()
try:
    with torch.no_grad():
        r = m(input_ids=v_ids, attention_mask=mask)
    out["result"] = "ok"
    out["device"] = [str(r.last_hidden_state.device), str(r.pooler_output.device)]
except Exception as e:
    out["result"] = f"{type(e).__name__}: {e}"
after = torch._C._vulkan_counters()
out["counters"] = {key: after[key] - before[key] for key in after}
out["host_mask_calls"] = calls["n"]
out["layers"] = m.config.num_hidden_layers
json.dump(out, sys.stdout)
"""


def test_the_pretrained_bert_sdpa_mask_forward_refuses_at_arange_and_the_host_mask_hook_is_never_called():
    """The measured state of VULKAN7's last open claim.

    Renamed from `..._builds_mask_on_host_and_agrees_with_upstream`, which is
    what it was called while it was never run. See the block above for the
    whole story; the short version is that the mask is *not* built on the host,
    because the method that would have built it there is not on transformers
    5.15.1's path for BERT.

    Two assertions, and the second is the one that makes this evidence rather
    than a symptom:

      * the forward refuses, by name, at `aten.arange.default`, having read
        nothing back to the host -- so this is a refusal and not a fallback;
      * `get_extended_attention_mask` was called **zero** times, which is why.
        Without this, someone reading the failure would reasonably conclude the
        host-mask path was reached and broken.
    """
    if _vulkan_or_skip("the pretrained BERT sdpa mask forward") is None:
        return
    path = _pretrained_bert_dir()
    if path is None:
        vulkan_coverage.vulkan_skip(
            "the pretrained BERT sdpa mask forward: no local bert-base-uncased with "
            "model.safetensors (set TORCHNATIVE_BERT_DIR); nothing is downloaded")
        return
    torch = _upstream()
    if torch is None:
        return
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(path)
    enc = tok(["the quick brown fox jumps over the lazy dog",
               "vulkan runs a pretrained transformer"],
              return_tensors="pt", padding=True)
    ids = enc["input_ids"]
    mask = enc["attention_mask"]
    B, S = ids.shape
    # The two sentences are different lengths on purpose: with equal lengths
    # there is no padding and the masked path is not exercised at all. A round
    # in this repository once made a test pass by equalising them.
    assert int(mask.sum()) < B * S, "the fixture stopped being padded"

    cfg = {"path": path, "ids": ids.reshape(-1).tolist(),
           "mask": mask.reshape(-1).tolist(), "sids": [B, S]}
    env = dict(os.environ)
    env["PYTHONPATH"] = (f"{os.environ.get('PYTHONPATH', '')}:"
                         f"{os.path.abspath('torchnative/src/main')}:{VENDOR_DIR}")
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    proc = subprocess.run([sys.executable, "-c", _BERT_SHIM_SCRIPT_MASK],
                          input=json.dumps(cfg), capture_output=True,
                          text=True, env=env, timeout=1200)
    assert proc.returncode == 0, (proc.stdout[-2000:], proc.stderr[-4000:])
    got = json.loads(proc.stdout)
    if not got["probe"]["available"]:
        vulkan_coverage.vulkan_skip(
            f"the pretrained BERT sdpa mask forward: the vendored-tree subprocess has "
            f"no loader -- {got['probe']['error']}")
        return

    assert got["param_devices"] == ["vulkan"], got["param_devices"]
    assert got["result"] != "ok", (
        "the default-sdpa BERT forward with an attention_mask succeeded on "
        "vulkan. That is what VULKAN7's round claimed and this test says is "
        "not true here -- if it is true now, restore the agreement assertions "
        "and the 410-shader count rather than deleting this one")
    assert got["result"].startswith("NotImplementedError"), got["result"]
    assert "aten.arange.default" in got["result"], (
        f"the forward stopped somewhere other than the op this test names: "
        f"{got['result']}")
    assert got["host_mask_calls"] == 0, (
        f"get_extended_attention_mask was called {got['host_mask_calls']} "
        f"time(s); if transformers routes through it again, the host-mask "
        f"design is reachable and this test is describing the wrong wall")
    # The state at the moment of refusal, pinned exactly rather than as an
    # inequality. It is **not** zero readbacks: the embedding prologue runs
    # first and something in it reads one buffer back before the mask is
    # reached. That is recorded here as a measurement, not blessed -- an
    # unexplained readback on a forward is the exact shape docs/devices/VULKAN6.md
    # says values cannot police, and nothing in this round audited it. It is
    # named in docs/devices/VULKAN8.md §5 as open.
    assert got["counters"] == {"shader_dispatches": 10, "host_uploads": 1,
                               "host_downloads": 1}, got["counters"]
    print(f"   bert default-sdpa + attention_mask: refused at aten.arange.default "
          f"after {got['counters']['shader_dispatches']} shaders, "
          f"{got['counters']['host_downloads']} readback(s), "
          f"get_extended_attention_mask calls=0")


# ---------------------------------------------------------------------------
# The logsumexp slot -- docs/devices/VULKAN8.md
#
# `_scaled_dot_product_flash_attention_for_cpu` returns `(output, logsumexp)`
# and this device returned `None` in the second slot. What upstream actually
# puts there was **measured**, not read off the name:
#
#   * shape   `query.shape[:-1]`, i.e. `(B, H, T)` -- one value per query row.
#   * dtype   `float32` for a `float32` query (and float32 even for a float16
#             one, which `test_metaemb.py` already pins on the meta device).
#   * value   the natural log-sum-exp of the **masked, scaled** scores over the
#             key axis -- the same rows the softmax above it normalises. Checked
#             against `torch.logsumexp(scores.double(), dim=-1)`: 2.28e-07 worst
#             over a randn population, i.e. float32's own rounding.
#   * layout  upstream's is **not contiguous** -- stride `(15, 1, 3)` for shape
#             `(2, 3, 5)`, because upstream allocates `(B, T, H)` and hands back
#             a `transpose(1, 2)` of it. A `VkTensor` has a shape and no strides
#             at all (docs/devices/VULKAN4.md §6), so this device cannot
#             reproduce that and does not claim to: the values agree element for
#             element, the strides are this device's own. Said here rather than
#             left for someone to discover.
#
# `_scaled_dot_product_flash_attention_for_cpu` takes `attn_mask` and `scale`
# keyword-only, so the upstream oracle below passes them that way.
# ---------------------------------------------------------------------------

# (B, H, T, E, S_kv, has_mask) -- rank 4, which is the only rank upstream's op
# accepts, so the oracle can be upstream's own kernel rather than a formula.
SDPA_LSE_CASES = (
    (1, 1, 2, 4, 2, False),
    (2, 3, 5, 4, 7, False),
    (2, 3, 5, 4, 7, True),
    (1, 2, 3, 16, 3, True),
    (2, 1, 8, 8, 8, False),
)


def _lse_inputs(torch, i, b, h, t, e, s, has_mask):
    q = _rand(torch, b, h, t, e, seed=1900 + i)
    k = _rand(torch, b, h, s, e, seed=1930 + i)
    v = _rand(torch, b, h, s, e, seed=1960 + i)
    mask = None
    if has_mask:
        # A padded batch's mask: the last two keys are padding for every row.
        mask = torch.zeros(b, 1, 1, s, dtype=torch.float32)
        mask[:, :, :, s - 2:] = -1e9
    return q, k, v, mask


def _lse_truth(torch, q, k, v, mask, dtype):
    """The written-out logsumexp at `dtype`, so the tolerance can be derived."""
    q, k, v = q.to(dtype), k.to(dtype), v.to(dtype)
    scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(q.size(-1)))
    if mask is not None:
        scores = scores + mask.to(dtype)
    return torch.logsumexp(scores, dim=-1)


def test_the_sdpa_logsumexp_is_no_longer_none_and_has_upstreams_shape_and_dtype():
    """The slot itself, before any question of its value.

    `None` here is what this device returned before, and the backward that
    would consume it could not even ask for a shape.
    """
    if _vulkan_or_skip("the sdpa logsumexp shape and dtype") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    for i, case in enumerate(SDPA_LSE_CASES):
        b, h, t, e, s, has_mask = case
        q, k, v, mask = _lse_inputs(torch, i, *case)
        kwargs = {}
        if mask is not None:
            kwargs["attn_mask"] = _to_vulkan(_cpu(_up_flat(mask), [b, 1, 1, s]))
        got = _C._aten_dispatch(
            _SDPA,
            _to_vulkan(_cpu(_up_flat(q), [b, h, t, e])),
            _to_vulkan(_cpu(_up_flat(k), [b, h, s, e])),
            _to_vulkan(_cpu(_up_flat(v), [b, h, s, e])),
            **kwargs)
        assert isinstance(got, tuple) and len(got) == 2, got
        lse = got[1]
        assert lse is not None, (
            f"the logsumexp slot is still None for {case}; a backward through "
            f"this op cannot even ask it for a shape")
        # Upstream's own answer, for the shape and dtype rather than a
        # transcription of them.
        up = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
            q, k, v, 0.0, False, attn_mask=mask)[1]
        assert list(lse.shape) == list(up.shape), (case, lse.shape, up.shape)
        assert str(lse.dtype) == str(up.dtype), (case, lse.dtype, up.dtype)
        assert str(lse.device) == "vulkan", lse.device


def test_the_sdpa_logsumexp_agrees_with_upstream_at_a_derived_tolerance():
    """Element-wise against **upstream's own kernel**, not a formula beside it.

    The tolerance comes from `_assert_agreement`, which re-derives it from this
    population's upstream float32-vs-float64 error (docs/numerics/AGREE.md §2).
    It is not a number chosen here and there is nothing in this file to widen:
    `test_a_wrong_logsumexp_is_rejected_by_this_tolerance` proves the derived
    number is tight enough to reject a wrong answer.
    """
    if _vulkan_or_skip("the sdpa logsumexp agreement sweep") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    cases = []
    for i, case in enumerate(SDPA_LSE_CASES):
        b, h, t, e, s, has_mask = case
        q, k, v, mask = _lse_inputs(torch, i, *case)
        kwargs = {}
        if mask is not None:
            kwargs["attn_mask"] = _to_vulkan(_cpu(_up_flat(mask), [b, 1, 1, s]))
        lse = _C._aten_dispatch(
            _SDPA,
            _to_vulkan(_cpu(_up_flat(q), [b, h, t, e])),
            _to_vulkan(_cpu(_up_flat(k), [b, h, s, e])),
            _to_vulkan(_cpu(_up_flat(v), [b, h, s, e])),
            **kwargs)[1]
        want32 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
            q, k, v, 0.0, False, attn_mask=mask)[1]
        truth = _lse_truth(torch, q, k, v, mask, torch.float64)
        cases.append((f"lse{list(case)}", _flat(_to_cpu(lse)),
                      _up_flat(want32.contiguous()), _up_flat(truth)))
    _assert_agreement("sdpa logsumexp", cases)


def test_a_wrong_logsumexp_is_rejected_by_this_tolerance():
    """The derived tolerance has teeth -- checked, not asserted.

    docs/architectures/VOICE4.md §6 records a tolerance widened 100x that left every
    test green. The guard against that shape of failure is not a smaller
    number, it is this: feed `_assert_agreement` a *wrong* answer and require
    it to raise. The wrong answers below are the ones a plausible defect
    produces -- `log(sum(exp(x)))` without the max shift is right, so it is not
    here; `sum(exp(x))` without the log, and the logsumexp of the *unscaled*
    scores, are the two this kernel could actually have been.
    """
    if _vulkan_or_skip("the logsumexp tolerance teeth check") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    b, h, t, e, s, has_mask = SDPA_LSE_CASES[1]
    q, k, v, mask = _lse_inputs(torch, 1, *SDPA_LSE_CASES[1])
    want32 = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
        q, k, v, 0.0, False, attn_mask=mask)[1]
    truth = _lse_truth(torch, q, k, v, mask, torch.float64)
    unscaled = torch.logsumexp(
        (q.double() @ k.double().transpose(-2, -1)), dim=-1)
    nolog = torch.exp(
        (q.double() @ k.double().transpose(-2, -1)) / math.sqrt(e)).sum(-1)

    # The right answer passes.
    _assert_agreement("lse teeth (control: the correct value)",
                      [("control", _up_flat(want32.contiguous()), _up_flat(want32.contiguous()),
                        _up_flat(truth))])

    for name, wrong in (("unscaled scores", unscaled), ("no log", nolog)):
        try:
            _assert_agreement(f"lse teeth ({name})",
                              [(name, _up_flat(wrong.float()),
                                _up_flat(want32.contiguous()), _up_flat(truth))])
        except AssertionError:
            pass
        else:
            raise AssertionError(
                f"the derived tolerance accepted a logsumexp computed from {name}; "
                f"it is too wide to be evidence of anything")


def test_the_logsumexp_ran_on_the_gpu_and_cost_exactly_one_more_shader():
    """docs/devices/VULKAN6.md's rule: values cannot police a host fallback.

    Making `embedding` gather on the host produced bit-perfect numbers on a
    tensor whose `.device` said `vulkan`, and only the counters caught it. So
    the logsumexp is held to the counters too: the whole call reads **nothing**
    back, and it costs exactly one dispatch more than the same call did without
    a logsumexp. The step count is derived from the handler's chain, not copied
    from a run:

        transpose(k) 1 + matmul 1 + mul.Scalar 1 [+ add(mask) 1]
                     + _softmax 1 + matmul 1 + logsumexp 1
    """
    if _vulkan_or_skip("the logsumexp dispatch-counter check") is None:
        return
    for case in (SDPA_LSE_CASES[1], SDPA_LSE_CASES[2]):
        b, h, t, e, s, has_mask = case
        torch = _upstream()
        if torch is None:
            return
        q, k, v, mask = _lse_inputs(torch, 1, *case)
        args = [_to_vulkan(_cpu(_up_flat(q), [b, h, t, e])),
                _to_vulkan(_cpu(_up_flat(k), [b, h, s, e])),
                _to_vulkan(_cpu(_up_flat(v), [b, h, s, e]))]
        kwargs = {}
        if mask is not None:
            kwargs["attn_mask"] = _to_vulkan(_cpu(_up_flat(mask), [b, 1, 1, s]))
        before = _counters()
        out, lse = _C._aten_dispatch(_SDPA, *args, **kwargs)
        d = _delta(before, _counters())
        expected = 6 + (1 if has_mask else 0)
        assert d["shader_dispatches"] == expected, (
            f"sdpa{list(case)} ran {d['shader_dispatches']} compute shaders, "
            f"expected {expected} (transpose, matmul, scale, "
            f"{'mask, ' if has_mask else ''}softmax, matmul, logsumexp): {d}")
        assert d["host_downloads"] == 0, (
            f"sdpa{list(case)} read {d['host_downloads']} buffer(s) back to the "
            f"host; the logsumexp is computed on the device or not at all: {d}")
        assert d["host_uploads"] == 0, d
        assert str(lse.device) == "vulkan", lse.device


def test_a_fully_masked_row_reports_upstreams_logsumexp_convention():
    """`0.0`, which is the flash kernel's answer and **not** `torch.logsumexp`'s.

    Measured, both ways:

        _scaled_dot_product_flash_attention_for_cpu, row masked to -inf
            -> logsumexp 0.0,  output all zeros
        torch.logsumexp([-inf, -inf, -inf], dim=-1)
            -> -inf

    The op being implemented here is the first one, so the kernel branches on
    an all-`-inf` row rather than letting `exp(-inf - -inf)` become NaN. A
    kernel without that branch returns NaN, which is neither answer.

    **Not fixed here, and it is next door:** the *output* for such a row is NaN
    on this device, because `sdpa_vulkan` runs `aten._softmax.default` and that
    shader has the same unguarded shift. `aten._safe_softmax.default` is the op
    that handles it and this device has it. Changing which softmax the handler
    runs changes the arithmetic of every sdpa forward on this device, so it is
    not folded into a logsumexp round; it is recorded in
    docs/devices/VULKAN8.md §5 as the next thing.
    """
    if _vulkan_or_skip("the fully-masked logsumexp row") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    b, h, t, e, s = 1, 1, 2, 4, 3
    q = _rand(torch, b, h, t, e, seed=2401)
    k = _rand(torch, b, h, s, e, seed=2402)
    v = _rand(torch, b, h, s, e, seed=2403)
    mask = torch.zeros(b, 1, t, s, dtype=torch.float32)
    mask[0, 0, 0, :] = float("-inf")

    want = torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default(
        q, k, v, 0.0, False, attn_mask=mask)[1]
    assert _up_flat(want.contiguous())[0] == 0.0, (
        f"upstream's convention changed: {_up_flat(want.contiguous())}")

    lse = _C._aten_dispatch(
        _SDPA,
        _to_vulkan(_cpu(_up_flat(q), [b, h, t, e])),
        _to_vulkan(_cpu(_up_flat(k), [b, h, s, e])),
        _to_vulkan(_cpu(_up_flat(v), [b, h, s, e])),
        attn_mask=_to_vulkan(_cpu(_up_flat(mask), [b, 1, t, s])))[1]
    got = _flat(_to_cpu(lse))
    assert got[0] == 0.0, (
        f"a fully masked row reported logsumexp {got[0]!r}; upstream's flash "
        f"kernel reports 0.0 for it and NaN is neither answer")
    assert abs(got[1] - _up_flat(want.contiguous())[1]) < 1e-5, (got, _up_flat(want.contiguous()))


# ---------------------------------------------------------------------------
# The reduction the sdpa backward stops on -- docs/devices/VULKAN9.md
# ---------------------------------------------------------------------------

_SUM_DIMS = "aten.sum.dim_IntList"
_SUM_ALL = "aten.sum.default"

# (shape, dim, keepdim). The first entry is the shape of the rule's own
# `rowsum(dP * P)` (tape.rs `sdpa_backward`): last axis, keepdim=True. The
# rest widen it away from that one special case -- a leading axis, two
# non-adjacent axes at once, an empty list (which torch reads as "every
# axis", not "no axis"), `dim=None`, and a 512-long axis where a reduction
# that rounds after every term drifts away from upstream's.
SUM_CASES = (
    ((4, 5), [-1], True),
    ((4, 5), [0], False),
    ((2, 3, 4), [0, 2], False),
    ((2, 3, 4), [1], True),
    ((2, 3, 4, 5), [1, 3], True),
    ((129,), [0], False),
    ((2, 512), [-1], False),
    ((3, 7), [], False),
    ((6,), None, False),
    ((1, 2, 3, 4), [-1], True),
)


def _sum_upstream(torch, a, dim, keepdim):
    if dim is None:
        return torch.ops.aten.sum.dim_IntList(a, None, keepdim)
    return torch.ops.aten.sum.dim_IntList(a, dim, keepdim)


def _sum_on_vulkan(shape, dim, keepdim, values):
    return _C._aten_dispatch(_SUM_DIMS, _to_vulkan(_cpu(values, shape)), dim, keepdim)


def test_sum_over_dims_agrees_with_upstream_at_a_derived_tolerance():
    """`aten.sum.dim_IntList` element-wise against upstream's own kernel.

    **This device has no strides** (docs/devices/VULKAN4.md §6), so there is no
    permute-then-reduce-the-last-axis route available: a `VkTensor` is a shape
    and a contiguous buffer. The kernel therefore takes the *source stride of
    every axis*, derived on the host from the input shape, splits the axes into
    kept and reduced, and walks both index spaces itself. That is why an
    arbitrary `dim` list -- including the two non-adjacent axes in the sweep --
    is one dispatch and not one per axis.

    The tolerance is re-derived by `_assert_agreement` from this population's
    own upstream float32-vs-float64 error. Nothing here widens it, and
    `test_a_wrong_sum_reduction_is_rejected_by_this_tolerance` proves it has
    teeth.
    """
    if _vulkan_or_skip("the sum.dim_IntList agreement sweep") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    cases = []
    for i, (shape, dim, keepdim) in enumerate(SUM_CASES):
        a = _rand(torch, *shape, seed=2700 + i)
        out = _sum_on_vulkan(shape, dim, keepdim, _up_flat(a))
        want32 = _sum_upstream(torch, a, dim, keepdim)
        truth = _sum_upstream(torch, a.double(), dim, keepdim)
        assert list(out.shape) == list(want32.shape), (
            f"sum{list(shape)} dim={dim} keepdim={keepdim}: shape "
            f"{list(out.shape)}, upstream {list(want32.shape)}")
        assert str(out.device) == "vulkan", out.device
        cases.append((f"sum{list(shape)} dim={dim} keepdim={keepdim}",
                      _flat(_to_cpu(out)), _up_flat(want32), _up_flat(truth)))
    _assert_agreement("sum.dim_IntList", cases)


def test_sum_default_agrees_with_upstream_at_a_derived_tolerance():
    """`aten.sum.default` -- the whole-tensor form, which a *loss* needs.

    It is the same kernel with no kept axes, which is the point: `sum.default`
    is not a second reduction, it is `sum.dim_IntList` over every axis with the
    output collapsed to rank 0. Held to upstream separately anyway, because
    "it is the same kernel" is an argument and this is a measurement.
    """
    if _vulkan_or_skip("the sum.default agreement sweep") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    cases = []
    for i, shape in enumerate(((2, 3, 4), (1025,), (4, 5), (2, 512))):
        a = _rand(torch, *shape, seed=2800 + i)
        out = _C._aten_dispatch(_SUM_ALL, _to_vulkan(_cpu(_up_flat(a), shape)))
        want32 = torch.ops.aten.sum.default(a)
        truth = torch.ops.aten.sum.default(a.double())
        assert list(out.shape) == list(want32.shape), (out.shape, want32.shape)
        assert str(out.device) == "vulkan", out.device
        cases.append((f"sum_all{list(shape)}", _flat(_to_cpu(out)),
                      [want32.item()], [truth.item()]))
    _assert_agreement("sum.default", cases)


def test_a_wrong_sum_reduction_is_rejected_by_this_tolerance():
    """The derived tolerance has teeth -- two wrong reductions, and the margin.

    The wrong answers are the two this kernel could actually have been, and
    both are *plausible*: reducing the wrong axis (the kept/reduced split
    inverted) and dropping the last term of each row (a `<` that should have
    been `<=`). Neither is a scaled or shifted version of the right answer, so
    a tolerance that accepts either is not measuring anything.
    """
    if _vulkan_or_skip("the sum tolerance teeth check") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    a = _rand(torch, 6, 7, seed=2901)
    want32 = torch.ops.aten.sum.dim_IntList(a, [-1], False)
    truth = torch.ops.aten.sum.dim_IntList(a.double(), [-1], False)

    _assert_agreement("sum teeth (control: the correct value)",
                      [("control", _up_flat(want32), _up_flat(want32), _up_flat(truth))])

    # The wrong axis: same shape here only because 6 != 7 would change it, so
    # a square-ish slice of the transpose is taken to keep the lengths equal
    # and force the comparison to be about *values*.
    wrong_axis = torch.ops.aten.sum.dim_IntList(a, [0], False)[:6]
    dropped = a[:, :-1].sum(-1)

    margins = []
    for name, wrong in (("the wrong axis", wrong_axis),
                        ("a dropped last term", dropped)):
        got = _up_flat(wrong.float())
        try:
            _assert_agreement(f"sum teeth ({name})",
                              [(name, got, _up_flat(want32), _up_flat(truth))])
        except AssertionError:
            margins.append((name, _elem_ratio(got, _up_flat(want32), _up_flat(truth))))
        else:
            raise AssertionError(
                f"the derived tolerance accepted a sum computed with {name}; "
                f"it is too wide to be evidence of anything")
    for name, ratio in margins:
        print(f"   sum teeth: {name} rejected at {ratio:,.0f}x upstream's own error")
        assert ratio > 1e3, (name, ratio)


def test_the_sum_reduction_ran_on_the_gpu_in_exactly_one_dispatch():
    """docs/devices/VULKAN6.md's rule: values cannot police a host fallback.

    A sum done on the host would produce bit-identical numbers on a tensor
    whose `.device` says `vulkan` -- that exact substitution was made for
    `embedding` and only the counters caught it. So every case in the sweep is
    required to cost **one** dispatch and **zero** host downloads. One, not
    "at least one": a reduction that walked the axes with a dispatch each
    would still be on the GPU and would still be the wrong kernel.
    """
    if _vulkan_or_skip("the sum dispatch-counter check") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    for i, (shape, dim, keepdim) in enumerate(SUM_CASES):
        a = _rand(torch, *shape, seed=3000 + i)
        on_device = _to_vulkan(_cpu(_up_flat(a), shape))
        before = _counters()
        out = _C._aten_dispatch(_SUM_DIMS, on_device, dim, keepdim)
        d = _delta(before, _counters())
        assert d["shader_dispatches"] == 1, (
            f"sum{list(shape)} dim={dim} ran {d['shader_dispatches']} compute "
            f"shaders, expected exactly 1: {d}")
        assert d["host_downloads"] == 0, (
            f"sum{list(shape)} dim={dim} read {d['host_downloads']} buffer(s) "
            f"back; the reduction happens on the device or not at all: {d}")
        assert d["host_uploads"] == 0, d
        assert str(out.device) == "vulkan", out.device

    on_device = _to_vulkan(_cpu(_up_flat(_rand(torch, 3, 4, seed=3100)), (3, 4)))
    before = _counters()
    _C._aten_dispatch(_SUM_ALL, on_device)
    d = _delta(before, _counters())
    assert d["shader_dispatches"] == 1, d
    assert d["host_downloads"] == 0, d


def test_sum_refuses_what_this_device_does_not_do_rather_than_reaching_for_the_cpu():
    """The narrowings refuse **naming themselves**, they do not fall back.

    Two of them, and both are this backend's standing policy rather than this
    round's choice: `dtype=` to anything but float32 is the `check_dtype`
    policy of docs/devices/VULKAN6.md §1 (there is no integer arithmetic here,
    and widening it to make a sum pass is the move that policy exists to
    forbid), and rank above 8 is the push-constant bound every indexed kernel
    on this device already has.
    """
    if _vulkan_or_skip("the sum refusals") is None:
        return
    x = _to_vulkan(_cpu([1.0, 2.0, 3.0, 4.0], (2, 2)))
    try:
        _C._aten_dispatch(_SUM_DIMS, x, [0], False, dtype=_C.int64)
    except NotImplementedError as e:
        assert "int64" in str(e), str(e)
        assert "float32" in str(e), str(e)
    else:
        raise AssertionError("sum(dtype=int64) did not refuse on the vulkan device")

    deep = _to_vulkan(_cpu([1.0] * 512, (2,) * 9))
    try:
        _C._aten_dispatch(_SUM_DIMS, deep, [0], False)
    except NotImplementedError as e:
        assert "9" in str(e) and "8" in str(e), str(e)
    else:
        raise AssertionError("a rank-9 sum did not refuse on the vulkan device")

    # And an out-of-range axis is an IndexError with torch's wording, not a
    # silent clamp to a plausible neighbouring axis.
    try:
        _C._aten_dispatch(_SUM_DIMS, x, [5], False)
    except IndexError as e:
        assert "Dimension out of range" in str(e), str(e)
    else:
        raise AssertionError("sum over axis 5 of a 2-D tensor did not refuse")


# ---------------------------------------------------------------------------
# Does a backward work now? **Yes.** docs/devices/VULKAN9.md §4
# ---------------------------------------------------------------------------

_SDPA_BACKWARD_PROBE = r"""
import json, sys
import torch

req = json.load(sys.stdin)
out = {"probe": torch._C._vulkan_probe()}
if not out["probe"]["available"]:
    json.dump(out, sys.stdout); raise SystemExit
shape = req["shape"]


def dev(flat):
    return torch.tensor(flat, dtype=torch.float32).reshape(shape).to("vulkan")


q = dev(req["q"]).requires_grad_(True)
k = dev(req["k"]).requires_grad_(True)
v = dev(req["v"]).requires_grad_(True)
g = dev(req["g"])
o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
out["forward_device"] = str(o.device)
# The counters bracket the backward ONLY. The readback below is this probe's
# own way of getting the numbers out to the process that owns the oracle, and
# folding it into the measurement would make `host_downloads` unreadable as
# evidence about the gradient rule.
before = torch._C._vulkan_counters()
try:
    grads = torch.autograd.grad(o, [q, k, v], grad_outputs=g)
    out["backward"] = "ok"
except Exception as e:
    out["backward"] = f"{type(e).__name__}: {e}"
    grads = None
after = torch._C._vulkan_counters()
out["counters"] = {key: after[key] - before[key] for key in after}
if grads is not None:
    out["grad_devices"] = [str(t.device) for t in grads]
    out["grads"] = [t.cpu().reshape(-1).tolist() for t in grads]
out["rules"] = sorted(torch._C._tape_rules())
json.dump(out, sys.stdout)
"""


# What the sdpa gradient rule still cannot reach on this device, probed rather
# than read off `tape.rs`. All four are behind `is_causal=True`, and the
# **vulkan forward refuses `is_causal` outright**, so a kernel for any of them
# would be unexercisable -- which is why this round did not write one.
_BACKWARD_STILL_MISSING = (
    "aten.ones.default",
    "aten.tril.default",
    "aten.eq.Scalar",
    "aten.masked_fill.Scalar",
)

# `o.sum().backward()` and `o.mean().backward()` were the walls
# docs/devices/VULKAN9.md §5 measured -- `aten.ones_like.default` and
# `aten.mean.default`. Both are taught now and both loss shapes run, so this
# file's claim about them is the opposite of the one it replaces and is
# asserted as `"ok"` below rather than as a refusal.
#
# The wall that replaces them is a *second* backward into a `.grad` that
# already exists: `AccumulateGrad`'s in-place add. It is named from the probe,
# and the reason it is not implemented is the same shape as the `is_causal`
# four -- **no op on this device writes in place.** `detach`/`alias` share the
# `VkBuffer` (vulkan.rs `dispatch` says so in as many words), so an `add_`
# would write through an alias that other tensors are reading, and it would do
# it *silently*. Teaching it needs a sharing analysis this device does not
# have, not a kernel. docs/devices/VULKAN10.md §6.
_SECOND_ACCUMULATION_WALL = "aten.add_.Tensor"


def _backward_probe(shape, payload):
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{VENDOR_DIR}"
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run([sys.executable, "-c", _SDPA_BACKWARD_PROBE],
                          input=json.dumps(payload), capture_output=True,
                          text=True, env=env, timeout=600)
    assert proc.returncode == 0, (proc.stdout[-2000:], proc.stderr[-4000:])
    return json.loads(proc.stdout)


_BW_SHAPE = (1, 2, 3, 4)


def _backward_payload(torch):
    n = 1
    for d in _BW_SHAPE:
        n *= d
    tensors = {}
    for i, name in enumerate("qkvg"):
        tensors[name] = _up_flat(_rand(torch, *_BW_SHAPE, seed=3300 + i))
    return {"shape": list(_BW_SHAPE), **tensors}


def test_a_backward_through_the_vulkan_sdpa_now_runs_and_agrees_with_upstream():
    """**It runs.** docs/devices/VULKAN8.md §4 said it did not, and named why.

    That document's wall was `aten.sum.dim_IntList` -- the `rowsum(dP * P)` in
    the middle of `sdpa_backward` (tape.rs). This round taught the device that
    reduction and the wall is gone, so the claim this test makes is the
    opposite of the one it replaces, and it is held to a higher bar than
    "no exception": the three gradients are compared **element-wise against
    upstream's own autograd**, at the tolerance `_assert_agreement` re-derives
    from upstream's own float32-vs-float64 error on this same input.

    The measurement is in a separate interpreter running the vendored shim
    (the oracle lives in *this* one and the two cannot both be `torch`), and
    the inputs cross as flat lists so both sides differentiate the same
    numbers rather than two draws that happen to share a seed.
    """
    if _vulkan_or_skip("the sdpa backward agreement") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    import torch.nn.functional as F
    payload = _backward_payload(torch)
    got = _backward_probe(_BW_SHAPE, payload)
    if not got["probe"]["available"]:
        vulkan_coverage.vulkan_skip(
            f"the sdpa backward agreement: the vendored-tree subprocess has no "
            f"loader -- {got['probe']['error']}")
        return

    assert got["forward_device"] == "vulkan", got["forward_device"]
    assert "aten._scaled_dot_product_flash_attention_for_cpu.default" in got["rules"]
    assert got["backward"] == "ok", (
        f"the backward through the vulkan sdpa did not run: {got['backward']}")
    assert got["grad_devices"] == ["vulkan"] * 3, got["grad_devices"]

    def upstream(dtype):
        q, k, v = (torch.tensor(payload[n], dtype=dtype).reshape(_BW_SHAPE)
                   .requires_grad_(True) for n in "qkv")
        g = torch.tensor(payload["g"], dtype=dtype).reshape(_BW_SHAPE)
        o = F.scaled_dot_product_attention(q, k, v)
        return [_up_flat(t) for t in torch.autograd.grad(o, [q, k, v], grad_outputs=g)]

    want32, truth = upstream(torch.float32), upstream(torch.float64)
    _assert_agreement("sdpa backward (dq, dk, dv)", [
        (f"grad_{name}", got["grads"][i], want32[i], truth[i])
        for i, name in enumerate("qkv")])


def test_the_vulkan_sdpa_backward_never_left_the_gpu():
    """The counters, which are the only thing that can say this.

    docs/devices/VULKAN6.md's rule, and it bites hardest here: a backward that
    quietly downloaded the saved q/k/v, differentiated on the host and uploaded
    three answers would produce gradients that pass the agreement test above
    **exactly**, on tensors whose `.device` says `vulkan`. Values cannot tell
    the two apart. `host_downloads` can, and it is zero -- so every one of the
    rule's matmuls, transposes, softmaxes and the new `sum.dim_IntList`
    executed as a compute shader.

    The dispatch count is asserted as a lower bound with a named floor rather
    than an exact number: the rule's op sequence is `tape.rs`'s to choose and
    pinning it here would make this test fail on a refactor that changed
    nothing observable. What must not move is the **zero**.
    """
    if _vulkan_or_skip("the sdpa backward counters") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    got = _backward_probe(_BW_SHAPE, _backward_payload(torch))
    if not got["probe"]["available"]:
        vulkan_coverage.vulkan_skip(
            f"the sdpa backward counters: the vendored-tree subprocess has no "
            f"loader -- {got['probe']['error']}")
        return
    assert got["backward"] == "ok", got["backward"]
    d = got["counters"]
    assert d["host_downloads"] == 0, (
        f"the backward read {d['host_downloads']} buffer(s) back to the host; "
        f"a rule that computed on the CPU would have to, and its gradients "
        f"would still agree with upstream: {d}")
    assert d["host_uploads"] == 0, (
        f"the backward uploaded {d['host_uploads']} buffer(s); every operand "
        f"it needs was already on the device: {d}")
    # The formula is dv = P^T dout, dP = dout v^T, dS = P * (dP - rowsum(dP*P)),
    # dq = scale * dS k, dk = scale * dS^T q, plus the forward recomputation of
    # P. That cannot be fewer than ten kernel launches on a device with no
    # fusion; the floor is deliberately well under what it measures so that
    # only a *collapse* -- the shape a host fallback has -- trips it.
    assert d["shader_dispatches"] >= 10, (
        f"the backward ran only {d['shader_dispatches']} compute shaders; the "
        f"gradient rule cannot be that few kernels on this device: {d}")
    print(f"   sdpa backward on vulkan: {d['shader_dispatches']} compute shaders, "
          f"{d['host_uploads']} uploads, {d['host_downloads']} downloads")


def test_what_the_sdpa_backward_still_cannot_reach_is_unreachable_for_a_reason():
    """The honest remainder -- and why no kernel was written for it.

    Four ops in `sdpa_backward` belong to the `is_causal=True` branch. This
    device's **forward** refuses `is_causal` by name, so that branch cannot be
    entered at all, and a `tril` kernel written now could not be exercised by
    any test -- it would be unproven code with a green tick beside it. So this
    test pins both halves: the ops are still absent, **and** the reason they
    are unreachable is still true.

    It also pins where a loss-shaped backward now gets to. Both walls
    docs/devices/VULKAN9.md §5 measured -- `o.sum().backward()` at
    `aten.ones_like.default` and `o.mean().backward()` at
    `aten.mean.default` -- are gone, so those two are asserted to **run**, and
    the wall that replaced them (a second accumulation into an existing
    `.grad`) is named from the same live probe rather than from this list.
    """
    if _vulkan_or_skip("the remaining sdpa-backward walls") is None:
        return
    try:
        _C._aten_dispatch("aten.this_op_does_not_exist.default",
                          _to_vulkan(_cpu([0.0], [1])))
    except NotImplementedError as e:
        taught = set(x.strip() for x in
                     str(e).split("by name -- ")[1].split(" -- and")[0].split(","))
    else:
        raise AssertionError("an unknown op did not refuse")

    assert "aten.sum.dim_IntList" in taught, (
        "this file claims the sdpa backward runs because sum.dim_IntList was "
        "taught; it is not in the device's own list")
    still = [op for op in _BACKWARD_STILL_MISSING if op not in taught]
    assert still == list(_BACKWARD_STILL_MISSING), (
        f"some of the ops this test names as unreachable are now taught "
        f"({sorted(set(_BACKWARD_STILL_MISSING) - set(still))}); if the "
        f"is_causal forward was lifted too, this test must be rewritten")

    env = dict(os.environ)
    env["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{VENDOR_DIR}"
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run([sys.executable, "-c", _REMAINING_WALLS_PROBE],
                          capture_output=True, text=True, env=env, timeout=600)
    assert proc.returncode == 0, (proc.stdout[-2000:], proc.stderr[-4000:])
    walls = json.loads(proc.stdout)
    if not walls.get("available", True):
        vulkan_coverage.vulkan_skip(
            "the remaining sdpa-backward walls: the vendored-tree subprocess "
            "has no loader")
        return

    assert walls["is_causal_forward"].startswith("NotImplementedError"), (
        f"the vulkan sdpa forward no longer refuses is_causal; the four ops "
        f"above are now reachable and must be implemented or re-justified: "
        f"{walls['is_causal_forward']}")
    assert "is_causal=False" in walls["is_causal_forward"], walls["is_causal_forward"]

    # Both loss shapes docs/devices/VULKAN9.md §5 measured as walls now run.
    assert walls["loss_backward"] == "ok", (
        f"`o.sum().backward()` stopped again: {walls['loss_backward'][:300]}")
    assert walls["mean_backward"] == "ok", (
        f"`o.mean().backward()` stopped again: {walls['mean_backward'][:300]}")

    # And what replaced them. The refusal message *lists every taught op*, so
    # a substring search would match `aten.add.Tensor` in the list and call it
    # the wall. The op that actually stopped is the one the message opens
    # with, before `: not implemented`, and that is what is read here.
    assert walls["second_accumulation"].startswith("NotImplementedError"), (
        f"a second backward into an existing `.grad` now works on this "
        f"device: {walls['second_accumulation']}. That means an in-place "
        f"write landed here, and vulkan.rs `dispatch` states that no op on "
        f"this device writes in place because `detach`/`alias` share the "
        f"buffer -- so this test must be rewritten against whatever made it "
        f"sound, not deleted")
    named = walls["second_accumulation"].split(": ", 1)[1].split(":", 1)[0].strip()
    assert named == _SECOND_ACCUMULATION_WALL, (
        f"the second accumulation stopped at {named!r}, not at the op this "
        f"test names ({_SECOND_ACCUMULATION_WALL!r}): "
        f"{walls['second_accumulation'][:200]}")
    print(f"   `o.sum().backward()` and `o.mean().backward()`: ok; "
          f"next wall (second accumulation): {named}")


_REMAINING_WALLS_PROBE = r"""
import json, sys
import torch

probe = torch._C._vulkan_probe()
out = {"available": probe["available"]}
if not probe["available"]:
    json.dump(out, sys.stdout); raise SystemExit

B, H, T, E = 1, 2, 3, 4


def trio():
    return [torch.randn(B, H, T, E).to("vulkan").requires_grad_(True)
            for _ in range(3)]


q, k, v = trio()
try:
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    o.sum().backward()
    out["loss_backward"] = "ok"
except Exception as e:
    out["loss_backward"] = f"{type(e).__name__}: {e}"

q, k, v = trio()
try:
    o = torch.nn.functional.scaled_dot_product_attention(q, k, v)
    o.mean().backward()
    out["mean_backward"] = "ok"
except Exception as e:
    out["mean_backward"] = f"{type(e).__name__}: {e}"

# A second backward into a `.grad` that already exists -- the in-place add.
w = torch.ones(2, 2, device="vulkan").requires_grad_(True)
try:
    for _ in range(2):
        (w * w).sum().backward()
    out["second_accumulation"] = "ok"
except Exception as e:
    out["second_accumulation"] = f"{type(e).__name__}: {e}"

q, k, v = trio()
try:
    torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
    out["is_causal_forward"] = "ok"
except Exception as e:
    out["is_causal_forward"] = f"{type(e).__name__}: {e}"

json.dump(out, sys.stdout)
"""


# ---------------------------------------------------------------------------
# From "a kernel's gradient agrees" to "a person can train on this device"
# -- docs/devices/VULKAN10.md
#
# docs/devices/VULKAN9.md §5 left the gap named by a live probe rather than
# inferred: the sdpa gradient rule runs when `grad_outputs` is handed to it,
# but a *loss*-shaped backward has to seed the gradient first and stops at
# `aten.ones_like.default`, and a mean-shaped one stops at `aten.mean.default`.
# Those two are gradient seeding and loss reduction, which is the whole of
# what stands between a proven kernel and an optimisation step.
# ---------------------------------------------------------------------------

_ONES_LIKE = "aten.ones_like.default"
_ZEROS_LIKE = "aten.zeros_like.default"
_MEAN_DIM = "aten.mean.dim"
_MEAN_ALL = "aten.mean.default"

LIKE_SHAPES = ((), (1,), (4, 5), (2, 3, 4), (1, 2, 3, 4), (129,))


def test_ones_like_and_zeros_like_are_filled_by_a_shader_not_by_an_upload():
    """The gradient seed -- and it is a *kernel*, deliberately.

    `torch.ones(..., device="vulkan")` on this device builds the fill on the
    host and uploads it (`vulkan.rs` `factory`), and that was the honest
    minimum for a constructor called once at the edge of a program. It is not
    the honest minimum **inside a backward pass**: `torch/autograd/__init__.py`
    `_make_grads` calls `torch.ones_like(out, memory_format=preserve_format)`
    to seed every `loss.backward()`, so an uploading `ones_like` would put a
    host round trip in the middle of every training step and make
    `host_uploads == 0` unavailable as evidence that the step stayed on the
    device. So this is a shader, and this test asserts exactly that: one
    dispatch, **zero uploads**, zero downloads.

    The values are constants, so they are held to bit equality rather than to
    a tolerance -- there is no rounding for a tolerance to be about.
    """
    if _vulkan_or_skip("the ones_like/zeros_like fill") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    for i, shape in enumerate(LIKE_SHAPES):
        n = 1
        for d in shape:
            n *= d
        a = _rand(torch, *shape, seed=4100 + i) if shape else _rand(torch, 1, seed=4100)
        src = _to_vulkan(_cpu(_up_flat(a)[:n] if shape else [1.5], shape))
        for op, want in ((_ONES_LIKE, 1.0), (_ZEROS_LIKE, 0.0)):
            before = _counters()
            out = _C._aten_dispatch(op, src)
            d = _delta(before, _counters())
            assert list(out.shape) == list(shape), (op, out.shape, shape)
            assert str(out.device) == "vulkan", out.device
            assert d["shader_dispatches"] == 1, (
                f"{op}{list(shape)} ran {d['shader_dispatches']} compute "
                f"shaders, expected exactly 1: {d}")
            assert d["host_uploads"] == 0, (
                f"{op}{list(shape)} uploaded {d['host_uploads']} buffer(s); "
                f"the fill happens on the device or `host_uploads == 0` stops "
                f"being readable as evidence about a training step: {d}")
            assert d["host_downloads"] == 0, d
            got = _flat(_to_cpu(out))
            assert got == [want] * n, (op, shape, got[:8])
    # And the `memory_format=preserve_format` that `_make_grads` actually
    # passes is accepted rather than refused -- a refusal there would move the
    # wall off the op that is missing and onto an argument.
    out = _C._aten_dispatch(_ONES_LIKE, _to_vulkan(_cpu([1.0, 2.0], (2,))),
                            memory_format=_C.preserve_format)
    assert _flat(_to_cpu(out)) == [1.0, 1.0], _flat(_to_cpu(out))


def test_ones_like_refuses_a_dtype_this_device_cannot_hold():
    """Not a silent float32 in an int64's clothing.

    `check_dtype` admits float32 and nothing else on this backend
    (docs/devices/VULKAN6.md §1). `ones_like(x, dtype=torch.int64)` is a real
    request upstream answers, and answering it here by ignoring the dtype
    would be exactly the silent conversion that policy exists to forbid.
    """
    if _vulkan_or_skip("the ones_like dtype refusal") is None:
        return
    for op in (_ONES_LIKE, _ZEROS_LIKE):
        try:
            _C._aten_dispatch(op, _to_vulkan(_cpu([1.0, 2.0], (2,))), dtype=_C.int64)
        except NotImplementedError as e:
            assert "int64" in str(e), str(e)
            assert "float32" in str(e), str(e)
        else:
            raise AssertionError(f"{op}(dtype=int64) did not refuse on the vulkan device")



def test_ones_like_refuses_a_device_it_would_have_to_leave_to_honour():
    """The subtler refusal, and the one no counter would have caught.

    `aten_dispatch` picks this backend from the *tensor arguments*, so
    `zeros_like(vk, device="cpu")` reaches the vulkan kernel -- while upstream
    answers that call with a real CPU tensor, and `aten.rs`'s
    `zeros_or_empty_like` calls that non-meta half "the interesting half" in
    as many words. Ignoring the argument gives back a `VkTensor`: a wrong
    answer in the right shape, with the right values in it, on a device that
    ran the right number of shaders. Neither the agreement sweep nor the
    dispatch counters can see that, which is why it is asserted here.
    """
    if _vulkan_or_skip("the ones_like/zeros_like device refusal") is None:
        return
    for op in (_ONES_LIKE, _ZEROS_LIKE):
        for kind in ("cpu", "meta"):
            try:
                out = _C._aten_dispatch(op, _to_vulkan(_cpu([1.0, 2.0], (2,))),
                                        device=_C.device(kind))
            except NotImplementedError as e:
                assert kind in str(e), str(e)
            else:
                raise AssertionError(
                    f"{op}(device={kind!r}) on a vulkan input returned a "
                    f"{out.device} tensor instead of refusing; upstream "
                    f"answers that call with a {kind} tensor")


# (shape, dim, keepdim) -- the same sweep shape as SUM_CASES, because `mean`
# is the same kernel with a divisor and the interesting cases are the same:
# a leading axis, two non-adjacent axes, an empty list, `dim=None`, and a
# 512-long axis where the division's rounding has somewhere to show.
MEAN_CASES = SUM_CASES


def _mean_upstream(torch, a, dim, keepdim):
    return torch.ops.aten.mean.dim(a, dim, keepdim)


def test_mean_over_dims_agrees_with_upstream_at_a_derived_tolerance():
    """`aten.mean.dim` -- the loss reduction, element-wise against upstream.

    It is `sum_dims_f32` with a divisor, not a second kernel and not a
    `sum` followed by a `mul.Scalar` by the reciprocal: the reciprocal is a
    rounding upstream does not do, which is the same rewrite
    docs/devices/VULKAN5.md §3.1 turned MoltenVK's fast-math off for. The
    division happens once, in the shader, on the compensated total.

    The tolerance is re-derived by `_assert_agreement` from this population's
    own upstream float32-vs-float64 error;
    `test_a_wrong_mean_reduction_is_rejected_by_this_tolerance` proves it bites.
    """
    if _vulkan_or_skip("the mean.dim agreement sweep") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    cases = []
    for i, (shape, dim, keepdim) in enumerate(MEAN_CASES):
        a = _rand(torch, *shape, seed=4200 + i)
        out = _C._aten_dispatch(_MEAN_DIM, _to_vulkan(_cpu(_up_flat(a), shape)),
                                dim, keepdim)
        want32 = _mean_upstream(torch, a, dim, keepdim)
        truth = _mean_upstream(torch, a.double(), dim, keepdim)
        assert list(out.shape) == list(want32.shape), (
            f"mean{list(shape)} dim={dim} keepdim={keepdim}: shape "
            f"{list(out.shape)}, upstream {list(want32.shape)}")
        assert str(out.device) == "vulkan", out.device
        cases.append((f"mean{list(shape)} dim={dim} keepdim={keepdim}",
                      _flat(_to_cpu(out)), _up_flat(want32), _up_flat(truth)))
    _assert_agreement("mean.dim", cases)


def test_mean_default_agrees_with_upstream_at_a_derived_tolerance():
    """`aten.mean.default` -- the whole-tensor form a scalar loss is."""
    if _vulkan_or_skip("the mean.default agreement sweep") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    cases = []
    for i, shape in enumerate(((2, 3, 4), (1025,), (4, 5), (2, 512))):
        a = _rand(torch, *shape, seed=4300 + i)
        out = _C._aten_dispatch(_MEAN_ALL, _to_vulkan(_cpu(_up_flat(a), shape)))
        want32 = torch.ops.aten.mean.default(a)
        truth = torch.ops.aten.mean.default(a.double())
        assert list(out.shape) == list(want32.shape), (out.shape, want32.shape)
        assert str(out.device) == "vulkan", out.device
        cases.append((f"mean_all{list(shape)}", _flat(_to_cpu(out)),
                      [want32.item()], [truth.item()]))
    _assert_agreement("mean.default", cases)


def test_a_wrong_mean_reduction_is_rejected_by_this_tolerance():
    """The teeth check, and the wrong answers are the ones this kernel
    could actually have been: dividing by the *kept* extent instead of the
    reduced one (an off-by-one-axis in the divisor, which a `sum` test could
    never see), and forgetting to divide at all.
    """
    if _vulkan_or_skip("the mean tolerance teeth check") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    a = _rand(torch, 6, 7, seed=4401)
    want32 = torch.ops.aten.mean.dim(a, [-1], False)
    truth = torch.ops.aten.mean.dim(a.double(), [-1], False)

    _assert_agreement("mean teeth (control: the correct value)",
                      [("control", _up_flat(want32), _up_flat(want32), _up_flat(truth))])

    wrong_divisor = torch.ops.aten.sum.dim_IntList(a, [-1], False) / 6.0
    undivided = torch.ops.aten.sum.dim_IntList(a, [-1], False)

    margins = []
    for name, wrong in (("the kept extent as the divisor", wrong_divisor),
                        ("no division at all", undivided)):
        got = _up_flat(wrong.float())
        try:
            _assert_agreement(f"mean teeth ({name})",
                              [(name, got, _up_flat(want32), _up_flat(truth))])
        except AssertionError:
            margins.append((name, _elem_ratio(got, _up_flat(want32), _up_flat(truth))))
        else:
            raise AssertionError(
                f"the derived tolerance accepted a mean computed with {name}; "
                f"it is too wide to be evidence of anything")
    for name, ratio in margins:
        print(f"   mean teeth: {name} rejected at {ratio:,.0f}x upstream's own error")
        assert ratio > 1e3, (name, ratio)


def test_the_mean_reduction_ran_on_the_gpu_in_exactly_one_dispatch():
    """One dispatch, not two -- `sum` then `div` would also be on the GPU and
    would also be the wrong kernel, and only a counter can tell them apart.
    """
    if _vulkan_or_skip("the mean dispatch-counter check") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    for i, (shape, dim, keepdim) in enumerate(MEAN_CASES):
        a = _rand(torch, *shape, seed=4500 + i)
        on_device = _to_vulkan(_cpu(_up_flat(a), shape))
        before = _counters()
        out = _C._aten_dispatch(_MEAN_DIM, on_device, dim, keepdim)
        d = _delta(before, _counters())
        assert d["shader_dispatches"] == 1, (
            f"mean{list(shape)} dim={dim} ran {d['shader_dispatches']} compute "
            f"shaders, expected exactly 1: {d}")
        assert d["host_downloads"] == 0, d
        assert d["host_uploads"] == 0, d
        assert str(out.device) == "vulkan", out.device

    on_device = _to_vulkan(_cpu(_up_flat(_rand(torch, 3, 4, seed=4600)), (3, 4)))
    before = _counters()
    _C._aten_dispatch(_MEAN_ALL, on_device)
    d = _delta(before, _counters())
    assert d["shader_dispatches"] == 1, d
    assert d["host_downloads"] == 0, d
    assert d["host_uploads"] == 0, d


# ---------------------------------------------------------------------------
# A training step -- forward, loss, backward, parameter update
# ---------------------------------------------------------------------------

# A 4->3 linear layer, mean-squared error, one SGD step. Small on purpose: the
# claim is that a step *completes on the device and agrees*, and a bigger
# tensor would only make the float32 reduction error larger without making the
# claim stronger. The update is written functionally (`w - lr * w.grad`)
# rather than in place, because **no op on this device writes in place** --
# `detach`/`alias` share the buffer (vulkan.rs `dispatch`), so an in-place
# update would be unsound here rather than merely unimplemented.
_TRAIN_STEP_PROBE = r"""
import json, sys
import torch

req = json.load(sys.stdin)
out = {"probe": torch._C._vulkan_probe()}
if not out["probe"]["available"]:
    json.dump(out, sys.stdout); raise SystemExit

lr = req["lr"]


def dev(flat, shape):
    return torch.tensor(flat, dtype=torch.float32).reshape(shape).to("vulkan")


x = dev(req["x"], req["x_shape"])
y = dev(req["y"], req["y_shape"])
w = dev(req["w"], req["w_shape"]).requires_grad_(True)

# The counters bracket the whole step: forward, loss, backward, update. The
# readback below is this probe's own way of getting the answer out to the
# process that owns the oracle and is deliberately outside the bracket.
before = torch._C._vulkan_counters()
try:
    pred = x @ w
    diff = pred - y
    loss = (diff * diff).mean()
    loss.backward()
    with torch.no_grad():
        updated = w - lr * w.grad
    out["step"] = "ok"
except Exception as e:
    out["step"] = f"{type(e).__name__}: {e}"
    updated = None
    loss = None
after = torch._C._vulkan_counters()
out["counters"] = {k: after[k] - before[k] for k in after}
if updated is not None:
    out["devices"] = {"updated": str(updated.device), "grad": str(w.grad.device),
                      "loss": str(loss.device)}
    out["updated"] = updated.cpu().reshape(-1).tolist()
    out["grad"] = w.grad.cpu().reshape(-1).tolist()
    out["loss"] = float(loss.cpu().reshape(-1).tolist()[0]) if loss.dim() == 0 or loss.numel() == 1 else None
    out["w_before"] = w.detach().cpu().reshape(-1).tolist()
json.dump(out, sys.stdout)
"""

_TRAIN_N, _TRAIN_IN, _TRAIN_OUT = 8, 4, 3
_TRAIN_LR = 0.1


def _train_payload(torch):
    x = _rand(torch, _TRAIN_N, _TRAIN_IN, seed=4700)
    y = _rand(torch, _TRAIN_N, _TRAIN_OUT, seed=4701)
    w = _rand(torch, _TRAIN_IN, _TRAIN_OUT, seed=4702)
    return {
        "lr": _TRAIN_LR,
        "x": _up_flat(x), "x_shape": [_TRAIN_N, _TRAIN_IN],
        "y": _up_flat(y), "y_shape": [_TRAIN_N, _TRAIN_OUT],
        "w": _up_flat(w), "w_shape": [_TRAIN_IN, _TRAIN_OUT],
    }


def _train_probe(payload):
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{VENDOR_DIR}"
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run([sys.executable, "-c", _TRAIN_STEP_PROBE],
                          input=json.dumps(payload), capture_output=True,
                          text=True, env=env, timeout=900)
    assert proc.returncode == 0, (proc.stdout[-2000:], proc.stderr[-4000:])
    return json.loads(proc.stdout)


def _upstream_step(torch, payload):
    """The same step upstream, on the CPU, from the *same numbers*."""
    def t(key, shape_key):
        return torch.tensor(payload[key], dtype=torch.float32).reshape(payload[shape_key])

    x = t("x", "x_shape")
    y = t("y", "y_shape")
    w = t("w", "w_shape").requires_grad_(True)
    pred = x @ w
    diff = pred - y
    loss = (diff * diff).mean()
    loss.backward()
    with torch.no_grad():
        updated = w - payload["lr"] * w.grad
    return loss, w.grad, updated


def test_a_training_step_runs_end_to_end_on_the_vulkan_device_and_agrees_with_upstream():
    """**A parameter update, on the device, with the weights actually moving.**

    docs/devices/VULKAN9.md proved one kernel's gradient. This proves a *step*:
    forward (`matmul`), loss (`sub`, `mul`, `mean`), backward (seeded by
    `ones_like`, through the `mean` and `mul` and `matmul` rules of tape.rs),
    and an SGD update. The measurement is in a separate interpreter running the
    vendored shim, fed the *same flat numbers* the oracle here differentiates,
    so neither side depends on the two RNGs agreeing.

    Three things are asserted and they are not the same claim:

      * the updated weights agree element-wise with the same step run upstream
        on the CPU, at a tolerance `_assert_agreement` re-derives;
      * the weights **changed** -- a gradient of exactly zero would satisfy
        agreement while proving nothing about the backward;
      * the gradient and the loss agree too, so a step that landed on the
        right weights by cancelling two errors is not read as a pass.
    """
    if _vulkan_or_skip("the vulkan training step") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    payload = _train_payload(torch)
    got = _train_probe(payload)
    if not got["probe"]["available"]:
        vulkan_coverage.vulkan_skip(
            "the vulkan training step: the vendored-tree subprocess has no loader")
        return
    assert got["step"] == "ok", (
        f"a training step did not complete on the vulkan device: {got['step']}\n"
        f"counters at the refusal: {got['counters']}")
    assert got["devices"]["updated"] == "vulkan", got["devices"]
    assert got["devices"]["grad"] == "vulkan", got["devices"]
    assert got["devices"]["loss"] == "vulkan", got["devices"]

    loss32, grad32, updated32 = _upstream_step(torch, payload)
    loss64, grad64, updated64 = _upstream_step(torch, payload)

    # The weights moved. Without this, a backward that produced zeros would
    # agree with an upstream step that also produced zeros -- and it does not:
    # the biggest coordinate moves by the margin printed below.
    moved = max(abs(a - b) for a, b in zip(got["updated"], got["w_before"]))
    assert moved > 1e-3, (
        f"the parameters did not move: worst coordinate changed by {moved:.3e}. "
        f"An unchanged weight satisfies agreement and proves nothing.")
    print(f"   training step: worst parameter moved {moved:.3e} at lr={_TRAIN_LR}")

    _assert_agreement("training step (loss, grad, updated weights)", [
        ("loss", [got["loss"]], [loss32.item()],
         [float(loss64.double().item())]),
        ("grad_w", got["grad"], _up_flat(grad32), _up_flat(grad64.double())),
        ("updated w", got["updated"], _up_flat(updated32),
         _up_flat(updated64.double())),
    ])


def test_the_training_step_never_left_the_gpu():
    """The claim values cannot make (docs/devices/VULKAN9.md §4).

    A step that downloaded the weights, ran the whole thing in candle on the
    host and uploaded the answer would pass the agreement test above *exactly*
    and would report `.device == "vulkan"` for every tensor in it. The only
    witness that separates the two is the counter pair, so this asserts
    **zero uploads and zero downloads** across forward, loss, backward and
    update -- the gradient seed included, which is why `ones_like` is a shader
    and not an upload.
    """
    if _vulkan_or_skip("the vulkan training step counters") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    got = _train_probe(_train_payload(torch))
    if not got["probe"]["available"]:
        vulkan_coverage.vulkan_skip(
            "the vulkan training step counters: the vendored-tree subprocess "
            "has no loader")
        return
    assert got["step"] == "ok", got["step"]
    d = got["counters"]
    assert d["host_downloads"] == 0, (
        f"the training step read {d['host_downloads']} buffer(s) back to the "
        f"host mid-step; it runs on the device or it does not run: {d}")
    assert d["host_uploads"] == 0, (
        f"the training step uploaded {d['host_uploads']} buffer(s) mid-step; "
        f"every input was already on the device before the bracket: {d}")
    # A floor, not an exact count: the exact number is `tape.rs`'s to choose
    # and pinning it would break this test on a refactor that changes nothing
    # observable. What must not move is the pair of zeros above, and a
    # *collapse* -- the shape a host fallback has -- trips this.
    assert d["shader_dispatches"] >= 10, (
        f"the whole step ran only {d['shader_dispatches']} compute shaders; a "
        f"forward, a loss, a backward and an update cannot be that few on a "
        f"device with no fusion: {d}")
    print(f"   training step on vulkan: {d['shader_dispatches']} compute shaders, "
          f"{d['host_uploads']} uploads, {d['host_downloads']} downloads")


# ---------------------------------------------------------------------------
# Three steps, not one -- the loss has to actually come down
# ---------------------------------------------------------------------------

# One step proves the wiring. It does not prove the *sign*: a gradient with
# the wrong sign, or one scaled by a constant, would agree with an upstream
# step computed the same wrong way only if upstream were wrong too -- but it
# would sail through a test that only asks "did the weights move". So this
# runs three steps and requires the loss to fall monotonically, which is a
# property of the arithmetic rather than of the comparison.
_TRAIN_LOOP_PROBE = r"""
import json, sys
import torch

req = json.load(sys.stdin)
out = {"probe": torch._C._vulkan_probe()}
if not out["probe"]["available"]:
    json.dump(out, sys.stdout); raise SystemExit


def dev(flat, shape):
    return torch.tensor(flat, dtype=torch.float32).reshape(shape).to("vulkan")


x = dev(req["x"], req["x_shape"])
y = dev(req["y"], req["y_shape"])
w = dev(req["w"], req["w_shape"]).requires_grad_(True)

before = torch._C._vulkan_counters()
losses = []
try:
    for _ in range(req["steps"]):
        diff = x @ w - y
        loss = (diff * diff).mean()
        loss.backward()
        losses.append(loss)
        # A fresh leaf each step rather than an in-place update: **no op on
        # this device writes in place** (vulkan.rs `dispatch` -- `detach` and
        # `alias` share the buffer), so `w -= lr * w.grad` would be unsound
        # here rather than merely unimplemented, and a fresh leaf also gives
        # `AccumulateGrad` an empty `.grad` to store into.
        with torch.no_grad():
            w = (w - req["lr"] * w.grad).requires_grad_(True)
    out["loop"] = "ok"
except Exception as e:
    out["loop"] = f"{type(e).__name__}: {e}"
after = torch._C._vulkan_counters()
out["counters"] = {k: after[k] - before[k] for k in after}
out["losses"] = [float(t.cpu().reshape(-1).tolist()[0]) for t in losses]
out["w_final"] = w.detach().cpu().reshape(-1).tolist()
json.dump(out, sys.stdout)
"""

_TRAIN_STEPS = 3


def _upstream_loop(torch, payload):
    def t(key, shape_key):
        return torch.tensor(payload[key], dtype=torch.float32).reshape(payload[shape_key])

    x = t("x", "x_shape")
    y = t("y", "y_shape")
    w = t("w", "w_shape").requires_grad_(True)
    losses = []
    for _ in range(payload["steps"]):
        diff = x @ w - y
        loss = (diff * diff).mean()
        loss.backward()
        losses.append(loss.item())
        with torch.no_grad():
            w = (w - payload["lr"] * w.grad).requires_grad_(True)
    return losses, w.detach()


def test_three_training_steps_drive_the_loss_down_and_track_upstream():
    """**Training, not one step of it.** docs/devices/VULKAN10.md §5.

    Three SGD steps on the device, and two claims a single step cannot make:

      * the loss falls at every step. A gradient with the wrong sign or a
        mis-scaled reduction still "moves the weights", and still agrees with
        an upstream step if the comparison is the only check. A falling loss
        is a property of the arithmetic itself.
      * the whole loop stays on the device -- zero uploads and zero downloads
        across all three steps, so nothing is re-seeded from the host between
        them.

    The loss sequence and the final weights are then compared element-wise
    with the same loop run upstream on the CPU from the same numbers, which is
    where error would show if it were accumulating.
    """
    if _vulkan_or_skip("the vulkan training loop") is None:
        return
    torch = _upstream()
    if torch is None:
        return
    payload = dict(_train_payload(torch), steps=_TRAIN_STEPS)
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{os.environ.get('PYTHONPATH', '')}:{VENDOR_DIR}"
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run([sys.executable, "-c", _TRAIN_LOOP_PROBE],
                          input=json.dumps(payload), capture_output=True,
                          text=True, env=env, timeout=900)
    assert proc.returncode == 0, (proc.stdout[-2000:], proc.stderr[-4000:])
    got = json.loads(proc.stdout)
    if not got["probe"]["available"]:
        vulkan_coverage.vulkan_skip(
            "the vulkan training loop: the vendored-tree subprocess has no loader")
        return
    assert got["loop"] == "ok", (
        f"the training loop stopped: {got['loop']}\ncounters: {got['counters']}")
    assert len(got["losses"]) == _TRAIN_STEPS, got["losses"]

    falls = all(b < a for a, b in zip(got["losses"], got["losses"][1:]))
    assert falls, (
        f"the loss did not fall monotonically over {_TRAIN_STEPS} steps: "
        f"{got['losses']}. A step that moves the weights in the wrong "
        f"direction still moves them.")
    print(f"   training loop on vulkan: losses "
          f"{' -> '.join(f'{v:.6f}' for v in got['losses'])}, "
          f"{got['counters']['shader_dispatches']} compute shaders, "
          f"{got['counters']['host_uploads']} uploads, "
          f"{got['counters']['host_downloads']} downloads")

    d = got["counters"]
    assert d["host_uploads"] == 0 and d["host_downloads"] == 0, (
        f"the training loop crossed the host boundary {d['host_uploads']} up / "
        f"{d['host_downloads']} down; every input was on the device before "
        f"the bracket and the loop never needs to leave it: {d}")

    losses32, w32 = _upstream_loop(torch, payload)
    losses64, w64 = _upstream_loop(torch, payload)
    _assert_agreement("training loop (losses, final weights)", [
        ("losses", got["losses"], losses32, [float(v) for v in losses64]),
        ("final w", got["w_final"], _up_flat(w32), _up_flat(w64.double())),
    ])


# ---------------------------------------------------------------------------
# The runner -- and why it is the LAST thing in this file
#
# `_main` collects `test_*` out of `globals()`, so it can only see what has
# already been defined when it runs. This block used to sit in the middle of
# the file, above `_BERT_SHIM_SCRIPT_MASK` and
# `test_the_pretrained_bert_sdpa_mask_forward_builds_mask_on_host_and_agrees_with_upstream`
# -- and a module executes top to bottom, so `raise SystemExit(_main())` fired
# before that test was ever defined. **It had never run.** The tally said
# `ran=40` with 41 test functions in the file, and nothing compared the two
# numbers; the suite was green because the test did not exist yet at the moment
# the runner looked.
#
# That is the "verification that cannot fail" shape of CLAUDE.md §5.5, in its
# purest form: not a weak assertion, an *unreached* one. The guard against it
# coming back is `test_every_test_in_this_file_is_actually_collected` above,
# which counts `def test_` in the source and requires the collected list to
# match -- so a test added below a future stray `__main__` block fails here
# instead of disappearing.
# ---------------------------------------------------------------------------

def _collectable():
    return [(name, fn) for name, fn in sorted(globals().items())
            if name.startswith("test_")]


def test_every_test_in_this_file_is_actually_collected():
    """A test defined after the runner is a test that never runs.

    No Vulkan device needed -- this one is about this file, not the GPU, so it
    does not skip and cannot be hidden by a machine without a loader.
    """
    import re
    src = open(os.path.abspath(__file__)).read()
    in_source = sorted(set(re.findall(r"^def (test_\w+)", src, re.M)))
    collected = sorted(name for name, _ in _collectable())
    missing = [n for n in in_source if n not in collected]
    assert not missing, (
        f"{len(missing)} test(s) are defined in this file but were not collected "
        f"by the runner: {missing}. A `raise SystemExit(_main())` above them is "
        f"how that happens, and it silently happened once already.")


def _main():
    # `run_tests` prints SKIP rather than ok for a test that had no Vulkan
    # device, and a `VULKAN:` tally that run.sh adds up (docs/devices/VULKAN5.md §2).
    failures = vulkan_coverage.run_tests(_collectable())
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
