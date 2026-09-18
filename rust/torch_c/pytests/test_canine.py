"""`aten.constant_pad_nd.default` on a non-contiguous meta tensor -- the tenth
architecture, and what it turned out actually to be.

`docs/graph/LIFTFRESH.md` §6 left `torch.export` at **9 of 10** under the
replay-and-agree bar and named the tenth: `canine`, stopping -- it recorded --
at `aten.zeros_like.default` on a non-contiguous meta tensor, in
`docs/graph/STRIDE.md` §3's territory, where *a dense tensor cannot be built
with a caller-chosen stride*.  That is a representation question and it was
left open for that reason.

**Re-measured at the head of this round, `canine` does not stop there.**  It
stops one wall earlier, and that wall is not a representation question at all:

    NotImplementedError: torch._C shim: the meta kernel for
    aten.constant_pad_nd.default has not been shown to produce upstream's
    output stride, and one of its inputs is not laid out contiguously
    (shape [1, 64, 8], stride [512, 1, 64])

That is `MetaStrideRule::Unverified`'s gate -- and behind the gate there is no
meta arm for the op at all.  So this is exactly the shape the brief asked to
check for first, the one that has caught this project repeatedly: **an operator
already in `_aten_implemented()` that merely lacks a meta kernel.**  Nothing
about it needs a dense tensor to carry a stride.

Upstream's rule was measured, in a separate subprocess, before anything was
written, and it is completely stated by `_constant_pad_nd_meta` in
`torch/_meta_registrations.py`:

* **every pad entry `<= 0`** -- the meta registration's test is `p <= 0`, not
  the reference implementation's `p < 0`, so an all-zero pad takes this branch
  too -- the input is `narrow`ed and `clone`d, so the output carries the
  *preserve_format* stride of the narrowed layout.  A transposed `(3, 4)` of
  stride `(1, 3)` padded `[-1, -1]` answers `(3, 2)` stride `(1, 3)`;
* **otherwise** the output is `empty(new_shape, memory_format=
  suggest_memory_format(input))` -- contiguous for every layout except a
  channels-last-strided one, which stays channels-last.  The same transposed
  input padded `[1, 1]` answers `(3, 6)` stride `(6, 1)`.

Those two rows are the whole measurement and they *disagree with each other*,
which is what makes the tests below able to fail.  A kernel spelled
`AlwaysContiguous` gets the positive-pad row right and the negative-pad row
wrong; one spelled `PreserveFormat` gets it the other way round.  So the
layout is asserted as the **difference between the two branches** rather than
as a stride written down beside the kernel, which is
`docs/graph/LIFTFRESH.md` §7.1's N2 lesson.

Nothing here asserts "export() returned".  The bar is three verdicts kept
apart -- exported / replayed / **agreed** -- and the export test below also
requires the operator to be *in* the graph, `docs/graph/COMPILE.md`'s trap.
"""

import json
import os
import subprocess
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

#: float32 machine epsilon.  Every tolerance below is a multiple of this and of
#: nothing else; `docs/numerics/AGREE.md` is the method.
_EPS32 = 1.1920929e-07


def _available():
    return os.path.isfile(_VENDOR_SHIM)


def _run(script, shim, timeout=1800):
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True, text=True, env=env, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ---------------------------------------------------------------------------
# the meta kernel's layout, over layouts x pads, against upstream
# ---------------------------------------------------------------------------

#: Input layouts, as `(shape, stride)` handed straight to `empty_strided` so
#: that both sides build the *same* layout without depending on whether either
#: can reach it through `permute`/`to(memory_format=)`.  The shim refuses
#: `to(memory_format=channels_last)` outright (docs/graph/EXPORT5.md §3), so
#: the channels-last rows could not be built any other way here.
_LAYOUTS = [
    ("contiguous_3d",       [1, 64, 8],       [512, 8, 1]),
    ("canine_3d",           [1, 64, 8],       [512, 1, 64]),   # canine's own
    ("permuted_3d",         [1, 64, 8],       [1, 8, 1]),
    ("gappy_3d",            [1, 64, 8],       [1024, 16, 1]),
    ("contiguous_2d",       [3, 4],           [4, 1]),
    ("transposed_2d",       [3, 4],           [1, 3]),
    ("gappy_2d",            [3, 4],           [8, 1]),
    ("contiguous_4d",       [2, 3, 4, 5],     [60, 20, 5, 1]),
    ("channels_last_4d",    [2, 3, 4, 5],     [60, 1, 15, 3]),
    ("n111_4d",             [2, 1, 1, 1],     [1, 1, 1, 1]),
    ("unit_channel_4d",     [2, 3, 1, 1],     [3, 1, 3, 3]),
    ("channels_last_3d_5d", [2, 3, 4, 5, 6],  [360, 1, 90, 18, 3]),
    ("contiguous_5d",       [2, 3, 4, 5, 6],  [360, 120, 30, 6, 1]),
    ("expanded_2d",         [3, 4],           [1, 0]),
]

#: Pads.  Both branches of upstream's rule are present and so is the boundary
#: between them: `[0, 0]` is all-`<= 0` and therefore takes the *clone* branch,
#: while `[0, 1]` takes the fill branch.  A rule written with the reference
#: implementation's `p < 0` instead of the meta registration's `p <= 0` answers
#: `[0, 0]` wrongly and nothing else, so that row is load-bearing.
_PADS = [
    [1, 1], [0, 2], [2, 0], [0, 1],
    [2, 0, 1, 1], [1, 1, 1, 1],
    [-1, -1], [0, -1], [-1, 0], [0, 0], [0, 0, 0, 0],
    [-1, 2], [2, -1],
]

_META_PROBE = r"""
import json
import torch

LAYOUTS = %(layouts)s
PADS = %(pads)s

out = {"is_shim": "torchnative" in (torch.__file__ or ""), "cases": {}}
op = torch.ops.aten.constant_pad_nd.default

for name, shape, stride in LAYOUTS:
    for pad in PADS:
        if len(pad) // 2 > len(shape):
            continue
        key = name + "|" + ",".join(str(p) for p in pad)
        try:
            x = torch.empty_strided(tuple(shape), tuple(stride), device="meta")
            r = op(x, list(pad), 0.0)
            out["cases"][key] = {
                "shape": list(r.shape), "stride": list(r.stride()),
                "dtype": str(r.dtype), "device": str(r.device),
            }
        except Exception as exc:
            out["cases"][key] = {"raised": type(exc).__name__, "msg": str(exc)[:200]}

print(json.dumps(out))
"""


def _meta_probe():
    return _META_PROBE % {
        "layouts": json.dumps(_LAYOUTS),
        "pads": json.dumps(_PADS),
    }


def test_constant_pad_nd_meta_layout_agrees_with_upstream():
    """Shape AND stride, exactly, for every layout x pad -- against upstream's
    own answer in a second process.

    Strides are integers; there is no tolerance here and none to widen.  The
    matrix contains upstream's *own* refusals as well as its answers (a pad
    longer than the rank), and those are compared by exception type too, so a
    kernel that answered where upstream refuses fails rather than scoring.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    probe = _meta_probe()
    shim, up = _run(probe, shim=True), _run(probe, shim=False)
    assert shim["is_shim"] and not up["is_shim"], (shim["is_shim"], up["is_shim"])

    bad = []
    for key, want in sorted(up["cases"].items()):
        got = shim["cases"][key]
        if "raised" in want:
            if "raised" not in got:
                bad.append(f"{key}: upstream raised {want['raised']}, shim answered {got}")
            continue
        if "raised" in got:
            bad.append(f"{key}: upstream {want['shape']}/{want['stride']}, "
                       f"shim raised {got['raised']}: {got['msg']}")
            continue
        if got["shape"] != want["shape"] or got["stride"] != want["stride"]:
            bad.append(f"{key}: upstream {want['shape']} stride {want['stride']}, "
                       f"shim {got['shape']} stride {got['stride']}")
    assert not bad, "constant_pad_nd meta disagrees with upstream:\n  " + "\n  ".join(bad)
    answered = sum(1 for v in up["cases"].values() if "raised" not in v)
    print(f"CANINEPAD: {len(up['cases'])} layout x pad cases, {answered} answered, "
          "shape and stride identical to upstream")


def test_constant_pad_nd_meta_negative_pad_preserves_where_positive_pad_contiguates():
    """The two branches must *differ*, on one and the same input.

    This is the assertion a single-rule kernel cannot pass, and it is why the
    layout is stated as a difference rather than as a stride beside the code
    (`docs/graph/LIFTFRESH.md` §7.1, N2).  On a transposed `(3, 4)` of stride
    `(1, 3)`:

        pad [1, 1]    -> (3, 6) stride (6, 1)   a fresh contiguous buffer
        pad [-1, -1]  -> (3, 2) stride (1, 3)   the narrowed input's layout

    An `AlwaysContiguous` rule passes the first row and fails the second; a
    `PreserveFormat` rule does the reverse.  Both values are upstream's,
    re-measured here rather than transcribed from the table above.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    probe = _meta_probe()
    shim, up = _run(probe, shim=True), _run(probe, shim=False)

    pos, neg = up["cases"]["transposed_2d|1,1"], up["cases"]["transposed_2d|-1,-1"]
    assert pos["stride"] == [6, 1] and neg["stride"] == [1, 3], (pos, neg)
    assert pos["stride"] != neg["stride"], "the oracle no longer separates the branches"

    for key in ("transposed_2d|1,1", "transposed_2d|-1,-1", "transposed_2d|0,0"):
        assert shim["cases"][key] == up["cases"][key], (
            key, shim["cases"][key], up["cases"][key])
    assert shim["cases"]["transposed_2d|1,1"]["stride"] != \
        shim["cases"]["transposed_2d|-1,-1"]["stride"], \
        "the shim answers one stride for both branches"
    print("CANINEBRANCH: positive pad contiguates, non-positive pad preserves, "
          "same input, both upstream's")


def test_constant_pad_nd_meta_keeps_a_channels_last_input_channels_last():
    """`suggest_memory_format` is the fill branch's memory format, so the one
    layout that does NOT come out contiguous is a channels-last-strided input.

    Upstream's two ambiguity fallbacks come with it and are asserted to *stay*
    ambiguous: an `N111` whose batch stride equals its channel stride, and a
    unit-channel `(2, 3, 1, 1)`, are the pair that a naive "is any stride out
    of order" test gets wrong in opposite directions.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    probe = _meta_probe()
    shim, up = _run(probe, shim=True), _run(probe, shim=False)

    cl = up["cases"]["channels_last_4d|1,1"]
    assert cl["stride"] == [84, 1, 21, 3], cl
    assert up["cases"]["contiguous_4d|1,1"]["stride"] == [84, 28, 7, 1]
    assert up["cases"]["n111_4d|1,1"]["stride"] == [3, 3, 3, 1], up["cases"]["n111_4d|1,1"]
    assert up["cases"]["unit_channel_4d|1,1"]["stride"] == [9, 1, 9, 3]

    for key in ("channels_last_4d|1,1", "contiguous_4d|1,1", "n111_4d|1,1",
                "unit_channel_4d|1,1", "channels_last_3d_5d|1,1"):
        assert shim["cases"][key] == up["cases"][key], (
            key, shim["cases"][key], up["cases"][key])
    print("CANINECL: channels-last in, channels-last out; both of upstream's "
          "ambiguity fallbacks stay contiguous")


# ---------------------------------------------------------------------------
# the bar: a module that pads a non-contiguous tensor, exported and replayed
# ---------------------------------------------------------------------------

#: `canine`'s own shape of expression, reduced to a module with no dependence
#: on `transformers`: transpose, then `F.pad` along the last axis.  The input
#: to the pad is non-contiguous, which is exactly the tensor whose meta
#: stride the gate refused.
_EXPORT_PROBE = r"""
import json
import torch

out = {"stage": "start"}
try:
    class M(torch.nn.Module):
        def forward(self, x):
            t = x.transpose(1, 2)
            p = torch.nn.functional.pad(t, (2, 3), "constant", 0.0)
            q = torch.nn.functional.pad(p, (-1, -1), "constant", 0.0)
            return (q * 2.0).transpose(1, 2).contiguous()

    m = M().eval()
    x = torch.arange(1 * 8 * 64, dtype=torch.float32).reshape(1, 8, 64) / 97.0
    with torch.no_grad():
        eager = m(x)
    out["stage"] = "eager"
    out["eager"] = [float(v) for v in eager.flatten().tolist()]
    out["shape"] = list(eager.shape)

    with torch.no_grad():
        x64 = x.double()
        eager64 = M().eval()(x64)
    out["oracle"] = [abs(float(a) - float(b)) for a, b in
                     zip(eager.flatten().tolist(), eager64.flatten().tolist())]

    ep = torch.export.export(m, (x,), strict=False)
    out["stage"] = "exported"
    out["targets"] = sorted({
        str(n.target) for n in ep.graph.nodes if n.op == "call_function"})
    with torch.no_grad():
        replayed = ep.module()(x)
    out["stage"] = "replayed"
    out["replayed"] = [float(v) for v in replayed.flatten().tolist()]
    out["replayed_shape"] = list(replayed.shape)
    out["stage"] = "done"
except Exception as exc:
    import traceback
    out["error"] = traceback.format_exc()[-4000:]

print(json.dumps(out))
"""


def test_padding_a_noncontiguous_tensor_exports_replays_and_agrees():
    """THE BAR, on `canine`'s shape of expression: three verdicts kept apart,
    none of which is "export() returned without raising".

    The tolerance is derived, `docs/numerics/AGREE.md` §2's method -- the p90
    of upstream's own float32-vs-float64 relative error on these very outputs,
    floored at 8 ulp.  There is no constant here to widen.

    `constant_pad_nd` is required to be in the graph: an `ExportedProgram`
    holding no operators would replay and agree trivially.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim, up = _run(_EXPORT_PROBE, shim=True), _run(_EXPORT_PROBE, shim=False)
    assert up["stage"] == "done", up.get("error")
    assert shim["stage"] == "done", (
        f"shim stopped at stage {shim['stage']}:\n{shim.get('error')}")
    assert shim["shape"] == up["shape"] == shim["replayed_shape"], (
        shim["shape"], up["shape"], shim["replayed_shape"])

    scale = max(abs(v) for v in up["eager"])
    rel = sorted(e / scale for e in up["oracle"])
    tol = max(rel[int(0.9 * (len(rel) - 1))], 8 * _EPS32) * scale
    replay = max(abs(a - b) for a, b in zip(shim["replayed"], shim["eager"]))
    cross = max(abs(a - b) for a, b in zip(shim["eager"], up["eager"]))
    assert replay <= tol, f"replay disagrees with eager: {replay} > {tol}"
    assert cross <= tol, f"shim eager disagrees with upstream: {cross} > {tol}"

    targets = shim["targets"]
    assert any("constant_pad_nd" in t for t in targets), (
        f"the exported graph does not contain constant_pad_nd: {targets}")
    print("CANINEEXPORT: exported+replayed+agreed, worst relative "
          f"{max(replay, cross) / scale:.3e}; {len(targets)} distinct call "
          "targets including constant_pad_nd")


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
