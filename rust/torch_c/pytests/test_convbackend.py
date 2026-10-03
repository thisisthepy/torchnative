"""`torch._C._select_conv_backend` -- a backend-selection query, not a kernel.

`docs/graph/STRIDE.md` §5 named this as the second of the two walls the ten
exportable architectures stop at, four of them here.  This file is that door,
and the first thing it records is what the door turned out to be.

It is **not** a convolution.  Its only consumer in the vendored tree is
`torch/_subclasses/fake_impls.py:1811`, which asks which kernel a convolution
would dispatch to and then asks `_conv_determine_backend_memory_format` what
memory format that kernel's output would carry.  The answer is used for one
thing: a `t.to(memory_format=...)` on the result.

So the honest answer is a statement about this shim's own convolution, and
upstream supplies both the vocabulary and the answer.  `_ConvBackend.Overrideable`
is upstream's name for "a backend outside this enumeration handles this", and
upstream returns exactly that for every meta-tensor convolution -- measured
here, in a subprocess, rather than taken on the word of the comment at
`torch/_meta_registrations.py:2793` that says so.

Everything below is compared with a live upstream torch in a separate process.
The enum's names and values in particular **cannot** come from the vendored
tree: `torch/_C/__init__.pyi` declares `class ConvBackend(Enum): ...` with zero
members.  They are transcribed in `bootstrap.py`, and this file is the check
on that transcription, the same relationship `verify_schemas.py` has with
`overloads.json`.

What this file cannot see: whether answering the query moves `torch.export`.
It does not, and that is measured in `docs/graph/VARMEAN.md` §4 rather than
here -- the four architectures behind this wall reach a further one.
"""

import json
import os
import subprocess
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")


def _available():
    return os.path.isfile(_VENDOR_SHIM)


def _run(script, shim, timeout=900):
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
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out["is_shim"] is shim, f"probe meant for shim={shim} ran on the other side"
    return out


_ENUM_PROBE = r"""
import json
import torch
out = {"is_shim": hasattr(torch._C, "_aten_implemented")}
CB = torch._C._ConvBackend
out["members"] = sorted(
    (n, int(getattr(CB, n).value)) for n in dir(CB)
    if not n.startswith("_") and n not in ("name", "value")
)
out["repr"] = repr(CB.Overrideable)
out["str"] = str(CB.Overrideable)
out["module"] = CB.__module__
print(json.dumps(out))
"""


def test_the_conv_backend_enum_is_upstreams_names_and_values():
    """The transcription check.

    `bootstrap.py` carries these twenty-two names and values as literals,
    because the vendored `.pyi` has none of them.  A literal table needs a
    re-derivation, not a review, so this reads upstream's own enum in a
    subprocess and diffs -- including the value gap at 9 and upstream's
    `MpsTranspose,`, whose trailing comma is transcribed as found.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim = _run(_ENUM_PROBE, shim=True)
    up = _run(_ENUM_PROBE, shim=False)
    assert shim["members"] == up["members"], (
        f"enum differs:\n  shim={shim['members']}\n  up  ={up['members']}")
    assert shim["repr"] == up["repr"], (shim["repr"], up["repr"])
    assert shim["str"] == up["str"], (shim["str"], up["str"])
    assert shim["module"] == up["module"] == "torch._C", (
        "`__module__` is not `torch._C`. The way this goes wrong is worth "
        "knowing: `torch/__init__.py:1091` rewrites `__module__` to `torch` "
        "for every PUBLIC name in `dir(_C)`, so exposing this class as "
        "`torch._C.ConvBackend` as well -- which the `.pyi` annotation "
        "invites -- moves it off `torch._C`. Upstream has no runtime "
        "`ConvBackend`, so the shim must not either.")
    # A list, not a tuple: JSON has no tuples, and the probe's answers come
    # back through it.
    assert ["MpsTranspose,", 22] in up["members"], (
        "upstream tidied the trailing comma; the transcription should follow")
    print(f"ENUM: {len(up['members'])} _ConvBackend members identical to upstream, "
          f"names and values")


_QUERY_PROBE = r"""
import json
import torch

out = {"is_shim": hasattr(torch._C, "_aten_implemented"), "cases": {}}
CB = torch._C._ConvBackend

SHAPES = {
    "2d":         ((1, 3, 8, 8),    (4, 3, 3, 3),    [1, 1],    [0, 0],    [1, 1],    False, [0, 0],    1),
    "3d":         ((1, 3, 4, 4, 4), (4, 3, 3, 3, 3), [1, 1, 1], [0, 0, 0], [1, 1, 1], False, [0, 0, 0], 1),
    "1d":         ((1, 3, 8),       (4, 3, 3),       [1],       [0],       [1],       False, [0],       1),
    "depthwise":  ((1, 8, 8, 8),    (8, 1, 3, 3),    [1, 1],    [1, 1],    [1, 1],    False, [0, 0],    8),
    "transposed": ((1, 3, 8, 8),    (3, 4, 3, 3),    [1, 1],    [0, 0],    [1, 1],    True,  [0, 0],    1),
    "dilated":    ((1, 3, 9, 9),    (4, 3, 3, 3),    [1, 1],    [0, 0],    [2, 2],    False, [0, 0],    1),
}


def case(name, fn):
    try:
        out["cases"][name] = fn()
    except Exception as exc:
        out["cases"][name] = {"raised": type(exc).__name__, "message": str(exc)}


# --- the query itself, on META tensors: the device this shim's convolution
# --- is traced on, and the one upstream answers `Overrideable` for ---------
for label, (xs, ws, st, pa, di, tr, op, g) in SHAPES.items():
    def ask(xs=xs, ws=ws, st=st, pa=pa, di=di, tr=tr, op=op, g=g):
        x = torch.empty(xs, device="meta")
        w = torch.empty(ws, device="meta")
        b = torch.empty((ws[1] * g if tr else ws[0],), device="meta")
        backend = torch._C._select_conv_backend(
            x, w, bias=b, stride=st, padding=pa, dilation=di,
            transposed=tr, output_padding=op, groups=g)
        fmt = torch._C._conv_determine_backend_memory_format(x, w, backend)
        return {"backend": str(backend), "memory_format": str(fmt)}
    case(f"meta/{label}", ask)

    # `bias=None` takes `fake_impls.py`'s other leg, which passes
    # `bias_sizes` instead of `bias`.
    def ask_nobias(xs=xs, ws=ws, st=st, pa=pa, di=di, tr=tr, op=op, g=g):
        x = torch.empty(xs, device="meta")
        w = torch.empty(ws, device="meta")
        backend = torch._C._select_conv_backend(
            x, w, bias=None, bias_sizes=None, stride=st, padding=pa,
            dilation=di, transposed=tr, output_padding=op, groups=g)
        return {"backend": str(backend)}
    case(f"meta-nobias/{label}", ask_nobias)

# --- `Overrideable`'s memory format is contiguous on EVERY input layout,
# --- including a channels-last one where `Slow2d` answers channels_last.
# --- That is the property this shim's answer rests on, so it is asserted
# --- rather than assumed.
def _cl():
    return torch.empty(1, 8, 8, 3, device="meta").permute(0, 3, 1, 2)


for lbl, build in (("contig", lambda: torch.empty(1, 3, 8, 8, device="meta")),
                   ("channels_last", _cl)):
    case(f"memfmt/{lbl}/Overrideable",
         lambda b=build: str(torch._C._conv_determine_backend_memory_format(
             b(), torch.empty(4, 3, 3, 3, device="meta"), CB.Overrideable)))
    case(f"memfmt/{lbl}/Empty",
         lambda b=build: str(torch._C._conv_determine_backend_memory_format(
             b(), torch.empty(4, 3, 3, 3, device="meta"), CB.Empty)))
    case(f"memfmt/{lbl}/Slow2d",
         lambda b=build: str(torch._C._conv_determine_backend_memory_format(
             b(), torch.empty(4, 3, 3, 3, device="meta"), CB.Slow2d)))

print(json.dumps(out))
"""


def test_select_conv_backend_answers_overrideable_exactly_where_upstream_does():
    """The query's answer, compared with upstream on meta tensors.

    Six convolution shapes x two bias spellings.  Upstream answers
    `_ConvBackend.Overrideable` for all of them on a meta tensor, and
    `torch.contiguous_format` for the memory format -- which is what this
    shim's convolution actually produces (`docs/graph/STRIDE.md` §3.1 measured
    its meta arm contiguous even for a channels-last input, and its dense side
    cannot hold a caller-chosen stride at all).

    Answering `Slow2d` would be the plausible-looking lie: it agrees with
    upstream on a *CPU* tensor and then makes
    `_conv_determine_backend_memory_format` answer `channels_last` for a
    channels-last input, recording a layout this build never produces.  The
    `memfmt/channels_last/*` cases are the measurement that says so, and they
    are why `Slow2d` is refused by name here rather than answered.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim = _run(_QUERY_PROBE, shim=True)
    up = _run(_QUERY_PROBE, shim=False)
    assert set(shim["cases"]) == set(up["cases"])
    same, differ = [], []
    for name in sorted(up["cases"]):
        s, u = shim["cases"][name], up["cases"][name]
        if name.endswith("/Slow2d"):
            continue  # asserted separately below
        (same if s == u else differ).append((name, s, u))
    assert not differ, f"{len(differ)} answers differ from upstream: {differ}"
    assert len(same) >= 16, f"only {len(same)} cases compared"
    print(f"QUERY: {len(same)} answers identical to upstream, across six "
          f"convolution shapes and two bias spellings")


_DENSE_PROBE = r"""
import json
import torch

out = {"is_shim": hasattr(torch._C, "_aten_implemented"), "cases": {}}
CB = torch._C._ConvBackend
w = torch.randn(4, 3, 3, 3)
for lbl, x in (("contig", torch.randn(1, 3, 8, 8)),
               ("channels_last", torch.randn(1, 8, 8, 3).permute(0, 3, 1, 2))):
    for name in ("Overrideable", "Empty", "Slow2d"):
        try:
            out["cases"][f"{lbl}/{name}"] = str(
                torch._C._conv_determine_backend_memory_format(
                    x, w, getattr(CB, name)))
        except Exception as exc:
            out["cases"][f"{lbl}/{name}"] = {
                "raised": type(exc).__name__, "message": str(exc)}
print(json.dumps(out))
"""


def test_the_one_place_this_constant_differs_from_upstream_is_recorded():
    """The divergence, named rather than hidden -- and shown to be unreachable.

    `_conv_determine_backend_memory_format` answers `torch.contiguous_format`
    unconditionally here.  On a **meta** tensor that is upstream's answer too,
    for every backend (the test above).  On a **dense** tensor it is not: a
    channels-last input with a native backend such as `Slow2d` answers
    `channels_last` upstream, because upstream's `Slow2d` kernel really does
    produce that layout and this build has no kernel that can.

    Two things are asserted, and the second is what makes the first
    acceptable:

    1. the divergence exists and is exactly where this docstring says --
       so a later change that removed it would redden this, and a later change
       that *widened* it would too;
    2. it is **unreachable through this shim's own query**, because
       `_select_conv_backend` never names a native backend.  A caller holding
       `Slow2d` did not get it from here.

    Written as a recorded divergence rather than a by-name refusal on purpose.
    The first draft refused, and refusing was measurably worse: it made the
    shim disagree with upstream on the meta path, which is the path
    `torch.export` actually takes.
    """
    if not _available():
        print("SKIP no vendored shim")
        return
    shim = _run(_DENSE_PROBE, shim=True)["cases"]
    up = _run(_DENSE_PROBE, shim=False)["cases"]
    agree = [k for k in sorted(up) if shim[k] == up[k]]
    differ = [(k, shim[k], up[k]) for k in sorted(up) if shim[k] != up[k]]
    assert differ == [("channels_last/Slow2d", "torch.contiguous_format",
                       "torch.channels_last")], (
        f"the divergence is not the one recorded: {differ}")
    assert len(agree) == 5, agree
    # And it cannot be reached through this shim's own selection.
    reached = _run(_QUERY_PROBE, shim=True)["cases"]
    named = {v["backend"] for k, v in reached.items()
             if k.startswith("meta") and isinstance(v, dict) and "backend" in v}
    assert named == {"_ConvBackend.Overrideable"}, named
    print("DIVERGENCE: one pair differs from upstream "
          "(dense channels-last + Slow2d), and this shim's own query never "
          "names a native backend, so nothing reaches it")


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failed else 0)
