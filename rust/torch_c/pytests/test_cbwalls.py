"""Issue #30: the three cpu walls on transformers' continuous-batching path that
sit above the dispatcher -- `torch.mps` memory queries, `pin_memory=True` on cpu
factories, and `torch.Generator()`.

Every comparison runs the shim and upstream in separate subprocesses (the shim
through the vendored tree, upstream through the interpreter's own torch) and
compares element-wise, dtype included. What each decision promises:

  1. `torch.mps.recommended_max_memory()` and `driver_allocated_memory()` are
     real numbers from `MTLDevice`. `current_allocated_memory()` is a count of
     live Metal storage bytes this crate allocated: allocating a known-size
     tensor moves it by exactly that size and freeing it moves it back.
     Where there is no Metal (a non-Apple build) all three refuse by name.
  2. `pin_memory=True` on a cpu factory is accepted and returns an ordinary cpu
     tensor; `is_pinned()` is False. **Upstream on a Mac returns an *mps* tensor
     for the same call** (and segfaults for `torch.tensor(..., pin_memory=True)`);
     `test_upstream_..._returns_an_mps_tensor` keeps that measurement honest. It
     is a deviation in the shim's favour, not an agreement claim -- so the
     values are compared, the device is not.
  3. `torch.Generator()` is an independent stream per generator, and because the
     global cpu RNG was measured to reproduce upstream's mt19937 sequence, a
     seeded generator must too.

Written before the build that would let it pass: against the tree built from
the commit before this one every test here that needs the new code fails.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import _skip  # noqa: E402

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.environ.get(
    "TORCHNATIVE_VENDOR_DIR", os.path.join(_REPO_ROOT, "torchnative", "src", "main")
)


def _run_side(script: str, shim: bool, timeout: int = 300) -> dict:
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"{'shim' if shim else 'upstream'} side failed "
            f"(rc={proc.returncode}):\n{proc.stderr[-3000:]}"
        )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    assert out.pop("is_shim") is shim, "the side imported the wrong torch"
    return out


def _both(script: str, upstream_script: str = None) -> tuple:
    return (
        _run_side(script, shim=True),
        _run_side(upstream_script or script, shim=False),
    )


_HEAD = r"""
import json, torch
IS_SHIM = hasattr(torch._C, "_aten_implemented")
out = {"is_shim": IS_SHIM}
"""
_TAIL = "\nprint(json.dumps(out))\n"


def _mps_available() -> bool:
    out = _run_side(
        _HEAD + 'out["mps"] = bool(torch.backends.mps.is_available())' + _TAIL, shim=True
    )
    return out["mps"]


# ---------------------------------------------------------------------------
# 2. pin_memory=True on cpu
# ---------------------------------------------------------------------------

_PIN_SCRIPT = _HEAD + r"""
makers = {
    "zeros": lambda: torch.zeros(4, pin_memory=True),
    "ones_cpu": lambda: torch.ones(2, 3, device="cpu", pin_memory=True),
    "full": lambda: torch.full((3,), 2.5, pin_memory=True),
    # (`arange` and float64 are left out: upstream's mps arange has no kernel and mps has no
    # float64 -- upstream cannot answer those, which is the wall the shim removes)
    "zeros_like": lambda: torch.zeros_like(torch.ones(2, 2), pin_memory=True),
    "empty_shape": lambda: torch.empty(2, 5, pin_memory=True),
}
res = {}
for name, make in makers.items():
    t = make()
    entry = {"device": str(t.device), "dtype": str(t.dtype), "shape": list(t.shape)}
    if name != "empty_shape":
        entry["values"] = t.cpu().tolist()
    if IS_SHIM:
        entry["is_pinned"] = t.is_pinned()  # upstream segfaults on an mps tensor
    res[name] = entry
out["res"] = res
""" + _TAIL


def test_pin_memory_true_on_cpu_factories_gives_unpinned_cpu_tensors_with_upstream_values():
    shim, upstream = _both(_PIN_SCRIPT)
    assert set(shim["res"]) == set(upstream["res"])
    for name, s in shim["res"].items():
        u = upstream["res"][name]
        assert s["device"] == "cpu", (name, s)
        assert s["is_pinned"] is False, (name, s)
        assert (s["dtype"], s["shape"]) == (u["dtype"], u["shape"]), (name, s, u)
        assert s.get("values") == u.get("values"), (name, s, u)


def test_pin_memory_true_through_rand_and_randn_is_accepted_on_cpu_and_keeps_the_stream():
    # `rand`/`randn` reach `empty` through `_empty_kwargs`, then draw. The draws
    # must be the unpinned ones: pinning changes where bytes live, not what they
    # are. float64 is the dtype Gemstone draws on cpu and mps cannot hold.
    script = _HEAD + r"""
torch.manual_seed(3); c = torch.randn(3, pin_memory=True)
torch.manual_seed(3); d = torch.randn(3)
torch.manual_seed(4); e = torch.rand(3, dtype=torch.float64, pin_memory=True)
torch.manual_seed(4); f = torch.rand(3, dtype=torch.float64)
out["devs"] = [str(t.device) for t in (c, e)]
out["pinned"] = [t.is_pinned() for t in (c, e)]
out["randn_same_as_unpinned"] = c.tolist() == d.tolist()
out["rand64_same_as_unpinned"] = e.tolist() == f.tolist()
out["e_dtype"] = str(e.dtype)
""" + _TAIL
    s = _run_side(script, shim=True)
    assert s["devs"] == ["cpu"] * 2, s
    assert s["pinned"] == [False] * 2, s
    assert s["randn_same_as_unpinned"] is True and s["rand64_same_as_unpinned"] is True, s
    assert s["e_dtype"] == "torch.float64", s


def test_pin_memory_true_through_tensor_and_new_tensor_is_accepted_on_cpu():
    # `torch.tensor` and `Tensor.new_tensor` carry their own Python refusal.
    # Upstream segfaults on `torch.tensor(..., pin_memory=True)` on a Mac (and has
    # no `pin_memory` keyword on `new_tensor`), so this is shim-only.
    script = _HEAD + r"""
a = torch.tensor([1.0, 2.0], pin_memory=True)
b = torch.ones(2).new_tensor([3.0], pin_memory=True)
out["devs"] = [str(t.device) for t in (a, b)]
out["pinned"] = [t.is_pinned() for t in (a, b)]
out["values"] = [a.tolist(), b.tolist()]
""" + _TAIL
    s = _run_side(script, shim=True)
    assert s["devs"] == ["cpu"] * 2 and s["pinned"] == [False] * 2, s
    assert s["values"] == [[1.0, 2.0], [3.0]], s


def test_pin_memory_true_on_mps_is_refused_by_name_not_served_unpinned():
    if not _mps_available():
        _skip.skip("no mps device: the non-cpu pin refusal cannot be exercised")
        return
    script = _HEAD + r"""
msgs = []
for call in (
    lambda: torch.zeros(2, device="mps", pin_memory=True),
    lambda: torch.zeros_like(torch.ones(2, device="mps"), pin_memory=True),
    lambda: torch.tensor([1.0], device="mps", pin_memory=True),
):
    try:
        call()
        msgs.append(None)
    except NotImplementedError as e:
        msgs.append(str(e))
out["msgs"] = msgs
""" + _TAIL
    s = _run_side(script, shim=True)
    for msg in s["msgs"]:
        assert msg is not None and "pin_memory" in msg, s


def test_upstream_cpu_pin_memory_returns_an_mps_tensor_on_a_mac():
    # The measured deviation `_pin_memory_is_cpu` documents. If upstream stops
    # doing this the documentation is stale and this goes red, which is the point.
    if not _mps_available():
        _skip.skip("no mps device: upstream pins through cuda here, not mps")
        return
    u = _run_side(
        _HEAD + 'out["dev"] = str(torch.zeros(2, device="cpu", pin_memory=True).device)' + _TAIL,
        shim=False,
    )
    assert u["dev"].startswith("mps"), u


def test_is_pinned_is_false_for_every_shim_tensor_and_takes_no_claim_on_values():
    script = _HEAD + r"""
out["r"] = [torch.ones(2).is_pinned(), torch.zeros(1, dtype=torch.int64).is_pinned()]
""" + _TAIL
    assert _run_side(script, shim=True)["r"] == [False, False]


# ---------------------------------------------------------------------------
# 1. torch.mps memory queries
# ---------------------------------------------------------------------------

_MEM_SCRIPT = _HEAD + r"""
import gc
mps = torch.mps
out["rec"] = mps.recommended_max_memory()
out["types"] = [type(f()).__name__ for f in
                (mps.recommended_max_memory, mps.driver_allocated_memory, mps.current_allocated_memory)]
"""


def test_recommended_max_memory_is_the_mtldevice_value_and_equals_upstreams():
    if not _mps_available():
        _skip.skip("no mps device")
        return
    shim, upstream = _both(_MEM_SCRIPT + _TAIL)
    assert shim["types"] == ["int", "int", "int"], shim
    assert shim["rec"] > 0, shim
    # Both read `MTLDevice.recommendedMaxWorkingSetSize` of the same GPU.
    assert shim["rec"] == upstream["rec"], (shim["rec"], upstream["rec"])


_ALLOC_SCRIPT = _HEAD + r"""
import gc
mps = torch.mps
def settle():
    # an in-flight command buffer holds its operands; reading a byte back waits it out
    torch.zeros(1, device="mps").cpu()
    gc.collect()
settle()
base_cur = mps.current_allocated_memory()
base_drv = mps.driver_allocated_memory()
x = torch.zeros(2 ** 22, dtype=torch.float32, device="mps")   # exactly 16 MiB
settle_x = mps.current_allocated_memory()
torch.zeros(1, device="mps").cpu()
with_x = mps.current_allocated_memory()
drv_with_x = mps.driver_allocated_memory()
y = torch.zeros(2 ** 20, dtype=torch.float32, device="mps")   # exactly 4 MiB
torch.zeros(1, device="mps").cpu()
with_xy = mps.current_allocated_memory()
del x
settle()
without_x = mps.current_allocated_memory()
del y
settle()
end = mps.current_allocated_memory()
out["d_x"] = with_x - base_cur
out["d_y"] = with_xy - with_x
out["freed_x"] = with_xy - without_x
out["back_to_base"] = end - base_cur
out["drv_delta"] = drv_with_x - base_drv
""" + _TAIL


def test_current_allocated_memory_moves_by_exactly_the_size_of_a_known_tensor():
    if not _mps_available():
        _skip.skip("no mps device")
        return
    shim, upstream = _both(_ALLOC_SCRIPT)
    mib = 1 << 20
    assert shim["d_x"] == 16 * mib, shim
    assert shim["d_y"] == 4 * mib, shim
    assert shim["freed_x"] == 16 * mib, shim
    assert shim["back_to_base"] == 0, shim
    # Upstream's own allocator agrees for sizes that need no rounding, which is
    # why these sizes were chosen. For a size that is not a power of two the two
    # round differently (ours to the next power of two, upstream's to its own
    # pages): not asserted, and documented on `_mps_currentAllocatedMemory`.
    for key in ("d_x", "d_y", "freed_x", "back_to_base"):
        assert shim[key] == upstream[key], (key, shim, upstream)


def test_driver_allocated_memory_covers_the_allocation_it_was_asked_about():
    if not _mps_available():
        _skip.skip("no mps device")
        return
    shim = _run_side(_ALLOC_SCRIPT, shim=True)
    # `MTLDevice.currentAllocatedSize` includes the 16 MiB buffer; the exact
    # figure also includes pool slack and is not asserted.
    assert shim["drv_delta"] >= 16 * (1 << 20), shim


def test_mps_memory_queries_refuse_by_name_where_there_is_no_metal():
    out = _run_side(
        _HEAD + 'out["built"] = bool(torch._C._mps_probe()["built"])' + _TAIL, shim=True
    )
    if out["built"]:
        _skip.skip("this build has Metal; the no-Metal refusal needs a non-Apple build")
        return
    script = _HEAD + r"""
msgs = {}
for name in ("recommended_max_memory", "driver_allocated_memory", "current_allocated_memory"):
    try:
        getattr(torch.mps, name)()
        msgs[name] = None
    except RuntimeError as e:
        msgs[name] = str(e)
out["avail"] = bool(torch.backends.mps.is_available())
out["msgs"] = msgs
""" + _TAIL
    s = _run_side(script, shim=True)
    assert s["avail"] is False, s
    for name, msg in s["msgs"].items():
        assert msg is not None and "no Metal" in msg and "_mps_" in msg, (name, s)


# ---------------------------------------------------------------------------
# 3. torch.Generator()
# ---------------------------------------------------------------------------

_GEN_SCRIPT = _HEAD + r"""
res = {}
res["initial_seed_fresh"] = torch.Generator().initial_seed()
for s in (0, 7, 1234):
    def g():
        return torch.Generator().manual_seed(s)
    res[f"rand_{s}"] = torch.rand(5, generator=g()).tolist()
    res[f"rand64_{s}"] = torch.rand(5, dtype=torch.float64, generator=g()).tolist()
    res[f"randn_{s}"] = torch.randn(20, generator=g()).tolist()
    res[f"randn64_{s}"] = torch.randn(5, dtype=torch.float64, generator=g()).tolist()
    res[f"randint_{s}"] = torch.randint(0, 100, (6,), generator=g()).tolist()
    res[f"randint_low_{s}"] = torch.randint(3, 40, (6,), generator=g()).tolist()
    res[f"randperm_{s}"] = torch.randperm(7, generator=g()).tolist()
    res[f"normal__{s}"] = torch.empty(5).normal_(generator=g()).tolist()
    res[f"uniform__{s}"] = torch.empty(5).uniform_(2, 3, generator=g()).tolist()
    p = torch.tensor([0.1, 0.2, 0.3, 0.4])
    res[f"multinomial_rep_{s}"] = torch.multinomial(p, 8, replacement=True, generator=g()).tolist()
    res[f"multinomial_{s}"] = torch.multinomial(p, 3, generator=g()).tolist()
    res[f"multinomial2d_{s}"] = torch.multinomial(
        torch.tensor([[0.1, 0.9], [0.5, 0.5]]), 1, generator=g()).tolist()
    res[f"normal_fn_{s}"] = torch.normal(0.0, 1.0, (4,), generator=g()).tolist()
    res[f"rand_like_{s}"] = torch.rand_like(torch.ones(3), generator=g()).tolist()
    res[f"bernoulli__{s}"] = torch.empty(6).bernoulli_(0.5, generator=g()).tolist()
out["res"] = res
""" + _TAIL


def test_seeded_generators_reproduce_upstream_draws_for_every_op_named_in_the_issue():
    shim, upstream = _both(_GEN_SCRIPT)
    assert set(shim["res"]) == set(upstream["res"])
    bad = [k for k in upstream["res"] if shim["res"][k] != upstream["res"][k]]
    assert not bad, {k: (shim["res"][k], upstream["res"][k]) for k in bad}


_INTERLEAVE_SCRIPT = _HEAD + r"""
def draws(g):
    return [torch.rand(3, generator=g).tolist(), torch.randn(3, generator=g).tolist(),
            torch.randint(0, 50, (3,), generator=g).tolist(),
            torch.multinomial(torch.tensor([.2, .3, .5]), 4, replacement=True, generator=g).tolist()]
# alone
alone = draws(torch.Generator().manual_seed(11))
# the same seed with another generator and the global stream drawing in between
a = torch.Generator().manual_seed(11)
b = torch.Generator().manual_seed(99)
torch.manual_seed(5)
res_a = []
steps = [lambda: torch.rand(3, generator=a).tolist(), lambda: torch.randn(3, generator=a).tolist(),
         lambda: torch.randint(0, 50, (3,), generator=a).tolist(),
         lambda: torch.multinomial(torch.tensor([.2, .3, .5]), 4, replacement=True, generator=a).tolist()]
for st in steps:
    torch.rand(7, generator=b)         # another generator draws
    torch.randn(2)                      # the global stream draws
    torch.manual_seed(5)                # and is reseeded
    res_a.append(st())
out["alone"] = alone
out["interleaved"] = res_a
# the global stream is untouched by generator draws: seed it, draw only from a generator, compare
torch.manual_seed(21)
before = torch.rand(4).tolist()
torch.manual_seed(21)
torch.rand(100, generator=torch.Generator().manual_seed(1))
after = torch.rand(4).tolist()
out["global_untouched"] = (before == after)
# and the reverse: generator unaffected by global draws
g1 = torch.Generator().manual_seed(8)
x1 = torch.rand(5, generator=g1).tolist()
g2 = torch.Generator().manual_seed(8)
torch.rand(1000); torch.randn(1000)
x2 = torch.rand(5, generator=g2).tolist()
out["generator_untouched_by_global"] = (x1 == x2)
# two generators with one seed are two streams: advancing one does not move the other
h1 = torch.Generator().manual_seed(4); h2 = torch.Generator().manual_seed(4)
torch.rand(10, generator=h1)
out["independent_state"] = (torch.rand(3, generator=h2).tolist()
                            == torch.rand(3, generator=torch.Generator().manual_seed(4)).tolist())
""" + _TAIL


def test_a_requests_own_generator_draws_the_same_numbers_whatever_other_requests_do():
    shim, upstream = _both(_INTERLEAVE_SCRIPT)
    assert shim["alone"] == upstream["alone"], (shim["alone"], upstream["alone"])
    assert shim["interleaved"] == shim["alone"], shim
    assert upstream["interleaved"] == upstream["alone"], upstream
    assert shim["global_untouched"] is True and shim["generator_untouched_by_global"] is True, shim
    assert shim["independent_state"] is True, shim
    assert (shim["global_untouched"], shim["generator_untouched_by_global"],
            shim["independent_state"]) == (
        upstream["global_untouched"], upstream["generator_untouched_by_global"],
        upstream["independent_state"]), (shim, upstream)


def test_generator_seed_state_matches_upstream_and_unseeded_default_is_reproducible():
    script = _HEAD + r"""
g = torch.Generator()
out["fresh"] = g.initial_seed()
out["after_seed"] = (g.manual_seed(77) is g, g.initial_seed())
out["neg"] = torch.Generator().manual_seed(-1).initial_seed()
out["fresh_draws_equal"] = (torch.rand(3, generator=torch.Generator()).tolist()
                            == torch.rand(3, generator=torch.Generator()).tolist())
out["device"] = str(g.device)
""" + _TAIL
    shim, upstream = _both(script)
    assert shim["fresh"] == 67280421310721, shim
    assert shim == upstream, (shim, upstream)


def test_generators_that_do_not_exist_here_refuse_by_name():
    script = _HEAD + r"""
res = {}
try:
    torch.Generator(device="cuda")
    res["cuda"] = None
except NotImplementedError as e:
    res["cuda"] = str(e)
class NotAGenerator:  # claims the API, owns no stream
    pass
try:
    torch.empty(2).uniform_(generator=NotAGenerator())
    res["foreign"] = None
except (NotImplementedError, TypeError) as e:
    res["foreign"] = str(e)
try:
    torch.Generator().get_state()
    res["get_state"] = None
except NotImplementedError as e:
    res["get_state"] = str(e)
out["res"] = res
""" + _TAIL
    s = _run_side(script, shim=True)["res"]
    assert s["cuda"] is not None and "Generator" in s["cuda"] and "cpu" in s["cuda"], s
    assert s["foreign"] is not None, s
    assert s["get_state"] is not None and "get_state" in s["get_state"], s


def test_freeing_generators_does_not_disturb_live_ones():
    script = _HEAD + r"""
import gc
live = torch.Generator().manual_seed(3)
first = torch.rand(2, generator=live).tolist()
for i in range(200):
    torch.rand(1, generator=torch.Generator().manual_seed(i))
gc.collect()
second = torch.rand(2, generator=live).tolist()
ref = torch.Generator().manual_seed(3)
out["same"] = (first + second) == torch.rand(4, generator=ref).tolist()
""" + _TAIL
    shim, upstream = _both(script)
    assert shim["same"] is True and upstream["same"] is True, (shim, upstream)


def _main():
    failures = _skip.run_tests(
        [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")],
        suite="test_cbwalls",
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
