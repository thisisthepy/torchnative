"""`Generator.get_state` / `set_state` and `torch.get_rng_state` /
`set_rng_state` give upstream's own 5056-byte blob.

Gemstone (LogitAI/Gemstone #177, run 39023653125 on 0.1.0b6) stopped on
`Generator.get_state`, which the shim refused by name. The CPU generator in
`rng.rs` is a port of upstream's MT19937 and keeps upstream's field names, so
the blob is upstream's `CPUGeneratorImplState` written field by field:

    offset  size  field
         0     8  the_initial_seed (u64)
         8     4  left (i32)
        12     4  seeded (i32)
        16     8  next (u64)
        24  4992  state[624], each u32 widened to u64
      5016     8  normal_x (always 0.0)
      5024     8  normal_y (the cached Box-Muller sample, or 0.0)
      5032     8  normal_rho (always 0.0)
      5040     4  normal_is_valid (i32)
      5044     4  padding (zero)
      5048     4  next_float_normal_sample (f32)
      5052     1  is_next_float_normal_sample_valid
      5053     3  padding (zero)

Every comparison runs the shim and upstream (the interpreter's own torch,
2.13.0 when this was written) in separate subprocesses and compares the bytes,
the draws after a restore, and the refusals' type and wording.

* *agrees*: the blob is byte-equal to upstream's after seeding, after draws
  that leave a Box-Muller sample cached, for the default generator and for a
  `torch.Generator()`; a restore replays upstream's draws; a blob taken from one
  side restores the other (the bytes are the interop format).
* the refusals upstream makes (wrong size, wrong dtype, not a tensor) carry
  upstream's exception type and message.
* *diverges, by name*: a blob whose next draw would read outside the 624-word
  state (for example `left == 0`). Upstream 2.13.0 accepts it and its next draw
  reads past the array, which is undefined behaviour in C++; the shim refuses
  it naming MT19937 instead.

What this cannot find: the float half of the normal cache. No CPU kernel
upstream fills it (`normal_kernel` draws doubles for every dtype), so every
blob here has it zero on both sides; a wrong offset for it would only show on
a blob crafted by hand, which the set-then-get case below does check.
"""

import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.join(str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())), "tests", "_support"))
import _skip  # noqa: E402

_REPO_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.environ.get(
    "TORCHNATIVE_VENDOR_DIR", os.path.join(_REPO_ROOT, "torchnative", "python")
)


def _run_side(script: str, shim: bool, stdin: str = "", timeout: int = 300) -> dict:
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"  # VENDOR.md wall 1
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", script],
        input=stdin,
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


def _upstream_available() -> bool:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    proc = subprocess.run([sys.executable, "-c", "import torch"],
                          capture_output=True, text=True, env=env, timeout=300)
    return proc.returncode == 0


def _both(script: str) -> tuple:
    if not _upstream_available():
        raise _skip.Skip("upstream torch is not importable by this interpreter: no oracle")
    return _run_side(script, shim=True), _run_side(script, shim=False)


_HEAD = r"""
import json, struct, sys, torch
IS_SHIM = hasattr(torch._C, "_aten_implemented")
out = {"is_shim": IS_SHIM}

def blob(t):
    return bytes(t.tolist()).hex()

def err(f):
    try:
        f()
        return None
    except Exception as e:
        return [type(e).__name__, str(e).splitlines()[0]]

def draws(g=None):
    kw = {} if g is None else {"generator": g}
    return {
        "rand": torch.rand(5, **kw).tolist(),
        "randn64": torch.randn(3, dtype=torch.float64, **kw).tolist(),
        "randn32": torch.randn(17, **kw).tolist(),
        "randint": torch.randint(0, 1000, (4,), **kw).tolist(),
        "randperm": torch.randperm(7, **kw).tolist(),
        "normal_": torch.empty(5, dtype=torch.float64).normal_(**kw).tolist(),
    }
"""
_TAIL = "\nprint(json.dumps(out))\n"


def test_the_blob_is_upstreams_bytes_after_seeding_and_after_draws():
    script = _HEAD + r"""
res = {}
for seed in (0, 1, 7, 2**40 + 7, 67280421310721):
    torch.manual_seed(seed)
    s = torch.get_rng_state()
    res[f"default {seed}"] = [str(s.dtype), list(s.shape), str(s.device), blob(s)]
    g = torch.Generator().manual_seed(seed)
    res[f"gen {seed}"] = blob(g.get_state())
    # `randn(3, float64)` takes the serial Box-Muller path and leaves one
    # sample cached on the generator: normal_y / normal_is_valid are set.
    torch.randn(3, dtype=torch.float64, generator=g)
    res[f"gen {seed} cached"] = blob(g.get_state())
    draws(g)
    res[f"gen {seed} after draws"] = blob(g.get_state())
res["fresh generator"] = blob(torch.Generator().get_state())
out["res"] = res
""" + _TAIL
    shim, upstream = _both(script)
    assert set(shim["res"]) == set(upstream["res"])
    for key, value in shim["res"].items():
        assert value == upstream["res"][key], (
            f"{key}: the shim's blob differs from upstream's"
            + (f" (first differing byte {_first_diff(value, upstream['res'][key])})"
               if isinstance(value, str) else f": {value[:3]} vs {upstream['res'][key][:3]}"))


def _first_diff(a, b):
    a, b = bytes.fromhex(a), bytes.fromhex(b)
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


@_skip.known_x86_64_linux_divergence(
    "#40",
    "randn32 (n=17) on x86_64 uses AVX2 NormalFill16<float, true> (sincos256_ps/log256_ps), "
    "giving last-bit differences from the scalar Box-Muller path (docs/numerics/RNG.md §3.3, §6)",
)
def test_a_restore_replays_upstreams_draws():
    script = _HEAD + r"""
res = {}
torch.manual_seed(11)
torch.randn(3, dtype=torch.float64)          # leave a cached sample behind
saved = torch.get_rng_state()
first = draws()
torch.set_rng_state(saved)
res["default replay"] = [first, draws()]

g = torch.Generator().manual_seed(5)
torch.randn(1, dtype=torch.float64, generator=g)
st = g.get_state()
a = draws(g)
g.set_state(st)
res["generator replay"] = [a, draws(g)]

# A blob moves between generators: default -> a fresh Generator.
torch.manual_seed(99)
torch.rand(10)
h = torch.Generator()
h.set_state(torch.get_rng_state())
res["moved"] = [draws(h), draws()]

# get_state hands out a copy: writing into it does not reach the generator.
k = torch.Generator().manual_seed(3)
s = k.get_state()
s.zero_()
res["copy"] = blob(k.get_state()) == blob(torch.Generator().manual_seed(3).get_state())

# set_state returns the generator, as upstream's does.
res["returns self"] = k.set_state(k.get_state()) is k

with torch.random.fork_rng():
    torch.manual_seed(1)
    torch.rand(4)
res["fork_rng"] = torch.rand(3).tolist()
out["res"] = res
""" + _TAIL
    # fork_rng restores the default generator to where it was before the
    # block; seed the parent first so both sides start from the same state.
    script = script.replace("with torch.random.fork_rng():",
                            "torch.manual_seed(42)\nwith torch.random.fork_rng():", 1)
    shim, upstream = _both(script)
    s, u = shim["res"], upstream["res"]
    assert s["default replay"][0] == s["default replay"][1], s["default replay"]
    assert s["generator replay"][0] == s["generator replay"][1], s["generator replay"]
    assert s["moved"][0] == s["moved"][1], s["moved"]
    assert s["copy"] is True and s["returns self"] is True, s
    for key in u:
        assert s[key] == u[key], (key, s[key], u[key])


@_skip.known_x86_64_linux_divergence(
    "#40",
    "randn32 (n=17) on x86_64 uses AVX2 NormalFill16<float, true> (sincos256_ps/log256_ps), "
    "giving last-bit differences from the scalar Box-Muller path (docs/numerics/RNG.md §3.3, §6)",
)
def test_a_blob_from_upstream_restores_the_shim_and_the_reverse():
    # The bytes are the interop format: a state saved by real torch (a
    # checkpoint's `rng_state`) has to restore here and replay its stream.
    make = _HEAD + r"""
g = torch.Generator().manual_seed(1234)
torch.randn(5, dtype=torch.float64, generator=g)   # odd count: a sample stays cached
torch.rand(3, generator=g)
out["blob"] = blob(g.get_state())
out["next"] = draws(g)
""" + _TAIL
    use = _HEAD + r"""
data = bytes.fromhex(sys.stdin.read().strip())
g = torch.Generator()
g.set_state(torch.tensor(list(data), dtype=torch.uint8))
out["next"] = draws(g)
out["blob"] = blob(torch.Generator().set_state(torch.tensor(list(data), dtype=torch.uint8)).get_state())
""" + _TAIL
    if not _upstream_available():
        raise _skip.Skip("upstream torch is not importable by this interpreter: no oracle")
    up = _run_side(make, shim=False)
    sh = _run_side(make, shim=True)
    into_shim = _run_side(use, shim=True, stdin=up["blob"])
    into_upstream = _run_side(use, shim=False, stdin=sh["blob"])
    assert into_shim["next"] == up["next"], "an upstream blob did not replay upstream's stream here"
    assert into_shim["blob"] == up["blob"], "set_state then get_state did not give the same bytes back"
    assert into_upstream["next"] == sh["next"], "upstream could not replay a blob the shim saved"


def test_refusals_carry_upstreams_type_and_wording():
    script = _HEAD + r"""
g = torch.Generator().manual_seed(0)
st = g.get_state()
import numpy
out["res"] = {
    "short": err(lambda: g.set_state(torch.zeros(10, dtype=torch.uint8))),
    "legacy 5048": err(lambda: g.set_state(st[:5048].clone())),
    "float": err(lambda: g.set_state(torch.zeros(5056))),
    "int64": err(lambda: g.set_state(torch.zeros(5056, dtype=torch.int64))),
    "numpy": err(lambda: g.set_state(numpy.zeros(5056, dtype=numpy.uint8))),
    "list": err(lambda: g.set_state([0] * 5056)),
    "2d accepted": err(lambda: g.set_state(st.clone().view(2, 2528))),
    "default short": err(lambda: torch.set_rng_state(torch.zeros(3, dtype=torch.uint8))),
}
""" + _TAIL
    shim, upstream = _both(script)
    assert shim["res"] == upstream["res"], (shim["res"], upstream["res"])


def test_a_blob_that_would_read_outside_the_state_is_refused_by_name():
    # Upstream 2.13.0 accepts these and reads past `state[623]` on the next
    # draw. The shim refuses them instead; the message names the engine.
    script = _HEAD + r"""
g = torch.Generator().manual_seed(0)
st = g.get_state()
res = {}
def with_fields(left, nxt):
    b = bytearray(bytes(st.tolist()))
    struct.pack_into("<i", b, 8, left)
    struct.pack_into("<Q", b, 16, nxt)
    return torch.tensor(list(b), dtype=torch.uint8)
for name, left, nxt in (("left 0", 0, 0), ("left -5", -5, 0), ("next 625", 2, 625),
                        ("past the end", 10, 620), ("left 626", 626, 0)):
    res[name] = err(lambda: g.set_state(with_fields(left, nxt)))
# The shapes a real state has stay accepted: freshly seeded (left 1, next 0)
# and mid-stream (left + next == 624). `left == 1` twists before it reads, so
# any `next` is in bounds there; otherwise the draws before the next twist
# read state[next .. next + left - 2].
res["fresh ok"] = err(lambda: g.set_state(with_fields(1, 0)))
res["left 1 any next ok"] = err(lambda: g.set_state(with_fields(1, 9999)))
res["left 625 ok"] = err(lambda: g.set_state(with_fields(625, 0)))
res["mid ok"] = err(lambda: g.set_state(with_fields(124, 500)))
res["end ok"] = err(lambda: g.set_state(with_fields(1, 623)))
res["still draws"] = torch.rand(2, generator=g.set_state(with_fields(124, 500))).tolist()
out["res"] = res
""" + _TAIL
    shim = _run_side(script, shim=True)["res"]
    for name in ("left 0", "left -5", "next 625", "past the end", "left 626"):
        assert shim[name] is not None, f"{name}: accepted a blob whose next draw reads outside the state"
        assert shim[name][0] == "RuntimeError" and "MT19937" in shim[name][1], (name, shim[name])
    for name in ("fresh ok", "left 1 any next ok", "left 625 ok", "mid ok", "end ok"):
        assert shim[name] is None, (name, shim[name])
    if _upstream_available():
        # Upstream accepts every blob above (it draws from none of the bad
        # ones); only the valid state's draws are compared.
        upstream = _run_side(script, shim=False)["res"]
        assert shim["still draws"] == upstream["still draws"], (shim["still draws"], upstream["still draws"])


def test_set_state_seeded_from_a_hand_made_blob_reads_every_field_back():
    # The float normal slot is never filled by a CPU kernel; write it by hand
    # and read it back, so its offset is checked at least through the shim's
    # own round trip and against upstream's.
    script = _HEAD + r"""
torch.manual_seed(0)
b = bytearray(bytes(torch.get_rng_state().tolist()))
struct.pack_into("<d", b, 5024, 0.625)   # normal_y
struct.pack_into("<i", b, 5040, 1)       # normal_is_valid
struct.pack_into("<f", b, 5048, 1.5)     # next_float_normal_sample
b[5052] = 1                              # is_next_float_normal_sample_valid
g = torch.Generator()
g.set_state(torch.tensor(list(b), dtype=torch.uint8))
back = bytes(g.get_state().tolist())
out["same"] = back == bytes(b)
out["fields"] = [struct.unpack_from("<d", back, 5024)[0], struct.unpack_from("<i", back, 5040)[0],
                 struct.unpack_from("<f", back, 5048)[0], back[5052]]
# The cached double is what the next serial normal returns first.
out["first normal"] = torch.randn(1, dtype=torch.float64, generator=g).tolist()
""" + _TAIL
    shim, upstream = _both(script)
    assert shim == upstream, (shim, upstream)


def _main():
    failures = _skip.run_tests(
        [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")],
        suite="test_rngstate",
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
