"""`torch.int8`: that the build that has it is shippable, and that its answers are upstream's.

docs/numerics/INT8.md is the specification. Two claims, graded separately.

**The fork builds anywhere.** `int8` needs `DType::I8`, which no released
`candle-core` has, so `rust/torch_c/Cargo.toml` patches candle. The first landing
pointed `[patch.crates-io]` at an absolute path on one developer's machine, and
the patch file carried in `vendor/` did not even apply -- the tree actually
built was a hand-fixed copy with Metal refusals the patch did not contain. The
tests below hold down each link that failure broke: the patch path is relative
and inside the repository; it is the same directory `vendor/vendor_candle.sh`
writes; that directory is the sha256-pinned published crate plus the patch,
byte for byte; and the check that says so can say no.

What these cannot see: whether a machine without this one's cargo cache can
download the crate (the check reads the cache first), and whether the CUDA
backend compiles with `I8` -- nothing here has an NVIDIA toolchain.

**The answers agree.** Exact equality against upstream torch in the same
interpreter -- integers have no tolerance to widen. The overflow edges are the
point: an implementation that silently computes in a wider type gives `128` for
`127 + 1` where upstream wraps to `-128`, and nothing else in the suite would see it.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import tomllib

from test_shim import _C

try:
    import torch as _upstream
except ImportError:  # pragma: no cover
    _upstream = None

_CRATE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_REPO_ROOT = os.path.abspath(os.path.join(_CRATE_DIR, "..", ".."))
_SCRIPT = os.path.join(_REPO_ROOT, "vendor", "vendor_candle.sh")
_FORK = os.path.join(_REPO_ROOT, "vendor", "candle-core")


# --------------------------------------------------------------------------
# the fork
# --------------------------------------------------------------------------


def _run_script(*args, env_extra=None):
    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.run(["sh", _SCRIPT, *args], capture_output=True, text=True,
                          env=env, timeout=300)


def test_the_candle_patch_path_is_relative_and_is_the_tree_the_script_writes():
    with open(os.path.join(_CRATE_DIR, "Cargo.toml"), "rb") as f:
        manifest = tomllib.load(f)
    path = manifest["patch"]["crates-io"]["candle-core"]["path"]
    assert not os.path.isabs(path), (
        "[patch.crates-io] candle-core is the absolute path %r, which exists on "
        "one machine. Every other checkout fails to build." % path)
    resolved = os.path.normpath(os.path.join(_CRATE_DIR, path))
    # Not merely "inside the repo": the SAME directory `--check` verifies. A
    # patch pointing at a second copy would let the check pass on a tree the
    # build never reads.
    assert resolved == _FORK, (resolved, _FORK)
    with open(os.path.join(resolved, "Cargo.toml"), "rb") as f:
        package = tomllib.load(f)["package"]
    assert (package["name"], package["version"]) == ("candle-core", "0.11.0"), package


def test_the_lock_resolves_candle_core_from_the_fork_not_the_registry():
    """A `[patch]` cargo did not use is only a warning. The lock is where it shows:
    a patched package has no `source`, a registry one does."""
    with open(os.path.join(_CRATE_DIR, "Cargo.lock"), "rb") as f:
        lock = tomllib.load(f)
    entries = [p for p in lock["package"] if p["name"] == "candle-core"]
    assert len(entries) == 1, entries
    assert entries[0]["version"] == "0.11.0", entries[0]
    assert "source" not in entries[0], (
        "Cargo.lock resolves candle-core from %s -- the fork is not in the build"
        % entries[0]["source"])


def test_the_committed_fork_is_the_pinned_crate_plus_the_patch():
    proc = _run_script("--check")
    assert proc.returncode == 0, "exit %d\n%s\n%s" % (proc.returncode, proc.stdout, proc.stderr)
    assert ("sha256 5ecb245093b0f791b89d3420c3df9c6d49c60ab63ba54db896bf8a3baf486706 verified"
            in proc.stdout), proc.stdout
    assert "byte for byte" in proc.stdout, proc.stdout


def test_the_fork_check_refuses_a_tree_that_drifted_from_the_patch():
    """Without this the test above could be a script that exits 0."""
    tmp = tempfile.mkdtemp(prefix="int8-drift-")
    try:
        copy = os.path.join(tmp, "candle-core")
        shutil.copytree(_FORK, copy)
        dtype_rs = os.path.join(copy, "src", "dtype.rs")
        with open(dtype_rs, "a", encoding="utf-8") as f:
            f.write("\n// an edit made to the fork and not to the patch\n")
        proc = _run_script("--check", env_extra={"TORCHNATIVE_CANDLE_DIR": copy})
        assert proc.returncode != 0, proc.stdout
        assert "dtype.rs" in proc.stderr, proc.stderr
    finally:
        shutil.rmtree(tmp)


def test_the_fork_script_refuses_a_crate_that_is_not_the_pinned_one():
    tmp = tempfile.mkdtemp(prefix="int8-crate-")
    try:
        bogus = os.path.join(tmp, "candle-core-0.11.0.crate")
        with open(bogus, "wb") as f:
            f.write(b"not the published crate")
        target = os.path.join(tmp, "must-not-be-written")
        proc = _run_script(env_extra={"TORCHNATIVE_CANDLE_CRATE": bogus,
                                      "TORCHNATIVE_CANDLE_DIR": target})
        assert proc.returncode != 0, proc.stdout
        assert "wrong sha256" in proc.stderr, proc.stderr
        assert not os.path.exists(target), "a refused crate still wrote a tree"
    finally:
        shutil.rmtree(tmp)


# --------------------------------------------------------------------------
# agreement with upstream
# --------------------------------------------------------------------------


def _oracle():
    # Loud rather than skipped: a skip here is counted as `ok`, and this file's
    # agreement half would then pass on an interpreter that checked nothing.
    assert _upstream is not None, "no upstream torch in this interpreter -- the oracle cannot run"
    assert not hasattr(_upstream._C, "_aten_implemented"), "the oracle imported the shim"
    assert hasattr(_C, "_aten_implemented"), "the shim side imported upstream"
    return _upstream


def _make(mod, values, dtype, via="int64"):
    """The same construction on both sides: a tensor of `via`, cast to `dtype`."""
    if mod is _C:
        src = _C._tensor_from_flat([float(v) for v in values], [len(values)], getattr(_C, via))
    else:
        src = mod.tensor(values, dtype=getattr(mod, via))
    return src.to(getattr(mod, dtype))


def _read(t):
    return (str(t.dtype).replace("torch.", ""), t.flatten().tolist())


def _disagreements(cases):
    """`cases` maps a label to `fn(mod) -> tensor`. Exact, dtype included."""
    up = _oracle()
    wrong = []
    for label, fn in cases.items():
        try:
            got = _read(fn(_C))
        except BaseException as e:  # noqa: BLE001
            got = ("refused", "%s: %s" % (type(e).__name__, str(e).splitlines()[0][:120]))
        want = _read(fn(up))
        if got != want:
            wrong.append("  %s: shim %r, upstream %r" % (label, got, want))
    return wrong


def _binary_cases(a_vals, b_vals, dtype="int8"):
    a = lambda m: _make(m, a_vals, dtype)  # noqa: E731
    b = lambda m: _make(m, b_vals, dtype)  # noqa: E731
    return a, b


def test_int8_arithmetic_agrees_with_upstream_on_ordinary_values():
    # No result here leaves [-128, 127] -- the largest is 42 * -3 = -126 -- so a
    # saturating implementation passes this test and only the edges test below
    # catches it. That split was measured, not assumed: an earlier `b` of 3
    # made -50 * 3 overflow and this test went red for the edges' reason.
    a, b = _binary_cases([-50, -7, -1, 0, 1, 3, 42, 50], [2, -2, 5, 1, -1, 7, -3, 2])
    cases = {
        "construct": lambda m: a(m),
        "add": lambda m: a(m) + b(m),
        "sub": lambda m: a(m) - b(m),
        "mul": lambda m: a(m) * b(m),
        "neg": lambda m: -a(m),
        "abs": lambda m: a(m).abs(),
        "div": lambda m: a(m) / b(m),
        "eq": lambda m: a(m) == b(m),
        "lt": lambda m: a(m) < b(m),
        "ge": lambda m: a(m) >= b(m),
        "sum": lambda m: a(m).sum(),
        "max": lambda m: a(m).max(),
        "min": lambda m: a(m).min(),
        "add_scalar": lambda m: a(m) + 3,
        "mul_scalar": lambda m: a(m) * 2,
    }
    wrong = _disagreements(cases)
    assert not wrong, "int8 disagrees with upstream on ordinary values:\n" + "\n".join(wrong)


def test_int8_wraps_at_the_edges_exactly_where_upstream_wraps():
    """Every row here has a different answer in any type wider than int8."""
    a, b = _binary_cases([127, -128, 127, -128, 100, -100, 64], [1, -1, 127, -128, 100, -100, 2])
    cases = {
        "add_overflow": lambda m: a(m) + b(m),
        "sub_overflow": lambda m: a(m) - b(m),
        "mul_overflow": lambda m: a(m) * b(m),
        "neg_of_min": lambda m: -a(m),
        "abs_of_min": lambda m: a(m).abs(),
        "add_scalar_overflow": lambda m: a(m) + 1,
        "sub_scalar_overflow": lambda m: a(m) - 1,
        "mul_scalar_overflow": lambda m: a(m) * 2,
        # The other direction: `sum` widens to int64 upstream, so an
        # accumulator that stays in int8 is the wrong one.
        "sum_does_not_wrap": lambda m: _make(m, [127, 127, 127, 127, -128], "int8").sum(),
        "max_min_edges": lambda m: a(m).max(),
        "min_edge": lambda m: a(m).min(),
    }
    wrong = _disagreements(cases)
    assert not wrong, "int8 overflow disagrees with upstream:\n" + "\n".join(wrong)


def test_casts_into_int8_truncate_exactly_as_upstream_does():
    cases = {
        "from_int64": lambda m: _make(m, [200, 255, 128, -129, 300, -300, 127, -128, 1000000], "int8"),
        "from_int32": lambda m: _make(m, [200, 255, 128, -129, 300, -300], "int8", via="int32"),
        "from_int16": lambda m: _make(m, [32767, -32768, 256, -257, 128], "int8", via="int16"),
        "from_uint8": lambda m: _make(m, [200, 255, 128, 127, 0], "int8", via="uint8"),
        # In range after truncation toward zero; out-of-range float -> int is
        # undefined in C++ and upstream's answer is platform-dependent.
        "from_float32": lambda m: _make(m, [1.9, -1.9, 127.4, -128.9, 0.5, -0.5], "int8", via="float32"),
        "from_float64": lambda m: _make(m, [2.5, -2.5, 99.99, -99.99], "int8", via="float64"),
        "from_bool": lambda m: _make(m, [1, 0, 1], "bool").to(m.int8),
    }
    wrong = _disagreements(cases)
    assert not wrong, "casts to int8 disagree with upstream:\n" + "\n".join(wrong)


def test_casts_out_of_int8_sign_extend_exactly_as_upstream_does():
    src = [127, -128, -1, 0, 1, 100, -100]
    cases = {
        "to_%s" % dt: (lambda dt: lambda m: _make(m, src, "int8").to(getattr(m, dt)))(dt)
        for dt in ("uint8", "int16", "int32", "int64", "float16", "float32", "float64", "bool")
    }
    wrong = _disagreements(cases)
    assert not wrong, "casts from int8 disagree with upstream:\n" + "\n".join(wrong)


def test_int8_with_uint8_promotes_to_int16_in_both_orders():
    """docs/numerics/INT8.md §3: giving `int8` the same rank as `uint8` alone sends
    this pair to float32. Upstream sends it to int16, the smallest type that holds
    both ranges -- so `-1 + 200` is `199` in int16, not `199.0`, and `-128 + 255`
    has no wrap to hide behind."""
    i8 = lambda m: _make(m, [-1, 1, -128, 127], "int8")  # noqa: E731
    u8 = lambda m: _make(m, [200, 1, 255, 0], "uint8")  # noqa: E731
    cases = {
        "int8+uint8": lambda m: i8(m) + u8(m),
        "uint8+int8": lambda m: u8(m) + i8(m),
        "int8*uint8": lambda m: i8(m) * u8(m),
        "uint8-int8": lambda m: u8(m) - i8(m),
        "int8<uint8": lambda m: i8(m) < u8(m),
        "int8+int16": lambda m: i8(m) + _make(m, [1000, -1000, 1, 1], "int16"),
        "int8+int32": lambda m: i8(m) + _make(m, [1, 2, 3, 4], "int32"),
        "int8+int64": lambda m: i8(m) + _make(m, [1, 2, 3, 4], "int64"),
        "int8+float32": lambda m: i8(m) + _make(m, [0.5, 0.5, 0.5, 0.5], "float32", via="float32"),
        "int8+bool": lambda m: i8(m) + _make(m, [1, 0, 1, 0], "bool"),
    }
    wrong = _disagreements(cases)
    assert not wrong, "int8 promotion disagrees with upstream:\n" + "\n".join(wrong)


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
