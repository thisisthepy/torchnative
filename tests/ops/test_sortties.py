"""`sort` / `argsort` / `topk`: index order among tied values (issue #33).

Top-p sampling sorts the probability row, so at the nucleus boundary a tie
decides which token can be sampled. The cpu kernels therefore have to put
tied indices where upstream puts them, not merely return the same values.

The oracle (`sortties_oracle.json`) is upstream torch 2.13.0 on the cpu
(macOS arm64, libc++); `sortties_oracle_linux.json` is the manylinux wheel
(libstdc++, GCC 13.3.1), recorded identically on aarch64 and x86_64 (#81),
recorded by `tests/_support/gen_sortties_oracle.py` over the cells that
`tests/_support/sortties_cells.py` enumerates: sort / argsort / topk, every
direction and flag, dtypes f32 f16 bf16 f64 i32 i64 u8 bool, tie patterns
(all-equal, two-valued, four-valued, +-0.0, NaN, distinct), row lengths
1 2 15 16 17 64 1000 5000, and 1-D / 2-D / 3-D tensors along every dim.
What the data says (derived, not assumed):

  * `stable=True` (sort and argsort) is a stable sort in both directions: ties
    keep ascending index order, NaN counts as the greatest value.
  * `stable=False`, and `argsort`/`sort` called without `stable`, is
    libc++'s `std::sort` over (value, index) pairs. That is stable only
    below 24 elements (insertion sort); from 24 up it is introsort and ties
    come out scrambled. All-equal rows happen to survive.
  * `topk` is libc++'s `std::partial_sort` when `k * 64 <= n`, otherwise
    `std::nth_element` followed (if `sorted`) by `std::sort` of the first
    `k - 1` elements. `sorted=False` leaves the nth_element partition as is.
  * `topk` refuses bool on cpu.

The shim is compared in a second interpreter, because this one may import
upstream. Each cell is one check; the tests report "N of M".
"""

import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.abspath(os.path.join(_HERE, "..", ".."))
_SUPPORT = os.path.join(_REPO, "tests", "_support")
_VENDOR = os.path.join(_REPO, "torchnative", "python")
_ORACLE_LIBCXX = os.path.join(_HERE, "sortties_oracle.json")
_ORACLE_LIBSTDCXX = os.path.join(_HERE, "sortties_oracle_linux.json")
_ORACLE_MSVC = os.path.join(_HERE, "sortties_oracle_windows.json")

sys.path.insert(0, _SUPPORT)
import sortties_cells as sc  # noqa: E402


def _worker():
    """Runs in the shim interpreter: print {cell id: encoded indices | {error}}."""
    import torch

    assert hasattr(torch._C, "_aten_implemented"), "this is upstream torch, not the shim"
    out = {}
    for c in sc.cells():
        try:
            out[c["id"]] = sc.encode(sc.run_cell(torch, c))
        except Exception as e:  # noqa: BLE001
            out[c["id"]] = {"error": f"{type(e).__name__}: {e}"}
    sys.stdout.write(json.dumps(out, separators=(",", ":")))


_cache = {}


def _shim_answers():
    if "a" in _cache:
        return _cache["a"]
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([_VENDOR, _SUPPORT])
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run(
        [sys.executable, os.path.abspath(__file__), "--worker"],
        capture_output=True, text=True, env=env, cwd=_REPO,
    )
    assert proc.returncode == 0, f"shim worker failed:\n{proc.stderr[-3000:]}"
    _cache["a"] = json.loads(proc.stdout)
    return _cache["a"]


def _oracle():
    if "o" not in _cache:
        with open(_ORACLE) as f:
            _cache["o"] = json.load(f)["cells"]
    return _cache["o"]


# Upstream's unstable order is its C++ runtime's `std::sort`: libc++ on Apple
# and Android (`sortties_oracle.json`, macOS arm64), libstdc++ on Linux
# (`sortties_oracle_linux.json`, manylinux aarch64 and x86_64, issue #81), MSVC's
# STL on Windows (`sortties_oracle_windows.json`, recorded by the
# `sortties-oracle-windows` workflow; `msvc_sort.rs` transcribes it). The shim
# transcribes all three. On Windows the unstable-order tests run only once that
# oracle file exists; until then they are skipped by name. Stable order and the
# bool refusal hold on every platform; they are checked against this host's
# oracle, or the libc++ one where none exists.
_LIBCXX_HOST = sys.platform == "darwin" or hasattr(sys, "getandroidapilevel")
_LIBSTDCXX_HOST = sys.platform.startswith("linux") and not _LIBCXX_HOST
_MSVC_HOST = sys.platform == "win32"
_MSVC_ORACLE_PRESENT = os.path.exists(_ORACLE_MSVC)
_ORACLE = (
    _ORACLE_LIBSTDCXX if _LIBSTDCXX_HOST
    else _ORACLE_MSVC if (_MSVC_HOST and _MSVC_ORACLE_PRESENT)
    else _ORACLE_LIBCXX
)


def _libcxx_order(fn):
    def run():
        if not (_LIBCXX_HOST or _LIBSTDCXX_HOST or (_MSVC_HOST and _MSVC_ORACLE_PRESENT)):
            sys.path.insert(0, _SUPPORT)
            import _skip
            raise _skip.Skip(
                f"upstream's unstable tie order on {sys.platform} is MSVC's STL and "
                "tests/ops/sortties_oracle_windows.json is absent (run the "
                "sortties-oracle-windows workflow; issue #81)")
        fn()
    run.__name__ = fn.__name__
    return run


def _check(select, what):
    """Every cell `select` keeps must match upstream exactly; report N of M."""
    answers, oracle = _shim_answers(), _oracle()
    cells = [c for c in sc.cells() if select(c)]
    assert cells, f"{what}: no cells selected"
    bad = []
    for c in cells:
        want, got = oracle[c["id"]], answers[c["id"]]
        if isinstance(want, dict) and "error" in want:
            ok = isinstance(got, dict) and "error" in got
        else:
            ok = got == want
        if not ok:
            bad.append(c["id"])
    assert not bad, (
        f"{what}: {len(cells) - len(bad)} of {len(cells)} cells agree with upstream; "
        f"{len(bad)} differ, first: {bad[:3]}"
    )


def test_worker_is_the_shim_and_the_oracle_is_upstream():
    import torch  # noqa: F401  (this interpreter may be upstream; the worker is what matters)

    answers = _shim_answers()
    assert len(answers) == len(_oracle()) == len(sc.cells())
    with open(_ORACLE) as f:
        assert json.load(f)["torch"].startswith("2."), "oracle was not recorded by torch 2.x"


def test_sort_stable_true_keeps_index_order_in_both_directions():
    _check(lambda c: c["op"] == "sort" and c["args"]["stable"] is True, "sort(stable=True)")


def test_argsort_stable_true_keeps_index_order_in_both_directions():
    _check(lambda c: c["op"] == "argsort" and c["args"]["stable"] is True, "argsort(stable=True)")


@_libcxx_order
def test_sort_stable_false_matches_libcxx_introsort_ties():
    _check(lambda c: c["op"] == "sort" and c["args"]["stable"] is False, "sort(stable=False)")


@_libcxx_order
def test_sort_without_stable_matches_libcxx_introsort_ties():
    _check(lambda c: c["op"] == "sort" and c["args"]["stable"] is None, "sort()")


@_libcxx_order
def test_argsort_without_stable_matches_libcxx_introsort_ties():
    _check(lambda c: c["op"] == "argsort" and c["args"]["stable"] is None, "argsort()")


@_libcxx_order
def test_argsort_stable_false_matches_libcxx_introsort_ties():
    _check(lambda c: c["op"] == "argsort" and c["args"]["stable"] is False, "argsort(stable=False)")


def test_unstable_rows_longer_than_23_are_in_the_oracle():
    # The cells that distinguish the unstable sort from a stable one: if upstream's
    # order were the stable one on every cell here, a stable kernel would pass.
    oracle = _oracle()
    differs = 0
    for c in sc.cells():
        if c["op"] != "sort" or c["args"]["stable"] is not False or c["n"] < 64:
            continue
        twin = c["id"].replace("stable=False", "stable=True")
        if oracle[c["id"]] != oracle[twin]:
            differs += 1
    assert differs > 100, f"only {differs} cells separate unstable from stable"


@_libcxx_order
def test_topk_partial_sort_path_k_times_64_le_n():
    _check(lambda c: c["op"] == "topk" and c["args"]["k"] * 64 <= c["n"], "topk partial_sort")


@_libcxx_order
def test_topk_nth_element_path_sorted():
    _check(lambda c: c["op"] == "topk" and c["args"]["k"] * 64 > c["n"] and c["args"]["sorted"],
           "topk nth_element + sort")


@_libcxx_order
def test_topk_nth_element_path_unsorted():
    _check(lambda c: c["op"] == "topk" and c["args"]["k"] * 64 > c["n"] and not c["args"]["sorted"],
           "topk sorted=False")


def test_topk_refuses_bool_like_upstream():
    _check(lambda c: c["op"] == "topk" and c["dtype"] == "bool", "topk bool refusal")


@_libcxx_order
def test_every_cell_is_checked():
    _check(lambda c: True, "all cells")


def test_linux_oracle_is_libstdcxx_not_libcxx():
    # The two oracles must disagree on the unstable cells, or the Linux one
    # would not be measuring a different runtime; and agree on every stable
    # cell, because stable order is not the runtime's.
    with open(_ORACLE_LIBCXX) as f:
        mac = json.load(f)["cells"]
    with open(_ORACLE_LIBSTDCXX) as f:
        lin = json.load(f)
    assert lin["torch"].startswith("2.13.0"), lin["torch"]
    lin = lin["cells"]
    assert set(mac) == set(lin) == {c["id"] for c in sc.cells()}
    stable = [k for k in mac if "stable=True" in k]
    assert all(mac[k] == lin[k] for k in stable), "stable order differs between runtimes"
    differ = sum(mac[k] != lin[k] for k in mac)
    assert differ > 1000, f"only {differ} cells separate libstdc++ from libc++"


def test_windows_oracle_is_msvc_when_present():
    # Recorded by .github/workflows/sortties-oracle-windows.yml on windows-latest
    # with the pinned torch. It must be a different runtime's order from libc++
    # and libstdc++, and agree with both on every stable cell.
    if not _MSVC_ORACLE_PRESENT:
        sys.path.insert(0, _SUPPORT)
        import _skip
        raise _skip.Skip("tests/ops/sortties_oracle_windows.json is absent (issue #81)")
    with open(_ORACLE_LIBCXX) as f:
        mac = json.load(f)["cells"]
    with open(_ORACLE_MSVC) as f:
        win = json.load(f)
    assert win["torch"].startswith("2.13.0"), win["torch"]
    win = win["cells"]
    assert set(mac) == set(win) == {c["id"] for c in sc.cells()}
    stable = [k for k in mac if "stable=True" in k]
    assert all(mac[k] == win[k] for k in stable), "stable order differs between runtimes"
    differ = sum(mac[k] != win[k] for k in mac)
    assert differ > 1000, f"only {differ} cells separate MSVC's STL from libc++"


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        _worker()
        raise SystemExit(0)
    import _skip

    raise SystemExit(1 if _skip.run_tests(
        [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)],
        "test_sortties") else 0)
