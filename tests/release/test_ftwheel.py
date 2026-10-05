"""The free-threaded wheel variant (issue #51): `cp315-cp315t`, beside the abi3 one.

`scripts/wheel/build.py --abi ft` builds a second wheel for CPython 3.15t. The
abi3 wheel cannot load there (a free-threaded interpreter has no stable ABI),
so the extension is built without `abi3` and carries a version-pinned name and
tag. Everything version-shaped is read out of the target interpreter's
sysconfig (`Py_GIL_DISABLED`, `VERSION`, `SOABI`, `EXT_SUFFIX`), the same way
the abi3 path reads its platform tag out of a target CPython, so nothing is
hand-written.

What this suite checks, and what each check cannot see:

* **Derivation** (`scripts/wheel/ftabi.py`), from fake sysconfig dicts for
  Linux, macOS and Windows: tag, extension suffix and archive member. It cannot
  see whether a real 3.15t interpreter agrees; that is `test_freethreaded.py`'s
  subprocess, which needs a 3.15t install.
* **The refusals by name**: a GIL interpreter, a missing `Py_GIL_DISABLED`, an
  inconsistent suffix, and every target that has no 3.15t interpreter.
* **build.py wiring**: `--abi ft` reaches those refusals before any build.
* **The workflow**: the `build-ft` job exists, covers exactly the targets that
  can be derived, fetches the pinned python-build-standalone freethreaded
  archive, and `collect` names the ft wheels.

What none of it can see: that a built wheel loads on a device. iOS and Android
are refused, so they are not even "builds".
"""

import os
import pathlib
import re
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "_support"))
import _skip  # noqa: E402

REPO = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
sys.path.insert(0, str(REPO / "scripts" / "wheel"))

import ftabi  # noqa: E402

WORKFLOW = REPO / ".github/workflows/publish-pypi.yml"
BUILD_PY = REPO / "scripts/wheel/build.py"

LINUX = {"Py_GIL_DISABLED": 1, "VERSION": "3.15", "SOABI": "cpython-315t-x86_64-linux-gnu",
         "EXT_SUFFIX": ".cpython-315t-x86_64-linux-gnu.so"}
MACOS = {"Py_GIL_DISABLED": 1, "VERSION": "3.15", "SOABI": "cpython-315t-darwin",
         "EXT_SUFFIX": ".cpython-315t-darwin.so"}
WINDOWS = {"Py_GIL_DISABLED": 1, "VERSION": "3.15", "SOABI": "cp315t-win_amd64",
           "EXT_SUFFIX": ".cp315t-win_amd64.pyd"}


def _refusal(variables):
    try:
        ftabi.ft_identity(variables)
    except ftabi.FtRefusal as exc:
        return str(exc)
    raise AssertionError(f"{variables} was accepted")


# ----------------------------------------------------------- derivation

def test_linux_identity_comes_from_the_interpreters_sysconfig():
    ident = ftabi.ft_identity(LINUX)
    assert ident.python_tag == "cp315", ident
    assert ident.abi_tag == "cp315t", ident
    assert ident.tag_prefix == "cp315-cp315t", ident
    assert ident.ext_suffix == ".cpython-315t-x86_64-linux-gnu.so", ident
    assert ident.member == "torch/_C.cpython-315t-x86_64-linux-gnu.so", ident


def test_macos_member_is_the_darwin_suffix():
    assert ftabi.ft_identity(MACOS).member == "torch/_C.cpython-315t-darwin.so"


def test_windows_member_is_a_pyd():
    ident = ftabi.ft_identity(WINDOWS)
    assert ident.member == "torch/_C.cp315t-win_amd64.pyd", ident
    assert ident.tag_prefix == "cp315-cp315t", ident


def test_the_tag_follows_the_version_not_a_constant():
    """A 3.16t interpreter must produce cp316, not a hard-coded cp315."""
    v = {"Py_GIL_DISABLED": 1, "VERSION": "3.16", "SOABI": "cpython-316t-darwin",
         "EXT_SUFFIX": ".cpython-316t-darwin.so"}
    assert ftabi.ft_identity(v).tag_prefix == "cp316-cp316t"


def test_value_may_be_the_string_one():
    """`sysconfig.get_config_var` returns int on POSIX; a pickled dict may not."""
    assert ftabi.ft_identity({**LINUX, "Py_GIL_DISABLED": "1"}).abi_tag == "cp315t"


# ------------------------------------------------------------- refusals

def test_a_gil_interpreter_is_refused_by_name():
    msg = _refusal({**LINUX, "Py_GIL_DISABLED": 0,
                    "SOABI": "cpython-315-x86_64-linux-gnu",
                    "EXT_SUFFIX": ".cpython-315-x86_64-linux-gnu.so"})
    assert "not free-threaded" in msg and "Py_GIL_DISABLED" in msg, msg


def test_a_missing_py_gil_disabled_is_refused_not_assumed_false_or_true():
    v = dict(LINUX)
    del v["Py_GIL_DISABLED"]
    msg = _refusal(v)
    assert "Py_GIL_DISABLED" in msg, msg
    assert _refusal({**LINUX, "Py_GIL_DISABLED": None}), "None was accepted"


def test_a_suffix_that_disagrees_with_the_version_is_refused():
    msg = _refusal({**LINUX, "EXT_SUFFIX": ".cpython-314t-x86_64-linux-gnu.so"})
    assert "EXT_SUFFIX" in msg and "315t" in msg, msg


def test_a_soabi_that_disagrees_with_the_version_is_refused():
    msg = _refusal({**LINUX, "SOABI": "cpython-315-x86_64-linux-gnu"})
    assert "SOABI" in msg and "315t" in msg, msg


def test_an_abi3_suffix_is_refused():
    """`.abi3.so` would mean the abi3 build, which cannot load on 3.15t."""
    msg = _refusal({**LINUX, "EXT_SUFFIX": ".abi3.so"})
    assert "EXT_SUFFIX" in msg, msg


def test_every_target_is_either_derivable_or_refused_by_name():
    import build  # noqa: E402
    keys = {"host", *build.TARGETS}
    derivable, refused = set(ftabi.FT_TARGETS), set(ftabi.FT_REFUSALS)
    assert not derivable & refused, derivable & refused
    assert derivable | refused == keys, (
        f"unclassified: {sorted(keys - derivable - refused)}; "
        f"stale: {sorted((derivable | refused) - keys)}")
    for key, why in ftabi.FT_REFUSALS.items():
        assert why.strip() and "3.15t" in why, (key, why)


def test_ios_and_android_name_why_no_interpreter_exists():
    for key in ("ios-arm64", "ios-arm64-sim", "android-arm64-v8a", "android-x86_64"):
        assert key in ftabi.FT_REFUSALS, key
        assert "free-threaded" in ftabi.FT_REFUSALS[key], key


# --------------------------------------------------------------- naming

def test_wheel_filename_is_retagged_to_the_ft_tag():
    ident = ftabi.ft_identity(MACOS)
    got = ftabi.ft_wheel_name("torchnative-0.1.0-cp313-abi3-macosx_11_0_arm64.whl", ident)
    assert got == "torchnative-0.1.0-cp315-cp315t-macosx_11_0_arm64.whl", got


def test_wheel_file_tag_lines_are_rewritten_whole():
    ident = ftabi.ft_identity(LINUX)
    old = b"Wheel-Version: 1.0\nTag: cp313-abi3-manylinux_2_17_x86_64\nRoot-Is-Purelib: false\n"
    new = ftabi.ft_wheel_metadata(old, ident)
    assert b"Tag: cp315-cp315t-manylinux_2_17_x86_64\n" in new, new
    assert b"abi3" not in new and b"cp313" not in new, new
    assert b"Root-Is-Purelib: false\n" in new, new


def test_a_wheel_name_that_is_not_five_parts_is_refused():
    ident = ftabi.ft_identity(LINUX)
    try:
        ftabi.ft_wheel_name("torchnative-0.1.0-py3-none-any-extra-parts.whl", ident)
    except ftabi.FtRefusal:
        return
    raise AssertionError("a malformed wheel name was retagged")


# ---------------------------------------------------------- build.py CLI

def test_verify_finds_the_ft_member_for_a_cross_target():
    # Dry run 37287742911: both Linux ft wheels were built and then refused by
    # verify(), which looked for the abi3 member name on cross targets (target
    # not None) and only remapped `expected`. The host passed only because its
    # target is None and skips that lookup.
    import tempfile
    import zipfile
    import build
    ident = ftabi.ft_identity(LINUX)
    seen = []

    class Cross:
        extension_member = build.Target.extension_member
        global_deps_name = None

        def check_image(self, data, where):
            seen.append(where)

    with tempfile.TemporaryDirectory(dir=REPO / ".scratch") as d:
        whl = pathlib.Path(d) / "torchnative-0-cp315-cp315t-manylinux_2_17_x86_64.whl"
        with zipfile.ZipFile(whl, "w") as zf:
            zf.writestr(ident.member, b"\x7fELF")
        build.verify(whl, {build.Target.extension_member}, Cross(), {}, "torchnative-0.dist-info", ft=ident)
    assert seen and seen[0].endswith(ident.member), seen


def _run_build(*args, env=None):
    return subprocess.run([sys.executable, str(BUILD_PY), *args], capture_output=True,
                          text=True, cwd=REPO, timeout=120, env=env)


def test_abi_ft_on_ios_refuses_by_name_before_building():
    r = _run_build("--abi", "ft", "--target", "ios-arm64")
    assert r.returncode != 0, r.stdout
    out = r.stdout + r.stderr
    assert "--abi ft" in out and "ios-arm64" in out and "free-threaded" in out, out


def test_abi_ft_on_android_refuses_by_name_before_building():
    r = _run_build("--abi", "ft", "--target", "android-arm64-v8a")
    assert r.returncode != 0, r.stdout
    assert "android-arm64-v8a" in (r.stdout + r.stderr), r.stderr


def test_abi_ft_with_a_gil_interpreter_refuses_by_name():
    r = _run_build("--abi", "ft", "--ft-python", sys.executable)
    assert r.returncode != 0, r.stdout
    assert "not free-threaded" in (r.stdout + r.stderr), r.stdout + r.stderr


def test_abi_ft_host_without_an_interpreter_refuses_by_name():
    env = {k: v for k, v in os.environ.items() if k != "PYO3_PYTHON"}
    r = _run_build("--abi", "ft", env=env)
    assert r.returncode != 0, r.stdout
    assert "--ft-python" in (r.stdout + r.stderr), r.stdout + r.stderr


def test_the_default_abi_is_still_abi3():
    text = BUILD_PY.read_text()
    assert re.search(r'add_argument\("--abi",[^)]*default="abi3"', text, re.S), (
        "build.py's --abi no longer defaults to abi3; the ft variant must sit "
        "beside the abi3 wheel, not replace it")


# -------------------------------------------------------------- workflow

def _yaml():
    try:
        import yaml
    except ImportError:
        return None
    return yaml.safe_load(WORKFLOW.read_text())


def test_the_workflow_has_a_build_ft_job_for_exactly_the_derivable_targets():
    doc = _yaml()
    if doc is None:
        raise _skip.Skip("pyyaml is not installed in this interpreter")
    job = doc["jobs"]["build-ft"]
    got = tuple(sorted(e["target"] for e in job["strategy"]["matrix"]["include"]))
    assert got == tuple(sorted(ftabi.FT_TARGETS)), got
    needs = job.get("needs") or []
    assert "preflight" in needs and "vendor" in needs, needs


def test_collect_needs_build_ft_and_names_the_ft_wheels():
    doc = _yaml()
    if doc is None:
        raise _skip.Skip("pyyaml is not installed in this interpreter")
    needs = doc["jobs"]["collect"]["needs"]
    assert "build-ft" in needs and "build" in needs and "gate" in needs, needs
    text = WORKFLOW.read_text()
    for plat in ("macosx_11_0_arm64", "manylinux_2_17_x86_64", "manylinux_2_17_aarch64"):
        assert f"cp315-cp315t-{plat}" in text, plat


def test_the_ft_interpreter_is_the_pinned_pbs_freethreaded_archive():
    text = WORKFLOW.read_text()
    assert re.search(r'PBS_FT_PYTHON: "3\.15\.0rc1"', text), "PBS_FT_PYTHON is not pinned to 3.15"
    assert "freethreaded-install_only.tar.gz" in text, (
        "the ft interpreter is not the freethreaded install_only archive")
    assert "--no-default-features" in text, "the ft cargo build keeps abi3"


def test_ft_build_does_not_disable_its_own_assertions():
    doc = _yaml()
    if doc is None:
        raise _skip.Skip("pyyaml is not installed in this interpreter")
    job = doc["jobs"]["build-ft"]
    assert not job.get("continue-on-error")
    for step in job["steps"]:
        assert not step.get("continue-on-error"), step.get("name")
        for line in (step.get("run") or "").splitlines():
            s = line.strip()
            if s.startswith("#"):
                continue
            assert not s.endswith("|| true") and "set +e" not in s, (step.get("name"), s)


def test_ft_build_does_not_revendor_and_uses_its_own_cargo_dir():
    doc = _yaml()
    if doc is None:
        raise _skip.Skip("pyyaml is not installed in this interpreter")
    job = doc["jobs"]["build-ft"]
    assert "vendor_torch.sh" not in str(job["steps"])
    assert "TORCHNATIVE_FT_CARGO_TARGET_DIR" in str(job), (
        "the ft build must not share a cargo target dir with the abi3 host shim")


if __name__ == "__main__":
    raise SystemExit(_skip.run_tests(
        [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)],
        "release/test_ftwheel",
    ))
