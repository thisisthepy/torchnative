#!/usr/bin/env python3
"""Build a platform wheel that actually contains a torch you can import.

    python tools/wheel/build.py                            # this machine
    python tools/wheel/build.py --target android-arm64-v8a
    python tools/wheel/build.py --target ios-arm64
    python tools/wheel/build.py --target ios-arm64-sim

The distribution on PyPI as `torchnative 0.0.1a0` is `py3-none-any` and holds
only the `torchnative/` skeleton: no `_C`, no vendored tree. `pip install
torchnative` gets you something where `import torch` raises ImportError. This
script exists so that the *default* way to build produces the other thing.

What it does beyond `pip wheel .`:

  preflight   Refuse to build unless `vendor/vendor_torch.sh` has laid down the
              upstream Python tree and `vendor/install_shim.sh` has put our `_C`
              in the hole. Both are gitignored, so a fresh clone has neither,
              and setuptools would happily emit the empty shell again -- which
              is how the PyPI one came to exist. The failure has to be loud.

  freshness   Refuse to build unless the `_C` about to be packaged was built
              from the source on disk now, read off cargo's own dep-info. This
              script does not *build* the cross artefacts -- it picks them up
              from `CARGO_TARGET_DIR/<triple>/release/` -- and until 2026-08-29
              nothing looked at their age, so a five-day-old extension shipped
              with an exit code of 0 and passed every check downstream (§
              `artefact_verdict`, docs/WHEEL.md §11).

  repack      Two fixes that have no setuptools spelling, applied to the
              finished archive (§ `_repack`):
                - `torch-<v>.dist-info/` from upstream, so that
                  `importlib.metadata.version("torch")` answers. transformers
                  gates the entire torch integration on that call
                  (docs/VENDOR.md), so it is load-bearing, not decoration.
                - the Mach-O install name of `_C.abi3.so`, which cargo sets to
                  the absolute path of the build machine's target directory.

  verify      Diff the archive against the source tree file-by-file and fail on
              anything missing. A wheel that is merely *smaller* than it should
              be installs fine and breaks later, at some import nobody ran.

Cross targets (`--target`) reuse all of that and swap three things, which are
exactly the three that are platform-shaped (§ `TARGETS`):

  the extension   `torch/_C.abi3.so` comes from the cross build's target
                  directory instead of from the source tree. The source tree
                  keeps the host artefact, so the suites that read it keep
                  measuring the host.
  global deps     built by the target's compiler, and named `.so` rather than
                  `.dylib` on iOS as well as Android -- `_load_global_deps()`
                  picks the extension with `platform.system() == "Darwin"`,
                  which is False on iOS.
  the tag         `android_<api>_<abi>` / `ios_<major>_<minor>_<multiarch>`,
                  derived from the *target CPython distribution* rather than
                  written down here, and cross-checked against what the
                  artefact itself says it needs.

Then check it for real -- building is not the proof:

    python tools/wheel/verify.py dist/<host wheel>        # installs it
    python tools/wheel/verify_cross.py dist/<cross wheel> # inspects it
    python tools/wheel/verify_android.py dist/<android wheel>    # imports it
    python tools/wheel/verify_ios_sim.py dist/<ios sim wheel>    # imports it
    python tools/wheel/verify_ios_device.py dist/<ios device wheel>  # links it

And check this script itself, which builds nothing and takes a second:

    python tools/wheel/build.py --self-test
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from binfmt import describe, elf_info, macho_arches, macho_info  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SRC = REPO / "torchnative" / "src" / "main"

# The two artefacts the vendor scripts produce. Their absence is the whole
# failure mode this script exists to catch, so they are named individually
# rather than checked as "does src/main look populated".
VENDORED_ROOT = SRC / "torch" / "__init__.py"
SHIM = SRC / "torch" / "_C.abi3.so"
STAMP = SRC / ".stamp"

SKIP_SUFFIXES = (".pyc", ".pyo")
SKIP_DIRS = {"__pycache__"}
#: Finder metadata. Excluded in pyproject too, but that only reaches files
#: *inside* a package -- the copy at the package root lands at the archive
#: root, which setuptools' package-data globs never see. Named here so the
#: preflight can refuse rather than ship one, as it did in 0.0.2a0's first
#: build.
SKIP_NAMES = {".DS_Store"}


def _fail(msg: str) -> None:
    sys.exit(f"tools/wheel/build.py: {msg}")


def preflight() -> dict[str, str]:
    """Refuse to build a wheel that would be missing its reason to exist."""
    if not VENDORED_ROOT.exists():
        _fail(
            f"no vendored torch tree at {VENDORED_ROOT.parent}\n"
            "  run: bash vendor/vendor_torch.sh\n"
            "  (the tree is gitignored -- a fresh clone never has it, and\n"
            "   building without it yields the py3-none-any shell on PyPI)"
        )
    if not SHIM.exists():
        _fail(
            f"no _C extension at {SHIM}\n"
            "  run: bash vendor/install_shim.sh\n"
            "  (without it the tree cannot import; that is by design --\n"
            "   vendor_torch.sh drops upstream's _C so the hole is visible)"
        )

    # Finder metadata, refused rather than merely excluded. `.DS_Store` got into
    # 0.0.2a0's first build: the copy at the package root lands at the *archive*
    # root, where setuptools' package-data globs never look, and `.gitignore`
    # cannot see it because the vendored tree is not in git. Excluding it fixes
    # this build; refusing is what makes the next one say so instead of quietly
    # shipping whatever Finder left behind. It was caught by the iOS check
    # comparing two wheels member by member -- 2,566 shared, 1 differing -- which
    # is a long way round for a file that should never have been packageable.
    junk = sorted(
        str(p.relative_to(SRC)) for name in SKIP_NAMES
        for p in SRC.rglob(name)
    )
    if junk:
        _fail(
            "the source tree carries files that must not be packaged:\n"
            + "".join(f"   {j}\n" for j in junk)
            + "   remove them and re-run (find . -name .DS_Store -delete)"
        )

    stamp: dict[str, str] = {}
    if STAMP.exists():
        for line in STAMP.read_text().splitlines():
            key, _, value = line.partition("=")
            stamp[key] = value

    # `native_left=0` is vendor_torch.sh's own measurement that it removed every
    # compiled artefact from the tree. If it is not 0, the tree still carries
    # upstream binaries and the wheel would ship somebody else's `_C` alongside
    # ours -- with upstream's `.cpython-313-darwin.so` suffix winning the
    # importlib race, so ours would be dead weight and nothing would say so.
    if stamp.get("native_left") not in (None, "0"):
        _fail(
            f"vendored tree still holds {stamp['native_left']} native artefact(s)"
            " -- re-run vendor/vendor_torch.sh"
        )
    return stamp


def tree_files(root: Path) -> set[str]:
    """Every file the wheel is expected to carry, as archive-relative paths."""
    out: set[str] = set()
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            if name.endswith(SKIP_SUFFIXES) or name == ".DS_Store":
                continue
            out.add(str((Path(dirpath) / name).relative_to(SRC)))
    return out


def run_pip_wheel(python: str, outdir: Path) -> Path:
    """Drive the PEP 517 build through pip.

    pip rather than `python -m build` because pip is in every virtualenv and
    `build` is not; `--no-build-isolation` because the build needs nothing that
    is not already installed, and isolation would download setuptools on every
    invocation. `--no-deps` because this distribution's dependency list is empty
    and resolving it would reach the network for no reason.
    """
    before = set(outdir.glob("*.whl")) if outdir.exists() else set()
    cmd = [
        python, "-m", "pip", "wheel", str(REPO),
        "--no-deps", "--no-build-isolation", "--wheel-dir", str(outdir),
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    made = sorted(set(outdir.glob("*.whl")) - before)
    if len(made) != 1:
        # Rebuilding without a version bump overwrites in place, so an empty
        # diff is normal; more than one is not.
        made = sorted(outdir.glob("torchnative-*.whl"))
        if len(made) != 1:
            _fail(f"expected exactly one torchnative wheel in {outdir}, found {made}")
    return made[0]


def _record_line(arcname: str, data: bytes) -> tuple[str, str, str]:
    digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=")
    return arcname, "sha256=" + digest.decode("ascii"), str(len(data))


def _fix_install_name(data: bytes) -> bytes:
    """Strip the build machine's absolute path out of the Mach-O install name.

    cargo links a `cdylib` with `-install_name <CARGO_TARGET_DIR>/release/deps/
    lib_C.dylib`, so a wheel built here announces
    `/Volumes/macMini/caches/cargo-target-wheel/...` to anyone who runs `otool
    -L` on it. It is inert -- `dlopen()` of a module by path ignores
    `LC_ID_DYLIB`, which is why the extension imports anyway -- but a published
    artefact should not name a directory on the machine that built it.

    Doing it here rather than in `vendor/install_shim.sh` keeps the file that
    the test suites load byte-identical to the one cargo emitted, so a green
    suite is evidence about the artefact and not about this rewrite.

    Applies to the iOS artefacts too -- cargo does the same thing there. Not to
    the Android one: that is an ELF, and `install_name_tool` on an ELF is not a
    no-op but an error, so the format is checked rather than the host platform.
    """
    if macho_info(data) is None:
        return data
    if sys.platform != "darwin" or not shutil.which("install_name_tool"):
        return data
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "_C.abi3.so"
        path.write_bytes(data)
        subprocess.run(
            ["install_name_tool", "-id", "@rpath/_C.abi3.so", str(path)],
            check=True, capture_output=True,
        )
        # codesign: install_name_tool invalidates the ad-hoc signature that the
        # linker put on the arm64 slice, and macOS refuses to load an arm64
        # Mach-O with a *broken* signature (an unsigned one is fine, a stale one
        # is not). Re-sign ad-hoc. If codesign is missing, better to ship the
        # untouched original than an unloadable one.
        if shutil.which("codesign"):
            r = subprocess.run(
                ["codesign", "--force", "--sign", "-", str(path)],
                capture_output=True,
            )
            if r.returncode != 0:
                print("  ! codesign failed, keeping original install name",
                      r.stderr.decode(errors="replace").strip())
                return data
        else:
            print("  ! no codesign, keeping original install name")
            return data
        return path.read_bytes()


def honest_macos_plat(plat: str, so_data: bytes) -> str:
    """Correct the architecture half of a macOS platform tag.

    setuptools derives the platform tag from `sysconfig.get_platform()` and then
    lets `calculate_macosx_platform_tag` walk the archive -- but that walk only
    ever raises the *deployment target*; it never touches the architecture. So a
    wheel built by the python.org universal2 interpreter is tagged
    `macosx_11_0_universal2` no matter what is inside it. Ours holds an arm64
    Mach-O and nothing else.

    The difference is not cosmetic. `universal2` tells pip on an Intel Mac that
    this wheel is for it; the install succeeds and `import torch` then dies in
    `dlopen` with a mach-o architecture mismatch, at which point the user's
    evidence points at their machine rather than at this tag. Narrowing it means
    those users get "no matching distribution", which is the truth.
    """
    m = re.match(r"^(macosx_\d+_\d+)_(.+)$", plat)
    if not m:
        return plat
    arches = set(macho_arches(so_data))
    if not arches:
        return plat
    if arches == {"arm64"}:
        arch = "arm64"
    elif arches == {"x86_64"}:
        arch = "x86_64"
    elif arches == {"arm64", "x86_64"}:
        arch = "universal2"
    else:
        return plat  # something unusual; leave setuptools' answer alone
    return f"{m.group(1)}_{arch}"


def _retag(name: str, plat: str) -> str:
    """Swap the platform component of a wheel filename."""
    stem, _, _ = name.rpartition(".whl")
    parts = stem.split("-")
    parts[-1] = plat
    return "-".join(parts) + ".whl"


# ------------------------------------------------------------- cross targets
#
# Everything platform-shaped about a wheel is one of three things: which
# `_C.abi3.so` goes in it, which compiler builds the empty global-deps library
# beside it, and what the tag says. A target is those three answers.
#
# The tag is *derived*, not written down. Both PEP 738 and PEP 730 encode a
# minimum OS version, and the honest value for that is a property of the CPython
# distribution the extension is built against -- `ANDROID_API_LEVEL` and
# `IPHONEOS_DEPLOYMENT_TARGET` in its `_sysconfigdata_*.py`. Hardcoding either
# would mean the tag stays put when the distribution is replaced, which is the
# same class of lie as the `universal2` tag in §3.3 of docs/WHEEL.md.

TARGET_PYTHON_ROOT = Path(os.environ.get(
    "TORCHNATIVE_TARGET_PYTHON", "/Volumes/macMini/caches/target-python"))

# Where `cargo` put the cross artefacts. Cargo's own default is `<crate>/target`
# and every build wiring in this repository overrides it, so read the same
# variable rather than inventing a third convention.
CARGO_TARGET_DIR = Path(os.environ.get(
    "CARGO_TARGET_DIR", REPO / "rust" / "torch_c" / "target"))

CRATE = REPO / "rust" / "torch_c"

# ------------------------------------------------------- artefact freshness
#
# The failure this exists to catch, measured 2026-08-29:
#
#     CARGO_TARGET_DIR/release/lib_C.dylib                 today
#     CARGO_TARGET_DIR/aarch64-apple-ios/release/...       4 days old
#     CARGO_TARGET_DIR/aarch64-apple-ios-sim/release/...   5 days old
#
# `build.py --target ios-arm64-sim` exited 0 and packaged the 5-day-old file.
# It does not build the cross artefact -- it picks one up from
# `CARGO_TARGET_DIR/<triple>/release/` -- and `preflight` looks only at the
# vendored tree and the host `_C`. So a wheel got built, verified and *passed*
# against source that had been superseded by a week of landed commits; the tell
# was an error message quoting a phrase no longer in the tree.
#
# Refuse rather than rebuild, for the reason `rust/torch_c/pytests/run.sh`
# refuses to install the shim it found stale:
#
#   * building here means writing down a second spelling of the cross build.
#     The device one needs PYO3_CONFIG_FILE (whose contents live in
#     docs/WHEEL.md §7.1, not in this repository), PYO3_CROSS_LIB_DIR and
#     TORCHNATIVE_PYTHON_FRAMEWORK_DIR; the Android one goes through
#     `scripts/device_android.sh build` and `cargo ndk --platform 21`. A second
#     spelling can drift from the first, and this repository's whole class of
#     recurring defect is a drift that is invisible until a device refuses the
#     file.
#   * this script already refuses when the artefact is *absent*, and points at
#     those same docs. Staleness is that question one notch further in;
#     answering it differently would mean "missing is your problem, stale is
#     mine".
#
# The criterion is cargo's own dep-info file (`lib_C.d`, next to the artefact),
# not a glob written here. That matters three ways:
#
#   * it lists what the build actually read, `include_str!` included -- this
#     crate pulls in `src/methods.json`, `src/overloads.json`, `src/surface.json`
#     and `src/bootstrap.py`, and a hand-written glob that missed one would be
#     silently blind to changes in it;
#   * it is per-target, sitting beside the artefact it describes, so the device
#     and simulator answers cannot be confused for each other;
#   * it records *absolute* paths, so an artefact built from a different
#     checkout is visible. A source glob cannot see that at all -- it would
#     compare this tree's mtimes against a binary built somewhere else and call
#     it fresh.
#
# `run.sh` compares bytes rather than mtimes, and deliberately: it *builds*, so
# it has a fresh reference, and a byte compare does not nag when a rebuild
# reproduces the previous output. Here there is nothing to compare against
# without building, which is what was just ruled out. mtime is what remains --
# and reading it off the dep-info makes it the same question cargo asks when it
# decides whether to rebuild, rather than a stricter one invented here.
#
# Three outcomes, kept apart. `run.sh` learned this from `cmp`, which
# distinguishes same (0), different (1) and "the comparison itself failed"
# (>1); folding the last into the middle once produced a staleness report about
# an artefact that was current. The same trap is here in the other direction and
# is the easier one to fall into: `max()` over an empty dependency list has no
# error to report, it just answers "nothing is newer than the artefact" -- so an
# unreadable, unparseable or foreign dep-info would read as *fresh*. Every path
# out of `artefact_verdict` therefore names which of the three it is, and only
# one of them is a staleness claim.

FRESH, STALE, UNKNOWN = "fresh", "stale", "unknown"


def _unescape_make(path: str) -> str:
    r"""Undo the `\ ` that cargo writes for a space inside a path."""
    return re.sub(r"\\(.)", r"\1", path)


def _make_rules(text: str) -> list[tuple[str, list[str]]]:
    r"""Split a Makefile-format dep-info into `(target, prerequisites)`.

    Only the shape cargo emits is handled: one rule per line, no line
    continuations, spaces inside paths escaped with a backslash. A line without
    an unescaped `:` is not a rule and is skipped -- cargo writes bare
    `<dep>:` lines in some configurations and those carry no information here.
    """
    rules: list[tuple[str, list[str]]] = []
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        # First unescaped colon that is not `C:\...`; cargo writes POSIX paths
        # on this platform, so the drive-letter case cannot arise, but the scan
        # is written to skip escaped colons regardless.
        idx = -1
        i = 0
        while i < len(line):
            if line[i] == "\\":
                i += 2
                continue
            if line[i] == ":":
                idx = i
                break
            i += 1
        if idx < 0:
            continue
        target = _unescape_make(line[:idx].strip())
        rest = line[idx + 1:]
        deps = [_unescape_make(tok)
                for tok in re.split(r"(?<!\\)\s+", rest.strip()) if tok]
        rules.append((target, deps))
    return rules


def artefact_verdict(artefact: Path) -> tuple[str, str]:
    """Was `artefact` built from the source that is on disk now?

    Returns `(FRESH | STALE | UNKNOWN, detail)`. `UNKNOWN` is never a staleness
    claim -- it means the question could not be answered, and the detail says
    why. See the block comment above for why the three are kept apart.
    """
    dep_file = artefact.with_suffix(".d")
    if not dep_file.exists():
        return UNKNOWN, (
            f"cargo wrote no dep-info beside it ({dep_file} does not exist), so "
            "what that build read is not recorded anywhere")
    try:
        text = dep_file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        return UNKNOWN, f"{dep_file} could not be read: {exc!r}"

    rules = _make_rules(text)
    if not rules:
        return UNKNOWN, f"{dep_file} holds no Makefile rule this can read"

    want = os.path.realpath(artefact)
    deps: list[str] | None = None
    for target, prerequisites in rules:
        if os.path.realpath(target) == want:
            deps = prerequisites
            break
    if deps is None:  # a target dir moved after the build; match on the name
        for target, prerequisites in rules:
            if os.path.basename(target) == artefact.name:
                deps = prerequisites
                break
    if deps is None:
        return UNKNOWN, (
            f"{dep_file} describes {[t for t, _ in rules][:3]}, none of which is "
            f"{artefact} -- it is some other build's record")
    if not deps:
        return UNKNOWN, (
            f"{dep_file} names {artefact.name} but lists no inputs, so there is "
            "nothing to compare it against")

    # Absolute paths from the machine that built it. If they are not inside this
    # crate then the artefact came out of a different checkout, and comparing it
    # against *this* tree's mtimes would answer a question nobody asked.
    crate = os.path.realpath(CRATE)
    foreign = [d for d in deps if not os.path.realpath(d).startswith(crate + os.sep)]
    if foreign:
        return UNKNOWN, (
            f"the build read {len(foreign)} input(s) from outside {CRATE}, e.g. "
            f"{foreign[0]} -- this artefact was built from a different checkout, "
            "so its age relative to this one says nothing")

    try:
        artefact_ns = artefact.stat().st_mtime_ns
    except OSError as exc:
        return UNKNOWN, f"{artefact} could not be stat'ed: {exc!r}"

    newest: tuple[int, str] | None = None
    for dep in deps:
        try:
            dep_ns = os.stat(dep).st_mtime_ns
        except OSError as exc:
            return UNKNOWN, (
                f"the build read {dep}, which cannot be stat'ed now: {exc!r}. "
                "Whether the artefact is current is therefore unknown")
        if newest is None or dep_ns > newest[0]:
            newest = (dep_ns, dep)

    assert newest is not None  # `deps` is non-empty and every stat succeeded
    if newest[0] > artefact_ns:
        gap = (newest[0] - artefact_ns) / 1e9
        return STALE, (
            f"{os.path.relpath(newest[1], REPO)} was modified {gap / 3600:.1f} h "
            f"({gap:.0f} s) after {artefact.name} was written, and it is one of "
            f"the {len(deps)} inputs that build read")
    gap = (artefact_ns - newest[0]) / 1e9
    return FRESH, (
        f"{len(deps)} recorded inputs, newest "
        f"{os.path.relpath(newest[1], REPO)}, {gap / 3600:.1f} h before it")


def _render_refusal(verdict: str, artefact: Path, detail: str,
                    rebuild_hint: str) -> str:
    """The text a reader gets. Separate from `require_current` so the self-test
    can check that a staleness report and a "cannot judge" read as different
    things, which is the whole reason the two verdicts exist."""
    if verdict == STALE:
        return (
            f"{artefact} is stale.\n"
            f"  {detail}.\n"
            "  Packaging it would put superseded machine code in the wheel, and\n"
            "  everything downstream -- the tag, the platform check, the file\n"
            "  list, verify_cross.py -- would pass, because none of them looks at\n"
            "  the age of the code. That has happened: a simulator wheel was built\n"
            "  and verified against source five days out of date, and the tell was\n"
            "  an error message quoting a phrase no longer in the tree.\n"
            f"  Fix: {rebuild_hint}"
        )
    return (
        f"cannot tell whether {artefact} is current.\n"
        f"  {detail}.\n"
        "  This is NOT a staleness report -- the artefact may well be fine. It is\n"
        "  the check failing to run, which is reported separately so that it is\n"
        "  never mistaken for a finding about the artefact.\n"
        f"  Fix: {rebuild_hint}, which writes the dep-info this reads."
    )


def require_current(artefact: Path, rebuild_hint: str) -> None:
    """`_fail` unless `artefact` was built from the source now on disk."""
    verdict, detail = artefact_verdict(artefact)
    if verdict == FRESH:
        print(f"  current: {detail}")
        return
    _fail(_render_refusal(verdict, artefact, detail, rebuild_hint))


HOST_REBUILD_HINT = "bash vendor/install_shim.sh"


def check_host_shim() -> None:
    """The host wheel's `_C` is a *copy*; check the thing it was copied from.

    `preflight` establishes that `torch/_C.abi3.so` exists. It cannot establish
    that it is current, because the file carries no record of what built it --
    `vendor/install_shim.sh` runs `cargo build --release` and then `cp`s the
    result into the vendored tree, and nothing ties the copy back to its source
    afterwards.

    So the question is asked in two halves, which is what
    `rust/torch_c/pytests/run.sh` already does for its own reasons: is the host
    cargo artefact current (dep-info), and is the shim that same artefact
    (bytes). Both have to hold. Bytes rather than mtime for the second half for
    run.sh's reason -- `cp` sets a fresh mtime on the copy, so mtime would say
    "newer" and mean nothing.

    The cross path does not need this second half: it reads the cargo artefact
    directly, so there is no copy in between.
    """
    host = None
    for name in ("lib_C.dylib", "lib_C.so"):
        candidate = CARGO_TARGET_DIR / "release" / name
        if candidate.exists():
            host = candidate
            break
    if host is None:
        _fail(
            "cannot tell whether the host _C.abi3.so is current.\n"
            f"  There is no lib_C.dylib or lib_C.so under {CARGO_TARGET_DIR}/"
            "release,\n"
            "  so the build it was copied from is not on this machine to compare\n"
            "  against. This is NOT a staleness report -- the shim may be fine.\n"
            "  CARGO_TARGET_DIR selects where to look; if the shim was installed\n"
            "  with a different one, set it to that.\n"
            f"  Fix: {HOST_REBUILD_HINT}"
        )
    require_current(host, HOST_REBUILD_HINT)

    try:
        installed = SHIM.read_bytes()
        built = host.read_bytes()
    except OSError as exc:
        _fail(
            f"cannot tell whether {SHIM} is the artefact {host} holds.\n"
            f"  Reading one of them failed: {exc!r}.\n"
            "  This is NOT a staleness report -- it is the comparison itself\n"
            "  failing, and the two are kept apart on purpose (see run.sh, where\n"
            "  a SIGKILLed `cmp` once got reported as a stale shim).\n"
            f"  Fix: retry; if it persists, {HOST_REBUILD_HINT}"
        )
    if installed != built:
        _fail(
            f"{SHIM} is not the artefact under {CARGO_TARGET_DIR}/release.\n"
            f"  {len(installed):,} B installed vs {len(built):,} B built.\n"
            "  install_shim.sh copies one to the other, so they differ only if\n"
            "  the crate has been rebuilt since -- meaning the wheel would carry\n"
            "  the older of the two, silently.\n"
            f"  Fix: {HOST_REBUILD_HINT}"
        )
    print(f"  current: torch/_C.abi3.so is byte-identical to {host.name}")


# NDK ABI names, keyed by the architecture half of CPython's `MULTIARCH`. This
# is the NDK's own mapping (developer.android.com/ndk/guides/abis), not a guess;
# `packaging.tags.android_platforms` normalises the hyphen to an underscore and
# that normalisation is applied here too, by `_normalise`.
_ANDROID_ABIS = {
    "aarch64": "arm64-v8a",
    "arm": "armeabi-v7a",
    "x86_64": "x86_64",
    "i686": "x86",
}


def _normalise(s: str) -> str:
    """`packaging.tags._normalize_string`: hyphens and periods become `_`."""
    return s.replace("-", "_").replace(".", "_")


def target_sysconfig(root: Path) -> dict[str, object]:
    """`build_time_vars` of a cross-compiled CPython, without importing it.

    The file is CPython's own generated `_sysconfigdata_<abi>_<multiarch>.py`,
    a single dict literal. Reading it is how the wheel tag gets to be a fact
    about the interpreter the extension will actually be loaded by, rather than
    a constant in this file.
    """
    found = sorted(root.glob("lib/python3.*/_sysconfigdata_*.py"))
    if not found:
        _fail(f"no _sysconfigdata_*.py under {root}/lib/python3.* -- "
              f"is {root} a cross-compiled CPython distribution?")
    namespace: dict[str, object] = {}
    exec(compile(found[0].read_text(), str(found[0]), "exec"), namespace)
    variables = namespace.get("build_time_vars")
    if not isinstance(variables, dict):
        _fail(f"{found[0]} defines no build_time_vars")
    return variables


class Target:
    """One cross target: an artefact, a compiler, and a derived tag."""

    def __init__(self, key: str, rust_target: str, python_root: Path,
                 artefact_name: str):
        self.key = key
        self.rust_target = rust_target
        self.python_root = python_root
        self.artefact = CARGO_TARGET_DIR / rust_target / "release" / artefact_name

    # The name `_load_global_deps()` will look for. It picks the extension with
    # `".dylib" if platform.system() == "Darwin" else ".so"` -- and
    # `platform.system()` is "iOS" on iOS and "Android" on Android, so both of
    # them want `.so` even though the iOS file is a Mach-O dylib. Getting this
    # wrong is silent: the `ctypes.CDLL` raises OSError, `_load_global_deps`
    # swallows it into `_preload_cuda_deps`, and the import fails somewhere else.
    global_deps_name = "libtorch_global_deps.so"

    # How to produce this artefact. Named rather than run -- see the freshness
    # block comment for why this script refuses instead of rebuilding.
    rebuild_hint = "rebuild the cross artefact"

    def sysconfig(self) -> dict[str, object]:
        return target_sysconfig(self.python_root)

    def platform_tag(self, artefact: bytes) -> str:      # pragma: no cover
        raise NotImplementedError

    def cc(self) -> list[str]:                           # pragma: no cover
        raise NotImplementedError

    def check_image(self, data: bytes, what: str) -> None:  # pragma: no cover
        """Is this machine code for this target? Raises through `_fail`."""
        raise NotImplementedError

    def check_artefact(self, data: bytes) -> None:
        self.check_image(data, str(self.artefact))

    def check_global_deps(self, data: bytes) -> None:
        self.check_image(data, f"the global-deps library from {self.cc()[0]}")


class AndroidTarget(Target):
    """PEP 738: `android_<api-level>_<abi>`.

    The API level in the tag is a *floor*: `packaging.tags.android_platforms`
    yields every level from the device's own down to 16, so a device at API 34
    accepts an `android_21_` wheel and a device at API 19 does not. The floor
    that is true for this build is the one CPython itself was configured with
    -- `scripts/device_android.sh build` passes `--platform 21` to `cargo ndk`
    for the same reason, and the two agreeing is checked below rather than
    assumed.
    """

    def __init__(self):
        super().__init__(
            "android-arm64-v8a", "aarch64-linux-android",
            TARGET_PYTHON_ROOT / "aarch64-linux-android" / "prefix", "lib_C.so",
        )

    rebuild_hint = "scripts/device_android.sh build"

    def _api_and_abi(self) -> tuple[int, str]:
        variables = self.sysconfig()
        api = int(variables["ANDROID_API_LEVEL"])
        arch = str(variables["MULTIARCH"]).split("-")[0]
        if arch not in _ANDROID_ABIS:
            _fail(f"unknown Android architecture {arch!r} in MULTIARCH="
                  f"{variables['MULTIARCH']!r}")
        return api, _normalise(_ANDROID_ABIS[arch])

    def platform_tag(self, artefact: bytes) -> str:
        api, abi = self._api_and_abi()
        tag = f"android_{api}_{abi}"
        _confirm_with_packaging(tag, "android", api_level=api, abi=abi)
        return tag

    def cc(self) -> list[str]:
        api, _ = self._api_and_abi()
        ndk = Path(os.environ.get(
            "ANDROID_NDK_HOME",
            Path.home() / "Library/Android/sdk/ndk/27.1.12297006"))
        found = sorted(ndk.glob(
            f"toolchains/llvm/prebuilt/*/bin/aarch64-linux-android{api}-clang"))
        if not found:
            _fail(
                f"no aarch64-linux-android{api}-clang under {ndk} -- set "
                "ANDROID_NDK_HOME. Without it the global-deps library would be "
                "built by the host cc, which puts a Mach-O inside an Android "
                "wheel and makes `import torch` fail on the device only"
            )
        return [str(found[0]), "-shared", "-fPIC"]

    def check_image(self, data: bytes, what: str) -> None:
        info = elf_info(data)
        if info is None:
            _fail(f"{what} is not an ELF image ({describe(data)}) -- "
                  "an Android wheel cannot carry it")
        if (info["bits"], info["machine"], info["type"]) != (64, "aarch64", "dyn"):
            _fail(f"{what} is {describe(data)}, expected ELF 64-bit aarch64 dyn")


class IOSTarget(Target):
    """PEP 730: `ios_<major>_<minor>_<multiarch>`.

    Same floor semantics as Android -- `packaging.tags.ios_platforms` walks down
    to 12.0 -- so the version comes from `IPHONEOS_DEPLOYMENT_TARGET`, and the
    `multiarch` half straight from CPython's `MULTIARCH`, which is exactly the
    `sys.implementation._multiarch` that `ios_platforms` reads at run time.

    The device and the simulator are two targets, not one: their Mach-O images
    differ only in the platform field of `LC_BUILD_VERSION` (2 against 7), so a
    wheel carrying the wrong one is indistinguishable by size, architecture or
    symbol table and fails only when a real device tries to load it.
    """

    def __init__(self, key: str, rust_target: str, subdir: str, sdk: str,
                 platform_id: str):
        super().__init__(key, rust_target, TARGET_PYTHON_ROOT / subdir,
                         "lib_C.dylib")
        self.sdk = sdk
        self.platform_id = platform_id
        self.rebuild_hint = (
            "re-run the cross build for this target -- docs/WHEEL.md §7.1 has "
            f"the exact command (cargo build --release --target {rust_target}, "
            "with PYO3_CONFIG_FILE and PYO3_CROSS_LIB_DIR"
            + (", TORCHNATIVE_PYTHON_FRAMEWORK_DIR"
               if platform_id == "ios" else "")
            + ")")

    def _multiarch_and_min(self) -> tuple[str, tuple[int, int]]:
        variables = self.sysconfig()
        multiarch = _normalise(str(variables["MULTIARCH"]))
        major, _, minor = str(variables["IPHONEOS_DEPLOYMENT_TARGET"]).partition(".")
        return multiarch, (int(major), int(minor or 0))

    def platform_tag(self, artefact: bytes) -> str:
        multiarch, (major, minor) = self._multiarch_and_min()
        # The artefact has its own floor, from `-mios-version-min`. Rust's
        # default for these targets need not match CPython's, and if it is
        # *higher* then the interpreter's number is the wrong one to publish --
        # the wheel would claim to run on an iOS that cannot load its own
        # extension. Take the higher of the two and say which won.
        info = macho_info(artefact) or {}
        art_min = info.get("minos")
        if art_min and art_min[:2] > (major, minor):
            print(f"  tag floor from the artefact ({art_min[0]}.{art_min[1]}), "
                  f"not from CPython ({major}.{minor})")
            major, minor = art_min[0], art_min[1]
        tag = f"ios_{major}_{minor}_{multiarch}"
        _confirm_with_packaging(tag, "ios", version=(major, minor),
                                multiarch=multiarch)
        return tag

    def cc(self) -> list[str]:
        _, (major, minor) = self._multiarch_and_min()
        triple = f"arm64-apple-ios{major}.{minor}"
        if self.platform_id == "iossimulator":
            triple += "-simulator"
        return [
            "xcrun", "--sdk", self.sdk, "clang",
            "-target", triple, "-dynamiclib",
            "-install_name", f"@rpath/{self.global_deps_name}",
        ]

    def check_image(self, data: bytes, what: str) -> None:
        info = macho_info(data)
        if info is None:
            _fail(f"{what} is not a thin 64-bit Mach-O ({describe(data)})")
        if info["arch"] != "arm64":
            _fail(f"{what} is {info['arch']}, expected arm64")
        if info["platform"] != self.platform_id:
            _fail(
                f"{what} is built for {info['platform']!r}, not "
                f"{self.platform_id!r}. A wheel tagged for one and holding the "
                "other installs and then fails only on the real thing"
            )

    def check_artefact(self, data: bytes) -> None:
        super().check_artefact(data)
        # Device only. The simulator resolves CPython symbols the way macOS
        # does -- `.cargo/config.toml` gives it `-undefined dynamic_lookup`, so
        # it carries no Python dependency at all and demanding one here would
        # fail a correct artefact. On a physical device there is no libpython to
        # fall back on, so the framework has to be named in the load commands.
        info = macho_info(data) or {}
        if self.platform_id == "ios" and not any(
            "Python.framework" in d for d in info.get("dylibs", ())
        ):
            _fail(
                f"{self.artefact} does not link Python.framework "
                f"(LC_LOAD_DYLIB: {info.get('dylibs')}). See "
                "docs/RUST_CROSSBUILD.md §0.5, and check "
                "TORCHNATIVE_PYTHON_FRAMEWORK_DIR"
            )


def _confirm_with_packaging(tag: str, family: str, **kwargs) -> None:
    """Ask `packaging` whether an installer would accept this tag.

    The point is not to *derive* the tag from packaging -- its generators need a
    running target interpreter's answers, which is precisely what a cross build
    does not have. It is to check the spelling against the code pip actually
    uses, so that a mistake here is a build failure rather than a wheel that no
    device will match. Skipped, loudly, if `packaging` is too old to know the
    family: that is the state in which the check is worthless, not the state in
    which it passes.
    """
    try:
        from packaging import tags as ptags
    except ImportError:
        print("  ! packaging not importable -- tag spelling unchecked")
        return
    generator = getattr(ptags, f"{family}_platforms", None)
    if generator is None:
        print(f"  ! packaging {getattr(__import__('packaging'), '__version__', '?')}"
              f" has no {family}_platforms -- tag spelling unchecked")
        return
    accepted = list(generator(**kwargs))
    if tag not in accepted:
        _fail(f"packaging.tags.{family}_platforms{kwargs} does not yield {tag!r};"
              f" it starts {accepted[:3]}")
    print(f"  tag {tag} accepted by packaging.tags.{family}_platforms")


TARGETS: dict[str, Target] = {
    t.key: t for t in (
        AndroidTarget(),
        IOSTarget("ios-arm64", "aarch64-apple-ios", "arm64-iphoneos",
                  "iphoneos", "ios"),
        IOSTarget("ios-arm64-sim", "aarch64-apple-ios-sim",
                  "arm64-iphonesimulator", "iphonesimulator", "iossimulator"),
    )
}


def _repack(wheel: Path, extra: dict[str, bytes], dist_info: str,
            plat: str | None = None,
            overrides: dict[str, bytes] | None = None) -> Path:
    """Rewrite the archive with `extra` added and RECORD regenerated.

    zipfile cannot delete or replace a member, so the whole archive is rebuilt.
    RECORD has to be regenerated anyway: it is a hash manifest, and pip verifies
    it on install, so appending files without touching it produces a wheel that
    fails at exactly the moment it looks like it worked.

    `plat` forces the platform tag (cross builds know theirs; the host derives
    it below from the extension). `overrides` replaces the content of members
    that are already there -- which is how a cross wheel gets the cross `_C`
    without the source tree ever holding it.
    """
    tmp = wheel.with_suffix(".whl.tmp")
    record_name = f"{dist_info}/RECORD"
    wheel_name = f"{dist_info}/WHEEL"
    rows: list[tuple[str, str, str]] = []
    overrides = overrides or {}
    old_plat = wheel.stem.split("-")[-1]
    new_plat: str | None = plat if plat and plat != old_plat else None

    # The architecture correction has to be known before WHEEL is written, and
    # it comes out of the extension, so read that first. Only for the host: a
    # cross target was told its tag and must not have it second-guessed by a
    # macOS-shaped rule.
    if plat is None:
        with zipfile.ZipFile(wheel) as src:
            so = next(
                (n for n in src.namelist() if n.endswith("torch/_C.abi3.so")), None
            )
            if so is not None and sys.platform == "darwin":
                so_data = src.read(so)
                honest = honest_macos_plat(old_plat, so_data)
                if honest != old_plat:
                    print(f"  retag: {old_plat} -> {honest} "
                          f"(extension is {'+'.join(macho_arches(so_data))})")
                    new_plat = honest
    elif new_plat:
        print(f"  retag: {old_plat} -> {new_plat}")

    unused = set(overrides) - set(zipfile.ZipFile(wheel).namelist())
    if unused:
        # An override that matches nothing is the wheel keeping its host
        # extension while everything else says it is a cross wheel -- silent,
        # and exactly the failure this whole path exists to avoid.
        _fail(f"override(s) name members the wheel does not have: {sorted(unused)}")

    with zipfile.ZipFile(wheel) as src, zipfile.ZipFile(
        tmp, "w", zipfile.ZIP_DEFLATED
    ) as dst:
        for item in src.infolist():
            if item.filename == record_name:
                continue  # regenerated at the end
            data = overrides.get(item.filename) or src.read(item.filename)
            if item.filename.endswith("torch/_C.abi3.so"):
                data = _fix_install_name(data)
                item = zipfile.ZipInfo(item.filename, date_time=item.date_time)
                item.external_attr = 0o755 << 16
                item.compress_type = zipfile.ZIP_DEFLATED
            elif item.filename == wheel_name and new_plat:
                # `Tag:` in WHEEL and the filename have to agree; installers read
                # the filename, `wheel unpack` and auditors read this.
                data = b"".join(
                    (line.rsplit(b"-", 1)[0] + b"-" + new_plat.encode() + b"\n")
                    if line.startswith(b"Tag: ") else line + b"\n"
                    for line in data.splitlines()
                )
            dst.writestr(item, data)
            rows.append(_record_line(item.filename, data))

        for arcname, data in sorted(extra.items()):
            info = zipfile.ZipInfo(arcname, date_time=(1980, 1, 1, 0, 0, 0))
            info.external_attr = 0o644 << 16
            info.compress_type = zipfile.ZIP_DEFLATED
            dst.writestr(info, data)
            rows.append(_record_line(arcname, data))

        buf = io.StringIO()
        writer = csv.writer(buf, lineterminator="\n")
        for row in sorted(rows):
            writer.writerow(row)
        writer.writerow([record_name, "", ""])
        dst.writestr(record_name, buf.getvalue())

    final = wheel if new_plat is None else wheel.with_name(
        _retag(wheel.name, new_plat)
    )
    tmp.replace(final)
    if final != wheel:
        wheel.unlink()
    return final


# A translation unit with one symbol, so that `nm` on the result says what it is
# instead of showing an empty library and leaving the next reader to guess.
_GLOBAL_DEPS_C = """
/* Generated by tools/wheel/build.py. See docs/WHEEL.md. */
const char torchnative_global_deps_note[] =
    "torchnative: this build has no global native dependencies; "
    "torch/__init__.py:_load_global_deps only needs this file to dlopen.";
"""


def global_deps_stub(target: "Target | None" = None) -> dict[str, bytes]:
    """An empty `torch/lib/libtorch_global_deps` so `import torch` needs no env.

    Wall 1 in docs/VENDOR.md: `torch/__init__.py:_load_global_deps()` does an
    unconditional `ctypes.CDLL(torch/lib/libtorch_global_deps.<ext>,
    RTLD_GLOBAL)`, and `vendor_torch.sh` drops `torch/lib/` entirely. Every
    invocation in this repository routes around it with
    `TORCH_USE_RTLD_GLOBAL=1`, which upstream provides for exactly this case
    ("a build environment where libtorch_global_deps is not available").

    That is fine for a source tree and unacceptable for a wheel. `pip install
    torchnative && python -c "import torch"` has to work, and an environment
    variable the user has never heard of is not a thing they will set. Worse,
    the wall-1 branch is not free: it also does
    `sys.setdlopenflags(RTLD_GLOBAL | RTLD_LAZY)` for the whole `from torch._C
    import *`, whose device-side effect VENDOR.md §7 item 4 lists as unverified.
    Shipping the file takes the ordinary branch instead.

    Upstream's copy exists to force MKL/OpenMP into the global namespace ahead
    of libtorch. This build links neither -- `_C.abi3.so` is self-contained
    against Accelerate -- so *empty* is not a stub standing in for something
    real, it is the correct content. The file has to be a loadable image
    though, not the zero-byte marker that `install_shim.sh` uses for
    `torch/bin/torch_shm_manager`; `CDLL` actually dlopens this one.

    Built here rather than in `install_shim.sh` so the source tree keeps
    behaving exactly as every existing doc describes -- wall 1 still stands
    there, and the suites that set the variable are still measuring what they
    say they measure.

    For a cross target the compiler is the target's, not the host's. Handing the
    host `cc` an Android wheel puts a Mach-O where the device expects an ELF;
    `CDLL` then raises OSError, `_load_global_deps` swallows it into the
    `_preload_cuda_deps` path, and the import dies somewhere with no mention of
    this file. The name changes too -- see `Target.global_deps_name`.
    """
    import tempfile

    if target is None:
        name = "libtorch_global_deps" + (
            ".dylib" if sys.platform == "darwin" else ".so")
        cc = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
        if not cc:
            _fail(
                "no C compiler (set CC) -- cannot build the empty "
                f"torch/lib/{name}, without which the installed "
                "wheel cannot `import torch` unless the user sets "
                "TORCH_USE_RTLD_GLOBAL=1 (docs/VENDOR.md wall 1)"
            )
        argv = [cc, "-shared", "-fPIC"]
        # Match the deployment target the wheel will be *tagged* with. Without
        # this the host `cc` stamps LC_BUILD_VERSION with the SDK it happens to
        # have -- measured at `macos 26.0+` inside a wheel tagged
        # `macosx_11_0_arm64` -- and dyld enforces that field, so the file the
        # tag promises would be unloadable on every macOS the tag claims. Same
        # class of mistake as the `universal2` tag in docs/WHEEL.md §3.3: the
        # archive contents and the tag disagreeing, in the direction that only
        # shows up on somebody else's machine.
        if sys.platform == "darwin":
            import sysconfig
            floor = sysconfig.get_config_var("MACOSX_DEPLOYMENT_TARGET")
            if floor:
                argv.append(f"-mmacosx-version-min={floor}")
    else:
        name = target.global_deps_name
        argv = target.cc()

    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "global_deps.c"
        out = Path(tmp) / name
        src.write_text(_GLOBAL_DEPS_C)
        r = subprocess.run([*argv, "-o", str(out), str(src)],
                           capture_output=True, text=True)
        if r.returncode != 0:
            _fail(f"{argv[0]} could not build the global-deps stub:\n{r.stderr}")
        data = out.read_bytes()
        # The compiler was asked for a target; check that it produced one. A
        # wrong `--sdk` or a host `cc` on $PATH is otherwise invisible until the
        # device refuses the file.
        if target is not None:
            target.check_global_deps(data)
        print(f"  + torch/lib/{name} "
              f"({len(data):,} B, empty by design -- VENDOR.md wall 1)"
              f"\n      {describe(data)}")
        return {f"torch/lib/{name}": data}


def upstream_dist_info(version: str) -> dict[str, bytes]:
    """Upstream's `torch-<v>.dist-info`, addressed into the wheel's purelib.

    Why it cannot simply sit at the archive root: pip resolves a wheel's own
    metadata directory by scanning top-level names ending in `.dist-info` and
    raising `UnsupportedWheel: multiple .dist-info directories found` if there
    is more than one. Routing it through `torchnative-<v>.data/purelib/` puts it
    in site-packages at install time while leaving exactly one at the root.

    Upstream's RECORD is deliberately not copied (vendor_torch.sh drops it too).
    Ours lists these files, so `pip uninstall torchnative` removes them; a
    RECORD of its own would additionally invite `pip uninstall torch` to delete
    the tree out from under us.
    """
    found = sorted(SRC.glob("torch-*.dist-info"))
    if not found:
        print("  ! no upstream torch-*.dist-info in the vendored tree --")
        print("    importlib.metadata.version('torch') will raise, and")
        print("    transformers' is_torch_available() will answer False")
        return {}
    root = found[0]
    prefix = f"torchnative-{version}.data/purelib/{root.name}"
    out: dict[str, bytes] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "RECORD":
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        out[f"{prefix}/{path.relative_to(root).as_posix()}"] = path.read_bytes()
    return out


def verify(wheel: Path, expected: set[str], target: "Target | None") -> None:
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
        missing = sorted(n for n in expected if n not in names)
        if missing:
            head = "\n".join("    " + m for m in missing[:20])
            _fail(
                f"{len(missing)} file(s) present in {SRC} but absent from the "
                f"wheel:\n{head}\n"
                + ("    ...\n" if len(missing) > 20 else "")
                + "  a wheel that is merely smaller than the tree installs fine "
                "and\n  fails later, at whichever import first needs the "
                "missing file"
            )
        if "-none-any" in wheel.name:
            _fail(f"{wheel.name} is tagged py3-none-any -- setup.py's "
                  "BinaryDistribution did not take effect")
        if "-abi3-" not in wheel.name:
            _fail(f"{wheel.name} is not abi3-tagged -- the bdist_wheel "
                  "py_limited_api option did not take effect")

        if target is None:
            return

        # Read the finished archive rather than trusting that the override and
        # the extra went in. This is the only place that looks at what is
        # actually in the file that will be published.
        plat = wheel.stem.split("-")[-1]
        member = next(n for n in names if n.endswith("torch/_C.abi3.so"))
        target.check_image(zf.read(member), f"{wheel.name}::{member}")
        deps = f"torch/lib/{target.global_deps_name}"
        if deps not in names:
            _fail(f"{wheel.name} has no {deps}")
        target.check_image(zf.read(deps), f"{wheel.name}::{deps}")
        strays = sorted(
            n for n in names
            if n.startswith("torch/lib/libtorch_global_deps") and n != deps
        )
        if strays:
            # `_load_global_deps` looks for exactly one name. A second one is a
            # host artefact that came along for the ride.
            _fail(f"{wheel.name} carries {strays} beside {deps}")
        if not plat.startswith(("android_", "ios_")):
            _fail(f"{wheel.name} was built for {target.key} but is tagged "
                  f"{plat!r}")


def self_test() -> None:
    """Drive `artefact_verdict` through all three of its answers.

    A freshness check has one way to be useless and it is not being wrong: it
    is answering "fresh" for a reason that has nothing to do with the artefact.
    Six of the eight cases below are the check *failing to run*, and each has to
    come back `unknown` rather than `fresh` -- particularly `no inputs listed`,
    where `max()` over an empty list has no error to raise and would simply
    conclude that nothing is newer than the artefact.

    Costs no build: the artefact is a scratch file with a chosen mtime and the
    dep-info is written by hand. The prerequisites are real files under
    `rust/torch_c`, because the containment rule is one of the things under
    test.
    """
    import tempfile

    real = sorted(str(p) for p in (CRATE / "src").glob("*.rs"))
    if len(real) < 2:
        _fail(f"self-test needs source files under {CRATE / 'src'}; found {real}")
    hour = 3600 * 10**9

    cases: list[tuple[str, str, str]] = []   # (label, expected, fragment)
    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)

        def scenario(name: str, dep_text: str | None, *, artefact_age_h: float
                     ) -> tuple[str, str]:
            art = tmpdir / f"{name}.dylib"
            art.write_bytes(b"\xcf\xfa\xed\xfe" + b"\0" * 60)
            when = art.stat().st_mtime_ns - int(artefact_age_h * hour)
            os.utime(art, ns=(when, when))
            dep = art.with_suffix(".d")
            if dep_text is not None:
                dep.write_text(dep_text.replace("@ART@", str(art)))
            return artefact_verdict(art)

        checks: list[tuple[str, str, str, str | None, float]] = [
            # label, expected verdict, expected phrase, dep-info, artefact age
            ("current artefact", FRESH, "recorded inputs",
             "@ART@: " + " ".join(real[:2]), 0.0),
            ("a prerequisite modified after the build", STALE,
             "was modified", "@ART@: " + " ".join(real[:2]), 48.0),
            ("no dep-info at all", UNKNOWN, "does not exist", None, 0.0),
            ("dep-info with no rule in it", UNKNOWN, "no Makefile rule",
             "not a makefile\njust prose\n", 0.0),
            ("dep-info describing a different artefact", UNKNOWN,
             "some other build's record",
             "/elsewhere/lib_Other.dylib: " + real[0], 0.0),
            ("rule with no prerequisites", UNKNOWN, "lists no inputs",
             "@ART@:", 0.0),
            ("a prerequisite that is gone", UNKNOWN, "cannot be stat'ed now",
             f"@ART@: {CRATE}/src/deleted_by_the_test.rs", 0.0),
            ("prerequisites from another checkout", UNKNOWN,
             "different checkout",
             "@ART@: /some/other/worktree/rust/torch_c/src/lib.rs", 0.0),
        ]

        print(f"SELF-TEST of the artefact freshness check ({len(checks)} cases)")
        bad = 0
        for i, (label, expected, phrase, dep_text, age) in enumerate(checks):
            verdict, detail = scenario(f"case{i}", dep_text, artefact_age_h=age)
            ok = verdict == expected and phrase in detail
            bad += not ok
            print(f"  {'ok    ' if ok else 'WRONG '}{verdict:<8}{label}")
            if not ok:
                print(f"          expected {expected} containing {phrase!r}")
                print(f"          got      {verdict}: {detail}")
            cases.append((label, verdict, detail))

    # The verdict is internal; what reaches a reader is the refusal message. The
    # trap this shape exists to avoid is a check reporting its own failure as a
    # finding about the artefact, so assert on the rendered text of both kinds
    # -- a `stale` message has to make the staleness claim and an `unknown` one
    # has to disclaim it, in so many words.
    counts = {FRESH: 0, STALE: 0, UNKNOWN: 0}
    for _, verdict, _ in cases:
        counts[verdict] += 1
    if not (counts[STALE] and counts[UNKNOWN] and counts[FRESH]):
        _fail(f"SELF-TEST: the cases do not reach all three verdicts: {counts}")
    for verdict, phrase, must_not in (
        (STALE, "is stale.", "NOT a staleness report"),
        (UNKNOWN, "This is NOT a staleness report", "is stale."),
    ):
        for label, got, detail in cases:
            if got != verdict:
                continue
            rendered = _render_refusal(verdict, Path("/scratch/x.dylib"), detail,
                                       "REBUILD-HINT")
            if phrase not in rendered or must_not in rendered:
                _fail(f"SELF-TEST: the {verdict} message for {label!r} does not "
                      f"read as one -- it must contain {phrase!r} and must not "
                      f"contain {must_not!r}:\n{rendered}")
    print(f"\n  {counts[STALE]} staleness report(s), "
          f"{counts[UNKNOWN]} 'cannot judge', {counts[FRESH]} clean -- and the "
          "two refusal texts assert they are not each other")
    if bad:
        sys.exit(f"SELF-TEST: FAIL -- {bad}/{len(cases)} cases answered wrongly")
    print(f"SELF-TEST: PASS -- {len(cases)}/{len(cases)} cases answered as "
          "specified")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter whose pip/setuptools drive the build")
    ap.add_argument("--outdir", default=str(REPO / "dist"), type=Path)
    ap.add_argument("--target", default="host",
                    choices=["host", *sorted(TARGETS)],
                    help="cross target; default is this machine")
    ap.add_argument("--self-test", action="store_true",
                    help="exercise the artefact freshness check and exit; "
                         "builds nothing")
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    target = None if args.target == "host" else TARGETS[args.target]

    # The host preflight runs for cross builds too, unchanged: `pip wheel` walks
    # the same source tree either way, so a missing vendored tree or a missing
    # host `_C` produces the same empty shell it always did. The cross artefact
    # is an *additional* requirement, never a substitute for those.
    stamp = preflight()
    print(f"vendored torch {stamp.get('version', '?')} "
          f"({stamp.get('py_modules', '?')} modules) + _C.abi3.so "
          f"({SHIM.stat().st_size:,} B)")

    # Every build packages the host shim -- a cross build overrides the member
    # afterwards, but `pip wheel` still walks the source tree to get there, and
    # `verify` compares the archive against that same tree. So the host artefact
    # is checked on both paths, exactly as `preflight` is.
    check_host_shim()

    overrides: dict[str, bytes] = {}
    plat: str | None = None
    if target is not None:
        if not target.artefact.exists():
            _fail(
                f"no cross-built extension at {target.artefact}\n"
                f"  build it for {target.rust_target} first "
                "(scripts/device_android.sh build, or docs/RUST_CROSSBUILD.md "
                "§0.5 for iOS)\n"
                "  CARGO_TARGET_DIR is currently "
                f"{CARGO_TARGET_DIR}"
            )
        # Existence was the only question this asked until 2026-08-29, and a
        # five-day-old artefact answers it just as well as a current one.
        require_current(target.artefact, target.rebuild_hint)
        cross = target.artefact.read_bytes()
        target.check_artefact(cross)
        print(f"target {target.key}: {target.artefact.name} "
              f"({len(cross):,} B)\n      {describe(cross)}")
        plat = target.platform_tag(cross)
        overrides["torch/_C.abi3.so"] = cross

    args.outdir.mkdir(parents=True, exist_ok=True)
    wheel = run_pip_wheel(args.python, args.outdir)

    version = wheel.name.split("-")[1]
    extra = {**upstream_dist_info(version), **global_deps_stub(target)}
    wheel = _repack(wheel, extra, f"torchnative-{version}.dist-info",
                    plat=plat, overrides=overrides)

    expected: set[str] = set()
    for pkg in ("torch", *stamp.get("packages", "").split(","), "torchnative"):
        if pkg and (SRC / pkg).is_dir():
            expected |= tree_files(SRC / pkg)
    verify(wheel, expected, target)

    with zipfile.ZipFile(wheel) as zf:
        entries = zf.namelist()
        uncompressed = sum(i.file_size for i in zf.infolist())
    print()
    print(f"{wheel}")
    print(f"  {len(entries):,} entries, "
          f"{wheel.stat().st_size / 1e6:.1f} MB compressed, "
          f"{uncompressed / 1e6:.1f} MB installed")
    print(f"  tag: {'-'.join(wheel.stem.split('-')[2:])}")
    print()
    if target is None:
        print(f"next: python tools/wheel/verify.py {wheel}")
    elif target.key.startswith("android"):
        print(f"next: python tools/wheel/verify_cross.py {wheel}")
        print(f"      python tools/wheel/verify_android.py {wheel}")
    else:
        print(f"next: python tools/wheel/verify_cross.py {wheel}")


if __name__ == "__main__":
    main()
