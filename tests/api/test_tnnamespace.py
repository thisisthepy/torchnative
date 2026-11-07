"""`import torchnative` and then reach for what the package says it is.

`torchnative/__init__.py`'s first line is "on-device test-time learning and
federated learning". Both of those are subpackages on disk -- `adapt/` and
`nn/federated/` -- and until 2026-09-12 neither could be reached through the
package object:

    >>> import torchnative
    >>> torchnative.adapt
    AttributeError: module 'torchnative' has no attribute 'adapt'

`__init__.py` resolves submodules through PEP 562, which is the right shape and
is there for a stated reason (the import must stay cheap and must not drag in
`torch`). The defect was that its `__all__` listed **two** of the nine
subpackages, and `__getattr__` refuses anything not in `__all__` -- so the two
that happened to be listed worked and the other seven raised, including both
of the two the docstring names as the package's purpose.

**Why this was invisible.** `from torchnative import adapt` works regardless:
that is the import system, which finds the submodule on `__path__` and *binds
it onto the parent module as a side effect*. Every docstring and document in
this tree spells it that way, so every reader and every test took the road that
cannot see the hole. Only the attribute road on a package that has not already
been imported hits it -- which is why the walk below runs each name in a
**fresh subprocess**. Done in-process, the first `from torchnative import x`
anywhere in the session would bind the attribute and the test would pass
against the unfixed code.

The list of names is read off the directory rather than written down here, so a
subpackage added later is covered without anyone remembering this file.

Nullifications this file is meant to catch, each verified by making the break
and watching it go red:

* a subpackage dropped from `__all__`      -> `test_every_subpackage_is_reachable_as_an_attribute`
* the walk run in-process (vacuous)        -> the subprocess is the test; see
                                              `test_the_subprocess_probe_can_still_fail`
* laziness given up to fix it              -> `test_importing_torchnative_does_not_import_torch`
"""

import os
import subprocess
import sys

_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "python")
_PKG = os.path.join(_VENDOR_DIR, "torchnative")

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")


def _subpackages():
    """Every subpackage of `torchnative`, read off the disk.

    `nn.federated` is deliberately not walked into: the attribute road to it
    goes through `nn`, and `nn` is what this file checks.
    """
    names = []
    for entry in sorted(os.listdir(_PKG)):
        if entry.startswith("_"):
            continue
        if os.path.isfile(os.path.join(_PKG, entry, "__init__.py")):
            names.append(entry)
    return names


def _fresh(code):
    """Run `code` in an interpreter that has never imported torchnative."""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(
        [_VENDOR_DIR] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p]
    )
    env["TORCH_USE_RTLD_GLOBAL"] = "1"
    return subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )


def test_the_subprocess_probe_can_still_fail():
    """Non-vacuity: a name that is genuinely absent must come back as absent.

    Without this, a `_fresh()` that silently always returned success would make
    every check below pass for the wrong reason.
    """
    got = _fresh("import torchnative; torchnative.no_such_subpackage")
    assert got.returncode != 0, got.stdout
    assert "AttributeError" in got.stderr, got.stderr
    print("ok   tnnamespace: the fresh-interpreter probe still reports an absent name")


def test_every_subpackage_is_reachable_as_an_attribute():
    names = _subpackages()
    assert len(names) >= 8, names
    bad = []
    for name in names:
        got = _fresh(f"import torchnative; torchnative.{name}")
        if got.returncode != 0:
            bad.append((name, got.stderr.strip().splitlines()[-1:]))
    assert not bad, (
        "these subpackages exist on disk and cannot be reached through the "
        f"package object: {bad}. `__init__.py`'s `__getattr__` refuses "
        "anything not in `__all__`."
    )
    print(
        f"ok   tnnamespace: all {len(names)} subpackages resolve as attributes "
        f"({', '.join(names)})"
    )


def test_dir_lists_every_subpackage():
    names = _subpackages()
    got = _fresh(
        "import torchnative, json; print(json.dumps(sorted(dir(torchnative))))"
    )
    assert got.returncode == 0, got.stderr
    import json

    listed = set(json.loads(got.stdout))
    missing = [n for n in names if n not in listed]
    assert not missing, missing
    print(f"ok   tnnamespace: dir() names all {len(names)} of them")


def test_importing_torchnative_does_not_import_torch():
    """The reason the resolution is lazy in the first place, kept honest.

    Fixing the attribute road by importing the subpackages eagerly would work
    and would cost exactly what `__init__.py`'s docstring says it must not.
    """
    got = _fresh(
        "import sys, torchnative; "
        "print('torch' in sys.modules, 'transformers' in sys.modules)"
    )
    assert got.returncode == 0, got.stderr
    assert got.stdout.strip() == "False False", got.stdout
    print("ok   tnnamespace: import torchnative pulls in neither torch nor transformers")



def test_the_package_reports_its_version():
    """`torchnative.__version__` is the version being built, not a copy of it.

    The 0.1.0 release audit found no `__version__` at all, so a user could not
    ask an installed wheel which release it was without `importlib.metadata`.
    It is read from the distribution's metadata (an installed wheel), or from
    `pyproject.toml` when the package runs from this source tree, so there is
    still one place a version is declared (`test_release.py`). Checked in a
    fresh interpreter, and checked to stay lazy: answering it must not import
    torch.
    """
    import tomllib

    with open(os.path.join(_ROOT, "pyproject.toml"), "rb") as f:
        want = tomllib.load(f)["project"]["version"]
    got = _fresh(
        "import sys, torchnative; "
        "print(torchnative.__version__); print('torch' in sys.modules)"
    )
    assert got.returncode == 0, got.stderr
    lines = got.stdout.split()
    assert lines[0] == want, (
        f"torchnative.__version__ is {lines[0]!r}, pyproject.toml says {want!r}"
    )
    assert lines[1] == "False", "reading __version__ imported torch"

    # The installed-wheel road: no pyproject.toml beside the package, so the
    # answer must come from the distribution metadata. A version no release
    # has used shows it is read, not echoed from pyproject.
    import shutil
    import tempfile

    with tempfile.TemporaryDirectory(prefix="tn-version-") as tmp:
        site = os.path.join(tmp, "site")
        os.makedirs(os.path.join(site, "torchnative"))
        shutil.copy(os.path.join(_PKG, "__init__.py"), os.path.join(site, "torchnative"))
        info = os.path.join(site, "torchnative-9.9.9b1.dist-info")
        os.makedirs(info)
        with open(os.path.join(info, "METADATA"), "w") as f:
            f.write("Metadata-Version: 2.4\nName: torchnative\nVersion: 9.9.9b1\n")
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        env["PYTHONPATH"] = site
        wheel = subprocess.run(
            [sys.executable, "-c", "import torchnative; print(torchnative.__version__)"],
            env=env, capture_output=True, text=True, cwd=tmp,
        )
    assert wheel.returncode == 0, wheel.stderr
    assert wheel.stdout.strip() == "9.9.9b1", (
        f"an installed package answered {wheel.stdout.strip()!r}, not its metadata's 9.9.9b1"
    )
    print(
        f"ok   tnnamespace: torchnative.__version__ is {want} from the source tree "
        f"and the metadata's version when installed, read without importing torch"
    )

if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
    raise SystemExit(1 if failures else 0)
