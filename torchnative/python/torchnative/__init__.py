"""torchnative: on-device test-time learning and federated learning.

See docs/design/DESIGN.md for the design and its reasoning.

**Every subpackage** is resolved lazily, through PEP 562, for the reason
`torchnative.quant` already gives: importing this package must stay cheap and
must not drag in `torch` (let alone `transformers`, which is not a hard
dependency) for someone who imported it for something else.
`import torchnative; torchnative.device.npu` works because the attribute lookup
lands here and imports the submodule then, and so does
`torchnative.adapt.wrap` -- which it did not until 2026-09-12, because only
`device` and `transformers` were listed below.

That laziness is also what makes the `nn.Module.to` patch safe to install at
`torchnative.device` import time: see `device/_module_to.py`.
"""

# Every subpackage, not a chosen two. `__getattr__` below refuses anything not
# named here, so a subpackage left off this list is unreachable through the
# package object -- `torchnative.adapt` raised `AttributeError` while
# `adapt/__init__.py` sat on disk with `wrap` in it, and the same for the other
# six. `from torchnative import adapt` always worked, which is why nothing
# noticed: the import system binds the submodule onto the parent as a side
# effect, and every example in this tree is spelled that way.
#
# Written out rather than scanned off `__path__`: a directory listing at import
# time is a stat call per entry on a road that exists to be cheap, and it would
# also make `__all__` depend on what happens to be installed. Added here is
# added on purpose. `tests/api/test_tnnamespace.py` compares this
# list against the disk and fails if a new subpackage is not added.
__all__ = [
    "adapt",
    "api",
    "delta",
    "device",
    "distributed",
    "export",
    "gguf",
    "kernels",
    "nn",
    "quant",
    "transformers",
]


def _version():
    """The release this package is, without declaring it a second time.

    `pyproject.toml` is the one place a version is written
    (`tests/release/test_release.py`). Run from this source tree, the package
    sits at `torchnative/python/torchnative/` and that file is three levels
    up; an installed wheel has no such file and answers from its
    distribution metadata instead.
    """
    import pathlib

    pyproject = pathlib.Path(__file__).resolve().parents[3] / "pyproject.toml"
    if pyproject.is_file():
        import tomllib

        with open(pyproject, "rb") as f:
            project = tomllib.load(f).get("project", {})
        if project.get("name") == "torchnative" and "version" in project:
            return project["version"]
    from importlib import metadata

    try:
        return metadata.version("torchnative")
    except metadata.PackageNotFoundError:
        raise AttributeError(
            "torchnative.__version__: no installed `torchnative` distribution "
            "metadata and no torchnative pyproject.toml beside this source "
            "tree, so nothing here records which release this is"
        ) from None


def __getattr__(name):
    if name in __all__:
        import importlib

        return importlib.import_module(f"{__name__}.{name}")
    if name == "__version__":
        # Lazy for the reason the subpackages are: `import torchnative` stays
        # cheap. Cached once read.
        globals()["__version__"] = version = _version()
        return version
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals()) + __all__)
