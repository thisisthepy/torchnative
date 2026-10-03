import os, sys  # moved from torchnative/src/test/ in #43; the gate now runs it

# The shim imports upstream extension libraries that resolve only with global
# symbols, as every other suite that imports torch sets.
os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "python"))


def test_import_torchnative():
    import torchnative  # noqa: F401


def test_import_submodules():
    from torchnative import adapt, delta, kernels  # noqa: F401


def test_import_export_does_not_need_torch():
    """`torchnative.export` must import without a `torch` on the interpreter.

    Same rule `torchnative.api` follows, for the reason its docstring gives:
    this distribution *provides* torch rather than depending on one, so a
    module that reaches for it at import time fails on a correctly-installed
    wheel. Every `import torch` in the decomposition pass is inside a function,
    and this is what says so.
    """
    from torchnative import export

    assert callable(export.decompose)
    assert callable(export.core_ops)


if __name__ == "__main__":
    import _skip
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_")]
    raise SystemExit(1 if _skip.run_tests(tests, suite="test_import") else 0)
