def test_import_torchnative():
    import torchnative  # noqa: F401


def test_import_submodules():
    from torchnative import adapt, delta, kernels  # noqa: F401
