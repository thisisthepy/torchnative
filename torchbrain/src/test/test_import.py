def test_import_torchbrain():
    import torchbrain  # noqa: F401


def test_import_submodules():
    from torchbrain import adapt, delta, kernels  # noqa: F401
