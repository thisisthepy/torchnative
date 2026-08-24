"""Deployment, lifetime policy and device orchestration.

The `torch` import is deferred rather than done at module scope. This
distribution *provides* `torch` -- the vendored tree plus our `_C` -- rather
than depending on someone else's, so at import time there may not be one yet:
the wheels that carry the tree are not wired up, and until they are, importing
`torchnative` should not require a PyTorch that this package is itself meant to
supply. See DESIGN.md §2 and the README's Status section.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch import nn


class TorchNativeAPI(object):
    def __init__(self, *args, **kwargs):
        pass

    def deploy(self, model: "nn.Module"):
        pass
