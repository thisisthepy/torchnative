"""``torch.nn.modules.test_time``: the module layer of test-time learning.

INTENT section 4 (#237). Test-time learning is a time axis, *when* learning
happens, so its module-side switch sits beside ``train()`` and ``eval()`` rather
than in a paradigm package. The names were set by the leader by convention
under the owner's delegation (owner, 2026-11-13: "컨벤션에 맞게 편성해").

:class:`TestTimeModuleMixin` is organised like
``torch.nn.modules.lazy.LazyModuleMixin``: an opt-in mixin listed before the
module class, reached as ``torch.nn.modules.test_time.TestTimeModuleMixin`` and
not exported at ``torch.nn`` top level::

    from torch.nn.modules.test_time import TestTimeModuleMixin

    class Detector(TestTimeModuleMixin, nn.Module):
        def online_parameters(self):
            return self.backbone.norm_parameters()

    model.online()     # eval(), freeze everything, unfreeze online_parameters()
    model.offline()    # train(), and requires_grad back to what it was

The semantics are test-time-adapters' (``ttadapters.methods.base``,
``AdaptationEngine.online``): online learning is evaluation-mode inference
during which a chosen subset of parameters stays trainable; offline learning
is ordinary training. What changes, and why:

* ``offline()`` restores the ``requires_grad`` each parameter had **when
  ``online`` was entered**. test-time-adapters reads its copy off
  ``state_dict()``, whose tensors are detached, so every flag it restores is
  ``False``. Snapshotting the parameters themselves is the fix.
* Propagation reaches the direct children whose **type** defines ``online``.
  ``hasattr(child, "online")`` would also be true for a module with a
  submodule *named* ``online``, and calling that submodule with ``mode`` would
  run a forward.

The rule that decides what is minimised and when parameters move is not here:
it is ``torch.optim.test_time`` (:mod:`torchnative.optim.test_time`).
"""

from __future__ import annotations

from typing import Iterable, Iterator, Union

import torch
from torch import nn

__all__ = ["TestTimeModuleMixin"]


_ParamsOrGroups = Union[Iterable[nn.Parameter], Iterable[dict]]


def _flatten(params):
    """Parameters, or optimizer-style groups ``{"params": [...], ...}``."""
    for item in params:
        if isinstance(item, dict):
            group = item.get("params")
            if group is None:
                raise ValueError(
                    "torch.nn.modules.test_time: a parameter group without a "
                    "'params' entry: %r" % (sorted(item),)
                )
            if isinstance(group, torch.Tensor):
                group = [group]
            yield from group
        else:
            yield item


class TestTimeModuleMixin:
    """Adds online and offline learning modes to a module.

    ``online(mode=True)`` puts the module in evaluation mode, sets
    ``requires_grad=False`` on every parameter, then ``True`` on each of
    :meth:`online_parameters`. ``offline()`` (``online(False)``) puts it in
    training mode and restores the ``requires_grad`` every parameter had when
    ``online`` was entered. Both then call ``online(mode)`` on each direct
    child whose class defines it, so a child that chooses its own online
    parameters keeps that choice.

    :attr:`adapting` is ``True`` between ``online()`` and ``offline()``, the
    way ``training`` reports ``train()``.
    """

    #: Whether the module is in online-learning mode.
    adapting: bool = False

    def online_parameters(self) -> Iterator[nn.Parameter] | _ParamsOrGroups:
        """The parameters that stay trainable while online.

        Every parameter by default, as test-time-adapters' ``AdaptationEngine``
        does. Override it to choose a subset; optimizer-style groups are
        accepted, so the same answer can build the optimizer.
        """
        return self.parameters()

    def online(self, mode: bool = True):
        """Online learning when ``mode`` is true, offline learning otherwise."""
        if not isinstance(mode, bool):
            raise ValueError(
                "torch.nn.modules.test_time: online(mode) takes a bool, got %r"
                % (type(mode).__name__,)
            )
        if mode:
            if not self.adapting:
                # Taken on entry only: a second online() would otherwise record
                # the frozen state as the original, and offline() would leave
                # the model frozen.
                self.__dict__["_test_time_requires_grad"] = [
                    (p, p.requires_grad) for p in self.parameters()
                ]
            self.eval()
            for p in self.parameters():
                p.requires_grad_(False)
            for p in _flatten(self.online_parameters()):
                p.requires_grad_(True)
            self.__dict__["adapting"] = True
            self._test_time_propagate(True)
        else:
            # Children first: a child entered online after this module had
            # already frozen it, so its own snapshot holds the frozen flags.
            # This module's snapshot, restored last, holds the originals.
            self._test_time_propagate(False)
            self.train()
            for p, flag in self.__dict__.pop("_test_time_requires_grad", ()):
                p.requires_grad_(flag)
            self.__dict__["adapting"] = False
        return self

    def _test_time_propagate(self, mode):
        for child in self.children():
            if callable(getattr(type(child), "online", None)):
                child.online(mode)

    def offline(self):
        """Offline learning: ``online(False)``."""
        return self.online(False)
