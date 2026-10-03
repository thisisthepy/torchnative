"""Stage-1 adaptation methods: the ones that descend a loss and need a backward.

DESIGN.md §3 splits adaptation by differentiation requirement, and INTENT §4
asks for the split to be **a type, so that a backward-free build rejects
gradient methods at import time** rather than at the first step. This module is
that type's home. :class:`GradientMethod` is the stage-1 base and :class:`Tent`
is the one stage-1 method shipped; ``torchnative.adapt`` itself holds only
stage 0 (:class:`~torchnative.adapt.StatisticsMethod`,
:class:`~torchnative.adapt.BatchNormStats`) and the wrapper, and reaches the
names here lazily -- so a stage-0 deployment never executes this file.

**What a backward-free build is, in this repository today.** No build of
``torch._C`` from this crate lacks the tape -- ``tape::register`` is
unconditional in ``torchnative/rust/torch_c/src/lib.rs`` -- so there is no artefact that
could be probed for the absence. What exists is a configuration:
``TORCHNATIVE_BACKWARD=off`` in the environment at ``torchnative.adapt`` import
time (``torchnative.adapt.BACKWARD`` reports what was read). Under it this
module refuses to import, naming itself and the reason, and every
:class:`~torchnative.adapt.Method` subclass declaring stage 1 or 2 is refused
when its class statement runs -- which is when the module that defines it is
imported. SPEC S6.5; held by ``tests/test_stagetype.py``.
"""

from __future__ import annotations

import torch

from torchnative import adapt as _adapt

# Before any class statement below. The class statements would be refused too
# -- `Method.__init_subclass__` reads the same configuration -- but that
# refusal names one class, and what a caller who wrote
# `from torchnative.adapt import Tent` needs to hear is that the whole module of
# stage-1 methods is out of this build, and why.
if not _adapt.BACKWARD:
    raise ImportError(
        "torchnative.adapt.gradient: this module holds the stage-1 (gradient) "
        "adaptation methods -- GradientMethod, Tent -- and stage 1 needs a "
        "backward, which this build is configured without (%s=%r). Refused at "
        "import time rather than at the first step, so that a gradient method "
        "cannot be wired into a stage-0 deployment (INTENT §4, SPEC S6.5).\n"
        "Stage 0 is available: torchnative.adapt.StatisticsMethod, "
        "torchnative.adapt.BatchNormStats.\n"
        "Check: torchnative.adapt.BACKWARD"
        % (_adapt.BACKWARD_ENV, _adapt.BACKWARD_SETTING),
        name=__name__,
    )

__all__ = ["GradientMethod", "Tent"]


class GradientMethod(_adapt.Method):
    """A stage-1 method: descend a scalar on selected parameters, through a backward.

    Declares :meth:`~torchnative.adapt.Method.select` (the *parameter* names a
    ``Delta`` is opened over) and :meth:`~torchnative.adapt.Method.objective`
    (the scalar to descend). :class:`~torchnative.adapt.Adapted` accepts a
    stage-1 method only as an instance of this class, so the stage is decided by
    the type and not by an attribute that anything could set.
    """

    stage = _adapt.STAGE_NARROW_BACKWARD


class Tent(GradientMethod):
    """Entropy minimisation on the normalisation affine parameters.

    Wang et al., *Tent: Fully Test-Time Adaptation by Entropy Minimization*
    (ICLR 2021): adapt to unlabelled test data by descending the entropy of the
    model's own predictions, moving only the affine parameters of the
    normalisation layers.

    It is DESIGN.md §3's stage 1 -- "optimisation-based, normalisation
    calibration, updating the affine {gamma, beta} by a loss" -- and that is
    the row of the survey table that needs a backward. The row above it, which
    only recomputes statistics, needs none; both are normalisation calibration,
    which is why the stage is declared by type and not read off a directory
    name.

    **What is deliberately not done.** The paper also puts normalisation layers
    into batch-statistic mode, because its models are BatchNorm ones and the
    test-time statistics are half the method. This selects and updates affine
    parameters only. On a model whose normalisation has no running statistics --
    LayerNorm, RMSNorm, so every transformer -- the two halves coincide and
    nothing is missing. On a BatchNorm model they do not.

        Check: any(hasattr(m, "running_mean") and m.running_mean is not None
                   for m in model.modules())

    If that is True for your model, this class is implementing half of Tent and
    :meth:`select` will tell you which layers it picked.
    """

    def __init__(self, select=None, affine_only=True):
        self._select = select
        self.affine_only = affine_only

    def select(self, model):
        if self._select is not None:
            return list(self._select(model) if callable(self._select) else self._select)
        names = []
        for mod_name, module in model.named_modules():
            if not _adapt._is_normalisation(module):
                continue
            for own_name, param in module.named_parameters(recurse=False):
                if self.affine_only and own_name not in ("weight", "bias"):
                    continue
                names.append(f"{mod_name}.{own_name}" if mod_name else own_name)
        return names

    def objective(self, outputs):
        """Mean prediction entropy, over every position of the batch.

        ``-(p * log p).sum(-1)`` with ``p`` from ``softmax`` and ``log p`` from
        ``log_softmax`` rather than from ``log(p)``: the second spelling is one
        op shorter and loses the large negative logits, which at a 49152-wide
        vocabulary is most of them.

        Both spellings have derivative rules, so the choice here is numerical
        and not a matter of what the tape can carry.
        """
        logits = _adapt._logits_of(outputs)
        p = torch.softmax(logits, dim=-1)
        logp = torch.log_softmax(logits, dim=-1)
        return -(p * logp).sum(-1).mean()
