"""``torch.optim.test_time``: the rule layer of test-time learning.

INTENT section 4 (#237); names set by the leader by convention under the owner's
delegation (owner, 2026-11-13: "컨벤션에 맞게 편성해"). The module layer, which
says *which mode* a model is in, is ``torch.nn.modules.test_time``
(:class:`~torchnative.nn.modules.test_time.TestTimeModuleMixin`). This layer
says what happens at test time:

* **what is minimised**: :meth:`TestTimeAdaptation.loss`;
* **which parameters move**: the parameters of the optimizer it holds;
* **when an update happens**: :meth:`TestTimeAdaptation.should_update`, and
  ``steps`` updates per batch;
* **reset**: :meth:`TestTimeAdaptation.reset`, parameters and buffers bit for
  bit, and the optimizer's state.

:class:`TestTimeAdaptation` holds an optimizer the way
``torch.optim.lr_scheduler.LRScheduler`` does and is not an optimizer itself::

    from torch.optim.test_time import EntropyMinimization

    model.online()                                   # TestTimeModuleMixin
    optimizer = torch.optim.SGD(model.online_parameters(), lr=1e-3)
    tta = EntropyMinimization(model, optimizer)
    for batch in stream:
        prediction = tta.step(batch)                 # predict, then adapt
    tta.reset()

The rules here are generic and named by objective type:
:class:`EntropyMinimization`, :class:`PseudoLabel` and :class:`NormStatistics`.
Paper-named methods (Tent, EATA, SAR, CoTTA) are compositions of these choices
and live in test-time-adapters, which builds on this layer the way
``transformers`` builds on ``torch``.

**A configuration without a backward** (SPEC S6.5). ``TORCHNATIVE_BACKWARD=off``
when this module is imported says the deployment carries no gradient
adaptation. :data:`BACKWARD` records what was read. Under it the gradient rules
are refused at import by name and reason, a :class:`TestTimeAdaptation`
subclass declaring ``requires_backward = True`` (the default) is refused when
its class statement runs, and :class:`NormStatistics`, which needs no backward,
still works.
"""

from __future__ import annotations

import copy
import os

import torch
import torch.nn.functional as F

from torchnative.delta import BufferSnapshot, Delta


# -- does this configuration have a backward? -------------------------------
#
# A configuration, because every `torch._C` this crate builds carries the tape,
# so there is no artefact whose missing backward a probe could detect (a probe
# whose "no" branch cannot be reached is a check that cannot fail, AGENTS.md
# section 17.5). Read once, at import: a setting that changed after the rules
# were defined would make "refused at import" mean nothing.
BACKWARD_ENV = "TORCHNATIVE_BACKWARD"
_BACKWARD_OFF = frozenset({"0", "off", "no", "false"})
_BACKWARD_ON = frozenset({"1", "on", "yes", "true"})


def _read_backward(value):
    """Unset or empty means a backward is available, which every build has.

    Any other spelling is refused by name: reading it as "on" would put gradient
    rules into a deployment that meant to leave them out, and reading it as
    "off" would refuse a working build over a typo.
    """
    if value is None or not value.strip():
        return True
    v = value.strip().lower()
    if v in _BACKWARD_OFF:
        return False
    if v in _BACKWARD_ON:
        return True
    raise ValueError(
        "torch.optim.test_time: %s=%r is neither on nor off. It says whether "
        "this configuration has a backward, which decides whether the gradient "
        "rules may be imported (SPEC S6.5), so it is not guessed at.\n"
        "Check: set it to one of %s for a configuration without a backward, one "
        "of %s (or unset it) otherwise."
        % (BACKWARD_ENV, value, sorted(_BACKWARD_OFF), sorted(_BACKWARD_ON))
    )


#: The raw ``TORCHNATIVE_BACKWARD`` value this module was imported under.
BACKWARD_SETTING = os.environ.get(BACKWARD_ENV)
#: Whether this configuration admits rules that need a backward. Read once.
BACKWARD = _read_backward(BACKWARD_SETTING)

_GRADIENT_RULES = ("EntropyMinimization", "PseudoLabel")

# This stack's eager autograd keeps one tape per thread (see `_release`).
_EAGER_TAPE = hasattr(torch._C, "_eager_backward")

__all__ = [
    "TestTimeAdaptation", "NormStatistics", "BACKWARD", "BACKWARD_ENV",
] + (list(_GRADIENT_RULES) if BACKWARD else [])


def _no_backward(name):
    return (
        "torch.optim.test_time: %s needs a backward, and this configuration is "
        "imported without one (%s=%r, SPEC S6.5). NormStatistics needs none and "
        "is available.\nCheck: unset %s, or set it to on, in a build that "
        "should adapt by gradient."
        % (name, BACKWARD_ENV, BACKWARD_SETTING, BACKWARD_ENV)
    )


def __getattr__(name):
    # PEP 562: `from torch.optim.test_time import EntropyMinimization` in a
    # configuration without a backward raises this ImportError unchanged,
    # rather than a bare "cannot import name".
    if name in _GRADIENT_RULES and not BACKWARD:
        raise ImportError(_no_backward(name), name=name)
    raise AttributeError("module %r has no attribute %r" % (__name__, name))


def _logits_of(outputs):
    """The prediction an objective is a function of.

    A ``transformers`` model returns a ``ModelOutput`` carrying ``.logits``; an
    ``nn.Module`` returns a tensor. Anything else is refused by name: a guess
    would minimise something about another tensor and still report a step.
    """
    if isinstance(outputs, torch.Tensor):
        return outputs
    logits = getattr(outputs, "logits", None)
    if isinstance(logits, torch.Tensor):
        return logits
    raise TypeError(
        "torch.optim.test_time: the model returned %r, which is neither a "
        "Tensor nor something with .logits, so there is no prediction to "
        "minimise over. Override loss(outputs)." % (type(outputs).__name__,)
    )


def _rows(logits):
    """Class scores as ``(samples, classes)``: a sequence model's positions are samples."""
    if logits.dim() < 2:
        raise ValueError(
            "torch.optim.test_time: logits of shape %s have no class dimension"
            % (tuple(logits.shape),)
        )
    return logits.reshape(-1, logits.shape[-1])


class TestTimeAdaptation:
    """Test-time adaptation of ``model`` by ``optimizer``.

    Holds the optimizer the way ``LRScheduler`` does; it is not an
    ``Optimizer``. The optimizer's parameters are the ones that move, and each
    must be a parameter of ``model`` (one that is not is refused by name, since
    stepping it would adapt another model). Gradients are taken for those
    parameters only, so no other parameter of the model gets a ``.grad``.

    ``step(*args, **kwargs)`` runs the model, takes :meth:`loss` of its output
    and, when :meth:`should_update` agrees, updates. It does so ``steps`` times
    per call and returns the last prediction, made before the last update: the
    answer an online loop has to emit before it has learnt from the batch.
    ``episodic=True`` resets before every call.

    :attr:`delta` is a :class:`torchnative.delta.Delta` over the moving
    parameters, recorded after every update, so ``tta.delta.norm()`` says how
    far the model has moved. :meth:`reset` restores the parameters and every
    buffer bit for bit and the optimizer's state as it was at construction.

    Subclasses override :meth:`loss`, and :meth:`should_update` when an update
    is conditional. ``requires_backward = False`` declares a rule that needs no
    gradient (see :class:`NormStatistics`).
    """

    #: Whether the rule updates by gradient. See the module docstring.
    requires_backward = True

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if cls.requires_backward and not BACKWARD:
            raise TypeError(_no_backward(cls.__name__))

    def __init__(self, model, optimizer, *, steps=1, episodic=False):
        if not isinstance(model, torch.nn.Module):
            raise TypeError(
                "torch.optim.test_time: model must be an nn.Module, got %r"
                % (type(model).__name__,)
            )
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError("torch.optim.test_time: steps must be an int >= 1, got %r" % (steps,))
        if optimizer is None:
            if self.requires_backward:
                raise TypeError(
                    "torch.optim.test_time: %s updates by gradient and needs an "
                    "optimizer over the parameters it moves" % (type(self).__name__,)
                )
        elif not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                "torch.optim.test_time: %r is not a torch.optim.Optimizer"
                % (type(optimizer).__name__,)
            )
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        self.episodic = episodic

        names = []
        if optimizer is not None:
            by_id = {id(p): n for n, p in model.named_parameters()}
            for gi, group in enumerate(optimizer.param_groups):
                for pi, p in enumerate(group["params"]):
                    name = by_id.get(id(p))
                    if name is None:
                        raise ValueError(
                            "torch.optim.test_time: optimizer parameter %d of "
                            "group %d (shape %s) is not a parameter of this "
                            "model, so stepping it would adapt something else"
                            % (pi, gi, tuple(p.shape))
                        )
                    if name not in names:
                        names.append(name)
            if not names:
                raise ValueError(
                    "torch.optim.test_time: the optimizer holds no parameters, "
                    "so every update would run and change nothing"
                )
        self._names = tuple(names)
        self.delta = Delta.over(model, names) if names else None
        buffers = [n for n, b in model.named_buffers() if b is not None]
        self._buffers = BufferSnapshot.over(model, buffers) if buffers else None
        self._optimizer_state = (
            None if optimizer is None else copy.deepcopy(optimizer.state_dict())
        )
        self.last_loss = None

    # -- what is minimised, and when ----------------------------------------

    def loss(self, outputs):
        """The scalar to minimise, from the model's outputs. ``None`` skips."""
        raise NotImplementedError(
            "torch.optim.test_time: %s does not define loss(outputs)"
            % (type(self).__name__,)
        )

    def should_update(self, outputs, loss):
        """Whether this batch updates. By default, whenever there is a loss."""
        return loss is not None

    # -- which parameters move -------------------------------------------

    @property
    def online_parameter_names(self):
        """The names of the parameters this rule moves, in optimizer order."""
        return self._names

    def online_parameters(self):
        """The parameters this rule moves."""
        params = dict(self.model.named_parameters())
        return [params[n] for n in self._names]

    # -- the step --------------------------------------------------------

    def step(self, *args, **kwargs):
        """Predict, then adapt; returns the model's outputs."""
        if self.episodic:
            self.reset()
        outputs = None
        for _ in range(self.steps):
            outputs = self.model(*args, **kwargs)
            loss = self.loss(outputs)
            self.last_loss = None if loss is None else float(loss.detach())
            if self.should_update(outputs, loss):
                self._update(loss)
            else:
                self._release(outputs, loss)
        return outputs

    def _release(self, outputs, loss):
        """Consume the recording of a forward that will not be differentiated.

        Upstream frees a graph when its outputs die. This stack's eager tape
        outlives them (docs/training/BACKWARD7.md section 10): a skipped batch's
        forward stays recorded until the next backward, and a write into the
        parameters it read (``reset()``, or any in-place update) in between
        makes that backward refuse. So a batch that does not update is walked
        once, its gradients discarded. Only on this stack; upstream needs and
        gets nothing here.
        """
        if not _EAGER_TAPE:
            return
        root = loss if isinstance(loss, torch.Tensor) and loss.requires_grad else None
        if root is None:
            try:
                logits = _logits_of(outputs)
            except TypeError:
                return
            if not logits.requires_grad:
                return
            root = logits.sum()
        leaves = [p for p in self.online_parameters() if p.requires_grad]
        if leaves:
            torch.autograd.grad(root, leaves, allow_unused=True)

    def _update(self, loss):
        params = self.online_parameters()
        frozen = [n for n, p in zip(self._names, params) if not p.requires_grad]
        if frozen:
            raise RuntimeError(
                "torch.optim.test_time: %d of the %d parameters this rule moves "
                "do not require grad, first is %r.\nCheck: call model.online() "
                "(TestTimeModuleMixin) with these among online_parameters(), or "
                "requires_grad_(True) on them." % (len(frozen), len(params), frozen[0])
            )
        if not loss.requires_grad:
            raise RuntimeError(
                "torch.optim.test_time: %s's loss does not require grad, so it "
                "is not a function of any parameter this rule moves (computed "
                "under no_grad, or on a detached tensor)" % (type(self).__name__,)
            )
        grads = torch.autograd.grad(loss, params, allow_unused=True)
        if all(g is None for g in grads):
            raise RuntimeError(
                "torch.optim.test_time: the loss reaches none of the %d "
                "parameters this rule moves, so an update would change nothing"
                % (len(params),)
            )
        for p, g in zip(params, grads):
            p.grad = g
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.delta.record(self.model)

    # -- reset -----------------------------------------------------------

    def reset(self):
        """Back to the state at construction: parameters, buffers, optimizer."""
        if self.delta is not None:
            self.delta.revert(self.model)
            self.delta.zero()
        if self._buffers is not None:
            self._buffers.revert(self.model)
        if self.optimizer is not None:
            self.optimizer.load_state_dict(copy.deepcopy(self._optimizer_state))
        self.last_loss = None
        return self

    def __repr__(self):
        return "%s(moves=%d, steps=%d, episodic=%s)" % (
            type(self).__name__, len(self._names), self.steps, self.episodic)


def _has_running_statistics(module):
    """By the buffers it registered, not by class name: ``BatchNorm*``,
    ``InstanceNorm*(track_running_stats=True)``, ``SyncBatchNorm`` and any
    third-party layer with the same two buffers. ``track_running_stats=False``
    registers them as ``None`` and has nothing to recalibrate."""
    own = dict(module.named_buffers(recurse=False))
    return own.get("running_mean") is not None and own.get("running_var") is not None


class NormStatistics(TestTimeAdaptation):
    """Re-estimate normalisation statistics on the test batches; no gradient.

    Every module that keeps running statistics sees each batch in training mode
    after the prediction is made, so its ``running_mean`` and ``running_var``
    take the batch in with the module's own momentum. The prediction is made
    first, in whatever mode the model is in: the answer an online loop emits is
    normalised by the statistics it had going in. No optimizer, no loss.
    """

    requires_backward = False

    def __init__(self, model, optimizer=None, *, steps=1, episodic=False):
        if optimizer is not None:
            raise TypeError(
                "torch.optim.test_time: NormStatistics moves no parameter, so it "
                "takes no optimizer. A method that also moves affine parameters "
                "composes this with a gradient rule."
            )
        super().__init__(model, None, steps=steps, episodic=episodic)
        self._modules = [m for m in model.modules() if _has_running_statistics(m)]
        if not self._modules:
            raise ValueError(
                "torch.optim.test_time: this model has no module with running "
                "statistics (LayerNorm and RMSNorm keep none), so NormStatistics "
                "would recalibrate nothing"
            )

    def loss(self, outputs):
        return None

    def step(self, *args, **kwargs):
        if self.episodic:
            self.reset()
        with torch.no_grad():
            outputs = self.model(*args, **kwargs)
            for _ in range(self.steps):
                was = [m.training for m in self._modules]
                for m in self._modules:
                    m.train()
                try:
                    self.model(*args, **kwargs)
                finally:
                    for m, mode in zip(self._modules, was):
                        m.train(mode)
        return outputs


if BACKWARD:

    class EntropyMinimization(TestTimeAdaptation):
        """Minimise the mean Shannon entropy of the softmax prediction."""

        def loss(self, outputs):
            logp = _rows(_logits_of(outputs)).log_softmax(-1)
            return -(logp.exp() * logp).sum(-1).mean()

    class PseudoLabel(TestTimeAdaptation):
        """Cross entropy against the model's own arg-max prediction.

        Only samples whose softmax confidence reaches ``threshold`` contribute;
        a batch with none of them does not update.
        """

        def __init__(self, model, optimizer, *, threshold=0.0, steps=1, episodic=False):
            super().__init__(model, optimizer, steps=steps, episodic=episodic)
            self.threshold = float(threshold)

        def loss(self, outputs):
            logits = _rows(_logits_of(outputs))
            conf, label = logits.detach().softmax(-1).max(-1)
            keep = conf >= self.threshold
            count = int(keep.sum())
            if count == 0:
                return None
            # Cross entropy over the kept samples, written as a masked mean of
            # `-log_softmax . one_hot`. The same quantity as
            # `F.cross_entropy(logits[keep], label[keep])`, spelled so that its
            # backward needs no rule this stack's tape lacks: boolean-mask
            # indexing (`aten.index.Tensor`) and `gather` have none, and
            # `nll_loss` with `reduction="none"` has a defective one.
            onehot = F.one_hot(label, logits.shape[-1]).to(logits.dtype)
            per_sample = -(logits.log_softmax(-1) * onehot).sum(-1)
            return (per_sample * keep.to(logits.dtype)).sum() / count
