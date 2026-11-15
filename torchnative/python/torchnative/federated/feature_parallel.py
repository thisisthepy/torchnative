"""``torch.federated.FederatedFeatureParallel``: vertical federated learning.

Issue #293, SPEC S6.8. The name is the owner's (2026-11-08, INTENT §4: "연합학습과
군집학습 부분은 확정하고 문서 개정하고, 리드미 수정하고, 지금 리펙토링은 바로 들어가도록
해."). **Everything below the name is an agent proposal, not confirmed**: the
split-learning shape, the constructor's arguments, the message protocol, the
tags, the concatenation, the alignment digest and ``backward``. The design and
what was weighed is ``docs/distributed/VFL.md``.

Vertical federated learning splits the *features* across participants: every
rank holds some columns of the same samples. Each rank wraps a **bottom
module** over its own columns. One rank, the **label holder** (the "active
party"), also holds the **top module** and the labels. The others are
**passive parties**.

    # every rank: its own bottom module over its own feature columns
    ffp = federated.FederatedFeatureParallel(bottom, label_holder=0,
                                             top=top if rank == 0 else None)
    optimizer = torch.optim.SGD(ffp.parameters(), lr=0.1)
    for ids, x_local, y in loader:          # y exists on the label holder only
        optimizer.zero_grad()
        out = ffp(x_local, sample_ids=ids)
        if ffp.is_label_holder:
            loss_fn(out, y).backward()      # an ordinary backward
        else:
            ffp.backward()                  # waits for this rank's gradient
        optimizer.step()

``ffp.backward(loss)`` with ``loss=None`` on passive parties is the same loop
written once for every rank.

What one step moves, and nothing else:

======================  ==========================  =========================
message                 direction                   shape
======================  ==========================  =========================
header                  passive to label holder     ``(6,)`` int64: step,
                                                    batch, width, dtype code,
                                                    grad mode, sample digest
verdict                 label holder to passive     ``(5,)`` int64: step, ok,
                                                    holder's digest, bad-rank
                                                    mask, reason bits
embedding               passive to label holder     ``(batch, width)``
status                  label holder to passive     ``(2,)`` int64: step, ok
gradient                label holder to passive     ``(batch, width)``
======================  ==========================  =========================

The raw features, the labels and the sample ids never leave their rank: the
features enter only the local bottom module, the labels are never handed to
this class, and the ids cross only as a 56-bit digest.

**The batch alignment is the caller's.** Every rank must feed the same samples
in the same order. That fails silently (the top module trains on rows that do
not belong together and the loss still goes down), so it is checked every
step: each passive party sends a digest of its ``sample_ids``, the label
holder compares them with its own before any embedding is sent, and every
rank raises :class:`SamplesMisaligned` naming the ranks that differ.

Graded *builds* (AGENTS.md §16): nothing here has run. ``tests/distributed/
test_vfl.py`` is written to hold it against the joined single-process model
and against upstream torch with gloo doing the same exchange by hand.
"""

from __future__ import annotations

import hashlib
import itertools
import struct

import torch

from . import _dist, _require_world, agree

__all__ = ["FederatedFeatureParallel", "SamplesMisaligned"]

#: Every message of one instance is tagged ``tag_base + kind``. The base moves
#: by ``_TAG_STRIDE`` per instance, so frames of an instance that refused
#: halfway can never be matched by the next one. It fits in 32 bits.
_TAG_ROOT = 0x76460000
_TAG_STRIDE = 8
TAGS = {"header": 0, "verdict": 1, "embedding": 2, "status": 3, "gradient": 4}

#: The embedding dtypes the header can name. The wire carries more, but the
#: top module concatenates floating embeddings, and these two are what the
#: tests hold against upstream (float64 is where the tolerance comes from).
_DTYPE_CODES = {"torch.float32": 0, "torch.float64": 1}

#: Integer dtypes a ``sample_ids`` tensor may have (compared by name, so the
#: check reads the same on the shim and on upstream).
_ID_DTYPES = ("torch.int8", "torch.int16", "torch.int32", "torch.int64",
              "torch.uint8")

#: The bad-rank mask is one int64, so the world it can name is bounded.
_MAX_WORLD = 62

# Reason bits of a verdict.
_R_SAMPLES = 1
_R_GRAD_MODE = 2
_R_DTYPE = 4
_R_STEP = 8

_INSTANCES = itertools.count()


class SamplesMisaligned(ValueError):
    """The ranks did not feed the same samples, in the same order, this step.

    Raised on every rank of the step, before any embedding is sent. ``step``
    is the step index, ``rank`` this process's rank and ``mismatched`` the
    ranks whose batch differs from the label holder's.
    """

    def __init__(self, message, step=None, rank=None, mismatched=()):
        super().__init__(message)
        self.step = step
        self.rank = rank
        self.mismatched = tuple(mismatched)


def _batch_digest(ids):
    """56 bits of SHA-256 over the ids, in order. Order is the point."""
    h = hashlib.sha256(b"torchnative.federated.FederatedFeatureParallel/ids\x00")
    h.update(struct.pack("<%dq" % len(ids), *ids))
    return int.from_bytes(h.digest()[:7], "big")


def _int64(values):
    return torch.tensor([int(v) for v in values], dtype=torch.int64)


def _reasons_text(bits):
    names = []
    if bits & _R_SAMPLES:
        names.append("the sample ids (batch size or order)")
    if bits & _R_STEP:
        names.append("the step index")
    if bits & _R_GRAD_MODE:
        names.append("whether gradients are enabled")
    if bits & _R_DTYPE:
        names.append("the embedding dtype")
    return ", ".join(names)


class FederatedFeatureParallel(torch.nn.Module):
    """Wrap this rank's bottom module for vertical federated learning.

    ``module`` is this rank's bottom module over its own feature columns. Its
    output is the rank's **embedding**, a CPU ``(batch, width)`` float32 or
    float64 tensor; widths may differ between ranks.

    ``label_holder`` is the rank that holds the labels and ``top``. ``top`` is
    given on that rank and on no other. Its input is the embeddings of every
    rank **concatenated along dim 1 in rank order** (the label holder's own at
    its rank's position), so its input width is the sum of the widths.

    ``forward(*args, sample_ids, **kwargs)`` calls ``module(*args, **kwargs)``.
    ``sample_ids`` is a 1-D integer tensor (or sequence) naming the batch's
    samples, one per row, in the order of the rows. On the label holder it
    returns ``top(concatenated embeddings)``; on a passive party it returns
    this rank's embedding.

    ``backward(loss=None)``: on the label holder ``loss.backward()`` and then
    a check that every passive party's gradient was sent; on a passive party
    (``loss`` must be None) it receives this rank's embedding gradient and
    runs ``embedding.backward(gradient)``. On the label holder a plain
    ``loss.backward()`` also works: the gradients leave as the backward
    reaches them, and the check runs at the next ``forward``.

    Under ``torch.no_grad()`` every rank must be in no-grad mode; no gradient
    is exchanged and no ``backward`` is called.

    Refused by name: a world of one, no ``label_holder``, a ``top`` on the
    wrong rank, ``group=`` (sub-groups are issue #221), ``merge=`` other than
    ``"cat"``, ``secure_aggregation=``, ``differential_privacy=``,
    ``encryption=``, a non-CPU embedding, misaligned batches.
    """

    def __init__(self, module, label_holder=None, top=None, group=None,
                 merge="cat", secure_aggregation=False,
                 differential_privacy=None, encryption=None):
        super().__init__()
        who = "torchnative.federated.FederatedFeatureParallel"
        if secure_aggregation:
            raise NotImplementedError(
                "%s: secure_aggregation= is not implemented. In vertical FL "
                "the label holder sees every passive party's embedding in "
                "the clear; hiding it needs a protocol (homomorphic "
                "encryption of the embeddings or of the gradients, or secure "
                "multi-party computation of the top module's first layer), "
                "not a flag on this wrapper. None of those is built." % who)
        if differential_privacy is not None:
            raise NotImplementedError(
                "%s: differential_privacy= is not implemented. A guarantee "
                "is an (epsilon, delta) at a stated granularity, and for "
                "embeddings and their gradients it needs per-example clipping "
                "of what crosses, noise calibrated to that clip, and an "
                "accountant over the steps. Adding noise without them would "
                "make the model worse and the guarantee absent, and report "
                "both as success." % who)
        if encryption is not None:
            raise NotImplementedError(
                "%s: encryption= is not implemented. The torchnative "
                "transport carries raw tensor bytes (docs/distributed/P2P.md "
                "§2) and binds loopback by default; an encrypted channel "
                "needs a key exchange and an authenticated cipher in the "
                "transport, not here." % who)
        if merge != "cat":
            raise NotImplementedError(
                "%s: merge=%r. Only 'cat' is built: the embeddings are "
                "concatenated along dim 1 in rank order. 'sum' would need "
                "every rank to agree on one width and makes the top module "
                "unable to tell the parties apart; it is not built rather "
                "than offered untested." % (who, merge))
        if not isinstance(module, torch.nn.Module):
            raise TypeError(
                "%s: module=%r is not a torch.nn.Module. Every rank wraps "
                "its own bottom module over its own feature columns."
                % (who, type(module).__name__))
        if group is not None:
            raise NotImplementedError(
                "%s: group= is not supported. Point-to-point inside a "
                "sub-group needs new_group/split_group, which the torchnative "
                "backend does not build yet (issue #221); this uses the "
                "default group's ranks." % who)
        world, rank = _require_world(None, "FederatedFeatureParallel")
        if world > _MAX_WORLD:
            raise NotImplementedError(
                "%s: world_size %d. The verdict names the misaligned ranks in "
                "one int64 mask, so at most %d ranks are built."
                % (who, world, _MAX_WORLD))
        if label_holder is None:
            raise TypeError(
                "%s: no label_holder=. Vertical FL needs one rank that holds "
                "the labels and the top module; without it there is no loss "
                "and nothing to send an embedding to.\n"
                "Check: label_holder=<rank>, the same on every rank." % who)
        if (isinstance(label_holder, bool) or not isinstance(label_holder, int)
                or not 0 <= label_holder < world):
            raise ValueError(
                "%s: label_holder=%r is not a rank of a world of %d"
                % (who, label_holder, world))
        if rank == label_holder and top is None:
            raise TypeError(
                "%s: rank %d is the label holder and was given no top=. The "
                "label holder computes the loss from the top module over "
                "every rank's embedding." % (who, rank))
        if rank != label_holder and top is not None:
            raise ValueError(
                "%s: rank %d is a passive party (the label holder is rank %d) "
                "and was given a top=. Only the label holder holds the top "
                "module; a passive party's top would never see an embedding."
                % (who, rank, label_holder))
        if top is not None and not isinstance(top, torch.nn.Module):
            raise TypeError("%s: top=%r is not a torch.nn.Module"
                            % (who, type(top).__name__))

        instance = next(_INSTANCES)
        # One int64 gather, at construction only: every rank must agree on the
        # world, the label holder and which instance this is (the tags are
        # derived from it). Two ranks that disagree would wait on each other
        # for the group timeout instead of saying why.
        config = hashlib.sha256(
            b"FederatedFeatureParallel\x00%d\x00%d\x00%d\x00cat"
            % (world, label_holder, instance)).digest()
        agree(int.from_bytes(config[:7], "big"), None,
              "the FederatedFeatureParallel configuration (world %d, "
              "label_holder %d, instance %d on rank %d)"
              % (world, label_holder, instance, rank))

        self.module = module
        self.top = top
        self.label_holder = label_holder
        self.world = world
        self.rank = rank
        self.tag_base = _TAG_ROOT + _TAG_STRIDE * instance
        self._step = 0
        self._pending = None     # passive: (step, embedding) awaiting backward
        self._awaiting = {}      # label holder: {peer: step} gradients not yet sent
        self._works = []         # label holder: isend works not yet waited
        self._federated_logging_data = {}

    def __repr__(self):
        return "<FederatedFeatureParallel rank %d of %d, label holder %d>" % (
            self.rank, self.world, self.label_holder)

    @property
    def is_label_holder(self):
        return self.rank == self.label_holder

    def _tag(self, kind):
        return self.tag_base + TAGS[kind]

    # -- the step --------------------------------------------------------

    def forward(self, *args, sample_ids=None, **kwargs):
        who = "torchnative.federated.FederatedFeatureParallel.forward"
        ids = self._ids(sample_ids, who)
        # Before the step index moves: a refusal here leaves every rank's
        # index where it was, so the next step is still aligned.
        if self.is_label_holder:
            self._settle()
        elif self._pending is not None:
            raise RuntimeError(
                "%s: rank %d's backward for step %d has not run. A passive "
                "party must call ffp.backward() after every forward in grad "
                "mode: the label holder has sent that step's gradient and "
                "this rank would train without it."
                % (who, self.rank, self._pending[0]))
        step = self._step
        self._step += 1

        emb = self.module(*args, **kwargs)
        self._check_embedding(emb, len(ids), who)
        grad_mode = bool(torch.is_grad_enabled())
        if grad_mode and not emb.requires_grad:
            raise ValueError(
                "%s: rank %d's embedding does not require grad in grad mode, "
                "so its bottom module has nothing to train. Freeze a party "
                "by running the step under torch.no_grad() on every rank."
                % (who, self.rank))
        digest = _batch_digest(ids)
        header = (step, len(ids), emb.shape[1],
                  _DTYPE_CODES[str(emb.dtype)], int(grad_mode), digest)
        self._federated_logging_data = {
            "rank": self.rank, "world": self.world,
            "label_holder": self.label_holder, "step": step,
            "batch": len(ids), "width": int(emb.shape[1]), "digest": digest,
            "grad_mode": grad_mode,
        }
        if self.is_label_holder:
            return self._forward_holder(emb, header)
        return self._forward_passive(emb, header)

    def _forward_passive(self, emb, header):
        dist = _dist()
        step, grad_mode, digest = header[0], header[4], header[5]
        dist.send(_int64(header), self.label_holder, tag=self._tag("header"))
        verdict = torch.zeros(5, dtype=torch.int64)
        dist.recv(verdict, self.label_holder, tag=self._tag("verdict"))
        v_step, ok, holder_digest, mask, bits = verdict.tolist()
        if not ok:
            self._refuse_from_verdict(step, v_step, digest, holder_digest,
                                      mask, bits)
        dist.send(emb.detach().contiguous(), self.label_holder,
                  tag=self._tag("embedding"))
        if grad_mode:
            self._pending = (step, emb)
        return emb

    def _forward_holder(self, emb, header):
        dist = _dist()
        step, batch, _, code, grad_mode, digest = header
        peers = [r for r in range(self.world) if r != self.rank]
        headers = {}
        for peer in peers:
            buf = torch.zeros(6, dtype=torch.int64)
            dist.recv(buf, peer, tag=self._tag("header"))
            headers[peer] = buf.tolist()
        mask, bits, why = 0, 0, []
        for peer in peers:
            p_step, p_batch, _, p_code, p_grad, p_digest = headers[peer]
            r = 0
            if p_step != step:
                r |= _R_STEP
            if p_batch != batch or p_digest != digest:
                r |= _R_SAMPLES
            if p_grad != grad_mode:
                r |= _R_GRAD_MODE
            if p_code != code:
                r |= _R_DTYPE
            if r:
                mask |= 1 << peer
                bits |= r
                why.append("rank %d (step %d, batch %d, digest %d, grad mode "
                           "%d, dtype code %d) differs in %s"
                           % (peer, p_step, p_batch, p_digest, p_grad, p_code,
                              _reasons_text(r)))
        verdict = _int64((step, 0 if mask else 1, digest, mask, bits))
        for peer in peers:
            dist.send(verdict, peer, tag=self._tag("verdict"))
        if mask:
            msg = ("torchnative.federated.FederatedFeatureParallel: step %d "
                   "refused on the label holder, rank %d (batch %d, digest "
                   "%d). %s. No embedding was sent.\n"
                   "Check: every rank feeds the same samples in the same "
                   "order and passes their ids as sample_ids=."
                   % (step, self.rank, batch, digest, "; ".join(why)))
            bad = [r for r in peers if mask >> r & 1]
            if bits & (_R_SAMPLES | _R_STEP):
                raise SamplesMisaligned(msg, step=step, rank=self.rank,
                                        mismatched=bad)
            raise ValueError(msg)

        parts = []
        for peer in range(self.world):
            if peer == self.rank:
                parts.append(emb)
                continue
            width = headers[peer][2]
            buf = torch.zeros(batch, width, dtype=emb.dtype)
            dist.recv(buf, peer, tag=self._tag("embedding"))
            if grad_mode:
                buf.requires_grad_(True)
                buf.register_hook(self._hook_for(peer, step))
                self._awaiting[peer] = step
            parts.append(buf)
        self._federated_logging_data["widths"] = [int(p.shape[1]) for p in parts]
        return self.top(self._merge(parts))

    def _merge(self, parts):
        """Concatenate the embeddings along dim 1, in rank order."""
        return torch.cat(parts, dim=1)

    def _hook_for(self, peer, step):
        def hook(grad):
            self._return_gradient(peer, step, grad)
            return None
        return hook

    def _return_gradient(self, peer, step, grad):
        if self._awaiting.get(peer) != step:
            raise RuntimeError(
                "torchnative.federated.FederatedFeatureParallel: a second "
                "gradient for rank %d's embedding at step %d. Its gradient "
                "was already sent; a backward run twice over one step "
                "(retain_graph=True) would send a sum the passive party "
                "cannot tell from one." % (peer, step))
        del self._awaiting[peer]
        dist = _dist()
        self._works.append(dist.isend(_int64((step, 1)), peer,
                                      tag=self._tag("status")))
        self._works.append(self._send_gradient(peer, grad))

    def _send_gradient(self, peer, grad):
        """Send ``grad``, the gradient of ``peer``'s embedding, to ``peer``."""
        return _dist().isend(grad.detach().contiguous(), peer,
                             tag=self._tag("gradient"))

    def _settle(self):
        """Label holder: every gradient of the last step sent, every send waited.

        A passive party whose embedding the loss did not reach would wait on
        its gradient for the group timeout. It is told instead (status 0) and
        both sides refuse by name.
        """
        missing = sorted(self._awaiting.items())
        if missing:
            dist = _dist()
            for peer, step in missing:
                self._works.append(dist.isend(_int64((step, 0)), peer,
                                              tag=self._tag("status")))
            self._awaiting = {}
        works, self._works = self._works, []
        for work in works:
            work.wait()
        if missing:
            raise RuntimeError(
                "torchnative.federated.FederatedFeatureParallel: the backward "
                "of step %d did not reach the embedding of rank(s) %s, so no "
                "gradient exists to send them. Either the loss does not "
                "depend on the top module's output, or backward was not "
                "called before the next forward. Those ranks were told and "
                "refuse too." % (missing[0][1], [p for p, _ in missing]))

    def backward(self, loss=None):
        """One step's backward on this rank. See the class docstring."""
        who = "torchnative.federated.FederatedFeatureParallel.backward"
        if self.is_label_holder:
            if loss is None:
                raise TypeError(
                    "%s: rank %d is the label holder and was given no loss. "
                    "Pass the loss computed from forward's output and the "
                    "labels." % (who, self.rank))
            loss.backward()
            self._settle()
            return None
        if loss is not None:
            raise ValueError(
                "%s: rank %d is a passive party and holds no labels, so it "
                "has no loss. Its gradient comes from the label holder; call "
                "ffp.backward() with no argument." % (who, self.rank))
        if self._pending is None:
            raise RuntimeError(
                "%s: rank %d has no step awaiting a gradient. Call forward in "
                "grad mode first; under torch.no_grad() no gradient is "
                "exchanged." % (who, self.rank))
        step, emb = self._pending
        self._pending = None
        dist = _dist()
        status = torch.zeros(2, dtype=torch.int64)
        dist.recv(status, self.label_holder, tag=self._tag("status"))
        s_step, ok = status.tolist()
        if s_step != step:
            raise RuntimeError(
                "%s: rank %d expected the gradient of step %d and the label "
                "holder sent step %d's" % (who, self.rank, step, s_step))
        if not ok:
            raise RuntimeError(
                "%s: the label holder's backward of step %d did not reach "
                "rank %d's embedding, so there is no gradient for it. The "
                "label holder refuses the step too." % (who, step, self.rank))
        grad = torch.zeros(tuple(emb.shape), dtype=emb.dtype)
        dist.recv(grad, self.label_holder, tag=self._tag("gradient"))
        emb.backward(grad)
        return None

    # -- checks ------------------------------------------------------------

    def _ids(self, sample_ids, who):
        if sample_ids is None:
            raise TypeError(
                "%s: sample_ids= is required. Every rank must feed the same "
                "samples in the same order; the ids are how that is checked "
                "each step (only a 56-bit digest of them is sent)." % who)
        if isinstance(sample_ids, torch.Tensor):
            if str(sample_ids.dtype) not in _ID_DTYPES:
                raise TypeError("%s: sample_ids has dtype %s; it must be an "
                                "integer tensor" % (who, sample_ids.dtype))
            if sample_ids.dim() != 1:
                raise ValueError("%s: sample_ids has shape %s; it must be 1-D, "
                                 "one id per row" % (who, tuple(sample_ids.shape)))
            ids = [int(v) for v in sample_ids.tolist()]
        else:
            try:
                ids = list(sample_ids)
            except TypeError:
                raise TypeError("%s: sample_ids=%r is not a sequence of ints"
                                % (who, sample_ids))
            if not all(isinstance(v, int) and not isinstance(v, bool)
                       for v in ids):
                raise TypeError("%s: sample_ids holds non-integers" % who)
        if not ids:
            raise ValueError("%s: an empty batch" % who)
        return ids

    def _check_embedding(self, emb, batch, who):
        if not isinstance(emb, torch.Tensor):
            raise TypeError("%s: rank %d's module returned %s, not a tensor"
                            % (who, self.rank, type(emb).__name__))
        if emb.device.type != "cpu":
            raise NotImplementedError(
                "%s: rank %d's embedding is on %s. Only CPU embeddings are "
                "exchanged: staging a vulkan, mps or npu tensor through host "
                "memory with counted transfers (SPEC S6.10) is not built for "
                "point-to-point. Move the embedding with .cpu() inside the "
                "bottom module." % (who, self.rank, emb.device))
        if emb.dim() != 2:
            raise ValueError("%s: rank %d's embedding has shape %s; it must "
                             "be (batch, width)" % (who, self.rank,
                                                    tuple(emb.shape)))
        if emb.shape[0] != batch:
            raise ValueError(
                "%s: rank %d's embedding has %d rows and sample_ids names %d"
                % (who, self.rank, emb.shape[0], batch))
        if str(emb.dtype) not in _DTYPE_CODES:
            raise NotImplementedError(
                "%s: rank %d's embedding is %s; only float32 and float64 are "
                "built" % (who, self.rank, emb.dtype))

    def _refuse_from_verdict(self, step, v_step, digest, holder_digest, mask,
                             bits):
        bad = [r for r in range(self.world) if mask >> r & 1]
        msg = ("torchnative.federated.FederatedFeatureParallel: step %d "
               "refused by the label holder, rank %d (its step %d, digest "
               "%d); this is rank %d (digest %d). Rank(s) %s differ in %s. "
               "No embedding was sent.\n"
               "Check: every rank feeds the same samples in the same order "
               "and passes their ids as sample_ids=."
               % (step, self.label_holder, v_step, holder_digest, self.rank,
                  digest, bad, _reasons_text(bits)))
        if bits & (_R_SAMPLES | _R_STEP):
            raise SamplesMisaligned(msg, step=step, rank=self.rank,
                                    mismatched=bad)
        raise ValueError(msg)

    def _get_federated_logging_data(self):
        """What the last forward did, as a plain dict (a copy).

        Keys: rank, world, label_holder, step, batch, width, digest,
        grad_mode, and on the label holder widths (every rank's, in rank
        order).
        """
        return dict(self._federated_logging_data)
