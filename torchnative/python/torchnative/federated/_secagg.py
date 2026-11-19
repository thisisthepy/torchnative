"""Fixed-point encoding and the pairwise-mask generator behind ``algorithms.SecAgg``.

Standard library only (``hashlib``, ``hmac``, ``struct``): no cryptographic
dependency was approved (leader, 2026-11-13: "secure aggregation via pairwise
masking (Bonawitz-style) is approved as the algorithm"). The algorithm is the
leader's decision. Everything in this file (the modulus, the fixed-point
scale, the generator, the domain string) is an **agent proposal, not
confirmed** (docs/distributed/SECAGG.md).

No torch here. Every function works on Python ints and floats, so the
arithmetic is exact and does not depend on how a shim kernel treats int64
overflow: the only torch arithmetic on masked values is the hub's int64
``add_`` in the all_reduce, and :func:`modulus_bits` is chosen so that sum
cannot leave int64.

The bound, derived once here and quoted everywhere else
-------------------------------------------------------

The hub folds ``world`` int64 contributions, each a residue in ``[0, 2**k)``.
Their sum is below ``world * 2**k``, which is at most ``2**63 - 1`` exactly
when ``world < 2**(63 - k)``. So ``k = 63 - world.bit_length()`` is the largest
modulus whose plain integer sum cannot overflow the accumulator
(``world < 2**world.bit_length()``).

A rank's plain value is ``q = round(w * d * 2**f)`` (``f`` fraction bits). The
decoded sum is the centred residue of ``sum(q) mod 2**k``, which is ``sum(q)``
itself exactly when ``|sum(q)| <= 2**(k-1) - 1``. That holds for any inputs when
every rank keeps ``|q| <= B = (2**(k-1) - 1) // world``. ``B / 2**f`` is
therefore the per-rank bound on ``|w * d|``, and a value past it is refused by
name rather than wrapped.

At world 3 and ``f = 32``: ``k = 61``, ``B = (2**60 - 1) // 3``, and
``|w * d| <= B / 2**32``, about 8.9e7.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import struct

#: Domain-separation prefix of every mask block. Versioned, so a later change
#: to the layout cannot produce masks that collide with this one's.
DOMAIN = b"torchnative.federated.secagg.v1"

#: Bytes per pairwise seed (256 bits, the HMAC-SHA-256 key size).
SEED_BYTES = 32

#: Default fixed-point fraction bits (agent proposal, not confirmed).
DEFAULT_FRACTION_BITS = 32

#: The accumulator on the hub is a signed int64.
_ACCUMULATOR_BITS = 63

#: Mask words per SHA-256 block: 32 bytes = four 64-bit words.
_WORDS_PER_BLOCK = 4


def modulus_bits(world):
    """``k`` with ``world * (2**k - 1) <= 2**63 - 1``: the largest safe modulus."""
    return _ACCUMULATOR_BITS - int(world).bit_length()


def per_rank_bound(world, k):
    """``B``: the largest ``|q|`` one rank may contribute so the sum decodes uniquely."""
    return ((1 << (k - 1)) - 1) // int(world)


def round_tag(round_index, fraction_bits):
    """One integer the ranks ``agree`` on: the round number and the scale.

    Both enter every mask (the round) or every encoding (the scale), and a pair
    that disagreed about either would produce masks that do not cancel, which
    is a wrong aggregate that nothing downstream notices.
    """
    return (int(round_index) << 8) | int(fraction_bits)


def quantise(values, weight, fraction_bits, bound):
    """``[round(v * weight * 2**f)]`` and a verdict. Returns ``(ints, worst, status)``.

    ``values`` are Python floats (``tolist()`` of a float tensor). The product
    ``v * weight`` is one float64 multiply; scaling by ``2**f`` is exact (a
    power of two), and ``round`` of a float64 is exact. So the only errors are
    the float64 product (relative ``2**-53``) and the rounding to an integer
    (at most ``1/2`` unit, that is ``2**-(f+1)`` in value).

    ``status`` is ``"ok"``, ``"overflow"`` (some ``|q| > bound``) or
    ``"nonfinite"`` (a nan or an inf, which has no fixed-point encoding).
    ``worst`` is the largest ``|v * weight|`` seen, for the refusal message.
    """
    scale = float(1 << fraction_bits)
    out = []
    worst = 0.0
    status = "ok"
    w = float(weight)
    for v in values:
        x = float(v) * w
        if not math.isfinite(x):
            status = "nonfinite"
            worst = float("inf")
            out.append(0)
            continue
        if abs(x) > worst:
            worst = abs(x)
        q = round(x * scale)
        if abs(q) > bound:
            if status == "ok":
                status = "overflow"
            out.append(0)
            continue
        out.append(int(q))
    return out, worst, status


def mask_words(seed, round_index, label, n, k):
    """``n`` mask words in ``[0, 2**k)``: HMAC-SHA-256 in counter mode.

    Block ``c`` is ``HMAC-SHA-256(seed, DOMAIN || 0x00 || round || len(label)
    || label || c)``, with the integers little-endian and fixed width, so the
    input is injective in ``(round, label, c)``. Each 32-byte block is four
    64-bit words and each word is reduced by masking its low ``k`` bits, which
    keeps it uniform because the modulus is a power of two.

    HMAC rather than a bare ``sha256(seed || ...)``: it is the keyed
    pseudo-random function the standard library offers, and it settles the
    length-extension question a bare secret-prefix hash would raise. Both are
    SHA-256 in counter mode over (seed, round, label, chunk), the shape the
    issue names (agent proposal, not confirmed).
    """
    if len(seed) != SEED_BYTES:
        raise ValueError(
            "torchnative.federated.SecAgg: a pairwise seed is %d bytes, not %d"
            % (SEED_BYTES, len(seed)))
    label_b = label if isinstance(label, bytes) else label.encode("utf-8")
    key = hmac.new(bytes(seed), digestmod=hashlib.sha256)
    key.update(DOMAIN + b"\x00" + struct.pack("<Q", int(round_index))
               + struct.pack("<I", len(label_b)) + label_b)
    low = (1 << k) - 1
    out = []
    blocks = (n + _WORDS_PER_BLOCK - 1) // _WORDS_PER_BLOCK
    for chunk in range(blocks):
        h = key.copy()
        h.update(struct.pack("<Q", chunk))
        out.extend(word & low for word in struct.unpack("<4Q", h.digest()))
    del out[n:]
    return out


def net_mask(label, n, seeds, rank, round_index, k):
    """``sum_{j>rank} PRG(s_rank,j) - sum_{i<rank} PRG(s_i,rank)`` mod ``2**k``.

    ``seeds`` maps each peer rank to the seed this rank shares with it. Pair
    ``(i, j)`` with ``i < j`` adds its block at ``i`` and subtracts the same
    block at ``j``, so over the whole world every block appears once with each
    sign and the masks cancel exactly in the integer sum mod ``2**k``.
    """
    modulus = 1 << k
    acc = [0] * n
    for peer in sorted(seeds):
        block = mask_words(seeds[peer], round_index, label, n, k)
        if peer > rank:
            acc = [(a + b) % modulus for a, b in zip(acc, block)]
        elif peer < rank:
            acc = [(a - b) % modulus for a, b in zip(acc, block)]
        else:
            raise ValueError(
                "torchnative.federated.SecAgg: rank %d holds a seed shared "
                "with itself" % (rank,))
    return acc


def add_mod(plain, mask, k):
    """``(q + m) mod 2**k`` element by element. ``plain`` may be negative."""
    modulus = 1 << k
    return [(q + m) % modulus for q, m in zip(plain, mask)]


def centred(residue, k):
    """The signed integer in ``[-2**(k-1), 2**(k-1))`` congruent to ``residue``."""
    modulus = 1 << k
    r = int(residue) % modulus
    return r - modulus if r >= (modulus >> 1) else r
