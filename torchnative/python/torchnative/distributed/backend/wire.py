"""The binary frame the `torchnative` backend puts on every transport.

BACKEND.md §4.2: "Binary frame: fixed header ... then raw little-endian
bytes. **No JSON.**" The layout below is an **agent proposal, not confirmed**
(BACKEND.md §11 lists "the binary frame layout" among the open proposals); the
decision it serves is D8, Python first, Rust only where measured slow.

Layout, little-endian, 24 fixed bytes then the shape then the payload::

    offset  size  field
    0       4     magic            b"TNP2"
    4       1     version          1
    5       1     kind             1 tensor, 2 object bytes, 3 control
    6       1     dtype code       0 for a frame that is not a tensor
    7       1     ndim             0..255
    8       8     tag              signed 64-bit
    16      8     payload length   unsigned 64-bit, in bytes
    24      8*n   shape            n = ndim unsigned 64-bit extents
    ...           payload          raw little-endian element bytes, row-major

A frame is self-describing so that the receiver can check, before it writes a
byte into anybody's tensor, that the dtype and element count are the ones it
was asked to receive. A mismatch refuses naming both sides (BACKEND.md §8,
"Cross-device mixed dtype in one collective: refused with both dtypes named").

**Getting bytes out of a tensor.** Two paths, and which one ran is reported by
`export_path` so a test can say which it measured:

* ``fast``: the shim's own numpy-view hook, ``TensorBase._shim_numpy_parts``,
  hands over the address of the CPU element buffer (tensor.rs documents the
  lifetime argument), and ``ctypes.string_at`` copies ``numel * itemsize``
  bytes from it. Exact for every dtype the hook knows, one ``memcpy``, no
  per-element Python. ``bfloat16`` is not one of them (the hook refuses it), so
  it goes out through ``float32``, which is exact because every bfloat16 value
  is a float32 whose low 16 bits are zero, and the high half of each
  little-endian float32 word is taken by slicing (C speed, no Python loop).
* ``slow``: ``struct.pack`` of ``tolist()``. Exact for every finite value and
  for infinities, but a NaN's payload bits are whatever ``struct`` writes for
  the Python float, so NaN *payloads* are not preserved on this path. It runs
  only when the fast path is unavailable (no hook, an unusual layout). D8: it
  is the slow exact path, and the place Rust would go if a measurement says so.

Into a tensor: ``torch.frombuffer`` (lib.rs ``_frombuffer``), which copies the
bytes into a new candle buffer for every dtype the shim can store.

Nothing here has run (BACKEND.md §9 step 2 is graded *builds*).
"""

from __future__ import annotations

import struct
import sys

MAGIC = b"TNP2"
VERSION = 1

KIND_TENSOR = 1
KIND_BYTES = 2
KIND_CONTROL = 3
_KINDS = (KIND_TENSOR, KIND_BYTES, KIND_CONTROL)

#: ``<4sBBBBqQ``: magic, version, kind, dtype code, ndim, tag, payload length.
_HEADER = struct.Struct("<4sBBBBqQ")
HEADER_SIZE = _HEADER.size  # 24

_TAG_MIN = -(1 << 63)
_TAG_MAX = (1 << 63) - 1

#: dtype name -> (wire code, itemsize, struct format for the slow path).
#: ``None`` as the format means "no struct letter" (bfloat16, handled apart).
#: Codes are an agent proposal; they only have to agree between two ends of
#: one frame, and VERSION changes if they ever change.
DTYPES = {
    "float32": (1, 4, "f"),
    "float64": (2, 8, "d"),
    "float16": (3, 2, "e"),
    "bfloat16": (4, 2, None),
    "int8": (5, 1, "b"),
    "int16": (6, 2, "h"),
    "int32": (7, 4, "i"),
    "int64": (8, 8, "q"),
    "uint8": (9, 1, "B"),
    "bool": (10, 1, "?"),
}
_BY_CODE = {code: name for name, (code, _, _) in DTYPES.items()}


class WireError(RuntimeError):
    """A frame that cannot be decoded. The message always names the peer rank."""


class PeerGone(RuntimeError):
    """The peer closed its end at a frame boundary. Names the peer rank.

    The text carries "connection closed", the phrase
    ``ProcessGroupLocal._is_gone`` already reads as a vanished peer, so the two
    halves of the backend describe a lost rank the same way.
    """


def _who(peer):
    return "rank %s" % (peer,) if peer is not None else "an unnamed peer"


def _refuse(what, why):
    raise NotImplementedError(
        "torchnative.distributed.backend.wire.%s: %s" % (what, why))


class Frame:
    """One decoded frame. ``payload`` is ``bytes``; ``shape`` a tuple of ints."""

    __slots__ = ("kind", "tag", "dtype", "shape", "payload")

    def __init__(self, kind, tag, dtype, shape, payload):
        self.kind = kind
        self.tag = tag
        self.dtype = dtype
        self.shape = tuple(shape)
        self.payload = payload

    def numel(self):
        n = 1
        for extent in self.shape:
            n *= extent
        return n

    def __repr__(self):
        return "Frame(kind=%d, tag=%d, dtype=%r, shape=%r, %d bytes)" % (
            self.kind, self.tag, self.dtype, self.shape, len(self.payload))


# -- tensor -> bytes ---------------------------------------------------------

def dtype_name(tensor):
    """``"float32"`` for a ``torch.float32`` tensor."""
    return str(tensor.dtype).rsplit(".", 1)[-1]


def _check_dtype(name, what):
    if name not in DTYPES:
        _refuse(what,
                "the wire has no code for torch.%s. Covered: %s. Complex "
                "tensors reach the backend as real pairs through upstream's "
                "view_as_real; any other dtype is refused here rather than "
                "reinterpreted as one that has a code"
                % (name, ", ".join("torch." + n for n in DTYPES)))


def _host_contiguous(tensor):
    t = tensor.detach()
    if t.device.type != "cpu":
        # BACKEND.md §4.3: a device tensor is staged through host memory.
        # Step 7 counts this download; here it is only performed.
        t = t.cpu()
    if not t.is_contiguous():
        t = t.contiguous()
    return t


def _c_strides(shape, itemsize):
    strides = []
    acc = itemsize
    for extent in reversed(shape):
        strides.append(acc)
        acc *= max(extent, 1)
    return list(reversed(strides))


def _fast_bytes(t, itemsize):
    """``numel * itemsize`` bytes straight out of the CPU buffer, or ``None``."""
    import ctypes
    if sys.byteorder != "little":
        return None
    parts = getattr(t, "_shim_numpy_parts", None)
    if parts is None:
        return None
    try:
        addr, _typestr, shape, strides = parts()
    except Exception:  # noqa: BLE001 -- the hook refuses dtypes it lacks
        return None
    shape = list(shape)
    strides = list(strides)
    expected = _c_strides(shape, itemsize)
    for extent, got, want in zip(shape, strides, expected):
        if extent > 1 and got != want:
            return None
    n = itemsize
    for extent in shape:
        n *= extent
    if n == 0:
        return b""
    # `t` is alive for the whole call, so the address is valid (tensor.rs
    # `_shim_numpy_parts`: the Vec never moves while a tensor holds its Arc).
    return ctypes.string_at(addr, n)


def _slow_bytes(t, name, fmt):
    flat = t.reshape(-1).tolist()
    if name == "bfloat16":
        raw = struct.pack("<%df" % len(flat), *flat)
        out = bytearray(2 * len(flat))
        out[0::2] = raw[2::4]
        out[1::2] = raw[3::4]
        return bytes(out)
    return struct.pack("<%d%s" % (len(flat), fmt), *flat)


def _bf16_fast(t):
    """bfloat16 through float32: exact, and the slicing runs in C."""
    wide = t.float()
    raw = _fast_bytes(_host_contiguous(wide), 4)
    if raw is None:
        return None
    out = bytearray(len(raw) // 2)
    out[0::2] = raw[2::4]
    out[1::2] = raw[3::4]
    return bytes(out)


def _export(tensor):
    name = dtype_name(tensor)
    _check_dtype(name, "encode_tensor")
    _, itemsize, fmt = DTYPES[name]
    t = _host_contiguous(tensor)
    shape = tuple(int(x) for x in t.shape)
    if name == "bfloat16":
        raw = _bf16_fast(t)
    else:
        raw = _fast_bytes(t, itemsize)
    if raw is not None:
        return name, shape, raw, "fast"
    return name, shape, _slow_bytes(t, name, fmt), "slow"


def tensor_bytes(tensor):
    """``(dtype name, shape, little-endian bytes)`` of ``tensor``, exactly."""
    name, shape, raw, _path = _export(tensor)
    return name, shape, raw


def export_path(tensor):
    """``"fast"`` or ``"slow"``: which bytes path ``tensor`` takes. For tests."""
    return _export(tensor)[3]


def bytes_to_tensor(payload, name, shape):
    """A new CPU tensor of dtype ``name`` and ``shape`` holding ``payload``."""
    import torch
    _check_dtype(name, "bytes_to_tensor")
    dtype = getattr(torch, name)
    n = 1
    for extent in shape:
        n *= extent
    if n == 0:
        return torch.empty(tuple(shape), dtype=dtype)
    flat = torch.frombuffer(bytes(payload), dtype=dtype)
    return flat.reshape(tuple(shape))


# -- frames ------------------------------------------------------------------

def _check_tag(tag):
    if isinstance(tag, bool) or not isinstance(tag, int):
        raise TypeError(
            "torchnative.distributed.backend.wire: a tag is an int, not %s"
            % (type(tag).__name__,))
    if not _TAG_MIN <= tag <= _TAG_MAX:
        raise ValueError(
            "torchnative.distributed.backend.wire: tag %d does not fit the "
            "frame's signed 64-bit field" % (tag,))


def encode(kind, tag, payload, dtype=None, shape=()):
    """One frame as ``bytes``."""
    if kind not in _KINDS:
        raise ValueError("torchnative.distributed.backend.wire: kind %r" % (kind,))
    _check_tag(tag)
    shape = tuple(int(x) for x in shape)
    if len(shape) > 255:
        _refuse("encode", "a tensor of %d dimensions; the frame's ndim field "
                          "holds at most 255" % len(shape))
    code = 0
    if dtype is not None:
        _check_dtype(dtype, "encode")
        code = DTYPES[dtype][0]
    payload = bytes(payload)
    head = _HEADER.pack(MAGIC, VERSION, kind, code, len(shape), tag,
                        len(payload))
    dims = struct.pack("<%dQ" % len(shape), *shape) if shape else b""
    return head + dims + payload


def encode_tensor(tensor, tag):
    """A tensor frame: the tensor's own dtype, shape and bytes."""
    name, shape, raw, _path = _export(tensor)
    return encode(KIND_TENSOR, tag, raw, dtype=name, shape=shape)


def encode_bytes(data, tag):
    return encode(KIND_BYTES, tag, data)


def encode_control(tag, data=b""):
    return encode(KIND_CONTROL, tag, data)


def _recv_exact(recv, n, peer, what, at_boundary):
    """Exactly ``n`` bytes from ``recv``, which may return fewer per call.

    ``recv(k)`` is a socket's ``recv``: it returns between 1 and ``k`` bytes,
    or ``b""`` at end of stream. A frame split across many calls decodes the
    same as one that arrives whole; a stream that ends inside a frame is
    truncation and raises naming the peer.
    """
    chunks = []
    got = 0
    while got < n:
        piece = recv(n - got)
        if not piece:
            if got == 0 and at_boundary:
                raise PeerGone(
                    "torchnative.distributed.backend: %s is gone: connection "
                    "closed at a frame boundary" % _who(peer))
            raise WireError(
                "torchnative.distributed.backend.wire: frame from %s "
                "truncated: the stream ended after %d of the %d bytes of its "
                "%s" % (_who(peer), got, n, what))
        chunks.append(bytes(piece))
        got += len(piece)
    return b"".join(chunks)


def read_frame(recv, peer=None):
    """Read one frame through ``recv`` (socket-``recv``-shaped). Returns a ``Frame``."""
    head = _recv_exact(recv, HEADER_SIZE, peer, "header", at_boundary=True)
    magic, version, kind, code, ndim, tag, length = _HEADER.unpack(head)
    if magic != MAGIC:
        raise WireError(
            "torchnative.distributed.backend.wire: frame from %s has bad "
            "magic %r (expected %r); the stream is not this backend's or is "
            "out of step" % (_who(peer), magic, MAGIC))
    if version != VERSION:
        raise WireError(
            "torchnative.distributed.backend.wire: frame from %s has version "
            "%d; this end speaks version %d" % (_who(peer), version, VERSION))
    if kind not in _KINDS:
        raise WireError(
            "torchnative.distributed.backend.wire: frame from %s has unknown "
            "kind %d" % (_who(peer), kind))
    dtype = None
    if code:
        dtype = _BY_CODE.get(code)
        if dtype is None:
            raise WireError(
                "torchnative.distributed.backend.wire: frame from %s has "
                "unknown dtype code %d" % (_who(peer), code))
    shape = ()
    if ndim:
        dims = _recv_exact(recv, 8 * ndim, peer, "shape", at_boundary=False)
        shape = struct.unpack("<%dQ" % ndim, dims)
    if kind == KIND_TENSOR:
        if dtype is None:
            raise WireError(
                "torchnative.distributed.backend.wire: tensor frame from %s "
                "carries no dtype" % (_who(peer),))
        n = 1
        for extent in shape:
            n *= extent
        if n * DTYPES[dtype][1] != length:
            raise WireError(
                "torchnative.distributed.backend.wire: tensor frame from %s "
                "says %d payload bytes for %d elements of torch.%s"
                % (_who(peer), length, n, dtype))
    payload = (_recv_exact(recv, length, peer, "payload", at_boundary=False)
               if length else b"")
    return Frame(kind, tag, dtype, shape, payload)


def decode(data, peer=None):
    """Decode one whole frame held in ``data``; trailing bytes are an error."""
    view = memoryview(bytes(data))
    pos = [0]

    def recv(n):
        chunk = view[pos[0]:pos[0] + n]
        pos[0] += len(chunk)
        return chunk.tobytes()

    frame = read_frame(recv, peer)
    if pos[0] != len(view):
        raise WireError(
            "torchnative.distributed.backend.wire: %d stray bytes after a "
            "frame from %s" % (len(view) - pos[0], _who(peer)))
    return frame


def frame_to_tensor(frame):
    """The tensor a tensor frame carries, as a new CPU tensor."""
    if frame.kind != KIND_TENSOR:
        raise WireError(
            "torchnative.distributed.backend.wire: expected a tensor frame, "
            "got kind %d" % (frame.kind,))
    return bytes_to_tensor(frame.payload, frame.dtype, frame.shape)
