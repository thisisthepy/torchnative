"""The GGUF container, read in plain Python.

**What this file is.** GGUF is a header, a table of typed key/value pairs, a
table of tensor descriptors, and then the tensor bytes at aligned offsets. The
container is small enough to read with `struct` and `mmap` and nothing else, so
this module imports neither `torch` nor `numpy` nor the `gguf` package:
`import torchnative.gguf` has to stay cheap for the same reason
`import torchnative` does (`torchnative/__init__.py`), and the container is not
where the hard part is.

**Where the hard part is, and why it is not here.** The tensor bytes are GGML
blocks -- Q8_0, Q4_K, Q6_K and the rest -- and those are not decoded in this
file at all. They are handed, byte for byte, to candle's `QTensor` through
`torch._C._quantized_from_blob` (`rust/torch_c/src/quant.rs`), which is the
same reinterpret-copy candle's own GGUF reader performs
(`vendor/candle-core/src/quantized/ggml_file.rs`, `qtensor_from_ggml` ->
`from_raw_data`, against `quant.rs` -> `QStorage::from_data` ->
`GgmlDType::from_data`; both are `as_t_slice::<BlockT>(bytes).to_vec()`). So the
blocks are never re-encoded on the way in, which is what makes "byte-exact"
something this reader can claim rather than approximate. docs/graph/GGUF.md §2
is the evidence; `rust/torch_c/pytests/test_gguf.py` is the measurement.

**What refuses, and how.** Every GGML type id is in `GGML_TYPES` with one of
three kinds: `dense` (read as a tensor of that dtype), `block` (a format candle
holds and has a matmul for), or `refused` (with the reason). An id that is not
in the table at all refuses naming the number. Nothing is skipped quietly and
no type is approximated by a neighbour -- AGENTS.md §18.

**Byte order.** Only little-endian GGUF is read. A big-endian file (llama.cpp
can write one for s390x) is recognised by its version field reading as a
byte-swapped small integer and refused by name, rather than parsed into
garbage.
"""

import mmap
import os
import struct

__all__ = [
    "GGUFFile",
    "GGUFTensorInfo",
    "GGML_TYPES",
    "GGUFFormatError",
    "UnsupportedGGUF",
    "type_name",
]

GGUF_MAGIC = 0x46554747  # b"GGUF", read little-endian
DEFAULT_ALIGNMENT = 32  # GGUF spec: `general.alignment` defaults to 32
GGML_MAX_DIMS = 4  # ggml.h GGML_MAX_DIMS


class GGUFFormatError(ValueError):
    """The file is not a well-formed GGUF file (bad magic, truncated, inconsistent)."""


class UnsupportedGGUF(NotImplementedError):
    """A well-formed GGUF construct this reader refuses, by name."""


# -- the GGML type table ------------------------------------------------------
#
# id -> (name, block_size, type_size, kind, format_or_reason)
#
#   kind "dense"   : format_or_reason is the torch dtype name to read it as.
#   kind "block"   : format_or_reason is the `torch._C._quantized_formats()`
#                    spelling (quant.rs `format_name`) the blocks go to.
#   kind "refused" : format_or_reason is why.
#
# Block and type sizes are ggml's (`ggml.c` `type_traits`). They are asserted
# against candle's `type_size()`/`block_size()` and against the `gguf` package's
# `GGML_QUANT_SIZES` in test_gguf.py, so a disagreement on either side is a red
# test and not a silent misread. **One such disagreement is already known**:
# `gguf` 0.17.1 lists Q8_1 as 40 bytes (`4 + 4 + 32`, from when `d` and `s`
# were `float`), while ggml and candle store it as 36 (`ggml_half2 ds` + 32).
# Q8_1 is refused below regardless, so the reader never depends on either.

_ACTIVATION_SIDE = (
    "is an activation-side block format: llama.cpp produces it only as the "
    "right-hand operand inside vec_dot, never as a stored weight, and "
    "torch._C._quantize refuses it as a weight for the same reason (quant.rs, "
    "docs/graph/QUANT2.md §6). A file carrying one is not a checkpoint this "
    "reader can map onto a matmul."
)
_NO_CANDLE_TYPE = (
    "has no candle-core GgmlDType, so there is no storage to put its blocks in "
    "and no kernel to run them; dequantising it here would mean transcribing "
    "its codebook into this reader, which is a format implementation, not a "
    "container read. Re-quantise the model to a k-quant (Q4_K_M, Q5_K_M, Q6_K) "
    "or Q8_0."
)
_NOT_WIRED = (
    "is a plain numeric tensor type that no LLM checkpoint this reader has met "
    "uses for weights, and it is not wired: refused rather than guessed at."
)
_REMOVED = (
    "is a repacked layout ggml removed from the file format (it now repacks at "
    "load time instead); a file carrying it predates that and is not read."
)

GGML_TYPES = {
    0: ("F32", 1, 4, "dense", "float32"),
    1: ("F16", 1, 2, "dense", "float16"),
    2: ("Q4_0", 32, 18, "block", "q4_0"),
    3: ("Q4_1", 32, 20, "block", "q4_1"),
    6: ("Q5_0", 32, 22, "block", "q5_0"),
    7: ("Q5_1", 32, 24, "block", "q5_1"),
    8: ("Q8_0", 32, 34, "block", "q8_0"),
    9: ("Q8_1", 32, 36, "refused", _ACTIVATION_SIDE),
    10: ("Q2_K", 256, 84, "block", "q2_k"),
    11: ("Q3_K", 256, 110, "block", "q3_k"),
    12: ("Q4_K", 256, 144, "block", "q4_k"),
    13: ("Q5_K", 256, 176, "block", "q5_k"),
    14: ("Q6_K", 256, 210, "block", "q6_k"),
    15: ("Q8_K", 256, 292, "refused", _ACTIVATION_SIDE),
    16: ("IQ2_XXS", 256, 66, "refused", _NO_CANDLE_TYPE),
    17: ("IQ2_XS", 256, 74, "refused", _NO_CANDLE_TYPE),
    18: ("IQ3_XXS", 256, 98, "refused", _NO_CANDLE_TYPE),
    19: ("IQ1_S", 256, 50, "refused", _NO_CANDLE_TYPE),
    20: ("IQ4_NL", 32, 18, "refused", _NO_CANDLE_TYPE),
    21: ("IQ3_S", 256, 110, "refused", _NO_CANDLE_TYPE),
    22: ("IQ2_S", 256, 82, "refused", _NO_CANDLE_TYPE),
    23: ("IQ4_XS", 256, 136, "refused", _NO_CANDLE_TYPE),
    24: ("I8", 1, 1, "refused", _NOT_WIRED),
    25: ("I16", 1, 2, "refused", _NOT_WIRED),
    26: ("I32", 1, 4, "refused", _NOT_WIRED),
    27: ("I64", 1, 8, "refused", _NOT_WIRED),
    28: ("F64", 1, 8, "refused", _NOT_WIRED),
    29: ("IQ1_M", 256, 56, "refused", _NO_CANDLE_TYPE),
    30: ("BF16", 1, 2, "dense", "bfloat16"),
    31: ("Q4_0_4_4", None, None, "refused", _REMOVED),
    32: ("Q4_0_4_8", None, None, "refused", _REMOVED),
    33: ("Q4_0_8_8", None, None, "refused", _REMOVED),
    34: ("TQ1_0", 256, 54, "refused", _NO_CANDLE_TYPE),
    35: ("TQ2_0", 256, 66, "refused", _NO_CANDLE_TYPE),
    36: ("IQ4_NL_4_4", None, None, "refused", _REMOVED),
    37: ("IQ4_NL_4_8", None, None, "refused", _REMOVED),
    38: ("IQ4_NL_8_8", None, None, "refused", _REMOVED),
    39: ("MXFP4", 32, 17, "refused", _NO_CANDLE_TYPE),
}


def type_name(ggml_type):
    entry = GGML_TYPES.get(ggml_type)
    return entry[0] if entry else f"<unknown GGML type id {ggml_type}>"


def refusal(ggml_type, tensor_name=None):
    """The `UnsupportedGGUF` for a type this reader will not load, or None."""
    where = f" (tensor {tensor_name!r})" if tensor_name else ""
    entry = GGML_TYPES.get(ggml_type)
    if entry is None:
        return UnsupportedGGUF(
            f"GGML type id {ggml_type}{where} is not one this reader knows. It is "
            "refused rather than guessed: a newer llama.cpp may have added it, and "
            "reading its bytes as anything else would load a wrong model quietly."
        )
    name, _, _, kind, why = entry
    if kind == "refused":
        return UnsupportedGGUF(f"GGML type {name}{where} {why}")
    return None


# -- metadata value types (GGUF spec, gguf_type) -------------------------------

_SCALARS = {
    0: ("<B", 1),   # UINT8
    1: ("<b", 1),   # INT8
    2: ("<H", 2),   # UINT16
    3: ("<h", 2),   # INT16
    4: ("<I", 4),   # UINT32
    5: ("<i", 4),   # INT32
    6: ("<f", 4),   # FLOAT32
    7: ("<?", 1),   # BOOL
    10: ("<Q", 8),  # UINT64
    11: ("<q", 8),  # INT64
    12: ("<d", 8),  # FLOAT64
}
VALUE_TYPE_NAMES = {
    0: "UINT8", 1: "INT8", 2: "UINT16", 3: "INT16", 4: "UINT32", 5: "INT32",
    6: "FLOAT32", 7: "BOOL", 8: "STRING", 9: "ARRAY", 10: "UINT64", 11: "INT64",
    12: "FLOAT64",
}
_STRING, _ARRAY = 8, 9


class GGUFTensorInfo:
    """One tensor descriptor. Nothing is read until asked for.

    `shape` is in **torch order** -- the reverse of GGUF's `ne`, whose first
    entry is the fastest-varying dimension. A linear weight stored as
    `ne = [in_features, out_features]` is therefore `(out_features,
    in_features)` here, which is `nn.Linear.weight`'s layout and the layout
    candle's `QMatMul` reads (quant.rs `_quantized_linear`). candle's own
    reader makes the same reversal (`gguf_file.rs`, `dimensions.reverse()`).
    """

    __slots__ = ("name", "ggml_type", "ne", "shape", "offset", "nbytes", "n_elements")

    def __init__(self, name, ggml_type, ne, offset, nbytes, n_elements):
        self.name = name
        self.ggml_type = ggml_type
        self.ne = tuple(ne)
        self.shape = tuple(reversed(ne))
        self.offset = offset  # absolute, in the file
        self.nbytes = nbytes
        self.n_elements = n_elements

    @property
    def type_name(self):
        return type_name(self.ggml_type)

    @property
    def kind(self):
        entry = GGML_TYPES.get(self.ggml_type)
        return entry[3] if entry else "refused"

    @property
    def format(self):
        """The `_C._quantized_formats()` spelling for a block type, else None."""
        entry = GGML_TYPES.get(self.ggml_type)
        return entry[4] if entry and entry[3] == "block" else None

    def __repr__(self):
        return (
            f"GGUFTensorInfo({self.name!r}, {self.type_name}, shape={self.shape}, "
            f"offset={self.offset}, nbytes={self.nbytes})"
        )


class _Cursor:
    def __init__(self, buf, size):
        self.buf = buf
        self.size = size
        self.pos = 0

    def take(self, n, what):
        if n < 0 or self.pos + n > self.size:
            raise GGUFFormatError(
                f"GGUF file is truncated: reading {what} needs {n} bytes at offset "
                f"{self.pos}, and the file is {self.size} bytes"
            )
        start = self.pos
        self.pos += n
        return start

    def scalar(self, fmt, size, what):
        start = self.take(size, what)
        return struct.unpack_from(fmt, self.buf, start)[0]

    def u32(self, what):
        return self.scalar("<I", 4, what)

    def u64(self, what):
        return self.scalar("<Q", 8, what)

    def string(self, what):
        n = self.u64(f"the length of {what}")
        start = self.take(n, what)
        raw = bytes(self.buf[start : start + n])
        # surrogateescape keeps every byte recoverable: a tokenizer entry that
        # is not valid UTF-8 round-trips through `.encode("utf-8",
        # "surrogateescape")` instead of being replaced or refused.
        return raw.decode("utf-8", "surrogateescape")

    def value(self, vtype, what, depth=0):
        if vtype in _SCALARS:
            fmt, size = _SCALARS[vtype]
            return self.scalar(fmt, size, what)
        if vtype == _STRING:
            return self.string(what)
        if vtype == _ARRAY:
            if depth > 0:
                # The spec allows arrays of arrays; no checkpoint uses them and
                # the reader does not pretend to have tested it.
                raise UnsupportedGGUF(f"{what}: nested GGUF arrays are not read")
            etype = self.u32(f"the element type of {what}")
            count = self.u64(f"the element count of {what}")
            # A corrupt count would otherwise be an allocation of whatever
            # number it happens to hold. Every element is at least one byte.
            if count > self.size - self.pos:
                raise GGUFFormatError(
                    f"{what}: array claims {count} elements with {self.size - self.pos} "
                    "bytes left in the file"
                )
            if etype in _SCALARS:
                fmt, size = _SCALARS[etype]
                start = self.take(size * count, what)
                items = list(struct.unpack_from(f"<{count}{fmt[1]}", self.buf, start))
            elif etype == _STRING:
                items = [self.string(f"{what}[{i}]") for i in range(count)]
            else:
                items = [self.value(etype, f"{what}[{i}]", depth + 1) for i in range(count)]
            return ArrayValue(etype, items)
        raise GGUFFormatError(f"{what}: unknown GGUF value type {vtype}")


class ArrayValue(list):
    """A GGUF array: a `list` that also remembers its element type."""

    def __init__(self, element_type, items):
        super().__init__(items)
        self.element_type = element_type


class GGUFFile:
    """A GGUF file, opened read-only and memory-mapped.

        with GGUFFile(path) as f:
            f.metadata["general.architecture"]       # 'qwen3'
            info = f.tensors["blk.0.attn_q.weight"]  # GGUFTensorInfo
            raw = f.raw(info.name)                   # the GGML block bytes

    Turning bytes into tensors needs `torch` and lives in
    `torchnative.gguf.load_tensor`, not here.
    """

    def __init__(self, path):
        self.path = os.fspath(path)
        self._fh = open(self.path, "rb")
        try:
            size = os.fstat(self._fh.fileno()).st_size
            if size == 0:
                raise GGUFFormatError(f"{self.path}: empty file, not GGUF")
            self._mm = mmap.mmap(self._fh.fileno(), 0, access=mmap.ACCESS_READ)
            self.size = size
            self._parse()
        except BaseException:
            self.close()
            raise

    # -- lifecycle -----------------------------------------------------------

    def close(self):
        mm = getattr(self, "_mm", None)
        if mm is not None:
            mm.close()
            self._mm = None
        fh = getattr(self, "_fh", None)
        if fh is not None:
            fh.close()
            self._fh = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- parsing -------------------------------------------------------------

    def _parse(self):
        c = _Cursor(self._mm, self.size)
        magic = c.u32("the magic")
        if magic != GGUF_MAGIC:
            raise GGUFFormatError(
                f"{self.path}: not a GGUF file (magic {magic:#010x}, expected "
                f"{GGUF_MAGIC:#010x} = b'GGUF')"
            )
        version = c.u32("the version")
        if version not in (2, 3):
            swapped = struct.unpack("<I", struct.pack(">I", version))[0]
            if swapped in (1, 2, 3):
                raise UnsupportedGGUF(
                    f"{self.path}: a big-endian GGUF file (version reads {version}, "
                    f"which is {swapped} byte-swapped). Only little-endian GGUF is read."
                )
            if version == 1:
                raise UnsupportedGGUF(
                    f"{self.path}: GGUF version 1 (32-bit counts, superseded in 2023) "
                    "is not read. Re-convert with a current llama.cpp."
                )
            raise UnsupportedGGUF(f"{self.path}: GGUF version {version} is not one this reader knows (2, 3)")
        self.version = version
        n_tensors = c.u64("the tensor count")
        n_kv = c.u64("the metadata count")
        # Each key/value pair and each tensor descriptor is at least 8 bytes of
        # length prefix; a count that cannot fit is corruption, refused here
        # rather than discovered as a loop that never ends.
        if n_kv * 8 > self.size or n_tensors * 8 > self.size:
            raise GGUFFormatError(
                f"{self.path}: header claims {n_kv} metadata entries and {n_tensors} "
                f"tensors, which cannot fit in {self.size} bytes"
            )

        self.metadata = {}
        self.metadata_types = {}
        for i in range(n_kv):
            key = c.string(f"metadata key #{i}")
            vtype = c.u32(f"the value type of {key!r}")
            if key in self.metadata:
                raise GGUFFormatError(f"{self.path}: metadata key {key!r} appears twice")
            self.metadata[key] = c.value(vtype, f"metadata {key!r}")
            self.metadata_types[key] = vtype

        alignment = self.metadata.get("general.alignment", DEFAULT_ALIGNMENT)
        if (
            not isinstance(alignment, int)
            or isinstance(alignment, bool)
            or alignment <= 0
            or alignment & (alignment - 1)
        ):
            raise GGUFFormatError(
                f"{self.path}: general.alignment = {alignment!r} is not a positive power of two"
            )
        self.alignment = alignment

        infos = []
        for i in range(n_tensors):
            name = c.string(f"tensor name #{i}")
            n_dims = c.u32(f"the dimension count of {name!r}")
            if n_dims == 0 or n_dims > GGML_MAX_DIMS:
                raise GGUFFormatError(
                    f"{self.path}: tensor {name!r} has {n_dims} dimensions; ggml allows 1..{GGML_MAX_DIMS}"
                )
            ne = [c.u64(f"dimension {d} of {name!r}") for d in range(n_dims)]
            ggml_type = c.u32(f"the type of {name!r}")
            rel = c.u64(f"the data offset of {name!r}")
            infos.append((name, ne, ggml_type, rel))

        # Tensor data starts at the next multiple of `alignment` after the
        # descriptors (GGUF spec; candle's gguf_file.rs computes the same).
        self.data_offset = -(-c.pos // alignment) * alignment

        self.tensors = {}
        for name, ne, ggml_type, rel in infos:
            if name in self.tensors:
                raise GGUFFormatError(f"{self.path}: tensor {name!r} appears twice")
            if rel % alignment:
                raise GGUFFormatError(
                    f"{self.path}: tensor {name!r} starts at data offset {rel}, which is "
                    f"not a multiple of the alignment {alignment}"
                )
            n_elements = 1
            for d in ne:
                n_elements *= d
            entry = GGML_TYPES.get(ggml_type)
            nbytes = None
            if entry is not None and entry[1] is not None:
                block, tsize = entry[1], entry[2]
                if ne[0] % block:
                    raise GGUFFormatError(
                        f"{self.path}: tensor {name!r} is {entry[0]} with ne[0] = {ne[0]}, "
                        f"which is not a multiple of its block size {block}"
                    )
                nbytes = n_elements // block * tsize
                end = self.data_offset + rel + nbytes
                if end > self.size:
                    raise GGUFFormatError(
                        f"{self.path}: tensor {name!r} ends at byte {end}, past the end "
                        f"of the file ({self.size} bytes) -- the file is truncated"
                    )
            # A type with no known size (removed or unknown) keeps nbytes None;
            # it can be listed but never read, and `raw` refuses it by name.
            self.tensors[name] = GGUFTensorInfo(
                name, ggml_type, ne, self.data_offset + rel, nbytes, n_elements
            )

    # -- access --------------------------------------------------------------

    def info(self, name):
        try:
            return self.tensors[name]
        except KeyError:
            raise KeyError(f"{self.path} has no tensor named {name!r}") from None

    def raw(self, name, refuse=True):
        """The tensor's bytes exactly as stored: GGML blocks, or packed scalars.

        With `refuse=True` (the default) a type the loader would refuse is
        refused here too, so nobody builds on bytes the rest of the stack cannot
        interpret. `refuse=False` returns them anyway when their size is known
        -- that is for tests and for tooling that copies tensors through
        untouched, never for computing with them.
        """
        info = self.info(name)
        if refuse:
            why = refusal(info.ggml_type, name)
            if why is not None:
                raise why
        if info.nbytes is None:
            raise refusal(info.ggml_type, name) or UnsupportedGGUF(
                f"tensor {name!r}: {info.type_name} has no known byte size"
            )
        return self._mm[info.offset : info.offset + info.nbytes]

    def __repr__(self):
        return (
            f"GGUFFile({self.path!r}, version={getattr(self, 'version', '?')}, "
            f"tensors={len(getattr(self, 'tensors', {}))}, "
            f"metadata={len(getattr(self, 'metadata', {}))})"
        )
