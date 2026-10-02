"""GGUF: the reader, the block mapping, and the `from_pretrained` door.

docs/graph/GGUF.md is the design record; this file is its measurement. Issue
#17's four completion criteria map onto it like this:

  1. *A GGUF checkpoint loads into a real transformers model through
     torchnative* -> `test_from_pretrained_gguf_loads_qwen3_with_the_reference_weights`.
  2. *Weights match the reference reader bit for bit; quantised blocks
     dequantise to the reference values* -> the `real_file` tests and
     `test_random_blocks_dequantise_like_gguf_py_bit_for_bit`.
  3. *Unsupported quantisation types refuse by name* ->
     `test_every_unsupported_type_refuses_by_name`.
  4. *Gate green* -> not claimable from this file.

**The reference.** llama.cpp's own Python package, `gguf` (0.17.1), run in a
**separate interpreter** -- `.caches/gguf-venv/bin/python`, a venv that holds
`gguf` and numpy and no torch -- through `gguf_ref.py`. It is never imported
here and never installed into the gate's interpreter (AGENTS.md §15.2). Its
reader and its numpy dequantisers share no code with candle or with
`torchnative.gguf`, which is what makes agreement mean something (AGENTS.md
§16). When that venv is absent, the tests that need it **skip by name**;
`ggml_ref.py` (in-process) still covers Q8_0, Q4_0 and Q4_K as a third,
independent spelling.

**The real files.** `.caches/gguf/Qwen3-0.6B-Q8_0.gguf` (Qwen's own, F32 +
Q8_0) and `.caches/gguf/Qwen3-0.6B-Q4_K_M.gguf` (unsloth's, F32 + Q4_K +
Q6_K), fetched once at pinned revisions with their LFS sha256 checked
(docs/graph/GGUF.md §4). **Nothing is downloaded here**; a missing file is a
skip by name. In a worktree `.caches/` lives in the main checkout and is found
there; `TORCHNATIVE_CACHES` overrides.

**No tolerance appears in this file.** Every comparison is equality of bytes:
the GGML blocks, the dequantised `float32`, the dense tensors. A tolerance
wide enough to admit a lossy format's error is wide enough to admit a wrong
kernel (ggml_ref.py's docstring), and for weights it is not acceptable at all.

**What these checks cannot find.** NaN and infinite block scales are excluded
from the random fixtures -- NaN payload propagation is allowed to differ
between `half`'s conversion and numpy's without either being wrong, so a
NaN-bearing block would test the float environment, not the format. No
checkpoint carries one. And agreement with `gguf.quants` is agreement with
llama.cpp's *Python* dequantiser; its C kernels are not run here.

Nullifications this file catches (each was reasoned through, none has been
run -- the round that wrote this could not build or gate):

* shape not reversed (`ne` handed to candle as-is) -> `real_file_*`, the
  linear test, and `test_the_comparisons_fail_when_the_bytes_or_layout_are_wrong`
* a block byte altered on the way in -> `test_blocks_reach_candle_unchanged`
* a refused type mapped to a neighbour -> `test_every_unsupported_type_refuses_by_name`
* the door dequantising differently from gguf-py -> the `from_pretrained` test
"""

import glob
import hashlib
import json
import os
import random
import shutil
import struct
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
_VENDOR_DIR = os.path.join(_ROOT, "torchnative", "src", "main")
sys.path.insert(0, _VENDOR_DIR)
sys.path.insert(0, _HERE)

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import torch  # noqa: E402

import _skip  # noqa: E402
import ggml_ref  # noqa: E402

_C = torch._C
_SUITE = "test_gguf"


def _caches_dir():
    env = os.environ.get("TORCHNATIVE_CACHES")
    if env:
        return env
    here = os.path.join(_ROOT, ".caches")
    if os.path.isdir(here):
        return here
    parts = _ROOT.split(os.sep)
    if ".worktrees" in parts:
        return os.path.join(os.sep.join(parts[: parts.index(".worktrees")]), ".caches")
    return here


_CACHES = _caches_dir()
_GGUF_PY = os.environ.get(
    "TORCHNATIVE_GGUF_PYTHON", os.path.join(_CACHES, "gguf-venv", "bin", "python")
)
_GGUF_DIR = os.environ.get("TORCHNATIVE_GGUF_DIR", os.path.join(_CACHES, "gguf"))
_REAL_FILES = {
    # name -> (file, the GGML types it is expected to carry)
    "Q8_0": ("Qwen3-0.6B-Q8_0.gguf", {"F32", "Q8_0"}),
    "Q4_K_M": ("Qwen3-0.6B-Q4_K_M.gguf", {"F32", "Q4_K", "Q6_K"}),
}
_TMP = os.path.join(_ROOT, ".tmp", f"test_gguf.{os.getpid()}")

# Byte offsets of every f16 field in each block type's block, from ggml's
# struct layouts (`ggml-common.h`). Used only to keep random fixtures finite.
_F16_FIELDS = {
    "q4_0": (0,), "q4_1": (0, 2), "q5_0": (0,), "q5_1": (0, 2), "q8_0": (0,),
    "q2_k": (80, 82), "q3_k": (108,), "q4_k": (0, 2), "q5_k": (0, 2), "q6_k": (208,),
}

_LOADABLE = {"F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0",
             "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K"}
_REFUSED = {"Q8_1", "Q8_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ1_S", "IQ4_NL",
            "IQ3_S", "IQ2_S", "IQ4_XS", "I8", "I16", "I32", "I64", "F64", "IQ1_M",
            "Q4_0_4_4", "Q4_0_4_8", "Q4_0_8_8", "TQ1_0", "TQ2_0", "IQ4_NL_4_4",
            "IQ4_NL_4_8", "IQ4_NL_8_8", "MXFP4"}


def _shim():
    assert hasattr(torch._C, "_aten_implemented"), (
        "not the torchnative shim -- torch._C has no _aten_implemented"
    )


def _gguf():
    import torchnative.gguf as g

    return g


# -- a GGUF writer, for fixtures ------------------------------------------------
#
# Written from the GGUF spec, independently of `reader.py`, so a misreading has
# to be made twice -- and the real-file tests compare the reader against
# gguf-py on files llama.cpp wrote, which closes that.

_FMT = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?",
        10: "<Q", 11: "<q", 12: "<d"}


def _s(text):
    raw = text.encode("utf-8", "surrogateescape") if isinstance(text, str) else text
    return struct.pack("<Q", len(raw)) + raw


def _val(vtype, value):
    if vtype in _FMT:
        return struct.pack(_FMT[vtype], value)
    if vtype == 8:
        return _s(value)
    if vtype == 9:
        etype, items = value
        return struct.pack("<IQ", etype, len(items)) + b"".join(_val(etype, v) for v in items)
    raise ValueError(vtype)


def _gguf_bytes(kv, tensors, alignment=32, version_bytes=None, magic=b"GGUF", rel=None):
    """kv: [(key, vtype, value)]; tensors: [(name, ggml_type, ne, raw)]."""
    out = bytearray(magic)
    out += version_bytes if version_bytes is not None else struct.pack("<I", 3)
    if alignment != 32:
        kv = list(kv) + [("general.alignment", 4, alignment)]
    out += struct.pack("<QQ", len(tensors), len(kv))
    for key, vtype, value in kv:
        out += _s(key) + struct.pack("<I", vtype) + _val(vtype, value)
    data = bytearray()
    for name, gtype, ne, raw in tensors:
        data += b"\0" * (-len(data) % alignment)
        offset = (rel or {}).get(name, len(data))
        out += _s(name) + struct.pack("<I", len(ne))
        out += b"".join(struct.pack("<Q", d) for d in ne)
        out += struct.pack("<IQ", gtype, offset)
        data += raw
    out += b"\0" * (-len(out) % alignment)
    return bytes(out + data)


def _write(name, blob):
    os.makedirs(_TMP, exist_ok=True)
    path = os.path.join(_TMP, name)
    with open(path, "wb") as fh:
        fh.write(blob)
    return path


def _type_id(name):
    g = _gguf()
    return next(i for i, e in g.GGML_TYPES.items() if e[0] == name)


def _random_blocks(fmt, n_blocks, seed):
    """Random GGML blocks with every f16 field forced finite (module docstring)."""
    g = _gguf()
    tid = next(i for i, e in g.GGML_TYPES.items() if e[4] == fmt and e[3] == "block")
    size = g.GGML_TYPES[tid][2]
    blob = bytearray(random.Random(seed).randbytes(size * n_blocks))
    for b in range(n_blocks):
        for off in _F16_FIELDS[fmt]:
            hi = b * size + off + 1
            if blob[hi] & 0x7C == 0x7C:  # exponent all ones: inf or NaN
                blob[hi] &= ~0x40 & 0xFF
    return bytes(blob)


def _random_dense(gtype_name, n, seed):
    """Random F32/F16/BF16 bit patterns, NaN and inf excluded."""
    rnd = random.Random(seed)
    if gtype_name == "F32":
        out = bytearray(rnd.randbytes(4 * n))
        for i in range(n):
            if out[4 * i + 3] & 0x7F == 0x7F and out[4 * i + 2] & 0x80:
                out[4 * i + 3] &= 0xBF
        return bytes(out)
    out = bytearray(rnd.randbytes(2 * n))
    for i in range(n):
        hi = 2 * i + 1
        if gtype_name == "F16" and out[hi] & 0x7C == 0x7C:
            out[hi] &= 0xBF
        if gtype_name == "BF16" and out[hi] & 0x7F == 0x7F and out[hi - 1] & 0x80:
            out[hi] &= 0xBF
    return bytes(out)


def _f32_bytes(t):
    """A dense float32 tensor's bytes, exactly: candle's F32 `QTensor` is a copy
    (`k_quants.rs`, `impl GgmlType for f32`: `copy_from_slice`)."""
    return bytes(_C._quantized_blob(_C._quantize(t.contiguous(), "f32")))


def _dense_bytes(t, gtype_name):
    """A dense tensor's bytes in its own dtype. f16/bf16 go through candle's
    F16/BF16 `QTensor`, which widens to f32 and narrows back -- exact for a
    value that started in that dtype."""
    fmt = {"F32": "f32", "F16": "f16", "BF16": "bf16"}[gtype_name]
    return bytes(_C._quantized_blob(_C._quantize(t.contiguous(), fmt)))


def _first_difference(a, b, width=4):
    if len(a) != len(b):
        return f"lengths differ: {len(a)} vs {len(b)}"
    for i in range(0, len(a), width):
        if a[i : i + width] != b[i : i + width]:
            return f"first differing {width}-byte word at offset {i}: {a[i:i+width].hex()} vs {b[i:i+width].hex()}"
    return None


# -- the reference subprocess -------------------------------------------------

_REF_CACHE = {}


def _ref_env():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "PYTHONHOME")}
    env["PYTHONNOUSERSITE"] = "1"
    return env


def _ref(args, stdin=None, timeout=3600):
    """Run gguf_ref.py under the gguf venv, or raise `_skip.Skip` naming why not."""
    if not os.path.isfile(_GGUF_PY):
        raise _skip.Skip(
            f"no reference interpreter at {_GGUF_PY} (a venv with llama.cpp's `gguf` "
            "package; docs/graph/GGUF.md §3 says how it was made)"
        )
    key = (tuple(args), stdin)
    if key in _REF_CACHE:
        return _REF_CACHE[key]
    proc = subprocess.run(
        [_GGUF_PY, "-B", os.path.join(_HERE, "gguf_ref.py"), *args],
        input=stdin, capture_output=True, text=True, env=_ref_env(), timeout=timeout,
    )
    assert proc.returncode == 0, f"gguf_ref.py {args[0]} failed:\n{proc.stderr[-3000:]}"
    out = json.loads(proc.stdout)
    _REF_CACHE[key] = out
    return out


def _real_file(label):
    fname, _ = _REAL_FILES[label]
    path = os.path.join(_GGUF_DIR, fname)
    if not os.path.isfile(path):
        raise _skip.Skip(f"{fname} is not cached at {_GGUF_DIR} (no download in the gate)")
    return path


# =============================================================================
# The container
# =============================================================================


def test_a_written_file_reads_back_header_metadata_and_descriptors():
    """Every metadata value type, a non-default alignment, and two tensors.

    The values are read back by equality, the shape comes back reversed (torch
    order), and every tensor offset is absolute and aligned.
    """
    g = _gguf()
    kv = [
        ("general.architecture", 8, "qwen3"),
        ("t.u8", 0, 250), ("t.i8", 1, -7), ("t.u16", 2, 65000), ("t.i16", 3, -30000),
        ("t.u32", 4, 4000000000), ("t.i32", 5, -2000000000), ("t.f32", 6, 0.1),
        ("t.bool", 7, True), ("t.u64", 10, 2**63 + 5), ("t.i64", 11, -(2**62)),
        ("t.f64", 12, 0.1),
        ("t.astr", 9, (8, ["a", "", "é中", "x" * 300])),
        ("t.ai32", 9, (5, [1, -2, 3])),
        ("t.empty", 9, (4, [])),
    ]
    q8 = _random_blocks("q8_0", 2 * 3, seed=1)
    f32 = struct.pack("<5f", 1.0, -2.5, 0.0, 3.25, -0.0)
    path = _write("container.gguf", _gguf_bytes(
        kv, [("norm", 0, [5], f32), ("w", 8, [64, 3], q8)], alignment=64,
    ))
    with g.GGUFFile(path) as f:
        assert f.version == 3 and f.alignment == 64, (f.version, f.alignment)
        assert f.data_offset % 64 == 0, f.data_offset
        md = f.metadata
        assert md["general.architecture"] == "qwen3"
        for key, _, value in kv:
            if key in ("t.f32",):
                assert md[key] == struct.unpack("<f", struct.pack("<f", value))[0], md[key]
            elif key.startswith("t.a") or key == "t.empty":
                assert list(md[key]) == value[1], (key, md[key])
                assert md[key].element_type == value[0], key
            else:
                assert md[key] == value, (key, md[key], value)
        assert f.metadata_types["t.astr"] == 9 and f.metadata_types["t.u8"] == 0
        assert list(f.tensors) == ["norm", "w"]
        norm, w = f.tensors["norm"], f.tensors["w"]
        assert norm.shape == (5,) and norm.nbytes == 20, norm
        assert w.ne == (64, 3) and w.shape == (3, 64), w
        assert w.nbytes == 6 * 34 and w.format == "q8_0" and w.kind == "block", w
        assert w.offset % 64 == 0 and w.offset >= f.data_offset, w
        assert f.raw("w") == q8, "the block bytes did not come back as written"
        assert f.raw("norm") == f32
    print(f"    gguf: header, {len(kv)} metadata types and 2 descriptors read back exactly")


def test_malformed_files_refuse_by_name():
    g = _gguf()
    q8 = _random_blocks("q8_0", 1, seed=2)
    good = [("w", 8, [32, 1], q8)]
    cases = [
        ("magic", _gguf_bytes([], good, magic=b"GGML"), g.GGUFFormatError, "not a GGUF file"),
        ("bigendian", _gguf_bytes([], good, version_bytes=struct.pack(">I", 3)), g.UnsupportedGGUF, "big-endian"),
        ("v1", _gguf_bytes([], good, version_bytes=struct.pack("<I", 1)), g.UnsupportedGGUF, "version 1"),
        ("v9", _gguf_bytes([], good, version_bytes=struct.pack("<I", 9)), g.UnsupportedGGUF, "version 9"),
        ("truncated", _gguf_bytes([], good)[:-5], g.GGUFFormatError, "truncated"),
        ("misaligned", _gguf_bytes([], good, rel={"w": 4}), g.GGUFFormatError, "not a multiple of the alignment"),
        ("partialblock", _gguf_bytes([], [("w", 8, [48, 1], q8 + q8[:17])]), g.GGUFFormatError, "block size 32"),
        ("duplicate", _gguf_bytes([], good + good), g.GGUFFormatError, "appears twice"),
        ("alignment", _gguf_bytes([], good, alignment=48), g.GGUFFormatError, "power of two"),
        ("dims", _gguf_bytes([], [("w", 8, [32, 1, 1, 1, 1], q8)]), g.GGUFFormatError, "dimensions"),
    ]
    for label, blob, exc_type, needle in cases:
        path = _write(f"bad_{label}.gguf", blob)
        try:
            g.GGUFFile(path)
        except exc_type as exc:
            assert needle in str(exc), (label, str(exc))
        else:
            raise AssertionError(f"{label}: a malformed file was accepted")
    print(f"    gguf: {len(cases)} malformed files each refused with their own reason")


def test_every_unsupported_type_refuses_by_name():
    """Refused types are named, not mapped onto a neighbour; the split is pinned.

    The two sets are written out here rather than read from `GGML_TYPES`, so
    moving a type from refused to loadable is an edit to this test as well --
    a deliberate act, not a side effect of a table change.
    """
    _shim()
    g = _gguf()
    names = {e[0]: (i, e) for i, e in g.GGML_TYPES.items()}
    loadable = {n for n, (_, e) in names.items() if e[3] in ("dense", "block")}
    refused = {n for n, (_, e) in names.items() if e[3] == "refused"}
    assert loadable == _LOADABLE, (sorted(loadable ^ _LOADABLE))
    assert refused == _REFUSED, (sorted(refused ^ _REFUSED))

    tensors = []
    for n in sorted(refused):
        tid, e = names[n]
        if e[1] is None:
            tensors.append((f"t.{n}", tid, [32, 1], b""))
        else:
            tensors.append((f"t.{n}", tid, [e[1], 1], b"\0" * e[2]))
    tensors.append(("t.unknown", 99, [32, 1], b""))
    path = _write("refused.gguf", _gguf_bytes([], tensors))
    with g.GGUFFile(path) as f:
        for name, tid, _, _ in tensors:
            label = name[2:]
            for attempt in (
                lambda: g.load_tensor(f, name),
                lambda: g.load_tensor(f, name, keep_blocks=True),
                lambda: g.quantized_linear(f, name),
                lambda: f.raw(name),
            ):
                try:
                    attempt()
                except g.UnsupportedGGUF as exc:
                    want = "99" if label == "unknown" else label
                    assert want in str(exc), (label, str(exc))
                else:
                    raise AssertionError(f"{label} (id {tid}) was not refused")
    print(f"    gguf: {len(refused)} named types and one unknown id refuse by name, four ways each")


# =============================================================================
# Blocks into candle
# =============================================================================


def _block_formats():
    g = _gguf()
    return sorted(e[4] for e in g.GGML_TYPES.values() if e[3] == "block")


def test_the_type_table_matches_candle():
    """Block and byte sizes as candle reports them, not as anyone restated them:
    a block-sized row quantises to exactly one block, one element short is refused.
    """
    _shim()
    g = _gguf()
    formats = set(_C._quantized_formats())
    for tid, (name, block, size, kind, fmt) in sorted(g.GGML_TYPES.items()):
        if kind != "block":
            continue
        assert fmt in formats, f"{name} maps to {fmt!r}, which this build does not have"
        q = _C._quantize(torch.zeros(1, block), fmt)
        assert _C._quantized_nbytes(q) == size, (name, _C._quantized_nbytes(q), size)
        if block > 1:
            try:
                _C._quantize(torch.zeros(1, block - 1), fmt)
            except NotImplementedError:
                pass
            else:
                raise AssertionError(f"candle took {block - 1} elements as {name}")
    print("    gguf: every block type's sizes are candle's")


def test_the_type_table_matches_gguf_py():
    """Type ids and sizes against llama.cpp's own table (`gguf.GGML_QUANT_SIZES`)."""
    g = _gguf()
    table = _ref(["table"])
    known_disagreement = {"Q8_1"}  # reader.py: gguf 0.17.1 says 40, ggml/candle 36
    for tid, (name, block, size, kind, _) in sorted(g.GGML_TYPES.items()):
        ref = table.get(name)
        if ref is None:
            continue  # removed or newer than gguf 0.17.1 (MXFP4); refused either way
        assert ref["id"] == tid, (name, ref["id"], tid)
        if name in known_disagreement:
            assert (ref["block"], ref["size"]) != (block, size), (
                f"gguf now agrees on {name}'s size -- drop it from known_disagreement "
                "and from reader.py's note"
            )
            continue
        assert (ref["block"], ref["size"]) == (block, size), (name, ref, block, size)
    print(f"    gguf: ids and block/type sizes agree with gguf-py ({len(table)} gguf types)")


def test_blocks_reach_candle_unchanged():
    """The file's bytes are candle's bytes: `_quantized_blob(load_tensor(keep_blocks=True))`
    returns exactly what the file holds, for every block type, through the reader.
    """
    _shim()
    g = _gguf()
    tensors = []
    for k, fmt in enumerate(_block_formats()):
        tid = _type_id(fmt.upper())
        block = g.GGML_TYPES[tid][1]
        tensors.append((f"w.{fmt}", tid, [block * 2, 3], _random_blocks(fmt, 6, seed=100 + k)))
    path = _write("blocks.gguf", _gguf_bytes([], tensors))
    with g.GGUFFile(path) as f:
        for name, tid, ne, raw in tensors:
            q = g.load_tensor(f, name, keep_blocks=True)
            assert _C._quantized_format(q) == f.tensors[name].format, name
            assert tuple(q.shape) == (ne[1], ne[0]), (name, tuple(q.shape))
            got = bytes(_C._quantized_blob(q))
            why = _first_difference(got, raw)
            assert why is None, f"{name}: {why}"
    print(f"    gguf: {len(tensors)} block types reach candle byte for byte")


def test_dense_types_read_back_exactly_in_their_own_dtype():
    _shim()
    g = _gguf()
    tensors = []
    for k, name in enumerate(("F32", "F16", "BF16")):
        width = {"F32": 4, "F16": 2, "BF16": 2}[name]
        raw = _random_dense(name, 7 * 5, seed=200 + k)
        assert len(raw) == 7 * 5 * width
        tensors.append((f"d.{name}", _type_id(name), [5, 7], raw))
    path = _write("dense.gguf", _gguf_bytes([], tensors))
    with g.GGUFFile(path) as f:
        for name, _, _, raw in tensors:
            kind = name[2:]
            t = g.load_tensor(f, name)
            want = {"F32": torch.float32, "F16": torch.float16, "BF16": torch.bfloat16}[kind]
            assert t.dtype == want, (name, t.dtype)
            assert tuple(t.shape) == (7, 5), (name, tuple(t.shape))
            why = _first_difference(_dense_bytes(t, kind), raw, width=2)
            assert why is None, f"{name}: {why}"
    print("    gguf: F32/F16/BF16 come back in their own dtype, bit for bit")


def test_random_blocks_dequantise_like_gguf_py_bit_for_bit():
    """candle's dequantisation of a block against gguf-py's, as float32 bytes.

    Random blocks rather than quantised weights: a dequantiser is a function of
    arbitrary bytes, and random bytes reach every scale, sign and nibble
    pattern a quantiser might not produce. The in-process `ggml_ref` spelling
    is checked first and does not need the reference venv.
    """
    _shim()
    rows = 3
    fixtures = {}
    for k, fmt in enumerate(_block_formats()):
        g = _gguf()
        tid = _type_id(fmt.upper())
        block = g.GGML_TYPES[tid][1]
        cols = block * 4
        blob = _random_blocks(fmt, rows * 4, seed=300 + k)
        got = _f32_bytes(_C._dequantize(_C._quantized_from_blob(blob, [rows, cols], fmt)))
        fixtures[fmt] = (blob, got)
        if fmt in ggml_ref.DEQUANTIZERS:
            want = struct.pack(f"<{rows * cols}f", *ggml_ref.DEQUANTIZERS[fmt](blob, rows * cols))
            why = _first_difference(got, want)
            assert why is None, f"{fmt} vs ggml_ref: {why}"

    items = [{"type": fmt.upper(), "rows": rows, "hex": blob.hex()} for fmt, (blob, _) in sorted(fixtures.items())]
    ref = _ref(["dequant"], stdin=json.dumps(items))
    for item, out in zip(items, ref):
        fmt = item["type"].lower()
        assert "error" not in out, (fmt, out)
        why = _first_difference(fixtures[fmt][1], bytes.fromhex(out["f32_hex"]))
        assert why is None, f"{fmt} vs gguf.quants.dequantize: {why}"
    print(f"    gguf: {len(items)} block types dequantise bit for bit like gguf-py")


def test_a_gguf_linear_holds_the_file_bytes_and_multiplies_exactly():
    """`quantized_linear` is the mapping onto torchnative's module, and it runs.

    Q8_0 with scale 1.0 and integer weights is lossless, and an activation whose
    every 32-block has absmax 127 quantises losslessly too, so the matmul is
    exact (test_shim.py's `_exactly_representable` argument). The output is then
    compared to a dense `linear` **bit for bit**. Non-square on purpose: a weight
    read transposed cannot multiply a 64-wide input into 5 outputs.

    **Refuses to run under `CANDLE_DEQUANTIZE_ALL` / `_F16`.** Either switch
    makes `QMatMul::from_arc` dequantise every weight and take a dense matmul
    (`vendor/candle-core/src/quantized/mod.rs`), and on these lossless operands
    the dense answer is the same bits -- so the test would pass while measuring
    a path that never touches the blocks. The Q4-quality round (#16) found this
    and refuses the same way (`tools/q4quality/measure.py`).
    """
    _shim()
    for var in ("CANDLE_DEQUANTIZE_ALL", "CANDLE_DEQUANTIZE_ALL_F16"):
        assert os.environ.get(var, "") in ("", "0"), (
            f"{var} is set: candle would multiply by dequantised weights, so this "
            "test cannot measure the quantised matmul. Refusing rather than passing."
        )
    g = _gguf()
    out_f, in_f = 5, 64
    rnd = random.Random(400)
    blob = bytearray()
    dense = []
    for _ in range(out_f):
        row = []
        for _ in range(in_f // 32):
            qs = [rnd.randint(-126, 126) for _ in range(32)]
            qs[rnd.randrange(32)] = 127
            blob += struct.pack("<e", 1.0) + struct.pack("<32b", *qs)
            row += [float(v) for v in qs]
        dense.append(row)
    path = _write("linear.gguf", _gguf_bytes([], [("blk.0.ffn_up.weight", 8, [in_f, out_f], bytes(blob))]))
    with g.GGUFFile(path) as f:
        layer = g.quantized_linear(f, "blk.0.ffn_up.weight")
    assert (layer.in_features, layer.out_features, layer.format) == (in_f, out_f, "q8_0"), layer
    assert bytes(_C._quantized_blob(layer.qweight)) == bytes(blob)

    xs = []
    for _ in range(2):
        row = [float(rnd.randint(-126, 126)) for _ in range(in_f)]
        row[0], row[32] = 127.0, -127.0
        xs.append(row)
    x = torch.tensor(xs)
    want = torch.nn.functional.linear(x, torch.tensor(dense))
    got = layer(x)
    assert _f32_bytes(got) == _f32_bytes(want), (got.tolist(), want.tolist())
    print("    gguf: a Q8_0 tensor becomes a QuantizedLinear holding its bytes and multiplies exactly")


def test_the_comparisons_fail_when_the_bytes_or_layout_are_wrong():
    """AGENTS.md §17.5: the comparisons above must be able to fail.

    Four faults shaped like real reader bugs, each fed to the same comparison
    the tests use: one scale bit altered, a square weight read transposed, a
    non-square weight handed over with GGUF's `ne` un-reversed, and a refused
    type smuggled in as a neighbour with the same block size.

    **What the flat digest cannot see, and what sees it instead.** An
    un-reversed shape keeps the same flat bytes, so the dequantised-bytes
    digest alone is blind to it. The real-file tests compare each shape with
    gguf-py's, and a non-square linear cannot multiply -- (c) below.
    """
    _shim()
    g = _gguf()
    # (a) one bit of block 0's super-scale `d`
    blob = _random_blocks("q4_k", 6, seed=500)
    good = _f32_bytes(_C._dequantize(_C._quantized_from_blob(blob, [3, 512], "q4_k")))
    bad_scale = bytearray(blob)
    bad_scale[0] ^= 0x01
    got = _f32_bytes(_C._dequantize(_C._quantized_from_blob(bytes(bad_scale), [3, 512], "q4_k")))
    assert _first_difference(got, good) is not None, "a scale bit flip went unseen"

    # (b) a square Q8_0 tensor read transposed: same bytes, values moved
    sq = _random_blocks("q8_0", 32, seed=501)  # 32 x 32
    a = _C._dequantize(_C._quantized_from_blob(sq, [32, 32], "q8_0"))
    assert _first_difference(_f32_bytes(a), _f32_bytes(a.t().contiguous())) is not None

    # (c) a 5 x 64 linear weight given as ne = [64, 5] un-reversed. Measured:
    # candle refuses it already at construction (`QTensor::new`: "last dim
    # divisible by block size [64, 5] 32"), so it cannot even reach a matmul.
    lin = _random_blocks("q8_0", 10, seed=502)
    try:
        wrong = _C._quantized_from_blob(lin, [64, 5], "q8_0")
        _C._quantized_linear(torch.ones(2, 64), wrong, None)
    except (RuntimeError, NotImplementedError) as exc:
        assert "block size" in str(exc) or "cannot be multiplied" in str(exc), str(exc)
    else:
        raise AssertionError("a weight with GGUF's ne order multiplied a 64-wide input")

    # (d) IQ4_NL is 18 bytes per 32 elements, like Q4_0
    iq = _gguf_bytes([], [("t", _type_id("IQ4_NL"), [32, 1], b"\0" * 18)])
    with g.GGUFFile(_write("smuggle.gguf", iq)) as f:
        raw = f.raw("t", refuse=False)
        # The same bytes read as Q4_0 (same 18-byte layout size) give numbers --
        # which is exactly why the reader refuses rather than mapping by size.
        _C._quantized_from_blob(raw, [1, 32], "q4_0")
        try:
            g.load_tensor(f, "t")
        except g.UnsupportedGGUF:
            pass
        else:
            raise AssertionError("IQ4_NL was loaded")
    print("    gguf: a flipped scale bit, a transposed read and a smuggled type are all visible")


# =============================================================================
# Real files, against the reference reader
# =============================================================================


def _canonical_kv(f, key):
    """The spelling gguf_ref.py `_kv_digest` hashes, built from our parse."""
    h = hashlib.sha256()
    vtype = f.metadata_types[key]
    value = f.metadata[key]
    h.update(struct.pack("<I", vtype))

    def one(t, v):
        if t == 8:
            raw = v.encode("utf-8", "surrogateescape")
            h.update(struct.pack("<Q", len(raw)))
            h.update(raw)
        else:
            h.update(struct.pack(_FMT[t], v))

    if vtype == 9:
        h.update(struct.pack("<IQ", value.element_type, len(value)))
        for v in value:
            one(value.element_type, v)
    else:
        one(vtype, value)
    return h.hexdigest()


def _real_checks(label):
    _shim()
    g = _gguf()
    path = _real_file(label)
    ref = _ref(["file", path])
    with g.GGUFFile(path) as f:
        assert (f.version, f.alignment, f.data_offset) == (
            ref["version"], ref["alignment"], ref["data_offset"]
        ), (f.version, f.alignment, f.data_offset, ref["version"], ref["alignment"], ref["data_offset"])
        assert sorted(f.metadata) == sorted(ref["kv"]), sorted(set(f.metadata) ^ set(ref["kv"]))
        bad_kv = [k for k in f.metadata if _canonical_kv(f, k) != ref["kv"][k]]
        assert not bad_kv, f"metadata values differ from gguf-py's: {bad_kv[:5]}"

        assert list(f.tensors) == [t["name"] for t in ref["tensors"]]
        seen = set()
        for t in ref["tensors"]:
            info = f.tensors[t["name"]]
            assert (info.type_name, list(info.shape), info.offset, info.nbytes) == (
                t["type"], t["shape"], t["offset"], t["nbytes"]
            ), (t["name"], info, t)
            seen.add(info.type_name)
            raw = f.raw(t["name"])
            assert hashlib.sha256(raw).hexdigest() == t["raw_sha256"], t["name"]
            assert t["f32_sha256"] is not None, (t["name"], t.get("f32_error"))

            # The blocks, as candle holds them.
            if info.kind == "block":
                q = g.load_tensor(f, t["name"], keep_blocks=True)
                assert hashlib.sha256(bytes(_C._quantized_blob(q))).hexdigest() == t["raw_sha256"], (
                    f"{t['name']}: candle does not hold the file's bytes"
                )
                del q

            # The dequantised values, row-chunked as the reference chunks them.
            cols = info.shape[-1]
            n_rows = info.n_elements // cols
            row_bytes = info.nbytes // n_rows
            h = hashlib.sha256()
            for start in range(0, n_rows, 4096):
                stop = min(n_rows, start + 4096)
                chunk = raw[start * row_bytes : stop * row_bytes]
                if info.kind == "block":
                    piece = _C._dequantize(_C._quantized_from_blob(chunk, [stop - start, cols], info.format))
                else:
                    # gguf-py's F16/BF16 "dequantisation" is a widening to
                    # float32, which is exact; F32 is the bytes themselves.
                    dtype = getattr(torch, g.GGML_TYPES[info.ggml_type][4])
                    piece = torch.frombuffer(chunk, dtype=dtype).reshape(stop - start, cols)
                    piece = piece.to(torch.float32)
                h.update(_f32_bytes(piece))
            assert h.hexdigest() == t["f32_sha256"], (
                f"{t['name']} ({info.type_name}): candle's dequantisation is not gguf-py's"
            )
        want_types = _REAL_FILES[label][1]
        assert seen == want_types, (label, seen, want_types)
    print(
        f"    gguf: {os.path.basename(path)}: {len(ref['tensors'])} tensors, "
        f"{len(ref['kv'])} metadata keys, types {sorted(seen)} -- blocks byte-exact, "
        "dequantisation bit-exact against gguf-py"
    )


def test_real_file_q8_0_matches_the_reference_reader():
    _real_checks("Q8_0")


def test_real_file_q4_k_m_matches_the_reference_reader():
    _real_checks("Q4_K_M")


# =============================================================================
# The from_pretrained door
# =============================================================================


def test_the_door_refuses_by_name_before_resolving_anything():
    """No network, no file: each refusal must come before the id is looked up."""
    _shim()
    import torchnative.transformers as tnt
    from torchnative.gguf import UnsupportedGGUF

    try:
        tnt.AutoModelForCausalLM.from_pretrained(
            "torchnative/no-such-repo", gguf_file="no-such.gguf", quantization_config=object()
        )
    except UnsupportedGGUF as exc:
        assert "quantization_config" in str(exc), str(exc)
    else:
        raise AssertionError("gguf_file + quantization_config was accepted")

    from transformers.utils.import_utils import is_gguf_available

    if is_gguf_available():
        _skip.skip("`gguf` is importable in the gate's interpreter, so its absence cannot be refused here")
        return
    try:
        tnt.AutoModelForCausalLM.from_pretrained("torchnative/no-such-repo", gguf_file="no-such.gguf")
    except ImportError as exc:
        assert "gguf" in str(exc), str(exc)
    else:
        raise AssertionError("a GGUF load without the gguf package was not refused")
    print("    gguf: quantization_config and a missing gguf package refuse before any lookup")


def test_a_tensor_rewriting_architecture_refuses_by_name():
    """Llama's GGUF q/k weights are permuted by a numpy processor upstream; not ported."""
    _shim()
    g = _gguf()
    from torchnative.gguf.hf import state_dict_from_gguf

    path = _write("llama.gguf", _gguf_bytes([("general.architecture", 8, "llama")], []))
    with g.GGUFFile(path) as f:
        try:
            state_dict_from_gguf(f, None, None)
        except g.UnsupportedGGUF as exc:
            assert "LlamaTensorProcessor" in str(exc) and "'llama'" in str(exc), str(exc)
        else:
            raise AssertionError("a llama GGUF was accepted without its q/k permutation")
    print("    gguf: an architecture whose tensors are rewritten upstream refuses by name")


_DOOR_SCRIPT = r"""
import hashlib, json, os, sys
sys.path.append(os.environ["TN_GGUF_SITE"])  # after site-packages: only `gguf` resolves from it
import torch
assert hasattr(torch._C, "_aten_implemented"), "not the shim"
from transformers.utils.import_utils import is_gguf_available
assert is_gguf_available(), "gguf did not become importable"
from torchnative.transformers import AutoModelForCausalLM

path = os.environ["TN_GGUF_FILE"]
model = AutoModelForCausalLM.from_pretrained(
    os.path.dirname(path), gguf_file=os.path.basename(path), dtype=torch.float32
)
_C = torch._C
def f32(t):
    return bytes(_C._quantized_blob(_C._quantize(t.contiguous(), "f32")))
sd = model.state_dict()
digests = {}
for hf_name, (gguf_name, kind) in model.torchnative_gguf.loaded.items():
    t = sd[hf_name]
    assert t.dtype == torch.float32, (hf_name, t.dtype)
    h = hashlib.sha256()
    rows = t.reshape(-1, t.shape[-1])
    for s in range(0, rows.shape[0], 4096):
        h.update(f32(rows[s : s + 4096]))
    digests[gguf_name] = h.hexdigest()
ids = torch.tensor([[1, 2, 3, 4]])
with torch.no_grad():
    logits = model(input_ids=ids).logits
print(json.dumps({
    "class": type(model).__module__ + "." + type(model).__name__,
    "digests": digests,
    "unmapped": model.torchnative_gguf.unmapped,
    "tied": model.lm_head.weight is model.model.embed_tokens.weight,
    "logits_shape": list(logits.shape),
    "logits_finite": bool(torch.isfinite(logits).all()),
}))
"""


def _door(path):
    """Load `path` through the door in a subprocess; return (its JSON, gguf-py's digests).

    The subprocess gets the gguf venv's site-packages **appended** to
    `sys.path`, so spike-venv's numpy and transformers win and only `gguf`
    resolves from there (docs/graph/GGUF.md §3).
    """
    _shim()
    ref = _ref(["file", path])
    if not os.path.isfile(_GGUF_PY):
        raise _skip.Skip(f"no reference interpreter at {_GGUF_PY}")
    sites = glob.glob(os.path.join(os.path.dirname(os.path.dirname(_GGUF_PY)), "lib", "python3*", "site-packages"))
    if not sites:
        raise _skip.Skip(f"no site-packages beside {_GGUF_PY}")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([_VENDOR_DIR] + [p for p in env.get("PYTHONPATH", "").split(os.pathsep) if p])
    env["TN_GGUF_SITE"] = sites[0]
    env["TN_GGUF_FILE"] = path
    env["HF_HUB_OFFLINE"] = "1"
    env["HF_HOME"] = os.path.join(_CACHES, "hf-home")
    proc = subprocess.run(
        [sys.executable, "-c", _DOOR_SCRIPT], capture_output=True, text=True, env=env, timeout=3600
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    return out, ref


def _door_checks(out, ref):
    assert out["class"].startswith("transformers.models.qwen3."), out["class"]
    want = {t["name"]: t["f32_sha256"] for t in ref["tensors"]}
    assert out["unmapped"] == [], out["unmapped"]
    assert set(out["digests"]) == set(want), sorted(set(want) ^ set(out["digests"]))
    bad = [n for n, d in out["digests"].items() if d != want[n]]
    assert not bad, f"{len(bad)} parameters differ from gguf-py's dequantisation, e.g. {bad[:5]}"
    assert out["tied"], "lm_head is not tied to the embedding, but the file has no output.weight"
    assert out["logits_shape"][:2] == [1, 4] and out["logits_finite"], out
    return (
        f"from_pretrained(gguf_file=) built {out['class']}; "
        f"{len(out['digests'])} of {len(want)} tensors agree with gguf-py bit for bit "
        f"({', '.join(sorted({t['type'] for t in ref['tensors']}))}); forward reaches finite logits"
    )


def test_from_pretrained_gguf_on_a_tiny_qwen3_written_by_gguf_py():
    """The door end to end, on a file llama.cpp's own writer produced.

    One layer, nine tensor types, a few hundred kilobytes -- so the door is
    exercised on every gate run without a real checkpoint, and on types the two
    Qwen files do not carry (F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1). Graded like the
    real-file test below: **agrees** on weights, **reaches** on the forward.
    """
    _shim()
    os.makedirs(_TMP, exist_ok=True)
    path = os.path.join(_TMP, "tiny-qwen3.gguf")
    _ref(["tiny", path])
    out, ref = _door(path)
    print("    gguf: tiny: " + _door_checks(out, ref))


def test_from_pretrained_gguf_loads_qwen3_with_the_reference_weights():
    """Issue #17 criterion 1, on Qwen's own Qwen3-0.6B Q8_0 file.

    The real `transformers` Qwen3 class, loaded through
    `torchnative.transformers.AutoModelForCausalLM.from_pretrained(dir,
    gguf_file=...)`; every loaded parameter's float32 bytes are hashed and
    compared with gguf-py's dequantisation of the tensor it came from
    (**agrees**). The forward is only checked to run and stay finite -- logits
    are not compared with upstream here, so no agreement is claimed for them
    (**reaches**). Loads ~2.4 GB of float32.
    """
    out, ref = _door(_real_file("Q8_0"))
    print("    gguf: Qwen3-0.6B-Q8_0: " + _door_checks(out, ref))


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items()) if name.startswith("test_")]
    try:
        failures = _skip.run_tests(items, suite=_SUITE)
    finally:
        if os.path.isdir(_TMP):
            shutil.rmtree(_TMP)
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
