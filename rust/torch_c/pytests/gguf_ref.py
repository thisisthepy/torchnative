"""The reference half of test_gguf.py. **Runs under a different interpreter.**

This file is executed by `.caches/gguf-venv/bin/python` -- a venv that holds
llama.cpp's own `gguf` package (0.17.1) and numpy, and **no torch** -- never by
the gate's interpreter, and nothing in the gate imports it. Two reasons for the
separation, both deliberate (docs/graph/GGUF.md §3):

  * AGENTS.md §15.2: nothing is installed into `.caches/spike-venv`. The `gguf`
    package lives in its own venv inside `.caches/`, and this script is the
    only thing that uses it.
  * AGENTS.md §16: *agrees* is measured in a separate subprocess, against an
    implementation that shares nothing with the one under test. `gguf`'s
    reader (`GGUFReader`) and dequantiser (`gguf.quants`) are numpy code
    maintained beside llama.cpp, written independently of candle and of
    `torchnative.gguf`. That independence is the point: ggml_ref.py's own
    docstring names "a GGUF file written by llama.cpp compared against
    `_quantized_blob`" as the check that would close what it cannot -- this is
    that check.

What it emits is digests and sizes, not tensors: the test compares sha256 of
the raw bytes and of the dequantised float32 bytes, so a 600 MB file crosses the
process boundary as a few kilobytes of JSON, and the comparison is equality of
bytes with no tolerance anywhere.

Commands (JSON on stdout):

    table                 GGML type ids and (block size, type size), from gguf
    file PATH             header, metadata digests, and per-tensor digests
    dequant               stdin: [{"type": NAME, "rows": R, "hex": BLOB}, ...]
                          stdout: [{"f32_hex": ...} | {"error": ...}, ...]
    tiny PATH             write a one-layer Qwen3 GGUF with gguf's GGUFWriter
"""

import hashlib
import json
import struct
import sys

import numpy as np
from gguf import GGML_QUANT_SIZES, GGMLQuantizationType, GGUFReader, GGUFValueType
from gguf.quants import dequantize

# Rows per dequantisation chunk. The embedding of Qwen3-0.6B is 151936 x 1024;
# dequantised whole it is 622 MB of float32 plus numpy's temporaries. Blocks
# never span rows, so a row-chunked digest is the digest of the whole tensor.
_CHUNK_ROWS = 4096

_SCALAR_NP = {
    GGUFValueType.UINT8, GGUFValueType.INT8, GGUFValueType.UINT16,
    GGUFValueType.INT16, GGUFValueType.UINT32, GGUFValueType.INT32,
    GGUFValueType.FLOAT32, GGUFValueType.BOOL, GGUFValueType.UINT64,
    GGUFValueType.INT64, GGUFValueType.FLOAT64,
}


def _kv_digest(field):
    """sha256 of a canonical spelling of one key/value pair.

    The spelling is the value's bytes as the file stores them: a scalar's
    little-endian bytes, a string's u64 length and UTF-8 bytes, an array's
    element type, count, and then its elements in that spelling. test_gguf.py
    builds the same spelling from what `torchnative.gguf.GGUFFile` parsed, so
    equal digests mean the two readers recovered the same values -- including
    the 151936 tokenizer strings, which is where an off-by-one in string
    handling would show.
    """
    h = hashlib.sha256()
    main = GGUFValueType(int(field.types[0]))
    h.update(struct.pack("<I", int(main)))
    # field.parts = [key_len, key_bytes, value_type, *value_parts]
    if main == GGUFValueType.ARRAY:
        itype = int(field.parts[3][0])
        count = int(field.parts[4][0])
        h.update(struct.pack("<IQ", itype, count))
        for idx in field.data:
            part = field.parts[idx]
            if itype == GGUFValueType.STRING:
                raw = bytes(part)
                h.update(struct.pack("<Q", len(raw)))
                h.update(raw)
            else:
                h.update(np.ascontiguousarray(part).tobytes())
    elif main == GGUFValueType.STRING:
        raw = bytes(field.parts[field.data[0]])
        h.update(struct.pack("<Q", len(raw)))
        h.update(raw)
    elif main in _SCALAR_NP:
        h.update(np.ascontiguousarray(field.parts[field.data[0]]).tobytes())
    else:
        raise ValueError(f"unhandled value type {main!r}")
    return h.hexdigest()


def _f32_digest(tensor):
    """sha256 of `gguf.quants.dequantize` over the whole tensor, as float32 LE."""
    data = tensor.data
    qtype = tensor.tensor_type
    h = hashlib.sha256()
    rows = data.reshape(-1, data.shape[-1]) if data.ndim > 1 else data.reshape(1, -1)
    for start in range(0, rows.shape[0], _CHUNK_ROWS):
        chunk = np.ascontiguousarray(rows[start : start + _CHUNK_ROWS])
        out = dequantize(chunk, qtype)
        h.update(np.ascontiguousarray(out, dtype="<f4").tobytes())
    return h.hexdigest()


def cmd_table():
    out = {}
    for t in GGMLQuantizationType:
        block, size = GGML_QUANT_SIZES[t]
        out[t.name] = {"id": int(t), "block": int(block), "size": int(size)}
    return out


def cmd_file(path):
    r = GGUFReader(path)
    kv = {}
    for key, field in r.fields.items():
        if key.startswith("GGUF."):
            continue  # gguf-py's synthetic header fields, not file metadata
        kv[key] = _kv_digest(field)
    tensors = []
    for t in r.tensors:
        raw = np.ascontiguousarray(t.data).view(np.uint8)
        entry = {
            "name": t.name,
            "type": t.tensor_type.name,
            "type_id": int(t.tensor_type),
            "shape": [int(d) for d in reversed(t.shape.tolist())],
            "offset": int(t.data_offset),
            "nbytes": int(t.n_bytes),
            "raw_sha256": hashlib.sha256(raw.tobytes()).hexdigest(),
        }
        try:
            entry["f32_sha256"] = _f32_digest(t)
        except NotImplementedError as exc:
            entry["f32_sha256"] = None
            entry["f32_error"] = str(exc)
        tensors.append(entry)
    version = int(r.fields["GGUF.version"].parts[-1][0])
    return {
        "version": version,
        "alignment": int(r.alignment),
        "data_offset": int(r.data_offset),
        "kv": kv,
        "tensors": tensors,
    }


def cmd_dequant(items):
    out = []
    for item in items:
        qtype = GGMLQuantizationType[item["type"]]
        blob = np.frombuffer(bytes.fromhex(item["hex"]), dtype=np.uint8)
        rows = int(item["rows"])
        try:
            got = dequantize(blob.reshape(rows, -1), qtype)
        except NotImplementedError as exc:
            out.append({"error": str(exc)})
            continue
        out.append({"f32_hex": np.ascontiguousarray(got, dtype="<f4").tobytes().hex()})
    return out


def _finite_random_blocks(rng, qtype, rows, cols, f16_offsets):
    """Random bytes for a block type gguf-py cannot quantise to (the k-quants),
    with every f16 field forced finite -- test_gguf.py explains why."""
    block, size = GGML_QUANT_SIZES[qtype]
    n_blocks = rows * cols // block
    blob = rng.integers(0, 256, size=(n_blocks, size), dtype=np.uint8)
    for off in f16_offsets:
        hi = blob[:, off + 1]
        bad = (hi & 0x7C) == 0x7C
        blob[bad, off + 1] = hi[bad] & 0xBF
    return blob.reshape(rows, -1)


def cmd_tiny(path):
    """Write a one-layer Qwen3 GGUF with **llama.cpp's own writer** (`GGUFWriter`).

    Small enough to load anywhere, and written by the reference package rather
    than by test_gguf.py's own writer, so the `from_pretrained` door is tested
    on a file neither of this repository's GGUF spellings produced. Nine
    tensor types: F32, F16, BF16, Q8_0, Q4_0, Q4_1, Q5_0, Q5_1, and Q4_K (random
    blocks: gguf-py has no k-quant quantiser).

    Shapes follow `Qwen3Config`'s defaults where GGUF does not carry the value:
    transformers' qwen3 mapping has no `attention.key_length`, so `head_dim`
    stays at its default of 128 and q/k/v are sized from it.
    """
    from gguf import GGUFWriter
    from gguf.quants import quantize

    hidden, heads, kv_heads, head_dim, ffn, vocab = 64, 2, 1, 128, 128, 96
    rng = np.random.default_rng(20261003)
    w = GGUFWriter(path, "qwen3")
    w.add_string("general.name", "torchnative-tiny-qwen3")
    w.add_uint32("qwen3.context_length", 128)
    w.add_uint32("qwen3.block_count", 1)
    w.add_uint32("qwen3.feed_forward_length", ffn)
    w.add_uint32("qwen3.embedding_length", hidden)
    w.add_uint32("qwen3.attention.head_count", heads)
    w.add_uint32("qwen3.attention.head_count_kv", kv_heads)
    w.add_uint32("qwen3.attention.key_length", head_dim)
    w.add_uint32("qwen3.attention.value_length", head_dim)
    w.add_float32("qwen3.attention.layer_norm_rms_epsilon", 1e-6)
    w.add_float32("qwen3.rope.freq_base", 1000000.0)
    w.add_uint32("qwen3.vocab_size", vocab)

    def weight(rows, cols):
        return rng.normal(0.0, 0.05, size=(rows, cols)).astype(np.float32)

    def norm(n):
        return rng.normal(1.0, 0.1, size=(n,)).astype(np.float32)

    def q(name, arr, qtype):
        w.add_tensor(name, quantize(arr, qtype), raw_dtype=qtype)

    T = GGMLQuantizationType
    q("token_embd.weight", weight(vocab, hidden), T.Q8_0)
    w.add_tensor("output_norm.weight", norm(hidden))
    w.add_tensor("blk.0.attn_norm.weight", norm(hidden))
    q("blk.0.attn_q.weight", weight(heads * head_dim, hidden), T.Q8_0)
    q("blk.0.attn_k.weight", weight(kv_heads * head_dim, hidden), T.Q4_0)
    q("blk.0.attn_v.weight", weight(kv_heads * head_dim, hidden), T.Q5_1)
    w.add_tensor(
        "blk.0.attn_output.weight",
        _finite_random_blocks(rng, T.Q4_K, hidden, heads * head_dim, (0, 2)),
        raw_dtype=T.Q4_K,
    )
    w.add_tensor("blk.0.attn_q_norm.weight", norm(head_dim).astype(np.float16))
    w.add_tensor("blk.0.attn_k_norm.weight", norm(head_dim))
    w.add_tensor("blk.0.ffn_norm.weight", norm(hidden))
    q("blk.0.ffn_gate.weight", weight(ffn, hidden), T.Q4_1)
    q("blk.0.ffn_up.weight", weight(ffn, hidden), T.Q5_0)
    q("blk.0.ffn_down.weight", weight(hidden, ffn), T.BF16)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    return {"path": path}


def main(argv):
    cmd = argv[1]
    if cmd == "table":
        result = cmd_table()
    elif cmd == "tiny":
        result = cmd_tiny(argv[2])
    elif cmd == "file":
        result = cmd_file(argv[2])
    elif cmd == "dequant":
        result = cmd_dequant(json.load(sys.stdin))
    else:
        raise SystemExit(f"unknown command {cmd!r}")
    json.dump(result, sys.stdout)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
