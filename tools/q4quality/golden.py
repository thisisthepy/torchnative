#!/usr/bin/env python3
"""Freeze llama.cpp's own answers for a handful of GGML blocks, so the gate can
check candle against them without having llama.cpp installed.

    # write (needs gguf-py; see measure.py's docstring for where it comes from)
    python tools/q4quality/golden.py --gguf-py <dir-or-whl> --write
    # re-derive and compare with the committed file; exit 1 on any difference
    python tools/q4quality/golden.py --gguf-py <dir-or-whl> --check

**Why a frozen file and not a live import.** The gate interpreter does not
have `gguf` and must not have it installed (AGENTS.md §15.2: the pinned venv
is never installed into). A literal answer is also the strongest form of the
check: `rust/torch_c/pytests/test_q4ref.py` compares candle's bytes and bits
with numbers that **neither candle nor this repository's `ggml_ref.py`
produced**. `ggml_ref.py` is a transcription by the same hands that read
candle, and it says so in its own docstring; this file is llama.cpp's.

**What the file can go wrong by.** It is generated once and then trusted, so a
stale or hand-edited copy is the risk. Three things stand against that: the
provenance block records the sha256 of the exact `gguf/quants.py` that
produced it, `--check` re-derives every entry and fails on one differing
byte, and the gate test re-checks the *coverage* properties each case was
built for (a zero block really has `d == 0`, the k-quant blob really sets the
high scale bits), so trimming a case to make a failure go away fails too.

**What the cases are for.** Each one is the input that would expose one
specific misreading of the format, named after it:

  q4_0  gauss          ordinary weights at a weight-like scale
  q4_0  pos_extreme    largest |x| is positive -> `d = max/-8` is negative
  q4_0  neg_extreme    largest |x| is negative -> `d` is positive
  q4_0  zero           `d = 0`, `id = 0`, every nibble 8
  q4_0  opp_sign       +a first and -a later: the first one sets the sign,
                       and -a lands at +8 steps, 16.5 -> clamped to 15
  q4_0  subnormal_d    `d` below f16's smallest normal
  q4_0  two_rows       2 x 64: block order across a row boundary
  q8_0  gauss, ties    `roundf` half away from zero on exact .5 (llama.cpp's
                       `_ref`; gguf-py's `np_roundf` reproduces it)
  q4_k  foreign        a blob **nobody's writer produced**: every scale and
                       min byte drawn at random, so all eight 6-bit pairs
                       and the split high bits of sub-blocks 4..7 are live

`q4_k` has no writer case because gguf-py has no k-quant writer.
"""

import argparse
import hashlib
import json
import math
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gguf_ref  # noqa: E402

GOLDEN = os.path.join(HERE, "golden_gguf_py.json")
SEED = 20261003


def _lcg(seed):
    state = seed & 0xFFFFFFFF

    def nxt():
        nonlocal state
        state = (1103515245 * state + 12345) & 0x7FFFFFFF
        return state / 0x7FFFFFFF

    return nxt


def _gauss(n, scale, seed):
    nxt = _lcg(seed)
    out = []
    while len(out) < n:
        u1 = max(nxt(), 1e-12)
        u2 = nxt()
        r = math.sqrt(-2.0 * math.log(u1))
        out += [r * math.cos(2 * math.pi * u2), r * math.sin(2 * math.pi * u2)]
    return np.array(out[:n], dtype=np.float32) * np.float32(scale)


def _cases():
    """`[(fmt, name, shape, float32 array or blob bytes)]`."""
    g = _gauss(32, 0.02, SEED)
    pos = g.copy()
    pos[5] = np.float32(0.1)
    neg = g.copy()
    neg[27] = np.float32(-0.1)
    opp = (g * np.float32(0.5)).astype(np.float32)
    opp[3] = np.float32(0.05)
    opp[20] = np.float32(-0.05)
    tiny = _gauss(32, 1e-6, SEED + 1)
    rows = _gauss(128, 0.03, SEED + 2).reshape(2, 64)

    ties = np.zeros(32, dtype=np.float32)
    ties[0] = 127.0  # absmax 127 -> d = 1 exactly, id = 1
    ties[1:9] = [2.5, -2.5, 0.5, -0.5, 1.5, -1.5, 126.5, -126.5]
    ties[9:] = _gauss(23, 30.0, SEED + 3)

    nxt = _lcg(SEED + 4)
    f16_normals = [0.0123, 0.0045, 0.25, 0.001, 0.0371, 0.5]
    blob = bytearray()
    for i in range(3):
        blob += struct.pack("<e", f16_normals[(2 * i) % len(f16_normals)])
        blob += struct.pack("<e", f16_normals[(2 * i + 1) % len(f16_normals)])
        blob += bytes(int(nxt() * 256) & 0xFF for _ in range(12 + 128))

    return [
        ("q4_0", "gauss", (32,), g),
        ("q4_0", "pos_extreme", (32,), pos),
        ("q4_0", "neg_extreme", (32,), neg),
        ("q4_0", "zero", (32,), np.zeros(32, dtype=np.float32)),
        ("q4_0", "opp_sign", (32,), opp),
        ("q4_0", "subnormal_d", (32,), tiny),
        ("q4_0", "two_rows", (2, 64), rows),
        ("q8_0", "gauss", (64,), _gauss(64, 0.5, SEED + 5)),
        ("q8_0", "ties", (32,), ties),
        ("q4_k", "foreign", (3 * 256,), bytes(blob)),
    ]


def _hex_f32(a):
    return np.ascontiguousarray(a, dtype="<f4").tobytes().hex()


def build(gq):
    entries = []
    for fmt, name, shape, data in _cases():
        n = int(np.prod(shape))
        e = {"format": fmt, "name": name, "shape": list(shape)}
        if isinstance(data, bytes):
            e["blob"] = data.hex()
        else:
            e["input_f32"] = _hex_f32(data)
            e["blob"] = gguf_ref.gguf_quantize(gq, fmt, data.reshape(shape)).hex()
        e["dequant_f32"] = _hex_f32(gguf_ref.gguf_dequantize(gq, fmt, bytes.fromhex(e["blob"]), n))
        entries.append(e)
    return entries


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--gguf-py", default=None, help="dir containing gguf/, or a gguf-*.whl")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    gq, prov = gguf_ref.load_gguf_quants(args.gguf_py)
    entries = build(gq)
    doc = {
        "what": "llama.cpp gguf-py answers for GGML blocks; checked by rust/torch_c/pytests/test_q4ref.py",
        "generator": "tools/q4quality/golden.py",
        "provenance": {
            "gguf_quants_sha256": prov["sha256"],
            "llama_cpp_commit_read": gguf_ref.LLAMACPP_COMMIT,
            "numpy": np.__version__,
        },
        "cases": entries,
    }
    if args.write:
        with open(GOLDEN, "w") as fh:
            json.dump(doc, fh, indent=1, sort_keys=True)
            fh.write("\n")
        print(f"wrote {GOLDEN}: {len(entries)} cases, gguf quants.py sha256 {prov['sha256']}")
        return 0

    with open(GOLDEN) as fh:
        have = json.load(fh)
    bad = []
    if have["provenance"]["gguf_quants_sha256"] != prov["sha256"]:
        # Not itself a failure: a newer gguf-py may give the same bytes. The
        # entries decide; this line says the comparison is across versions.
        print(f"note: committed file came from quants.py {have['provenance']['gguf_quants_sha256']}, "
              f"this one is {prov['sha256']}")
    old = {(c["format"], c["name"]): c for c in have["cases"]}
    for e in entries:
        k = (e["format"], e["name"])
        if k not in old:
            bad.append(f"{k}: missing from committed file")
            continue
        for field in ("blob", "dequant_f32", "input_f32"):
            if e.get(field) != old[k].get(field):
                bad.append(f"{k}: {field} differs")
    for k in old:
        if k not in {(e["format"], e["name"]) for e in entries}:
            bad.append(f"{k}: in committed file but not generated")
    for line in bad:
        print("DIFF", line)
    print(f"checked {len(entries)} cases against {GOLDEN}: {'OK' if not bad else f'{len(bad)} differences'}")
    print("sha256(golden file) =", hashlib.sha256(open(GOLDEN, "rb").read()).hexdigest())
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
