"""`UntypedStorage.from_buffer`, `ByteStorage._from_buffer` and `ByteTensor(storage)` (issue #229).

`torch.distributed`'s object helpers are upstream code and the shim runs them
unchanged. `_object_to_tensor` (distributed_c10d.py) makes exactly these calls:

    f = io.BytesIO(); _pickler(f).dump(obj)
    byte_storage = torch.ByteStorage._from_buffer(f.getvalue())
    byte_tensor  = torch.ByteTensor(byte_storage).to(device)
    local_size   = torch.LongTensor([byte_tensor.numel()]).to(device)

and `_tensor_to_object` reads them back with

    buf = tensor.cpu().numpy().tobytes()[:tensor_size]

`ByteStorage._from_buffer` is the vendored `TypedStorage._from_buffer`, which
calls `torch.UntypedStorage.from_buffer(buf, dtype=torch.uint8)`. Before this
round the shim had no `from_buffer` on its storage class, so the inherited
`_StorageBase.from_buffer` stub raised a bare `NotImplementedError` (measured
on the main checkout's shim, 2026-11-13) and every object list failed on the
sender.

Each case below runs in a child process against upstream and again against
the shim, and the two answers must be identical, so every expectation is
upstream's own output. The payload tests also check bytes in == bytes out
against the payload itself, so upstream is not the only oracle. The shim child
asserts it imported the shim, the upstream child that it did not.
"""

import json
import os
import subprocess
import sys

os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")

import _skip


_REPO_ROOT = str(next(p for p in __import__("pathlib").Path(__file__).resolve().parents
                      if (p / "AGENTS.md").is_file()))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "python")

_SCRIPT = r"""
import hashlib, io, json, pickle, sys
A = json.loads(sys.argv[1])
import torch
assert hasattr(torch._C, "_aten_implemented") == A["shim"], "wrong torch in this child"
from torch.distributed import distributed_c10d as c10d

out = {}

def refusal(f):
    try:
        f()
    except Exception as e:
        return [type(e).__name__, str(e)]
    return ["no error", ""]

def digest(b):
    b = bytes(b)
    return {"len": len(b), "sha256": hashlib.sha256(b).hexdigest(),
            "head": list(b[:8]), "tail": list(b[-8:])}

def storage_bytes(s):
    # Read back through the legacy constructor and numpy, the same two calls
    # `_object_to_tensor` / `_tensor_to_object` make, not through anything
    # the shim could answer from the storage's own bookkeeping.
    return torch.ByteTensor(s).numpy().tobytes()

def lcg(n, seed):
    # Deterministic bytes without numpy or random, identical in both children.
    out_ = bytearray(n)
    x = seed
    for i in range(n):
        x = (x * 1103515245 + 12345) & 0x7FFFFFFF
        out_[i] = (x >> 16) & 0xFF
    return bytes(out_)

NON_ASCII = {"ko": "한국어", "e": "été", "emoji": "\U0001F600",
             "nul": "a\x00b", "nested": [("ß", 1.5), {"中": None}]}
PAYLOADS = {
    "empty": b"",
    "one": b"\x00",
    "small": b"hello, storage",
    "all_bytes": bytes(range(256)),
    "large": lcg(3 * 1024 * 1024 + 7, 229),
    "pickle_non_ascii_p2": pickle.dumps(NON_ASCII, protocol=2),
    "pickle_non_ascii_p5": pickle.dumps(NON_ASCII, protocol=5),
    "pickle_large_str": pickle.dumps("é" * 200000 + "\U0001F600", protocol=4),
}

def roundtrip(p, from_buffer=None):
    s = torch.ByteStorage._from_buffer(p)
    got = storage_bytes(s)
    return {"digest": digest(got), "equal_to_payload": got == p,
            "nbytes": s.nbytes(), "dtype": str(s.dtype), "size": s.size(),
            "type": type(s).__name__, "untyped": type(s._untyped_storage).__name__}

for name, p in PAYLOADS.items():
    out["rt_" + name] = roundtrip(p)

# The exact upstream pair, with an object that pickles to non-ASCII bytes.
OBJECTS = {"non_ascii": NON_ASCII, "empty_list": [], "none": None,
           "bytes": b"\xff\x00\xfe", "big": "x" * 500000 + "é"}
for name, obj in OBJECTS.items():
    def pair(obj=obj):
        t, size = c10d._object_to_tensor(obj, "cpu", None)
        back = c10d._tensor_to_object(t, size, None)
        return {"dtype": str(t.dtype), "shape": list(t.shape),
                "size": size.tolist(), "size_dtype": str(size.dtype),
                "repr_equal": repr(back) == repr(obj), "back": repr(back)[:200],
                "digest": digest(t.numpy().tobytes())}
    try:
        out["pair_" + name] = pair()
    except Exception as e:
        out["pair_" + name] = ["raised", type(e).__name__, str(e)]

# UntypedStorage.from_buffer itself, read back byte by byte.
def ubytes(*args, **kwargs):
    s = torch.UntypedStorage.from_buffer(*args, **kwargs)
    return {"type": type(s).__name__, "nbytes": s.nbytes(),
            "bytes": [s[i] for i in range(s.nbytes())]}

B8 = bytes(range(1, 9))
B16 = bytes(range(16))
ORDER_CASES = {
    "u8": ((B8,), {"dtype": torch.uint8}),
    "u8_bytearray": ((bytearray(B8),), {"dtype": torch.uint8}),
    "u8_memoryview": ((memoryview(B8),), {"dtype": torch.uint8}),
    "u8_kw_buffer": ((), {"buffer": B8, "dtype": torch.uint8}),
    "u8_big_ignored": ((B8, "big"), {"dtype": torch.uint8}),
    "i8": ((b"\xff\x01",), {"dtype": torch.int8}),
    "i16_big": ((B8, "big"), {"dtype": torch.int16}),
    "i16_little": ((B8, "little"), {"dtype": torch.int16}),
    "i16_native": ((B8, "native"), {"dtype": torch.int16}),
    "i16_kw_order": ((B8,), {"byte_order": "big", "dtype": torch.int16}),
    "i32_big": ((B8, "big"), {"dtype": torch.int32}),
    "i64_big": ((B8, "big"), {"dtype": torch.int64}),
    "f16_big": ((b"\x3c\x00", "big"), {"dtype": torch.float16}),
    "bf16_big": ((b"\x3f\x80", "big"), {"dtype": torch.bfloat16}),
    "f32_big": ((b"\x3f\x80\x00\x00", "big"), {"dtype": torch.float32}),
    "f64_big": ((B8, "big"), {"dtype": torch.float64}),
    "c64_big": ((B8, "big"), {"dtype": torch.complex64}),
    "c128_big": ((B16, "big"), {"dtype": torch.complex128}),
    "bool_big": ((b"\x00\x01\x02\xff", "big"), {"dtype": torch.bool}),
    "bool_native": ((b"\x00\x05", "native"), {"dtype": torch.bool}),
    "count_offset_u8": ((b"abcdef",), {"count": 2, "offset": 1, "dtype": torch.uint8}),
    "count_offset_i16": ((B8, "big"), {"count": 2, "offset": 1, "dtype": torch.int16}),
    "offset_rest_i16": ((B8, "big"), {"offset": 2, "dtype": torch.int16}),
    "positional_five": ((B8, "little", 1, 4, torch.int32), {}),
    "count_zero": ((B8,), {"count": 0, "dtype": torch.uint8}),
    "count_negative": ((B8,), {"count": -2, "dtype": torch.uint8}),
    "offset_at_end": ((B8,), {"offset": 8, "dtype": torch.uint8}),
    "empty": ((b"",), {"dtype": torch.uint8}),
}
for name, (args, kwargs) in ORDER_CASES.items():
    try:
        out["order_" + name] = ubytes(*args, **kwargs)
    except Exception as e:
        out["order_" + name] = ["raised", type(e).__name__, str(e)]

# The bytes are copied at the call, as upstream copies them.
ba = bytearray(b"abc")
s = torch.UntypedStorage.from_buffer(ba, dtype=torch.uint8)
ba[0] = 0
out["copied_at_call"] = [s[i] for i in range(3)]

# Typed storages, read back through their own legacy tensor class.
TYPED = {
    "short_big": lambda: torch.ShortTensor(torch.ShortStorage._from_buffer(B8, "big")),
    "int_big": lambda: torch.IntTensor(torch.IntStorage._from_buffer(B8, "big")),
    "long_big": lambda: torch.LongTensor(torch.LongStorage._from_buffer(B8, "big")),
    "float_little": lambda: torch.FloatTensor(torch.FloatStorage._from_buffer(b"\x00\x00\x80\x3f", "little")),
    "double_big": lambda: torch.DoubleTensor(torch.DoubleStorage._from_buffer(b"\x3f\xf0" + bytes(6), "big")),
    "half_big": lambda: torch.HalfTensor(torch.HalfStorage._from_buffer(b"\x3c\x00", "big")),
    "bf16_big": lambda: torch.BFloat16Tensor(torch.BFloat16Storage._from_buffer(b"\x3f\x80", "big")),
    "char": lambda: torch.CharTensor(torch.CharStorage._from_buffer(b"\xff\x01")),
    "bool_big": lambda: torch.BoolTensor(torch.BoolStorage._from_buffer(b"\x00\x02", "big")),
    "byte_empty": lambda: torch.ByteTensor(torch.ByteStorage._from_buffer(b"")),
    "typed_from_buffer": lambda: torch.ShortTensor(torch.TypedStorage.from_buffer(B8, "big", dtype=torch.int16)),
    "short_of_untyped_3": lambda: torch.ShortTensor(torch.UntypedStorage.from_buffer(b"\x01\x02\x03", dtype=torch.uint8)),
    "byte_of_untyped": lambda: torch.ByteTensor(torch.UntypedStorage.from_buffer(B8, dtype=torch.uint8)),
    "byte_storage_kw": lambda: torch.ByteTensor(storage=torch.ByteStorage._from_buffer(B8)),
    "byte_device_cpu": lambda: torch.ByteTensor(torch.ByteStorage._from_buffer(B8), device="cpu"),
}
for name, f in TYPED.items():
    try:
        t = f()
        out["typed_" + name] = {"dtype": str(t.dtype), "shape": list(t.shape),
                                "values": t.tolist(), "type": type(t).__name__}
    except Exception as e:
        out["typed_" + name] = ["raised", type(e).__name__, str(e)]

# Refusals with upstream's own wording.
REFUSE = {
    "no_dtype": lambda: torch.UntypedStorage.from_buffer(B8),
    "dtype_not_dtype": lambda: torch.UntypedStorage.from_buffer(B8, dtype="uint8"),
    "missing_byte_order": lambda: torch.UntypedStorage.from_buffer(B8, dtype=torch.int16),
    "missing_byte_order_bool": lambda: torch.UntypedStorage.from_buffer(B8, dtype=torch.bool),
    "byte_order_none": lambda: torch.UntypedStorage.from_buffer(B8, None, dtype=torch.int16),
    "bad_byte_order": lambda: torch.UntypedStorage.from_buffer(B8, "middle", dtype=torch.int16),
    "not_a_multiple": lambda: torch.UntypedStorage.from_buffer(b"\x01\x02\x03", "big", dtype=torch.int16),
    "offset_not_a_multiple": lambda: torch.UntypedStorage.from_buffer(B8, "big", offset=1, dtype=torch.int16),
    "count_too_big": lambda: torch.UntypedStorage.from_buffer(b"abc", count=5, dtype=torch.uint8),
    "offset_negative": lambda: torch.UntypedStorage.from_buffer(b"abc", offset=-1, dtype=torch.uint8),
    "offset_past_end": lambda: torch.UntypedStorage.from_buffer(B8, offset=9, dtype=torch.uint8),
    "str_buffer": lambda: torch.UntypedStorage.from_buffer("abc", dtype=torch.uint8),
    "list_buffer": lambda: torch.UntypedStorage.from_buffer([1, 2], dtype=torch.uint8),
    "non_contiguous": lambda: torch.UntypedStorage.from_buffer(memoryview(B8)[::2], dtype=torch.uint8),
    "uint32": lambda: torch.UntypedStorage.from_buffer(B8, "big", dtype=torch.uint32),
    "qint8": lambda: torch.UntypedStorage.from_buffer(B8, "big", dtype=torch.qint8),
    "legacy_with_dtype": lambda: torch.ByteStorage._from_buffer(B8, dtype=torch.uint8),
    "legacy_with_device": lambda: torch.ByteStorage._from_buffer(B8, device="cpu"),
    "typed_meta": lambda: torch.TypedStorage.from_buffer(B8, dtype=torch.uint8, device="meta"),
    "byte_of_bool_storage": lambda: torch.ByteTensor(torch.BoolStorage._from_buffer(b"\x01", "big")),
    "long_of_byte_storage": lambda: torch.LongTensor(torch.ByteStorage._from_buffer(B8)),
    "double_of_float_storage": lambda: torch.DoubleTensor(torch.FloatStorage._from_buffer(B8, "big")),
    "byte_device_meta": lambda: torch.ByteTensor(torch.ByteStorage._from_buffer(B8), device="meta"),
}
for name, f in REFUSE.items():
    out["refuse_" + name] = refusal(f)

# The nullification, run in the shim child only: `from_buffer` replaced by one
# that delivers zeros of the right length. The payload digests it produces
# must differ from upstream's, or the round-trip comparison above could not
# have failed.
if A["shim"]:
    real = torch._C.StorageBase.__dict__["from_buffer"]
    def zeros_from_buffer(cls, buffer, *args, **kwargs):
        n = memoryview(buffer).nbytes
        return real.__get__(None, cls)(bytes(n), *args, **kwargs)
    torch._C.StorageBase.from_buffer = classmethod(zeros_from_buffer)
    try:
        out["mutant_zeros"] = {name: roundtrip(p) for name, p in PAYLOADS.items()}
    finally:
        torch._C.StorageBase.from_buffer = real
    out["restored"] = roundtrip(PAYLOADS["small"])
    # What the shim says about the dtypes it refuses that upstream accepts.
    out["shim_refuse_float8"] = refusal(
        lambda: torch.UntypedStorage.from_buffer(B8, "big", dtype=torch.float8_e5m2))
print(json.dumps(out))
"""


def _run(upstream):
    env = dict(os.environ)
    env.pop("TORCH_C_ARTEFACT", None)
    if upstream:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    else:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    proc = subprocess.run([sys.executable, "-c", _SCRIPT, json.dumps({"shim": not upstream})],
                          capture_output=True, text=True, timeout=900, env=env,
                          cwd=os.path.dirname(os.path.abspath(__file__)))
    lines = proc.stdout.strip().splitlines()
    assert lines, "child produced no JSON (rc=%d):\n%s" % (proc.returncode, proc.stderr[-3000:])
    return json.loads(lines[-1])


_CACHE = {}


def _both(what):
    if not os.path.isfile(os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")):
        _skip.skip("   (skipped %s: no vendored shim at %s -- run scripts/vendor/install_shim.sh)"
                   % (what, _VENDOR_DIR))
        return None
    if not _CACHE:
        _CACHE["up"] = _run(True)
        _CACHE["shim"] = _run(False)
    return _CACHE["up"], _CACHE["shim"]


def _same(keys, what):
    both = _both(what)
    if both is None:
        return None
    up, shim = both
    for k in keys:
        assert k in up and k in shim, (k, sorted(up)[:5], sorted(shim)[:5])
        assert shim[k] == up[k], "%s differs from upstream:\n  shim:     %r\n  upstream: %r" % (
            k, shim[k], up[k])
    return both


_PAYLOADS = ("empty", "one", "small", "all_bytes", "large", "pickle_non_ascii_p2",
             "pickle_non_ascii_p5", "pickle_large_str")


def test_bytes_in_are_bytes_out_for_empty_small_large_and_non_ascii_pickles():
    """`ByteStorage._from_buffer(p)` then `ByteTensor(storage).numpy().tobytes()` is `p`, as upstream."""
    both = _same(["rt_" + n for n in _PAYLOADS], "from_buffer round trip")
    if both is None:
        return
    for side in both:
        for n in _PAYLOADS:
            assert side["rt_" + n]["equal_to_payload"] is True, (n, side["rt_" + n])
    # The large payload really is large, so a size-dependent path is exercised.
    assert both[1]["rt_large"]["nbytes"] == 3 * 1024 * 1024 + 7


def test_the_exact_upstream_object_to_tensor_pair_round_trips():
    """`_object_to_tensor` then `_tensor_to_object`, the helpers every object collective uses."""
    names = ("non_ascii", "empty_list", "none", "bytes", "big")
    both = _same(["pair_" + n for n in names], "_object_to_tensor pair")
    if both is None:
        return
    for n in names:
        got = both[1]["pair_" + n]
        assert isinstance(got, dict), (n, got)
        assert got["repr_equal"] is True and got["dtype"] == "torch.uint8", (n, got)
        assert got["size_dtype"] == "torch.int64", (n, got)


def test_byte_order_and_dtype_are_honoured():
    """Each element is byte-swapped per `byte_order`, complex per component, bool normalised to 0/1."""
    keys = [k for k in ("order_u8", "order_u8_bytearray", "order_u8_memoryview",
                        "order_u8_kw_buffer", "order_u8_big_ignored", "order_i8",
                        "order_i16_big", "order_i16_little", "order_i16_native",
                        "order_i16_kw_order", "order_i32_big", "order_i64_big",
                        "order_f16_big", "order_bf16_big", "order_f32_big",
                        "order_f64_big", "order_c64_big", "order_c128_big",
                        "order_bool_big", "order_bool_native")]
    both = _same(keys, "from_buffer byte order")
    if both is None:
        return
    shim = both[1]
    # Pinned independently of upstream, so a shared mistake cannot pass.
    assert shim["order_i16_big"]["bytes"] == [2, 1, 4, 3, 6, 5, 8, 7], shim["order_i16_big"]
    assert shim["order_c64_big"]["bytes"] == [4, 3, 2, 1, 8, 7, 6, 5], shim["order_c64_big"]
    assert shim["order_bool_big"]["bytes"] == [0, 1, 1, 1], shim["order_bool_big"]
    assert shim["order_u8"]["type"] == "UntypedStorage", shim["order_u8"]


def test_count_and_offset_select_the_same_bytes_as_upstream():
    _same(["order_" + k for k in ("count_offset_u8", "count_offset_i16", "offset_rest_i16",
                                  "positional_five", "count_zero", "count_negative",
                                  "offset_at_end", "empty")] + ["copied_at_call"],
          "from_buffer count and offset")


def test_legacy_tensor_constructor_takes_a_storage():
    """`ByteTensor(storage)` and its siblings, typed and untyped, as upstream answers them."""
    keys = ["typed_" + k for k in ("short_big", "int_big", "long_big", "float_little",
                                   "double_big", "half_big", "bf16_big", "char", "bool_big",
                                   "byte_empty", "typed_from_buffer", "short_of_untyped_3",
                                   "byte_of_untyped", "byte_storage_kw", "byte_device_cpu")]
    both = _same(keys, "legacy constructor from storage")
    if both is None:
        return
    for k in keys:
        assert isinstance(both[1][k], dict), (k, both[1][k])


def test_from_buffer_refuses_with_upstreams_words():
    """Every misuse raises upstream's exception type and text, never a guessed storage."""
    keys = ["refuse_" + k for k in (
        "no_dtype", "dtype_not_dtype", "missing_byte_order", "missing_byte_order_bool",
        "byte_order_none", "bad_byte_order", "not_a_multiple", "offset_not_a_multiple",
        "count_too_big", "offset_negative", "offset_past_end", "str_buffer", "list_buffer",
        "non_contiguous", "uint32", "qint8", "legacy_with_dtype", "legacy_with_device",
        "typed_meta", "byte_of_bool_storage", "long_of_byte_storage",
        "double_of_float_storage", "byte_device_meta")]
    both = _same(keys, "from_buffer refusals")
    if both is None:
        return
    for k in keys:
        assert both[1][k][0] != "no error", (k, both[1][k])


def test_a_dtype_the_shim_does_not_build_refuses_by_name():
    """float8 is accepted upstream; the shim refuses it naming the dtype rather than guessing."""
    both = _both("from_buffer float8 refusal")
    if both is None:
        return
    kind, msg = both[1]["shim_refuse_float8"]
    assert kind == "NotImplementedError", (kind, msg)
    assert "from_buffer" in msg and "float8_e5m2" in msg, msg


def test_a_from_buffer_that_delivers_zeros_is_caught():
    """The nullification: zeros of the right length must turn the round-trip comparison red."""
    both = _both("from_buffer nullification")
    if both is None:
        return
    up, shim = both
    caught = [n for n in _PAYLOADS
              if shim["mutant_zeros"][n]["digest"] != up["rt_" + n]["digest"]]
    # Every payload except the two whose bytes are all zero already.
    assert sorted(caught) == sorted(n for n in _PAYLOADS if n not in ("empty", "one")), caught
    assert all(shim["mutant_zeros"][n]["equal_to_payload"] is False for n in caught), caught
    assert shim["restored"] == up["rt_small"], (shim["restored"], up["rt_small"])


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_")]
    failures = _skip.run_tests(items, suite="test_storagebuf")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
