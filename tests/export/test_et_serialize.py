"""exir's serializer step in Rust, against exir's own, byte for byte (issue #423, step R5-S1).

Owner, 2026-11-21 (asked by the leader, option (c)): "Rust로 재구현": the serializer
step is re-implemented in Rust and only exir's passes are vendored, so no executorch
Python package goes into the wheel. The pieces: `torch._C._et_serialize_pte`
(`et_pte.rs`, every target), `torch._C._et_dataclass_json` (`et_json.rs`, every
target), `torch._C._et_serialize_xnnpack` (`et.rs`, where the runtime and its linked
flatbuffers parser are built), and `torchnative.export.et_serialize`, which keeps
exir's signatures and hooks them into exir.

**Grade (AGENTS section 16): agrees, byte for byte.** The specification is executorch
1.4.1's own output for the same input, computed in the same process.

How. `tests/_support/et_serialize_probe.py` runs in the interpreter named by
`TORCHNATIVE_EXIR_PYTHON` with the vendored tree on `PYTHONPATH` (exir on the shim,
issue #306), so exir's functions and the Rust ones see the same dataclasses. It wraps
exir's two serializers during eight lowerings and serialises every argument both
ways; serialises hand-built programs and XNNPACK graphs that reach every union arm
and option; feeds both inputs exir refuses; and lowers once more with the hook
installed and flatc unreachable.

**What makes each test able to fail.**

* `test_the_bindings_are_in_torch_c`: a shim without `_et_serialize_pte` or
  `_et_dataclass_json` (module not registered) fails everywhere; on macOS arm64,
  where the gate builds the runtime, a missing `_et_serialize_xnnpack` fails.
* `test_the_json_text_is_jsons_for_dataclasses_unions_and_scalars`: in this
  process, against a copy of exir's 15-line `_DataclassEncoder` and `json.dumps`:
  float repr, escapes, IntEnum, bytes, nested dataclasses, the `_type` key of a
  string-annotated Union and its absence for an `Optional`. Any formatting slip
  changes the text.
* `test_every_program_exir_serialised_while_lowering_is_reproduced`: one byte of
  difference in any `serialize_pte_binary` call of the lowerings (layout, vtable
  sharing, padding, header, segment order) fails and names the first offset. The
  portable and XNNPACK Linear lowerings must complete, so the check cannot pass on
  zero calls.
* `test_every_xnnpack_payload_and_its_json_text_is_reproduced`: the same for
  `serialize_xnnpack_binary` and for the JSON text exir hands flatc.
* `test_programs_reaching_every_union_arm_and_option_are_reproduced`: hand-built
  programs (every `KernelTypes` and `InstructionArguments` arm, inf, -inf, nan and
  -0.0 doubles, non-ASCII strings, stack traces, shared and empty delegate blobs,
  named data with alignments, mutable data, four alignment settings) and XNNPACK
  graphs (quantised values, `"inf"` strings in OutputMinMax, an empty graph).
* `test_what_exir_refuses_the_rust_serializer_refuses`: each case must be refused
  by exir (so it is a real refusal) and by the Rust code.
* `test_with_the_hook_a_lowering_runs_no_exir_serializer_and_no_flatc` and
  `test_the_hooked_program_loads_and_agrees_with_eager`: with the hook, the program
  equals the unhooked one, no frame of exir's two serializer functions ran, flatc
  was never started, and the program runs in `executorch_rt` within 1e-6 of eager.

**What this suite cannot find.** Inputs no lowering or hand-built case produces
(other partitioners' payloads, programs above 2 GB). The PTD (external data) files
exir writes beside a program: not serialised here, and not in the owner's scope of
an in-memory program. Android: the Rust code is the same, the run is on macOS arm64
and Linux.
"""

import dataclasses
import enum
import json
import os
import pathlib
import platform
import subprocess
import sys
import typing

_TESTS_DIR = pathlib.Path(__file__).resolve().parents[1]
_SUPPORT_DIR = _TESTS_DIR / "_support"
if str(_SUPPORT_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPPORT_DIR))

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_VENDOR_DIR = str(_REPO_ROOT / "torchnative" / "python")
_PROBE = str(_SUPPORT_DIR / "et_serialize_probe.py")
_MARK = "ET_SERIALIZE_JSON "

import _skip  # noqa: E402

_cache = {}


def _on_macos_arm64():
    return sys.platform == "darwin" and platform.machine() == "arm64"


def _torch():
    if not os.path.isfile(os.path.join(_VENDOR_DIR, "torch", "__init__.py")):
        return None
    if _VENDOR_DIR not in sys.path:
        sys.path.insert(0, _VENDOR_DIR)
    # The vendored tree has no lib/libtorch_global_deps; upstream's __init__
    # loads it unless TORCH_USE_RTLD_GLOBAL is set, as test_devicens.py sets it
    # (run 44973431737: every test here failed on that dlopen).
    os.environ.setdefault("TORCH_USE_RTLD_GLOBAL", "1")
    import torch

    assert hasattr(torch._C, "_aten_implemented"), "imported upstream torch, not the shim"
    return torch


def _exir_python():
    path = os.environ.get("TORCHNATIVE_EXIR_PYTHON")
    return path if path and os.path.isfile(path) else None


def _report():
    """The probe's report, or None after a skip."""
    if _torch() is None:
        _skip.skip("no vendored torch tree")
        return None
    if _exir_python() is None:
        _skip.skip("TORCHNATIVE_EXIR_PYTHON unset or not a file: no interpreter with executorch "
                   "(the gate builds one with executorch==1.4.1; issue #306)")
        return None
    if "report" not in _cache:
        scratch = _REPO_ROOT / ".scratch"
        scratch.mkdir(exist_ok=True)
        env = dict(os.environ)
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
        env["TMPDIR"] = str(scratch)
        proc = subprocess.run([_exir_python(), _PROBE], env=env, cwd=str(_SUPPORT_DIR),
                              capture_output=True, text=True, timeout=1800)
        line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith(_MARK)), None)
        if line is None:
            raise AssertionError(f"the probe exited {proc.returncode} without a report; stderr tail:\n"
                                 + "\n".join(proc.stderr.splitlines()[-25:]))
        _cache["report"] = json.loads(line[len(_MARK):])
    report = _cache["report"]
    assert "fatal" not in report, report.get("fatal")
    assert report["shim"], "the probe imported upstream torch, not the shim"
    assert report["pte_binding"], "the probe's torch._C has no _et_serialize_pte"
    return report


def _bad(entry):
    return not entry.get("equal") or "error" in entry


def test_the_bindings_are_in_torch_c():
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    assert hasattr(torch._C, "_et_serialize_pte")
    assert hasattr(torch._C, "_et_dataclass_json")
    if _on_macos_arm64():
        assert hasattr(torch._C, "_et_serialize_xnnpack"), torch._C._et_runtime_info()
    elif not hasattr(torch._C, "_et_serialize_xnnpack"):
        print("ok   et_serialize: no _et_serialize_xnnpack here: " + str(torch._C._et_runtime_info().get("reason")))


# exir/_serialize/_dataclass.py::_DataclassEncoder, executorch 1.4.1, verbatim logic.
class _DataclassEncoder(json.JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            props = {}
            for field in dataclasses.fields(o):
                props[field.name] = getattr(o, field.name)
                origin = typing.get_origin(typing.get_type_hints(type(o))[field.name])
                if isinstance(field.type, str) and origin == typing.Union:
                    props[f"{field.name}_type"] = type(getattr(o, field.name)).__name__
            return props
        if isinstance(o, bytes):
            return list(o)
        return super().default(o)


class _Kind(enum.IntEnum):
    A = 0
    B = 7


@dataclasses.dataclass
class _Leaf:
    x: float
    s: str


@dataclasses.dataclass
class _Other:
    k: _Kind
    raw: bytes


_Either = typing.Union[_Leaf, _Other]


@dataclasses.dataclass
class _Base:
    flag: bool


@dataclasses.dataclass
class _Root(_Base):
    choice: "_Either"
    maybe: typing.Optional[_Leaf]
    items: typing.List[_Leaf]
    pair: tuple
    nothing: typing.Optional[int] = None


def test_the_json_text_is_jsons_for_dataclasses_unions_and_scalars():
    torch = _torch()
    if torch is None:
        _skip.skip("no vendored torch tree")
        return
    floats = [0.0, -0.0, 1.0, 0.1, 1e-4, 1e-5, 1.5e-7, 1e15, 1e16, 1.2345e16, 3.4028234663852886e38,
              5e-324, float("inf"), float("-inf"), float("nan"), 2.0 ** 70, 1 / 3]
    objs = [
        _Root(flag=True, choice=_Leaf(0.5, "a\"b\\c\né\U0001F600\x01\x7f"), maybe=None,
              items=[_Leaf(x, "s") for x in floats], pair=(1, "two", None, False)),
        _Root(flag=False, choice=_Other(_Kind.B, b"\x00\xff\x10"), maybe=_Leaf(-2.5, ""), items=[],
              pair=(), nothing=2 ** 80),
        [1, 2.5, "x", None, True],
        {"k": [_Leaf(1e-300, "v")], "e": {}},
    ]
    for o in objs:
        want = json.dumps(o, cls=_DataclassEncoder)
        got = torch._C._et_dataclass_json(o)
        assert got == want, f"\nwant {want}\ngot  {got}"
    for bad in (bytearray(b"x"), object()):
        try:
            torch._C._et_dataclass_json(bad)
        except TypeError:
            continue
        raise AssertionError(f"{type(bad).__name__} was encoded; json.dumps refuses it")
    print(f"ok   et_serialize: JSON text equal for {len(objs)} objects")


def test_every_program_exir_serialised_while_lowering_is_reproduced():
    report = _report()
    if report is None:
        return
    lowered = report["lowered"]
    for required in ("portable_linear", "xnnpack_linear"):
        assert lowered[required]["ok"], (required, lowered[required])
    n = 0
    for name, low in lowered.items():
        if not low["ok"]:
            print(f"note et_serialize: lowering {name} stopped on the shim: {low.get('error')}")
        for c in low["calls"]:
            if c["kind"] != "pte":
                continue
            n += 1
            assert not _bad(c), (name, c)
    assert n >= 2, n
    print(f"ok   et_serialize: {n} serialize_pte_binary calls byte-identical across "
          f"{sum(1 for v in lowered.values() if v['ok'])} lowerings")


def test_every_xnnpack_payload_and_its_json_text_is_reproduced():
    report = _report()
    if report is None:
        return
    n = 0
    for name, low in report["lowered"].items():
        for c in low["calls"]:
            if c["kind"] != "xnnpack":
                continue
            n += 1
            assert "error" not in c, (name, c)
            assert c["json_equal"], (name, c)
            if report["xnnpack_binding"]:
                assert c["equal"], (name, c)
    assert n >= 2, n
    if not report["xnnpack_binding"]:
        _skip.skip(f"{n} JSON texts equal; payload bytes need _et_serialize_xnnpack (runtime builds only)")
        return
    print(f"ok   et_serialize: {n} XNNPACK payloads and JSON texts byte-identical")


def test_programs_reaching_every_union_arm_and_option_are_reproduced():
    report = _report()
    if report is None:
        return
    syn = report["synthetic"]
    for name, entry in syn.items():
        assert "upstream" not in entry, f"exir itself refused the hand-built case {name}: {entry}"
        if name.startswith("xnnpack_"):
            assert "error" not in entry and entry["json_equal"], (name, entry)
            if report["xnnpack_binding"]:
                assert entry["equal"], (name, entry)
        else:
            assert not _bad(entry), (name, entry)
    assert sum(1 for k in syn if not k.startswith("xnnpack_")) >= 7, sorted(syn)
    print(f"ok   et_serialize: {len(syn)} hand-built programs and graphs byte-identical")


def test_what_exir_refuses_the_rust_serializer_refuses():
    report = _report()
    if report is None:
        return
    for name, res in report["errors"].items():
        assert res["upstream"] is not None, f"{name}: exir accepted it, so it is not a refusal case"
        assert res["rust"] is not None, f"{name}: exir raised {res['upstream']}, the Rust serializer accepted it"
    print(f"ok   et_serialize: {len(report['errors'])} refusals shared")


def test_with_the_hook_a_lowering_runs_no_exir_serializer_and_no_flatc():
    report = _report()
    if report is None:
        return
    if not report["xnnpack_binding"]:
        _skip.skip("the hooked XNNPACK lowering needs _et_serialize_xnnpack (runtime builds only)")
        return
    h = report["hooked"]
    assert "error" not in h, h
    names = {(m, a) for m, a, _ in h["replaced"]}
    assert ("executorch.exir._serialize._serialize", "serialize_pte_binary") in names, h["replaced"]
    assert ("executorch.backends.xnnpack.xnnpack_preprocess", "serialize_xnnpack_binary") in names, h["replaced"]
    assert h["equal"], h
    assert h["exir_serializer_ran"] == [], h["exir_serializer_ran"]
    assert h["flatc_attempts"] == [], h["flatc_attempts"]
    print(f"ok   et_serialize: hooked lowering equals exir's ({h['rust_bytes']} bytes), no exir serializer, no flatc")


def test_the_hooked_program_loads_and_agrees_with_eager():
    report = _report()
    if report is None:
        return
    if not (report["xnnpack_binding"] and report["runtime"]):
        _skip.skip("needs the runtime (torch._C._et_load), built on macOS arm64 and Android only")
        return
    ran = report["hooked"].get("ran")
    assert ran and "error" not in ran, ran
    assert ran["max_abs_diff"] <= 1e-6, ran
    print(f"ok   et_serialize: hooked program ran in executorch_rt, max |diff| {ran['max_abs_diff']:.3g}")


#: How exir reads a constant's bytes (`backends/xnnpack/operators/node_visitor.py`:
#: through a temporary tensor's temporary storage; `exir/emit/_emitter.py`: through a
#: storage kept after its tensor is gone), asked of a torch in its own process.
_DATA_PTR_PROBE = r'''
import ctypes, gc, json, sys
import torch
assert hasattr(torch._C, "_aten_implemented") == (sys.argv[1] == "shim"), "the wrong torch"
def read(ptr, n):
    return bytes(ctypes.cast(ptr, ctypes.POINTER(ctypes.c_char * n)).contents).hex()
w = torch.arange(12, dtype=torch.float32).reshape(3, 4) / 8 - 0.5
out = {}
c = w.to(dtype=torch.float32).contiguous()
out["node_visitor"] = read(c.untyped_storage().data_ptr(), c.untyped_storage().nbytes())
out["tensor_data_ptr"] = read(c.data_ptr(), c.numel() * 4)
s = (torch.arange(6, dtype=torch.int64) * 3 - 7).untyped_storage()
gc.collect()
out["emitter_storage_outlives_tensor"] = read(s.data_ptr(), s.nbytes())
v = w[1:]
out["view_offset_bytes"] = v.data_ptr() - w.data_ptr()
out["view_storage_is_the_base"] = v.untyped_storage().data_ptr() == w.untyped_storage().data_ptr()
out["view_reads_its_rows"] = read(v.data_ptr(), v.numel() * 4)
print("DATA_PTR_JSON " + json.dumps(out))
'''


def _data_ptr_answers(shim):
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run([sys.executable, "-c", _DATA_PTR_PROBE, "shim" if shim else "upstream"],
                          capture_output=True, text=True, env=env, timeout=600)
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith("DATA_PTR_JSON ")), None)
    assert proc.returncode == 0 and line, (shim, proc.returncode, proc.stderr[-3000:])
    return json.loads(line[len("DATA_PTR_JSON "):])


def test_reading_through_data_ptr_gives_the_tensors_bytes_as_upstream():
    """exir reads every constant's bytes with ctypes through `data_ptr()`.

    Run 44980924096: the hooked XNNPACK program ran 7e11 away from eager and
    every XNNPACK key (the sha256 of a weight's bytes) differed between two
    lowerings of one model. `untyped_storage().data_ptr()` and
    `tensor.data_ptr()` answered the address of candle's `Storage` struct, an
    identity token, so exir hashed and serialized heap bytes. Asked of upstream
    and the shim, each in its own process: the bytes read through both
    addresses, through a temporary storage (node_visitor) and through a storage
    whose tensor is gone (the emitter), and a view's offset and shared base.
    Reverting either address to the struct's fails the first two rows; dropping
    the storage's keep-alive fails the third (it reads freed memory)."""
    if not os.path.isfile(os.path.join(_VENDOR_DIR, "torch", "__init__.py")):
        _skip.skip("no vendored torch tree")
        return
    shim = _data_ptr_answers(True)
    try:
        up = _data_ptr_answers(False)
    except AssertionError:
        up = None
    if up is not None:
        assert shim == up, {k: (up.get(k), shim.get(k)) for k in up if up.get(k) != shim.get(k)}
    else:
        # No upstream torch here: the values themselves are known.
        import struct
        want = (struct.pack("<12f", *[i / 8 - 0.5 for i in range(12)])).hex()
        assert shim["node_visitor"] == want and shim["tensor_data_ptr"] == want, shim
        assert shim["emitter_storage_outlives_tensor"] == struct.pack("<6q", *[i * 3 - 7 for i in range(6)]).hex()
        assert shim["view_offset_bytes"] == 16 and shim["view_storage_is_the_base"], shim
    print(f"ok   et_serialize: bytes read through data_ptr() are the tensor's "
          f"({'equal to upstream' if up else 'upstream not importable; checked against the values'})")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(list(globals().items())):
        if not name.startswith("test_") or not callable(fn):
            continue
        _skip._state["current"] = name
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(exc).__name__}: {exc}")
        else:
            if name in _skip._state["skipped"]:
                print(f"SKIP test_et_serialize: {name} -- {_skip._state['skipped'][name]}")
        _skip._state["current"] = None
    raise SystemExit(1 if failures else 0)
