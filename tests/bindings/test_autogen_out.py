"""Every `autogen:` `.out` variant of `native_functions.yaml` is in its packet, with upstream's schema (issue #423).

Run 44973431737: executorch's `ToOutVarPass`, lowering `x * 1.5` and `y - 2` on the
shim, stopped with "Missing out variants: {'aten::mul', 'aten::sub',
'aten::expand_copy'}". It finds an edge op's `.out` by walking
`torch._C._jit_get_operation(name)[1]` and reading each overload's `_schema`
(`exir/dialects/edge/op/api.py::to_variant`). `mul.Scalar_out`, `sub.Scalar_out`
and `expand_copy.out` are not `- func:` lines of the yaml: an entry's `autogen:`
line names them, and upstream's build generates them with torchgen. The shim's
packets listed only the `- func:` overloads, plus a hand-transcribed few.

Now the yaml scan records every `autogen:` `out` / `*_out` name with the entry that
names it, lists it in its packet, and derives its schema with torchgen's own
`self_to_out_signature` / `functional_to_out_signature` /
`mutable_to_out_signature` when first asked (`bootstrap.py::_autogen_out_text`).

**What makes each test able to fail.**

* `test_every_autogen_out_is_listed_with_upstreams_schema`: this file scans the yaml
  itself (not through the shim) for the names, asks the shim for each packet's
  overload list and each schema's text, and asks upstream torch, in a separate
  process, for the same. A name missing from a packet, a placeholder, or one
  character of difference from upstream's text fails, naming the overload. At
  least 400 names must be compared, so a scan that found nothing cannot pass.
* `test_the_three_exir_needed_are_there`: the three of run 44973431737, by name.

**What this cannot find.** Whether a kernel runs any of them: the shim has no `.out`
kernels, and calling one refuses by name in the one door. Only the lookup is
needed by exir, which never runs the graph it converts.
"""

import json
import os
import pathlib
import subprocess
import sys

_REPO_ROOT = next(p for p in pathlib.Path(__file__).resolve().parents if (p / "AGENTS.md").is_file())
_VENDOR_DIR = str(_REPO_ROOT / "torchnative" / "python")
_SUPPORT_DIR = str(_REPO_ROOT / "tests" / "_support")
if _SUPPORT_DIR not in sys.path:
    sys.path.insert(0, _SUPPORT_DIR)
_YAML = pathlib.Path(_VENDOR_DIR) / "torchgen" / "packaged" / "ATen" / "native" / "native_functions.yaml"

import _skip  # noqa: E402

_MARK = "AUTOGEN_OUT_JSON "
_cache = {}

#: Asked of a torch, upstream or the shim, in its own process.
_ASK = r'''
import json, sys
import torch
keys = json.loads(sys.stdin.read())
out = {"shim": hasattr(torch._C, "_aten_implemented"), "answers": {}}
for name, overload in keys:
    entry = {}
    try:
        entry["listed"] = overload in torch._C._jit_get_operation("aten::" + name)[1]
    except Exception as exc:
        entry["listed"] = "error: %s: %s" % (type(exc).__name__, exc)
    try:
        schema = getattr(getattr(torch.ops.aten, name), overload)._schema
        entry["text"] = str(schema)
        entry["placeholder"] = bool(getattr(schema, "is_placeholder", False))
    except Exception as exc:
        entry["text"] = "error: %s: %s" % (type(exc).__name__, exc)
    out["answers"]["%s.%s" % (name, overload)] = entry
print("AUTOGEN_OUT_JSON " + json.dumps(out))
'''


def _autogen_out_keys():
    """(name, overload) for every `autogen:` `out` / `*_out` name, read here."""
    keys = []
    for line in _YAML.read_text(encoding="utf-8").splitlines():
        if not line.startswith("  autogen:"):
            continue
        for generated in line[len("  autogen:"):].split(","):
            name, _, overload = generated.strip().partition(".")
            if name and (overload == "out" or overload.endswith("_out")):
                keys.append((name, overload))
    return sorted(set(keys))


def _ask(shim, keys):
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run([sys.executable, "-c", _ASK], input=json.dumps(keys), capture_output=True,
                          text=True, env=env, timeout=600)
    line = next((ln for ln in reversed(proc.stdout.splitlines()) if ln.startswith(_MARK)), None)
    if proc.returncode != 0 or line is None:
        raise AssertionError(f"{'shim' if shim else 'upstream'} exited {proc.returncode}:\n"
                             + "\n".join(proc.stderr.splitlines()[-20:]))
    rec = json.loads(line[len(_MARK):])
    assert rec["shim"] == shim, f"the {'shim' if shim else 'upstream'} side imported the wrong torch"
    return rec["answers"]


def _answers():
    if not _YAML.is_file() or not os.path.isfile(os.path.join(_VENDOR_DIR, "torch", "__init__.py")):
        _skip.skip("no vendored torch tree (native_functions.yaml comes with it)")
        return None
    if "a" not in _cache:
        keys = _autogen_out_keys()
        _cache["a"] = (keys, _ask(True, keys), _ask(False, keys))
    return _cache["a"]


def test_every_autogen_out_is_listed_with_upstreams_schema():
    got = _answers()
    if got is None:
        return
    keys, shim, up = got
    assert len(keys) >= 400, f"the yaml scan found {len(keys)} autogen out names"
    bad = []
    compared = 0
    for name, overload in keys:
        k = f"{name}.{overload}"
        u, s = up[k], shim[k]
        if u["listed"] is not True:
            continue  # upstream has no such overload in its packet: nothing to match
        compared += 1
        if s["listed"] is not True or s.get("placeholder") or s["text"] != u["text"]:
            bad.append({"overload": k, "upstream": u["text"], "shim": s["text"],
                        "listed": s["listed"], "placeholder": s.get("placeholder")})
    assert compared >= 400, f"only {compared} autogen out names are in upstream's packets"
    assert not bad, f"{len(bad)} of {compared} differ, first: " + json.dumps(bad[:5], indent=1)
    print(f"ok   autogen_out: {compared} autogen .out overloads listed with upstream's schema text")


def test_the_three_exir_needed_are_there():
    got = _answers()
    if got is None:
        return
    _, shim, up = got
    for k in ("mul.Scalar_out", "sub.Scalar_out", "expand_copy.out"):
        assert shim[k]["listed"] is True and shim[k]["text"] == up[k]["text"], (k, shim[k], up[k])
    print("ok   autogen_out: mul.Scalar_out, sub.Scalar_out, expand_copy.out are listed as upstream lists them")


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
                print(f"SKIP test_autogen_out: {name} -- {_skip._state['skipped'][name]}")
        _skip._state["current"] = None
    raise SystemExit(1 if failures else 0)
