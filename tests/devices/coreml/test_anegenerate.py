"""`from_pretrained` -> `to(device.npu)` -> `generate` on the Apple Neural Engine.

The path a caller takes, in one process with the shim as `torch`: SmolLM2-135M
loads with the real `transformers`, `to(torchnative.device.npu)` replaces its
decoder layers with grouped CoreML programs (ANEDECODE3: two layers per program
clear the ~4.7M weight threshold; the KV rows live in the `transformers` Cache
and are concatenated inside the program), and the real `generate` runs greedy
decoding to the end.

What is asserted, and graded against upstream torch in a separate subprocess:

* every decoder layer is inside a group, and every group's `MLComputePlan`
  prefers the Neural Engine for every computing op;
* the greedy tokens are upstream float32's until the first position where the
  two differ, and that position is a near tie on upstream's side: the token
  the Neural Engine chose is upstream's runner-up, within `_NEAR_TIE` logits.
  float16 cannot be held to float32's exact argmax at a 0.26-logit tie;
  upstream's own bfloat16 run leaves float32's tokens at the second step.

What this cannot find: whether the Neural Engine, rather than the CPU, executed
the program. `MLComputePlan` is CoreML's placement answer, not a measurement.
`docs/graph/ANEGEN.md` records the timing comparison and the `powermetrics`
command that measures it.

<!-- DOCWATCH: symbol-in-file tests/devices/coreml/test_anegenerate.py test_every_decoder_layer_is_in_a_neural_engine_group present -->
<!-- DOCWATCH: symbol-in-file tests/devices/coreml/test_anegenerate.py test_generate_follows_upstream_greedy_until_a_near_tie present -->
"""

import json
import os
import subprocess
import sys

import _skip
from test_shim import _CKPT_VENDOR_SHIM, _STDOUT_GUARD, _npu_fixture

_MODEL = "HuggingFaceTB/SmolLM2-135M"
_PROMPT = "The capital of France is"
_NEW = 20
#: The shortest agreeing prefix accepted before a near tie.
_MIN_PREFIX = 8
#: Upstream float32's top-1 minus top-2 logit at the first differing step.
_NEAR_TIE = 0.5

_SHIM_SCRIPT = r"""
@STDOUT_GUARD@
import json, warnings
warnings.simplefilter("ignore")
out = {}
try:
    import coremltools  # noqa: F401
except Exception as error:
    out["skip"] = f"coremltools unavailable: {type(error).__name__}: {error}"
    print(json.dumps(out), file=_stdout, flush=True)
    raise SystemExit(0)
import torch
assert hasattr(torch._C, "_aten_implemented")
import torchnative
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(@MODEL@)
m = AutoModelForCausalLM.from_pretrained(@MODEL@, dtype=torch.float32).eval()
m = m.to(torchnative.device.npu)
rep = m.torchnative_offload
out["groups"] = rep["decoder_groups"]
out["refused"] = rep["decoder_refused"]
out["plans"] = rep.get("decoder_plans", [])
out["layers"] = len(m.model.layers)
ids = tok(@PROMPT@, return_tensors="pt").input_ids
with torch.no_grad():
    g = m.generate(ids, max_new_tokens=@NEW@, do_sample=False)
out["tokens"] = [int(t) for t in g[0, ids.shape[1]:].tolist()]
print(json.dumps(out), file=_stdout, flush=True)
"""

_UPSTREAM_SCRIPT = r"""
import json, sys, warnings
warnings.simplefilter("ignore")
import torch
assert not hasattr(torch._C, "_aten_implemented")
from transformers import AutoModelForCausalLM, AutoTokenizer
tok = AutoTokenizer.from_pretrained(@MODEL@)
m = AutoModelForCausalLM.from_pretrained(@MODEL@, dtype=torch.float32).eval()
ids = tok(@PROMPT@, return_tensors="pt").input_ids
shim = json.loads(sys.argv[1])
with torch.no_grad():
    g = m.generate(ids, max_new_tokens=@NEW@, do_sample=False)[0, ids.shape[1]:].tolist()
    # upstream's logits along the shim's own tokens, to grade a divergence
    seq = torch.cat([ids, torch.tensor([shim])], 1)
    logits = m(seq).logits[0, ids.shape[1] - 1:].float()
top = logits.topk(2, dim=-1)
print(json.dumps({"tokens": g, "top2": top.indices.tolist(),
                  "margin": (top.values[:, 0] - top.values[:, 1]).tolist()}))
"""


def _fill(script):
    return (script.replace("@STDOUT_GUARD@", _STDOUT_GUARD)
            .replace("@MODEL@", repr(_MODEL)).replace("@PROMPT@", repr(_PROMPT))
            .replace("@NEW@", str(_NEW)))


_CACHE = {}


def _runs():
    if "shim" in _CACHE:
        return _CACHE["shim"], _CACHE["up"]
    if sys.platform != "darwin":
        _skip.skip("the Apple Neural Engine exists only on macOS")
    if not os.path.isfile(_CKPT_VENDOR_SHIM):
        _skip.skip("vendored tree has no _C.abi3.so")
    shim = _npu_fixture(_fill(_SHIM_SCRIPT))
    if "skip" in shim:
        _skip.skip(shim["skip"])
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    proc = subprocess.run(
        [sys.executable, "-c", _fill(_UPSTREAM_SCRIPT), json.dumps(shim["tokens"])],
        capture_output=True, text=True, env=env, timeout=900)
    assert proc.returncode == 0, proc.stderr[-2000:]
    up = json.loads(proc.stdout.strip().splitlines()[-1])
    _CACHE["shim"], _CACHE["up"] = shim, up
    return shim, up


def test_every_decoder_layer_is_in_a_neural_engine_group():
    shim, _ = _runs()
    assert shim["refused"] == [], shim["refused"]
    covered = []
    for path in shim["groups"]:
        first, last = path.rsplit(".", 1)[1].split("-")
        covered.extend(range(int(first), int(last) + 1))
    assert covered == list(range(shim["layers"])), (covered, shim["layers"])
    assert len(shim["plans"]) == len(shim["groups"])
    for plan in shim["plans"]:
        assert plan.get("preferred") == ["NeuralEngine"], plan


def test_generate_follows_upstream_greedy_until_a_near_tie():
    shim, up = _runs()
    ours, theirs = shim["tokens"], up["tokens"]
    assert len(ours) == _NEW, ours
    first = next((i for i, (a, b) in enumerate(zip(ours, theirs)) if a != b), None)
    print(f"ANEGENERATE agree_prefix={_NEW if first is None else first}/{_NEW}"
          + ("" if first is None else
             f" tie_margin={up['margin'][first]:.4f}"))
    if first is None:
        return
    assert first >= _MIN_PREFIX, (
        f"diverged at step {first}: ours {ours[:first + 1]} upstream {theirs[:first + 1]}")
    assert ours[first] == up["top2"][first][1], (
        f"step {first}: ours {ours[first]} is not upstream's runner-up {up['top2'][first]}")
    assert up["margin"][first] <= _NEAR_TIE, (
        f"step {first}: upstream's margin {up['margin'][first]:.4f} is not a near tie")


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except _skip.Skip as s:
            print(f"SKIP test_anegenerate: {name} -- {s}")
        except Exception as e:  # noqa: BLE001
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
