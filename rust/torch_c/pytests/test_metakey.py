"""`_dispatch_has_kernel_for_dispatch_key(name, "Meta")` -- what it costs, measured.

`docs/graph/VARMEAN.md` §4.1 named this gap, rejected it as the cause of the
`native_layer_norm` export wall, and left behind one number: of upstream's
**2083** aten overloads, **408 answer `False`** to this question too. So
"agrees with upstream" and "answers `True`" are different targets and only the
first is correct. `docs/graph/STRUCTSEQ.md` found the real cause and left this
untouched on purpose. Neither round measured the gap itself.

This file is that measurement, kept as tests rather than as prose, and every
upstream answer in it is computed **in a separate subprocess** from the probe
text below -- never a table written beside the implementation.

What it establishes, and what each test would catch:

* The shim answers `False` to every name and **never raises**. That is not an
  oversight, it is the paired half of `_dispatch_has_kernel = True`:
  `docs/design/REGISTRATIONS.md` §3.3 measured that upstream's own
  `activate_meta()` dies with `RuntimeError: operator aten::ldexp does not
  exist` when the two are mixed. A future round that makes this predicate
  name-validating reddens `test_the_meta_predicate_answers_false_for_every_name_and_never_raises`.

* **The obvious repair is measurably wrong.** Deriving the answer from what
  this shim implements -- the `meta_dispatch` arms, which is where
  `_aten_implemented()`'s knowledge of meta support lives -- would answer
  `True` for twenty-eight ops upstream answers `False` for, because upstream
  reaches them through `CompositeExplicitAutograd` (the views and copies:
  `clone`, `detach`, `expand`, `t`, `slice.Tensor`, `_to_copy`, ...) or
  `CompositeImplicitAutograd` (`matmul`, `var_mean.default`). Those are not
  Meta-key registrations and upstream says so. That is the harmful direction:
  a caller told `True` attempts something that then fails.

* **The gap is not 1375.** `OpOverload.has_kernel_for_dispatch_key` is
  `self.py_kernels` **or** this predicate, and this tree runs upstream's own
  `activate_meta()`, which puts `DispatchKey.Meta` into `py_kernels` for 1227
  of upstream's 1375 `True` overloads. The caller-visible disagreement is the
  residual, and it is dominated by `.out` variants.

* **There is exactly one caller.** Instrumenting upstream through an export
  shows every `Meta` query arriving from one place -- `resolve_key` in
  `torch/_ops.py`, reached from `OpOverload._get_dispatch` -- and asking mostly
  about `prims::` ops, not `aten::` ones.

What this file cannot see, stated so nobody reads more into it: it measures the
predicate and its one caller. It does not measure whether a *different* future
caller would be harmed, and it deliberately does not assert that the shim keeps
making zero `Meta` queries -- a test that reddens when the shim improves is
worse than no test.
"""

import json
import os
import subprocess
import sys


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
_VENDOR_DIR = os.path.join(_REPO_ROOT, "torchnative", "src", "main")
_VENDOR_SHIM = os.path.join(_VENDOR_DIR, "torch", "_C.abi3.so")

#: Names upstream's dispatcher **raises** on rather than answering: the
#: TorchScript residue `docs/design/REGISTRATIONS.md` §3.2 enumerated. The shim
#: answers `False` for these, and that is load-bearing (§3.4).
_JUNK_NAMES = (
    "aten::ldexp",
    "aten::acos.int",
    "aten::acos.float",
    "aten::_list_to_tensor",
    "aten::_unsafe_index.Tensor_hacked_twin",
)


def _available():
    return os.path.isfile(_VENDOR_SHIM)


def _run(script, shim, stdin="", timeout=900):
    env = dict(os.environ)
    if shim:
        env["PYTHONPATH"] = _VENDOR_DIR
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run(
        [sys.executable, "-c", script], input=stdin,
        capture_output=True, text=True, env=env, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"subprocess exited {proc.returncode}\n"
            f"--- stdout ---\n{proc.stdout}\n--- stderr ---\n{proc.stderr}"
        )
    out = json.loads(proc.stdout.strip().splitlines()[-1])
    # The side each probe landed on is asserted, not assumed: an empty vendored
    # tree makes the "shim" side silently import upstream.
    assert out["is_shim"] is shim, f"probe meant for shim={shim} ran on the other side"
    return out


_PREAMBLE = """
import json, warnings
warnings.filterwarnings("ignore")
import torch
IS_SHIM = hasattr(torch._C, "_aten_implemented")
"""


# --- the whole aten surface, on whichever side the probe runs ---------------
_SURFACE = _PREAMBLE + """
rows, errs = {}, {}
for opname in dir(torch.ops.aten):
    if opname.startswith("__"):
        continue
    try:
        packet = getattr(torch.ops.aten, opname)
        overloads = packet.overloads()
    except Exception:
        continue
    for ov in overloads:
        try:
            name = getattr(packet, ov).name()
        except Exception:
            continue
        try:
            rows[name] = bool(
                torch._C._dispatch_has_kernel_for_dispatch_key(name, "Meta"))
        except Exception as exc:
            errs[name] = type(exc).__name__
print(json.dumps({"is_shim": IS_SHIM, "rows": rows, "errs": errs}))
"""


# --- does the shim's own meta dispatcher have an arm for this op? ----------
#
# Derived by *calling* it, not by reading a list beside it: the only arm that
# produces "has no meta kernel for" is `meta_table`'s fallthrough, so the
# refusal wording is the implementation answering the question about itself.
_SHIM_ARMS = _PREAMBLE + """
assert IS_SHIM
x = torch.zeros(2, 2, device="meta")
has, no = [], []
for key in sorted(set(torch._C._aten_all_implemented())):
    parts = key.split(".")
    if len(parts) != 3 or parts[0] != "aten":
        continue
    try:
        op = getattr(getattr(torch.ops.aten, parts[1]), parts[2])
    except Exception:
        continue
    try:
        op(x)
        has.append(key)
        continue
    except Exception as exc:
        message = str(exc)
    if "meta kernel for" in message and (
            "has no meta kernel" in message or "will not have a meta kernel" in message):
        no.append(key)
    else:
        has.append(key)
print(json.dumps({"is_shim": IS_SHIM, "has": has, "no": no}))
"""


# --- how many of upstream's `True` names already answer True here ----------
_SHIM_PY_KERNELS = _PREAMBLE + """
import sys
from torch._C import DispatchKey
wanted = json.loads(sys.stdin.read())
covered, residual = [], []
for full in wanted:
    body = full[len("aten::"):]
    name, _, ov = body.partition(".")
    try:
        op = getattr(getattr(torch.ops.aten, name), ov or "default")
    except Exception:
        residual.append(full)
        continue
    if DispatchKey.Meta in getattr(op, "py_kernels", {}):
        covered.append(full)
    else:
        residual.append(full)
print(json.dumps({"is_shim": IS_SHIM, "covered": covered, "residual": residual}))
"""


# --- who asks the question, during a real export ---------------------------
_CALLERS = _PREAMBLE + """
import collections, traceback
original = torch._C._dispatch_has_kernel_for_dispatch_key
asked = collections.Counter()
sites = collections.Counter()

def recording(name, key, *args, **kwargs):
    label = getattr(key, "name", str(key))
    if label == "Meta":
        asked[str(name)] += 1
        frame = traceback.extract_stack()[-3]
        sites[f"{frame.filename.split('/torch/')[-1]}:{frame.name}"] += 1
    return original(name, key, *args, **kwargs)

torch._C._dispatch_has_kernel_for_dispatch_key = recording


class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = torch.nn.LayerNorm(6)

    def forward(self, x):
        return self.ln(x)


exported = True
try:
    torch.export.export(M().eval(), (torch.randn(4, 6),))
except Exception:
    exported = False
print(json.dumps({"is_shim": IS_SHIM, "exported": exported,
                  "asked": dict(asked), "sites": dict(sites)}))
"""


_JUNK_PROBE = _PREAMBLE + """
names = %r
answers, raised = {}, {}
for name in names:
    try:
        answers[name] = torch._C._dispatch_has_kernel_for_dispatch_key(name, "Meta")
    except Exception as exc:
        raised[name] = type(exc).__name__
print(json.dumps({"is_shim": IS_SHIM, "answers": answers, "raised": raised}))
""" % (list(_JUNK_NAMES),)


_CACHE = {}


def _surface(shim):
    key = ("surface", shim)
    if key not in _CACHE:
        _CACHE[key] = _run(_SURFACE, shim)
    return _CACHE[key]


def _skip(name):
    print(f"SKIP {name}: no vendored shim at {_VENDOR_SHIM}")


# ---------------------------------------------------------------------------
# 1. The predicate's shape here, and why it is not an oversight.
# ---------------------------------------------------------------------------
def test_the_meta_predicate_answers_false_for_every_name_and_never_raises():
    """`False` for everything, including names upstream refuses to answer at all.

    `docs/design/REGISTRATIONS.md` §3.4: this predicate never validating a name
    is what keeps `_dispatch_has_kernel = True`'s 251 TorchScript residue
    overloads from killing `activate_meta()`. The two are a set. A round that
    makes this one "more realistic" on its own reddens this test, which is the
    point.
    """
    if not _available():
        return _skip("test_the_meta_predicate_answers_false_for_every_name_and_never_raises")
    shim = _surface(True)
    assert not shim["errs"], (
        f"the shim raised for {len(shim['errs'])} names; it must answer, not "
        f"validate: {sorted(shim['errs'])[:10]}")
    wrong = sorted(n for n, v in shim["rows"].items() if v)
    assert not wrong, f"the shim answered True for {len(wrong)} names: {wrong[:10]}"

    junk = _run(_JUNK_PROBE, True)
    assert not junk["raised"], (
        f"the shim raised for the TorchScript residue {sorted(junk['raised'])}; "
        f"REGISTRATIONS.md §3.3 measured that this is what kills activate_meta()")
    assert set(junk["answers"].values()) == {False}, junk["answers"]
    print(f"METAKEY: shim answers False for all {len(shim['rows'])} aten overloads "
          f"and raises for none, the residue included")


def test_upstream_answers_false_for_a_measured_minority_so_true_is_not_the_target():
    """VARMEAN.md §4.1's 408, re-derived rather than quoted.

    A blanket `True` would agree with upstream *less* than the constant `False`
    does on these, and upstream raises on a third group that neither answer
    covers.
    """
    up = _surface(False)
    yes = sum(1 for v in up["rows"].values() if v)
    no = sum(1 for v in up["rows"].values() if not v)
    raised = len(up["errs"])
    assert no >= 400, f"upstream answers False for only {no}; VARMEAN.md §4.1 measured 408"
    assert raised >= 250, f"upstream raised for only {raised}; REGISTRATIONS.md §3.2 measured ~300"
    assert yes + no + raised >= 2000, (yes, no, raised)
    print(f"METAKEY: upstream over {yes + no + raised} aten overloads -- "
          f"True {yes}, False {no}, raises {raised}")


# ---------------------------------------------------------------------------
# 2. The obvious repair, and the measurement that rejects it.
# ---------------------------------------------------------------------------
def test_deriving_the_answer_from_this_shims_meta_arms_would_disagree_in_the_harmful_direction():
    """"Compute it from what the shim implements" is wrong, by 28 ops.

    `meta_dispatch` answers "can this be computed without storage". Upstream's
    predicate answers "is there a registration at the Meta key". They are
    different questions, and the views and copies are where they part: upstream
    reaches `clone`/`detach`/`expand`/`t`/`slice` through
    `CompositeExplicitAutograd` and `matmul`/`var_mean.default` through
    `CompositeImplicitAutograd`, so its honest answer is `False` while this shim
    has a real meta arm for every one of them.

    Answering `True` there is the direction that harms: `resolve_key` would
    return `Meta` where upstream returns the alias key.
    """
    if not _available():
        return _skip("test_deriving_the_answer_from_this_shims_meta_arms_would_disagree_in_the_harmful_direction")
    arms = _run(_SHIM_ARMS, True)
    up = _surface(False)["rows"]

    def upstream_says(key):
        _, name, ov = key.split(".")
        return up.get("aten::" + name + ("" if ov == "default" else "." + ov))

    have = set(arms["has"])
    assert len(have) >= 100, f"only {len(have)} meta arms found; the probe is not reaching them"
    harmful = sorted(k for k in have if upstream_says(k) is False)
    named = {
        "aten.clone.default", "aten.detach.default", "aten.expand.default",
        "aten.t.default", "aten.transpose.int", "aten.slice.Tensor",
        "aten._to_copy.default", "aten.matmul.default", "aten.permute.default",
        "aten.reshape.default",
    }
    missing = sorted(named - set(harmful))
    assert not missing, (
        f"these are the ops the disagreement is made of and upstream no longer "
        f"answers False for them: {missing}")
    assert len(harmful) >= 20, (
        f"only {len(harmful)} ops disagree; if this has collapsed, the derivation "
        f"this test rejects may have become correct -- re-measure before trusting it")
    print(f"METAKEY: {len(harmful)} of {len(have)} ops with a meta arm here are ops "
          f"upstream answers False for -- a derived predicate would be wrong for "
          f"every one, in the harmful direction")


# ---------------------------------------------------------------------------
# 3. The size of the disagreement a caller can actually see.
# ---------------------------------------------------------------------------
def test_most_of_upstreams_meta_registrations_already_answer_true_here_through_py_kernels():
    """The caller-visible gap is the residual, not the 1375.

    `OpOverload.has_kernel_for_dispatch_key` is `dk in self.py_kernels` **or**
    this predicate, and this tree runs upstream's `activate_meta()`. So the
    constant `False` is invisible for every op that got a Python meta
    registration, which is most of them.
    """
    if not _available():
        return _skip("test_most_of_upstreams_meta_registrations_already_answer_true_here_through_py_kernels")
    up = _surface(False)["rows"]
    wanted = sorted(n for n, v in up.items() if v)
    out = _run(_SHIM_PY_KERNELS, True, stdin=json.dumps(wanted))
    covered, residual = out["covered"], out["residual"]
    assert len(covered) >= 1100, (
        f"only {len(covered)} of upstream's {len(wanted)} Meta registrations answer "
        f"True here through py_kernels; activate_meta() may have stopped running")
    assert len(residual) <= 300, (
        f"{len(residual)} overloads answer False here and True upstream")
    outs = [n for n in residual if ".out" in n or n.endswith("_")]
    assert len(outs) >= len(residual) // 2, (
        f"the residual is no longer dominated by out-variants and in-place "
        f"spellings ({len(outs)} of {len(residual)}) -- it has changed shape")
    print(f"METAKEY: of upstream's {len(wanted)} Meta-True overloads, {len(covered)} "
          f"already answer True here via py_kernels; residual {len(residual)}, of "
          f"which {len(outs)} are out-variants or in-place spellings")


# ---------------------------------------------------------------------------
# 4. Who asks, and about what.
# ---------------------------------------------------------------------------
def test_the_only_caller_that_asks_about_the_meta_key_is_resolve_key_and_it_asks_about_prims():
    """One call site, and it is mostly not asking about `aten` at all.

    Upstream, exporting a four-line LayerNorm module, asks this question nine
    times and every one arrives through `torch/_ops.py`'s `resolve_key`, reached
    from `OpOverload._get_dispatch`. Most of the names are `prims::`. That
    reframes the gap: a registry of this shim's *aten* meta arms would not have
    been the answer to the question being asked.

    The shim's own count is printed and not asserted -- a test that reddens when
    the shim starts reaching this path would be a test against progress.
    """
    up = _run(_CALLERS, False)
    assert up["exported"], "upstream failed to export the probe module"
    assert up["asked"], "upstream asked nothing about the Meta key; the probe moved"
    assert set(up["sites"]) == {"_ops.py:resolve_key"}, (
        f"more than one call site now asks about the Meta key: {up['sites']} -- "
        f"this file's conclusion rests on there being exactly one")
    prims = [n for n in up["asked"] if n.startswith("prims::")]
    assert len(prims) > len(up["asked"]) // 2, (
        f"the names asked about are no longer mostly prims: {sorted(up['asked'])}")
    line = (f"METAKEY: upstream asks {sum(up['asked'].values())} Meta questions while "
            f"exporting a LayerNorm, all from {sorted(up['sites'])}, "
            f"{len(prims)} of {len(up['asked'])} names in prims::")
    if _available():
        shim = _run(_CALLERS, True)
        line += (f"; the shim asks {sum(shim['asked'].values())} "
                 f"(exported={shim['exported']})")
    print(line)


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"ok   {name}")
            except Exception as exc:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {exc}")
    sys.exit(1 if failed else 0)
