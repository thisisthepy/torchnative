"""Issue #372: which CompositeImplicitAutograd ops upstream's exports decompose.

Direction: leader decision 2026-11-16, agent proposal; not owner-confirmed.
The leader's condition (1) for the CIA frame is that the ops covered are the
ones upstream's exports actually decompose, measured. This suite is that
measurement for the two models the project names, SmolLM2-135M and
Qwen3-0.6B. It **prints and does not assert on the census**: the list is the
work queue for the next rounds (docs/design/DECOMPOSE.md section 5), and a
count that rises as models change is not a failure.

What it runs: upstream torch only (the oracle, `.caches/spike-venv` locally,
the CI venv in CI), in a subprocess with the vendored tree off `PYTHONPATH`.
`torch.export.export(model, (ids,), {"use_cache": False})` and then
`ep.run_decompositions()`, with `OpOverload.decompose` wrapped to count the
ops it decomposed and `FunctionalTensorMode.__torch_dispatch__` wrapped to
record the ops that mode received. It prints, per model: the CIA ops in the
exported (pre-decomposition) graph, the CIA ops `FunctionalTensorMode`
received and decomposed, and every op `decompose` answered for (FakeTensorMode
internals included, such as `aten.to.dtype` under `addmm`'s Python
decomposition). When the vendored tree is built it also asks the shim for its
`_shim_cia_table()` and prints which of them are covered, pending (a native
kernel here, no body) or neither.

The models are the tiny ones `tests/models/test_stream.py` builds (the same
architectures, `qwen3` and `llama`, SmolLM2's; random weights, no download):
the real checkpoints are in the repository's HF cache only on the
maintainer's machine, and the CIA ops an export goes through are set by the
architecture, not the sizes. NOT MEASURED: the real checkpoints' configs
(`tie_word_embeddings`, `rope_scaling` and attention settings could add ops),
generation with a cache, and the shim side (it cannot export these models
through `run_decompositions` until the census is covered).

What this cannot find: ops reached only by inputs other than an 8-token
prompt with no cache, and ops that decompose without passing through
`OpOverload.decompose` (a Python dispatcher `py_kernel` called directly).

Written before it ran: agents do not run models (leader rule). The first CI
run is the measurement; an upstream export that fails is a SKIP naming where,
not a FAIL.
"""
import json
import os
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
VENDOR = ROOT / "torchnative" / "python"

sys.path.insert(0, str(ROOT / "tests" / "_support"))
import _skip  # noqa: E402

#: `tests/models/test_stream.py::_TINY_CFG`, copied: that suite builds them on
#: disk with a tokenizer, this one only needs the module.
_TINY_CFG = {
    "qwen3": dict(
        model_type="qwen3", vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=64, tie_word_embeddings=True,
        rope_theta=1e6, bos_token_id=1, eos_token_id=[2, 3], pad_token_id=1),
    "smollm2": dict(
        model_type="llama", vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, tie_word_embeddings=True, rope_theta=1e5,
        bos_token_id=1, eos_token_id=2, pad_token_id=1),
}
_IDS = [[5, 7, 9, 11, 13, 17, 19, 23]]


def _run(script, shim, *extra, timeout=1800):
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"
    if shim:
        env["PYTHONPATH"] = str(VENDOR)
        env["TORCH_USE_RTLD_GLOBAL"] = "1"
    else:
        env.pop("PYTHONPATH", None)
        env.pop("TORCH_USE_RTLD_GLOBAL", None)
    proc = subprocess.run([sys.executable, "-c", script, "shim" if shim else "upstream", *extra],
                          capture_output=True, text=True, env=env, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(
            f"probe ({'shim' if shim else 'upstream'}) exited {proc.returncode}\n"
            f"--- stderr ---\n{proc.stderr[-4000:]}"
        )
    return json.loads(proc.stdout.strip().splitlines()[-1])


def _have_shim():
    return (VENDOR / "torch" / "__init__.py").is_file()


def _have_upstream():
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONPATH", "TORCH_USE_RTLD_GLOBAL")}
    proc = subprocess.run([sys.executable, "-c", "import torch, sys; "
                           "sys.exit(1 if hasattr(torch._C, '_aten_implemented') else 0)"],
                          capture_output=True, env=env, timeout=600)
    return proc.returncode == 0


_CENSUS = r'''
import collections, json, sys, traceback
import torch
assert not hasattr(torch._C, "_aten_implemented"), "the census runs on upstream only"  # AGENTS.md 15.2
A = json.loads(sys.argv[2])
out = {}
try:
    import transformers
except Exception as exc:  # noqa: BLE001
    print(json.dumps({"skip": f"transformers not importable: {type(exc).__name__}: {exc}"}))
    raise SystemExit(0)
from torch._subclasses.functional_tensor import FunctionalTensorMode

CIA = torch._C.DispatchKey.CompositeImplicitAutograd


#: Overloads the C++ dispatcher does not know (TorchScript builtins such as
#: `prim::device`, which `FunctionalTensorMode` receives as metadata queries).
#: `_dispatch_has_kernel_for_dispatch_key` raises "operator prim::device does
#: not exist" for them on upstream (first CI run of #373), so they are asked
#: nothing, recorded here, and printed as "not a dispatcher operator".
NOT_IN_DISPATCHER = set()


def known_to_dispatcher(op):
    schema = op._schema
    try:
        torch._C._dispatch_find_schema_or_throw(schema.name, schema.overload_name)
    except RuntimeError:
        return False
    return True


def is_cia(op):
    if not isinstance(op, torch._ops.OpOverload):
        return False
    if CIA in op.py_kernels:
        return True
    if not known_to_dispatcher(op):
        NOT_IN_DISPATCHER.add(str(op))
        return False
    return torch._C._dispatch_has_kernel_for_dispatch_key(op.name(), CIA)


def where(exc):
    frames = traceback.extract_tb(exc.__traceback__)
    inside = [f for f in frames if "/torch/" in f.filename.replace("\\", "/")
              or "/transformers/" in f.filename.replace("\\", "/")]
    f = inside[-1] if inside else (frames[-1] if frames else None)
    site = None if f is None else f.filename.replace("\\", "/").split("/site-packages/")[-1] + f":{f.lineno}"
    return f"{type(exc).__name__} at {site}: {str(exc)[:300]}"


decomposed = collections.Counter()
received = []
_orig_decompose = torch._ops.OpOverload.decompose


def logged_decompose(self, *args, **kwargs):
    r = _orig_decompose(self, *args, **kwargs)
    if r is not NotImplemented:
        decomposed[str(self)] += 1
    return r


_orig_functional = FunctionalTensorMode.__torch_dispatch__


def logged_functional(self, func, types, args=(), kwargs=None):
    received.append(func)
    return _orig_functional(self, func, types, args, kwargs)


for name, cfg in A["configs"].items():
    row = {}
    out[name] = row
    NOT_IN_DISPATCHER.clear()
    try:
        if isinstance(cfg, str):
            # A real checkpoint, read from the local hub cache (HF_HUB_OFFLINE).
            model = transformers.AutoModelForCausalLM.from_pretrained(
                cfg, attn_implementation="sdpa").eval()
            row["dtype"] = str(next(model.parameters()).dtype)
        else:
            config = transformers.AutoConfig.for_model(**cfg)
            config._attn_implementation = "sdpa"
            torch.manual_seed(0)
            model = transformers.AutoModelForCausalLM.from_config(config).eval()
        ids = torch.tensor(A["ids"])
        ep = torch.export.export(model, (ids,), {"use_cache": False})
    except Exception as exc:  # noqa: BLE001
        row["export_wall"] = where(exc)
        continue
    row["exported_cia"] = sorted({str(n.target) for n in ep.graph.nodes
                                  if n.op == "call_function" and is_cia(n.target)})
    decomposed.clear()
    received.clear()
    torch._ops.OpOverload.decompose = logged_decompose
    FunctionalTensorMode.__torch_dispatch__ = logged_functional
    try:
        ep.run_decompositions()
    except Exception as exc:  # noqa: BLE001
        row["run_decompositions_wall"] = where(exc)
    finally:
        torch._ops.OpOverload.decompose = _orig_decompose
        FunctionalTensorMode.__torch_dispatch__ = _orig_functional
    cia_received = sorted({str(f) for f in received if is_cia(f)})
    row["functional_cia_received"] = cia_received
    row["functional_cia_decomposed"] = sorted(op for op in cia_received if decomposed.get(op))
    row["decomposed_all"] = dict(sorted(decomposed.items()))
    row["not_dispatcher_ops"] = sorted(NOT_IN_DISPATCHER)
print(json.dumps({"out": out}))
'''

_SHIM_TABLE = r'''
import json, sys
import torch
assert hasattr(torch._C, "_aten_implemented"), "the shim, not upstream"  # AGENTS.md 15.2
print(json.dumps(torch._C._shim_cia_table()))
'''


def test_print_the_cia_ops_upstreams_exports_of_smollm2_and_qwen3_decompose():
    """Prints `CIACENSUS <model> ...` lines. The only assertion is that the
    probe measured something for every model that exported (a census that is
    empty for an exported LLM means the wrappers saw nothing, AGENTS.md 17.5)."""
    if not _have_upstream():
        _skip.skip("upstream torch not importable without the vendored tree")
        return
    _census(_TINY_CFG)


#: The models the end goal names (issue #377). No hosted runner has them: the
#: suites that use them (`tests/models/test_qwen3.py`) read a local hub cache
#: and skip by name, and so does this. Leader decision 2026-11-16: no
#: checkpoint cache is added to CI; the leader runs this once by hand.
_REAL = {"qwen3-0.6b": "Qwen/Qwen3-0.6B", "smollm2-135m": "HuggingFaceTB/SmolLM2-135M"}


def _hub():
    home = os.environ.get("HF_HOME") or str(ROOT / ".caches" / "hf-home")
    return pathlib.Path(home) / "hub"


def test_print_the_cia_ops_upstreams_exports_of_the_real_checkpoints_decompose():
    """The same census on the real SmolLM2-135M and Qwen3-0.6B checkpoints,
    for whichever of them the local hub cache holds. Prints only."""
    if not _have_upstream():
        _skip.skip("upstream torch not importable without the vendored tree")
        return
    hub = _hub()
    have = {name: repo for name, repo in _REAL.items()
            if (hub / ("models--" + repo.replace("/", "--")) / "snapshots").is_dir()}
    if not have:
        _skip.skip(f"no real checkpoint in the hub cache {hub} (wanted {sorted(_REAL.values())})")
        return
    for name in sorted(set(_REAL) - set(have)):
        print(f"CIACENSUS {name} not in the hub cache {hub}: not measured")
    os.environ["HF_HOME"] = str(hub.parent)
    _census(have)


def _census(configs):
    got = _run(_CENSUS, False, json.dumps({"configs": configs, "ids": _IDS}))
    if "skip" in got:
        _skip.skip(got["skip"])
        return
    table = None
    if _have_shim():
        try:
            table = _run(_SHIM_TABLE, True)
        except RuntimeError as exc:
            print(f"CIACENSUS shim table unavailable: {str(exc)[:300]}")
    walls = []
    for name, row in sorted(got["out"].items()):
        if "export_wall" in row:
            walls.append(f"{name}: export: {row['export_wall']}")
            print(f"CIACENSUS {name} upstream export stopped: {row['export_wall']}")
            continue
        if "run_decompositions_wall" in row:
            print(f"CIACENSUS {name} upstream run_decompositions stopped: {row['run_decompositions_wall']}")
        if "dtype" in row:
            print(f"CIACENSUS {name} loaded parameter dtype: {row['dtype']}")
        print(f"CIACENSUS {name} exported graph CIA ops: {row['exported_cia']}")
        print(f"CIACENSUS {name} FunctionalTensorMode received CIA ops: {row['functional_cia_received']}")
        print(f"CIACENSUS {name} ... of which decomposed: {row['functional_cia_decomposed']}")
        print(f"CIACENSUS {name} every decompose answer (Fake internals included): {row['decomposed_all']}")
        print(f"CIACENSUS {name} not a dispatcher operator (not asked): {row.get('not_dispatcher_ops', [])}")
        if table is not None:
            ops = sorted(set(row["exported_cia"]) | set(row["functional_cia_received"]))
            covered = [op for op in ops if op in table["covered"]]
            pending = [op for op in ops if op in table["pending"]]
            neither = [op for op in ops if op not in covered and op not in pending]
            print(f"CIACENSUS {name} shim: covered {covered}; pending (native kernel, no body) "
                  f"{pending}; no body and no native kernel {neither}")
        assert row["decomposed_all"] or row["exported_cia"], (
            name, "the census saw nothing: the wrappers were not reached", row)
    if walls and len(walls) == len(got["out"]):
        _skip.skip("; ".join(walls))
        return


if __name__ == "__main__":
    sys.exit(1 if _skip.run_tests(
        [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)],
        suite="test_cia_census",
    ) else 0)
