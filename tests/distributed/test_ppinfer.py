"""Pipeline-parallel generation of a real `transformers` model on backend `torchnative`.

BACKEND.md section 9 step 10 (issue #295), SPEC S6.13, design in
docs/distributed/PPINFER.md. A tiny Qwen3 (the project's test model family,
owner 2026-10-03) with four layers is written once by upstream `from_config`
under a fixed seed, so every side loads identical weights. Then:

* single-process, on the shim and on upstream: transformers' own greedy
  `generate`, the reference, and `load_layer_subset` of several ranges with
  the reads recorded;
* pipelined over `init_process_group("torchnative")` on the shim and over
  gloo on upstream torch: `PipelineStage.from_pretrained(path, split=...)`
  then `generate` on every rank, at splits 2:2, 3:1, 1:2:1 (given as ranges)
  and, over a tied-embedding checkpoint, 1:3.

The runner is torchnative's (`torchnative/distributed/pipeline.py`), but it
calls only public `torch.distributed` and transformers API, so the upstream
children import it too (appended to `sys.path` after upstream `torch` is
imported, and asserted to be upstream). That makes two independent checks:
the runner over gloo on upstream torch must reproduce upstream's own
`generate`, and the runner over the shim's backend must reproduce the shim's
own `generate`. Both bit for bit.

Why `tests/distributed/`: it reuses `test_collect2.py`'s subprocess-world
harness (`_c2_spawn`) and `test_tpinfer.py`'s helpers by import, which works
only from this folder (`tests/run.sh` puts the suite's folder at
`sys.path[0]` and only `tests/_support/` on PYTHONPATH).

**What can fail** (AGENTS.md section 17.5):

* the tokens: every rank's sequences must equal the single-process sequences
  of its own torch, element for element;
* what crossed ranks is read from the backend (`record_traffic`), not from
  the runner, and must equal the protocol's exact list: a header and a hidden
  state of `(batch, length, hidden)` per step per forward edge, a header and
  the step's token ids back to stage 0, one final sequences message down the
  chain. A runner that shipped logits, or sent a step twice, fails there;
* every stage's cache must hold exactly its own layer count, each at the
  length the step count implies, and its layers' `layer_idx` must be local;
* the checkpoint reads are recorded by wrapping the `safe_open` the installed
  loader calls and logging every slice it materialises (`tensor[...]`). The
  read set must equal the subset's keys exactly. The instrument is checked
  too: a full `from_pretrained` under it must record every key in the file,
  so a recorder that saw nothing cannot pass;
* sabotage 1 (stale hidden state): stage 0 sends the previous decode step's
  hidden state from the second decode step on. The tokens must change, on
  both sides;
* sabotage 2 (a stage drops its cache updates): with the runner's position
  check left in, that rank must raise `PipelineDesync` naming itself; with the
  check also disabled, the tokens must change. So the check and the token
  comparison each catch it on their own;
* every shim child asserts `hasattr(torch._C, "_aten_implemented")` and every
  upstream child its absence (AGENTS.md section 15.2).

**What this cannot see** (AGENTS.md section 17.4): padded batches (refused),
sampling and beam search (refused), dtypes other than float32, a checkpoint
sharded over several files, any device other than CPU (a vulkan or npu stage
is a device test the coordinating session runs), memory actually saved (the
reads and the allocated parameters are asserted, not resident memory), and
speed.
"""

import functools
import json
import os
import shutil
import tempfile
import atexit

import _skip
import gloo_loopback

from test_collect2 import _c2_spawn
from test_shim import _CKPT_REPO_ROOT, _CKPT_VENDOR_DIR
from test_tpinfer import _reached, _require_shim

_LAYERS = 4
_HIDDEN = 48
_NEW = 6

#: Untied, so every stage's head and embedding are separate tensors. Same
#: shape family as test_tpinfer.py's checkpoint, with four layers so that a
#: three-stage split and a 3:1 split both exist.
_CFG = dict(
    vocab_size=96, hidden_size=_HIDDEN, intermediate_size=60,
    num_hidden_layers=_LAYERS, num_attention_heads=12, num_key_value_heads=6,
    head_dim=8, max_position_embeddings=64, tie_word_embeddings=False,
    rope_theta=1e6, bos_token_id=1, eos_token_id=2, pad_token_id=1,
)
#: The small Qwen3 checkpoints tie their embeddings; one world covers that.
_CFG_TIED = dict(_CFG, tie_word_embeddings=True)

#: Two rows, equal length, no pad token (id 1) in them: a batch, but no padding.
_IDS = [[5, 7, 9, 11, 13, 17, 19, 23], [29, 31, 37, 41, 43, 47, 53, 59]]
_BATCH, _PROMPT = len(_IDS), len(_IDS[0])

#: name -> (world, split as given to the runner, tied checkpoint?)
_WORLDS = {
    "even2": (2, [2, 2], False),
    "uneven2": (2, [3, 1], False),
    "three": (3, [[0, 1], [1, 3], [3, 4]], False),
    "tied2": (2, [1, 3], True),
}

#: Ranges probed by the single-process loader test: (start, stop, tied?).
_LOADER_RANGES = [(0, 2, False), (2, 4, False), (1, 3, False), (3, 4, False),
                  (0, 4, False), (0, 1, True), (1, 3, True), (3, 4, True)]

_TAG_HEADER, _TAG_HIDDEN, _TAG_IDS, _TAG_RESULT = 7301, 7302, 7303, 7304


# ---------------------------------------------------------------------------
# One probe body, shared verbatim by the shim and the upstream children.
# ---------------------------------------------------------------------------

_PP_PROBE_SRC = r'''
def _install_read_recorder():
    """Log every checkpoint slice the installed loader materialises."""
    import transformers.modeling_utils as mu
    real = mu.safe_open
    reads = []

    class _Slice:
        def __init__(self, inner, key):
            self._inner, self._key = inner, key
        def __getitem__(self, idx):
            reads.append(self._key)
            return self._inner[idx]
        def __getattr__(self, name):
            return getattr(self._inner, name)

    class _File:
        def __init__(self, *a, **k):
            self._f = real(*a, **k)
        def get_slice(self, key):
            return _Slice(self._f.get_slice(key), key)
        def __getattr__(self, name):
            return getattr(self._f, name)
        def __enter__(self):
            self._f.__enter__()
            return self
        def __exit__(self, *exc):
            return self._f.__exit__(*exc)

    mu.safe_open = _File
    return reads, lambda: setattr(mu, "safe_open", real)


def _ckpt_tensors(path):
    """{key: tensor} straight from the file, outside the recorder."""
    import glob
    from safetensors import safe_open
    out = {}
    for name in sorted(glob.glob(path + "/*.safetensors")):
        with safe_open(name, framework="pt") as f:
            for key in f.keys():
                out[key] = f.get_slice(key)[...]
    return out


def _refusal(fn):
    try:
        fn()
    except Exception as exc:
        return "%s: %s" % (type(exc).__name__, exc)
    return None


def _stage_facts(torch, model):
    info = dict(model._torchnative_stage)
    info.pop("loading_info", None)
    layers = model.model.layers
    return {
        "class": type(model).__name__,
        "module": type(model).__module__,
        "params": sorted(n for n, _ in model.named_parameters()),
        "meta_params": sorted(n for n, p in model.named_parameters()
                              if p.device.type == "meta"),
        "layer_idx": [l.self_attn.layer_idx for l in layers],
        "layer_keys": sorted({int(k.split(".")[2]) for k in model.state_dict()
                              if k.startswith("model.layers.")}),
        "n_layers": len(layers),
        "config_layers": model.config.num_hidden_layers,
        "embed_is_none": model.model.embed_tokens is None,
        "head_is_none": model.lm_head is None,
        "norm_is_identity": isinstance(model.model.norm, torch.nn.Identity),
        "head_shares_embed": (model.lm_head is not None
                              and model.model.embed_tokens is not None
                              and model.lm_head.weight is model.model.embed_tokens.weight),
        "stage": {k: (list(v) if isinstance(v, tuple) else v) for k, v in info.items()
                  if k != "global_layer_idx"},
        "global_layer_idx": {str(k): v for k, v in info["global_layer_idx"].items()},
    }


def _values_match(model, ckpt):
    """Names whose loaded value differs from the file's bytes (empty = all equal)."""
    bad = []
    for name, p in model.named_parameters():
        ref = ckpt.get(name)
        if ref is None or tuple(ref.shape) != tuple(p.shape) or not bool((ref == p.detach()).all()):
            bad.append(name)
    return bad


def pp_child(torch, dist, rank, world, A, shim, init):
    import os, traceback
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    os.environ["HF_HOME"] = A["hf_home"]
    out = {"stage": "start", "rank": rank, "world": world, "shim": shim}
    try:
        from transformers import AutoModelForCausalLM, Qwen3Config
        dt = getattr(torch, A["dtype"])

        if A["mode"] == "write":
            out["stage"] = "write"
            torch.manual_seed(0)
            m = AutoModelForCausalLM.from_config(Qwen3Config(**A["config"])).eval()
            with torch.no_grad():
                for p in m.parameters():
                    p.normal_(0.0, 0.3)
            m.save_pretrained(A["path"])
            out["file_keys"] = sorted(_ckpt_tensors(A["path"]))
            out["stage"] = "done"
            return out

        import torchnative
        out["torchnative_file"] = torchnative.__file__
        from torchnative.distributed import pipeline as pp
        from torchnative.transformers import load_layer_subset

        ids = torch.tensor(A["ids"], dtype=torch.long)
        gen = dict(max_new_tokens=A["new"], min_new_tokens=A["new"], do_sample=False)

        if A["mode"] == "single":
            out["stage"] = "reference"
            reads, restore = _install_read_recorder()
            model = AutoModelForCausalLM.from_pretrained(A["path"], dtype=dt).eval()
            restore()
            out["full_reads"] = sorted(reads)
            with torch.no_grad():
                seq = model.generate(ids, attention_mask=torch.ones_like(ids), **gen)
            out["sequences"] = seq.tolist()
            del model

            out["stage"] = "loader"
            ckpt = _ckpt_tensors(A["path"])
            probes = []
            for start, stop in A["loader_ranges"]:
                reads, restore = _install_read_recorder()
                try:
                    m = load_layer_subset(A["path"], layers=(start, stop), dtype=dt)
                finally:
                    restore()
                facts = _stage_facts(torch, m)
                facts["range"] = [start, stop]
                facts["reads"] = sorted(set(reads))
                facts["value_mismatch"] = _values_match(m, ckpt)
                from transformers import DynamicCache
                facts["cache_entries"] = len(DynamicCache(config=m.config).layers)
                probes.append(facts)
                del m
            out["probes"] = probes

            out["stage"] = "refusals"
            P = A["path"]
            out["refusals"] = {
                "empty_range": _refusal(lambda: load_layer_subset(P, layers=(2, 2))),
                "reversed_range": _refusal(lambda: load_layer_subset(P, layers=(3, 1))),
                "outside_range": _refusal(lambda: load_layer_subset(P, layers=(0, 5))),
                "non_contiguous_layers": _refusal(lambda: load_layer_subset(P, layers=[0, 1, 3])),
                "device_map": _refusal(lambda: load_layer_subset(P, layers=(0, 2), device_map="cpu")),
                "split_empty_stage": _refusal(lambda: pp.split_layers(4, 3, [2, 0, 2])),
                "split_gap": _refusal(lambda: pp.split_layers(4, 2, [[0, 2], [3, 4]])),
                "split_overlap": _refusal(lambda: pp.split_layers(4, 2, [[0, 3], [2, 4]])),
                "split_short": _refusal(lambda: pp.split_layers(4, 2, [1, 1])),
                "split_one_stage": _refusal(lambda: pp.split_layers(4, 1, [4])),
                "split_wrong_count": _refusal(lambda: pp.split_layers(4, 2, [1, 1, 2])),
                "split_hook": pp.split_layers(4, 2, lambda n, w: [n - 1, 1]),
                "stage_without_group": _refusal(lambda: pp.PipelineStage.from_pretrained(P, split=[2, 2])),
            }
            out["stage"] = "done"
            return out

        # -- pipeline --------------------------------------------------------
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world)
        out["stage"] = "init_process_group"
        init()
        out["backend"] = dist.get_backend()
        traffic = None
        if shim:
            from torchnative.distributed.backend import record_traffic
            traffic = record_traffic()

        out["stage"] = "load"
        reads, restore = _install_read_recorder()
        try:
            stage = pp.PipelineStage.from_pretrained(A["path"], split=A["split"], dtype=dt)
        finally:
            restore()
        out["reads"] = sorted(set(reads))
        out["facts"] = _stage_facts(torch, stage.model)
        out["ranges"] = [list(r) for r in stage.ranges]

        if A["mode"] == "pp_desync":
            out["stage"] = "desync"
            if rank == A["skip_rank"]:
                _drop_cache_updates(stage)
            try:
                stage.generate(ids, **gen)
                out["desync_error"] = None
            except Exception as exc:
                out["desync_error"] = "%s: %s" % (type(exc).__name__, exc)
            out["stage"] = "done"
            return out

        out["stage"] = "generate"
        seq = stage.generate(ids, **gen)
        out["sequences"] = seq.tolist()
        out["steps"] = stage.steps
        cache = stage.cache
        out["cache"] = {
            "entries": len(cache.layers),
            "lengths": [cache.get_seq_length(i) for i in range(len(cache.layers))],
            "key_shapes": [list(layer.keys.shape) for layer in cache.layers],
        }
        if traffic is not None:
            out["traffic"] = [list(t[:4]) + [list(t[4])] for t in traffic]

        out["stage"] = "generate_refusals"
        out["generate_refusals"] = {
            "sampling": _refusal(lambda: stage.generate(ids, do_sample=True, **{k: v for k, v in gen.items() if k != "do_sample"})),
            "beam_search": _refusal(lambda: stage.generate(ids, num_beams=2, max_new_tokens=2)),
            "streamer": _refusal(lambda: stage.generate(ids, streamer=object(), max_new_tokens=2)),
            "no_cache": _refusal(lambda: stage.generate(ids, use_cache=False, max_new_tokens=2)),
        }
        if traffic is not None:
            out["traffic_after_refusals"] = len(traffic)

        out["stage"] = "sabotage_stale_hidden"
        if rank == A["stale_rank"]:
            real_send = stage._send_hidden
            held = {}
            def stale(hidden, past):
                # From the second decode step on, send the step before's.
                if hidden.shape[1] == 1:
                    prev = held.get("prev")
                    held["prev"] = hidden.clone()
                    if prev is not None:
                        hidden = prev
                return real_send(hidden, past)
            stage._send_hidden = stale
        out["stale_sequences"] = stage.generate(ids, **gen).tolist()
        stage.__dict__.pop("_send_hidden", None)

        out["stage"] = "sabotage_dropped_cache_unchecked"
        if rank == A["skip_rank"]:
            _drop_cache_updates(stage)
            stage._check_past = lambda past, src: None
        out["unchecked_sequences"] = stage.generate(ids, **gen).tolist()
        stage.__dict__.pop("_new_cache", None)
        stage.__dict__.pop("_check_past", None)

        out["stage"] = "done"
    except BaseException as exc:
        out["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:1500])
        out["traceback"] = traceback.format_exc()[-4000:]
    return out


def _drop_cache_updates(stage):
    """Sabotage: this stage's cache returns each step's keys and stores nothing."""
    real_new = stage._new_cache
    def new_cache():
        cache = real_new()
        cache.update = lambda key_states, value_states, layer_idx, *a, **k: (
            key_states, value_states)
        return cache
    stage._new_cache = new_cache
'''

_PP_SHIM_SRC = (
    "import json, sys, datetime\n"
    "import torch\n"
    'assert hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded upstream torch, not the shim")\n'
    "import torch.distributed as dist\n"
    "import torchnative.distributed  # noqa: F401 (registers backend='torchnative')\n"
    + _PP_PROBE_SRC
    + r'''
rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
A = json.loads(sys.argv[5])
def init():
    dist.init_process_group(backend="torchnative",
                            init_method="tcp://127.0.0.1:%d" % port,
                            rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=A["timeout"]))
out = pp_child(torch, dist, rank, world, A, True, init)
out["_meta"] = {"shim": hasattr(torch._C, "_aten_implemented"), "rank": rank,
                "torch_file": torch.__file__}
with open(dest, "w") as handle:
    json.dump(out, handle)
'''
)

_PP_GLOO_SRC = (
    gloo_loopback.child_pin_source()
    + "import json, os, sys, datetime\n"
    "import torch\n"
    'assert not hasattr(torch._C, "_aten_implemented"), (\n'
    '    "this subprocess loaded the shim, not upstream torch")\n'
    "import torch.distributed as dist\n"
    + _PP_PROBE_SRC
    + r'''
rank, port, world, dest = (int(sys.argv[1]), int(sys.argv[2]),
                           int(sys.argv[3]), sys.argv[4])
A = json.loads(sys.argv[5])
# Appended after upstream torch is imported, so `torch` stays upstream's and
# only `torchnative` (the runner under test) comes from the source tree.
sys.path.append(A["tn_path"])
def init():
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(backend="gloo", rank=rank, world_size=world,
                            timeout=datetime.timedelta(seconds=A["timeout"]))
out = pp_child(torch, dist, rank, world, A, False, init)
out["_meta"] = {"shim": hasattr(torch._C, "_aten_implemented"), "rank": rank,
                "torch_file": torch.__file__}
with open(dest, "w") as handle:
    json.dump(out, handle)
if dist.is_initialized() and "error" not in out:
    dist.destroy_process_group()
'''
)


# ---------------------------------------------------------------------------
# Fixtures, each run once
# ---------------------------------------------------------------------------

@functools.cache
def _workdir():
    """Inside the repository (AGENTS.md section 2): `.scratch/` is git-ignored."""
    scratch = os.path.join(_CKPT_REPO_ROOT, ".scratch")
    os.makedirs(scratch, exist_ok=True)
    root = tempfile.mkdtemp(prefix="ppinfer-", dir=scratch)
    atexit.register(shutil.rmtree, root, ignore_errors=True)
    return root


def _path(tied):
    return os.path.join(_workdir(), "tiny-qwen3-tied" if tied else "tiny-qwen3")


def _args(mode, tied=False, **extra):
    base = {"mode": mode, "dtype": "float32", "ids": _IDS, "new": _NEW,
            "path": _path(tied), "hf_home": os.path.join(_workdir(), "hf-home"),
            "tn_path": _CKPT_VENDOR_DIR, "timeout": 120,
            "loader_ranges": [[a, b] for a, b, t in _LOADER_RANGES if t == tied]}
    base.update(extra)
    return json.dumps(base)


def _side(shim):
    return "shim" if shim else "gloo"


@functools.cache
def _checkpoint(tied):
    """Written once by upstream; returns the keys in the file."""
    _require_shim()
    [rep] = _c2_spawn(_PP_GLOO_SRC, 1, _args("write", tied,
                                             config=_CFG_TIED if tied else _CFG),
                      False, "pp-write-%s" % ("tied" if tied else "untied"))
    _reached(rep, "write")
    assert rep["file_keys"], rep
    return rep["file_keys"]


@functools.cache
def _single(shim, tied):
    _checkpoint(tied)
    src = _PP_SHIM_SRC if shim else _PP_GLOO_SRC
    [rep] = _c2_spawn(src, 1, _args("single", tied), shim,
                      "pp-single-%s-%s" % (_side(shim), "tied" if tied else "untied"))
    _reached(rep, "single %s" % _side(shim))
    return rep


@functools.cache
def _pp(shim, name):
    world, split, tied = _WORLDS[name]
    _checkpoint(tied)
    src = _PP_SHIM_SRC if shim else _PP_GLOO_SRC
    # Sabotage 2 needs a stage that serves (not the last): stage 0 at world 2,
    # the middle stage at world 3.
    reps = _c2_spawn(src, world, _args("pp", tied, split=split, stale_rank=0,
                                       skip_rank=0 if world == 2 else 1),
                     shim, "pp-%s-%s" % (name, _side(shim)))
    for rep in reps:
        _reached(rep, "pp %s %s" % (name, _side(shim)))
    return reps


@functools.cache
def _pp_desync():
    _checkpoint(False)
    reps = _c2_spawn(_PP_SHIM_SRC, 3, _args("pp_desync", False,
                                            split=[[0, 1], [1, 3], [3, 4]],
                                            skip_rank=1, timeout=60),
                     True, "pp-desync-shim")
    return reps


# ---------------------------------------------------------------------------
# What a stage should hold, read, and send
# ---------------------------------------------------------------------------

def _ranges(name):
    world, split, _ = _WORLDS[name]
    if all(isinstance(c, int) for c in split):
        out, start = [], 0
        for count in split:
            out.append([start, start + count])
            start += count
        return out
    return [list(r) for r in split]


def _expected_reads(file_keys, start, stop, tied):
    """The file keys a stage of layers start..stop-1 must materialise, exactly."""
    embed = start == 0 or (tied and stop == _LAYERS)
    head = stop == _LAYERS or (tied and start == 0)
    out = set()
    for key in file_keys:
        if key.startswith("model.layers."):
            if start <= int(key.split(".")[2]) < stop:
                out.add(key)
        elif key.startswith("model.embed_tokens."):
            if embed:
                out.add(key)
        elif key.startswith("model.norm."):
            if stop == _LAYERS:
                out.add(key)
        elif key.startswith("lm_head."):
            if head:
                out.add(key)
        else:
            raise AssertionError("checkpoint key nobody expected: %r" % key)
    return sorted(out)


def _check_stage_facts(facts, start, stop, tied, file_keys, where):
    width = stop - start
    assert facts["class"] == "Qwen3ForCausalLM", (where, facts["class"])
    assert facts["module"].startswith("transformers."), (where, facts["module"])
    assert facts["meta_params"] == [], (where, facts["meta_params"])
    # Allocated parameters are exactly the keys this stage needed from the file
    # (a tied head is the embedding's tensor, listed once under its first name).
    want = set(_expected_reads(file_keys, start, stop, tied))
    if tied and not facts["embed_is_none"]:
        want.discard("lm_head.weight")
    assert facts["params"] == sorted(want), (where, facts["params"], sorted(want))
    assert facts["n_layers"] == width and facts["config_layers"] == width, (where, facts)
    assert facts["layer_idx"] == list(range(width)), (where, facts["layer_idx"])
    assert facts["layer_keys"] == list(range(start, stop)), (where, facts["layer_keys"])
    assert facts["global_layer_idx"] == {str(i): start + i for i in range(width)}, (
        where, facts["global_layer_idx"])
    assert facts["norm_is_identity"] == (stop != _LAYERS), (where, facts)
    assert facts["embed_is_none"] == (not (start == 0 or (tied and stop == _LAYERS))), (
        where, facts)
    assert facts["head_is_none"] == (not (stop == _LAYERS or (tied and start == 0))), (
        where, facts)
    if tied and not facts["embed_is_none"] and not facts["head_is_none"]:
        assert facts["head_shares_embed"], (where, "tied head is a separate tensor")


def _expected_traffic(rank, world, steps):
    """The exact ordered list of (direction, peer, tag, dtype, shape) at `rank`."""
    last = world - 1
    i64, f32 = "torch.int64", "torch.float32"
    total = _PROMPT + _NEW
    out = []
    for step in range(steps):
        length = _PROMPT if step == 0 else 1
        ids = [_BATCH, length]
        hidden = [_BATCH, length, _HIDDEN]
        if rank == last:
            out += [["send", 0, _TAG_HEADER, i64, [4]], ["send", 0, _TAG_IDS, i64, ids],
                    ["recv", rank - 1, _TAG_HEADER, i64, [4]],
                    ["recv", rank - 1, _TAG_HIDDEN, f32, hidden]]
        elif rank == 0:
            out += [["recv", last, _TAG_HEADER, i64, [4]], ["recv", last, _TAG_IDS, i64, ids],
                    ["send", 1, _TAG_HEADER, i64, [4]], ["send", 1, _TAG_HIDDEN, f32, hidden]]
        else:
            out += [["recv", rank - 1, _TAG_HEADER, i64, [4]],
                    ["recv", rank - 1, _TAG_HIDDEN, f32, hidden],
                    ["send", rank + 1, _TAG_HEADER, i64, [4]],
                    ["send", rank + 1, _TAG_HIDDEN, f32, hidden]]
    result = [_BATCH, total]
    if rank == last:
        out += [["send", 0, _TAG_HEADER, i64, [4]], ["send", 0, _TAG_RESULT, i64, result]]
    else:
        src = last if rank == 0 else rank - 1
        out += [["recv", src, _TAG_HEADER, i64, [4]], ["recv", src, _TAG_RESULT, i64, result]]
        if rank + 1 < last:
            out += [["send", rank + 1, _TAG_HEADER, i64, [4]],
                    ["send", rank + 1, _TAG_RESULT, i64, result]]
    return out


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def _check_tokens(name):
    world, _, tied = _WORLDS[name]
    for shim in (True, False):
        ref = _single(shim, tied)["sequences"]
        assert len(ref) == _BATCH and all(len(r) == _PROMPT + _NEW for r in ref), ref
        assert [r[:_PROMPT] for r in ref] == _IDS, ref
        for rep in _pp(shim, name):
            assert rep["backend"] == ("torchnative" if shim else "gloo"), rep["backend"]
            assert rep["sequences"] == ref, (
                "%s %s rank %d: pipelined tokens %r differ from the single-process "
                "run %r" % (name, _side(shim), rep["rank"], rep["sequences"], ref))
            assert rep["steps"] == _NEW, (name, _side(shim), rep["rank"], rep["steps"])


def test_two_stages_generate_the_single_process_tokens_bit_for_bit():
    _check_tokens("even2")


def test_three_stages_generate_the_single_process_tokens_bit_for_bit():
    _check_tokens("three")


def test_an_uneven_three_to_one_split_generates_the_same_tokens():
    _check_tokens("uneven2")


def test_a_tied_embedding_checkpoint_split_one_to_three_generates_the_same_tokens():
    _check_tokens("tied2")


def test_each_stage_cache_holds_only_its_own_layers():
    for name, (world, _, tied) in sorted(_WORLDS.items()):
        ranges = _ranges(name)
        for shim in (True, False):
            for rep in _pp(shim, name):
                start, stop = ranges[rep["rank"]]
                cache = rep["cache"]
                where = (name, _side(shim), rep["rank"])
                assert cache["entries"] == stop - start, (where, cache)
                # The last token is never fed back, so each layer holds
                # prompt + new - 1 positions.
                assert cache["lengths"] == [_PROMPT + _NEW - 1] * (stop - start), (
                    where, cache)
                assert all(s == [_BATCH, 6, _PROMPT + _NEW - 1, 8]
                           for s in cache["key_shapes"]), (where, cache)
                assert rep["facts"]["layer_idx"] == list(range(stop - start)), where


def test_only_hidden_states_and_token_ids_cross_ranks_as_read_from_the_backend():
    """The backend's own record, compared with the protocol's exact list."""
    for name, (world, _, _) in sorted(_WORLDS.items()):
        for rep in _pp(True, name):
            got = rep["traffic"]
            want = _expected_traffic(rep["rank"], world, _NEW)
            assert got == want, (
                "%s rank %d: the backend saw\n%r\nexpected\n%r"
                % (name, rep["rank"], got, want))
            # Refused generate calls must not have sent or posted anything.
            assert rep["traffic_after_refusals"] == len(want), (name, rep["rank"])


def test_each_pipeline_rank_reads_only_its_layers_from_the_checkpoint():
    for name, (world, _, tied) in sorted(_WORLDS.items()):
        file_keys = _checkpoint(tied)
        ranges = _ranges(name)
        for shim in (True, False):
            for rep in _pp(shim, name):
                start, stop = ranges[rep["rank"]]
                where = "%s %s rank %d" % (name, _side(shim), rep["rank"])
                assert rep["ranges"] == ranges, (where, rep["ranges"])
                want = _expected_reads(file_keys, start, stop, tied)
                assert rep["reads"] == want, (
                    "%s: materialised %r, expected exactly %r" % (where, rep["reads"], want))
                _check_stage_facts(rep["facts"], start, stop, tied, file_keys, where)


def test_load_layer_subset_reads_and_allocates_only_the_requested_layers():
    for tied in (False, True):
        file_keys = _checkpoint(tied)
        for shim in (True, False):
            rep = _single(shim, tied)
            # The instrument itself: a full load must record every file key.
            assert sorted(set(rep["full_reads"])) == sorted(file_keys), (
                "the read recorder did not see a full from_pretrained read every key; "
                "it cannot vouch for a subset either", rep["full_reads"])
            wanted = [[a, b] for a, b, t in _LOADER_RANGES if t == tied]
            assert [p["range"] for p in rep["probes"]] == wanted, rep["probes"]
            for probe in rep["probes"]:
                start, stop = probe["range"]
                where = "%s tied=%s layers %d..%d" % (_side(shim), tied, start, stop)
                want = _expected_reads(file_keys, start, stop, tied)
                assert probe["reads"] == want, (where, probe["reads"], want)
                assert probe["value_mismatch"] == [], (where, probe["value_mismatch"])
                assert probe["cache_entries"] == stop - start, (where, probe["cache_entries"])
                _check_stage_facts(probe, start, stop, tied, file_keys, where)


def test_what_is_not_built_refuses_by_name():
    for shim in (True, False):
        ref = _single(shim, False)["refusals"]
        expect = {
            "empty_range": ("UnsupportedLayerSubset", "at least one layer"),
            "reversed_range": ("UnsupportedLayerSubset", "at least one layer"),
            "outside_range": ("UnsupportedLayerSubset", "outside a model of 4"),
            "non_contiguous_layers": ("UnsupportedLayerSubset", "non-contiguous"),
            "device_map": ("UnsupportedLayerSubset", "`device_map=`"),
            "split_empty_stage": ("UnsupportedPipeline", "leaves a stage with no layers"),
            "split_gap": ("UnsupportedPipeline", "contiguous"),
            "split_overlap": ("UnsupportedPipeline", "contiguous"),
            "split_short": ("UnsupportedPipeline", "covers layers 0..2"),
            "split_one_stage": ("UnsupportedPipeline", "model itself"),
            "split_wrong_count": ("UnsupportedPipeline", "3 stages but the world has 2"),
            "stage_without_group": ("UnsupportedPipeline", "not initialised"),
        }
        for key, (cls, words) in expect.items():
            got = ref[key]
            assert got is not None and got.startswith(cls + ":") and words in got, (
                _side(shim), key, got)
        assert ref["split_hook"] == [[0, 3], [3, 4]], ref["split_hook"]
        for name in ("even2", "three"):
            for rep in _pp(shim, name):
                g = rep["generate_refusals"]
                for key, words in (("sampling", "do_sample=True"),
                                   ("beam_search", "beam search"),
                                   ("streamer", "`streamer=`"),
                                   ("no_cache", "use_cache=False")):
                    assert g[key] and g[key].startswith("UnsupportedPipeline:") and \
                        words in g[key], (_side(shim), name, rep["rank"], key, g[key])


def test_sending_a_stale_hidden_state_changes_the_tokens():
    """Sabotage 1: stage 0 resends the previous decode step's hidden state."""
    for name in ("even2", "three"):
        world, _, tied = _WORLDS[name]
        for shim in (True, False):
            ref = _single(shim, tied)["sequences"]
            for rep in _pp(shim, name):
                assert rep["stale_sequences"] != ref, (
                    "%s %s rank %d: stage 0 sent stale hidden states and the tokens "
                    "did not change; the token comparison cannot see a wrong boundary"
                    % (name, _side(shim), rep["rank"]))
                # Prompt and the first two new tokens precede the first stale send.
                assert [r[:_PROMPT + 2] for r in rep["stale_sequences"]] == \
                    [r[:_PROMPT + 2] for r in ref], (name, _side(shim), rep["rank"])


def test_a_stage_that_drops_its_cache_updates_is_caught_twice():
    """Sabotage 2, both ways: the position check names it; without the check the tokens move."""
    bad = _pp_desync()[1]
    assert bad.get("desync_error"), bad
    assert bad["desync_error"].startswith("PipelineDesync:"), bad["desync_error"]
    assert "stage 1 (layers (1, 3))" in bad["desync_error"], bad["desync_error"]
    for name in ("even2", "three"):
        world, _, tied = _WORLDS[name]
        for shim in (True, False):
            ref = _single(shim, tied)["sequences"]
            for rep in _pp(shim, name):
                assert rep["unchecked_sequences"] != ref, (
                    "%s %s rank %d: a stage that stored no cache produced the "
                    "reference tokens" % (name, _side(shim), rep["rank"]))


def test_every_child_ran_the_torch_it_claims():
    for tied in (False, True):
        for shim in (True, False):
            meta = _single(shim, tied)["_meta"]
            assert meta["shim"] is shim, meta
    for name in sorted(_WORLDS):
        for shim in (True, False):
            for rank, rep in enumerate(_pp(shim, name)):
                assert rep["_meta"]["shim"] is shim and rep["_meta"]["rank"] == rank, rep["_meta"]
                if not shim:
                    # Upstream torch, with only torchnative from the tree.
                    assert not rep["_meta"]["torch_file"].startswith(_CKPT_VENDOR_DIR), rep["_meta"]
                assert rep["torchnative_file"].startswith(_CKPT_VENDOR_DIR), rep["torchnative_file"]


def _main():
    items = [(name, fn) for name, fn in sorted(globals().items())
             if name.startswith("test_") and callable(fn)]
    failures = _skip.run_tests(items, suite="test_ppinfer")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(_main())
