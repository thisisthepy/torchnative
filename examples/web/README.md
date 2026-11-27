# Web example

Runs torchnative in a browser tab, on the WASM `cpu` build, inside Pyodide. It does not run the
common task (`../README.md`): SmolLM2-135M's download and memory do not fit a tab's wasm32 heap.
It runs tensor basics instead and checks every answer against upstream torch.

| File | What it is |
|---|---|
| `index.html` | The page. It loads Pyodide 314.0.8 from the jsDelivr CDN, installs torchnative's `pyemscripten_2026_0_wasm32` wheel with micropip, and runs `basics.py` |
| `basics.py` | The cases and the comparison. The page and `run.mjs` both run it |
| `reference.json` | Upstream torch 2.13.0's answers, written by `reference.py` on a host |
| `serve.py` | Serves the page over loopback, opens it, and prints the page's result |
| `run.mjs` | The same run in Node, with the same Pyodide (`package.json` pins it) |

## The cases

Eleven cases, each a small use of the API a page would make: creation, elementwise maths,
reductions, `argmax` and `argsort`, matmul, indexing, integer ops, the seeded generators
(`rand`, `randn`, `randint`, `randperm`), `nn` layers (`Linear`, `Conv1d`, softmax, layer norm,
GELU), autograd through a `Linear`, and float16 and bfloat16 round trips.

- `exact` cases (integers, booleans, shapes, dtypes, the seeded generators) must equal upstream's
  value exactly. Python's types count too, so `3` is not `3.0`.
- `float` cases must lie within a tolerance derived from upstream itself (AGENTS.md §16). Per
  element, it is the larger of upstream's own float32-vs-float64 difference and 8 float32 ulps.
  `reference.py` runs each case in float64 to find it. Nothing is chosen by hand.

A case that raises is reported with its error, and nothing replaces it. The script refuses to run
on a `torch` that is not torchnative's.

## Run

In a browser:

```sh
uv run python serve.py                 # opens the page; torchnative comes from PyPI
uv run python serve.py --no-browser    # prints the URL; open it yourself
```

The page needs a server: it fetches `basics.py` and `reference.json`, and a page opened as a file
cannot. Any static server works; `serve.py` also waits for the page's result, prints it, and exits
with the page's status.

In Node (22 or later):

```sh
npm install
node run.mjs
```

To try a wheel built from a checkout (`scripts/wheel/build.py --target wasm32-emscripten`), pass it
to either runner with `--wheel <path>`.

Neither runner stubs a module before `import torch`. Pyodide's CPython has no `_multiprocessing`
and no `_posixshmem`, and upstream's `torch/__init__.py` imports `torch.multiprocessing`, which
needs both. The shim stands in for them on Emscripten, as it does on Android (issue #437): the
semaphore module is empty and the two shared-memory functions refuse by name. The `PLATFORM` line
prints which stand-ins were installed.

torchnative 0.1.0b11 on PyPI does not carry these stand-ins yet: its `import torch` stops with
`ModuleNotFoundError: No module named '_multiprocessing'`, and both runners exit 2. The first
release after 0.1.0b11 carries them. Until then, use `--wheel` with a wheel built from `develop`.

## Expected output

```
INSTALLED 0.1.0b11 from ...
PLATFORM {"sys_platform": "emscripten", "python": "3.14.2", "torch": "2.13.0", "torch_C": "/lib/python3.14/site-packages/torch/_C.abi3.so", "ipc_stand_ins": ["_multiprocessing", "_posixshmem"]}
BASICS {"case": "creation", "kind": "float", "agrees_with_upstream": true}
...
SUMMARY {"cases": 11, "agree": 11, "failed": [], "upstream_torch": "2.13.0"}
```

Exit status: 0 when every case agrees, 1 when one disagrees or raises, 2 when torch cannot be
installed or imported.

## Last recorded result

Mac mini M1, macOS 26.5.1, 2026-11-27. The wheel was built from branch `feat/437-examples-web`
(develop 8f11083d plus the Emscripten stand-ins) with `scripts/wheel/build.py --target
wasm32-emscripten`.

| Runner | Pyodide | Cases equal to upstream |
|---|---|---|
| Naver Whale 5.39.412.57 (Chromium), through `serve.py --wheel` | 314.0.8 (CPython 3.14.2) | 11 of 11 |
| Node 22.23.2, through `run.mjs --wheel` | 314.0.8 (CPython 3.14.2) | 11 of 11 |
| Node 22.23.2, `run.mjs` with 0.1.0b11 from PyPI | 314.0.8 (CPython 3.14.2) | not reached: exit 2 at `import torch` (`_multiprocessing`) |

These are agreement results, not measurements. Nothing here measures speed.
