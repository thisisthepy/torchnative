<div align="center">

# torchnative

**Run the real PyTorch ecosystem on device — not a reimplementation of it.**

[![PyPI](https://img.shields.io/pypi/v/torchnative?color=blue)](https://pypi.org/project/torchnative/)
[![Python](https://img.shields.io/badge/python-3.13%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Platforms](https://img.shields.io/badge/platforms-macOS%20%7C%20Android%20%7C%20iOS-lightgrey)](#install)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange)](#status)

</div>

---

`torchnative` replaces PyTorch's compiled core — `torch._C` — with a native extension, so the
genuine `torch` and `transformers` packages run on a phone the way they run on a workstation.

Models are not ported, converted, or re-expressed. They are imported.

```python
from transformers import AutoModelForCausalLM     # the real one
model = AutoModelForCausalLM.from_pretrained("...")
model.generate(...)                                # on the device
```

> [!WARNING]
> **Pre-alpha.** The operator layer matches upstream PyTorch numerically, 18 of 20 tested
> architectures reach zero missing operators, real checkpoints load, and an Android device runs
> the built artefact — but `import transformers` does not work yet and there is no accelerator
> backend. See [Status](#status) before depending on this.

---

## Why not a reimplementation

Every other route to on-device inference re-expresses the model somewhere else.

| | approach | cost |
|---|---|---|
| llama.cpp | architectures rewritten in C++ | each new architecture is a porting task |
| ExecuTorch · CoreML | ahead-of-time compiled graph | export step, and what runs is not what you wrote |
| MLC | lowered to its own runtime | same |
| **torchnative** | **the real Python package** | **the substrate is hard; architectures are free** |

The reason nobody runs the real thing is that `torch._C` cannot be built for mobile. PyTorch's own
build sets `INTERN_BUILD_MOBILE` for any Android or iOS toolchain, and that path forces
`BUILD_PYTHON` off — so the mobile build is structurally incapable of producing the Python
extension module the Python package needs.

`torchnative` supplies that module instead. Everything above it is upstream source, unmodified.

---

## What it does

### 1 · LLM inference

Run `transformers` models directly. No conversion step, no per-architecture port — if
`transformers` supports it and the operators are covered, it runs.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-1B")
tok   = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")

out = model.generate(**tok("On-device inference is", return_tensors="pt"),
                     max_new_tokens=32, do_sample=True)
```

**Today:** this runs. `SmolLM2-135M` is pulled from the Hub through `from_pretrained` — 273
tensors, weights bit-identical to upstream — and `generate` emits the same twenty tokens upstream
does when the model is loaded in `float32`.

Loaded in the checkpoint's native `bfloat16`, which is what you get if you pass no `dtype`, the
tokens diverge. That is not a defect to fix: upstream disagrees with *itself* on one prompt in
three under a mathematically equivalent change of accumulation order, so bitwise agreement in
bf16 is not a bar any independent implementation can clear. See [Status](#status).

### 2 · Federated learning

Devices train locally and share updates, not data. Federated averaging *is* collective
communication, so this is built on `torch.distributed` rather than beside it — broadcast the
model, gather the updates, weighted all-reduce.

```python
from torchnative.nn import federated

engine = federated.Engine(model, rounds=..., aggregator=federated.FedAvg())
engine.participate()          # local epochs, then contribute a delta
```

**Today:** the transport under it stands. `torch.distributed` works at `world_size = 1`, and its
sixteen value-producing collectives are byte-identical to upstream's `gloo`; what needs a second
rank refuses by name rather than pretending. `torchnative.nn.federated` above it is still empty.

### 3 · Test-time adaptation & training

A model that ships to a device meets data the training set never had. TTA, TTT and the wider
test-time learning family let it adapt in place — and every method reduces to the same thing: a
weight delta over base weights, differing only in lifetime and destination.

```python
from torchnative import adapt

model = adapt.wrap(model, method=adapt.Tent())   # or TTT, memory-based, entropy-based
model.online()                                   # adapt as it serves
```

**Today:** planned; the delta abstraction is specified in [`docs/DESIGN.md`](docs/DESIGN.md) §3.
Lifetime is driven by system events — backgrounding, user switch, sync window — rather than by
the domain boundaries a benchmark hands you.

---

## How it works

```
your code · transformers · torch/*.py        upstream Python, unmodified
──────────────────────────────────────────
torch._C                                     ← replaced
  ├── _aten_dispatch                         the single door every operator passes
  ├── Python spellings                       torch.mm, x.softmax(), F.linear, ...
  └── kernels                                Rust, backed by candle
──────────────────────────────────────────
CPU today · Metal, Vulkan, NPU planned
```

**One door.** Every operator reaches its kernel through `_aten_dispatch`, and nothing bypasses
it. That makes the surface measurable — an unimplemented operator names itself rather than
failing downstream — and it gives graph capture, which NPU backends will need, exactly one place
to attach.

**Demand-driven.** Nothing is implemented because it might be needed. The shim refuses by name,
the refusal names the next thing to build, and that list comes from running real models.

**Stable ABI.** Built against CPython's limited API (`abi3-py313`), so one binary per platform
loads on 3.13, 3.14 and later without a rebuild.

---

## Status

<table>
<tr><th align="left">Working</th><th align="left"></th></tr>
<tr><td>ATen operators</td><td><b>119</b>, each compared against upstream</td></tr>
<tr><td>Golden comparison cases</td><td><b>2811 / 2811</b> — values, shapes, dtypes</td></tr>
<tr><td>Signature and schema tables</td><td><b>4203</b> entries checked against upstream</td></tr>
<tr><td>Architectures complete</td><td><b>19 of 20</b> measured — Mixtral needs <code>_grouped_mm</code> alone</td></tr>
<tr><td>Checkpoints</td><td><code>torch.load</code> and safetensors, round-tripped against upstream</td></tr>
<tr><td>Build targets</td><td>macOS arm64 · Android arm64 · iOS arm64 — <b>Linux and Windows are in the target matrix and not yet built</b> (<a href="docs/DESIGN.md">DESIGN.md</a> §722)</td></tr>
<tr><td>Devices run</td><td>Android arm64 — <code>import torch</code>, 119 ops, <code>nn</code> forward</td></tr>
</table>

Complete: Llama · GPT-2 · Qwen2 · Mistral · Gemma · GPT-NeoX · OPT · MPT · StarCoder2 ·
Persimmon · Cohere · StableLM · OLMo · Phi · BERT · Falcon · BLOOM · GPT-BigCode

`uniform_` and `normal_` are **bit-identical** to upstream, and `multinomial` consumes the same
generator stream — a seeded run reproduces exactly.

**Not working yet**

- Mixtral is the one incomplete architecture — `_grouped_mm`, an offset-based grouped GEMM
  with no equivalent here yet.
- `torch.compile` does not work; it stops inside Dynamo. Eager execution is the supported path.
- CPU only. No GPU or NPU backend.
- The Android run is an emulator, not a phone. No number here describes real silicon.
- Apple is much faster than Android at `f32` matmul, and that is the hardware. Accelerate
  reaches the AMX coprocessor; ARMv8.2-A NEON has no equivalent. Our Android throughput equals
  our own throughput on the same core under the same backend, at 88% of that core's NEON peak —
  so the kernels are not the gap. Upstream PyTorch has no Android wheel, so how we compare to it
  *there* is unmeasured. See [`docs/PERF_ANDROID.md`](docs/PERF_ANDROID.md).

Tracked with the measurements behind them in [`docs/DESIGN.md`](docs/DESIGN.md) §11.1.

---

## Verification

Correctness here means *agreeing with upstream PyTorch*, so the strategy is comparison rather
than assertion.

| | |
|---|---|
| **Golden comparison** | Every operator runs on both upstream torch and this shim, compared on value, shape and dtype. It has caught a `float16` GEMM accumulating in `float16` where torch accumulates in `float32`, `cumsum` routed through the wrong kernel, and integer overflow where torch refuses. |
| **The harness tests itself** | `--self-test` injects a fault shaped like a plausible misimplementation at each comparator and fails if the comparator accepts it — 11 comparators × 11 fault modes, with any comparator never exercised reported as failure. It found that the previous fault injection reached exactly one case out of 1781. |
| **Tokens are not enough** | A wrong `gelu` approximation produced *identical tokens* while logits differed by 5.9e-04. End-to-end tests compare logits too, with a tolerance measured to sit between normal float32 noise and that failure. |

```sh
sh rust/torch_c/pytests/run.sh                  # smoke tests + harness self-test
python tools/golden/compare.py                  # golden comparison against upstream
python rust/torch_c/pytests/verify_schemas.py   # signature tables vs upstream
```

---

## Roadmap

The next milestone is the device abstraction, because everything waits on it — a distributed rank
needs a device to point at, and every accelerator attaches there.

```
torchnative.nn.federated    rounds · client selection · aggregation · dropout
  └ torch.distributed       ProcessGroup · collectives (transport)
      └ backends            ours, via register_backend
          └ devices         CPU · Metal · Vulkan · NPU
```

| | |
|---|---|
| **Device abstraction** | `torch.device`, per-device dispatch. Everything else waits on it. |
| **Metal** | candle already has the backend; disabled here for build isolation, not absent. |
| **`torch.distributed`** | From `world_size = 1` upward. Unblocks `transformers` as a side effect. |
| **Vulkan** | No candle backend and no `vulkan` slot in the `kernels` contract — genuinely new work. Wiring and correctness are testable on an emulator; only the performance question needs a phone. |
| **NPU** | NNAPI, CoreML and QNN compile at runtime, so no export step is added — but they take a whole subgraph, not one operator. That needs a capture layer, and the single door is where it attaches. |

---

## Install

```sh
pip install torchnative
```

`0.0.2a0` ships three platform wheels, all `cp313-abi3` — one binary per platform, loadable by
CPython 3.13 and every later release. Each carries the `_C` extension and the vendored upstream
tree, so `import torch` resolves to *this* build.

**They are not all verified to the same depth, and the table says which is which.**

| wheel | built | installed | `import torch` | computes |
|---|:--:|:--:|:--:|:--:|
| `macosx_11_0_arm64` | ✅ | ✅ | ✅ | ✅ |
| `android_21_arm64_v8a` | ✅ | ✅ | ✅ | ✅ |
| `ios_12_0_arm64_iphoneos` | ✅ | — | — | — |

macOS is checked in a clean virtualenv and Android on a device, unpacked into its CPython's
`site-packages` — in both, `torch.__file__` lands inside the install, `aten.mm` returns the right
answer and an `nn.Linear` forward runs ([`docs/WHEEL.md`](docs/WHEEL.md) §7).

> [!IMPORTANT]
> **The iOS wheel has never been executed.** What is verified is everything short of running it:
> its 222 undefined symbols all resolve against the device `Python.framework` and the iOS SDK,
> checked through the two-level namespace bindings dyld itself uses, and every file in it outside
> the extension is byte-identical to the simulator wheel, which does import and compute. What is
> not verified is the load itself, `@rpath` resolution inside a real app bundle, and code signing
> — none of which can be answered without a device ([`docs/IOS.md`](docs/IOS.md)).
>
> If you run it on a phone, we would like to hear either way.

> [!NOTE]
> `0.0.1a0` is still on PyPI and does **not** work — it is `py3-none-any` and carries the
> `torchnative` skeleton alone, no `_C` and no `torch`, so it installs cleanly and then fails to
> import. Ask for `0.0.2a0` or later.
>
> There is no source distribution. Building needs a Rust toolchain and a vendoring step that
> `pip` cannot drive, so an sdist would install and then fail; the recipe is below instead.

### Building from source

Requires a Rust toolchain and CPython 3.13+.

```sh
bash vendor/vendor_torch.sh     # assemble the vendored torch tree
bash vendor/install_shim.sh     # build the extension and install it
```

### Building a wheel

Additionally requires `pip`, `setuptools` and `wheel` in the building interpreter, and a C
compiler for the empty `libtorch_global_deps` (see [`docs/WHEEL.md`](docs/WHEEL.md) §3.2).

```sh
bash vendor/vendor_torch.sh
bash vendor/install_shim.sh
python tools/wheel/build.py                            # -> dist/*.whl
python tools/wheel/verify.py dist/torchnative-*.whl    # clean venv, real import
```

`verify.py` is the part that matters: it installs into a throwaway virtualenv and asserts that
`torch.__file__` resolves *inside* it. A check that lets the development tree answer proves
nothing about the wheel.

Cross-compilation is documented in [`docs/RUST_CROSSBUILD.md`](docs/RUST_CROSSBUILD.md),
including the PyO3 configuration iOS needs in order not to link `libpython`.

---

## Repository layout

```
torchnative/     the Python library
rust/torch_c/    the torch._C replacement (Rust · PyO3 · candle)
tools/golden/    the upstream comparison harness
tools/wheel/     build a platform wheel, and prove it installs (docs/WHEEL.md)
vendor/          scripts that assemble the vendored torch tree (not checked in)
docs/            design, measurements, and the reasoning behind open decisions
```

`docs/` is written to be read. It records what was measured, what was assumed, and where an
earlier conclusion turned out to be wrong — corrections are left visible rather than edited away.
Start with [`DESIGN.md`](docs/DESIGN.md); [`SURFACE_HONESTY.md`](docs/SURFACE_HONESTY.md) and
[`HARNESS.md`](docs/HARNESS.md) show the standard the rest aims for.

---

## Related

- [PythonMultiplatform](https://github.com/thisisthepy/PythonMultiplatform) — embeds CPython 3.13
  into Kotlin Multiplatform; the deployment target for this library
- [pypackpack](https://github.com/thisisthepy/pypackpack) — the build and bundling tool
- [Hugging Face `kernels`](https://github.com/huggingface/kernels) — the fused-kernel contract
  this adopts, with resolution moved from runtime download to build time, since downloading
  executable code is not permitted on every target platform

---

## License

MIT — see [LICENSE](LICENSE).

PyTorch is vendored under its own BSD-3-Clause license. The vendored tree is assembled at build
time and is not redistributed in this repository.
