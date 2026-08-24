# torchnative

**Run the real PyTorch ecosystem on device — not a reimplementation of it.**

`torchnative` replaces PyTorch's compiled core (`torch._C`) with a native extension, so the
genuine `torch` and `transformers` Python packages run on a phone the same way they run on a
workstation. Models are not ported, converted, or re-expressed in another framework. They are
imported.

On top of that substrate it provides on-device learning — Test-Time Training, Test-Time
Adaptation and Federated Learning — because a model that ships to a device should be able to keep
improving there.

> **Status: pre-alpha, under active development.** The substrate runs real transformer
> architectures and matches upstream PyTorch numerically, but several load-bearing paths are not
> finished. See [Status](#status) before depending on anything here.

---

## Why not a reimplementation

Every other route to on-device inference re-expresses the model somewhere else. `llama.cpp`
rewrites architectures in C++. ExecuTorch and CoreML ahead-of-time compile a frozen graph. MLC
lowers to its own runtime. Each conversion is a place where a model can quietly stop being the
model you trained, and each new architecture is a porting task someone has to do first.

The reason nobody runs the real thing is that `torch._C` cannot be built for mobile. PyTorch's
own build sets `INTERN_BUILD_MOBILE` for any Android or iOS toolchain, and that path forces
`BUILD_PYTHON` off. The mobile build is structurally incapable of producing the Python extension
module that the Python package needs.

`torchnative` supplies that module instead. Everything above it — `torch/`, `transformers/`, your
training loop — is upstream source, unmodified.

The trade is explicit: this is harder than a reimplementation, and it is the only approach where
a new architecture costs nothing, because the architecture was never ported in the first place.

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

Three properties hold this together.

**One door.** Every operator reaches its kernel through `_aten_dispatch`, and nothing bypasses
it. That makes the surface measurable — an unimplemented operator names itself instead of failing
somewhere downstream — and it gives graph capture, which NPU backends will need, exactly one
place to attach.

**Demand-driven.** Nothing is implemented because it might be needed. The shim refuses by name,
the refusal names the next thing to build, and that list comes from running real models rather
than from reading headers.

**Stable ABI.** The extension is built against CPython's limited API (`abi3-py313`), so one
binary loads on 3.13, 3.14 and later without a rebuild.

---

## Status

The substrate is real and measured. The paths above it are not finished.

### Working

| | |
|---|---|
| ATen operators | **91**, each compared against upstream |
| Golden comparison cases | **2095 / 2095** passing — values, shapes and dtypes |
| Python spellings | **204** entries verified against upstream signatures |
| Architectures with no missing operators | **15 of 20** measured |
| Build targets | macOS/Linux, `aarch64-linux-android`, `aarch64-apple-ios` |

Llama and GPT-2 produce **the same tokens and the same logits** as upstream PyTorch, for greedy
decoding and for sampling. `uniform_` and `normal_` are **bit-identical** to upstream, and
`multinomial` consumes the same generator stream — so a seeded run reproduces exactly.

Architectures currently complete: Llama, GPT-2, Qwen2, Mistral, Gemma, GPT-NeoX, OPT, MPT,
StarCoder2, Persimmon, Cohere, StableLM, OLMo, Phi, BERT.

### Not working yet

- **`import transformers` fails.** `torch.distributed` is not implemented, and an unguarded
  import inside `torch._dynamo` reaches `dist.Store`. Every result above was therefore measured
  against models transcribed by hand rather than loaded through `transformers`.
- **No checkpoint has ever been loaded.** All weights so far are randomly initialised;
  `from_pretrained` and the `torch.load` path are untested.
- **Mobile is link-verified only.** The Android and iOS artefacts build and link, but no device
  has loaded one.
- **CPU only.** No GPU or NPU backend yet.

These are tracked in [`docs/DESIGN.md`](docs/DESIGN.md) §11.1, with the measurements behind them.

---

## Verification

Correctness here means *agreeing with upstream PyTorch*, so the strategy is comparison rather
than assertion.

**Golden comparison.** Every implemented operator runs on both upstream torch and this shim, and
is compared on value, shape and dtype. It has caught real defects — a `float16` GEMM accumulating
in `float16` where torch accumulates in `float32`, `cumsum` routed through the wrong kernel,
integer overflow where torch refuses.

**The harness tests itself.** `compare.py --self-test` injects a fault shaped like a plausible
misimplementation at each comparator and fails if the comparator accepts it: 11 comparators
against 11 fault modes, with any comparator that was never exercised reported as a failure. That
was worth building — it found that the previous fault injection reached exactly one case out of
1781.

**Tokens are not enough.** A wrong `gelu` approximation produced *identical tokens* while the
logits differed by 5.9e-04. End-to-end tests therefore compare logits too, with a tolerance
measured to sit between normal float32 noise and that failure.

```sh
sh rust/torch_c/pytests/run.sh                    # smoke tests + harness self-test
python tools/golden/compare.py                    # golden comparison against upstream
python rust/torch_c/pytests/verify_schemas.py     # signature tables vs upstream
```

---

## Roadmap

The next milestone is the device abstraction, because everything else waits on it — a distributed
rank needs a device to point at, and every accelerator attaches there.

```
torchnative.nn.federated    rounds · client selection · aggregation · dropout
  └ torch.distributed       ProcessGroup · collectives (transport)
      └ backends            ours, via register_backend
          └ devices         CPU · Metal · Vulkan · NPU
```

`torch.distributed` is on that list for a reason beyond unblocking imports: federated averaging
*is* collective communication — broadcast the model, gather the updates, weighted all-reduce. The
subsystem that looked irrelevant to an on-device library turns out to be the foundation of its
federated axis.

NPU support is the genuinely hard one. ANE, NNAPI and QNN take a whole graph ahead of time, which
is at odds with dispatching one operator at a time, so it needs a capture layer rather than
another device. The single door is where that attaches.

---

## Repository layout

```
torchnative/     the Python library
rust/torch_c/    the torch._C replacement (Rust, PyO3, candle)
tools/golden/    the upstream comparison harness
vendor/          scripts that assemble the vendored torch tree (not checked in)
docs/            design, measurements, and the reasoning behind open decisions
```

`docs/` is written to be read. It records what was measured, what was assumed, and where an
earlier conclusion turned out to be wrong — corrections are left visible rather than edited away.
Start with [`DESIGN.md`](docs/DESIGN.md); [`SURFACE_HONESTY.md`](docs/SURFACE_HONESTY.md) and
[`HARNESS.md`](docs/HARNESS.md) are good examples of the standard the rest aims for.

---

## Building

Requires a Rust toolchain and CPython 3.13 or later.

```sh
bash vendor/vendor_torch.sh     # assemble the vendored torch tree
bash vendor/install_shim.sh     # build the extension and install it
```

Cross-compilation for Android and iOS is documented in
[`docs/RUST_CROSSBUILD.md`](docs/RUST_CROSSBUILD.md), including the PyO3 configuration iOS needs
in order not to link `libpython`.

---

## Related projects

- [PythonMultiplatform](https://github.com/thisisthepy/PythonMultiplatform) — embeds CPython 3.13
  into Kotlin Multiplatform; the deployment target for this library
- [pypackpack](https://github.com/thisisthepy/pypackpack) — the build tool
- [Hugging Face `kernels`](https://github.com/huggingface/kernels) — the fused-kernel contract
  this adopts, with resolution moved from runtime download to build time, since downloading
  executable code is not permitted on every target platform

---

## License

MIT — see [LICENSE](LICENSE).

PyTorch is vendored under its own BSD-3-Clause license. The vendored tree is assembled at build
time and is not redistributed in this repository.
