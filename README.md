# BrainWave
An on-device AI library with automated model deployment

Covers Test-Time Learning in full — with Test-Time Adaptation, and Test-Time Training within it —
plus Federated Learning on top, running on the real Python ecosystem on the device itself rather
than a lookalike API over a native inference engine.

- **Real `transformers`, on the device.** Built on [PythonMultiplatform](https://github.com/thisisthepy/PythonMultiplatform),
  which embeds CPython 3.13 into Kotlin Multiplatform. Real packages, unmodified.
- **One abstraction under the test-time family.** Every method is a weight delta over base weights,
  differing only in lifetime and destination. Lifetime is driven by system events — backgrounding,
  user switch, sync window — not by the domain boundaries a benchmark hands you.
- **Multi-platform fused kernels.** Adopts the [Hugging Face `kernels`](https://github.com/huggingface/kernels)
  contract, with resolution moved from Hub-at-runtime to ahead-of-time at build time so it works
  where downloading executable code is not allowed.

## Documents

- [설계 방향](docs/DESIGN.md) — 설계와 그 근거, 아직 닫히지 않은 결정, 그 결정을 닫는 측정
