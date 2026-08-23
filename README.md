# BrainWave
An on-device AI library with automated model deployment

Covers the whole test-time family — Federated Learning, Test-Time Training, Test-Time Adaptation,
and Test-Time Learning — on top of the real Python ecosystem running on the device itself, not a
lookalike API over a native inference engine.

- **Real `transformers`, on the device.** Built on [PythonMultiplatform](https://github.com/thisisthepy/PythonMultiplatform),
  which embeds CPython 3.13 into Kotlin Multiplatform. Real packages, unmodified.
- **One abstraction under the test-time family.** FL, TTT and TTA are all a weight delta over base
  weights, differing only in lifetime and destination. (TTL is not yet defined here — see the
  open question in the design doc.)
- **Multi-platform fused kernels.** Adopts the [Hugging Face `kernels`](https://github.com/huggingface/kernels)
  contract, with resolution moved from Hub-at-runtime to ahead-of-time at build time so it works
  where downloading executable code is not allowed.

## Documents

- [설계 방향](docs/DESIGN.md) — 설계와 그 근거, 아직 닫히지 않은 결정, 그 결정을 닫는 측정
