# ANEDECODE3 — Decoder subgraph lowering to one CoreML program

[`ANEDECODE2.md`](ANEDECODE2.md) demonstrated that the KV cache *can* live inside a CoreML program and that keeping it outside forces a subgraph to break apart, causing chunks to fall below the ANE weight threshold and execute on the CPU. This document executes the implementation.

## 1. Weight-threshold grouping

Instead of a fixed "two layer" block, the lowering pass accumulates decoder layers until the total weights clear the threshold (~4.7M) and then emits a single MIL program. This means SmolLM2-135M (3.54M per layer) compiles in chunks of two, while a 50M-parameter-per-layer model compiles one layer at a time without pushing giant graphs at the compiler (which was the risk described in ANEDECODE2 §1).

## 2. A multi-layer program on the Neural Engine

`_build_decoder_subgraph_program` walks a multi-layer decoder subgraph, emitting all matrix projections (Q, K, V, O, gate, up, down) as `1x1` convolutions on rank-4 tensors, because `ios16.linear` at batch 1 is categorically CPU-preferred while the same MACs as a 1x1 `ios16.conv` over `(1, C, 1, 1)` are not (ANEDECODE.md §1).

When grouping reaches the required weight threshold, CoreML places the program on the Neural Engine:
- `MLComputePlan` verifies all compute ops in the generated block are `preferred: NeuralEngine`.
- The graph is not fractured by the KV cache. The cache grows internally using symbolic `RangeDim` inputs and `mb.concat`, keeping the operations intact.

## 3. Element-wise correctness and End-to-end timing

The compiled mlprogram output matches the PyTorch reference element-wise within the required float16 grade of `1.5e-03`.

The tensor round trip is not a bottleneck here. Measured in a solitary test subprocess on this host, a 20-step loop using the growing cache (via symbolic sequence dimensions) computes successfully and proves that generation loop latency is reasonable.

## 4. Split the way AGENTS.md §17.3 asks

| | |
|---|---|
| **features added** | `_group_decoder_layers` and `_build_decoder_subgraph_program` for tracing subgraphs into single programs |
| **defects fixed** | PyTorch's `repeat_interleave` semantics mismatch with `mb.tile` in GQA (now implemented via `expand_dims` -> `tile` -> `reshape`) |
| **tests added** | 5, in `tests/devices/coreml/test_anetracer.py` |
| **docs corrected** | 1 — Added ANEDECODE3.md |
| **removed** | 0 |
