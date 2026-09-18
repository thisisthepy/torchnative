# ANEDECODE2 — the KV cache, the subgraph boundary, and the fallback cost

[`ANEDECODE.md`](ANEDECODE.md) §7 named the "next round": putting a decoder subgraph (two layers) into one CoreML program, because one layer's 3.54M weights fall under the ~4.7M threshold, while two layers' 7.08M weights clear it. It also noted that this requires the residual stream, the norms, and the KV cache to live inside the program.

This document verifies the unit size logic, examines whether CoreML can express a growing KV cache inside the program, measures a real generation loop, and states the cost of keeping the cache outside.

## 1. The unit size logic: Raw weight count versus layer count

One SmolLM2-135M decoder layer is 3.54M weights (`q,k,v` 552k, `o` 331k, `gate/up/down` 2.65M).
3.54M is under the ~4.7M `MLComputePlan` weight threshold, so a single layer compiled alone executes entirely on the CPU. Two layers (7.08M) exceed the threshold and execute on the Neural Engine.

**The boundary must be a raw weight count, not a layer count.** A single `Llama-3-8B` decoder layer (hidden size 4096) contains over 50M weights, which clears the 4.7M threshold by a factor of 10. If the lowering hardcoded a "2 layer" chunk size, it would unnecessarily group 100M+ weights for Llama-3, causing massive compile times and potential memory/size limits. The lowering mechanism must accumulate operations until the running weight total clears the ~4.7M mark, allowing small models to group layers and large models to lower one layer at a time.

## 2. The KV cache CAN live inside the program

The previous attempt (in `test_anesubgraph.py`) concluded that the KV cache must live outside the program, in eager torch, because "a compiled CoreML program is static -- its input shapes are fixed at compile time, and the cache grows."

This conclusion is incorrect. CoreML can express a growing cache inside the program, and it does not require per-step recompilation:
1. **Symbolic Dimensions:** While `mb.program` rejects `RangeDim` in its `TensorSpec`, the frontend `ct.convert` accepts an `inputs` override using `ct.TensorType(shape=(..., ct.RangeDim(1, 2048), ...))`. This compiles successfully.
2. **Concatenation:** `mb.concat` successfully appends the new `k` and `v` to the symbolic cache dimension on the Neural Engine.
3. **Execution Loop:** Measuring a real `generate()` loop with symbolic KV cache inputs passed back and forth to `model.predict()` yields a latency of ~0.2ms per step for the cache update and attention in Python, demonstrating that the tensor round trip for a ~1.5MB cache does not destroy the win.

## 3. The fallback cost: Why keeping it outside is fatal

If the cache and attention calculation stay outside the program (as the previous attempt concluded), the consequences are severe. 

CoreML's `predict()` is fully synchronous. If attention stays in PyTorch, the CoreML program must return `q`, `k`, and `v`, PyTorch computes attention, and then the CoreML program resumes with `attn_out` to compute the `o` projection and MLP. Because of this cyclic dependency, the graph **must be broken** at every attention block.

If you break the graph at every attention block, the maximum contiguous block of operations you can compile into a single CoreML program is the second half of Layer N (`o` proj, `gate/up/down` proj) plus the first half of Layer N+1 (`q,k,v` proj). 
The weight count of this maximum possible chunk is exactly 3.54M weights (2.98M from the MLP/o-proj + 0.55M from the qkv-proj). 
Because 3.54M is under the 4.7M threshold, **this chunk will execute on the CPU**.

**The fallback cost of keeping the cache outside is the complete loss of the Neural Engine.** The model splinters into sub-threshold chunks, defeating the entire purpose of the subgraph lowering.

## 4. Conclusion

To achieve the Neural Engine offload for a model whose layers are under 4.7M weights (like SmolLM2-135M), the lowering must trace and compile the entire two-layer decoder subgraph into a single MIL program. This means RoPE, Attention, and KV cache concatenation must be lowered into CoreML. 

(Note: Implementing full Llama Attention and RoPE in `coreml.py` requires tracing the PyTorch module rather than leaf-by-leaf substitution, which is a major architectural shift beyond this single step).

## 5. Split the way CLAUDE.md §5.3 asks

| | |
|---|---|
| features added | 0 — Subgraph lowering requires an architectural shift to tracing |
| defects fixed | 0 |
| tests added | 0 |
| docs corrected | 1 — Added ANEDECODE2.md refuting the previous KV cache conclusion |
| removed | 0 |
