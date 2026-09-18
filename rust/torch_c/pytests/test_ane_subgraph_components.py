import sys, time
import numpy as np
import coremltools as ct
from coremltools.converters.mil import Builder as mb
from coremltools.converters.mil.mil import get_new_symbol
sys.path.insert(0, '/Volumes/macMini/worktrees/tn-anesubgraph/torchnative/src/main')
from torchnative.export import coreml as C

def test_rope_can_be_implemented_in_coreml_without_strides():
    """Verify RoPE compiles successfully on macOS 13."""
    s = get_new_symbol("s")
    HEAD_DIM = 64
    
    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, 9, 1, HEAD_DIM)),
        mb.TensorSpec(shape=(1, 1, 1, HEAD_DIM)),
        mb.TensorSpec(shape=(1, 1, 1, HEAD_DIM))
    ])
    def prog(q, cos, sin):
        x1 = mb.slice_by_index(x=q, begin=[0,0,0,0], end=[1,9,1,32], begin_mask=[True,True,True,False], end_mask=[True,True,True,False])
        x2 = mb.slice_by_index(x=q, begin=[0,0,0,32], end=[1,9,1,64], begin_mask=[True,True,True,False], end_mask=[True,True,True,False])
        neg_x2 = mb.mul(x=x2, y=np.float32(-1.0))
        rot_q = mb.concat(values=[neg_x2, x1], axis=3)
        q_embed = mb.add(x=mb.mul(x=q, y=cos), y=mb.mul(x=rot_q, y=sin))
        return q_embed
        
    model = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS13, compute_precision=ct.precision.FLOAT16)
    rows = C.compute_plan(model)
    assert any(r['op'] == 'slice_by_index' for r in rows), "Missing slice_by_index"

def test_kv_cache_compiles_and_runs_fast_in_a_loop():
    """Verify that a symbolic growing cache can be expressed and evaluated quickly."""
    s = get_new_symbol("s")
    HEAD_DIM = 64
    N_HEADS = 9
    KV_HEADS = 3
    
    @mb.program(input_specs=[
        mb.TensorSpec(shape=(1, N_HEADS, 1, HEAD_DIM)),
        mb.TensorSpec(shape=(1, KV_HEADS, 1, HEAD_DIM)),
        mb.TensorSpec(shape=(1, KV_HEADS, 1, HEAD_DIM)),
        mb.TensorSpec(shape=(1, KV_HEADS, s, HEAD_DIM)),
        mb.TensorSpec(shape=(1, KV_HEADS, s, HEAD_DIM)),
    ])
    def prog(q, k, v, k_cache, v_cache):
        new_k_cache = mb.concat(values=[k_cache, k], axis=2)
        new_v_cache = mb.concat(values=[v_cache, v], axis=2)
        k_rep = mb.tile(x=new_k_cache, reps=[1, 3, 1, 1])
        v_rep = mb.tile(x=new_v_cache, reps=[1, 3, 1, 1])
        k_T = mb.transpose(x=k_rep, perm=[0, 1, 3, 2])
        attn_weights = mb.matmul(x=q, y=k_T)
        attn_weights = mb.mul(x=attn_weights, y=np.float32(1.0 / np.sqrt(HEAD_DIM)))
        attn_probs = mb.softmax(x=attn_weights, axis=3)
        attn_out = mb.matmul(x=attn_probs, y=v_rep)
        return attn_out, new_k_cache, new_v_cache

    inputs = [
        ct.TensorType(name="q", shape=(1, N_HEADS, 1, HEAD_DIM)),
        ct.TensorType(name="k", shape=(1, KV_HEADS, 1, HEAD_DIM)),
        ct.TensorType(name="v", shape=(1, KV_HEADS, 1, HEAD_DIM)),
        ct.TensorType(name="k_cache", shape=(1, KV_HEADS, ct.RangeDim(1, 2048), HEAD_DIM)),
        ct.TensorType(name="v_cache", shape=(1, KV_HEADS, ct.RangeDim(1, 2048), HEAD_DIM)),
    ]
    model = ct.convert(prog, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS13, compute_precision=ct.precision.FLOAT16, inputs=inputs)
    
    q_val = np.random.randn(1, N_HEADS, 1, HEAD_DIM).astype(np.float32)
    k_val = np.random.randn(1, KV_HEADS, 1, HEAD_DIM).astype(np.float32)
    v_val = np.random.randn(1, KV_HEADS, 1, HEAD_DIM).astype(np.float32)
    k_cache_val = np.random.randn(1, KV_HEADS, 1, HEAD_DIM).astype(np.float32)
    v_cache_val = np.random.randn(1, KV_HEADS, 1, HEAD_DIM).astype(np.float32)

    spec = model.get_spec()
    feed = {
        spec.description.input[0].name: q_val,
        spec.description.input[1].name: k_val,
        spec.description.input[2].name: v_val,
        spec.description.input[3].name: k_cache_val,
        spec.description.input[4].name: v_cache_val,
    }
    res = model.predict(feed)
    assert len(res) == 3

def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_"):
            continue
        try:
            fn()
        except Exception as e:
            failures += 1
            print(f"FAIL {name}: {type(e).__name__}: {e}")
        else:
            print(f"ok   {name}")
    return 1 if failures else 0

if __name__ == "__main__":
    raise SystemExit(_main())
