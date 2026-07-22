// ═══════════════════════════════════════════════════════════════════════════════
// fused_add_norm — fused residual_add + rms_norm_forward in one kernel
// ═══════════════════════════════════════════════════════════════════════════════
// Per-thread: each thread handles D/tpg elements, keeps merged values in registers.
// Eliminates the global round-trip between the add and the norm.
//
// Grid: [B*S, 1, 1] — one TG per token
// Threads/TG: 256
//
// Shared: 40 B  (1 float per SIMD group × 8 warps)
// ═══════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

inline float warp_sum(float v) {
    v += simd_shuffle_down(v, 16);
    v += simd_shuffle_down(v, 8);
    v += simd_shuffle_down(v, 4);
    v += simd_shuffle_down(v, 2);
    v += simd_shuffle_down(v, 1);
    return v;
}

kernel void fused_add_norm(
    device float*       x_residual [[buffer(0)]],  // h (in-place, read-write)
    device const float* residual   [[buffer(1)]],  // attn_out or ffn_out
    device const float* weight     [[buffer(2)]],  // norm weight [D]
    device float*       output     [[buffer(3)]],  // normalized output
    constant uint&      D          [[buffer(4)]],
    constant float&     eps        [[buffer(5)]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tpg     [[threads_per_threadgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint ln_id   [[thread_index_in_simdgroup]]
) {
    threadgroup float shared_sum[8];
    uint base = row_idx * D;

    // Phase 1: load x and residual, compute merged value in register
    float local_sq = 0.0f;
    for (uint c = tid; c < D; c += tpg) {
        uint idx = base + c;
        float xv = x_residual[idx];
        float rv = residual[idx];
        float nv = xv + rv;
        x_residual[idx] = nv;   // write back for backward
        local_sq += nv * nv;
    }

    // Phase 2: warp-level reduction
    float warp_sq = warp_sum(local_sq);
    if (ln_id == 0) shared_sum[sg_id] = warp_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 3: final reduction, broadcast rms
    float rms = 0.0f;
    if (sg_id == 0) {
        float total = (ln_id < 8) ? shared_sum[ln_id] : 0.0f;
        total = warp_sum(total);
        if (ln_id == 0) rms = rsqrt(total / (float)D + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 4: re-read merged value from x_residual (L1-cached, no global round-trip)
    // and write normalized output
    for (uint c = tid; c < D; c += tpg) {
        uint idx = base + c;
        output[idx] = x_residual[idx] * rms * weight[c];
    }
}
