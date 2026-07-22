// ═══════════════════════════════════════════════════════════════════════════════
// fused_backward_add_norm — BF16 gradients, FP32 weight
// ═══════════════════════════════════════════════════════════════════════════════

#include <metal_stdlib>
using namespace metal;

inline float simd_sum_f(float v) {
    v += simd_shuffle_down(v, 16);
    v += simd_shuffle_down(v, 8);
    v += simd_shuffle_down(v, 4);
    v += simd_shuffle_down(v, 2);
    v += simd_shuffle_down(v, 1);
    return v;
}

kernel void fused_backward_add_norm(
    device const bfloat* grad_output [[buffer(0)]],  // BF16 (128 MB, halved)
    device const bfloat* input       [[buffer(1)]],  // BF16 (h_mid, 128 MB)
    device const float*  weight      [[buffer(2)]],  // FP32 (4 KB, not worth converting)
    device const bfloat* residual    [[buffer(3)]],  // FP32 (grad_output, 128 MB)
    device bfloat*       grad_input  [[buffer(4)]],  // FP32 output (kept FP32 for now)
    constant uint&       D           [[buffer(5)]],
    constant float&      eps         [[buffer(6)]],
    uint row_idx [[threadgroup_position_in_grid]],
    uint tid     [[thread_position_in_threadgroup]],
    uint tpg     [[threads_per_threadgroup]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint ln_id   [[thread_index_in_simdgroup]]
) {
    threadgroup float scratch[8];
    uint base = row_idx * D;

    float local_sq = 0.0f;
    for (uint c = tid; c < D; c += tpg) {
        float x = (float)input[base + c];
        local_sq += x * x;
    }
    float warp_sq = simd_sum_f(local_sq);
    if (ln_id == 0) scratch[sg_id] = warp_sq;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float rms;
    if (sg_id == 0) {
        float total = (ln_id < 8) ? scratch[ln_id] : 0.0f;
        total = simd_sum_f(total);
        if (ln_id == 0) rms = sqrt(total / (float)D + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float local_gwx = 0.0f;
    for (uint c = tid; c < D; c += tpg) {
        uint idx = base + c;
        float g = (float)grad_output[idx];
        float w = weight[c];
        float x = (float)input[idx];
        float xhat = x / rms;
        local_gwx += g * w * xhat;
    }
    float warp_gwx = simd_sum_f(local_gwx);
    if (ln_id == 0) scratch[sg_id] = warp_gwx;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float sum_gwx;
    if (sg_id == 0) {
        float total = (ln_id < 8) ? scratch[ln_id] : 0.0f;
        total = simd_sum_f(total);
        if (ln_id == 0) sum_gwx = total / (float)D;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = tid; c < D; c += tpg) {
        uint idx = base + c;
        float g  = (float)grad_output[idx];
        float w  = weight[c];
        float x  = (float)input[idx];
        float r  = (float)residual[idx];
        float xhat = x / rms;
        float dx = (1.0f / rms) * (g * w - xhat * sum_gwx) + r;
        grad_input[idx] = (bfloat)dx;
    }
}
