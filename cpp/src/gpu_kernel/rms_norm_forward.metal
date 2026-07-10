#include <metal_stdlib>
using namespace metal;

// Inline helper for SIMD group reduction using intra-register shuffles
inline float simd_sum(float val) {
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    return val;
}

kernel void rms_norm_forward(
    device const float* input       [[buffer(0)]],
    device float*       output      [[buffer(1)]],
    device const float* weight      [[buffer(2)]],
    constant float&     eps         [[buffer(3)]],
    constant uint&      dims        [[buffer(4)]],
    uint                row_idx     [[threadgroup_position_in_grid]],
    uint                tid         [[thread_position_in_threadgroup]],
    uint                tpg         [[threads_per_threadgroup]],
    uint                simd_id     [[simdgroup_index_in_threadgroup]],
    uint                lane_id     [[thread_index_in_simdgroup]]
) {
    // Threadgroup scratch space for SIMDgroup sums (max 8 SIMD groups for 256 threads)
    threadgroup float shared_scratch[8];
    
    // Each thread calculates its local sum of squares
    float local_sum = 0.0f;
    for (uint col = tid; col < dims; col += tpg) {
        float val = input[row_idx * dims + col];
        local_sum += val * val;
    }
    
    // Reduce within the SIMDgroup
    float simd_sq = simd_sum(local_sum);
    if (lane_id == 0) {
        shared_scratch[simd_id] = simd_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // First SIMD group reduces the 8 intermediate sums
    threadgroup float rms;
    if (simd_id == 0) {
        float val = (lane_id < 8) ? shared_scratch[lane_id] : 0.0f;
        val += simd_shuffle_down(val, 4);
        val += simd_shuffle_down(val, 2);
        val += simd_shuffle_down(val, 1);
        if (lane_id == 0) {
            rms = rsqrt(val / (float)dims + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Normalize and scale the output row
    for (uint col = tid; col < dims; col += tpg) {
        uint idx = row_idx * dims + col;
        output[idx] = (input[idx] * rms) * weight[col];
    }
}
