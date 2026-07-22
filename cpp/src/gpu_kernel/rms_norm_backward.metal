#include <metal_stdlib>
using namespace metal;

// Helper function to perform atomic float addition in device memory
inline void atomic_add_float(device atomic_uint* addr, float value) {
    uint expected = atomic_load_explicit(addr, memory_order_relaxed);
    while (true) {
        float expected_val = as_type<float>(expected);
        float new_val = expected_val + value;
        uint desired = as_type<uint>(new_val);
        if (atomic_compare_exchange_weak_explicit(addr, &expected, desired, memory_order_relaxed, memory_order_relaxed)) {
            break;
        }
    }
}

// Inline helper for SIMD group reduction using intra-register shuffles
inline float simd_sum(float val) {
    val += simd_shuffle_down(val, 16);
    val += simd_shuffle_down(val, 8);
    val += simd_shuffle_down(val, 4);
    val += simd_shuffle_down(val, 2);
    val += simd_shuffle_down(val, 1);
    return val;
}

// Pass 1: Compute input gradient dx and accumulate weight gradient dw directly in a single pass
kernel void rms_norm_backward_dx(
    device const bfloat* grad_output     [[buffer(0)]],
    device const bfloat* input           [[buffer(1)]],
    device const bfloat* weight          [[buffer(2)]],
    device bfloat*       grad_input      [[buffer(3)]],
    device bfloat*       grad_weight     [[buffer(4)]],
    constant float&     eps             [[buffer(5)]],
    constant uint&      dims            [[buffer(6)]],
    uint                row_idx         [[threadgroup_position_in_grid]],
    uint                tid             [[thread_position_in_threadgroup]],
    uint                tpg             [[threads_per_threadgroup]],
    uint                simd_id         [[simdgroup_index_in_threadgroup]],
    uint                lane_id         [[thread_index_in_simdgroup]]
) {
    // Shared memory for SIMD group reduction results (max 8 SIMD groups for 256 threads)
    threadgroup float shared_scratch[8];
    
    // 1. Calculate sum of squares for this row to compute RMS
    float local_sq_sum = 0.0f;
    for (uint col = tid; col < dims; col += tpg) {
        float val = input[row_idx * dims + col];
        local_sq_sum += val * val;
    }
    
    // Reduce local sum within each SIMD group
    float simd_sq = simd_sum(local_sq_sum);
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
            rms = sqrt(val / (float)dims + eps);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 2. Compute inner sum term: mean(g * w * xhat)
    float local_g_w_xhat = 0.0f;
    for (uint col = tid; col < dims; col += tpg) {
        uint idx = row_idx * dims + col;
        float xhat = input[idx] / rms;
        local_g_w_xhat += grad_output[idx] * weight[col] * xhat;
    }
    
    // Reduce local sum within each SIMD group
    float simd_g_w_xhat = simd_sum(local_g_w_xhat);
    if (lane_id == 0) {
        shared_scratch[simd_id] = simd_g_w_xhat;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    threadgroup float sum_g_w_xhat;
    if (simd_id == 0) {
        float val = (lane_id < 8) ? shared_scratch[lane_id] : 0.0f;
        val += simd_shuffle_down(val, 4);
        val += simd_shuffle_down(val, 2);
        val += simd_shuffle_down(val, 1);
        if (lane_id == 0) {
            sum_g_w_xhat = val / (float)dims;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 3. Compute gradients w.r.t input and write directly to global grad_weight
    for (uint col = tid; col < dims; col += tpg) {
        uint idx = row_idx * dims + col;
        float xhat = input[idx] / rms;
        
        // grad_input calculation
        grad_input[idx] = (bfloat)((1.0f / rms) * ((float)grad_output[idx] * (float)weight[col] - xhat * sum_g_w_xhat));
        
        // Accumulate to global weight gradient directly using atomic floats
        float dw_val = (float)grad_output[idx] * xhat;
        atomic_add_float((device atomic_uint*)&grad_weight[col], dw_val);
    }
}
