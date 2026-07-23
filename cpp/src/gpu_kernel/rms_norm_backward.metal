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

// Pass 1: Compute input gradient dx and write row-wise inv_rms to a temporary buffer
kernel void rms_norm_backward_dx(
    device const bfloat* grad_output     [[buffer(0)]],
    device const bfloat* input           [[buffer(1)]],
    device const bfloat* weight          [[buffer(2)]],
    device bfloat*       grad_input      [[buffer(3)]],
    device float*        inv_rms_buf     [[buffer(4)]],
    constant float&      eps             [[buffer(5)]],
    constant uint&       dims            [[buffer(6)]],
    uint                 row_idx         [[threadgroup_position_in_grid]],
    uint                 tid             [[thread_position_in_threadgroup]],
    uint                 tpg             [[threads_per_threadgroup]],
    uint                 simd_id         [[simdgroup_index_in_threadgroup]],
    uint                 lane_id         [[thread_index_in_simdgroup]]
) {
    // Shared memory for SIMD group reduction results (max 8 SIMD groups for 256 threads)
    threadgroup float shared_scratch[8];
    threadgroup float rms;
    threadgroup float sum_g_w_xhat;

    if (tid == 0) {
        rms = 1.0f;
        sum_g_w_xhat = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Each thread handles 4 elements (coalesced 128-bit load)
    uint col_idx = tid * 4;
    
    float4 x = float4(0.0f);
    float4 dy = float4(0.0f);
    float4 w = float4(0.0f);
    
    if (col_idx < dims) {
        bfloat4 x_bf  = *((device const bfloat4*)(input + row_idx * dims + col_idx));
        bfloat4 dy_bf = *((device const bfloat4*)(grad_output + row_idx * dims + col_idx));
        bfloat4 w_bf  = *((device const bfloat4*)(weight + col_idx));
        
        x  = float4(x_bf);
        dy = float4(dy_bf);
        w  = float4(w_bf);
    }
    
    // 1. Calculate sum of squares for this thread
    float local_sq_sum = x.x * x.x + x.y * x.y + x.z * x.z + x.w * x.w;
    
    // Reduce local sum within each SIMD group
    float simd_sq = simd_sum(local_sq_sum);
    if (lane_id == 0) {
        shared_scratch[simd_id] = simd_sq;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // First SIMD group reduces the 8 intermediate sums
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
    
    // Compute thread-local normalized input xhat
    float4 xhat = x / rms;
    
    // 2. Compute inner sum term: mean(g * w * xhat)
    float local_g_w_xhat = (dy.x * w.x * xhat.x) + (dy.y * w.y * xhat.y) + (dy.z * w.z * xhat.z) + (dy.w * w.w * xhat.w);
    
    // Reduce local sum within each SIMD group
    float simd_g_w_xhat = simd_sum(local_g_w_xhat);
    if (lane_id == 0) {
        shared_scratch[simd_id] = simd_g_w_xhat;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
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
    
    // 3. Compute gradients w.r.t input and write directly to global grad_input
    if (col_idx < dims) {
        float4 dx = (1.0f / rms) * (dy * w - xhat * sum_g_w_xhat);
        *((device bfloat4*)(grad_input + row_idx * dims + col_idx)) = (bfloat4)dx;
    }
    
    // Write out inverse RMS for this row (only done by thread 0)
    if (tid == 0) {
        inv_rms_buf[row_idx] = 1.0f / rms;
    }
}

// Pass 2: Compute weight gradient dw parallel across columns (coalesced, non-atomic)
kernel void rms_norm_backward_dw(
    device const bfloat* grad_output     [[buffer(0)]],
    device const bfloat* input           [[buffer(1)]],
    device const float*  inv_rms_buf     [[buffer(2)]],
    device bfloat*       grad_weight     [[buffer(3)]],
    constant uint&       num_rows        [[buffer(4)]],
    constant uint&       dims            [[buffer(5)]],
    uint3                ti              [[thread_position_in_threadgroup]]
) {
    uint tid = ti.x;
    uint col_idx = tid * 4;
    if (col_idx >= dims) return;
    
    float4 dw = float4(0.0f);
    
    // Loop over all rows (B * S)
    for (uint i = 0; i < num_rows; ++i) {
        float inv_rms = inv_rms_buf[i];
        
        bfloat4 x_bf  = *((device const bfloat4*)(input + i * dims + col_idx));
        bfloat4 dy_bf = *((device const bfloat4*)(grad_output + i * dims + col_idx));
        
        float4 x = float4(x_bf);
        float4 dy = float4(dy_bf);
        
        float4 xhat = x * inv_rms;
        dw += dy * xhat;
    }
    
    // Write the final accumulated weight gradient using a single non-atomic store
    *((device bfloat4*)(grad_weight + col_idx)) = (bfloat4)dw;
}
