#include <metal_stdlib>
using namespace metal;

// Pass 1: Compute input gradient dx and row-wise weight gradient terms dw_rows
kernel void rms_norm_backward_dx(
    device const float* grad_output     [[buffer(0)]],
    device const float* input           [[buffer(1)]],
    device const float* weight          [[buffer(2)]],
    device float*       grad_input      [[buffer(3)]],
    device float*       dw_rows         [[buffer(4)]],
    constant float&     eps             [[buffer(5)]],
    constant uint&      dims            [[buffer(6)]],
    uint                row_idx         [[threadgroup_position_in_grid]],
    uint                tid         [[thread_position_in_threadgroup]],
    uint                tpg         [[threads_per_threadgroup]]
) {
    // Threadgroup shared memory for parallel reductions
    threadgroup float shared_sq_sum[256];
    threadgroup float shared_sum_g_w_xhat[256];
    
    // 1. Calculate sum of squares for this row to compute RMS
    float local_sq_sum = 0.0f;
    for (uint col = tid; col < dims; col += tpg) {
        float val = input[row_idx * dims + col];
        local_sq_sum += val * val;
    }
    shared_sq_sum[tid] = local_sq_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction for sum of squares
    for (uint stride = tpg / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    threadgroup float rms;
    if (tid == 0) {
        rms = sqrt(shared_sq_sum[0] / (float)dims + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // 2. Compute inner sum term: mean(g * w * xhat)
    float local_g_w_xhat = 0.0f;
    for (uint col = tid; col < dims; col += tpg) {
        uint idx = row_idx * dims + col;
        float xhat = input[idx] / rms;
        local_g_w_xhat += grad_output[idx] * weight[col] * xhat;
    }
    shared_sum_g_w_xhat[tid] = local_g_w_xhat;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Tree reduction for inner sum term
    for (uint stride = tpg / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sum_g_w_xhat[tid] += shared_sum_g_w_xhat[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    float sum_g_w_xhat = shared_sum_g_w_xhat[0] / (float)dims;
    
    // 3. Compute gradients w.r.t input and write weight gradient term
    for (uint col = tid; col < dims; col += tpg) {
        uint idx = row_idx * dims + col;
        float xhat = input[idx] / rms;
        
        // grad_input calculation
        grad_input[idx] = (1.0f / rms) * (grad_output[idx] * weight[col] - xhat * sum_g_w_xhat);
        
        // dw_row term
        dw_rows[idx] = grad_output[idx] * xhat;
    }
}

// Pass 2: Sum columns of dw_rows to produce final grad_weight
kernel void rms_norm_backward_dw_reduce(
    device const float* dw_rows     [[buffer(0)]],
    device float*       grad_weight [[buffer(1)]],
    constant uint&      num_rows    [[buffer(2)]],
    constant uint&      dims        [[buffer(3)]],
    uint                col         [[thread_position_in_grid]]
) {
    if (col >= dims) return;
    
    float sum = 0.0f;
    for (uint r = 0; r < num_rows; ++r) {
        sum += dw_rows[r * dims + col];
    }
    
    // Accumulate to global weight gradient
    grad_weight[col] += sum;
}
