#include <metal_stdlib>
using namespace metal;

kernel void rms_norm_forward(
    device const float* input       [[buffer(0)]],
    device float*       output      [[buffer(1)]],
    device const float* weight      [[buffer(2)]],
    constant float&     eps         [[buffer(3)]],
    constant uint&      dims        [[buffer(4)]],
    uint                row_idx     [[threadgroup_position_in_grid]],
    uint                tid         [[thread_position_in_threadgroup]],
    uint                tpg         [[threads_per_threadgroup]]
) {
    // Threadgroup shared memory to perform parallel reduction
    threadgroup float shared_sq_sum[256];
    
    // Each thread calculates its local sum of squares
    float local_sum = 0.0f;
    for (uint col = tid; col < dims; col += tpg) {
        float val = input[row_idx * dims + col];
        local_sum += val * val;
    }
    shared_sq_sum[tid] = local_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Perform tree reduction in shared memory to sum across all threads
    for (uint stride = tpg / 2; stride > 0; stride /= 2) {
        if (tid < stride) {
            shared_sq_sum[tid] += shared_sq_sum[tid + stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    
    // Thread 0 calculates the final RMS divisor
    threadgroup float rms;
    if (tid == 0) {
        rms = rsqrt(shared_sq_sum[0] / (float)dims + eps);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    
    // Normalize and scale the output row
    for (uint col = tid; col < dims; col += tpg) {
        uint idx = row_idx * dims + col;
        output[idx] = (input[idx] * rms) * weight[col];
    }
}
