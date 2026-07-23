#include <metal_stdlib>
using namespace metal;

kernel void rope_backward(device float* grad [[buffer(0)]],
                        device const float* cos_table [[buffer(1)]],
                        device const float* sin_table [[buffer(2)]],
                        constant uint&      batch        [[buffer(3)]],
                        constant uint&      heads        [[buffer(4)]],
                        constant uint&      seq_len      [[buffer(5)]],
                        constant uint&      head_dim     [[buffer(6)]],
                        uint                 idx          [[thread_position_in_grid]]
){

    uint half_dim = head_dim / 2;
    uint total_pairs = batch * heads * seq_len * half_dim;

    if (idx >= total_pairs) return;
    
    uint seq_idx = (idx % (heads * seq_len * half_dim)) % (seq_len * half_dim) / half_dim;
    uint dim_idx = (idx % (heads * seq_len * half_dim)) % (seq_len * half_dim) % half_dim;

    uint idx_0 = idx * 2;
    uint idx_1 = idx * 2 + 1;

    uint cos_idx = seq_idx * half_dim + dim_idx;
    float cos_val = cos_table[cos_idx];
    float sin_val = sin_table[cos_idx];

    float x0 = grad[idx_0];
    float x1 = grad[idx_1];

    float x0_new = x0 * cos_val + x1 * sin_val;
    float x1_new = x1 * cos_val - x0 * sin_val; 

    grad[idx_0] = (bfloat)x0_new;
    grad[idx_1] = (bfloat)x1_new;

}
    