// Write your SwiGLU backward MSL shader here under the Teacher's guidance!
#include <metal_stdlib>
using namespace metal;



kernel void swiglu_backward(
    device const bfloat* grad_output [[buffer(0)]],
    device const bfloat* gate        [[buffer(1)]],
    device const bfloat* up          [[buffer(2)]],
    device bfloat*       grad_gate   [[buffer(3)]],
    device bfloat*       grad_up     [[buffer(4)]],
    constant uint&      n           [[buffer(5)]],
    uint                idx         [[thread_position_in_grid]]
) {
    if (idx >= n) return;
    
    float g = gate[idx];
    float u = up[idx];
    float dy = grad_output[idx];
    
    float sig = 1.0f / (1.0f + exp(-g));
    float silu_val = g * sig;
    float dsilu = sig * (1.0f + g * (1.0f - sig));
    
    grad_up[idx] = (bfloat)(dy * silu_val);
    grad_gate[idx] = (bfloat)(dy * u * dsilu);
}   