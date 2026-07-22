#include <metal_stdlib>
using namespace metal;

// Element-wise SwiGLU activation: output = (gate / (1 + exp(-gate))) * up
kernel void swiglu_forward(
    device const bfloat* gate  [[buffer(0)]],
    device const bfloat* up    [[buffer(1)]],
    device bfloat*       out   [[buffer(2)]],
    constant uint&      n     [[buffer(3)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    float g = gate[gid], u = up[gid];
    out[gid] = (bfloat)((g / (1.0f + exp(-g))) * u);
}
