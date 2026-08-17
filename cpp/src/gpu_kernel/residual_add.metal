// ==============================================================================
// KERNEL: residual_add
// WHAT: Performs element-wise in-place addition: a[i] += b[i]
// WHY:  Replaces the CPU residual additions (h += attn_out, h += ffn_out)
//       that currently force a CPU readback between GPU kernel dispatches.
//       By running this on GPU, the hidden state `a` never needs to leave
//       the GPU command stream between layers.
// SHAPE: Both tensors are flat float arrays of identical length n.
//        Thread layout: 1D grid of (n) threads, each handles one element.
// ==============================================================================

#include <metal_stdlib>
using namespace metal;

kernel void residual_add(
    device bfloat*       a   [[buffer(0)]],  // WHAT: In-place accumulation target (h)
    device const bfloat* b   [[buffer(1)]],  // WHAT: Tensor to add (attn_out or ffn_out)
    constant uint&      n   [[buffer(2)]],  // WHAT: Total number of float elements
    uint gid [[thread_position_in_grid]]
) {
    // WHAT: Bounds check — threads past the tensor length do nothing.
    // WHY:  GPU threadgroup sizes are powers of 2; the last group may overshoot n.
    if (gid >= n) return;

    // WHAT: Single in-place add.
    // WHY:  No synchronization needed — each thread owns exactly one element.
    a[gid] += b[gid];
}

kernel void convert_fp32_to_bf16(
    device const float* src [[buffer(0)]],
    device bfloat*      dst [[buffer(1)]],
    constant uint&      n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    dst[gid] = (bfloat)src[gid];
}

kernel void convert_bf16_to_fp32(
    device const bfloat* src [[buffer(0)]],
    device float*        dst [[buffer(1)]],
    constant uint&       n   [[buffer(2)]],
    uint gid [[thread_position_in_grid]]
) {
    if (gid >= n) return;
    dst[gid] = (float)src[gid];
}
