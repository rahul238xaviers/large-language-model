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
    device float*       a   [[buffer(0)]],  // WHAT: In-place accumulation target (h)
    device const float* b   [[buffer(1)]],  // WHAT: Tensor to add (attn_out or ffn_out)
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
