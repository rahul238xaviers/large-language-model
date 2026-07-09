#include <metal_stdlib>
using namespace metal;

kernel void gemm_gqa(
    device const float* Q    [[buffer(0)]],
    device const float* K    [[buffer(1)]],
    device const float* V    [[buffer(2)]],
    device float*       out  [[buffer(3)]]

) {
    // We will build the logic step-by-step
}
