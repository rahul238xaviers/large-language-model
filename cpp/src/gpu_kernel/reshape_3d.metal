#include <metal_stdlib>
using namespace metal;

kernel void reshape_to_3d(
    device const bfloat* src    [[buffer(0)]],
    device bfloat*       dst    [[buffer(1)]],
    constant uint&      batch   [[buffer(2)]],
    constant uint&      n_heads [[buffer(3)]],
    constant uint&      seq_len [[buffer(4)]],
    constant uint&      head_dim[[buffer(5)]],
    uint gid [[thread_position_in_grid]]
) {
    uint total = batch * n_heads * seq_len * head_dim;
    if (gid >= total) return;

    uint tmp = gid;
    uint d = tmp % head_dim; tmp /= head_dim;
    uint s = tmp % seq_len;  tmp /= seq_len;
    uint h = tmp % n_heads;  tmp /= n_heads;
    uint b = tmp;

    uint src_idx = (b * n_heads * seq_len + h * seq_len + s) * head_dim + d;
    uint dst_idx = (b * seq_len * n_heads + s * n_heads + h) * head_dim + d;
    dst[dst_idx] = src[src_idx];
}
