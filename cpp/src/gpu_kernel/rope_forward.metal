#include <metal_stdlib>
using namespace metal;

kernel void rope_forward(
    device float*       q         [[buffer(0)]],
    device float*       k         [[buffer(1)]],
    device const float* cos_table [[buffer(2)]],
    device const float* sin_table [[buffer(3)]],
    constant uint&      batch     [[buffer(4)]],
    constant uint&      q_heads   [[buffer(5)]],
    constant uint&      kv_heads  [[buffer(6)]],
    constant uint&      seq_len   [[buffer(7)]],
    constant uint&      head_dim  [[buffer(8)]],
    uint gid [[thread_position_in_grid]]
) {
    uint half_dim = head_dim / 2;
    uint total_q  = batch * q_heads * seq_len * half_dim;
    uint total_kv = batch * kv_heads * seq_len * half_dim;
    uint total    = total_q + total_kv;

    if (gid >= total) return;

    if (gid < total_q) {
        uint tmp = gid;
        uint i = tmp % half_dim; tmp /= half_dim;
        uint s = tmp % seq_len;  tmp /= seq_len;
        uint h = tmp % q_heads;
        uint b = tmp / q_heads;

        uint base = (b * q_heads * seq_len + h * seq_len + s) * head_dim;
        uint idx0 = 2 * i;
        uint idx1 = 2 * i + 1;

        float cos_val = cos_table[s * half_dim + i];
        float sin_val = sin_table[s * half_dim + i];

        float x0 = q[base + idx0];
        float x1 = q[base + idx1];
        q[base + idx0] = x0 * cos_val - x1 * sin_val;
        q[base + idx1] = x0 * sin_val + x1 * cos_val;
    } else {
        uint gid_k = gid - total_q;
        uint tmp = gid_k;
        uint i = tmp % half_dim; tmp /= half_dim;
        uint s = tmp % seq_len;  tmp /= seq_len;
        uint h = tmp % kv_heads;
        uint b = tmp / kv_heads;

        uint base = (b * kv_heads * seq_len + h * seq_len + s) * head_dim;
        uint idx0 = 2 * i;
        uint idx1 = 2 * i + 1;

        float cos_val = cos_table[s * half_dim + i];
        float sin_val = sin_table[s * half_dim + i];

        float x0 = k[base + idx0];
        float x1 = k[base + idx1];
        k[base + idx0] = x0 * cos_val - x1 * sin_val;
        k[base + idx1] = x0 * sin_val + x1 * cos_val;
    }
}
