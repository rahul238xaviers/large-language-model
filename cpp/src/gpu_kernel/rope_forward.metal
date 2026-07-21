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
    uint total_pairs = batch * max(q_heads, kv_heads) * seq_len * half_dim;
    if (gid >= total_pairs) return;

    uint tmp = gid;
    uint i = tmp % half_dim; tmp /= half_dim;
    uint s = tmp % seq_len;  tmp /= seq_len;

    uint idx0 = 2 * i;
    uint idx1 = 2 * i + 1;

    float cos_val = cos_table[s * half_dim + i];
    float sin_val = sin_table[s * half_dim + i];

    // Apply to Q (q_heads)
    if (gid < batch * q_heads * seq_len * half_dim) {
        uint tmp2 = gid;
        uint i2 = tmp2 % half_dim; tmp2 /= half_dim;
        uint s2 = tmp2 % seq_len;  tmp2 /= seq_len;
        uint h_q = tmp2 % q_heads;
        uint b = tmp2 / q_heads;

        uint q_base = (b * q_heads * seq_len + h_q * seq_len + s2) * head_dim;
        float x0 = q[q_base + idx0];
        float x1 = q[q_base + idx1];
        q[q_base + idx0] = x0 * cos_val - x1 * sin_val;
        q[q_base + idx1] = x0 * sin_val + x1 * cos_val;
    }

    // Apply to K (kv_heads)
    uint k_offset = batch * q_heads * seq_len * half_dim;
    if (gid >= k_offset) {
        uint gid_k = gid - k_offset;
        uint tmp3 = gid_k;
        uint i3 = tmp3 % half_dim; tmp3 /= half_dim;
        uint s3 = tmp3 % seq_len;  tmp3 /= seq_len;
        uint h_k = tmp3 % kv_heads;
        uint b = tmp3 / kv_heads;

        uint k_base = (b * kv_heads * seq_len + h_k * seq_len + s3) * head_dim;
        float x0 = k[k_base + idx0];
        float x1 = k[k_base + idx1];
        k[k_base + idx0] = x0 * cos_val - x1 * sin_val;
        k[k_base + idx1] = x0 * sin_val + x1 * cos_val;
    }
}
