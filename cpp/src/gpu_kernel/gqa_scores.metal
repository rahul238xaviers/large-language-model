#include <metal_stdlib>
using namespace metal;

// Computes Q·K^T scores with causal mask: scores[b,h,s_q,s_k] = Q[b,h,s_q]·K[b,kv_h,s_k] * scale
// Grid: [batch, n_q_heads, seq_len * seq_len] — last dim flattened.
// Thread extracts s_q = flat / seq_len, s_k = flat % seq_len.
// Only processes s_k ≤ s_q (causal mask).

kernel void gqa_scores(
    device const float* Q        [[buffer(0)]],
    device const float* K        [[buffer(1)]],
    device float*       scores   [[buffer(2)]],
    constant uint&      batch    [[buffer(3)]],
    constant uint&      n_heads  [[buffer(4)]],
    constant uint&      n_kv     [[buffer(5)]],
    constant uint&      seq_len  [[buffer(6)]],
    constant uint&      head_dim [[buffer(7)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint b = tid.x;
    uint h = tid.y;
    uint flat = tid.z;
    uint s_q = flat / seq_len;
    uint s_k = flat % seq_len;

    if (b >= batch || h >= n_heads || s_q >= seq_len || s_k > s_q) return;

    uint kv_h = h / (n_heads / n_kv);
    float scale = 1.0f / sqrt((float)head_dim);

    uint q_off = (b * n_heads * seq_len + h * seq_len + s_q) * head_dim;
    uint k_off = (b * n_kv * seq_len + kv_h * seq_len + s_k) * head_dim;

    float sum = 0.0f;
    for (uint d = 0; d < head_dim; ++d)
        sum += Q[q_off + d] * K[k_off + d];

    uint score_off = ((b * n_heads + h) * seq_len + s_q) * seq_len + s_k;
    scores[score_off] = sum * scale;
}
