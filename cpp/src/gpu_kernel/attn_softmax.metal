#include <metal_stdlib>
using namespace metal;

// Softmax with causal mask for attention probabilities.
// Each thread handles one row (s_q) of one attention head.
// Grid: [batch, n_heads, seq_len]
kernel void attn_softmax(
    device const float* scores   [[buffer(0)]], // [B, nH, S, S]
    device float*       probs    [[buffer(1)]], // [B, nH, S, S]
    constant uint&      batch    [[buffer(2)]],
    constant uint&      n_heads  [[buffer(3)]],
    constant uint&      seq_len  [[buffer(4)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint b = tid.x, h = tid.y, sq = tid.z;
    if (b >= batch || h >= n_heads || sq >= seq_len) return;

    uint row_base = ((b * n_heads + h) * seq_len + sq) * seq_len;
    device const float* row_scores = scores + row_base;
    device float* row_probs = probs + row_base;

    float max_val = -INFINITY;
    for (uint sk = 0; sk <= sq; ++sk)
        if (row_scores[sk] > max_val) max_val = row_scores[sk];

    float sum_exp = 0.0f;
    for (uint sk = 0; sk <= sq; ++sk)
        sum_exp += exp(row_scores[sk] - max_val);

    float inv_sum = 1.0f / sum_exp;
    for (uint sk = 0; sk <= sq; ++sk)
        row_probs[sk] = exp(row_scores[sk] - max_val) * inv_sum;
    for (uint sk = sq + 1; sk < seq_len; ++sk)
        row_probs[sk] = 0.0f;
}
