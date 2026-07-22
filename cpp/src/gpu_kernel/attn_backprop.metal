#include <metal_stdlib>
using namespace metal;

// Compute dS = P * (dP_row - sum_dP_prob) for one row.
// Grid: [B, nH, S] — each thread processes one (b, h, sq).
// Output: dS_row[S] per thread (stored as [B,nH,S,S]).
kernel void attn_ds(
    device const float* probs   [[buffer(0)]], // [B, nH, S, S]
    device const float* dP_row  [[buffer(1)]], // [B, nH, S, S]
    device float*       dS_out  [[buffer(2)]], // [B, nH, S, S]
    constant uint&      batch   [[buffer(3)]],
    constant uint&      n_heads [[buffer(4)]],
    constant uint&      seq_len [[buffer(5)]],
    uint3 tid [[thread_position_in_grid]]
) {
    uint b = tid.x, h = tid.y, sq = tid.z;
    if (b >= batch || h >= n_heads || sq >= seq_len) return;
    uint row_base = ((b * n_heads + h) * seq_len + sq) * seq_len;
    device const float* P  = probs  + row_base;
    device const float* dP = dP_row + row_base;
    device float*       dS = dS_out + row_base;

    float sum_dP_prob = 0.0f;
    for (uint sk = 0; sk <= sq; ++sk)
        sum_dP_prob += P[sk] * dP[sk];

    for (uint sk = 0; sk <= sq; ++sk)
        dS[sk] = P[sk] * (dP[sk] - sum_dP_prob);
    for (uint sk = sq + 1; sk < seq_len; ++sk)
        dS[sk] = 0.0f;
}
