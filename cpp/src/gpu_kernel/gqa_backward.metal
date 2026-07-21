// ==============================================================================
// GQA BACKWARD — score-fed, 3-pass, no Q·K recomputation
// ==============================================================================
// Scores are precomputed by gqa_scores kernel and passed via scores[S×S] buffer.
// Each pass reads scores[s_k] from global memory instead of recomputing dot(Q,K).
// Eliminates 2× of the 3× Q·K passes — only softmax math remains in the loop.
// ==============================================================================

#include <metal_stdlib>
using namespace metal;

struct GQABackwardParams {
    uint batch;
    uint n_q_heads;
    uint n_kv_heads;
    uint seq_len;
    uint head_dim;
};

kernel void gqa_backward(
    device const float*  Q                [[buffer(0)]],
    device const float*  K                [[buffer(1)]],
    device const float*  V                [[buffer(2)]],
    device const float*  grad_attn_output [[buffer(3)]],
    device float*        grad_Q           [[buffer(4)]],
    device atomic_float* grad_K           [[buffer(5)]],
    device atomic_float* grad_V           [[buffer(6)]],
    constant GQABackwardParams& params    [[buffer(7)]],
    device const float*  scores           [[buffer(8)]], // precomputed Q·K^T
    uint3 tid [[thread_position_in_grid]]
) {
    uint b   = tid.x;
    uint h   = tid.y;
    uint s_q = tid.z;

    if (b >= params.batch || h >= params.n_q_heads || s_q >= params.seq_len) return;

    uint group_size = params.n_q_heads / params.n_kv_heads;
    uint kv_h = h / group_size;
    float scale = 1.0f / sqrt((float)params.head_dim);
    const uint HD = params.head_dim;
    const uint S  = params.seq_len;

    uint q_offset = (b * params.n_q_heads * S + h * S + s_q) * HD;
    uint kv_base  = (b * params.n_kv_heads * S + kv_h * S) * HD;
    uint grad_out_offset = (b * S + s_q) * (params.n_q_heads * HD) + h * HD;
    uint score_row_base = ((b * params.n_q_heads + h) * S + s_q) * S;

    device const float* q_ptr      = Q + q_offset;
    device const float* grad_o_ptr = grad_attn_output + grad_out_offset;

    // ── Pass 1: Online softmax from precomputed scores ──
    float max_val = -INFINITY;
    float sum_exp = 0.0f;
    for (uint s_k = 0; s_k <= s_q; ++s_k) {
        float score = scores[score_row_base + s_k];
        if (s_k == 0) { max_val = score; sum_exp = 1.0f; }
        else if (score > max_val) { float om = max_val; max_val = score; sum_exp = sum_exp * exp(om - max_val) + 1.0f; }
        else { sum_exp += exp(score - max_val); }
    }

    // ── Pass 2: prob, dP, sum_dP_prob, grad_V ──
    float sum_dP_prob = 0.0f;
    for (uint s_k = 0; s_k <= s_q; ++s_k) {
        float score = scores[score_row_base + s_k];
        float prob = exp(score - max_val) / sum_exp;

        device const float* v_ptr = V + kv_base + s_k * HD;
        uint kv_off = kv_base + s_k * HD;

        float dP = 0.0f;
        for (uint d = 0; d < HD; ++d) {
            float g = grad_o_ptr[d];
            dP += g * v_ptr[d];
            atomic_fetch_add_explicit(&grad_V[kv_off + d], prob * g, memory_order_relaxed);
        }
        sum_dP_prob += dP * prob;
    }

    // ── Pass 3: dS → grad_Q, grad_K ──
    for (uint s_k = 0; s_k <= s_q; ++s_k) {
        float score = scores[score_row_base + s_k];
        float prob = exp(score - max_val) / sum_exp;

        device const float* k_ptr = K + kv_base + s_k * HD;
        uint kv_off = kv_base + s_k * HD;

        float dP = 0.0f;
        for (uint d = 0; d < HD; ++d) dP += grad_o_ptr[d] * (V + kv_base + s_k * HD)[d];

        float dS = prob * (dP - sum_dP_prob);
        float dS_scaled = dS * scale;
        for (uint d = 0; d < HD; ++d) {
            grad_Q[q_offset + d] += dS_scaled * k_ptr[d];
            atomic_fetch_add_explicit(&grad_K[kv_off + d], dS_scaled * q_ptr[d], memory_order_relaxed);
        }
    }
}
