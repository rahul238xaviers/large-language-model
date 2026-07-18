// ==============================================================================
// TECHNICAL SPECIFICATION: GQA ATTENTION BACKWARD PASS
// ==============================================================================
//
// WHAT: Computes gradients for Q, K, and V during the attention backward pass.
//
// WHY: The forward pass computed attention scores (Q·K), applied softmax, and
//      weighted V vectors. The backward pass reverses this to propagate gradients
//      back through Q, K, and V for weight updates.
//
// THREAD GRID: 3D dispatch [batch, n_q_heads, seq_len]
//   - X-axis -> Batch index `b`
//   - Y-axis -> Query Head index `h`
//   - Z-axis -> Query Token index `s_q`
//
// ATOMIC WRITES: Multiple Q heads map to the same KV head in GQA. We use
//   atomic_fetch_add_explicit for grad_K and grad_V to avoid race conditions.
//
// TENSOR LAYOUTS:
//   - Q, grad_Q:         [batch, n_q_heads, seq_len, head_dim]
//   - K, V, grad_K, grad_V: [batch, n_kv_heads, seq_len, head_dim]
//   - grad_attn_output:  [batch, seq_len, n_q_heads * head_dim]
// ==============================================================================

#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN 2048
#endif

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
    uint3 tid [[thread_position_in_grid]]
) {
    uint b   = tid.x;
    uint h   = tid.y;
    uint s_q = tid.z;

    // WHAT: Bounds check to prevent out-of-range GPU threads from executing.
    if (b >= params.batch || h >= params.n_q_heads || s_q >= params.seq_len)
        return;

    // WHAT: Map this query head to its corresponding KV head using GQA grouping.
    // WHY: In GQA, multiple query heads share a single KV head to save memory.
    uint group_size = params.n_q_heads / params.n_kv_heads;
    uint kv_h = h / group_size;

    float scale = 1.0f / sqrt((float)params.head_dim);

    // WHAT: Compute base pointers for this thread's Q, K, V, and gradient vectors.
    uint q_offset = (b * params.n_q_heads * params.seq_len + h * params.seq_len + s_q) * params.head_dim;
    uint kv_base  = (b * params.n_kv_heads * params.seq_len + kv_h * params.seq_len) * params.head_dim;
    uint grad_out_offset = (b * params.seq_len + s_q) * (params.n_q_heads * params.head_dim) + h * params.head_dim;

    device const float* q_ptr       = Q + q_offset;
    device const float* grad_o_ptr  = grad_attn_output + grad_out_offset;

    // =========================================================================
    // Step A: Recompute attention probabilities (forward recompute)
    // WHAT: Dot-product Q with all valid Keys (causal mask: s_k <= s_q),
    //       then apply scaled softmax to get attention probabilities.
    // WHY: We don't store probabilities from the forward pass to save memory.
    //       Recomputing them here is the standard "memory-efficient attention" approach.
    // =========================================================================
    float raw_scores[MAX_SEQ_LEN];
    float max_score = -INFINITY;

    for (uint s_k = 0; s_k <= s_q; ++s_k) {
        device const float* k_ptr = K + kv_base + s_k * params.head_dim;
        float dot = 0.0f;
        for (uint d = 0; d < params.head_dim; ++d) {
            dot += q_ptr[d] * k_ptr[d];
        }
        raw_scores[s_k] = dot * scale;
        if (raw_scores[s_k] > max_score) {
            max_score = raw_scores[s_k];
        }
    }

    // WHAT: Compute softmax probabilities from raw scores.
    float prob[MAX_SEQ_LEN];
    float sum_exp = 0.0f;
    for (uint s_k = 0; s_k <= s_q; ++s_k) {
        sum_exp += exp(raw_scores[s_k] - max_score);
    }
    for (uint s_k = 0; s_k <= s_q; ++s_k) {
        prob[s_k] = exp(raw_scores[s_k] - max_score) / sum_exp;
    }

    // =========================================================================
    // Step B: Backprop w.r.t probabilities (dP) and accumulate grad_V
    // WHAT: Compute dP[s_k] = dot(grad_output, V[s_k]) for each key position.
    //       Simultaneously accumulate grad_V[s_k] += prob[s_k] * grad_output.
    // WHY: dP represents how much each attention weight needs to change.
    //       grad_V captures how the Value vectors should change.
    // =========================================================================
    float dP[MAX_SEQ_LEN];
    float sum_dP_prob = 0.0f;

    for (uint s_k = 0; s_k <= s_q; ++s_k) {
        device const float* v_ptr = V + kv_base + s_k * params.head_dim;
        uint grad_v_offset = kv_base + s_k * params.head_dim;

        float dot_g_v = 0.0f;
        for (uint d = 0; d < params.head_dim; ++d) {
            float g = grad_o_ptr[d];
            dot_g_v += g * v_ptr[d];
            // WHAT: Atomic add to grad_V because multiple Q heads share the same KV head.
            // WHY: Without atomics, concurrent threads writing to the same KV head would
            //      create a race condition and produce incorrect gradients.
            atomic_fetch_add_explicit(&grad_V[grad_v_offset + d],
                                     prob[s_k] * g,
                                     memory_order_relaxed);
        }
        dP[s_k] = dot_g_v;
        sum_dP_prob += dP[s_k] * prob[s_k];
    }

    // =========================================================================
    // Step C: Backprop through Softmax and Scaled Dot-Product to grad_Q & grad_K
    // WHAT: Compute the softmax Jacobian: dS = prob * (dP - sum(dP * prob)),
    //       then propagate through the Q·K dot product.
    // WHY: The softmax derivative has a special structure where each output
    //       depends on all inputs, captured by the sum_dP_prob correction term.
    // =========================================================================
    for (uint s_k = 0; s_k <= s_q; ++s_k) {
        float dS = prob[s_k] * (dP[s_k] - sum_dP_prob);
        float dS_scaled = dS * scale;

        device const float* k_ptr = K + kv_base + s_k * params.head_dim;
        uint grad_k_offset = kv_base + s_k * params.head_dim;

        for (uint d = 0; d < params.head_dim; ++d) {
            // WHAT: grad_Q is written exclusively by this thread (no contention).
            grad_Q[q_offset + d] += dS_scaled * k_ptr[d];
            // WHAT: Atomic add to grad_K because multiple Q heads share the same KV head.
            atomic_fetch_add_explicit(&grad_K[grad_k_offset + d],
                                     dS_scaled * q_ptr[d],
                                     memory_order_relaxed);
        }
    }
}
