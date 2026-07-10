// ==============================================================================
// TECHNICAL SPECIFICATION & ARCHITECTURAL REFERENCE: GROUPED QUERY ATTENTION (GQA)
// ==============================================================================
//
// 1. WHAT IS GQA?
//    Grouped Query Attention (GQA) is an optimization that reduces memory bandwidth
//    overhead during LLM generation. Instead of having equal numbers of Query, Key,
//    and Value heads, multiple Query heads share a single Key and Value head.
//
// 2. DATA FLOW & ALGORITHMIC STEPS:
//    For each Query vector Q[b, s_q, h]:
//    - Step A (Dot Product): Multiply Q by all Keys K[b, s_k, h_kv] to get scores.
//      `score[s_k] = sum(Q[d] * K[d])` for d in 0..head_dim-1, and s_k in 0..s_q.
//    - Step B (Softmax): Convert scores into probabilities:
//      `softmax_score[s_k] = exp(score[s_k] - max_score) / sum(exp(score - max_score))`
//    - Step C (Weighted Sum): Multiply probabilities by Values V[b, s_k, h_kv]:
//      `out[b, s_q, h, d] = sum(softmax_score[s_k] * V[d])` for s_k in 0..s_q.
//
// 3. TENSOR SHAPES & LAYOUTS:
//    - Q (Query):    [batch, seq_len, n_q_heads,  head_dim]
//    - K (Key):      [batch, seq_len, n_kv_heads, head_dim]
//    - V (Value):    [batch, seq_len, n_kv_heads, head_dim]
//    - out (Output): [batch, seq_len, n_q_heads,  head_dim]
//
// 4. THREAD GRID MAPPING:
//    We launch a 3D grid of threads where each thread is responsible for computing
//    the output vector of size `head_dim` for a single query head.
//    - X-axis (thread_position_in_grid.x) -> Batch index `b` (0 .. batch-1)
//    - Y-axis (thread_position_in_grid.y) -> Query Token index `s_q` (0 .. seq_len-1)
//    - Z-axis (thread_position_in_grid.z) -> Query Head index `h` (0 .. n_q_heads-1)
//
// ==============================================================================

//Defined the max sequence length in case, it is not set at the build stage
//This will cause a compile error if not set at the build stage, which is the desired be    havior
//as it will force the user to set the max sequence length at the build stage
#ifndef MAX_SEQ_LEN
#define MAX_SEQ_LEN 2048
#endif

#include <metal_stdlib>
using namespace metal;

struct GQAParams {
  uint batch;
  uint n_q_heads;
  uint n_kv_heads;
  uint seq_len;
  uint head_dim;
};


kernel void gemm_gqa(
    device const float* Q    [[buffer(0)]],
    device const float* K    [[buffer(1)]],
    device const float* V    [[buffer(2)]],
    device float*       out  [[buffer(3)]],
    constant GQAParams& gqa  [[buffer(4)]],
    uint3 tg_id   [[thread_position_in_grid]]
) {
    // We pass this as 3-D grid because we wanted to process Q, K and V as 3-D arrays
    // The b is for the batch dimension, s_q is the sequence length, h is the query head index
    // The size of the grid is [gqa.batch, gqa.seq_len, gqa.n_q_heads]   
    uint b = tg_id.x;
    uint s_q = tg_id.y;
    uint h = tg_id.z;

    uint group_size = gqa.n_q_heads / gqa.n_kv_heads;
    uint h_kv = h / group_size;

    device const float* q_ptr = Q + (b * gqa.n_q_heads * gqa.seq_len + h * gqa.seq_len + s_q) * gqa.head_dim;
    device float* out_ptr =  out + (b * gqa.seq_len * gqa.n_q_heads + s_q * gqa.n_q_heads + h) * gqa.head_dim;
  
    // the shape of K and V is [B, H_KV, S, D] and indices are batch = b, token index = 0, key head index = h_kv
    uint kv_common_factor = (b * gqa.n_kv_heads * gqa.seq_len + h_kv * gqa.seq_len) * gqa.head_dim;
    device const float* k_base = K + kv_common_factor;
    device const float* v_base = V + kv_common_factor;

    float scale = 1.0f / sqrt((float)gqa.head_dim);

    // Initialize online softmax with s_k = 0
    device const float* k_ptr_0 = k_base;
    float sum_0 = 0.0f;
    for (uint d = 0; d < gqa.head_dim; d++) {
        sum_0 += q_ptr[d] * k_ptr_0[d];
    }
    float max_val = sum_0 * scale;
    float sum_exp = 1.0f;

    device const float* v_ptr_0 = v_base;
    float accum_val[128];
    for (uint d = 0; d < gqa.head_dim; d++) {
        accum_val[d] = v_ptr_0[d];
    }

    // Process remaining keys s_k from 1 to s_q
    for (uint s_k = 1; s_k <= s_q; s_k++) {
        device const float* k_ptr = k_base + s_k * gqa.head_dim;

        // 1. Compute dot product Q * K
        float sum = 0.0f;
        for (uint d = 0; d < gqa.head_dim; d++) {
            sum += q_ptr[d] * k_ptr[d];
        }
        float score = sum * scale;

        // 2. Online softmax update
        float m_old = max_val;
        if (score > max_val) {
            max_val = score;
            float exp_scale = exp(m_old - max_val);
            sum_exp = sum_exp * exp_scale + 1.0f;

            device const float* v_ptr = v_base + s_k * gqa.head_dim;
            for (uint d = 0; d < gqa.head_dim; d++) {
                accum_val[d] = accum_val[d] * exp_scale + v_ptr[d];
            }
        } else {
            float exp_term = exp(score - max_val);
            sum_exp += exp_term;

            device const float* v_ptr = v_base + s_k * gqa.head_dim;
            for (uint d = 0; d < gqa.head_dim; d++) {
                accum_val[d] += exp_term * v_ptr[d];
            }
        }
    }

    // 3. Normalize and write to VRAM
    for (uint d = 0; d < gqa.head_dim; ++d) {
        out_ptr[d] = accum_val[d] / sum_exp;
    }
}
