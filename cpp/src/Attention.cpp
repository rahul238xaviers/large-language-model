/**
 * @file Attention.cpp
 * @brief Implementation of the GQA (Grouped Query Attention) layer forward &
 * backward passes
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * This file implements the projections, reshaping, GQA forward operations, and
 * analytical backpropagation for Grouped Query Attention (GQA).
 */

#include "Attention.hpp"
#include "RMSNorm.hpp"
#include "Tensor.hpp"
#define ACCELERATE_NEW_LAPACK
#include "gpu_kernel/MetalBridge.hpp"
#include <Accelerate/Accelerate.h>
#include <cmath>
#include <cstddef>
#include <dispatch/dispatch.h>
#include <stdexcept>
#include <vector>

/**
 * @brief Construct a new Attention object and initialize the projection
 * weights.
 *
 * Wq_, Wk_, Wv_, and Wo_ are the weight matrices for Query, Key, Value,
 * and Output projections.
 *
 * @param config ModelConfig containing n_heads, n_kv_heads, head_dim, and
 * hidden_dim.
 */
Attention::Attention(const ModelConfig &config)
    : config_(config),
      Wq_({config.hidden_dim, config.n_heads * config.head_dim}),
      Wk_({config.hidden_dim, config.n_kv_heads * config.head_dim}),
      Wv_({config.hidden_dim, config.n_kv_heads * config.head_dim}),
      Wo_({config.n_heads * config.head_dim, config.hidden_dim}) {

  if (config.n_kv_heads > config.n_heads) {
    throw std::invalid_argument("n_kv_heads cannot be greater than n_heads");
  }
  if (config.n_heads % config.n_kv_heads != 0) {
    throw std::invalid_argument("n_heads must be a multiple of n_kv_heads");
  }
}
/**
 * @brief Reshapes a 3D tensor of shape [batch, seq_len, n_heads * head_dim]
 *        into a 4D tensor of shape [batch, n_heads, seq_len, head_dim].
 *
 * Used to isolate attention heads so self-attention can be computed
 * head-by-head.
 *
 * Example:
 *   If src shape is [2, 128, 256], calling reshape_to_4d(src, 8, 32)
 *   returns a tensor of shape [2, 8, 128, 32].
 *
 * @param src Input 3D tensor to reshape.
 * @param n_heads Number of attention heads.
 * @param head_dim Dimension of each attention head.
 * @return Tensor Reshaped 4D tensor.
 */
Tensor reshape_to_4d(const Tensor &src, size_t n_heads, size_t head_dim) {

  size_t batch = src.shape()[0];
  size_t seq_len = src.shape()[1];

  Tensor dest({batch, n_heads, seq_len, head_dim}, 0.0f);

  for (size_t b = 0; b < batch; ++b) {

    for (size_t h = 0; h < n_heads; h++) {
      for (size_t s = 0; s < seq_len; ++s) {
        for (size_t d = 0; d < head_dim; ++d) {
          dest(b, h, s, d) = src(b, s, h * head_dim + d);
        }
      }
    }
  }
  return dest;
}

/**
 * @brief Reshapes a 4D tensor of shape [batch, n_heads, seq_len, head_dim]
 *        into a 3D tensor of shape [batch, seq_len, n_heads * head_dim].
 *
 * Used to isolate attention heads so self-attention can be computed
 * head-by-head.
 *
 * Example:
 *   If src shape is [2, 128, 256], calling reshape_to_4d(src, 8, 32)
 *   returns a tensor of shape [2, 8, 128, 32].
 *
 * @param src Input 3D tensor to reshape.
 * @param n_heads Number of attention heads.
 * @param head_dim Dimension of each attention head.
 * @return Tensor Reshaped 4D tensor.
 */

Tensor reshape_to_3d(const Tensor &src) {
  size_t batch = src.shape()[0];
  size_t n_heads = src.shape()[1];
  size_t seq_len = src.shape()[2];
  size_t head_dim = src.shape()[3];

  Tensor dest({batch, seq_len, n_heads * head_dim}, 0.0f);

  for (size_t b = 0; b < batch; ++b) {

    for (size_t h = 0; h < n_heads; h++) {
      for (size_t s = 0; s < seq_len; ++s) {
        for (size_t d = 0; d < head_dim; ++d) {
          dest(b, s, h * head_dim + d) = src(b, h, s, d);
        }
      }
    }
  }
  return dest;
}

/**
 * @brief Performs the forward pass of the Attention layer.
 *
 * Flow:
 * 1. Normalize input using RMSNorm.
 * 2. Project normalized states to Q, K, V.
 * 3. Reshape Q, K, V to 4D to separate heads.
 * 4. Apply Rotary Positional Embeddings (RoPE) to Q and K.
 * 5. Compute GQA scores, apply causal mask, softmax, and weighted sum with V.
 * 6. Concat heads and project with Output projection Wo.
 *
 * @param x Input tensor of shape [batch_size, seq_len, hidden_dim].
 * @param rope Rotary Position Embedding (RoPE) helper.
 * @param cache Optional pointer to Key-Value Cache.
 * @param pos_offset Optional position offset for cache mapping.
 * @return Tensor Attention output of shape [batch_size, seq_len, hidden_dim].
 */
Tensor Attention::forward(const Tensor &x, const RoPE &rope, KVCache *cache,
                          size_t pos_offset) const {

  if (x.shape().size() != 3) {
    throw std::invalid_argument(
        "Input tensor must have 3 dimensions (batch, seq_len, hidden_dim)");
  }
  if (x.shape()[2] != config_.hidden_dim) {
    throw std::invalid_argument(
        "Input tensor hidden dim must match model config");
  }

  const char *gpu_enabled_env = std::getenv("GPU_ENABLED");
  bool use_gpu = false;
  if (gpu_enabled_env && std::string(gpu_enabled_env) == "1") {
    metal_bridge::initialize();
    if (metal_bridge::is_available()) {
      use_gpu = true;
    }
  }

  RMSNorm rms_norm(config_.hidden_dim, config_.rms_norm_eps);

  Tensor x_norm = rms_norm.forward(x);

  Tensor q4 =
      reshape_to_4d(x_norm.matmul(Wq_), config_.n_heads, config_.head_dim);
  Tensor k4 =
      reshape_to_4d(x_norm.matmul(Wk_), config_.n_kv_heads, config_.head_dim);
  Tensor v4 =
      reshape_to_4d(x_norm.matmul(Wv_), config_.n_kv_heads, config_.head_dim);

  rope.forward(q4, k4);

  size_t batch = x.shape()[0];
  size_t seq_len = x.shape()[1];
  size_t gqa_factor = config_.n_heads / config_.n_kv_heads;
  float scale = 1.0f / std::sqrt(static_cast<float>(config_.head_dim));

  // Initialize the output tensor with shape [batch, seq_len, hidden_dim]
  Tensor attn_output({batch, seq_len, config_.hidden_dim}, 0.0f);

  // GCD-parallel dispatch over (batch × n_heads): each (b,h) pair writes to
  // non-overlapping output channels [h*head_dim, (h+1)*head_dim), so no mutex
  // needed.
  const float *q4_ptr = q4.data().data();
  const float *k4_ptr = k4.data().data();
  const float *v4_ptr = v4.data().data();
  float *out_ptr = attn_output.data().data();

  const size_t n_h = config_.n_heads;
  const size_t h_d = config_.head_dim;
  const size_t n_kv = config_.n_kv_heads;

  if (use_gpu) {

    metal_bridge::GQAParams gqa_params = {
        .batch = static_cast<uint32_t>(batch),
        .n_q_heads = static_cast<uint32_t>(n_h),
        .n_kv_heads = static_cast<uint32_t>(n_kv),
        .seq_len = static_cast<uint32_t>(seq_len),
        .head_dim = static_cast<uint32_t>(h_d),
    };
    auto start = std::chrono::high_resolution_clock::now();
    metal_bridge::gemm_gqa(gqa_params, q4_ptr, k4_ptr, v4_ptr, out_ptr);
    auto end = std::chrono::high_resolution_clock::now();
    metal_bridge::accum_gpu_time_ms +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1000.0;
    metal_bridge::count_gpu_calls++;
  } else {
    dispatch_apply(
        batch * n_h, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0),
        ^(size_t bh) {
          const size_t b = bh / n_h;
          const size_t h = bh % n_h;
          const size_t kv_h = h / gqa_factor;

          std::vector<float> scores(seq_len, 0.0f); // thread-local scratch

          for (size_t s_q = 0; s_q < seq_len; ++s_q) {
            const float *q_row =
                q4_ptr + (b * n_h * seq_len + h * seq_len + s_q) * h_d;

            float max_score = -INFINITY;
            for (size_t s_k = 0; s_k <= s_q; ++s_k) {
              const float *k_row =
                  k4_ptr + (b * n_kv * seq_len + kv_h * seq_len + s_k) * h_d;
              float dot = 0.0f;
              for (size_t d = 0; d < h_d; ++d)
                dot += q_row[d] * k_row[d];
              scores[s_k] = dot * scale;
              if (scores[s_k] > max_score)
                max_score = scores[s_k];
            }

            // Stable softmax: compute exp in-place, reuse stored values
            float sum_exp = 0.0f;
            for (size_t s_k = 0; s_k <= s_q; ++s_k) {
              scores[s_k] = std::exp(scores[s_k] - max_score);
              sum_exp += scores[s_k];
            }

            // Weighted accumulation — thread-safe: h writes to
            // h*h_d..(h+1)*h_d-1
            float *out_row =
                out_ptr + (b * seq_len + s_q) * (n_h * h_d) + h * h_d;
            for (size_t s_k = 0; s_k <= s_q; ++s_k) {
              const float prob = scores[s_k] / sum_exp;
              const float *v_row =
                  v4_ptr + (b * n_kv * seq_len + kv_h * seq_len + s_k) * h_d;
              for (size_t d = 0; d < h_d; ++d)
                out_row[d] += prob * v_row[d];
            }
          }
        });
  }
  Tensor out = attn_output.matmul(Wo_);
  return out;
}
/**
 * @brief Helper function to perform GQA backward pass calculations for a single
 * query token.
 *
 * Implements softmax backward and accumulates gradients w.r.t queries, keys,
 * and values.
 */
static void
gqa_backward_query_token(size_t b, size_t h, size_t kv_h, size_t s_q,
                         size_t seq_len, size_t head_dim, float scale,
                         const Tensor &q4, const Tensor &k4, const Tensor &v4,
                         const Tensor &grad_attn_output, Tensor &grad_q4,
                         Tensor &grad_k4, Tensor &grad_v4) {

  std::vector<float> raw_scores(seq_len);
  std::vector<float> prob(seq_len);
  std::vector<float> dP(seq_len);

  // Step A: Recompute probabilities (prob[s_k])
  float max_score = -INFINITY;
  for (size_t s_k = 0; s_k <= s_q; ++s_k) {
    float dot = 0.0f;
    for (size_t d = 0; d < head_dim; ++d) {
      dot += q4(b, h, s_q, d) * k4(b, kv_h, s_k, d);
    }
    raw_scores[s_k] = dot * scale;
    if (raw_scores[s_k] > max_score) {
      max_score = raw_scores[s_k];
    }
  }

  float sum_exp = 0.0f;
  for (size_t s_k = 0; s_k <= s_q; ++s_k) {
    sum_exp += std::exp(raw_scores[s_k] - max_score);
  }

  for (size_t s_k = 0; s_k <= s_q; ++s_k) {
    prob[s_k] = std::exp(raw_scores[s_k] - max_score) / sum_exp;
  }

  // Step B: Backprop w.r.t probabilities (dP) and accumulate grad_v4
  float sum_dP_prob = 0.0f;
  for (size_t s_k = 0; s_k <= s_q; ++s_k) {
    float dot_g_v = 0.0f;
    for (size_t d = 0; d < head_dim; ++d) {
      float g = grad_attn_output(b, s_q, h * head_dim + d);
      dot_g_v += g * v4(b, kv_h, s_k, d);
      grad_v4(b, kv_h, s_k, d) += prob[s_k] * g;
    }
    dP[s_k] = dot_g_v;
    sum_dP_prob += dP[s_k] * prob[s_k];
  }

  // Step C: Backprop through Softmax and Scaled Dot-Product to grad_q4 &
  // grad_k4
  for (size_t s_k = 0; s_k <= s_q; ++s_k) {
    float dS = prob[s_k] * (dP[s_k] - sum_dP_prob);
    float dS_scaled = dS * scale;
    for (size_t d = 0; d < head_dim; ++d) {
      grad_q4(b, h, s_q, d) += dS_scaled * k4(b, kv_h, s_k, d);
      grad_k4(b, kv_h, s_k, d) += dS_scaled * q4(b, h, s_q, d);
    }
  }
}

/**
 * @brief Performs the backward pass of the Attention layer.
 *
 * @param grad_output Gradient from the next layer.
 * @param x Input tensor.
 * @param rope Rotary Position Embedding (RoPE) helper.
 * @param grad_Wq Gradient for Wq.
 * @param grad_Wk Gradient for Wk.
 * @param grad_Wv Gradient for Wv.
 * @param grad_Wo Gradient for Wo.
 * @param cache Optional pointer to Key-Value Cache.
 * @param pos_offset Optional position offset for cache mapping.
 * @return Tensor Gradient for the input tensor x.
 *
 */
Tensor Attention::backward(const Tensor &grad_output, const Tensor &x,
                           const RoPE &rope, Tensor &grad_Wq, Tensor &grad_Wk,
                           Tensor &grad_Wv, Tensor &grad_Wo, KVCache *cache,
                           size_t pos_offset) const {
  if (grad_output.shape() != x.shape()) {
    throw std::invalid_argument("grad_output shape must match input x shape");
  }
  if (grad_Wq.shape() != Wq_.shape()) {
    throw std::invalid_argument("grad_Wq shape must match Wq_ shape");
  }
  if (grad_Wk.shape() != Wk_.shape()) {
    throw std::invalid_argument("grad_Wk shape must match Wk_ shape");
  }
  if (grad_Wv.shape() != Wv_.shape()) {
    throw std::invalid_argument("grad_Wv shape must match Wv_ shape");
  }
  if (grad_Wo.shape() != Wo_.shape()) {
    throw std::invalid_argument("grad_Wo shape must match Wo_ shape");
  }
  size_t batch = x.shape()[0];
  size_t seq_len = x.shape()[1];
  size_t hidden_dim = x.shape()[2];
  size_t n_heads = config_.n_heads;
  size_t n_kv_heads = config_.n_kv_heads;
  size_t head_dim = config_.head_dim;
  size_t gqa_factor = n_heads / n_kv_heads;
  float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));

  // 1. Recompute forward states
  RMSNorm rms_norm(hidden_dim, config_.rms_norm_eps);
  Tensor x_norm = rms_norm.forward(x);

  Tensor q4 = reshape_to_4d(x_norm.matmul(Wq_), n_heads, head_dim);
  Tensor k4 = reshape_to_4d(x_norm.matmul(Wk_), n_kv_heads, head_dim);
  Tensor v4 = reshape_to_4d(x_norm.matmul(Wv_), n_kv_heads, head_dim);
  rope.forward(q4, k4);

  Tensor attn_output({batch, seq_len, hidden_dim}, 0.0f);

  // --- Run the backward recomputed forward pass with GCD parallel dispatch ---
  {
    const float *q4_ptr = q4.data().data();
    const float *k4_ptr = k4.data().data();
    const float *v4_ptr = v4.data().data();
    float *out_ptr = attn_output.data().data();
    const size_t n_h = n_heads;
    const size_t h_d = head_dim;
    const size_t n_kv = n_kv_heads;

    dispatch_apply(
        batch * n_h, dispatch_get_global_queue(DISPATCH_QUEUE_PRIORITY_HIGH, 0),
        ^(size_t bh) {
          const size_t b = bh / n_h;
          const size_t h = bh % n_h;
          const size_t kv_h = h / gqa_factor;

          std::vector<float> scores(seq_len, 0.0f);

          for (size_t s_q = 0; s_q < seq_len; ++s_q) {
            const float *q_row =
                q4_ptr + (b * n_h * seq_len + h * seq_len + s_q) * h_d;
            float max_score = -INFINITY;
            for (size_t s_k = 0; s_k <= s_q; ++s_k) {
              const float *k_row =
                  k4_ptr + (b * n_kv * seq_len + kv_h * seq_len + s_k) * h_d;
              float dot = 0.0f;
              for (size_t d = 0; d < h_d; ++d)
                dot += q_row[d] * k_row[d];
              scores[s_k] = dot * scale;
              if (scores[s_k] > max_score)
                max_score = scores[s_k];
            }
            float sum_exp = 0.0f;
            for (size_t s_k = 0; s_k <= s_q; ++s_k) {
              scores[s_k] = std::exp(scores[s_k] - max_score);
              sum_exp += scores[s_k];
            }
            float *out_row =
                out_ptr + (b * seq_len + s_q) * (n_h * h_d) + h * h_d;
            for (size_t s_k = 0; s_k <= s_q; ++s_k) {
              const float prob = scores[s_k] / sum_exp;
              const float *v_row =
                  v4_ptr + (b * n_kv * seq_len + kv_h * seq_len + s_k) * h_d;
              for (size_t d = 0; d < h_d; ++d)
                out_row[d] += prob * v_row[d];
            }
          }
        });
  }

  // 2. grad_Wo[NH*HD, H] = attn_output[B*S, NH*HD].T @ grad_output[B*S, H]
  grad_Wo = attn_output.reshape({batch * seq_len, hidden_dim}).transpose().matmul(grad_output.reshape({batch * seq_len, hidden_dim}));

  // 3. Backpropagate to attention head outputs
  Tensor grad_attn_output = grad_output.matmul(Wo_.transpose());

  // 4. Initialize head gradients
  Tensor grad_q4({batch, n_heads, seq_len, head_dim}, 0.0f);
  Tensor grad_k4({batch, n_kv_heads, seq_len, head_dim}, 0.0f);
  Tensor grad_v4({batch, n_kv_heads, seq_len, head_dim}, 0.0f);

  // 5. GQA attention backpropagation loops
  const char *gpu_enabled_env = std::getenv("GPU_ENABLED");
  bool use_gpu = false;
  if (gpu_enabled_env && std::string(gpu_enabled_env) == "1") {
    metal_bridge::initialize();
    if (metal_bridge::is_available()) {
      use_gpu = true;
    }
  }

  if (use_gpu) {
    metal_bridge::GQABackwardParams params;
    params.batch = batch;
    params.n_q_heads = n_heads;
    params.n_kv_heads = n_kv_heads;
    params.seq_len = seq_len;
    params.head_dim = head_dim;

    metal_bridge::gqa_backward(params,
                               q4.data().data(), k4.data().data(), v4.data().data(),
                               grad_attn_output.data().data(),
                               grad_q4.data().data(), grad_k4.data().data(), grad_v4.data().data());
  } else {
    for (size_t b = 0; b < batch; ++b) {
      for (size_t h = 0; h < n_heads; ++h) {
        size_t kv_h = h / gqa_factor;
        for (size_t s_q = 0; s_q < seq_len; ++s_q) {
          gqa_backward_query_token(b, h, kv_h, s_q, seq_len, head_dim, scale, q4,
                                   k4, v4, grad_attn_output, grad_q4, grad_k4,
                                   grad_v4);
        }
      }
    }
  }

  // 6. Backpropagate RoPE
  rope.backward(grad_q4, grad_k4);

  // 7. Reshape gradients back to 3D
  Tensor grad_q_proj = reshape_to_3d(grad_q4);
  Tensor grad_k_proj = reshape_to_3d(grad_k4);
  Tensor grad_v_proj = reshape_to_3d(grad_v4);

  // 8. Replace 3-nested weight accumulation loops with cblas_sgemm
  // grad_Wq[H, NH*HD] += x_norm[B*S, H].T @ grad_q_proj[B*S, NH*HD]
  // grad_Wk[H, NKV*HD] += x_norm[B*S, H].T @ grad_k_proj[B*S, NKV*HD]
  // grad_Wv[H, NKV*HD] += x_norm[B*S, H].T @ grad_v_proj[B*S, NKV*HD]
  grad_Wq = x_norm.reshape({batch * seq_len, hidden_dim}).transpose().matmul(grad_q_proj.reshape({batch * seq_len, n_heads * head_dim}));
  grad_Wk = x_norm.reshape({batch * seq_len, hidden_dim}).transpose().matmul(grad_k_proj.reshape({batch * seq_len, n_kv_heads * head_dim}));
  grad_Wv = x_norm.reshape({batch * seq_len, hidden_dim}).transpose().matmul(grad_v_proj.reshape({batch * seq_len, n_kv_heads * head_dim}));

  // 9. Compute gradient w.r.t normalized input
  Tensor grad_x_norm = grad_q_proj.matmul(Wq_.transpose());
  grad_x_norm.add_(grad_k_proj.matmul(Wk_.transpose()));
  grad_x_norm.add_(grad_v_proj.matmul(Wv_.transpose()));

  // 10. Backpropagate through RMSNorm to get gradient w.r.t raw input x
  Tensor grad_weight_dummy({hidden_dim}, 0.0f);
  Tensor grad_x = rms_norm.backward(grad_x_norm, x, grad_weight_dummy);

  return grad_x;
}
