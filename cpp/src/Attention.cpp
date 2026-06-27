/**
 * @file Attention.cpp
 * @brief Implementation of the GQA (Grouped Query Attention) layer
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * This file implements the projections, reshaping, and self-attention
 * operations for Grouped Query Attention (GQA).
 */

#include "Attention.hpp"
#include "RMSNorm.hpp"
#include "Tensor.hpp"
#include <cmath>
#include <cstddef>
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
      Wo_({config.n_heads * config.head_dim, config.hidden_dim}) {};
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

  // We allocate a small vector to hold intermediate raw scores for softmax
  std::vector<float> raw_scores(seq_len);

  for (size_t b = 0; b < batch; ++b) {
    for (size_t h = 0; h < config_.n_heads; ++h) {
      size_t kv_h = h / gqa_factor;

      for (size_t s_q = 0; s_q < seq_len; ++s_q) {
        // --- 1. Compute raw dot-product scores (up to s_q for causal mask) ---
        float max_score = -INFINITY;
        for (size_t s_k = 0; s_k <= s_q; ++s_k) {
          float dot = 0.0f;
          for (size_t d = 0; d < config_.head_dim; ++d) {
            dot += q4(b, h, s_q, d) * k4(b, kv_h, s_k, d);
          }
          raw_scores[s_k] = dot * scale;
          if (raw_scores[s_k] > max_score) {
            max_score = raw_scores[s_k];
          }
        }

        // --- 2. Compute Softmax denominator (sum of exp) ---
        float sum_exp = 0.0f;
        for (size_t s_k = 0; s_k <= s_q; ++s_k) {
          sum_exp += std::exp(raw_scores[s_k] - max_score);
        }

        // --- 3. Weighted accumulation of Values ---
        for (size_t s_k = 0; s_k <= s_q; ++s_k) {
          float prob = std::exp(raw_scores[s_k] - max_score) / sum_exp;
          for (size_t d = 0; d < config_.head_dim; ++d) {
            size_t out_channel = h * config_.head_dim + d;
            attn_output(b, s_q, out_channel) += prob * v4(b, kv_h, s_k, d);
          }
        }
      }
    }
  }
  Tensor out = attn_output.matmul(Wo_);
  return out;
}
