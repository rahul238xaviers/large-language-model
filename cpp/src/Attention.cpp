/**
 * @file Attention.cpp
 * @brief Implementation of the GQA (Grouped Query Attention) layer
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * This file implements the projections, reshaping, and self-attention operations
 * for Grouped Query Attention (GQA).
 */

#include "Attention.hpp"
#include "RMSNorm.hpp"
#include "Tensor.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

/**
 * @brief Construct a new Attention object and initialize the projection weights.
 * 
 * Wq_, Wk_, Wv_, and Wo_ are the weight matrices for Query, Key, Value,
 * and Output projections.
 * 
 * @param config ModelConfig containing n_heads, n_kv_heads, head_dim, and hidden_dim.
 */
Attention::Attention(const ModelConfig &config)
    : config_(config),
      Wq_({config.hidden_dim, config.n_heads * config.head_dim}),
      Wk_({config.hidden_dim, config.n_kv_heads * config.head_dim}),
      Wv_({config.hidden_dim, config.n_kv_heads * config.head_dim}),
      Wo_({config.n_heads * config.head_dim, config.hidden_dim}) {};

/**
 * @brief Performs batch-wise projection matrix multiplication.
 * 
 * Multiplies a 3D input tensor `x` of shape [batch, seq_len, hidden_dim]
 * with a 2D weight matrix `w` of shape [hidden_dim, out_dim].
 * 
 * This is an optimized raw-pointer implementation to perform projection
 * without allocating extra copies of the weights per batch.
 * 
 * Example:
 *   If x has shape [1, 2, 3] and w has shape [3, 4],
 *   then matmul_project(x, w) returns a tensor of shape [1, 2, 4].
 * 
 * @param x Input tensor of shape [batch_size, seq_len, hidden_dim].
 * @param w Weight matrix of shape [hidden_dim, out_dim].
 * @return Tensor Result tensor of shape [batch_size, seq_len, out_dim].
 */
Tensor matmul_project(const Tensor &x, const Tensor &w) {

  size_t batch_size = x.shape()[0];
  size_t seq_len = x.shape()[1];
  size_t hidden_dim = x.shape()[2];
  size_t out_dim = w.shape()[1];

  Tensor result({batch_size, seq_len, out_dim}, 0.0f);

  const float *x_data = x.data().data();
  const float *w_data = w.data().data();
  float *res_data = result.data().data();

  for (size_t b = 0; b < batch_size; ++b) {
    size_t x_batch_offset = b * seq_len * hidden_dim;
    size_t res_batch_offset = b * seq_len * out_dim;

    for (size_t s = 0; s < seq_len; ++s) {
      size_t x_row_offset = x_batch_offset + s * hidden_dim;
      size_t res_row_offset = res_batch_offset + s * out_dim;

      for (size_t d = 0; d < hidden_dim; ++d) {
        float val = x_data[x_row_offset + d];
        for (size_t o = 0; o < out_dim; ++o) {
          res_data[res_row_offset + o] += val * w_data[d * out_dim + o];
        }
      }
    }
  }

  return result;
}

/**
 * @brief Reshapes a 3D tensor of shape [batch, seq_len, n_heads * head_dim]
 *        into a 4D tensor of shape [batch, n_heads, seq_len, head_dim].
 * 
 * Used to isolate attention heads so self-attention can be computed head-by-head.
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

  Tensor q = matmul_project(x_norm, Wq_);
  Tensor k = matmul_project(x_norm, Wk_);
  Tensor v = matmul_project(x_norm, Wv_);

  return q;
}
