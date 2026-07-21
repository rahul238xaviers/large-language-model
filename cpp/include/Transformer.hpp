/**
 * @file Transformer.hpp
 * @brief Top-level Transformer model and layers declarations
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * Defines the TransformerLayer and Transformer classes. Coordinates the token
 * embedding lookup, stacked transformer layers, and logits projection.
 */

#pragma once

#include "Attention.hpp"
#include "Positional.hpp"
#include "RMSNorm.hpp"
#include "Tensor.hpp"
#include "TransformerConfig.hpp"
#include <vector>

/**
 * @brief Represents a single Transformer block layer.
 *
 * Each block applies self-attention (GQA) and a Feed-Forward Network (SwiGLU).
 */
struct TransformerLayer {
  RMSNorm attn_norm;
  Attention attn;
  RMSNorm ffn_norm;
  Tensor w_gate;
  Tensor w_up;
  Tensor w_down;

  TransformerLayer(const ModelConfig &config);
  // Single layer backward pass
  Tensor backward(const Tensor &grad_output, const Tensor &h_in,
                  Tensor &grad_w_gate, Tensor &grad_w_up, Tensor &grad_w_down,
                  Tensor &grad_Wq, Tensor &grad_Wk, Tensor &grad_Wv,
                  Tensor &grad_Wo, const RoPE &rope,
                  KVCache *cache = nullptr) const;
};

/**
 * @brief The complete Transformer model.
 *
 * Takes token IDs, maps them to embeddings, runs them through the stacked
 * layers, normalizes them, and projects them to token probability logits.
 */
class Transformer {
public:
  Transformer(const ModelConfig &config);

  /**
   * @brief Performs the forward pass of the entire Transformer model.
   *
   * Maps tokens to embeddings, runs them through each stack layer sequentially
   * using residual connections, normalizes, and projects to logits.
   *
   * @param tokens 2D integer tensor of token IDs with shape [batch_size,
   * seq_len].
   * @param cache Optional pointer to Key-Value Cache.
   * @return Tensor Logits tensor of shape [batch_size, seq_len, vocab_size].
   */
  Tensor forward(const Tensor &tokens, KVCache *cache = nullptr) const;
  // Store intermediate hidden states during forward for reuse in backward.
  // This eliminates the expensive run_forward_cache recomputation.
  void clear_h_cache() const { h_cache_.clear(); }
  const std::vector<Tensor> &h_cache() const { return h_cache_; }

  // Entire model backward pass
  Tensor backward(const Tensor &grad_logits, const Tensor &tokens,
                  std::vector<Tensor> &grad_w_gate,
                  std::vector<Tensor> &grad_w_up,
                  std::vector<Tensor> &grad_w_down,
                  std::vector<Tensor> &grad_Wq, std::vector<Tensor> &grad_Wk,
                  std::vector<Tensor> &grad_Wv, std::vector<Tensor> &grad_Wo,
                  Tensor &grad_embeddings, Tensor &grad_output_projection,
                  const RoPE &rope) const;

  // Accessors
  const Tensor &token_embeddings() const { return token_embeddings_; }
  Tensor &token_embeddings() { return token_embeddings_; }
  const Tensor &output_projection() const { return output_projection_; }
  Tensor &output_projection() { return output_projection_; }
   const std::vector<TransformerLayer> &layers() const { return layers_; }
  std::vector<TransformerLayer> &layers() { return layers_; }
  const RMSNorm &final_norm() const { return final_norm_; }
  RMSNorm &final_norm() { return final_norm_; }
  const ModelConfig &config() const { return config_; }

private:
  ModelConfig config_;
  Tensor token_embeddings_;
  std::vector<TransformerLayer> layers_;
  RMSNorm final_norm_;
  Tensor output_projection_;
  RoPE rope_;
  mutable std::vector<Tensor> h_cache_;
};
