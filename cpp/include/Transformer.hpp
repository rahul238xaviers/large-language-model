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

  // Accessors
  const Tensor &token_embeddings() const { return token_embeddings_; }
  Tensor &token_embeddings() { return token_embeddings_; }
  const Tensor &output_projection() const { return output_projection_; }
  Tensor &output_projection() { return output_projection_; }
  const std::vector<TransformerLayer> &layers() const { return layers_; }
  std::vector<TransformerLayer> &layers() { return layers_; }

private:
  ModelConfig config_;
  Tensor token_embeddings_;
  std::vector<TransformerLayer> layers_;
  RMSNorm final_norm_;
  Tensor output_projection_;
  RoPE rope_;
};
