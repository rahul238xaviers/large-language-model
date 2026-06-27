/**
 * @file Transformer.cpp
 * @brief Implementation of the Transformer model and layer blocks
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * Implements constructors and forward pass operations for TransformerLayer and
 * the Transformer network.
 */

#include "Transformer.hpp"
#include "Activations.hpp"
#include "Tensor.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <vector>

// Construct a single transformer layer block
TransformerLayer::TransformerLayer(const ModelConfig &config)
    : attn_norm(config.hidden_dim, config.rms_norm_eps), attn(config),
      ffn_norm(config.hidden_dim, config.rms_norm_eps),
      w_gate({config.hidden_dim, config.intermediate_dim}, 0.0f),
      w_up({config.hidden_dim, config.intermediate_dim}, 0.0f),
      w_down({config.intermediate_dim, config.hidden_dim}, 0.0f) {}

// Construct the complete Transformer model
Transformer::Transformer(const ModelConfig &config)
    : config_(config),
      token_embeddings_({config.vocab_size, config.hidden_dim}, 0.0f),
      final_norm_(config.hidden_dim, config.rms_norm_eps),
      output_projection_({config.hidden_dim, config.vocab_size}, 0.0f),
      rope_(config.head_dim, config.max_seq_len, config.rope_base) {

  layers_.reserve(config.n_layers);
  for (size_t i = 0; i < config.n_layers; ++i) {
    layers_.emplace_back(config);
  }
}
Tensor Transformer::forward(const Tensor &tokens, KVCache *cache) const {
  if (tokens.shape().size() != 2) {
    throw std::runtime_error("Input tokens must be 2D (batch_size, seq_len)");
  }

  size_t batch_size = tokens.shape()[0];
  size_t seq_len = tokens.shape()[1];

  Tensor h = Tensor({batch_size, seq_len, config_.hidden_dim});

  for (size_t b = 0; b < batch_size; b++) {
    for (size_t s = 0; s < seq_len; s++) {

      float token_id = tokens(b, s);
      if (token_id > config_.vocab_size - 1) {
        throw std::runtime_error("token_id is out of range");
      }
      size_t id = static_cast<size_t>(token_id);
      for (size_t d = 0; d < config_.hidden_dim; d++) {
        h(b, s, d) = token_embeddings_(id, d);
      }
    }
  }
  return h;
}