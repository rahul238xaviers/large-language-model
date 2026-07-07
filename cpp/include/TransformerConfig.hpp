/**
 * @file TransformerConfig.hpp
 * @brief Configuration structures for the Transformer model and KV cache
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * Defines ModelConfig (holding hyperparameters like dimensions, layers, and
 * heads) and KVCache structures used during autoregressive generation.
 */

#pragma once
#include "Tensor.hpp"
#include <cstddef>

struct ModelConfig {
  /**
   * @brief Vocabulary size of the language model tokenizer.
   * @note Dictates final logits classifier shape [M, vocab_size].
   */
  size_t vocab_size;

  /**
   * @brief Dimension of the token embeddings and hidden states (H).
   * @note Direct impact on Projection GEMM (gemm_proj.metal).
   *       For optimal performance on GPU hardware matrix cores, hidden_dim
   *       should be a multiple of the threadgroup tile size (64 elements).
   *       Examples of well-aligned values: 512, 1024, 2048, 4096.
   */
  size_t hidden_dim;

  /**
   * @brief Hidden dimension of the Feed-Forward Network (FFN) expansion layer
   * (typically ~4H).
   * @note Direct impact on Feed-Forward GEMM (gemm_ffn.metal).
   *       For optimal GPU ALU occupancy and to prevent execution
   * padding/divergence, intermediate_dim should be a multiple of the FFN tile
   * size (128 elements). If non-aligned (e.g., 5461), the GPU shader fallback
   * path relies on zero-padding, which incurs memory boundary check overhead.
   * Optimal recommendation: 5632.
   */
  size_t intermediate_dim;

  /**
   * @brief Number of Transformer block layers.
   * @note Governs loop iterations and pipeline depth.
   */
  size_t n_layers;

  /**
   * @brief Number of Attention query heads.
   * @note Used to split hidden_dim into head dimensions for multi-head
   * attention.
   */
  size_t n_heads;

  /**
   * @brief Number of Key and Value attention heads (for GQA/MQA).
   * @note Direct impact on GQA Attention GEMM (gemm_gqa.metal). Must divide
   * n_heads evenly.
   */
  size_t n_kv_heads;

  /**
   * @brief Dimensionality of each individual attention head (hidden_dim /
   * n_heads).
   * @note Dictates the query-key dot product dimensions [seq_len, head_dim] x
   * [head_dim, seq_len]. Must be a multiple of 8 (e.g. 64, 128) to utilize
   * physical AMX matrix units.
   */
  size_t head_dim;

  /**
   * @brief Maximum context window length supported.
   * @note Limits the KV cache allocation size in VRAM/Unified memory.
   */
  size_t max_seq_len;

  /**
   * @brief Base frequency for Rotary Position Embeddings (RoPE).
   */
  float rope_base;

  /**
   * @brief Epsilon constant for root-mean-square normalization to prevent
   * division by zero.
   */
  float rms_norm_eps;

  static ModelConfig make_default() {
    ModelConfig model_config;
    model_config.vocab_size = 100256;
    model_config.hidden_dim = 2048;
    model_config.intermediate_dim = 5461;
    model_config.n_layers = 24;
    model_config.n_heads = 16;
    model_config.n_kv_heads = 8;
    model_config.head_dim = 128;
    model_config.max_seq_len = 2048;
    model_config.rope_base = 10000.0f;
    model_config.rms_norm_eps = 1e-5f;
    return model_config;
  }

  static ModelConfig make_toy() {
    ModelConfig model_toy;
    model_toy.vocab_size = 100256;
    model_toy.hidden_dim = 16;
    model_toy.intermediate_dim = 64;
    model_toy.n_layers = 2;
    model_toy.n_heads = 4;
    model_toy.n_kv_heads = 4;
    model_toy.head_dim = 4;
    model_toy.max_seq_len = 128;
    model_toy.rope_base = 10000.0f;
    model_toy.rms_norm_eps = 1e-5f;
    return model_toy;
  }
};

struct KVCache {
  std::vector<Tensor> key_cache;
  std::vector<Tensor> value_cache;
};