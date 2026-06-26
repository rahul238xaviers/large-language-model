#pragma once
#include "Tensor.hpp"
#include <cstddef>

struct ModelConfig {
  size_t vocab_size;
  size_t hidden_dim;
  size_t intermediate_dim;
  size_t n_layers;
  size_t n_heads;
  size_t n_kv_heads;
  size_t head_dim;
  size_t max_seq_len;
  float rope_base;

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
    return model_toy;
  }
};

struct KVCache {
  std::vector<Tensor> key_cache;
  std::vector<Tensor> value_cache;
};