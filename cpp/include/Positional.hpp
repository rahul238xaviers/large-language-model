/**
 * @file Positional.hpp
 * @brief Rotary Position Embeddings (RoPE) layer declaration
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * RoPE embeds relative positional information into the Queries and Keys of
 * the attention mechanism by rotating pairs of dimensions in the complex plane.
 */

#pragma once

#include "Tensor.hpp"
#include <vector>

class RoPE {
public:
  // Constructor: head_dim is the attention head size
  // max_seq_len is the maximum sentence size
  RoPE(size_t head_dim, size_t max_seq_len, float base);

  // Applies rotary position embeddings to Query and Key tensors in-place
  void forward(Tensor &q, Tensor &k) const;

  // Accessors for testing
  std::vector<std::vector<float>> cos_table() const {
    size_t half_dim = head_dim_ / 2;
    std::vector<std::vector<float>> res(max_seq_len_, std::vector<float>(half_dim));
    for (size_t s = 0; s < max_seq_len_; s++) {
      for (size_t i = 0; i < half_dim; i++) {
        res[s][i] = cos_table_[s * half_dim + i];
      }
    }
    return res;
  }
  std::vector<std::vector<float>> sin_table() const {
    size_t half_dim = head_dim_ / 2;
    std::vector<std::vector<float>> res(max_seq_len_, std::vector<float>(half_dim));
    for (size_t s = 0; s < max_seq_len_; s++) {
      for (size_t i = 0; i < half_dim; i++) {
        res[s][i] = sin_table_[s * half_dim + i];
      }
    }
    return res;
  }
  // Applies inverse rotary position embeddings to Query and Key gradients
  // in-place
  void backward(Tensor &grad_q, Tensor &grad_k) const;

private:
  size_t head_dim_;
  size_t max_seq_len_;
  float base_;

  // Pre-computed tables of shape [max_seq_len * (head_dim / 2)]
  std::vector<float> cos_table_;
  std::vector<float> sin_table_;

  void precompute_tables();
};
