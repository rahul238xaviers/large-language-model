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
  const std::vector<std::vector<float>> &cos_table() const {
    return cos_table_;
  }
  const std::vector<std::vector<float>> &sin_table() const {
    return sin_table_;
  }

private:
  size_t head_dim_;
  size_t max_seq_len_;
  float base_;

  // Pre-computed tables of shape [max_seq_len, head_dim / 2]
  std::vector<std::vector<float>> cos_table_;
  std::vector<std::vector<float>> sin_table_;

  void precompute_tables();
};
