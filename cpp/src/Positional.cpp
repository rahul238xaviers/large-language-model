/**
 * @file Positional.cpp
 * @brief Implementation of Rotary Position Embeddings (RoPE) forward & backward passes
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * This file implements precomputation of trigonometric rotation tables,
 * forward rotation application to Query and Key tensors, and inverse rotation
 * gradient backpropagation.
 *
 * By grouping adjacent coordinates into pairs (x0, x1) and rotating them,
 * we embed relative distance information directly into the dot-product
 * attention scores.
 */

#include "Positional.hpp"
#include <cmath>
#include <cstddef>
#include <stdexcept>

/**
 * @brief Construct a new RoPE object.
 *
 * Verifies head dimension properties and triggers table precomputation.
 *
 * @param head_dim Size of individual attention heads (must be even).
 * @param max_seq_len Maximum sequence length supported by precomputed tables.
 * @param base Base constant for frequency computations (e.g. 10000.0f).
 */
RoPE::RoPE(size_t head_dim, size_t max_seq_len, float base)
    : head_dim_(head_dim), max_seq_len_(max_seq_len), base_(base) {

  if (head_dim % 2 != 0) {
    throw std::invalid_argument("head_dim must be divisible by 2");
  }

  precompute_tables();
}

/**
 * @brief Precomputes the rotation matrices (sine and cosine tables)
 *
 * Sets up frequency values (theta) based on the dimensions and base constant,
 * then evaluates cos(pos * theta) and sin(pos * theta) for all positions.
 */
void RoPE::precompute_tables() {

  size_t half_dim = head_dim_ / 2;

  cos_table_.resize(max_seq_len_, std::vector<float>(half_dim));
  sin_table_.resize(max_seq_len_, std::vector<float>(half_dim));
  std::vector<float> theta(half_dim);

  for (size_t i = 0; i < half_dim; i++) {
    float exponent = (2.0f * i) / head_dim_;
    theta[i] = 1.0f / std::pow(base_, exponent);
  }

  for (size_t pos = 0; pos < max_seq_len_; pos++) {

    for (size_t i = 0; i < half_dim; i++) {
      cos_table_[pos][i] = std::cos(pos * theta[i]);
      sin_table_[pos][i] = std::sin(pos * theta[i]);
    }
  }
}

/**
 * @brief Applies the precomputed rotary positional embeddings to Q and K.
 *
 * Both tensors must be 4D shape: [batch, heads, seq_len, head_dim].
 * Rotation is applied to coordinate pairs in-place.
 *
 * @param q Query tensor to rotate in-place.
 * @param k Key tensor to rotate in-place.
 */
void RoPE::forward(Tensor &q, Tensor &k) const {

  if (q.shape().size() != 4 || k.shape().size() != 4) {
    throw std::invalid_argument("Q and K must have same shape and that is 4");
  }

  if (q.shape()[0] != k.shape()[0]) {
    throw std::invalid_argument("Batch sizes must match");
  }

  if (q.shape()[2] != k.shape()[2] || q.shape()[2] > max_seq_len_) {
    throw std::invalid_argument(
        "Sequence lengths must match and should be less than {max_seq_len}" +
        std::to_string(max_seq_len_));
  }

  if (q.shape()[3] != head_dim_ || k.shape()[3] != head_dim_) {
    throw std::invalid_argument("Head dimensions must match");
  }

  auto result_t = [&](Tensor &t) {
    size_t batch_size = t.shape()[0];
    size_t head_count = t.shape()[1];
    size_t seq_len = t.shape()[2];
    size_t half_dim = head_dim_ / 2;

    for (size_t b = 0; b < batch_size; b++) {
      for (size_t h = 0; h < head_count; h++) {
        for (size_t s = 0; s < seq_len; s++) {
          for (size_t i = 0; i < half_dim; i++) {
            size_t idx0 = 2 * i;
            size_t idx1 = 2 * i + 1;

            float x0 = t(b, h, s, idx0);
            float x1 = t(b, h, s, idx1);

            float cos_val = cos_table_[s][i];
            float sin_val = sin_table_[s][i];

            t(b, h, s, idx0) = x0 * cos_val - x1 * sin_val;
            t(b, h, s, idx1) = x0 * sin_val + x1 * cos_val;
          }
        }
      }
    }
  };

  result_t(q);
  result_t(k);
}

/**
@brief Applies inverse rotary position embeddings to Query and Key gradients
       in-place.
@param grad_q Gradient w.r.t. the Query tensor.
@param grad_k Gradient w.r.t. the Key tensor.
@throws std::invalid_argument if the input tensors have invalid shapes.
*/
void RoPE::backward(Tensor &grad_q, Tensor &grad_k) const {

  if (grad_q.shape().size() != 4 || grad_k.shape().size() != 4) {
    throw std::invalid_argument("Q and K must have same shape and that is 4");
  }

  if (grad_q.shape()[0] != grad_k.shape()[0]) {
    throw std::invalid_argument("Batch sizes must match");
  }

  if (grad_q.shape()[2] != grad_k.shape()[2] ||
      grad_q.shape()[2] > max_seq_len_) {
    throw std::invalid_argument(
        "Sequence lengths must match and should be less than {max_seq_len}" +
        std::to_string(max_seq_len_));
  }

  if (grad_q.shape()[3] != head_dim_ || grad_k.shape()[3] != head_dim_) {
    throw std::invalid_argument("Head dimensions must match");
  }

  auto result_t = [&](Tensor &t) {
    size_t batch_size = t.shape()[0];
    size_t head_count = t.shape()[1];
    size_t seq_len = t.shape()[2];
    size_t half_dim = head_dim_ / 2;

    for (size_t b = 0; b < batch_size; b++) {
      for (size_t h = 0; h < head_count; h++) {
        for (size_t s = 0; s < seq_len; s++) {
          for (size_t i = 0; i < half_dim; i++) {
            size_t idx0 = 2 * i;
            size_t idx1 = 2 * i + 1;

            float x0 = t(b, h, s, idx0);
            float x1 = t(b, h, s, idx1);

            float cos_val = cos_table_[s][i];
            float sin_val = sin_table_[s][i];

            t(b, h, s, idx0) = x0 * cos_val + x1 * sin_val;
            t(b, h, s, idx1) = -x0 * sin_val + x1 * cos_val;
          }
        }
      }
    }
  };

  result_t(grad_q);
  result_t(grad_k);
}
