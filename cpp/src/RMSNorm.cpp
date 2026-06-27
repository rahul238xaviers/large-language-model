/**
 * @file RMSNorm.cpp
 * @brief Implementation of Root Mean Square Normalization (RMSNorm)
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * This file implements the mathematical formulas for RMSNorm. For each row:
 * 1. Calculate the mean of the squared elements along the hidden dimension.
 * 2. Add an epsilon for numerical stability.
 * 3. Take the square root to find the Root Mean Square (RMS).
 * 4. Divide each element by the RMS and scale by the learnable weight vector.
 */

#include "RMSNorm.hpp"
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

/**
 * @brief Construct a new RMSNorm object
 * 
 * @param dims Dimension size to normalize over (typically the hidden dimension).
 * @param eps Small epsilon to prevent division by zero (defaults to 1e-5f).
 */
RMSNorm::RMSNorm(size_t dims, float eps) : weight_({dims}, 1.0f), eps_(eps) {}

/**
 * @brief Performs RMSNorm normalization on the input tensor in the forward pass.
 * 
 * Each row (representing the features of a single token) is normalized
 * independently along its last dimension.
 * 
 * Mathematical Formula:
 *   y = (x / RMS(x)) * weight
 *   where RMS(x) = sqrt(1/d * sum(x_i^2) + eps)
 * 
 * @param x Input tensor of shape (batch, seq_len, hidden_dim) or (seq_len, hidden_dim).
 * @return Tensor Normalized tensor of the same shape as input x.
 */
Tensor RMSNorm::forward(const Tensor &x) const {

  const std::vector<size_t> &x_shape = x.shape();
  if (x.shape().size() < 2) {

    throw std::invalid_argument("Input tensor must have at least 2 dimensions");
  }

  size_t dims = weight_.shape()[0];
  if (x_shape.back() != dims) {
    throw std::invalid_argument(
        "Input tensor last dimension must match RMSNorm dims");
  }

  Tensor result(x_shape);
  size_t total_elements = x.size();
  size_t num_rows = total_elements / dims;

  const float *x_data = x.data().data();
  const float *w_data = weight_.data().data();
  float *res_data = result.data().data();

  for (size_t r = 0; r < num_rows; r++) {

    size_t offset = r * dims;

    float sum_sq = std::inner_product(
        x_data + offset,        // Start of the row
        x_data + offset + dims, // End of the row
        x_data + offset,        // Start of the row (multiplied against itself)
        0.0f                    // Initial sum value
    );

    float rms = std::sqrt(sum_sq / static_cast<float>(dims) + eps_);

    for (size_t col_idx = 0; col_idx < dims; col_idx++) {
      res_data[offset + col_idx] =
          (x_data[offset + col_idx] / rms) * w_data[col_idx];
    }
  }

  return result;
}