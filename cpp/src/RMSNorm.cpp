#include "RMSNorm.hpp"
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>

RMSNorm::RMSNorm(size_t dims, float eps) : weight_({dims}, 1.0f), eps_(eps) {}

// This function squares each row of the Tensor sum it up. Then add the epsilon
// Then takes the square root of it. Once the square root is calculated then
// that value is used to divide the input tensor. Finally the result is
// multiplied with the weight vector.

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