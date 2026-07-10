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
#include "gpu_kernel/MetalBridge.hpp"
#include <cmath>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <vector>
#include <thread>
#include <cstdlib>

/**
 * @brief Construct a new RMSNorm object
 *
 * @param dims Dimension size to normalize over (typically the hidden
 * dimension).
 * @param eps Small epsilon to prevent division by zero (defaults to 1e-5f).
 */
RMSNorm::RMSNorm(size_t dims, float eps) : weight_({dims}, 1.0f), eps_(eps) {}

/**
 * @brief Performs RMSNorm normalization on the input tensor in the forward
 * pass.
 *
 * Each row (representing the features of a single token) is normalized
 * independently along its last dimension.
 *
 * Mathematical Formula:
 *   y = (x / RMS(x)) * weight
 *   where RMS(x) = sqrt(1/d * sum(x_i^2) + eps)
 *
 * @param x Input tensor of shape (batch, seq_len, hidden_dim) or (seq_len,
 * hidden_dim).
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

  const char *gpu_enabled_env = std::getenv("GPU_ENABLED");
  bool use_gpu = false;
  if (gpu_enabled_env && std::string(gpu_enabled_env) == "1") {
    metal_bridge::initialize();
    if (metal_bridge::is_available()) {
      use_gpu = true;
    }
  }

  if (use_gpu) {
    metal_bridge::rms_norm_forward(x.data().data(), result.data().data(), weight_.data().data(),
                                   eps_, num_rows, dims);
    return result;
  }

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 4;

  std::vector<std::thread> workers;
  size_t rows_per_thread = (num_rows + num_threads - 1) / num_threads;

  for (unsigned int t = 0; t < num_threads; ++t) {
    size_t start_row = t * rows_per_thread;
    size_t end_row = std::min(start_row + rows_per_thread, num_rows);

    if (start_row >= end_row) continue;

    workers.emplace_back([this, start_row, end_row, dims, &x, &result]() {
      const float *x_data = x.data().data();
      const float *w_data = weight_.data().data();
      float *res_data = result.data().data();

      for (size_t r = start_row; r < end_row; ++r) {
        size_t offset = r * dims;

        float sum_sq = std::inner_product(
            x_data + offset,
            x_data + offset + dims,
            x_data + offset,
            0.0f
        );

        float rms = std::sqrt(sum_sq / static_cast<float>(dims) + eps_);

        for (size_t col_idx = 0; col_idx < dims; col_idx++) {
          res_data[offset + col_idx] =
              (x_data[offset + col_idx] / rms) * w_data[col_idx];
        }
      }
    });
  }

  for (auto &worker : workers) {
    worker.join();
  }

  return result;
}

Tensor RMSNorm::backward(const Tensor &grad_output, const Tensor &input,
                         Tensor &grad_weight) const {
  if (grad_output.shape() != input.shape()) {
    throw std::invalid_argument(
        "grad_output and input shapes must match in RMSNorm::backward");
  }

  size_t dims = weight_.shape()[0];
  if (grad_weight.shape() != std::vector<size_t>{dims}) {
    throw std::invalid_argument(
        "grad_weight must match RMSNorm weight dimensions");
  }

  Tensor grad_input(input.shape(), 0.0f);
  size_t total_elements = input.size();
  size_t num_rows = total_elements / dims;

  const char *gpu_enabled_env = std::getenv("GPU_ENABLED");
  bool use_gpu = false;
  if (gpu_enabled_env && std::string(gpu_enabled_env) == "1") {
    metal_bridge::initialize();
    if (metal_bridge::is_available()) {
      use_gpu = true;
    }
  }

  if (use_gpu) {
    metal_bridge::rms_norm_backward(grad_output.data().data(), input.data().data(), weight_.data().data(),
                                    grad_input.data().data(), grad_weight.data().data(), eps_,
                                    num_rows, dims);
    return grad_input;
  }

  float *dw_data = grad_weight.data().data();

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 4;

  std::vector<std::thread> workers;
  std::vector<std::vector<float>> thread_dw(num_threads, std::vector<float>(dims, 0.0f));
  size_t rows_per_thread = (num_rows + num_threads - 1) / num_threads;

  for (unsigned int t = 0; t < num_threads; ++t) {
    size_t start_row = t * rows_per_thread;
    size_t end_row = std::min(start_row + rows_per_thread, num_rows);

    if (start_row >= end_row) continue;

    workers.emplace_back([this, start_row, end_row, dims, &input, &grad_output, &grad_input, &thread_dw, t]() {
      const float *x_data = input.data().data();
      const float *g_data = grad_output.data().data();
      const float *w_data = weight_.data().data();
      float *dx_data = grad_input.data().data();
      float *dw_local = thread_dw[t].data();

      for (size_t r = start_row; r < end_row; ++r) {
        size_t offset = r * dims;

        // 1. Calculate sum of squares for this row to compute RMS
        float sum_sq = 0.0f;
        for (size_t col = 0; col < dims; ++col) {
          float val = x_data[offset + col];
          sum_sq += val * val;
        }
        float rms = std::sqrt(sum_sq / static_cast<float>(dims) + eps_);

        // 2. Compute inner sum_term: mean(g * w * xhat)
        float sum_g_w_xhat = 0.0f;
        for (size_t col = 0; col < dims; ++col) {
          float xhat = x_data[offset + col] / rms;
          sum_g_w_xhat += g_data[offset + col] * w_data[col] * xhat;
        }
        sum_g_w_xhat /= static_cast<float>(dims);

        // 3. Compute gradients w.r.t input and accumulate weights gradients locally
        for (size_t col = 0; col < dims; ++col) {
          float xhat = x_data[offset + col] / rms;

          // grad_input calculation
          dx_data[offset + col] =
              (1.0f / rms) *
              (g_data[offset + col] * w_data[col] - xhat * sum_g_w_xhat);

          // Accumulate weight gradient locally
          dw_local[col] += g_data[offset + col] * xhat;
        }
      }
    });
  }

  for (auto &worker : workers) {
    worker.join();
  }

  // Sum up thread-local gradients into global grad_weight
  for (unsigned int t = 0; t < num_threads; ++t) {
    for (size_t col = 0; col < dims; ++col) {
      dw_data[col] += thread_dw[t][col];
    }
  }

  return grad_input;
}
