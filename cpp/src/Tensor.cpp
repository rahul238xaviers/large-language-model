/**
 * @file Tensor.cpp
 * @brief Implementation of the multidimensional Tensor structure
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * Implements index-flattening calculations (strides) and arithmetic
 * methods (add, mul, scale, matmul) for the Tensor class.
 */

#include "Tensor.hpp"
#define ACCELERATE_NEW_LAPACK
#include "gpu_kernel/MetalBridge.hpp"
#include <Accelerate/Accelerate.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <numeric>
#include <stdexcept>

/**
 * @brief Construct a new empty Tensor object.
 */
Tensor::Tensor() : shape_({}), strides_({}), data_({}) {}

/**
 * @brief Helper to calculate the total flat size of a given shape.
 */
static size_t get_total_size(const std::vector<size_t> &shape) {
  if (shape.empty()) {
    return 1;
  }
  return std::accumulate(shape.begin(), shape.end(), 1ULL,
                         std::multiplies<size_t>());
}

/**
 * @brief Construct a new Tensor object with shape, zero-initialized.
 *
 * @param shape Vector containing dimension sizes.
 */
Tensor::Tensor(const std::vector<size_t> &shape)
    : shape_(shape), data_(get_total_size(shape), 0.0f) {
  compute_strides();
}

/**
 * @brief Construct a new Tensor object with shape, initialized to a scalar
 * value.
 *
 * @param shape Vector containing dimension sizes.
 * @param val Scalar value to fill the tensor with.
 */
Tensor::Tensor(const std::vector<size_t> &shape, float val)
    : shape_(shape), data_(get_total_size(shape), val) {
  compute_strides();
}

/**
 * @brief Precomputes the row-major index offset multiplier (stride) for each
 * dimension.
 */
void Tensor::compute_strides() {
  strides_.resize(shape_.size(), 1);
  if (shape_.empty())
    return;
  size_t stride = 1;
  for (int i = static_cast<int>(shape_.size()) - 1; i >= 0; --i) {
    strides_[i] = stride;
    stride *= shape_[i];
  }
}

/**
 * @brief Flattens multi-dimensional indices into a single row-major 1D index
 * offset.
 *
 * @param indices Vector of indices per dimension.
 * @return size_t Flat index offset.
 */
size_t Tensor::get_index(const std::vector<size_t> &indices) const {
  if (indices.size() != shape_.size()) {
    throw std::invalid_argument("Dimensionality mismatch");
  }
  size_t idx = 0;
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i] >= shape_[i]) {
      throw std::out_of_range("Index out of bounds");
    }
    idx += indices[i] * strides_[i];
  }
  return idx;
}

/**
 * @brief Access element at multi-dimensional coordinate vector (mutable).
 */
float &Tensor::operator()(const std::vector<size_t> &indices) {
  return data_[get_index(indices)];
}

/**
 * @brief Access element at multi-dimensional coordinate vector (read-only).
 */
const float &Tensor::operator()(const std::vector<size_t> &indices) const {
  return data_[get_index(indices)];
}

/**
 * @brief Access element at flat index (mutable).
 */
float &Tensor::operator()(size_t i) { return data_[i]; }

/**
 * @brief Access element at flat index (read-only).
 */
const float &Tensor::operator()(size_t i) const { return data_[i]; }

/**
 * @brief Access element at 2D coordinates (mutable).
 */
float &Tensor::operator()(size_t i, size_t j) {
  return data_[i * strides_[0] + j];
}

/**
 * @brief Access element at 2D coordinates (read-only).
 */
const float &Tensor::operator()(size_t i, size_t j) const {
  return data_[i * strides_[0] + j];
}

/**
 * @brief Access element at 3D coordinates (mutable).
 */
float &Tensor::operator()(size_t i, size_t j, size_t k) {
  return data_[i * strides_[0] + j * strides_[1] + k];
}

/**
 * @brief Access element at 3D coordinates (read-only).
 */
const float &Tensor::operator()(size_t i, size_t j, size_t k) const {
  return data_[i * strides_[0] + j * strides_[1] + k];
}

/**
 * @brief Access element at 4D coordinates (mutable).
 */
float &Tensor::operator()(size_t i, size_t j, size_t k, size_t l) {
  return data_[i * strides_[0] + j * strides_[1] + k * strides_[2] + l];
}

/**
 * @brief Access element at 4D coordinates (read-only).
 */
const float &Tensor::operator()(size_t i, size_t j, size_t k, size_t l) const {
  return data_[i * strides_[0] + j * strides_[1] + k * strides_[2] + l];
}

/**
 * @brief Construct a new Tensor object from an existing flat data vector.
 *
 * @param shape Shape of the new tensor.
 * @param data Flat vector of data values.
 */
Tensor::Tensor(const std::vector<size_t> &shape, const std::vector<float> &data)
    : shape_(shape), data_(data) {
  compute_strides();
  if (data_.size() != get_total_size(shape_)) {
    throw std::invalid_argument("Data size does not match shape dimensions");
  }
}

/**
 * @brief Fills the tensor elements with a single scalar value.
 *
 * @param val Scalar value.
 */
void Tensor::fill(const float val) {
  std::fill(data_.begin(), data_.end(), val);
}

/**
 * @brief Prints the shape and metadata values of the tensor to stdout.
 *
 * @param name Optional label to print before the shape.
 */
void Tensor::print(const std::string &name) const {
  if (!name.empty()) {
    std::cout << name << " ";
  }
  std::cout << "Tensor shape: [";
  for (size_t i = 0; i < shape_.size(); ++i) {
    std::cout << shape_[i] << (i == shape_.size() - 1 ? "" : ", ");
  }
  std::cout << "], size: " << data_.size() << "\n";

  if (data_.empty()) {
    std::cout << "  []\n";
    return;
  }

  // Print first few elements (up to 20)
  std::cout << "  ";
  for (size_t i = 0; i < std::min(data_.size(), size_t(20)); ++i) {
    std::cout << data_[i] << " ";
  }
  if (data_.size() > 20) {
    std::cout << "...";
  }
  std::cout << "\n";
}

/**
 * @brief Adds another tensor to this tensor in-place.
 *
 * Uses Apple vDSP for vectorized SIMD element-wise addition.
 *
 * @param other Tensor to add (must have matching shape).
 */
void Tensor::add_(const Tensor &other) {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("Shape mismatch for in-place addition");
  }
  // vDSP_vadd: NEON-vectorized element-wise addition
  vDSP_vadd(data_.data(), 1, other.data_.data(), 1, data_.data(), 1,
            static_cast<vDSP_Length>(data_.size()));
}

/**
 * @brief Multiplies another tensor with this tensor element-wise in-place.
 *
 * Uses Apple vDSP for vectorized SIMD element-wise multiplication.
 *
 * @param other Tensor to multiply (must have matching shape).
 */
void Tensor::mul_(const Tensor &other) {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("Shape mismatch for in-place multiplication");
  }
  // vDSP_vmul: NEON-vectorized element-wise multiply
  vDSP_vmul(data_.data(), 1, other.data_.data(), 1, data_.data(), 1,
            static_cast<vDSP_Length>(data_.size()));
}

/**
 * @brief Scales this tensor's elements by a scalar factor in-place.
 *
 * Uses Apple vDSP for vectorized SIMD scalar multiplication.
 *
 * @param factor Scalar scaling factor.
 */
void Tensor::scale_(float factor) {
  // vDSP_vsmul: NEON-vectorized scalar multiply
  vDSP_vsmul(data_.data(), 1, &factor, data_.data(), 1,
             static_cast<vDSP_Length>(data_.size()));
}

/**
 * @brief Returns the element-wise sum of this tensor and another tensor.
 *
 * @param other Tensor to add.
 * @return Tensor New sum tensor.
 */
Tensor Tensor::add(const Tensor &other) const {
  Tensor result = *this;
  result.add_(other);
  return result;
}

/**
 * @brief Returns the element-wise product of this tensor and another tensor.
 *
 * @param other Tensor to multiply.
 * @return Tensor New product tensor.
 */
Tensor Tensor::mul(const Tensor &other) const {
  Tensor result = *this;
  result.mul_(other);
  return result;
}

/**
 * @brief Returns a scaled copy of this tensor.
 *
 * @param factor Scalar scaling factor.
 * @return Tensor New scaled tensor.
 */
Tensor Tensor::scale(float factor) const {
  Tensor result = *this;
  result.scale_(factor);
  return result;
}

/**
 * @brief Performs batched or 2D matrix multiplication between this tensor and
 * another.
 *
 * Preconditions:
 * 1. Both tensors must have at least 2 dimensions.
 * 2. Non-matrix dimensions (batch dimensions) must match exactly.
 * 3. Inner matrix dimensions must match: this->shape().back() ==
 * other.shape()[second_to_last].
 *
 * @param other Multiplier tensor.
 * @return Tensor Result tensor.
 */
// Tensor::matmul — reviewed and fixed.
//
// Bugs fixed vs. the original:
//   1. The "broadcast 2D weight across batch dims" fast path was dead code:
//      it required shape_.size()==3 && other.shape().size()==2, but the
//      strict rank-equality check just above it threw first, so that branch
//      could never be reached. It is now a real, reachable code path, and
//      generalized to work for ANY input rank (not just rank 3), since the
//      broadcasting logic doesn't actually depend on rank.
//   2. Inconsistent zero-initialization: one branch explicitly zero-filled
//      the result tensor, the other didn't. Both paths now explicitly
//      zero-initialize, since we accumulate with +=.
//   3. Signed/unsigned loop counters (`int i` compared against
//      `shape_.size() - 2`, a size_t) replaced with size_t throughout, and
//      the loop bound rewritten as `i + 2 < rank` to avoid any risk of
//      unsigned underflow if this code is ever reused with a smaller rank.
//
// Behavior preserved: same cache-friendly i-k-j loop order, same exception
// messages/conditions for the still-supported error cases.

Tensor Tensor::matmul(const Tensor &other) const {
  // Matrix multiplication rules:
  // 1. Both tensors must have at least 2 dimensions.
  // 2. If `other` is exactly 2D, it's treated as a shared weight/projection
  //    matrix and broadcast across every leading ("batch") dimension of
  //    `this` — e.g. (B, S, K) @ (K, N) -> (B, S, N), or even
  //    (B, S1, S2, K) @ (K, N) -> (B, S1, S2, N).
  // 3. Otherwise, both tensors must have the same rank, matching leading
  //    (batch) dimensions, and a matching inner dimension:
  //    (..., M, K) @ (..., K, N) -> (..., M, N)

  if (shape_.size() < 2 || other.shape().size() < 2) {
    throw std::invalid_argument(
        "Matmul not implemented for tensors with less than 2 dimensions");
  }

  // --- Case 1: broadcast a 2D weight matrix across all leading dimensions ---
  // e.g. x:(B, S, K) @ W:(K, N) -> (B, S, N)
  if (other.shape().size() == 2 && shape_.size() != 2) {
    const size_t K = shape_.back();
    const size_t N = other.shape()[1];

    if (K != other.shape()[0]) {
      throw std::invalid_argument("Inner dimensions must match for projection");
    }

    // Flatten every dimension except the last into a single "row" count.
    // e.g. (B, S, K) -> M = B * S rows of length K.
    size_t M = 1;
    for (size_t i = 0; i + 1 < shape_.size(); ++i) {
      M *= shape_[i];
    }

    std::vector<size_t> result_shape = shape_;
    result_shape.back() = N;
    Tensor result(result_shape, 0.0f);

    const float *x_data = data_.data();
    const float *w_data = other.data().data();
    float *res_data = result.data().data();

    const char *gpu_enabled_env = std::getenv("GPU_ENABLED");
    bool use_gpu = false;
    if (gpu_enabled_env && std::string(gpu_enabled_env) == "1") {
      metal_bridge::initialize();
      if (metal_bridge::is_available() && (M % 8 == 0) && (N % 8 == 0)) {
        use_gpu = true;
      }
    }

    if (use_gpu) {
      auto start = std::chrono::high_resolution_clock::now();
      if (K == N) {
        metal_bridge::gemm_proj(x_data, w_data, res_data, M, N, K);
      } else {
        metal_bridge::gemm_ffn(x_data, w_data, res_data, M, N, K);
      }
      auto end = std::chrono::high_resolution_clock::now();
      metal_bridge::accum_gpu_time_ms +=
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count() /
          1000.0;
      metal_bridge::count_gpu_calls++;
    } else {
      auto start = std::chrono::high_resolution_clock::now();
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  static_cast<int>(M), static_cast<int>(N), static_cast<int>(K),
                  1.0f, x_data, static_cast<int>(K), w_data,
                  static_cast<int>(N), 0.0f, res_data, static_cast<int>(N));
      auto end = std::chrono::high_resolution_clock::now();
      metal_bridge::accum_cpu_time_ms +=
          std::chrono::duration_cast<std::chrono::microseconds>(end - start)
              .count() /
          1000.0;
      metal_bridge::count_cpu_calls++;
    }

    return result;
  }

  // --- Case 2: standard batched matmul, both tensors share the same rank ---
  if (shape_.size() != other.shape().size()) {
    throw std::invalid_argument(
        "Batch size must match for matrix multiplication");
  }

  const size_t rank = shape_.size();

  for (size_t i = 0; i + 2 < rank; ++i) {
    if (shape_[i] != other.shape()[i]) {
      throw std::invalid_argument(
          "Batch dimension must match for matrix multiplication");
    }
  }

  if (shape_[rank - 1] != other.shape()[rank - 2]) {
    throw std::invalid_argument(
        "Inner dimensions must match for matrix multiplication");
  }

  // Result takes the shape of `this`, but with the trailing dimension
  // replaced by the trailing dimension of `other`.
  // e.g. A[3,4,5,6] @ B[3,4,6,7] -> result[3,4,5,7]
  std::vector<size_t> result_shape = shape_;
  result_shape.back() = other.shape().back();
  Tensor result(result_shape, 0.0f);

  // Every dimension except the trailing two is a batch dimension, since
  // matmul only operates on the last two dimensions of each slice.
  size_t num_batches = 1;
  for (size_t i = 0; i + 2 < rank; ++i) {
    num_batches *= shape_[i];
  }

  const size_t M = shape_[rank - 2];     // rows of A / rows of result
  const size_t K = shape_[rank - 1];     // cols of A / rows of B
  const size_t N = other.shape().back(); // cols of B / cols of result

  const size_t batch_offset_A = M * K;
  const size_t batch_offset_B = K * N;
  const size_t batch_offset_C = M * N;

  const float *a_data = data_.data();
  const float *b_data = other.data().data();
  float *c_data = result.data().data();

  for (size_t b = 0; b < num_batches; ++b) {
    const float *dataA = a_data + b * batch_offset_A;
    const float *dataB = b_data + b * batch_offset_B;
    float *dataC = c_data + b * batch_offset_C;

    // Use Apple Accelerate BLAS for SIMD-vectorized matrix multiplication
    auto start = std::chrono::high_resolution_clock::now();
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, static_cast<int>(M),
                static_cast<int>(N), static_cast<int>(K), 1.0f, dataA,
                static_cast<int>(K), dataB, static_cast<int>(N), 0.0f, dataC,
                static_cast<int>(N));
    auto end = std::chrono::high_resolution_clock::now();
    metal_bridge::accum_cpu_time_ms +=
        std::chrono::duration_cast<std::chrono::microseconds>(end - start)
            .count() /
        1000.0;
    metal_bridge::count_cpu_calls++;
  }

  return result;
}
