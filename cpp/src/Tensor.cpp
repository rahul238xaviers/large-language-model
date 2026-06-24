#include "Tensor.hpp"
#include <algorithm>
#include <functional>
#include <iostream>
#include <numeric>

Tensor::Tensor() : shape_({}), strides_({}), data_({}) {}

static size_t get_total_size(const std::vector<size_t> &shape) {
  if (shape.empty()) {
    return 1;
  }
  return std::accumulate(shape.begin(), shape.end(), 1ULL,
                         std::multiplies<size_t>());
}
Tensor::Tensor(const std::vector<size_t> &shape)
    : shape_(shape), data_(get_total_size(shape), 0.0f) {
  compute_strides();
}

Tensor::Tensor(const std::vector<size_t> &shape, float val)
    : shape_(shape), data_(get_total_size(shape), val) {
  compute_strides();
}

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

size_t Tensor::get_index(const std::vector<size_t> &indices) const {

  if (indices.size() != shape_.size()) {
    throw std::invalid_argument("Dimensionality mismatch");
  }
  size_t idx = 0;
  for (size_t i = 0; i < indices.size(); i++) {
    if (indices[i] >= shape_[i]) {
      throw std::invalid_argument("Index out of bounds");
    }

    idx += indices[i] * strides_[i];
  }
  return idx;
}

float &Tensor::operator()(size_t i) { return data_[i]; }

const float &Tensor::operator()(size_t i) const { return data_[i]; }

float &Tensor::operator()(size_t i, size_t j) {
  return data_[i * strides_[0] + j];
}

const float &Tensor::operator()(size_t i, size_t j) const {
  return data_[i * strides_[0] + j];
}

float &Tensor::operator()(size_t i, size_t j, size_t k) {
  return data_[i * strides_[0] + j * strides_[1] + k];
}

const float &Tensor::operator()(size_t i, size_t j, size_t k) const {
  return data_[i * strides_[0] + j * strides_[1] + k];
}

float &Tensor::operator()(size_t i, size_t j, size_t k, size_t l) {
  return data_[i * strides_[0] + j * strides_[1] + k * strides_[2] + l];
}

const float &Tensor::operator()(size_t i, size_t j, size_t k, size_t l) const {
  return data_[i * strides_[0] + j * strides_[1] + k * strides_[2] + l];
}

Tensor::Tensor(const std::vector<size_t> &shape, const std::vector<float> &data)
    : shape_(shape), data_(data) {
  compute_strides();
  if (data_.size() != get_total_size(shape_)) {
    throw std::invalid_argument("Data size does not match shape dimensions");
  }
}
void Tensor::fill(const float val) {
  std::fill(data_.begin(), data_.end(), val);
}
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

void Tensor::add_(const Tensor &other) {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("Shape mismatch for in-place addition");
  }
  std::transform(data_.begin(), data_.end(), other.data_.begin(), data_.begin(),
                 std::plus<float>());
}
void Tensor::mul_(const Tensor &other) {
  if (shape_ != other.shape_) {
    throw std::invalid_argument("Shape mismatch for in-place multiplication");
  }
  std::transform(data_.begin(), data_.end(), other.data_.begin(), data_.begin(),
                 std::multiplies<float>());
}

void Tensor::scale_(float factor) {
  std::transform(data_.begin(), data_.end(), data_.begin(),
                 [factor](float const &val) { return val * factor; });
}

Tensor Tensor::add(const Tensor &other) const {
  Tensor result = *this;
  result.add_(other);
  return result;
}

Tensor Tensor::mul(const Tensor &other) const {
  Tensor result = *this;
  result.mul_(other);
  return result;
}

Tensor Tensor::scale(float factor) const {
  Tensor result = *this;
  result.scale_(factor);
  return result;
}

Tensor Tensor::matmul(const Tensor &other) const {

  // Checking conditions for matrix multiplication
  // 1. Both tensors must have at least 2 dimensions
  // 2. Batch size must match for matrix multiplication
  // 3. Inner dimensions must match for matrix multiplication
  // Let the shape of the first tensor be (B1, M, K)
  // Let the shape of the second tensor be (B2, K, N)
  // For matrix multiplication to be valid, we must have B1 == B2 and K == K
  // The resulting tensor will have the shape (B1, M, N)

  if (shape_.size() < 2 || other.shape().size() < 2) {
    throw std::invalid_argument(
        "Matmul not implemented for tensors with less than 2 dimensions");
  }

  if (shape_.size() != other.shape().size()) {
    throw std::invalid_argument(
        "Batch size must match for matrix multiplication");
  }

  for (int i = 0; i < shape_.size() - 2; i++) {
    if (shape_[i] != other.shape()[i]) {
      throw std::invalid_argument(
          "Batch dimension must match for matrix multiplication");
    }
  }

  if (shape_[shape_.size() - 1] != other.shape()[other.shape().size() - 2]) {
    throw std::invalid_argument(
        "Inner dimensions must match for matrix multiplication");
  }

  // First we set the shape as the primary tension shape and then replace the
  // last element in the shape with the multiplier tensor For example,
  // A[3,4,5,6] and B[3,4,6,7] then result shape is initialised as [3,4,5,6]
  // then it is updated with other.shape().back() so it becomes [3,4,5,7]
  std::vector<size_t> result_shape = shape_;
  result_shape.back() = other.shape().back();
  Tensor result(result_shape);

  size_t num_batches = 1;

  // Except the last 2 dimension the multiplication of all the
  // other dimensions is considered as batch_size
  // This is because matmul is performed on the last 2 dimensions only
  for (int i = 0; i < shape_.size() - 2; i++) {
    num_batches *= shape_[i];
  }

  // Last but one dimension of the shape of the primary tensor
  // For example, if the tensor A is [3,4,5,6] then the M is 5.
  size_t M = shape_[shape_.size() - 2];

  // last dimension of the shape of the primary tensor
  // For example, if the tensor A is [3,4,5,6] then the K is 6.
  size_t K = shape_[shape_.size() - 1];

  // Last dimension of the shape of the multiplier tensor
  // For example, if the tensor B is [3,4,6,7] then the N is 7.
  size_t N = other.shape().back();

  // This represents in each batch dimension the size of the the matrix.
  // For example, A[3,4,5,6] then M*K = 5*6=30 is the size of matrix for A.
  // Similarly for B[3,4,6,7] then K*N=6*7=42 is the size of matrix for B
  // and result will be [3,4,5,7] so batch_offset_C = 5*7 =35
  size_t batch_offset_A = M * K;
  size_t batch_offset_B = K * N;
  size_t batch_offset_C = M * N;

  for (size_t b = 0; b < num_batches; ++b) {

    const float *dataA = data_.data() + b * batch_offset_A;
    const float *dataB = other.data().data() + b * batch_offset_B;
    float *dataC = result.data().data() + b * batch_offset_C;

    for (size_t i = 0; i < M; ++i) {
      for (size_t k = 0; k < K; ++k) {
        float val = dataA[i * K + k];
        for (size_t j = 0; j < N; ++j) {
          dataC[i * N + j] += val * dataB[k * N + j];
        }
      }
    }
  }
  return result;
}
