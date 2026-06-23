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
void Tensor::fill(float val) { std::fill(data_.begin(), data_.end(), val); }
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
