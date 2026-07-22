#pragma once

#include "PageAlignedAllocator.hpp"
#include <cstddef>
#include <cstdint>
#include <vector>

enum class DType : uint8_t { FP32, BF16 };

template <typename T>
using AlignedVector = std::vector<T, PageAlignedAllocator<T>>;

class Tensor {
public:
  Tensor();
  explicit Tensor(const std::vector<size_t> &shape, DType dtype = DType::FP32);
  Tensor(const std::vector<size_t> &shape, float val, DType dtype = DType::FP32);
  Tensor(const std::vector<size_t> &shape, const AlignedVector<float> &data, DType dtype = DType::FP32);
  Tensor(const std::vector<size_t> &shape, const std::vector<float> &data, DType dtype = DType::FP32);

  DType dtype() const { return dtype_; }
  const std::vector<size_t> &shape() const { return shape_; }
  const std::vector<size_t> &strides() const { return strides_; }
  size_t size() const { return shape_.empty() ? 0 : (dtype_ == DType::BF16 ? bf16_data_.size() : data_.size()); }

  // FP32 access (optimizer m/v, CPU fallback)
  const AlignedVector<float> &data() const { return data_; }
  AlignedVector<float> &data() { return data_; }

  // BF16 access (model weights, activations, gradients)
  const AlignedVector<__bf16> &bf16() const { return bf16_data_; }
  AlignedVector<__bf16> &bf16() { return bf16_data_; }

  // Raw pointer for GPU dispatch (dtype-aware, page-aligned for zero-copy)
  const void *raw_ptr() const { return dtype_ == DType::BF16 ? (const void*)bf16_data_.data() : (const void*)data_.data(); }
  void *raw_ptr() { return dtype_ == DType::BF16 ? (void*)bf16_data_.data() : (void*)data_.data(); }
  size_t raw_bytes() const { return dtype_ == DType::BF16 ? bf16_data_.size() * sizeof(__bf16) : data_.size() * sizeof(float); }
  size_t num_elements() const { return dtype_ == DType::BF16 ? bf16_data_.size() : data_.size(); }

  size_t get_index(const std::vector<size_t> &indices) const;

  float &operator()(const std::vector<size_t> &indices);
  const float &operator()(const std::vector<size_t> &indices) const;
  float &operator()(size_t i);
  const float &operator()(size_t i) const;
  float &operator()(size_t i, size_t j);
  const float &operator()(size_t i, size_t j) const;
  float &operator()(size_t i, size_t j, size_t k);
  const float &operator()(size_t i, size_t j, size_t k) const;
  float &operator()(size_t i, size_t j, size_t k, size_t l);
  const float &operator()(size_t i, size_t j, size_t k, size_t l) const;

  void fill(float val);
  void print(const std::string &name = "") const;

  Tensor add(const Tensor &other) const;
  Tensor mul(const Tensor &other) const;
  Tensor scale(float factor) const;
  void add_(const Tensor &other);
  void mul_(const Tensor &other);
  void scale_(float factor);

  Tensor matmul(const Tensor &other) const;
  Tensor transpose() const;
  Tensor reshape(const std::vector<size_t> &new_shape) const;

private:
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;
  AlignedVector<float> data_;        // FP32 storage (optimizer m/v) — page-aligned
  AlignedVector<__bf16> bf16_data_;  // BF16 storage (weights, activations) — page-aligned
  DType dtype_ = DType::FP32;
  void compute_strides();
};
