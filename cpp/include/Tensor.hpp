/**
 * @file Tensor.hpp
 * @brief Multidimensional Tensor structure declaration
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * The Tensor class provides a general-purpose, multidimensional array representation
 * for floating-point values. It supports shape metadata, strided indexing, and basic
 * tensor operations (addition, multiplication, matrix multiplication).
 */

#pragma once

#include <cstddef>
#include <vector>

class Tensor {

public:
  // Default Constructor
  Tensor();
  // Shape Constructor
  explicit Tensor(const std::vector<size_t> &shape);
  // Fill Constructor
  Tensor(const std::vector<size_t> &shape, float val);
  // Data Constructor
  Tensor(const std::vector<size_t> &shape, const std::vector<float> &data);

  const std::vector<size_t> &shape() const { return shape_; }
  const std::vector<size_t> &strides() const { return strides_; }
  const std::vector<float> &data() const { return data_; }
  std::vector<float> &data() { return data_; }
  size_t size() const { return data_.size(); }

  // Indexing helper to convert multi-dimensional indices to flat 1D index
  size_t get_index(const std::vector<size_t> &indices) const;
  // Generic multi-dimensional indexing (both const and non-const)
  float &operator()(const std::vector<size_t> &indices);
  const float &operator()(const std::vector<size_t> &indices) const;
  // Fast 1D indexers
  float &operator()(size_t i);
  const float &operator()(size_t i) const;
  // Fast 2D indexers
  float &operator()(size_t i, size_t j);
  const float &operator()(size_t i, size_t j) const;
  // Fast 3D indexers
  float &operator()(size_t i, size_t j, size_t k);
  const float &operator()(size_t i, size_t j, size_t k) const;
  // Fast 4D indexers
  float &operator()(size_t i, size_t j, size_t k, size_t l);
  const float &operator()(size_t i, size_t j, size_t k, size_t l) const;

  // Utility methods
  void fill(float val);
  void print(const std::string &name = "") const;

  // Element-wise operations (Out-of-place: returns new tensor)
  Tensor add(const Tensor &other) const;
  Tensor mul(const Tensor &other) const;
  Tensor scale(float factor) const;

  // In-place operations (Modifies this tensor directly)
  void add_(const Tensor &other);
  void mul_(const Tensor &other);
  void scale_(float factor);

  // Matrix multiplication (2D or Batched 2D)
  Tensor matmul(const Tensor &other) const;

  // 2D Transpose (uses Accelerate vDSP_mtrans)
  Tensor transpose() const;

  // Reshape tensor to a new shape of matching size
  Tensor reshape(const std::vector<size_t> &new_shape) const;

private:
  std::vector<size_t> shape_;
  std::vector<size_t> strides_;
  std::vector<float> data_;
  void compute_strides();
};