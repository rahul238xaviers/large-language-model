#pragma once

#include "Tensor.hpp"
#include <cstddef>

class RMSNorm {
public:
  // Constructor: dims is the feature dimension, eps is a small float to prevent
  // div-by-zero
  RMSNorm(size_t dims, float eps = 1e-5f);

  // Forward pass: normalizes the input tensor along its last dimension
  Tensor forward(const Tensor &x) const;

  // Accessors
  const Tensor &weight() const { return weight_; }
  Tensor &weight() { return weight_; }
  float eps() const { return eps_; }

private:
  Tensor weight_;
  float eps_;
};
