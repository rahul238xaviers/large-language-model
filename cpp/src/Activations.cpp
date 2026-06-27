/**
 * @file Activations.cpp
 * @brief Implementation of neural network activation functions (SiLU, SwiGLU)
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * This file implements the mathematical formulas for:
 * 1. SiLU (Sigmoid Linear Unit): f(x) = x * sigmoid(x).
 * 2. SwiGLU: swiglu(gate, up) = SiLU(gate) * up.
 */

#include "Activations.hpp"
#include "Tensor.hpp"
#include <algorithm>
#include <cmath>
#include <vector>

namespace activatations {

/**
 * @brief Applies the Sigmoid Linear Unit (SiLU) activation function in-place.
 * 
 * Formula: x_i = x_i * (1 / (1 + exp(-x_i)))
 * 
 * @param x Input tensor to be modified in-place.
 */
void silu_(Tensor &x) {
  std::transform(x.data().begin(), x.data().end(), x.data().begin(),
                 [](const float val) { return val * 1 / (1 + exp(-val)); });
}

/**
 * @brief Computes the SwiGLU (Swish Gated Linear Unit) activation.
 * 
 * Takes a gating tensor and an up-projection tensor, applies SiLU to the
 * gate tensor, and multiplies the result element-wise with the up-projection.
 * 
 * Formula: SwiGLU(gate, up) = SiLU(gate) * up
 * 
 * @param gate Gating projection tensor.
 * @param up Up-projection tensor (must have same shape as gate).
 * @return Tensor Result of SwiGLU activation.
 */
Tensor swiglu(const Tensor &gate, const Tensor &up) {

  if (gate.shape() != up.shape()) {
    throw std::invalid_argument("Dimension mismatch for SwiGLU activation");
  }
  Tensor gated = gate;
  silu_(gated);
  gated.mul_(up);
  return gated;
}
} // namespace activatations
