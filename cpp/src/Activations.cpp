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

/**
 * @brief Computes the backward pass (gradients) for the SwiGLU activation.
 *
 * Calculates gradients with respect to the gate and up-projection inputs based
 * on the chain rule, given the gradient flowing from the output.
 *
 * @param grad_output Incoming gradient from the subsequent layer.
 * @param gate The gate tensor used in the forward pass.
 * @param up The up-projection tensor used in the forward pass.
 * @param grad_gate Output tensor to store gradients w.r.t. the gate.
 * @param grad_up Output tensor to store gradients w.r.t. the up-projection.
 */
void swiglu_backward(const Tensor &grad_output, const Tensor &gate,
                     const Tensor &up, Tensor &grad_gate, Tensor &grad_up) {

  if (grad_output.shape() != gate.shape() ||
      grad_output.shape() != up.shape() ||
      grad_output.shape() != grad_gate.shape() ||
      grad_output.shape() != grad_up.shape()) {
    throw std::invalid_argument("Dimension mismatch for SwiGLU backward pass");
  }
  size_t total_elements = gate.size();

  for (size_t i = 0; i < total_elements; i++) {
    float g = gate.data()[i];
    float u = up.data()[i];
    float dy = grad_output.data()[i];

    float sig = 1.0f / (1.0f + std::exp(-g));
    float silu_val = g * sig;

    float dsilu = sig * (1.0f + g * (1.0f - sig));

    grad_up.data()[i] = dy * silu_val;
    grad_gate.data()[i] = dy * u * dsilu;
  }
}
} // namespace activatations
