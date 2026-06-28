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
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h>

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
  const int n = static_cast<int>(x.size());
  float *ptr = x.data().data();

  // Step 1: compute neg_x = -x using vDSP_vneg
  std::vector<float> neg_x(n);
  vDSP_vneg(ptr, 1, neg_x.data(), 1, static_cast<vDSP_Length>(n));

  // Step 2: exp(-x) using vForce vectorized exp
  vvexpf(neg_x.data(), neg_x.data(), &n);

  // Step 3: denom = 1 + exp(-x)  —  vDSP_vsadd adds a scalar
  float one = 1.0f;
  vDSP_vsadd(neg_x.data(), 1, &one, neg_x.data(), 1,
             static_cast<vDSP_Length>(n));

  // Step 4: sigmoid = 1 / (1 + exp(-x))  —  element-wise reciprocal
  vvrecf(neg_x.data(), neg_x.data(), &n);

  // Step 5: x = x * sigmoid(x)  —  in-place element multiply
  vDSP_vmul(ptr, 1, neg_x.data(), 1, ptr, 1, static_cast<vDSP_Length>(n));
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
  const int n = static_cast<int>(gate.size());
  const float *g_ptr = gate.data().data();
  const float *u_ptr = up.data().data();
  const float *dy_ptr = grad_output.data().data();
  float *dg_ptr = grad_gate.data().data();
  float *du_ptr = grad_up.data().data();

  // Compute sig = sigmoid(gate) using vForce exp
  std::vector<float> neg_g(n), sig(n), silu_val(n), dsilu(n);
  vDSP_vneg(g_ptr, 1, neg_g.data(), 1, static_cast<vDSP_Length>(n));
  vvexpf(neg_g.data(), neg_g.data(), &n); // exp(-gate)
  float one = 1.0f;
  vDSP_vsadd(neg_g.data(), 1, &one, sig.data(), 1,
             static_cast<vDSP_Length>(n)); // 1 + exp(-gate)
  vvrecf(sig.data(), sig.data(), &n);      // sigmoid = 1 / (1 + exp(-gate))

  // silu_val = gate * sigmoid
  vDSP_vmul(g_ptr, 1, sig.data(), 1, silu_val.data(), 1,
            static_cast<vDSP_Length>(n));

  // dsilu = sigmoid * (1 + gate * (1 - sigmoid))
  for (int i = 0; i < n; ++i) {
    dsilu[i] = sig[i] * (1.0f + g_ptr[i] * (1.0f - sig[i]));
  }

  // grad_up = dy * silu_val
  vDSP_vmul(dy_ptr, 1, silu_val.data(), 1, du_ptr, 1,
            static_cast<vDSP_Length>(n));

  // grad_gate = dy * up * dsilu
  vDSP_vmul(dy_ptr, 1, u_ptr, 1, dg_ptr, 1,
            static_cast<vDSP_Length>(n)); // dy * up
  vDSP_vmul(dg_ptr, 1, dsilu.data(), 1, dg_ptr, 1,
            static_cast<vDSP_Length>(n)); // * dsilu
}
} // namespace activatations
