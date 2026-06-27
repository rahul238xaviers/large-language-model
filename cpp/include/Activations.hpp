/**
 * @file Activations.hpp
 * @brief Declaration of neural network activation functions (SiLU, SwiGLU)
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * Defines non-linear activation functions used within the Feed-Forward Network
 * (FFN) blocks of modern transformers. This includes the SiLU activation and
 * the SwiGLU gated activation mechanism.
 */

#pragma once
#include "Tensor.hpp"

namespace activatations {

void silu_(Tensor &x);
Tensor swiglu(const Tensor &gate, const Tensor &up);
void swiglu_backward(const Tensor &grad_output, const Tensor &gate,
                     const Tensor &up, Tensor &grad_gate, Tensor &grad_up);

} // namespace activatations