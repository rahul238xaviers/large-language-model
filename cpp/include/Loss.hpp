#pragma once

#include "Tensor.hpp"

class CrossEntropyLoss {
public:
  // Forward pass: computes averate cross-entory loss over the batch.
  float forward(const Tensor &logits, const Tensor &targets);
  // Backward pass: computes gradient w.r.t logits
  Tensor backward(const Tensor &targets) const;
  // Direct access to GPU gradient buffer (for async dispatch — avoid copy)
  const Tensor &grad_logits() const { return grad_logits_; }
  float loss_val() const { return loss_val_; }

private:
  // Saved probabilities from the forward pass (CPU path)
  Tensor probs_;
  // Saved pre-computed gradients from the forward pass (GPU path)
  mutable Tensor grad_logits_;
  // Persistent float for GPU loss scalar copy-back
  mutable float loss_val_ = 0.0f;
};
