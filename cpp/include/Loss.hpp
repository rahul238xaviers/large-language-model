#pragma once

#include "Tensor.hpp"

class CrossEntropyLoss {
public:
  // Forward pass: computes averate cross-entory loss over the batch.
  float forward(const Tensor &logits, const Tensor &targets);
  // Backward pass: computes gradient w.r.t logits
  Tensor backward(const Tensor &targets) const;

private:
  // Saved probabilities from the forward pass
  Tensor probs_;
};
