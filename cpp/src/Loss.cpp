/**
 * @file Loss.cpp
 * @brief Implementation of the Cross-Entropy Loss function and its analytical
 * gradients
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * Implements CrossEntropyLoss forward and backward pass operations:
 * 1. Computes the average token-wise cross-entropy loss over a batch.
 * 2. Uses the log-softmax subtraction trick for numerical stability (preventing
 * float overflow).
 * 3. Computes the analytical gradient of loss with respect to input logits:
 *    dLoss / dlogit_i = (1 / N) * (probs_i - indicator(i == target))
 */

#include "Loss.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>

/**
 * @brief Computes the forward pass of the Cross-Entropy Loss function.
 *
 * This function calculates:
 * 1. Softmax probabilities per token with subtraction stabilization.
 * 2. Log-probability of each target token.
 * 3. Averages the loss across all tokens in the batch.
 *
 * Preconditions:
 * 1. logits shape must be [batch, seq_len, vocab_size].
 * 2. targets shape must be [batch, seq_len].
 *
 * @param logits Logits tensor of shape [batch, seq_len, vocab_size].
 * @param targets Target token indices tensor of shape [batch, seq_len].
 * @return float Average loss scalar.
 */
float CrossEntropyLoss::forward(const Tensor &logits, const Tensor &targets) {
  size_t batch = logits.shape()[0];
  size_t seq_len = logits.shape()[1];
  size_t vocab_size = logits.shape()[2];

  // Initialize probs_ container to save activations for backward pass
  probs_ = Tensor({batch, seq_len, vocab_size}, 0.0f);

  float total_loss = 0.0f;

  for (size_t b = 0; b < batch; b++) {
    for (size_t s = 0; s < seq_len; s++) {

      // Step 1: Find the maximum logit value for numerical stability (softmax
      // subtraction trick)
      float max_val = -std::numeric_limits<float>::infinity();
      for (size_t v = 0; v < vocab_size; v++) {
        if (logits(b, s, v) > max_val) {
          max_val = logits(b, s, v);
        }
      }

      // Step 2: Compute the sum of exponentials (denominator of softmax)
      float sum_exp = 0.0f;
      for (size_t v = 0; v < vocab_size; v++) {
        sum_exp += std::exp(logits(b, s, v) - max_val);
      }

      // Step 3: Compute and store token-wise softmax probabilities
      for (size_t v = 0; v < vocab_size; v++) {
        probs_(b, s, v) = std::exp(logits(b, s, v) - max_val) / sum_exp;
      }

      // Step 4: Accumulate loss for the target token ID
      size_t target_id = static_cast<size_t>(targets(b, s));
      float target_prob = probs_(b, s, target_id);
      total_loss += -std::log(
          std::max(target_prob, 1e-15f)); // Floor probability to prevent log(0)
    }
  }
  return total_loss / (batch * seq_len);
}

/**
 * @brief Computes the backward pass (gradients of loss w.r.t input logits).
 *
 * Utilizes the saved probabilities from the forward pass:
 * dLoss / dlogit = (1 / N) * (probs - 1) for the target class
 * dLoss / dlogit = (1 / N) * probs for non-target classes
 *
 * @param targets Target token indices tensor of shape [batch, seq_len].
 * @return Tensor Gradients w.r.t logits of shape [batch, seq_len, vocab_size].
 */
Tensor CrossEntropyLoss::backward(const Tensor &targets) const {
  size_t batch = probs_.shape()[0];
  size_t seq_len = probs_.shape()[1];
  size_t vocab_size = probs_.shape()[2];

  Tensor grad_logits({batch, seq_len, vocab_size}, 0.0f);
  float scale = 1.0f / (batch * seq_len);

  for (size_t b = 0; b < batch; ++b) {
    for (size_t s = 0; s < seq_len; ++s) {
      size_t target_id = static_cast<size_t>(targets(b, s));
      for (size_t v = 0; v < vocab_size; ++v) {
        // Apply target-class indicator subtraction and batch normalization
        // scaling
        if (v == target_id) {
          grad_logits(b, s, v) = (probs_(b, s, v) - 1.0f) * scale;
        } else {
          grad_logits(b, s, v) = probs_(b, s, v) * scale;
        }
      }
    }
  }
  return grad_logits;
}
