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
#include "gpu_kernel/MetalBridge.hpp"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <string>
#include <thread>
#include <vector>

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
  size_t total_tokens = batch * seq_len;

  const char *gpu_enabled_env = std::getenv("GPU_ENABLED");
  if (gpu_enabled_env && std::string(gpu_enabled_env) == "1") {
    metal_bridge::initialize();
    if (metal_bridge::is_available()) {
      std::vector<uint32_t> targets_uint32(total_tokens);
      for (size_t i = 0; i < total_tokens; ++i) {
        targets_uint32[i] = static_cast<uint32_t>(targets.data()[i]);
      }

      loss_val_ = 0.0f;
      grad_logits_ = Tensor({batch, seq_len, vocab_size}, 0.0f);

      metal_bridge::cross_entropy(
          logits.data(),
          targets_uint32.data(),
          &loss_val_,
          grad_logits_.data(),
          total_tokens,
          vocab_size
      );

      return loss_val_;
    }
  }

  // Initialize probs_ container to save activations for backward pass
  probs_ = Tensor({batch, seq_len, vocab_size}, 0.0f);

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 4;

  std::vector<std::thread> workers;
  std::vector<float> thread_loss(num_threads, 0.0f);
  size_t tokens_per_thread = (total_tokens + num_threads - 1) / num_threads;

  for (unsigned int t = 0; t < num_threads; ++t) {
    size_t start_tok = t * tokens_per_thread;
    size_t end_tok = std::min(start_tok + tokens_per_thread, total_tokens);

    if (start_tok >= end_tok) continue;

    workers.emplace_back([this, start_tok, end_tok, vocab_size, seq_len, &logits, &targets, &thread_loss, t]() {
      float local_loss = 0.0f;
      for (size_t tok = start_tok; tok < end_tok; ++tok) {
        size_t b = tok / seq_len;
        size_t s = tok % seq_len;

        // Step 1: Find the maximum logit value for numerical stability (softmax subtraction trick)
        float max_val = -std::numeric_limits<float>::infinity();
        for (size_t v = 0; v < vocab_size; v++) {
          float val = logits(b, s, v);
          if (val > max_val) {
            max_val = val;
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
        local_loss += -std::log(std::max(target_prob, 1e-15f));
      }
      thread_loss[t] = local_loss;
    });
  }

  for (auto &worker : workers) {
    worker.join();
  }

  float total_loss = 0.0f;
  for (float loss : thread_loss) {
    total_loss += loss;
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
  const char *gpu_enabled_env = std::getenv("GPU_ENABLED");
  if (gpu_enabled_env && std::string(gpu_enabled_env) == "1") {
    if (metal_bridge::is_available() && grad_logits_.size() > 0) {
      return grad_logits_;
    }
  }

  size_t batch = probs_.shape()[0];
  size_t seq_len = probs_.shape()[1];
  size_t vocab_size = probs_.shape()[2];

  Tensor grad_logits({batch, seq_len, vocab_size}, 0.0f);
  float scale = 1.0f / (batch * seq_len);

  unsigned int num_threads = std::thread::hardware_concurrency();
  if (num_threads == 0) num_threads = 4;

  std::vector<std::thread> workers;
  size_t total_tokens = batch * seq_len;
  size_t tokens_per_thread = (total_tokens + num_threads - 1) / num_threads;

  for (unsigned int t = 0; t < num_threads; ++t) {
    size_t start_tok = t * tokens_per_thread;
    size_t end_tok = std::min(start_tok + tokens_per_thread, total_tokens);

    if (start_tok >= end_tok) continue;

    workers.emplace_back([this, start_tok, end_tok, vocab_size, seq_len, scale, &targets, &grad_logits]() {
      for (size_t tok = start_tok; tok < end_tok; ++tok) {
        size_t b = tok / seq_len;
        size_t s = tok % seq_len;

        size_t target_id = static_cast<size_t>(targets(b, s));
        for (size_t v = 0; v < vocab_size; ++v) {
          if (v == target_id) {
            grad_logits(b, s, v) = (probs_(b, s, v) - 1.0f) * scale;
          } else {
            grad_logits(b, s, v) = probs_(b, s, v) * scale;
          }
        }
      }
    });
  }

  for (auto &worker : workers) {
    worker.join();
  }

  return grad_logits;
}
