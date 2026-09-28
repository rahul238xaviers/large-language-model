/**
 * @file Optimizer.cpp
 * @brief Implementation of neural network parameters optimizers (SGD, AdamW)
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * This file implements parameter optimization logic:
 * 1. SGD (Stochastic Gradient Descent) updates weights using simple scaled gradients.
 * 2. AdamW (Adam with Weight Decay) decouples L2 regularization from gradient momentum
 *    and scales parameter updates using running first and second moments.
 */

#include "Optimizer.hpp"
#include "gpu_kernel/MetalBridge.hpp"
#include <stdexcept>
#include<iostream>
#include <vector>
#include <thread>
#include <cmath>

/**
 * @brief Registers a parameter tensor and its corresponding gradient tensor for optimization updates.
 *
 * @param param Pointer to the parameter tensor to be updated.
 * @param grad Pointer to the gradient tensor containing gradients w.r.t param.
 */
void Optimizer::register_parameter(Tensor *param, const Tensor *grad) {
  if (!param || !grad) {
    throw std::invalid_argument("Parameter and gradient pointers must not be null");
  }
  if (param->shape() != grad->shape()) {
    throw std::invalid_argument("Parameter and gradient shapes must match");
  }
  params_.push_back(param);
  grads_.push_back(grad);
}

/**
 * @brief Clears all registered parameters and gradient tracking references.
 */
void Optimizer::clear() {
  params_.clear();
  grads_.clear();
}

/**
 * @brief Performs a single optimization step using Stochastic Gradient Descent (SGD) algorithm.
 *
 * Updates all registered parameters using the formula: weight = weight - lr * gradient.
 *
 * @param lr Learning rate.
 */
void SGDOptimizer::step(float lr) {
  for (size_t i = 0; i < params_.size(); ++i) {
    Tensor &param = *params_[i];
    const Tensor &grad = *grads_[i];
    float *p_data = param.data();
    const float *g_data = grad.data();
    size_t n = param.size();
    for (size_t j = 0; j < n; ++j) {
      p_data[j] -= lr * g_data[j];
    }
  }
}

/**
 * @brief Construct a new AdamWOptimizer object.
 *
 * Sets hyperparameters for momentum decay, squared gradient decay, and weight decay.
 *
 * @param beta1 First moment decay hyperparameter (default 0.9).
 * @param beta2 Second moment decay hyperparameter (default 0.999).
 * @param eps Small constant to prevent division by zero (default 1e-8).
 * @param weight_decay Decoupled weight decay parameter (default 0.01).
 */
AdamWOptimizer::AdamWOptimizer(float beta1, float beta2, float eps, float weight_decay)
    : beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay), step_count_(0) {}

/**
 * @brief Registers a parameter tensor and initializes first and second moment tracking buffers.
 *
 * @param param Pointer to the parameter tensor.
 * @param grad Pointer to the gradient tensor.
 */
void AdamWOptimizer::register_parameter(Tensor *param, const Tensor *grad) {
  Optimizer::register_parameter(param, grad);
  m_states_.push_back(Tensor(param->shape(), 0.0f));
  v_states_.push_back(Tensor(param->shape(), 0.0f));
}

/**
 * @brief Helper to perform AdamW parameter update for a single tensor block.
 *
 * Keeps loops under 20 lines and maintains cache locality.
 */
static void adamw_update_parameter(
    Tensor &param, const Tensor &grad, Tensor &m, Tensor &v,
    float lr, float beta1, float beta2, float eps, float weight_decay,
    float bias_correction1, float bias_correction2) {
  size_t n = param.size();
  if (getenv("DEBUG_OPT")) {
    fprintf(stderr, "[OPT] param.size()=%zu itemsize=%zu bytes=%zu\n",
            n, param.itemsize(), param.raw_bytes());
  }
  float *p_data = param.data();
  const float *g_data = grad.data();
  float *m_data = m.data();
  float *v_data = v.data();

  const char *gpu_enabled_env = std::getenv("GPU_ENABLED");
  bool use_gpu = false;
  if (gpu_enabled_env && std::string(gpu_enabled_env) == "1") {
    metal_bridge::initialize();
    if (metal_bridge::is_available()) {
      use_gpu = true;
    }
  }

  metal_bridge::AdamWStepParams p;
  p.lr = lr;
  p.beta1 = beta1;
  p.beta2 = beta2;
  p.eps = eps;
  p.weight_decay = weight_decay;
  p.bias_correction1 = bias_correction1;
  p.bias_correction2 = bias_correction2;
  p.n = static_cast<uint32_t>(n);

  static size_t opt_call_idx = 0;
  if (getenv("DEBUG_STATS") && opt_call_idx < 3) {
    float g0 = g_data ? g_data[0] : 0.0f;
    float m0 = m_data ? m_data[0] : 0.0f;
    float v0 = v_data ? v_data[0] : 0.0f;
    float p0 = 0.0f;
    if (param.dtype() == DType::BF16) {
      uint16_t raw = ((const uint16_t*)param.raw_ptr())[0];
      uint32_t u32 = static_cast<uint32_t>(raw) << 16;
      std::memcpy(&p0, &u32, sizeof(float));
    } else {
      p0 = p_data ? p_data[0] : 0.0f;
    }
    float m_next = beta1 * m0 + (1.0f - beta1) * g0;
    float v_next = beta2 * v0 + (1.0f - beta2) * g0 * g0;
    float m_hat = m_next / bias_correction1;
    float v_hat = v_next / bias_correction2;
    float delta = -lr * (m_hat / (std::sqrt(v_hat) + eps)) - lr * weight_decay * p0;
    float p_next = p0 + delta;

    std::cout << "\n[ADAMW-DEBUG Param #" << opt_call_idx << "]\n"
              << "  old_weight:       " << p0 << "\n"
              << "  grad:             " << g0 << "\n"
              << "  m (first moment): " << m_next << "\n"
              << "  v (sec moment):   " << v_next << "\n"
              << "  lr * update_step: " << delta << "\n"
              << "  new_weight:       " << p_next << "\n"
              << "  (lr=" << lr << ", eps=" << eps << ", bc1=" << bias_correction1 << ", bc2=" << bias_correction2 << ")\n"
              << std::endl;
    opt_call_idx++;
  }

  metal_bridge::adamw_step(p_data, g_data, m_data, v_data, p);
}

/**
 * @brief Performs a single optimization step using the AdamW algorithm.
 *
 * Computes bias corrections, and updates all registered parameters in parallel
 * utilizing first and second momentum tracking states.
 *
 * @param lr Learning rate.
 */
void AdamWOptimizer::step(float lr) {
  step_count_++;
  float bias_correction1 = 1.0f - std::pow(beta1_, step_count_);
  float bias_correction2 = 1.0f - std::pow(beta2_, step_count_);

  for (size_t i = 0; i < params_.size(); ++i) {
    adamw_update_parameter(*params_[i], *grads_[i], m_states_[i], v_states_[i],
                           lr, beta1_, beta2_, eps_, weight_decay_,
                           bias_correction1, bias_correction2);
  }
}
