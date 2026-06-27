/**
 * @file Optimizer.cpp
 * @brief Implementation of Neural Network Optimizers
 */

#include "Optimizer.hpp"
#include <stdexcept>

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

void Optimizer::clear() {
  params_.clear();
  grads_.clear();
}

void SGDOptimizer::step(float lr) {
  for (size_t i = 0; i < params_.size(); ++i) {
    Tensor &param = *params_[i];
    const Tensor &grad = *grads_[i];
    for (size_t j = 0; j < param.size(); ++j) {
      param(j) -= lr * grad(j);
    }
  }
}

AdamWOptimizer::AdamWOptimizer(float beta1, float beta2, float eps, float weight_decay)
    : beta1_(beta1), beta2_(beta2), eps_(eps), weight_decay_(weight_decay), step_count_(0) {}

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
  for (size_t j = 0; j < param.size(); ++j) {
    float g = grad(j);
    m(j) = beta1 * m(j) + (1.0f - beta1) * g;
    v(j) = beta2 * v(j) + (1.0f - beta2) * g * g;

    if (weight_decay > 0.0f) {
      param(j) -= lr * weight_decay * param(j);
    }

    float m_hat = m(j) / bias_correction1;
    float v_hat = v(j) / bias_correction2;
    param(j) -= lr * m_hat / (std::sqrt(v_hat) + eps);
  }
}

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
