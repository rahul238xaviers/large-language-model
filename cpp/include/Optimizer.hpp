/**
 * @file Optimizer.hpp
 * @brief Extensible optimizer framework for training deep neural networks
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * Exposes a base Optimizer class and concrete subclasses:
 * 1. SGDOptimizer: Standard Stochastic Gradient Descent.
 * 2. AdamWOptimizer: Adam optimizer with decoupled weight decay to prevent
 *    parameter expansion and improve generalization.
 */

#pragma once

#include "Tensor.hpp"
#include <vector>

/**
 * @brief Base Optimizer class.
 *
 * Tracks references to model parameters and their corresponding gradient
 * tensors.
 */
class Optimizer {
public:
  virtual ~Optimizer() = default;

  /**
   * @brief Registers a parameter tensor and its corresponding gradient tensor.
   *
   * @param param Pointer to the parameter tensor to optimize.
   * @param grad Pointer to the gradient tensor.
   */
  virtual void register_parameter(Tensor *param, const Tensor *grad);

  /**
   * @brief Performs a single optimization update step.
   *
   * @param lr Learning rate for the current step.
   */
  virtual void step(float lr) = 0;

  /**
   * @brief Clears all registered parameters.
   */
  void clear();

protected:
  std::vector<Tensor *> params_;
  std::vector<const Tensor *> grads_;
};

/**
 * @brief SGDOptimizer subclass.
 *
 * Implements simple gradient descent update step: w = w - lr * grad.
 */
class SGDOptimizer : public Optimizer {
public:
  /**
   * @brief Performs a single SGD step.
   *
   * @param lr Learning rate.
   */
  void step(float lr) override;
};

/**
 * @brief AdamWOptimizer subclass.
 *
 * Implements AdamW (Adam with Decoupled Weight Decay).
 */
class AdamWOptimizer : public Optimizer {
public:
  /**
   * @brief Construct a new AdamWOptimizer object.
   *
   * @param beta1 Exponential decay rate for the first moment estimates (default
   * 0.9).
   * @param beta2 Exponential decay rate for the second moment estimates
   * (default 0.999).
   * @param eps Small epsilon to prevent division by zero (default 1e-8).
   * @param weight_decay Decoupled weight decay coefficient (default 0.01).
   */
  AdamWOptimizer(float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f,
                 float weight_decay = 0.01f);

  /**
   * @brief Registers a parameter tensor and allocates matching moment states.
   *
   * @param param Pointer to the parameter tensor.
   * @param grad Pointer to the gradient tensor.
   */
  void register_parameter(Tensor *param, const Tensor *grad) override;

  /**
   * @brief Performs a single AdamW step.
   *
   * @param lr Learning rate.
   */
  void step(float lr) override;

  const std::vector<Tensor> &m_states() const { return m_states_; }
  std::vector<Tensor> &m_states() { return m_states_; }
  const std::vector<Tensor> &v_states() const { return v_states_; }
  std::vector<Tensor> &v_states() { return v_states_; }
  size_t step_count() const { return step_count_; }
  size_t &step_count() { return step_count_; }

private:
  float beta1_;
  float beta2_;
  float eps_;
  float weight_decay_;
  size_t step_count_;
  std::vector<Tensor> m_states_;
  std::vector<Tensor> v_states_;
};
