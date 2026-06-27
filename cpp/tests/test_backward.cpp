/**
 * @file test_backward.cpp
 * @brief Numerical gradient checking and overfitting verification for the complete backpropagation flow
 */

#include "Transformer.hpp"
#include "Positional.hpp"
#include "Tensor.hpp"
#include "Loss.hpp"
#include <cassert>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>

void print_test_result(const std::string &desc, bool passed) {
  std::cout << std::left << std::setw(60) << desc
            << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m")
            << std::endl;
}

// Helper to compute model loss for a specific set of weights
float compute_model_loss(const Transformer &model, const Tensor &tokens, const Tensor &targets, const RoPE &rope) {
  Tensor logits = model.forward(tokens);
  CrossEntropyLoss loss_fn;
  return loss_fn.forward(logits, targets);
}

int main() {
  std::cout << "[INFO] Starting Complete Backpropagation Verification Suite" << std::endl;
  std::cout << "[INFO] Target: test_backward" << std::endl << std::endl;

  int passed_checks = 0;
  int total_checks = 0;

  // Initialize a tiny configuration for numerical gradient checking
  ModelConfig config;
  config.vocab_size = 10;
  config.hidden_dim = 8;
  config.intermediate_dim = 16;
  config.n_layers = 1;
  config.n_heads = 2;
  config.n_kv_heads = 2;
  config.head_dim = 4;
  config.max_seq_len = 8;
  config.rope_base = 10000.0f;
  config.rms_norm_eps = 1e-5f;

  Transformer model(config);
  RoPE rope(config.head_dim, config.max_seq_len, config.rope_base);

  // Initialize parameters to small non-zero values
  model.token_embeddings().fill(0.1f);
  model.output_projection().fill(0.05f);
  for (auto &layer : model.layers()) {
    layer.w_gate.fill(0.1f);
    layer.w_up.fill(0.08f);
    layer.w_down.fill(0.1f);
    layer.attn.Wq().fill(0.05f);
    layer.attn.Wk().fill(0.05f);
    layer.attn.Wv().fill(0.05f);
    layer.attn.Wo().fill(0.05f);
  }

  // Create simple input tokens and target outputs (batch=1, seq_len=3)
  Tensor tokens({1, 3});
  tokens(0, 0) = 1.0f;
  tokens(0, 1) = 2.0f;
  tokens(0, 2) = 3.0f;

  Tensor targets({1, 3});
  targets(0, 0) = 2.0f;
  targets(0, 1) = 3.0f;
  targets(0, 2) = 4.0f;

  // ============================================================================
  // TEST 1: NUMERICAL VS ANALYTICAL GRADIENT CHECKING
  // ============================================================================
  std::cout << "================================================================================" << std::endl;
  std::cout << "SECTION 1: FINITE-DIFFERENCE NUMERICAL GRADIENT VERIFICATION" << std::endl;
  std::cout << "================================================================================" << std::endl;

  // Run analytical backward pass
  Tensor logits = model.forward(tokens);
  CrossEntropyLoss loss_fn;
  float loss = loss_fn.forward(logits, targets);
  Tensor grad_logits = loss_fn.backward(targets);

  // Allocate analytical parameter gradients
  std::vector<Tensor> grad_w_gate = {Tensor(model.layers()[0].w_gate.shape(), 0.0f)};
  std::vector<Tensor> grad_w_up = {Tensor(model.layers()[0].w_up.shape(), 0.0f)};
  std::vector<Tensor> grad_w_down = {Tensor(model.layers()[0].w_down.shape(), 0.0f)};
  std::vector<Tensor> grad_Wq = {Tensor(model.layers()[0].attn.Wq().shape(), 0.0f)};
  std::vector<Tensor> grad_Wk = {Tensor(model.layers()[0].attn.Wk().shape(), 0.0f)};
  std::vector<Tensor> grad_Wv = {Tensor(model.layers()[0].attn.Wv().shape(), 0.0f)};
  std::vector<Tensor> grad_Wo = {Tensor(model.layers()[0].attn.Wo().shape(), 0.0f)};
  Tensor grad_embeddings(model.token_embeddings().shape(), 0.0f);
  Tensor grad_output_projection(model.output_projection().shape(), 0.0f);

  model.backward(grad_logits, tokens, grad_w_gate, grad_w_up, grad_w_down,
                 grad_Wq, grad_Wk, grad_Wv, grad_Wo, grad_embeddings,
                 grad_output_projection, rope);

  // We will perturb a few weights and check if the relative/absolute error is small
  const float eps = 1e-4f;
  bool grad_check_passed = true;

  auto verify_gradient = [&](Tensor &param, Tensor &grad, const std::string &name, size_t idx) {
    float original_val = param(idx);

    // Compute Loss(w + eps)
    param(idx) = original_val + eps;
    float loss_plus = compute_model_loss(model, tokens, targets, rope);

    // Compute Loss(w - eps)
    param(idx) = original_val - eps;
    float loss_minus = compute_model_loss(model, tokens, targets, rope);

    // Restore weight
    param(idx) = original_val;

    // Numerical gradient via central differences
    float num_grad = (loss_plus - loss_minus) / (2.0f * eps);
    float ana_grad = grad(idx);

    float abs_diff = std::abs(num_grad - ana_grad);
    float norm = std::max(std::abs(num_grad) + std::abs(ana_grad), 1e-5f);
    float rel_diff = abs_diff / norm;

    // We allow a tolerance of 1e-2 for relative difference, or 1e-3 absolute difference for floats
    bool passed = (rel_diff < 1e-2f || abs_diff < 1e-3f);
    if (!passed) {
      grad_check_passed = false;
      std::cout << "[FAIL] Mismatch in " << name << " at index " << idx 
                << ": Numerical = " << num_grad << ", Analytical = " << ana_grad 
                << " (rel_diff=" << rel_diff << ", abs_diff=" << abs_diff << ")" << std::endl;
    }
  };

  // Run checks on various parameter groups
  verify_gradient(model.token_embeddings(), grad_embeddings, "token_embeddings", 12);
  verify_gradient(model.output_projection(), grad_output_projection, "output_projection", 5);
  verify_gradient(model.layers()[0].w_gate, grad_w_gate[0], "w_gate", 4);
  verify_gradient(model.layers()[0].w_up, grad_w_up[0], "w_up", 10);
  verify_gradient(model.layers()[0].w_down, grad_w_down[0], "w_down", 8);
  verify_gradient(model.layers()[0].attn.Wq(), grad_Wq[0], "Wq", 14);
  verify_gradient(model.layers()[0].attn.Wo(), grad_Wo[0], "Wo", 6);

  total_checks++;
  if (grad_check_passed) passed_checks++;
  print_test_result("TC-01: Finite-difference numerical gradient matches analytical", grad_check_passed);

  // ============================================================================
  // TEST 2: MONOTONIC LOSS DECREASE ON OVERFITTING
  // ============================================================================
  std::cout << std::endl;
  std::cout << "================================================================================" << std::endl;
  std::cout << "SECTION 2: MONOTONIC LOSS DECREASE ON OVERFITTING SEQUENCE" << std::endl;
  std::cout << "================================================================================" << std::endl;

  // Let's run a simple SGD training loop to verify that loss decreases monotonically
  Tensor overfit_tokens({1, 1}, 1.0f);
  Tensor overfit_targets({1, 1}, 2.0f);
  float initial_loss = compute_model_loss(model, overfit_tokens, overfit_targets, rope);
  float last_loss = initial_loss;
  bool monotonic_decrease = true;
  const float learning_rate = 0.5f;

  std::cout << "  Initial loss: " << initial_loss << std::endl;

  for (int step = 0; step < 50; ++step) {
    // 1. Forward pass
    Tensor step_logits = model.forward(overfit_tokens);
    float step_loss = loss_fn.forward(step_logits, overfit_targets);
    Tensor step_grad_logits = loss_fn.backward(overfit_targets);

    // 2. Reset and compute analytical backward pass
    grad_w_gate[0].fill(0.0f);
    grad_w_up[0].fill(0.0f);
    grad_w_down[0].fill(0.0f);
    grad_Wq[0].fill(0.0f);
    grad_Wk[0].fill(0.0f);
    grad_Wv[0].fill(0.0f);
    grad_Wo[0].fill(0.0f);
    grad_embeddings.fill(0.0f);
    grad_output_projection.fill(0.0f);

    model.backward(step_grad_logits, overfit_tokens, grad_w_gate, grad_w_up, grad_w_down,
                   grad_Wq, grad_Wk, grad_Wv, grad_Wo, grad_embeddings,
                   grad_output_projection, rope);

    // 3. Simple SGD weight update: w = w - lr * grad
    auto sgd_update = [learning_rate](Tensor &param, const Tensor &grad) {
      for (size_t i = 0; i < param.size(); ++i) {
        param(i) -= learning_rate * grad(i);
      }
    };

    sgd_update(model.token_embeddings(), grad_embeddings);
    sgd_update(model.output_projection(), grad_output_projection);
    sgd_update(model.layers()[0].w_gate, grad_w_gate[0]);
    sgd_update(model.layers()[0].w_up, grad_w_up[0]);
    sgd_update(model.layers()[0].w_down, grad_w_down[0]);
    sgd_update(model.layers()[0].attn.Wq(), grad_Wq[0]);
    sgd_update(model.layers()[0].attn.Wk(), grad_Wk[0]);
    sgd_update(model.layers()[0].attn.Wv(), grad_Wv[0]);
    sgd_update(model.layers()[0].attn.Wo(), grad_Wo[0]);

    std::cout << "  Step " << std::setw(2) << (step + 1) << " | Loss: " << std::fixed << std::setprecision(6) << step_loss << std::endl;

    if (step > 0 && step_loss >= last_loss + 1e-4f) {
      // Allow minor fluctuations due to float limits, but it must generally decrease
      monotonic_decrease = false;
    }
    last_loss = step_loss;
  }

  total_checks++;
  if (monotonic_decrease) passed_checks++;
  print_test_result("TC-02: Loss decreases monotonically under simple gradient descent", monotonic_decrease);

  total_checks++;
  bool target_loss_reached = (last_loss < 0.3f);
  if (target_loss_reached) passed_checks++;
  print_test_result("TC-03: Overfit target loss is successfully reached (< 0.3)", target_loss_reached);

  std::cout << std::endl;
  std::cout << "================================================================================" << std::endl;
  std::cout << "SECTION 3: PERFORMANCE PROFILING & BENCHMARKS" << std::endl;
  std::cout << "================================================================================" << std::endl;

  {
    size_t bench_batch = 2;
    size_t bench_seq_len = 32;
    size_t bench_vocab_size = 1000;
    size_t bench_hidden_dim = 512;
    size_t bench_intermediate_dim = 1384;
    size_t bench_n_layers = 2;
    size_t bench_n_heads = 8;
    size_t bench_n_kv_heads = 4;
    size_t bench_head_dim = 64;

    std::cout << "Complete Transformer Model Benchmark (Forward & Backward Passes):" << std::endl;
    std::cout << "  Input Dimensions:     [" << bench_batch << ", " << bench_seq_len << "]" << std::endl;
    std::cout << "  Model Layers:         " << bench_n_layers << std::endl;
    std::cout << "  Hidden Dimension:     " << bench_hidden_dim << std::endl;
    std::cout << "  Vocab Size:           " << bench_vocab_size << std::endl;

    ModelConfig bench_config;
    bench_config.vocab_size = bench_vocab_size;
    bench_config.hidden_dim = bench_hidden_dim;
    bench_config.intermediate_dim = bench_intermediate_dim;
    bench_config.n_layers = bench_n_layers;
    bench_config.n_heads = bench_n_heads;
    bench_config.n_kv_heads = bench_n_kv_heads;
    bench_config.head_dim = bench_head_dim;
    bench_config.max_seq_len = 128;
    bench_config.rope_base = 10000.0f;
    bench_config.rms_norm_eps = 1e-5f;

    Transformer bench_model(bench_config);
    RoPE bench_rope(bench_head_dim, 128, 10000.0f);

    bench_model.token_embeddings().fill(0.01f);
    bench_model.output_projection().fill(0.01f);
    for (auto &layer : bench_model.layers()) {
      layer.w_gate.fill(0.01f);
      layer.w_up.fill(0.01f);
      layer.w_down.fill(0.01f);
      layer.attn.Wq().fill(0.01f);
      layer.attn.Wk().fill(0.01f);
      layer.attn.Wv().fill(0.01f);
      layer.attn.Wo().fill(0.01f);
    }

    std::vector<float> bench_token_data(bench_batch * bench_seq_len);
    std::vector<float> bench_target_data(bench_batch * bench_seq_len);
    for (size_t i = 0; i < bench_token_data.size(); ++i) {
      bench_token_data[i] = static_cast<float>(i % bench_vocab_size);
      bench_target_data[i] = static_cast<float>((i + 1) % bench_vocab_size);
    }
    Tensor bench_tokens({bench_batch, bench_seq_len}, bench_token_data);
    Tensor bench_targets({bench_batch, bench_seq_len}, bench_target_data);

    // Warm-up pass
    Tensor logits_warm = bench_model.forward(bench_tokens);
    CrossEntropyLoss bench_loss_fn;
    float loss_warm = bench_loss_fn.forward(logits_warm, bench_targets);
    Tensor grad_logits_warm = bench_loss_fn.backward(bench_targets);

    std::vector<Tensor> g_w_gate, g_w_up, g_w_down, g_Wq, g_Wk, g_Wv, g_Wo;
    for (size_t l = 0; l < bench_n_layers; ++l) {
      g_w_gate.push_back(Tensor(bench_model.layers()[l].w_gate.shape(), 0.0f));
      g_w_up.push_back(Tensor(bench_model.layers()[l].w_up.shape(), 0.0f));
      g_w_down.push_back(Tensor(bench_model.layers()[l].w_down.shape(), 0.0f));
      g_Wq.push_back(Tensor(bench_model.layers()[l].attn.Wq().shape(), 0.0f));
      g_Wk.push_back(Tensor(bench_model.layers()[l].attn.Wk().shape(), 0.0f));
      g_Wv.push_back(Tensor(bench_model.layers()[l].attn.Wv().shape(), 0.0f));
      g_Wo.push_back(Tensor(bench_model.layers()[l].attn.Wo().shape(), 0.0f));
    }
    Tensor g_embeddings(bench_model.token_embeddings().shape(), 0.0f);
    Tensor g_output_projection(bench_model.output_projection().shape(), 0.0f);

    bench_model.backward(grad_logits_warm, bench_tokens, g_w_gate, g_w_up, g_w_down,
                         g_Wq, g_Wk, g_Wv, g_Wo, g_embeddings,
                         g_output_projection, bench_rope);

    const int runs = 5;

    // Forward pass timing
    std::cout << "  [INFO] Running Forward Pass benchmark..." << std::endl;
    std::vector<double> fwd_latencies;
    for (int r = 0; r < runs; ++r) {
      auto start = std::chrono::high_resolution_clock::now();
      Tensor logits_loop = bench_model.forward(bench_tokens);
      float loss_loop = bench_loss_fn.forward(logits_loop, bench_targets);
      auto end = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
      fwd_latencies.push_back(ms);
      std::cout << "    Run " << (r + 1) << ": " << ms << " ms" << std::endl;
    }
    double fwd_avg_ms = 0.0;
    for (double v : fwd_latencies) fwd_avg_ms += v;
    fwd_avg_ms /= runs;

    // Backward pass timing
    std::cout << "  [INFO] Running Backward Pass benchmark..." << std::endl;
    std::vector<double> bwd_latencies;
    for (int r = 0; r < runs; ++r) {
      // reset grads
      g_embeddings.fill(0.0f);
      g_output_projection.fill(0.0f);
      for (size_t l = 0; l < bench_n_layers; ++l) {
        g_w_gate[l].fill(0.0f);
        g_w_up[l].fill(0.0f);
        g_w_down[l].fill(0.0f);
        g_Wq[l].fill(0.0f);
        g_Wk[l].fill(0.0f);
        g_Wv[l].fill(0.0f);
        g_Wo[l].fill(0.0f);
      }
      auto start = std::chrono::high_resolution_clock::now();
      bench_model.backward(grad_logits_warm, bench_tokens, g_w_gate, g_w_up, g_w_down,
                           g_Wq, g_Wk, g_Wv, g_Wo, g_embeddings,
                           g_output_projection, bench_rope);
      auto end = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
      bwd_latencies.push_back(ms);
      std::cout << "    Run " << (r + 1) << ": " << ms << " ms" << std::endl;
    }
    double bwd_avg_ms = 0.0;
    for (double v : bwd_latencies) bwd_avg_ms += v;
    bwd_avg_ms /= runs;

    std::cout << std::endl;
    std::cout << "  Forward Average Latency:  " << fwd_avg_ms << " ms" << std::endl;
    std::cout << "  Backward Average Latency: " << bwd_avg_ms << " ms" << std::endl;
  }

  std::cout << "================================================================================" << std::endl;
  std::cout << "TEST EXECUTION SUMMARY" << std::endl;
  std::cout << "================================================================================" << std::endl;
  std::cout << "  Total Checks:            " << total_checks << std::endl;
  std::cout << "  Passed Checks:           " << passed_checks << std::endl;
  std::cout << "  Failed Checks:           " << (total_checks - passed_checks) << std::endl;
  std::cout << "  Status:                  " << (passed_checks == total_checks ? "\033[32mSUCCESS\033[0m" : "\033[31mFAILURE\033[0m") << std::endl;
  std::cout << "================================================================================" << std::endl;

  return (passed_checks == total_checks) ? 0 : 1;
}
