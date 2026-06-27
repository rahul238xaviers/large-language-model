/**
 * @file test_trainer.cpp
 * @brief Comprehensive verification for C++ Optimizer, Cosine Scheduler, and training loop
 */

#include "Trainer.hpp"
#include "Positional.hpp"
#include "Tensor.hpp"
#include "Loss.hpp"
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>

void print_test_row(const std::string &id, const std::string &desc,
                    const std::string &expected, const std::string &actual,
                    bool passed) {
  std::cout << std::left << std::setw(10) << id << std::setw(50) << desc
            << std::setw(20) << expected << std::setw(20) << actual
            << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m")
            << std::endl;
}

// Access internal helper run_training_step declaration for local deterministic overfitting verification
static void run_training_step(
    Transformer &model, Optimizer &optimizer, const RoPE &rope,
    const Tensor &tokens, const Tensor &targets, float lr,
    std::vector<Tensor> &grad_w_gate, std::vector<Tensor> &grad_w_up,
    std::vector<Tensor> &grad_w_down, std::vector<Tensor> &grad_Wq,
    std::vector<Tensor> &grad_Wk, std::vector<Tensor> &grad_Wv,
    std::vector<Tensor> &grad_Wo, Tensor &grad_embeddings,
    Tensor &grad_output_projection, float &loss_out);

int main() {
  std::cout << "[INFO] Starting Optimizer & Trainer Verification Target" << std::endl;
  std::cout << "[INFO] Target: test_trainer" << std::endl << std::endl;

  std::cout << "================================================================================" << std::endl;
  std::cout << "SECTION 1: FUNCTIONAL & SCHEDULER VERIFICATION" << std::endl;
  std::cout << "================================================================================" << std::endl;
  std::cout << std::left << std::setw(10) << "Test ID" << std::setw(50)
            << "Test Description" << std::setw(20) << "Expected Value"
            << std::setw(20) << "Actual Value"
            << "Status" << std::endl;
  std::cout << std::string(106, '-') << std::endl;

  int passed_checks = 0;
  int total_checks = 0;

  // Set up model configurations
  ModelConfig config = ModelConfig::make_toy();
  config.vocab_size = 10;
  config.hidden_dim = 8;
  config.intermediate_dim = 16;
  config.n_layers = 1;
  config.n_heads = 2;
  config.n_kv_heads = 2;
  config.head_dim = 4;
  config.max_seq_len = 8;

  Transformer model(config);
  RoPE rope(config.head_dim, config.max_seq_len, config.rope_base);

  // Initialize data loader with toy parameters (pointing to files validated in previous runs)
  std::string data_dir = "data/raw_chunks/bigcode/the-stack-v2/data/c++/ ";
  // Strip trailing space to yield correct path
  data_dir = "data/raw_chunks/bigcode/the-stack-v2/data/c++/";
  std::string single_file = "train-00000-of-00214.parquet";
  std::string vocab_path = "data/raw_chunks/vocabulary/cl100k_base.tiktoken";

  // TC-01: Optimizer Registration & Moment Allocation
  {
    total_checks++;
    AdamWOptimizer optimizer(0.9f, 0.999f, 1e-8f, 0.01f);
    Tensor param({4, 4}, 0.5f);
    Tensor grad({4, 4}, 0.1f);
    
    bool allocated = true;
    try {
      optimizer.register_parameter(&param, &grad);
    } catch (...) {
      allocated = false;
    }
    if (allocated) passed_checks++;
    print_test_row("TC-01", "Optimizer param moment state allocation check", "Success",
                   allocated ? "Success" : "Allocation Fail", allocated);
  }

  // TC-02: Scheduler Warmup Bound Verification
  // Construct trainer config
  TrainerConfig train_cfg;
  train_cfg.max_steps = 100;
  train_cfg.warmup_steps = 20;
  train_cfg.lr_max = 2e-4f;
  train_cfg.lr_min = 2e-5f;
  train_cfg.log_interval = 2;

  // Allocate a placeholder optimizer and data loader for trainer instantiation
  AdamWOptimizer dummy_opt;
  DataIngestion data_loader(data_dir, single_file, 100 * 1024 * 1024, 2, 8, vocab_path);
  Trainer trainer(train_cfg, model, dummy_opt, data_loader, rope);

  {
    total_checks++;
    float lr_step_0 = trainer.get_scheduled_lr(0);
    float lr_step_10 = trainer.get_scheduled_lr(10);
    float lr_step_20 = trainer.get_scheduled_lr(20);
    
    bool bounds_pass = (lr_step_0 == 0.0f) && 
                       (std::abs(lr_step_10 - 1e-4f) < 1e-6f) &&
                       (std::abs(lr_step_20 - 2e-4f) < 1e-6f);
    if (bounds_pass) passed_checks++;
    print_test_row("TC-02", "Verify linear warmup schedule bounds", "0.0 -> 1e-4 -> 2e-4",
                   std::to_string(lr_step_0) + " -> " + std::to_string(lr_step_10) + " -> " + std::to_string(lr_step_20), bounds_pass);
  }

  // TC-03: Scheduler Cosine Decay Bound Verification
  {
    total_checks++;
    float lr_step_60 = trainer.get_scheduled_lr(60); // Midpoint of decay (ratio = 40/80 = 0.5)
    float lr_step_100 = trainer.get_scheduled_lr(100); // End of decay
    
    // At ratio = 0.5, cos decay multiplier is 0.5 * (1 + cos(pi/2)) = 0.5.
    // Midpoint LR = lr_min + 0.5 * (lr_max - lr_min) = 2e-5 + 0.5 * 1.8e-4 = 1.1e-4.
    bool decay_pass = (std::abs(lr_step_60 - 1.1e-4f) < 1e-6f) &&
                      (std::abs(lr_step_100 - 2e-5f) < 1e-6f);
    if (decay_pass) passed_checks++;
    print_test_row("TC-03", "Verify cosine decay schedule endpoints", "1.1e-4 -> 2e-5",
                   std::to_string(lr_step_60) + " -> " + std::to_string(lr_step_100), decay_pass);
  }

  // TC-04: Local Deterministic Overfitting Convergence
  {
    total_checks++;
    
    // Reinitialize parameters to non-zero values
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

    AdamWOptimizer opt(0.9f, 0.999f, 1e-8f, 0.0f); // Disable weight decay for pure overfitting

    // Setup local gradients
    std::vector<Tensor> grad_w_gate, grad_w_up, grad_w_down, grad_Wq, grad_Wk, grad_Wv, grad_Wo;
    for (const auto &layer : model.layers()) {
      grad_w_gate.push_back(Tensor(layer.w_gate.shape(), 0.0f));
      grad_w_up.push_back(Tensor(layer.w_up.shape(), 0.0f));
      grad_w_down.push_back(Tensor(layer.w_down.shape(), 0.0f));
      grad_Wq.push_back(Tensor(layer.attn.Wq().shape(), 0.0f));
      grad_Wk.push_back(Tensor(layer.attn.Wk().shape(), 0.0f));
      grad_Wv.push_back(Tensor(layer.attn.Wv().shape(), 0.0f));
      grad_Wo.push_back(Tensor(layer.attn.Wo().shape(), 0.0f));
    }
    Tensor grad_embeddings(model.token_embeddings().shape(), 0.0f);
    Tensor grad_output_projection(model.output_projection().shape(), 0.0f);

    opt.register_parameter(&model.token_embeddings(), &grad_embeddings);
    opt.register_parameter(&model.output_projection(), &grad_output_projection);
    for (size_t l = 0; l < model.layers().size(); ++l) {
      opt.register_parameter(&model.layers()[l].w_gate, &grad_w_gate[l]);
      opt.register_parameter(&model.layers()[l].w_up, &grad_w_up[l]);
      opt.register_parameter(&model.layers()[l].w_down, &grad_w_down[l]);
      opt.register_parameter(&model.layers()[l].attn.Wq(), &grad_Wq[l]);
      opt.register_parameter(&model.layers()[l].attn.Wk(), &grad_Wk[l]);
      opt.register_parameter(&model.layers()[l].attn.Wv(), &grad_Wv[l]);
      opt.register_parameter(&model.layers()[l].attn.Wo(), &grad_Wo[l]);
    }

    Tensor tokens({1, 1}, 1.0f);
    Tensor targets({1, 1}, 2.0f);

    float initial_loss = 0.0f;
    float final_loss = 0.0f;
    
    // Step 0 forward loss
    {
      Tensor logits = model.forward(tokens);
      CrossEntropyLoss loss_fn;
      initial_loss = loss_fn.forward(logits, targets);
    }

    // Run SGD step updates for 40 steps
    for (int step = 0; step < 40; ++step) {
      run_training_step(model, opt, rope, tokens, targets, 0.8f,
                        grad_w_gate, grad_w_up, grad_w_down, grad_Wq,
                        grad_Wk, grad_Wv, grad_Wo, grad_embeddings,
                        grad_output_projection, final_loss);
    }

    bool overfit_pass = (final_loss < 0.1f) && (final_loss < initial_loss);
    if (overfit_pass) passed_checks++;
    print_test_row("TC-04", "Deterministic local overfitting loss convergence", "< 0.1",
                   "Loss: " + std::to_string(final_loss), overfit_pass);
  }

  // TC-05: Data loader Integration & Trainer Run Check
  {
    total_checks++;
    
    // Create optimizer
    AdamWOptimizer opt(0.9f, 0.999f, 1e-8f, 0.01f);
    
    // Create a trainer configuration for a short run
    TrainerConfig integration_cfg;
    integration_cfg.max_steps = 4;
    integration_cfg.warmup_steps = 2;
    integration_cfg.lr_max = 1e-4f;
    integration_cfg.lr_min = 1e-5f;
    integration_cfg.log_interval = 1;

    Trainer pretrain_trainer(integration_cfg, model, opt, data_loader, rope);
    
    bool trainer_success = true;
    try {
      pretrain_trainer.train();
    } catch (...) {
      trainer_success = false;
    }
    if (trainer_success) passed_checks++;
    print_test_row("TC-05", "End-to-end trainer execution with DataIngestion", "Success",
                   trainer_success ? "Success" : "Runtime Crash", trainer_success);
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

// Standard copy of run_training_step matching implementation in Trainer.cpp
static void run_training_step(
    Transformer &model, Optimizer &optimizer, const RoPE &rope,
    const Tensor &tokens, const Tensor &targets, float lr,
    std::vector<Tensor> &grad_w_gate, std::vector<Tensor> &grad_w_up,
    std::vector<Tensor> &grad_w_down, std::vector<Tensor> &grad_Wq,
    std::vector<Tensor> &grad_Wk, std::vector<Tensor> &grad_Wv,
    std::vector<Tensor> &grad_Wo, Tensor &grad_embeddings,
    Tensor &grad_output_projection, float &loss_out) {
  Tensor logits = model.forward(tokens);
  CrossEntropyLoss loss_fn;
  loss_out = loss_fn.forward(logits, targets);
  Tensor grad_logits = loss_fn.backward(targets);

  grad_embeddings.fill(0.0f);
  grad_output_projection.fill(0.0f);
  for (size_t l = 0; l < model.layers().size(); ++l) {
    grad_w_gate[l].fill(0.0f);
    grad_w_up[l].fill(0.0f);
    grad_w_down[l].fill(0.0f);
    grad_Wq[l].fill(0.0f);
    grad_Wk[l].fill(0.0f);
    grad_Wv[l].fill(0.0f);
    grad_Wo[l].fill(0.0f);
  }

  model.backward(grad_logits, tokens, grad_w_gate, grad_w_up, grad_w_down,
                 grad_Wq, grad_Wk, grad_Wv, grad_Wo, grad_embeddings,
                 grad_output_projection, rope);
  optimizer.step(lr);
}
