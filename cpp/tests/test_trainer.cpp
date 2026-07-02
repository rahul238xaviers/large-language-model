/**
 * @file test_trainer.cpp
 * @brief Comprehensive verification for C++ Optimizer, Cosine Scheduler, and training loop
 */

#include "Trainer.hpp"
#include "Positional.hpp"
#include "Tensor.hpp"
#include "Loss.hpp"
#include "gpu_kernel/MetalBridge.hpp"
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <vector>
#include <cstdlib>
#include <sys/resource.h>

static double get_peak_rss_mb() {
  struct rusage usage;
  if (getrusage(RUSAGE_SELF, &usage) == 0) {
    // macOS returns ru_maxrss in bytes.
    return static_cast<double>(usage.ru_maxrss) / (1024.0 * 1024.0);
  }
  return 0.0;
}

void print_test_row(const std::string &id, const std::string &desc,
                    const std::string &expected, const std::string &actual,
                    bool passed) {
  std::cout << std::left << std::setw(10) << id << std::setw(50) << desc
            << std::setw(20) << expected << std::setw(20) << actual
            << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m")
            << std::endl;
}

static void prepare_batch_tensors(
    const std::vector<std::vector<int>> &batch, size_t batch_size, size_t seq_len,
    Tensor &tokens, Tensor &targets) {
  for (size_t b = 0; b < batch_size; ++b) {
    for (size_t s = 0; s < seq_len - 1; ++s) {
      tokens(b, s) = static_cast<float>(batch[b][s]);
      targets(b, s) = static_cast<float>(batch[b][s + 1]);
    }
  }
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

  // TC-05: Data loader Integration & Trainer CPU vs GPU Profiling Benchmark
  {
    total_checks++;
    
    bool benchmark_success = true;
    try {
      // Configuration for benchmark model (2048 embedding, 24 layers, 16 heads)
      ModelConfig integration_config;
      integration_config.vocab_size = 100300;
      integration_config.hidden_dim = 2048;
      integration_config.intermediate_dim = 5504;
      integration_config.n_layers = 24;
      integration_config.n_heads = 16;
      integration_config.n_kv_heads = 8;
      integration_config.head_dim = 128; // 2048 / 16
      integration_config.max_seq_len = 128;
      integration_config.rope_base = 10000.0f;
      integration_config.rms_norm_eps = 1e-5f;

      RoPE rope_bench(integration_config.head_dim, integration_config.max_seq_len, integration_config.rope_base);

      // Create a dedicated loader for larger benchmark shapes: batch=32, seq_len=128
      std::cout << "\n[INFO] Loading and tokenizing Parquet data once for benchmark cache..." << std::endl;
      DataIngestion benchmark_loader(data_dir, single_file, 100 * 1024 * 1024, 32, 128, vocab_path);

      std::vector<std::vector<std::vector<int>>> cached_batches;
      for (int step = 0; step < 4; ++step) {
        auto batch = benchmark_loader.get_batch();
        if (batch.empty()) {
          std::cerr << "[ERROR] Benchmark loader returned empty batch!" << std::endl;
          benchmark_success = false;
          break;
        }
        cached_batches.push_back(batch);
      }

      if (benchmark_success) {
        // Prepare shared inputs to ensure bit-wise parity and avoid loading twice
        std::vector<Tensor> cached_tokens;
        std::vector<Tensor> cached_targets;
        for (int step = 0; step < 4; ++step) {
          Tensor tokens({32, 128});
          Tensor targets({32, 128});
          prepare_batch_tensors(cached_batches[step], 32, 129, tokens, targets);
          cached_tokens.push_back(tokens);
          cached_targets.push_back(targets);
        }

        // ==========================================
        // 1. CPU RUN
        // ==========================================
        double cpu_total_ms = 0.0;
        {
          std::cout << "\n>>> Starting CPU Benchmark (GPU_ENABLED=0) <<<" << std::endl;
          setenv("GPU_ENABLED", "0", 1);
          metal_bridge::reset_profile_stats();

          Transformer cpu_model(integration_config);
          cpu_model.token_embeddings().fill(0.1f);
          cpu_model.output_projection().fill(0.05f);
          for (auto &layer : cpu_model.layers()) {
            layer.w_gate.fill(0.1f);
            layer.w_up.fill(0.08f);
            layer.w_down.fill(0.1f);
            layer.attn.Wq().fill(0.05f);
            layer.attn.Wk().fill(0.05f);
            layer.attn.Wv().fill(0.05f);
            layer.attn.Wo().fill(0.05f);
          }

          AdamWOptimizer cpu_opt(0.9f, 0.999f, 1e-8f, 0.01f);
          std::vector<Tensor> cpu_grad_w_gate, cpu_grad_w_up, cpu_grad_w_down, cpu_grad_Wq, cpu_grad_Wk, cpu_grad_Wv, cpu_grad_Wo;
          for (const auto &layer : cpu_model.layers()) {
            cpu_grad_w_gate.push_back(Tensor(layer.w_gate.shape(), 0.0f));
            cpu_grad_w_up.push_back(Tensor(layer.w_up.shape(), 0.0f));
            cpu_grad_w_down.push_back(Tensor(layer.w_down.shape(), 0.0f));
            cpu_grad_Wq.push_back(Tensor(layer.attn.Wq().shape(), 0.0f));
            cpu_grad_Wk.push_back(Tensor(layer.attn.Wk().shape(), 0.0f));
            cpu_grad_Wv.push_back(Tensor(layer.attn.Wv().shape(), 0.0f));
            cpu_grad_Wo.push_back(Tensor(layer.attn.Wo().shape(), 0.0f));
          }
          Tensor cpu_grad_embeddings(cpu_model.token_embeddings().shape(), 0.0f);
          Tensor cpu_grad_output_projection(cpu_model.output_projection().shape(), 0.0f);

          cpu_opt.register_parameter(&cpu_model.token_embeddings(), &cpu_grad_embeddings);
          cpu_opt.register_parameter(&cpu_model.output_projection(), &cpu_grad_output_projection);
          for (size_t l = 0; l < cpu_model.layers().size(); ++l) {
            cpu_opt.register_parameter(&cpu_model.layers()[l].w_gate, &cpu_grad_w_gate[l]);
            cpu_opt.register_parameter(&cpu_model.layers()[l].w_up, &cpu_grad_w_up[l]);
            cpu_opt.register_parameter(&cpu_model.layers()[l].w_down, &cpu_grad_w_down[l]);
            cpu_opt.register_parameter(&cpu_model.layers()[l].attn.Wq(), &cpu_grad_Wq[l]);
            cpu_opt.register_parameter(&cpu_model.layers()[l].attn.Wk(), &cpu_grad_Wk[l]);
            cpu_opt.register_parameter(&cpu_model.layers()[l].attn.Wv(), &cpu_grad_Wv[l]);
            cpu_opt.register_parameter(&cpu_model.layers()[l].attn.Wo(), &cpu_grad_Wo[l]);
          }

          auto cpu_start_all = std::chrono::high_resolution_clock::now();
          for (int step = 0; step < 4; ++step) {
            auto start_time = std::chrono::high_resolution_clock::now();
            float loss = 0.0f;
            run_training_step(cpu_model, cpu_opt, rope_bench, cached_tokens[step], cached_targets[step], 1e-4f,
                              cpu_grad_w_gate, cpu_grad_w_up, cpu_grad_w_down, cpu_grad_Wq,
                              cpu_grad_Wk, cpu_grad_Wv, cpu_grad_Wo, cpu_grad_embeddings,
                              cpu_grad_output_projection, loss);
            auto end_time = std::chrono::high_resolution_clock::now();
            double step_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
            std::cout << "  Step " << step << " | Loss: " << std::fixed << std::setprecision(5) << loss 
                      << " | Step Latency: " << std::fixed << std::setprecision(2) << step_ms << " ms"
                      << " | Peak RAM: " << std::fixed << std::setprecision(2) << get_peak_rss_mb() << " MB"
                      << " [Profile] GPU GEMM: " << metal_bridge::count_gpu_calls << " calls (" << metal_bridge::accum_gpu_time_ms << " ms) | "
                      << "CPU GEMM: " << metal_bridge::count_cpu_calls << " calls (" << metal_bridge::accum_cpu_time_ms << " ms)" << std::endl;
            metal_bridge::reset_profile_stats();
          }
          auto cpu_end_all = std::chrono::high_resolution_clock::now();
          cpu_total_ms = std::chrono::duration_cast<std::chrono::microseconds>(cpu_end_all - cpu_start_all).count() / 1000.0;
        }

        // ==========================================
        // 2. GPU RUN
        // ==========================================
        double gpu_total_ms = 0.0;
        {
          std::cout << "\n>>> Starting GPU Benchmark (GPU_ENABLED=1) <<<" << std::endl;
          setenv("GPU_ENABLED", "1", 1);
          metal_bridge::reset_profile_stats();

          Transformer gpu_model(integration_config);
          gpu_model.token_embeddings().fill(0.1f);
          gpu_model.output_projection().fill(0.05f);
          for (auto &layer : gpu_model.layers()) {
            layer.w_gate.fill(0.1f);
            layer.w_up.fill(0.08f);
            layer.w_down.fill(0.1f);
            layer.attn.Wq().fill(0.05f);
            layer.attn.Wk().fill(0.05f);
            layer.attn.Wv().fill(0.05f);
            layer.attn.Wo().fill(0.05f);
          }

          AdamWOptimizer gpu_opt(0.9f, 0.999f, 1e-8f, 0.01f);
          std::vector<Tensor> gpu_grad_w_gate, gpu_grad_w_up, gpu_grad_w_down, gpu_grad_Wq, gpu_grad_Wk, gpu_grad_Wv, gpu_grad_Wo;
          for (const auto &layer : gpu_model.layers()) {
            gpu_grad_w_gate.push_back(Tensor(layer.w_gate.shape(), 0.0f));
            gpu_grad_w_up.push_back(Tensor(layer.w_up.shape(), 0.0f));
            gpu_grad_w_down.push_back(Tensor(layer.w_down.shape(), 0.0f));
            gpu_grad_Wq.push_back(Tensor(layer.attn.Wq().shape(), 0.0f));
            gpu_grad_Wk.push_back(Tensor(layer.attn.Wk().shape(), 0.0f));
            gpu_grad_Wv.push_back(Tensor(layer.attn.Wv().shape(), 0.0f));
            gpu_grad_Wo.push_back(Tensor(layer.attn.Wo().shape(), 0.0f));
          }
          Tensor gpu_grad_embeddings(gpu_model.token_embeddings().shape(), 0.0f);
          Tensor gpu_grad_output_projection(gpu_model.output_projection().shape(), 0.0f);

          gpu_opt.register_parameter(&gpu_model.token_embeddings(), &gpu_grad_embeddings);
          gpu_opt.register_parameter(&gpu_model.output_projection(), &gpu_grad_output_projection);
          for (size_t l = 0; l < gpu_model.layers().size(); ++l) {
            gpu_opt.register_parameter(&gpu_model.layers()[l].w_gate, &gpu_grad_w_gate[l]);
            gpu_opt.register_parameter(&gpu_model.layers()[l].w_up, &gpu_grad_w_up[l]);
            gpu_opt.register_parameter(&gpu_model.layers()[l].w_down, &gpu_grad_w_down[l]);
            gpu_opt.register_parameter(&gpu_model.layers()[l].attn.Wq(), &gpu_grad_Wq[l]);
            gpu_opt.register_parameter(&gpu_model.layers()[l].attn.Wk(), &gpu_grad_Wk[l]);
            gpu_opt.register_parameter(&gpu_model.layers()[l].attn.Wv(), &gpu_grad_Wv[l]);
            gpu_opt.register_parameter(&gpu_model.layers()[l].attn.Wo(), &gpu_grad_Wo[l]);
          }

          auto gpu_start_all = std::chrono::high_resolution_clock::now();
          for (int step = 0; step < 4; ++step) {
            auto start_time = std::chrono::high_resolution_clock::now();
            float loss = 0.0f;
            run_training_step(gpu_model, gpu_opt, rope_bench, cached_tokens[step], cached_targets[step], 1e-4f,
                              gpu_grad_w_gate, gpu_grad_w_up, gpu_grad_w_down, gpu_grad_Wq,
                              gpu_grad_Wk, gpu_grad_Wv, gpu_grad_Wo, gpu_grad_embeddings,
                              gpu_grad_output_projection, loss);
            auto end_time = std::chrono::high_resolution_clock::now();
            double step_ms = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count() / 1000.0;
            std::cout << "  Step " << step << " | Loss: " << std::fixed << std::setprecision(5) << loss 
                      << " | Step Latency: " << std::fixed << std::setprecision(2) << step_ms << " ms"
                      << " | Peak RAM: " << std::fixed << std::setprecision(2) << get_peak_rss_mb() << " MB"
                      << " [Profile] GPU GEMM: " << metal_bridge::count_gpu_calls << " calls (" << metal_bridge::accum_gpu_time_ms << " ms) | "
                      << "CPU GEMM: " << metal_bridge::count_cpu_calls << " calls (" << metal_bridge::accum_cpu_time_ms << " ms)" << std::endl;
            metal_bridge::reset_profile_stats();
          }
          auto gpu_end_all = std::chrono::high_resolution_clock::now();
          gpu_total_ms = std::chrono::duration_cast<std::chrono::microseconds>(gpu_end_all - gpu_start_all).count() / 1000.0;
        }

        std::cout << "\n=======================================================" << std::endl;
        std::cout << "BENCHMARK PERFORMANCE SUMMARY (4 steps, hidden=2048, layers=24, batch=32, seq=128)" << std::endl;
        std::cout << "=======================================================" << std::endl;
        std::cout << "  CPU Total Execution Time: " << std::fixed << std::setprecision(2) << cpu_total_ms << " ms" << std::endl;
        std::cout << "  GPU Total Execution Time: " << std::fixed << std::setprecision(2) << gpu_total_ms << " ms" << std::endl;
        if (gpu_total_ms > 0) {
          std::cout << "  Speedup Factor:           " << std::fixed << std::setprecision(2) << (cpu_total_ms / gpu_total_ms) << "x" << std::endl;
        }
        std::cout << "  Peak Physical RAM (RSS):  " << std::fixed << std::setprecision(2) << get_peak_rss_mb() << " MB" << std::endl;
        std::cout << "=======================================================" << std::endl;
      }
    } catch (const std::exception &e) {
      std::cout << "[ERROR] Captured exception in CPU/GPU comparative benchmark: " << e.what() << std::endl;
      benchmark_success = false;
    } catch (...) {
      std::cout << "[ERROR] Captured unknown exception in CPU/GPU comparative benchmark" << std::endl;
      benchmark_success = false;
    }

    if (benchmark_success) passed_checks++;
    print_test_row("TC-05", "End-to-end comparative trainer CPU/GPU execution", "Success",
                   benchmark_success ? "Success" : "Runtime Crash", benchmark_success);
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
