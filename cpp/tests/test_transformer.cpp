/**
 * @file test_transformer.cpp
 * @brief Functional verification and performance benchmarking for the complete Transformer
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * Verifies that the top-level Transformer class:
 * 1. Correctly maps token IDs to logits of shape [batch_size, seq_len, vocab_size].
 * 2. Compiles and executes without runtime exceptions.
 * 3. Benchmarks forward-pass latency and outputs throughput metrics.
 */

#include "Transformer.hpp"
#include "Tensor.hpp"
#include "TransformerConfig.hpp"
#include <cassert>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

void print_test_row(const std::string &id, const std::string &desc,
                    const std::string &expected, const std::string &actual,
                    bool passed) {
  std::cout << std::left << std::setw(10) << id << std::setw(40) << desc
            << std::setw(20) << expected << std::setw(20) << actual
            << (passed ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m")
            << std::endl;
}

int main() {
  std::cout << "[INFO] Starting Transformer Model Test & Benchmark Suite" << std::endl;
  std::cout << "[INFO] Target: test_transformer" << std::endl;
  std::cout << "[INFO] System: macOS arm64 (Apple Silicon)" << std::endl
            << std::endl;

  std::cout << "================================================================================"
            << std::endl;
  std::cout << "SECTION 1: FUNCTIONAL & SHAPE VERIFICATION" << std::endl;
  std::cout << "================================================================================"
            << std::endl;
  std::cout << std::left << std::setw(10) << "Test ID" << std::setw(40)
            << "Test Description" << std::setw(20) << "Expected Value"
            << std::setw(20) << "Actual Value"
            << "Status" << std::endl;
  std::cout << std::string(96, '-') << std::endl;

  int passed_checks = 0;
  int total_checks = 0;

  // Initialize a toy model configuration for testing
  ModelConfig config = ModelConfig::make_toy();
  config.vocab_size = 1000; // Small vocab size for faster test runs
  config.hidden_dim = 16;
  config.intermediate_dim = 32;
  config.n_layers = 2;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.head_dim = 4;
  config.max_seq_len = 32;

  // Instantiate the Transformer model
  Transformer model(config);

  // Initialize parameters to small non-zero values
  model.token_embeddings().fill(0.01f);
  model.output_projection().fill(0.02f);
  for (auto &layer : model.layers()) {
    layer.w_gate.fill(0.05f);
    layer.w_up.fill(0.05f);
    layer.w_down.fill(0.05f);
    layer.attn.Wq().fill(0.01f);
    layer.attn.Wk().fill(0.01f);
    layer.attn.Wv().fill(0.01f);
    layer.attn.Wo().fill(0.01f);
  }

  // Create input token IDs (batch_size = 2, seq_len = 4)
  size_t batch = 2;
  size_t seq_len = 4;
  std::vector<float> token_data = {
      10.0f, 20.0f, 30.0f, 40.0f, // Batch 1
      50.0f, 60.0f, 70.0f, 80.0f  // Batch 2
  };
  Tensor tokens({batch, seq_len}, token_data);

  // TC-01: Verify Output Logits Shape
  {
    total_checks++;
    Tensor logits = model.forward(tokens);

    std::vector<size_t> expected_shape = {batch, seq_len, config.vocab_size};
    bool pass = (logits.shape() == expected_shape);
    if (pass) passed_checks++;

    std::string expected_str = "[2, 4, 1000]";
    std::string actual_str = "[" + std::to_string(logits.shape()[0]) + ", " +
                             std::to_string(logits.shape()[1]) + ", " +
                             std::to_string(logits.shape()[2]) + "]";
    print_test_row("TC-01", "Verify logits shape matches config",
                   expected_str, actual_str, pass);
  }

  // TC-02: Ensure Output Values are Finite (No NaNs/Infs)
  {
    total_checks++;
    Tensor logits = model.forward(tokens);

    bool all_finite = true;
    for (size_t i = 0; i < logits.size(); ++i) {
      if (!std::isfinite(logits(i))) {
        all_finite = false;
        break;
      }
    }
    if (all_finite) passed_checks++;

    print_test_row("TC-02", "Ensure no NaNs/Infs in final logits",
                   "All Finite", all_finite ? "All Finite" : "NaN/Inf Found", all_finite);
  }

  std::cout << std::endl;
  std::cout << "================================================================================"
            << std::endl;
  std::cout << "SECTION 2: PERFORMANCE PROFILING & BENCHMARKS" << std::endl;
  std::cout << "================================================================================"
            << std::endl;

  // Benchmark complete forward pass on the configured toy model
  {
    std::cout << "Transformer Forward Pass Benchmark:" << std::endl;
    std::cout << "  Configuration:        2 Layers, 16 Hidden Dim, 32 FFN Dim, 1000 Vocab" << std::endl;
    std::cout << "  Input size:           Batch=" << batch << ", SeqLen=" << seq_len << std::endl;

    // Warm-up pass
    Tensor logits = model.forward(tokens);

    const int runs = 5;
    std::vector<double> latencies;
    latencies.reserve(runs);

    for (int r = 0; r < runs; ++r) {
      auto start = std::chrono::high_resolution_clock::now();
      Tensor out = model.forward(tokens);
      auto end = std::chrono::high_resolution_clock::now();
      double duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
      latencies.push_back(duration_ms);
      std::cout << "    Run " << (r + 1) << ": " << duration_ms << " ms" << std::endl;
    }

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg_ms = sum / runs;

    std::cout << "  Average Latency:      " << avg_ms << " ms" << std::endl << std::endl;
  }

  std::cout << "================================================================================"
            << std::endl;
  std::cout << "TEST EXECUTION SUMMARY" << std::endl;
  std::cout << "================================================================================"
            << std::endl;
  std::cout << "  Total Functional Checks: " << total_checks << std::endl;
  std::cout << "  Passed Checks:           " << passed_checks << std::endl;
  std::cout << "  Failed Checks:           " << (total_checks - passed_checks) << std::endl;
  std::cout << "  Status:                  " << (passed_checks == total_checks ? "\033[32mSUCCESS\033[0m" : "\033[31mFAILURE\033[0m") << std::endl;
  std::cout << "================================================================================" << std::endl;

  return (passed_checks == total_checks) ? 0 : 1;
}
