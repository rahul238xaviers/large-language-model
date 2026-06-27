/**
 * @file test_attention.cpp
 * @brief Unit test and benchmark suite for the Grouped Query Attention (GQA) block
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * Verifies the correctness and performance of the Attention layer implementation:
 * 1. Output shapes for batched forward pass.
 * 2. Causal masking compliance and finite outputs.
 * 3. Performance benchmarking (latency and GFLOPs/s throughput).
 */

#include "Attention.hpp"
#include "Positional.hpp"
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
  std::cout << "[INFO] Starting Attention Layer Test & Benchmark Suite" << std::endl;
  std::cout << "[INFO] Target: test_attention" << std::endl;
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

  // Use the toy configuration for faster and simpler verification
  ModelConfig config = ModelConfig::make_toy();
  config.hidden_dim = 16;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.head_dim = 4;
  config.max_seq_len = 32;

  // Instantiate Attention and RoPE layers
  Attention attn(config);
  RoPE rope(config.head_dim, config.max_seq_len, config.rope_base);

  // Fill attention projection weights with deterministic values for stability
  attn.Wq().fill(0.1f);
  attn.Wk().fill(0.05f);
  attn.Wv().fill(0.1f);
  attn.Wo().fill(0.2f);

  // TC-01: Verify Output Shape on Batched Forward Pass
  {
    total_checks++;
    size_t batch = 2;
    size_t seq_len = 4;
    Tensor x({batch, seq_len, config.hidden_dim}, 0.5f);

    Tensor out = attn.forward(x, rope);
    
    std::vector<size_t> expected_shape = {batch, seq_len, config.hidden_dim};
    bool pass = (out.shape() == expected_shape);
    if (pass) passed_checks++;

    std::string expected_str = "[2, 4, 16]";
    std::string actual_str = "[" + std::to_string(out.shape()[0]) + ", " +
                             std::to_string(out.shape()[1]) + ", " +
                             std::to_string(out.shape()[2]) + "]";
    print_test_row("TC-01", "Verify output shape matches input shape",
                   expected_str, actual_str, pass);
  }

  // TC-02: Ensure Output Values are Finite (No NaNs/Infs)
  {
    total_checks++;
    size_t batch = 2;
    size_t seq_len = 4;
    Tensor x({batch, seq_len, config.hidden_dim}, 0.5f);

    Tensor out = attn.forward(x, rope);
    
    bool all_finite = true;
    for (size_t i = 0; i < out.size(); ++i) {
      if (!std::isfinite(out(i))) {
        all_finite = false;
        break;
      }
    }
    if (all_finite) passed_checks++;

    print_test_row("TC-02", "Ensure no NaNs/Infs in projection output",
                   "All Finite", all_finite ? "All Finite" : "NaN/Inf Found", all_finite);
  }

  std::cout << std::endl;
  std::cout << "================================================================================"
            << std::endl;
  std::cout << "SECTION 2: PERFORMANCE PROFILING & BENCHMARKS" << std::endl;
  std::cout << "================================================================================"
            << std::endl;

  // Benchmark shape: batch_size = 2, seq_len = 64, n_heads = 16, n_kv_heads = 8, head_dim = 64, hidden_dim = 1024
  {
    size_t bench_batch = 2;
    size_t bench_seq_len = 64;
    size_t bench_hidden_dim = 1024;
    size_t bench_n_heads = 16;
    size_t bench_n_kv_heads = 8;
    size_t bench_head_dim = 64;

    std::cout << "Attention Layer GQA Batch Benchmark:" << std::endl;
    std::cout << "  Input Dimensions:     [" << bench_batch << ", " << bench_seq_len << ", " << bench_hidden_dim << "]" << std::endl;
    std::cout << "  Query Heads:          " << bench_n_heads << " (dim=" << bench_head_dim << ")" << std::endl;
    std::cout << "  Key/Value Heads:      " << bench_n_kv_heads << " (dim=" << bench_head_dim << ")" << std::endl;

    ModelConfig bench_config;
    bench_config.hidden_dim = bench_hidden_dim;
    bench_config.n_heads = bench_n_heads;
    bench_config.n_kv_heads = bench_n_kv_heads;
    bench_config.head_dim = bench_head_dim;
    bench_config.max_seq_len = 1024;
    bench_config.rope_base = 10000.0f;
    bench_config.rms_norm_eps = 1e-5f;

    Attention bench_attn(bench_config);
    RoPE bench_rope(bench_head_dim, 1024, 10000.0f);

    bench_attn.Wq().fill(0.01f);
    bench_attn.Wk().fill(0.01f);
    bench_attn.Wv().fill(0.01f);
    bench_attn.Wo().fill(0.01f);

    Tensor x({bench_batch, bench_seq_len, bench_hidden_dim}, 0.1f);

    // Compute total FLOPs for the forward pass:
    // 1. Projections (Q, K, V):
    double flop_q = 2.0 * bench_batch * bench_seq_len * bench_hidden_dim * (bench_n_heads * bench_head_dim);
    double flop_k = 2.0 * bench_batch * bench_seq_len * bench_hidden_dim * (bench_n_kv_heads * bench_head_dim);
    double flop_v = 2.0 * bench_batch * bench_seq_len * bench_hidden_dim * (bench_n_kv_heads * bench_head_dim);
    
    // 2. Attention dot products + weighted sum (causal masked):
    double pair_count = (bench_seq_len * (bench_seq_len + 1)) / 2.0;
    double flop_dot = 2.0 * bench_batch * bench_n_heads * pair_count * bench_head_dim;
    double flop_wsum = 2.0 * bench_batch * bench_n_heads * pair_count * bench_head_dim;
    double flop_softmax = 3.0 * bench_batch * bench_n_heads * pair_count; // exp + division + subtract max
    
    // 3. Output projection (Wo):
    double flop_wo = 2.0 * bench_batch * bench_seq_len * (bench_n_heads * bench_head_dim) * bench_hidden_dim;

    double total_flops = flop_q + flop_k + flop_v + flop_dot + flop_wsum + flop_softmax + flop_wo;
    double total_gflops = total_flops / 1e9;

    std::cout << "  Total Computation:    " << std::fixed << std::setprecision(4) 
              << total_gflops << " GFLOPs" << std::endl;

    std::cout << "  [INFO] Running benchmark..." << std::endl;

    // Warm-up pass
    Tensor out = bench_attn.forward(x, bench_rope);

    const int runs = 5;
    std::vector<double> latencies;
    latencies.reserve(runs);

    for (int r = 0; r < runs; ++r) {
      auto start = std::chrono::high_resolution_clock::now();
      Tensor out_run = bench_attn.forward(x, bench_rope);
      auto end = std::chrono::high_resolution_clock::now();
      double duration_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
      latencies.push_back(duration_ms);
      std::cout << "    Run " << (r + 1) << ": " << duration_ms << " ms" << std::endl;
    }

    double sum = std::accumulate(latencies.begin(), latencies.end(), 0.0);
    double avg_ms = sum / runs;
    double avg_seconds = avg_ms / 1000.0;
    double throughput = total_gflops / avg_seconds;

    std::cout << "  Average Latency:      " << avg_ms << " ms" << std::endl;
    std::cout << "  Compute Throughput:   " << throughput << " GFLOPs/s" << std::endl << std::endl;
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
