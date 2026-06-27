/**
 * @file test_loss.cpp
 * @brief Robust functional verification and memory bandwidth profiling for Cross-Entropy Loss
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * 1. Checks basic shape and analytical mathematical correctness.
 * 2. Checks numerical stability under extreme logit ranges (preventing NaN/Inf).
 * 3. Profiles forward & backward pass latency under training shapes.
 * 4. Computes memory bandwidth throughput in GB/s.
 */

#include "Loss.hpp"
#include "Tensor.hpp"
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
  std::cout << "[INFO] Starting CrossEntropyLoss Test & Benchmark Suite" << std::endl;
  std::cout << "[INFO] Target: test_loss" << std::endl;
  std::cout << "[INFO] System: macOS arm64 (Apple Silicon)" << std::endl
            << std::endl;

  std::cout << "================================================================================"
            << std::endl;
  std::cout << "SECTION 1: FUNCTIONAL & NUMERICAL STABILITY VERIFICATION" << std::endl;
  std::cout << "================================================================================"
            << std::endl;
  std::cout << std::left << std::setw(10) << "Test ID" << std::setw(40)
            << "Test Description" << std::setw(20) << "Expected Value"
            << std::setw(20) << "Actual Value"
            << "Status" << std::endl;
  std::cout << std::string(96, '-') << std::endl;

  int passed_checks = 0;
  int total_checks = 0;

  CrossEntropyLoss loss_fn;

  // TC-01: Uniform Logits Math Verification
  {
    total_checks++;
    Tensor logits({1, 1, 4}, 0.0f);
    Tensor targets({1, 1}, 2.0f);

    float loss = loss_fn.forward(logits, targets);
    float expected_loss = -std::log(0.25f);
    bool pass = (std::abs(loss - expected_loss) < 1e-5f);
    if (pass) passed_checks++;

    print_test_row("TC-01", "Uniform logits forward pass loss",
                   std::to_string(expected_loss), std::to_string(loss), pass);
  }

  // TC-02: Uniform Logits Gradients Verification
  {
    total_checks++;
    Tensor targets({1, 1}, 2.0f);
    Tensor grad = loss_fn.backward(targets);

    bool pass = (std::abs(grad(0, 0, 0) - 0.25f) < 1e-5f) &&
                (std::abs(grad(0, 0, 1) - 0.25f) < 1e-5f) &&
                (std::abs(grad(0, 0, 2) - (-0.75f)) < 1e-5f) &&
                (std::abs(grad(0, 0, 3) - 0.25f) < 1e-5f);
    if (pass) passed_checks++;

    print_test_row("TC-02", "Uniform logits backward pass grad",
                   "[0.25, 0.25, -0.75, 0.25]",
                   "[" + std::to_string(grad(0, 0, 0)) + ", ..., " + std::to_string(grad(0, 0, 2)) + ", ...]",
                   pass);
  }

  // TC-03: Numerical Stability Check (Extremely large logits should NOT overflow/NaN)
  {
    total_checks++;
    Tensor stable_logits({1, 1, 4}, 1000.0f); // Large value
    Tensor targets({1, 1}, 1.0f);

    float loss = loss_fn.forward(stable_logits, targets);
    bool pass = std::isfinite(loss) && (std::abs(loss - (-std::log(0.25f))) < 1e-5f);
    if (pass) passed_checks++;

    print_test_row("TC-03", "Logits subtraction trick stability check",
                   "1.386294 (Finite)", std::to_string(loss), pass);
  }

  // TC-04: Multi-Batch / Multi-Seq Shape check
  {
    total_checks++;
    size_t B = 4;
    size_t S = 8;
    size_t V = 32;
    Tensor logits({B, S, V}, 1.0f);
    Tensor targets({B, S}, 0.0f);

    float loss = loss_fn.forward(logits, targets);
    Tensor grad = loss_fn.backward(targets);

    bool shape_pass = (grad.shape() == std::vector<size_t>{B, S, V});
    if (shape_pass) passed_checks++;

    print_test_row("TC-04", "Multi-batch [4, 8, 32] output shape test",
                   "[4, 8, 32]",
                   "[" + std::to_string(grad.shape()[0]) + ", " + std::to_string(grad.shape()[1]) + ", " + std::to_string(grad.shape()[2]) + "]",
                   shape_pass);
  }

  std::cout << std::endl;
  std::cout << "================================================================================"
            << std::endl;
  std::cout << "SECTION 2: PERFORMANCE PROFILING & BENCHMARKS" << std::endl;
  std::cout << "================================================================================"
            << std::endl;

  // Profiling under practical memory loads
  {
    size_t B = 8;
    size_t S = 512;
    size_t V = 32000;

    double footprint_mb = (B * S * V * sizeof(float)) / (1024.0 * 1024.0);
    // Forward reads: B * S * V floats (logits) + writes B * S * V floats (probs)
    // Backward reads: B * S * V floats (probs) + writes B * S * V floats (grads)
    double fwd_traffic_gb = (2.0 * B * S * V * sizeof(float)) / (1.0e9);
    double bwd_traffic_gb = (2.0 * B * S * V * sizeof(float)) / (1.0e9);

    std::cout << "Memory Bandwidth Benchmark Settings:" << std::endl;
    std::cout << "  Input Dimensions:    [" << B << ", " << S << ", " << V << "]" << std::endl;
    std::cout << "  Activation Size:     " << std::fixed << std::setprecision(2) << footprint_mb << " MB" << std::endl;
    std::cout << "  Forward Traffic:     " << fwd_traffic_gb << " GB" << std::endl;
    std::cout << "  Backward Traffic:    " << bwd_traffic_gb << " GB" << std::endl << std::endl;

    Tensor logits({B, S, V}, 0.5f);
    std::vector<float> target_data(B * S);
    for (size_t i = 0; i < B * S; ++i) {
      target_data[i] = static_cast<float>(i % V);
    }
    Tensor targets({B, S}, target_data);

    // Warm-up pass
    float loss = loss_fn.forward(logits, targets);
    Tensor grad = loss_fn.backward(targets);

    const int runs = 10;
    
    // Forward Pass profiling
    std::cout << "  [INFO] Running Forward Pass benchmark..." << std::endl;
    std::vector<double> fwd_latencies;
    for (int r = 0; r < runs; ++r) {
      auto start = std::chrono::high_resolution_clock::now();
      float l = loss_fn.forward(logits, targets);
      auto end = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
      fwd_latencies.push_back(ms);
      std::cout << "    Run " << (r + 1) << ": " << ms << " ms" << std::endl;
    }
    double fwd_avg_ms = std::accumulate(fwd_latencies.begin(), fwd_latencies.end(), 0.0) / runs;
    double fwd_bandwidth = fwd_traffic_gb / (fwd_avg_ms / 1000.0);

    // Backward Pass profiling
    std::cout << "  [INFO] Running Backward Pass benchmark..." << std::endl;
    std::vector<double> bwd_latencies;
    for (int r = 0; r < runs; ++r) {
      auto start = std::chrono::high_resolution_clock::now();
      Tensor g = loss_fn.backward(targets);
      auto end = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
      bwd_latencies.push_back(ms);
      std::cout << "    Run " << (r + 1) << ": " << ms << " ms" << std::endl;
    }
    double bwd_avg_ms = std::accumulate(bwd_latencies.begin(), bwd_latencies.end(), 0.0) / runs;
    double bwd_bandwidth = bwd_traffic_gb / (bwd_avg_ms / 1000.0);

    std::cout << std::endl;
    std::cout << "  Forward Average Latency:  " << fwd_avg_ms << " ms (" << fwd_bandwidth << " GB/s)" << std::endl;
    std::cout << "  Backward Average Latency: " << bwd_avg_ms << " ms (" << bwd_bandwidth << " GB/s)" << std::endl;
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
