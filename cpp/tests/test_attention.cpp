/**
 * @file test_attention.cpp
 * @brief Comprehensive functional verification and performance benchmarks for GQA & Backpropagation
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * Runs 28 distinct functional checks to verify GQA forward and backward passes:
 * 1. Shape, finite value, causal masking compliance.
 * 2. Exception/validation checks (rank, dimensions, grouping factor).
 * 3. Numerical stability under extreme ranges (underflow/overflow).
 * 4. Mathematical symmetry, uniform distribution properties.
 * 5. Analytical gradient shapes, non-zero values, and boundary validations.
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
  std::cout << std::left << std::setw(10) << id << std::setw(50) << desc
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
  std::cout << "SECTION 1: COMPREHENSIVE FUNCTIONAL & EDGE-CASE VERIFICATION" << std::endl;
  std::cout << "================================================================================"
            << std::endl;
  std::cout << std::left << std::setw(10) << "Test ID" << std::setw(50)
            << "Test Description" << std::setw(20) << "Expected Value"
            << std::setw(20) << "Actual Value"
            << "Status" << std::endl;
  std::cout << std::string(106, '-') << std::endl;

  int passed_checks = 0;
  int total_checks = 0;

  // Use toy model configuration for verification
  ModelConfig config = ModelConfig::make_toy();
  config.hidden_dim = 16;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.head_dim = 4;
  config.max_seq_len = 32;

  Attention attn(config);
  RoPE rope(config.head_dim, config.max_seq_len, config.rope_base);

  // Fill attention projection weights with deterministic values for stability
  attn.Wq().fill(0.1f);
  attn.Wk().fill(0.05f);
  attn.Wv().fill(0.1f);
  attn.Wo().fill(0.2f);

  // --- Category A: Core Functionality (TC-01 to TC-06) ---
  {
    total_checks++;
    Tensor x({2, 4, 16}, 0.5f);
    Tensor out = attn.forward(x, rope);
    bool pass = (out.shape() == std::vector<size_t>{2, 4, 16});
    if (pass) passed_checks++;
    print_test_row("TC-01", "Verify forward shape matches input shape", "[2, 4, 16]",
                   "[" + std::to_string(out.shape()[0]) + ", " + std::to_string(out.shape()[1]) + ", " + std::to_string(out.shape()[2]) + "]", pass);
  }

  {
    total_checks++;
    Tensor x({2, 4, 16}, 0.5f);
    Tensor out = attn.forward(x, rope);
    bool all_finite = true;
    for (size_t i = 0; i < out.size(); ++i) {
      if (!std::isfinite(out(i))) all_finite = false;
    }
    if (all_finite) passed_checks++;
    print_test_row("TC-02", "Ensure no NaNs/Infs in forward output", "All Finite",
                   all_finite ? "All Finite" : "NaN/Inf Found", all_finite);
  }

  {
    total_checks++;
    Tensor x1({1, 3, 16}, 0.5f);
    Tensor x2 = x1;
    x2(0, 2, 0) = 100.0f; // Modify token 2 (future)
    Tensor out1 = attn.forward(x1, rope);
    Tensor out2 = attn.forward(x2, rope);
    bool causal = true;
    for (size_t s = 0; s < 2; ++s) {
      for (size_t d = 0; d < 16; ++d) {
        if (std::abs(out1(0, s, d) - out2(0, s, d)) > 1e-6f) causal = false;
      }
    }
    if (causal) passed_checks++;
    print_test_row("TC-03", "Causal mask strict compliance test", "Exact Match",
                   causal ? "Exact Match" : "Leaked Future", causal);
  }

  {
    total_checks++;
    Tensor x({1, 2, 16}, 10000.0f); // Massive logits
    Tensor out = attn.forward(x, rope);
    bool pass = std::isfinite(out(0, 0, 0));
    if (pass) passed_checks++;
    print_test_row("TC-04", "Extreme input stability (overflow check)", "Finite",
                   pass ? "Finite" : "NaN/Inf", pass);
  }

  {
    total_checks++;
    Tensor x({2, 4, 16}, 0.5f);
    Tensor grad_output({2, 4, 16}, 0.1f);
    Tensor grad_Wq(attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(attn.Wo().shape(), 0.0f);
    Tensor grad_x = attn.backward(grad_output, x, rope, grad_Wq, grad_Wk, grad_Wv, grad_Wo);
    bool pass = (grad_x.shape() == x.shape());
    if (pass) passed_checks++;
    print_test_row("TC-05", "Verify backward pass input gradient shape", "[2, 4, 16]",
                   "[" + std::to_string(grad_x.shape()[0]) + ", " + std::to_string(grad_x.shape()[1]) + ", " + std::to_string(grad_x.shape()[2]) + "]", pass);
  }

  {
    total_checks++;
    Tensor x({2, 4, 16}, 0.5f);
    Tensor grad_output({2, 4, 16}, 0.1f);
    Tensor grad_Wq(attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(attn.Wo().shape(), 0.0f);
    attn.backward(grad_output, x, rope, grad_Wq, grad_Wk, grad_Wv, grad_Wo);
    bool pass = (grad_Wq.shape() == attn.Wq().shape()) &&
                (grad_Wk.shape() == attn.Wk().shape()) &&
                (grad_Wv.shape() == attn.Wv().shape()) &&
                (grad_Wo.shape() == attn.Wo().shape());
    if (pass) passed_checks++;
    print_test_row("TC-06", "Verify backward weights gradient shapes", "All Match",
                   pass ? "All Match" : "Mismatch", pass);
  }

  // --- Category B: Input Shape Validation (TC-07 to TC-11) ---
  {
    total_checks++;
    Tensor x_2d({4, 16}, 0.5f); // 2D shape (invalid)
    bool throws = false;
    try {
      attn.forward(x_2d, rope);
    } catch (const std::invalid_argument &) {
      throws = true;
    }
    if (throws) passed_checks++;
    print_test_row("TC-07", "Validate rank exception for 2D inputs", "Throw Exception",
                   throws ? "Thrown" : "Not Thrown", throws);
  }

  {
    total_checks++;
    Tensor x_4d({1, 2, 3, 4}, 0.5f); // 4D shape (invalid)
    bool throws = false;
    try {
      attn.forward(x_4d, rope);
    } catch (const std::invalid_argument &) {
      throws = true;
    }
    if (throws) passed_checks++;
    print_test_row("TC-08", "Validate rank exception for 4D inputs", "Throw Exception",
                   throws ? "Thrown" : "Not Thrown", throws);
  }

  {
    total_checks++;
    Tensor x_bad_dim({1, 4, 32}, 0.5f); // Mismatched hidden dim (32 instead of 16)
    bool throws = false;
    try {
      attn.forward(x_bad_dim, rope);
    } catch (const std::invalid_argument &) {
      throws = true;
    }
    if (throws) passed_checks++;
    print_test_row("TC-09", "Validate dimension mismatch exception", "Throw Exception",
                   throws ? "Thrown" : "Not Thrown", throws);
  }

  {
    total_checks++;
    ModelConfig bad_gqa = config;
    bad_gqa.n_heads = 5;
    bad_gqa.n_kv_heads = 2; // GQA factor 5/2 = 2.5 (non-integer)
    bool throws = false;
    try {
      Attention bad_attn(bad_gqa);
    } catch (const std::invalid_argument &) {
      throws = true;
    }
    // GQA check happens implicitly during reshape/division or constructor
    // (If not thrown in constructor, we mark it passed if it throws during forward/backward)
    if (!throws) {
      try {
        Attention bad_attn(bad_gqa);
        Tensor x({1, 2, 16}, 0.5f);
        bad_attn.forward(x, rope);
      } catch (const std::runtime_error &) {
        throws = true;
      } catch (const std::invalid_argument &) {
        throws = true;
      }
    }
    if (throws) passed_checks++;
    print_test_row("TC-10", "Validate GQA non-integer division check", "Throw Exception",
                   throws ? "Thrown" : "Not Thrown", throws);
  }

  {
    total_checks++;
    ModelConfig bad_gqa = config;
    bad_gqa.n_heads = 4;
    bad_gqa.n_kv_heads = 8; // KV heads > Query heads (invalid)
    bool throws = false;
    try {
      Attention bad_attn(bad_gqa);
      Tensor x({1, 2, 16}, 0.5f);
      bad_attn.forward(x, rope);
    } catch (const std::invalid_argument &) {
      throws = true;
    } catch (const std::runtime_error &) {
      throws = true;
    }
    if (throws) passed_checks++;
    print_test_row("TC-11", "Validate n_kv_heads <= n_heads constraint", "Throw Exception",
                   throws ? "Thrown" : "Not Thrown", throws);
  }

  // --- Category C: Value Edge-Cases (TC-12 to TC-17) ---
  {
    total_checks++;
    Tensor x({1, 1, 16}, 0.5f); // seq_len = 1
    Tensor out = attn.forward(x, rope);
    bool pass = (out.shape() == std::vector<size_t>{1, 1, 16});
    if (pass) passed_checks++;
    print_test_row("TC-12", "Single-token sequence validation (seq_len=1)", "[1, 1, 16]",
                   "[" + std::to_string(out.shape()[0]) + ", " + std::to_string(out.shape()[1]) + ", " + std::to_string(out.shape()[2]) + "]", pass);
  }

  {
    total_checks++;
    Tensor x({1, 2, 16}, 0.0f); // All zeros input
    Tensor out = attn.forward(x, rope);
    bool pass = std::isfinite(out(0, 0, 0));
    if (pass) passed_checks++;
    print_test_row("TC-13", "Zero-filled inputs verification", "Finite",
                   pass ? "Finite" : "NaN/Inf", pass);
  }

  {
    total_checks++;
    Tensor x({1, 2, 16}, -0.5f); // All negative values
    Tensor out = attn.forward(x, rope);
    bool pass = std::isfinite(out(0, 0, 0));
    if (pass) passed_checks++;
    print_test_row("TC-14", "Negative values inputs verification", "Finite",
                   pass ? "Finite" : "NaN/Inf", pass);
  }

  {
    total_checks++;
    Tensor x({1, 2, 16}, 1e-35f); // Extremely small floats (near underflow)
    Tensor out = attn.forward(x, rope);
    bool pass = std::isfinite(out(0, 0, 0));
    if (pass) passed_checks++;
    print_test_row("TC-15", "Near underflow input values verification", "Finite",
                   pass ? "Finite" : "NaN/Inf", pass);
  }

  {
    total_checks++;
    Tensor x({1, 35, 16}, 0.5f); // seq_len = 35 (max_seq_len is 32)
    bool throws = false;
    try {
      attn.forward(x, rope);
    } catch (const std::invalid_argument &) {
      throws = true;
    }
    if (throws) passed_checks++;
    print_test_row("TC-16", "Validate seq_len > max_seq_len check", "Throw Exception",
                   throws ? "Thrown" : "Not Thrown", throws);
  }

  {
    total_checks++;
    Tensor x({1, 2, 16}, 0.5f);
    Tensor out = attn.forward(x, rope);
    // Average value check to verify model doesn't output pure zero
    double sum = 0.0;
    for (size_t i = 0; i < out.size(); ++i) sum += std::abs(out(i));
    bool pass = (sum > 1e-5);
    if (pass) passed_checks++;
    print_test_row("TC-17", "Verify output activation energy is non-zero", "Non-Zero",
                   pass ? "Non-Zero" : "Pure Zero Output", pass);
  }

  // --- Category D: Mathematical & Symmetry (TC-18 to TC-23) ---
  {
    total_checks++;
    Tensor x({1, 2, 16}, 1.0f);
    Tensor out = attn.forward(x, rope);
    // Because input tokens are identical, outputs at seq index 0 and 1 should be identical
    // (since causal mask on token 0 only attends to 0, and token 1 attends to 0 and 1 which are identical)
    bool symmetric = true;
    for (size_t d = 0; d < 16; ++d) {
      if (std::abs(out(0, 0, d) - out(0, 1, d)) > 1e-5f) symmetric = false;
    }
    if (symmetric) passed_checks++;
    print_test_row("TC-18", "Identical input tokens output symmetry check", "Symmetric",
                   symmetric ? "Symmetric" : "Asymmetric", symmetric);
  }

  {
    total_checks++;
    // Testing scaling: multiplying inputs by zero should yield zero output
    Tensor x({1, 2, 16}, 0.0f);
    Tensor out = attn.forward(x, rope);
    bool pass = true;
    for (size_t i = 0; i < out.size(); ++i) {
      if (std::abs(out(i)) > 1e-9f) pass = false;
    }
    if (pass) passed_checks++;
    print_test_row("TC-19", "Input scaling extreme bounds (scale=0)", "Pure Zeros",
                   pass ? "Pure Zeros" : "Non-Zeros Found", pass);
  }

  {
    total_checks++;
    // Verify that Wo weight gradient is non-zero after backward pass
    Tensor x({1, 2, 16});
    for (size_t i = 0; i < x.size(); ++i) x(i) = 0.05f * (i + 1);
    Tensor grad_output({1, 2, 16});
    for (size_t i = 0; i < grad_output.size(); ++i) grad_output(i) = 0.01f * (i + 1);

    Tensor grad_Wq(attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(attn.Wo().shape(), 0.0f);
    attn.backward(grad_output, x, rope, grad_Wq, grad_Wk, grad_Wv, grad_Wo);
    double sum = 0.0;
    for (size_t i = 0; i < grad_Wo.size(); ++i) sum += std::abs(grad_Wo(i));
    bool pass = (sum > 1e-6);
    if (pass) passed_checks++;
    print_test_row("TC-20", "Verify Wo parameter gradient accumulation", "Non-Zero",
                   pass ? "Non-Zero" : "Zero Gradient", pass);
  }

  {
    total_checks++;
    // Verify that Wq weight gradient is non-zero after backward pass
    Tensor x({1, 2, 16});
    for (size_t i = 0; i < x.size(); ++i) x(i) = 0.05f * (i + 1);
    Tensor grad_output({1, 2, 16});
    for (size_t i = 0; i < grad_output.size(); ++i) grad_output(i) = 0.01f * (i + 1);

    Tensor grad_Wq(attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(attn.Wo().shape(), 0.0f);
    attn.backward(grad_output, x, rope, grad_Wq, grad_Wk, grad_Wv, grad_Wo);
    double sum = 0.0;
    for (size_t i = 0; i < grad_Wq.size(); ++i) sum += std::abs(grad_Wq(i));
    bool pass = (sum > 1e-6);
    if (pass) passed_checks++;
    print_test_row("TC-21", "Verify Wq parameter gradient accumulation", "Non-Zero",
                   pass ? "Non-Zero" : "Zero Gradient", pass);
  }

  {
    total_checks++;
    // Verify that Wk weight gradient is non-zero after backward pass
    Tensor x({1, 2, 16});
    for (size_t i = 0; i < x.size(); ++i) x(i) = 0.05f * (i + 1);
    Tensor grad_output({1, 2, 16});
    for (size_t i = 0; i < grad_output.size(); ++i) grad_output(i) = 0.01f * (i + 1);

    Tensor grad_Wq(attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(attn.Wo().shape(), 0.0f);
    attn.backward(grad_output, x, rope, grad_Wq, grad_Wk, grad_Wv, grad_Wo);
    double sum = 0.0;
    for (size_t i = 0; i < grad_Wk.size(); ++i) sum += std::abs(grad_Wk(i));
    bool pass = (sum > 1e-6);
    if (pass) passed_checks++;
    print_test_row("TC-22", "Verify Wk parameter gradient accumulation", "Non-Zero",
                   pass ? "Non-Zero" : "Zero Gradient", pass);
  }

  {
    total_checks++;
    // Verify that Wv weight gradient is non-zero after backward pass
    Tensor x({1, 2, 16});
    for (size_t i = 0; i < x.size(); ++i) x(i) = 0.05f * (i + 1);
    Tensor grad_output({1, 2, 16});
    for (size_t i = 0; i < grad_output.size(); ++i) grad_output(i) = 0.01f * (i + 1);

    Tensor grad_Wq(attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(attn.Wo().shape(), 0.0f);
    attn.backward(grad_output, x, rope, grad_Wq, grad_Wk, grad_Wv, grad_Wo);
    double sum = 0.0;
    for (size_t i = 0; i < grad_Wv.size(); ++i) sum += std::abs(grad_Wv(i));
    bool pass = (sum > 1e-6);
    if (pass) passed_checks++;
    print_test_row("TC-23", "Verify Wv parameter gradient accumulation", "Non-Zero",
                   pass ? "Non-Zero" : "Zero Gradient", pass);
  }

  // --- Category E: Backward Pass Edge Cases (TC-24 to TC-28) ---
  {
    total_checks++;
    // Zero grad_output should propagate zero gradients everywhere
    Tensor x({1, 2, 16}, 0.5f);
    Tensor grad_output({1, 2, 16}, 0.0f); // Zero grad incoming
    Tensor grad_Wq(attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(attn.Wo().shape(), 0.0f);
    Tensor grad_x = attn.backward(grad_output, x, rope, grad_Wq, grad_Wk, grad_Wv, grad_Wo);
    double sum = 0.0;
    for (size_t i = 0; i < grad_x.size(); ++i) sum += std::abs(grad_x(i));
    bool pass = (sum < 1e-9);
    if (pass) passed_checks++;
    print_test_row("TC-24", "Incoming zero gradient propagation pass", "Pure Zeros",
                   pass ? "Pure Zeros" : "Non-Zeros Found", pass);
  }

  {
    total_checks++;
    // Mismatched grad_output shape should throw exception in backward
    Tensor x({1, 2, 16}, 0.5f);
    Tensor grad_output_bad({1, 4, 16}, 0.1f); // Mismatched sequence length
    Tensor grad_Wq(attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(attn.Wo().shape(), 0.0f);
    bool throws = false;
    try {
      attn.backward(grad_output_bad, x, rope, grad_Wq, grad_Wk, grad_Wv, grad_Wo);
    } catch (const std::invalid_argument &) {
      throws = true;
    }
    if (throws) passed_checks++;
    print_test_row("TC-25", "Validate grad_output shape mismatch check", "Throw Exception",
                   throws ? "Thrown" : "Not Thrown", throws);
  }

  {
    total_checks++;
    // Mismatched grad_Wq shape should throw exception
    Tensor x({1, 2, 16}, 0.5f);
    Tensor grad_output({1, 2, 16}, 0.1f);
    Tensor grad_Wq_bad({8, 8}, 0.0f); // Mismatched shape
    Tensor grad_Wk(attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(attn.Wo().shape(), 0.0f);
    bool throws = false;
    try {
      attn.backward(grad_output, x, rope, grad_Wq_bad, grad_Wk, grad_Wv, grad_Wo);
    } catch (const std::invalid_argument &) {
      throws = true;
    }
    if (throws) passed_checks++;
    print_test_row("TC-26", "Validate grad_Wq shape mismatch check", "Throw Exception",
                   throws ? "Thrown" : "Not Thrown", throws);
  }

  {
    total_checks++;
    // Mismatched grad_Wo shape should throw exception
    Tensor x({1, 2, 16}, 0.5f);
    Tensor grad_output({1, 2, 16}, 0.1f);
    Tensor grad_Wq(attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(attn.Wv().shape(), 0.0f);
    Tensor grad_Wo_bad({8, 8}, 0.0f); // Mismatched shape
    bool throws = false;
    try {
      attn.backward(grad_output, x, rope, grad_Wq, grad_Wk, grad_Wv, grad_Wo_bad);
    } catch (const std::invalid_argument &) {
      throws = true;
    }
    if (throws) passed_checks++;
    print_test_row("TC-27", "Validate grad_Wo shape mismatch check", "Throw Exception",
                   throws ? "Thrown" : "Not Thrown", throws);
  }

  {
    total_checks++;
    // Checking that backpropagating through scaling preserves scale properties
    Tensor x({1, 2, 16}, 0.5f);
    Tensor grad_output1({1, 2, 16}, 0.1f);
    Tensor grad_output2({1, 2, 16}, 0.2f); // 2x scaled grad_output
    Tensor grad_Wq1(attn.Wq().shape(), 0.0f);
    Tensor grad_Wk1(attn.Wk().shape(), 0.0f);
    Tensor grad_Wv1(attn.Wv().shape(), 0.0f);
    Tensor grad_Wo1(attn.Wo().shape(), 0.0f);
    Tensor grad_x1 = attn.backward(grad_output1, x, rope, grad_Wq1, grad_Wk1, grad_Wv1, grad_Wo1);

    Tensor grad_Wq2(attn.Wq().shape(), 0.0f);
    Tensor grad_Wk2(attn.Wk().shape(), 0.0f);
    Tensor grad_Wv2(attn.Wv().shape(), 0.0f);
    Tensor grad_Wo2(attn.Wo().shape(), 0.0f);
    Tensor grad_x2 = attn.backward(grad_output2, x, rope, grad_Wq2, grad_Wk2, grad_Wv2, grad_Wo2);

    bool proportional = (std::abs(grad_x2(0, 0, 0) - 2.0f * grad_x1(0, 0, 0)) < 1e-5f);
    if (proportional) passed_checks++;
    print_test_row("TC-28", "Validate gradient scaling proportionality", "Proportional",
                   proportional ? "Proportional" : "Disproportional", proportional);
  }

  std::cout << std::endl;
  std::cout << "================================================================================"
            << std::endl;
  std::cout << "SECTION 2: PERFORMANCE PROFILING & BENCHMARKS" << std::endl;
  std::cout << "================================================================================"
            << std::endl;

  // Benchmark GQA batch configurations
  {
    size_t bench_batch = 2;
    size_t bench_seq_len = 64;
    size_t bench_hidden_dim = 1024;
    size_t bench_n_heads = 16;
    size_t bench_n_kv_heads = 8;
    size_t bench_head_dim = 64;

    std::cout << "Attention Layer GQA Batch Benchmark (Forward & Backward Passes):" << std::endl;
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
    Tensor grad_output({bench_batch, bench_seq_len, bench_hidden_dim}, 0.05f);

    Tensor grad_Wq(bench_attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(bench_attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(bench_attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(bench_attn.Wo().shape(), 0.0f);

    // Warm-up pass
    Tensor out = bench_attn.forward(x, bench_rope);
    Tensor dx = bench_attn.backward(grad_output, x, bench_rope, grad_Wq, grad_Wk, grad_Wv, grad_Wo);

    const int runs = 5;
    
    // Forward pass timing
    std::cout << "  [INFO] Running Forward Pass benchmark..." << std::endl;
    std::vector<double> fwd_latencies;
    for (int r = 0; r < runs; ++r) {
      auto start = std::chrono::high_resolution_clock::now();
      Tensor y = bench_attn.forward(x, bench_rope);
      auto end = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
      fwd_latencies.push_back(ms);
      std::cout << "    Run " << (r + 1) << ": " << ms << " ms" << std::endl;
    }
    double fwd_avg_ms = std::accumulate(fwd_latencies.begin(), fwd_latencies.end(), 0.0) / runs;

    // Backward pass timing
    std::cout << "  [INFO] Running Backward Pass benchmark..." << std::endl;
    std::vector<double> bwd_latencies;
    for (int r = 0; r < runs; ++r) {
      auto start = std::chrono::high_resolution_clock::now();
      Tensor dy = bench_attn.backward(grad_output, x, bench_rope, grad_Wq, grad_Wk, grad_Wv, grad_Wo);
      auto end = std::chrono::high_resolution_clock::now();
      double ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count() / 1000.0;
      bwd_latencies.push_back(ms);
      std::cout << "    Run " << (r + 1) << ": " << ms << " ms" << std::endl;
    }
    double bwd_avg_ms = std::accumulate(bwd_latencies.begin(), bwd_latencies.end(), 0.0) / runs;

    std::cout << std::endl;
    std::cout << "  Forward Average Latency:  " << fwd_avg_ms << " ms" << std::endl;
    std::cout << "  Backward Average Latency: " << bwd_avg_ms << " ms" << std::endl;
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
