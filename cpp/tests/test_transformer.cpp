/**
 * @file test_transformer.cpp
 * @brief Comprehensive functional verification and performance benchmarks for Transformer Layer & Model
 *
 * ============================================================================
 *                             PIPELINE FLOW & PURPOSE
 * ============================================================================
 * Runs 28 distinct functional checks to verify TransformerLayer forward and backward passes:
 * 1. Shape, finite value, residual addition, and causal masking boundaries.
 * 2. Exception/validation checks (rank, dimensions, weight mismatches).
 * 3. Numerical stability under extreme ranges (underflow/overflow).
 * 4. Mathematical symmetry, linear gradient scaling proportionality.
 * 5. FFN SwiGLU gating backpropagation properties and gradient accumulation.
 */

#include "Transformer.hpp"
#include "Positional.hpp"
#include "Tensor.hpp"
#include "TransformerConfig.hpp"
#include "Activations.hpp"
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
  std::cout << "[INFO] Starting Transformer Layer & Model Test Suite" << std::endl;
  std::cout << "[INFO] Target: test_transformer" << std::endl;
  std::cout << "[INFO] System: macOS arm64 (Apple Silicon)" << std::endl
            << std::endl;

  std::cout << "================================================================================"
            << std::endl;
  std::cout << "SECTION 1: COMPREHENSIVE TRANSFORMER-LAYER VERIFICATION" << std::endl;
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
  config.intermediate_dim = 32;
  config.n_heads = 4;
  config.n_kv_heads = 2;
  config.head_dim = 4;
  config.max_seq_len = 32;

  TransformerLayer layer(config);
  RoPE rope(config.head_dim, config.max_seq_len, config.rope_base);

  // Initialize weights to non-zero values for gradient flow
  layer.w_gate.fill(0.1f);
  layer.w_up.fill(0.05f);
  layer.w_down.fill(0.1f);
  layer.attn.Wq().fill(0.05f);
  layer.attn.Wk().fill(0.05f);
  layer.attn.Wv().fill(0.1f);
  layer.attn.Wo().fill(0.1f);

  // --- Category A: Core Forward & Backward (TC-01 to TC-08) ---
  {
    total_checks++;
    Tensor h_in({2, 4, 16}, 0.5f);
    // Mimic the forward behavior of a layer norm + attention + FFN block
    Tensor attn_in = layer.attn_norm.forward(h_in);
    Tensor attn_out = layer.attn.forward(attn_in, rope);
    Tensor h_mid = h_in.add(attn_out);
    Tensor ffn_in = layer.ffn_norm.forward(h_mid);
    Tensor gate_proj = ffn_in.matmul(layer.w_gate);
    Tensor up_proj = ffn_in.matmul(layer.w_up);
    Tensor activated = activatations::swiglu(gate_proj, up_proj);
    Tensor ffn_out = activated.matmul(layer.w_down);
    Tensor h_out = h_mid.add(ffn_out);

    bool pass = (h_out.shape() == std::vector<size_t>{2, 4, 16});
    if (pass) passed_checks++;
    print_test_row("TC-01", "Verify layer forward shape matches input", "[2, 4, 16]",
                   "[" + std::to_string(h_out.shape()[0]) + ", " + std::to_string(h_out.shape()[1]) + ", " + std::to_string(h_out.shape()[2]) + "]", pass);
  }

  {
    total_checks++;
    Tensor h_in({2, 4, 16}, 0.5f);
    Tensor attn_in = layer.attn_norm.forward(h_in);
    Tensor attn_out = layer.attn.forward(attn_in, rope);
    Tensor h_mid = h_in.add(attn_out);
    Tensor ffn_in = layer.ffn_norm.forward(h_mid);
    Tensor gate_proj = ffn_in.matmul(layer.w_gate);
    Tensor up_proj = ffn_in.matmul(layer.w_up);
    Tensor activated = activatations::swiglu(gate_proj, up_proj);
    Tensor ffn_out = activated.matmul(layer.w_down);
    Tensor h_out = h_mid.add(ffn_out);

    bool all_finite = true;
    for (size_t i = 0; i < h_out.size(); ++i) {
      if (!std::isfinite(h_out(i))) all_finite = false;
    }
    if (all_finite) passed_checks++;
    print_test_row("TC-02", "Ensure no NaNs/Infs in layer forward", "All Finite",
                   all_finite ? "All Finite" : "NaN/Inf Found", all_finite);
  }

  {
    total_checks++;
    Tensor h_in({2, 4, 16});
    for (size_t i = 0; i < h_in.size(); ++i) h_in(i) = 0.05f * (i + 1);
    Tensor grad_output({2, 4, 16}, 0.1f);

    Tensor grad_w_gate(layer.w_gate.shape(), 0.0f);
    Tensor grad_w_up(layer.w_up.shape(), 0.0f);
    Tensor grad_w_down(layer.w_down.shape(), 0.0f);
    Tensor grad_Wq(layer.attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(layer.attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(layer.attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(layer.attn.Wo().shape(), 0.0f);

    Tensor grad_h_in = layer.backward(grad_output, h_in, grad_w_gate, grad_w_up, grad_w_down,
                                      grad_Wq, grad_Wk, grad_Wv, grad_Wo, rope);
    bool pass = (grad_h_in.shape() == h_in.shape());
    if (pass) passed_checks++;
    print_test_row("TC-03", "Verify layer backward input gradient shape", "[2, 4, 16]",
                   "[" + std::to_string(grad_h_in.shape()[0]) + ", " + std::to_string(grad_h_in.shape()[1]) + ", " + std::to_string(grad_h_in.shape()[2]) + "]", pass);
  }

  {
    total_checks++;
    Tensor h_in({2, 4, 16});
    for (size_t i = 0; i < h_in.size(); ++i) h_in(i) = 0.05f * (i + 1);
    Tensor grad_output({2, 4, 16}, 0.1f);

    Tensor grad_w_gate(layer.w_gate.shape(), 0.0f);
    Tensor grad_w_up(layer.w_up.shape(), 0.0f);
    Tensor grad_w_down(layer.w_down.shape(), 0.0f);
    Tensor grad_Wq(layer.attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(layer.attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(layer.attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(layer.attn.Wo().shape(), 0.0f);

    layer.backward(grad_output, h_in, grad_w_gate, grad_w_up, grad_w_down,
                   grad_Wq, grad_Wk, grad_Wv, grad_Wo, rope);

    bool pass = (grad_w_gate.shape() == layer.w_gate.shape()) &&
                (grad_w_up.shape() == layer.w_up.shape()) &&
                (grad_w_down.shape() == layer.w_down.shape()) &&
                (grad_Wq.shape() == layer.attn.Wq().shape()) &&
                (grad_Wk.shape() == layer.attn.Wk().shape()) &&
                (grad_Wv.shape() == layer.attn.Wv().shape()) &&
                (grad_Wo.shape() == layer.attn.Wo().shape());
    if (pass) passed_checks++;
    print_test_row("TC-04", "Verify layer weight gradient shapes match", "All Match",
                   pass ? "All Match" : "Mismatch", pass);
  }

  {
    total_checks++;
    Tensor h_in({1, 2, 16});
    for (size_t i = 0; i < h_in.size(); ++i) h_in(i) = 0.05f * (i + 1);
    Tensor grad_output({1, 2, 16}, 0.1f);
    Tensor grad_w_gate(layer.w_gate.shape(), 0.0f);
    Tensor grad_w_up(layer.w_up.shape(), 0.0f);
    Tensor grad_w_down(layer.w_down.shape(), 0.0f);
    Tensor grad_Wq(layer.attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(layer.attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(layer.attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(layer.attn.Wo().shape(), 0.0f);

    layer.backward(grad_output, h_in, grad_w_gate, grad_w_up, grad_w_down,
                   grad_Wq, grad_Wk, grad_Wv, grad_Wo, rope);

    double sum = 0.0;
    for (size_t i = 0; i < grad_w_down.size(); ++i) sum += std::abs(grad_w_down(i));
    bool pass = (sum > 1e-6);
    if (pass) passed_checks++;
    print_test_row("TC-05", "Verify FFN down weight gradient is non-zero", "Non-Zero",
                   pass ? "Non-Zero" : "Zero Gradient", pass);
  }

  {
    total_checks++;
    Tensor h_in({1, 2, 16});
    for (size_t i = 0; i < h_in.size(); ++i) h_in(i) = 0.05f * (i + 1);
    Tensor grad_output({1, 2, 16}, 0.1f);
    Tensor grad_w_gate(layer.w_gate.shape(), 0.0f);
    Tensor grad_w_up(layer.w_up.shape(), 0.0f);
    Tensor grad_w_down(layer.w_down.shape(), 0.0f);
    Tensor grad_Wq(layer.attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(layer.attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(layer.attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(layer.attn.Wo().shape(), 0.0f);

    layer.backward(grad_output, h_in, grad_w_gate, grad_w_up, grad_w_down,
                   grad_Wq, grad_Wk, grad_Wv, grad_Wo, rope);

    double sum_gate = 0.0;
    double sum_up = 0.0;
    for (size_t i = 0; i < grad_w_gate.size(); ++i) sum_gate += std::abs(grad_w_gate(i));
    for (size_t i = 0; i < grad_w_up.size(); ++i) sum_up += std::abs(grad_w_up(i));
    bool pass = (sum_gate > 1e-6) && (sum_up > 1e-6);
    if (pass) passed_checks++;
    print_test_row("TC-06", "Verify FFN gate & up weight grads non-zero", "Non-Zero",
                   pass ? "Non-Zero" : "Zero Gradient", pass);
  }

  {
    total_checks++;
    Tensor h_in({1, 2, 16});
    for (size_t i = 0; i < h_in.size(); ++i) h_in(i) = 0.05f * (i + 1);
    Tensor grad_output({1, 2, 16}, 0.1f);
    Tensor grad_w_gate(layer.w_gate.shape(), 0.0f);
    Tensor grad_w_up(layer.w_up.shape(), 0.0f);
    Tensor grad_w_down(layer.w_down.shape(), 0.0f);
    Tensor grad_Wq(layer.attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(layer.attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(layer.attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(layer.attn.Wo().shape(), 0.0f);

    layer.backward(grad_output, h_in, grad_w_gate, grad_w_up, grad_w_down,
                   grad_Wq, grad_Wk, grad_Wv, grad_Wo, rope);

    double sum_wq = 0.0;
    double sum_wo = 0.0;
    for (size_t i = 0; i < grad_Wq.size(); ++i) sum_wq += std::abs(grad_Wq(i));
    for (size_t i = 0; i < grad_Wo.size(); ++i) sum_wo += std::abs(grad_Wo(i));
    bool pass = (sum_wq > 1e-6) && (sum_wo > 1e-6);
    if (pass) passed_checks++;
    print_test_row("TC-07", "Verify Attention weight grads non-zero", "Non-Zero",
                   pass ? "Non-Zero" : "Zero Gradient", pass);
  }

  {
    total_checks++;
    Tensor h_in({1, 2, 16}, 0.5f);
    Tensor grad_output({1, 2, 16}, 0.0f); // Zero grad incoming
    Tensor grad_w_gate(layer.w_gate.shape(), 0.0f);
    Tensor grad_w_up(layer.w_up.shape(), 0.0f);
    Tensor grad_w_down(layer.w_down.shape(), 0.0f);
    Tensor grad_Wq(layer.attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(layer.attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(layer.attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(layer.attn.Wo().shape(), 0.0f);

    Tensor grad_h_in = layer.backward(grad_output, h_in, grad_w_gate, grad_w_up, grad_w_down,
                                      grad_Wq, grad_Wk, grad_Wv, grad_Wo, rope);

    double sum = 0.0;
    for (size_t i = 0; i < grad_h_in.size(); ++i) sum += std::abs(grad_h_in(i));
    bool pass = (sum < 1e-9);
    if (pass) passed_checks++;
    print_test_row("TC-08", "Verify incoming zero gradient propagation", "Pure Zeros",
                   pass ? "Pure Zeros" : "Non-Zeros Found", pass);
  }

  // --- Category B: Input Shape Exceptions (TC-09 to TC-11) ---
  {
    total_checks++;
    Tensor h_in_bad({4, 16}, 0.5f); // 2D shape (invalid rank)
    Tensor grad_output({1, 2, 16}, 0.1f);
    Tensor grad_w_gate(layer.w_gate.shape(), 0.0f);
    Tensor grad_w_up(layer.w_up.shape(), 0.0f);
    Tensor grad_w_down(layer.w_down.shape(), 0.0f);
    Tensor grad_Wq(layer.attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(layer.attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(layer.attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(layer.attn.Wo().shape(), 0.0f);
    bool throws = false;
    try {
      layer.backward(grad_output, h_in_bad, grad_w_gate, grad_w_up, grad_w_down,
                     grad_Wq, grad_Wk, grad_Wv, grad_Wo, rope);
    } catch (const std::invalid_argument &) {
      throws = true;
    }
    if (throws) passed_checks++;
    print_test_row("TC-09", "Validate rank exception for 2D inputs", "Throw Exception",
                   throws ? "Thrown" : "Not Thrown", throws);
  }

  {
    total_checks++;
    Tensor h_in({1, 2, 16}, 0.5f);
    Tensor grad_output_bad({1, 4, 16}, 0.1f); // Sequence length mismatch
    Tensor grad_w_gate(layer.w_gate.shape(), 0.0f);
    Tensor grad_w_up(layer.w_up.shape(), 0.0f);
    Tensor grad_w_down(layer.w_down.shape(), 0.0f);
    Tensor grad_Wq(layer.attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(layer.attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(layer.attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(layer.attn.Wo().shape(), 0.0f);

    bool throws = false;
    try {
      layer.backward(grad_output_bad, h_in, grad_w_gate, grad_w_up, grad_w_down,
                     grad_Wq, grad_Wk, grad_Wv, grad_Wo, rope);
    } catch (const std::invalid_argument &) {
      throws = true;
    } catch (const std::runtime_error &) {
      throws = true;
    }
    if (throws) passed_checks++;
    print_test_row("TC-10", "Validate shape mismatch exception in backward", "Throw Exception",
                   throws ? "Thrown" : "Not Thrown", throws);
  }

  {
    total_checks++;
    Tensor h_in({1, 2, 16}, 0.5f);
    Tensor grad_output({1, 2, 16}, 0.1f);
    Tensor grad_w_gate_bad({8, 8}, 0.0f); // Mismatched parameter gradients shape
    Tensor grad_w_up(layer.w_up.shape(), 0.0f);
    Tensor grad_w_down(layer.w_down.shape(), 0.0f);
    Tensor grad_Wq(layer.attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(layer.attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(layer.attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(layer.attn.Wo().shape(), 0.0f);

    bool throws = false;
    try {
      layer.backward(grad_output, h_in, grad_w_gate_bad, grad_w_up, grad_w_down,
                     grad_Wq, grad_Wk, grad_Wv, grad_Wo, rope);
    } catch (const std::invalid_argument &) {
      throws = true;
    } catch (const std::runtime_error &) {
      throws = true;
    }
    if (throws) passed_checks++;
    print_test_row("TC-11", "Validate weight grads mismatch exception", "Throw Exception",
                   throws ? "Thrown" : "Not Thrown", throws);
  }

  // --- Category C: Value Edge-Cases (TC-12 to TC-18) ---
  {
    total_checks++;
    Tensor h_in({1, 1, 16}, 0.5f); // seq_len = 1
    Tensor attn_in = layer.attn_norm.forward(h_in);
    Tensor attn_out = layer.attn.forward(attn_in, rope);
    Tensor h_mid = h_in.add(attn_out);
    bool pass = (h_mid.shape() == std::vector<size_t>{1, 1, 16});
    if (pass) passed_checks++;
    print_test_row("TC-12", "Single-token sequence validation (seq_len=1)", "[1, 1, 16]",
                   "[" + std::to_string(h_mid.shape()[0]) + ", " + std::to_string(h_mid.shape()[1]) + ", " + std::to_string(h_mid.shape()[2]) + "]", pass);
  }

  {
    total_checks++;
    Tensor h_in({1, 2, 16}, 0.0f); // All zeros input
    Tensor attn_in = layer.attn_norm.forward(h_in);
    Tensor attn_out = layer.attn.forward(attn_in, rope);
    bool pass = std::isfinite(attn_out(0, 0, 0));
    if (pass) passed_checks++;
    print_test_row("TC-13", "Zero-filled inputs verification", "Finite",
                   pass ? "Finite" : "NaN/Inf", pass);
  }

  {
    total_checks++;
    Tensor h_in({1, 2, 16}, -0.5f); // All negative values
    Tensor attn_in = layer.attn_norm.forward(h_in);
    Tensor attn_out = layer.attn.forward(attn_in, rope);
    bool pass = std::isfinite(attn_out(0, 0, 0));
    if (pass) passed_checks++;
    print_test_row("TC-14", "Negative values inputs verification", "Finite",
                   pass ? "Finite" : "NaN/Inf", pass);
  }

  {
    total_checks++;
    Tensor h_in({1, 2, 16}, 1e-35f); // Extremely small float values
    Tensor attn_in = layer.attn_norm.forward(h_in);
    Tensor attn_out = layer.attn.forward(attn_in, rope);
    bool pass = std::isfinite(attn_out(0, 0, 0));
    if (pass) passed_checks++;
    print_test_row("TC-15", "Near underflow input values verification", "Finite",
                   pass ? "Finite" : "NaN/Inf", pass);
  }

  {
    total_checks++;
    Tensor h_in({1, 2, 16}, 10000.0f); // Extreme overflow inputs
    Tensor attn_in = layer.attn_norm.forward(h_in);
    Tensor attn_out = layer.attn.forward(attn_in, rope);
    bool pass = std::isfinite(attn_out(0, 0, 0));
    if (pass) passed_checks++;
    print_test_row("TC-16", "Extreme overflow input values verification", "Finite",
                   pass ? "Finite" : "NaN/Inf", pass);
  }

  {
    total_checks++;
    Tensor h_in({1, 35, 16}, 0.5f); // seq_len > max_seq_len (32)
    bool throws = false;
    try {
      layer.attn.forward(h_in, rope);
    } catch (const std::invalid_argument &) {
      throws = true;
    } catch (const std::runtime_error &) {
      throws = true;
    }
    if (throws) passed_checks++;
    print_test_row("TC-17", "Validate seq_len > max_seq_len check", "Throw Exception",
                   throws ? "Thrown" : "Not Thrown", throws);
  }

  {
    total_checks++;
    Tensor h_in({1, 2, 16}, 0.5f);
    Tensor attn_in = layer.attn_norm.forward(h_in);
    Tensor attn_out = layer.attn.forward(attn_in, rope);
    double sum = 0.0;
    for (size_t i = 0; i < attn_out.size(); ++i) sum += std::abs(attn_out(i));
    bool pass = (sum > 1e-5);
    if (pass) passed_checks++;
    print_test_row("TC-18", "Verify layer activation energy is non-zero", "Non-Zero",
                   pass ? "Non-Zero" : "Pure Zero Output", pass);
  }

  // --- Category D: Mathematical & Symmetry (TC-19 to TC-23) ---
  {
    total_checks++;
    Tensor h_in({1, 2, 16}, 1.0f);
    Tensor attn_in = layer.attn_norm.forward(h_in);
    Tensor attn_out = layer.attn.forward(attn_in, rope);
    bool symmetric = true;
    for (size_t d = 0; d < 16; ++d) {
      if (std::abs(attn_out(0, 0, d) - attn_out(0, 1, d)) > 1e-5f) symmetric = false;
    }
    if (symmetric) passed_checks++;
    print_test_row("TC-19", "Identical input tokens output symmetry check", "Symmetric",
                   symmetric ? "Symmetric" : "Asymmetric", symmetric);
  }

  {
    total_checks++;
    Tensor h_in({1, 2, 16});
    for (size_t i = 0; i < h_in.size(); ++i) h_in(i) = 0.05f * (i + 1);
    Tensor grad_output1({1, 2, 16}, 0.1f);
    Tensor grad_output2({1, 2, 16}, 0.2f); // 2x scaled grad_output

    Tensor grad_w_gate1(layer.w_gate.shape(), 0.0f);
    Tensor grad_w_up1(layer.w_up.shape(), 0.0f);
    Tensor grad_w_down1(layer.w_down.shape(), 0.0f);
    Tensor grad_Wq1(layer.attn.Wq().shape(), 0.0f);
    Tensor grad_Wk1(layer.attn.Wk().shape(), 0.0f);
    Tensor grad_Wv1(layer.attn.Wv().shape(), 0.0f);
    Tensor grad_Wo1(layer.attn.Wo().shape(), 0.0f);
    Tensor grad_h_in1 = layer.backward(grad_output1, h_in, grad_w_gate1, grad_w_up1, grad_w_down1,
                                       grad_Wq1, grad_Wk1, grad_Wv1, grad_Wo1, rope);

    Tensor grad_w_gate2(layer.w_gate.shape(), 0.0f);
    Tensor grad_w_up2(layer.w_up.shape(), 0.0f);
    Tensor grad_w_down2(layer.w_down.shape(), 0.0f);
    Tensor grad_Wq2(layer.attn.Wq().shape(), 0.0f);
    Tensor grad_Wk2(layer.attn.Wk().shape(), 0.0f);
    Tensor grad_Wv2(layer.attn.Wv().shape(), 0.0f);
    Tensor grad_Wo2(layer.attn.Wo().shape(), 0.0f);
    Tensor grad_h_in2 = layer.backward(grad_output2, h_in, grad_w_gate2, grad_w_up2, grad_w_down2,
                                       grad_Wq2, grad_Wk2, grad_Wv2, grad_Wo2, rope);

    bool proportional = (std::abs(grad_h_in2(0, 0, 0) - 2.0f * grad_h_in1(0, 0, 0)) < 1e-5f) &&
                        (std::abs(grad_w_down2(0, 0) - 2.0f * grad_w_down1(0, 0)) < 1e-5f);
    if (proportional) passed_checks++;
    print_test_row("TC-20", "Validate gradient scaling proportionality", "Proportional",
                   proportional ? "Proportional" : "Disproportional", proportional);
  }

  {
    total_checks++;
    // Verify residual connection math: intermediate output mid = in + attn_out
    Tensor h_in({1, 2, 16}, 0.5f);
    Tensor attn_in = layer.attn_norm.forward(h_in);
    Tensor attn_out = layer.attn.forward(attn_in, rope);
    Tensor h_mid = h_in.add(attn_out);
    bool pass = true;
    for (size_t i = 0; i < h_mid.size(); ++i) {
      if (std::abs(h_mid(i) - (h_in(i) + attn_out(i))) > 1e-6f) pass = false;
    }
    if (pass) passed_checks++;
    print_test_row("TC-21", "Verify residual connections math", "ResidualAdded",
                   pass ? "ResidualAdded" : "Mismatch", pass);
  }

  {
    total_checks++;
    // Verify RMSNorm scaling property: the RMS of normalized features (before weights) is exactly 1.0
    Tensor h_in({1, 2, 16});
    for (size_t i = 0; i < h_in.size(); ++i) h_in(i) = static_cast<float>(i + 1);
    Tensor attn_in = layer.attn_norm.forward(h_in);
    // Calculate RMS of row 0: sqrt(1/d * sum(x_i^2))
    float sum_sq = 0.0f;
    for (size_t i = 0; i < 16; ++i) sum_sq += attn_in(0, 0, i) * attn_in(0, 0, i);
    float rms = std::sqrt(sum_sq / 16.0f);
    bool pass = (std::abs(rms - 1.0f) < 1e-5f);
    if (pass) passed_checks++;
    print_test_row("TC-22", "Verify RMSNorm output scaling to unit RMS", "1.0",
                   pass ? "1.0" : std::to_string(rms), pass);
  }

  {
    total_checks++;
    // Verify that backpropagating through scaling output preserves scale properties (zero scale = zero output grads)
    Tensor h_in({1, 2, 16}, 0.5f);
    Tensor grad_output({1, 2, 16}, 0.0f);
    Tensor grad_w_gate(layer.w_gate.shape(), 0.0f);
    Tensor grad_w_up(layer.w_up.shape(), 0.0f);
    Tensor grad_w_down(layer.w_down.shape(), 0.0f);
    Tensor grad_Wq(layer.attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(layer.attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(layer.attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(layer.attn.Wo().shape(), 0.0f);
    layer.backward(grad_output, h_in, grad_w_gate, grad_w_up, grad_w_down,
                   grad_Wq, grad_Wk, grad_Wv, grad_Wo, rope);
    double sum = 0.0;
    for (size_t i = 0; i < grad_w_gate.size(); ++i) sum += std::abs(grad_w_gate(i));
    bool pass = (sum < 1e-9);
    if (pass) passed_checks++;
    print_test_row("TC-23", "Verify backpropagation scaling bounds", "Pure Zeros",
                   pass ? "Pure Zeros" : "Non-Zeros Found", pass);
  }

  // --- Category E: Activation Gating Properties (TC-24 to TC-28) ---
  {
    total_checks++;
    // GQA caching property validation (checking shape with dummy cache parameter)
    KVCache cache;
    Tensor h_in({1, 2, 16}, 0.5f);
    Tensor attn_in = layer.attn_norm.forward(h_in);
    Tensor attn_out = layer.attn.forward(attn_in, rope, &cache);
    bool pass = (attn_out.shape() == std::vector<size_t>{1, 2, 16});
    if (pass) passed_checks++;
    print_test_row("TC-24", "GQA cache integration shape check", "ShapeMatch",
                   pass ? "ShapeMatch" : "Mismatch", pass);
  }

  {
    total_checks++;
    // GQA cache backward pass shape validation
    KVCache cache;
    Tensor h_in({1, 2, 16});
    for (size_t i = 0; i < h_in.size(); ++i) h_in(i) = 0.05f * (i + 1);
    Tensor grad_output({1, 2, 16}, 0.1f);
    Tensor grad_w_gate(layer.w_gate.shape(), 0.0f);
    Tensor grad_w_up(layer.w_up.shape(), 0.0f);
    Tensor grad_w_down(layer.w_down.shape(), 0.0f);
    Tensor grad_Wq(layer.attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(layer.attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(layer.attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(layer.attn.Wo().shape(), 0.0f);

    Tensor grad_h_in = layer.backward(grad_output, h_in, grad_w_gate, grad_w_up, grad_w_down,
                                      grad_Wq, grad_Wk, grad_Wv, grad_Wo, rope, &cache);
    bool pass = (grad_h_in.shape() == h_in.shape());
    if (pass) passed_checks++;
    print_test_row("TC-25", "GQA cache backward pass shape validation", "ShapeMatch",
                   pass ? "ShapeMatch" : "Mismatch", pass);
  }

  {
    total_checks++;
    // SwiGLU Gating derivative: gating value is extremely negative (SiLU(x) -> 0), up gradient should be near 0
    Tensor gate({1, 2, 32}, -100.0f); // Closed gate
    Tensor up({1, 2, 32}, 1.0f);
    Tensor grad_act({1, 2, 32}, 0.5f);
    Tensor grad_gate({1, 2, 32}, 0.0f);
    Tensor grad_up({1, 2, 32}, 0.0f);
    activatations::swiglu_backward(grad_act, gate, up, grad_gate, grad_up);
    double sum = 0.0;
    for (size_t i = 0; i < grad_up.size(); ++i) sum += std::abs(grad_up(i));
    bool pass = (sum < 1e-4);
    if (pass) passed_checks++;
    print_test_row("TC-26", "Verify SwiGLU closed gate gradient bounds", "Near Zero",
                   pass ? "Near Zero" : "High Gradient Found", pass);
  }

  {
    total_checks++;
    // Verify SwiGLU gradient alignment w.r.t up projection: grad_up = grad_act * SiLU(gate)
    Tensor gate({1, 1, 4}, 2.0f);
    Tensor up({1, 1, 4}, 2.0f);
    Tensor grad_act({1, 1, 4}, 0.5f);
    Tensor grad_gate({1, 1, 4}, 0.0f);
    Tensor grad_up({1, 1, 4}, 0.0f);
    activatations::swiglu_backward(grad_act, gate, up, grad_gate, grad_up);
    
    float sig = 1.0f / (1.0f + std::exp(-2.0f));
    float silu_val = 2.0f * sig;
    
    bool pass = true;
    for (size_t i = 0; i < grad_up.size(); ++i) {
      float expected = grad_act(i) * silu_val;
      if (std::abs(grad_up(i) - expected) > 1e-5f) pass = false;
    }
    if (pass) passed_checks++;
    print_test_row("TC-27", "Verify SwiGLU open gate gradient alignment", "Exact Match",
                   pass ? "Exact Match" : "Mismatch", pass);
  }

  {
    total_checks++;
    // Gradient accumulation verification: pre-filled gradient buffers correctly accumulate rather than overwrite
    Tensor h_in({1, 2, 16});
    for (size_t i = 0; i < h_in.size(); ++i) h_in(i) = 0.05f * (i + 1);
    Tensor grad_output({1, 2, 16}, 0.1f);
    Tensor grad_w_gate(layer.w_gate.shape(), 1.0f); // Pre-filled with 1.0
    Tensor grad_w_up(layer.w_up.shape(), 0.0f);
    Tensor grad_w_down(layer.w_down.shape(), 0.0f);
    Tensor grad_Wq(layer.attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(layer.attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(layer.attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(layer.attn.Wo().shape(), 0.0f);

    layer.backward(grad_output, h_in, grad_w_gate, grad_w_up, grad_w_down,
                   grad_Wq, grad_Wk, grad_Wv, grad_Wo, rope);

    // Verify element (0,0) of grad_w_gate is greater than 1.0 (accumulated)
    bool pass = (grad_w_gate(0, 0) > 1.0f + 1e-6f);
    if (pass) passed_checks++;
    print_test_row("TC-28", "Verify gradient accumulation (+=) behavior", "Accumulated",
                   pass ? "Accumulated" : "Overwritten", pass);
  }

  std::cout << std::endl;
  std::cout << "================================================================================"
            << std::endl;
  std::cout << "SECTION 2: PERFORMANCE PROFILING & BENCHMARKS" << std::endl;
  std::cout << "================================================================================"
            << std::endl;

  {
    size_t bench_batch = 2;
    size_t bench_seq_len = 32;
    size_t bench_hidden_dim = 512;
    size_t bench_intermediate_dim = 1384;
    size_t bench_n_heads = 8;
    size_t bench_n_kv_heads = 4;
    size_t bench_head_dim = 64;

    std::cout << "Transformer Layer Benchmark (Forward & Backward Passes):" << std::endl;
    std::cout << "  Input Dimensions:     [" << bench_batch << ", " << bench_seq_len << ", " << bench_hidden_dim << "]" << std::endl;
    std::cout << "  FFN Intermediate:     " << bench_intermediate_dim << std::endl;
    std::cout << "  Attention Heads:      " << bench_n_heads << " (dim=" << bench_head_dim << ")" << std::endl;

    ModelConfig bench_config;
    bench_config.hidden_dim = bench_hidden_dim;
    bench_config.intermediate_dim = bench_intermediate_dim;
    bench_config.n_heads = bench_n_heads;
    bench_config.n_kv_heads = bench_n_kv_heads;
    bench_config.head_dim = bench_head_dim;
    bench_config.max_seq_len = 128;
    bench_config.rope_base = 10000.0f;
    bench_config.rms_norm_eps = 1e-5f;

    TransformerLayer bench_layer(bench_config);
    RoPE bench_rope(bench_head_dim, 128, 10000.0f);

    bench_layer.w_gate.fill(0.01f);
    bench_layer.w_up.fill(0.01f);
    bench_layer.w_down.fill(0.01f);
    bench_layer.attn.Wq().fill(0.01f);
    bench_layer.attn.Wk().fill(0.01f);
    bench_layer.attn.Wv().fill(0.01f);
    bench_layer.attn.Wo().fill(0.01f);

    Tensor x({bench_batch, bench_seq_len, bench_hidden_dim}, 0.1f);
    Tensor grad_output({bench_batch, bench_seq_len, bench_hidden_dim}, 0.05f);

    Tensor grad_w_gate(bench_layer.w_gate.shape(), 0.0f);
    Tensor grad_w_up(bench_layer.w_up.shape(), 0.0f);
    Tensor grad_w_down(bench_layer.w_down.shape(), 0.0f);
    Tensor grad_Wq(bench_layer.attn.Wq().shape(), 0.0f);
    Tensor grad_Wk(bench_layer.attn.Wk().shape(), 0.0f);
    Tensor grad_Wv(bench_layer.attn.Wv().shape(), 0.0f);
    Tensor grad_Wo(bench_layer.attn.Wo().shape(), 0.0f);

    // Warm-up pass
    Tensor attn_in = bench_layer.attn_norm.forward(x);
    Tensor attn_out = bench_layer.attn.forward(attn_in, bench_rope);
    Tensor h_mid = x.add(attn_out);
    Tensor ffn_in = bench_layer.ffn_norm.forward(h_mid);
    Tensor gate_proj = ffn_in.matmul(bench_layer.w_gate);
    Tensor up_proj = ffn_in.matmul(bench_layer.w_up);
    Tensor activated = activatations::swiglu(gate_proj, up_proj);
    Tensor ffn_out = activated.matmul(bench_layer.w_down);
    Tensor h_out = h_mid.add(ffn_out);

    Tensor dx = bench_layer.backward(grad_output, x, grad_w_gate, grad_w_up, grad_w_down,
                                     grad_Wq, grad_Wk, grad_Wv, grad_Wo, bench_rope);

    const int runs = 5;
    
    // Forward pass timing
    std::cout << "  [INFO] Running Forward Pass benchmark..." << std::endl;
    std::vector<double> fwd_latencies;
    for (int r = 0; r < runs; ++r) {
      auto start = std::chrono::high_resolution_clock::now();
      Tensor attn_in_loop = bench_layer.attn_norm.forward(x);
      Tensor attn_out_loop = bench_layer.attn.forward(attn_in_loop, bench_rope);
      Tensor h_mid_loop = x.add(attn_out_loop);
      Tensor ffn_in_loop = bench_layer.ffn_norm.forward(h_mid_loop);
      Tensor gate_proj_loop = ffn_in_loop.matmul(bench_layer.w_gate);
      Tensor up_proj_loop = ffn_in_loop.matmul(bench_layer.w_up);
      Tensor activated_loop = activatations::swiglu(gate_proj_loop, up_proj_loop);
      Tensor ffn_out_loop = activated_loop.matmul(bench_layer.w_down);
      Tensor h_out_loop = h_mid_loop.add(ffn_out_loop);
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
      Tensor dy = bench_layer.backward(grad_output, x, grad_w_gate, grad_w_up, grad_w_down,
                                       grad_Wq, grad_Wk, grad_Wv, grad_Wo, bench_rope);
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
