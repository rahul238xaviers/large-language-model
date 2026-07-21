---
type: Guide
title: Testing Guidance
description: Test suite structure, target-by-target coverage, how to run tests, and guidance for writing new tests for the C++ LLM training engine.
tags: [testing, ci, unit-tests, integration-tests]
---

# Testing Guidance

The test suite covers every component from individual tensor operations to the full training loop. All tests link against the `data_ingestion` static library and run as standalone executables.

## Test Targets

Configured in [`CMakeLists.txt`](../CMakeLists.txt) (lines 71–148). Each test is a separate executable:

| Test Executable | Source | Lines | What It Tests |
|---|---|---|---|
| `test_data_ingestion` | `test_data_ingestion.cpp` | 7,967 | Data loading, batch generation from .bin, token count verification |
| `test_math_ops` | `test_math_ops.cpp` | 14,584 | Tensor arithmetic (add, mul, scale), matmul (2D + batched), indexing, transpose, reshape |
| `test_rmsnorm` | `test_rmsnorm.cpp` | 9,701 | RMSNorm forward with various shapes, backward gradient correctness |
| `test_activations` | `test_activations.cpp` | 7,367 | SiLU forward, SwiGLU forward, activation backward passes |
| `test_rope` | `test_rope.cpp` | 11,062 | RoPE table construction, forward application, dimension handling |
| `test_attention` | `test_attention.cpp` | 24,994 | GQA forward and backward, QKV projection correctness, head grouping |
| `test_transformer` | `test_transformer.cpp` | 32,840 | Full model forward (logits shape), backward (gradient propagation through all layers) |
| `test_loss` | `test_loss.cpp` | 8,838 | Cross-entropy forward (loss value), backward (grad_logits) |
| `test_backward` | `test_backward.cpp` | 16,176 | End-to-end gradient propagation from loss through all layers |
| `test_trainer` | `test_trainer.cpp` | 28,993 | Training loop, optimizer step, LR schedule, checkpoint save/load/resume, batch iteration |
| `test_gpu_kernels` | `test_gpu_kernels.cpp` | 13,116 | All Metal bridge kernel calls: GEMM, GQA, RMSNorm, SwiGLU backward, RoPE backward, AdamW step |

## Running Tests

```bash
# Build all tests
cmake --build build -j$(sysctl -n hw.ncpu)

# Run all tests via CTest
ctest --test-dir build --output-on-failure

# Run a specific test
./build/test_gpu_kernels
./build/test_trainer --gtest_filter="*Forward*"

# Run with verbose output
./build/test_attention --gtest_print_time=1
```

## Test Structure

Tests use **Google Test** (gtest). The pattern:

```cpp
TEST(TestSuiteName, TestName) {
  // 1. Setup: create tensors, configs, model components
  ModelConfig config = ModelConfig::make_default();
  Transformer model(config);

  // 2. Execute: run the operation
  Tensor tokens = /* ... */;
  Tensor logits = model.forward(tokens);

  // 3. Verify: check shapes and values
  EXPECT_EQ(logits.shape()[0], batch_size);
  EXPECT_EQ(logits.shape()[1], seq_len);
  EXPECT_EQ(logits.shape()[2], vocab_size);
  EXPECT_FLOAT_EQ(loss_value, expected_value);
}
```

## Coverage Map

| Component | Forward | Backward | GPU Path | CPU Fallback |
|---|---|---|---|---|
| Tensor operations | ✅ `test_math_ops` | N/A | ✅ `test_gpu_kernels` | ✅ (implicit) |
| RMSNorm | ✅ `test_rmsnorm` | ✅ `test_rmsnorm` | ✅ `test_gpu_kernels` | ✅ (implicit) |
| Activations | ✅ `test_activations` | ✅ `test_activations` | ✅ `test_gpu_kernels` | ✅ (implicit) |
| RoPE | ✅ `test_rope` | ✅ (GPU only) | ✅ `test_gpu_kernels` | ✅ (implicit) |
| GQA Attention | ✅ `test_attention` | ✅ `test_attention` | ✅ `test_gpu_kernels` | ❌ (GPU-only) |
| Transformer | ✅ `test_transformer` | ✅ `test_transformer` | ✅ (GPU dispatch) | ❌ |
| Cross-entropy Loss | ✅ `test_loss` | ✅ `test_loss` | ✅ `test_gpu_kernels` | ✅ `test_loss` |
| AdamW Optimizer | N/A | N/A | ✅ `test_gpu_kernels` | ✅ `test_trainer` |
| Data Ingestion | ✅ `test_data_ingestion` | N/A | N/A | ✅ |
| Training Loop | ✅ `test_trainer` | ✅ `test_trainer` | ✅ `test_trainer` | ✅ |
| GPU Kernels | N/A | N/A | ✅ `test_gpu_kernels` | N/A |

## Writing New Tests

### Guidelines
1. **Component isolation**: Test each component independently. A RMSNorm test should not depend on the full Transformer.
2. **Shape coverage**: Test at boundary conditions: `batch=1`, `seq_len=1`, `head_dim=min/max`, edge values at vocab boundaries
3. **Numerical verification**: Use `EXPECT_FLOAT_EQ` / `EXPECT_NEAR` for float comparisons
4. **Device path**: Always test both CPU and GPU paths if a component supports both
5. **Backward pass**: Verify with numerical gradient checking when adding new differentiable operations
6. **No external data**: Tests should not depend on `.bin` files or external data — use synthetic tensors

### Adding a New Test

```cmake
# In CMakeLists.txt (add after existing test targets):
add_executable(test_my_component
    cpp/tests/test_my_component.cpp
)
target_link_libraries(test_my_component PUBLIC
    data_ingestion
)
```

### Key Test Files to Reference

- `test_gpu_kernels.cpp` — Best template for GPU kernel tests (device init, input setup, kernel launch, result verification)
- `test_math_ops.cpp` — Best template for Tensor operation tests (shape generation, operation, shape/value assertions)
- `test_trainer.cpp` — Best template for integration tests (model + optimizer + data + checkpoint cycle)

## GPU Test Notes

GPU tests in `test_gpu_kernels.cpp`:
- Initialize Metal device in `SetUp()` via `metal_bridge::initialize()`
- Use `metal_bridge::is_available()` to skip if no GPU
- Verify with known-reference computations on CPU
- Clean up with `metal_bridge::reset_profile_stats()` between tests
- Ensure host buffers are 16KB-aligned for UMA zero-copy mapping

## Git History Insights

Key testing-evolution commits from git history:

- `241689c` — Transitioned test suite from model-level to layer-level verification
- `3d726cf` — Removed debug print statements from GPU kernel tests (production hardening)
- `7ced276` — Added input dimension validation tests for Attention forward/backward + expanded GQA test suite
- `5e448f0` — Comprehensive unit tests for Tensor construction, indexing, arithmetic, and performance

## Source Files

| File | Lines | Role |
|---|---|---|
| `cpp/tests/test_data_ingestion.cpp` | 7,967 | Data pipeline tests |
| `cpp/tests/test_math_ops.cpp` | 14,584 | Tensor operation tests |
| `cpp/tests/test_rmsnorm.cpp` | 9,701 | RMSNorm tests |
| `cpp/tests/test_activations.cpp` | 7,367 | Activation function tests |
| `cpp/tests/test_rope.cpp` | 11,062 | RoPE tests |
| `cpp/tests/test_attention.cpp` | 24,994 | GQA attention tests |
| `cpp/tests/test_transformer.cpp` | 32,840 | Full model tests |
| `cpp/tests/test_loss.cpp` | 8,838 | Loss function tests |
| `cpp/tests/test_backward.cpp` | 16,176 | End-to-end backward tests |
| `cpp/tests/test_trainer.cpp` | 28,993 | Training loop integration tests |
| `cpp/tests/test_gpu_kernels.cpp` | 13,116 | GPU kernel unit tests |
| `CMakeLists.txt` (lines 71–168) | — | Test target definitions |
