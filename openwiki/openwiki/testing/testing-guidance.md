---
type: Guide
title: Testing Guidance
description: Test suite structure, target-by-target coverage, how to run tests, and guidance for writing new tests.
tags: [testing, ci, unit-tests, integration-tests]
---

# Testing Guidance

The test suite covers every component from individual tensor operations to the full training loop. All tests link against the `data_ingestion` static library and run as standalone Google Test executables.

## Test Targets

| Test Executable | Lines | What It Tests |
|---|---|---|
| `test_data_ingestion` | 7,967 | Data loading, batch generation from .bin |
| `test_math_ops` | 14,584 | Tensor arithmetic, matmul, indexing, transpose, reshape |
| `test_rmsnorm` | 9,701 | RMSNorm forward/backward with various shapes |
| `test_activations` | 7,367 | SiLU, SwiGLU forward/backward |
| `test_rope` | 11,062 | RoPE table construction, forward application |
| `test_attention` | 24,994 | GQA forward/backward, QKV projection, head grouping |
| `test_transformer` | 32,840 | Full model forward/backward through all layers |
| `test_loss` | 8,838 | Cross-entropy forward/backward |
| `test_backward` | 16,176 | End-to-end gradient propagation |
| `test_trainer` | 28,993 | Training loop, optimizer, LR schedule, checkpoint save/load/resume |
| `test_gpu_kernels` | 13,116 | All Metal bridge kernel calls |

## Running Tests

```bash
# Build all tests
cmake --build build -j$(sysctl -n hw.ncpu)

# Run all tests via CTest
ctest --test-dir build --output-on-failure

# Run a specific test
./build/test_gpu_kernels
./build/test_trainer --gtest_filter="*Forward*"

# Verbose output
./build/test_attention --gtest_print_time=1
```

## Coverage Map

| Component | Forward | Backward | GPU Path | CPU Fallback |
|---|---|---|---|---|
| Tensor operations | ✅ | N/A | ✅ | ✅ |
| RMSNorm | ✅ | ✅ | ✅ | ✅ |
| Activations | ✅ | ✅ | ✅ | ✅ |
| RoPE | ✅ | ✅ (GPU only) | ✅ | ✅ |
| GQA Attention | ✅ | ✅ | ✅ | ❌ (GPU-only) |
| Transformer | ✅ | ✅ | ✅ | ❌ |
| Cross-entropy Loss | ✅ | ✅ | ✅ | ✅ |
| AdamW Optimizer | N/A | N/A | ✅ | ✅ |
| Data Ingestion | ✅ | N/A | N/A | ✅ |
| Training Loop | ✅ | ✅ | ✅ | ✅ |
| GPU Kernels | N/A | N/A | ✅ | N/A |

## Writing New Tests

### Guidelines
1. **Component isolation**: Test each component independently
2. **Shape coverage**: Test at boundaries (`batch=1`, `seq_len=1`, `head_dim=min/max`)
3. **Numerical verification**: Use `EXPECT_FLOAT_EQ` / `EXPECT_NEAR`
4. **Device path**: Test both CPU and GPU paths if a component supports both
5. **Backward pass**: Verify with numerical gradient checking for new differentiable ops
6. **No external data**: Use synthetic tensors, never depend on `.bin` files

### Adding a New Test Target

In `CMakeLists.txt`:
```cmake
add_executable(test_my_component
    cpp/tests/test_my_component.cpp)
target_link_libraries(test_my_component PRIVATE
    data_ingestion gtest gtest_main)
add_test(NAME test_my_component COMMAND test_my_component)
```

## Source Files

- `CMakeLists.txt` (lines 71–148) — Test target definitions
- `cpp/tests/` — All test source files
- `testing/testing-guidance.md` — Root testing guidance (authoritative)

For the complete test structure and all guidelines, see the root [Testing Guidance](/testing/testing-guidance.md).
