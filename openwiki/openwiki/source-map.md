---
type: Reference
title: Source Map
description: Compact file-by-file directory map for the LLM Training Engine repository, organized by subsystem.
tags: [reference, source-map, navigation]
---

# Source Map

## Root

| Path | Role |
|---|---|
| `CMakeLists.txt` | Build system: defines `data_ingestion` library, all test executables, `run_trainer` binary, Metal metallib compilation |
| `pretokenize.py` | Reads Parquet → tokenizes with tiktoken cl100k_base → writes flat uint32 `.bin` |
| `fix_metal.py` | Hot-patches `MetalBridge.mm` for persistent buffer cache tracking |
| `compile_commands.json` | Clangd/LSP compilation database (auto-generated) |

## `cpp/include/` — Public Headers

| File | Key Types | Role |
|---|---|---|
| `TransformerConfig.hpp` | `ModelConfig`, `KVCache` | Model hyperparameters (vocab_size, hidden_dim, n_layers, n_heads, head_dim, ...) |
| `Tensor.hpp` | `Tensor` | Multi-dimensional float array, shape/strides, matmul (Accelerate + GPU dispatch), transpose, element-wise ops |
| `Transformer.hpp` | `Transformer`, `TransformerLayer` | Owns token_embeddings, layers, output_projection, final_norm, rope |
| `Attention.hpp` | `Attention` | GQA attention: Wq, Wk, Wv, Wo weights; forward/backward |
| `RMSNorm.hpp` | `RMSNorm` | Root Mean Square Normalization |
| `Positional.hpp` | `RoPE` | Rotary Position Embeddings — precomputed sin/cos table |
| `Activations.hpp` | — | `silu` (SiLU/Swish), `swiglu` |
| `Loss.hpp` | `CrossEntropyLoss` | Cross-entropy loss forward and backward |
| `Trainer.hpp` | `Trainer`, `TrainerConfig` | Training loop orchestrator: max_steps, warmup_steps, lr schedule, checkpointing |
| `Optimizer.hpp` | `Optimizer`, `SGDOptimizer`, `AdamWOptimizer` | Optimization algorithms |
| `DataIngestion.hpp` | `DataIngestion` | Reads pretokenized .bin, yields batches of token IDs |
| `Checkpoint.hpp` | — | Safetensors save/load |
| `gpu_kernel/MetalBridge.hpp` | `metal_bridge::` namespace | `initialize()`, `gemm_ffn()`, `gemm_proj()`, `gemm_gqa()`, RMS norm, backward kernels, AdamW step |

## `cpp/src/` — Implementation Files

| File | Lines | Role |
|---|---|---|
| `Tensor.cpp` | 16,378 | Tensor storage, indexing, arithmetic, matmul dispatch, transpose, reshape |
| `Transformer.cpp` | 23,204 | Full model forward/backward, residual connections |
| `Trainer.cpp` | 16,579 | Training loop: batch fetch, forward/loss/backward, optimizer step, logging, LR schedule |
| `RunTrainer.cpp` | 11,468 | CLI entry point: config parsing, weight init, checkpoint resume |
| `Checkpoint.cpp` | 39,888 | Safetensors read/write, weight fusion for MLX compatibility |
| `Attention.cpp` | 20,684 | GQA attention forward (GPU) and backward (gradients for Q/K/V/O) |
| `RMSNorm.cpp` | 7,388 | RMSNorm forward/backward (GPU + CPU paths) |
| `Positional.cpp` | 6,719 | RoPE table precomputation and forward application |
| `Activations.cpp` | 5,718 | SiLU, SwiGLU forward/backward |
| `Optimizer.cpp` | 6,593 | SGD and AdamW optimizer step implementations |
| `Loss.cpp` | 6,672 | Cross-entropy loss forward/backward |
| `DataIngestion.cpp` | 2,934 | Parquet reading and .bin file token ingestion |

All source files live in `cpp/`. See the [root source map](/source-map.md) for the full canonical version.

## `cpp/src/gpu_kernel/` — Metal GPU Kernels

| File | Lines | Role |
|---|---|---|
| `MetalBridge.mm` | ~53KB | C++ ↔ Metal bridge: device init, buffer cache, pipeline states, all kernel dispatch |
| `gemm_ffn.metal` | — | Tiled GEMM for SwiGLU FFN (64x64 tiles) |
| `gemm_proj.metal` | — | Tiled GEMM for Q/K/V/O projections |
| `gemm_proj_trans_b.metal` | — | Projection GEMM with pre-transposed weights |
| `gemm_gqa.metal` | — | Fused GQA attention (Flash Attention-style) |
| `gemm_backward.metal` | — | GEMM backward pass |
| `rms_norm_forward.metal` | — | RMSNorm forward |
| `rms_norm_backward.metal` | — | RMSNorm backward (dx and dw) |
| `swiglu_backward.metal` | — | SwiGLU activation backward |
| `rope_backward.metal` | — | RoPE backward |
| `gqa_backward.metal` | — | GQA attention backward |
| `adamw_step.metal` | — | AdamW optimizer step on GPU |
| `cross_entropy.metal` | — | Cross-entropy loss forward on GPU |
| `embedding_forward.metal` | — | Token embedding lookup on GPU |
| `residual_add.metal` | — | Residual addition |

## `cpp/tests/` — Test Executables

| Test Executable | Lines | What It Tests |
|---|---|---|
| `test_data_ingestion.cpp` | 7,967 | Data loading, batch generation |
| `test_math_ops.cpp` | 14,584 | Tensor arithmetic, matmul, indexing, transpose |
| `test_rmsnorm.cpp` | 9,701 | RMSNorm forward/backward |
| `test_activations.cpp` | 7,367 | SiLU, SwiGLU forward/backward |
| `test_rope.cpp` | 11,062 | RoPE table and forward application |
| `test_attention.cpp` | 24,994 | GQA forward/backward, QKV projection |
| `test_transformer.cpp` | 32,840 | Full model forward/backward |
| `test_loss.cpp` | 8,838 | Cross-entropy forward/backward |
| `test_backward.cpp` | 16,176 | End-to-end gradient propagation |
| `test_trainer.cpp` | 28,993 | Training loop, optimizer, checkpoint |
| `test_gpu_kernels.cpp` | 13,116 | All Metal bridge kernel calls |

## `cpp/doc/` — C++ Documentation

| Document | Content |
|---|---|
| `architecture.md` | Original MLX-era architecture (partially outdated) |
| `METAL_KERNEL_ARCHITECTURE.md` | GPU kernel dispatch strategy, heterogeneous operation split |
| `apple_silicon_gpu_guide.md` | Comprehensive Metal GPU programming guide |

## `python/` — Python Pipeline

See the [root source map](/source-map.md#python--python-pipeline) for the Python directory structure.

For the authoritative version, see [Root Source Map](/source-map.md).
