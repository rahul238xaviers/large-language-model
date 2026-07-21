---
type: Architecture
title: Metal GPU Kernel Architecture
description: Custom Metal Shading Language kernels for GEMM, GQA attention, RMSNorm, activations, optimizer, and the Objective-C++ bridge with heterogeneous CPU/GPU dispatch.
tags: [gpu, metal, kernels, gemm, gqa, apple-silicon, shader]
---

# Metal GPU Kernel Architecture

The engine includes **15 custom Metal Shading Language (MSL) kernels** and a ~53KB Objective-C++ bridge (`MetalBridge.mm`) that manages GPU device initialization, pipeline state compilation, buffer caching, and kernel dispatch.

## Kernel Inventory

All kernels live in `cpp/src/gpu_kernel/`:

| Kernel File | Purpose |
|---|---|
| `gemm_ffn.metal` | SwiGLU FFN GEMM (gate/up/down projections) — 64x64 tiles |
| `gemm_proj.metal` | Q/K/V/O projection GEMM |
| `gemm_proj_trans_b.metal` | Projection GEMM with transposed weights (backward pass) |
| `gemm_gqa.metal` | Fused GQA attention (Flash Attention-style, no materialized SxS matrix) |
| `gemm_backward.metal` | GEMM backward (gradient w.r.t. weights) |
| `rms_norm_forward.metal` | RMSNorm forward |
| `rms_norm_backward.metal` | RMSNorm backward (dx and dw) |
| `swiglu_backward.metal` | SwiGLU activation backward |
| `rope_backward.metal` | RoPE backward |
| `gqa_backward.metal` | GQA attention backward |
| `adamw_step.metal` | AdamW optimizer step on GPU |
| `cross_entropy.metal` | Cross-entropy loss forward on GPU |
| `embedding_forward.metal` | Token embedding lookup on GPU |
| `residual_add.metal` | Residual addition |

## The Metal Bridge (`MetalBridge.mm`)

The bridge is the single entry point for all GPU operations:

- **Initialization**: Creates `MTLDevice`, `MTLCommandQueue`, compiles all 15 pipeline states from `default.metallib`
- **Buffer Cache**: Maps raw `float*` → `id<MTLBuffer>` via `std::unordered_map`; weight tensors are persistent, activations/gradients are transient with a free pool
- **Command Batching**: `begin_batch()` / `commit_batch()` lifecycle — all kernel dispatches in a training step share one encoder
- **Dispatch Strategies**:
  1. **Pure GPU GEMM** — Large single matmuls (projection layers)
  2. **Fused GQA** — `gemm_gqa.metal` fuses softmax (Flash Attention pattern)
  3. **Heterogeneous split** — Independent paired matmuls (FFN gate + up) dispatched to GPU + CPU concurrently

## Tiling Architecture (FFN GEMM Example)

`gemm_ffn.metal` uses three-level tiling tuned for M3 Ultra's 64KB L1 cache:

```
Level 1 — Threadgroup tile:     64x64 output elements
Level 2 — SIMD group subtile:   32x32
Level 3 — simdgroup matrix op:  8x8 (Apple GPU native matmul)
```

Tile sizes are CMake-configurable:
```cmake
set(FFN_TILE_M 64)   # FFN tile M dimension
set(FFN_TILE_N 64)   # FFN tile N dimension
set(FFN_TILE_K 32)   # FFN tile K dimension
```

## Profiling

```cpp
extern double accum_gpu_time_ms;
extern double accum_cpu_time_ms;
extern size_t count_gpu_calls;
extern size_t count_cpu_calls;
```

Cumulative timing reported in training logs. `reset_profile_stats()` resets counters.

## Hot-Patching with fix_metal.py

After building (or after regenerating `MetalBridge.mm`), apply the buffer cache persistence patch:

```bash
python fix_metal.py
cmake --build build -j$(sysctl -n hw.ncpu)  # Rebuild
```

This modifies `MetalBridge.mm` to add persistent buffer tracking.

## Source Files

- `cpp/src/gpu_kernel/MetalBridge.mm` — Objective-C++ bridge (~53KB)
- `cpp/src/gpu_kernel/*.metal` — 15 MSL kernel files
- `fix_metal.py` — Post-build hot-patch script
- `cpp/doc/METAL_KERNEL_ARCHITECTURE.md` — Original kernel design rationale

For detailed tile configuration, kernel parameters, and the full bridge API, see the root [GPU Kernels](/domain/gpu-kernels.md).
