---
type: Architecture
title: Metal GPU Kernel Architecture
description: Custom Metal Shading Language kernels for GEMM, GQA attention, RMSNorm, activations, optimizer, and more — plus the Objective-C++ bridge and heterogeneous CPU/GPU dispatch strategy.
tags: [gpu, metal, kernels, gemm, gqa, apple-silicon, shader]
---

# Metal GPU Kernel Architecture

The engine includes **15 custom Metal Shading Language (MSL) kernels** and a **53KB Objective-C++ bridge** (`MetalBridge.mm`) that manages GPU device initialization, pipeline state compilation, buffer caching, and kernel dispatch.

## Kernel Inventory

All kernels live in [`cpp/src/gpu_kernel/`](../cpp/src/gpu_kernel/):

| Kernel File | Purpose | Key Parameters |
|---|---|---|
| `gemm_ffn.metal` | SwiGLU FFN GEMM (gate, up, down projections) | TILE_M=64, TILE_N=64, TILE_K=32 |
| `gemm_proj.metal` | Q/K/V/O projection GEMM | 64×64 threadgroup tiles |
| `gemm_proj_trans_b.metal` | Projection GEMM with transposed weights | For backwards pass |
| `gemm_gqa.metal` | Fused GQA attention (Flash Attention-style) | seq_len, head_dim, n_heads, n_kv_heads |
| `gemm_backward.metal` | GEMM backward pass (gradient w.r.t. weights) | M, N, K dimensions |
| `rms_norm_forward.metal` | RMSNorm: normalize and scale | num_rows, dims, eps |
| `rms_norm_backward.metal` | RMSNorm backward: dx and dw | num_rows, dims, eps |
| `swiglu_backward.metal` | SwiGLU activation backward | Total elements |
| `rope_backward.metal` | RoPE rotation backward | seq_len, head_dim |
| `gqa_backward.metal` | GQA attention backward | Batch, heads, seq, head_dim |
| `adamw_step.metal` | AdamW optimizer parameter update | Total parameters |
| `cross_entropy.metal` | Cross-entropy loss forward | Batch, seq, vocab |
| `embedding_forward.metal` | Token embedding lookup | vocab_size, hidden_dim |
| `residual_add.metal` | Residual connection: output = x + residual | Total elements |

## The Metal Bridge ([`MetalBridge.mm`](../cpp/src/gpu_kernel/MetalBridge.mm))

The bridge is the single entry point for all GPU operations. Key subsystems:

### Initialization
- `metal_bridge::initialize()` — Creates `MTLDevice`, `MTLCommandQueue`, compiles all 15 pipeline states from `default.metallib`
- Pipeline states use `#include` in the Metal source to share common tile-size defines

### Buffer Cache
- Maps raw `float*` host pointers to `id<MTLBuffer>` GPU buffers via `std::unordered_map`
- **Persistent buffers**: Weight tensors (projection matrices, norm weights) — cached once, never evicted
- **Transient buffers**: Activation/gradient tensors — created per-batch, evicted after `commit_batch()`
- A free pool reuses deallocated buffer memory to avoid re-allocation

### Command Batching
- `begin_batch()` / `commit_batch()` lifecycle:
  - `begin_batch`: Creates `MTLCommandBuffer` and `MTLComputeCommandEncoder`
  - All kernel dispatches in the step enqueue to the same encoder
  - `commit_batch`: Ends the encoder, commits the command buffer, waits for completion, copies results back

### Dispatch Strategy

The bridge implements three dispatch patterns:

1. **Pure GPU GEMM** — For single matrix multiplications large enough to justify GPU launch overhead (e.g., projection layers)
2. **Fused GQA** — The `gemm_gqa.metal` kernel fuses softmax into the attention computation (Flash Attention pattern), never materializing the full `[B, nH, S, S]` attention matrix
3. **Heterogeneous split** — Independent paired matmuls (e.g., FFN gate + up projection) dispatched to GPU and CPU simultaneously:
   - GPU computes `ffn_in @ w_gate` via Metal
   - CPU computes `ffn_in @ w_up` via `cblas_sgemm` (Accelerate)
   - Both run concurrently on the same unified memory; no data copy needed

### Profiling
```cpp
extern double accum_gpu_time_ms;
extern double accum_cpu_time_ms;
extern size_t count_gpu_calls;
extern size_t count_cpu_calls;
```
Cumulative timing is reported in training logs. `reset_profile_stats()` resets counters.

## Tiling Architecture (FFN GEMM Example)

The `gemm_ffn.metal` kernel uses three-level tiling tuned for M3 Ultra's 64KB L1 cache:

```
Level 1 — Threadgroup tile:     64×64 output elements (1 threadgroup per tile)
Level 2 — SIMD group subtile:   32×32 (4 SIMD groups per threadgroup)
Level 3 — simdgroup matrix op:  8×8 (Apple GPU's native matrix multiply unit)
```

A threadgroup loads two 64×32 slices (A and B) into threadgroup memory (16KB each), then iterates through K-dim in steps of 32.

### Tile Configuration (from CMakeLists.txt)
```cmake
set(FFN_TILE_M 64 CACHE STRING "FFN tile M dimension size")
set(FFN_TILE_N 64 CACHE STRING "FFN tile N dimension size")
set(FFN_TILE_K 32 CACHE STRING "FFN tile K dimension size")
```

The projection GEMM (`gemm_proj.metal`) uses 64×64 tiles. The GQA kernel uses whole-head tiles (head_dim × seq_len in threadgroup memory).

## Hot-Patching with fix_metal.py

The `fix_metal.py` script patches `MetalBridge.mm` to add persistent buffer tracking. Run it when the bridge code is regenerated or modified:

```bash
python fix_metal.py
```

It:
1. Replaces the simple `unordered_map` with a `CachedBuffer` struct containing `(id<MTLBuffer> buf, bool is_persistent)`
2. Adds `is_persistent` parameter to `get_or_create_buffer`
3. Modifies `commit_batch` to only evict non-persistent buffers
4. Marks weight buffers (FFN, projection, RMSNorm) as persistent

## GPU Programming Reference

Two comprehensive guides in `cpp/doc/`:

- **[`apple_silicon_gpu_guide.md`](../cpp/doc/apple_silicon_gpu_guide.md)** — Complete Metal GPU programming guide: UMA architecture, MMU page alignment, 7-step GPU pipeline, compilation toolchain
- **[`gpu_tiling_guide.md`](../cpp/doc/gpu_tiling_guide.md)** — Tile size selection methodology: hardware profile (M3 Ultra: 80 cores, 128 ALUs/core, 64KB L1), occupancy calculation, register pressure analysis

## Source Files

| File | Role |
|---|---|
| `cpp/src/gpu_kernel/MetalBridge.mm` | All GPU dispatch, buffer management, profiling |
| `cpp/include/gpu_kernel/MetalBridge.hpp` | Public GPU API declarations |
| `cpp/src/gpu_kernel/*.metal` | 15 Metal Shading Language kernels |
| `default.metallib` | Pre-compiled Metal library (auto-generated at build) |
| `fix_metal.py` | Buffer cache persistence hot-patch |
| `cpp/doc/METAL_KERNEL_ARCHITECTURE.md` | Design rationale and heterogeneous split strategy |
| `cpp/doc/gpu_tiling_guide.md` | Tile sizing methodology |
| `cpp/doc/apple_silicon_gpu_guide.md` | Complete Metal programming bible |
