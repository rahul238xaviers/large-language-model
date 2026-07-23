# C4 Planning: Kernel Audit & Architecture

## Kernel Inventory — Usage & Replacement Status

### ACTIVE — Called from training loop (hot path)

| Kernel | Calls/step | Called from | Purpose |
|--------|-----------|-------------|---------|
| `flash_attn_fwd` | 24 | `Attention::forward()` (GPU path) | Fused QK^T→softmax→PV (replaces `gemm_gqa` in forward) |
| `gemm_gqa` | 24 | `Attention::backward()` recomputation (GPU) | Still used in BACKWARD recomputation for attention forward |
| `gemm_bf16` | ~168 | `Tensor::matmul()` | All GEMMs: QKV, gate/up, down, output projection |
| `fused_attn_bwd` | 24 | `Attention::backward()` | Single-kernel GQA attention backward (replaces `gqa_scores`+`attn_ds`+`gqa_backward`) |
| `fused_add_norm` | 24 | `Transformer::forward()` | Fused residual_add + rms_norm_forward (replaces both) |
| `fused_backward_add_norm` | 48 | `TransformerLayer::backward()` | Fused rms_norm_backward + residual_add (replaces both) |
| `rms_norm_forward` | 25 | `Transformer::forward()` (final norm) + backward recomputation | Standalone RMSNorm forward (final norm + cache path) |
| `rms_norm_backward` | 0 (hot) | Called only from CPU fallback | Replaced by `fused_backward_add_norm` in GPU path |
| `residual_add` | 75 | `Transformer.cpp`, `Attention.cpp`, `Tensor::add_()` | In-place addition (small tensors via `add_()`, large residual connections) |
| `swiglu_forward` | 24 | `Transformer::forward()` | SwiGLU activation |
| `swiglu_backward` | 24 | `TransformerLayer::backward()` | SwiGLU gradient (recompute activated, grad_gate, grad_up) |
| `embedding_forward` | 1 | `Transformer::forward()` | Token embedding lookup |
| `cross_entropy` | 1 | `CrossEntropyLoss::forward()` | Fused softmax + cross-entropy loss + gradient |
| `adamw_step` | ~170 | `AdamWOptimizer::step()` | Fused AdamW parameter update |
| `gemm_backward` | ~48 | `Transformer.cpp`, `Attention.cpp` | Weight gradient accumulation (dL/dW = A^T @ dL/dY) |
| `gemm_proj_trans_b` | ~48 | `Transformer.cpp`, `Attention.cpp` | Input gradient propagation (dL/dX = dL/dY @ W^T) |
| `rope_forward` | 24 | `Attention::forward()` | Rotary Position Embedding |
| `rope_backward` | 24 | `Attention::backward()` | RoPE gradient |
| `reshape_to_4d` | 6 | `Attention::forward()`, `Attention::backward()` | 3D→4D view for head-dim split |
| `reshape_to_3d` | 6 | `Attention::forward()`, `Attention::backward()` | 4D→3D view after attention |

### UNUSED — Called only from tests/benchmarks

| Kernel | Why unused | Replacement |
|--------|-----------|-------------|
| `gemm_ffn` | Legacy fused FFN kernel; training uses `gemm_bf16` + `swiglu_forward` separately | `gemm_bf16` |
| `gemm_proj` | Legacy AMX cblas_sgemm wrapper; training uses `gemm_bf16` | `gemm_bf16` |
| `gqa_scores` | Standalone QK^T score kernel; training uses `flash_attn_fwd` or `fused_attn_bwd` | `flash_attn_fwd`, `fused_attn_bwd` |
| `attn_softmax` | Standalone softmax kernel | `flash_attn_fwd` (fused) |
| `attn_ds` | Standalone attention dS kernel | `fused_attn_bwd` (fused) |
| `gqa_backward` | Multi-kernel GQA backward | `fused_attn_bwd` (single kernel) |
| `gemm_bf16_math_only` | Diagnostic roofline isolate (math throughput) | N/A — diagnostics only |
| `gemm_bf16_mem_only` | Diagnostic roofline isolate (memory bandwidth) | N/A — diagnostics only |
| `output_proj.metal` | Placeholder/legacy — not wired | `gemm_bf16` |

### CONDITIONAL — Called from CPU fallback path only

| Kernel | When called |
|--------|------------|
| `rms_norm_backward` | When `use_gpu == false` or `is_available()` returns false |
| `gqa_backward_query_token` (CPU function) | When `use_gpu == false` |

## Replacement History

```
FORWARD ATTENTION PIPELINE (BEFORE → AFTER):
  gqa_scores → QK^T scores     \
  attn_softmax → softmax        | → flash_attn_fwd (single fused kernel, online softmax)
  gemm_gqa → P × V             /
  
BACKWARD ATTENTION PIPELINE (BEFORE → AFTER):
  gqa_scores → recompute S     \
  attn_ds → dS                  | → fused_attn_bwd (single kernel, BF16 tiled)
  gqa_backward → dQ,dK,dV      /

FORWARD NORM PIPELINE (BEFORE → AFTER):
  residual_add → h += attn_out  \
  rms_norm_forward → ffn_in     | → fused_add_norm (single kernel)
  → repeated at FFN output      /

BACKWARD NORM PIPELINE (BEFORE → AFTER):
  rms_norm_backward → grad_h    \
  add_() → grad_h += grad_out   | → fused_backward_add_norm (single kernel)
  → repeated at attention norm  /

GEMM PIPELINE (BEFORE → AFTER):
  gemm_proj → cblas_sgemm AMX   → gemm_bf16 (Metal, BF16 DRAM, FP32 accum)
  gemm_ffn → fused gate+up GEMM → gemm_bf16 (two separate calls) + swiglu_forward
```

## Data Types

| Storage | Type | Bytes/element |
|---------|------|--------------|
| Weights (all) | BF16 | 2 |
| Activations (forward) | BF16 | 2 |
| Gradients (backward) | BF16 | 2 |
| Optimizer m/v states | FP32 | 4 |
| Gradient accumulators | FP32 | 4 |
| Loss scalar | FP32 | 4 |

## Pipeline State Metrics (from `GPU_PROFILE=1`)

| Kernel | tgMem | maxThreads | Occupancy limit |
|--------|-------|-----------|----------------|
| `gemm_bf16` | 32 KB | 1024 | Shared memory (32/64 KB) |
| `fused_attn_bwd` | 24.4 KB | 1024 | Shared memory |
| `flash_attn_fwd` | 20 KB | 1024 | Shared memory |
| `fused_backward_add_norm` | 32 B | 1024 | Register |
| All element-wise | 0-48 B | 1024 | Register |

## Next Optimizations (C4 targets)

1. **Kernel fusion**: Combine `gemm_bf16` + `residual_add` gradients at FFN output
2. **Tile resizing**: Reduce `gemm_bf16` BM/BN from 128→64 to double occupancy
3. **Forward norm fusion**: Replace remaining `rms_norm_forward` with fused variants
4. **Remove dead kernels**: Delete unused `.metal` files and bridge functions
