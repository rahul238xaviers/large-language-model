---
type: Architecture
title: 1.6B GPT Model Architecture
description: Detailed architecture of the GPT-style transformer model with GQA attention, SwiGLU FFN, RoPE, and RMSNorm — covering configuration, forward pass, and backward pass.
tags: [model, transformer, gpt, gqa, swiglu, rope, rmsnorm]
---

# 1.6B GPT Model Architecture

The model is a **decoder-only GPT-style transformer** inspired by LLaMA-2 and Mistral, with 1.6 billion parameters.

## Configuration

Defined in [`TransformerConfig.hpp`](../cpp/include/TransformerConfig.hpp) via `ModelConfig::make_default()`:

| Parameter | Value | Notes |
|---|---|---|
| `vocab_size` | 100,256 | cl100k_base tokenizer (tiktoken) |
| `hidden_dim` | 2,048 | Embedding/hidden size (d_model) |
| `intermediate_dim` | 5,461 | SwiGLU FFN expansion dimension |
| `n_layers` | 24 | Transformer blocks |
| `n_heads` | 16 | Query heads (per layer) |
| `n_kv_heads` | 8 | Key/Value heads (GQA grouping) |
| `head_dim` | 128 | hidden_dim / n_heads |
| `max_seq_len` | 2,048 | Max context window |
| `rope_base` | 10,000.0 | RoPE base frequency |
| `rms_norm_eps` | 1e-6 | Norm stability epsilon |

**Parameter count**: ~1.6B (computed from the shape composition of all weight tensors).

## Component Composition

The model is built from the following components, each with its own header and source:

### Tensor ([`Tensor.hpp`](../cpp/include/Tensor.hpp), [`Tensor.cpp`](../cpp/src/Tensor.cpp))
- Multi-dimensional `float` array with shape `[d1, d2, ...]` and computed strides
- Row-major storage in a flat `std::vector<float>`
- Supports: add, mul, matmul (Accelerate + GPU dispatch via Tensor.cpp), transpose (vDSP_mtrans), reshape
- 1D/2D/3D/4D indexing overloads

### RMSNorm ([`RMSNorm.hpp`](../cpp/include/RMSNorm.hpp), [`RMSNorm.cpp`](../cpp/src/RMSNorm.cpp))
- **Forward**: `x / sqrt(mean(x²) + eps) * weight`
- **Backward**: GPU kernel (`rms_norm_backward.metal`) for dx and dw
- CPU fallback for small sizes

### Rotary Position Embeddings ([`Positional.hpp`](../cpp/include/Positional.hpp), [`Positional.cpp`](../cpp/src/Positional.cpp))
- Precomputes `[max_seq_len, head_dim]` sin/cos table
- Applies rotation to Q and K tensors: rotate pairs `(x_2i, x_{2i+1})` by angle `m * theta_i`
- GPU backward kernel (`rope_backward.metal`)

### GQA Attention ([`Attention.hpp`](../cpp/include/Attention.hpp), [`Attention.cpp`](../cpp/src/Attention.cpp))
- **Grouped Query Attention**: 16 Q heads, 8 KV heads (each KV head shared by 2 Q heads)
- Forward: Q/K/V projections → RoPE → GQA scores (GPU kernel `gemm_gqa.metal`) → Output projection
- Backward: gradients for Wq, Wk, Wv, Wo; GPU kernels for GQA backward, GEMM backward
- No causal mask in current GPU implementation (TODO)

### SwiGLU FFN (in [`Transformer.cpp`](../cpp/src/Transformer.cpp))
- `gate = silu(x @ W_gate)`, `up = x @ W_up`, `output = (gate * up) @ W_down`
- Gate and up projections can be dispatched **simultaneously** (GPU + CPU) via the heterogeneous operation split
- GPU backward kernel (`swiglu_backward.metal`)

### Cross-Entropy Loss ([`Loss.hpp`](../cpp/include/Loss.hpp), [`Loss.cpp`](../cpp/src/Loss.cpp))
- Forward: softmax(logits) → cross-entropy with target tokens
- Backward: `grad_logits = probs - targets` (GPU kernel: `cross_entropy.metal`)
- Supports both CPU and GPU paths

## Forward Pass Sequence

For one batch of tokens `[batch_size, seq_len]`:

```
1. token_embeddings[tokens]           → [B, S, D]   (embedding lookup)
2. For each of 24 layers:
   a. h = rms_norm(h)                  → RMSNorm forward
   b. attn_out = attention(h)          → QKV proj, RoPE, GQA scores, Wo proj
   c. h = h + attn_out                 → residual
   d. h = rms_norm(h)                  → RMSNorm forward
   e. ffn_out = swiglu_ffn(h)          → gate/up proj, SwiGLU, down proj
   f. h = h + ffn_out                  → residual
3. h = rms_norm(h)                     → final RMSNorm
4. logits = h @ output_projection      → [B, S, vocab_size]
```

## Backward Pass Sequence

Reverses the forward pass, computing gradients for all parameters:

```
1. grad_logits = cross_entropy_backward(targets)
2. grad_output_projection = h.T @ grad_logits
   grad_h = grad_logits @ output_projection.T
3. For each of 24 layers (in reverse):
   a. grad_ffn_out = grad_h (residual branch)
   b. grad_h_ffn = rms_norm_backward(grad_ffn_out)  → grad_ffn_norm
   c. SwiGLU backward: grad_gate, grad_up, grad_x
   d. grad_w_down = x_prev.T @ grad_swiglu
   e. grad_attn_out = grad_h (residual branch)
   f. grad_h_attn = rms_norm_backward(grad_attn_out) → grad_attn_norm
   g. Attention backward: grad_Wq, grad_Wk, grad_Wv, grad_Wo
4. grad_token_embeddings[token_ids] = grad_h_final
```

## Files

| File | Role |
|---|---|
| `cpp/include/TransformerConfig.hpp` | `ModelConfig` struct and `make_default()` |
| `cpp/include/Transformer.hpp` | `TransformerLayer` and `Transformer` declarations |
| `cpp/src/Transformer.cpp` | Forward and backward pass orchestration |
| `cpp/include/Attention.hpp` | `Attention` declarations |
| `cpp/src/Attention.cpp` | GQA attention forward/backward, QKV projection dispatch |
| `cpp/include/RMSNorm.hpp` | `RMSNorm` declarations |
| `cpp/src/RMSNorm.cpp` | Norm forward/backward with GPU dispatch |
| `cpp/include/Positional.hpp` | `RoPE` declarations |
| `cpp/src/Positional.cpp` | RoPE table build and forward application |
| `cpp/include/Activations.hpp` | SiLU/SwiGLU declarations |
| `cpp/src/Activations.cpp` | SiLU forward/backward, SwiGLU forward/backward |
| `cpp/include/Loss.hpp` | `CrossEntropyLoss` declarations |
| `cpp/src/Loss.cpp` | Loss forward/backward (CPU + GPU paths) |

## GPU Kernel Mapping

Each model operation that has a GPU implementation maps to a Metal kernel:

| Operation | Metal Kernel | Bridge Function |
|---|---|---|
| Q/K/V projection | `gemm_proj.metal` | `metal_bridge::gemm_proj()` |
| GQA attention | `gemm_gqa.metal` | `metal_bridge::gemm_gqa()` |
| FFN gate/up/down | `gemm_ffn.metal` | `metal_bridge::gemm_ffn()` |
| RMSNorm forward | `rms_norm_forward.metal` | `metal_bridge::rms_norm_forward()` |
| RMSNorm backward | `rms_norm_backward.metal` | `metal_bridge::rms_norm_backward()` |
| SwiGLU backward | `swiglu_backward.metal` | `metal_bridge::swiglu_backward()` |
| RoPE backward | `rope_backward.metal` | `metal_bridge::rope_backward()` |
| GQA backward | `gqa_backward.metal` | `metal_bridge::gqa_backward()` |
| AdamW step | `adamw_step.metal` | `metal_bridge::adamw_step()` |
| Embedding forward | `embedding_forward.metal` | `metal_bridge::embedding_forward()` |
| Cross-entropy | `cross_entropy.metal` | `metal_bridge::cross_entropy_forward()` |

See [GPU Metal Kernels](gpu-kernels.md) for detailed kernel architecture.
