---
type: Architecture
title: 1.6B GPT Model Architecture
description: Detailed architecture of the GPT-style transformer model with GQA attention, SwiGLU FFN, RoPE, and RMSNorm.
tags: [model, transformer, gpt, gqa, swiglu, rope, rmsnorm]
---

# 1.6B GPT Model Architecture

A **decoder-only GPT-style transformer** inspired by LLaMA-2 and Mistral, with ~1.6 billion parameters.

## Configuration

Defined in `TransformerConfig.hpp` via `ModelConfig::make_default()`:

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

## Forward Pass

For one batch `[B, S]`:

```
1. token_embeddings[tokens]           → [B, S, D]   (embedding lookup)
2. For each of 24 layers:
   a. h = rms_norm(h)                  → RMSNorm forward
   b. attn_out = attention(h)          → QKV proj, RoPE, GQA, Wo proj
   c. h = h + attn_out                 → residual
   d. h = rms_norm(h)                  → RMSNorm forward
   e. ffn_out = swiglu_ffn(h)          → gate/up proj, SwiGLU, down proj
   f. h = h + ffn_out                  → residual
3. h = rms_norm(h)                     → final RMSNorm
4. logits = h @ output_projection      → [B, S, V]
```

## Backward Pass

Reverses the forward pass, computing gradients for all parameters. See the root [Model Architecture](/domain/model-architecture.md) for the full backward pass sequence.

## Components at a Glance

| Component | Header | Source | Lines | Key Behavior |
|---|---|---|---|---|
| **Tensor** | `Tensor.hpp` | `Tensor.cpp` | 16,378 | Multi-dimensional float array, matmul (Accelerate + GPU), transpose (vDSP_mtrans) |
| **RMSNorm** | `RMSNorm.hpp` | `RMSNorm.cpp` | 7,388 | `x / sqrt(mean(x²) + eps) * weight` — GPU + CPU paths |
| **RoPE** | `Positional.hpp` | `Positional.cpp` | 6,719 | Precomputed sin/cos table, rotation on Q/K pairs |
| **GQA Attention** | `Attention.hpp` | `Attention.cpp` | 20,684 | 16 Q heads, 8 KV heads; fused GQA kernel (Flash Attention-style) |
| **SwiGLU FFN** | (in Transformer) | `Transformer.cpp` | 23,204 | `silu(x@W_gate) * (x@W_up) @ W_down` — gate/up dispatched concurrently |
| **Cross-Entropy Loss** | `Loss.hpp` | `Loss.cpp` | 6,672 | Softmax → CE; CPU + GPU paths |

## Key Source Files

- `cpp/include/TransformerConfig.hpp` — Config struct
- `cpp/include/Transformer.hpp` — Model class
- `cpp/include/Attention.hpp` — GQA attention class
- `cpp/include/RMSNorm.hpp` — RMS norm class
- `cpp/include/Positional.hpp` — RoPE class
- `cpp/include/Loss.hpp` — Loss class

For complete detail on all components, forward pass, backward pass, and the parameter count breakdown, see the root [Model Architecture](/domain/model-architecture.md).
