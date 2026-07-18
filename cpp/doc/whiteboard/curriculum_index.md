# 📚 LLM Training Curriculum Index

A complete, sequential index of every concept in LLM training, ordered exactly as they occur.

---

## 🔵 PART 1: Data Pipeline (Before the Model)

| # | Topic | Status |
|---|---|---|
| 01 | Text Corpus → Tokenization (converting words to integer IDs) | ✅ Done |
| 02 | Batching: building the 2D input grid `[batch_size, seq_len]` | ✅ Done |

---

## 🟢 PART 2: Forward Pass (Inside Each Transformer Block)

| # | Topic | Status |
|---|---|---|
| 03 | Token Embedding Lookup: 2D `[batch, seq_len]` → 3D `[batch, seq_len, hidden_dim]` | ✅ Done |
| 04 | RMSNorm: normalize each token's embedding vector individually | ✅ Done |
| 05 | Q, K, V Projections: token-local matrix multiplication | ✅ Done |
| 06 | Head Splitting: reshape Q, K, V into groups of size `head_dim` | ✅ Done |
| 07 | RoPE: precompute global sin/cos table → apply to Q and K heads | ✅ Done |
| 08 | Causal Mask: prevent each token from attending to future tokens | 🔄 In Progress |
| 09 | Attention Scores: scaled dot product of Q · K^T | ⬜ Pending |
| 10 | Softmax: convert raw scores to probabilities row-by-row | ⬜ Pending |
| 11 | Context Output: weighted sum of V vectors using softmax weights | ⬜ Pending |
| 12 | Concat Heads + Output Projection (`Wo`): map back to `hidden_dim` | ⬜ Pending |
| 13 | Residual Connection: add original input back to attention output | ⬜ Pending |
| 14 | Second RMSNorm: normalize before Feed-Forward Network | ⬜ Pending |
| 15 | Feed-Forward Network (FFN) Gate Projection | ⬜ Pending |
| 16 | FFN Up Projection | ⬜ Pending |
| 17 | SwiGLU Activation: gating mechanism to select which features to pass | ⬜ Pending |
| 18 | FFN Down Projection: map back to `hidden_dim` | ⬜ Pending |
| 19 | Second Residual Connection | ⬜ Pending |

---

## 🟡 PART 3: Model Output & Loss

| # | Topic | Status |
|---|---|---|
| 20 | Final RMSNorm (after all transformer blocks) | ⬜ Pending |
| 21 | Language Model Head: linear projection to vocabulary size | ⬜ Pending |
| 22 | Output Softmax: convert logits to next-token probabilities | ⬜ Pending |
| 23 | Cross-Entropy Loss: measure how wrong the prediction was | ⬜ Pending |

---

## 🔴 PART 4: Backward Pass (Gradients, Reverse Order)

| # | Topic | Status |
|---|---|---|
| 24 | Loss Gradient | ⬜ Pending |
| 25 | Language Model Head Backward | ⬜ Pending |
| 26 | Final RMSNorm Backward | ⬜ Pending |
| 27 | FFN Backward: SwiGLU and projection gradients | ⬜ Pending |
| 28 | Second RMSNorm Backward | ⬜ Pending |
| 29 | Attention Output Projection Backward | ⬜ Pending |
| 30 | V Backward | ⬜ Pending |
| 31 | Softmax Backward | ⬜ Pending |
| 32 | Attention Scores Backward (scaled Q · K^T) | ⬜ Pending |
| 33 | RoPE Backward: Q and K gradients | ⬜ Pending |
| 34 | Q, K, V Projection Backward | ⬜ Pending |
| 35 | First RMSNorm Backward | ⬜ Pending |

---

## 🟣 PART 5: Parameter Update

| # | Topic | Status |
|---|---|---|
| 36 | AdamW Optimizer Step: moments, learning rate, weight decay | ⬜ Pending |

---

> **Current Position:** Topic 07 — RoPE Table Construction
