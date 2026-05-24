# 🦀 The Training Journey of Rust-GPT: A 1.6B Parameter LLM

Welcome to the technical history and engineering journal of **Rust-GPT**, a custom 1.6B parameter decoder-only language model built from scratch and optimized via MLX for Apple Silicon. 

This multi-chapter book chronicles our end-to-end journey: from diagnosing 500GB+ memory OOM crashes on an M3 Ultra, to GQA acceleration, mixed precision, and solving the model repetition loops using a custom hybrid penalty.

---

## 📖 Table of Contents

Navigate through the chapters below:

### 🏗️ [Chapter 1: The Architectural Blueprint](chapter1_architecture.md)
*   *An overview of our custom 1.6B parameter decoder-only GPT model, layers, attention heads, context window, and tokenizer selection.*

### 💥 [Chapter 2: The M3 Ultra & The OOM Crash](chapter2_oom_crash.md)
*   *The technical diagnosis of the quadratic scaling bottleneck of $O(\text{seq}^2)$ attention score activation memory caching during the backward pass, which caused a 500GB+ kernel crash.*

### 🛡️ [Chapter 3: Stabilization & Memory Control](chapter3_stabilization.md)
*   *Re-engineering training using gradient accumulation (`micro_batch=4`, `accumulation=32`) to shrink activation memory by 87.5% and secure a 460GB+ headroom margin.*

### ⚡ [Chapter 4: Hardware Optimization & Scaling](chapter4_hardware_acceleration.md)
*   *Replacing tensor repetition with native MLX broadcasting in Grouped Query Attention (achieving a 3.26x speedup), mixed Bfloat16 precision, and prefetch queue tuning.*

### 🌀 [Chapter 5: The Repetition Crisis & Decoding Engine](chapter5_decoding_upgrades.md)
*   *Tackling severe infinite repetition loops by replacing greedy argmax decoding with stochastic categorical sampling, Top-K/Top-P, and an advanced hybrid (multiplicative + additive) repetition penalty.*

### 🎨 [Chapter 6: The Interactive Playground UI](chapter6_developer_playground.md)
*   *Building a compact, obsidian-themed Gradio app featuring real-time token streaming and an invisible global clipboard polyfill that guarantees a 100% copy success rate.*

---

*This engineering log stands as a comprehensive record of the solutions, benchmarks, and architectures developed during this project.*
