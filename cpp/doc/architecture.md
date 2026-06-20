# GPT-style Model Architecture on Apple Silicon (MLX)

This document details the high-level architecture of the 1.6 Billion parameter Generative Pre-trained Transformer (GPT) style model, uniquely optimized for training natively on Apple Silicon using the `mlx` framework.

## 1. Core Model Architecture

Our implementation is heavily inspired by modern transformer architectures (such as LLaMA-2 and Mistral), favoring efficient sub-components over the original Vanilla Transformer (Attention Is All You Need).

### 1.1 Tokenization and Embedding
- **Tokenizer**: `tiktoken` (cl100k_base) is used to compress raw text down to sub-word integers.
- **Embedding Layer**: A standard lookup table mapping integer token IDs to high-dimensional continuous vectors ($d_{model} = 2048$).
- **Context Window**: $T = 2048$ tokens.

### 1.2 Rotary Position Embeddings (RoPE)
Instead of adding absolute positional embeddings at the base of the model, we use **Rotary Position Embeddings**.
- RoPE injects positional information directly into the Query and Key matrices at every attention layer.
- Mathematically, it rotates the vector pairs $(x_{2i}, x_{2i+1})$ by an angle proportional to their position $m$ in the sequence:
  $$ f_q(x_m, m) = (x_m \cos(m\theta)) + (x_m \otimes \text{reversal}) \sin(m\theta) $$
- **Advantage**: It allows the model to extrapolate to longer sequence lengths naturally.

### 1.3 Grouped Query Attention (GQA)
Traditional Multi-Head Attention (MHA) has one Key and Value head for every Query head. This requires a massive memory bandwidth during decoding.
- We use **Grouped Query Attention**.
- **Mechanism**: We have 16 Query heads, but only 8 Key/Value heads. Each Key/Value head is shared ("grouped") across 2 Query heads.
- **Advantage**: Massively reduces the memory footprint of the KV-cache during inference without significantly impacting model accuracy.

### 1.4 SwiGLU FeedForward Network (FFN)
We replace the standard ReLU-based MLP with a **SwiGLU** activated network.
- **Math**: $ \text{SwiGLU}(x) = \text{Swish}(x W_1) \otimes (x W_2) $ where $ \text{Swish}(x) = x \sigma(x) $
- It utilizes three projection matrices instead of two.
- **Advantage**: Proven empirically to converge faster and achieve lower loss than traditional activations.

### 1.5 RMSNorm (Root Mean Square Normalization)
Instead of standard Layer Normalization, we use **RMSNorm**.
- **Math**: $ \text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d} \sum_{i=1}^{d} x_i^2 + \epsilon}} \odot \gamma $
- **Advantage**: It ignores mean-centering and only scales by the root-mean-square. It is computationally cheaper and highly effective.

---

## 2. System Design and Data Pipeline

Training a 1.6B parameter model requires massive data throughput. The system is split across decoupled modules.

### 2.1 Asynchronous Data Streaming (`data.py`)
Hugging Face's `datasets` library streams a 3TB Parquet dataset (`bigcode/the-stack`).
To prevent CPU bottlenecking:
1. **Parallel Workers**: A pool of background threads download independent shards of the dataset.
2. **Token Queue**: Workers encode text to integers and push to a shared thread-safe queue.
3. **AsyncBatchPrefetcher**: Another background thread chunks these integers into micro-batches $X$ and $Y$ (shifted by 1), and places them in a fast-access buffer.

### 2.2 Gradient Accumulation & Memory Management
The M3 Ultra GPU has 512GB of unified memory, but calculating the backward pass for a massive batch is still computationally unfeasible.
- **Micro-batching**: The effective batch size (e.g., 128) is divided into micro-batches (e.g., 4).
- **Gradient Accumulation**: The loss gradients are calculated for the micro-batch and added to a rolling tally. The optimizer only steps once the tally reaches 128.
- **Lazy Evaluation**: `mlx` builds computational graphs lazily. We must call `mx.eval(accumulated_grads)` inside the inner micro-batch loop to collapse the graph, otherwise the AST grows infinitely and crashes the OS (Out Of Memory).

### 2.3 Learning Rate Scheduler (`utils.py`)
We employ a **Cosine Decay with Linear Warmup**.
- **Warmup**: The learning rate linearly scales from 0 to $\eta_{max}$ to prevent early gradient explosions.
- **Cosine Decay**: The rate follows the cosine curve down to $\eta_{min}$, allowing fine-grained optimization at the end of training.
