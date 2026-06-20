# 📖 Chapter 2: The M3 Ultra & The OOM Crash

[⬅️ Previous Chapter](chapter1_architecture.md) | [📖 Table of Contents](training_journey.md) | [Next Chapter ➡️](chapter3_stabilization.md)

---

## 🏎️ The Initial Training Configuration

Equipped with a Mac M3 Ultra and **512 GB of Unified Memory**, we configured our training launch with a bold baseline layout:
*   **Micro-batch size**: `32`
*   **Gradient accumulation steps**: `4`
*   **Effective batch size**: `128`

The goal was to maximize GPU thread occupancy and execute high-throughput training steps. Shortly after launching, however, the system froze, and the training pipeline was terminated by a **macOS Out Of Memory (OOM) kernel kill** after memory consumption spiked past **500 GB**.

---

## 🔬 Diagnosing the Activation Memory Bottleneck

We conducted a strict activation memory audit. While many developers focus exclusively on weight and optimizer memory, the actual culprit in long-context training is **activation caching for the backward pass**.

Particularly, the attention score calculation ($QK^T$) scales quadratically with sequence length:
$$\text{Per-Layer Attention Score Memory} = \text{batch\_size} \times \text{heads} \times \text{seq\_len} \times \text{seq\_len} \times \text{dtype\_size}$$

Inputting our parameters at `batch=32` and `seq_len=2048` using 16-bit floats:
$$\text{Memory} = 32 \times 16 \times 2048 \times 2048 \times 2 \text{ bytes}$$
$$\text{Memory} = 4,294,967,296 \text{ bytes} \approx \mathbf{4.3 \text{ GB per layer}}$$

Multiplying this across all **24 layers** of our model revealed that the backward pass cached a staggering **103 GB strictly for attention scores**!

---

## 📊 The OOM Spike Anatomy

Here is the exact diagnostic breakdown of the memory allocations that led to the 500GB+ OOM spike:

| Allocation Component | Memory Usage | Nature |
| :--- | :--- | :--- |
| **Model Weights & Grads (BF16)** | ~6.4 GB | Constant |
| **Optimizer States (Adam FP32)** | ~12.8 GB | Constant |
| **Master Weights (FP32)** | ~6.4 GB | Constant |
| **Attention Scores ($QK^T$ backward cache)** | ~103 GB | Dynamic (caching) |
| **Other Layer Activations** | ~6.4 GB | Linear |
| **MLX Trace + Metal Graph Buffers** | ~350+ GB | Compile-time overhead |
| **OS Overhead & Fragmentation** | ~20 GB | Baseline overhead |
| **Total Peak Footprint** | **500+ GB** | **Result: OOM Kill** |

> [!CAUTION]
> The massive compilation trace overhead (~350 GB) occurs because MLX builds a unified computation graph before execution. With a batch size of 32, this compile-time graph became extremely complex, pushing the hardware beyond the 512 GB limit.

---

[⬅️ Previous Chapter](chapter1_architecture.md) | [📖 Table of Contents](training_journey.md) | [Next Chapter ➡️](chapter3_stabilization.md)
