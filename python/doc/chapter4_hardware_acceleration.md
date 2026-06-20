# 📖 Chapter 4: Hardware Optimization & Scaling

[⬅️ Previous Chapter](chapter3_stabilization.md) | [📖 Table of Contents](training_journey.md) | [Next Chapter ➡️](chapter5_decoding_upgrades.md)

---

## ⚡ 1. Grouped Query Attention (GQA) Acceleration

In our initial design of `GroupedQueryAttention` (`src/model.py`), we used manual tensor repetition (`mx.repeat`) to tile key and value heads to query heads. This manual memory allocation was causing a heavy bottleneck in our attention layers.

*   **The Optimization**: We refactored `scaled_dot_product_attention` to utilize native **MLX broadcasting** instead of `mx.repeat`.
*   **The Impact**: This bypassed redundant memory allocation entirely. We observed a **3.26x speedup** in raw attention layer forward/backward passes:
    *   *Manual `mx.repeat` attention block*: **1.71s**
    *   *Broadcast-based attention block*: **0.52s**

---

## 🔬 2. Bfloat16 Mixed Precision

Initially, parts of the model ran in standard Float32 precision. To reduce compute overhead and memory footprint, we transitioned the entire architecture to **Bfloat16 precision**:

*   **Parameters & Activations**: Cast directly to `mx.bfloat16`.
*   **Result**: 
    *   Cut memory bandwidth consumption in half.
    *   Reduced our initial compilation overhead, dropping graph compilation step time from 90 seconds down to **63 seconds**.

---

## 📦 3. Data Pipeline Prefetch Tuning

With the GPU executing steps at a vastly accelerated rate, we discovered that training was periodically stalling. The data pipeline streaming from our Parquet shards could not supply tokens fast enough to keep the GPU fully occupied.

To eliminate this input lag, we scaled our asynchronous prefetch buffers inside `src/config.py`:

```python
# Prefetch Configuration
num_worker_threads = 24       # Scaled up from 12 to maximize multi-core pre-processing
prefetch_size = 2000000       # Expanded to 2 million tokens in memory buffer
num_prefetch_batches = 32     # Increased from 16 to queue 32 batches ahead of training
```

This successfully removed all data ingestion bottlenecks, ensuring our M3 Ultra GPU remained at **100% steady-state compute occupancy**.

---

[⬅️ Previous Chapter](chapter3_stabilization.md) | [📖 Table of Contents](training_journey.md) | [Next Chapter ➡️](chapter5_decoding_upgrades.md)
