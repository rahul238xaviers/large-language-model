# C++ Engine Benchmark Baseline & Reference Standards

> **Document Location:** `cpp/doc/benchmark_baseline_reference.md`
> **Purpose:** Permanent memory reference for performance benchmarks, model parameters, matrix shapes, and verification targets comparing the native C++ engine against the Python/MLX baseline.

---

## 🎯 Primary Optimization Goal

The C++ LLM Training Engine must **beat the Python MLX baseline** on Apple Silicon (M3 Ultra):
- **Micro-batch Latency Target:** `< 32.4 seconds` per step of `batch_size 32` (32,768 tokens).
- **MFU Target:** `> 10.5%` MFU.
- **Throughput Target:** `> 2,080 tokens/sec`.

---

## 📊 Recorded Python MLX Baseline (Step 30,000 Checkpoint)

**Source Run:** `python/apple-silicon/runs/run_20260615_185212/`

### 1. `metrics.csv` Entry (Line 30001)

| Metric | Python MLX Value | C++ Target |
|---|---|---|
| **Step** | `30000` | — |
| **Train Loss** | `7.4375` | Same numerical convergence |
| **Tokens / Sec** | `1,987.34 tok/s` | `> 2,080 tok/s` |
| **Learning Rate** | `1.249e-4` | Same scheduler |
| **VRAM Usage** | `5.21 GB` | Flat pre-allocated budget (~51 GB) |
| **MFU %** | `10.03%` | `> 10.5%` |

---

### 2. Log Timing Decomposition (`train.log` lines 384818–384834)

```text
2026-07-19 01:04:27 [INFO] Iter 30000 | Micro-batch 1/4 | Starting forward/backward pass...
2026-07-19 01:04:59 [INFO] Iter 30000 | Micro-batch 1/4 | Completed in 32.41s
2026-07-19 01:05:32 [INFO] Iter 30000 | Micro-batch 2/4 | Completed in 33.13s
2026-07-19 01:06:06 [INFO] Iter 30000 | Micro-batch 3/4 | Completed in 33.06s
2026-07-19 01:06:39 [INFO] Iter 30000 | Micro-batch 4/4 | Completed in 33.09s
2026-07-19 01:06:39 [INFO] Iter 30000 | Loss 7.4375 | tok/s 1987 | MFU 10.0% | Mem 5.2GB
2026-07-19 01:06:39 [INFO] Checkpoint saved: step_030000.safetensors
```

- **Per Micro-Batch (size 32):** `32.41s` to `33.13s`
- **Total Iteration (128 seq):** `131.68s`
- **Data Precision:** Python uses `bfloat16`; C++ uses `float32`.

---

## 📐 Model Configuration & Shape Reference

**Source Config:** `python/apple-silicon/runs/run_20260615_185212/config.json`

### 1. Architecture & Tensor Dimensions

| Parameter | Python Value | C++ Value | Description / Matrix Shape |
|---|---|---|---|
| **n_layer** | `24` | `24` | Number of Transformer layers |
| **n_embd (hidden_dim)** | `1024` | `1024` | Hidden dimension $H = 1024$ |
| **intermediate_dim** | `2730` | `2752` | SwiGLU FFN expansion dimension (padded to multiple of 32 for Metal) |
| **n_head** | `16` | `16` | Attention query heads |
| **n_kv_head** | `8` | `8` | GQA Key/Value heads |
| **head_dim** | `64` | `64` | Per-head dimension ($1024 / 16 = 64$) |
| **block_size (seq_len)** | `1024` | `1024` | Max context sequence length |
| **vocab_size** | `100277` | `100352` | Tokenizer vocabulary (padded to multiple of 64 for Metal) |
| **Total Parameters** | **~380M** | **~380M** | Weights + embeddings |

---

### 2. Batching & Token Accounting

| Parameter | Value | Calculation |
|---|---|---|
| **micro_batch_size** | `32` | 32 sequences of 1024 tokens |
| **tokens_per_micro_batch** | `32,768` | `32 × 1024` tokens |
| **gradient_accumulation_steps** | `4` | 4 micro-batches per optimizer step |
| **effective_batch_size** | `128` | `32 × 4` = 128 sequences |
| **total_tokens_per_iter** | `131,072` | `128 × 1024` tokens per optimizer update |

---

## 🛠️ Verification Command

All performance validation of the C++ engine must be run with:

```bash
make -C build run_trainer && GPU_ENABLED=1 ./build/run_trainer --max_steps 5 --checkpoint_interval 1000 --batch_size 32
```
