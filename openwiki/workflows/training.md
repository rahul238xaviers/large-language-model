---
type: Workflow
title: Training Workflows
description: End-to-end training pipeline from data ingestion through pretokenization, batch streaming, forward/backward pass, gradient accumulation, optimizer step, checkpointing, and metrics logging.
tags: [training, pipeline, data-ingestion, checkpoint, optimizer, lr-schedule]
---

# Training Workflows

## End-to-End Pipeline

```
Raw Parquet dataset (The Stack v1)
    → pretokenize.py → .bin file (uint32 token IDs)
    → DataIngestion → token batches [B, S]
    → Trainer::train()
        → for each step:
            → get_batch() → tokens [B, S]
            → forward() → logits [B, S, V]
            → loss.forward() → scalar loss
            → loss.backward() → grad_logits
            → backward() → parameter gradients
            → optimizer.step(lr) → parameter update
            → log metrics → metrics.csv
            → save checkpoint → .safetensors
```

## 1. Data Pre-tokenization ([`pretokenize.py`](../pretokenize.py))

A one-step Python script:

```bash
python pretokenize.py
```

1. Reads a Parquet shard (e.g., `data/datasets/rust/train-00000-of-00040.parquet`)
2. Extracts text from the `content` column
3. Tokenizes with tiktoken `cl100k_base` (appends EOT=100257 after each document)
4. Writes flat `uint32` tokens to `train.bin`

The `.bin` format has no metadata or offsets — it is a raw array of token IDs. The C++ `DataIngestion` class splits it into contiguous sequences of `seq_len` tokens (no document boundaries; documents are simply concatenated with EOT tokens).

## 2. Data Ingestion ([`DataIngestion.hpp`](../cpp/include/DataIngestion.hpp), [`DataIngestion.cpp`](../cpp/src/DataIngestion.cpp))

- Constructor: `DataIngestion(data_dir, single_data_file, max_shard_bytes, batch_size, sequence_length, vocab_path)`
- `get_batch()`: Returns `vector<vector<int>>` shape `[batch_size, seq_len]`
- Loads the `.bin` file into memory; generates flat token batches by splitting at sequence boundaries
- `skip_sequences(n)`: Offsets the read position by n sequences (used for checkpoint resume)
- `batch_size()`: Returns configured batch size (used by Trainer's inner loop)
- `EOT = 100257`: End-of-text token id

No on-the-fly tokenization — all encoding is done offline by `pretokenize.py`.

## 3. Training Loop ([`Trainer.hpp`](../cpp/include/Trainer.hpp), [`Trainer.cpp`](../cpp/src/Trainer.cpp))

### TrainerConfig (from `Trainer.hpp`)
```cpp
struct TrainerConfig {
  size_t max_steps = 100;         // Total training steps
  size_t warmup_steps = 10;       // LR warmup steps
  float lr_max = 3e-4f;           // Peak learning rate
  float lr_min = 3e-5f;           // Minimum LR after decay
  size_t log_interval = 10;       // Log every N steps
  std::string checkpoint_dir = "checkpoints";
  std::string metrics_filepath = "metrics.csv";
  size_t checkpoint_interval = 500;
  size_t keep_last_n_checkpoints = 3;
  bool resume = true;             // Auto-resume from latest checkpoint
};
```

### Training Step (pseudocode)
```
for step in 0..max_steps:
    tokens = data_loader.get_batch()
    logits = model.forward(tokens)
    loss = criterion.forward(logits, targets)
    grad_logits = criterion.backward(targets)
    model.backward(grad_logits, tokens)
    lr = get_scheduled_lr(step)
    optimizer.step(lr)
    if step % log_interval == 0:
        log(loss, lr, tok/s, MFU)
    if step % checkpoint_interval == 0:
        save_checkpoint(step)
```

### Learning Rate Schedule
- **Warmup**: Linear from 0 to `lr_max` over `warmup_steps` steps
- **Cosine decay**: From `lr_max` to `lr_min` over the remaining steps

Implemented in `Trainer::get_scheduled_lr(step)`.

### Gradient Accumulation (not yet in C++ engine)
The current C++ implementation performs one optimizer step per batch. The MLX baseline uses gradient accumulation with 4 micro-batches (effective batch size 128 → 4 × 32). This is documented as a future optimization in the sprint plan.

## 4. Optimizer ([`Optimizer.hpp`](../cpp/include/Optimizer.hpp), [`Optimizer.cpp`](../cpp/src/Optimizer.cpp))

Two optimizer implementations:
- **SGDOptimizer**: `w = w - lr * grad` (baseline, for testing)
- **AdamWOptimizer**: Adam with decoupled weight decay:
  - `m = beta1 * m + (1-beta1) * g` (first moment)
  - `v = beta2 * v + (1-beta2) * g²` (second moment, raw)
  - `m_hat = m / (1 - beta1^t)`, `v_hat = v / (1 - beta2^t)` (bias correction)
  - `w = w - lr * (m_hat / (sqrt(v_hat) + eps) + weight_decay * w)`

AdamW also has a GPU kernel path (`adamw_step.metal`) for batched parameter updates.

## 5. Checkpointing ([`Checkpoint.hpp`](../cpp/include/Checkpoint.hpp), [`Checkpoint.cpp`](../cpp/src/Checkpoint.cpp))

- **Format**: Safetensors (`.safetensors`) — compatible with Python/MLX's `mlx.nn.save_weights()` and Hugging Face `safetensors`
- **Content**: All weight tensors (embeddings, 24× layer weights, output projection, norm scales)
- **Fusion helpers**: `Checkpoint.cpp` includes weight fusion (concatenating Q, K, V weights) and transposition helpers for MLX compatibility
- **Resume logic**: `Trainer` scans the checkpoint directory for the latest step, loads weights, and skips consumed sequences

## 6. Production Runner ([`RunTrainer.cpp`](../cpp/src/RunTrainer.cpp))

CLI entry point and the primary way to launch pre-training:

```bash
./build/run_trainer \
  --data_dir data/datasets/rust \
  --single_data_file train.bin \
  --batch_size 32 \
  --seq_len 2048 \
  --max_steps 100000 \
  --checkpoint_interval 500 \
  --lr_max 3e-4 \
  --lr_min 3e-5 \
  --warmup_steps 1000
```

Features:
- **Tee logging**: Duplicates stdout/stderr to a timestamped log file in `logs/`
- **Weight initialization**: GPT-2 style normal distribution (stddev = 1/sqrt(d_model))
- **Config JSON**: Saves `config.json` alongside checkpoints for reproducibility
- **Checkpoint auto-resume**: Scans `checkpoints/` for latest step, loads weights and skips data

## Source Files

| File | Role |
|---|---|
| `pretokenize.py` | Offline data tokenization |
| `cpp/include/DataIngestion.hpp` | Data loader declaration |
| `cpp/src/DataIngestion.cpp` | .bin file reading and batch generation |
| `cpp/include/Trainer.hpp` | Trainer and TrainerConfig |
| `cpp/src/Trainer.cpp` | Training loop, LR schedule, metrics logging |
| `cpp/src/RunTrainer.cpp` | CLI entry point, weight init, resume logic |
| `cpp/include/Optimizer.hpp` | Optimizer base, SGD, AdamW declarations |
| `cpp/src/Optimizer.cpp` | Optimizer step implementations |
| `cpp/include/Checkpoint.hpp` | Checkpoint save/load declarations |
| `cpp/src/Checkpoint.cpp` | Safetensors serialization, fusion, transposition |
| `cpp/doc/DataIngestion.md` | Detailed data ingestion CPU execution blueprint |
