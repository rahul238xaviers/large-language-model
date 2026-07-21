---
type: Workflow
title: Training Workflows
description: End-to-end training pipeline from data pre-tokenization through batch streaming, forward/backward pass, optimizer step, checkpointing, and metrics logging.
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

## 1. Data Pre-tokenization (`pretokenize.py`)

A one-step Python script:

```bash
python pretokenize.py
```

1. Reads a Parquet shard (e.g., `data/datasets/rust/train-00000.parquet`)
2. Extracts text from the `content` column
3. Tokenizes with tiktoken `cl100k_base` (appends EOT=100257 after each document)
4. Writes flat `uint32` tokens to `train.bin`

## 2. Data Ingestion (`DataIngestion`)

- Constructor: `DataIngestion(data_dir, single_data_file, max_shard_bytes, batch_size, sequence_length, vocab_path)`
- `get_batch()`: Returns `vector<vector<int>>` shape `[batch_size, seq_len]`
- `skip_sequences(n)`: Offsets read position (used for checkpoint resume)
- No on-the-fly tokenization — all encoding is done offline

## 3. Training Loop (`Trainer`)

Key config (`TrainerConfig`):

| Field | Default | Purpose |
|---|---|---|
| `max_steps` | 100 | Total training steps |
| `warmup_steps` | 10 | LR warmup steps |
| `lr_max` | 3e-4 | Peak learning rate |
| `lr_min` | 3e-5 | Minimum LR after decay |
| `checkpoint_interval` | 500 | Steps between checkpoints |
| `keep_last_n_checkpoints` | 3 | Checkpoint rotation |

### Learning Rate Schedule
- **Warmup**: Linear 0 → `lr_max` over `warmup_steps`
- **Cosine decay**: `lr_max` → `lr_min` over remaining steps

## 4. Optimizer

Two implementations:
- **SGDOptimizer**: `w = w - lr * grad` (baseline for testing)
- **AdamWOptimizer**: Full AdamW with decoupled weight decay, GPU path via `adamw_step.metal`

## 5. Checkpointing

- Format: **Safetensors** (`.safetensors`) — directly loadable by Python/MLX
- Saves all model weights + optimizer state (AdamW moments)
- Auto-resume from latest checkpoint via `--resume` flag
- `keep_last_n_checkpoints` rotates old checkpoints

## Source Files

- `pretokenize.py` — Data pre-tokenization
- `cpp/include/DataIngestion.hpp` / `cpp/src/DataIngestion.cpp` — Data streaming
- `cpp/include/Trainer.hpp` / `cpp/src/Trainer.cpp` — Training loop
- `cpp/include/Optimizer.hpp` / `cpp/src/Optimizer.cpp` — Optimizers
- `cpp/src/RunTrainer.cpp` — CLI entry point

For the full documentation including gradient accumulation plans and MLX baseline comparison, see the root [Training Workflows](/workflows/training.md).
