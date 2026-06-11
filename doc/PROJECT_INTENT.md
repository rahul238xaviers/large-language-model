Project Intent: 1.6B Rust-GPT on Apple Silicon (MLX)

## 1) Vision and Non-Negotiables

This project exists to build, understand, and optimize a GPT-style language model from first principles on Apple Silicon, not to treat model training as a black box.

The objective is not only quality loss curves, but a robust engineering system that can sustain long training runs with predictable throughput, stable memory behavior, and clear observability.

## 2) Execution Rule: User-Only Training/Test Commands

Hard rule for this repository:

- Any training or testing related command must be executed only by the user.
- The assistant must never run training or test commands.
- The assistant may still inspect code, refine docs, propose fixes, and provide command sequences for the user to run.

This rule applies to:

- Training entrypoints (for example `apple-silicon/src/train.py` workflows).
- Unit/performance test scripts under `apple-silicon/tests/`.
- Any benchmark or profiling script that executes model compute loops.

## 3) Current Implementation Ground Truth

The intent must match what already exists in code.

### 3.1 Model architecture (implemented)

- 24 transformer layers.
- Embedding dimension: 2048.
- Query heads: 16.
- KV heads: 8 (Grouped Query Attention).
- Head dimension: 128.
- Context window: 2048 tokens.
- Vocabulary size: 100277 (`cl100k_base`).
- RMSNorm + SwiGLU FFN + RoPE + causal masking + tied token/output embeddings.

### 3.2 Training architecture (implemented)

- MLX compiled update step (`@mx.compile`) with internal micro-batch gradient accumulation.
- Default micro-batch size: 32.
- Default accumulation steps: 16.
- Effective batch size: 512.
- Optimizer: AdamW.
- LR policy: warmup then cosine decay.
- Metrics logging to CSV: loss, token throughput, LR, memory, MFU estimate.

### 3.3 Data pipeline (implemented)

- Multiprocessing token workers read local Rust parquet files from `data/rust`.
- `tiktoken` (`cl100k_base`) with EOT append.
- Async prefetch thread builds full effective iterations and queues them for training.

### 3.4 Operational tooling (implemented)

- Run folders under `runs/run_YYYYMMDD_HHMMSS` with `config.json`, `metrics.csv`, and logs.
- Plotting utility in `apple-silicon/src/plot_results.py` for loss/throughput/memory/LR curves.
- Data download helper in `apple-silicon/scripts/download_data.py`.

## 4) System Capability Harness Plan

To fully use the machine while preserving stability:

1. Keep compute in bfloat16 for throughput and memory efficiency.
2. Preserve compiled step shape consistency (batch and sequence dimensions fixed during a run).
3. Tune only one major axis at a time:
	- micro-batch size,
	- accumulation steps,
	- number of worker processes,
	- prefetch queue depth.
4. Track and compare each run by:
	- tokens/sec,
	- memory usage,
	- MFU percentage,
	- loss slope stability.
5. Prefer sustained throughput over short spikes if the goal is multi-day reliability.

## 5) Debug Priorities and Known Risk Areas

When debugging, prioritize failures that halt training or silently distort metrics.

### P0: API mismatches between tests and core model

Some unit/perf scripts call `model.set_dtype(...)`, but the main model path uses `model.update(...)` casting. This mismatch is a likely breakage source in local test scripts.

### P1: Data stream starvation/timeouts

`ParallelTokenStream.get_batch()` can raise timeout if worker throughput drops or data path/layout is incomplete.

### P2: Process/thread pressure in data ingestion

`num_worker_threads` is used as process count; too high can increase contention and reduce effective throughput.

### P3: Learning-rate schedule consistency

There are multiple LR helper implementations across files; keep behavior aligned to avoid analysis confusion between intended and actual LR dynamics.

### P4: Checkpoint lifecycle drift

Checkpoint manager exists, but training loop must remain consistent with actual save cadence and retention expectations.

## 6) Debug Workflow Contract (Assistant + User)

1. Assistant inspects code and pinpoints likely root causes.
2. Assistant proposes minimal patches and exact commands.
3. User runs all training/testing/benchmark commands.
4. User shares logs/errors.
5. Assistant iterates on fixes until stable.

This preserves your rule while still enabling fast debugging.

## 7) Success Criteria

Success means:

- A reproducible, observable training system for a 1.6B class GPT model on Apple Silicon.
- Stable long-run behavior (no repeated memory fragmentation crashes, no silent pipeline stalls).
- Clear performance envelope documentation per run.
- Debug loops that are fast, evidence-driven, and user-controlled for execution.