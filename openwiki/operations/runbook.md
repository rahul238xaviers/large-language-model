---
type: Playbook
title: Operations & Runbook
description: How to build, run, benchmark, debug, and troubleshoot the C++ LLM training engine on Apple Silicon.
tags: [operations, runbook, build, debug, benchmark, metal]
---

# Operations & Runbook

## Building

### Prerequisites

```bash
# macOS Apple Silicon, Xcode Command Line Tools
xcode-select --install

# Homebrew dependencies
brew install apache-arrow cmake ninja

# Python (for data tooling)
python3 -m venv .venv
source .venv/bin/activate
pip install tiktoken pandas numpy
```

### Build Commands

```bash
# Configure (with Ninja for faster builds)
cmake -B build -G Ninja

# Build all targets
cmake --build build -j$(sysctl -n hw.ncpu)

# Build specific target
cmake --build build --target run_trainer -j$(sysctl -n hw.ncpu)
cmake --build build --target test_gpu_kernels -j$(sysctl -n hw.ncpu)

# Clean build
cmake --build build --clean-first
```

The build produces the `data_ingestion` static library and links all test executables and `run_trainer` against it.

### Metal Kernel Compilation

Metal shaders are compiled offline by CMake:

```cmake
add_custom_command(OUTPUT default.metallib
  COMMAND ${XCRUN} -sdk macosx metal -c ... -o default.metallib
  DEPENDS ${METAL_SOURCES})
```

The `.metallib` is embedded into the binary and loaded at runtime by `MetalBridge.mm::initialize()`. If you modify a `.metal` file, rebuild will recompile the metallib automatically.

### Post-Build Hot-Patch

After building (or after regenerating `MetalBridge.mm`), apply the buffer cache persistence patch:

```bash
python fix_metal.py
```

This modifies `MetalBridge.mm` to add persistent buffer tracking. After running, rebuild.

## Running

### Data Preparation

```bash
# 1. Download dataset shards (optional — you can bring your own .bin)
python python/scripts/download_cpp_blobs.py

# 2. Pretokenize a shard
python pretokenize.py
```

### Pre-training

```bash
# Basic run
./build/run_trainer \
  --data_dir data/datasets/rust \
  --single_data_file train.bin \
  --batch_size 32 \
  --max_steps 100000 \
  --checkpoint_interval 500

# With custom config
./build/run_trainer \
  --data_dir data/datasets/rust \
  --single_data_file train.bin \
  --batch_size 32 \
  --seq_len 2048 \
  --max_steps 100000 \
  --lr_max 3e-4 \
  --lr_min 3e-5 \
  --warmup_steps 1000 \
  --checkpoint_interval 500 \
  --log_interval 10
```

Output:
- Training logs (tee'd) → `logs/train_<timestamp>.log`
- Metrics CSV → `metrics.csv` (columns: step, loss, tok/s, lr, mfu)
- Checkpoints → `checkpoints/step_<step>.safetensors`
- Config JSON → `checkpoints/config.json`

### Tests

```bash
# Build and run all tests
cmake --build build -j$(sysctl -n hw.ncpu)
ctest --test-dir build

# Or run individually
./build/test_trainer
./build/test_gpu_kernels
./build/test_transformer
./build/test_backward
./build/test_attention
./build/test_math_ops
./build/test_rmsnorm
./build/test_rope
./build/test_activations
./build/test_loss
./build/test_data_ingestion
```

## Benchmarking

### Performance Targets

Reference: [`cpp/doc/benchmark_baseline_reference.md`](../cpp/doc/benchmark_baseline_reference.md)

| Metric | MLX Baseline (bf16) | C++ Target (fp32) |
|---|---|---|
| Micro-batch latency (B=32) | 32.4s | < 32.4s |
| Tokens/sec | 1,987 | > 2,080 |
| MFU | 10.03% | > 10.5% |
| VRAM (MLX) | 5.21 GB | ~51 GB budget |

### Profiling

The bridge exposes profiling variables:

```cpp
metal_bridge::reset_profile_stats();  // Reset at start of training
// ... run training ...
// Read at end or log interval:
metal_bridge::accum_gpu_time_ms
metal_bridge::accum_cpu_time_ms
metal_bridge::count_gpu_calls
metal_bridge::count_cpu_calls
```

Metrics are logged automatically at `log_interval` during training.

### Python Metrics Analysis

```bash
python python/scripts/analyze_metrics.py metrics.csv
```

## Debugging

### Common Issues

#### Metal Buffer Allocation Failure
**Symptom**: `get_or_create_buffer` fails or returns nil.
**Cause**: Host pointer not 16KB aligned (Apple Silicon MMU page requirement).
**Fix**: Ensure buffers are aligned with `posix_memalign` or use the Metal-bridged allocation path.

#### GPU Kernel Compilation Errors
**Symptom**: Build fails on `.metal` file compilation.
**Cause**: `MAX_SEQ_LEN` or tile-size defines mismatch.
**Fix**: Check `CMakeLists.txt` for `set(MAX_SEQ_LEN ...)` and tile config settings. The GQA kernel uses `MAX_SEQ_LEN` to size threadgroup memory.

#### Buffer Cache Corruption
**Symptom**: Incorrect results or crashes after adding new kernel dispatch.
**Cause**: Persistent buffer flag not set for weight tensors, so `commit_batch` evicts them.
**Fix**: Run `python fix_metal.py` and rebuild, or manually set `is_persistent=true` in the `get_or_create_buffer` call for weight buffers.

#### Low GPU Occupancy
**Symptom**: GPU utilization is low (< 50%) despite large matmuls.
**Cause**: Tile sizes too large for the operation; L1 cache occupancy drops below 2 threadgroups per core.
**Fix**: Reduce TILE_M or TILE_N (see `gpu_tiling_guide.md` for occupancy calculation).

#### Training Loss Not Converging
**Symptom**: Loss stays near vocab_size (≈ 11.5) or diverges.
**Causes**:
- Learning rate too high: reduce `lr_max`
- Weight initialization too large: check `initialize_gpt_weights` stddev
- No causal mask in GQA (known limitation): tokens attend to future tokens
- CPU fallback path divergence: verify that `metal_bridge::is_available()` returns true

### Known Limitations

| Issue | Status | Workaround |
|---|---|---|
| No causal mask in GQA | Known | Add causal mask to `gemm_gqa.metal` (Flash Attention pattern) |
| No gradient accumulation | Planned | Submit 4 micro-batches per optimizer step |
| Float32 only | By design | Can add bfloat16 support via Metal 3.1+ `bfloat` type |
| Single-file .bin ingestion | Current | Add multi-shard support for larger datasets |
| No distributed training | Future scope | N/A for single-device Apple Silicon training |

## File Layout

Runtime directories created during training:

```
./
├── data/datasets/rust/train.bin     # Pretokenized data (generated)
├── logs/train_<timestamp>.log       # Tee'd console output (generated)
├── metrics.csv                       # Step metrics (generated)
├── checkpoints/
│   ├── config.json                   # Training config (generated)
│   ├── step_000500.safetensors       # Checkpoint (generated)
│   ├── step_001000.safetensors
│   └── ...
├── build/                            # Build output
│   └── run_trainer                   # Production training binary
├── default.metallib                  # Compiled Metal library (generated)
└── scratch/                          # Scratch experiments (git-ignored)
```
