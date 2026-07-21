---
type: Playbook
title: Operations & Runbook
description: How to build, run, benchmark, debug, and troubleshoot the C++ LLM training engine on Apple Silicon.
tags: [operations, runbook, build, debug, benchmark, metal]
---

# Operations & Runbook

## Prerequisites

```bash
# macOS Apple Silicon, Xcode Command Line Tools
xcode-select --install

# Homebrew dependencies
brew install apache-arrow cmake ninja

# Python (for data tooling)
python3 -m venv .venv && source .venv/bin/activate
pip install tiktoken pandas numpy
```

## Building

```bash
# Configure with Ninja
cmake -B build -G Ninja

# Build all targets
cmake --build build -j$(sysctl -n hw.ncpu)

# Build specific target
cmake --build build --target run_trainer -j$(sysctl -n hw.ncpu)
cmake --build build --target test_gpu_kernels -j$(sysctl -n hw.ncpu)
```

### Metal Kernel Compilation
Metal shaders are compiled offline by CMake into `default.metallib`, embedded into the binary. Modify `.metal` files → rebuild automatically recompiles the metallib.

### Post-Build Hot-Patch
```bash
python fix_metal.py              # Patches MetalBridge.mm buffer cache
cmake --build build -j$(sysctl -n hw.ncpu)  # Rebuild with patch
```

## Running

### Data Preparation
```bash
python python/scripts/download_cpp_blobs.py   # Download dataset shards (optional)
python pretokenize.py                          # Pretokenize a shard
```

### Pre-training
```bash
./build/run_trainer \
  --data_dir data/datasets/rust \
  --single_data_file train.bin \
  --batch_size 32 \
  --max_steps 100000 \
  --checkpoint_interval 500
```

### Checkpoint Resume
```bash
./build/run_trainer \
  --data_dir data/datasets/rust \
  --single_data_file train.bin \
  --resume                    # Auto-resume from latest checkpoint
```

## Debugging

- **Metal validation**: Run with `MTL_DEBUG_LAYER=1` or `MTL_DEBUG_LAYER_VALIDATE=1` environment variables
- **GPU timing**: `accum_gpu_time_ms` / `accum_cpu_time_ms` externals in `MetalBridge.mm` track per-step GPU/CPU time
- **Assertions**: C++ engine uses `assert()` widely; build with `-DCMAKE_BUILD_TYPE=Debug` for debug symbols
- **Known issue**: `get_batch()` may segfault when `.bin` is empty — ensure `pretokenize.py` produced a valid file

## Benchmarking

| Metric | Python/MLX Baseline (bf16) | C++ Target (fp32) |
|---|---|---|
| Tokens/sec | 1,987 | > 2,080 |
| MFU | 10.03% | > 10.5% |
| Micro-batch latency (batch=32) | 32.4s | < 32.4s |

Reference: `cpp/doc/benchmark_baseline_reference.md`

## Source Files

- `CMakeLists.txt` — Build system
- `fix_metal.py` — Post-build hot-patch
- `cpp/src/gpu_kernel/MetalBridge.mm` — GPU bridge (debug/profiling hooks)
- `cpp/src/RunTrainer.cpp` — CLI entry point
- `cpp/doc/benchmark_baseline_reference.md` — Baseline performance reference

For the complete runbook with all CLI flags, build variants, and troubleshooting guidance, see the root [Operations Runbook](/operations/runbook.md).
