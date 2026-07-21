---
type: Project Overview
title: LLM Training Engine — Quickstart
description: Native C++ 1.6B parameter GPT-style LLM training engine with Metal GPU acceleration on Apple Silicon. Entrypoint for navigating the codebase, architecture, workflows, and operations.
tags: [llm, training, gpt, metal, apple-silicon, cpp]
---

# LLM Training Engine — Quickstart

This repository builds a **native C++ training engine** for a 1.6B-parameter GPT-style language model (LLaMA-2/Mistral-inspired architecture) on Apple Silicon (M3 Ultra). It uses:

- **C++20** with Accelerate (BLAS/vDSP), **Metal** GPU shaders, and **MetalPerformanceShaders**
- **Apache Arrow/Parquet** for data ingestion from [The Stack v1](https://huggingface.co/datasets/bigcode/the-stack) dataset
- **tokenizers-cpp** (tiktoken cl100k_base) for subword tokenization
- **Safetensors**-based checkpoint serialization for Python/MLX compatibility

A companion **Python/MLX pipeline** (`python/`) handles higher-level orchestration, data selection, SFT, evaluation, and inference serving.

## Quick Start

### Prerequisites
- macOS (Apple Silicon), Xcode Command Line Tools
- Homebrew: `brew install apache-arrow cmake`
- Python 3.10+ for data tooling and the MLX pipeline

### Build
```bash
cmake -B build -G Ninja
cmake --build build -j$(sysctl -n hw.ncpu)
```

### Pretokenize Training Data
```bash
python pretokenize.py
# Produces data/datasets/rust/train.bin (uint32 token IDs)
```

### Run Pre-training
```bash
./build/run_trainer --data_dir data/datasets/rust \
  --single_data_file train.bin --batch_size 32 \
  --max_steps 1000 --checkpoint_interval 500
```

### Run Tests
```bash
cmake --build build -j$(sysctl -n hw.ncpu)
./build/test_trainer
./build/test_gpu_kernels
# See testing/testing-guidance.md for the full test suite
```

## Repository Map

| Area | Path | Purpose |
|---|---|---|
| **C++ Engine** | `cpp/` | Core training engine (headers, src, tests, docs) |
| **C++ Headers** | `cpp/include/` | All public type declarations |
| **C++ Sources** | `cpp/src/` | Implementation files |
| **GPU Kernels** | `cpp/src/gpu_kernel/` | Metal Shading Language (.metal) kernels + Objective-C++ bridge |
| **C++ Tests** | `cpp/tests/` | Unit and integration test executables |
| **C++ Docs** | `cpp/doc/` | Architecture, kernel architecture, data ingestion, sprint plans, whiteboard pedagogy |
| **Python Pipeline** | `python/` | MLX training, inference, data selection, evaluation, serving |
| **Python Scripts** | `python/scripts/` | Data download, metrics analysis, RAG processing |
| **Root Scripts** | `pretokenize.py`, `fix_metal.py` | Data pre-tokenization, Metal bridge hot-patching |

## Major Domains

- **[Architecture Overview](architecture/overview.md)** — System context, C++ engine design, CPU/GPU cooperative dispatch
- **[Source Map](source-map.md)** — Detailed file-by-file directory map
- **[Model Architecture](domain/model-architecture.md)** — 1.6B GPT model: config, transformer layers, forward/backward pass
- **[GPU Metal Kernels](domain/gpu-kernels.md)** — Custom GEMM tile engine, GQA attention, element-wise kernels, hybrid CPU/GPU dispatch
- **[Training Workflows](workflows/training.md)** — Data ingestion, trainer loop, gradient accumulation, checkpointing, learning rate schedule
- **[Operations & Runbook](operations/runbook.md)** — Building, running, benchmarking, debugging, and known issues
- **[Testing Guidance](testing/testing-guidance.md)** — Test targets, structure, and how to write new tests

## Key Performance Targets

| Metric | Python/MLX Baseline (bf16) | C++ Target (fp32) |
|---|---|---|
| Tokens/sec | 1,987 | > 2,080 |
| MFU | 10.03% | > 10.5% |
| Micro-batch latency (batch=32) | 32.4s | < 32.4s |

Reference: [`cpp/doc/benchmark_baseline_reference.md`](../cpp/doc/benchmark_baseline_reference.md)

## Backlog

- **Python pipeline deep-doc**: The `python/pipeline/` directory has substantial orchestration, training, inference, data, and evaluation subsystems that deserve their own wiki pages.
- **Whiteboard curriculum synthesis**: The `cpp/doc/whiteboard/` series is a pedagogical resource for LLM training from first principles. A synthesized page linking it to source code would be valuable.
- **Python/MLX comparison**: Direct A/B comparison between the C++ and Python forward/backward passes at the tensor level.
