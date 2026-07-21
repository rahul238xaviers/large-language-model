---
type: Project Overview
title: LLM Training Engine — OpenWiki
description: Native C++ 1.6B-parameter GPT-style LLM training engine with Metal GPU acceleration on Apple Silicon. Entrypoint for the OpenWiki knowledge base.
tags: [llm, training, gpt, metal, apple-silicon, cpp, openwiki]
---

# LLM Training Engine — OpenWiki

This is the **OpenWiki knowledge base** for a native C++20 training engine for a ~1.6B-parameter GPT-style language model (LLaMA-2/Mistral-inspired) on Apple Silicon (M3 Ultra). It uses:

- **C++20** with Accelerate (BLAS/vDSP), **Metal** GPU shaders, and MetalPerformanceShaders
- **Apache Arrow/Parquet** for data ingestion from [The Stack v1](https://huggingface.co/datasets/bigcode/the-stack) dataset
- **tokenizers-cpp** (tiktoken cl100k_base) for subword tokenization
- **Safetensors**-based checkpoint serialization for Python/MLX compatibility
- A companion **Python/MLX pipeline** (`python/`) for orchestration, data selection, SFT, evaluation, and inference

> **Canonical entrypoint**: The root `quickstart.md` has full build/run detail. This wiki provides curated navigation, agent-oriented summaries, and cross-linked concept pages.

## Quick Start

### Build
```bash
cmake -B build -G Ninja
cmake --build build -j$(sysctl -n hw.ncpu)
python fix_metal.py                # Post-build hot-patch for Metal buffer cache
cmake --build build -j$(sysctl -n hw.ncpu)  # Rebuild with patch
```

### Pretokenize Data
```bash
python pretokenize.py              # Produces data/datasets/rust/train.bin
```

### Run Pre-training
```bash
./build/run_trainer --data_dir data/datasets/rust --single_data_file train.bin \
  --batch_size 32 --max_steps 1000 --checkpoint_interval 500
```

### Run Tests
```bash
cmake --build build -j$(sysctl -n hw.ncpu)
ctest --test-dir build --output-on-failure
```

## Wiki Pages

| Page | What It Covers |
|---|---|
| [Source Map](source-map.md) | Complete file-by-file directory map |
| [Architecture Overview](architecture/overview.md) | System context, CPU/GPU hybrid compute model, component containers |
| [Model Architecture](domain/model-architecture.md) | 1.6B GPT model: config, transformer layers, forward/backward |
| [GPU Metal Kernels](domain/gpu-kernels.md) | Custom GEMM tile engine, GQA attention, dispatch strategies |
| [Training Workflows](workflows/training.md) | Data ingestion, trainer loop, optimizer, checkpointing |
| [Operations & Runbook](operations/runbook.md) | Building, running, benchmarking, debugging, known issues |
| [Testing Guidance](testing/testing-guidance.md) | Test targets, coverage map, writing new tests |

## Root Documentation (Authoritative Source)

The following root-level docs contain the most detailed content. The wiki links to them as primary references:

- [Root Quickstart](/quickstart.md) — Full build/run/test instructions
- [Root Source Map](/source-map.md) — Detailed directory and file map
- [Architecture Overview](/architecture/overview.md) — System architecture with Mermaid diagrams
- [Model Architecture](/domain/model-architecture.md) — Full model component deep-dive
- [GPU Kernels](/domain/gpu-kernels.md) — Metal kernel inventory and tiling detail
- [Training Workflows](/workflows/training.md) — End-to-end pipeline documentation
- [Operations Runbook](/operations/runbook.md) — Build, run, debug, benchmark instructions
- [Testing Guidance](/testing/testing-guidance.md) — Test suite structure and coverage

## Agent Guidance

When working with this repository:

1. **Start here** — This wiki gives you the conceptual map. Follow links to root docs for detail.
2. **GPU vs CPU** — The hybrid dispatch strategy is the most important architectural decision. See [Architecture Overview](architecture/overview.md).
3. **Checkpoints are safetensors** — Compatible with Python/MLX. The checkpoint format is critical for eval + inference interop.
4. **Metal kernel changes** — After modifying `.metal` files, rebuild and re-run `python fix_metal.py`. See [GPU Kernels](domain/gpu-kernels.md).
5. **No git history** — This repository checkout does not include `.git`. Agent reasoning should be grounded in source code and existing docs.

## Backlog

- **Python pipeline deep-doc**: The `python/pipeline/` directory has substantial orchestration, training, inference, data, and evaluation subsystems that deserve their own wiki pages.
- **Whiteboard curriculum synthesis**: The `cpp/doc/whiteboard/` series is a pedagogical resource for LLM training from first principles. A synthesized page linking it to source code would be valuable.
- **Python/MLX comparison**: Direct A/B comparison between the C++ and Python forward/backward passes at the tensor level.
