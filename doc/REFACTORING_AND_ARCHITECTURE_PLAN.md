# REFACTORING_AND_ARCHITECTURE_PLAN

## 1. Objective and Scope

Refactor the current monolithic repository into a clean, reusable, production-grade ML engineering pipeline optimized for Apple Silicon (MLX), with strict separation between:

- Data lifecycle (download, profiling, scoring, selection)
- Pre-training and fine-tuning workflows
- Evaluation and analytics
- Inference UI and model registry metadata

This document is a systems architecture and migration plan only. No core pipeline implementation is included at this stage.

## 2. Current-State Analysis (What Exists Today)

Based on repository inspection, the current implementation is centered around `apple-silicon/src` and has strong core training logic but limited modular boundaries.

### 2.1 Current strengths

- Robust MLX training loop with gradient accumulation, explicit memory diagnostics, checkpoint retention, and run directory snapshots.
- Efficient model architecture (`RMSNorm`, `GroupedQueryAttention`, `SwiGLU`, `RoPE`) in dedicated modules.
- Token streaming and async prefetching with multiprocessing workers.
- Basic plotting and an interactive Gradio app for inference from checkpoint.

### 2.2 Current bottlenecks and coupling

- **Hardcoded paths**: data and metrics paths are hardcoded in multiple places (`data/rust`, static run IDs in plotting/scripts).
- **Mixed responsibilities**: training loop owns orchestration, instrumentation, data coordination, and checkpoint policy in one script.
- **Notebook-driven logic**: `marimo/notebook.py` contains data download + profiling + selection logic mixed with ad-hoc experimentation.
- **No formal run registry contract**: metadata is split across `config.json`, `metrics.csv`, and naming conventions but lacks a canonical lineage asset.
- **Inference is not fully runtime-config driven**: Gradio reads checkpoint from `.env` but does not consume structured model metadata lineage.
- **Evaluation stage gap**: no dedicated benchmark/evaluation pipeline for coding/agentic task suites.

## 3. Target Architecture Principles

1. **Configuration-driven execution**: no hardcoded run, dataset, or checkpoint paths in pipeline code.
2. **Run isolation and reproducibility**: each run gets a unique run ID and sealed artifact directory.
3. **Strict component decoupling**: data, train, finetune, eval, analytics, and serving are independently executable.
4. **Apple Silicon first**: MLX-native implementation is primary; `windows-nvidia` remains isolated legacy reference.
5. **Deterministic interfaces**: each stage consumes typed config and emits defined artifacts.

## 4. Proposed Repository Tree (Target)

```text
large-language-model/
  configs/
    base/
      app.yaml
      data.yaml
      model.yaml
      train_pretrain.yaml
      train_sft.yaml
      eval.yaml
      inference.yaml
    profiles/
      local-dev.yaml
      m3-ultra-prod.yaml

  env/
    .env.example

  pipeline/
    __init__.py
    cli.py
    orchestration/
      run_context.py
      stage_runner.py
      artifact_registry.py
      config_loader.py

    data/
      download/
        downloader.py
        sources.py
      profiling/
        doc_sampler.py
        token_sampler.py
        profile_report.py
      selection/
        scoring_engine.py
        filters.py
        dataset_writer.py

    training/
      common/
        callbacks.py
        checkpointing.py
        metrics_logger.py
      pretrain/
        trainer.py
        datamodule.py
      sft/
        synthetic_dataset.py
        trainer.py
        datamodule.py

    generation/
      ollama_client.py
      prompt_templates/
        instruction_generation.j2
      quality_filters.py

    evaluation/
      harness.py
      swebench_adapter.py
      python_workbench_adapter.py
      reports.py

    analytics/
      plot_training.py
      plot_selection.py
      plot_eval.py
      run_dashboard.py

    serving/
      gradio_app/
        app.py
        loader.py
        runtime_config.py
      registry/
        model_metadata.py

  mlx_core/
    __init__.py
    config.py
    model.py
    data_stream.py
    optim.py
    schedules.py
    train_step.py

  runs/
    <run_id>/
      config/
        resolved_config.yaml
        env_snapshot.json
      data/
        downloaded/
        profiles/
        selected/
      pretrain/
        checkpoints/
        metrics/
      synthetic/
        prompts/
        dataset/
      sft/
        checkpoints/
        metrics/
      eval/
        raw/
        summary/
      analytics/
        figures/
        dashboard/
      serving/
        runtime_config.yaml
        model_metadata.json

  legacy/
    apple-silicon/   # moved in phased migration with compatibility shims
    windows-nvidia/

  tests/
    unit/
    integration/
    e2e/
```

## 5. Cross-Cutting Contracts

### 5.1 Run ID contract

- Format: `run_<UTC_YYYYMMDD_HHMMSS>_<short_hash>`
- Generated once per top-level pipeline execution.
- Every stage writes only inside its run-scoped directory.

### 5.2 Configuration contract

- Resolution order: `base yaml` -> `profile yaml` -> `.env` -> CLI overrides.
- Persist resolved config snapshot to `runs/<run_id>/config/resolved_config.yaml`.
- Every stage receives a typed config subsection, not global mutable dicts.

### 5.3 Artifact registry contract

- Maintain canonical artifact pointers in `artifact_registry.json` per run.
- Stages read dependencies from registry instead of hardcoded paths.

### 5.4 Metadata lineage contract

`model_metadata.json` must include:

- `model_id`, `parent_model_id`, `run_id`
- `base_architecture` (layers, heads, context, vocab)
- `train_config_hash`, `data_config_hash`
- `training_duration_sec`, `global_steps`, `best_checkpoint`
- `eval_scores` (task-wise)
- `created_at_utc`, `git_commit`

## 6. Stage-by-Stage Component Specifications

## Stage 1. Data Download Component

### Purpose

Centralize all raw dataset acquisition and caching with configuration-driven sources.

### Proposed files

- `pipeline/data/download/downloader.py`
- `pipeline/data/download/sources.py`

### Inputs

- `data.download.sources[]` from config (dataset name, subset, split, auth mode)
- Env: `HF_TOKEN` (optional, if needed)

### Outputs

- `runs/<run_id>/data/downloaded/<source_name>/manifest.json`
- Symlink or registry pointer to canonical cache store under `data/datasets/`

### Notes

- Keep physical raw cache centralized and immutable where possible.
- Emit deterministic download manifests for reproducibility.

## Stage 2. Data Analysis and Profiling

### Purpose

Audit dataset composition before training using document and token sampling metrics.

### Proposed files

- `pipeline/data/profiling/doc_sampler.py`
- `pipeline/data/profiling/token_sampler.py`
- `pipeline/data/profiling/profile_report.py`

### Inputs

- Download artifacts from Stage 1
- Tokenizer configuration (`cl100k_base` initially)

### Outputs

- `runs/<run_id>/data/profiles/doc_profile.parquet`
- `runs/<run_id>/data/profiles/token_profile.parquet`
- `runs/<run_id>/data/profiles/profile_summary.json`

### Metrics

- Document length distribution, duplicate estimate, code-likeness ratio
- Token coverage and high-frequency token concentration

## Stage 3. Data Selection Pipeline Engine

### Purpose

Score and filter data per run to generate pre-training-ready datasets.

### Proposed files

- `pipeline/data/selection/scoring_engine.py`
- `pipeline/data/selection/filters.py`
- `pipeline/data/selection/dataset_writer.py`

### Inputs

- Profile outputs from Stage 2
- Selection policy config (quality thresholds, language ratio, dedupe policy)

### Outputs

- `runs/<run_id>/data/selected/pretrain_dataset.arrow`
- `runs/<run_id>/data/selected/selection_report.json`

### Scoring interface

- `score(document) -> {quality_score, code_score, keep_probability, reasons[]}`

## Stage 4. Pre-Training Pipeline (MLX Optimized)

### Purpose

Refactor existing MLX trainer into reusable module with explicit train/val split handling.

### Proposed files

- `pipeline/training/pretrain/trainer.py`
- `pipeline/training/pretrain/datamodule.py`
- `pipeline/training/common/checkpointing.py`
- `mlx_core/train_step.py`

### Inputs

- Selected dataset artifact from Stage 3
- `train_pretrain.yaml` config (batching, lr schedule, precision, checkpoint cadence)

### Outputs

- `runs/<run_id>/pretrain/checkpoints/*.safetensors`
- `runs/<run_id>/pretrain/metrics/train_metrics.csv`
- `runs/<run_id>/pretrain/metrics/val_metrics.csv`

### Performance requirements

- Preserve current memory instrumentation and lazy-eval safeguards.
- Keep MLX dtype controls and Apple Silicon profiling hooks.

## Stage 5. Synthetic Instruction Data Generation (Ollama)

### Purpose

Generate 100-500 high-quality instruction samples using local Ollama models.

### Proposed files

- `pipeline/generation/ollama_client.py`
- `pipeline/generation/prompt_templates/instruction_generation.j2`
- `pipeline/training/sft/synthetic_dataset.py`

### Inputs

- Prompt template config
- Source seed corpus or prompts from Stage 3/4 outputs
- Env: `OLLAMA_BASE_URL`, `OLLAMA_MODEL`

### Outputs

- `runs/<run_id>/synthetic/dataset/synthetic_instructions.jsonl`
- `runs/<run_id>/synthetic/dataset/quality_report.json`

### Quality gates

- Schema validation (instruction/input/output)
- Deduplication and minimum diversity thresholds

## Stage 6. Instruction Fine-Tuning Pipeline

### Purpose

Mirror pre-training architecture for SFT with explicit train/val splits and consistent logging.

### Proposed files

- `pipeline/training/sft/trainer.py`
- `pipeline/training/sft/datamodule.py`
- Reuse common checkpoint + metrics modules

### Inputs

- Synthetic dataset from Stage 5
- Optional parent checkpoint from Stage 4

### Outputs

- `runs/<run_id>/sft/checkpoints/*.safetensors`
- `runs/<run_id>/sft/metrics/train_metrics.csv`
- `runs/<run_id>/sft/metrics/val_metrics.csv`

## Stage 7. Evaluation and Workbench Analytics

### Purpose

Benchmark fine-tuned model against coding/agentic workbench tasks.

### Proposed files

- `pipeline/evaluation/harness.py`
- `pipeline/evaluation/swebench_adapter.py`
- `pipeline/evaluation/python_workbench_adapter.py`
- `pipeline/evaluation/reports.py`

### Inputs

- Target model checkpoint + tokenizer + runtime decode config
- Task suite config and sampling budget

### Outputs

- `runs/<run_id>/eval/raw/*.jsonl`
- `runs/<run_id>/eval/summary/eval_report.json`

### Baseline metrics

- Task pass@k, success rate, latency, token usage

## Stage 8. End-to-End Analytics and Visualization

### Purpose

Generate unified plots and summaries across data selection, training, fine-tuning, and evaluation.

### Proposed files

- `pipeline/analytics/plot_training.py`
- `pipeline/analytics/plot_selection.py`
- `pipeline/analytics/plot_eval.py`
- `pipeline/analytics/run_dashboard.py`

### Inputs

- Metrics from Stages 3, 4, 6, 7

### Outputs

- `runs/<run_id>/analytics/figures/*.png`
- `runs/<run_id>/analytics/dashboard/run_summary.html`

## Stage 9. Segregated Gradio Inference and Metadata Registry

### Purpose

Fully decouple inference UI from training scripts and make model loading metadata-driven.

### Proposed files

- `pipeline/serving/gradio_app/app.py`
- `pipeline/serving/gradio_app/loader.py`
- `pipeline/serving/gradio_app/runtime_config.py`
- `pipeline/serving/registry/model_metadata.py`

### Inputs

- `runtime_config.yaml` specifying model/checkpoint/tokenizer/decode settings
- `model_metadata.json` produced at model finalization

### Outputs

- Inference service that can switch checkpoints by config only
- Metadata panel in UI showing active model lineage

### Hard requirement

- Gradio must never import training orchestration modules directly.

## 7. Migration Strategy (Notebook and Monolith to Modular Pipeline)

This roadmap extracts logic incrementally from existing code without breaking legacy paths.

### Phase A: Foundation and compatibility shell

1. Introduce `configs/`, `pipeline/`, and `mlx_core/` skeleton.
2. Add `pipeline/cli.py` with no-op stage commands and run ID generation.
3. Keep existing `apple-silicon/src/train.py` operational.

### Phase B: Extract notebook data flows first

1. Extract dataset download logic from `marimo/notebook.py` into Stage 1 downloader module.
2. Extract analysis functions (`analyze_dataset`, token/doc metrics) into Stage 2 profiling modules.
3. Produce profile JSON/Parquet artifacts compatible with future selection engine.

### Phase C: Selection engine and deterministic outputs

1. Implement Stage 3 scoring/filtering module as pure functions + config.
2. Write selected dataset artifact to run-scoped location and register pointer.

### Phase D: Training refactor with adapters

1. Move reusable model and train-step internals from `apple-silicon/src` into `mlx_core/`.
2. Implement Stage 4 trainer that wraps existing behavior with explicit train/val split.
3. Add compatibility adapter so legacy script can call new trainer backend.

### Phase E: Synthetic + SFT

1. Add Stage 5 Ollama generation component and dataset quality filters.
2. Implement Stage 6 SFT trainer reusing shared callbacks/checkpointing.

### Phase F: Evaluation, analytics, serving decoupling

1. Add Stage 7 evaluation harness/adapters.
2. Add Stage 8 analytics aggregator and dashboard.
3. Migrate Gradio into Stage 9 serving module with runtime config and metadata lineage.

### Phase G: Legacy isolation

1. Keep legacy entry points as wrappers for one transition cycle.
2. Move old scripts into `legacy/apple-silicon` and mark deprecation schedule.

## 8. Implementation Roadmap by Approval Gates

### Gate 1 (Stage 1 only)

- Deliver central download component + config schema + manifest output.
- Acceptance: run-scoped download manifest and no hardcoded paths.

### Gate 2 (Stages 2-3)

- Deliver profiling + selection engine with artifact registry integration.

### Gate 3 (Stage 4)

- Deliver MLX pretraining module parity with current training behavior.

### Gate 4 (Stages 5-6)

- Deliver synthetic instruction generation + SFT pipeline.

### Gate 5 (Stages 7-9)

- Deliver evaluation harness, analytics suite, and fully decoupled Gradio inference with metadata registry.

## 9. Risks and Controls

- **Performance regression risk**: preserve existing MLX kernels and memory instrumentation while refactoring boundaries.
- **Config drift risk**: enforce strict schema validation and resolved-config snapshots.
- **Artifact incompatibility risk**: use artifact registry IDs rather than direct path assumptions.
- **Migration disruption risk**: maintain compatibility wrappers until each stage is verified.

## 10. Immediate Next Step (Pending Approval)

After plan approval, implement **Stage 1** only:

1. Create config schema for data source definitions.
2. Build Stage 1 downloader module and run manifest generation.
3. Add CLI entrypoint for `data-download` stage.
4. Keep all existing training and notebook paths untouched.
