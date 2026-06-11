"""Pipeline CLI – entry point for all pipeline stages.

Usage:
    # Stage 1: download all configured sources
    python -m pipeline data-download

    # Stage 1: download a specific named source only
    python -m pipeline data-download --source rust_stack

    # Stage 1: resume an existing run and add another source to it
    python -m pipeline data-download --run-id run_20260529_091500_abc1234 --source fineweb_edu

    # Use a non-default profile
    python -m pipeline --profile m3-ultra-prod data-download

Global flags (must come before the sub-command):
    --config-dir    Path to the configs/ directory (default: "configs")
    --profile       Config profile to load    (default: "local-dev")
    --log-level     Logging verbosity         (default: INFO)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from pipeline.orchestration.artifact_registry import ArtifactRegistry
from pipeline.orchestration.config_loader import ConfigLoader
from pipeline.orchestration.run_context import RunContext


# ------------------------------------------------------------------ #
# Logging                                                               #
# ------------------------------------------------------------------ #

def _setup_logging(level: str) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
        stream=sys.stdout,
        force=True,
    )


# ------------------------------------------------------------------ #
# Stage handlers                                                        #
# ------------------------------------------------------------------ #

def cmd_data_download(args: argparse.Namespace) -> None:
    """Stage 1: Download datasets and write per-source manifests."""
    load_dotenv()
    _setup_logging(args.log_level)
    logger = logging.getLogger("pipeline.cli")

    cfg = _load_config(args)
    ctx = _resolve_run_context(args, cfg)
    logger.info("Run ID: %s", ctx.run_id)

    # Persist a snapshot of the resolved config for reproducibility
    config_dir = ctx.config_dir()
    with open(config_dir / "resolved_config.json", "w") as fh:
        json.dump(cfg, fh, indent=2, default=str)
    # Also persist a snapshot of relevant env vars (values redacted for secrets)
    env_snapshot = {
        k: ("***" if "TOKEN" in k or "SECRET" in k or "KEY" in k else v)
        for k, v in os.environ.items()
        if k.startswith(("HF_", "PIPELINE_", "OLLAMA_", "CHECKPOINT_"))
    }
    with open(config_dir / "env_snapshot.json", "w") as fh:
        json.dump(env_snapshot, fh, indent=2)

    registry = ArtifactRegistry(ctx.run_dir)
    download_cfg = cfg.get("download", {})

    hf_token: str | None = os.environ.get("HF_TOKEN") or download_cfg.get("hf_token") or None
    cache_dir = Path(download_cfg.get("cache_dir", "data/datasets"))
    sources_cfg: list[dict] = download_cfg.get("sources", [])

    # Optional source filter from --source flag
    if args.source:
        requested = set(args.source)
        sources_cfg = [s for s in sources_cfg if s.get("name") in requested]
        if not sources_cfg:
            logger.error("No matching sources found for filter: %s", args.source)
            sys.exit(1)

    if not sources_cfg:
        logger.warning("No sources configured in download.sources – nothing to do.")
        sys.exit(0)

    from pipeline.data.download.downloader import Downloader
    from pipeline.data.download.sources import HFDatasetSource

    downloader = Downloader(cache_dir=cache_dir, hf_token=hf_token)

    for source_dict in sources_cfg:
        source = HFDatasetSource.from_dict(source_dict)
        logger.info("Processing source: %s (%s)", source.name, source.repo_id)
        manifest_dir = ctx.stage_dir("data/downloaded") / source.name
        manifest_path = downloader.download(source, manifest_dir)
        registry.register("download", source.name, manifest_path)
        logger.info("Registered artifact: download.%s → %s", source.name, manifest_path)

    logger.info("Stage 1 complete.  Run ID: %s", ctx.run_id)
    # Print run ID to stdout so it can be captured by shell scripts
    print(ctx.run_id)


def cmd_data_profile(args: argparse.Namespace) -> None:
    """Stage 2: Profile downloaded datasets and write profiling artifacts."""
    load_dotenv()
    _setup_logging(args.log_level)
    logger = logging.getLogger("pipeline.cli")

    if not args.run_id:
        logger.error(
            "--run-id is required for data-profile. "
            "Run 'pipeline data-download' first and pass its run ID here."
        )
        sys.exit(1)

    cfg = _load_config(args)
    ctx = RunContext.resume(args.run_id, base_dir=Path(cfg.get("runs_dir", "runs")))
    logger.info("Run ID: %s", ctx.run_id)

    registry = ArtifactRegistry(ctx.run_dir)
    download_cfg = cfg.get("download", {})
    profiling_cfg = cfg.get("profiling", {})

    hf_token: str | None = os.environ.get("HF_TOKEN") or download_cfg.get("hf_token") or None
    sample_size: int = int(profiling_cfg.get("sample_size", 100_000))
    tokenizer_name: str = profiling_cfg.get("tokenizer", "cl100k_base")
    text_columns: dict = profiling_cfg.get("text_columns", {})

    # Sources to profile = all Stage 1 artifacts unless --source filter given
    registered_sources = list(registry.list_stage("download").keys())
    if not registered_sources:
        logger.error(
            "No Stage 1 artifacts found in run %s. Run data-download first.",
            ctx.run_id,
        )
        sys.exit(1)

    sources_to_profile = registered_sources
    if args.source:
        requested = set(args.source)
        sources_to_profile = [s for s in registered_sources if s in requested]
        if not sources_to_profile:
            logger.error(
                "Source filter %s matched nothing. Available: %s",
                args.source,
                registered_sources,
            )
            sys.exit(1)

    from pipeline.data.profiling.profile_report import build_and_write_profile

    profile_dir = ctx.stage_dir("data/profiles")

    for source_name in sources_to_profile:
        manifest_path = registry.get("download", source_name)
        text_column = text_columns.get(source_name, "content")

        logger.info(
            "Profiling source: %s  (text_column=%s, sample_size=%d)",
            source_name, text_column, sample_size,
        )

        artifacts = build_and_write_profile(
            source_name=source_name,
            manifest_path=manifest_path,
            text_column=text_column,
            sample_size=sample_size,
            tokenizer_name=tokenizer_name,
            output_dir=profile_dir,
            hf_token=hf_token,
        )

        for artifact_name, artifact_path in artifacts.items():
            registry_key = f"{source_name}_{artifact_name}"
            registry.register("profiling", registry_key, artifact_path)
            logger.info(
                "Registered: profiling.%s → %s", registry_key, artifact_path
            )

    logger.info("Stage 2 complete.  Run ID: %s", ctx.run_id)
    print(ctx.run_id)


def cmd_data_select(args: argparse.Namespace) -> None:
    """Stage 3: Score and filter profiled documents; write the training dataset."""
    load_dotenv()
    _setup_logging(args.log_level)
    logger = logging.getLogger("pipeline.cli")

    cfg = _load_config(args)
    ctx = RunContext.resume(args.run_id, base_dir=Path(cfg.get("runs_dir", "runs")))
    logger.info("Run ID: %s", ctx.run_id)

    registry = ArtifactRegistry(ctx.run_dir)
    download_cfg   = cfg.get("download", {})
    profiling_cfg  = cfg.get("profiling", {})
    selection_cfg  = cfg.get("selection", {})

    hf_token: str | None = os.environ.get("HF_TOKEN") or download_cfg.get("hf_token") or None
    text_columns: dict    = profiling_cfg.get("text_columns", {})

    from pipeline.data.selection.filters       import SelectionPolicy
    from pipeline.data.selection.scoring_engine import score_profile, summarise_scoring
    from pipeline.data.selection.dataset_writer import (
        write_selected_dataset, merge_arrow_files, write_selection_report,
    )

    policy = SelectionPolicy.from_dict(selection_cfg.get("policy", {}))
    batch_size: int = int(selection_cfg.get("batch_size", 2048))

    # Resolve which sources to process
    registered_sources = list(registry.list_stage("download").keys())
    if not registered_sources:
        logger.error("No Stage 1 artifacts in run %s – run data-download first.", ctx.run_id)
        sys.exit(1)

    sources_to_select = registered_sources
    if args.source:
        requested = set(args.source)
        sources_to_select = [s for s in registered_sources if s in requested]
        if not sources_to_select:
            logger.error("Source filter %s matched nothing. Available: %s",
                         args.source, registered_sources)
            sys.exit(1)

    selection_dir  = ctx.stage_dir("data/selected")
    per_source_stats: dict[str, dict] = {}
    arrow_files: list[Path] = []

    for source_name in sources_to_select:
        # Locate Stage 2 doc_profile artifact
        profile_key = f"{source_name}_doc_profile"
        try:
            doc_profile_path = registry.get("profiling", profile_key)
        except KeyError:
            logger.error(
                "No profiling artifact for source %r (key: profiling.%s). "
                "Run data-profile first.", source_name, profile_key,
            )
            sys.exit(1)

        manifest_path = registry.get("download", source_name)
        text_column   = text_columns.get(source_name, "content")

        logger.info("Scoring source: %s", source_name)
        scored_df = score_profile(doc_profile_path, policy)
        stats = summarise_scoring(scored_df)
        per_source_stats[source_name] = stats
        logger.info(
            "  kept=%d (%.1f%%)  dropped=%d",
            stats["kept"], stats["keep_rate_pct"], stats["dropped"],
        )

        if stats["kept"] == 0:
            logger.warning("No documents kept for source %r – skipping Arrow write.", source_name)
            continue

        arrow_path = write_selected_dataset(
            source_name=source_name,
            manifest_path=manifest_path,
            scored_df=scored_df,
            output_dir=selection_dir,
            text_column=text_column,
            batch_size=batch_size,
            hf_token=hf_token,
        )
        arrow_files.append(arrow_path)
        registry.register("selection", f"{source_name}_arrow", arrow_path)

    # Merge all per-source Arrow files into a single dataset
    if arrow_files:
        merged_path = selection_dir / "pretrain_dataset.arrow"
        merge_arrow_files(arrow_files, merged_path)
        registry.register("selection", "pretrain_dataset", merged_path)
        logger.info("Merged dataset → %s", merged_path)

    # Write human-readable selection report
    report_path = selection_dir / "selection_report.json"
    write_selection_report(per_source_stats, report_path)
    registry.register("selection", "selection_report", report_path)

    logger.info("Stage 3 complete.  Run ID: %s", ctx.run_id)
    print(ctx.run_id)


def cmd_data_tokenise(args: argparse.Namespace) -> None:
    """Stage 4: Tokenise the selected dataset and write training sequences."""
    load_dotenv()
    _setup_logging(args.log_level)
    logger = logging.getLogger("pipeline.cli")

    cfg = _load_config(args)
    ctx = RunContext.resume(args.run_id, base_dir=Path(cfg.get("runs_dir", "runs")))
    logger.info("Run ID: %s", ctx.run_id)

    registry = ArtifactRegistry(ctx.run_dir)
    tokenisation_cfg = cfg.get("tokenisation", {})

    encoding_name: str = tokenisation_cfg.get("encoding", "cl100k_base")
    block_size:    int  = int(tokenisation_cfg.get("block_size", 2048))
    stride_cfg          = tokenisation_cfg.get("stride", None)
    stride:        int | None = int(stride_cfg) if stride_cfg is not None else None
    text_column:   str  = tokenisation_cfg.get("text_column", "text")

    # Locate Stage 3 merged dataset
    try:
        arrow_path = registry.get("selection", "pretrain_dataset")
    except KeyError:
        logger.error(
            "No Stage 3 merged dataset found in run %s. "
            "Run data-select first.", ctx.run_id,
        )
        sys.exit(1)

    from pipeline.data.tokenisation.tokeniser    import Tokeniser
    from pipeline.data.tokenisation.batch_engine import run_tokenisation_engine

    tokeniser     = Tokeniser(encoding_name)
    tokenise_dir  = ctx.stage_dir("data/tokenised")

    logger.info(
        "Tokenising: encoding=%s  block_size=%d  stride=%s",
        encoding_name, block_size, stride if stride is not None else block_size,
    )

    artifacts = run_tokenisation_engine(
        arrow_path=arrow_path,
        output_dir=tokenise_dir,
        tokeniser=tokeniser,
        block_size=block_size,
        stride=stride,
        text_column=text_column,
    )

    for artifact_name, artifact_path in artifacts.items():
        registry.register("tokenisation", artifact_name, artifact_path)
        logger.info("Registered: tokenisation.%s → %s", artifact_name, artifact_path)

    logger.info("Stage 4 complete.  Run ID: %s", ctx.run_id)
    print(ctx.run_id)


def cmd_train_pretrain(args: argparse.Namespace) -> None:
    """Stage 5: Train the GPT model on the tokenised sequences."""
    load_dotenv()
    _setup_logging(args.log_level)
    logger = logging.getLogger("pipeline.cli")

    cfg = _load_config(args)
    ctx = RunContext.resume(args.run_id, base_dir=Path(cfg.get("runs_dir", "runs")))
    logger.info("Run ID: %s", ctx.run_id)

    registry = ArtifactRegistry(ctx.run_dir)

    # Locate Stage 4 sequences artifact
    try:
        sequences_path = registry.get("tokenisation", "sequences")
    except KeyError:
        logger.error(
            "No tokenisation artifact in run %s. Run data-tokenise first.",
            ctx.run_id,
        )
        sys.exit(1)

    from pipeline.training.pretrain.trainer import Trainer, TrainingRunConfig

    train_cfg = TrainingRunConfig.from_dict(cfg)

    # Allow CLI overrides for rapid experimentation
    if args.max_steps is not None:
        # TrainingRunConfig is frozen; rebuild with the override
        train_cfg = TrainingRunConfig.from_dict({
            **cfg,
            "training": {**cfg.get("training", {}), "max_steps": args.max_steps},
        })

    trainer = Trainer(
        config=train_cfg,
        run_dir=ctx.run_dir,
        sequences_path=sequences_path,
    )
    trainer.train()

    # Register output artefacts produced by the Trainer
    train_dir = ctx.run_dir / "training"
    metrics_path = train_dir / "metrics.csv"
    if metrics_path.exists():
        registry.register("training", "metrics", metrics_path)

    ckpt_dir = train_dir / "checkpoints"
    for ckpt in sorted(ckpt_dir.glob("step_*.safetensors")):
        key = ckpt.stem   # e.g. step_0010000
        registry.register("training", key, ckpt)

    logger.info("Stage 5 complete.  Run ID: %s", ctx.run_id)
    print(ctx.run_id)


def cmd_eval_checkpoint(args: argparse.Namespace) -> None:
    """Stage 6: Evaluate checkpoint(s) and write perplexity reports."""
    load_dotenv()
    _setup_logging(args.log_level)
    logger = logging.getLogger("pipeline.cli")

    cfg = _load_config(args)
    ctx = RunContext.resume(args.run_id, base_dir=Path(cfg.get("runs_dir", "runs")))
    logger.info("Run ID: %s", ctx.run_id)

    registry  = ArtifactRegistry(ctx.run_dir)
    model_cfg = cfg.get("model",    {})
    eval_cfg  = cfg.get("eval",     {})

    # ── Sequences path from Stage 4 ──────────────────────────────── #
    try:
        sequences_path = registry.get("tokenisation", "sequences")
    except KeyError:
        logger.error(
            "No tokenisation artifact in run %s. Run data-tokenise first.",
            ctx.run_id,
        )
        sys.exit(1)

    # ── Resolve checkpoints ──────────────────────────────────────── #
    if args.checkpoint:
        # Explicit path provided via --checkpoint
        checkpoint_paths = [Path(args.checkpoint)]
    else:
        # Discover from Stage 5 artifacts or scan the training/checkpoints dir
        ckpt_dir = ctx.run_dir / "training" / "checkpoints"
        mode = eval_cfg.get("checkpoints", "all")
        all_ckpts = sorted(ckpt_dir.glob("step_*.safetensors"))

        if not all_ckpts:
            # Fall back to registry
            training_artifacts = registry.list_stage("training")
            all_ckpts = sorted(
                [p for k, p in training_artifacts.items() if k.startswith("step_")],
                key=lambda p: p.stem,
            )

        if not all_ckpts:
            logger.error(
                "No checkpoint files found for run %s. Run train-pretrain first.",
                ctx.run_id,
            )
            sys.exit(1)

        if mode == "last" or mode == "latest":
            checkpoint_paths = [all_ckpts[-1]]
        else:   # "all"
            checkpoint_paths = all_ckpts

    from pipeline.evaluation.eval_runner import run_evaluation

    results = run_evaluation(
        run_dir          = ctx.run_dir,
        sequences_path   = sequences_path,
        model_cfg        = model_cfg,
        eval_cfg         = eval_cfg,
        checkpoint_paths = checkpoint_paths,
    )

    for r in results:
        registry.register(
            "evaluation",
            f"eval_report_step_{r['step']:07d}",
            ctx.run_dir / "evaluation" / f"eval_report_step_{r['step']:07d}.json",
        )
        logger.info(
            "step=%7d  ppl=%.2f  loss=%.4f",
            r["step"], r["perplexity"], r["mean_loss"],
        )

    history_path = ctx.run_dir / "evaluation" / "eval_history.json"
    if history_path.exists():
        registry.register("evaluation", "eval_history", history_path)

    logger.info("Stage 6 complete.  Run ID: %s", ctx.run_id)
    print(ctx.run_id)


def cmd_serve(args: argparse.Namespace) -> None:
    """Stage 7: Launch the Gradio inference playground."""
    load_dotenv()
    _setup_logging(args.log_level)
    logger = logging.getLogger("pipeline.cli")

    cfg = _load_config(args)
    ctx = RunContext.resume(args.run_id, base_dir=Path(cfg.get("runs_dir", "runs")))
    logger.info("Run ID: %s", ctx.run_id)

    registry     = ArtifactRegistry(ctx.run_dir)
    model_cfg    = cfg.get("model",     {})
    inf_cfg      = cfg.get("inference", {})

    # ── Resolve checkpoint ──────────────────────────────────────── #
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        # Use latest checkpoint discovered from training artifacts
        ckpt_dir  = ctx.run_dir / "training" / "checkpoints"
        all_ckpts = sorted(ckpt_dir.glob("step_*.safetensors"))
        if not all_ckpts:
            training_artifacts = registry.list_stage("training")
            all_ckpts = sorted(
                [p for k, p in training_artifacts.items() if k.startswith("step_")],
                key=lambda p: p.stem,
            )
        if not all_ckpts:
            logger.error(
                "No checkpoint files found for run %s. Run train-pretrain first.",
                ctx.run_id,
            )
            sys.exit(1)
        ckpt_path = all_ckpts[-1]   # latest by step

    logger.info("Serving checkpoint: %s", ckpt_path)

    from pipeline.inference.model_server import InferenceServer

    server = InferenceServer(
        checkpoint_path = ckpt_path,
        model_cfg       = model_cfg,
        inference_cfg   = inf_cfg,
    )
    server.load()

    port        = args.port or inf_cfg.get("port",        7860)
    server_name = inf_cfg.get("server_name", "0.0.0.0")
    share       = inf_cfg.get("share",       False)

    server.launch_gradio(port=port, server_name=server_name, share=share)


def cmd_analytics(args: argparse.Namespace) -> None:
    """Stage 8: Generate analytics figures and HTML dashboard for a run."""
    load_dotenv()
    _setup_logging(args.log_level)
    logger = logging.getLogger("pipeline.cli")

    cfg = _load_config(args)
    ctx = RunContext.resume(args.run_id, base_dir=Path(cfg.get("runs_dir", "runs")))
    logger.info("Run ID: %s", ctx.run_id)

    registry  = ArtifactRegistry(ctx.run_dir)
    figures_dir = ctx.run_dir / "analytics" / "figures"
    dash_dir    = ctx.run_dir / "analytics" / "dashboard"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # ── Training curves ──────────────────────────────────────────── #
    metrics_path = ctx.run_dir / "training" / "metrics.csv"
    if metrics_path.exists():
        from pipeline.analytics.plot_training import plot_training_metrics
        p = plot_training_metrics(metrics_path, figures_dir)
        registry.register("analytics", "training_curves", p)
        logger.info("Training curves → %s", p)
    else:
        logger.warning("No training metrics.csv found — skipping training plot")

    # ── Selection breakdown ───────────────────────────────────────── #
    selection_report = ctx.run_dir / "selection" / "selection_report.json"
    if selection_report.exists():
        from pipeline.analytics.plot_selection import plot_selection_report
        p = plot_selection_report(selection_report, figures_dir)
        registry.register("analytics", "selection_breakdown", p)
        logger.info("Selection breakdown → %s", p)
    else:
        logger.warning("No selection_report.json found — skipping selection plot")

    # ── Eval perplexity curve ─────────────────────────────────────── #
    eval_history = ctx.run_dir / "evaluation" / "eval_history.json"
    if eval_history.exists():
        from pipeline.analytics.plot_eval import plot_eval_history
        p = plot_eval_history(eval_history, figures_dir)
        registry.register("analytics", "eval_perplexity", p)
        logger.info("Eval perplexity curve → %s", p)
    else:
        logger.warning("No eval_history.json found — skipping eval plot")

    # ── HTML dashboard ────────────────────────────────────────────── #
    from pipeline.analytics.run_dashboard import build_run_dashboard
    html_path = build_run_dashboard(ctx.run_dir, dash_dir)
    registry.register("analytics", "dashboard", html_path)
    logger.info("Run dashboard → %s", html_path)

    logger.info("Stage 8 complete.  Run ID: %s", ctx.run_id)
    print(ctx.run_id)


def cmd_finalize(args: argparse.Namespace) -> None:
    """Stage 9: Finalize a run and write model_metadata.json."""
    load_dotenv()
    _setup_logging(args.log_level)
    logger = logging.getLogger("pipeline.cli")

    cfg = _load_config(args)
    ctx = RunContext.resume(args.run_id, base_dir=Path(cfg.get("runs_dir", "runs")))
    logger.info("Run ID: %s", ctx.run_id)

    registry  = ArtifactRegistry(ctx.run_dir)
    model_cfg = cfg.get("model",    {})
    train_cfg = cfg.get("training", {})

    ckpt_path: Path | None = Path(args.checkpoint) if args.checkpoint else None

    from pipeline.registry.model_registry import finalize_run
    metadata = finalize_run(
        run_dir         = ctx.run_dir,
        checkpoint_path = ckpt_path,
        model_cfg       = model_cfg,
        train_cfg       = train_cfg,
    )

    registry.register(
        "registry",
        "model_metadata",
        ctx.run_dir / "model_metadata.json",
    )
    logger.info(
        "Finalized: step=%d  checkpoint=%s",
        metadata.step,
        Path(metadata.checkpoint_path).name,
    )
    if metadata.eval_summary:
        logger.info(
            "Best eval: perplexity=%.2f  loss=%.4f  (step %d)",
            metadata.eval_summary.get("perplexity", float("nan")),
            metadata.eval_summary.get("mean_loss",  float("nan")),
            metadata.eval_summary.get("step", -1),
        )

    logger.info("Stage 9 complete.  Run ID: %s", ctx.run_id)
    print(ctx.run_id)


# ------------------------------------------------------------------ #
# Shared helpers                                                        #
# ------------------------------------------------------------------ #

def _load_config(args: argparse.Namespace) -> dict:
    loader = ConfigLoader(Path(args.config_dir))
    return loader.load(profile=args.profile)


def _resolve_run_context(args: argparse.Namespace, cfg: dict) -> RunContext:
    runs_dir = Path(cfg.get("runs_dir", "runs"))
    if getattr(args, "run_id", None):
        return RunContext.resume(args.run_id, base_dir=runs_dir)
    return RunContext.create(base_dir=runs_dir)


# ------------------------------------------------------------------ #
# Argument parser                                                       #
# ------------------------------------------------------------------ #

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="pipeline",
        description="Production ML Engineering Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-dir", default="configs", metavar="DIR",
                        help="Path to the configs/ directory  [default: configs]")
    parser.add_argument("--profile", default="local-dev", metavar="NAME",
                        help="Config profile to load  [default: local-dev]")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
                        help="Logging verbosity  [default: INFO]")

    sub = parser.add_subparsers(dest="command", required=True, metavar="STAGE")

    # -- Stage 1: data-download --
    dl = sub.add_parser("data-download", help="Stage 1: Download configured datasets")
    dl.add_argument("--run-id", default=None, metavar="RUN_ID",
                    help="Resume an existing run instead of creating a new one")
    dl.add_argument("--source", nargs="*", metavar="NAME",
                    help="Restrict download to these named source(s)")
    dl.set_defaults(func=cmd_data_download)

    # -- Stage 2: data-profile --
    prof = sub.add_parser(
        "data-profile",
        help="Stage 2: Profile downloaded datasets (requires --run-id from Stage 1)",
    )
    prof.add_argument(
        "--run-id", required=True, metavar="RUN_ID",
        help="Run ID produced by a previous data-download invocation",
    )
    prof.add_argument(
        "--source", nargs="*", metavar="NAME",
        help="Restrict profiling to these named source(s)",
    )
    prof.set_defaults(func=cmd_data_profile)

    # -- Stage 3: data-select --
    sel = sub.add_parser(
        "data-select",
        help="Stage 3: Score, filter and write the training dataset (requires --run-id from Stage 2)",
    )
    sel.add_argument(
        "--run-id", required=True, metavar="RUN_ID",
        help="Run ID that contains completed Stage 1 + Stage 2 artifacts",
    )
    sel.add_argument(
        "--source", nargs="*", metavar="NAME",
        help="Restrict selection to these named source(s)",
    )
    sel.set_defaults(func=cmd_data_select)

    # -- Stage 4: data-tokenise --
    tok = sub.add_parser(
        "data-tokenise",
        help="Stage 4: Tokenise selected dataset and write training sequences (requires --run-id from Stage 3)",
    )
    tok.add_argument(
        "--run-id", required=True, metavar="RUN_ID",
        help="Run ID that contains a completed Stage 3 selection artifact",
    )
    tok.set_defaults(func=cmd_data_tokenise)

    # -- Stage 5: train-pretrain --
    tr = sub.add_parser(
        "train-pretrain",
        help="Stage 5: Pretrain GPT on tokenised sequences (requires --run-id from Stage 4)",
    )
    tr.add_argument(
        "--run-id", required=True, metavar="RUN_ID",
        help="Run ID that contains a completed Stage 4 tokenisation artifact",
    )
    tr.add_argument(
        "--max-steps", type=int, default=None, metavar="N",
        help="Override training.max_steps from config (useful for short smoke runs)",
    )
    tr.set_defaults(func=cmd_train_pretrain)

    # -- Stage 6: eval-checkpoint --
    ev = sub.add_parser(
        "eval-checkpoint",
        help="Stage 6: Evaluate checkpoint(s) and write perplexity reports (requires --run-id from Stage 5)",
    )
    ev.add_argument(
        "--run-id", required=True, metavar="RUN_ID",
        help="Run ID that contains Stage 4 tokenisation + Stage 5 training artifacts",
    )
    ev.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help="Evaluate a specific .safetensors file instead of discovering from the run",
    )
    ev.set_defaults(func=cmd_eval_checkpoint)

    # -- Stage 7: serve --
    sv = sub.add_parser(
        "serve",
        help="Stage 7: Launch the Gradio inference playground (requires --run-id from Stage 5)",
    )
    sv.add_argument(
        "--run-id", required=True, metavar="RUN_ID",
        help="Run ID that contains Stage 5 training checkpoints",
    )
    sv.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help="Serve a specific .safetensors file instead of the latest checkpoint in the run",
    )
    sv.add_argument(
        "--port", type=int, default=None, metavar="PORT",
        help="HTTP port for the Gradio server  [default: inference.port from config]",
    )
    sv.set_defaults(func=cmd_serve)

    # -- Stage 8: analytics --
    an = sub.add_parser(
        "analytics",
        help="Stage 8: Generate training plots, selection breakdown, eval curve, and HTML dashboard",
    )
    an.add_argument(
        "--run-id", required=True, metavar="RUN_ID",
        help="Run ID to generate analytics for",
    )
    an.set_defaults(func=cmd_analytics)

    # -- Stage 9: finalize --
    fn = sub.add_parser(
        "finalize",
        help="Stage 9: Finalize a run and write model_metadata.json to the run directory",
    )
    fn.add_argument(
        "--run-id", required=True, metavar="RUN_ID",
        help="Run ID to finalize",
    )
    fn.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help="Finalize using a specific checkpoint instead of the latest one",
    )
    fn.set_defaults(func=cmd_finalize)

    return parser


# ------------------------------------------------------------------ #
# Entry point                                                           #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
