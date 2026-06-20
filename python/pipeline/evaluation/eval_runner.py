"""Stage 6 — evaluation orchestrator.

Resolves which checkpoints to evaluate, loads each one, runs perplexity
computation, and writes a JSON report per checkpoint plus a summary
``eval_history.json`` that accumulates results across all evaluated steps.

Output layout (within ``runs/<run_id>/evaluation/``):
    eval_report_step_0000500.json    — single-checkpoint report
    eval_report_step_0001000.json
    ...
    eval_history.json               — all steps ordered by step number

Public interface:
    run_evaluation(
        run_dir, sequences_path, model_cfg, eval_cfg,
        checkpoint_paths          # list[Path]
    ) → list[dict]   (one dict per evaluated checkpoint)
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _step_from_path(path: Path) -> int:
    """Extract the integer step number from a ``step_NNNNNNN.safetensors`` filename."""
    stem = path.stem  # e.g. "step_0000500"
    try:
        return int(stem.split("_")[-1])
    except (ValueError, IndexError):
        return 0


def run_evaluation(
    run_dir: Path,
    sequences_path: Path,
    model_cfg: dict,
    eval_cfg: dict,
    checkpoint_paths: list[Path],
) -> list[dict]:
    """Evaluate one or more checkpoints and persist JSON reports.

    Args:
        run_dir:           Pipeline run directory root.
        sequences_path:    Path to ``sequences.npy`` from Stage 4.
        model_cfg:         Model architecture config dict.
        eval_cfg:          Evaluation config dict (``eval_fraction``, etc.).
        checkpoint_paths:  Ordered list of ``.safetensors`` files to evaluate.

    Returns:
        List of result dicts (one per checkpoint), ordered by step number.
        Each dict contains all ``EvalResult`` fields plus ``step`` and
        ``checkpoint``.
    """
    from pipeline.evaluation.checkpoint_loader import load_checkpoint
    from pipeline.evaluation.perplexity import compute_perplexity

    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir(parents=True, exist_ok=True)

    # Sort by step number so reports are written in order
    ordered = sorted(checkpoint_paths, key=_step_from_path)
    all_results: list[dict] = []

    for ckpt_path in ordered:
        step = _step_from_path(ckpt_path)
        logger.info(
            "Evaluating checkpoint step=%d  (%s)", step, ckpt_path.name
        )

        model       = load_checkpoint(ckpt_path, model_cfg)
        eval_result = compute_perplexity(model, sequences_path, eval_cfg)

        result = {
            "step":       step,
            "checkpoint": str(ckpt_path),
            **eval_result.to_dict(),
        }
        all_results.append(result)

        # Per-checkpoint report
        report_path = eval_dir / f"eval_report_step_{step:07d}.json"
        report_path.write_text(json.dumps(result, indent=2))
        logger.info(
            "  → ppl=%.2f  loss=%.4f  written: %s",
            result["perplexity"], result["mean_loss"], report_path.name,
        )

    # Cumulative history (append-safe: re-read existing history if present)
    history_path = eval_dir / "eval_history.json"
    existing: list[dict] = []
    if history_path.exists():
        try:
            existing = json.loads(history_path.read_text())
        except (json.JSONDecodeError, OSError):
            existing = []

    # Merge: replace entries with same step, keep others
    existing_by_step = {r["step"]: r for r in existing}
    for r in all_results:
        existing_by_step[r["step"]] = r

    history = sorted(existing_by_step.values(), key=lambda r: r["step"])
    history_path.write_text(json.dumps(history, indent=2))
    logger.info("Eval history updated: %d entries → %s", len(history), history_path)

    return all_results
