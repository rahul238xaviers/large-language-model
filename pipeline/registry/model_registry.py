"""Model registry — run finalization and metadata lineage.

``finalize_run`` is the terminal operation on a training run.  It:
1. Picks the best or specified checkpoint.
2. Summarises eval history (best perplexity step).
3. Writes ``model_metadata.json`` to the run directory.
4. Returns a ``ModelMetadata`` object.

The metadata file is consumed by the Stage 7 inference server and by the
Stage 8 dashboard, making checkpoint selection entirely config-driven.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

METADATA_FILENAME = "model_metadata.json"


# ------------------------------------------------------------------ #
# Data model                                                            #
# ------------------------------------------------------------------ #

@dataclass
class ModelMetadata:
    """Canonical lineage record for a finalized training run."""
    run_id:           str
    checkpoint_path:  str
    step:             int
    model_cfg:        dict  = field(default_factory=dict)
    train_cfg:        dict  = field(default_factory=dict)
    eval_summary:     dict  = field(default_factory=dict)   # best-step eval metrics
    created_at:       str   = ""
    pipeline_version: str   = "1.0"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelMetadata":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# ------------------------------------------------------------------ #
# Helpers                                                               #
# ------------------------------------------------------------------ #

def _step_from_path(p: Path) -> int:
    """Extract integer step from a ``step_NNNNNNN.safetensors`` filename."""
    stem = p.stem  # e.g. step_0001000
    return int(stem.split("_")[-1])


def _best_eval(eval_history: list[dict]) -> dict:
    """Return the eval record with the lowest perplexity."""
    if not eval_history:
        return {}
    return min(eval_history, key=lambda r: r.get("perplexity", float("inf")))


# ------------------------------------------------------------------ #
# Public API                                                            #
# ------------------------------------------------------------------ #

def finalize_run(
    run_dir:            Path | str,
    checkpoint_path:    Path | str | None = None,
    model_cfg:          dict | None = None,
    train_cfg:          dict | None = None,
) -> ModelMetadata:
    """Finalize a training run and write ``model_metadata.json``.

    If *checkpoint_path* is *None*, the latest checkpoint in
    ``<run_dir>/training/checkpoints/`` is used.

    Returns the ``ModelMetadata`` instance written to disk.
    """
    run_dir = Path(run_dir)
    model_cfg = model_cfg or {}
    train_cfg = train_cfg or {}

    # ── Resolve checkpoint ─────────────────────────────────────────── #
    if checkpoint_path is None:
        ckpt_dir  = run_dir / "training" / "checkpoints"
        all_ckpts = sorted(ckpt_dir.glob("step_*.safetensors"))
        if not all_ckpts:
            raise FileNotFoundError(
                f"No checkpoint files found in {ckpt_dir}. "
                "Run train-pretrain first."
            )
        checkpoint_path = all_ckpts[-1]

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    step = _step_from_path(checkpoint_path)

    # ── Load eval history (optional) ──────────────────────────────── #
    eval_history_path = run_dir / "evaluation" / "eval_history.json"
    eval_history: list[dict] = []
    if eval_history_path.exists():
        try:
            eval_history = json.loads(eval_history_path.read_text())
        except Exception as exc:
            logger.warning("Could not read eval_history.json: %s", exc)

    eval_summary = _best_eval(eval_history)

    # ── Build metadata ─────────────────────────────────────────────── #
    metadata = ModelMetadata(
        run_id          = run_dir.name,
        checkpoint_path = str(checkpoint_path),
        step            = step,
        model_cfg       = model_cfg,
        train_cfg       = train_cfg,
        eval_summary    = eval_summary,
        created_at      = datetime.now(timezone.utc).isoformat(),
    )

    # ── Write to disk ─────────────────────────────────────────────── #
    out_path = run_dir / METADATA_FILENAME
    out_path.write_text(json.dumps(metadata.to_dict(), indent=2), encoding="utf-8")
    logger.info(
        "Run finalized. step=%d  checkpoint=%s  metadata→%s",
        step, checkpoint_path.name, out_path,
    )
    return metadata


def load_metadata(metadata_path: Path | str) -> ModelMetadata:
    """Load a ``model_metadata.json`` file and return a ``ModelMetadata`` object."""
    path = Path(metadata_path)
    if not path.exists():
        raise FileNotFoundError(f"model_metadata.json not found: {path}")
    return ModelMetadata.from_dict(json.loads(path.read_text()))
