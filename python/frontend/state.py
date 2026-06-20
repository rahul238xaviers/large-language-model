"""Global state management for the GPT Pipeline UI.

This module is the single source of truth for cross-tab workflow state.

Design principles
-----------------
- All persistent state lives in a single ``PipelineState`` dataclass that is
  stored inside a ``gr.State`` component created in ``main.py`` and threaded
  through every tab's render function.
- Individual component files must **never** hold module-level mutable state.
  They receive the current state dict from Gradio event callbacks and return
  an updated copy.
- Long-running stage jobs write their output to the ``PipelineState`` once
  complete so that downstream tabs can auto-populate their inputs.

State lifecycle
---------------
  1. ``initial_state()`` — creates a fresh blank state dict.
  2. ``update_state(current, **kwargs)`` — returns a new dict with the given
     keys overwritten (immutable-update pattern; Gradio detects the new object
     and re-renders any bound ``gr.State`` consumers).
  3. Individual render functions call ``update_state`` inside their ``.click``
     / ``.change`` callbacks and pass the result back to the shared state.
"""
from __future__ import annotations

import json
import os
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from pipeline.orchestration.artifact_registry import ArtifactRegistry
from pipeline.orchestration.config_loader import ConfigLoader
from pipeline.orchestration.run_context import RunContext


# ── Stage completion sentinel values ──────────────────────────────────
STAGE_PENDING  = "pending"
STAGE_RUNNING  = "running"
STAGE_DONE     = "done"
STAGE_ERROR    = "error"

ALL_STAGES = [
    "download",      # Stage 1
    "profile",       # Stage 2
    "select",        # Stage 3
    "tokenise",      # Stage 4
    "pretrain",      # Stage 5
    "eval",          # Stage 6
    "serve",         # Stage 7
    "analytics",     # Stage 8
    "finalize",      # Stage 9
]


# ── PipelineState dataclass ────────────────────────────────────────────

@dataclass
class PipelineState:
    """Immutable-by-convention snapshot of the entire pipeline workflow."""

    # ── Active run ─────────────────────────────────────── #
    run_id:   str  = ""
    runs_dir: str  = "runs"
    profile:  str  = "local-dev"

    # ── Per-stage completion status ────────────────────── #
    stage_status: dict[str, str] = field(
        default_factory=lambda: {s: STAGE_PENDING for s in ALL_STAGES}
    )

    # ── Key artifact paths (populated after each stage) ── #
    # Stage 1
    manifest_paths: dict[str, str] = field(default_factory=dict)   # {source: manifest_path}
    # Stage 2
    profile_dir:    str            = ""
    # Stage 3
    selection_dir:  str            = ""
    arrow_path:     str            = ""
    # Stage 4
    sequences_path: str            = ""
    tokenisation_meta: dict        = field(default_factory=dict)
    # Stage 5
    training_metrics_path: str     = ""
    latest_checkpoint:     str     = ""
    # Stage 6
    eval_history_path:     str     = ""
    best_checkpoint:       str     = ""
    best_perplexity:       float   = float("inf")
    # Stage 7  (inference server — launched separately; no artifact path)
    serving_checkpoint:    str     = ""
    # Stage 8
    dashboard_path:        str     = ""
    # Stage 9
    metadata_path:         str     = ""

    # ── Resolved config (merged YAML + env) ────────────── #
    config: dict = field(default_factory=dict)

    # ── UI-only ephemeral fields (not persisted) ───────── #
    active_tab: int = 0   # 0=data  1=train  2=finetune  3=analytics  4=inference

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineState":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})


# ── Public helpers ─────────────────────────────────────────────────────

def initial_state() -> dict:
    """Return the blank pipeline state as a plain dict (required by gr.State)."""
    cfg = ConfigLoader().load("local-dev")
    state = PipelineState(
        config   = cfg,
        runs_dir = str(cfg.get("runs_dir", "runs")),
        profile  = "local-dev",
    )
    return state.to_dict()


def update_state(current: dict, **kwargs: Any) -> dict:
    """Return a new state dict with the given fields overwritten.

    Usage inside a Gradio callback::

        new_state = update_state(state, run_id="run_20260529_...", stage_status={...})
        return new_state, gr.update(value=new_state["run_id"])
    """
    updated = deepcopy(current)
    updated.update(kwargs)
    return updated


def mark_stage(state: dict, stage: str, status: str) -> dict:
    """Convenience wrapper to update a single stage status."""
    new_statuses = deepcopy(state.get("stage_status", {}))
    new_statuses[stage] = status
    return update_state(state, stage_status=new_statuses)


# ── Run context helpers ────────────────────────────────────────────────

def create_run(state: dict) -> tuple[dict, RunContext]:
    """Create a new RunContext and update the state with the new run_id."""
    ctx = RunContext.create(base_dir=Path(state.get("runs_dir", "runs")))
    new_state = update_state(state, run_id=ctx.run_id)
    return new_state, ctx


def resume_run(state: dict) -> RunContext:
    """Resume the RunContext recorded in *state*.  Raises if run_id is empty."""
    run_id = state.get("run_id", "")
    if not run_id:
        raise ValueError("No active run_id in state. Start a new run first.")
    return RunContext.resume(run_id, base_dir=Path(state.get("runs_dir", "runs")))


def get_registry(state: dict) -> ArtifactRegistry:
    """Return the ArtifactRegistry for the current run."""
    ctx = resume_run(state)
    return ArtifactRegistry(ctx.run_dir)


# ── Existing runs discovery ────────────────────────────────────────────

def list_runs(runs_dir: str = "runs") -> list[str]:
    """Return a sorted list of existing run IDs (newest first)."""
    base = Path(runs_dir)
    if not base.exists():
        return []
    return sorted(
        [d.name for d in base.iterdir() if d.is_dir() and d.name.startswith("run_")],
        reverse=True,
    )


def runs_with_metadata(runs_dir: str = "runs") -> list[dict]:
    """Return a list of dicts for runs that have ``model_metadata.json``."""
    result = []
    for run_id in list_runs(runs_dir):
        meta_path = Path(runs_dir) / run_id / "model_metadata.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text())
                result.append({"run_id": run_id, **meta})
            except Exception:
                pass
    return result


# ── Progress HTML helper ───────────────────────────────────────────────

def progress_html(stage_status: dict[str, str]) -> str:
    """Render a small HTML progress-dot bar from the current stage_status dict."""
    dots = []
    for stage in ALL_STAGES:
        s = stage_status.get(stage, STAGE_PENDING)
        cls = {"done": "done", "running": "active"}.get(s, "")
        title = f"{stage}: {s}"
        dots.append(f'<div class="progress-dot {cls}" title="{title}"></div>')
    inner = "\n".join(dots)
    return f'<div class="progress-track">{inner}</div>'
