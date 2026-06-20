"""Evaluation perplexity-curve plot.

Reads ``eval_history.json`` written by Stage 6 and produces a line plot
of perplexity (and mean loss) versus checkpoint step.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_COLOR_PPL  = "#f97316"   # orange — perplexity
_COLOR_LOSS = "#8b5cf6"   # indigo — mean loss
_GRID_COLOR = "#292524"


def plot_eval_history(eval_history_path: Path | str, output_dir: Path | str) -> Path:
    """Read ``eval_history.json`` and write ``eval_perplexity.png``.

    Returns the path of the saved figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for analytics plots. "
            "Install it with: pip install matplotlib"
        ) from exc

    eval_history_path = Path(eval_history_path)
    output_dir        = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not eval_history_path.exists():
        raise FileNotFoundError(f"Eval history not found: {eval_history_path}")

    history: list[dict] = json.loads(eval_history_path.read_text())
    if not history:
        raise ValueError("eval_history.json is empty — nothing to plot")

    steps       = [r["step"]        for r in history]
    perplexity  = [r["perplexity"]  for r in history]
    mean_loss   = [r["mean_loss"]   for r in history]

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Checkpoint Evaluation", fontsize=14, fontweight="bold", color="#f97316")

    # ── Left: perplexity ─────────────────────────────────────────── #
    ax = axes[0]
    ax.plot(steps, perplexity, color=_COLOR_PPL, linewidth=2.0,
            marker="o", markersize=5, label="Perplexity")
    best_idx = perplexity.index(min(perplexity))
    ax.axvline(steps[best_idx], color=_COLOR_PPL, linestyle=":", alpha=0.6)
    ax.annotate(
        f"Best: {perplexity[best_idx]:.2f}",
        xy=(steps[best_idx], perplexity[best_idx]),
        xytext=(10, 10), textcoords="offset points",
        color=_COLOR_PPL, fontsize=9,
        arrowprops={"arrowstyle": "->", "color": _COLOR_PPL, "lw": 0.8},
    )
    ax.set_title("Perplexity vs Checkpoint Step", fontsize=12, pad=8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity")
    ax.grid(True, linestyle="--", color=_GRID_COLOR, alpha=0.7)
    ax.legend(frameon=True, facecolor="#1c1917", edgecolor="#44403c")

    # ── Right: mean loss ─────────────────────────────────────────── #
    ax2 = axes[1]
    ax2.plot(steps, mean_loss, color=_COLOR_LOSS, linewidth=2.0,
             marker="s", markersize=5, label="Mean CE loss")
    ax2.set_title("Mean Cross-Entropy Loss", fontsize=12, pad=8)
    ax2.set_xlabel("Step")
    ax2.set_ylabel("Loss")
    ax2.grid(True, linestyle="--", color=_GRID_COLOR, alpha=0.7)
    ax2.legend(frameon=True, facecolor="#1c1917", edgecolor="#44403c")

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    out = output_dir / "eval_perplexity.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0c0a09")
    plt.close(fig)
    logger.info("Saved eval perplexity curve → %s", out)
    return out
