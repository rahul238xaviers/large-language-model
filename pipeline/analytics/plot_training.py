"""Training metrics plot — 4-panel dark-theme figure.

Reads the ``metrics.csv`` written by the Stage 5 trainer and produces:
  - Loss convergence
  - Token throughput
  - VRAM utilisation
  - Learning-rate schedule
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# Column names written by pipeline/training/pretrain/trainer.py
_EXPECTED_COLUMNS = {"step", "train_loss", "learning_rate"}
_OPTIONAL_COLUMNS = {"tokens_per_sec", "vram_usage_gb", "mfu_pct"}

# Dark-theme palette (consistent with legacy plot_journey.py)
_COLORS = {
    "loss":       "#f97316",   # Rust orange
    "throughput": "#10b981",   # Teal green
    "vram":       "#ef4444",   # Crimson
    "lr":         "#8b5cf6",   # Indigo
    "avg":        "#64748b",   # Slate
    "grid":       "#292524",   # Dark stone
}


def plot_training_metrics(metrics_path: Path | str, output_dir: Path | str) -> Path:
    """Read ``metrics.csv`` and write ``training_curves.png`` to *output_dir*.

    Returns the path of the saved figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")          # non-interactive backend — safe in headless envs
        import matplotlib.pyplot as plt
        import pandas as pd
    except ImportError as exc:
        raise ImportError(
            "matplotlib and pandas are required for analytics plots. "
            "Install them with: pip install matplotlib pandas"
        ) from exc

    metrics_path = Path(metrics_path)
    output_dir   = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")

    df = pd.read_csv(metrics_path)
    missing = _EXPECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"metrics.csv is missing expected columns: {missing}")

    has_tp   = "tokens_per_sec" in df.columns
    has_vram = "vram_usage_gb"  in df.columns
    has_mfu  = "mfu_pct"        in df.columns

    run_name = metrics_path.parent.parent.name  # e.g. run_20260529_...

    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Training Metrics: {run_name}",
        fontsize=15, fontweight="bold", color=_COLORS["loss"],
    )

    # ── 1. Loss ────────────────────────────────────────────────────── #
    ax = axes[0, 0]
    ax.plot(df["step"], df["train_loss"], color=_COLORS["loss"], linewidth=1.5, label="Train loss")
    ax.set_title("Loss Convergence", fontsize=12, fontweight="semibold", pad=8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(True, linestyle="--", color=_COLORS["grid"], alpha=0.7)
    ax.legend(frameon=True, facecolor="#1c1917", edgecolor="#44403c")

    # ── 2. Throughput ─────────────────────────────────────────────── #
    ax = axes[0, 1]
    if has_tp:
        ax.plot(df["step"], df["tokens_per_sec"], color=_COLORS["throughput"], linewidth=1.2, label="tok/s")
        avg = df["tokens_per_sec"].mean()
        ax.axhline(avg, color=_COLORS["avg"], linestyle=":", label=f"Avg {avg:.0f} tok/s")
        ax.legend(frameon=True, facecolor="#1c1917", edgecolor="#44403c")
    else:
        ax.text(0.5, 0.5, "tokens_per_sec\nnot logged",
                ha="center", va="center", transform=ax.transAxes, color=_COLORS["avg"])
    ax.set_title("Throughput", fontsize=12, fontweight="semibold", pad=8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Tokens / second")
    ax.grid(True, linestyle="--", color=_COLORS["grid"], alpha=0.7)

    # ── 3. VRAM ───────────────────────────────────────────────────── #
    ax = axes[1, 0]
    if has_vram:
        ax.plot(df["step"], df["vram_usage_gb"], color=_COLORS["vram"], linewidth=2.0, label="VRAM usage")
        ax.axhline(512.0, color="#b91c1c", linestyle="--", linewidth=1.2, label="M3 Ultra ceiling (512 GB)")
        ax.legend(frameon=True, facecolor="#1c1917", edgecolor="#44403c")
    elif has_mfu:
        ax.plot(df["step"], df["mfu_pct"], color=_COLORS["vram"], linewidth=1.2, label="MFU %")
        ax.set_ylabel("MFU (%)")
        ax.legend(frameon=True, facecolor="#1c1917", edgecolor="#44403c")
    else:
        ax.text(0.5, 0.5, "vram_usage_gb\nnot logged",
                ha="center", va="center", transform=ax.transAxes, color=_COLORS["avg"])
    ax.set_title("Memory / Compute Utilisation", fontsize=12, fontweight="semibold", pad=8)
    ax.set_xlabel("Step")
    ax.grid(True, linestyle="--", color=_COLORS["grid"], alpha=0.7)

    # ── 4. Learning rate ─────────────────────────────────────────── #
    ax = axes[1, 1]
    ax.plot(df["step"], df["learning_rate"], color=_COLORS["lr"], linewidth=1.2, label="LR")
    ax.set_title("Learning-Rate Schedule", fontsize=12, fontweight="semibold", pad=8)
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning rate")
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))
    ax.grid(True, linestyle="--", color=_COLORS["grid"], alpha=0.7)
    ax.legend(frameon=True, facecolor="#1c1917", edgecolor="#44403c")

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    output_path = output_dir / "training_curves.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved training curves → %s", output_path)
    return output_path
