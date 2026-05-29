"""Data-selection breakdown plot.

Reads the ``selection_report.json`` written by Stage 3 and produces a
horizontal stacked-bar chart showing kept / dropped document counts per
source, plus a drop-reason breakdown panel.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_KEEP_COLOR = "#10b981"   # teal
_DROP_COLOR = "#ef4444"   # red
_GRID_COLOR = "#292524"


def plot_selection_report(report_path: Path | str, output_dir: Path | str) -> Path:
    """Read ``selection_report.json`` and write ``selection_breakdown.png``.

    Returns the path of the saved figure.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "matplotlib is required for analytics plots. "
            "Install it with: pip install matplotlib"
        ) from exc

    report_path = Path(report_path)
    output_dir  = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not report_path.exists():
        raise FileNotFoundError(f"Selection report not found: {report_path}")

    report: dict = json.loads(report_path.read_text())
    # Expected schema (from pipeline/data/selection/dataset_writer.py):
    # { "sources": { "<name>": { "total": N, "kept": K, "dropped": D, "keep_rate_pct": ...,
    #                             "drop_reasons": { reason: count } }, ... } }
    sources_data: dict = report.get("sources", {})

    if not sources_data:
        logger.warning("selection_report.json has no 'sources' key — nothing to plot")
        # Write a blank placeholder
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "No selection data available",
                ha="center", va="center", transform=ax.transAxes, color="#a8a29e")
        ax.axis("off")
        out = output_dir / "selection_breakdown.png"
        plt.savefig(out, dpi=100, bbox_inches="tight", facecolor="#0c0a09")
        plt.close(fig)
        return out

    names  = list(sources_data.keys())
    kept   = [sources_data[n].get("kept",    0) for n in names]
    dropped = [sources_data[n].get("dropped", 0) for n in names]

    # Collect all distinct drop reasons across all sources
    all_reasons: dict[str, list[int]] = {}
    for n in names:
        for reason, cnt in sources_data[n].get("drop_reasons", {}).items():
            all_reasons.setdefault(reason, [0] * len(names))
            all_reasons[reason][names.index(n)] = cnt

    n_panels = 2 if all_reasons else 1
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, max(3, len(names) * 0.8 + 2)))
    if n_panels == 1:
        axes = [axes]

    fig.suptitle("Data Selection Breakdown", fontsize=14, fontweight="bold", color="#f97316")

    # ── Left panel: kept vs dropped per source ───────────────────── #
    ax = axes[0]
    y  = range(len(names))
    ax.barh(y, kept,    color=_KEEP_COLOR, label="Kept",    height=0.5)
    ax.barh(y, dropped, left=kept, color=_DROP_COLOR, label="Dropped", height=0.5)
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=10)
    ax.set_xlabel("Documents")
    ax.set_title("Kept vs Dropped per Source", fontsize=12, pad=8)
    ax.grid(True, axis="x", linestyle="--", color=_GRID_COLOR, alpha=0.7)
    ax.legend(frameon=True, facecolor="#1c1917", edgecolor="#44403c")
    # Keep rate annotations
    for i, (k, d) in enumerate(zip(kept, dropped)):
        total = k + d
        if total > 0:
            pct = 100 * k / total
            ax.text(total + max(total * 0.01, 1), i, f"{pct:.0f}%",
                    va="center", fontsize=8, color="#a8a29e")

    # ── Right panel: drop reasons (if any) ───────────────────────── #
    if all_reasons:
        ax2 = axes[1]
        reason_names  = list(all_reasons.keys())
        reason_totals = [sum(v) for v in all_reasons.values()]
        colors = plt.cm.Reds(  # type: ignore[attr-defined]
            [0.4 + 0.5 * (i / max(len(reason_names) - 1, 1)) for i in range(len(reason_names))]
        )
        ax2.barh(range(len(reason_names)), reason_totals, color=colors, height=0.5)
        ax2.set_yticks(list(range(len(reason_names))))
        ax2.set_yticklabels(reason_names, fontsize=9)
        ax2.set_xlabel("Documents dropped")
        ax2.set_title("Drop Reasons (all sources)", fontsize=12, pad=8)
        ax2.grid(True, axis="x", linestyle="--", color=_GRID_COLOR, alpha=0.7)

    plt.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    out = output_dir / "selection_breakdown.png"
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0c0a09")
    plt.close(fig)
    logger.info("Saved selection breakdown → %s", out)
    return out
