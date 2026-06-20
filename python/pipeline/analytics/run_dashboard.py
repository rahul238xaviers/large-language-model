"""Self-contained HTML run dashboard.

Aggregates all available run artifacts and writes a single
``run_summary.html`` that can be opened in any browser without a server.

Embedded figures (PNG) are base64-encoded so the file is fully portable.
"""
from __future__ import annotations

import base64
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Helpers ────────────────────────────────────────────────────────── #

def _img_tag(path: Path) -> str:
    """Return an ``<img>`` tag with the PNG embedded as base64."""
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{data}" style="max-width:100%;border-radius:8px;">'


def _load_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _safe_csv_summary(metrics_path: Path) -> dict:
    """Return a lightweight dict summary from metrics.csv."""
    try:
        import pandas as pd
        df = pd.read_csv(metrics_path)
        summary: dict[str, Any] = {
            "steps_logged": len(df),
            "first_step":   int(df["step"].iloc[0])  if "step"        in df.columns else None,
            "last_step":    int(df["step"].iloc[-1]) if "step"        in df.columns else None,
            "final_loss":   round(float(df["train_loss"].iloc[-1]), 4)
                            if "train_loss" in df.columns else None,
            "min_loss":     round(float(df["train_loss"].min()), 4)
                            if "train_loss" in df.columns else None,
        }
        if "tokens_per_sec" in df.columns:
            summary["avg_tokens_per_sec"] = round(float(df["tokens_per_sec"].mean()), 1)
        return summary
    except Exception:
        return {}


# ── Public API ─────────────────────────────────────────────────────── #

def build_run_dashboard(run_dir: Path | str, output_dir: Path | str) -> Path:
    """Aggregate all run artifacts and write a self-contained HTML dashboard.

    Figures that already exist in *output_dir* (created by ``plot_training``,
    ``plot_selection``, ``plot_eval``) are embedded automatically.

    Returns the path to the written HTML file.
    """
    run_dir    = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id      = run_dir.name
    generated   = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    # ── Gather data ────────────────────────────────────────────────── #
    registry_data  = _load_json(run_dir / "artifact_registry.json")
    eval_history   = _load_json(run_dir / "evaluation" / "eval_history.json") or []
    metadata       = _load_json(run_dir / "model_metadata.json")
    tokenise_meta  = _load_json(run_dir / "tokenisation" / "tokenisation_meta.json")
    selection_rpt  = _load_json(run_dir / "selection" / "selection_report.json")

    metrics_path = run_dir / "training" / "metrics.csv"
    train_summary = _safe_csv_summary(metrics_path) if metrics_path.exists() else {}

    # ── Embedded figures ───────────────────────────────────────────── #
    # Figures are always written by the CLI to run_dir/analytics/figures/
    figures_dir = run_dir / "analytics" / "figures"
    figure_tags: dict[str, str] = {}
    for stem in ("training_curves", "selection_breakdown", "eval_perplexity"):
        p = figures_dir / f"{stem}.png"
        if p.exists():
            figure_tags[stem] = _img_tag(p)

    # ── Eval summary table ─────────────────────────────────────────── #
    eval_rows = ""
    if eval_history:
        best_step = min(eval_history, key=lambda r: r.get("perplexity", float("inf")))
        for r in eval_history:
            bold = ' style="color:#f97316;font-weight:bold;"' if r is best_step else ""
            eval_rows += (
                f"<tr{bold}>"
                f"<td>{r.get('step','—')}</td>"
                f"<td>{r.get('perplexity','—')}</td>"
                f"<td>{r.get('mean_loss','—')}</td>"
                f"<td>{r.get('n_sequences','—')}</td>"
                f"</tr>\n"
            )

    # ── Data pipeline summary ──────────────────────────────────────── #
    data_rows = ""
    if tokenise_meta:
        for k, v in tokenise_meta.items():
            data_rows += f"<tr><td>{k}</td><td>{v}</td></tr>\n"
    if selection_rpt:
        src_data = selection_rpt.get("sources", {})
        for src, stats in src_data.items():
            keep_rate = stats.get("keep_rate_pct", "—")
            data_rows += (
                f"<tr><td>selection:{src}</td>"
                f"<td>kept={stats.get('kept','—')}  "
                f"dropped={stats.get('dropped','—')}  "
                f"rate={keep_rate}%</td></tr>\n"
            )

    # ── Registry summary ───────────────────────────────────────────── #
    registry_rows = ""
    if registry_data:
        for stage, artifacts in registry_data.items():
            for name, path in artifacts.items():
                registry_rows += (
                    f"<tr><td>{stage}</td><td>{name}</td>"
                    f"<td style='font-size:0.78em;color:#a8a29e;'>{path}</td></tr>\n"
                )

    # ── Model metadata box ─────────────────────────────────────────── #
    meta_html = ""
    if metadata:
        meta_items = "".join(
            f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
            for k, v in metadata.items()
            if k not in ("model_cfg", "train_cfg")
        )
        meta_html = f"""
        <section>
          <h2>Model Metadata</h2>
          <table>{meta_items}</table>
        </section>"""

    # ── Training summary box ───────────────────────────────────────── #
    train_html = ""
    if train_summary:
        train_items = "".join(
            f"<tr><td><b>{k}</b></td><td>{v}</td></tr>"
            for k, v in train_summary.items()
        )
        train_html = f"""
        <section>
          <h2>Training Summary</h2>
          <table>{train_items}</table>
        </section>"""

    # ── Assemble HTML ─────────────────────────────────────────────── #
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Run Dashboard — {run_id}</title>
  <style>
    :root {{
      --bg:      #0c0a09;
      --surface: #1c1917;
      --border:  #292524;
      --accent:  #f97316;
      --muted:   #a8a29e;
      --text:    #e7e5e4;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      background: var(--bg);
      color: var(--text);
      font-family: "Outfit", "Segoe UI", sans-serif;
      padding: 2rem 4vw;
      line-height: 1.6;
    }}
    h1 {{ color: var(--accent); font-size: 1.8rem; margin-bottom: 0.25rem; }}
    h2 {{ color: var(--accent); font-size: 1.1rem; margin: 1.5rem 0 0.5rem; border-bottom: 1px solid var(--border); padding-bottom: 4px; }}
    .subtitle {{ color: var(--muted); font-size: 0.9rem; margin-bottom: 2rem; }}
    section {{ margin-bottom: 2.5rem; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
    th, td {{ padding: 6px 10px; border: 1px solid var(--border); text-align: left; }}
    th {{ background: var(--surface); color: var(--accent); font-weight: 600; }}
    tr:nth-child(even) {{ background: #14100e; }}
    .figures {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(500px, 1fr)); gap: 1.5rem; }}
    .fig-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 10px; padding: 1rem; }}
    .fig-card h3 {{ font-size: 0.9rem; color: var(--muted); margin-bottom: 0.5rem; }}
    footer {{ margin-top: 3rem; color: var(--muted); font-size: 0.8rem; text-align: center; }}
  </style>
</head>
<body>
  <h1>Run Dashboard</h1>
  <p class="subtitle">Run ID: <code>{run_id}</code> &nbsp;|&nbsp; Generated: {generated}</p>

  {"<!-- model metadata -->" + meta_html if meta_html else ""}

  {"<!-- training summary -->" + train_html if train_html else ""}

  {"<!-- data pipeline -->" + ("""
  <section>
    <h2>Data Pipeline</h2>
    <table>
      <tr><th>Key</th><th>Value</th></tr>
      """ + data_rows + """
    </table>
  </section>""") if data_rows else ""}

  {"<!-- eval history -->" + ("""
  <section>
    <h2>Evaluation History</h2>
    <table>
      <tr><th>Step</th><th>Perplexity</th><th>Mean Loss</th><th>Sequences evaluated</th></tr>
      """ + eval_rows + """
    </table>
  </section>""") if eval_rows else ""}

  {"<!-- figures -->" + ("""
  <section>
    <h2>Figures</h2>
    <div class="figures">
      """ + "".join(
          f'<div class="fig-card"><h3>{stem.replace("_"," ").title()}</h3>{tag}</div>'
          for stem, tag in figure_tags.items()
      ) + """
    </div>
  </section>""") if figure_tags else ""}

  {"<!-- artifact registry -->" + ("""
  <section>
    <h2>Artifact Registry</h2>
    <table>
      <tr><th>Stage</th><th>Artifact</th><th>Path</th></tr>
      """ + registry_rows + """
    </table>
  </section>""") if registry_rows else ""}

  <footer>Generated by pipeline/analytics/run_dashboard.py</footer>
</body>
</html>
"""

    out = output_dir / "run_summary.html"
    out.write_text(html, encoding="utf-8")
    logger.info("Saved run dashboard → %s", out)
    return out
