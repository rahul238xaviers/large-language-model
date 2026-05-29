"""Stages 6 + 8: Checkpoint Evaluation and Analytics Workbench tab.

Stage 6 (eval-checkpoint): runs perplexity evaluation on one or all checkpoints,
streams the CLI log, then renders the eval history as an HTML table with
summary metric cards.

Stage 8 (analytics): generates all four matplotlib figures and the self-contained
HTML run dashboard; renders the PNGs inline and provides a file-path link to the
dashboard.

Public symbol: ``render_analytics_tab(state)``
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Iterator

import gradio as gr

from frontend.state import (
    STAGE_DONE,
    STAGE_ERROR,
    STAGE_RUNNING,
    mark_stage,
    update_state,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]


# ── Shared subprocess streamer ─────────────────────────────────────────

def _stream_pipeline(
    cmd: list[str],
    env: dict | None = None,
) -> Iterator[tuple[str, int | None]]:
    merged_env = {**os.environ, **(env or {})}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(_REPO_ROOT),
        env=merged_env,
    )
    output = ""
    assert proc.stdout is not None
    for line in proc.stdout:
        output += line
        yield output, None
    proc.wait()
    yield output, proc.returncode


def _badge(label: str, variant: str = "") -> str:
    cls = f"stage-badge {variant}".strip()
    return f"<span class='{cls}'>{label}</span>"


# ── Checkpoint discovery ───────────────────────────────────────────────

def _discover_checkpoints(run_id: str, runs_dir: str) -> list[str]:
    """Return sorted list of step_*.safetensors names under a run's training/checkpoints/."""
    if not run_id:
        return []
    ckpt_dir = _REPO_ROOT / runs_dir / run_id / "training" / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted([f.name for f in ckpt_dir.glob("step_*.safetensors")])


def _checkpoint_choices(run_id: str, runs_dir: str) -> list[str]:
    ckpts = _discover_checkpoints(run_id, runs_dir)
    if not ckpts:
        return ["all (auto-discover)"]
    return ["all (auto-discover)", "latest only"] + ckpts


# ── Stage 6 callback ───────────────────────────────────────────────────

def _on_eval(
    checkpoint_choice: str,
    eval_fraction: float,
    eval_batch_size: int,
    profile: str,
    state: dict,
) -> Iterator[tuple]:
    run_id = state.get("run_id", "")
    if not run_id:
        yield (
            "⚠  No active run. Complete Stage 5 (Pre-train) first.",
            "<span class='status-error'>●  Error</span>",
            state,
            _eval_placeholder(),
        )
        return

    cmd = [
        sys.executable, "-m", "pipeline",
        "--profile", profile,
        "eval-checkpoint",
        "--run-id", run_id,
    ]
    # Pass specific checkpoint if user selected one
    if checkpoint_choice not in ("all (auto-discover)", "latest only"):
        runs_dir = state.get("runs_dir", "runs")
        ckpt_path = str(
            _REPO_ROOT / runs_dir / run_id / "training" / "checkpoints" / checkpoint_choice
        )
        cmd += ["--checkpoint", ckpt_path]
    elif checkpoint_choice == "latest only":
        # CLI picks the latest when no --checkpoint flag — we signal via env
        cmd += ["--checkpoint", ""]   # will be ignored; use env override instead

    env = {
        "PIPELINE_EVAL__EVAL_FRACTION":  str(eval_fraction),
        "PIPELINE_EVAL__EVAL_BATCH_SIZE": str(int(eval_batch_size)),
    }
    if checkpoint_choice == "latest only":
        env["PIPELINE_EVAL__CHECKPOINTS"] = "last"

    new_state    = mark_stage(state, "eval", STAGE_RUNNING)
    running_html = "<span class='status-running'>⟳  Evaluating…</span>"
    runs_dir     = state.get("runs_dir", "runs")
    history_path = _REPO_ROOT / runs_dir / run_id / "evaluation" / "eval_history.json"

    for output, returncode in _stream_pipeline(cmd, env=env):
        if returncode is None:
            yield output, running_html, new_state, _eval_placeholder()
        else:
            stage_st = STAGE_DONE if returncode == 0 else STAGE_ERROR
            new_state = mark_stage(
                update_state(new_state, eval_history_path=str(history_path)),
                "eval", stage_st,
            )
            status_html = (
                "<span class='status-success'>✓  Evaluation complete</span>"
                if returncode == 0
                else f"<span class='status-error'>✗  Exit {returncode}</span>"
            )
            suffix = (
                "\n\n[Stage 6 complete]"
                if returncode == 0
                else f"\n\n[Stage 6 failed — exit code {returncode}]"
            )
            eval_html = _render_eval_history(history_path) if returncode == 0 else _eval_placeholder()
            yield output + suffix, status_html, new_state, eval_html


# ── Stage 8 callback ───────────────────────────────────────────────────

def _on_analytics(
    profile: str,
    state: dict,
) -> Iterator[tuple]:
    run_id = state.get("run_id", "")
    if not run_id:
        yield (
            "⚠  No active run. Complete Stage 5 (Pre-train) first.",
            "<span class='status-error'>●  Error</span>",
            state,
            None, None, None,   # three gr.Image outputs
            "<span style='color:#a8a29e;'>No dashboard yet.</span>",
        )
        return

    cmd = [
        sys.executable, "-m", "pipeline",
        "--profile", profile,
        "analytics",
        "--run-id", run_id,
    ]

    new_state    = mark_stage(state, "analytics", STAGE_RUNNING)
    running_html = "<span class='status-running'>⟳  Generating…</span>"
    runs_dir     = state.get("runs_dir", "runs")
    figs_dir     = _REPO_ROOT / runs_dir / run_id / "analytics" / "figures"
    dash_path    = _REPO_ROOT / runs_dir / run_id / "analytics" / "dashboard" / "run_summary.html"

    for output, returncode in _stream_pipeline(cmd):
        if returncode is None:
            yield output, running_html, new_state, None, None, None, _dash_placeholder()
        else:
            stage_st = STAGE_DONE if returncode == 0 else STAGE_ERROR
            new_state = mark_stage(
                update_state(new_state, dashboard_path=str(dash_path)),
                "analytics", stage_st,
            )
            status_html = (
                "<span class='status-success'>✓  Analytics complete</span>"
                if returncode == 0
                else f"<span class='status-error'>✗  Exit {returncode}</span>"
            )
            suffix = (
                "\n\n[Stage 8 complete]"
                if returncode == 0
                else f"\n\n[Stage 8 failed — exit code {returncode}]"
            )
            # Locate generated PNGs
            training_fig  = _find_fig(figs_dir, "training")
            selection_fig = _find_fig(figs_dir, "selection")
            eval_fig      = _find_fig(figs_dir, "eval")
            dash_html     = _dash_link(dash_path) if returncode == 0 else _dash_placeholder()
            yield output + suffix, status_html, new_state, training_fig, selection_fig, eval_fig, dash_html


# ── Eval history HTML renderer ─────────────────────────────────────────

def _eval_placeholder() -> str:
    return (
        "<div style='padding:16px;text-align:center;color:#a8a29e;font-size:0.85rem;'>"
        "Eval results will appear here after Stage 6 completes."
        "</div>"
    )


def _render_eval_history(history_path: Path) -> str:
    if not history_path.exists():
        return _eval_placeholder()
    try:
        records = json.loads(history_path.read_text())
        if not records:
            return _eval_placeholder()

        # ── Summary cards ─── #
        best = min(records, key=lambda r: r.get("perplexity", float("inf")))
        cards_html = ""
        for key, label, fmt in [
            ("perplexity", "Best PPL",    lambda v: f"{v:.2f}"),
            ("mean_loss",  "Best Loss",   lambda v: f"{v:.4f}"),
            ("step",       "Best Step",   lambda v: str(int(v))),
        ]:
            val = best.get(key, "—")
            display = fmt(val) if isinstance(val, (int, float)) else str(val)
            cards_html += (
                f"<div class='metric-card'>"
                f"<span class='metric-value'>{display}</span>"
                f"<span class='metric-label'>{label}</span>"
                f"</div>"
            )
        header = (
            f"<div style='display:flex;gap:10px;margin-bottom:14px;flex-wrap:wrap;'>"
            f"{cards_html}</div>"
        )

        # ── Table ─── #
        keys = ["step", "perplexity", "mean_loss", "num_batches", "checkpoint"]
        th_cells = "".join(
            f"<th style='padding:5px 12px;color:#a8a29e;font-size:0.75rem;"
            f"text-transform:uppercase;border-bottom:1px solid #292524;'>{k}</th>"
            for k in keys
        )
        rows_html = ""
        for i, r in enumerate(sorted(records, key=lambda x: x.get("step", 0))):
            bg = "#14100e" if i % 2 == 0 else "#1c1917"
            highlight = " border-left: 3px solid #10b981;" if r is best else ""
            tds = "".join(
                f"<td style='padding:5px 12px;font-family:\"Fira Code\",monospace;"
                f"font-size:0.78rem;color:#e7e5e4;'>{_fmt_eval(r.get(k, '—'))}</td>"
                for k in keys
            )
            rows_html += f"<tr style='background:{bg};{highlight}'>{tds}</tr>"

        table = (
            f"<div style='overflow-x:auto;'>"
            f"<table style='width:100%;border-collapse:collapse;'>"
            f"<thead><tr>{th_cells}</tr></thead>"
            f"<tbody>{rows_html}</tbody>"
            f"</table></div>"
        )
        return header + table
    except Exception as exc:
        return f"<span style='color:#ef4444;font-size:0.8rem;'>Error reading eval history: {exc}</span>"


def _fmt_eval(v: object) -> str:
    if isinstance(v, float):
        if v == float("inf"):
            return "∞"
        if abs(v) < 0.001:
            return f"{v:.3e}"
        return f"{v:.4f}"
    if isinstance(v, int):
        return f"{v:,}"
    if isinstance(v, str) and v.endswith(".safetensors"):
        return Path(v).name   # shorten to filename only
    return str(v)


# ── Analytics figure helpers ───────────────────────────────────────────

def _find_fig(figs_dir: Path, keyword: str) -> str | None:
    """Return path string of first PNG containing *keyword* in its name, or None."""
    if not figs_dir.exists():
        return None
    for f in figs_dir.glob("*.png"):
        if keyword in f.name.lower():
            return str(f)
    return None


def _dash_placeholder() -> str:
    return "<span style='color:#a8a29e;font-size:0.85rem;'>Dashboard will appear after Stage 8 completes.</span>"


def _dash_link(dash_path: Path) -> str:
    if dash_path.exists():
        return (
            f"<div style='padding:12px;background:#14100e;border:1px solid #292524;"
            f"border-radius:8px;'>"
            f"<span style='color:#10b981;font-weight:700;'>✓  Dashboard ready</span><br>"
            f"<code style='font-size:0.8rem;color:#f97316;'>{dash_path}</code>"
            f"</div>"
        )
    return _dash_placeholder()


# ── Main render function ───────────────────────────────────────────────

def render_analytics_tab(state: gr.State) -> None:
    """Render Stage 6 (eval) and Stage 8 (analytics) inside the current Blocks context."""

    with gr.Column():

        gr.HTML(
            _badge("Stage 6 · 8", "")
            + "<h3 style='margin:6px 0 2px;color:#e7e5e4;'>Analytics</h3>"
            "<p style='color:#a8a29e;font-size:0.85rem;margin:0 0 12px;'>"
            "Evaluate checkpoints against a held-out split, then generate "
            "training-curve figures and an HTML run dashboard."
            "</p>"
        )

        with gr.Tabs():

            # ── Stage 6: Eval ─────────────────────────────────────── #
            with gr.Tab("📐  Stage 6 · Evaluate"):

                with gr.Row():
                    with gr.Column(scale=3):
                        checkpoint_dd = gr.Dropdown(
                            choices     = ["all (auto-discover)"],
                            value       = "all (auto-discover)",
                            label       = "Checkpoints to evaluate",
                            interactive = True,
                            info        = "Refresh after Stage 5 completes to see available checkpoints.",
                        )
                        refresh_ckpt_btn = gr.Button("↻  Refresh checkpoints", size="sm", variant="secondary")

                    with gr.Column(scale=2, elem_classes=["panel-glass"]):
                        eval_fraction_slider = gr.Slider(
                            minimum = 0.01,
                            maximum = 0.20,
                            step    = 0.01,
                            value   = 0.05,
                            label   = "Eval fraction  (tail split of sequences)",
                        )
                        eval_batch_num = gr.Number(
                            value=8, label="Eval batch size", precision=0,
                        )

                with gr.Row():
                    eval_btn    = gr.Button("📐  Run evaluation", variant="primary", scale=2)
                    eval_status = gr.HTML("<span class='status-idle'>●  Idle</span>", label="", scale=1)

                eval_log = gr.Textbox(
                    label="Evaluation log",
                    lines=10, max_lines=25,
                    interactive=False,
                    elem_classes=["log-terminal"],
                    show_copy_button=True,
                )

                eval_results_panel = gr.HTML(
                    _eval_placeholder(),
                    label="Eval history",
                )

            # ── Stage 8: Analytics ────────────────────────────────── #
            with gr.Tab("📊  Stage 8 · Analytics"):

                with gr.Row():
                    analytics_btn    = gr.Button("📊  Generate analytics", variant="primary", scale=2)
                    analytics_status = gr.HTML("<span class='status-idle'>●  Idle</span>", label="", scale=1)

                analytics_log = gr.Textbox(
                    label="Analytics log",
                    lines=8, max_lines=20,
                    interactive=False,
                    elem_classes=["log-terminal"],
                    show_copy_button=True,
                )

                gr.Markdown("#### Figures")
                with gr.Row():
                    training_img  = gr.Image(
                        label="Training curves",
                        type="filepath",
                        interactive=False,
                        height=320,
                    )
                    eval_img = gr.Image(
                        label="Eval perplexity",
                        type="filepath",
                        interactive=False,
                        height=320,
                    )

                selection_img = gr.Image(
                    label="Selection breakdown",
                    type="filepath",
                    interactive=False,
                    height=280,
                )

                gr.Markdown("#### Dashboard")
                dashboard_panel = gr.HTML(_dash_placeholder())

    # ── Hidden state mirrors ──────────────────────────────────────── #
    hidden_profile = gr.Textbox(value="local-dev", visible=False)
    hidden_run_id  = gr.Textbox(value="",           visible=False)

    def _sync_hidden(s: dict) -> tuple[str, str]:
        return s.get("profile", "local-dev"), s.get("run_id", "")

    state.change(
        fn=_sync_hidden,
        inputs=[state],
        outputs=[hidden_profile, hidden_run_id],
    )

    # ── Checkpoint refresh ────────────────────────────────────────── #
    def _refresh_checkpoints(s: dict) -> object:
        run_id   = s.get("run_id", "")
        runs_dir = s.get("runs_dir", "runs")
        choices  = _checkpoint_choices(run_id, runs_dir)
        return gr.update(choices=choices, value=choices[0])

    refresh_ckpt_btn.click(
        fn=_refresh_checkpoints,
        inputs=[state],
        outputs=[checkpoint_dd],
    )

    # Also refresh when state changes (e.g. after Stage 5 completes)
    state.change(
        fn=_refresh_checkpoints,
        inputs=[state],
        outputs=[checkpoint_dd],
    )

    # ── Stage 6 wiring ────────────────────────────────────────────── #
    eval_btn.click(
        fn=_on_eval,
        inputs=[checkpoint_dd, eval_fraction_slider, eval_batch_num, hidden_profile, state],
        outputs=[eval_log, eval_status, state, eval_results_panel],
    )

    # ── Stage 8 wiring ────────────────────────────────────────────── #
    analytics_btn.click(
        fn=_on_analytics,
        inputs=[hidden_profile, state],
        outputs=[
            analytics_log,
            analytics_status,
            state,
            training_img,
            selection_img,
            eval_img,
            dashboard_panel,
        ],
    )
