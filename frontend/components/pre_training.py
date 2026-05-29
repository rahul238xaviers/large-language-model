"""Stages 4–5: Data tokenisation and MLX pre-training tab.

All layout and event wiring lives in ``render_pretraining_tab``.
Backend work is executed via subprocess so the UI stays responsive.
Live training metrics are polled from ``metrics.csv`` while training runs.

Public symbol: ``render_pretraining_tab(state)``
"""
from __future__ import annotations

import csv
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import gradio as gr

from frontend.state import (
    STAGE_DONE,
    STAGE_ERROR,
    STAGE_RUNNING,
    mark_stage,
    progress_html,
    update_state,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Shared subprocess streamer (mirrors data_preparation.py) ──────────

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


def _pill(run_id: str) -> str:
    if run_id:
        return f"<span class='run-id-pill'>{run_id}</span>"
    return "<span class='run-id-pill'>No active run</span>"


def _badge(label: str, variant: str = "") -> str:
    cls = f"stage-badge {variant}".strip()
    return f"<span class='{cls}'>{label}</span>"


# ── Stage 4 callback ───────────────────────────────────────────────────

def _on_tokenise(
    block_size: int,
    encoding: str,
    profile: str,
    state: dict,
) -> Iterator[tuple]:
    run_id = state.get("run_id", "")
    if not run_id:
        yield (
            "⚠  No active run. Complete Stage 3 (Select) first.",
            "<span class='status-error'>●  Error</span>",
            state,
        )
        return

    cmd = [
        sys.executable, "-m", "pipeline",
        "--profile", profile,
        "data-tokenise",
        "--run-id", run_id,
    ]
    env = {
        "PIPELINE_TOKENISATION__BLOCK_SIZE":  str(int(block_size)),
        "PIPELINE_TOKENISATION__ENCODING":    encoding,
    }

    new_state = mark_stage(state, "tokenise", STAGE_RUNNING)
    running_html = "<span class='status-running'>⟳  Running…</span>"

    for output, returncode in _stream_pipeline(cmd, env=env):
        if returncode is None:
            yield output, running_html, new_state
        else:
            stage_st = STAGE_DONE if returncode == 0 else STAGE_ERROR
            new_state = mark_stage(new_state, "tokenise", stage_st)
            runs_dir = state.get("runs_dir", "runs")
            seq_path = str(
                _REPO_ROOT / runs_dir / run_id / "data" / "tokenised" / "sequences.npy"
            )
            if returncode == 0:
                new_state = update_state(new_state, sequences_path=seq_path)
            status_html = (
                "<span class='status-success'>✓  Done</span>"
                if returncode == 0
                else f"<span class='status-error'>✗  Exit {returncode}</span>"
            )
            suffix = (
                f"\n\n[Stage 4 complete — sequences: {seq_path}]"
                if returncode == 0
                else f"\n\n[Stage 4 failed — exit code {returncode}]"
            )
            yield output + suffix, status_html, new_state


# ── Stage 5 callback ───────────────────────────────────────────────────

def _on_train(
    max_steps: int,
    batch_size: int,
    accum_steps: int,
    lr_max: float,
    lr_min: float,
    warmup_steps: int,
    grad_clip: float,
    weight_decay: float,
    checkpoint_interval: int,
    keep_last_n: int,
    seed: int,
    profile: str,
    state: dict,
) -> Iterator[tuple]:
    run_id = state.get("run_id", "")
    if not run_id:
        yield (
            "⚠  No active run. Complete Stage 4 (Tokenise) first.",
            "<span class='status-error'>●  Error</span>",
            state,
            _metrics_placeholder(),
            "",
        )
        return

    cmd = [
        sys.executable, "-m", "pipeline",
        "--profile", profile,
        "train-pretrain",
        "--run-id", run_id,
        "--max-steps", str(int(max_steps)),
    ]
    env = {
        "PIPELINE_TRAINING__BATCH_SIZE":             str(int(batch_size)),
        "PIPELINE_TRAINING__ACCUM_STEPS":            str(int(accum_steps)),
        "PIPELINE_TRAINING__LR_MAX":                 str(lr_max),
        "PIPELINE_TRAINING__LR_MIN":                 str(lr_min),
        "PIPELINE_TRAINING__WARMUP_STEPS":           str(int(warmup_steps)),
        "PIPELINE_TRAINING__GRAD_CLIP":              str(grad_clip),
        "PIPELINE_TRAINING__WEIGHT_DECAY":           str(weight_decay),
        "PIPELINE_TRAINING__CHECKPOINT_INTERVAL":    str(int(checkpoint_interval)),
        "PIPELINE_TRAINING__KEEP_LAST_N_CHECKPOINTS": str(int(keep_last_n)),
        "PIPELINE_TRAINING__SEED":                   str(int(seed)),
    }

    new_state  = mark_stage(state, "pretrain", STAGE_RUNNING)
    runs_dir   = state.get("runs_dir", "runs")
    metrics_csv = _REPO_ROOT / runs_dir / run_id / "training" / "metrics.csv"
    running_html = "<span class='status-running'>⟳  Training…</span>"
    live_metrics = _metrics_placeholder()

    for output, returncode in _stream_pipeline(cmd, env=env):
        # Poll metrics.csv while the process is running
        metrics_html = _read_metrics_html(metrics_csv)
        if returncode is None:
            # Derive latest checkpoint from log
            ckpt = _latest_ckpt_from_log(output, run_id, runs_dir)
            yield output, running_html, new_state, metrics_html, ckpt
        else:
            stage_st = STAGE_DONE if returncode == 0 else STAGE_ERROR
            # Find the best checkpoint on disk
            ckpt_dir = _REPO_ROOT / runs_dir / run_id / "training" / "checkpoints"
            latest_ckpt = _best_checkpoint(ckpt_dir)
            new_state = mark_stage(
                update_state(new_state,
                    training_metrics_path=str(metrics_csv),
                    latest_checkpoint=latest_ckpt,
                ),
                "pretrain", stage_st,
            )
            status_html = (
                "<span class='status-success'>✓  Training complete</span>"
                if returncode == 0
                else f"<span class='status-error'>✗  Exit {returncode}</span>"
            )
            suffix = (
                f"\n\n[Stage 5 complete — latest checkpoint: {latest_ckpt}]"
                if returncode == 0
                else f"\n\n[Stage 5 failed — exit code {returncode}]"
            )
            yield output + suffix, status_html, new_state, _read_metrics_html(metrics_csv), latest_ckpt


# ── Metrics helpers ────────────────────────────────────────────────────

def _metrics_placeholder() -> str:
    return (
        "<div class='metric-card' style='text-align:center;padding:20px;'>"
        "<span style='color:#a8a29e;font-size:0.85rem;'>"
        "Metrics will appear here once training starts…"
        "</span></div>"
    )


def _read_metrics_html(metrics_csv: Path) -> str:
    """Read the last 5 rows of metrics.csv and render as an HTML table."""
    if not metrics_csv.exists():
        return _metrics_placeholder()
    try:
        rows = []
        with open(metrics_csv, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        if not rows:
            return _metrics_placeholder()
        last_rows = rows[-5:]
        headers = list(rows[0].keys())
        th_cells = "".join(f"<th style='padding:4px 10px;color:#a8a29e;font-size:0.75rem;text-transform:uppercase;'>{h}</th>" for h in headers)
        tr_rows = ""
        for i, row in enumerate(last_rows):
            bg = "#14100e" if i % 2 == 0 else "#1c1917"
            tds = "".join(
                f"<td style='padding:4px 10px;font-family:\"Fira Code\",monospace;font-size:0.78rem;color:#e7e5e4;'>{_fmt_val(row.get(h,''))}</td>"
                for h in headers
            )
            tr_rows += f"<tr style='background:{bg};'>{tds}</tr>"
        # Latest step summary cards
        last = last_rows[-1]
        cards = ""
        for key, label in [("loss", "Loss"), ("tokens_per_sec", "tok/s"), ("lr", "LR"), ("step", "Step")]:
            if key in last:
                cards += (
                    f"<div class='metric-card'>"
                    f"<span class='metric-value'>{_fmt_val(last[key])}</span>"
                    f"<span class='metric-label'>{label}</span>"
                    f"</div>"
                )
        cards_html = f"<div style='display:flex;gap:10px;margin-bottom:12px;flex-wrap:wrap;'>{cards}</div>" if cards else ""
        table_html = (
            f"<table style='width:100%;border-collapse:collapse;'>"
            f"<thead><tr style='border-bottom:1px solid #292524;'>{th_cells}</tr></thead>"
            f"<tbody>{tr_rows}</tbody>"
            f"</table>"
        )
        return cards_html + table_html
    except Exception as exc:
        return f"<span style='color:#ef4444;font-size:0.8rem;'>Error reading metrics: {exc}</span>"


def _fmt_val(v: str) -> str:
    """Format numeric strings for display."""
    try:
        f = float(v)
        if abs(f) < 1e-3 and f != 0:
            return f"{f:.3e}"
        if abs(f) >= 1000:
            return f"{f:,.0f}"
        return f"{f:.4f}".rstrip("0").rstrip(".")
    except (ValueError, TypeError):
        return v


def _latest_ckpt_from_log(output: str, run_id: str, runs_dir: str) -> str:
    """Parse the latest step_*.safetensors path from training log output."""
    last = ""
    for line in output.splitlines():
        if "step_" in line and ".safetensors" in line:
            for token in line.split():
                if token.endswith(".safetensors"):
                    last = token.strip("'\"")
    return last


def _best_checkpoint(ckpt_dir: Path) -> str:
    """Return the path of the highest-step safetensors file on disk."""
    if not ckpt_dir.exists():
        return ""
    files = sorted(ckpt_dir.glob("step_*.safetensors"))
    return str(files[-1]) if files else ""


# ── Main render function ───────────────────────────────────────────────

def render_pretraining_tab(state: gr.State) -> None:
    """Render Stage 4 (tokenise) and Stage 5 (pretrain) inside the current Blocks context."""

    with gr.Column():

        gr.HTML(
            _badge("Stage 4 · 5", "")
            + "<h3 style='margin:6px 0 2px;color:#e7e5e4;'>Pre-training</h3>"
            "<p style='color:#a8a29e;font-size:0.85rem;margin:0 0 12px;'>"
            "Pack token sequences then train the 1.6B-parameter GQA GPT on Apple Silicon MLX."
            "</p>"
        )

        with gr.Tabs():

            # ── Stage 4: Tokenise ────────────────────────────────── #
            with gr.Tab("🔤  Stage 4 · Tokenise"):

                with gr.Row():
                    with gr.Column(scale=3):
                        encoding_radio = gr.Radio(
                            choices = ["cl100k_base", "p50k_base", "r50k_base"],
                            value   = "cl100k_base",
                            label   = "tiktoken encoding",
                        )
                        block_size_slider = gr.Slider(
                            minimum = 512,
                            maximum = 4096,
                            step    = 128,
                            value   = 2048,
                            label   = "Block size (tokens per sequence)",
                            info    = "Must match model.block_size in training config.",
                        )
                    with gr.Column(scale=2, elem_classes=["panel-glass"]):
                        gr.Markdown(
                            "**Input:** Stage 3 `pretrain_dataset.arrow`  \n"
                            "**Output:** `sequences.npy` — memory-mappable int32 array  \n"
                            "Sequences are packed back-to-back with an `<|endoftext|>` boundary token."
                        )

                with gr.Row():
                    tokenise_btn    = gr.Button("🔤  Pack sequences", variant="primary", scale=2)
                    tokenise_status = gr.HTML("<span class='status-idle'>●  Idle</span>", label="", scale=1)

                tokenise_log = gr.Textbox(
                    label="Tokenisation log",
                    lines=10, max_lines=25,
                    interactive=False,
                    elem_classes=["log-terminal"],
                )

            # ── Stage 5: Pretrain ─────────────────────────────────── #
            with gr.Tab("🧠  Stage 5 · Pre-train"):

                gr.Markdown("#### Hyperparameters")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("**Schedule**")
                        max_steps_num = gr.Number(
                            value=10_000, label="Max steps", precision=0,
                            info="Total weight-update steps (after gradient accumulation).",
                        )
                        warmup_steps_num = gr.Number(
                            value=500, label="Warmup steps", precision=0,
                        )
                        lr_max_num = gr.Number(
                            value=3e-4, label="Peak LR",
                            info="AdamW peak learning rate.",
                        )
                        lr_min_num = gr.Number(
                            value=3e-5, label="Min LR  (cosine floor)",
                        )

                    with gr.Column():
                        gr.Markdown("**Batch / Accumulation**")
                        batch_size_num = gr.Number(
                            value=4, label="Micro-batch size (sequences)",
                            precision=0,
                            info="Sequences per forward-backward pass.",
                        )
                        accum_steps_num = gr.Number(
                            value=32, label="Gradient accumulation steps",
                            precision=0,
                            info="Effective batch = micro_batch × accum × block_size tokens.",
                        )
                        gr.HTML(
                            "<div class='metric-card' id='eff-batch-info'>"
                            "<span class='metric-label'>Effective tokens/step</span>"
                            "<span class='metric-value' id='eff-tokens'>262 144</span>"
                            "</div>"
                        )

                    with gr.Column():
                        gr.Markdown("**Regularisation & Checkpointing**")
                        grad_clip_num = gr.Number(
                            value=1.0, label="Gradient clip norm",
                        )
                        weight_decay_num = gr.Number(
                            value=0.1, label="Weight decay",
                        )
                        ckpt_interval_num = gr.Number(
                            value=500, label="Checkpoint every N steps",
                            precision=0,
                        )
                        keep_last_n_num = gr.Number(
                            value=3, label="Keep last N checkpoints",
                            precision=0,
                        )
                        seed_num = gr.Number(
                            value=42, label="Random seed",
                            precision=0,
                        )

                gr.Markdown("---")

                with gr.Row():
                    train_btn    = gr.Button("🧠  Start training", variant="primary", scale=2)
                    train_status = gr.HTML("<span class='status-idle'>●  Idle</span>", label="", scale=1)

                # ── Live metrics ─────────────────────────────────── #
                with gr.Row():
                    with gr.Column(scale=3):
                        train_log = gr.Textbox(
                            label="Training log",
                            lines=14, max_lines=40,
                            interactive=False,
                            elem_classes=["log-terminal"],
                        )
                    with gr.Column(scale=2):
                        metrics_panel = gr.HTML(
                            _metrics_placeholder(),
                            label="Live metrics  (last 5 steps)",
                        )

                latest_ckpt_display = gr.Textbox(
                    label="Latest checkpoint path",
                    interactive=False,
                    placeholder="will update as checkpoints are saved…",
                )

    # ── Hidden state mirrors ─────────────────────────────────────────── #
    hidden_profile = gr.Textbox(value="local-dev", visible=False)
    hidden_run_id  = gr.Textbox(value="",           visible=False)

    def _sync_hidden(s: dict) -> tuple[str, str]:
        return s.get("profile", "local-dev"), s.get("run_id", "")

    state.change(
        fn=_sync_hidden,
        inputs=[state],
        outputs=[hidden_profile, hidden_run_id],
    )

    # ── Stage 4 wiring ─────────────────────────────────────────────── #
    tokenise_btn.click(
        fn=_on_tokenise,
        inputs=[block_size_slider, encoding_radio, hidden_profile, state],
        outputs=[tokenise_log, tokenise_status, state],
    )

    # ── Stage 5 wiring ─────────────────────────────────────────────── #
    train_btn.click(
        fn=_on_train,
        inputs=[
            max_steps_num,
            batch_size_num,
            accum_steps_num,
            lr_max_num,
            lr_min_num,
            warmup_steps_num,
            grad_clip_num,
            weight_decay_num,
            ckpt_interval_num,
            keep_last_n_num,
            seed_num,
            hidden_profile,
            state,
        ],
        outputs=[train_log, train_status, state, metrics_panel, latest_ckpt_display],
    )
