"""Stages 1–3: Data Download, Profiling, and Quality-Scored Selection tab.

All layout and event wiring lives inside ``render_data_preparation_tab``.
Backend calls go through ``python -m pipeline`` subprocesses so the UI
stays responsive and the streaming log mirrors exactly what the CLI prints.

Public symbol: ``render_data_preparation_tab(state, *, run_id_display, progress_bar)``
"""
from __future__ import annotations

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
    progress_html,
    update_state,
)

# Repo root so subprocess CWD is always correct regardless of where the app is launched.
_REPO_ROOT = Path(__file__).resolve().parents[2]


# ------------------------------------------------------------------ #
# Shared subprocess streaming helper                                    #
# ------------------------------------------------------------------ #

def _stream_pipeline(
    cmd: list[str],
    env: dict | None = None,
) -> Iterator[tuple[str, int | None]]:
    """Run a pipeline CLI command and yield ``(cumulative_stdout, returncode|None)`` pairs.

    Yields ``(text, None)`` while running and ``(text, returncode)`` on completion.
    """
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


def _last_run_id_line(text: str) -> str:
    """Extract the run_id emitted by the CLI's final ``print(ctx.run_id)``."""
    for line in reversed(text.strip().splitlines()):
        stripped = line.strip()
        if stripped.startswith("run_"):
            return stripped
    return ""


# ------------------------------------------------------------------ #
# Stage 1 callbacks                                                     #
# ------------------------------------------------------------------ #

def _on_download(
    sources: list[str],
    hf_token: str,
    profile: str,
    state: dict,
) -> Iterator[tuple]:
    """Stream Stage 1 download; update state with new run_id on completion."""
    if not sources:
        yield (
            "⚠  No sources selected — please tick at least one dataset.",
            "<span class='status-error'>●  Error</span>",
            state,
            "<span class='run-id-pill'>No active run</span>",
            progress_html(state.get("stage_status", {})),
        )
        return

    cmd = [sys.executable, "-m", "pipeline", "--profile", profile, "data-download"]
    # Pass --source for each selected source
    for s in sources:
        cmd += ["--source", s]
    # Preserve existing run if one is already active
    existing_run_id = state.get("run_id", "")
    if existing_run_id:
        cmd += ["--run-id", existing_run_id]

    env = {"HF_TOKEN": hf_token} if hf_token.strip() else {}

    new_state = mark_stage(state, "download", STAGE_RUNNING)
    status_running = "<span class='status-running'>⟳  Running…</span>"

    for output, returncode in _stream_pipeline(cmd, env=env):
        if returncode is None:
            yield (
                output,
                status_running,
                new_state,
                _pill(new_state.get("run_id", "")),
                progress_html(new_state["stage_status"]),
            )
        else:
            run_id = _last_run_id_line(output) or existing_run_id
            stage_status = STAGE_DONE if returncode == 0 else STAGE_ERROR
            new_state = mark_stage(
                update_state(new_state, run_id=run_id),
                "download", stage_status,
            )
            status_html = (
                "<span class='status-success'>✓  Done</span>"
                if returncode == 0
                else f"<span class='status-error'>✗  Exit {returncode}</span>"
            )
            suffix = (
                f"\n\n[Stage 1 complete — Run ID: {run_id}]"
                if returncode == 0
                else f"\n\n[Stage 1 failed — exit code {returncode}]"
            )
            yield (
                output + suffix,
                status_html,
                new_state,
                _pill(run_id),
                progress_html(new_state["stage_status"]),
            )


# ------------------------------------------------------------------ #
# Stage 2 callbacks                                                     #
# ------------------------------------------------------------------ #

def _on_profile(
    sample_size: int,
    profile: str,
    state: dict,
) -> Iterator[tuple]:
    """Stream Stage 2 profiling."""
    run_id = state.get("run_id", "")
    if not run_id:
        yield (
            "⚠  No active run. Complete Stage 1 (Download) first.",
            "<span class='status-error'>●  Error</span>",
            state,
        )
        return

    cmd = [
        sys.executable, "-m", "pipeline",
        "--profile", profile,
        "data-profile",
        "--run-id", run_id,
    ]

    new_state = mark_stage(state, "profile", STAGE_RUNNING)
    status_running = "<span class='status-running'>⟳  Running…</span>"

    for output, returncode in _stream_pipeline(cmd):
        if returncode is None:
            yield output, status_running, new_state
        else:
            stage_status = STAGE_DONE if returncode == 0 else STAGE_ERROR
            new_state = mark_stage(new_state, "profile", stage_status)
            # Resolve profile_dir from run directory
            runs_dir = state.get("runs_dir", "runs")
            profile_dir = str(
                _REPO_ROOT / runs_dir / run_id / "data" / "profiles"
            )
            if returncode == 0:
                new_state = update_state(new_state, profile_dir=profile_dir)
            status_html = (
                "<span class='status-success'>✓  Done</span>"
                if returncode == 0
                else f"<span class='status-error'>✗  Exit {returncode}</span>"
            )
            suffix = (
                "\n\n[Stage 2 complete]"
                if returncode == 0
                else f"\n\n[Stage 2 failed — exit code {returncode}]"
            )
            yield output + suffix, status_html, new_state


# ------------------------------------------------------------------ #
# Stage 3 callbacks                                                     #
# ------------------------------------------------------------------ #

def _on_select(
    min_quality: float,
    min_code: float,
    max_code: float,
    min_chars: int,
    max_chars: int,
    deduplicate: bool,
    max_dup_rate: float,
    profile: str,
    state: dict,
) -> Iterator[tuple]:
    """Stream Stage 3 selection with inline policy overrides via env vars."""
    run_id = state.get("run_id", "")
    if not run_id:
        yield (
            "⚠  No active run. Complete Stage 1 (Download) first.",
            "<span class='status-error'>●  Error</span>",
            state,
        )
        return

    cmd = [
        sys.executable, "-m", "pipeline",
        "--profile", profile,
        "data-select",
        "--run-id", run_id,
    ]

    # Pass policy overrides as env vars that ConfigLoader merges
    env = {
        "PIPELINE_SELECTION__POLICY__MIN_QUALITY_SCORE": str(min_quality),
        "PIPELINE_SELECTION__POLICY__MIN_CODE_SCORE":    str(min_code),
        "PIPELINE_SELECTION__POLICY__MAX_CODE_SCORE":    str(max_code),
        "PIPELINE_SELECTION__POLICY__MIN_CHAR_LENGTH":   str(int(min_chars)),
        "PIPELINE_SELECTION__POLICY__MAX_CHAR_LENGTH":   str(int(max_chars)),
        "PIPELINE_SELECTION__POLICY__DEDUPLICATE":       "true" if deduplicate else "false",
        "PIPELINE_SELECTION__POLICY__MAX_DUPLICATE_RATE": str(max_dup_rate),
    }

    new_state = mark_stage(state, "select", STAGE_RUNNING)
    status_running = "<span class='status-running'>⟳  Running…</span>"

    for output, returncode in _stream_pipeline(cmd, env=env):
        if returncode is None:
            yield output, status_running, new_state
        else:
            stage_status = STAGE_DONE if returncode == 0 else STAGE_ERROR
            new_state = mark_stage(new_state, "select", stage_status)
            runs_dir = state.get("runs_dir", "runs")
            selection_dir = str(_REPO_ROOT / runs_dir / run_id / "data" / "selected")
            arrow_path    = str(_REPO_ROOT / runs_dir / run_id / "data" / "selected" / "training.arrow")
            if returncode == 0:
                new_state = update_state(
                    new_state,
                    selection_dir=selection_dir,
                    arrow_path=arrow_path,
                )
            status_html = (
                "<span class='status-success'>✓  Done</span>"
                if returncode == 0
                else f"<span class='status-error'>✗  Exit {returncode}</span>"
            )
            suffix = (
                "\n\n[Stage 3 complete]"
                if returncode == 0
                else f"\n\n[Stage 3 failed — exit code {returncode}]"
            )
            yield output + suffix, status_html, new_state


# ------------------------------------------------------------------ #
# Small HTML helpers                                                    #
# ------------------------------------------------------------------ #

def _pill(run_id: str) -> str:
    if run_id:
        return f"<span class='run-id-pill'>{run_id}</span>"
    return "<span class='run-id-pill'>No active run</span>"


def _badge(label: str, variant: str = "") -> str:
    cls = f"stage-badge {variant}".strip()
    return f"<span class='{cls}'>{label}</span>"


# ------------------------------------------------------------------ #
# Main render function                                                  #
# ------------------------------------------------------------------ #

def render_data_preparation_tab(
    state: gr.State,
    *,
    run_id_display: gr.HTML | None = None,
    progress_bar: gr.HTML | None = None,
) -> None:
    """Render all three data-preparation stages inside the current Blocks context."""

    with gr.Column():

        gr.HTML(
            _badge("Stage 1 · 2 · 3", "")
            + "<h3 style='margin:6px 0 2px;color:#e7e5e4;'>Data Preparation</h3>"
            "<p style='color:#a8a29e;font-size:0.85rem;margin:0 0 12px;'>"
            "Download → Profile → Quality-score and select your training corpus."
            "</p>"
        )

        with gr.Tabs():

            # ── Stage 1: Download ────────────────────────────────── #
            with gr.Tab("⬇  Stage 1 · Download"):

                with gr.Row():
                    with gr.Column(scale=3):
                        source_selector = gr.CheckboxGroup(
                            choices = ["rust_stack", "starcoder_python", "fineweb_edu"],
                            value   = ["rust_stack", "starcoder_python", "fineweb_edu"],
                            label   = "Dataset sources",
                        )
                        hf_token_box = gr.Textbox(
                            label       = "HuggingFace token  (optional — for gated datasets)",
                            type        = "password",
                            placeholder = "hf_…",
                            value       = "",
                        )
                    with gr.Column(scale=2):
                        gr.Markdown(
                            "**Cache location:** `data/datasets/`  \n"
                            "Physical files are shared across all runs.  \n"
                            "A fresh `manifest.json` is written per run.",
                            elem_classes=["panel-glass"],
                        )

                with gr.Row():
                    download_btn    = gr.Button("⬇  Download datasets", variant="primary", scale=2)
                    download_status = gr.HTML(
                        "<span class='status-idle'>●  Idle</span>",
                        label="",
                        scale=1,
                    )

                download_log = gr.Textbox(
                    label         = "Download log",
                    lines         = 12,
                    max_lines     = 30,
                    interactive   = False,
                    elem_classes  = ["log-terminal"],
                )

            # ── Stage 2: Profile ─────────────────────────────────── #
            with gr.Tab("🔬  Stage 2 · Profile"):

                with gr.Row():
                    sample_size_slider = gr.Slider(
                        minimum = 1_000,
                        maximum = 500_000,
                        step    = 1_000,
                        value   = 100_000,
                        label   = "Max documents to sample per source",
                    )
                    gr.Markdown(
                        "Profiling computes quality scores, code-likeness, "
                        "token-length distributions, and exact-hash dedup rates.",
                        elem_classes=["panel-glass"],
                    )

                with gr.Row():
                    profile_btn    = gr.Button("🔬  Run profiling", variant="primary", scale=2)
                    profile_status = gr.HTML(
                        "<span class='status-idle'>●  Idle</span>",
                        label="",
                        scale=1,
                    )

                profile_log = gr.Textbox(
                    label         = "Profiling log",
                    lines         = 12,
                    max_lines     = 30,
                    interactive   = False,
                    elem_classes  = ["log-terminal"],
                )

            # ── Stage 3: Select ──────────────────────────────────── #
            with gr.Tab("✂  Stage 3 · Select"):

                gr.Markdown("#### Selection Policy")

                with gr.Row():
                    with gr.Column():
                        min_quality_slider = gr.Slider(
                            0.0, 1.0, step=0.01, value=0.20,
                            label="Min quality score",
                            info="Documents below this threshold are dropped.",
                        )
                        min_code_slider = gr.Slider(
                            0.0, 1.0, step=0.01, value=0.00,
                            label="Min code score",
                            info="0.0 = include all prose.",
                        )
                        max_code_slider = gr.Slider(
                            0.0, 1.0, step=0.01, value=1.00,
                            label="Max code score",
                            info="1.0 = no upper limit.",
                        )
                    with gr.Column():
                        min_chars_number = gr.Number(
                            value=150, label="Min character length",
                            precision=0,
                        )
                        max_chars_number = gr.Number(
                            value=0, label="Max character length  (0 = unlimited)",
                            precision=0,
                        )
                        dedup_checkbox = gr.Checkbox(
                            value=True,
                            label="Hash-based deduplication",
                        )
                        max_dup_rate_slider = gr.Slider(
                            0.0, 1.0, step=0.01, value=0.50,
                            label="Abort if duplicate rate exceeds",
                        )

                gr.Markdown("---")

                with gr.Row():
                    select_btn    = gr.Button("✂  Run selection", variant="primary", scale=2)
                    select_status = gr.HTML(
                        "<span class='status-idle'>●  Idle</span>",
                        label="",
                        scale=1,
                    )

                select_log = gr.Textbox(
                    label         = "Selection log",
                    lines         = 12,
                    max_lines     = 30,
                    interactive   = False,
                    elem_classes  = ["log-terminal"],
                )

    # ── Derive the profile string from state for callbacks ─────────── #
    # We read `state["profile"]` by binding a hidden Textbox that is
    # kept in sync with state.  This avoids passing the gr.State object
    # directly into non-streaming inputs.
    hidden_profile = gr.Textbox(value="local-dev", visible=False)
    hidden_run_id  = gr.Textbox(value="",           visible=False)

    # Keep hidden fields in sync whenever state changes
    def _sync_hidden(s: dict) -> tuple[str, str]:
        return s.get("profile", "local-dev"), s.get("run_id", "")

    state.change(
        fn      = _sync_hidden,
        inputs  = [state],
        outputs = [hidden_profile, hidden_run_id],
    )

    # ── Stage 1 wiring ─────────────────────────────────────────────── #
    _dl_outputs = [download_log, download_status, state]
    if run_id_display is not None:
        _dl_outputs.append(run_id_display)
    if progress_bar is not None:
        _dl_outputs.append(progress_bar)

    # Wrap callback to conditionally include optional outputs
    def _download_wrapped(sources, hf_token, profile, s):
        for result in _on_download(sources, hf_token, profile, s):
            yield result[: len(_dl_outputs)]

    download_btn.click(
        fn      = _download_wrapped,
        inputs  = [source_selector, hf_token_box, hidden_profile, state],
        outputs = _dl_outputs,
    )

    # ── Stage 2 wiring ─────────────────────────────────────────────── #
    profile_btn.click(
        fn      = _on_profile,
        inputs  = [sample_size_slider, hidden_profile, state],
        outputs = [profile_log, profile_status, state],
    )

    # ── Stage 3 wiring ─────────────────────────────────────────────── #
    select_btn.click(
        fn      = _on_select,
        inputs  = [
            min_quality_slider,
            min_code_slider,
            max_code_slider,
            min_chars_number,
            max_chars_number,
            dedup_checkbox,
            max_dup_rate_slider,
            hidden_profile,
            state,
        ],
        outputs = [select_log, select_status, state],
    )
