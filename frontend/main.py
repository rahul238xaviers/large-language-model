"""Central entry point and layout orchestrator for the GPT Pipeline UI.

Launch
------
    python -m frontend          # default port 7860
    python -m frontend --port 7861 --share

Architecture
------------
This file owns:
  - The top-level ``gr.Blocks`` application shell.
  - The shared ``gr.State`` that threads ``PipelineState`` through all tabs.
  - The global header, workflow-progress bar, and run-selector sidebar.
  - Tab registration: it imports each component module's ``render_*`` function
    and calls it inside its own ``gr.Tab`` — no layout logic from the
    components leaks into this file.
  - The ``_on_run_select`` callback that restores persisted state when the user
    picks an existing run from the sidebar dropdown.

Gradio never imports training/evaluation modules directly from this file.
All backend calls live exclusively in ``frontend/components/``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import gradio as gr

# ── Logging ── configure once at import time so every module's logger
# writes timestamped lines to stdout, visible in both Docker and native logs.
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
    stream  = sys.stdout,
)
log = logging.getLogger("frontend")

from frontend.theme import THEME, CSS
from frontend.state import (
    ALL_STAGES,
    initial_state,
    list_runs,
    mark_stage,
    progress_html,
    update_state,
)

# ── Component render functions (imported lazily so Gradio can start fast) ──
from frontend.components.data_preparation import render_data_preparation_tab
from frontend.components.pre_training     import render_pretraining_tab
from frontend.components.fine_tuning      import render_fine_tuning_tab
from frontend.components.analytics        import render_analytics_tab
from frontend.components.inference        import render_inference_tab


# ------------------------------------------------------------------ #
# Application builder                                                   #
# ------------------------------------------------------------------ #

def build_app() -> gr.Blocks:
    """Construct and return the complete Gradio Blocks application."""

    with gr.Blocks(
        title      = "GPT Pipeline",
        fill_height= True,
    ) as app:

        # ── Global shared state ─────────────────────────────────── #
        # ``state`` carries a PipelineState dict across all components.
        # Every callback that mutates workflow data must accept it as input
        # and return an updated copy as output.
        state = gr.State(value=initial_state)

        # ── Header ──────────────────────────────────────────────── #
        with gr.Row(elem_classes=["pipeline-header"]):
            with gr.Column(scale=7):
                gr.HTML(
                    "<h1 class='pipeline-title'>GPT Pipeline</h1>"
                    "<p class='pipeline-subtitle'>"
                    "9-stage production ML workflow · Apple Silicon · MLX"
                    "</p>"
                )
            with gr.Column(scale=3, min_width=260):
                run_id_display = gr.HTML(
                    "<span class='run-id-pill'>No active run</span>",
                    label="",
                )

        # ── Sidebar + main content ───────────────────────────────── #
        with gr.Row():

            # ── Left sidebar: run selector + progress ──────────── #
            with gr.Column(scale=1, min_width=240, elem_classes=["panel-glass"]):
                gr.Markdown("#### Run Manager")

                new_run_btn = gr.Button(
                    "＋  New Run",
                    variant = "primary",
                    size    = "sm",
                )

                existing_runs = gr.Dropdown(
                    choices  = list_runs(),
                    label    = "Resume existing run",
                    value    = None,
                    interactive = True,
                )

                refresh_runs_btn = gr.Button("↻  Refresh", size="sm", variant="secondary")

                gr.Markdown("---")
                gr.Markdown("#### Workflow Progress")
                progress_bar = gr.HTML(progress_html({s: "pending" for s in ALL_STAGES}))

                gr.Markdown("---")
                gr.Markdown("#### Configuration")
                profile_selector = gr.Dropdown(
                    choices = ["local-dev", "m3-ultra-prod"],
                    value   = "local-dev",
                    label   = "Config profile",
                    interactive = True,
                )

            # ── Main content: tabbed stage panels ──────────────── #
            with gr.Column(scale=5):
                with gr.Tabs() as tabs:

                    with gr.Tab("📦  Data", id="tab_data"):
                        render_data_preparation_tab(
                            state,
                            run_id_display=run_id_display,
                            progress_bar=progress_bar,
                        )

                    with gr.Tab("🧠  Pre-training", id="tab_pretrain"):
                        render_pretraining_tab(state)

                    with gr.Tab("🎯  Fine-tuning", id="tab_finetune"):
                        render_fine_tuning_tab(state)

                    with gr.Tab("📊  Analytics", id="tab_analytics"):
                        render_analytics_tab(state)

                    with gr.Tab("🚀  Inference", id="tab_inference"):
                        render_inference_tab(state)

        # ---------------------------------------------------------------- #
        # Global event wiring                                               #
        # ---------------------------------------------------------------- #

        def _on_new_run(current_state: dict) -> tuple[dict, str, str]:
            """Create a new run directory and update UI elements."""
            from frontend.state import create_run
            new_state, ctx = create_run(current_state)
            pill    = f"<span class='run-id-pill'>{ctx.run_id}</span>"
            bar     = progress_html(new_state["stage_status"])
            return new_state, pill, bar

        new_run_btn.click(
            fn      = _on_new_run,
            inputs  = [state],
            outputs = [state, run_id_display, progress_bar],
        )

        def _on_run_select(run_id: str, current_state: dict) -> tuple[dict, str, str]:
            """Restore a previously created run from disk."""
            if not run_id:
                return current_state, current_state.get("run_id", ""), ""
            import json
            from pathlib import Path as _Path
            runs_dir = current_state.get("runs_dir", "runs")
            meta_path = _Path(runs_dir) / run_id / "model_metadata.json"
            # Restore metadata if it exists, otherwise just set run_id
            new_state = update_state(current_state, run_id=run_id)
            if meta_path.exists():
                meta = json.loads(meta_path.read_text())
                new_state = update_state(new_state,
                    latest_checkpoint = meta.get("checkpoint_path", ""),
                    best_perplexity   = meta.get("eval_summary", {}).get("perplexity", float("inf")),
                )
            pill = f"<span class='run-id-pill'>{run_id}</span>"
            bar  = progress_html(new_state["stage_status"])
            return new_state, pill, bar

        existing_runs.change(
            fn      = _on_run_select,
            inputs  = [existing_runs, state],
            outputs = [state, run_id_display, progress_bar],
        )

        def _on_refresh_runs(current_state: dict) -> tuple[dict, object]:
            runs_dir = current_state.get("runs_dir", "runs")
            choices  = list_runs(runs_dir)
            return current_state, gr.update(choices=choices)

        refresh_runs_btn.click(
            fn      = _on_refresh_runs,
            inputs  = [state],
            outputs = [state, existing_runs],
        )
        # Keep global progress bar in sync with any state mutations
        def _sync_progress(s: dict) -> str:
            return progress_html(s.get("stage_status", {}))

        state.change(
            fn      = _sync_progress,
            inputs  = [state],
            outputs = [progress_bar],
        )
        def _on_profile_change(profile: str, current_state: dict) -> tuple[dict, str]:
            from pipeline.orchestration.config_loader import ConfigLoader
            cfg       = ConfigLoader().load(profile)
            new_state = update_state(current_state, profile=profile, config=cfg)
            return new_state, profile

        profile_selector.change(
            fn      = _on_profile_change,
            inputs  = [profile_selector, state],
            outputs = [state, profile_selector],
        )

    return app


# ------------------------------------------------------------------ #
# Entry point                                                           #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        prog        = "python -m frontend",
        description = "GPT Pipeline Gradio UI",
    )
    parser.add_argument("--port",   type=int,  default=7860,  help="HTTP port")
    parser.add_argument("--host",   default="0.0.0.0",        help="Bind address")
    parser.add_argument("--share",  action="store_true",       help="Create a public Gradio share link")
    parser.add_argument("--debug",  action="store_true",       help="Enable Gradio debug mode")
    args = parser.parse_args()

    log.info("Starting GPT Pipeline UI  port=%s  host=%s", args.port, args.host)
    app = build_app()
    log.info("App built — launching Gradio")
    app.launch(
        server_name = args.host,
        server_port = args.port,
        share       = args.share,
        debug       = args.debug,
        show_error  = True,
        theme       = THEME,
        css         = CSS,
    )


if __name__ == "__main__":
    main()
