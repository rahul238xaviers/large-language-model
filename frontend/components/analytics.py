"""Stages 6 + 8: Checkpoint Evaluation and Analytics Workbench tab.

Placeholder — full implementation delivered in the next phase.
``render_analytics_tab`` is the only public symbol.
"""
from __future__ import annotations
import gradio as gr


def render_analytics_tab(state: gr.State) -> None:
    """Render the Analytics tab layout inside the current Blocks context."""
    gr.Markdown(
        "### 📊 Analytics\n"
        "> **Stage 6**: Checkpoint perplexity evaluation.  "
        "**Stage 8**: Training metrics plots and run dashboard.\n\n"
        "*Full controls will be wired in the next implementation phase.*"
    )
