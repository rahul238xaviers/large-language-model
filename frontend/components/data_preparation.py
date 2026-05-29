"""Stages 1–3: Data Download, Profiling, and Selection tab.

Placeholder — full implementation delivered in the next phase.
``render_data_preparation_tab`` is the only public symbol.
"""
from __future__ import annotations
import gradio as gr


def render_data_preparation_tab(state: gr.State) -> None:
    """Render the Data Preparation tab layout inside the current Blocks context."""
    gr.Markdown(
        "### 📦 Data Preparation\n"
        "> **Stages 1 – 3**: Dataset download, document profiling, and quality-scored selection.\n\n"
        "*Full controls will be wired in the next implementation phase.*"
    )
