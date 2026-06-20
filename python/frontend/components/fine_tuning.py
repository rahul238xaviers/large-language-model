"""Fine-tuning tab — future SFT pipeline extension.

Placeholder — full implementation delivered in a later phase.
``render_fine_tuning_tab`` is the only public symbol.
"""
from __future__ import annotations
import gradio as gr


def render_fine_tuning_tab(state: gr.State) -> None:
    """Render the Fine-tuning tab layout inside the current Blocks context."""
    gr.Markdown(
        "### 🎯 Fine-tuning\n"
        "> Supervised fine-tuning stages (planned future pipeline extension).\n\n"
        "*Not yet available.*"
    )
