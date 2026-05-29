"""Stages 7 + 9: Inference serving and model registry tab.

Placeholder — full implementation delivered in the next phase.
``render_inference_tab`` is the only public symbol.
"""
from __future__ import annotations
import gradio as gr


def render_inference_tab(state: gr.State) -> None:
    """Render the Inference tab layout inside the current Blocks context."""
    gr.Markdown(
        "### 🚀 Inference\n"
        "> **Stage 7**: Load a checkpoint and stream generated text.  "
        "**Stage 9**: Finalize run into the model registry.\n\n"
        "*Full controls will be wired in the next implementation phase.*"
    )
