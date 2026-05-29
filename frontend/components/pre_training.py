"""Stages 4–5: Tokenisation and MLX Pre-training tab.

Placeholder — full implementation delivered in the next phase.
``render_pretraining_tab`` is the only public symbol.
"""
from __future__ import annotations
import gradio as gr


def render_pretraining_tab(state: gr.State) -> None:
    """Render the Pre-training tab layout inside the current Blocks context."""
    gr.Markdown(
        "### 🧠 Pre-training\n"
        "> **Stages 4 – 5**: Token sequence packing and MLX GQA GPT training with live metrics.\n\n"
        "*Full controls will be wired in the next implementation phase.*"
    )
