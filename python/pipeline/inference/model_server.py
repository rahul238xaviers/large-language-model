"""Inference model server.

Loads a GPT checkpoint, exposes a streaming ``generate_text()`` generator,
and optionally launches a Gradio playground UI.

Gradio is a **soft dependency**: if it is not installed the server still
works in headless mode; only ``launch_gradio()`` will raise ``ImportError``.
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# InferenceServer                                                       #
# ------------------------------------------------------------------ #

class InferenceServer:
    """Loads a checkpoint and provides streaming text generation.

    Parameters
    ----------
    checkpoint_path:
        Path to a ``.safetensors`` checkpoint file produced by Stage 5.
    model_cfg:
        Dict with model architecture keys (``n_layer``, ``n_embd``, …).
        If *None*, defaults from ``configs/base/training.yaml`` are used.
    inference_cfg:
        Dict with decode defaults (``max_new_tokens``, ``temperature``, …).
    """

    def __init__(
        self,
        checkpoint_path: Path | str,
        model_cfg: dict | None = None,
        inference_cfg: dict | None = None,
    ) -> None:
        self._ckpt_path   = Path(checkpoint_path)
        self._model_cfg   = model_cfg or {}
        self._inf_cfg     = inference_cfg or {}
        self._model: Any  = None
        self._tokenizer: Any = None
        self._block_size: int = self._model_cfg.get("block_size", 2048)

    # ---------------------------------------------------------------- #
    # Loading                                                            #
    # ---------------------------------------------------------------- #

    def load(self) -> "InferenceServer":
        """Load model weights and tokenizer into memory."""
        import tiktoken
        from pipeline.evaluation.checkpoint_loader import load_checkpoint

        logger.info("Loading checkpoint: %s", self._ckpt_path)
        self._model     = load_checkpoint(self._ckpt_path, self._model_cfg)
        self._tokenizer = tiktoken.get_encoding("cl100k_base")
        self._block_size = self._model_cfg.get("block_size", 2048)
        logger.info("Model + tokenizer ready")
        return self

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def _ensure_loaded(self) -> None:
        if not self.is_loaded:
            raise RuntimeError("Call InferenceServer.load() before generating text.")

    # ---------------------------------------------------------------- #
    # Text generation                                                    #
    # ---------------------------------------------------------------- #

    def generate_text(
        self,
        prompt: str,
        *,
        max_new_tokens: int | None = None,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        repetition_penalty: float | None = None,
    ) -> Iterator[str]:
        """Yield decoded text fragments one token at a time.

        Keyword overrides take precedence over ``inference_cfg`` defaults.
        """
        self._ensure_loaded()
        from pipeline.inference.sampler import SamplerConfig, generate

        cfg = SamplerConfig.from_dict({
            **self._inf_cfg,
            **({"max_new_tokens":     max_new_tokens}     if max_new_tokens     is not None else {}),
            **({"temperature":        temperature}        if temperature        is not None else {}),
            **({"top_k":              top_k}              if top_k              is not None else {}),
            **({"top_p":              top_p}              if top_p              is not None else {}),
            **({"repetition_penalty": repetition_penalty} if repetition_penalty is not None else {}),
        })

        prompt_ids = list(self._tokenizer.encode(prompt))
        eot_id     = self._tokenizer.eot_token

        cumulative = ""
        for token_id in generate(
            model       = self._model,
            prompt_ids  = prompt_ids,
            eot_id      = eot_id,
            block_size  = self._block_size,
            cfg         = cfg,
        ):
            cumulative = self._tokenizer.decode([token_id])
            yield cumulative

    # ---------------------------------------------------------------- #
    # Gradio UI                                                          #
    # ---------------------------------------------------------------- #

    def launch_gradio(self, port: int = 7860, server_name: str = "0.0.0.0", share: bool = False) -> None:
        """Build and launch the Gradio playground.

        Requires ``gradio`` to be installed (``pip install gradio>=4.0``).
        The UI is a faithful port of the legacy ``apple-silicon/tests/functional/gradio_app.py``
        with the same premium theme, but model loading is delegated to this server
        rather than importing training modules directly.
        """
        try:
            import gradio as gr
            import gradio.themes as gradio_themes
        except ImportError as exc:
            raise ImportError(
                "Gradio is required for the playground UI. "
                "Install it with: pip install 'gradio>=4.0'"
            ) from exc

        self._ensure_loaded()
        server = self  # capture for closure

        def _generate_streaming(
            prompt: str,
            max_new_tokens: int,
            temperature: float,
            repetition_penalty: float,
            top_k: int,
            top_p: float,
        ):
            if not prompt:
                yield ""
                return
            accumulated = ""
            for fragment in server.generate_text(
                prompt,
                max_new_tokens     = max_new_tokens,
                temperature        = temperature,
                top_k              = top_k,
                top_p              = top_p,
                repetition_penalty = repetition_penalty,
            ):
                accumulated += fragment
                yield accumulated

        # ── Premium Glassmorphism Theme (ported from legacy app) ── #
        theme = gradio_themes.Soft(
            primary_hue   = "orange",
            secondary_hue = "slate",
            neutral_hue   = "stone",
            font      = [gradio_themes.GoogleFont("Outfit"),    "sans-serif"],
            font_mono = [gradio_themes.GoogleFont("Fira Code"), "monospace"],
        ).set(
            body_background_fill      = "*neutral_950",
            body_background_fill_dark = "*neutral_950",
            block_background_fill      = "*neutral_900",
            block_background_fill_dark = "*neutral_900",
            block_border_width  = "1px",
            block_border_color  = "*neutral_800",
            block_shadow        = "0 4px 20px 0 rgba(0, 0, 0, 0.4)",
            button_primary_background_fill       = "linear-gradient(90deg, #f97316 0%, #ea580c 100%)",
            button_primary_background_fill_hover = "linear-gradient(90deg, #ea580c 0%, #c2410c 100%)",
            button_primary_text_color = "white",
            slider_color = "#ea580c",
        )

        css = """
        .gradient-text {
            background: linear-gradient(90deg, #ff7e33 0%, #ea580c 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 1.8rem;
            margin-bottom: 0.1rem;
            text-align: center;
        }
        .header-desc {
            color: #a8a29e;
            font-size: 0.95rem;
            margin-bottom: 0.8rem;
            text-align: center;
        }
        .panel-glass {
            background: rgba(28, 25, 23, 0.65) !important;
            backdrop-filter: blur(16px);
            border: 1px solid rgba(234, 88, 12, 0.15) !important;
            border-radius: 12px !important;
            padding: 12px !important;
            margin-bottom: 8px;
        }
        .generate-btn {
            box-shadow: 0 4px 15px rgba(234, 88, 12, 0.3);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            font-weight: 700 !important;
            font-size: 1rem !important;
            max-width: 220px !important;
            margin: 0 auto !important;
            display: block !important;
            height: 42px !important;
        }
        .generate-btn:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(234, 88, 12, 0.5);
        }
        .clear-btn {
            max-width: 120px !important;
            margin: 0 auto !important;
            display: block !important;
            height: 42px !important;
        }
        textarea {
            font-family: 'Fira Code', 'Courier New', Courier, monospace !important;
            font-size: 0.9rem !important;
        }
        .footer-text {
            color: #78716c;
            font-size: 0.8rem;
            text-align: center;
            margin-top: 8px;
        }
        """

        ckpt_name = self._ckpt_path.stem

        with gr.Blocks(title="GPT Playground", css=css, theme=theme) as demo:
            with gr.Row():
                with gr.Column():
                    gr.Markdown("<h1 class='gradient-text'>GPT Playground</h1>")
                    gr.Markdown(
                        f"<p class='header-desc'>Checkpoint: <code>{ckpt_name}</code>"
                        " &nbsp;|&nbsp; Powered by MLX on Apple Silicon</p>"
                    )

            with gr.Accordion("⚙️ Decode Hyperparameters", open=False, elem_classes=["panel-glass"]):
                with gr.Row():
                    temperature        = gr.Slider(0.0, 2.0,  value=self._inf_cfg.get("temperature",        0.7),  step=0.05, label="Temperature")
                    top_k              = gr.Slider(0,   100,  value=self._inf_cfg.get("top_k",              50),   step=1,    label="Top-K")
                    top_p              = gr.Slider(0.1, 1.0,  value=self._inf_cfg.get("top_p",              0.9),  step=0.05, label="Top-P")
                with gr.Row():
                    max_tokens         = gr.Slider(1,   512,  value=self._inf_cfg.get("max_new_tokens",     128),  step=1,    label="Max New Tokens")
                    repetition_penalty = gr.Slider(1.0, 2.0,  value=self._inf_cfg.get("repetition_penalty", 1.15), step=0.05, label="Repetition Penalty")

            with gr.Row():
                with gr.Column(scale=1, elem_classes=["panel-glass"]):
                    gr.Markdown("### Prompt")
                    prompt_box = gr.Code(
                        label       = "Input Context",
                        value       = "fn main() {\n    // Start typing your prompt here...\n}",
                        lines       = 12,
                        show_label  = False,
                        language    = None,
                        interactive = True,
                    )
                with gr.Column(scale=1, elem_classes=["panel-glass"]):
                    gr.Markdown("### Completion")
                    output_box = gr.Code(
                        label       = "Generated Text",
                        lines       = 12,
                        show_label  = False,
                        language    = None,
                        interactive = False,
                    )

            with gr.Row():
                with gr.Column(scale=1):
                    submit_btn = gr.Button("Generate", variant="primary", elem_classes=["generate-btn"])
                with gr.Column(scale=1):
                    clear_btn = gr.Button("Clear", variant="secondary", elem_classes=["clear-btn"])

            submit_btn.click(
                fn      = _generate_streaming,
                inputs  = [prompt_box, max_tokens, temperature, repetition_penalty, top_k, top_p],
                outputs = output_box,
            )
            clear_btn.click(fn=lambda: ("", ""), outputs=[prompt_box, output_box])

            gr.Markdown("<p class='footer-text'>Running on Apple Silicon via MLX.</p>")

        logger.info("Launching Gradio on http://%s:%d", server_name, port)
        demo.launch(server_name=server_name, server_port=port, share=share)
