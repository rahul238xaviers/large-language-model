import os
import sys
from pathlib import Path
from typing import Any, cast

import gradio as gr
import gradio.themes as gradio_themes
import tiktoken
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "pre-training"
sys.path.insert(0, str(SRC))

from config import TrainingConfig
from model import GPTModel


def load_checkpoint(model: GPTModel, checkpoint_path: str):
    import mlx.core as mx

    if hasattr(model, "load_weights"):
        model.load_weights(checkpoint_path)
        return

    load_fn = getattr(mx, "load_safetensors", None)
    if load_fn is not None:
        state = load_fn(checkpoint_path)
    else:
        from safetensors import safe_open

        state = {}
        with safe_open(checkpoint_path, framework="numpy") as f:
            for key in f.keys():
                state[key] = f.get_tensor(key)
    model.update(state)


def build_generator():
    load_dotenv()
    config = TrainingConfig()
    checkpoint_path = os.getenv("CHECKPOINT_PATH")
    if not checkpoint_path:
        raise RuntimeError("CHECKPOINT_PATH is not set in the environment")
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    model = GPTModel(config)
    model.tie_weights()
    model.set_dtype(config.mx_dtype)
    load_checkpoint(model, str(checkpoint_path))
    model.tie_weights()

    tokenizer = tiktoken.get_encoding("cl100k_base")

    return config, model, tokenizer


def generate_completion(prompt: str, max_new_tokens: int = 64, temperature: float = 1.0, repetition_penalty: float = 1.2, top_k: int = 50, top_p: float = 0.9):
    if not prompt:
        yield ""
        return

    global _config, _model, _tokenizer
    prompt_tokens = list(_tokenizer.encode(prompt))
    max_len = _config.block_size
    generated = list(prompt_tokens)

    # Cast variables to concrete types to resolve Pylance static type analysis warnings
    temp_val = float(temperature)
    rep_val = float(repetition_penalty)
    k_val = int(top_k)
    p_val = float(top_p)

    import mlx.core as mx

    for _ in range(max_new_tokens):
        context = generated[-max_len:]
        x = mx.array([context], dtype=mx.int32)
        logits = _model(x)
        next_token_logits = logits[0, -1]

        # Apply repetition penalty to break infinite loops
        if rep_val != 1.0:
            logits_list = next_token_logits.tolist()
            if isinstance(logits_list, list):
                logits_np = cast(list[float], logits_list)
                # Penalize non-whitespace tokens generated in the last 128 steps
                seen = set(generated[-128:])
                for token_id in seen:
                    if token_id in [220, 256, 198, 385]:  # Common whitespace tokens in cl100k_base
                        continue
                    if logits_np[token_id] > 0.0:
                        logits_np[token_id] /= rep_val
                    else:
                        logits_np[token_id] *= rep_val
                    # Additive penalty to guarantee high-confidence loops are broken
                    logits_np[token_id] -= (rep_val - 1.0) * 35.0
                next_token_logits = mx.array(logits_np)

        # Apply Top-K filtering
        if k_val > 0:
            sorted_logits = mx.sort(next_token_logits)
            threshold = sorted_logits[-k_val]
            next_token_logits = mx.where(next_token_logits >= threshold, next_token_logits, mx.array(float('-inf')))

        # Apply Top-P (Nucleus) filtering
        if p_val < 1.0:
            sorted_idx = mx.argsort(-next_token_logits)
            sorted_logits = next_token_logits[sorted_idx]
            probs = mx.softmax(sorted_logits)
            cum_probs = mx.cumsum(probs)
            
            mask = cum_probs > p_val
            mask_list = cast(list[bool], mask.tolist())
            cutoff_idx = -1
            for idx, val in enumerate(mask_list):
                if val:
                    cutoff_idx = idx
                    break
            
            if cutoff_idx != -1:
                cutoff_idx = max(1, cutoff_idx)
                threshold_val = sorted_logits[cutoff_idx]
                next_token_logits = mx.where(next_token_logits >= threshold_val, next_token_logits, mx.array(float('-inf')))

        if temp_val > 0.0:
            next_token_arr = mx.random.categorical(next_token_logits / temp_val)
        else:
            next_token_arr = mx.argmax(next_token_logits, axis=-1)

        if hasattr(next_token_arr, "item"):
            next_token = int(next_token_arr.item())
        else:
            next_token = int(next_token_arr)
        if next_token == _tokenizer.eot_token:
            break

        generated.append(next_token)

        # Yield the generated-so-far tokens to stream them in real time
        generated_only = generated[len(prompt_tokens):]
        yield _tokenizer.decode(generated_only)


def build_gradio():
    # Premium Custom Styling & Glassmorphism Theme
    theme = gradio_themes.Soft(
        primary_hue="orange",
        secondary_hue="slate",
        neutral_hue="stone",
        font=[gradio_themes.GoogleFont("Outfit"), "sans-serif"],
        font_mono=[gradio_themes.GoogleFont("Fira Code"), "monospace"],
    ).set(
        body_background_fill="*neutral_950",
        body_background_fill_dark="*neutral_950",
        block_background_fill="*neutral_900",
        block_background_fill_dark="*neutral_900",
        block_border_width="1px",
        block_border_color="*neutral_800",
        block_shadow="0 4px 20px 0 rgba(0, 0, 0, 0.4)",
        button_primary_background_fill="linear-gradient(90deg, #f97316 0%, #ea580c 100%)",
        button_primary_background_fill_hover="linear-gradient(90deg, #ea580c 0%, #c2410c 100%)",
        button_primary_text_color="white",
        slider_color="#ea580c",
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

    with gr.Blocks(title="Rust-GPT Play") as demo:
        # Header block
        with gr.Row():
            with gr.Column():
                gr.Markdown("<h1 class='gradient-text'>🦀 Rust-GPT Playground</h1>")
                gr.Markdown("<p class='header-desc'>State-of-the-art <b>1.6B parameter</b> Decoder-only model optimized via MLX for Apple Silicon.</p>")

        # Collapsible Hyperparameters (Saves 100% of vertical space, keeps app inside single screen frame)
        with gr.Accordion("⚙️ Advanced Model Hyperparameters (Click to expand)", open=False, elem_classes=["panel-glass"]):
            with gr.Row():
                temperature = gr.Slider(0.0, 2.0, value=0.7, step=0.05, label="Temperature")
                top_k = gr.Slider(0, 100, value=50, step=1, label="Top-K")
                top_p = gr.Slider(0.1, 1.0, value=0.90, step=0.05, label="Top-P")
            with gr.Row():
                max_tokens = gr.Slider(1, 512, value=128, step=1, label="Max New Tokens")
                repetition_penalty = gr.Slider(1.0, 2.0, value=1.15, step=0.05, label="Repetition Penalty")

        # Side-by-Side Workspace Layout (Clean, balanced look with Copy-to-Clipboard buttons via gr.Code)
        with gr.Row():
            with gr.Column(scale=1, elem_classes=["panel-glass"]):
                gr.Markdown("### 📝 Code Prompt")
                prompt = gr.Code(
                    label="Input Context",
                    value="fn main() {\n    // Type your Rust context or code prefix here...\n}",
                    lines=12,
                    show_label=False,
                    language=None,
                    interactive=True
                )
                
            with gr.Column(scale=1, elem_classes=["panel-glass"]):
                gr.Markdown("### 💻 Output")
                output = gr.Code(
                    label="Completed Code",
                    lines=12,
                    show_label=False,
                    language=None,
                    interactive=False
                )
        
        # Centered, compact buttons (Short and beautiful)
        with gr.Row():
            with gr.Column(scale=1):
                submit = gr.Button("🦀 Generate Code", variant="primary", elem_classes=["generate-btn"])
            with gr.Column(scale=1):
                clear = gr.Button("🧹 Clear", variant="secondary", elem_classes=["clear-btn"])

        # Link actions
        submit.click(
            generate_completion, 
            inputs=[prompt, max_tokens, temperature, repetition_penalty, top_k, top_p], 
            outputs=output
        )
        
        # Clear action
        def clear_inputs():
            return "", ""
        clear.click(clear_inputs, outputs=[prompt, output])

        gr.Markdown("<p class='footer-text'>⚡ Running on Apple Silicon via MLX.</p>")

    head = """
    <script>
    (function() {
        // Fallback copying function using temporary off-screen textarea
        function fallbackCopyText(text) {
            var textArea = document.createElement("textarea");
            textArea.value = text;
            textArea.style.position = "fixed";
            textArea.style.top = "0";
            textArea.style.left = "0";
            textArea.style.width = "2em";
            textArea.style.height = "2em";
            textArea.style.padding = "0";
            textArea.style.border = "none";
            textArea.style.outline = "none";
            textArea.style.boxShadow = "none";
            textArea.style.background = "transparent";
            document.body.appendChild(textArea);
            textArea.focus();
            textArea.select();
            try {
                document.execCommand("copy");
            } catch (err) {
                console.error("Fallback copy failed", err);
            }
            document.body.removeChild(textArea);
        }

        // If navigator.clipboard is undefined (insecure HTTP context), polyfill it
        if (typeof navigator.clipboard === "undefined") {
            Object.defineProperty(navigator, "clipboard", {
                value: {
                    writeText: function(text) {
                        return new Promise(function(resolve, reject) {
                            try {
                                fallbackCopyText(text);
                                resolve();
                            } catch (err) {
                                reject(err);
                            }
                        });
                    }
                },
                writable: true,
                configurable: true
            });
        } else {
            // If it exists, wrap writeText to fallback if the native call fails
            var nativeWriteText = navigator.clipboard.writeText;
            navigator.clipboard.writeText = function(text) {
                return nativeWriteText.call(navigator.clipboard, text).catch(function(err) {
                    fallbackCopyText(text);
                });
            };
        }
    })();
    </script>
    """

    return demo, theme, css, head


if __name__ == "__main__":
    _config, _model, _tokenizer = build_generator()
    demo, theme, css, head = build_gradio()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False, theme=theme, css=css, head=head)
