import os
import sys
from pathlib import Path
from typing import Any, cast

import gradio as gr
import tiktoken
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
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
        return ""

    global _config, _model, _tokenizer
    prompt_tokens = list(_tokenizer.encode(prompt))
    max_len = _config.block_size
    generated = list(prompt_tokens)

    import mlx.core as mx

    for _ in range(max_new_tokens):
        context = generated[-max_len:]
        x = mx.array([context], dtype=mx.int32)
        logits = _model(x)
        next_token_logits = logits[0, -1]

        # Apply repetition penalty to break infinite loops
        if repetition_penalty != 1.0:
            logits_np = next_token_logits.tolist()
            # Penalize non-whitespace tokens generated in the last 40 steps
            seen = set(generated[-40:])
            for token_id in seen:
                if token_id in [220, 256, 198, 385]:  # Common whitespace tokens in cl100k_base
                    continue
                if logits_np[token_id] > 0:
                    logits_np[token_id] /= repetition_penalty
                else:
                    logits_np[token_id] *= repetition_penalty
            next_token_logits = mx.array(logits_np)

        # Apply Top-K filtering
        if top_k > 0:
            sorted_logits = mx.sort(next_token_logits)
            threshold = sorted_logits[-top_k]
            next_token_logits = mx.where(next_token_logits >= threshold, next_token_logits, mx.array(float('-inf')))

        # Apply Top-P (Nucleus) filtering
        if top_p < 1.0:
            sorted_idx = mx.argsort(-next_token_logits)
            sorted_logits = next_token_logits[sorted_idx]
            probs = mx.softmax(sorted_logits)
            cum_probs = mx.cumsum(probs)
            
            mask = cum_probs > top_p
            mask_list = mask.tolist()
            cutoff_idx = -1
            for idx, val in enumerate(mask_list):
                if val:
                    cutoff_idx = idx
                    break
            
            if cutoff_idx != -1:
                cutoff_idx = max(1, cutoff_idx)
                threshold_val = sorted_logits[cutoff_idx]
                next_token_logits = mx.where(next_token_logits >= threshold_val, next_token_logits, mx.array(float('-inf')))

        if temperature > 0.0:
            next_token_arr = mx.random.categorical(next_token_logits / temperature)
        else:
            next_token_arr = mx.argmax(next_token_logits, axis=-1)

        if hasattr(next_token_arr, "item"):
            next_token = int(next_token_arr.item())
        else:
            next_token = int(next_token_arr)
        generated.append(next_token)
        if next_token == _tokenizer.eot_token:
            break

    generated_only = generated[len(prompt_tokens):]
    return _tokenizer.decode(generated_only)


def build_gradio():
    with gr.Blocks() as demo:
        gr.Markdown("## GPT Model Chat")
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", lines=4, placeholder="Type your message here...")
        with gr.Row():
            max_tokens = gr.Slider(1, 256, value=64, step=1, label="Max new tokens")
            temperature = gr.Slider(0.1, 2.0, value=0.7, step=0.1, label="Temperature")
            repetition_penalty = gr.Slider(1.0, 2.0, value=1.1, step=0.05, label="Repetition Penalty")
        with gr.Row():
            top_k = gr.Slider(0, 100, value=50, step=1, label="Top-K (0 to disable)")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-P (1.0 to disable)")
        output = gr.Textbox(label="Response", lines=10)
        submit = gr.Button("Generate")
        submit.click(generate_completion, [prompt, max_tokens, temperature, repetition_penalty, top_k, top_p], output)
        gr.Markdown("*The model uses the checkpoint path set by `CHECKPOINT_PATH` in `.env`. Use a short prompt to avoid long decode time.*")
    return demo


if __name__ == "__main__":
    _config, _model, _tokenizer = build_generator()
    demo = build_gradio()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
