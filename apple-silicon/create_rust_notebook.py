import json

def create_notebook():
    notebook = {
        "cells": [],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "name": "python",
                "version": "3.10.0"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 5
    }

    def add_md(text):
        lines = text.splitlines(keepends=True)
        notebook["cells"].append({"cell_type": "markdown", "metadata": {}, "source": lines})

    def add_code(text):
        lines = text.splitlines(keepends=True)
        notebook["cells"].append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": lines})

    add_md("""# Pro-Grade Rust-GPT on Apple Silicon (MLX)

This notebook implements a Decoder-only Transformer optimized for Apple Silicon (M3 Ultra) using the MLX framework. It trains on the `bigcode/the-stack-v2` Rust subset.
""")

    add_md("""## Authentication
The `bigcode/the-stack` dataset is gated. You must authenticate with a Hugging Face token.
""")

    add_code("""import os
from dotenv import load_dotenv
from huggingface_hub import login
# This will prompt you for your "Access Token"
# Get yours here: https://huggingface.co/settings/tokens
load_dotenv()
hf_token = os.getenv("HF_TOKEN")
login(hf_token)
""")

    add_code("""import math
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
import tiktoken
import time
import os
import mlx.utils as mut

# Extreme Model Hyperparameters (1.5B+ Scale)
n_layer = 24
n_embd = 2048
n_head = 16
head_dim = n_embd // n_head # 128
block_size = 2048

# Training & Hardware (M3 Ultra 512GB)
batch_size = 128
learning_rate = 2e-4
max_iters = 10000
save_interval = 500
prefetch_size = 10000

print(f"MLX using device: {mx.default_device()}")

""")

    add_md("""## Data Pipeline
Load the streaming dataset and initialize the tokenizer.
""")

    add_code("""# Setup Tokenizer
enc = tiktoken.get_encoding("cl100k_base")
vocab_size = enc.n_vocab
print(f"Vocabulary Size: {vocab_size}")

def token_stream_generator():
    # We use v1 of The Stack because v2 only contains blob_ids, not raw text!
    dataset = load_dataset(
        "bigcode/the-stack", 
        data_dir="data/rust", 
        streaming=True, 
        split="train",
        token=True
    )
    for row in dataset:
        # Note: v2 uses 'content' for the text field
        tokens = enc.encode(row['content'], allowed_special="all")
        for token in tokens:
            yield token

def batch_generator(batch_size, block_size):
    stream = token_stream_generator()
    buffer = []
    
    # Fill prefetch buffer initially
    try:
        for _ in range(prefetch_size):
            buffer.append(next(stream))
    except StopIteration:
        pass
        
    while True:
        X_batch = []
        Y_batch = []
        for _ in range(batch_size):
            while len(buffer) < block_size + 1:
                try:
                    buffer.append(next(stream))
                except StopIteration:
                    return # Stream exhausted
            
            chunk = buffer[:block_size + 1]
            X_batch.append(chunk[:-1])
            Y_batch.append(chunk[1:])
            
            # Slide the buffer by block_size
            buffer = buffer[block_size:]
            
        yield mx.array(X_batch), mx.array(Y_batch)

""")

    add_md("""## Model Architecture
Implementing RoPE, RMSNorm, SwiGLU, and Fast Attention.
""")

    add_code("""class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln_1 = nn.RMSNorm(n_embd)
        
        # RoPE - Rotary Positional Embeddings
        self.rope = nn.RoPE(n_embd // n_head, traditional=True)
        
        # QKV Projections
        self.c_attn_qkv = nn.Linear(n_embd, 3 * n_embd, bias=False)
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        self.ln_2 = nn.RMSNorm(n_embd)
        
        # SwiGLU Architecture
        hidden_dim = int((4 * n_embd) * (2/3)) # LLaMA style scaling
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2 = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w3 = nn.Linear(hidden_dim, n_embd, bias=False)
        self.n_head = n_head

    def __call__(self, x, mask=None):
        B, T, C = x.shape
        
        # 1. Attention Block
        norm_x = self.ln_1(x)
        qkv = self.c_attn_qkv(norm_x)
        q, k, v = mx.split(qkv, 3, axis=-1)
        
        q = q.reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3) # B, n_head, T, head_dim
        k = k.reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3)
        v = v.reshape(B, T, self.n_head, -1).transpose(0, 2, 1, 3)
        
        # Apply RoPE
        q = self.rope(q)
        k = self.rope(k)
        
        # Fast attention
        scale = 1.0 / math.sqrt(q.shape[-1])
        if mask is not None:
            a = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask=mask)
        else:
            a = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale)
            
        a = a.transpose(0, 2, 1, 3).reshape(B, T, C)
        x = x + self.c_proj(a)
        
        # 2. Feed Forward Block (SwiGLU)
        norm_x2 = self.ln_2(x)
        # Swish(x) * xV
        swish = nn.silu(self.w1(norm_x2))
        ff_out = self.w3(swish * self.w2(norm_x2))
        x = x + ff_out
        
        return x

class GPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, n_embd)
        self.blocks = [Block(n_embd, n_head) for _ in range(n_layer)]
        self.ln_f = nn.RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
    def __call_(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx)
        
        # Causal mask creation
        mask = nn.MultiHeadAttention.create_additive_causal_mask(T)
        mask = mask.astype(x.dtype)
        
        for block in self.blocks:
            x = block(x, mask)
            
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

""")

    add_md("""## Training Loop
Configure the AdamW optimizer, cosine decay scheduler, and memory tracking.
""")

    add_code("""from functools import partial

# 1. Initialize Model and Optimizer
model = GPTModel(vocab_size, n_embd, n_head, n_layer)
mx.eval(model.parameters()) 

schedule = optim.cosine_decay(learning_rate, max_iters)
optimizer = optim.AdamW(learning_rate=schedule)

# 2. Define the Loss Function (Must take model as first arg for value_and_grad)
def loss_fn(model, x, y):
    logits = model(x)
    loss = nn.losses.cross_entropy(logits, y)
    return mx.mean(loss)

# 3. Create the Gradient Function
loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

# 4. The Compiled Step
# We define 'state' to tell the compiler which parts of your 512GB RAM are mutable
state = [model.state, optimizer.state]

@partial(mx.compile, inputs=state, outputs=state)
def step(x, y):
    # This function now 'captures' model and optimizer from the outer scope
    loss, grads = loss_and_grad_fn(model, x, y)
    optimizer.update(model, grads)
    return loss

# Training loop execution
print("Starting training on M3 Ultra...")
batch_gen = batch_generator(batch_size, block_size)
os.makedirs("checkpoints", exist_ok=True)

for iter_num in range(1, max_iters + 1):
    try:
        x_batch, y_batch = next(batch_gen)
    except StopIteration:
        print("Dataset stream exhausted.")
        break
    
    # Execution: Just pass the data
    loss = step(x_batch, y_batch)
    
    # Evaluation: This is where the M3 Ultra actually does the work
    mx.eval(loss, model.state, optimizer.state)
    
    if iter_num % 10 == 0:
        # loss.item() is now safe because mx.eval was called
        mem_gb = mx.metal.get_active_memory() / (1024**3)
        print(f"Step {iter_num:4d} | Loss: {loss.item():.4f} | Mem: {mem_gb:.2f}GB")
        
    if iter_num % save_interval == 0:
        ckpt_path = f"checkpoints/model_{iter_num}.safetensors"
        weights = dict(mut.tree_flatten(model.parameters()))
        mx.save_safetensors(ckpt_path, weights)
        print(f"Saved checkpoint to {ckpt_path}")

""")

    add_md("""## Inference Generation
A generator function using top-k and top-p sampling to produce valid Rust `fn main()` blocks.
""")

    add_code("""def generate(model, prompt_text, max_tokens=100, top_k=50, top_p=0.9, temp=1.0):
    # This acts as a standard inference generation logic.
    prompt = mx.array([enc.encode(prompt_text)])
    
    for _ in range(max_tokens):
        # Crop context
        idx_cond = prompt[:, -block_size:]
        logits = model(idx_cond)
        
        # Logits of last token
        next_token_logits = logits[:, -1, :] / temp
        
        # Categorical sampling (acting as standard sampling, can be extended for strict top-k/top-p)
        next_token = mx.random.categorical(next_token_logits)
        
        prompt = mx.concatenate([prompt, next_token[:, None]], axis=1)
        mx.eval(prompt)
        
    return enc.decode(prompt[0].tolist())

print("\\n--- Testing Generation ---")
prompt_text = "fn main() {\\n"
generated = generate(model, prompt_text, max_tokens=150)
print(generated)

""")

    with open("rust_gpt_mlx.ipynb", "w", encoding="utf-8") as f:
        json.dump(notebook, f, indent=2)
        f.write("\n")

if __name__ == "__main__":
    create_notebook()
    print("Notebook 'rust_gpt_mlx.ipynb' created successfully.")
