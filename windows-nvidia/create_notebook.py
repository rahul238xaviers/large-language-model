import json

notebook = {
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {
                "name": "ipython",
                "version": 3
            },
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbconvert_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.0"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

def add_markdown_cell(source):
    notebook["cells"].append({
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split("\n")]
    })

def add_code_cell(source):
    notebook["cells"].append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split("\n")]
    })

add_markdown_cell("# GPT-style Large Language Model (LLM) Training\n\nThis notebook implements a Decoder-only Transformer (GPT architecture) from scratch to perform next-token prediction on '../data/the-verdict.txt'.")

add_code_cell("""import urllib.request
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
import tiktoken
from tqdm import tqdm

# Hyperparameters
batch_size = 4
block_size = 256
n_embd = 128
n_head = 4
n_layer = 4
dropout = 0.1
learning_rate = 1e-3

device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 10
eval_interval = 100
max_iters = 500

print(f"Using device: {device}")""")

add_markdown_cell("## Data Preparation\nDownload the dataset, tokenize it using `tiktoken` (`cl100k_base`), and split it into training (90%) and validation (10%) sets.")

add_code_cell("""# Download dataset
url = "https://raw.githubusercontent.com/rasbt/LLMs-from-scratch/main/ch02/01_main-chapter-code/the-verdict.txt"
filepath = "../data/the-verdict.txt"
try:
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()
except FileNotFoundError:
    print(f"Downloading {filepath}...")
    urllib.request.urlretrieve(url, filepath)
    with open(filepath, "r", encoding="utf-8") as f:
        text = f.read()

print(f"Dataset length in characters: {len(text)}")

# Tokenize
enc = tiktoken.get_encoding("cl100k_base")
tokens = enc.encode(text)
print(f"Dataset length in tokens: {len(tokens)}")
vocab_size = enc.n_vocab
print(f"Vocabulary size: {vocab_size}")

# Split into train and val (90/10)
n = int(0.9 * len(tokens))
train_data = torch.tensor(tokens[:n], dtype=torch.long)
val_data = torch.tensor(tokens[n:], dtype=torch.long)""")

add_markdown_cell("## Dataset Class\nImplement `GPTDataset` to return sliding window chunks of `block_size` tokens for `x` and shifted tokens for `y`.")

add_code_cell("""class GPTDataset(Dataset):
    def __init__(self, data, block_size):
        self.data = data
        self.block_size = block_size

    def __len__(self):
        return len(self.data) - self.block_size

    def __getitem__(self, idx):
        # Get chunk of size block_size + 1
        chunk = self.data[idx:idx + self.block_size + 1]
        x = chunk[:-1] # Inputs
        y = chunk[1:]  # Targets
        return x, y

train_dataset = GPTDataset(train_data, block_size)
val_dataset = GPTDataset(val_data, block_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)""")

add_markdown_cell("## Model Architecture\nBuilding the Transformer components: Multi-Head Attention, FeedForward Network, and the Transformer Block.")

add_code_cell("""class MultiHeadAttention(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        assert n_embd % n_head == 0, "Embedding dimension must be divisible by number of heads"
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        
        # Key, query, value projections for all heads
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Output projection
        self.c_proj = nn.Linear(n_embd, n_embd, bias=False)
        
        # Regularization
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        
        # Causal mask to ensure attention is only applied to past tokens
        self.register_buffer("bias", torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size() # Batch size, Sequence length, Embedding dim

        # Calculate query, key, values for all heads
        qkv = self.c_attn(x)
        q, k, v = qkv.split(C, dim=2)
        
        # Reshape to (B, n_head, T, head_dim)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # Causal self-attention: (B, n_head, T, head_dim) x (B, n_head, head_dim, T) -> (B, n_head, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / (self.head_dim ** 0.5))
        
        # Apply causal mask
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)
        
        # Weighted sum of values: (B, n_head, T, T) x (B, n_head, T, head_dim) -> (B, n_head, T, head_dim)
        y = att @ v
        
        # Re-assemble all head outputs side by side
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.resid_dropout(self.c_proj(y))
        return y""")

add_code_cell("""class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)""")

add_code_cell("""class Block(nn.Module):
    def __init__(self, n_embd, n_head, dropout=0.1):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = MultiHeadAttention(n_embd, n_head, dropout)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.ffwd = FeedForward(n_embd, dropout)

    def forward(self, x):
        # Pre-LayerNorm formulation with residual connections
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x""")

add_code_cell("""class GPTModel(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, dropout=0.1):
        super().__init__()
        self.block_size = block_size
        
        # Token and Position embeddings
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        # Transformer Blocks
        self.blocks = nn.Sequential(*[Block(n_embd, n_head, dropout) for _ in range(n_layer)])
        
        # Final LayerNorm
        self.ln_f = nn.LayerNorm(n_embd)
        
        # Language Model Head
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        
        # Weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        
        # Ensure sequence length does not exceed block_size
        idx = idx[:, -self.block_size:]
        B, T = idx.shape
        
        # Get embeddings
        tok_emb = self.token_embedding_table(idx) # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T, n_embd)
        x = tok_emb + pos_emb # (B, T, n_embd)
        
        # Pass through blocks and final layernorm
        x = self.blocks(x) # (B, T, n_embd)
        x = self.ln_f(x) # (B, T, n_embd)
        
        # Compute logits
        logits = self.lm_head(x) # (B, T, vocab_size)
        
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            logits_reshaped = logits.view(B * T, C)
            targets_reshaped = targets.view(B * T)
            loss = F.cross_entropy(logits_reshaped, targets_reshaped)
            
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # Crop idx to the last block_size tokens
            idx_cond = idx[:, -self.block_size:]
            # Get the predictions
            logits, _ = self(idx_cond)
            # Focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # Apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # Sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # Append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx""")

add_markdown_cell("## Training Setup and Loop\nInitialize the model and AdamW optimizer. Define an evaluation function, and run the training loop.")

add_code_cell("""# Initialize model and optimizer
model = GPTModel(vocab_size, n_embd, n_head, n_layer, block_size, dropout)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split, loader in [('train', train_loader), ('val', val_loader)]:
        losses = torch.zeros(eval_iters)
        # Manually grab batches to estimate loss
        iterator = iter(loader)
        for k in range(eval_iters):
            try:
                X, Y = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                X, Y = next(iterator)
            X, Y = X.to(device), Y.to(device)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# Training Loop
iterator = iter(train_loader)
for iter_num in tqdm(range(max_iters), desc="Training"):
    # Evaluate the loss periodically
    if iter_num % eval_interval == 0 or iter_num == max_iters - 1:
        losses = estimate_loss()
        print(f"step {iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # Sample a batch of data
    try:
        xb, yb = next(iterator)
    except StopIteration:
        iterator = iter(train_loader)
        xb, yb = next(iterator)
        
    xb, yb = xb.to(device), yb.to(device)

    # Forward pass
    logits, loss = model(xb, yb)
    
    # Backward pass and optimize
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()""")

add_markdown_cell("## Inference\nUse the trained model to generate a sequence of tokens.")

add_code_cell("""# Generate text from the trained model
context_text = "The "
context_tokens = enc.encode(context_text)
context_tensor = torch.tensor([context_tokens], dtype=torch.long, device=device)

model.eval()
print("Generating text...")
generated_tokens = model.generate(context_tensor, max_new_tokens=100)
generated_text = enc.decode(generated_tokens[0].tolist())

print("\\n--- Generated Text ---")
print(generated_text)""")

with open("gpt_training.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("gpt_training.ipynb created successfully.")
