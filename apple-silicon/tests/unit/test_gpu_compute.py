import time
import mlx.core as mx
import mlx.nn as nn
import sys
import os
import math

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from config import TrainingConfig
from model import GroupedQueryAttention

class OptimizedGQA(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_rep = self.n_head // self.n_kv_head
        self.head_dim = config.head_dim
        
        self.wq = nn.Linear(config.n_embd, self.n_head * self.head_dim, bias=False)
        self.wk = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wv = nn.Linear(config.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.wo = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)
        
        self.rope = nn.RoPE(self.head_dim, traditional=False)

    def __call__(self, x, mask=None):
        B, T, _ = x.shape
        xq = self.wq(x).reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        xk = self.wk(x).reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        xv = self.wv(x).reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        
        xq = self.rope(xq)
        xk = self.rope(xk)
        
        # Native MLX SDPA handles broadcasting GQA heads without manual repeat
        scale = 1.0 / math.sqrt(self.head_dim)
        out = mx.fast.scaled_dot_product_attention(xq, xk, xv, scale=scale, mask=mask)
        return self.wo(out.transpose(0, 2, 1, 3).reshape(B, T, -1))

def profile_gqa():
    config = TrainingConfig()
    config.n_head = 16
    config.n_kv_head = 8
    config.n_embd = 2048
    
    x = mx.random.normal((16, 2048, config.n_embd))
    mask = nn.MultiHeadAttention.create_additive_causal_mask(2048)
    
    print("Initializing Original GQA (with manual repeat)...")
    gqa_orig = GroupedQueryAttention(config)
    mx.eval(gqa_orig.parameters())
    
    print("Initializing Optimized GQA (native broadcasting)...")
    gqa_opt = OptimizedGQA(config)
    mx.eval(gqa_opt.parameters())
    
    # Warmup
    _ = gqa_orig(x, mask)
    _ = gqa_opt(x, mask)
    mx.eval(_)
    
    print("\nProfiling Original GQA...")
    start = time.time()
    for _ in range(10):
        out = gqa_orig(x, mask)
        mx.eval(out)
    dt_orig = (time.time() - start) / 10
    print(f"Original GQA Step Time: {dt_orig:.4f}s")
    
    print("\nProfiling Optimized GQA...")
    start = time.time()
    for _ in range(10):
        out = gqa_opt(x, mask)
        mx.eval(out)
    dt_opt = (time.time() - start) / 10
    print(f"Optimized GQA Step Time: {dt_opt:.4f}s")
    
    print(f"\nSpeedup: {dt_orig / dt_opt:.2f}x")

if __name__ == "__main__":
    profile_gqa()
