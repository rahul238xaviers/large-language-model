import math
import mlx.core as mx
import mlx.nn as nn
import mlx.utils as mut
from config import TrainingConfig

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps
    def __call__(self, x):
        return mx.fast.rms_norm(x, self.weight, self.eps)

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.head_dim = config.head_dim
        self.q_size = self.n_head * self.head_dim
        self.kv_size = self.n_kv_head * self.head_dim
        self.wqkv = nn.Linear(config.n_embd, self.q_size + 2 * self.kv_size, bias=False)
        self.wo = nn.Linear(self.n_head * self.head_dim, config.n_embd, bias=False)
        self.rope = nn.RoPE(self.head_dim, traditional=False)

    def __call__(self, x, mask=None, cache=None):
        B, T, _ = x.shape
        qkv = self.wqkv(x)
        xq = qkv[:, :, :self.q_size].reshape(B, T, self.n_head, self.head_dim).transpose(0, 2, 1, 3)
        xk = qkv[:, :, self.q_size:self.q_size + self.kv_size].reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        xv = qkv[:, :, self.q_size + self.kv_size:].reshape(B, T, self.n_kv_head, self.head_dim).transpose(0, 2, 1, 3)
        
        if cache is not None:
            xq = self.rope(xq, offset=cache.offset)
            xk = self.rope(xk, offset=cache.offset)
            xk, xv = cache.update(xk, xv)
        else:
            xq = self.rope(xq)
            xk = self.rope(xk)
        
        scale = 1.0 / math.sqrt(self.head_dim)
        out = mx.fast.scaled_dot_product_attention(xq, xk, xv, scale=scale, mask=mask)
        return self.wo(out.transpose(0, 2, 1, 3).reshape(B, T, -1))

class FeedForward(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.hidden_dim = int((4 * config.n_embd) * 2 / 3)
        self.w12 = nn.Linear(config.n_embd, 2 * self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, config.n_embd, bias=False)
    def __call__(self, x):
        out12 = self.w12(x)
        out1, out2 = mx.split(out12, 2, axis=-1)
        return self.w3(nn.silu(out1) * out2)

class TransformerBlock(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.attention_norm = RMSNorm(config.n_embd)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.feed_forward = FeedForward(config)
    def __call__(self, x, mask=None, cache=None):
        x = x + self.attention(self.attention_norm(x), mask, cache)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
        self.layers = [TransformerBlock(config) for _ in range(config.n_layer)]
        self.norm = RMSNorm(config.n_embd)
        self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Causal mask cache
        self._mask = None
        self._mask_T = -1

    def tie_weights(self):
        """Perform weight tying after precision conversion."""
        self.output.weight = self.tok_embeddings.weight

    def set_dtype(self, dtype):
        """Compatibility helper used by existing benchmark/test scripts."""
        self.update(mut.tree_map(lambda p: p.astype(dtype), self.parameters()))

    def __call__(self, idx, mask=None, cache=None):
        B, T = idx.shape
        if mask is None and cache is None:
            if T != self._mask_T:
                self._mask = nn.MultiHeadAttention.create_additive_causal_mask(T).astype(self.tok_embeddings.weight.dtype)
                self._mask_T = T
            mask = self._mask
        x = self.tok_embeddings(idx)
        for layer in self.layers:
            x = layer(x, mask, cache)
        return self.output(self.norm(x))
