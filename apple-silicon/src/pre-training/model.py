import math
import mlx.core as mx
import mlx.nn as nn
from typing import Callable, Optional
from config import TrainingConfig
from utils import timed_log

class RMSNorm(nn.Module):
    def __init__(self, dims: int, eps: float = 1e-5):
        """
        Initialize Root Mean Square normalization.

        Unlike LayerNorm, RMSNorm skips mean-centering and only scales by the
        RMS of each vector, reducing compute by one reduction operation.

        Learnable scale γ (`self.weight`) is initialized to ones so the layer
        is an identity at the start of training.

        Args:
            dims: Feature dimension d (must match the last axis of inputs).
            eps:  Small constant added inside the sqrt to prevent division by
                  zero when the input vector is all-zeros.

        Example:
            norm = RMSNorm(dims=2048)
            # norm.weight.shape == (2048,)  — all ones initially
        """
        super().__init__()
        self.weight = mx.ones((dims,))
        self.eps = eps

    def __call__(self, x):
        """
        Apply RMS normalization to x.

        Math:
            rms(x) = sqrt( (1/d) * sum_i(x_i^2) + eps )
            out    = (x / rms(x)) * γ

        The division by rms(x) makes each token vector unit-norm (in the
        RMS sense). Multiplying by the learned γ restores expressive capacity.

        Args:
            x: Tensor of shape (..., dims).

        Returns:
            Tensor of the same shape with each vector rescaled.

        Example:
            x   = mx.ones((2, 4, 2048))  # batch=2, seq=4, d=2048
            out = norm(x)               # out.shape == (2, 4, 2048)
            # Each of the 2×4 token vectors is divided by its RMS then scaled by γ
        """
        return mx.fast.rms_norm(x, self.weight, self.eps)

class GroupedQueryAttention(nn.Module):
    def __init__(self, config: TrainingConfig):
        """
        Grouped Query Attention (GQA) layer.

        GQA generalizes Multi-Head Attention (MHA) and Multi-Query Attention (MQA):
        - MHA: n_head Q heads, each paired with its own K and V head (n_kv_head == n_head).
        - MQA: all Q heads share a single K/V pair        (n_kv_head == 1).
        - GQA: Q heads are divided into groups; each group shares one K/V head.

        This model uses n_head=16, n_kv_head=8 → 2 Q heads per K/V group.
        That halves K/V memory and bandwidth vs MHA with minimal quality loss.

        Shapes after initialisation:
            wqkv weight: (n_embd,  n_head*head_dim + 2*n_kv_head*head_dim)
                       = (2048,    16*128 + 2*8*128) = (2048, 4096)
            wo   weight: (n_head*head_dim, n_embd)  = (2048, 2048)

        Args:
            config: TrainingConfig carrying n_head, n_kv_head, head_dim, n_embd.

        Example:
            attn = GroupedQueryAttention(config)
            # attn.wqkv.weight.shape == (4096, 2048)  [MLX stores weights transposed]
        """
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
        """
        Grouped Query Attention forward pass with RoPE positional encoding.

        Math (per attention head h):
            Q_h = x · W_Q_h    shape: (B, n_head,    T, head_dim)
            K_h = x · W_K_h    shape: (B, n_kv_head, T, head_dim)
            V_h = x · W_V_h    shape: (B, n_kv_head, T, head_dim)

            # RoPE rotates Q and K by position-dependent angles:
            Q_h, K_h ← RoPE(Q_h, K_h)

            # Scaled dot-product attention (causal when mask is provided):
            Attn_h = softmax( (Q_h · K_h^T) / sqrt(head_dim) + mask ) · V_h

            # Concatenate all heads and project back:
            out = concat(Attn_0, ..., Attn_{n_head-1}) · W_o

        MLX's `scaled_dot_product_attention` handles K/V head broadcasting
        internally, so no explicit repeat is needed in user code.

        Args:
            x:     Input tensor, shape (B, T, n_embd).
            mask:  Additive causal mask, shape (T, T); -inf for masked positions.
            cache: Optional KV-cache for autoregressive inference.

        Returns:
            Output tensor, shape (B, T, n_embd).

        Example:
            x   = mx.zeros((2, 4, 2048))          # batch=2, seq=4
            out = attn(x, mask=causal_mask)        # out.shape == (2, 4, 2048)
        """
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
        """
        SwiGLU Feed-Forward Network (FFN) block.

        The hidden dimension follows the LLaMA convention:
            hidden_dim = int( (4 * n_embd) * 2/3 )

        For n_embd=2048: hidden_dim = int(8192 * 0.667) = 5461.

        Two conceptually separate projections (W1 gate, W2 up) are fused into
        a single matrix `w12` of shape (n_embd, 2*hidden_dim) to halve the
        number of matmul kernel launches on the GPU.

        Args:
            config: TrainingConfig with n_embd field.

        Example:
            ffn = FeedForward(config)
            # ffn.w12.weight.shape == (2*5461, 2048) = (10922, 2048)
            # ffn.w3.weight.shape  == (2048,  5461)
        """
        super().__init__()
        self.hidden_dim = int((4 * config.n_embd) * 2 / 3)
        self.w12 = nn.Linear(config.n_embd, 2 * self.hidden_dim, bias=False)
        self.w3 = nn.Linear(self.hidden_dim, config.n_embd, bias=False)

    def __call__(self, x):
        """
        SwiGLU gated activation forward pass.

        Math:
            [gate, up] = split( x · W12, dim=-1 )   # each half is (B, T, hidden_dim)
            hidden     = SiLU(gate) * up             # element-wise gating
            out        = hidden · W3                 # project back to n_embd

        SiLU (Sigmoid Linear Unit / Swish):
            SiLU(z) = z * sigmoid(z) = z / (1 + e^{-z})

        Compared to ReLU-FFN, the multiplicative gate lets the network suppress
        features by learning gate ≈ 0, improving gradient flow and model quality.

        Args:
            x: Input tensor, shape (B, T, n_embd).

        Returns:
            Output tensor, shape (B, T, n_embd).

        Example:
            x   = mx.zeros((2, 4, 2048))
            out = ffn(x)   # out.shape == (2, 4, 2048)
            # gate path: SiLU(x·W1) controls which features are let through
            # up   path: x·W2 provides the values
        """
        out12 = self.w12(x)
        out1, out2 = mx.split(out12, 2, axis=-1)
        return self.w3(nn.silu(out1) * out2)

class TransformerBlock(nn.Module):
    def __init__(self, config: TrainingConfig):
        """
        A single Pre-Norm Transformer block (attention + FFN with residual connections).

        Architecture ("Pre-LN" formulation used by LLaMA / GPT-NeoX):
            1. Attention sub-layer:  x ← x + GQA( RMSNorm(x) )
            2. FFN sub-layer:        x ← x + FFN( RMSNorm(x) )

        Normalizing before the sub-layer (Pre-LN) rather than after (Post-LN)
        stabilises gradients in deep models by preventing the residual stream
        from growing unbounded.

        Args:
            config: TrainingConfig.

        Example:
            block = TransformerBlock(config)
            # block.attention: GroupedQueryAttention
            # block.feed_forward: FeedForward (SwiGLU)
        """
        super().__init__()
        self.attention_norm = RMSNorm(config.n_embd)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.n_embd)
        self.feed_forward = FeedForward(config)

    def __call__(self, x, mask=None, cache=None):
        """
        Forward pass through one Pre-Norm Transformer block.

        Math:
            h = x + GQA( RMSNorm(x),  mask, cache )   # attention residual
            y = h + FFN( RMSNorm(h) )                  # FFN residual

        The residual additions allow gradients to flow unimpeded through the
        entire depth of the network ("highway" connections).

        Args:
            x:     Input residual stream, shape (B, T, n_embd).
            mask:  Causal mask passed to attention.
            cache: Optional KV cache for inference.

        Returns:
            Updated residual stream of the same shape.

        Example:
            x_in  = mx.zeros((2, 4, 2048))
            x_out = block(x_in, mask=causal_mask)  # x_out.shape == (2, 4, 2048)
        """
        x = x + self.attention(self.attention_norm(x), mask, cache)
        x = x + self.feed_forward(self.ffn_norm(x))
        return x

class GPTModel(nn.Module):
    def __init__(self, config: TrainingConfig):
        """
        Build a 1.518B-parameter GPT model with the following architecture:

            tok_embeddings : Embedding(vocab_size=100277, n_embd=2048)
            layers         : 24 × TransformerBlock (GQA + SwiGLU + RMSNorm)
            norm           : RMSNorm(n_embd=2048)   — final layer norm
            output         : Linear(n_embd=2048, vocab_size=100277, bias=False)

        Weight tying: `output.weight` is shared with `tok_embeddings.weight`
        (same tensor), reducing parameter count by ~205M and aligning the
        embedding and unembedding spaces.

        Parameter budget (approximate, before weight tying):
            tok_embeddings : 100277 × 2048 ≈  205M
            24 × attn       : 24 × (2048×4096 + 2048×2048) ≈ 302M
            24 × ffn        : 24 × (2048×10922 + 5461×2048) ≈ 804M
            norms           : negligible
            output          : tied → 0 additional
            ─────────────────────────────────────────────────────
            Total           : ~1.518B parameters

        Args:
            config: TrainingConfig carrying all architecture hyperparameters.
        """
        super().__init__()
        self.config = config
        with timed_log("GPTModel.__init__", getattr(config, "profile_methods", False)):
            self.tok_embeddings = nn.Embedding(config.vocab_size, config.n_embd)
            self.layers = [TransformerBlock(config) for _ in range(config.n_layer)]
            self.norm = RMSNorm(config.n_embd)
            self.output = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        
        # Causal mask cache
        self._mask = None
        self._mask_T = -1

    def tie_weights(self):
        """
        Share the token embedding matrix with the output projection (weight tying).

        By assigning `output.weight = tok_embeddings.weight`, both layers read
        from the same underlying tensor in memory.  This means:
            - Gradients from the output softmax flow back through the same
              matrix that receives embedding gradients → faster alignment.
            - ~205M fewer stored parameters (100277 × 2048 = 205,367,296 values).

        Must be called after every `model.update(...)` or `set_dtype(...)` call
        that replaces parameter tensors, because MLX `update()` creates new
        tensor objects, breaking the reference.

        Example:
            model.tie_weights()
            assert model.output.weight is model.tok_embeddings.weight  # True
        """
        with timed_log("GPTModel.tie_weights", getattr(self.config, "profile_methods", False)):
            self.output.weight = self.tok_embeddings.weight

    def set_dtype(
        self,
        dtype,
        predicate: Optional[Callable[[mx.Dtype], bool]] = lambda x: mx.issubdtype(x, mx.floating),
        **kwargs
    ):
        """
        Cast all floating-point parameters to `dtype` in place.

        Wraps `nn.Module.set_dtype` with a custom predicate that limits the
        cast to floating-point leaves, leaving integer or bool tensors untouched.

        The `predicate` keyword-only argument matches the MLX Module base
        class signature; passing it as keyword avoids a TypeError on Python 3.13
        where positional-or-keyword collision with the base signature raises.

        Args:
            dtype:     Target MLX dtype, e.g. `mx.bfloat16` or `mx.float32`.
            predicate: A function (MLX dtype) → bool controlling which leaves
                       are cast.  Defaults to all floating subtypes.
            **kwargs:  Forwarded to `nn.Module.set_dtype`.

        Example:
            model.set_dtype(mx.bfloat16)
            # All weight matrices are now bfloat16; tie_weights() still needed
            # after this to re-share the embedding / output tensor.
        """
        with timed_log("GPTModel.set_dtype", getattr(self.config, "profile_methods", False)):
            super().set_dtype(dtype, predicate=predicate, **kwargs)

    def __call__(self, idx, mask=None, cache=None):
        """
        Full GPT forward pass: token IDs → logits over vocabulary.

        Math:
            x      = tok_embeddings(idx)              # (B, T, n_embd)  lookup
            for each TransformerBlock layer:
                x  = layer(x, mask, cache)            # self-attention + FFN
            logits = output( norm(x) )                # (B, T, vocab_size)

        Causal mask (generated once and cached per sequence length T):
            mask[i, j] = 0     if j <= i  (token j can attend to token i)
            mask[i, j] = -inf  if j > i   (future positions are masked out)

        The mask is additive (added to the QK^T dot-product before softmax) so
        that -inf positions become 0 after softmax.

        Weight tying means the final linear layer reuses the embedding matrix:
            logits = x_norm · tok_embeddings.weight^T

        Args:
            idx:   Integer token indices, shape (B, T).
            mask:  Optional explicit additive causal mask, shape (T, T).
                   If None and no cache, the causal mask is generated/cached.
            cache: Optional KV cache list for autoregressive inference.

        Returns:
            Logits tensor of shape (B, T, vocab_size).

        Example:
            idx    = mx.array([[1, 2, 3, 4]])        # batch=1, seq=4
            logits = model(idx)                       # shape (1, 4, 100277)
            next_token = mx.argmax(logits[0, -1])    # greedy decode next token
        """
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
