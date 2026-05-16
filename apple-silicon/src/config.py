import os
from pathlib import Path
from dataclasses import dataclass

@dataclass
class HardwareConfig:
    total_memory_gb: int = 512
    gpu_cores: int = 80
    gpu_freq_hz: float = 1.4e9
    flops_per_core_per_cycle: int = 512
    
    @property
    def theoretical_tflops(self) -> float:
        """Calculates theoretical peak TFLOPS of the M3 Ultra GPU."""
        return (self.gpu_cores * self.gpu_freq_hz * self.flops_per_core_per_cycle) / 1e12

@dataclass
class TrainingConfig:
    # Model Architecture (1.6B Parameters)
    n_layer: int = 24
    n_embd: int = 2048
    n_head: int = 16
    n_kv_head: int = 8
    head_dim: int = 128 # n_embd // n_head
    block_size: int = 2048
    vocab_size: int = 100277 # cl100k_base
    
    # Training Hyperparameters (Aggressive Scaling for M3 Ultra)
    micro_batch_size: int = 4 # Safer starting point for 1.6B + 2048 context
    gradient_accumulation_steps: int = 32 # Keeps effective batch high without OOM spikes
    effective_batch_size: int = 128 # 4 * 32
    
    # Optimizer
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    warmup_iters: int = 500
    max_iters: int = 100000
    weight_decay: float = 0.1
    grad_clip_norm: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    
    # Data Pipeline
    prefetch_size: int = 50000 # Deprecated (kept for backward compatibility)
    token_chunk_size: int = 4096 # Small bounded chunks to cap queue memory
    token_queue_max_chunks: int = 256 # Queue depth in chunks (not documents)
    num_prefetch_batches: int = 64
    num_worker_threads: int = 8 # Lower process pressure; tune upward after stability
    
    # Hardware & Precision
    dtype: str = "bfloat16" # Critical for M3 AMX performance
    profile_methods: bool = False
    save_interval: int = 500
    keep_checkpoints: int = 3
    checkpoint_dir: Path = Path("checkpoints")
    
    @property
    def mx_dtype(self):
        import mlx.core as mx
        return getattr(mx, self.dtype)

    @property
    def total_tokens_per_iter(self) -> int:
        """Total tokens per optimization step (1,048,576 tokens)."""
        return self.effective_batch_size * self.block_size
