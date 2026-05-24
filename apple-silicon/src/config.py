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
    
    def __post_init__(self):
        """
        Apply environment-variable overrides after dataclass initialisation.

        This keeps runtime tuning external to the source code, so you can
        change batch sizes, iteration count, worker limits, and dtype without
        editing `src/config.py`.
        """
        self.micro_batch_size = int(os.getenv("TRAIN_MICRO_BATCH_SIZE", self.micro_batch_size))
        self.gradient_accumulation_steps = int(os.getenv("TRAIN_GRAD_ACC_STEPS", self.gradient_accumulation_steps))
        self.num_worker_threads = int(os.getenv("TRAIN_NUM_WORKERS", self.num_worker_threads))
        self.token_queue_max_chunks = int(os.getenv("TRAIN_TOKEN_QUEUE_MAX_CHUNKS", self.token_queue_max_chunks))
        self.num_prefetch_batches = int(os.getenv("TRAIN_NUM_PREFETCH_BATCHES", self.num_prefetch_batches))
        self.max_iters = int(os.getenv("TRAIN_MAX_ITERS", self.max_iters))
        self.warmup_iters = int(os.getenv("TRAIN_WARMUP_ITERS", self.warmup_iters))
        self.learning_rate = float(os.getenv("TRAIN_LEARNING_RATE", self.learning_rate))
        self.min_lr = float(os.getenv("TRAIN_MIN_LR", self.min_lr))
        self.save_interval = int(os.getenv("TRAIN_SAVE_INTERVAL", self.save_interval))
        self.keep_checkpoints = int(os.getenv("TRAIN_KEEP_CHECKPOINTS", self.keep_checkpoints))
        self.dtype = os.getenv("TRAIN_DTYPE", self.dtype)
        profile_default = "1" if self.max_iters <= 5 else "0"
        self.profile_methods = os.getenv("TRAIN_PROFILE_METHODS", profile_default).lower() in {"1", "true", "yes", "on"}

    @property
    def mx_dtype(self):
        import mlx.core as mx
        return getattr(mx, self.dtype)

    @property
    def effective_batch_size(self) -> int:
        """
        Number of sequences in one optimizer update.

        This is derived from the current micro-batch size and gradient
        accumulation steps, allowing env overrides to change it dynamically.
        """
        return self.micro_batch_size * self.gradient_accumulation_steps

    @property
    def total_tokens_per_iter(self) -> int:
        """Total tokens per optimization step."""
        return self.effective_batch_size * self.block_size
