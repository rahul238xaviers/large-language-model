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
    # Model Architecture (Optimised 350M Parameters)
    n_layer: int = 24
    n_embd: int = 1024
    n_head: int = 16
    n_kv_head: int = 8
    head_dim: int = 64 # n_embd // n_head
    block_size: int = 1024
    vocab_size: int = 100277 # cl100k_base
    
    # Training Hyperparameters (Highly Efficient for M3 Ultra + 1024 Context)
    micro_batch_size: int = 32 # Safe and high GPU saturation for 350M + 1024 context
    gradient_accumulation_steps: int = 4 # Keeps effective batch size at 128

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
    num_worker_threads: int = 4  # 4 workers saturate the GPU pipeline; 8 wastes ~400 MB with no throughput gain
    
    # Hardware & Precision
    dtype: str = "bfloat16" # Critical for M3 AMX performance
    grad_checkpoint: bool = True  # Gradient checkpointing: ~70% less activation memory, ~33% slower compute
    profile_methods: bool = False
    eager_mode: bool = False
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
        self.n_layer = int(os.getenv("TRAIN_N_LAYER", self.n_layer))
        self.n_embd = int(os.getenv("TRAIN_N_EMBD", self.n_embd))
        self.n_head = int(os.getenv("TRAIN_N_HEAD", self.n_head))
        self.n_kv_head = int(os.getenv("TRAIN_N_KV_HEAD", self.n_kv_head))
        self.head_dim = int(os.getenv("TRAIN_HEAD_DIM", self.head_dim))
        self.block_size = int(os.getenv("TRAIN_BLOCK_SIZE", self.block_size))

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
        # Gradient checkpointing: on by default, disable with TRAIN_GRAD_CHECKPOINT=0
        gc_env = os.getenv("TRAIN_GRAD_CHECKPOINT", "1")
        self.grad_checkpoint = gc_env.lower() not in {"0", "false", "no", "off"}
        profile_default = "1" if self.max_iters <= 5 else "0"
        self.profile_methods = os.getenv("TRAIN_PROFILE_METHODS", profile_default).lower() in {"1", "true", "yes", "on"}
        
        # Eager mode: off by default, enable with TRAIN_EAGER_MODE=1
        eager_env = os.getenv("TRAIN_EAGER_MODE", "0")
        self.eager_mode = eager_env.lower() in {"1", "true", "yes", "on"}

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
