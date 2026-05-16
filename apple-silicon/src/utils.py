import math
import pickle
import logging
import time
from contextlib import contextmanager
from typing import Optional
from dataclasses import asdict
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mut
from config import TrainingConfig

logger = logging.getLogger(__name__)


@contextmanager
def timed_log(label: str, enabled: bool = True, log: Optional[logging.Logger] = None):
    """
    Context manager that measures wall-clock time for a code block and emits
    a structured log line when `enabled` is True.

    When `enabled` is False the block executes normally with zero overhead
    (no timing, no logging).

    Implementation note:
        Uses `time.perf_counter()` (monotonic, sub-microsecond resolution on
        macOS) rather than `time.time()` to avoid being affected by NTP adjustments.

    Args:
        label:   Human-readable name for the timed region.
        enabled: If False the context manager is a no-op.  Defaults to True.
        log:     Logger to emit to.  Falls back to the "train" logger if None.

    Example:
        with timed_log("model_forward", enabled=config.profile_methods, log=logger):
            logits = model(x)
        # Emits: "model_forward | 0.342s"

        # Disabled (zero overhead):
        with timed_log("data_fetch", enabled=False):
            batch = get_batch()
    """
    if not enabled:
        yield
        return
    active_logger = log or logging.getLogger("train")
    start = time.perf_counter()
    try:
        yield
    finally:
        active_logger.info("%s | %.3fs", label, time.perf_counter() - start)

def get_lr(iteration: int, config: TrainingConfig) -> float:
    """
    Computes the current learning rate using a Linear Warmup and Cosine Decay schedule.
    
    Math:
        1. Warmup (if iter < warmup_iters):
           lr = lr_max * (iter / warmup_iters)
           
        2. Cosine Decay (if iter >= warmup_iters):
           decay_ratio = (iter - warmup_iters) / (max_iters - warmup_iters)
           coeff = 0.5 * (1.0 + cos(pi * decay_ratio))
           lr = lr_min + coeff * (lr_max - lr_min)
           
    Example:
        Iter 0: lr = 0
        Iter 500 (warmup ends): lr = 3e-4 (lr_max)
        Iter 50,000 (halfway): cos(pi/2) = 0 -> coeff = 0.5 -> lr = halfway between max and min.
    """
    if iteration < config.warmup_iters:
        return config.learning_rate * iteration / config.warmup_iters
    decay_ratio = (iteration - config.warmup_iters) / (config.max_iters - config.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

class CheckpointManager:
    """
    Manages saving and pruning of MLX model checkpoints and optimizer states.

    Ensures that only the `keep_checkpoints` most recent states are kept on
    disk to prevent storage exhaustion during long training runs.

    Checkpoint format:
        Model weights → `.safetensors`  (flat key-value, safe to load cross-framework)
        Metadata      → `.meta.pkl`     (iteration, step, val_loss, config dict)

    A `best_model.safetensors` is maintained separately for the weights
    corresponding to the lowest validation loss seen so far.
    """
    def __init__(self, config: TrainingConfig):
        """
        Initialise the checkpoint manager and create the checkpoint directory.

        Args:
            config: TrainingConfig carrying `checkpoint_dir` and
                    `keep_checkpoints` fields.

        Example:
            mgr = CheckpointManager(config)
            # config.checkpoint_dir/ is created if it did not exist
        """
        self.config = config
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints = []
        self.best_val_loss = float('inf')

    def save(self, model: nn.Module, optimizer: optim.Optimizer, iteration: int, 
             step: int, val_loss: Optional[float] = None):
        """
        Saves the current model weights and optimizer state to disk using `.safetensors`.
        
        Args:
            model: The GPT model instance.
            optimizer: The MLX optimizer.
            iteration: Global training iteration.
            step: Effective training step.
            val_loss: Optional validation loss for tracking best model.
        """
        ckpt_path = self.config.checkpoint_dir / f"model_iter_{iteration:06d}_step_{step:06d}.safetensors"
        flat_weights = dict(mut.tree_flatten(model.parameters()))
        mx.save_safetensors(str(ckpt_path), flat_weights)
        
        metadata = {"iteration": iteration, "step": step, "val_loss": val_loss or float('inf'),
                   "config": asdict(self.config)}
        with open(ckpt_path.with_suffix('.meta.pkl'), 'wb') as f:
            pickle.dump(metadata, f)
        
        if val_loss and val_loss < self.best_val_loss:
            self.best_val_loss = val_loss
            mx.save_safetensors(str(self.config.checkpoint_dir / "best_model.safetensors"), flat_weights)
            logger.info(f"New best model! val_loss={val_loss:.4f}")
        
        self.checkpoints.append(ckpt_path)
        if len(self.checkpoints) > self.config.keep_checkpoints:
            old = self.checkpoints.pop(0)
            old.unlink(missing_ok=True)
            old.with_suffix('.meta.pkl').unlink(missing_ok=True)
        
        logger.info(f"Saved checkpoint: {ckpt_path.name}")
