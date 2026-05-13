import math
import pickle
import logging
from typing import Optional
from dataclasses import asdict
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mut
from config import TrainingConfig

logger = logging.getLogger(__name__)

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
    
    Ensures that only the 'keep_checkpoints' most recent states are kept on disk
    to prevent storage exhaustion.
    """
    def __init__(self, config: TrainingConfig):
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
