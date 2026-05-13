import time
import logging
import math
import os
import json
import csv
from functools import partial
from typing import Any
from datetime import datetime
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mut

from config import TrainingConfig, HardwareConfig
from model import GPTModel
from data import ParallelTokenStream, AsyncBatchPrefetcher

def setup_run_dir(config):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"runs/run_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()}
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)
    return run_dir

class MetricsLogger:
    def __init__(self, run_dir):
        self.metrics_path = run_dir / "metrics.csv"
        self.headers = ["step", "train_loss", "tokens_per_sec", "learning_rate", "vram_usage_gb", "mfu_pct"]
        with open(self.metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
    def log(self, metrics_dict):
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([metrics_dict.get(h, "") for h in self.headers])

def get_lr(it, config):
    if it < config.warmup_iters: return config.learning_rate * it / config.warmup_iters
    if it > config.max_iters: return config.min_lr
    decay_ratio = it / config.max_iters
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

def make_step(model, optimizer, config):
    """
    Build a compiled micro-step. We keep gradient accumulation outside compile
    and force evaluation each micro-batch to avoid graph growth.
    """
    def loss_fn(model, x, y):
        logits = model(x)
        return mx.mean(nn.losses.cross_entropy(logits, y))

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)
    step_state = [model.state]

    @partial(mx.compile, inputs=step_state)
    def micro_step(x, y):
        return loss_and_grad_fn(model, x, y)

    return micro_step, step_state

def train():
    config = TrainingConfig()
    hw_config = HardwareConfig()
    run_dir = setup_run_dir(config)
    metrics_logger = MetricsLogger(run_dir)
    
    log_file = run_dir / "train.log"
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.FileHandler(log_file), logging.StreamHandler()])
    logger = logging.getLogger(__name__)

    logger.info(f"Production Run: {run_dir.name} | Architecture: 1.6B Pure Closure")
    model = GPTModel(config)
    
    # Cast to dtype and tie weights
    model.update(mut.tree_map(lambda p: p.astype(config.mx_dtype), model.parameters()))
    model.tie_weights()
    mx.eval(model.parameters())
    
    param_count = sum(getattr(v, "size", 0) for _, v in mut.tree_flatten(model.parameters()) if not isinstance(v, str))
    optimizer = optim.AdamW(learning_rate=config.learning_rate, betas=[config.beta1, config.beta2])
    
    # Build compiled micro-step (accumulation is done outside compile)
    micro_step, step_state = make_step(model, optimizer, config)

    # Data Pipeline
    token_stream = ParallelTokenStream(config)
    prefetcher = AsyncBatchPrefetcher(config, token_stream)
    
    iteration = 0
    throughput_history = []

    try:
        while iteration < config.max_iters:
            iter_start = time.time()
            
            # 1. Fetch full iteration (list of micro-batches)
            batch_x, batch_y = prefetcher.get_full_iteration()
            
            # 2. Execute micro-steps with explicit per-step evaluation
            total_loss = mx.array(0.0)
            accumulated_grads = mut.tree_map(lambda p: mx.zeros_like(p), model.parameters())

            for x_mb, y_mb in zip(batch_x, batch_y):
                loss_mb, grads_mb = micro_step(x_mb, y_mb)
                total_loss = total_loss + loss_mb
                accumulated_grads = mut.tree_map(lambda ag, g: ag + g, accumulated_grads, grads_mb)
                # Collapse lazy graph per micro-batch to cap memory.
                mx.eval(accumulated_grads)

            scale = 1.0 / len(batch_x)
            loss_arr = total_loss * scale
            accumulated_grads = mut.tree_map(lambda g: g * scale, accumulated_grads)

            total_sq_norm = mx.array(0.0)
            for _, v in mut.tree_flatten(accumulated_grads):
                arr: Any = v
                if isinstance(arr, str):
                    continue
                total_sq_norm = total_sq_norm + mx.sum(arr * arr)
            norm_arr = mx.sqrt(total_sq_norm)

            clip_scale = mx.where(
                norm_arr > config.grad_clip_norm,
                config.grad_clip_norm / (norm_arr + 1e-6),
                1.0,
            )
            accumulated_grads = mut.tree_map(lambda g: g * clip_scale, accumulated_grads)

            optimizer.update(model, accumulated_grads)

            # 3. Synchronous evaluation of updated state and scalars
            mx.eval(step_state, optimizer.state, loss_arr, norm_arr)
            
            # 4. Instrumentation
            dt = time.time() - iter_start
            tokens_per_sec = config.total_tokens_per_iter / dt
            throughput_history.append(tokens_per_sec)
            
            step_loss = loss_arr.item()
            step_norm = norm_arr.item()
            # Modernized memory API
            active_mem = mx.get_active_memory() / (1024**3)
            lr = get_lr(iteration, config)
            optimizer.learning_rate = lr
            
            mfu = ((6 * param_count * config.total_tokens_per_iter) / dt) / (hw_config.theoretical_tflops * 1e12)

            if iteration % 10 == 0:
                metrics_logger.log({
                    "step": iteration, "train_loss": step_loss, 
                    "tokens_per_sec": tokens_per_sec, "learning_rate": lr, 
                    "vram_usage_gb": active_mem, "mfu_pct": mfu * 100
                })
                logger.info(f"Iter {iteration:4d} | Loss {step_loss:.4f} | tok/s {tokens_per_sec:6.0f} | MFU {mfu*100:4.1f}% | Mem {active_mem:.1f}GB")

            iteration += 1
            if iteration % 20 == 0:
                mx.clear_cache()
                
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception as e:
        logger.error(f"Training crashed: {e}")
    finally:
        prefetcher.stop()
        token_stream.stop()

if __name__ == "__main__":
    train()
