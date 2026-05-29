import time
import logging
import math
import os
import json
import csv
import resource
import sys
from functools import partial
from typing import Any
from datetime import datetime
from pathlib import Path
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import mlx.utils as mut
from utils import timed_log

from config import TrainingConfig, HardwareConfig
from model import GPTModel
from data import ParallelTokenStream, AsyncBatchPrefetcher


def build_run_logger(log_file: Path) -> logging.Logger:
    """
    Create an isolated "train" logger that writes to both a file and stdout.

    The logger is detached from the root logger (`propagate = False`) so that
    third-party libraries (e.g. tiktoken, pyarrow) that attach handlers to the
    root logger do not pollute the training log file.

    Format: `<timestamp> [<level>] <message>`

    Args:
        log_file: Path to the `.log` file; opened in append mode so that
                  multiple training restarts accumulate in the same file.

    Returns:
        Configured `logging.Logger` named "train".

    Example:
        logger = build_run_logger(Path("runs/run_20260514/train.log"))
        logger.info("Training started")   # → file + stdout
    """
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    stream_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)
    return logger

def setup_run_dir(config: TrainingConfig) -> Path:
    """
    Create a timestamped run directory and save the resolved config as JSON.

    Directory layout:
        runs/run_YYYYMMDD_HHMMSS/
            config.json   — all TrainingConfig fields + computed properties
            train.log     — written later by build_run_logger
            metrics.csv   — written later by MetricsLogger

    The config snapshot includes computed properties (`effective_batch_size`,
    `total_tokens_per_iter`) so the file is self-contained and can be used to
    reproduce or audit the run without the source code.

    Args:
        config: TrainingConfig instance (after env-var overrides are applied).

    Returns:
        `pathlib.Path` pointing to the newly created run directory.

    Example:
        run_dir = setup_run_dir(config)
        # run_dir == Path("runs/run_20260514_083022")
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(f"runs/run_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    config_dict = {k: str(v) if isinstance(v, Path) else v for k, v in config.__dict__.items()}
    # Include computed properties so the saved params file is self-contained
    config_dict["effective_batch_size"] = config.effective_batch_size
    config_dict["total_tokens_per_iter"] = config.total_tokens_per_iter
    with open(run_dir / "config.json", "w") as f:
        json.dump(config_dict, f, indent=4)
    return run_dir

class MetricsLogger:
    def __init__(self, run_dir: Path):
        """
        Initialise the CSV metrics logger and write the header row.

        Columns written per optimizer step:
            step            — training iteration index (0-based)
            train_loss      — mean cross-entropy loss over the effective batch
            tokens_per_sec  — total tokens processed per wall-clock second
            learning_rate   — current LR after warmup / cosine decay
            vram_usage_gb   — MLX active memory in GiB
            mfu_pct         — Model FLOP Utilisation as a percentage

        Args:
            run_dir: `pathlib.Path` to the run directory (must exist).

        Example:
            ml = MetricsLogger(run_dir)
            # Creates run_dir/metrics.csv with header row
        """
        self.metrics_path = run_dir / "metrics.csv"
        self.headers = ["step", "train_loss", "tokens_per_sec", "learning_rate", "vram_usage_gb", "mfu_pct"]
        with open(self.metrics_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)

    def log(self, metrics_dict: dict[str, float]) -> None:
        """
        Append one row of scalar metrics to the CSV file.

        Missing keys are written as empty strings so the column count stays
        consistent and the file remains valid CSV.

        Args:
            metrics_dict: Dict mapping header names to scalar values.

        Example:
            ml.log({"step": 5, "train_loss": 3.21, "tokens_per_sec": 512.0,
                    "learning_rate": 2.8e-4, "vram_usage_gb": 22.6, "mfu_pct": 8.3})
        """
        with open(self.metrics_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([metrics_dict.get(h, "") for h in self.headers])

def get_lr(it: int, config: TrainingConfig) -> float:
    """
    Cosine learning-rate schedule with linear warmup.

    Math:
        Phase 1 — linear warmup (it < warmup_iters):
            lr = lr_max × (it / warmup_iters)

        Phase 2 — cosine decay (warmup_iters ≤ it ≤ max_iters):
            decay_ratio = it / max_iters
            coeff       = 0.5 × (1 + cos(π × decay_ratio))
            lr          = lr_min + coeff × (lr_max - lr_min)

        Phase 3 — floor (it > max_iters):
            lr = lr_min

    The cosine term sweeps from 1 → 0 as `decay_ratio` goes 0 → 1, so `lr`
    smoothly decays from `lr_max` to `lr_min` without abrupt drops.

    Args:
        it:     Current iteration index (0-based).
        config: TrainingConfig with `learning_rate`, `min_lr`, `warmup_iters`,
                and `max_iters`.

    Returns:
        Scalar learning rate for iteration `it`.

    Example:
        # warmup_iters=1, max_iters=5, lr_max=3e-4, lr_min=3e-5
        get_lr(0, config)  # 0.0               (start of warmup)
        get_lr(1, config)  # 3e-4              (peak, warmup done)
        get_lr(3, config)  # ~1.65e-4          (midway through cosine)
        get_lr(5, config)  # 3e-5              (floor)
    """
    if it < config.warmup_iters: return config.learning_rate * it / config.warmup_iters
    if it > config.max_iters: return config.min_lr
    decay_ratio = it / config.max_iters
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return config.min_lr + coeff * (config.learning_rate - config.min_lr)

def make_step(model: GPTModel, optimizer: optim.Optimizer, config: TrainingConfig):
    """
    Build and return a compiled MLX micro-step function.

    The micro-step computes the cross-entropy loss and its gradients for a
    single micro-batch `(x, y)`.  Gradient accumulation is intentionally kept
    outside the compiled region so the computation graph stays bounded in
    memory — each call produces a fixed-size graph regardless of how many
    micro-batches are accumulated.

    Math (per call):
        logits  = model(x)                             # (B, T, vocab_size)
        loss    = mean( CrossEntropy(logits, y) )      # scalar
        grads   = ∇_θ loss                            # same tree shape as model.parameters()

    `mx.compile` traces the graph once (on the first call), caches the compiled
    Metal kernel schedule, and re-uses it for all subsequent calls with the same
    shapes — eliminating Python-side graph construction overhead.

    `inputs=model.state` / `outputs=model.state` tells the compiler that the
    model's parameter tensors may be read (and not mutated) inside the step;
    this is required for correct gradient capture through the closure.

    Args:
        model:     Initialised GPTModel instance.
        optimizer: MLX optimizer (used only to register state; not called here).
        config:    TrainingConfig (reserved for future conditional compilation).

    Returns:
        `micro_step(x, y) → (loss_scalar, grad_tree)` — an `mx.compile`-d
        callable.

    Example:
        micro_step = make_step(model, optimizer, config)
        loss, grads = micro_step(x_mb, y_mb)   # lazy; call mx.eval to materialise
    """
    def loss_fn(model, x, y):
        logits = model(x)
        return mx.mean(nn.losses.cross_entropy(logits, y))

    loss_and_grad_fn = nn.value_and_grad(model, loss_fn)

    @partial(mx.compile, inputs=model.state, outputs=model.state)
    def micro_step(x, y):
        return loss_and_grad_fn(model, x, y)

    return micro_step


def _dtype_nbytes(dtype_name: str) -> int:
    """
    Return the number of bytes occupied by one scalar element of `dtype_name`.

    Used to estimate memory footprints without instantiating actual tensors.

    Supported dtype strings (case-insensitive, substring match):
        "bfloat16" | "float16" | "half"  → 2 bytes
        "float32"  | "int32"             → 4 bytes
        "float64"  | "int64"             → 8 bytes
        "int16"                          → 2 bytes
        "int8"     | "uint8" | "bool"    → 1 byte
        anything else                    → 4 bytes (conservative default)

    Args:
        dtype_name: String representation of an MLX or NumPy dtype.

    Returns:
        Byte width of one element.

    Example:
        _dtype_nbytes("bfloat16")  # → 2
        _dtype_nbytes("float32")   # → 4
        _dtype_nbytes(str(mx.bfloat16))  # → 2
    """
    name = dtype_name.lower()
    if "bfloat16" in name or "float16" in name or "half" in name:
        return 2
    if "float32" in name or "int32" in name:
        return 4
    if "float64" in name or "int64" in name:
        return 8
    if "int16" in name:
        return 2
    if "int8" in name or "uint8" in name or "bool" in name:
        return 1
    return 4


def _tree_array_gb(tree: Any) -> float:
    """
    Compute the total memory footprint (in GiB) of all leaf arrays in a
    nested parameter / optimizer-state tree.

    Shared tensors (e.g. tied embedding / output weights that appear at two
    paths in the tree) are counted only once via an `id()` set, so the
    estimate does not double-count weight tying.

    Math:
        bytes = sum over unique leaves: leaf.size × dtype_bytes(leaf.dtype)
        GiB   = bytes / 1024^3

    Args:
        tree: Any nested dict/list/tuple of MLX arrays (e.g. `model.parameters()`).

    Returns:
        Total memory in gibibytes (GiB).

    Example:
        gb = _tree_array_gb(model.parameters())
        # For 1.518B bf16 params: 1.518e9 × 2 / 1024^3 ≈ 2.83 GiB
    """
    total_bytes = 0
    seen_ids: set[int] = set()
    for _, value in mut.tree_flatten(tree):
        if isinstance(value, str):
            continue
        if not hasattr(value, "size"):
            continue
        # Avoid double-counting shared/tied tensors that can appear multiple
        # times in parameter trees (for example tied embeddings).
        value_id = id(value)
        if value_id in seen_ids:
            continue
        seen_ids.add(value_id)
        dtype_bytes = _dtype_nbytes(str(getattr(value, "dtype", "float32")))
        total_bytes += int(getattr(value, "size", 0)) * dtype_bytes
    return total_bytes / (1024 ** 3)


def _tree_dtype_summary(tree: Any, max_items: int = 6) -> str:
    """
    Collect the set of unique dtype strings present in a parameter tree and
    return them as a comma-separated summary string.

    Useful for verifying that a cast operation (e.g. bfloat16 conversion)
    succeeded uniformly and that no float32 leaves remain after the cast-back.

    Args:
        tree:      Nested MLX parameter tree.
        max_items: Maximum number of dtype names to include before truncating.

    Returns:
        Comma-separated sorted dtype names, or "unknown" if no arrays found.
        If more than `max_items` distinct dtypes are present, the remainder is
        summarised as "... (+N)".

    Example:
        # After successful bfloat16 cast:
        _tree_dtype_summary(model.parameters())  # → "mlx.core.bfloat16"

        # Mixed (fp32 contamination detected):
        _tree_dtype_summary(model.parameters())  # → "mlx.core.bfloat16, mlx.core.float32"
    """
    dtypes: set[str] = set()
    for _, value in mut.tree_flatten(tree):
        if isinstance(value, str):
            continue
        if hasattr(value, "dtype"):
            dtypes.add(str(value.dtype))
    if not dtypes:
        return "unknown"
    ordered = sorted(dtypes)
    if len(ordered) <= max_items:
        return ", ".join(ordered)
    return ", ".join(ordered[:max_items]) + f", ... (+{len(ordered) - max_items})"


def _estimate_expected_memory_gb(config: TrainingConfig, param_count: int) -> dict[str, float]:
    """
    Estimate how much GPU memory each category of tensor will consume.

    This provides a pre-run sanity check: if the sum exceeds available
    unified memory, an OOM crash is likely before the first optimizer step.

    Categories and assumptions:
        params      — model weights stored in `config.dtype` (e.g. bf16 = 2 B/param)
        grads       — autograd gradients promoted to fp32 by MLX (4 B/param)
        opt_state   — Adam first moment m + second moment v, both fp32 (8 B/param)
        attn_act    — attention score matrices retained for backprop:
                        mbs × n_head × T × T × dtype_bytes × n_layer
        other_act   — residual stream activations:
                        mbs × T × n_embd × dtype_bytes × n_layer

    Math example (mbs=16, T=2048, n_embd=2048, n_head=16, n_layer=24, bf16):
        attn_act = 16 × 16 × 2048^2 × 2 × 24 / 1024^3  ≈ 48 GiB
        (This is why Flash Attention is so valuable — it avoids materialising
        the full T×T matrix in HBM.)

    Args:
        config:      TrainingConfig.
        param_count: Total number of scalar parameters (int).

    Returns:
        Dict with keys: "params", "grads", "opt_state", "attn_act", "other_act".
        Values are GiB floats.
    """
    act_dtype_bytes = _dtype_nbytes(config.dtype)
    # After the fp32-cast-back fix, model weights stay in config.dtype (bfloat16 = 2 bytes).
    expected_params_gb = (param_count * act_dtype_bytes) / (1024 ** 3)
    expected_grads_gb = (param_count * 4) / (1024 ** 3)  # fp32 gradients from autograd
    expected_opt_state_gb = (param_count * 8) / (1024 ** 3)  # Adam m and v in fp32
    expected_attn_act_gb = (
        config.micro_batch_size
        * config.n_head
        * config.block_size
        * config.block_size
        * act_dtype_bytes
        * config.n_layer
    ) / (1024 ** 3)
    expected_other_act_gb = (
        config.micro_batch_size
        * config.block_size
        * config.n_embd
        * act_dtype_bytes
        * config.n_layer
    ) / (1024 ** 3)
    return {
        "params": expected_params_gb,
        "grads": expected_grads_gb,
        "opt_state": expected_opt_state_gb,
        "attn_act": expected_attn_act_gb,
        "other_act": expected_other_act_gb,
    }


def _mx_memory_gb(name: str) -> float | None:
    """
    Safely read one of the MLX memory-reporting functions by name.

    MLX's memory API evolves between releases; some functions (`get_cache_memory`,
    `get_peak_memory`) are not present in all versions.  This helper centralises
    the version check so callers don't need try/except everywhere.

    Args:
        name: MLX function name, e.g. "get_active_memory", "get_cache_memory",
              "get_peak_memory".

    Returns:
        Memory in GiB, or None if the function does not exist or raises.

    Example:
        active_gb = _mx_memory_gb("get_active_memory")  # e.g. 22.6
        cache_gb  = _mx_memory_gb("get_cache_memory")   # None if unavailable
    """
    if not hasattr(mx, name):
        return None
    try:
        return getattr(mx, name)() / (1024 ** 3)
    except Exception:
        return None


def _active_memory_gb() -> float:
    """
    Return the current MLX "active" memory (live tensors not in the cache)
    in GiB, with a fallback for older MLX builds.

    "Active" memory = tensors that are referenced by at least one live Python
    object.  This excludes the buffer cache (freed tensors kept warm for
    reuse) and is the most meaningful number for diagnosing memory pressure.

    Returns:
        Active memory in GiB.

    Example:
        mem = _active_memory_gb()   # e.g. 22.6 during training at mbs=16
    """
    active = _mx_memory_gb("get_active_memory")
    if active is not None:
        return active
    return mx.get_active_memory() / (1024 ** 3)


def _process_rss_gb() -> float:
    """
    Return the current process Resident Set Size (RSS) in GiB.

    RSS measures how much physical RAM the process is currently occupying,
    including MLX GPU buffers (which live in Apple’s unified memory), Python
    heap, tokenizer subprocesses mapped pages, and OS overhead.

    Platform note:
        `resource.getrusage(RUSAGE_SELF).ru_maxrss` units vary by OS:
          • macOS: kilobytes  (typical value ≈10^8)
          • Linux: bytes      (typical value ≈10^11)
        A magnitude heuristic (> 1e9 → bytes; otherwise KB) handles both.

    Returns:
        RSS in GiB (float).

    Example:
        rss = _process_rss_gb()   # e.g. 103.7 GiB when training at mbs=16
        # Much larger than MLX active memory because it includes CPU-side
        # Python objects, tokenizer worker mapped pages, and OS page cache.
    """
    ru = resource.getrusage(resource.RUSAGE_SELF)
    # ru_maxrss units differ by platform/build: often KB, sometimes bytes.
    # Use a magnitude heuristic to avoid under-reporting on macOS.
    rss_raw = float(ru.ru_maxrss)
    if rss_raw > 1e9:
        # Likely already bytes.
        return rss_raw / (1024 ** 3)
    # Likely kilobytes.
    return (rss_raw * 1024.0) / (1024 ** 3)


def _reset_peak_memory() -> None:
    """
    Reset MLX's internal peak-memory watermark to zero.

    After calling this, `mx.get_peak_memory()` will report the maximum
    memory allocated since this reset — useful for measuring the peak
    footprint of a specific code region (e.g. a single training iteration)
    without contamination from the initialization phase.

    No-ops gracefully if `mx.reset_peak_memory` does not exist in the
    installed MLX version.
    """
    if not hasattr(mx, "reset_peak_memory"):
        return
    try:
        mx.reset_peak_memory()
    except Exception:
        pass


def _log_throughput_projection(config: TrainingConfig, hw_config: HardwareConfig, param_count: int, logger: logging.Logger) -> None:
    """
    Log a projection table of tokens/s and time-per-iter for candidate
    (micro_batch_size, gradient_accumulation_steps) combos.

    Assumes per-micro-batch GPU time scales linearly with mbs (conservative).
    Actual may be better if larger mbs improves GPU occupancy.
    """
    # Baseline: from empirical run — ~61s per micro-batch at mbs=16 (stable iters 1+)
    # We use the current config to derive a per-token-time baseline, so the
    # projection automatically updates if the user has changed mbs.
    MB_SEC_PER_TOKEN_BASE = 61.0 / (config.micro_batch_size * config.block_size)  # s/token per micro-batch

    candidates = [
        (8,  8),
        (8,  16),
        (16, 4),   # fewer acc steps, faster iters
        (16, 8),   # current
        (16, 16),
        (32, 4),
        (32, 8),
    ]

    logger.info("ThroughputProjection | (mbs × acc) | tokens/iter | sec/iter | tok/s  | mfu%%  | note")
    logger.info("ThroughputProjection | %-12s | %-11s | %-8s | %-6s | %-5s | %s",
                "mbs × acc", "tokens/iter", "sec/iter", "tok/s", "mfu%", "note")
    for mbs, acc in candidates:
        tokens_per_iter = mbs * acc * config.block_size
        sec_per_mb = MB_SEC_PER_TOKEN_BASE * mbs * config.block_size
        sec_per_iter = sec_per_mb * acc
        toks = tokens_per_iter / sec_per_iter
        mfu = ((6 * param_count * tokens_per_iter) / sec_per_iter) / (hw_config.theoretical_tflops * 1e12) * 100
        note = "← current" if (mbs == config.micro_batch_size and acc == config.gradient_accumulation_steps) else ""
        logger.info("ThroughputProjection | mbs=%-2d acc=%-2d | %11d | %8.1f | %6.0f | %5.1f | %s",
                    mbs, acc, tokens_per_iter, sec_per_iter, toks, mfu, note)


def _check_model_dtype(model: Any, expected_dtype_str: str, iteration: int, logger: logging.Logger) -> None:
    """Warn if any model parameter has been promoted away from the expected dtype."""
    actual = _tree_dtype_summary(model.parameters())
    if expected_dtype_str not in actual or "float32" in actual.replace(expected_dtype_str, ""):
        logger.warning("DtypeCheck FAIL | Iter %d | expected %s | got [%s]", iteration, expected_dtype_str, actual)
    else:
        logger.info("DtypeCheck OK   | Iter %d | model params are [%s]", iteration, actual)

def train() -> None:
    """
    Main training entry point for the 1.518B GPT model on Apple Silicon.

    High-level loop structure per iteration:
        1. Fetch a full iteration (all micro-batches) from the async prefetcher.
        2. For each micro-batch:
               loss, grads = micro_step(x, y)    # compiled forward + backward
               mx.eval(loss); mx.eval(grads)      # materialise (split when profiling)
               accumulate grads
        3. Scale + clip accumulated gradients (global L2 norm clipping).
        4. Optimizer step (AdamW) + cast model weights back to bfloat16.
        5. Synchronise all lazy tensors (`mx.eval`).
        6. Log metrics; optionally save checkpoint.

    Gradient clipping math:
        norm  = sqrt( sum_i( ||g_i||^2 ) )       # global L2 norm across all params
        scale = min(1, grad_clip_norm / norm)     # clip factor (1.0 if already small)
        g_i   ← g_i × scale                       # rescale in place

    Model FLOP Utilisation (MFU):
        MFU = (6 × N × T_tokens) / (t_iter × TFLOPS_hw)
        where N = param count, T_tokens = tokens per iter, t_iter = wall-clock iter time.
        Factor 6 = 2 (FWD) + 4 (BWD) FLOPs per multiply-add.

    AdamW bfloat16 cast-back:
        AdamW promotes model parameters to fp32 during the update step.
        After `optimizer.update()`, weights are explicitly cast back to bf16
        to halve parameter memory and restore fast bfloat16 matmuls.
        Adam m/v states remain fp32 (correct for numerical stability).

    Example (diagnostic 5-iter run):
        python src/train.py
        # Iter 0: ~559s (compile + 8 micro-batches)
        # Iter 1–4: ~490–504s each
        # Bottleneck: micro-batch compute (~61s each at mbs=16, block=2048)
    """
    config = TrainingConfig()
    hw_config = HardwareConfig()
    run_dir = setup_run_dir(config)
    metrics_logger = MetricsLogger(run_dir)
    
    log_file = run_dir / "train.log"
    logger = build_run_logger(log_file)

    logger.info(f"Production Run: {run_dir.name} | Architecture: 1.6B Pure Closure")
    logger.info(
        "Config | mbs=%d acc=%d eff_batch=%d workers=%d prefetch=%d queue_chunks=%d",
        config.micro_batch_size,
        config.gradient_accumulation_steps,
        config.effective_batch_size,
        config.num_worker_threads,
        config.num_prefetch_batches,
        config.token_queue_max_chunks,
    )
    cache_limit_gb_env = os.getenv("TRAIN_CACHE_LIMIT_GB")
    if cache_limit_gb_env:
        try:
            cache_limit_gb = float(cache_limit_gb_env)
            if cache_limit_gb > 0 and hasattr(mx, "set_cache_limit"):
                mx.set_cache_limit(int(cache_limit_gb * (1024 ** 3)))
                logger.info("Runtime | cache_limit_gb=%.1f applied via mx.set_cache_limit", cache_limit_gb)
            else:
                logger.info("Runtime | cache limit not applied (invalid value or API unavailable)")
        except Exception:
            logger.exception("Runtime | failed to apply TRAIN_CACHE_LIMIT_GB=%s", cache_limit_gb_env)
    try:
        with timed_log("train.initialize_model_and_optimizer", config.profile_methods, logger):
            model = GPTModel(config)

            # Cast to dtype and tie weights
            model.update(mut.tree_map(lambda p: p.astype(config.mx_dtype), model.parameters()))
            model.tie_weights()
            mx.eval(model.parameters())

            param_count = sum(getattr(v, "size", 0) for _, v in mut.tree_flatten(model.parameters()) if not isinstance(v, str))
            optimizer = optim.AdamW(learning_rate=config.learning_rate, betas=[config.beta1, config.beta2])
            expected_mem = _estimate_expected_memory_gb(config, param_count)
            model_dtypes = _tree_dtype_summary(model.parameters())
            opt_dtypes = _tree_dtype_summary(optimizer.state)

            # Build compiled micro-step (accumulation is done outside compile)
            micro_step = make_step(model, optimizer, config)

            # Data Pipeline
            token_stream = ParallelTokenStream(config)
            prefetcher = AsyncBatchPrefetcher(config, token_stream)
        logger.info("Data pipeline initialized; waiting for first prefetched full iteration")
        logger.info(
            "ExpectedMem[GB] | params %.1f | grads %.1f | opt %.1f | act_attn %.1f | act_other %.1f | total_no_temp %.1f",
            expected_mem["params"],
            expected_mem["grads"],
            expected_mem["opt_state"],
            expected_mem["attn_act"],
            expected_mem["other_act"],
            expected_mem["params"]
            + expected_mem["grads"]
            + expected_mem["opt_state"]
            + expected_mem["attn_act"]
            + expected_mem["other_act"],
        )
        logger.info(
            "ModelInfo | params %d (%.3fB) | model_dtypes [%s] | optimizer_state_dtypes [%s] | config_dtype %s",
            param_count,
            param_count / 1e9,
            model_dtypes,
            opt_dtypes,
            config.dtype,
        )
        logger.info("ExpectedMem note | grads and optimizer states are estimated conservatively as fp32.")
        logger.info(
            "ProgressMode | first iteration may be long (compile + %d micro-batches). Heartbeats enabled for iter 0.",
            config.gradient_accumulation_steps,
        )
        if config.profile_methods:
            _log_throughput_projection(config, hw_config, param_count, logger)
        _reset_peak_memory()
    except Exception:
        logger.exception("Initialization failed — aborting run")
        raise
    
    iteration = 0
    throughput_history = []

    try:
        while iteration < config.max_iters:
            iter_start = time.time()
            
            # 1. Fetch full iteration (list of micro-batches)
            with timed_log(f"Iter {iteration} fetch_full_iteration", config.profile_methods, logger):
                batch_x, batch_y = prefetcher.get_full_iteration()
            if iteration == 0:
                logger.info("First prefetched iteration received; entering compute step")
            
            # 2. Execute micro-steps with explicit per-step evaluation
            total_loss_value = 0.0
            accumulated_grads = mut.tree_map(lambda p: mx.zeros_like(p), model.parameters())

            # bf16 pre-step dtype assertion — confirms cast-back is holding
            if config.profile_methods:
                _check_model_dtype(model, config.dtype, iteration, logger)

            num_micro_batches = len(batch_x)
            for mb_idx, (x_mb, y_mb) in enumerate(zip(batch_x, batch_y), start=1):
                loss_mb, grads_mb = micro_step(x_mb, y_mb)

                if config.profile_methods and iteration > 0:
                    # Split eval: forward then backward separately to measure each cost.
                    # mx.eval(loss_mb) triggers the forward pass only (grads not yet needed).
                    # mx.eval(grads_mb) then triggers the backward pass.
                    t0 = time.perf_counter()
                    mx.eval(loss_mb)
                    t_fwd = time.perf_counter() - t0
                    t1 = time.perf_counter()
                    mx.eval(grads_mb)
                    t_bwd = time.perf_counter() - t1
                    ratio = t_fwd / t_bwd if t_bwd > 1e-6 else float("inf")
                    logger.info(
                        "Iter %d mb %d/%d | fwd %.3fs | bwd %.3fs | fwd:bwd=%.2f",
                        iteration, mb_idx, num_micro_batches, t_fwd, t_bwd, ratio,
                    )
                else:
                    with timed_log(f"Iter {iteration} micro_batch {mb_idx}/{num_micro_batches}", config.profile_methods, logger):
                        mx.eval(loss_mb, grads_mb)

                total_loss_value += loss_mb.item()
                accumulated_grads = mut.tree_map(lambda ag, g: ag + g, accumulated_grads, grads_mb)
                if iteration == 0 and (mb_idx == 1 or mb_idx % 4 == 0 or mb_idx == num_micro_batches):
                    logger.info(
                        "Iter 0 progress | micro-batch %d/%d | elapsed %.1fs",
                        mb_idx,
                        num_micro_batches,
                        time.time() - iter_start,
                    )

            with timed_log(f"Iter {iteration} reduce_and_norm", config.profile_methods, logger):
                scale = 1.0 / len(batch_x)
                loss_arr = mx.array(total_loss_value * scale)
                accumulated_grads = mut.tree_map(lambda g: g * scale, accumulated_grads)
                mx.eval(accumulated_grads)

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

            with timed_log(f"Iter {iteration} optimizer_update", config.profile_methods, logger):
                optimizer.update(model, accumulated_grads)

                # Cast model weights back to bfloat16 after the optimizer step.
                # AdamW computes the parameter update in fp32 (gradients get promoted
                # during the backward pass), leaving model.parameters() in fp32.
                # Casting back here halves parameter memory and restores fast
                # bfloat16 matmuls in the next forward/backward pass.
                # Adam m/v states remain fp32 — that is correct for numerical stability.
                model.update(mut.tree_map(lambda p: p.astype(config.mx_dtype), model.parameters()))
                model.tie_weights()

            # 3. Synchronous evaluation of updated state and scalars
            with timed_log(f"Iter {iteration} eval_and_sync", config.profile_methods, logger):
                mx.eval(model.parameters(), optimizer.state, loss_arr, norm_arr)
            
            # 4. Instrumentation
            dt = time.time() - iter_start
            tokens_per_sec = config.total_tokens_per_iter / dt
            throughput_history.append(tokens_per_sec)
            
            step_loss = loss_arr.item()
            step_norm = norm_arr.item()
            # Modernized memory API
            active_mem = _active_memory_gb()
            lr = get_lr(iteration, config)
            optimizer.learning_rate = lr
            
            mfu = ((6 * param_count * config.total_tokens_per_iter) / dt) / (hw_config.theoretical_tflops * 1e12)

            metrics_logger.log({
                "step": iteration, "train_loss": step_loss,
                "tokens_per_sec": tokens_per_sec, "learning_rate": lr,
                "vram_usage_gb": active_mem, "mfu_pct": mfu * 100
            })
            if iteration < 5 or iteration % 10 == 0:
                logger.info(f"Iter {iteration:4d} | Loss {step_loss:.4f} | tok/s {tokens_per_sec:6.0f} | MFU {mfu*100:4.1f}% | Mem {active_mem:.1f}GB")
                actual_params_gb = _tree_array_gb(model.parameters())
                actual_grads_gb = _tree_array_gb(accumulated_grads)
                actual_opt_gb = _tree_array_gb(optimizer.state)
                model_dtypes_now = _tree_dtype_summary(model.parameters())
                grad_dtypes_now = _tree_dtype_summary(accumulated_grads)
                opt_dtypes_now = _tree_dtype_summary(optimizer.state)
                cache_gb = _mx_memory_gb("get_cache_memory")
                peak_gb = _mx_memory_gb("get_peak_memory")
                rss_gb = _process_rss_gb()
                residual_gb = active_mem - (actual_params_gb + actual_grads_gb + actual_opt_gb)
                logger.info(
                    "ActualMem[GB]   | params %.1f | grads %.1f | opt %.1f | residual %.1f | mx_active %.1f | mx_cache %s | mx_peak %s | os_rss %.1f",
                    actual_params_gb,
                    actual_grads_gb,
                    actual_opt_gb,
                    residual_gb,
                    active_mem,
                    f"{cache_gb:.1f}" if cache_gb is not None else "n/a",
                    f"{peak_gb:.1f}" if peak_gb is not None else "n/a",
                    rss_gb,
                )
                logger.info(
                    "DtypeInfo      | model [%s] | grads [%s] | optimizer_state [%s]",
                    model_dtypes_now,
                    grad_dtypes_now,
                    opt_dtypes_now,
                )
                logger.info(
                    "ExpectedMem[GB] | params %.1f | grads %.1f | opt %.1f | act_attn %.1f | act_other %.1f",
                    expected_mem["params"],
                    expected_mem["grads"],
                    expected_mem["opt_state"],
                    expected_mem["attn_act"],
                    expected_mem["other_act"],
                )
                _reset_peak_memory()

            # Checkpoint saving
            if iteration > 0 and iteration % config.save_interval == 0:
                ckpt_dir = run_dir / "checkpoints"
                ckpt_dir.mkdir(exist_ok=True)
                ckpt_path = ckpt_dir / f"step_{iteration:06d}.safetensors"
                model.save_weights(str(ckpt_path))
                all_ckpts = sorted(ckpt_dir.glob("step_*.safetensors"))
                for old_ckpt in all_ckpts[:-config.keep_checkpoints]:
                    old_ckpt.unlink()
                logger.info(f"Checkpoint saved: {ckpt_path.name} (kept last {config.keep_checkpoints})")

            iteration += 1
            if iteration % 20 == 0:
                mx.clear_cache()
                
    except KeyboardInterrupt:
        logger.info("Interrupted.")
    except Exception:
        logger.exception("Training crashed")
    finally:
        prefetcher.stop()
        token_stream.stop()

if __name__ == "__main__":
    train()
