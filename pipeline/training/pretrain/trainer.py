"""Stage 5 — GPT pretraining wrapper.

Wraps the legacy GPT model architecture (``apple-silicon/src/model.py``) and
optional utility helpers (``apple-silicon/src/utils.py``) with a clean,
reproducible training loop that:

  * Reads tokenised sequences produced by Stage 4 via ``SequenceLoader``
  * Accumulates gradients over ``accum_steps`` micro-batches for large
    effective batch sizes without OOM
  * Applies global gradient clipping and AdamW weight updates
  * Schedules learning rate via cosine decay with linear warmup
  * Casts parameters back to bfloat16 after each optimizer step
  * Saves ``.safetensors`` checkpoints and a ``metrics.csv`` into the
    pipeline run directory for full Stage 6 compatibility

Legacy files in ``apple-silicon/src/`` are imported via ``sys.path``
injection and are **never modified** — Stage 5 consumes the model class only.

Public interface:
    TrainingRunConfig.from_dict(cfg_dict)  → frozen dataclass
    Trainer(config, run_dir, sequences_path).train()
"""

from __future__ import annotations

import csv
import logging
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

logger = logging.getLogger(__name__)

# ── Legacy src path injection ────────────────────────────────────────── #
# apple-silicon/src/ is intentionally NOT a Python package so that it
# remains usable as a standalone script collection (legacy contract).
# We inject it into sys.path at import time without mutating any source.

_REPO_ROOT  = Path(__file__).parents[3]      # .../large-language-model/
_LEGACY_SRC = _REPO_ROOT / "apple-silicon" / "src" / "pre-training"


def _ensure_legacy_on_path() -> None:
    resolved = str(_LEGACY_SRC.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)
        logger.debug("Injected legacy src into sys.path: %s", resolved)


# ────────────────────────────────────────────────────────────────────────────
# Configuration
# ────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TrainingRunConfig:
    """Immutable snapshot of all hyperparameters for a single pretraining run.

    Construct via ``TrainingRunConfig.from_dict(merged_config_dict)`` where
    *merged_config_dict* is the full pipeline config (the ``model:`` and
    ``training:`` top-level sections are read separately).
    """

    # ── Model architecture ─────────────────────────────────────────── #
    n_layer:    int   = 24
    n_embd:     int   = 2048
    n_head:     int   = 16
    n_kv_heads: int   = 8
    block_size: int   = 2048
    vocab_size: int   = 100_277
    dropout:    float = 0.0

    # ── Optimisation ─────────────────────────────────────────────────── #
    batch_size:    int   = 4        # sequences per micro-forward-backward pass
    accum_steps:   int   = 32       # gradient accumulation → effective batch = 128
    max_steps:     int   = 10_000
    warmup_steps:  int   = 500
    lr_max:        float = 3e-4
    lr_min:        float = 3e-5
    weight_decay:  float = 0.1
    grad_clip:     float = 1.0
    seed:          int   = 42

    # ── Checkpointing & logging ───────────────────────────────────────── #
    log_interval:            int = 10
    eval_interval:           int = 500
    checkpoint_interval:     int = 500
    keep_last_n_checkpoints: int = 3

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingRunConfig":
        """Build config from the merged pipeline config dict."""
        m = d.get("model",    {})
        t = d.get("training", {})
        return cls(
            n_layer    = int(m.get("n_layer",    24)),
            n_embd     = int(m.get("n_embd",     2048)),
            n_head     = int(m.get("n_head",     16)),
            n_kv_heads = int(m.get("n_kv_heads", 8)),
            block_size = int(m.get("block_size", 2048)),
            vocab_size = int(m.get("vocab_size", 100_277)),
            dropout    = float(m.get("dropout",  0.0)),
            batch_size   = int(t.get("batch_size",   4)),
            accum_steps  = int(t.get("accum_steps",  32)),
            max_steps    = int(t.get("max_steps",    10_000)),
            warmup_steps = int(t.get("warmup_steps", 500)),
            lr_max       = float(t.get("lr_max",     3e-4)),
            lr_min       = float(t.get("lr_min",     3e-5)),
            weight_decay = float(t.get("weight_decay", 0.1)),
            grad_clip    = float(t.get("grad_clip",    1.0)),
            seed         = int(t.get("seed", 42)),
            log_interval            = int(t.get("log_interval",            10)),
            eval_interval           = int(t.get("eval_interval",           500)),
            checkpoint_interval     = int(t.get("checkpoint_interval",     500)),
            keep_last_n_checkpoints = int(t.get("keep_last_n_checkpoints", 3)),
        )

    @property
    def effective_batch_size(self) -> int:
        """Sequences consumed per weight update (micro_batch × accum_steps)."""
        return self.batch_size * self.accum_steps

    @property
    def tokens_per_step(self) -> int:
        """Tokens processed per weight update."""
        return self.effective_batch_size * self.block_size


# ────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ────────────────────────────────────────────────────────────────────────────

def _cosine_lr(step: int, cfg: TrainingRunConfig) -> float:
    """Linear warmup followed by cosine decay to ``lr_min``."""
    if step < cfg.warmup_steps:
        # step=0 → 0.0 (cold start); step=warmup_steps-1 → just below lr_max
        return cfg.lr_max * step / max(cfg.warmup_steps, 1)
    if step >= cfg.max_steps:
        return cfg.lr_min
    progress = (step - cfg.warmup_steps) / max(cfg.max_steps - cfg.warmup_steps, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return cfg.lr_min + coeff * (cfg.lr_max - cfg.lr_min)


def _clip_grads(grads: dict, max_norm: float) -> tuple[dict, float]:
    """Global gradient norm clipping over the full parameter tree.

    Returns the (possibly rescaled) gradient tree and the pre-clip norm.
    """
    import mlx.core as mx

    def _iter_leaves(node):
        """Yield all mx.array leaves in a nested dict/list/tuple tree."""
        if isinstance(node, mx.array):
            yield node
        elif isinstance(node, dict):
            for v in node.values():
                yield from _iter_leaves(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                yield from _iter_leaves(v)

    leaves = list(_iter_leaves(grads))
    if not leaves:
        return grads, 0.0

    sq_sum = sum(
        float(mx.sum(g.astype(mx.float32) ** 2).item()) for g in leaves
    )
    norm = sq_sum ** 0.5

    if norm > max_norm:
        scale = max_norm / (norm + 1e-6)

        def _rescale(node):
            if isinstance(node, mx.array):
                return node * scale
            elif isinstance(node, dict):
                return {k: _rescale(v) for k, v in node.items()}
            elif isinstance(node, list):
                return [_rescale(v) for v in node]
            elif isinstance(node, tuple):
                return tuple(_rescale(v) for v in node)
            return node

        grads = _rescale(grads)

    return grads, norm


def _add_grad_trees(tree_a: dict, tree_b: dict):
    """Element-wise addition of two gradient trees (same structure required)."""
    import mlx.core as mx

    if isinstance(tree_a, mx.array) and isinstance(tree_b, mx.array):
        return tree_a + tree_b
    if isinstance(tree_a, dict) and isinstance(tree_b, dict):
        return {k: _add_grad_trees(tree_a[k], tree_b[k]) for k in tree_a}
    if isinstance(tree_a, (list, tuple)) and isinstance(tree_b, (list, tuple)):
        merged = [_add_grad_trees(a, b) for a, b in zip(tree_a, tree_b)]
        return type(tree_a)(merged)
    # Scalars / None: keep tree_a
    return tree_a


def _scale_grad_tree(tree, scale: float):
    """Multiply every mx.array in the gradient tree by *scale*."""
    import mlx.core as mx

    if isinstance(tree, mx.array):
        return tree * scale
    if isinstance(tree, dict):
        return {k: _scale_grad_tree(v, scale) for k, v in tree.items()}
    if isinstance(tree, list):
        return [_scale_grad_tree(v, scale) for v in tree]
    if isinstance(tree, tuple):
        return tuple(_scale_grad_tree(v, scale) for v in tree)
    return tree


def _infinite_loader(loader: "SequenceLoader") -> Iterator:  # noqa: F821
    """Cycle a SequenceLoader forever, reshuffling at each epoch boundary."""
    epoch = 0
    while True:
        for batch in loader:
            yield batch
        epoch += 1
        logger.debug("Data epoch %d exhausted — reshuffling", epoch)


# ────────────────────────────────────────────────────────────────────────────
# Trainer
# ────────────────────────────────────────────────────────────────────────────

class Trainer:
    """Pretraining loop for the pipeline GPT model.

    Reads tokenised sequences from Stage 4, trains the legacy GPT architecture,
    and writes artefacts into the standard pipeline run directory layout:

        <run_dir>/training/
            metrics.csv
            checkpoints/
                step_0000500.safetensors
                step_0001000.safetensors
                ...

    Args:
        config:          ``TrainingRunConfig`` with all hyperparameters.
        run_dir:         Pipeline run directory root.
        sequences_path:  Path to ``sequences.npy`` from Stage 4.
    """

    def __init__(
        self,
        config: TrainingRunConfig,
        run_dir: Path,
        sequences_path: Path,
    ) -> None:
        self._config         = config
        self._run_dir        = Path(run_dir)
        self._sequences_path = Path(sequences_path)

        self._train_dir      = self._run_dir / "training"
        self._ckpt_dir       = self._train_dir / "checkpoints"
        self._metrics_path   = self._train_dir / "metrics.csv"

        self._train_dir.mkdir(parents=True, exist_ok=True)
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Setup helpers ────────────────────────────────────────────────── #

    def _load_model(self):
        """Instantiate and return a freshly-initialised ``GPTModel`` in bfloat16."""
        import mlx.core as mx

        _ensure_legacy_on_path()
        from model import GPTModel  # type: ignore[import]  — apple-silicon/src/

        cfg = self._config
        model = GPTModel(
            vocab_size = cfg.vocab_size,
            n_embd     = cfg.n_embd,
            n_layer    = cfg.n_layer,
            n_head     = cfg.n_head,
            n_kv_heads = cfg.n_kv_heads,
            block_size = cfg.block_size,
            dropout    = cfg.dropout,
        )

        # Cast all parameters to bfloat16 for Apple Silicon memory efficiency
        def _to_bf16(leaf):
            return leaf.astype(mx.bfloat16) if isinstance(leaf, mx.array) else leaf

        def _cast_tree(node):
            if isinstance(node, mx.array):
                return _to_bf16(node)
            if isinstance(node, dict):
                return {k: _cast_tree(v) for k, v in node.items()}
            if isinstance(node, (list, tuple)):
                casted = [_cast_tree(v) for v in node]
                return type(node)(casted)
            return node

        model.update(_cast_tree(model.parameters()))
        mx.eval(model)

        n_params = sum(p.size for _, p in model.parameters() if hasattr(p, "size"))
        logger.info(
            "Model: ~%.2fB params  dtype=bfloat16  "
            "effective_batch=%d  tokens/step=%d",
            n_params / 1e9,
            cfg.effective_batch_size,
            cfg.tokens_per_step,
        )
        return model

    def _build_optimizer(self):
        """Build AdamW with the configured learning rate and weight decay."""
        import mlx.optimizers as optim

        cfg = self._config
        return optim.AdamW(
            learning_rate=cfg.lr_max,
            weight_decay=cfg.weight_decay,
        )

    # ── Training loop ────────────────────────────────────────────────── #

    def train(self) -> None:
        """Execute the full pretraining loop.

        Raises:
            FileNotFoundError: If ``sequences_path`` does not exist.
        """
        import mlx.core as mx
        import mlx.nn as nn

        from pipeline.training.pretrain.sequence_loader import SequenceLoader

        cfg   = self._config
        model = self._load_model()
        opt   = self._build_optimizer()

        loader = SequenceLoader(
            self._sequences_path, cfg.batch_size, shuffle=True, seed=cfg.seed
        )
        data = _infinite_loader(loader)

        # ── Forward pass (used by value_and_grad) ─────────────────────── #
        def _loss(mdl, x: "mx.array") -> "mx.array":
            logits  = mdl(x[:, :-1])
            targets = x[:, 1:].astype(mx.int32)
            return nn.losses.cross_entropy(logits, targets, reduction="mean")

        # ── Metrics CSV ───────────────────────────────────────────────── #
        if not self._metrics_path.exists():
            with open(self._metrics_path, "w", newline="") as f:
                csv.writer(f).writerow(
                    ["step", "train_loss", "learning_rate",
                     "tok_per_sec", "grad_norm", "memory_gb"]
                )

        # ── Try to get memory helper from legacy utils ─────────────────── #
        _ensure_legacy_on_path()
        try:
            from utils import _active_memory_gb  # type: ignore[import]
        except Exception:
            def _active_memory_gb():  # type: ignore[no-redef]
                return 0.0

        logger.info(
            "Training: max_steps=%d  warmup=%d  lr_max=%.1e  lr_min=%.1e",
            cfg.max_steps, cfg.warmup_steps, cfg.lr_max, cfg.lr_min,
        )

        # ── Main loop ─────────────────────────────────────────────────── #
        for step in range(1, cfg.max_steps + 1):
            lr = _cosine_lr(step, cfg)
            opt.learning_rate = lr

            t0 = time.perf_counter()
            accum_grads = None
            accum_loss  = 0.0
            scale       = 1.0 / cfg.accum_steps

            # Gradient accumulation: accum_steps micro-forward-backward passes
            for _ in range(cfg.accum_steps):
                x = next(data)
                loss, grads = nn.value_and_grad(model, _loss)(model, x)
                mx.eval(loss)
                accum_loss += float(loss.item()) * scale

                grads = _scale_grad_tree(grads, scale)
                if accum_grads is None:
                    accum_grads = grads
                else:
                    accum_grads = _add_grad_trees(accum_grads, grads)

            # Clip + update
            accum_grads, grad_norm = _clip_grads(accum_grads, cfg.grad_clip)
            opt.update(model, accum_grads)

            # Cast parameters back to bfloat16 (optimizer states are float32)
            def _bf16(node):
                if isinstance(node, mx.array) and node.dtype != mx.bfloat16:
                    return node.astype(mx.bfloat16)
                if isinstance(node, dict):
                    return {k: _bf16(v) for k, v in node.items()}
                if isinstance(node, (list, tuple)):
                    return type(node)(_bf16(v) for v in node)
                return node

            model.update(_bf16(model.parameters()))
            mx.eval(model)

            elapsed    = time.perf_counter() - t0
            tok_per_sec = cfg.tokens_per_step / max(elapsed, 1e-9)
            mem_gb      = _active_memory_gb()

            # ── Log ───────────────────────────────────────────────────── #
            if step % cfg.log_interval == 0:
                logger.info(
                    "step=%7d  loss=%.4f  lr=%.2e  tok/s=%.0f  "
                    "grad_norm=%.3f  mem=%.1fGB",
                    step, accum_loss, lr, tok_per_sec, grad_norm, mem_gb,
                )

            with open(self._metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([
                    step,
                    round(accum_loss,  6),
                    round(lr,          8),
                    round(tok_per_sec, 1),
                    round(grad_norm,   4),
                    round(mem_gb,      2),
                ])

            # ── Checkpoint ────────────────────────────────────────────── #
            if step % cfg.checkpoint_interval == 0 or step == cfg.max_steps:
                ckpt_path = self._ckpt_dir / f"step_{step:07d}.safetensors"
                mx.save_safetensors(str(ckpt_path), dict(model.parameters()))
                logger.info("Saved checkpoint: %s", ckpt_path.name)

                # Prune old checkpoints beyond keep_last_n
                ckpts = sorted(self._ckpt_dir.glob("step_*.safetensors"))
                for old in ckpts[: -cfg.keep_last_n_checkpoints]:
                    old.unlink(missing_ok=True)
                    logger.debug("Pruned: %s", old.name)

        logger.info("Training complete — %d steps finished.", cfg.max_steps)
