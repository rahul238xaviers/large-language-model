"""Stage 6 — token-level perplexity evaluation.

Perplexity is the standard language-model evaluation metric:

    PPL = exp( (1/N) Σ -log p(tᵢ | t₁…tᵢ₋₁) )

where the sum runs over all N predicted tokens across the eval corpus.

This module operates on the ``sequences.npy`` file from Stage 4 using the
last ``eval_fraction`` fraction of sequences as a held-out eval split.

Design notes
------------
* Sequences are **not** shuffled before splitting so the held-out set is
  always the tail of the dataset (deterministic across runs).
* The forward pass is run with ``model.eval()`` (disables dropout) and
  MLX does not track gradients here so no ``value_and_grad`` overhead.
* Batches are evaluated in ``eval_batch_size`` chunks and losses are
  accumulated in float32 regardless of model dtype.

Public interface:
    EvalResult  (dataclass)
    compute_perplexity(model, sequences_path, eval_cfg) → EvalResult
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class EvalResult:
    """Evaluation metrics for a single checkpoint.

    Attributes:
        n_sequences:    Number of sequences evaluated.
        n_tokens:       Total predicted tokens (n_sequences × (block_size - 1)).
        mean_loss:      Average cross-entropy loss per token.
        perplexity:     ``exp(mean_loss)``.
        eval_fraction:  Fraction of the dataset used for evaluation.
    """
    n_sequences:  int
    n_tokens:     int
    mean_loss:    float
    perplexity:   float
    eval_fraction: float

    def to_dict(self) -> dict:
        return {
            "n_sequences":   self.n_sequences,
            "n_tokens":      self.n_tokens,
            "mean_loss":     round(self.mean_loss,  6),
            "perplexity":    round(self.perplexity, 4),
            "eval_fraction": self.eval_fraction,
        }


def compute_perplexity(
    model: object,
    sequences_path: Path,
    eval_cfg: dict,
) -> EvalResult:
    """Compute token-level perplexity on the held-out eval split.

    Args:
        model:           Loaded ``GPTModel`` instance (weights applied).
        sequences_path:  Path to ``sequences.npy`` from Stage 4.
        eval_cfg:        Dict from ``configs/base/eval.yaml``::

                            eval_fraction: 0.05
                            eval_batch_size: 8

    Returns:
        ``EvalResult`` with loss and perplexity.

    Raises:
        FileNotFoundError: If ``sequences_path`` does not exist.
        ValueError:        If the eval split is empty.
    """
    import mlx.core as mx
    import mlx.nn as nn

    if not sequences_path.exists():
        raise FileNotFoundError(f"Sequences file not found: {sequences_path}")

    eval_fraction: float = float(eval_cfg.get("eval_fraction", 0.05))
    eval_batch_size: int = int(eval_cfg.get("eval_batch_size", 8))

    # ── Load & split ─────────────────────────────────────────────────── #
    arr = np.load(str(sequences_path), mmap_mode="r")  # (N, T) uint32
    n_total = arr.shape[0]
    n_eval  = max(1, int(n_total * eval_fraction))

    # Tail split — deterministic, no shuffle
    eval_arr = arr[n_total - n_eval :]
    logger.info(
        "Eval split: last %d/%d sequences (%.1f%%)",
        n_eval, n_total, eval_fraction * 100,
    )

    if len(eval_arr) == 0:
        raise ValueError("Eval split is empty. Increase eval_fraction or dataset size.")

    # ── Evaluation loop ───────────────────────────────────────────────── #
    total_loss   = 0.0
    total_tokens = 0
    n_batches    = math.ceil(n_eval / eval_batch_size)

    for i in range(n_batches):
        batch_np = np.array(
            eval_arr[i * eval_batch_size : (i + 1) * eval_batch_size],
            dtype=np.int32,
        )
        x       = mx.array(batch_np)          # (B, T)
        logits  = model(x[:, :-1])            # (B, T-1, V)
        targets = x[:, 1:].astype(mx.int32)   # (B, T-1)

        # cross_entropy returns per-token loss; we want the mean over the batch
        loss_per_token = nn.losses.cross_entropy(logits, targets, reduction="none")
        mx.eval(loss_per_token)

        batch_tokens = loss_per_token.size
        batch_loss   = float(mx.sum(loss_per_token.astype(mx.float32)).item())
        total_loss   += batch_loss
        total_tokens += batch_tokens

    mean_loss  = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(mean_loss, 20.0))   # cap at exp(20) ≈ 485M to avoid inf

    logger.info(
        "Eval complete: n_seq=%d  n_tokens=%d  loss=%.4f  ppl=%.2f",
        n_eval, total_tokens, mean_loss, perplexity,
    )

    return EvalResult(
        n_sequences   = n_eval,
        n_tokens      = total_tokens,
        mean_loss     = mean_loss,
        perplexity    = perplexity,
        eval_fraction = eval_fraction,
    )
