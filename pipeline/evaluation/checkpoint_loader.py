"""Stage 6 — checkpoint loader.

Loads a ``.safetensors`` checkpoint produced by the Stage 5 trainer into
a freshly-instantiated ``GPTModel``, ready for evaluation or inference.

The function validates that every parameter key expected by the model is
present in the checkpoint and logs a warning for any unexpected extra keys
(e.g. from a future model version) so mismatches are caught early.

Public interface:
    load_checkpoint(checkpoint_path, model_cfg) → GPTModel
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

_REPO_ROOT  = Path(__file__).parents[2]       # .../large-language-model/
_LEGACY_SRC = _REPO_ROOT / "apple-silicon" / "src"


def _ensure_legacy_on_path() -> None:
    resolved = str(_LEGACY_SRC.resolve())
    if resolved not in sys.path:
        sys.path.insert(0, resolved)


def load_checkpoint(
    checkpoint_path: Path,
    model_cfg: dict,
) -> object:
    """Load a ``.safetensors`` checkpoint into a ``GPTModel``.

    Args:
        checkpoint_path:  Path to a ``step_*.safetensors`` file from Stage 5.
        model_cfg:        Dict with model architecture keys (``n_layer``,
                          ``n_embd``, ``n_head``, ``n_kv_heads``,
                          ``block_size``, ``vocab_size``, ``dropout``).

    Returns:
        ``GPTModel`` instance with weights loaded from the checkpoint,
        in bfloat16 and ready for ``mx.eval``.

    Raises:
        FileNotFoundError: If ``checkpoint_path`` does not exist.
        KeyError:          If required parameter keys are missing from
                           the checkpoint.
    """
    import mlx.core as mx

    _ensure_legacy_on_path()
    from model import GPTModel  # type: ignore[import]  — apple-silicon/src/

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # ── Build model skeleton ──────────────────────────────────────────── #
    model = GPTModel(
        vocab_size  = int(model_cfg.get("vocab_size",  100_277)),
        n_embd      = int(model_cfg.get("n_embd",      2048)),
        n_layer     = int(model_cfg.get("n_layer",     24)),
        n_head      = int(model_cfg.get("n_head",      16)),
        n_kv_heads  = int(model_cfg.get("n_kv_heads",  8)),
        block_size  = int(model_cfg.get("block_size",  2048)),
        dropout     = float(model_cfg.get("dropout",   0.0)),
    )

    # ── Load weights ─────────────────────────────────────────────────── #
    saved_weights = mx.load(str(checkpoint_path))

    # Key validation
    model_keys    = set(k for k, _ in model.parameters())
    ckpt_keys     = set(saved_weights.keys())
    missing  = model_keys - ckpt_keys
    extra    = ckpt_keys  - model_keys

    if missing:
        raise KeyError(
            f"Checkpoint is missing {len(missing)} parameter keys: "
            f"{sorted(missing)[:10]}{'...' if len(missing) > 10 else ''}"
        )
    if extra:
        logger.warning(
            "Checkpoint has %d unexpected extra keys (ignored): %s",
            len(extra), sorted(extra)[:10],
        )

    model.load_weights(str(checkpoint_path))
    mx.eval(model)

    logger.info(
        "Loaded checkpoint: %s  (n_layer=%d  n_embd=%d)",
        checkpoint_path.name,
        model_cfg.get("n_layer", "?"),
        model_cfg.get("n_embd", "?"),
    )
    return model
