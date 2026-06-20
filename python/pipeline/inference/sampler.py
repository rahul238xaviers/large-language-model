"""Token sampling utilities — Top-K, Top-P, repetition penalty, and the
auto-regressive ``generate`` loop.

All functions operate on MLX arrays and are model-agnostic.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

import mlx.core as mx


# ------------------------------------------------------------------ #
# Sampler configuration                                                 #
# ------------------------------------------------------------------ #

@dataclass
class SamplerConfig:
    """Decode hyper-parameters."""
    max_new_tokens:      int   = 128
    temperature:         float = 0.7
    top_k:               int   = 50
    top_p:               float = 0.9
    repetition_penalty:  float = 1.15
    # Token IDs treated as whitespace — never penalised during rep-penalty.
    # cl100k_base common whitespace IDs: space(220), tab(256), newline(198), indent(385)
    whitespace_ids: list[int] = field(default_factory=lambda: [220, 256, 198, 385])

    @classmethod
    def from_dict(cls, d: dict) -> "SamplerConfig":
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        return asdict(self)


# ------------------------------------------------------------------ #
# Sampling primitives                                                   #
# ------------------------------------------------------------------ #

def apply_repetition_penalty(
    logits: mx.array,
    generated_ids: list[int],
    penalty: float,
    whitespace_ids: set[int] | None = None,
    window: int = 128,
) -> mx.array:
    """Reduce the probability of recently generated tokens.

    The penalty is applied multiplicatively (divides positive logits,
    multiplies negative ones) plus an additive term that breaks high-
    confidence repetition loops.  Whitespace tokens are excluded.
    """
    if penalty == 1.0 or not generated_ids:
        return logits
    ws = whitespace_ids or {220, 256, 198, 385}
    seen = set(generated_ids[-window:]) - ws
    if not seen:
        return logits
    logits_list: list[float] = logits.tolist()  # type: ignore[assignment]
    for tid in seen:
        v = logits_list[tid]
        logits_list[tid] = (
            (v / penalty if v > 0.0 else v * penalty)
            - (penalty - 1.0) * 35.0
        )
    return mx.array(logits_list)


def apply_top_k(logits: mx.array, k: int) -> mx.array:
    """Zero out all logits below the k-th highest value."""
    if k <= 0:
        return logits
    sorted_vals = mx.sort(logits)           # ascending order
    threshold = sorted_vals[-k]
    return mx.where(logits >= threshold, logits, mx.array(float("-inf")))


def apply_top_p(logits: mx.array, p: float) -> mx.array:
    """Zero out the bottom-probability mass so only the top-p nucleus remains."""
    if p >= 1.0:
        return logits
    sorted_idx    = mx.argsort(-logits)     # descending index order
    sorted_logits = logits[sorted_idx]
    probs         = mx.softmax(sorted_logits)
    cum_probs     = mx.cumsum(probs)
    mx.eval(cum_probs)                      # materialise before .tolist()
    cum_list = cum_probs.tolist()
    if isinstance(cum_list, float):         # guard for 1-element tensors
        cum_list = [cum_list]
    cutoff = next((i for i, v in enumerate(cum_list) if v > p), -1)
    if cutoff == -1:
        return logits
    cutoff = max(1, cutoff)
    threshold_val = sorted_logits[cutoff]
    return mx.where(logits >= threshold_val, logits, mx.array(float("-inf")))


def sample_token(logits: mx.array, temperature: float) -> int:
    """Draw a token from the (temperature-scaled) distribution."""
    if temperature > 0.0:
        token_arr = mx.random.categorical(logits / temperature)
    else:
        token_arr = mx.argmax(logits, axis=-1)
    mx.eval(token_arr)
    return int(token_arr.item()) if hasattr(token_arr, "item") else int(token_arr)


# ------------------------------------------------------------------ #
# Auto-regressive generate loop                                         #
# ------------------------------------------------------------------ #

def generate(
    model: Any,
    prompt_ids: list[int],
    eot_id: int,
    block_size: int,
    cfg: SamplerConfig,
) -> Iterator[int]:
    """Yield new token IDs one at a time via auto-regressive decoding.

    Stops when ``eot_id`` is produced or ``cfg.max_new_tokens`` is reached.
    ``model`` must be callable as ``model(x: mx.array) -> mx.array`` where
    the output is ``(batch=1, T, vocab_size)``.
    """
    generated = list(prompt_ids)
    ws_set = set(cfg.whitespace_ids)

    for _ in range(cfg.max_new_tokens):
        context = generated[-block_size:]
        x       = mx.array([context], dtype=mx.int32)
        logits_all = model(x)
        mx.eval(logits_all)
        logits = logits_all[0, -1]   # (vocab_size,)

        logits = apply_repetition_penalty(
            logits, generated, cfg.repetition_penalty, whitespace_ids=ws_set
        )
        logits = apply_top_k(logits, cfg.top_k)
        logits = apply_top_p(logits, cfg.top_p)

        token = sample_token(logits, cfg.temperature)
        if token == eot_id:
            break
        generated.append(token)
        yield token
