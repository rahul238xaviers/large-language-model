import os
import sys
from pathlib import Path
from typing import Optional

import pytest

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "pre-training"
sys.path.insert(0, str(SRC))

from config import TrainingConfig
from model import GPTModel


def _find_latest_checkpoint() -> Optional[Path]:
    env_path = os.getenv("CHECKPOINT_PATH")
    if not env_path:
        return None
    return Path(env_path)


def test_saved_checkpoint_loads_and_infers():
    checkpoint_path = _find_latest_checkpoint()
    if checkpoint_path is None or not checkpoint_path.exists():
        pytest.skip("No saved checkpoint available in runs/*/checkpoints/step_*.safetensors")

    config = TrainingConfig()
    model = GPTModel(config)
    model.tie_weights()
    model.set_dtype(config.mx_dtype)

    if hasattr(model, "load_weights"):
        model.load_weights(str(checkpoint_path))
    else:
        import mlx.core as mx

        load_fn = getattr(mx, "load_safetensors", None)
        if load_fn is not None:
            state = load_fn(str(checkpoint_path))
        else:
            from safetensors import safe_open

            state = {}
            with safe_open(str(checkpoint_path), framework="numpy") as f:
                for key in f.keys():
                    state[key] = f.get_tensor(key)
        model.update(state)

    model.tie_weights()

    import mlx.core as mx

    # Use a short sequence for inference to keep the test lightweight.
    input_ids = mx.random.randint(0, config.vocab_size, (1, 16), dtype=mx.int32)
    logits = model(input_ids)
    mx.eval(logits)

    assert logits.shape == (1, 16, config.vocab_size)
    assert logits.dtype == config.mx_dtype
    assert mx.all(mx.isfinite(logits)).item(), "Logits contain non-finite values"
