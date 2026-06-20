"""Stage 5 smoke test — run via: python3 -m pipeline._tests.test_stage5_smoke"""
import json
import tempfile
from pathlib import Path

import numpy as np

# ── Stage 5 imports ───────────────────────────────────────────────────
from pipeline.training.pretrain.sequence_loader import SequenceLoader
from pipeline.training.pretrain.trainer import (
    TrainingRunConfig,
    _cosine_lr,
    _clip_grads,
    _scale_grad_tree,
    _add_grad_trees,
)

print("All Stage 5 imports OK")

# ── SequenceLoader unit test ──────────────────────────────────────────
with tempfile.TemporaryDirectory() as tmp:
    npy_path = Path(tmp) / "sequences.npy"

    # 50 sequences of length 32, values in [0, 99]
    arr = np.random.randint(0, 100, size=(50, 32), dtype=np.uint32)
    np.save(str(npy_path), arr)

    loader = SequenceLoader(npy_path, batch_size=8, shuffle=True, seed=0)
    assert len(loader) == 6,   f"Expected 6 batches, got {len(loader)}"
    assert loader.block_size == 32

    import mlx.core as mx
    batches = list(loader)
    assert len(batches) == 6
    assert batches[0].shape == (8, 32)
    assert batches[0].dtype == mx.int32   # cast from uint32

    # Confirm shuffle reproducibility
    loader2 = SequenceLoader(npy_path, batch_size=8, shuffle=True, seed=0)
    batches2 = list(loader2)
    # Same seed → same first batch
    assert np.allclose(np.array(batches[0]), np.array(batches2[0]))

    print(f"SequenceLoader OK: {len(batches)} batches  shape={batches[0].shape}  dtype={batches[0].dtype}")

# ── TrainingRunConfig round-trip ──────────────────────────────────────
cfg = TrainingRunConfig.from_dict({
    "model":    {"n_layer": 2, "n_embd": 64, "n_head": 2, "n_kv_heads": 2,
                 "block_size": 32, "vocab_size": 100},
    "training": {"batch_size": 2, "accum_steps": 4, "max_steps": 10,
                 "warmup_steps": 2, "lr_max": 1e-3, "lr_min": 1e-5,
                 "grad_clip": 1.0, "weight_decay": 0.1},
})
assert cfg.n_layer == 2
assert cfg.effective_batch_size == 8     # 2 × 4
assert cfg.tokens_per_step == 256        # 8 × 32
print(f"TrainingRunConfig OK: effective_batch={cfg.effective_batch_size}  tokens/step={cfg.tokens_per_step}")

# ── LR schedule ───────────────────────────────────────────────────────
lr_step0 = _cosine_lr(0, cfg)
lr_step1 = _cosine_lr(1, cfg)            # warmup
lr_step5 = _cosine_lr(5, cfg)            # mid-cosine
lr_end   = _cosine_lr(cfg.max_steps, cfg)
assert lr_step0 == 0.0, f"LR at step 0 should be 0, got {lr_step0}"
assert lr_step1 > 0
assert lr_step5 > cfg.lr_min
assert abs(lr_end - cfg.lr_min) < 1e-9
print(f"LR schedule OK: step0={lr_step0}  step1={lr_step1:.4e}  mid={lr_step5:.4e}  end={lr_end:.4e}")

# ── Gradient helpers ──────────────────────────────────────────────────
a = mx.array([1.0, 2.0, 3.0])
b = mx.array([4.0, 5.0, 6.0])

grads = {"layer": {"w": a, "b": b}}
scaled = _scale_grad_tree(grads, 0.5)
assert np.allclose(np.array(scaled["layer"]["w"]), [0.5, 1.0, 1.5])

summed = _add_grad_trees(grads, grads)
assert np.allclose(np.array(summed["layer"]["w"]), [2.0, 4.0, 6.0])

# Clipping: norm > max_norm should rescale
big_grads = {"w": mx.array([10.0, 10.0, 10.0])}
clipped, norm = _clip_grads(big_grads, max_norm=1.0)
clipped_arr = np.array(clipped["w"])
assert norm > 1.0
assert np.linalg.norm(clipped_arr) <= 1.01   # within tolerance
print(f"Gradient helpers OK: pre-clip norm={norm:.3f}  post-clip norm={np.linalg.norm(clipped_arr):.4f}")

# ── Config carries training section ──────────────────────────────────
from pipeline.orchestration.config_loader import ConfigLoader
full_cfg = ConfigLoader().load("local-dev")
assert full_cfg.get("model", {}).get("n_layer") == 24
assert full_cfg.get("training", {}).get("max_steps") == 10000
print("training.yaml merged into config OK:", {
    "n_layer": full_cfg["model"]["n_layer"],
    "max_steps": full_cfg["training"]["max_steps"],
})

# ── CLI parses train-pretrain ─────────────────────────────────────────
from pipeline.cli import _build_parser
p = _build_parser()
args = p.parse_args(["train-pretrain", "--run-id", "run_fake_000"])
print("CLI train-pretrain parse OK, func:", args.func.__name__)

args2 = p.parse_args(["train-pretrain", "--run-id", "run_fake_000", "--max-steps", "100"])
assert args2.max_steps == 100
print("CLI --max-steps override OK:", args2.max_steps)

print("\nALL Stage 5 smoke tests PASSED")
