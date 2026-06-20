"""Stage 6 smoke test — run via: python3 -m pipeline._tests.test_stage6_smoke"""
import json
import math
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

# ── Stage 6 imports ───────────────────────────────────────────────────
from pipeline.evaluation.perplexity import EvalResult, compute_perplexity
from pipeline.evaluation.eval_runner import run_evaluation, _step_from_path

print("All Stage 6 imports OK")

# ── _step_from_path ───────────────────────────────────────────────────
assert _step_from_path(Path("step_0000500.safetensors")) == 500
assert _step_from_path(Path("step_0010000.safetensors")) == 10000
print("_step_from_path OK")

# ── EvalResult.to_dict ────────────────────────────────────────────────
er = EvalResult(n_sequences=10, n_tokens=200, mean_loss=3.5,
                perplexity=math.exp(3.5), eval_fraction=0.05)
d = er.to_dict()
assert d["n_sequences"] == 10
assert d["perplexity"] == round(math.exp(3.5), 4)
print("EvalResult.to_dict OK:", d)

# ── compute_perplexity with a tiny mock model ─────────────────────────
import mlx.core as mx
import mlx.nn as nn

class _TinyModel:
    """Minimal model that returns uniform logits (loss ≈ log(vocab_size))."""
    vocab_size = 50
    block_size = 16

    def __call__(self, x):
        B, T = x.shape[0], x.shape[1]
        # Uniform logits → loss = log(vocab_size)
        return mx.zeros((B, T, self.vocab_size))

    def eval(self):
        return self

with tempfile.TemporaryDirectory() as tmp:
    npy_path = Path(tmp) / "sequences.npy"
    # 40 sequences of length 16 with token IDs in [0, 49]
    arr = np.random.randint(0, 50, size=(40, 16), dtype=np.uint32)
    np.save(str(npy_path), arr)

    eval_cfg = {"eval_fraction": 0.25, "eval_batch_size": 4}
    result = compute_perplexity(_TinyModel(), npy_path, eval_cfg)

    print(f"compute_perplexity: n_seq={result.n_sequences}  n_tokens={result.n_tokens}  "
          f"loss={result.mean_loss:.4f}  ppl={result.perplexity:.2f}")
    assert result.n_sequences == 10                # 25% of 40
    assert result.n_tokens == 10 * 15             # (T-1) = 15 predicted tokens/seq
    # Uniform logits → loss = log(50) ≈ 3.912; allow ±0.1
    assert abs(result.mean_loss - math.log(50)) < 0.1, f"Unexpected loss: {result.mean_loss}"
    assert result.perplexity > 1.0

    # ── eval_runner end-to-end with patched checkpoint_loader ─────────
    with tempfile.TemporaryDirectory() as run_tmp:
        run_dir   = Path(run_tmp) / "run_test_000"
        run_dir.mkdir()
        # Fake checkpoint files
        ckpt_dir = run_dir / "training" / "checkpoints"
        ckpt_dir.mkdir(parents=True)
        ckpt500  = ckpt_dir / "step_0000500.safetensors"
        ckpt1000 = ckpt_dir / "step_0001000.safetensors"
        ckpt500.write_bytes(b"")
        ckpt1000.write_bytes(b"")

        tiny = _TinyModel()

        with patch("pipeline.evaluation.checkpoint_loader.load_checkpoint", return_value=tiny):
            results = run_evaluation(
                run_dir          = run_dir,
                sequences_path   = npy_path,
                model_cfg        = {"vocab_size": 50, "block_size": 16},
                eval_cfg         = eval_cfg,
                checkpoint_paths = [ckpt500, ckpt1000],
            )

        assert len(results) == 2
        assert results[0]["step"] == 500
        assert results[1]["step"] == 1000
        for r in results:
            report = run_dir / "evaluation" / f"eval_report_step_{r['step']:07d}.json"
            assert report.exists(), f"Missing report: {report}"
            loaded = json.loads(report.read_text())
            assert loaded["step"] == r["step"]

        history = json.loads((run_dir / "evaluation" / "eval_history.json").read_text())
        assert len(history) == 2
        print(f"eval_runner OK: {len(results)} reports written, history={len(history)} entries")

# ── Config carries eval section ───────────────────────────────────────
from pipeline.orchestration.config_loader import ConfigLoader
cfg = ConfigLoader().load("local-dev")
ev_cfg = cfg.get("eval", {})
print("Eval cfg:", ev_cfg)
assert ev_cfg.get("eval_fraction") == 0.05
assert ev_cfg.get("eval_batch_size") == 8

# ── CLI parses eval-checkpoint ─────────────────────────────────────────
from pipeline.cli import _build_parser
p = _build_parser()
args = p.parse_args(["eval-checkpoint", "--run-id", "run_fake_000"])
print("CLI eval-checkpoint parse OK, func:", args.func.__name__)

args2 = p.parse_args(["eval-checkpoint", "--run-id", "run_fake_000",
                       "--checkpoint", "runs/x/training/checkpoints/step_0001000.safetensors"])
assert args2.checkpoint is not None
print("CLI --checkpoint override OK:", Path(args2.checkpoint).name)

print("\nALL Stage 6 smoke tests PASSED")
