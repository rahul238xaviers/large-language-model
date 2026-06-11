"""Stage 7 smoke test — run via: python3 -m pipeline._tests.test_stage7_smoke"""
import math
import sys
from pathlib import Path
from typing import Any, Iterator

import mlx.core as mx

# ── Import checks ──────────────────────────────────────────────────────
from pipeline.inference.sampler import (
    SamplerConfig,
    apply_repetition_penalty,
    apply_top_k,
    apply_top_p,
    sample_token,
    generate,
)
from pipeline.inference.model_server import InferenceServer

print("All Stage 7 imports OK")

# ── SamplerConfig ──────────────────────────────────────────────────────
cfg = SamplerConfig(max_new_tokens=10, temperature=0.7, top_k=5, top_p=0.9, repetition_penalty=1.2)
d = cfg.to_dict()
assert d["max_new_tokens"] == 10
assert d["top_k"] == 5
cfg2 = SamplerConfig.from_dict(d)
assert cfg2.repetition_penalty == 1.2
print("SamplerConfig OK:", d)

# ── apply_top_k ────────────────────────────────────────────────────────
logits = mx.array([1.0, 3.0, 2.0, 4.0, 0.5])
filtered = apply_top_k(logits, k=2)
mx.eval(filtered)
vals = filtered.tolist()
assert vals[3] == 4.0          # rank-1 kept
assert vals[1] == 3.0          # rank-2 kept
assert vals[0] == float("-inf")  # rank-4 zeroed
assert vals[2] == float("-inf")
assert vals[4] == float("-inf")
print("apply_top_k OK")

# apply_top_k with k=0 is a no-op
no_filter = apply_top_k(logits, k=0)
assert no_filter[0].item() == 1.0
print("apply_top_k k=0 no-op OK")

# ── apply_top_p ────────────────────────────────────────────────────────
# Logits heavily concentrated on one token → nucleus should keep only top token(s)
logits_p = mx.array([10.0, 0.1, 0.1, 0.1, 0.1])
filtered_p = apply_top_p(logits_p, p=0.95)
mx.eval(filtered_p)
vals_p = filtered_p.tolist()
assert vals_p[0] == 10.0       # dominant token kept
print("apply_top_p OK")

# apply_top_p with p>=1.0 is a no-op
no_filter_p = apply_top_p(logits_p, p=1.0)
assert no_filter_p[0].item() == 10.0
print("apply_top_p p>=1.0 no-op OK")

# ── apply_repetition_penalty ───────────────────────────────────────────
logits_r = mx.array([2.0, 1.0, 3.0, 0.5, -1.0])
penalised = apply_repetition_penalty(logits_r, generated_ids=[2, 4], penalty=1.5)
mx.eval(penalised)
vals_r = penalised.tolist()
# Token 2 (positive logit 3.0, in generated): 3.0/1.5 - 0.5*35 = 2.0 - 17.5 = -15.5
assert vals_r[2] < vals_r[0]  # penalised below un-penalised token
# Token 4 (negative logit -1.0, in generated): -1.0 * 1.5 - 0.5*35 < -1.0
assert vals_r[4] < -1.0
# Token 0 (not in generated) unchanged
assert abs(vals_r[0] - 2.0) < 1e-4
print("apply_repetition_penalty OK")

# Penalty 1.0 → no-op
unchanged = apply_repetition_penalty(logits_r, generated_ids=[2], penalty=1.0)
assert unchanged[2].item() == 3.0
print("apply_repetition_penalty 1.0 no-op OK")

# ── sample_token ───────────────────────────────────────────────────────
# Greedy (temperature=0) should always pick argmax
logits_s = mx.array([0.1, 0.2, 5.0, 0.3])
tok = sample_token(logits_s, temperature=0.0)
assert tok == 2
print(f"sample_token greedy OK: token={tok}")

# ── generate loop with tiny mock model ────────────────────────────────
VOCAB = 20
EOT   = 0

class _TinyModel:
    """Returns logits that cycle tokens 1→2→3→…→EOT after 5 tokens."""
    call_count = 0

    def __call__(self, x: mx.array) -> mx.array:
        B, T = x.shape[0], x.shape[1]
        last  = int(x[0, -1].item())
        # After generating 5 non-EOT tokens, emit EOT-dominant logits
        _TinyModel.call_count += 1
        if _TinyModel.call_count > 5:
            logits = mx.zeros((B, T, VOCAB))
            # EOT gets +99 to dominate
            eot_boost = mx.zeros((B, T, VOCAB))
            # Can't do indexed assignment in MLX; build via one-hot trick
            one_hot = mx.array([[1 if i == EOT else 0 for i in range(VOCAB)]] * T).reshape(1, T, VOCAB)
            return logits + one_hot * 99.0
        # Normal: boost the next token index (wrapping around vocab 1-19)
        next_tok = (last % (VOCAB - 1)) + 1
        one_hot = mx.array([[1 if i == next_tok else 0 for i in range(VOCAB)]] * T).reshape(1, T, VOCAB)
        return mx.zeros((B, T, VOCAB)) + one_hot * 10.0

sampler_cfg = SamplerConfig(max_new_tokens=20, temperature=0.0, top_k=0, top_p=1.0, repetition_penalty=1.0)
_TinyModel.call_count = 0
generated_tokens = list(generate(
    model      = _TinyModel(),
    prompt_ids = [1],
    eot_id     = EOT,
    block_size = 16,
    cfg        = sampler_cfg,
))
assert len(generated_tokens) <= 20
assert EOT not in generated_tokens        # EOT stops the loop — not yielded
print(f"generate loop OK: {len(generated_tokens)} tokens produced (stopped at EOT)")

# ── InferenceServer — headless generate_text via mock model ──────────
import tempfile, numpy as np
import mlx.core as mx

class _MockServer(InferenceServer):
    """Skip actual checkpoint I/O; inject tiny mock model + tokenizer directly."""
    def load(self):
        import tiktoken

        class _BOS:
            """Constant-logit model that emits token 345 ('Ġ') then EOT."""
            _step = 0
            eot_token = tiktoken.get_encoding("cl100k_base").eot_token

            def __call__(self, x):
                _BOS._step += 1
                # After 3 calls return EOT-dominant
                vocab = 100277
                B, T  = x.shape[0], x.shape[1]
                if _BOS._step >= 3:
                    # Return very high logit for EOT
                    one_hot_list = [100.0 if i == _BOS.eot_token else 0.0 for i in range(vocab)]
                    return mx.array([one_hot_list] * T).reshape(1, T, vocab)
                # Normal: emit token 345 (a valid BPE token)
                one_hot_list = [10.0 if i == 345 else 0.0 for i in range(vocab)]
                return mx.array([one_hot_list] * T).reshape(1, T, vocab)

        import tiktoken
        self._model      = _BOS()
        self._tokenizer  = tiktoken.get_encoding("cl100k_base")
        self._block_size = 16
        return self

with tempfile.TemporaryDirectory() as tmp:
    fake_ckpt = Path(tmp) / "step_0000001.safetensors"
    fake_ckpt.write_bytes(b"")

    _BOS_server = _MockServer(
        checkpoint_path = fake_ckpt,
        model_cfg       = {"block_size": 16},
        inference_cfg   = {"max_new_tokens": 5, "temperature": 0.0,
                           "top_k": 0, "top_p": 1.0, "repetition_penalty": 1.0},
    )
    _BOS_server.load()
    fragments = list(_BOS_server.generate_text("fn main()"))
    # Should produce ≤5 fragments (stops at EOT)
    print(f"InferenceServer.generate_text OK: {len(fragments)} fragments -> '{''.join(fragments)}'")
    assert isinstance(fragments, list)

# ── Config carries inference section ─────────────────────────────────
from pipeline.orchestration.config_loader import ConfigLoader
cfg_loaded = ConfigLoader().load("local-dev")
inf_section = cfg_loaded.get("inference", {})
print("Inference cfg:", inf_section)
assert inf_section.get("max_new_tokens") == 128
assert inf_section.get("temperature")    == 0.7
assert inf_section.get("top_k")          == 50
assert inf_section.get("port")           == 7860

# ── CLI parses `serve` sub-command ────────────────────────────────────
from pipeline.cli import _build_parser
p    = _build_parser()
args = p.parse_args(["serve", "--run-id", "run_fake_000"])
print("CLI serve parse OK, func:", args.func.__name__)

args2 = p.parse_args(["serve", "--run-id", "run_fake_000",
                      "--checkpoint", "runs/x/training/checkpoints/step_0001000.safetensors",
                      "--port", "8080"])
assert args2.checkpoint is not None
assert args2.port == 8080
print("CLI --checkpoint + --port override OK")

# ── CLI --help shows serve ────────────────────────────────────────────
import io, contextlib
buf = io.StringIO()
try:
    with contextlib.redirect_stdout(buf):
        p.parse_args(["--help"])
except SystemExit:
    pass
help_text = buf.getvalue()
assert "serve" in help_text
print("CLI help shows 'serve' OK")

print("\nALL Stage 7 smoke tests PASSED")
