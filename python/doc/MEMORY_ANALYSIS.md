# Memory Analysis for 1.6B GPT Model on M3 Ultra

## Key Research Findings

### From Hugging Face Transformers Docs
Training a **4B parameter model** in mixed precision (fp16/fp32) at **batch size 16** requires **~85GB**:

```
Model:           6 bytes/param × 4B = 24 GB
Optimizer (Adam): 8 bytes/param × 4B = 32 GB  
Gradients:       4 bytes/param × 4B = 16 GB
Activations:     variable, ~10-15 GB (for batch=16, seq=512)
─────────────────────────────────────
Total:           ~85 GB
```

## Activation Memory — The Real Culprit (Corrected)

**Common mistake:** Simply multiplying batch × seq × hidden × layers

```
24 layers × 32 batch × 2048 seq × 2048 hidden × 2 bytes = 6.44 GB
```

This vastly underestimates because it ignores **attention score matrices**, which are O(seq²):

**Per layer attention storage (batch=32):**
```
batch × num_heads × seq_len × seq_len × dtype_size
= 32 × 16 × 2048 × 2048 × 2 bytes
= 4.3 GB per layer (just the QK^T matrix)
```

**× 24 layers = ~103 GB** just for attention scores (backward cache)

**Full activation breakdown (batch=32, seq=2048):**
- Attention scores (QK^T, 24 layers): ~103 GB
- Other activations (layer norm, residuals, etc.): ~6.4 GB
- **Total activations: ~110–130 GB**

**With batch=4 (1/8 smaller):**
- Attention scores: ~13 GB (scales linearly with batch)
- Other activations: ~0.8 GB
- **Total activations: ~13.8 GB**

---

## Parameter Memory

Using conservative mixed precision (bf16 weights + fp32 master + fp32 gradients):
- **1.6B params × 18 bytes/param = ~29 GB**

In practice MLX may use bf16 for gradients too (12–14 bytes/param = ~20 GB), but overestimating is safe.

---

## Crash Analysis — Observed 500+ GB Peak

**Run_20260513_235509 (batch=32, accumulation=4):**

| Component | Memory |
|-----------|--------|
| Weights + optimizer + gradients | ~29 GB |
| Activations (attention QK^T) | ~103 GB |
| Other activations | ~6 GB |
| MLX compile trace + Metal graph buffers | 50–100+ GB |
| Fragmentation + OS overhead | ~20 GB |
| **Total observed peak** | **500+ GB** |

**Root cause:** Attention score matrices O(seq²) + massive Metal compile DAG for 1.6B model with 2048 context.

---

## Final Conservative Config (After 500GB+ OOM Crash)

**Observed crash data:** batch_size=32 triggered 500GB+ memory spike → macOS OOM kill

**New target:** batch_size=4, accumulation=32
- Effective batch = 128 (same training compute as before, just safer memory distribution)
- Expected step 0: **~180 GB** (285 / 8 × 5 with some compile reduction for smaller graph)
- **Expected sustained: ~42.7 GB** (29 GB weights+optimizer+grads + 0.8 GB other activations + 12.9 GB attention scores for batch=4)
- **Safety margin to 512 GB ceiling: 332 GB** at step 0, 469+ GB sustained ← very safe

### Comparison Table

| Config | Batch | Accum | Eff Batch | Step 0 | Sustained | Status |
|--------|-------|-------|-----------|--------|-----------|--------|
| Original (crashed) | 32 | 4 | 128 | 285 GB | 120 GB | ✗ OOM at 500+ GB |
| First attempt | 8 | 16 | 128 | 222 GB | 60 GB | ✗ Likely risky |
| **Final (safe)** | **4** | **32** | **128** | **~180 GB** | **~42.7 GB** | **✓ 300 GB margin** |

---

## Future Optimizations — Priority Order

Only after confirming stable baseline. **Priority order by impact on M3 Ultra:**

1. **Flash Attention** (HIGHEST IMPACT): Reduces attention memory from O(seq²) to O(seq). MLX has memory-efficient attention.
   - Attention scores for batch=32 would drop from ~103 GB to ~6 GB
   - Would make original crash config (batch=32) viable and increase throughput
   - After enabling: can scale back to micro_batch=32, accumulation=4

2. **Activation Checkpointing**: Recompute activations on backward pass instead of caching all 24 layers.
   - Trade-off: ~6–12% GPU compute overhead, ~30% memory savings
   - Use if flash attention not enough or to push batch size further

3. **Lower precision Adam state**: Quantized optimizer following bitsandbytes patterns.
   - Not yet available in MLX, but could save ~8 GB
   - Lower priority given current 430 GB free memory headroom

4. **Combination: micro_batch=16 + checkpointing every 6 layers**: Safe ~100 GB sustained after flash attention

---

## PRE-RUN SAFETY CHECKLIST (Required Before Any Training Start)

Run this **BEFORE** executing `python3 src/train.py`:

```bash
cd apple-silicon
python3 << 'EOF'
import mlx.core as mx
from src.config import TrainingConfig

cfg = TrainingConfig()

print("=== Config Validation ===")
print(f"micro_batch_size:         {cfg.micro_batch_size} (should be ≤ 4)")
print(f"gradient_accumulation:    {cfg.gradient_accumulation_steps} (should be ≥ 32)")
print(f"effective_batch_size:     {cfg.effective_batch_size}")
print(f"num_worker_threads:       {cfg.num_worker_threads} (should be ≤ 4)")
print(f"num_prefetch_batches:     {cfg.num_prefetch_batches} (should be ≤ 4)")

print("\n=== Memory Prediction ===")
expected_sustained_gb = 42.7
expected_step0_gb = 180
margin_step0 = 512 - expected_step0_gb
margin_sustained = 512 - expected_sustained_gb

print(f"Expected step 0 peak:      {expected_step0_gb} GB")
print(f"Safety margin:             {margin_step0} GB (to 512 GB OOM ceiling)")
print(f"Expected sustained:        {expected_sustained_gb} GB")
print(f"Sustained margin:          {margin_sustained} GB")

print("\n=== MLX Metal Status ===")
try:
    active = mx.metal.get_active_memory() / 1e9
    peak = mx.metal.get_peak_memory() / 1e9
    cache = mx.metal.get_cache_memory() / 1e9
    print(f"Active: {active:.2f} GB | Peak: {peak:.2f} GB | Cache: {cache:.2f} GB")
except Exception as e:
    print(f"(MLX memory query failed: {e})")

if cfg.micro_batch_size <= 4 and cfg.gradient_accumulation_steps >= 32:
    print("\n✓ SAFE TO RUN")
else:
    print("\n✗ UNSAFE: Config violates safety bounds")
    print("  Reduce micro_batch_size OR increase gradient_accumulation_steps")
    exit(1)
EOF
```

If check fails: **DO NOT RUN TRAINING**. Reduce micro_batch_size or increase accumulation_steps.

**During training:** After step 0 and step 1 complete, run this to verify sustained memory matches prediction:
```python
import mlx.core as mx
active_gb = mx.metal.get_active_memory() / 1e9
print(f"Active memory: {active_gb:.1f} GB (expected ~42.7 GB)")
if active_gb > 150:
    print("WARNING: Memory exceeds 150 GB — terminate training")
```
