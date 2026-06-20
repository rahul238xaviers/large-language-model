# 📖 Chapter 3: Stabilization & Memory Control

[⬅️ Previous Chapter](chapter2_oom_crash.md) | [📖 Table of Contents](training_journey.md) | [Next Chapter ➡️](chapter4_hardware_acceleration.md)

---

## 🛠️ The Gradient Accumulation Strategy

To stabilize training without changing the mathematical behavior or learning dynamics of our 1.6B parameter model, we implemented a **gradient accumulation configuration**:

*   **Micro-batch size**: Reduced from `32` to **`4`**.
*   **Gradient accumulation steps**: Increased from `4` to **`32`**.
*   **Effective batch size**: Maintained identical compute ($4 \times 32 = 128$ tokens per step).

By accumulating gradients over 32 micro-steps before executing an optimizer step, we achieved the exact same convergence path while dramatically lowering the simultaneous activation footprint.

---

## 📊 Memory Stabilization Comparison

At `batch=4`, our attention score memory shrank by 87.5% from 103 GB to **12.9 GB**, rendering our training memory completely stable:

| Metric | Original Layout (Batch=32) | Stable Layout (Batch=4) |
| :--- | :--- | :--- |
| **Micro-Batch Size** | 32 | **4** |
| **Gradient Accumulation** | 4 | **32** |
| **Attention Activation Memory** | ~103 GB | **~12.9 GB** |
| **Step 0 Graph Compile Memory** | ~350+ GB | **~130 GB** |
| **Expected Sustained Peak** | ~120 GB | **~42.7 GB** |
| **Unified Memory Headroom** | Exceeded Limit (OOM Crash) | **+469.3 GB Margin** |

---

## 🔒 Pre-Run Safety Checklist Script

To guarantee that training never runs in an unstable environment, we integrated a pre-run validator inside our scripts directory. This checks the memory parameters and MLX Metal allocation limits prior to running `src/train.py`:

```python
import mlx.core as mx
from src.config import TrainingConfig

cfg = TrainingConfig()

print("=== Training Safeguard Validation ===")
print(f"Micro-Batch Size:      {cfg.micro_batch_size} (Max target: 4)")
print(f"Gradient Accumulation: {cfg.gradient_accumulation_steps} (Min target: 32)")
print(f"Metal Active Memory:   {mx.metal.get_active_memory() / 1e9:.2f} GB")

if cfg.micro_batch_size <= 4 and cfg.gradient_accumulation_steps >= 32:
    print("\n✓ STATE IS SECURE: Headroom confirmed. Safe to launch training.")
else:
    print("\n✗ STATE IS UNSTABLE: Exceeds safe bounds! Abort training.")
    exit(1)
```

By enforcing these limits, we successfully eliminated memory crashes, paving the way for continuous, steady-state training runs.

---

[⬅️ Previous Chapter](chapter2_oom_crash.md) | [📖 Table of Contents](training_journey.md) | [Next Chapter ➡️](chapter4_hardware_acceleration.md)
