# 🎨 Whiteboard Topic 07: RoPE Table Construction (COMPLETED) ✅

This topic covers how we precompute the global sine and cosine tables for Rotary Position Embeddings.

---

## ⚙️ Table Configuration
* **`head_dimension`:** `64` (32 column pairs, `num_pairs = 32`)
* **`max_seq_len`:** `5` (token positions `0` to `4`)
* **`base`:** `10000.0`

---

## 💻 Python Table Generator (User Completed)
The table is generated as a `[max_seq_len, head_dim // 2]` grid:

```python
import builtins
import numpy as np

head_dim = 64
max_seq_len = 5
base = 10000.0
num_pairs = head_dim // 2

cos_table = np.zeros((max_seq_len, num_pairs))
sin_table = np.zeros((max_seq_len, num_pairs))

for i in builtins.range(num_pairs):
    theta = 1.0 / (base ** (i * 2.0 / head_dim))
    for pos in builtins.range(max_seq_len):
        angle = pos * theta
        cos_table[pos][i] = np.cos(angle)
        sin_table[pos][i] = np.sin(angle)
```

---

## 🏆 Key Takeaways
1. **Dimension Alignment:** The row index of the table corresponds to the token's position in the sentence (`pos`). The column index corresponds to the internal feature pairs of the head (`i`).
2. **Weight Independence:** Because this table is completely independent of the model's weights and data, we build it once at startup and reuse it for all iterations.
