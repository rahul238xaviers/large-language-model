# 🎨 Active Whiteboard: Coding the RoPE Table Generator 🌈

Let's write a complete Python function that builds the cosine and sine tables exactly like the C++ model does.

---

## 🧮 The Formulas

### 1. Frequencies (`theta`)
For each column pair `i` (where `i` goes from `0` to `(head_dim / 2) - 1`):
* **Exponent:** `exponent = (2.0 * i) / head_dim`
* **Frequency:** `theta[i] = 1.0 / (base ^ exponent)`

### 2. Table Elements
For each position `pos` (from `0` to `max_seq_len - 1`) and each pair `i`:
* `cos_table[pos][i] = cos(pos * theta[i])`
* `sin_table[pos][i] = sin(pos * theta[i])`

---

## ⚙️ The Configuration
You will write a Python generator for the following production-grade configuration:
* **`head_dimension`:** `64` (so `head_dim // 2 = 32` column pairs)
* **`max_seq_len`:** `5` (positions `pos` go from `0` to `4`)
* **`base`:** `10000.0` (standard LLaMA/Transformer base constant)

Your output tables must have the shape `[5, 32]`.

---

## 💻 Python Coding Challenge

Write a Python function with the following signature:

```python
import numpy as np

def precompute_rope_tables(head_dim: int, max_seq_len: int, base: float = 10000.0):
    # 1. Initialize empty arrays for cos_table and sin_table of shape [max_seq_len, head_dim // 2]
    # 2. Compute theta for each pair i in range(head_dim // 2)
    # 3. For each position pos in range(max_seq_len), compute cos and sin
    # 4. Return cos_table, sin_table
    pass
```

---

## 🔍 Validation Checkpoints

To verify your function is 100% correct, run it and check if it yields the following values at these coordinates:

1. **At `pos = 3`, pair `i = 0`:** 
   What is the value of `cos_table[3][0]`?
2. **At `pos = 2`, pair `i = 1`:**
   What is the value of `sin_table[2][1]`?
3. **At `pos = 4`, pair `i = 16`:**
   What is the value of `cos_table[4][16]`?
