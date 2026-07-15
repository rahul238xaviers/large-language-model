# 🎨 Whiteboard Topic 03: Q, K, V Projections (COMPLETED)

This topic explains how we project the normalized token vector using matrix multiplication (dot products).

---

## 🔍 1. The Token-by-Token Projection
We take each token's normalized vector (size `H = 8`) and multiply it by the weight matrices (`Wq`, `Wk`, `Wv`).

```
[ Normalized Token x_norm ] (size 8)
       |
       +---> x_norm * Wq [8 x 8] ----> Query (q) vector  (size 8)
       +---> x_norm * Wk [8 x 4] ----> Key (k) vector    (size 4)
       +---> x_norm * Wv [8 x 4] ----> Value (v) vector  (size 4)
```

---

## 🧮 2. The Projection Dimensions
* **Query (`q`):** `head_dim * n_heads = 2 * 4 = 8`.
  * Outputs a vector of size **8**.
* **Key (`k`) & Value (`v`):** `head_dim * n_kv_heads = 2 * 2 = 4`.
  * Outputs a vector of size **4**.

---

## ✏️ Interactive Challenge (User Completed)
Using:
*   `x_norm = [ 0.333333, -0.333333, 1.0, -1.666667, 0.333333, -0.333333, 1.0, -1.666667 ]`
*   `Column 0 of Wq = [ 3.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0 ]`

### 1. Calculation:
*   `q[0] = (0.333333 * 3.0) + (-0.333333 * 3.0) + (1.0 * 2.0)`
*   `q[0] = 1.0 - 1.0 + 2.0 = 2.0`

*(User verified the dot-product implementation in Python using `(normalised_vector_array * weight_q).sum()`)*
