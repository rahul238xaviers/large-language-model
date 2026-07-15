# 🎨 Whiteboard Topic 02: RMSNorm (Root Mean Square Normalization) (COMPLETED)

This topic explains how RMSNorm standardizes the scale of each token vector individually.

---

## 🔍 1. The Goal of RMSNorm
RMSNorm forces the values of each token's vector to have a consistent scale (a "root mean square" of 1.0) before passing them forward. This stabilizes training.

RMSNorm is done **token-by-token** (independent of other tokens).

---

## 🧮 2. The Step-by-Step Math (For a Single Token Vector)
Let's normalize a single token vector `x` of size `H = 8`.

### The Token Vector `x`:
`x = [ +0.7, +0.1, -0.5, +0.2, -0.9, +0.3, +0.8, +0.4 ]`

### Step A: Square all the elements
`x_squared = [ 0.49, 0.01, 0.25, 0.04, 0.81, 0.09, 0.64, 0.16 ]`

### Step B: Compute the Mean of the squares (MS)
Sum of squares = `0.49 + 0.01 + 0.25 + 0.04 + 0.81 + 0.09 + 0.64 + 0.16 = 2.49`
Mean of squares = `2.49 / 8 = 0.31125`

### Step C: Compute the Root Mean Square (RMS)
Add a tiny constant epsilon (e.g. `1e-5`) to avoid division by zero:
`rms = sqrt(0.31125 + 1e-5) ≈ sqrt(0.31126) ≈ 0.5579`

### Step D: Divide the original vector by the RMS
We divide each element of `x` by `0.5579` to get the normalized vector `x_norm`:
```
x_norm = [
  +0.7 / 0.5579,   +0.1 / 0.5579,   -0.5 / 0.5579,  ...
]
```

### Step E: Scale by the learnable weight parameter `gamma`
Finally, we multiply each element by a learnable scaling parameter `gamma` (a vector of size 8):
```
output_i = x_norm_i * gamma_i
```

---

## ✏️ Interactive Challenge (User Completed)
Vector to normalize:
```
x = [ 1.0, -1.0, 3.0, -5.0, 1.0, -1.0, 3.0, -5.0 ]
```

### 1. RMS Value Calculation:
*   Sum of squares = `1 + 1 + 9 + 25 + 1 + 1 + 9 + 25 = 72.0`
*   Mean of squares = `72.0 / 8 = 9.0`
*   RMS value = `sqrt(9.0) = 3.0`

### 2. Normalized Vector (`x_norm`):
*   `x_norm = [ 0.333333, -0.333333, 1.0, -1.666667, 0.333333, -0.333333, 1.0, -1.666667 ]`
