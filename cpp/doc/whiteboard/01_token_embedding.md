# 🎨 Whiteboard Topic 01: The Token Embedding Color-Map 🌈 (COMPLETED)

This topic explains how the input shape transforms from **2D** to **3D**.

---

## 🟢 Color Key
Each Token ID is assigned a unique color block:
* **ID 9** = 🟢 Green
* **ID 12** = 🟡 Yellow
* **ID 3** = 🔵 Blue
* **ID 2** = 🔴 Red

---

## 🔢 Step 1: The 2D Input Grid [Batch=4, SeqLen=4]
Each cell holds a single token ID:

| | Token 0 | Token 1 | Token 2 | Token 3 |
|---|---|---|---|---|
| **Sentence 0** | 🟢 (ID 9) | 🟡 (ID 12) | 🔵 (ID 3) | 🟢 (ID 9) |
| **Sentence 1** | 🔴 (ID 2) | 🔵 (ID 3) | 🔴 (ID 2) | 🟡 (ID 12) |
| **Sentence 2** | 🟡 (ID 12) | 🟢 (ID 9) | 🟡 (ID 12) | 🔵 (ID 3) |
| **Sentence 3** | 🔵 (ID 3) | 🔴 (ID 2) | 🟢 (ID 9) | 🔴 (ID 2) |

---

## 📖 Step 2: The Embedding Lookup Table (Vocab x H)
The table holds a vector of 8 floats for each color:

* 🟢 **Row 9:** `[ +0.7, +0.1, -0.5, +0.2, -0.9, +0.3, +0.8, +0.4 ]`
* 🟡 **Row 12:** `[ +0.2, +0.5, +0.9, -0.1, +0.3, -0.7, +0.0, +0.1 ]`
* 🔵 **Row 3:** `[ +0.1, -0.4, +0.3, +0.8, -0.2, +0.5, -0.1, +0.9 ]`
* 🔴 **Row 2:** `[ -0.3, +0.6, -0.1, -0.7, +0.8, -0.2, +0.4, +0.5 ]`

---

## 🔄 Step 3: Substitution (From 1D value to 8D Vector)
Each single cell from **Step 1** is replaced by the corresponding row from **Step 2**.

For **Sentence 0**, the substitution looks like this:

| Token 0 (🟢) | Token 1 (🟡) | Token 2 (🔵) | Token 3 (🟢) |
|---|---|---|---|
| `[ +0.7, +0.1, -0.5, +0.2, -0.9, +0.3, +0.8, +0.4 ]` | `[ +0.2, +0.5, +0.9, -0.1, +0.3, -0.7, +0.0, +0.1 ]` | `[ +0.1, -0.4, +0.3, +0.8, -0.2, +0.5, -0.1, +0.9 ]` | `[ +0.7, +0.1, -0.5, +0.2, -0.9, +0.3, +0.8, +0.4 ]` |

---

## 🧊 Step 4: The Final 3D Tensor [Batch=4, SeqLen=4, HiddenDim=8]
By performing this substitution for all sentences, our 2D grid becomes a 3D block. 

### Sentence 0
| | Float 0 | Float 1 | Float 2 | Float 3 | Float 4 | Float 5 | Float 6 | Float 7 |
|---|---|---|---|---|---|---|---|---|
| **Token 0 (🟢)** | 🟢 +0.7 | 🟢 +0.1 | 🟢 -0.5 | 🟢 +0.2 | 🟢 -0.9 | 🟢 +0.3 | 🟢 +0.8 | 🟢 +0.4 |
| **Token 1 (🟡)** | 🟡 +0.2 | 🟡 +0.5 | 🟡 +0.9 | 🟡 -0.1 | 🟡 +0.3 | 🟡 -0.7 | 🟡 +0.0 | 🟡 +0.1 |
| **Token 2 (🔵)** | 🔵 +0.1 | 🔵 -0.4 | 🔵 +0.3 | 🔵 +0.8 | 🔵 -0.2 | 🔵 +0.5 | 🔵 -0.1 | 🔵 +0.9 |
| **Token 3 (🟢)** | 🟢 +0.7 | 🟢 +0.1 | 🟢 -0.5 | 🟢 +0.2 | 🟢 -0.9 | 🟢 +0.3 | 🟢 +0.8 | 🟢 +0.4 |

### Sentence 1 (User Corrected)
| | Float 0 | Float 1 | Float 2 | Float 3 | Float 4 | Float 5 | Float 6 | Float 7 |
|---|---|---|---|---|---|---|---|---|
| **Token 0 (🔴)** | 🔴 -0.3 | 🔴 +0.6 | 🔴 -0.1 | 🔴 -0.7 | 🔴 +0.8 | 🔴 -0.2 | 🔴 +0.4 | 🔴 +0.5 |
| **Token 1 (🔵)** | 🔵 +0.1 | 🔵 -0.4 | 🔵 +0.3 | 🔵 +0.8 | 🔵 -0.2 | 🔵 +0.5 | 🔵 -0.1 | 🔵 +0.9 |
| **Token 2 (🔴)** | 🔴 -0.3 | 🔴 +0.6 | 🔴 -0.1 | 🔴 -0.7 | 🔴 +0.8 | 🔴 -0.2 | 🔴 +0.4 | 🔴 +0.5 |
| **Token 3 (🟡)** | 🟡 +0.2 | 🟡 +0.5 | 🟡 +0.9 | 🟡 -0.1 | 🟡 +0.3 | 🟡 -0.7 | 🟡 +0.0 | 🟡 +0.1 |
