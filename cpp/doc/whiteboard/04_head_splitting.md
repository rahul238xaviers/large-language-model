# 🎨 Whiteboard Topic 04: Splitting Projections into Attention Heads (COMPLETED)

This topic explains how we partition the large query projection vector into individual attention heads.

---

## ✂️ 1. Why do we split?
Multi-Head Attention works by having different "heads" focus on different parts of the embedding. 
* Total `query_projection_vector` size = `8`
* Number of Query Heads (`nH`) = `4`
* Head Dimension (`d`) = `8 / 4 = 2`

---

## 🗺️ 2. The Head Mapping Grid
We divide the 8 elements of the `query_projection_vector` into 4 sequential buckets:

| Element 0 | Element 1 | Element 2 | Element 3 | Element 4 | Element 5 | Element 6 | Element 7 |
|---|---|---|---|---|---|---|---|
| 🟧 🟧 | 🟧 🟧 | 🟪 🟪 | 🟪 🟪 | 🟫 🟫 | 🟫 🟫 | 🟨 🟨 | 🟨 🟨 |
| **Head 0** (🟧) | | **Head 1** (🟪) | | **Head 2** (🟫) | | **Head 3** (🟨) | |

* **Head 0 (Orange):** `[ query_projection_vector[0], query_projection_vector[1] ]`
* **Head 1 (Purple):** `[ query_projection_vector[2], query_projection_vector[3] ]`
* **Head 2 (Brown):** `[ query_projection_vector[4], query_projection_vector[5] ]`
* **Head 3 (Yellow):** `[ query_projection_vector[6], query_projection_vector[7] ]`

---

## ✏️ Interactive Challenge (User Completed)
Input `query_projection_vector = [ 2.0, -1.0, 4.5, 0.0, -3.2, 1.8, 0.5, 9.9 ]`

*   **Head 0 (🟧):** `[ 2.0, -1.0 ]`
*   **Head 1 (🟪):** `[ 4.5, 0.0 ]`
*   **Head 2 (🟫):** `[ -3.2, 1.8 ]`
*   **Head 3 (🟨):** `[ 0.5, 9.9 ]`
