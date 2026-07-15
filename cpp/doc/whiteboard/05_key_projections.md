# 🎨 Whiteboard Topic 05: Key Projections with Split Groups (COMPLETED)

This topic explains how we multiply the normalized token vector by the key weight matrix to get the Key Projection vector, and how its columns are grouped.

---

## 🗺️ 1. The Group Structure
* **`token_normalized_vector`:** size `hidden_dimension = 8`.
* **`key_weight_matrix`:**
  * Rows = `hidden_dimension = 8`.
  * Columns = 2 groups, with 2 columns in each group (total of 4 columns).

---

## ✏️ Interactive Practice (User Completed)
Input `token_normalized_vector = [ 1.0, 0.0, 2.0, -1.0, 0.0, 0.0, 0.0, 0.0 ]`

*   **Column 0:** `[ 3.0, 1.0, 2.0, 1.0, 0.0, 0.0, 0.0, 0.0 ]`
*   **Column 1:** `[ 1.0, -1.0, 0.0, 4.0, 0.0, 0.0, 0.0, 0.0 ]`

### 1. Outputs for Group 0 (Head 0):
*   `k[0] = 6.0`
*   `k[1] = -3.0`

*(User verified the calculations in Python using vector dot-products)*
