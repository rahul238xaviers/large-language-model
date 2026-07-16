# 🎨 Active Whiteboard: Full Self-Attention Flow (Visual) 🌈

---

## 📝 Input: Our 6-Token Sentence
```
[ tok1, tok2, tok3, tok4, tok5, tok6 ]
```
Each token has been normalized and projected. Now each token has its own Q, K, and V vector.

---

## 🔢 Step 1: Q, K, V Projections (Token-Local)
Each token independently computes its own Q, K, V:
```
tok1 → Q1, K1, V1
tok2 → Q2, K2, V2
tok3 → Q3, K3, V3
tok4 → Q4, K4, V4
tok5 → Q5, K5, V5
tok6 → Q6, K6, V6
```

---

## 🔀 Step 2: Raw Attention Scores (Cross-Token)
Every Q is dot-producted with every K → fills a 6x6 score grid:
```
             K1      K2      K3      K4      K5      K6
           +-------+-------+-------+-------+-------+-------+
Q1         |Q1·K1  |Q1·K2  |Q1·K3  |Q1·K4  |Q1·K5  |Q1·K6  |
           +-------+-------+-------+-------+-------+-------+
Q2         |Q2·K1  |Q2·K2  |Q2·K3  |Q2·K4  |Q2·K5  |Q2·K6  |
           +-------+-------+-------+-------+-------+-------+
Q3         |Q3·K1  |Q3·K2  |Q3·K3  |Q3·K4  |Q3·K5  |Q3·K6  |
           +-------+-------+-------+-------+-------+-------+
Q4         |Q4·K1  |Q4·K2  |Q4·K3  |Q4·K4  |Q4·K5  |Q4·K6  |
           +-------+-------+-------+-------+-------+-------+
Q5         |Q5·K1  |Q5·K2  |Q5·K3  |Q5·K4  |Q5·K5  |Q5·K6  |
           +-------+-------+-------+-------+-------+-------+
Q6         |Q6·K1  |Q6·K2  |Q6·K3  |Q6·K4  |Q6·K5  |Q6·K6  |
           +-------+-------+-------+-------+-------+-------+
```

---

## 📊 Step 3: Softmax (Row-Wise)
We apply softmax to EACH ROW independently.
Each row of 6 raw scores is converted into 6 probabilities that sum to 1.0:
```
             K1      K2      K3      K4      K5      K6      SUM
           +-------+-------+-------+-------+-------+-------+-----+
Q1         | 0.05  | 0.40  | 0.20  | 0.15  | 0.10  | 0.10  | 1.0 |
Q2         | 0.30  | 0.10  | 0.20  | 0.10  | 0.20  | 0.10  | 1.0 |
Q3         | 0.10  | 0.05  | 0.50  | 0.20  | 0.10  | 0.05  | 1.0 |
...
           +-------+-------+-------+-------+-------+-------+-----+
```

---

## 🎯 Step 4: Weighted Sum with V (Output Calculation)
For each token, we mix the V vectors using that token's row of attention weights:
```
output_tok3 = 0.10 * V1
            + 0.05 * V2
            + 0.50 * V3   ← tok3 pays most attention to itself here!
            + 0.20 * V4
            + 0.10 * V5
            + 0.05 * V6
```

The result is a new, context-enriched vector for tok3 that contains information from all other tokens it paid attention to.

---

## ✏️ Interactive Challenge!
Look at the softmax row for Q3 above: `[0.10, 0.05, 0.50, 0.20, 0.10, 0.05]`.

1. Which token does tok3 pay the **most** attention to?
2. If we want tok3's output to be heavily influenced by tok6, what would we need the weight for K6 to be: higher or lower than `0.05`?
