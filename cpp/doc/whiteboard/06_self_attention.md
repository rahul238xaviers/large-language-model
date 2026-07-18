# 🎨 Whiteboard Topic 06: Full Self-Attention Flow (COMPLETED) ✅

## 📝 Input
`[ tok1, tok2, tok3, tok4, tok5, tok6 ]`

## Step 1: Q, K, V Projections (Token-Local)
Each token independently computes its own Q, K, V.

## Step 2: Raw Scores (Cross-Token)
Every Q is dot-producted with every K → 6x6 score grid.

## Step 3: Softmax (Row-Wise)
Each row of scores is converted to probabilities summing to 1.0.

## Step 4: Weighted Sum with V
Each token's output = weighted mix of all V vectors using its attention probability row.

## 🏆 Key Insight
* Q and K are used to **decide who to attend to** (compute scores).
* V is the **actual content** that gets mixed together.
* RoPE encodes position inside Q and K before they interact cross-token.
