# 🎨 Active Whiteboard: Topic 08 — Causal Masking 🌈

Now that we have computed our Q, K, and V vectors and applied RoPE, we are ready to calculate how tokens talk to each other. 

But before we do the raw multiplications, we must enforce a strict rule of language: **tokens cannot look into the future.**

---

## 🚫 1. The Rule: No Peeking Ahead
If the model is trying to predict the word after *"The cat sat on the"*, it is currently looking at the word **"the"** (position 4). 

It is allowed to look at *"The"* (0), *"cat"* (1), *"sat"* (2), *"on"* (3), and *"the"* (4). 
But it must **never** be allowed to look at *"bank"* (position 5), because that is the very word it is trying to predict!

---

## 🗺️ 2. The Score Grid Mask
To enforce this, we apply a mask to our `6 x 6` attention score grid. 

For any cell `[Query token i][Key token j]`:
* If `j > i` (future token): We set the score to **`-INFINITY`** (negative infinity).
* If `j <= i` (past or current token): We keep the original calculated score.

Here is the masked grid:
```
                K_tok0    K_tok1    K_tok2    K_tok3    K_tok4    K_tok5
Q_tok0 (Pos 0)   Keep     -INF      -INF      -INF      -INF      -INF
Q_tok1 (Pos 1)   Keep      Keep     -INF      -INF      -INF      -INF
Q_tok2 (Pos 2)   Keep      Keep      Keep     -INF      -INF      -INF
Q_tok3 (Pos 3)   Keep      Keep      Keep      Keep     -INF      -INF
Q_tok4 (Pos 4)   Keep      Keep      Keep      Keep      Keep     -INF
Q_tok5 (Pos 5)   Keep      Keep      Keep      Keep      Keep      Keep
```

---

## 🧮 3. Why Negative Infinity (`-INFINITY`)?
We use `-INFINITY` because of how the **Softmax** step behaves:

* The formula uses exponentiation: `e^score`.
* Mathematically, `e^(-INFINITY) = 0.0`.

By setting the future scores to `-INFINITY`, we guarantee that their attention probability (weight) becomes exactly **`0.0`** after Softmax. The model is forced to completely ignore them.

---

## ✏️ Interactive Challenge!
Suppose we have a sequence of 4 tokens: `[tok0, tok1, tok2, tok3]`.

Look at **Row 1** (representing Query token 1, `Q_tok1`):
1. Which column keys are allowed to be looked at (kept)?
2. Which column keys are blocked (set to `-INFINITY`)?
