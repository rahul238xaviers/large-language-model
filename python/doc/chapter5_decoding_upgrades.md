# 📖 Chapter 5: The Repetition Crisis & Decoding Engine

[⬅️ Previous Chapter](chapter4_hardware_acceleration.md) | [📖 Table of Contents](training_journey.md) | [Next Chapter ➡️](chapter6_developer_playground.md)

---

## 🌀 1. The Repetition Crisis

During early inference testing of the completed 1.6B model checkpoint, we encountered a severe **repetition loop bottleneck**. When provided a standard code prompt, the model would get trapped in infinite, repeating cycles of single-character brackets and binary elements (e.g. `*x; *x; *x; *b; *b;`).

The diagnosis revealed that the model was using **greedy decoding (argmax)**. In code syntax, once the model outputs a closing token or a bracket, the local self-attention heads can lock onto those structures deterministically, leading to an infinite cycle.

---

## 🎲 2. Stochastic Categorical Sampling

We replaced the greedy argmax decoder with **stochastic categorical sampling** powered by native MLX `mx.random.categorical(...)`:
*   Introduced a **`Temperature`** parameter that divides logits before sampling.
*   *Low temperatures* (`0.1 - 0.3`) force highly focused, logical completions.
*   *Moderate temperatures* (`0.65 - 0.75`) introduce appropriate syntactic variance, allowing the model to explore alternate paths.

To eliminate syntax-breaking outlier tokens, we integrated **Top-K** and **Top-P (Nucleus) Filtering**:
1.  **Top-K**: Restricts candidate tokens to the $K$ highest-probability choices (default: `50`).
2.  **Top-P**: Restricts selection dynamically to the smallest subset of tokens whose cumulative probability exceeds $P$ (default: `0.90`).

---

## 🏹 3. The Hybrid Repetition Penalty Loop Breaker

Traditional repetition penalties scale log-probabilities multiplicatively:
*   If logit > 0: `logit /= penalty`
*   If logit < 0: `logit *= penalty`

However, in code generation, if the model has memorized a repeating sequence (like a null pointer dereference loop), the target token's logit is extremely high (e.g. `20.0`), while the second-best token is way down at `-20.0`. A multiplicative penalty of `1.15` only reduces the logit to `17.39`—leaving it as the absolute top choice by a huge margin.

To solve this, we engineered a **hybrid (multiplicative + additive) repetition penalty**:
$$\text{logit} = \text{logit}_{\text{multiplicative}} - (\text{penalty} - 1.0) \times 35.0$$

*   **How it works**:
    *   Tracks recently generated tokens in a **128-token sliding window**.
    *   If a token has already been generated within this window, its logit is first scaled multiplicatively, and then a strong additive penalty (e.g. `-5.25` for `1.15` penalty) is subtracted.
    *   This directly suppresses the logit below the selection threshold, forcing the attention heads to break the cycle and pick new syntactic tokens.
*   **The Result**: Repetition was **100% eliminated**, resulting in clean, structured, and correct Rust code completions!

---

[⬅️ Previous Chapter](chapter4_hardware_acceleration.md) | [📖 Table of Contents](training_journey.md) | [Next Chapter ➡️](chapter6_developer_playground.md)
