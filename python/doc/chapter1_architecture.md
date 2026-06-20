# 📖 Chapter 1: The Architectural Blueprint

[📖 Table of Contents](training_journey.md) | [Next Chapter ➡️](chapter2_oom_crash.md)

---

## 🏗️ Model Specifications

To build a high-performance LLM tailored specifically for Rust code completion, we carefully selected a **decoder-only Transformer architecture**. This layout is structurally optimized to learn the formal, highly nested syntax of systems programming.

The architectural layout for our custom **1.6 Billion Parameter** model is structured as follows:

| Attribute | Specification | Rationale |
| :--- | :--- | :--- |
| **Parameter Count** | 1.6 Billion | Perfect balance between learning capacity and real-time local inference speeds. |
| **Number of Layers** | 24 | High depth to capture deep, complex syntactic relations and lifetime constraints in Rust. |
| **Attention Heads** | 16 | Allows the model to attend to multiple tokens in parallel (e.g. tracking types, scopes, and variables). |
| **Hidden Dimension** | 2048 | Provides ample capacity to represent dense code features and semantics. |
| **Context Window (Seq Len)** | 2048 tokens | Essential for holding large blocks of functional context, dependencies, and structs. |
| **Tokenizer** | `cl100k_base` | The vocabulary standard utilized by tiktoken, containing ~100k tokens. Extremely efficient for syntax-heavy keywords. |

---

## 🔬 Vocabulary & Tokenization Selection

A major factor in code-generation efficiency is the tokenization scheme. Selecting `cl100k_base` (the standard vocabulary used by GPT-4) was crucial:
*   **Compression**: Rather than splitting rare Rust constructs into dozens of single characters, `cl100k_base` compresses keywords (like `struct`, `impl`, `match`, `unwrap`) into single tokens.
*   **Efficiency**: Reduces the effective context window size consumed by boilerplate, leaving more space for functional code.

> [!NOTE]
> Training a model of this magnitude requires robust computational infrastructure. For our training grounds, we deployed this architecture on the high-performance **Mac M3 Ultra** equipped with 512 GB of unified memory.

---

[📖 Table of Contents](training_journey.md) | [Next Chapter ➡️](chapter2_oom_crash.md)
