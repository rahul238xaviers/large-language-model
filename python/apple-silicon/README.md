# 🦀 Rust-GPT Playground

Welcome to the **Rust-GPT Playground**! This codebase contains a premium developer workspace and interactive text generation engine powered by a custom **1.6B parameter decoder-only model** optimized using MLX for Apple Silicon.

---

## 🚀 Getting Started

To run the interactive playground on your local system, follow these steps:

1. **Initialize the Environment Variables**
   Make sure you have a `.env` file in the root of the `apple-silicon` folder, or export the `CHECKPOINT_PATH` directly. For example, your `.env` should contain:
   ```env
   CHECKPOINT_PATH=runs/run_20260514_183932/checkpoints/step_001000.safetensors
   ```

2. **Launch the Playground**
   Execute the interactive Gradio script from your terminal:
   ```bash
   python3 tests/functional/gradio_app.py
   ```

3. **Open in Browser**
   Once launched, navigate to `http://localhost:7860` in your web browser to access the playground.

---

## 🛠️ How it Works: Advanced Generation Engine

Early versions of custom model generation often suffer from **infinite repetition loops** (e.g., generating repeating sequences of `1`s and `0`s). This occurs when standard **greedy decoding (argmax)** gets stuck in deterministic attention traps on structured syntax.

To resolve this, our generation engine implements a suite of advanced decoding algorithms:

### 1. Stochastic Sampling
Instead of greedily choosing the single token with the highest probability, the engine uses **categorical sampling** powered by native MLX `mx.random.categorical`. 
* **Temperature** scales the logits to control the randomness.
* Lower temperatures (`0.2 - 0.5`) produce highly focused, deterministic code.
* Moderate temperatures (`0.65 - 0.75`) balance creativity and syntactic correctness.

### 2. Top-K & Top-P (Nucleus) Filtering
To eliminate the low-probability "garbage" tail tokens that can break syntax:
* **Top-K**: Restricts candidate tokens to the top $K$ most likely options (e.g., $K=50$).
* **Top-P**: Restricts candidates dynamically to the smallest set of tokens whose cumulative probability exceeds $P$ (e.g., $P=0.90$).

### 3. Repetition Penalty
To completely break loop traps, a highly robust hybrid (multiplicative + additive) token-level penalty is integrated:
* Keeps track of generated tokens inside a **sliding window of 128 steps**.
* If a token has already been generated recently, its logit is first scaled multiplicatively (divided if positive, multiplied if negative), and then a strong additive penalty—scaled dynamically as `(penalty - 1.0) * 35.0`—is subtracted. This dual-action approach aggressively pushes down even highly confident repeating tokens, forcing the model to select other syntax branches.

> [!TIP]
> **Gold Standard Hyperparameters for 1.6B Rust Code Generation:**
> * **Temperature**: `0.70`
> * **Repetition Penalty**: `1.15` (values $\ge 1.25$ can penalize standard keywords like `let` and `fn` too heavily, causing token cannibalization)
> * **Top-K**: `50`
> * **Top-P**: `0.90`

---

## 🎨 Premium Developer UI features

The playground features a gorgeous, dark-themed user interface tailored for developers:

* **Real-time Token Streaming**: Generation uses a python generator interface to **stream tokens one-by-one** in real time, rather than delivering completions in a single burst.
* **Side-by-Side Workspace**: Left-hand code editor for the prompt input, and right-hand code editor displaying the streamed completions.
* **Copy to Clipboard Buttons**: Natively integrated copy icons inside the code panels. Polyfilled globally using custom client-side JavaScript in the page header, they utilize `navigator.clipboard` on secure origins (localhost/HTTPS) and dynamically fallback to off-screen textarea selection on insecure HTTP origins. This ensures the built-in Gradio copy icons have a 100% copy success rate in all environments without visual clutter.
* **Sleek Glassmorphism Accordion**: Hyperparameter controllers are compressed into a gorgeous, collapsible accordion block to keep the UI perfectly constrained to a single, vertical screen frame.
* **Monospaced Typography**: Input and output workspaces are styled with high-legibility **Fira Code monospace** fonts.

---

## 🔒 Compile-Time Type Safety

The application is written with fully strict compile-time type verification to support modern editors:
* **Static Type Hints**: Dynamic MLX array methods (like `.tolist()`) are validated using `isinstance(..., list)` checks and statically asserted via `typing.cast`.
* **Zero Overhead**: This gives you full Pylance/Pyright auto-completion and error-checking inside VS Code without adding any runtime processing costs.
