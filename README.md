# 🦀 Large Language Model from Scratch (Apple Silicon MLX)

This repository houses a custom **1.6B parameter decoder-only GPT model** optimized for Rust code completion, built and trained on Apple Silicon via MLX.

---

## 📖 The Training Journey Book

We documented our entire engineering path—from architectural layout and resolving 500GB+ OOM crashes on an M3 Ultra to GQA acceleration and implementing advanced decoding engines—in a comprehensive, book-style documentation structure inside the `doc` folder:

👉 **[Read the Full Rust-GPT LLM Training Journey Book (Chapter-by-Chapter)](doc/training_journey.md)**

*   **[Chapter 1: The Architectural Blueprint](doc/chapter1_architecture.md)**
*   **[Chapter 2: The M3 Ultra & The OOM Crash](doc/chapter2_oom_crash.md)**
*   **[Chapter 3: Stabilization & Memory Control](doc/chapter3_stabilization.md)**
*   **[Chapter 4: Hardware Optimization & Scaling](doc/chapter4_hardware_acceleration.md)**
*   **[Chapter 5: The Repetition Crisis & Decoding Engine](doc/chapter5_decoding_upgrades.md)**
*   **[Chapter 6: The Interactive Playground UI](doc/chapter6_developer_playground.md)**

---

## 🚀 Quick Setup & Playground Launch

### 1. Requirements & Setup
Ensure you are using **Python 3.10+** on Apple Silicon (M-series processor recommended) and install dependencies inside your virtual environment:

```bash
cd apple-silicon
python3 -m venv .apple_env
source .apple_env/bin/activate
pip install -r requirements.txt
```

### 2. Configure Checkpoint Location
Set the `CHECKPOINT_PATH` inside a `.env` file at the root of `apple-silicon/`:

```env
CHECKPOINT_PATH=runs/run_20260514_183932/checkpoints/step_001000.safetensors
```

### 3. Launch the Developer Playground
Launch the Gradio 6.0 playground workspace:

```bash
python3 tests/functional/gradio_app.py
```

Open `http://localhost:7860` in your web browser to generate Rust code completions in real time with built-in copy-to-clipboard fallbacks!

---

*Educational project inspired by Sebastian Raschka's "Large Language Models from Scratch" and fully scaled to 1.6 Billion parameters on Metal/Apple Silicon.*
