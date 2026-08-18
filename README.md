<p align="center">
  <br>
  <img src="https://img.shields.io/badge/Large_Language_Model-CRAB-E34F26?style=for-the-badge&logo=rust&logoColor=white" alt="LLM Logo" />
  <img src="https://img.shields.io/badge/Apple_Silicon-METAL-000000?style=for-the-badge&logo=apple&logoColor=white" alt="Metal Logo" />
  <br><br>
  <b>×</b>
  &nbsp;&nbsp;
  <i>Building an LLM from scratch natively on Apple Silicon.</i>
  &nbsp;&nbsp;
  <b>×</b>
  <br><br>
</p>

### `├─[ PIPELINES ]────────────────────────────────────────────`

This repository contains two pipelines for training a Large Language Model from scratch: a native **C++ Training Engine** and an **Apple Silicon MLX (Python) Pipeline**.

### `├─[ C++_NATIVE_ENGINE ]────────────────────────────────────`

Contains the native C++ implementation of Parquet data ingestion, BPE tokenization, and model training.

![C++](https://img.shields.io/badge/C++-00599C?style=for-the-badge&logo=c%2B%2B&logoColor=white)
![Metal](https://img.shields.io/badge/Metal-000000?style=for-the-badge&logo=apple&logoColor=white)

*   👉 **[C4 Architecture & Development Sprint Plan](cpp/doc/c4_sprint_plan.md)**
*   👉 **[System Container Architecture](cpp/doc/architecture.md)**

<br>

### `├─[ PYTHON_MLX_PIPELINE ]──────────────────────────────────`

Contains the Python-based 1.6B parameter decoder-only GPT model optimization journey on Apple Silicon.

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)

👉 **[Read the Full Python MLX Training Journey Book](python/doc/training_journey.md)**

*   **[Chapter 1: The Architectural Blueprint](python/doc/chapter1_architecture.md)**
*   **[Chapter 2: The M3 Ultra & The OOM Crash](python/doc/chapter2_oom_crash.md)**
*   **[Chapter 3: Stabilization & Memory Control](python/doc/chapter3_stabilization.md)**
*   **[Chapter 4: Hardware Optimization & Scaling](python/doc/chapter4_hardware_acceleration.md)**
*   **[Chapter 5: The Repetition Crisis & Decoding Engine](python/doc/chapter5_decoding_upgrades.md)**
*   **[Chapter 6: The Interactive Playground UI](python/doc/chapter6_developer_playground.md)**
*   **[Memory Analysis & Safety Checklist](python/doc/MEMORY_ANALYSIS.md)**

<br>

### `└─[ QUICK_SETUP ]──────────────────────────────────────────`

#### 1. Requirements & Setup
Ensure you are using **Python 3.10+** and install dependencies using **uv** inside your virtual environment:

```bash
uv venv .venv
source .venv/bin/activate
uv pip install -r requirements/requirements.txt
```

#### 2. Configure Checkpoint Location
Set the `CHECKPOINT_PATH` inside a `.env` file at the root of `apple-silicon/`:

```env
CHECKPOINT_PATH=runs/run_20260514_183932/checkpoints/step_001000.safetensors
```

#### 3. Launch the Developer Playground
Launch the Gradio 6.0 playground workspace:

```bash
python3 tests/functional/gradio_app.py
```

Open `http://localhost:7860` in your web browser to generate Rust code completions in real time with built-in copy-to-clipboard fallbacks!

---

<p align="center">
  <i>Educational project inspired by Sebastian Raschka's "Large Language Models from Scratch" and fully scaled to 1.6 Billion parameters on Metal/Apple Silicon.</i>
</p>
