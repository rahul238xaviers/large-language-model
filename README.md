# Large Language Model from Scratch

This project implements a Large Language Model (LLM) from scratch, based on the book **"Large Language Model from Scratch"** by **Sebastian Raschka**.

## Overview

The goal of this project is to demonstrate how to build a language model step by step, starting from the very basics. We will cover:

- Tokenization
- Building a vocabulary
- Implementing a Transformer architecture
- Training the model on a dataset
- Generating text using the trained model

This is an educational project aimed at understanding the core concepts behind modern language models like GPT.

## Requirements

To run this project, you need:

- Python 3.10+
- A virtual environment (recommended)
- Required Python packages listed in `requirements.txt`

## Setup Instructions

1. Clone this repository.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Download the dataset (`the-verdict.txt`) as described in the notebook.
2. Run the Jupyter notebook `create_large_language_mode.ipynb` step by step.
3. Each cell is accompanied by markdown explanations describing what the code does.

## Project Structure

- `create_large_language_mode.ipynb` – Main notebook with step-by-step implementation
- `requirements.txt` – Python dependencies
- `README.md` – This file

## Author

Based on the book **"Large Language Model from Scratch"** by **Sebastian Raschka**.

## License

See [LICENSE](LICENSE) for details.
