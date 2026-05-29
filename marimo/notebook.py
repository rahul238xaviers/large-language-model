import marimo

__generated_with = "0.23.6"
app = marimo.App(width="medium")


@app.cell
def _():
    # Import necessary libraries
    from datasets import load_dataset
    import pandas as pd
    import polars as pl
    import altair as alt
    import datasets
    import os
    from dotenv import load_dotenv
    from huggingface_hub import HfFileSystem, hf_hub_download, login
    from datasets import logging
    from gemma import gm
    import marimo as mo

    return (
        datasets,
        gm,
        hf_hub_download,
        load_dataset,
        load_dotenv,
        logging,
        login,
        mo,
        os,
    )


@app.cell
def _(datasets, load_dotenv, os):
    # This automatically finds your local .env file and loads HF_TOKEN into the system environment
    load_dotenv()
    has_token = "HF_TOKEN" in os.environ

    # 1. Choose a high-capacity local directory for your raw training data
    local_storage_base = "/Users/rahulkumar/dev/large-language-model/data/datasets"
    os.makedirs(local_storage_base, exist_ok=True)
    datasets.utils.logging.set_verbosity_info()
    return (local_storage_base,)


@app.cell
def _(load_dataset, local_storage_base, os):
    print("--- Starting Local Downloads ---")

    # 3. Pull down and build the Python file cache locally
    print("Downloading Python Dataset (21.5 GB compressed)...")
    local_python_dataset = load_dataset(
        "bigcode/starcoderdata",
        data_dir="python",
        split="train",
        cache_dir=os.path.join(local_storage_base, "starcoder_python_cache"),
    )
    return (local_python_dataset,)


@app.cell
def _(load_dataset, local_storage_base, logging, os):
    print("Downloading English NLP Dataset (28.5 GB compressed)...")
    local_english_dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        cache_dir=os.path.join(local_storage_base, "fineweb_english_cache"),
    )
    logging.set_verbosity_info()
    return (local_english_dataset,)


@app.cell
def _(gm):
    # 1. Import your specified Gemma 4 Tokenizer
    try:
        from gemma.gm.text import Gemma4Tokenizer

        print("Successfully imported Gemma4Tokenizer from gemma.gm.text!")
    except ImportError:
        try:
            Gemma4Tokenizer = gm.text.Gemma4Tokenizer
            print("Successfully imported Gemma4Tokenizer via gemma.gm!")
        except ImportError as e:
            print("Error importing Gemma4Tokenizer.")
            print(
                "Please ensure your environment has the 'gemma' library installed."
            )
            raise e
    return (Gemma4Tokenizer,)


@app.cell
def _(local_storage_base, os):
    python_cache_dir = os.path.join(local_storage_base, "starcoder_python_cache")
    print(f"\n[Step 1] Loading local Python dataset from: {python_cache_dir}")
    return


@app.cell
def _(Gemma4Tokenizer, hf_hub_download, login, os):
    # Go to the URL https://huggingface.co/google/gemma-2-9b and accept the license terms after reading and understanding it.
    # Once you acknowledge and accept the below download will work.
    print("\nInitializing gm.text.Gemma4Tokenizer...")
    tokenizer_model_path = "google/gemma-2-9b"
    print("Successfully initialized Gemma4Tokenizer directly!")
    hf_token = os.environ.get("HF_TOKEN")
    login()
    # Automatically download/locate the SentencePiece model file locally
    try:
        tokenizer_model_path = hf_hub_download(
            repo_id=tokenizer_model_path,
            token=hf_token,
            filename="tokenizer.model",
            local_files_only=False,  # Allows downloading the single small file if not cached
        )
        print(
            f"Resolved tokenizer.model file to local path: {tokenizer_model_path}"
        )
    except Exception as e:
        print(
            f"Failed to automatically resolve tokenizer.model from '{tokenizer_model_path}': {e}"
        )
        print(
            "Please make sure you have authenticated via HF_TOKEN or have the 'tokenizer.model' file downloaded locally."
        )

    tokenizer = Gemma4Tokenizer(tokenizer_model_path)
    return (tokenizer,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Imports and Setup
    """)
    return


@app.cell
def _():
    from collections import Counter
    from tqdm import tqdm
    from multiprocessing import Pool
    import numpy as np
    import hashlib
    import psutil

    print("✅ Imports loaded")
    print(f"CPU Cores: {psutil.cpu_count()}")
    print(f"RAM Available: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    return Counter, hashlib, np, tqdm


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Define Analysis Function
    """)
    return


@app.cell
def _(hashlib, tokenizer):
    def analyze_single_batch(batch_data):
        """Analyze a single batch and return all metrics."""
        texts, batch_id = batch_data

        batch_metrics = {
            'tokens': [],
            'lengths': [],
            'code_score': 0,
            'quality_score': 0,
            'hashes': []
        }

        for text in texts:
            if not text:
                continue

            # Tokenize
            tokens = tokenizer.encode(text)
            batch_metrics['tokens'].extend(tokens)
            batch_metrics['lengths'].append(len(tokens))

            # Code detection
            code_indicators = ['def ', 'import ', 'class ', 'function(', 'if __name__', 'print(']
            is_code = sum(1 for ind in code_indicators if ind in text[:500]) >= 2
            batch_metrics['code_score'] += (1 if is_code else 0)

            # Quality score
            quality = 0
            quality += min(len(text) / 2000, 0.5)
            quality += 0.3 if text.count('\n') > 10 else 0
            quality += 0.2 if any(p in text for p in '.,;:!?') else 0
            batch_metrics['quality_score'] += quality

            # Hash for duplicates
            text_hash = hashlib.md5(text[:1000].encode()).hexdigest()
            batch_metrics['hashes'].append(text_hash)

        return batch_metrics

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Main Analysis Function
    """)
    return


@app.cell
def _(Counter, hashlib, time, tokenizer, tqdm):
    def analyze_dataset(dataset, name, text_field, sample_size=100000, batch_size=1000):
        """Simple non-parallel analysis for marimo."""

        print(f"\n{'='*60}")
        print(f"📊 ANALYZING: {name}")
        print(f"{'='*60}")

        if sample_size < len(dataset):
            print(f"Taking {sample_size:,} documents (from {len(dataset):,} total)...")
            dataset = dataset.select(range(sample_size))

        print(f"Processing {len(dataset):,} documents...")
        start_time = time.time()

        all_tokens = []
        all_lengths = []
        code_count = 0
        total_quality = 0
        all_hashes = []

        for i in tqdm(range(0, len(dataset), batch_size), desc="Processing"):
            batch = dataset[i:i+batch_size]
            texts = batch[text_field]

            for text in texts:
                if not text:
                    continue

                tokens = tokenizer.encode(text)
                all_tokens.extend(tokens)
                all_lengths.append(len(tokens))

                # Code detection
                code_indicators = ['def ', 'import ', 'class ', 'function(', 'if __name__']
                is_code = sum(1 for ind in code_indicators if ind in text[:500]) >= 2
                code_count += (1 if is_code else 0)

                # Quality score
                quality = min(len(text) / 2000, 0.5)
                quality += 0.3 if text.count('\n') > 10 else 0
                quality += 0.2 if any(p in text for p in '.,;:!?') else 0
                total_quality += quality

                # Hash for duplicates
                text_hash = hashlib.md5(text[:1000].encode()).hexdigest()
                all_hashes.append(text_hash)

        elapsed = time.time() - start_time

        return {
            'name': name,
            'sample_size': len(dataset),
            'total_tokens': len(all_tokens),
            'unique_tokens': len(set(all_tokens)),
            'lengths': all_lengths,
            'code_count': code_count,
            'avg_quality': total_quality / len(dataset),
            'duplicate_rate': 1 - (len(set(all_hashes)) / len(dataset)),
            'token_counts': Counter(all_tokens),
            'time_minutes': elapsed / 60
        }

    return (analyze_dataset,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run Python Dataset Analysis
    """)
    return


@app.cell
def _(analyze_dataset, local_python_dataset):
    print("🔍 Analyzing Python dataset...")
    python_metrics = analyze_dataset(
        local_python_dataset, 
        "PYTHON DATASET", 
        "content", 
        sample_size=1000000,  # Adjust as needed
        batch_size=2000
    )

    print(f"\n✅ Python analysis complete in {python_metrics['time_minutes']:.1f} minutes")
    return (python_metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Run English Dataset Analysis
    """)
    return


@app.cell
def _(analyze_dataset, local_english_dataset):
    print("🔍 Analyzing English dataset...")
    english_metrics = analyze_dataset(
        local_english_dataset, 
        "ENGLISH DATASET", 
        "text", 
        sample_size=1000000,  # Adjust as needed
        batch_size=2000
    )

    print(f"\n✅ English analysis complete in {english_metrics['time_minutes']:.1f} minutes")
    return (english_metrics,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Display Python Dataset Results
    """)
    return


@app.cell
def _(np, python_metrics, tokenizer):
    print(f"\n{'='*60}")
    print(f"📈 PYTHON DATASET RESULTS")
    print(f"{'='*60}")

    lengths = python_metrics['lengths']
    code_ratio = python_metrics['code_count'] / python_metrics['sample_size']
    vocab_coverage = (python_metrics['unique_tokens'] / tokenizer.vocab_size) * 100

    print(f"\n📄 Documents: {python_metrics['sample_size']:,}")
    print(f"🔤 Total tokens: {python_metrics['total_tokens']:,}")
    print(f"✨ Unique tokens: {python_metrics['unique_tokens']:,}")
    print(f"📊 Vocab coverage: {vocab_coverage:.1f}%")
    print(f"💻 Code ratio: {code_ratio:.1%}")
    print(f"⭐ Quality score: {python_metrics['avg_quality']:.2f}/1.00")
    print(f"🔄 Duplicate rate: {python_metrics['duplicate_rate']:.1%}")

    print(f"\n📏 Document Lengths:")
    print(f"   Min: {min(lengths):,}")
    print(f"   Max: {max(lengths):,}")
    print(f"   Mean: {np.mean(lengths):.1f}")
    print(f"   Median: {np.median(lengths):.1f}")
    print(f"   95th percentile: {np.percentile(lengths, 95):.0f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Display English Dataset Results
    """)
    return


@app.cell
def _(english_metrics, np, tokenizer):
    print(f"\n{'='*60}")
    print(f"📈 ENGLISH DATASET RESULTS")
    print(f"{'='*60}")

    eng_lengths = english_metrics['lengths']
    eng_code_ratio = english_metrics['code_count'] / english_metrics['sample_size']
    eng_vocab_coverage = (english_metrics['unique_tokens'] / tokenizer.vocab_size) * 100

    print(f"\n📄 Documents: {english_metrics['sample_size']:,}")
    print(f"🔤 Total tokens: {english_metrics['total_tokens']:,}")
    print(f"✨ Unique tokens: {english_metrics['unique_tokens']:,}")
    print(f"📊 Vocab coverage: {eng_vocab_coverage:.1f}%")
    print(f"💻 Code ratio: {eng_code_ratio:.1%}")
    print(f"⭐ Quality score: {english_metrics['avg_quality']:.2f}/1.00")
    print(f"🔄 Duplicate rate: {english_metrics['duplicate_rate']:.1%}")

    print(f"\n📏 Document Lengths:")
    print(f"   Min: {min(eng_lengths):,}")
    print(f"   Max: {max(eng_lengths):,}")
    print(f"   Mean: {np.mean(eng_lengths):.1f}")
    print(f"   Median: {np.median(eng_lengths):.1f}")
    print(f"   95th percentile: {np.percentile(eng_lengths, 95):.0f}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Comparison & Recommendations
    """)
    return


@app.cell
def _(english_metrics, np, python_metrics, tokenizer):
    print(f"\n{'='*60}")
    print(f"🎯 PRETRAINING RECOMMENDATIONS")
    print(f"{'='*60}")

    # Calculate recommendations
    python_code_ratio = python_metrics['code_count'] / python_metrics['sample_size']
    english_code_ratio = english_metrics['code_count'] / english_metrics['sample_size']

    # Max sequence length
    max_len = max(np.percentile(python_metrics['lengths'], 95), 
                  np.percentile(english_metrics['lengths'], 95))

    # Combined vocabulary
    combined_vocab = set(python_metrics['token_counts'].keys()) | set(english_metrics['token_counts'].keys())

    print(f"\n📚 Combined unique tokens: {len(combined_vocab):,}")
    print(f"   ({len(combined_vocab)/tokenizer.vocab_size:.1%} of tokenizer capacity)")

    print(f"\n🔧 Training Configuration:")
    print(f"   max_length = {int(max_len)}")
    print(f"   Estimated memory per sample: {max_len * 4 / 1e6:.1f} MB")

    print(f"\n⚖️ Mixing Strategy:")
    print(f"   Option 1 (Balanced): 50% Python + 50% English")
    print(f"   Option 2 (Code-focused): 70% Python + 30% English")
    print(f"   Option 3 (Language-focused): 30% Python + 70% English")

    # Warnings
    if python_metrics['duplicate_rate'] > 0.1:
        print(f"\n⚠️ WARNING: Python dataset has {python_metrics['duplicate_rate']:.1%} duplicates!")
    if english_metrics['duplicate_rate'] > 0.1:
        print(f"\n⚠️ WARNING: English dataset has {english_metrics['duplicate_rate']:.1%} duplicates!")
    if python_metrics['avg_quality'] < 0.5:
        print(f"\n⚠️ WARNING: Python dataset quality is low ({python_metrics['avg_quality']:.2f})!")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Top Tokens Visualization
    """)
    return


@app.cell
def _(english_metrics, python_metrics, tokenizer):
    # Display most common tokens
    print("\n🔝 TOP 20 TOKENS IN PYTHON DATASET:")
    python_top = python_metrics['token_counts'].most_common(20)
    for i, (token_id, count) in enumerate(python_top, 1):
        token_str = tokenizer.decode([token_id])[:30]
        print(f"  {i:2d}. {token_str:30s} - {count:,} times")

    print("\n🔝 TOP 20 TOKENS IN ENGLISH DATASET:")
    english_top = english_metrics['token_counts'].most_common(20)
    for i, (token_id, count) in enumerate(english_top, 1):
        token_str = tokenizer.decode([token_id])[:30]
        print(f"  {i:2d}. {token_str:30s} - {count:,} times")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 📚 Data-Efficient Pretraining Strategy (Based on arXiv:2402.09668)

    ### The Problem
    Training LLMs on 100% of available data is expensive and inefficient. The paper "How to Train Data-Efficient LLMs" proves we can achieve **better results with 70-90% less data** by intelligently selecting which documents to train on.

    ### Our Analysis Results (1M Documents Each)

    | Dataset | Docs | Total Tokens | Unique Tokens | Vocab Coverage | Code Ratio |
    |---------|------|--------------|---------------|----------------|------------|
    | **Python** | 1,000,000 | 1.44B | 230,100 | 89.9% | 57.3% |
    | **English** | 1,000,000 | 1.02B | 217,910 | 85.1% | 0.0% |
    | **Combined** | 2,000,000 | 2.46B | 244,423 | **95.5%** | 28.7% |

    ### Our Implementation of Ask-LLM (Next Step)

    **What the paper found:**
    - Quality-focused sampling (Ask-LLM) outperforms training on the full dataset
    - Coverage sampling (Density) recovers full-data performance
    - Small proxy models (7B) are sufficient for reliable quality scoring

    **Our approach for Python + English pretraining:**

    1. **Judge Model Selection**
       - Python documents → `qwen2.5-coder:7b` (specialized for code quality)
       - English documents → `qwen2.5:7b-instruct` (general text quality)

    2. **Scoring Strategy**
       - Each document gets a quality score (0-1) from the judge
       - Score based on: correctness, clarity, educational value
       - Temperature=0 for deterministic, reproducible scores

    3. **Sampling Decision**
       - Keep top 30% of documents by Ask-LLM score
       - Paper shows this yields better models than using 100% of data
       - Saves 70% of training time and compute

    ### Why This Works for Us

    | Metric (from 1M analysis) | Value | Insight |
    |---------------------------|-------|---------|
    | Combined vocab coverage | **95.5%** | Almost full tokenizer utilization |
    | Duplicate rate | 0.2-0.3% | Data is exceptionally clean |
    | Existing quality score | 0.85-0.88 | Starting from high baseline |
    | Python code ratio | 57.3% | Perfect balance for code+English |

    The Ask-LLM judge will identify the **best of the best** - documents that teach the model both coding patterns AND natural language understanding.

    ### Expected Outcome After Ask-LLM Scoring

    | | Full Data (Current) | After Ask-LLM (Top 30%) |
    |--|---------------------|-------------------------|
    | Documents | 1M + 1M | 300K + 300K |
    | Tokens | 2.46B | ~740M |
    | Training time | 100% | ~30% |
    | Model quality | Baseline | **Higher** |

    ### Next Steps

    1. ✅ Complete 1M analysis (in progress)
    2. ⏳ Run Ask-LLM scoring on both datasets
    3. ⏳ Filter to top 30% by quality score
    4. ⏳ Start pretraining on M3 Ultra

    *"Even when we reject 90% of the original dataset, models trained on Ask-LLM data consistently outperform full-data training"* - Sachdeva et al., 2024
    """)
    return


@app.cell
def _():
    import requests
    from tqdm import tqdm
    from concurrent.futures import ThreadPoolExecutor, as_completed

    def get_score(text, doc_type):
        model = "qwen2.5-coder:7b" if doc_type == "python" else "qwen2.5:7b-instruct"
        prompt = f"Rate this {'code' if doc_type=='python' else 'text'} quality 0-1. Reply only number:\n\n{text[:2000]}"
    
        try:
            r = requests.post("http://localhost:11434/api/generate", 
                             json={"model": model, "prompt": prompt, "stream": False, "options": {"temperature": 0}})
            score = float(r.json()["response"].strip().split()[0])
            return max(0.0, min(1.0, score))
        except:
            return 0.5

    return ThreadPoolExecutor, as_completed, get_score, tqdm


@app.cell
def _(
    ThreadPoolExecutor,
    as_completed,
    get_score,
    local_english_dataset,
    local_python_dataset,
    tqdm,
):
    def add_scores(dataset, text_field, doc_type, sample_size=50000):
        dataset = dataset.select(range(min(sample_size, len(dataset))))
        scores = [None] * len(dataset)
    
        with ThreadPoolExecutor(max_workers=8) as ex:
            futures = {ex.submit(get_score, doc[text_field], doc_type): i for i, doc in enumerate(dataset)}
            for f in tqdm(as_completed(futures), total=len(futures)):
                scores[futures[f]] = f.result()
    
        return dataset.add_column("ask_llm_score", scores)

    # Run
    python_scored = add_scores(local_python_dataset, "content", "python", 50000)
    english_scored = add_scores(local_english_dataset, "text", "english", 50000)
    return english_scored, python_scored


@app.cell
def _(english_scored, python_scored):
    py_scores = python_scored["ask_llm_score"]
    en_scores = english_scored["ask_llm_score"]

    print(f"Python: min={min(py_scores):.2f}, mean={sum(py_scores)/len(py_scores):.2f}")
    print(f"English: min={min(en_scores):.2f}, mean={sum(en_scores)/len(en_scores):.2f}")
    print(f"\nTop 30% threshold - Python: {sorted(py_scores, reverse=True)[int(len(py_scores)*0.3)]:.2f}")
    print(f"Top 30% threshold - English: {sorted(en_scores, reverse=True)[int(len(en_scores)*0.3)]:.2f}")
    return


if __name__ == "__main__":
    app.run()
