#!/usr/bin/env python3
"""Process rust_coder and rust_github_issues datasets into unified JSONL files for RAG training.

Usage:
    python scripts/process_rag_data.py
"""

import json
import os
import random
import sys
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

def load_records(dataset_name, cache_dir):
    from datasets import load_dataset
    print(f"📂 Loading {dataset_name} from {cache_dir}...")
    dataset = load_dataset(dataset_name, cache_dir=cache_dir)
    records = []
    if isinstance(dataset, dict):
        for split_name, split in dataset.items():
            print(f"  - Extracting split: {split_name} ({len(split)} records)")
            for row in split:
                records.append(row)
    else:
        print(f"  - Extracting main split ({len(dataset)} records)")
        for row in dataset:
            records.append(row)
    return records

def main():
    combined_data = []

    # 1. Process rust_coder
    rust_coder_dir = "/Users/rahulkumar/dev/large-language-model/data/datasets/rust_coder_cache"
    try:
        rust_coder_records = load_records("Convence/Rust-Coder", rust_coder_dir)
        for row in rust_coder_records:
            instruction = row.get("instruction") or ""
            context = row.get("explanation") or ""
            response = row.get("code") or ""
            
            # Formatting template
            formatted_text = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}<|endoftext|>"
            combined_data.append({"text": formatted_text})
    except Exception as e:
        print(f"⚠️ Error loading rust_coder: {e}")

    # 2. Process rust_github_issues
    rust_issues_dir = "/Users/rahulkumar/dev/large-language-model/data/datasets/rust_github_issues_cache"
    try:
        rust_github_issues_records = load_records("matteopilotto/rust-github-issues", rust_issues_dir)
        for row in rust_github_issues_records:
            instruction = row.get("title") or ""
            context = row.get("body") or ""
            response = ""  # No response field in issues dataset
            
            # Formatting template
            formatted_text = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}<|endoftext|>"
            combined_data.append({"text": formatted_text})
    except Exception as e:
        print(f"⚠️ Error loading rust_github_issues: {e}")

    print(f"\n📊 Total combined records: {len(combined_data)}")
    if not combined_data:
        print("❌ No data processed. Exiting.")
        return

    # Shuffle the dataset using a fixed seed
    print("🎲 Shuffling dataset with seed 42...")
    random.seed(42)
    random.shuffle(combined_data)

    # Split into 85% train, 15% valid
    split_idx = int(len(combined_data) * 0.85)
    train_data = combined_data[:split_idx]
    valid_data = combined_data[split_idx:]
    print(f"📝 Split data: {len(train_data)} train records, {len(valid_data)} validation records.")

    # Save to data/processed/
    processed_dir = repo_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    train_file = processed_dir / "train_sft.jsonl"
    valid_file = processed_dir / "valid_sft.jsonl"

    print(f"💾 Saving outputs to {processed_dir}...")
    
    with open(train_file, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
    with open(valid_file, "w", encoding="utf-8") as f:
        for item in valid_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"✅ Successfully created {train_file.name} and {valid_file.name}!")

if __name__ == "__main__":
    main()
