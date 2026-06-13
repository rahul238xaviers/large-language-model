#!/usr/bin/env python3
"""Script to unify, validate, and deduplicate SFT data from all past runs.

This script scans all subdirectories in data/processed/, extracts all successful (PASS)
records, runs compilation checks using our local Cargo project verification engine,
performs repetition filtering, deduplicates based on instructions, and splits the result
into high-quality train and validation files.
"""

import os
import re
import sys
import json
import random
import asyncio
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

def get_record_hash(record: Dict[str, Any]) -> str:
    """Generate SHA-256 hash for instruction-response pair."""
    instruction = record.get("instruction", "").strip()
    response = record.get("response", "").strip()
    content = f"{instruction}\n{response}"
    return hashlib.sha256(content.encode("utf-8")).hexdigest()

# Set repository root absolutely
repo_root = Path("/Users/rahulkumar/dev/large-language-model")
sys.path.insert(0, str(repo_root))

# Try importing tomllib, fallback to toml
try:
    import tomllib
except ImportError:
    import toml as tomllib

from scripts.process_rag_data_rapid_mlx import (
    ensure_validation_workspace,
    validate_cargo_toml,
    compile_rust_code,
    has_catastrophic_repetition,
    robust_extract_response,
    robust_extract_context
)

# Configuration
PROCESSED_DIR = repo_root / "data" / "processed"
TARGET_DIR = PROCESSED_DIR / "unified_sft_dataset"
WORKSPACE_DIR = PROCESSED_DIR / "temp_validation_cargo_project"
TRAIN_RATIO = 0.85
SEED = 42

async def collect_records() -> List[Dict[str, Any]]:
    """Scan all processed runs and collect passed records."""
    records = []
    seen_records = set()
    
    run_folders = [f for f in PROCESSED_DIR.iterdir() if f.is_dir() and re.match(r'^\d{8}_\d{6}$', f.name)]
    print(f"🔍 Found {len(run_folders)} run directories in {PROCESSED_DIR}")
    
    for folder in sorted(run_folders):
        audit_file = folder / "evaluation_audit.jsonl"
        if not audit_file.exists():
            continue
            
        print(f"📂 Reading audit logs from {folder.name}...")
        try:
            with open(audit_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                        if record.get("decision") == "PASS":
                            # Use (instruction, response) as a key to avoid exact duplicate pairs immediately
                            key = (record.get("instruction", "").strip(), record.get("response", "").strip())
                            if key not in seen_records:
                                seen_records.add(key)
                                records.append(record)
                    except Exception as e:
                        print(f"  ⚠️ Failed to parse line in {audit_file.name}: {e}")
        except Exception as e:
            print(f"  ❌ Error reading {audit_file}: {e}")
            
    print(f"✨ Collected {len(records)} unique candidate records across all runs.")
    return records

async def validate_candidate(
    record: Dict[str, Any], 
    lock: asyncio.Lock
) -> Optional[Tuple[Dict[str, Any], str]]:
    """Validate candidate record for repetition, complexity, and compilation.
    
    Returns (cleaned_record, success_reason) if valid, None if invalid.
    """
    instruction = record.get("instruction") or ""
    context = record.get("context") or ""
    response = record.get("response") or ""
    source = record.get("source") or "unknown"
    
    # 1. Check for repetition
    if has_catastrophic_repetition(response):
        return None, "Failed repetition check"
        
    if source == "matteopilotto/rust-github-issues":
        # Sanitization check and basic roadmap checks (lines count)
        clean_context = context.strip()
        if len(response.strip().splitlines()) < 5:
            return None, "Roadmap response is too short"
        return record, "Pass"
        
    elif source == "Convence/Rust-Coder":
        # 2. Extract context and response robustly
        new_response = robust_extract_response(response)
        new_context = robust_extract_context(context, context)
        
        if not new_response:
            return None, "Failed robust response extraction"
            
        is_toml = "cargo.toml" in instruction.lower() or new_response.strip().startswith("[")
        
        # 3. Complexity check
        lines = new_response.strip().splitlines()
        has_rust_structure = any(kw in new_response for kw in ["fn ", "struct ", "impl ", "enum ", "macro_rules!", "trait "])
        if not is_toml and (len(lines) < 8 or not has_rust_structure):
            return None, "Failed complexity check (too simple)"
            
        # 4. Syntactic validation
        if is_toml:
            toml_err = validate_cargo_toml(new_response)
            if toml_err:
                return None, f"TOML validation error: {toml_err}"
        else:
            passed, compile_err = await compile_rust_code(new_response, WORKSPACE_DIR, lock)
            if not passed:
                return None, f"Rust compilation error: {compile_err.strip()[:100]}"
                
        cleaned_record = {
            "source": source,
            "instruction": instruction,
            "context": new_context,
            "response": new_response
        }
        return cleaned_record, "Pass"
        
    return None, "Unknown source format"

async def main():
    # Setup workspace
    ensure_validation_workspace(WORKSPACE_DIR)
    compilation_lock = asyncio.Lock()
    
    # Collect records
    candidates = await collect_records()
    if not candidates:
        print("❌ No candidates collected. Exiting.")
        return
        
    # Setup cache directory and file path
    cache_file = TARGET_DIR / "validation_cache.json"
    validation_cache = {}
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                validation_cache = json.load(f)
            print(f"📦 Loaded {len(validation_cache)} entries from validation cache.")
        except Exception as e:
            print(f"⚠️ Failed to load validation cache: {e}")

    # Validate candidates
    print("\n⚡ Validating and filtering candidates...")
    valid_records = []
    validated_count = 0
    cache_hits = 0
    cache_updated = False
    
    for record in candidates:
        validated_count += 1
        r_hash = get_record_hash(record)
        
        if r_hash in validation_cache:
            cache_hits += 1
            entry = validation_cache[r_hash]
            is_valid = entry.get("is_valid", False)
            reason = entry.get("reason", "Unknown cached reason")
            cleaned = entry.get("cleaned_record")
            
            if is_valid and cleaned:
                valid_records.append(cleaned)
        else:
            cleaned, reason = await validate_candidate(record, compilation_lock)
            is_valid = cleaned is not None
            validation_cache[r_hash] = {
                "is_valid": is_valid,
                "reason": reason,
                "cleaned_record": cleaned
            }
            cache_updated = True
            
            if is_valid:
                valid_records.append(cleaned)
                print(f"  [{validated_count}/{len(candidates)}] ✅ VALID (Compiled): {record.get('instruction')[:60]}...")
            else:
                print(f"  [{validated_count}/{len(candidates)}] ❌ DISCARDED ({reason}): {record.get('instruction')[:60]}...")
                
    print(f"\n⚡ Validation stats: {cache_hits} cache hits, {len(candidates) - cache_hits} new evaluations compiled.")
    print(f"✨ Validation phase complete. {len(valid_records)}/{len(candidates)} records passed validation.")
    
    # Save validation cache if new evaluations occurred
    if cache_updated:
        try:
            TARGET_DIR.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(validation_cache, f, indent=2)
            print(f"💾 Saved updated validation cache to {cache_file} ({len(validation_cache)} total entries).")
        except Exception as e:
            print(f"⚠️ Failed to save validation cache: {e}")
    
    # Deduplicate by instruction (keeping the longest/richest response)
    print("\n⚡ Deduplicating by instruction...")
    deduped_map = {}
    for r in valid_records:
        inst = r["instruction"].strip().lower()
        if inst in deduped_map:
            # Keep the longer response (better entropy/details)
            if len(r["response"]) > len(deduped_map[inst]["response"]):
                deduped_map[inst] = r
        else:
            deduped_map[inst] = r
            
    final_records = list(deduped_map.values())
    print(f"✨ Deduplication complete. {len(final_records)} unique high-quality samples remaining.")
    
    if not final_records:
        print("❌ No records left after validation and deduplication.")
        return
        
    # Split and write
    random.seed(SEED)
    random.shuffle(final_records)
    
    split_idx = int(len(final_records) * TRAIN_RATIO)
    train_data = final_records[:split_idx]
    valid_data = final_records[split_idx:]
    
    TARGET_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save chat format files
    train_file = TARGET_DIR / "train.jsonl"
    valid_file = TARGET_DIR / "valid.jsonl"
    audit_file = TARGET_DIR / "evaluation_audit.jsonl"
    
    print(f"\n💾 Saving datasets to {TARGET_DIR}...")
    
    with open(train_file, "w", encoding="utf-8") as f:
        for r in train_data:
            formatted = {
                "messages": [
                    {"role": "system", "content": "You are a helpful Rust programming assistant trained in RAG context retrieval."},
                    {"role": "user", "content": f"Context:\n{r['context']}\n\nQuestion: {r['instruction']}"},
                    {"role": "assistant", "content": r['response']}
                ]
            }
            f.write(json.dumps(formatted) + "\n")
            
    with open(valid_file, "w", encoding="utf-8") as f:
        for r in valid_data:
            formatted = {
                "messages": [
                    {"role": "system", "content": "You are a helpful Rust programming assistant trained in RAG context retrieval."},
                    {"role": "user", "content": f"Context:\n{r['context']}\n\nQuestion: {r['instruction']}"},
                    {"role": "assistant", "content": r['response']}
                ]
            }
            f.write(json.dumps(formatted) + "\n")
            
    with open(audit_file, "w", encoding="utf-8") as f:
        for r in final_records:
            # Log as audit format
            audit_item = {
                "source": r["source"],
                "instruction": r["instruction"],
                "context": r["context"],
                "response": r["response"],
                "decision": "PASS",
                "reason": "Passed unification validation checks"
            }
            f.write(json.dumps(audit_item) + "\n")
            
    print(f"✅ SUCCESS! Generated {len(train_data)} train and {len(valid_data)} validation samples.")
    print(f"📁 Target folder: {TARGET_DIR}")

if __name__ == "__main__":
    asyncio.run(main())
