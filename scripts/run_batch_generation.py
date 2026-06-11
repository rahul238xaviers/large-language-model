#!/usr/bin/env python3
"""Batch generation runner for SFT data.

This script automates running the SFT generator in consecutive small batches (e.g. of size 30)
using the newly implemented --offset parameter. This prevents the model server from
crashing due to cumulative memory/caching overhead during long single runs.
"""

import sys
import argparse
import subprocess
import time
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Automate long SFT data generation runs in safe, sequential batches.")
    parser.add_argument("--start-offset", type=int, default=60, help="Offset to start generating from (default: 60)")
    parser.add_argument("--total", type=int, default=300, help="Total number of records to generate across all batches (default: 300)")
    parser.add_argument("--batch-size", type=int, default=30, help="Size of each generation batch (default: 30)")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Max concurrent workers per batch (default: 4)")
    parser.add_argument("--rate-limit", type=int, default=20, help="API calls per second limit (default: 20)")
    return parser.parse_args()

def main():
    args = parse_args()
    repo_root = Path("/Users/rahulkumar/dev/large-language-model")
    python_bin = repo_root / ".venv" / "bin" / "python3"
    
    if not python_bin.exists():
        python_bin = "python3"
        
    generator_script = repo_root / "scripts" / "process_rag_data_rapid_mlx.py"
    unifier_script = repo_root / "scripts" / "unify_and_clean_runs.py"
    
    start = args.start_offset
    end = start + args.total
    step = args.batch_size
    
    print("="*60)
    print("🚀 STARTING AUTOMATED SFT BATCH GENERATION")
    print("="*60)
    print(f"• Start Offset:   {start}")
    print(f"• Total Records:  {args.total}")
    print(f"• Batch Size:     {step}")
    print(f"• End Offset:     {end}")
    print(f"• Generator:      {generator_script.name}")
    print(f"• Unifier:        {unifier_script.name}")
    print("="*60)
    
    success_batches = 0
    total_batches = (args.total + step - 1) // step
    
    for offset in range(start, end, step):
        batch_num = success_batches + 1
        limit = min(step, end - offset)
        
        print(f"\n📦 [Batch {batch_num}/{total_batches}] Generating records from offset {offset} to {offset + limit}...")
        
        cmd = [
            str(python_bin),
            str(generator_script),
            "--limit", str(limit),
            "--offset", str(offset),
            "--workers", str(args.workers),
            "--rate-limit", str(args.rate_limit)
        ]
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, check=True, cwd=str(repo_root))
            elapsed = time.time() - start_time
            print(f"✅ [Batch {batch_num}/{total_batches}] Completed successfully in {elapsed:.1f}s.")
            success_batches += 1
            
            # Run unification after each successful batch to update the train/valid files and log progress
            print(f"🔄 Updating unified dataset...")
            subprocess.run([str(python_bin), str(unifier_script)], check=True, cwd=str(repo_root))
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ [Batch {batch_num}/{total_batches}] FAILED at offset {offset} with exit code {e.returncode}.")
            print("💡 The model server might have OOM'd or disconnected. Please check the model server logs.")
            print("💡 Once the model server is running again, you can resume batch generation by running this script again with:")
            print(f"   --start-offset {offset} --total {end - offset}")
            sys.exit(1)
            
    print("\n" + "="*60)
    print("🎉 ALL BATCHES PROCESSED SUCCESSFULLY!")
    print(f"Successfully processed {success_batches}/{total_batches} batches.")
    print("All records have been validated, cleaned, and unified into:")
    print("  data/processed/unified_sft_dataset/train.jsonl")
    print("="*60)

if __name__ == "__main__":
    main()
