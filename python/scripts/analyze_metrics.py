#!/usr/bin/env python3
"""Analyze and plot performance metrics of the SFT RAG MLX dataset processing pipeline.

Reads 'metrics.jsonl' from a run directory, computes summary stats, prints a text representation of
the latency degradation, writes a CSV for spreadsheet plotting, and attempts to plot using matplotlib.
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

def analyze(metrics_file: Path):
    if not metrics_file.exists():
        print(f"❌ Metrics file not found: {metrics_file}")
        sys.exit(1)

    print(f"📊 Analyzing metrics from: {metrics_file}")
    
    records = []
    with open(metrics_file, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    if not records:
        print("⚠️ No metric records found in the file.")
        return
        
    print(f"📈 Loaded {len(records)} metric events.")
    
    # Separate phases
    gen_records = [r for r in records if r["phase"] == "generation"]
    val_records = [r for r in records if r["phase"] == "validation"]
    val_heur_records = [r for r in records if r["phase"] == "validation_heuristic"]
    
    print("\n" + "="*50)
    print("SUMMARY STATISTICS")
    print("="*50)
    
    for phase_name, phase_data in [
        ("Generation (Phase 1)", gen_records), 
        ("Validation LLM (Phase 2)", val_records),
        ("Validation Heuristic Filter", val_heur_records)
    ]:
        if not phase_data:
            print(f"\n{phase_name}: No records.")
            continue
            
        count = len(phase_data)
        if phase_name == "Validation Heuristic Filter":
            print(f"\n{phase_name}:")
            print(f"  - Total events filtered: {count}")
            continue
            
        total_time = sum(r["elapsed"] for r in phase_data)
        avg_time = total_time / count
        avg_prompt = sum(r["prompt_tokens"] for r in phase_data) / count
        avg_completion = sum(r["completion_tokens"] for r in phase_data) / count
        avg_tps = sum(r["tokens_per_sec"] for r in phase_data) / count
        max_time = max(r["elapsed"] for r in phase_data)
        min_time = min(r["elapsed"] for r in phase_data)
        
        print(f"\n{phase_name}:")
        print(f"  - Count: {count} requests")
        print(f"  - Avg Latency: {avg_time:.2f}s (range: {min_time:.2f}s - {max_time:.2f}s)")
        print(f"  - Avg Prompt Tokens (Input): {avg_prompt:.1f}")
        print(f"  - Avg Completion Tokens (Output): {avg_completion:.1f}")
        print(f"  - Avg Generation Speed: {avg_tps:.2f} tokens/sec")
        print(f"  - Total Time: {total_time:.2f}s")
        
    # Write CSV for plotting
    csv_file = metrics_file.with_name("latency_metrics.csv")
    print(f"\n💾 Writing CSV for spreadsheet plotting to: {csv_file}")
    with open(csv_file, "w") as f:
        f.write("timestamp,phase,row_idx,model,prompt_tokens,completion_tokens,elapsed,tokens_per_sec\n")
        for r in records:
            f.write(f"{r['timestamp']},{r['phase']},{r['row_idx']},{r['model']},{r['prompt_tokens']},{r['completion_tokens']},{r['elapsed']},{r['tokens_per_sec']}\n")

    # Try to plot with matplotlib
    try:
        import matplotlib.pyplot as plt
        
        plt.figure(figsize=(12, 6))
        
        # Plot generation latency over row index
        if gen_records:
            gen_x = [r["row_idx"] for r in gen_records]
            gen_y = [r["elapsed"] for r in gen_records]
            plt.plot(gen_x, gen_y, marker='o', linestyle='-', label=f"Generation Latency (Avg: {sum(gen_y)/len(gen_y):.1f}s)", color='#1f77b4')
            
        # Plot validation latency over row index
        if val_records:
            val_x = [r["row_idx"] for r in val_records]
            val_y = [r["elapsed"] for r in val_records]
            plt.plot(val_x, val_y, marker='s', linestyle='-', label=f"Validation Latency (Avg: {sum(val_y)/len(val_y):.1f}s)", color='#ff7f0e')
            
        plt.xlabel("Row Index (Chronological Execution)")
        plt.ylabel("Latency per Request (seconds)")
        plt.title("API Request Latency Trend (Detecting Performance Degradation / Swapping)")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        
        plot_file = metrics_file.with_name("latency_plot.png")
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📈 Saved latency plot graph to: {plot_file}")
        
    except ImportError:
        print("\n💡 Tip: Install 'matplotlib' (pip install matplotlib) to automatically generate PNG graphs.")
        
    # Print ASCII trend
    if len(records) > 5:
        print("\n📊 Trend analysis (Request Latency over time):")
        # Split into 5 buckets to show degradation
        chunk_size = max(1, len(records) // 5)
        for i in range(5):
            chunk = records[i*chunk_size : (i+1)*chunk_size]
            if not chunk:
                break
            # Only count actual LLM calls (elapsed > 0)
            llm_calls = [r["elapsed"] for r in chunk if r["elapsed"] > 0]
            if llm_calls:
                avg_lat = sum(llm_calls) / len(llm_calls)
                bar = "█" * int(min(20, avg_lat * 2))
                print(f"  - Exec chunk {i+1}/5 (Rows {chunk[0]['row_idx']}-{chunk[-1]['row_idx']}): {avg_lat:5.2f}s {bar}")

def main():
    parser = argparse.ArgumentParser(description="Analyze performance metrics logs.")
    parser.add_argument("metrics_file", nargs="?", help="Path to metrics.jsonl file")
    args = parser.parse_args()
    
    if args.metrics_file:
        metrics_path = Path(args.metrics_file)
    else:
        # Find latest run
        processed_dir = Path("data/processed")
        if not processed_dir.exists():
            print("❌ No 'data/processed' directory found.")
            sys.exit(1)
            
        runs = sorted([d for d in processed_dir.iterdir() if d.is_dir()])
        if not runs:
            print("❌ No run directories found inside 'data/processed'.")
            sys.exit(1)
            
        latest_run = runs[-1]
        metrics_path = latest_run / "metrics.jsonl"
        print(f"ℹ️ Found latest run directory: {latest_run}")
        
    analyze(metrics_path)

if __name__ == "__main__":
    main()
