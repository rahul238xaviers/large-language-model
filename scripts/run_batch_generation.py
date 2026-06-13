#!/usr/bin/env python3
"""Batch generation runner for SFT data.

This script automates running the SFT generator in consecutive small batches (e.g. of size 30)
using the newly implemented --offset parameter. This prevents the model server from
crashing due to cumulative memory/caching overhead during long single runs.
"""

import os
import sys
import signal
import shutil
import argparse
import subprocess
import time
import urllib.request
import urllib.error
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description="Automate long SFT data generation runs in safe, sequential batches.")
    parser.add_argument("--start-offset", type=int, default=60, help="Offset to start generating from (default: 60)")
    parser.add_argument("--total", type=int, default=300, help="Total number of records to generate across all batches (default: 300)")
    parser.add_argument("--batch-size", type=int, default=30, help="Size of each generation batch (default: 30)")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Max concurrent workers per batch (default: 4)")
    parser.add_argument("--rate-limit", type=int, default=20, help="API calls per second limit (default: 20)")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit", help="Model name to serve")
    parser.add_argument("--port", type=int, default=8000, help="Port to run the model server on")
    return parser.parse_args()

def find_server_binary():
    for name in ["rapid-mlx", "vllm-mlx"]:
        path = shutil.which(name)
        if path:
            return path
    for path in ["/opt/homebrew/bin/rapid-mlx", "/opt/homebrew/bin/vllm-mlx"]:
        if Path(path).exists():
            return path
    raise FileNotFoundError("Could not find rapid-mlx or vllm-mlx server binaries on system PATH or in /opt/homebrew/bin.")

def start_server(server_bin, model, port, repo_root):
    log_dir = repo_root / "logs"
    log_dir.mkdir(exist_ok=True)
    server_log_path = log_dir / "model_server.log"
    print(f"📡 Starting model server ({model}) on port {port}...")
    print(f"📝 Logging server output to {server_log_path}")
    
    server_log = open(server_log_path, "w", encoding="utf-8")
    
    server_cmd = [
        server_bin, "serve",
        model,
        "--port", str(port),
        "--use-paged-cache",
        "--kv-cache-quantization",
        "--gpu-memory-utilization", "0.6"
    ]
    
    proc = subprocess.Popen(
        server_cmd,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,
        cwd=str(repo_root)
    )
    
    print("⏳ Waiting for model to load and server to respond...")
    for i in range(90):  # Wait up to 180 seconds
        time.sleep(2)
        try:
            req = urllib.request.Request(f"http://localhost:{port}/v1/models")
            with urllib.request.urlopen(req, timeout=2) as response:
                if response.status == 200:
                    print("✅ Server is healthy and online.")
                    return proc, server_log
        except urllib.error.HTTPError as e:
            print(f"✅ Server is online (returned status {e.code}).")
            return proc, server_log
        except (urllib.error.URLError, ConnectionResetError, ConnectionRefusedError):
            if i % 10 == 0 and i > 0:
                print(f"   ... still waiting ({i*2}s elapsed) ...")
            pass
            
    print("❌ Server failed to respond within 180 seconds.")
    kill_server(proc, server_log)
    sys.exit(1)

def kill_server(proc, server_log):
    print("🔌 Stopping model server to release Metal memory allocations...")
    if server_log:
        try:
            server_log.close()
        except Exception:
            pass
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=15)
        print("✅ Server stopped successfully.")
    except subprocess.TimeoutExpired:
        print("⚠️ Server did not exit on SIGTERM, sending SIGKILL...")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait()
            print("✅ Server killed successfully.")
        except Exception as e:
            print(f"⚠️ Error while killing server: {e}")
    except Exception as e:
        print(f"⚠️ Warning while stopping server: {e}")

def main():
    args = parse_args()
    repo_root = Path("/Users/rahulkumar/dev/large-language-model")
    python_bin = repo_root / ".venv" / "bin" / "python3"
    
    if not python_bin.exists():
        python_bin = "python3"
        
    generator_script = repo_root / "scripts" / "process_rag_data_rapid_mlx.py"
    unifier_script = repo_root / "scripts" / "unify_and_clean_runs.py"
    
    try:
        server_bin = find_server_binary()
        print(f"🔍 Found server binary: {server_bin}")
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)
        
    start = args.start_offset
    end = start + args.total
    step = args.batch_size
    
    print("="*60)
    print("🚀 STARTING AUTOMATED SFT BATCH GENERATION WITH RECYCLING")
    print("="*60)
    print(f"• Start Offset:   {start}")
    print(f"• Total Records:  {args.total}")
    print(f"• Batch Size:     {step}")
    print(f"• End Offset:     {end}")
    print(f"• Model Served:   {args.model}")
    print(f"• Server Port:    {args.port}")
    print(f"• Generator:      {generator_script.name}")
    print(f"• Unifier:        {unifier_script.name}")
    print("="*60)
    
    success_batches = 0
    total_batches = (args.total + step - 1) // step
    
    for offset in range(start, end, step):
        batch_num = success_batches + 1
        limit = min(step, end - offset)
        
        # Start a fresh model server process for this batch
        server_proc, server_log = start_server(server_bin, args.model, args.port, repo_root)
        
        print(f"\n📦 [Batch {batch_num}/{total_batches}] Generating records from offset {offset} to {offset + limit}...")
        
        cmd = [
            str(python_bin),
            str(generator_script),
            "--limit", str(limit),
            "--offset", str(offset),
            "--workers", str(args.workers),
            "--rate-limit", str(args.rate_limit),
            "--gen-model", args.model,
            "--judge-model", args.model,
            "--gen-url", f"http://localhost:{args.port}/v1",
            "--judge-url", f"http://localhost:{args.port}/v1"
        ]
        
        start_time = time.time()
        try:
            result = subprocess.run(cmd, check=True, cwd=str(repo_root))
            elapsed = time.time() - start_time
            print(f"✅ [Batch {batch_num}/{total_batches}] Completed successfully in {elapsed:.1f}s.")
            success_batches += 1
            
            # Shut down server to clear Metal/GPU memory leaks
            kill_server(server_proc, server_log)
            
            # Run unification after each successful batch to update the train/valid files and log progress
            print(f"🔄 Updating unified dataset...")
            subprocess.run([str(python_bin), str(unifier_script)], check=True, cwd=str(repo_root))
            
        except subprocess.CalledProcessError as e:
            print(f"\n❌ [Batch {batch_num}/{total_batches}] FAILED at offset {offset} with exit code {e.returncode}.")
            kill_server(server_proc, server_log)
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

