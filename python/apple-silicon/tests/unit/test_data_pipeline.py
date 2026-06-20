import time
import sys
import os
import queue

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/pre-training'))

from config import TrainingConfig
from data import ParallelTokenStream

def test_data_throughput():
    config = TrainingConfig()
    
    print(f"Initializing ParallelTokenStream with {config.num_worker_threads} workers...")
    stream = ParallelTokenStream(config)
    
    print("Starting token stream...")
    start_time = time.time()
    total_tokens = 0
    target_tokens = 200000 # Measure 200k tokens
    
    last_print = start_time
    
    try:
        while total_tokens < target_tokens:
            try:
                # get_batch returns x, y which are lists of tokens
                x, y = stream.get_batch()
                total_tokens += len(x)
                
                now = time.time()
                if now - last_print > 5:
                    elapsed = now - start_time
                    tok_per_sec = total_tokens / elapsed
                    print(f"Progress: {total_tokens}/{target_tokens} tokens | Throughput: {tok_per_sec:.0f} tok/s")
                    last_print = now
            except Exception as e:
                print(f"Error in stream: {e}")
                break
                
    finally:
        stream.stop()
        
    end_time = time.time()
    total_elapsed = end_time - start_time
    print(f"\nFinal Results:")
    print(f"Total Tokens: {total_tokens}")
    print(f"Total Time: {total_elapsed:.2f}s")
    print(f"Average Throughput: {total_tokens / total_elapsed:.0f} tok/s")

if __name__ == "__main__":
    test_data_throughput()
