import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/pre-training'))

from config import TrainingConfig
from model import GPTModel

def benchmark_config(mbs, acc):
    config = TrainingConfig()
    config.micro_batch_size = mbs
    config.gradient_accumulation_steps = acc
    
    print(f"\n🚀 Initializing Model for MBS={mbs}, ACC={acc}...")
    model = GPTModel(config)
    model.set_dtype(config.mx_dtype)
    mx.eval(model.parameters())
    
    optimizer = optim.AdamW(learning_rate=1e-4)
    
    def loss_fn(model, x, y):
        logits = model(x)
        loss = mx.mean(nn.losses.cross_entropy(logits, y))
        return loss
        
    @mx.compile
    def step(x, y):
        return nn.value_and_grad(model, loss_fn)(model, x, y)
    
    # Mock data
    x = mx.random.randint(0, config.vocab_size, (mbs, config.block_size))
    y = mx.random.randint(0, config.vocab_size, (mbs, config.block_size))
    mx.eval(x, y)
    
    print("⏳ Compiling and Warming up...")
    t_start = time.time()
    loss, grads = step(x, y)
    mx.eval(loss, grads)
    print(f"✅ Compilation finished in {time.time() - t_start:.2f}s")
    
    print(f"📊 Starting benchmark (8 accumulation steps equivalent)...")
    start = time.time()
    # We simulate 'acc' steps but for comparison we can just do 8 steps total for both
    # so we compare raw throughput.
    for i in range(8):
        loss, grads = step(x, y)
        optimizer.update(model, grads)
        mx.eval(model.parameters(), optimizer.state)
        if (i+1) % 2 == 0:
            print(f"   Step {i+1}/8 finished...")
            
    dt = time.time() - start
    print(f"✨ Total Execution Time: {dt:.2f}s")
    
    # Cleanup to free memory
    del model, optimizer, grads, loss
    mx.clear_cache()
    
    return dt

if __name__ == "__main__":
    # Test 16x8 (What we are doing now)
    t16 = benchmark_config(16, 8)
    
    # Test 32x4 (Proposed)
    # Since we want to compare SAME total tokens, we run 4 steps of MBS=32
    # but we just ran 8 steps above. So we'll normalize by tokens per second.
    t32 = benchmark_config(32, 8) # Run 8 steps of 32 to get stable throughput
    
    print(f"\n{'='*40}")
    print(f"FINAL RESULTS (Steady State Throughput)")
    print(f"MBS=16: {t16:.2f}s for 8 steps")
    print(f"MBS=32: {t32:.2f}s for 8 steps")
    print(f"Relative Efficiency: {(t16*2 / t32):.2f}x")
    print(f"{'='*40}")
