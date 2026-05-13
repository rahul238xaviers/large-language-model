import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src'))

from config import TrainingConfig
from model import GPTModel

def test_compilation_time():
    config = TrainingConfig()
    # Use standard batch settings
    config.micro_batch_size = 16
    config.block_size = 2048
    
    print("Initializing Model...")
    model = GPTModel(config)
    model.set_dtype(mx.bfloat16)
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
    x = mx.random.randint(0, config.vocab_size, (config.micro_batch_size, config.block_size))
    y = mx.random.randint(0, config.vocab_size, (config.micro_batch_size, config.block_size))
    mx.eval(x, y)
    
    print("\n--- Starting Iteration 1 (Expect compilation delay) ---")
    start = time.time()
    loss, grads = step(x, y)
    mx.eval(loss, grads) # Force execution
    dt1 = time.time() - start
    print(f"Iteration 1 Time: {dt1:.2f}s")
    
    print("\n--- Starting Iteration 2 (Expect pure execution) ---")
    start = time.time()
    loss, grads = step(x, y)
    mx.eval(loss, grads) # Force execution
    dt2 = time.time() - start
    print(f"Iteration 2 Time: {dt2:.2f}s")
    
    print("\n--- Starting Iteration 3 ---")
    start = time.time()
    loss, grads = step(x, y)
    mx.eval(loss, grads) # Force execution
    dt3 = time.time() - start
    print(f"Iteration 3 Time: {dt3:.2f}s")
    
    print(f"\nCompilation Overhead: {dt1 - dt2:.2f}s")
    print(f"Steady State Step Time: {dt2:.2f}s")

if __name__ == "__main__":
    test_compilation_time()
