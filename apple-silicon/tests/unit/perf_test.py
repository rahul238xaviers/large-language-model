import time
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import sys
import os

# Add src to path so we can import modules
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/pre-training'))

from config import TrainingConfig
from model import GPTModel

def test_gpu_speed():
    config = TrainingConfig()
    config.micro_batch_size = 16
    config.gradient_accumulation_steps = 8
    config.block_size = 2048
    
    print("Initializing Model...")
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
    
    print("Generating random batch...")
    # Mock data
    x = mx.random.randint(0, config.vocab_size, (config.micro_batch_size, config.block_size))
    y = mx.random.randint(0, config.vocab_size, (config.micro_batch_size, config.block_size))
    mx.eval(x, y)
    
    print("Testing compiled step...")
    start = time.time()
    
    import mlx.utils as mut
    # Lazy accumulation
    accumulated_grads = None
    for i in range(config.gradient_accumulation_steps):
        loss, grads = step(x, y)
        if accumulated_grads is None:
            accumulated_grads = grads
        else:
            accumulated_grads = mut.tree_map(lambda ag, g: mx.add(ag, g), accumulated_grads, grads)
            
    optimizer.update(model, accumulated_grads)
    # Single evaluation for the entire step
    mx.eval(model.parameters(), optimizer.state)
    
    dt = time.time() - start
    print(f"GPU Step Time (pure compute): {dt:.2f} seconds")
    tokens = config.micro_batch_size * config.block_size * config.gradient_accumulation_steps
    print(f"GPU Throughput: {tokens / dt:.0f} tokens/sec")

if __name__ == "__main__":
    test_gpu_speed()
