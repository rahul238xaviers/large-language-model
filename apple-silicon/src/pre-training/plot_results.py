import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path

def plot_run(run_dir):
    run_path = Path(run_dir)
    metrics_file = run_path / "metrics.csv"
    
    if not metrics_file.exists():
        print(f"Error: Could not find {metrics_file}")
        return

    df = pd.read_csv(metrics_file)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f"Training Progress: {run_path.name}", fontsize=16)

    # 1. Loss Curve
    axes[0, 0].plot(df['step'], df['train_loss'], label='Train Loss', color='blue')
    axes[0, 0].set_title('Loss vs. Iterations')
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # 2. Throughput (Tokens/sec)
    axes[0, 1].plot(df['step'], df['tokens_per_sec'], label='tok/s', color='green')
    axes[0, 1].set_title('Throughput')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Tokens / Second')
    axes[0, 1].grid(True, alpha=0.3)

    # 3. Memory Usage
    axes[1, 0].plot(df['step'], df['vram_usage_gb'], label='VRAM (GB)', color='red')
    axes[1, 0].set_title('Memory Utilization')
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('GB')
    axes[1, 0].grid(True, alpha=0.3)

    # 4. Learning Rate
    axes[1, 1].plot(df['step'], df['learning_rate'], label='LR', color='purple')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('LR')
    axes[1, 1].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0.03, 1, 0.95))
    
    output_path = run_path / "training_curves.png"
    plt.savefig(output_path)
    print(f"Plot saved to {output_path}")
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        plot_run(sys.argv[1])
    else:
        # Try to find the latest run
        runs = sorted(list(Path("runs").glob("run_*")))
        if runs:
            plot_run(runs[-1])
        else:
            print("No runs found in ./runs/")
