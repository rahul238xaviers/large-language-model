import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

def plot_interactive_metrics():
    # 1. Resolve the path relative to the script's own folder (apple-silicon/src/)
    script_dir = Path(__file__).resolve().parent
    apple_silicon_root = script_dir.parent  # Points to apple-silicon/
    
    metrics_path = apple_silicon_root / "runs/run_20260514_183932/metrics.csv"
    if not metrics_path.exists():
        print(f"Error: Metrics file not found at {metrics_path}")
        return

    # 2. Read the training log
    df = pd.read_csv(metrics_path)
    
    # 3. Apply Premium Dark Theme Styling
    plt.style.use('dark_background')
    
    # Create the 2x2 grid layout
    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Rust-GPT 1.6B Training Journey: 1,472 Steps on Apple Silicon (MLX)", 
                 fontsize=16, fontweight='bold', color='#ea580c')

    # Accent colors matching Rust / MLX branding
    color_orange = '#f97316'  # Rust Orange
    color_green = '#10b981'   # Teal / Slate Green
    color_red = '#ef4444'     # Crimson Red
    color_purple = '#8b5cf6'  # Deep Indigo/Purple
    color_grid = '#292524'    # Dark Stone Gray

    # --- Subplot 1: Loss Convergence ---
    axes[0, 0].plot(df['step'], df['train_loss'], label='Training Loss', color=color_orange, linewidth=1.5)
    axes[0, 0].set_title('Loss Convergence (Cross-Entropy)', fontsize=12, fontweight='semibold', pad=10)
    axes[0, 0].set_xlabel('Training Steps', fontsize=10)
    axes[0, 0].set_ylabel('Loss Value', fontsize=10)
    axes[0, 0].grid(True, linestyle='--', color=color_grid, alpha=0.7)
    axes[0, 0].legend(loc='upper right', frameon=True, facecolor='#1c1917', edgecolor='#44403c')

    # --- Subplot 2: Throughput (Tokens/sec) ---
    axes[0, 1].plot(df['step'], df['tokens_per_sec'], label='Throughput', color=color_green, linewidth=1.2)
    # Highlight stable steady-state average throughput
    avg_throughput = df['tokens_per_sec'].mean()
    axes[0, 1].axhline(avg_throughput, color='#64748b', linestyle=':', 
                        label=f'Avg: {avg_throughput:.1f} tok/s')
    axes[0, 1].set_title('Ingestion & Processing Throughput', fontsize=12, fontweight='semibold', pad=10)
    axes[0, 1].set_xlabel('Training Steps', fontsize=10)
    axes[0, 1].set_ylabel('Tokens / Second', fontsize=10)
    axes[0, 1].grid(True, linestyle='--', color=color_grid, alpha=0.7)
    axes[0, 1].legend(loc='lower right', frameon=True, facecolor='#1c1917', edgecolor='#44403c')

    # --- Subplot 3: VRAM Memory Utilization ---
    axes[1, 0].plot(df['step'], df['vram_usage_gb'], label='VRAM Usage', color=color_red, linewidth=2.0)
    # Draw hardware ceiling line to mathematically demonstrate safety headroom
    axes[1, 0].axhline(512.0, color='#b91c1c', linestyle='--', linewidth=1.5, label='M3 Ultra Ceiling (512 GB)')
    axes[1, 0].set_title('Stable Memory Utilization (VRAM)', fontsize=12, fontweight='semibold', pad=10)
    axes[1, 0].set_xlabel('Training Steps', fontsize=10)
    axes[1, 0].set_ylabel('Memory (GB)', fontsize=10)
    axes[1, 0].set_ylim(0, 560)  # Add height to clearly display headroom
    axes[1, 0].grid(True, linestyle='--', color=color_grid, alpha=0.7)
    axes[1, 0].legend(loc='center right', frameon=True, facecolor='#1c1917', edgecolor='#44403c')

    # --- Subplot 4: Compute Efficiency (MFU %) ---
    axes[1, 1].plot(df['step'], df['mfu_pct'], label='Model FLOPs Util (MFU)', color=color_purple, linewidth=1.2)
    axes[1, 1].set_title('Model FLOPs Utilization (Metal Efficiency)', fontsize=12, fontweight='semibold', pad=10)
    axes[1, 1].set_xlabel('Training Steps', fontsize=10)
    axes[1, 1].set_ylabel('Efficiency (%)', fontsize=10)
    axes[1, 1].grid(True, linestyle='--', color=color_grid, alpha=0.7)
    axes[1, 1].legend(loc='lower right', frameon=True, facecolor='#1c1917', edgecolor='#44403c')

    # Tighten layout (using strict float tuple to satisfy type checks) and display naturally
    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
    print("Launching natural Matplotlib interactive diagram window...")
    plt.show()

if __name__ == "__main__":
    plot_interactive_metrics()