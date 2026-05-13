import json

with open('gpt_training.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = cell.get('source', [])
        if not source:
            continue
        
        source_text = "".join(source)
        
        old_device_code = (
            "if torch.cuda.is_available():\n"
            "    device = 'cuda' \n"
            "    # Speed up matrix multiplications on NVIDIA hardware with Tensor Cores\n"
            "    torch.set_float32_matmul_precision('high')\n"
            "else:\n"
            "    device = 'cpu'\n"
        )
        
        new_device_code = (
            "if torch.cuda.is_available():\n"
            "    device = 'cuda'\n"
            "    torch.set_float32_matmul_precision('high')\n"
            "elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():\n"
            "    device = 'mps'\n"
            "else:\n"
            "    device = 'cpu'\n"
        )

        if old_device_code in source_text:
            source_text = source_text.replace(old_device_code, new_device_code)
            source_text = source_text.replace(
                "# Optimized Hyperparameters for a 4GB GPU",
                "# Optimized Hyperparameters for Mac Studio (Apple Silicon)"
            )
            source_text = source_text.replace(
                "batch_size = 16       # Increased from 4 for better GPU utilization",
                "batch_size = 64       # Increased for Mac Studio unified memory"
            )
            
            cell['source'] = source_text.splitlines(keepends=True)

with open('gpt_training_mac_studio.ipynb', 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=2)
    # add trailing newline to match usual format
    f.write("\n")

print("Created gpt_training_mac_studio.ipynb")
