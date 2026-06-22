import os
import sys
import shutil
from pathlib import Path
from huggingface_hub import hf_hub_download

def load_env_file(env_path):
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    key, val = line.split("=", 1)
                    os.environ[key.strip()] = val.strip().strip('"').strip("'")

def main():
    # Load environment variables (to get HF_TOKEN if set)
    python_root = Path(__file__).resolve().parent.parent
    repo_root = python_root.parent
    load_env_file(repo_root / "env" / ".env")
    
    hf_token = os.environ.get("HF_TOKEN")
    
    output_dir = repo_root / "data/raw_chunks/bigcode/the-stack-v2/data/c++"
    
    # Clean the directory to remove existing files before starting fresh
    if output_dir.exists():
        print(f"Cleaning existing files in {output_dir}...")
        for item in output_dir.iterdir():
            if item.is_file() or item.is_symlink():
                os.remove(item)
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Download all 214 shards of the-stack (v1) C++ split
    # Each shard has documents with the actual raw code in the "content" column
    total_shards = 214
    print(f"Downloading all {total_shards} shards of C++ source files (The Stack v1)...")
    
    for i in range(total_shards):
        local_name = f"train-{i:05d}-of-00214.parquet"
        filename = f"data/c++/{local_name}"
        dst_path = output_dir / local_name
        
        print(f"\n[{i+1}/{total_shards}] Downloading {filename} from HF...")
        try:
            cached_path = hf_hub_download(
                repo_id="bigcode/the-stack",
                repo_type="dataset",
                filename=filename,
                token=hf_token
            )
            
            print(f"Copying {local_name} to local workspace...")
            shutil.copy(cached_path, dst_path)
            print(f"Successfully saved locally: {dst_path}")
            
        except Exception as e:
            print(f"Error downloading shard {i}: {e}")
            sys.exit(1)
            
    print("\nAll 214 C++ parquet files with raw code content downloaded and saved locally!")
    # Force clean exit to prevent any library destructor hang
    os._exit(0)

if __name__ == "__main__":
    main()
