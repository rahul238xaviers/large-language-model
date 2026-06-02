import os
from huggingface_hub import hf_hub_download, HfApi

def download_rust_data():
    from pathlib import Path
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parent.parent  # large-language-model/
    local_dir = str(repo_root / "data" / "datasets")
    repo_id = "bigcode/the-stack"
    data_dir = "data/rust"
    
    print(f"🚀 Initializing download from {repo_id}...")
    api = HfApi()
    
    try:
        # Get list of files in the rust directory
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        rust_files = [f for f in files if f.startswith(data_dir)]
        
        if not rust_files:
            print("❌ No files found in 'data/rust'. Please check your Hugging Face agreement.")
            return

        print(f"📦 Found {len(rust_files)} files to download.")
        
        for i, file_path in enumerate(rust_files):
            print(f"\n[{i+1}/{len(rust_files)}] Downloading {file_path}...")
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=file_path,
                local_dir=local_dir,
                token=True
            )
            
        print(f"\n✅ All files downloaded to {os.path.join(local_dir, data_dir)}")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")

if __name__ == "__main__":
    download_rust_data()
