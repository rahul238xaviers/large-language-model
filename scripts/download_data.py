#!/usr/bin/env python3
"""Generic dataset downloader using ConfigLoader.

Usage:
    python scripts/download_data.py --profile local-dev
    python scripts/download_data.py --profile local-dev --source rust_stack --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

# Add repository root to path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from dotenv import load_dotenv
load_dotenv(repo_root / "env" / ".env")

from pipeline.orchestration.config_loader import ConfigLoader

def parse_args():
    parser = argparse.ArgumentParser(description="Generic HF dataset downloader.")
    parser.add_argument(
        "--profile", "-p",
        default="local-dev",
        help="Configuration profile to use (default: local-dev)"
    )
    parser.add_argument(
        "--source", "-s",
        help="Name of specific source to download. If not specified, downloads all sources."
    )
    parser.add_argument(
        "--dry-run", "-d",
        action="store_true",
        help="Print what would be downloaded without performing any downloads."
    )
    return parser.parse_args()

def main():
    args = parse_args()
    
    config_dir = repo_root / "configs"
    loader = ConfigLoader(config_dir)
    cfg = loader.load(profile=args.profile)
    
    download_cfg = cfg.get("download", {})
    hf_token = os.environ.get("HF_TOKEN") or download_cfg.get("hf_token")
    cache_dir_str = download_cfg.get("cache_dir", "data/datasets")
    
    # Resolve cache directory relative to repo root
    cache_dir = Path(cache_dir_str)
    if not cache_dir.is_absolute():
        cache_dir = repo_root / cache_dir
        
    print(f"📂 Cache directory resolved to: {cache_dir}")
    if not args.dry_run:
        cache_dir.mkdir(parents=True, exist_ok=True)
        
    sources_cfg = download_cfg.get("sources", [])
    if not sources_cfg:
        print("⚠️ No dataset sources configured.")
        return
        
    target_sources = sources_cfg
    if args.source:
        target_sources = [s for s in sources_cfg if s.get("name") == args.source]
        if not target_sources:
            print(f"❌ Source '{args.source}' not found in configuration.")
            return

    for src_dict in target_sources:
        name = src_dict.get("name")
        repo_id = src_dict.get("repo_id")
        method = src_dict.get("method", "load_dataset")
        repo_type = src_dict.get("repo_type", "dataset")
        subset = src_dict.get("subset")
        split = src_dict.get("split")
        data_dir = src_dict.get("data_dir")
        
        print(f"\n──────────────────────────────────────────────────")
        print(f"📦 Processing Source: {name}")
        print(f"   Repo ID: {repo_id}")
        print(f"   Method:  {method}")
        
        if method == "load_dataset":
            source_cache = cache_dir / f"{name}_cache"
            print(f"   Destination: {source_cache}")
            if args.dry_run:
                print(f"   [Dry Run] Would load dataset '{repo_id}' (subset={subset}, split={split}) using datasets.load_dataset")
            else:
                from datasets import load_dataset
                kwargs = {"cache_dir": str(source_cache)}
                if subset:
                    kwargs["name"] = subset
                if split:
                    kwargs["split"] = split
                if hf_token:
                    kwargs["token"] = hf_token
                print(f"   🚀 Downloading split '{split}'...")
                load_dataset(repo_id, **kwargs)
                print(f"   ✅ Done loading {name}")
                
        elif method == "hf_hub_files":
            source_cache = cache_dir / name
            print(f"   Destination: {source_cache}")
            
            from huggingface_hub import HfApi, hf_hub_download
            api = HfApi()
            
            try:
                print(f"   🔍 Listing files in repository...")
                files = api.list_repo_files(repo_id=repo_id, repo_type=repo_type, token=hf_token or True)
                if data_dir:
                    target_files = [f for f in files if f.startswith(data_dir)]
                else:
                    target_files = files
                    
                if not target_files:
                    print(f"   ⚠️ No files found matching prefix '{data_dir or ''}'")
                    continue
                    
                print(f"   🎯 Found {len(target_files)} files to download.")
                
                if args.dry_run:
                    print(f"   [Dry Run] Would download files:")
                    for f in target_files[:5]:
                        print(f"     - {f}")
                    if len(target_files) > 5:
                        print(f"     - ... and {len(target_files) - 5} more files.")
                else:
                    source_cache.mkdir(parents=True, exist_ok=True)
                    try:
                        from tqdm import tqdm
                        pbar = tqdm(target_files, desc=f"Downloading {name}")
                    except ImportError:
                        pbar = target_files
                        
                    for i, file_path in enumerate(pbar):
                        if isinstance(pbar, list):
                            print(f"   [{i+1}/{len(target_files)}] Downloading {file_path}...")
                        hf_hub_download(
                            repo_id=repo_id,
                            repo_type=repo_type,
                            filename=file_path,
                            local_dir=str(source_cache),
                            token=hf_token or True
                        )
                    print(f"   ✅ Done downloading {name}")
            except Exception as e:
                print(f"   ❌ Error: {e}")
        else:
            print(f"   ❌ Unknown download method: '{method}'")

if __name__ == "__main__":
    main()
