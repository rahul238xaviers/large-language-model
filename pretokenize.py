import os
import glob
import tiktoken
import pandas as pd
import numpy as np
from tqdm import tqdm

def main():
    data_dir = "data/datasets/rust/"
    out_path = os.path.join(data_dir, "train.bin")
    
    parquet_files = sorted(glob.glob(os.path.join(data_dir, "train-*.parquet")))
    print(f"Found {len(parquet_files)} parquet files to tokenize.")
    
    print("Loading tiktoken 'cl100k_base'...")
    enc = tiktoken.get_encoding("cl100k_base")
    eot_token = enc.eot_token if hasattr(enc, 'eot_token') else 100257
    
    all_tokens = []
    
    for parquet_path in parquet_files:
        print(f"Loading parquet from {parquet_path}...")
        df = pd.read_parquet(parquet_path)
        
        text_col = 'content' if 'content' in df.columns else 'text'
        if text_col not in df.columns:
            for c in df.columns:
                if df[c].dtype == object:
                    text_col = c
                    break
        
        texts = df[text_col].dropna().tolist()
        print(f"Tokenizing {len(texts)} documents from {os.path.basename(parquet_path)}...")
        
        for text in tqdm(texts):
            tokens = enc.encode(text, allowed_special={"<|endoftext|>"})
            tokens.append(eot_token)
            all_tokens.extend(tokens)
            
    print(f"Total tokens generated: {len(all_tokens)}")
    
    print(f"Saving to {out_path} as uint32...")
    tokens_np = np.array(all_tokens, dtype=np.uint32)
    tokens_np.tofile(out_path)
    
    print("Done!")

if __name__ == "__main__":
    main()
