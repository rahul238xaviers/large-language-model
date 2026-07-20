import os
import tiktoken
import pandas as pd
import numpy as np
from tqdm import tqdm

def main():
    parquet_path = "data/datasets/rust/train-00000-of-00040.parquet"
    out_path = "data/datasets/rust/train.bin"
    
    print(f"Loading parquet from {parquet_path}...")
    df = pd.read_parquet(parquet_path)
    
    text_col = 'content' if 'content' in df.columns else 'text'
    if text_col not in df.columns:
        for c in df.columns:
            if df[c].dtype == object:
                text_col = c
                break
    
    print(f"Using column: '{text_col}'")
    texts = df[text_col].dropna().tolist()
    
    print("Loading tiktoken 'cl100k_base'...")
    enc = tiktoken.get_encoding("cl100k_base")
    eot_token = enc.eot_token if hasattr(enc, 'eot_token') else 100257
    
    print(f"Tokenizing {len(texts)} documents...")
    all_tokens = []
    
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
