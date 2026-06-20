import os
import sys
import glob
import pyarrow.parquet as pq
import tiktoken
import multiprocessing as mp

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '../../src/pre-training'))

def process_file(file_path):
    encoder = tiktoken.get_encoding("cl100k_base")
    unique_tokens = set()
    try:
        table = pq.read_table(file_path, columns=["content"])
        contents = table.column("content").to_pylist()
        for content in contents:
            if content:
                unique_tokens.update(encoder.encode_ordinary(content))
    except Exception as e:
        print(f"Error processing {os.path.basename(file_path)}: {e}")
    return unique_tokens

def test_vocab_coverage():
    # Find dataset path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    local_data_path = os.path.join(script_dir, "../../../data/datasets/rust")
    local_files = sorted(glob.glob(os.path.join(local_data_path, "*.parquet")))
    
    assert len(local_files) > 0, f"No training parquet files found in {local_data_path}!"
    
    print(f"\nFound {len(local_files)} training files.")
    print("Running parallel vocabulary coverage analysis across all files...")
    
    num_cores = min(mp.cpu_count(), 8)
    print(f"Using {num_cores} cores.")
    
    unique_tokens = set()
    from tqdm import tqdm
    with mp.Pool(processes=num_cores) as pool:
        iterator = pool.imap_unordered(process_file, local_files)
        for res in tqdm(iterator, total=len(local_files), desc="Analyzing vocabulary coverage"):
            unique_tokens.update(res)
        
    total_vocab = 100277
    coverage_pct = len(unique_tokens) / total_vocab * 100
    
    print(f"\n--- Vocabulary Coverage Results ---")
    print(f"Total unique tokens found in dataset: {len(unique_tokens)} / {total_vocab}")
    print(f"Coverage of cl100k_base vocabulary: {coverage_pct:.2f}%")
    
    assert len(unique_tokens) > 0

if __name__ == "__main__":
    test_vocab_coverage()
