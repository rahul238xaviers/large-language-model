import base64
import json
import sys

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~")+1)) + list(range(ord("¡"), ord("¬")+1)) + list(range(ord("®"), ord("ÿ")+1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}

def convert_tiktoken_to_hf(tiktoken_path, output_json_path):
    print(f"Reading {tiktoken_path}...")
    with open(tiktoken_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    
    vocab_ranks = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split(" ")
        if len(parts) != 2:
            continue
        token_bytes = base64.b64decode(parts[0])
        rank = int(parts[1])
        vocab_ranks[token_bytes] = rank
        
    print("Sorting vocabulary and preparing base bytes...")
    sorted_vocab = sorted(vocab_ranks.items(), key=lambda x: x[1])
    
    vocab_ranks_restricted = {}
    for token_bytes, rank in sorted_vocab:
        if len(token_bytes) == 1:
            vocab_ranks_restricted[token_bytes] = rank

    byte_encoder = bytes_to_unicode()
    merges = []
    
    print("Reconstructing BPE merges...")
    for token_bytes, rank in sorted_vocab:
        if len(token_bytes) <= 1:
            continue
        
        # Tokenize token_bytes using restricted vocabulary of lower ranks
        parts = [bytes([b]) for b in token_bytes]
        while len(parts) > 1:
            min_rank = float('inf')
            best_idx = -1
            for i in range(len(parts) - 1):
                pair = parts[i] + parts[i+1]
                pair_rank = vocab_ranks_restricted.get(pair, float('inf'))
                if pair_rank < min_rank:
                    min_rank = pair_rank
                    best_idx = i
            if min_rank == float('inf'):
                break
            parts[best_idx] = parts[best_idx] + parts[best_idx+1]
            parts.pop(best_idx+1)
            
        if len(parts) == 2:
            left = "".join(byte_encoder[b] for b in parts[0])
            right = "".join(byte_encoder[b] for b in parts[1])
            merges.append(f"{left} {right}")
            
        vocab_ranks_restricted[token_bytes] = rank
        
    print("Constructing Hugging Face format mapping...")
    hf_vocab = {}
    for token_bytes, rank in sorted_vocab:
        encoded_token = "".join(byte_encoder[b] for b in token_bytes)
        hf_vocab[encoded_token] = rank
        
    hf_tokenizer = {
      "version": "1.0",
      "truncation": None,
      "padding": None,
      "added_tokens": [
        {
          "id": 100257,
          "content": "<|endoftext|>",
          "single_word": False,
          "lstrip": False,
          "rstrip": False,
          "normalized": False,
          "special": True
        },
        {
          "id": 100258,
          "content": "<|fim_prefix|>",
          "single_word": False,
          "lstrip": False,
          "rstrip": False,
          "normalized": False,
          "special": True
        },
        {
          "id": 100259,
          "content": "<|fim_middle|>",
          "single_word": False,
          "lstrip": False,
          "rstrip": False,
          "normalized": False,
          "special": True
        },
        {
          "id": 100260,
          "content": "<|fim_suffix|>",
          "single_word": False,
          "lstrip": False,
          "rstrip": False,
          "normalized": False,
          "special": True
        },
        {
          "id": 100276,
          "content": "<|detokenizer_all_special_tokens|>",
          "single_word": False,
          "lstrip": False,
          "rstrip": False,
          "normalized": False,
          "special": True
        }
      ],
      "normalizer": None,
      "pre_tokenizer": {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": False,
        "use_regex": True
      },
      "post_processor": {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": False,
        "use_regex": True
      },
      "decoder": {
        "type": "ByteLevel",
        "add_prefix_space": False,
        "trim_offsets": False,
        "use_regex": True
      },
      "model": {
        "type": "BPE",
        "dropout": None,
        "unk_token": None,
        "continuing_subword_prefix": None,
        "end_of_word_suffix": None,
        "fuse_unk": False,
        "vocab": hf_vocab,
        "merges": merges
      }
    }
    
    print(f"Writing to {output_json_path}...")
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(hf_tokenizer, f, indent=2)
    print("Conversion complete!")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python convert_tiktoken.py <input.tiktoken> <output.json>")
        sys.exit(1)
    convert_tiktoken_to_hf(sys.argv[1], sys.argv[2])
