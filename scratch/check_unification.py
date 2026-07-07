import os
import re
import json
from pathlib import Path

repo_root = Path("/Users/rahulkumar/dev/large-language-model")
processed_dir = repo_root / "data" / "processed"
unified_audit_path = processed_dir / "unified_sft_dataset" / "evaluation_audit.jsonl"

def main():
    print("=== Analyzing Unified Dataset Merge Status ===")
    
    # 1. Load unified audit records
    unified_keys = set()
    unified_count = 0
    if unified_audit_path.exists():
        with open(unified_audit_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                    # Use same key definition as unify_and_clean_runs.py
                    key = (record.get("instruction", "").strip().lower(), record.get("response", "").strip())
                    unified_keys.add(key)
                    unified_count += 1
                except Exception as e:
                    print(f"Error reading unified audit line: {e}")
    else:
        print("❌ Unified audit file does not exist!")
        return

    print(f"Loaded {unified_count} records from unified_sft_dataset/evaluation_audit.jsonl.")
    print(f"Unique keys in unified dataset: {len(unified_keys)}")

    # 2. Iterate through run directories
    run_folders = [f for f in processed_dir.iterdir() if f.is_dir() and re.match(r'^\d{8}_\d{6}$', f.name)]
    run_folders = sorted(run_folders)
    
    total_run_pass = 0
    total_run_fail = 0
    total_unresolved = 0
    
    print("\nRun folder details:")
    print(f"{'Run Folder':<20} | {'Total':<6} | {'PASS':<6} | {'Merged':<6} | {'Not Merged':<10} | {'Status'}")
    print("-" * 75)
    
    for folder in run_folders:
        audit_file = folder / "evaluation_audit.jsonl"
        if not audit_file.exists():
            print(f"{folder.name:<20} | {'N/A':<6} | {'N/A':<6} | {'N/A':<6} | {'N/A':<10} | Missing evaluation_audit.jsonl")
            continue
            
        pass_records = []
        fail_records = 0
        total_lines = 0
        
        try:
            with open(audit_file, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    total_lines += 1
                    try:
                        record = json.loads(line)
                        if record.get("decision") == "PASS":
                            pass_records.append(record)
                        else:
                            fail_records += 1
                    except Exception as e:
                        pass
        except Exception as e:
            print(f"{folder.name:<20} | Error reading: {e}")
            continue
            
        merged_count = 0
        not_merged_count = 0
        not_merged_examples = []
        
        for r in pass_records:
            key = (r.get("instruction", "").strip().lower(), r.get("response", "").strip())
            # Let's check if there's any matching instruction (even if response differs)
            inst_only = r.get("instruction", "").strip().lower()
            
            # Find if this exact key is in unified_keys
            if key in unified_keys:
                merged_count += 1
            else:
                not_merged_count += 1
                not_merged_examples.append(r)
                
        total_run_pass += len(pass_records)
        total_run_fail += fail_records
        
        status = "OK"
        if len(pass_records) > 0 and merged_count == 0:
            status = "⚠️ NOT MERGED"
        elif not_merged_count > 0:
            status = f"⚠️ {not_merged_count} filtered/deduped"
            
        print(f"{folder.name:<20} | {total_lines:<6} | {len(pass_records):<6} | {merged_count:<6} | {not_merged_count:<10} | {status}")
        
        # If there are not merged but they are NOT just due to deduplication, print some details
        # We can check if their instruction itself is merged
        if not_merged_count > 0:
            real_misses = []
            for r in not_merged_examples:
                inst = r.get("instruction", "").strip().lower()
                # Check if this instruction exists in unified_keys (any response)
                inst_in_unified = any(u[0] == inst for u in unified_keys)
                if not inst_in_unified:
                    real_misses.append(r)
            if real_misses:
                print(f"   ↳ 🚨 REAL MISSES (Instruction not found in unified dataset): {len(real_misses)}")
                for i, rm in enumerate(real_misses[:3]):
                    print(f"      - [{i+1}] {rm.get('instruction')[:80]}...")
            else:
                pass # all differences were due to instruction-level deduplication (a different response was chosen/kept)

    print("\nSummary:")
    print(f"Total PASS records across all folders: {total_run_pass}")
    print(f"Total FAIL records across all folders: {total_run_fail}")
    print(f"Total records in unified dataset: {unified_count}")

if __name__ == "__main__":
    main()
