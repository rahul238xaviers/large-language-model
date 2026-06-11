#!/usr/bin/env python3
"""High-performance dataset processor for RAG SFT training with LLM-as-a-Judge and parallel workers.

Optimizations:
- Connection pooling for Ollama API
- Batch processing with progress bars
- LRU caching for repeated operations
- Asynchronous I/O with semaphore control
- Memory-efficient streaming for large datasets
"""

import argparse
import asyncio
import json
import os
import random
import sys
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from functools import lru_cache, partial
import signal

import aiohttp
import orjson
from tqdm import tqdm
from datasets import load_dataset

# Add repository root to path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

# Default configuration
DEFAULT_CONFIG = {
    "sources": [
        {
            "name": "Convence/Rust-Coder",
            "cache_dir": "data/datasets/rust_coder_cache",
            "field_mapping": {
                "instruction": "instruction",
                "context": "explanation",
                "response": "code"
            }
        },
        {
            "name": "matteopilotto/rust-github-issues",
            "cache_dir": "data/datasets/rust_github_issues_cache",
            "field_mapping": {
                "instruction": "title",
                "context": "body"
            }
        }
    ],
    "output_dir": "data/processed",
    "seed": 42,
    "train_ratio": 0.85
}

class OllamaClient:
    """High-performance async Ollama client using aiohttp with connection pooling and retries."""
    
    def __init__(self, base_url: str, timeout: int = 300, max_retries: int = 3):
        self.base_url = base_url.rstrip('/')
        self.timeout = timeout
        self.max_retries = max_retries
        
        # Configure TCP connector with connection pooling limits
        self.connector = aiohttp.TCPConnector(
            limit=50,
            limit_per_host=20,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        self.session = aiohttp.ClientSession(
            connector=self.connector,
            timeout=aiohttp.ClientTimeout(total=timeout)
        )
    
    async def generate(
        self, 
        model: str, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        repetition_penalty: float = 1.0
    ) -> str:
        """Asynchronously generate response from Ollama with native async/await and retries."""
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "repetition_penalty": repetition_penalty
            }
        }
        if system_prompt:
            payload["system"] = system_prompt
        
        url = f"{self.base_url}/api/generate"
        
        for attempt in range(self.max_retries + 1):
            try:
                async with self.session.post(
                    url,
                    data=orjson.dumps(payload),
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status in [429, 500, 502, 503, 504]:
                        if attempt < self.max_retries:
                            backoff = 0.5 * (2 ** attempt)
                            await asyncio.sleep(backoff)
                            continue
                    response.raise_for_status()
                    
                    resp_bytes = await response.read()
                    resp_json = orjson.loads(resp_bytes)
                    return resp_json.get("response", "").strip()
            except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                if attempt < self.max_retries:
                    backoff = 0.5 * (2 ** attempt)
                    await asyncio.sleep(backoff)
                    continue
                raise
                
        raise RuntimeError(f"Failed to generate after {self.max_retries} attempts.")
    
    async def close(self):
        """Clean up ClientSession and connector resources."""
        if not self.session.closed:
            await self.session.close()
        await self.connector.close()

class RateLimiter:
    """Thread/async-safe token bucket rate limiter for API calls."""
    
    def __init__(self, max_calls_per_second: int = 10):
        self.max_calls = max_calls_per_second
        self.calls = []
        self.lock = asyncio.Lock()
    
    async def acquire(self):
        """Acquire a token, waiting if necessary under an async lock."""
        async with self.lock:
            now = datetime.now().timestamp()
            # Remove calls older than 1 second
            self.calls = [t for t in self.calls if t > now - 1]
            
            if len(self.calls) >= self.max_calls:
                wait_time = 1.0 - (now - self.calls[-self.max_calls])
                if wait_time > 0:
                    await asyncio.sleep(wait_time)
                    now = datetime.now().timestamp()
            
            self.calls.append(now)

def load_config_file(config_path: str) -> Dict:
    """Load configuration from JSON or YAML file using orjson/yaml."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(path, "rb") as f:
        content = f.read()
        if path.suffix.lower() in (".yaml", ".yml"):
            import yaml
            return yaml.safe_load(content)
        return orjson.loads(content)

def load_records(dataset_name: str, cache_dir: str, limit: int = -1, offset: int = 0) -> List[Dict]:
    """Memory-efficient loader for local datasets."""
    cache_path = Path(cache_dir)
    if not cache_path.is_absolute():
        cache_path = repo_root / cache_path
    
    print(f"📂 Loading {dataset_name} from {cache_path} (limit={limit}, offset={offset})...")
    dataset = load_dataset(dataset_name, cache_dir=str(cache_path))
    
    records = []
    if isinstance(dataset, dict):
        for split_name, split in dataset.items():
            print(f"  - Split: {split_name}")
            if limit > 0:
                end_idx = min(offset + limit, len(split))
                if offset < len(split):
                    selected = split.select(range(offset, end_idx))
                    records.extend(list(selected))
            else:
                if offset > 0:
                    selected = split.select(range(offset, len(split)))
                    records.extend(list(selected))
                else:
                    records.extend(list(split))
    else:
        if limit > 0:
            end_idx = min(offset + limit, len(dataset))
            if offset < len(dataset):
                selected = dataset.select(range(offset, end_idx))
                records.extend(list(selected))
        else:
            if offset > 0:
                selected = dataset.select(range(offset, len(dataset)))
                records.extend(list(selected))
            else:
                records.extend(list(dataset))
    
    return records


@lru_cache(maxsize=1000)
def extract_json_cached(text: str) -> Dict:
    """Cached JSON extraction for repeated patterns using orjson and json fallback."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    
    clean_text = text.strip()
    try:
        return orjson.loads(clean_text)
    except Exception:
        # Fallback to standard json loads with strict=False to handle control characters (like newlines) in strings
        return json.loads(clean_text, strict=False)

def sanitize_github_issue_context(context: str) -> str:
    """Strip HTML comments, template instructions, and placeholder reminders from the issue body."""
    # Remove HTML comments
    context = re.sub(r'<!--.*?-->', '', context, flags=re.DOTALL)
    
    # Remove common template lines/reminders
    lines = []
    for line in context.splitlines():
        # Strip lines containing XXX template placeholders
        if "XXX" in line:
            continue
        # Strip lines that match typical comments or templates
        stripped = line.strip()
        if stripped.startswith("Include each step required to complete") or \
           stripped.startswith("Include a list of all the PRs") or \
           stripped.startswith("Thank you for creating a tracking issue") or \
           stripped.startswith("Remember to add team labels to the tracking issue") or \
           stripped.startswith("For a language team feature, this would e.g., be") or \
           stripped.startswith("Such a feature should also be labeled with") or \
           stripped.startswith("This label is used to associate issues") or \
           stripped.startswith("Unresolved Questions") and "list all the" in line:
            continue
        lines.append(line)
        
    return "\n".join(lines).strip()

def extract_field_xml(text: str, tag: str) -> Optional[str]:
    """Extract content between XML tags, returning stripped string if found."""
    # Handle case where tag is opened but closing tag is missing (e.g. stop sequence matched)
    if f"<{tag}>" in text and f"</{tag}>" not in text:
        text = text + f"</{tag}>"
    pattern = rf"<{tag}>(.*?)</{tag}>"
    match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

class SFTDataProcessor:
    """Optimized processor for RAG SFT data."""
    
    def __init__(
        self,
        # Models definition - Downscaled for local 40+ tokens/sec execution speed
        gen_model: str = "qwen2.5-coder:14b-instruct",    # High-throughput formatting engine
        fallback_model: str = "qwen3.6:35b-a3b-coding-mxfp8",
        judge_model: str = "qwen2.5:14b-instruct",
        ollama_base_url: str = "http://localhost:11434",
        max_concurrent: int = 10,
        rate_limit: int = 20
    ):
        self.gen_model = gen_model
        self.fallback_model = fallback_model
        self.judge_model = judge_model
        self.ollama_client = OllamaClient(ollama_base_url)
        self.rate_limiter = RateLimiter(rate_limit)
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
    
    async def generate_with_fallback(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        repetition_penalty: float = 1.0
    ) -> str:
        """Generate with automatic fallback and rate limiting."""
        await self.rate_limiter.acquire()
        
        salt = str(uuid.uuid4())
        salt_system_prompt = f"System ID: {salt}\n{system_prompt}" if system_prompt else None
        salt_prompt = f"Prompt ID: {salt}\n{prompt}"
        
        try:
            return await self.ollama_client.generate(
                self.gen_model, salt_prompt, salt_system_prompt, temperature, repetition_penalty
            )
        except Exception as e:
            print(f"  ⚠️ Generator failed: {e}. Trying fallback...")
            await self.rate_limiter.acquire()
            try:
                return await self.ollama_client.generate(
                    self.fallback_model, salt_prompt, salt_system_prompt, temperature, repetition_penalty
                )
            except Exception as fallback_err:
                print(f"  ❌ Fallback also failed: {fallback_err}")
                raise
    
    async def process_rust_coder_row(
        self, 
        instruction: str, 
        context: str, 
        response: str,
        row_idx: int,
        total_rows: int
    ) -> Optional[Tuple[str, str]]:
        """Process Rust-Coder dataset row with LLM refinement."""
        gen_prompt = f"""Analyze the following Rust training entry for low-entropy boilerplate or repetitive placeholder templates (such as `let x = 42;`, `println!("Value: {{}}", x);`, or generic empty loops).

Instruction: {instruction}
Original Explanation: {context}
Original Code:
{response}

Rules:
1. FORMAT CONSTRAINT: Your output structure must strictly match the requested file format. If the instruction references a configuration format (e.g., Cargo.toml), you must return raw, valid TOML blocks inside the "response" field, NOT Rust code files.
2. NO SIMULATED OR DUMMY LOGIC: Every code example must be a fully functional, logical, and complete implementation of a realistic operation.
   - Do NOT simulate conditions by writing `panic!("error")` or `assert!(false)` directly in the main path or tests. Instead, execute the logical code path that naturally triggers the condition (e.g., attempt to read past the end of a custom buffer).
   - If calling C functions or FFI, write real, correct signatures and pass actual pointers/data correctly (e.g., if using `libc::printf`, pass both the format string pointer and the actual data pointer correctly).
   - Do NOT write placeholder logic (such as `let result = value * 2;` or empty loops) to satisfy an instruction. Write a complete, realistic function that performs a real operation.
3. EXPLANATION INTEGRITY: You must provide a comprehensive, conceptual explanation of how the feature works inside the "context" field. Do not skip the explanation.
4. IF the entry is ALREADY clean, idiomatic, highly detailed, AND strictly relevant to the instruction, you MUST return the original explanation and code completely unaltered.
5. VERBOSITY BUDGET: Provide exactly ONE concise, production-ready implementation example. Do NOT generate redundant architectural variations or boilerplate structural permutations. Keep your output focused and under 800 tokens.
6. NO DUMMY LOGIC: Do NOT write empty `fn main()`, do NOT use dummy print statements like `println!("Value: {{}}", x);` or `println!("Hello, World!");` as placeholders. However, if the instruction explicitly asks to demonstrate printing, logging, or debugging, you may use standard print/debug macros (like println!, eprintln!, dbg!, or log!) to show the functionality, ensuring they output meaningful runtime values. Do NOT use dummy variable assignments like `let x = 42;`, `10`, `1`, or `0` as placeholders. Avoid hardcoded dummy strings in your code examples; if inputs/filenames are needed, pass them as arguments to the function.
7. COMPILE CLEAN: The code/config must be syntactically valid and compilation-ready. Check for matching quotes, correct braces, and ensure methods are called only on types that support them.
8. NO C CODE: If the instruction asks for FFI or calling C functions, you must write Rust declarations (`extern "C"`) and safe Rust wrappers. Never output C code. Do NOT link to fictional C libraries; link only to standard system libraries (e.g., `#[link(name = "m")]` or `libc`) or declare them realistically without dummy attribute placeholders.
9. EMBEDDED CONSTRAINTS: If the instruction mentions "embedded system", "no_std", or strict memory constraints, you must write code compatible with `#![no_std]`.
   - Remember that `Rc` and `Arc` require a heap allocator and are located in the `alloc` crate (`alloc::rc::Rc`), NOT `core::rc::Rc`. You must use `extern crate alloc;` and show proper allocator usage, or demonstrate stack/static interior mutability alternatives (like `core::cell::RefCell` with static variables or safe concurrency locks).
10. Your output must be wrapped in XML tags <context>...</context> and <response>...</response>. Do not wrap in markdown backticks or add any other text outside these tags.

<context>
[explanation text here]
</context>
<response>
[code/config block here]
</response>"""
        
        try:
            res_raw = await self.generate_with_fallback(
                gen_prompt,
                system_prompt="You are an expert Rust code refactoring engine that outputs only valid XML.",
                temperature=0.1,
                repetition_penalty=1.0
            )
            
            # Try XML tag extraction first
            new_context = extract_field_xml(res_raw, "context")
            new_response = extract_field_xml(res_raw, "response")
            
            # Fallback to JSON if XML tags are missing
            if not new_context or not new_response:
                try:
                    parsed = extract_json_cached(res_raw)
                    new_context = parsed.get("context")
                    new_response = parsed.get("response")
                except Exception:
                    pass
            
            if not new_context or not new_response:
                print(f"  [Row {row_idx}/{total_rows}] ⚠️ Empty fields, skipping")
                return None
            
            return new_context, new_response
        except Exception as e:
            print(f"  [Row {row_idx}/{total_rows}] ❌ Processing failed: {e}")
            return None
    
    async def process_github_issues_row(
        self,
        instruction: str,
        context: str,
        row_idx: int,
        total_rows: int
    ) -> Optional[str]:
        """Process GitHub issues dataset row."""
        gen_prompt = f"""You are a senior Rust core compiler maintainer. We are processing an official Rust compiler tracking issue, RFC implementation text, or architectural debate.

Issue Title: {instruction}
Issue Body:
{context}

Based on this issue context, write a detailed, highly accurate, and production-ready Rust technical roadmap or documentation document.

CRITICAL INSTRUCTIONS:
1. DO NOT implement, invent, or include fictional Rust source code, attributes, or signatures (such as `#[abi("wasm")]` or `#[feature(...)])` for core compiler feature gates, custom compiler types, or unstable features. You are compiling a high-level tracking or documentation document, NOT writing or demonstrating the feature logic itself. Stick strictly to stable standard Rust concepts or the exact names/syntax provided in the Context.
2. Restrict your output to an actionable roadmap, documentation steps, or a rigorous evaluation of the unresolved questions established directly within the Context.
3. Apply a strict verbosity budget: Keep your explanation high-entropy and under 800 tokens total. No boilerplate repetitions.
4. Output your response as clean, raw Markdown.
5. DO NOT repeat headers, lists, or sections. The output must be a clean, non-repetitive, linear markdown document.
6. NO PLACEHOLDERS: Do NOT copy verbatim template placeholders, HTML comments, or template reminders (such as 'XXX --- list all the unresolved questions') from the issue body. Either resolve them or omit them completely. Do NOT use fake code comments or ellipsis stubs.
7. CONCLUSION constraint: End the document cleanly and concisely. Once you have covered the necessary roadmap items, stop generating immediately to prevent looping."""
        
        try:
            return await self.generate_with_fallback(
                gen_prompt,
                system_prompt="You are a senior Rust core maintainer who writes production-ready Rust code.",
                temperature=0.1,
                repetition_penalty=1.1
            )
        except Exception as e:
            print(f"  [Row {row_idx}/{total_rows}] ❌ Generation failed: {e}")
            return None
    
    async def judge_sample(
        self,
        instruction: str,
        context: str,
        response: str,
        row_idx: int,
        total_rows: int
    ) -> bool:
        """LLM-as-a-Judge validation."""
        judge_prompt = f"""You are an SFT training data auditor. Evaluate the following RAG SFT training sample against these explicit quality criteria:
1. Crucial Constraint: The Response MUST NOT contain placeholder logic, dummy code stubs (e.g., `let x = 42;`, `println!("Value: {{}}", x);`), or generic syntax-free placeholders. Note: A clean, complete, and functional library function, struct, enum, module, or unit test is NOT a dummy code stub even if it is short. Only code containing unfinished stubs (like `todo!`, `unimplemented!`, ellipsis `...`), dummy variables (like `let x = 42;`), or placeholder comments is considered a dummy code stub.
2. RAG Grounding: The Response must strictly answer the Instruction using the facts established within the Context block.

RAG Sample to Audit:
---
### Instruction:
{instruction}

### Context:
{context}

### Response:
{response}
---

Output your evaluation in this exact JSON format:
{{"decision": "PASS" or "FAIL", "reason": "short explanation"}}"""
        
        try:
            salt = str(uuid.uuid4())
            judge_res_raw = await self.ollama_client.generate(
                self.judge_model,
                f"Prompt ID: {salt}\n{judge_prompt}",
                system_prompt=f"System ID: {salt}\nYou are a strict code quality auditor that outputs only valid JSON.",
                temperature=0.1,
                repetition_penalty=1.0
            )
            parsed = extract_json_cached(judge_res_raw)
            decision = parsed.get("decision", "FAIL").upper()
            reason = parsed.get("reason", "unknown")
            
            if decision == "PASS":
                print(f"  [Row {row_idx}/{total_rows}] ✅ PASS: {reason[:50]}")
                return True
            else:
                print(f"  [Row {row_idx}/{total_rows}] ❌ FAIL: {reason[:50]}")
                return False
        except Exception as e:
            print(f"  [Row {row_idx}/{total_rows}] ⚠️ Judge failed: {e}")
            return False
    
    async def process_row(
        self,
        row: Dict,
        source_name: str,
        inst_col: str,
        ctx_col: str,
        resp_col: str,
        row_idx: int,
        total_rows: int
    ) -> Optional[Dict]:
        """Process a single row with full pipeline."""
        async with self.semaphore:
            try:
                instruction = row.get(inst_col) or ""
                context = row.get(ctx_col) or ""
                response = row.get(resp_col) or ""
                
                # Step 1: Generate/refine response
                if source_name == "matteopilotto/rust-github-issues":
                    # Sanitize context programmatically first
                    sanitized_context = sanitize_github_issue_context(context)
                    generated_response = await self.process_github_issues_row(
                        instruction, sanitized_context, row_idx, total_rows
                    )
                    if not generated_response:
                        return None
                    context = sanitized_context
                    response = generated_response
                
                elif source_name == "Convence/Rust-Coder":
                    result = await self.process_rust_coder_row(
                        instruction, context, response, row_idx, total_rows
                    )
                    if not result:
                        return None
                    context, response = result
                
                # Step 2: Judge validation
                passed = await self.judge_sample(
                    instruction, context, response, row_idx, total_rows
                )
                
                if not passed:
                    return None
                
                # Step 3: Format output
                formatted_text = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}<|endoftext|>"
                return {"text": formatted_text}
                
            except Exception as e:
                print(f"  [Row {row_idx}/{total_rows}] ❌ Unexpected error: {e}")
                return None
    
    async def close(self):
        await self.ollama_client.close()

async def process_source_async(
    source: Dict,
    processor: SFTDataProcessor,
    limit: int,
    offset: int,
    workers: int
) -> List[Dict]:
    """Process a single data source asynchronously."""
    name = source.get("name")
    cache_dir = source.get("cache_dir")
    mapping = source.get("field_mapping", {})
    
    if not name or not cache_dir:
        print(f"⚠️ Skipping invalid source: {source}")
        return []
    
    print(f"\n{'─'*50}")
    print(f"📦 Processing Source: {name}")
    
    # Load records without streaming logic
    records = load_records(name, cache_dir, limit, offset)
    if not records:
        print(f"⚠️ No records loaded from {name}")
        return []
    
    inst_col = mapping.get("instruction", "instruction")
    ctx_col = mapping.get("context", "context")
    resp_col = mapping.get("response", "response")
    
    # Create tasks with progress bar
    tasks = []
    for idx, row in enumerate(records):
        task = processor.process_row(
            row, name, inst_col, ctx_col, resp_col,
            idx + 1, len(records)
        )
        tasks.append(task)
    
    # Execute with controlled concurrency
    results = []
    with tqdm(total=len(tasks), desc=f"Processing {name}", unit="rows") as pbar:
        for coro in asyncio.as_completed(tasks):
            result = await coro
            if result:
                results.append(result)
            pbar.update(1)
    
    print(f"\n✨ Validated {len(results)}/{len(records)} records for '{name}'")
    return results

def parse_args():
    parser = argparse.ArgumentParser(description="High-performance RAG SFT Dataset Processor.")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--output-dir", "-o", help="Override output directory")
    parser.add_argument("--seed", "-s", type=int, help="Random seed")
    parser.add_argument("--train-ratio", "-r", type=float, help="Training ratio (default: 0.85)")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Limit records per source (-1 for all)")
    parser.add_argument("--offset", type=int, default=0, help="Offset to start reading records from")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Max concurrent workers")
    parser.add_argument("--rate-limit", type=int, default=20, help="API calls per second")
    return parser.parse_args()

async def main_async():
    """Main async entry point."""
    args = parse_args()
    
    # Load configuration
    if args.config:
        print(f"🛠️ Loading config from {args.config}")
        config = load_config_file(args.config)
    else:
        print("ℹ️ Using default configuration")
        config = DEFAULT_CONFIG
    
    # Apply overrides
    output_dir = args.output_dir or config.get("output_dir", "data/processed")
    run_dir = datetime.now().strftime("%Y%m%d_%H%M%S")
    seed = args.seed or config.get("seed", 42)
    train_ratio = args.train_ratio or config.get("train_ratio", 0.85)
    limit = args.limit
    offset = args.offset
    workers = min(args.workers, 50)  # Cap at 50
    rate_limit = args.rate_limit
    
    ollama_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    print(f"🔌 Ollama: {ollama_url} | Workers: {workers} | Rate: {rate_limit}/s")
    
    # Initialize processor
    processor = SFTDataProcessor(
        ollama_base_url=ollama_url,
        max_concurrent=workers,
        rate_limit=rate_limit
    )
    
    try:
        # Process all sources
        all_results = []
        for source in config.get("sources", []):
            results = await process_source_async(source, processor, limit, offset, workers)
            all_results.extend(results)
        
        print(f"\n📊 Total validated records: {len(all_results)}")
        if not all_results:
            print("❌ No valid data, exiting")
            sys.exit(1)
        
        # Shuffle and split
        print(f"🎲 Shuffling with seed {seed}")
        random.seed(seed)
        random.shuffle(all_results)
        
        split_idx = int(len(all_results) * train_ratio)
        train_data = all_results[:split_idx]
        valid_data = all_results[split_idx:]
        
        print(f"📝 Split: {len(train_data)} train / {len(valid_data)} valid")
        
        # Save outputs
        output_path = Path(output_dir, run_dir)
        if not output_path.is_absolute():
            output_path = repo_root / output_path
        output_path.mkdir(parents=True, exist_ok=True)
        
        train_file = output_path / "train_sft.jsonl"
        valid_file = output_path / "valid_sft.jsonl"
        
        print(f"💾 Saving to {output_path}")
        
        # Batch write for performance using orjson in binary mode
        with open(train_file, "wb") as f:
            for item in train_data:
                f.write(orjson.dumps(item) + b"\n")
        
        with open(valid_file, "wb") as f:
            for item in valid_data:
                f.write(orjson.dumps(item) + b"\n")
        
        print(f"✅ Success! Train: {train_file.name}, Valid: {valid_file.name}")
        
    finally:
        await processor.close()

def main():
    """Synchronous entry point with signal handling."""
    # Handle Ctrl+C gracefully
    signal.signal(signal.SIGINT, lambda sig, frame: sys.exit(0))
    
    # Run async main
    asyncio.run(main_async())

if __name__ == "__main__":
    main()