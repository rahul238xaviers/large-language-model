#!/usr/bin/env python3
"""High-performance dataset processor for MLX-LM/Rapid-MLX SFT training with LLM-as-a-Judge.

Optimizations:
- Native asynchronous I/O with aiohttp and TCP connection pooling
- Concurrency-safe token bucket RateLimiter using asyncio.Lock
- Rust-powered fast JSON parsing and serialization via orjson
- Slicing datasets in memory with Dataset.select() instead of loops
- Output structures compatible with MLX-LM (chat, completions, text)
"""

import argparse
import asyncio
import json
import os
import random
import re
import sys
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from functools import lru_cache, partial
import signal
try:
    import tomllib
except ImportError:
    import toml as tomllib

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

class ConnectionLostError(Exception):
    """Custom exception raised when connection to the model server is permanently lost."""
    pass

class RapidMLXClient:
    """High-performance async Rapid-MLX client using OpenAI-compatible endpoints."""
    
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
    
    async def chat_completion(
        self, 
        model: str, 
        messages: List[Dict[str, str]],
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        repetition_penalty: float = 1.1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ) -> Dict[str, Any]:
        """Asynchronously call chat completions and return content and usage/timing metrics."""
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
            "repetition_penalty": repetition_penalty,
            "frequency_penalty": frequency_penalty,
            "presence_penalty": presence_penalty,
            "user": str(uuid.uuid4())
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        if stop is not None:
            payload["stop"] = stop
        
        url = f"{self.base_url}/chat/completions"
        
        start_time = time.time()
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
                    elapsed = time.time() - start_time
                    resp_json = orjson.loads(resp_bytes)
                    choices = resp_json.get("choices", [])
                    usage = resp_json.get("usage", {})
                    
                    content = ""
                    if choices:
                        content = choices[0].get("message", {}).get("content", "").strip()
                        
                    return {
                        "content": content,
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "elapsed": elapsed
                    }
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
        try:
            return json.loads(clean_text, strict=False)
        except Exception:
            # Fallback to robust regex extraction for unescaped double quotes inside values
            parsed = {}
            # Parse is_rewritten
            ir_match = re.search(r'"is_rewritten"\s*:\s*(true|false)', clean_text, re.IGNORECASE)
            if ir_match:
                parsed["is_rewritten"] = ir_match.group(1).lower() == "true"
            
            # Try order 1: context then response
            ctx_match = re.search(r'"context"\s*:\s*"(.*)"\s*,\s*"response"\s*:', clean_text, re.DOTALL)
            if ctx_match:
                val = ctx_match.group(1)
                try:
                    val = bytes(val, "utf-8").decode("unicode_escape")
                except Exception:
                    val = val.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
                parsed["context"] = val
                
                resp_match = re.search(r'"response"\s*:\s*"(.*)"\s*(?:\}\s*|\Z)', clean_text, re.DOTALL)
                if resp_match:
                    val = resp_match.group(1).strip()
                    if val.endswith('"'):
                        val = val[:-1]
                    try:
                        val = bytes(val, "utf-8").decode("unicode_escape")
                    except Exception:
                        val = val.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
                    parsed["response"] = val
            else:
                # Try order 2: response then context
                resp_match = re.search(r'"response"\s*:\s*"(.*)"\s*,\s*"context"\s*:', clean_text, re.DOTALL)
                if resp_match:
                    val = resp_match.group(1)
                    try:
                        val = bytes(val, "utf-8").decode("unicode_escape")
                    except Exception:
                        val = val.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
                    parsed["response"] = val
                    
                    ctx_match = re.search(r'"context"\s*:\s*"(.*)"\s*(?:\}\s*|\Z)', clean_text, re.DOTALL)
                    if ctx_match:
                        val = ctx_match.group(1).strip()
                        if val.endswith('"'):
                            val = val[:-1]
                        try:
                            val = bytes(val, "utf-8").decode("unicode_escape")
                        except Exception:
                            val = val.replace('\\n', '\n').replace('\\t', '\t').replace('\\"', '"').replace('\\\\', '\\')
                        parsed["context"] = val
            return parsed

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

def ensure_validation_workspace(workspace_dir: Path):
    """Ensure a validation Cargo project exists with standard dependencies."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    cargo_toml_path = workspace_dir / "Cargo.toml"
    src_dir = workspace_dir / "src"
    src_dir.mkdir(exist_ok=True)
    
    lib_rs = src_dir / "lib.rs"
    if not lib_rs.exists():
        lib_rs.write_text("// Validation file\n", encoding="utf-8")
        
    cargo_toml_content = """[package]
name = "temp_validation_cargo_project"
version = "0.1.0"
edition = "2021"

[dependencies]
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
reqwest = { version = "0.11", features = ["blocking", "json"] }
libc = "0.2"
futures = "0.3"
rand = "0.8"
anyhow = "1.0"
lazy_static = "1.4"
once_cell = "1.18"
"""
    cargo_toml_path.write_text(cargo_toml_content, encoding="utf-8")

def validate_cargo_toml(content: str) -> Optional[str]:
    """Validate Cargo.toml content. Returns error message if invalid, None if valid."""
    if not tomllib:
        if "[package]" not in content and "[dependencies]" not in content:
            return "Missing essential sections in Cargo.toml"
        return None
    try:
        tomllib.loads(content)
        return None
    except Exception as e:
        return f"TOML parsing failed: {e}"

async def compile_rust_code(code: str, workspace_dir: Path, lock: asyncio.Lock) -> Tuple[bool, str]:
    """Compile Rust code by writing to src/lib.rs and running cargo check.
    
    Uses asyncio.Lock to prevent concurrent validation collisions.
    """
    async with lock:
        lib_rs_path = workspace_dir / "src" / "lib.rs"
        lib_rs_path.write_text(code, encoding="utf-8")
        
        try:
            process = await asyncio.create_subprocess_exec(
                "cargo", "check", "--lib",
                "--manifest-path", str(workspace_dir / "Cargo.toml"),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return True, ""
            else:
                return False, stderr.decode("utf-8", errors="replace")
        except Exception as e:
            return False, f"Cargo execution error: {e}"

def has_catastrophic_repetition(text: str) -> bool:
    """Detect paragraph-level or block-level repetitions in the generated text."""
    # 1. Check for duplicate sentences/lines
    sentences = re.split(r'[.!?\n]+', text)
    sentences = [s.strip() for s in sentences if len(s.strip()) > 15]
    for s in sentences:
        if sentences.count(s) > 3:
            return True
            
    # 2. Check for repeating chunks (consecutive duplicate windows)
    text_len = len(text)
    for chunk_len in range(30, min(text_len // 2, 250)):
        for i in range(text_len - 2 * chunk_len):
            chunk = text[i:i+chunk_len]
            if text[i+chunk_len:i+2*chunk_len] == chunk:
                repetitions = 1
                idx = i + chunk_len
                while idx + chunk_len <= text_len and text[idx:idx+chunk_len] == chunk:
                    repetitions += 1
                    idx += chunk_len
                if repetitions >= 3:
                    return True
    return False

def robust_extract_response(res_raw: str) -> Optional[str]:
    """Robustly extract response (code/toml block) from raw LLM output."""
    # 1. Try XML tag response first
    res = extract_field_xml(res_raw, "response")
    if res:
        return res
        
    # 2. Try JSON fallback
    try:
        parsed = extract_json_cached(res_raw)
        res = parsed.get("response")
        if res:
            return res
    except Exception:
        pass
        
    # 3. Try markdown code block fence
    matches = re.findall(r'```(?:rust|toml)?\n(.*?)```', res_raw, re.DOTALL | re.IGNORECASE)
    if matches:
        return matches[0].strip()
        
    # 4. If no tags or fences are found, check if it looks like raw code
    if "fn " in res_raw or "struct " in res_raw or "impl " in res_raw or "[package]" in res_raw:
        # Strip other XML tags to clean up
        clean_res = res_raw.replace("<response>", "").replace("</response>", "").strip()
        return clean_res
        
    return None

def robust_extract_context(res_raw: str, default_context: str) -> str:
    """Robustly extract context/explanation from raw LLM output, falling back to default_context."""
    # 1. Try XML tag context first
    ctx = extract_field_xml(res_raw, "context")
    if ctx:
        return ctx
        
    # 2. Try JSON fallback
    try:
        parsed = extract_json_cached(res_raw)
        ctx = parsed.get("context")
        if ctx:
            return ctx
    except Exception:
        pass
        
    # 3. Try extracting text before response tags/fences
    parts = re.split(r'(?:<response>|```)', res_raw, maxsplit=1)
    if parts and parts[0].strip():
        clean_part = parts[0].replace("<context>", "").replace("</context>", "").strip()
        if clean_part:
            return clean_part
            
    return default_context

class SFTDataProcessor:
    """Optimized processor for RAG SFT data with MLX format support."""
    
    def __init__(
        self,
        gen_model: str,
        judge_model: str,
        gen_url: str,
        judge_url: str,
        max_concurrent: int = 10,
        rate_limit: int = 20,
        output_format: str = "chat",
        gen_max_tokens: int = 3072,
        judge_max_tokens: int = 1024,
        chunk_size: int = 5
    ):
        self.gen_model = gen_model
        self.judge_model = judge_model
        self.gen_client = RapidMLXClient(gen_url)
        self.judge_client = RapidMLXClient(judge_url)
        self.rate_limiter = RateLimiter(rate_limit)
        self.max_concurrent = max_concurrent
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.output_format = output_format
        self.audit_log = []
        self.metrics_log = []
        self.gen_max_tokens = gen_max_tokens
        self.judge_max_tokens = judge_max_tokens
        self.chunk_size = chunk_size
        self.workspace_dir = repo_root / "data" / "processed" / "temp_validation_cargo_project"
        self.compilation_lock = asyncio.Lock()
        ensure_validation_workspace(self.workspace_dir)
        self.consecutive_failures = 0
        self.connection_lost = False
    
    async def generate_with_fallback(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: float = 0.1,
        max_tokens: Optional[int] = None,
        stop: Optional[List[str]] = None,
        repetition_penalty: float = 1.1,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0
    ) -> Dict[str, Any]:
        """Generate using the generator client with rate limiting."""
        if self.connection_lost:
            raise ConnectionLostError("Connection to model server was lost.")
            
        await self.rate_limiter.acquire()
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        try:
            res = await self.gen_client.chat_completion(
                self.gen_model, messages, temperature, max_tokens, stop,
                repetition_penalty, frequency_penalty, presence_penalty
            )
            self.consecutive_failures = 0  # Reset on success
            return res
        except Exception as e:
            if "connect" in str(e).lower() or "connection" in str(e).lower():
                self.consecutive_failures += 1
                if self.consecutive_failures >= 3:
                    self.connection_lost = True
            raise

    @staticmethod
    def check_heuristics(response: str) -> Optional[str]:
        """Fast local check for common placeholder and code stub patterns to avoid LLM overhead."""
        # 1. Look for todo! or unimplemented! macros in Rust code
        if re.search(r"\b(todo|unimplemented)!", response):
            return "Response contains 'todo!' or 'unimplemented!' macro."
        
        # 2. Look for comments indicating todo or placeholder
        if re.search(r"(?i)//\s*(todo|placeholder|insert|code\s+here|write\s+your|stub)", response):
            return "Response contains TODO/placeholder comments."
        
        # 3. Look for ellipsis stubs or omissions (e.g. standalone or inside comments)
        if re.search(r'(?m)^\s*(\/\/\s*)?\.\.\.\s*$', response) or "// ..." in response or "/* ... */" in response:
            return "Response contains ellipsis stubs ('...')."
            
        # 4. Look for dummy variable stubs like let x = 42; or let mut x = 42;
        if re.search(r"\blet\s+(mut\s+)?([a-zA-Z0-9_]+)\s*=\s*(42|10|1|0)\s*;", response):
            return "Response contains dummy variable assignments (e.g., let x = 42;)."
        
        # 5. Generic print stubs
        if re.search(r'println!\(\s*"\s*(Value:|Hello,\s*world|test|here|debug)\s*.*"\s*\)', response):
            return "Response contains generic print stubs."
            
        return None
            
    async def refine_rust_code(
        self,
        instruction: str,
        failed_context: str,
        failed_code: str,
        compile_error: str,
        row_idx: int,
        total_rows: int,
        attempt: int
    ) -> Optional[Tuple[str, str]]:
        """Compiler-guided self-correction loop for Rust compilation failures."""
        refine_prompt = f"""You are a senior Rust core developer. The Rust code you previously generated has compilation errors. Please fix all compiler errors, undefined functions, missing imports, or type mismatches.
        
Instruction: {instruction}

Original Explanation:
{failed_context}

Previously Generated Code with Errors:
```rust
{failed_code}
```

Compiler Error Output (stderr):
```
{compile_error}
```

Rules:
1. FORMAT CONSTRAINT: Your output structure must strictly match the requested file format. If the instruction references a configuration format (e.g., Cargo.toml), you must return raw, valid TOML blocks inside the "response" field, NOT Rust code files.
2. NO SIMULATED OR DUMMY LOGIC: Every code example must be a fully functional, logical, and complete implementation of a realistic operation.
3. Correct all errors identified by the compiler. Make sure all standard library imports (e.g. `use std::...`) and external crate references are present.
4. Output your response wrapped strictly in XML tags <context>...</context> and <response>...</response>. Do not wrap in markdown backticks or add any other text outside these tags.

<context>
[explanation text here]
</context>
<response>
[corrected code/config block here]
</response>"""
        
        try:
            print(f"  [Row {row_idx}/{total_rows}] 🛠️ Attempting compilation self-correction ({attempt}/2)...")
            print(f"  [Row {row_idx}/{total_rows}] 🔌 Awaiting self-correction response from LLM (max 1500 tokens)...")
            res_dict = await self.generate_with_fallback(
                refine_prompt,
                system_prompt="You are an expert Rust code refactoring engine that outputs only valid XML to fix compiler errors.",
                temperature=0.1,
                max_tokens=min(self.gen_max_tokens, 1500),
                repetition_penalty=1.1,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            elapsed = res_dict["elapsed"]
            self.metrics_log.append({
                "timestamp": datetime.now().isoformat(),
                "phase": f"refinement_attempt_{attempt}",
                "row_idx": row_idx,
                "model": self.gen_model,
                "prompt_tokens": res_dict["prompt_tokens"],
                "completion_tokens": res_dict["completion_tokens"],
                "elapsed": elapsed,
                "tokens_per_sec": res_dict["completion_tokens"] / elapsed if elapsed > 0 else 0
            })
            
            res_raw = res_dict["content"]
            new_response = robust_extract_response(res_raw)
            new_context = robust_extract_context(res_raw, failed_context)
            
            return new_context, new_response
        except ConnectionLostError:
            raise
        except Exception as e:
            print(f"  [Row {row_idx}/{total_rows}] ❌ Self-correction query failed: {e}")
            return None

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
            res_dict = await self.generate_with_fallback(
                gen_prompt,
                system_prompt="You are an expert Rust code refactoring engine that outputs only valid XML.",
                temperature=0.1,
                max_tokens=self.gen_max_tokens,
                repetition_penalty=1.1,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            elapsed = res_dict["elapsed"]
            self.metrics_log.append({
                "timestamp": datetime.now().isoformat(),
                "phase": "generation",
                "row_idx": row_idx,
                "model": self.gen_model,
                "prompt_tokens": res_dict["prompt_tokens"],
                "completion_tokens": res_dict["completion_tokens"],
                "elapsed": elapsed,
                "tokens_per_sec": res_dict["completion_tokens"] / elapsed if elapsed > 0 else 0
            })
            
            res_raw = res_dict["content"]
            
            # Robustly extract response and context
            new_response = robust_extract_response(res_raw)
            new_context = robust_extract_context(res_raw, context)
            
            if not new_response:
                print(f"  [Row {row_idx}/{total_rows}] ⚠️ Empty response field, skipping. Raw LLM response: {res_raw!r}")
                return None
            
            # Determine if target format is TOML/Cargo.toml
            is_toml = "cargo.toml" in instruction.lower() or new_response.strip().startswith("[")
            
            # Code complexity checks
            # Must have at least 8 lines and define a Rust structure (fn, struct, impl, enum, macro_rules!, trait) or be a Cargo.toml TOML
            lines = new_response.strip().splitlines()
            has_rust_structure = any(kw in new_response for kw in ["fn ", "struct ", "impl ", "enum ", "macro_rules!", "trait "])
            if not is_toml and (len(lines) < 8 or not has_rust_structure):
                print(f"  [Row {row_idx}/{total_rows}] ❌ Code rejected by complexity/entropy check (lines={len(lines)}, has_structure={has_rust_structure})")
                return None
                
            # Perform validation
            if is_toml:
                toml_err = validate_cargo_toml(new_response)
                if toml_err:
                    print(f"  [Row {row_idx}/{total_rows}] ❌ Cargo.toml validation failed: {toml_err}")
                    refined = await self.refine_rust_code(
                        instruction, new_context, new_response, toml_err, row_idx, total_rows, attempt=1
                    )
                    if refined:
                        new_context, new_response = refined
                        toml_err_2 = validate_cargo_toml(new_response)
                        if toml_err_2:
                            print(f"  [Row {row_idx}/{total_rows}] ❌ Cargo.toml validation failed after refinement: {toml_err_2}")
                            return None
                        else:
                            print(f"  [Row {row_idx}/{total_rows}] 🎉 Cargo.toml self-correction succeeded!")
                    else:
                        return None
            else:
                print(f"  [Row {row_idx}/{total_rows}] 📦 Running cargo compiler check...")
                passed, compile_err = await compile_rust_code(new_response, self.workspace_dir, self.compilation_lock)
                if not passed:
                    # Attempt self-correction loop 1
                    refined = await self.refine_rust_code(
                        instruction, new_context, new_response, compile_err, row_idx, total_rows, attempt=1
                    )
                    if refined:
                        new_context, new_response = refined
                        print(f"  [Row {row_idx}/{total_rows}] 📦 Running cargo compiler check (after attempt 1)...")
                        passed, compile_err = await compile_rust_code(new_response, self.workspace_dir, self.compilation_lock)
                        if not passed:
                            # Attempt self-correction loop 2
                            refined = await self.refine_rust_code(
                                instruction, new_context, new_response, compile_err, row_idx, total_rows, attempt=2
                            )
                            if refined:
                                new_context, new_response = refined
                                print(f"  [Row {row_idx}/{total_rows}] 📦 Running cargo compiler check (after attempt 2)...")
                                passed, compile_err = await compile_rust_code(new_response, self.workspace_dir, self.compilation_lock)
                                if not passed:
                                    print(f"  [Row {row_idx}/{total_rows}] ❌ Compilation failed after 2 self-correction attempts: {compile_err[:120].strip()}...")
                                    return None
                                else:
                                    print(f"  [Row {row_idx}/{total_rows}] 🎉 Compilation self-correction succeeded on attempt 2!")
                            else:
                                return None
                        else:
                            print(f"  [Row {row_idx}/{total_rows}] 🎉 Compilation self-correction succeeded on attempt 1!")
                    else:
                        return None
            
            return new_context, new_response
        except ConnectionLostError:
            raise
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
            res_dict = await self.generate_with_fallback(
                gen_prompt,
                system_prompt="You are a senior Rust core compiler maintainer.",
                temperature=0.1,
                max_tokens=1024,
                repetition_penalty=1.1,
                frequency_penalty=0.0,
                presence_penalty=0.0
            )
            
            elapsed = res_dict["elapsed"]
            self.metrics_log.append({
                "timestamp": datetime.now().isoformat(),
                "phase": "generation",
                "row_idx": row_idx,
                "model": self.gen_model,
                "prompt_tokens": res_dict["prompt_tokens"],
                "completion_tokens": res_dict["completion_tokens"],
                "elapsed": elapsed,
                "tokens_per_sec": res_dict["completion_tokens"] / elapsed if elapsed > 0 else 0
            })
            
            return res_dict["content"]
        except ConnectionLostError:
            raise
        except Exception as e:
            print(f"  [Row {row_idx}/{total_rows}] ❌ Generation failed: {e}")
            return None
    
    @staticmethod
    def parse_batch_results_xml(text: str, total_rows: int) -> Dict[int, Tuple[bool, str]]:
        """Extract multiple <sample id="X">...</sample> blocks from batch response."""
        results = {}
        # Match from <sample id="X"> until </sample> or the next <sample or the end of the string
        sample_pattern = r'<sample\s+id=["\'](\d+)["\']>(.*?)(?:</sample>|(?=<sample)|$)'
        samples = re.findall(sample_pattern, text, re.DOTALL | re.IGNORECASE)
        
        for sample_id_str, sample_content in samples:
            try:
                sample_id = int(sample_id_str)
                decision = extract_field_xml(sample_content, "decision")
                reason = extract_field_xml(sample_content, "reason")
                if decision:
                    decision = decision.upper().strip()
                    passed = (decision == "PASS")
                    results[sample_id] = (passed, reason or "unknown")
                    
                    # Print real-time status in the same format
                    icon = "✅ PASS" if passed else "❌ FAIL"
                    print(f"  [Row {sample_id}/{total_rows}] {icon} (Batch): {reason[:50] if reason else ''}")
            except Exception:
                pass
                
        return results

    async def judge_batch(
        self,
        samples: List[Tuple[int, str, str, str]],
        total_rows: int
    ) -> Dict[int, Tuple[bool, str]]:
        """LLM-as-a-Judge validation for a batch of samples."""
        if self.connection_lost:
            raise ConnectionLostError("Connection to model server was lost.")
            
        prompt_parts = []
        prompt_parts.append("You are an SFT training data auditor. Evaluate the following RAG SFT training samples against these explicit quality criteria:")
        prompt_parts.append("1. Crucial Constraint: The Response MUST NOT contain placeholder logic, dummy code stubs (e.g., `let x = 42;`, `println!(\"Value: {}\");`), or generic syntax-free placeholders. Note: A clean, complete, and functional library function, struct, enum, module, or unit test is NOT a dummy code stub even if it is short. Only code containing unfinished stubs (like `todo!`, `unimplemented!`, ellipsis `...`), dummy variables (like `let x = 42;`), or placeholder comments is considered a dummy code stub.")
        prompt_parts.append("2. RAG Grounding: The Response must strictly answer the Instruction using the facts established within the Context block.")
        prompt_parts.append("\nSamples to Audit:")
        
        for row_idx, instruction, context, response in samples:
            prompt_parts.append(f"""\n=== Sample ID: {row_idx} ===
Instruction:
{instruction}

Context:
{context}

Response:
{response}
----------------------------------------""")
            
        prompt_parts.append("\nOutput your evaluation in XML format using the structure below. Include a <sample> block for every single sample audited:")
        prompt_parts.append("""<results>
  <sample id="[Sample ID]">
    <decision>PASS or FAIL</decision>
    <reason>short explanation</reason>
  </sample>
</results>""")
        
        judge_prompt = "\n".join(prompt_parts)
        
        try:
            messages = [
                {"role": "system", "content": "You are a strict code quality auditor that outputs only valid XML. Do not output any <think> tags or chain-of-thought reasoning, start directly with the XML results block."},
                {"role": "user", "content": judge_prompt}
            ]
            await self.rate_limiter.acquire()
            start_time = time.time()
            res_dict = await self.judge_client.chat_completion(
                self.judge_model, messages, temperature=0.1, max_tokens=1536, stop=["</results>"],
                repetition_penalty=1.1, frequency_penalty=0.0, presence_penalty=0.0
            )
            self.consecutive_failures = 0  # Reset on success
            elapsed = time.time() - start_time
            
            judge_res_raw = res_dict["content"]
            
            # Log metrics for the batch request
            self.metrics_log.append({
                "timestamp": datetime.now().isoformat(),
                "phase": "validation_batch",
                "row_idx": samples[0][0],  # Store the first row_idx of the batch as reference
                "model": self.judge_model,
                "prompt_tokens": res_dict["prompt_tokens"],
                "completion_tokens": res_dict["completion_tokens"],
                "elapsed": elapsed,
                "tokens_per_sec": res_dict["completion_tokens"] / elapsed if elapsed > 0 else 0
            })
            
            parsed_results = self.parse_batch_results_xml(judge_res_raw, total_rows)
            return parsed_results
            
        except ConnectionLostError:
            raise
        except Exception as e:
            if "connect" in str(e).lower() or "connection" in str(e).lower():
                self.consecutive_failures += 1
                if self.consecutive_failures >= 3:
                    self.connection_lost = True
            print(f"  ⚠️ Batch judge API call failed: {e}")
            return {}

    async def judge_sample(
        self,
        instruction: str,
        context: str,
        response: str,
        row_idx: int,
        total_rows: int
    ) -> Tuple[bool, str]:
        """LLM-as-a-Judge validation using XML and fallbacks. Returns (passed, reason)."""
        if self.connection_lost:
            raise ConnectionLostError("Connection to model server was lost.")
            
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

Output your evaluation using XML tags as shown below:
<decision>PASS or FAIL</decision>
<reason>short explanation</reason>"""
        
        try:
            messages = [
                {"role": "system", "content": "You are a strict code quality auditor that outputs only valid XML. Do not output any <think> tags or chain-of-thought reasoning, start directly with the XML decision."},
                {"role": "user", "content": judge_prompt}
            ]
            await self.rate_limiter.acquire()
            res_dict = await self.judge_client.chat_completion(
                self.judge_model, messages, temperature=0.1, max_tokens=self.judge_max_tokens, stop=["</reason>"],
                repetition_penalty=1.1, frequency_penalty=0.0, presence_penalty=0.0
            )
            self.consecutive_failures = 0  # Reset on success
            
            elapsed = res_dict["elapsed"]
            self.metrics_log.append({
                "timestamp": datetime.now().isoformat(),
                "phase": "validation",
                "row_idx": row_idx,
                "model": self.judge_model,
                "prompt_tokens": res_dict["prompt_tokens"],
                "completion_tokens": res_dict["completion_tokens"],
                "elapsed": elapsed,
                "tokens_per_sec": res_dict["completion_tokens"] / elapsed if elapsed > 0 else 0
            })
            
            judge_res_raw = res_dict["content"]
            
            # Try XML tag extraction first
            decision = extract_field_xml(judge_res_raw, "decision")
            reason = extract_field_xml(judge_res_raw, "reason")
            
            # Fallback to JSON if XML tags are missing
            if not decision:
                try:
                    parsed = extract_json_cached(judge_res_raw)
                    decision = parsed.get("decision", "FAIL")
                    reason = parsed.get("reason", "unknown")
                except Exception:
                    # Final text-search fallback
                    if "PASS" in judge_res_raw.upper():
                        decision = "PASS"
                        reason = "extracted PASS from raw text"
                    else:
                        decision = "FAIL"
                        reason = f"extracted FAIL or failed parsing raw text: {judge_res_raw[:100]}"
            
            decision = decision.upper().strip()
            reason = reason or "unknown"
            
            if decision == "PASS":
                print(f"  [Row {row_idx}/{total_rows}] ✅ PASS (in {elapsed:.2f}s): {reason[:50]}")
                return True, reason
            else:
                print(f"  [Row {row_idx}/{total_rows}] ❌ FAIL (in {elapsed:.2f}s): {reason[:50]}")
                return False, reason
        except ConnectionLostError:
            raise
        except Exception as e:
            if "connect" in str(e).lower() or "connection" in str(e).lower():
                self.consecutive_failures += 1
                if self.consecutive_failures >= 3:
                    self.connection_lost = True
            print(f"  [Row {row_idx}/{total_rows}] ⚠️ Judge failed: {e}")
            return False, str(e)
    
    async def generate_row(
        self,
        row: Dict,
        source_name: str,
        inst_col: str,
        ctx_col: str,
        resp_col: str,
        row_idx: int,
        total_rows: int
    ) -> Optional[Tuple[int, Dict, str, str]]:
        """Phase 1: Generate/refine a response for the row. Returns (row_idx, original_row, context, response) if successful."""
        if self.connection_lost:
            raise ConnectionLostError("Connection to model server was lost.")
            
        async with self.semaphore:
            try:
                instruction = row.get(inst_col) or ""
                context = row.get(ctx_col) or ""
                response = row.get(resp_col) or ""
                
                if source_name == "matteopilotto/rust-github-issues":
                    # Sanitize context programmatically first to strip placeholders and comments
                    sanitized_context = sanitize_github_issue_context(context)
                    generated_response = await self.process_github_issues_row(
                        instruction, sanitized_context, row_idx, total_rows
                    )
                    if not generated_response:
                        self.audit_log.append({
                            "timestamp": datetime.now().isoformat(),
                            "source": source_name,
                            "row_idx": row_idx,
                            "instruction": instruction,
                            "context": sanitized_context,
                            "response": "",
                            "decision": "FAIL",
                            "reason": "Generator LLM returned empty response or failed to generate"
                        })
                        return None
                    context = sanitized_context
                    response = generated_response
                
                elif source_name == "Convence/Rust-Coder":
                    result = await self.process_rust_coder_row(
                        instruction, context, response, row_idx, total_rows
                    )
                    if not result:
                        self.audit_log.append({
                            "timestamp": datetime.now().isoformat(),
                            "source": source_name,
                            "row_idx": row_idx,
                            "instruction": instruction,
                            "context": context,
                            "response": response,
                            "decision": "FAIL",
                            "reason": "Generator LLM failed to refine or returned empty context/response"
                        })
                        return None
                    context, response = result
                
                # Check for catastrophic text repetition
                if response and has_catastrophic_repetition(response):
                    print(f"  [Row {row_idx}/{total_rows}] ❌ FAIL (Repetition): Catastrophic text repetition detected.")
                    self.audit_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "source": source_name,
                        "row_idx": row_idx,
                        "instruction": instruction,
                        "context": context,
                        "response": response,
                        "decision": "FAIL",
                        "reason": "Generated response failed catastrophic text repetition check"
                    })
                    return None
                
                return row_idx, row, context, response
            except ConnectionLostError:
                raise
            except Exception as e:
                self.audit_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "source": source_name,
                    "row_idx": row_idx,
                    "instruction": row.get(inst_col) or "",
                    "context": row.get(ctx_col) or "",
                    "response": row.get(resp_col) or "",
                    "decision": "FAIL",
                    "reason": f"Unexpected generation error: {str(e)}"
                })
                print(f"  [Row {row_idx}/{total_rows}] ❌ Unexpected generation error: {e}")
                return None

    async def judge_row(
        self,
        row: Dict,
        context: str,
        response: str,
        source_name: str,
        inst_col: str,
        row_idx: int,
        total_rows: int
    ) -> Optional[Dict]:
        """Phase 2: Validate the generated response. Returns formatted dict if passed."""
        if self.connection_lost:
            raise ConnectionLostError("Connection to model server was lost.")
            
        async with self.semaphore:
            try:
                instruction = row.get(inst_col) or ""
                
                # Check local heuristics first to avoid slow LLM calls
                heuristic_reason = self.check_heuristics(response)
                if heuristic_reason:
                    print(f"  [Row {row_idx}/{total_rows}] ❌ FAIL (Heuristic): {heuristic_reason}")
                    self.audit_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "source": source_name,
                        "row_idx": row_idx,
                        "instruction": instruction,
                        "context": context,
                        "response": response,
                        "decision": "FAIL",
                        "reason": f"Heuristic check: {heuristic_reason}"
                    })
                    # Log heuristic metrics record
                    self.metrics_log.append({
                        "timestamp": datetime.now().isoformat(),
                        "phase": "validation_heuristic",
                        "row_idx": row_idx,
                        "model": self.judge_model,
                        "prompt_tokens": 0,
                        "completion_tokens": 0,
                        "elapsed": 0.0,
                        "tokens_per_sec": 0.0
                    })
                    return None
                
                passed, reason = await self.judge_sample(
                    instruction, context, response, row_idx, total_rows
                )
                
                self.audit_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "source": source_name,
                    "row_idx": row_idx,
                    "instruction": instruction,
                    "context": context,
                    "response": response,
                    "decision": "PASS" if passed else "FAIL",
                    "reason": reason
                })
                
                if not passed:
                    return None
                
                # Format output based on requested format
                if self.output_format == "chat":
                    return {
                        "messages": [
                            {"role": "system", "content": "You are a helpful Rust programming assistant trained in RAG context retrieval."},
                            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {instruction}"},
                            {"role": "assistant", "content": response}
                        ]
                    }
                elif self.output_format == "completions":
                    return {
                        "prompt": f"Instruction:\n{instruction}\n\nContext:\n{context}",
                        "completion": response
                    }
                else:
                    formatted_text = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}<|endoftext|>"
                    return {"text": formatted_text}
            except ConnectionLostError:
                raise
            except Exception as e:
                self.audit_log.append({
                    "timestamp": datetime.now().isoformat(),
                    "source": source_name,
                    "row_idx": row_idx,
                    "instruction": instruction,
                    "context": context,
                    "response": response,
                    "decision": "FAIL",
                    "reason": f"Unexpected judging error: {str(e)}"
                })
                print(f"  [Row {row_idx}/{total_rows}] ❌ Unexpected judging error: {e}")
                return None
    
    async def close(self):
        """Clean up generator and judge client resources."""
        await self.gen_client.close()
        await self.judge_client.close()

async def process_source_async(
    source: Dict,
    processor: SFTDataProcessor,
    limit: int,
    offset: int,
    workers: int,
    train_file: Path,
    valid_file: Path,
    audit_file: Path,
    metrics_file: Path,
    train_ratio: float
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
    
    chunk_size = processor.chunk_size
    all_results = []
    
    num_train_written = 0
    num_valid_written = 0
    total_validated_count = 0
    
    print(f"🔄 Processing {len(records)} records in chunks of {chunk_size}...")
    
    try:
        with tqdm(total=len(records), desc=f"Processing {name}", unit="rows") as pbar:
            for chunk_start in range(0, len(records), chunk_size):
                chunk_records = records[chunk_start : chunk_start + chunk_size]
                
                # Phase 1: Generation
                gen_tasks = []
                for offset, row in enumerate(chunk_records):
                    row_idx = chunk_start + offset + 1
                    task = processor.generate_row(
                        row, name, inst_col, ctx_col, resp_col,
                        row_idx, len(records)
                    )
                    gen_tasks.append(task)
                
                generated_chunk_rows = await asyncio.gather(*gen_tasks)
                generated_chunk_rows = [r for r in generated_chunk_rows if r is not None]
                generated_chunk_rows.sort(key=lambda x: x[0])
                
                # Phase 2: Heuristics & Validation
                passed_heuristics = []
                for row_idx, row, context, response in generated_chunk_rows:
                    instruction = row.get(inst_col) or ""
                    heuristic_reason = processor.check_heuristics(response)
                    if heuristic_reason:
                        print(f"  [Row {row_idx}/{len(records)}] ❌ FAIL (Heuristic): {heuristic_reason}")
                        audit_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "source": name,
                            "row_idx": row_idx,
                            "instruction": instruction,
                            "context": context,
                            "response": response,
                            "decision": "FAIL",
                            "reason": f"Heuristic check: {heuristic_reason}"
                        }
                        processor.audit_log.append(audit_entry)
                        with open(audit_file, "ab") as f:
                            f.write(orjson.dumps(audit_entry) + b"\n")
                        
                        metric_entry = {
                            "timestamp": datetime.now().isoformat(),
                            "phase": "validation_heuristic",
                            "row_idx": row_idx,
                            "model": processor.judge_model,
                            "prompt_tokens": 0,
                            "completion_tokens": 0,
                            "elapsed": 0.0,
                            "tokens_per_sec": 0.0
                        }
                        processor.metrics_log.append(metric_entry)
                        with open(metrics_file, "ab") as f:
                            f.write(orjson.dumps(metric_entry) + b"\n")
                    else:
                        passed_heuristics.append((row_idx, row, context, response))
                
                # Batch Judge
                if passed_heuristics:
                    samples_for_batch = [(item[0], item[1].get(inst_col) or "", item[2], item[3]) for item in passed_heuristics]
                    
                    metrics_before = len(processor.metrics_log)
                    batch_results = await processor.judge_batch(samples_for_batch, len(records))
                    metrics_after = len(processor.metrics_log)
                    
                    # Write any new metrics entries written during judge_batch
                    with open(metrics_file, "ab") as f:
                        for m_idx in range(metrics_before, metrics_after):
                            f.write(orjson.dumps(processor.metrics_log[m_idx]) + b"\n")
                    
                    fallback_items = []
                    chunk_validated_results = []
                    
                    for row_idx, row, context, response in passed_heuristics:
                        instruction = row.get(inst_col) or ""
                        if row_idx in batch_results:
                            passed, reason = batch_results[row_idx]
                            audit_entry = {
                                "timestamp": datetime.now().isoformat(),
                                "source": name,
                               "row_idx": row_idx,
                                "instruction": instruction,
                                "context": context,
                                "response": response,
                                "decision": "PASS" if passed else "FAIL",
                                "reason": reason
                            }
                            processor.audit_log.append(audit_entry)
                            with open(audit_file, "ab") as f:
                                f.write(orjson.dumps(audit_entry) + b"\n")
                                
                            if passed:
                                if processor.output_format == "chat":
                                    formatted = {
                                        "messages": [
                                            {"role": "system", "content": "You are a helpful Rust programming assistant trained in RAG context retrieval."},
                                            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {instruction}"},
                                            {"role": "assistant", "content": response}
                                        ]
                                    }
                                elif processor.output_format == "completions":
                                    formatted = {
                                        "prompt": f"Instruction:\n{instruction}\n\nContext:\n{context}",
                                        "completion": response
                                    }
                                else:
                                    formatted_text = f"### Instruction:\n{instruction}\n\n### Context:\n{context}\n\n### Response:\n{response}<|endoftext|>"
                                    formatted = {"text": formatted_text}
                                chunk_validated_results.append((row_idx, formatted))
                            else:
                                fallback_items.append((row_idx, row, context, response))
                        else:
                            fallback_items.append((row_idx, row, context, response))
                            
                    if fallback_items:
                        print(f"  [Chunk Fallback] ⚠️ Incomplete or failed batch results. Running individual fallback for {len(fallback_items)} items...")
                        for row_idx, row, context, response in fallback_items:
                            metrics_before = len(processor.metrics_log)
                            audit_before = len(processor.audit_log)
                            
                            formatted = await processor.judge_row(row, context, response, name, inst_col, row_idx, len(records))
                            
                            metrics_after = len(processor.metrics_log)
                            audit_after = len(processor.audit_log)
                            
                            with open(metrics_file, "ab") as f:
                                for m_idx in range(metrics_before, metrics_after):
                                    f.write(orjson.dumps(processor.metrics_log[m_idx]) + b"\n")
                            with open(audit_file, "ab") as f:
                                for a_idx in range(audit_before, audit_after):
                                    f.write(orjson.dumps(processor.audit_log[a_idx]) + b"\n")
                                    
                            if formatted:
                                chunk_validated_results.append((row_idx, formatted))
                                
                    # Sort chunk validated results by row_idx before splitting/writing
                    chunk_validated_results.sort(key=lambda x: x[0])
                    
                    # Append to train/valid files
                    with open(train_file, "ab") as f_train, open(valid_file, "ab") as f_valid:
                        for row_idx, formatted in chunk_validated_results:
                            is_train = random.random() < train_ratio
                            if is_train:
                                f_train.write(orjson.dumps(formatted) + b"\n")
                                num_train_written += 1
                            else:
                                f_valid.write(orjson.dumps(formatted) + b"\n")
                                num_valid_written += 1
                            total_validated_count += 1
                            all_results.append(formatted)
                
                pbar.update(len(chunk_records))
    except ConnectionLostError:
        print(f"\n❌ Aborting source processing for {name}: Connection to model server was lost.")
        raise
            
    print(f"\n✨ Source processing complete. Validated and wrote {total_validated_count} records (Train: {num_train_written}, Valid: {num_valid_written}).")
    return all_results

def parse_args():
    parser = argparse.ArgumentParser(description="High-performance RAG MLX Dataset Processor powered by Rapid-MLX.")
    parser.add_argument("--config", "-c", help="Path to configuration file")
    parser.add_argument("--output-dir", "-o", help="Override output directory")
    parser.add_argument("--seed", "-s", type=int, help="Random seed")
    parser.add_argument("--train-ratio", "-r", type=float, help="Training ratio (default: 0.85)")
    parser.add_argument("--limit", "-l", type=int, default=10, help="Limit records per source (-1 for all)")
    parser.add_argument("--offset", type=int, default=0, help="Offset to start reading records from")
    parser.add_argument("--workers", "-w", type=int, default=4, help="Max concurrent workers")
    parser.add_argument("--rate-limit", type=int, default=20, help="API calls per second")
    
    # Models configuration
    parser.add_argument(
        "--gen-model", 
        default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit", 
        help="Generator model served by Rapid-MLX"
    )
    parser.add_argument(
        "--judge-model", 
        default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit", 
        help="Judge model served by Rapid-MLX"
    )
    
    # URLs configuration
    parser.add_argument(
        "--gen-url", 
        default="http://localhost:8000/v1", 
        help="Base URL of OpenAI-compatible API served for generator"
    )
    parser.add_argument(
        "--judge-url", 
        default="http://localhost:8000/v1", 
        help="Base URL of OpenAI-compatible API served for judge"
    )
    
    parser.add_argument(
        "--format", "-f", 
        choices=["chat", "completions", "text"], 
        default="chat", 
        help="Target output format for MLX training (default: chat)"
    )
    parser.add_argument(
        "--gen-max-tokens",
        type=int,
        default=3072,
        help="Max output tokens for the generator model (default: 3072)"
    )
    parser.add_argument(
        "--judge-max-tokens",
        type=int,
        default=1024,
        help="Max output tokens for the judge model (default: 1024)"
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5,
        help="Chunk size for batch auditing of generated rows (default: 5)"
    )
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
    output_format = args.format
    
    gen_model = args.gen_model
    judge_model = args.judge_model
    gen_url = args.gen_url
    judge_url = args.judge_url
    
    print(f"🔌 Generator Endpoint: {gen_url} ({gen_model})")
    print(f"🔌 Judge Endpoint: {judge_url} ({judge_model})")
    print(f"🚀 Concurrency Workers: {workers} | Rate: {rate_limit}/s | Format: {output_format}")
    
    # Prepare output directory immediately
    output_path = Path(output_dir, run_dir)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Paths for files
    train_file = output_path / "train.jsonl"
    valid_file = output_path / "valid.jsonl"
    audit_file = output_path / "evaluation_audit.jsonl"
    metrics_file = output_path / "metrics.jsonl"
    
    # Touch files to ensure they exist
    for f in [train_file, valid_file, audit_file, metrics_file]:
        f.touch(exist_ok=True)
        
    print(f"📂 Output files initialized in {output_path}")
    print(f"🎲 Shuffling seed set to {seed}")
    random.seed(seed)
    
    # Initialize processor pointing to Rapid-MLX OpenAI-compatible endpoints
    processor = SFTDataProcessor(
        gen_model=gen_model,
        judge_model=judge_model,
        gen_url=gen_url,
        judge_url=judge_url,
        max_concurrent=workers,
        rate_limit=rate_limit,
        output_format=output_format,
        gen_max_tokens=args.gen_max_tokens,
        judge_max_tokens=args.judge_max_tokens,
        chunk_size=args.chunk_size
    )
    
    try:
        # Process all sources
        all_results = []
        for source in config.get("sources", []):
            results = await process_source_async(
                source=source,
                processor=processor,
                limit=limit,
                offset=offset,
                workers=workers,
                train_file=train_file,
                valid_file=valid_file,
                audit_file=audit_file,
                metrics_file=metrics_file,
                train_ratio=train_ratio
            )
            all_results.extend(results)
        
        print(f"\n📊 Total validated records across all sources: {len(all_results)}")
        if not all_results:
            print("❌ No valid data generated, exiting")
            sys.exit(1)
        
        # Count lines written to train/valid files
        train_count = 0
        valid_count = 0
        if train_file.exists():
            with open(train_file, "r") as f:
                train_count = sum(1 for _ in f)
        if valid_file.exists():
            with open(valid_file, "r") as f:
                valid_count = sum(1 for _ in f)
                
        print(f"📝 Final split: {train_count} train / {valid_count} valid")
        print(f"✅ Success! Train: {train_file}, Valid: {valid_file}, Audit Log: {audit_file}")
        
    except ConnectionLostError:
        print("\n🛑 Run aborted: The connection to the model server was lost (it likely crashed/OOM'd).")
        print(f"💡 All successfully generated records up to the crash have been safely saved to: {output_path}")
        sys.exit(1)
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
