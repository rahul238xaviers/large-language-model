"""Stage 4 smoke test — run via: python3 -m pipeline._tests.test_stage4_smoke"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.ipc as ipc

from pipeline.data.tokenisation.tokeniser import Tokeniser
from pipeline.data.tokenisation.batch_engine import run_tokenisation_engine

print("All Stage 4 imports OK")

# ── Tokeniser unit checks ─────────────────────────────────────────────
tok = Tokeniser("cl100k_base")
assert tok.vocab_size == 100277, f"unexpected vocab size: {tok.vocab_size}"
assert tok.eot_id > 0

ids = tok.encode_doc("def hello(): pass")
assert ids[-1] == tok.eot_id, "EOT not appended"
assert len(ids) > 1

batch_results = list(tok.encode_batch(["hello world", "fn main() {}", "the quick fox"]))
assert len(batch_results) == 3
print(f"Tokeniser OK: vocab={tok.vocab_size}  eot={tok.eot_id}  sample_len={len(ids)}")

# ── Build a tiny synthetic Arrow dataset ─────────────────────────────
docs = [
    ("rust_stack",  "fn fibonacci(n: u64) -> u64 { if n <= 1 { n } else { fibonacci(n-1) + fibonacci(n-2) } } " * 30),
    ("fineweb_edu", "The transformer architecture introduced by Vaswani et al. in 2017 has become the backbone of modern NLP systems. " * 25),
    ("rust_stack",  "use std::collections::HashMap; fn main() { let mut map = HashMap::new(); map.insert('key', 42); } " * 30),
    ("starcoder",   "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x <= pivot]\n    right = [x for x in arr[1:] if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)\n" * 20),
]

schema = pa.schema([
    pa.field("source", pa.string()),
    pa.field("doc_id", pa.int64()),
    pa.field("text",   pa.large_utf8()),
])

with tempfile.TemporaryDirectory() as tmp:
    arrow_path = Path(tmp) / "pretrain_dataset.arrow"
    with ipc.new_file(str(arrow_path), schema) as writer:
        batch = pa.record_batch(
            {
                "source": [d[0] for d in docs],
                "doc_id": list(range(len(docs))),
                "text":   [d[1] for d in docs],
            },
            schema=schema,
        )
        writer.write_batch(batch)

    out_dir = Path(tmp) / "tokenised"
    artifacts = run_tokenisation_engine(
        arrow_path=arrow_path,
        output_dir=out_dir,
        tokeniser=tok,
        block_size=128,   # small for smoke test
        stride=64,        # 50% overlap
        text_column="text",
    )

    print("Artifacts returned:", {k: str(v.name) for k, v in artifacts.items()})

    # Validate sequences.npy
    arr = np.load(str(artifacts["sequences"]))
    print(f"sequences.npy: shape={arr.shape}  dtype={arr.dtype}")
    assert arr.ndim == 2
    assert arr.shape[1] == 128, f"expected block_size=128, got {arr.shape[1]}"
    assert arr.dtype == np.uint32
    assert arr.shape[0] > 0, "No sequences produced"
    assert arr.max() < tok.vocab_size, "Token ID exceeds vocab size"

    # Validate metadata JSON
    meta = json.loads(artifacts["meta"].read_text())
    print("Meta:", json.dumps(meta, indent=2))
    assert meta["block_size"] == 128
    assert meta["stride"] == 64
    assert meta["n_sequences"] == arr.shape[0]
    assert meta["vocab_size"] == 100277
    assert set(meta["sources_seen"]) == {"rust_stack", "fineweb_edu", "starcoder"}

# ── Config carries tokenisation section ──────────────────────────────
from pipeline.orchestration.config_loader import ConfigLoader
cfg = ConfigLoader().load("local-dev")
tok_cfg = cfg.get("tokenisation", {})
print("Tokenisation cfg:", tok_cfg)
assert tok_cfg.get("encoding") == "cl100k_base"
assert tok_cfg.get("block_size") == 2048

# ── CLI parses data-tokenise ──────────────────────────────────────────
from pipeline.cli import _build_parser
p = _build_parser()
args = p.parse_args(["data-tokenise", "--run-id", "run_fake_000"])
print("CLI data-tokenise parse OK, func:", args.func.__name__)

print("\nALL Stage 4 smoke tests PASSED")
