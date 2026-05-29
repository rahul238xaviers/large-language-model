"""Stage 3 smoke test — run via: python3 pipeline/_tests/test_stage3_smoke.py"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.data.selection.filters import SelectionPolicy
from pipeline.data.selection.scoring_engine import score_profile, summarise_scoring
from pipeline.data.selection.dataset_writer import write_selection_report

print("All Stage 3 imports OK")

# ── Policy round-trip ────────────────────────────────────────────────
policy = SelectionPolicy.from_dict({
    "min_quality_score": 0.2,
    "min_code_score": 0.0,
    "max_char_length": 0,
    "min_char_length": 150,
    "deduplicate": True,
})
row_pass = {"quality_score": 0.8, "code_score": 0.5, "char_length": 500}
row_fail = {"quality_score": 0.1, "code_score": 0.0, "char_length": 10}
print("pass reasons:", policy.apply_reason(row_pass))
print("fail reasons:", policy.apply_reason(row_fail))
assert policy.apply_reason(row_pass) == []
assert "low_quality" in policy.apply_reason(row_fail)
assert "too_short" in policy.apply_reason(row_fail)

# ── score_profile on synthetic DataFrame ─────────────────────────────
df = pd.DataFrame({
    "doc_id":        list(range(10)),
    "text_hash":     [f"h{i}" for i in range(10)],
    "char_length":   [50, 200, 300, 400, 100, 20000, 600, 800, 900, 1000],
    "line_count":    [2] * 10,
    "word_count":    [10] * 10,
    "code_score":    [0.8] * 10,
    "quality_score": [0.9, 0.9, 0.1, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
})

with tempfile.TemporaryDirectory() as tmp:
    profile_path = Path(tmp) / "test_doc_profile.parquet"
    df.to_parquet(profile_path, engine="pyarrow")

    scored = score_profile(profile_path, policy)
    print("Scored columns:", list(scored.columns))

    stats = summarise_scoring(scored)
    print("Stats:", stats)

    assert stats["total"] == 10
    # doc 0: char=50 → too_short
    # doc 2: quality=0.1 → low_quality
    # doc 4: char=100 → too_short
    assert stats["dropped"] >= 3, f"Expected >=3 drops, got {stats['dropped']}"
    assert "too_short" in stats["drop_reasons"]
    assert "low_quality" in stats["drop_reasons"]

    # ── selection report JSON ─────────────────────────────────────────
    report_path = Path(tmp) / "report.json"
    write_selection_report({"test_source": stats}, report_path)
    loaded = json.loads(report_path.read_text())
    print("Report total_kept:", loaded["total_kept"])
    assert loaded["total_documents"] == 10

# ── Config carries selection section ─────────────────────────────────
from pipeline.orchestration.config_loader import ConfigLoader
cfg = ConfigLoader().load("local-dev")
print("Selection cfg:", cfg.get("selection"))
assert cfg.get("selection", {}).get("policy") is not None

# ── CLI parses data-select ────────────────────────────────────────────
from pipeline.cli import _build_parser
p = _build_parser()
args = p.parse_args(["data-select", "--run-id", "run_fake_000"])
print("CLI data-select parse OK, func:", args.func.__name__)

print("\nALL Stage 3 smoke tests PASSED")
