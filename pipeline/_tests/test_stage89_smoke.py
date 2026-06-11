"""Stage 8 + 9 smoke test — run via: python3 -m pipeline._tests.test_stage89_smoke"""
import json
import math
import os
import tempfile
from pathlib import Path

# ── Stage 8 imports ───────────────────────────────────────────────────
from pipeline.analytics.plot_training   import plot_training_metrics
from pipeline.analytics.plot_selection  import plot_selection_report
from pipeline.analytics.plot_eval       import plot_eval_history
from pipeline.analytics.run_dashboard   import build_run_dashboard

# ── Stage 9 imports ───────────────────────────────────────────────────
from pipeline.registry.model_registry import (
    ModelMetadata, finalize_run, load_metadata,
)

print("All Stage 8+9 imports OK")

# ─────────────────────────────────────────────────────────────────────
# Build a synthetic run directory with all expected artifact files
# ─────────────────────────────────────────────────────────────────────
with tempfile.TemporaryDirectory() as tmp:
    run_dir = Path(tmp) / "run_smoke_000"

    # ── training/metrics.csv ─────────────────────────────────────── #
    training_dir = run_dir / "training"
    ckpt_dir     = training_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True)

    metrics_csv = training_dir / "metrics.csv"
    with metrics_csv.open("w") as f:
        f.write("step,train_loss,learning_rate,tokens_per_sec,vram_usage_gb\n")
        for step in range(0, 1001, 100):
            loss = 5.0 * math.exp(-step / 400)
            lr   = 3e-4 * min(step / 100, 1.0) * (0.5 + 0.5 * math.cos(math.pi * step / 1000))
            f.write(f"{step},{loss:.4f},{lr:.6f},{800 + step * 0.1:.1f},{18.5:.1f}\n")

    # Fake checkpoints
    ckpt500  = ckpt_dir / "step_0000500.safetensors"
    ckpt1000 = ckpt_dir / "step_0001000.safetensors"
    ckpt500.write_bytes(b"")
    ckpt1000.write_bytes(b"")

    # ── selection/selection_report.json ──────────────────────────── #
    sel_dir = run_dir / "selection"
    sel_dir.mkdir()
    sel_report = {
        "sources": {
            "fineweb_edu": {
                "total": 100, "kept": 75, "dropped": 25, "keep_rate_pct": 75,
                "drop_reasons": {"low_quality": 15, "too_short": 10},
            },
            "rust_stack": {
                "total": 80, "kept": 68, "dropped": 12, "keep_rate_pct": 85,
                "drop_reasons": {"low_quality": 8, "duplicate": 4},
            },
        }
    }
    (sel_dir / "selection_report.json").write_text(json.dumps(sel_report))

    # ── evaluation/eval_history.json ─────────────────────────────── #
    eval_dir = run_dir / "evaluation"
    eval_dir.mkdir()
    eval_history = [
        {"step": 500,  "perplexity": 80.5,  "mean_loss": 4.39, "n_sequences": 50},
        {"step": 1000, "perplexity": 55.2,  "mean_loss": 4.01, "n_sequences": 50},
    ]
    (eval_dir / "eval_history.json").write_text(json.dumps(eval_history))

    # ── tokenisation/tokenisation_meta.json ──────────────────────── #
    tok_dir = run_dir / "tokenisation"
    tok_dir.mkdir()
    tok_meta = {"n_sequences": 1200, "block_size": 2048, "stride": 1024,
                "encoding": "cl100k_base", "sources": ["fineweb_edu", "rust_stack"]}
    (tok_dir / "tokenisation_meta.json").write_text(json.dumps(tok_meta))

    # ── artifact_registry.json ────────────────────────────────────── #
    registry = {"training": {"metrics": str(metrics_csv)},
                "evaluation": {"eval_history": str(eval_dir / "eval_history.json")}}
    (run_dir / "artifact_registry.json").write_text(json.dumps(registry))

    # ─────────────────────────────────────────────────────────────── #
    # Stage 8: plot_training_metrics                                   #
    # ─────────────────────────────────────────────────────────────── #
    figures_dir = run_dir / "analytics" / "figures"
    p_train = plot_training_metrics(metrics_csv, figures_dir)
    assert p_train.exists(), f"Missing: {p_train}"
    assert p_train.stat().st_size > 1000
    print(f"plot_training_metrics OK → {p_train.name} ({p_train.stat().st_size:,} bytes)")

    # ─────────────────────────────────────────────────────────────── #
    # Stage 8: plot_selection_report                                   #
    # ─────────────────────────────────────────────────────────────── #
    p_sel = plot_selection_report(sel_dir / "selection_report.json", figures_dir)
    assert p_sel.exists()
    assert p_sel.stat().st_size > 1000
    print(f"plot_selection_report OK → {p_sel.name} ({p_sel.stat().st_size:,} bytes)")

    # ─────────────────────────────────────────────────────────────── #
    # Stage 8: plot_eval_history                                        #
    # ─────────────────────────────────────────────────────────────── #
    p_eval = plot_eval_history(eval_dir / "eval_history.json", figures_dir)
    assert p_eval.exists()
    assert p_eval.stat().st_size > 1000
    print(f"plot_eval_history OK → {p_eval.name} ({p_eval.stat().st_size:,} bytes)")

    # ─────────────────────────────────────────────────────────────── #
    # Stage 8: build_run_dashboard                                      #
    # ─────────────────────────────────────────────────────────────── #
    dash_dir = run_dir / "analytics" / "dashboard"
    html_path = build_run_dashboard(run_dir, dash_dir)
    assert html_path.exists()
    html_text = html_path.read_text()
    assert "run_smoke_000" in html_text
    assert "Perplexity" in html_text         # eval table rendered
    assert "fineweb_edu" in html_text        # selection data embedded
    assert "tokenisation_meta" in html_text or "n_sequences" in html_text
    assert "data:image/png;base64," in html_text   # all three figures embedded
    print(f"build_run_dashboard OK → {html_path.name} ({html_path.stat().st_size:,} bytes)")

    # ─────────────────────────────────────────────────────────────── #
    # Stage 9: ModelMetadata round-trip                                 #
    # ─────────────────────────────────────────────────────────────── #
    meta = ModelMetadata(
        run_id          = "run_smoke_000",
        checkpoint_path = str(ckpt1000),
        step            = 1000,
        model_cfg       = {"n_layer": 24, "n_embd": 2048},
        eval_summary    = {"perplexity": 55.2, "step": 1000},
        created_at      = "2026-05-29T00:00:00+00:00",
    )
    d = meta.to_dict()
    assert d["step"] == 1000
    assert d["eval_summary"]["perplexity"] == 55.2
    meta2 = ModelMetadata.from_dict(d)
    assert meta2.run_id == "run_smoke_000"
    print("ModelMetadata round-trip OK")

    # ─────────────────────────────────────────────────────────────── #
    # Stage 9: finalize_run                                             #
    # ─────────────────────────────────────────────────────────────── #
    finalized = finalize_run(
        run_dir  = run_dir,
        model_cfg = {"n_layer": 24, "n_embd": 2048},
        train_cfg = {"max_steps": 1000},
    )
    assert finalized.step == 1000          # latest checkpoint
    assert finalized.run_id == "run_smoke_000"
    assert finalized.eval_summary.get("perplexity") == 55.2   # best from history
    assert (run_dir / "model_metadata.json").exists()

    # ── Round-trip load from disk ─────────────────────────────────── #
    loaded = load_metadata(run_dir / "model_metadata.json")
    assert loaded.step == 1000
    assert loaded.eval_summary.get("perplexity") == 55.2
    print(f"finalize_run OK: step={finalized.step}  best_ppl={finalized.eval_summary.get('perplexity')}")

    # ─────────────────────────────────────────────────────────────── #
    # Stage 9: finalize with explicit checkpoint override               #
    # ─────────────────────────────────────────────────────────────── #
    finalized2 = finalize_run(
        run_dir         = run_dir,
        checkpoint_path = ckpt500,
    )
    assert finalized2.step == 500
    print(f"finalize_run explicit ckpt OK: step={finalized2.step}")

    # ─────────────────────────────────────────────────────────────── #
    # Config carries inference + analytics is loadable                  #
    # ─────────────────────────────────────────────────────────────── #
    from pipeline.orchestration.config_loader import ConfigLoader
    cfg = ConfigLoader().load("local-dev")
    assert "inference" in cfg
    print("Config inference section OK")

    # ─────────────────────────────────────────────────────────────── #
    # CLI parses analytics + finalize                                   #
    # ─────────────────────────────────────────────────────────────── #
    from pipeline.cli import _build_parser
    p = _build_parser()

    args_an = p.parse_args(["analytics", "--run-id", "run_smoke_000"])
    assert args_an.func.__name__ == "cmd_analytics"
    print("CLI analytics parse OK")

    args_fn = p.parse_args(["finalize", "--run-id", "run_smoke_000"])
    assert args_fn.func.__name__ == "cmd_finalize"
    print("CLI finalize parse OK")

    args_fn2 = p.parse_args(["finalize", "--run-id", "run_smoke_000",
                              "--checkpoint", str(ckpt500)])
    assert args_fn2.checkpoint == str(ckpt500)
    print("CLI finalize --checkpoint override OK")

    # ── All 9 stages visible in --help ───────────────────────────── #
    import io, contextlib
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            p.parse_args(["--help"])
    except SystemExit:
        pass
    help_text = buf.getvalue()
    for cmd in ("data-download", "data-profile", "data-select", "data-tokenise",
                "train-pretrain", "eval-checkpoint", "serve", "analytics", "finalize"):
        assert cmd in help_text, f"Missing '{cmd}' from --help"
    print("CLI --help shows all 9 stage commands OK")

print("\nALL Stage 8+9 smoke tests PASSED")
