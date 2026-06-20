"""Stages 7 + 9: Inference serving and model registry (finalize) tab.

Stage 7 (serve): loads a checkpoint in-process via ``InferenceServer``,
exposes decode-parameter sliders, and streams generated tokens directly
into a Gradio Chatbot component — no separate server process needed.

Stage 9 (finalize): invokes ``pipeline finalize`` via subprocess to write
``model_metadata.json``, then displays the metadata summary.

Public symbol: ``render_inference_tab(state)``
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Iterator

import gradio as gr

from frontend.state import (
    STAGE_DONE,
    STAGE_ERROR,
    STAGE_RUNNING,
    mark_stage,
    runs_with_metadata,
    update_state,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ── Module-level server singleton ─────────────────────────────────────
# Kept here so the server survives Gradio callback re-invocations.
_server_lock   = threading.Lock()
_active_server: "Any | None" = None   # InferenceServer instance or None
_active_ckpt:   str          = ""     # path of the currently loaded checkpoint


def _get_or_load_server(
    checkpoint_path: str,
    model_cfg: dict,
    inference_cfg: dict,
) -> "Any":
    """Return a loaded InferenceServer, reusing it if the same checkpoint is already loaded."""
    global _active_server, _active_ckpt
    with _server_lock:
        if _active_server is not None and _active_ckpt == checkpoint_path:
            return _active_server
        from pipeline.inference.model_server import InferenceServer
        server = InferenceServer(
            checkpoint_path = checkpoint_path,
            model_cfg       = model_cfg,
            inference_cfg   = inference_cfg,
        )
        server.load()
        _active_server = server
        _active_ckpt   = checkpoint_path
        return server


def _unload_server() -> None:
    global _active_server, _active_ckpt
    with _server_lock:
        _active_server = None
        _active_ckpt   = ""


# ── Subprocess streamer (for Stage 9) ─────────────────────────────────

def _stream_pipeline(
    cmd: list[str],
    env: dict | None = None,
) -> Iterator[tuple[str, int | None]]:
    merged_env = {**os.environ, **(env or {})}
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        cwd=str(_REPO_ROOT),
        env=merged_env,
    )
    output = ""
    assert proc.stdout is not None
    for line in proc.stdout:
        output += line
        yield output, None
    proc.wait()
    yield output, proc.returncode


# ── Checkpoint discovery ───────────────────────────────────────────────

def _discover_checkpoints(run_id: str, runs_dir: str) -> list[str]:
    if not run_id:
        return []
    ckpt_dir = _REPO_ROOT / runs_dir / run_id / "training" / "checkpoints"
    if not ckpt_dir.exists():
        return []
    return sorted([f.name for f in ckpt_dir.glob("step_*.safetensors")])


def _full_ckpt_path(run_id: str, runs_dir: str, ckpt_name: str) -> str:
    return str(_REPO_ROOT / runs_dir / run_id / "training" / "checkpoints" / ckpt_name)


# ── Stage 7 callbacks ──────────────────────────────────────────────────

def _on_load_model(
    ckpt_choice: str,
    state: dict,
) -> tuple[str, str]:
    """Load the chosen checkpoint into the in-process InferenceServer."""
    run_id   = state.get("run_id", "")
    runs_dir = state.get("runs_dir", "runs")

    if not ckpt_choice or not run_id:
        return (
            "<span class='status-error'>●  No checkpoint selected</span>",
            "Model not loaded.",
        )

    ckpt_path = _full_ckpt_path(run_id, runs_dir, ckpt_choice)
    if not Path(ckpt_path).exists():
        return (
            "<span class='status-error'>●  File not found</span>",
            f"Checkpoint not found: {ckpt_path}",
        )

    try:
        cfg        = state.get("config", {})
        model_cfg  = cfg.get("model", {})
        inf_cfg    = cfg.get("inference", {})
        _get_or_load_server(ckpt_path, model_cfg, inf_cfg)
        short_name = Path(ckpt_path).name
        return (
            f"<span class='status-success'>✓  Loaded: {short_name}</span>",
            f"Model ready — {ckpt_path}",
        )
    except Exception as exc:
        _unload_server()
        return (
            "<span class='status-error'>●  Load failed</span>",
            f"Error loading model:\n{exc}",
        )


def _on_generate(
    history: list[list[str | None]],
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    rep_penalty: float,
) -> Iterator[tuple[list[list[str | None]], str]]:
    """Stream generated tokens into the Chatbot history."""
    if not prompt.strip():
        yield history, ""
        return

    with _server_lock:
        server = _active_server

    if server is None:
        history = history + [[prompt, "⚠ Model not loaded. Use 'Load model' first."]]
        yield history, ""
        return

    # Add user message immediately
    history = history + [[prompt, ""]]
    yield history, ""

    accumulated = ""
    try:
        for token_text in server.generate_text(
            prompt,
            max_new_tokens     = max_new_tokens,
            temperature        = temperature,
            top_k              = top_k,
            top_p              = top_p,
            repetition_penalty = rep_penalty,
        ):
            accumulated += token_text
            history[-1][1] = accumulated
            yield history, ""
    except Exception as exc:
        history[-1][1] = f"⚠ Generation error: {exc}"
        yield history, ""


def _on_clear_chat() -> list:
    return []


# ── Stage 9 callback ───────────────────────────────────────────────────

def _on_finalize(
    ckpt_choice: str,
    profile: str,
    state: dict,
) -> Iterator[tuple]:
    run_id = state.get("run_id", "")
    if not run_id:
        yield (
            "⚠  No active run. Complete Stage 5 (Pre-train) first.",
            "<span class='status-error'>●  Error</span>",
            state,
            _meta_placeholder(),
        )
        return

    cmd = [
        sys.executable, "-m", "pipeline",
        "--profile", profile,
        "finalize",
        "--run-id", run_id,
    ]

    runs_dir  = state.get("runs_dir", "runs")
    if ckpt_choice:
        ckpt_path = _full_ckpt_path(run_id, runs_dir, ckpt_choice)
        cmd += ["--checkpoint", ckpt_path]

    new_state    = mark_stage(state, "finalize", STAGE_RUNNING)
    running_html = "<span class='status-running'>⟳  Finalizing…</span>"
    meta_path    = _REPO_ROOT / runs_dir / run_id / "model_metadata.json"

    for output, returncode in _stream_pipeline(cmd):
        if returncode is None:
            yield output, running_html, new_state, _meta_placeholder()
        else:
            stage_st  = STAGE_DONE if returncode == 0 else STAGE_ERROR
            new_state = mark_stage(
                update_state(new_state, metadata_path=str(meta_path)),
                "finalize", stage_st,
            )
            status_html = (
                "<span class='status-success'>✓  Finalized</span>"
                if returncode == 0
                else f"<span class='status-error'>✗  Exit {returncode}</span>"
            )
            suffix = (
                f"\n\n[Stage 9 complete — metadata: {meta_path}]"
                if returncode == 0
                else f"\n\n[Stage 9 failed — exit code {returncode}]"
            )
            meta_html = _render_metadata(meta_path) if returncode == 0 else _meta_placeholder()
            yield output + suffix, status_html, new_state, meta_html


# ── Metadata renderer ──────────────────────────────────────────────────

def _meta_placeholder() -> str:
    return (
        "<div style='padding:16px;text-align:center;color:#a8a29e;font-size:0.85rem;'>"
        "Metadata will appear here after Stage 9 completes."
        "</div>"
    )


def _render_metadata(meta_path: Path) -> str:
    if not meta_path.exists():
        return _meta_placeholder()
    try:
        m = json.loads(meta_path.read_text())

        def _card(value: str, label: str) -> str:
            return (
                f"<div class='metric-card'>"
                f"<span class='metric-value' style='font-size:1.1rem;'>{value}</span>"
                f"<span class='metric-label'>{label}</span>"
                f"</div>"
            )

        eval_s = m.get("eval_summary", {})
        ppl    = eval_s.get("perplexity", None)
        loss   = eval_s.get("mean_loss", None)
        step   = m.get("step", "—")

        cards_html = (
            _card(str(int(step)) if isinstance(step, (int, float)) else str(step), "Step")
            + (_card(f"{ppl:.2f}", "Best PPL")  if isinstance(ppl,  (int, float)) else "")
            + (_card(f"{loss:.4f}", "Best Loss") if isinstance(loss, (int, float)) else "")
            + _card(m.get("pipeline_version", "1.0"), "Version")
        )
        cards_row = (
            f"<div style='display:flex;gap:10px;flex-wrap:wrap;margin-bottom:14px;'>"
            f"{cards_html}</div>"
        )

        ckpt_short = Path(m.get("checkpoint_path", "")).name or "—"
        details = (
            f"<table style='width:100%;border-collapse:collapse;font-size:0.82rem;'>"
            f"<tr><td style='padding:4px 10px;color:#a8a29e;width:160px;'>Run ID</td>"
            f"<td style='padding:4px 10px;font-family:\"Fira Code\",monospace;color:#f97316;'>{m.get('run_id','—')}</td></tr>"
            f"<tr style='background:#14100e;'>"
            f"<td style='padding:4px 10px;color:#a8a29e;'>Checkpoint</td>"
            f"<td style='padding:4px 10px;font-family:\"Fira Code\",monospace;color:#e7e5e4;'>{ckpt_short}</td></tr>"
            f"<tr><td style='padding:4px 10px;color:#a8a29e;'>Created at</td>"
            f"<td style='padding:4px 10px;color:#e7e5e4;'>{m.get('created_at','—')}</td></tr>"
            f"</table>"
        )
        return cards_row + details
    except Exception as exc:
        return f"<span style='color:#ef4444;font-size:0.8rem;'>Error reading metadata: {exc}</span>"


# ── Registry browser helper ────────────────────────────────────────────

def _registry_rows(runs_dir: str) -> list[list[str]]:
    """Return table rows for all finalized runs with metadata."""
    rows = []
    for m in runs_with_metadata(runs_dir):
        eval_s = m.get("eval_summary", {})
        ppl    = eval_s.get("perplexity", "—")
        rows.append([
            m.get("run_id",   "—"),
            str(m.get("step", "—")),
            f"{ppl:.2f}" if isinstance(ppl, float) else str(ppl),
            Path(m.get("checkpoint_path", "")).name or "—",
            m.get("created_at", "—")[:19],
        ])
    return rows or [["No finalized runs found", "", "", "", ""]]


def _badge(label: str, variant: str = "") -> str:
    cls = f"stage-badge {variant}".strip()
    return f"<span class='{cls}'>{label}</span>"


# ── Main render function ───────────────────────────────────────────────

def render_inference_tab(state: gr.State) -> None:
    """Render Stage 7 (serve/chat) and Stage 9 (finalize) inside the current Blocks context."""

    with gr.Column():

        gr.HTML(
            _badge("Stage 7 · 9", "")
            + "<h3 style='margin:6px 0 2px;color:#e7e5e4;'>Inference</h3>"
            "<p style='color:#a8a29e;font-size:0.85rem;margin:0 0 12px;'>"
            "Load a checkpoint and stream generated text, then finalize the run into the model registry."
            "</p>"
        )

        with gr.Tabs():

            # ── Stage 7: Chat playground ──────────────────────────── #
            with gr.Tab("💬  Stage 7 · Playground"):

                with gr.Row():
                    with gr.Column(scale=2):
                        ckpt_dd_serve = gr.Dropdown(
                            choices     = [],
                            value       = None,
                            label       = "Checkpoint",
                            interactive = True,
                            info        = "Refresh after Stage 5 completes.",
                        )
                        refresh_serve_btn = gr.Button("↻  Refresh checkpoints", size="sm", variant="secondary")

                    with gr.Column(scale=3):
                        load_btn    = gr.Button("⚡  Load model", variant="primary")
                        load_status = gr.HTML(
                            "<span class='status-idle'>●  No model loaded</span>",
                            label="",
                        )
                        load_info   = gr.Textbox(
                            label="",
                            interactive=False,
                            placeholder="checkpoint path will appear here…",
                            lines=1,
                        )

                gr.Markdown("---")
                gr.Markdown("#### Decode parameters")

                with gr.Row():
                    max_tokens_slider = gr.Slider(
                        16, 1024, step=16, value=128,
                        label="Max new tokens",
                    )
                    temperature_slider = gr.Slider(
                        0.01, 2.0, step=0.01, value=0.7,
                        label="Temperature",
                    )
                    top_k_slider = gr.Slider(
                        1, 200, step=1, value=50,
                        label="Top-K",
                    )

                with gr.Row():
                    top_p_slider = gr.Slider(
                        0.0, 1.0, step=0.01, value=0.9,
                        label="Top-P",
                    )
                    rep_penalty_slider = gr.Slider(
                        1.0, 2.0, step=0.01, value=1.15,
                        label="Repetition penalty",
                    )

                gr.Markdown("---")
                gr.Markdown("#### Chat")

                chatbot = gr.Chatbot(
                    label="",
                    height=420,
                    elem_classes=["chatbot"],
                    avatar_images=(None, None),
                )

                with gr.Row():
                    prompt_box  = gr.Textbox(
                        label="",
                        placeholder="Enter a prompt and press Enter or click Generate…",
                        lines=2,
                        scale=5,
                        show_label=False,
                    )
                    generate_btn = gr.Button("▶  Generate", variant="primary", scale=1)
                    clear_btn    = gr.Button("🗑  Clear", variant="secondary", scale=1)

            # ── Stage 9: Finalize ─────────────────────────────────── #
            with gr.Tab("🏁  Stage 9 · Finalize"):

                with gr.Row():
                    with gr.Column(scale=3):
                        ckpt_dd_final = gr.Dropdown(
                            choices     = [],
                            value       = None,
                            label       = "Checkpoint to finalize  (leave blank for latest)",
                            interactive = True,
                        )
                        refresh_final_btn = gr.Button("↻  Refresh checkpoints", size="sm", variant="secondary")

                    with gr.Column(scale=2, elem_classes=["panel-glass"]):
                        gr.Markdown(
                            "Finalizing picks the best checkpoint, summarises eval history, "
                            "and writes `model_metadata.json` into the run directory.  \n"
                            "This file is used by the Stage 7 server and Stage 8 dashboard."
                        )

                with gr.Row():
                    finalize_btn    = gr.Button("🏁  Finalize run", variant="primary", scale=2)
                    finalize_status = gr.HTML("<span class='status-idle'>●  Idle</span>", label="", scale=1)

                finalize_log = gr.Textbox(
                    label="Finalize log",
                    lines=8, max_lines=20,
                    interactive=False,
                    elem_classes=["log-terminal"],
                )

                metadata_panel = gr.HTML(_meta_placeholder(), label="Model metadata")

                # ── Registry browser ──────────────────────────────── #
                gr.Markdown("#### Model Registry  (all finalized runs)")

                registry_table = gr.Dataframe(
                    headers = ["Run ID", "Step", "PPL", "Checkpoint", "Created"],
                    value   = [["—", "", "", "", ""]],
                    interactive = False,
                    wrap        = False,
                )
                refresh_registry_btn = gr.Button("↻  Refresh registry", size="sm", variant="secondary")

    # ── Hidden state mirrors ─────────────────────────────────────────── #
    hidden_profile = gr.Textbox(value="local-dev", visible=False)

    def _sync_hidden(s: dict) -> str:
        return s.get("profile", "local-dev")

    state.change(fn=_sync_hidden, inputs=[state], outputs=[hidden_profile])

    # ── Checkpoint refresh (serve tab) ───────────────────────────────── #
    def _refresh_serve(s: dict) -> object:
        run_id   = s.get("run_id", "")
        runs_dir = s.get("runs_dir", "runs")
        choices  = _discover_checkpoints(run_id, runs_dir)
        val      = choices[-1] if choices else None
        return gr.update(choices=choices, value=val)

    refresh_serve_btn.click(fn=_refresh_serve, inputs=[state], outputs=[ckpt_dd_serve])
    state.change(fn=_refresh_serve, inputs=[state], outputs=[ckpt_dd_serve])

    # ── Checkpoint refresh (finalize tab) ───────────────────────────── #
    def _refresh_final(s: dict) -> object:
        run_id   = s.get("run_id", "")
        runs_dir = s.get("runs_dir", "runs")
        choices  = ["latest (auto)"] + _discover_checkpoints(run_id, runs_dir)
        return gr.update(choices=choices, value=choices[0])

    refresh_final_btn.click(fn=_refresh_final, inputs=[state], outputs=[ckpt_dd_final])
    state.change(fn=_refresh_final, inputs=[state], outputs=[ckpt_dd_final])

    # ── Load model ───────────────────────────────────────────────────── #
    load_btn.click(
        fn      = _on_load_model,
        inputs  = [ckpt_dd_serve, state],
        outputs = [load_status, load_info],
    )

    # ── Generate (streaming) ─────────────────────────────────────────── #
    generate_btn.click(
        fn      = _on_generate,
        inputs  = [
            chatbot, prompt_box,
            max_tokens_slider, temperature_slider,
            top_k_slider, top_p_slider, rep_penalty_slider,
        ],
        outputs = [chatbot, prompt_box],
    )
    prompt_box.submit(
        fn      = _on_generate,
        inputs  = [
            chatbot, prompt_box,
            max_tokens_slider, temperature_slider,
            top_k_slider, top_p_slider, rep_penalty_slider,
        ],
        outputs = [chatbot, prompt_box],
    )
    clear_btn.click(fn=_on_clear_chat, outputs=[chatbot])

    # ── Finalize ─────────────────────────────────────────────────────── #
    def _finalize_ckpt(choice: str) -> str:
        """Strip the 'latest (auto)' sentinel before passing to callback."""
        return "" if (not choice or choice == "latest (auto)") else choice

    def _on_finalize_wrapped(ckpt_choice, profile, s):
        for result in _on_finalize(_finalize_ckpt(ckpt_choice), profile, s):
            yield result

    finalize_btn.click(
        fn      = _on_finalize_wrapped,
        inputs  = [ckpt_dd_final, hidden_profile, state],
        outputs = [finalize_log, finalize_status, state, metadata_panel],
    )

    # ── Registry refresh ─────────────────────────────────────────────── #
    def _refresh_registry(s: dict) -> list[list[str]]:
        return _registry_rows(s.get("runs_dir", "runs"))

    refresh_registry_btn.click(fn=_refresh_registry, inputs=[state], outputs=[registry_table])

    # Auto-populate registry when state changes (e.g. after finalize)
    state.change(fn=_refresh_registry, inputs=[state], outputs=[registry_table])
