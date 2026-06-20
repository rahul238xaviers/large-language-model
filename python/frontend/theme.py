"""Design system for the GPT Pipeline UI.

Provides:
- ``THEME``  — a Gradio Soft theme with Rust/MLX brand palette
- ``CSS``    — custom stylesheet injected via ``gr.Blocks(css=CSS)``
- ``COLORS`` — shared colour constants used in Matplotlib figures
"""
from __future__ import annotations

import gradio as gr
import gradio.themes as gt

# ── Brand palette ──────────────────────────────────────────────────────
COLORS = {
    "accent":     "#f97316",   # Rust orange
    "accent_dark":"#c2410c",   # Deep orange
    "success":    "#10b981",   # Teal green
    "danger":     "#ef4444",   # Crimson
    "info":       "#8b5cf6",   # Indigo
    "muted":      "#a8a29e",   # Stone muted
    "bg":         "#0c0a09",   # Near-black
    "surface":    "#1c1917",   # Dark stone
    "border":     "#292524",   # Stone border
    "text":       "#e7e5e4",   # Light stone
    "grid":       "#292524",   # Chart grid
}

# ── Gradio theme ───────────────────────────────────────────────────────
THEME = gt.Soft(
    primary_hue   = "orange",
    secondary_hue = "slate",
    neutral_hue   = "stone",
    font      = [gt.GoogleFont("Outfit"),    "sans-serif"],
    font_mono = [gt.GoogleFont("Fira Code"), "monospace"],
).set(
    body_background_fill           = COLORS["bg"],
    body_background_fill_dark      = COLORS["bg"],
    block_background_fill          = COLORS["surface"],
    block_background_fill_dark     = COLORS["surface"],
    block_border_width             = "1px",
    block_border_color             = COLORS["border"],
    block_shadow                   = "0 4px 24px 0 rgba(0,0,0,0.45)",
    button_primary_background_fill = f"linear-gradient(90deg, {COLORS['accent']} 0%, {COLORS['accent_dark']} 100%)",
    button_primary_background_fill_hover = f"linear-gradient(90deg, {COLORS['accent_dark']} 0%, #9a3412 100%)",
    button_primary_text_color      = "#ffffff",
    slider_color                   = COLORS["accent"],
    input_background_fill          = "#14100e",
    input_background_fill_dark     = "#14100e",
    input_border_color             = COLORS["border"],
    checkbox_background_color      = "#14100e",
    table_even_background_fill     = "#14100e",
    table_odd_background_fill      = COLORS["surface"],
)

# ── Custom CSS ─────────────────────────────────────────────────────────
CSS = """
/* ── Global resets ─────────────────────────────────────── */
* { box-sizing: border-box; }

/* ── App header ─────────────────────────────────────────── */
.pipeline-header {
    background: linear-gradient(135deg, #14100e 0%, #1c1917 100%);
    border-bottom: 1px solid #292524;
    padding: 1.25rem 2rem 1rem;
    margin-bottom: 0;
}
.pipeline-title {
    background: linear-gradient(90deg, #f97316 0%, #ea580c 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 1.75rem;
    letter-spacing: -0.02em;
    line-height: 1.2;
}
.pipeline-subtitle {
    color: #a8a29e;
    font-size: 0.9rem;
    margin-top: 2px;
}

/* ── Stage badges ────────────────────────────────────────── */
.stage-badge {
    display: inline-block;
    background: rgba(249,115,22,0.15);
    border: 1px solid rgba(249,115,22,0.35);
    color: #f97316;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 2px 8px;
    border-radius: 99px;
    margin-bottom: 6px;
}
.stage-badge.success {
    background: rgba(16,185,129,0.12);
    border-color: rgba(16,185,129,0.35);
    color: #10b981;
}
.stage-badge.muted {
    background: rgba(168,162,158,0.10);
    border-color: rgba(168,162,158,0.25);
    color: #a8a29e;
}

/* ── Panel glass card ────────────────────────────────────── */
.panel-glass {
    background: rgba(28,25,23,0.70) !important;
    backdrop-filter: blur(18px);
    border: 1px solid rgba(249,115,22,0.12) !important;
    border-radius: 12px !important;
    padding: 14px !important;
    margin-bottom: 10px;
}

/* ── Run-ID pill ─────────────────────────────────────────── */
.run-id-pill {
    font-family: 'Fira Code', monospace;
    font-size: 0.8rem;
    background: #14100e;
    border: 1px solid #292524;
    color: #f97316;
    padding: 3px 10px;
    border-radius: 99px;
    letter-spacing: 0.03em;
    display: inline-block;
    margin-top: 4px;
}

/* ── Status indicators ───────────────────────────────────── */
.status-idle    { color: #a8a29e; }
.status-running { color: #f97316; }
.status-success { color: #10b981; }
.status-error   { color: #ef4444; }

/* ── Log terminal ────────────────────────────────────────── */
.log-terminal textarea {
    font-family: 'Fira Code', 'Courier New', monospace !important;
    font-size: 0.78rem !important;
    background: #0c0a09 !important;
    color: #d6d3d1 !important;
    border: 1px solid #292524 !important;
    resize: vertical;
}

/* ── Metric cards row ────────────────────────────────────── */
.metric-card {
    background: #14100e;
    border: 1px solid #292524;
    border-radius: 10px;
    padding: 14px 18px;
    text-align: center;
}
.metric-value {
    font-size: 1.6rem;
    font-weight: 800;
    color: #f97316;
    font-family: 'Fira Code', monospace;
    display: block;
}
.metric-label {
    font-size: 0.78rem;
    color: #a8a29e;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    margin-top: 2px;
    display: block;
}

/* ── Tab accent ──────────────────────────────────────────── */
.tab-nav button.selected {
    border-bottom: 2px solid #f97316 !important;
    color: #f97316 !important;
}

/* ── Primary / danger buttons ────────────────────────────── */
.btn-danger {
    background: linear-gradient(90deg, #ef4444 0%, #dc2626 100%) !important;
}
.btn-success {
    background: linear-gradient(90deg, #10b981 0%, #059669 100%) !important;
}

/* ── Workflow progress bar ───────────────────────────────── */
.progress-track {
    display: flex;
    gap: 4px;
    align-items: center;
    margin: 4px 0 10px;
}
.progress-dot {
    width: 10px; height: 10px;
    border-radius: 50%;
    background: #292524;
    transition: background 0.3s;
}
.progress-dot.done    { background: #10b981; }
.progress-dot.active  { background: #f97316; box-shadow: 0 0 8px #f97316; }

/* ── Chat bubbles ────────────────────────────────────────── */
.chatbot .user    { background: rgba(249,115,22,0.12) !important; border-radius: 10px !important; }
.chatbot .bot     { background: rgba(28,25,23,0.80)  !important; border-radius: 10px !important; }
"""
