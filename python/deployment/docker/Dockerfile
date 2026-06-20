# =============================================================================
# GPT Pipeline — Gradio Frontend + Data Stages
# =============================================================================
# Runs on Apple Silicon Mac via Docker Desktop (linux/arm64 platform).
#
# IMPORTANT: mlx (Apple Metal) is NOT available inside Linux containers.
#   Stage 5  (pre-training)   → run natively via: make train
#   Stage 7  (live inference) → run natively via: make run
#   All other stages + the Gradio UI work fully in this image.
# =============================================================================

FROM python:3.13-slim

LABEL maintainer="rahul238xaviers@gmail.com"
LABEL description="GPT Pipeline — Gradio frontend and data pipeline stages (non-mlx)"

# --- System dependencies ---
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        curl \
        git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# --- Python dependencies (installed early for layer-cache efficiency) ---
COPY deployment/docker/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# --- Application code ---
# pipeline/  — 9-stage ML pipeline (data + eval + analytics + finalize)
# frontend/  — Gradio UI
# configs/   — YAML base configs and profiles (mounted at runtime, but baked in as fallback)
# apple-silicon/src/ — legacy GPTModel + utils (imported by pipeline at runtime)
COPY pipeline/          ./pipeline/
COPY frontend/          ./frontend/
COPY configs/           ./configs/
COPY apple-silicon/src/ ./apple-silicon/src/

# --- Entrypoint ---
COPY deployment/scripts/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Gradio binds to this port inside the container
EXPOSE 7860

# Health check: Gradio serves an HTTP root at /
HEALTHCHECK --interval=30s --timeout=10s --start-period=25s --retries=3 \
    CMD curl -sf http://localhost:7860/ > /dev/null || exit 1

ENTRYPOINT ["/entrypoint.sh"]
