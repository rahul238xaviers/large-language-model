#!/usr/bin/env sh
# =============================================================================
# Docker entrypoint for the GPT Pipeline frontend.
# =============================================================================
# Injects the legacy apple-silicon/src/ into PYTHONPATH so pipeline modules
# can import GPTModel without modifying the legacy source tree.
# All arguments passed to `docker run` (or docker-compose command:) are
# forwarded to `python3 -m frontend`.
# =============================================================================
set -e

export PYTHONPATH="/app/apple-silicon/src:/app${PYTHONPATH:+:$PYTHONPATH}"

exec python3 -m frontend "$@"
