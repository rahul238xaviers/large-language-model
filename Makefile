# =============================================================================
# GPT Pipeline — Makefile
# =============================================================================
# Two execution paths are provided:
#
#   DOCKER  (make build / make up)
#     Runs the Gradio frontend + data pipeline stages 1-3, 6, 8, 9 inside a
#     linux/arm64 container. No mlx — training and live inference are excluded.
#
#   NATIVE  (make run / make train / make <stage>)
#     Runs all 9 stages directly in the apple-silicon/.apple_env virtualenv.
#     Requires macOS Apple Silicon. Stages 5 and 7 (mlx) only work here.
#
# Quick start:
#   make build && make up      # Docker path
#   make install && make run   # Native path
# =============================================================================

SHELL        := /bin/bash
COMPOSE_FILE := deployment/docker/docker-compose.yml
VENV         := apple-silicon/.apple_env/bin/activate
PYTHON       := python3
PROFILE      ?= local-dev
RUN_ID       ?=
PORT         ?= 7860

.DEFAULT_GOAL := help

.PHONY: help \
        build up down restart logs shell ps \
        install run train \
        download profile select tokenise pretrain eval analytics finalize \
        clean check env-init

# ── Help ──────────────────────────────────────────────────────────────────────
help: ## Show all available targets
	@printf "\n\033[1mGPT Pipeline\033[0m — available make targets\n\n"
	@printf "  \033[33m%-22s\033[0m %s\n" "VARIABLE" "DEFAULT"
	@printf "  \033[33m%-22s\033[0m %s\n" "PROFILE" "$(PROFILE)"
	@printf "  \033[33m%-22s\033[0m %s\n" "PORT"    "$(PORT)"
	@printf "  \033[33m%-22s\033[0m %s\n" "RUN_ID"  "(empty — auto-generated)"
	@printf "\n"
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'
	@printf "\n"

# ── Docker ────────────────────────────────────────────────────────────────────
build: ## Build the Docker image (linux/arm64)
	docker compose -f $(COMPOSE_FILE) build

up: env-init ## Start the frontend container (detached)
	docker compose -f $(COMPOSE_FILE) up -d
	@printf "\n\033[32m✓ Gradio UI → http://localhost:$(PORT)\033[0m\n\n"

down: ## Stop and remove containers
	docker compose -f $(COMPOSE_FILE) down

restart: ## Restart running containers
	docker compose -f $(COMPOSE_FILE) restart

logs: ## Tail container logs (Ctrl-C to stop)
	docker compose -f $(COMPOSE_FILE) logs -f

shell: ## Open an interactive shell inside the running container
	docker compose -f $(COMPOSE_FILE) exec frontend /bin/sh

ps: ## Show container status
	docker compose -f $(COMPOSE_FILE) ps

# ── Native · Apple Silicon ────────────────────────────────────────────────────
install: ## Create / update the Python virtualenv and install all deps
	@if [ ! -f "$(VENV)" ]; then \
	  $(PYTHON) -m venv apple-silicon/.apple_env; \
	  echo "Virtualenv created."; \
	fi
	source $(VENV) && pip install --upgrade pip && \
	  pip install -r pipeline/requirements.txt
	@echo "Done — activate with: source $(VENV)"

run: ## Launch the Gradio frontend natively (mlx available, all 9 stages)
	source $(VENV) && $(PYTHON) -m frontend --port $(PORT)

# ── Pipeline stage shortcuts (native) ────────────────────────────────────────
# Override defaults with: make <target> PROFILE=m3-ultra-prod RUN_ID=run_xxx

download: ## Stage 1 · Download datasets to data/datasets/
	source $(VENV) && $(PYTHON) -m pipeline --profile $(PROFILE) data-download \
	  $(if $(RUN_ID),--run-id $(RUN_ID),)

profile: ## Stage 2 · Profile downloaded data
	source $(VENV) && $(PYTHON) -m pipeline --profile $(PROFILE) data-profile \
	  $(if $(RUN_ID),--run-id $(RUN_ID),)

select: ## Stage 3 · Score and filter data into run-specific arrow file
	source $(VENV) && $(PYTHON) -m pipeline --profile $(PROFILE) data-select \
	  $(if $(RUN_ID),--run-id $(RUN_ID),)

tokenise: ## Stage 4 · Tokenise selected data (cl100k_base by default)
	source $(VENV) && $(PYTHON) -m pipeline --profile $(PROFILE) data-tokenise \
	  $(if $(RUN_ID),--run-id $(RUN_ID),)

train: ## Stage 5 · Pre-train with mlx (Apple Silicon only — not available in Docker)
	source $(VENV) && $(PYTHON) -m pipeline --profile $(PROFILE) train-pretrain \
	  $(if $(RUN_ID),--run-id $(RUN_ID),)

pretrain: train ## Alias for make train

eval: ## Stage 6 · Evaluate checkpoints and write eval_history.json
	source $(VENV) && $(PYTHON) -m pipeline --profile $(PROFILE) eval-checkpoint \
	  $(if $(RUN_ID),--run-id $(RUN_ID),)

analytics: ## Stage 8 · Generate training and eval plots
	source $(VENV) && $(PYTHON) -m pipeline --profile $(PROFILE) analytics \
	  $(if $(RUN_ID),--run-id $(RUN_ID),)

finalize: ## Stage 9 · Write model_metadata.json and register the run
	source $(VENV) && $(PYTHON) -m pipeline --profile $(PROFILE) finalize \
	  $(if $(RUN_ID),--run-id $(RUN_ID),)

# ── Utility ───────────────────────────────────────────────────────────────────
env-init: ## Copy env/.env.example → env/.env if not present
	@if [ ! -f env/.env ]; then \
	  cp env/.env.example env/.env; \
	  printf "\033[33mCreated env/.env from example — set HF_TOKEN before running.\033[0m\n"; \
	fi

clean: ## Remove Python bytecode and __pycache__ directories
	find . \( -path './.git' -o -path './apple-silicon/.apple_env' -o -path './.venv' \) \
	  -prune -o \( -type d -name '__pycache__' -print \) | xargs -r rm -rf
	find . -name '*.pyc' -not -path './.git/*' -delete

check: ## AST syntax-check all Python source files
	source $(VENV) && $(PYTHON) deployment/scripts/syntax_check.py
