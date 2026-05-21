# ===========================================================================
# Makefile — Developer Ergonomics for the AI-Driven Dynamic Pricing System
# ===========================================================================
#
# Every target here is a thin wrapper around a Docker, pytest, or Python
# command. The goal is a single source of truth for how you run things —
# no more "wait, what was the exact docker compose flag?" in the PR review.
#
# Usage:
#   make <target>
#
# Run `make help` (or just `make`) for a list of all available targets.
# ---------------------------------------------------------------------------

# The name that shows up in `docker compose ps` and in the health-check step
# of the CI pipeline. Keep this consistent with docker-compose.yml.
IMAGE_NAME := pricing-system

# The Python interpreter to use for local (non-Docker) commands.
# Override from the shell if you use pyenv or a different venv path:
#   make test PYTHON=python3.10
PYTHON := python

# The virtual environment directory. All local commands activate this first.
VENV := .venv

.DEFAULT_GOAL := help

# ---------------------------------------------------------------------------
# Self-documenting help target.
#
# This works by scanning the Makefile for targets that have a ## comment on
# the same line. `awk` extracts the target name and comment and formats them
# into a tidy table. This is the "make help" convention used by many major
# open-source projects (Kubernetes, Prometheus, etc.) — it means the
# Makefile IS the documentation, not a separate wiki page that goes stale.
# ---------------------------------------------------------------------------
.PHONY: help
help: ## Show this help message and exit
	@echo ""
	@echo "  AI-Driven Dynamic Pricing System — Developer Commands"
	@echo "  ======================================================"
	@awk 'BEGIN {FS = ":.*##"; printf ""} \
	      /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 }' \
	     $(MAKEFILE_LIST)
	@echo ""

# ---------------------------------------------------------------------------
# Docker targets
#
# These mirror what CI does, so if `make build` passes locally, `make build`
# will pass in GitHub Actions too.  The only difference is that CI uses a
# distinct image tag ('ci') while local builds use 'latest' for convenience.
# ---------------------------------------------------------------------------

.PHONY: build
build: ## Build the Docker image (tag: pricing-system:latest)
	@echo "→ Building Docker image '$(IMAGE_NAME):latest'..."
	docker build --tag $(IMAGE_NAME):latest --file Dockerfile .
	@echo "✓ Build complete."

.PHONY: up
up: ## Start the app in detached mode (docker compose up -d)
	@echo "→ Starting services in detached mode..."
	docker compose up --detach
	@echo "✓ App is running at http://localhost:8501"
	@echo "  Run 'make logs' to tail the output."

.PHONY: down
down: ## Stop and remove containers, networks (docker compose down)
	@echo "→ Stopping services..."
	docker compose down
	@echo "✓ All containers stopped and removed."

.PHONY: logs
logs: ## Tail live logs from the running container (Ctrl+C to stop)
	@echo "→ Tailing Docker logs (Ctrl+C to stop)..."
	docker compose logs --follow

# ---------------------------------------------------------------------------
# Local development targets
#
# These run against the local .venv rather than Docker.  This is faster for
# tight edit-run-test loops during active development.  The venv is created
# by `make install`; `make test` assumes it already exists.
# ---------------------------------------------------------------------------

.PHONY: install
install: ## Create .venv and install all dependencies (first-time setup)
	@echo "→ Creating virtual environment at $(VENV)/..."
	$(PYTHON) -m venv $(VENV)
	@echo "→ Installing production dependencies from requirements.txt..."
	$(VENV)/bin/pip install --quiet --upgrade pip
	$(VENV)/bin/pip install --quiet -r requirements.txt
	@echo "→ Installing test dependencies (pytest, pytest-cov)..."
	$(VENV)/bin/pip install --quiet pytest pytest-cov
	@echo "✓ Environment ready. Activate with: source $(VENV)/bin/activate"

.PHONY: test
test: ## Run the full pytest suite with coverage (requires active venv or `make install`)
	@echo "→ Running pytest suite..."
	$(VENV)/bin/python -m pytest \
		--cov=src \
		--cov-report=term-missing \
		-v
	@echo "✓ All tests complete."

.PHONY: test-fast
test-fast: ## Run pytest without coverage reporting (faster feedback loop)
	@echo "→ Running pytest (no coverage)..."
	$(VENV)/bin/python -m pytest -q

.PHONY: generate-data
generate-data: ## Run the synthetic data generation script (outputs to data/)
	@echo "→ Generating synthetic ride data..."
	$(VENV)/bin/python scripts/generate_data.py
	@echo "✓ Data written to data/. Check the scripts/generate_data.py header"
	@echo "  for flags like --samples and --output-dir."

.PHONY: run-local
run-local: ## Launch the Streamlit app locally (no Docker required)
	@echo "→ Starting Streamlit on http://localhost:8501 ..."
	$(VENV)/bin/streamlit run src/app.py \
		--server.port 8501 \
		--server.address localhost

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

.PHONY: clean
clean: ## Remove __pycache__, .pytest_cache, coverage artefacts, and stale .joblib files
	@echo "→ Cleaning build artefacts..."
	find . -type d -name "__pycache__" -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "coverage.xml" -delete 2>/dev/null || true
	find . -name ".coverage"    -delete 2>/dev/null || true
	find . -name "*.joblib" -not -path "./.venv/*" -delete 2>/dev/null || true
	@echo "✓ Clean complete."
