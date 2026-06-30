# =============================================================================
# AI Automation Platform — Makefile
# Provides common development and operational commands.
# =============================================================================

.PHONY: help install install-dev lint format type-check test test-cov \
        docker-up docker-down docker-build migrate db-shell clean \
        run dev

# Default shell
SHELL := /bin/bash

# Python interpreter
PYTHON := python3
PIP    := pip3

# Application
APP_MODULE := app.main:app
HOST       := 0.0.0.0
PORT       := 8000

help: ## Show this help message
	@echo "AI Automation Platform — available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Setup
# =============================================================================

install: ## Install production dependencies
	$(PIP) install -r requirements/prod.txt

install-dev: ## Install development dependencies
	$(PIP) install -r requirements/dev.txt
	pre-commit install

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run Ruff linter
	ruff check app/ tests/

lint-fix: ## Run Ruff linter with auto-fix
	ruff check --fix app/ tests/

format: ## Format code with Black
	black app/ tests/

format-check: ## Check formatting without modifying files
	black --check app/ tests/

type-check: ## Run MyPy type checking
	mypy app/

check: lint format-check type-check ## Run all code quality checks

# =============================================================================
# Testing
# =============================================================================

test: ## Run all unit tests
	pytest tests/ -v --asyncio-mode=auto

test-cov: ## Run tests with coverage report
	pytest tests/ -v --asyncio-mode=auto --cov=app --cov-report=html --cov-report=term-missing

test-ai: ## Run AI provider tests only
	pytest tests/ai/ -v

test-rag: ## Run RAG pipeline tests only
	pytest tests/rag/ -v

test-integration: ## Run integration tests
	pytest tests/integration/ -v --asyncio-mode=auto

# =============================================================================
# Development Server
# =============================================================================

run: ## Start production server (gunicorn)
	gunicorn $(APP_MODULE) \
		--worker-class uvicorn.workers.UvicornWorker \
		--workers 4 \
		--bind $(HOST):$(PORT) \
		--timeout 120

dev: ## Start development server with hot reload
	uvicorn $(APP_MODULE) --host $(HOST) --port $(PORT) --reload --log-level debug

# =============================================================================
# Docker
# =============================================================================

docker-build: ## Build Docker images
	docker compose build

docker-up: ## Start all services
	docker compose up -d

docker-down: ## Stop all services
	docker compose down

docker-logs: ## Follow API container logs
	docker compose logs -f api

docker-restart: ## Restart the API container
	docker compose restart api

docker-clean: ## Remove containers, networks, and volumes
	docker compose down -v --remove-orphans

# =============================================================================
# Database
# =============================================================================

migrate: ## Apply pending database migrations
	alembic upgrade head

migrate-create: ## Create a new migration (use: make migrate-create MSG="description")
	alembic revision --autogenerate -m "$(MSG)"

migrate-rollback: ## Roll back the last migration
	alembic downgrade -1

migrate-history: ## Show migration history
	alembic history --verbose

db-shell: ## Open a psql shell to the database
	docker compose exec postgres psql -U postgres -d ai_platform

# =============================================================================
# Utilities
# =============================================================================

clean: ## Remove Python cache files and build artifacts
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache htmlcov .mypy_cache .ruff_cache dist build

env-example: ## Copy .env.example to .env (will not overwrite)
	@test -f .env || (cp .env.example .env && echo "Created .env from .env.example")

pre-commit-run: ## Run pre-commit hooks on all files
	pre-commit run --all-files

logs: ## Tail application logs
	tail -f logs/app.log
