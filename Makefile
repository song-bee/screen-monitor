# ASAM Development Makefile

# Variables
PYTHON = python3
VENV = venv
VENV_ACTIVATE = source $(VENV)/bin/activate
PIP = $(VENV_ACTIVATE) && pip
PYTEST = $(VENV_ACTIVATE) && python -m pytest
ASAM = $(VENV_ACTIVATE) && python -m asam.main

.PHONY: help install dev-install clean test lint format setup run

# Default target
help:
	@echo "ASAM Development Commands:"
	@echo ""
	@echo "  setup         - Set up development environment"
	@echo "  install       - Install package dependencies"
	@echo "  dev-install   - Install development dependencies"
	@echo "  test          - Run test suite"
	@echo "  test-cov      - Run tests with coverage report"
	@echo "  lint          - Run linting checks"
	@echo "  format        - Format code with black and isort"
	@echo "  type-check    - Run mypy type checking"
	@echo "  run           - Run ASAM in development mode"
	@echo "  run-prod      - Run ASAM in production mode"
	@echo "  clean         - Clean up build artifacts"
	@echo "  build         - Build distribution packages"
	@echo ""

# Environment setup
setup:
	@echo "Setting up ASAM development environment..."
	$(PYTHON) scripts/dev_setup.py

# Install dependencies
install:
	$(PIP) install -e .

dev-install:
	$(PIP) install -e ".[dev,macos]"

# Testing
test:
	$(PYTEST) tests/ -v

test-cov:
	$(PYTEST) tests/ --cov=src/asam --cov-report=html --cov-report=term-missing -v

test-watch:
	$(PYTEST) tests/ -f

# Code quality
lint:
	$(VENV_ACTIVATE) && ruff check src/ tests/
	$(VENV_ACTIVATE) && black --check src/ tests/
	$(VENV_ACTIVATE) && isort --check-only src/ tests/

format:
	$(VENV_ACTIVATE) && black src/ tests/
	$(VENV_ACTIVATE) && isort src/ tests/
	$(VENV_ACTIVATE) && ruff check --fix src/ tests/

type-check:
	$(VENV_ACTIVATE) && mypy src/asam

# Running the application
run:
	$(ASAM) --dev-mode

run-prod:
	$(ASAM)

run-config:
	$(ASAM) --config resources/configs/default_config.yaml --dev-mode

# Building and packaging
build:
	$(VENV_ACTIVATE) && python -m build

install-local:
	$(PIP) install -e .

# Cleanup
clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	rm -rf .mypy_cache/
	rm -rf .tox/
	find . -type d -name __pycache__ -delete
	find . -type f -name "*.pyc" -delete

# Database operations
db-init:
	@echo "Initializing ASAM database..."
	$(ASAM) --init-db

db-reset:
	@echo "Resetting ASAM database..."
	rm -f ~/.asam/asam.db
	$(ASAM) --init-db

# Service management (macOS)
install-service:
	$(ASAM) --install-service

uninstall-service:
	$(ASAM) --uninstall-service

# Development utilities
check-deps:
	@echo "Checking system dependencies..."
	@which ollama || echo "⚠️  Ollama not found - install with: curl -fsSL https://ollama.ai/install.sh | sh"
	@which terminal-notifier || echo "⚠️  terminal-notifier not found - install with: brew install terminal-notifier"

install-ollama:
	@echo "Installing Ollama..."
	curl -fsSL https://ollama.ai/install.sh | sh
	@echo "Pulling Llama 3.2 model..."
	ollama pull llama3.2:3b

# CI/CD targets
ci-test: dev-install lint type-check test

ci-build: clean build

# Documentation
docs-serve:
	@echo "Documentation is in docs/ directory"
	@echo "View README.md and docs/*.md files"

# All-in-one development setup
dev-setup: setup dev-install check-deps
	@echo ""
	@echo "🎉 Development environment ready!"
	@echo "Run 'make run' to start ASAM in development mode"
