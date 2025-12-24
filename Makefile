.PHONY: help install dev-install test test-cov check lint format typecheck security clean clean-build clean-all build publish publish-manual publish-test version

# Variables
PYTHON := python3
PIP := $(PYTHON) -m pip
PYTEST := $(PYTHON) -m pytest
RUFF := ruff
MYPY := mypy

# Directories
SRC_DIR := src
TEST_DIR := tests
EXAMPLES_DIR := examples

# Default target
help:
	@echo "Lazarus (chuk-mlx) - Available Commands"
	@echo "======================================="
	@echo ""
	@echo "Setup & Installation:"
	@echo "  make install          - Install production dependencies"
	@echo "  make dev-install      - Install development dependencies"
	@echo ""
	@echo "Running:"
	@echo "  make run-infer        - Run inference example"
	@echo "  make run-tokenizer    - Run tokenizer encode example"
	@echo ""
	@echo "Testing:"
	@echo "  make test             - Run tests"
	@echo "  make test-cov         - Run tests with coverage report"
	@echo "  make test-watch       - Run tests in watch mode"
	@echo "  make serve-coverage   - Serve HTML coverage report on localhost:8000"
	@echo ""
	@echo "Code Quality:"
	@echo "  make check            - Run all checks (lint + type check + test)"
	@echo "  make lint             - Run linter (ruff)"
	@echo "  make format           - Format code with ruff"
	@echo "  make format-check     - Check code formatting"
	@echo "  make typecheck        - Run type checker (mypy)"
	@echo "  make security         - Run security checks (bandit)"
	@echo ""
	@echo "Build & Publishing:"
	@echo "  make build            - Build distribution packages"
	@echo "  make publish          - Build and publish to PyPI (via GitHub trusted publishing)"
	@echo "  make publish-manual   - Build and publish to PyPI (manual with twine)"
	@echo "  make publish-test     - Build and publish to TestPyPI"
	@echo ""
	@echo "Examples:"
	@echo "  make example-tokenize         - Tokenize sample text"
	@echo "  make example-vocab            - Show vocabulary info"
	@echo "  make example-train-sft        - Run SFT training example"
	@echo ""
	@echo "Cleanup:"
	@echo "  make clean            - Remove generated files"
	@echo "  make clean-all        - Remove all generated files and caches"

# Setup & Installation
install:
	@echo "Installing production dependencies..."
	@if command -v uv >/dev/null 2>&1; then \
		uv sync; \
	else \
		$(PIP) install -e .; \
	fi

dev-install:
	@echo "Installing development dependencies..."
	@if command -v uv >/dev/null 2>&1; then \
		uv sync --all-extras; \
	else \
		$(PIP) install -e ".[dev]"; \
	fi

# Running
run-infer:
	@echo "Running inference..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run lazarus infer --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Hello, world!"; \
	else \
		lazarus infer --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --prompt "Hello, world!"; \
	fi

run-tokenizer:
	@echo "Running tokenizer encode..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run lazarus tokenizer encode -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --text "Hello world"; \
	else \
		lazarus tokenizer encode -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --text "Hello world"; \
	fi

# Testing
test:
	@echo "Running tests..."
	@if command -v uv >/dev/null 2>&1; then \
		PYTHONPATH=$(SRC_DIR) uv run pytest $(TEST_DIR) -v; \
	else \
		PYTHONPATH=$(SRC_DIR) $(PYTEST) $(TEST_DIR) -v; \
	fi

test-cov coverage:
	@echo "Running tests with coverage..."
	@if command -v uv >/dev/null 2>&1; then \
		PYTHONPATH=$(SRC_DIR) uv run pytest $(TEST_DIR) --cov=$(SRC_DIR)/chuk_lazarus --cov-report=term-missing --cov-report=html; \
	else \
		PYTHONPATH=$(SRC_DIR) $(PYTEST) $(TEST_DIR) --cov=$(SRC_DIR)/chuk_lazarus --cov-report=term-missing --cov-report=html; \
	fi
	@echo ""
	@echo "Coverage report generated in htmlcov/index.html"

test-watch:
	@echo "Running tests in watch mode..."
	@if command -v uv >/dev/null 2>&1; then \
		PYTHONPATH=$(SRC_DIR) uv run pytest-watch $(TEST_DIR); \
	else \
		PYTHONPATH=$(SRC_DIR) $(PYTHON) -m pytest_watch $(TEST_DIR); \
	fi

# Code Quality
check: lint typecheck test
	@echo ""
	@echo "All checks passed!"

lint:
	@echo "Running linter..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run ruff check $(SRC_DIR) $(TEST_DIR); \
		uv run ruff format --check $(SRC_DIR) $(TEST_DIR); \
	elif command -v $(RUFF) >/dev/null 2>&1; then \
		$(RUFF) check $(SRC_DIR) $(TEST_DIR); \
		$(RUFF) format --check $(SRC_DIR) $(TEST_DIR); \
	else \
		echo "Ruff not found. Install with: pip install ruff"; \
		exit 1; \
	fi

format:
	@echo "Formatting code..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run ruff format $(SRC_DIR) $(TEST_DIR); \
		uv run ruff check --fix $(SRC_DIR) $(TEST_DIR); \
	elif command -v $(RUFF) >/dev/null 2>&1; then \
		$(RUFF) format $(SRC_DIR) $(TEST_DIR); \
		$(RUFF) check --fix $(SRC_DIR) $(TEST_DIR); \
	else \
		echo "Ruff not found. Install with: pip install ruff"; \
		exit 1; \
	fi

format-check:
	@echo "Checking code formatting..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run ruff format --check $(SRC_DIR) $(TEST_DIR); \
	elif command -v $(RUFF) >/dev/null 2>&1; then \
		$(RUFF) format --check $(SRC_DIR) $(TEST_DIR); \
	else \
		echo "Ruff not found. Install with: pip install ruff"; \
		exit 1; \
	fi

typecheck:
	@echo "Running type checker..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run mypy $(SRC_DIR) --no-site-packages || echo "Type check found issues (non-blocking)"; \
	elif command -v $(MYPY) >/dev/null 2>&1; then \
		$(MYPY) $(SRC_DIR) --no-site-packages || echo "Type check found issues (non-blocking)"; \
	else \
		echo "MyPy not found. Install with: pip install mypy"; \
		exit 1; \
	fi

# Alias for typecheck
type-check: typecheck

# Security
security:
	@echo "Running security checks..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run bandit -r $(SRC_DIR) -f txt || echo "Security issues found (non-blocking)"; \
	elif command -v bandit >/dev/null 2>&1; then \
		bandit -r $(SRC_DIR) -f txt || echo "Security issues found (non-blocking)"; \
	else \
		echo "Bandit not found. Install with: pip install bandit"; \
		exit 1; \
	fi

# Build & Publishing
build: clean-build
	@echo "Building distribution packages..."
	@if command -v uv >/dev/null 2>&1; then \
		uv build; \
	else \
		python3 -m build; \
	fi
	@echo ""
	@echo "Build complete. Distributions are in the 'dist' folder."
	@ls -lh dist/

publish:
	@echo "Starting automated release process..."
	@echo ""
	@# Get current version
	@version=$$(grep '^version = ' pyproject.toml | cut -d'"' -f2); \
	tag="v$$version"; \
	echo "Version: $$version"; \
	echo "Tag: $$tag"; \
	echo ""; \
	\
	echo "Pre-flight checks:"; \
	echo "=================="; \
	\
	if git diff --quiet && git diff --cached --quiet; then \
		echo "Working directory is clean"; \
	else \
		echo "Working directory has uncommitted changes"; \
		echo ""; \
		git status --short; \
		echo ""; \
		echo "Please commit or stash your changes before releasing."; \
		exit 1; \
	fi; \
	\
	if git tag -l | grep -q "^$$tag$$"; then \
		echo "Tag $$tag already exists"; \
		echo ""; \
		echo "To delete and recreate:"; \
		echo "  git tag -d $$tag"; \
		echo "  git push origin :refs/tags/$$tag"; \
		exit 1; \
	else \
		echo "Tag $$tag does not exist yet"; \
	fi; \
	\
	current_branch=$$(git rev-parse --abbrev-ref HEAD); \
	echo "Current branch: $$current_branch"; \
	echo ""; \
	\
	echo "This will:"; \
	echo "  1. Create and push tag $$tag"; \
	echo "  2. Trigger GitHub Actions to:"; \
	echo "     - Create a GitHub release with changelog"; \
	echo "     - Run tests on all platforms"; \
	echo "     - Build and publish to PyPI"; \
	echo ""; \
	read -p "Continue? (y/N) " -n 1 -r; \
	echo ""; \
	if [[ ! $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "Aborted."; \
		exit 1; \
	fi; \
	\
	echo ""; \
	echo "Creating and pushing tag..."; \
	git tag -a "$$tag" -m "Release $$tag"; \
	git push origin "$$tag"; \
	echo ""; \
	echo "Tag $$tag created and pushed"; \
	echo ""; \
	echo "GitHub Actions will now:"; \
	echo "  - Run tests"; \
	echo "  - Create GitHub release"; \
	echo "  - Publish to PyPI"; \
	echo ""; \
	echo "Monitor progress at:"; \
	echo "  https://github.com/chrishayuk/chuk-mlx/actions"

publish-manual: build
	@echo "Manual PyPI Publishing"
	@echo "======================"
	@echo ""
	@if [ -n "$$PYPI_TOKEN" ]; then \
		if command -v uv >/dev/null 2>&1; then \
			uv run twine upload --username __token__ --password "$$PYPI_TOKEN" dist/*; \
		else \
			python3 -m twine upload --username __token__ --password "$$PYPI_TOKEN" dist/*; \
		fi; \
	else \
		if command -v uv >/dev/null 2>&1; then \
			uv run twine upload dist/*; \
		else \
			python3 -m twine upload dist/*; \
		fi; \
	fi

publish-test: build
	@echo "Publishing to TestPyPI..."
	@echo ""
	@if [ -n "$$PYPI_TOKEN" ]; then \
		if command -v uv >/dev/null 2>&1; then \
			uv run twine upload --repository testpypi --username __token__ --password "$$PYPI_TOKEN" dist/*; \
		else \
			python3 -m twine upload --repository testpypi --username __token__ --password "$$PYPI_TOKEN" dist/*; \
		fi; \
	else \
		if command -v uv >/dev/null 2>&1; then \
			uv run twine upload --repository testpypi dist/*; \
		else \
			python3 -m twine upload --repository testpypi dist/*; \
		fi; \
	fi

# Cleanup
clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage htmlcov 2>/dev/null || true

clean-build:
	@echo "Cleaning build artifacts..."
	@rm -rf build/ dist/ *.egg-info $(SRC_DIR)/*.egg-info 2>/dev/null || true
	@rm -rf .eggs/ 2>/dev/null || true
	@find . -name '*.egg' -exec rm -f {} + 2>/dev/null || true

clean-all: clean clean-build
	@echo "Cleaning all files..."
	rm -rf venv env .eggs 2>/dev/null || true
	find . -name '.DS_Store' -delete 2>/dev/null || true
	find . -name '*.log' -delete 2>/dev/null || true

# Examples
example-tokenize:
	@echo "Running tokenizer example..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run lazarus tokenizer encode -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --text "The quick brown fox jumps over the lazy dog."; \
	else \
		lazarus tokenizer encode -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --text "The quick brown fox jumps over the lazy dog."; \
	fi

example-vocab:
	@echo "Running vocabulary info example..."
	@if command -v uv >/dev/null 2>&1; then \
		uv run lazarus tokenizer vocab -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --search "hello"; \
	else \
		lazarus tokenizer vocab -t TinyLlama/TinyLlama-1.1B-Chat-v1.0 --search "hello"; \
	fi

example-train-sft:
	@echo "SFT training example (requires training data)..."
	@echo "lazarus train sft --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --data train.jsonl --use-lora"

# Development helpers
serve-coverage:
	@echo "Serving coverage report on http://localhost:8000..."
	cd htmlcov && $(PYTHON) -m http.server 8000

# Quick development workflow
dev: dev-install
	@echo "Development environment ready!"
	@echo "Run 'make run-tokenizer' to test the CLI"

quick-check: format lint
	@echo "Quick check complete!"

# CI/CD helpers
ci: dev-install check
	@echo "CI pipeline complete!"

# Version info
version:
	@echo "Python version:"
	@$(PYTHON) --version
	@echo ""
	@echo "Package version:"
	@grep '^version = ' pyproject.toml
	@echo ""
	@echo "Installed packages:"
	@$(PIP) list | grep -E "(chuk-lazarus|mlx|pytest|ruff|mypy)" || echo "  (none found)"
