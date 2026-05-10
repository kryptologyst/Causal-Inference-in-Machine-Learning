"""Makefile for common development tasks."""

.PHONY: help install install-dev test lint format type-check clean run-demo run-notebook

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"
	pre-commit install

test: ## Run tests
	pytest tests/ -v --cov=src --cov-report=term-missing

test-fast: ## Run tests quickly (no coverage)
	pytest tests/ -v

lint: ## Run linting
	ruff check src/ tests/
	black --check src/ tests/

format: ## Format code
	black src/ tests/
	ruff check --fix src/ tests/

type-check: ## Run type checking
	mypy src/

clean: ## Clean up generated files
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	rm -rf build/
	rm -rf dist/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/

run-demo: ## Run Streamlit demo
	streamlit run demo/streamlit_app.py

run-notebook: ## Start Jupyter notebook server
	jupyter notebook notebooks/

run-quick-start: ## Run quick start script
	python quick_start.py --data-type synthetic --n-samples 1000

run-experiment: ## Run full experiment
	python scripts/train.py --data-type synthetic --n-samples 1000

check-all: lint type-check test ## Run all checks

build: ## Build the package
	python -m build

publish: ## Publish to PyPI (requires authentication)
	python -m twine upload dist/*

docs: ## Generate documentation
	sphinx-build -b html docs/ docs/_build/html

setup: install-dev ## Initial setup for development
	@echo "✅ Development environment setup complete!"
	@echo "Run 'make help' to see available commands"
