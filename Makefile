.PHONY: help install install-dev test lint format type-check clean docs

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install:  ## Install package in production mode
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e .

install-dev:  ## Install package in development mode with all extras
	pip install --upgrade pip
	pip install -r requirements.txt
	pip install -e ".[dev,all]"
	pre-commit install

install-fast:  ## Fast install using our install script
	./install_deps.sh

install-apple:  ## Apple Silicon optimized install (M1/M2/M3)
	./install_apple_silicon.sh

install-simple:  ## Simple no-fuss install
	./simple_install.sh

test:  ## Run tests
	pytest tests/ -v --cov=sports_analyzer --cov-report=html --cov-report=term

test-fast:  ## Run tests without slow tests
	pytest tests/ -v -m "not slow" --cov=sports_analyzer --cov-report=term

lint:  ## Run linting (flake8)
	flake8 sports_analyzer tests

format:  ## Format code with black and isort
	black sports_analyzer tests examples
	isort sports_analyzer tests examples

type-check:  ## Run type checking with mypy
	mypy sports_analyzer

quality:  ## Run all quality checks
	make format
	make lint
	make type-check

clean:  ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

docs:  ## Build documentation
	cd docs && make html

docs-serve:  ## Serve documentation locally
	cd docs/_build/html && python -m http.server 8000

benchmark:  ## Run performance benchmarks
	python examples/soccer/benchmark.py

install-models:  ## Download required models and data
	cd examples/soccer && ./setup.sh

demo:  ## Run demo with sample data
	python examples/soccer/main.py --mode PLAYER_TRACKING --source_video_path examples/soccer/data/sample.mp4

# Development shortcuts
dev-setup: install-dev install-models  ## Complete development setup

check: quality test  ## Run all checks before commit