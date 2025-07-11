# PRESTO Development Makefile
.PHONY: help test test-verbose coverage clean lint format mypy install dev-install pre-commit

help:  ## Show this help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install:  ## Install package
	pip install -e .

dev-install:  ## Install package with development dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	python -m pytest tests/

test-verbose:  ## Run tests with verbose output
	python -m pytest tests/ -v

coverage:  ## Run tests with coverage report
	python -m pytest tests/ --cov=ornl_presto --cov-report=term-missing --cov-report=html

format:  ## Format code with black
	black ornl_presto/ tests/

lint:  ## Run flake8 linting
	flake8 ornl_presto/ tests/ --exclude=ornl_presto/core_backup.py,ornl_presto/core_cleaned.py --max-line-length=88 --extend-ignore=E203,W503,E501,W293,W291,F401,E402

mypy:  ## Run mypy type checking
	mypy ornl_presto/ --exclude='(core_backup|core_cleaned)' --ignore-missing-imports --no-strict-optional

quality:  ## Run all quality checks (format, lint, test)
	$(MAKE) format
	$(MAKE) lint
	$(MAKE) test

pre-commit:  ## Run pre-commit hooks
	pre-commit run --all-files

clean:  ## Clean up build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build package
	python -m build

docs:  ## Generate documentation (if docs exist)
	@echo "Documentation generation not yet implemented"

ci:  ## Run full CI pipeline locally
	$(MAKE) clean
	$(MAKE) quality
	$(MAKE) coverage
