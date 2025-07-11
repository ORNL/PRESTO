# PRESTO Development Makefile
.PHONY: help test test-verbose coverage clean lint format mypy install dev-install pre-commit docs docs-serve docs-check

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
	rm -rf docs/_build/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build:  ## Build package
	python -m build

# Documentation targets
install-docs-deps:  ## Install documentation dependencies
	pip install -r docs/requirements.txt

docs:  ## Build documentation
	@echo "Building documentation..."
	@sphinx-build -b html docs docs/_build/html
	@echo "Documentation built. Open docs/_build/html/index.html to view."

docs-serve: docs  ## Serve documentation locally
	@echo "Serving documentation at http://localhost:8000"
	@cd docs/_build/html && python -m http.server 8000

docs-check:  ## Check documentation for errors
	@echo "Checking documentation for errors..."
	@sphinx-build -b linkcheck docs docs/_build/linkcheck

docs-clean:  ## Clean documentation build
	rm -rf docs/_build/
