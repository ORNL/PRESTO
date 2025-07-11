# CI/CD Pipeline Troubleshooting Guide

## Common Issues and Solutions

### 1. PyTorch Installation Timeouts
**Problem**: PyTorch installation takes too long and times out in GitHub Actions.
**Solution**: We now use CPU-only PyTorch which is much faster to install.

### 2. Memory Issues
**Problem**: Tests fail with out-of-memory errors.
**Solution**: The updated CI configuration uses optimized settings for the GitHub Actions environment.

### 3. Pre-commit Hook Failures
**Problem**: MyPy type checking fails due to missing type stubs.
**Solution**: Pre-commit hooks are now excluded from CI and run separately.

### 4. Dependency Conflicts
**Problem**: Package dependencies conflict across Python versions.
**Solution**: We now test on Python 3.9-3.11 (removed 3.8 for better compatibility).

## Monitoring CI/CD Status

You can monitor the pipeline status at:
- **GitHub Actions**: https://github.com/ORNL/PRESTO/actions
- **README Badge**: Shows current build status
- **Coverage Badge**: Shows test coverage percentage

## Manual CI Testing

To test the CI pipeline locally:

```bash
# Run the exact same tests as CI
python -m pytest tests/ --cov=ornl_presto --cov-report=xml --cov-report=term-missing

# Run linting checks
make lint

# Run formatting checks
make format

# Run all quality checks
make quality
```

## CI/CD Pipeline Structure

The pipeline consists of three jobs:

1. **Test Job**: Runs tests across Python 3.9-3.11 with coverage
2. **Lint Job**: Runs Black and Flake8 code quality checks
3. **Build Job**: Builds and validates the package (only on main branch)

## Troubleshooting Steps

If the CI/CD pipeline fails:

1. **Check the GitHub Actions tab** for detailed error logs
2. **Run tests locally** using the commands above
3. **Check for dependency issues** in the installation step
4. **Verify code quality** using `make quality`
5. **Check timeout limits** - some steps have 10-20 minute timeouts

## Performance Optimizations

The current CI configuration includes:

- **Pip caching**: Speeds up dependency installation
- **CPU-only PyTorch**: Faster installation and smaller footprint
- **Separated jobs**: Better parallelization and faster feedback
- **Timeouts**: Prevents hanging builds
- **Matrix testing**: Tests across multiple Python versions

This should resolve most CI/CD pipeline issues you might encounter.
