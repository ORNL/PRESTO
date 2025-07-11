# Contributing to presto

Thank you for considering contributing to presto! This document provides guidelines and instructions for contributing to this project.

## Code of Conduct

Please note that this project adheres to the [Contributor Covenant Code of Conduct](CODE_OF_CONDUCT.md). By participating, you are expected to uphold this code. Please report unacceptable behavior to the project maintainers.

## How to Contribute

### Reporting Bugs

If you find a bug in the software, please create an issue on GitHub with the following information:

- A clear, descriptive title
- A detailed description of the issue
- Steps to reproduce the bug
- Expected behavior
- Actual behavior
- Screenshots (if applicable)
- Your environment (OS, Python version, package versions)

### Suggesting Enhancements

We welcome suggestions for enhancements! Please create an issue on GitHub with:

- A clear, descriptive title
- A detailed description of the proposed enhancement
- Any relevant examples or use cases
- If applicable, references to similar features in other projects

### Pull Requests

1. Fork the repository
2. Create a new branch for your feature or bugfix (`git checkout -b feature/your-feature-name`)
3. Make your changes
4. Add or update tests as necessary
5. Update documentation as needed
6. Ensure all tests pass
7. Commit your changes with clear, descriptive commit messages
8. Push to your branch
9. Submit a pull request to the main repository

#### Pull Request Guidelines

- Follow the existing code style
- Include tests for new features
- Update documentation for any changed functionality
- Keep pull requests focused on a single topic
- Reference any relevant issues in your PR description

## Development Setup

1. Clone the repository
2. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```
3. Run tests to ensure everything is working:
   ```bash
   pytest
   ```

## Testing

We use pytest for testing. To run the tests:

```bash
pytest
```

## Documentation

Please update documentation when making changes:

- Update docstrings for any modified functions or classes
- Update README.md if necessary
- Update example code if your changes affect the API

## License

By contributing to this project, you agree that your contributions will be licensed under the project's [MIT License](LICENSE).

## Questions?

If you have any questions about contributing, please create an issue on GitHub.
