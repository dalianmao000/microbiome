# Contributing to biomekit

Thank you for your interest in contributing to biomekit!

## Development Setup

1. Clone the repository:
```bash
git clone https://github.com/dalianmao000/biomekit.git
cd biomekit
```

2. Install development dependencies:
```bash
pip install -e ".[dev]"
```

3. Install pre-commit hooks:
```bash
pre-commit install
```

## Code Style

- Follow PEP 8
- Use type hints where possible
- Write docstrings for all public functions
- Maximum line length: 100 characters

## Testing

All new features must include tests:

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=biomekit
```

## Commit Messages

Follow conventional commits:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `test:` Test additions
- `refactor:` Code refactoring

## Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## Reporting Issues

Please report bugs and feature requests via GitHub Issues.

## Code of Conduct

Please be respectful and constructive in all interactions.