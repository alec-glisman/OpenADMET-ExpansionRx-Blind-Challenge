# Code Quality Checks

Run linting and type checking tools.

## All Checks

```bash
# Run all pre-commit hooks
pre-commit run --all-files
```

## Individual Tools

```bash
# Style check (flake8)
flake8 src/ tests/

# Type checking (mypy)
mypy src/admet/

# Deep analysis (pylint) - must score >= 9.0
pylint src/admet/ --fail-under=9.0

# Security check (optional)
bandit -r src/admet/
```

## Quick Check (CI-like)

```bash
flake8 src/ tests/ && mypy src/admet/ && pylint src/admet/ --fail-under=9.0
```

## Configuration

- Line length: 120 characters
- Config files: `pyproject.toml`, `.flake8`
- Pylint threshold: 9.0

## Skip Pre-commit Hooks

```bash
SKIP=pytest,mypy git commit -m "message"
```
