# Code Formatting

Auto-format code with black and isort.

## Format All

```bash
black src/ tests/ && isort src/ tests/
```

## Check Only (No Changes)

```bash
black --check src/ tests/
isort --check-only src/ tests/
```

## Format Single File

```bash
black path/to/file.py
isort path/to/file.py
```

## Configuration

- Line length: 120 characters
- Black profile for isort
- Config in `pyproject.toml`

## Pre-commit Integration

```bash
# Install hooks (one-time)
pre-commit install

# Run formatters via pre-commit
pre-commit run black --all-files
pre-commit run isort --all-files
```
