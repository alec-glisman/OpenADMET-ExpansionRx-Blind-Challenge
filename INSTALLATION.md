# Installation Guide

This guide covers setting up the development environment for the OpenADMET Challenge project. The project uses `uv` for fast, reliable Python package management.

## Prerequisites

- Python 3.11 (required, not 3.12+)
- CUDA 13.0+ (for GPU acceleration)
- Linux or macOS (Windows may work but is untested)

## Quick Start

```bash
# 1. Create virtual environment
uv venv

# 2. Activate environment
source .venv/bin/activate

# 3. Install project with all dependencies
uv pip install \
    -e ".[dev,docs]"


# 4. Synchronize environment (if uv.lock is updated)
uv sync --extra "dev" --extra "docs"

# 5. Install pre-commit hooks
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg
```

## Installation Options

### Minimal Installation (Runtime Only)

```bash
uv pip install -e .
```

Installs only the core dependencies needed to run models and predictions.

### Development Installation (Recommended)

```bash
uv pip install -e ".[dev]"
```

Includes:

- Code formatters (black, isort)
- Linters (flake8, pylint, mypy)
- Testing tools (pytest, pytest-cov, pytest-xdist)
- Pre-commit hooks
- Shell formatters (beautysh, shfmt-py)
- Commit message linting (commitizen)

### Full Installation (Dev + Docs)

```bash
uv pip install -e ".[dev,docs]"
```

Adds documentation tools:

- Sphinx documentation generator
- Sphinx themes (furo, sphinx-rtd-theme)
- Sphinx extensions (autodoc, autobuild, myst-parser)

## Verify Installation

```bash
# Check CLI is available
admet --help

# Verify Python version
python --version  # Should be 3.11.x

# Check PyTorch with CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# Run tests
pytest tests/ -q
```

## Environment Sync

If `uv.lock` is updated:

```bash
uv sync --extra "dev" --extra "docs"
```

## Pre-commit Hooks

After installation, set up pre-commit hooks to automatically check code quality:

```bash
# Install git hooks
pre-commit install
pre-commit install --hook-type commit-msg

# Run all hooks on all files (optional)
pre-commit run --all-files
```

See [CONTRIBUTING.md](./CONTRIBUTING.md#pre-commit-hooks) for details on what each hook does.

## Troubleshooting

### PyTorch CUDA Issues

If PyTorch doesn't detect CUDA:

```bash
# Reinstall PyTorch with CUDA support
uv pip install --force-reinstall torch torchvision torchaudio \
    --extra-index-url https://download.pytorch.org/whl/cu130
```

### Import Errors

If you get import errors for `admet` package:

```bash
# Ensure editable install
uv pip install -e .

# Verify package is installed
pip list | grep openadmet-challenge
```

### Pre-commit Hook Failures

If pre-commit hooks are too slow or failing:

```bash
# Skip hooks for a single commit (use sparingly)
git commit --no-verify -m "your message"

# Update hook versions
pre-commit autoupdate
```

## Next Steps

After installation:

1. Read [CONTRIBUTING.md](./CONTRIBUTING.md) for development guidelines
2. Review [README.md](./README.md) for project overview
3. Check [MODEL_CARD.md](./MODEL_CARD.md) for methodology details
4. Explore example configs in `configs/`
5. Try training a small model with `admet model train --help`
