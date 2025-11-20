# Installation

These instructions create a virtual environment and install the runtime dependencies plus developer utilities (formatters, linters, and documentation tools). The commands assume a Unix-like shell (Linux/macOS).

## Create and activate a virtual environment

```bash
# create venv
uv venv

# activate
source .venv/bin/activate

# install project in editable mode with dev and docs extras
uv pip install -e ".[dev,docs]" --extra-index-url https://download.pytorch.org/whl/cu130
```

## Notes and caveats

If you want editable installs from this repository using PEP-517/pyproject, add a `[build-system]` section to `pyproject.toml` (setuptools/poetry) and then you can run `uv pip install -e ".[dev,docs]"` to install project extras.
