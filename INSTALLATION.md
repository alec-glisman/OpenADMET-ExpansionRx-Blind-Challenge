# Installation (env/dev)

These instructions create a virtual environment at `env/dev` and install the runtime dependencies plus developer utilities (formatters and linters). The commands assume macOS with `zsh`.

## Create and activate a virtual environment at `env/dev`

```bash
# create venv
uv venv

# activate (zsh)
source .venv/bin/activate

# upgrade packaging tools inside the venv via uv
uv pip install -e '.[dev]' --extra-index-url https://download.pytorch.org/whl/cu130
```

## Notes and caveats

If you want editable installs from this repository using PEP-517/pyproject, add a `[build-system]` section to `pyproject.toml` (setuptools/poetry) and then you can run `pip install -e '.[dev]'` to install project extras.
