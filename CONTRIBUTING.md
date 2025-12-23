# CONTRIBUTING

## Purpose

This file documents how to contribute to this machine-learning project. It covers repository workflow, environment and dependency setup, data handling, experiments, testing, and norms for pull requests and issues. Follow these guidelines to keep contributions reproducible, reviewable, and safe.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Pre-commit Hooks](#pre-commit-hooks)
3. [Branching and Commits](#branching-and-commits)
4. [Testing and CI](#testing-and-ci)
5. [Code Quality](#code-quality)
6. [Documentation](#documentation)
7. [Data and Artifacts](#data-and-artifacts)
8. [Experiments and Reproducibility](#experiments-and-reproducibility)
9. [Contributor Checklist](#contributor-checklist)

## Getting Started

### Initial Setup

1. Follow [INSTALLATION.md](./INSTALLATION.md) to set up your environment
2. Install pre-commit hooks: `pre-commit install`
3. Familiarize yourself with the [README.md](./README.md) and project structure
4. Run tests to verify setup: `pytest tests -q`

### Project Structure

```
.
├── src/admet/          # Main package
│   ├── cli/            # CLI commands (admet data/model/leaderboard)
│   ├── model/          # Model implementations (Chemprop, ensemble, HPO)
│   ├── data/           # Data loading, splitting, preprocessing
│   ├── evaluation/     # Metrics, visualization, leaderboard
│   └── train/          # Training loops, callbacks, curriculum learning
├── configs/            # YAML configs for experiments, HPO, ensemble
├── scripts/            # Bash scripts for training, data prep, analysis
├── tests/              # Pytest test suite
├── docs/               # Sphinx documentation
└── notebooks/          # Jupyter/Marimo notebooks for EDA
```

### Key Components

- **CLI**: Typer-based `admet` command with subcommands for data, model, leaderboard
- **ML Framework**: PyTorch with Chemprop for molecular property prediction
- **Experiment Tracking**: MLflow for logging experiments, metrics, and artifacts
- **Parallelization**: Ray for distributed HPO and ensemble training
- **Testing**: Pytest with markers for unit/integration/slow tests
- **Package Management**: uv for dependency management

## Pre-commit Hooks

This project uses pre-commit hooks to maintain code quality and consistency. Hooks run automatically on `git commit` and block commits if checks fail.

### Installing Hooks

```bash
# Install pre-commit framework (included in dev dependencies)
uv pip install pre-commit

# Install git hooks
uv run pre-commit install                    # For pre-commit stage
uv run pre-commit install --hook-type commit-msg  # For commit message linting

# Verify installation
uv run pre-commit --version

# Run hooks on all files initially
uv run pre-commit run --all-files
```

### Configured Hooks

See [.pre-commit-config.yaml](./.pre-commit-config.yaml) for complete configuration. Key hooks include:

**Standard Checks**: Trailing whitespace, merge conflicts, private keys, large files (>1MB), syntax validation

**Formatting**: black (Python), isort (imports), prettier (YAML/TOML), beautysh/shfmt (shell scripts)

**Linting**: flake8, pylint (≥9.0 score), mypy (static type checking)

**Testing**: pytest (fast unit tests only, parallel execution)

**Notebooks**: nbstripout (removes outputs before commit)

**Commit Messages**: commitizen (enforces conventional commit format)

### Performance Tips

**Typical timing**: Formatters/linters (<5s), mypy/pylint (5-30s, cached), pytest (30s-2min)

**Speed up commits**: Run `pre-commit run` (changed files only) or skip temporarily with `SKIP=pytest,mypy git commit -m "message"`

**Excluded paths**: `src/bitbirch/`, `docs/`, `notebooks/`, `archive/`

## Branching and Commits

### Branch Naming

Use descriptive prefixes:

- `feature/<description>` - New functionality (e.g., `feature/task-affinity`)
- `fix/<description>` - Bug fixes (e.g., `fix/validation-mae-calculation`)
- `experiment/<description>` - Experimental changes (e.g., `experiment/moe-ffn`)
- `docs/<description>` - Documentation updates
- `refactor/<description>` - Code restructuring without functional changes

### Commit Message Format

This project uses [Conventional Commits](https://www.conventionalcommits.org/) enforced by commitizen:

```text
<type>: <short summary>

<optional body>

<optional footer>
```

**Types:**

- `feat:` - New feature (e.g., `feat: add MoE FFN decoder`)
- `fix:` - Bug fix (e.g., `fix: correct ASHA early stopping logic`)
- `docs:` - Documentation only (e.g., `docs: update HPO guide`)
- `style:` - Formatting changes (e.g., `style: apply black formatting`)
- `refactor:` - Code restructuring (e.g., `refactor: extract data splitting logic`)
- `test:` - Adding/updating tests (e.g., `test: add curriculum sampler tests`)
- `chore:` - Maintenance tasks (e.g., `chore: update dependencies`)
- `perf:` - Performance improvements (e.g., `perf: optimize Ray parallelization`)

**Multi-line commits:**

```bash
git commit -m "feat: add ensemble prediction aggregation

Implements mean and std aggregation across 25 models.
Includes epistemic uncertainty estimation.

Closes #42"
```

### PR Checklist

- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] All tests pass (`pytest tests/ -q`)
- [ ] New code has appropriate tests (aim for >80% coverage)
- [ ] CLI changes include updated help text and examples
- [ ] Config changes include example YAML files in `configs/`
- [ ] Changes to datasets/models are documented (see Data & Artifacts section)
- [ ] Experiments include:
  - [ ] MLflow run ID and experiment name
  - [ ] Dataset version/commit
  - [ ] Hyperparameters (or config file path)
  - [ ] Key metrics (MAE, RMSE, etc.)
  - [ ] Compute resources used (GPUs, runtime)
- [ ] Documentation updated if public API changed:
  - [ ] Docstrings updated (NumPy style)
  - [ ] README updated if CLI or major features changed
  - [ ] docs/ updated if architectural changes

## Testing and CI

### Issues and Bug Reports

- Use issues for bugs, feature requests, or proposing experiments
- Provide minimal reproduction (error messages, stack traces, commands)
- Label clearly: `bug`, `enhancement`, `experiment`, `data`, `question`

### Test Markers

Tests use pytest markers (see `pyproject.toml`): `slow`, `integration`, `unit`, `no_mlflow_runs`

**Run specific tests**: `pytest -m slow` or `pytest -m "not slow"`

## Code Quality

### Formatting Standards

**Automatically enforced by pre-commit hooks:**

- **Line length**: 120 characters, Python 3.11, Black-compatible style
- **Tools**: black (Python), isort (imports), beautysh/shfmt (shell), prettier (YAML/TOML)

**Manual formatting**: `black src/ tests/ && isort src/ tests/` or `pre-commit run --all-files`

### Linting

**flake8**: Style enforcement (max line 120, ignores E203/W503)

**pylint**: Deep analysis (must score ≥9.0, see `pyproject.toml`)

**mypy**: Static type checking (Python 3.11, non-strict mode)

**Run linters**: `flake8 src/ tests/ && pylint src/admet/ && mypy src/admet/`

### Type Annotations

**Add types to**: Public API functions/classes, complex functions, non-obvious return types
**Skip types for**: Test functions

## Documentation

### Docstring Standards

**Use NumPy-style docstrings** for all public modules, functions, and classes:

```python
def train_model(
    config: ChempropConfig,
    train_data: pd.DataFrame,
    val_data: Optional[pd.DataFrame] = None,
    output_dir: Optional[Path] = None
) -> ChempropModel:
    """Train a Chemprop model with given configuration.

    Trains a message passing neural network (MPNN) for molecular property
    prediction using the provided training data and configuration.

    Parameters
    ----------
    config : ChempropConfig
        Model configuration including architecture, training parameters,
        and hyperparameters.
    train_data : pd.DataFrame
        Training dataset with columns [SMILES, <target_columns>].
        Shape: (n_samples, n_columns).
    val_data : Optional[pd.DataFrame], default=None
        Validation dataset with same schema as train_data.
        If None, uses 10% of train_data for validation.
    output_dir : Optional[Path], default=None
        Directory to save model checkpoints and logs.
        If None, uses temporary directory.

    Returns
    -------
    ChempropModel
        Trained model ready for prediction.

    Raises
    ------
    ValueError
        If train_data is empty or missing required columns.
    FileNotFoundError
        If output_dir doesn't exist and cannot be created.

    Notes
    -----
    - Uses MLflow for experiment tracking if configured
    - Automatically handles missing values (NaN) in targets
    - Saves best model based on validation MAE

    Examples
    --------
    >>> config = ChempropConfig(epochs=50, hidden_size=300)
    >>> model = train_model(config, train_df, val_df)
    >>> predictions = model.predict(test_df)

    See Also
    --------
    ModelEnsemble : Train ensemble of models
    ChempropHPO : Hyperparameter optimization
    """
    ...
```

### Building Documentation

**Sphinx docs** in `docs/`: Run `make -C docs html` or `sphinx-autobuild docs docs/_build/html` for live-reload

**CLI help**: Keep concise with usage examples

## Data and Artifacts

Handling datasets and model artifacts is the most sensitive part of an ML repository. Follow these rules:

- Do not commit raw or sensitive data to the repository. Use `assets/` for small example files, metadata, and pointers, but keep full datasets out of source control.
- If you add a dataset, include a README in `assets/dataset/` describing source, license, preprocessing steps, and versioning information.
- Store large artifacts (models, datasets) outside the git repo (object storage, Hugging Face, or a mounted drive). Include a small script or instructions to download them reproducibly.
- When modifying or augmenting data, record the transformation steps and output files under `assets/preprocessing/` or `assets/augmentation/` as appropriate.

Minimal dataset README example to include with assets:

```yaml
source: Hugging Face dataset XYZ
version: v1.2
preprocessing: canonicalized SMILES, filtered by MW < 800
note: training/validation/test split uses stratified splits by endpoint
```

## Experiments and Reproducibility

- Record experiment details: dataset version (or commit/hash that produced the processed dataset), random seed, hyperparameters, and environment (Python version, key package versions).
- Prefer experiment tracking (MLflow, Weights & Biases, or a simple CSV) so results can be compared and reproduced.
- Save model checkpoints and logs to `assets/models/` for small models or provide links for larger ones.
- For reproducible evaluation, include an `evaluate.py` or small script that accepts a model checkpoint and dataset path and outputs numeric metrics.

Example experiment record (in PR or notebook):

```yaml
seed: 42
dataset: expansion_data_train_v1.csv
model: chemprop-multitask, commit: abc123
hyperparameters: {...}
metrics: {"rmse": 0.32}
```

## Contributor Checklist

- [ ] Read README and CONTRIBUTING guide
- [ ] Changes are scoped and documented
- [ ] Pre-commit hooks pass (`pre-commit run --all-files`)
- [ ] Tests pass (`pytest tests/ -q`)
- [ ] Documentation updated (docstrings, CLI help, README, docs/)
- [ ] Experiments include MLflow run ID, dataset version, metrics, and config

---

**Thank you for contributing!** Questions? Open an issue or contact maintainers in [README.md](./README.md).
