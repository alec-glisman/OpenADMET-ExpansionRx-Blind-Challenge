# CONTRIBUTING

## Purpose

This file documents how to contribute to this machine-learning project. It covers repository workflow, environment and dependency setup, data handling, experiments, testing, and norms for pull requests and issues. Follow these guidelines to keep contributions reproducible, reviewable, and safe.

## Getting started

## Branching and commit messages

- Use a short, descriptive branch name: `feature/<short-desc>`, `fix/<short-desc>`, or `experiment/<short-desc>`.
- Keep commits small and focused. Squash or rebase locally before opening a pull request if it helps keep history clean.
- Write concise commit messages with a one-line summary and optional body. Follow conventional commits if possible (e.g., `feat:`, `fix:`, `docs:`).

## Pull request process

1. Open an issue first for non-trivial work or experiments that may affect other contributors.
1. Create a PR from your branch and target `main` (or the branch named in the issue).
1. Include the following in the PR description:
   - Summary of the change.
   - How to run or validate the change (commands, notebooks, small dataset subset).
   - Any new dependencies or environment changes.
   - If the change affects model results, include a short reproducibility checklist (dataset version, random seed, hyperparameters).
1. Add reviewers and wait for at least one approval before merging. Address review comments with new commits.

### PR checklist

- [ ] Code is linted and formatted.
- [ ] Unit tests and/or small integration tests were added or updated.
- [ ] Changes to datasets or models are documented and saved under `assets/` (see Data & Artifacts).
- [ ] Any long-running experiments include a short note on compute used and runtime.

## Issues

- Use issues for bugs, feature requests, or proposing large experiments.
- Provide a minimal reproduction when reporting a bug (error messages, stack traces, commands).
- Label issues clearly: `bug`, `enhancement`, `experiment`, `data`, `question`.

## Testing and CI

- Add fast unit tests for deterministic code paths. Heavy model training should not run in CI unless it is a short smoke test.
- Use `pytest` for tests. Keep CI jobs fast by running only unit tests and linters; reserve longer integration tests for scheduled pipelines.

Testing the CLI

- The Typer app is available as ``admet.cli.app`` and subcommands are registered at import time. When writing CLI tests, prefer invoking the top-level app with Typer's ``CliRunner`` to ensure parsing matches the installed console script:

```python
from typer.testing import CliRunner
from admet.cli import app as main_app

runner = CliRunner()
result = runner.invoke(main_app, ["data", "split", "--output", "./out", "data.csv"])
assert result.exit_code == 0
```

- Avoid invoking sub-``Typer`` instances (for example ``data_app``) directly in tests because argument parsing behavior may differ.

Note: Tests that require MLflow (e.g., start MLflow runs) are marked with `no_mlflow_runs` and are excluded by default from CI and standard `pytest` runs. To run them explicitly, use:

```bash
pytest -m no_mlflow_runs -q
```

Example local test run:

```bash
pytest tests/ -q
```

## Style and linting

- Prefer tools like `black`, `isort`, and `flake8` (or project-specific equivalents). Run formatters before committing.
- Python typing is encouraged for public APIs and large modules.

Format example:

```bash
black .
isort .
flake8
```

### Documentation Standard

All Python modules now follow a unified style:

- Module headers: RST sections listing purpose, key components, examples.
- Functions / classes: NumPy‑style docstrings with Parameters / Returns / Raises sections (and Notes / Examples where helpful).
- Shape and schema conventions explicitly documented for data loading, model interfaces, splitting logic, and visualization utilities.

## Data and artifacts

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

## Experiments and reproducibility

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

## Notebooks

- Keep notebooks focused and small. Include a short text summary at the top describing goal and inputs.
- Clear outputs and remove very large inline outputs before committing. Prefer linking to rendered notebook artifacts (nbviewer, GitHub-rendered) or converting to MD when useful.

## Security, licensing, and data privacy

- Check upstream dataset licenses before using or redistributing data. Add license notes to `assets/dataset/README.md`.
- Never commit secrets, API keys, or credentials. Use environment variables or a secrets manager for CI.

## When you find technical debt

- Create an issue documenting the debt, its impact, and a suggested remediation. Tag it with `technical-debt` and a priority.
- If the debt is small and well-scoped, include a follow-up PR that fixes it.

## Communication and reviews

- Be respectful during code review. Provide concrete, actionable feedback.
- When requesting a review, add a clear summary and testing steps so reviewers can validate quickly.

## Contact

If you have questions about contributing, open an issue with the `question` label or contact the maintainers listed in `README.md`.

## Small checklist for contributors

- [ ] I have read the repo README and this CONTRIBUTING guide.
- [ ] My changes are limited in scope and documented.
- [ ] I ran linters and tests locally.
- [ ] I added or updated documentation where applicable.

Thank you for contributing!
