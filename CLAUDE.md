# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

OpenADMET Challenge: ML pipeline for predicting ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties of small molecules. Uses Chemprop MPNN as the primary model with ensemble training across 5 splits × 5 folds.

**Tech Stack:** Python 3.11, PyTorch, Chemprop (v2), Ray Tune (HPO/parallelization), MLflow (experiment tracking), OmegaConf (configs), Typer/Rich (CLI).

## Common Commands

### Environment Setup

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,docs]"
uv run pre-commit install && uv run pre-commit install --hook-type commit-msg
```

### Testing

```bash
pytest tests/ -q                          # Run all tests
pytest tests/test_chemprop_model.py -v    # Single test file
pytest -m "not slow" -q                   # Skip slow tests
pytest -m "not no_mlflow_runs" -q         # Skip MLflow-dependent tests (default)
pytest -n auto -q                         # Parallel execution
```

### Linting & Formatting

```bash
black src/ tests/ && isort src/ tests/    # Format code
flake8 src/ tests/                        # Style check
mypy src/admet/                           # Type check
pylint src/admet/ --fail-under=9.0        # Deep analysis
pre-commit run --all-files                # Run all hooks
```

### CLI Usage

```bash
admet --help                              # Show all commands
admet model train -c configs/0-experiment/chemprop.yaml
admet model ensemble -c configs/3-production/ensemble_chemprop_hpo_001.yaml --max-parallel 4
admet model hpo -c configs/1-hpo-single/hpo_chemprop.yaml --num-samples 50
admet data split data.csv --cluster-method bitbirch
admet leaderboard scrape --user <username>
```

### Documentation

```bash
make -C docs html                         # Build docs
sphinx-autobuild docs docs/_build/html    # Live preview
```

## Architecture

### Source Layout

```
src/admet/
├── cli/                 # Typer CLI (data, model, leaderboard subcommands)
├── model/
│   ├── config.py        # UnifiedModelConfig schema (OmegaConf dataclasses)
│   ├── chemprop/        # Chemprop MPNN: model.py, hpo.py, ensemble.py
│   ├── chemeleon/       # Pretrained encoder: model.py, hpo.py
│   ├── classical/       # XGBoost, LightGBM, CatBoost wrappers
│   ├── ensemble.py      # Generic ensemble orchestration
│   └── ffn_factory.py   # FFN architectures (MLP, MoE, Branched)
├── data/                # Data loading, splitting, preprocessing
├── features/            # Molecular fingerprints (Morgan, RDKit, MACCS, Mordred)
└── leaderboard/         # Challenge leaderboard scraping and reports
```

### Key Architectural Patterns

**Unified Config System:** All models use `UnifiedModelConfig` (src/admet/model/config.py). The `model.type` field discriminates which nested section applies (chemprop, chemeleon, xgboost, etc.). Training strategies (joint_sampling, task_affinity) are root-level and validated for model compatibility.

**Model Types:**

- `chemprop`, `chemeleon`: PyTorch-based, support curriculum learning and task affinity
- `xgboost`, `lightgbm`, `catboost`: Classical models using fingerprint features

**FFN Architectures:** All neural models support three FFN types via `ffn_factory.py`:

- `regression`: Standard MLP
- `mixture_of_experts`: MoE with gating network
- `branched`: Shared trunk with task-specific heads

**Ensemble Training:** Ray-parallelized training across split_N/fold_M directory structure. Configured via `ensemble.enabled: true` and `data.data_dir` pointing to split parent directory.

**HPO:** Ray Tune + ASHA scheduler. Search spaces defined in `hpo_search_space.py` with conditional parameters (MoE experts, branched layers). Results saved to `hpo_results/top_k_configs.json`.

### Config Directory Structure

```
configs/
├── 0-experiment/        # Single model experiments
├── 1-hpo-single/        # HPO configs
├── 2-hpo-ensemble/      # Ensemble HPO configs
├── 3-production/        # Production ensemble configs
├── 4-more-models/       # Classical model examples
├── curriculum/          # Curriculum learning configs
└── task-affinity/       # Task affinity grouping configs
```

## Development Workflow

**Commit Format:** Conventional Commits enforced by commitizen: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

**Pre-commit Hooks:** black, isort, flake8, pylint (≥9.0), mypy, pytest. Skip with: `SKIP=pytest,mypy git commit -m "message"`

**NumPy Docstrings:** Required for public API functions/classes.

**Line Length:** 120 characters (Black/flake8/pylint configured).

## Important Files

- `src/admet/model/config.py`: Master configuration schema
- `src/admet/cli/model.py`: CLI commands for train/ensemble/hpo
- `src/admet/model/chemprop/model.py`: ChempropModel training logic
- `src/admet/model/ensemble.py`: Ensemble orchestration
- `tests/conftest.py`: Shared pytest fixtures

## Predicted Endpoints

9 ADMET properties: LogD, Log KSOL, Log HLM CLint, Log MLM CLint, Log Caco-2 Permeability Papp A>B, Log Caco-2 Permeability Efflux, Log MPPB, Log MBPB, Log MGMB

## Recent Changes (January 2026)

### Dependency Version Pinning (Jan 2026)

- **All dependencies pinned to exact versions** for reproducibility and deterministic builds
- Main dependencies: numpy==1.26.4, pandas==2.3.3, torch==2.7.1+cu118, chemprop==2.2.1, rdkit==2023.9.6
- HPO dependencies: optuna==4.6.0, bayesian-optimization==3.2.0, hyperopt==0.2.7
- Dev/docs dependencies also pinned (black==25.11.0, sphinx==7.3.7, etc.)
- See `INSTALLATION.md` for version update procedures

### Weight Decay Regularization

- Added `weight_decay` parameter to `OptimizationConfig` (default: 0.0)
- Implemented `MPNNWithWeightDecay` subclass using AdamW optimizer
- Updated all 117+ config files with `weight_decay: 0.0` for backward compatibility
- HPO search space includes conditional weight_decay exploration (1e-6 to 1e-3)

### Bayesian Optimization Support

- Added `SearchAlgorithmConfig` to HPO configuration schemas
- Implemented `_build_search_algorithm()` in both ChempropHPO and ChemeleonHPO
- Supports Optuna (TPE), BayesOptSearch, and HyperOpt search algorithms
- Default: Optuna with 20 initial random trials for exploration
- 3-5x efficiency improvement over pure random sampling

**Usage:**

```yaml
optimization:
  weight_decay: 0.0  # Set to 1e-5 for regularization

search_algorithm:
  type: optuna       # optuna, bayesopt, hyperopt, random
  seed: 42
  n_initial_points: 20
```

**Documentation Updated:**

- `docs/guide/hpo.rst`: Added search algorithm and weight_decay sections
- `docs/guide/configuration.rst`: Added weight_decay to optimization examples
- `README.md`: Updated HPO section with Bayesian optimization features
