# OpenADMET Challenge - Copilot Instructions

ML pipeline for predicting ADMET (Absorption, Distribution, Metabolism, Excretion, Toxicity) properties of small molecules. Uses Chemprop MPNN as the primary model with ensemble training across 5 splits × 5 folds.

Before running any commands, ensure your environment is set up as per the instructions below.
Use `source .venv/bin/activate` in the project root to activate the virtual environment.
Use the `uv` tool for environment and package management.

## Tech Stack

- **Runtime:** Python 3.11
- **ML Framework:** PyTorch, Chemprop v2
- **Parallelization:** Ray Tune (HPO and ensemble training)
- **Experiment Tracking:** MLflow
- **Configuration:** OmegaConf (dataclass-based configs)
- **CLI:** Typer + Rich
- **Package Manager:** uv

## Environment Setup (Validated)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev,docs]"
uv run pre-commit install && uv run pre-commit install --hook-type commit-msg
```

## Testing Commands

```bash
pytest tests/ -q                          # All tests
pytest tests/test_chemprop_model.py -v    # Single file
pytest -m "not slow" -q                   # Skip slow tests
pytest -m "not no_mlflow_runs" -q         # Skip MLflow tests (default)
pytest -n auto -q                         # Parallel execution
```

## Linting & Formatting

```bash
black src/ tests/ && isort src/ tests/    # Format
flake8 src/ tests/                        # Style check
mypy src/admet/                           # Type check
pylint src/admet/ --fail-under=9.0        # Deep analysis
pre-commit run --all-files                # All hooks
```

**Line length:** 120 characters (Black/flake8/pylint configured).

## CLI Commands

```bash
admet --help
admet model train -c configs/0-experiment/0-single-fold/chemprop.yaml
admet model ensemble -c configs/3-hpo-ensemble-production/0_chemprop_v1/ensemble_chemprop_hpo_001.yaml --max-parallel 4
admet model hpo -c configs/1-hpo-single-fold/hpo_chemprop.yaml --num-samples 50
admet model hpo-list-studies --verbose    # List Optuna studies for warmstart
admet data split data.csv --cluster-method bitbirch
admet leaderboard scrape --user <username>
```

## Project Layout

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
configs/
├── 0-experiment/        # Single model experiments
│   ├── 0-single-fold/   # Single fold training configs
│   ├── 1-ensemble/      # Ensemble training configs
│   ├── 2-classical-models-ensemble/  # XGBoost, LightGBM, CatBoost
│   ├── curriculum-learning/  # Curriculum learning configs
│   └── task-affinity/   # Task affinity grouping configs
├── 1-hpo-single-fold/   # HPO configs for single fold
├── 2-hpo-ensemble/      # Ensemble HPO configs
└── 3-hpo-ensemble-production/  # Production ensemble configs
tests/                   # pytest test files
```

## Key Architectural Patterns

**Unified Config System:** All models use `UnifiedModelConfig` (src/admet/model/config.py). The `model.type` field discriminates which nested section applies (chemprop, chemeleon, xgboost, etc.).

**Model Types:**
- `chemprop`, `chemeleon`: PyTorch-based, support curriculum learning and task affinity
- `xgboost`, `lightgbm`, `catboost`: Classical models using fingerprint features

**FFN Architectures:** All neural models support three FFN types via `ffn_factory.py`:
- `regression`: Standard MLP
- `mixture_of_experts`: MoE with gating network
- `branched`: Shared trunk with task-specific heads

**Ensemble Training:** Ray-parallelized across split_N/fold_M directories. Enable via `ensemble.enabled: true`.

**HPO:** Ray Tune + ASHA scheduler. Results saved to `hpo_results/top_k_configs.json`.

## Important Files

- `src/admet/model/config.py`: Master configuration schema
- `src/admet/cli/model.py`: CLI commands for train/ensemble/hpo
- `src/admet/model/chemprop/model.py`: ChempropModel training logic
- `src/admet/model/ensemble.py`: Ensemble orchestration
- `tests/conftest.py`: Shared pytest fixtures

## Development Workflow

**Commit Format:** Conventional Commits enforced by commitizen: `feat:`, `fix:`, `docs:`, `refactor:`, `test:`, `chore:`

**Pre-commit Hooks:** black, isort, flake8, pylint (≥9.0), mypy, pytest.
Skip hooks: `SKIP=pytest,mypy git commit -m "message"`

**Docstrings:** NumPy-style required for public API functions/classes.

## Predicted Endpoints

9 ADMET properties: LogD, Log KSOL, Log HLM CLint, Log MLM CLint, Log Caco-2 Permeability Papp A>B, Log Caco-2 Permeability Efflux, Log MPPB, Log MBPB, Log MGMB
