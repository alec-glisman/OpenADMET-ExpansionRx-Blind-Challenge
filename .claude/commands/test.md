# Run Test Suite

Run the pytest test suite for this ADMET ML pipeline.

## Commands

```bash
# Run all tests (excludes no_mlflow_runs by default)
pytest tests/ -q

# Skip slow tests
pytest -m "not slow" -q

# Parallel execution (faster)
pytest -n auto -q

# With coverage report
pytest --cov=src/admet --cov-report=term-missing tests/

# Verbose output
pytest tests/ -v
```

## Common Markers

- `@pytest.mark.slow` - Long-running tests
- `@pytest.mark.integration` - Integration tests
- `@pytest.mark.no_mlflow_runs` - Requires MLflow (excluded by default)

## Test Directories

- `tests/model/chemprop/` - Chemprop model tests
- `tests/model/chemeleon/` - Chemeleon model tests
- `tests/model/hpo/` - HPO orchestration tests
- `tests/cli/` - CLI command tests

## Quick Validation

```bash
# Fast smoke test
pytest tests/ -q -x --tb=short -m "not slow"
```
