# Run Single Test

Run a specific test file or test function.

## Usage

Provide a test path as argument: `/test-single tests/model/test_ffn_factory.py`

## Commands

```bash
# Single file
pytest $ARGUMENTS -v

# Single test function
pytest tests/model/test_file.py::test_function_name -v

# Single test class
pytest tests/model/test_file.py::TestClassName -v

# With output capture disabled (see print statements)
pytest $ARGUMENTS -v -s
```

## Common Test Files

- `tests/model/chemprop/test_hpo.py` - HPO tests
- `tests/model/test_ffn_factory.py` - FFN architecture tests
- `tests/model/test_unified_config.py` - Config validation
- `tests/cli/test_cli_model.py` - CLI command tests
- `tests/test_hpo_metrics.py` - Metric computation tests

## Debug Mode

```bash
# Stop on first failure, show locals
pytest $ARGUMENTS -v -x --tb=long -l
```
