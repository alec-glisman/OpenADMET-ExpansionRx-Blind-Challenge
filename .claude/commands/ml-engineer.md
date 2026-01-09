# ML Engineer: Implement Model Improvements

Implement ML improvements with proper engineering practices.

## Your Role

Act as an ML engineer. Take a specific improvement idea and implement it end-to-end: code changes, tests, config updates, and validation.

## Before Starting

1. Read the relevant source files to understand current implementation
2. Identify all files that need modification
3. Plan backward-compatible changes

## Implementation Steps

### Step 1: Config Schema (if needed)
Add new parameters to `src/admet/model/config.py`:
```python
@dataclass
class NewFeatureConfig:
    param_name: float = 0.0
```

### Step 2: Core Implementation
Modify the appropriate module:
- Architecture: `src/admet/model/chemprop/model.py`, `ffn.py`
- Training: `curriculum.py`, `task_affinity.py`, `joint_sampler.py`
- HPO: `hpo.py`, `hpo_search_space.py`, `hpo_config.py`
- Data: `src/admet/data/split.py`, `src/admet/features/`

### Step 3: Add Tests
Create or update tests in `tests/`:
```bash
pytest tests/model/test_new_feature.py -v
```

### Step 4: Update Configs
Add defaults to existing YAML configs for backward compatibility.

### Step 5: Validate
```bash
# Run tests
pytest tests/ -q

# Lint check
pre-commit run --all-files

# Quick training test
admet model train -c configs/0-experiment/chemprop.yaml
```

## Quality Checklist

- [ ] Code follows existing patterns in codebase
- [ ] Tests cover new functionality
- [ ] Existing tests still pass
- [ ] Configs load without errors
- [ ] Docstrings added for new functions
- [ ] No breaking changes to existing behavior
