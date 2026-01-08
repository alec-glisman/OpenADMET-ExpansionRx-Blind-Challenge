---
agent: 'pre-commit'
description: 'Execute all pre-commit hooks and iteratively fix all failures without prompting. Ensures code quality, formatting, linting, type checking, and tests all pass before commit.'
tools: ['search/changes', 'search/codebase', 'edit/editFiles', 'read/problems', 'execute/getTerminalOutput', 'execute/runInTerminal', 'execute/runTests', 'search', 'search/usages', 'github', 'pylance-mcp-server/*', 'ms-python.python/getPythonEnvironmentInfo', 'ms-python.python/getPythonExecutableCommand', 'ms-python.python/configurePythonEnvironment']
---

# Run Pre-Commit Hooks

Execute all pre-commit hooks autonomously and fix failures iteratively until all hooks pass or escalation is required.

## Purpose

This agent ensures your code meets all quality standards before committing by:

- Running all configured pre-commit hooks (formatting, linting, type checking, testing)
- Automatically fixing issues that can be resolved programmatically
- Iterating until all hooks pass or maximum attempts reached
- Providing detailed error context and escalating unresolvable issues

## When to Use

Invoke this agent when you want to:

- **Before committing**: Ensure all pre-commit hooks pass without manual intervention
- **After code changes**: Validate that your changes meet project quality standards
- **During development**: Fix linting, formatting, and type errors automatically
- **Pre-merge**: Verify branch is ready for pull request creation

## Quick Start

Simply invoke the agent without any parameters:

```
@workspace /run-pre-commit-hooks
```

Or use the agent directly in chat:

```
Run all pre-commit hooks and fix any failures
```

## What It Does

### Automatic Fixes

The agent will automatically fix:

- **Formatting**: black, isort, prettier, shfmt (shell scripts)
- **Whitespace**: trailing whitespace, end-of-file newlines
- **Imports**: unused imports, import organization
- **Line length**: break long lines at 120 characters
- **Docstrings**: add missing NumPy-style docstrings
- **Type hints**: add missing type annotations
- **Notebooks**: strip Jupyter notebook outputs

### Validation Checks

The agent validates and fixes:

- **flake8**: PEP 8 compliance, syntax errors
- **pylint**: code quality (minimum score 9.0/10.0)
- **mypy**: static type checking
- **pytest**: run test suite with parallel execution

### Safety Checks

The agent will escalate for manual review:

- **Merge conflicts**: Manual resolution required
- **Private keys**: Security concern requiring manual removal
- **Large files**: Files exceeding 1000KB size limit
- **Missing dependencies**: Package installation requires approval
- **Complex test failures**: Business logic errors requiring domain knowledge

## Configuration

The agent respects all settings in:

- `.pre-commit-config.yaml` - Hook configurations and exclusions
- `pyproject.toml` - Tool settings (black, isort, flake8, pylint, mypy)
- `.github/instructions/python.instructions.md` - Python coding conventions

## Environment Requirements

The agent automatically handles environment setup:

1. **Detects `uv` availability** (preferred) or falls back to virtual environment
2. **Validates Python 3.11.x** is active
3. **Checks development dependencies** are installed
4. **Activates virtual environment** if needed

No manual environment configuration required!

## Execution Phases

### Phase 0: Environment Setup
- Auto-detect `uv` or virtual environment
- Validate Python version and dependencies
- Configure command prefix (`uv run` or direct execution)

### Phase 1: Initial Scan
- Run all pre-commit hooks
- Capture full output for error analysis
- Parse failures by hook type

### Phase 2: Auto-Fix Formatting (Parallel)
- Run black, isort, prettier, shfmt concurrently
- Strip notebook outputs
- Remove trailing whitespace
- Performance gain: ~40-60% faster than sequential

### Phase 3: Fix Validation Issues (Parallel)
- Run flake8, pylint, mypy concurrently
- Parse errors with full context
- Apply automated fixes

### Phase 4: Fix Structural Issues
- Fix JSON/YAML/TOML syntax errors
- Correct Python AST errors
- Move docstrings to proper locations
- Remove debug statements

### Phase 5: Fix Tests
- Run pytest with parallel execution
- Fix import errors
- Update failing assertions
- Correct fixture usage

### Phase 6: Final Verification
- Re-run all hooks
- Confirm zero failures
- Generate summary report

## Iteration Behavior

- **Maximum iterations**: 5
- **Error tracking**: Full context captured across iterations
- **Git diff tracking**: Changes documented per iteration
- **Progress reporting**: Clear status updates with visual indicators (✓, ✗, ⚠️)
- **Log files**: Full hook output saved to timestamped files

## Enhanced Error Context

For each error, the agent captures:

- Hook name and error code
- File path and line number
- Surrounding code (5 lines before/after)
- Git blame information
- Previous fix attempts and outcomes
- Iteration history

## Success Criteria

All of the following must be satisfied:

- ✓ All formatting hooks pass
- ✓ All validation hooks pass (flake8, pylint ≥9.0, mypy)
- ✓ All structural hooks pass
- ✓ All safety hooks pass (or escalated)
- ✓ All tests pass (`pytest -n auto -q -m "not no_mlflow_runs"`)
- ✓ No pre-commit hook returns non-zero exit code

## Escalation

The agent will escalate to you when:

1. **Maximum iterations reached** (5) without resolving all issues
2. **Merge conflicts detected** - Manual resolution required
3. **Private keys found** - Security concern
4. **Large files added** - Manual review needed
5. **Missing dependencies** - Installation approval required
6. **Complex test failures** - Business logic requiring domain knowledge

### Escalation Format

When escalated, you'll receive:

- Comprehensive error summary with full context
- Git diff of all changes attempted
- Links to full output logs
- Prioritized list of next steps
- Affected files with line numbers

## Examples

### Example 1: Standard Usage

```
User: Run pre-commit hooks
Agent:
  Iteration 1/5: Running pre-commit hooks...
  - ✗ black failed (3 files)
  - ✗ flake8 failed (12 issues)
  - ✓ All other hooks passed

  Fixing formatting issues...
  Applied black to src/admet/model/config.py, tests/test_ensemble.py

  Iteration 2/5: Re-running pre-commit hooks...
  - ✓ All hooks passed!

  ✓ Success! All pre-commit hooks passed in 2 iterations.
```

### Example 2: With Escalation

```
User: Fix all pre-commit issues
Agent:
  Iteration 5/5 reached

  ⚠️ Pre-Commit Escalation Required

  Unresolved Hooks:
  - pylint: 2 issues in src/admet/model/chemprop/ensemble.py
    - C0116: Missing function docstring (line 245)
    - R0913: Too many arguments (line 267)

  Recommended Actions:
  1. Add docstring to `_aggregate_predictions` function
  2. Refactor function to use dataclass for parameters

  Full logs: .pre-commit-output-20260107_143022.log
```

### Example 3: Environment Auto-Configuration

```
User: Run hooks
Agent:
  ✓ Using uv for environment management
  ✓ Python version: 3.11.7
  ✓ All dev dependencies installed

  Iteration 1/5: Running pre-commit hooks...
  [continues with hook execution]
```

## Advanced Options

While the agent runs autonomously, you can provide context:

```
User: Run pre-commit hooks but skip pytest (it's slow)
Agent: [Will use SKIP=pytest environment variable]
```

```
User: Only run formatting hooks
Agent: [Will execute only black, isort, prettier, shfmt]
```

```
User: Run hooks with verbose output
Agent: [Will use --verbose flag and show detailed output]
```

## Troubleshooting

### Issue: "uv: command not found"
- **Solution**: Install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- **Fallback**: Agent will automatically use virtual environment

### Issue: "Virtual environment not activated"
- **Solution**: Agent will attempt to activate `.venv` automatically
- **Manual**: Run `source .venv/bin/activate`

### Issue: "Missing dev dependencies"
- **Solution**: Agent will report missing packages
- **Install**: Run `uv pip install -e '.[dev]'`

### Issue: "Wrong Python version"
- **Solution**: Recreate venv with Python 3.11
- **Command**: `uv venv --python 3.11`

## Performance Notes

- **Parallel execution**: Formatters and validators run concurrently (~50% faster)
- **Incremental fixes**: Only re-runs failed hooks after fixes applied
- **Cache aware**: Respects pre-commit's built-in caching
- **Resource efficient**: Uses pytest-xdist for parallel test execution

## Related Documentation

- [Pre-Commit Configuration](.pre-commit-config.yaml)
- [Python Coding Conventions](.github/instructions/python.instructions.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Installation Guide](INSTALLATION.md)

## Agent Behavior Guarantees

**The agent WILL**:
- Execute fixes immediately without confirmation
- Iterate until all hooks pass or max iterations reached
- Preserve original code intent and logic
- Document all changes with progress updates
- Capture full error context for debugging
- Use parallel execution for efficiency

**The agent WILL NOT**:
- Ask for permission before fixing
- Skip any failing hooks without escalation
- Make assumptions about business logic
- Modify excluded directories (src/bitbirch/)
- Change test assertions without understanding intent
- Stop before completion unless escalation required

## Technical Implementation

Powered by the **pre-commit.agent.md** located in `.github/agents/`, which provides:

- Comprehensive hook categorization
- Parallel execution strategies
- Enhanced error context tracking
- Git diff integration
- Automatic environment detection
- Fallback mechanisms

For detailed implementation, see [.github/agents/pre-commit.agent.md](.github/agents/pre-commit.agent.md).
