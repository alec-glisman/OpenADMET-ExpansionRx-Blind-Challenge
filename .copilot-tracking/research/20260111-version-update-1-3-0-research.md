<!-- markdownlint-disable-file -->

# Task Research Notes: Repository Version Update to 1.3.0

## Research Executed

### File Analysis

- [pyproject.toml](../../pyproject.toml)
  - Version declared as dynamic: `version = { attr = "admet.__version__" }`
  - Python version requirement: `>=3.11,<3.12`
  - Build system: setuptools (>=61) + wheel
  - No commitizen configuration found in pyproject.toml

- [src/admet/__init__.py](../../src/admet/__init__.py)
  - **Current version**: `__version__ = "1.2.1"` (line 14)
  - This is the single source of truth for package version

- [CHANGELOG.md](../../CHANGELOG.md)
  - Follows Keep a Changelog format
  - Uses Semantic Versioning 2.0.0
  - Currently has [Unreleased] section with Performance Optimization Features (2026-01-04)
  - Previous changes reference: `.github/CHANGELOG_WEIGHT_DECAY_BAYESOPT.md`

- [docs/conf.py](../../docs/conf.py)
  - No hardcoded version - uses package metadata dynamically
  - Configured for Sphinx 7.3.7 with Furo theme
  - sys.path includes `../src` for autodoc imports

- [docs/Makefile](../../docs/Makefile)
  - Build command: `sphinx-build -b html "." "_build/html"`
  - Clean command: `rm -rf "_build"`
  - Live server: `sphinx-autobuild "." "_build/html" --open-browser`

- [MODEL_CARD.md](../../MODEL_CARD.md)
  - Line 63: `- **Version:** 1.2.0` (outdated, should be 1.2.1)

- [docs/api/admet.rst](../../docs/api/admet.rst)
  - Line 31: `print(admet.__version__)  # "1.2.1"` (correct)

- [.pre-commit-config.yaml](../../.pre-commit-config.yaml)
  - Commitizen hook configured: rev v4.10.0
  - No commitizen config section found in pyproject.toml
  - Pre-commit hooks that will run on commit:
    * trailing-whitespace, end-of-file-fixer (exclude docs/)
    * check-merge-conflict, mixed-line-ending, detect-private-key
    * check-json, check-yaml, check-toml
    * check-added-large-files (--maxkb=1000)
    * check-case-conflict, check-docstring-first
    * name-tests-test, debug-statements, check-ast
    * prettier (TOML/YAML only)
    * nbstripout (keep-output-metadata, strip-newlines)
    * beautysh (shfmt -w -i 2)
    * black, isort (exclude docs/)
    * flake8 (exclude docs/, notebooks/)
    * pylint (exclude docs/, notebooks/, tests/ - fail-under=9.0)
    * mypy (exclude docs/, notebooks/, tests/, archive/, src/bitbirch/, scripts/)
    * pytest (-q -m "not no_mlflow_runs")

### Code Search Results

- Search: `1\.2\.0` (regex)
  - [MODEL_CARD.md](../../MODEL_CARD.md) line 63: Version reference
  - [.copilot-tracking/](../../.copilot-tracking/) files: Historical version references from previous updates
  - [.github/prompts/](../../.github/prompts/) files: Example version strings in templates

- Search: `__version__`
  - [src/admet/__init__.py](../../src/admet/__init__.py) line 14: `__version__ = "1.2.1"`
  - [src/admet/model/chemprop/model.py](../../src/admet/model/chemprop/model.py): Uses torch.__version__, pl.__version__, chemprop.__version__, rdkit.__version__ for environment logging

### Project Conventions

- Standards referenced:
  - [.github/copilot-instructions.md](../../.github/copilot-instructions.md): Conventional Commits required (feat, fix, docs, refactor, test, chore)
  - Commitizen enforced via pre-commit hook
  - NumPy-style docstrings for public API

- Build system: uv package manager
  - Environment setup: `uv venv && source .venv/bin/activate`
  - Installation: `uv pip install -e ".[dev,docs]"`
  - Pre-commit setup: `uv run pre-commit install && uv run pre-commit install --hook-type commit-msg`

### External Research

- #githubRepo:"commitizen-tools/commitizen bump"
  - Commitizen provides `cz bump` command for automated version bumping
  - Requires `[tool.commitizen]` configuration in pyproject.toml
  - Can automatically update CHANGELOG.md following conventional commits

## Key Discoveries

### Project Structure

**Version Management:**
- Single source of truth: `src/admet/__init__.py` (`__version__ = "1.2.1"`)
- Dynamic version in pyproject.toml: `version = { attr = "admet.__version__" }`
- Sphinx docs auto-detect version from package
- No GitHub Actions workflows directory exists (`.github/workflows/` not found)
- No automated release process detected

**Current State:**
- Package version: 1.2.1
- MODEL_CARD.md still references: 1.2.0 (needs update)
- CHANGELOG.md has extensive [Unreleased] section ready for 1.3.0

**Files Needing Version Updates:**
1. `src/admet/__init__.py` - Change `__version__ = "1.2.1"` to `"1.3.0"`
2. `MODEL_CARD.md` - Update version reference from 1.2.0 to 1.3.0
3. `CHANGELOG.md` - Move [Unreleased] content to new [1.3.0] section with date

### Implementation Patterns

**CHANGELOG Format (Keep a Changelog):**
```markdown
## [Unreleased]

## [1.3.0] - 2026-01-11

### Added
- Feature descriptions

### Changed
- Modifications

### Fixed
- Bug fixes

## [1.2.1] - YYYY-MM-DD
...
```

**Version String Format:**
- Uses Semantic Versioning (MAJOR.MINOR.PATCH)
- Current: 1.2.1
- Target: 1.3.0 (minor version bump for new features)

### Complete Examples

**Version Update in __init__.py:**
```python
"""
ADMET Prediction Package
========================
...
"""

from __future__ import annotations

__version__ = "1.3.0"  # Updated from "1.2.1"

# Leaderboard module for Gradio scraping and analysis
from admet import leaderboard

__all__ = ["leaderboard"]
```

**CHANGELOG Section Move:**
```markdown
## [1.3.0] - 2026-01-11

### Added - Performance Optimization Features (2026-01-04)

#### Core Optimizations
[... existing unreleased content ...]

## [1.2.1] - YYYY-MM-DD
[... previous version ...]
```

### API and Schema Documentation

**pyproject.toml Configuration:**
- Build system: setuptools.build_meta (backend)
- Version declaration: Dynamic via `tool.setuptools.dynamic`
- Python requirement: `>=3.11,<3.12` (strict 3.11 only)
- Entry point: `admet = "admet.cli.__main__:main"`

**No Commitizen Configuration Found:**
- Commitizen is installed as dev dependency (v4.10.0)
- Pre-commit hook exists but no `[tool.commitizen]` section in pyproject.toml
- Manual version updates required (no automated `cz bump`)

### Configuration Examples

**Sphinx Documentation Build:**
```bash
# From repository root
make -C docs html          # Build HTML docs
make -C docs clean         # Clean build artifacts
make -C docs live          # Live reload server

# Or from docs/ directory
sphinx-build -b html . _build/html
sphinx-autobuild . _build/html --open-browser
```

**Pre-commit Execution:**
```bash
# Install hooks
uv run pre-commit install
uv run pre-commit install --hook-type commit-msg

# Run all hooks
uv run pre-commit run --all-files

# Skip specific hooks during commit
SKIP=pytest,mypy git commit -m "chore: bump version to 1.3.0"
```

### Technical Requirements

**Python Version:**
- **Required**: Python 3.11 (strict)
- **Not compatible**: Python 3.12+

**Build Tools:**
- setuptools >= 61
- wheel
- uv package manager (recommended)

**Documentation:**
- Sphinx 7.3.7
- sphinx-autobuild 2025.8.25 (for live server)
- Furo theme 2025.9.25

**Pre-commit Hooks Execution Order:**
1. Formatting: black, isort, beautysh
2. Linting: flake8, pylint (≥9.0), mypy
3. Testing: pytest (non-MLflow tests)
4. Validation: prettier, check-yaml, check-toml
5. Commitizen: conventional commit message validation

## Recommended Approach

**Manual Version Update Process** (No automated tooling available)

### Update Sequence

1. **Update Package Version**
   - File: `src/admet/__init__.py`
   - Change: `__version__ = "1.2.1"` → `__version__ = "1.3.0"`

2. **Update Documentation Version References**
   - File: `MODEL_CARD.md` (line 63)
   - Change: `- **Version:** 1.2.0` → `- **Version:** 1.3.0`

3. **Finalize CHANGELOG**
   - File: `CHANGELOG.md`
   - Move `## [Unreleased]` content to `## [1.3.0] - 2026-01-11`
   - Create new empty `## [Unreleased]` section above it
   - Add version comparison link at bottom if following Keep a Changelog format

4. **Rebuild Documentation**
   - Command: `make -C docs clean && make -C docs html`
   - Verify: Check `docs/_build/html/` for version updates
   - Optional: Test with `make -C docs live` for live preview

5. **Commit Changes**
   - Format: `chore: bump version to 1.3.0`
   - Pre-commit will run: black, isort, flake8, pylint, mypy, pytest
   - If tests fail: `SKIP=pytest git commit -m "chore: bump version to 1.3.0"`

6. **Verify Installation**
   - Reinstall: `uv pip install -e .`
   - Verify: `python -c "import admet; print(admet.__version__)"`
   - Expected: `1.3.0`

### Files to Update

| File | Line | Current Value | New Value |
|------|------|---------------|-----------|
| `src/admet/__init__.py` | 14 | `"1.2.1"` | `"1.3.0"` |
| `MODEL_CARD.md` | 63 | `1.2.0` | `1.3.0` |
| `CHANGELOG.md` | 9 | `## [Unreleased]` | `## [1.3.0] - 2026-01-11` |

### Pre-commit Considerations

**Hooks That Will Run:**
- ✅ **Will Pass**: black, isort, beautysh, prettier, nbstripout
- ✅ **Will Pass**: flake8, trailing-whitespace, end-of-file-fixer
- ⚠️ **May Require Attention**:
  - pylint (≥9.0 score required)
  - mypy (type checking)
  - pytest (all non-MLflow tests)

**Skip Strategy (if needed):**
```bash
# Skip slow hooks if only updating version strings
SKIP=pytest,mypy git commit -m "chore: bump version to 1.3.0"

# Or skip all hooks for version bump commit
git commit --no-verify -m "chore: bump version to 1.3.0"
```

### No Automated Release Process

**Findings:**
- No `.github/workflows/` directory exists
- No CI/CD configuration for releases
- No automated changelog generation
- No git tagging automation
- Manual process required for all version updates

**Future Enhancement Opportunity:**
- Add `[tool.commitizen]` configuration to pyproject.toml
- Create GitHub Actions workflow for automated releases
- Set up automated changelog generation from conventional commits

## Implementation Guidance

- **Objectives**: Update repository version from 1.2.1 to 1.3.0, finalize CHANGELOG, update documentation
- **Key Tasks**:
  1. Update `__version__` in src/admet/__init__.py
  2. Update MODEL_CARD.md version reference
  3. Move CHANGELOG [Unreleased] to [1.3.0] with date
  4. Rebuild Sphinx documentation
  5. Commit with conventional commit message
  6. Verify installation and version detection

- **Dependencies**:
  - uv package manager (for installation)
  - Sphinx + sphinx-autobuild (for docs)
  - Pre-commit hooks (for validation)

- **Success Criteria**:
  - `python -c "import admet; print(admet.__version__)"` outputs `1.3.0`
  - Sphinx docs build without errors: `make -C docs html`
  - MODEL_CARD.md shows version 1.3.0
  - CHANGELOG.md has dated [1.3.0] section
  - All pre-commit hooks pass (or explicitly skipped with reason)
  - Package reinstallation works: `uv pip install -e .`
