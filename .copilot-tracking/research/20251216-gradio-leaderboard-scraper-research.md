<!-- markdownlint-disable-file -->

# Research: Productionize Gradio Leaderboard Scraper

## Overview

Research findings for integrating the `scrape_gradio.py` script into the `admet` package as a modular, production-ready CLI tool with comprehensive documentation and tests.

## Source Script Analysis

### Current Script Location
- **File**: `scrape_gradio.py` (project root, 1873 lines)
- **Purpose**: Scrapes OpenADMET ExpansionRx Challenge HuggingFace Space leaderboard, computes user rankings, generates plots and reports

### Key Components Identified

#### 1. Data Fetching (Lines 113-140)
```python
def _find_refresh_dependency(config: Dict[str, Any], n_expected_outputs: int) -> int:
    """Find the Gradio dependency (fn_index) that outputs all leaderboard tables."""
```

#### 2. DataFrame Conversion (Lines 60-112)
```python
def _to_dataframe(gradio_value: Any) -> pd.DataFrame:
    """Convert common Gradio table payload formats into a pandas DataFrame."""
```

#### 3. Value Parsing (Lines 148-165)
```python
def extract_value_uncertainty(val: Any) -> Tuple[Optional[float], Optional[float]]:
    """Parse strings like '0.35 +/- 0.01' or '0.35 ± 0.01'."""
```

#### 4. User Rank Extraction (Lines 167-194)
```python
def _row_rank_for_user(df: pd.DataFrame, target_user: str) -> Optional[int]:
    """Return 1-based row rank for target_user based on displayed row order."""
```

#### 5. Plot Generation (Lines 196-1483)
- 20+ publication-quality plots including:
  - Overall ranking distribution (histogram + ECDF)
  - Task-specific rankings bar chart
  - Delta MAE comparison
  - MAE comparison bar chart
  - Performance category pie chart
  - Multi-metric heatmaps
  - Rank vs R²/MAE/Spearman/Kendall scatter plots
  - Distribution KDE plots
  - Radar/spider chart
  - Percentile ranking analysis
  - Gap-to-leader waterfall
  - Metric correlation matrix
  - Rank improvement potential
  - Task difficulty vs performance
  - Priority matrix

#### 6. Report Generation (Lines 1622-1867)
- Markdown report matching SUBMISSIONS.md format
- CSV data export for analysis

#### 7. Constants and Configuration (Lines 54-58)
```python
SPACE = "openadmet/OpenADMET-ExpansionRx-Challenge"
TARGET_USER = "aglisman"  # case-insensitive match
CACHE_DIR = "assets/submissions"  # Base directory for caching
```

## Project Structure Analysis

### Package Layout
```
src/admet/
├── __init__.py          # Package root, version info
├── cli/                  # CLI entrypoint (planned via pyproject.toml)
├── data/                 # Data processing modules
├── model/                # ML model implementations
├── plot/                 # Visualization modules
│   ├── __init__.py      # Exports, matplotlib/seaborn setup
│   ├── density.py       # Distribution plots
│   ├── heatmap.py       # Heatmap utilities
│   ├── latex.py         # LaTeX string sanitization
│   ├── metrics.py       # Metric bar charts (571 lines)
│   ├── parity.py        # Parity plots
│   └── split.py         # Split visualization
└── util/
    ├── logging.py       # Logging configuration (166 lines)
    └── utils.py         # General utilities
```

### Logging Pattern (src/admet/util/logging.py, Lines 63-100)
```python
def configure_logging(
    level: str = "INFO",
    fmt: Optional[str] = None,
    file: Optional[str] = None,
    structured: bool = False,
) -> None:
    """Configure application-wide logging."""
```

Standard usage across codebase:
```python
import logging
logger = logging.getLogger(__name__)
```

### Plot Module Pattern (src/admet/plot/metrics.py)

#### Function Signature Style (Lines 143-186)
```python
def plot_metric_bar(
    values: np.ndarray,
    labels: Sequence[str],
    metric_name: str,
    *,
    errors: Optional[np.ndarray] = None,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
    color: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None,
    show_values: bool = True,
    show_mean: bool = True,
    value_fontsize: int = 9,
) -> Tuple[Figure, Axes]:
```

#### Return Type Convention
- All plot functions return `Tuple[Figure, Axes]` or `Tuple[Figure, ndarray]`

### CLI Pattern (src/admet/model/chemprop/hpo.py, Lines 405-420)

Uses argparse with RawDescriptionHelpFormatter:
```python
parser = argparse.ArgumentParser(
    description="Run HPO for Chemprop models",
    formatter_class=argparse.RawDescriptionHelpFormatter,
)
```

### pyproject.toml CLI Entry Point (Lines 119-121)
```toml
[project.scripts]
admet = "admet.cli.__main__:main"
```

Note: The CLI module doesn't exist yet but is referenced in pyproject.toml.

### Test Pattern (tests/test_hpo.py)

```python
"""Unit tests for HPO orchestrator module."""
from pathlib import Path
import pytest
from admet.model.chemprop.hpo import ChempropHPO, _flatten_dict

class TestFlattenDict:
    """Tests for _flatten_dict helper function."""

    def test_empty_dict(self) -> None:
        """Test flattening empty dictionary."""
        result = _flatten_dict({})
        assert result == {}
```

### Documentation Structure (docs/)
```
docs/
├── index.rst           # Main index with toctree
├── conf.py             # Sphinx configuration
├── api/                # API reference
└── guide/
    ├── cli.rst         # CLI usage documentation
    ├── modeling.rst    # Modeling guide
    └── ...
```

## External Dependencies

### Required (from pyproject.toml, Lines 34-36)
```toml
"gradio-client==2.0.1",
"matplotlib>=3.6",
"seaborn>=0.12",
"colorcet>=3.0",
"scienceplots==2.1.1",
```

### Visualization Setup (src/admet/plot/__init__.py, Lines 1-36)
```python
import colorcet as cc
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

GLASBEY_PALETTE = list(cc.glasbey)
matplotlib.use("Agg")  # Non-interactive backend
plt.style.use("science")  # or fallback
plt.rcParams["axes.prop_cycle"] = cycler(color=GLASBEY_PALETTE)
sns.set_palette(GLASBEY_PALETTE)
```

## Proposed Architecture

### New Package Structure
```
src/admet/
├── leaderboard/
│   ├── __init__.py          # Public API exports
│   ├── client.py            # GradioClient wrapper for scraping
│   ├── parser.py            # DataFrame conversion, value parsing
│   ├── config.py            # Configuration dataclass/constants
│   └── report.py            # Markdown/CSV report generation
├── plot/
│   ├── leaderboard.py       # New: Leaderboard-specific plots (20+ functions)
│   └── ... (existing)
└── cli/
    ├── __init__.py
    ├── __main__.py          # Main CLI entrypoint (typer app)
    └── leaderboard.py       # Leaderboard CLI commands
```

### Module Responsibilities

1. **admet.leaderboard.client** (~150 lines)
   - `LeaderboardClient` class wrapping gradio_client
   - `fetch_all_tables()` method
   - Error handling and retry logic

2. **admet.leaderboard.parser** (~200 lines)
   - `to_dataframe()` function
   - `extract_value_uncertainty()` function
   - `find_user_rank()` function
   - `normalize_user_cell()` helper

3. **admet.leaderboard.config** (~50 lines)
   - `LeaderboardConfig` dataclass
   - Default constants (SPACE, endpoints, etc.)

4. **admet.leaderboard.report** (~300 lines)
   - `generate_markdown_report()` function
   - `save_csv_data()` function
   - `save_summary_statistics()` function

5. **admet.plot.leaderboard** (~1200 lines)
   - Individual plot functions matching project patterns
   - Each returns `Tuple[Figure, Axes]`

6. **admet.cli.leaderboard** (~150 lines)
   - Typer CLI commands
   - `scrape` command with options
   - `report` command
   - `plot` command

### CLI Interface Design

```bash
# Basic scrape and report
admet leaderboard scrape --user aglisman

# With custom output directory
admet leaderboard scrape --user aglisman --output ./my_results

# Generate specific plots only
admet leaderboard plot --type rankings --type priority-matrix

# Generate report only (from cached data)
admet leaderboard report --data-dir assets/submissions/2025-12-16
```

## Test Coverage Requirements

### Unit Tests
1. `test_leaderboard_parser.py`
   - Test `to_dataframe()` with various input formats
   - Test `extract_value_uncertainty()` edge cases
   - Test `find_user_rank()` with different user formats

2. `test_leaderboard_config.py`
   - Test configuration loading
   - Test default values

3. `test_leaderboard_report.py`
   - Test markdown generation
   - Test CSV output format

### Integration Tests (marked with pytest.mark.integration)
1. `test_leaderboard_client.py`
   - Test actual Gradio API connection (optional, slow)
   - Mock tests for client behavior

### Plot Tests
1. `test_plot_leaderboard.py`
   - Test each plot function generates valid Figure
   - Test with mock data

## Documentation Requirements

### API Documentation
- Docstrings following PEP 257/Google style
- Type hints for all public functions

### Guide Documentation
- `docs/guide/leaderboard.rst` - User guide with examples
- Update `docs/guide/cli.rst` with leaderboard commands

### README/Examples
- Example usage in README or dedicated examples folder

## Implementation Notes

### Logging Integration
Replace print statements with:
```python
import logging
logger = logging.getLogger(__name__)

# Instead of: print(f"Cached {label} leaderboard...")
logger.info("Cached %s leaderboard to %s", label, table_path)
```

### Plot Function Refactoring Pattern
Current monolithic `_generate_plots()` becomes individual functions:
```python
def plot_overall_rank_distribution(
    df_overall: pd.DataFrame,
    user_rank: Optional[int] = None,
    *,
    ax: Optional[Axes] = None,
    title: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Plot overall ranking distribution with histogram and ECDF."""
```

### Configuration Management
Use dataclass for type safety:
```python
from dataclasses import dataclass, field

@dataclass
class LeaderboardConfig:
    space: str = "openadmet/OpenADMET-ExpansionRx-Challenge"
    target_user: str = "aglisman"
    cache_dir: str = "assets/submissions"
    endpoints: list = field(default_factory=lambda: [
        "LogD", "KSOL", "MLM CLint", "HLM CLint",
        "Caco-2 Permeability Efflux", "Caco-2 Permeability Papp A>B",
        "MPPB", "MBPB", "MGMB"
    ])
```

## File Changes Summary

### New Files
1. `src/admet/leaderboard/__init__.py`
2. `src/admet/leaderboard/client.py`
3. `src/admet/leaderboard/parser.py`
4. `src/admet/leaderboard/config.py`
5. `src/admet/leaderboard/report.py`
6. `src/admet/plot/leaderboard.py`
7. `src/admet/cli/__init__.py`
8. `src/admet/cli/__main__.py`
9. `src/admet/cli/leaderboard.py`
10. `tests/test_leaderboard_parser.py`
11. `tests/test_leaderboard_client.py`
12. `tests/test_leaderboard_report.py`
13. `tests/test_plot_leaderboard.py`
14. `docs/guide/leaderboard.rst`

### Modified Files
1. `src/admet/plot/__init__.py` - Add leaderboard exports
2. `docs/guide/cli.rst` - Add leaderboard CLI documentation
3. `docs/index.rst` - Add leaderboard to toctree

### Deleted Files
1. `scrape_gradio.py` - After migration complete
