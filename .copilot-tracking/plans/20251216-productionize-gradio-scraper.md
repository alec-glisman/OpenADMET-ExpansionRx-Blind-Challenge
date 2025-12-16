# Plan: Productionize Gradio Leaderboard Scraper

## Summary

Integrate the `scrape_gradio.py` script into the `admet` package as a modular, production-ready CLI tool with comprehensive documentation and tests.

## Research Reference

- [Research File](../research/20251216-gradio-leaderboard-scraper-research.md)

---

## Phase 1: Core Infrastructure

### Task 1.1: Create Leaderboard Configuration Module

**File**: `src/admet/leaderboard/config.py`
**Estimated Lines**: ~60

Create a dataclass-based configuration module with:

- `LeaderboardConfig` dataclass with type-safe fields
- Default constants (SPACE, endpoints, cache directory)
- Configuration validation

### Task 1.2: Create Leaderboard Parser Module

**File**: `src/admet/leaderboard/parser.py`
**Estimated Lines**: ~200

Extract and refactor parsing functions:

- `to_dataframe()` - Convert Gradio payloads to pandas DataFrame
- `extract_value_uncertainty()` - Parse "0.35 +/- 0.01" format strings
- `find_user_rank()` - Find user's rank in leaderboard
- `normalize_user_cell()` - Normalize user cell text for matching

### Task 1.3: Create Leaderboard Client Module

**File**: `src/admet/leaderboard/client.py`
**Estimated Lines**: ~180

Create a client wrapper for Gradio API:

- `LeaderboardClient` class with connection management
- `fetch_all_tables()` method with retry logic
- `_find_refresh_dependency()` helper
- Error handling and logging

### Task 1.4: Create Leaderboard Package Init

**File**: `src/admet/leaderboard/__init__.py`
**Estimated Lines**: ~30

Export public API:

- `LeaderboardClient`
- `LeaderboardConfig`
- Parser functions

---

## Phase 2: Report Generation

### Task 2.1: Create Report Generation Module

**File**: `src/admet/leaderboard/report.py`
**Estimated Lines**: ~350

Extract and refactor report generation:

- `generate_markdown_report()` - Create SUBMISSIONS.md-style report
- `save_csv_data()` - Export tables to CSV
- `save_summary_statistics()` - Save summary JSON/CSV
- `ResultsData` dataclass for structured results

---

## Phase 3: Visualization Integration

### Task 3.1: Create Leaderboard Plot Module

**File**: `src/admet/plot/leaderboard.py`
**Estimated Lines**: ~1400

Refactor all 20+ plots into individual functions following project patterns:

- `plot_overall_rank_distribution()` - Histogram + ECDF
- `plot_task_rankings_bar()` - Task-specific rankings
- `plot_delta_mae_comparison()` - Performance gap analysis
- `plot_mae_comparison()` - MAE bar chart with minimums
- `plot_performance_category_pie()` - Performance distribution
- `plot_metrics_heatmap()` - Multi-metric heatmap
- `plot_rank_vs_metric_scatter()` - Rank correlation scatter plots
- `plot_distribution_kde()` - Distribution KDE plots
- `plot_radar_chart()` - Multi-metric radar/spider chart
- `plot_percentile_analysis()` - Percentile ranking
- `plot_gap_to_leader_waterfall()` - Gap analysis waterfall
- `plot_metric_correlation_matrix()` - Metric correlations
- `plot_rank_improvement_potential()` - Improvement analysis
- `plot_task_difficulty_performance()` - Difficulty vs performance
- `plot_priority_matrix()` - Priority matrix

All functions return `Tuple[Figure, Axes]` per project convention.

### Task 3.2: Update Plot Package Exports

**File**: `src/admet/plot/__init__.py`
**Action**: Add leaderboard module exports

---

## Phase 4: CLI Implementation

### Task 4.1: Create CLI Package Structure

**File**: `src/admet/cli/__init__.py`
**Estimated Lines**: ~20

Create CLI package with typer app initialization.

### Task 4.2: Create CLI Main Entry Point

**File**: `src/admet/cli/__main__.py`
**Estimated Lines**: ~30

Create main entry point that:

- Imports and registers subcommands
- Configures logging
- Handles global options

### Task 4.3: Create Leaderboard CLI Commands

**File**: `src/admet/cli/leaderboard.py`
**Estimated Lines**: ~200

Implement CLI commands:

- `scrape` - Full scrape, analyze, and report pipeline
- `report` - Generate report from cached data
- `plot` - Generate specific plots

Options:

- `--user` / `-u`: Target username
- `--output` / `-o`: Output directory
- `--space`: HuggingFace Space URL
- `--no-plots`: Skip plot generation
- `--format`: Output format (markdown, json, csv)

---

## Phase 5: Testing

### Task 5.1: Create Parser Tests

**File**: `tests/test_leaderboard_parser.py`
**Estimated Lines**: ~200

Test cases:

- `to_dataframe()` with dict, list, DataFrame inputs
- `extract_value_uncertainty()` with valid/invalid strings
- `find_user_rank()` with various user formats
- Edge cases: empty data, missing columns, None values

### Task 5.2: Create Client Tests

**File**: `tests/test_leaderboard_client.py`
**Estimated Lines**: ~150

Test cases:

- `_find_refresh_dependency()` logic
- Mock Gradio client responses
- Error handling scenarios
- Integration test (marked `@pytest.mark.integration`)

### Task 5.3: Create Report Tests

**File**: `tests/test_leaderboard_report.py`
**Estimated Lines**: ~150

Test cases:

- Markdown report generation format
- CSV output correctness
- Summary statistics calculation

### Task 5.4: Create Plot Tests

**File**: `tests/test_plot_leaderboard.py`
**Estimated Lines**: ~300

Test cases:

- Each plot function produces valid Figure/Axes
- Correct handling of missing data
- Parameter validation

---

## Phase 6: Documentation

### Task 6.1: Create Leaderboard User Guide

**File**: `docs/guide/leaderboard.rst`
**Estimated Lines**: ~200

Contents:

- Overview and purpose
- Installation requirements
- CLI usage examples
- Python API examples
- Configuration options
- Output file descriptions

### Task 6.2: Update CLI Documentation

**File**: `docs/guide/cli.rst`
**Action**: Add leaderboard commands section

### Task 6.3: Update Documentation Index

**File**: `docs/index.rst`
**Action**: Add leaderboard guide to toctree

### Task 6.4: Add API Documentation

**File**: `docs/api/leaderboard.rst`
**Estimated Lines**: ~50

Autodoc configuration for leaderboard modules.

---

## Phase 7: Cleanup and Finalization

### Task 7.1: Archive Original Script

**Action**: Move `scrape_gradio.py` to `archive/` or delete after verification

### Task 7.2: Update Package Init

**File**: `src/admet/__init__.py`
**Action**: Add leaderboard module to package exports

### Task 7.3: Verify pyproject.toml

**File**: `pyproject.toml`
**Action**: Ensure CLI entry point is correctly configured

---

## Implementation Order

```
Phase 1 (Infrastructure) ──┬── Task 1.1 (config)
                           ├── Task 1.2 (parser)
                           ├── Task 1.3 (client)
                           └── Task 1.4 (__init__)
                                   │
Phase 2 (Reports) ─────────────── Task 2.1 (report)
                                   │
Phase 3 (Visualization) ───┬── Task 3.1 (plots)
                           └── Task 3.2 (exports)
                                   │
Phase 4 (CLI) ─────────────┬── Task 4.1 (__init__)
                           ├── Task 4.2 (__main__)
                           └── Task 4.3 (commands)
                                   │
Phase 5 (Testing) ─────────┬── Task 5.1 (parser)
                           ├── Task 5.2 (client)
                           ├── Task 5.3 (report)
                           └── Task 5.4 (plots)
                                   │
Phase 6 (Documentation) ───┬── Task 6.1 (guide)
                           ├── Task 6.2 (cli docs)
                           ├── Task 6.3 (index)
                           └── Task 6.4 (api)
                                   │
Phase 7 (Cleanup) ─────────┬── Task 7.1 (archive)
                           ├── Task 7.2 (exports)
                           └── Task 7.3 (verify)
```

---

## New Files Summary

| File | Lines | Description |
|------|-------|-------------|
| `src/admet/leaderboard/__init__.py` | ~30 | Package exports |
| `src/admet/leaderboard/config.py` | ~60 | Configuration dataclass |
| `src/admet/leaderboard/parser.py` | ~200 | Data parsing utilities |
| `src/admet/leaderboard/client.py` | ~180 | Gradio API client |
| `src/admet/leaderboard/report.py` | ~350 | Report generation |
| `src/admet/plot/leaderboard.py` | ~1400 | Visualization functions |
| `src/admet/cli/__init__.py` | ~20 | CLI package |
| `src/admet/cli/__main__.py` | ~30 | CLI entry point |
| `src/admet/cli/leaderboard.py` | ~200 | CLI commands |
| `tests/test_leaderboard_parser.py` | ~200 | Parser tests |
| `tests/test_leaderboard_client.py` | ~150 | Client tests |
| `tests/test_leaderboard_report.py` | ~150 | Report tests |
| `tests/test_plot_leaderboard.py` | ~300 | Plot tests |
| `docs/guide/leaderboard.rst` | ~200 | User guide |
| `docs/api/leaderboard.rst` | ~50 | API docs |
| **Total** | **~3520** | |

## Modified Files Summary

| File | Action |
|------|--------|
| `src/admet/__init__.py` | Add leaderboard exports |
| `src/admet/plot/__init__.py` | Add leaderboard plot exports |
| `docs/guide/cli.rst` | Add leaderboard CLI section |
| `docs/index.rst` | Add leaderboard to toctree |

## Deleted Files

| File | Reason |
|------|--------|
| `scrape_gradio.py` | Migrated to `admet.leaderboard` |

---

## Success Criteria

1. ✅ All functionality from `scrape_gradio.py` preserved
2. ✅ CLI works: `admet leaderboard scrape --user aglisman`
3. ✅ Tests pass with >90% coverage on new code
4. ✅ Documentation builds without warnings
5. ✅ Type hints pass mypy checks
6. ✅ Code passes ruff linting
7. ✅ Generated plots match original quality
