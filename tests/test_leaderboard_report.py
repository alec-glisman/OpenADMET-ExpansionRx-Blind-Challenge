"""Unit tests for leaderboard report module."""

from __future__ import annotations

import pandas as pd
import pytest

from admet.leaderboard.report import (
    ResultsData,
    _calc_delta_pct,
    _get_performance_note,
    generate_markdown_report,
    save_csv_data,
    save_summary_statistics,
)


class TestCalcDeltaPct:
    """Tests for _calc_delta_pct helper function."""

    def test_simple_calculation(self) -> None:
        """Test simple delta calculation."""
        result = _calc_delta_pct("0.40", "0.30")
        assert result == "25.0%"

    def test_with_uncertainty(self) -> None:
        """Test with +/- uncertainty."""
        result = _calc_delta_pct("0.35 +/- 0.01", "0.30")
        assert result == "14.3%"

    def test_with_pm_symbol(self) -> None:
        """Test with ± symbol."""
        result = _calc_delta_pct("0.35 ± 0.01", "0.30")
        assert result == "14.3%"

    def test_zero_mean(self) -> None:
        """Test with zero mean value."""
        result = _calc_delta_pct("0.0", "0.0")
        assert result == "0.0%"

    def test_invalid_input(self) -> None:
        """Test with invalid input."""
        result = _calc_delta_pct("invalid", "0.30")
        assert result == "N/A"


class TestGetPerformanceNote:
    """Tests for _get_performance_note helper function."""

    def test_excellent_performance(self) -> None:
        """Test excellent performance categorization."""
        result = _get_performance_note(5, 100)
        assert "Excellent" in result
        assert "Top 5.0%" in result

    def test_good_performance(self) -> None:
        """Test good performance categorization."""
        result = _get_performance_note(15, 100)
        assert "Good" in result
        assert "Top 15.0%" in result

    def test_okay_performance(self) -> None:
        """Test okay performance categorization."""
        result = _get_performance_note(30, 100)
        assert "Okay" in result

    def test_poor_performance(self) -> None:
        """Test poor performance categorization."""
        result = _get_performance_note(50, 100)
        assert "Poor" in result

    def test_needs_improvement(self) -> None:
        """Test needs improvement categorization."""
        result = _get_performance_note(70, 100)
        assert "Needs improvement" in result

    def test_without_total(self) -> None:
        """Test without total count."""
        result = _get_performance_note(5)
        assert "Excellent" in result
        assert "%" not in result

    def test_invalid_rank(self) -> None:
        """Test with invalid rank."""
        result = _get_performance_note(None)
        assert "unavailable" in result


class TestGenerateMarkdownReport:
    """Tests for generate_markdown_report function."""

    @pytest.fixture
    def sample_results(self) -> ResultsData:
        """Create sample results data."""
        summary_df = pd.DataFrame(
            [
                {
                    "task": "OVERALL",
                    "rank": 5,
                    "n_rows": 100,
                    "ma-rae": "0.35 +/- 0.01",
                    "r2": "0.85",
                    "spearman r": "0.90",
                    "kendall's tau": "0.75",
                },
                {
                    "task": "LogD",
                    "rank": 10,
                    "n_rows": 50,
                    "mae": "0.25 +/- 0.02",
                    "r2": "0.80",
                    "spearman r": "0.85",
                    "kendall's tau": "0.70",
                },
                {
                    "task": "KSOL",
                    "rank": 20,
                    "n_rows": 50,
                    "mae": "0.30 +/- 0.01",
                    "r2": "0.75",
                    "spearman r": "0.80",
                    "kendall's tau": "0.65",
                },
            ]
        )

        return ResultsData(
            summary_df=summary_df,
            tables={"Average": pd.DataFrame(), "LogD": pd.DataFrame(), "KSOL": pd.DataFrame()},
            task_mins={"LogD": 0.20, "KSOL": 0.25},
            overall_min=0.30,
            target_user="testuser",
            timestamp="2025-12-16 10:00:00+00:00",
        )

    def test_generate_report_creates_file(self, sample_results, tmp_path) -> None:
        """Test report file is created."""
        output_path = tmp_path / "report.md"

        generate_markdown_report(sample_results, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_generate_report_contains_user(self, sample_results, tmp_path) -> None:
        """Test report contains target user."""
        output_path = tmp_path / "report.md"

        generate_markdown_report(sample_results, output_path)

        content = output_path.read_text()
        assert "testuser" in content

    def test_generate_report_contains_overall_stats(self, sample_results, tmp_path) -> None:
        """Test report contains overall statistics."""
        output_path = tmp_path / "report.md"

        generate_markdown_report(sample_results, output_path)

        content = output_path.read_text()
        assert "Overall" in content
        assert "5/100" in content  # rank/total

    def test_generate_report_contains_task_stats(self, sample_results, tmp_path) -> None:
        """Test report contains task statistics."""
        output_path = tmp_path / "report.md"

        generate_markdown_report(sample_results, output_path)

        content = output_path.read_text()
        assert "LogD" in content
        assert "KSOL" in content

    def test_generate_report_with_figures(self, sample_results, tmp_path) -> None:
        """Test report includes figure links."""
        output_path = tmp_path / "report.md"

        generate_markdown_report(sample_results, output_path, include_figures=True)

        content = output_path.read_text()
        assert "figures/" in content
        assert ".png" in content

    def test_generate_report_without_figures(self, sample_results, tmp_path) -> None:
        """Test report excludes figures when requested."""
        output_path = tmp_path / "report.md"

        generate_markdown_report(sample_results, output_path, include_figures=False)

        content = output_path.read_text()
        assert "Visual Highlights" not in content

    def test_generate_report_creates_parent_dir(self, sample_results, tmp_path) -> None:
        """Test report creates parent directories."""
        output_path = tmp_path / "subdir" / "report.md"

        generate_markdown_report(sample_results, output_path)

        assert output_path.exists()


class TestSaveCsvData:
    """Tests for save_csv_data function."""

    @pytest.fixture
    def sample_results(self) -> ResultsData:
        """Create sample results data."""
        summary_df = pd.DataFrame([{"task": "LogD", "rank": 10, "mae": "0.25"}])
        tables = {
            "Average": pd.DataFrame([{"rank": 1, "user": "alice"}]),
            "LogD": pd.DataFrame([{"rank": 1, "user": "bob"}]),
        }

        return ResultsData(
            summary_df=summary_df,
            tables=tables,
            task_mins={},
            overall_min=None,
            target_user="testuser",
            timestamp="2025-12-16",
        )

    def test_saves_summary_csv(self, sample_results, tmp_path) -> None:
        """Test summary CSV is saved."""
        save_csv_data(sample_results, tmp_path)

        summary_path = tmp_path / "summary.csv"
        assert summary_path.exists()

        df = pd.read_csv(summary_path)
        assert "task" in df.columns
        assert "rank" in df.columns

    def test_saves_endpoint_csvs(self, sample_results, tmp_path) -> None:
        """Test endpoint CSVs are saved."""
        save_csv_data(sample_results, tmp_path)

        assert (tmp_path / "Average.csv").exists()
        assert (tmp_path / "LogD.csv").exists()

    def test_sanitizes_filenames(self, sample_results, tmp_path) -> None:
        """Test filenames are sanitized."""
        sample_results.tables["Task / Name"] = pd.DataFrame([{"a": 1}])

        save_csv_data(sample_results, tmp_path)

        # Spaces and slashes should be replaced
        assert (tmp_path / "Task___Name.csv").exists()


class TestSaveSummaryStatistics:
    """Tests for save_summary_statistics function."""

    @pytest.fixture
    def sample_results(self) -> ResultsData:
        """Create sample results data."""
        summary_df = pd.DataFrame(
            [
                {"task": "OVERALL", "rank": 5, "n_rows": 100, "ma-rae": "0.35", "r2": "0.85"},
                {"task": "LogD", "rank": 10, "n_rows": 50, "mae": "0.25", "r2": "0.80"},
            ]
        )

        return ResultsData(
            summary_df=summary_df,
            tables={},
            task_mins={},
            overall_min=None,
            target_user="testuser",
            timestamp="2025-12-16 10:00:00+00:00",
        )

    def test_creates_summary_file(self, sample_results, tmp_path) -> None:
        """Test summary statistics file is created."""
        output_path = tmp_path / "summary.txt"

        save_summary_statistics(sample_results, output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_contains_timestamp(self, sample_results, tmp_path) -> None:
        """Test summary contains timestamp."""
        output_path = tmp_path / "summary.txt"

        save_summary_statistics(sample_results, output_path)

        content = output_path.read_text()
        assert "2025-12-16" in content

    def test_contains_user(self, sample_results, tmp_path) -> None:
        """Test summary contains user."""
        output_path = tmp_path / "summary.txt"

        save_summary_statistics(sample_results, output_path)

        content = output_path.read_text()
        assert "testuser" in content

    def test_contains_overall_stats(self, sample_results, tmp_path) -> None:
        """Test summary contains overall statistics."""
        output_path = tmp_path / "summary.txt"

        save_summary_statistics(sample_results, output_path)

        content = output_path.read_text()
        assert "Overall Performance" in content
        assert "5/100" in content

    def test_contains_task_stats(self, sample_results, tmp_path) -> None:
        """Test summary contains task statistics."""
        output_path = tmp_path / "summary.txt"

        save_summary_statistics(sample_results, output_path)

        content = output_path.read_text()
        assert "LogD" in content
        assert "10/50" in content
