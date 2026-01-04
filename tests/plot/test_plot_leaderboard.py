"""Unit tests for leaderboard plotting functions."""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from admet.plot.leaderboard import (
    plot_delta_mae_comparison,
    plot_mae_comparison_bar,
    plot_metrics_heatmap,
    plot_overall_rank_distribution,
    plot_performance_category_pie,
    plot_rank_vs_metric_scatter,
    plot_task_rankings_bar,
    save_figure_formats,
)

# Use non-interactive backend for tests
matplotlib.use("Agg")


@pytest.fixture
def sample_overall_df() -> pd.DataFrame:
    """Create sample overall leaderboard DataFrame."""
    return pd.DataFrame({"rank": [1, 2, 3, 5, 10, 15, 20, 30, 50, 100]})


@pytest.fixture
def sample_task_data() -> pd.DataFrame:
    """Create sample task data."""
    return pd.DataFrame(
        [
            {
                "task": "LogD",
                "rank": 5,
                "mae": "0.25 +/- 0.02",
                "r2": "0.85",
                "spearman r": "0.90",
                "kendall's tau": "0.75",
            },
            {
                "task": "KSOL",
                "rank": 15,
                "mae": "0.30 +/- 0.01",
                "r2": "0.80",
                "spearman r": "0.85",
                "kendall's tau": "0.70",
            },
            {
                "task": "MLM CLint",
                "rank": 25,
                "mae": "0.35 +/- 0.03",
                "r2": "0.75",
                "spearman r": "0.80",
                "kendall's tau": "0.65",
            },
        ]
    )


@pytest.fixture
def sample_task_mins() -> dict:
    """Create sample task minimum values."""
    return {"LogD": 0.20, "KSOL": 0.25, "MLM CLint": 0.30}


class TestPlotOverallRankDistribution:
    """Tests for plot_overall_rank_distribution function."""

    def test_returns_figure_and_axes(self, sample_overall_df) -> None:
        """Test function returns Figure and axes."""
        fig, axs = plot_overall_rank_distribution(sample_overall_df)

        assert isinstance(fig, plt.Figure)
        assert isinstance(axs, np.ndarray)
        assert len(axs) == 2
        plt.close(fig)

    def test_with_user_rank(self, sample_overall_df) -> None:
        """Test plot with user rank highlighted."""
        fig, axs = plot_overall_rank_distribution(sample_overall_df, user_rank=5)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_without_user_rank(self, sample_overall_df) -> None:
        """Test plot without user rank."""
        fig, axs = plot_overall_rank_distribution(sample_overall_df, user_rank=None)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)

    def test_custom_figsize(self, sample_overall_df) -> None:
        """Test with custom figure size."""
        fig, axs = plot_overall_rank_distribution(sample_overall_df, figsize=(10, 5))

        assert fig.get_figwidth() == pytest.approx(10)
        assert fig.get_figheight() == pytest.approx(5)
        plt.close(fig)

    def test_empty_dataframe(self) -> None:
        """Test with empty DataFrame."""
        df = pd.DataFrame({"rank": []})
        fig, axs = plot_overall_rank_distribution(df)

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotTaskRankingsBar:
    """Tests for plot_task_rankings_bar function."""

    def test_returns_figure_and_axes(self, sample_task_data) -> None:
        """Test function returns Figure and Axes."""
        fig, ax = plot_task_rankings_bar(sample_task_data, "testuser")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_title_includes_user(self, sample_task_data) -> None:
        """Test plot title includes username."""
        fig, ax = plot_task_rankings_bar(sample_task_data, "testuser")

        title = ax.get_title()
        assert "testuser" in title
        plt.close(fig)

    def test_bars_created(self, sample_task_data) -> None:
        """Test horizontal bars are created."""
        fig, ax = plot_task_rankings_bar(sample_task_data, "testuser")

        # Check that bars were created
        patches = ax.patches
        assert len(patches) > 0
        plt.close(fig)


class TestPlotDeltaMaeComparison:
    """Tests for plot_delta_mae_comparison function."""

    def test_returns_figure_and_axes(self, sample_task_data, sample_task_mins) -> None:
        """Test function returns Figure and Axes."""
        fig, ax = plot_delta_mae_comparison(sample_task_data, sample_task_mins, "testuser")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_with_empty_mins(self, sample_task_data) -> None:
        """Test with empty task minimums."""
        fig, ax = plot_delta_mae_comparison(sample_task_data, {}, "testuser")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotMaeComparisonBar:
    """Tests for plot_mae_comparison_bar function."""

    def test_returns_figure_and_axes(self, sample_task_data, sample_task_mins) -> None:
        """Test function returns Figure and Axes."""
        fig, ax = plot_mae_comparison_bar(sample_task_data, sample_task_mins, "testuser")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_with_no_valid_tasks(self, sample_task_data) -> None:
        """Test with no valid tasks."""
        # Empty mins dict means no valid tasks
        fig, ax = plot_mae_comparison_bar(sample_task_data, {}, "testuser")

        assert isinstance(fig, plt.Figure)
        plt.close(fig)


class TestPlotPerformanceCategoryPie:
    """Tests for plot_performance_category_pie function."""

    def test_returns_figure_and_axes(self, sample_task_data) -> None:
        """Test function returns Figure and Axes."""
        fig, ax = plot_performance_category_pie(sample_task_data)

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_pie_wedges_created(self, sample_task_data) -> None:
        """Test pie chart wedges are created."""
        fig, ax = plot_performance_category_pie(sample_task_data)

        # Pie chart should have wedges
        patches = ax.patches
        assert len(patches) > 0
        plt.close(fig)


class TestPlotMetricsHeatmap:
    """Tests for plot_metrics_heatmap function."""

    def test_returns_figure_and_axes(self, sample_task_data) -> None:
        """Test function returns Figure and axes array."""
        fig, axs = plot_metrics_heatmap(sample_task_data)

        assert isinstance(fig, plt.Figure)
        assert isinstance(axs, np.ndarray)
        assert len(axs) == 4  # 4 metrics
        plt.close(fig)


class TestPlotRankVsMetricScatter:
    """Tests for plot_rank_vs_metric_scatter function."""

    def test_returns_figure_and_axes(self, sample_task_data) -> None:
        """Test function returns Figure and Axes."""
        fig, ax = plot_rank_vs_metric_scatter(sample_task_data, "r2", "R²")

        assert isinstance(fig, plt.Figure)
        assert isinstance(ax, plt.Axes)
        plt.close(fig)

    def test_with_different_metrics(self, sample_task_data) -> None:
        """Test with different metric columns."""
        for metric, label in [("mae", "MAE"), ("spearman r", "Spearman"), ("kendall's tau", "Kendall")]:
            fig, ax = plot_rank_vs_metric_scatter(sample_task_data, metric, label)
            assert isinstance(fig, plt.Figure)
            plt.close(fig)

    def test_axis_labels(self, sample_task_data) -> None:
        """Test axis labels are set correctly."""
        fig, ax = plot_rank_vs_metric_scatter(sample_task_data, "r2", "R²")

        assert "Rank" in ax.get_xlabel()
        assert "R²" in ax.get_ylabel()
        plt.close(fig)


class TestSaveFigureFormats:
    """Tests for save_figure_formats function."""

    def test_saves_png(self, tmp_path) -> None:
        """Test saves PNG format."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        output_path = tmp_path / "test_figure"
        save_figure_formats(fig, output_path, formats=["png"])

        assert (tmp_path / "png" / "test_figure.png").exists()

    def test_saves_multiple_formats(self, tmp_path) -> None:
        """Test saves multiple formats."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        output_path = tmp_path / "test_figure"
        save_figure_formats(fig, output_path, formats=["png", "svg", "pdf"])

        assert (tmp_path / "png" / "test_figure.png").exists()
        assert (tmp_path / "svg" / "test_figure.svg").exists()
        assert (tmp_path / "pdf" / "test_figure.pdf").exists()

    def test_creates_directories(self, tmp_path) -> None:
        """Test creates format directories."""
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3])

        output_path = tmp_path / "subdir" / "test_figure"
        save_figure_formats(fig, output_path, formats=["png"])

        assert (tmp_path / "subdir" / "png" / "test_figure.png").exists()

    def test_closes_figure(self, tmp_path) -> None:
        """Test figure is closed after saving."""
        fig, ax = plt.subplots()
        fig_num = fig.number

        output_path = tmp_path / "test_figure"
        save_figure_formats(fig, output_path, formats=["png"])

        # Figure should be closed
        assert fig_num not in plt.get_fignums()
