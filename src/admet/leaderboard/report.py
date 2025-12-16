"""Report generation for leaderboard results."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class ResultsData:
    """Structured container for leaderboard analysis results.

    Attributes
    ----------
    summary_df : pd.DataFrame
        Summary DataFrame with columns: task, rank, n_rows, mae, r2, spearman, kendall
    tables : Dict[str, pd.DataFrame]
        Raw leaderboard tables by endpoint name
    task_mins : Dict[str, float]
        Minimum MAE value for each task (from rank #1)
    overall_min : Optional[float]
        Minimum MA-RAE from overall leaderboard rank #1
    target_user : str
        Username being analyzed
    timestamp : str
        ISO format timestamp of analysis
    """

    summary_df: pd.DataFrame
    tables: Dict[str, pd.DataFrame]
    task_mins: Dict[str, float]
    overall_min: Optional[float]
    target_user: str
    timestamp: str


def _calc_delta_pct(mean_val: str, min_val: str) -> str:
    """Calculate percentage delta: ((mean - min) / mean) * 100%.

    Parameters
    ----------
    mean_val : str
        Mean value string (may include "+/-" uncertainty)
    min_val : str
        Minimum value string

    Returns
    -------
    str
        Formatted delta percentage, or "N/A" if calculation fails

    Examples
    --------
    >>> _calc_delta_pct("0.35 +/- 0.01", "0.30")
    '14.3%'
    """
    try:
        # Extract value from uncertainty format
        mean_str = str(mean_val)
        if "+/-" in mean_str:
            mean_str = mean_str.split("+/-")[0].strip()
        elif "±" in mean_str:
            mean_str = mean_str.split("±")[0].strip()

        mean_num = float(mean_str)
        min_num = float(str(min_val).strip())

        if mean_num == 0:
            return "0.0%"

        delta_pct = ((mean_num - min_num) / mean_num) * 100
        return f"{delta_pct:.1f}%"
    except (ValueError, TypeError, ZeroDivisionError):
        return "N/A"


def _get_performance_note(rank_val, total: Optional[int] = None) -> str:
    """Categorize performance based on rank position.

    Parameters
    ----------
    rank_val : Any
        Rank value (int or string)
    total : Optional[int]
        Total number of submissions (for percentile calculation)

    Returns
    -------
    str
        Performance category note

    Examples
    --------
    >>> _get_performance_note(5, 100)
    'Excellent performance - Top 5.0%'
    >>> _get_performance_note(50, 100)
    'Poor performance - Top 50.0%'
    """
    try:
        rank_int = int(rank_val) if pd.notna(rank_val) else None
        if rank_int is None:
            return "Performance data unavailable"

        # Add percentile if total is provided
        pct_suffix = ""
        if total and total > 0:
            pct = round((rank_int / total) * 100, 1)
            pct_suffix = f" - Top {pct:.1f}%"

        if rank_int <= 10:
            return f"Excellent performance{pct_suffix}"
        elif rank_int <= 20:
            return f"Good performance{pct_suffix}"
        elif rank_int <= 40:
            return f"Okay performance{pct_suffix}"
        elif rank_int <= 60:
            return f"Poor performance{pct_suffix}"
        else:
            return f"Needs improvement{pct_suffix}"
    except (ValueError, TypeError):
        return "Performance data unavailable"


def generate_markdown_report(
    results: ResultsData,
    output_path: Path,
    include_figures: bool = True,
) -> None:
    """Generate comprehensive markdown report matching SUBMISSIONS.md format.

    Parameters
    ----------
    results : ResultsData
        Structured results data
    output_path : Path
        Output file path for the markdown report
    include_figures : bool, default=True
        Whether to include figure links in the report

    Examples
    --------
    >>> results = ResultsData(...)
    >>> generate_markdown_report(results, Path("report.md"))
    """
    md_lines = []

    # Header
    md_lines.append("# OpenADMET + ExpansionRx Blind Challenge Submissions\n")
    md_lines.append("* [Submission Link](https://huggingface.co/spaces/openadmet/" "OpenADMET-ExpansionRx-Challenge)\n")
    md_lines.append(f"## {datetime.now().strftime('%B %d, %Y')}\n")
    md_lines.append("### Statistics\n")
    md_lines.append("#### Overall\n")

    # Overall section
    overall_row = results.summary_df[results.summary_df["task"] == "OVERALL"]
    if not overall_row.empty:
        overall = overall_row.iloc[0]
        ma_rae = overall.get("ma-rae", overall.get("mae", "N/A"))
        r2 = overall.get("r2", "N/A")
        spearman = overall.get("spearman r", overall.get("spearman", "N/A"))
        kendall = overall.get("kendall's tau", overall.get("kendall", "N/A"))
        rank = overall["rank"]
        total = overall.get("n_rows", 0)

        # Calculate delta MA-RAE
        min_ma_rae_str = str(results.overall_min) if results.overall_min is not None else "N/A"
        delta_ma_rae = "N/A"
        if results.overall_min is not None and ma_rae != "N/A":
            delta_ma_rae = _calc_delta_pct(ma_rae, str(results.overall_min))

        note = _get_performance_note(rank, total)

        md_lines.append(
            "| Rank | User | MA-RAE | Min MA-RAE | "
            "$\\Delta$ MA-RAE to min (\\%)[^1] | R2 | Spearman R | "
            "Kendall's Tau | Submission Time | Notes |"
        )
        md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|")

        row_parts = [
            f"{rank}/{total}",
            results.target_user,
            str(ma_rae),
            min_ma_rae_str,
            delta_ma_rae,
            str(r2),
            str(spearman),
            str(kendall),
            results.timestamp,
            note,
        ]
        md_lines.append("| " + " | ".join(row_parts) + " |")

    # By Task section
    md_lines.append("\n#### By Task\n")
    md_lines.append(
        "| Rank | Task | User | MAE | Min MAE | "
        "$\\Delta$ MAE to min (\\%)[^2] | R2 | Spearman R | "
        "Kendall's Tau | Submission Time | Notes |"
    )
    md_lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|")

    for _, row in results.summary_df[results.summary_df["task"] != "OVERALL"].iterrows():
        task = row["task"]
        rank = row["rank"] if pd.notna(row["rank"]) else "—"
        mae = row.get("mae", "N/A")
        r2 = row.get("r2", "N/A")
        spearman = row.get("spearman r", row.get("spearman", "N/A"))
        kendall = row.get("kendall's tau", row.get("kendall", "N/A"))
        total = row.get("n_rows", 0)

        # Get minimum and calculate delta
        min_mae_val = results.task_mins.get(task)
        min_mae_str = str(min_mae_val) if min_mae_val is not None else "N/A"
        delta_mae = "N/A"
        if min_mae_val is not None and mae != "N/A":
            delta_mae = _calc_delta_pct(mae, str(min_mae_val))

        note = _get_performance_note(rank, total)

        row_parts = [
            str(rank),
            task,
            results.target_user,
            mae,
            min_mae_str,
            delta_mae,
            str(r2),
            str(spearman),
            str(kendall),
            results.timestamp,
            note,
        ]
        md_lines.append("| " + " | ".join(row_parts) + " |")

    # Visual Highlights section
    if include_figures:
        md_lines.append("\n### Visual Highlights\n")
        highlight_specs = [
            (
                "Overall rank distribution",
                "01_overall_rank_hist_ecdf.png",
                "Histogram + ECDF contextualizing leaderboard spread.",
            ),
            (
                "Task-specific rankings",
                "02_task_rankings_bar.png",
                "Horizontal bar chart with shaded performance zones.",
            ),
            (
                "MAE comparison",
                "04_mae_comparison_bar.png",
                "User vs. top-performer MAE with uncertainty bars.",
            ),
            (
                "Metrics heatmap",
                "06_metrics_heatmap_multi.png",
                r"Per-task heatmaps for $R^2$, Spearman R, Kendall's $\tau$, and MAE.",
            ),
        ]
        for title, filename, blurb in highlight_specs:
            rel_path = f"figures/{filename}"
            png_path = f"figures/png/{filename}"
            md_lines.append(f"- ![{title}]({png_path}) [{title}]({rel_path}): {blurb}")

        # Actionable Insights section
        md_lines.append("\n### Actionable Insights for Next Round\n")
        action_specs = [
            (
                "Priority Matrix",
                "20_priority_matrix.png",
                "Identifies quick wins (high impact, low effort) vs. major projects.",
            ),
            (
                "Gap to Leader",
                "16_gap_to_leader_waterfall.png",
                "Absolute MAE improvement needed per task to match top performer.",
            ),
            (
                "Percentile Rankings",
                "15_percentile_ranking.png",
                "Shows where you stand relative to all submissions per task.",
            ),
            (
                "Task Difficulty vs Performance",
                "19_task_difficulty_vs_performance.png",
                "Reveals if hard tasks are dragging down overall rank.",
            ),
            (
                "Rank Improvement Potential",
                "18_rank_improvement_potential.png",
                "Visualizes how much rank could improve if MAE matched leader.",
            ),
            (
                "Multi-Metric Radar",
                "14_radar_task_profile.png",
                "Spider chart showing balanced performance across metrics per task.",
            ),
        ]
        for title, filename, blurb in action_specs:
            rel_path = f"figures/{filename}"
            png_path = f"figures/png/{filename}"
            md_lines.append(f"- ![{title}]({png_path}) [{title}]({rel_path}): {blurb}")

    # Footnotes
    md_lines.append(
        "\n* [^1]: $\\Delta$ MA-RAE to min (\\%) = ((mean MA-RAE - minimum MA-RAE) / "
        "mean MA-RAE) $\\times$ 100\\%, rounded to 1 decimal place."
    )
    md_lines.append(
        "* [^2]: $\\Delta$ MAE to min (\\%) = ((mean MAE - minimum MAE) / mean MAE) "
        "$\\times$ 100\\%, rounded to 1 decimal place."
    )

    # Write to file
    md_content = "\n".join(md_lines)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        logger.info("Markdown report written to: %s", output_path)
    except Exception as e:
        logger.error("Failed to write markdown report: %s", e)
        raise


def save_csv_data(
    results: ResultsData,
    output_dir: Path,
) -> None:
    """Save leaderboard tables to CSV files.

    Creates CSV files for each endpoint in the output directory.

    Parameters
    ----------
    results : ResultsData
        Structured results data
    output_dir : Path
        Output directory for CSV files

    Examples
    --------
    >>> results = ResultsData(...)
    >>> save_csv_data(results, Path("output/csv"))
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save summary table
    summary_path = output_dir / "summary.csv"
    results.summary_df.to_csv(summary_path, index=False)
    logger.info("Summary CSV saved to: %s", summary_path)

    # Save individual endpoint tables
    for endpoint, df in results.tables.items():
        # Sanitize filename
        safe_name = endpoint.replace(" ", "_").replace("/", "_")
        csv_path = output_dir / f"{safe_name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info("Table '%s' saved to: %s", endpoint, csv_path)


def save_summary_statistics(
    results: ResultsData,
    output_path: Path,
) -> None:
    """Save summary statistics to a text file.

    Parameters
    ----------
    results : ResultsData
        Structured results data
    output_path : Path
        Output file path for summary statistics

    Examples
    --------
    >>> results = ResultsData(...)
    >>> save_summary_statistics(results, Path("summary.txt"))
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines = []
    lines.append(f"Leaderboard Analysis Summary - {results.timestamp}")
    lines.append("=" * 70)
    lines.append(f"User: {results.target_user}\n")

    # Overall statistics
    overall = results.summary_df[results.summary_df["task"] == "OVERALL"]
    if not overall.empty:
        row = overall.iloc[0]
        lines.append("Overall Performance:")
        lines.append(f"  Rank: {row['rank']}/{row.get('n_rows', 'N/A')}")
        lines.append(f"  MA-RAE: {row.get('ma-rae', row.get('mae', 'N/A'))}")
        lines.append(f"  R²: {row.get('r2', 'N/A')}")
        lines.append(f"  Spearman R: {row.get('spearman r', row.get('spearman', 'N/A'))}")
        kendall_val = row.get("kendall's tau", row.get("kendall", "N/A"))
        lines.append(f"  Kendall's Tau: {kendall_val}")
        lines.append("")

    # Task breakdown
    lines.append("Task Performance:")
    for _, row in results.summary_df[results.summary_df["task"] != "OVERALL"].iterrows():
        lines.append(f"\n  {row['task']}:")
        lines.append(f"    Rank: {row['rank']}/{row.get('n_rows', 'N/A')}")
        lines.append(f"    MAE: {row.get('mae', 'N/A')}")
        lines.append(f"    R²: {row.get('r2', 'N/A')}")

    # Write to file
    content = "\n".join(lines)
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("Summary statistics saved to: %s", output_path)
    except Exception as e:
        logger.error("Failed to write summary statistics: %s", e)
        raise
