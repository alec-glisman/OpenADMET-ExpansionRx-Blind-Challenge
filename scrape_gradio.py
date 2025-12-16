#!/usr/bin/env python3
"""
Scrape leaderboard tables from the OpenADMET ExpansionRx Challenge HF Space (Gradio UI),
and compute the row-order rank for a target user for OVERALL + each endpoint tab.

Requires:
  pip install -U gradio_client pandas matplotlib

Notes:
- OVERALL ("Average") has an explicit 'rank' column in the table; per-endpoint tabs do not.
- Per-endpoint rank is computed as (row_index + 1) in the displayed sorted table.
"""

from __future__ import annotations

import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from gradio_client import Client

plt: Any = None
sns: Any = None
np: Any = None

# Import plotting configuration from project
try:
    import colorcet as cc
    import matplotlib
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from cycler import cycler

    matplotlib.use("Agg")  # Non-interactive backend
    GLASBEY_PALETTE = list(cc.glasbey)

    try:
        import scienceplots  # noqa: F401

        plt.style.use("science")
    except Exception:
        sns.set_theme(style="whitegrid", palette=GLASBEY_PALETTE)
        try:
            plt.style.use("seaborn-v0_8")
        except Exception:
            plt.style.use("default")

    plt.rcParams["axes.prop_cycle"] = cycler(color=GLASBEY_PALETTE)
    sns.set_palette(GLASBEY_PALETTE)
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


SPACE = "openadmet/OpenADMET-ExpansionRx-Challenge"
TARGET_USER = "aglisman"  # case-insensitive match inside the displayed 'user' cell
CACHE_DIR = "assets/submissions"  # Base directory for caching leaderboards


def _to_dataframe(gradio_value: Any) -> pd.DataFrame:
    """
    Convert common Gradio table payload formats into a pandas DataFrame.

    Handles:
    - {"headers": [...], "data": [[...], ...]}
    - {"columns": [...], "data": [[...], ...]}
    - {"data": [[...], ...]} (headers inferred as col0..)
    - [[...], ...] (list of rows)
    - pandas DataFrame already
    """
    if gradio_value is None:
        return pd.DataFrame()

    if isinstance(gradio_value, pd.DataFrame):
        return gradio_value

    if isinstance(gradio_value, dict):
        headers = gradio_value.get("headers") or gradio_value.get("columns")
        data = gradio_value.get("data")

        # Some components wrap deeper; try common nests:
        if data is None and "value" in gradio_value:
            return _to_dataframe(gradio_value["value"])

        if data is None:
            # last resort: find the first list-of-lists inside
            for v in gradio_value.values():
                if isinstance(v, list) and v and isinstance(v[0], list):
                    data = v
                    break

        if data is None:
            return pd.DataFrame()

        if headers:
            return pd.DataFrame(data, columns=headers)
        return pd.DataFrame(data)

    if isinstance(gradio_value, list):
        if not gradio_value:
            return pd.DataFrame()
        if isinstance(gradio_value[0], dict):
            # list of records
            return pd.DataFrame(gradio_value)
        if isinstance(gradio_value[0], list):
            # list of rows
            return pd.DataFrame(gradio_value)

    # Unknown format
    return pd.DataFrame()


def _find_refresh_dependency(config: Dict[str, Any], n_expected_outputs: int) -> int:
    """
    Find the Gradio dependency (fn_index) that outputs all leaderboard tables in one call.
    In this Space, refresh_if_changed returns [per_ep[ep] for ep in ALL_EPS],
    where ALL_EPS = ['Average'] + ENDPOINTS. :contentReference[oaicite:4]{index=4}
    """
    deps = config.get("dependencies") or []
    candidates: List[Tuple[int, int, int]] = []  # (fn_index, n_in, n_out)

    for i, d in enumerate(deps):
        inputs = d.get("inputs") or []
        outputs = d.get("outputs") or []
        if len(outputs) == n_expected_outputs:
            candidates.append((i, len(inputs), len(outputs)))

    if not candidates:
        raise RuntimeError(
            f"Could not find any dependency with {n_expected_outputs} outputs. "
            f"Found output counts: {sorted({len((d.get('outputs') or [])) for d in deps})}"
        )

    # Prefer the one with 0 inputs (refresh_if_changed has no inputs in app.py). :contentReference[oaicite:5]{index=5}
    candidates.sort(key=lambda t: (t[1] != 0, t[0]))
    return candidates[0][0]


def _normalize_user_cell(x: Any) -> str:
    s = "" if x is None else str(x)
    return s.strip().lower()


def extract_value_uncertainty(val: Any) -> Tuple[Optional[float], Optional[float]]:
    """Parse strings like '0.35 +/- 0.01' or '0.35 ± 0.01'.

    Returns (value, uncertainty). If parsing fails, returns (None, None).
    """
    if val is None:
        return (None, None)
    s = str(val).strip()
    try:
        if "+/-" in s:
            parts = s.split("+/-")
            return (float(parts[0].strip()), float(parts[1].strip()))
        if "±" in s:
            parts = s.split("±")
            return (float(parts[0].strip()), float(parts[1].strip()))
        # plain number
        return (float(s), None)
    except Exception:
        return (None, None)


def _row_rank_for_user(df: pd.DataFrame, target_user: str) -> Optional[int]:
    """
    Return 1-based row rank for target_user based on displayed row order.
    Matches both plain 'aglisman' and markdown links like '[aglisman](...)'. :contentReference[oaicite:6]{index=6}
    """
    if df.empty:
        return None

    # Identify user column (Space uses 'user' in Leaderboard tables). :contentReference[oaicite:7]{index=7}
    user_col = None
    for c in df.columns:
        if str(c).strip().lower() == "user":
            user_col = c
            break
    if user_col is None:
        # fallback: search any column containing username
        user_col = df.columns[0]

    t = target_user.strip().lower()

    for idx, val in enumerate(df[user_col].tolist()):
        cell = _normalize_user_cell(val)
        if t in cell:
            return idx + 1
    return None


def _generate_plots(
    results_df: pd.DataFrame,
    task_mins: dict,
    overall_min: Optional[float],
    figures_path: str,
    data_path: str,
    target_user: str,
    task_labels: list,
) -> None:
    """Generate publication-quality plots for leaderboard analysis."""
    if not HAS_MATPLOTLIB:
        return

    def save_figure(fig, filename_base: str) -> None:
        for ext in ("png", "svg", "pdf"):
            ext_dir = os.path.join(figures_path, ext)
            os.makedirs(ext_dir, exist_ok=True)
            out_path = os.path.join(ext_dir, f"{filename_base}.{ext}")
            fig.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close(fig)

    # 1. Overall Ranking Distribution (Histogram + ECDF) using cached Overall.csv
    overall_csv = os.path.join(data_path, "Overall.csv")
    if os.path.exists(overall_csv):
        try:
            df_overall = pd.read_csv(overall_csv)
            # Extract rank column and drop NaNs/non-numeric
            rank_col = next((c for c in df_overall.columns if str(c).strip().lower() == "rank"), None)
            if rank_col:
                ranks_all = pd.to_numeric(df_overall[rank_col], errors="coerce").dropna().astype(int).values
                overall_row = results_df[results_df["task"] == "OVERALL"].iloc[0]
                user_rank = int(overall_row["rank"]) if pd.notna(overall_row["rank"]) else None

                fig, axs = plt.subplots(1, 2, figsize=(14, 6))
                # Histogram of ranks
                sns.histplot(ranks_all, bins=30, kde=False, ax=axs[0], color="#118AB2")
                if user_rank is not None:
                    axs[0].axvline(user_rank, color="#E63946", linewidth=2, linestyle="--", label=f"Rank = {user_rank}")
                axs[0].set_xlabel("Rank Position", fontsize=12)
                axs[0].set_ylabel("Count", fontsize=12)
                axs[0].set_title("Overall Rank Distribution", fontsize=14, fontweight="bold")
                axs[0].grid(True, alpha=0.3)
                leg0 = axs[0].legend(loc="upper right")
                if leg0:
                    leg0.set_frame_on(True)
                    leg0.get_frame().set_alpha(0.9)
                    leg0.get_frame().set_facecolor("white")
                    leg0.get_frame().set_edgecolor("gray")

                # ECDF
                x = np.sort(ranks_all)
                y = np.arange(1, len(x) + 1) / len(x)
                axs[1].plot(x, y, color="#06D6A0", linewidth=2)
                if user_rank is not None:
                    axs[1].axvline(user_rank, color="#E63946", linewidth=2, linestyle=":", label=f"Rank = {user_rank}")
                axs[1].set_xlabel("Rank Position", fontsize=12)
                axs[1].set_ylabel("ECDF", fontsize=12)
                axs[1].set_title("Overall Rank ECDF", fontsize=14, fontweight="bold")
                axs[1].grid(True, alpha=0.3)
                leg1 = axs[1].legend(loc="lower right")
                if leg1:
                    leg1.set_frame_on(True)
                    leg1.get_frame().set_alpha(0.9)
                    leg1.get_frame().set_facecolor("white")
                    leg1.get_frame().set_edgecolor("gray")

                plt.tight_layout()
                save_figure(fig, "01_overall_rank_hist_ecdf")
        except Exception:
            pass

    # 2. Task-Specific Rankings Bar Chart with Performance Zones
    task_data = results_df[results_df["task"] != "OVERALL"].copy()
    task_data = task_data.sort_values("rank")

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = [
        (
            "#06D6A0"
            if r <= 10
            else "#118AB2" if r <= 20 else "#FFD166" if r <= 40 else "#EF476F" if r <= 60 else "#E63946"
        )
        for r in task_data["rank"]
    ]
    ax.barh(
        task_data["task"],
        task_data["rank"],
        color=colors,
        alpha=0.9,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )
    ax.set_xlabel("Rank Position (lower is better)", fontsize=12)
    ax.set_ylabel("Task", fontsize=12)
    ax.set_title(
        f"Task-Specific Rankings for {target_user}",
        fontsize=14,
        fontweight="bold",
    )
    ax.invert_xaxis()

    # Add performance zone shading
    ax.axvspan(0, 10, alpha=0.1, color="#06D6A0", label="Excellent (1-10)", zorder=1)
    ax.axvspan(10, 20, alpha=0.1, color="#118AB2", label="Good (11-20)", zorder=1)
    ax.axvspan(20, 40, alpha=0.1, color="#FFD166", label="Okay (21-40)", zorder=1)
    ax.axvspan(40, 60, alpha=0.1, color="#EF476F", label="Poor (41-60)", zorder=1)
    ax.axvspan(60, 100, alpha=0.1, color="#E63946", label="Terrible (60+)", zorder=1)

    ax.grid(True, alpha=0.3, axis="x")
    leg = ax.legend(loc="lower left", fontsize=9)
    if leg:
        leg.set_frame_on(True)
        leg.get_frame().set_alpha(0.9)
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("gray")

    # Add rank labels on bars
    for i, (task, rank) in enumerate(zip(task_data["task"], task_data["rank"])):
        # Place label just to the right of the bar tip with a fixed pixel offset
        ax.annotate(
            f"{int(rank)}",
            (rank, i),
            xytext=(12, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=10,
            fontweight="bold",
            color="black",
            zorder=4,
            bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
        )

    plt.tight_layout()
    save_figure(fig, "02_task_rankings_bar")

    # 3. Delta MAE Comparison
    task_data_with_delta = task_data[task_data["task"].isin(task_mins.keys())].copy()
    deltas = []
    for _, row in task_data_with_delta.iterrows():
        task = row["task"]
        mae_str = str(row.get("mae", "N/A"))
        if "+/-" in mae_str:
            mae_val = float(mae_str.split("+/-")[0].strip())
        else:
            mae_val = float(mae_str.split("±")[0].strip()) if "±" in mae_str else float(mae_str)
        min_mae = task_mins.get(task, mae_val)
        delta_pct = ((mae_val - min_mae) / mae_val) * 100 if mae_val > 0 else 0
        deltas.append(delta_pct)

    task_data_with_delta["delta_mae_pct"] = deltas
    task_data_with_delta = task_data_with_delta.sort_values("delta_mae_pct")

    fig, ax = plt.subplots(figsize=(12, 8))
    colors_delta = [
        "#06D6A0" if d < 10 else "#FFD166" if d < 20 else "#EF476F" for d in task_data_with_delta["delta_mae_pct"]
    ]
    ax.barh(
        task_data_with_delta["task"],
        task_data_with_delta["delta_mae_pct"],
        color=colors_delta,
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
        zorder=3,
    )
    ax.set_xlabel(r"$\Delta$ MAE to Minimum (\%)", fontsize=12)
    ax.set_ylabel("Task", fontsize=12)
    ax.set_title(
        f"Performance Gap Analysis: {target_user} vs. Top Performer",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(True, alpha=0.3, axis="x")

    # Add percentage labels
    for i, (task, delta) in enumerate(zip(task_data_with_delta["task"], task_data_with_delta["delta_mae_pct"])):
        ax.annotate(
            f"{delta:.1f}%",
            (delta, i),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray", "pad": 2},
        )

    plt.tight_layout()
    save_figure(fig, "03_delta_mae_comparison")

    # 4. MAE Values with Minimum Comparison
    mae_values = []
    mae_errors = []
    min_values = []
    min_errors = []
    valid_tasks = []
    for _, row in task_data.iterrows():
        task = row["task"]
        mae_str = str(row.get("mae", "N/A"))
        if mae_str != "N/A" and task in task_mins:
            val, err = extract_value_uncertainty(mae_str)
            if val is not None:
                mae_values.append(val)
                mae_errors.append(err if err is not None else 0.0)
                # Attempt to parse error for min from top row of cached CSV
                min_val = task_mins[task]
                min_err = 0.0
                endpoint_csv = os.path.join(data_path, f"{task.replace(' ', '_').replace('>', 'gt')}.csv")
                try:
                    df_task = pd.read_csv(endpoint_csv)
                    if not df_task.empty:
                        mae_col = next(
                            (c for c in df_task.columns if str(c).strip().lower() == "mae"),
                            "mae",
                        )
                        top_mae = str(df_task.iloc[0][mae_col])
                        mv, me = extract_value_uncertainty(top_mae)
                        if me is not None:
                            min_err = me
                except Exception:
                    pass
                min_values.append(min_val)
                min_errors.append(min_err)
                valid_tasks.append(task)

    if mae_values:
        x = np.arange(len(valid_tasks))
        width = 0.35

        fig, ax = plt.subplots(figsize=(14, 8))
        bars1 = ax.bar(
            x - width / 2,
            min_values,
            width,
            label="Minimum MAE (Rank \\#1)",
            color="#118AB2",
            alpha=0.85,
            yerr=min_errors,
            capsize=4,
            zorder=3,
        )
        bars2 = ax.bar(
            x + width / 2,
            mae_values,
            width,
            label=f"{target_user} MAE",
            color="#EF476F",
            alpha=0.85,
            yerr=mae_errors,
            capsize=4,
            zorder=3,
        )

        ax.set_xlabel("Task", fontsize=12)
        ax.set_ylabel("MAE", fontsize=12)
        ax.set_title("MAE Comparison: User vs. Top Performer", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(valid_tasks, rotation=45, ha="right")
        leg = ax.legend()
        if leg:
            leg.set_frame_on(True)
            leg.get_frame().set_alpha(0.9)
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_edgecolor("gray")
        ax.grid(True, alpha=0.3, axis="y")

        # Add value labels on bars lifted above error bars with a bounding box
        for idx, bar in enumerate(bars1):
            height = bar.get_height()
            err = min_errors[idx] if idx < len(min_errors) else 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + err + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray", "pad": 2},
            )
        for idx, bar in enumerate(bars2):
            height = bar.get_height()
            err = mae_errors[idx] if idx < len(mae_errors) else 0.0
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + err + 0.02,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray", "pad": 2},
            )

        plt.tight_layout()
        save_figure(fig, "04_mae_comparison_bar")

    # 5. Performance Category Pie Chart
    performance_categories = {
        r"Excellent ($\leq$10)": 0,
        "Good (11-20)": 0,
        "Okay (21-40)": 0,
        "Poor (41-60)": 0,
        "Terrible (>60)": 0,
    }

    for rank in task_data["rank"]:
        if rank <= 10:
            performance_categories[r"Excellent ($\leq$10)"] += 1
        elif rank <= 20:
            performance_categories["Good (11-20)"] += 1
        elif rank <= 40:
            performance_categories["Okay (21-40)"] += 1
        elif rank <= 60:
            performance_categories["Poor (41-60)"] += 1
        else:
            performance_categories["Terrible (>60)"] += 1

    # Filter out zero categories
    perf_labels = [k for k, v in performance_categories.items() if v > 0]
    perf_values = [v for v in performance_categories.values() if v > 0]
    perf_colors = ["#06D6A0", "#FFD166", "#118AB2", "#EF476F", "#E63946"][: len(perf_labels)]

    fig, ax = plt.subplots(figsize=(10, 8))
    pie_segments = ax.pie(
        perf_values,
        labels=perf_labels,
        autopct="%1.1f%%",
        startangle=90,
        colors=perf_colors,
        textprops={"fontsize": 11, "fontweight": "bold"},
    )
    if len(pie_segments) == 3:
        _, label_texts, autopct_texts = pie_segments
    else:
        _, label_texts = pie_segments
        autopct_texts = []
    for text in list(label_texts) + list(autopct_texts):
        text.set_bbox({"facecolor": "white", "alpha": 0.85, "edgecolor": "gray", "pad": 2})
    ax.set_title(f"Performance Distribution Across Tasks\n{target_user}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_figure(fig, "05_performance_category_pie")

    # 6. Metrics Heatmap (separate colormap per metric)
    metrics_for_heatmap = []
    for _, row in task_data.iterrows():
        task = row["task"]
        rank = row["rank"]
        r2_str = str(row.get("r2", "N/A"))
        spearman_str = str(row.get("spearman r", "N/A"))
        kendall_str = str(row.get("kendall's tau", "N/A"))
        mae_str = str(row.get("mae", "N/A"))

        def extract_val(s):
            if s == "N/A" or s == "nan":
                return np.nan
            if "+/-" in s:
                return float(s.split("+/-")[0].strip())
            elif "±" in s:
                return float(s.split("±")[0].strip())
            return float(s)

        metrics_for_heatmap.append(
            {
                "Task": task,
                "Rank": rank,
                r"$R^2$": extract_val(r2_str),
                r"Spearman R": extract_val(spearman_str),
                r"Kendall's $\tau$": extract_val(kendall_str),
                r"MAE": extract_val(mae_str),
            }
        )

    metrics_df = pd.DataFrame(metrics_for_heatmap)
    metrics_df.set_index("Task", inplace=True)

    fig, axs = plt.subplots(1, 4, figsize=(22, 6), sharey=True)
    axs = axs.flatten()
    # Per-metric heatmaps with tailored colormaps
    cmap_specs = [
        (r"$R^2$", "viridis", 0.5),
        (r"Spearman R", "coolwarm", 0.0),
        (r"Kendall's $\\tau$", "magma", 0.0),
        (r"MAE", "cividis", None),
    ]
    for ax, (col, cmap, center) in zip(axs, cmap_specs):
        if col not in metrics_df.columns:
            ax.set_visible(False)
            continue
        sns.heatmap(
            metrics_df[[col]],
            annot=True,
            fmt=".2f",
            cmap=cmap,
            center=center,
            cbar_kws={"label": col},
            ax=ax,
            linewidths=0.5,
        )
        ax.set_title(col, fontsize=14, fontweight="bold")
        ax.set_xlabel("")
    axs[0].set_ylabel("Task", fontsize=12)
    plt.tight_layout()
    save_figure(fig, "06_metrics_heatmap_multi")

    # 7. Rank vs R² Scatter Plot
    r2_vals = []
    r2_errs = []
    rank_vals = []
    task_names = []
    for _, row in task_data.iterrows():
        r2_str = str(row.get("r2", "N/A"))
        if r2_str != "N/A" and r2_str != "nan":
            r2_val, r2_err = extract_value_uncertainty(r2_str)
            if r2_val is None:
                continue
            r2_vals.append(r2_val)
            r2_errs.append(r2_err if r2_err is not None else 0.0)
            rank_vals.append(int(row["rank"]))
            task_names.append(row["task"])

    if r2_vals:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            r2_vals,
            rank_vals,
            s=200,
            c=rank_vals,
            cmap="RdYlGn_r",
            alpha=0.8,
            edgecolors="black",
            linewidth=1.2,
            zorder=3,
        )
        # Add horizontal error bars for R^2 uncertainty
        ax.errorbar(r2_vals, rank_vals, xerr=r2_errs, fmt="none", ecolor="black", elinewidth=1.0, capsize=3, zorder=2)
        for i, task in enumerate(task_names):
            ax.annotate(
                task,
                (r2_vals[i], rank_vals[i]),
                fontsize=9,
                ha="center",
                va="bottom",
                xytext=(0, 8),
                textcoords="offset points",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray", "pad": 2},
            )
        ax.set_xlabel(r"$R^2$ Score", fontsize=12)
        ax.set_ylabel("Rank Position", fontsize=12)
        ax.set_title(
            f"Ranking vs. $R^2$ Performance\n{target_user}",
            fontsize=14,
            fontweight="bold",
        )
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Rank Position", fontsize=11)
        plt.tight_layout()
        save_figure(fig, "07_rank_vs_r2_scatter")

    # 7b. Rank vs MAE Scatter Plot
    mae_vals = []
    mae_errs = []
    rank_vals_mae = []
    task_names_mae = []
    for _, row in task_data.iterrows():
        mae_str = str(row.get("mae", "N/A"))
        if mae_str != "N/A" and mae_str != "nan":
            val, err = extract_value_uncertainty(mae_str)
            if val is None:
                continue
            mae_vals.append(val)
            mae_errs.append(err if err is not None else 0.0)
            rank_vals_mae.append(int(row["rank"]))
            task_names_mae.append(row["task"])

    if mae_vals:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            mae_vals,
            rank_vals_mae,
            s=200,
            c=rank_vals_mae,
            cmap="RdYlGn_r",
            alpha=0.8,
            edgecolors="black",
            linewidth=1.2,
            zorder=3,
        )
        ax.errorbar(
            mae_vals,
            rank_vals_mae,
            xerr=mae_errs,
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
            zorder=2,
        )
        for i, task in enumerate(task_names_mae):
            ax.annotate(
                task,
                (mae_vals[i], rank_vals_mae[i]),
                fontsize=9,
                ha="center",
                va="bottom",
                xytext=(0, 8),
                textcoords="offset points",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray", "pad": 2},
            )
        ax.set_xlabel("MAE", fontsize=12)
        ax.set_ylabel("Rank Position", fontsize=12)
        ax.set_title(f"Ranking vs. MAE Performance\n{target_user}", fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Rank Position", fontsize=11)
        plt.tight_layout()

    # 7c. Rank vs Spearman R Scatter Plot
    sp_vals = []
    sp_errs = []
    rank_vals_sp = []
    task_names_sp = []
    for _, row in task_data.iterrows():
        s_str = str(row.get("spearman r", "N/A"))
        if s_str != "N/A" and s_str != "nan":
            val, err = extract_value_uncertainty(s_str)
            if val is None:
                continue
            sp_vals.append(val)
            sp_errs.append(err if err is not None else 0.0)
            rank_vals_sp.append(int(row["rank"]))
            task_names_sp.append(row["task"])

    if sp_vals:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            sp_vals,
            rank_vals_sp,
            s=200,
            c=rank_vals_sp,
            cmap="RdYlGn_r",
            alpha=0.8,
            edgecolors="black",
            linewidth=1.2,
            zorder=3,
        )
        ax.errorbar(
            sp_vals,
            rank_vals_sp,
            xerr=sp_errs,
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
            zorder=2,
        )
        for i, task in enumerate(task_names_sp):
            ax.annotate(
                task,
                (sp_vals[i], rank_vals_sp[i]),
                fontsize=9,
                ha="center",
                va="bottom",
                xytext=(0, 8),
                textcoords="offset points",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray", "pad": 2},
            )
        ax.set_xlabel(r"Spearman R", fontsize=12)
        ax.set_ylabel("Rank Position", fontsize=12)
        ax.set_title(f"Ranking vs. Spearman R Performance\n{target_user}", fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Rank Position", fontsize=11)
        plt.tight_layout()

    # 7d. Rank vs Kendall's tau Scatter Plot
    kd_vals = []
    kd_errs = []
    rank_vals_kd = []
    task_names_kd = []
    for _, row in task_data.iterrows():
        k_str = str(row.get("kendall's tau", "N/A"))
        if k_str != "N/A" and k_str != "nan":
            val, err = extract_value_uncertainty(k_str)
            if val is None:
                continue
            kd_vals.append(val)
            kd_errs.append(err if err is not None else 0.0)
            rank_vals_kd.append(int(row["rank"]))
            task_names_kd.append(row["task"])

    if kd_vals:
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            kd_vals,
            rank_vals_kd,
            s=200,
            c=rank_vals_kd,
            cmap="RdYlGn_r",
            alpha=0.8,
            edgecolors="black",
            linewidth=1.2,
            zorder=3,
        )
        ax.errorbar(
            kd_vals,
            rank_vals_kd,
            xerr=kd_errs,
            fmt="none",
            ecolor="black",
            elinewidth=1.0,
            capsize=3,
            zorder=2,
        )
        for i, task in enumerate(task_names_kd):
            ax.annotate(
                task,
                (kd_vals[i], rank_vals_kd[i]),
                fontsize=9,
                ha="center",
                va="bottom",
                xytext=(0, 8),
                textcoords="offset points",
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray", "pad": 2},
            )
        ax.set_xlabel(r"Kendall's $\tau$", fontsize=12)
        ax.set_ylabel("Rank Position", fontsize=12)
        ax.set_title(f"Ranking vs. Kendall's $\tau$ Performance\n{target_user}", fontsize=14, fontweight="bold")
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3)
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label("Rank Position", fontsize=11)
        plt.tight_layout()

    # 8. MAE Distributions with KDE for Overall (MA-RAE) and all endpoints
    # Build subplots grid: 2x5 for 10 tables (Overall + 9 endpoints)
    label_files = [
        ("Overall (MA-RAE)", os.path.join(data_path, "Overall.csv"), "ma-rae"),
        ("LogD", os.path.join(data_path, "LogD.csv"), "mae"),
        ("KSOL", os.path.join(data_path, "KSOL.csv"), "mae"),
        ("MLM CLint", os.path.join(data_path, "MLM_CLint.csv"), "mae"),
        ("HLM CLint", os.path.join(data_path, "HLM_CLint.csv"), "mae"),
        ("Caco-2 Permeability Efflux", os.path.join(data_path, "Caco-2_Permeability_Efflux.csv"), "mae"),
        ("Caco-2 Permeability Papp A>B", os.path.join(data_path, "Caco-2_Permeability_Papp_AgtB.csv"), "mae"),
        ("MPPB", os.path.join(data_path, "MPPB.csv"), "mae"),
        ("MBPB", os.path.join(data_path, "MBPB.csv"), "mae"),
        ("MGMB", os.path.join(data_path, "MGMB.csv"), "mae"),
    ]

    fig, axs = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
    axs = axs.flatten()
    for ax, (label, path, colname) in zip(axs, label_files):
        try:
            df_ep = pd.read_csv(path)
            col = next((c for c in df_ep.columns if str(c).strip().lower() == colname), None)
            if not col:
                ax.set_visible(False)
                continue
            # Parse values for histogram
            vals = []
            for v in df_ep[col].astype(str):
                val, _ = extract_value_uncertainty(v)
                if val is not None:
                    vals.append(val)
            if not vals:
                ax.set_visible(False)
                continue
            sns.histplot(
                vals,
                bins=60,
                kde=True,
                ax=ax,
                color="#118AB2",
            )
            # Add vertical line for user's submission
            user_row = results_df[
                results_df["task"].str.lower() == (label if label != "Overall (MA-RAE)" else "overall").lower()
            ]
            if not user_row.empty:
                user_val_raw = user_row.iloc[0].get(colname, None)
                uv, ue = extract_value_uncertainty(user_val_raw)
                if uv is not None:
                    ax.axvline(uv, color="#E63946", linewidth=2, linestyle="--")
                    if ue is not None and ue > 0:
                        ax.axvspan(max(0, uv - ue), uv + ue, color="#E63946", alpha=0.15)
            ax.set_title(label, fontsize=12, fontweight="bold")
            ax.grid(True, alpha=0.3)
        except Exception:
            ax.set_visible(False)
    save_figure(fig, "11_mae_distributions_kde")

    # 9. Spearman R distributions with KDE (Overall + endpoints)
    try:
        spearman_files = [
            ("Overall (Spearman R)", os.path.join(data_path, "Overall.csv"), "spearman r"),
            ("LogD", os.path.join(data_path, "LogD.csv"), "spearman r"),
            ("KSOL", os.path.join(data_path, "KSOL.csv"), "spearman r"),
            ("MLM CLint", os.path.join(data_path, "MLM_CLint.csv"), "spearman r"),
            ("HLM CLint", os.path.join(data_path, "HLM_CLint.csv"), "spearman r"),
            ("Caco-2 Permeability Efflux", os.path.join(data_path, "Caco-2_Permeability_Efflux.csv"), "spearman r"),
            (
                "Caco-2 Permeability Papp A>B",
                os.path.join(data_path, "Caco-2_Permeability_Papp_AgtB.csv"),
                "spearman r",
            ),
            ("MPPB", os.path.join(data_path, "MPPB.csv"), "spearman r"),
            ("MBPB", os.path.join(data_path, "MBPB.csv"), "spearman r"),
            ("MGMB", os.path.join(data_path, "MGMB.csv"), "spearman r"),
        ]

        fig, axs = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
        axs = axs.flatten()
        for ax, (label, path, target_lower) in zip(axs, spearman_files):
            try:
                df_ep = pd.read_csv(path)
                # find column by lower-case match
                col = next((c for c in df_ep.columns if str(c).strip().lower() == target_lower), None)
                if not col:
                    ax.set_visible(False)
                    continue
                vals = []
                for v in df_ep[col].astype(str):
                    val, _ = extract_value_uncertainty(v)
                    if val is not None:
                        vals.append(val)
                if not vals:
                    ax.set_visible(False)
                    continue
                sns.histplot(
                    vals,
                    bins=60,
                    kde=True,
                    ax=ax,
                    color="#06D6A0",
                )
                # user marker
                key = label if label != "Overall (Spearman R)" else "OVERALL"
                user_row = results_df[results_df["task"].str.lower() == key.lower()]
                if not user_row.empty:
                    user_val_raw = user_row.iloc[0].get("spearman r", None)
                    uv, ue = extract_value_uncertainty(user_val_raw)
                    if uv is not None:
                        ax.axvline(uv, color="#E63946", linewidth=2, linestyle="--")
                        if ue is not None and ue > 0:
                            ax.axvspan(max(0, uv - ue), uv + ue, color="#E63946", alpha=0.15)
                ax.set_title(label, fontsize=12, fontweight="bold")
                ax.grid(True, alpha=0.3)
            except Exception:
                ax.set_visible(False)
        save_figure(fig, "12_spearman_distributions_kde")
    except Exception:
        pass

    # 10. Kendall's tau distributions with KDE (Overall + endpoints)
    try:
        kendall_files = [
            ("Overall (Kendall's tau)", os.path.join(data_path, "Overall.csv"), "kendall's tau"),
            ("LogD", os.path.join(data_path, "LogD.csv"), "kendall's tau"),
            ("KSOL", os.path.join(data_path, "KSOL.csv"), "kendall's tau"),
            ("MLM CLint", os.path.join(data_path, "MLM_CLint.csv"), "kendall's tau"),
            ("HLM CLint", os.path.join(data_path, "HLM_CLint.csv"), "kendall's tau"),
            ("Caco-2 Permeability Efflux", os.path.join(data_path, "Caco-2_Permeability_Efflux.csv"), "kendall's tau"),
            (
                "Caco-2 Permeability Papp A>B",
                os.path.join(data_path, "Caco-2_Permeability_Papp_AgtB.csv"),
                "kendall's tau",
            ),
            ("MPPB", os.path.join(data_path, "MPPB.csv"), "kendall's tau"),
            ("MBPB", os.path.join(data_path, "MBPB.csv"), "kendall's tau"),
            ("MGMB", os.path.join(data_path, "MGMB.csv"), "kendall's tau"),
        ]

        fig, axs = plt.subplots(2, 5, figsize=(20, 8), constrained_layout=True)
        axs = axs.flatten()
        for ax, (label, path, target_lower) in zip(axs, kendall_files):
            try:
                df_ep = pd.read_csv(path)
                col = next((c for c in df_ep.columns if str(c).strip().lower() == target_lower), None)
                if not col:
                    ax.set_visible(False)
                    continue
                vals = []
                for v in df_ep[col].astype(str):
                    val, _ = extract_value_uncertainty(v)
                    if val is not None:
                        vals.append(val)
                if not vals:
                    ax.set_visible(False)
                    continue
                sns.histplot(
                    vals,
                    bins=60,
                    kde=True,
                    ax=ax,
                    color="#FFD166",
                )
                key = label if label != "Overall (Kendall's tau)" else "OVERALL"
                user_row = results_df[results_df["task"].str.lower() == key.lower()]
                if not user_row.empty:
                    user_val_raw = user_row.iloc[0].get("kendall's tau", None)
                    uv, ue = extract_value_uncertainty(user_val_raw)
                    if uv is not None:
                        ax.axvline(uv, color="#E63946", linewidth=2, linestyle="--")
                        if ue is not None and ue > 0:
                            ax.axvspan(max(0, uv - ue), uv + ue, color="#E63946", alpha=0.15)
                ax.set_title(label, fontsize=12, fontweight="bold")
                ax.grid(True, alpha=0.3)
            except Exception:
                ax.set_visible(False)
        save_figure(fig, "13_kendall_distributions_kde")
    except Exception:
        pass

    # 14. Radar/Spider Chart - Multi-metric profile per task
    try:
        radar_tasks = task_data["task"].tolist()
        metrics_radar = {"Rank (inverted)": [], r"$R^2$": [], "Spearman R": [], "MAE (inverted)": []}
        max_rank = 100  # normalize ranks
        for _, row in task_data.iterrows():
            rank = row["rank"]
            metrics_radar["Rank (inverted)"].append(1 - rank / max_rank)
            r2_val, _ = extract_value_uncertainty(row.get("r2", None))
            metrics_radar[r"$R^2$"].append(r2_val if r2_val is not None else 0)
            sp_val, _ = extract_value_uncertainty(row.get("spearman r", None))
            metrics_radar["Spearman R"].append(sp_val if sp_val is not None else 0)
            mae_val, _ = extract_value_uncertainty(row.get("mae", None))
            metrics_radar["MAE (inverted)"].append(1 - mae_val if mae_val is not None else 0)

        categories = list(metrics_radar.keys())
        n_cats = len(categories)
        angles = [n / float(n_cats) * 2 * np.pi for n in range(n_cats)]
        angles += angles[:1]

        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={"polar": True})
        for i, task in enumerate(radar_tasks):
            values = [metrics_radar[cat][i] for cat in categories]
            values += values[:1]
            ax.plot(angles, values, linewidth=2, linestyle="solid", label=task)
            ax.fill(angles, values, alpha=0.1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories, fontsize=11)
        ax.set_title(f"Multi-Metric Profile by Task\n{target_user}", fontsize=14, fontweight="bold")
        leg = ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.0), fontsize=9)
        if leg:
            leg.set_frame_on(True)
            leg.get_frame().set_facecolor("white")
            leg.get_frame().set_edgecolor("gray")
        save_figure(fig, "14_radar_task_profile")
    except Exception:
        pass

    # 15. Percentile Ranking Analysis - Where does user fall in distribution?
    try:
        percentile_data = []
        for _, row in task_data.iterrows():
            task = row["task"]
            csv_path = os.path.join(data_path, f"{task.replace(' ', '_').replace('>', 'gt')}.csv")
            if os.path.exists(csv_path):
                df_task = pd.read_csv(csv_path)
                n_total = len(df_task)
                user_rank = row["rank"]
                percentile = ((n_total - user_rank + 1) / n_total) * 100 if n_total > 0 else 0
                percentile_data.append({"Task": task, "Rank": user_rank, "Total": n_total, "Percentile": percentile})
        if percentile_data:
            pct_df = pd.DataFrame(percentile_data).sort_values("Percentile", ascending=False)

            fig, ax = plt.subplots(figsize=(12, 8))
            colors = [
                "#06D6A0" if p >= 90 else "#118AB2" if p >= 75 else "#FFD166" if p >= 50 else "#EF476F"
                for p in pct_df["Percentile"]
            ]
            ax.barh(pct_df["Task"], pct_df["Percentile"], color=colors, edgecolor="black", linewidth=0.5)
            ax.axvline(50, color="gray", linestyle="--", linewidth=1.5, label="Median")
            ax.axvline(75, color="#118AB2", linestyle=":", linewidth=1.5, label="75th percentile")
            ax.axvline(90, color="#06D6A0", linestyle=":", linewidth=1.5, label="90th percentile")
            ax.set_xlabel("Percentile Rank (higher is better)", fontsize=12)
            ax.set_ylabel("Task", fontsize=12)
            ax.set_title(f"Percentile Ranking by Task\n{target_user}", fontsize=14, fontweight="bold")
            ax.set_xlim(0, 100)
            ax.grid(True, alpha=0.3, axis="x")
            for i, (task, pct) in enumerate(zip(pct_df["Task"], pct_df["Percentile"])):
                ax.annotate(
                    f"{pct:.1f}%",
                    (pct, i),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=10,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
                )
            leg = ax.legend(loc="lower right")
            if leg:
                leg.set_frame_on(True)
                leg.get_frame().set_facecolor("white")
                leg.get_frame().set_edgecolor("gray")
            plt.tight_layout()
            save_figure(fig, "15_percentile_ranking")
            pct_df.to_csv(os.path.join(data_path, "percentile_rankings.csv"), index=False)
    except Exception:
        pass

    # 16. Gap-to-Leader Waterfall - Improvement needed per task
    try:
        gap_data = []
        for _, row in task_data.iterrows():
            task = row["task"]
            mae_val, _ = extract_value_uncertainty(row.get("mae", None))
            min_mae = task_mins.get(task, None)
            if mae_val is not None and min_mae is not None:
                gap = mae_val - min_mae
                gap_data.append({"Task": task, "UserMAE": mae_val, "MinMAE": min_mae, "Gap": gap})
        if gap_data:
            gap_df = pd.DataFrame(gap_data).sort_values("Gap", ascending=False)

            fig, ax = plt.subplots(figsize=(12, 8))
            colors = ["#E63946" if g > 0.05 else "#FFD166" if g > 0.02 else "#06D6A0" for g in gap_df["Gap"]]
            ax.barh(gap_df["Task"], gap_df["Gap"], color=colors, edgecolor="black", linewidth=0.5)
            ax.set_xlabel("MAE Gap to Leader (absolute)", fontsize=12)
            ax.set_ylabel("Task", fontsize=12)
            ax.set_title(
                f"Improvement Opportunity: Gap to Leader\n{target_user}",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3, axis="x")
            for i, (task, gap) in enumerate(zip(gap_df["Task"], gap_df["Gap"])):
                ax.annotate(
                    f"{gap:.3f}",
                    (gap, i),
                    xytext=(5, 0),
                    textcoords="offset points",
                    va="center",
                    fontsize=10,
                    bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2},
                )
            plt.tight_layout()
            save_figure(fig, "16_gap_to_leader_waterfall")
            gap_df.to_csv(os.path.join(data_path, "gap_to_leader.csv"), index=False)
    except Exception:
        pass

    # 17. Metric Correlation Matrix - Understanding metric relationships
    try:
        corr_data = []
        for _, row in task_data.iterrows():
            r2_val, _ = extract_value_uncertainty(row.get("r2", None))
            sp_val, _ = extract_value_uncertainty(row.get("spearman r", None))
            kd_val, _ = extract_value_uncertainty(row.get("kendall's tau", None))
            mae_val, _ = extract_value_uncertainty(row.get("mae", None))
            corr_data.append(
                {
                    "Rank": row["rank"],
                    r"$R^2$": r2_val,
                    "Spearman R": sp_val,
                    r"Kendall's $\tau$": kd_val,
                    "MAE": mae_val,
                }
            )
        corr_df = pd.DataFrame(corr_data).dropna()
        if len(corr_df) >= 3:
            corr_matrix = corr_df.corr()
            fig, ax = plt.subplots(figsize=(10, 8))
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
            sns.heatmap(
                corr_matrix,
                mask=mask,
                annot=True,
                fmt=".2f",
                cmap="coolwarm",
                center=0,
                ax=ax,
                linewidths=0.5,
                cbar_kws={"label": "Correlation"},
            )
            ax.set_title(
                f"Metric Correlation Matrix\n{target_user}",
                fontsize=14,
                fontweight="bold",
            )
            plt.tight_layout()
            save_figure(fig, "17_metric_correlation_matrix")
    except Exception:
        pass

    # 18. Rank Improvement Potential - Estimated rank if MAE matched leader
    try:
        improvement_data = []
        for _, row in task_data.iterrows():
            task = row["task"]
            current_rank = row["rank"]
            csv_path = os.path.join(data_path, f"{task.replace(' ', '_').replace('>', 'gt')}.csv")
            if os.path.exists(csv_path):
                df_task = pd.read_csv(csv_path)
                n_total = len(df_task)
                # If you matched the leader's MAE, you'd be ~rank 1-5
                potential_rank = min(5, n_total)
                rank_improvement = current_rank - potential_rank
                improvement_data.append(
                    {
                        "Task": task,
                        "CurrentRank": current_rank,
                        "PotentialRank": potential_rank,
                        "RankImprovement": rank_improvement,
                        "TotalSubmissions": n_total,
                    }
                )
        if improvement_data:
            imp_df = pd.DataFrame(improvement_data).sort_values("RankImprovement", ascending=False)

            fig, ax = plt.subplots(figsize=(12, 8))
            ax.barh(imp_df["Task"], imp_df["CurrentRank"], color="#EF476F", alpha=0.7, label="Current Rank")
            ax.barh(
                imp_df["Task"],
                imp_df["PotentialRank"],
                color="#06D6A0",
                alpha=0.9,
                label="Potential Rank (if top MAE)",
            )
            ax.set_xlabel("Rank Position", fontsize=12)
            ax.set_ylabel("Task", fontsize=12)
            ax.set_title(
                f"Rank Improvement Potential\n{target_user}",
                fontsize=14,
                fontweight="bold",
            )
            ax.invert_xaxis()
            ax.grid(True, alpha=0.3, axis="x")
            leg = ax.legend(loc="lower left")
            if leg:
                leg.set_frame_on(True)
                leg.get_frame().set_facecolor("white")
                leg.get_frame().set_edgecolor("gray")
            plt.tight_layout()
            save_figure(fig, "18_rank_improvement_potential")
            imp_df.to_csv(os.path.join(data_path, "rank_improvement_potential.csv"), index=False)
    except Exception:
        pass

    # 19. Task Difficulty vs Performance - Are hard tasks dragging you down?
    try:
        difficulty_data = []
        for _, row in task_data.iterrows():
            task = row["task"]
            user_rank = row["rank"]
            min_mae = task_mins.get(task, None)
            csv_path = os.path.join(data_path, f"{task.replace(' ', '_').replace('>', 'gt')}.csv")
            if os.path.exists(csv_path) and min_mae is not None:
                df_task = pd.read_csv(csv_path)
                mae_col = next((c for c in df_task.columns if str(c).strip().lower() == "mae"), None)
                if mae_col:
                    mae_vals_list = []
                    for v in df_task[mae_col].astype(str):
                        val, _ = extract_value_uncertainty(v)
                        if val is not None:
                            mae_vals_list.append(val)
                    if mae_vals_list:
                        median_mae = np.median(mae_vals_list)
                        std_mae = np.std(mae_vals_list)
                        difficulty_data.append(
                            {
                                "Task": task,
                                "UserRank": user_rank,
                                "MinMAE": min_mae,
                                "MedianMAE": median_mae,
                                "StdMAE": std_mae,
                                "Difficulty": median_mae,  # Higher median = harder task
                            }
                        )
        if difficulty_data:
            diff_df = pd.DataFrame(difficulty_data)

            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(
                diff_df["Difficulty"],
                diff_df["UserRank"],
                s=200,
                c=diff_df["UserRank"],
                cmap="RdYlGn_r",
                edgecolors="black",
                linewidth=1.2,
                alpha=0.8,
            )
            for i, task in enumerate(diff_df["Task"]):
                ax.annotate(
                    task,
                    (diff_df["Difficulty"].iloc[i], diff_df["UserRank"].iloc[i]),
                    xytext=(8, 8),
                    textcoords="offset points",
                    fontsize=9,
                    bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray", "pad": 2},
                )
            ax.set_xlabel("Task Difficulty (Median MAE across all submissions)", fontsize=12)
            ax.set_ylabel("Your Rank Position", fontsize=12)
            ax.set_title(
                f"Task Difficulty vs. Your Performance\n{target_user}",
                fontsize=14,
                fontweight="bold",
            )
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Rank Position", fontsize=11)
            plt.tight_layout()
            save_figure(fig, "19_task_difficulty_vs_performance")
            diff_df.to_csv(os.path.join(data_path, "task_difficulty_analysis.csv"), index=False)
    except Exception:
        pass

    # 20. Priority Matrix - Combine impact (rank gap) with effort (MAE gap)
    try:
        priority_data = []
        for _, row in task_data.iterrows():
            task = row["task"]
            mae_val, _ = extract_value_uncertainty(row.get("mae", None))
            min_mae = task_mins.get(task, None)
            user_rank = row["rank"]
            csv_path = os.path.join(data_path, f"{task.replace(' ', '_').replace('>', 'gt')}.csv")
            if mae_val is not None and min_mae is not None and os.path.exists(csv_path):
                df_task = pd.read_csv(csv_path)
                n_total = len(df_task)
                mae_gap = mae_val - min_mae
                rank_gap = user_rank - 1  # gap to #1
                priority_data.append(
                    {
                        "Task": task,
                        "MAE_Gap": mae_gap,
                        "Rank_Gap": rank_gap,
                        "Impact": (rank_gap / n_total) * 100 if n_total > 0 else 0,
                    }
                )
        if priority_data:
            pri_df = pd.DataFrame(priority_data)

            fig, ax = plt.subplots(figsize=(14, 11))
            scatter = ax.scatter(
                pri_df["MAE_Gap"],
                pri_df["Rank_Gap"],
                s=400,
                c=pri_df["Impact"],
                cmap="YlOrRd",
                edgecolors="black",
                linewidth=1.5,
                alpha=0.85,
                zorder=3,
            )
            # Quadrant lines
            ax.axhline(pri_df["Rank_Gap"].median(), color="gray", linestyle="--", alpha=0.5, zorder=1)
            ax.axvline(pri_df["MAE_Gap"].median(), color="gray", linestyle="--", alpha=0.5, zorder=1)
            # Labels with collision avoidance (offset based on quadrant)
            for i, task in enumerate(pri_df["Task"]):
                x = pri_df["MAE_Gap"].iloc[i]
                y = pri_df["Rank_Gap"].iloc[i]
                # Determine quadrant and offset accordingly
                x_med = pri_df["MAE_Gap"].median()
                y_med = pri_df["Rank_Gap"].median()
                if x < x_med and y > y_med:  # Top-left (quick wins)
                    offset = (-15, 15)
                elif x >= x_med and y > y_med:  # Top-right (major projects)
                    offset = (15, 15)
                elif x < x_med and y <= y_med:  # Bottom-left (fill-ins)
                    offset = (-15, -15)
                else:  # Bottom-right (low priority)
                    offset = (15, -15)
                ax.annotate(
                    task,
                    (x, y),
                    xytext=offset,
                    textcoords="offset points",
                    fontsize=9,
                    fontweight="bold",
                    bbox={"facecolor": "white", "alpha": 0.95, "edgecolor": "gray", "pad": 3},
                    arrowprops={"arrowstyle": "->", "connectionstyle": "arc3,rad=0.2", "lw": 0.5},
                    zorder=4,
                )
            ax.set_xlabel("MAE Gap to Leader (Effort Required)", fontsize=12)
            ax.set_ylabel("Rank Gap to Leader (Impact)", fontsize=12)
            ax.set_title(
                f"Priority Matrix: Impact vs. Effort\n{target_user}\n"
                "(Top-left = Quick Wins, Bottom-right = Low Priority)",
                fontsize=14,
                fontweight="bold",
            )
            ax.grid(True, alpha=0.3, zorder=0)
            # Quadrant annotations (moved to edges to avoid collision)
            ax.text(
                0.02,
                0.98,
                "QUICK WINS",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                va="top",
                ha="left",
                bbox={"facecolor": "#06D6A0", "alpha": 0.4, "edgecolor": "#06D6A0", "pad": 6},
            )
            ax.text(
                0.98,
                0.98,
                "MAJOR PROJECTS",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                va="top",
                ha="right",
                bbox={"facecolor": "#FFD166", "alpha": 0.4, "edgecolor": "#FFD166", "pad": 6},
            )
            ax.text(
                0.02,
                0.02,
                "FILL-INS",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                va="bottom",
                ha="left",
                bbox={"facecolor": "#118AB2", "alpha": 0.4, "edgecolor": "#118AB2", "pad": 6},
            )
            ax.text(
                0.98,
                0.02,
                "LOW PRIORITY",
                transform=ax.transAxes,
                fontsize=11,
                fontweight="bold",
                va="bottom",
                ha="right",
                bbox={"facecolor": "#EF476F", "alpha": 0.4, "edgecolor": "#EF476F", "pad": 6},
            )
            cbar = plt.colorbar(scatter, ax=ax)
            cbar.set_label("Relative Impact (%)", fontsize=11)
            plt.tight_layout()
            save_figure(fig, "20_priority_matrix")
            pri_df.to_csv(os.path.join(data_path, "priority_matrix.csv"), index=False)
    except Exception:
        pass

    # 9. Summary statistics table (CSV) for user vs minima
    try:
        summary_rows = []
        # Overall
        overall_row = results_df[results_df["task"] == "OVERALL"]
        if not overall_row.empty:
            o = overall_row.iloc[0]
            val, err = extract_value_uncertainty(o.get("ma-rae", None))
            min_val = overall_min if overall_min is not None else None
            delta_pct = None
            if val is not None and min_val is not None and val > 0:
                delta_pct = ((val - min_val) / val) * 100.0
            summary_rows.append(
                {
                    "Task": "OVERALL",
                    "Rank": o.get("rank", None),
                    "Value": val,
                    "Uncertainty": err,
                    "Minimum": min_val,
                    "DeltaToMinPct": delta_pct,
                    "R2": o.get("r2", None),
                    "SpearmanR": o.get("spearman r", None),
                    "KendallsTau": o.get("kendall's tau", None),
                }
            )
        # Endpoints
        for _, row in results_df[results_df["task"] != "OVERALL"].iterrows():
            task = row["task"]
            val, err = extract_value_uncertainty(row.get("mae", None))
            min_val = task_mins.get(task, None)
            delta_pct = None
            if val is not None and min_val is not None and val > 0:
                delta_pct = ((val - min_val) / val) * 100.0
            summary_rows.append(
                {
                    "Task": task,
                    "Rank": row.get("rank", None),
                    "Value": val,
                    "Uncertainty": err,
                    "Minimum": min_val,
                    "DeltaToMinPct": delta_pct,
                    "R2": row.get("r2", None),
                    "SpearmanR": row.get("spearman r", None),
                    "KendallsTau": row.get("kendall's tau", None),
                }
            )
        summary_df = pd.DataFrame(summary_rows)
        summary_df.to_csv(os.path.join(data_path, "summary_statistics.csv"), index=False)
    except Exception:
        pass

    print(f"Generated publication-quality plots in {figures_path}")


def main() -> None:
    client = Client(SPACE)
    config = client.config  # fetched from the Space runtime

    # Setup caching directory with timestamp (year-month-day format)
    now = datetime.now()
    cache_subdir = now.strftime("%Y-%m-%d")
    cache_path = os.path.join(CACHE_DIR, cache_subdir)
    data_path = os.path.join(cache_path, "data")
    figures_path = os.path.join(cache_path, "figures")
    os.makedirs(cache_path, exist_ok=True)
    os.makedirs(data_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)

    # The Space defines ALL_EPS = ['Average'] + ENDPOINTS (9 endpoints => 10 tables)
    n_expected_outputs = 10
    fn_index = _find_refresh_dependency(config, n_expected_outputs)

    # Call the refresh function to get all current leaderboard tables
    try:
        outputs = client.predict(fn_index=fn_index, api_name=None, data=[])
    except TypeError:
        # older gradio_client signature
        outputs = client.predict(fn_index=fn_index)

    # Convert tuple to list if needed
    if isinstance(outputs, tuple):
        outputs = list(outputs)

    if not isinstance(outputs, list) or len(outputs) != n_expected_outputs:
        raise RuntimeError(f"Unexpected outputs: type={type(outputs)} " f"len={getattr(outputs, '__len__', None)}")

    # Labels in the same order the app returns them: ['Average'] + ENDPOINTS. :contentReference[oaicite:9]{index=9}
    labels = [
        "Average",
        "LogD",
        "KSOL",
        "MLM CLint",
        "HLM CLint",
        "Caco-2 Permeability Efflux",
        "Caco-2 Permeability Papp A>B",
        "MPPB",
        "MBPB",
        "MGMB",
    ]

    results = []
    task_mins = {}  # Store minimum MAE for each task from rank #1
    overall_min_ma_rae = None

    for label, table_val in zip(labels, outputs):
        df = _to_dataframe(table_val)
        rank = _row_rank_for_user(df, TARGET_USER)

        # Cache the full leaderboard table to data/ subdirectory
        table_label = "Overall" if label == "Average" else label
        table_filename = f"{table_label.replace(' ', '_').replace('>', 'gt')}.csv"
        table_path = os.path.join(data_path, table_filename)
        try:
            df.to_csv(table_path, index=False)
            print(f"Cached {label} leaderboard to {table_path}")
        except Exception as e:
            print(f"Warning: Could not cache {label} table: {e}")

        # Extract minimum value from top-ranked entry (rank #1)
        if not df.empty:
            top_row = df.iloc[0]
            print(f"Debug: Processing {label}, columns: {df.columns.tolist()}")
            if label == "Average":
                # Extract MA-RAE for overall (column is "MA-RAE")
                ma_rae_col = next(
                    (c for c in df.columns if str(c).strip().lower() in ["ma-rae"]),
                    None,
                )
                if ma_rae_col is not None:
                    raw_val = ""
                    try:
                        raw_val = str(top_row[ma_rae_col])
                        # Handle both +/- and ± formats
                        ma_rae_str = raw_val.split("+/-")[0].split("±")[0].strip()
                        overall_min_ma_rae = float(ma_rae_str)
                        print(f"  -> Extracted overall min MA-RAE: {overall_min_ma_rae}")
                    except (ValueError, TypeError) as e:
                        print(f"  -> Failed to extract MA-RAE: {e}, raw='{raw_val}'")
            else:
                # Extract MAE for tasks (column is "MAE")
                mae_col = next((c for c in df.columns if str(c).strip().lower() == "mae"), None)
                if mae_col is not None:
                    raw_val = ""
                    try:
                        raw_val = str(top_row[mae_col])
                        # Handle both +/- and ± formats
                        mae_str = raw_val.split("+/-")[0].split("±")[0].strip()
                        task_mins[label] = float(mae_str)
                        print(f"  -> Extracted {label} min MAE: {task_mins[label]}")
                    except (ValueError, TypeError) as e:
                        print(f"  -> Failed to extract MAE for {label}: {e}, raw='{raw_val}'")

        # Extract user row metrics
        user_metrics = {}
        if not df.empty:
            # Find user column
            user_col = next((c for c in df.columns if str(c).strip().lower() == "user"), None)
            if user_col is not None:
                mask = df[user_col].astype(str).str.lower().str.contains(TARGET_USER.lower(), na=False)
                if mask.any():
                    user_row = df[mask].iloc[0]

                    # Extract all metrics from the row
                    for col in df.columns:
                        col_lower = str(col).strip().lower()
                        if col_lower not in ["user", "rank"]:
                            user_metrics[col_lower] = user_row[col]

                    # For OVERALL, try to get explicit rank
                    rank_col = next((c for c in df.columns if str(c).strip().lower() == "rank"), None)
                    if rank_col is not None:
                        try:
                            explicit_rank = int(pd.to_numeric(user_row[rank_col], errors="coerce"))
                            rank = explicit_rank
                        except (ValueError, TypeError):
                            pass

        results.append(
            {
                "task": "OVERALL" if label == "Average" else label,
                "rank": rank,
                "n_rows": int(len(df)) if isinstance(df, pd.DataFrame) else None,
                **user_metrics,
            }
        )

    out = pd.DataFrame(results)

    # Print table output
    print("\n=== Raw Data ===")
    print(out.to_string(index=False))

    # Generate comprehensive markdown report matching SUBMISSIONS.md format

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S+00:00")

    # Helper to calculate delta percentage
    def calc_delta_pct(mean_val: str, min_val: str) -> str:
        """Calculate (mean - min) / mean * 100%"""
        try:
            # Handle both +/- and ± formats
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

    # Minimum values already extracted during results processing
    # task_mins: dict mapping task label to minimum MAE from rank #1
    # overall_min_ma_rae: minimum MA-RAE from overall leaderboard rank #1

    # Build comprehensive markdown content
    md_lines = []
    md_lines.append("# OpenADMET + ExpansionRx Blind Challenge Submissions\n")
    md_lines.append("* [Submission Link](https://huggingface.co/spaces/openadmet/OpenADMET-ExpansionRx-Challenge)\n")
    md_lines.append(f"## {datetime.now().strftime('%B %d, %Y')}\n")
    md_lines.append("### Statistics\n")
    md_lines.append("#### Overall\n")

    overall_row = out[out["task"] == "OVERALL"]
    if not overall_row.empty:
        overall = overall_row.iloc[0]
        ma_rae = overall.get("ma-rae", overall.get("mae", "N/A"))
        r2 = overall.get("r2", "N/A")
        spearman = overall.get("spearman r", overall.get("spearman", "N/A"))
        kendall = overall.get("kendall's tau", overall.get("kendall", "N/A"))
        rank = overall["rank"]
        total = overall.get("n_rows", 0)

        # Calculate percentile and note
        try:
            rank_int = int(rank)
            pct = round((rank_int / total) * 100, 1) if total > 0 else 0
            note = f"Top {pct:.1f}% overall"
        except (ValueError, TypeError):
            note = "Performance data unavailable"

        # Calculate delta MA-RAE if minimum is available
        delta_ma_rae = "N/A"
        min_ma_rae_str = str(overall_min_ma_rae) if overall_min_ma_rae is not None else "N/A"
        if overall_min_ma_rae is not None and ma_rae != "N/A":
            delta_ma_rae = calc_delta_pct(ma_rae, str(overall_min_ma_rae))

        md_lines.append(
            "| Rank | User | MA-RAE | Min MA-RAE | "
            "$\\Delta$ MA-RAE to min (\\%)[^1] | R2 | Spearman R | "
            "Kendall's Tau | Submission Time | Notes |"
        )
        md_lines.append("|---|---|---:|---:|---:|---:|---:|---:|---|---|")
        overall_row_parts = [
            f"{rank}/{total}",
            TARGET_USER,
            str(ma_rae),
            min_ma_rae_str,
            delta_ma_rae,
            str(r2),
            str(spearman),
            str(kendall),
            timestamp,
            note,
        ]
        md_lines.append("| " + " | ".join(overall_row_parts) + " |")

    # By Task section - exactly matching SUBMISSIONS.md format
    md_lines.append("\n#### By Task\n")
    md_lines.append(
        "| Rank | Task | User | MAE | Min MAE | "
        "$\\Delta$ MAE to min (\\%)[^2] | R2 | Spearman R | "
        "Kendall's Tau | Submission Time | Notes |"
    )
    md_lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---|")

    def get_note_from_rank(rank_val):
        """Categorize performance based on rank position"""
        try:
            rank_int = int(rank_val) if pd.notna(rank_val) else None
            if rank_int is None:
                return "Performance data unavailable"
            if rank_int <= 10:
                return "Excellent performance"
            elif rank_int <= 20:
                return "Good performance"
            elif rank_int <= 40:
                return "Okay performance"
            elif rank_int <= 60:
                return "Poor performance"
            else:
                return "Terrible performance"
        except (ValueError, TypeError):
            return "Performance data unavailable"

    for _, row in out[out["task"] != "OVERALL"].iterrows():
        task = row["task"]
        rank = row["rank"] if pd.notna(row["rank"]) else "—"
        mae_full = row.get("mae", "N/A")
        r2 = row.get("r2", "N/A")
        spearman = row.get("spearman r", row.get("spearman", "N/A"))
        kendall = row.get("kendall's tau", row.get("kendall", "N/A"))

        # Get minimum and calculate delta - task names must match labels list
        min_mae_val = task_mins.get(task)
        delta_mae = "N/A"
        min_mae_str = str(min_mae_val) if min_mae_val is not None else "N/A"
        note = get_note_from_rank(rank)

        if min_mae_val is not None and mae_full != "N/A":
            delta_mae = calc_delta_pct(mae_full, str(min_mae_val))
        else:
            # Debug: print if minimum is missing
            if mae_full != "N/A":
                print(f"Debug: No minimum found for task '{task}', available: {list(task_mins.keys())}")

        # Build row with all columns in order
        row_parts = [
            str(rank),
            task,
            TARGET_USER,
            mae_full,
            min_mae_str,
            delta_mae,
            str(r2),
            str(spearman),
            str(kendall),
            timestamp,
            note,
        ]
        md_lines.append("| " + " | ".join(row_parts) + " |")

    # Visual summary section linking to key plots
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
            "Per-task heatmaps for $R^2$, Spearman R, Kendall's $\\tau$, and MAE.",
        ),
    ]
    for title, filename, blurb in highlight_specs:
        rel_path = os.path.join("figures", filename)
        png_path = os.path.join("figures", "png", filename)
        md_lines.append(f"- ![{title}]({png_path}) [{title}]({rel_path}): {blurb}")

    # Actionable insights section with new diagnostic plots
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
        rel_path = os.path.join("figures", filename)
        png_path = os.path.join("figures", "png", filename)
        md_lines.append(f"- ![{title}]({png_path}) [{title}]({rel_path}): {blurb}")

    # Add footnotes
    md_lines.append(
        "\n* [^1]: $\\Delta$ MA-RAE to min (\\%) = ((mean MA-RAE - minimum MA-RAE) / "
        "mean MA-RAE) $\\times$ 100\\%, rounded to 1 decimal place."
    )
    md_lines.append(
        "* [^2]: $\\Delta$ MAE to min (\\%) = ((mean MAE - minimum MAE) / mean MAE) "
        "$\\times$ 100\\%, rounded to 1 decimal place."
    )
    # Generate publication-quality plots
    if HAS_MATPLOTLIB:
        print("\n=== Generating Plots ===")
        _generate_plots(
            out,
            task_mins,
            overall_min_ma_rae,
            figures_path,
            data_path,
            TARGET_USER,
            labels[1:],
        )
    # Write report only to cache directory
    md_content = "\n".join(md_lines)
    report_cache_path = os.path.join(cache_path, "report.md")
    try:
        with open(report_cache_path, "w", encoding="utf-8") as f:
            f.write(md_content)
        print(f"\n\n=== Markdown Report Written to: {report_cache_path} ===")
    except Exception as e:
        print(f"Error writing report to cache: {e}")
    print(md_content)


if __name__ == "__main__":
    main()
