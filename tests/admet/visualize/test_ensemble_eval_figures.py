"""Visualization tests for ensemble evaluation plotting utilities.

Ensure labeled evaluation and blind distribution plots are generated and
contain expected files.
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from admet.visualize.ensemble_eval import plot_blind_distributions, plot_labeled_eval_outputs


def _make_eval_frames():
    df_eval = pd.DataFrame(
        {
            "Molecule Name": [f"mol{i}" for i in range(5)],
            "SMILES": [f"C{i}H{i}" for i in range(5)],
            "LogD": np.linspace(0.1, 0.5, 5),
            "KSOL": np.linspace(1.0, 2.0, 5),
        }
    )
    preds_log = df_eval.copy()
    preds_log["LogD"] = preds_log["LogD"] + 0.1
    preds_log["KSOL"] = preds_log["KSOL"] - 0.1
    preds_linear = preds_log.copy()
    preds_linear["KSOL"] = np.power(10.0, preds_linear["KSOL"])
    metrics_entries = []
    for metric in ["mae", "rmse", "R2", "pearson_r2", "spearman_rho2", "kendall_tau"]:
        for ep in ["LogD", "KSOL", "macro"]:
            metrics_entries.append(
                {
                    "type": "models_agg_mean_log",
                    "endpoint": ep,
                    "space": "log",
                    "metric": metric,
                    "value": 0.1,
                }
            )
            metrics_entries.append(
                {
                    "type": "models_agg_stderr_log",
                    "endpoint": ep,
                    "space": "log",
                    "metric": metric,
                    "value": 0.01,
                }
            )
            metrics_entries.append(
                {"type": "ensemble_log", "endpoint": ep, "space": "log", "metric": metric, "value": 0.2}
            )
            metrics_entries.append(
                {
                    "type": "models_agg_mean_linear",
                    "endpoint": ep,
                    "space": "linear",
                    "metric": metric,
                    "value": 1.0,
                }
            )
            metrics_entries.append(
                {
                    "type": "models_agg_stderr_linear",
                    "endpoint": ep,
                    "space": "linear",
                    "metric": metric,
                    "value": 0.5,
                }
            )
            metrics_entries.append(
                {
                    "type": "ensemble_linear",
                    "endpoint": ep,
                    "space": "linear",
                    "metric": metric,
                    "value": 1.5,
                }
            )
    metrics_log = pd.DataFrame(metrics_entries)
    metrics_lin = metrics_log[metrics_log["space"] == "linear"].copy()
    metrics_log = metrics_log[metrics_log["space"] == "log"].copy()
    return df_eval, preds_log, preds_linear, metrics_log, metrics_lin


def test_labeled_eval_plots(tmp_path: Path) -> None:
    df_eval, preds_log, preds_lin, metrics_log, metrics_lin = _make_eval_frames()
    figures_dir = tmp_path / "figures"
    plot_labeled_eval_outputs(
        df_eval=df_eval,
        preds_log_df=preds_log,
        preds_linear_df=preds_lin,
        metrics_log_df=metrics_log,
        metrics_linear_df=metrics_lin,
        figures_dir=figures_dir,
        dpi=100,
        n_jobs=1,
    )

    parity_log = list((figures_dir / "parity" / "log").glob("*.png"))
    parity_lin = list((figures_dir / "parity" / "linear").glob("*.png"))
    assert parity_log and parity_lin
    for subdir in [
        figures_dir / "metrics" / "models_agg_mean_log",
        figures_dir / "metrics" / "ensemble_log",
        figures_dir / "metrics" / "models_agg_mean_linear",
        figures_dir / "metrics" / "ensemble_linear",
    ]:
        files = list(subdir.glob("*.png"))
        assert files, f"Expected metric plots in {subdir}"
    log_dists = list((figures_dir / "distributions" / "log").glob("*.png"))
    lin_dists = list((figures_dir / "distributions" / "linear").glob("*.png"))
    assert log_dists and lin_dists


def test_blind_distribution_plots(tmp_path: Path) -> None:
    df_blind_log = pd.DataFrame(
        {
            "Molecule Name": ["mol1", "mol2"],
            "SMILES": ["C", "CC"],
            "LogD": [0.1, 0.2],
            "KSOL": [1.0, 1.2],
        }
    )
    df_blind_lin = df_blind_log.copy()
    df_blind_lin["LogD"] = np.power(10.0, df_blind_lin["LogD"])
    df_blind_lin["KSOL"] = np.power(10.0, df_blind_lin["KSOL"])

    figures_dir = tmp_path / "blind_figs"
    plot_blind_distributions(
        preds_log_df=df_blind_log,
        preds_linear_df=df_blind_lin,
        figures_dir=figures_dir,
        dpi=100,
    )

    log_files = list((figures_dir / "distributions" / "log").glob("*.png"))
    lin_files = list((figures_dir / "distributions" / "linear").glob("*.png"))
    assert log_files and lin_files
