"""CLI subcommand for ensemble evaluation across labeled and blind CSVs.

Add a Typer command `admet ensemble-eval` that accepts a list of trained model
directories (comma separated), an optional labeled CSV, optional blind CSV,
output directory, and aggregation function. The command will load models from
disk, run ensemble predictions, compute metrics (for labeled eval), and save
predictions/metrics/plots to a standardized output directory.
"""

from __future__ import annotations

from pathlib import Path

import logging
import pandas as pd
import typer
import yaml

from admet.data.chem import parallel_canonicalize_smiles
from admet.evaluate.ensemble import (
    EnsemblePredictConfig,
    run_ensemble_predictions_from_root,
)
from admet.visualize.ensemble_performance import (
    plot_blind_distributions,
    plot_labeled_ensemble,
)

logger = logging.getLogger(__name__)


def _read_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df


def _ensure_outdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def ensemble_eval(
    config: Path = typer.Option(..., "--config", "-c", help="YAML config for ensemble evaluation."),
):
    """Run ensemble inference and evaluation from a parent models directory.

    The provided YAML configuration must define at least::

        models_root: path/to/parent/models_dir
        eval_csv: path/to/eval.csv  # optional
        blind_csv: path/to/blind.csv  # optional
        agg_fn: mean  # or median
        output_dir: path/to/output_dir
        n_jobs: 4

    Predictions and metrics will be written under ``training.output_dir``.
    """
    cfg_dict = yaml.safe_load(config.read_text())
    models_root = Path(cfg_dict["models_root"]).expanduser()
    eval_csv = cfg_dict.get("eval_csv")
    blind_csv = cfg_dict.get("blind_csv")
    agg_fn = cfg_dict.get("agg_fn", "mean")
    output_dir_raw = cfg_dict.get("output_dir", "ensemble_eval_output")
    output_dir = Path(output_dir_raw)
    n_jobs = int(cfg_dict.get("n_jobs", 1))

    if eval_csv is None and blind_csv is None:
        raise typer.BadParameter("At least one of eval_csv or blind_csv must be provided in the config.")

    _ensure_outdir(output_dir)

    df_eval = None
    df_blind = None
    if eval_csv is not None:
        df_eval = _read_csv(Path(eval_csv))
        df_eval["SMILES"] = parallel_canonicalize_smiles(df_eval["SMILES"].astype(str))
    if blind_csv is not None:
        df_blind = _read_csv(Path(blind_csv))
        df_blind["SMILES"] = parallel_canonicalize_smiles(df_blind["SMILES"].astype(str))

    pred_cfg = EnsemblePredictConfig(
        models_root=models_root,
        eval_csv=Path(eval_csv) if eval_csv is not None else None,
        blind_csv=Path(blind_csv) if blind_csv is not None else None,
        agg_fn=agg_fn,
        n_jobs=n_jobs,
    )

    summary = run_ensemble_predictions_from_root(pred_cfg, df_eval=df_eval, df_blind=df_blind)

    # Save outputs using the same directory structure as before.
    if summary.preds_log_eval is not None and df_eval is not None:
        eval_outdir = output_dir / "eval"
        eval_outdir.mkdir(parents=True, exist_ok=True)
        summary.preds_log_eval.to_csv(eval_outdir / "predictions_log.csv", index=False)
        assert summary.preds_linear_eval is not None
        assert summary.metrics_log is not None
        assert summary.metrics_linear is not None
        summary.preds_linear_eval.to_csv(eval_outdir / "predictions_linear.csv", index=False)
        summary.metrics_log.to_csv(eval_outdir / "metrics_log.csv", index=False)
        summary.metrics_linear.to_csv(eval_outdir / "metrics_linear.csv", index=False)
        # Plots
        plot_labeled_ensemble(
            df_eval,
            summary.preds_log_eval,
            summary.preds_linear_eval,
            list(summary.preds_log_eval.columns[2::]),
            eval_outdir,
            n_jobs=n_jobs,
        )

    if summary.preds_log_blind is not None:
        blind_outdir = output_dir / "blind"
        blind_outdir.mkdir(parents=True, exist_ok=True)
        summary.preds_log_blind.to_csv(blind_outdir / "predictions_log.csv", index=False)
        assert summary.preds_linear_blind is not None
        summary.preds_linear_blind.to_csv(blind_outdir / "predictions_linear.csv", index=False)
        plot_blind_distributions(summary.preds_log_blind, summary.preds_linear_blind, [], blind_outdir)


__all__ = ["ensemble_eval"]
