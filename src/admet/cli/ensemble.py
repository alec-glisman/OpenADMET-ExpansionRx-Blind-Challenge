"""CLI subcommand for ensemble evaluation across labeled and blind CSVs.

Add a Typer command `admet ensemble-eval` that accepts a list of trained model
directories (comma separated), an optional labeled CSV, optional blind CSV,
output directory, and aggregation function. The command will load models from
disk, run ensemble predictions, compute metrics (for labeled eval), and save
predictions/metrics to a standardized output directory.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import typer
import yaml  # type: ignore[import-not-found]

from admet.data.chem import parallel_canonicalize_smiles
from admet.data.fingerprinting import DEFAULT_FINGERPRINT_CONFIG, FingerprintConfig
from admet.evaluate.ensemble import EnsemblePredictConfig, run_ensemble_predictions_from_root
from admet.visualize.ensemble_eval import plot_blind_distributions, plot_labeled_eval_outputs

logger = logging.getLogger(__name__)


def _read_csv(path: Path) -> pd.DataFrame:
    """Read a CSV file into a DataFrame.

    Parameters
    ----------
    path : Path
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame with original columns.
    """
    return pd.read_csv(path)


def _ensure_outdir(path: Path) -> None:
    """Ensure directory exists (create parents as needed)."""
    path.mkdir(parents=True, exist_ok=True)


def ensemble_eval(
    config: Path = typer.Option(..., "--config", "-c", help="YAML config for ensemble evaluation."),
) -> None:
    """Run ensemble inference and evaluation from a parent models directory.

    Parameters
    ----------
    config : Path
        YAML configuration file specifying the evaluation inputs.

    YAML Structure
    --------------
    Required keys / examples::

        models_root: path/to/parent/models_dir
        eval_csv: path/to/eval.csv            # optional labeled evaluation set
        blind_csv: path/to/blind.csv          # optional blind set (no targets)
        train_data_root: path/to/training_splits_root  # optional HF splits root used for training
        agg_fn: mean                          # aggregation function (mean|median)
        output_dir: path/to/output_dir        # directory for artifacts
        n_jobs: 4                             # parallelism for plotting/canonicalization

    Behavior
    --------
    * When ``eval_csv`` is supplied, metrics (log & linear space) and prediction
      CSVs are produced plus summary visualizations.
    * When ``blind_csv`` is supplied, only predictions (log & linear) and
      distribution plots are generated.
    * At least one of ``eval_csv`` or ``blind_csv`` must be present.
    """
    cfg_raw = yaml.safe_load(config.read_text()) or {}
    cfg_dict: Dict[str, Any] = cfg_raw if isinstance(cfg_raw, dict) else {}
    if "models_root" not in cfg_dict:
        raise typer.BadParameter("'models_root' key missing from config YAML.")
    models_root = Path(str(cfg_dict["models_root"])).expanduser()
    eval_csv: Optional[str] = cfg_dict.get("eval_csv")
    blind_csv: Optional[str] = cfg_dict.get("blind_csv")
    train_data_root_raw: Optional[str] = cfg_dict.get("train_data_root")
    train_data_root: Optional[Path] = (
        Path(str(train_data_root_raw)).expanduser() if train_data_root_raw is not None else None
    )
    agg_fn: str = str(cfg_dict.get("agg_fn", "mean"))
    output_dir_raw: str = str(cfg_dict.get("output_dir", "ensemble_eval_output"))
    output_dir = Path(output_dir_raw)
    n_jobs: int = int(cfg_dict.get("n_jobs", 1))
    fp_cfg = FingerprintConfig.from_mapping(cfg_dict.get("fingerprint"), default=DEFAULT_FINGERPRINT_CONFIG)

    if eval_csv is None and blind_csv is None:
        raise typer.BadParameter("At least one of eval_csv or blind_csv must be provided in the config.")

    _ensure_outdir(output_dir)
    data_root = output_dir / "data"
    figures_root = output_dir / "figures"
    _ensure_outdir(data_root)
    _ensure_outdir(figures_root)

    df_eval: Optional[pd.DataFrame] = None
    if eval_csv is not None:
        df_eval = _read_csv(Path(eval_csv))
        logger.info("Read eval CSV with %d rows from %s", len(df_eval), eval_csv)
        df_eval["SMILES"] = parallel_canonicalize_smiles(df_eval["SMILES"].astype(str))

    df_blind: Optional[pd.DataFrame] = None
    if blind_csv is not None:
        df_blind = _read_csv(Path(blind_csv))
        logger.info("Read blind CSV with %d rows from %s", len(df_blind), blind_csv)
        df_blind["SMILES"] = parallel_canonicalize_smiles(df_blind["SMILES"].astype(str))

    pred_cfg = EnsemblePredictConfig(
        models_root=models_root,
        eval_csv=Path(eval_csv) if eval_csv is not None else None,
        blind_csv=Path(blind_csv) if blind_csv is not None else None,
        agg_fn=agg_fn,
        n_jobs=n_jobs,
        fingerprint_config=fp_cfg,
        train_data_root=train_data_root,
    )

    summary = run_ensemble_predictions_from_root(pred_cfg, df_eval=df_eval, df_blind=df_blind)
    eval_shape = summary.preds_log_eval.shape if summary.preds_log_eval is not None else None
    blind_shape = summary.preds_log_blind.shape if summary.preds_log_blind is not None else None
    logger.info(
        "Completed ensemble predictions with shape eval=%s, blind=%s",
        eval_shape,
        blind_shape,
    )

    if summary.preds_log_blind is not None:
        blind_data_dir = data_root / "blind"
        blind_data_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Writing blind predictions to %s", blind_data_dir)
        summary.preds_log_blind.to_csv(blind_data_dir / "predictions_log.csv", index=False)
        assert summary.preds_linear_blind is not None
        summary.preds_linear_blind.to_csv(blind_data_dir / "predictions_linear.csv", index=False)
        plot_blind_distributions(
            preds_log_df=summary.preds_log_blind,
            preds_linear_df=summary.preds_linear_blind,
            figures_dir=figures_root / "blind",
            dpi=600,
        )

    if summary.train_split_evaluations:
        train_data_dir = data_root / "train"
        train_figures_dir = figures_root / "train"
        logger.info(
            "Writing training split predictions and metrics to %s (splits=%s)",
            train_data_dir,
            ",".join(sorted(summary.train_split_evaluations.keys())),
        )
        for split_name, artifacts in summary.train_split_evaluations.items():
            split_dir = train_data_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            artifacts.preds_log.to_csv(split_dir / "predictions_log.csv", index=False)
            artifacts.preds_linear.to_csv(split_dir / "predictions_linear.csv", index=False)
            artifacts.metrics_log.to_csv(split_dir / "metrics_log.csv", index=False)
            artifacts.metrics_linear.to_csv(split_dir / "metrics_linear.csv", index=False)

            plot_labeled_eval_outputs(
                df_eval=artifacts.df_true,
                preds_log_df=artifacts.preds_log,
                preds_linear_df=artifacts.preds_linear,
                metrics_log_df=artifacts.metrics_log,
                metrics_linear_df=artifacts.metrics_linear,
                figures_dir=train_figures_dir / split_name,
                dpi=600,
                n_jobs=n_jobs,
                parity_split=split_name if split_name in {"train", "validation", "test"} else "validation",
            )

    # Save outputs using the same directory structure as before.
    if summary.preds_log_eval is not None and df_eval is not None:
        eval_data_dir = data_root / "eval"
        eval_data_dir.mkdir(parents=True, exist_ok=True)
        summary.preds_log_eval.to_csv(eval_data_dir / "predictions_log.csv", index=False)

        assert summary.preds_linear_eval is not None
        assert summary.metrics_log_eval is not None
        assert summary.metrics_linear_eval is not None
        logger.info("Writing evaluation predictions and metrics to %s", eval_data_dir)
        summary.preds_linear_eval.to_csv(eval_data_dir / "predictions_linear.csv", index=False)
        summary.metrics_log_eval.to_csv(eval_data_dir / "metrics_log.csv", index=False)
        summary.metrics_linear_eval.to_csv(eval_data_dir / "metrics_linear.csv", index=False)

        plot_labeled_eval_outputs(
            df_eval=df_eval,
            preds_log_df=summary.preds_log_eval,
            preds_linear_df=summary.preds_linear_eval,
            metrics_log_df=summary.metrics_log_eval,
            metrics_linear_df=summary.metrics_linear_eval,
            figures_dir=figures_root / "eval",
            dpi=600,
            n_jobs=n_jobs,
        )

    logger.info("Ensemble evaluation complete. Outputs written to %s", output_dir)


__all__ = ["ensemble_eval"]
