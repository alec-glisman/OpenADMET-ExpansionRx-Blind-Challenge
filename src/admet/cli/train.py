"""CLI entrypoint for training XGBoost models (initial)."""

from __future__ import annotations

import json
from pathlib import Path
import logging
import typer
import yaml

from admet.data.load import load_dataset
from admet.train.xgb_train import train_xgb_models

logger = logging.getLogger(__name__)
app = typer.Typer(help="Model training commands.")


@app.command("xgb")
def train_xgb(
    data_root: Path = typer.Argument(..., help="Directory containing train.csv, val.csv, test.csv"),
    config: Path = typer.Option(..., help="YAML configuration file with models.xgboost section."),
    output_dir: Path = typer.Option(Path("xgb_artifacts"), help="Directory to write model artifacts."),
    seed: int | None = typer.Option(None, help="Random seed for reproducibility."),
) -> None:
    """Train XGBoost per-endpoint models using a simplified config.

    The YAML config should define:

    models:
      xgboost:
        model_params: {...}
        early_stopping_rounds: 50
    training:
      sample_weights:
        enabled: true
        weights: { dataset_a: 1.0, default: 1.0 }
    data:
      endpoints: [LogD, KSOL, ...]
    """
    cfg = yaml.safe_load(config.read_text())
    endpoints = cfg.get("data", {}).get("endpoints")
    xgb_cfg = cfg.get("models", {}).get("xgboost", {})
    model_params = xgb_cfg.get("model_params")
    early_stopping_rounds = xgb_cfg.get("early_stopping_rounds", 50)
    sw_cfg = cfg.get("training", {}).get("sample_weights", {})
    sw_enabled = sw_cfg.get("enabled", False)
    sw_mapping = sw_cfg.get("weights") if sw_enabled else None

    dataset = load_dataset(data_root, endpoints=endpoints, n_fingerprint_bits=16)
    metrics = train_xgb_models(
        dataset,
        model_params=model_params,
        early_stopping_rounds=early_stopping_rounds,
        sample_weight_mapping=sw_mapping,
        output_dir=output_dir,
        seed=seed,
    )
    typer.echo(json.dumps(metrics["val"]["macro"], indent=2))


__all__ = ["app", "train_xgb"]
