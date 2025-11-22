"""admet.cli.train
===================

Train XGBoost models for ADMET endpoints via Typer CLI.

Features
--------
* Single HF ``DatasetDict`` training.
* Multi-dataset batch training using Ray (``--multi``).
* YAML-driven configuration (objective, early stopping, endpoints, sample weights).
* Automatic objective mapping shortcuts (``mae``/``rmse``).

Configuration File (YAML)
-------------------------
Example minimal structure::

        data:
            endpoints: [LogD, KSOL, HLM CLint]
        models:
            xgboost:
                objective: rmse
                early_stopping_rounds: 75
                model_params:
                    max_depth: 6
                    learning_rate: 0.05
        training:
            sample_weights:
                enabled: true
                weights:
                    default: 1.0
                    expansion_teaser: 1.2

Usage Examples
--------------
Single dataset training::

        admet train xgb path/to/hf_dataset --config cfg.yaml

Multi-dataset Ray training::

        admet train xgb path/to/parent --multi --config cfg.yaml --ray-address auto
"""

from __future__ import annotations

import json
from pathlib import Path
import logging

import typer
import yaml
import numpy as np

from admet.data.load import load_dataset
from admet.train.xgb_train import XGBoostTrainer
from admet.train.base import train_model, train_ensemble, BaseEnsembleTrainer
from admet.model.xgb_wrapper import XGBoostMultiEndpoint

logger = logging.getLogger(__name__)
app = typer.Typer(help="Model training commands.")


@app.command("xgb")
def xgb(
    data_root: Path = typer.Argument(
        ...,
        help=(
            "Path to a saved HF DatasetDict directory or a parent "
            "directory containing multiple 'hf_dataset' subdirectories."
        ),
    ),
    config: Path = typer.Option(..., help="YAML configuration file with models.xgboost section."),
    output_dir: Path = typer.Option(Path("xgb_artifacts"), help="Directory to write model artifacts."),
    multi: bool = typer.Option(
        False,
        "--multi",
        help=(
            "If set, treat data_root as a parent directory and train "
            "models for all discovered 'hf_dataset' subdirectories "
            "using Ray."
        ),
    ),
    ray_address: str | None = typer.Option(
        None,
        "--ray-address",
        help=(
            "Ray address to connect to (e.g. 'auto' or 'ray://host:10001'). "
            "If omitted, a local Ray runtime is started."
        ),
    ),
    seed: int | None = typer.Option(None, help="Random seed for reproducibility."),
    n_fingerprint_bits: int = typer.Option(2048, help="Number of fingerprint bits for dataset loading."),
) -> None:
    """Train per-endpoint XGBoost models using a YAML configuration.

    Parameters
    ----------
    data_root : pathlib.Path
        HF ``DatasetDict`` directory or parent containing multiple ``hf_dataset`` dirs.
    config : pathlib.Path
        YAML file with ``models.xgboost`` hyperparameters and optional sections
        for endpoints (``data.endpoints``) and sample weights (``training.sample_weights``).
    output_dir : pathlib.Path
        Directory where artifacts are written.
    multi : bool, optional
        Batch mode; treat ``data_root`` as parent and train each nested dataset via Ray.
    ray_address : str, optional
        Ray cluster address (``auto`` or explicit ``ray://``). Local runtime started if omitted.
    seed : int, optional
        Global seed for reproducibility.
    n_fingerprint_bits : int, optional
        Fingerprint bit count used during loading (default 2048).
    """
    cfg = yaml.safe_load(config.read_text()) or {}
    # Use `or {}` to guard against keys being present with a null value in YAML
    # (yaml.safe_load can return None for empty files, and a key that maps to
    # `null` will cause cfg.get("...", {}) to return None instead of a dict).
    endpoints = (cfg.get("data") or {}).get("endpoints")
    xgb_cfg = (cfg.get("models") or {}).get("xgboost") or {}
    model_params = xgb_cfg.get("model_params", {})
    objective = xgb_cfg.get("objective", "rmse")
    if objective == "mae":
        model_params["objective"] = "reg:absoluteerror"
        model_params["eval_metric"] = "mae"
    elif objective == "rmse":
        model_params["objective"] = "reg:squarederror"
        model_params["eval_metric"] = "rmse"
    early_stopping_rounds = xgb_cfg.get("early_stopping_rounds", 50)
    sw_cfg = (cfg.get("training") or {}).get("sample_weights") or {}
    sw_enabled = sw_cfg.get("enabled", False)
    sw_mapping = sw_cfg.get("weights") if sw_enabled else None

    def _format_value(v):
        # Format numeric values to 4 decimal places; for dicts, format inner numeric values;
        # otherwise fall back to JSON-serializable Python types (strings or None).
        if v is None:
            return None
        if isinstance(v, (int, float)):
            return np.round(v, 4)
        if isinstance(v, dict):
            formatted = {}
            for kk, vv in v.items():
                if vv is None:
                    formatted[kk] = None
                elif isinstance(vv, (int, float)):
                    formatted[kk] = np.round(vv, 4)
                else:
                    formatted[kk] = vv
            return formatted
        return str(v)

    if not multi:
        dataset = load_dataset(data_root, endpoints=endpoints, n_fingerprint_bits=n_fingerprint_bits)
        run_metrics, summary = train_model(
            dataset,
            trainer_cls=XGBoostTrainer,
            model_cls=XGBoostMultiEndpoint,
            model_params=model_params,
            early_stopping_rounds=early_stopping_rounds,
            sample_weight_mapping=sw_mapping,
            output_dir=output_dir,
            seed=seed,
        )

        # output macro metrics for train, val, and test with 4 decimal places
        for split in ["train", "validation", "test"]:
            typer.echo(f"Metrics for {split} split:")
            macro_metrics = run_metrics[split]["macro"]
            formatted_metrics = {k: _format_value(v) for k, v in macro_metrics.items()}
            typer.echo(json.dumps(formatted_metrics, indent=2))
    else:
        # Optional Ray configuration from YAML. Guard against `ray: null` cases
        # which would cause cfg.get("ray", {}) to return None and break .get.
        ray_cfg = cfg.get("ray") or {}
        num_cpus = ray_cfg.get("num_cpus", None)
        # CLI flag takes precedence over config file for address
        ray_addr_effective = ray_address or ray_cfg.get("address")

        results = train_ensemble(
            data_root,
            ensemble_trainer_cls=BaseEnsembleTrainer,
            trainer_cls=XGBoostTrainer,
            model_cls=XGBoostMultiEndpoint,
            model_params=model_params,
            early_stopping_rounds=early_stopping_rounds,
            sample_weight_mapping=sw_mapping,
            output_root=output_dir,
            seed=seed,
            n_fingerprint_bits=n_fingerprint_bits,
            num_cpus=num_cpus,
            ray_address=ray_addr_effective,
        )

        # Print metrics per discovered dataset
        for rel_key, payload in results.items():
            typer.echo(f"=== Results for dataset: {rel_key} ===")
            run_metrics = payload["run_metrics"]
            for split in ["train", "validation", "test"]:
                typer.echo(f"Metrics for {split} split:")
                macro_metrics = run_metrics[split]["macro"]
                formatted_metrics = {k: _format_value(v) for k, v in macro_metrics.items()}
                typer.echo(json.dumps(formatted_metrics, indent=2))


__all__ = ["app", "xgb"]
