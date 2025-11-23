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
import datetime
from pathlib import Path
import logging

import typer
import yaml
import numpy as np
import mlflow

from admet.data.load import load_dataset
from admet.train.xgb_train import XGBoostTrainer
from admet.train.base import train_model, train_ensemble, BaseEnsembleTrainer
from admet.model.xgb_wrapper import XGBoostMultiEndpoint
from admet.train.mlflow_utils import flatten_metrics, flatten_params, set_mlflow_tracking
from admet.utils import get_git_commit_hash

logger = logging.getLogger(__name__)
app = typer.Typer(help="Model training commands.")


@app.command("xgb")
def xgb(
    config: Path = typer.Option(
        ...,
        "--config",
        "-c",
        help="YAML configuration file with models.xgboost section.",
    ),
    data_root: Path | None = typer.Option(
        None,
        "--data-root",
        "-d",
        help=(
            "Optional override for the HF DatasetDict directory (single) or parent "
            "directory containing multiple 'hf_dataset' subdirectories (multi). "
            "Defaults to the `data.root` entry in the YAML config."
        ),
    ),
) -> None:
    """Train per-endpoint XGBoost models using a YAML configuration.

    Parameters
    ----------
    config : pathlib.Path
        YAML file with ``models.xgboost`` hyperparameters and optional sections
        for endpoints (``data.endpoints``), sample weights (``training.sample_weights``),
        output directory (``training.output_dir``), fingerprint size
        (``training.n_fingerprint_bits``), dataset location (``data.root``), and
        Ray configuration (``ray.address``, ``ray.num_cpus``).
    data_root : pathlib.Path, optional
        Override for the dataset directory. If omitted, ``data.root`` from the YAML
        configuration is used. Required when ``data.root`` is missing.
    """
    cfg = yaml.safe_load(config.read_text()) or {}
    # Use `or {}` to guard against keys being present with a null value in YAML
    # (yaml.safe_load can return None for empty files, and a key that maps to
    # `null` will cause cfg.get("...", {}) to return None instead of a dict).
    data_cfg = cfg.get("data") or {}
    endpoints = data_cfg.get("endpoints")
    cfg_data_root = data_cfg.get("root")
    effective_data_root: Path | None
    if data_root is not None:
        effective_data_root = data_root.expanduser()
    elif cfg_data_root is not None:
        effective_data_root = Path(cfg_data_root).expanduser()
    else:
        raise typer.BadParameter(
            "Dataset directory is not provided. Supply --data-root or set data.root in the config."
        )
    data_cfg["resolved_root"] = str(effective_data_root)
    xgb_cfg = (cfg.get("models") or {}).get("xgboost") or {}
    model_params = xgb_cfg.get("model_params", {})
    objective = xgb_cfg.get("objective", "rmse")
    if objective == "mae":
        model_params["objective"] = "reg:absoluteerror"
        model_params["eval_metric"] = "mae"
    elif objective == "rmse":
        model_params["objective"] = "reg:squarederror"
        model_params["eval_metric"] = "rmse"
    xgb_cfg["objective"] = objective
    early_stopping_rounds = xgb_cfg.get("early_stopping_rounds", 50)
    xgb_cfg["early_stopping_rounds"] = early_stopping_rounds
    training_cfg = cfg.get("training") or {}
    sw_cfg = training_cfg.get("sample_weights") or {}
    sw_enabled = sw_cfg.get("enabled", False)
    sw_mapping = sw_cfg.get("weights") if sw_enabled else None
    output_dir_raw = training_cfg.get("output_dir", "xgb_artifacts")
    training_cfg["output_dir"] = output_dir_raw
    output_dir = Path(output_dir_raw)
    seed = training_cfg.get("seed")
    training_cfg.setdefault("seed", seed)
    n_fingerprint_bits = int(training_cfg.get("n_fingerprint_bits", 2048))
    training_cfg["n_fingerprint_bits"] = n_fingerprint_bits
    tracking_uri = training_cfg.get("tracking_uri")
    experiment_name = training_cfg.get("experiment_name")
    training_cfg.setdefault("experiment_name", experiment_name)
    training_cfg.setdefault("tracking_uri", tracking_uri)

    ray_cfg = cfg.get("ray") or {}
    multi = bool(ray_cfg.get("multi", False))
    ray_cfg["multi"] = multi
    num_cpus = ray_cfg.get("num_cpus", None)
    ray_cfg["num_cpus"] = num_cpus
    ray_address = ray_cfg.get("address")
    ray_cfg["address"] = ray_address

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

    flattened_cfg_params = flatten_params(cfg, prefix="cfg")
    cli_params = flatten_params(
        {
            "cli": {
                "config_path": str(config),
                "data_root_override": str(data_root) if data_root is not None else "",
                "effective_data_root": str(effective_data_root),
                "multi": multi,
                "ray_address": ray_address or "",
                "num_cpus": num_cpus if num_cpus is not None else "",
            }
        }
    )
    git_commit = get_git_commit_hash()

    if not multi:
        set_mlflow_tracking(tracking_uri, experiment_name)
        single_tags = {"mode": "single", "data_root": str(effective_data_root)}
        if git_commit:
            single_tags["git_commit"] = git_commit
        with mlflow.start_run(run_name=effective_data_root.name, tags=single_tags):
            mlflow.log_params(flattened_cfg_params)
            mlflow.log_params(cli_params)
            mlflow.set_tag("featurization", "morgan_fp")
            if endpoints:
                mlflow.set_tag("endpoints", ",".join(endpoints))
            if config.is_file():
                mlflow.log_artifact(str(config))
            start_ts = datetime.datetime.now()
            try:
                dataset = load_dataset(
                    effective_data_root,
                    endpoints=endpoints,
                    n_fingerprint_bits=n_fingerprint_bits,
                )
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
            except Exception as exc:  # noqa: BLE001
                mlflow.set_tag("status", "error")
                mlflow.set_tag("error", str(exc))
                mlflow.log_metric("duration_seconds", (datetime.datetime.now() - start_ts).total_seconds())
                raise
            duration_seconds = (datetime.datetime.now() - start_ts).total_seconds()
            mlflow.set_tag("status", "ok")
            mlflow.log_metric("duration_seconds", duration_seconds)
            mlflow.log_metrics(flatten_metrics(run_metrics))
            if output_dir.exists():
                mlflow.log_artifacts(str(output_dir))

        # output macro metrics for train, val, and test with 4 decimal places
        for split in ["train", "validation", "test"]:
            typer.echo(f"Metrics for {split} split:")
            macro_metrics = run_metrics[split]["macro"]
            formatted_metrics = {k: _format_value(v) for k, v in macro_metrics.items()}
            typer.echo(json.dumps(formatted_metrics, indent=2))
    else:
        set_mlflow_tracking(tracking_uri, experiment_name)
        ensemble_tags = {"mode": "ensemble", "data_root": str(effective_data_root)}
        if git_commit:
            ensemble_tags["git_commit"] = git_commit
        with mlflow.start_run(run_name=f"ensemble:{effective_data_root.name}", tags=ensemble_tags) as parent_run:
            mlflow.log_params(flattened_cfg_params)
            mlflow.log_params(cli_params)
            if config.is_file():
                mlflow.log_artifact(str(config))
            start_ts = datetime.datetime.now()
            try:
                results = train_ensemble(
                    effective_data_root,
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
                    ray_address=ray_address,
                    mlflow_tracking_uri=tracking_uri,
                    mlflow_experiment_name=experiment_name,
                    mlflow_parent_run_id=parent_run.info.run_id if multi else None,
                    mlflow_params=flattened_cfg_params,
                    mlflow_cli_params=cli_params,
                )
            except Exception as exc:  # noqa: BLE001
                mlflow.set_tag("status", "error")
                mlflow.set_tag("error", str(exc))
                mlflow.log_metric("ensemble.duration_seconds", (datetime.datetime.now() - start_ts).total_seconds())
                raise
            duration_seconds = (datetime.datetime.now() - start_ts).total_seconds()
            mlflow.log_metric("ensemble.duration_seconds", duration_seconds)

            status_counts: dict[str, int] = {}
            for rel_key, payload in results.items():
                status = str(payload.get("status", "unknown"))
                status_counts[status] = status_counts.get(status, 0) + 1
                run_metrics = payload.get("run_metrics")
                if run_metrics:
                    mlflow.log_metrics(flatten_metrics(run_metrics, prefix=f"ensemble.{rel_key}"))
            status_counts["total"] = len(results)
            if status_counts:
                mlflow.log_metrics({f"ensemble.status.{k}": v for k, v in status_counts.items()})
            failures = status_counts.get("error", 0) + status_counts.get("partial", 0) + status_counts.get("timeout", 0)
            parent_status = "ok" if failures == 0 else ("error" if failures and failures == status_counts.get("total", 0) else "partial")
            mlflow.set_tag("status", parent_status)
            for summary_file in [output_dir / "metrics_summary.csv", output_dir / "metrics_summary.json"]:
                if summary_file.exists():
                    mlflow.log_artifact(str(summary_file))

            # Print metrics per discovered dataset
            for rel_key, payload in results.items():
                typer.echo(f"=== Results for dataset: {rel_key} ===")
                run_metrics = payload.get("run_metrics")
                if not run_metrics:
                    typer.echo(f"No metrics available (status: {payload.get('status', 'unknown')})")
                    continue
                for split in ["train", "validation", "test"]:
                    typer.echo(f"Metrics for {split} split:")
                    macro_metrics = run_metrics[split]["macro"]
                    formatted_metrics = {k: _format_value(v) for k, v in macro_metrics.items()}
                    typer.echo(json.dumps(formatted_metrics, indent=2))


__all__ = ["app", "xgb"]
