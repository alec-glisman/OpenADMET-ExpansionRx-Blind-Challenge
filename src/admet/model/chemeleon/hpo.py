"""CheMeleon Hyperparameter Optimization with Ray Tune.

This module provides the main orchestrator class for running hyperparameter
optimization of CheMeleon models using Ray Tune with ASHA scheduler.

CheMeleon HPO differs from Chemprop HPO:
- The message passing encoder is frozen (pre-trained)
- Only FFN architecture and training dynamics are tuned
- Supports all FFN types: regression, mixture_of_experts, branched

Example usage
-------------
CLI:
    python -m admet.model.chemeleon.hpo --config configs/1-hpo-single/hpo_chemeleon.yaml

Python:
    from admet.model.chemeleon.hpo import ChemeleonHPO
    from admet.model.chemeleon.hpo_config import ChemeleonHPOConfig

    config = ChemeleonHPOConfig(
        experiment_name="chemeleon_hpo",
        data_path="data/train.csv",
        target_columns=["logD", "solubility"],
    )
    hpo = ChemeleonHPO(config)
    results = hpo.run()
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
from omegaconf import OmegaConf
from ray import tune
from ray.air.integrations.mlflow import MLflowLoggerCallback
from ray.tune.schedulers import ASHAScheduler

from admet.model.chemeleon.hpo_config import ChemeleonHPOConfig
from admet.model.chemeleon.hpo_search_space import build_chemeleon_search_space
from admet.util.logging import configure_logging

logger = logging.getLogger("admet.model.chemeleon.hpo")


def _trial_dirname_creator(trial) -> str:
    """Create a short directory name for the trial."""
    return f"trial_{trial.trial_id}"


def train_chemeleon_trial(config: dict[str, Any]) -> None:
    """Ray Tune trainable function for CheMeleon HPO trials.

    This function is called by Ray Tune for each trial. It creates a
    CheMeleon model with the sampled hyperparameters and trains it.

    Parameters
    ----------
    config : dict[str, Any]
        Hyperparameter configuration sampled by Ray Tune.
    """
    import pandas as pd
    from ray import tune

    from admet.model.chemeleon import ChemeleonModel

    # Extract fixed parameters
    data_path = config.pop("data_path")
    val_data_path = config.pop("val_data_path", None)
    smiles_column = config.pop("smiles_column")
    target_columns = config.pop("target_columns")
    max_epochs = config.pop("max_epochs")
    metric = config.pop("metric")
    seed = config.pop("seed", 42)
    checkpoint_path = config.pop("checkpoint_path", "auto")
    freeze_encoder = config.pop("freeze_encoder", True)

    # Load data
    df_train = pd.read_csv(data_path)
    df_val = pd.read_csv(val_data_path) if val_data_path else None

    # Build model config
    model_config = OmegaConf.create(
        {
            "model": {
                "type": "chemeleon",
                "chemeleon": {
                    "checkpoint_path": checkpoint_path,
                    "freeze_encoder": freeze_encoder,
                    "ffn_type": config.get("ffn_type", "regression"),
                    "ffn_hidden_dim": config.get("ffn_hidden_dim", 300),
                    "ffn_num_layers": config.get("ffn_num_layers", 2),
                    "dropout": config.get("dropout", 0.0),
                    "batch_norm": config.get("batch_norm", False),
                    "n_experts": config.get("n_experts"),
                    "trunk_n_layers": config.get("trunk_n_layers"),
                    "trunk_hidden_dim": config.get("trunk_hidden_dim"),
                },
            },
            "data": {
                "smiles_col": smiles_column,
                "target_cols": list(target_columns),
            },
            "optimization": {
                "max_epochs": max_epochs,
                "batch_size": config.get("batch_size", 32),
                "patience": config.get("patience", 15),
                "learning_rate": config.get("learning_rate", 1e-4),
            },
            "mlflow": {"enabled": False},
        }
    )

    # Create and train model
    model = ChemeleonModel(model_config)
    model.fit(df_train, df_validation=df_val)

    # Report metrics to Ray Tune
    metrics = model.get_validation_metrics() if hasattr(model, "get_validation_metrics") else {}

    # Report the target metric
    if metric in metrics:
        tune.report(**{metric: metrics[metric], "epoch": max_epochs})
    else:
        # Default to val_loss if available
        tune.report(**{"val_mae": metrics.get("val_mae", float("inf")), "epoch": max_epochs})


class ChemeleonHPO:
    """Orchestrator for CheMeleon hyperparameter optimization.

    This class manages the full HPO workflow:
    1. Builds Ray Tune search space from configuration
    2. Runs HPO trials with ASHA early stopping
    3. Logs results to MLflow
    4. Saves top-k configurations for downstream use

    Attributes
    ----------
    config : ChemeleonHPOConfig
        HPO configuration
    results : tune.ResultGrid | None
        Ray Tune results after running HPO
    """

    def __init__(self, config: ChemeleonHPOConfig) -> None:
        """Initialize the HPO orchestrator.

        Parameters
        ----------
        config : ChemeleonHPOConfig
            HPO configuration dataclass
        """
        self.config = config
        self.results: tune.ResultGrid | None = None
        self._mlflow_run_id: str | None = None
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def run(self) -> tune.ResultGrid:
        """Run hyperparameter optimization.

        Returns
        -------
        tune.ResultGrid
            Ray Tune ResultGrid containing all trial results
        """
        self._setup_mlflow()
        search_space = self._build_search_space()
        scheduler = self._build_scheduler()

        tune_config = tune.TuneConfig(
            scheduler=scheduler,
            num_samples=self.config.resources.num_samples,
            max_concurrent_trials=self.config.resources.max_concurrent_trials,
            trial_dirname_creator=_trial_dirname_creator,
        )

        trainable = tune.with_resources(
            train_chemeleon_trial,
            resources={
                "cpu": self.config.resources.cpus_per_trial,
                "gpu": self.config.resources.gpus_per_trial,
            },
        )

        storage_path = self.config.ray_storage_path
        if storage_path is None:
            storage_path = str(Path(self.config.output_dir) / "ray_results")
        storage_path = str(Path(storage_path).resolve())

        ray_temp_dir = str(Path(storage_path) / "_ray_tmp")
        Path(ray_temp_dir).mkdir(parents=True, exist_ok=True)

        import ray

        if not ray.is_initialized():
            ray.init(_temp_dir=ray_temp_dir, include_dashboard=False)
            logger.info("Ray initialized with temp dir: %s", ray_temp_dir)

        logger.info(
            "Starting CheMeleon HPO: %d trials, metric=%s, mode=%s",
            self.config.resources.num_samples,
            self.config.asha.metric,
            self.config.asha.mode,
        )

        tags: dict[str, str] = {"parent_run_id": self._mlflow_run_id or ""}
        if self._mlflow_run_id:
            tags["mlflow.parentRunId"] = self._mlflow_run_id

        mlflow_callback = MLflowLoggerCallback(
            tracking_uri=mlflow.get_tracking_uri(),
            experiment_name=self.config.experiment_name,
            save_artifact=True,
            tags=tags,
        )

        tuner = tune.Tuner(
            trainable,
            param_space=search_space,
            tune_config=tune_config,
            run_config=tune.RunConfig(
                name=self.config.experiment_name,
                storage_path=storage_path,
                verbose=1,
                callbacks=[mlflow_callback],
                sync_config=tune.SyncConfig(),
            ),
        )

        try:
            self.results = tuner.fit()
        except Exception as e:
            logger.error("HPO failed or interrupted: %s", e)
            try:
                logger.info("Attempting to restore Tuner to retrieve partial results...")
                tuner = tune.Tuner.restore(
                    path=str(Path(storage_path) / self.config.experiment_name),
                    trainable=trainable,
                )
                self.results = tuner.get_results()
            except Exception as restore_error:
                logger.warning("Could not restore Tuner results: %s", restore_error)
        finally:
            if self.results:
                self._log_results()
            else:
                logger.warning("No results to log to MLflow.")
                if self._mlflow_run_id:
                    mlflow.end_run()

        if self.results is None:
            raise RuntimeError("HPO failed to produce any results.")

        return self.results

    def _build_search_space(self) -> dict[str, Any]:
        """Build the Ray Tune search space."""
        space = build_chemeleon_search_space(self.config.search_space)

        # Add fixed parameters
        space["data_path"] = str(Path(self.config.data_path).resolve())
        space["val_data_path"] = str(Path(self.config.val_data_path).resolve()) if self.config.val_data_path else None
        space["smiles_column"] = self.config.smiles_column
        space["target_columns"] = self.config.target_columns
        space["max_epochs"] = self.config.asha.max_t
        space["metric"] = self.config.asha.metric
        space["seed"] = self.config.seed
        space["checkpoint_path"] = self.config.checkpoint_path
        space["freeze_encoder"] = self.config.freeze_encoder

        return space

    def _build_scheduler(self) -> ASHAScheduler:
        """Build the ASHA scheduler."""
        return ASHAScheduler(
            time_attr="epoch",
            metric=self.config.asha.metric,
            mode=self.config.asha.mode,
            max_t=self.config.asha.max_t,
            grace_period=self.config.asha.grace_period,
            reduction_factor=self.config.asha.reduction_factor,
        )

    def _setup_mlflow(self) -> None:
        """Setup MLflow tracking."""
        if self.config.mlflow_tracking_uri:
            mlflow.set_tracking_uri(self.config.mlflow_tracking_uri)

        mlflow.set_experiment(self.config.experiment_name)

        run = mlflow.start_run(run_name=f"hpo_{self.timestamp}")
        self._mlflow_run_id = run.info.run_id

        mlflow.log_params(
            {
                "num_samples": self.config.resources.num_samples,
                "max_epochs": self.config.asha.max_t,
                "grace_period": self.config.asha.grace_period,
                "checkpoint_path": self.config.checkpoint_path,
                "freeze_encoder": self.config.freeze_encoder,
            }
        )

    def _log_results(self) -> None:
        """Log HPO results to MLflow."""
        if self.results is None:
            return

        try:
            best_result = self.results.get_best_result(
                metric=self.config.asha.metric,
                mode=self.config.asha.mode,
            )

            if best_result and best_result.config:
                mlflow.log_params({f"best_{k}": v for k, v in best_result.config.items() if not k.startswith("_")})

            if best_result and best_result.metrics:
                mlflow.log_metrics(
                    {f"best_{k}": v for k, v in best_result.metrics.items() if isinstance(v, (int, float))}
                )

        except Exception as e:
            logger.warning("Could not log best result: %s", e)

        mlflow.end_run()

    def save_top_configs(self, k: int | None = None) -> list[dict[str, Any]]:
        """Save top-k configurations to file.

        Parameters
        ----------
        k : int | None
            Number of top configs to save. Defaults to transfer_learning.top_k.

        Returns
        -------
        list[dict[str, Any]]
            List of top configurations.
        """
        if self.results is None:
            raise RuntimeError("No HPO results available.")

        k = k or self.config.transfer_learning.top_k

        top_configs = []
        for result in self.results:
            if result.config:
                top_configs.append(
                    {
                        "config": result.config,
                        "metrics": result.metrics,
                    }
                )

        # Sort by metric
        metric = self.config.asha.metric
        reverse = self.config.asha.mode == "max"
        top_configs.sort(key=lambda x: x["metrics"].get(metric, float("inf")), reverse=reverse)
        top_configs = top_configs[:k]

        # Save to file
        output_path = Path(self.config.output_dir) / f"top_{k}_configs_{self.timestamp}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(top_configs, f, indent=2, default=str)

        logger.info("Saved top %d configs to %s", k, output_path)
        return top_configs


def main() -> None:
    """CLI entry point for CheMeleon HPO."""
    configure_logging()

    parser = argparse.ArgumentParser(description="Run CheMeleon HPO with Ray Tune")
    parser.add_argument("--config", type=str, required=True, help="Path to HPO config YAML")
    args = parser.parse_args()

    config_dict = OmegaConf.load(args.config)
    config = OmegaConf.structured(ChemeleonHPOConfig(**config_dict))

    hpo = ChemeleonHPO(config)
    results = hpo.run()

    logger.info("HPO complete. %d trials finished.", len(results))
    hpo.save_top_configs()


if __name__ == "__main__":
    main()
