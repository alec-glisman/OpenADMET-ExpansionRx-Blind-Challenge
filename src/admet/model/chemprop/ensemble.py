"""
Ensemble training for Chemprop models.

This module provides the ChempropEnsemble class for training multiple Chemprop
models across different data splits and folds, with Ray-based parallelization
and MLflow nested run tracking.

Key features:
- Automatic discovery of split/fold directory structure
- Ray-based parallel training with configurable concurrency
- MLflow nested runs for organized experiment tracking
- Ensemble prediction aggregation with uncertainty estimates
- Visualization with error bars from ensemble variance

Examples
--------
>>> from omegaconf import OmegaConf
>>> from admet.model.chemprop.ensemble import ChempropEnsemble
>>> from admet.model.chemprop.config import EnsembleConfig
>>>
>>> config = OmegaConf.merge(
...     OmegaConf.structured(EnsembleConfig),
...     OmegaConf.load("configs/ensemble_chemprop.yaml")
... )
>>> ensemble = ChempropEnsemble.from_config(config)
>>> ensemble.train_all()
>>> predictions = ensemble.predict_ensemble(test_df)
>>> ensemble.close()
"""

from __future__ import annotations

import logging
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import numpy as np
import pandas as pd
import ray
from matplotlib import pyplot as plt
from mlflow import MlflowClient
from omegaconf import DictConfig, OmegaConf

from admet.model.chemprop.config import (
    ChempropConfig,
    DataConfig,
    EnsembleConfig,
    InterTaskAffinityConfig,
    MlflowConfig,
    ModelConfig,
    OptimizationConfig,
    TaskAffinityConfig,
)
from admet.model.chemprop.model import ChempropModel
from admet.plot.latex import latex_sanitize
from admet.plot.metrics import plot_metric_bar
from admet.plot.parity import plot_parity
from admet.util.utils import parse_data_dir_params

# Configure module-level logger
logger = logging.getLogger("admet.model.chemprop.ensemble")


def _sanitize_metric_label(label: str) -> str:
    """
    Sanitize label for use in MLflow metric names.

    Converts labels to lowercase and replaces special characters with underscores
    to ensure valid MLflow metric names.

    Parameters
    ----------
    label : str
        Raw label (e.g., "Log KSOL", "Spearman $\\rho$")

    Returns
    -------
    str
        Sanitized label (e.g., "log_ksol", "spearman_rho")

    Examples
    --------
    >>> _sanitize_metric_label("Log KSOL")
    'log_ksol'
    >>> _sanitize_metric_label("Spearman $\\rho$")
    'spearman_rho'
    >>> _sanitize_metric_label("$R^2$")
    'r2'
    """
    return (
        label.lower()
        .replace(" ", "_")
        .replace(">", "gt")
        .replace("<", "lt")
        .replace("-", "_")
        .replace("$", "")
        .replace("^", "")
        .replace("\\", "")
        .replace("ρ", "rho")
        .replace("τ", "tau")
        .replace("²", "2")
    )


@dataclass
class SplitFoldInfo:
    """Information about a single split/fold combination."""

    split_idx: int
    fold_idx: int
    data_dir: Path
    train_file: Path
    validation_file: Path


class ChempropEnsemble:
    """
    Ensemble trainer for Chemprop models across multiple splits and folds.

    This class orchestrates training of multiple Chemprop models using Ray
    for parallelization and MLflow for experiment tracking with nested runs.

    Parameters
    ----------
    config : EnsembleConfig or DictConfig
        Configuration object containing data paths, model architecture,
        optimization settings, and MLflow configuration.

    Attributes
    ----------
    config : EnsembleConfig
        The ensemble configuration.
    split_fold_infos : List[SplitFoldInfo]
        Discovered split/fold combinations.
    models : Dict[str, ChempropModel]
        Trained models keyed by "split_{i}_fold_{j}".
    parent_run_id : str or None
        MLflow parent run ID for nested runs.

    Examples
    --------
    >>> ensemble = ChempropEnsemble.from_config(config)
    >>> ensemble.discover_splits_folds()
    >>> ensemble.train_all(max_parallel=2)
    >>> predictions = ensemble.predict_ensemble(test_df)
    """

    def __init__(
        self,
        config: Union[EnsembleConfig, DictConfig],
    ) -> None:
        """
        Initialize ChempropEnsemble with configuration.

        Parameters
        ----------
        config : EnsembleConfig or DictConfig
            Ensemble configuration object.
        """
        self.config = config
        self.split_fold_infos: List[SplitFoldInfo] = []
        self.models: Dict[str, ChempropModel] = {}
        self.predictions: Dict[str, pd.DataFrame] = {}

        # MLflow tracking
        self.parent_run_id: Optional[str] = None
        self._mlflow_client: Optional[MlflowClient] = None
        self._temp_dir: Optional[Path] = None

        # Initialize MLflow
        if self.config.mlflow.tracking:
            self._init_mlflow()

    @classmethod
    def from_config(
        cls,
        config: Union[EnsembleConfig, DictConfig],
    ) -> "ChempropEnsemble":
        """
        Create a ChempropEnsemble from configuration.

        Parameters
        ----------
        config : EnsembleConfig or DictConfig
            Configuration loaded from YAML or created programmatically.

        Returns
        -------
        ChempropEnsemble
            Initialized ensemble ready for training.
        """
        ensemble = cls(config)
        ensemble.discover_splits_folds()
        return ensemble

    def _init_mlflow(self) -> None:
        """Initialize MLflow tracking with parent run."""
        if self.config.mlflow.tracking_uri:
            mlflow.set_tracking_uri(self.config.mlflow.tracking_uri)

        mlflow.set_experiment(self.config.mlflow.experiment_name)
        self._mlflow_client = MlflowClient()

        # Start parent run (use configured name or let MLflow generate default)
        parent_run = mlflow.start_run(run_name=self.config.mlflow.run_name)
        self.parent_run_id = parent_run.info.run_id

        # Log ensemble configuration
        config_dict = OmegaConf.to_container(self.config, resolve=True)
        mlflow.log_params(self._flatten_dict(config_dict, max_depth=2))  # type: ignore[arg-type]

        # Parse and log data_dir parameters

        data_params = parse_data_dir_params(self.config.data.data_dir)
        for key, value in data_params.items():
            if value is not None:
                mlflow.log_param(f"data.{key}", value)

        logger.info("Started MLflow parent run: %s", self.parent_run_id)

    def _flatten_dict(
        self, d: Dict[str, Any], parent_key: str = "", sep: str = ".", max_depth: int = 3
    ) -> Dict[str, Any]:
        """Flatten nested dictionary for MLflow params."""
        items: List[Tuple[str, Any]] = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict) and max_depth > 0:
                items.extend(self._flatten_dict(v, new_key, sep, max_depth - 1).items())
            else:
                # Truncate long values
                str_val = str(v)
                if len(str_val) > 250:
                    str_val = str_val[:247] + "..."
                items.append((new_key, str_val))
        return dict(items)

    def discover_splits_folds(self) -> List[SplitFoldInfo]:
        """
        Discover available split/fold combinations in data directory.

        Scans the data directory for subdirectories matching the pattern
        split_*/fold_*/ containing train.csv and validation.csv files.

        Returns
        -------
        List[SplitFoldInfo]
            List of discovered split/fold combinations.

        Raises
        ------
        FileNotFoundError
            If data directory does not exist.
        ValueError
            If no valid split/fold combinations are found.
        """
        data_dir = Path(self.config.data.data_dir)
        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        self.split_fold_infos = []
        split_pattern = re.compile(r"split_(\d+)")
        fold_pattern = re.compile(r"fold_(\d+)")

        # Find all split directories and sort by numeric split index
        split_dirs = []
        for p in data_dir.iterdir():
            if not p.is_dir():
                continue
            m = split_pattern.match(p.name)
            if not m:
                continue
            split_dirs.append((int(m.group(1)), p))
        split_dirs.sort(key=lambda x: x[0])
        for split_idx, split_dir in split_dirs:
            if not split_dir.is_dir():
                continue
            split_match = split_pattern.match(split_dir.name)
            if not split_match:
                continue
            split_idx = int(split_match.group(1))

            # Filter by specified splits
            if self.config.data.splits is not None:
                if split_idx not in self.config.data.splits:
                    continue

            # Find all fold directories within this split
            # Find all fold directories and sort by numeric fold index
            fold_dirs = []
            for p in split_dir.iterdir():
                if not p.is_dir():
                    continue
                m = fold_pattern.match(p.name)
                if not m:
                    continue
                fold_dirs.append((int(m.group(1)), p))
            fold_dirs.sort(key=lambda x: x[0])
            for fold_idx, fold_dir in fold_dirs:
                if not fold_dir.is_dir():
                    continue
                # fold_idx and fold_dir are set from fold_dirs (see above)

                # Filter by specified folds
                if self.config.data.folds is not None:
                    if fold_idx not in self.config.data.folds:
                        continue

                train_file = fold_dir / "train.csv"
                val_file = fold_dir / "validation.csv"

                if train_file.exists() and val_file.exists():
                    self.split_fold_infos.append(
                        SplitFoldInfo(
                            split_idx=split_idx,
                            fold_idx=fold_idx,
                            data_dir=fold_dir,
                            train_file=train_file,
                            validation_file=val_file,
                        )
                    )
                else:
                    logger.warning("Missing train.csv or validation.csv in %s", fold_dir)

        if not self.split_fold_infos:
            raise ValueError(
                f"No valid split/fold combinations found in {data_dir}. "
                "Expected structure: split_*/fold_*/{{train,validation}}.csv"
            )

        logger.info("Discovered %d split/fold combinations", len(self.split_fold_infos))
        for info in self.split_fold_infos:
            logger.debug("  split_%d/fold_%d: %s", info.split_idx, info.fold_idx, info.data_dir)

        return self.split_fold_infos

    def _create_single_model_config(self, split_fold_info: SplitFoldInfo) -> ChempropConfig:
        """
        Create a single-model config for a specific split/fold.

        Parameters
        ----------
        split_fold_info : SplitFoldInfo
            Information about the split/fold to configure.
        run_id : str, optional
            MLflow run ID to attach to. If provided, the model will log
            to this existing run instead of creating a new one.

        Returns
        -------
        ChempropConfig
            Configuration for training a single model.
        """
        return ChempropConfig(
            data=DataConfig(
                data_dir=str(split_fold_info.data_dir),
                test_file=self.config.data.test_file,
                blind_file=self.config.data.blind_file,
                smiles_col=self.config.data.smiles_col,
                target_cols=list(self.config.data.target_cols),
                target_weights=list(self.config.data.target_weights) if self.config.data.target_weights else [],
                output_dir=self.config.data.output_dir,
            ),
            model=ModelConfig(
                depth=self.config.model.depth,
                message_hidden_dim=self.config.model.message_hidden_dim,
                dropout=self.config.model.dropout,
                num_layers=self.config.model.num_layers,
                hidden_dim=self.config.model.hidden_dim,
                batch_norm=self.config.model.batch_norm,
                ffn_type=self.config.model.ffn_type,
                trunk_n_layers=self.config.model.trunk_n_layers,
                trunk_hidden_dim=self.config.model.trunk_hidden_dim,
                n_experts=self.config.model.n_experts,
            ),
            optimization=OptimizationConfig(
                criterion=self.config.optimization.criterion,
                init_lr=self.config.optimization.init_lr,
                max_lr=self.config.optimization.max_lr,
                final_lr=self.config.optimization.final_lr,
                warmup_epochs=self.config.optimization.warmup_epochs,
                patience=self.config.optimization.patience,
                max_epochs=self.config.optimization.max_epochs,
                batch_size=self.config.optimization.batch_size,
                num_workers=self.config.optimization.num_workers,
                seed=self.config.optimization.seed,
                progress_bar=self.config.optimization.progress_bar,
            ),
            mlflow=MlflowConfig(
                tracking=True,  # Enable MLflow tracking in model
                tracking_uri=self.config.mlflow.tracking_uri,
                experiment_name=self.config.mlflow.experiment_name,
                run_name=f"split_{split_fold_info.split_idx}_fold_{split_fold_info.fold_idx}",
                run_id=None,  # Model will create its own nested run
                parent_run_id=None,  # Will be set by caller for ensemble runs
                nested=False,  # Will be set by caller for ensemble runs
            ),
            joint_sampling=self.config.joint_sampling,
            task_affinity=TaskAffinityConfig(
                enabled=self.config.task_affinity.enabled,
                affinity_epochs=self.config.task_affinity.affinity_epochs,
                affinity_batch_size=self.config.task_affinity.affinity_batch_size,
                affinity_lr=self.config.task_affinity.affinity_lr,
                n_groups=self.config.task_affinity.n_groups,
                clustering_method=self.config.task_affinity.clustering_method,
                affinity_type=self.config.task_affinity.affinity_type,
                seed=self.config.task_affinity.seed,
            ),
            inter_task_affinity=InterTaskAffinityConfig(
                enabled=self.config.inter_task_affinity.enabled,
                compute_every_n_steps=self.config.inter_task_affinity.compute_every_n_steps,
                log_every_n_steps=self.config.inter_task_affinity.log_every_n_steps,
                log_epoch_summary=self.config.inter_task_affinity.log_epoch_summary,
                log_step_matrices=self.config.inter_task_affinity.log_step_matrices,
                lookahead_lr=self.config.inter_task_affinity.lookahead_lr,
                use_optimizer_lr=self.config.inter_task_affinity.use_optimizer_lr,
                shared_param_patterns=list(self.config.inter_task_affinity.shared_param_patterns),
                exclude_param_patterns=list(self.config.inter_task_affinity.exclude_param_patterns),
                n_groups=self.config.inter_task_affinity.n_groups,
                clustering_method=self.config.inter_task_affinity.clustering_method,
                clustering_linkage=self.config.inter_task_affinity.clustering_linkage,
                device=self.config.inter_task_affinity.device,
                log_to_mlflow=self.config.inter_task_affinity.log_to_mlflow,
                save_plots=self.config.inter_task_affinity.save_plots,
                plot_formats=list(self.config.inter_task_affinity.plot_formats),
                plot_dpi=self.config.inter_task_affinity.plot_dpi,
            ),
        )

    def train_all(self, max_parallel: Optional[int] = None) -> None:
        """
        Train all ensemble models with Ray parallelization.

        Parameters
        ----------
        max_parallel : int, optional
            Maximum number of models to train in parallel.
            Overrides config.ray.max_parallel if provided.
        """
        max_parallel = max_parallel or self.config.ray.max_parallel

        if not self.split_fold_infos:
            self.discover_splits_folds()

        logger.info(
            "Training %d models with max_parallel=%d",
            len(self.split_fold_infos),
            max_parallel,
        )

        # Initialize Ray
        ray_kwargs: Dict[str, Any] = {}
        if self.config.ray.num_cpus is not None:
            ray_kwargs["num_cpus"] = self.config.ray.num_cpus
        if self.config.ray.num_gpus is not None:
            ray_kwargs["num_gpus"] = self.config.ray.num_gpus

        if not ray.is_initialized():
            ray.init(**ray_kwargs, ignore_reinit_error=True)
            logger.info("Initialized Ray cluster")

        # Ensure max_parallel is set
        if max_parallel is None or max_parallel < 1:
            max_parallel = 1

        # Calculate GPU fraction per task
        gpu_fraction = 1.0 / max_parallel if max_parallel > 1 else 1.0

        # Create training tasks
        @ray.remote(num_gpus=gpu_fraction)
        def train_single_model(
            config_dict: Dict[str, Any],
            split_idx: int,
            fold_idx: int,
            parent_run_id: Optional[str],
            tracking_uri: Optional[str],
            experiment_name: str,
        ) -> Tuple[str, Dict[str, float], Optional[pd.DataFrame], Optional[pd.DataFrame]]:
            """
            Train a single model as a Ray task.

            The model creates its own nested MLflow run under the parent run,
            so all artifacts (checkpoints, metrics, plots) are logged directly.
            """
            # Reconstruct config from dict - use deep merge to ensure nested values override defaults
            base_config = OmegaConf.structured(ChempropConfig)
            override_config = OmegaConf.create(config_dict)
            config = OmegaConf.merge(base_config, override_config)

            # Force resolve to ensure all values are concrete
            OmegaConf.resolve(config)

            model_key = f"split_{split_idx}_fold_{fold_idx}"

            # Debug: Log key config values to verify they match YAML
            import logging

            _logger = logging.getLogger("admet.model.chemprop.ensemble")
            _logger.info(
                "[%s] Config values - depth=%s, dropout=%s, hidden_dim=%s, "
                "batch_size=%s, max_lr=%s, joint_sampling=%s",
                model_key,
                config.model.depth,
                config.model.dropout,
                config.model.hidden_dim,
                config.optimization.batch_size,
                config.optimization.max_lr,
                config.joint_sampling.enabled if config.joint_sampling else False,
            )

            # Configure model to create a nested MLflow run
            config.mlflow.parent_run_id = parent_run_id
            config.mlflow.nested = parent_run_id is not None

            # Create and train model (MLflow enabled, creating nested run)
            model = ChempropModel.from_config(config)  # type: ignore[arg-type]
            model.fit()

            # Get metrics from trainer
            metrics = {}
            if model.trainer and model.trainer.callback_metrics:
                for key, val in model.trainer.callback_metrics.items():
                    if hasattr(val, "item"):
                        metrics[key] = val.item()
                    else:
                        metrics[key] = float(val)

            # Get predictions for test and blind
            test_preds = None
            blind_preds = None
            smiles_col = config.data.smiles_col
            target_cols = list(config.data.target_cols)

            if model.dataframes["test"] is not None:
                test_df = model.dataframes["test"]
                pred_df = model.predict(
                    test_df,
                    generate_plots=True,  # Generate plots for each model
                    split_name="test",
                )
                # Prepend SMILES and Molecule Name columns to predictions if present
                if "Molecule Name" in test_df.columns:
                    pred_df["Molecule Name"] = test_df["Molecule Name"].values
                    cols = pred_df.columns.tolist()
                    cols.insert(0, cols.pop(cols.index("Molecule Name")))
                    pred_df = pred_df[cols]

                if smiles_col in test_df.columns:
                    pred_df[smiles_col] = test_df[smiles_col].values
                    cols = pred_df.columns.tolist()
                    cols.insert(0, cols.pop(cols.index(smiles_col)))
                    pred_df = pred_df[cols]

                test_preds = pred_df.copy()
                test_preds[smiles_col] = test_df[smiles_col].values
                for col in target_cols:
                    if col in test_df.columns:
                        test_preds[f"{col}_actual"] = test_df[col].values

            if model.dataframes["blind"] is not None:
                blind_df = model.dataframes["blind"]
                pred_df = model.predict(
                    blind_df,
                    generate_plots=False,  # No ground truth for blind
                    split_name="blind",
                )
                # Prepend SMILES and Molecule Name columns to predictions if present
                if "Molecule Name" in blind_df.columns:
                    pred_df["Molecule Name"] = blind_df["Molecule Name"].values
                    cols = pred_df.columns.tolist()
                    cols.insert(0, cols.pop(cols.index("Molecule Name")))
                    pred_df = pred_df[cols]

                if smiles_col in blind_df.columns:
                    pred_df[smiles_col] = blind_df[smiles_col].values
                    cols = pred_df.columns.tolist()
                    cols.insert(0, cols.pop(cols.index(smiles_col)))
                    pred_df = pred_df[cols]

                blind_preds = pred_df.copy()

            # Close model (ends nested MLflow run)
            model.close()

            return model_key, metrics, test_preds, blind_preds

        # Submit tasks in batches
        all_results = []
        pending_tasks = []
        task_infos = []

        from dataclasses import asdict

        for info in self.split_fold_infos:
            config = self._create_single_model_config(info)
            # Convert dataclass instance to plain dict for Ray serialization.
            # Use asdict to preserve values from the dataclass instance
            # (avoids accidentally using dataclass defaults via OmegaConf.structured).
            config_dict = asdict(config)

            task = train_single_model.remote(
                config_dict,  # type: ignore[arg-type]
                info.split_idx,
                info.fold_idx,
                self.parent_run_id,
                self.config.mlflow.tracking_uri,
                self.config.mlflow.experiment_name,
            )
            pending_tasks.append(task)
            task_infos.append(info)

            # Wait for batch completion if at capacity
            if len(pending_tasks) >= max_parallel:
                done_ids, pending_tasks = ray.wait(pending_tasks, num_returns=1, timeout=None)
                for done_id in done_ids:
                    result = ray.get(done_id)
                    all_results.append(result)
                    logger.info("Completed training: %s", result[0])

        # Wait for remaining tasks
        for task in pending_tasks:
            result = ray.get(task)
            all_results.append(result)
            logger.info("Completed training: %s", result[0])

        # Store predictions
        self._all_test_predictions: List[pd.DataFrame] = []
        self._all_blind_predictions: List[pd.DataFrame] = []
        self._all_metrics: Dict[str, Dict[str, float]] = {}

        for model_key, metrics, test_preds, blind_preds in all_results:
            self._all_metrics[model_key] = metrics
            if test_preds is not None:
                self._all_test_predictions.append(test_preds)
            if blind_preds is not None:
                self._all_blind_predictions.append(blind_preds)

        logger.info("Completed training all %d models", len(all_results))

        # Generate ensemble predictions and plots
        self._generate_ensemble_outputs()

    def _generate_ensemble_outputs(self) -> None:
        """Generate ensemble predictions, metrics, and plots."""
        # Create temp directory for outputs
        self._temp_dir = Path(tempfile.mkdtemp(prefix="ensemble_"))

        # Aggregate test predictions
        if self._all_test_predictions:
            test_ensemble = self._aggregate_predictions(self._all_test_predictions, split_name="test")
            self._save_ensemble_predictions(test_ensemble, "test")
            self._generate_ensemble_plots(test_ensemble, "test")
            self._generate_unlabeled_ensemble_plots(test_ensemble, "test")

        # Aggregate blind predictions
        if self._all_blind_predictions:
            blind_ensemble = self._aggregate_predictions(self._all_blind_predictions, split_name="blind")
            self._save_ensemble_predictions(blind_ensemble, "blind")
            self._generate_unlabeled_ensemble_plots(blind_ensemble, "blind")

        # Log ensemble metrics
        self._log_ensemble_metrics()

    def _aggregate_predictions(self, predictions_list: List[pd.DataFrame], split_name: str) -> pd.DataFrame:
        """
        Aggregate predictions from multiple models.

        Computes mean and standard error for each target column.
        For Log columns, applies 10^x transform after averaging.

        Parameters
        ----------
        predictions_list : List[pd.DataFrame]
            List of prediction DataFrames from individual models.
        split_name : str
            Name of the split (e.g., "test", "blind").

        Returns
        -------
        pd.DataFrame
            Aggregated predictions with mean, std, and stderr columns.
        """
        if not predictions_list:
            return pd.DataFrame()

        smiles_col = self.config.data.smiles_col
        target_cols = list(self.config.data.target_cols)

        # Stack predictions for each target
        result = pd.DataFrame()
        result[smiles_col] = predictions_list[0][smiles_col].copy()

        # Preserve Molecule Name if present
        if "Molecule Name" in predictions_list[0].columns:
            result["Molecule Name"] = predictions_list[0]["Molecule Name"].copy()

        # Also preserve any actual values if present
        for col in predictions_list[0].columns:
            if col.endswith("_actual"):
                result[col] = predictions_list[0][col].copy()

        n_models = len(predictions_list)

        for target in target_cols:
            # Collect predictions from all models
            pred_col = target  # Predictions are in target column
            preds = np.array([df[pred_col].values for df in predictions_list])

            # Calculate statistics
            mean_pred = np.mean(preds, axis=0)
            std_pred = np.std(preds, axis=0, ddof=1)
            stderr_pred = std_pred / np.sqrt(n_models)

            result[f"{target}_mean"] = mean_pred
            result[f"{target}_std"] = std_pred
            result[f"{target}_stderr"] = stderr_pred

            # For "Log " columns (with space), compute transformed values
            # Average first, then transform: 10^mean(x_i)
            # Note: "Log" (no space) columns are already in correct form, no transform needed
            if target.startswith("Log "):
                result[f"{target}_transformed_mean"] = np.power(10, mean_pred)
                # Propagate uncertainty through log transform
                # For y = 10^x, dy = ln(10) * 10^x * dx
                result[f"{target}_transformed_stderr"] = np.log(10) * np.power(10, mean_pred) * stderr_pred

        logger.info("Aggregated %d model predictions for %s split", n_models, split_name)
        return result

    def _save_ensemble_predictions(self, predictions: pd.DataFrame, split_name: str) -> None:
        """Save ensemble predictions to CSV files."""
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="ensemble_"))

        # Save detailed predictions
        detailed_path = self._temp_dir / f"{split_name}_ensemble_predictions.csv"
        predictions.to_csv(detailed_path, index=False)

        # Save submissions format (transformed values for Log columns)
        submissions = pd.DataFrame()
        smiles_col = self.config.data.smiles_col
        submissions[smiles_col] = predictions[smiles_col]

        # Include Molecule Name if present in predictions
        if "Molecule Name" in predictions.columns:
            submissions["Molecule Name"] = predictions["Molecule Name"]

        target_cols = list(self.config.data.target_cols)
        for target in target_cols:
            if target.startswith("Log "):
                # Use transformed mean for submissions, remove "Log " prefix
                clean_name = target.replace("Log ", "")
                submissions[clean_name] = predictions[f"{target}_transformed_mean"]
            else:
                # "Log" (no space) and other columns use mean directly
                submissions[target] = predictions[f"{target}_mean"]

        submissions_path = self._temp_dir / f"{split_name}_ensemble_submissions.csv"
        submissions.to_csv(submissions_path, index=False)

        # Log to MLflow
        if self._mlflow_client and self.parent_run_id:
            self._mlflow_client.log_artifact(self.parent_run_id, str(detailed_path), artifact_path="predictions")
            self._mlflow_client.log_artifact(self.parent_run_id, str(submissions_path), artifact_path="submissions")

        logger.info("Saved ensemble predictions for %s", split_name)

    def _generate_unlabeled_ensemble_plots(self, predictions: pd.DataFrame, split_name: str) -> None:
        """
        Generate ensemble visualizations when the predictions lack ground truth.

        Creates prediction distribution plots for mean predictions and
        standard errors, then logs them to MLflow.

        Parameters
        ----------
        predictions : pd.DataFrame
            Aggregated ensemble predictions with mean and stderr columns.
        split_name : str
            Name of the split (e.g., "test", "blind").
        """
        from admet.plot.density import plot_endpoint_distributions

        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="ensemble_"))

        plot_dir = self._temp_dir / f"plots_{split_name}"
        plot_dir.mkdir(exist_ok=True)

        target_cols = list(self.config.data.target_cols)

        # Collect mean and stderr columns for distribution plots
        mean_cols = [f"{target}_mean" for target in target_cols]
        stderr_cols = [f"{target}_stderr" for target in target_cols]

        # Plot mean prediction distributions
        mean_plot_path = plot_dir / "prediction_distributions.png"
        fig, _ = plot_endpoint_distributions(
            predictions,
            columns=mean_cols,
            title=f"Ensemble Mean Predictions ({split_name})",
            save_path=mean_plot_path,
        )
        plt.close(fig)

        # Plot standard error distributions
        stderr_plot_path = plot_dir / "uncertainty_distributions.png"
        fig, _ = plot_endpoint_distributions(
            predictions,
            columns=stderr_cols,
            title=f"Ensemble Standard Errors ({split_name})",
            save_path=stderr_plot_path,
        )
        plt.close(fig)

        # Log plots to MLflow
        if self._mlflow_client and self.parent_run_id:
            for plot_file in plot_dir.iterdir():
                if plot_file.is_file():
                    self._mlflow_client.log_artifact(
                        self.parent_run_id,
                        str(plot_file),
                        artifact_path=f"plots/{split_name}",
                    )

        logger.info("Generated unlabeled ensemble plots for %s", split_name)

    def _log_plot_metrics(
        self,
        split_name: str,
        metric_type: str,
        safe_metric: str,
        labels: List[str],
        means: List[float],
        errors: List[float],
        n_models: int,
    ) -> None:
        """
        Log bar plot data as MLflow metrics under plots/ prefix using batch API.

        This method logs all bar values and standard errors in a single batch operation
        for better performance. It also includes metadata about the ensemble size.

        Parameters
        ----------
        split_name : str
            Split name (e.g., "test", "blind")
        metric_type : str
            Display metric type (e.g., "MAE", r"$R^2$")
        safe_metric : str
            Sanitized metric name for MLflow (e.g., "MAE", "R2")
        labels : List[str]
            Bar labels (target names + "Mean")
        means : List[float]
            Bar values (metric means across models)
        errors : List[float]
            Error bar values (standard errors)
        n_models : int
            Number of models in the ensemble

        Notes
        -----
        Metrics are logged under the hierarchical structure:
        - plots/{split_name}/{safe_metric}/{target}
        - plots/{split_name}/{safe_metric}/{target}_stderr
        - plots/{split_name}/{safe_metric}/n_models

        NaN values are skipped with a warning. All metrics are logged in a single
        batch call for performance.
        """
        if not self._mlflow_client or not self.parent_run_id:
            return

        # Prepare batch metrics dictionary
        metrics_dict = {}

        # Log each bar's value and stderr
        for label, mean_val, stderr_val in zip(labels, means, errors):
            # Handle NaN values
            if np.isnan(mean_val):
                logger.warning(f"Skipping NaN metric for {label} in {metric_type} ({split_name})")
                continue

            # Sanitize label using shared utility
            safe_label = _sanitize_metric_label(label)

            # Add to batch
            metrics_dict[f"plots/{split_name}/{safe_metric}/{safe_label}"] = float(mean_val)
            metrics_dict[f"plots/{split_name}/{safe_metric}/{safe_label}_stderr"] = float(stderr_val)

        # Add metadata: number of models
        metrics_dict[f"plots/{split_name}/{safe_metric}/n_models"] = float(n_models)

        # Log all metrics in a single batch
        try:
            mlflow.log_metrics(metrics_dict)
            logger.debug(f"Logged {len(metrics_dict)} plot metrics for {safe_metric} ({split_name})")
        except Exception as e:
            logger.warning(f"Failed to log plot metrics for {safe_metric} ({split_name}): {e}")

    def _generate_ensemble_plots(self, predictions: pd.DataFrame, split_name: str) -> None:
        """
        Generate ensemble visualizations with error bars.

        Creates parity plots showing mean prediction with uncertainty bands
        and bar plots showing metric values with standard errors.

        Parameters
        ----------
        predictions : pd.DataFrame
            Aggregated ensemble predictions.
        split_name : str
            Name of the split for labeling.
        """
        if self._temp_dir is None:
            self._temp_dir = Path(tempfile.mkdtemp(prefix="ensemble_"))

        plot_dir = self._temp_dir / f"plots_{split_name}"
        plot_dir.mkdir(exist_ok=True)

        target_cols = list(self.config.data.target_cols)

        # Generate parity plots with error bands for each target
        for target in target_cols:
            actual_col = f"{target}_actual"
            if actual_col not in predictions.columns:
                continue

            actual = predictions[actual_col].values
            mean_pred = predictions[f"{target}_mean"].values
            stderr = predictions[f"{target}_stderr"].values

            # Use the shared plot_parity function with yerr support
            fig, _ = plot_parity(
                actual,
                mean_pred,
                yerr=stderr,
                title=f"Ensemble: {latex_sanitize(target)} ({split_name})",
                xlabel=f"Actual {latex_sanitize(target)}",
                ylabel=f"Predicted {latex_sanitize(target)}",
                show_stats=True,
                alpha=0.6,
                s=16,
            )

            plot_path = plot_dir / f"parity_{target.replace(' ', '_')}.png"
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")
            plt.close(fig)

        # Generate bar plot of metrics computed from predictions
        self._generate_metrics_bar_plot(plot_dir, split_name, predictions)

        # Log plots to MLflow
        if self._mlflow_client and self.parent_run_id:
            for plot_file in plot_dir.iterdir():
                if plot_file.is_file():
                    self._mlflow_client.log_artifact(
                        self.parent_run_id,
                        str(plot_file),
                        artifact_path=f"plots/{split_name}",
                    )

        logger.info("Generated ensemble plots for %s", split_name)

    def _generate_metrics_bar_plot(self, plot_dir: Path, split_name: str, predictions: pd.DataFrame) -> None:
        """Generate bar plots of metrics computed from individual model predictions.

        Creates separate bar plots for each metric type (MAE, RMSE, R², RAE,
        Spearman ρ, Pearson r, Kendall τ), showing the mean value across all
        fold/split models with error bars representing the standard error.

        Parameters
        ----------
        plot_dir : Path
            Directory to save the plots.
        split_name : str
            Name of the split for labeling.
        predictions : pd.DataFrame
            Aggregated ensemble predictions (used for target column names).
        """
        from scipy.stats import kendalltau, pearsonr, spearmanr
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        # Get individual model predictions for computing per-model metrics
        if split_name == "test":
            model_predictions = self._all_test_predictions
        else:
            return  # No individual predictions for blind set

        if not model_predictions:
            return

        target_cols = list(self.config.data.target_cols)

        # Compute metrics for each model and each target
        # Structure: {metric_type: {target: [values across models]}}
        metrics_by_type: Dict[str, Dict[str, List[float]]] = {
            "MAE": {},
            "RMSE": {},
            r"$R^2$": {},
            "RAE": {},
            r"Spearman $\rho$": {},
            r"Pearson $r$": {},
            r"Kendall $\tau$": {},
        }

        # Mapping from display names to sanitized MLflow names
        metric_name_map = {
            "MAE": "MAE",
            "RMSE": "RMSE",
            r"$R^2$": "R2",
            "RAE": "RAE",
            r"Spearman $\rho$": "spearman_rho",
            r"Pearson $r$": "pearson_r",
            r"Kendall $\tau$": "kendall_tau",
        }

        for target in target_cols:
            actual_col = f"{target}_actual"

            # Initialize lists for each target
            for metric_type in metrics_by_type:
                metrics_by_type[metric_type][target] = []

            # Compute metrics for each model's predictions
            for model_preds in model_predictions:
                if actual_col not in model_preds.columns or target not in model_preds.columns:
                    continue

                actual = model_preds[actual_col].values
                pred = model_preds[target].values

                # Remove NaN values
                mask = ~(np.isnan(actual) | np.isnan(pred))
                if mask.sum() == 0:
                    continue

                actual_clean = actual[mask]
                pred_clean = pred[mask]

                # Compute error metrics for this model
                mae = mean_absolute_error(actual_clean, pred_clean)
                rmse = np.sqrt(mean_squared_error(actual_clean, pred_clean))
                r2 = r2_score(actual_clean, pred_clean)

                # Relative Absolute Error (RAE): sum(|pred - actual|) / sum(|actual - mean(actual)|)
                baseline_error = np.sum(np.abs(actual_clean - np.mean(actual_clean)))
                if baseline_error > 0:
                    rae = np.sum(np.abs(pred_clean - actual_clean)) / baseline_error
                else:
                    rae = np.nan

                # Correlation metrics
                spearman_rho, _ = spearmanr(actual_clean, pred_clean)
                pearson_r, _ = pearsonr(actual_clean, pred_clean)
                kendall_tau, _ = kendalltau(actual_clean, pred_clean)

                metrics_by_type["MAE"][target].append(mae)
                metrics_by_type["RMSE"][target].append(rmse)
                metrics_by_type[r"$R^2$"][target].append(r2)
                metrics_by_type["RAE"][target].append(rae)
                metrics_by_type[r"Spearman $\rho$"][target].append(spearman_rho)
                metrics_by_type[r"Pearson $r$"][target].append(pearson_r)
                metrics_by_type[r"Kendall $\tau$"][target].append(kendall_tau)

        # Generate a separate plot for each metric type and log metrics to MLflow
        for metric_type, target_metrics in metrics_by_type.items():
            labels = []
            means = []
            errors = []
            all_values_for_mean = []  # Collect all values for computing overall mean

            for target in target_cols:
                if target not in target_metrics or not target_metrics[target]:
                    continue

                values = target_metrics[target]
                # Filter out NaN values (e.g., RAE when baseline is zero)
                values = [v for v in values if not np.isnan(v)]
                if not values:
                    continue

                # Use clean target name (remove "Log " prefix for display)
                clean_target = target.replace("Log ", "")

                labels.append(clean_target)
                mean_val = np.mean(values)
                stderr_val = np.std(values, ddof=1) / np.sqrt(len(values))
                means.append(mean_val)
                errors.append(stderr_val)
                all_values_for_mean.extend(values)

                # Log individual target metrics to MLflow under test/ prefix
                if self._mlflow_client and self.parent_run_id:
                    safe_metric = metric_name_map.get(metric_type, metric_type)
                    safe_target = clean_target.replace(" ", "_").replace(">", "gt").replace("<", "lt").replace("-", "_")
                    try:
                        self._mlflow_client.log_metric(
                            self.parent_run_id,
                            f"{split_name}/{safe_target}_{safe_metric}",
                            float(mean_val),
                        )
                        self._mlflow_client.log_metric(
                            self.parent_run_id,
                            f"{split_name}/{safe_target}_{safe_metric}_stderr",
                            float(stderr_val),
                        )
                    except Exception:
                        pass  # Silently ignore metric logging failures

            if not labels:
                continue

            # Add "Mean" bar with overall mean and stderr across all endpoints and models
            if all_values_for_mean:
                labels.append("Mean")
                overall_mean = np.mean(all_values_for_mean)
                overall_stderr = np.std(all_values_for_mean, ddof=1) / np.sqrt(len(all_values_for_mean))
                means.append(overall_mean)
                errors.append(overall_stderr)

                # Log overall mean metrics to MLflow under test/ prefix
                if self._mlflow_client and self.parent_run_id:
                    safe_metric = metric_name_map.get(metric_type, metric_type)
                    try:
                        self._mlflow_client.log_metric(
                            self.parent_run_id,
                            f"{split_name}/mean_{safe_metric}",
                            float(overall_mean),
                        )
                        self._mlflow_client.log_metric(
                            self.parent_run_id,
                            f"{split_name}/mean_{safe_metric}_stderr",
                            float(overall_stderr),
                        )
                    except Exception:
                        pass  # Silently ignore metric logging failures

            # Use shared plot_metric_bar function with error bars
            fig, _ = plot_metric_bar(
                np.array(means),
                labels,
                metric_name=metric_type,
                errors=np.array(errors),
                title=f"Ensemble {metric_type} ({split_name})\nMean ± Standard Error (n={len(model_predictions)})",
                show_mean=False,
            )

            # Save plot with metric type in filename (sanitize special characters)
            safe_metric_name = (
                metric_type.replace("$", "").replace(" ", "_").replace("^", "").replace("ρ", "rho").replace("τ", "tau")
            )
            plot_path = plot_dir / f"ensemble_{safe_metric_name.lower()}.png"
            fig.savefig(plot_path, dpi=300, bbox_inches="tight")

            # Log plot data as MLflow metrics under plots/ prefix
            safe_metric = metric_name_map.get(metric_type, metric_type)
            self._log_plot_metrics(
                split_name=split_name,
                metric_type=metric_type,
                safe_metric=safe_metric,
                labels=labels,
                means=[float(m) for m in means],
                errors=[float(e) for e in errors],
                n_models=len(model_predictions),
            )

            plt.close(fig)

    def _log_ensemble_metrics(self) -> None:
        """Log aggregated ensemble metrics to MLflow."""
        if not self._all_metrics or not self._mlflow_client or not self.parent_run_id:
            return

        # Calculate ensemble statistics for each metric
        metric_names: set[str] = set()
        for metrics in self._all_metrics.values():
            metric_names.update(metrics.keys())

        ensemble_metrics = {}
        for metric_name in metric_names:
            values = [m[metric_name] for m in self._all_metrics.values() if metric_name in m]
            if values:
                ensemble_metrics[f"ensemble_{metric_name}_mean"] = np.mean(values)
                ensemble_metrics[f"ensemble_{metric_name}_std"] = np.std(values, ddof=1)
                ensemble_metrics[f"ensemble_{metric_name}_stderr"] = np.std(values, ddof=1) / np.sqrt(len(values))

        mlflow.log_metrics({k: float(v) for k, v in ensemble_metrics.items()})
        logger.info("Logged ensemble metrics to MLflow")

    def predict_ensemble(
        self,
        df: pd.DataFrame,
        split_name: str = "prediction",
    ) -> pd.DataFrame:
        """
        Generate ensemble predictions for new data.

        This method requires that models have been trained via train_all().
        It uses the stored predictions if available, or raises an error.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing SMILES for prediction.
        split_name : str, default="prediction"
            Name for the prediction split.

        Returns
        -------
        pd.DataFrame
            Aggregated predictions with mean, std, and stderr columns.
        """
        # For now, we use stored predictions from training
        # In future, could support loading trained models and making new predictions
        if split_name == "test" and hasattr(self, "_all_test_predictions"):
            return self._aggregate_predictions(self._all_test_predictions, split_name)
        elif split_name == "blind" and hasattr(self, "_all_blind_predictions"):
            return self._aggregate_predictions(self._all_blind_predictions, split_name)
        else:
            raise ValueError(
                f"No predictions available for {split_name}. " "Call train_all() first to generate predictions."
            )

    def close(self) -> None:
        """Close MLflow run and clean up resources."""
        if self.parent_run_id:
            mlflow.end_run()
            logger.info("Ended MLflow parent run: %s", self.parent_run_id)

        if ray.is_initialized():
            ray.shutdown()
            logger.info("Shut down Ray cluster")

        # Clean up temp directory
        if self._temp_dir and self._temp_dir.exists():
            import shutil

            shutil.rmtree(self._temp_dir, ignore_errors=True)


def train_ensemble_from_config(config_path: str, log_level: str = "INFO") -> None:
    """
    Train an ensemble of ChempropModels from a YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the YAML configuration file.
    log_level : str, default="INFO"
        Logging level. Options: "DEBUG", "INFO", "WARNING", "ERROR".
    """
    # Configure logging
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Loading ensemble configuration from: %s", config_path)

    # Load configuration
    config = OmegaConf.merge(
        OmegaConf.structured(EnsembleConfig),
        OmegaConf.load(config_path),
    )

    logger.info("Configuration:\n%s", OmegaConf.to_yaml(config))

    # Create and train ensemble
    ensemble = ChempropEnsemble.from_config(config)  # type: ignore[arg-type]
    ensemble.train_all()
    ensemble.close()

    logger.info("Ensemble training complete!")


def main() -> None:
    """CLI entrypoint for ensemble training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train an ensemble of Chemprop models from YAML configuration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m admet.model.chemprop.ensemble --config configs/ensemble_chemprop.yaml
  python -m admet.model.chemprop.ensemble -c configs/ensemble.yaml --max-parallel 2
  python -m admet.model.chemprop.ensemble -c configs/ensemble.yaml --log-level DEBUG

Configuration file should have the structure:
  data:
    data_dir: "path/to/splits"  # Contains split_*/fold_*/
    test_file: "path/to/test.csv"
    blind_file: "path/to/blind.csv"
    target_cols: [...]
  model: {...}
  optimization: {...}
  mlflow: {...}
  ray:
    max_parallel: 2
    num_cpus: null
    num_gpus: null
        """,
    )
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file",
    )
    parser.add_argument(
        "--max-parallel",
        type=int,
        default=None,
        help="Maximum models to train in parallel (overrides config)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    args = parser.parse_args()

    # Load config and potentially override max_parallel
    config = OmegaConf.merge(
        OmegaConf.structured(EnsembleConfig),
        OmegaConf.load(args.config),
    )

    if args.max_parallel is not None:
        config.max_parallel = args.max_parallel

    # Configure logging
    numeric_level = getattr(logging, args.log_level.upper(), logging.INFO)
    logger.setLevel(numeric_level)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info("Loading ensemble configuration from: %s", args.config)
    logger.info("Configuration:\n%s", OmegaConf.to_yaml(config))

    # Create and train ensemble
    ensemble = ChempropEnsemble.from_config(config)  # type: ignore[arg-type]
    ensemble.train_all()
    ensemble.close()

    logger.info("Ensemble training complete!")


if __name__ == "__main__":
    main()
