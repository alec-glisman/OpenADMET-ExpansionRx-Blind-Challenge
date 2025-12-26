"""MLflow tracking mixin for consistent logging across all model types.

This module provides a mixin class that standardizes MLflow tracking behavior
for all ADMET models, ensuring consistent parameter logging, metric tracking,
and artifact management.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import mlflow
from mlflow.tracking import MlflowClient
from omegaconf import OmegaConf

if TYPE_CHECKING:
    from omegaconf import DictConfig

logger = logging.getLogger(__name__)


class MLflowMixin:
    """Mixin providing consistent MLflow tracking across all model types.

    This mixin extracts common MLflow tracking logic to ensure all models
    (chemprop, chemeleon, xgboost, etc.) have consistent logging behavior.

    Attributes:
        config: Expected from the model class (DictConfig).
        _mlflow_run_id: ID of the active MLflow run.
        _mlflow_client: MLflow tracking client instance.

    Example:
        >>> class MyModel(BaseModel, MLflowMixin):
        ...     def fit(self, smiles, y, val_smiles=None, val_y=None):
        ...         self.init_mlflow(run_name="my_run")
        ...         self.log_params_from_config()
        ...         # Training loop
        ...         self.log_metrics({"train_loss": 0.1})
        ...         self.end_mlflow()
        ...         return self
    """

    config: DictConfig
    _mlflow_run_id: str | None = None
    _mlflow_client: MlflowClient | None = None
    _mlflow_run: Any = None

    def init_mlflow(
        self,
        run_name: str | None = None,
        nested: bool = False,
        tags: dict[str, str] | None = None,
    ) -> str | None:
        """Initialize an MLflow run.

        Parameters:
            run_name: Name for the MLflow run. If None, auto-generated.
            nested: Whether this is a nested run (e.g., ensemble member).
            tags: Additional tags to add to the run.

        Returns:
            The run ID if MLflow is enabled, None otherwise.
        """
        mlflow_config = self.config.get("mlflow", {})
        if not mlflow_config.get("enabled", True):
            logger.debug("MLflow tracking disabled")
            return None

        # Set tracking URI if configured
        tracking_uri = mlflow_config.get("tracking_uri")
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        # Set experiment
        experiment_name = mlflow_config.get("experiment_name", "admet")
        mlflow.set_experiment(experiment_name)

        # Initialize client
        self._mlflow_client = MlflowClient()

        # Generate run name if not provided
        if run_name is None:
            model_type = getattr(self, "model_type", "model")
            config_run_name = mlflow_config.get("run_name")
            run_name = f"{model_type}_{config_run_name}" if config_run_name else model_type

        # Prepare tags
        run_tags = {"model_type": getattr(self, "model_type", "unknown")}
        if tags:
            run_tags.update(tags)

        # Start run
        self._mlflow_run = mlflow.start_run(run_name=run_name, nested=nested, tags=run_tags)
        self._mlflow_run_id = self._mlflow_run.info.run_id

        logger.info(f"Started MLflow run: {self._mlflow_run_id} (name={run_name})")
        return self._mlflow_run_id

    def log_params_from_config(self, max_depth: int = 3) -> None:
        """Log configuration parameters to MLflow.

        Flattens the nested config structure and logs all parameters.

        Parameters:
            max_depth: Maximum nesting depth to flatten.
        """
        if not self._mlflow_run_id:
            return

        config_dict = OmegaConf.to_container(self.config, resolve=True)
        flat_params = self._flatten_dict(config_dict, max_depth=max_depth)

        # MLflow has a limit on param value length
        truncated_params = {}
        for key, value in flat_params.items():
            str_val = str(value)
            if len(str_val) > 250:
                str_val = str_val[:247] + "..."
            truncated_params[key] = str_val

        try:
            mlflow.log_params(truncated_params)
        except Exception as e:
            logger.warning(f"Failed to log params to MLflow: {e}")

    def log_params(self, params: dict[str, Any]) -> None:
        """Log arbitrary parameters to MLflow.

        Parameters:
            params: Dictionary of parameter names to values.
        """
        if not self._mlflow_run_id:
            return

        try:
            mlflow.log_params(params)
        except Exception as e:
            logger.warning(f"Failed to log params to MLflow: {e}")

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log metrics to MLflow.

        Parameters:
            metrics: Dictionary of metric names to values.
            step: Optional step number for time-series metrics.
        """
        if not self._mlflow_run_id:
            return

        try:
            mlflow.log_metrics(metrics, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metrics to MLflow: {e}")

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a single metric to MLflow.

        Parameters:
            key: Metric name.
            value: Metric value.
            step: Optional step number.
        """
        if not self._mlflow_run_id:
            return

        try:
            mlflow.log_metric(key, value, step=step)
        except Exception as e:
            logger.warning(f"Failed to log metric '{key}' to MLflow: {e}")

    def log_artifact(self, local_path: str, artifact_path: str | None = None) -> None:
        """Log an artifact file to MLflow.

        Parameters:
            local_path: Path to the local file to log.
            artifact_path: Optional subdirectory in artifacts.
        """
        if not self._mlflow_run_id:
            return

        try:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
        except Exception as e:
            logger.warning(f"Failed to log artifact to MLflow: {e}")

    def log_model(
        self,
        model: Any,
        artifact_path: str = "model",
        registered_model_name: str | None = None,
    ) -> None:
        """Log a model artifact to MLflow.

        Parameters:
            model: The model object to log.
            artifact_path: Subdirectory for the model artifact.
            registered_model_name: Optional name for model registry.
        """
        if not self._mlflow_run_id:
            return

        mlflow_config = self.config.get("mlflow", {})
        if not mlflow_config.get("log_model", True):
            return

        try:
            mlflow.pyfunc.log_model(
                artifact_path=artifact_path,
                python_model=model,
                registered_model_name=registered_model_name,
            )
        except Exception as e:
            logger.warning(f"Failed to log model to MLflow: {e}")

    def set_mlflow_tag(self, key: str, value: str) -> None:
        """Set a tag on the current MLflow run.

        Parameters:
            key: Tag name.
            value: Tag value.
        """
        if not self._mlflow_run_id:
            return

        try:
            mlflow.set_tag(key, value)
        except Exception as e:
            logger.warning(f"Failed to set MLflow tag: {e}")

    def end_mlflow(self, status: str = "FINISHED") -> None:
        """End the current MLflow run.

        Parameters:
            status: Run status (FINISHED, FAILED, KILLED).
        """
        if self._mlflow_run_id:
            try:
                mlflow.end_run(status=status)
                logger.debug(f"Ended MLflow run: {self._mlflow_run_id}")
            except Exception as e:
                logger.warning(f"Failed to end MLflow run: {e}")
            finally:
                self._mlflow_run_id = None
                self._mlflow_run = None

    @property
    def mlflow_run_id(self) -> str | None:
        """Get the current MLflow run ID."""
        return self._mlflow_run_id

    @property
    def mlflow_client(self) -> MlflowClient | None:
        """Get the MLflow client instance."""
        return self._mlflow_client

    @staticmethod
    def _flatten_dict(
        d: dict[str, Any],
        parent_key: str = "",
        sep: str = ".",
        max_depth: int = 3,
    ) -> dict[str, Any]:
        """Flatten nested dictionary for MLflow parameter logging.

        Parameters:
            d: Dictionary to flatten.
            parent_key: Prefix for nested keys.
            sep: Separator between nested keys.
            max_depth: Maximum recursion depth.

        Returns:
            Flattened dictionary with dot-separated keys.
        """
        items: list[tuple[str, Any]] = []
        for key, value in d.items():
            new_key = f"{parent_key}{sep}{key}" if parent_key else key
            if isinstance(value, dict) and max_depth > 0:
                items.extend(MLflowMixin._flatten_dict(value, new_key, sep, max_depth - 1).items())
            else:
                items.append((new_key, value))
        return dict(items)
