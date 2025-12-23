"""Generic ensemble training for ADMET models.

This module provides a model-agnostic ensemble class that works with any
model type registered in the ModelRegistry.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from admet.model.base import BaseModel
from admet.model.mlflow_mixin import MLflowMixin
from admet.model.registry import ModelRegistry

logger = logging.getLogger(__name__)


class Ensemble(MLflowMixin):
    """Generic ensemble of ADMET models.

    Creates and manages an ensemble of models of any registered type.
    Supports training across multiple data splits/folds with aggregated
    predictions and uncertainty estimation.

    Parameters
    ----------
    config : DictConfig
        Ensemble configuration with model, data, and mlflow settings.

    Attributes
    ----------
    models : list[BaseModel]
        List of fitted model instances.
    model_type : str
        Type of model in the ensemble.

    Examples
    --------
    >>> config = OmegaConf.create({
    ...     "model": {"type": "xgboost", "xgboost": {"n_estimators": 100}},
    ...     "ensemble": {"n_models": 5},
    ...     "data": {"smiles_col": "smiles", "target_cols": ["target"]},
    ...     "mlflow": {"enabled": False},
    ... })
    >>> ensemble = Ensemble(config)
    >>> ensemble.fit(smiles_list, targets_array)
    >>> mean_preds, std_preds = ensemble.predict_with_uncertainty(test_smiles)
    """

    def __init__(self, config: DictConfig) -> None:
        """Initialize ensemble.

        Parameters
        ----------
        config : DictConfig
            Ensemble configuration.
        """
        self.config = config
        self.models: list[BaseModel] = []

        model_section = config.get("model", {})
        self.model_type = model_section.get("type", "chemprop")

        ensemble_section = config.get("ensemble", {})
        self.n_models = ensemble_section.get("n_models", 5)
        self.aggregation = ensemble_section.get("aggregation", "mean")

        mlflow_config = config.get("mlflow", {})
        if mlflow_config.get("enabled", False):
            self.init_mlflow(mlflow_config)

    @property
    def is_fitted(self) -> bool:
        """Return whether the ensemble has been fitted."""
        return len(self.models) > 0 and all(m.is_fitted for m in self.models)

    def _create_model(self, seed: int | None = None) -> BaseModel:
        """Create a single model instance.

        Parameters
        ----------
        seed : int | None
            Optional random seed for this model.

        Returns
        -------
        BaseModel
            Unfitted model instance.
        """
        model_config = OmegaConf.to_container(self.config, resolve=True)

        if seed is not None:
            model_section = model_config.get("model", {})
            type_key = self.model_type
            if type_key in model_section:
                model_section[type_key]["random_state"] = seed
                model_section[type_key]["random_seed"] = seed

        return ModelRegistry.create(OmegaConf.create(model_config))

    def fit(
        self,
        smiles: list[str],
        targets: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        seeds: list[int] | None = None,
        **kwargs: Any,
    ) -> Ensemble:
        """Fit ensemble of models.

        Parameters
        ----------
        smiles : list[str]
            List of SMILES strings.
        targets : np.ndarray
            Target values.
        sample_weight : np.ndarray | None
            Optional sample weights.
        seeds : list[int] | None
            Random seeds for each model (default: 0, 1, ..., n_models-1).
        **kwargs : Any
            Additional keyword arguments passed to model.fit().

        Returns
        -------
        Ensemble
            Fitted ensemble instance.
        """
        if seeds is None:
            seeds = list(range(self.n_models))

        logger.info(f"Training ensemble of {self.n_models} {self.model_type} models")

        self.models = []
        for i, seed in enumerate(seeds[: self.n_models]):
            logger.info(f"Training model {i + 1}/{self.n_models} (seed={seed})")

            model = self._create_model(seed=seed)
            model.fit(smiles, targets, sample_weight=sample_weight, **kwargs)
            self.models.append(model)

        if hasattr(self, "_mlflow_run") and self._mlflow_run:
            self.log_metric("n_ensemble_models", len(self.models))
            self.log_metric("model_type", self.model_type)

        logger.info(f"Ensemble training complete: {len(self.models)} models")
        return self

    def predict(self, smiles: list[str], **kwargs: Any) -> np.ndarray:
        """Generate ensemble predictions.

        Parameters
        ----------
        smiles : list[str]
            List of SMILES strings.
        **kwargs : Any
            Additional keyword arguments passed to model.predict().

        Returns
        -------
        np.ndarray
            Aggregated predictions.

        Raises
        ------
        RuntimeError
            If ensemble has not been fitted.
        """
        if not self.is_fitted:
            msg = "Ensemble has not been fitted"
            raise RuntimeError(msg)

        predictions = self._collect_predictions(smiles, **kwargs)
        return self._aggregate_predictions(predictions)

    def predict_with_uncertainty(self, smiles: list[str], **kwargs: Any) -> tuple[np.ndarray, np.ndarray]:
        """Generate predictions with uncertainty estimates.

        Parameters
        ----------
        smiles : list[str]
            List of SMILES strings.
        **kwargs : Any
            Additional keyword arguments passed to model.predict().

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            Mean predictions and standard deviation.
        """
        if not self.is_fitted:
            msg = "Ensemble has not been fitted"
            raise RuntimeError(msg)

        predictions = self._collect_predictions(smiles, **kwargs)
        mean_pred = np.mean(predictions, axis=0)
        std_pred = np.std(predictions, axis=0)

        return mean_pred, std_pred

    def _collect_predictions(self, smiles: list[str], **kwargs: Any) -> np.ndarray:
        """Collect predictions from all models.

        Parameters
        ----------
        smiles : list[str]
            List of SMILES strings.
        **kwargs : Any
            Additional keyword arguments.

        Returns
        -------
        np.ndarray
            Stacked predictions of shape (n_models, n_samples, n_targets).
        """
        all_preds = []
        for model in self.models:
            preds = model.predict(smiles, **kwargs)
            all_preds.append(preds)

        return np.stack(all_preds, axis=0)

    def _aggregate_predictions(self, predictions: np.ndarray) -> np.ndarray:
        """Aggregate predictions across models.

        Parameters
        ----------
        predictions : np.ndarray
            Predictions of shape (n_models, n_samples, n_targets).

        Returns
        -------
        np.ndarray
            Aggregated predictions of shape (n_samples, n_targets).
        """
        if self.aggregation == "mean":
            return np.mean(predictions, axis=0)
        elif self.aggregation == "median":
            return np.median(predictions, axis=0)
        else:
            msg = f"Unknown aggregation method: {self.aggregation}"
            raise ValueError(msg)

    def save(self, path: str | Path) -> None:
        """Save ensemble to disk.

        Parameters
        ----------
        path : str | Path
            Directory path to save ensemble.
        """
        import joblib

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        metadata = {
            "model_type": self.model_type,
            "n_models": len(self.models),
            "aggregation": self.aggregation,
            "config": OmegaConf.to_container(self.config, resolve=True),
        }
        joblib.dump(metadata, path / "metadata.pkl")

        for i, model in enumerate(self.models):
            # Save entire model object to preserve state
            joblib.dump(model, path / f"model_{i}.pkl")

        logger.info(f"Ensemble saved to {path}")

    def load(self, path: str | Path) -> Ensemble:
        """Load ensemble from disk.

        Parameters
        ----------
        path : str | Path
            Directory path containing saved ensemble.

        Returns
        -------
        Ensemble
            Loaded ensemble instance.
        """
        import joblib

        path = Path(path)

        metadata = joblib.load(path / "metadata.pkl")
        self.model_type = metadata["model_type"]
        self.aggregation = metadata["aggregation"]

        self.models = []
        for i in range(metadata["n_models"]):
            model_path = path / f"model_{i}.pkl"
            model = joblib.load(model_path)
            self.models.append(model)

        logger.info(f"Ensemble loaded from {path}: {len(self.models)} models")
        return self

    @classmethod
    def from_config(cls, config: DictConfig) -> Ensemble:
        """Create ensemble from configuration.

        Parameters
        ----------
        config : DictConfig
            Ensemble configuration.

        Returns
        -------
        Ensemble
            Configured ensemble instance.
        """
        return cls(config)
