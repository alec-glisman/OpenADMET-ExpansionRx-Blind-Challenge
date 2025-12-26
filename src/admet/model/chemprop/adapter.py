"""BaseModel adapter for ChempropModel.

This module provides a thin wrapper around the existing ChempropModel that
implements the BaseModel interface, enabling use with ModelRegistry and
consistent API across different model types.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from omegaconf import DictConfig, OmegaConf

from admet.model.base import BaseModel
from admet.model.chemprop.model import ChempropHyperparams, ChempropModel
from admet.model.mlflow_mixin import MLflowMixin
from admet.model.registry import ModelRegistry

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


@ModelRegistry.register("chemprop")
class ChempropModelAdapter(BaseModel, MLflowMixin):
    """BaseModel adapter wrapping the existing ChempropModel.

    This adapter provides the unified BaseModel interface for ChempropModel,
    enabling it to work with ModelRegistry and other multi-model tooling.

    The adapter delegates to the underlying ChempropModel for actual training
    and prediction, handling the interface translation between the new unified
    API and the existing implementation.

    Parameters
    ----------
    config : DictConfig
        Configuration object. Can use either:
        - New structure: config.model.chemprop containing model params
        - Legacy structure: config.model containing params directly

    Attributes
    ----------
    model_type : str
        Model type identifier ("chemprop").
    config : DictConfig
        The configuration object.
    _model : ChempropModel | None
        The underlying ChempropModel instance.

    Examples
    --------
    Create from config:

    >>> from omegaconf import OmegaConf
    >>> config = OmegaConf.load("configs/chemprop.yaml")
    >>> model = ChempropModelAdapter.from_config(config)
    >>> model.fit(smiles, y)
    >>> predictions = model.predict(test_smiles)

    Create via registry:

    >>> from admet.model.registry import ModelRegistry
    >>> config = OmegaConf.create({"model": {"type": "chemprop", ...}})
    >>> model = ModelRegistry.create(config)
    """

    model_type = "chemprop"

    def __init__(self, config: DictConfig) -> None:
        """Initialize adapter with configuration.

        Parameters
        ----------
        config : DictConfig
            Configuration object.
        """
        super().__init__(config)
        self._model: ChempropModel | None = None
        self._smiles_col: str = "smiles"
        self._target_cols: list[str] = []

    def fit(
        self,
        smiles: list[str],
        y: np.ndarray,
        val_smiles: list[str] | None = None,
        val_y: np.ndarray | None = None,
    ) -> "ChempropModelAdapter":
        """Train the model on SMILES and targets.

        Creates a ChempropModel from the configuration and trains it.

        Parameters
        ----------
        smiles : list[str]
            Training SMILES strings.
        y : np.ndarray
            Training target values. Shape: (n_samples,) or (n_samples, n_tasks).
        val_smiles : list[str] | None, optional
            Validation SMILES strings.
        val_y : np.ndarray | None, optional
            Validation target values.

        Returns
        -------
        ChempropModelAdapter
            Self, for method chaining.
        """
        # Extract config sections
        data_config = self._get_data_config()

        self._smiles_col = data_config.get("smiles_col", "smiles")
        self._target_cols = list(data_config.get("target_cols", []))

        # If target_cols not specified, infer from y shape
        if not self._target_cols:
            if y.ndim == 1:
                self._target_cols = ["target"]
            else:
                self._target_cols = [f"target_{i}" for i in range(y.shape[1])]

        # Create training DataFrame
        df_train = self._create_dataframe(smiles, y)

        # Create validation DataFrame if provided
        df_val = None
        if val_smiles is not None and val_y is not None:
            df_val = self._create_dataframe(val_smiles, val_y)

        # Build hyperparams
        hyperparams = self._build_hyperparams()

        # Create ChempropModel
        self._model = ChempropModel(
            df_train=df_train,
            df_validation=df_val,
            smiles_col=self._smiles_col,
            target_cols=self._target_cols,
            target_weights=list(data_config.get("target_weights", [])),
            progress_bar=self.config.get("optimization", {}).get("progress_bar", False),
            hyperparams=hyperparams,
            mlflow_tracking=self.config.get("mlflow", {}).get("tracking", False),
            mlflow_tracking_uri=self.config.get("mlflow", {}).get("tracking_uri"),
            mlflow_experiment_name=self.config.get("mlflow", {}).get("experiment_name", "chemprop"),
            mlflow_run_name=self.config.get("mlflow", {}).get("run_name"),
        )

        # Train
        self._model.fit()
        self._fitted = True

        return self

    def predict(self, smiles: list[str]) -> np.ndarray:
        """Generate predictions for SMILES.

        Parameters
        ----------
        smiles : list[str]
            SMILES strings to predict.

        Returns
        -------
        np.ndarray
            Predictions. Shape: (n_samples,) or (n_samples, n_tasks).

        Raises
        ------
        RuntimeError
            If model has not been fitted.
        """
        if not self._fitted or self._model is None:
            raise RuntimeError("Model has not been fitted. Call fit() first.")

        # Create DataFrame for prediction
        df = pd.DataFrame({self._smiles_col: smiles})

        # Get predictions
        predictions = self._model.predict(df)

        return predictions

    @classmethod
    def from_config(cls, config: DictConfig) -> "ChempropModelAdapter":
        """Create model from configuration.

        Parameters
        ----------
        config : DictConfig
            Configuration object.

        Returns
        -------
        ChempropModelAdapter
            Initialized model adapter.
        """
        return cls(config)

    def _get_data_config(self) -> DictConfig:
        """Get data configuration section.

        Returns
        -------
        DictConfig
            Data configuration.
        """
        return self.config.get("data", OmegaConf.create({}))

    def _get_model_config(self) -> DictConfig:
        """Get model configuration section.

        Handles both new and legacy config structures.

        Returns
        -------
        DictConfig
            Model configuration.
        """
        model_section = self.config.get("model", OmegaConf.create({}))

        # New structure: model.type and model.chemprop
        if "type" in model_section and "chemprop" in model_section:
            return model_section.get("chemprop", OmegaConf.create({}))

        # Legacy structure: model contains params directly
        return model_section

    def _build_hyperparams(self) -> ChempropHyperparams:
        """Build ChempropHyperparams from config.

        Returns
        -------
        ChempropHyperparams
            Hyperparameters object.
        """
        model_config = self._get_model_config()
        opt_config = self.config.get("optimization", OmegaConf.create({}))

        return ChempropHyperparams(
            # Optimization
            init_lr=opt_config.get("init_lr", 1e-4),
            max_lr=opt_config.get("max_lr", 1e-3),
            final_lr=opt_config.get("final_lr", 1e-4),
            warmup_epochs=opt_config.get("warmup_epochs", 5),
            patience=opt_config.get("patience", 15),
            max_epochs=opt_config.get("max_epochs", 150),
            batch_size=opt_config.get("batch_size", 32),
            num_workers=opt_config.get("num_workers", 0),
            seed=opt_config.get("seed", 12345),
            criterion=opt_config.get("criterion", "MSE"),
            # Architecture
            depth=model_config.get("depth", 5),
            message_hidden_dim=model_config.get("message_hidden_dim", 600),
            dropout=model_config.get("dropout", 0.1),
            num_layers=model_config.get("num_layers", 2),
            hidden_dim=model_config.get("hidden_dim", 600),
            batch_norm=model_config.get("batch_norm", True),
            ffn_type=model_config.get("ffn_type", "regression"),
            trunk_n_layers=model_config.get("trunk_n_layers"),
            trunk_hidden_dim=model_config.get("trunk_hidden_dim"),
            n_experts=model_config.get("n_experts"),
        )

    def _create_dataframe(self, smiles: list[str], y: np.ndarray) -> pd.DataFrame:
        """Create DataFrame from SMILES and targets.

        Parameters
        ----------
        smiles : list[str]
            SMILES strings.
        y : np.ndarray
            Target values.

        Returns
        -------
        pd.DataFrame
            DataFrame with smiles and target columns.
        """
        df = pd.DataFrame({self._smiles_col: smiles})

        if y.ndim == 1:
            df[self._target_cols[0]] = y
        else:
            for i, col in enumerate(self._target_cols):
                df[col] = y[:, i]

        return df

    @property
    def underlying_model(self) -> ChempropModel | None:
        """Get the underlying ChempropModel instance.

        Returns
        -------
        ChempropModel | None
            The underlying model, or None if not fitted.
        """
        return self._model

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get model parameters.

        Parameters
        ----------
        deep : bool, optional
            If True, include nested parameters.

        Returns
        -------
        dict
            Parameter dictionary.
        """
        params = super().get_params(deep)
        if self._model is not None:
            params["underlying_model"] = self._model
        return params
