"""Base class for classical ML models with fingerprint features.

Provides shared functionality for XGBoost, LightGBM, and CatBoost models
including fingerprint generation, configuration handling, and MLflow integration.
"""

from __future__ import annotations

import logging
from abc import abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from omegaconf import DictConfig, OmegaConf

from admet.features.fingerprints import FingerprintGenerator
from admet.model.base import BaseModel
from admet.model.config import FingerprintConfig
from admet.model.mlflow_mixin import MLflowMixin

if TYPE_CHECKING:
    from sklearn.multioutput import MultiOutputRegressor

logger = logging.getLogger(__name__)


class ClassicalModelBase(BaseModel, MLflowMixin):
    """Base class for classical ML models with fingerprint features.

    Provides common functionality for gradient boosting models (XGBoost, LightGBM,
    CatBoost) that operate on molecular fingerprints rather than raw SMILES.

    Parameters
    ----------
    config : DictConfig
        Model configuration with model params, fingerprint settings, and MLflow config.

    Attributes
    ----------
    model_type : str
        Type identifier for the model.
    fingerprint_generator : FingerprintGenerator
        Generator for molecular fingerprints.
    """

    model_type: str = "classical"

    def __init__(self, config: DictConfig) -> None:
        """Initialize classical model base.

        Parameters
        ----------
        config : DictConfig
            Configuration dictionary with model, fingerprint, and MLflow settings.
        """
        self.config = config
        self._is_fitted = False
        self._model: MultiOutputRegressor | None = None
        self._target_cols: list[str] = []

        self._model_params = self._get_model_params()
        self._fingerprint_config = self._get_fingerprint_config()
        self.fingerprint_generator = FingerprintGenerator(self._fingerprint_config)

        mlflow_config = config.get("mlflow", {})
        if mlflow_config.get("enabled", False):
            self.init_mlflow(mlflow_config)

    @property
    def is_fitted(self) -> bool:
        """Return whether the model has been fitted."""
        return self._is_fitted

    def _get_model_params(self) -> dict[str, Any]:
        """Extract model-specific parameters from config.

        Returns
        -------
        dict[str, Any]
            Model parameters for the underlying estimator.
        """
        model_section = self.config.get("model", {})

        params_key = self.model_type
        if params_key in model_section:
            params = model_section.get(params_key, {})
            if isinstance(params, DictConfig):
                container = OmegaConf.to_container(params, resolve=True)
                if isinstance(container, dict):
                    return {str(k): v for k, v in container.items()}
                return {}
            return dict(params) if params else {}

        return {}

    def _get_fingerprint_config(self) -> FingerprintConfig:
        """Extract fingerprint configuration from config.

        Returns
        -------
        FingerprintConfig
            Configuration for fingerprint generation.
        """
        model_section = self.config.get("model", {})
        fp_section = model_section.get("fingerprint", {})

        fp_type = fp_section.get("type", "morgan")

        morgan_config = fp_section.get("morgan", {})
        rdkit_config = fp_section.get("rdkit", {})
        maccs_config = fp_section.get("maccs", {})
        mordred_config = fp_section.get("mordred", {})

        # Convert configs to proper types
        default_fp = FingerprintConfig()
        morgan = OmegaConf.to_object(morgan_config) if morgan_config else default_fp.morgan
        rdkit = OmegaConf.to_object(rdkit_config) if rdkit_config else default_fp.rdkit
        maccs = OmegaConf.to_object(maccs_config) if maccs_config else default_fp.maccs
        mordred = OmegaConf.to_object(mordred_config) if mordred_config else default_fp.mordred

        return FingerprintConfig(
            type=fp_type,
            morgan=morgan,  # type: ignore[arg-type]
            rdkit=rdkit,  # type: ignore[arg-type]
            maccs=maccs,  # type: ignore[arg-type]
            mordred=mordred,  # type: ignore[arg-type]
        )

    @abstractmethod
    def _build_estimator(self, params: dict[str, Any]) -> Any:
        """Build the underlying estimator.

        Parameters
        ----------
        params : dict[str, Any]
            Model parameters.

        Returns
        -------
        Any
            Base estimator instance (XGBRegressor, LGBMRegressor, CatBoostRegressor).
        """

    def _generate_features(self, smiles: list[str]) -> np.ndarray:
        """Generate fingerprint features from SMILES.

        Parameters
        ----------
        smiles : list[str]
            List of SMILES strings.

        Returns
        -------
        np.ndarray
            Feature matrix of shape (n_samples, n_features).
        """
        return self.fingerprint_generator.generate(smiles)

    def fit(  # type: ignore[override]
        self,
        smiles: list[str],
        targets: np.ndarray,
        *,
        sample_weight: np.ndarray | None = None,
        val_smiles: list[str] | None = None,
        val_y: np.ndarray | None = None,
        **kwargs: Any,
    ) -> ClassicalModelBase:
        """Fit the model on SMILES and targets.

        Parameters
        ----------
        smiles : list[str]
            List of SMILES strings.
        targets : np.ndarray
            Target values of shape (n_samples,) or (n_samples, n_targets).
        sample_weight : np.ndarray | None
            Optional sample weights.
        **kwargs : Any
            Additional keyword arguments passed to the estimator.

        Returns
        -------
        ClassicalModelBase
            Fitted model instance.
        """
        from sklearn.multioutput import MultiOutputRegressor

        logger.info(f"Fitting {self.model_type} model on {len(smiles)} samples")

        X = self._generate_features(smiles)
        y = np.atleast_2d(targets)
        if y.shape[0] == 1:
            y = y.T

        n_targets = y.shape[1]
        self._target_cols = [f"target_{i}" for i in range(n_targets)]

        base_estimator = self._build_estimator(self._model_params)
        self._model = MultiOutputRegressor(base_estimator)

        fit_params = {}
        if sample_weight is not None:
            fit_params["sample_weight"] = sample_weight

        self._model.fit(X, y, **fit_params)
        self._is_fitted = True

        if hasattr(self, "_mlflow_run") and self._mlflow_run:
            self.log_params_from_config()
            self.log_metric("n_samples", len(smiles))
            self.log_metric("n_features", X.shape[1])
            self.log_metric("n_targets", n_targets)

        logger.info(f"{self.model_type} model fitted successfully")
        return self

    def predict(self, smiles: list[str], **kwargs: Any) -> np.ndarray:
        """Generate predictions for SMILES.

        Parameters
        ----------
        smiles : list[str]
            List of SMILES strings.
        **kwargs : Any
            Additional keyword arguments (ignored).

        Returns
        -------
        np.ndarray
            Predictions of shape (n_samples, n_targets).

        Raises
        ------
        RuntimeError
            If model has not been fitted.
        """
        if not self._is_fitted or self._model is None:
            msg = f"{self.model_type} model has not been fitted"
            raise RuntimeError(msg)

        X = self._generate_features(smiles)
        predictions = self._model.predict(X)

        return predictions

    def save(self, path: str | Path) -> None:
        """Save model to disk.

        Parameters
        ----------
        path : str | Path
            Path to save the model.
        """
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "model": self._model,
            "model_type": self.model_type,
            "fingerprint_config": self._fingerprint_config,
            "target_cols": self._target_cols,
            "is_fitted": self._is_fitted,
        }

        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str | Path) -> ClassicalModelBase:
        """Load model from disk.

        Parameters
        ----------
        path : str | Path
            Path to load the model from.

        Returns
        -------
        ClassicalModelBase
            Loaded model instance.
        """
        import joblib

        model_data = joblib.load(path)

        self._model = model_data["model"]
        self._fingerprint_config = model_data["fingerprint_config"]
        self.fingerprint_generator = FingerprintGenerator(self._fingerprint_config)
        self._target_cols = model_data["target_cols"]
        self._is_fitted = model_data["is_fitted"]

        logger.info(f"Model loaded from {path}")
        return self

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get model parameters (sklearn interface).

        Parameters
        ----------
        deep : bool
            If True, return nested parameters.

        Returns
        -------
        dict[str, Any]
            Model parameters.
        """
        params = {
            "model_type": self.model_type,
            "fingerprint_type": self._fingerprint_config.type,
        }
        params.update(self._model_params)
        return params

    def set_params(self, **params: Any) -> ClassicalModelBase:
        """Set model parameters (sklearn interface).

        Parameters
        ----------
        **params : Any
            Parameters to set.

        Returns
        -------
        ClassicalModelBase
            Self with updated parameters.
        """
        for key, value in params.items():
            if key in self._model_params:
                self._model_params[key] = value
        return self

    @classmethod
    def from_config(cls, config: DictConfig) -> ClassicalModelBase:
        """Create model from configuration.

        Parameters
        ----------
        config : DictConfig
            Model configuration.

        Returns
        -------
        ClassicalModelBase
            Configured model instance.
        """
        return cls(config)
