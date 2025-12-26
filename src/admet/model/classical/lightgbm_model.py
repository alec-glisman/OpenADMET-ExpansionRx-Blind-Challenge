"""LightGBM model implementation for ADMET prediction.

Provides LightGBM gradient boosting with molecular fingerprint features.
"""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig

from admet.model.classical.base import ClassicalModelBase
from admet.model.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("lightgbm")
class LightGBMModel(ClassicalModelBase):
    """LightGBM model for ADMET property prediction.

    Uses molecular fingerprints as features and LightGBM gradient boosting
    for regression. Supports multi-output prediction via MultiOutputRegressor.

    Parameters
    ----------
    config : DictConfig
        Configuration with model.lightgbm params and fingerprint settings.

    Examples
    --------
    >>> config = OmegaConf.create({
    ...     "model": {
    ...         "type": "lightgbm",
    ...         "lightgbm": {
    ...             "n_estimators": 100,
    ...             "num_leaves": 31,
    ...             "learning_rate": 0.1,
    ...         },
    ...         "fingerprint": {"type": "morgan"},
    ...     },
    ...     "mlflow": {"enabled": False},
    ... })
    >>> model = LightGBMModel(config)
    >>> model.fit(smiles_list, targets)
    >>> predictions = model.predict(new_smiles)
    """

    model_type: str = "lightgbm"

    def __init__(self, config: DictConfig) -> None:
        """Initialize LightGBM model.

        Parameters
        ----------
        config : DictConfig
            Model configuration.
        """
        super().__init__(config)

    def _build_estimator(self, params: dict[str, Any]) -> Any:
        """Build LGBMRegressor with given parameters.

        Parameters
        ----------
        params : dict[str, Any]
            LightGBM parameters.

        Returns
        -------
        LGBMRegressor
            Configured LightGBM regressor.
        """
        from lightgbm import LGBMRegressor

        default_params = {
            "n_estimators": 100,
            "num_leaves": 31,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

        merged_params = {**default_params, **params}
        return LGBMRegressor(**merged_params)

    @classmethod
    def from_config(cls, config: DictConfig) -> LightGBMModel:
        """Create LightGBM model from configuration.

        Parameters
        ----------
        config : DictConfig
            Model configuration.

        Returns
        -------
        LightGBMModel
            Configured model instance.
        """
        return cls(config)
