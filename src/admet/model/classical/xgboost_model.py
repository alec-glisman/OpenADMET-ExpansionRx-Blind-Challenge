"""XGBoost model implementation for ADMET prediction.

Provides XGBoost gradient boosting with molecular fingerprint features.
"""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig

from admet.model.classical.base import ClassicalModelBase
from admet.model.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("xgboost")
class XGBoostModel(ClassicalModelBase):
    """XGBoost model for ADMET property prediction.

    Uses molecular fingerprints as features and XGBoost gradient boosting
    for regression. Supports multi-output prediction via MultiOutputRegressor.

    Parameters
    ----------
    config : DictConfig
        Configuration with model.xgboost params and fingerprint settings.

    Examples
    --------
    >>> config = OmegaConf.create({
    ...     "model": {
    ...         "type": "xgboost",
    ...         "xgboost": {
    ...             "n_estimators": 100,
    ...             "max_depth": 6,
    ...             "learning_rate": 0.1,
    ...         },
    ...         "fingerprint": {"type": "morgan"},
    ...     },
    ...     "mlflow": {"enabled": False},
    ... })
    >>> model = XGBoostModel(config)
    >>> model.fit(smiles_list, targets)
    >>> predictions = model.predict(new_smiles)
    """

    model_type: str = "xgboost"

    def __init__(self, config: DictConfig) -> None:
        """Initialize XGBoost model.

        Parameters
        ----------
        config : DictConfig
            Model configuration.
        """
        super().__init__(config)

    def _build_estimator(self, params: dict[str, Any]) -> Any:
        """Build XGBRegressor with given parameters.

        Parameters
        ----------
        params : dict[str, Any]
            XGBoost parameters.

        Returns
        -------
        XGBRegressor
            Configured XGBoost regressor.
        """
        from xgboost import XGBRegressor

        default_params = {
            "n_estimators": 100,
            "max_depth": 6,
            "learning_rate": 0.1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }

        merged_params = {**default_params, **params}
        return XGBRegressor(**merged_params)

    @classmethod
    def from_config(cls, config: DictConfig) -> XGBoostModel:
        """Create XGBoost model from configuration.

        Parameters
        ----------
        config : DictConfig
            Model configuration.

        Returns
        -------
        XGBoostModel
            Configured model instance.
        """
        return cls(config)
