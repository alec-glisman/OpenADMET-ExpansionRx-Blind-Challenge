"""CatBoost model implementation for ADMET prediction.

Provides CatBoost gradient boosting with molecular fingerprint features.
"""

from __future__ import annotations

import logging
from typing import Any

from omegaconf import DictConfig

from admet.model.classical.base import ClassicalModelBase
from admet.model.registry import ModelRegistry

logger = logging.getLogger(__name__)


@ModelRegistry.register("catboost")
class CatBoostModel(ClassicalModelBase):
    """CatBoost model for ADMET property prediction.

    Uses molecular fingerprints as features and CatBoost gradient boosting
    for regression. Supports multi-output prediction via MultiOutputRegressor.

    Parameters
    ----------
    config : DictConfig
        Configuration with model.catboost params and fingerprint settings.

    Examples
    --------
    >>> config = OmegaConf.create({
    ...     "model": {
    ...         "type": "catboost",
    ...         "catboost": {
    ...             "iterations": 100,
    ...             "depth": 6,
    ...             "learning_rate": 0.1,
    ...         },
    ...         "fingerprint": {"type": "morgan"},
    ...     },
    ...     "mlflow": {"enabled": False},
    ... })
    >>> model = CatBoostModel(config)
    >>> model.fit(smiles_list, targets)
    >>> predictions = model.predict(new_smiles)
    """

    model_type: str = "catboost"

    def __init__(self, config: DictConfig) -> None:
        """Initialize CatBoost model.

        Parameters
        ----------
        config : DictConfig
            Model configuration.
        """
        super().__init__(config)

    def _build_estimator(self, params: dict[str, Any]) -> Any:
        """Build CatBoostRegressor with given parameters.

        Parameters
        ----------
        params : dict[str, Any]
            CatBoost parameters.

        Returns
        -------
        CatBoostRegressor
            Configured CatBoost regressor.
        """
        from catboost import CatBoostRegressor

        default_params = {
            "iterations": 100,
            "depth": 6,
            "learning_rate": 0.1,
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
        }

        merged_params = {**default_params, **params}
        return CatBoostRegressor(**merged_params)

    @classmethod
    def from_config(cls, config: DictConfig) -> CatBoostModel:
        """Create CatBoost model from configuration.

        Parameters
        ----------
        config : DictConfig
            Model configuration.

        Returns
        -------
        CatBoostModel
            Configured model instance.
        """
        return cls(config)
