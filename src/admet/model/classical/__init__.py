"""Classical machine learning models for ADMET prediction.

This module provides implementations of classical ML models (XGBoost, LightGBM,
CatBoost) that use molecular fingerprints as features.
"""

from __future__ import annotations

# Legacy imports for backward compatibility
from admet.model.classical import models
from admet.model.classical.base import ClassicalModelBase
from admet.model.classical.catboost_model import CatBoostModel
from admet.model.classical.lightgbm_model import LightGBMModel
from admet.model.classical.xgboost_model import XGBoostModel

__all__ = [
    "ClassicalModelBase",
    "XGBoostModel",
    "LightGBMModel",
    "CatBoostModel",
    "models",
]
