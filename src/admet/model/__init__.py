from __future__ import annotations

import logging
import warnings

from admet.model import chemeleon, chemprop, classical, hpo
from admet.model.base import BaseModel
from admet.model.config import (
    BaseDataConfig,
    BaseMlflowConfig,
    BaseModelConfig,
    CatBoostModelParams,
    ChemeleonModelParams,
    FingerprintConfig,
    LightGBMModelParams,
    UnfreezeScheduleConfig,
    XGBoostModelParams,
)
from admet.model.ensemble import Ensemble
from admet.model.mlflow_mixin import MLflowMixin
from admet.model.registry import ModelRegistry

__all__ = [
    # Base classes
    "BaseModel",
    "BaseDataConfig",
    "BaseModelConfig",
    "BaseMlflowConfig",
    # Configuration
    "FingerprintConfig",
    "XGBoostModelParams",
    "LightGBMModelParams",
    "CatBoostModelParams",
    "ChemeleonModelParams",
    "UnfreezeScheduleConfig",
    # Utilities
    "MLflowMixin",
    "ModelRegistry",
    "Ensemble",
    # Sub-modules
    "classical",
    "chemprop",
    "chemeleon",
    "hpo",
]

# Suppress overly verbose logging from dependencies
logging.getLogger("lightning.pytorch").setLevel(logging.WARNING)
logging.getLogger("lightning.fabric").setLevel(logging.WARNING)
logging.getLogger("lightning.pytorch.accelerators.cuda").setLevel(logging.WARNING)
logging.getLogger("mlflow").setLevel(logging.WARNING)

# Enable INFO logging for curriculum callbacks (per-quality metrics)
logging.getLogger("admet.model.chemprop.curriculum").setLevel(logging.INFO)

# Suppress specific warnings
warnings.filterwarnings("ignore", message=".*srun.*")
warnings.filterwarnings("ignore", message=".*num_workers.*")
warnings.filterwarnings("ignore", message=".*Please use `name`.*")
warnings.filterwarnings("ignore", message=".*Please set `input_example`.*")
