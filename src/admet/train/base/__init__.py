"""Training base package split from monolithic `base_trainer`.

Submodules:
    utils          – feature/target extraction helpers & metadata dataclasses
    model_trainer  – `BaseModelTrainer`, `RunSummary`, `FeaturizationMethod`
    ray_trainer    – Ray remote worker + `BaseEnsembleTrainer` orchestration
    api            – Convenience wrappers `train_model`, `train_ensemble`

Public re‑exports maintained for previous names (without deprecation shims).
"""

from .api import train_ensemble, train_model
from .model_trainer import BaseModelTrainer, FeaturizationMethod, RunSummary
from .ray_trainer import BaseEnsembleTrainer
from .utils import (
    SplitMetadata,
    _extract_features,
    _extract_targets,
    _target_mask,
    infer_split_metadata,
    metadata_from_dict,
)

__all__ = [
    "_extract_features",
    "_extract_targets",
    "_target_mask",
    "infer_split_metadata",
    "SplitMetadata",
    "metadata_from_dict",
    "BaseModelTrainer",
    "RunSummary",
    "FeaturizationMethod",
    "BaseEnsembleTrainer",
    "train_model",
    "train_ensemble",
]
