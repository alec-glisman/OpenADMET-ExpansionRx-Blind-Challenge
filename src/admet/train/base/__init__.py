"""Training base package split from monolithic `base_trainer`.

Submodules:
    utils          – feature/target extraction helpers & metadata dataclasses
    model_trainer  – `BaseModelTrainer`, `RunOutputs`, `FeaturizationMethod`
    ray_trainer    – Ray remote worker + `BaseEnsembleTrainer` orchestration
    api            – Convenience wrappers `train_model`, `train_ensemble`

Public re‑exports maintained for previous names (without deprecation shims).
"""

from .utils import (
    _extract_features,
    _extract_targets,
    _target_mask,
    infer_split_metadata,
    SplitMetadata,
    metadata_from_dict,
)
from .model_trainer import (
    BaseModelTrainer,
    RunOutputs,
    FeaturizationMethod,
)
from .ray_trainer import BaseEnsembleTrainer
from .api import train_model, train_ensemble

__all__ = [
    "_extract_features",
    "_extract_targets",
    "_target_mask",
    "infer_split_metadata",
    "SplitMetadata",
    "metadata_from_dict",
    "BaseModelTrainer",
    "RunOutputs",
    "FeaturizationMethod",
    "BaseEnsembleTrainer",
    "train_model",
    "train_ensemble",
]
