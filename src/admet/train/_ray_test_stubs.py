"""Stub trainer classes for Ray status tests.

These lightweight trainers are defined in the package namespace so Ray
workers can import them when deserializing pickled class definitions.
They purposely implement minimal logic tailored to the test scenarios:
error handling, partial metrics, timeout simulation, etc.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List
import time
import numpy as np

from admet.train.base_trainer import BaseModelTrainer, BaseRayMultiDatasetTrainer
from admet.model.base import BaseModel


# ---------------------------------------------------------------------------
# Base minimal model used by stubs
# ---------------------------------------------------------------------------
class _MinimalModel(BaseModel):
    def __init__(self, endpoints: List[str]):
        self.endpoints = list(endpoints)

    def fit(self, *args, **kwargs):
        return None

    def predict(self, X):
        return np.zeros((X.shape[0], len(self.endpoints)))

    def save(self, path):
        return None

    @classmethod
    def load(cls, path):
        return cls([])

    def get_config(self):
        return {}

    def get_metadata(self):
        return {}


# ---------------------------------------------------------------------------
# Shared helper to build masks
# ---------------------------------------------------------------------------
_def_mask = lambda arr: np.ones_like(arr, dtype=bool)


# ---------------------------------------------------------------------------
# Failing trainer: raises during run to force error status
# ---------------------------------------------------------------------------
class FailingTrainer(BaseModelTrainer):
    def __init__(self, **kwargs):
        kwargs.setdefault("model_cls", BaseModel)
        super().__init__(**kwargs)

    def prepare_features(self, dataset):
        import numpy as _np

        return (
            _np.zeros((len(dataset.train), 1)),
            _np.zeros((len(dataset.val), 1)),
            _np.zeros((len(dataset.test), 1)),
        )

    def prepare_targets(self, dataset):
        import numpy as _np

        D = len(dataset.endpoints)
        return (
            _np.zeros((len(dataset.train), D)),
            _np.zeros((len(dataset.val), D)),
            _np.zeros((len(dataset.test), D)),
        )

    def prepare_masks(self, Y_train, Y_val, Y_test):
        import numpy as _np

        return (
            _np.zeros_like(Y_train, dtype=bool),
            _np.zeros_like(Y_val, dtype=bool),
            _np.zeros_like(Y_test, dtype=bool),
        )

    def build_sample_weights(self, dataset, sample_weight_mapping=None):
        return None

    def build_model(self, endpoints):
        return _MinimalModel(endpoints)

    def compute_metrics(self, *args, **kwargs):
        return {}

    def save_artifacts(self, *args, **kwargs):
        return None

    def run(self, dataset, *args, **kwargs):
        raise RuntimeError("failing trainer run")


# ---------------------------------------------------------------------------
# Trivial trainer: returns empty macro dicts quickly
# ---------------------------------------------------------------------------
class TrivialTrainer(BaseModelTrainer):
    def __init__(self, **kwargs):
        kwargs.setdefault("model_cls", BaseModel)
        super().__init__(**kwargs)

    def prepare_features(self, dataset):
        import numpy as _np

        return (
            _np.zeros((len(dataset.train), 1)),
            _np.zeros((len(dataset.val), 1)),
            _np.zeros((len(dataset.test), 1)),
        )

    def prepare_targets(self, dataset):
        import numpy as _np

        D = len(dataset.endpoints)
        return (
            _np.zeros((len(dataset.train), D)),
            _np.zeros((len(dataset.val), D)),
            _np.zeros((len(dataset.test), D)),
        )

    def prepare_masks(self, Y_train, Y_val, Y_test):
        import numpy as _np

        return (
            _np.ones_like(Y_train, dtype=bool),
            _np.ones_like(Y_val, dtype=bool),
            _np.ones_like(Y_test, dtype=bool),
        )

    def build_sample_weights(self, dataset, sample_weight_mapping=None):
        return None

    def build_model(self, endpoints):
        return _MinimalModel(endpoints)

    def compute_metrics(self, *args, **kwargs):
        return {"train": {"macro": {}}, "validation": {"macro": {}}, "test": {"macro": {}}}

    def save_artifacts(self, *args, **kwargs):
        return None

    def run(self, dataset, *args, **kwargs):  # type: ignore[override]
        # Minimal orchestration: skip fitting, directly return metrics structure.
        return self.compute_metrics()


# ---------------------------------------------------------------------------
# Slow trainer: sleeps to trigger timeout
# ---------------------------------------------------------------------------
class SlowTrainer(BaseModelTrainer):
    def __init__(self, **kwargs):
        kwargs.setdefault("model_cls", BaseModel)
        super().__init__(**kwargs)

    def prepare_features(self, dataset):
        return (
            np.zeros((len(dataset.train), 1)),
            np.zeros((len(dataset.val), 1)),
            np.zeros((len(dataset.test), 1)),
        )

    def prepare_targets(self, dataset):
        D = len(dataset.endpoints)
        return (
            np.zeros((len(dataset.train), D)),
            np.zeros((len(dataset.val), D)),
            np.zeros((len(dataset.test), D)),
        )

    def prepare_masks(self, Y_train, Y_val, Y_test):
        return (
            np.ones_like(Y_train, dtype=bool),
            np.ones_like(Y_val, dtype=bool),
            np.ones_like(Y_test, dtype=bool),
        )

    def build_sample_weights(self, dataset, sample_weight_mapping=None):
        return None

    def build_model(self, endpoints):
        return _MinimalModel(endpoints)

    def compute_metrics(self, *args, **kwargs):
        return {"train": {"macro": {}}, "validation": {"macro": {}}, "test": {"macro": {}}}

    def save_artifacts(self, *args, **kwargs):
        return None

    def run(self, dataset, *args, **kwargs):
        time.sleep(0.01)
        return self.compute_metrics()


# ---------------------------------------------------------------------------
# Partial trainer: omit validation split to force 'partial' status
# ---------------------------------------------------------------------------
class PartialTrainer(BaseModelTrainer):
    def __init__(self, **kwargs):
        kwargs.setdefault("model_cls", BaseModel)
        super().__init__(**kwargs)

    def prepare_features(self, dataset):
        return (
            np.zeros((len(dataset.train), 1)),
            np.zeros((len(dataset.val), 1)),
            np.zeros((len(dataset.test), 1)),
        )

    def prepare_targets(self, dataset):
        D = len(dataset.endpoints)
        return (
            np.zeros((len(dataset.train), D)),
            np.zeros((len(dataset.val), D)),
            np.zeros((len(dataset.test), D)),
        )

    def prepare_masks(self, Y_train, Y_val, Y_test):
        return (
            np.ones_like(Y_train, dtype=bool),
            np.ones_like(Y_val, dtype=bool),
            np.ones_like(Y_test, dtype=bool),
        )

    def build_sample_weights(self, dataset, sample_weight_mapping=None):
        return None

    def build_model(self, endpoints):
        return _MinimalModel(endpoints)

    def compute_metrics(self, *args, **kwargs):
        return {"train": {"macro": {}}, "test": {"macro": {}}}

    def save_artifacts(self, *args, **kwargs):
        return None

    def run(self, dataset, *args, **kwargs):  # type: ignore[override]
        # Return metrics missing 'validation' to force partial classification.
        return self.compute_metrics()


# ---------------------------------------------------------------------------
# Ray multi-dataset wrappers
# ---------------------------------------------------------------------------
class MinimalRayTrainer(BaseRayMultiDatasetTrainer):
    def discover_datasets(self, root: Path):
        return [p for p in root.rglob("hf_dataset") if p.is_dir()]

    def build_output_dir(self, base: Path, meta: Dict[str, Any]) -> Path:
        cluster = str(meta.get("cluster", "unknown_method"))
        split = str(meta.get("split", "unknown_split"))
        fold = str(meta.get("fold", "unknown_fold"))
        return base / cluster / f"split_{split}" / f"fold_{fold}"


class SlowRayTrainer(BaseRayMultiDatasetTrainer):
    def discover_datasets(self, root: Path):
        return [p for p in root.rglob("hf_dataset") if p.is_dir()]

    def build_output_dir(self, base: Path, meta: Dict[str, Any]) -> Path:
        cluster = str(meta.get("cluster", "unknown_method"))
        split = str(meta.get("split", "unknown_split"))
        fold = str(meta.get("fold", "unknown_fold"))
        return base / cluster / f"split_{split}" / f"fold_{fold}"


class PartialRayTrainer(BaseRayMultiDatasetTrainer):
    def discover_datasets(self, root: Path):
        return [p for p in root.rglob("hf_dataset") if p.is_dir()]

    def build_output_dir(self, base: Path, meta: Dict[str, Any]) -> Path:
        cluster = str(meta.get("cluster", "unknown_method"))
        split = str(meta.get("split", "unknown_split"))
        fold = str(meta.get("fold", "unknown_fold"))
        return base / cluster / f"split_{split}" / f"fold_{fold}"


__all__ = [
    "FailingTrainer",
    "TrivialTrainer",
    "SlowTrainer",
    "PartialTrainer",
    "MinimalRayTrainer",
    "SlowRayTrainer",
    "PartialRayTrainer",
]
