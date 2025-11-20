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
    """Extremely small stand-in model used by trainer stubs.

    Produces all-zero predictions for the requested endpoints and stores
    only the endpoint list. Persistence methods are no-ops; configuration
    and metadata are intentionally empty.
    """

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


# ---------------------------------------------------------------------------
# Failing trainer: raises during run to force error status
# ---------------------------------------------------------------------------
class FailingTrainer(BaseModelTrainer):
    """Trainer that raises during ``run`` to exercise error paths.

    Used to verify Ray orchestration correctly captures failures and
    returns a status payload with ``error`` classification.
    """

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
    """Trainer returning empty metrics without performing a fit.

    Exercises the success path while producing a minimal metrics structure
    used in status classification tests.
    """

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
    """Trainer that sleeps briefly to simulate latency / timeout.

    Introduces a small delay so test cases can assert timeout handling
    behavior in Ray worker execution.
    """

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
    """Trainer that omits validation metrics to produce a 'partial' status.

    Returns metrics for train and test splits only, enabling tests to
    confirm that missing expected splits yields a partial classification.
    """

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
    """Ray multi-dataset trainer stub performing only path discovery.

    Discovers datasets and computes output directory layout without
    executing any model training logic.
    """

    def discover_datasets(self, root: Path):
        return [p for p in root.rglob("hf_dataset") if p.is_dir()]

    def build_output_dir(self, base: Path, meta: Dict[str, Any]) -> Path:
        cluster = str(meta.get("cluster", "unknown_method"))
        split = str(meta.get("split", "unknown_split"))
        fold = str(meta.get("fold", "unknown_fold"))
        return base / cluster / f"split_{split}" / f"fold_{fold}"


class SlowRayTrainer(BaseRayMultiDatasetTrainer):
    """Ray trainer stub used for timeout / latency scenarios.

    Shares logic with :class:`MinimalRayTrainer` but identified separately
    in tests for classification purposes.
    """

    def discover_datasets(self, root: Path):
        return [p for p in root.rglob("hf_dataset") if p.is_dir()]

    def build_output_dir(self, base: Path, meta: Dict[str, Any]) -> Path:
        cluster = str(meta.get("cluster", "unknown_method"))
        split = str(meta.get("split", "unknown_split"))
        fold = str(meta.get("fold", "unknown_fold"))
        return base / cluster / f"split_{split}" / f"fold_{fold}"


class PartialRayTrainer(BaseRayMultiDatasetTrainer):
    """Ray trainer stub producing partial artifact directory layout.

    Used to validate behavior when expected outputs or splits are missing
    compared to the full multi-dataset training pipeline.
    """

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
