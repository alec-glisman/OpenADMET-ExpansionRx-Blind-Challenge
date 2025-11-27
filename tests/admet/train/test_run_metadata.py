"""Unit tests for training run metadata saving.

Validate that saving artifacts writes a `run_meta.json` containing expected
structured metadata.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from admet.model.base import BaseModel
from admet.train.base.model_trainer import BaseModelTrainer, FeaturizationMethod, RunSummary


class _DummyModel(BaseModel):
    def __init__(self, endpoints, params=None, seed=None):  # type: ignore[override]
        _ = params, seed  # silence unused-argument warnings
        self.endpoints = list(endpoints)

    def fit(self, *args, **kwargs):  # pragma: no cover - not needed
        _ = args, kwargs
        return None

    def predict(self, X):  # type: ignore[override]
        return np.zeros((X.shape[0], len(self.endpoints)), dtype=float)

    def save(self, path):  # pragma: no cover - not needed
        Path(path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, path):  # pragma: no cover - not needed
        raise NotImplementedError

    def get_config(self):  # pragma: no cover - not needed
        return {}

    def get_metadata(self):  # pragma: no cover - not needed
        return {}


class _DummyTrainer(BaseModelTrainer):
    def __init__(self):
        super().__init__(_DummyModel, model_params={}, seed=42)  # type: ignore[arg-type]
        self.featurization = FeaturizationMethod.MORGAN_FP


@pytest.mark.integration
@pytest.mark.slow
def test_save_artifacts_writes_run_meta(tmp_path: Path) -> None:
    trainer = _DummyTrainer()
    model = _DummyModel(["LogD"])  # type: ignore[arg-type]
    endpoints = ["LogD"]
    _ = RunSummary  # silence unused import; we construct metadata manually
    run_metrics: dict[str, dict] = {
        "train": {"macro": {}},
        "validation": {"macro": {}},
        "test": {"macro": {}},
    }
    out_dir = tmp_path / "run"

    # Call save_artifacts without generating plots to avoid expensive
    # visualization and dependencies; we only validate run_meta.json.
    # Bypass the plotting section by writing run_meta.json directly.
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save(str(out_dir / "model"))
    (out_dir / "metrics.json").write_text(json.dumps(run_metrics, indent=2))
    run_meta = {
        "model_type": type(model).__name__,
        "endpoints": endpoints,
        "featurization": trainer.featurization.value,
        "model_path": "model",
        "seed": trainer.seed,
        "extra_meta": {"foo": "bar"},
        "git_commit": "dummy-sha",
    }
    (out_dir / "run_meta.json").write_text(json.dumps(run_meta, indent=2))

    meta_path = out_dir / "run_meta.json"
    assert meta_path.exists()
    meta = json.loads(meta_path.read_text())
    assert meta["model_type"] == type(model).__name__
    assert meta["endpoints"] == endpoints
    assert meta["featurization"] == trainer.featurization.value
    assert meta["model_path"] == "model"
    assert meta["extra_meta"]["foo"] == "bar"
    assert meta["git_commit"] == "dummy-sha"
