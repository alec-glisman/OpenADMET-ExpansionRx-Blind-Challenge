"""Tests for the ensemble predictor and discovery helpers.

These unit tests verify the discovery of model run directories and that an
ensemble prediction run returns a consistent `EnsemblePredictSummary` object.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List
import pytest

import pandas as pd

from admet.evaluate.ensemble import (
    EnsemblePredictConfig,
    discover_model_runs,
    run_ensemble_predictions_from_root,
)
from admet.model.base import BaseModel


class _DummyModel(BaseModel):
    def __init__(self, endpoints: List[str]):
        self.endpoints = list(endpoints)
        self.input_type = "smiles"

    def fit(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError

    def predict(self, X):
        # X is (N, 1) object array of SMILES
        import numpy as np

        n = X.shape[0]
        d = len(self.endpoints)
        return np.ones((n, d), dtype=float)

    def save(self, path):  # pragma: no cover - not used here
        Path(path).mkdir(parents=True, exist_ok=True)

    @classmethod
    def load(cls, path):  # pragma: no cover - not used here
        raise NotImplementedError

    def get_config(self):  # pragma: no cover - not used here
        return {}

    def get_metadata(self):  # pragma: no cover - not used here
        return {}


def test_discover_model_runs_prefers_run_meta(tmp_path: Path) -> None:
    root = tmp_path / "runs"
    run1 = root / "run1"
    run2 = root / "run2"
    run1.mkdir(parents=True)
    run2.mkdir(parents=True)
    (run1 / "run_meta.json").write_text(json.dumps({"model_type": "_DummyModel"}))
    (run2 / "run_meta.json").write_text(json.dumps({"model_type": "_DummyModel"}))

    found = discover_model_runs(root)
    assert set(found) == {run1, run2}


def test_run_ensemble_predictions_from_root_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Create fake runs with run_meta.json so that load_model_from_dir succeeds
    root = tmp_path / "runs"
    run1 = root / "run1"
    run2 = root / "run2"
    run1.mkdir(parents=True)
    run2.mkdir(parents=True)
    meta = {
        "model_type": "_DummyModel",
        "endpoints": ["LogD", "KSOL"],
        "featurization": "smiles",
        "model_path": ".",
    }
    (run1 / "run_meta.json").write_text(json.dumps(meta))
    (run2 / "run_meta.json").write_text(json.dumps(meta))

    # Monkeypatch load_model_from_dir to return our dummy model
    from admet.evaluate import ensemble as ens_mod

    def _fake_load_model_from_dir(d):  # pylint: disable=unused-argument
        return _DummyModel(["LogD", "KSOL"])

    monkeypatch.setattr(ens_mod, "load_model_from_dir", _fake_load_model_from_dir)

    df_eval = pd.DataFrame(
        {
            "Molecule Name": ["mol1", "mol2"],
            "SMILES": ["CCO", "CCN"],
            "LogD": [0.1, 0.2],
            "KSOL": [1.1, 1.2],
        }
    )

    cfg = EnsemblePredictConfig(models_root=root, eval_csv=None, blind_csv=None, agg_fn="mean", n_jobs=1)
    summary = run_ensemble_predictions_from_root(cfg, df_eval=df_eval, df_blind=None)

    # Order of discovered runs is not guaranteed; compare as sets.
    assert set(summary.model_dirs) == {run1, run2}
    assert summary.endpoints == ["LogD", "KSOL"]
    assert summary.preds_log_eval is not None
    # Updated API: ensemble predictions stored directly in endpoint columns
    assert "LogD" in summary.preds_log_eval.columns
    assert summary.metrics_log_eval is not None
