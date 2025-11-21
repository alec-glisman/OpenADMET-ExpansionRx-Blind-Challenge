"""Integration tests for train_model/train_ensemble.

pyright: ignore-all
flake8: noqa
mypy: ignore-errors
"""

import os
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pytest
from admet.train.base import (
    BaseModelTrainer,
    BaseEnsembleTrainer,
    FeaturizationMethod,
    train_model,
    train_ensemble,
)


class DummyModel:

    def __init__(self, endpoints, params, seed):
        self.endpoints = endpoints
        self.params = params or {}
        self.seed = seed

    def fit(self, X_train, Y_train, **kwargs):  # type: ignore[unused-argument]
        pass

    def predict(self, X):  # type: ignore[unused-argument]
        return np.zeros((X.shape[0], len(self.endpoints)))

    def save(self, path: str):  # type: ignore[unused-argument]
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "dummy.json").write_text(json.dumps({"endpoints": self.endpoints}))


class FPTrainer(BaseModelTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.featurization = FeaturizationMethod.MORGAN_FP


def make_dataset(n_train=5, n_val=3, n_test=2):
    cols = {
        "fp1": np.random.rand(n_train + n_val + n_test),
        "fp2": np.random.rand(n_train + n_val + n_test),
        "fp3": np.random.rand(n_train + n_val + n_test),
        "A": np.random.rand(n_train + n_val + n_test),
    }
    df = pd.DataFrame(cols)
    train = df.iloc[:n_train].copy()
    val = df.iloc[n_train : n_train + n_val].copy()
    test = df.iloc[n_train + n_val :].copy()

    class DS:
        endpoints = ["A"]
        fingerprint_cols = ["fp1", "fp2", "fp3"]
        smiles_col = "smiles"
        train = train
        val = val
        test = test

    return DS()


@pytest.fixture
def synthetic_dataset():
    return make_dataset()


def test_train_model_end_to_end(tmp_path, synthetic_dataset):
    metrics = train_model(
        synthetic_dataset,
        trainer_cls=FPTrainer,
        model_cls=DummyModel,
        model_params={"alpha": 0.1},
        output_dir=tmp_path / "single",
        early_stopping_rounds=10,
    )
    assert set(metrics.keys()) == {"train", "validation", "test"}
    assert (tmp_path / "single" / "model").is_dir()
    assert (tmp_path / "single" / "metrics.json").is_file()


@pytest.mark.skipif("RAY_DISABLE_INTEGRATION" in os.environ, reason="Ray integration disabled by env var")
def test_train_ensemble_with_monkeypatched_loader(tmp_path, monkeypatch):
    root = tmp_path / "root"
    d1 = root / "clusterA" / "split_train" / "fold_0" / "hf_dataset"
    d2 = root / "clusterA" / "split_train" / "fold_1" / "hf_dataset"
    d1.mkdir(parents=True)
    d2.mkdir(parents=True)
    synthetic_ds = make_dataset()

    def fake_load_dataset(path, n_fingerprint_bits=None):
        return synthetic_ds

    import admet.data.load as load_mod

    monkeypatch.setattr(load_mod, "load_dataset", fake_load_dataset)
    res = train_ensemble(
        root,
        ensemble_trainer_cls=BaseEnsembleTrainer,
        trainer_cls=FPTrainer,
        model_cls=DummyModel,
        model_params={"alpha": 0.2},
        output_root=tmp_path / "ensemble",
        num_cpus=1,
        dry_run=True,
    )
    assert len(res) == 2
    assert (tmp_path / "ensemble" / "metrics_summary.csv").exists()
