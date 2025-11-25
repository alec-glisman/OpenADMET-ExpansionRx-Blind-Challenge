"""Unit tests for BaseModelTrainer utilities.

pyright: ignore-all
flake8: noqa
mypy: ignore-errors
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, cast
import numpy as np
import pandas as pd
import pytest
from admet.train.base import BaseModelTrainer, FeaturizationMethod


class DummyModel:

    def __init__(self, endpoints: List[str], params: Optional[Dict], seed: Optional[int]):
        self.endpoints = endpoints
        self.params = params or {}
        self.seed = seed
        # conform to ModelProtocol
        self.input_type = "fingerprint"
        self.fitted = False
        self.fit_calls = []

    def fit(self, X_train, Y_train, **kwargs):  # type: ignore[unused-argument]
        self.fitted = True
        self.fit_calls.append({"X_train_shape": X_train.shape, "Y_train_shape": Y_train.shape, **kwargs})

    def predict(self, X):  # type: ignore[unused-argument]
        mean_feat = X.mean(axis=1, keepdims=True)
        return np.repeat(mean_feat, len(self.endpoints), axis=1)

    def save(self, path: str):  # type: ignore[unused-argument]
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "dummy_model.json").write_text(json.dumps({"endpoints": self.endpoints}))

    @classmethod
    def load(cls, path: str):  # type: ignore[override]
        # Minimal load implementation for tests: instantiate with empty params
        return cls(endpoints=["A", "B"], params={}, seed=None)


class DummyDataset:

    def __init__(self, featurization: FeaturizationMethod = FeaturizationMethod.MORGAN_FP):
        self.endpoints = ["A", "B"]
        self.fingerprint_cols = (
            ["fp1", "fp2", "fp3"] if featurization == FeaturizationMethod.MORGAN_FP else []
        )
        self.smiles_col = "smiles"
        data_train = {
            "fp1": [0.1, 0.2, 0.3],
            "fp2": [1.0, 1.1, 1.2],
            "fp3": [2.0, 2.1, 2.2],
            "smiles": ["CC", "CCC", "CCCC"],
            "A": [0.5, 0.6, 0.7],
            "B": [1.5, 1.6, 1.7],
            "Dataset": ["d1", "d2", "d1"],
        }
        data_val = {
            "fp1": [0.15, 0.25],
            "fp2": [1.05, 1.15],
            "fp3": [2.05, 2.15],
            "smiles": ["CCO", "CCN"],
            "A": [0.55, 0.65],
            "B": [1.55, 1.65],
            "Dataset": ["d1", "d2"],
        }
        data_test = {
            "fp1": [0.12],
            "fp2": [1.02],
            "fp3": [2.02],
            "smiles": ["CO"],
            "A": [0.52],
            "B": [1.52],
            "Dataset": ["d2"],
        }
        self.train = pd.DataFrame(data_train)
        self.val = pd.DataFrame(data_val)
        self.test = pd.DataFrame(data_test)


class FPTrainer(BaseModelTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.featurization = FeaturizationMethod.MORGAN_FP


class SMILESTrainer(BaseModelTrainer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.featurization = FeaturizationMethod.SMILES


@pytest.fixture
def dataset_fp():
    return DummyDataset(FeaturizationMethod.MORGAN_FP)


@pytest.fixture
def dataset_smiles():
    return DummyDataset(FeaturizationMethod.SMILES)


@pytest.fixture
def dataset_fp_no_precomputed():
    ds = DummyDataset(FeaturizationMethod.MORGAN_FP)
    ds.train = ds.train.drop(columns=["fp1", "fp2", "fp3"])
    ds.val = ds.val.drop(columns=["fp1", "fp2", "fp3"])
    ds.test = ds.test.drop(columns=["fp1", "fp2", "fp3"])
    ds.fingerprint_cols = []
    return ds


def test_prepare_features_fingerprint(dataset_fp):
    trainer = FPTrainer(model_cls=DummyModel)
    X_train, X_val, X_test = trainer.prepare_features(dataset_fp)
    assert X_train.shape == (3, 3)
    assert X_val.shape == (2, 3)
    assert X_test.shape == (1, 3)


def test_prepare_features_fingerprint_generated(dataset_fp_no_precomputed):
    pytest.importorskip("rdkit")
    trainer = FPTrainer(model_cls=DummyModel)
    X_train, X_val, X_test = trainer.prepare_features(dataset_fp_no_precomputed)
    assert X_train.shape == (3, 1024)
    assert X_val.shape == (2, 1024)
    assert X_test.shape == (1, 1024)


def test_prepare_features_smiles(dataset_smiles):
    trainer = SMILESTrainer(model_cls=DummyModel)
    X_train, X_val, X_test = trainer.prepare_features(dataset_smiles)
    assert X_train.shape == (3, 1)
    assert X_val.shape == (2, 1)
    assert X_test.shape == (1, 1)


def test_prepare_targets(dataset_fp):
    trainer = FPTrainer(model_cls=DummyModel)
    Y_train, Y_val, Y_test = trainer.prepare_targets(dataset_fp)
    assert Y_train.shape == (3, 2)
    assert Y_val.shape == (2, 2)
    assert Y_test.shape == (1, 2)


def test_prepare_masks(dataset_fp):
    trainer = FPTrainer(model_cls=DummyModel)
    Y_train, Y_val, Y_test = trainer.prepare_targets(dataset_fp)
    m_tr, m_val, m_test = trainer.prepare_masks(Y_train, Y_val, Y_test)
    assert m_tr.all() and m_val.all() and m_test.all()


def test_sample_weights(dataset_fp):
    trainer = FPTrainer(model_cls=DummyModel)
    mapping = {"d1": 2.0, "default": 1.0}
    sw = trainer.build_sample_weights(dataset_fp, mapping)
    assert sw is not None and sw.tolist() == [2.0, 1.0, 2.0]


def test_sample_weights_default_only(dataset_fp):
    trainer = FPTrainer(model_cls=DummyModel)
    mapping = {"default": 1.5}
    sw = trainer.build_sample_weights(dataset_fp, mapping)
    assert sw is not None and sw.tolist() == [1.5, 1.5, 1.5]


def test_build_model_and_fit_call(tmp_path, dataset_fp):
    trainer = FPTrainer(model_cls=DummyModel, seed=42)
    out = tmp_path / "out"
    run_metrics, summary = trainer.fit(dataset_fp, early_stopping_rounds=5, output_dir=out)
    assert {"train", "validation", "test"} == set(run_metrics.keys())
    assert trainer.model is not None and getattr(trainer.model, "fitted", False)
    call = getattr(trainer.model, "fit_calls", [])[0]
    assert "Y_mask" in call and call["Y_mask"].shape == (3, 2)
    assert call["early_stopping_rounds"] == 5


def test_dry_run_returns_empty_metrics(tmp_path, dataset_fp):
    trainer = FPTrainer(model_cls=DummyModel)
    out = tmp_path / "out"
    run_metrics, summary = trainer.fit(dataset_fp, dry_run=True, output_dir=out)
    assert run_metrics["train"]["macro"] == {}
    assert run_metrics["validation"]["macro"] == {}
    assert run_metrics["test"]["macro"] == {}


def test_artifact_saving(tmp_path, dataset_fp):
    pytest.importorskip("matplotlib")
    trainer = FPTrainer(model_cls=DummyModel)
    out_dir = tmp_path / "artifacts"
    trainer.fit(dataset_fp, output_dir=out_dir)
    assert (out_dir / "model").is_dir()
    assert (out_dir / "metrics.json").is_file()
    with (out_dir / "metrics.json").open() as f:
        data = json.load(f)
    assert "train" in data and "macro" in data["train"]
    assert (out_dir / "figures" / "log").is_dir()
    assert (out_dir / "figures" / "linear").is_dir()


def test_errors_missing_features_without_smiles():
    ds = DummyDataset(FeaturizationMethod.MORGAN_FP)
    ds.train = ds.train.drop(columns=["fp1", "fp2", "fp3", "smiles"])
    trainer = FPTrainer(model_cls=DummyModel)
    with pytest.raises(ValueError):
        trainer.prepare_features(cast(Any, ds))


def test_errors_missing_endpoints(dataset_fp):
    ds = dataset_fp
    ds.train = ds.train.drop(columns=["A", "B"])
    trainer = FPTrainer(model_cls=DummyModel)
    with pytest.raises(ValueError):
        trainer.prepare_targets(ds)
