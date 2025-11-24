"""Integration tests for train_model/train_ensemble.

pyright: ignore-all
flake8: noqa
mypy: ignore-errors
"""

from pathlib import Path
import json
from typing import Sequence, Any, cast
import numpy as np
import pandas as pd
import pytest
from admet.train.base import (
    BaseModelTrainer,
    BaseEnsembleTrainer,
    FeaturizationMethod,
    train_model,
)


class DummyModel:
    endpoints: Sequence[str]
    input_type: str

    def __init__(self, endpoints, params, seed):
        self.endpoints = endpoints
        self.params = params or {}
        self.seed = seed
        self.input_type = "fingerprint"

    def fit(self, X_train, Y_train, **kwargs):  # type: ignore[unused-argument]
        pass

    def predict(self, X):  # type: ignore[unused-argument]
        return np.zeros((X.shape[0], len(self.endpoints)))

    def save(self, path: str):  # type: ignore[unused-argument]
        Path(path).mkdir(parents=True, exist_ok=True)
        (Path(path) / "dummy.json").write_text(json.dumps({"endpoints": self.endpoints}))

    @classmethod
    def load(cls, path: str):  # type: ignore[override]
        return cls(endpoints=["A"], params={}, seed=None)


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
        endpoints: list[str]
        fingerprint_cols: list[str]
        smiles_col: str
        train: pd.DataFrame
        val: pd.DataFrame
        test: pd.DataFrame

    DS.endpoints = ["A"]
    DS.fingerprint_cols = ["fp1", "fp2", "fp3"]
    DS.smiles_col = "smiles"
    DS.train = train
    DS.val = val
    DS.test = test

    return DS()


@pytest.fixture
def synthetic_dataset():
    return make_dataset()


def test_train_model_end_to_end(tmp_path: Path, synthetic_dataset) -> None:
    run_metrics, summary = train_model(
        synthetic_dataset,
        trainer_cls=FPTrainer,
        model_cls=cast(Any, DummyModel),
        model_params={"alpha": 0.1},
        output_dir=tmp_path / "single",
        early_stopping_rounds=10,
    )
    assert set(run_metrics.keys()) == {"train", "validation", "test"}
    assert (tmp_path / "single" / "model").is_dir()
    assert (tmp_path / "single" / "metrics.json").is_file()


def test_discover_datasets(tmp_path: Path) -> None:
    root = tmp_path / "root"
    d1 = root / "clusterA" / "split_train" / "fold_0" / "hf_dataset"
    d2 = root / "clusterB" / "split_test" / "fold_1" / "hf_dataset"
    d3 = root / "other" / "file.txt"
    d1.mkdir(parents=True)
    d2.mkdir(parents=True)
    d3.parent.mkdir(parents=True)
    d3.write_text("not a dataset")
    trainer = BaseEnsembleTrainer(trainer_cls=FPTrainer)
    datasets = trainer.discover_datasets(root)
    assert len(datasets) == 2
    assert d1 in datasets
    assert d2 in datasets
    assert d3 not in datasets
