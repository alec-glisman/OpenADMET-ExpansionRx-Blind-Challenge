"""Edge-case tests for BaseModelTrainer error conditions.

Tests focus on missing dataset columns, unsupported featurization, and empty
dataset split handling to ensure clear errors are raised.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from admet.train.base import BaseModelTrainer, FeaturizationMethod


class DummyDataset:

    def __init__(self, featurization: FeaturizationMethod = FeaturizationMethod.MORGAN_FP):
        self.endpoints = ["A", "B"]
        self.fingerprint_cols = ["fp1", "fp2", "fp3"] if featurization == FeaturizationMethod.MORGAN_FP else []
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
        self.train = pd.DataFrame(data_train)


class DummyEmptyDS:
    endpoints: list[str] = []


class PipeTrainer(BaseModelTrainer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # deliberately unsupported featurization for tests
        self.featurization = FeaturizationMethod.NONE


@pytest.fixture
def dataset_fp() -> DummyDataset:
    return DummyDataset(FeaturizationMethod.MORGAN_FP)


@pytest.mark.unit
def test_sample_weights_raises_if_dataset_column_missing(dataset_fp: DummyDataset) -> None:
    ds = dataset_fp
    ds.train = ds.train.drop(columns=["Dataset"])
    trainer = PipeTrainer(model_cls=object)
    with pytest.raises(ValueError):
        trainer.build_sample_weights(ds, {"default": 1.0})


@pytest.mark.unit
def test_prepare_features_unsupported_featurization(dataset_fp: DummyDataset) -> None:
    trainer = PipeTrainer(model_cls=object)
    with pytest.raises(ValueError):
        trainer.prepare_features(dataset_fp)


@pytest.mark.unit
def test_fit_requires_splits(tmp_path: Path) -> None:
    trainer = PipeTrainer(model_cls=object)
    with pytest.raises(ValueError):
        trainer.fit(DummyEmptyDS(), output_dir=tmp_path / "out")
