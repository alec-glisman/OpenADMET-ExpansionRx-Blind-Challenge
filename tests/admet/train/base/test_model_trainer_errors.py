"""Edge-case tests for BaseModelTrainer error conditions."""

from typing import Any
import pytest
import pandas as pd
from admet.train.base import BaseModelTrainer, FeaturizationMethod


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
        self.train = pd.DataFrame(data_train)


class DummyEmptyDS:
    endpoints = []


class PipeTrainer(BaseModelTrainer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        # deliberately unsupported featurization for tests
        self.featurization = FeaturizationMethod.NONE


@pytest.fixture
def dataset_fp():
    return DummyDataset(FeaturizationMethod.MORGAN_FP)


def test_sample_weights_raises_if_dataset_column_missing(dataset_fp):
    ds = dataset_fp
    ds.train = ds.train.drop(columns=["Dataset"])
    trainer = PipeTrainer(model_cls=object)
    with pytest.raises(ValueError):
        trainer.build_sample_weights(ds, {"default": 1.0})


def test_prepare_features_unsupported_featurization(dataset_fp):
    trainer = PipeTrainer(model_cls=object)
    with pytest.raises(ValueError):
        trainer.prepare_features(dataset_fp)


def test_fit_requires_splits():
    trainer = PipeTrainer(model_cls=object)
    with pytest.raises(ValueError):
        trainer.fit(DummyEmptyDS())
