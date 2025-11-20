from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from admet.data.load import ENDPOINT_COLUMNS, expected_fingerprint_columns, load_dataset
from admet.train.xgb_train import train_xgb_models, train_xgb_models_ray, XGBoostTrainer
from admet.train.base_trainer import infer_split_metadata
from admet.model.xgb_wrapper import XGBoostMultiEndpoint
from admet.model.base import BaseModel
from admet.train.base_trainer import BaseRayMultiDatasetTrainer, BaseModelTrainer
import ray

# imported for monkeypatching in tests


def _make_hf_like_dataset(root: Path, n_rows: int = 30, n_bits: int = 16) -> Path:
    """Create a minimal on-disk HF DatasetDict layout that ``load_dataset`` can load."""

    fp_cols = expected_fingerprint_columns(2048)
    splits = {}
    for split in ["train", "validation", "test"]:
        data = {
            "Molecule Name": [f"MOL_{i}" for i in range(n_rows)],
            "SMILES": ["C" for _ in range(n_rows)],
            "Dataset": ["dataset_a" for _ in range(n_rows)],
        }
        for ep in ENDPOINT_COLUMNS:
            vals = np.random.randn(n_rows).astype(float)
            if split == "train":
                vals[0] = np.nan
            data[ep] = list(vals)
        for col in fp_cols:
            data[col] = list(np.random.randn(n_rows).astype(float))
        splits[split] = Dataset.from_pandas(pd.DataFrame(data), preserve_index=False)

    dset = DatasetDict(splits)
    dset.save_to_disk(str(root))
    return root


def test_infer_split_metadata_parses_cluster_split_fold(tmp_path: Path):
    root = tmp_path / "assets" / "dataset" / "splits" / "high_quality"
    path = root / "random_cluster" / "split_0" / "fold_1" / "hf_dataset"
    path.mkdir(parents=True)

    meta = infer_split_metadata(path, root)
    assert "absolute_path" in meta
    assert meta["relative_path"].endswith("random_cluster/split_0/fold_1/hf_dataset")
    assert meta["split"] == "0"
    assert meta["fold"] == "1"
    assert meta["quality"] == "high_quality"
    assert meta["cluster"] == "high_quality/random_cluster"


def test_train_xgb_models_runs_end_to_end(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir)

    dataset = load_dataset(data_dir, n_fingerprint_bits=16)
    metrics = train_xgb_models(dataset, model_params={"n_estimators": 10}, early_stopping_rounds=5)

    for split in ["train", "validation", "test"]:
        assert split in metrics
        assert "macro" in metrics[split]


def test_xgb_trainer_runs_end_to_end(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir)

    dataset = load_dataset(data_dir, n_fingerprint_bits=16)
    trainer = XGBoostTrainer(model_cls=XGBoostMultiEndpoint, model_params={"n_estimators": 10}, seed=123)

    metrics = trainer.run(dataset, early_stopping_rounds=5)

    for split in ["train", "validation", "test"]:
        assert split in metrics
        assert "macro" in metrics[split]


def test_build_model_uses_provided_model_cls(tmp_path: Path):
    # Minimal fake model to verify dependency injection
    class FakeModel(BaseModel):
        def __init__(self, endpoints, model_params=None, random_state=None):
            self.endpoints = list(endpoints)
            self.model_params = model_params or {}
            self.random_state = random_state

        def fit(
            self,
            X_train,
            Y_train,
            *,
            Y_mask=None,
            X_val=None,
            Y_val=None,
            Y_val_mask=None,
            sample_weight=None,
            early_stopping_rounds=None,
        ):
            return None

        def predict(self, X):
            import numpy as np

            return np.zeros((X.shape[0], len(self.endpoints)))

        def save(self, path):
            return None

        @classmethod
        def load(cls, path):
            return cls([], {})

        def get_config(self):
            return {}

        def get_metadata(self):
            return {}

    trainer = XGBoostTrainer(model_cls=FakeModel, model_params={"n_estimators": 10}, seed=123)
    model = trainer.build_model(["A", "B"])  # should be FakeModel
    assert isinstance(model, FakeModel)


def test_build_sample_weights_vectorized(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=10, n_bits=16)

    dataset = load_dataset(data_dir, n_fingerprint_bits=16)
    mapping = {"dataset_a": 2.0, "default": 1.0}
    trainer = XGBoostTrainer(model_cls=XGBoostMultiEndpoint, model_params={"n_estimators": 10}, seed=123)
    sw = trainer.build_sample_weights(dataset, mapping)
    assert sw is not None
    assert sw.shape[0] == len(dataset.train)
    assert all(sw == 2.0)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_train_xgb_models_ray_multiple_datasets_and_summary(tmp_path: Path):
    # Create two synthetic hf_dataset directories under a random_cluster root
    root = tmp_path / "assets" / "dataset" / "splits" / "high_quality" / "random_cluster"
    for split_idx in range(2):
        for fold_idx in range(1):
            ds_dir = root / f"split_{split_idx}" / f"fold_{fold_idx}" / "hf_dataset"
            ds_dir.mkdir(parents=True)
            _make_hf_like_dataset(ds_dir)

    output_root = tmp_path / "xgb_artifacts"
    results = train_xgb_models_ray(
        root,
        model_params={"n_estimators": 10},
        early_stopping_rounds=5,
        output_root=output_root,
        seed=123,
        num_cpus=1,
    )

    # Expect one entry per hf_dataset
    assert len(results) == 2

    # Check that per-dataset metrics and meta exist
    for payload in results.values():
        assert isinstance(payload, dict)
        assert "metrics" in payload
        assert "meta" in payload
        metrics_obj = payload["metrics"]
        assert isinstance(metrics_obj, dict)
        for split in ["train", "validation", "test"]:
            assert split in metrics_obj
            assert "macro" in metrics_obj[split]

    # Summary files should exist at the output root
    summary_csv = output_root / "metrics_summary.csv"
    summary_json = output_root / "metrics_summary.json"
    assert summary_csv.exists() and summary_csv.stat().st_size > 0
    assert summary_json.exists() and summary_json.stat().st_size > 0

    # Summary CSV should have one row per (dataset, split)
    df = pd.read_csv(summary_csv)
    assert df.shape[0] == 2 * 3  # 2 datasets * 3 splits


def test_train_xgb_models_ray_handles_remote_errors(tmp_path: Path):
    # Create two synthetic hf_dataset directories under a random_cluster root
    root = tmp_path / "assets" / "dataset" / "splits" / "high_quality" / "random_cluster"
    for split_idx in range(2):
        for fold_idx in range(1):
            ds_dir = root / f"split_{split_idx}" / f"fold_{fold_idx}" / "hf_dataset"
            ds_dir.mkdir(parents=True)
            _make_hf_like_dataset(ds_dir)

    # Define a failing trainer that raises during run
    class FailingTrainer(BaseModelTrainer):
        def prepare_features(self, dataset):
            import numpy as _np

            X_train = _np.zeros((len(dataset.train), 1))
            X_val = _np.zeros((len(dataset.val), 1))
            X_test = _np.zeros((len(dataset.test), 1))
            return X_train, X_val, X_test

        def prepare_targets(self, dataset):
            import numpy as _np

            D = len(dataset.endpoints)
            Y_train = _np.zeros((len(dataset.train), D))
            Y_val = _np.zeros((len(dataset.val), D))
            Y_test = _np.zeros((len(dataset.test), D))
            return Y_train, Y_val, Y_test

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
            # return a minimal fake model implementing BaseModel interface
            class MinimalModel(BaseModel):
                def __init__(self):
                    self.endpoints = list(endpoints)

                def fit(
                    self,
                    X_train,
                    Y_train,
                    *,
                    Y_mask=None,
                    X_val=None,
                    Y_val=None,
                    Y_val_mask=None,
                    sample_weight=None,
                    early_stopping_rounds=None,
                ):
                    return None

                def predict(self, X):
                    import numpy as _np

                    return _np.zeros((X.shape[0], len(self.endpoints)))

                def save(self, path):
                    return None

                @classmethod
                def load(cls, path):
                    return cls()

                def get_config(self):
                    return {}

                def get_metadata(self):
                    return {}

            return MinimalModel()

        def compute_metrics(self, *args, **kwargs):
            return {}

        def save_artifacts(self, model, metrics, output_dir, *, dataset, extra_meta=None):
            return None

        def run(self, dataset, *args, **kwargs):
            # Explicitly raise during run so the Ray worker triggers error handling
            raise RuntimeError("failing trainer run")

    # Build a small Ray-based runner using our failing trainer
    class MinimalRayTrainer(BaseRayMultiDatasetTrainer):
        def discover_datasets(self, root):
            return [p for p in root.rglob("hf_dataset") if p.is_dir()]

        def build_output_dir(self, base, meta):
            cluster = str(meta.get("cluster", "unknown_method"))
            split = str(meta.get("split", "unknown_split"))
            fold = str(meta.get("fold", "unknown_fold"))
            return base / cluster / f"split_{split}" / f"fold_{fold}"

    ray_trainer = MinimalRayTrainer(trainer_cls=FailingTrainer, trainer_kwargs={})
    results = ray_trainer.run_all(root, num_cpus=1)

    # Each payload should include error and have no metrics
    assert len(results) == 2
    for payload in results.values():
        assert isinstance(payload, dict)
        assert "metrics" in payload
        assert payload["metrics"] is None
        assert "error" in payload


def test_ray_shutdown_after_run_all(tmp_path: Path):
    # create a single dataset
    root = tmp_path / "assets" / "dataset" / "splits" / "high_quality" / "random_cluster"
    ds_dir = root / "split_0" / "fold_0" / "hf_dataset"
    ds_dir.mkdir(parents=True)
    _make_hf_like_dataset(ds_dir)

    # Ensure no active Ray
    if ray.is_initialized():
        ray.shutdown()

    class TrivialTrainer(BaseModelTrainer):
        def prepare_features(self, dataset):
            import numpy as _np

            X_train = _np.zeros((len(dataset.train), 1))
            X_val = _np.zeros((len(dataset.val), 1))
            X_test = _np.zeros((len(dataset.test), 1))
            return X_train, X_val, X_test

        def prepare_targets(self, dataset):
            import numpy as _np

            D = len(dataset.endpoints)
            Y_train = _np.zeros((len(dataset.train), D))
            Y_val = _np.zeros((len(dataset.val), D))
            Y_test = _np.zeros((len(dataset.test), D))
            return Y_train, Y_val, Y_test

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
            class MinimalModel(BaseModel):
                def __init__(self):
                    self.endpoints = list(endpoints)

                def fit(
                    self,
                    X_train,
                    Y_train,
                    *,
                    Y_mask=None,
                    X_val=None,
                    Y_val=None,
                    Y_val_mask=None,
                    sample_weight=None,
                    early_stopping_rounds=None,
                ):
                    return None

                def predict(self, X):
                    import numpy as _np

                    return _np.zeros((X.shape[0], len(self.endpoints)))

                def save(self, path):
                    return None

                @classmethod
                def load(cls, path):
                    return cls()

                def get_config(self):
                    return {}

                def get_metadata(self):
                    return {}

            return MinimalModel()

        def compute_metrics(self, *args, **kwargs):
            return {}

        def save_artifacts(self, model, metrics, output_dir, *, dataset, extra_meta=None):
            return None

        def run(self, dataset, *args, **kwargs):
            return {"train": {"macro": {}}, "validation": {"macro": {}}, "test": {"macro": {}}}

    class MinimalRayTrainer(BaseRayMultiDatasetTrainer):
        def discover_datasets(self, root):
            return [p for p in root.rglob("hf_dataset") if p.is_dir()]

        def build_output_dir(self, base, meta):
            cluster = str(meta.get("cluster", "unknown_method"))
            split = str(meta.get("split", "unknown_split"))
            fold = str(meta.get("fold", "unknown_fold"))
            return base / cluster / f"split_{split}" / f"fold_{fold}"

    # Make sure Ray is not running, then run all and assert it's stopped afterward
    if ray.is_initialized():
        ray.shutdown()
    ray_trainer = MinimalRayTrainer(trainer_cls=TrivialTrainer, trainer_kwargs={})
    _ = ray_trainer.run_all(root, num_cpus=1)
    assert not ray.is_initialized()


def test_dry_run_returns_minimal_metrics(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=10, n_bits=16)

    dataset = load_dataset(data_dir, n_fingerprint_bits=16)
    trainer = XGBoostTrainer(model_cls=XGBoostMultiEndpoint, model_params={"n_estimators": 10}, seed=123)
    metrics = trainer.run(dataset, dry_run=True)
    assert set(metrics.keys()) == {"train", "validation", "test"}
    for split in ["train", "validation", "test"]:
        assert "macro" in metrics[split]


def test_missing_fingerprint_columns_errors(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=10, n_bits=16)
    dataset = load_dataset(data_dir, n_fingerprint_bits=16)
    dataset.fingerprint_cols = []
    trainer = XGBoostTrainer(model_cls=XGBoostMultiEndpoint, model_params={"n_estimators": 10}, seed=123)
    with pytest.raises(ValueError):
        trainer.run(dataset)


def test_xgb_gpu_fallback_retry(tmp_path: Path, monkeypatch):
    # Create a small dataset
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=10, n_bits=16)
    dataset = load_dataset(data_dir, n_fingerprint_bits=16)

    # Fake regressor to simulate failure on first fit then succeed
    class FakeRegressor:
        call_count = 0

        def __init__(self, **params):
            self.params = params

        def fit(self, X, y, **kwargs):
            FakeRegressor.call_count += 1
            if FakeRegressor.call_count == 1:
                raise RuntimeError("fake GPU failure")
            return None

        def predict(self, X):
            import numpy as _np

            return _np.zeros(X.shape[0])

        def save_model(self, path):
            return None

        def load_model(self, path):
            return None

    monkeypatch.setattr("admet.model.xgb_wrapper.XGBRegressor", FakeRegressor)

    trainer = XGBoostTrainer(
        model_cls=XGBoostMultiEndpoint,
        model_params={"n_estimators": 1, "device": "cuda"},
        seed=123,
    )
    metrics = trainer.run(dataset, early_stopping_rounds=1)
    # Training completed even though the first call raised an exception due to fallback
    for split in ["train", "validation", "test"]:
        assert split in metrics
        assert "macro" in metrics[split]


__all__ = [
    "test_infer_split_metadata_parses_cluster_split_fold",
    "test_train_xgb_models_runs_end_to_end",
    "test_train_xgb_models_ray_multiple_datasets_and_summary",
]
