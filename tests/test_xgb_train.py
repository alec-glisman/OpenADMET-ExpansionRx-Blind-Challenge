from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from admet.data.load import ENDPOINT_COLUMNS, expected_fingerprint_columns, load_dataset
from admet.train.xgb_train import train_xgb_models, train_xgb_models_ray, XGBoostTrainer
from admet.model.xgb_wrapper import XGBoostMultiEndpoint
from admet.model.base import BaseModel
from admet.train.base_trainer import infer_split_metadata
from admet.train._ray_test_stubs import (
    FailingTrainer,
    TrivialTrainer,
    SlowTrainer,
    PartialTrainer,
    MinimalRayTrainer,
    SlowRayTrainer,
    PartialRayTrainer,
)
from typing import Any, cast
import ray

# imported for monkeypatching in tests


def _make_hf_like_dataset(root: Path, n_rows: int = 30, n_bits: int = 16) -> Path:
    """Create a minimal on-disk HF DatasetDict layout that ``load_dataset`` can load."""

    fp_cols = expected_fingerprint_columns(n_bits)
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

    # Helper trainer classes moved to admet.train._ray_test_stubs


def test_infer_split_metadata_parses_cluster_split_fold(tmp_path: Path):
    root = tmp_path / "assets" / "dataset" / "splits" / "high_quality"
    path = root / "random_cluster" / "split_0" / "fold_1" / "hf_dataset"
    path.mkdir(parents=True)

    meta = infer_split_metadata(path, root)
    if "absolute_path" not in meta:
        pytest.fail("absolute_path missing from inferred metadata")
    if not str(meta["relative_path"]).endswith("random_cluster/split_0/fold_1/hf_dataset"):
        pytest.fail("relative_path not parsed as expected")
    if meta["split"] != "0":
        pytest.fail(f"split parsed incorrectly: {meta.get('split')}")
    if meta["fold"] != "1":
        pytest.fail(f"fold parsed incorrectly: {meta.get('fold')}")
    if meta["quality"] != "high_quality":
        pytest.fail(f"quality parsed incorrectly: {meta.get('quality')}")
    if meta["cluster"] != "high_quality/random_cluster":
        pytest.fail(f"cluster parsed incorrectly: {meta.get('cluster')}")


def test_train_xgb_models_runs_end_to_end(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir)

    dataset = load_dataset(data_dir, n_fingerprint_bits=16)
    metrics = train_xgb_models(dataset, model_params={"n_estimators": 10}, early_stopping_rounds=5)

    for split in ["train", "validation", "test"]:
        if split not in metrics:
            pytest.fail(f"Expected split {split} in metrics")
        if "macro" not in metrics[split]:
            pytest.fail(f"Expected macro in metrics[{split}]")


def test_xgb_trainer_runs_end_to_end(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir)

    dataset = load_dataset(data_dir, n_fingerprint_bits=16)
    trainer = XGBoostTrainer(
        model_cls=cast(Any, XGBoostMultiEndpoint), model_params={"n_estimators": 10}, seed=123
    )

    metrics = trainer.run(dataset, early_stopping_rounds=5)

    for split in ["train", "validation", "test"]:
        if split not in metrics:
            pytest.fail(f"Expected split {split} in metrics")
        if "macro" not in metrics[split]:
            pytest.fail(f"Expected macro in metrics[{split}]")


def test_build_model_uses_provided_model_cls():
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

    trainer = XGBoostTrainer(model_cls=cast(Any, FakeModel), model_params={"n_estimators": 10}, seed=123)
    model = trainer.build_model(["A", "B"])  # should be FakeModel
    if not isinstance(model, FakeModel):
        pytest.fail("Built model is not instance of provided FakeModel class")


def test_build_sample_weights_vectorized(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=10, n_bits=16)

    dataset = load_dataset(data_dir, n_fingerprint_bits=16)
    mapping = {"dataset_a": 2.0, "default": 1.0}
    trainer = XGBoostTrainer(
        model_cls=cast(Any, XGBoostMultiEndpoint), model_params={"n_estimators": 10}, seed=123
    )
    sw = trainer.build_sample_weights(dataset, mapping)
    if sw is None:
        pytest.fail("Sample weights vector should not be None")
    if sw.shape[0] != len(dataset.train):
        pytest.fail("Sample weights shape does not match training set length")
    if not all(sw == 2.0):
        pytest.fail("Sample weights are not all 2.0 as expected")


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
        n_fingerprint_bits=16,
    )

    # Expect one entry per hf_dataset
    if len(results) != 2:
        pytest.fail(f"Expected 2 results from ray trainer, got {len(results)}")

    # Check that per-dataset metrics and meta exist
    for payload in results.values():
        if not isinstance(payload, dict):
            pytest.fail("Expected payload dict from ray worker")
        if "metrics" not in payload:
            pytest.fail("Expected payload to contain metrics")
        if "meta" not in payload:
            pytest.fail("Expected payload to contain meta")
        metrics_obj = payload["metrics"]
        if metrics_obj is None:
            pytest.fail(f"Remote worker failed for {payload.get('meta')}: {payload.get('error')}")
        if not isinstance(metrics_obj, dict):
            pytest.fail("Expected metrics object dict")
        for split in ["train", "validation", "test"]:
            if split not in metrics_obj:
                pytest.fail(f"Expected split {split} in metrics_obj")
            if "macro" not in metrics_obj[split]:
                pytest.fail(f"Expected macro in metrics_obj[{split}]")
        # Ensure status is present and indicates ok
        if payload.get("status") != "ok":
            pytest.fail(f"Expected OK status in payload; got: {payload.get('status')}")
        # Timing info included in payload
        if payload.get("start_time") is None:
            pytest.fail("Expected start_time in payload")
        if payload.get("end_time") is None:
            pytest.fail("Expected end_time in payload")
        if not isinstance(payload.get("duration_seconds"), float):
            pytest.fail("Expected duration_seconds to be float in payload")

    # Summary files should exist at the output root
    summary_csv = output_root / "metrics_summary.csv"
    summary_json = output_root / "metrics_summary.json"
    if not (summary_csv.exists() and summary_csv.stat().st_size > 0):
        pytest.fail("Expected non-empty metrics_summary.csv file")
    if not (summary_json.exists() and summary_json.stat().st_size > 0):
        pytest.fail("Expected non-empty metrics_summary.json file")

    # Summary CSV should have one row per (dataset, split)
    df = pd.read_csv(summary_csv)
    if df.shape[0] != 2 * 3:
        pytest.fail(f"Expected {2*3} rows in summary, got {df.shape[0]}")
    if "status" not in df.columns:
        pytest.fail("Expected status column in summary CSV")
    # times and duration included
    if "start_time" not in df.columns:
        pytest.fail("Expected start_time column in summary CSV")
    if "end_time" not in df.columns:
        pytest.fail("Expected end_time column in summary CSV")
    if "duration_seconds" not in df.columns:
        pytest.fail("Expected duration_seconds column in summary CSV")


def test_train_xgb_models_ray_handles_remote_errors(tmp_path: Path):
    # Create two synthetic hf_dataset directories under a random_cluster root
    root = tmp_path / "assets" / "dataset" / "splits" / "high_quality" / "random_cluster"
    for split_idx in range(2):
        for fold_idx in range(1):
            ds_dir = root / f"split_{split_idx}" / f"fold_{fold_idx}" / "hf_dataset"
            ds_dir.mkdir(parents=True)
            _make_hf_like_dataset(ds_dir)

    # Use stub FailingTrainer & MinimalRayTrainer from package stubs
    ray_trainer = MinimalRayTrainer(trainer_cls=cast(Any, FailingTrainer), trainer_kwargs={})
    results = ray_trainer.run_all(root, num_cpus=1)

    # Each payload should include error and have no metrics
    if len(results) != 2:
        pytest.fail(f"Expected 2 results for failing trainer, got {len(results)}")
    for payload in results.values():
        if not isinstance(payload, dict):
            pytest.fail("Expected payload to be a dict")
        if "metrics" not in payload:
            pytest.fail("Expected payload to contain metrics key")
        if payload["metrics"] is not None:
            pytest.fail("Expected metrics to be None for failing trainer payload")
        if "error" not in payload:
            pytest.fail("Expected error key in payload for failing trainer")
        if payload.get("status") != "error":
            pytest.fail(f"Expected status 'error' in payload but got {payload.get('status')}")


def test_ray_shutdown_after_run_all(tmp_path: Path):
    # create a single dataset
    root = tmp_path / "assets" / "dataset" / "splits" / "high_quality" / "random_cluster"
    ds_dir = root / "split_0" / "fold_0" / "hf_dataset"
    ds_dir.mkdir(parents=True)
    _make_hf_like_dataset(ds_dir)

    # Ensure no active Ray
    if ray.is_initialized():
        ray.shutdown()

    # Make sure Ray is not running, then run all and assert it's stopped afterward
    if ray.is_initialized():
        ray.shutdown()
    ray_trainer = MinimalRayTrainer(trainer_cls=TrivialTrainer, trainer_kwargs={})
    _ = ray_trainer.run_all(root, num_cpus=1)
    if ray.is_initialized():
        pytest.fail("Ray should be shut down after run_all")


def test_dry_run_returns_minimal_metrics(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=10, n_bits=16)

    dataset = load_dataset(data_dir, n_fingerprint_bits=16)
    trainer = XGBoostTrainer(
        model_cls=cast(Any, XGBoostMultiEndpoint), model_params={"n_estimators": 10}, seed=123
    )
    metrics = trainer.run(dataset, dry_run=True)
    if set(metrics.keys()) != {"train", "validation", "test"}:
        pytest.fail(f"Expected metrics keys to be train/validation/test; got {set(metrics.keys())}")
    for split in ["train", "validation", "test"]:
        if "macro" not in metrics[split]:
            pytest.fail(f"Expected macro in metrics[{split}]")


def test_train_xgb_models_ray_dry_run_skipped_status(tmp_path: Path):
    # Create a small dataset and run with dry_run True
    root = tmp_path / "assets" / "dataset" / "splits" / "high_quality" / "random_cluster"
    for split_idx in range(1):
        ds_dir = root / f"split_{split_idx}" / "fold_0" / "hf_dataset"
        ds_dir.mkdir(parents=True)
        _make_hf_like_dataset(ds_dir)

    out = tmp_path / "out"
    results = train_xgb_models_ray(
        root,
        model_params={"n_estimators": 5},
        output_root=out,
        seed=42,
        num_cpus=1,
        dry_run=True,
        n_fingerprint_bits=16,
    )
    if not results:
        pytest.fail("Expected results from dry_run invocation")
    for payload in results.values():
        if payload.get("status") != "skipped":
            pytest.fail(f"Expected status 'skipped' in payload, got {payload.get('status')}")
        if payload.get("start_time") is None:
            pytest.fail("Expected start_time in skipped payload")
        if payload.get("end_time") is None:
            pytest.fail("Expected end_time in skipped payload")
        if not isinstance(payload.get("duration_seconds"), float):
            pytest.fail("Expected duration_seconds to be float in skipped payload")


def test_train_xgb_models_ray_timeout_status(tmp_path: Path):
    # Create a single dataset
    root = tmp_path / "assets" / "dataset" / "splits" / "high_quality" / "random_cluster"
    ds_dir = root / "split_0" / "fold_0" / "hf_dataset"
    ds_dir.mkdir(parents=True)
    _make_hf_like_dataset(ds_dir)

    # Create a slow trainer that sleeps in run so we can trigger timeout
    # time import removed; slow behavior handled in stub trainers

    slow_ray_trainer = SlowRayTrainer(trainer_cls=SlowTrainer, trainer_kwargs={})
    results = slow_ray_trainer.run_all(root, num_cpus=1, max_duration_seconds=0.0, n_fingerprint_bits=16)
    if not results:
        pytest.fail("Expected results from timeout run_all invocation")
    for payload in results.values():
        if payload.get("status") != "timeout":
            pytest.fail(f"Expected timeout payload but got {payload.get('status')}: {payload.get('error')}")
        if payload.get("start_time") is None:
            pytest.fail("Expected start_time in timeout payload")
        if payload.get("end_time") is None:
            pytest.fail("Expected end_time in timeout payload")
        if not isinstance(payload.get("duration_seconds"), float):
            pytest.fail("Expected duration_seconds to be float in timeout payload")


def test_train_xgb_models_ray_partial_status(tmp_path: Path):
    # Create a single dataset
    root = tmp_path / "assets" / "dataset" / "splits" / "high_quality" / "random_cluster"
    ds_dir = root / "split_0" / "fold_0" / "hf_dataset"
    ds_dir.mkdir(parents=True)
    _make_hf_like_dataset(ds_dir)

    partial_trainer = PartialRayTrainer(trainer_cls=PartialTrainer, trainer_kwargs={"model_cls": BaseModel})
    results = partial_trainer.run_all(root, num_cpus=1, n_fingerprint_bits=16)
    if not results:
        pytest.fail("Expected results from partial status run_all invocation")
    for payload in results.values():
        if payload.get("status") != "partial":
            pytest.fail(f"Expected partial payload but got {payload.get('status')}: {payload.get('error')}")
        assert payload.get("start_time") is not None
        assert payload.get("end_time") is not None
        assert isinstance(payload.get("duration_seconds"), float)


def test_missing_fingerprint_columns_errors(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=10, n_bits=16)
    dataset = load_dataset(data_dir, n_fingerprint_bits=16)
    dataset.fingerprint_cols = []
    trainer = XGBoostTrainer(
        model_cls=cast(Any, XGBoostMultiEndpoint), model_params={"n_estimators": 10}, seed=123
    )
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
        model_cls=cast(Any, XGBoostMultiEndpoint),
        model_params={"n_estimators": 1, "device": "cuda"},
        seed=123,
    )
    metrics = trainer.run(dataset, early_stopping_rounds=1)
    # Training completed even though the first call raised an exception due to fallback
    for split in ["train", "validation", "test"]:
        assert split in metrics
        assert "macro" in metrics[split]


def test_save_artifacts_receives_outputs(tmp_path: Path):
    # setup dataset
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=20, n_bits=16)
    dataset = load_dataset(data_dir, n_fingerprint_bits=16)

    captured = []

    class CaptureTrainer(XGBoostTrainer):
        def save_artifacts(self, model, metrics, output_dir, outputs, *, dataset, extra_meta=None):
            # store the outputs for inspection
            captured.append(outputs)
            super().save_artifacts(
                model, metrics, output_dir, outputs, dataset=dataset, extra_meta=extra_meta
            )

    trainer = CaptureTrainer(
        model_cls=cast(Any, XGBoostMultiEndpoint), model_params={"n_estimators": 5}, seed=42
    )
    out = tmp_path / "out"
    _ = trainer.run(dataset, output_dir=out)
    # ensure we captured outputs
    assert captured, "save_artifacts did not capture outputs"
    outputs = captured[0]
    from admet.train.base_trainer import RunOutputs

    assert isinstance(outputs, RunOutputs)
    # shapes are consistent
    assert outputs.Y_train.shape[0] == len(dataset.train)
    assert outputs.Y_train.shape[1] == len(dataset.endpoints)
    assert outputs.pred_train.shape[0] == len(dataset.train)
    assert outputs.mask_train.shape == outputs.Y_train.shape


def test_build_sample_weights_missing_dataset_column_raises(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=10, n_bits=16)
    ds = load_dataset(data_dir, n_fingerprint_bits=16)
    # Remove Dataset column from training split
    ds.train = ds.train.drop(columns=["Dataset"])

    # cast and Any are imported at the module level (to satisfy typing/Protocol casts)

    trainer = XGBoostTrainer(
        model_cls=cast(Any, XGBoostMultiEndpoint),
        model_params={"n_estimators": 5},
        seed=1,
    )
    with pytest.raises(ValueError):
        trainer.build_sample_weights(ds, {"default": 1.0, "dataset_a": 2.0})


__all__ = [
    "test_infer_split_metadata_parses_cluster_split_fold",
    "test_train_xgb_models_runs_end_to_end",
    "test_train_xgb_models_ray_multiple_datasets_and_summary",
]
