"""Comprehensive tests for XGBoost training & Ray orchestration.

pyright: ignore-all
flake8: noqa
mypy: ignore-errors
"""

from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from datasets import Dataset, DatasetDict
from typing import Any, cast
import ray

from admet.data.load import ENDPOINT_COLUMNS, expected_fingerprint_columns, load_dataset
from admet.train.xgb_train import XGBoostTrainer
from admet.train.base import (
    train_model,
    train_ensemble,
    BaseEnsembleTrainer,
    infer_split_metadata,
    RunSummary,
)
from admet.model.xgb_wrapper import XGBoostMultiEndpoint
from admet.model.base import BaseModel
from admet.train._ray_test_stubs import (
    FailingTrainer,
    TrivialTrainer,
    SlowTrainer,
    PartialTrainer,
    MinimalRayTrainer,
    SlowRayTrainer,
    PartialRayTrainer,
)


def _make_hf_like_dataset(root: Path, n_rows: int = 30, n_bits: int = 16) -> Path:
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


def test_infer_split_metadata_parses_cluster_split_fold(tmp_path: Path):
    root = tmp_path / "assets" / "dataset" / "splits" / "high_quality"
    path = root / "random_cluster" / "split_0" / "fold_1" / "hf_dataset"
    path.mkdir(parents=True)
    meta = infer_split_metadata(path, root)
    assert "absolute_path" in meta
    assert str(meta["relative_path"]).endswith("random_cluster/split_0/fold_1/hf_dataset")
    assert meta["split"] == "0"
    assert meta["fold"] == "1"
    assert meta["quality"] == "high_quality"
    assert meta["cluster"] == "high_quality/random_cluster"


def test_train_model_runs_end_to_end(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir)
    dataset = load_dataset(data_dir, n_fingerprint_bits=16)
    out = tmp_path / "out"
    run_metrics, summary = train_model(
        dataset,
        trainer_cls=XGBoostTrainer,
        model_cls=cast(Any, XGBoostMultiEndpoint),
        model_params={"n_estimators": 10},
        early_stopping_rounds=5,
        output_dir=out,
    )
    for split in ["train", "validation", "test"]:
        assert split in run_metrics and "macro" in run_metrics[split]


def test_xgb_trainer_fit_runs_end_to_end(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir)
    dataset = load_dataset(data_dir, n_fingerprint_bits=16)
    trainer = XGBoostTrainer(
        model_cls=cast(Any, XGBoostMultiEndpoint),
        model_params={"n_estimators": 10},
        seed=123,
    )
    out = tmp_path / "out"
    run_metrics, summary = trainer.fit(dataset, early_stopping_rounds=5, output_dir=out)
    for split in ["train", "validation", "test"]:
        assert split in run_metrics and "macro" in run_metrics[split]


def test_build_model_uses_provided_model_cls():
    class FakeModel(BaseModel):
        def __init__(self, endpoints, model_params=None, random_state=None):
            self.endpoints = list(endpoints)
            self.model_params = model_params or {}
            self.random_state = random_state

        def fit(self, X_train, Y_train, **kwargs):
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
    model = trainer.build_model(["A", "B"])
    assert isinstance(model, FakeModel)


def test_build_sample_weights_vectorized(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=10, n_bits=16)
    dataset = load_dataset(data_dir, n_fingerprint_bits=16)
    mapping = {"dataset_a": 2.0, "default": 1.0}
    trainer = XGBoostTrainer(
        model_cls=cast(Any, XGBoostMultiEndpoint),
        model_params={"n_estimators": 10},
        seed=123,
    )
    sw = trainer.build_sample_weights(dataset, mapping)
    assert sw is not None and sw.shape[0] == len(dataset.train) and all(sw == 2.0)


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_train_xgb_models_ray_multiple_datasets_and_summary(tmp_path: Path):
    root = tmp_path / "assets" / "dataset" / "splits" / "high_quality" / "random_cluster"
    for split_idx in range(2):
        for fold_idx in range(1):
            ds_dir = root / f"split_{split_idx}" / f"fold_{fold_idx}" / "hf_dataset"
            ds_dir.mkdir(parents=True)
            _make_hf_like_dataset(ds_dir)
    output_root = tmp_path / "xgb_artifacts"
    results = train_ensemble(
        root,
        ensemble_trainer_cls=BaseEnsembleTrainer,
        trainer_cls=XGBoostTrainer,
        model_cls=cast(Any, XGBoostMultiEndpoint),
        model_params={"n_estimators": 10},
        early_stopping_rounds=5,
        output_root=output_root,
        seed=123,
        num_cpus=1,
        n_fingerprint_bits=16,
    )
    assert len(results) == 2
    for payload in results.values():
        run_metrics_obj = payload["run_metrics"]
        assert isinstance(run_metrics_obj, dict)
        for split in ["train", "validation", "test"]:
            assert split in run_metrics_obj and "macro" in run_metrics_obj[split]
        assert payload.get("status") == "ok"
        assert payload.get("start_time")
        assert payload.get("end_time")
        assert isinstance(payload.get("duration_seconds"), float)
    summary_csv = output_root / "metrics_summary.csv"
    summary_json = output_root / "metrics_summary.json"
    assert summary_csv.exists() and summary_csv.stat().st_size > 0
    assert summary_json.exists() and summary_json.stat().st_size > 0


def test_train_xgb_models_ray_handles_remote_errors(tmp_path: Path):
    root = tmp_path / "assets" / "dataset" / "splits" / "high_quality" / "random_cluster"
    for split_idx in range(2):
        for fold_idx in range(1):
            ds_dir = root / f"split_{split_idx}" / f"fold_{fold_idx}" / "hf_dataset"
            ds_dir.mkdir(parents=True)
            _make_hf_like_dataset(ds_dir)
    ray_trainer = MinimalRayTrainer(trainer_cls=cast(Any, FailingTrainer), trainer_kwargs={})
    out = tmp_path / "out"
    results = ray_trainer.fit_ensemble(root, num_cpus=1, output_root=out)
    assert len(results) == 2
    for payload in results.values():
        assert payload["run_metrics"] is None
        assert "error" in payload
        assert payload.get("status") == "error"


def test_ray_shutdown_after_run_all(tmp_path: Path):
    root = tmp_path / "assets" / "dataset" / "splits" / "high_quality" / "random_cluster"
    ds_dir = root / "split_0" / "fold_0" / "hf_dataset"
    ds_dir.mkdir(parents=True)
    _make_hf_like_dataset(ds_dir)
    if ray.is_initialized():
        ray.shutdown()
    ray_trainer = MinimalRayTrainer(trainer_cls=TrivialTrainer, trainer_kwargs={})
    _ = ray_trainer.fit_ensemble(root, num_cpus=1, output_root=tmp_path / "out")
    assert not ray.is_initialized(), "Ray should be shut down after run_all"


def test_dry_run_returns_minimal_metrics(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=10, n_bits=16)
    dataset = load_dataset(data_dir, n_fingerprint_bits=16)
    trainer = XGBoostTrainer(
        model_cls=cast(Any, XGBoostMultiEndpoint),
        model_params={"n_estimators": 10},
        seed=123,
    )
    out = tmp_path / "out"
    run_metrics, summary = trainer.fit(dataset, dry_run=True, output_dir=out)
    assert set(run_metrics.keys()) == {"train", "validation", "test"}
    for split in ["train", "validation", "test"]:
        assert "macro" in run_metrics[split]


def test_train_xgb_models_ray_dry_run_skipped_status(tmp_path: Path):
    root = tmp_path / "assets" / "dataset" / "splits" / "high_quality" / "random_cluster"
    for split_idx in range(1):
        ds_dir = root / f"split_{split_idx}" / "fold_0" / "hf_dataset"
        ds_dir.mkdir(parents=True)
        _make_hf_like_dataset(ds_dir)
    out = tmp_path / "out"
    results = train_ensemble(
        root,
        ensemble_trainer_cls=BaseEnsembleTrainer,
        trainer_cls=XGBoostTrainer,
        model_cls=cast(Any, XGBoostMultiEndpoint),
        model_params={"n_estimators": 5},
        output_root=out,
        seed=42,
        num_cpus=1,
        dry_run=True,
        n_fingerprint_bits=16,
    )
    assert results
    for payload in results.values():
        assert payload.get("run_metrics") is None
        assert payload.get("summary") is None
    for payload in results.values():
        assert payload.get("status") == "skipped"
        assert payload.get("start_time")
        assert payload.get("end_time")
        assert isinstance(payload.get("duration_seconds"), float)


def test_train_xgb_models_ray_timeout_status(tmp_path: Path):
    root = tmp_path / "assets" / "dataset" / "splits" / "high_quality" / "random_cluster"
    ds_dir = root / "split_0" / "fold_0" / "hf_dataset"
    ds_dir.mkdir(parents=True)
    _make_hf_like_dataset(ds_dir)
    slow_ray_trainer = SlowRayTrainer(trainer_cls=SlowTrainer, trainer_kwargs={})
    out = tmp_path / "out"
    results = slow_ray_trainer.fit_ensemble(
        root,
        num_cpus=1,
        max_duration_seconds=0.0,
        n_fingerprint_bits=16,
        output_root=out,
    )
    assert results
    for payload in results.values():
        assert payload.get("status") == "timeout"
        assert payload.get("start_time")
        assert payload.get("end_time")
        assert isinstance(payload.get("duration_seconds"), float)


def test_train_xgb_models_ray_partial_status(tmp_path: Path):
    root = tmp_path / "assets" / "dataset" / "splits" / "high_quality" / "random_cluster"
    ds_dir = root / "split_0" / "fold_0" / "hf_dataset"
    ds_dir.mkdir(parents=True)
    _make_hf_like_dataset(ds_dir)
    partial_trainer = PartialRayTrainer(trainer_cls=PartialTrainer, trainer_kwargs={"model_cls": BaseModel})
    results = partial_trainer.fit_ensemble(
        root, num_cpus=1, n_fingerprint_bits=16, output_root=tmp_path / "out"
    )
    assert results
    for payload in results.values():
        assert payload.get("status") == "partial"
        assert payload.get("start_time")
        assert payload.get("end_time")
        assert isinstance(payload.get("duration_seconds"), float)


def test_missing_fingerprint_columns_errors(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=10, n_bits=16)
    dataset = load_dataset(data_dir, n_fingerprint_bits=16)
    dataset.fingerprint_cols = []
    trainer = XGBoostTrainer(
        model_cls=cast(Any, XGBoostMultiEndpoint),
        model_params={"n_estimators": 10},
        seed=123,
    )
    with pytest.raises(ValueError):
        trainer.fit(dataset, output_dir=tmp_path / "out")


def test_xgb_gpu_fallback_retry(tmp_path: Path, monkeypatch):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=10, n_bits=16)
    dataset = load_dataset(data_dir, n_fingerprint_bits=16)

    class FakeRegressor:
        call_count = 0

        def __init__(self, **params):  # type: ignore[unused-argument]
            self.params = params

        def fit(self, X, y, **kwargs):  # type: ignore[unused-argument]
            FakeRegressor.call_count += 1
            if FakeRegressor.call_count == 1:
                raise RuntimeError("fake GPU failure")

        def predict(self, X):  # type: ignore[unused-argument]
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
    run_metrics, summary = trainer.fit(dataset, early_stopping_rounds=1, output_dir=tmp_path / "out")
    for split in ["train", "validation", "test"]:
        assert split in run_metrics and "macro" in run_metrics[split]


def test_save_artifacts_receives_outputs(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=20, n_bits=16)
    dataset = load_dataset(data_dir, n_fingerprint_bits=16)
    captured = []

    class CaptureTrainer(XGBoostTrainer):
        def save_artifacts(self, model, run_metrics, output_dir, summary, *, dataset, extra_meta=None):
            captured.append(summary)
            super().save_artifacts(
                model, run_metrics, output_dir, summary, dataset=dataset, extra_meta=extra_meta
            )

    trainer = CaptureTrainer(
        model_cls=cast(Any, XGBoostMultiEndpoint), model_params={"n_estimators": 5}, seed=42
    )
    out = tmp_path / "out"
    _run_metrics, summary = trainer.fit(dataset, output_dir=out)
    assert captured
    summary = captured[0]
    assert isinstance(summary, RunSummary)
    assert summary.Y_train.shape[0] == len(dataset.train)
    assert summary.Y_train.shape[1] == len(dataset.endpoints)
    assert summary.pred_train.shape[0] == len(dataset.train)
    assert summary.mask_train.shape == summary.Y_train.shape


def test_build_sample_weights_missing_dataset_column_raises(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=10, n_bits=16)
    ds = load_dataset(data_dir, n_fingerprint_bits=16)
    ds.train = ds.train.drop(columns=["Dataset"])
    trainer = XGBoostTrainer(
        model_cls=cast(Any, XGBoostMultiEndpoint), model_params={"n_estimators": 5}, seed=1
    )
    with pytest.raises(ValueError):
        trainer.build_sample_weights(ds, {"default": 1.0, "dataset_a": 2.0})


def test_build_sample_weights_explicit_default(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=10, n_bits=16)
    ds = load_dataset(data_dir, n_fingerprint_bits=16)
    ds.train.loc[0:4, "Dataset"] = "unknown_label"
    mapping = {"dataset_a": 2.0, "default": 1.0}
    trainer = XGBoostTrainer(
        model_cls=cast(Any, XGBoostMultiEndpoint), model_params={"n_estimators": 10}, seed=123
    )
    sw = trainer.build_sample_weights(ds, mapping)
    assert sw is not None and sw.shape[0] == len(ds.train) and all(sw[:5] == 1.0) and all(sw[5:] == 2.0)


def test_build_sample_weights_unknown_labels(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=10, n_bits=16)
    ds = load_dataset(data_dir, n_fingerprint_bits=16)
    mapping = {"dataset_a": 2.0, "unknown_in_mapping": 3.0, "default": 1.0}
    trainer = XGBoostTrainer(
        model_cls=cast(Any, XGBoostMultiEndpoint), model_params={"n_estimators": 10}, seed=123
    )
    sw = trainer.build_sample_weights(ds, mapping)
    assert sw is not None and sw.shape[0] == len(ds.train) and all(sw == 2.0)


def test_build_sample_weights_no_mapping(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=10, n_bits=16)
    ds = load_dataset(data_dir, n_fingerprint_bits=16)
    trainer = XGBoostTrainer(
        model_cls=cast(Any, XGBoostMultiEndpoint), model_params={"n_estimators": 10}, seed=123
    )
    sw = trainer.build_sample_weights(ds, None)
    assert sw is None
    sw_empty = trainer.build_sample_weights(ds, {})
    assert sw_empty is None


def test_prepare_features_missing_fingerprint_columns_raises(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=10, n_bits=16)
    dataset = load_dataset(data_dir, n_fingerprint_bits=16)
    dataset.fingerprint_cols = ["missing_fp1", "missing_fp2"] + list(dataset.fingerprint_cols)
    trainer = XGBoostTrainer(
        model_cls=cast(Any, XGBoostMultiEndpoint), model_params={"n_estimators": 10}, seed=123
    )
    with pytest.raises(ValueError, match="Missing fingerprint columns"):
        trainer.prepare_features(dataset)


def test_prepare_targets_missing_endpoint_columns_raises(tmp_path: Path):
    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=10, n_bits=16)
    dataset = load_dataset(data_dir, n_fingerprint_bits=16)
    dataset.endpoints = ["missing_ep1", "missing_ep2"] + list(dataset.endpoints)
    trainer = XGBoostTrainer(
        model_cls=cast(Any, XGBoostMultiEndpoint), model_params={"n_estimators": 10}, seed=123
    )
    with pytest.raises(ValueError, match="Missing endpoint columns"):
        trainer.prepare_targets(dataset)
