from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from admet.data.load import ENDPOINT_COLUMNS, expected_fingerprint_columns, load_dataset
from admet.train.xgb_train import train_xgb_models, train_xgb_models_ray, XGBoostTrainer
from admet.train.base_trainer import infer_split_metadata
from admet.model.xgb_wrapper import XGBoostMultiEndpoint


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


__all__ = [
    "test_infer_split_metadata_parses_cluster_split_fold",
    "test_train_xgb_models_runs_end_to_end",
    "test_train_xgb_models_ray_multiple_datasets_and_summary",
]
