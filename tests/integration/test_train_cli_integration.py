"""Integration tests for CLI training with MLflow and Ray.

These tests create temporary HF-like datasets and run the CLI to assert that
training and MLflow tracking produce expected artifacts. Tests are intentionally
small and use minimal estimators to keep runtime manageable in CI.
"""

from __future__ import annotations

from pathlib import Path

# No additional typing imports necessary for this test module
import numpy as np
import pandas as pd
import pytest
import yaml
from datasets import Dataset, DatasetDict
from typer.testing import CliRunner

from admet.cli import app
from admet.data.load import ENDPOINT_COLUMNS, expected_fingerprint_columns


def _make_hf_like_dataset(root: Path, n_rows: int = 24, n_bits: int = 16) -> Path:
    fp_cols = expected_fingerprint_columns(n_bits)
    splits = {}
    for split in ["train", "validation", "test"]:
        data = {
            "Molecule Name": [f"MOL_{i}" for i in range(n_rows)],
            "SMILES": ["C" for _ in range(n_rows)],
            "Dataset": ["dataset_a" for _ in range(n_rows)],
        }
        for ep in ENDPOINT_COLUMNS:
            vals = np.random.randn(n_rows).astype(np.float32)
            if split == "train":
                vals[0] = np.nan
            data[ep] = list(vals)
        for col in fp_cols:
            data[col] = list(np.random.randn(n_rows).astype(np.float32))
        splits[split] = Dataset.from_pandas(pd.DataFrame(data), preserve_index=False)
    dset = DatasetDict(splits)
    dset.save_to_disk(str(root))
    return root


def _write_config(
    cfg_path: Path,
    *,
    data_root: Path,
    output_dir: Path,
    tracking_dir: Path,
    multi: bool,
) -> None:
    cfg = {
        "models": {
            "xgboost": {
                "objective": "mae",
                "early_stopping_rounds": 3,
                "model_params": {"n_estimators": 5, "max_depth": 3, "tree_method": "hist", "n_jobs": 1},
            }
        },
        "training": {
            "output_dir": str(output_dir),
            "experiment_name": "xgb_integration",
            "tracking_uri": f"file://{tracking_dir}",
            "seed": 123,
            "n_fingerprint_bits": 16,
        },
        "ray": {"multi": multi, "num_cpus": 1, "address": "local"},
        "data": {"root": str(data_root), "endpoints": ENDPOINT_COLUMNS[:2]},
    }
    cfg_path.write_text(yaml.safe_dump(cfg))


@pytest.fixture(autouse=True)
def limit_threads(monkeypatch):
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"]:
        monkeypatch.setenv(var, "1")
    monkeypatch.setenv("MLFLOW_ENABLE_SYSTEM_METRICS", "false")
    yield


def test_cli_single_training_logs_to_mlflow(tmp_path: Path) -> None:
    data_root = tmp_path / "hf_dataset"
    data_root.mkdir(parents=True)
    _make_hf_like_dataset(data_root)
    cfg_path = tmp_path / "single.yaml"
    output_dir = tmp_path / "artifacts_single"
    tracking_dir = tmp_path / "mlruns_single"
    _write_config(
        cfg_path, data_root=data_root, output_dir=output_dir, tracking_dir=tracking_dir, multi=False
    )

    runner = CliRunner()
    result = runner.invoke(app, ["train", "xgb", "--config", str(cfg_path)])
    if result.exit_code != 0:
        pytest.fail(f"CLI single training failed: {result.output}")

    assert tracking_dir.exists()
    assert output_dir.exists()
    # Expect at least one run recorded under the file-based store
    mlruns = list(tracking_dir.glob("*/**/meta.yaml"))
    assert mlruns, "No MLflow runs recorded for single training"


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_cli_ensemble_training_logs_parent_and_child(tmp_path: Path) -> None:
    root = tmp_path / "ensemble_root"
    ds_dir = root / "split_0" / "fold_0" / "hf_dataset"
    ds_dir.mkdir(parents=True)
    _make_hf_like_dataset(ds_dir)

    cfg_path = tmp_path / "ensemble.yaml"
    output_dir = tmp_path / "artifacts_ensemble"
    tracking_dir = tmp_path / "mlruns_ensemble"
    _write_config(cfg_path, data_root=root, output_dir=output_dir, tracking_dir=tracking_dir, multi=True)

    runner = CliRunner()
    result = runner.invoke(app, ["train", "xgb", "--config", str(cfg_path)])
    if result.exit_code != 0:
        pytest.fail(f"CLI ensemble training failed: {result.output}")

    # MLflow tracking data exists
    mlruns = list(tracking_dir.glob("*/**/meta.yaml"))
    assert mlruns, "No MLflow runs recorded for ensemble training"

    # Artifacts from ensemble summary should be present
    assert (output_dir / "metrics_summary.csv").exists()
    assert (output_dir / "metrics_summary.json").exists()
