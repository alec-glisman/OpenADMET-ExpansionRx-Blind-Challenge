"""Integration tests for logging behaviour with Ray workers and CLI.

These tests ensure that structured JSON logs are produced by Ray worker
processes and that CLI `--log-file` option writes a file.
"""

from __future__ import annotations

import json
from pathlib import Path
from typer.testing import CliRunner
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict
import pytest

from admet.logging import configure_logging
from admet.cli import app
from admet.train.base import train_ensemble, BaseEnsembleTrainer
from admet.train.xgb_train import XGBoostTrainer
from admet.model.xgb_wrapper import XGBoostMultiEndpoint
from admet.data.load import expected_fingerprint_columns, ENDPOINT_COLUMNS


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


def _make_config(tmp_path: Path) -> Path:
    cfg = {
        "models": {"xgboost": {"model_params": {"n_estimators": 5}, "early_stopping_rounds": 1}},
        "training": {
            "output_dir": str(tmp_path / "out"),
            "seed": 42,
            "n_fingerprint_bits": 16,
            "sample_weights": {"enabled": False, "weights": {"default": 1.0}},
        },
        "data": {"endpoints": ["LogD"]},
    }
    import yaml

    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_ray_worker_writes_structured_json_log(tmp_path: Path) -> None:
    ds_root = tmp_path / "splits" / "high_quality" / "random_cluster" / "split_0" / "fold_0" / "hf_dataset"
    ds_root.mkdir(parents=True)
    _make_hf_like_dataset(ds_root, n_rows=10, n_bits=16)
    logfile = tmp_path / "ray_worker.log"
    configure_logging(level="DEBUG", file=str(logfile), structured=True)
    results = train_ensemble(
        tmp_path / "splits",
        ensemble_trainer_cls=BaseEnsembleTrainer,
        trainer_cls=XGBoostTrainer,
        model_cls=XGBoostMultiEndpoint,
        model_params={"n_estimators": 5},
        output_root=tmp_path / "out",
        seed=42,
        num_cpus=1,
        n_fingerprint_bits=16,
    )
    if not results:
        pytest.fail("Expected results from train_ensemble but got empty/None")
    if not logfile.exists():
        pytest.fail("Expected log file to exist after Ray worker run")
    with logfile.open("r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    if not lines:
        pytest.fail("Expected log file to contain at least one non-empty line")
    json.loads(lines[0])


def test_cli_log_file_option_creates_file(tmp_path: Path) -> None:
    ds_root = tmp_path / "hf_dataset"
    ds_root.mkdir(parents=True)
    _make_hf_like_dataset(ds_root, n_rows=10, n_bits=16)
    cfg_path = _make_config(tmp_path)
    runner = CliRunner()
    logfile = tmp_path / "cli.log"
    cmd = [
        "--log-file",
        str(logfile),
        "--log-json",
        "train",
        "xgb",
        str(ds_root),
        "--config",
        str(cfg_path),
    ]
    _ = runner.invoke(app, cmd)
    if not logfile.exists():
        pytest.fail("Expected CLI-created logfile to exist")
