import json
from pathlib import Path
from typer.testing import CliRunner

from admet.logging import configure_logging
from admet.cli import app
from admet.train.xgb_train import train_xgb_models_ray


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


def _make_config(tmp_path: Path) -> Path:
    cfg = {
        "models": {
            "xgboost": {
                "model_params": {"n_estimators": 5},
                "early_stopping_rounds": 1,
            }
        },
        "training": {"sample_weights": {"enabled": False, "weights": {"default": 1.0}}},
        "data": {"endpoints": ["LogD"]},
    }
    import yaml

    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return p


def test_ray_worker_writes_structured_json_log(tmp_path: Path):
    # Create a dataset and config
    ds_root = tmp_path / "splits" / "high_quality" / "random_cluster" / "split_0" / "fold_0" / "hf_dataset"
    ds_root.mkdir(parents=True)
    # Build a very small dataset using helper used elsewhere
    from .test_xgb_train import _make_hf_like_dataset

    _make_hf_like_dataset(ds_root, n_rows=10, n_bits=16)
    # configure logging to write to file in structured JSON
    logfile = tmp_path / "ray_worker.log"
    configure_logging(level="DEBUG", file=str(logfile), structured=True)

    # Call the trainer - the BaseRayMultiDatasetTrainer.run_all picks up the current logging config
    results = train_xgb_models_ray(
        tmp_path / "splits",
        model_params={"n_estimators": 5},
        output_root=tmp_path / "out",
        seed=42,
        num_cpus=1,
    )
    assert results
    assert logfile.exists()
    # Ensure file contains JSON lines
    with logfile.open("r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    assert lines
    json.loads(lines[0])  # should parse


def test_cli_log_file_option_creates_file(tmp_path: Path):
    # create dataset and config
    ds_root = tmp_path / "hf_dataset"
    ds_root.mkdir(parents=True)
    from .test_xgb_train import _make_hf_like_dataset

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
        "--output-dir",
        str(tmp_path / "out"),
    ]
    _ = runner.invoke(app, cmd)
    # CLI might exit with non-zero if training fails for other reasons, but logging should have been created
    assert logfile.exists()
    with logfile.open("r") as fh:
        lines = [ln.strip() for ln in fh if ln.strip()]
    assert lines
    json.loads(lines[0])
