from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from datasets import Dataset, DatasetDict

from admet.data.load import load_dataset, expected_fingerprint_columns, ENDPOINT_COLUMNS


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


@pytest.fixture
def synth_dataset(tmp_path: Path, n_rows: int = 40, n_bits: int = 16):
    fp_cols = expected_fingerprint_columns(n_bits)
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
        pd.DataFrame(data).to_csv(tmp_path / f"{split}.csv", index=False)
    ds = load_dataset(tmp_path, n_fingerprint_bits=n_bits)
    return ds


def test_xgboost_multiendpoint_save_load_roundtrip(tmp_path: Path):
    from admet.model.xgb_wrapper import XGBoostMultiEndpoint

    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=20, n_bits=16)
    ds = load_dataset(data_dir, n_fingerprint_bits=16)
    X_train = ds.train.loc[:, ds.fingerprint_cols].to_numpy(dtype=float)
    Y_train = ds.train.loc[:, ds.endpoints].to_numpy(dtype=float)
    mask_train = (~np.isnan(Y_train)).astype(bool)
    model = XGBoostMultiEndpoint(endpoints=ds.endpoints, model_params={"n_estimators": 1}, random_state=42)
    model.fit(X_train, Y_train, Y_mask=mask_train, early_stopping_rounds=None)
    preds_before = model.predict(X_train)
    out_dir = tmp_path / "xgb_model"
    model.save(str(out_dir))
    loaded = XGBoostMultiEndpoint.load(str(out_dir))
    preds_after = loaded.predict(X_train)
    assert preds_before.shape == preds_after.shape
    mask = ~np.isnan(preds_before)
    assert np.allclose(preds_before[mask], preds_after[mask], equal_nan=True)


def test_predict_dimension_checks(tmp_path: Path):
    from admet.model.xgb_wrapper import XGBoostMultiEndpoint

    data_dir = tmp_path / "hf_dataset"
    data_dir.mkdir(parents=True)
    _make_hf_like_dataset(data_dir, n_rows=10, n_bits=16)
    ds = load_dataset(data_dir, n_fingerprint_bits=16)
    X_train = ds.train.loc[:, ds.fingerprint_cols].to_numpy(dtype=float)
    Y_train = ds.train.loc[:, ds.endpoints].to_numpy(dtype=float)
    mask_train = (~np.isnan(Y_train)).astype(bool)
    model = XGBoostMultiEndpoint(endpoints=ds.endpoints, model_params={"n_estimators": 1}, random_state=42)
    model.fit(X_train, Y_train, Y_mask=mask_train, early_stopping_rounds=None)
    import numpy as _np

    with pytest.raises(ValueError):
        model.predict(_np.zeros((len(X_train), X_train.shape[1] - 1)))


def test_predict_errors_when_not_fit():
    from admet.model.xgb_wrapper import XGBoostMultiEndpoint

    model = XGBoostMultiEndpoint(endpoints=["LogD"], model_params={"n_estimators": 1}, random_state=42)
    import numpy as _np

    with pytest.raises(ValueError):
        model.predict(_np.zeros((2, 16)))
