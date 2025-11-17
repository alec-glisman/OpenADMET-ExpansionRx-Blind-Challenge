from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from admet.data.load import load_dataset, expected_fingerprint_columns, ENDPOINT_COLUMNS


@pytest.fixture
def synth_dataset(tmp_path: Path, n_rows: int = 40, n_bits: int = 16):
    """Create a small synthetic dataset saved as CSV splits and return the
    LoadedDataset via `load_dataset` for tests to consume.
    """
    fp_cols = expected_fingerprint_columns(n_bits)
    for split in ["train", "val", "test"]:
        data = {
            "Molecule Name": [f"MOL_{i}" for i in range(n_rows)],
            "SMILES": ["C" for _ in range(n_rows)],
            "Dataset": ["dataset_a" for _ in range(n_rows)],
        }
        for ep in ENDPOINT_COLUMNS:
            vals = np.random.randn(n_rows).astype(float)
            if split == "train":
                vals[0] = np.nan  # missing first row
            data[ep] = list(vals)
        for col in fp_cols:
            data[col] = list(np.random.randn(n_rows).astype(float))
        pd.DataFrame(data).to_csv(tmp_path / f"{split}.csv", index=False)

    ds = load_dataset(tmp_path, n_fingerprint_bits=n_bits)
    return ds


def test_xgb_fit_predict_save_load(synth_dataset: object, tmp_path: Path):
    try:
        from admet.model.xgb_wrapper import XGBoostMultiEndpoint
    except Exception:
        pytest.skip("xgboost not installed; skipping XGB wrapper tests")

    ds = synth_dataset
    X = ds.train[ds.fingerprint_cols].to_numpy()
    Y = ds.train[ds.endpoints].to_numpy()
    mask = (~np.isnan(Y)).astype(int)
    model = XGBoostMultiEndpoint(endpoints=ds.endpoints)
    model.fit(X, Y, Y_mask=mask, early_stopping_rounds=5)
    preds = model.predict(X)
    assert preds.shape == Y.shape
    # Save and load
    save_dir = tmp_path / "model"
    model.save(str(save_dir))
    loaded = XGBoostMultiEndpoint.load(str(save_dir))
    preds_loaded = loaded.predict(X)
    assert np.allclose(preds, preds_loaded, atol=1e-6, equal_nan=True)
