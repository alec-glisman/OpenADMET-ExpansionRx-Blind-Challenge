import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from admet.data.load import (
    validate_dataset_schema,
    expected_fingerprint_columns,
    ENDPOINT_COLUMNS,
)


def _make_split(path: Path, n_rows: int = 20, n_bits: int = 16):
    fp_cols = expected_fingerprint_columns(n_bits)
    data = {
        "Molecule Name": [f"MOL_{i}" for i in range(n_rows)],
        "SMILES": ["C" for _ in range(n_rows)],
        "Dataset": ["dataset_a" for _ in range(n_rows)],
    }
    for ep in ENDPOINT_COLUMNS:
        vals = np.random.randn(n_rows).astype(float)
        vals[0] = np.nan
        data[ep] = list(vals)
    for i, col in enumerate(fp_cols):
        data[col] = list(np.random.randn(n_rows).astype(float))
    df = pd.DataFrame(data)
    df.to_csv(path, index=False)


def test_validate_dataset_schema_missing_column(tmp_path: Path):
    _make_split(tmp_path / "train.csv")
    df = pd.read_csv(tmp_path / "train.csv")
    fp_cols = expected_fingerprint_columns(16)
    remove_col = fp_cols[0]
    df = df.drop(columns=[remove_col])
    with pytest.raises(ValueError) as exc_info:
        validate_dataset_schema(df, fp_cols, ENDPOINT_COLUMNS)
    assert "Missing fingerprint" in str(exc_info.value)
