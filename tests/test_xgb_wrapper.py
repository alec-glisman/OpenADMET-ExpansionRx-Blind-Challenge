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
    for split in ["train", "validation", "test"]:
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
