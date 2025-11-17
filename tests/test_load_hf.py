# smoke/load test for HF datasets
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

try:
    from datasets import Dataset, DatasetDict
except Exception:
    pytest.skip("datasets not installed", allow_module_level=True)

from admet.data.load import load_dataset


def make_df(n, start=0):
    df = pd.DataFrame(
        {
            "Molecule Name": [f"mol_{i+start}" for i in range(n)],
            "SMILES": ["C" for _ in range(n)],
            "Dataset": ["smoke" for _ in range(n)],
            "LogD": np.random.randn(n),
            "KSOL": np.random.randn(n) * 1.5,
        }
    )
    for i in range(16):
        df[f"Morgan_FP_{i}"] = np.random.randint(0, 2, size=n)
    return df


def test_load_hf_roundtrip(tmp_path: Path):
    train_df = make_df(8, start=0)
    val_df = make_df(2, start=8)
    test_df = make_df(3, start=10)

    train_hf = Dataset.from_pandas(train_df, preserve_index=False)
    val_hf = Dataset.from_pandas(val_df, preserve_index=False)
    test_hf = Dataset.from_pandas(test_df, preserve_index=False)

    dset = DatasetDict({"train": train_hf, "validation": val_hf, "test": test_hf})
    out = tmp_path / "hf_dataset"
    dset.save_to_disk(str(out))

    loaded = load_dataset(out, endpoints=["LogD", "KSOL"], n_fingerprint_bits=16)

    assert loaded.train.shape[0] == 8
    assert loaded.val.shape[0] == 2
    assert loaded.test.shape[0] == 3

    # check fingerprint columns exist
    for i in range(16):
        assert f"Morgan_FP_{i}" in loaded.train.columns

    # spot-check endpoint presence and dtype
    assert "LogD" in loaded.train.columns
    assert "KSOL" in loaded.train.columns
    assert loaded.train["LogD"].dtype.kind in "fi"
