"""Dataset loading and validation utilities (initial XGBoost-focused implementation).

This is a minimal implementation to support early XGBoost model training. It
will be extended later to fully match the planning document for multiple
splits/folds and HF datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, cast
import logging

from datasets import load_from_disk
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

REQUIRED_BASE_COLUMNS: Sequence[str] = [
    "Molecule Name",
    "SMILES",
    "Dataset",
]

ENDPOINT_COLUMNS: Sequence[str] = [
    "LogD",
    "KSOL",
    "HLM CLint",
    "MLM CLint",
    "Caco-2 Permeability Efflux",
    "Caco-2 Permeability Papp A>B",
    "MPPB",
    "MBPB",
    "MGMB",
]


def expected_fingerprint_columns(n_bits: int = 2046) -> List[str]:  # small for tests
    """Return list of fingerprint column names.

    NOTE: Full implementation will use 2048 bits; a reduced size is used for
    unit tests to reduce memory/time.
    """
    return [f"Morgan_FP_{i}" for i in range(n_bits)]


@dataclass
class LoadedDataset:
    """Container for loaded dataset splits."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    endpoints: Sequence[str]
    fingerprint_cols: Sequence[str]


def validate_dataset_schema(
    df: pd.DataFrame,
    fingerprint_cols: Sequence[str],
    endpoints: Sequence[str],
) -> None:
    """Validate required columns and types. Raises ValueError on failure."""
    missing = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required base columns: {missing}. Found: {df.columns.tolist()}")
    fp_missing = [c for c in fingerprint_cols if c not in df.columns]
    if fp_missing:
        raise ValueError(f"Missing fingerprint columns: {fp_missing[:10]} .... Found: {df.columns.tolist()}")
    ep_missing = [c for c in endpoints if c not in df.columns]
    if ep_missing:
        raise ValueError(f"Missing endpoint columns: {ep_missing}. Found: {df.columns.tolist()}")

    # Basic dtype checks (float for endpoints)
    bad_types = []
    for c in endpoints:
        series = df[c].dropna()
        if series.empty:
            continue
        if not np.issubdtype(series.dtype, np.number):  # type: ignore[arg-type]
            bad_types.append(c)
    if bad_types:
        raise ValueError(f"Endpoint columns not float-like: {bad_types}")


def load_dataset_from_hf(
    root: Path,
    *,
    endpoints: Optional[Sequence[str]] = None,
    n_fingerprint_bits: int = 2048,
) -> LoadedDataset:
    """Load a Hugging Face DatasetDict saved with save_to_disk.

    Expects a directory containing a serialized DatasetDict with keys
    'train', 'validation', and 'test'. Each split is converted to a
    pandas.DataFrame and validated.
    """
    dset = load_from_disk(str(root))

    # Check for required splits
    required_splits = ["train", "validation", "test"]
    missing = [s for s in required_splits if s not in dset]
    if missing:
        raise ValueError(f"HF dataset missing required splits: {missing}")

    train_df = cast(pd.DataFrame, dset["train"].to_pandas())  # type: ignore[attr-defined]
    val_df = cast(pd.DataFrame, dset["validation"].to_pandas())  # type: ignore[attr-defined]
    test_df = cast(pd.DataFrame, dset["test"].to_pandas())  # type: ignore[attr-defined]

    endpoints = endpoints or ENDPOINT_COLUMNS
    fp_cols = expected_fingerprint_columns(n_fingerprint_bits)

    for name, df in [("train", train_df), ("validation", val_df), ("test", test_df)]:
        try:
            validate_dataset_schema(df, fp_cols, endpoints)
        except Exception as e:  # pragma: no cover - surface errors to user
            raise ValueError(f"Schema validation failed for split '{name}': {e}") from e

    return LoadedDataset(
        train=train_df,
        val=val_df,
        test=test_df,
        endpoints=endpoints,
        fingerprint_cols=fp_cols,
    )


def load_dataset(
    root: Path,
    *,
    endpoints: Optional[Sequence[str]] = None,
    n_fingerprint_bits: int = 2048,
) -> LoadedDataset:
    """Load train/validation/test splits from a saved Hugging Face DatasetDict.

    Expected directory containing DatasetDict with splits: 'train', 'validation', 'test'.
    """
    return load_dataset_from_hf(root, endpoints=endpoints, n_fingerprint_bits=n_fingerprint_bits)


def load_blinded_dataset(path: Path) -> pd.DataFrame:
    """Load blinded test CSV containing Molecule Name + SMILES only."""
    df = pd.read_csv(path)
    for col in ["Molecule Name", "SMILES"]:
        if col not in df.columns:
            raise ValueError(f"Blinded dataset missing column '{col}'")
    return df


def iter_splits(
    split_ids: Sequence[int],
    fold_ids: Optional[Sequence[int]] = None,
) -> Iterable[Tuple[int, Optional[int]]]:
    """Yield (split_id, fold_id) pairs. Initial simple implementation (no nesting)."""
    if not split_ids:
        yield from []
    if not fold_ids:
        for s in split_ids:
            yield s, None
    else:
        for s in split_ids:
            for f in fold_ids:
                yield s, f


__all__ = [
    "LoadedDataset",
    "load_dataset",
    "load_blinded_dataset",
    "validate_dataset_schema",
    "iter_splits",
    "ENDPOINT_COLUMNS",
    "expected_fingerprint_columns",
]
