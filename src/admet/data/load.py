"""admet.data.load
===================

Dataset loading, schema validation, and convenience helpers for Hugging Face
``DatasetDict`` objects saved with ``save_to_disk``.

The current implementation targets the initial XGBoost training pipeline and
expects three splits: ``train``, ``validation``, ``test``. Each split is
converted to a ``pandas.DataFrame`` and validated for required columns:

Base Columns
------------
``['Molecule Name', 'SMILES', 'Dataset']``

Fingerprint Columns
-------------------
Generated Morgan fingerprint bit columns named ``Morgan_FP_{i}`` for
``i in [0, n_bits)``. A reduced bit length may be used in tests to decrease
resource usage.

Endpoint Columns
----------------
Default endpoints (ADMET properties) included by ``ENDPOINT_COLUMNS``. Custom
endpoint subsets may be provided when loading.

Dataclass Container
-------------------
The :class:`LoadedDataset` dataclass provides split DataFrames plus endpoint
and fingerprint column lists to avoid repeatedly passing separate values.

Examples
--------
Load a dataset and inspect shapes::

    from pathlib import Path
    from admet.data.load import load_dataset

    ds = load_dataset(Path("/path/to/hf_dataset"), n_fingerprint_bits=256)
    print(ds.train.shape, ds.val.shape, ds.test.shape)
    print(ds.endpoints)

Schema Validation Failures
--------------------------
Validation functions raise ``ValueError`` with diagnostic information.
Endpoint columns may contain NaNs (missing labels); numeric dtype is required
for non-null values.
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
    """Return ordered Morgan fingerprint column names.

    Parameters
    ----------
    n_bits : int, optional
        Number of fingerprint bits (columns). Tests may override with a
        smaller value; production defaults to ``2048``.

    Returns
    -------
    list of str
        Column names in the form ``Morgan_FP_{i}``.
    """
    return [f"Morgan_FP_{i}" for i in range(n_bits)]


@dataclass
class LoadedDataset:
    """Container for loaded dataset splits and metadata.

    Attributes
    ----------
    train : pandas.DataFrame
        Training split with required base, fingerprint, and endpoint columns.
    val : pandas.DataFrame
        Validation split.
    test : pandas.DataFrame
        Test split.
    endpoints : Sequence[str]
        Endpoint (target) column names included across all splits.
    fingerprint_cols : Sequence[str]
        Ordered list of fingerprint bit column names.
    """

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
    """Validate dataset columns and basic endpoint dtypes.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame to validate.
    fingerprint_cols : Sequence[str]
        Expected fingerprint bit columns.
    endpoints : Sequence[str]
        Expected endpoint (target) columns.

    Raises
    ------
    ValueError
        If base, fingerprint, or endpoint columns are missing, or endpoints
        contain non-numeric dtypes (excluding NaNs).
    """
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
    """Load and validate a Hugging Face ``DatasetDict`` from disk.

    Parameters
    ----------
    root : pathlib.Path
        Directory containing a serialized ``DatasetDict`` (``save_to_disk``).
    endpoints : Sequence[str], optional
        Override endpoint columns (defaults to ``ENDPOINT_COLUMNS``).
    n_fingerprint_bits : int, optional
        Number of fingerprint bit columns; default ``2048``.

    Returns
    -------
    LoadedDataset
        Dataclass with validated splits and column metadata.

    Raises
    ------
    ValueError
        If required splits are missing or schema validation fails.
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
    """Public wrapper to load a Hugging Face dataset splits directory.

    Parameters
    ----------
    root : pathlib.Path
        Directory containing a saved ``DatasetDict``.
    endpoints : Sequence[str], optional
        Subset of endpoint columns to retain.
    n_fingerprint_bits : int, optional
        Fingerprint bit length (affects expected columns).

    Returns
    -------
    LoadedDataset
        Loaded splits and metadata.
    """
    return load_dataset_from_hf(root, endpoints=endpoints, n_fingerprint_bits=n_fingerprint_bits)


def load_blinded_dataset(path: Path) -> pd.DataFrame:
    """Load blinded test CSV with identifier and structure only.

    Parameters
    ----------
    path : pathlib.Path
        CSV path containing required columns ``['Molecule Name', 'SMILES']``.

    Returns
    -------
    pandas.DataFrame
        DataFrame restricted to provided columns (additional cols retained).

    Raises
    ------
    ValueError
        If required columns are missing.
    """
    df = pd.read_csv(path)
    for col in ["Molecule Name", "SMILES"]:
        if col not in df.columns:
            raise ValueError(f"Blinded dataset missing column '{col}'")
    return df


def iter_splits(
    split_ids: Sequence[int],
    fold_ids: Optional[Sequence[int]] = None,
) -> Iterable[Tuple[int, Optional[int]]]:
    """Iterate over (split_id, fold_id) pairs.

    Parameters
    ----------
    split_ids : Sequence[int]
        Split identifiers (e.g., ``[0, 1, 2]``).
    fold_ids : Sequence[int], optional
        Optional fold identifiers; if omitted yields ``(split_id, None)``.

    Yields
    ------
    tuple[int, Optional[int]]
        Pair containing a split id and optional fold id.
    """
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
