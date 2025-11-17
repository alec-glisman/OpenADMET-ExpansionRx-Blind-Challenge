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

import pandas as pd
import numpy as np

# Note: Hugging Face `datasets` is imported lazily inside the HF loader so
# environments without the package don't fail at module import time.

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


def expected_fingerprint_columns(n_bits: int = 16) -> List[str]:  # small for tests
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

    def all_splits(self) -> Iterable[Tuple[str, pd.DataFrame]]:
        yield "train", self.train
        yield "val", self.val
        yield "test", self.test


def validate_dataset_schema(
    df: pd.DataFrame,
    fingerprint_cols: Sequence[str],
    endpoints: Sequence[str],
) -> None:
    """Validate required columns and types. Raises ValueError on failure."""
    missing = [c for c in REQUIRED_BASE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required base columns: {missing}")
    fp_missing = [c for c in fingerprint_cols if c not in df.columns]
    if fp_missing:
        raise ValueError(f"Missing fingerprint columns: {fp_missing[:10]} ...")
    ep_missing = [c for c in endpoints if c not in df.columns]
    if ep_missing:
        raise ValueError(f"Missing endpoint columns: {ep_missing}")
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


def _load_split_csv(root: Path, name: str) -> pd.DataFrame:
    path = root / f"{name}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected split file: {path}")
    df = pd.read_csv(path)
    return df


def load_dataset_from_hf(
    root: Path,
    *,
    endpoints: Optional[Sequence[str]] = None,
    n_fingerprint_bits: int = 16,
) -> LoadedDataset:
    """Load a Hugging Face `DatasetDict` saved with `save_to_disk`.

    This expects a directory containing a serialized DatasetDict with keys
    'train', 'validation' (or 'val'), and 'test'. Each split is converted to a
    pandas.DataFrame and validated with the existing `validate_dataset_schema`.
    """
    try:
        from datasets import load_from_disk  # type: ignore
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise ImportError(
            "The 'datasets' package is required to load HF datasets. "
            "Install with `pip install datasets` and try again.`"
        ) from e

    # Accept either the directory itself or a subdirectory `hf_dataset`.
    tried = []
    dset = None
    candidates = [root, root / "hf_dataset"]
    for p in candidates:
        tried.append(str(p))
        if not p.exists():
            continue
        try:
            loaded = load_from_disk(str(p))
        except (OSError, ValueError):
            # Try next candidate on IO/parsing errors
            continue
        # Basic structural check for a DatasetDict-like object with splits
        if not hasattr(loaded, "keys"):
            continue
        if not ("train" in loaded and "test" in loaded and ("validation" in loaded or "val" in loaded)):
            continue
        dset = loaded
        break

    if dset is None:
        raise FileNotFoundError(
            f"No HF dataset found at any candidate paths: {tried}. "
            "Ensure you saved with `DatasetDict(...).save_to_disk(...)`."
        )

    # Determine validation split key name
    val_key = "validation" if "validation" in dset else ("val" if "val" in dset else None)
    if val_key is None or "train" not in dset or "test" not in dset:
        raise ValueError("HF dataset missing required splits 'train','validation'/'val' or 'test'")

    train_df = cast(pd.DataFrame, dset["train"].to_pandas())  # type: ignore[attr-defined]
    val_df = cast(pd.DataFrame, dset[val_key].to_pandas())  # type: ignore[attr-defined]
    test_df = cast(pd.DataFrame, dset["test"].to_pandas())  # type: ignore[attr-defined]

    endpoints = endpoints or ENDPOINT_COLUMNS
    fp_cols = expected_fingerprint_columns(n_fingerprint_bits)

    for name, df in [("train", train_df), ("val", val_df), ("test", test_df)]:
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
    n_fingerprint_bits: int = 16,
) -> LoadedDataset:
    """Load train/val/test CSVs from a directory and validate schema.

    Expected files: train.csv, val.csv, test.csv.
    This function is a bootstrap utility; later versions will integrate the
    full HF dataset + split hierarchy logic.
    """
    # Prefer Hugging Face DatasetDict saved with `save_to_disk`.
    root = Path(root)
    try:
        return load_dataset_from_hf(root, endpoints=endpoints, n_fingerprint_bits=n_fingerprint_bits)
    except (FileNotFoundError, ImportError, ValueError) as hf_err:
        # If HF loader isn't available or the path doesn't contain HF data,
        # fall back to CSV loader (for backward compatibility) and log the
        # reason for inspection.
        logger.debug("HF dataset load failed (%s); falling back to CSV loader", hf_err)

    endpoints = endpoints or ENDPOINT_COLUMNS
    fp_cols = expected_fingerprint_columns(n_fingerprint_bits)
    train = _load_split_csv(root, "train")
    val = _load_split_csv(root, "val")
    test = _load_split_csv(root, "test")
    for name, df in [("train", train), ("val", val), ("test", test)]:
        try:
            validate_dataset_schema(df, fp_cols, endpoints)
        except Exception as e:  # pragma: no cover - re-raise with context
            raise ValueError(f"Schema validation failed for split '{name}': {e}") from e
    return LoadedDataset(train=train, val=val, test=test, endpoints=endpoints, fingerprint_cols=fp_cols)


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
