"""Unit tests for train.base.utils helpers.

This module ensures helpers used by training code behave as expected for
feature/target extraction, mask creation, metadata inference and round-trips.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from admet.train.base.utils import (
    _extract_features,
    _extract_targets,
    _target_mask,
    infer_split_metadata,
    metadata_from_dict,
)


@pytest.mark.unit
def test_extract_features_and_targets() -> None:
    df = pd.DataFrame({"fp1": [0.1, 0.2], "fp2": [1.0, 1.1], "A": [0.5, 0.6]})
    X = _extract_features(df.loc[:, ["fp1", "fp2"]], ["fp1", "fp2"])  # type: ignore[arg-type]
    assert X.shape == (2, 2)
    assert np.allclose(X, [[0.1, 1.0], [0.2, 1.1]])
    Y = _extract_targets(df, ["A"])  # type: ignore[arg-type]
    assert Y.shape == (2, 1)
    assert np.allclose(Y, [[0.5], [0.6]])


@pytest.mark.unit
def test_target_mask_nan_handling() -> None:
    Y = np.array([[0.5, np.nan], [np.nan, 1.2]])
    mask = _target_mask(Y)
    assert mask.shape == Y.shape
    assert mask.dtype == bool
    assert mask.tolist() == [[True, False], [False, True]]


@pytest.mark.unit
def test_infer_split_metadata_and_roundtrip(tmp_path: Path) -> None:
    # Create nested path: <root>/quality/cluster/method/split_0/fold_0/hf_dataset
    root = tmp_path / "root"
    hf_dataset = root / "high_quality" / "kmeans" / "method" / "split_0" / "fold_0" / "hf_dataset"
    hf_dataset.mkdir(parents=True)
    meta = infer_split_metadata(hf_dataset, root)
    assert "relative_path" in meta
    assert meta.get("split") == "0"
    assert meta.get("fold") == "0"
    assert "cluster" in meta
    # Roundtrip
    sd = metadata_from_dict(meta)
    assert sd.relative_path == meta["relative_path"]
