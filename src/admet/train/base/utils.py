"""Utility helpers for training.

Includes feature/target extraction, mask building, and split metadata parsing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def _extract_features(df, fingerprint_cols: Sequence[str]) -> np.ndarray:
    """Extract fingerprint features from dataframe.

    Args:
        df: Input dataframe
        fingerprint_cols: Column names for fingerprint features

    Returns:
        Float32 numpy array of shape (n_samples, n_features)
    """
    return df.loc[:, fingerprint_cols].to_numpy(dtype=np.float32, copy=False)


def _extract_targets(df, endpoints: Sequence[str]) -> np.ndarray:
    """Extract target endpoints from dataframe.

    Args:
        df: Input dataframe
        endpoints: Column names for target endpoints

    Returns:
        Float32 numpy array of shape (n_samples, n_endpoints)
    """
    return df[endpoints].to_numpy(dtype=np.float32, copy=False)


def _target_mask(y: np.ndarray) -> np.ndarray:
    """Create mask for non-NaN target values.

    Args:
        y: Target array

    Returns:
        Boolean mask where True indicates non-NaN values
    """
    return ~np.isnan(y)


def infer_split_metadata(hf_path: Path, root: Path) -> Dict[str, object]:
    try:
        rel = hf_path.relative_to(root)
        rel_parts: List[str] = [p for p in rel.parts if p]
        meta: Dict[str, object] = {
            "relative_path": str(rel),
            "absolute_path": str(hf_path.resolve()),
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to infer relative path for %s: %s", hf_path, e)
        rel = None
        rel_parts = []
        meta = {"relative_path": str(hf_path)}

    try:
        full_path = str(hf_path.resolve())
    except Exception as e:  # noqa: BLE001
        logger.warning("Failed to resolve full path for %s: %s", hf_path, e)
        full_path = str(hf_path)
    full_parts: List[str] = [p for p in Path(full_path).parts if p and p != "/"]
    meta["full_path"] = full_path

    for part in rel_parts:
        if part.startswith("split_"):
            meta["split"] = part.replace("split_", "")
        elif part.startswith("fold_"):
            meta["fold"] = part.replace("fold_", "")

    if len(full_parts) > 5:
        cluster_method = full_parts[-4]
        quality = full_parts[-5]
        meta["quality"] = quality
        meta["cluster"] = f"{quality}/{cluster_method}"

    logger.info("Inferred metadata for %s: %s", rel, meta)
    return meta


@dataclass
class SplitMetadata:
    """Metadata for a dataset split including paths and clustering info.

    Attributes:
        relative_path: Path relative to root directory
        absolute_path: Absolute filesystem path
        full_path: Full resolved path
        quality: Data quality level (e.g., 'high_quality')
        cluster: Clustering method and quality (e.g., 'high_quality/kmeans_cluster')
        split: Split identifier (e.g., 'split_0')
        fold: Fold identifier (e.g., 'fold_0')
    """

    relative_path: str
    absolute_path: str
    full_path: str
    quality: Optional[str] = None
    cluster: Optional[str] = None
    split: Optional[str] = None
    fold: Optional[str] = None


def metadata_from_dict(d: Dict[str, object]) -> SplitMetadata:
    """Convert dictionary to SplitMetadata dataclass.

    Args:
        d: Dictionary with metadata keys

    Returns:
        SplitMetadata instance
    """

    def _opt_str(k: str) -> Optional[str]:
        v = d.get(k)
        return str(v) if v is not None else None

    return SplitMetadata(
        relative_path=str(d.get("relative_path", "")),
        absolute_path=str(d.get("absolute_path", "")),
        full_path=str(d.get("full_path", "")),
        quality=_opt_str("quality"),
        cluster=_opt_str("cluster"),
        split=_opt_str("split"),
        fold=_opt_str("fold"),
    )


__all__ = [
    "_extract_features",
    "_extract_targets",
    "_target_mask",
    "infer_split_metadata",
    "SplitMetadata",
    "metadata_from_dict",
]
