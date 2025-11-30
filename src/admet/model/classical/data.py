from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold


QUALITY_MAP = {
    "high": 1.0,
    "medium": 0.5,
    "low": 0.2,
}


def load_data(path: str) -> pd.DataFrame:
    """Load CSV into a DataFrame."""
    return pd.read_csv(path)


def augment_quality(
    df: pd.DataFrame,
    quality_col: str = "quality",
    quality_weights: Dict[str, float] = None,
) -> pd.DataFrame:
    """Attach quality_bucket and sample_weight to df."""
    if quality_weights is None:
        quality_weights = QUALITY_MAP

    df = df.copy()
    df["quality_bucket"] = df[quality_col].astype(str)
    df["sample_weight"] = df["quality_bucket"].map(quality_weights).fillna(1.0)
    return df


def make_folds(
    df: pd.DataFrame,
    n_splits: int = 5,
    random_state: int = 42,
    shuffle: bool = True,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Simple KFold splitter.

    This is a placeholder; you can replace this with a BitBirch + StratifiedGroupKFold
    implementation that respects cluster_id and stratifies on (task_bin, quality).
    """
    indices = np.arange(len(df))
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    folds = []
    for tr_idx, val_idx in kf.split(indices):
        folds.append((tr_idx, val_idx))
    return folds


def get_xyw(
    df: pd.DataFrame,
    target_cols: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract X, y, sample_weight arrays from DataFrame.

    All non-target, non-quality columns are treated as features.
    """
    ignore_cols = set(target_cols) | {"quality_bucket", "sample_weight"}
    feature_cols = [c for c in df.columns if c not in ignore_cols]

    X = df[feature_cols].values.astype(np.float32)
    y = df[target_cols].values.astype(np.float32)
    w = df["sample_weight"].values.astype(np.float32)
    return X, y, w
