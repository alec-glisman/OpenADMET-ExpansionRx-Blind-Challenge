"""General Utilities
====================

Utility helpers for reproducibility, seeding, and small conveniences.

Contents
--------
Functions
^^^^^^^^^
* :func:`set_global_seeds` – Seed Python/NumPy/Torch RNGs for reproducibility.
* :func:`get_git_commit_hash` – Retrieve the current git commit hash if available.
"""

from __future__ import annotations

import os
import random
import subprocess
from pathlib import Path
from typing import Optional, Union

# Re-export placed near top to satisfy lint rule about import grouping.
# Removed previous re-export of configure_logging; import no longer needed.


def set_global_seeds(seed: Optional[int]) -> None:
    """Seed common RNGs for reproducibility.

    Applies the seed to Python's ``random`` module, NumPy, and (if available)
    PyTorch CPU/CUDA RNGs. Also sets ``PYTHONHASHSEED`` for deterministic
    hashing and configures CuDNN deterministic behavior where possible.

    Parameters
    ----------
    seed : int or None
        Seed to apply. If ``None`` this function is a no-op.

    Returns
    -------
    None
        This function mutates global RNG state and returns nothing.

    Notes
    -----
    Tree-based libraries (XGBoost, LightGBM) still require an explicit
    ``seed``/``random_state`` in their estimator parameters; this utility
    does not override those settings.
    """
    if seed is None:
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    try:
        import numpy as np  # local import to avoid mandatory dependency here

        np.random.seed(seed)
    except ImportError:
        # Optional dependency missing; continue
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if getattr(torch, "cuda", None) is not None and torch.cuda.is_available():  # pragma: no cover - depends on env
            torch.cuda.manual_seed_all(seed)
        # Make CuDNN deterministic
        try:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except AttributeError:
            # Older torch or missing backend attributes; ignore
            pass
    except ImportError:
        # PyTorch not installed; ok to ignore
        pass


def get_git_commit_hash(repo_path: Union[str, Path, None] = None) -> Optional[str]:
    """Return the active git commit hash if the repository is available.

    Parameters
    ----------
    repo_path : str or pathlib.Path, optional
        Directory inside the git repository. Defaults to current working directory.

    Returns
    -------
    str or None
        The commit hash if available; otherwise ``None`` when git is missing or
        the path is not within a repository.
    """

    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_path) if repo_path is not None else None,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    commit = completed.stdout.strip()
    return commit or None


def parse_data_dir_params(data_dir: Union[str, Path]) -> dict[str, Optional[str]]:
    """Parse dataset path to extract structured parameters.

    Parses paths following the convention:
    ``<base>/split_<split_type>/<version>/<quality>/<cluster_method>/<split_method>/data/split_<N>/fold_<M>``

    For example:
    ``assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data/split_0/fold_0``

    Parameters
    ----------
    data_dir : str or pathlib.Path
        Path to the dataset directory following the convention above.

    Returns
    -------
    dict[str, str | None]
        Dictionary containing parsed parameters:
        - split_type: Type of data split (e.g., "train_val", "train_val_local_test")
        - version: Version string (e.g., "v3")
        - quality: Quality level (e.g., "high", "all_quality")
        - cluster_method: Clustering method (e.g., "bitbirch", "random")
        - split_method: Split method (e.g., "multilabel_stratified_kfold")
        - split: Split index as string (e.g., "0")
        - fold: Fold index as string (e.g., "0")

        Any parameters that cannot be parsed will have ``None`` values.

    Examples
    --------
    >>> path = "assets/dataset/split_train_val/v3/quality_high/bitbirch/multilabel_stratified_kfold/data/split_0/fold_0"
    >>> params = parse_data_dir_params(path)
    >>> params["split_type"]
    'train_val'
    >>> params["quality"]
    'high'
    >>> params["split"]
    '0'
    """
    import re

    result: dict[str, Optional[str]] = {
        "split_type": None,
        "version": None,
        "quality": None,
        "cluster_method": None,
        "split_method": None,
        "split": None,
        "fold": None,
    }

    path = Path(data_dir)
    parts = path.parts

    # Extract split and fold from the end of the path
    # Pattern: split_<N>/fold_<M> or just split_<N>
    for i, part in enumerate(parts):
        if part.startswith("split_") and not part.startswith("split_train"):
            # This is a split index, not split_train_val
            match = re.match(r"split_(\d+)", part)
            if match:
                result["split"] = match.group(1)
        if part.startswith("fold_"):
            match = re.match(r"fold_(\d+)", part)
            if match:
                result["fold"] = match.group(1)

    # Find the split_train_val or similar component
    # Expected patterns: split_train_val, split_train_val_local_test, etc.
    for i, part in enumerate(parts):
        if part.startswith("split_train"):
            # Extract everything after "split_" as split_type
            result["split_type"] = part.replace("split_", "")

            # The next parts should be version, quality, cluster_method, split_method
            # But we need to find "data" to know where the structure ends
            remaining = list(parts[i + 1 :])

            # Find "data" marker
            if "data" in remaining:
                data_idx = remaining.index("data")
                structure_parts = remaining[:data_idx]

                if len(structure_parts) >= 1:
                    result["version"] = structure_parts[0]
                if len(structure_parts) >= 2:
                    # quality_high -> high, all_quality -> all_quality
                    quality = structure_parts[1]
                    if quality.startswith("quality_"):
                        result["quality"] = quality.replace("quality_", "")
                    else:
                        result["quality"] = quality
                if len(structure_parts) >= 3:
                    result["cluster_method"] = structure_parts[2]
                if len(structure_parts) >= 4:
                    result["split_method"] = structure_parts[3]
            break

    return result


__all__ = ["set_global_seeds", "get_git_commit_hash", "parse_data_dir_params"]
