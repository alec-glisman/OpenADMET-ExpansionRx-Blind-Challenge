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
    except Exception:
        pass
    try:
        import torch  # type: ignore

        torch.manual_seed(seed)
        if torch.cuda.is_available():  # pragma: no cover - depends on env
            torch.cuda.manual_seed_all(seed)
        # Make CuDNN deterministic
        try:
            torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = False  # type: ignore[attr-defined]
        except Exception:
            pass
    except Exception:
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


__all__ = ["set_global_seeds", "get_git_commit_hash"]
