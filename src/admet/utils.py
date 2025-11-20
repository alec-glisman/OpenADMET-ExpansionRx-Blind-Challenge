"""Utility helpers for reproducibility, seeding, and small conveniences.

This module centralizes functions that are cross-cutting across subpackages
but too small to deserve their own module.
"""

from __future__ import annotations

import os
import random
from typing import Optional


def set_global_seeds(seed: Optional[int]) -> None:
    """Seed Python, NumPy, Torch (if available) RNGs for reproducibility.

    Parameters
    ----------
    seed : Optional[int]
        Seed to apply. If ``None``, this function is a no-op.

    Notes
    -----
    - Sets ``PYTHONHASHSEED`` for deterministic hashing.
    - Seeds Python's ``random`` and NumPy. If PyTorch is installed, also seeds
      CPU and CUDA RNGs and enables deterministic CuDNN behavior.
    - Tree-based libraries like XGBoost/LightGBM require passing the seed via
      their estimators (``random_state``/``seed``). This function does not
      override those parameters; use the returned seed value there.
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


from admet.logging import configure_logging  # re-export for backwards compatibility

__all__ = ["set_global_seeds", "configure_logging"]
