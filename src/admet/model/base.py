"""Base model interface for ADMET multi-output regression (initial version)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Sequence
import numpy as np


class BaseModel(ABC):
    """Abstract base class for multi-output regression models."""

    endpoints: Sequence[str]
    input_type: str  # "fingerprint" | "smiles"

    @abstractmethod
    def fit(
        self,
        X_train: np.ndarray,
        Y_train: np.ndarray,
        *,
        Y_mask: np.ndarray,
        X_val: np.ndarray | None = None,
        Y_val: np.ndarray | None = None,
        Y_val_mask: np.ndarray | None = None,
        sample_weight: np.ndarray | None = None,
        early_stopping_rounds: int | None = None,
    ) -> None:  # pragma: no cover - interface definition
        """Train the model.

        Y_mask: 1 where target present; 0 where missing.
        If model trains per-endpoint, rows with missing endpoint may be dropped.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        """Return predictions with shape (N, D) for all endpoints."""

    @abstractmethod
    def save(self, path: str) -> None:  # pragma: no cover - interface
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":  # pragma: no cover - interface
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:  # pragma: no cover - interface
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:  # pragma: no cover - interface
        pass
