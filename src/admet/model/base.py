"""admet.model.base
====================

Core abstract interfaces and structural protocols for multi‑endpoint ADMET
regression models.

Design Goals
------------
* Keep trainer requirements minimal (fit, predict, save/load, config/meta).
* Support both fingerprint and raw SMILES inputs (``input_type`` attribute).
* Allow non‑``BaseModel`` classes (e.g. lightweight test doubles) through a
    structural :class:`ModelProtocol`.

Shape Conventions
-----------------
``X_*`` arrays: ``(N, F)`` where ``F`` is feature dimension (e.g., fingerprint bits).
``Y_*`` arrays: ``(N, D)`` where ``D`` equals ``len(endpoints)``; missing targets
represented by ``NaN`` with a parallel mask ``Y_mask`` of the same shape (1 = present).

Early Stopping
--------------
Implementations may leverage ``early_stopping_rounds``; if unsupported, they
should ignore the argument gracefully.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Protocol, Sequence, runtime_checkable

import numpy as np


class BaseModel(ABC):  # pylint: disable=invalid-name
    """Abstract base class for multi‑output regression models.

    Attributes
    ----------
    endpoints : Sequence[str]
        Ordered list of endpoint (target) names.
    input_type : str
        String identifier for expected input modality (``'fingerprint'`` or ``'smiles'``).

    Note
    ----
    Parameters use capital X/Y letters to follow scientific computing convention where
    X denotes feature matrices and Y denotes target matrices. This is widespread in
    scikit-learn and related ML libraries and takes precedence over PEP 8 naming style.
    """

    endpoints: Sequence[str]
    input_type: str  # "fingerprint" | "smiles"

    @abstractmethod
    def fit(  # pylint: disable=invalid-name
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
        """Fit the model on training (and optional validation) data.

        Parameters
        ----------
        X_train : numpy.ndarray
            Feature matrix of shape ``(N_train, F)``.
        Y_train : numpy.ndarray
            Target matrix of shape ``(N_train, D)`` with ``NaN`` for missing labels.
        Y_mask : numpy.ndarray
            Mask matrix same shape as ``Y_train`` (1 = present, 0 = missing).
        X_val : numpy.ndarray, optional
            Validation feature matrix ``(N_val, F)``.
        Y_val : numpy.ndarray, optional
            Validation targets ``(N_val, D)`` with ``NaN`` placeholders.
        Y_val_mask : numpy.ndarray, optional
            Validation target mask.
        sample_weight : numpy.ndarray, optional
            Optional per‑sample weights for training rows (shape ``(N_train,)``).
        early_stopping_rounds : int, optional
            Backend‑specific early stopping patience; ignored if unsupported.

        Raises
        ------
        ValueError
            Implementations may raise if shapes/invariants are violated.
        """

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        """Generate predictions for all endpoints.

        Parameters
        ----------
        X : numpy.ndarray
            Feature matrix ``(N, F)``.

        Returns
        -------
        numpy.ndarray
            Predicted values ``(N, D)`` aligned with ``self.endpoints``.
        """

    @abstractmethod
    def save(self, path: str) -> None:  # pragma: no cover - interface
        """Persist model artifacts to ``path``.

        Parameters
        ----------
        path : str
            Destination directory or file depending on backend.
        """
        pass

    @classmethod
    @abstractmethod
    def load(cls, path: str) -> "BaseModel":  # pragma: no cover - interface
        """Load a previously saved model from ``path``.

        Parameters
        ----------
        path : str
            Source path used by :meth:`save`.

        Returns
        -------
        BaseModel
            Instantiated model ready for inference.
        """
        pass

    @abstractmethod
    def get_config(self) -> Dict[str, Any]:  # pragma: no cover - interface
        """Return backend configuration / hyperparameters.

        Returns
        -------
        dict
            Serializable configuration for reproducibility.
        """
        pass

    @abstractmethod
    def get_metadata(self) -> Dict[str, Any]:  # pragma: no cover - interface
        """Return model metadata (e.g., training stats, version info)."""
        pass


@runtime_checkable
class ModelProtocol(Protocol):
    """Structural protocol for trainer‑compatible models.

    Provides a duck‑typed alternative to subclassing :class:`BaseModel`. Any
    class implementing this surface can be consumed by training code. Shape
    and semantic conventions match those documented for :class:`BaseModel`.
    """

    endpoints: Sequence[str]
    input_type: str

    def fit(  # pylint: disable=invalid-name
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
    ) -> None: ...

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Generate predictions for input features.

        Args:
            X: Feature matrix of shape (N, F)

        Returns:
            Prediction matrix of shape (N, D) where D is number of endpoints
        """

    def save(self, path: str) -> None:
        """Save model to filesystem.

        Args:
            path: Directory path to save model artifacts
        """

    @classmethod
    def load(cls, path: str) -> "ModelProtocol":
        """Load model from filesystem.

        Args:
            path: Directory path containing model artifacts

        Returns:
            Loaded model instance
        """
