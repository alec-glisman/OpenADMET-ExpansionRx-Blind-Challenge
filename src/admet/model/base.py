"""Abstract base class for all ADMET models.

This module defines the common interface that all model implementations must follow,
enabling seamless model swapping and unified ensemble support.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, TypeVar

import numpy as np

if TYPE_CHECKING:
    from omegaconf import DictConfig

T = TypeVar("T", bound="BaseModel")


class BaseModel(ABC):
    """Abstract base class for all ADMET models.

    All model implementations (chemprop, chemeleon, xgboost, lightgbm, catboost)
    must inherit from this class and implement the required abstract methods.

    Attributes:
        model_type: Class attribute identifying the model type (set by registry).
        config: The OmegaConf configuration for this model.

    Example:
        >>> from admet.model.registry import ModelRegistry
        >>> @ModelRegistry.register("my_model")
        ... class MyModel(BaseModel):
        ...     def fit(self, smiles, y, val_smiles=None, val_y=None):
        ...         # Training implementation
        ...         self._fitted = True
        ...         return self
        ...     def predict(self, smiles):
        ...         # Prediction implementation
        ...         return np.zeros(len(smiles))
        ...     @classmethod
        ...     def from_config(cls, config):
        ...         return cls(config)
    """

    model_type: str = "base"

    def __init__(self, config: DictConfig) -> None:
        """Initialize the base model.

        Parameters:
            config: OmegaConf configuration containing model parameters.
        """
        self.config = config
        self._fitted = False

    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._fitted

    @abstractmethod
    def fit(
        self,
        smiles: list[str],
        y: np.ndarray,
        val_smiles: list[str] | None = None,
        val_y: np.ndarray | None = None,
    ) -> BaseModel:
        """Train the model on the provided data.

        Parameters:
            smiles: List of SMILES strings for training.
            y: Target values as numpy array.
            val_smiles: Optional validation SMILES strings.
            val_y: Optional validation target values.

        Returns:
            Self for method chaining.

        Raises:
            ValueError: If input data is invalid.
        """
        ...

    @abstractmethod
    def predict(self, smiles: list[str]) -> np.ndarray:
        """Make predictions for the given SMILES.

        Parameters:
            smiles: List of SMILES strings for prediction.

        Returns:
            Numpy array of predictions.

        Raises:
            RuntimeError: If model has not been fitted.
            ValueError: If input SMILES are invalid.
        """
        ...

    @classmethod
    @abstractmethod
    def from_config(cls: type[T], config: DictConfig) -> T:
        """Create a model instance from OmegaConf configuration.

        This factory method extracts model-specific parameters from the
        unified config structure and instantiates the appropriate model.

        Parameters:
            config: Full configuration with nested model parameters.

        Returns:
            Configured model instance.

        Example:
            >>> config = OmegaConf.load("config.yaml")
            >>> model = ChempropModel.from_config(config)
        """
        ...

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        """Get model parameters (sklearn compatibility).

        Parameters:
            deep: If True, return nested parameters.

        Returns:
            Dictionary of parameter names to values.
        """
        return {"config": self.config}

    def set_params(self, **params: Any) -> BaseModel:
        """Set model parameters (sklearn compatibility).

        Parameters:
            **params: Parameter names and values to set.

        Returns:
            Self for method chaining.
        """
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def __repr__(self) -> str:
        """Return string representation of the model."""
        return f"{self.__class__.__name__}(model_type={self.model_type!r}, fitted={self._fitted})"
