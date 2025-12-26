"""Model registry for dynamic model type resolution.

This module provides a registry pattern that allows models to be registered
with a decorator and instantiated dynamically based on configuration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from omegaconf import DictConfig

    from admet.model.base import BaseModel

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Registry for model types with factory pattern.

    Provides decorator-based registration and factory methods for creating
    model instances from configuration.

    Example:
        >>> from admet.model.registry import ModelRegistry
        >>> from admet.model.base import BaseModel
        >>>
        >>> @ModelRegistry.register("my_model")
        ... class MyModel(BaseModel):
        ...     # Implementation
        ...     pass
        >>>
        >>> # Later, create model from config
        >>> config = OmegaConf.create({"model": {"type": "my_model"}})
        >>> model = ModelRegistry.create(config)
    """

    _registry: dict[str, type[BaseModel]] = {}

    @classmethod
    def register(cls, model_type: str) -> Callable[[type[BaseModel]], type[BaseModel]]:
        """Decorator to register a model class.

        Parameters:
            model_type: String identifier for the model type.

        Returns:
            Decorator function that registers the model class.

        Raises:
            ValueError: If model_type is already registered.

        Example:
            >>> @ModelRegistry.register("xgboost")
            ... class XGBoostModel(BaseModel):
            ...     pass
        """

        def decorator(model_cls: type[BaseModel]) -> type[BaseModel]:
            if model_type in cls._registry:
                raise ValueError(
                    f"Model type '{model_type}' already registered by {cls._registry[model_type].__name__}"
                )
            cls._registry[model_type] = model_cls
            model_cls.model_type = model_type
            logger.debug(f"Registered model type '{model_type}' -> {model_cls.__name__}")
            return model_cls

        return decorator

    @classmethod
    def create(cls, config: DictConfig) -> BaseModel:
        """Create model instance from configuration.

        Looks up the model type from config.model.type and instantiates
        the corresponding registered model class.

        Parameters:
            config: OmegaConf configuration with model.type field.

        Returns:
            Instantiated model of the appropriate type.

        Raises:
            ValueError: If model type is unknown or not registered.
            KeyError: If config.model.type is missing.

        Example:
            >>> config = OmegaConf.create({
            ...     "model": {"type": "chemprop", "chemprop": {...}}
            ... })
            >>> model = ModelRegistry.create(config)
        """
        model_type = config.model.type
        if model_type not in cls._registry:
            available = list(cls._registry.keys())
            raise ValueError(f"Unknown model type: '{model_type}'. Available types: {available}")
        model_cls = cls._registry[model_type]
        logger.info(f"Creating model of type '{model_type}'")
        return model_cls.from_config(config)

    @classmethod
    def list_models(cls) -> list[str]:
        """List all registered model types.

        Returns:
            List of registered model type identifiers.
        """
        return list(cls._registry.keys())

    @classmethod
    def get(cls, model_type: str) -> type[BaseModel]:
        """Get model class by type identifier.

        Parameters:
            model_type: String identifier for the model type.

        Returns:
            The registered model class.

        Raises:
            KeyError: If model type is not registered.
        """
        if model_type not in cls._registry:
            available = list(cls._registry.keys())
            raise KeyError(f"Model type '{model_type}' not found. Available: {available}")
        return cls._registry[model_type]

    @classmethod
    def is_registered(cls, model_type: str) -> bool:
        """Check if a model type is registered.

        Parameters:
            model_type: String identifier to check.

        Returns:
            True if the model type is registered.
        """
        return model_type in cls._registry

    @classmethod
    def clear(cls) -> None:
        """Clear all registered models (primarily for testing)."""
        cls._registry.clear()
        logger.debug("Cleared model registry")
