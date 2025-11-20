"""
Model
=====

Model abstractions and backend implementations.

.. module:: admet.model

"""

from .base import BaseModel
from .xgb_wrapper import XGBoostMultiEndpoint

__all__ = ["BaseModel", "XGBoostMultiEndpoint"]
